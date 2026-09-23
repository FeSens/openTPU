"""Chat with Qwen3 running on openTPU.

    python3 tools/chat.py                                 # ISA simulator (~3 s/token on a laptop)
    python3 tools/chat.py --backend board                 # the FPGA over PCIe (host/board.py)
    python3 tools/chat.py --backend board-sim             # the Verilator board model (very slow)
    python3 tools/chat.py --prompt "Why is the sky blue?" # one-shot
    python3 tools/chat.py --think                         # Qwen3 thinking mode

The model runs token by token on the device; the host only tokenizes, looks up the embedding
row, applies the chat template and samples from the logits. The KV cache stays in device DRAM
across turns; only the new turn's tokens are fed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu.llm.qwen3 import Engine, Spec, load_weights  # noqa: E402


def sampler(temperature: float, top_k: int, top_p: float, seed: int | None):
    rng = np.random.default_rng(seed)

    def pick(logits):
        if temperature <= 0:
            return int(np.argmax(logits))
        z = logits.astype(np.float64) / temperature
        idx = np.argpartition(-z, top_k)[:top_k] if top_k else np.arange(len(z))
        z = z[idx]
        order = np.argsort(-z)
        idx, z = idx[order], z[order]
        p = np.exp(z - z[0])
        p /= p.sum()
        keep = np.searchsorted(np.cumsum(p), top_p) + 1
        p = p[:keep] / p[:keep].sum()
        return int(idx[rng.choice(keep, p=p)])

    return pick


class Chat:
    def __init__(self, engine: Engine, tok, think: bool, pick, max_new: int,
                 clock_mhz: float = 0.0):
        self.eng, self.tok, self.think, self.pick, self.max_new = engine, tok, think, pick, max_new
        self.clock_mhz = clock_mhz
        self.history: list[dict] = []
        self.fed: list[int] = []            # tokens whose K/V are in the device cache

    def _template(self, add_prompt=True) -> list[int]:
        ids = self.tok.apply_chat_template(self.history, add_generation_prompt=add_prompt,
                                           enable_thinking=self.think, tokenize=True)
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)

    def ask(self, text: str, stream=sys.stdout) -> str:
        self.history.append({"role": "user", "content": text})
        ids = self._template()
        n = len(self.fed)
        if ids[:n] != self.fed:              # template rewrote history: start over
            self.eng.reset()
            self.fed, n = [], 0
        t0 = time.time()
        logits = None
        for t in ids[n:]:
            logits = self.eng.step(t)
            self.fed.append(t)
        t1 = time.time()
        out, shown = [], ""
        for _ in range(self.max_new):
            t = self.pick(logits)
            if t in self.eng.spec.eos:
                break
            out.append(t)
            text_now = self.tok.decode(out, skip_special_tokens=True)
            stream.write(text_now[len(shown):])
            stream.flush()
            shown = text_now
            if self.eng.pos >= self.eng.cap:
                break
            logits = self.eng.step(t)
            self.fed.append(t)
        t2 = time.time()
        stream.write("\n")
        n_pre, n_gen = len(ids) - n, len(out)
        dev = ""
        cyc = [st.get("cycles", 0) for st in self.eng.stats[-max(n_gen, 1):] if st]
        if cyc and self.clock_mhz:
            c = float(np.mean(cyc))
            dev = (f", device {c / 1e6:.2f} Mcycles/token = "
                   f"{self.clock_mhz * 1e6 / c:.1f} tok/s at {self.clock_mhz:.0f} MHz")
        stream.write(f"[{n_pre} prompt tokens in {t1 - t0:.1f}s, {n_gen} tokens in "
                     f"{t2 - t1:.1f}s ({n_gen / max(t2 - t1, 1e-9):.2f} tok/s){dev}, "
                     f"context {self.eng.pos}/{self.eng.cap}]\n")
        reply = self.tok.decode(out, skip_special_tokens=True)
        self.history.append({"role": "assistant", "content": reply})
        return reply


def make_backend(name: str, spec: Spec, cap: int, dev: str):
    """(backend, configuration) for Engine."""
    if name == "isa":
        return "isa", None
    if name == "board":
        from host.board import BoardBackend, XdmaTransport      # PCIe driver (host/board.py)
        from opentpu.isasim import board_config
        return (lambda c, imgs: BoardBackend(c, imgs, transport=XdmaTransport(dev))), \
            board_config()
    if name == "board-sim":
        from host.board import BoardBackend, SimTransport, sim_config
        cfg = sim_config(spec, cap)
        tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2)
        return (lambda c, imgs: BoardBackend(c, imgs, transport=tr)), cfg
    if name == "rtl":
        from opentpu.llm.rtl_backend import RtlBackend
        return RtlBackend, None
    raise SystemExit(f"unknown backend {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3-0.6B"))
    ap.add_argument("--backend", default="isa", choices=["isa", "board", "board-sim", "rtl"])
    ap.add_argument("--dev", default="/dev/xdma0", help="XDMA device prefix (--backend board)")
    ap.add_argument("--clock-mhz", type=float, default=100.0,
                    help="core clock, to turn device cycles into tokens/s")
    ap.add_argument("--cap", type=int, default=2048, help="KV cache capacity (tokens)")
    ap.add_argument("--prompt", help="ask one question and exit")
    ap.add_argument("--think", action="store_true", help="enable Qwen3 thinking mode")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-new", type=int, default=256)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    spec = Spec.from_hf(a.model)
    print(f"loading {Path(a.model).name} onto openTPU ({a.backend}) ...", flush=True)
    backend, cfg = make_backend(a.backend, spec, a.cap, a.dev)
    eng = Engine(spec, load_weights(a.model), cap=a.cap, cfg=cfg, backend=backend)
    pick = sampler(0 if a.greedy else a.temperature, a.top_k, a.top_p, a.seed)
    chat = Chat(eng, tok, a.think, pick, a.max_new,
                clock_mhz=a.clock_mhz if a.backend.startswith("board") else 0.0)
    if a.prompt:
        chat.ask(a.prompt)
        return
    print("type a message (empty line or Ctrl-D to quit)")
    while True:
        try:
            text = input("\n> ").strip()
        except EOFError:
            break
        if not text:
            break
        chat.ask(text)


if __name__ == "__main__":
    main()
