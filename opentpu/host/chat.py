"""otpu-chat: chat with Qwen3 (or LFM2, or Qwen3.5) running on openTPU.

    otpu-chat                                 # ISA simulator (~3 s/token on a laptop)
    otpu-chat --model lfm2                    # LFM2.5-230M instead of Qwen3-0.6B
    otpu-chat --model qwen35                  # Qwen3.5-0.8B (text only)
    otpu-chat --backend board                 # the FPGA over PCIe (opentpu/host/board.py)
    otpu-chat --backend board-sim             # the Verilator board model (very slow)
    otpu-chat --prompt "Why is the sky blue?" # one-shot
    otpu-chat --think                         # Qwen3 thinking mode

The model runs token by token on the device; the host only tokenizes, looks up the embedding
row, applies the chat template and samples from the logits. The KV cache stays in device DRAM
across turns; only the new turn's tokens are fed. On the card the tool holds the device lock
and publishes its status (model, DRAM, tokens/s) for otpu-smi; the next token's program is
compiled while the card runs the current one (Engine pipelining).
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from opentpu.llm import MODELS, load_spec, model_dir
from opentpu.llm.qwen3 import Engine, load_weights


# Sampling defaults per model family (Spec module); command-line flags override them. LFM2's
# are its generation_config.json; Qwen3.5's its model card's non-thinking settings (without the
# presence penalty).
SAMPLING = {"qwen3": dict(temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.0),
            "lfm2": dict(temperature=0.1, top_k=50, top_p=1.0, repetition_penalty=1.05),
            "qwen35": dict(temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.0)}


def sampler(temperature: float, top_k: int, top_p: float, seed: int | None,
            repetition_penalty: float = 1.0):
    """pick(logits, context) -> token id. The repetition penalty (as Hugging Face's) divides
    the positive logits and multiplies the negative ones of every token in `context`; it
    applies to greedy decoding (temperature 0) too."""
    rng = np.random.default_rng(seed)

    def pick(logits, context=()):
        if repetition_penalty != 1.0 and len(context):
            logits = logits.copy()
            seen = np.unique(np.asarray(context, np.int64))
            v = logits[seen]
            logits[seen] = np.where(v > 0, v / repetition_penalty, v * repetition_penalty)
        if temperature <= 0:
            return int(np.argmax(logits))
        z = logits.astype(np.float64) / temperature
        idx = np.argpartition(-z, top_k)[:top_k] if top_k else np.arange(len(z))
        z = z[idx]
        order = np.argsort(-z)
        idx, z = idx[order], z[order]
        p = np.exp(z - z[0])
        p /= p.sum()
        keep = min(len(p), np.searchsorted(np.cumsum(p), top_p) + 1)
        p = p[:keep] / p[:keep].sum()
        return int(idx[rng.choice(keep, p=p)])

    return pick


def sampling(spec, args) -> dict:
    """The model family's SAMPLING defaults, overridden by the flags given on the command
    line (None when not given)."""
    d = dict(SAMPLING[type(spec).__module__.rsplit(".", 1)[-1]])
    d.update({k: getattr(args, k) for k in d if getattr(args, k, None) is not None})
    return d


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
            t = self.pick(logits, self.fed)
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


def make_backend(name: str, spec, cap: int, dev: str, model: str | None = None):
    """(backend, configuration) for Engine."""
    if name == "isa":
        return "isa", None
    if name == "board":
        from opentpu.host.board import BoardBackend, XdmaTransport   # the PCIe driver
        from opentpu.isasim import board_config
        return (lambda c, imgs: BoardBackend(c, imgs, transport=XdmaTransport(dev),
                                             model=model)), board_config()
    if name == "board-sim":
        from opentpu.host.board import BoardBackend, SimTransport, sim_config
        cfg = sim_config(spec, cap)
        tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2)
        return (lambda c, imgs: BoardBackend(c, imgs, transport=tr, model=model)), cfg
    if name == "rtl":
        from opentpu.llm.rtl_backend import RtlBackend
        return RtlBackend, None
    raise SystemExit(f"unknown backend {name}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="otpu-chat", description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen3",
                    help=f"{' or '.join(MODELS)} (models/<name>), or a checkpoint directory")
    ap.add_argument("--backend", default="isa", choices=["isa", "board", "board-sim", "rtl"])
    ap.add_argument("--dev", default="/dev/xdma0", help="XDMA device prefix (--backend board)")
    ap.add_argument("--clock-mhz", type=float,
                    help="core clock, to turn device cycles into tokens/s (default: the "
                         "bitstream's CORE_KHZ, or 100 on a register map 1 bitstream)")
    ap.add_argument("--cap", type=int, default=2048, help="KV cache capacity (tokens)")
    ap.add_argument("--prompt", help="ask one question and exit")
    ap.add_argument("--think", action="store_true", help="enable Qwen3 / Qwen3.5 thinking mode")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float,
                    help="sampling flags default per model: " + "; ".join(
                        f"{m} " + " ".join(f"{k}={v}" for k, v in d.items())
                        for m, d in SAMPLING.items()))
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--repetition-penalty", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-new", type=int, default=256)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    path = model_dir(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    spec = load_spec(path)
    print(f"loading {path.name} onto openTPU ({a.backend}) ...", flush=True)
    backend, cfg = make_backend(a.backend, spec, a.cap, a.dev, path.name)
    eng = Engine(spec, load_weights(path), cap=a.cap, cfg=cfg, backend=backend)
    sp = sampling(spec, a)
    pick = sampler(0 if a.greedy else sp["temperature"], sp["top_k"], sp["top_p"], a.seed,
                   sp["repetition_penalty"])
    clock = 0.0
    if a.backend.startswith("board"):
        khz = eng.backend.info.get("core_khz")
        clock = a.clock_mhz or (khz / 1e3 if khz else 100.0)
    chat = Chat(eng, tok, a.think, pick, a.max_new, clock_mhz=clock)
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
