"""otpu-chat: chat with Qwen3 (or LFM2, or Qwen3.5) running on openTPU.

    otpu-chat                                 # ISA simulator (~3 s/token on a laptop)
    otpu-chat --plain                         # a plain REPL instead of the full-screen one
    otpu-chat --model lfm2                    # LFM2.5-230M instead of Qwen3-0.6B
    otpu-chat --model qwen35                  # Qwen3.5-0.8B (text only)
    otpu-chat --backend board                 # the FPGA over PCIe (opentpu/host/board.py)
    otpu-chat --backend board-sim             # the Verilator board model (very slow)
    otpu-chat --prompt "Why is the sky blue?" # one-shot
    otpu-chat --think                         # Qwen3 thinking mode

The interactive mode is a full-screen interface (opentpu/host/chat_tui.py): the conversation,
and a panel with TTFT, prefill and decode tokens/s (wall and device), the KV context, DRAM and
session totals, updated while the reply streams. --plain and --prompt print the same numbers
as one line per reply.

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
from dataclasses import dataclass

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


@dataclass
class Turn:
    """The numbers of one reply. Wall times are seconds from the submit; device numbers come
    from the engine's per-step cycles at `clock_mhz` (0: no device clock, wall only).
    Prefill: the prompt tokens fed this turn (the KV cache keeps the earlier turns). TTFT:
    submit -> first generated token. Decode: the tokens after the first, over the time since
    the first."""
    clock_mhz: float = 0.0
    cap: int = 0
    prefill_tokens: int = 0
    prefill_s: float = 0.0
    prefill_cycles: int = 0
    ttft_s: float | None = None
    gen_tokens: int = 0
    decode_s: float = 0.0             # first -> latest generated token
    decode_steps: int = 0             # device steps after the prefill
    decode_cycles: int = 0
    context: int = 0                  # KV positions filled
    restarted: bool = False           # the template changed the history: KV rebuilt
    stopped: bool = False             # stopped by the user

    def _dev(self, n: int, cycles: int) -> float | None:
        return n * self.clock_mhz * 1e6 / cycles if self.clock_mhz and cycles else None

    @property
    def prefill_tok_s(self) -> float | None:
        return self.prefill_tokens / self.prefill_s if self.prefill_s else None

    @property
    def prefill_dev_tok_s(self) -> float | None:
        return self._dev(self.prefill_tokens, self.prefill_cycles)

    @property
    def decode_tok_s(self) -> float | None:
        return (self.gen_tokens - 1) / self.decode_s if self.gen_tokens > 1 and self.decode_s \
            else None

    @property
    def decode_dev_tok_s(self) -> float | None:
        return self._dev(self.decode_steps, self.decode_cycles)

    @property
    def mcycles_per_token(self) -> float | None:
        return self.decode_cycles / self.decode_steps / 1e6 if self.decode_steps else None

    def line(self) -> str:
        """The plain-mode summary: TTFT, prefill tok/s, decode tok/s, context."""
        def r(x, dev):
            return "n/a" if x is None else f"{x:.2f}" + ("" if dev is None else
                                                         f" (device {dev:.1f})")
        ttft = "n/a" if self.ttft_s is None else f"{self.ttft_s:.2f}s"
        mc = "" if self.mcycles_per_token is None else \
            f", {self.mcycles_per_token:.2f} Mcycles/token at {self.clock_mhz:.0f} MHz"
        return (f"[TTFT {ttft}; prefill {self.prefill_tokens} tokens, "
                f"{r(self.prefill_tok_s, self.prefill_dev_tok_s)} tok/s; decode "
                f"{self.gen_tokens} tokens, {r(self.decode_tok_s, self.decode_dev_tok_s)} tok/s"
                f"{mc}; context {self.context}/{self.cap}"
                + (", stopped" if self.stopped else "") + "]")


@dataclass
class Session:
    turns: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    decode_s: float = 0.0
    decode_tokens: int = 0            # tokens counted in decode_s (the first of a turn is not)

    def add(self, t: Turn) -> None:
        self.turns += 1
        self.tokens_in += t.prefill_tokens
        self.tokens_out += t.gen_tokens
        if t.gen_tokens > 1:
            self.decode_s += t.decode_s
            self.decode_tokens += t.gen_tokens - 1

    @property
    def decode_tok_s(self) -> float | None:
        return self.decode_tokens / self.decode_s if self.decode_s else None


class Chat:
    """A conversation on an Engine: the KV cache keeps every fed token across turns, so a turn
    feeds only what the chat template added since (when the template rewrites the history, it
    starts over)."""

    def __init__(self, engine: Engine, tok, think: bool, pick, max_new: int,
                 clock_mhz: float = 0.0):
        self.eng, self.tok, self.think, self.pick, self.max_new = engine, tok, think, pick, max_new
        self.clock_mhz = clock_mhz
        self.history: list[dict] = []
        self.fed: list[int] = []            # tokens whose K/V are in the device cache
        self.session = Session()
        self.last: Turn | None = None

    def _template(self, add_prompt=True) -> list[int]:
        ids = self.tok.apply_chat_template(self.history, add_generation_prompt=add_prompt,
                                           enable_thinking=self.think, tokenize=True)
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)

    def reset(self) -> None:
        """Forget the conversation and the KV cache."""
        self.history, self.fed = [], []
        self.eng.reset()

    def _cycles(self, k0: int, k1: int | None = None) -> int:
        return int(sum(st.get("cycles", 0) for st in self.eng.stats[k0:k1] if st))

    def ask(self, text: str, on_update=None, stop=lambda: False) -> tuple[str, Turn]:
        """One turn. on_update(delta_text, turn) after the prefill (delta "") and after every
        generated token; stop() is polled between steps (the reply so far is kept)."""
        on_update = on_update or (lambda delta, turn: None)
        t0 = time.perf_counter()
        turn = Turn(clock_mhz=self.clock_mhz, cap=self.eng.cap)
        self.history.append({"role": "user", "content": text})
        ids = self._template()
        n = len(self.fed)
        if ids[:n] != self.fed:              # template rewrote history: start over
            self.eng.reset()
            self.fed, n, turn.restarted = [], 0, True
        k0 = len(self.eng.stats)
        logits = None
        for t in ids[n:]:
            if stop():
                turn.stopped = True
                break
            logits = self.eng.step(t)
            self.fed.append(t)
            turn.prefill_tokens += 1
        turn.prefill_s = time.perf_counter() - t0
        k1 = len(self.eng.stats)
        turn.prefill_cycles = self._cycles(k0, k1)
        turn.context = self.eng.pos
        on_update("", turn)
        out, shown, t_first = [], "", None
        while logits is not None and len(out) < self.max_new and not turn.stopped:
            t = self.pick(logits, self.fed)
            if t in self.eng.spec.eos:
                break
            out.append(t)
            now = time.perf_counter()
            if t_first is None:
                t_first, turn.ttft_s = now, now - t0
            turn.gen_tokens, turn.decode_s = len(out), now - t_first
            text_now = self.tok.decode(out, skip_special_tokens=True)
            delta, shown = text_now[len(shown):], text_now
            on_update(delta, turn)
            if self.eng.pos >= self.eng.cap:
                break
            if stop():
                turn.stopped = True
                break
            logits = self.eng.step(t)
            self.fed.append(t)
            turn.decode_steps = len(self.eng.stats) - k1
            turn.decode_cycles = self._cycles(k1)
            turn.context = self.eng.pos
        reply = self.tok.decode(out, skip_special_tokens=True)
        self.history.append({"role": "assistant", "content": reply})
        self.session.add(turn)
        self.last = turn
        return reply, turn

    def ask_plain(self, text: str, stream=sys.stdout) -> str:
        """ask() printing the reply as it streams, then the turn's numbers."""
        def show(delta, turn):
            stream.write(delta)
            stream.flush()
        reply, turn = self.ask(text, show)
        stream.write("\n" + turn.line() + "\n")
        return reply


def make_backend(name: str, spec, cap: int, dev: str, model: str | None = None):
    """(backend, configuration) for Engine."""
    if name == "isa":
        return "isa", None
    if name == "board":
        from opentpu.host.board import (Board, BoardBackend, XdmaTransport,   # the PCIe driver
                                        device_config)
        tr = XdmaTransport(dev)
        cfg = device_config(Board(tr, lock=False).info())     # MCOLS / LANES of the bitstream
        return (lambda c, imgs: BoardBackend(c, imgs, transport=tr, model=model)), cfg
    if name == "board-sim":
        from opentpu.host.board import BoardBackend, SimTransport, sim_config
        cfg = sim_config(spec, cap)
        tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2)
        return (lambda c, imgs: BoardBackend(c, imgs, transport=tr, model=model)), cfg
    if name == "rtl":
        from opentpu.llm.rtl_backend import RtlBackend
        return RtlBackend, None
    raise SystemExit(f"unknown backend {name}")


def panel_meta(eng, backend: str, dev: str, model: str, sp: dict, max_new: int) -> dict:
    """What the interface shows about the model, the device and the sampling."""
    info = getattr(eng.backend, "info", None)
    meta = {"model": model, "backend": backend, "sampling": {**sp, "max_new": max_new},
            "device": {"board": dev, "board-sim": "Verilator board model"}.get(
                backend, "ISA simulator (host)"), "dram": None}
    c = eng.cfg
    if info:
        meta["bitstream"] = [f"D={info['D']} MCOLS={info['MCOLS']} LANES={info['LANES']}",
                             "build " + ("n/a" if info["build_id"] is None
                                         else f"{info['build_id']:08x}")
                             + (f", {info['core_khz'] / 1e3:g} MHz" if info["core_khz"]
                                else "")]
    else:
        meta["bitstream"] = [f"D={c.D} MCOLS={c.MCOLS} (simulated)"]
    if hasattr(eng.backend, "image_bytes"):
        from opentpu.host.board import dram_layout
        be = eng.backend
        meta["dram"] = lambda: dram_layout(c, be.image_bytes, be.prog_at, eng.image,
                                           list(eng.poss))
    return meta


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
    ap.add_argument("--prompt", help="ask one question and exit (plain output)")
    ap.add_argument("--plain", action="store_true",
                    help="a line-by-line REPL instead of the full-screen interface")
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
    from opentpu.host.board import ConfigMismatch
    try:
        backend, cfg = make_backend(a.backend, spec, a.cap, a.dev, path.name)
    except ConfigMismatch as e:
        raise SystemExit(f"otpu-chat: {e}") from None
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
        chat.ask_plain(a.prompt)
        return
    if not a.plain:
        from opentpu.host.chat_tui import ChatApp
        ChatApp(chat, panel_meta(eng, a.backend, a.dev, path.name,
                                 dict(sp, greedy=a.greedy), a.max_new)).run()
        return
    print("type a message (empty line or Ctrl-D to quit)")
    while True:
        try:
            text = input("\n> ").strip()
        except EOFError:
            break
        if not text:
            break
        chat.ask_plain(text)


if __name__ == "__main__":
    main()
