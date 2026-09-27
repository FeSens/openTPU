"""Where the host time of one decode token goes: a timeline of otpu-chat's decode loop.

    python3 tools/decode_profile.py --model lfm2 [--backend board | fake] [--tokens 64]
                                    [--prompt "..."] [--greedy] [--json out.json]

Runs one Chat turn (plain mode, the reply printed to /dev/null) on the card (--backend
board) or on FakeTransport (--backend fake: a card that computes nothing and halts after
--fake-ms; the host path only), with timers around every piece of the per-token work:

  main thread   io-write (x / cos / sin), wait-compile (the step waits for the precompiled
                program), prog-upload (program DMA), imem-load (LOAD .. not LOADING), run
                (Board.start .. HALTED seen, split into device = CYCLES / CORE_KHZ and overshoot =
                the rest: poll wake-up and PCIe round trips), counters (register reads after
                the run), status (the status file), logits-read (DMA + unpack), sample,
                detok, ui (the on_update callback), and "other" (the step's wall time not in
                any of these);
  compile thread  trace (compile_step) and assemble (prepare), overlapped with the run.

Prints the mean per decode token (the first generated token and the prefill are excluded)
and wall vs device tokens/s.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu.host import board as B  # noqa: E402
from opentpu.host import chat as C  # noqa: E402
from opentpu.host import runstate  # noqa: E402
from opentpu.llm import load_spec, model_dir  # noqa: E402
from opentpu.llm import qwen3 as Q  # noqa: E402

T = defaultdict(float)            # label -> seconds (while `on`)
N = defaultdict(int)
on = threading.Event()


def timed(owner, name: str, label: str):
    f = getattr(owner, name)

    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            if on.is_set():
                T[label] += time.perf_counter() - t0
                N[label] += 1
    setattr(owner, name, w)
    return w


def instrument(khz_box: list) -> None:
    timed(Q.Engine, "_program", "wait-compile")
    timed(Q.Engine, "step", "step")
    timed(B.BoardBackend, "prepare", "assemble")
    timed(B.BoardBackend, "write", "io-write")
    timed(B.BoardBackend, "read", "logits-read")
    timed(runstate.RunnerStatus, "token", "status")


    def load_program(self, addr, words):
        words = np.asarray(words, "<u4")
        t0 = time.perf_counter()
        self.write(addr, words.view(np.uint8))
        t1 = time.perf_counter()
        t = self.t
        t.reg_write(B.R_CTRL, 0)
        t.reg_write(B.R_PROG_ADDR, addr)
        t.reg_write(B.R_PROG_N, len(words) // 8)
        t.reg_write(B.R_CTRL, B.CTRL_LOAD)
        t.poll(B.R_STATUS, B.ST_LOADING, 0)
        t2 = time.perf_counter()
        if on.is_set():
            T["prog-upload"] += t1 - t0
            T["imem-load"] += t2 - t1
            T["prog-bytes"] += 4 * len(words)
    B.Board.load_program = load_program

    t_run = [0.0]

    def start_(self, trace=None):
        t_run[0] = time.perf_counter()
        self.t.reg_write(B.R_CTRL, B.CTRL_CLEAR)
        self.t.reg_write(B.R_CTRL, B.CTRL_RUN)

    def wait(self, timeout=600.0, expect=0.0):
        t = self.t
        if expect:
            expect = max(expect - (time.perf_counter() - t_run[0]), 1e-9)
            t.poll(B.R_STATUS, B.ST_HALTED, B.ST_HALTED, timeout, expect)
        else:                               # the driver before the expect hint
            t.poll(B.R_STATUS, B.ST_HALTED, B.ST_HALTED, timeout)
        t1 = time.perf_counter()
        vals = t.reg_read_many(self.RUN_OFFS)
        t2 = time.perf_counter()
        st, lo, hi = vals[:3]
        cyc = lo | hi << 32
        t.reg_write(B.R_CTRL, 0)
        if on.is_set():
            dev = cyc / (khz_box[0] * 1e3)
            T["run"] += t1 - t_run[0]
            T["run.device"] += dev
            T["run.overshoot"] += t1 - t_run[0] - dev
            T["counters"] += t2 - t1
        return {"cycles": cyc, "instructions": [vals[3]], "b_reads": vals[4],
                "b_writes": vals[5], "a_reads": vals[6], "a_writes": vals[7],
                "b_stall": vals[8], "status": st}
    B.Board.start, B.Board.wait = start_, wait

    def run(self, timeout=600.0, trace=None, expect=0.0):     # a Board without start / wait
        start_(self, trace)
        return wait(self, timeout, expect)
    B.Board.run = run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="lfm2")
    ap.add_argument("--backend", default="board", choices=["board", "fake"])
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--prompt", default="Write a short story about a lighthouse keeper.")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake-ms", type=float, default=80.0)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    path = model_dir(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    spec = load_spec(path)
    khz = [100_000]
    instrument(khz)
    if a.backend == "board":
        backend, cfg = C.make_backend("board", spec, a.cap, a.dev, path.name)
    else:
        from opentpu.host.fake import FakeTransport
        from opentpu.isasim import board_config
        probe = spec.image(board_config(DRAM_BYTES=1 << 32), a.cap)
        ch = 1 << max(20, (probe.nbytes // 2 + (1 << 20)).bit_length())
        tr = FakeTransport(ch_bytes=ch, run_s=a.fake_ms / 1e3,
                           cycles=int(a.fake_ms * 1e5), devname=None)
        cfg = B.device_config(B.Board(tr, lock=False).info(), DRAM_BYTES=2 * ch)
        backend = lambda c, imgs: B.BoardBackend(c, imgs, transport=tr, model=path.name)  # noqa
    eng = Q.Engine(spec, Q.load_weights(path), cap=a.cap, cfg=cfg, backend=backend)
    khz[0] = (getattr(eng.backend, "info", {}) or {}).get("core_khz") or 100_000
    sp = C.sampling(spec, argparse.Namespace())
    pick = C.sampler(0 if a.greedy else sp["temperature"], sp["top_k"], sp["top_p"], a.seed,
                     sp["repetition_penalty"])

    # the trace (compile_step) of every Image class
    img_cls = type(eng.image)
    timed(img_cls, "compile_step", "trace")
    def pick_t(logits, ctx=()):
        t0 = time.perf_counter()
        r = pick(logits, ctx)
        if on.is_set():
            T["sample"] += time.perf_counter() - t0
        return r
    chat = C.Chat(eng, tok, False, pick_t, a.tokens, clock_mhz=khz[0] / 1e3)
    timed(tok, "decode", "detok")
    sink = io.StringIO()
    step0 = {}

    def upd(delta, turn):
        t0 = time.perf_counter()
        sink.write(delta)
        if turn.gen_tokens == 1:            # decode steps start after the first token
            on.set()
            step0["t"] = time.perf_counter()
            step0["n"] = len(eng.stats)
        if on.is_set():
            T["ui"] += time.perf_counter() - t0
    chat.ask(a.prompt, upd)
    on.clear()
    wall = time.perf_counter() - step0["t"]
    n = len(eng.stats) - step0["n"]
    cyc = sum(s["cycles"] for s in eng.stats[step0["n"]:])
    per = {k: 1e3 * v / n for k, v in T.items() if k != "prog-bytes"}
    known = ["io-write", "wait-compile", "prog-upload", "imem-load", "run", "counters",
             "status", "logits-read", "sample", "detok", "ui"]
    # nested: _program and the backend calls happen inside step; step excludes sample/detok/ui
    per["other"] = 1e3 * wall / n - sum(per.get(k, 0) for k in known)
    print(f"{path.name} on {a.backend}: {n} decode steps, prompt fed "
          f"{step0['n']} tokens; program {T['prog-bytes'] / max(n, 1) / 1024:.1f} KiB/token")
    print(f"{'item':<16} {'ms/token':>9}")
    for k in known + ["run.device", "run.overshoot", "other", "trace", "assemble", "step"]:
        if k in per:
            print(f"{k:<16} {per[k]:9.2f}" + ("   (compile thread)" if k in ("trace", "assemble")
                                               else ""))
    print(f"wall {n / wall:.2f} tok/s, device {n * khz[0] * 1e3 / cyc:.2f} tok/s "
          f"({cyc / n / 1e6:.2f} Mcycles/token)")
    if a.json:
        Path(a.json).write_text(json.dumps({"model": path.name, "steps": n, "ms": per,
                                            "wall_tok_s": n / wall,
                                            "dev_tok_s": n * khz[0] * 1e3 / cyc}, indent=1))
    eng._drain()
    eng.backend.close()


if __name__ == "__main__":
    main()
