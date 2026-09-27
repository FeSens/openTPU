"""Where the host time of one decode token goes: a timeline of otpu-chat's decode loop.

    python3 tools/decode_profile.py --model lfm2 [--backend board | fake] [--tokens 64]
                                    [--prompt "..."] [--greedy] [--no-stream] [--json out.json]
                                    [--wformat int8|fp4|int4] [--head-format int8|fp4|int4]

Runs one Chat turn (plain mode, the reply printed to /dev/null) on the card (--backend
board) or on FakeTransport (--backend fake: a card that computes nothing and halts after
--fake-ms; the host path only, without streamed logits).

The number that matters is the host's critical path per token: from the moment the host sees
HALTED to the moment it writes RUN for the next token (the card idles in between). The rest of
the host's work overlaps a run. Every transport operation (DMA write / read, register read /
write, the HALTED poll with its sleeps) is timed and filed under the host step it serves, as
"critical" (between HALTED and RUN) or "overlapped" (while the card runs):

  io-write        x / cos / sin of the next token (BoardBackend.write)
  prog-upload     the program's DMA to the program area (inside Board.load_program)
  imem-load       LOAD .. not LOADING (Board.load_program's registers and poll)
  start           CLEAR, RUN (and the trace registers)
  counters        the HALTED poll and the counter registers after it (Board.wait); its critical
                  part starts when the run ends: the poll's wake-up and the register reads
  logits-stream   the logits pieces read while the card runs (streamed logits) and their
                  sentinel re-marking; logits-tail: what is read after HALTED
  logits-read     the logits read after HALTED without streaming (--no-stream)
  sample, detok, ui, status   host computation (no transport)
  compile-wait    the step waiting for the precompiled program (Engine._program)
  other           the critical window's time not in any item above

Prints the mean per decode token (the first generated token and the prefill are excluded),
the transport operations per token (count, bytes, time) and wall vs device tokens/s.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu.host import board as B  # noqa: E402
from opentpu.host import chat as C  # noqa: E402
from opentpu.host import regs as R  # noqa: E402
from opentpu.host import runstate  # noqa: E402
from opentpu.llm import load_spec, model_dir  # noqa: E402
from opentpu.llm import qwen3 as Q  # noqa: E402

COMPUTE = ("sample", "detok", "ui", "status", "compile-wait")    # items timed directly
KNOWN = ["io-write", "args-write", "compile-wait", "prog-upload", "imem-load", "start",
         "counters", "logits-stream", "logits-tail", "logits-read", "status", "sample", "detok",
         "ui"]


class Profiler:
    """Per-token accounting. `stack` holds the host step being run (the innermost wins); the
    outermost transport operation of a call chain adds its time to [step][critical or
    overlapped] and to ops[op] (calls, bytes, seconds); the COMPUTE steps add their own time."""

    def __init__(self):
        self.on = threading.Event()
        self.t = defaultdict(lambda: [0.0, 0.0])        # item -> [critical, overlapped] s
        self.ops = defaultdict(lambda: [0, 0, 0.0])     # op -> [calls, bytes, seconds]
        self.stack = ["other"]
        self.depth = 0                                  # transport operations in progress
        self.running = False                            # between RUN and HALTED seen
        self.t_halt = self.t_run = None
        self.crit, self.windows = 0.0, 0                # sum of HALTED seen -> next RUN
        self.runs = []                                  # RUN written -> HALTED seen, s

    def add(self, item, dt, running=None):
        if self.on.is_set():
            self.t[item][int(self.running if running is None else running)] += dt

    def step(self, owner, name, item):
        """Time owner.name as host step `item`."""
        f, prof = getattr(owner, name), self

        def w(*a, **k):
            prof.stack.append(item)
            t0, run0 = time.perf_counter(), prof.running
            try:
                return f(*a, **k)
            finally:
                prof.stack.pop()
                if item in COMPUTE:
                    prof.add(item, time.perf_counter() - t0, run0)
        setattr(owner, name, w)

    def op(self, t, name, op, nbytes=None, after=None):
        """Time transport method t.name as operation `op`; after(args, result) runs last."""
        f, prof = getattr(t, name), self

        def w(*a, **k):
            prof.depth += 1
            t0, run0 = time.perf_counter(), prof.running
            try:
                r = f(*a, **k)
            finally:
                prof.depth -= 1
            dt = time.perf_counter() - t0
            if after is not None:                       # HALTED seen / RUN written: first
                after(a, r)
            if prof.depth == 0 and prof.on.is_set():
                o = prof.ops[op]
                o[0] += 1
                o[1] += nbytes(a) if nbytes else 0
                o[2] += dt
                if run0 and not prof.running:           # HALTED seen inside: split the time
                    prof.add(prof.stack[-1], prof.t_halt - t0, True)
                    prof.add(prof.stack[-1], time.perf_counter() - prof.t_halt, False)
                else:
                    prof.add(prof.stack[-1], dt, run0)
            return r
        setattr(t, name, w)

    def instrument_transport(self, t):
        prof = self

        def run_written(a, r):
            off, val = a[0], a[1]
            if off == R.R_CTRL and val & R.CTRL_RUN:
                now = time.perf_counter()
                if prof.on.is_set() and prof.t_halt is not None:
                    prof.crit += now - prof.t_halt
                    prof.windows += 1
                prof.t_halt, prof.t_run, prof.running = None, now, True

        def status_read(a, r):
            if prof.running and (
                    (a[0] == R.R_STATUS and r & R.ST_HALTED) or           # reg_read
                    (len(a) > 2 and a[0] == R.R_STATUS and a[1] & R.ST_HALTED)):  # poll
                prof.t_halt, prof.running = time.perf_counter(), False
                if prof.on.is_set():
                    prof.runs.append(prof.t_halt - prof.t_run)
        self.op(t, "mem_write", "dma-write", lambda a: len(a[2]))
        self.op(t, "mem_read", "dma-read", lambda a: a[2])
        self.op(t, "reg_write", "reg-write", after=run_written)
        self.op(t, "reg_read", "reg-read", after=status_read)
        self.op(t, "reg_read_many", "reg-read-many")
        self.op(t, "poll", "poll", after=status_read)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="lfm2")
    ap.add_argument("--backend", default="board", choices=["board", "fake"])
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--prompt", default="Write a short story about a lighthouse keeper.")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--no-stream", action="store_true",
                    help="read the logits after the run (no streamed logits)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake-ms", type=float, default=80.0)
    ap.add_argument("--per-position", action="store_true",
                    help="a decode program per position (default: the resident one when the "
                         "bitstream takes run arguments)")
    ap.add_argument("--fake-no-args", action="store_true",
                    help="--backend fake: a bitstream without run arguments (CAPS bit24)")
    ap.add_argument("--json")
    ap.add_argument("--wformat", default="int8", choices=["int8", "fp4", "int4"],
                    help="weight format of the layers (docs/quant.md)")
    ap.add_argument("--head-format", default=None, choices=["int8", "fp4", "int4"],
                    help="weight format of the LM head (default: --wformat)")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    path = model_dir(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    spec = load_spec(path)
    P = Profiler()
    wkw = dict(wformat=a.wformat, head_format=a.head_format)
    if a.backend == "board":
        backend, cfg = C.make_backend("board", spec, a.cap, a.dev, path.name)
    else:
        from opentpu.host.fake import FakeTransport
        from opentpu.isasim import board_config
        probe = spec.image(board_config(DRAM_BYTES=1 << 32), a.cap, **wkw,
                           **({"lookup": True} if not a.per_position and Q.has_lookup(spec)
                              else {}))
        ch = 1 << max(20, (probe.nbytes // 2 + (1 << 20)).bit_length())
        tr = FakeTransport(ch_bytes=ch, run_s=a.fake_ms / 1e3,
                           cycles=int(a.fake_ms * 1e5), devname=None,
                           args=not a.fake_no_args)
        cfg = B.device_config(B.Board(tr, lock=False).info(), DRAM_BYTES=2 * ch)
        backend = lambda c, imgs: B.BoardBackend(c, imgs, transport=tr, model=path.name)  # noqa
    # host steps: their transport operations are filed under them
    P.step(B.BoardBackend, "write", "io-write")
    P.step(B.Board, "load_program", "imem-load")
    P.step(B.Board, "set_args", "args-write")
    P.step(B.Board, "start", "start")
    P.step(B.Board, "wait", "counters")
    P.step(B.BoardBackend, "_stream_logits", "logits-stream")
    P.step(B.BoardBackend, "_stream_tail", "logits-tail")
    P.step(B.BoardBackend, "read", "logits-read")
    P.step(runstate.RunnerStatus, "token", "status")
    P.step(Q.Engine, "_program", "compile-wait")
    P.step(Q.Engine, "_decode", "compile-wait")
    b_write = B.Board.write

    def board_write(self, addr, data):                  # the program's DMA in load_program
        if P.stack[-1] != "imem-load":
            return b_write(self, addr, data)
        P.stack.append("prog-upload")
        try:
            return b_write(self, addr, data)
        finally:
            P.stack.pop()
    B.Board.write = board_write

    eng = Q.Engine(spec, Q.load_weights(path), cap=a.cap, cfg=cfg, backend=backend, **wkw,
                   resident=not a.per_position)
    eng.stream_logits = not a.no_stream
    if a.backend == "fake":         # logits the sampler works on as on real ones (no ties)
        import numpy as np
        lg = np.random.default_rng(0).normal(0, 3, eng.image.v_loc).astype(np.float32)
        eng.backend.board.write(eng.image.io["logits"], lg)
    khz = (getattr(eng.backend, "info", {}) or {}).get("core_khz") or 100_000
    P.instrument_transport(eng.backend.board.t)
    sp = C.sampling(spec, argparse.Namespace())
    pick = C.sampler(0 if a.greedy else sp["temperature"], sp["top_k"], sp["top_p"], a.seed,
                     sp["repetition_penalty"])
    step0 = {}

    def first():                                        # the first pick: the prefill is done
        if "t" not in step0:
            step0["t"] = time.perf_counter()
            step0["n"] = len(eng.stats)
            P.t_halt = P.t_halt or time.perf_counter()
            P.on.set()

    def timed_pick(logits, ctx=()):
        first()
        t0 = time.perf_counter()
        r = pick(logits, ctx)
        P.add("sample", time.perf_counter() - t0)
        return r

    class TimedStream:                                  # pick.stream, its work timed
        def __init__(self, ctx):
            self.s = pick.stream(ctx)

        def begin(self, n, *x):
            self.s.begin(n, *x)

        def feed(self, lo, v):
            t0 = time.perf_counter()
            self.s.feed(lo, v)
            P.add("sample", time.perf_counter() - t0)

        def result(self):
            first()
            t0 = time.perf_counter()
            r = self.s.result()
            P.add("sample", time.perf_counter() - t0)
            return r
    timed_pick.stream = TimedStream
    chat = C.Chat(eng, tok, False, timed_pick, a.tokens, clock_mhz=khz / 1e3)
    dec = tok.decode

    def decode(*x, **k):
        t0 = time.perf_counter()
        try:
            return dec(*x, **k)
        finally:
            P.add("detok", time.perf_counter() - t0)
    tok.decode = decode
    sink = io.StringIO()

    def upd(delta, turn):
        t0 = time.perf_counter()
        sink.write(delta)
        P.add("ui", time.perf_counter() - t0)
    chat.ask(a.prompt, upd)
    P.on.clear()
    wall = time.perf_counter() - step0["t"]
    n = len(eng.stats) - step0["n"]
    cyc = sum(s["cycles"] for s in eng.stats[step0["n"]:])
    dev_ms = 1e3 * cyc / n / (khz * 1e3)
    per = {k: (1e3 * v[0] / n, 1e3 * v[1] / n) for k, v in P.t.items()}
    crit = 1e3 * P.crit / max(P.windows, 1)
    crit_known = sum(per.get(k, (0, 0))[0] for k in KNOWN)
    bid = (getattr(eng.backend, "info", {}) or {}).get("build_id")
    fmt = a.wformat + (f", head {a.head_format}" if a.head_format else "")
    streamed = eng.stream_logits and getattr(eng.backend, "streams", False)
    print(f"{path.name} ({fmt}, {'streamed logits' if streamed else 'no stream'}) on "
          f"{a.backend}" + ("" if bid is None else f" (build {bid:08x})")
          + f": {n} decode steps, prompt fed {step0['n']} tokens")
    print(f"{'ms per token':<15} {'critical':>9} {'overlapped':>11}")
    for k in KNOWN:
        if k in per:
            print(f"{k:<15} {per[k][0]:9.3f} {per[k][1]:11.3f}")
    print(f"{'other':<15} {max(0.0, crit - crit_known):9.3f}")
    over = 1e3 * sum(P.runs) / max(len(P.runs), 1) - dev_ms
    print(f"host critical path (HALTED seen -> next RUN): {crit:.3f} ms/token over "
          f"{P.windows} tokens; device {dev_ms:.3f} ms/token; HALTED seen {over:.3f} ms after "
          f"the run's end (poll overshoot)")
    print("transport per token: " + ", ".join(
        f"{k} {v[0] / n:.1f}x {v[1] / n / 1024:.1f} KiB {1e3 * v[2] / n:.3f} ms"
        for k, v in sorted(P.ops.items())))
    ls = getattr(eng.backend, "last_stream", None) if streamed else None
    if ls:
        print(f"streamed logits (last token): {ls.get('during')} of {ls.get('pieces')} pieces "
              f"during the run, {ls.get('probes')} probes, then "
              f"{ls.get('tail_bytes', 0) / 1024:.0f} KiB in {1e3 * ls.get('tail_s', 0):.3f} ms")
    print(f"wall {n / wall:.2f} tok/s, device {n * khz * 1e3 / cyc:.2f} tok/s "
          f"({cyc / n / 1e6:.3f} Mcycles/token)")
    if a.json:
        Path(a.json).write_text(json.dumps({
            "model": path.name, "wformat": a.wformat, "head_format": a.head_format,
            "streamed": streamed, "steps": n, "build_id": bid,
            "ms": {k: {"critical": c, "overlapped": o} for k, (c, o) in per.items()},
            "critical_ms": crit, "device_ms": dev_ms, "overshoot_ms": over,
            "ops": {k: {"calls": v[0] / n, "bytes": v[1] / n, "ms": 1e3 * v[2] / n}
                    for k, v in P.ops.items()},
            "wall_tok_s": n / wall, "dev_tok_s": n * khz * 1e3 / cyc,
            "reply_ids": [int(x) for x in chat._reply]}, indent=1))
    eng._drain()
    eng.backend.close()


if __name__ == "__main__":
    main()
