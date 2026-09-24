"""otpu-lens: openTPU Lens profiles recorded on the card, from its hardware trace buffer.

    otpu-lens record --dev /dev/xdma0 --model models/Qwen3-0.6B --prompt "Hi" --tokens 2 \\
                     [--pos P] [--keep first|last] -o card.otpuprof
    otpu-lens record --sim [--workload mlp-small | qwen-tiny] [--pos P] -o sim.otpuprof
    otpu-lens open|html|info|list ...          # passed to opentpu.lens (python -m opentpu.lens)

record runs Qwen3 decode steps on the card (the prompt's tokens, then greedy tokens); the steps
at positions P .. P+N-1 (default P: the prompt's last token, whose step yields the first new
token) run with the trace buffer on (TRACE_CTRL). After each traced step the 64-bit records
are read back (TRACE_ADDR / TRACE_LO / TRACE_HI) and rebuilt into the simulator's +trace text by
opentpu.hwtrace.records_to_trace; opentpu.profile.parse and opentpu.lens.to_data turn that into
the same profile JSON `lens record` makes from an RTL simulation (kind "hw", one profile per
traced step). --sim does the same on the Verilator board model with a small workload (a kernel
workload of opentpu.lens, or one qwen-tiny step).

Lost data is reported in each profile's "hwtrace" metadata, as a note, and on stdout:
TRACE_DROP (events the capture queue dropped) and the records that did not fit the buffer
(--keep first: the run's tail is missing; --keep last: the ring wrapped and the head is
missing). A bitstream without the trace buffer (register map 1) gives a counters-only profile
(kind "board", opentpu.lens.board_data).
"""
from __future__ import annotations

import argparse
import inspect
import re
import sys
import time
from pathlib import Path

import numpy as np

from opentpu import lens as L
from opentpu import rtlsim

ROOT = Path(__file__).resolve().parents[2]
_D = re.compile(r"^T(\d+) D c=\d+ s=\d+ pc=(\d+)\s*$")


class NoHwTrace(RuntimeError):
    pass


# ------------------------------------------------------------------------------ records -> trace
def records_to_trace(records: np.ndarray, programs: list, sid: int = 0) -> str:
    """The trace text of one slice's records, via opentpu.hwtrace (imported here: it comes with
    the RTL side of the observability work). D lines get their op from the program the host
    loaded when the rebuild leaves it out (profile.parse needs it)."""
    try:
        from opentpu import hwtrace
    except ImportError as e:
        raise NoHwTrace("opentpu.hwtrace is missing (records_to_trace(records, sid=0) -> str, "
                        "docs/observability.md): this checkout has the host side only") from e
    fn = hwtrace.records_to_trace
    ps = inspect.signature(fn).parameters
    kw = {}
    if "programs" in ps:
        kw["programs"] = programs
    elif "program" in ps:
        kw["program"] = programs[sid]
    return fill_ops(fn(records, sid=sid, **kw), programs)


def fill_ops(text: str, programs: list) -> str:
    """`T<s> D c= s= pc=` -> `... op=<op> w1= w2= w3=` from the static program (w1..w3 are the
    unresolved operand words: the RTL prints them after register offsets, parse ignores them)."""
    out = []
    for line in text.splitlines():
        m = _D.match(line.strip())
        if m:
            ins = programs[int(m.group(1))][int(m.group(2))]
            line = (f"{line.strip()} op={ins.op:02x} w1={ins.w[0] & 0xFFFFFFFF:08x} "
                    f"w2={ins.w[1] & 0xFFFFFFFF:08x} w3={ins.w[2] & 0xFFFFFFFF:08x}")
        out.append(line)
    return "\n".join(out)


def drop_orphans(text: str) -> str:
    """After a ring wrap the first records may belong to instructions whose D record was
    overwritten: drop S / G / E lines of slots with no D before them. (U lines are paired with
    completions by order per unit; a few at the head may then be misattributed.)"""
    open_: set = set()
    out = []
    ev = re.compile(r"^T(\d+) ([DSGE]) c=\d+ s=(\d+)")
    for line in text.splitlines():
        m = ev.match(line.strip())
        if m:
            key = (m.group(1), m.group(3))
            if m.group(2) == "D":
                open_.add(key)
            elif key not in open_:
                continue
            elif m.group(2) == "E":
                open_.discard(key)
        out.append(line)
    return "\n".join(out)


def hw_profile(name: str, cfg, programs: list, stats: dict, core_khz: int | None,
               uarch: dict | None = None) -> dict:
    """A traced run (Board.run(trace=...) stats) -> Lens profile dict (kind "hw")."""
    from opentpu.profile import parse
    uarch = dict(rtlsim.BOARD_UARCH) if uarch is None else uarch
    tr = stats["trace"]
    text = records_to_trace(tr["records"], programs)
    if tr["wrapped"]:
        text = drop_orphans(text)
    p = parse(text, cfg, programs, name)
    last = p.cycles
    open_ = 0
    for r in p.recs:                    # the buffer filled up (keep first): close what is open
        if r.start >= 0 and r.end < 0:
            r.end, open_ = last, open_ + 1
    p.cycles = stats["cycles"]
    d = L.to_data(p, uarch, kind="hw")
    d["clock_mhz"] = core_khz / 1e3 if core_khz else L.CLOCK_MHZ
    d["config"]["MEM"] = "ddr3"
    d["board"] = {k: v for k, v in stats.items() if k != "trace"}
    meta = {k: tr[k] for k in ("count", "drop", "depth", "keep", "wrapped", "lost")}
    meta.update(records=int(len(tr["records"])), open_at_end=open_,
                traced_to=int(last), complete=not tr["drop"] and not tr["lost"])
    d["hwtrace"] = meta
    for msg in lost_text(meta):
        d["notes"].insert(0, {"level": "warn", "text": msg})
    return d


def lost_text(m: dict) -> list[str]:
    out = []
    if m["drop"]:
        out.append(f"The trace capture queue dropped {m['drop']} events (TRACE_DROP): "
                   "the timeline misses them.")
    if m["lost"]:
        out.append(f"{m['lost']} of {m['count']} trace records did not fit the {m['depth']}-record "
                   "buffer: " + ("the ring kept the last ones, the start of the run is missing."
                                 if m["keep"] == "last" else
                                 f"the buffer kept the first ones, the run after cycle "
                                 f"{m.get('traced_to', '?')} is missing."))
    return out


def counters_profile(name: str, cfg, programs: list, stats: dict, core_khz) -> dict:
    d = L.board_data(name, cfg, programs, {k: v for k, v in stats.items() if k != "trace"})
    d["clock_mhz"] = core_khz / 1e3 if core_khz else L.CLOCK_MHZ
    d["notes"].insert(0, {"level": "warn", "text": "No trace buffer in this bitstream (register "
                          "map 1): counters only."})
    return d


# ------------------------------------------------------------------------------ recording
def _profiles_of_steps(steps: list, name: str, cfg, core_khz, save_raw=None) -> list:
    out = []
    for pos, (programs, st) in steps:
        nm = f"{name} token at pos {pos}"
        if "trace" not in st:
            out.append(counters_profile(nm, cfg, programs, st, core_khz))
            continue
        try:
            out.append(hw_profile(nm, cfg, programs, st, core_khz))
        except NoHwTrace:
            if save_raw:
                np.save(save_raw, st["trace"]["records"])
                print(f"raw trace records saved to {save_raw}", file=sys.stderr)
            raise
    return out


def run_steps(eng, tokens: list[int], pos0: int, n: int, keep: str, prompt_len: int) -> list:
    """Engine steps from position 0: tokens[i] while i < prompt_len, then greedy; the steps at
    positions pos0 .. pos0 + n - 1 traced. Returns [(pos, (programs, stats))]."""
    be = eng.backend
    trace_ok = be.info["regmap"] >= 2 and be.info["caps"]["trace"]
    got, tok = [], tokens[0]
    for pos in range(pos0 + n):
        be.trace = {"keep": keep} if pos >= pos0 and trace_ok else None
        logits = eng.step(tok)
        if pos >= pos0:
            got.append((pos, be.last))
        tok = tokens[pos + 1] if pos + 1 < prompt_len else int(np.argmax(logits))
    be.trace = None
    return got


def record_card(a) -> list:
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import Engine, Spec, load_weights
    from .board import BoardBackend, XdmaTransport
    model = Path(a.model)
    spec = Spec.from_hf(model)
    if a.prompt_ids:
        ids = [int(x) for x in a.prompt_ids.split(",")]
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                      add_generation_prompt=True, enable_thinking=False,
                                      tokenize=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    pos0 = len(ids) - 1 if a.pos is None else a.pos
    cap = max(256, -(-(pos0 + a.tokens + 1) // 128) * 128)
    tr = XdmaTransport(a.dev)
    eng = Engine(spec, load_weights(model), cap=cap, cfg=board_config(),
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr, model=model.name))
    return _record_engine(eng, ids, pos0, a, f"{model.name} on {a.dev}")


def _record_engine(eng, ids, pos0, a, name) -> list:
    be = eng.backend
    if be.info["regmap"] < 2 or not be.info["caps"]["trace"]:
        print(f"note: no trace buffer (register map {be.info['regmap']}): counters-only "
              "profiles", file=sys.stderr)
    t0 = time.time()
    steps = run_steps(eng, ids, pos0, a.tokens, a.keep, len(ids))
    print(f"{pos0 + a.tokens} steps in {time.time() - t0:.1f}s, {len(steps)} traced",
          flush=True)
    raw = Path(a.out).with_suffix(".records.npy")
    return _profiles_of_steps(steps, name, eng.cfg, be.info["core_khz"], raw)


def record_sim(a) -> list:
    """The board model: a kernel workload of opentpu.lens (one run), or qwen-tiny steps."""
    from .board import Board, BoardBackend, SimTransport, sim_config
    if a.workload == "qwen-tiny":
        from opentpu.llm.qwen3 import Engine
        spec, W = L._tiny_qwen()
        pos0 = 3 if a.pos is None else a.pos
        cap = max(256, -(-(pos0 + a.tokens + 1) // 128) * 128)
        cfg = sim_config(spec, cap)
        tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2)
        eng = Engine(spec, W, cap=cap, cfg=cfg,
                     backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
        ids = [int(t) for t in np.random.default_rng(1).integers(0, spec.vocab, pos0 + 1)]
        return _record_engine(eng, ids, pos0, a, "tiny Qwen3 (board model)")
    from opentpu import isa as I
    from opentpu.runtime import compile_kernel
    wl = L._kernel_workloads()
    if a.workload not in wl:
        raise SystemExit(f"unknown workload {a.workload!r}: {', '.join(wl)} or qwen-tiny")
    kernel, cfg, args, name = wl[a.workload](True)
    comp, imgs = compile_kernel(kernel, cfg, **args)
    # the program goes above the kernel's DRAM (the model's memory is twice as large)
    t = SimTransport(ch_bytes=cfg.DRAM_BYTES)
    b = Board(t)
    info = b.info()
    b.write(0, imgs[0])
    b.load_program(cfg.DRAM_BYTES, np.asarray(I.assemble(comp.programs[0]), np.uint32))
    traced = info["regmap"] >= 2 and info["caps"]["trace"]
    if not traced:
        print(f"note: the board model has no trace buffer (register map {info['regmap']}): "
              "counters only", file=sys.stderr)
    st = b.run(trace={"keep": a.keep} if traced else None)
    nm = f"{name} (board model)"
    if not traced:
        return [counters_profile(nm, cfg, comp.programs, st, info["core_khz"])]
    return [hw_profile(nm, cfg, comp.programs, st, info["core_khz"])]


# ------------------------------------------------------------------------------ CLI
PASS = ("open", "html", "info", "list")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in PASS:
        L.main(argv)
        return 0
    ap = argparse.ArgumentParser(prog="otpu-lens", description="openTPU Lens on the card: "
                                 "record profiles from the hardware trace buffer. Also: "
                                 "otpu-lens open|html|info|list (opentpu.lens).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="run Qwen3 steps on the card (or a workload on the board "
                       "model) with the trace buffer on; write a profile file")
    r.add_argument("-o", "--out", required=True)
    r.add_argument("--dev", default="/dev/xdma0", help="XDMA device prefix")
    r.add_argument("--sim", action="store_true", help="the Verilator board model")
    r.add_argument("--model", default=str(ROOT / "models" / "Qwen3-0.6B"))
    r.add_argument("--prompt", default="What is the capital of France?")
    r.add_argument("--prompt-ids", help="comma-separated token ids instead of --prompt "
                   "(no tokenizer needed)")
    r.add_argument("--tokens", type=int, default=1, help="steps to trace (one profile each)")
    r.add_argument("--pos", type=int, help="position of the first traced step (default: the "
                   "prompt's last token; --sim qwen-tiny: 3)")
    r.add_argument("--keep", choices=["first", "last"], default="first",
                   help="when the buffer fills: keep the first records (STOP_WHEN_FULL) or the "
                   "last (ring)")
    r.add_argument("--workload", default="mlp-small",
                   help="--sim: a kernel workload of `lens list`, or qwen-tiny")
    r.add_argument("--open", action="store_true", help="open the app afterwards")
    for c in PASS:
        sub.add_parser(c, help=f"-> python -m opentpu.lens {c}")
    a = ap.parse_args(argv)
    try:
        profs = record_sim(a) if a.sim else record_card(a)
    except NoHwTrace as e:
        print(f"otpu-lens: {e}", file=sys.stderr)
        return 2
    for d in profs:
        h = d.get("hwtrace")
        extra = ""
        if h:
            extra = (f", {h['records']} trace records (count {h['count']}, drop {h['drop']}, "
                     f"lost {h['lost']}{', ring wrapped' if h['wrapped'] else ''})")
        print(f"{d['name']}: {d['cycles']} cycles, "
              f"{100 * d['roofline']['efficiency']:.1f}% of roofline{extra}", flush=True)
        for msg in lost_text(h) if h else []:
            print(f"  warning: {msg}")
    out = L.save(profs, a.out)
    print(f"wrote {out}")
    if a.open:
        L.serve(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
