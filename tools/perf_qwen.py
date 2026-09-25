"""Profile one decode token on the RTL at the board configuration (AXI memory path).

    python3 tools/perf_qwen.py [--model qwen3|lfm2|DIR] [--layers N] [--pos P] [--bw 100]
                               [--check]

Uses the real weights (models/Qwen3-0.6B, or --model lfm2: models/LFM2.5-230M), optionally only
the first N layers (the LM head is always complete). Prints cycles, the DRAM roofline (port-B
chunk transfers: weights, KV, LD/ST chunks), efficiency, tokens/s at an assumed 100 MHz, MM time
per kernel source line, and the MXU idle gaps with their causes.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu import isa as I  # noqa: E402
from opentpu import rtlsim  # noqa: E402
from opentpu.isasim import Machine, board_config  # noqa: E402
from opentpu.llm import MODELS, load_spec, model_dir  # noqa: E402
from opentpu.llm.qwen3 import load_weights, rope_tables  # noqa: E402
from opentpu.profile import parse  # noqa: E402


def _src(ins, depth=1):
    s = ins.src[depth] if len(ins.src) > depth else (ins.src[0] if ins.src else ("?", 0, "?"))
    return f"{Path(s[0]).name}:{s[1]} {s[2]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3",
                    help=f"{' or '.join(MODELS)} (models/<name>), or a checkpoint directory")
    ap.add_argument("--layers", type=int, default=2, help="0: all")
    ap.add_argument("--pos", type=int, default=9)
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--bw", type=int, default=100)
    ap.add_argument("--lat", type=int, default=30)
    ap.add_argument("--stall", type=int, default=0)
    ap.add_argument("--block", type=int, default=None, help="attention block (tokens)")
    ap.add_argument("--depth", type=int, default=None, help="attention score blocks in flight")
    ap.add_argument("--check", action="store_true", help="compare with the ISA simulator")
    ap.add_argument("--timeline", help="print the instructions of dynamic index range A:B")
    ap.add_argument("--idle", action="store_true", help="list DRAM-idle stretches (64-cycle windows)")
    a = ap.parse_args()
    path = model_dir(a.model)
    spec = load_spec(path)
    if a.layers:
        spec = dataclasses.replace(spec, **({"kinds": spec.kinds[:a.layers]}
                                            if hasattr(spec, "kinds") else {"layers": a.layers}))
    W = load_weights(path)
    need = spec.image(board_config(DRAM_BYTES=1 << 40), a.cap).nbytes
    cfg = board_config(DRAM_BYTES=1 << max(20, (need - 1).bit_length()))
    img = spec.image(cfg, a.cap)
    dram = img.build(W)[0]
    # this token's inputs (the KV cache before pos stays zero: timing does not depend on it)
    emb = np.asarray(W["model.embed_tokens.weight"][791], np.float32)
    c, s = rope_tables(spec, a.pos)
    for key, v in (("x", emb), ("cos", c), ("sin", s)):
        b = np.ascontiguousarray(v, np.float32).view(np.uint8)
        dram[img.io[key]:img.io[key] + b.size] = b
    if a.depth:
        import opentpu.llm.qwen3 as Q
        Q.ATTN_DEPTH = a.depth
    progs = img.compile_step(a.pos, *([a.block] if a.block else []))
    t = time.time()
    drams, _, st = rtlsim.run(cfg, progs, [dram], trace=True, uarch=rtlsim.BOARD_UARCH,
                              axi=True, boot=True, stall=a.stall, bw=a.bw, lat=a.lat,
                              max_cycles=1 << 40)
    wall = time.time() - t
    p = parse(st["trace"], cfg, progs, path.name)
    p.cycles = st["cycles"]
    rl = p.roofline()
    ideal = rl["bound"] * 100 / a.bw
    print(f"layers={spec.layers} pos={a.pos} bw={a.bw}% lat={a.lat}: {p.cycles} cycles "
          f"({wall:.0f}s sim), roofline {rl['bound']} chunks -> {ideal:.0f} cycles at this "
          f"bandwidth, efficiency {100 * ideal / p.cycles:.1f}%")
    # useful-bytes roofline: weights + their fp32 block scales + KV + activations, at D bytes
    # per cycle (both channels at 100%)
    D = cfg.D
    useful = 0
    for r in p.recs:
        ins = progs[0][r.pc]
        if r.op == I.MM:
            useful += r.portb * D + r.porta * 4
        elif r.op in (I.LD, I.ST):
            useful += 4 * ins.w[2]
        elif r.op == I.QST:
            useful += r.porta
    ub = useful / D * 100 / a.bw
    print(f"useful bytes {useful} -> {ub:.0f} cycles at this bandwidth: "
          f"efficiency {100 * ub / p.cycles:.1f}%")
    print(f"at an assumed 100 MHz: {p.cycles / 1e5:.1f} ms/token, {1e8 / p.cycles:.1f} tok/s "
          f"(simulated cycles, no host time)")
    print(p.summary())
    ph = defaultdict(lambda: [0, 0, 0])
    for r in p.recs:
        if r.op != I.MM or r.end < 0:
            continue
        k = _src(progs[0][r.pc])
        ph[k][0] += 1
        ph[k][1] += r.portb
        ph[k][2] += r.end - max(r.start, r.release)
    print("MM by source: count, chunks, release->end cycles")
    for k, v in sorted(ph.items(), key=lambda x: -x[1][1])[:20]:
        print(f"  {k:50s} {v[0]:6d} {v[1]:9d} {v[2]:9d}")
    g = p.mxu_gaps(0)
    print(f"MXU prologue {g['prologue']} epilogue {g['epilogue']} "
          f"gaps {sum(b - a for a, b, _ in g['gaps'])}")
    for key, v in sorted(g["blame"].items(), key=lambda x: -x[1])[:15]:
        src = _src(progs[0][key[1]], 0) if key[1] >= 0 else "?"
        print(f"  gap {v:8d}  after {key[0]:6s} pc={key[1]:5d} {src}")
    for k, v in sorted(p.by_class(0).items(), key=lambda x: -x[1]["busy"]):
        print(f"  {k:7s} n={v['count']:6d} busy={v['busy']:9d} work={v['work']:9d} "
              f"wait_dep={v['wait_dep']:9d} wait_unit={v['wait_unit']:9d}")
    if a.timeline:
        lo, hi = (int(x) for x in a.timeline.split(":"))
        for r in p.slice_recs(0)[lo:hi]:
            print(f"  #{r.idx:5d} pc={r.pc:4d} {r.name:9s} {r.detail:28s} disp={r.dispatch:8d} "
                  f"rdy={r.ready:8d} st={r.start:8d} rel={r.release:8d} end={r.end:8d}  "
                  f"{_src(progs[0][r.pc], 0)}")
    if a.idle:
        b = p.buckets[0]
        util = [(c, n, bm + bd) for c, n, bm, bd in zip(b["c"], b["n"], b["bm"], b["bd"])]
        tot, run = 0, None
        for c, n, u in util:
            lost = n - u
            tot += max(0, lost)
            if lost > n // 8:
                if run and c - n <= run[1] + 1:
                    run = (run[0], c, run[2] + lost)
                else:
                    if run and run[2] > 300:
                        print(f"  idle {run[2]:6d} in [{run[0]}, {run[1]}]")
                    run = (c - n, c, lost)
        if run and run[2] > 300:
            print(f"  idle {run[2]:6d} in [{run[0]}, {run[1]}]")
        print(f"  port-B idle cycles total {tot}")
    if a.check:
        m = Machine(cfg, [progs[0]], [dram.copy()])
        m.run()
        ok = np.array_equal(m.slices[0].dram[:img.nbytes], drams[0][:img.nbytes])
        print("bit-exact vs ISA:", ok)


if __name__ == "__main__":
    main()
