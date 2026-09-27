"""Profile one decode token on the RTL at the board configuration (AXI memory path).

    python3 tools/perf_qwen.py [--model qwen3|lfm2|qwen35|DIR] [--layers N] [--pos P]
                               [--bw 100] [--check]

Uses the real weights (models/Qwen3-0.6B, or --model lfm2: models/LFM2.5-230M, --model qwen35:
models/Qwen3.5-0.8B), optionally only the first N layers (the LM head is always complete).
Prints cycles, the DRAM roofline (port-B chunk transfers: weights, KV, LD/ST chunks), the
useful-bytes roofline, efficiency, tokens/s at an assumed 100 MHz, the cycles per phase of the
token (DeltaNet, attention, MLP, LM head) with their bytes, MM time per kernel source line, and
the MXU idle gaps with their causes.
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


# kernel functions that name a phase of the token (the innermost one on an instruction's source
# stack wins)
PHASE_NAMES = {"head_step": "DeltaNet", "_deltanet": "DeltaNet", "_pair_segment": "DeltaNet",
               "_attention": "attention", "_attend_heads": "attention", "_conv": "conv",
               "_mlp": "MLP", "swiglu_down": "MLP", "_lm_head": "LM head"}


def _phase(ins):
    for _, _, fn in ins.src:
        if fn in PHASE_NAMES:
            return PHASE_NAMES[fn]
    return "other"


def _src(ins, depth=1):
    s = ins.src[depth] if len(ins.src) > depth else (ins.src[0] if ins.src else ("?", 0, "?"))
    return f"{Path(s[0]).name}:{s[1]} {s[2]}"


def _gap_table(p, progs, pbytes, bpc):
    """Per phase: cycles, roofline (useful bytes at bpc bytes per cycle) and the excess, and the
    phase's cycles split by the MXU's state (MAC / starved: chunk FIFO empty / blocked: chunks
    but no MAC (scales or the drain) / no MM streaming) and port B's idle cycles. A 64-cycle
    counter window goes to the phase whose run (as Profile.phases charges them) holds its end."""
    runs, done = [], 0
    for r in sorted(p.slice_recs(0), key=lambda r: r.idx):
        if r.end > done:
            ph = _phase(progs[0][r.pc])
            if runs and runs[-1][2] == ph:
                runs[-1][1] = r.end
            else:
                runs.append([done, r.end, ph])
            done = r.end
    b = p.buckets[0]
    acc = defaultdict(lambda: defaultdict(int))
    k = 0
    for i, c in enumerate(b["c"]):
        while k + 1 < len(runs) and runs[k][1] < c:
            k += 1
        t = acc[runs[k][2] if runs else "other"]
        n = b["n"][i]
        t["n"] += n
        t["mac"] += b["mx"][i]
        t["starve"] += b["ms"][i]
        t["block"] += b["mb"][i]
        t["bidle"] += n - b["bm"][i] - b["bd"][i]
    cyc = p.phases(lambda r: _phase(progs[0][r.pc]))
    print("gap table: cycles, roofline, excess; MXU MAC / starved / blocked / no MM; "
          "port B idle")
    rows = []
    for ph, v in cyc.items():
        rl = pbytes[ph] / bpc
        t = acc[ph]
        idle = t["n"] - t["mac"] - t["starve"] - t["block"]
        rows.append((v["cycles"] - rl, ph, v["cycles"], rl, t, idle))
    for ex, ph, cy, rl, t, idle in sorted(rows, key=lambda x: -x[0]):
        print(f"  {ph:12s} {cy:9d} {rl:9.0f} {ex:+9.0f}   MAC {t['mac']:8d}  starved "
              f"{t['starve']:7d}  blocked {t['block']:7d}  no MM {idle:7d}   B idle {t['bidle']:8d}")
    # the MXU's starved / blocked windows by MM source line (the MM consuming at the window's end)
    mms = sorted((r for r in p.slice_recs(0) if r.op == I.MM and r.end >= 0),
                 key=lambda r: max(r.start, r.release))
    by = defaultdict(lambda: [0, 0, 0])
    k = 0
    for i, c in enumerate(b["c"]):
        while k + 1 < len(mms) and max(mms[k + 1].start, mms[k + 1].release) <= c:
            k += 1
        if mms and max(mms[k].start, mms[k].release) <= c <= mms[k].end + b["n"][i]:
            t = by[_src(progs[0][mms[k].pc], 0)]
            t[0] += b["ms"][i]
            t[1] += b["mb"][i]
            t[2] += b["mx"][i]
    print("  MXU starved / blocked / MAC by MM source (top 8 by starved + blocked)")
    for src, (ms, mb, mx) in sorted(by.items(), key=lambda x: -(x[1][0] + x[1][1]))[:8]:
        print(f"    {src:50s} starved {ms:7d}  blocked {mb:7d}  MAC {mx:8d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3",
                    help=f"{' or '.join(MODELS)} (models/<name>), or a checkpoint directory")
    ap.add_argument("--layers", type=int, default=2, help="0: all")
    ap.add_argument("--pos", type=int, default=9)
    ap.add_argument("--cap", type=int, default=None,
                    help="KV cache capacity (tokens); default: the multiple of 256 above pos")
    ap.add_argument("--bw", type=int, default=100)
    ap.add_argument("--lat", type=int, default=30)
    ap.add_argument("--stall", type=int, default=0)
    ap.add_argument("--arc", type=int, default=0,
                    help="cycles per AXI read transaction and channel (the board's is about 4)")
    ap.add_argument("--bl", type=int, default=8,
                    help="AXI read burst, beats (1: single-beat reads, as before bursts)")
    ap.add_argument("--dram", choices=["off", "brc", "rbc"], default="off",
                    help="DDR3 bank / row timing (otpu_axi_mem.sv) with the MIG's address map "
                         "BANK_ROW_COLUMN or ROW_BANK_COLUMN (replaces --bw)")
    ap.add_argument("--plus", action="append", default=[],
                    help="extra simulator argument, e.g. --plus +axi_trfc=26 (repeatable)")
    ap.add_argument("--block", type=int, default=None, help="attention block (tokens)")
    ap.add_argument("--depth", type=int, default=None, help="attention score blocks in flight")
    ap.add_argument("--check", action="store_true", help="compare with the ISA simulator")
    ap.add_argument("--timeline", help="print the instructions of dynamic index range A:B")
    ap.add_argument("--idle", action="store_true", help="list DRAM-idle stretches (64-cycle windows)")
    ap.add_argument("--gaps", action="store_true",
                    help="per phase: the cycles above its roofline, split by what the MXU and "
                         "port B did (64-cycle windows)")
    a = ap.parse_args()
    path = model_dir(a.model)
    spec = load_spec(path)
    if a.layers:
        spec = dataclasses.replace(spec, **({"kinds": spec.kinds[:a.layers]}
                                            if hasattr(spec, "kinds") else {"layers": a.layers}))
    if a.cap is None:
        a.cap = 256 * (a.pos // 256 + 1)
    if a.pos >= a.cap:
        ap.error(f"--pos {a.pos} needs --cap above it (the KV write would land past the cache)")
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
    drams, _, st = rtlsim.run(cfg, progs, [dram], trace=True,
                              uarch={**rtlsim.BOARD_UARCH, "AXI_BL": a.bl},
                              axi=True, boot=True, stall=a.stall, bw=a.bw, lat=a.lat, arc=a.arc,
                              max_cycles=1 << 40,
                              plusargs=([] if a.dram == "off" else
                                        ["+axi_dram=1", f"+axi_map={int(a.dram == 'rbc')}"])
                              + a.plus)
    wall = time.time() - t
    p = parse(st["trace"], cfg, progs, path.name)
    p.cycles = st["cycles"]
    rl = p.roofline()
    ideal = rl["bound"] * 100 / a.bw
    print(f"layers={spec.layers} pos={a.pos} bw={a.bw}% lat={a.lat}: {p.cycles} cycles "
          f"({wall:.0f}s sim), roofline {rl['bound']} chunks -> {ideal:.0f} cycles at this "
          f"bandwidth, efficiency {100 * ideal / p.cycles:.1f}%")
    ar = st.get("axi_reads", [])
    if ar:
        print("AXI reads per channel (transactions, beats, beats/transaction): " +
              ", ".join(f"{n}, {b}, {b / max(n, 1):.2f}" for n, b in ar) +
              f"; arc={a.arc} bl={a.bl} dram={a.dram}")
    b = p.buckets[0]
    if b.get("ms") is not None:
        mxb = sum(b.get("mx", []))
        print(f"MXU starved (chunk FIFO empty while streaming, the card's MXU_STARVE) "
              f"{sum(b['ms'])} cycles = {100 * sum(b['ms']) / p.cycles:.1f}%; blocked (chunks "
              f"but no MAC) {100 * sum(b.get('mb', [])) / p.cycles:.1f}%; MAC "
              f"{100 * mxb / p.cycles:.1f}%; DRAM read beats/cycle/channel "
              f"{sum(n for _, n in ar) / 2 / p.cycles:.3f}")
    for c, d in enumerate(st.get("axi_detail", [])):
        print(f"  ch{c}: port A reads {d['ar_a']}, DDR3 row opens {d['row_miss']}, "
              f"read-modify-writes {d['rmw']} (port A / QST {d['rmw_a']})")
    # useful-bytes roofline: weights + their fp32 block scales + KV + activations, at D bytes
    # per cycle (both channels at 100%)
    D = cfg.D

    def nbytes(r):
        if r.op == I.MM:
            return r.portb * D + r.porta * 4
        if r.op in (I.LD, I.ST):
            return 4 * progs[0][r.pc].w[2]
        return r.porta if r.op == I.QST else 0

    useful = sum(nbytes(r) for r in p.recs)
    ub = useful / D * 100 / a.bw
    print(f"useful bytes {useful} -> {ub:.0f} cycles at this bandwidth: "
          f"efficiency {100 * ub / p.cycles:.1f}%")
    print(f"at an assumed 100 MHz: {p.cycles / 1e5:.1f} ms/token, {1e8 / p.cycles:.1f} tok/s "
          f"(simulated cycles, no host time)")
    print(p.summary())
    print(f"phases: cycles, share, useful bytes -> cycles at {a.bw}% (their roofline), "
          f"VPU busy, MXU busy")
    pb = defaultdict(int)
    for r in p.recs:
        pb[_phase(progs[0][r.pc])] += nbytes(r)
    for k, v in sorted(p.phases(lambda r: _phase(progs[0][r.pc])).items(),
                       key=lambda x: -x[1]["cycles"]):
        rl_ph = pb[k] / D * 100 / a.bw
        print(f"  {k:12s} {v['cycles']:9d} {100 * v['cycles'] / p.cycles:5.1f}%  "
              f"{pb[k]:10d} B -> {rl_ph:9.0f} ({100 * rl_ph / max(1, v['cycles']):5.1f}%)  "
              f"VPU {v['VPU']:9d}  MXU {v['MXU']:9d}")
    if a.gaps:
        _gap_table(p, progs, pb, D * a.bw / 100)
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
