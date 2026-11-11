"""Two physical board paths in one Verilator process, joined by otpu_link_sim at L=40.

Reuse board_perf.decode_case's configuration-only cuts, zero weights, run-time position and
present_words expert directory (the decode cuts' flow), with tb_x2 in place of tb_bcuwg's
cross-card approximation. Check every byte of the image's low area, all instruction counts,
logits / argmax and issued collective counts against an S=4 ISA machine. No model tensors
are loaded. One run at a time; compilation uses four workers on split C++ units,
nice 15 externally.

Example (from this tree, with the host's Python environment and PYTHONPATH=.):
  python tools/x2_cosim.py q36 CUT_DIR/q36-4:fp4:int8:r60 --json q36.json

The cut's measured body is repeated 10 / 16 / 12 times for q36 / q38 / goss. The prologue
occurs once. Qwen heads use the study's half-of-S=1 estimate (full-head co-sim is separate
work); GPT-OSS's cut already contains the full head. These are full-model projections,
not full-model runs or card measurements. Useful bandwidth is useful bytes / (1024 * cycles).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np


def phase_blocks(programs) -> dict:
    """The gap map's source-function blocks, on slice 0.
    Source frames come from this compile; no PC ranges from another tree are reused."""
    groups = {
        "DN": {"_deltanet_dstep", "gates", "conv", "_silu_dn", "_l2n", "_key_heads",
               "_pair_segment", "gatez", "post", "_emit", "_dn_ahead", "_cv"},
        "ATT": {"_attention", "_norm_heads", "_rope_padded", "prime", "_attend_heads",
                "_attend_split"},
        "MLP": {"_mlp"}, "MoE": {"moe_ffn", "route", "expert", "glu"},
        "HEAD": {"_lm_head"},
    }
    block, cur = {}, "prologue"
    for pc, ins in enumerate(programs[0]):
        src = ins.src[1] if len(ins.src) > 1 else (ins.src[0] if ins.src else ("?", 0, "?"))
        cur = next((b for b, fs in groups.items() if src[2] in fs), cur)
        block[pc] = cur
    return block


def phase_spans(profile, programs=None, *, blocks=None) -> dict:
    """Latest-end accounting, with source blocks from this compile or a prepared pack."""
    block = phase_blocks(programs) if blocks is None else blocks
    span, done = collections.Counter(), 0
    for rec in sorted(profile.slice_recs(0), key=lambda r: r.idx):
        if rec.end > done:
            span[block[rec.pc]] += rec.end - done
            done = rec.end
    return dict(span)


def project(model: str, cycles: int, spans: dict) -> dict:
    repeats, useful = {"q36": (10, 1957e6), "q38": (16, 14596e6),
                       "goss": (12, 2510208000)}[model]
    pro, head = spans.get("prologue", 0), spans.get("HEAD", 0)
    if not head:
        raise ValueError("head missing from trace: cannot project a full token")
    body = cycles - pro - head
    # pfsum.py's q36 anchor: 4.97M = 10 * (414135 - 17145 - 14618) + 17145 + head1.
    full_head = {"q36": (4970000 - 10 * (414135 - 17145 - 14618) - 17145) / 2,
                 "q38": 2720000 / 2, "goss": head}[model]
    full = repeats * body + pro + full_head
    return {"repeats": repeats, "body_cycles": body, "prologue_cycles": pro,
            "cut_head_cycles": head, "full_head_cycles": full_head,
            "head_estimated": model != "goss", "full_cycles": full,
            "full_tok_s_150mhz": 150e6 / full, "useful_bytes": int(useful),
            "useful_bandwidth_fraction": useful / (512 * 2 * full)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", choices=["q36", "q38", "goss"])
    ap.add_argument("point", help="board_perf decode point (configuration-only cut)")
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--target", default="bcu1525-x2")
    ap.add_argument("--trace-depth", type=int, default=32768,
                    help="trace records (default 32768; old cuts used at most 28594)")
    a = ap.parse_args()
    os.environ["OTPU_TARGET"] = a.target
    from opentpu import rtlsim, target as T
    from opentpu.host.board import Board
    from opentpu.host.cards import CardsTransport, X2SimTransport
    from opentpu.host.hwlens import drop_orphans, records_to_trace
    from opentpu.isasim import Machine
    from opentpu.profile import parse
    from board_perf import decode_case, present_words
    from importlib.util import module_from_spec, spec_from_file_location

    flags = list(rtlsim.VFLAGS)
    flags[flags.index("-j") + 1] = "4"
    flags += ["--output-groups", "0"]
    rtlsim.VFLAGS = flags
    # The real card's two-slice plan c, used by gx.sh --params FP=c, on each card.
    spec = spec_from_file_location("bcu_fp", "boards/bcu1525/scripts/bcu_fp.py")
    fp = module_from_spec(spec)
    spec.loader.exec_module(fp)
    plan = fp.plan("c")
    cfg, img, progs, at, args = decode_case(T.config(T.load(a.target)), a.point)
    low = img.low_end
    # A GPT-OSS cut's low area includes a multi-GiB zero embedding table. Keep only a small
    # dense prefix; the program and the few touched beats beyond it use the existing sparse
    # channel store. All three cuts then share one Verilator build and stay below 20 GiB.
    dense = 1 << 21
    params = {"SE_R": 2, "DMA_DEPTH": 512, "RLY": plan["rly"], "DMAP": plan["dmap"],
              "TRACE_DEPTH": a.trace_depth, "TRACE_QD": 256}
    plus = ["+cp=3333", "+dimm_mc=1", "+mc_ba=15", "+mc_rw=62", "+mc_strict=1",
            "+mc_xbsw=16", "+mc_xrtw=8", "+mc_xwtr=8", "+mc_xgq=12",
            "+max_cycles=100000000", "+x2_debug"]
    print(json.dumps({"point": a.point, "low_bytes": low, "program_at": at,
          "image_bytes": img.nbytes, "dense_bytes_per_channel": dense,
          "params": params, "plusargs": plus, "link_lat": 40}), flush=True)
    # Sparse zero DRAM avoids materializing the cut's multi-GiB, unwritten weight arrays.
    ref = Machine(cfg, progs, [None] * 4, args)
    for s in ref.slices:
        for addr, data in present_words(img):
            s.dram[addr:addr + len(data)] = data
    print("ISA start", flush=True)
    ref.run(max_steps=1 << 30)
    counts, issued = [s.icount for s in ref.slices], ref.collectives
    print(f"ISA done: instructions={counts}, issued={issued}", flush=True)
    sim = X2SimTransport(target=a.target, ch_bytes=dense, params=params, plusargs=plus, link_lat=40)
    ct = CardsTransport(sim.cards)
    board = Board(ct, lock=False)
    for k in range(4):
        for addr, data in present_words(img):
            board.write(addr, data, k)
    board.load(at, progs)
    ct.collectives = issued               # includes run-time LOOP counts known from the ISA
    if args is not None:
        board.set_args(args)
    st = board.run(trace={"keep": "first"})
    assert st["instructions"] == counts, (st["instructions"], counts)
    checked = 1 << 22
    for k, s in enumerate(ref.slices):
        for off in range(0, low, checked):
            end = min(off + checked, low)
            got = board.read(off, end - off, k)
            diff = np.flatnonzero(got != s.dram[off:end])
            assert not diff.size, (k, off + int(diff[0]), len(diff))
    io = img.io["logits"]
    vloc = img.v_loc
    logits = np.concatenate([board.read(io + 4 * k * vloc, 4 * vloc, k).view("<u4")
                             for k in range(4)])
    want_logits = np.concatenate([s.dram[io + 4 * k * vloc:io + 4 * (k + 1) * vloc]
                                  .view("<u4") for k, s in enumerate(ref.slices)])
    assert np.array_equal(logits, want_logits)
    token = int(np.argmax(logits.view("<f4")))
    assert token == int(np.argmax(want_logits.view("<f4")))
    trace = st["trace"]
    assert not trace["lost"] and not trace["drop"], trace
    profile = parse(drop_orphans(records_to_trace(trace["records"], progs)), cfg, progs, "card")
    spans = phase_spans(profile, progs)
    baselines = {"q36": (191851, 367844), "q38": (936484, 1839124),
                 "goss": (807765, 1580314)}
    approx, one = baselines[a.model]
    out = {"model": a.model, "point": a.point, "target": a.target, "ok": True, "link_lat": 40,
           "cycles": st["cycles"], "instructions": counts, "collectives_mod65536": issued,
           "checked_low_bytes_per_slice": low, "logit_words_checked": len(logits),
           "argmax": token, "sequences": ct.sequences, "collective_delta": ct.collective_delta,
           "params": params, "plusargs": plus, "trace_count": trace["count"],
           "board_projection_cycles": approx, "delta_projection_percent":
           100 * (st["cycles"] / approx - 1), "one_card_cycles": one,
           "one_card_speedup": one / st["cycles"], "phase_cycles": spans,
           "projection": project(a.model, st["cycles"], spans)}
    a.json.write_text(json.dumps(out, indent=2) + "\n")
    print("X2_CUT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
