"""Prefill cycles per token against the rows a weight pass serves (docs/tiled_prefill.md): RTL
co-simulation of one prefill run of T rows, the first L layers of a model, no logits.

    python3 tools/perf_tiled.py [--model qwen3|qwen35|DIR | --shape qwen3-4b] [--layers 2]
        [--pos 256] [--rows 4,8,16] [--mcols 4] [--act-rows 32] [--tmem 65536]
        [--imem 131072] [--wformat fp4] [--mem ldc|port] [--lat 40] [--kernel rows|tile]
        [--phases] [--compile-only]

--kernel rows: today's prefill run (qwen3_rows / qwen35_rows, compile_rows) with every MM over
the run's T rows at once: the ACT RAM holds max(--act-rows, MCOLS, T) rows, so each weight
chunk is streamed once and replayed for ceil(T / MCOLS) groups (docs/isa.md, MM). TMEM and IMEM
may be virtual (--tmem, --imem words; the card's TMEM is 64K words): today's kernel holds every
row's activations at once, so more than 8 rows of Qwen3-0.6B need more. A TMEM over 64K words
needs otpu_seq's TMEM hazard ranges to follow TMEM_WORDS (without it every instruction past
word 65,535 waits for everything: docs/tiled_prefill.md 3.5).
--kernel tile: the tiled prefill run (opentpu/llm/tiled.py): the same, with the projections
split by KV head and the MLP's down projection by output columns (TMEM independent of H), and
the attention's rows packed MCOLS / G to an MM (K / V streamed once per group of rows).

--mem ldc: the card's memory path (LiteDRAM's controllers co-simulated, DDR3-1066, the core at
133.33 MHz); --mem port: the board's adapter in front of a memory at the core port's full rate
(one 128-byte chunk a cycle on two channels, --lat cycles of latency): the BCU1525's DDR4, whose
bandwidth exceeds what one slice's port takes.

The images are zeros (timing does not depend on the data). One JSON line per run: cycles,
cycles per row, the core-port roofline (port-B chunks, the MXU's own cycles), and with
--phases the cycles per phase (projections, attention, MLP, DeltaNet, other) with their port-B
chunks and MXU / VPU busy cycles.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu import isa as I  # noqa: E402
from opentpu import rtlsim  # noqa: E402
from opentpu.isasim import board_config  # noqa: E402
from opentpu.llm import load_spec, model_dir  # noqa: E402
from opentpu.profile import ddr3_plusargs, parse  # noqa: E402

MHZ = 133.33
SHAPES = {   # dense shapes whose checkpoints are not needed (timing does not depend on data)
    "qwen3-4b": dict(hidden=2560, layers=36, n_q=32, n_kv=8, head_dim=128, ffn=9728,
                     vocab=151936),
}
ATTN_CORE = {"scores", "finish", "prime", "_attend_heads", "_attend_tile", "_tile_scores",
             "_tile_finish"}
PHASES = {"_deltanet_rows": "deltanet", "_deltanet_dstep": "deltanet",
          "_mlp": "mlp", "swiglu_down": "mlp", "_mlp_tile": "mlp", "_down_split": "mlp",
          "_attention_rows": "attn-proj", "_attention_tile": "attn-proj"}


def phase(ins) -> str:
    fns = [fn for _, _, fn in ins.src]
    if any(f in ATTN_CORE for f in fns):
        return "attn-core"
    for f in fns:
        if f in PHASES:
            return PHASES[f]
    return "other"


def spec_of(a):
    if a.shape:
        from opentpu.llm.qwen3 import Spec
        return Spec(**SHAPES[a.shape])
    return load_spec(model_dir(a.model))


def cut(spec, n):
    if hasattr(spec, "kinds"):
        return dataclasses.replace(spec, kinds=spec.kinds[:n])
    return dataclasses.replace(spec, layers=n)


def compile_run(img, kernel, pos, T, block=256):
    rows = [(0, pos + r) for r in range(T)]
    if kernel == "rows":
        return img.compile_rows(rows, [], block)
    from opentpu.llm import tiled
    return tiled.compile_tile(img, pos, T, block)


def one(a, spec, T):
    act = max(a.act_rows or 0, a.mcols, T)
    kw = dict(MCOLS=a.mcols, ACT_ROWS=act, TMEM_WORDS=a.tmem, IMEM_WORDS=a.imem, PAIR=True,
              DSTEP=True, STREAM=True)
    cap = 256 * ((a.pos + T - 1) // 256 + 1)
    rec = dict(model=a.shape or Path(str(model_dir(a.model))).name, layers=a.layers,
               pos=a.pos, T=T, mcols=a.mcols, act_rows=act, tmem=a.tmem, wformat=a.wformat,
               mem=a.mem, kernel=a.kernel)
    probe = spec.image(board_config(DRAM_BYTES=1 << 40, **kw), cap, rows=T,
                       wformat=a.wformat, head_format="int8")
    cfg = board_config(DRAM_BYTES=1 << max(20, (probe.nbytes - 1).bit_length()), **kw)
    img = spec.image(cfg, cap, rows=T, wformat=a.wformat, head_format="int8")
    t = time.time()
    try:
        progs = compile_run(img, a.kernel, a.pos, T)
    except Exception as e:                  # TMEM / IMEM / ACT RAM: the fit answer
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        return rec
    rec["instructions"] = max(len(p) for p in progs)
    mm = [i for i in progs[0] if i.op == I.MM]
    rec["mms"] = len(mm)
    rec["compile_s"] = round(time.time() - t, 1)
    if a.compile_only:
        return rec
    dram = np.zeros(img.nbytes, np.uint8)
    if a.mem == "ldc":
        plus = rtlsim.ldc_plusargs(1066, MHZ) + ddr3_plusargs(3200 / 3, MHZ)[2:]
        mem = dict(ldc=1066, bw=100, lat=round(0.3 * MHZ),
                   plusargs=["+axi_dram=1", "+axi_map=1"] + plus)
    else:
        mem = dict(ldc=0, bw=100, lat=a.lat, plusargs=[])
    t = time.time()
    _, _, st = rtlsim.run(cfg, progs, [dram], trace=True, uarch=rtlsim.BOARD_UARCH, axi=True,
                          boot=True, stall=0, max_cycles=1 << 40, **mem)
    rec["sim_s"] = round(time.time() - t)
    p = parse(st["trace"], cfg, progs, "tiled")
    p.cycles = st["cycles"]
    ps = p.roofline()["per_slice"][0]
    rec.update(cycles=p.cycles, per_row=round(p.cycles / T), portb=ps["portb"],
               porta=ps["porta_cycles"], mxu=ps["mxu"],
               busy={k: v for k, v in p.unit_busy(0).items() if v})
    if a.phases:
        ph = p.phases(lambda r: phase(progs[0][r.pc]))
        rec["phases"] = {k: dict(cycles=v["cycles"], portb=v["portb"], VPU=v["VPU"],
                                 MXU=v["MXU"]) for k, v in ph.items()}
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--shape", choices=sorted(SHAPES), default=None)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pos", type=int, default=256)
    ap.add_argument("--rows", default="4,8,16")
    ap.add_argument("--mcols", type=int, default=4)
    ap.add_argument("--act-rows", type=int, default=0,
                    help="ACT RAM rows (at least max(MCOLS, T), the default)")
    ap.add_argument("--tmem", type=int, default=1 << 16, help="TMEM words (the card's: 64K)")
    ap.add_argument("--imem", type=int, default=1 << 17)
    ap.add_argument("--wformat", default="fp4", choices=["int8", "int4", "fp4"])
    ap.add_argument("--mem", choices=["ldc", "port"], default="ldc")
    ap.add_argument("--lat", type=int, default=40, help="--mem port: latency, core cycles")
    ap.add_argument("--kernel", choices=["rows", "tile"], default="rows")
    ap.add_argument("--phases", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    a = ap.parse_args()
    spec = cut(spec_of(a), a.layers)
    for T in [int(x) for x in a.rows.split(",")]:
        print(json.dumps(one(a, spec, T)), flush=True)


if __name__ == "__main__":
    main()
