"""MXU width study: RTL cycles of one Qwen3-0.6B-sized MLP layer (H=1024, F=3072) at the board
configuration for several MXU widths (MCOLS) and activation-row counts M (batched decode rows or
prefill tokens), then a projection to full-model prefill / batched decode / TTFT.

    python3 tools/mxu_bench.py [--mcols 2,4,8,16] [--rows 1,2,4,8,16] [--impl 0|1] [--json out]

The MLP layer streams 9.4 MB of int8 weights; with M rows the compiler splits the rows into
ceil(M / MCOLS) groups and streams the weights once per group, so the measured cycles per group
("pass") is what a weight-bound layer costs. The projection scales the measured pass cost to the
full model's weight bytes (tools/perf_qwen.py: 4.84M useful-byte cycles per token at 100%) and
adds the KV-cache traffic of attention (per sequence, per position).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from opentpu import rtlsim  # noqa: E402
from opentpu.isasim import board_config  # noqa: E402
from opentpu.kernels import mlp  # noqa: E402
from opentpu.profile import profile  # noqa: E402
from test_kernels import mlp_args  # noqa: E402

H, F = 1024, 3072
# Full Qwen3-0.6B decode token at the board configuration (tools/perf_qwen.py, all 28 layers,
# pos 9, 100% bandwidth): useful-byte roofline and measured cycles.
TOKEN_ROOF, TOKEN_CYC = 4_839_968, 4_945_002
KV_BYTES_PER_POS = 28 * 2 * 8 * 128      # K and V, 8 KV heads x 128, int8, 28 layers
GQA = 2                                  # query heads per KV head


def measure(mcols: int, rows: list[int], impl: int, bw: int, lat: int) -> dict:
    uarch = dict(rtlsim.BOARD_UARCH, MXU_IMPL=impl)
    out = {}
    for m in rows:
        args, _ = mlp_args(np.random.default_rng(m), M=m, H=H, Fd=F)
        # 16 rows of fp32 temporaries do not fit the 64K-word TMEM; a larger TMEM does not
        # change the MXU's timing, so the 16-row runs get one
        cfg = board_config(MCOLS=mcols, DRAM_BYTES=1 << 25,
                           TMEM_WORDS=(1 << 17) if m >= 16 else (1 << 16))
        t = time.time()
        p = profile(mlp, cfg, uarch=uarch, run_kw=dict(axi=True, bw=bw, lat=lat, stall=0), **args)
        passes = math.ceil(m / mcols)
        out[m] = {"cycles": p.cycles, "passes": passes, "roofline": p.roofline()["bound"],
                  "per_pass": p.cycles / passes, "wall": round(time.time() - t, 1)}
        print(f"MCOLS={mcols:2d} impl={impl} M={m:3d}: {p.cycles:8d} cycles, {passes} passes, "
              f"{p.cycles / passes:9.0f}/pass, roofline {p.roofline()['bound']}", flush=True)
    return out


def project(mcols: int, meas: dict, mhz: float = 100.0, bw: float = 1.0) -> dict:
    """Full-model projections from the MLP measurements (bw: DRAM efficiency factor).

    A pass with r rows (r <= MCOLS) costs the measured full-model decode token scaled by the
    measured MLP-layer ratio pass(r) / pass(1): that captures the extra vector/quantizer work of
    more rows (at 16 rows the 2 composite VPU lanes' exp2/recip begin to dominate)."""
    ms = sorted(meas)
    one = meas[ms[0]]["per_pass"]

    def pass_cyc(r: int) -> float:
        r = min(r, mcols)
        k = min((m for m in ms if m >= r), default=ms[-1])   # nearest measured row count
        return TOKEN_CYC / bw * meas[k]["per_pass"] / one

    def step(rows: int, pos: int) -> float:                # rows share the weights, not the KV
        full, rest = divmod(rows, mcols)
        cyc = full * pass_cyc(mcols) + (pass_cyc(rest) if rest else 0)
        return cyc + rows * pos * KV_BYTES_PER_POS / 128 / bw

    res = {"decode_tok_s": {}}
    for b in (1, 2, 4, 8):
        res["decode_tok_s"][b] = b * mhz * 1e6 / step(b, 256)
    # prefill of T tokens: T / MCOLS full passes, plus causal attention whose K/V stream is
    # shared by the MCOLS / GQA query tokens of a pass
    for T in (128, 512, 2048):
        lin = math.ceil(T / mcols) * pass_cyc(mcols)
        att = (T * T / 2) * KV_BYTES_PER_POS / 128 / max(1, mcols // GQA) / bw
        cyc = lin + att
        res.setdefault("prefill_tok_s", {})[T] = T * mhz * 1e6 / cyc
        res.setdefault("ttft_s", {})[T] = cyc / (mhz * 1e6)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcols", default="2,4,8,16")
    ap.add_argument("--rows", default="1,2,4,8,16")
    ap.add_argument("--impl", type=int, default=0)
    ap.add_argument("--bw", type=int, default=100)
    ap.add_argument("--lat", type=int, default=30)
    ap.add_argument("--json")
    ap.add_argument("--from-json", help="re-project saved measurements")
    a = ap.parse_args()
    rows = [int(x) for x in a.rows.split(",")]
    allm = {}
    if a.from_json:
        saved = json.loads(Path(a.from_json).read_text())["measure"]
        allm = {int(mc): {int(m): v for m, v in d.items()} for mc, d in saved.items()}
    else:
        for mc in (int(x) for x in a.mcols.split(",")):
            allm[mc] = measure(mc, rows, a.impl, a.bw, a.lat)
    print("\nprojection at 100 MHz (bw = measured bandwidth; x0.8 for realistic DDR3-800):")
    table = {}
    for mc, meas in allm.items():
        for eff in (1.0, 0.8):
            pr = project(mc, meas, bw=eff)
            table[f"{mc}@{eff}"] = pr
            d = pr["decode_tok_s"]
            print(f"MCOLS={mc:2d} DDR x{eff}: decode b1 {d[1]:5.1f} b2 {d[2]:5.1f} b4 {d[4]:5.1f} "
                  f"b8 {d[8]:5.1f} tok/s | prefill@512 {pr['prefill_tok_s'][512]:6.1f} tok/s | "
                  f"TTFT 128/512/2048 {pr['ttft_s'][128]:.2f}/{pr['ttft_s'][512]:.2f}/"
                  f"{pr['ttft_s'][2048]:.1f} s")
    if a.json:
        Path(a.json).write_text(json.dumps({"measure": allm, "projection": table}, indent=1))


if __name__ == "__main__":
    main()
