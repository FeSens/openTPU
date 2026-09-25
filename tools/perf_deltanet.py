"""Measure the Gated DeltaNet recurrence on the RTL at the board configuration (AXI memory path).

    python3 tools/perf_deltanet.py [--heads 1 8 16] [--bw 80] [--cl 2]

Runs `kernels/deltanet.gated_deltanet_step` (128 x 128 fp32 states streamed through TMEM by LD
and ST) with the new VOPs (RDOT, OUTER: 3 state passes per head) and on the older ISA (7
passes), checks the RTL against the ISA simulator bit for bit, and prints the cycles. The
old-ISA kernel does not compile for 16 heads (its 16K-word temporaries and the per-head tiles
fragment the 64K-word TMEM): use the per-head slope between 1 and 8 heads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from opentpu import isa as I, rtlsim  # noqa: E402
from opentpu.isasim import Machine, board_config  # noqa: E402
from opentpu.compiler import CompileError  # noqa: E402
from opentpu.kernels.deltanet import gated_deltanet_step  # noqa: E402
from opentpu.runtime import collect, compile_kernel  # noqa: E402
from test_vops import deltanet_args  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, nargs="+", default=[1, 8, 16])
    ap.add_argument("--bw", type=int, default=80)
    ap.add_argument("--lat", type=int, default=30)
    ap.add_argument("--cl", type=int, default=2, help="VPU composite lanes (VPU_CL)")
    a = ap.parse_args()
    cfg = board_config(DRAM_BYTES=1 << 24)
    uarch = dict(rtlsim.BOARD_UARCH, VPU_CL=a.cl)
    for H in a.heads:
        args, (Sw, ow) = deltanet_args(np.random.default_rng(H), H=H)
        for fused in (True, False):
            try:
                comp, imgs = compile_kernel(gated_deltanet_step, cfg, **args, fused=fused)
            except CompileError as e:
                print(f"H={H:2d} {'old ISA':22s} does not compile: {e}")
                continue
            prog = comp.programs[0]
            passes = sum(1 for i in prog if i.op == I.VOP and i.w[3] == 128 | (128 << 16))
            m = Machine(cfg, comp.programs, [imgs[0].copy()]).run()
            drams, tmems, st = rtlsim.run(cfg, comp.programs, [imgs[0].copy()], uarch=uarch,
                                          axi=True, boot=True, bw=a.bw, lat=a.lat, stall=0)
            exact = np.array_equal(drams[0], m.slices[0].dram) and \
                np.array_equal(tmems[0], m.slices[0].tmem)
            out = collect(comp, drams, tmems, st).outputs
            err = np.max(np.abs(out["state_out"].transpose(0, 2, 1) - Sw)) / np.abs(Sw).max()
            print(f"H={H:2d} {'new ISA (RDOT, OUTER)' if fused else 'old ISA':22s} "
                  f"CL={a.cl} bw={a.bw}%: {st['cycles']:8d} cycles  {len(prog):4d} instrs  "
                  f"{passes:3d} state passes  bit-exact vs ISA sim: {exact}  "
                  f"state rel err vs float64 {err:.1e}", flush=True)


if __name__ == "__main__":
    main()
