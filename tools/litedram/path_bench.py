"""Cycles of a few board workloads on the two memory paths in simulation: the AXI adapter
(otpu_axi_dram) and the native one (otpu_native_dram, OTPU_NATIVE), on the same DDR3-1066 bank
model at the core clock of 120.755 MHz and a 300 ns read latency (opentpu.profile.ddr3_plusargs).
The AXI path runs twice: with the per-transaction costs fitted on the card (arc, tgap, bgap,
wgap) and without them; the native model has none (docs/litedram.md section 3).

    python3 tools/litedram/path_bench.py [micro] [rw] [qwen]      (default: all three)

micro: a 1 MiB LD and ST, a transposed V append (a byte per beat: the SW queue's read-fill) and a
contiguous K append (QSTs); rw: tools/rw_bench.py sim --mb 2; qwen: tools/perf_qwen.py, 2 layers
at position 300 (needs the Qwen3-0.6B weights). Run it on the test machine, not the Mac.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from opentpu import Config, isa as I, rtlsim  # noqa: E402
from opentpu.isasim import Machine  # noqa: E402
from opentpu.profile import ddr3_plusargs  # noqa: E402

MHZ = 120.755
DDR = ddr3_plusargs(3200 / 3, MHZ)
NOCOST = ["+axi_arc=0", "+axi_tgap=0", "+axi_bgap=0", "+axi_wgap=0"]
LAT = round(0.3 * MHZ)
MODES = [("axi", False, []), ("axi-nocost", False, NOCOST), ("native", True, [])]


def micro():
    cfg = Config(S=1, D=128, ACT_BLOCKS=16, DRAM_BYTES=1 << 22)
    D = cfg.D
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, 1 << 22, dtype=np.uint8)
    img[:4 * 8 * D] = rng.standard_normal(8 * D).astype(np.float32).view(np.uint8)
    CAP, H = 256, 8
    progs = {
        "ld 1 MiB": [I.ld(k << 18, 0, 1 << 16) for k in range(4)] + [I.halt()],
        "st 1 MiB": [I.st((k + 4) << 18, 0, 1 << 16) for k in range(4)] + [I.halt()],
        "vt append x4": [I.ld(0, 0, 4 * H * D)] +
                        [I.qst(0, 0x20000 + t, 0xE0000 + 64 * t, H, 1, D, D * CAP, CAP)
                         for t in range(4)] + [I.halt()],
        "k append x16": [I.ld(0, 0, 4 * H * D)] +
                        [I.qst(0, 0x200000 + t * H * D, 0x300000 + 32 * t, H, 1, D, D, 1)
                         for t in range(16)] + [I.halt()],
    }
    for name, prog in progs.items():
        m = Machine(cfg, [prog], [img.copy()]).run()
        row = []
        for mode, native, extra in MODES:
            d, _, st = rtlsim.run(cfg, [prog], [img.copy()], axi=True, boot=True, native=native,
                                  stall=0, lat=LAT, arc=4, uarch=rtlsim.BOARD_UARCH,
                                  plusargs=extra + DDR, max_cycles=1 << 40)
            ok = np.array_equal(d[0], m.slices[0].dram)
            row.append(f"{mode} {st['cycles']}{'' if ok else ' MISMATCH'}")
        print(f"{name:14s} " + "  ".join(row), flush=True)


def _tool(args, native, extra):
    env = {**os.environ, "OTPU_NATIVE": "1" if native else "0"}
    return subprocess.run([sys.executable, *args] + [f"--plus={p}" for p in extra], env=env,
                          capture_output=True, text=True, cwd=ROOT)


def rw():
    for mode, native, extra in MODES:
        r = _tool(["tools/rw_bench.py", "sim", "--mb", "2",
                   "--modes", "mm,mm+st,mm+ld,mm+dstep,st,ld"], native, extra)
        print(f"== rw_bench {mode}\n{r.stdout}{r.stderr[-2000:] if r.returncode else ''}",
              flush=True)


def qwen():
    for mode, native, extra in MODES:
        t = time.time()
        r = _tool(["tools/perf_qwen.py", "--layers", "2", "--pos", "300", "--ddr", "1066",
                   "--mhz", str(MHZ)], native, extra)
        keep = [x for x in r.stdout.splitlines()
                if re.match(r"(layers=|MXU starved|DRAM efficiency|at \d|  ch\d)", x)]
        print(f"== perf_qwen {mode} ({time.time() - t:.0f}s)\n" + "\n".join(keep) +
              (r.stderr[-2000:] if r.returncode else ""), flush=True)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["micro", "rw", "qwen"]:
        {"micro": micro, "rw": rw, "qwen": qwen}[w]()
