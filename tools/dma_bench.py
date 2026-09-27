"""Host <-> card DMA latency and bandwidth through the XDMA driver, for comparing the driver's
completion modes (poll_mode=1 vs interrupts; docs/host.md section 2).

    python3 tools/dma_bench.py [--dev /dev/xdma0] [--json out.json]

Takes the device lock. Uses channel 0 from 1 GiB up (writes before it reads: the card's ECC
DRAM hangs on a read of a never-written beat) and checks every read against what was written.
Reports per-call latency (median, p99) for small transfers, GB/s for large ones, and the
CPU time per transfer of the process plus the driver's completion threads (poll mode waits
there).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opentpu.host.board import XdmaTransport  # noqa: E402
from opentpu.host.runstate import DeviceLock  # noqa: E402

BASE = 1 << 30                                 # channel-0 offset of the test region


def cpu() -> float:
    """CPU seconds of this process plus the XDMA driver's kernel threads (cmpl_status_th*, where
    poll mode waits for completions), so other load on the PC does not count."""
    tot = sum(os.times()[:2])
    hz = os.sysconf("SC_CLK_TCK")
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            st = open(f"/proc/{d}/stat").read()
        except OSError:
            continue
        name = st[st.index("(") + 1:st.rindex(")")]
        if name.startswith(("cmpl_status_th", "xdma")):
            f = st[st.rindex(")") + 2:].split()
            tot += (int(f[11]) + int(f[12])) / hz                 # utime, stime
    return tot


def timed(fn, reps: int) -> tuple[np.ndarray, float]:
    """Per-call wall times (s) and the CPU seconds (cpu()) of reps calls."""
    t = np.empty(reps)
    c0 = cpu()
    for i in range(reps):
        a = time.perf_counter()
        fn()
        t[i] = time.perf_counter() - a
    return t, cpu() - c0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--json")
    a = ap.parse_args()
    lock = DeviceLock(Path(a.dev).name)
    t = XdmaTransport(a.dev)
    mode = Path("/sys/module/xdma/parameters/poll_mode").read_text().strip()
    rng = np.random.default_rng(1)
    out = {"poll_mode": int(mode), "latency": [], "bandwidth": []}
    print(f"poll_mode={mode}")

    reg = timed(lambda: t.reg_read(0), 10000)[0]
    out["reg_read_us"] = {"median": float(np.median(reg) * 1e6), "p99": float(np.percentile(reg, 99) * 1e6)}
    print(f"register read          median {out['reg_read_us']['median']:7.2f} us  "
          f"p99 {out['reg_read_us']['p99']:7.2f} us")

    for n, reps in ((64, 2000), (4096, 2000), (65536, 1000)):
        buf = rng.integers(0, 256, n).astype(np.uint8)
        wt, wc = timed(lambda: t.mem_write(0, BASE, buf), reps)
        got = np.empty(n, np.uint8)
        rt, rc = timed(lambda: t.mem_read(0, BASE, n, got), reps)
        if not np.array_equal(got, buf):
            raise SystemExit(f"{n} B: read back differs")
        row = {"bytes": n, "reps": reps,
               "write_us": [float(np.median(wt) * 1e6), float(np.percentile(wt, 99) * 1e6)],
               "read_us": [float(np.median(rt) * 1e6), float(np.percentile(rt, 99) * 1e6)],
               "write_cpu_us": wc / reps * 1e6, "read_cpu_us": rc / reps * 1e6}
        out["latency"].append(row)
        print(f"{n:>6} B  write median {row['write_us'][0]:7.1f} us p99 {row['write_us'][1]:7.1f}"
              f" cpu {row['write_cpu_us']:6.1f} | read median {row['read_us'][0]:7.1f} us"
              f" p99 {row['read_us'][1]:7.1f} cpu {row['read_cpu_us']:6.1f}")

    for n, reps in ((1 << 20, 64), (8 << 20, 16), (64 << 20, 4)):
        buf = rng.integers(0, 256, n).astype(np.uint8)
        wt, wc = timed(lambda: t.mem_write(0, BASE, buf), reps)
        got = np.empty(n, np.uint8)
        rt, rc = timed(lambda: t.mem_read(0, BASE, n, got), reps)
        if not np.array_equal(got, buf):
            raise SystemExit(f"{n} B: read back differs")
        row = {"bytes": n, "reps": reps,
               "write_gbs": n * reps / wt.sum() / 1e9, "read_gbs": n * reps / rt.sum() / 1e9,
               "write_cpu_frac": wc / wt.sum(), "read_cpu_frac": rc / rt.sum()}
        out["bandwidth"].append(row)
        print(f"{n >> 20:>4} MiB  write {row['write_gbs']:5.2f} GB/s (cpu {row['write_cpu_frac']:4.2f} cores)"
              f" | read {row['read_gbs']:5.2f} GB/s (cpu {row['read_cpu_frac']:4.2f} cores)")

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    lock.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
