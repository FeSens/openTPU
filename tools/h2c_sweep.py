"""Host -> card DMA (H2C) rate sweep: what holds XDMA's host->card writes at ~2.3 GB/s on PCIe Gen2
x8 (docs/host.md section 2). For each point (call size, DRAM channel, calls in flight, the host
buffer's pages) it makes back-to-back writes on h2c_0 for a while and records, per call, the wall
time and the calling thread's CPU time (CLOCK_THREAD_CPUTIME_ID: the system call's own work, i.e.
the driver's pinning, mapping, descriptors and unpinning; in interrupt mode the wait for the
engine sleeps). So for one call in flight, n / (wall - CPU) is the engine's rate with the host's
serial share taken out, and the rate with 2 or 4 calls in flight (threads; the driver pins a call's
pages before it takes the engine) is what overlapping that share gives.

    python3 tools/h2c_sweep.py [--dev /dev/xdma0] [--json out.json] [--secs 0.3] [--reps 2]
                               [--sizes 65536,...] [--qd 1,2,4] [--quick]

The points:
- sizes: 64 KiB to 8 MiB, with offload's record halves (35B 0.83 MB, 26B 1.72 MB);
- channel: 0, 1, or alternating (call k on channel k % 2: offload's two halves of a record);
- qd: calls in flight, one thread each, each with its own buffer and card region;
- pages: the host buffer private on transparent huge pages (thp) or on 4 KiB pages (4k), or shared
  anonymous memory (shm: board.placed's and the staging buffer's, shmem on 4 KiB pages);
- the bounced path (XdmaTransport.mem_write from a buffer 16 bytes off: the staging copy, as
  dma_bench's and the selftest's writes may take) and card->host reads, one call at a time.
Every buffer is placed (board.DMA_PLACE) and touched before its point; the order of the points is
shuffled (fixed seed) and each is run --reps times.

Safety: takes the device lock and holds the card's DMA lock (board._DmaLock) throughout; no run
may be in flight (H2C alone: docs/host.md, "XDMA's H2C overrun"); a card->host read never overlaps
a write. After each point the first and last 4 KiB of every thread's last call are read back and
compared; a mismatch stops the sweep (FAIL). The ECC counters are read before and after.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opentpu.host import board as B  # noqa: E402

REGION = 1 << 30                        # each channel's test region starts here (as dma_bench's)
SPAN = 64 << 20                         # card bytes per thread
HUGE = 2 << 20
MADV_HUGEPAGE = getattr(mmap, "MADV_HUGEPAGE", 14)
MADV_NOHUGEPAGE = getattr(mmap, "MADV_NOHUGEPAGE", 15)
PAGES = ("thp", "4k", "shm")
SIZES = [64 << 10, 256 << 10, 834_944, 1 << 20, 1_724_992, 4 << 20, 8 << 20]
QUICK = [256 << 10, 834_944, 8 << 20]


def card_addr(ch: int, thread: int) -> int:
    return B.BASE[ch] + REGION + thread * SPAN


HUGE_KB: dict[int, int] = {}            # a buffer's address -> its mapping's AnonHugePages (kB)


def alloc(n: int, card: int, pages: str) -> np.ndarray:
    """n bytes placed for `card` (d % 4096 == DMA_PLACE) in a 2 MiB-aligned mapping, touched (random
    bytes): private on huge pages (thp) or 4 KiB pages (4k), or shared anonymous (shm, as
    board.placed's: shmem, which THP leaves alone)."""
    if pages == "shm":
        m = mmap.mmap(-1, n + 2 * HUGE + 4096)
    else:
        m = mmap.mmap(-1, n + 2 * HUGE + 4096, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    base = np.frombuffer(m, np.uint8)
    a0 = B._addr(base)
    start = (-a0) % HUGE
    if hasattr(m, "madvise") and pages != "shm":
        try:
            m.madvise(MADV_HUGEPAGE if pages == "thp" else MADV_NOHUGEPAGE)
        except OSError:
            pass
    off = start + (card + B.DMA_PLACE - (a0 + start)) % 4096
    buf = base[off:off + n]
    buf[:] = np.random.default_rng(n ^ card).integers(0, 256, n, np.uint8)
    HUGE_KB[B._addr(buf)] = huge_kb(B._addr(buf))
    return buf


def huge_kb(addr: int) -> int:
    """AnonHugePages (kB) of the mapping that holds addr, or -1 where /proc has no smaps."""
    try:
        inside = False
        for line in Path("/proc/self/smaps").read_text().splitlines():
            f = line.split()
            if f and "-" in f[0] and ":" not in f[0]:
                lo, hi = (int(x, 16) for x in f[0].split("-"))
                inside = lo <= addr < hi
            elif inside and line.startswith("AnonHugePages:"):
                return int(f[1])
    except (OSError, ValueError):
        pass
    return -1


def run_point(fd: int, bufs: list[np.ndarray], n: int, chmode: str, secs: float) -> dict:
    """Back-to-back h2c calls of n bytes from len(bufs) threads for `secs`: per-call wall and CPU
    (ns) and the aggregate rate."""
    qd = len(bufs)
    walls: list[list[int]] = [[] for _ in range(qd)]
    cpus: list[list[int]] = [[] for _ in range(qd)]
    last_ch = [0] * qd
    t_first, t_last = [0] * qd, [0] * qd
    start = threading.Barrier(qd)
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            mv = memoryview(bufs[i])
            start.wait()
            end = time.perf_counter_ns() + int(secs * 1e9)
            k = 0
            t_first[i] = time.perf_counter_ns()
            while k < 3 or time.perf_counter_ns() < end:
                ch = k % 2 if chmode == "alt" else int(chmode)
                w0 = time.perf_counter_ns()     # the wall interval holds the CPU one
                c0 = time.thread_time_ns()
                got = os.pwrite(fd, mv, card_addr(ch, i))
                c1 = time.thread_time_ns()
                w1 = time.perf_counter_ns()
                if got != n:
                    raise OSError(f"h2c write returned {got} of {n}")
                walls[i].append(w1 - w0)
                cpus[i].append(c1 - c0)
                last_ch[i] = ch
                k += 1
            t_last[i] = time.perf_counter_ns()
        except BaseException as e:     # noqa: BLE001 (reported by the caller)
            errors.append(e)

    th = [threading.Thread(target=worker, args=(i,)) for i in range(qd)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errors:
        raise errors[0]
    calls = sum(len(w) for w in walls)
    span = max(t_last) - min(t_first)
    w = np.concatenate([np.array(x) for x in walls])
    c = np.concatenate([np.array(x) for x in cpus])
    wm, cm = float(np.median(w)), float(np.median(c))
    return {"calls": calls, "gbs": calls * n / span, "wall_us": wm / 1e3, "cpu_us": cm / 1e3,
            "wall_p90_us": float(np.percentile(w, 90)) / 1e3,
            "cpu_us_per_page": cm / 1e3 / max(1, n // 4096),
            "engine_gbs": n / (wm - cm) if qd == 1 and wm > cm else None,
            "last_ch": last_ch}


def check(t, bufs: list[np.ndarray], n: int, last_ch: list[int]) -> str | None:
    """The first and last 4 KiB of each thread's last call, read back: None if equal."""
    k = min(4096, n)
    for i, b in enumerate(bufs):
        for off in sorted({0, n - k}):
            got = t.mem_read(last_ch[i], card_addr(last_ch[i], i) - B.BASE[last_ch[i]] + off, k)
            if not np.array_equal(got, b[off:off + k]):
                bad = int(np.flatnonzero(got != b[off:off + k])[0])
                return f"thread {i} ch {last_ch[i]} byte {off + bad} of {n} differs"
    return None


def ecc_counts(t) -> list[tuple[int, int]] | None:
    """Both channels' (sec_errors, ded_errors), as tools/qual/turnaround.py reads them."""
    try:
        from opentpu.host import ddrcal, memcal
        c = memcal.csr(t)
        return [(ddrcal.Chan(c, ch).r("ecc_sec_errors"), ddrcal.Chan(c, ch).r("ecc_ded_errors"))
                for ch in (0, 1)]
    except Exception:                  # noqa: BLE001 (a bitstream without the CSRs: not counted)
        return None


def host_info(dev: str) -> dict:
    """What the rates depend on, on this PC: the link, the driver's mode, THP, the CPU."""
    info: dict = {}
    rd = lambda p: Path(p).read_text().strip() if Path(p).exists() else None  # noqa: E731
    for p in Path("/sys/bus/pci/devices").glob("*") if Path("/sys/bus/pci/devices").exists() else []:
        if rd(p / "vendor") == "0x10ee":
            info.update(pci=p.name, link_speed=rd(p / "current_link_speed"),
                        link_width=rd(p / "current_link_width"))
    info["poll_mode"] = rd("/sys/module/xdma/parameters/poll_mode")
    info["thp"] = rd("/sys/kernel/mm/transparent_hugepage/enabled")
    info["governor"] = rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    info["kernel"] = os.uname().release
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--json")
    ap.add_argument("--secs", type=float, default=0.3, help="seconds per point")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--sizes", help="comma-separated bytes (multiples of 64)")
    ap.add_argument("--qd", default="1,2,4", help="calls in flight (threads)")
    ap.add_argument("--quick", action="store_true", help=f"sizes {QUICK}, one rep")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    sizes = [int(s) for s in a.sizes.split(",")] if a.sizes else QUICK if a.quick else SIZES
    reps = 1 if a.quick else a.reps
    qds = [int(q) for q in a.qd.split(",")]
    if any(s % B.BEAT or s <= 0 or s > SPAN for s in sizes):
        raise SystemExit(f"sizes must be whole 64-byte beats up to {SPAN}: {sizes}")

    t = B.XdmaTransport(a.dev)
    out = {"host": host_info(a.dev), "args": vars(a), "points": [], "c2h": [], "bounced": []}
    out["build_id"] = f"{t.reg_read(B.R.R_BUILD_ID):08x}"
    st = t.reg_read(B.R_STATUS)
    if st & B.R.ST_RUN and not st & B.R.ST_HALTED:  # a run in flight: the overrun's case
        t.close()
        raise SystemExit("the card is running a program: the sweep needs it idle")
    print(f"h2c_sweep: {out['host']}  BUILD_ID {out.get('build_id')}", flush=True)
    grid = [(n, ch, qd, pg) for n in sizes for ch in ("0", "1", "alt") for qd in qds
            for pg in PAGES]
    order = [p for _ in range(reps) for p in grid]
    random.Random(a.seed).shuffle(order)
    bufs: dict = {}
    fail = None
    with t._dma:                                # no other DMA on this card meanwhile
        out["ecc_before"] = ecc_counts(t)
        for n, chmode, qd, pg in order:
            key = (n, pg)
            have = bufs.setdefault(key, [])
            while len(have) < qd:               # per thread; placed for its card address
                have.append(alloc(n, card_addr(0, len(have)), pg))
            p = run_point(t.h2c, have[:qd], n, chmode, a.secs)
            p.update(bytes=n, ch=chmode, qd=qd, pages=pg,
                     huge_kb=min(HUGE_KB.get(B._addr(b), -1) for b in have[:qd]))
            out["points"].append(p)
            eng = f"{p['engine_gbs']:5.2f}" if p["engine_gbs"] else "    -"
            print(f"{n:>9} B ch {chmode:>3} qd {qd} {pg:3s} {p['gbs']:5.2f} GB/s"
                  f"  call {p['wall_us']:8.1f} us (p90 {p['wall_p90_us']:8.1f})  cpu {p['cpu_us']:7.1f}"
                  f" us ({p['cpu_us_per_page']:.3f}/page)  engine {eng}  calls {p['calls']}",
                  flush=True)
            fail = check(t, have[:qd], n, p["last_ch"])
            if fail:
                break
            if len(bufs) > 6:                   # keep the host memory bounded
                bufs.pop(next(iter(bufs)))
        bufs.clear()
        if not fail:
            for n in sizes:                     # the staging copy path, then card->host reads
                raw = np.empty(n + 128, np.uint8)       # 16 bytes past a beat: bounced
                off = (16 - B._addr(raw)) % 64
                src = raw[off:off + n]
                src[:] = np.random.default_rng(n).integers(0, 256, n, np.uint8)
                c0 = REGION
                k, t0 = 0, time.perf_counter()
                while k < 3 or time.perf_counter() - t0 < a.secs:
                    t.mem_write(0, c0, src)
                    k += 1
                dt = time.perf_counter() - t0
                got = t.mem_read(0, c0, n)
                if not np.array_equal(got, src):
                    fail = f"bounced {n} B: read back differs"
                    break
                out["bounced"].append({"bytes": n, "gbs": k * n / dt / 1e9, "calls": k,
                                       "placed": B._write_ok(B._addr(src), B.BASE[0] + c0)})
                dst = B.placed(n, B.BASE[0] + c0)
                k, t0 = 0, time.perf_counter()
                while k < 3 or time.perf_counter() - t0 < a.secs:
                    t.mem_read(0, c0, n, dst)
                    k += 1
                out["c2h"].append({"bytes": n, "gbs": k * n / (time.perf_counter() - t0) / 1e9,
                                   "calls": k})
                print(f"{n:>9} B  bounced h2c {out['bounced'][-1]['gbs']:5.2f} GB/s  "
                      f"c2h {out['c2h'][-1]['gbs']:5.2f} GB/s", flush=True)
        out["ecc_after"] = ecc_counts(t)
    t.close()
    out["result"] = "FAIL: " + fail if fail else "PASS"
    print(f"ECC before {out['ecc_before']} after {out['ecc_after']}")
    print(out["result"])
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
