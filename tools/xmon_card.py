"""The DMA overlap fault on the card, with the debug build's monitors (XMON=1; otpu_xmon.sv,
opentpu/host/xmon.py). Run on the card's host, alone on the card (under otpu-lock).

    xmon_card.py run SECONDS [overlap|serial] [SEED]
        Fills a read region of each channel with self-describing data, CLEARs the monitors, then
        writes self-describing data (a new tag per write) to random places on both channels and
        reads each write back, while another thread with its own transport reads the read region
        (overlap: host->card and card->host at once, the fault's trigger; serial: the same with
        one lock around every DMA call, the control). Every read is checked on the host. Stops at
        the first bad byte, a failed DMA or a monitor flag, then SNAPs and prints the monitors.
        Exit 0 clean, 3 bad data on the host, 4 a DMA failed, 5 a flag with the host's data clean.
    xmon_card.py snap       SNAP and print the monitors
    xmon_card.py clear      CLEAR them

Writes only 0x40000000 .. 0x7effffff of each channel; reads 0x7f000000 + 16 MiB.
"""
from __future__ import annotations

import sys
import threading
import time

import numpy as np

from opentpu.host import xmon as X
from opentpu.host.board import XdmaTransport

LO, HI = 0x40000000, 0x7F000000
RD, RDN = 0x7F000000, 16 << 20


def now() -> str:
    return time.strftime("%H:%M:%S")


def report(t, why: str) -> None:
    print(f"=== {why} ({now()})", flush=True)
    try:
        print(X.describe(X.snap(t)), flush=True)
    except Exception as e:              # noqa: BLE001 - a report never stops the run
        print(f"monitors unavailable: {e!r}", flush=True)


def reader(stop: threading.Event, lock, stats: dict, seed: int) -> None:
    t2 = XdmaTransport("/dev/xdma0")
    rng = np.random.default_rng(seed + 1000)
    try:
        while not stop.is_set():
            ch = int(rng.integers(2))
            n = int(rng.choice([4096, 65536, 1 << 20]))
            off = RD + int(rng.integers(0, (RDN - n) // 64)) * 64
            with lock:
                got = t2.mem_read(ch, off, n)
            stats["rd"] += n
            b = X.first_bad(got, ch, off)
            if b is not None:
                stats["bad"] = f"reader: ch{ch} {off + b[0]:#x} (of {off:#x}+{n}) described {b[1]:#x}"
                stop.set()
    except Exception as e:              # noqa: BLE001
        stats["bad"] = f"reader DMA failed: {e!r}"
        stats["dma"] = True
        stop.set()
    finally:
        t2.close()


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def run(t, secs: float, mode: str, seed: int) -> int:
    rng = np.random.default_rng(seed)
    for ch in (0, 1):
        for o in range(0, RDN, 1 << 22):
            t.mem_write(ch, RD + o, X.data(ch, RD + o, 1 << 22))
    X.clear(t)
    time.sleep(0.01)
    lock = threading.Lock() if mode == "serial" else _NoLock()
    stats = {"rd": 0, "bad": None}
    stop = threading.Event()
    th = threading.Thread(target=reader, args=(stop, lock, stats, seed), daemon=True)
    th.start()
    print(f"run {mode} {secs:.0f} s seed {seed}, {now()}; flags {X.flags(t):#06x}", flush=True)
    t0 = last = lastf = time.time()
    nw = nb = 0
    rc = 0
    tag = 1
    try:
        while time.time() - t0 < secs and not stop.is_set():
            ch = int(rng.integers(2))
            n = int(rng.choice([64, 128, 256, 512, 4096, 65536, 1 << 20]))
            off = LO + int(rng.integers(0, (HI - LO - n) // 64)) * 64
            d = X.data(ch, off, n, tag)
            with lock:
                t.mem_write(ch, off, d)
            with lock:
                got = t.mem_read(ch, off, n)
            nw += 1
            nb += n
            tag += 1
            b = X.first_bad(got, ch, off)
            if b is not None:
                stats["bad"] = (f"writer: ch{ch} {off + b[0]:#x} (of {off:#x}+{n}, tag {tag - 1}) described"
                                f" {b[1]:#x}, after {time.time() - t0:.1f} s, {nw} writes")
                break
            if time.time() - lastf > 0.5:
                lastf = time.time()
                f = X.flags(t)
                if f:
                    rc = 5
                    print(f"FLAGS {f:#06x} {' '.join(X.flag_names(f))} after {lastf - t0:.1f} s", flush=True)
                    break
            if time.time() - last > 30:
                last = time.time()
                print(f"  {now()} +{last - t0:5.0f} s: {nw} writes, {nb / 1e9:.2f} GB written,"
                      f" {stats['rd'] / 1e9:.2f} GB read; flags {X.flags(t):#06x}", flush=True)
    except Exception as e:              # noqa: BLE001 - a failed DMA is a result
        stats["bad"] = f"writer DMA failed after {time.time() - t0:.1f} s, {nw} writes: {e!r}"
        stats["dma"] = True
    finally:
        stop.set()
        th.join(timeout=30)
    if stats["bad"]:
        print(f"BAD {stats['bad']}", flush=True)
        rc = 4 if stats.get("dma") else 3
    report(t, f"end, rc {rc}: {nw} writes, {nb / 1e9:.2f} GB written, {stats['rd'] / 1e9:.2f} GB read,"
              f" {time.time() - t0:.0f} s")
    return rc


def main(a: list[str]) -> int:
    if not a or a[0] not in ("run", "snap", "clear"):
        print(__doc__)
        return 2
    t = XdmaTransport("/dev/xdma0")
    if not X.present(t):
        print(f"no DMA monitors in this bitstream (0xF00 reads {t.reg_read(X.REG):#010x})")
        return 2
    if a[0] == "snap":
        report(t, "snap")
        return 0
    if a[0] == "clear":
        X.clear(t)
        return 0
    return run(t, float(a[1]), a[2] if len(a) > 2 else "overlap", int(a[3]) if len(a) > 3 else 1)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
