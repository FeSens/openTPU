"""Host DMA in one direction against the accelerator's DRAM traffic in the other, on the same
channels, with no host DMA in the other direction (the qual soak's pattern: decode's streamed
logits are card->host reads while the run writes; its SENTINEL marks are host->card writes while
the run reads). Run on the card's host, alone on the card (under otpu-lock).

    acc_overlap.py SECONDS rd|wr [SEED]
      rd  the accelerator stores 32 MiB per run (rw_bench's "st": 64 KiB tiles), run after run,
          while another thread reads a self-describing region of each channel (card->host) and
          checks every byte
      wr  the accelerator loads 32 MiB per run ("ld") while another thread writes self-describing
          data to that region (host->card), each write checked afterwards
    Then, with nothing overlapping: the stores' region read back (every word 0), and a
    self-describing write and read back on each channel at a fresh place (a host->card slip that
    stuck shows there). Exit 0 clean, 3 bad data, 4 a DMA or a run failed.

The accelerator's region: logical 0 .. ~40 MB (channel offsets below 0x02000000); the host's:
channel offsets 0x7f000000 + 16 MiB, and 0x7e000000 for the closing check.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import rw_bench as RB  # noqa: E402
from opentpu import isa as I  # noqa: E402
from opentpu.host import xmon as X  # noqa: E402
from opentpu.host.board import Board, XdmaTransport, device_config  # noqa: E402

RD, RDN = 0x7F000000, 16 << 20
CHK = 0x7E000000
MAXN = int(os.environ.get("XMON_MAXN", 1 << 20))   # the host's largest transfer (4 KiB, 64 KiB, 1 MiB)
N_DMA = 512                           # 64 KiB tiles per run: 32 MiB


def now() -> str:
    return time.strftime("%H:%M:%S")


def host_side(mode: str, stop: threading.Event, stats: dict, seed: int) -> None:
    t2 = XdmaTransport("/dev/xdma0")
    rng = np.random.default_rng(seed)
    tag = 1
    try:
        while not stop.is_set():
            ch = int(rng.integers(2))
            n = int(rng.choice([k for k in (4096, 65536, 1 << 20) if k <= MAXN] or [MAXN]))
            off = RD + int(rng.integers(0, (RDN - n) // 64)) * 64
            if mode == "rd":
                got = t2.mem_read(ch, off, n)
                b = X.first_bad(got, ch, off)
                if b is not None:
                    stats["bad"] = f"host read: ch{ch} {off + b[0]:#x} (of {off:#x}+{n}) described {b[1]:#x}"
                    stop.set()
            else:
                t2.mem_write(ch, off, X.placed(ch, off, X.data(ch, off, n, tag)))
                tag += 1
            stats["bytes"] += n
    except Exception as e:              # noqa: BLE001 - a failed DMA is a result
        stats["bad"] = f"host DMA failed: {e!r}"
        stats["dma"] = True
        stop.set()
    finally:
        t2.close()


def main(a: list[str]) -> int:
    if len(a) < 2 or a[1] not in ("rd", "wr"):
        print(__doc__)
        return 2
    secs, mode, seed = float(a[0]), a[1], int(a[2]) if len(a) > 2 else 1
    t = XdmaTransport("/dev/xdma0")
    board = Board(t)
    board.scrub()
    cfg = device_config(board.info())
    am = "st" if mode == "rd" else "ld"
    img, lay = RB.image(1, N_DMA)
    prog = RB.bench.trace(cfg, 0, {"m": RB.descs(lay, 1, N_DMA), "mode": am, "n_mm": 1,
                                   "n_dma": N_DMA}).finish()
    words = np.asarray(I.assemble(prog), np.uint32)
    if am == "st":
        img[lay["dma"]:lay["end"]] = 0xAB             # the stores' region: overwritten with 0
    board.write(0, img)
    board.load_program(-(-len(img) // 4096) * 4096, words)
    for ch in (0, 1):                                   # the host's region, self-describing
        for o in range(0, RDN, 1 << 22):
            t.mem_write(ch, RD + o, X.data(ch, RD + o, 1 << 22))
    print(f"{mode}: accelerator '{am}' {N_DMA * 64} KiB per run, {secs:.0f} s, seed {seed}, {now()}"
          f"{'' if os.environ.get('XMON_PLACE') is None else ', host = card + ' + os.environ['XMON_PLACE']}"
          f"{f', transfers <= {MAXN} B' if MAXN < 1 << 20 else ''}",
          flush=True)
    stats = {"bytes": 0, "bad": None}
    stop = threading.Event()
    th = threading.Thread(target=host_side, args=(mode, stop, stats, seed), daemon=True)
    t0 = time.time()
    runs, cyc = 0, 0
    rc = 0
    th.start()
    try:
        while time.time() - t0 < secs and not stop.is_set():
            st = board.run()
            runs += 1
            cyc += st["cycles"]
    except Exception as e:              # noqa: BLE001
        stats["bad"] = stats["bad"] or f"run failed: {e!r}"
        stats["dma"] = True
    finally:
        stop.set()
        th.join(timeout=60)
    dt = time.time() - t0
    print(f"  {runs} runs ({cyc / max(runs, 1) / 1e3:.0f} k cycles each), host {stats['bytes'] / 1e9:.2f} GB"
          f" {'read' if mode == 'rd' else 'written'} in {dt:.0f} s", flush=True)
    if stats["bad"]:
        print(f"BAD during the overlap: {stats['bad']}", flush=True)
        rc = 4 if stats.get("dma") else 3
    # ---- after, nothing overlapping
    try:
        if am == "st":
            got = board.read(lay["dma"], lay["end"] - lay["dma"])
            nz = np.flatnonzero(got)
            print(f"  stores' region: {'all 0' if len(nz) == 0 else f'{len(nz)} bytes not 0, first +{nz[0]:#x}'}",
                  flush=True)
            rc = rc or (3 if len(nz) else 0)
        else:
            for ch in (0, 1):
                got = t.mem_read(ch, RD, RDN)
                w = got.view(np.uint32).reshape(-1, 4)
                a = (np.uint64(0x80000000 * ch) + RD + 16 * np.arange(len(w), dtype=np.uint64)).astype(np.uint32)
                ok = (w[:, 0] == X.MAGIC) & (w[:, 1] == a) & (w[:, 2] == ~a)
                print(f"  host-written region ch{ch}: {'clean' if ok.all() else f'{(~ok).sum()} beats off, first +{16 * int(np.flatnonzero(~ok)[0]):#x}'}",
                      flush=True)
                rc = rc or (0 if ok.all() else 3)
        slip = False
        for ch in (0, 1):
            for k in range(4):
                off, n = CHK + k * (1 << 20), 1 << 20
                t.mem_write(ch, off, X.data(ch, off, n, 0xC0DE + k))
                b = X.first_bad(t.mem_read(ch, off, n), ch, off)
                if b is not None:
                    print(f"  SLIP check ch{ch}: {off + b[0]:#x} described {b[1]:#x}"
                          f" ({b[1] - (0x80000000 * ch + off + b[0]):+d} B)", flush=True)
                    rc = rc or 3
                    slip = True
                    break
        print(f"  host->card check after: {'SLIPPED' if slip else 'clean'}", flush=True)
    except Exception as e:              # noqa: BLE001
        print(f"  check after failed: {e!r}", flush=True)
        rc = 4
    print(f"RESULT {mode} rc {rc} ({now()})", flush=True)
    board.close()
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
