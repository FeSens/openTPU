"""The host side of Board.write's pipelined write, no card: GB/s of the split before (each PIPE
piece copied, hash_swap'd and copied again a channel in the calling thread, the DMA worker
writing the piece before it from the other of two buffer sets; kept here as old_write) and of
Board.write now (_piped: gather by the split threads straight into PIPE_SETS sets of placed
channel buffers, PIPE / 2 bytes a channel a piece), on a fake card whose DMA calls take no time,
or take their bytes' time at --dma-gbs (a sleep: the GIL released, as the driver's write).

    python3 tools/h2c_split_bench.py [--gib 1] [--cases 4x64,2x64,8x128] [--threads 4]
                                     [--dma-gbs 2.9] [--no-chash] [--image FILE]

A case is NCHxBEAT, a slice's channels and their beat bytes (the YPCB-00338's 2x64, the
bcu1525-wg's 8x128). The source is --gib GiB of random bytes, or of --image's file from
--offset (an image cache entry's: its pages through the page cache, as the chat's writes read
them), written at --addr (4 GiB: the weights' blocks, CHASH folds not 0). Before timing, a piece
of each case is split both ways and compared (the bytes per channel must be equal). Prints per
case and way the best and median of --reps writes, in GB/s and s/GB.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opentpu.host import board as B  # noqa: E402
from opentpu.host.fake import FakeTransport, _target_cfg  # noqa: E402


class NullCard(FakeTransport):
    """The fake card's registers (one slice of nch channels of beat bytes, a platform this tree
    has not), its DMA writes dropped, or each one sleeping until its bytes' time at `gbs` after
    the last one's end (an engine's clock: a sleep's overshoot, ~1.4 ms on macOS, is not counted
    against the next call)."""
    threaded = True
    bases = B.BASE

    def __init__(self, nch: int, beat: int, chash: bool, gbs: float = 0.0):
        from opentpu.isasim import board_config
        c = board_config()
        cfg = dict(_target_cfg(c.D, c.MCOLS, c.LANES, False, False, False, c.act_rows),
                   hash=0, S=1, channels=nch, channel_bytes=1 << 33, DRAM_BYTES=nch << 33,
                   beat_bytes=beat, platform=0xEE)
        super().__init__(devname=None, regmap=4, cfg=cfg, ch_bytes=4096, chash=chash)
        self.gbs, self.done, self.late = gbs, 0.0, 0.0

    def mem_write(self, ch, off, data):
        if self.gbs:
            now = time.perf_counter()
            self.done = max(self.done, now - self.late) + len(data) / (self.gbs * 1e9)
            if self.done > now:
                time.sleep(self.done - now)
            self.late = max(0.0, time.perf_counter() - self.done)


def old_split(addr: int, v: np.ndarray, slots: list, chash: bool) -> list:
    """The old pipelined path's split of one piece [n, nch, beat] into its channels' buffers."""
    if chash:                                       # its channel order (a copy of the piece)
        v = v.copy()
        B.hash_swap(addr, v)
    out = []
    for c in range(v.shape[1]):
        buf = slots[c][:len(v) * v.shape[2]]
        np.copyto(buf.reshape(-1, v.shape[2]), v[:, c, :])
        out.append(buf)
    return out


def old_write(b: B.Board, a0: int, data: np.ndarray) -> None:
    """Board._chunks's pipelined path before boardload (PIPE pieces, two buffer sets, the split
    in this thread while the DMA worker writes the piece before)."""
    nch, beat, chash = b.nch, b.beat, b.chash
    h = B.PIPE // nch
    slots = [[B.placed(h, b._base(c) + a0 // nch) for c in range(nch)] for _ in range(2)]
    ex, prev = b._worker(), None
    for i, pos in enumerate(range(0, len(data), B.PIPE)):
        v = data[pos:pos + B.PIPE].reshape(-1, nch, beat)
        bufs = old_split(a0 + pos, v, slots[i % 2], chash)
        parts = [(c, a0 // nch + i * h, buf) for c, buf in enumerate(bufs)]
        if prev is not None:
            prev.result()
        prev = ex.submit(lambda ps: [b.t.mem_write(*p) for p in ps], parts)
    prev.result()


def check(nch: int, beat: int, chash: bool, src: np.ndarray, addr: int) -> None:
    """One piece of the source split both ways: the same bytes on every channel."""
    P = nch * beat
    n = min(len(src), B.PIPE) // P
    v = src[:n * P]
    old = old_split(addr, v.reshape(n, nch, beat), [np.empty(n * beat, np.uint8)
                                                    for _ in range(nch)], chash)
    new = [np.empty((n, beat), np.uint8) for _ in range(nch)]
    for m0 in (0, n // 3):                          # (two threads' parts, one mid-block)
        B.gather(addr, v.reshape(n * nch, beat), new, chash, nch, beat, m0, n if m0 else n // 3)
    for c in range(nch):
        if not np.array_equal(old[c], new[c].reshape(-1)):
            sys.exit(f"{nch}x{beat}: the new split's channel {c} differs from the old one's")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gib", type=float, default=1.0, help="GiB written a case and way")
    ap.add_argument("--cases", default="4x64,2x64,8x128", help="NCHxBEAT,...")
    ap.add_argument("--threads", type=int, default=B.SPLIT_THREADS, help="split threads")
    ap.add_argument("--dma-gbs", type=float, default=0.0,
                    help="also with each DMA call taking its bytes at this rate (GB/s)")
    ap.add_argument("--no-chash", action="store_true", help="the channel interleave unhashed")
    ap.add_argument("--image", help="a file to read the source from (default random bytes)")
    ap.add_argument("--offset", type=int, default=0, help="--image's first byte")
    ap.add_argument("--addr", type=int, default=1 << 32, help="the logical address written")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(argv)
    B.SPLIT_THREADS = a.threads
    n = int(a.gib * (1 << 30)) // B.PIPE * B.PIPE
    if a.image:
        src = np.memmap(a.image, np.uint8, "r", offset=a.offset, shape=(n,))
    else:
        src = np.random.default_rng(0).integers(0, 256, n, dtype=np.uint8)
    chash = not a.no_chash
    print(f"host: {os.cpu_count()} CPUs, numpy {np.__version__}; {n / 2**30:.2f} GiB "
          f"{'from ' + a.image if a.image else 'random'}, CHASH {'on' if chash else 'off'}, "
          f"PIPE {B.PIPE >> 20} MiB, {a.threads} split threads, {B.PIPE_SETS} buffer sets")
    for case in a.cases.split(","):
        nch, beat = (int(x) for x in case.split("x"))
        check(nch, beat, chash, src, a.addr)
        rates = [("free", 0.0)] + ([(f"{a.dma_gbs:g} GB/s", a.dma_gbs)] if a.dma_gbs else [])
        for label, gbs in rates:
            for way, fn in (("old", old_write), ("new", lambda b, x, d: b.write(x, d))):
                b = B.Board(NullCard(nch, beat, chash, gbs), lock=False)
                fn(b, a.addr, src[:B.PIPE * 2])     # (warm: the buffers, the threads)
                ts = []
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    fn(b, a.addr, src)
                    ts.append(time.perf_counter() - t0)
                b.close()
                best, med = min(ts), statistics.median(ts)
                print(f"{nch}x{beat:<4} {way} DMA {label:<9} best {n / best / 1e9:6.2f} GB/s "
                      f"({best / n * 1e9:5.3f} s/GB)  median {n / med / 1e9:6.2f} GB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
