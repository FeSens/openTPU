"""Host driver for the openTPU board (YPCB-00338 over PCIe, Xilinx XDMA).

The card exposes, through the XDMA bridge:
  - its memory channels on the bridge's AXI master: channel j at BASE[j] (the YPCB-00338's two
    DDR3 channels, 2 GiB each; a register map 4 bitstream's from its CFG_* registers:
    channel_map), written with /dev/xdma0_h2c_0 and read with /dev/xdma0_c2h_0 (file offset =
    AXI address);
  - the control registers (rtl/shell/otpu_ctrl.sv; map in regs.py and
    docs/observability.md) on BAR0, /dev/xdma0_user.

The accelerator addresses one logical DRAM interleaved over a slice's NCH channels in beats of
CB bytes (the YPCB-00338's and the BCU1525's two channels of 64 bytes on LiteDRAM, the BCU1525 WG targets'
eight channels of 128; register map 4's CFG_DRAM): logical beat b of wide chunk m = b // NCH
lives at BASE[c] + m * CB on channel c = b % NCH, or, on a bitstream with CAPS.chash,
c = (b % NCH) ^ h(m), h the XOR fold of m's bits into log2(NCH) bits (at NCH 2 the parity of m;
rtl/mem/otpu_native_dram.sv). This driver applies the same map, so the host works with logical
addresses only. On a multi-slice bitstream (CFG_SLICES S > 1: bcu1525-s2's two)
each slice has a logical DRAM of its own on its NCH channels, slice s's channel c being the
card's channel NCH s + c (Board.write / read take the slice).

BoardBackend implements the Engine backend interface (write / read / run, plus prepare and
attach), so `Engine(..., cfg=device_config(board.info()), backend=BoardBackend)` runs Qwen3,
LFM2 or Qwen3.5 on the card; device_config takes MCOLS and LANES from the bitstream's VERSION
register. With transport=SimTransport the identical protocol runs against the Verilator model
of the board (sim/verilator/tb_board.sv) -- the bring-up rehearsal.

A Board takes the device's exclusive lock (runstate.DeviceLock, /tmp/otpu/<dev>.lock) when its
transport names a device (XdmaTransport, FakeTransport); monitors pass lock=False. Register map
version 1 bitstreams (no REGMAP register) work for everything but the counters, the trace and
the temperature: info() reports regmap 1 and snapshot() returns None.

A locked Board ends what an earlier holder left on the card before anything writes it
(Board.quiesce: a process killed mid-run leaves its program running, storing into DRAM), and
stops its own run on the way out (Board.stop: an error, KeyboardInterrupt, SIGTERM / SIGHUP,
which raise SystemExit while the lock is held, or the interpreter's exit).
"""
from __future__ import annotations

import collections
import concurrent.futures
import fcntl
import functools
import glob
import mmap
import os
import struct
import sys
import threading
import time
import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import memcal
from . import regs as R
from .regs import *  # noqa: F401,F403  (the v1 names stay importable from here)
from .regs import (CTRL_CLEAR, CTRL_LOAD, CTRL_RUN, ID_OTPU, R_CTRL, R_CYCLES, R_CYCLES_HI,
                   R_ICOUNT, R_ID, R_PROG_ADDR, R_PROG_N, R_STATUS, ST_AXI_ERR, ST_CALIB0,
                   ST_CALIB1, ST_ERROR, ST_HALTED, ST_LOADING, ST_RUN)
from .runstate import DeviceLock, LockLost, RunnerStatus, ensure_run_dir, open_shared

BEAT = 64                       # bytes per interleave beat (the cards'; Board.beat: the device's)
BASE = (0x0000_0000, 0x8000_0000)
CH_BYTES = 1 << 31              # 2 GiB per channel
DMA_CHUNK = 8 << 20             # bytes per XDMA read/write call: the driver pins the call's user
                                # pages and builds one descriptor list for them; 8 MiB bounds that
                                # (2048 pages) while the per-call cost (~20 us) stays < 1% of the
                                # transfer (8 MiB at 1.7 GB/s is 4.9 ms)
# Host buffer placement for full DMA speed, measured on the card (docs/host.md section 2,
# "DMA and buffer placement"): with d = host address - card address,
#   writes (h2c) need d % 64 == 0, else 0.77 GB/s instead of 1.7;
#   reads (c2h) need d % 4096 in [32, 4064], else 1.16 GB/s instead of 1.65 (so a page-aligned
#   buffer read from a page-aligned card address is the slow case).
# d % 4096 == DMA_PLACE satisfies both. Buffers that do not are bounced through a staging buffer.
DMA_PLACE = 2048
# XDMA's H2C engine laps its read buffer (8 KiB) when the card holds up its writes: a host->card
# call of more than 4 KiB while the channel is busy (a run's traffic, or a card->host read)
# writes data from 8 KiB further on, or stale data, and the engine keeps that slip until the
# bitstream is loaded again (docs/host.md, "XDMA's H2C overrun"). Calls of 4 KiB were clean
# under both. So while a run is in flight Board.write moves at most RUN_H2C bytes per call and
# channel (XdmaTransport.run_h2c); larger writes belong before or after the run.
RUN_H2C = 4096
PIPE = 2 * DMA_CHUNK            # Board.read: logical bytes per pipelined piece; Board.write: a
                                # channel's PIPE / 2 a piece (one DMA call, at any NCH)
PIPE_SETS = 3                   # Board.write: the pieces' sets of channel buffers (the DMA worker
                                # writes one while the split threads fill the next)
SPLIT_THREADS = 4               # Board.write: the threads splitting a piece into its channels
KEEP_ALIGN = 1 << 20            # keep_ranges: an image's data ranges widened to this (logical)
KEEP_GAP = 4 << 20              # keep_ranges: data ranges less apart are one (fewer DMA calls)
POLL_SPIN = 100e-6              # poll: seconds of back-to-back register reads before sleeping
POLL_MAX_SLEEP = 1e-3           # poll: longest sleep between reads
STATUS_INTERVAL = 0.25          # BoardBackend: the status file is rewritten at most this often
POLL_EARLY = 1.5e-3             # poll with an expected wait: wake this much (+ 3%) before it
# (a sleep on a loaded host wakes up to ~1 ms late: at 0.5 ms + 1% the card's LFM2 decode saw
# HALTED 0.35-0.86 ms after the run's end, 0.02 ms when the poll spun throughout; 2026-09-28)
POLL_MIN_SLEEP = 20e-6          # poll: shortest sleep between reads (past the expected end)
# Streamed logits (BoardBackend.start(stream=...)): a word the device never stores marks the
# logits not written yet (the device's NaN is canonical: 0x7FC00000, or 0xFFC00000 after a sign
# flip), so the host reads each piece of the LM head's output as soon as it is complete.
SENTINEL = 0xFFFFFFFF
# A step program compiled with fill (opentpu.llm.qwen3.fill_logits) writes this over its logits
# region itself, at its start (-inf: VOP FILL keeps it; the LM head's logits are finite), and
# the host polls ICOUNT until the fill has landed (start(stream=(..., gate))): no host write.
FILL_SENTINEL = 0xFF800000
STREAM_EARLY = 0.3e-3           # probe a piece this long before it came complete last token
STREAM_RETRY = 0.1e-3           # probe again after this when it is not complete yet
STREAM_PROBE = 0.5e-3           # probe interval when no completion time is known yet
TAIL_SETTLE = 5e-3              # after HALTED: how long the last stores may take to land
WR_SETTLE = 2e-3                # after HALTED: WR_IDLE read back to back this long, then polled
                                # up to QUIET_WAIT (not seen then: an error)
RUN_TIMEOUT = 600.0             # the longest run (Board.wait), and the longest a generate run may
                                # go without a token (run_generate)
# Ending a run (Board.stop; docs/observability.md, "Stopping a run"): CTRL = 0, then STATUS
# WR_IDLE (QUIET), then DRAM writes, a LOAD or a RUN. The run's writes given to the memory
# still land after CTRL = 0: WR_IDLE orders the host's DMA writes after them. Its reads in
# flight the card drops itself (from fix-board on: the loader and the next RUN start once none
# is in flight). Older bitstreams released the core with port B reads in flight (up to 128
# beats a channel) whose data a program load right after took for instruction rows:
# CUT_SETTLE, before QUIET, gives a cut-short run's reads time to come back there (a few us;
# a fallback only, the whole sequence on a current bitstream is CTRL = 0 and QUIET).
QUIET = R.ST_WR_IDLE
QUIET_WAIT = 1.0                # after CTRL = 0: the longest wait for QUIET
CUT_SETTLE = 1e-3               # (bitstreams before fix-board)
STOP_WAIT = 1.0                 # a run stopped by an error or an exit: its time to halt by itself
                                # (the token in flight) before it is cut short
QUIESCE_WAIT = 120.0            # Board(): the longest wait for another process's run to halt
SNAP_TRIES = 8                  # snapshot(): reads again when another process's SNAP came between
RUN_CLOCK_N = 16                # run_clock: a program's time with no waits, the least of its last 16 runs
EXPECT_N = 1024                 # BoardBackend: the run expectations kept (by program length and run
                                # arguments: a decode token's are new every token), the oldest dropped
HOST_IDLE = 50e-6               # BoardBackend.host: the sleep between polls it had nothing for
                                # (a transport's host_idle instead: XdmaTransport's 0)
GEN_POLL_FAR = 8e-3             # run_generate: its longest sleep while the next token is not due yet (out[]
                                # and STATUS are read once a wake-up: at 33 tokens/s a 1 ms cap read the
                                # card ~27 times a token, this ~9; stop() is seen within it)
HOST_TAKE = 1e-3                # run_generate with a host hook: out[] read at most this often (its
                                # two DMA calls would double the hook's poll period)


# ------------------------------------------------------------------------------ address map
# A slice's map: nch channels, beat bytes a beat (the defaults: the cards' two of 64). A wide
# chunk (nch x beat bytes) is one beat on every channel; at nch 2 it is the D = 128 chunk.
def chunk_hash(addr: int, n: int, nch: int = 2, beat: int = BEAT) -> np.ndarray:
    """CHASH: the channel permutation of the n wide chunks from logical `addr` (wide-chunk
    aligned): the XOR fold of each chunk's index into log2(nch) bits (beat j of chunk m on
    channel j ^ h(m)); at nch 2 the parity of the index."""
    m = np.arange(addr // (nch * beat), addr // (nch * beat) + n, dtype=np.uint64)
    cl = (nch - 1).bit_length()
    h = np.zeros(n, np.uint64)
    while m.any():
        h ^= m & np.uint64(nch - 1)
        m >>= np.uint64(cl)
    return h.astype(np.int64)


def swapped(addr: int, n: int) -> np.ndarray:
    """CHASH on two channels: which of the n chunks from logical `addr` (chunk aligned) have
    their halves swapped between the channels (the parity of the chunk index)."""
    return chunk_hash(addr, n).astype(bool)


def hash_swap(addr: int, v: np.ndarray) -> None:
    """Wide chunks [n, nch, beat] at logical `addr`: permute each one's beats as CHASH does, in
    place (its own inverse: logical order <-> channel order; at nch 2 the halves swapped)."""
    n, nch, beat = v.shape
    h = chunk_hash(addr, n, nch, beat)
    if nch == 2:
        p = h.astype(bool)
        v[p] = v[p][:, ::-1]
    else:
        v[:] = v[np.arange(n)[:, None], np.arange(nch)[None, :] ^ h[:, None]]


def beat_at(addr: int, chash: bool = False, nch: int = 2, beat: int = BEAT) -> tuple[int, int]:
    """(channel, channel offset) of the logical byte `addr`: beat b = addr // beat % nch of wide
    chunk m = addr // (nch x beat) on channel b, or b ^ h(m) with CHASH (chunk_hash)."""
    m, c = addr // (nch * beat), addr // beat % nch
    if chash:
        c ^= int(chunk_hash(m * nch * beat, 1, nch, beat)[0])
    return c, m * beat + addr % beat


def split(addr: int, data: np.ndarray, chash: bool = False, nch: int = 2,
          beat: int = BEAT) -> list[tuple[int, int, np.ndarray]]:
    """Logical bytes at `addr` -> [(channel, channel offset, bytes)], beat-aligned pieces
    merged into one contiguous run per channel. `addr` and len(data) must be multiples of
    nch x beat (the caller widens unaligned ranges)."""
    assert addr % (nch * beat) == 0 and len(data) % (nch * beat) == 0
    v = data.reshape(-1, nch, beat)
    if chash:
        v = v.copy()
        hash_swap(addr, v)
    off = addr // nch
    return [(c, off, np.ascontiguousarray(v[:, c, :]).reshape(-1)) for c in range(nch)]


def join(parts: list[np.ndarray], addr: int = 0, chash: bool = False,
         beat: int = BEAT) -> np.ndarray:
    """Inverse of split: the channels' contiguous runs (from logical `addr`) -> logical
    bytes."""
    v = np.stack([p.reshape(-1, beat) for p in parts], axis=1)
    if chash:
        hash_swap(addr, v)
    return v.reshape(-1)


@functools.lru_cache(maxsize=None)
def _gather_tables(nch: int, chash: bool) -> tuple[int, int, tuple[np.ndarray, ...]]:
    """gather's tables: (bl, log2 nch, tabs), a block being bl = nch^k wide chunks (4,096 up to 16
    channels) and tabs[x] [bl] the rows of each chunk's beat for a channel x: m x nch + (x ^
    h(m)) with chash (h(m) of chunk m of the block alone; a channel c of a block whose index's
    fold is H takes tabs[c ^ H]: CHASH's fold of m is the low k digits' ^ the block index's), else
    m x nch + x."""
    cl = (nch - 1).bit_length()
    bl = nch ** max(1, 12 // cl)
    hl = chunk_hash(0, bl, nch, 1) if chash else np.zeros(bl, np.int64)
    m = np.arange(bl, dtype=np.intp) * nch
    return bl, cl, tuple(m + (x ^ hl) for x in range(nch))


def gather(addr: int, rows: np.ndarray, outs: list, chash: bool, nch: int, beat: int,
           m0: int = 0, m1: int | None = None) -> None:
    """split's bytes straight into each channel's buffer: wide chunks m0 .. m1 of the n from
    logical `addr` (wide-chunk aligned), `rows` their beats [n x nch, beat], into outs[c] [n,
    beat] for channel c (CHASH's channel order with chash). One np.take a block of chunks and
    channel, by a table (_gather_tables), the GIL released: no copy of the bytes before it, and
    the split threads of a pipelined write run side by side."""
    bl, cl, tabs = _gather_tables(nch, chash)
    m1 = len(rows) // nch if m1 is None else m1
    g0 = addr // (nch * beat)
    m = m0
    while m < m1:
        b, r = divmod(g0 + m, bl)                   # (the chunk's block, its index there)
        e = min(m1, m + bl - r)
        x = 0                                       # h(the block's index): its digits' fold
        while chash and b:
            x ^= b & (nch - 1)
            b >>= cl
        src = rows[m * nch:e * nch]
        for c in range(nch):
            t = tabs[c ^ x]
            if r or e - m < bl:                     # (a part of the block)
                t = t[r:r + e - m] - r * nch
            np.take(src, t, axis=0, out=outs[c][m:e], mode="clip")  # (raise buffers the out)
        m = e


def channel_map(info: dict | None) -> tuple[int, ...]:
    """The card (AXI) addresses of the memory channels, channel j = NCH s + c being slice s's
    channel c, at j x the bytes a slice addresses per channel: from a register map 4 bitstream's
    CFG_* registers (CFG_SLICES, CFG_DRAM; the YPCB-00338's are BASE), BASE before them."""
    c = (info or {}).get("cfg")
    if not c:
        return BASE
    return tuple(j * c["channel_bytes"] for j in range(c["S"] * c["channels"]))


# ------------------------------------------------------------------------------ transports
def _readinto(fd: int, mv: memoryview, off: int) -> int:
    """A synchronous read at `off` straight into `mv`. Not preadv: vectored reads reach the
    XDMA driver's read_iter, which queues every request as asynchronous I/O and completes it
    from a worker -- on a synchronous request the kernel stops the call (BUG in
    do_iter_readv_writev, kernel 7.1) and the worker jumps to a NULL completion."""
    if hasattr(os, "readinto"):                     # Python 3.14: the driver's plain .read
        os.lseek(fd, off, os.SEEK_SET)
        return os.readinto(fd, mv)
    b = os.pread(fd, len(mv), off)                  # one extra copy
    mv[:len(b)] = b
    return len(b)


class _DmaLock:
    """One DMA call on a card at a time: a host->card and a card->host call never overlap. On the
    card (2026-10-01, builds B 79c5707a at Gen1 and ccae1763 at Gen2), XDMA writes from one thread
    while another thread read from the card slipped the host->card data by 64 bytes for good (every
    later write landed 64 B behind its address on both channels, until the FPGA was reloaded), or
    left the host->card engine BUSY (every later write timed out): none in 225 GB of writes or 11 GB
    of reads alone. A thread lock for the process's threads, and flock on <run dir>/<card>.dma
    for other processes (the device lock does not keep them apart inside one otpu-lock). Calls in
    one direction keep their pipelining with host work (Board.write / read's worker); a call waits
    at most one DMA_CHUNK of the other direction. Reentrant in its thread: a thread that makes
    calls back to back (offload.BoardDram's worker, a request's) holds it around them once, and
    each call inside skips the flock's two system calls (each one more wait for the GIL in a
    busy process)."""

    def __init__(self, name: str, flock: bool = True):
        self.name, self.flock = name, flock
        self._t = threading.RLock()
        self._fd, self._pid = -1, 0
        self._depth = 0                 # the holder's nesting (the flock at the outermost)

    def __enter__(self) -> _DmaLock:
        self._t.acquire()
        if self.flock and not self._depth:
            try:
                if self._pid != os.getpid():           # a forked child opens its own description
                    self._fd = open_shared(ensure_run_dir() / f"{self.name}.dma")
                    self._pid = os.getpid()
                fcntl.flock(self._fd, fcntl.LOCK_EX)
            except BaseException:
                self._t.release()
                raise
        self._depth += 1
        return self

    def __exit__(self, *exc) -> None:
        self._depth -= 1
        if self.flock and not self._depth:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        self._t.release()


_DMA_LOCKS: dict[str, _DmaLock] = {}


def _dma_lock(name: str) -> _DmaLock:
    """The process's DMA lock of card `name` (the device node's name, as the device lock's)."""
    return _DMA_LOCKS.setdefault(name, _DmaLock(name))


def _addr(a: np.ndarray) -> int:
    return a.__array_interface__["data"][0]


def _write_ok(ptr: int, card: int) -> bool:
    return (ptr - card) % 64 == 0


def _read_ok(ptr: int, card: int) -> bool:
    return 32 <= (ptr - card) % 4096 <= 4064


def placed(n: int, card: int) -> np.ndarray:
    """n bytes of fresh host memory at DMA_PLACE from the card address `card` (mod 4096): full
    DMA speed both ways (see DMA_PLACE)."""
    m = mmap.mmap(-1, n + 4096)
    p = (card + DMA_PLACE) % 4096
    return np.frombuffer(m, np.uint8, n, p)


def _placed_in(buf: np.ndarray, card: int, n: int) -> np.ndarray:
    """n bytes of `buf` (page aligned, n + 4096 bytes or more) placed as placed() places them."""
    p = (card + DMA_PLACE) % 4096
    return buf[p:p + n]


class XdmaTransport:
    """The card through the Xilinx XDMA driver (dma_ip_drivers/XDMA/linux-kernel).
    dma=False opens only the register BAR (monitors: no DMA channel is touched)."""
    ecc = True                  # the card's DRAM needs Board.scrub after configuration
    threaded = True             # Board may run the DMA calls in a worker thread (Board.write/read)
    streams = True              # DMA while the accelerator runs (BoardBackend streamed logits)
    run_h2c = RUN_H2C           # Board.write's bytes per call and channel during a run
    # BoardBackend.host's sleep between polls that found nothing: none, the hook's read of the
    # card (a DMA call, ~30 us) paces the loop, so the card's request is seen ~45 us after its
    # post, not ~100 (HOST_IDLE's 50 us slept ~110 on the card's host; docs/offload.md 10.8)
    host_idle = 0.0
    _dma = _DmaLock("", flock=False)    # a transport built bare (tests): its threads only
    bases = BASE                        # the channels' card addresses (Board.info: channel_map)

    def __init__(self, dev: str = "/dev/xdma0", dma: bool = True):
        self.dev, self.devname = dev, Path(dev).name
        # The device lock comes before any open: a process waiting for its turn
        # (OTPU_LOCK_WAIT) must hold no file on the card, or `otpu-setup --rescan` of the
        # holder, which refuses while the device is open, is blocked by the queue.
        self._otpu_lock = DeviceLock(self.devname) if dma else None
        self._dma = _dma_lock(self.devname)     # the card's DMA calls, one at a time
        self.h2c = self.c2h = -1
        try:
            if dma:
                self.h2c = os.open(f"{dev}_h2c_0", os.O_WRONLY)
                self.c2h = os.open(f"{dev}_c2h_0", os.O_RDONLY)
            fd = os.open(f"{dev}_user", os.O_RDWR | os.O_SYNC)
        except BaseException:
            self.close()
            raise
        # BAR0 (1 MiB, the XDMA AXI-Lite master in the block design since the first bitstream)
        # through its 64 KiB windows: the control registers (their first 4 KiB), the LiteDRAM
        # CSRs (R_MEMCAL: memcal; the MIG bitstreams' ECC registers), the XADC / SYSMON
        # (0x30000), and on a DDR4 board each DIMM's core's CSRs (R_DDR4: ddr4); the first
        # four alone if the driver maps less
        try:
            self.regs = mmap.mmap(fd, R.R_DDR4 + 4 * R.DDR4_STRIDE, mmap.MAP_SHARED,
                                  mmap.PROT_READ | mmap.PROT_WRITE)
        except OSError:
            self.regs = mmap.mmap(fd, R.R_DDR4, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        # One 32-bit load / store per register access. A slice of the mmap is copied byte by
        # byte (CPython 3.14): four 1-byte AXI-Lite writes, and otpu_ctrl, which takes the
        # whole word on every write, would keep only the last byte.
        self.words = memoryview(self.regs).cast("I")

    def close(self) -> None:
        """Close the device files and release the lock (the transport is unusable after)."""
        if getattr(self, "words", None) is not None:
            self.words.release()
            self.words = None
        if getattr(self, "regs", None) is not None:
            self.regs.close()
            self.regs = None
        for name in ("h2c", "c2h"):
            if getattr(self, name, -1) >= 0:
                os.close(getattr(self, name))
                setattr(self, name, -1)
        if self._otpu_lock is not None:
            self._otpu_lock.release()
            self._otpu_lock = None

    def _stage(self, n: int, card: int) -> np.ndarray:
        """A slice of the persistent staging buffer, placed for `card` (see DMA_PLACE)."""
        if getattr(self, "_stage_buf", None) is None:
            self._stage_buf = np.frombuffer(mmap.mmap(-1, DMA_CHUNK + 8192), np.uint8)
        p = (card + DMA_PLACE) % 4096
        return self._stage_buf[p:p + n]

    def mem_write(self, ch: int, off: int, data: np.ndarray) -> None:
        """Whole 64-byte beats only: a range that does not start and end on a beat boundary is
        widened, its edge beats read and merged here. Sub-beat host writes (partial strobes, a
        read-modify-write in the ECC controller) can wedge the card's write path for good
        (docs/host.md, "Sub-beat writes")."""
        a = np.ascontiguousarray(data, np.uint8).reshape(-1)
        if len(a) == 0:
            return
        if off % BEAT or len(a) % BEAT:
            o0, o1 = off // BEAT * BEAT, -(-(off + len(a)) // BEAT) * BEAT
            buf = self.mem_read(ch, o0, BEAT) if o1 - o0 == BEAT else None
            if buf is None:
                buf = np.empty(o1 - o0, np.uint8)
                buf[:BEAT] = self.mem_read(ch, o0, BEAT)
                buf[-BEAT:] = self.mem_read(ch, o1 - BEAT, BEAT)
            buf[off - o0:off - o0 + len(a)] = a
            a, off = buf, o0
        base, pos = self.bases[ch] + off, 0
        while pos < len(a):
            k = min(DMA_CHUNK, len(a) - pos)
            src = a[pos:pos + k]
            with self._dma:                         # never during a card->host call
                if not _write_ok(_addr(src), base + pos):   # bounce: 0.77 -> 1.7 GB/s
                    st = self._stage(k, base + pos)
                    st[:] = src
                    src = st
                n = os.pwrite(self.h2c, memoryview(src), base + pos)
            if n <= 0:
                raise IOError("XDMA h2c write failed")
            pos += n

    def mem_read(self, ch: int, off: int, n: int, out: np.ndarray | None = None) -> np.ndarray:
        """n bytes of channel `ch` at `off`, DMA'd into `out` (or a new buffer placed for full
        speed); an `out` that is placed badly (see DMA_PLACE) gets them through the staging
        buffer."""
        base = self.bases[ch] + off
        if out is None:     # small reads are latency-bound: no mmap for them
            buf = placed(n, base) if n >= 64 << 10 else np.empty(n, np.uint8)
        else:
            buf = out
        mv = memoryview(buf).cast("B")
        direct = n < 64 << 10 or _read_ok(_addr(buf), base)
        pos = 0
        while pos < n:
            want = min(DMA_CHUNK, n - pos)
            with self._dma:                         # never during a host->card call
                if direct:
                    k = _readinto(self.c2h, mv[pos:pos + want], base + pos)
                else:
                    st = self._stage(want, base + pos)
                    k = _readinto(self.c2h, memoryview(st), base + pos)
                    mv[pos:pos + k] = memoryview(st)[:k]
            if k <= 0:
                raise IOError("XDMA c2h read failed")
            pos += k
        return buf

    def reg_write(self, off: int, val: int) -> None:
        self.words[off >> 2] = val & 0xFFFFFFFF

    def reg_read(self, off: int) -> int:
        return self.words[off >> 2]

    def reg_read_many(self, offs: list[int]) -> list[int]:
        return [self.reg_read(o) for o in offs]

    def poll(self, off: int, mask: int, val: int, timeout: float = 600.0,
             expect: float = 0.0) -> int:
        """Wait until (reg & mask) == val. Back-to-back reads (~1 us each over PCIe) for the
        first POLL_SPIN seconds catch short waits (program loads) without a sleep's latency;
        then sleeps of elapsed/32, at most POLL_MAX_SLEEP: the wake-up comes at most ~3% of
        the run late (a Qwen3-0.6B token is ~50 ms: <= 1 ms, 2%), the register is read about
        32 ln(T / 100 us) + T / 1 ms times instead of T / 1 us, and the sleeps release the
        GIL to the thread compiling the next token's program (Engine pipelining).

        expect: the wait the caller expects (seconds; the previous token's run time). The
        poll then sleeps once until POLL_EARLY (+ 3%) before it and reads back to back from there to
        POLL_SPIN past it, so a run that ends on time is seen within a register read; past
        that the sleeps are 1/32 of the time since the expected end (at least POLL_MIN_SLEEP),
        so a run a little late is still seen within ~POLL_MIN_SLEEP, not a whole
        POLL_MAX_SLEEP later (before: 1.0-1.2 ms per Qwen3 token on the card)."""
        t0 = time.perf_counter()
        early = POLL_EARLY + expect * 0.03
        while (left := expect - early - (time.perf_counter() - t0)) > 0:
            time.sleep(left / 2 if left > 1e-3 else left)   # halving: a long sleep overshoots
        spin = max(expect, 0.0) + POLL_SPIN                  # by a few % (macOS: 2-3 ms of 50)
        # exact: the condition was seen by back-to-back reads that saw it false just before
        # (not at the first read, nor after a sleep): its time is known to a read
        self.poll_exact = False
        while True:
            r = self.reg_read(off)
            if r & mask == val:
                return r
            self.poll_exact = True
            el = time.perf_counter() - t0
            if el > timeout:
                raise TimeoutError(f"register {off:#x} = {r:#x}, waiting for {val:#x}/{mask:#x}")
            if el > spin:
                late = el - max(expect, 0.0)
                time.sleep(min(POLL_MAX_SLEEP, max(POLL_MIN_SLEEP, late / 32)))
                self.poll_exact = False


def present_devs() -> list[str]:
    """The XDMA devices of this host, by name order: "/dev/xdma0", ..."""
    return sorted(p[:-len("_user")] for p in glob.glob("/dev/xdma*_user"))


def default_dev(present=None, ident=None) -> str:
    """The card's name when none is given: OTPU_DEV; else, of the XDMA devices present, the
    only one, or the only one whose ID register reads openTPU's (a BCU1525 in SAFE fences its
    registers, another design's card has another ID: with one card up beside them, it is that
    one); else /dev/xdma0, as before. One register read a device, nothing written, no lock.
    present, ident: the devices' list and a device's ID (tests)."""
    env = os.environ.get("OTPU_DEV")
    if env:
        return env
    devs = present_devs() if present is None else list(present)
    if len(devs) == 1:
        return devs[0]

    def read_id(dev: str):
        try:
            t = XdmaTransport(dev, dma=False)
        except OSError:
            return None
        try:
            return t.reg_read(R_ID)
        finally:
            t.close()
    up = [d for d in devs if (ident or read_id)(d) == ID_OTPU]
    return up[0] if len(up) == 1 else "/dev/xdma0"


def open_transport(dev: str | None = None, dma: bool = True, target: str | None = None):
    """The card's transport by name: an XDMA device's path, "/dev/xdmaN" (XdmaTransport: the
    YPCB-00338, the BCU1525; any path, a test's too); cards in tensor parallel as
    "/dev/xdma0+/dev/xdma1" (card 0 first: opentpu.host.cards.open_cards; target: the build
    they must hold, a variant by name, bcu1525-x2:st16: check_target); None: default_dev().
    dma=False: a monitor's (no DMA channel, no device lock)."""
    dev = dev or default_dev()
    if "+" in dev:
        from .cards import open_cards
        return open_cards(dev.split("+"), dma=dma, target=target)
    if "/" in dev:
        return XdmaTransport(dev, dma=dma)
    raise ValueError(f"no transport for {dev!r} (/dev/xdmaN or another path)")


def sim_sources(bcuwg: bool = False) -> tuple[str, list[Path]]:
    """The board model's top (tb_board; bcuwg: tb_bcuwg, the BCU1525 at WG 8's bridges, DIMM pairs and hmuxes around it) and its sources
    (SimTransport's build; test_wg_rtl's port-width lint)."""
    from opentpu import rtlsim
    root = Path(__file__).resolve().parents[2]
    srcs = [rtlsim.RTL / s for s in rtlsim.RTL_SOURCES if not s.endswith("otpu_top.sv")]
    shell = root / "rtl/shell"                  # the FPGA boards' shell (docs/targets.md)
    srcs += [shell / "otpu_ctrl.sv", shell / "otpu_trace.sv", shell / "otpu_board.sv"]
    srcs += [rtlsim.TB / "otpu_native_mem.sv", shell / "otpu_afifo.sv", shell / "otpu_mem_ch.sv",
             shell / "otpu_axi_split2.sv",      # otpu_native_reg (tb_board NREG)
             rtlsim.TB / "otpu_chmem.sv", rtlsim.TB / "otpu_ldn_model.sv"]
    top = "tb_bcuwg" if bcuwg else "tb_board"
    if bcuwg:
        srcs += [root / "rtl/mem" / f for f in ("otpu_hbm_ch.sv", "otpu_axi_rs.sv")]
        srcs += [root / "rtl/boards/bcu1525" / f for f in ("otpu_mig_pair.sv", "otpu_mig_hmux.sv",
                                                           "otpu_bcu_wg_sys.sv")]
        srcs += [rtlsim.TB / "otpu_dimm_model.sv"]
    return top, srcs + [rtlsim.TB / f"{top}.sv"]


class SimTransport:
    """The Verilator model of the board (sim/verilator/tb_board.sv): the memory lives here as
    the channels' physical images (`nch` per slice of `target`'s build, `beat`-byte beats; an
    FPGA target's channels other than the cards' two of 64 bytes run the native memory model
    with the target's channel rate and latency, opentpu.target.mem_model), register operations
    are queued and replayed by the testbench when a result is needed (a flush).

    Every flush is a fresh simulation: DRAM persists (through the channel image files), but the
    control registers, IMEM and TMEM start from reset. The Board protocol is built for that --
    a program load, its run and the reads of its counters happen in one flush -- and the Qwen3
    step programs do not rely on TMEM surviving between runs.

    Batched use (batched = True): queue_read() queues a read and returns its index in the next
    flush's `results` (reads keep their order and may repeat an address); wait_cycles(n) queues
    the testbench's `C` command (wait n core cycles). Together they sample the free-running
    counters twice in one simulation (otpu-smi --sim) and read the trace buffer out in the flush
    that ran the program.

    tb_bcuwg at `caw` (bcuwg; otpu_bcu_wg_sys's CAW, a channel 2^(caw + 7) bytes: 25 the 4 GiB
    of a 32 GB card's channels, 26 64 GB's; None the target's, bcu1525-wg's 25, bcu1525-s2's 24
    (two slices, a channel a quarter DIMM: otpu_bcu_wg_sys S 2, DMAP and RLY in `params`); the model's
    CFG_HASH the target's build's at the DRAM its channels give): a channel's bytes past its dense
    image (ch_bytes, DENSE at most) are held here sparsely, in 32-byte beats (`hi`: {beat: 32
    bytes}; a beat not there reads 0), handed to the DIMM models as ch<c>_hi.txt and read back from
    ch<c>_hi_out.txt (otpu_dimm_model; docs/bcu1525_wg.md)."""

    batched = True
    devname = None              # private to this process: no device lock
    keeps_state = False         # IMEM and the ARG registers start from reset every flush
    DENSE = 1 << 29             # (caw) the dense image's bytes a channel at most: tb_bcuwg's WORDS
                                # is an int, a slice's 8 channels' words, and wraps from 2^30
                                # bytes a channel (bcu1525-s2's 16 GiB a slice: WORDS 2^32 = 0)

    def __init__(self, ch_bytes: int = 1 << 24, stall: int = 20, seed: int = 1,
                 params: dict | None = None, plusargs: list | None = None,
                 native: bool | str | None = None, target=None,
                 bcuwg: bool = False, caw: int | None = None):
        from opentpu.target import current, load
        # the target whose build the model is (default the current one): its slices (two
        # channels each), CFG_PLATFORM and CFG_HASH
        self.target = load(target) if isinstance(target, str) else (target or current())
        self.slices = self.target.arch["S"]
        pl = self.target.platform
        self.nch, self.beat = (pl["channels_per_slice"], pl["beat_bytes"]) \
            if self.target.kind == "fpga" else (2, BEAT)
        # the BCU1525 at WG 8 (sim/verilator/tb_bcuwg.sv: otpu_bcu_wg_sys, the bridges and the
        # DIMMs' pairs in front of models of the DIMMs' AXI4 ports): one slice of 8 channels, or
        # two (bcu1525-s2: each on two DIMMs)
        self.bcuwg = bcuwg
        if bcuwg and ((self.nch, self.beat) != (8, 128) or self.slices not in (1, 2)):
            raise ValueError(f"bcuwg: {self.target.name} is not one or two slices of 8 channels "
                             f"of 128 bytes")
        if bcuwg and caw is None:                   # (the target's channels: its build's CAW)
            caw = pl["channel_bytes"].bit_length() - 8
        self.caw = caw
        if caw is not None and not (bcuwg and 22 <= caw <= 26 and ch_bytes <= 1 << caw + 7):
            raise ValueError(f"caw {caw}: tb_bcuwg's (bcuwg), 22 .. 26, a channel 2^(caw + 7) "
                             f"bytes from its dense image ({ch_bytes})")
        if caw is not None:
            ch_bytes = min(ch_bytes, self.DENSE)
        self.ch = [np.zeros(ch_bytes, np.uint8) for _ in range(self.nch * self.slices)]
        self.hi: list[dict] = [{} for _ in self.ch]     # (caw: the beats past the dense image)
        self.stall, self.seed = stall, seed
        self.params = params or {}
        self.native = native                    # the channels (rtlsim.MEMORY["NATIVE"]):
                                                # "ld" the LiteDRAM build's (otpu_chmem), True
                                                # the native memory model alone
        self.plusargs = list(plusargs or [])     # extra simulator arguments (e.g. "+trace")
        self.out = ""                            # the last flush's simulator output
        self.script: list[str] = []
        self.nreads = 0                         # reads queued in the pending script
        self.results: list[int] = []            # values of the last flush's reads, in order
        self.regs_seen: dict[int, int] = {}     # last value read per address
        self.cycles = 0

    @property
    def gather(self) -> bool:
        """The model's collective unit runs GATHER (CAPS2 bit0): every multi-slice build, and a
        one-slice build unless its COLL_GATHER parameter is 0 (`params`', else the simulation's
        uarch: rtlsim.UARCH, under OTPU_UARCH=board the card's BAR-only unit)."""
        return self.slices > 1 or str(self._coll_gather()) not in ("0", "1'b0")

    @property
    def icache(self) -> bool:
        """The model's IMEM is an instruction cache over the program in DRAM (CAPS2 bit12,
        otpu_seq ICACHE): `params`' ICACHE, else the simulation's (rtlsim.UARCH: OTPU_ICACHE),
        else its target's."""
        if "ICACHE" in self.params:
            return bool(int(self.params["ICACHE"]))
        from opentpu import rtlsim              # (the simulator's: not on the card's path)
        return bool(int(rtlsim.UARCH.get("ICACHE", self.target.arch["ICACHE"])))

    def _coll_gather(self):
        if "COLL_GATHER" in self.params:
            return self.params["COLL_GATHER"]
        from opentpu import rtlsim              # (the simulator's: not on the card's path)
        return rtlsim.UARCH.get("COLL_GATHER", 1)

    def mem_write(self, ch: int, off: int, data: np.ndarray) -> None:
        self.flush()
        if self.caw is None:
            self.ch[ch][off:off + len(data)] = data
            return
        data = np.ascontiguousarray(data, np.uint8).reshape(-1)
        k = min(len(data), max(0, len(self.ch[ch]) - off))
        self.ch[ch][off:off + k] = data[:k]
        if k < len(data):
            self._hi_write(ch, off + k, data[k:])

    def mem_read(self, ch: int, off: int, n: int, out: np.ndarray | None = None) -> np.ndarray:
        self.flush()
        if self.caw is None:
            if out is None:
                return self.ch[ch][off:off + n].copy()
            out[:] = self.ch[ch][off:off + n]
            return out
        r = np.empty(n, np.uint8) if out is None else out
        k = min(n, max(0, len(self.ch[ch]) - off))
        r[:k] = self.ch[ch][off:off + k]
        if k < n:
            r[k:] = self._hi_read(ch, off + k, n - k)
        return r

    HI_BEAT = 32                # otpu_dimm_model's sparse beats (bytes)

    def _hi_beats(self, ch: int, b0: int, b1: int) -> list:
        """The sparse store's beats of channel ch in [b0, b1), ascending."""
        hi = self.hi[ch]
        if b1 - b0 < len(hi):
            return [b for b in range(b0, b1) if b in hi]
        return sorted(b for b in hi if b0 <= b < b1)

    def _hi_read(self, ch: int, off: int, n: int) -> np.ndarray:
        """n bytes of channel ch from byte off, at or past its dense image."""
        B = self.HI_BEAT
        b0, b1 = off // B, -(-(off + n) // B)
        buf = np.zeros((b1 - b0) * B, np.uint8)
        for b in self._hi_beats(ch, b0, b1):
            buf[(b - b0) * B:(b - b0 + 1) * B] = self.hi[ch][b]
        return buf[off - b0 * B:off - b0 * B + n]

    def _hi_write(self, ch: int, off: int, data: np.ndarray) -> None:
        """data to channel ch from byte off, at or past its dense image: its beats in the sparse
        store (a partial beat's other bytes kept; an all-zero beat leaves the store)."""
        B, end = self.HI_BEAT, off + len(data)
        if end > 1 << self.caw + 7:
            raise ValueError(f"channel {ch}: bytes {off:#x} .. {end:#x} past its 2^{self.caw + 7}"
                             f" (CAW {self.caw})")
        b0, b1 = off // B, -(-end // B)
        if off % B or end % B:                  # (the edge beats: read, modify, write)
            buf = self._hi_read(ch, b0 * B, (b1 - b0) * B).copy()
            buf[off - b0 * B:end - b0 * B] = data
            data = buf
        rows = data.reshape(-1, B)
        nz = rows.any(axis=1)
        hi = self.hi[ch]
        for b in self._hi_beats(ch, b0, b1):
            if not nz[b - b0]:
                del hi[b]
        for i in np.flatnonzero(nz):
            hi[b0 + int(i)] = rows[i].copy()

    def reg_write(self, off: int, val: int) -> None:
        self.script.append(f"W {off:x} {val & 0xFFFFFFFF:x}")

    def queue_read(self, off: int) -> int:
        self.script.append(f"R {off:x}")
        self.nreads += 1
        return self.nreads - 1

    def wait_cycles(self, n: int) -> None:
        self.script.append(f"C {n:x}")

    def reg_read(self, off: int) -> int:
        return self.reg_read_many([off])[0]

    def reg_read_many(self, offs: list[int]) -> list[int]:
        """All reads in one simulation, in order (a later flush starts a fresh machine)."""
        idx = [self.queue_read(o) for o in offs]
        self.flush()
        return [self.results[i] for i in idx]

    def poll(self, off: int, mask: int, val: int, timeout: float = 0, expect: float = 0) -> int:
        self.script.append(f"P {off:x} {mask:x} {val:x}")
        return val

    @staticmethod
    def _wrap(srcs: list, p: dict) -> tuple[list, dict]:
        """tb_bcuwg's build from tb_board's parameters: S, WORDS, CORE_KHZ and RLY (the far
        bridges' relays) and otpu_bcu_wg_sys's MWAIT, ARD, PRD, CAW, BHW and DMAP tb_bcuwg's (at
        two slices RLY a literal of its 64 bits), the channels the wrapper's own (its native
        ports: MEM_NATIVE, NCH, CB), every other one otpu_board's, given as the wrapper's
        OTPU_BOARD (as make_wg.py gives the bitstream's) in a source before the others; a key
        otpu_board does not declare is an error (boardmacro.board_macro)."""
        import hashlib
        from opentpu import rtlsim
        from . import boardmacro
        p = dict(p)
        tb = {k: p.pop(k) for k in ("S", "WORDS", "CORE_KHZ", "RLY", "MWAIT", "ARD", "PRD",
                                    "CAW", "BHW", "DMAP") if k in p}
        if tb.get("S", 1) not in (1, 2):            # (tb_bcuwg: otpu_bcu_wg_sys's one or two)
            raise ValueError("bcuwg: one or two slices")
        if tb.get("S", 1) > 1 and isinstance(tb.get("RLY"), int):   # (4 S NCH bits)
            tb["RLY"] = f"{32 * tb['S']}'h{tb['RLY']:x}"
        for k in ("MEM_NATIVE", "NCH", "CB"):
            p.pop(k, None)
        p.setdefault("BUILD_ID", 0x0B0A_4D00)      # (tb_board's)
        text = boardmacro.board_defines(p)
        d = rtlsim.BUILD / "bcuwg"
        d.mkdir(parents=True, exist_ok=True)
        vh = d / f"otpu_board_{hashlib.sha1(text.encode()).hexdigest()[:12]}.sv"
        if not vh.exists():
            tmp = vh.with_name(f".{vh.name}.{os.getpid()}")
            tmp.write_text(text)
            tmp.replace(vh)
        return [vh, *srcs], tb

    def sim_build(self) -> tuple[str, list, dict]:
        """The model's Verilator top, sources and parameters (flush's build; test_wg_rtl's
        port-width lint)."""
        from opentpu import rtlsim
        native = rtlsim.MEMORY["NATIVE"] if self.native is None else self.native
        top, srcs = sim_sources(self.bcuwg)
        from opentpu.isasim import board_config
        cfg = board_config()                        # OTPU_MCOLS / OTPU_LANES: the "bitstream"
        # VPU_CL and ULANES as the bitstream builds them (make bit: VPU_CL 2, ULANES 8)
        p = {"S": self.slices, "WORDS": self.nch * len(self.ch[0]) // 4, "MCOLS": cfg.MCOLS,
             "LANES": cfg.LANES,
             "ACT_ROWS": cfg.act_rows, "TMEM_WORDS": cfg.TMEM_WORDS,      # (release r2: 128K)
             "VPU_CL": rtlsim.UARCH.get("VPU_CL", 2), "ULANES": rtlsim.UARCH.get("ULANES", 8),
             "MXU_IMPL": rtlsim.UARCH.get("MXU_IMPL", 2),
             # the MXU's epilogue lanes as the target builds them (OTPU_MXU_E overrides)
             "MXU_E": rtlsim.UARCH.get("MXU_E", self.target.uarch.get("MXU_E", 0)),
             "MEM_NATIVE": 2 if native == "ld" else 1}
        # RSCALE's row factor read once a row as the target builds it (OTPU_QUANT_R1 overrides)
        if rtlsim.UARCH.get("QUANT_R1", self.target.uarch.get("QUANT_R1", 0)):
            p["QUANT_R1"] = 1
        # and a buffered QACT pass 1's chunks a cycle (OTPU_QUANT_PW overrides)
        pw = rtlsim.UARCH.get("QUANT_PW", self.target.uarch.get("QUANT_PW", 1))
        if pw > 1:
            p["QUANT_PW"] = int(pw)
        # and QST's SW port words (OTPU_QUANT_SWW overrides)
        sww = rtlsim.UARCH.get("QUANT_SWW", self.target.uarch.get("QUANT_SWW", 1))
        if sww > 1:
            p["QUANT_SWW"] = int(sww)
        # and QST rows through the row buffer (OTPU_QUANT_QSTB overrides)
        if rtlsim.UARCH.get("QUANT_QSTB", self.target.uarch.get("QUANT_QSTB", 0)):
            p["QUANT_QSTB"] = 1
        own = (self.nch, self.beat) != (2, BEAT)   # the WG targets' channels: the native memory model
        if own:
            p.update(NCH=self.nch, CB=self.beat, MEM_NATIVE=1)
        # the streams' engine as the target builds it (OTPU_SE_R overrides)
        se_r = rtlsim.UARCH.get("SE_R", self.target.uarch.get("SE_R", 0))
        if se_r:
            p["SE_R"] = se_r
        # the MXU's command queue as the target builds it (OTPU_MXU_QD overrides)
        qd = rtlsim.UARCH.get("MXU_QD", self.target.uarch.get("MXU_QD", 2))
        if qd != 2:
            p["MXU_QD"] = int(qd)
        if not cfg.OUTER:                           # a VPU without VOP OUTER (CAPS2 bit1)
            p["OUTER"] = 0
        if cfg.DSTEP2:                              # DSTEP's pair (CAPS2 bit2)
            p["DSTEP2"] = 1
        if cfg.KT:                                  # MM's K tail (CAPS2 bit5)
            p["KT"] = 1
        from opentpu.target import rtl_params
        rp = rtl_params(self.target)
        # DSTEP's state in 16 bits (CAPS2 bits 14:13), STREAM's ST16R / ST16W (bit18), QACT / QST
        # flag LO (bit17), QACT flag HL (bit21), QST HL (bit22) and TMEM's writes a bank a cycle
        # (otpu_board WPB, 1 unless given: x2:wmx's pumped TMEM 2) as the model's target builds
        # them, never the environment's board_config (OTPU_TARGET, the YPCB by default: x2's board
        # models were built at WPB 1 without the quantizer's flags, docs/pf500.md)
        if rp.get("ST16"):
            p["ST16"] = rp["ST16"]
        for k in ("ST16S", "QUANT_SPLIT", "QUANT_HL", "QUANT_QSTHL", "MXU_SPL"):
            if rp.get(k):
                p[k] = 1
        if rp["WPB"] != 1:
            p["WPB"] = rp["WPB"]
        # the VPU's options: the bitstream's, or the target's build (a model compiled for the
        # target's FSUM, SIGMOID / SILU or MAC needs its VPU built with them, as VPU_CNS below)
        if cfg.FSUM or rp.get("VPU_FSUM"):          # EXP2SUB's fused row sums (CAPS2 bit16)
            p["VPU_FSUM"] = 1
        if cfg.VSIG or rp.get("VPU_SIG"):           # VOP SIGMOID / SILU (CAPS2 bit19)
            p["VPU_SIG"] = 1
        if cfg.VMAC or rp.get("VPU_MAC"):           # VOP MAC (CAPS2 bit20)
            p["VPU_MAC"] = 1
        if (cfg.OUTER1 or rp.get("VPU_OUTER1")) and not cfg.OUTER:   # OUTER decay one (bit23)
            p["VPU_OUTER1"] = 1
        if cfg.QONCE or rp.get("QONCE"):            # QACT ST, LDA (CAPS2 bit24)
            p["QONCE"] = 1
        # the VPU's composites (VPU_CNS 5: two passes), the MXU's pumped array (MXU_PUMP, on
        # the board's clk_f) and the DMA's commands in flight (DMA_LDQ) as the target builds them
        # (OTPU_UARCH's rtlsim.UARCH overrides)
        from opentpu.target import sim_uarch
        su = sim_uarch(self.target)
        for k in ("VPU_CNS", "MXU_PUMP", "DMA_LDQ"):
            v = rtlsim.UARCH.get(k, su.get(k))
            if v is not None:
                p[k] = int(v)
        if rtlsim.UARCH.get("SEQ_WC"):              # (EXPERIMENTAL: WAITC, CFG_UARCH bit24)
            p["SEQ_WC"] = 1
        # IMEM an instruction cache (CAPS2 bit12; the target's, OTPU_ICACHE overrides), its
        # lines and its prefetch as the target builds them (OTPU_IC_LINES / _PF override), and
        # its longest program (CFG_IMEM) where it is not tb_board's: an ICACHE target's (the
        # model's target, not the environment's board_config: bcu1525-wg:ic's 16,384
        # instructions; the CFG_HASH below takes the same), else the board configuration's
        if self.icache:
            p["ICACHE"] = 1
            for k in ("IC_LINES", "IC_PF"):
                v = rtlsim.UARCH.get(k, self.target.uarch.get(k))
                if v is not None:
                    p[k] = int(v)
            w = self.target.arch["IMEM_WORDS"] if self.target.arch["ICACHE"] else cfg.IMEM_WORDS
            if w != 1 << 15:
                p["IMEM_WORDS"] = w
        # the sequencer's window as the target builds it (otpu_board WIN, 16 unless given: the
        # BCU's bhw variant and bcu1525-s2 32, their entries' counts saturated, Config.WC_WIN)
        from opentpu.target import sim_uarch
        if sim_uarch(self.target)["WIN"] != 16:
            p["WIN"] = sim_uarch(self.target)["WIN"]
        if "COLL_GATHER" in rtlsim.UARCH:           # the board's: BAR alone at S = 1 (gather)
            p["COLL_GATHER"] = int(rtlsim.UARCH["COLL_GATHER"])
        # the TMEM as the target builds it (TMEM_IMPL 3: the one copy and its knobs, pumped on
        # tb_board's clk_f with PMP 1); OTPU_TMEM_* (rtlsim.UARCH) override
        from opentpu.target import sim_uarch
        tm = {k: v for k, v in sim_uarch(self.target).items() if k.startswith("TMEM_")}
        tm.update({k: v for k, v in rtlsim.UARCH.items()
                   if k.startswith("TMEM_") and k not in ("TMEM_RQ", "TMEM_NCP")})
        if tm.get("TMEM_IMPL") == 3:
            tm.setdefault("TMEM_SBP", max(3, (cfg.LANES - 1).bit_length()))   # (rtlsim's)
            p.update(tm)
        # the adapter's depths as the target builds them (otpu_board NAT_*; the cards' default)
        p.update({k: v for k, v in sim_uarch(self.target).items() if k.startswith("NAT_")})
        for k in ("NAT_XSTG", "NAT_XSTG_P"):        # (OTPU_NAT_XSTG / _P override)
            if k in rtlsim.UARCH:
                p[k] = rtlsim.UARCH[k]
        if not p.get("NAT_XSTG"):
            p.pop("NAT_XSTG", None)
        # the collective hub's reads in flight and the crossings' stages as the target builds them
        # (otpu_board COLL_PIPE, XRS; OTPU_COLL_PIPE / OTPU_XRS override; S > 1)
        from opentpu.target import COLL_KEYS
        for k in COLL_KEYS:
            v = os.environ.get(f"OTPU_{k}") or self.target.uarch.get(k)
            if v is not None and self.slices > 1:
                p[k] = int(v)
        # and the MXU's chunk FIFO, its lanes and its result FIFO (otpu_board FIFO_DEPTH, the
        # the WG targets' 2,048; MXU_FIFO_STYLE: the WG targets' in UltraRAM, its posted beats staged across the
        # clocks; MXU_RF_STYLE: the WG targets' in LUT RAM; MXU_RF_ROWS: the WG targets' 32)
        p.update({k: v for k, v in sim_uarch(self.target).items()
                  if k in ("FIFO_DEPTH", "MXU_FIFO_STYLE", "MXU_RF_STYLE", "MXU_RF_ROWS")})
        # the DRAM addresses' bits as the target builds them (otpu_board AW: above 32 only) and
        # tb_bcuwg's channels' beat address bits (caw)
        if self.target.arch.get("AW", 32) > 32:
            p["AW"] = self.target.arch["AW"]
        if self.caw is not None:
            p["CAW"] = self.caw
        # and its pairs' bank-half window as the target builds it (otpu_mig_pair BHW: a
        # channel's bytes; Config.BHW the slice's)
        if self.bcuwg and self.target.arch.get("BHW"):
            p["BHW"] = self.target.arch["BHW"] - (self.nch - 1).bit_length()
        p.update(self.params)
        # VOP OUTER of decay one (CAPS2 bit23) as the target builds it on a VPU built without
        # OUTER (the params' OUTER, the target's): the VPU's options above test the environment's
        # board_config's OUTER (the YPCB's, on), so x2:b2's cards (X2SimTransport: OUTER 0, the
        # target's OUTER1) were built without it and refused b2's programs (check_fits)
        if rp.get("VPU_OUTER1") and not int(p.get("OUTER", 1)):
            p["VPU_OUTER1"] = 1
        # the posted W beats (NAT_XPOST) where they apply: the native model at WG > 1 (a board
        # model built at WG 1 builds without them, and so with column reuse: CAPS bit5)
        if p.get("NAT_XPOST") and (p.get("WG", 1) == 1 or p["MEM_NATIVE"] != 1):
            del p["NAT_XPOST"]
        # and the channels' half beats (NAT_RHB: with the posted W beats; NAT_WHB: with NAT_WPOST)
        if not p.get("NAT_XPOST"):
            p.pop("NAT_RHB", None)
        if not p.get("NAT_WPOST") or p["MEM_NATIVE"] != 1:
            p.pop("NAT_WHB", None)
        # the configuration registers (register map 4) as the target's build of this
        # configuration reports them (run_vivado.sh: opentpu.target build_config / arch_hash);
        # WG as built (tb_board's default 1 unless params give it, not the target's: a target
        # with WG 8, the WG targets', at the board model's MCOLS 4 is built at WG 1)
        # (tb_bcuwg: the DRAM its channels at CAW report, CFG_DRAM)
        from opentpu.target import arch_hash, build_config
        t = self.target
        p.setdefault("PLATFORM", t.platform_id)
        bc = build_config(
            t, MCOLS=p["MCOLS"], LANES=p["LANES"], ACT_ROWS=p["ACT_ROWS"],
            TMEM_WORDS=p["TMEM_WORDS"], IMEM_WORDS=p.get("IMEM_WORDS"),
            DSTEP=p.get("DSTEP", 1), SEQ_WC=p.get("SEQ_WC", 0),
            WG=p.get("WG", 1), CHASH=p.get("CHASH", 1), OUTER=p.get("OUTER", 1),
            DSTEP2=p.get("DSTEP2", 0), SE_R=p.get("SE_R", 0), KT=p.get("KT", 0),
            ICACHE=p.get("ICACHE", 0), ST16=p.get("ST16", 0), FSUM=p.get("VPU_FSUM", 0),
            SPLITQ=p.get("QUANT_SPLIT", 0), ST16S=p.get("ST16S", 0), QHL=p.get("QUANT_HL", 0),
            QSTHL=p.get("QUANT_QSTHL", 0), VSIG=p.get("VPU_SIG", 0), VMAC=p.get("VPU_MAC", 0),
            OUTER1=p.get("VPU_OUTER1", 0), QONCE=p.get("QONCE", 0),
            SPLMM=p.get("MXU_SPL", 0))
        if self.bcuwg:
            bc = replace(bc, DRAM_BYTES=self.nch << self.caw + 7)
        else:                       # tb_board: the DRAM its otpu_board reports (DRAM_LOG2, 4 GiB
            # unless given: a target of 16 GiB slices hashed its own and was refused,
            # check_hash) and no bank-half window (the DIMM pairs', otpu_mig_pair: tb_bcuwg's)
            bc = replace(bc, BHW=0, DRAM_BYTES=1 << int(p.get("DRAM_LOG2", 32)))
        p.setdefault("ARCH_HASH", "32'h%08x" % arch_hash(bc))
        if self.bcuwg:
            srcs, p = self._wrap(srcs, p)
        return top, srcs, p

    def flush(self) -> None:
        if not self.script:
            return
        from opentpu import rtlsim, simtmp     # (the simulator's: not on the card's path)
        top, srcs, p = self.sim_build()
        exe = rtlsim.build(top, srcs, p)
        own = (self.nch, self.beat) != (2, BEAT)   # the WG targets' channels: the native memory model
        with simtmp.tempdir("otpu_board_") as d:
            for c in range(len(self.ch)):
                self.ch[c].view("<u4").astype(">u4").tofile(d / f"ch{c}.bin")
                if self.caw is not None:            # (byte j of a beat its bits 8 j .. 8 j + 7)
                    (d / f"ch{c}_hi.txt").write_text("".join(
                        f"{b:x} {self.hi[c][b][::-1].tobytes().hex()}\n"
                        for b in sorted(self.hi[c])))
            (d / "host.txt").write_text("\n".join(self.script) + "\n")
            self.script, self.nreads = [], 0
            from opentpu.target import mem_model
            r = rtlsim.run_sim([str(exe), f"+dir={d}", f"+axi_stall={self.stall}",
                                f"+axi_seed={self.seed}", *self.plusargs,
                                *(mem_model(self.target)
                                  if own and not self.bcuwg else [])])
            out = self.out = r.stdout + r.stderr
            if "DONE" not in out:
                raise RuntimeError(f"board simulation failed:\n{out[-3000:]}")
            self.results = []
            for line in out.splitlines():
                if line.startswith("REG "):
                    _, a, v = line.split()
                    self.results.append(int(v, 16))
                    self.regs_seen[int(a, 16)] = int(v, 16)
                elif line.startswith("DONE"):
                    self.cycles += int(line.split("=")[1])
            for c in range(len(self.ch)):
                self.ch[c] = np.fromfile(d / f"ch{c}_out.bin", np.uint8)
                if self.caw is not None:
                    self.hi[c] = self._hi_parse((d / f"ch{c}_hi_out.txt").read_text())

    @staticmethod
    def _hi_parse(text: str) -> dict:
        """ch<c>_hi_out.txt's beats: "<beat> <its 32 bytes, byte 0 last>" (hex) a line; the
        all-zero ones left out (a beat not in the store reads 0)."""
        hi = {}
        for line in text.splitlines():
            if line.strip():
                b, v = line.split()
                v = np.frombuffer(bytes.fromhex(v.rjust(64, "0"))[::-1], np.uint8)
                if v.any():
                    hi[int(b, 16)] = v.copy()
        return hi


# ------------------------------------------------------------------------------ the board
def _lock(t) -> DeviceLock | None:
    """The device lock, shared by every Board on the same transport object (a transport is one
    open of the device); a second transport on the same device, in this process or another,
    raises runstate.DeviceBusy naming the holder's pid."""
    lk = getattr(t, "_otpu_lock", None)
    name = getattr(t, "devname", None)
    if lk is None and name:
        lk = DeviceLock(name)
        t._otpu_lock = lk
    return lk


def counter_units(t=None) -> tuple[int, int]:
    """The bytes a count of DRAM_RD and of DRAM_WR stands for on target `t`'s build (default the
    current one): a channel beat (beat_bytes; the DDR3 cards' 64), half of one where the channels
    move their read / write data in half beats (NAT_RHB / NAT_WHB: otpu_hbm_ch raises n_rvalid /
    n_wvalid each half, and the counters count those). No register says so: the target does."""
    from opentpu.target import current, load
    t = load(t) if isinstance(t, str) else (t or current())
    b = t.platform["beat_bytes"] if t.kind == "fpga" else BEAT
    return (b // 2 if t.uarch.get("NAT_RHB") else b, b // 2 if t.uarch.get("NAT_WHB") else b)


def rates(a: dict, b: dict, core_khz: int | None, beat: int = R.DRAM_BEAT) -> dict:
    """Two snapshots -> utilization of every cycle counter (delta / UPTIME delta), DRAM GB/s
    (beats x `beat` bytes, the channels' (Board.beat), over the device time, which needs
    CORE_KHZ) and instructions per cycle."""
    dt = b["UPTIME"] - a["UPTIME"]
    sec = dt / (core_khz * 1e3) if core_khz else None
    d = {k: b[k] - a[k] for k in R.COUNTERS if k in a and k in b}
    out = {"cycles": dt, "seconds": sec,
           "util": {k: d[k] / dt if dt else 0.0 for k in d
                    if k not in R.EVENTS and k != "UPTIME"},
           "dram_beats": d["DRAM_RD"] + d["DRAM_WR"],
           "ipc": d["INSTR"] / dt if dt else 0.0,
           "dram_rd_gbs": None, "dram_wr_gbs": None, "dram_gbs": None}
    if sec:
        out["dram_rd_gbs"] = d["DRAM_RD"] * beat / sec / 1e9
        out["dram_wr_gbs"] = d["DRAM_WR"] * beat / sec / 1e9
        out["dram_gbs"] = out["dram_beats"] * beat / sec / 1e9
    return out


class CardRunning(RuntimeError):
    """The card still runs a program another process started (Board.quiesce)."""


class NoGather(RuntimeError):
    """A program with GATHER for a bitstream whose collective unit runs BAR alone (CAPS2 bit0 =
    0: a one-slice build without GATHER, docs/infra_hunt.md 5)."""


def check_no_gather(progs) -> None:
    """Raise NoGather when a program (words, 8 per instruction: the opcode in word 0's low
    byte) has a GATHER: such a bitstream would only acknowledge it and raise ERROR."""
    from opentpu import isa as I
    for s, w in enumerate(progs):
        w = np.asarray(w, "<u4")
        ops = w[:len(w) // 8 * 8].reshape(-1, 8)[:, 0] & 0xFF
        hit = np.flatnonzero(ops == I.GATHER)
        if hit.size:
            raise NoGather(f"slice {s}'s program has GATHER (instruction {int(hit[0])}), and "
                           "this bitstream's collective unit runs BAR alone (CAPS2 bit0 = 0)")


class Board:
    """Logical-address access to the card's DRAM, program loading and runs, the counters and
    the trace buffer."""

    def __init__(self, transport=None, check: bool = True, lock: bool = True,
                 calibrate: bool = True):
        """check: the card must answer as openTPU; calibrate: then calibrate its DDR3 if the
        bitstream wants the host to (LiteDRAM, CAPS bit27: memcal.ensure; once per
        configuration, as the STATUS calibration bits say)."""
        self.t = transport or XdmaTransport()
        self.lock = _lock(self.t) if lock else None
        self._info = None
        self._trace = None              # (depth, keep_first) of a started traced run
        self._t_run = 0.0               # when the started run began (perf_counter)
        self.in_run = False             # started and not seen halted: writes in RUN_H2C calls
        self._ded0 = None               # STATUS ECC_DED as the run started (None: not read)
        self._ded_said = False
        self._pool = None           # the DMA worker thread (large reads / writes on the card)
        self._split = None          # the split threads (pipelined writes: gather)
        self._sets = None           # ((nch, bytes), the pipelined writes' channel buffers)
        self._piping = threading.Lock()     # (one pipelined write at a time: its buffers)
        self._prog = None           # ICACHE: the loaded program's bytes [a, b) (write refuses them)
        try:
            ident = self.t.reg_read(R_ID) if check or self.lock is not None else ID_OTPU
            if check and ident != ID_OTPU:
                raise RuntimeError(f"no openTPU on the card (ID register {ident:#x})")
            if self.lock is not None and ident == ID_OTPU:
                self.quiesce()          # (before the calibration: it moves the DDR3 timing)
                self.lock.cleanup.append(weakref.WeakMethod(self._at_exit))
            if check and calibrate:
                memcal.ensure(self.t, log=lambda m: print(m, file=sys.stderr))
        except BaseException:
            self.close()
            raise

    def sharing(self) -> "Board":
        """A Board on this one's transport, its information and no lock of its own: writes
        beside this one's (its own pipeline), BoardBackend's to another card's slices."""
        b = Board(self.t, check=False, lock=False, calibrate=False)
        b._info = self._info or self.info()
        return b

    def close(self) -> None:
        """Release the device lock (for every Board on this transport); a run of this Board's
        not seen halted is stopped first (stop)."""
        if self.in_run:
            self.abort(STOP_WAIT)
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        if self._split is not None:
            self._split.shutdown()
            self._split = None
        self._sets = None
        if self.lock is not None:
            self.lock.release()
            self.t._otpu_lock = None
        self.lock = None

    def _at_exit(self) -> None:
        """The lock is going (DeviceLock.release, at exit too: SystemExit, an uncaught
        KeyboardInterrupt): a run of this Board's in flight is stopped first."""
        if self.in_run:
            self.abort(STOP_WAIT)

    # ------------------------------------------------------------------ ending runs
    def quiesce(self, wait: float | None = None) -> None:
        """End what an earlier holder of the card left, before this process writes it: a process
        killed mid-run (SIGKILL, the OOM killer) leaves RUN set and its program storing into
        DRAM, which would overwrite a new image written meanwhile, silently. A run still going
        is waited for up to `wait` seconds (QUIESCE_WAIT), then refused (CardRunning); with
        OTPU_STOP_RUN=1 it is cut short at once. Then stop(): CTRL = 0, the adapter quiet. RUN
        clear (an idle card, a new configuration before its calibration), nothing is done."""
        st = self.t.reg_read(R_STATUS)
        if not st & ST_RUN:
            return
        if not st & ST_HALTED and os.environ.get("OTPU_STOP_RUN") != "1":
            wait = QUIESCE_WAIT if wait is None else wait
            print(f"otpu: the card is still running a program started before this process; "
                  f"waiting up to {wait:.0f} s for it to halt", file=sys.stderr, flush=True)
            try:
                self.t.poll(R_STATUS, ST_HALTED, ST_HALTED, wait)
            except TimeoutError:
                raise CardRunning(f"the card is still running a program started before this "
                                  f"process ({wait:.0f} s, not halted): its process ended "
                                  f"without stopping it. Wait for it, or cut it short with "
                                  f"OTPU_STOP_RUN=1 (CTRL = 0)") from None
        self.stop()

    def stop(self, wait: float = 0.0) -> bool:
        """End the card's run, if one goes: up to `wait` seconds for HALTED (a run that halts by
        itself leaves DRAM as its program does: a whole token), then CTRL = 0 (the core held in
        reset: a run still going is cut short) and STATUS QUIET (its writes landed), so nothing
        of the run lands after the host's next writes or in a program load. Nothing
        with RUN clear (the core is held in reset already: every CTRL = 0 of a run's is this
        one's), nor on a card this process no longer holds (LockLost: it may be another
        runner's). True when no run was cut short."""
        t = self.t
        self.in_run = False
        if getattr(t, "batched", False):        # (the board model: a script, no run left)
            t.reg_write(R_CTRL, 0)
            return True
        lk = self.lock
        if lk is not None and lk.holder is not None and lk.lost():
            return False
        st = t.reg_read(R_STATUS)
        if not st & ST_RUN:
            return True
        done = bool(st & ST_HALTED)
        if not done and wait > 0:
            try:
                t.poll(R_STATUS, ST_HALTED, ST_HALTED, wait)
                done = True
            except TimeoutError:
                pass
        t.reg_write(R_CTRL, 0)
        if not done:                # (reads in flight: bitstreams before fix-board only)
            time.sleep(CUT_SETTLE)
        try:
            t.poll(R_STATUS, QUIET, QUIET, QUIET_WAIT)
        except TimeoutError:
            raise RuntimeError(f"the memory adapter still has DRAM writes outstanding "
                               f"{QUIET_WAIT:g} s after CTRL = 0 (STATUS WR_IDLE clear): one "
                               "never got its response; reload the bitstream") from None
        return done

    def abort(self, wait: float = 0.0) -> None:
        """stop() on an error's way out: its own errors are dropped (the error being raised is
        the one to see)."""
        try:
            self.stop(wait)
        except Exception:               # noqa: BLE001
            pass

    def _own(self) -> None:
        """LockLost when the otpu-lock this process runs under has exited."""
        lk = self.lock
        if lk is not None and lk.holder is not None and lk.lost():
            raise LockLost(lk.name, lk.holder)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ identity and state
    def info(self) -> dict:
        """D / MCOLS / LANES, STATUS, calibration and, with register map 2: CAPS, CORE_KHZ,
        BUILD_ID, the die temperature and the DDR3 data rate (None where the bitstream lacks
        them); with map 4 the configuration registers ("cfg": regs.cfg; None before); with map 5
        CAPS2 ("caps2": regs.caps2, every capability present before)."""
        v, st, rm = self.t.reg_read_many([R.R_VERSION, R_STATUS, R.R_REGMAP])
        d = {"D": v >> 16, "MCOLS": (v >> 8) & 0xFF, "LANES": v & 0xFF, "status": st,
             "calib": [bool(st & ST_CALIB0), bool(st & ST_CALIB1)],
             "calibrated": bool(st & ST_CALIB0) and bool(st & ST_CALIB1),
             "running": bool(st & ST_RUN) and not st & ST_HALTED,
             "regmap": R.regmap(rm), "caps": None, "core_khz": None, "build_id": None,
             "temp_c": None, "ddr_mts": None, "act_rows": (v >> 8) & 0xFF, "cfg": None,
             "caps2": R.caps2(None)}
        if d["regmap"] >= 2:
            offs = [R.R_CAPS, R.R_CORE_KHZ, R.R_BUILD_ID, R.R_TEMP, R.R_DDR_MTS]
            if d["regmap"] >= R.CAPS2_SINCE:
                offs.append(R.R_CAPS2)
            cp, khz, bid, tp, mts, *c2 = self.t.reg_read_many(offs)
            if c2:
                d["caps2"] = R.caps2(c2[0])
            d.update(caps=R.caps(cp), core_khz=khz or None, build_id=bid,
                     temp_c=round(R.temp_c(tp), 2) if cp & R.CAP_TEMP and tp & R.TEMP_VALID
                     else None,
                     ddr_mts=(mts or None) if cp & R.CAP_DDR else None)
            if cp & R.CAP_ACT:
                d["act_rows"] = self.t.reg_read(R.R_ACT_ROWS)
        if d["regmap"] >= R.CFG_SINCE:
            d["cfg"] = R.cfg(self.t.reg_read_many(list(R.CFG_REGS)))
            if hasattr(self.t, "bases"):            # the DMA's channel addresses
                self.t.bases = channel_map(d)
            # the target it is a build of (check_target: the transport's want_target, a WG
            # variant by name, refused on another build)
            t = check_target(d, getattr(self.t, "want_target", None))
            if t is not None:
                d["target"] = t
            # the sensor: XADC, or UltraScale+'s SYSMON
            if t is not None and t.family != "xilinx7" and d["temp_c"] is not None:
                d["temp_c"] = round(R.temp_c(tp, t.family), 2)
        self._info = d
        return d

    @property
    def slices(self) -> int:
        """The bitstream's slices (CFG_SLICES; 1 before register map 4). The board model's
        transport knows them (a register read there is a simulation of its own: SimTransport)."""
        if self._info is None and getattr(self.t, "batched", False):
            return getattr(self.t, "slices", 1)
        c = (self._info or self.info())["cfg"]
        return c["S"] if c else 1

    def _base(self, ch: int) -> int:
        """Channel ch's card address (where the DMA buffers are placed for: DMA_PLACE)."""
        return getattr(self.t, "bases", BASE)[ch]

    @property
    def waitc(self) -> bool:
        """The bitstream's sequencer takes wait entries (CFG_UARCH bit24, SEQ_WC, EXPERIMENTAL:
        docs/seq_vpu.md): every program loads with its entries (load_program). Its window must
        be 16 or less, or 32 (the entries' counts name a unit's last 15, saturating at 32:
        Config.WC_WIN, device_wc_win): a target of this tree with another is refused
        (ConfigMismatch)."""
        i = self._info or self.info()
        c = i["cfg"]
        if not (c and c.get("SEQ_WC")):
            return False
        if getattr(self, "_waitc_of", None) is not i:    # (once per info())
            from opentpu.target import by_platform
            t = by_platform(c["platform"])
            if t is not None and 16 < t.uarch["WIN"] != 32:
                raise ConfigMismatch(f"the bitstream takes wait entries (CFG_UARCH bit24) and its "
                                     f"target {t.name} has a window of {t.uarch['WIN']}: the "
                                     f"entries need 16 or less, or 32 (docs/isa.md, Wait counts)")
            self._waitc_of = i
        return True

    @property
    def chash(self) -> bool:
        """The bitstream hashes the channel interleave (CAPS bit7)."""
        c = (self._info or self.info())["caps"]
        return bool(c and c.get("chash"))

    @property
    def nch(self) -> int:
        """A slice's memory channels (CFG_DRAM; 2 before register map 4). The board model's
        transport knows them (as `slices`)."""
        if self._info is None and getattr(self.t, "batched", False):
            return getattr(self.t, "nch", 2)
        c = (self._info or self.info())["cfg"]
        return c["channels"] if c else 2

    @property
    def beat(self) -> int:
        """The channels' beat bytes (CFG_DRAM [31:24]; 64 before register map 4)."""
        if self._info is None and getattr(self.t, "batched", False):
            return getattr(self.t, "beat", BEAT)
        c = (self._info or self.info())["cfg"]
        return c.get("beat_bytes", BEAT) if c else BEAT

    def beat_at(self, addr: int, s: int = 0) -> tuple[int, int]:
        """(card channel, channel offset) of slice s's logical byte `addr` (the bitstream's map)."""
        c, off = beat_at(addr, self.chash, self.nch, self.beat)
        return self.nch * s + c, off

    @property
    def gather(self) -> bool:
        """The collective unit runs GATHER (CAPS2 bit0; every bitstream before register map 5).
        The board model's transport knows it (as `slices`)."""
        if self._info is None and getattr(self.t, "batched", False):
            return getattr(self.t, "gather", True)
        return (self._info or self.info())["caps2"]["gather"]

    @property
    def icache(self) -> bool:
        """The bitstream's IMEM is an instruction cache over the program in DRAM (CAPS2 bit12,
        otpu_seq ICACHE, docs/icache.md; clear on every bitstream before it): load_program writes
        the program to DRAM alone and LOAD copies nothing (the slices fetch its lines as they run
        them, from PROG_ADDR), so the program stays there, unchanged, while it runs; CFG_IMEM is
        the longest program. The board model's transport knows it."""
        if self._info is None and getattr(self.t, "batched", False):
            return getattr(self.t, "icache", False)
        return (self._info or self.info())["caps2"]["icache"]

    @property
    def v2(self) -> bool:
        return (self._info or self.info())["regmap"] >= 2

    @staticmethod
    def snap_offs(regmap: int) -> list[int]:
        """The registers a snapshot reads: the map's counter shadows (low, high), then SNAP."""
        return [o + k for o in R.counters(regmap).values() for k in (0, 4)] + [R.R_SNAP]

    def snapshot(self) -> dict | None:
        """SNAP, then the free-running counters' shadows (one consistent instant): {name: count}
        plus "snaps" (SNAP's read value). None on a register map 1 bitstream."""
        if not self.v2:
            return None
        rm = (self._info or self.info())["regmap"]
        # SNAP is the card's, not this process's: another one's (otpu-smi -l) between these
        # reads latches the shadows again, mixing two instants (or a counter's two halves).
        # SNAP's count read before the shadows and after them: equal, they are one latch's.
        offs = [R.R_SNAP] + self.snap_offs(rm)
        for _ in range(SNAP_TRIES):
            self.t.reg_write(R.R_SNAP, 1)
            v = self.t.reg_read_many(offs)
            if v[0] == v[-1]:
                break
        return self.snap_dict(v[1:], rm)

    @staticmethod
    def snap_dict(v: list[int], regmap: int) -> dict:
        """Values read at snap_offs(regmap) -> {counter: 64-bit count, "snaps": n}."""
        d = {n: v[2 * i] | v[2 * i + 1] << 32 for i, n in enumerate(R.counters(regmap))}
        d["snaps"] = v[-1]
        return d

    # ------------------------------------------------------------------ DRAM
    def write(self, addr: int, data, s: int = 0) -> None:
        """Logical bytes to the card (slice s's DRAM). Only the beats they fall in are written: the
        bytes off the wide-chunk grid go beat by beat to their channels (a partial beat's other
        bytes kept by the transport: XdmaTransport.mem_write reads that beat alone), never the
        rest of their chunks, which the card may store meanwhile (a generate run's state words
        beside the host's stop word: widened to its chunk, the stop write put back the words it
        had read before the card's store). On an ICACHE bitstream the loaded program and its
        wait entries stay in DRAM while it runs and between runs, the cache fetching their lines
        from there (load_program): a write over them raises ValueError (one compare against their
        range; nothing on other bitstreams)."""
        self._own()
        data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        if len(data) == 0:
            return
        end = addr + len(data)
        self._not_prog(addr, end)
        P = self.nch * self.beat                        # a wide chunk: a beat on every channel
        a0 = -(-addr // P) * P                          # the whole chunks inside
        a1 = end // P * P
        if a1 <= a0:
            a0 = a1 = end                               # (none: beats only)
        self._beats(addr, data[:a0 - addr], s)
        if a1 > a0:
            self._chunks(a0, data[a0 - addr:a1 - addr], s)
        self._beats(a1, data[a1 - addr:], s)

    def write_chunks(self, pieces, s: int = 0) -> None:
        """Whole wide chunks to slice s: `pieces` (addr, bytes), each address and length a
        multiple of the wide chunk (nch x beat), written in turn as write() writes them, but on a
        threaded transport through one pipeline (_piped: no fill and drain between pieces, short
        ones split there too). BoardBackend's image ranges (keep_ranges, image_pieces)."""
        self._own()
        P = self.nch * self.beat
        segs = []
        for addr, data in pieces:
            data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
            if addr % P or len(data) % P:
                raise ValueError(f"write_chunks: [{addr:#x}, +{len(data):#x}) is not whole wide "
                                 f"chunks of {P} bytes")
            if len(data):
                self._not_prog(addr, addr + len(data))
                segs.append((addr, data))
        if self.in_run and getattr(self.t, "run_h2c", 0) or not getattr(self.t, "threaded",
                                                                         False):
            for addr, data in segs:
                self._chunks(addr, data, s)
            return
        self._piped(segs, s)

    def _not_prog(self, addr: int, end: int) -> None:
        if self._prog is not None and addr < self._prog[1] and end > self._prog[0]:
            raise ValueError(f"a write of [{addr:#x}, {end:#x}) over the loaded program at "
                             f"[{self._prog[0]:#x}, {self._prog[1]:#x}) (ICACHE: the slices "
                             "fetch it from DRAM)")

    def _beats(self, addr: int, data: np.ndarray, s: int = 0) -> None:
        """Bytes of at most a chunk's worth (a write's edges), beat by beat to their channels."""
        x, end, beat = addr, addr + len(data), self.beat
        while x < end:
            k = min(end, (x // beat + 1) * beat) - x
            self.t.mem_write(*self.beat_at(x, s), data[x - addr:x - addr + k])
            x += k

    def _chunks(self, a0: int, data: np.ndarray, s: int = 0) -> None:
        """Whole wide chunks from a0 (aligned) to slice s's channels. CHASH's channel order in a
        copy when not pipelined; pipelined (_piped), gathered piece by piece straight into the
        channels' buffers (an image's whole copy, and hash_swap's gather beside it, took three
        times its span in host memory: GPT-OSS's 15.8 GiB on the BCU1525)."""
        nch, beat = self.nch, self.beat
        j = nch * s                                     # its first channel
        lim = getattr(self.t, "run_h2c", 0) if self.in_run else 0
        piped = not lim and getattr(self.t, "threaded", False) and len(data) > PIPE
        if piped:
            self._piped([(a0, data)], s)
            return
        if self.chash:                                  # channel order (a copy)
            data = data.copy()
            hash_swap(a0, data.reshape(-1, nch, beat))
        if lim:                                         # a run is in flight (RUN_H2C)
            for c, off, part in split(a0, data, nch=nch, beat=beat):
                for i in range(0, len(part), lim):
                    self.t.mem_write(j + c, off + i, part[i:i + lim])
            return
        for c, off, part in split(a0, data, nch=nch, beat=beat):
            self.t.mem_write(j + c, off, part)

    def _piped(self, segs: list, s: int = 0) -> None:
        """Whole wide chunks to slice s's channels, piece by piece: segs [(a0, bytes)], each
        aligned. A piece is PIPE / 2 bytes a channel (one DMA call) of a segment; the split
        threads gather its beats straight into its set of channel buffers (gather: CHASH's order
        with no copy before it; the threads' parts of the piece side by side) while the DMA
        worker writes the pieces before it from the other sets (PIPE_SETS, placed for full DMA
        speed: DMA_PLACE). Each DMA task waits for its own piece's split: the main thread queues
        pieces ahead, and waits only for a set whose last piece is still being written."""
        nch, beat, chash = self.nch, self.beat, self.chash
        j, P, h = nch * s, nch * beat, PIPE // 2
        bl = _gather_tables(nch, chash)[0]
        with self._piping:
            sets = self._piece_sets(nch, h)
            pool, ex = self._splitter(), self._worker()
            busy: list = [None] * len(sets)             # each set's DMA in flight
            split_f: list = []                          # the last piece's split (not yet queued)
            k, ok = 0, False

            def put(fs, parts):
                concurrent.futures.wait(fs)
                for f in fs:
                    f.result()
                for p in parts:
                    self.t.mem_write(*p)
            try:
                for a0, data in segs:
                    for pos in range(0, len(data), nch * h):
                        v = data[pos:pos + nch * h]
                        n, i = len(v) // P, k % len(sets)   # its chunks; its set
                        k += 1
                        if busy[i] is not None:
                            busy[i].result()
                            busy[i] = None
                        off = (a0 + pos) // nch             # (every channel's offset)
                        bufs = [_placed_in(sets[i][c], self._base(j + c) + off, n * beat)
                                for c in range(nch)]
                        rows, outs = v.reshape(n * nch, beat), [b.reshape(n, beat) for b in bufs]
                        step = -(-n // (SPLIT_THREADS * bl)) * bl   # (whole blocks a thread)
                        split_f = [pool.submit(gather, a0 + pos, rows, outs, chash, nch, beat,
                                               m, min(n, m + step)) for m in range(0, n, step)]
                        busy[i] = ex.submit(put, split_f, [(j + c, off, bufs[c])
                                                           for c in range(nch)])
                        split_f = []
                ok = True
            finally:                                    # (nothing left using the buffers)
                concurrent.futures.wait([f for f in busy if f is not None] + split_f)
                if ok:
                    for f in busy:
                        if f is not None:
                            f.result()

    def _piece_sets(self, nch: int, h: int) -> list:
        """_piped's PIPE_SETS sets of nch channel buffers of h bytes (and a page, to place them:
        _placed_in), kept for the Board's later writes (an image's ranges, one call each)."""
        if self._sets is None or self._sets[0] != (nch, h):
            self._sets = ((nch, h), [[np.frombuffer(mmap.mmap(-1, h + 4096), np.uint8)
                                      for _ in range(nch)] for _ in range(PIPE_SETS)])
        return self._sets[1]

    def _worker(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="otpu-dma")
        return self._pool

    def _splitter(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._split is None:
            self._split = concurrent.futures.ThreadPoolExecutor(SPLIT_THREADS,
                                                                thread_name_prefix="otpu-split")
        return self._split

    def scrub(self, force: bool = False, keep=None) -> bool:
        """Write the whole DRAM once per configuration. The DDR3 controllers run with ECC, and
        after power-up the check bits are random: a read of a 64-byte beat never written since is
        an uncorrectable error, which hangs the read (a DMA timeout, or a stuck accelerator) instead
        of failing it. Zeros everywhere (~6 s at Gen1 x8), then SCRATCH = DRAM_INIT, which a new
        bitstream resets. Only the card: the simulated boards have no ECC. True when it scrubbed.
        The first scrub of a configuration whose core clock is not its controllers' also centres
        the read taps under the core's load (loadcal.center, ~40 s), between two passes of
        zeros.

        keep: for slice s at keep[s], logical ranges [lo, hi) (whole wide chunks) the caller
        writes right after a scrub that ran, every beat of them (BoardBackend: an image's data,
        keep_ranges): no zeros there, so each beat is still written once. With the centring the
        first pass is whole (the load's reads may be of any beat) and the second leaves them."""
        if not getattr(self.t, "ecc", False):
            return False
        first = self.t.reg_read(R.R_SCRATCH) != R.DRAM_INIT
        if not force and not first:
            return False
        c = (self._info or self.info())["cfg"]  # every slice's channels, what a slice addresses
        S, k, beat = (c["S"], c["channels"], c.get("beat_bytes", BEAT)) if c else (1, 2, BEAT)
        nch, size = S * k, c["channel_bytes"] if c else CH_BYTES
        if keep is not None and len(keep) > S:
            raise ValueError(f"scrub: ranges kept for {len(keep)} slices, the card has {S}")
        skip = [[] for _ in range(nch)]     # each channel's offsets left out
        for s, rs in enumerate(keep or ()):
            for lo, hi in rs:
                if lo % (k * beat) or hi % (k * beat) or not 0 <= lo <= hi <= k * size:
                    raise ValueError(f"scrub: a kept range [{lo:#x}, {hi:#x}) is not whole "
                                     f"wide chunks of slice {s}'s DRAM")
                for ch in range(k * s, k * s + k):
                    skip[ch].append((lo // k, hi // k))
        z = placed(64 << 20, 0)             # zeros, placed: no bounce copies

        def zeros(skip):
            def some(chs):
                for ch in chs:
                    x = 0
                    for lo, hi in sorted(skip[ch]) + [(size, size)]:
                        for off in range(x, lo, len(z)):
                            self.t.mem_write(ch, off, z[:min(len(z), lo - off)])
                        x = max(x, hi)
            # (cards in tensor parallel: each card's channels on a thread of its own)
            _side_by_side(some, [(g,) for g in _by_card(self.t, range(nch),
                                                        k * getattr(self.t, "spc", S))])
        from opentpu.host import ddr4, loadcal, memcal
        cal = first and loadcal.wanted_on(self)
        zeros([[]] * nch if cal else skip)
        if cal:
            loadcal.center(self, log=lambda m: print(m, file=sys.stderr))
            zeros(skip)                     # the BIST's region and the load's, with check bits
        # the ECC counters (STATUS ECC_DED) from here on: a read of a beat not yet written (an
        # earlier run's, a tool's) counted its random check bits as uncorrectable
        memcal.ecc(self.t, clear=True)
        ddr4.ecc(self.t, clear=True)
        self.t.reg_write(R.R_SCRATCH, R.DRAM_INIT)
        return True

    def read(self, addr: int, n: int, s: int = 0) -> np.ndarray:
        """n logical bytes from slice s's DRAM at addr."""
        nch, beat = self.nch, self.beat
        j = nch * s                                     # its first channel
        a0 = addr // (nch * beat) * (nch * beat)
        a1 = -(-(addr + n) // (nch * beat)) * (nch * beat)
        half = (a1 - a0) // nch                         # bytes a channel
        out = np.empty((half // beat, nch, beat), np.uint8)
        if not (getattr(self.t, "threaded", False) and nch * half > PIPE):
            for c in range(nch):                        # each channel's run, interleaved
                out[:, c, :] = self.t.mem_read(j + c, a0 // nch, half).reshape(-1, beat)
        else:
            # pieces of PIPE / nch bytes per channel: the worker reads piece i + 1 into one set
            # of buffers while this thread interleaves piece i from the other set
            h = PIPE // nch
            slots = [[placed(h, self._base(j + c) + a0 // nch) for c in range(nch)]
                     for _ in range(2)]

            def get(i: int):
                k = min(h, half - i * h)
                return [self.t.mem_read(j + c, a0 // nch + i * h, k, slots[i % 2][c][:k])
                        for c in range(nch)]

            npieces = -(-half // h)
            ex = self._worker()
            fut = ex.submit(get, 0)
            for i in range(npieces):
                nxt = ex.submit(get, i + 1) if i + 1 < npieces else None
                rows = slice(i * h // beat, (i * h + h) // beat)
                for c, part in enumerate(fut.result()):
                    out[rows, c, :] = part.reshape(-1, beat)
                fut = nxt
        if self.chash:                                  # channel order -> logical
            hash_swap(a0, out)
        flat = out.reshape(-1)
        return flat if (a0, a1) == (addr, addr + n) else flat[addr - a0:addr - a0 + n].copy()

    # ------------------------------------------------------------------ programs
    def cross_card_entries(self, ents) -> np.ndarray:
        """A program's wait entries as a machine of several cards loads them: FOLLOW cleared
        (every load: load_program's, and a chain slot's, BoardBackend.slot_words, which HALT
        CHAIN loads). Phase 1's cross-card hub advances FOLLOW's contiguous top on local writes
        only. The compiler's single-card pass can follow a reduction of the whole gathered row:
        a later local piece would then expose unwritten remote data. Keep k_4's ordinary
        completion dependency and clear FOLLOW. This is conservative even for local-only
        reductions; remote per-piece validity is needed before unrestricted compiler FOLLOW is
        safe at NC > 1. One card: the entries unchanged."""
        ents = np.asarray(ents, np.uint64)
        if getattr(self.t, "ncards", 1) <= 1:
            return ents
        from opentpu import waits as W
        return ents & ~np.uint64(W.FOLLOW)

    def load_program(self, addr: int, words, entries=None) -> None:
        """Copy a program into DRAM at `addr` (chunk aligned) and into IMEM (an ICACHE
        bitstream's, `icache`: DRAM alone, LOAD sets the slices' program base and drops the
        cache's lines; the program stays there while it runs). A run started and not seen halted
        (a wait that raised) is stopped first: its reads in flight would land in IMEM as the new
        program's rows. On a multi-slice bitstream `words` may be a list,
        slice s's program at its index, each in its own DRAM at `addr` (the slices load the same
        PROG_ADDR and PROG_N: the shorter ones padded with NOPs); slices it names no program
        for get a HALT.

        entries: the programs' wait entries (opentpu.waits.entries; a list like `words`), which
        a bitstream whose sequencer takes them (waitc: CFG_UARCH bit24) needs: it loads them
        after the program (waits.loadable), and a program without them is refused
        (ConfigMismatch): its sequencer would run it with no waits. Other bitstreams load the
        words alone (`entries` unused). load() computes them."""
        if not 0 <= addr < 1 << 32:             # (PROG_ADDR: 32 bits, reg_write would wrap it)
            raise ValueError(f"a program at {addr:#x}: PROG_ADDR is a byte address in the "
                             "slice's first 4 GiB")
        progs = [np.asarray(w, "<u4") for w in words] if isinstance(words, (list, tuple)) \
            else [np.asarray(words, "<u4")]
        if not self.gather:
            check_no_gather(progs)
        waitc = self.waitc
        if waitc:
            if entries is None:
                raise ConfigMismatch("this bitstream's sequencer takes wait entries (CFG_UARCH "
                                     "bit24, docs/seq_vpu.md): a program loads with them "
                                     "(Board.load, opentpu.waits.entries)")
            ents = [np.asarray(e, np.uint64) for e in entries] if \
                isinstance(entries, (list, tuple)) else [np.asarray(entries, np.uint64)]
            if len(ents) != len(progs) or any(len(e) != len(w) // 8
                                              for e, w in zip(ents, progs)):
                raise ValueError("the wait entries do not match the programs")
            ents = [self.cross_card_entries(e) for e in ents]
        S = self.slices
        if len(progs) > S:
            raise ValueError(f"{len(progs)} programs for a bitstream of {S} slices")
        if len(progs) < S:
            from opentpu import isa as I
            progs += [np.asarray(I.assemble([I.halt()]), "<u4")] * (S - len(progs))
            if waitc:
                from opentpu import waits as W
                ents += [np.full(1, W.DEFAULT, np.uint64)] * (S - len(ents))
        n = max(len(w) for w in progs)
        if self.in_run:
            self.stop()
        if hasattr(self.t, "collective_program"):
            self.t.collective_program(progs)
        self._prog = None
        nb = 0
        for s, w in enumerate(progs):
            if len(w) < n:
                w = np.concatenate([w, np.zeros(n - len(w), "<u4")])
            if waitc:                               # its entries after it (the NOPs': none)
                from opentpu import waits as W
                e = ents[s]
                w = W.loadable(w, np.concatenate([e, np.zeros(n // 8 - len(e), np.uint64)]),
                               (self._info or self.info())["D"])
            self.write(addr, w.view(np.uint8), s)
            nb = max(nb, w.nbytes)
        if self.icache:                             # (the cache reads them for as long as loaded)
            self._prog = (addr, addr + nb)
        t = self.t
        t.reg_write(R_CTRL, 0)
        t.reg_write(R_PROG_ADDR, addr)
        t.reg_write(R_PROG_N, n // 8)
        t.reg_write(R_CTRL, CTRL_LOAD)
        t.poll(R_STATUS, ST_LOADING, 0)

    def release_program(self) -> None:
        """Forget the loaded program (ICACHE: write refuses its bytes while loaded), so a new
        image may cover them; a run not seen halted is stopped first. The cores must not run it
        again before the next load_program."""
        if self.in_run:
            self.stop()
        self._prog = None

    def load(self, addr: int, progs) -> None:
        """load_program of programs as instructions (a list of Instr or compiler.Program; a list
        of them, one per slice), assembled here, with their wait entries when the bitstream
        takes them (waits.entries: kept per program)."""
        from opentpu import isa as I
        multi = bool(progs) and isinstance(progs[0], (list, tuple))
        ps = list(progs) if multi else [progs]
        words = [np.asarray(I.assemble(p), np.uint32) for p in ps]
        entries = None
        if self.waitc:
            from opentpu import waits as W
            info = self._info or self.info()
            D, S, aw = info["D"], self.slices, device_aw(info)  # (the card's address bits)
            win = device_wc_win(info, aw)                       # (its entries' window)
            entries = [W.entries(w, D, S, prog=p, aw=aw, win=win) for w, p in zip(words, ps)]
        self.load_program(addr, words if multi else words[0],
                          entries if multi or entries is None else entries[0])

    def set_args(self, words) -> None:
        """The next runs' arguments ARG0..7 (CAPS bit25; R8..R15 at the start): only the words
        that differ from the last ones written (8 at most)."""
        last = getattr(self, "_args", None) if getattr(self.t, "keeps_state", True) else None
        last = last or [None] * 8
        for k, w in enumerate(words):
            w = int(w) & 0xFFFFFFFF
            if last[k] != w:
                self.t.reg_write(R.R_ARG0 + 4 * k, w)
        self._args = [int(w) & 0xFFFFFFFF for w in words]

    RUN_OFFS = [R_STATUS, R_CYCLES, R_CYCLES_HI, R_ICOUNT, R.R_B_RD, R.R_B_WR, R.R_A_RD,
                R.R_A_WR, R.R_B_STALL]

    def run(self, timeout: float = RUN_TIMEOUT, trace: dict | None = None,
            expect: float = 0.0) -> dict:
        """Run the loaded program until it halts; returns the counters.

        trace={"keep": "first" | "last"} records the run in the trace buffer (register map 2
        with CAPS.trace): STOP_WHEN_FULL keeps the first DEPTH records, the ring the last. It
        adds stats["trace"]: records (uint64, oldest first), count (TRACE_COUNT), drop
        (TRACE_DROP: events the capture queue lost), depth, keep, wrapped (the ring overwrote
        records), lost (records written but not in the buffer: count - depth, or 0) and
        saturated (TRACE_COUNT's high bits saturated past 2^32 - depth records: lost is more).
        expect: the run's expected wall time in seconds, a hint for the poll (Transport.poll).

        start() and wait() are the two halves: the host can work between them while the card
        runs (the transport must not be used in between)."""
        self.start(trace)
        return self.wait(timeout, expect)

    def start(self, trace: dict | None = None) -> None:
        """Start the loaded program (run's first half)."""
        self._own()
        t = self.t
        self._trace = None
        if trace is not None:
            i = self._info or self.info()
            if i["regmap"] < 2 or not i["caps"]["trace"]:
                raise RuntimeError("this bitstream has no trace buffer (register map "
                                   f"{i['regmap']}{'' if i['regmap'] < 2 else ', CAPS.trace = 0'})")
            keep_first = trace.get("keep", "first") == "first"
            self._trace = (i["caps"]["trace_depth"], keep_first)
            t.reg_write(R.R_TRACE_CTRL, R.TR_CLEAR)
            t.reg_write(R.R_TRACE_CTRL, R.TR_ENABLE | (R.TR_STOP_WHEN_FULL if keep_first else 0))
        t.reg_write(R_CTRL, CTRL_CLEAR)
        self.in_run = True
        t.reg_write(R_CTRL, CTRL_RUN)
        self._t_run = time.perf_counter()
        self._ded0 = None
        if not getattr(t, "batched", False):    # (the board model: a script, read at its end)
            # CTRL takes RUN (and LOAD) only while both channels are calibrated: else RUN
            # stays 0 and a wait would poll HALTED until its timeout
            st = t.reg_read(R_STATUS)
            if not st & ST_RUN:
                self.in_run = False
                low = [ch for ch, bit in enumerate((ST_CALIB0, ST_CALIB1)) if not st & bit]
                raise RuntimeError(
                    f"the card did not start the run (STATUS {st:#x}, RUN clear): DDR3 channel "
                    f"{' and '.join(map(str, low))} not calibrated, and CTRL takes neither RUN "
                    "nor LOAD before both are (memcal: otpu-memcal cal; or reload the bitstream)"
                    if low else f"the card did not start the run (STATUS {st:#x}, RUN clear)")
            self._ded0 = bool(st & R.ST_ECC_DED)

    def wait(self, timeout: float = RUN_TIMEOUT, expect: float = 0.0) -> dict:
        """Wait for the started program to halt; returns the counters (run's second half).
        expect counts from the start (the host's work in between is taken off). A run that does
        not halt in `timeout` seconds is cut short (stop) before the TimeoutError; one this
        wait leaves another way (KeyboardInterrupt, SystemExit) gets STOP_WAIT to halt first."""
        t = self.t
        if expect:
            expect = max(expect - (time.perf_counter() - self._t_run), 1e-9)
        traced = self._trace is not None
        depth, keep_first = self._trace or (0, True)
        try:
            st0 = t.poll(R_STATUS, ST_HALTED, ST_HALTED, timeout, expect)
        except TimeoutError:
            self.abort()
            raise
        except BaseException:
            self.abort(STOP_WAIT)
            raise
        self.t_seen = time.perf_counter()           # HALTED seen (BoardBackend's run time)
        self.seen_exact = getattr(t, "poll_exact", False)   # else an upper bound
        # HALTED rises once the last store has left the core, not when the DRAM has it
        # (WR_IDLE: the memory adapter's writes all answered); the host reads the results
        # next, and a read may pass writes in flight: wait for WR_IDLE, read back to back for
        # WR_SETTLE, then polled up to QUIET_WAIT; never seen, the run's results are not all
        # in DRAM: an error (the board model replays its register script in order: no such race)
        if not getattr(t, "batched", False) and not st0 & R.ST_WR_IDLE:
            while not st0 & R.ST_WR_IDLE and time.perf_counter() - self.t_seen < WR_SETTLE:
                st0 = t.reg_read(R_STATUS)
            if not st0 & R.ST_WR_IDLE:
                try:
                    st0 = t.poll(R_STATUS, R.ST_WR_IDLE, R.ST_WR_IDLE, QUIET_WAIT)
                except TimeoutError:
                    self.abort()
                    raise RuntimeError(f"the run halted, but its DRAM writes had not all landed "
                                       f"{QUIET_WAIT:g} s later (STATUS WR_IDLE clear): its "
                                       "results are not all there; reload the bitstream") \
                        from None
        self.in_run = False
        if traced:                      # the last cycles' events still drain into the buffer
            t.poll(R.R_TRACE_CTRL, R.TR_BUSY, 0, timeout)
        S = self.slices
        ics = [R.R_ICOUNT_S + 4 * s for s in range(1, S)]  # (slice 0's: ICOUNT)
        offs = self.RUN_OFFS + ics + ([R.R_TRACE_COUNT, R.R_TRACE_DROP] if traced else [])
        raw = None
        if traced and getattr(t, "batched", False):
            # the board model: one simulation, so the whole buffer is read after the counters
            # (the count is not known before the flush)
            idx = [t.queue_read(o) for o in offs]
            t.reg_write(R.R_TRACE_ADDR, 0)
            ridx = [t.queue_read(o) for _ in range(depth) for o in (R.R_TRACE_LO, R.R_TRACE_HI)]
            t.flush()
            vals = [t.results[k] for k in idx]
            w = np.array([t.results[k] for k in ridx], np.uint64)
            raw = w[0::2] | w[1::2] << np.uint64(32)
        else:
            vals = t.reg_read_many(offs)
        st, lo, hi, ic, brd, bwr, ard, awr, bst = vals[:9]
        ic = [ic] + vals[9:9 + len(ics)]
        vals = vals[:9] + vals[9 + len(ics):]
        stats = {"cycles": lo | hi << 32, "instructions": ic, "b_reads": brd,
                 "b_writes": bwr, "a_reads": ard, "a_writes": awr, "b_stall": bst,
                 "status": st}
        if traced:
            stats["trace"] = self._trace_out(vals[9], vals[10], depth, keep_first, raw)
            t.reg_write(R.R_TRACE_CTRL, 0)
        t.reg_write(R_CTRL, 0)
        if st & ST_ERROR:
            raise RuntimeError("a WAITW timed out (STATUS WAIT_TO)" if st & R.ST_WAIT_TO else
                               "the program stopped on an illegal instruction")
        if st & ST_AXI_ERR:
            raise RuntimeError("the memory path reported an error (STATUS AXI_ERR: a DDR3 controller "
                               "broke its port contract; on MIG bitstreams, an AXI error response)")
        # ECC_DED is sticky (until the ECC counters' clear: Board.scrub): set during this run, a
        # word it read was wrong; set before it, the run is not checked (said once a Board)
        stats["ecc_ded"] = bool(st & R.ST_ECC_DED)
        if stats["ecc_ded"] and self._ded0 is False:
            raise RuntimeError("an uncorrectable ECC error during the run (STATUS ECC_DED): a "
                               "word it read from DRAM was wrong, and so may be its results "
                               "(memcal.ecc, on a DDR4 card ddr4.ecc, has the counts; "
                               "Board.scrub clears them)")
        if stats["ecc_ded"] and self._ded0 and not self._ded_said:
            self._ded_said = True
            print("otpu: STATUS ECC_DED was set before this run (an uncorrectable ECC error "
                  "since the counters' last clear): runs are not checked for one until Board.scrub "
                  "clears it", file=sys.stderr, flush=True)
        return stats

    def _trace_out(self, count: int, drop: int, depth: int, keep_first: bool,
                   raw: np.ndarray | None) -> dict:
        n = min(count, depth)
        wrapped = count > depth and not keep_first
        start = count % depth if wrapped else 0          # the oldest record in the ring
        if raw is None:
            raw = self.read_trace(start, n, depth)
        else:
            raw = np.concatenate([raw[start:], raw[:start]]) if wrapped else raw[:n]
        # TRACE_COUNT saturates in its high bits (its low log2(depth) bits, the ring's write
        # slot, keep counting): there, lost is a lower bound
        return {"records": raw, "count": count, "drop": drop, "depth": depth,
                "keep": "first" if keep_first else "last", "wrapped": wrapped,
                "lost": max(0, count - depth), "saturated": count >= (1 << 32) - depth}

    def read_trace(self, start: int, n: int, depth: int) -> np.ndarray:
        """n records from ring index `start` on (TRACE_HI reads step TRACE_ADDR; the address is
        set again where the ring wraps)."""
        out = np.empty(n, np.uint64)
        pos = 0
        while pos < n:
            a = (start + pos) % depth
            k = min(n - pos, depth - a)
            self.t.reg_write(R.R_TRACE_ADDR, a)
            v = np.array(self.t.reg_read_many([R.R_TRACE_LO, R.R_TRACE_HI] * k), np.uint64)
            out[pos:pos + k] = v[0::2] | v[1::2] << np.uint64(32)
            pos += k
        return out


# ------------------------------------------------------------------------------ configuration
class ConfigMismatch(RuntimeError):
    """The bitstream on the card and the host's configuration disagree."""


def device_config(info: dict, **kw):
    """The Config of the bitstream that `info` (Board.info()) describes: MCOLS and LANES come
    from its VERSION register, PAIR from CAPS bit5, DSTEP from bit6 and STREAM from bit26, and
    with register map 4 the slices, TMEM, IMEM, ACT RAM and DRAM sizes from its CFG_* registers
    and SEQ_WC (the sequencer's wait entries, EXPERIMENTAL) from CFG_UARCH bit24, WG from its
    bits 27:25 and at WG > 1 CHASH (the group chunks' row order) from CAPS bit7 and GALL
    (every MM in groups: otpu_mxu at WG > 1) (before map 4: the YPCB-00338's, the current
    target's), OUTER from CAPS2 bit1 (set: a VPU without VOP OUTER; clear on every bitstream
    before it), DSTEP2 from its bit2 (DSTEP's pair), SE_OWN from its bit3 (the streams on an
    engine of their own), KT from its bit5 (MM's K tail), BHW from its bits 11:6 (the memory's
    bank-half window: the images' layout), ST16 from its bits 14:13 (DSTEP's state in 16
    bits, its format), FSUM from its bit16 (EXP2SUB's fused row sums), SPLITQ from its bit17
    (QACT / QST flag LO), ST16S from its bit18 (STREAM's ST16R / ST16W), VSIG from its bit19
    (VOP SIGMOID / SILU), VMAC from its bit20 (VOP MAC), QHL from its bit21 (QACT flag HL),
    QSTHL from its bit22 (QST HL), OUTER1 from its bit23 (VOP OUTER decay one), QONCE from
    its bit24 (QACT ST, LDA) and SPLMM from its bit25 (MM SPL), so the card needs no OTPU_MCOLS /
    OTPU_LANES / OTPU_PAIR / OTPU_DSTEP / OTPU_STREAM. When one is set in the environment it must
    name the bitstream's value (ConfigMismatch otherwise). Keyword arguments set other fields
    (DRAM_BYTES). A bitstream without the MX numerics (CAPS bit29) or whose CFG_HASH is not
    this host's hash of what its registers report is refused (check_numerics, check_hash)."""
    from opentpu.isasim import board_config
    for k in ("MCOLS", "LANES"):
        env = os.environ.get(f"OTPU_{k}")
        if env is not None and int(env) != info[k]:
            raise ConfigMismatch(f"the bitstream was built with {k}={info[k]} but OTPU_{k}={env}"
                                 f": unset OTPU_{k} (the host follows the bitstream) or load "
                                 f"a {k}={env} bitstream")
    caps = info.get("caps") or {}
    pair, dstep = bool(caps.get("pair")), bool(caps.get("dstep"))
    stream = bool(caps.get("stream"))
    for k, have, what in (("PAIR", pair, "column reuse (CAPS bit5)"),
                          ("DSTEP", dstep, "DSTEP (CAPS bit6)"),
                          ("STREAM", stream, "the stream engine (CAPS bit26)")):
        env = os.environ.get(f"OTPU_{k}")
        if env is not None and bool(int(env)) != have:
            raise ConfigMismatch(f"the bitstream {'has' if have else 'lacks'} {what} but "
                                 f"OTPU_{k}={env}: unset OTPU_{k} (the host follows the "
                                 f"bitstream)")
    rows = info.get("act_rows") or 0              # 0: MCOLS rows (no MM replay)
    dev = {"MCOLS": info["MCOLS"], "LANES": info["LANES"], "PAIR": pair, "DSTEP": dstep,
           "STREAM": stream, "ACT_ROWS": rows if rows > info["MCOLS"] else 0,
           "OUTER": _outer(info), "DSTEP2": _dstep2(info), "SE_OWN": _se_own(info),
           "KT": _kt(info), "ICACHE": _icache(info), "ST16": _st16(info),
           "FSUM": _fsum(info), "SPLITQ": _splitq(info), "ST16S": _st16s(info),
           "QHL": _qhl(info), "QSTHL": _qsthl(info), "VSIG": _vsig(info), "VMAC": _vmac(info),
           "OUTER1": _outer1(info), "QONCE": _qonce(info), "SPLMM": _splmm(info),
           "BHW": (info.get("caps2") or {}).get("bhw", 0)}
    c = info.get("cfg")
    if c:                                         # register map 4: the device says the rest
        dev.update(D=info["D"], SEQ_WC=bool(c.get("SEQ_WC")), WG=c.get("WG", 1),
                   CHASH=c.get("WG", 1) > 1 and bool(caps.get("chash")),
                   AW=device_aw(info), SEXACT=_sexact(info), KVSPLIT=_kvsplit(info),
                   WC_WIN=device_wc_win(info, device_aw(info)),
                   **{k: c[k] for k in ("S", "TMEM_WORDS", "IMEM_WORDS", "ACT_BLOCKS",
                                        "DRAM_BYTES")})
    cfg = board_config(**dev)
    if c and c["DRAM_BYTES"] > 1 << cfg.AW:
        from opentpu.target import current
        raise ConfigMismatch(f"the bitstream's slices have {c['DRAM_BYTES'] >> 20} MiB of DRAM "
                             f"each and its addresses are {cfg.AW} bits "
                             f"({(1 << cfg.AW) >> 20} MiB: platform {c['platform']} is not a "
                             f"target of this tree, the target {current().name}'s): select "
                             f"the bitstream's target (OTPU_TARGET, e.g. bcu1525-wg: AW 36)")
    check_numerics(info)
    check_hash(info, cfg)
    if info["D"] != cfg.D:
        raise ConfigMismatch(f"the bitstream has D={info['D']}, the board configuration "
                             f"D={cfg.D}: not a YPCB-00338 openTPU build")
    return replace(cfg, **kw) if kw else cfg


def _sexact(info: dict) -> bool:
    """S-exact images (Config.SEXACT, a compile choice no register says: qwen35.s_exact_on):
    the bitstream's target's (info's, else its CFG_PLATFORM's: bcu1525-s2's on), off for a
    platform this tree has not."""
    from opentpu.target import by_platform
    t = info.get("target") or by_platform(info["cfg"]["platform"])
    return bool(t is not None and t.arch.get("SEXACT"))


def _kvsplit(info: dict) -> int:
    """The decode buckets that split a copied KV head (Config.KVSPLIT, a compile choice no
    register says): the bitstream's target's, as _sexact's (bcu1525-x2's), 0 for a platform
    this tree has not."""
    from opentpu.target import by_platform
    t = info.get("target") or by_platform(info["cfg"]["platform"])
    return int(t.arch.get("KVSPLIT", 0)) if t is not None else 0


def _outer(info: dict) -> bool:
    """The VPU runs VOP OUTER: CAPS2 bit1 clear (every bitstream before the bit and before
    register map 5)."""
    return (info.get("caps2") or {}).get("outer", True)


def _dstep2(info: dict) -> bool:
    """The DMA and the stream engine run DSTEP's pair: CAPS2 bit2 (clear on every bitstream
    before it)."""
    return (info.get("caps2") or {}).get("dstep2", False)


def _icache(info: dict) -> bool:
    """IMEM is an instruction cache over the program in DRAM: CAPS2 bit12 (clear on every
    bitstream before it)."""
    return (info.get("caps2") or {}).get("icache", False)


def _st16(info: dict) -> str:
    """DSTEP's (STREAM's) DRAM state in 16 bits with its flag ST16, the format: CAPS2 bits 14:13
    ("" on every bitstream before them)."""
    st = (info.get("caps2") or {}).get("st16", "")
    if st not in ("", "e6m9", "bf16"):
        raise ConfigMismatch(f"the bitstream's CAPS2 ST16 is {st}: this host knows e6m9 and bf16")
    return st


def _se_own(info: dict) -> bool:
    """The streams run on an engine of their own beside the VPU: CAPS2 bit3 (clear on every
    bitstream before it)."""
    return (info.get("caps2") or {}).get("se_own", False)


def _kt(info: dict) -> bool:
    """The MXU takes MM's K tail: CAPS2 bit5 (clear on every bitstream before it)."""
    return (info.get("caps2") or {}).get("kt", False)


def _fsum(info: dict) -> bool:
    """The VPU writes EXP2SUB's fused row sums: CAPS2 bit16 (clear on every bitstream before
    it)."""
    return (info.get("caps2") or {}).get("fsum", False)


def _splitq(info: dict) -> bool:
    """The quantizer takes QACT / QST flag LO (the split's lo part): CAPS2 bit17 (clear on every
    bitstream before it)."""
    return (info.get("caps2") or {}).get("splitq", False)


def _st16s(info: dict) -> bool:
    """The DMA takes STREAM's flags ST16R / ST16W (one end alone 16-bit): CAPS2 bit18 (clear
    on every bitstream before it)."""
    return (info.get("caps2") or {}).get("st16s", False)


def _qhl(info: dict) -> bool:
    """The quantizer takes QACT's flag HL (a split operand's hi and lo from one read): CAPS2
    bit21 (clear on every bitstream before it)."""
    return (info.get("caps2") or {}).get("qhl", False)


def _qsthl(info: dict) -> bool:
    """The quantizer takes QST HL (a GRP ROW QST's hi and lo parts from one read): CAPS2 bit22
    (clear on every bitstream before it)."""
    return (info.get("caps2") or {}).get("qsthl", False)


def _vsig(info: dict) -> bool:
    """The VPU runs VOP SIGMOID and SILU: CAPS2 bit19 (clear on every bitstream before it)."""
    return (info.get("caps2") or {}).get("vsig", False)


def _vmac(info: dict) -> bool:
    """The VPU runs VOP MAC: CAPS2 bit20 (clear on every bitstream before it)."""
    return (info.get("caps2") or {}).get("vmac", False)


def _outer1(info: dict) -> bool:
    """The VPU, without OUTER, runs VOP OUTER of decay one on MAC's datapath: CAPS2 bit23 (clear
    on every bitstream before it)."""
    return (info.get("caps2") or {}).get("outer1", False)


def _qonce(info: dict) -> bool:
    """QACT stores its rows to DRAM (flag ST) and LDA loads them back: CAPS2 bit24 (clear on
    every bitstream before it)."""
    return (info.get("caps2") or {}).get("qonce", False)


def _splmm(info: dict) -> bool:
    """The MXU runs MM SPL (a split's terms as one MM): CAPS2 bit25 (clear on every bitstream
    before it)."""
    return (info.get("caps2") or {}).get("splmm", False)


def config_summary(info: dict) -> str:
    """One line describing the bitstream `info` (Board.info()) reports (otpu-selftest, otpu-diag):
    the MXU, the register map, with map 4 the target it was built from and its slices, the core
    clock, the memory and the build."""
    from opentpu.target import by_platform, full_name
    i, c = info, info.get("cfg")
    msg = f"D={i['D']} MCOLS={i['MCOLS']} LANES={i['LANES']}, register map {i['regmap']}"
    t = (i.get("target") or by_platform(c["platform"])) if c else None
    if c:
        beat = c.get("beat_bytes", BEAT)
        msg += (f", {full_name(t) if t else 'platform %d' % c['platform']} (S={c['S']}, "
                f"{c['channels']} channels x {c['channel_bytes'] >> 20} MiB"
                + (f" in {beat}-byte beats)" if beat != BEAT else ")"))
    if i["core_khz"]:
        msg += f", core {i['core_khz'] / 1e3:g} MHz"
    if i["ddr_mts"]:
        msg += f", {(t.platform.get('mem', 'ddr3') if t else 'ddr3').upper()}-{i['ddr_mts']}"
    if i["build_id"] is not None:
        msg += f", build {i['build_id']:08x}"
    return msg


def check_numerics(info: dict) -> None:
    """Raise ConfigMismatch unless the bitstream has the MX numerics (CAPS bit29: MM's exact
    accumulation under power-of-two block scales, QACT / QST's row scale and offsets;
    docs/isa.md, docs/mx_mxu.md), which every program and image of this host assumes. A
    bitstream without the bit (or without CAPS: register map 1) is of the MXU before MX: this
    host's programs would compute something else there."""
    if not (info.get("caps") or {}).get("mx"):
        raise ConfigMismatch("the bitstream lacks the MX numerics (CAPS bit29): its MXU and "
                             "quantizer are the ones before MX (docs/mx_mxu.md), and this host's "
                             "programs and images would compute something else there; load an "
                             "MX bitstream (this tree's) or run the bitstream's own host")


def _reg_config(info: dict, aw: int, wc_win: int = 16):
    """The Config the registers of `info` (register map 4) report, at `aw` DRAM address bits
    and a wait-entry window of `wc_win` (check_hash's, without device_config's environment
    checks)."""
    from opentpu.isasim import board_config
    c, caps = info["cfg"], info.get("caps") or {}
    rows = info.get("act_rows") or 0
    sizes = {k: c[k] for k in ("S", "TMEM_WORDS", "IMEM_WORDS", "ACT_BLOCKS", "DRAM_BYTES")}
    return board_config(D=info["D"], MCOLS=info["MCOLS"], LANES=info["LANES"],
                        PAIR=bool(caps.get("pair")), DSTEP=bool(caps.get("dstep")),
                        STREAM=bool(caps.get("stream")), SEQ_WC=bool(c.get("SEQ_WC")),
                        WG=c.get("WG", 1), CHASH=c.get("WG", 1) > 1 and bool(caps.get("chash")),
                        ACT_ROWS=rows if rows > info["MCOLS"] else 0, OUTER=_outer(info),
                        DSTEP2=_dstep2(info), SE_OWN=_se_own(info), KT=_kt(info),
                        ICACHE=_icache(info), ST16=_st16(info), FSUM=_fsum(info),
                        SPLITQ=_splitq(info), ST16S=_st16s(info), QHL=_qhl(info),
                        QSTHL=_qsthl(info), VSIG=_vsig(info), VMAC=_vmac(info),
                        OUTER1=_outer1(info), QONCE=_qonce(info), SPLMM=_splmm(info), AW=aw,
                        BHW=(info.get("caps2") or {}).get("bhw", 0),
                        WC_WIN=wc_win if c.get("SEQ_WC") else 16, **sizes)


def device_wc_win(info: dict, aw: int) -> int:
    """The window the wait entries of the bitstream `info` describes assume (Config.WC_WIN): 16
    without SEQ_WC; with it, as device_aw finds AW, the one of 16 and 32 whose hash is its CFG_HASH
    (arch_hash holds WC_WIN above 16), else 16 (a bitstream before the field, or no hash)."""
    from opentpu.target import arch_hash
    c = info.get("cfg")
    if not c or not c.get("SEQ_WC") or not c["hash"]:
        return 16
    for win in (16, 32):
        if arch_hash(_reg_config(info, aw, win)) == c["hash"]:
            return win
    return 16


def device_aw(info: dict) -> int:
    """The DRAM address bits of the bitstream `info` (Board.info()) describes. No register says
    them, its CFG_HASH does (arch_hash holds AW above 32): the first of its target's (info's,
    else its CFG_PLATFORM's: bcu1525-wg's 36), 32 and the current target's whose hash is its
    CFG_HASH (so a bcu1525-wg build from before AW, at 32, is told apart); with no hash, or none
    of them its hash (check_hash refuses it), its target's, and the current target's for a
    platform this tree has not or a bitstream before register map 4. The host follows the
    bitstream: a host of another target compiles a bcu1525-wg card's programs at AW 36."""
    from opentpu.target import arch_hash, by_platform, current
    c = info.get("cfg")
    if not c:
        return current().arch["AW"]
    t = info.get("target") or by_platform(c["platform"])
    have = [t.arch["AW"]] if t is not None else []
    have += [aw for aw in (32, current().arch["AW"]) if aw not in have]
    if c["hash"]:
        for aw in have:
            if any(arch_hash(_reg_config(info, aw, win)) == c["hash"] for win in (16, 32)):
                return aw
    return have[0]


def check_hash(info: dict, cfg=None) -> None:
    """Raise ConfigMismatch when the bitstream's CFG_HASH (register map 4) is not this host's
    hash of the configuration its registers report (`cfg`: device_config's, default built
    here without its environment checks): the host and the build disagree on the Config's
    fields or on what the hash holds besides them (opentpu/target.py arch_hash), so programs
    compiled here would compute something else there. Nothing to check before map 4 or with
    a zero hash."""
    from opentpu.target import arch_hash
    c = info.get("cfg")
    if not c or not c["hash"]:
        return
    if cfg is None:
        aw = device_aw(info)
        cfg = _reg_config(info, aw, device_wc_win(info, aw))
    if arch_hash(cfg) != c["hash"]:
        raise ConfigMismatch(f"the bitstream's CFG_HASH {c['hash']:08x} is not the hash of the "
                             f"configuration its registers report ({arch_hash(cfg):08x}): the "
                             f"host and the build disagree on the Config's fields or on what "
                             f"the hash holds besides them, e.g. the MXU's numerics "
                             f"(opentpu/target.py arch_hash): load a bitstream of this tree or "
                             f"run the bitstream's host")


def check_target(info: dict, want=None):
    """The target `info`'s bitstream (register map 4) is a build of: `want` (a Target, or a name:
    a variant by name, target:variant), else the one of its CFG_PLATFORM (by_platform; None for a
    platform this tree has not). ConfigMismatch when it is not `want`'s build: its platform,
    slices and channels not `want`'s, or its CFG_HASH not the hash of `want`'s build (the build's
    ARCH_HASH: arch_hash(build_config(want)); a build with other knobs, or of another variant,
    computes something else); and, without `want`, when it is a variant's build (its slices not
    its platform's target's: a variant shares the platform id, so the host has to be told which
    one, e.g. OTPU_TARGET=<target>:<variant>). One card of a target over
    several (bcu1525-x2: [platform] cards) reports its own slices, S / cards (opentpu.host.cards):
    that is its target's build, not a variant's."""
    from opentpu.target import (arch_hash, build_config, by_platform, cards, full_name, load,
                                variants)
    c = info.get("cfg")
    if not c:
        return None
    base = by_platform(c["platform"])
    if want is None:
        if base is not None and c["S"] not in (base.arch["S"], base.arch["S"] // cards(base)) \
                and variants(base):
            raise ConfigMismatch(f"the bitstream is a {c['S']}-slice build of platform "
                                 f"{c['platform']} ({base.name} has {base.arch['S']}): a "
                                 f"variant's, which shares the platform id. Name it: "
                                 f"OTPU_TARGET={base.name}:<variant> ("
                                 f"{', '.join(variants(base))})")
        return base
    t = load(want) if isinstance(want, str) else want
    pl = t.platform
    have = (c["platform"], c["S"], c["channels"], c["channel_bytes"], c.get("beat_bytes", BEAT))
    exp = (t.platform_id, t.arch["S"], pl.get("channels_per_slice", 2),
           pl.get("channel_bytes", have[3]), pl.get("beat_bytes", BEAT))
    if have != exp:
        raise ConfigMismatch(f"the bitstream is not a build of {full_name(t)}: platform, slices, "
                             f"channels, channel and beat bytes {have}, {full_name(t)}'s {exp}")
    h = arch_hash(build_config(t))
    if c["hash"] != h:
        raise ConfigMismatch(f"the bitstream's CFG_HASH {c['hash']:08x} is not {full_name(t)}'s "
                             f"build's ({h:08x}, its ARCH_HASH): another variant's, a "
                             f"build with other knobs or of another tree; load {full_name(t)}'s "
                             f"bitstream or name the build it holds")
    return t


def check_fits(cfg, info: dict) -> None:
    """Raise ConfigMismatch unless programs compiled for `cfg` run on the device `info`
    describes: the MX numerics (check_numerics), its CFG_HASH this host's (check_hash), the
    same MXU, lanes and ACT RAM rows, and (register map 4) no more slices, TMEM, IMEM, ACT RAM
    blocks or DRAM than it has, and room for the wait entries if its sequencer takes them
    (SEQ_WC: the programs' chain slots and area; a configuration with the room runs on a
    bitstream without them too); programs with OUTER of decay one on a VPU without OUTER
    (OUTER1) need a VPU that runs one (CAPS2 bit23, or OUTER itself: bit1 clear), programs
    with QACT ST and LDA (QONCE) a bitstream with them (CAPS2 bit24), and programs with FSUM,
    SPLITQ, ST16S, VSIG, VMAC, QHL, QSTHL or SPLMM (MM SPL) a bitstream with it (its CAPS2 bit)."""
    check_numerics(info)
    check_hash(info)
    if getattr(cfg, "OUTER1", False) and not cfg.OUTER and not (_outer(info) or _outer1(info)):
        raise ConfigMismatch("the programs take VOP OUTER of decay one (Config.OUTER1) and the "
                             "bitstream's VPU runs no OUTER (CAPS2 bit1 set, bit23 clear: a "
                             "build without OUTER1) (use device_config)")
    if getattr(cfg, "QONCE", False) and not _qonce(info):
        raise ConfigMismatch("the programs take QACT ST and LDA (Config.QONCE) and the "
                             "bitstream has neither (CAPS2 bit24 clear) (use device_config)")
    for k, has, what in (("FSUM", _fsum, "EXP2SUB's fused row sums (CAPS2 bit16)"),
                         ("SPLITQ", _splitq, "QACT / QST flag LO (CAPS2 bit17)"),
                         ("ST16S", _st16s, "STREAM's ST16R / ST16W (CAPS2 bit18)"),
                         ("VSIG", _vsig, "VOP SIGMOID / SILU (CAPS2 bit19)"),
                         ("VMAC", _vmac, "VOP MAC (CAPS2 bit20)"),
                         ("QHL", _qhl, "QACT flag HL (CAPS2 bit21)"),
                         ("QSTHL", _qsthl, "QST HL (CAPS2 bit22)"),
                         ("SPLMM", _splmm, "MM SPL, a split's terms as one MM (CAPS2 bit25)")):
        if getattr(cfg, k, False) and not has(info):
            raise ConfigMismatch(f"the programs take {what} (Config.{k}) and the bitstream has "
                                 f"none (e.g. x2:wmx's programs on B2, x2:b2's on B1) (use "
                                 f"device_config)")
    rows = info.get("act_rows") or info["MCOLS"]
    if (info["D"], info["MCOLS"], info["LANES"], rows) != \
            (cfg.D, cfg.MCOLS, cfg.LANES, cfg.act_rows):
        raise ConfigMismatch(f"the bitstream is D={info['D']} MCOLS={info['MCOLS']} "
                             f"LANES={info['LANES']}, the configuration D={cfg.D} "
                             f"MCOLS={cfg.MCOLS} LANES={cfg.LANES} (use device_config)")
    c = info.get("cfg")
    if not c:
        return
    if c.get("WG", 1) != cfg.WG:
        raise ConfigMismatch(f"the bitstream's MXU takes weights in groups of {c.get('WG', 1)} "
                             f"rows (CFG_UARCH [27:25]), the configuration WG={cfg.WG}: the "
                             f"images' layout and the programs' MMs follow WG (use device_config)")
    chash = c.get("WG", 1) > 1 and bool((info.get("caps") or {}).get("chash"))
    if cfg.WG > 1 and chash != cfg.CHASH:
        raise ConfigMismatch(f"the bitstream's memory {'hashes' if chash else 'does not hash'} "
                             f"its channels (CAPS bit7), the configuration CHASH={cfg.CHASH}: "
                             f"the images' grouped matrices store their rows in its order "
                             f"(docs/isa.md \"Weight groups\"; use device_config)")
    if c.get("SEQ_WC") and not getattr(cfg, "SEQ_WC", False):
        raise ConfigMismatch("the bitstream's sequencer takes wait entries (CFG_UARCH bit24) "
                             "and the configuration has no room for them (SEQ_WC off: the "
                             "chain slots and the program area hold the words alone): use "
                             "device_config")
    for k in ("S", "TMEM_WORDS", "IMEM_WORDS", "ACT_BLOCKS", "DRAM_BYTES"):
        if getattr(cfg, k) > c[k]:
            raise ConfigMismatch(f"the programs need {k}={getattr(cfg, k)} and the device has "
                                 f"{c[k]} (use device_config)")


# ------------------------------------------------------------------------------ Engine backend
def sim_config(spec, cap: int, base=None, rows: int | None = None, lookup: bool = False,
               wformat: str = "int8", head_format: str | None = None):
    """`base` (default board_config()) with the DRAM cut to what the model needs (power of
    two): the image with I/O rows for `rows` tokens per run (default the Engine's
    PREFILL_ROWS; lookup: with the resident decode's tables, Engine(resident=True)), then the
    program area. The board model's memory, and the ISA reference that runs the same
    layout. The image sized is the int8 one (the largest: every format of a model gets the
    same DRAM), or where that is over the card's DRAM (base's, at most 2^AW: 4 GiB at AW 32,
    Qwen3.5-4B's; the BCU's 32 GiB at AW 36, Qwen3.6-35B-A3B's, which 2^AW gave 64 GiB) the one
    in `wformat` / `head_format`."""
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import PREFILL_ROWS, has_lookup
    base = base or board_config()
    lookup = lookup or getattr(spec, "embed", "f32") == "int8"     # as Engine: always tables
    kw = {"lookup": True} if lookup and has_lookup(spec) else {}
    big = replace(base, DRAM_BYTES=min(1 << base.AW, base.DRAM_BYTES))   # (AW 32: 4 GiB)
    try:
        probe = spec.image(big, cap, 1, rows or PREFILL_ROWS, **kw)
    except MemoryError:
        probe = spec.image(big, cap, 1, rows or PREFILL_ROWS, wformat=wformat,
                           head_format=head_format, **kw)
    need = -(-probe.nbytes // 4096) * 4096 + 4 * base.IMEM_WORDS
    return replace(base, DRAM_BYTES=1 << max(22, (need - 1).bit_length()))


def dram_layout(cfg, image_bytes: int, prog_at: int, image=None, poss=None,
                snapshots: int = 0) -> dict:
    """The device DRAM in bytes: the image (weights, norms, I/O area and KV capacity, and the
    Engine's `snapshots` area after it: opentpu/llm/snapshot.py), the program area after it,
    free; with a model Image also the KV cache capacity (LFM2: with the convolution state;
    Qwen3.5: with the convolution and DeltaNet state, which do not grow with the position but
    are counted as if they did) and the part filled at positions `poss` (one per sequence); an
    Image without kv_bytes (GPT-OSS's) counts none."""
    from opentpu import waits as W
    prog = W.area(cfg.IMEM_WORDS, getattr(cfg, "SEQ_WC", False))     # (with its wait entries)
    d = {"total": cfg.DRAM_BYTES, "image": image_bytes, "weights": image_bytes - snapshots,
         "kv_capacity": 0, "kv_used": 0, "program": prog, "snapshots": snapshots,
         "free": max(0, cfg.DRAM_BYTES - max(prog_at + prog, image_bytes))}
    if image is not None:
        cap, per_seq = image.cap, getattr(image, "kv_bytes", 0)
        d["kv_capacity"] = per_seq * image.batch
        d["weights"] = image_bytes - snapshots - d["kv_capacity"]
        d["kv_used"] = int(sum(per_seq * min(p, cap) / cap for p in (poss or [0])))
    return d


def keep_ranges(runs, spans, chunk: int, top: int, align: int | None = None,
                gap: int | None = None) -> list[tuple[int, int]]:
    """A slice's image data (`runs`: byte ranges holding every non-zero byte the backend writes
    of it, images.runs) -> the ranges BoardBackend writes after a scrub that leaves them out
    (Board.scrub keep): each run's part inside `spans` (the ranges written), out to `align`
    (KEEP_ALIGN, in whole wide chunks of `chunk` bytes) within [0, top), the ones less than
    `gap` (KEEP_GAP) apart joined (large DMA calls, in the scrub and the write)."""
    gap = KEEP_GAP if gap is None else gap
    a = -(-(KEEP_ALIGN if align is None else align) // chunk) * chunk
    rs = sorted((x0 // a * a, min(top, -(-x1 // a) * a)) for lo, hi in runs for sa, sb in spans
                for x0, x1 in [(max(lo, sa), min(hi, sb))] if x0 < x1)
    out: list[list[int]] = []
    for x0, x1 in rs:
        if out and x0 - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], x1)
        else:
            out.append([x0, x1])
    return [(x0, x1) for x0, x1 in out]


def image_pieces(m: np.ndarray, spans, keep, chunk: int):
    """The image `m`'s bytes over the ranges `keep` (whole wide chunks of `chunk` bytes) as a
    scrub and the write of its `spans` leave them: m's inside the spans, zeros outside them and
    past m's end. (addr, bytes) of whole chunks in turn: m's own where a piece is inside a span,
    a copy where a span's unaligned edge falls in it (its chunk) or none does (zeros)."""
    n = len(m)
    for lo, hi in keep:
        cuts = {lo, hi}
        for x in (x for ab in spans for x in ab if lo < x < hi):
            cuts |= {x // chunk * chunk, -(-x // chunk) * chunk}
        cuts = sorted(cuts)
        for x0, x1 in zip(cuts, cuts[1:]):
            if any(a <= x0 and x1 <= b for a, b in spans) and x1 <= n:
                yield x0, m[x0:x1]
                continue
            buf = np.zeros(x1 - x0, np.uint8)
            for a, b in spans:
                a, b = max(a, x0), min(b, x1, n)
                if a < b:
                    buf[a - x0:b - x0] = m[a:b]
            yield x0, buf


def _spans(images, n: int) -> list[tuple[int, int]]:
    """The byte ranges BoardBackend writes of an image array of n bytes (qwen3.Images.spans; a
    list of arrays: the whole of each)."""
    return images.spans(n) if hasattr(images, "spans") else [(0, n)]


def _by_card(t, items, per: int) -> list:
    """items (channels or slices, `per` of them a card) one list a card when t is cards in
    tensor parallel whose DMA can run side by side (CardsTransport, threaded); else one list."""
    n = getattr(t, "ncards", 1)
    if n < 2 or not getattr(t, "threaded", False):
        return [list(items)]
    return [[i for i in items if i // per == c] for c in range(n)]


def _side_by_side(fn, groups) -> None:
    """fn(*group) for each group, each on a thread of its own when there are more (the cards'
    writes at once: each card's link its own)."""
    if len(groups) == 1:
        fn(*groups[0])
        return
    with concurrent.futures.ThreadPoolExecutor(len(groups), thread_name_prefix="otpu-card") as ex:
        for f in [ex.submit(fn, *g) for g in groups]:
            f.result()


def _nwords(words) -> int:
    """The longest program's words (one array, or a list of them: one per slice)."""
    return max(len(w) for w in words) if isinstance(words, (list, tuple)) else len(words)


class BoardBackend:
    """Engine backend on the card: images are written once, each token's program is copied to
    the program area right after the image and loaded into IMEM, then run.

    prepare(programs) assembles ahead of time (the Engine calls it on its compile thread);
    start(programs) and wait() split run(programs), so the Engine compiles the next program
    while the card runs (and not while the host copies the program);
    attach(engine) lets the status file follow the engine's KV cache; `trace` (None, or run()'s
    trace options) records the following runs in the trace buffer; `last` is (programs, stats)
    of the latest run. With a device transport it holds the device lock and publishes
    /tmp/otpu/<dev>.json (runstate.RunnerStatus) until close() or exit."""

    def __init__(self, cfg, images: list, transport=None, model: str | None = None,
                 status: bool = True):
        from opentpu import isa as I
        self.I = I
        self.cfg = cfg
        self.board = Board(transport)
        # images.runs (for slice s, byte ranges holding every non-zero byte of its spans): the
        # scrub leaves their ranges out (keep_ranges) and only those are written after it; a
        # scrub skipped (SCRATCH = DRAM_INIT: the DRAM holds what it held) writes the spans
        runs = getattr(images, "runs", None)
        keep = None if runs is None else self._keep(images, runs)
        scrubbed = self.board.scrub() if keep is None else self.board.scrub(keep=keep)
        info = self.info = self.board.info()
        try:
            check_fits(cfg, info)
            c = info.get("cfg")
            if c and (c["channels"] < 2 or c["channels"] & (c["channels"] - 1)
                      or c.get("beat_bytes", BEAT) not in (64, 128)):
                raise ConfigMismatch(f"the bitstream's slices have {c['channels']} memory "
                                     f"channels of {c.get('beat_bytes', BEAT)}-byte beats each: "
                                     "this driver interleaves a power of two of at least two, "
                                     "in 64- or 128-byte beats")
            # its sequencer takes wait entries (EXPERIMENTAL): each program loads with its own
            # (check_fits: the configuration has their room; Board.waitc: the window)
            self.waitc = self.board.waitc
        except ConfigMismatch:
            self.board.close()
            raise
        # slice s's image in its own DRAM (cfg.S of the device's slices; the others run a HALT:
        # Board.load_program), every slice's program at prog_at, after the largest image; an
        # image with its weights from 4 GiB on (qwen3.Images: high) has its low part to `end`,
        # the program area after it below 4 GiB
        imgs = [np.asarray(images[s], np.uint8) for s in range(cfg.S)]
        self.image_bytes = max(len(m) for m in imgs)
        end, self.high = getattr(images, "end", self.image_bytes), getattr(images, "high", None)
        self.prog_at = -(-end // 4096) * 4096
        from opentpu import waits as W
        self.W = W
        if self.prog_at + W.area(cfg.IMEM_WORDS, cfg.SEQ_WC) > \
                (cfg.DRAM_BYTES if self.high is None else self.high):
            self.board.close()
            raise MemoryError("no room for the program area")
        self.engine = None
        self.trace: dict | None = None
        self.last = None
        self._prep: dict = {}               # id(programs) -> (programs, words, entries)
        self._running = None                # the started programs
        self._resident = None               # (programs, words) in IMEM (start() skips the load)
        self._expect = 0.0                  # the run's expected seconds (the poll's hint)
        # the expected seconds per program length and run arguments: a run is expected to take
        # what the last run of its length and arguments took (the first decode run after a
        # prefill chunk, expected to take the chunk's 25 ms, was seen 14 ms late on the card),
        # one with arguments not seen before the least of its length's last RUN_CLOCK_N runs
        # (a layer loop's program has one length for any run count: a 2-run loop expected to
        # take an 8-run loop's time slept past its end, +17 s on a 2,689-token prompt's 62.7 s
        # on the card)
        self._expects: dict = {}
        self._gen_gaps: dict[int, float] = {}  # program length -> run_generate's seconds per token
        self._t0s: dict = {}                # program length -> its last runs' seconds (run_clock)
        self._key = 0                       # the started program's length
        self._akey = None                   # (its length, its run arguments or None)
        self._ratio = 1.0                   # run wall time / (CYCLES / CORE_KHZ), smoothed
        self._seen = None                   # HALTED seen by the streamed wait (perf_counter)
        # path (a)'s expert server (opentpu/host/offload.py, ExpertServer.poll), set by the
        # Engine: called over and over while a run is in flight, it serves the card's requests
        # (a MoE layer's WAITWs wait for it, docs/offload.md 5.2); nonzero when it served one
        self.host = None
        # the host's mailbox servers (the Engine's: ExpertServer, RowServer), rebased before each
        # run (their seq back to 0 before fp32's 2^24; offload.SEQ_REBASE)
        self.servers: list = []
        # streamed logits (start(stream=...)): the region holding the sentinel, its marking
        # after a run (on the DMA worker, waited for before anything else writes the card),
        # the pieces' completion times last token
        self.streams = bool(getattr(self.board.t, "streams", False))
        self._armed = None                  # (addr, nbytes, piece) filled with SENTINEL
        self._rearm = None                  # the marking in flight (a Future)
        self._marks = np.zeros(0, np.uint32)    # SENTINEL words for it
        self._stream = None                 # the running program's (addr, nbytes, piece)
        self._due: dict = {}
        self.last_stream: dict = {}         # the last streamed wait: pieces during the run...
        t = self.board.t
        self.status = RunnerStatus(t.devname, STATUS_INTERVAL, dev=getattr(t, "dev", t.devname),
                                   model=model,
                                   core_khz=info["core_khz"], dram=self._layout()) \
            if status and self.board.lock is not None else None
        def put(board, ss):
            for s in ss:
                m = imgs[s]
                if keep is not None and scrubbed:   # (every beat the scrub left: m's or zeros)
                    board.write_chunks(image_pieces(m, _spans(images, len(m)), keep[s],
                                                    board.nch * board.beat), s)
                elif self.high is None:
                    board.write(0, m, s)
                else:                   # (the bytes between are the image's unread zeros; and
                    for lo, hi in images.spans(len(m)):     # the bank-half window's past the
                        board.write(lo, m[lo:hi], s)        # mixers')
        # cards in tensor parallel: each card's slices on a thread of its own, through a Board
        # of their own on the transport (its own pipeline: Board._piped's buffers and threads)
        groups = _by_card(t, range(len(imgs)), getattr(t, "spc", cfg.S))
        boards = [self.board] + [self.board.sharing() for _ in groups[1:]]
        try:
            _side_by_side(put, list(zip(boards, groups)))
        finally:
            for b in boards[1:]:
                b.close()

    def _keep(self, images, runs) -> list:
        """Board.scrub's keep for images.runs: slice s's keep_ranges (the device's slices of
        cfg.S: check_fits refuses more)."""
        b = self.board
        c = b.info()["cfg"]
        top = b.nch * (c["channel_bytes"] if c else CH_BYTES)     # a slice's logical bytes
        return [keep_ranges(runs[s], _spans(images, len(np.asarray(images[s], np.uint8))),
                            b.nch * b.beat, top) for s in range(min(self.cfg.S, b.slices))]

    def _layout(self, next_token: bool = False) -> dict:
        e = self.engine
        poss = list(getattr(e, "poss", [])) or None
        if poss and next_token:
            poss[0] += 1                    # the run in flight fills the next position
        return dram_layout(self.cfg, self.image_bytes, self.prog_at,
                           getattr(e, "image", None), poss, getattr(e, "snapshot_bytes", 0))

    def attach(self, engine) -> None:
        """Called by the Engine once it exists. A model image with 4-bit weights needs a
        bitstream whose MXU runs them (CAPS bit4)."""
        img = getattr(engine, "image", None)
        fmts = {getattr(img, "wformat", "int8"), getattr(img, "head_format", "int8")}
        caps = self.info.get("caps") or {}
        if fmts != {"int8"} and not caps.get("w4"):
            self.board.close()
            raise ConfigMismatch("the model image has 4-bit weights and this bitstream's MXU "
                                 "runs int8 weights only (CAPS bit4 clear): use --wformat int8 "
                                 "or load a newer bitstream")
        if self.cfg.PAIR and not caps.get("pair"):
            self.board.close()
            raise ConfigMismatch("the programs use column reuse (MM PAIR / QACT DUP) and this "
                                 "bitstream lacks it (CAPS bit5 clear): use device_config")
        if self.cfg.DSTEP and not caps.get("dstep"):
            self.board.close()
            raise ConfigMismatch("the programs use DSTEP and this bitstream lacks it (CAPS bit6 "
                                 "clear): use device_config")
        if self.cfg.STREAM and not caps.get("stream"):
            self.board.close()
            raise ConfigMismatch("the programs use STREAM and this bitstream lacks the stream "
                                 "engine (CAPS bit26 clear): use device_config")
        self.engine = engine
        if self.status is not None:
            self.status.update(dram=self._layout())

    def write(self, s: int, addr: int, data: np.ndarray) -> None:
        self._settle()
        a = self._armed
        if s == 0 and a is not None and addr < a[0] + a[1] and \
                a[0] < addr + np.asarray(data).nbytes:
            self._armed = None
        self.board.write(addr, data, s)

    def read(self, s: int, addr: int, nbytes: int) -> np.ndarray:
        return self.board.read(addr, nbytes, s)

    def _assemble(self, programs):
        """The programs' words: one array (one slice), or slice s's at index s."""
        if self.cfg.S == 1:
            return np.asarray(self.I.assemble(programs[0]), np.uint32)
        return [np.asarray(self.I.assemble(p), np.uint32) for p in programs[:self.cfg.S]]

    def _entries(self, words, programs=None):
        """The wait entries of the programs' words (as _assemble gives them; a WAITC bitstream:
        waits.entries, kept per program), or None on a bitstream without them. programs: the
        instructions, when at hand (else the words are decoded; a compile worker's words,
        compiler.Words, carry their run_args)."""
        if not self.waitc:
            return None
        D, S = self.cfg.D, self.cfg.S
        if isinstance(words, np.ndarray):
            if getattr(words, "waits", None) is not None:   # (a compile worker's)
                return words.waits
            p = programs[0] if programs is not None and not isinstance(programs, np.ndarray) \
                else None
            return self.W.entries(words, D, S, prog=p, aw=self.cfg.AW, win=self.cfg.WC_WIN)
        ps = programs if programs is not None and not all(
            isinstance(w, np.ndarray) for w in programs) else [None] * len(words)
        return [w.waits if getattr(w, "waits", None) is not None else
                self.W.entries(w, D, S, prog=p, aw=self.cfg.AW, win=self.cfg.WC_WIN)
                for w, p in zip(words, ps)]

    def slot_words(self, prog) -> np.ndarray:
        """A chain slot's contents for `prog` (one slice's: generate.slot_words): its words,
        and on a WAITC bitstream its entries after them (waits.loadable), as load_program's
        on several cards (Board.cross_card_entries: FOLLOW cleared; a chained bucket with FOLLOW
        read the two cards' gathered rows before the remote pieces landed, x2a 11-04)."""
        w = np.asarray(self.I.assemble(prog), np.uint32)
        if not self.waitc:
            return w
        e = self.W.entries(w, self.cfg.D, self.cfg.S, prog=prog, aw=self.cfg.AW,
                           win=self.cfg.WC_WIN)
        return self.W.loadable(w, self.board.cross_card_entries(e), self.cfg.D)

    def prepare(self, programs: list) -> None:
        """Assemble ahead of run(programs), and on a WAITC bitstream compute the wait entries.
        Called from the Engine's compile thread while run() of the previous token goes on: one
        dict entry per program list (dict operations are atomic), dropped by run()."""
        words = self._assemble(programs)
        self._prep[id(programs)] = (programs, words, self._entries(words, programs))

    runs_words = True           # start() takes assembled words too (the Engine's worker process)

    @property
    def fills(self) -> bool:
        """Streamed runs may fill their logits region themselves (qwen3.fill_logits: an RLD, so
        a bitstream with CAPS bit30), the host polling ICOUNT for its end."""
        return self.streams and bool((self.info.get("caps") or {}).get("gen"))

    @property
    def args(self) -> bool:
        """The bitstream takes run arguments (CAPS bit25): start(programs, args=words)."""
        return bool((self.info.get("caps") or {}).get("args"))

    def start(self, programs, stream: tuple | None = None, args=None) -> None:
        """Copy the program to the card, load it and start it (run's first half: the Engine
        compiles the next token's program between start and wait). `programs`: the programs,
        or the program already assembled (uint32 words).

        stream=(addr, nbytes, piece): the run writes its logits to [addr, addr + nbytes) in
        pieces of `piece` bytes (the LM head's chunks, late in the run), and wait(feed) hands
        each piece over as soon as it is complete. The region holds SENTINEL words before the
        run: written here when it does not (the first streamed run, or after anything else
        wrote the region), else it was marked again after the last run (_stream_tail, on the
        DMA worker while the host takes the token; waited for here, before RUN: a write that
        size must not meet a run, RUN_H2C). Needs a transport that allows DMA during a run
        (`streams`). stream=(addr, nbytes, piece, gate): the program fills the region with
        FILL_SENTINEL itself (qwen3.fill_logits; `fills`), and its fill has landed once ICOUNT
        reaches `gate` (qwen3.fill_gate): the host writes nothing there, before or after.

        args: the run's arguments (up to 8 words, ARG0..7; a bitstream with CAPS bit25). The
        program in IMEM stays there: starting the same `programs` object again (a program
        that takes its position as arguments) loads nothing, on a transport that keeps the
        device's state between runs (not SimTransport: it loads again)."""
        if args is not None and not self.args:
            raise ConfigMismatch("this bitstream takes no run arguments (CAPS bit25 clear)")
        n = 1 if isinstance(programs, np.ndarray) else len(programs)
        if n < self.cfg.S:
            # (load_program pads the other slices with HALT: a run that waits on them never ends)
            raise ValueError(f"{n} slice's program on an S = {self.cfg.S} machine: give every "
                             f"slice's (a one-slice compile worker's words in the program cache?)")
        self._settle()
        for srv in self.servers:
            srv.rebase()
        res = self._resident
        if res is None or res[0] is not programs or not getattr(self.board.t, "keeps_state",
                                                                  True):
            prep = self._prep.pop(id(programs), None)
            while len(self._prep) > 1:              # stale entries (discarded compiles)
                self._prep.pop(next(iter(self._prep)), None)
            if isinstance(programs, np.ndarray) or \
                    (self.cfg.S > 1 and all(isinstance(w, np.ndarray) for w in programs)):
                words, ents = programs, self._entries(programs)
            elif prep is not None and prep[0] is programs:
                words, ents = prep[1], prep[2]
            else:
                words = self._assemble(programs)
                ents = self._entries(words, programs)
            if _nwords(words) > self.cfg.IMEM_WORDS:
                raise ValueError("program does not fit IMEM" if not self.cfg.ICACHE else
                                 f"program longer than the bitstream's longest (CFG_IMEM "
                                 f"{self.cfg.IMEM_WORDS // 8} instructions: ICACHE)")
            self._resident = None
            if ents is None:
                self.board.load_program(self.prog_at, words)
            else:                                   # (WAITC: with the programs' entries)
                self.board.load_program(self.prog_at, words, ents)
            self._resident = (programs, words)
        if args is not None:
            self.board.set_args(list(args) + [0] * (8 - len(args)))
        if stream is not None and len(stream) > 3:
            self._armed = None                      # the program fills the region itself
        elif stream is not None and self._armed != stream:
            self.board.write(stream[0], np.full(stream[1] // 4, SENTINEL, np.uint32))
            self._armed = stream
        elif stream is None:
            self._armed = None                      # the run may write the region
        self._seen = None
        self._key = _nwords(self._resident[1])
        self._akey = (self._key, None if args is None else tuple(int(a) for a in args))
        self._expect = self._expects.get(self._akey)
        if self._expect is None:
            ts = self._t0s.get(self._key)
            self._expect = min(ts) if ts and args is not None else 0.0
        self.board.start(trace=self.trace)
        self._running, self._stream = programs, stream

    def run_clock(self) -> tuple | None:
        """(the started run's start, perf_counter; its time with no waits: the shortest of the
        last RUN_CLOCK_N runs of its program's length, seen wall time per device second as
        _next_expect's ratio), or None with no run started or none of its length before
        (docs/offload.md 13.12: the expert server adds the run's own waits on it and holds idle
        parts back near its end)."""
        ts = self._t0s.get(self._key)
        if self._running is None or not ts:
            return None
        return self.board._t_run, min(ts)

    def wait(self, feed=None) -> dict:
        """Wait for the started program; returns its counters (run's second half). After a
        start(stream=...), feed(offset, words) gets every piece of the logits (byte offset in
        the region, uint32 words), most of them while the run goes on."""
        programs = self._running              # (running until it halts: run_clock, polled
        try:                                    # by the host hook during _serve)
            if self._stream is not None:
                self._stream_logits(feed)
            elif self.host is not None:
                self._serve()
            # halted already (seen by the streamed wait): read the counters, no sleep
            st = self.board.wait(expect=0.0 if self._seen else self._expect)
            if self._stream is not None:
                self._stream_tail(feed)
        except BaseException:
            self._armed, self._stream = None, None  # the region's state is unknown
            self.board.abort(STOP_WAIT)             # (the run, if it still goes)
            raise
        finally:
            self._running = None
        khz = self.info["core_khz"]
        self._expects.pop(self._akey, None)
        self._expects[self._akey] = self._next_expect(st["cycles"] / (khz * 1e3) if khz else 0.0)
        while len(self._expects) > EXPECT_N:
            self._expects.pop(next(iter(self._expects)))
        if khz:
            self._t0s.setdefault(self._key, collections.deque(maxlen=RUN_CLOCK_N)).append(
                st["cycles"] / (khz * 1e3) * self._ratio)
        self.last = (programs, st)
        if self.status is not None:
            self.status.token(st["cycles"], khz, dram=self._layout(True))
        return st

    def _serve(self) -> None:
        """Serve the host hook until the run halts (not on the board model, which runs a
        script: no host during a run)."""
        t = self.board.t
        if getattr(t, "batched", False):
            return
        idle = getattr(t, "host_idle", HOST_IDLE)
        end = time.perf_counter() + RUN_TIMEOUT
        while not t.reg_read(R_STATUS) & ST_HALTED:
            if not self.host() and idle:
                time.sleep(idle)
            if time.perf_counter() > end:
                raise TimeoutError(f"the run did not halt in {RUN_TIMEOUT:.0f} s while the host "
                                   "served its requests")
        self._seen = time.perf_counter()

    def _next_expect(self, dev: float) -> float:
        """The next run's expected wall time: this run's device time (CYCLES / CORE_KHZ) times
        the smoothed ratio of the seen run times to it (the clock may differ from CORE_KHZ by
        a fraction of a percent: 0.25% of a 53 ms token is past the poll's spin; the seen time
        includes the poll's lateness, which the ratio's average keeps small)."""
        streamed, self._seen = self._seen, None
        seen = streamed or getattr(self.board, "t_seen", None)
        t_run = getattr(self.board, "_t_run", None)
        if dev > 0 and seen is not None and t_run is not None:
            r = (seen - t_run) / dev
            # halted at the first read after a sleep (or by the streamed wait's probes): the
            # run ended some time before, so the ratio may only come down to it; halted by
            # back-to-back reads that saw it running: within a read of the end
            upper = streamed is not None or not getattr(self.board, "seen_exact", False)
            if 0.9 < r < 1.1 and (r < self._ratio or not upper):
                self._ratio = 0.5 * self._ratio + 0.5 * r
        return dev * self._ratio

    def _pieces(self):
        addr, n, piece = self._stream[:3]
        return [(o, min(piece, n - o)) for o in range(0, n, piece)]

    def _stream_logits(self, feed) -> None:
        """Hand over the pieces before the last one while the run goes on: probe a piece's
        last beat from STREAM_EARLY before its completion time of the last token (or every
        STREAM_PROBE), read it once the beat is written, check every word, mark it again.
        Stops at HALTED; the last piece (written just before it) is left to _stream_tail. A
        program that fills the region itself: no piece is probed before ICOUNT shows its fill
        landed (until then the region holds the last token's logits)."""
        t, b = self.board.t, self.board
        addr = self._stream[0]
        gate = self._stream[3] if len(self._stream) > 3 else None
        mark = SENTINEL if gate is None else FILL_SENTINEL
        pieces, due = self._pieces(), {}
        t0, i, probes, tries = b._t_run, 0, 0, 0
        halted = False
        # from POLL_EARLY (+ 3%) before the expected end on, the slices are short, so a run
        # that ends while a piece is awaited is seen within ~POLL_MIN_SLEEP (1 ms slices: Qwen3,
        # whose next-to-last piece comes near the end, saw HALTED 0.7-0.8 ms late on the card)
        near = self._expect - POLL_EARLY - 0.03 * self._expect if self._expect else float("inf")
        while i < len(pieces) - 1:
            if time.perf_counter() - t0 > RUN_TIMEOUT:
                raise TimeoutError(f"streamed logits: the run did not halt in {RUN_TIMEOUT:.0f} s "
                                   f"({i} of {len(pieces)} pieces handed over)")
            d = self._due.get(i)
            # a piece's time of the last token may be past this run's end (a token delayed
            # by the host): HALTED is checked after every slice, not only at the piece's time
            while d is not None and (now := time.perf_counter() - t0) < d - STREAM_EARLY:
                cap = 1e-3 if now < near else POLL_MIN_SLEEP
                time.sleep(min(d - STREAM_EARLY - now, cap))    # slices: sleeps overshoot
                if t.reg_read(R_STATUS) & ST_HALTED:
                    halted = True
                    break
            if halted or t.reg_read(R_STATUS) & ST_HALTED:
                self._seen = time.perf_counter()
                break
            if gate is not None:
                if t.reg_read(R_ICOUNT) < gate:        # the fill has not landed yet
                    time.sleep(POLL_MIN_SLEEP)
                    continue
                gate = None
            o, k = pieces[i]
            last = addr + o + k - BEAT                  # the piece's last 64 bytes (aligned)
            probes += 1
            tries += 1
            beat = t.mem_read(*b.beat_at(last), BEAT)
            w = None
            if not (beat.view(np.uint32) == mark).any():
                w = b.read(addr + o, k).view(np.uint32)
                if (w == mark).any():                   # the beats land out of order: soon
                    w = None
            if w is None:
                # near the expected end the retry is short too: the run may end while this
                # piece is awaited (a 0.1 ms retry left HALTED seen 0.15 ms late on the card)
                late_run = time.perf_counter() - t0 >= near
                time.sleep(POLL_MIN_SLEEP if late_run else
                           STREAM_RETRY if d is not None else STREAM_PROBE)
                continue
            feed(o, w)
            # complete at the first probe: it may have been for a while, so probe earlier
            # next token; else it came between the last two probes
            due[i] = d - STREAM_EARLY if tries == 1 and d is not None else \
                time.perf_counter() - t0
            # its SENTINEL marks go back after the run (with the tail's): a write here
            # kept the host busy when the run ended (HALTED seen up to 0.15 ms late), and
            # more than RUN_H2C a call must not meet the run
            i, tries = i + 1, 0
        self._due.update(due)
        self.last_stream = {"during": i, "pieces": len(pieces), "probes": probes}

    def _stream_tail(self, feed) -> None:
        """After HALTED: the pieces not handed over yet, in one read and one feed."""
        addr, n = self._stream[:2]
        fill = len(self._stream) > 3
        mark = FILL_SENTINEL if fill else SENTINEL
        pieces = self._pieces()
        i = self.last_stream.get("during", 0)
        o = pieces[i][0]
        t0 = time.perf_counter()
        # HALTED rises once the last store has left the core, not when the DRAM has it: its
        # last beats may still be on the way (WR_IDLE clear), and a read may pass them. Wait
        # for WR_IDLE, then read; words still unwritten are read again for a while.
        t = self.board.t
        while not t.reg_read(R_STATUS) & R.ST_WR_IDLE and time.perf_counter() - t0 < TAIL_SETTLE:
            pass
        w = self.board.read(addr + o, n - o).view(np.uint32)
        tries = 0
        while (w == mark).any() and time.perf_counter() - t0 < TAIL_SETTLE:
            tries += 1
            time.sleep(50e-6)
            w = self.board.read(addr + o, n - o).view(np.uint32)
        if (w == mark).any():
            self._armed = None
            bad = np.flatnonzero(w == mark)
            raise RuntimeError(f"streamed logits: the run left words unwritten ({len(bad)}, "
                               f"bytes {o + 4 * bad[0]}..{o + 4 * bad[-1] + 3} of the region, "
                               f"{tries} re-reads over {1e3 * TAIL_SETTLE:.0f} ms): the program "
                               "does not write the whole logits region, or a marking raced it")
        self.last_stream["tail_retries"] = tries
        feed(o, w)                                  # the rest in one piece
        if fill:                                    # the next run fills it again
            self._stream = None
            self.last_stream.update(tail_bytes=n - o, tail_s=time.perf_counter() - t0)
            return
        # every piece marked again, on the DMA worker while the host takes the token (the run
        # has ended: whole calls); start() and write() wait for it
        if len(self._marks) != n // 4:
            self._marks = np.full(n // 4, SENTINEL, np.uint32)
        if n <= PIPE:                               # (a larger write pipelines on the worker)
            self._rearm = self.board._worker().submit(self.board.write, addr, self._marks)
        else:
            self.board.write(addr, self._marks)
        self._stream = None
        self.last_stream.update(tail_bytes=n - o, tail_s=time.perf_counter() - t0)

    def _settle(self) -> None:
        """Wait for the marking of the streamed logits' region (_stream_tail's), raising its
        error."""
        f, self._rearm = self._rearm, None
        if f is not None:
            f.result()

    def run(self, programs: list) -> dict:
        self.start(programs)
        return self.wait()

    # ---- the decode loop on the card (Engine.generate_card)
    @property
    def generates(self) -> bool:
        """The bitstream runs the generate programs: RLD, VOP ARGMAX, HALT CHAIN (CAPS bit30)."""
        return bool((self.info.get("caps") or {}).get("gen"))

    chains = generates

    def run_generate(self, programs, out: int, n: int, on_token=None, stop=None,
                     state: int | None = None):
        """Start a generate program and hand over its tokens as they land in out[] (up to n
        fp32 ids from byte address `out`, OUT_MARK until written) while the card runs; stop()
        true writes the state block's stop word (at `state`): the card halts after the token in
        flight. Returns (the run's counters, the tokens). The host reads 64-byte beats of out[]
        only, never the logits. The stop word's beat holds the host's words only (generate.py's
        state block), and Board.write writes that beat alone: the card's state stores beside it
        are never rewritten. A run that hands over no token for RUN_TIMEOUT seconds is an error;
        on any error (KeyboardInterrupt, SystemExit too) the stop word goes out and the run gets
        STOP_WAIT to halt after its token before it is cut short (Board.stop). With the host's
        hook (self.host: a streamed MoE's servers) it is served until the card halts, past the
        last token: the MTP loop's last D runs the MTP layer after the verify that wrote it (its
        MoE request, its two rows' post) and waits for the answers (session mtp1, 2026-10-14:
        served only up to the n-th token, the run waited on its request until RUN_TIMEOUT)."""
        from opentpu.llm import generate as G
        b, t = self.board, self.board.t
        self.start(programs)
        got, k, asked = [], 0, False
        gap = self._gen_gaps.get(self._key, 0.0) or 0.002   # seconds per token
        t_tok = time.perf_counter()

        def take() -> int:
            nonlocal k, t_tok
            a = out + 4 * k
            m = min(n - k, (64 - a % 64) // 4 + 16)            # to the next beat's end
            w = b.read(a, 4 * m).view(np.uint32)
            new = w[:int(np.argmax(w == G.OUT_MARK))] if (w == G.OUT_MARK).any() else w
            for x in new.view(np.float32):
                got.append(int(x))
                if on_token is not None:
                    on_token(int(x))
            k += len(new)
            if len(new):
                t_tok = time.perf_counter()
            return len(new)

        # the board model replays its register script in one simulation per flush: no reads
        # while the program runs, the tokens are read after it halts
        def halt() -> None:     # every slice's stop word (the generate loop gathers them at S > 1)
            for s in range(b.slices):
                b.write(state + 4 * G.S_HALT, np.ones(1, np.float32), s)

        host, t_take, idle = self.host, 0.0, getattr(t, "host_idle", HOST_IDLE)
        try:
            while (k < n or host is not None) and not getattr(t, "batched", False):
                served = host() if host is not None else 0
                if k < n and (host is None or time.perf_counter() - t_take >= HOST_TAKE):
                    t_take = time.perf_counter()
                    if take():
                        continue
                if t.reg_read(R_STATUS) & ST_HALTED:
                    break
                if time.perf_counter() - t_tok > RUN_TIMEOUT:
                    raise TimeoutError(f"the generate run handed over no token in "
                                       f"{RUN_TIMEOUT:.0f} s ({k} of {n} so far)" if k < n else
                                       f"the generate run did not halt in {RUN_TIMEOUT:.0f} s "
                                       f"after its last token, served all along")
                if stop is not None and not asked and state is not None and stop():
                    halt()
                    asked = True
                if served:
                    continue
                if host is not None:            # the card's MoE layers wait for the hook
                    if idle:
                        time.sleep(idle)
                    continue
                # half the time to the next token's due (the last run's pace), GEN_POLL_FAR at
                # most, then every 50 us: few reads of the card while a token is far, none late
                due = t_tok + gap - time.perf_counter()
                time.sleep(min(max(due * 0.5, 5e-5), GEN_POLL_FAR))
            st = b.wait(expect=0.0)
        except BaseException:
            self._running = None
            if b.in_run and state is not None and not asked:
                try:                            # (the card halts after the token in flight)
                    halt()
                except Exception:               # noqa: BLE001 (the error raised is the one)
                    pass
            b.abort(STOP_WAIT)
            raise
        self._running = None
        while k < n and take():                            # the last tokens, after HALTED
            pass
        khz = self.info["core_khz"]
        if got and khz:
            self._gen_gaps[self._key] = st["cycles"] / (khz * 1e3) / len(got)
        self.last = (programs, st)
        if self.status is not None:
            self.status.token(st["cycles"], khz, dram=self._layout(True))
        return st, got

    def close(self) -> None:
        if self.status is not None:
            self.status.remove()
            self.status = None
        self.board.close()
