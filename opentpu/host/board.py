"""Host driver for the openTPU board (YPCB-00338 over PCIe, Xilinx XDMA).

The card exposes, through the XDMA bridge:
  - its two DDR3 channels on the bridge's AXI master: channel c at BASE[c] (2 GiB each),
    written with /dev/xdma0_h2c_0 and read with /dev/xdma0_c2h_0 (file offset = AXI address);
  - the control registers (rtl/boards/ypcb-00338/otpu_ctrl.sv; map in regs.py and
    docs/observability.md) on BAR0, /dev/xdma0_user.

The accelerator addresses one logical DRAM interleaved over the channels in 64-byte beats:
logical beat b of chunk m = b // 2 lives at BASE[c] + m * 64 on channel c = b % 2, or, on a
bitstream with CAPS.chash, c = (b % 2) ^ parity(m) (rtl/mem/otpu_native_dram.sv). This driver
applies the same map, so the host works with logical addresses only.

BoardBackend implements the Engine backend interface (write / read / run, plus prepare and
attach), so `Engine(..., cfg=device_config(board.info()), backend=BoardBackend)` runs Qwen3,
LFM2 or Qwen3.5 on the card; device_config takes MCOLS and LANES from the bitstream's VERSION
register. With transport=SimTransport the identical protocol runs against the Verilator model
of the board (sim/verilator/tb_board.sv) -- the bring-up rehearsal.

A Board takes the device's exclusive lock (runstate.DeviceLock, /tmp/otpu/<dev>.lock) when its
transport names a device (XdmaTransport, FakeTransport); monitors pass lock=False. Register map
version 1 bitstreams (no REGMAP register) work for everything but the counters, the trace and
the temperature: info() reports regmap 1 and snapshot() returns None.
"""
from __future__ import annotations

import concurrent.futures
import mmap
import os
import struct
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import memcal
from . import regs as R
from .regs import *  # noqa: F401,F403  (the v1 names stay importable from here)
from .regs import (CTRL_CLEAR, CTRL_LOAD, CTRL_RUN, ID_OTPU, R_CTRL, R_CYCLES, R_CYCLES_HI,
                   R_ICOUNT, R_ID, R_PROG_ADDR, R_PROG_N, R_STATUS, ST_AXI_ERR, ST_CALIB0,
                   ST_CALIB1, ST_ERROR, ST_HALTED, ST_LOADING, ST_RUN)
from .runstate import DeviceLock, RunnerStatus

BEAT = 64                       # bytes per interleave beat
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
PIPE = 2 * DMA_CHUNK            # Board.write / read: logical bytes per pipelined piece
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
STREAM_EARLY = 0.3e-3           # probe a piece this long before it came complete last token
STREAM_RETRY = 0.1e-3           # probe again after this when it is not complete yet
STREAM_PROBE = 0.5e-3           # probe interval when no completion time is known yet
TAIL_SETTLE = 5e-3              # after HALTED: how long the last stores may take to land
WR_SETTLE = 2e-3                # after HALTED: the longest wait for WR_IDLE


# ------------------------------------------------------------------------------ address map
def swapped(addr: int, n: int) -> np.ndarray:
    """CHASH: which of the n chunks from logical `addr` (chunk aligned) have their halves
    swapped between the channels (the parity of the chunk index)."""
    m = np.arange(addr // (2 * BEAT), addr // (2 * BEAT) + n, dtype=np.uint64)
    for s in (32, 16, 8, 4, 2, 1):
        m ^= m >> np.uint64(s)
    return (m & np.uint64(1)).astype(bool)


def hash_swap(addr: int, v: np.ndarray) -> None:
    """Chunks [n, 2, BEAT] at logical `addr`: swap the halves CHASH swaps, in place (its own
    inverse: logical order <-> channel order)."""
    p = swapped(addr, len(v))
    v[p] = v[p][:, ::-1]


def split(addr: int, data: np.ndarray, chash: bool = False) -> list[tuple[int, int, np.ndarray]]:
    """Logical bytes at `addr` -> [(channel, channel offset, bytes)], beat-aligned pieces
    merged into one contiguous run per channel. `addr` and len(data) must be multiples of
    2 * BEAT (the caller widens unaligned ranges)."""
    assert addr % (2 * BEAT) == 0 and len(data) % (2 * BEAT) == 0
    v = data.reshape(-1, 2, BEAT)
    if chash:
        v = v.copy()
        hash_swap(addr, v)
    off = addr // 2
    return [(c, off, np.ascontiguousarray(v[:, c, :]).reshape(-1)) for c in (0, 1)]


def join(parts: list[np.ndarray], addr: int = 0, chash: bool = False) -> np.ndarray:
    """Inverse of split: the two channels' contiguous runs (from logical `addr`) -> logical
    bytes."""
    a, b = (p.reshape(-1, BEAT) for p in parts)
    v = np.stack([a, b], axis=1)
    if chash:
        hash_swap(addr, v)
    return v.reshape(-1)


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


class XdmaTransport:
    """The card through the Xilinx XDMA driver (dma_ip_drivers/XDMA/linux-kernel).
    dma=False opens only the register BAR (monitors: no DMA channel is touched)."""
    ecc = True                  # the card's DRAM needs Board.scrub after configuration
    threaded = True             # Board may run the DMA calls in a worker thread (Board.write/read)
    streams = True              # DMA while the accelerator runs (BoardBackend streamed logits)

    def __init__(self, dev: str = "/dev/xdma0", dma: bool = True):
        self.dev, self.devname = dev, Path(dev).name
        # The device lock comes before any open: a process waiting for its turn
        # (OTPU_LOCK_WAIT) must hold no file on the card, or `otpu-setup --rescan` of the
        # holder, which refuses while the device is open, is blocked by the queue.
        self._otpu_lock = DeviceLock(self.devname) if dma else None
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
        # through its four 64 KiB windows: the control registers (their first 4 KiB), the
        # LiteDRAM CSRs (R_MEMCAL: memcal; the MIG bitstreams' ECC registers), and the XADC
        # (0x30000)
        self.regs = mmap.mmap(fd, 0x40000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
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
        base, pos = BASE[ch] + off, 0
        while pos < len(a):
            k = min(DMA_CHUNK, len(a) - pos)
            src = a[pos:pos + k]
            if not _write_ok(_addr(src), base + pos):       # bounce: 0.77 -> 1.7 GB/s
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
        base = BASE[ch] + off
        if out is None:     # small reads are latency-bound: no mmap for them
            buf = placed(n, base) if n >= 64 << 10 else np.empty(n, np.uint8)
        else:
            buf = out
        mv = memoryview(buf).cast("B")
        direct = n < 64 << 10 or _read_ok(_addr(buf), base)
        pos = 0
        while pos < n:
            want = min(DMA_CHUNK, n - pos)
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


class SimTransport:
    """The Verilator model of the board (sim/verilator/tb_board.sv): the memory lives here as
    the two channels' physical images, register operations are queued and replayed by the
    testbench when a result is needed (a flush).

    Every flush is a fresh simulation: DRAM persists (through the channel image files), but the
    control registers, IMEM and TMEM start from reset. The Board protocol is built for that --
    a program load, its run and the reads of its counters happen in one flush -- and the Qwen3
    step programs do not rely on TMEM surviving between runs.

    Batched use (batched = True): queue_read() queues a read and returns its index in the next
    flush's `results` (reads keep their order and may repeat an address); wait_cycles(n) queues
    the testbench's `C` command (wait n core cycles). Together they sample the free-running
    counters twice in one simulation (otpu-smi --sim) and read the trace buffer out in the flush
    that ran the program."""

    batched = True
    devname = None              # private to this process: no device lock
    keeps_state = False         # IMEM and the ARG registers start from reset every flush

    def __init__(self, ch_bytes: int = 1 << 24, stall: int = 20, seed: int = 1,
                 params: dict | None = None, plusargs: list | None = None,
                 native: bool | str | None = None):
        self.ch = [np.zeros(ch_bytes, np.uint8) for _ in range(2)]
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

    def mem_write(self, ch: int, off: int, data: np.ndarray) -> None:
        self.flush()
        self.ch[ch][off:off + len(data)] = data

    def mem_read(self, ch: int, off: int, n: int, out: np.ndarray | None = None) -> np.ndarray:
        self.flush()
        if out is None:
            return self.ch[ch][off:off + n].copy()
        out[:] = self.ch[ch][off:off + n]
        return out

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

    def flush(self) -> None:
        if not self.script:
            return
        from opentpu import rtlsim
        root = Path(__file__).resolve().parents[2]
        srcs = [rtlsim.RTL / s for s in rtlsim.RTL_SOURCES if not s.endswith("otpu_top.sv")]
        board = root / "rtl/boards/ypcb-00338"
        srcs += [board / "otpu_ctrl.sv", board / "otpu_board.sv"]
        if (board / "otpu_trace.sv").exists():             # register map 2
            srcs.insert(-2, board / "otpu_trace.sv")
        native = rtlsim.MEMORY["NATIVE"] if self.native is None else self.native
        srcs += [rtlsim.TB / "otpu_native_mem.sv", board / "otpu_afifo.sv", board / "otpu_mem_ch.sv",
                 rtlsim.TB / "otpu_chmem.sv", rtlsim.TB / "otpu_ldn_model.sv"]
        srcs += [rtlsim.TB / "tb_board.sv"]
        from opentpu.isasim import board_config
        cfg = board_config()                        # OTPU_MCOLS / OTPU_LANES: the "bitstream"
        # VPU_CL and ULANES as the bitstream builds them (make bit: VPU_CL 2, ULANES 8)
        p = {"WORDS": 2 * len(self.ch[0]) // 4, "MCOLS": cfg.MCOLS, "LANES": cfg.LANES,
             "ACT_ROWS": cfg.act_rows,
             "VPU_CL": rtlsim.UARCH.get("VPU_CL", 2), "ULANES": rtlsim.UARCH.get("ULANES", 8),
             "MXU_IMPL": rtlsim.UARCH.get("MXU_IMPL", 2),
             "MEM_NATIVE": 2 if native == "ld" else 1}
        p.update(self.params)
        exe = rtlsim.build("tb_board", srcs, p)
        with tempfile.TemporaryDirectory(prefix="otpu_board_") as d:
            d = Path(d)
            for c in (0, 1):
                self.ch[c].view("<u4").astype(">u4").tofile(d / f"ch{c}.bin")
            (d / "host.txt").write_text("\n".join(self.script) + "\n")
            self.script, self.nreads = [], 0
            r = rtlsim.run_sim([str(exe), f"+dir={d}", f"+axi_stall={self.stall}",
                                f"+axi_seed={self.seed}", *self.plusargs])
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
            for c in (0, 1):
                self.ch[c] = np.fromfile(d / f"ch{c}_out.bin", np.uint8)


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


def rates(a: dict, b: dict, core_khz: int | None) -> dict:
    """Two snapshots -> utilization of every cycle counter (delta / UPTIME delta), DRAM GB/s
    (beats x 64 B over the device time, which needs CORE_KHZ) and instructions per cycle."""
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
        out["dram_rd_gbs"] = d["DRAM_RD"] * R.DRAM_BEAT / sec / 1e9
        out["dram_wr_gbs"] = d["DRAM_WR"] * R.DRAM_BEAT / sec / 1e9
        out["dram_gbs"] = out["dram_beats"] * R.DRAM_BEAT / sec / 1e9
    return out


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
        self._pool = None           # the DMA worker thread (large reads / writes on the card)
        if check:
            ident = self.t.reg_read(R_ID)
            if ident != ID_OTPU:
                self.close()
                raise RuntimeError(f"no openTPU on the card (ID register {ident:#x})")
            if calibrate:
                try:
                    memcal.ensure(self.t, log=lambda m: print(m, file=sys.stderr))
                except Exception:
                    self.close()
                    raise

    def close(self) -> None:
        """Release the device lock (for every Board on this transport)."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        if self.lock is not None:
            self.lock.release()
            self.t._otpu_lock = None
        self.lock = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ identity and state
    def info(self) -> dict:
        """D / MCOLS / LANES, STATUS, calibration and, with register map 2: CAPS, CORE_KHZ,
        BUILD_ID, the die temperature and the DDR3 data rate (None where the bitstream lacks
        them)."""
        v, st, rm = self.t.reg_read_many([R.R_VERSION, R_STATUS, R.R_REGMAP])
        d = {"D": v >> 16, "MCOLS": (v >> 8) & 0xFF, "LANES": v & 0xFF, "status": st,
             "calib": [bool(st & ST_CALIB0), bool(st & ST_CALIB1)],
             "calibrated": bool(st & ST_CALIB0) and bool(st & ST_CALIB1),
             "running": bool(st & ST_RUN) and not st & ST_HALTED,
             "regmap": R.regmap(rm), "caps": None, "core_khz": None, "build_id": None,
             "temp_c": None, "ddr_mts": None, "act_rows": (v >> 8) & 0xFF}
        if d["regmap"] >= 2:
            cp, khz, bid, tp, mts = self.t.reg_read_many([R.R_CAPS, R.R_CORE_KHZ, R.R_BUILD_ID,
                                                          R.R_TEMP, R.R_DDR_MTS])
            d.update(caps=R.caps(cp), core_khz=khz or None, build_id=bid,
                     temp_c=round(R.temp_c(tp), 2) if cp & R.CAP_TEMP and tp & R.TEMP_VALID
                     else None,
                     ddr_mts=(mts or None) if cp & R.CAP_DDR else None)
            if cp & R.CAP_ACT:
                d["act_rows"] = self.t.reg_read(R.R_ACT_ROWS)
        self._info = d
        return d

    @property
    def chash(self) -> bool:
        """The bitstream hashes the channel interleave (CAPS bit7)."""
        c = (self._info or self.info())["caps"]
        return bool(c and c.get("chash"))

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
        self.t.reg_write(R.R_SNAP, 1)
        return self.snap_dict(self.t.reg_read_many(self.snap_offs(rm)), rm)

    @staticmethod
    def snap_dict(v: list[int], regmap: int) -> dict:
        """Values read at snap_offs(regmap) -> {counter: 64-bit count, "snaps": n}."""
        d = {n: v[2 * i] | v[2 * i + 1] << 32 for i, n in enumerate(R.counters(regmap))}
        d["snaps"] = v[-1]
        return d

    # ------------------------------------------------------------------ DRAM
    def write(self, addr: int, data) -> None:
        data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        if len(data) == 0:
            return
        a0 = addr // (2 * BEAT) * (2 * BEAT)
        a1 = -(-(addr + len(data)) // (2 * BEAT)) * (2 * BEAT)
        own = a0 != addr or a1 != addr + len(data)      # data is our own buffer
        if own:                                         # widen: read-modify-write the edges
            buf = np.empty(a1 - a0, np.uint8)
            head, tail = addr - a0, a1 - addr - len(data)
            if head:
                buf[:2 * BEAT] = self.read(a0, 2 * BEAT)
            if tail:
                buf[-2 * BEAT:] = self.read(a1 - 2 * BEAT, 2 * BEAT)
            buf[head:head + len(data)] = data
            data = buf
        if self.chash:                                  # channel order (a copy)
            data = data if own else data.copy()
            hash_swap(a0, data.reshape(-1, 2, BEAT))
        if not (getattr(self.t, "threaded", False) and len(data) > PIPE):
            for c, off, part in split(a0, data):
                self.t.mem_write(c, off, part)
            return
        # pieces of PIPE bytes: the worker DMAs piece i while this thread splits piece i + 1
        # into the other pair of buffers (placed for full DMA speed, see DMA_PLACE)
        h = PIPE // 2
        slots = [[placed(h, BASE[c] + a0 // 2) for c in (0, 1)] for _ in range(2)]
        ex, prev = self._worker(), None
        for i, pos in enumerate(range(0, len(data), PIPE)):
            v = data[pos:pos + PIPE].reshape(-1, 2, BEAT)
            parts = []
            for c in (0, 1):
                buf = slots[i % 2][c][:len(v) * BEAT]
                np.copyto(buf.reshape(-1, BEAT), v[:, c, :])
                parts.append((c, a0 // 2 + i * h, buf))
            if prev is not None:
                prev.result()
            prev = ex.submit(lambda ps: [self.t.mem_write(*p) for p in ps], parts)
        prev.result()

    def _worker(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="otpu-dma")
        return self._pool

    def scrub(self, force: bool = False) -> bool:
        """Write the whole DRAM once per configuration. The DDR3 controllers run with ECC, and
        after power-up the check bits are random: a read of a 64-byte beat never written since is
        an uncorrectable error, which hangs the read (a DMA timeout, or a stuck accelerator) instead
        of failing it. Zeros everywhere (~6 s at Gen1 x8), then SCRATCH = DRAM_INIT, which a new
        bitstream resets. Only the card: the simulated boards have no ECC. True when it scrubbed."""
        if not getattr(self.t, "ecc", False):
            return False
        if not force and self.t.reg_read(R.R_SCRATCH) == R.DRAM_INIT:
            return False
        z = placed(64 << 20, 0)             # zeros, placed: no bounce copies
        for c in (0, 1):
            for off in range(0, CH_BYTES, len(z)):
                self.t.mem_write(c, off, z)
        self.t.reg_write(R.R_SCRATCH, R.DRAM_INIT)
        return True

    def read(self, addr: int, n: int) -> np.ndarray:
        a0 = addr // (2 * BEAT) * (2 * BEAT)
        a1 = -(-(addr + n) // (2 * BEAT)) * (2 * BEAT)
        half = (a1 - a0) // 2
        out = np.empty((half // BEAT, 2, BEAT), np.uint8)
        if not (getattr(self.t, "threaded", False) and 2 * half > PIPE):
            for c in (0, 1):                            # each channel's run, interleaved
                out[:, c, :] = self.t.mem_read(c, a0 // 2, half).reshape(-1, BEAT)
        else:
            # pieces of PIPE / 2 bytes per channel: the worker reads piece i + 1 into one pair of
            # buffers while this thread interleaves piece i from the other pair
            h = PIPE // 2
            slots = [[placed(h, BASE[c] + a0 // 2) for c in (0, 1)] for _ in range(2)]

            def get(i: int):
                k = min(h, half - i * h)
                return [self.t.mem_read(c, a0 // 2 + i * h, k, slots[i % 2][c][:k]) for c in (0, 1)]

            npieces = -(-half // h)
            ex = self._worker()
            fut = ex.submit(get, 0)
            for i in range(npieces):
                nxt = ex.submit(get, i + 1) if i + 1 < npieces else None
                rows = slice(i * h // BEAT, (i * h + h) // BEAT)
                for c, part in enumerate(fut.result()):
                    out[rows, c, :] = part.reshape(-1, BEAT)
                fut = nxt
        if self.chash:                                  # channel order -> logical
            hash_swap(a0, out)
        flat = out.reshape(-1)
        return flat if (a0, a1) == (addr, addr + n) else flat[addr - a0:addr - a0 + n].copy()

    # ------------------------------------------------------------------ programs
    def load_program(self, addr: int, words: np.ndarray) -> None:
        """Copy a program into DRAM at `addr` (chunk aligned) and into IMEM."""
        words = np.asarray(words, "<u4")
        self.write(addr, words.view(np.uint8))
        t = self.t
        t.reg_write(R_CTRL, 0)
        t.reg_write(R_PROG_ADDR, addr)
        t.reg_write(R_PROG_N, len(words) // 8)
        t.reg_write(R_CTRL, CTRL_LOAD)
        t.poll(R_STATUS, ST_LOADING, 0)

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

    def run(self, timeout: float = 600.0, trace: dict | None = None,
            expect: float = 0.0) -> dict:
        """Run the loaded program until it halts; returns the counters.

        trace={"keep": "first" | "last"} records the run in the trace buffer (register map 2
        with CAPS.trace): STOP_WHEN_FULL keeps the first DEPTH records, the ring the last. It
        adds stats["trace"]: records (uint64, oldest first), count (TRACE_COUNT), drop
        (TRACE_DROP: events the capture queue lost), depth, keep, wrapped (the ring overwrote
        records) and lost (records written but not in the buffer: count - depth, or 0).
        expect: the run's expected wall time in seconds, a hint for the poll (Transport.poll).

        start() and wait() are the two halves: the host can work between them while the card
        runs (the transport must not be used in between)."""
        self.start(trace)
        return self.wait(timeout, expect)

    def start(self, trace: dict | None = None) -> None:
        """Start the loaded program (run's first half)."""
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
        t.reg_write(R_CTRL, CTRL_RUN)
        self._t_run = time.perf_counter()

    def wait(self, timeout: float = 600.0, expect: float = 0.0) -> dict:
        """Wait for the started program to halt; returns the counters (run's second half).
        expect counts from the start (the host's work in between is taken off)."""
        t = self.t
        if expect:
            expect = max(expect - (time.perf_counter() - self._t_run), 1e-9)
        traced = self._trace is not None
        depth, keep_first = self._trace or (0, True)
        st0 = t.poll(R_STATUS, ST_HALTED, ST_HALTED, timeout, expect)
        self.t_seen = time.perf_counter()           # HALTED seen (BoardBackend's run time)
        # HALTED rises once the last store has left the core, not when the DRAM has it
        # (WR_IDLE: the memory adapter's writes all answered); the host reads the results
        # next, and a read may pass writes in flight: wait for WR_IDLE (bounded; the board
        # model replays its register script in order: no such race)
        while not getattr(t, "batched", False) and not st0 & R.ST_WR_IDLE and \
                time.perf_counter() - self.t_seen < WR_SETTLE:
            st0 = t.reg_read(R_STATUS)
        self.seen_exact = getattr(t, "poll_exact", False)   # else an upper bound
        if traced:                      # the last cycles' events still drain into the buffer
            t.poll(R.R_TRACE_CTRL, R.TR_BUSY, 0, timeout)
        offs = self.RUN_OFFS + ([R.R_TRACE_COUNT, R.R_TRACE_DROP] if traced else [])
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
        stats = {"cycles": lo | hi << 32, "instructions": [ic], "b_reads": brd,
                 "b_writes": bwr, "a_reads": ard, "a_writes": awr, "b_stall": bst,
                 "status": st}
        if traced:
            stats["trace"] = self._trace_out(vals[9], vals[10], depth, keep_first, raw)
            t.reg_write(R.R_TRACE_CTRL, 0)
        t.reg_write(R_CTRL, 0)
        if st & ST_ERROR:
            raise RuntimeError("the program stopped on an illegal instruction")
        if st & ST_AXI_ERR:
            raise RuntimeError("a DRAM access got an AXI error response")
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
        return {"records": raw, "count": count, "drop": drop, "depth": depth,
                "keep": "first" if keep_first else "last", "wrapped": wrapped,
                "lost": max(0, count - depth)}

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
    """The board_config of the bitstream that `info` (Board.info()) describes: MCOLS and LANES
    come from its VERSION register, PAIR from CAPS bit5, DSTEP from bit6 and STREAM from bit26,
    so the card needs no OTPU_MCOLS / OTPU_LANES / OTPU_PAIR / OTPU_DSTEP / OTPU_STREAM. When one is set in the environment it
    must name the bitstream's value (ConfigMismatch otherwise). Keyword arguments set other fields (DRAM_BYTES)."""
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
    cfg = board_config(**{"MCOLS": info["MCOLS"], "LANES": info["LANES"], "PAIR": pair,
                          "DSTEP": dstep, "STREAM": stream,
                          "ACT_ROWS": rows if rows > info["MCOLS"] else 0,
                          **kw})
    if info["D"] != cfg.D:
        raise ConfigMismatch(f"the bitstream has D={info['D']}, the board configuration "
                             f"D={cfg.D}: not a YPCB-00338 openTPU build")
    return cfg


# ------------------------------------------------------------------------------ Engine backend
def sim_config(spec, cap: int, base=None, rows: int | None = None, lookup: bool = False):
    """`base` (default board_config()) with the DRAM cut to what the model needs (power of
    two): the image with I/O rows for `rows` tokens per run (default the Engine's
    PREFILL_ROWS; lookup: with the resident decode's tables, Engine(resident=True)), then the
    program area. The board model's memory, and the ISA reference that runs the same
    layout."""
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import PREFILL_ROWS, has_lookup
    base = base or board_config()
    probe = spec.image(replace(base, DRAM_BYTES=1 << 32), cap, 1, rows or PREFILL_ROWS,
                       **({"lookup": True} if lookup and has_lookup(spec) else {}))
    need = -(-probe.nbytes // 4096) * 4096 + 4 * base.IMEM_WORDS
    return replace(base, DRAM_BYTES=1 << max(22, (need - 1).bit_length()))


def dram_layout(cfg, image_bytes: int, prog_at: int, image=None, poss=None) -> dict:
    """The device DRAM in bytes: the image (weights, norms, I/O area and KV capacity), the
    program area after it, free; with a model Image also the KV cache capacity (LFM2: with the
    convolution state; Qwen3.5: with the convolution and DeltaNet state, which do not grow with
    the position but are counted as if they did) and the part filled at positions `poss` (one
    per sequence)."""
    prog = 4 * cfg.IMEM_WORDS
    d = {"total": cfg.DRAM_BYTES, "image": image_bytes, "weights": image_bytes,
         "kv_capacity": 0, "kv_used": 0, "program": prog,
         "free": max(0, cfg.DRAM_BYTES - prog_at - prog)}
    if image is not None:
        cap, per_seq = image.cap, image.kv_bytes
        d["kv_capacity"] = per_seq * image.batch
        d["weights"] = image_bytes - d["kv_capacity"]
        d["kv_used"] = int(sum(per_seq * min(p, cap) / cap for p in (poss or [0])))
    return d


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
        if cfg.S != 1:
            raise ValueError("the board has one slice: use opentpu.isasim.board_config()")
        self.cfg = cfg
        self.board = Board(transport)
        self.board.scrub()
        info = self.info = self.board.info()
        if (info["D"], info["MCOLS"], info["LANES"], info.get("act_rows") or info["MCOLS"]) != \
                (cfg.D, cfg.MCOLS, cfg.LANES, cfg.act_rows):
            self.board.close()
            raise ConfigMismatch(f"the bitstream is D={info['D']} MCOLS={info['MCOLS']} "
                                 f"LANES={info['LANES']}, the configuration D={cfg.D} "
                                 f"MCOLS={cfg.MCOLS} LANES={cfg.LANES} (use device_config)")
        img = np.asarray(images[0], np.uint8)
        self.image_bytes = len(img)
        self.prog_at = -(-len(img) // 4096) * 4096
        if self.prog_at + 4 * cfg.IMEM_WORDS > cfg.DRAM_BYTES:
            self.board.close()
            raise MemoryError("no room for the program area")
        self.engine = None
        self.trace: dict | None = None
        self.last = None
        self._prep: dict = {}               # id(programs) -> (programs, words), from prepare()
        self._running = None                # the started programs
        self._resident = None               # (programs, words) in IMEM (start() skips the load)
        self._expect = 0.0                  # the run's expected seconds (the poll's hint)
        # the expected seconds per program length: a run is expected to take what the last
        # run of a program of its length took (the first decode run after a prefill chunk,
        # expected to take the chunk's 25 ms, was seen 14 ms late on the card)
        self._expects: dict[int, float] = {}
        self._key = 0
        self._ratio = 1.0                   # run wall time / (CYCLES / CORE_KHZ), smoothed
        self._seen = None                   # HALTED seen by the streamed wait (perf_counter)
        # streamed logits (start(stream=...)): the region holding the sentinel, the pieces to
        # mark again once the next run has started, the pieces' completion times last token
        self.streams = bool(getattr(self.board.t, "streams", False))
        self._armed = None                  # (addr, nbytes, piece) filled with SENTINEL
        self._rearm: list = []
        self._stream = None                 # the running program's (addr, nbytes, piece)
        self._due: dict = {}
        self.last_stream: dict = {}         # the last streamed wait: pieces during the run...
        t = self.board.t
        self.status = RunnerStatus(t.devname, STATUS_INTERVAL, dev=getattr(t, "dev", t.devname),
                                   model=model,
                                   core_khz=info["core_khz"], dram=self._layout()) \
            if status and self.board.lock is not None else None
        self.board.write(0, img)

    def _layout(self, next_token: bool = False) -> dict:
        e = self.engine
        poss = list(getattr(e, "poss", [])) or None
        if poss and next_token:
            poss[0] += 1                    # the run in flight fills the next position
        return dram_layout(self.cfg, self.image_bytes, self.prog_at,
                           getattr(e, "image", None), poss)

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
        a = self._armed
        if a is not None and addr < a[0] + a[1] and a[0] < addr + np.asarray(data).nbytes:
            self._armed = None
        self.board.write(addr, data)

    def read(self, s: int, addr: int, nbytes: int) -> np.ndarray:
        return self.board.read(addr, nbytes)

    def prepare(self, programs: list) -> None:
        """Assemble ahead of run(programs). Called from the Engine's compile thread while
        run() of the previous token goes on: one dict entry per program list (dict operations
        are atomic), dropped by run()."""
        self._prep[id(programs)] = (programs, np.asarray(self.I.assemble(programs[0]),
                                                         np.uint32))

    runs_words = True           # start() takes assembled words too (the Engine's worker process)

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
        wrote the region), else the whole region is marked again right
        after this start, while the run is still far from its LM head. Needs a transport that
        allows DMA during a run (`streams`).

        args: the run's arguments (up to 8 words, ARG0..7; a bitstream with CAPS bit25). The
        program in IMEM stays there: starting the same `programs` object again (a program
        that takes its position as arguments) loads nothing, on a transport that keeps the
        device's state between runs (not SimTransport: it loads again)."""
        if args is not None and not self.args:
            raise ConfigMismatch("this bitstream takes no run arguments (CAPS bit25 clear)")
        res = self._resident
        if res is None or res[0] is not programs or not getattr(self.board.t, "keeps_state",
                                                                  True):
            prep = self._prep.pop(id(programs), None)
            while len(self._prep) > 1:              # stale entries (discarded compiles)
                self._prep.pop(next(iter(self._prep)), None)
            if isinstance(programs, np.ndarray):
                words = programs
            else:
                words = prep[1] if prep is not None and prep[0] is programs else \
                    np.asarray(self.I.assemble(programs[0]), np.uint32)
            if len(words) > self.cfg.IMEM_WORDS:
                raise ValueError("program does not fit IMEM")
            self._resident = None
            self.board.load_program(self.prog_at, words)
            self._resident = (programs, words)
        if args is not None:
            self.board.set_args(list(args) + [0] * (8 - len(args)))
        if stream is not None and self._armed != stream:
            self.board.write(stream[0], np.full(stream[1] // 4, SENTINEL, np.uint32))
            self._armed, self._rearm = stream, []
        elif stream is None:
            self._armed = None                      # the run may write the region
        self._seen = None
        self._key = len(self._resident[1])
        self._expect = self._expects.get(self._key, 0.0)
        self.board.start(trace=self.trace)
        self._running, self._stream = programs, stream
        for a, n in self._rearm:
            self.board.write(a, np.full(n // 4, SENTINEL, np.uint32))
        self._rearm = []

    def wait(self, feed=None) -> dict:
        """Wait for the started program; returns its counters (run's second half). After a
        start(stream=...), feed(offset, words) gets every piece of the logits (byte offset in
        the region, uint32 words), most of them while the run goes on."""
        programs, self._running = self._running, None
        try:
            if self._stream is not None:
                self._stream_logits(feed)
            # halted already (seen by the streamed wait): read the counters, no sleep
            st = self.board.wait(expect=0.0 if self._seen else self._expect)
            if self._stream is not None:
                self._stream_tail(feed)
        except BaseException:
            self._armed, self._stream = None, None  # the region's state is unknown
            raise
        khz = self.info["core_khz"]
        self._expects[self._key] = self._next_expect(st["cycles"] / (khz * 1e3) if khz else 0.0)
        self.last = (programs, st)
        if self.status is not None:
            self.status.token(st["cycles"], khz, dram=self._layout(True))
        return st

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
        addr, n, piece = self._stream
        return [(o, min(piece, n - o)) for o in range(0, n, piece)]

    def _stream_logits(self, feed) -> None:
        """Hand over the pieces before the last one while the run goes on: probe a piece's
        last beat from STREAM_EARLY before its completion time of the last token (or every
        STREAM_PROBE), read it once the beat is written, check every word, mark it again.
        Stops at HALTED; the last piece (written just before it) is left to _stream_tail."""
        t, b = self.board.t, self.board
        addr = self._stream[0]
        pieces, due = self._pieces(), {}
        t0, i, probes, tries = b._t_run, 0, 0, 0
        halted = False
        # from POLL_EARLY (+ 3%) before the expected end on, the slices are short, so a run
        # that ends while a piece is awaited is seen within ~POLL_MIN_SLEEP (1 ms slices: Qwen3,
        # whose next-to-last piece comes near the end, saw HALTED 0.7-0.8 ms late on the card)
        near = self._expect - POLL_EARLY - 0.03 * self._expect if self._expect else float("inf")
        while i < len(pieces) - 1:
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
            o, k = pieces[i]
            last = addr + o + k - BEAT                  # the piece's last beat (64-byte aligned)
            probes += 1
            tries += 1
            beat = t.mem_read(last // BEAT % 2, last // (2 * BEAT) * BEAT, BEAT)
            w = None
            if not (beat.view(np.uint32) == SENTINEL).any():
                w = b.read(addr + o, k).view(np.uint32)
                if (w == SENTINEL).any():               # the beats land out of order: soon
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
            # its SENTINEL marks go back after the next start (with the tail's): a write
            # here kept the host busy when the run ended (HALTED seen up to 0.15 ms late)
            i, tries = i + 1, 0
        self._due.update(due)
        self.last_stream = {"during": i, "pieces": len(pieces), "probes": probes}

    def _stream_tail(self, feed) -> None:
        """After HALTED: the pieces not handed over yet, in one read and one feed."""
        addr, n, _ = self._stream
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
        while (w == SENTINEL).any() and time.perf_counter() - t0 < TAIL_SETTLE:
            tries += 1
            time.sleep(50e-6)
            w = self.board.read(addr + o, n - o).view(np.uint32)
        if (w == SENTINEL).any():
            self._armed = None
            bad = np.flatnonzero(w == SENTINEL)
            raise RuntimeError(f"streamed logits: the run left words unwritten ({len(bad)}, "
                               f"bytes {o + 4 * bad[0]}..{o + 4 * bad[-1] + 3} of the region, "
                               f"{tries} re-reads over {1e3 * TAIL_SETTLE:.0f} ms): the program "
                               "does not write the whole logits region, or a marking raced it")
        self.last_stream["tail_retries"] = tries
        feed(o, w)                                  # the rest in one piece
        self._rearm = [(addr, n)]                  # every piece: marked after the next start
        self._stream = None
        self.last_stream.update(tail_bytes=n - o, tail_s=time.perf_counter() - t0)

    def run(self, programs: list) -> dict:
        self.start(programs)
        return self.wait()

    # ---- the decode loop on the card (Engine.generate_card)
    @property
    def generates(self) -> bool:
        """The bitstream runs the generate programs: RLD, VOP ARGMAX, HALT CHAIN (CAPS bit28)."""
        return bool((self.info.get("caps") or {}).get("gen"))

    chains = generates

    def run_generate(self, programs, out: int, n: int, on_token=None, stop=None,
                     state: int | None = None):
        """Start a generate program and hand over its tokens as they land in out[] (up to n
        fp32 ids from byte address `out`, OUT_MARK until written) while the card runs; stop()
        true writes the state block's stop word (at `state`): the card halts after the token in
        flight. Returns (the run's counters, the tokens). The host reads 64-byte beats of out[]
        only, never the logits."""
        from opentpu.llm import generate as G
        b, t = self.board, self.board.t
        self.start(programs)
        got, k, asked = [], 0, False
        gap = self._expects.get(("gen", self._key), 0.0) or 0.002   # seconds per token
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

        while k < n:
            if take():
                continue
            if t.reg_read(R_STATUS) & ST_HALTED:
                break
            if stop is not None and not asked and state is not None and stop():
                b.write(state + 4 * G.S_HALT, np.ones(1, np.float32))
                asked = True
            # wake up a little before the next token is due, then every 50 us
            due = t_tok + gap - time.perf_counter()
            time.sleep(min(max(due * 0.5, 5e-5), 1e-3))
        st = b.wait(expect=0.0)
        self._running = None
        while k < n and take():                            # the last tokens, after HALTED
            pass
        khz = self.info["core_khz"]
        if got and khz:
            self._expects[("gen", self._key)] = st["cycles"] / (khz * 1e3) / len(got)
        self.last = (programs, st)
        if self.status is not None:
            self.status.token(st["cycles"], khz, dram=self._layout(True))
        return st, got

    def close(self) -> None:
        if self.status is not None:
            self.status.remove()
            self.status = None
        self.board.close()
