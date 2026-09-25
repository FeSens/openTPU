"""Host driver for the openTPU board (YPCB-00338 over PCIe, Xilinx XDMA).

The card exposes, through the XDMA bridge:
  - its two DDR3 channels on the bridge's AXI master: channel c at BASE[c] (2 GiB each),
    written with /dev/xdma0_h2c_0 and read with /dev/xdma0_c2h_0 (file offset = AXI address);
  - the control registers (rtl/boards/ypcb-00338/otpu_ctrl.sv; map in regs.py and
    docs/observability.md) on BAR0, /dev/xdma0_user.

The accelerator addresses one logical DRAM interleaved over the channels in 64-byte beats
(logical beat b lives on channel b % 2 at BASE[b % 2] + (b // 2) * 64; rtl/mem/otpu_axi_dram.sv).
This driver applies the same map, so the host works with logical addresses only.

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

import mmap
import os
import struct
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

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
                                # transfer (8 MiB at ~3 GB/s is 2.7 ms)
POLL_SPIN = 100e-6              # poll: seconds of back-to-back register reads before sleeping
POLL_MAX_SLEEP = 1e-3           # poll: longest sleep between reads


# ------------------------------------------------------------------------------ address map
def split(addr: int, data: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    """Logical bytes at `addr` -> [(channel, channel offset, bytes)], beat-aligned pieces
    merged into one contiguous run per channel. `addr` and len(data) must be multiples of
    2 * BEAT (the caller widens unaligned ranges)."""
    assert addr % (2 * BEAT) == 0 and len(data) % (2 * BEAT) == 0
    v = data.reshape(-1, 2, BEAT)
    off = addr // 2
    return [(c, off, np.ascontiguousarray(v[:, c, :]).reshape(-1)) for c in (0, 1)]


def join(parts: list[np.ndarray]) -> np.ndarray:
    """Inverse of split: the two channels' contiguous runs -> logical bytes."""
    a, b = (p.reshape(-1, BEAT) for p in parts)
    return np.stack([a, b], axis=1).reshape(-1)


# ------------------------------------------------------------------------------ transports
def _readinto(fd: int, mv: memoryview, off: int) -> int:
    if hasattr(os, "preadv"):
        return os.preadv(fd, [mv], off)
    b = os.pread(fd, len(mv), off)                  # no preadv (old macOS): one extra copy
    mv[:len(b)] = b
    return len(b)


class XdmaTransport:
    """The card through the Xilinx XDMA driver (dma_ip_drivers/XDMA/linux-kernel).
    dma=False opens only the register BAR (monitors: no DMA channel is touched)."""

    def __init__(self, dev: str = "/dev/xdma0", dma: bool = True):
        self.dev, self.devname = dev, Path(dev).name
        self.h2c = os.open(f"{dev}_h2c_0", os.O_WRONLY) if dma else -1
        self.c2h = os.open(f"{dev}_c2h_0", os.O_RDONLY) if dma else -1
        fd = os.open(f"{dev}_user", os.O_RDWR | os.O_SYNC)
        self.regs = mmap.mmap(fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)

    def mem_write(self, ch: int, off: int, data: np.ndarray) -> None:
        mv = memoryview(np.ascontiguousarray(data, np.uint8)).cast("B")
        pos = 0
        while pos < len(mv):
            n = os.pwrite(self.h2c, mv[pos:pos + DMA_CHUNK], BASE[ch] + off + pos)
            if n <= 0:
                raise IOError("XDMA h2c write failed")
            pos += n

    def mem_read(self, ch: int, off: int, n: int, out: np.ndarray | None = None) -> np.ndarray:
        """n bytes of channel `ch` at `off`, DMA'd straight into `out` (or a new buffer)."""
        buf = np.empty(n, np.uint8) if out is None else out
        mv = memoryview(buf).cast("B")
        pos = 0
        while pos < n:
            k = _readinto(self.c2h, mv[pos:pos + min(DMA_CHUNK, n - pos)], BASE[ch] + off + pos)
            if k <= 0:
                raise IOError("XDMA c2h read failed")
            pos += k
        return buf

    def reg_write(self, off: int, val: int) -> None:
        self.regs[off:off + 4] = struct.pack("<I", val & 0xFFFFFFFF)

    def reg_read(self, off: int) -> int:
        return struct.unpack("<I", self.regs[off:off + 4])[0]

    def reg_read_many(self, offs: list[int]) -> list[int]:
        return [self.reg_read(o) for o in offs]

    def poll(self, off: int, mask: int, val: int, timeout: float = 600.0) -> int:
        """Wait until (reg & mask) == val. Back-to-back reads (~1 us each over PCIe) for the
        first POLL_SPIN seconds catch short waits (program loads) without a sleep's latency;
        then sleeps of elapsed/32, at most POLL_MAX_SLEEP: the wake-up comes at most ~3% of
        the run late (a Qwen3-0.6B token is ~50 ms: <= 1 ms, 2%), the register is read about
        32 ln(T / 100 us) + T / 1 ms times instead of T / 1 us, and the sleeps release the
        GIL to the thread compiling the next token's program (Engine pipelining)."""
        t0 = time.perf_counter()
        while True:
            r = self.reg_read(off)
            if r & mask == val:
                return r
            el = time.perf_counter() - t0
            if el > timeout:
                raise TimeoutError(f"register {off:#x} = {r:#x}, waiting for {val:#x}/{mask:#x}")
            if el > POLL_SPIN:
                time.sleep(min(POLL_MAX_SLEEP, el / 32))


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

    def __init__(self, ch_bytes: int = 1 << 24, stall: int = 20, seed: int = 1,
                 params: dict | None = None, plusargs: list | None = None):
        self.ch = [np.zeros(ch_bytes, np.uint8) for _ in range(2)]
        self.stall, self.seed = stall, seed
        self.params = params or {}
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

    def poll(self, off: int, mask: int, val: int, timeout: float = 0) -> int:
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
        srcs += [rtlsim.TB / "otpu_axi_mem.sv", rtlsim.TB / "tb_board.sv"]
        from opentpu.isasim import board_config
        cfg = board_config()                        # OTPU_MCOLS / OTPU_LANES: the "bitstream"
        # VPU_CL and ULANES as the bitstream builds them (make bit: VPU_CL 2, ULANES 8)
        p = {"WORDS": 2 * len(self.ch[0]) // 4, "MCOLS": cfg.MCOLS, "LANES": cfg.LANES,
             "VPU_CL": rtlsim.UARCH.get("VPU_CL", 2), "ULANES": rtlsim.UARCH.get("ULANES", 8)}
        p.update(self.params)
        exe = rtlsim.build("tb_board", srcs, p)
        with tempfile.TemporaryDirectory(prefix="otpu_board_") as d:
            d = Path(d)
            for c in (0, 1):
                self.ch[c].view("<u4").astype(">u4").tofile(d / f"ch{c}.bin")
            (d / "host.txt").write_text("\n".join(self.script) + "\n")
            self.script, self.nreads = [], 0
            r = subprocess.run([str(exe), f"+dir={d}", f"+axi_stall={self.stall}",
                                f"+axi_seed={self.seed}", *self.plusargs],
                               capture_output=True, text=True)
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
    d = {k: b[k] - a[k] for k in R.COUNTERS}
    out = {"cycles": dt, "seconds": sec,
           "util": {k: d[k] / dt if dt else 0.0 for k in R.COUNTERS
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

    def __init__(self, transport=None, check: bool = True, lock: bool = True):
        self.t = transport or XdmaTransport()
        self.lock = _lock(self.t) if lock else None
        self._info = None
        if check:
            ident = self.t.reg_read(R_ID)
            if ident != ID_OTPU:
                self.close()
                raise RuntimeError(f"no openTPU on the card (ID register {ident:#x})")

    def close(self) -> None:
        """Release the device lock (for every Board on this transport)."""
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
        BUILD_ID and the die temperature (None where the bitstream lacks them)."""
        v, st, rm = self.t.reg_read_many([R.R_VERSION, R_STATUS, R.R_REGMAP])
        d = {"D": v >> 16, "MCOLS": (v >> 8) & 0xFF, "LANES": v & 0xFF, "status": st,
             "calib": [bool(st & ST_CALIB0), bool(st & ST_CALIB1)],
             "calibrated": bool(st & ST_CALIB0) and bool(st & ST_CALIB1),
             "running": bool(st & ST_RUN) and not st & ST_HALTED,
             "regmap": R.regmap(rm), "caps": None, "core_khz": None, "build_id": None,
             "temp_c": None}
        if d["regmap"] >= 2:
            cp, khz, bid, tp = self.t.reg_read_many([R.R_CAPS, R.R_CORE_KHZ, R.R_BUILD_ID,
                                                     R.R_TEMP])
            d.update(caps=R.caps(cp), core_khz=khz or None, build_id=bid,
                     temp_c=round(R.temp_c(tp), 2) if cp & R.CAP_TEMP and tp & R.TEMP_VALID
                     else None)
        self._info = d
        return d

    @property
    def v2(self) -> bool:
        return (self._info or self.info())["regmap"] >= 2

    SNAP_OFFS = [o + k for o in R.COUNTERS.values() for k in (0, 4)] + [R.R_SNAP]

    def snapshot(self) -> dict | None:
        """SNAP, then the free-running counters' shadows (one consistent instant): {name: count}
        plus "snaps" (SNAP's read value). None on a register map 1 bitstream."""
        if not self.v2:
            return None
        self.t.reg_write(R.R_SNAP, 1)
        return self.snap_dict(self.t.reg_read_many(self.SNAP_OFFS))

    @staticmethod
    def snap_dict(v: list[int]) -> dict:
        """Values read at SNAP_OFFS -> {counter: 64-bit count, "snaps": n}."""
        d = {n: v[2 * i] | v[2 * i + 1] << 32 for i, n in enumerate(R.COUNTERS)}
        d["snaps"] = v[-1]
        return d

    # ------------------------------------------------------------------ DRAM
    def write(self, addr: int, data) -> None:
        data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        if len(data) == 0:
            return
        a0 = addr // (2 * BEAT) * (2 * BEAT)
        a1 = -(-(addr + len(data)) // (2 * BEAT)) * (2 * BEAT)
        if a0 != addr or a1 != addr + len(data):        # widen: read-modify-write the edges
            buf = np.empty(a1 - a0, np.uint8)
            head, tail = addr - a0, a1 - addr - len(data)
            if head:
                buf[:2 * BEAT] = self.read(a0, 2 * BEAT)
            if tail:
                buf[-2 * BEAT:] = self.read(a1 - 2 * BEAT, 2 * BEAT)
            buf[head:head + len(data)] = data
            data = buf
        for c, off, part in split(a0, data):
            self.t.mem_write(c, off, part)

    def read(self, addr: int, n: int) -> np.ndarray:
        a0 = addr // (2 * BEAT) * (2 * BEAT)
        a1 = -(-(addr + n) // (2 * BEAT)) * (2 * BEAT)
        half = (a1 - a0) // 2
        out = np.empty((half // BEAT, 2, BEAT), np.uint8)
        for c in (0, 1):                                # each channel's run, interleaved
            out[:, c, :] = self.t.mem_read(c, a0 // 2, half).reshape(-1, BEAT)
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

    RUN_OFFS = [R_STATUS, R_CYCLES, R_CYCLES_HI, R_ICOUNT, R.R_B_RD, R.R_B_WR, R.R_A_RD,
                R.R_A_WR, R.R_B_STALL]

    def run(self, timeout: float = 600.0, trace: dict | None = None) -> dict:
        """Run the loaded program until it halts; returns the counters.

        trace={"keep": "first" | "last"} records the run in the trace buffer (register map 2
        with CAPS.trace): STOP_WHEN_FULL keeps the first DEPTH records, the ring the last. It
        adds stats["trace"]: records (uint64, oldest first), count (TRACE_COUNT), drop
        (TRACE_DROP: events the capture queue lost), depth, keep, wrapped (the ring overwrote
        records) and lost (records written but not in the buffer: count - depth, or 0)."""
        t = self.t
        depth, keep_first = 0, True
        if trace is not None:
            i = self._info or self.info()
            if i["regmap"] < 2 or not i["caps"]["trace"]:
                raise RuntimeError("this bitstream has no trace buffer (register map "
                                   f"{i['regmap']}{'' if i['regmap'] < 2 else ', CAPS.trace = 0'})")
            depth = i["caps"]["trace_depth"]
            keep_first = trace.get("keep", "first") == "first"
            t.reg_write(R.R_TRACE_CTRL, R.TR_CLEAR)
            t.reg_write(R.R_TRACE_CTRL, R.TR_ENABLE | (R.TR_STOP_WHEN_FULL if keep_first else 0))
        t.reg_write(R_CTRL, CTRL_CLEAR)
        t.reg_write(R_CTRL, CTRL_RUN)
        t.poll(R_STATUS, ST_HALTED, ST_HALTED, timeout)
        if trace is not None:           # the last cycles' events still drain into the buffer
            t.poll(R.R_TRACE_CTRL, R.TR_BUSY, 0, timeout)
        offs = self.RUN_OFFS + ([R.R_TRACE_COUNT, R.R_TRACE_DROP] if trace is not None else [])
        raw = None
        if trace is not None and getattr(t, "batched", False):
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
        if trace is not None:
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
    come from its VERSION register, so the card needs no OTPU_MCOLS / OTPU_LANES. When either
    is set in the environment it must name the bitstream's value (ConfigMismatch otherwise).
    Keyword arguments set other fields (DRAM_BYTES)."""
    from opentpu.isasim import board_config
    for k in ("MCOLS", "LANES"):
        env = os.environ.get(f"OTPU_{k}")
        if env is not None and int(env) != info[k]:
            raise ConfigMismatch(f"the bitstream was built with {k}={info[k]} but OTPU_{k}={env}"
                                 f": unset OTPU_{k} (the host follows the bitstream) or load "
                                 f"a {k}={env} bitstream")
    cfg = board_config(**{"MCOLS": info["MCOLS"], "LANES": info["LANES"], **kw})
    if info["D"] != cfg.D:
        raise ConfigMismatch(f"the bitstream has D={info['D']}, the board configuration "
                             f"D={cfg.D}: not a YPCB-00338 openTPU build")
    return cfg


# ------------------------------------------------------------------------------ Engine backend
def sim_config(spec, cap: int, base=None):
    """`base` (default board_config()) with the DRAM cut to what the model needs (power of
    two): the image, then the program area. The board model's memory, and the ISA reference
    that runs the same layout."""
    from opentpu.isasim import board_config
    base = base or board_config()
    probe = spec.image(replace(base, DRAM_BYTES=1 << 32), cap)
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
        info = self.info = self.board.info()
        if (info["D"], info["MCOLS"], info["LANES"]) != (cfg.D, cfg.MCOLS, cfg.LANES):
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
        t = self.board.t
        self.status = RunnerStatus(t.devname, dev=getattr(t, "dev", t.devname), model=model,
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
        """Called by the Engine once it exists."""
        self.engine = engine
        if self.status is not None:
            self.status.update(dram=self._layout())

    def write(self, s: int, addr: int, data: np.ndarray) -> None:
        self.board.write(addr, data)

    def read(self, s: int, addr: int, nbytes: int) -> np.ndarray:
        return self.board.read(addr, nbytes)

    def prepare(self, programs: list) -> None:
        """Assemble ahead of run(programs). Called from the Engine's compile thread while
        run() of the previous token goes on: one dict entry per program list (dict operations
        are atomic), dropped by run()."""
        self._prep[id(programs)] = (programs, np.asarray(self.I.assemble(programs[0]),
                                                         np.uint32))

    def run(self, programs: list) -> dict:
        prep = self._prep.pop(id(programs), None)
        while len(self._prep) > 1:                  # stale entries (discarded compiles)
            self._prep.pop(next(iter(self._prep)), None)
        words = prep[1] if prep is not None and prep[0] is programs else \
            np.asarray(self.I.assemble(programs[0]), np.uint32)
        if len(words) > self.cfg.IMEM_WORDS:
            raise ValueError("program does not fit IMEM")
        self.board.load_program(self.prog_at, words)
        st = self.board.run(trace=self.trace)
        self.last = (programs, st)
        if self.status is not None:
            self.status.token(st["cycles"], self.info["core_khz"], dram=self._layout(True))
        return st

    def close(self) -> None:
        if self.status is not None:
            self.status.remove()
            self.status = None
        self.board.close()
