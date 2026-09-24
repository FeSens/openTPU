"""FakeTransport: an in-memory card for tests and demos (otpu-smi --fake).

It implements the register protocol the driver uses -- register map 2 (or 1, regmap=1: 8-bit
address decode, 0xDEADBEEF for registers v1 does not have) -- over two channel memories in RAM.
It computes nothing: a run takes `run_s` seconds of wall time (HALTED rises then), reports
`cycles` and retires PROG_N instructions; the DRAM keeps what the host wrote.

The free-running counters advance deterministically: every SNAP adds `step` cycles of uptime
and RATES x step to the others, so two snapshots give exactly RATES as utilization (and a DRAM
bandwidth of (DRAM_RD + DRAM_WR rates) x 64 B x CORE_KHZ). The trace buffer serves `trace`
(a list of 64-bit records) with TRACE_COUNT = len(trace) + `trace_extra` (records that fell
out of the buffer) and TRACE_DROP = `trace_drop`.
"""
from __future__ import annotations

import time

import numpy as np

from . import regs as R

RATES = {"RUNNING": 0.80, "MXU_BUSY": 0.60, "MXU_MAC": 0.50, "VPU_BUSY": 0.12,
         "QNT_BUSY": 0.05, "DMA_BUSY": 0.03, "TMEM_DENY": 0.01, "DRAM_RD": 0.70,
         "DRAM_WR": 0.02, "DRAM_WAIT": 0.20, "INSTR": 0.005}


class FakeTransport:
    batched = False

    def __init__(self, ch_bytes: int = 1 << 20, regmap: int = 2, devname: str | None = "fake0",
                 run_s: float = 0.0, cycles: int = 1_000_000, core_khz: int = 100_000,
                 build_id: int = 0x1234ABCD, temp_code: int = 0x9C4, trace_log2: int = 12,
                 step: int = 1_000_000, trace: list[int] | None = None, trace_extra: int = 0,
                 trace_drop: int = 0, D: int = 128, MCOLS: int = 2, LANES: int = 8):
        self.ch = [np.zeros(ch_bytes, np.uint8) for _ in range(2)]
        self.v, self.devname, self.dev = regmap, devname, devname and f"/dev/{devname}"
        self.run_s, self.cycles_per_run = run_s, cycles
        self.core_khz, self.build_id, self.temp_code = core_khz, build_id, temp_code
        self.trace_log2, self.step = trace_log2, step
        self.trace, self.trace_extra, self.trace_drop = list(trace or []), trace_extra, trace_drop
        self.version = D << 16 | MCOLS << 8 | LANES
        self.regs = {R.R_CTRL: 0, R.R_PROG_ADDR: 0, R.R_PROG_N: 0, R.R_SCRATCH: 0,
                     R.R_TRACE_CTRL: 0, R.R_TRACE_ADDR: 0}
        self.count = {k: 0 for k in R.COUNTERS}
        self.shadow = dict(self.count)
        self.snaps = 0
        self.t_run = None               # wall time RUN rose
        self.runs = 0
        self.reads = 0                  # register reads (poll cost)

    # ---- memory
    def mem_write(self, ch, off, data):
        self.ch[ch][off:off + len(data)] = data

    def mem_read(self, ch, off, n, out=None):
        if out is None:
            return self.ch[ch][off:off + n].copy()
        out[:] = self.ch[ch][off:off + n]
        return out

    # ---- registers
    def _halted(self) -> bool:
        return self.t_run is not None and time.perf_counter() - self.t_run >= self.run_s

    def reg_write(self, off, val):
        if self.v < 2:
            off &= 0xFF
            if off >= R.R_REGMAP:
                return
        if off == R.R_CTRL:
            if val & R.CTRL_RUN and not self.regs[R.R_CTRL] & R.CTRL_RUN:
                self.t_run = time.perf_counter()
                self.runs += 1
            elif not val & R.CTRL_RUN:
                self.t_run = None
        if off == R.R_SNAP:
            self.snaps += 1
            for k in self.count:
                self.count[k] += self.step if k == "UPTIME" else int(RATES[k] * self.step)
            self.shadow = dict(self.count)
            return
        if off == R.R_TRACE_CTRL and val & R.TR_CLEAR:
            return
        self.regs[off] = val & 0xFFFFFFFF

    def reg_read(self, off):
        self.reads += 1
        if self.v < 2:
            off &= 0xFF                                  # v1 decodes 8 address bits
            if off >= R.R_REGMAP:
                return R.UNMAPPED
        if off == R.R_ID:
            return R.ID_OTPU
        if off == R.R_VERSION:
            return self.version
        if off == R.R_STATUS:
            run = bool(self.regs[R.R_CTRL] & R.CTRL_RUN)
            return (R.ST_HALTED if run and self._halted() else 0) | R.ST_CALIB0 | R.ST_CALIB1 \
                | R.ST_WR_IDLE | (R.ST_RUN if run else 0)
        if off in (R.R_CYCLES, R.R_CYCLES_HI):
            c = self.cycles_per_run if self.runs else 0
            return c & 0xFFFFFFFF if off == R.R_CYCLES else c >> 32
        if off == R.R_ICOUNT:
            return self.regs[R.R_PROG_N] if self.runs else 0
        if off in (R.R_B_RD, R.R_B_WR, R.R_A_RD, R.R_A_WR, R.R_B_STALL):
            return 0
        if off == R.R_REGMAP:
            return 2
        if off == R.R_CAPS:
            return R.CAP_TRACE | R.CAP_TEMP | self.trace_log2 << 8 | 6 << 16
        if off == R.R_CORE_KHZ:
            return self.core_khz
        if off == R.R_BUILD_ID:
            return self.build_id
        if off == R.R_TEMP:
            return R.TEMP_VALID | self.temp_code
        if off == R.R_SNAP:
            return self.snaps
        for k, o in R.COUNTERS.items():
            if off in (o, o + 4):
                v = self.shadow[k]
                return v & 0xFFFFFFFF if off == o else v >> 32
        if off == R.R_TRACE_COUNT:
            return len(self.trace) + self.trace_extra
        if off == R.R_TRACE_DROP:
            return self.trace_drop
        if off in (R.R_TRACE_LO, R.R_TRACE_HI):
            a = self.regs[R.R_TRACE_ADDR]
            rec = self.trace[a] if a < len(self.trace) else 0
            if off == R.R_TRACE_HI:
                self.regs[R.R_TRACE_ADDR] = (a + 1) % (1 << self.trace_log2)
                return rec >> 32
            return rec & 0xFFFFFFFF
        return self.regs.get(off, R.UNMAPPED)

    def reg_read_many(self, offs):
        return [self.reg_read(o) for o in offs]

    def poll(self, off, mask, val, timeout=600.0):
        from .board import XdmaTransport
        return XdmaTransport.poll(self, off, mask, val, timeout)
