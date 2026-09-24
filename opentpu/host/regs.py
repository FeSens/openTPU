"""The control register map of the board (docs/observability.md; otpu_ctrl.sv, otpu_trace.sv).

Byte offsets from BAR0, 32-bit registers. Version 1 bitstreams implement 0x00..0x38 only and
decode 8 address bits: an unknown register reads 0xDEADBEEF, and an offset >= 0x100 aliases
into 0x00..0xFF (so the counters and the trace registers must not be read before REGMAP says
2). REGMAP (0x3C) reads 0xDEADBEEF on version 1; 0 is treated the same (a map with the
register tied off).
"""
from __future__ import annotations

# ---- version 1 (the offsets are kept in version 2)
R_ID, R_VERSION, R_CTRL, R_STATUS = 0x00, 0x04, 0x08, 0x0C
R_PROG_ADDR, R_PROG_N, R_CYCLES, R_CYCLES_HI, R_ICOUNT = 0x10, 0x14, 0x18, 0x1C, 0x20
R_B_RD, R_B_WR, R_A_RD, R_A_WR, R_B_STALL, R_SCRATCH = 0x24, 0x28, 0x2C, 0x30, 0x34, 0x38
R_SW_WR = R_A_WR                 # the v2 name: scalar (QST) write requests
CTRL_RUN, CTRL_LOAD, CTRL_CLEAR = 1, 2, 4
ST_HALTED, ST_ERROR, ST_LOADING, ST_WR_IDLE, ST_AXI_ERR = 1, 2, 4, 8, 16
ST_CALIB0, ST_CALIB1, ST_RUN = 32, 64, 128
ID_OTPU = 0x4F545055
UNMAPPED = 0xDEADBEEF            # what version 1 returns for a register it does not have

# ---- version 2
R_REGMAP, R_CAPS, R_CORE_KHZ, R_BUILD_ID, R_TEMP, R_SNAP = 0x3C, 0x40, 0x44, 0x48, 0x4C, 0x50
CAP_TRACE, CAP_TEMP = 1, 2       # CAPS bit0, bit1; [15:8] log2 trace depth, [23:16] log2 P/Q window
TEMP_VALID = 1 << 31

# free-running 64-bit counters: shadows latched by a SNAP write; low word at the offset
COUNTERS = {"UPTIME": 0x100, "RUNNING": 0x108, "MXU_BUSY": 0x110, "MXU_MAC": 0x118,
            "VPU_BUSY": 0x120, "QNT_BUSY": 0x128, "DMA_BUSY": 0x130, "TMEM_DENY": 0x138,
            "DRAM_RD": 0x140, "DRAM_WR": 0x148, "DRAM_WAIT": 0x150, "INSTR": 0x158}
EVENTS = ("DRAM_RD", "DRAM_WR", "INSTR")    # count events, not cycles
DRAM_BEAT = 64                              # bytes per DRAM_RD / DRAM_WR event

# trace buffer
R_TRACE_CTRL, R_TRACE_COUNT, R_TRACE_DROP = 0x200, 0x204, 0x208
R_TRACE_ADDR, R_TRACE_LO, R_TRACE_HI = 0x20C, 0x210, 0x214
TR_ENABLE, TR_CLEAR, TR_STOP_WHEN_FULL = 1, 2, 4


def temp_c(code: int) -> float:
    """XADC die-temperature code (12 bits) -> degrees Celsius (UG480)."""
    return (code & 0xFFF) * 503.975 / 4096 - 273.15


def caps(v: int) -> dict:
    return {"trace": bool(v & CAP_TRACE), "temp": bool(v & CAP_TEMP),
            "trace_depth": 1 << ((v >> 8) & 0xFF) if v & CAP_TRACE else 0,
            "pq_window": 1 << ((v >> 16) & 0xFF)}


def regmap(v: int) -> int:
    """REGMAP register value -> map version (1 when the register is missing)."""
    return 1 if v in (UNMAPPED, 0) else v
