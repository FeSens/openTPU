"""The control register map of the board (docs/observability.md; otpu_ctrl.sv, otpu_trace.sv).

Byte offsets from BAR0, 32-bit registers. Version 1 bitstreams implement 0x00..0x38 only and
decode 8 address bits: an unknown register reads 0xDEADBEEF, and an offset >= 0x100 aliases
into 0x00..0xFF (so the counters and the trace registers must not be read before REGMAP says
2). REGMAP (0x3C) reads 0xDEADBEEF on version 1; 0 is treated the same (a map with the
register tied off). Version 3 adds the MXU_STARVE counter; a version 2 bitstream reads
0xDEADBEEF there, so the host reads only counters(regmap). Version 4 adds the configuration
registers CFG_* (0x80..0x9C: the target the bitstream was built from, opentpu/target.py), from
which the host builds its Config (board.device_config); before it, the YPCB-00338's values.
Version 5 adds CAPS2 (0x5C: CAPS is full); before it every capability CAPS2 names is present.
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
ST_WAIT_TO = 256                 # with ERROR: a WAITW timed out (bitstreams before: ERROR alone)
# a channel's ECC counted an uncorrectable 64-bit word since its counters were last cleared
# (memcal.ecc; LiteDRAM bitstreams from fix-board on, older ones leave it 0): data the card read
# was wrong, and a partial write merged into such a beat wrote it back with good check bits
ST_ECC_DED = 512
ID_OTPU = 0x4F545055
DRAM_INIT = 0x5C2B_ED00          # SCRATCH after Board.scrub (configuration resets it to 0)
UNMAPPED = 0xDEADBEEF            # what version 1 returns for a register it does not have

# ---- version 2
R_REGMAP, R_CAPS, R_CORE_KHZ, R_BUILD_ID, R_TEMP, R_SNAP = 0x3C, 0x40, 0x44, 0x48, 0x4C, 0x50
CAP_TRACE, CAP_TEMP, CAP_I2C = 1, 2, 4   # CAPS bit0..2; [15:8] log2 trace depth,
                                         # [23:16] log2 P/Q window
# CAPS bit3: DDR_MTS holds the DDR3 data rate the bitstream was built for (MT/s). Older
# bitstreams leave the bit clear and read 0xDEADBEEF there: the rate is unknown.
CAP_DDR, R_DDR_MTS = 8, 0x54
# CAPS bit4: the MXU runs 4-bit weights (MM WF); bit5: column reuse (MM PAIR + QACT DUP);
# bit6: DSTEP (the DMA's Gated DeltaNet head step, docs/isa.md)
CAP_W4, CAP_PAIR, CAP_DSTEP = 16, 32, 64
# CAPS bit7: CHASH, the hashed channel interleave (rtl/mem/otpu_native_dram.sv): chunk m's two
# beats swap channels when m has odd parity. Older bitstreams: plain interleave.
CAP_CHASH = 0x80
# CAPS bit24: ACT_ROWS holds the ACT RAM rows (the most rows of one MM; older bitstreams: MCOLS)
CAP_ACT, R_ACT_ROWS = 1 << 24, 0x58
# CAPS bit25: ARG0..7 (0x60 + 4k) are the run's arguments, R8..R15 at the start (docs/isa.md
# "Arguments"): one program serves every decode position
CAP_ARGS, R_ARG0 = 1 << 25, 0x60
# CAPS bit26: the stream engine (docs/stream.md) runs STREAM's hardware subset
# (opentpu.isa.stream_hw_cfg); DSTEP (bit6) runs on it too
CAP_STREAM = 1 << 26
# CAPS bit27: the DDR3 controllers are LiteDRAM (docs/litedram.md): the host calibrates them
# (opentpu/host/memcal.py) through their CSRs in the BAR0 window at R_MEMCAL; STATUS CALIB0/1
# rise when it has. The MIG bitstreams (before LiteDRAM) calibrate themselves.
CAP_HOSTCAL, R_MEMCAL = 1 << 27, 0x10000
# A DDR4 board's LiteDRAM cores (the BCU1525: one per DIMM, calibrated by their own CPUs;
# opentpu/host/ddr4.py): DIMM d's CSRs in BAR0 at R_DDR4 + d x DDR4_STRIDE
R_DDR4, DDR4_STRIDE = 0x40000, 0x10000
# CAPS bit29: the MX numerics (docs/isa.md MM, QACT, QST; docs/mx_mxu.md): MM accumulates each
# output exactly with power-of-two block scales, QACT / QST quantize to a row scale and
# power-of-two offsets. Programs and images of the previous numerics compute garbage on such a
# bitstream and vice versa (the CFG_HASH differs too: opentpu/target.py arch_hash)
CAP_MX = 1 << 29
# CAPS bit30: the decode loop on the card (docs/isa.md RLD, VOP ARGMAX, HALT CHAIN; the
# generate programs of opentpu/llm/generate.py)
CAP_GEN = 1 << 30
CAP_WAITW = 1 << 31              # WAITW: the DMA waits for a word the host writes (docs/isa.md)
TEMP_VALID = 1 << 31

# ---- version 5: a second capability register (CAPS is full)
CAPS2_SINCE, R_CAPS2 = 5, 0x5C
# CAPS2 bit0: the collective unit runs GATHER. Always at S > 1; a one-slice bitstream built
# without it (otpu_board COLL_GATHER = 0, docs/infra_hunt.md 5) runs BAR alone and raises ERROR
# on a GATHER, and the host refuses to load a program with one (Board.load_program). Every
# bitstream before version 5 has GATHER.
CAP2_GATHER = 1


def caps2(v: int | None) -> dict:
    """CAPS2's value -> its capabilities; None (a register map before 5): all present."""
    if v is None:
        return {"gather": True}
    return {"gather": bool(v & CAP2_GATHER)}


# ---- version 4: the configuration (otpu_ctrl CFG_*; docs/targets.md)
CFG_SINCE = 4
R_CFG_SLICES, R_CFG_TMEM, R_CFG_IMEM, R_CFG_ACT = 0x80, 0x84, 0x88, 0x8C
R_CFG_DRAM, R_CFG_UARCH, R_CFG_PLATFORM, R_CFG_HASH = 0x90, 0x94, 0x98, 0x9C
CFG_REGS = (R_CFG_SLICES, R_CFG_TMEM, R_CFG_IMEM, R_CFG_ACT, R_CFG_DRAM, R_CFG_UARCH,
            R_CFG_PLATFORM, R_CFG_HASH)
# reserved: the slices' DRAM windows (0xA0 + 4s), for platforms with more memory per channel than
# a slice addresses (docs/targets.md)
R_CFG_WIN0 = 0xA0
# slice s's ICOUNT at R_ICOUNT_S + 4s on a multi-slice bitstream (CFG_SLICES > 1; ICOUNT is slice
# 0's)
R_ICOUNT_S = 0xC0


def cfg(vals: list[int]) -> dict:
    """The CFG_* registers' values (in CFG_REGS order) -> the configuration they report."""
    s, tmem, imem, act, dram, ua, plat, h = vals
    return {"S": s, "TMEM_WORDS": tmem, "IMEM_WORDS": imem, "ACT_BLOCKS": act,
            "DRAM_BYTES": 1 << (dram & 0xFF), "channels": (dram >> 8) & 0xFF,
            "channel_bytes": 1 << ((dram >> 16) & 0xFF),
            "MXU_IMPL": ua & 0xF, "TMEM_IMPL": (ua >> 4) & 0xF, "VPU_CL": (ua >> 8) & 0xFF,
            "ULANES": (ua >> 16) & 0xFF, "SEQ_WC": (ua >> 24) & 1, "QUAD": (ua >> 25) & 1,
            "platform": plat, "hash": h}


def cfg_words(c: dict) -> list[int]:
    """cfg()'s inverse: the register values for a configuration (the fake card's)."""
    return [c["S"], c["TMEM_WORDS"], c["IMEM_WORDS"], c["ACT_BLOCKS"],
            (c["DRAM_BYTES"] - 1).bit_length() | c["channels"] << 8
            | (c["channel_bytes"] - 1).bit_length() << 16,
            c["MXU_IMPL"] | c["TMEM_IMPL"] << 4 | c["VPU_CL"] << 8 | c["ULANES"] << 16
            | c.get("SEQ_WC", 0) << 24 | c.get("QUAD", 0) << 25,
            c["platform"], c["hash"]]


# free-running 64-bit counters: shadows latched by a SNAP write; low word at the offset
COUNTERS = {"UPTIME": 0x100, "RUNNING": 0x108, "MXU_BUSY": 0x110, "MXU_MAC": 0x118,
            "VPU_BUSY": 0x120, "QNT_BUSY": 0x128, "DMA_BUSY": 0x130, "TMEM_DENY": 0x138,
            "DRAM_RD": 0x140, "DRAM_WR": 0x148, "DRAM_WAIT": 0x150, "INSTR": 0x158,
            "MXU_STARVE": 0x160}
COUNTER_SINCE = {"MXU_STARVE": 3}           # the register map version that added a counter;
                                            # the others are version 2
EVENTS = ("DRAM_RD", "DRAM_WR", "INSTR")    # count events, not cycles
DRAM_BEAT = 64                              # bytes per DRAM_RD / DRAM_WR event

# trace buffer
R_TRACE_CTRL, R_TRACE_COUNT, R_TRACE_DROP = 0x200, 0x204, 0x208
R_TRACE_ADDR, R_TRACE_LO, R_TRACE_HI = 0x20C, 0x210, 0x214
TR_ENABLE, TR_CLEAR, TR_STOP_WHEN_FULL = 1, 2, 4
TR_BUSY = 8                      # TRACE_CTRL bit3 (read only): events not yet in the buffer

# I2C pins (CAPS bit2; opentpu/host/i2c.py): I2C_CTRL bit 2b drives bus b's SCL low, bit 2b+1
# its SDA (1 = low, 0 = released); I2C_IN reads the levels the same way, plus ALERT0 at bit4.
# Bus 0 is the LM73 sensor's, bus 1 the PCIe edge connector's SMBus.
R_I2C_CTRL, R_I2C_IN = 0x220, 0x224
I2C_ALERT0 = 1 << 4              # I2C_IN: the LM73's ALERT pin (active low)


def temp_c(code: int, family: str = "xilinx7") -> float:
    """Die-temperature code (12 bits) -> degrees Celsius: the 7-series XADC's (UG480), or on
    UltraScale+ (family "xilinxus") SYSMONE4's with its internal reference (UG580; the BCU1525's
    top gives its 10-bit code as the 12-bit code's upper bits)."""
    if family == "xilinxus":
        return (code & 0xFFF) * 509.3140064 / 4096 - 280.2308787
    return (code & 0xFFF) * 503.975 / 4096 - 273.15


def caps(v: int) -> dict:
    return {"trace": bool(v & CAP_TRACE), "temp": bool(v & CAP_TEMP), "i2c": bool(v & CAP_I2C),
            "ddr": bool(v & CAP_DDR), "w4": bool(v & CAP_W4),
            "pair": bool(v & CAP_PAIR), "dstep": bool(v & CAP_DSTEP),
            "chash": bool(v & CAP_CHASH), "act_rows": bool(v & CAP_ACT),
            "args": bool(v & CAP_ARGS), "stream": bool(v & CAP_STREAM),
            "hostcal": bool(v & CAP_HOSTCAL), "gen": bool(v & CAP_GEN),
            "waitw": bool(v & CAP_WAITW), "mx": bool(v & CAP_MX),
            "trace_depth": 1 << ((v >> 8) & 0xFF) if v & CAP_TRACE else 0,
            "pq_window": 1 << ((v >> 16) & 0xFF)}


def counters(regmap: int) -> dict:
    """The free-running counters a register map version has ({} before version 2)."""
    return {k: o for k, o in COUNTERS.items() if regmap >= COUNTER_SINCE.get(k, 2)}


# RDOT / OUTER / LOG2 (ddec900) predate register map 3 (3281a7b): a bitstream reporting map 3
# or later has them, so wrong results there are a fault, not an old bitstream
VOPS_SINCE = 3


def regmap(v: int) -> int:
    """REGMAP register value -> map version (1 when the register is missing)."""
    return 1 if v in (UNMAPPED, 0) else v
