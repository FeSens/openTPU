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
# CAPS2 bit1: the VPU has no VOP OUTER (otpu_vpu OUTER 0: the WG targets', whose DeltaNet runs DSTEP;
# Config.OUTER off). Clear on every bitstream before it: OUTER present.
CAP2_NO_OUTER = 2
# CAPS2 bit2: the DMA and the stream engine run DSTEP's pair (otpu_board DSTEP2: two heads'
# rows in one DSTEP, isa.F_DPAIR; Config.DSTEP2). Clear on every bitstream before it.
CAP2_DSTEP2 = 4
# CAPS2 bit3: the streams on an engine of their own beside the VPU (otpu_board SE_R > 0, the
# slice's otpu_se_eng; Config.SE_OWN: the compiler's DeltaNet bursts a pair each). Clear on every
# bitstream before it.
CAP2_SE_OWN = 8
# CAPS2 bit4: the board management controller's UART bridge (the BCU1525's otpu_bmc_uart, BAR0
# R_BMC; opentpu/host/bmc.py): the VCCINT regulator's telemetry through the BMC
CAP2_BMC = 16
R_BMC = 0x20000
# CAPS2 bit5: the MXU takes MM's K tail (otpu_board KT: a grouped MM with rs bit 0, a pair of
# groups sharing their last K-block's chunk, docs/isa.md "K tail"; Config.KT). Clear on every
# bitstream before it.
CAP2_KT = 32
# CAPS2 bits 11:6: the memory's bank-half window (the BCU1525's otpu_mig_pair BHW): the slice's
# first 2^n bytes, the images' layout (Config.BHW: the DeltaNet mixers' weights in its upper half,
# what a decode writes in its lower; 0 none, every bitstream before)
CAP2_BHW_SHIFT, CAP2_BHW_MASK = 6, 0x3F
# CAPS2 bit12: the IMEM an instruction cache over the program in DRAM (otpu_seq ICACHE,
# docs/icache.md; Config.ICACHE): LOAD copies nothing, the slice fetches the program's lines
# from DRAM as it runs them, and CFG_IMEM is the longest program, not an IMEM's words. Clear on
# every bitstream before it.
CAP2_ICACHE = 1 << 12
# CAPS2 bits 14:13: DSTEP's (STREAM's) DRAM state in 16 bits with its flag ST16 (otpu_dma ST16,
# docs/isa.md "ST16"; Config.ST16): 1 e6m9, 2 bf16; 0 none, every bitstream before it
CAP2_ST16_SHIFT, CAP2_ST16_MASK = 13, 0x3
ST16_NAMES = ("", "e6m9", "bf16")
# CAPS2 bit15: the card's collective spans cards (the link port, rtl/link/otpu_link: tensor
# parallel over QSFP28, opentpu/host/cards.py), its CSRs in the X2 window. Clear on every bitstream
# before it.
CAP2_XCARD = 1 << 15
# CAPS2 bit16: VOP EXP2SUB's fused row sums (otpu_vpu FSUM, flag FSUM: T[imm + r] = isum_64 of
# row r's results, docs/isa.md VOP; Config.FSUM). Clear on every bitstream before it.
CAP2_FSUM = 1 << 16
# CAPS2 bit17: QACT / QST flag LO, the split's lo part (otpu_quant SPLIT, docs/isa.md
# "Quantization"; Config.SPLITQ). Clear on every bitstream before it.
CAP2_SPLITQ = 1 << 17
# CAPS2 bit18: STREAM's flags ST16R / ST16W, one end alone 16-bit (otpu_dma ST16S, docs/isa.md
# "ST16"; Config.ST16S). Clear on every bitstream before it.
CAP2_ST16S = 1 << 18
# CAPS2 bit19: VOP SIGMOID and SILU (otpu_vpu VSIG: recip(exp2(A*B) + 1) and A times it, docs/isa.md
# VOP; Config.VSIG). Clear on every bitstream before it. (Bits 17-18: bf16st's SPLITQ, ST16 on
# STREAM; bit20: VOP MAC.)
CAP2_VSIG = 1 << 19
# CAPS2 bit20: VOP MAC (otpu_vpu MAC: dst += A * B in place, B per column or a scalar, docs/isa.md
# VOP; Config.VMAC). Clear on every bitstream before it.
CAP2_VMAC = 1 << 20
# CAPS2 bit21: QACT flag HL, a split operand's hi and lo from one read (otpu_quant HL,
# docs/isa.md "Quantization"; Config.QHL). Clear on every bitstream before it. (Bits 19 and 20:
# wide-mxu's VSIG / VMAC.)
CAP2_QHL = 1 << 21
# CAPS2 bit22: QST HL, a GRP ROW QST's hi and lo parts from one read, lo HLD KiB after hi (otpu_quant
# QSTHL, docs/isa.md QST; Config.QSTHL). Clear on every bitstream before it.
CAP2_QSTHL = 1 << 22
# CAPS2 bit23: VOP OUTER of decay one on MAC's datapath, a VPU without OUTER (otpu_vpu OUTER1,
# docs/isa.md VOP OUTER; Config.OUTER1). Clear on every bitstream before it.
CAP2_OUTER1 = 1 << 23
# CAPS2 bit24: quantize-once, QACT flag ST (each row's ACT bytes and records to DRAM) and LDA (DMA:
# such rows into ACT RAM) (otpu_quant QONCE, otpu_dma LDA, docs/isa.md "Quantize once";
# Config.QONCE). Clear on every bitstream before it.
CAP2_QONCE = 1 << 24
# CAPS2 bit25: MM SPL, a split's terms as one MM (otpu_mxu SPL, docs/isa.md "Split terms";
# Config.SPLMM). Clear on every bitstream before it.
CAP2_SPLMM = 1 << 25
# X2 window, docs/x2_link.md (otpu_ctrl's mux, not
# otpu_link_pkg's old A_* map). X2_SEQ and COLL_STATUS / WD are the hub's additions.
R_X2_CARD, R_X2_CTRL, R_X2_INFO, R_X2_SEQ = 0x300, 0x304, 0x308, 0x30C
R_X2_STATUS = (0x310, 0x320)
R_X2_HELLO = (0x314, 0x324)
R_X2_CRC = (0x318, 0x328)
R_X2_FRAMES = (0x31C, 0x32C)
R_X2_PING = (0x330, 0x334)
R_X2_COUNTERS = (0x340, 0x350)       # +0 TX words, +4 RX words, +8 credit wait, +C soft errors
R_COLL_STATUS, R_COLL_WD = 0x360, 0x364
X2_CLEAR, X2_HELLO, X2_LRST, X2_PING = 1, 2, 4, 8
X2_CHANNEL, X2_SEEN, X2_CRC, X2_WATCHDOG = 1, 2, 4, 8
X2_ABORT, X2_LANES, X2_DROPPED, X2_UP = 16, 32, 64, 128
X2_ERRORS = X2_CRC | X2_WATCHDOG | X2_ABORT | X2_DROPPED
X2_HELLO_MAGIC = 0xA5
COLL_ABORTED, COLL_WATCHDOG, COLL_HEADER, COLL_LINK, COLL_ABORT_RX = 1, 2, 4, 8, 16
COLL_ERRORS = 0x1F
COLL_CAUSES = ("aborted", "watchdog", "header out of place", "link check", "ABORT received")
COLL_WD_DEFAULT = 1 << 26          # core cycles; 0 disables the watchdog


def caps2(v: int | None) -> dict:
    """CAPS2's value -> its capabilities; None (a register map before 5): all present but
    DSTEP2, SE_OWN, BMC, KT and ICACHE (which came after), no bank-half window (bhw 0), no
    16-bit state (st16 ""), no cross-card port (xcard False), no fused row sums (fsum False),
    no split LO flag (splitq False), no one-sided ST16 (st16s False), no QACT HL (qhl
    False), no QST HL (qsthl False), no SIGMOID / SILU (vsig False), no VOP MAC (vmac False), no
    OUTER decay one on MAC's (outer1 False), no QACT ST / LDA (qonce False), no MM SPL (splmm
    False)."""
    if v is None:
        return {"gather": True, "outer": True, "dstep2": False, "se_own": False, "bmc": False,
                "kt": False, "bhw": 0, "icache": False, "st16": "", "xcard": False,
                "fsum": False, "splitq": False, "st16s": False, "qhl": False, "qsthl": False,
                "vsig": False, "vmac": False, "outer1": False, "qonce": False, "splmm": False}
    st = v >> CAP2_ST16_SHIFT & CAP2_ST16_MASK
    return {"gather": bool(v & CAP2_GATHER), "outer": not v & CAP2_NO_OUTER,
            "dstep2": bool(v & CAP2_DSTEP2), "se_own": bool(v & CAP2_SE_OWN),
            "bmc": bool(v & CAP2_BMC), "kt": bool(v & CAP2_KT),
            "bhw": v >> CAP2_BHW_SHIFT & CAP2_BHW_MASK, "icache": bool(v & CAP2_ICACHE),
            "st16": ST16_NAMES[st] if st < len(ST16_NAMES) else f"reserved {st}",
            "xcard": bool(v & CAP2_XCARD), "fsum": bool(v & CAP2_FSUM),
            "splitq": bool(v & CAP2_SPLITQ), "st16s": bool(v & CAP2_ST16S),
            "qhl": bool(v & CAP2_QHL), "qsthl": bool(v & CAP2_QSTHL),
            "vsig": bool(v & CAP2_VSIG), "vmac": bool(v & CAP2_VMAC),
            "outer1": bool(v & CAP2_OUTER1), "qonce": bool(v & CAP2_QONCE),
            "splmm": bool(v & CAP2_SPLMM)}


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
            "beat_bytes": 64 << ((dram >> 24) & 0xFF),     # (0 before the WG targets' 128-byte beats)
            "MXU_IMPL": ua & 0xF, "TMEM_IMPL": (ua >> 4) & 0xF, "VPU_CL": (ua >> 8) & 0xFF,
            "ULANES": (ua >> 16) & 0xFF, "SEQ_WC": (ua >> 24) & 1,
            "WG": 1 << ((ua >> 25) & 7),                  # (weight groups; 0 before them: 1)
            "ICACHE": (ua >> 28) & 1,                     # (CAPS2 bit12's, in the hash's words)
            "platform": plat, "hash": h}


def cfg_words(c: dict) -> list[int]:
    """cfg()'s inverse: the register values for a configuration (the fake card's)."""
    return [c["S"], c["TMEM_WORDS"], c["IMEM_WORDS"], c["ACT_BLOCKS"],
            (c["DRAM_BYTES"] - 1).bit_length() | c["channels"] << 8
            | (c["channel_bytes"] - 1).bit_length() << 16
            | (c.get("beat_bytes", 64) // 64 - 1).bit_length() << 24,
            c["MXU_IMPL"] | c["TMEM_IMPL"] << 4 | c["VPU_CL"] << 8 | c["ULANES"] << 16
            | c.get("SEQ_WC", 0) << 24 | (c.get("WG", 1) - 1).bit_length() << 25
            | c.get("ICACHE", 0) << 28,
            c["platform"], c["hash"]]


# free-running 64-bit counters: shadows latched by a SNAP write; low word at the offset
COUNTERS = {"UPTIME": 0x100, "RUNNING": 0x108, "MXU_BUSY": 0x110, "MXU_MAC": 0x118,
            "VPU_BUSY": 0x120, "QNT_BUSY": 0x128, "DMA_BUSY": 0x130, "TMEM_DENY": 0x138,
            "DRAM_RD": 0x140, "DRAM_WR": 0x148, "DRAM_WAIT": 0x150, "INSTR": 0x158,
            "MXU_STARVE": 0x160}
COUNTER_SINCE = {"MXU_STARVE": 3}           # the register map version that added a counter;
                                            # the others are version 2
EVENTS = ("DRAM_RD", "DRAM_WR", "INSTR")    # count events, not cycles
DRAM_BEAT = 64                              # bytes per DRAM_RD / DRAM_WR event (the cards'; a
                                            # channel beat: CFG_DRAM's beat_bytes)

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
