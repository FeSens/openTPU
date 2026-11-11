"""openTPU instruction encoding (docs/isa.md). Every instruction is eight 32-bit words."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NOP, HALT, LI, ADDI, LOOP, BAR, RLD, WAITW = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
LD, ST, DSTEP, STREAM, LDA = 0x10, 0x11, 0x12, 0x13, 0x14
MM, QACT, QST = 0x20, 0x21, 0x22
VOP = 0x30
GATHER = 0x40

OPNAMES = {NOP: "NOP", HALT: "HALT", LI: "LI", ADDI: "ADDI", LOOP: "LOOP", BAR: "BAR", RLD: "RLD",
           WAITW: "WAITW",
           LD: "LD", ST: "ST", DSTEP: "DSTEP", STREAM: "STREAM", LDA: "LDA", MM: "MM",
           QACT: "QACT",
           QST: "QST", VOP: "VOP", GATHER: "GATHER"}

# MM / QACT / QST flags
F_UNIT, F_ACC, F_RMAX, F_ASCALE = 0x1, 0x2, 0x4, 0x8     # MM
# MM weight format, flags[5:4] (docs/isa.md "Weight formats")
WF_SHIFT = 4
# int8 | int4 / E2M1 with a 6-bit multiplier per D/4 | E2M1 with an E4M3 scale per D/8; every
# format a per-row fp32 scale Sw and power-of-two offsets dw under it (MX, docs/mx_mxu.md)
W8, W4I, W4F, WNV = 0, 1, 2, 3
WFORMATS = {"int8": W8, "int4": W4I, "fp4": W4F, "nvfp4": WNV}
MX_W = {W8: 16, W4I: 11, W4F: 11, WNV: 13}   # the shift window W of each format
F_PAIR = 0x40                   # MM, 4-bit only: column reuse, two K-blocks per cycle and row
F_RSC = 0x80                    # MM: row scales only (each scale row is Sw[n]; dw 0, m 1)
MM_SPL = 1 << 31                # MM w7 (srs) bit31, SPLMM: a split's terms as one MM (KB 3: lo.hi +
MM_HLD = 20                     #   hi.hi + hi.lo; KB 2: hi + lo), lo at + w7[30:20] KiB; srs w7[19:0]
F_ROW, F_CSCALE, F_RSCALE = 0x1, 0x2, 0x4   # QACT (QST: F_ROW)
F_DUP = 0x8                     # QACT: also write the rows to ACT rows rows..2*rows-1 (PAIR)
F_QLO = 0x10                    # QACT, SPLITQ: the split's lo part, q8lo(x') with S * 2^-8 (fp32.quantize_mx)
F_QHL = 0x20                    # QACT, QHL: hi into blocks ab .., lo into ab + KB .. in one pass
QHL_ELEMS = 8192                # QACT HL: a row's elements at most (KB x D; otpu_quant's row buffer)
F_QST = 0x40                    # QACT, QONCE: each row's ACT bytes and records to DRAM too (w3)
QREC = 8                        # QACT ST / LDA: a block's record: its fp32 S, its NSUB offsets
QALIGN = 64                     # QACT ST / LDA: their DRAM rows' addresses and stride, multiples of it
F_HALF = 0x2                    # QST, ROW mode: write only the first half of each row
F_QGRP = 0x4                    # QST: into a grouped int8 matrix (WG > 1, GALL): rows R[rd] + w6 ..
F_QCOL = 0x8                    # QST GRP: the rows are the grouped matrix's columns (V^T)
QST_LG = 4                      # QST GRP: flags 6:4 = LG, the matrix's groups of 2^LG rows (WG)
F_SLO = 0x80                    # QST, SPLITQ: the split's lo part (as QACT's F_QLO)
QST_HL = 1 << 31                # QST w5 (srs) bit31, QSTHL: hi and lo from one read, lo at
QST_HLD = 20                    #   + w5[30:20] KiB (data and scales; HLD); srs w5[19:0]
F_RAW = 0x1                     # RLD: the word's bits (no fp32 -> int conversion)
F_MUL = 0x2                     # RLD: times R[rb] + w2 (mod 2^AW)
F_BOUND = 0x4                   # RLD: {w5, w3} <= the value <= {w6, w4} (the waits pass's; the hardware ignores it)
VARIANT = 0x8000_0000           # RLD w7: a variant's count word (rld's variant; the hardware ignores it)
F_CHAIN = 0x1                   # HALT: then load and start the program at R[ra], R[rb] words
C_EQ, C_NE, C_GE = 0, 1, 2      # WAITW flags[1:0]: the word ==, != or >= (signed difference) ref

# VOP functions
V_ADD, V_SUB, V_RSUB, V_MUL, V_MAX, V_MIN, V_OUTER = 0, 1, 2, 3, 4, 5, 6
V_COPY, V_EXP2, V_RECIP, V_RSQRT, V_ABS, V_FILL, V_EXP2SUB, V_LOG2 = 8, 9, 10, 11, 12, 13, 14, 15
V_RSUM, V_RMAX, V_RSSQ, V_RDOT, V_ARGMAX = 16, 17, 18, 19, 20
V_SIGMOID, V_SILU = 21, 22             # Config.VSIG: recip(exp2(A*B) + 1), A times it
V_MAC = 23                           # dst += A * B in place (Config.VMAC: B per column or a scalar)
VFUNCS = {V_ADD: "add", V_SUB: "sub", V_RSUB: "rsub", V_MUL: "mul", V_MAX: "max",
          V_MIN: "min", V_OUTER: "outer", V_COPY: "copy", V_EXP2: "exp2", V_RECIP: "recip",
          V_RSQRT: "rsqrt", V_ABS: "abs", V_FILL: "fill", V_EXP2SUB: "exp2sub",
          V_LOG2: "log2", V_RSUM: "rsum", V_RMAX: "rmax", V_RSSQ: "rssq", V_RDOT: "rdot",
          V_ARGMAX: "argmax", V_SIGMOID: "sigmoid", V_SILU: "silu", V_MAC: "mac"}
BINARY = {V_ADD, V_SUB, V_RSUB, V_MUL, V_MAX, V_MIN, V_FILL, V_EXP2SUB, V_SIGMOID, V_SILU}
REDUCE = {V_RSUM, V_RMAX, V_RSSQ, V_RDOT, V_ARGMAX}
READS_B = BINARY | {V_RDOT, V_MAC}   # functions that read operand B in its bmode (OUTER: B_ROW;
#                                      MAC: B_COL or B_SCALAR)
F_DSCALAR, F_DONE = 0x1, 0x2         # VOP OUTER: decay T[d] for every column / decay 1.0
F_FSUM = 0x4                         # VOP EXP2SUB: also T[imm + r] = isum_64 of row r (FSUM),
#                                      written by the time its take (VOP RSUM with it) is done
OUTER_MAX_COLS = 256                 # OUTER: its column vectors are held in 256-word buffers
FSUM_MAX_ROWS, FSUM_MIN_COLS = 256, 64   # FSUM: row sums buffered (RDOT's 256), whole partials
MAC_MAX_COLS = 256                   # MAC per column: its column vector is held in OUTER's buffer

# VOP broadcast modes for operand B
B_FULL, B_ROW, B_COL, B_SCALAR = 0, 1, 2, 3


def u32(x: int) -> int:
    return int(x) & 0xFFFFFFFF


def f32bits(x: float) -> int:
    return int(np.asarray(x, dtype=np.float32).view(np.uint32))


@dataclass
class Instr:
    op: int
    ra: int = 0
    rb: int = 0
    rc: int = 0
    rd: int = 0
    flags: int = 0
    w: list = field(default_factory=lambda: [0] * 7)   # w1..w7
    comment: str = ""
    src: tuple = ()          # kernel source frames that emitted it (set by the compiler)

    def encode(self) -> list[int]:
        _check(all(0 <= r < 16 for r in (self.ra, self.rb, self.rc, self.rd)),
               f"{OPNAMES.get(self.op, self.op)}: a register field outside R0..R15")
        _check(0 <= self.flags < 256, f"{OPNAMES.get(self.op, self.op)}: flags {self.flags:#x}")
        w0 = (self.op & 0xFF) | (self.ra << 8) | (self.rb << 12) | (self.rc << 16) \
            | (self.rd << 20) | ((self.flags & 0xFF) << 24)
        return [u32(w0)] + [u32(x) for x in self.w]

    @staticmethod
    def decode(words) -> "Instr":
        w0 = int(words[0])
        return Instr(op=w0 & 0xFF, ra=(w0 >> 8) & 15, rb=(w0 >> 12) & 15, rc=(w0 >> 16) & 15,
                     rd=(w0 >> 20) & 15, flags=(w0 >> 24) & 0xFF,
                     w=[int(x) for x in words[1:8]])

    def __str__(self) -> str:
        name = OPNAMES.get(self.op, f"op{self.op:#x}")
        regs = f"ra=R{self.ra} rb=R{self.rb} rc=R{self.rc} rd=R{self.rd}"
        c = f"  ; {self.comment}" if self.comment else ""
        return f"{name:6s} {regs} fl={self.flags:#x} w={[hex(x) for x in self.w]}{c}"


def _check(ok: bool, what: str) -> None:
    """A field that does not fit its bits, or a flag combination the instruction does not
    have: an error (an assert would let python -O truncate the word silently)."""
    if not ok:
        raise ValueError(what)


def _w(*vals) -> list:
    """Immediate words w1.. (the rest 0): each a 32-bit word, a negative one in two's
    complement; one that does not fit 32 bits is an error, not truncated (an address of AW >
    32 bits is a base register's: LI / ADDI's {w2, w1}, imm64)."""
    for x in vals:
        _check(-(1 << 32) < int(x) < 1 << 32, f"immediate {int(x):#x} does not fit 32 bits")
    v = [u32(x) for x in vals]
    return v + [0] * (7 - len(v))


def imm64(imm: int, aw: int = 32) -> list:
    """LI's and ADDI's immediate words {w2, w1} = imm mod 2^64 (docs/isa.md "Address bits"): at
    AW 32 w1 alone, w2 0 (the hardware ignores w2: a negative imm is w1's two's complement); at
    AW > 32 w2 the high word, 0xFFFFFFFF for a negative imm of 32 bits."""
    imm = int(imm)
    if aw == 32:
        return _w(imm)
    _check(32 < aw <= 64 and -(1 << aw) < imm < 1 << aw,
           f"LI / ADDI immediate {imm:#x} at AW {aw}")
    return _w(imm & 0xFFFFFFFF, (imm >> 32) & 0xFFFFFFFF)


def nop(comment=""):
    return Instr(NOP, comment=comment)


def halt(chain=False, ra=0, rb=0, comment=""):
    """HALT; with `chain`, the board then loads R[rb] instructions from DRAM byte address R[ra]
    into IMEM and starts them (docs/isa.md "HALT CHAIN")."""
    return Instr(HALT, ra=ra, rb=rb, flags=F_CHAIN if chain else 0, comment=comment)


def wrap(v: int, aw: int = 32) -> int:
    """v mod 2^AW as a signed AW-bit number: an increment LI / ADDI take whatever v's size
    (register arithmetic is mod 2^AW), e.g. a loop's reset of a stride times its count."""
    m = 1 << aw
    v = int(v) % m
    return v - m if v >= m >> 1 else v


def li(rd, imm, comment="", aw: int = 32):
    """R[rd] = imm mod 2^AW ({w2, w1}: imm64)."""
    return Instr(LI, rd=rd, w=imm64(imm, aw), comment=comment)


def addi(rd, ra, imm, comment="", aw: int = 32):
    """R[rd] = R[ra] + imm mod 2^AW ({w2, w1}: imm64; a negative imm at AW > 32 sets w2)."""
    return Instr(ADDI, rd=rd, ra=ra, w=imm64(imm, aw), comment=comment)


def loop(body_len, count, rcount=0, comment=""):
    return Instr(LOOP, ra=rcount, w=_w(body_len, count), comment=comment)


def bar():
    return Instr(BAR)


def rld(rd, tmem, ra=0, raw=False, mul=None, rb=0, bound=None, variant=None, comment=""):
    """R[rd] = f2i(T[R[ra] + tmem]) (raw: the word's bits), times R[rb] + mul (mod 2^AW: the
    word zero-extended) if mul is given; docs/isa.md "RLD". bound (lo, hi): the values the
    register may take, lo <= v <= hi as unsigned numbers (lo below 2^AW and hi past it: v from
    lo to 2^AW - 1 or v + 2^AW up to hi, a negative multiplier's product: compiler.mul_bound),
    what the kernel knows of the word (an index below E, a slot of the pool, a resident
    expert's place: id x expert bytes, past 32 bits above AW 32): the waits pass bounds the
    footprints computed from the register by it
    (opentpu.waits), the ISA simulator checks it, the hardware ignores it (flag bit 2; {w5, w3}
    and {w6, w4}, w5 and w6 0 below 2^32, so 32-bit bounds keep their words). variant (k, K):
    the count word of variant k of K (attention._Variants: words tmem - k .. tmem - k + K - 1 of
    one table, exactly one of them 1 at a run, bound (0, 1)), w7 = VARIANT | K << 16 | k: the
    waits pass then runs the K paths apart (opentpu.waits); the hardware ignores w7."""
    flags = (F_RAW if raw else 0) | (F_MUL if mul is not None else 0)
    lo = hi = 0
    if bound is not None:
        lo, hi = (int(x) for x in bound)
        _check(0 <= lo <= hi < 1 << 64, f"RLD: bound {bound}")
        flags |= F_BOUND
    tag = 0
    if variant is not None:
        k, K = (int(x) for x in variant)
        _check(0 <= k < K < 1 << 15 and bound == (0, 1) and ra == 0 and not raw and mul is None,
               f"RLD: variant {variant} (a constant word, bound (0, 1))")
        tag = VARIANT | K << 16 | k
    return Instr(RLD, ra=ra, rb=rb, rd=rd, flags=flags,
                 w=_w(tmem, 0 if mul is None else mul, lo & 0xFFFFFFFF, hi & 0xFFFFFFFF,
                      lo >> 32, hi >> 32, tag), comment=comment)


def rld_mul(rd, tmem, c, ra=0, raw=False, bound=None, comment="", aw: int = 32) -> list:
    """R[rd] = the word at T[R[ra] + tmem] times the integer c (an RLD MUL), as instructions: at
    AW 32, or c >= 0, the RLD with w2 = c (a negative c in two's complement: the low 32 bits of
    the product, signed and unsigned alike); at AW > 32 a negative c is AW bits, so it goes
    into R[rd] first (LI, w2 0xFFFFFFFF) and the RLD multiplies by R[rd] + 0 (its multiplier
    read before R[rd] is written). The word is zero-extended: c * a non-negative value."""
    if aw == 32 or c >= 0:
        return [rld(rd, tmem, ra=ra, raw=raw, mul=c, bound=bound, comment=comment)]
    _check(rd != 0 and ra != rd, f"RLD MUL by {c} at AW {aw}: R{rd} takes the multiplier first "
                                 f"(rd R1..R15, not ra)")
    return [li(rd, c, comment=f"{comment}: x {c}", aw=aw),
            rld(rd, tmem, ra=ra, raw=raw, mul=0, rb=rd, bound=bound, comment=comment)]


def waitw(dram, tmem, ref, cmp, ra=0, rb=0, rc=0, mask=0xFFFFFFFF, interval=0, timeout=0,
          comment=""):
    """WAITW (DMA): wait until cmp(M32[R[ra] + dram] & mask, R[rc] + ref) holds, polling DRAM
    every `interval` cycles (the first read at once; timeout cycles, 0: none), then
    T[R[rb] + tmem] = the word's bits (an RLD raw takes it into a register); docs/isa.md
    "WAITW"."""
    if cmp not in (C_EQ, C_NE, C_GE):
        raise ValueError(f"WAITW compare {cmp}")
    return Instr(WAITW, ra=ra, rb=rb, rc=rc, flags=cmp,
                 w=_w(dram, tmem, ref, mask, interval, timeout), comment=comment)


def waitw_holds(word: int, ref: int, cmp: int, mask: int = 0xFFFFFFFF) -> bool:
    """WAITW's condition on the word it read."""
    v, ref = u32(word) & u32(mask), u32(ref)
    if cmp == C_EQ:
        return v == ref
    if cmp == C_NE:
        return v != ref
    return u32(v - ref) < 1 << 31               # GE: the difference >= 0 as signed


def f2i(bits: int) -> int:
    """RLD's conversion of fp32 bits to a 32-bit word: truncation toward zero; values below 1 in
    magnitude (denormals, +-0) give 0, and |x| >= 2^31, infinities and NaN give 0x80000000."""
    bits = u32(bits)
    e = (bits >> 23) & 0xFF
    if e < 127:
        return 0
    if e >= 127 + 31:
        return 0x80000000
    m = (bits & 0x7FFFFF) | 0x800000
    v = m << (e - 150) if e >= 150 else m >> (150 - e)
    return u32(-v if bits >> 31 else v)


def ld(dram, tmem, nwords, ra=0, rb=0, comment=""):
    return Instr(LD, ra=ra, rb=rb, w=_w(dram, tmem, nwords), comment=comment)


def st(dram, tmem, nwords, ra=0, rb=0, comment=""):
    return Instr(ST, ra=ra, rb=rb, w=_w(dram, tmem, nwords), comment=comment)


DSTEP_LANES = 8                      # DSTEP: state words per cycle (timing only)
DSTEP_MAX_ROWS = 256                 # DSTEP: o is held in a 256-word buffer
F_DZERO = 0x1                        # DSTEP: the state starts at +0 (DRAM is not read)
F_DPAIR = 0x2                        # DSTEP: two heads (Config.DSTEP2), rows / 2 rows each
F_DQKB = 0x4                         # DSTEP pair: head b's q | k after head a's, else head a's
F_DST16 = 0x8                        # DSTEP: the DRAM state in 16 bits (Config.ST16; fp32 inside)
DSTEP_PAIR_COLS = 128                # DSTEP pair: the heads' cols (both vectors in SE's buffers)
DSTEP_PAIR_ROWS = 16                 # DSTEP pair: rows a multiple (a head's rows: the engine's
#                                      row groups, up to 8, each in one head)


def dstep(dram, qk, v, rows, cols, g, gs, o, zero=False, pair=False, qkb=False, ra=0, rb=0,
          rc=0, comment="", st16=False):
    """DSTEP (DMA): one Gated DeltaNet head step on an fp32 state St [rows, cols] (row-major)
    in DRAM at byte address R[ra] + dram, updated in place. q = T[qk + c], k = T[qk + cols + c]
    (qk: R[rb] + w2), v(r) = T[v + r] (R[rc] + w3), the decay e = T[g], beta = T[g + gs], and
    o(r) = T[o + r] is written. Row by row: kv = rdot(St[r], k), d = (v(r) - kv * e) * beta,
    St[r] = St[r] * e + d * k, o(r) = rdot(St[r], q) -- the rounding of RDOT, MUL, SUB, MUL,
    OUTER, RDOT (docs/isa.md). `zero`: the state starts at +0 and is not read (position 0).
    `pair` (Config.DSTEP2): two heads' steps, rows r >= rows / 2 head b's, with e = T[g + 1],
    beta = T[g + gs + 1] and (`qkb`) q | k at qk + 2 cols; cols 128, rows a multiple of 16.
    `st16` (Config.ST16): the state is 16-bit in DRAM (2 rows cols bytes; the format the
    configuration's, docs/isa.md "ST16"), fp32 in the step."""
    _check(0 < rows < 65536 and 0 < cols < 65536, f"DSTEP: rows {rows}, cols {cols}")
    _check(not qkb or pair, "DSTEP: qkb is a pair's")
    _check(not pair or (rows % DSTEP_PAIR_ROWS == 0 and cols == DSTEP_PAIR_COLS),
           f"DSTEP pair: rows {rows} (a multiple of {DSTEP_PAIR_ROWS}), cols {cols} "
           f"({DSTEP_PAIR_COLS})")
    fl = (F_DZERO if zero else 0) | (F_DPAIR if pair else 0) | (F_DQKB if qkb else 0) | \
        (F_DST16 if st16 else 0)
    return Instr(DSTEP, ra=ra, rb=rb, rc=rc, flags=fl,
                 w=_w(dram, qk, v, rows | (cols << 16), g, o, gs), comment=comment)


# ------------------------------------------------------------------------------------ STREAM
# The stream engine's instruction (docs/stream.md): rows of a stream (DRAM or TMEM) pass
# REDUCE A -> SCALAR -> MAP U -> MAP/REDUCE Q under a descriptor of a few TMEM words.
F_SZERO = 0x1                        # the stream reads as +0 (nothing is read)
F_SRC_T = 0x2                        # the stream source is TMEM (words), not DRAM (bytes)
F_DST_T = 0x4                        # the stream destination is TMEM
F_NODST = 0x8                        # the stream is not written back
F_SST16 = 0x10                       # the DRAM stream in 16 bits, source and destination
#                                      (Config.ST16, as DSTEP's ST16; fp32 inside)
F_SST16R = 0x20                      # the source alone in 16 bits, the destination fp32
F_SST16W = 0x40                      # the destination alone in 16 bits, the source fp32

# ST16 (docs/isa.md "ST16"): a DSTEP's (STREAM's) DRAM state in 16 bits, its format the
# configuration's (Config.ST16, CAPS2 bits 14:13): e6m9 (6 exponent bits, bias 31, 9 mantissa
# bits) or bf16
ST16_CODE = {"": 0, "e6m9": 1, "bf16": 2}

A_SLOT, A_SELF, A_CONST, A_MAX = 0, 1, 2, 3      # REDUCE A: A = isum_64(S[r] * opA), or
#                                                   the row's max (A_MAX, VOP RMAX's order)
U_PASS, U_FMMA, U_MUL, U_ADD = 0, 1, 2, 3         # MAP U: Y = S | S*G + D*B | S*G | S + D*B
G_REG, G_SLOT, G_CONST, G_ONE = 0, 1, 2, 3        # G: register, column slot, constant, 1.0
B_SLOT, B_CONST, B_REG = 0, 1, 2                  # B: column slot, constant, register
# SCALAR ops: dst = op(a, b), each rounded as fp32.py's function of the same name
SC_ADD, SC_SUB, SC_MUL, SC_MAX, SC_MIN, SC_MOV = 0, 1, 2, 3, 4, 5
SC_RSQRT, SC_RECIP, SC_EXP2, SC_LOG2, SC_EXP2SUB = 6, 7, 8, 9, 10
# SCALAR operands: r0..r7, the row's A, its row scalar X[r], constants K0..K3, +0, +1
OP_A, OP_X, OP_K0, OP_ZERO, OP_ONE = 8, 9, 10, 14, 15
STREAM_SLOTS = 4                     # column slot i at vec + i * cols
STREAM_MAX_OPS = 15
STREAM_DESC_FIXED = 5                # descriptor words before the scalar ops
STREAM_DESC_AREA = 64                # the compiler's descriptor words at the top of TMEM


def _fsafe(p: int) -> int:
    """A 24-bit payload as a float-safe word: exponent field 0x80 (a normal fp32 in [2, 4)),
    payload in the sign bit and the 23 mantissa bits, so a FILL writes it exactly."""
    _check(0 <= p < 1 << 24, f"STREAM descriptor payload {p:#x} is not 24 bits")
    return ((p >> 23) & 1) << 31 | 0x80 << 23 | (p & 0x7FFFFF)


def _payload(w: int) -> int:
    w = int(w) & 0xFFFFFFFF
    if (w >> 23) & 0xFF != 0x80:
        raise ValueError(f"STREAM descriptor word {w:#010x} is not float-safe")
    return ((w >> 31) & 1) << 23 | (w & 0x7FFFFF)


@dataclass(frozen=True)
class StreamDesc:
    """A STREAM descriptor (docs/stream.md, section 3.2). Per row r of the stream S [rows, cols]:
      A     = isum_64(S[r] * opA)                    (a_en; opA: slot a_idx, S itself, K[a_idx];
                                                      A_MAX: the row's max)
      regs  : the scalar ops in order, over r0..r7, A, X[r], K0..K3, +0, +1
      Y     = S | S*G + D*B | S*G | S + D*B          (u_mode; D = r[d_reg])
      O[r]  = isum_64(Y * Q) (q_en; Q = slot q_idx, or Y with q_self), or r[o_reg] (o_reg_en)
      the stream row written back is Y, or Y * C[q_idx] with out_p2
    r4..r7 start as K0..K3 where rinit has their bit (else +0) and carry from row to row; r0..r3
    start each instruction at +0 and also carry. rsave writes r4..r7 back to K0..K3's words."""
    rows: int
    cols: int
    a_en: bool = False
    a_op: int = A_SLOT
    a_idx: int = 0
    u_mode: int = U_PASS
    g_src: int = G_ONE
    g_idx: int = 0
    b_src: int = B_SLOT
    b_idx: int = 0
    d_reg: int = 0
    q_en: bool = False
    q_idx: int = 0
    q_self: bool = False
    out_p2: bool = False
    o_reg_en: bool = False
    o_reg: int = 0
    rinit: int = 0
    rsave: int = 0
    srs: int = 0                     # TMEM stream row strides (0: cols)
    drs: int = 0
    ops: tuple = ()                  # ((op, dst, a, b), ...)

    def words(self) -> list[int]:
        _check(0 < self.rows < 1 << 12 and 0 < self.cols < 1 << 12,
               f"STREAM descriptor: rows {self.rows}, cols {self.cols} (1 .. 4095)")
        _check(len(self.ops) <= STREAM_MAX_OPS, f"STREAM descriptor: {len(self.ops)} ops")
        _check(0 <= self.srs < 1 << 12 and 0 <= self.drs < 1 << 12,
               f"STREAM descriptor: srs {self.srs}, drs {self.drs} (0 .. 4095)")
        c1 = (int(self.a_en) | self.a_op << 1 | self.a_idx << 3 | self.u_mode << 5
              | self.g_src << 7 | self.g_idx << 9 | self.b_src << 12 | self.b_idx << 14
              | self.d_reg << 17 | int(self.q_en) << 20 | self.q_idx << 21
              | int(self.out_p2) << 23)
        c2 = (len(self.ops) | self.rinit << 4 | self.rsave << 8 | int(self.q_self) << 12
              | int(self.o_reg_en) << 13 | self.o_reg << 14)
        p = [self.rows | self.cols << 12, c1, c2, 0, self.srs | self.drs << 12]   # d3 reserved
        p += [op | dst << 4 | a << 7 | b << 11 for op, dst, a, b in self.ops]
        return [_fsafe(x) for x in p]

    @staticmethod
    def decode(words) -> "StreamDesc":
        p = [_payload(w) for w in words[:STREAM_DESC_FIXED]]
        c1, c2 = p[1], p[2]
        n = c2 & 15
        if len(words) < STREAM_DESC_FIXED + n:
            raise ValueError("STREAM descriptor: fewer words than its ops")
        ops = tuple((x & 15, (x >> 4) & 7, (x >> 7) & 15, (x >> 11) & 15)
                    for x in (_payload(w) for w in
                              words[STREAM_DESC_FIXED:STREAM_DESC_FIXED + n]))
        return StreamDesc(rows=p[0] & 0xFFF, cols=p[0] >> 12, a_en=bool(c1 & 1),
                          a_op=(c1 >> 1) & 3, a_idx=(c1 >> 3) & 3, u_mode=(c1 >> 5) & 3,
                          g_src=(c1 >> 7) & 3, g_idx=(c1 >> 9) & 7, b_src=(c1 >> 12) & 3,
                          b_idx=(c1 >> 14) & 7, d_reg=(c1 >> 17) & 7,
                          q_en=bool((c1 >> 20) & 1), q_idx=(c1 >> 21) & 3,
                          out_p2=bool((c1 >> 23) & 1), rinit=(c2 >> 4) & 15,
                          rsave=(c2 >> 8) & 15, q_self=bool((c2 >> 12) & 1),
                          o_reg_en=bool((c2 >> 13) & 1), o_reg=(c2 >> 14) & 7,
                          srs=p[4] & 0xFFF, drs=p[4] >> 12, ops=ops)

    def nwords(self) -> int:
        return STREAM_DESC_FIXED + len(self.ops)


def stream(desc, src, dst, vec, x, k, out, ks=1, zero=False, src_t=False, dst_t=False,
           nodst=False, ra=0, rb=0, rc=0, rd=0, comment="", st16=False, st16r=False,
           st16w=False):
    """STREAM (docs/stream.md): the descriptor at TMEM `desc` over the rows of the stream at
    src + R[ra] (DRAM bytes, or TMEM words with src_t), Y written to dst + R[ra]; column slots
    at vec + R[rb] (slot i at + i * cols), row scalars X[r] = T[x + R[rc] + r], constants
    K_j = T[k + R[rd] + j * ks], row outputs O[r] = T[out + r]. desc and ks share w1 (desc in
    [15:0], ks in [31:16]) so the scoreboard knows the constants' range. `st16`
    (Config.ST16): the DRAM stream's values in 16 bits (2 bytes each), both ends in DRAM;
    `st16r` (`st16w`) the source's (the destination's) alone, the other end fp32 and apart
    from it."""
    _check(0 <= desc < 1 << 16 and 0 <= ks < 1 << 16, f"STREAM: desc {desc}, ks {ks} (16 bits)")
    r16, w16 = st16 or st16r, st16 or st16w
    _check(not (r16 or w16) or not (src_t or dst_t or nodst), "STREAM: ST16 is a DRAM stream's")
    _check(r16 == w16 or src != dst, "STREAM: ST16 on one side needs dst apart from src")
    fl = (F_SZERO if zero else 0) | (F_SRC_T if src_t else 0) | (F_DST_T if dst_t else 0) | \
        (F_NODST if nodst else 0) | \
        (F_SST16 if r16 and w16 else F_SST16R if r16 else F_SST16W if w16 else 0)
    return Instr(STREAM, ra=ra, rb=rb, rc=rc, rd=rd, flags=fl,
                 w=_w(desc | ks << 16, src, dst, vec, x, k, out), comment=comment)


# the scalar programs of the hardware's d stage (docs/stream.md, 4.4), d in r2
D_DELTA = ((SC_MUL, 0, OP_A, OP_K0), (SC_SUB, 1, OP_X, 0), (SC_MUL, 2, 1, OP_K0 + 1))
D_DELTA1 = ((SC_SUB, 1, OP_X, OP_A), (SC_MUL, 2, 1, OP_K0 + 1))
D_SCALE = ((SC_MUL, 2, OP_X, OP_K0 + 1),)
D_DOT = ((SC_MUL, 2, OP_A, OP_K0 + 1),)
DMODES = {D_DELTA: 0, D_DELTA1: 1, D_SCALE: 2, D_DOT: 3}


def state_desc(rows, cols, dprog=D_DELTA, a_idx=1, gate="const", q=True) -> StreamDesc:
    """A linear-recurrence step (docs/stream.md, 5.2): slots q (0), b = k (1), a (2), gate
    column (3); X = the rows' inputs (v); K0 = the scalar gate, K1 = beta.
    dprog: D_DELTA / D_DELTA1 / D_SCALE / D_DOT; a_idx: dot A's slot (1 or 2; unused unless
    dprog reads A); gate: "const" (K0), "col" (slot 3) or "one"."""
    uses_a = any(OP_A in (a, b) for _, _, a, b in dprog)
    g_src, g_idx = {"const": (G_CONST, 0), "col": (G_SLOT, 3), "one": (G_ONE, 0)}[gate]
    return StreamDesc(rows, cols, a_en=uses_a, a_op=A_SLOT, a_idx=a_idx if uses_a else 0,
                      u_mode=U_FMMA, g_src=g_src, g_idx=g_idx, b_src=B_SLOT, b_idx=1,
                      d_reg=2, q_en=q, q_idx=0, ops=tuple(dprog))


def gdn_desc(rows, cols) -> StreamDesc:
    """Gated DeltaNet, exactly DSTEP (ks = DSTEP's gs): slots q (0), k (1); X = v; K0 = the
    decay e, K1 = beta."""
    return state_desc(rows, cols, D_DELTA, 1, "const", True)


def stream_hw_cfg(d: StreamDesc, flags: int = 0) -> dict | None:
    """The v1 hardware's configuration for a descriptor (docs/stream.md, 4.3 / 4.4), or None if
    the board's engine cannot run it (the compiler then falls back to VOPs)."""
    if flags & (F_SRC_T | F_DST_T | F_NODST):
        return None
    if not (0 < d.rows <= DSTEP_MAX_ROWS) or d.cols % 64 or not (0 < d.cols <= 256):
        return None
    if d.q_self or d.out_p2 or d.o_reg_en or d.rinit or d.rsave or d.srs or d.drs:
        return None
    if d.u_mode != U_FMMA or d.b_src != B_SLOT or d.b_idx != 1 or d.d_reg != 2:
        return None
    g = {(G_CONST, 0): 0, (G_SLOT, 3): 1, (G_ONE, 0): 2}.get(
        (d.g_src, d.g_idx if d.g_src != G_ONE else 0))
    dm = DMODES.get(tuple(d.ops))
    if g is None or dm is None or (d.q_en and d.q_idx != 0):
        return None
    uses_a = dm in (0, 1, 3)
    if uses_a != d.a_en or (d.a_en and (d.a_op != A_SLOT or d.a_idx not in (1, 2))):
        return None
    return {"ns": d.cols // DSTEP_LANES, "rows": d.rows, "a_en": d.a_en,
            "a_sel": int(d.a_idx == 2), "dmode": dm, "g_src": g, "q_en": d.q_en}


def mm(sa, ssa, out, n, kb, rs, ors, m, ab, srs, unit=False, acc=False, rmax=False,
       ascale=None, wf=W8, pair=False, k0=0, rsc=False, grp=1, kt=False, hld=0, ra=0, rb=0,
       rc=0, comment=""):
    """MM (docs/isa.md). With `ascale` (a TMEM address; needs unit and acc) the old accumulator
    is first multiplied by a per-row factor: y = T[out] * T[ascale + j] + a.w (the
    flash-attention rescale, done in the MXU epilogue). The address travels in the (unused)
    scale field. `wf`: the streamed weights' format (W8, W4I, W4F, WNV). `pair` (4-bit only):
    column reuse, ACT row j + m carries the odd K-blocks of row j ("Column reuse"). `k0`: the
    MM's first block within the scale rows at `ssa` (each row starts with its Sw). `rsc`: the
    scale rows are Sw alone (one word a row: the K cache). `grp`: the operand's rows a group, 1
    or quant.grp_r(fmt, Config.WG) (WG int8, 2 WG 4-bit; docs/isa.md "Weight groups"): rs and
    srs are then the groups' strides; its log2 is MM's rd field (no flag is free). `kt`
    (Config.KT, groups only; docs/isa.md "K tail"): the groups in pairs that share their last
    K-block's chunk, rs the pairs' stride; rs's bit 0 (a group stride is a multiple of WG x D).
    hld > 0 (MM_SPL, Config.SPLMM; docs/isa.md "Split terms"): the weights' lo part hld bytes
    after their hi part, one block each: kb 3 the split's three terms (ACT blocks ab the hi
    and ab + 1 the lo against hi, ab against lo), kb 2 a pair's (ACT block ab against hi, then lo),
    the lo terms 2^-8 under the hi block's scales; groups, int8 RSC, k0 0, not PAIR, KT, RMAX
    or ASCALE."""
    _check(0 < n < 65536 and 0 < kb < 256 and 0 < m < 256 and 0 <= ab < 256
           and 0 <= ors < 65536 and 0 <= k0 < 256,
           f"MM: n {n}, kb {kb}, m {m}, ab {ab}, ors {ors}, k0 {k0}")
    _check(wf in (W8, W4I, W4F, WNV), f"MM: weight format {wf}")
    _check(not pair or wf != W8, "MM: PAIR needs 4-bit weights")
    _check(not pair or k0 % 2 == 0, "MM: PAIR needs an even k0")
    _check(not rsc or wf != WNV, "MM: RSC does not take NVFP4")
    _check(0 < grp < 1 << 15 and grp & (grp - 1) == 0, f"MM: {grp} rows a group")
    # (RMAX and ASCALE on a group: GALL machines only, the simulator's check)
    _check(grp == 1 or not pair, "MM: groups take no PAIR")
    _check(not kt or (grp > 1 and k0 == 0 and wf != WNV and rs % 2 == 0),
           f"MM: KT needs groups, k0 0, an int8 / int4 / fp4 operand and an even rs ({rs})")
    _check(hld % 1024 == 0 and 0 <= hld >> 10 < 1 << (31 - MM_HLD), f"MM: SPL offset {hld}")
    _check(not hld or (kb in (2, 3) and grp > 1 and wf == W8 and rsc and not unit and k0 == 0
                       and not (pair or kt or rmax or ascale is not None)),
           f"MM: SPL needs KB 2 or 3, groups, int8 RSC, k0 0 and not PAIR, KT, RMAX or ASCALE")
    _check(not hld or 0 <= srs < 1 << MM_HLD, f"MM: SPL with srs {srs} past 20 bits")
    if ascale is not None:
        _check(unit and acc, "MM: ASCALE needs UNIT and ACC")
        ssa = ascale
    fl = (F_UNIT if unit else 0) | (F_ACC if acc else 0) | (F_RMAX if rmax else 0) | \
        (F_ASCALE if ascale is not None else 0) | (F_PAIR if pair else 0) | \
        (F_RSC if rsc else 0) | (wf << WF_SHIFT)
    return Instr(MM, ra=ra, rb=rb, rc=rc, rd=grp.bit_length() - 1, flags=fl,
                 w=_w(sa, ssa, out, n | (kb << 16) | (k0 << 24), rs | int(kt),
                      ors | (m << 16) | (ab << 24),
                      (srs | MM_SPL | (hld >> 10) << MM_HLD) if hld else srs),
                 comment=comment)


def qact(src, rows, ab, kb, srs, row=False, cscale=None, rscale=None, dup=False, ra=0,
         lo=False, hl=False, store=None, rc=0, comment=""):
    """QACT. `cscale`/`rscale`: TMEM addresses of a per-column / per-row factor applied before
    quantization, x' = (x * T[rscale + r]) * T[cscale + c]. `dup`: ACT row r + rows receives
    a copy of row r in the same cycles (the odd K-blocks' column of an MM PAIR). `lo` (F_QLO,
    SPLITQ): the split's lo part instead of q, its blocks' scale S * 2^-8. `hl` (F_QHL, QHL):
    both, q into blocks ab .. ab + kb - 1 and lo into ab + kb .. ab + 2 kb - 1, from one read.
    `store` (F_QST, QONCE): (dram, drs), each row's ACT bytes and records also to DRAM, row r at
    R[rc] + dram + r drs (qrow_bytes a row, LDA's); w3 is then that address, srs moves to w6
    and drs is w7."""
    _check(0 < rows < 256 and 0 <= ab < 256 and 0 < kb < 65536,
           f"QACT: rows {rows}, ab {ab}, kb {kb}")
    _check(not hl or not (lo or dup), "QACT: HL with LO or DUP")
    _check(store is None or not hl, "QACT: ST with HL")
    fl = (F_ROW if row else 0) | (F_CSCALE if cscale is not None else 0) | \
        (F_RSCALE if rscale is not None else 0) | (F_DUP if dup else 0) | (F_QLO if lo else 0) \
        | (F_QHL if hl else 0) | (F_QST if store is not None else 0)
    if store is None:
        w = _w(src, rows | (ab << 8) | (kb << 16), srs, cscale or 0, rscale or 0)
    else:
        w = _w(src, rows | (ab << 8) | (kb << 16), store[0], cscale or 0, rscale or 0, srs,
               store[1])
    return Instr(QACT, ra=ra, rc=rc if store is not None else 0, flags=fl, w=w,
                 comment=comment)


def qrow_bytes(kb: int, D: int) -> int:
    """QACT ST's and LDA's bytes a row (docs/isa.md "Quantize once"): its KB blocks' D bytes as
    ACT RAM holds them, then KB records of QREC bytes (the block's fp32 S, its NSUB offsets)."""
    return kb * D + kb * QREC


def lda(dram, rows, ab, kb, drs, arow=0, ra=0, comment=""):
    """LDA (DMA, Config.QONCE): rows rows of ACT bytes and records (QACT ST's, qrow_bytes a
    row), row r from DRAM R[ra] + dram + r drs, into ACT rows arow + r at blocks ab .. ab + kb
    - 1: the operand a QACT ST stored, unchanged."""
    _check(0 < rows < 256 and 0 <= ab < 256 and 0 < kb < 65536 and 0 <= arow < 65536,
           f"LDA: rows {rows}, ab {ab}, kb {kb}, ACT row {arow}")
    return Instr(LDA, ra=ra, w=_w(dram, rows | (ab << 8) | (kb << 16), drs, arow),
                 comment=comment)


def qst(src, dst, sdst, rows, kb, srs, drs, es, row=False, half=False, ra=0, rb=0, rc=0,
        grp=1, col=False, rd=0, lo=False, hld=0, comment=""):
    """grp > 1 (F_QGRP, docs/isa.md QST): rows R[rd] + drs .. of a grouped int8 matrix at dst in
    groups of grp rows (the machine's WG; LG = log2 grp in flags 6:4), a group's chunks P = grp
    x D bytes: es is the scale areas' group stride; col (F_QCOL): the rows are the matrix's
    columns, es its group stride. lo (F_SLO, SPLITQ): the split's lo bytes, the scale S * 2^-8.
    hld > 0 (QST_HL, QSTHL; GRP ROW rows): hi and lo from one read, the lo part's bytes and
    scales hld bytes (a multiple of 1 KiB) after hi's."""
    lg = int(grp).bit_length() - 1
    _check(0 < rows < 65536 and 0 < kb < 65536, f"QST: rows {rows}, kb {kb}")
    _check(grp >= 1 and grp == 1 << lg and lg < 8, f"QST: groups of {grp} rows")
    _check(row or not half, "QST: HALF needs ROW mode")
    _check(grp > 1 or not col, "QST: COL needs GRP")
    _check(not col or row, "QST: COL needs ROW mode (a column's scale is its row's)")
    _check(not hld or (grp > 1 and row and not (col or half or lo)),
           "QST: HL needs GRP rows in ROW mode, not COL, HALF or LO")
    _check(hld % 1024 == 0 and 0 <= hld >> 10 < 1 << (31 - QST_HLD), f"QST: HL offset {hld}")
    _check(not hld or 0 <= srs < 1 << QST_HLD, f"QST: HL with srs {srs} past 20 bits")
    return Instr(QST, ra=ra, rb=rb, rc=rc, rd=rd,
                 flags=(F_ROW if row else 0) | (F_HALF if half else 0) |
                 ((F_QGRP | lg << QST_LG) if grp > 1 else 0) | (F_QCOL if col else 0) |
                 (F_SLO if lo else 0),
                 w=_w(src, dst, sdst, rows | (kb << 16),
                      (srs | QST_HL | (hld >> 10) << QST_HLD) if hld else srs, drs, es),
                 comment=comment)


def vop(func, dst, a, b, rows, cols, drs, ars, brs, bmode=B_FULL, imm=0.0,
        ra=0, rb=0, rc=0, rd=0, comment=""):
    """imm: an fp32 immediate (ARGMAX: an int, the index base)."""
    _check(0 < rows < 65536 and 0 < cols < 65536, f"VOP: rows {rows}, cols {cols}")
    _check(0 <= drs < 65536 and 0 <= ars < 65536 and 0 <= brs < 65536,
           f"VOP: drs {drs}, ars {ars}, brs {brs} (16 bits)")
    iw = u32(imm) if func == V_ARGMAX else f32bits(imm)
    return Instr(VOP, ra=ra, rb=rb, rc=rc, rd=rd,
                 w=_w(dst, a, b, rows | (cols << 16), drs | (ars << 16),
                      brs | (func << 16) | (bmode << 24), iw),
                 comment=comment)


def argmax(dst, a, rows, cols, drs=2, ars=0, base=0, ra=0, rb=0, rd=0, comment="argmax"):
    """VOP ARGMAX: T[dst + r*drs] = the row maximum (RMAX), T[dst + r*drs + 1] =
    i2f(c + R[rd] + base) for c the first column holding it (rows > 1: drs >= 2)."""
    _check(rows == 1 or drs >= 2, "ARGMAX: the rows' (max, index) pairs need drs >= 2")
    return vop(V_ARGMAX, dst, a, 0, rows, cols, drs, ars, 0, B_FULL, base, ra=ra, rb=rb, rd=rd,
               comment=comment)


def outer(dst, d, b, c, rows, cols, drs, brs, dmode="scalar", ra=0, rb=0, rc=0, rd=0,
          comment=""):
    """VOP OUTER: T[dst + r*drs + j] = T[dst + r*drs + j] * Dv(j) + T[b + r*brs] * T[c + j],
    Dv(j) = T[d] (dmode "scalar"), T[d + j] ("column") or 1.0 ("one", d unused). The decay
    address travels in the A field and the column vector's in the immediate word (w7 += R[rd])."""
    _check(0 < rows < 65536 and 0 < cols <= OUTER_MAX_COLS and 0 <= drs < 65536
           and 0 <= brs < 65536, f"OUTER: rows {rows}, cols {cols}, drs {drs}, brs {brs}")
    fl = {"scalar": F_DSCALAR, "column": 0, "one": F_DONE}[dmode]
    return Instr(VOP, ra=ra, rb=rb, rc=rc, rd=rd, flags=fl,
                 w=_w(dst, d, b, rows | (cols << 16), drs, brs | (V_OUTER << 16) | (B_ROW << 24),
                      c),
                 comment=comment)


def mac(dst, a, b, rows, cols, drs, ars, bmode=B_COL, imm=0.0, ra=0, rb=0, rc=0,
        comment="mac"):
    """VOP MAC (Config.VMAC): T[dst + r*drs + c] = add(T[dst + r*drs + c], mul(A(r, c), B)) in
    place, B per column (T[b + c], cols <= 256: read before anything is written, so it may
    overlap dst) or the scalar imm; the MUL then ADD of `dst + A * B` bit for bit."""
    _check(bmode in (B_COL, B_SCALAR) and (bmode == B_SCALAR or cols <= MAC_MAX_COLS),
           f"MAC: bmode {bmode}, cols {cols} (B per column, cols <= {MAC_MAX_COLS}, or a scalar)")
    return vop(V_MAC, dst, a, b if bmode == B_COL else 0, rows, cols, drs, ars, 0, bmode, imm,
               ra=ra, rb=rb, rc=rc, comment=comment)


def exp2sub_sum(dst, a, b, sums, rows, cols, drs, ars, brs, bmode=B_ROW, ra=0, rb=0, rc=0,
                rd=0, comment="exp2sub"):
    """VOP EXP2SUB with FSUM (Config.FSUM): T[dst + r*drs + c] = exp2(A - B) and T[sums + R[rd] +
    r] = isum_64 of row r's results (RSUM's of them, bit for bit), written by its take
    (fsum_take; untouched until then). B not a scalar (the immediate word is the sums' address),
    rows <= 256, cols >= 64."""
    _check(bmode != B_SCALAR and 0 < rows <= FSUM_MAX_ROWS and FSUM_MIN_COLS <= cols < 65536,
           f"EXP2SUB FSUM: bmode {bmode}, rows {rows}, cols {cols} (B not a scalar, rows <= "
           f"{FSUM_MAX_ROWS}, cols >= {FSUM_MIN_COLS})")
    ins = vop(V_EXP2SUB, dst, a, b, rows, cols, drs, ars, brs, bmode, 0.0, ra=ra, rb=rb, rc=rc,
              rd=rd, comment=comment)
    ins.flags = F_FSUM
    ins.w[6] = u32(sums)
    return ins


def fsum_take(sums, a, rows, cols, ars, ra=0, rb=0, comment="rsum fsum"):
    """VOP RSUM with FSUM: the take of the oldest EXP2SUB FSUM's row sums not taken yet, at
    T[sums + R[ra] + r] (that EXP2SUB's): done once they are written, its footprint RSUM's of A
    (T[a + R[rb]]: that EXP2SUB's results, [rows, cols] at row stride ars). It reads and
    computes nothing."""
    ins = vop(V_RSUM, sums, a, 0, rows, cols, 1, ars, 0, B_FULL, 0.0, ra=ra, rb=rb,
              comment=comment)
    ins.flags = F_FSUM
    return ins


def gather(src, dst, rows, cols, srs, drs, seg, ra=0, rb=0, comment=""):
    _check(0 < rows < 65536 and 0 < cols < 65536, f"GATHER: rows {rows}, cols {cols}")
    return Instr(GATHER, ra=ra, rb=rb, w=_w(src, dst, rows | (cols << 16), srs, drs, seg),
                 comment=comment)


def assemble(prog: list[Instr]) -> np.ndarray:
    return np.array([x for ins in prog for x in ins.encode()], dtype=np.uint32)


def disassemble(prog: list[Instr]) -> str:
    return "\n".join(f"{i:4d}: {ins}" for i, ins in enumerate(prog))
