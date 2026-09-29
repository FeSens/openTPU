"""openTPU instruction encoding (docs/isa.md). Every instruction is eight 32-bit words."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NOP, HALT, LI, ADDI, LOOP, BAR = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05
LD, ST, DSTEP, STREAM = 0x10, 0x11, 0x12, 0x13
MM, QACT, QST = 0x20, 0x21, 0x22
VOP = 0x30
GATHER = 0x40

OPNAMES = {NOP: "NOP", HALT: "HALT", LI: "LI", ADDI: "ADDI", LOOP: "LOOP", BAR: "BAR",
           LD: "LD", ST: "ST", DSTEP: "DSTEP", STREAM: "STREAM", MM: "MM", QACT: "QACT",
           QST: "QST", VOP: "VOP", GATHER: "GATHER"}

# MM / QACT / QST flags
F_UNIT, F_ACC, F_RMAX, F_ASCALE = 0x1, 0x2, 0x4, 0x8     # MM
# MM weight format, flags[5:4] (docs/isa.md "Weight formats")
WF_SHIFT = 4
W8, W4I, W4F = 0, 1, 2          # int8 + fp32 scale | int4 / E2M1 + two-level scale word
WFORMATS = {"int8": W8, "int4": W4I, "fp4": W4F}
F_PAIR = 0x40                   # MM, 4-bit only: column reuse, two K-blocks per cycle and row
F_ROW, F_CSCALE, F_RSCALE = 0x1, 0x2, 0x4   # QACT (QST: F_ROW)
F_DUP = 0x8                     # QACT: also write the rows to ACT rows rows..2*rows-1 (PAIR)
F_HALF = 0x2                    # QST, ROW mode: write only the first half of each row

# VOP functions
V_ADD, V_SUB, V_RSUB, V_MUL, V_MAX, V_MIN, V_OUTER = 0, 1, 2, 3, 4, 5, 6
V_COPY, V_EXP2, V_RECIP, V_RSQRT, V_ABS, V_FILL, V_EXP2SUB, V_LOG2 = 8, 9, 10, 11, 12, 13, 14, 15
V_RSUM, V_RMAX, V_RSSQ, V_RDOT = 16, 17, 18, 19
VFUNCS = {V_ADD: "add", V_SUB: "sub", V_RSUB: "rsub", V_MUL: "mul", V_MAX: "max",
          V_MIN: "min", V_OUTER: "outer", V_COPY: "copy", V_EXP2: "exp2", V_RECIP: "recip",
          V_RSQRT: "rsqrt", V_ABS: "abs", V_FILL: "fill", V_EXP2SUB: "exp2sub",
          V_LOG2: "log2", V_RSUM: "rsum", V_RMAX: "rmax", V_RSSQ: "rssq", V_RDOT: "rdot"}
BINARY = {V_ADD, V_SUB, V_RSUB, V_MUL, V_MAX, V_MIN, V_FILL, V_EXP2SUB}
REDUCE = {V_RSUM, V_RMAX, V_RSSQ, V_RDOT}
READS_B = BINARY | {V_RDOT}          # functions that read operand B in its bmode (OUTER: B_ROW)
F_DSCALAR, F_DONE = 0x1, 0x2         # VOP OUTER: decay T[d] for every column / decay 1.0
OUTER_MAX_COLS = 256                 # OUTER: its column vectors are held in 256-word buffers

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
        for r in (self.ra, self.rb, self.rc, self.rd):
            assert 0 <= r < 16
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


def _w(*vals) -> list:
    v = [u32(x) for x in vals]
    return v + [0] * (7 - len(v))


def nop(comment=""):
    return Instr(NOP, comment=comment)


def halt():
    return Instr(HALT)


def li(rd, imm, comment=""):
    return Instr(LI, rd=rd, w=_w(imm), comment=comment)


def addi(rd, ra, imm, comment=""):
    return Instr(ADDI, rd=rd, ra=ra, w=_w(imm), comment=comment)


def loop(body_len, count, rcount=0, comment=""):
    return Instr(LOOP, ra=rcount, w=_w(body_len, count), comment=comment)


def bar():
    return Instr(BAR)


def ld(dram, tmem, nwords, ra=0, rb=0, comment=""):
    return Instr(LD, ra=ra, rb=rb, w=_w(dram, tmem, nwords), comment=comment)


def st(dram, tmem, nwords, ra=0, rb=0, comment=""):
    return Instr(ST, ra=ra, rb=rb, w=_w(dram, tmem, nwords), comment=comment)


DSTEP_LANES = 8                      # DSTEP: state words per cycle (timing only)
DSTEP_MAX_ROWS = 256                 # DSTEP: o is held in a 256-word buffer
F_DZERO = 0x1                        # DSTEP: the state starts at +0 (DRAM is not read)


def dstep(dram, qk, v, rows, cols, g, gs, o, zero=False, ra=0, rb=0, rc=0, comment=""):
    """DSTEP (DMA): one Gated DeltaNet head step on an fp32 state St [rows, cols] (row-major)
    in DRAM at byte address R[ra] + dram, updated in place. q = T[qk + c], k = T[qk + cols + c]
    (qk: R[rb] + w2), v(r) = T[v + r] (R[rc] + w3), the decay e = T[g], beta = T[g + gs], and
    o(r) = T[o + r] is written. Row by row: kv = rdot(St[r], k), d = (v(r) - kv * e) * beta,
    St[r] = St[r] * e + d * k, o(r) = rdot(St[r], q) -- the rounding of RDOT, MUL, SUB, MUL,
    OUTER, RDOT (docs/isa.md). `zero`: the state starts at +0 and is not read (position 0)."""
    return Instr(DSTEP, ra=ra, rb=rb, rc=rc, flags=F_DZERO if zero else 0,
                 w=_w(dram, qk, v, rows | (cols << 16), g, o, gs), comment=comment)


# ------------------------------------------------------------------------------------ STREAM
# The stream engine's instruction (docs/stream.md): rows of a stream (DRAM or TMEM) pass
# REDUCE A -> SCALAR -> MAP U -> MAP/REDUCE Q under a descriptor of a few TMEM words.
F_SZERO = 0x1                        # the stream reads as +0 (nothing is read)
F_SRC_T = 0x2                        # the stream source is TMEM (words), not DRAM (bytes)
F_DST_T = 0x4                        # the stream destination is TMEM
F_NODST = 0x8                        # the stream is not written back

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
    assert 0 <= p < 1 << 24
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
        assert 0 < self.rows < 1 << 12 and 0 < self.cols < 1 << 12
        assert len(self.ops) <= STREAM_MAX_OPS
        assert self.srs < 1 << 12 and self.drs < 1 << 12
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
           nodst=False, ra=0, rb=0, rc=0, rd=0, comment=""):
    """STREAM (docs/stream.md): the descriptor at TMEM `desc` over the rows of the stream at
    src + R[ra] (DRAM bytes, or TMEM words with src_t), Y written to dst + R[ra]; column slots
    at vec + R[rb] (slot i at + i * cols), row scalars X[r] = T[x + R[rc] + r], constants
    K_j = T[k + R[rd] + j * ks], row outputs O[r] = T[out + r]. desc and ks share w1 (desc in
    [15:0], ks in [31:16]) so the scoreboard knows the constants' range."""
    assert 0 <= desc < 1 << 16 and 0 <= ks < 1 << 16
    fl = (F_SZERO if zero else 0) | (F_SRC_T if src_t else 0) | (F_DST_T if dst_t else 0) | \
        (F_NODST if nodst else 0)
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
       ascale=None, wf=W8, pair=False, ra=0, rb=0, rc=0, comment=""):
    """MM. With `ascale` (a TMEM address; needs unit and acc) the old accumulator is first
    multiplied by a per-row factor: y = T[out] * T[ascale + j] + a.w (the flash-attention
    rescale, done in the MXU epilogue). The address travels in the (unused) scale field.
    `wf`: the streamed weights' format (W8, W4I, W4F). `pair` (4-bit only): column reuse,
    ACT row j + m carries the odd K-blocks of row j and both terms of a chunk are summed
    before the partial sums (docs/isa.md, "Column reuse")."""
    assert 0 < n < 65536 and 0 < kb < 65536 and 0 < m < 256 and 0 <= ab < 256 and ors < 65536
    assert wf in (W8, W4I, W4F)
    assert not pair or wf != W8, "PAIR needs 4-bit weights"
    if ascale is not None:
        assert unit and acc, "ASCALE needs UNIT and ACC"
        ssa = ascale
    fl = (F_UNIT if unit else 0) | (F_ACC if acc else 0) | (F_RMAX if rmax else 0) | \
        (F_ASCALE if ascale is not None else 0) | (F_PAIR if pair else 0) | (wf << WF_SHIFT)
    return Instr(MM, ra=ra, rb=rb, rc=rc, flags=fl,
                 w=_w(sa, ssa, out, n | (kb << 16), rs, ors | (m << 16) | (ab << 24), srs),
                 comment=comment)


def qact(src, rows, ab, kb, srs, row=False, cscale=None, rscale=None, dup=False, ra=0,
         comment=""):
    """QACT. `cscale`/`rscale`: TMEM addresses of a per-column / per-row factor applied before
    quantization, x' = (x * T[rscale + r]) * T[cscale + c]. `dup`: ACT row r + rows receives
    a copy of row r in the same cycles (the odd K-blocks' column of an MM PAIR)."""
    assert 0 < rows < 256 and 0 <= ab < 256 and 0 < kb < 65536
    fl = (F_ROW if row else 0) | (F_CSCALE if cscale is not None else 0) | \
        (F_RSCALE if rscale is not None else 0) | (F_DUP if dup else 0)
    return Instr(QACT, ra=ra, flags=fl,
                 w=_w(src, rows | (ab << 8) | (kb << 16), srs, cscale or 0, rscale or 0),
                 comment=comment)


def qst(src, dst, sdst, rows, kb, srs, drs, es, row=False, half=False, ra=0, rb=0, rc=0,
        comment=""):
    assert 0 < rows < 65536 and 0 < kb < 65536
    assert row or not half, "QST HALF needs ROW mode"
    return Instr(QST, ra=ra, rb=rb, rc=rc, flags=(F_ROW if row else 0) | (F_HALF if half else 0),
                 w=_w(src, dst, sdst, rows | (kb << 16), srs, drs, es), comment=comment)


def vop(func, dst, a, b, rows, cols, drs, ars, brs, bmode=B_FULL, imm=0.0,
        ra=0, rb=0, rc=0, comment=""):
    assert 0 < rows < 65536 and 0 < cols < 65536
    assert drs < 65536 and ars < 65536 and brs < 65536
    return Instr(VOP, ra=ra, rb=rb, rc=rc,
                 w=_w(dst, a, b, rows | (cols << 16), drs | (ars << 16),
                      brs | (func << 16) | (bmode << 24), f32bits(imm)),
                 comment=comment)


def outer(dst, d, b, c, rows, cols, drs, brs, dmode="scalar", ra=0, rb=0, rc=0, rd=0,
          comment=""):
    """VOP OUTER: T[dst + r*drs + j] = T[dst + r*drs + j] * Dv(j) + T[b + r*brs] * T[c + j],
    Dv(j) = T[d] (dmode "scalar"), T[d + j] ("column") or 1.0 ("one", d unused). The decay
    address travels in the A field and the column vector's in the immediate word (w7 += R[rd])."""
    assert 0 < rows < 65536 and 0 < cols <= OUTER_MAX_COLS and drs < 65536 and brs < 65536
    fl = {"scalar": F_DSCALAR, "column": 0, "one": F_DONE}[dmode]
    return Instr(VOP, ra=ra, rb=rb, rc=rc, rd=rd, flags=fl,
                 w=_w(dst, d, b, rows | (cols << 16), drs, brs | (V_OUTER << 16) | (B_ROW << 24),
                      c),
                 comment=comment)


def gather(src, dst, rows, cols, srs, drs, seg, ra=0, rb=0, comment=""):
    return Instr(GATHER, ra=ra, rb=rb, w=_w(src, dst, rows | (cols << 16), srs, drs, seg),
                 comment=comment)


def assemble(prog: list[Instr]) -> np.ndarray:
    return np.array([x for ins in prog for x in ins.encode()], dtype=np.uint32)


def disassemble(prog: list[Instr]) -> str:
    return "\n".join(f"{i:4d}: {ins}" for i, ins in enumerate(prog))
