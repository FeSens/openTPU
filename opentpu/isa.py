"""openTPU instruction encoding (docs/isa.md). Every instruction is eight 32-bit words."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NOP, HALT, LI, ADDI, LOOP, BAR = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05
LD, ST = 0x10, 0x11
MM, QACT, QST = 0x20, 0x21, 0x22
VOP = 0x30
GATHER = 0x40

OPNAMES = {NOP: "NOP", HALT: "HALT", LI: "LI", ADDI: "ADDI", LOOP: "LOOP", BAR: "BAR",
           LD: "LD", ST: "ST", MM: "MM", QACT: "QACT", QST: "QST", VOP: "VOP",
           GATHER: "GATHER"}

# MM / QACT / QST flags
F_UNIT, F_ACC, F_RMAX, F_ASCALE = 0x1, 0x2, 0x4, 0x8     # MM
F_ROW, F_CSCALE, F_RSCALE = 0x1, 0x2, 0x4   # QACT (QST: F_ROW)

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


def mm(sa, ssa, out, n, kb, rs, ors, m, ab, srs, unit=False, acc=False, rmax=False,
       ascale=None, ra=0, rb=0, rc=0, comment=""):
    """MM. With `ascale` (a TMEM address; needs unit and acc) the old accumulator is first
    multiplied by a per-row factor: y = T[out] * T[ascale + j] + a.w (the flash-attention
    rescale, done in the MXU epilogue). The address travels in the (unused) scale field."""
    assert 0 < n < 65536 and 0 < kb < 65536 and 0 < m < 256 and 0 <= ab < 256 and ors < 65536
    if ascale is not None:
        assert unit and acc, "ASCALE needs UNIT and ACC"
        ssa = ascale
    fl = (F_UNIT if unit else 0) | (F_ACC if acc else 0) | (F_RMAX if rmax else 0) | \
        (F_ASCALE if ascale is not None else 0)
    return Instr(MM, ra=ra, rb=rb, rc=rc, flags=fl,
                 w=_w(sa, ssa, out, n | (kb << 16), rs, ors | (m << 16) | (ab << 24), srs),
                 comment=comment)


def qact(src, rows, ab, kb, srs, row=False, cscale=None, rscale=None, ra=0, comment=""):
    """QACT. `cscale`/`rscale`: TMEM addresses of a per-column / per-row factor applied before
    quantization, x' = (x * T[rscale + r]) * T[cscale + c]."""
    assert 0 < rows < 256 and 0 <= ab < 256 and 0 < kb < 65536
    fl = (F_ROW if row else 0) | (F_CSCALE if cscale is not None else 0) | \
        (F_RSCALE if rscale is not None else 0)
    return Instr(QACT, ra=ra, flags=fl,
                 w=_w(src, rows | (ab << 8) | (kb << 16), srs, cscale or 0, rscale or 0),
                 comment=comment)


def qst(src, dst, sdst, rows, kb, srs, drs, es, row=False, ra=0, rb=0, rc=0, comment=""):
    assert 0 < rows < 65536 and 0 < kb < 65536
    return Instr(QST, ra=ra, rb=rb, rc=rc, flags=F_ROW if row else 0,
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
