"""Chunked Gated DeltaNet prefill on openTPU (docs/deltanet_chunk.md): the kernel's operations by unit and
their issue cycles, per value head and chunk, on one slice of bcu1525-x2:wmx (MXU_PUMP, VPU_CNS 5, GALL
WG 8, LANES 8, two DIMMs), at tools/pf500.py's rates (no stalls, no overlap; the busiest unit bounds):
  MXU   an MM's advances: ceil(N / WG) x ceil(M / 2) x KB (int8 streamed rows in groups, two
        activation rows a pass)
  VPU   rows x ceil(cols / LANES) x passes (EXP2 / EXP2SUB 2 at CNS 5); OUTER max(2, ch) + rows ch
  Q     QACT: its TMEM reads, LANES words a cycle (pass 1 overlaps the next row's pass 0); QST: pass 0
        at LANES, pass 1 a word (4 bytes) a cycle in rows mode, a byte a cycle in COL mode
        (otpu_quant.sv), unless --qstw sets pass 1's bytes a cycle
  DRAM  bytes over 256 a cycle (2 DIMMs x 19.2 GB/s at 150 MHz)
The 35B: 32 value heads on 16 key heads (q, k shared by a pair), d 128, so 8 value heads a slice at S 4.

The kernel (docs in the Phase 0 note): per key head and chunk K's rows QST'd (M1 / M4's streamed
operand) and K^T (M3 / M8's: an identity MM from K's rows, then rows mode, or a COL QST, whichever is
cheaper), Q QACT'd and A0 = Q K^T (M4); per value head the decays (cumsum, exp2), Gamma's two masked
tiles (FILL, EXP2SUB, MIN against the 0/1 masks), M1 (beta in QACT RSCALE), L and A (MUL), T = (I+L)^-1
(forward substitution: OUTER, else FILL + MUL + SUB a column), V^T (as K^T), M2, M3 (-beta e^gc in
CSCALE), M5 (ACC into U0), M6 (e^gc in RSCALE), U^T (identity MM, as K^T), M7 (U^T's rows QST'd), M8
(St' = e^gl St + U^T Kd: ACT U^T with e^(gl - gc) in CSCALE, ASCALE e^gl, into the fp32 state in
TMEM), St's rows QST'd for the next chunk's M5 / M6. A matmul's precision (dn_chunk's letters): q one
int8 pass, h three (hi.hi + hi.lo + lo.hi), a / w two (the activation's / the streamed side's split).
A split side's lo part: the quantizer's (--splitq: QACT / QST write hi and lo in one pass), else
today's units: deq(hi) by an MM against an int8 identity (UNIT: exactly the hi values the MXU uses;
a streamed side's hi through a QACT ROW, whose q8 values are QST's at KB 1), the residual x - deq(hi)
a VPU SUB, and its own QACT / QST (the VPU alone would take 7 passes: ABS, RMAX, MUL, q8 as ADD and
SUB of 1.5 2^23, MUL, SUB).

    python tools/dn_cost.py [--C 64] [--mode q1h2345678] [--outer] [--splitq] [--qstw 4] [--tout]
                            [--R 512] [--heads 8]
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict

D, WG, LANES, DRAM_BPC = 128, 8, 8, 256.0
EXP_PASSES = 2                                  # VPU_CNS 5


class Kernel:
    def __init__(self, a):
        self.a = a
        self.cyc = defaultdict(float)           # unit -> cycles
        self.ops = defaultdict(float)           # (unit, op) -> cycles
        self.dram = 0.0
        self.f = 1.0                            # the share of the work this value head takes

    def add(self, unit, op, c):
        self.cyc[unit] += c * self.f
        self.ops[unit, op] += c * self.f

    # ---------------------------------------------------------------- units
    def mm(self, op, N, M, K, passes=1, w_bytes=True):
        """out [M, N] = ACT [M, K] . W [N, K]^T: MMs of up to 16 activation rows; W streamed per MM."""
        KB = -(-K // D)
        mms = -(-M // 16)
        c = 0
        for i in range(mms):
            m = min(16, M - 16 * i)
            c += -(-N // WG) * -(-m // 2) * KB
        self.add("MXU", op, c * passes)
        if w_bytes:
            self.dram += passes * mms * N * KB * D * self.f

    def vop(self, op, rows, cols, passes=1):
        self.add("VPU", op, rows * -(-cols // LANES) * passes)

    def outer(self, op, rows, cols):
        ch = -(-cols // LANES)
        self.add("VPU", op, max(2, ch) + rows * ch)

    def qact(self, op, rows, cols, split=False):
        cols = -(-cols // D) * D
        self.add("Q", op, rows * cols / LANES)
        if split and self.a.lo == "all":        # a second QACT with LO: lo alone, the same rows
            self.add("Q", op + " LO", rows * cols / LANES)
        if split and not self.a.splitq:         # deq(hi) by the identity MM, the SUB, lo's QACT
            self.mm(op + " lo: deq(hi)", cols, rows, cols)
            self.vop(op + " lo: SUB", rows, cols)
            self.add("Q", op + " lo", rows * cols / LANES)

    def qst(self, op, rows, cols, col=False, split=False):
        """rows x cols elements to DRAM int8 (COL: the transposed matrix)."""
        cols_p = -(-cols // D) * D if not col else cols
        bpc = (self.a.qstw_col if col else self.a.qstw)
        e = rows * cols_p
        # (--splitq: pass 1 writes hi's and lo's bytes, both through pass 1's bpc bytes a cycle)
        self.add("Q", op, e / LANES + (2 if split and self.a.splitq and not self.a.lo else 1) * e / bpc)
        if split and self.a.lo:                 # a second QST with LO: both passes again, lo alone
            self.add("Q", op + " LO", e / LANES + e / bpc)
        self.dram += (2 if split else 1) * e * self.f
        if split and not self.a.splitq:         # hi again by a QACT ROW, deq(hi), SUB, lo's QST
            self.add("Q", op + " lo: QACT ROW", e / LANES)
            self.mm(op + " lo: deq(hi)", cols_p, rows, cols_p)
            self.vop(op + " lo: SUB", rows, cols_p)
            self.add("Q", op + " lo", e / LANES + e / bpc)

    def ld(self, op, words):
        self.dram += 4 * words * self.f

    # ---------------------------------------------------------------- the transposes
    def transpose(self, op, rows, cols, split):
        """X [rows, cols] (TMEM) -> its transpose as a streamed operand in DRAM (rows mode): an
        identity MM from X's rows (QST'd first) and the transposed tile's rows QST'd, or a COL QST of X,
        the cheaper on Q; with --tout an MM writes X^T, so only the rows mode QST of X^T is left.
        Returns whether X^T is in TMEM (the identity path or --tout)."""
        if self.a.tout:
            self.qst(op + " X^T rows", cols, rows, split=split)
            return True
        e = rows * cols
        col_q = e / LANES + e / self.a.qstw_col
        id_q = 2 * (e / LANES + e / self.a.qstw)
        if col_q <= id_q:
            self.qst(op + " COL", rows, cols, col=True, split=split)
            return False
        self.qst(op + " X rows", rows, cols, split=split)
        self.mm(op + " identity", rows, cols, rows, 2 if split else 1)
        self.qst(op + " X^T rows", cols, rows, split=split)
        return True


def parse(mode: str) -> dict:
    how = {i: "q" for i in range(1, 9)}
    for h, ids in re.findall(r"([qhaw])(\d+)", mode):
        how.update({int(i): h for i in ids})
    return how


PASSES = {"q": 1, "a": 2, "w": 2, "h": 3}


def chunk(a) -> Kernel:
    """One value head's share of a chunk of C rows."""
    k = Kernel(a)
    C, how = a.C, parse(a.mode)
    sa = {i: how[i] in "ah" for i in how}       # activation side split
    sw = {i: how[i] in "wh" for i in how}       # streamed side split
    p = {i: PASSES[how[i]] for i in how}
    # inputs: q, k (a key head's, shared by its two value heads: half each), v [C, d] fp32 from
    # DRAM (item 2's rows by block)
    k.ld("v", C * D)
    k.f = 0.5
    k.ld("q k", 2 * C * D)
    # key head: K rows (M1, M4's streamed side), K^T (M3, M8's), Q's QACT and M4
    k.qst("K rows", C, D, split=sw[1] or sw[4])
    kt_tmem = k.transpose("K^T", C, D, sw[3] or sw[8])
    k.qact("Q", C, D, split=sa[4])
    k.mm("M4 Q.K^T", C, C, D, p[4])
    k.f = 1.0
    # decays: cumsum (6 ADDs over [heads, C], a head's share), exp2(gc), exp2(gl - gc), the vectors
    k.vop("gc cumsum, exp2", 1, C, 6 + 3 * EXP_PASSES + 4)
    # Gamma: FILL (gc_r), EXP2SUB (gc_c), MIN with the strict and the lower masks; L, A
    k.vop("Gamma FILL", C, C)
    k.vop("Gamma EXP2SUB", C, C, EXP_PASSES)
    k.vop("Gamma MIN x2", C, C, 2)
    k.qact("K beta", C, D, split=sa[1])
    k.mm("M1 Kb.K^T", C, C, D, p[1])
    k.vop("L, A MUL", C, C, 2)
    # T = (I + L)^-1: from I (a COPY), a column s at a time, rows s+1.. and columns 0..s (OUTER with
    # DONE: T[r, c] += (-L[r, s]) T[s, c], -L from M1's -beta in its RSCALE)
    k.vop("solve T = I", C, C)
    for s in range(C - 1):
        if a.outer:
            k.outer("solve OUTER", C - 1 - s, s + 1)
        else:
            k.vop("solve FILL MUL SUB", C - 1 - s, s + 1, 3)
    # M2 U0 = T . Vb (V^T streamed), M3 Wc = -T . Kbe (K^T), M5 U = U0 + Wc . S, M6 O = Qe . S
    k.transpose("V^T", C, D, sw[2])
    k.qact("T (beta vs)", C, C, split=sa[2])
    k.mm("M2 T.Vb", D, C, C, p[2])
    k.qact("T (-beta e^gc ks)", C, C, split=sa[3])
    k.mm("M3 T.Kbe", D, C, C, p[3])
    k.qact("Wc", C, D, split=sa[5])
    k.mm("M5 Wc.S", D, C, D, p[5])
    k.qact("Q e^gc", C, D, split=sa[6])
    k.mm("M6 Qe.S", D, C, D, p[6])
    # U^T: M7's streamed side (its rows) and M8's activation (in TMEM)
    ut_tmem = k.transpose("U^T", C, D, sw[7] or sa[8])
    if not ut_tmem:                             # (a COL QST leaves U^T in DRAM only: M8 needs it in
        k.qst("U rows (M8)", C, D, split=sa[8])  # TMEM, so the identity MM after all)
        k.mm("U^T identity (M8)", D, C, D, 2 if sa[8] else 1)
    k.qact("A", C, C, split=sa[7])
    k.mm("M7 A.U", D, C, C, p[7])
    k.qact("U^T (e^(gl-gc))", D, C, split=sa[8])
    k.mm("M8 U^T.Kd", D, D, C, p[8])
    if kt_tmem or a.tout:           # K^T's rows quantized a row: not UNIT, so no ASCALE: the decay
        k.vop("St decay MUL", D, D)  # e^gl on the VPU (a COL K^T is UNIT: ASCALE in the epilogue)
    k.qst("St rows", D, D, split=sw[5] or sw[6])
    return k


def per_row(a) -> dict:
    k = chunk(a)
    # a pass's state: load and store (16-bit, LD / ST with ST16), once per R rows
    k.ld("St in/out", 2 * D * D * 0.5 * a.C / a.R)
    out = {u: k.cyc[u] / a.C for u in ("MXU", "VPU", "Q")}
    out["DRAM"] = k.dram / DRAM_BPC / a.C
    out["ops"] = {kk: v / a.C for kk, v in k.ops.items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--mode", default="h12345678")
    ap.add_argument("--outer", action="store_true", help="VOP OUTER (Config.OUTER) for the solve")
    ap.add_argument("--splitq", action="store_true", help="QACT / QST emit hi and lo in one pass")
    ap.add_argument("--lo", nargs="?", const="all", default="", choices=("all", "qst"),
                    help="with --splitq: lo by a second QACT / QST (qst: QST only) with flag LO "
                    "(both passes again) instead of in the same pass")
    ap.add_argument("--qstw", type=float, default=4, help="QST rows mode pass 1 bytes a cycle")
    ap.add_argument("--qstw-col", type=float, default=1, help="QST COL pass 1 bytes a cycle")
    ap.add_argument("--tout", action="store_true", help="MM writes its output transposed")
    ap.add_argument("--R", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8, help="value heads a slice")
    ap.add_argument("--ops", action="store_true")
    a = ap.parse_args(argv)
    r = per_row(a)
    u = {k: v for k, v in r.items() if k != "ops"}
    b = max(u, key=u.get)
    print(f"C {a.C} mode {a.mode} outer {a.outer} splitq {a.splitq} qstw {a.qstw}/{a.qstw_col} "
          f"tout {a.tout}: per row and value head " + "  ".join(f"{k} {v:.0f}" for k, v in u.items())
          + f"  bound {u[b]:.0f} ({b}); a row's DN core on the slice ({a.heads} heads) "
          f"{a.heads * u[b]:.0f} cycles (DSTEP's 2,048 a head: {a.heads * 2048})")
    if a.ops:
        for (unit, op), v in sorted(r["ops"].items(), key=lambda t: -t[1]):
            print(f"  {unit:4s} {op:28s} {v:8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
