"""Wait entries: the compiler's half of the sequencer's explicit dependencies (otpu_seq WAITC;
docs/isa.md "Wait counts", docs/seq_vpu.md).

With WAITC the sequencer computes no footprints and compares no ranges. Each instruction carries a
64-bit wait entry beside it: a word of seven 4-bit counts, k_j in bits [4j+3:4j], and a word with
the VPU mask in bits [30:0].
  count k_j = k > 0 holds the instruction until the k-th last instruction of unit U(j) before it
  (in program order) has completed, and with it every older one of that unit (DMA, MXU, QUANT and
  COLL start and complete in order); k = 0 waits for nothing on U(j).
    j = 0, 1, 2, 4: unit j (DMA, MXU, QUANT, COLL), before the instruction starts (an MM: before
                    its release, when it reads the ACT RAM and writes TMEM);
    j = 5, 6:       an MM's stream start (its DRAM reads): the DMA's and the quantizer's counts
                    over DRAM conflicts only;
    j = 3:          unused (0): the VPU's is the mask.
  FOLLOW, bit 28: a VOP's k_4 = 1 holds it only until the GATHER it names has started, and the VPU
  reads its A behind the GATHER's writes into its slice's TMEM (follows()).
  VPU mask bit r - 1 holds the instruction until the r-th last VPU instruction before it has
  completed (r = 1..31). The VPU starts its oldest ready instruction, so they complete out of
  order: the mask names each producer, not the nearest and all older.
An in-order unit's own count is not needed (it starts in order and finishes an instruction before
the next touches memory); the VPU's own producers are in its mask. A producer 16 or more of its
unit back has completed: for an in-order unit an incomplete one holds a window slot, and so does
each younger one of its unit (the window has 16 slots); the sequencer dispatches a VPU
instruction only once the one 31 before it has completed (a VPU instruction's slot is freed when
it completes, so younger ones pass an old one blocked on its producers; docs/seq_vpu.md 4.5: a
hold at 15 cost up to 0.8% of prefill). So 4-bit counts and a 31-bit mask are exact.

A window of 32 (Config.WC_WIN 32, `win`; docs/bcu1525_s2.md): an in-order
unit's producer 16 to 31 back may still run. The pass keeps each in-order unit's last 31, and a
count saturates at 15: the unit completes in order, so its 15th last done means every older one
done (conservative: such a producer is waited for through a younger one of its unit). 32 or
more back has completed (the window's 32 slots). The VPU's 31-bit mask is the same at both. At
16 the entries are the same as before the field.

wait_counts() computes the entries from the program alone, conservatively: an abstract execution
of the program (registers as intervals, the run's arguments R8..R15 and every RLD unknown unless
given or bounded), each instruction's footprint the otpu_pkg footprint() of every value its
registers may hold, and per unit the last 15 instructions' footprints (the VPU's last 31), joined
where paths meet (a loop's exits, a count the program does not know). A pc's count is the smallest
over all its visits, its mask the union. Loops the budget cannot run (or whose count is unknown
beyond CAP iterations) run on widened registers to a fixed point, which covers every later
iteration. The pass visits every instruction a run executes; _MemoPass (the default) keeps
each visit's checks of the TMEM and ACT footprints, which repeat, and checks the DRAM ranges
apart: the same entries, 2-5 times faster (MiniCPM-V's tower at 1,024 patches: 9.5 -> 2.4 s).

Checker checks entries against a concrete execution (the ISA simulator's): every conflict otpu_pkg
conflict() finds with an older instruction that may still be running is waited for.

At AW > 32 (Config.AW, docs/isa.md "Address bits": the BCU's 36) the registers' intervals and the
DRAM ranges are taken mod 2^AW, the TMEM and ACT ranges mod 2^32 (an index's low bits), LOOP's
count and QST GRP's row 32 bits, and WAITW's DRAM footprint all of 2^AW: mod 2^32 a range across
a 4 GiB line would be clamped there, a dependency missed (with SEQ_WC there is no footprint
scoreboard behind the entries to catch it), and ranges 4 GiB apart would alias.

On a WAITC device (Config.SEQ_WC; the bitstream's CFG_UARCH bit24) the host loads every program
with its entries (opentpu/host/board.py Board.load_program refuses one without them): loadable()
places them after the program's last chunk, where the slice's loader reads them, IMEM load and
HALT CHAIN alike; area() is a program's room in DRAM; entries() computes them once per program
(its words, its arguments' bounds) and keeps them beside the program cache's programs.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from . import isa as I

M32 = 0xFFFFFFFF
TOP = (0, M32)
SP_TMEM, SP_DRAM, SP_ACT = 0, 1, 2
U_DMA, U_MXU, U_Q, U_VPU, U_COLL = 0, 1, 2, 3, 4
NUNITS, NW = 5, 7
WIN = 16                          # the window the entries assume by default (Config.WC_WIN)
WINS = (16, 32)                   # the windows they may assume
KMAX = 15                         # ranks a 4-bit count names (at a window of 16: every producer
                                  #   that may still run; at 32 the counts saturate here)
VMAX = 31                         # VPU instructions the mask names (the sequencer's hold)
VMASK = (1 << VMAX) - 1           # the VPU mask: the last 31 VPU instructions
DEFAULT = 0x0111_0111 | (VMASK << 32)  # a pc never reached: every unit's last, the VPU's 31
FOLLOW = 1 << 28                  # the counts' word: a VOP follows the GATHER its k_4 names
FOLLOWS = (I.V_RSUM, I.V_RSSQ, I.V_RMAX, I.V_ARGMAX)    # the functions that may (A alone)
UNIT = {I.LD: U_DMA, I.ST: U_DMA, I.DSTEP: U_DMA, I.STREAM: U_DMA, I.WAITW: U_DMA,
        I.LDA: U_DMA, I.MM: U_MXU, I.QACT: U_Q, I.QST: U_Q, I.VOP: U_VPU,
        I.BAR: U_COLL, I.GATHER: U_COLL, I.RLD: U_COLL}
WUNIT = (U_DMA, U_MXU, U_Q, U_VPU, U_COLL, U_DMA, U_Q)
DEPTH = tuple(VMAX if u == U_VPU else KMAX for u in range(NUNITS))   # history per unit (window 16)


def _depth(win: int) -> tuple:
    """The history per unit at a window of `win`: an in-order unit's last win - 1, the VPU's
    VMAX."""
    if win not in WINS:
        raise ValueError(f"waits: a window of {win} (the entries assume one of {WINS})")
    return tuple(VMAX if u == U_VPU else win - 1 for u in range(NUNITS))
CAP = 4                           # iterations of an unknown count run before widening
PATHS = 16                        # variant_paths' most runs passed apart
BUDGET = 4_000_000                # instruction visits before known counts widen too
MAXR = 128                        # ranges a joined footprint keeps per (space, read / write);
# past it they become one interval a space. A run of 0/1-count loops (attention's sub-block
# variants, 32 a layer in GPT-OSS's decode) joins paths whose histories are shifted against each
# other, so a slot's footprint gathers one range per path: at 12 they collapsed and an RLD of a
# table word written once waited for every TMEM write in flight, holding dispatch at its LOOP
# (GPT-OSS's decode at r60: 9,901 -> 4,527 VOP mask bits, 0.23 -> 0.40 s; the prefill tiles' and
# MiniCPM-V's tower's entries and times the same)


# ------------------------------------------------------------------------- values (intervals)
# A register's values: an interval [lo, hi] of integers taken mod 2^AW (32 but on the BCU), lo in
# [0, 2^AW) and hi < lo + 2^AW (hi >= 2^AW: the interval wraps, [lo, 2^AW) and [0, hi - 2^AW]);
# TOP: any 32-bit value (aw 32: any), _top(aw) any.
def _top(aw: int):
    return TOP if aw == 32 else (0, (1 << aw) - 1)


def _norm(lo, hi, aw: int = 32):
    m = M32 if aw == 32 else (1 << aw) - 1
    if hi - lo >= m:
        return (0, m)
    k = lo >> aw
    return (lo - (k << aw), hi - (k << aw))


def _add(v, imm, aw: int = 32):
    """v + imm mod 2^aw, imm an address offset (32 bits, zero-extended) or LI / ADDI's immediate
    (_imm)."""
    m = M32 if aw == 32 else (1 << aw) - 1
    return _norm(v[0] + (imm & m), v[1] + (imm & m), aw)


def _imm(ins: I.Instr, aw: int) -> int:
    """LI's and ADDI's immediate: {w2, w1} above AW 32, w1 at it."""
    return ins.w[0] & M32 if aw == 32 else (ins.w[0] & M32 | (ins.w[1] & M32) << 32)


def _low(v):
    """An interval mod 2^AW as the values of its low 32 bits (a TMEM index, a 32-bit field)."""
    return v if v[1] <= M32 else _norm(v[0], v[1])


def _hull(a, b, aw: int = 32):
    lo, hi = min(a[0], b[0]), max(a[1], b[1])
    m = M32 if aw == 32 else (1 << aw) - 1
    return (0, m) if hi - lo >= m else (lo, hi)


def _span(v):
    """The values as one interval within [0, 2^32) (a wrapped one: all of it)."""
    return v if v[1] <= M32 else TOP


# ------------------------------------------------------------------------- footprints
@dataclass(frozen=True)
class Fp:
    """What an instruction may touch: half-open [lo, hi) ranges per space, read and written;
    all: conflicts with everything (BAR); cg: a GATHER whose writes ascend from its dst through
    one contiguous range (one row, seg = cols: a VOP may follow it)."""
    all: bool = False
    rd: tuple = ()
    wr: tuple = ()
    cg: bool = False


def _mk(sp, base, ln, aw: int = 32):
    """otpu_pkg mk() over every value of the base: [lo, hi) clamped to the last byte; a list of
    ranges (two when the base's values wrap). DRAM's at AW bits, TMEM's and ACT's at 32 (the
    base's low 32 bits)."""
    ln &= M32
    if ln == 0:
        return []
    m = M32
    if aw > 32:
        if sp == SP_DRAM:
            m = (1 << aw) - 1
        else:
            base = _low(base)
    lo, hi = base
    if hi <= m:
        return [(sp, lo, min(hi + ln, m))]
    return [(sp, lo, m), (sp, 0, min(hi - (m + 1) + ln, m))]


def _beats(rs):
    return [(sp, lo & ~63, hi | 63) for sp, lo, hi in rs]


def _p(a, w):
    """otpu_pkg fp_prod: the 24-bit a times the 32-bit w, low 32 bits."""
    return ((a & 0xFFFFFF) * (w & M32)) & M32


def resolve(ins: I.Instr, R, aw: int = 32) -> list:
    """The register-relative fields as otpu_seq resolves them (w1..w7, intervals mod 2^aw: a
    TMEM index's or a 32-bit field's are its low 32 bits, _mk and _low)."""
    w = [(x & M32, x & M32) for x in ins.w]
    op = ins.op

    def r(k):
        return (0, 0) if k == 0 else R[k]

    def add(i, k):
        v = r(k)
        if v != (0, 0):
            w[i] = _add(v, w[i][0], aw)

    if op == I.STREAM:
        add(1, ins.ra); add(2, ins.ra); add(3, ins.rb); add(4, ins.rc); add(5, ins.rd)
        return w
    add(0, ins.ra)
    if op not in (I.QACT, I.LDA):
        add(1, ins.rb)
    if op not in (I.QACT, I.LD, I.ST, I.GATHER, I.LDA) or op == I.QACT and ins.flags & I.F_QST:
        add(2, ins.rc)                  # (QACT ST: its DRAM rows, R[rc] + w3)
    if op == I.VOP:
        add(6, ins.rd)
    if op == I.QST and ins.flags & I.F_QGRP:
        add(5, ins.rd)                  # (QST GRP's first row: R[rd] + w6)
    return w


def footprint(ins: I.Instr, R, D: int = 128, S: int = 1, aw: int = 32) -> Fp:
    """otpu_pkg footprint() of the instruction for every value its registers may hold (R: 16
    intervals mod 2^aw)."""
    op, fl = ins.op, ins.flags
    if op == I.BAR:
        return Fp(all=True)
    w = resolve(ins, R, aw)
    mk = _mk if aw == 32 else (lambda sp, base, ln: _mk(sp, base, ln, aw))
    c = [x[0] for x in w]          # immediates (the fields the RTL does not resolve)
    rd, wr = [], []
    if op == I.LD:
        rd.append(mk(SP_DRAM, w[0], c[2] << 2)); wr.append(mk(SP_TMEM, w[1], c[2]))
    elif op == I.ST:
        rd.append(mk(SP_TMEM, w[1], c[2])); wr.append(mk(SP_DRAM, w[0], c[2] << 2))
    elif op == I.STREAM:
        wr.append(mk(SP_DRAM, w[1], 4 * 256 * 256)); wr.append(mk(SP_DRAM, w[2], 4 * 256 * 256))
        wr.append(mk(SP_TMEM, w[6], 256))
        rd.append(mk(SP_TMEM, ((c[0] & 0xFFFF,) * 2), 8))
        rd.append(mk(SP_TMEM, w[3], 4 * 256)); rd.append(mk(SP_TMEM, w[4], 256))
        rd.append(mk(SP_TMEM, w[5], (c[0] >> 16) + 1))
    elif op == I.DSTEP:
        rows, cols = c[3] & 0xFFFF, c[3] >> 16
        if rows and cols:
            p0 = _p(rows - 1, cols)
            # (the state's 16-bit values with ST16: half its bytes)
            wr.append(mk(SP_DRAM, w[0], (p0 + cols) << (1 if fl & I.F_DST16 else 2)))
            wr.append(mk(SP_TMEM, w[5], rows))
            # (a pair's: head b's q | k after head a's with QKB, its e and beta a word on)
            rd.append(mk(SP_TMEM, w[1], cols << (2 if fl & I.F_DQKB else 1)))
            rd.append(mk(SP_TMEM, w[2], rows))
            rd.append(mk(SP_TMEM, w[4], c[6] + (2 if fl & I.F_DPAIR else 1)))
    elif op == I.MM:
        n, kb, k0, m = c[3] & 0xFFFF, (c[3] >> 16) & 0xFF, c[3] >> 24, (c[5] >> 16) & 0xFF
        lg = ins.rd                    # groups of 2^lg rows (docs/isa.md "Weight groups")
        if n and kb and m:
            # SPL (docs/isa.md "Split terms"): a block of hi's rows and a block of lo's HLD after
            # them, KB - 1 ACT blocks; srs w7[19:0]
            spl = bool(c[6] & I.MM_SPL)
            hld = (c[6] >> I.MM_HLD & 0x7FF) << 10 if spl else 0
            srs = c[6] & (1 << I.MM_HLD) - 1 if spl else c[6]
            p0, p1 = _p((n - 1) >> lg, c[4]), _p((n - 1) >> lg, srs)
            p2, p3 = _p(m - 1, c[5] & 0xFFFF), _p(m, c[5] & 0xFFFF)
            for h in (0, hld) if spl else (0,):
                rd.append(mk(SP_DRAM, _add(w[0], h, aw) if h else w[0],
                             p0 + (((1 if spl else kb) * D) << lg)))
            if not fl & I.F_UNIT:
                # a scale row's bytes up to block k0 + KB's (MX, docs/isa.md "Weight formats"):
                # RSC Sw; int8x Sw and a byte a block; NVFP4 Sw, three reserved words and 8
                # bytes a block; int4x / fp4x6 Sw, a reserved word and a word a block; a group's
                # scale area those of its 2^lg rows
                wf = (fl >> I.WF_SHIFT) & 3
                sb = (4 if fl & I.F_RSC else 4 + k0 + kb if wf == I.W8 else
                      24 + ((k0 + kb) << 3) if wf == I.WNV else 12 + ((k0 + kb) << 2))
                rd.append(mk(SP_DRAM, w[1], p1 + (sb << lg)))
            if fl & I.F_ASCALE:
                rd.append(mk(SP_TMEM, w[1], m))
            ab = (c[5] >> 24) & 0xFF
            rd.append(mk(SP_ACT, (ab, ab), kb - 1 if spl else kb))
            wr.append(mk(SP_TMEM, w[2], p2 + n))
            if fl & I.F_RMAX:
                wr.append(mk(SP_TMEM, _add(_low(w[2]), p3), m))
    elif op == I.QACT:
        rows, kb = c[1] & 0xFF, c[1] >> 16
        st = fl & I.F_QST                       # (ST: srs w6, its rows to DRAM at w3, w7 apart)
        if rows and kb:
            rd.append(mk(SP_TMEM, w[0], _p(rows - 1, c[5] if st else c[2]) + kb * D))
            if st:
                wr.append(mk(SP_DRAM, w[2], _p(rows - 1, c[6]) + I.qrow_bytes(kb, D)))
            if fl & I.F_CSCALE:
                rd.append(mk(SP_TMEM, w[3], kb * D))
            if fl & I.F_RSCALE:
                rd.append(mk(SP_TMEM, w[4], rows))
            ab = (c[1] >> 8) & 0xFF
            wr.append(mk(SP_ACT, (ab, ab), 2 * kb if fl & I.F_QHL else kb))   # (HL: q, lo)
    elif op == I.LDA:
        rows, kb = c[1] & 0xFF, c[1] >> 16
        if rows and kb:                         # (QACT ST's rows, w3 apart, into ACT RAM)
            rd.append(mk(SP_DRAM, w[0], _p(rows - 1, c[2]) + I.qrow_bytes(kb, D)))
            ab = (c[1] >> 8) & 0xFF
            wr.append(mk(SP_ACT, (ab, ab), kb))
    elif op == I.QST and fl & I.F_QGRP:
        rows, kb = c[3] & 0xFFFF, c[3] >> 16
        if rows and kb:
            # QST GRP (GALL, docs/isa.md QST): rows n = R[rd] + w6 .. of a grouped matrix in
            # groups of R = 2^LG rows, chunks P = R D: from the first row's group (COL: its chunk
            # of columns) over every group (chunk) the rows can reach; a row's scale area (COL:
            # the rows' words) likewise
            R, (nlo, nhi) = 1 << ((fl >> I.QST_LG) & 7), _span(_low(w[5]))
            P, es = R * D, c[6]
            hld = (c[4] >> I.QST_HLD & 0x7FF) << 10 if c[4] & I.QST_HL else 0   # (HL: lo's part)
            srs = c[4] & (1 << I.QST_HLD) - 1 if hld else c[4]
            rd.append(mk(SP_TMEM, w[0], _p(rows - 1, srs) + kb * D))
            if fl & I.F_QCOL:
                lo, hi = nlo // D * P, nhi // D * P
                ln = ((kb * D - 1) // R) * es + ((D + rows - 2) // D + 1) * P
                slo, shi, sln = 0, 0, 4 * rows
            else:
                gs = kb * P
                lo, hi = nlo // R * gs, nhi // R * gs
                ln = ((R + rows - 2) // R + 1) * gs
                slo, shi, sln = nlo // R * es, nhi // R * es, ((R + rows - 2) // R + 1) * es
            for h in (0, hld) if hld else (0,):
                wr.append(_beats(mk(SP_DRAM, _norm(w[1][0] + lo + h, w[1][1] + hi + h, aw), ln)))
                wr.append(_beats(mk(SP_DRAM, _norm(w[2][0] + slo + h, w[2][1] + shi + h, aw),
                                    sln)))
    elif op == I.QST:
        rows, kb = c[3] & 0xFFFF, c[3] >> 16
        if rows and kb:
            p0, p1 = _p(rows - 1, c[4]), _p(rows - 1, c[5])
            # a scale row's words (not ROW): Sw and ceil(KB / 4) words of offsets
            p2, p3 = _p(kb * D - 1, c[6]), _p(rows, 1 + ((kb + 3) >> 2))
            rd.append(mk(SP_TMEM, w[0], p0 + kb * D))
            wr.append(_beats(mk(SP_DRAM, w[1], p1 + p2 + 1)))
            wr.append(_beats(mk(SP_DRAM, w[2], 4 * (rows if fl & I.F_ROW else p3))))
    elif op == I.VOP:
        rows, cols = c[3] & 0xFFFF, c[3] >> 16
        f, bm = (c[5] >> 16) & 0xFF, (c[5] >> 24) & 3
        if rows and cols:
            p0, p1, p2 = _p(rows - 1, c[4] >> 16), _p(rows - 1, c[5] & 0xFFFF), \
                _p(rows - 1, c[4] & 0xFFFF)
            if f == I.V_OUTER:
                if not fl & I.F_DONE:
                    rd.append(mk(SP_TMEM, w[1], 1 if fl & I.F_DSCALAR else cols))
                rd.append(mk(SP_TMEM, w[6], cols))
            elif f != I.V_FILL:
                rd.append(mk(SP_TMEM, w[1], p0 + cols))
            if f == I.V_MAC:                            # dst read in place (and written)
                rd.append(mk(SP_TMEM, w[0], p2 + cols))
            if f in I.BINARY or f in (I.V_RDOT, I.V_OUTER, I.V_MAC):
                if bm == I.B_FULL:
                    rd.append(mk(SP_TMEM, w[2], p1 + cols))
                elif bm == I.B_ROW:
                    rd.append(mk(SP_TMEM, w[2], p1 + 1))
                elif bm == I.B_COL:
                    rd.append(mk(SP_TMEM, w[2], cols))
            if f == I.V_ARGMAX:
                wr.append(mk(SP_TMEM, w[0], p2 + 2))
            elif f in I.REDUCE:
                wr.append(mk(SP_TMEM, w[0], p2 + 1))
            else:
                wr.append(mk(SP_TMEM, w[0], p2 + cols))
            if fl & I.F_FSUM and f == I.V_EXP2SUB:      # its row sums, a word a row
                wr.append(mk(SP_TMEM, w[6], rows))
    elif op == I.GATHER:
        rows, cols = c[2] & 0xFFFF, c[2] >> 16
        if rows and cols:
            rd.append(mk(SP_TMEM, w[0], _p(rows - 1, c[3]) + cols))
            wr.append(mk(SP_TMEM, w[1], _p(S - 1, c[5]) + _p(rows - 1, c[4]) + cols))
            if rows == 1 and c[5] == cols:
                return Fp(False, tuple(x for rs in rd for x in rs),
                          tuple(x for rs in wr for x in rs), True)
    elif op == I.RLD:
        rd.append(mk(SP_TMEM, w[0], 1))
    elif op == I.WAITW:
        wr.append(mk(SP_DRAM, (0, 0), M32) if aw == 32 else [(SP_DRAM, 0, (1 << aw) - 1)])
        wr.append(mk(SP_TMEM, w[1], 1))
    return Fp(False, tuple(x for rs in rd for x in rs), tuple(x for rs in wr for x in rs))


def _ov(a, b):
    return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]


def conflict(n: Fp, e: Fp, dram_only: bool = False) -> bool:
    """otpu_pkg conflict() (RAW, WAR, WAW); dram_only: conflict_dram()."""
    if n.all or e.all:
        return True
    nr, nw, er, ew = n.rd, n.wr, e.rd, e.wr
    if dram_only:
        nr = [x for x in nr if x[0] == SP_DRAM]; nw = [x for x in nw if x[0] == SP_DRAM]
        er = [x for x in er if x[0] == SP_DRAM]; ew = [x for x in ew if x[0] == SP_DRAM]
    for a in nw:
        for b in er:
            if _ov(a, b):
                return True
        for b in ew:
            if _ov(a, b):
                return True
    for a in ew:
        for b in nr:
            if _ov(a, b):
                return True
    return False


def follows(ins: I.Instr, n: Fp, e: Fp | None) -> bool:
    """A VOP (footprint n) may follow the collective instruction before it (e, None: none): a
    reduction of A alone of one row (FOLLOWS) whose only conflict with e is a read of what e, a
    contiguous GATHER, writes. The VPU then reads A's chunks as the GATHER's writes reach them
    (otpu_vpu FOLLOW; every chunk at or below the top of the GATHER's writes into its slice's
    TMEM). Without a conflict with e there is nothing to follow, and FOLLOW is harmless: the VOP
    still waits for e's start, so every older collective instruction has completed."""
    if e is None or not conflict(n, e):
        return True
    return (ins.op == I.VOP and e.cg and not e.all and (ins.w[5] >> 16) & 0xFF in FOLLOWS and
            ins.w[3] & 0xFFFF == 1 and not conflict(Fp(n.all, (), n.wr), e))


def _coalesce(rs):
    rs = tuple(sorted(set(rs)))
    if len(rs) <= MAXR:
        return rs
    out = []
    for sp in (SP_TMEM, SP_DRAM, SP_ACT):
        x = [r for r in rs if r[0] == sp]
        if x:
            out.append((sp, min(r[1] for r in x), max(r[2] for r in x)))
    return tuple(out)


def _fjoin(a: Fp, b: Fp) -> Fp:
    if a == b:
        return a
    return Fp(a.all or b.all, _coalesce(a.rd + b.rd), _coalesce(a.wr + b.wr), a.cg and b.cg)


# ------------------------------------------------------------------------- abstract state
class _State:
    """Registers (intervals) and, per unit, its last DEPTH[u] instructions' footprints
    (newest first: KMAX, the VPU's VMAX). HL, HD: the same histories' local projections' ids and
    DRAM footprints (_MemoPass's; None: not computed yet, e.g. after a join)."""
    __slots__ = ("R", "H", "HL", "HD")

    def __init__(self, R, H, HL=None, HD=None):
        self.R, self.H, self.HL, self.HD = R, H, HL, HD

    def key(self):
        return (tuple(self.R), self.H)


def _join(a: _State | None, b: _State | None, aw: int = 32) -> _State | None:
    if a is None:
        return b
    if b is None:
        return a
    R = [x if x == y else _hull(x, y, aw) for x, y in zip(a.R, b.R)]
    H = []
    for ha, hb in zip(a.H, b.H):
        if ha == hb:
            H.append(ha)
            continue
        n = max(len(ha), len(hb))
        h = []
        for i in range(n):
            if i >= len(ha):
                h.append(hb[i])
            elif i >= len(hb):
                h.append(ha[i])
            else:
                h.append(_fjoin(ha[i], hb[i]))
        H.append(tuple(h))
    return _State(R, tuple(H))


def _writes(prog, lo, hi) -> set:
    """Registers an instruction in [lo, hi) may write."""
    out = set()
    for ins in prog[lo:hi]:
        if ins.op in (I.LI, I.ADDI, I.RLD) and ins.rd:
            out.add(ins.rd)
    return out


class _Pass:
    def __init__(self, prog, D, S, rld, budget, aw: int = 32, win: int = WIN):
        self.prog, self.D, self.S, self.aw = prog, D, S, aw
        self.win, self.depth = win, _depth(win)
        self.rld = rld or {}
        self.kmin = np.full((len(prog), NW), win, np.int64)    # (win: none within the window)
        self.vmask = np.zeros(len(prog), np.int64)
        self.fol = np.ones(len(prog), bool)        # FOLLOW on every visit (follows())
        self.seen = np.zeros(len(prog), bool)
        self.visits = 0
        self.budget = budget
        self.widened = []              # (pc of the LOOP, reason)

    # one instruction; returns the state after it (None: the program halted)
    def visit(self, pc, st: _State) -> _State | None:
        ins = self.prog[pc]
        op = ins.op
        self.visits += 1
        if op == I.HALT:
            return None
        if op in (I.NOP,):
            return st
        if op == I.LI:
            if ins.rd:
                R = list(st.R); R[ins.rd] = (_imm(ins, self.aw),) * 2
                return _State(R, st.H)
            return st
        if op == I.ADDI:
            if ins.rd:
                R = list(st.R)
                R[ins.rd] = _add((0, 0) if ins.ra == 0 else st.R[ins.ra], _imm(ins, self.aw),
                                 self.aw)
                return _State(R, st.H)
            return st
        u = UNIT.get(op)
        if u is None:
            raise ValueError(f"waits: pc {pc}: opcode {op:#x}")
        fp = footprint(ins, st.R, self.D, self.S, self.aw)
        self.seen[pc] = True
        km = self.kmin[pc]
        for j in range(NW):
            uj = WUNIT[j]
            if (j >= NUNITS and op != I.MM) or uj == u or uj == U_VPU:
                continue
            for i, e in enumerate(st.H[uj]):
                if i + 1 >= km[j]:
                    break
                if conflict(fp, e, dram_only=j >= NUNITS):
                    km[j] = i + 1
                    break
        m = int(self.vmask[pc])
        for i, e in enumerate(st.H[U_VPU]):
            if not m >> i & 1 and conflict(fp, e):
                m |= 1 << i
        self.vmask[pc] = m
        if op == I.VOP and self.fol[pc]:
            self.fol[pc] = follows(ins, fp, st.H[U_COLL][0] if st.H[U_COLL] else None)
        H = list(st.H)
        H[u] = (fp,) + H[u][:self.depth[u] - 1]
        R = st.R
        if op == I.RLD and ins.rd:
            # the kernel's bound on the value (isa.rld's bound: flag bit 2, w3..w4), else any
            R = list(R)
            R[ins.rd] = self.rld_value(pc, ins)
        return _State(R, tuple(H))

    def rld_value(self, pc, ins) -> tuple:
        """An RLD's values: rld's, else its bound (isa.rld's: flag bit 2, {w5, w3} .. {w6, w4},
        at most 2^AW - 1, or from lo below 2^AW to hi past it: the interval wraps, a negative
        multiplier's), else any (a 32-bit word; with MUL above AW 32 any AW-bit product)."""
        v = self.rld.get(pc)
        if v is None:
            w, m = ins.w, M32 if self.aw == 32 else (1 << self.aw) - 1
            lo, hi = w[2] | w[4] << 32, w[3] | w[5] << 32
            v = ((_norm(lo, hi, self.aw) if lo <= m < hi else (min(lo, m), min(hi, m)))
                 if ins.flags & I.F_BOUND else _top(self.aw) if ins.flags & I.F_MUL else TOP)
        return v

    def block(self, lo, hi, st: _State | None) -> _State | None:
        pc = lo
        while pc < hi and st is not None:
            ins = self.prog[pc]
            if ins.op == I.LOOP:
                body = ins.w[0]
                st = self.loop(pc, pc + 1, pc + 1 + body, st)
                pc += 1 + body
            else:
                st = self.visit(pc, st)
                pc += 1
        return st

    def loop(self, at, lo, hi, st: _State) -> _State | None:
        ins = self.prog[at]
        self.visits += 1
        cnt = _span(_add(_low((0, 0) if ins.ra == 0 else st.R[ins.ra]), ins.w[1]))
        clo, chi = cnt
        exits = st if clo == 0 else None
        cap = CAP if clo != chi else chi
        if self.visits + cap * (hi - lo) > self.budget:
            cap = min(cap, CAP)
        s = st
        t = 0
        while t < min(chi, cap) and s is not None:
            prev = s
            s = self.block(lo, hi, s)
            t += 1
            if s is not None and t >= clo:
                exits = _join(exits, s, self.aw)
            if s is not None and s.key() == prev.key():
                # a fixed point: every later iteration repeats this one
                if chi > t:
                    exits = _join(exits, s, self.aw)
                return exits
        if s is not None and chi > t:
            self.widened.append((at, "unknown count" if clo != chi else "budget"))
            mod, top = _writes(self.prog, lo, hi), _top(self.aw)
            R = [top if k in mod else v for k, v in enumerate(s.R)]
            s = _State(R, s.H)
            for _ in range(4 * self.win):
                n = self.block(lo, hi, s)
                if n is None:
                    break
                n = _State([top if k in mod else v for k, v in enumerate(n.R)], n.H)
                j = _join(s, n, self.aw)
                if j.key() == s.key():
                    break
                s = j
            else:
                raise RuntimeError(f"waits: the loop at pc {at} did not reach a fixed point")
            exits = _join(exits, s, self.aw)
        return exits


class _MemoPass(_Pass):
    """_Pass with each visit's checks split by address space: conflict(n, e) is a conflict in
    the TMEM and ACT ranges (or `all`) or in the DRAM ranges. A run's TMEM and ACT footprints
    repeat (its tiles, the ACT RAM's blocks) where its DRAM addresses advance (attend at 1,024
    patches: 977 distinct histories of local footprints in 214,534 visits), so the local part
    of a visit, each unit's first conflicting rank and the VPU mask, is kept by (the local
    footprint, the unit, MM or not, the histories' local footprints), and the DRAM ranges are
    checked only for an instruction that has some, against the entries that do. The entries
    are _Pass's (tests/test_waits.py), in a quarter of its time on the tower's programs."""
    MEMO_N = 1 << 18                # kept visits; past it the memo starts again

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.memo = {}
        self.pj = {}                # Fp -> (its local projection's id, its DRAM Fp or None)
        self.lfp = []               # local id -> its local Fp
        self.lids = {}

    def proj(self, fp: Fp):
        v = self.pj.get(fp)
        if v is None:
            lrd = tuple(r for r in fp.rd if r[0] != SP_DRAM)
            lwr = tuple(r for r in fp.wr if r[0] != SP_DRAM)
            k = (fp.all, lrd, lwr, fp.cg)
            lid = self.lids.get(k)
            if lid is None:
                lid = self.lids[k] = len(self.lfp)
                self.lfp.append(Fp(fp.all, lrd, lwr, fp.cg))
            drd = tuple(r for r in fp.rd if r[0] == SP_DRAM)
            dwr = tuple(r for r in fp.wr if r[0] == SP_DRAM)
            v = self.pj[fp] = (lid, Fp(False, drd, dwr) if drd or dwr else None)
        return v

    def _local(self, lid, u, mm, HL):
        """Each count's first conflicting rank over the local footprints (the window: none), and
        the VPU mask's bits."""
        n = self.lfp[lid]
        first = [self.win] * NW
        for j in range(NW):
            uj = WUNIT[j]
            if (j >= NUNITS and not mm) or uj == u or uj == U_VPU:
                continue
            for i, e in enumerate(HL[uj]):
                f = self.lfp[e]
                # (the stream start's counts: DRAM conflicts only, but `all` conflicts)
                if (n.all or f.all) if j >= NUNITS else conflict(n, f):
                    first[j] = i
                    break
        m = 0
        for i, e in enumerate(HL[U_VPU]):
            if conflict(n, self.lfp[e]):
                m |= 1 << i
        return first, m

    def visit(self, pc, st: _State) -> _State | None:
        ins = self.prog[pc]
        op = ins.op
        u = UNIT.get(op)
        if u is None:
            r = super().visit(pc, st)
            if r is not None and r is not st and r.H is st.H:
                r.HL, r.HD = st.HL, st.HD
            return r
        self.visits += 1
        if st.HL is None:
            st.HL = tuple(tuple(self.proj(e)[0] for e in h) for h in st.H)
            st.HD = tuple(tuple(self.proj(e)[1] for e in h) for h in st.H)
        fp = footprint(ins, st.R, self.D, self.S, self.aw)
        lid, dfp = self.proj(fp)
        self.seen[pc] = True
        mm = op == I.MM
        key = (lid, u, mm, st.HL)
        res = self.memo.get(key)
        if res is None:
            if len(self.memo) >= self.MEMO_N:
                self.memo.clear()
            res = self.memo[key] = self._local(lid, u, mm, st.HL)
        first, m = res
        km = self.kmin[pc]
        for j in range(NW):
            uj = WUNIT[j]
            if (j >= NUNITS and not mm) or uj == u or uj == U_VPU:
                continue
            f = first[j]
            if dfp is not None:
                lim = min(f, km[j] - 1)
                for i, e in enumerate(st.HD[uj]):
                    if i >= lim:
                        break
                    if e is not None and conflict(dfp, e):
                        f = i
                        break
            if f + 1 < km[j]:
                km[j] = f + 1
        if dfp is not None:
            for i, e in enumerate(st.HD[U_VPU]):
                if e is not None and not m >> i & 1 and conflict(dfp, e):
                    m |= 1 << i
        self.vmask[pc] |= m
        if op == I.VOP and self.fol[pc]:
            self.fol[pc] = follows(ins, fp, st.H[U_COLL][0] if st.H[U_COLL] else None)
        d = self.depth[u]
        H, HL, HD = list(st.H), list(st.HL), list(st.HD)
        H[u] = (fp,) + H[u][:d - 1]
        HL[u] = (lid,) + HL[u][:d - 1]
        HD[u] = (dfp,) + HD[u][:d - 1]
        R = st.R
        if op == I.RLD and ins.rd:
            R = list(R)
            R[ins.rd] = self.rld_value(pc, ins)
        return _State(R, tuple(H), tuple(HL), tuple(HD))


def arg_ranges(run_args) -> dict:
    """{register: values} of a program's run-time arguments (compiler.Builder.run_args: argument
    k, c * var, in R15 - k): c * [0, var.bound) where the variable has a bound, else any."""
    out = {}
    for k, (v, c) in enumerate(run_args or []):
        bound = getattr(v, "bound", None)
        if bound:
            lo, hi = sorted((0, int(c) * (int(bound) - 1)))
            out[15 - k] = _norm(lo, hi)
    return out


def variant_paths(prog: list) -> list:
    """The runs' paths through the program's variant tables (isa.rld's variant: the RLDs of
    table words b .. b + K - 1, exactly one 1 at a run), as rld maps {pc: (v, v)}: one map a
    combination of the tables' variants, at most PATHS of them; [None] (the RLDs' bounds, one
    pass) without tables or past PATHS. Each path is one straight run of the variant loops of
    0 or 1 iterations, so its histories are not shifted against another path's at the joins: a
    table's loops (attention's sub-block variants: one table a program, 32 loops a layer in
    GPT-OSS's decode) joined as independent counts paired each VOP with the ones of a path no
    run takes, and the waits named VOPs of other heads (GPT-OSS's decode at r60: 4,527 -> 2,883
    mask bits). The table's words stay as written once a program (the builder holds the tile),
    so a base is one table."""
    tabs = {}
    for pc, ins in enumerate(prog):
        if ins.op == I.RLD and ins.w[6] & I.VARIANT:
            k, K = ins.w[6] & 0x7FFF, (ins.w[6] >> 16) & 0x7FFF
            t = tabs.setdefault(ins.w[0] - k, (K, []))
            if t[0] != K or ins.ra:
                return [None]
            t[1].append((pc, k))
    n = 1
    for K, _ in tabs.values():
        n *= K
    if not tabs or n > PATHS:
        return [None]
    paths = [{}]
    for K, rl in tabs.values():
        paths = [m | {pc: (int(k == c),) * 2 for pc, k in rl} for m in paths for c in range(K)]
    return paths


def wait_counts(prog: list, D: int = 128, S: int = 1, args=None, rld: dict | None = None,
                budget: int = BUDGET, stats: dict | None = None,
                run_args=None, memo: bool = True, aw: int = 32, win: int = WIN) -> np.ndarray:
    """The wait entries of a program (uint64 per instruction: the counts' word, the VPU mask's
    word above it). args: the run's arguments if known (R8..R15); else run_args (default: the
    program's own, compiler.Program.run_args) bounds those whose variable has a bound, and the
    others are unknown. rld: {pc: (lo, hi)} the values an RLD may load (default: its own
    bound, isa.rld's, else unknown; None: each of variant_paths' runs, their waits joined).
    stats: filled with the visits and the loops run on widened registers. memo: _MemoPass (the
    same entries as _Pass's, faster). aw: the machine's address bits (Config.AW; the arguments
    are 32-bit words at any). win: the sequencer's window (Config.WC_WIN, 16 or 32: at 32 the
    counts saturate at KMAX)."""
    R = [(0, 0)] * 8 + [TOP] * 8
    if args is not None:
        a = list(args) + [0] * (8 - len(args))
        R = [(0, 0)] * 8 + [(int(x) & M32,) * 2 for x in a[:8]]
    else:
        for r, v in arg_ranges(run_args if run_args is not None
                               else getattr(prog, "run_args", None)).items():
            R[r] = v
    ps = []
    for rv in ([rld] if rld is not None else variant_paths(prog)):
        p = (_MemoPass if memo else _Pass)(prog, D, S, rv, budget, aw, win)
        p.block(0, len(prog), _State(R, ((),) * NUNITS))
        ps.append(p)
    p = ps[0]
    for q in ps[1:]:                    # the paths' union: every path's waits
        p.kmin, p.vmask = np.minimum(p.kmin, q.kmin), p.vmask | q.vmask
        p.fol, p.seen = p.fol & q.fol, p.seen | q.seen
        p.visits, p.widened = p.visits + q.visits, p.widened + q.widened
    k = np.where(p.kmin >= win, 0, np.minimum(p.kmin, KMAX)).astype(np.uint64)
    words = p.vmask.astype(np.uint64) << np.uint64(32)
    for j in range(NW):
        if WUNIT[j] != U_VPU:
            words |= k[:, j] << np.uint64(4 * j)
    # FOLLOW where the GATHER followed is the last collective instruction (k_4 = 1), whose start
    # the VOP then waits for instead of its completion
    fol = p.fol & (k[:, U_COLL] == 1) & np.array([ins.op == I.VOP for ins in prog], bool)
    words |= fol.astype(np.uint64) << np.uint64(28)
    for pc, ins in enumerate(prog):
        if ins.op in UNIT and not p.seen[pc]:
            words[pc] = DEFAULT
    if stats is not None:
        stats.update(visits=p.visits, widened=p.widened, unreached=int(
            sum(1 for pc, ins in enumerate(prog) if ins.op in UNIT and not p.seen[pc])))
    return words


def counts(entry: int) -> list:
    """An entry's seven counts (k_3 is 0: the VPU's is the mask)."""
    return [(int(entry) >> (4 * j)) & 15 for j in range(NW)]


def vmask(entry: int) -> int:
    """An entry's VPU mask: bit r - 1, the r-th last VPU instruction."""
    return (int(entry) >> 32) & VMASK


def follow(entry: int) -> bool:
    """An entry's FOLLOW bit."""
    return bool(int(entry) & FOLLOW)


def sync_file(words, path) -> None:
    """otpu_seq's +hz_sync file: lines 'pc k0 .. k6 n mask follow' (n unused)."""
    with open(path, "w") as f:
        for pc, w in enumerate(words):
            f.write(f"{pc} " + " ".join(str(x) for x in counts(w)) +
                    f" 0 {vmask(w)} {int(follow(w))}\n")


# ------------------------------------------------------------------------- loading (the host)
def area(imem_words: int, waitc: bool = False) -> int:
    """Bytes of DRAM a program of IMEM's instructions takes: its words (4 bytes each), and on a
    WAITC device its entries after them (8 bytes an instruction; IMEM's words fill whole
    chunks, so the entries need no padding before them)."""
    return 4 * imem_words + (imem_words if waitc else 0)


def loadable(words, entries, D: int) -> np.ndarray:
    """A program as a WAITC device loads it from DRAM (uint32 words): its words, zeros to the
    next chunk (D bytes), then its entries, 64 bits each (little-endian: the counts' word, then
    the mask's), one per instruction (docs/isa.md "Wait counts")."""
    w = np.asarray(words, "<u4")
    e = np.asarray(entries, np.uint64)
    if len(w) % 8 or len(e) != len(w) // 8:
        raise ValueError(f"{len(e)} wait entries for {len(w) / 8:g} instructions")
    pad = np.zeros(-len(w) % (D // 4), "<u4")
    return np.concatenate([w, pad, e.astype("<u8").view("<u4")])


MEM_N = 256                       # entries kept in this process (the most recently used)
_mem: OrderedDict = OrderedDict()
stats = {"memory": 0, "disk": 0, "computed": 0, "seconds": 0.0}


def entries_key(words, D: int, S: int, run_args=None, aw: int = 32, win: int = WIN) -> str:
    """The key of a program's entries: its words, the bounds its arguments take (run_args:
    compiler.Program's), D, S, AW above 32, the window above 16, and the source of the compiler
    and of this pass (progcache.source(): every module of opentpu that decides a program word,
    this one too)."""
    from . import progcache as PC
    b = sorted(arg_ranges(run_args).items())
    k = ("waits", D, S, b, PC.source()) + ((aw,) if aw != 32 else ()) + \
        ((("win", win),) if win != WIN else ())
    h = hashlib.blake2b(repr(k).encode(), digest_size=20)
    h.update(np.ascontiguousarray(words, "<u4").tobytes())
    return h.hexdigest()


def entries(words, D: int = 128, S: int = 1, run_args=None, prog=None,
            aw: int = 32, win: int = WIN) -> np.ndarray:
    """wait_counts() of a program given as its assembled words (prog: its instructions, if at
    hand; else the words decoded), its arguments bounded by run_args (default: prog's or the
    words', compiler.Program / compiler.Words), kept: in this process (MEM_N) and, when the
    program cache has a disk (progcache.cache_dir), as an entry of it (evicted with its
    programs). `stats` counts the hits, the computes and their seconds."""
    from . import progcache as PC
    if run_args is None:
        run_args = getattr(prog, "run_args", None) or getattr(words, "run_args", None)
    k = entries_key(words, D, S, run_args, aw, win)
    if k in _mem:
        _mem.move_to_end(k)
        stats["memory"] += 1
        return _mem[k]
    d = PC.cache_dir()
    f = None if d is None else d / k[:2] / f"{k}.npz"
    e = None
    if f is not None:
        try:
            with np.load(f) as z:
                e = z["waits"]
            os.utime(f)
            stats["disk"] += 1
        except FileNotFoundError:
            pass
        except Exception:                   # a damaged entry: compute again
            f.unlink(missing_ok=True)
    if e is None:
        t0 = time.perf_counter()
        if prog is None:
            w = np.asarray(words, np.uint32)
            prog = [I.Instr.decode(w[i:i + 8]) for i in range(0, len(w), 8)]
        e = wait_counts(prog, D, S, run_args=run_args if run_args is not None else (), aw=aw,
                        win=win)
        stats["computed"] += 1
        stats["seconds"] += time.perf_counter() - t0
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(f.parent).free - e.nbytes >= PC.free_floor():
                tmp = f.with_name(f".{f.stem}.{os.getpid()}.npz")
                np.savez(tmp, waits=e)
                os.replace(tmp, f)
                PC._evict(d)
    _mem[k] = e
    while len(_mem) > MEM_N:
        _mem.popitem(last=False)
    return e


def _compute(words, D: int, S: int, run_args, aw: int = 32, win: int = WIN) -> np.ndarray:
    """entries() in a worker process (entries_many): kept on the disk there, returned here."""
    return entries(words, D, S, run_args, aw=aw, win=win)


def entries_many(items, D: int, S: int, pool=None, aw: int = 32, win: int = WIN) -> list:
    """entries() of several programs [(words, run_args or None, prog or None)], those not kept
    yet computed in `pool` (a concurrent.futures executor: the engine's compile worker
    processes) at once, then kept here; without a pool one after another."""
    out = [None] * len(items)
    todo = []
    for i, (w, ra, p) in enumerate(items):
        if ra is None:
            ra = getattr(p, "run_args", None) or getattr(w, "run_args", None)
        k = entries_key(w, D, S, ra, aw, win)
        if k in _mem:
            _mem.move_to_end(k)
            stats["memory"] += 1
            out[i] = _mem[k]
        else:
            todo.append((i, w, ra, p, k))
    if pool is not None and len(todo) > 1:
        futs = [(i, k, pool.submit(_compute, np.asarray(w, np.uint32), D, S, ra, aw, win))
                for i, w, ra, p, k in todo]
        for i, k, f in futs:
            out[i] = _mem[k] = f.result()
            stats["computed"] += 1
        while len(_mem) > MEM_N:
            _mem.popitem(last=False)
    else:
        for i, w, ra, p, k in todo:
            out[i] = entries(w, D, S, ra, prog=p, aw=aw, win=win)
    return out


# ------------------------------------------------------------------------- checking
class Checker:
    """Checks wait entries on a concrete execution: feed it each unit instruction as it issues
    in program order (pc, the registers then). A conflict with an older instruction of another
    unit (or a VPU instruction's with an older one) within the window of its unit (win - 1; VMAX
    of the VPU) that the entry does not wait for raises (a saturated count, through a younger one
    of the unit, waits for it)."""

    def __init__(self, prog, words, D: int = 128, S: int = 1, aw: int = 32, win: int = WIN):
        self.prog, self.words, self.D, self.S, self.aw = prog, words, D, S, aw
        self.depth = _depth(win)
        self.H = [[] for _ in range(NUNITS)]
        self.n = 0

    def issue(self, pc, regs) -> None:
        ins = self.prog[pc]
        u = UNIT.get(ins.op)
        if u is None:
            return
        am = _top(self.aw)[1]
        R = [(int(x) & am,) * 2 for x in regs]
        fp = footprint(ins, R, self.D, self.S, self.aw)
        k = counts(self.words[pc])
        m = vmask(self.words[pc])
        if follow(self.words[pc]) and (
                ins.op != I.VOP or k[U_COLL] != 1 or
                not follows(ins, fp, self.H[U_COLL][0][0] if self.H[U_COLL] else None)):
            raise AssertionError(f"waits: pc {pc} ({I.OPNAMES.get(ins.op)}) FOLLOW, but it "
                                 "may not follow the last collective instruction")
        for i, (e, epc) in enumerate(self.H[U_VPU]):
            if not m >> i & 1 and conflict(fp, e):
                raise AssertionError(
                    f"waits: pc {pc} ({I.OPNAMES.get(ins.op)}) VPU mask {m:#06x} misses the "
                    f"{i + 1}-th last VPU instruction (pc {epc})")
        for j in range(NW):
            uj = WUNIT[j]
            if (j >= NUNITS and ins.op != I.MM) or uj == u or uj == U_VPU:
                continue
            for i, (e, epc) in enumerate(self.H[uj]):
                if conflict(fp, e, dram_only=j >= NUNITS):
                    if k[j] == 0 or k[j] > i + 1:
                        raise AssertionError(
                            f"waits: pc {pc} ({I.OPNAMES.get(ins.op)}) wait {j} k={k[j]} misses "
                            f"the {i + 1}-th last {('DMA', 'MXU', 'QUANT', 'VPU', 'COLL')[uj]} "
                            f"instruction (pc {epc})")
                    break
        self.H[u].insert(0, (fp, pc))
        del self.H[u][self.depth[u]:]
        self.n += 1


def check_isasim(machine, words_per_slice, max_steps: int = 50_000_000) -> int:
    """Run an isasim.Machine to its halt, checking every slice's wait words on the way (no
    chains, no collectives across slices). Returns the instructions checked."""
    sl = machine.slices
    chk = [Checker(s.prog, w, machine.cfg.D, machine.cfg.S, machine.cfg.AW, machine.cfg.WC_WIN)
           for s, w in zip(sl, words_per_slice)]
    n = 0
    steps = 0
    while not all(s.halted for s in sl):
        progressed = False
        for s, c in zip(sl, chk):
            while not s.halted and s.waiting is None and s.polling is None:
                pc = s.pc
                if 0 <= pc < len(s.prog) and s.prog[pc].op in UNIT:
                    c.issue(pc, [s.reg(r) for r in range(16)])
                s.step()
                steps += 1
                progressed = True
                if steps > max_steps:
                    raise RuntimeError("check_isasim: step limit")
                if s.halted and s.chain is not None:
                    raise RuntimeError("check_isasim: HALT CHAIN")
        live = [s for s in sl if not s.halted]
        if live and all(s.waiting is not None for s in live):
            if any(s.waiting.op == I.GATHER for s in live):
                machine._gather()
            for s in live:
                s.waiting = None
                s.advance()
            progressed = True
        if not progressed:
            for s in sl:
                if s.polling is not None and s.poll(s.polling):
                    s.polling = None
                    s.advance()
                    progressed = True
        if not progressed:
            raise RuntimeError("check_isasim: deadlock")
    return sum(c.n for c in chk)
