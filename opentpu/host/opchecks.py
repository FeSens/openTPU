"""One small program per instruction variant, for otpu-diag's ISA coverage (and
tests/test_board.py on the board model).

op_checks(cfg) -> [(group, name, program)]; every program starts from diag_image() in DRAM,
stores what it computed below checks.PROG_AT, and is compared with the ISA simulator bit for bit
by checks.run_demo. The programs are self-contained on the card too, where TMEM and the ACT RAM
keep the previous run's contents: they only store TMEM words they wrote, and every MM reads ACT
RAM blocks its own QACT filled.

Groups (otpu-diag's diagnosis keys on them): control, dma, mxu, quant, collective, vpu,
vpu-reduce, vpu-composite, vpu-edge, vpu-new (RDOT / OUTER / LOG2, which bitstreams before
ddec900 lack).

The edge values leave out NaN inputs: the RTL and the ISA simulator disagree there (board
model, 3c270c9): RECIP of a NaN gives 0 in the RTL, NaN in the simulator; MAX / ABS / COPY pass
a signalling NaN (0x7F800001) through unchanged where the simulator returns 0x7FC00000. The
models never feed NaN, so this is a documentation gap, not a bring-up blocker.
"""
from __future__ import annotations

import numpy as np

from opentpu import isa as I

from .checks import DATA, OUT, SC, W8, demo_image

EDGE = 0x80000                  # DRAM: edge-value fp32 vectors (EDGE_BITS, then reversed)
EDGE_BITS = [                   # fp32 bit patterns
    0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x3F000000, 0x40000000, 0x40400000,
    0xBF000000, 0x00800000, 0x80800000, 0x006CE3EE, 0x00000001, 0x7149F2CA, 0xF149F2CA,
    0x7F7FFFFF, 0xFF7FFFFF, 0x42FC0000, 0x42FE0000, 0x43000000, 0x43480000, 0xC2FC0000,
    0xC3150000, 0xC3160000, 0xC3480000, 0x7F800000, 0xFF800000, 0x3EAAAAAB, 0xC0490FDB,
    0x3F7FFFEF, 0x3F800008, 0x1E3CE508, 0x477FE000]
NE = len(EDGE_BITS)             # 32: 0, -0, +-1, 0.5, 2, 3, -0.5, +-min normal, denormals,
#                                 +-1e30, +-max, 126, 127, 128, 200, -126, -149, -150, -200,
#                                 +-inf, 1/3, -pi, 1 -+ ulps, 1e-20, 65504

# TMEM layout (words): the data at 0.., results from RES on
A, B, RES, TE, TER = 0, 1024, 8192, 12288, 12352


def diag_image() -> np.ndarray:
    img = demo_image()
    e = np.array(EDGE_BITS, np.uint32)
    img[EDGE:EDGE + 4 * NE] = e.view(np.uint8)
    img[EDGE + 4 * NE:EDGE + 8 * NE] = e[::-1].copy().view(np.uint8)
    return img


def _p(*body) -> list:
    return [I.ld(DATA, 0, 4096), *body, I.halt()]


def _vop(func, bmode=I.B_FULL, rows=3, cols=100) -> list:
    return _p(I.vop(func, RES, A, B, rows, cols, cols, 256, 256, bmode, imm=0.75),
              I.st(OUT, RES, rows * cols))


def _edge(func, **kw) -> list:
    return [I.ld(EDGE, TE, 2 * NE),
            I.vop(func, RES, TE, TER, 1, NE, NE, NE, NE, **kw),
            I.st(OUT, RES, NE), I.halt()]


def _mm(cfg, m=None, qflags=None, pre=(), ab=0, ors=16, **kw) -> list:
    """QACT m rows (2 blocks) into ACT RAM block ab, then MM over the 16 x 256 int8 weights.
    The output area is filled from DRAM first (the accumulator of ACC; the gaps of ors > 16)."""
    m = m or cfg.MCOLS
    n = ors * m + (m if kw.get("rmax") else 0)
    return _p(I.qact(0, m, ab, 2, 256, **(qflags or {})), I.ld(DATA + 0x2000, RES, n), *pre,
              I.mm(W8, SC, RES, 16, 2, 256, ors, m, ab, 8, **kw), I.st(OUT, RES, n))


def op_checks(cfg) -> list[tuple[str, str, list]]:
    M = cfg.MCOLS
    out = []

    def add(group, name, prog):
        out.append((group, name, prog))

    # ---- control flow
    add("control", "HALT", [I.halt()])
    add("control", "NOP", _p(I.nop(), I.nop(), I.st(OUT, A, 64)))
    add("control", "LI / ADDI (register offsets)",
        _p(I.li(1, 0x400), I.addi(2, 1, 0x100), I.addi(3, 2, -0x40), I.st(OUT, A, 16, ra=1),
           I.st(OUT, 64, 16, ra=2), I.st(OUT, 128, 16, ra=3)))
    add("control", "LOOP", _p(I.loop(2, 5), I.addi(1, 1, 256), I.st(OUT, A, 32, ra=1)))
    add("control", "LOOP nested",
        _p(I.loop(4, 3), I.loop(2, 4), I.addi(1, 1, 128), I.st(OUT, A, 16, ra=1),
           I.addi(2, 2, 64), I.st(OUT + 0x4000, 32, 8, ra=2)))
    add("control", "LOOP count from a register (rcount)",
        _p(I.li(4, 3), I.loop(2, 2, rcount=4), I.addi(1, 1, 256), I.st(OUT, A, 32, ra=1)))
    add("control", "LOOP count 0 (skips its body)",
        _p(I.loop(1, 0), I.st(OUT, A, 32), I.st(OUT + 256, 64, 32)))
    add("control", "BAR", _p(I.bar(), I.st(OUT, A, 64)))

    # ---- DMA
    add("dma", "LD / ST aligned", _p(I.st(OUT, A, 1024)))
    add("dma", "LD unaligned source",
        [I.ld(DATA + 4, A, 777), I.ld(DATA + 0x1004, 1024, 3), I.st(OUT, A, 777),
         I.st(OUT + 0x1000, 1024, 3), I.halt()])
    add("dma", "ST unaligned target",
        _p(I.st(OUT + 4, A, 777), I.st(OUT + 0x1008, 64, 1), I.st(OUT + 0x103C, 100, 17)))
    add("dma", "LD / ST short (1..17 words)",
        _p(*[I.st(OUT + 0x100 * k + 4 * k, 64 * k, 1 + k) for k in range(17)]))
    add("dma", "LD / ST register offsets (ra dram, rb tmem)",
        [I.li(1, 0x200), I.li(2, 300), I.ld(DATA, 0, 256, ra=1, rb=2),
         I.st(OUT, 0, 256, ra=1, rb=2), I.halt()])

    # ---- MXU (and QACT feeding it)
    add("mxu", "MM", _mm(cfg))
    add("mxu", "MM UNIT", _mm(cfg, unit=True))
    add("mxu", "MM ACC", _mm(cfg, acc=True))
    add("mxu", "MM RMAX", _mm(cfg, rmax=True))
    add("mxu", "MM ACC RMAX", _mm(cfg, acc=True, rmax=True))
    add("mxu", "MM UNIT ACC ASCALE",
        _mm(cfg, pre=[I.ld(SC, 2048, M)], unit=True, acc=True, ascale=2048))
    add("mxu", "MM M=1", _mm(cfg, m=1))
    add("mxu", "MM ACT block ab=3", _mm(cfg, ab=3))
    add("mxu", "MM output row stride ors=40", _mm(cfg, ors=40))
    add("mxu", "MM registers (ra sa, rb ssa, rc out)",
        _p(I.qact(0, M, 0, 2, 256), I.li(1, 256), I.li(2, 8), I.li(3, 64),
           I.ld(DATA + 0x2000, RES, 64 + 16 * M),
           I.mm(W8, SC, RES, 15, 2, 256, 16, M, 0, 8, ra=1, rb=2, rc=3),
           I.st(OUT, RES, 64 + 16 * M)))

    # ---- quantizer
    add("quant", "QACT ROW", _mm(cfg, qflags={"row": True}))
    add("quant", "QACT CSCALE",
        _p(I.vop(I.V_ABS, 2048, 3072, 0, 1, 256, 256, 256, 0),
           I.qact(0, M, 0, 2, 256, cscale=2048),
           I.mm(W8, SC, RES, 16, 2, 256, 16, M, 0, 8), I.st(OUT, RES, 16 * M)))
    add("quant", "QACT RSCALE",
        _p(I.qact(0, M, 0, 2, 256, rscale=3072), I.mm(W8, SC, RES, 16, 2, 256, 16, M, 0, 8),
           I.st(OUT, RES, 16 * M)))
    add("quant", "QST dense", _p(I.qst(A, OUT + 0x800, OUT + 0x1800, 2, 2, 256, 256, 1)))
    add("quant", "QST strided (es=3)",
        _p(I.qst(512, OUT + 0x2001, OUT + 0x5000, 3, 1, 128, 1024, 3)))
    add("quant", "QST ROW", _p(I.qst(A, OUT + 0x800, OUT + 0x1800, 3, 2, 256, 256, 1,
                                     row=True)))

    # ---- collective (one slice: a strided TMEM copy)
    add("collective", "GATHER", _p(I.gather(A, RES, 4, 64, 256, 64, 0), I.st(OUT, RES, 256)))

    # ---- VPU: every function under its legal broadcast modes
    for f, name in I.VFUNCS.items():
        if f == I.V_OUTER:
            continue
        if f in I.READS_B:
            modes = [("FULL", I.B_FULL), ("ROW", I.B_ROW), ("COL", I.B_COL),
                     ("SCALAR", I.B_SCALAR)]
        else:
            modes = [("", I.B_FULL)]
        group = ("vpu-new" if f in (I.V_RDOT, I.V_LOG2) else "vpu-reduce" if f in I.REDUCE
                 else "vpu-composite" if f in (I.V_EXP2, I.V_RECIP, I.V_RSQRT, I.V_EXP2SUB)
                 else "vpu")
        for mname, bm in modes:
            nm = f"VOP {name.upper()}" + (f" {mname}" if mname else "")
            if f in I.REDUCE:
                add(group, nm, _p(I.vop(f, RES, A, B, 3, 200, 1, 256, 256, bm, imm=0.75),
                                  I.st(OUT, RES, 3)))
            else:
                add(group, nm, _vop(f, bm))
    for dname in ("scalar", "column", "one"):
        add("vpu-new", f"VOP OUTER {dname}",
            _p(I.vop(I.V_COPY, RES, A, 0, 4, 64, 64, 64, 0),
               I.outer(RES, 2048, 1024, 3072, 4, 64, 64, 1, dname),
               I.st(OUT, RES, 256)))

    # ---- VPU on edge values (zeros, denormals, huge, inf, NaN)
    for f in (I.V_EXP2, I.V_RECIP, I.V_RSQRT, I.V_EXP2SUB, I.V_LOG2, I.V_ADD, I.V_SUB, I.V_MUL,
              I.V_MAX, I.V_MIN, I.V_ABS, I.V_COPY):
        add("vpu-new" if f == I.V_LOG2 else "vpu-edge", f"VOP {I.VFUNCS[f].upper()} edge values",
            _edge(f))
    for f in (I.V_RSUM, I.V_RMAX, I.V_RSSQ):
        add("vpu-edge", f"VOP {I.VFUNCS[f].upper()} edge values",
            [I.ld(EDGE, TE, 2 * NE), I.vop(f, RES, TE, 0, 2, NE // 2, 1, NE // 2, 0),
             I.st(OUT, RES, 2), I.halt()])
    return out
