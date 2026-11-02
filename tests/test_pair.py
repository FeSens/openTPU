"""Column reuse (docs/isa.md, "Column reuse"): a 4-bit MM of M <= MCOLS/2 rows with PAIR takes
two K-blocks per cycle, the even ones from ACT rows 0..M-1 and the odd ones from ACT rows
M..2M-1, which QACT DUP fills with the same rows. The sums are one exact integer per output
either way (tests/test_mx.py: PAIR changes no bit); here the ISA simulator's row choice, the
epilogues on a PAIR result, QACT DUP and the field checks."""
import dataclasses

import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I, quant as Q
from opentpu.isasim import Machine

WA, SA, XA, OUT, AS = 16384, 32768, 1 << 17, 8192, 1536


def _setup(rng, fmt, KB, N, D, x):
    """x's rows at XA, W [N, KB D] in `fmt` at WA (rows rs apart) and its scale rows at SA (srs
    apart): (rs, srs, dram)."""
    W = (rng.standard_normal((N, KB * D)) * 0.1).astype(np.float32)
    wrows, srows = Q.quantize_mxu(W, fmt, D)
    assert WA + wrows.size <= SA and SA + srows.size <= XA
    dram = np.zeros(1 << 18, np.uint8)
    dram[XA:XA + x.nbytes] = x.view(np.uint8).reshape(-1)
    dram[WA:WA + wrows.size] = wrows.reshape(-1)
    dram[SA:SA + srows.size] = srows.reshape(-1)
    return wrows.shape[1], srows.shape[1], dram


def _run(cfg, dram, prog):
    cfg = dataclasses.replace(cfg, DRAM_BYTES=max(cfg.DRAM_BYTES, dram.size))
    return Machine(cfg, [prog + [I.halt()]], [dram.copy()]).run().slices[0]


@pytest.mark.parametrize("fmt", ["int4", "fp4", "nvfp4"])
@pytest.mark.parametrize("MCOLS,M", [(2, 1), (4, 1), (4, 2)])
@pytest.mark.parametrize("KB", [1, 2, 3, 5, 8])
def test_pair_takes_the_odd_blocks_from_row_j_plus_m(fmt, MCOLS, M, KB):
    """ACT rows j + M = -(row j): the same Sa and offsets, other data. PAIR's output j is then
    the MM without PAIR of row j with its odd blocks negated, bit for bit (every quantized value
    and scale of -x is that of x negated)."""
    D = 128 if fmt == "nvfp4" else 32
    cfg = Config(S=1, D=D, MCOLS=MCOLS, PAIR=True)
    rng = np.random.default_rng(100 * KB + 10 * MCOLS + M + 7 * len(fmt))
    N = 7
    x = rng.standard_normal((M, KB * D)).astype(np.float32)
    x[:, ::D // 4] *= 2.0 ** -9                    # some sub-blocks with large offsets
    neg = np.where((np.arange(KB * D) // D) % 2 == 1, -1.0, 1.0).astype(np.float32)
    rows = np.concatenate([x, -x, x * neg])        # [x; -x] for PAIR, x with odd blocks negated
    rs, srs, dram = _setup(rng, fmt, KB, N, D, rows)
    wf = Q.mxu_wf(fmt)
    for unit in (False, True):
        s = _run(cfg, dram, [I.ld(XA, 0, rows.size), I.qact(0, 2 * M, 0, KB, KB * D),
                             I.mm(WA, SA, OUT, N, KB, rs, N, M, 0, srs, unit=unit, wf=wf,
                                  pair=True)])
        got = s.tget(OUT + np.arange(M * N)).reshape(M, N)
        s = _run(cfg, dram, [I.ld(XA, 0, rows.size), I.qact(2 * M * KB * D, M, 0, KB, KB * D),
                             I.mm(WA, SA, OUT, N, KB, rs, N, M, 0, srs, unit=unit, wf=wf)])
        want = s.tget(OUT + np.arange(M * N)).reshape(M, N)
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32))


@pytest.mark.parametrize("fmt", ["int4", "fp4", "nvfp4"])
def test_pair_epilogues(fmt):
    """ACC, RMAX and ASCALE act on the pair result as on any MM result."""
    D = 128 if fmt == "nvfp4" else 32
    cfg = Config(S=1, D=D, MCOLS=2, PAIR=True)
    rng = np.random.default_rng(5)
    KB, N, M = 5, 9, 1
    x = rng.standard_normal((1, KB * D)).astype(np.float32)
    rs, srs, dram = _setup(rng, fmt, KB, N, D, x)
    base = [I.ld(XA, 0, x.size), I.qact(0, 1, 0, KB, KB * D, dup=True),
            I.vop(I.V_FILL, OUT, 0, 0, 1, N + 1, N + 1, 0, 0, I.B_SCALAR, 0.75),
            I.vop(I.V_FILL, AS, 0, 0, 1, 1, 1, 0, 0, I.B_SCALAR, -1.5)]
    mm = lambda **k: I.mm(WA, SA, OUT, N, KB, rs, N, M, 0, srs, wf=Q.mxu_wf(fmt), pair=True,
                          **k)
    y = _run(cfg, dram, base + [mm()]).tget(OUT + np.arange(N))
    yu = _run(cfg, dram, base + [mm(unit=True)]).tget(OUT + np.arange(N))
    s = _run(cfg, dram, base + [mm(acc=True, rmax=True)])
    want = F.add(np.float32(0.75), y)
    assert np.array_equal(s.tget(OUT + np.arange(N)), want)
    assert s.tget(OUT + N) == F.chain_max(want[None])[0]
    s = _run(cfg, dram, base + [mm(unit=True, acc=True, ascale=AS)])
    assert np.array_equal(s.tget(OUT + np.arange(N)),
                          F.add(F.mul(np.float32(0.75), np.float32(-1.5)), yu))


@pytest.mark.parametrize("row", [False, True])
def test_qact_dup_writes_both_rows(row):
    cfg = Config(S=1, D=32, MCOLS=4, PAIR=True)
    x = np.random.default_rng(1).standard_normal((2, 96)).astype(np.float32)
    dram = np.zeros(1 << 16, np.uint8)
    dram[:x.nbytes] = x.view(np.uint8).reshape(-1)
    rsc = 512
    prog = [I.ld(0, 0, x.size), I.vop(I.V_FILL, rsc, 0, 0, 1, 2, 2, 0, 0, I.B_SCALAR, 3.0),
            I.qact(0, 2, 5, 3, 96, row=row, rscale=rsc, dup=True)]
    s = _run(cfg, dram, prog)
    one = _run(cfg, dram, prog[:2] + [I.qact(0, 2, 5, 3, 96, row=row, rscale=rsc)])
    assert np.array_equal(s.act[:2], one.act[:2]) and np.array_equal(s.act[2:], one.act[:2])
    assert np.array_equal(s.ascale[2:], one.ascale[:2])
    with pytest.raises(Exception, match="bounds"):
        _run(Config(S=1, D=32, MCOLS=2), dram, prog[:2] + [I.qact(0, 2, 5, 3, 96, dup=True)])


@pytest.mark.parametrize("wf,ssa,srs,align", [(I.W4F, SA + 4, 16, 8), (I.W4I, SA, 12, 8),
                                              (I.WNV, SA + 8, 32, 16), (I.WNV, SA, 24, 16)])
def test_pair_scales_are_aligned(wf, ssa, srs, align):
    """A chunk's two blocks' scale words are read together: ssa and srs multiples of 8 (NVFP4:
    16)."""
    D = 128 if wf == I.WNV else 32
    dram = np.zeros(1 << 16, np.uint8)
    with pytest.raises(Exception, match=f"multiples of {align}"):
        _run(Config(S=1, D=D, MCOLS=2), dram,
             [I.mm(WA, ssa, OUT, 1, 3, 2 * D, 1, 1, 0, srs, wf=wf, pair=True)])
    _run(Config(S=1, D=D, MCOLS=2), dram,             # UNIT reads no scales
         [I.mm(WA, ssa, OUT, 1, 3, 2 * D, 1, 1, 0, srs, wf=wf, pair=True, unit=True)])


def test_pair_needs_4bit_and_room():
    with pytest.raises(ValueError, match="even k0"):
        I.mm(0, 0, 0, 1, 2, 32, 1, 1, 0, 16, wf=I.W4F, pair=True, k0=1)
    dram = np.zeros(1 << 16, np.uint8)
    # int8 PAIR encodes (a DUAL configuration's); without DUAL the machine refuses it
    with pytest.raises(Exception, match="PAIR needs 4-bit"):
        _run(Config(S=1, D=32, MCOLS=4), dram, [I.mm(WA, SA, OUT, 1, 2, 32, 1, 1, 0, 4, pair=True)])
    with pytest.raises(Exception, match="PAIR needs 2\\*M <= MCOLS|bounds"):
        _run(Config(S=1, D=32, MCOLS=2), dram,
             [I.mm(WA, SA, OUT, 1, 2, 32, 1, 2, 0, 16, wf=I.W4F, pair=True)])
