"""Column reuse (docs/isa.md, "Column reuse"): a 4-bit MM of M <= MCOLS/2 rows with PAIR takes
two K-blocks per cycle, the even ones from ACT rows 0..M-1 and the odd ones from ACT rows
M..2M-1, which QACT DUP fills with the same rows. The ISA simulator against a scalar
float32 model of the definition, and the compiler's use of it."""
import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I, quant as Q
from opentpu.isasim import Machine

WA, SA, OUT, AS = 16384, 32768, 2048, 1536


def _srs(KB):
    """Scale row stride: PAIR reads a chunk's two scale words as one 8-byte pair."""
    return 8 * -(-KB // 2)


def _setup(rng, fmt, M, KB, N, D, rows_x):
    x = rng.standard_normal((rows_x, KB * D)).astype(np.float32)
    W = (rng.standard_normal((N, KB * D)) * 0.1).astype(np.float32)
    wrows, words = Q.quantize_mxu(W, fmt, D)
    dram = np.zeros(1 << 16, np.uint8)
    dram[:x.nbytes] = x.view(np.uint8).reshape(-1)
    dram[WA:WA + wrows.size] = wrows.reshape(-1)
    sw = np.zeros((N, _srs(KB) // 4), np.uint32)
    sw[:, :KB] = words
    dram[SA:SA + sw.nbytes] = sw.view(np.uint8).reshape(-1)
    return x, wrows.shape[1], dram


def _run(cfg, dram, prog):
    return Machine(cfg, [prog + [I.halt()]], [dram.copy()]).run().slices[0]


def _terms(cfg, dram, x, fmt, R, KB, N, rs, unit):
    """t[r, n, k]: the fp32 term of ACT row r and K-block k, from single-block MMs (KB = 1:
    the partial sums add nothing to one term)."""
    D = cfg.D
    t = np.zeros((R, N, KB), np.float32)
    for k in range(KB):
        prog = [I.ld(0, 0, x.size), I.qact(k * D, R, 0, 1, KB * D)]
        # block k's weights: the chunk k // 2, its half k % 2 (4-bit: two blocks a chunk)
        d = dram.copy()
        wv = d[WA:WA + N * rs].reshape(N, rs)
        half = wv[:, (k // 2) * D + (k % 2) * D // 2:][:, :D // 2].copy()
        wv[:, :D // 2] = half
        sw = d[SA:SA + N * _srs(KB)].view(np.uint32).reshape(N, -1)
        sw[:, 0] = sw[:, k].copy()
        prog.append(I.mm(WA, SA, OUT, N, 1, rs, N, R, 0, _srs(KB), unit=unit,
                         wf=Q.mxu_wf(fmt)))
        s = _run(cfg, d, prog)
        t[:, :, k] = s.tget(OUT + np.arange(R * N)).reshape(R, N)
    return t


def _isum4(p):
    """The MXU's sum of one output: four interleaved partials from +0, left to right, then
    (p0 + p2) + (p1 + p3); scalar float32 with flush to zero."""
    f = lambda v: np.float32(F.ftz(np.float32(v)))
    part = [np.float32(0)] * 4
    for i, v in enumerate(p):
        part[i % 4] = f(part[i % 4] + v)
    return f(f(part[0] + part[2]) + f(part[1] + part[3]))


def _pair_model(t, M):
    """y[j, n] of MM PAIR from the terms of ACT rows 0..2M-1."""
    R, N, KB = t.shape
    f = lambda v: np.float32(F.ftz(np.float32(v)))
    y = np.zeros((M, N), np.float32)
    for j in range(M):
        for n in range(N):
            p = [f(t[j, n, 2 * i] + (t[j + M, n, 2 * i + 1] if 2 * i + 1 < KB else 0))
                 for i in range(-(-KB // 2))]
            y[j, n] = _isum4(p)
    return y


@pytest.mark.parametrize("fmt", ["int4", "fp4"])
@pytest.mark.parametrize("MCOLS,M", [(2, 1), (4, 1), (4, 2)])
@pytest.mark.parametrize("KB", [1, 2, 3, 5, 8, 11])
def test_pair_follows_definition(fmt, MCOLS, M, KB):
    """Distinct data in ACT rows j and j+M: the odd blocks come from row j+M."""
    cfg = Config(S=1, D=32, MCOLS=MCOLS, PAIR=True)
    rng = np.random.default_rng(100 * KB + 10 * MCOLS + M)
    N = 7
    x, rs, dram = _setup(rng, fmt, M, KB, N, cfg.D, 2 * M)
    for unit in (False, True):
        t = _terms(cfg, dram, x, fmt, 2 * M, KB, N, rs, unit)
        s = _run(cfg, dram, [I.ld(0, 0, x.size), I.qact(0, 2 * M, 0, KB, KB * cfg.D),
                             I.mm(WA, SA, OUT, N, KB, rs, N, M, 0, _srs(KB), unit=unit,
                                  wf=Q.mxu_wf(fmt), pair=True)])
        got = s.tget(OUT + np.arange(M * N)).reshape(M, N)
        assert np.array_equal(got.view(np.uint32), _pair_model(t, M).view(np.uint32))


@pytest.mark.parametrize("fmt", ["int4", "fp4"])
def test_pair_epilogues(fmt):
    """ACC, RMAX and ASCALE act on the pair result as on any MM result."""
    cfg = Config(S=1, D=32, MCOLS=2, PAIR=True)
    rng = np.random.default_rng(5)
    KB, N, M = 5, 9, 1
    x, rs, dram = _setup(rng, fmt, M, KB, N, cfg.D, 1)
    base = [I.ld(0, 0, x.size), I.qact(0, 1, 0, KB, KB * cfg.D, dup=True),
            I.vop(I.V_FILL, OUT, 0, 0, 1, N + 1, N + 1, 0, 0, I.B_SCALAR, 0.75),
            I.vop(I.V_FILL, AS, 0, 0, 1, 1, 1, 0, 0, I.B_SCALAR, -1.5)]
    mm = lambda **k: I.mm(WA, SA, OUT, N, KB, rs, N, M, 0, _srs(KB), wf=Q.mxu_wf(fmt),
                          pair=True, **k)
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


@pytest.mark.parametrize("ssa,srs", [(SA + 4, 8), (SA, 12)])
def test_pair_scales_are_8_byte_aligned(ssa, srs):
    dram = np.zeros(1 << 16, np.uint8)
    with pytest.raises(Exception, match="multiples of 8"):
        _run(Config(S=1, D=32, MCOLS=2), dram,
             [I.mm(WA, ssa, OUT, 1, 3, 64, 1, 1, 0, srs, wf=I.W4F, pair=True)])
    _run(Config(S=1, D=32, MCOLS=2), dram,             # UNIT reads no scales
         [I.mm(WA, ssa, OUT, 1, 3, 64, 1, 1, 0, srs, wf=I.W4F, pair=True, unit=True)])


def test_pair_needs_4bit_and_room():
    with pytest.raises(AssertionError):
        I.mm(0, 0, 0, 1, 1, 32, 1, 1, 0, 4, pair=True)
    dram = np.zeros(1 << 16, np.uint8)
    with pytest.raises(Exception, match="bounds"):
        _run(Config(S=1, D=32, MCOLS=2), dram,
             [I.mm(WA, SA, OUT, 1, 2, 32, 1, 2, 0, 8, wf=I.W4F, pair=True)])
