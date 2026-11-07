"""QUAD (Config.QUAD on a DUAL configuration; docs/isa.md "Column reuse"): a 4-bit MM PAIR of
M == MCOLS/4 rows (not UNIT, not RSC) takes four blocks an advance (two chunks, the DUAL port's
pair), block 4a + g from ACT row j + g*M, and QACT DUP of rows <= MCOLS/4 fills four copies. The
ISA simulator: the same integer as the MM without PAIR, the copies, and the fallback to two blocks
(4*M != MCOLS); int8 PAIR stays DUAL's. The RTL (otpu_mxu QUAD) against it: the same TMEM bit for
bit, over k0 that put an advance's block data across two scale lines, odd KB and every format."""
import numpy as np
import pytest

from opentpu import Config, isa as I
from opentpu.isasim import Machine, SimError

from opentpu import rtlsim

from test_rtl import DATA, INT8, MX_BASE, MX_PITCH, _images, _mx_scales


def _cfg(quad=True, mcols=4):
    return Config(S=1, D=32, MCOLS=mcols, PAIR=True, DUAL=True, QUAD=quad)


def _prog(M, KB, k0, pair, wf, N=20, off=0, rows=None):
    D = 32
    bpb = D if wf == I.W8 else D // 2
    srs = KB * D
    rs = -(-KB * bpb // D) * D + D * off
    return [I.ld(DATA, 0, 4096), I.qact(0, M, 0, KB, srs, dup=pair),
            I.mm(INT8 + D * off, MX_BASE[wf], 4096, N if rows is None else rows, KB, rs, N, M, 0,
                 MX_PITCH[wf], wf=wf, pair=pair, k0=k0),
            I.halt()]


CASES = [(1, 4, 0, 0), (1, 6, 0, 1), (1, 9, 4, 0), (1, 6, 2, 1), (2, 5, 0, 0), (1, 2, 0, 0),
         (1, 7, 6, 0), (1, 3, 2, 1), (1, 1, 0, 0)]


@pytest.mark.parametrize("wf", [I.W4F, I.WNV, I.W4I, I.W8])
@pytest.mark.parametrize("M,KB,k0,off", CASES)
def test_quad_pair_is_the_same_integer(wf, M, KB, k0, off):
    """QUAD's 4-bit PAIR (four blocks an advance at 4*M == MCOLS, else two) and int8's (DUAL's two
    chunks) give the MM's result without PAIR, bit for bit; the copies are ACT rows j + g*M."""
    if wf == I.W8 and k0 % 2:
        pytest.skip("int8 PAIR needs an even k0")
    img = _images(np.random.default_rng(KB + 7 * M + 13 * k0 + wf), 1)[0]
    ref = Machine(_cfg(quad=False), [_prog(M, KB, k0, False, wf, off=off)], [img.copy()]).run()
    q = Machine(_cfg(), [_prog(M, KB, k0, True, wf, off=off)], [img.copy()]).run()
    assert np.array_equal(q.slices[0].tmem, ref.slices[0].tmem)
    s = q.slices[0]
    copies = 4 if 4 * M <= 4 else 2
    for g in range(1, copies):
        assert np.array_equal(s.act[g * M:(g + 1) * M], s.act[:M])
    if copies == 2:                                # 2 * M == MCOLS: rows 2M.. untouched
        assert not s.act[2 * M:].any()


def test_quad_dup_bounds():
    """QACT DUP still needs 2*rows <= MCOLS."""
    img = _images(np.random.default_rng(1), 1)[0]
    with pytest.raises(SimError, match="QACT out of ACT RAM bounds"):
        Machine(_cfg(), [[I.qact(0, 3, 0, 2, 64, dup=True), I.halt()]], [img]).run()


def _same(cfg, prog, img, **kw):
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [img.copy()], **kw)
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    return st


@pytest.mark.parametrize("wf", [I.W4F, I.WNV, I.W4I])
@pytest.mark.parametrize("M,KB,k0,off", CASES)
def test_quad_rtl(have_verilator, wf, M, KB, k0, off):
    """The RTL with QUAD (MCOLS 4) runs the 4-bit PAIR MM as the ISA simulator: the same TMEM;
    M = 1 in ceil(KB/4) advances a row (k0 0, 2, 4, 6: block data within a scale line, across
    two, the row's first advance across), M = 2 in PAIR's ceil(KB/2)."""
    cfg = _cfg()
    N = 20
    st = _same(cfg, _prog(M, KB, k0, True, wf, N=N, off=off),
               _images(np.random.default_rng(KB + 7 * M + 13 * k0 + wf), 1)[0])
    a = st["act_reads"][0]
    assert a["pop"] == N * -(-KB // (4 if 4 * M == cfg.MCOLS else 2))


@pytest.mark.parametrize("seed,axi", [(s, False) for s in range(6)] + [(4, True), (5, True),
                                                                        (6, True), (7, True)])
def test_fuzz_quad(have_verilator, seed, axi):
    """Random QACT DUP + MM sequences on a QUAD configuration: 4-bit PAIR of M = MCOLS/4 (QUAD)
    and of more rows (PAIR), int8 PAIR, plain MMs, MX / RSC / UNIT scales, ACC, RMAX, ASCALE, on
    the same operands, back to back (the MXU merges commands alike). axi: the board's memory path,
    otpu_native_dram2 with random stalls and latency (D = 128)."""
    rng = np.random.default_rng(700 + seed)
    cfg = Config(S=1, D=32 if seed < 4 else 128, MCOLS=4 if seed % 2 == 0 else 8, PAIR=True,
                 DUAL=True, QUAD=True)
    D = cfg.D
    prog = [I.ld(DATA, 0, 4096)]
    out = 4096
    for _ in range(8):
        M = cfg.MCOLS // 4 if rng.random() < 0.6 else int(rng.integers(1, cfg.MCOLS // 2 + 1))
        KB = int(rng.integers(1, 10 if D == 32 else 4))
        srs = KB * D + int(rng.integers(0, 4))
        M = min(M, 4096 // srs)
        prog.append(I.qact(int(rng.integers(0, 4096 - M * srs)), M, 0, KB, srs,
                           row=bool(rng.integers(2)), dup=True))
        for _ in range(int(rng.integers(1, 4))):
            wf = [I.W4F, I.WNV, I.W4I, I.W8][int(rng.choice([0, 0, 1, 1, 2, 3]))]
            pair = rng.random() < 0.85
            N = int(rng.integers(1, 24))
            rb = KB * D if wf == I.W8 else -(-KB // 2) * D
            rs = rb + D * int(rng.integers(0, 3))
            sa = INT8 + D * int(rng.integers(0, (32768 - N * rs) // D + 1))
            unit = bool(rng.integers(5) == 0)
            ssa, srs_s, k0, rsc = _mx_scales(rng, wf, N, KB, pair, unit)
            acc = bool(rng.integers(2)) and out > 4096
            asc = None
            if unit and acc and rng.integers(2):
                asc = int(rng.integers(0, 4096 - M))
            prog.append(I.mm(sa, ssa, 4096 if acc else out, N, KB, rs, N, M, 0, srs_s, unit=unit,
                             acc=acc, rmax=bool(rng.integers(2)), ascale=asc, wf=wf,
                             pair=pair, k0=k0, rsc=rsc))
            if not acc:
                out += M * N + M + 8
    prog.append(I.halt())
    _same(cfg, prog, _images(rng, 1)[0], **({"axi": True, "stall": 20, "seed": seed} if axi else {}))
