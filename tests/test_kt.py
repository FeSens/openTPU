"""MM's K tail (KT, docs/isa.md "K tail"; Config.KT): a grouped operand whose K ends in the middle
of its last K-block, its groups in pairs sharing that block's chunk (quant.kt_rows / kt_scales),
gives the bits of the same MM over the padded groups, for int8, int4 and fp4, UNIT, RSC, ACC,
odd and even group counts, KB 1 to 23, M past MCOLS and the channel hash; and the refusals."""
import numpy as np
import pytest

from opentpu import Config, isa as I, quant as Q
from opentpu.isasim import Machine, SimError

X, WA, SA, OUT = 0, 1 << 16, 3 << 18, 1 << 12
FMT = {I.W8: "int8", I.W4I: "int4", I.W4F: "fp4"}
WG = 8


def _cfg(D, **kw):
    return Config(S=1, D=D, MCOLS=8, ACT_BLOCKS=32, ACT_ROWS=16, WG=WG, GALL=True,
                  DRAM_BYTES=1 << 22, **kw)


def _operand(rng, wf, N, KB, D):
    """Natural rows [N, .] and scale rows of K = (KB - 1/2) D: the last block's upper half 0."""
    fmt = FMT[wf]
    K = KB * D - D // 2
    if wf == I.W8:
        el = rng.integers(-127, 128, (N, KB * D))
        el[:, K:] = 0
        data = el.astype(np.int8).view(np.uint8)
    else:
        el = rng.integers(0, 16, (N, KB * D))
        el[:, K:] = 0
        data = np.zeros((N, Q.row_bytes(KB * D, fmt, D)), np.uint8)
        data[:, :KB * D // 2] = Q.pack4(el)
    sw = (rng.standard_normal(N) * 0.01).astype(np.float32)
    rows = np.zeros((N, Q.scale_bytes(KB * D, fmt, D)), np.uint8)
    rows[:, :4] = sw[:, None].view(np.uint8)
    h = Q.swp(fmt, 1)
    if wf == I.W8:
        rows[:, h:h + KB] = rng.integers(0, 20, (N, KB))
    else:
        word = rng.integers(0, 20, (N, KB)) | sum(
            rng.integers(0, 64, (N, KB)) << (8 + 6 * b) for b in range(4))
        rows[:, h:h + 4 * KB] = word.astype(np.uint32).view(np.uint8).reshape(N, -1)
    return data, rows


def _act(rng, M, KB, D):
    x = (rng.standard_normal((M, KB * D)) * np.exp(rng.uniform(-6, 2, (M, KB * 4, 1)))
         .repeat(D // 4, axis=2).reshape(M, -1)).astype(np.float32)
    x[:, KB * D - D // 2:] = 0
    return x


def _place(dram, at, a):
    b = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
    dram[at:at + b.size] = b


def _mm_both(wf, N, KB, M, unit=False, rsc=False, acc=False, chash=False, D=32, seed=0):
    """The MM's TMEM outputs over the padded groups and over KT's pairs."""
    rng = np.random.default_rng(seed)
    fmt = FMT[wf]
    x = _act(rng, M, KB, D)
    data, rows = _operand(rng, wf, N, KB, D)
    g = Q.group_rows(data, WG, D, fmt, KB * D)
    s = Q.group_scales(rows, fmt, KB * D, WG, D, rsc)
    ors, outs = N + 1, []
    for kt in (False, True):
        cfg = _cfg(D, CHASH=chash, KT=kt)
        d, a = (Q.kt_rows(g, WG, D), Q.kt_scales(s, fmt, KB, WG, rsc)) if kt else (g, s)
        if chash:
            d = Q.hash_rows(d, WA, WG, D)
        dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
        _place(dram, X, x)
        _place(dram, WA, d)
        _place(dram, SA, a)
        mm = dict(sa=WA, ssa=SA, out=OUT, n=N, kb=KB, rs=d.shape[1], ors=ors, m=M, ab=1,
                  srs=a.shape[1], unit=unit, wf=wf, rsc=rsc, grp=Q.grp_r(fmt, WG), kt=kt)
        prog = [I.ld(X, 0, x.size), I.qact(0, M, 1, KB, KB * D), I.mm(**mm)]
        if acc:
            prog.append(I.mm(**mm, acc=True))
        sl = Machine(cfg, [prog + [I.halt()]], [dram]).run().slices[0]
        outs.append(sl.tget(OUT + np.arange(M)[:, None] * ors + np.arange(N)[None, :]))
    return outs


CASES = [(I.W8, False, False), (I.W8, True, False), (I.W8, False, True),
         (I.W4I, False, False), (I.W4I, False, True), (I.W4F, False, False),
         (I.W4F, True, False), (I.W4F, False, True)]


@pytest.mark.parametrize("wf,unit,rsc", CASES)
@pytest.mark.parametrize("chash", [False, True])
def test_kt_mm_is_the_padded_mm(wf, unit, rsc, chash):
    i = 0
    R = Q.grp_r(FMT[wf], WG)
    for N in (1, R, 2 * R, 3 * R - 5, 4 * R):          # one group, two, three, four
        for KB in (1, 2, 5):
            for M in (1, 3, 9):
                a, b = _mm_both(wf, N, KB, M, unit, rsc, acc=i % 3 == 0, chash=chash,
                                seed=1000 * wf + 100 * unit + 10 * rsc + 7 * i)
                i += 1
                assert a.view(np.uint32).tolist() == b.view(np.uint32).tolist(), \
                    (FMT[wf], N, KB, M)


@pytest.mark.parametrize("wf", [I.W8, I.W4F])
def test_kt_mm_at_d128(wf):
    """GPT-OSS's shapes at D 128: K 2880 (KB 23) and the experts' last part (KB 1)."""
    for N, KB, M in ((40, 23, 1), (64, 23, 2), (48, 1, 1)):
        a, b = _mm_both(wf, N, KB, M, chash=True, D=128, seed=N + KB)
        assert a.view(np.uint32).tolist() == b.view(np.uint32).tolist(), (N, KB, M)


def test_kt_layout_saves_a_chunk_a_pair():
    D, KB = 32, 5
    rng = np.random.default_rng(3)
    data, _ = _operand(rng, I.W4F, 64, KB, D)
    g = Q.group_rows(data, WG, D, "fp4", KB * D)
    k = Q.kt_rows(g, WG, D)
    assert g.shape == (4, KB * WG * D) and k.shape == (2, (2 * KB - 1) * WG * D)
    with pytest.raises(ValueError, match="upper halves"):
        Q.kt_rows(g + 1, WG, D)


def test_kt_refusals():
    D, KB, N, M = 32, 3, 16, 1
    rng = np.random.default_rng(5)
    x = _act(rng, M, KB, D)
    data, rows = _operand(rng, I.W8, N, KB, D)
    d = Q.kt_rows(Q.group_rows(data, WG, D), WG, D)
    a = Q.kt_scales(Q.group_scales(rows, "int8", KB * D, WG, D), "int8", KB, WG)
    base = dict(sa=WA, ssa=SA, out=OUT, n=N, kb=KB, rs=d.shape[1], ors=N, m=M, ab=1,
                srs=a.shape[1], grp=WG, kt=True)

    def run(cfg, mm, xx=x):
        dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
        _place(dram, X, xx)
        _place(dram, WA, d)
        _place(dram, SA, a)
        prog = [I.ld(X, 0, xx.size), I.qact(0, M, 1, KB, KB * D), I.mm(**mm), I.halt()]
        Machine(cfg, [prog], [dram]).run()

    run(_cfg(D, KT=True), base)
    with pytest.raises(SimError, match="without it"):
        run(_cfg(D), base)
    with pytest.raises(SimError, match="upper half zero"):
        y = x.copy()
        y[:, -1] = 1.0
        run(_cfg(D, KT=True), base, y)
    with pytest.raises(SimError, match="pair stride"):
        run(_cfg(D, KT=True), dict(base, rs=WG * D))
    with pytest.raises(ValueError, match="KT"):
        I.mm(**dict(base, k0=1))
    with pytest.raises(ValueError, match="KT"):
        I.mm(**dict(base, grp=1))


def test_kt_footprint_covers_the_reads():
    """waits.footprint (otpu_pkg footprint()) of a KT MM: rs (bit 0 set) over (2 KB - 1) P
    covers the pairs' chunks."""
    from opentpu import waits as WT
    D = 32
    for wf in (I.W8, I.W4F):
        fmt = FMT[wf]
        R = Q.grp_r(fmt, WG)
        for N, KB in ((1, 1), (R, 3), (2 * R, 3), (5 * R - 1, 4)):
            P = WG * D
            rs = (2 * KB - 1) * P
            ins = I.mm(WA, SA, OUT, N, KB, rs, N, 1, 0, 64, wf=wf, grp=R, kt=True)
            fp = WT.footprint(ins, [(0, 0)] * 16, D)
            (_, d0, d1) = fp.rd[0]
            G = -(-N // R)
            assert d0 == WA and d1 >= WA + (G - 1) // 2 * rs + (KB if G % 2 else 2 * KB - 1) * P


def test_kt_layout_helpers_and_descriptors():
    """The compiler's side: quant.grouped kt (kt_rows and kt_scales of group_rows' and
    group_scales'), grp_bytes kt (whole pairs), kt_fits (a K that ends in the first half of its
    last block); a KT QTensor's row slices start at a pair, and its K is streamed whole."""
    from opentpu import compiler as C
    rng = np.random.default_rng(3)
    D, N, KB = 32, 3 * 2 * WG, 3                      # int8: 6 groups, 3 pairs
    q, s = _operand(rng, I.W8, N, KB, D)
    d0, s0 = Q.grouped(q, s, "int8", KB * D, WG, D)
    d1, s1 = Q.grouped(q, s, "int8", KB * D, WG, D, kt=True)
    assert np.array_equal(d1, Q.kt_rows(d0, WG, D)) and np.array_equal(
        s1, Q.kt_scales(s0, "int8", KB, WG))
    assert d1.size == Q.grp_bytes(N, KB * D, "int8", D, WG, kt=True)[0] == \
        3 * Q.kt_pair_bytes(KB * D, "int8", D, WG) == 3 * (2 * KB - 1) * WG * D
    assert Q.grp_bytes(N + 1, KB * D, "int8", D, WG, kt=True)[0] == 4 * (2 * KB - 1) * WG * D
    assert [Q.kt_fits(k, 128) for k in (2880, 2944, 2900, 2817, 64, 128)] == \
        [True, False, False, True, True, False]
    with pytest.raises(ValueError):
        Q.grouped(q, s, "int8", KB * D, 1, D, kt=True)
    w = C.QTensor.weight(WA, SA, N, KB * D, "int8", D, grp=WG, kt=True)
    assert w.kt and w.rs == (2 * KB - 1) * WG * D
    v = w[2 * WG:N, :]                                 # from pair 1
    assert v.kt and (v.data - w.data).static() == w.rs and (v.scale - w.scale).static() == \
        2 * w.srs
    with pytest.raises(C.CompileError, match="pair"):
        w[WG:N, :]                                     # group 1: inside pair 0
    with pytest.raises(C.CompileError, match="whole K"):
        w[:, 0:D]
