"""The stream engine's streams as their op lists (docs/stream.md 4.3 / 4.4): the reference
(every dmode, every gate, A on slot 1 or 2, Q on or off) and the random streams with special
values that tests/test_se_vpu.py runs through the VPU with its tail (otpu_se_tail shares the
VPU's tree, so it is tested inside otpu_vpu, not on its own)."""
import numpy as np

from opentpu import fp32 as F

L = 8
SF_Q, SF_K, SF_X, SF_K0, SF_K1, SF_A, SF_G = 1, 2, 3, 4, 5, 6, 7
DELTA, DELTA1, SCALE, DOT = 0, 1, 2, 3
G_K0, G_COL, G_ONE = 0, 1, 2


def _special(rng, n, inf=0.01):
    """Normals with +-0, denormals (flushed), a few huge values and +-inf (fraction inf)."""
    x = rng.standard_normal(n).astype(np.float32)
    u = rng.random(n)
    x[u < 0.04] = 0.0
    x[(u >= 0.04) & (u < 0.06)] = -0.0
    x[(u >= 0.06) & (u < 0.08)] = np.float32(1e-40) * np.sign(x[(u >= 0.06) & (u < 0.08)])
    x[(u >= 0.08) & (u < 0.09)] = np.float32(3e38) * np.sign(x[(u >= 0.08) & (u < 0.09)])
    x[(u >= 0.09) & (u < 0.09 + inf)] = np.inf * np.sign(x[(u >= 0.09) & (u < 0.09 + inf)])
    return x


def reference(S, q, k, a, g, x, K0, K1, a_en, a_sel, dmode, g_src, q_en):
    """One stream on the hardware subset, as its op list: A = RDOT(S, k or a) (+0 without A),
    d by dmode, Y = OUTER(S, G, d, k) = add(mul(S, G), mul(d, k)), O = RDOT(Y, q)."""
    rows, cols = S.shape
    A = F.rdot(S, (a if a_sel else k)[None, :]) if a_en else np.zeros(rows, np.float32)
    K0a, K1a = np.float32(K0)[None], np.float32(K1)[None]
    if dmode == DELTA:
        d = F.mul(F.sub(x, F.mul(A, K0a)), K1a)
    elif dmode == DELTA1:
        d = F.mul(F.sub(x, A), K1a)
    elif dmode == SCALE:
        d = F.mul(x, K1a)
    else:
        d = F.mul(A, K1a)
    G = {G_K0: np.full(cols, K0, np.float32), G_COL: g, G_ONE: np.ones(cols, np.float32)}[g_src]
    Y = F.outer(S, G[None, :], d[:, None], k[None, :])
    O = F.rdot(Y, q[None, :]) if q_en else None
    return F.f32(Y), O


def _stream(rng, special, dmode=None, g_src=None):
    rows = int(rng.choice([1, 7, 8, 9, 63, 64, 128, 129, 200, 256]))
    cols = int(rng.choice([64, 128, 192, 256]))
    dmode = int(rng.integers(4)) if dmode is None else dmode
    g_src = int(rng.integers(3)) if g_src is None else g_src
    a_en = dmode != SCALE
    a_sel = bool(a_en and rng.integers(2))
    q_en = bool(rng.integers(4) != 0)
    # specials: +-0, denormals and huge values everywhere; infinities in a few state words
    # and, rarely, in a vector (an infinite k or a turns every row's d into NaN)
    gen = (lambda n, inf=0.0: _special(rng, n, inf)) if special else \
        (lambda n, inf=0.0: rng.standard_normal(n).astype(np.float32))
    S = (0.3 * gen(rows * cols, 0.0005)).astype(np.float32).reshape(rows, cols)
    vi = 0.002 if rng.integers(4) == 0 else 0.0
    q, k, a, x = gen(cols, vi), gen(cols, vi), gen(cols, vi), gen(rows, 0.01)
    g = rng.uniform(0.2, 1, cols).astype(np.float32)
    K0, K1 = np.float32(rng.uniform(0.2, 1)), np.float32(rng.uniform(0, 1))
    if special and rng.integers(3) == 0:
        K0, K1 = rng.choice(_special(rng, 64, 0.0), 2)
    return dict(S=S, q=q, k=k, a=a, g=g, x=x, K0=K0, K1=K1, a_en=a_en, a_sel=a_sel,
                dmode=dmode, g_src=g_src, q_en=q_en)


def _fills(st, rng):
    """The fill steps (docs/stream.md 4.3), in the DMA's order or shuffled (the tail's writes
    do not depend on it)."""
    ns = st["S"].shape[1] // L
    rows = st["S"].shape[0]
    b = lambda v: F.bits(np.asarray(v, np.float32))
    f = [(SF_Q, i, b(st["q"][L * i:L * i + L])) for i in range(ns)]
    f += [(SF_K, i, b(st["k"][L * i:L * i + L])) for i in range(ns)]
    if st["a_sel"]:
        f += [(SF_A, i, b(st["a"][L * i:L * i + L])) for i in range(ns)]
    if st["g_src"] == G_COL:
        f += [(SF_G, i, b(st["g"][L * i:L * i + L])) for i in range(ns)]
    xp = np.zeros(-(-rows // L) * L, np.float32)
    xp[:rows] = st["x"]
    f += [(SF_X, i, b(xp[L * i:L * i + L])) for i in range(len(xp) // L)]
    f += [(SF_K0, 0, b([st["K0"]] + [0] * (L - 1))), (SF_K1, 0, b([st["K1"]] + [0] * (L - 1)))]
    if rng.integers(2):
        f = [f[i] for i in rng.permutation(len(f))]
    return f


def _check(streams, outs):
    for i, (st, got) in enumerate(zip(streams, outs)):
        Y, O = reference(**st)
        gy = np.array(got["Y"], np.uint32).reshape(Y.shape)
        bad = np.argwhere(gy != F.bits(Y))
        tag = {k: st[k] for k in ("a_en", "a_sel", "dmode", "g_src", "q_en")}
        assert len(bad) == 0, f"stream {i} {st['S'].shape} {tag}: {len(bad)} Y words differ, " \
            f"first at {bad[:4].tolist()}"
        if O is None:
            assert got["O"] == []
        else:
            go = np.array(got["O"], np.uint32).reshape(-1)
            bad = np.nonzero(go != F.bits(O))[0]
            assert len(bad) == 0, f"stream {i} {st['S'].shape} {tag}: {len(bad)} O differ, " \
                f"first at {bad[:8]}"


def test_reference_is_dstep():
    """GDN mode of the reference is the ISA simulator's DSTEP (RDOT, MUL, SUB, MUL, OUTER,
    RDOT)."""
    from opentpu import isa as I
    from opentpu.isasim import Config, Machine
    rng = np.random.default_rng(7)
    rows, cols = 129, 192
    st = _stream(rng, True, DELTA, G_K0)
    st.update(S=(0.3 * _special(rng, rows * cols, 0.0005)).reshape(rows, cols),
              q=_special(rng, cols, 0.0), k=_special(rng, cols, 0.0), x=_special(rng, rows),
              a_sel=False, q_en=True)
    qk, v, g, o = 1000, 2000, 3000, 4000
    tm = np.zeros(1 << 16, np.uint32)
    tm[qk:qk + cols] = F.bits(st["q"])
    tm[qk + cols:qk + 2 * cols] = F.bits(st["k"])
    tm[v:v + rows] = F.bits(st["x"])
    tm[g], tm[g + 1] = F.bits(np.float32(st["K0"])), F.bits(np.float32(st["K1"]))
    img = np.zeros(1 << 20, np.uint8)
    img[:rows * cols * 4] = st["S"].view(np.uint8).reshape(-1)
    img[0x80000:0x80000 + 4 * len(tm)] = tm.view(np.uint8)
    prog = [I.ld(0x80000, 0, len(tm)), I.dstep(0, qk, v, rows, cols, g, 1, o), I.halt()]
    m = Machine(Config(S=1, D=128, DRAM_BYTES=1 << 20, DSTEP=True), [prog], [img]).run()
    Y, O = reference(**st)
    assert np.array_equal(m.slices[0].dram[:rows * cols * 4].view(np.uint32),
                          F.bits(Y).reshape(-1))
    assert np.mean(np.isnan(Y)) < 0.5
    assert np.array_equal(m.slices[0].tmem[o:o + rows], F.bits(O))
