"""RTL fp32 package vs the Python bit-exact model (and the model vs float64 math)."""
import re

import numpy as np
import pytest

from opentpu import fp32 as F
from opentpu import rtlsim

rng = np.random.default_rng(1)


def _vals(n):
    """Random fp32 values spanning many exponents plus edge cases."""
    mant = rng.integers(0, 1 << 23, n, dtype=np.uint32)
    exp = rng.integers(90, 165, n, dtype=np.uint32)
    sign = rng.integers(0, 2, n, dtype=np.uint32)
    v = F.from_bits((sign << 31) | (exp << 23) | mant).copy()
    edge = F.f32([0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 2.5, -2.5, 127.5, -127.5, 126.9, 3e38, -3e38,
                  1.2e-38, -1.2e-38, 1e-45, 2.0 ** -126, 1 - 2 ** -24, 1 + 2 ** -23, 65504.0,
                  -126.0, -126.5, 127.99, 128.0, 0.4999999, 0.5000001, np.inf, -np.inf,
                  2.0 ** 126, -2.0 ** 126, 2.0 ** 125.9, 1.7e38, 2.0 ** 127])
    return np.concatenate([v, edge]).astype(np.float32)


def _vectors(n=20000):
    a, b = _vals(n), _vals(n)
    # near-cancellation pairs for add/sub
    near = F.from_bits(F.bits(a[: n // 4]) ^ np.uint32(0x80000000)).copy()
    near = F.from_bits(F.bits(near) + rng.integers(-3, 4, n // 4).astype(np.uint32)).copy()
    rows = []
    def add(op, x, y, e):
        rows.append(np.stack([np.full(len(x), op, np.uint32), F.bits(x), F.bits(y),
                              np.asarray(e, np.uint32)], 1))
    add(0, a, b, F.bits(F.add(a, b)))
    add(0, a[: n // 4], near, F.bits(F.add(a[: n // 4], near)))
    add(1, a, b, F.bits(F.sub(a, b)))
    add(2, a, b, F.bits(F.mul(a, b)))
    add(3, a, b, F.bits(F.fmax(a, b)))
    add(4, a, b, F.bits(F.fmin(a, b)))
    x = F.f32(rng.uniform(-140, 130, n))
    add(5, x, x, F.bits(F.exp2(x)))
    add(6, a, a, F.bits(F.recip(a)))
    add(7, a, a, F.bits(F.rsqrt(a)))
    ints = rng.integers(-(1 << 22), 1 << 22, n).astype(np.int32)
    add(8, ints.view(np.float32), ints.view(np.float32), F.bits(F.i2f(ints)))
    y = F.f32(rng.uniform(-140, 140, n))
    add(9, y, y, F.q8(y).view(np.uint8).astype(np.uint32))
    add(10, a, b, F.gt(a, b).astype(np.uint32))
    add(11, a, a, F.bits(F.fabs(a)))
    lg = np.concatenate([a, F.from_bits(rng.integers(0, 1 << 32, n, dtype=np.int64)
                                        .astype(np.uint32)),
                         F.from_bits(0x3F3504F3 + np.arange(-8, 8)),
                         F.from_bits(0x3FB504F3 + np.arange(-8, 8))])
    add(12, lg, lg, F.bits(F.log2(lg)))
    # the compares on every pair of classes (signed zeros, subnormals, normals, inf, the
    # canonical NaN) and ties: fp_gt fixes up the flushed operands beside its comparator;
    # fp_mm is MAX / MIN on one
    cls = np.array([(s << 31) | (e << 23) | m for s in (0, 1) for e in (0, 1, 0x7F, 0xFE)
                    for m in (0, 1, 0x2AAAAA, 0x7FFFFF)]
                   + [0x7F800000, 0xFF800000, 0x7FC00000], np.uint32)
    ca, cb = (F.from_bits(x).copy() for x in np.meshgrid(cls, cls))
    ca, cb = np.concatenate([ca.ravel(), a]), np.concatenate([cb.ravel(), a])
    add(10, ca, cb, F.gt(ca, cb).astype(np.uint32))
    add(3, ca, cb, F.bits(F.fmax(ca, cb)))
    add(4, ca, cb, F.bits(F.fmin(ca, cb)))
    for x, y in ((a, b), (ca, cb)):
        add(13, x, y, F.bits(F.fmax(x, y)))
        add(14, x, y, F.bits(F.fmin(x, y)))
    # every exponent (0..254 and inf; no NaN), mantissas biased to their edges
    w, v = _full(n), _full(n)
    for op, fn in ((0, F.add), (1, F.sub), (2, F.mul), (3, F.fmax), (4, F.fmin)):
        add(op, w, v, F.bits(fn(w, v)))
    add(10, w, v, F.gt(w, v).astype(np.uint32))
    for op, fn in ((6, F.recip), (7, F.rsqrt), (12, F.log2)):
        add(op, w, w, F.bits(fn(w)))
    # products at the flush boundary: a power of two times an all-ones mantissa lands on
    # 2^-126 - 2^-150 (or near it); IEEE rounds that up to 2^-126 on the subnormal grid
    e1 = rng.integers(1, 127, n // 4).astype(np.int64)   # e1 + e2 = 127: 2^-126 (1 - 2^-24)
    e2 = np.clip(127 - e1 + rng.integers(-1, 2, n // 4), 1, 254)
    s1, s2 = (rng.integers(0, 2, n // 4).astype(np.int64) << 31 for _ in range(2))
    m2 = np.where(rng.random(n // 4) < 0.7, 0x7FFFFF, rng.integers(0x7FFFF0, 0x800000, n // 4))
    pa = F.from_bits((s1 | (e1 << 23)).astype(np.uint32))
    pb = F.from_bits((s2 | (e2 << 23) | m2).astype(np.uint32))
    pa = np.concatenate([F.f32([0.5, -0.5, 0.25]), pa])
    pb = np.concatenate([F.from_bits(np.uint32([0x00FFFFFF, 0x00FFFFFF, 0x017FFFFF])), pb])
    add(2, pa, pb, F.bits(F.mul(pa, pb)))
    add(2, pb, pa, F.bits(F.mul(pb, pa)))
    # recip / rsqrt over their scaled ranges (exponent fields 0..5 and 246..255), both signs
    m = np.concatenate([[0, 1, 0x7FFFFF, 0x7311C3, 0x7311C4], rng.integers(0, 1 << 23, 2000)])
    sc = np.array([(s << 31) | (e << 23) | (0 if e == 255 else int(k)) for s in (0, 1)
                   for e in list(range(6)) + list(range(246, 256)) for k in m], np.uint32)
    scx = F.from_bits(sc)
    add(6, scx, scx, F.bits(F.recip(scx)))
    add(7, scx, scx, F.bits(F.rsqrt(scx)))
    # exp2 around every integer, near its range limits and where x + 1 rounds to 1 (f = 1)
    k = np.arange(-130, 131).astype(np.float32)
    up, dn = np.nextafter(k, np.float32(np.inf)), np.nextafter(k, np.float32(-np.inf))
    ex = np.concatenate([k, up, dn, F.from_bits(0x43000000 - np.arange(1, 200, dtype=np.uint32)),
                         F.from_bits(0xC2FC0000 + np.arange(-100, 100).astype(np.uint32)),
                         -np.exp2(-np.arange(1, 40)).astype(np.float32)]).astype(np.float32)
    add(5, ex, ex, F.bits(F.exp2(ex)))
    # q8 of a NaN (quantize's 0 * inf) is 0
    qn = F.from_bits(np.uint32([0x7FC00000, 0xFFC00000, 0x7F800001]))
    add(9, qn, qn, F.q8(qn).view(np.uint8).astype(np.uint32))
    # NaN operands (out of spec, but produced by inf - inf, 0 * inf, log2 of a negative and
    # sign flips): every function as the RTL defines it (fp32.py's module docstring)
    nans = np.uint32([0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFBFFFFF, 0x7FFFFFFF])
    na, nb = (F.from_bits(x).copy() for x in np.meshgrid(nans, np.concatenate([nans, cls])))
    na, nb = na.ravel(), nb.ravel()
    for x, y in ((na, nb), (nb, na)):
        for op, fn in ((0, F.add), (1, F.sub), (2, F.mul), (3, F.fmax), (4, F.fmin),
                       (13, F.fmax), (14, F.fmin)):
            add(op, x, y, F.bits(fn(x, y)))
        add(10, x, y, F.gt(x, y).astype(np.uint32))
    nx = F.from_bits(nans)
    for op, fn in ((5, F.exp2), (6, F.recip), (7, F.rsqrt), (11, F.fabs), (12, F.log2)):
        add(op, nx, nx, F.bits(fn(nx)))
    return np.concatenate(rows)


def _full(n):
    """Random fp32 words over every exponent field (0..254, and inf), never NaN."""
    e = rng.integers(0, 256, n).astype(np.uint32)
    m = rng.integers(0, 1 << 23, n, dtype=np.uint32)
    k = rng.random(n)
    m = np.where(k < 0.1, 0, np.where(k < 0.2, 0x7FFFFF, m)).astype(np.uint32)
    m = np.where(e == 255, 0, m).astype(np.uint32)
    s = rng.integers(0, 2, n, dtype=np.uint32)
    return F.from_bits((s << 31) | (e << 23) | m).copy()


def test_mul_rounds_on_the_subnormal_grid_before_the_flush():
    # 2^-126 - 2^-150 is a tie on the subnormal grid: it rounds to 2^-126, which is kept
    assert F.bits(F.mul(F.f32(0.5), F.from_bits(np.uint32(0x00FFFFFF)))) == 0x00800000
    assert F.bits(F.mul(F.f32(-0.5), F.from_bits(np.uint32(0x00FFFFFF)))) == 0x80800000
    assert F.bits(F.mul(F.f32(0.5), F.from_bits(np.uint32(0x00FFFFFE)))) == 0


def test_recip_rsqrt_every_exponent():
    """recip and rsqrt within 2e-7 of the exact value for every exponent field, wherever the
    result is a normal number (unscaled, rsqrt failed at fields 1 and >= 252, recip near 2^126)."""
    m = np.concatenate([[0, 1, 0x7FFFFF, 0x400000, 0x7311C3], rng.integers(0, 1 << 23, 3000)])
    x = F.from_bits(np.array([(e << 23) | int(k) for e in range(1, 255) for k in m], np.uint32))
    xd = x.astype(np.float64)
    r = F.recip(x).astype(np.float64)
    ok = xd < 2.0 ** 126
    assert np.max(np.abs(r[ok] * xd[ok] - 1)) < 2e-7
    assert np.all(r[~ok] == 0)
    assert np.all(F.recip(-x)[ok] == -F.recip(x)[ok])
    q = F.rsqrt(x).astype(np.float64)
    assert np.max(np.abs(q * np.sqrt(xd) - 1)) < 2e-7
    assert np.all(F.rsqrt(-x) == 0)


def test_q8_nan_is_zero():
    assert F.q8(F.from_bits(np.uint32([0x7FC00000, 0xFFC00000]))).tolist() == [0, 0]
    # a tiny amax: inv = 127 * recip(amax) is +inf, and the zeros (0 * inf) quantize to 0
    x = np.zeros((1, 32), np.float32)
    x[0, 5] = F.from_bits(np.uint32(0x01000000))
    q, s = F.quantize(x, axis=1)
    assert q[0].tolist() == [0] * 5 + [127] + [0] * 26 and s[0] == 0


def test_mm_partials_have_no_pad_terms():
    """MM's isum_4 adds no +0 pad terms (a -0 partial, from a sum that flushed, stays -0);
    the VOP sums pad with +0 terms (added: -0 becomes +0). otpu_mxu's order, written out."""
    m = F.f32(2.0 ** -126)
    t = F.f32([-1.5 * m] * 4 + [m] * 4 + [-0.0, -0.0])              # KB = 10

    def mxu(t):
        pacc = []
        for k in range(len(t)):
            pacc.append(F.add(F.f32(0) if k < 4 else pacc[k - 4], t[k]))
        C = len(t)
        cy = lambda i: F.add(pacc[i], pacc[i - 2] if i >= 2 else F.f32(0))
        return F.add(cy(C - 2) if C >= 2 else F.f32(0), cy(C - 1))
    assert F.bits(mxu(t)) == 0x80000000
    assert F.bits(F.interleaved_sum(t[None, :], F.MM_PARTIALS, pad=-0.0)[0]) == 0x80000000
    assert F.bits(F.interleaved_sum(t[None, :], F.MM_PARTIALS)[0]) == 0
    r = np.random.default_rng(4)
    for _ in range(2000):
        C = int(r.integers(1, 20))
        v = F.ftz((r.integers(-3, 4, C) * 0.5 * m).astype(np.float32))
        assert F.bits(mxu(v)) == F.bits(F.interleaved_sum(v[None, :], 4, pad=-0.0)[0])


def test_model_accuracy():
    x = F.f32(np.linspace(-30, 30, 10001))
    assert np.max(np.abs(F.exp2(x) / np.exp2(x.astype(np.float64)) - 1)) < 2e-6
    p = F.f32(np.exp(rng.uniform(-40, 40, 10000)))
    assert np.max(np.abs(F.recip(p) * p.astype(np.float64) - 1)) < 1e-6
    assert np.max(np.abs(F.rsqrt(p) * np.sqrt(p.astype(np.float64)) - 1)) < 1e-6
    q, s = F.quantize(F.f32(rng.standard_normal((16, 32))), axis=1)
    assert np.all(np.abs(q) <= 127) and np.all(s > 0)


def _ulps(y, ref):
    """|y - ref| in ulps of the binade of ref (float64, normal)."""
    e = np.floor(np.log2(np.abs(ref)))
    return np.abs(y.astype(np.float64) - ref) / np.exp2(np.maximum(e, -126) - 23)


def _binade(e0, sign=0):
    """Every fp32 in [2^e0, 2^(e0+1)), with the sign bit `sign`."""
    return F.from_bits(np.uint32(sign << 31) | np.uint32((e0 + 127) << 23)
                       | np.arange(1 << 23, dtype=np.uint32)).copy()


def test_exp2_error_bound():
    """exp2 (docs/isa.md: minimax coefficients) within 1.3 ulp where f = x - floor(x) is exact
    (every fp32 in [0.5, 1) and (-1, -0.5]) and 1.65 ulp where x + 1 rounds to f (every fp32 in
    (-0.5, -0.25] and (-2^-8, -2^-10]: the worst binades), and on samples of [-126, 128). The
    Taylor coefficients were 13 ulp off near f = 1."""
    r = np.random.default_rng(5)
    for x, bound in ((_binade(-1), 1.3), (_binade(-1, 1), 1.3),
                     (_binade(-2, 1), 1.65), (_binade(-9, 1), 1.65), (_binade(-10, 1), 1.65),
                     (F.f32(r.uniform(-126, 128, 1 << 22)), 1.65)):
        u = _ulps(F.exp2(x), np.exp2(x.astype(np.float64)))
        assert u.max() < bound, (u.max(), x[np.argmax(u)])


def test_exp2_exact_at_integers_and_monotone():
    """exp2(i) = 2^i for every integer in [-126, 127] (C0 = 1); non-decreasing over every fp32 in
    [0.25, 1), around every integer and on [-2^-22, 0] (p(f) <= 2 as f reaches 1, also where x + 1
    rounds to 1), and on a sorted sample of [-126, 128)."""
    i = np.arange(-126, 128)
    assert np.array_equal(F.exp2(i.astype(np.float32)), np.exp2(i).astype(np.float32))
    x = np.concatenate([_binade(-2), _binade(-1)])
    assert np.all(np.diff(F.exp2(x).astype(np.float64)) >= 0)
    k = np.arange(-125, 128).astype(np.float32)[:, None]
    x = np.concatenate([np.nextafter(k, np.float32(-np.inf)), k,
                        np.nextafter(k, np.float32(np.inf))], axis=1).astype(np.float32)
    assert np.all(np.diff(F.exp2(x).astype(np.float64), axis=1) >= 0)
    x = np.concatenate([-_binade(-23)[::-1], -_binade(-24)[::-1], -_binade(-25)[::-1], F.f32([0])])
    y = F.exp2(x)
    assert np.all(np.diff(y.astype(np.float64)) >= 0) and y[-1] == 1.0
    r = np.random.default_rng(6)
    x = np.sort(F.f32(r.uniform(-126, 128, 1 << 21)))
    assert np.all(np.diff(F.exp2(x).astype(np.float64)) >= 0)


def test_rtl_fp_bit_exact(tmp_path):
    vec = _vectors()
    f = tmp_path / "vec.txt"
    np.savetxt(f, vec, fmt="%08x")
    out = rtlsim.run_fp_vectors(f)
    m = re.search(r"FPTEST cases=(\d+) errors=(\d+)", out)
    assert m, out
    assert int(m.group(1)) == len(vec)
    assert int(m.group(2)) == 0, out
