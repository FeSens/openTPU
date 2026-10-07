"""Bit-exact fp32 arithmetic of the openTPU (docs/isa.md, "Arithmetic").

Every function accepts and returns numpy float32 arrays (or scalars). Semantics are IEEE-754
binary32 round-to-nearest-even with flush-to-zero on inputs and outputs. The RTL implements the
same functions; tests compare the two bit for bit.

NaN (docs/isa.md: no NaN inputs are expected) follows the RTL: add / sub / mul return the
canonical 0x7FC00000 for any NaN; ftz, max / min, abs, COPY / FILL keep a NaN's bits; the
compares order raw sign-magnitude bits (a NaN with the sign set is below -inf).
"""
from __future__ import annotations

import numpy as np

F32 = np.float32
MIN_NORMAL = F32(2.0 ** -126)

# 2^f ~ C0 + f*(C1 + ... + f*C7) on [0, 1]: a minimax fit of the relative error (degree 7) with
# C0 = 1, rounded to fp32, then searched in fp32 for the smallest error through the Horner steps.
EXP2_COEFFS = [np.uint32(b).view(np.float32) for b in (
    0x3F800000, 0x3F317218, 0x3E75FDE9, 0x3D63593E, 0x3C1D8534, 0x3AAFE2F6, 0x3916C07C, 0x37B3C7D6)]
# log2(1+t) ~ t*(C1 + t*(C2 + ... + t*C9)) on [sqrt(1/2)-1, sqrt(2)-1]: a minimax fit of the
# relative error (degree 8 in t), rounded to fp32. Horner from C9 down to C1.
LOG2_COEFFS = [np.uint32(b).view(np.float32) for b in (
    0x3FB8AA3B, 0xBF38AA38, 0x3EF639EB, 0xBEB8AE27, 0x3E9369C2, 0xBE74ADF2, 0x3E5CE48E,
    0xBE543E8E, 0x3E00DB73)]
LOG2_SQRT2 = 0x3504F3        # mantissa bits of sqrt(2): m >= sqrt(2) is halved
INV127 = F32(1.0 / 127.0)
RECIP_MAGIC = 0x7EF311C3
RSQRT_MAGIC = 0x5F3759DF


def f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def bits(x) -> np.ndarray:
    return f32(x).view(np.uint32)


def from_bits(b) -> np.ndarray:
    return np.asarray(b, dtype=np.uint32).view(np.float32)


def ftz(x) -> np.ndarray:
    """Flush denormals to signed zero (a NaN keeps its bits, as in the RTL's ftz)."""
    x = f32(x)
    if not ((x.view(np.uint32) & np.uint32(0x7F800000)) == 0).any():
        return x
    den = (np.abs(x) < MIN_NORMAL) & (x != 0)
    if den.any():
        x = np.where(den, np.copysign(F32(0), x), x).astype(np.float32)
    return x


def _canon(x) -> np.ndarray:
    """An arithmetic result: every NaN is the canonical 0x7FC00000 (the RTL's fp_add / fp_mul;
    numpy's NaN bits depend on the platform)."""
    x = f32(x)
    nan = np.isnan(x)
    if nan.any():
        x = np.where(nan, from_bits(np.uint32(0x7FC00000)), x).astype(np.float32)
    return x


def chain_sum(a) -> np.ndarray:
    """Row sums of a [rows, cols], accumulated strictly left to right with add() (FTZ at every
    step), i.e. acc = add(...add(add(0, a[:, 0]), a[:, 1])..., a[:, cols-1]).

    Fast path: numpy's float32 accumulate is the same sequential RNE sum; FTZ only matters if
    a partial sum is denormal, and those rows are redone step by step.
    """
    a = ftz(a)
    z = np.zeros((a.shape[0], 1), np.float32)          # starts from +0 (so -0 + -0 ... = +0)
    with np.errstate(over="ignore", invalid="ignore"):
        part = np.add.accumulate(np.concatenate([z, a], axis=1), axis=1, dtype=np.float32)
    out = _canon(ftz(part[:, -1])).copy()
    bad = ((np.abs(part) < MIN_NORMAL) & (part != 0)).any(axis=1)
    for r in np.nonzero(bad)[0]:
        acc = F32(0)
        for c in range(a.shape[1]):
            acc = add(acc, a[r, c])
        out[r] = acc
    return out


def _run(fn, *args):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return _canon(ftz(fn(*[ftz(a) for a in args])))


def add(a, b):
    return _run(lambda x, y: (x + y).astype(np.float32), a, b)


def sub(a, b):
    return _run(lambda x, y: (x - y).astype(np.float32), a, b)


def mul(a, b):
    return _run(lambda x, y: (x * y).astype(np.float32), a, b)


def gt(a, b):
    """a > b in the total order of flushed values (-0 < +0; the canonical NaN is largest, a NaN
    with the sign set smallest: raw sign-magnitude bits, as the RTL's fp_gt)."""
    return _key(a) > _key(b)


def _key(x) -> np.ndarray:
    """Total-order key of flushed fp32 bits: -NaN < -inf < ... < -0 < +0 < ... < +inf < NaN."""
    u = bits(ftz(x)).astype(np.uint32)
    return np.where(u >> 31, ~u, u | np.uint32(0x80000000)).astype(np.uint32)


def chain_max(a) -> np.ndarray:
    """Row maxima of a [rows, cols] under the total order of gt() (any order of evaluation
    gives the same bits; the hardware reduces in a tree)."""
    a = ftz(a)
    idx = np.argmax(_key(a), axis=1)
    return a[np.arange(a.shape[0]), idx].astype(np.float32)


RED_PARTIALS = 64      # VOP RSUM/RSSQ: interleaved partial sums, then a pairwise tree
MM_PARTIALS = 4        # MM: interleaved partial sums over the K blocks, then a pairwise tree


def interleaved_sum(a, P: int, pad: float = 0.0) -> np.ndarray:
    """Row sums of a [rows, cols] as the hardware defines them (docs/isa.md, "Sums"):
    partial p = +0 + a[:, p] + a[:, p+P] + ... (left to right), then a folding tree over the
    P partials: while n > 1: n /= 2; x[i] = x[i] + x[i+n] for i < n. P is a power of two.
    Rows are padded to a multiple of P with `pad` terms: +0 for the VOP sums (the VPU adds
    them: a -0 partial, from a sum that flushed, becomes +0), -0 for MM, which adds none
    (x + -0 = x for every x, +-0 included)."""
    a = ftz(a)
    R, C = a.shape
    n = -(-C // P) * P
    if n != C:
        a = np.concatenate([a, np.full((R, n - C), pad, np.float32)], axis=1)
    part = chain_sum(a.reshape(R, n // P, P).transpose(0, 2, 1).reshape(R * P, n // P))
    x = part.reshape(R, P)
    while x.shape[1] > 1:
        n = x.shape[1] // 2
        x = add(x[:, :n], x[:, n:])
    return x[:, 0]


def fmax(a, b):
    a, b = ftz(a), ftz(b)
    return np.where(gt(a, b), a, b).astype(np.float32)


def fmin(a, b):
    a, b = ftz(a), ftz(b)
    return np.where(gt(b, a), a, b).astype(np.float32)


def fabs(a):
    return from_bits(bits(ftz(a)) & np.uint32(0x7FFFFFFF)).copy()


def i2f(i):
    """int -> fp32 with RNE (numpy's conversion from int64 is correctly rounded)."""
    return np.asarray(i, dtype=np.int64).astype(np.float32)


def q8(x):
    """Round half to even, saturate to [-127, 127], as int8; NaN -> 0 (quantize's 0 * inf: a zero
    in a group whose amax is so small that inv = 127 * recip(amax) is +inf)."""
    x = ftz(x)
    r = np.rint(np.where(np.isnan(x), F32(0), x).astype(np.float64))
    return np.clip(r, -127, 127).astype(np.int8)


def exp2(x):
    # the range flags look at x after the chains' first op (x*1 + -0, or x - y for EXP2SUB),
    # which makes any NaN the canonical one: exp2(NaN) = +inf
    x = _canon(ftz(x))
    lo, hi = gt(F32(-126), x), ~gt(F32(128), x)           # x < -126; x >= 128 (or NaN)
    xf = np.where(~lo & ~hi, x, F32(0)).astype(np.float32)
    i = np.floor(xf.astype(np.float64)).astype(np.int64)
    f = sub(xf, i2f(i))
    p = np.broadcast_to(EXP2_COEFFS[7], f.shape).astype(np.float32)
    for c in reversed(EXP2_COEFFS[:7]):
        p = add(mul(p, f), c)
    pb = bits(p).astype(np.int64) + (i << 23)
    res = from_bits(pb.astype(np.uint32)).copy()
    res = np.where(lo, F32(0), res)
    res = np.where(hi, F32(np.inf), res)
    return res.astype(np.float32)


def log2(x):
    """log2(x) (docs/isa.md): x = 2^e * m with m in [sqrt(1/2), sqrt(2)), t = m - 1 (exact),
    q = C9, q = q*t + Ck for k = 8..1, result = q*t + i2f(e) (each step a mul then an add).
    +-0 -> -inf, x < 0 -> NaN, +inf -> +inf."""
    x = ftz(x)
    b = bits(x).astype(np.int64)
    frac, ex = b & 0x7FFFFF, (b >> 23) & 0xFF
    ge = frac >= LOG2_SQRT2
    e = np.where(ex == 0, 0, ex - 127 + ge)
    m = from_bits((np.where(ge, 126, 127) << 23 | frac).astype(np.uint32))
    t = add(m, F32(-1))
    q = np.broadcast_to(LOG2_COEFFS[8], t.shape).astype(np.float32)
    for c in reversed(LOG2_COEFFS[:8]):
        q = add(mul(q, t), c)
    r = add(mul(q, t), i2f(e))
    zero = (b & 0x7FFFFFFF) == 0
    r = np.where(ex == 255, _canon(x), r)                       # +inf; NaN: the canonical one
    r = np.where((b >> 31).astype(bool) & ~zero, from_bits(np.uint32(0x7FC00000)), r)
    r = np.where(zero, F32(-np.inf), r)
    return ftz(r)


def rdot(a, b):
    """VOP RDOT row sums: isum_64 of the rounded products a*b."""
    return interleaved_sum(mul(a, b), RED_PARTIALS)


def outer(a, d, b, c):
    """VOP OUTER: a*d + b*c, two rounded products and one rounded add."""
    return add(mul(a, d), mul(b, c))


def _exp_add(x, k):
    """x with k added to its exponent field (x * 2^k, exact while the field stays in 1..254)."""
    return from_bits((bits(x).astype(np.int64) + (np.asarray(k, np.int64) << 23)).astype(np.uint32))


def recip(x):
    """docs/isa.md recip: |x| with exponent field >= 250 runs on |x|/16, then the result /16
    (the Newton steps are exact under that scaling; unscaled, the seed flushes for
    |x| > 0x7E7311C3). |x| >= 2^126 (and inf / NaN bits) returns a zero with x's sign."""
    x = ftz(x)
    xb = bits(x)
    ab = xb & np.uint32(0x7FFFFFFF)
    zero = ab == 0
    big = ab >= np.uint32(0x7E800000)                     # 1/|x| <= 2^-126 flushes (incl. inf)
    sc = (ab >= np.uint32(250 << 23)) & ~big
    ax = _exp_add(from_bits(np.where(zero, bits(F32(1)), ab)), np.where(sc, -4, 0))
    seed = (np.uint64(RECIP_MAGIC) - bits(ax).astype(np.uint64)) & np.uint64(0xFFFFFFFF)
    y = ftz(from_bits(seed.astype(np.uint32)))
    for _ in range(3):
        y = mul(y, sub(F32(2), mul(ax, y)))
    y = np.where(big, F32(0), _exp_add(y, np.where(sc & ~big, -4, 0)))
    y = from_bits(bits(y) | (xb & np.uint32(0x80000000)))  # x's sign (also on a NaN's zero)
    return np.where(zero, F32(0), y).astype(np.float32)


def rsqrt(x):
    """docs/isa.md rsqrt: x <= 0 (any sign bit), +0 and +inf return +0. Exponent field <= 2 runs
    on 16x, then the result * 4; field 250..254 on x/16, then / 4 (unscaled, h = x/2 flushes
    at field 1 and y*y at field >= 252; the steps are exact under the scaling)."""
    x = ftz(x)
    b = bits(x)
    f = (b >> np.uint32(23)) & np.uint32(0xFF)
    bad = (b >> np.uint32(31) == 1) | (b == 0) | (b == np.uint32(0x7F800000))
    k = np.where(bad, 0, np.where(f <= 2, 4, np.where((f >= 250) & (f < 255), -4, 0)))
    xs = np.where(bad, F32(1), _exp_add(x, k)).astype(np.float32)
    y = from_bits(np.uint32(RSQRT_MAGIC) - (bits(xs) >> np.uint32(1))).copy()
    h = mul(F32(0.5), xs)
    for _ in range(3):
        y = mul(y, sub(F32(1.5), mul(h, mul(y, y))))
    return np.where(bad, F32(0), _exp_add(y, k // 2)).astype(np.float32)


def quantize(x, axis=-1):
    """Quantize groups along `axis` (the whole axis is one group). Returns (int8 q, fp32 scale).

    Follows docs/isa.md exactly: amax -> s = amax*inv127, inv = 127*recip(amax), q = q8(x*inv).
    """
    x = ftz(x)
    # the largest |x| by its bits (the RTL's unsigned compare of x[30:0]: a NaN is above +inf)
    amax = from_bits(np.max(bits(fabs(x)), axis=axis, keepdims=True))
    zero = amax == 0
    s = np.where(zero, F32(0), mul(amax, INV127)).astype(np.float32)
    inv = np.where(zero, F32(0), mul(F32(127), recip(amax))).astype(np.float32)
    q = q8(mul(x, inv))
    return q, np.squeeze(s, axis=axis)
