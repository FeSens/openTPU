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
    return np.concatenate(rows)


def test_model_accuracy():
    x = F.f32(np.linspace(-30, 30, 10001))
    assert np.max(np.abs(F.exp2(x) / np.exp2(x.astype(np.float64)) - 1)) < 2e-6
    p = F.f32(np.exp(rng.uniform(-40, 40, 10000)))
    assert np.max(np.abs(F.recip(p) * p.astype(np.float64) - 1)) < 1e-6
    assert np.max(np.abs(F.rsqrt(p) * np.sqrt(p.astype(np.float64)) - 1)) < 1e-6
    q, s = F.quantize(F.f32(rng.standard_normal((16, 32))), axis=1)
    assert np.all(np.abs(q) <= 127) and np.all(s > 0)


def test_rtl_fp_bit_exact(tmp_path):
    vec = _vectors()
    f = tmp_path / "vec.txt"
    np.savetxt(f, vec, fmt="%08x")
    out = rtlsim.run_fp_vectors(f)
    m = re.search(r"FPTEST cases=(\d+) errors=(\d+)", out)
    assert m, out
    assert int(m.group(1)) == len(vec)
    assert int(m.group(2)) == 0, out
