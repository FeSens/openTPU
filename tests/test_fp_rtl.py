"""The fp32 corners on the RTL against the ISA simulator, through the units that compute them:
the VPU's composite functions on their scaled ranges and on NaN words (the long lanes' slot
programs and SE's otpu_se_comp), the quantizer's recip on a huge amax and its 0 * inf on a tiny
one (QST: the bytes and scales land in DRAM), and an MM whose partial sums flush to -0 (no pad
terms in the MXU's isum_4). Each runs the same program on both and compares DRAM and TMEM."""
import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I, rtlsim
from opentpu.isasim import Machine


def _both(cfg, prog, dram):
    m = Machine(cfg, [prog], [dram.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [dram.copy()])
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, (f"{len(bad)} TMEM words differ, first {bad[:6]}: RTL "
                           f"{[hex(int(v)) for v in tmems[0][bad[:6]]]} ISA "
                           f"{[hex(int(v)) for v in m.slices[0].tmem[bad[:6]]]}")
    assert np.array_equal(drams[0], m.slices[0].dram)
    return m


def _corner_words(rng, n):
    """n words: rsqrt's scaled ranges (exponent fields 0..5 and 246..255, both signs), random
    words of every field, every power of two and its neighbours (recip's significand edges,
    rsqrt's powers of four), exp2 near its range limits and where x + 1 rounds to 1, NaNs."""
    p = np.arange(1, 255, dtype=np.int64) << 23
    p = np.concatenate([p, p - 1, p + 1])
    k = np.float32([-127, -126, -125, -1, 0, 1, 126, 127, 128])
    t = np.exp2(-np.arange(20, 31))
    ex = np.concatenate([np.nextafter(k, np.float32(-np.inf)), np.nextafter(k, np.float32(np.inf)),
                         -t, t]).astype(np.float32)
    extra = np.concatenate([
        np.uint32([0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFBFFFFF, 0x7F800000, 0xFF800000,
                   0x7E800000, 0x7E7311C3, 0x7E7311C4, 0x7E7FFFFF, 0x00800000, 0x00FFFFFF,
                   0x3F800000]),
        np.concatenate([p, p | (1 << 31)]).astype(np.uint32), ex.view(np.uint32)])
    n = n - len(extra)
    e = np.where(rng.random(n) < 0.5, rng.choice(list(range(6)) + list(range(246, 255)), n),
                 rng.integers(1, 255, n)).astype(np.uint32)
    m = rng.integers(0, 1 << 23, n, dtype=np.uint32)
    m[rng.random(n) < 0.1] = 0
    m[rng.random(n) < 0.1] = 0x7FFFFF
    s = rng.integers(0, 2, n, dtype=np.uint32)
    w = (s << 31) | (e << 23) | m
    return F.from_bits(np.concatenate([w, extra])).copy()


@pytest.mark.parametrize("lanes,dstep", [(8, True), (8, False), (4, False), (16, False)])
def test_composites_on_their_scaled_ranges_rtl(have_verilator, lanes, dstep):
    """RECIP / RSQRT / EXP2 / EXP2SUB / LOG2 / MAX / ABS / COPY on the corner words: dstep with
    8 lanes is otpu_se_comp (the board's), the others the long lanes' slot programs."""
    cfg = Config(S=1, LANES=lanes, MCOLS=min(8, lanes), DSTEP=dstep, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(11 + lanes)
    cols = 64
    x = _corner_words(rng, 40 * cols)
    n = len(x)
    rows = n // cols
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    dram[:4 * n] = x.view(np.uint8)
    prog = [I.ld(0, 0, n)]
    out = n
    for func in (I.V_RECIP, I.V_RSQRT, I.V_EXP2, I.V_LOG2, I.V_ABS, I.V_COPY):
        prog.append(I.vop(func, out, 0, 0, rows, cols, cols, cols, 0))
        out += n
    for func in (I.V_EXP2SUB, I.V_MAX, I.V_MIN):              # x op x[reversed rows]
        prog.append(I.vop(func, out, 0, cols, rows - 1, cols, cols, cols, cols))
        out += n
    prog.append(I.halt())
    _both(cfg, prog, dram)


def test_quantizer_huge_and_tiny_amax_rtl(have_verilator):
    """QST (row groups and blocks): an amax in [2^123, 2^126) (once recip's scaled range), at
    2^126 (inv = 0), tiny ones (inv = 127 * recip(amax) is +inf: the zeros give 0 * inf = NaN,
    q 0) and an inf element (inf * 0)."""
    cfg = Config(S=1, DRAM_BYTES=1 << 20)
    D, KB = cfg.D, 2
    rng = np.random.default_rng(3)
    rows = []
    for amax in (2.0 ** 123, 1.5 * 2.0 ** 125, 0x7E7311C4, 2.0 ** 126, 2.0 ** -126, 1e-37,
                 3.7e-37, 3.8e-37, np.inf, 1.0):
        a = F.from_bits(np.uint32(amax)) if isinstance(amax, int) else F.f32(amax)
        r = (rng.standard_normal(KB * D) * 0.3).astype(np.float32) * a
        r[rng.random(KB * D) < 0.4] = 0.0
        r[rng.random(KB * D) < 0.1] = -0.0
        r[0] = a
        rows.append(np.where(np.isfinite(r) | np.isinf(a), r, 0).astype(np.float32))
    x = np.stack(rows)
    R = len(rows)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    dram[:4 * x.size] = x.view(np.uint8).reshape(-1)
    prog = [I.ld(0, 0, x.size),
            I.qst(0, 0x40000, 0x48000, R, KB, KB * D, KB * D, 1),                 # blocks
            I.qst(0, 0x50000, 0x58000, R, KB, KB * D, KB * D, 1, row=True),       # rows
            I.halt()]
    _both(cfg, prog, dram)


def test_quantizer_recip_every_exponent_rtl(have_verilator):
    """QST (blocks and rows): an amax at every power of two (recip on the significand 1.0, the
    result's exponent field 127 - field away), its neighbours, random ones of every field, the
    largest below 2^126, and amax across the point where inv = 127 * recip(amax) overflows to
    +inf (otpu_qscale scales 127 * y after the multiply and saturates the field there)."""
    cfg = Config(S=1, DRAM_BYTES=1 << 20)
    D, KB = cfg.D, 2
    rng = np.random.default_rng(4)
    p = np.arange(1, 253, dtype=np.int64) << 23
    am = np.concatenate([p, p[1:] - 1, p + 1, p | rng.integers(0, 1 << 23, len(p)),
                         (5 << 23) + np.arange(8257526, 8257546), [0x7E7FFFFF, 0x7E800000]])
    am = F.from_bits(am.astype(np.uint32))
    am = np.resize(am, -(-len(am) // KB) * KB)
    R = len(am) // KB
    r = (rng.standard_normal((R * KB, D)) * 0.3).astype(np.float32) * am[:, None]
    r[rng.random(r.shape) < 0.3] = 0.0
    r[:, 0] = am
    x = r.reshape(R, KB * D)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    dram[:4 * x.size] = x.view(np.uint8).reshape(-1)
    prog = [I.ld(0, 0, x.size),
            I.qst(0, 0x40000, 0x60000, R, KB, KB * D, KB * D, 1),                 # blocks
            I.qst(0, 0x70000, 0x90000, R, KB, KB * D, KB * D, 1, row=True),       # rows
            I.halt()]
    _both(cfg, prog, dram)


def test_mm_partials_flushed_to_minus_zero_rtl(have_verilator):
    """KB = 10: each block's term t_k = (i2f(127) * ws_k) * (1/127); blocks 0..3 and 4..7 are
    -(2^-126 + 3 ulp) and 2^-126, so every partial of two real terms flushes to -0, and t8, t9
    are -0. The MXU adds no pad terms: the row sum is -0 (a +0 pad would make it +0)."""
    cfg = Config(S=1, DRAM_BYTES=1 << 20)
    D, KB = cfg.D, 10
    s = F.mul(F.f32(1.0), F.INV127)
    t_of = lambda ws: F.mul(F.mul(F.f32(127.0), F.f32(ws)), s)
    ws_b = F.from_bits(np.uint32(0x00800000))
    while F.bits(t_of(ws_b)) < 0x00800000:
        ws_b = F.from_bits(F.bits(ws_b) + np.uint32(1))
    ws_a = F.from_bits((F.bits(ws_b) + np.uint32(3)) | np.uint32(0x80000000))
    ws = np.array([ws_a] * 4 + [ws_b] * 4 + [F.f32(-0.0)] * 2, np.float32)
    x = np.zeros(KB * D, np.float32)
    x[np.arange(KB) * D] = 1.0
    W = np.zeros(KB * D, np.int8)
    W[np.arange(KB) * D] = 1
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    dram[:4 * KB * D] = x.view(np.uint8)
    dram[0x10000:0x10000 + KB * D] = W.view(np.uint8)
    dram[0x20000:0x20000 + 4 * KB] = ws.view(np.uint8)
    out = 2 * KB * D
    m = _both(cfg, [I.ld(0, 0, KB * D), I.qact(0, 1, 0, KB, KB * D),
                    I.mm(0x10000, 0x20000, out, 1, KB, KB * D, 1, 1, 0, 4 * KB), I.halt()], dram)
    assert int(m.slices[0].tmem[out].view(np.uint32)) == 0x80000000
