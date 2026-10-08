"""The quantization path's corners on the RTL against the ISA simulator (docs/isa.md, "MM",
"Weight formats", "QACT", "QST"): every 4-bit code under every sub-block multiplier, with and
without column reuse; the scale words' corners (4-bit bf16 scales and int8 fp32 scales:
negative, signed zeros, denormals, the largest finite, inf, NaN) and int8's -128, which the
host's quantizers never write and the fuzzers never draw; and the quantizer's rounding ties
(q8 rounds x * inv half to even and saturates at +-127), through QST (blocks and rows) and
QACT (blocks and rows, read back by an MM with identity weights). Each runs the same program on
both and compares DRAM and TMEM bit for bit."""
import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I, quant as Q, rtlsim
from opentpu.isasim import Machine

# bf16 scale corners (the top 16 bits of an fp32): -1.5, +0, -0, the smallest and largest
# denormals, the smallest normal, the largest finite, +inf, NaN
BF16_CORNERS = [0xBFC0, 0x0000, 0x8000, 0x0001, 0x007F, 0x0080, 0x7F7F, 0x7F80, 0x7FC0]
F32_CORNERS = [0xBF400000, 0x00000000, 0x80000000, 0x00000001, 0x007FFFFF, 0x00800000,
               0x7F7FFFFF, 0x7F800000, 0x7FC00000]


def _both(cfg, prog, dram):
    m = Machine(cfg, [prog], [dram.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [dram.copy()])
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, (f"{len(bad)} TMEM words differ, first {bad[:6]}: RTL "
                           f"{[hex(int(v)) for v in tmems[0][bad[:6]]]} ISA "
                           f"{[hex(int(v)) for v in m.slices[0].tmem[bad[:6]]]}")
    assert np.array_equal(drams[0], m.slices[0].dram)
    return m


def _put(dram, at, a):
    b = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
    dram[at:at + b.size] = b


X, WA, SA, OUT = 0, 0x40000, 0x80000, 0x8000


@pytest.mark.parametrize("fmt,D,pair", [("int4", 32, False), ("fp4", 32, False),
                                        ("int4", 128, True), ("fp4", 128, True),
                                        ("fp4", 128, False)])
def test_mm_4bit_codes_multipliers_and_scale_corners_rtl(have_verilator, fmt, D, pair):
    """Rows 0..15: random positive and negative normal bf16 scales; every code at every
    position class under every multiplier 0..15 (block k of row n: code (i + n + k) % 16 at
    element i, multiplier m_b = (4 (n KB + k) + b) % 16). Rows 16..: one scale corner each, in
    all its blocks. KB = 3 (an odd count: PAIR's last chunk has one block)."""
    cfg = Config(S=1, D=D, MCOLS=4, PAIR=pair, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(D + pair)
    KB, M = 3, 2 if pair else 4
    N = 16 + len(BF16_CORNERS)
    x = rng.standard_normal((M, KB * D)).astype(np.float32)
    i = np.arange(D)
    codes = np.array([[(i + n + k) % 16 for k in range(KB)] for n in range(N)]).reshape(N, -1)
    rb = -(-KB // 2) * D                                    # the row: whole D-byte chunks
    rows = np.zeros((N, rb), np.uint8)
    rows[:, :KB * D // 2] = Q.pack4(codes)
    s = rng.uniform(2.0 ** -10, 2.0 ** -3, (16, KB)).astype(np.float32)
    s[rng.random((16, KB)) < 0.3] *= -1
    sb = np.concatenate([F.bits(s) >> 16, np.repeat(np.uint32(BF16_CORNERS)[:, None], KB, 1)])
    mb = (4 * (np.arange(N)[:, None, None] * KB + np.arange(KB)[None, :, None])
          + np.arange(4)) % 16
    sh = 16 + 4 * np.arange(4, dtype=np.uint32)
    words = (sb.astype(np.uint32) | (mb.astype(np.uint32) << sh).sum(-1)).astype(np.uint32)
    srs = 8 * -(-KB // 2)                                   # PAIR: 8-byte aligned pairs
    wpad = np.zeros((N, srs // 4), np.uint32)
    wpad[:, :KB] = words
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    _put(dram, WA, rows)
    _put(dram, SA, wpad)
    wf = Q.mxu_wf(fmt)
    prog = [I.ld(X, 0, x.size), I.qact(0, M, 0, KB, KB * D, dup=pair),
            I.mm(WA, SA, OUT, N, KB, rb, N, M, 0, srs, wf=wf, pair=pair),
            I.mm(WA, SA, OUT + 2 * M * N, N, KB, rb, N, M, 0, srs, wf=wf, pair=pair, unit=True),
            I.halt()]
    m = _both(cfg, prog, dram)
    # the finite rows against float64 math on the dequantized weights (the ISA's own check)
    got = m.slices[0].tget(OUT + np.arange(M * N)).reshape(M, N)[:, :16].astype(np.float64)
    q, sx = F.quantize(x.reshape(M, KB, D), axis=2)
    xa = (q.astype(np.float64) * sx[..., None]).reshape(M, -1)
    want = xa @ Q.dequantize_w4(rows[:16, :KB * D // 2], words[:16], fmt, D).T
    assert np.allclose(got, want, rtol=1e-5, atol=1e-6 * np.abs(want).max())


def test_mm_int8_minus_128_and_scale_corners_rtl(have_verilator):
    """int8 weights of -128 (a code the host never writes) beside random ones, and every fp32
    scale corner, one per row; UNIT and with scales."""
    cfg = Config(S=1, D=32, MCOLS=4, DRAM_BYTES=1 << 20)
    D, KB, M = cfg.D, 3, 4
    rng = np.random.default_rng(7)
    N = 8 + len(F32_CORNERS)
    x = rng.standard_normal((M, KB * D)).astype(np.float32)
    w = rng.integers(-128, 128, (N, KB * D)).astype(np.int8)
    w[:, ::5] = -128
    s = F.from_bits(np.concatenate([F.bits(rng.uniform(0.01, 0.1, (8, KB)).astype(np.float32)),
                                    np.repeat(np.uint32(F32_CORNERS)[:, None], KB, 1)]))
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    _put(dram, WA, w)
    _put(dram, SA, s)
    prog = [I.ld(X, 0, x.size), I.qact(0, M, 0, KB, KB * D),
            I.mm(WA, SA, OUT, N, KB, KB * D, N, M, 0, 4 * KB),
            I.mm(WA, SA, OUT + 2 * M * N, N, KB, KB * D, N, M, 0, 4 * KB, unit=True),
            I.halt()]
    _both(cfg, prog, dram)


def _ties(amax, rng):
    """Values x with F.mul(x, inv) exactly k + 0.5 (inv = 127 * recip(amax), the quantizer's)
    for the k where such an x <= amax exists, their 1-ulp neighbours, and +-0, shuffled."""
    inv = F.mul(F.f32(127), F.recip(F.f32(amax)))
    out = []
    for k in range(-128, 128):
        t = F.f32(k + 0.5)
        x0 = F.f32(np.float64(t) / np.float64(inv))
        for d in range(-4, 5):
            x = F.from_bits(np.uint32(int(F.bits(x0)) + d))
            if F.mul(x, inv) == t:
                out += [x, F.from_bits(F.bits(x) + np.uint32(1)),
                        F.from_bits(F.bits(x) - np.uint32(1))]
                break
    out = np.array(out + [F.f32(v) for v in (0.0, -0.0)], np.float32)
    out = out[np.abs(out) <= amax]
    return out[rng.permutation(len(out))]


def _tie_rows(D, rng, row: bool, nrows=8, KB=6):
    """Rows of KB blocks: the group's amax (each block's, or with `row` the row's) at element
    0 of each block, with either sign, and ties of that amax elsewhere. Returns x and how many
    of its elements are exact ties."""
    x = np.zeros((nrows, KB * D), np.float32)
    amaxes = F.f32(rng.uniform(0.5, 3.0, (nrows, KB)))
    amaxes[0, :] = F.f32([1.0, 2.0, 0.75, 127.0, 3.0, 1.5])
    if row:
        amaxes[:] = amaxes[:, :1]
    for r in range(nrows):
        for k in range(KB):
            blk = np.resize(_ties(amaxes[r, k], rng), D)
            blk[0] = amaxes[r, k] * (1 if (r + k) % 2 else -1)
            x[r, k * D:(k + 1) * D] = blk
    a = np.abs(x).reshape(nrows, KB, D).max(-1, keepdims=True)
    p = F.mul(x.reshape(nrows, KB, D), F.mul(F.f32(127), F.recip(a)))
    return x, int((p - np.floor(p) == 0.5).sum())


@pytest.mark.parametrize("D,row", [(32, False), (32, True), (128, False), (128, True)])
def test_quantizer_ties_qst_rtl(have_verilator, D, row):
    """QST (blocks, rows) on exact ties of x * inv and their neighbours: the bytes and scales
    land in DRAM."""
    cfg = Config(S=1, D=D, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(D + row)
    x, ties = _tie_rows(D, rng, row)
    assert ties > 100
    R, KB = x.shape[0], x.shape[1] // D
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    prog = [I.ld(X, 0, x.size),
            I.qst(0, 0x40000, 0x48000, R, KB, KB * D, KB * D, 1, row=row),
            I.qst(0, 0x50000, 0x58000, R, KB, KB * D, 2 * KB * D, 2, row=row),
            I.halt()]
    _both(cfg, prog, dram)


@pytest.mark.parametrize("row", [False, True])
def test_quantizer_ties_qact_rtl(have_verilator, row):
    """QACT (blocks, rows) on the same ties, read back by MMs with identity int8 weights and
    UNIT scales: T[out + j, n] = i2f(q[j][n]) * s[j] for each block."""
    cfg = Config(S=1, D=32, MCOLS=8, DRAM_BYTES=1 << 20)
    D = cfg.D
    rng = np.random.default_rng(5 + row)
    x, ties = _tie_rows(D, rng, row)
    assert ties > 100
    R, KB = x.shape[0], x.shape[1] // D
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    _put(dram, WA, np.eye(D, dtype=np.int8))
    prog = [I.ld(X, 0, x.size), I.qact(0, R, 0, KB, KB * D, row=row)]
    for k in range(KB):
        prog.append(I.mm(WA, 0, OUT + k * R * D, D, 1, D, D, R, k, 0, unit=True))
    prog.append(I.halt())
    m = _both(cfg, prog, dram)
    got = m.slices[0].tget(OUT + np.arange(KB * R * D)).reshape(KB, R, D)
    q, s = F.quantize(x if row else x.reshape(R, KB, D), axis=-1 if row else 2)
    q = q.reshape(R, KB, D)
    s = np.repeat(s[:, None], KB, 1) if row else s
    assert np.array_equal(got, F.mul(F.i2f(q), s[..., None]).transpose(1, 0, 2))
