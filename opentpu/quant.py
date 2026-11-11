"""Weight quantization formats: reference quantizers (numpy) for 8-bit and 4-bit weights.

Every quantizer takes an fp32 matrix W [N, K] (rows = output features, K = the inner dimension
the MXU reduces over) and quantizes blocks of `g` consecutive elements of a row, like the MXU's
per-(row, D-block) scales. It returns a `Quantized`: the element codes, the block scales and
the dequantized matrix (what the model computes with), plus the storage cost in bits per
weight (elements + block scales; a per-tensor scale is negligible and not counted).

Formats (the survey; the MXU's own formats are below, "the MXU's formats")
  int8      the MXU's format before MX: int8 in [-127, 127], one fp32 scale per 128 (g = D),
            bit-identical to that hardware quantizer (fp32.quantize)
  int4      symmetric int4 in [-7, 7], one scale per g (g = 32, 64, 128); the scale is stored
            as fp32, fp16, e4m3 (FP8, with a per-tensor fp32 scale as in NVFP4) or e8m0 (a
            power of two)
  mxfp4     OCP Microscaling MXFP4: E2M1 elements, blocks of 32, a shared E8M0 (power-of-two)
            scale. `rule="ocp"` is the spec's scale choice 2^(floor(log2 amax) - 2), which clips
            the block maximum when amax >= 6 * 2^e; `rule="ceil"` picks the smallest power of
            two that does not clip
  nvfp4     NVIDIA NVFP4: E2M1 elements, blocks of 16, an E4M3 (FP8) block scale and an fp32
            per-tensor scale (amax_tensor / (6 * 448))
  e2m1      E2M1 elements with other block sizes / scale types (the general form of the two)
  int4k, e2m1k  two-level (k-quant style) scales that fit one 32-bit word per 128 elements: a
            bf16 scale per 128 and an unsigned 4-bit multiplier m in [1, 15] per 32, so the
            sub-block scale is bf16 * m (the elements are int4 or E2M1; the MXU's fp4 before MX)
  mx-int8, mx-int4, mx-fp4, mx-nvfp4  the MX MXU's formats (quantize_mxu: int8x, int4x,
            fp4x6, nvfp4), their bits the element and scale-row bytes

`search=True` replaces the round-to-nearest scale (amax / qmax) by the candidate that minimizes
the block's squared error (a small grid of scales below and around it, each rounded to the
scale format), the cheap stand-in for GPTQ/AWQ-style error-minimizing quantization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# E2M1 (FP4): 1 sign, 2 exponent, 1 mantissa bit, no inf/NaN. Codes 0..7 are the magnitudes.
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
E2M1_MAX = 6.0
E4M3_MAX = 448.0


def _e4m3_table() -> np.ndarray:
    """The 127 non-negative finite E4M3 values (code 0..126; 127 is NaN), ascending."""
    v = []
    for c in range(127):
        e, m = c >> 3, c & 7
        v.append(m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
    return np.array(v, np.float64)


E4M3 = _e4m3_table()


def round_to_grid(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Round |x| to the nearest value of an ascending non-negative grid (ties to the even code,
    as IEEE round-to-nearest-even: the even code has the even mantissa bit), saturating at the
    largest; the sign is kept. Returns the codes (index into grid) and the signs."""
    a = np.abs(x)
    i = np.clip(np.searchsorted(grid, a), 1, len(grid) - 1)       # grid[i-1] < a <= grid[i]
    lo, hi = grid[i - 1], grid[i]
    dl, dh = a - lo, hi - a
    code = np.where(dl < dh, i - 1, np.where(dh < dl, i, np.where((i - 1) % 2 == 0, i - 1, i)))
    code = np.where(a >= grid[-1], len(grid) - 1, code)
    return code


def e4m3(x: np.ndarray) -> np.ndarray:
    """Round non-negative values to E4M3 (RNE, saturating at 448)."""
    return E4M3[round_to_grid(np.asarray(x, np.float64), E4M3)]


@dataclass
class Quantized:
    fmt: str
    deq: np.ndarray           # dequantized [N, K] fp32
    codes: np.ndarray         # element codes [N, K] (int8 values, or 4-bit codes)
    scales: np.ndarray        # block scales [N, K // g] (as values, fp64)
    g: int
    bits: float               # storage bits per weight (elements + block scales)


def _blocks(W: np.ndarray, g: int) -> np.ndarray:
    N, K = W.shape
    if K % g:
        raise ValueError(f"inner dimension {K} is not a multiple of the block size {g}")
    return W.reshape(N, K // g, g).astype(np.float64)


SCALE_BITS = {"fp32": 32, "fp16": 16, "bf16": 16, "e4m3": 8, "e8m0": 8}


def _round_scale(s: np.ndarray, kind: str, tensor_scale: float) -> np.ndarray:
    """Round positive block scales to the scale format (e4m3 relative to the tensor scale)."""
    if kind == "fp32":
        return s.astype(np.float32).astype(np.float64)
    if kind == "fp16":
        return np.minimum(s, 65504).astype(np.float16).astype(np.float64)
    if kind == "bf16":
        b = s.astype(np.float32).view(np.uint32)
        b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000               # RNE to bf16
        return b.astype(np.uint32).view(np.float32).astype(np.float64)
    if kind == "e4m3":
        return e4m3(s / tensor_scale) * tensor_scale
    if kind == "e8m0":                                               # round up: never clips
        return np.exp2(np.ceil(np.log2(np.maximum(s, 2.0 ** -127))))
    raise ValueError(kind)


def e2m1_code(x: np.ndarray) -> np.ndarray:
    """E2M1 magnitude code of |x| (0..7): round to nearest, ties to the even code, saturating.
    The same as round_to_grid(x, E2M1), as seven compares (fast on large arrays)."""
    a = np.abs(x)
    return ((a > 0.25).astype(np.int8) + (a >= 0.75) + (a > 1.25) + (a >= 1.75) + (a > 2.5)
            + (a >= 3.5) + (a > 5.0))


def _elem(x: np.ndarray, grid: str) -> np.ndarray:
    """Quantize scaled elements to the element grid; returns the dequantized element values."""
    if grid == "int4":
        return np.clip(np.rint(x), -7, 7)
    if grid == "e2m1":
        return np.sign(x) * E2M1[e2m1_code(x)]
    raise ValueError(grid)


QMAX = {"int4": 7.0, "e2m1": E2M1_MAX}
SEARCH = np.linspace(0.64, 1.12, 13)          # scale candidates, x the round-to-nearest scale


def quant_block(W: np.ndarray, g: int, grid: str, scale: str, search: bool = False,
                rule: str = "rtn") -> tuple[np.ndarray, np.ndarray, float]:
    """Blockwise 4-bit quantization. Returns (deq [N, K], scales [N, K/g], tensor scale)."""
    N, K = W.shape
    ts = 1.0
    if scale == "e4m3":                       # NVFP4's second level: amax -> 448 * qmax
        ts = float(np.abs(W).max()) / (E4M3_MAX * QMAX[grid]) or 1.0
    R = max(1, (1 << 22) // K)                # rows per piece (bounded temporaries)
    parts = [_quant_rows(W[r:r + R], g, grid, scale, search, rule, ts) for r in range(0, N, R)]
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), ts)


def _quant_rows(W, g, grid, scale, search, rule, ts):
    N, K = W.shape
    xb = _blocks(W, g)
    amax = np.abs(xb).max(-1)
    qmax = QMAX[grid]
    if scale == "e8m0" and rule == "ocp":    # OCP MX: 2^(floor(log2 amax) - emax_elem)
        base = np.exp2(np.floor(np.log2(np.maximum(amax, 2.0 ** -127))) - 2)
        cands = [base] + ([base * 2, base / 2] if search else [])
    elif scale == "e8m0":
        base = _round_scale(amax / qmax, "e8m0", ts)
        cands = [base] + ([base / 2] if search else [])
    else:
        mul = SEARCH if search else [1.0]
        cands = [_round_scale(amax * m / qmax, scale, ts) for m in mul]
    best_s = best_q = best_e = None
    for s in cands:
        s = np.where((amax == 0) | (s == 0), 1.0, s)          # a scale that underflowed: all 0
        q = _elem(xb / s[..., None], grid) * s[..., None]
        e = ((q - xb) ** 2).sum(-1)
        if best_e is None:
            best_s, best_q, best_e = s, q, e
        else:
            m = e < best_e
            best_s = np.where(m, s, best_s)
            best_q = np.where(m[..., None], q, best_q)
            best_e = np.where(m, e, best_e)
    return best_q.reshape(N, K).astype(np.float32), best_s


def quant_two(W: np.ndarray, grid: str, G: int = 128, sub: int = 32, search: bool = False):
    """Two-level scales: S (bf16) per G elements, m in [1, 15] per `sub`; returns deq, S, m."""
    N, K = W.shape
    R = max(1, (1 << 22) // K)
    parts = [_two_rows(W[r:r + R], grid, G, sub, search) for r in range(0, N, R)]
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def _two_rows(W, grid, G, sub, search):
    N, K = W.shape
    xb = W.reshape(N, K // G, G // sub, sub).astype(np.float64)
    qmax = QMAX[grid]
    sb = np.abs(xb).max(-1) / qmax                                   # ideal sub-block scales
    S = _round_scale(sb.max(-1) / 15, "bf16", 1.0)                   # [N, K/G]
    S = np.where(S == 0, 1.0, S)
    ms = range(1, 16) if search else [None]
    best_m = best_q = best_e = None
    for m in ms:
        if m is None:
            m = np.clip(np.rint(sb / S[..., None]), 1, 15)
        else:
            m = np.full(sb.shape, float(m))
        s = (S[..., None] * m)[..., None]
        q = _elem(xb / s, grid) * s
        e = ((q - xb) ** 2).sum(-1)
        if best_e is None:
            best_m, best_q, best_e = m, q, e
        else:
            k = e < best_e
            best_m = np.where(k, m, best_m)
            best_q = np.where(k[..., None], q, best_q)
            best_e = np.where(k, e, best_e)
    return best_q.reshape(N, K).astype(np.float32), S, best_m


def quantize(W: np.ndarray, fmt: str, g: int | None = None, scale: str | None = None,
             search: bool = False, rule: str = "rtn") -> Quantized:
    """Quantize-dequantize W [N, K] in one of the formats of the module docstring."""
    W = np.asarray(W, np.float32)
    if fmt.startswith("mx-"):           # the MX MXU's formats (D = 128)
        f, K = fmt[3:], W.shape[1]
        rows, srows = quantize_mxu(W, f)
        deq = dequantize_mxu(rows, srows, f, K).astype(np.float32)
        return Quantized(fmt, deq, elements(rows, f, K), srows, 128,
                         8 * (row_bytes(K, f) + scale_bytes(K, f)) / K)
    if fmt == "int8":                   # an fp32 scale per group of g (the MXU before MX)
        from . import fp32 as F
        g = g or 128
        q, s = F.quantize(W.reshape(W.shape[0], -1, g), axis=2)
        q = q.reshape(W.shape)
        deq = (q.reshape(W.shape[0], -1, g).astype(np.float32) * s[..., None]).reshape(W.shape)
        return Quantized(fmt, deq, q, s.astype(np.float64), g, 8 + 32 / g)
    if fmt == "int4":
        g, scale = g or 32, scale or "fp16"
        deq, s, _ = quant_block(W, g, "int4", scale, search)
    elif fmt == "mxfp4":
        g, scale = g or 32, "e8m0"
        deq, s, _ = quant_block(W, g, "e2m1", scale, search, rule=rule if rule != "rtn" else "ocp")
    elif fmt == "nvfp4":
        g, scale = g or 16, "e4m3"
        deq, s, _ = quant_block(W, g, "e2m1", scale, search)
    elif fmt == "e2m1":
        g, scale = g or 32, scale or "e4m3"
        deq, s, _ = quant_block(W, g, "e2m1", scale, search, rule=rule)
    elif fmt in ("int4k", "e2m1k"):
        deq, S, m = quant_two(W, fmt[:-1], search=search)
        return Quantized(fmt, deq, np.zeros(0), S[..., None] * m, 32, 4 + 32 / 128)
    else:
        raise ValueError(f"unknown format {fmt!r}")
    return Quantized(fmt, deq, np.zeros(0), s, g, 4 + SCALE_BITS[scale] / g)


def rel_err(W: np.ndarray, deq: np.ndarray) -> float:
    """Relative squared reconstruction error ||W - deq||^2 / ||W||^2."""
    W = np.asarray(W, np.float64)
    return float(((deq - W) ** 2).sum() / (W ** 2).sum())


# =============================================================================== the MXU's formats
# The streamed weight formats of MM (docs/isa.md "Weight formats", docs/mx_mxu.md). A weight row
# has an fp32 scale Sw and, under it, power-of-two offsets dw per K-block of D: int8 (int8x, WF 0)
# one dw byte a block; int4 / fp4 (int4x / fp4x6, WF 1 / 2) a word a block of dw and four 6-bit
# multipliers m, one per sub-block of D/4; nvfp4 (WF 3) an E4M3 scale per group of D/8 (its
# mantissa the multiplier, its exponent the offset). Elements: int8, or two 4-bit codes per byte
# (element 2i in the low nibble), one K-block per D/2 bytes. int4 codes are two's complement in
# [-8, 7]; E2M1 codes (fp4, nvfp4) are taken as twice their value, {0, 1, 2, 3, 4, 6, 8, 12} with
# the sign in bit 3, so an fp4 row stores Sw / 2 and an nvfp4 row 16 times its fp32 scale (the
# E4M3 value m 2^(max(E,1) - 10) times 2^-5 Sw for a code twice the E2M1 value).
# The names "int8", "int4", "fp4" are the MX formats (they replaced the per-block fp32 / bf16
# scales of the MXU before docs/mx_mxu.md); every cache keyed on the quantizer holds MX_NUMERICS.
MX_NUMERICS = "mx1"
DEC_INT4 = np.array([0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1], np.int8)
DEC_FP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.int8)
DEC = {"int4": DEC_INT4, "fp4": DEC_FP4, "nvfp4": DEC_FP4}
NSUB = 4                       # sub-blocks per K-block (QACT's offsets, int4 / fp4's multipliers)
NVG = 8                        # nvfp4: E4M3 groups per K-block
MXU_FORMATS = ("int8", "int4", "fp4", "nvfp4")
FOUR = ("int4", "fp4", "nvfp4")
MMAX = 63                      # int4 / fp4: the largest 6-bit multiplier
DW_MAX = 255                   # an offset byte


def unpack4(b: np.ndarray) -> np.ndarray:
    """Bytes [..., n] -> nibble codes [..., 2n], the low nibble first."""
    b = np.asarray(b, np.uint8)
    return np.stack([b & 15, b >> 4], -1).reshape(*b.shape[:-1], 2 * b.shape[-1])


def pack4(codes: np.ndarray) -> np.ndarray:
    """Nibble codes [..., 2n] (0..15) -> bytes [..., n]."""
    c = np.asarray(codes, np.uint8)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)


def row_bytes(K: int, fmt: str, D: int = 128) -> int:
    """DRAM bytes of one streamed row of K elements (4-bit rows padded to whole D-byte chunks)."""
    return K if fmt == "int8" else -(-K // (2 * D)) * D


def scale_bytes(K: int, fmt: str, D: int = 128, rsc: bool = False) -> int:
    """Bytes of one weight row's scale row (its stride srs) for K elements: Sw, then int8 a byte
    a block, int4 / fp4 a reserved word and a word a block, nvfp4 three reserved words and eight
    bytes a block, padded so that every row (and a PAIR's block pair) stays aligned (4, 8, 16
    bytes). rsc: Sw alone (MM RSC: the K cache of one block a token)."""
    if K % D:
        raise ValueError(f"inner dimension {K} is not a multiple of D={D}")
    KB = K // D
    if rsc:
        return 4
    if fmt == "int8":
        return 4 + 4 * -(-KB // 4)
    if fmt in ("int4", "fp4"):
        return 8 + 8 * -(-KB // 2)
    if fmt == "nvfp4":
        return 16 + 16 * -(-KB // 2)
    raise ValueError(f"unknown MXU format {fmt!r}")


def mxu_wf(fmt: str) -> int:
    from .isa import WFORMATS
    return WFORMATS[fmt]


def _int8x(W: np.ndarray, D: int):
    """int8: the hardware quantizer's rows (fp32.quantize_mx in groups of D, as QST writes an
    int8 scale row; docs/isa.md "Quantization"): Sw = amax * f32(1/127) and each block's offset
    d, the largest (<= 31) with the block's amax * 2^d <= the row's. (q int8 [N, K], Sw [N],
    dw [N, KB])"""
    from . import fp32 as F
    return F.quantize_mx(np.asarray(W, np.float32), D)


def _x4(W: np.ndarray, fmt: str, D: int):
    """int4 / fp4 (int4x / fp4x6: docs/mx_mxu.md 3.2's w_fp4x(a, 6)): sub-block scales
    Sw 2^-d m (m in 1..63) with Sw = the row's largest ideal sub-block scale / 63 (fp32), d per
    block the largest power of two that keeps the block's largest ideal sub-block scale within
    63 Sw 2^-d, and m per sub-block the candidate of SEARCH (around the ideal one) with the
    smallest squared error. Returns (codes [N, K] uint8, Sw [N] f32 as the MXU takes it (fp4:
    halved), dw [N, KB] uint8, m [N, KB, NSUB] uint8, deq [N, K] float64)."""
    N, K = W.shape
    KB, sub = K // D, D // NSUB
    grid = "int4" if fmt == "int4" else "e2m1"
    xb = np.asarray(W, np.float64).reshape(N, KB, NSUB, sub)
    sb = np.abs(xb).max(-1) / QMAX[grid]                         # ideal sub-block scales
    top = sb.max(-1).max(-1)                                     # [N]
    Sw = np.where(top == 0, 1.0, top / MMAX).astype(np.float32).astype(np.float64)
    bmax = sb.max(-1)                                            # [N, KB]
    with np.errstate(divide="ignore"):
        d = np.floor(np.log2(MMAX * Sw[:, None] / bmax))
    d = np.where(bmax > 0, np.clip(d, 0, DW_MAX), 0)
    unit = (Sw[:, None] * np.exp2(-d))[..., None]                # [N, KB, 1]
    m0 = sb / unit
    best_m = best_q = best_e = None
    for f in SEARCH:
        m = np.clip(np.rint(m0 * f), 1, MMAX)
        s = (unit * m)[..., None]
        q = _elem(xb / s, grid) * s
        e = ((q - xb) ** 2).sum(-1)
        if best_e is None:
            best_m, best_q, best_e = m, q, e
        else:
            k = e < best_e
            best_m = np.where(k, m, best_m)
            best_q = np.where(k[..., None], q, best_q)
            best_e = np.where(k, e, best_e)
    x = best_q / (unit * best_m)[..., None]                      # grid values (exact)
    if grid == "int4":
        code = np.rint(x).astype(np.int64) & 15
        sw = Sw
    else:
        mag = e2m1_code(x)
        code = mag | np.where(np.signbit(x) & (mag != 0), 8, 0)
        sw = Sw * 0.5                                            # codes are twice the value
    return (code.reshape(N, K).astype(np.uint8), sw.astype(np.float32), d.astype(np.uint8),
            best_m.astype(np.uint8), best_q.reshape(N, K))


def _nv(W: np.ndarray, D: int):
    """nvfp4 with a per-row fp32 scale (docs/mx_mxu.md 3.2's w_nvfp4(a, per_row=True)): E2M1
    elements, an E4M3 scale per group of D/8 relative to ts = the row's amax / (448 * 6) (fp32),
    each picked by SEARCH (squared error). Returns (codes [N, K] uint8, Sw [N] f32 = 16 ts as the
    MXU takes it, the E4M3 bytes [N, KB, NVG] uint8, deq [N, K] float64)."""
    N, K = W.shape
    g = D // NVG
    a = np.asarray(W, np.float64)
    ts = np.abs(a).max(-1, keepdims=True) / (E4M3_MAX * E2M1_MAX)
    ts = np.where(ts == 0, 1.0, ts).astype(np.float32).astype(np.float64)
    xb = a.reshape(N, K // g, g) / ts[..., None]
    amax = np.abs(xb).max(-1)
    best = None
    for f in SEARCH:
        s = e4m3(amax * f / E2M1_MAX)
        s = np.where((amax == 0) | (s == 0), 1.0, s)
        q = np.sign(xb) * E2M1[e2m1_code(xb / s[..., None])] * s[..., None]
        e = ((q - xb) ** 2).sum(-1)
        if best is None:
            best = [s, q, e]
        else:
            k = e < best[2]
            best = [np.where(k, s, best[0]), np.where(k[..., None], q, best[1]),
                    np.where(k, e, best[2])]
    s, q, _ = best
    x = q / s[..., None]
    mag = e2m1_code(x)
    code = (mag | np.where(np.signbit(x) & (mag != 0), 8, 0)).reshape(N, K)
    e4 = round_to_grid(s, E4M3)                                  # exact E4M3 values: their codes
    return (code.astype(np.uint8), (16 * ts[:, 0]).astype(np.float32),
            e4.reshape(N, K // D, NVG).astype(np.uint8), (q * ts[..., None]).reshape(N, K))


def pack_scales(fmt: str, Sw, dw=None, m=None, e4=None) -> np.ndarray:
    """The scale rows [N, scale_bytes] (uint8) of Sw [N] (fp32, as the MXU takes it) and the
    blocks' offsets dw [N, KB] (int8, int4, fp4), multipliers m [N, KB, NSUB] (int4, fp4) or
    E4M3 bytes e4 [N, KB, NVG] (nvfp4)."""
    Sw = np.asarray(Sw, np.float32).reshape(-1)
    N = len(Sw)
    KB = (e4 if fmt == "nvfp4" else dw).shape[1]
    rows = np.zeros((N, scale_bytes(KB, fmt, 1)), np.uint8)
    rows[:, :4] = Sw[:, None].view(np.uint8)
    if fmt == "int8":
        rows[:, 4:4 + KB] = dw
    elif fmt in ("int4", "fp4"):
        word = np.asarray(dw, np.uint32)
        for b in range(NSUB):
            word = word | (np.asarray(m[..., b], np.uint32) << np.uint32(8 + 6 * b))
        rows[:, 8:8 + 4 * KB] = word.astype(np.uint32).view(np.uint8).reshape(N, 4 * KB)
    elif fmt == "nvfp4":
        if np.any(np.asarray(e4) & 0x80):
            raise ValueError("an NVFP4 scale with the sign bit set")
        rows[:, 16:16 + 8 * KB] = np.asarray(e4, np.uint8).reshape(N, 8 * KB)
    else:
        raise ValueError(f"unknown MXU format {fmt!r}")
    return rows


def unpack_scales(srows: np.ndarray, fmt: str, KB: int, k0: int = 0, rsc: bool = False):
    """What MM reads from scale rows [N, srs] (isasim's _mx_scales): Sw [N] fp32, and per (row,
    block k0 .. k0 + KB - 1, group) the offset dw and the multiplier m [N, KB, G] (G = NVG groups
    of D/8 for nvfp4, else the NSUB sub-blocks of D/4)."""
    srows = np.ascontiguousarray(srows, np.uint8)
    N = srows.shape[0]
    G = NVG if fmt == "nvfp4" else NSUB
    Sw = srows[:, :4].copy().view(np.float32)[:, 0]
    if rsc:
        return Sw, np.zeros((N, KB, G), np.int64), np.ones((N, KB, G), np.int64)
    kk = k0 + np.arange(KB)
    if fmt == "int8":
        dw = np.repeat(srows[:, 4 + kk].astype(np.int64)[:, :, None], G, axis=2)
        return Sw, dw, np.ones((N, KB, G), np.int64)
    if fmt in ("int4", "fp4"):
        word = srows[:, 8:].copy().view(np.uint32)[:, kk].astype(np.int64)
        dw = np.repeat((word & 0xFF)[:, :, None], G, axis=2)
        m = np.stack([(word >> (8 + 6 * b)) & 63 for b in range(G)], -1)
        return Sw, dw, m
    e4 = srows[:, 16:].reshape(N, -1, NVG)[:, kk].astype(np.int64)
    E, M = (e4 >> 3) & 15, e4 & 7
    return Sw, 15 - np.maximum(E, 1), np.where(E > 0, 8 + M, M)


def elements(rows: np.ndarray, fmt: str, K: int) -> np.ndarray:
    """The integer elements w [N, K] the MXU multiplies (int8; 4-bit codes decoded)."""
    rows = np.asarray(rows, np.uint8)
    if fmt == "int8":
        return rows[:, :K].view(np.int8)
    return DEC[fmt][unpack4(rows[:, :K // 2])]


def dequantize_mxu(rows: np.ndarray, srows: np.ndarray | None, fmt: str, K: int,
                   D: int = 128, rsc: bool = False) -> np.ndarray:
    """The matrix [N, K] (float64) that MXU rows and their scale rows stand for: w m 2^-dw Sw
    (srows None: UNIT, the elements themselves). A denormal Sw is 0 (the MXU's epilogue
    flushes it)."""
    w = elements(rows, fmt, K).astype(np.float64)
    if srows is None:
        return w
    N, KB = w.shape[0], K // D
    Sw, dw, m = unpack_scales(srows, fmt, KB, rsc=rsc)
    G = dw.shape[2]
    sc = m * np.exp2(-dw.astype(np.float64))                     # [N, KB, G]
    Sw = np.where(np.abs(Sw) < np.float32(2.0 ** -126), 0.0, Sw.astype(np.float64))
    return (w.reshape(N, KB, G, D // G) * sc[..., None]).reshape(N, K) * Sw[:, None]


def group_offsets(srows: np.ndarray, fmt: str, K: int, D: int = 128,
                  rsc: bool = False) -> np.ndarray:
    """Each group of D/8 elements' weight offset dw [N, K / (D/8)] (the MM's drop rule: a group
    whose dw and activation offset add up to more than the window W does not count)."""
    _, dw, _ = unpack_scales(srows, fmt, K // D, rsc=rsc)
    return np.repeat(dw, NVG // dw.shape[2], axis=2).reshape(dw.shape[0], -1)


def quantize_mx(W: np.ndarray, fmt: str, D: int = 128):
    """W [N, K] in an MXU format: (element bytes [N, row_bytes], scale rows [N, scale_bytes]),
    every row on its own."""
    W = np.asarray(W, np.float32)
    N, K = W.shape
    if K % D:
        raise ValueError(f"inner dimension {K} is not a multiple of D={D}")
    if fmt == "int8":
        q, Sw, dw = _int8x(W, D)
        return q.view(np.uint8), pack_scales(fmt, Sw, dw=dw)
    rows = np.zeros((N, row_bytes(K, fmt, D)), np.uint8)
    if fmt in ("int4", "fp4"):
        code, Sw, dw, m, _ = _x4(W, fmt, D)
        rows[:, :K // 2] = pack4(code)
        return rows, pack_scales(fmt, Sw, dw=dw, m=m)
    if fmt == "nvfp4":
        code, Sw, e4, _ = _nv(W, D)
        rows[:, :K // 2] = pack4(code)
        return rows, pack_scales(fmt, Sw, e4=e4)
    raise ValueError(f"unknown MXU format {fmt!r}")


QUANT_ROWS = 1 << 14    # quantize_mxu: rows per piece, so its temporaries stay small (a 248K x
                        # 2048 LM head is 2 GB in fp32; its float temporaries were several more)


def quantize_mxu(W: np.ndarray, fmt: str, D: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """W [N, K] in an MXU weight format ("int8", "int4", "fp4" or "nvfp4"): the DRAM rows
    [N, row_bytes] (uint8) and the scale rows [N, scale_bytes] (uint8), ready to place (rows
    rs and srs bytes apart). Rows are quantized on their own, QUANT_ROWS at a time."""
    N = W.shape[0]
    if N <= QUANT_ROWS:
        return quantize_mx(W, fmt, D)
    q0, s0 = quantize_mx(W[:QUANT_ROWS], fmt, D)
    q = np.empty((N,) + q0.shape[1:], q0.dtype)
    sc = np.empty((N,) + s0.shape[1:], s0.dtype)
    q[:QUANT_ROWS], sc[:QUANT_ROWS] = q0, s0
    for i in range(QUANT_ROWS, N, QUANT_ROWS):
        q[i:i + QUANT_ROWS], sc[i:i + QUANT_ROWS] = quantize_mx(W[i:i + QUANT_ROWS], fmt, D)
    return q, sc


def part_scales(srows: np.ndarray, fmt: str, k0: int, kb: int) -> np.ndarray:
    """The scale rows of blocks k0 .. k0 + kb - 1 of rows with scale rows srows, as a matrix of
    their own (a column part, compiler.QTensor parts): the rows' Sw and those blocks'
    offsets."""
    srows = np.asarray(srows, np.uint8)
    out = np.zeros((srows.shape[0], scale_bytes(kb, fmt, 1)), np.uint8)
    out[:, :4] = srows[:, :4]
    if fmt == "int8":
        out[:, 4:4 + kb] = srows[:, 4 + k0:4 + k0 + kb]
    elif fmt in ("int4", "fp4"):
        out[:, 8:8 + 4 * kb] = srows[:, 8 + 4 * k0:8 + 4 * (k0 + kb)]
    else:
        out[:, 16:16 + 8 * kb] = srows[:, 16 + 8 * k0:16 + 8 * (k0 + kb)]
    return out


def part_rows(rows: np.ndarray, fmt: str, c0: int, c1: int, D: int = 128) -> np.ndarray:
    """Columns c0 .. c1 of MXU element rows as rows of their own (4-bit: c0 a multiple of 2D,
    a whole D-byte chunk)."""
    rows = np.asarray(rows, np.uint8)
    if fmt == "int8":
        return np.ascontiguousarray(rows[:, c0:c1])
    if c0 % (2 * D):
        raise ValueError(f"a 4-bit column part starts on a whole chunk ({c0})")
    out = np.zeros((rows.shape[0], row_bytes(c1 - c0, fmt, D)), np.uint8)
    n = (c1 - c0) // 2
    out[:, :n] = rows[:, c0 // 2:c0 // 2 + n]
    return out


def column_parts(rows: np.ndarray, srows: np.ndarray, fmt: str, K: int, C: int,
                 D: int = 128) -> list:
    """Quantized rows of K columns (quantize_mxu's) as column parts of C: [(element rows, scale
    rows)] of columns j C .. (j + 1) C, each part's scale rows the rows' Sw and its blocks'
    offsets (a matrix stored in parts, compiler.QTensor: the same values as the whole rows)."""
    if K % C:
        raise ValueError(f"{K} columns in parts of {C}")
    return [(part_rows(rows, fmt, j * C, (j + 1) * C, D),
             part_scales(srows, fmt, j * C // D, C // D)) for j in range(K // C)]


def gather_order(K: int, C: int, S: int) -> np.ndarray:
    """The columns of a down projection's K = S F' inputs (each slice's F' in chunks of C) in
    the all-gather's order: chunk c of slice 0, of slice 1, ..., then chunk c + 1
    (kernels.mlp.swiglu_down: each chunk's activations of every slice are one QACT range and
    W_down's part c one MM). S = 1: in order."""
    f = K // S
    if K % S or f % C:
        raise ValueError(f"{K} columns: {S} slices in chunks of {C}")
    return np.arange(K).reshape(S, f // C, C).transpose(1, 0, 2).reshape(-1)


def gathered_rows(rows: np.ndarray, srows: np.ndarray, fmt: str, K: int, C: int, S: int,
                  D: int = 128) -> tuple:
    """Quantized rows of K = S F' columns (quantize_mxu's) with their columns in gather_order:
    whole C-column chunks moved (C whole D-byte chunks of elements: int8 D, 4-bit 2 D), the
    rows' Sw kept and the blocks' offsets moved with them, so the values are the rows'. Stored
    in column parts of S C (column_parts), part c is chunk c of every slice."""
    if S == 1:
        return rows, srows
    f = K // S
    if K % S or f % C or C % (D if fmt == "int8" else 2 * D):
        raise ValueError(f"{K} columns: {S} slices in chunks of {C} ({fmt}, D {D})")
    starts = [t * f + c * C for c in range(f // C) for t in range(S)]
    er = np.concatenate([part_rows(rows, fmt, a, a + C, D) for a in starts], 1)
    hdr, per = (4, 1) if fmt == "int8" else (8, 4) if fmt in ("int4", "fp4") else (16, 8)
    sr = np.array(srows, np.uint8)
    kb = C // D
    sr[:, hdr:hdr + per * (K // D)] = np.concatenate(
        [sr[:, hdr + per * (a // D):hdr + per * (a // D + kb)] for a in starts], 1)
    return er, sr


def gathered_parts(rows: np.ndarray, srows: np.ndarray, fmt: str, K: int, C: int, S: int,
                   D: int = 128) -> list:
    """column_parts of S C of the rows in gather_order (gathered_rows): part c chunk c of
    every slice's F' (S = 1: column parts of C)."""
    return column_parts(*gathered_rows(rows, srows, fmt, K, C, S, D), fmt, K, S * C, D)


def gather_scales(srows: np.ndarray, fmt: str, K: int, D: int = 128) -> np.ndarray:
    """A table's scales as kernels.gather streams its rows (a block, 4-bit a chunk of two blocks,
    a streamed row each): int8 an fp32 per block, Sw 2^-dw [N, KB] (MM RSC, 4 bytes apart);
    4-bit a scale row per chunk (MM k0 0, two blocks): Sw, the reserved words and the chunk's
    two blocks' word (int4, fp4: [N, KB/2, 16]) or bytes (nvfp4: [N, KB/2, 32]), uint8."""
    srows = np.asarray(srows, np.uint8)
    N, KB = srows.shape[0], K // D
    if fmt == "int8":
        Sw = srows[:, :4].copy().view(np.float32)
        return np.ldexp(Sw, -srows[:, 4:4 + KB].astype(np.int32)).astype(np.float32)
    nc = -(-KB // 2)
    hdr, per = (8, 4) if fmt in ("int4", "fp4") else (16, 8)    # header bytes, a block's bytes
    blk = np.zeros((N, 2 * nc, per), np.uint8)
    blk[:, :KB] = srows[:, hdr:hdr + per * KB].reshape(N, KB, per)
    out = np.zeros((N, nc, hdr + 2 * per), np.uint8)
    out[:, :, :4] = srows[:, None, :4]
    out[:, :, hdr:] = blk.reshape(N, nc, 2 * per)
    return out


def gather_scale_bytes(K: int, fmt: str, D: int = 128) -> int:
    """Bytes of one table row's gather_scales."""
    KB = K // D
    if fmt == "int8":
        return 4 * KB
    return -(-KB // 2) * (16 if fmt in ("int4", "fp4") else 32)


# =============================================================================== weight groups
# A matrix in groups (docs/isa.md "Weight groups", MM GRP) on a machine of WG lanes (Config.WG,
# the MXU's columns and port W's channels): a group is R = grp_r(fmt, WG) rows, WG of int8 and
# 2 WG of a 4-bit format, and its chunk c, P = WG x D contiguous bytes at g rs + c P, a lane's D
# bytes apiece: int8 chunk c (K-block c) of row g R + l in lane l; 4-bit K-block c of rows
# g R + 2l and g R + 2l + 1 in lane l, element i of the pair in byte i (row 2l's the low nibble).
# Group g's scale area at g srs: the rows' Sw words, padded to swp(), then per block the R rows'
# block data (an int8 dw byte, a 4-bit word, NVFP4's eight E4M3 bytes). WG = 1 is the natural
# layout (quantize_mxu's rows and scale rows) byte for byte. N is padded to whole groups with zero
# rows. On a machine with the channel hash (Config.CHASH) each group chunk's lanes are stored
# permuted by the chunk's address (hash_rows): lane l of the chunk at A at lane l ^
# chunk_fold(A / P, WG). The functions below take the machine's WG (`wg`) and the format; swp
# and grp_rows take a group's rows.
SCALE_DATA = {"int8": 1, "int4": 4, "fp4": 4, "nvfp4": 8}   # a block's scale data bytes a row


def grp_r(fmt: str, wg: int) -> int:
    """The rows of a group on a machine of `wg` lanes: wg (int8), 2 wg (a 4-bit format: two rows
    a lane); 1 at wg = 1."""
    return wg if wg == 1 or fmt == "int8" else 2 * wg


def swp(fmt: str, R: int, rsc: bool = False) -> int:
    """The Sw part of a scale area of R rows: R words, padded so that a block's entries stay
    aligned: 4 R (int8, RSC); int4 / fp4 8 at R = 1 (PAIR's two words), 4 R above (a block's
    R words); NVFP4 16 at R = 1, 8 R above. At R = 1, the scale row's header."""
    if rsc or fmt == "int8":
        return 4 * R
    if fmt in ("int4", "fp4"):
        return 8 if R == 1 else 4 * R
    return 16 if R == 1 else 8 * R


def grp_rows(N: int, R: int) -> int:
    """Rows of a grouped matrix of N rows in groups of R rows: whole groups."""
    return -(-N // R) * R


def grp_row_bytes(K: int, fmt: str, D: int = 128, wg: int = 1) -> int:
    """The group stride of a grouped [., K] matrix: its chunks, wg x D bytes each (int8: a
    D-block a chunk; 4-bit at wg > 1: a D-block of the group's 2 wg rows a chunk)."""
    if wg == 1 or fmt == "int8":
        return row_bytes(K, fmt, D) * wg
    if K % D:
        raise ValueError(f"inner dimension {K} is not a multiple of D={D}")
    return (K // D) * wg * D


def grp_scale_bytes(K: int, fmt: str, D: int = 128, wg: int = 1, rsc: bool = False) -> int:
    """The group scale stride: the Sw part and K / D blocks' entries, padded to swp (at wg = 1
    scale_bytes)."""
    if K % D:
        raise ValueError(f"inner dimension {K} is not a multiple of D={D}")
    R = grp_r(fmt, wg)
    p = swp(fmt, R, rsc)
    n = p if rsc else p + (K // D) * R * SCALE_DATA[fmt]
    return -(-n // p) * p


def group_rows(rows: np.ndarray, wg: int, D: int = 128, fmt: str = "int8",
               K: int | None = None) -> np.ndarray:
    """Natural element rows [N, nc D] (whole D-byte chunks; 4-bit: K elements, default every
    block of the rows) -> the grouped matrix's bytes [ceil(N / R), grp_row_bytes] (a group a
    row), the padding rows 0."""
    rows = np.asarray(rows, np.uint8)
    N, w = rows.shape
    if w % D:
        raise ValueError(f"rows of {w} bytes are not whole {D}-byte chunks")
    R = grp_r(fmt, wg)
    G, nc = -(-N // R), w // D
    if R == wg:
        out = np.zeros((G * R, w), np.uint8)
        out[:N] = rows
        return np.ascontiguousarray(out.reshape(G, R, nc, D).transpose(0, 2, 1, 3)).reshape(G, -1)
    KB = 2 * nc if K is None else K // D
    if K is not None and (K % D or KB > 2 * nc):
        raise ValueError(f"K {K}: not whole D-blocks of rows of {w} bytes")
    c = np.zeros((G * R, KB * D), np.uint8)
    c[:N] = unpack4(rows)[:, :KB * D]
    c = c.reshape(G, wg, 2, KB, D)                              # [g, lane, row of the pair, k, i]
    out = c[:, :, 0] | (c[:, :, 1] << 4)                        # [g, lane, k, i]
    return np.ascontiguousarray(out.transpose(0, 2, 1, 3)).reshape(G, -1)


def chunk_fold(m, R: int):
    """The channel hash of group chunk index `m` (an int or an integer array): the XOR fold of
    its bits into log2 R bits, bit i into bit i mod log2 R (otpu_native_dram's fold at NCH = R);
    0 at R = 1."""
    m = np.asarray(m, np.int64)
    b = (R - 1).bit_length()
    h = np.zeros_like(m)
    if b:
        while m.any():
            h ^= m & (R - 1)
            m = m >> b
    return h


def hash_rows(data: np.ndarray, at: int, wg: int, D: int = 128) -> np.ndarray:
    """A grouped matrix's bytes (any shape, whole wg x D chunks) as a machine with the channel
    hash stores them at DRAM address `at` (a multiple of P = wg x D; docs/isa.md "Weight
    groups"): in the chunk at A lane l's D bytes at lane l ^ chunk_fold(A / P, wg), so that
    channel c holds lane c of every chunk. Its own inverse at the same address; the same bytes at
    another address are hash_rows(hash_rows(x, at, wg, D), new, wg, D)."""
    data = np.asarray(data, np.uint8)
    P = wg * D
    if at % P or data.size % P:
        raise ValueError(f"grouped bytes at {at:#x} ({data.size} bytes) are not whole {P}-byte "
                         f"chunks")
    c = data.reshape(-1, wg, D)
    h = chunk_fold(at // P + np.arange(c.shape[0]), wg)
    return c[np.arange(c.shape[0])[:, None], np.arange(wg)[None, :] ^ h[:, None]].reshape(
        data.shape)


def ungroup_rows(data: np.ndarray, N: int, wg: int, D: int = 128,
                 fmt: str = "int8") -> np.ndarray:
    """group_rows' inverse: the grouped bytes [G, grp_row_bytes] -> natural rows [N, row_bytes]
    (4-bit: K = the chunks' blocks, padded to whole D-byte chunks)."""
    data = np.asarray(data, np.uint8)
    G = data.shape[0]
    nc = data.shape[1] // (wg * D)
    R = grp_r(fmt, wg)
    if R == wg:
        return data.reshape(G, nc, R, D).transpose(0, 2, 1, 3).reshape(G * R, nc * D)[:N]
    c = data.reshape(G, nc, wg, D).transpose(0, 2, 1, 3)        # [g, lane, k, i]
    codes = np.stack([c & 15, c >> 4], 2).reshape(G * R, nc * D)
    out = np.zeros((G * R, row_bytes(nc * D, fmt, D)), np.uint8)
    out[:, :nc * D // 2] = pack4(codes)
    return out[:N]


def _scale_parts(srows: np.ndarray, fmt: str, KB: int, rsc: bool):
    """Natural scale rows -> (Sw bytes [N, 4], block data [N, KB, sb] or None)."""
    srows = np.asarray(srows, np.uint8)
    if rsc:
        return srows[:, :4], None
    h, sb = swp(fmt, 1), SCALE_DATA[fmt]
    return srows[:, :4], srows[:, h:h + KB * sb].reshape(-1, KB, sb)


def group_scales(srows: np.ndarray, fmt: str, K: int, wg: int, D: int = 128,
                 rsc: bool = False) -> np.ndarray:
    """Natural scale rows [N, >= scale_bytes] (quantize_mxu's) -> the grouped scale areas
    [ceil(N / R), grp_scale_bytes] (uint8), the padding rows' Sw and data 0."""
    KB = K // D
    sw, dat = _scale_parts(srows, fmt, KB, rsc)
    N = sw.shape[0]
    R = grp_r(fmt, wg)
    G, p = -(-N // R), swp(fmt, R, rsc)
    out = np.zeros((G, grp_scale_bytes(K, fmt, D, wg, rsc)), np.uint8)
    s = np.zeros((G * R, 4), np.uint8)
    s[:N] = sw
    out[:, :4 * R] = s.reshape(G, 4 * R)
    if dat is not None:
        sb = dat.shape[2]
        d = np.zeros((G * R, KB, sb), np.uint8)
        d[:N] = dat
        out[:, p:p + KB * R * sb] = d.reshape(G, R, KB, sb).transpose(0, 2, 1, 3).reshape(G, -1)
    return out


def ungroup_scales(areas: np.ndarray, fmt: str, K: int, N: int, wg: int, D: int = 128,
                   rsc: bool = False) -> np.ndarray:
    """group_scales' inverse: grouped areas -> natural scale rows [N, scale_bytes]."""
    areas = np.asarray(areas, np.uint8)
    KB, G = K // D, areas.shape[0]
    R = grp_r(fmt, wg)
    out = np.zeros((G * R, scale_bytes(K, fmt, D, rsc)), np.uint8)
    out[:, :4] = areas[:, :4 * R].reshape(G * R, 4)
    if not rsc:
        h, sb, p = swp(fmt, 1), SCALE_DATA[fmt], swp(fmt, R)
        d = areas[:, p:p + KB * R * sb].reshape(G, KB, R, sb).transpose(0, 2, 1, 3)
        out[:, h:h + KB * sb] = d.reshape(G * R, KB * sb)
    return out[:N]


def grouped(q: np.ndarray, s: np.ndarray, fmt: str, K: int, wg: int, D: int = 128,
            at: int | None = None, kt: bool = False):
    """quantize_mxu's (rows, scale rows) of a [., K] matrix grouped for a machine of wg lanes:
    (data [ceil(N/R), grp_row_bytes], scale areas [ceil(N/R), grp_scale_bytes]); wg = 1: as
    given. `at`: the data's DRAM address on a machine with the channel hash (Config.CHASH): its
    lanes as stored there (hash_rows); None: group_rows' order. kt: in MM KT's pairs (data
    [ceil(G / 2), kt_pair_bytes]: kt_rows, kt_scales; the matrix's K ends in the first half of
    its last D-block, kt_fits)."""
    if wg == 1:
        if kt:
            raise ValueError("grouped: KT's pairs are of groups (wg > 1)")
        return q, s
    d, sc = group_rows(q, wg, D, fmt, K), group_scales(s, fmt, K, wg, D)
    if kt:
        d, sc = kt_rows(d, wg, D), kt_scales(sc, fmt, K // D, wg)
    return (d if at is None else hash_rows(d, at, wg, D)), sc


def kt_fits(k: int, D: int = 128) -> bool:
    """A matrix whose own K (k columns, before an image pads it to whole D-blocks) ends in the
    first half of its last D-block: the rest of that block is zero, MM KT's pairs hold it
    (docs/isa.md "K tail")."""
    return 0 < k % D <= D // 2


def kt_pair_bytes(K: int, fmt: str, D: int = 128, wg: int = 1) -> int:
    """The pair stride of a [., K] matrix in MM KT's pairs: two groups' chunks but one, (2 KB -
    1) P (P = wg x D; docs/isa.md "K tail")."""
    return 2 * grp_row_bytes(K, fmt, D, wg) - wg * D


def kt_rows(data: np.ndarray, wg: int, D: int = 128) -> np.ndarray:
    """A grouped matrix's bytes [G, KB P] (group_rows'; P = wg x D) whose K ends in the middle
    of its last K-block (every lane's bytes D/2 .. D-1 of chunk KB-1 zero) -> MM KT's pairs
    [ceil(G / 2), (2 KB - 1) P] (docs/isa.md "K tail"): pair p is group 2p's chunks 0 .. KB-2,
    the tail chunk (lane l: group 2p's half block, then group 2p + 1's), group 2p + 1's chunks
    0 .. KB-2; an odd G's last pair has group 2p alone (its tail's upper halves 0, then zeros).
    One chunk a pair less than the groups': the half blocks of zeros are not streamed."""
    data = np.asarray(data, np.uint8)
    P, G = wg * D, data.shape[0]
    KB = data.shape[1] // P
    c = np.zeros((2 * -(-G // 2), KB, wg, D), np.uint8)
    c[:G] = data.reshape(G, KB, wg, D)
    if np.any(c[:, KB - 1, :, D // 2:]):
        raise ValueError("kt_rows: the last K-block's upper halves are not zero")
    ev, od = c[0::2], c[1::2]
    t = np.concatenate([ev[:, KB - 1, :, :D // 2], od[:, KB - 1, :, :D // 2]], -1)
    return np.concatenate([ev[:, :KB - 1], t[:, None], od[:, :KB - 1]], 1).reshape(len(ev), -1)


def kt_scales(areas: np.ndarray, fmt: str, KB: int, wg: int, rsc: bool = False) -> np.ndarray:
    """Grouped scale areas [G, .] (group_scales', KB blocks' entries) -> MM KT's (docs/isa.md "K
    tail"): an odd group's entries in the order it takes its chunks, the tail block's first
    (int4 / fp4: its sub-blocks 0 and 1's multipliers also at 2 and 3, the half that meets its
    swapped activation), then blocks 0 .. KB-2; the even groups' as they are."""
    areas = np.array(areas, np.uint8)
    if rsc or len(areas) < 2:
        return areas
    R = grp_r(fmt, wg)
    p, sb = swp(fmt, R), SCALE_DATA[fmt]
    e = areas[1::2, p:p + KB * R * sb].reshape(-1, KB, R, sb)
    tail = e[:, KB - 1:].copy()
    if fmt in ("int4", "fp4"):
        w = tail.view("<u4")
        w[:] = (w & 0xFFFFF) | (((w >> 8) & 0xFFF) << 20)
    elif fmt != "int8":
        raise ValueError(f"kt_scales: {fmt} has no K tail")
    areas[1::2, p:p + KB * R * sb] = np.concatenate([tail, e[:, :KB - 1]], 1).reshape(len(e), -1)
    return areas


def quantize_grouped(W: np.ndarray, fmt: str, wg: int, D: int = 128, at: int | None = None):
    """W [N, K] in an MXU format, grouped for wg lanes (grouped(quantize_mxu(W)), `at` its)."""
    return grouped(*quantize_mxu(W, fmt, D), fmt, W.shape[1], wg, D, at)


def grp_bytes(N: int, K: int, fmt: str, D: int = 128, wg: int = 1, kt: bool = False) -> tuple:
    """(data, scale) bytes of an [N, K] matrix grouped for wg lanes (wg = 1: N rows' row_bytes
    and scale_bytes); kt: its data in MM KT's pairs (whole pairs, grouped's)."""
    G = -(-N // grp_r(fmt, wg))
    d = -(-G // 2) * kt_pair_bytes(K, fmt, D, wg) if kt else G * grp_row_bytes(K, fmt, D, wg)
    return d, G * grp_scale_bytes(K, fmt, D, wg)


# =============================================================================== emulation
class MXAct(np.ndarray):
    """Activations as QACT leaves them (float64 values) with `da`, each group of D/8 elements'
    power-of-two offset below its row's scale [..., K / (D/8)] (MM's drop rule; mx_matmul).
    Any array derived from one (a slice, a reshape, arithmetic) is a plain one: da = None."""

    def __array_finalize__(self, obj):
        self.da = None

    @staticmethod
    def of(values, da) -> "MXAct":
        a = np.asarray(values).view(MXAct)
        a.da = None if da is None else np.asarray(da)
        return a

    def rows(self, sl) -> "MXAct":
        """Rows sl (an index or slice of the leading axes) with their offsets."""
        return MXAct.of(np.asarray(self).view(np.ndarray)[sl],
                        None if self.da is None else self.da[sl])


class MXW(np.ndarray):
    """A weight as the MXU holds it: its values (float64 [N, K]) with the groups' offsets dw
    [N or 1, K / (D/8)] and the window W (`mx`); np.matmul with it, w @ h (h [K]) or h @ w.T,
    sums as MM does (mx_matmul: the groups the window drops left out where h is an MXAct).
    Anything else derived from it (a slice, arithmetic) is a plain array."""

    def __array_finalize__(self, obj):
        self.mx = None                  # (dw, W, D, transposed)

    @staticmethod
    def of(values, dw, W: int, D: int = 128) -> "MXW":
        """(float32 values stay float32: a big matrix held at half the memory.)"""
        v = np.asarray(values)
        w = (v if v.dtype in (np.float32, np.float64) else v.astype(np.float64)).view(MXW)
        w.mx = (np.asarray(dw), W, D, False)
        return w

    def astype(self, dtype, *a, **kw):
        """Its values in another float type, the offsets kept."""
        return MXW.of(np.asarray(self).view(np.ndarray).astype(dtype, *a, **kw), self.mx[0],
                      self.mx[1], self.mx[2]) if self.mx is not None and not self.mx[3] else \
            np.asarray(self).view(np.ndarray).astype(dtype, *a, **kw)

    @property
    def T(self):
        t = np.ndarray.transpose(self).view(MXW)
        if self.mx is not None:
            dw, W, D, tr = self.mx
            t.mx = (dw, W, D, not tr)
        return t

    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        if ufunc is np.matmul and method == "__call__" and not kw and len(inputs) == 2:
            a, b = inputs
            if isinstance(b, MXW) and b.mx is not None and b.mx[3]:          # h @ w.T
                dw, W, D, _ = b.mx
                return mx_matmul(a, np.asarray(b).view(np.ndarray).T, dw, W, D)
            if isinstance(a, MXW) and a.mx is not None and not a.mx[3] and np.ndim(b) == 1:
                dw, W, D, _ = a.mx                                             # w @ h
                return mx_matmul(b, np.asarray(a).view(np.ndarray), dw, W, D)
        args = [np.asarray(x).view(np.ndarray) if isinstance(x, np.ndarray) else x
                for x in inputs]
        if "out" in kw:
            kw["out"] = tuple(np.asarray(o).view(np.ndarray) for o in kw["out"])
        return getattr(ufunc, method)(*args, **kw)


def fake_w(W: np.ndarray, fmt: str, D: int = 128, quantized: tuple | None = None) -> MXW:
    """W [N, K] as the MXU holds it in format fmt (quantize_mxu, dequantized; `quantized`: its
    (rows, scale rows) already, e.g. opentpu.qcache's): an MXW."""
    from .isa import MX_W, WFORMATS
    rows, srows = quantize_mxu(W, fmt, D) if quantized is None else quantized
    return MXW.of(dequantize_mxu(rows, srows, fmt, W.shape[1], D),
                  group_offsets(srows, fmt, W.shape[1], D), MX_W[WFORMATS[fmt]], D)


def fake_act(x, D: int = 128, n: int | None = None, row: bool = False,
             g: int | None = None):
    """x [..., K] quantized as QACT / QST quantize it (docs/isa.md "Quantization"), in float64:
    each row in ranges of n columns (one QACT each; default the whole row; a sequence: the
    ranges' widths, one after the other) has the scale S =
    amax / 127, its groups of g (QACT: D/4, the default; QST: D) the offset d, the largest
    (<= 31) with the group's amax 2^d <= amax (0 in ROW mode), and its values
    rint(x 127 / amax 2^d) S 2^-d, clipped to 127. A value within 2^-31 of a rounding tie rounds
    as the device's quantizer rounds it in fp32 (fp32.quantize_mx: x * (127 recip(amax) 2^d)),
    not as float64 noise has it (a block whose values have exact ratios has exact ties).
    Returns an MXAct (its da per D/8)."""
    from . import fp32 as F
    x = np.asarray(x, np.float64)
    sh = x.shape
    K = sh[-1]
    if n is not None and not isinstance(n, (int, np.integer)):        # ranges of their own
        if sum(n) != K:
            raise ValueError(f"fake_act: ranges {list(n)} over {K} columns")
        c, parts = 0, []
        for w in n:
            parts.append(fake_act(x[..., c:c + w], D, None, row, g))
            c += w
        out = np.concatenate([np.asarray(p).view(np.ndarray) for p in parts], -1).view(MXAct)
        out.da = np.concatenate([p.da for p in parts], -1)
        return out
    n = K if n is None else n
    g = D // NSUB if g is None else g
    if K % n or n % g:
        raise ValueError(f"fake_act: ranges of {n} over {K} columns, groups of {g}")
    xr = x.reshape(-1, n)                                       # one QACT range a row
    amax = np.abs(xr).max(-1, keepdims=True)
    xb = xr.reshape(len(xr), n // g, g)
    ag = np.abs(xb).max(-1)
    if row:
        d = np.zeros(ag.shape, np.int64)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            fa, ea = np.frexp(ag)
            fm, em = np.frexp(np.broadcast_to(amax, ag.shape))
        d = np.where(ag > 0, np.clip(em - ea - (fa > fm), 0, 31), 0)
    s = np.where(amax == 0, 1.0, amax / 127)                    # [rows, 1]
    unit = s * np.exp2(-d)                                      # [rows, n/g]
    r = xb / unit[..., None]
    q = np.rint(r)
    tie = np.abs(r - q) > 0.5 - 2.0 ** -31
    if tie.any():
        b = tie.any(-1).any(-1)
        qd = F.quantize_mx(xr[b].astype(np.float32), g, row)[0].astype(np.float64)
        q[b] = np.where(tie[b], qd.reshape(q[b].shape), q[b])
    out = (np.clip(q, -127, 127) * unit[..., None]).reshape(sh).view(MXAct)
    out.da = np.repeat(d, g * NVG // D, axis=-1).reshape(*sh[:-1], K // (D // NVG))
    return out


def fake_act_gathered(x, D: int = 128, C: int | None = None, S: int = 1):
    """A down projection's input x [..., S F'] quantized as kernels.mlp.swiglu_down's QACTs on S
    slices (fake_act): chunk c of every slice's F' (gather_order: S C columns) one range; C None:
    the whole row. S = 1: fake_act's ranges of C."""
    if C is None or S == 1:
        return fake_act(x, D, C)
    g = D // NVG                                                # da: an offset per D/8
    if C % g:
        raise ValueError(f"fake_act_gathered: chunks of {C} are not whole groups of {g}")
    x = np.asarray(x, np.float64)
    order = gather_order(x.shape[-1], C, S)
    a = fake_act(x[..., order], D, S * C)
    return MXAct.of(np.asarray(a).view(np.ndarray)[..., np.argsort(order)],
                    a.da[..., np.argsort(order[::g] // g)])


def mx_matmul(h, w: np.ndarray, dw: np.ndarray | None = None, W: int = 16,
              D: int = 128) -> np.ndarray:
    """h @ w.T (float64) as MM sums it: w [N, K] the streamed matrix's values with its groups'
    offsets dw [N or 1, K / (D/8)] (None: 0, UNIT or RSC) and the format's window W
    (isa.MX_W); h an MXAct (a plain array: no offsets known, nothing dropped), the groups whose
    two offsets add up to more than W left out."""
    hv = np.asarray(h).view(np.ndarray)
    y = hv @ w.T
    da = getattr(h, "da", None)
    if da is None:
        return y
    K = w.shape[1]
    dw = np.zeros((1, K // (D // NVG)), np.int64) if dw is None else np.asarray(dw, np.int64)
    hv = hv.reshape(-1, K)
    dg = np.asarray(da).reshape(-1, dw.shape[1])
    yf = y.reshape(-1, w.shape[0])
    G = D // NVG
    dwmax = dw.max(0)
    for gi in np.nonzero(dg.max(0) + dwmax > W)[0]:
        rows = np.nonzero(dg[:, gi] + dwmax[gi] > W)[0]
        m = (dg[rows, gi][:, None] + dw[None, :, gi]) > W           # [rows, N] (dw [1, K]: UNIT)
        if not m.any():
            continue
        sl = slice(G * gi, G * (gi + 1))
        yf[rows] -= (hv[rows, sl] @ w[:, sl].T) * m
    return y


# ============================================================================ checkpoint imports
class NotExact(ValueError):
    """A checkpoint's quantized weights that no MXU format holds exactly."""


def _pow2_f32(e: np.ndarray, what: str) -> np.ndarray:
    """2^e (integers) as normal fp32 values (NotExact past fp32's normal range)."""
    e = np.asarray(e, np.int64)
    if np.any(e < -126) or np.any(e > 127):
        raise NotExact(f"{what}: a row scale 2^{int(e.min())}..2^{int(e.max())} outside fp32's "
                       f"normal range")
    return np.ldexp(np.ones(e.shape, np.float32), e.astype(np.int32)).astype(np.float32)


def import_mxfp4(codes: np.ndarray, e8m0: np.ndarray, D: int = 128, want: str | None = None):
    """An OCP MXFP4 matrix (E2M1 element codes [N, K], 0..15 with the sign in bit 3; an E8M0
    scale per 32 elements [N, K / 32], value 2^(e - 127) E2M1) in an MXU format, exactly:
    (rows, scale rows, format). fp4 (fp4x6) when every K-block's sub-blocks' exponents span at
    most 5 (m_b = 2^(e_b - the block's smallest e), dw the block's below the row's largest such
    smallest e; Sw a power of two); else nvfp4 (two groups of D/8 a scale, E4M3 bytes of
    mantissa 0, subnormals 1, 2, 4 at the bottom) when every row's exponents span at most 17;
    else NotExact. Sub-blocks of zeros have any exponent. MM's window then drops what lies more
    than W below the row's top (fp4 11, nvfp4 13 octaves with the activations' offsets; the
    representation itself is exact). want "fp4" / "nvfp4": that format or NotExact (a matrix
    that fp4 holds in nvfp4: a slot format of a model's experts)."""
    codes = np.asarray(codes, np.uint8)
    e = np.asarray(e8m0, np.int64)
    N, K = codes.shape
    if D != 128 or K % D or e.shape != (N, K // 32):
        raise ValueError(f"MXFP4 [{N}, {K}] with scales {e.shape}: blocks of 32, D = 128")
    if np.any(e == 0xFF):
        raise NotExact("MXFP4: an E8M0 NaN scale")
    KB = K // D
    live = (codes.reshape(N, K // 32, 32) & 7).any(-1)                 # [N, K/32]
    eb = e.reshape(N, KB, NSUB)
    lv = live.reshape(N, KB, NSUB)
    big, small = 1 << 20, -(1 << 20)
    emin = np.where(lv, eb, big).min(-1)                                # [N, KB]
    emax = np.where(lv, eb, small).max(-1)
    blive = lv.any(-1)
    rows = np.zeros((N, row_bytes(K, "fp4", D)), np.uint8)
    rows[:, :K // 2] = pack4(codes)
    if want not in (None, "fp4", "nvfp4"):
        raise ValueError(f"MXFP4 into {want!r}: fp4 or nvfp4")
    if want != "nvfp4" and np.all(np.where(blive, emax - emin, 0) <= 5):
        top = np.where(blive, emin, small).max(-1)                      # [N]
        top = np.where(top == small, 129, top)                          # an all-zero row
        dw = np.where(blive, top[:, None] - emin, 0)
        if np.any(dw > DW_MAX):
            raise NotExact("MXFP4: a block more than 255 octaves below its row's top")
        m = np.where(lv, np.exp2(eb - np.where(blive, emin, 0)[..., None]), 1).astype(np.uint8)
        Sw = _pow2_f32(top - 128, "MXFP4")                              # codes are 2x E2M1
        return rows, pack_scales("fp4", Sw, dw=dw.astype(np.uint8), m=m), "fp4"
    if want == "fp4":
        raise NotExact("MXFP4: a K-block's exponents span more than fp4's 5 octaves")
    top = np.where(live, e, small).max(-1)                              # [N]
    span = np.where(live, top[:, None] - e, 0)
    if np.any(span > 17):
        raise NotExact(f"MXFP4: a row's scales span {int(span.max())} octaves, nvfp4's E4M3 "
                       f"holds 17 with mantissa 0")
    E = 15 - span                                                       # [N, K/32]
    e4 = np.where(E >= 1, E << 3, np.where(E == 0, 4, np.where(E == -1, 2, 1)))
    e4 = np.where(live, e4, 0).astype(np.uint8)
    e4 = np.repeat(e4, 2, axis=1).reshape(N, KB, NVG)                  # a scale per D/8
    top = np.where(top == small, 131, top)
    return rows, pack_scales("nvfp4", _pow2_f32(top - 131, "MXFP4"), e4=e4), "nvfp4"


def import_nvfp4(codes: np.ndarray, e4m3_scales: np.ndarray, tensor_scale: float,
                 D: int = 128):
    """An NVFP4 matrix (E2M1 element codes [N, K], the sign in bit 3; an E4M3 scale per 16
    elements [N, K / 16], uint8, and the per-tensor fp32 scale: value E2M1 e4m3 ts) in the MXU's
    nvfp4, exactly: (rows, scale rows). A scale m 2^p (m = 8 + M, or M for a subnormal) is
    normalized to m' in 8..15 and each row re-based so that its largest has dw 0 (exponent
    field 15; the MXU reads 0x7F as 15 2^0, not as a NaN, so a row whose largest scale's
    mantissa is 7 needs no other shift); a scale more than 14 octaves below the row's largest
    becomes an MXU subnormal (M' = m' 2^(14 - dw)) where that is exact. Sw = ts 2^(top - 1)
    for the row's largest p (codes are 2x E2M1). NotExact on a NaN or negative scale (the
    MXU's E4M3 scales are unsigned), a scale too far below its row's largest to hold exactly
    (dw > 14 with low bits a subnormal drops; past 17 none), or a row scale outside fp32's
    normal range."""
    codes = np.asarray(codes, np.uint8)
    s = np.asarray(e4m3_scales, np.uint8)
    N, K = codes.shape
    if D != 128 or K % D or s.shape != (N, K // 16):
        raise ValueError(f"NVFP4 [{N}, {K}] with scales {s.shape}: groups of 16, D = 128")
    if np.any(s & 0x80):
        raise NotExact("NVFP4: a negative E4M3 scale")
    if np.any(s == 0x7F):
        raise NotExact("NVFP4: an E4M3 NaN scale")
    E, M = (s >> 3).astype(np.int64) & 15, (s & 7).astype(np.int64)
    m = np.where(E > 0, 8 + M, M)                                       # scale m 2^p
    p = np.maximum(E, 1) - 10
    live = m > 0
    sh = np.where(live, 3 - np.floor(np.log2(np.maximum(m, 1))).astype(np.int64), 0)
    mn, pn = m << sh, p - sh                                            # m' in 8..15
    low = -(1 << 20)
    top = np.where(live, pn, low).max(-1)                               # [N]
    top = np.where(top == low, 1, top)                                  # an all-zero row
    dw = np.where(live, top[:, None] - pn, 0)
    sub = live & (dw > 14)                                              # MXU subnormals
    ms = np.where(sub, mn >> np.clip(dw - 14, 0, 3), 0)
    if np.any(dw > 17) or np.any(sub & ((ms << np.clip(dw - 14, 0, 3)) != mn)):
        raise NotExact("NVFP4: a scale more than 14 octaves below its row's largest whose "
                       "low bits an MXU subnormal does not hold")
    e4 = np.where(~live, 0, np.where(dw <= 14, ((15 - dw) << 3) | (mn - 8), ms))
    sw = np.float64(np.float32(tensor_scale)) * np.exp2(top - 1.0)
    if np.any((sw != 0) & ((np.abs(sw) < 2.0 ** -126) | (np.abs(sw) >= 2.0 ** 128))):
        raise NotExact("NVFP4: a row scale outside fp32's normal range")
    rows = np.zeros((N, row_bytes(K, "nvfp4", D)), np.uint8)
    rows[:, :K // 2] = pack4(codes)
    return rows, pack_scales("nvfp4", sw.astype(np.float32),
                             e4=e4.reshape(N, K // D, NVG).astype(np.uint8))


