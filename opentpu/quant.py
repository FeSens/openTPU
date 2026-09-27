"""Weight quantization formats: reference quantizers (numpy) for 8-bit and 4-bit weights.

Every quantizer takes an fp32 matrix W [N, K] (rows = output features, K = the inner dimension
the MXU reduces over) and quantizes blocks of `g` consecutive elements of a row, like the MXU's
per-(row, D-block) scales. It returns a `Quantized`: the element codes, the block scales and
the dequantized matrix (what the model computes with), plus the storage cost in bits per
weight (elements + block scales; a per-tensor scale is negligible and not counted).

Formats
  int8      the current openTPU format: int8 in [-127, 127], one fp32 scale per 128 (g = D),
            bit-identical to the hardware quantizer (runtime.quantize_rows)
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
            sub-block scale is bf16 * m (the elements are int4 or E2M1)

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
        s = np.where(amax == 0, 1.0, s)
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
    if fmt == "int8":
        from .runtime import quantize_rows
        g = g or 128
        q, s = quantize_rows(W, g)
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


# =============================================================================== MXU 4-bit
# The streamed 4-bit weight formats of the MM instruction (docs/isa.md "Weight formats"): two
# elements per byte (element 2i in the low nibble), 128 elements (one K-block) per 64 bytes, and
# one 32-bit scale word per K-block: bits [15:0] a bf16 block scale S, bits [16+4b, 20+4b) the
# unsigned multiplier m_b of sub-block b (elements 32b .. 32b+31). The MXU multiplies integers:
# int4 codes are two's complement in [-8, 7]; E2M1 codes are taken as twice their value,
# {0, 1, 2, 3, 4, 6, 8, 12} with the sign in bit 3, so an E2M1 matrix stores S / 2.
DEC_INT4 = np.array([0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1], np.int8)
DEC_FP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.int8)
DEC = {"int4": DEC_INT4, "fp4": DEC_FP4}
NSUB = 4                       # sub-blocks per K-block


def unpack4(b: np.ndarray) -> np.ndarray:
    """Bytes [..., n] -> nibble codes [..., 2n], the low nibble first."""
    b = np.asarray(b, np.uint8)
    return np.stack([b & 15, b >> 4], -1).reshape(*b.shape[:-1], 2 * b.shape[-1])


def pack4(codes: np.ndarray) -> np.ndarray:
    """Nibble codes [..., 2n] (0..15) -> bytes [..., n]."""
    c = np.asarray(codes, np.uint8)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """fp32 values that are bf16 already -> their 16 bits."""
    return (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint32)


def quantize_w4(W: np.ndarray, fmt: str, D: int = 128, search: bool = True):
    """Quantize W [N, K] to the MXU's 4-bit format ("int4" or "fp4", two-level scales over
    K-blocks of D). Returns (bytes [N, K/2] uint8, scale words [N, K/D] uint32, dequantized
    [N, K] fp32). The dequantized matrix is exactly S * m_b * dec(code) per element."""
    W = np.asarray(W, np.float32)
    N, K = W.shape
    if K % D:
        raise ValueError(f"inner dimension {K} is not a multiple of D={D}")
    grid = {"int4": "int4", "fp4": "e2m1"}[fmt]
    deq, S, m = quant_two(W, grid, G=D, sub=D // NSUB, search=search)
    S = S.astype(np.float32)                                        # bf16 values, [N, K/D]
    s = (S[..., None] * m).astype(np.float64)[..., None]             # [N, K/D, 4, 1]
    x = deq.reshape(N, K // D, NSUB, D // NSUB) / s                  # grid values (exact)
    if grid == "int4":
        code = np.rint(x).astype(np.int64) & 15
    else:
        mag = e2m1_code(x)                                           # exact grid points
        code = mag | np.where(np.signbit(x) & (mag != 0), 8, 0)
        S = S * np.float32(0.5)                                      # codes are 2x the value
    code = code.reshape(N, K)
    word = bf16_bits(S) | sum(m[..., b].astype(np.uint32) << np.uint32(16 + 4 * b)
                              for b in range(NSUB))
    return pack4(code), word.astype(np.uint32), deq


def dequantize_w4(data: np.ndarray, words: np.ndarray, fmt: str, D: int = 128) -> np.ndarray:
    """The matrix a packed 4-bit weight stands for (the MXU's arithmetic, in float64)."""
    N = data.shape[0]
    w = DEC[fmt][unpack4(data)].astype(np.float64).reshape(N, -1, NSUB, D // NSUB)
    S = (np.asarray(words, np.uint32) << 16).view(np.float32).astype(np.float64)
    m = np.stack([(words >> (16 + 4 * b)) & 15 for b in range(NSUB)], -1).astype(np.float64)
    return (w * (S[..., None] * m)[..., None]).reshape(N, -1)


def row_bytes(K: int, fmt: str, D: int = 128) -> int:
    """DRAM bytes of one streamed row of K elements (4-bit rows padded to whole D-byte chunks)."""
    return K if fmt == "int8" else -(-K // (2 * D)) * D


def quantize_mxu(W: np.ndarray, fmt: str, D: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """W [N, K] in an MXU weight format ("int8", "int4" or "fp4"): the DRAM rows [N, row_bytes]
    (uint8) and the block scales [N, K/D] (fp32 for int8, scale words for 4-bit), ready to
    place (the scale rows are 4 * K/D bytes apart)."""
    if fmt == "int8":
        from .runtime import quantize_rows
        q, s = quantize_rows(np.asarray(W, np.float32), D)
        return q.view(np.uint8), s.view(np.uint32)
    b, words, _ = quantize_w4(W, fmt, D)
    rows = np.zeros((b.shape[0], row_bytes(W.shape[1], fmt, D)), np.uint8)
    rows[:, :b.shape[1]] = b
    return rows, words


def mxu_wf(fmt: str) -> int:
    from .isa import WFORMATS
    return WFORMATS[fmt]
