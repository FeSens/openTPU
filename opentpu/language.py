"""openTPU kernel language (import as `ol`), in the spirit of triton.language / Gluon.

Kernels are SPMD over slices. Inside a kernel:

    pid = ol.program_id()                 # this slice
    x   = ol.load(desc)                   # DRAM fp32 -> TMEM tile
    y   = ol.dot(x, w)                    # x: tile [M, K]; w: streamed QTensor [N, K] -> [M, N]
    m   = ol.max(y, axis=1)               # row reductions
    p   = ol.exp2(y - m[:, None])         # broadcasting maps onto the VPU operand modes
    for i in ol.range(n): ...             # hardware loop (body traced once; use t.set(...))
    for i in ol.static_range(n): ...      # unrolled
    z   = ol.all_gather(y_shard)          # sharded -> replicated across slices
    ol.store(out_desc, z)

Everything the hardware does is visible here: each call lowers to one or a few instructions.
"""
from __future__ import annotations

import builtins
import math
import sys

from . import isa as I
from .compiler import (TEMP_RC_FN, Affine, Bcast, CompileError, KVDesc, QTensor, Stationary,
                       Tensor, Tile, current, jit)

__all__ = ["jit", "program_id", "num_programs", "block_size", "tmem_words", "load", "store", "dot", "quantize", "exp2",
           "recip", "rsqrt", "abs", "maximum", "minimum", "max", "sum", "full", "zeros",
           "empty", "all_gather", "all_reduce", "range", "static_range", "kv_append", "Tensor", "QTensor",
           "KVDesc", "Tile", "CompileError", "LOG2E"]

LOG2E = 1.0 / math.log(2.0)


def program_id() -> int:
    return current().sid


def num_programs() -> int:
    return current().S


def block_size() -> int:
    """The MXU depth D (quantization block)."""
    return current().cfg.D


def tmem_words() -> int:
    """TMEM capacity in 32-bit words."""
    return current().cfg.TMEM_WORDS


# ---- memory
def load(desc: Tensor) -> Tile:
    return current().load(desc)


def store(desc: Tensor, value) -> None:
    current().store(desc, value)


def quantize(x: Tile) -> Stationary:
    """Quantize rows of `x` into ACT RAM once, to reuse as the stationary operand of dot().

    `quantize(a * v[None, :])` on an unnamed product fuses the column scaling into the
    quantizer (QACT CSCALE): no separate vector pass.
    """
    temp = sys.getrefcount(x) <= TEMP_RC_FN
    return current().quantize(x, temp)


def dot(a, w: QTensor, acc: Tile | None = None, out: Tile | None = None,
        rowmax: bool = False, acc_scale: Tile | None = None) -> Tile:
    """a[M, K] @ w[N, K]^T with block-scaled int8 operands and fp32 accumulation.

    `a` is a TMEM tile (quantized on the fly) or the result of `quantize`. With `acc`, the
    result is added into `acc` in place; with `out`, it overwrites `out` (e.g. one of two
    score buffers of a software-pipelined loop). `ol.max(dot(...), axis=1)` is computed by
    the MXU epilogue for free (MM RMAX); `rowmax=True` requests the maxima explicitly and
    they are then read as `out.rowmax`. `acc_scale=alpha` rescales the accumulator first,
    acc = acc * alpha[:, None] + a @ w^T, in the MXU epilogue (the flash-attention correction).
    """
    temp = sys.getrefcount(a) <= TEMP_RC_FN
    return current().dot(a, w, acc, out, temp, rowmax, acc_scale)


def all_gather(x: Tile) -> Tile:
    """Concatenate every slice's `x` along the last axis, replicated in all slices.
    With a single slice this is `x` itself."""
    b = current()
    if b.S == 1:
        return x
    return b.all_gather(x, b.S)


def all_reduce(x: Tile) -> Tile:
    """Sum `x` over all slices, replicated in all slices (an all-gather, then S-1 adds)."""
    b = current()
    if b.S == 1:
        return x
    g = b.all_gather(x, b.S)
    n = x.cols
    if len(x.shape) == 1:
        acc = g[0:n] + g[n:2 * n]
        for s in builtins.range(2, b.S):
            acc = acc + g[s * n:(s + 1) * n]
    else:
        acc = g[:, 0:n] + g[:, n:2 * n]
        for s in builtins.range(2, b.S):
            acc = acc + g[:, s * n:(s + 1) * n]
    return acc


# ---- construction
def _acc_layout(shape) -> int | None:
    """Constructed 2-D tiles are typically MXU accumulators: give them an odd row stride so the
    MXU writes a streamed row's results to distinct TMEM banks in one cycle."""
    shape = tuple(shape)
    if len(shape) == 2 and shape[0] > 1 and shape[1] % 2 == 0:
        return shape[1] + 1
    return None


def _spare(shape) -> int:
    shape = tuple(shape)
    return shape[0] if len(shape) == 2 else 0


def empty(shape) -> Tile:
    return current().alloc(tuple(shape), _acc_layout(shape), _spare(shape))


def full(shape, value: float) -> Tile:
    t = current().alloc(tuple(shape), _acc_layout(shape), _spare(shape))
    return t.set(float(value))


def zeros(shape) -> Tile:
    return full(shape, 0.0)


# ---- elementwise / reductions
def exp2(x) -> Tile:
    """2**x. `exp2(a - b)` on an unnamed temporary fuses into one EXP2SUB pass."""
    temp = sys.getrefcount(x) <= TEMP_RC_FN     # measured before x is passed on
    return current().unop(I.V_EXP2, x, temp=temp)


def recip(x) -> Tile:
    return current().unop(I.V_RECIP, x)


def rsqrt(x) -> Tile:
    return current().unop(I.V_RSQRT, x)


def abs(x) -> Tile:  # noqa: A001
    return current().unop(I.V_ABS, x)


def maximum(x, y) -> Tile:
    return current().binop(I.V_MAX, x, y)


def minimum(x, y) -> Tile:
    return current().binop(I.V_MIN, x, y)


def max(x, axis: int = -1) -> Tile:  # noqa: A001
    return current().reduce(I.V_RMAX, x, axis)


def sum(x, axis: int = -1) -> Tile:  # noqa: A001
    """Row sums. `sum(a * a)` on an unnamed square is one RSSQ pass (sum of squares)."""
    temp = sys.getrefcount(x) <= TEMP_RC_FN
    return current().reduce(I.V_RSUM, x, axis, temp)


# ---- control
def range(n: int):  # noqa: A001
    """Hardware loop. The body is traced once; carry values across iterations with `.set()`."""
    b = current()
    n = int(n)
    if n <= 0:
        return
    loop = b.begin_loop(n)
    try:
        yield loop
    finally:
        b.end_loop(loop)


def static_range(*args):
    return builtins.range(*args)


# ---- KV cache
def kv_append(kv: KVDesc, h: int, pos, k: Tile, v: Tile) -> None:
    """Quantize and append rows of k, v ([n, d]) at token position `pos` of KV head `h`.

    K rows are written token-major with per-block scales; V is written transposed (one byte per
    dimension, stride = capacity) with one scale per token.
    """
    b = current()
    pos = Affine.of(pos)
    kd = kv.k(h)
    b.store_quantized(k, kd.data + pos * kd.rs, kd.scale + pos * kd.srs, kd.rs, 1,
                      row_scale=False)
    vt = kv.vt(h)
    vs = kv.vscale(h)
    b.store_quantized(v, vt.data + pos, vs.base + pos * 4, 1, vt.rs, row_scale=True)
