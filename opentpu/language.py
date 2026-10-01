"""openTPU kernel language (import as `ol`), in the spirit of triton.language / Gluon.

Kernels are SPMD over slices. Inside a kernel:

    pid = ol.program_id()                 # this slice
    x   = ol.load(desc)                   # DRAM fp32 -> TMEM tile
    y   = ol.dot(x, w)                    # x: tile [M, K]; w: streamed QTensor [N, K] -> [M, N]
    m   = ol.max(y, axis=1)               # row reductions
    p   = ol.exp2(y - m[:, None])         # broadcasting maps onto the VPU operand modes
    kv  = S @ k                           # row dot products of a tile with a vector (RDOT)
    ol.outer(d, k, acc=S, decay=a)        # S = a * S + d k^T in place (OUTER)
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
                       Tensor, Tile, current, jit, tile_split, unnamed)

__all__ = ["jit", "program_id", "num_programs", "block_size", "tmem_words", "mxu_columns",
           "load", "store", "dot", "quantize", "exp2", "log2", "recip", "rsqrt", "abs", "maximum",
           "minimum", "max", "sum", "outer", "full", "zeros",
           "empty", "all_gather", "all_reduce", "range", "static_range", "kv_append", "Tensor", "QTensor",
           "KVDesc", "Tile", "CompileError", "LOG2E", "LN2"]

LOG2E = 1.0 / math.log(2.0)
LN2 = math.log(2.0)


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


def mxu_columns() -> int:
    """MCOLS: the most rows a stationary operand of one MM can have."""
    return current().cfg.MCOLS


# ---- memory
def load(desc: Tensor, out: Tile | None = None) -> Tile:
    """DRAM -> TMEM. With `out`, into that existing tile (e.g. one of two buffers a hardware
    loop fills in turn)."""
    return current().load(desc, out)


def store(desc: Tensor, value) -> None:
    current().store(desc, value)


def quantize(x: Tile) -> Stationary:
    """Quantize rows of `x` into ACT RAM once, to reuse as the stationary operand of dot().

    `quantize(a * v[None, :])` on an unnamed product fuses the column scaling into the
    quantizer (QACT CSCALE): no separate vector pass.
    """
    temp = unnamed(sys.getrefcount(x), TEMP_RC_FN, x)
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
    temp = unnamed(sys.getrefcount(a), TEMP_RC_FN, a)
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


def empty(shape, dense: bool = False) -> Tile:
    """A new tile. dense: rows back to back (row stride = columns), as a load allocates them:
    the destination of loads in pieces (ol.load(..., out=view))."""
    if dense:
        return current().alloc(tuple(shape))
    return current().alloc(tuple(shape), _acc_layout(shape), _spare(shape))


def full(shape, value: float) -> Tile:
    t = current().alloc(tuple(shape), _acc_layout(shape), _spare(shape))
    return t.set(float(value))


def zeros(shape) -> Tile:
    return full(shape, 0.0)


# ---- elementwise / reductions
def exp2(x) -> Tile:
    """2**x. `exp2(a - b)` on an unnamed temporary fuses into one EXP2SUB pass."""
    temp = unnamed(sys.getrefcount(x), TEMP_RC_FN, x)     # measured before x is passed on
    return current().unop(I.V_EXP2, x, temp=temp)


def log2(x) -> Tile:
    """log2(x): -inf at 0, NaN below it (VOP LOG2, within 2.3 ulp)."""
    return current().unop(I.V_LOG2, x)


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
    """Row sums. `sum(a * b)` on an unnamed product is one RDOT pass (b may be broadcast:
    `sum(S * k[None, :], axis=1)` is S @ k); `sum(a * a)` is RSSQ."""
    temp = unnamed(sys.getrefcount(x), TEMP_RC_FN, x)
    return current().reduce(I.V_RSUM, x, axis, temp)


def outer(x: Tile, y: Tile, acc: Tile | None = None, decay: Tile | None = None) -> Tile:
    """The rank-1 tile x[:, None] * y[None, :] (one MUL pass).

    With `acc`, the state update of a linear recurrence in one in-place pass (VOP OUTER):
    acc = acc * decay + x[:, None] * y[None, :], decay a [1] tile (one factor), a [cols] tile
    (per column) or None (1.0). As dot(acc=, acc_scale=) is for attention's accumulator."""
    return current().outer(x, y, acc, decay)


def deltanet_step(state: Tensor, qk: Tile, v: Tile, decay: Tile, beta: Tile, o: Tile,
                  zero: bool = False) -> None:
    """One Gated DeltaNet head step in the DMA (DSTEP, Config.DSTEP): the fp32 state [rows,
    cols] in DRAM is updated in place, row by row, kv = S[r] . k, d = (v[r] - kv * decay) *
    beta, S[r] = S[r] * decay + d * k, o[r] = S[r] . q, with qk = [q | k]. Bit-identical to
    the RDOT, MUL, SUB, MUL, OUTER, RDOT the VPU would run on the loaded state. `zero`: the
    state starts at +0 and is not read (the first token)."""
    current().deltanet_step(state, qk, v, decay, beta, o, zero)


def has_dstep() -> bool:
    """deltanet_step runs on the state in DRAM: the DMA's DSTEP (Config.DSTEP; CAPS bit6 on
    the card) or the stream engine (Config.STREAM; CAPS bit26)."""
    cfg = current().cfg
    return cfg.DSTEP or cfg.STREAM


def has_stream() -> bool:
    """The stream engine runs STREAM's hardware subset (Config.STREAM; CAPS bit26)."""
    return current().cfg.STREAM


_DPROGS = {"delta": I.D_DELTA, "delta1": I.D_DELTA1, "scale": I.D_SCALE, "dot": I.D_DOT}


def state_step(state: Tensor, vec: Tile, x: Tile | None, k0: Tile | None, k1: Tile,
               o: Tile | None = None, mode: str = "delta", a_slot: int = 1,
               gate: str = "const", zero: bool = False, tmp: Tile | None = None) -> None:
    """One step of a linear recurrence on the fp32 state [rows, cols] in DRAM, updated in
    place, row by row (docs/stream.md, 5.2). vec holds column slots of cols words: q (slot
    0), k (1), a (2), the gate column (3), as many as are used. x [rows] holds the rows'
    inputs, k0 and k1 are [1] tiles (k1 after k0 in TMEM; k0 may be None when unused):
      A    = S[r] . slot[a_slot]                              (modes delta, delta1, dot)
      d    = delta: (x[r] - A*k0)*k1 | delta1: (x[r] - A)*k1 | scale: x[r]*k1 | dot: A*k1
      S[r] = S[r]*G + d*k,  G = k0 ("const"), slot 3 ("col") or 1 ("one")
      o[r] = S[r] . q                                         (when o is given)
    Gated DeltaNet is mode "delta" with gate "const"; DeltaNet "delta1"/"one"; KDA "delta1"
    with a = alpha*k in slot 2 and gate "col"; GLA "scale"/"col"; RetNet, Mamba2 and mLSTM
    "scale"/"const". On a Config.STREAM machine it is one STREAM; otherwise the same rounding
    as VOPs on the loaded state (in `tmp`, [rows, cols], or a new tile). `zero`: the state
    starts at +0."""
    b = current()
    rows, cols = state.shape
    if mode not in _DPROGS or gate not in ("const", "col", "one") or a_slot not in (1, 2):
        raise CompileError("state_step: mode delta/delta1/scale/dot, gate const/col/one, "
                           "a_slot 1 or 2")
    uses_a = mode != "scale"
    if (mode == "delta" or gate == "const") and k0 is None:
        raise CompileError(f"state_step: mode {mode} with gate {gate} needs k0")
    if k0 is not None and k1.base - k0.base <= 0:
        raise CompileError("state_step: k1 must follow k0 in TMEM")
    ks = 1 if k0 is None else k1.base - k0.base
    nslot = builtins.max(1, a_slot if uses_a else 0, 3 if gate == "col" else 0) + 1
    if not isinstance(vec, Tile) or vec.cols < nslot * cols:
        raise CompileError(f"state_step: vec must hold {nslot} slots of {cols} words")
    if mode != "dot" and (x is None or x.cols != rows):
        raise CompileError(f"state_step: x must be a [{rows}] tile")
    d = I.state_desc(rows, cols, _DPROGS[mode], a_slot, gate, q=o is not None)
    if b.cfg.STREAM and I.stream_hw_cfg(d) is not None:
        # without k0 (never read), K0's word is the one before k1 (ks = 1)
        b.stream(d, state, vec, x, k1 if k0 is None else k0, o, zero, ks=ks,
                 k_off=-1 if k0 is None else 0)
        return
    slot = [vec[i * cols:(i + 1) * cols] for i in builtins.range(4) if (i + 1) * cols <= vec.cols]
    St = empty([rows * cols]).reshape(rows, cols) if tmp is None else tmp
    if zero:
        St.set(0.0)
    else:
        load(state, out=St)
    w = empty([rows])
    if uses_a:
        w.set(St @ slot[a_slot])
    if mode == "delta":
        w.set((x - w * k0) * k1)
    elif mode == "delta1":
        w.set((x - w) * k1)
    elif mode == "scale":
        w.set(x * k1)
    else:
        w.set(w * k1)
    outer(w, slot[1], acc=St, decay={"const": k0, "col": slot[3] if gate == "col" else None,
                                     "one": None}[gate])
    if o is not None:
        o.set(St @ slot[0])
    store(state, St)


# ---- control
def release(var) -> None:
    """The kernel no longer uses the run-time value var (compiler.RunVar): its argument
    registers may serve addresses from here on (Builder.release_arg)."""
    current().release_arg(var)


def range(n: int):  # noqa: A001
    """Hardware loop. The body is traced once; carry values across iterations with `.set()`."""
    b = current()
    n = int(n)
    if n <= 0:
        return
    loop = b.begin_loop(n)
    try:
        yield loop
    except GeneratorExit:       # closed while suspended: the build was given up inside the
        return                  # body (its loop stack is not this loop's to end)
    b.end_loop(loop)


def static_range(*args):
    return builtins.range(*args)


# ---- KV cache
def kv_append(kv: KVDesc, h: int, pos, k: Tile | None, v: Tile | None) -> None:
    """Quantize and append rows of k, v ([n, d]) at token position `pos` of KV head `h` (either
    may be None: only the other is appended).

    K rows are written token-major with per-block scales; V is written transposed (one byte per
    dimension, stride = the V^T tile, see KVDesc) with one scale per token.
    """
    b = current()
    pos = Affine.of(pos)
    if k is not None:
        kd = kv.k(h)
        b.store_quantized(k, kd.data + pos * kd.rs, kd.scale + pos * kd.srs, kd.rs, 1,
                          row_scale=False)
    if v is None:
        return
    vt = kv.vt(h)
    vs = kv.vscale(h)
    # a head half as wide as its padded row (LFM2: 64 of D = 128) writes only its own V^T rows:
    # P.V never reads the padding rows, and each transposed byte is a separate DRAM write
    half = v.cols == 2 * kv.dv
    # V^T is stored in tiles of vt.rs tokens (KVDesc): one store per tile the rows reach
    t, r0 = pos, 0
    while r0 < v.rows:
        r = tile_split(t, vt.rs)[1]
        if not isinstance(r, int) and v.rows > 1:
            raise CompileError("V^T rows at a run-time token: one row per append")
        n = min(v.rows - r0, vt.rs - r) if isinstance(r, int) else 1
        vr = v if n == v.rows else v[r0:r0 + n, :]
        b.store_quantized(vr, kv.vt_column(h, t), vs.base + t * 4, 1, vt.rs, row_scale=True,
                          half=half)
        t, r0 = t + n, r0 + n
