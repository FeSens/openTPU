"""openTPU compiler: traces kernels written in `opentpu.language` and lowers them to the ISA.

Model (Triton/Gluon flavoured):
  * A kernel is SPMD over slices: it is traced once per slice with a concrete `program_id()`.
  * Values live in TMEM as fp32 `Tile`s with an explicit row-major layout (base, shape, row
    stride). Views (`t[:, a:b]`, `t[i]`) alias the same buffer; broadcasts (`v[:, None]`,
    `v[None, :]`) map to the VPU's per-row / per-column operand modes.
  * DRAM operands are descriptors: `Tensor` (fp32) and `QTensor` (int8 rows + fp32 block scales,
    the MXU's streamed operand). Their addresses are affine in hardware-loop induction variables.
  * `range(n)` emits a hardware LOOP; its body is traced once. Loop-carried values are updated
    in place with `Tile.set()`. `static_range` is unrolled by Python.
  * Every layout transition is one instruction: DRAM->TMEM `LD`, TMEM->ACT RAM `QACT`,
    streamed operand `MM`, TMEM->DRAM `ST`/`QST`, sharded->replicated `GATHER`.
"""
from __future__ import annotations

import builtins
import itertools
import gc
import os
import sys
import threading
import weakref
from dataclasses import dataclass, field

import numpy as np

from . import isa as I
from .isasim import Config


class CompileError(RuntimeError):
    pass


REGION = 1 << 32                # a DRAM region (isa.py "DRAM address width"): 4 GiB


def region_of(addr: int, nbytes: int = 1) -> int:
    """The 4 GiB region of the DRAM range [addr, addr + nbytes); CompileError when it crosses a
    region boundary (no instruction's DRAM range may: isa.py "DRAM address width")."""
    r = int(addr) // REGION
    if nbytes > 0 and (int(addr) + int(nbytes) - 1) // REGION != r:
        raise CompileError(f"the DRAM range {int(addr):#x} + {int(nbytes):#x} crosses a 4 GiB "
                           f"boundary")
    return r


# =============================================================================== affine addresses
_loop_ids = itertools.count()


class Loop:
    """Hardware loop induction variable. `rcount`: a register whose value (read when the loop
    starts) is the trip count instead of `count` (a device-computed count; nothing may step
    with such a loop: its registers could not be reset)."""

    def __init__(self, count: int, rcount: int = 0):
        self.id = next(_loop_ids)
        self.count = count
        self.rcount = rcount

    def _aff(self) -> "Affine":
        return Affine(0, {self: 1})

    def __mul__(self, k):
        return self._aff() * k

    __rmul__ = __mul__

    def __add__(self, k):
        return self._aff() + k

    __radd__ = __add__

    def __repr__(self):
        return f"iv{self.id}"


class RunVar:
    """A value known only when the program runs (the token's position, its id). An address may
    add `c * var` for one (var, c): the host puts c times the value into one of the run's
    argument registers (docs/isa.md "Arguments"), one per distinct (var, c) of the program
    (Builder.run_args), so the same program serves every value. bound: the value is below it
    (a V^T append at a run-time token stays in its tile: tile_split)."""

    def __init__(self, name: str, bound: int | None = None):
        self.name, self.bound = name, bound

    def _aff(self) -> "Affine":
        return Affine(0, {self: 1})

    def __add__(self, o):
        return self._aff() + o

    __radd__ = __add__

    def __sub__(self, o):
        return self._aff() - o

    def __mul__(self, k):
        return self._aff() * k

    __rmul__ = __mul__

    def __repr__(self):
        return self.name


class DevVar(RunVar):
    """A value the program computes itself, in register `reg` (a Builder.scratch set by RLD:
    e.g. an expert's slot address, which WAITW copied from a directory, docs/offload.md). An
    address may add 1 * var; it then uses R[reg] as its base, like a run-time argument's
    register."""

    def __init__(self, name: str, reg: int, align: int = 1):
        super().__init__(name)
        self.reg, self.align = reg, align       # align: the value is a multiple of it


class Counter(DevVar):
    """A value the program steps itself (Builder.counter), in register `reg`: 0 where it is
    taken (before the loops it counts in), and Builder.step adds to it where the program says,
    e.g. in a loop of 0 or 1 iterations (tiled._lfm2_stack: an LFM2 stack's layers in one loop,
    the counter the bytes of the blocks before the layer's, each its kind's size). An address
    adds 1 * counter (its register as the base, as a DevVar's), and with it at most one run-time
    value of the host's (a register of their sum, stepped with it: Builder._derived)."""


ARG0 = 8                        # the run's arguments ARG0..7 are R8..R15 at the start


class Program(list):
    """A program's instructions (Builder.finish), with the run-time arguments it reads
    (Builder.run_args: [(RunVar, c)], argument k in R15 - k): opentpu.waits bounds them by
    their variables' bounds (an otpu_seq WAITC build's wait counts); and run_seen, every
    (RunVar, c) its addresses add (Builder.run_seen): with run words (a prompt run's state
    words, qwen3.RunWords) also those read only into a register that a loop steps, which are
    no argument, so the host writes every word the program reads (prefill._prepared)."""

    def __init__(self, instrs=(), run_args=(), run_seen=()):
        super().__init__(instrs)
        self.run_args = list(run_args)
        self.run_seen = list(run_seen)


class Words(np.ndarray):
    """A program's assembled words (uint32) with its run_args, as Program keeps them (opentpu.waits
    bounds the arguments with them on a WAITC device), and its wait entries when a compile
    worker process computed them (waits: None, or uint64 per instruction). Pickled with both."""

    run_args: list = []
    waits = None

    @staticmethod
    def of(words, run_args=(), waits=None) -> "Words":
        """words (any words; a Words keeps its entries unless `waits` replaces them)."""
        w = np.asarray(words, np.uint32).view(Words)
        w.run_args = list(run_args or ())
        w.waits = getattr(words, "waits", None) if waits is None else waits
        return w

    def __reduce__(self):
        fn, args, state = super().__reduce__()
        return fn, args, (state, self.run_args, self.waits)

    def __setstate__(self, s):
        state, self.run_args, self.waits = s
        super().__setstate__(state)


def arg_reg(k: int) -> int:
    """The register of a program's k-th run-time argument (Builder.run_args): from R15 down,
    as address registers are taken from R1 up."""
    return 15 - k


def arg_words(run_args: list, values: dict) -> list[int]:
    """The words ARG0..7 of a program's Builder.run_args for these values: argument k,
    c * var, in ARG(7 - k) (register R15 - k)."""
    w = [0] * (16 - ARG0)
    for k, (v, c) in enumerate(run_args):
        w[arg_reg(k) - ARG0] = (c * values[v.name]) & 0xFFFFFFFF
    return w


class Affine:
    def __init__(self, const: int = 0, terms: dict | None = None):
        self.const = int(const)
        self.terms = {l: c for l, c in (terms or {}).items() if c != 0}

    @staticmethod
    def of(x) -> "Affine":
        if isinstance(x, Affine):
            return x
        if isinstance(x, (Loop, RunVar)):
            return x._aff()
        if isinstance(x, (int, np.integer)):
            return Affine(int(x))
        raise CompileError(f"not an integer address expression: {x!r}")

    def __add__(self, o):
        o = Affine.of(o)
        t = dict(self.terms)
        for l, c in o.terms.items():
            t[l] = t.get(l, 0) + c
        return Affine(self.const + o.const, t)

    __radd__ = __add__

    def __sub__(self, o):
        return self + (Affine.of(o) * -1)

    def __rsub__(self, o):
        return Affine.of(o) - self

    def __mul__(self, k):
        if not isinstance(k, (int, np.integer)):
            raise CompileError("affine expressions can only be scaled by integers")
        return Affine(self.const * k, {l: c * k for l, c in self.terms.items()})

    __rmul__ = __mul__

    def div_exact(self, k: int) -> "Affine":
        if self.const % k or any(c % k for c in self.terms.values()):
            raise CompileError(f"address {self} is not a multiple of {k}")
        return Affine(self.const // k, {l: c // k for l, c in self.terms.items()})

    @property
    def is_static(self) -> bool:
        return not self.terms

    def static(self) -> int:
        if self.terms:
            raise CompileError("expected a compile-time constant, got a loop-dependent value")
        return self.const

    def __repr__(self):
        s = " + ".join(f"{c}*{l}" for l, c in self.terms.items())
        return f"({self.const}{' + ' + s if s else ''})"


def _static(x) -> int:
    return Affine.of(x).static()


# =============================================================================== descriptors
@dataclass
class Tensor:
    """fp32 tensor in the slice's DRAM. Addresses in bytes, strides in elements."""
    base: Affine
    shape: tuple
    strides: tuple
    host: object = None            # host argument this view belongs to (for outputs)
    offset: Affine = field(default_factory=Affine)   # element offset inside the host array

    def __getitem__(self, key) -> "Tensor":
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (len(self.shape) - len(key))
        base, off, shape, strides = Affine.of(self.base), Affine.of(self.offset), [], []
        for k, n, st in zip(key, self.shape, self.strides):
            if isinstance(k, slice):
                if k.step not in (None, 1):
                    raise CompileError("strided slices are not supported")
                start = Affine.of(0 if k.start is None else k.start)
                stop = Affine.of(n if k.stop is None else k.stop)
                size = (stop - start).static()
                base, off = base + start * (4 * st), off + start * st
                shape.append(size)
                strides.append(st)
            else:
                base, off = base + Affine.of(k) * (4 * st), off + Affine.of(k) * st
        return Tensor(base, tuple(shape), tuple(strides), self.host, off)


@dataclass
class QTensor:
    """A weight matrix [rows, cols] in an MXU format -- the MXU's streamed operand: element rows
    rs bytes apart and a scale row per row, srs bytes apart (docs/isa.md "Weight formats": the
    row's fp32 Sw, then its blocks' power-of-two offsets and multipliers).

    `scale is None` means unit scales (e.g. V^T, whose per-token scale is folded into P).
    `k0`: the matrix's first block within its scale rows (a column slice keeps the rows' scale
    address and moves k0: MM's k0). `rsc`: the scale rows are Sw alone (MM RSC: the K cache of
    one block a token, quantized by QST ROW).
    `parts`: the matrix is stored as separate column slices of width `pw` (each with its own
    contiguous rows and scale rows, so a column-slice MM streams contiguous scales and a part
    at a loop-dependent column is an address); it can then only be sliced into whole parts.
    `wf`: the element format (isa.W8 int8; W4I / W4F / WNV: 4-bit elements, D/2 bytes per
    D-block, docs/isa.md "Weight formats").
    `tiles`: (tb, tstride): the columns are stored in tiles of tb (each tile rows x tb, row
    stride rs = tb, tiles tstride bytes apart; int8, unscaled: the V^T cache). A column slice
    must stay within one tile.
    """
    data: Affine
    scale: Affine | None
    shape: tuple
    rs: int            # row stride, bytes
    srs: int           # scale row stride, bytes
    D: int
    parts: tuple | None = None
    pw: int = 0
    wf: int = 0
    tiles: tuple | None = None
    k0: int = 0
    rsc: bool = False

    @staticmethod
    def weight(data, scale, n: int, k: int, fmt: str, D: int, **kw) -> "QTensor":
        """The descriptor of an [n, k] matrix in MXU format `fmt` ("int8", "int4", "fp4",
        "nvfp4", opentpu/quant.py quantize_mxu's rows): element rows row_bytes apart, scale rows
        scale_bytes apart."""
        from . import quant as Q
        return QTensor(Affine.of(data), Affine.of(scale), (n, k), Q.row_bytes(k, fmt, D),
                       Q.scale_bytes(k, fmt, D), D, wf=Q.mxu_wf(fmt), **kw)

    def padded_cols(self, r0, nr: int, c0, nc: int, valid: int) -> "QTensor":
        """Rows [r0, r0+nr), columns [c0, c0+nc) of which the first `valid` are real (the rest
        pad an MM operand to whole D-blocks: the caller weights them 0). In a tiled matrix
        the real columns must stay within one tile; the padding may run past it (into the
        next row, then read with weight 0)."""
        if self.tiles is None:
            return self[r0:Affine.of(r0) + nr, c0:Affine.of(c0) + nc]
        tb, tstride = self.tiles
        q, r = tile_split(c0, tb)
        if r + valid > tb:
            raise CompileError(f"a column slice of a tiled matrix must stay within one tile "
                               f"of {tb} (columns {c0} + {valid})")
        return QTensor(self.data + q * tstride + Affine.of(r0) * self.rs + r, None, (nr, nc),
                       self.rs, 0, self.D)

    def __getitem__(self, key) -> "QTensor":
        if not isinstance(key, tuple):
            key = (key, slice(None))
        rk, ck = key
        if not (isinstance(rk, slice) and isinstance(ck, slice)):
            raise CompileError("QTensor supports 2-D slicing only")
        if self.parts is not None:
            c0 = _static(0 if ck.start is None else ck.start)
            c1 = _static(self.shape[1] if ck.stop is None else ck.stop)
            if c0 % self.pw or c1 - c0 != self.pw:
                raise CompileError(f"this matrix is stored in column parts of {self.pw}: "
                                   f"slice one part at a time")
            return self.parts[c0 // self.pw][rk, :]
        r0 = Affine.of(0 if rk.start is None else rk.start)
        r1 = Affine.of(self.shape[0] if rk.stop is None else rk.stop)
        c0 = Affine.of(0 if ck.start is None else ck.start)
        c1 = Affine.of(self.shape[1] if ck.stop is None else ck.stop)
        nr, nc = (r1 - r0).static(), (c1 - c0).static()
        if nc % self.D:
            raise CompileError("QTensor column slices must be multiples of D")
        if self.tiles is not None:
            return self.padded_cols(r0, nr, c0, nc, nc)
        if self.wf == I.W8:
            data = self.data + r0 * self.rs + c0
        else:                     # 4-bit rows stream whole D-byte chunks of two D-blocks
            data = self.data + r0 * self.rs + c0.div_exact(2 * self.D) * self.D
        scale, k0 = None, self.k0
        if self.scale is not None:
            # the rows' scale rows stay where they are (each starts with its Sw): a column
            # slice starts at block k0 of them, an immediate of the MM
            if not c0.is_static and not self.rsc:
                raise CompileError(f"a column slice of a scaled matrix at a run-time or "
                                   f"loop-dependent column ({c0}): its first block is MM's k0, "
                                   f"an immediate; store the matrix in column parts")
            scale = self.scale + r0 * self.srs
            if not self.rsc:
                k0 += c0.div_exact(self.D).static()
        return QTensor(data, scale, (nr, nc), self.rs, self.srs, self.D, wf=self.wf, k0=k0,
                       rsc=self.rsc)


VT_TILE = 256          # V^T cache tile (tokens): the attention block (qwen3.ATTN_BLOCK)
QACT_KB_MAX = 255      # QACT's blocks (the RTL's field)
QST_KB_MAX = 1024      # QST's blocks without ROW (the RTL's limit)


def vt_tile(cap: int) -> int:
    """Tokens per V^T tile for a cache of `cap` tokens (cap itself: one tile, untiled)."""
    return VT_TILE if cap % VT_TILE == 0 else cap


def tile_split(t: Affine, tb: int) -> tuple:
    """t = q * tb + r with r static (0 <= r < tb): (q, r). A loop-dependent t needs its loop
    terms to be multiples of tb. Run-time terms (RunVar) go to r, which is then an Affine: they
    need a bound that keeps r below tb."""
    t = Affine.of(t)
    run = {v: c for v, c in t.terms.items() if isinstance(v, RunVar)}
    if run:
        q, r = tile_split(Affine(t.const, {v: c for v, c in t.terms.items() if v not in run}),
                          tb)
        if any(v.bound is None or c < 0 for v, c in run.items()) or \
                r + sum(c * (v.bound - 1) for v, c in run.items()) >= tb:
            raise CompileError(f"token index {t} may cross a tile of {tb}")
        return q, Affine(r, run)
    r = t.const % tb
    try:
        return (t - r).div_exact(tb), r
    except CompileError:
        raise CompileError(f"token index {t} does not split into tiles of {tb}") from None


def vt_tiled(vt: np.ndarray) -> np.ndarray:
    """A V^T [d, cap] in its DRAM order (tiles of vt_tile(cap) tokens, see KVDesc), flat."""
    d, cap = vt.shape
    tb = vt_tile(cap)
    return np.ascontiguousarray(vt.reshape(d, cap // tb, tb).transpose(1, 0, 2)).reshape(-1)


def vt_untiled(flat: np.ndarray, d: int, cap: int) -> np.ndarray:
    """Inverse of vt_tiled: DRAM bytes -> V^T [d, cap]."""
    tb = vt_tile(cap)
    return flat.reshape(cap // tb, d, tb).transpose(1, 0, 2).reshape(d, cap)


def k_scale_bytes(d: int, D: int) -> int:
    """Bytes of a K cache token's scale row (KVDesc.k): 4 (its fp32 scale) for a head of one
    block, else an int8 scale row of d / D blocks."""
    from . import quant as Q
    return Q.scale_bytes(d, "int8", D, rsc=d == D)


class KVDesc:
    """Per-slice view of a KV cache whose heads are dealt round-robin over slices.

    Layout per head: K token-major int8 [cap, d] + a scale row per token (k_scale_bytes: its
    fp32 scale at d = D, else an int8 scale row); V^T dim-major int8
    [d, cap] in tiles of tb = vt_tile(cap) tokens: tile i holds tokens [i*tb, (i+1)*tb) of every
    row, [d, tb] contiguous (so P.V of an attention block streams one contiguous tile, and a
    token's column stays within a few DRAM rows at any capacity); one V scale per token [cap]
    (folded into P before P.V). With head_dim < D the rows are zero-padded to d = D and
    dv = head_dim: P.V reads only the first dv rows of V^T.
    """

    def __init__(self, heads: dict, cap: int, d: int, D: int, S: int, sid: int,
                 dv: int | None = None):
        self.heads, self.cap, self.d, self.D, self.S, self.sid = heads, cap, d, D, S, sid
        self.dv = d if dv is None else dv

    def owned_heads(self, n_kv_heads: int):
        return builtins.range(self.sid, n_kv_heads, self.S)

    def _h(self, h: int) -> dict:
        if h not in self.heads:
            raise CompileError(f"KV head {h} does not live on slice {self.sid}")
        return self.heads[h]

    def k(self, h: int) -> QTensor:
        """K [cap, d]: int8 rows, a token's scale row per row (k_scale_bytes): at d = D its
        fp32 scale alone (QST ROW, MM RSC), else an int8 scale row (QST: the token's S and an
        offset per block)."""
        e = self._h(h)
        return QTensor(Affine.of(e["k"]), Affine.of(e["ks"]), (self.cap, self.d), self.d,
                       k_scale_bytes(self.d, self.D), self.D, rsc=self.d == self.D)

    def vt(self, h: int) -> QTensor:
        e = self._h(h)
        tb = vt_tile(self.cap)
        return QTensor(Affine.of(e["vt"]), None, (self.dv, self.cap), tb, 0, self.D,
                       tiles=(tb, self.d * tb) if tb < self.cap else None)

    def vt_column(self, h: int, t) -> Affine:
        """The address of token t's byte in V^T row 0 (the next rows are vt(h).rs apart)."""
        tb = vt_tile(self.cap)
        q, r = tile_split(t, tb)
        return Affine.of(self._h(h)["vt"]) + q * (self.d * tb) + r

    def vscale(self, h: int) -> Tensor:
        e = self._h(h)
        return Tensor(Affine.of(e["vs"]), (self.cap,), (1,))


# =============================================================================== TMEM tiles
class _Buf:
    """Identity of one TMEM allocation. Every Tile (and view) of the allocation holds it; when
    the last one is gone the region is free again (see Builder.alloc)."""
    __slots__ = ("__weakref__",)


def _probe_fn(x):
    return sys.getrefcount(x)


class _Probe:
    def m(self, x):
        return sys.getrefcount(x)


# Reference count seen inside a one-argument function / method when the argument is an unnamed
# temporary (as numpy does for temporary elision). A larger count means the caller holds the
# value in a variable, so it may be used again and must not be overwritten in place.
TEMP_RC_FN = _probe_fn(object())
TEMP_RC_METHOD = _Probe().m(object())


def unnamed(rc: int, limit: int, x) -> bool:
    """`x` is an unnamed temporary, so the op may overwrite it in place. `rc` is its reference
    count taken in the ol function that received it (the first argument, so it is counted
    before `x` is pushed for this call), `limit` the matching TEMP_RC_*; call this from that
    function. Python 3.14 loads local variables as borrowed references, so a named
    argument counts no more than a temporary: the count still sees references held by
    containers and attributes, and the variables of the calling frames are checked here."""
    if rc > limit:
        return False
    f = sys._getframe(2)            # skip this function and the ol function that received x
    while f is not None and f.f_code is not _TRACE_CODE:     # frames above the trace hold no tiles
        if builtins.any(v is x for v in f.f_locals.values()):
            return False
        f = f.f_back
    return True


def _span(k: slice, n: int) -> tuple:
    """[a, e) of slice k over n (Python's bounds and negative indices); a view is contiguous
    rows or columns: a step other than 1, or an empty span, is refused."""
    a, e, step = k.indices(n)
    if step != 1:
        raise CompileError(f"strided tile views are not supported ({k})")
    if e <= a:
        raise CompileError(f"an empty tile view ({k} of {n})")
    return a, e


class Tile:
    """fp32 values in TMEM: `base` word address, `shape` (1-D or 2-D), row stride `rs`."""

    def __init__(self, b: "Builder", base: int, shape: tuple, rs: int | None = None,
                 buf: int | None = None):
        self.b, self.base, self.shape = b, base, tuple(int(s) for s in shape)
        self.rs = rs if rs is not None else self.shape[-1]
        self.buf = _Buf() if buf is None else buf

    @property
    def rows(self) -> int:
        return self.shape[0] if len(self.shape) == 2 else 1

    @property
    def cols(self) -> int:
        return self.shape[-1]

    @property
    def rowmax(self) -> "Tile":
        """The row maxima the MXU writes after this tile's last row (dot(..., rowmax=True))."""
        if len(self.shape) != 2 or getattr(self, "spare", 0) < self.rows:
            raise CompileError("rowmax: only for 2-D tiles from ol.empty/zeros/full or dot()")
        return Tile(self.b, self.base + self.rows * self.rs, (self.rows,), None, self.buf)

    @property
    def contiguous(self) -> bool:
        return len(self.shape) == 1 or self.rs == self.cols

    def __repr__(self):
        return f"Tile(@{self.base}, {self.shape}, rs={self.rs})"

    # ---- views
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(self.shape) == 1:
            if key == (slice(None), None):
                return Bcast(self, B_ROWVIEW)
            if key == (None, slice(None)):
                return Bcast(self, B_COLVIEW)
            if len(key) == 1 and isinstance(key[0], slice):
                a, e = _span(key[0], self.cols)
                return Tile(self.b, self.base + a, (e - a,), None, self.buf)
            raise CompileError(f"unsupported 1-D index {key}")
        rk = key[0]
        ck = key[1] if len(key) > 1 else slice(None)
        if isinstance(rk, (int, np.integer)) and isinstance(ck, slice):
            r = int(rk) + (self.rows if rk < 0 else 0)          # a negative row: from the end
            if not 0 <= r < self.rows:
                raise CompileError(f"row {rk} of a {self.rows}-row tile")
            a, e = _span(ck, self.cols)
            return Tile(self.b, self.base + r * self.rs + a, (e - a,), None, self.buf)
        if isinstance(rk, slice) and isinstance(ck, slice):
            r0, r1 = _span(rk, self.rows)
            c0, c1 = _span(ck, self.cols)
            return Tile(self.b, self.base + r0 * self.rs + c0, (r1 - r0, c1 - c0), self.rs,
                        self.buf)
        raise CompileError(f"unsupported 2-D index {key}")

    def column(self) -> "Tile":
        """A 1-D tile as an [n, 1] column (row stride 1): `v.column().set(t[:, c:c + 1])`
        copies column c of a 2-D tile into v (one VOP; a transpose column by column)."""
        if len(self.shape) != 1:
            raise CompileError(f"column(): {self} is not 1-D")
        return Tile(self.b, self.base, (self.cols, 1), 1, self.buf)

    def row_stride_view(self, r0: int, n: int, step: int) -> "Tile":
        """Rows r0, r0+step, ... (n of them) of a 2-D tile, as a [n, cols] view."""
        if len(self.shape) != 2 or r0 + (n - 1) * step >= self.rows:
            raise CompileError(f"row_stride_view({r0}, {n}, {step}) outside {self}")
        return Tile(self.b, self.base + r0 * self.rs, (n, self.cols), self.rs * step, self.buf)

    def reshape(self, rows: int, cols: int) -> "Tile":
        """The same words as a [rows, cols] tile (contiguous tiles only; one row always is)."""
        if not (self.contiguous or self.rows == 1) or rows * cols != self.rows * self.cols:
            raise CompileError(f"reshape: {self} is not {rows}x{cols} contiguous words")
        return Tile(self.b, self.base, (rows, cols), cols, self.buf)

    # ---- in-place update (loop-carried values)
    def set(self, value) -> "Tile":
        temp = unnamed(sys.getrefcount(value), TEMP_RC_METHOD, value)
        if temp and isinstance(value, Tile) and self.b.retarget_matvec(value, self):
            self.b.bump_version(self.buf)
            return self
        if temp and isinstance(value, Tile) and self.b.retarget(value, self):
            self.b.bump_version(self.buf)
            return self
        if isinstance(value, (int, float)):
            self.b.vop(I.V_FILL, self, None, float(value))
        elif isinstance(value, Bcast):
            self.b.vop(I.V_FILL, self, None, value)
        else:
            value = self.b.materialize(value, self.shape)
            if value.shape != self.shape:
                raise CompileError(f"set: shape {value.shape} != {self.shape}")
            self.b.vop(I.V_COPY, self, value, None)
        self.b.bump_version(self.buf)
        return self

    # ---- arithmetic
    def __add__(self, o):
        return self.b.binop(I.V_ADD, self, o)

    def __radd__(self, o):
        return self.b.binop(I.V_ADD, self, o)

    def __sub__(self, o):
        return self.b.binop(I.V_SUB, self, o)

    def __rsub__(self, o):
        return self.b.binop(I.V_RSUB, self, o)

    def __mul__(self, o):
        return self.b.binop(I.V_MUL, self, o)

    def __rmul__(self, o):
        return self.b.binop(I.V_MUL, self, o)

    def __truediv__(self, o):
        if isinstance(o, (int, float)):
            return self * (1.0 / float(o))
        if isinstance(o, Bcast):
            return self * Bcast(self.b.unop(I.V_RECIP, o.t), o.kind)
        return self * self.b.unop(I.V_RECIP, o)

    def __neg__(self):
        return self * -1.0

    def __matmul__(self, v):
        """x @ v for a 1-D tile v: the row dot products (one RDOT pass, no product tile)."""
        return self.b.matvec(self, v)


B_ROWVIEW, B_COLVIEW = "row", "col"


class Bcast:
    """A 1-D tile broadcast along columns (`v[:, None]`) or rows (`v[None, :]`)."""

    def __init__(self, t: Tile, kind: str):
        self.t, self.kind, self.b = t, kind, t.b

    def __add__(self, o):
        return self.b.binop(I.V_ADD, self, o)

    __radd__ = __add__

    def __sub__(self, o):
        return self.b.binop(I.V_SUB, self, o)

    def __rsub__(self, o):
        return self.b.binop(I.V_RSUB, self, o)

    def __mul__(self, o):
        return self.b.binop(I.V_MUL, self, o)

    __rmul__ = __mul__


class _Parts:
    """The shape of a stationary operand filled in parts (Builder.stationary), or of one whose
    source tile was let go once quantized (Builder.quantize(free=True))."""

    def __init__(self, b, rows: int, cols: int, shape: tuple | None = None):
        self.b, self.rows, self.cols = b, rows, cols
        self.shape = shape or (rows, cols)


class Stationary:
    """A tile quantized into ACT RAM blocks [ab, ab+KB), in chunks of up to ACT_ROWS rows.
    `pair`: its rows are also in ACT rows M..2M-1 (QACT DUP), so 4-bit MMs run at full rate
    (MM PAIR). `qranges`: the block ranges [k0, k1) each QACT wrote (None: one, all KB blocks):
    every row has one Sa per QACT, and an MM reads the blocks of one Sa only (docs/isa.md
    "MM"), so dot() runs an MM per range and adds them up (MM ACC)."""

    def __init__(self, src: Tile, chunks: list, KB: int, owners: list, pair: bool = False,
                 qranges: list | None = None):
        self.src, self.chunks, self.KB, self.owners = src, chunks, KB, owners
        self.pair = pair
        self.qranges = qranges

    def add_qrange(self, k0: int, k1: int) -> None:
        """Blocks [k0, k1) were written by one QACT (a part filled by quantize_into): they
        replace what other ranges they overlap."""
        rs = [(a, b) for a, b in (self.qranges or []) if b <= k0 or a >= k1]
        self.qranges = sorted(rs + [(k0, k1)])

    def ranges(self) -> list:
        """The QACT ranges [(k0, k1)] covering its KB blocks, in order (CompileError on a block
        no QACT wrote)."""
        rs = self.qranges if self.qranges is not None else [(0, self.KB)]
        k = 0
        for a, b in rs:
            if a != k:
                break
            k = b
        if k != self.KB:
            raise CompileError(f"a stationary operand's blocks {k}.. were not quantized "
                               f"(QACT ranges {rs} of {self.KB} blocks)")
        return list(rs)

    def blocks(self, k: int, kb: int = 1) -> "Stationary":
        """Its ACT RAM blocks [k, k + kb) as a stationary operand of their own (the same rows,
        K = kb * D): a dot with part of an operand quantized once (kernels/gather.py's one-hot
        blocks). One chunk of rows only; the view keeps this operand, and so its blocks,
        alive."""
        if len(self.chunks) != 1 or self.pair or not 0 <= k < k + kb <= self.KB:
            raise CompileError(f"blocks({k}, {kb}) of a {self.KB}-block operand in "
                               f"{len(self.chunks)} chunks")
        (ab, mc), = self.chunks
        D = self.src.b.cfg.D
        src = self.src[:, k * D:(k + kb) * D] if len(self.src.shape) == 2 else \
            self.src[k * D:(k + kb) * D]
        qr = None
        if self.qranges is not None:
            qr = [(max(a, k) - k, min(b, k + kb) - k) for a, b in self.qranges
                  if a < k + kb and b > k]
        v = Stationary(src, [(ab + k, mc)], kb, list(self.owners), qranges=qr)
        v.loop, v.parent = getattr(self, "loop", None), self
        return v


# =============================================================================== builder
@dataclass
class LoopBlock:
    loop: Loop
    items: list = field(default_factory=list)
    steps: list = field(default_factory=list)     # (register, increment per iteration)


_INTERNAL = (os.path.join("opentpu", "compiler.py"), os.path.join("opentpu", "language.py"))


def _user_frames(depth: int = 4) -> tuple:
    """The kernel source lines that led to the instruction being emitted, innermost first:
    ((file, line, function), ...). Profiles attribute cycles to these lines."""
    out = []
    f = sys._getframe(2)
    while f is not None and len(out) < depth:
        fn = f.f_code.co_filename
        if not fn.endswith(_INTERNAL):
            if f.f_code.co_name in ("trace", "compile_kernel", "profile", "launch"):
                break
            out.append((fn, f.f_lineno, f.f_code.co_name))
        f = f.f_back
    return tuple(out)


class Builder:
    """Per-slice trace state and instruction emission."""

    def __init__(self, cfg: Config, sid: int):
        self.cfg, self.sid = cfg, sid
        self.root: list = []
        self.stack: list = [self.root]
        self.loops: list[LoopBlock] = []
        self.tmem_next = 0
        self.fit = "next"               # TMEM placement: next-fit, or "best" (trace_fit's retry)
        self.act_next = 0
        self.act_owner = [None] * cfg.ACT_BLOCKS
        self.act_live = [None] * cfg.ACT_BLOCKS      # weakref to the Stationary holding a block
        self.regs: dict = {}          # live: frozenset(terms) -> register
        # Registers whose inner loops have ended but that still hold outer-loop terms:
        # (terms, loops live when retired, register). One may take a new key that adds terms
        # of loops begun since (those are 0 at the loop start and stepped from then on).
        self.spare_regs: list = []
        # free registers (value 0), each with the loops that were live when it was freed: it
        # may not take an address with terms of those loops (their step at the end of the
        # body would leave it non-zero for its earlier use in the next iteration)
        self.free_regs = [(r, frozenset()) for r in range(15, 0, -1)]
        self.used_regs: set = set()
        self.run_args: list = []      # (RunVar, coefficient) of argument k (arg_reg(k))
        self.run_seen: list = []      # every (RunVar, coefficient) an address adds
        # The generate loop (opentpu/llm/generate.py) keeps the run-time values in TMEM words,
        # {name: word}: a register that starts at c * var and steps with loops then starts with
        # an RLD MUL of its word, so only an address of c * var alone takes an argument register
        self.run_words: dict | None = None
        # arguments whose registers were given back (release_arg), and those registers while
        # they still hold the argument (zeroed when an address takes one); each one's release:
        # the program's last instruction then and the registers taken by then
        self.released: set = set()
        self.dirty: set = set()
        self.release_at: dict = {}
        self.moved: dict = {}         # argument k -> the released register it was copied to
        self.late_zero: set = set()   # address registers that hold an argument at the start:
                                      # zeroed at the release or before their loop, not there
        # A loop that may run 0 times (a device-computed count: a guard) skips the inits and
        # zeroings inside it: register -> the loops of that kind its value was last set in.
        # When one ends, such a register (spare, live or free) holds an unknown value: it is
        # free again, taken last and zeroed when taken (unset)
        self.set_in: dict = {}
        self.unset: set = set()
        self.scratch_loops: dict = {}  # scratch register -> the loops live when it was taken
        # register -> loops (ids) whose body zeroes it before any address takes it there (a
        # scratch taken before the loop, given back in its body: generate.py's token count)
        self.zero_head: dict = {}
        self.versions = weakref.WeakKeyDictionary()
        self.tmem_regions: list = []     # (base, end, weakref to the allocation's _Buf)
        self.tmem_peak = 0
        self.stores: list = []        # (host, element offsets, dram byte address)
        self.loop_uses: list = []     # stationaries used inside loops: (loop depth, stat)
        # STREAM descriptors live at the top of TMEM, filled once at the program's start; in a
        # TMEM above 64K words, at the top of its first 64K (STREAM's descriptor field is 16
        # bits), a region no tile takes
        self.desc_area = I.STREAM_DESC_AREA if cfg.STREAM else 0
        self.desc_at = min(cfg.TMEM_WORDS, 1 << 16) - self.desc_area
        if self.desc_area and cfg.TMEM_WORDS > 1 << 16:
            self.tmem_regions.append((self.desc_at, 1 << 16, lambda: True))
        self.descs: dict = {}         # descriptor words -> TMEM address
        self.desc_fills = 0

    # ---- emission
    def emit(self, ins: I.Instr) -> None:
        ins.src = _user_frames()
        self.stack[-1].append(ins)

    def addr(self, a) -> tuple[int, int]:
        a = Affine.of(a)
        if a.is_static:
            return 0, a.const
        run = [(v, c) for v, c in a.terms.items() if isinstance(v, RunVar)]
        for l in a.terms:
            if not isinstance(l, RunVar) and not any(lb.loop is l for lb in self.loops):
                raise CompileError(f"address uses {l} outside its loop")
        dev = None              # a Counter and a run-time value: the register of their sum
        if len(run) == 2 and any(isinstance(v, Counter) for v, _ in run):
            (k, kc), (v, c) = sorted(run, key=lambda t: not isinstance(t[0], Counter))
            if kc != 1 or isinstance(v, DevVar):
                raise CompileError(f"address {a}: a counter adds 1 * itself and one run-time "
                                   f"value of the host's")
            if (v, c) not in self.run_seen:
                self.run_seen.append((v, c))
            dev = arg = self._derived(k, v, c)
            if len(a.terms) == 2:
                return arg, a.const
        elif len(run) > 1:
            raise CompileError(f"address {a} adds more than one run-time value")
        words = self.run_words
        if run and dev is None:
            v, c = run[0]
            if isinstance(v, DevVar):               # the program's own register
                if c != 1:
                    raise CompileError(f"address {a}: a device value adds only 1 * {v}")
                arg = v.reg
                if len(a.terms) == 1:
                    return arg, a.const
            else:
                if run[0] not in self.run_seen:
                    self.run_seen.append(run[0])
                # c * var alone takes an argument register; with run_words, once all of them
                # are taken, an address register loaded from var's word where it is used
                room = run[0] in self.run_args or len(self.run_args) < 16 - ARG0
                if words is None or (len(a.terms) == 1 and room):
                    arg = self.arg_reg(v, c)
                    if len(a.terms) == 1:           # the argument register itself
                        return arg, a.const
        key = frozenset(a.terms.items())
        if key not in self.regs:
            r = self._spare_for(key)
            if r is None:
                r = self._free_for(key)
                if r is not None and run:          # it starts at the run-time value
                    (v, c), what = run[0], " + ".join(f"{c_}*{v_}" for v_, c_ in run)
                    if dev is not None or words is None or isinstance(v, DevVar):
                        init = I.addi(r, arg, 0, comment=f"{what} (argument)")
                    elif v.name not in words:
                        raise CompileError(f"run-time value {v}: no TMEM word (run_words)")
                    else:
                        init = I.rld(r, words[v.name], mul=int(c), comment=what)
                    self._init_before_loops(key, init)
                elif r in self.dirty or r in self.unset:      # it starts at 0
                    self._init_before_loops(key, I.li(r, 0, comment="released argument"
                                                      if r in self.dirty else "register reset"))
                if r in self.dirty:
                    self.dirty.discard(r)
                    self.late_zero.add(r)
                self.unset.discard(r)
            if r is None:
                raise CompileError("out of address registers")
            self.regs[key] = r
            self.used_regs.add(r)
        return self.regs[key], a.const

    # ---- DRAM addresses: {REG, (R + imm) mod 2^32} above DAW 32 (isa.py "DRAM address width")
    def region(self, a, span: int = 1, what: str = "DRAM access") -> int:
        """The 4 GiB region (an instruction's REG) of the DRAM address `a` (an Affine / int):
        every execution's range [const + least offset, const + greatest offset + span) must lie
        in it (span: the bytes one execution touches from the address; the offsets: a loop's
        c * (count - 1), a run-time value's c * (bound - 1)). The register of the address
        (Builder.addr) holds its terms only, and the immediate is the constant: the hardware's
        (R + imm) mod 2^32 is then the address's offset in its region, so the program's words
        are those of the same tensors 4 GiB lower but for REG.
        Region 0 holds everything a register or table word names absolutely (isa.py): a device
        value (DevVar, Counter: an absolute slot or row address, a count the program steps) and a
        run-time value without a bound address region 0 only (their range is unknown there: it
        is not checked, as at DAW 32). CompileError names the access when its range crosses a
        4 GiB boundary, leaves the DRAM's 2^DAW bytes, or needs an unchecked value above 4 GiB.
        At DAW 32 the address is as ever: region 0, nothing checked (the layout-only probes'
        images past 4 GiB compile as before)."""
        if self.cfg.DAW <= 32:
            return 0
        a = Affine.of(a)
        lo = hi = 0
        free = None                     # a value whose range is not known
        for v, c in a.terms.items():
            if isinstance(v, DevVar):
                free = free or v
                continue
            n = v.bound if isinstance(v, RunVar) else (None if v.rcount else v.count)
            if not n or n < 1:
                free = free or v
                continue
            e = c * (int(n) - 1)
            lo, hi = lo + min(e, 0), hi + max(e, 0)
        bot, top = a.const + lo, a.const + hi + max(int(span), 1)

        def fail(msg: str):
            where = f"{what} at {a.const:#x}" + "".join(f" + {c}*{v}" for v, c in a.terms.items())
            frame = next(iter(_user_frames(1)), None)           # the kernel line that asked
            if frame is not None:
                where += f" ({os.path.basename(frame[0])}:{frame[1]} in {frame[2]})"
            raise CompileError(f"{where}: {msg}")

        if top <= REGION:               # region 0 (a wrapped range too: the mask rows)
            return 0
        if free is not None:
            fail(f"{free} ({'a device value' if isinstance(free, DevVar) else 'no bound'}) "
                 f"addresses region 0 only, the range reaches {top:#x}")
        if bot < 0 or top > 1 << self.cfg.DAW:
            fail(f"the range {bot:#x}..{top:#x} leaves the DRAM's {self.cfg.DAW}-bit addresses")
        r = bot // REGION
        if (top - 1) // REGION != r:
            fail(f"the range {bot:#x}..{top:#x} crosses a 4 GiB boundary (regions {r} and "
                 f"{(top - 1) // REGION}): the layout keeps a tensor in one region, the run "
                 f"planners split loops at 4 GiB")
        return r

    def dram(self, a, span: int = 1, what: str = "DRAM access") -> tuple[int, int, int]:
        """(register, immediate, region) of the DRAM address `a`: Builder.addr and the region
        whose REG the instruction takes (Builder.region; span bytes from it, `what` names it)."""
        g = self.region(a, span, what)
        ra, imm = self.addr(a)
        return ra, imm, g

    def same_region(self, what: str, *regs) -> int:
        """The one region of an instruction's DRAM operands (regs: each operand's; None: the
        operand is not in DRAM): CompileError when they differ (isa.py: REG is the
        instruction's, so a matrix and its scale rows, a cache and its scale rows, a stream's
        source and destination share a region; the layout co-locates them)."""
        rs = {r for r in regs if r is not None}
        if len(rs) > 1:
            raise CompileError(f"{what}: its DRAM operands lie in regions {sorted(rs)}: one "
                               f"instruction's DRAM operands share its 4 GiB region (co-locate "
                               f"them)")
        return rs.pop() if rs else 0

    def arg_reg(self, var: RunVar, c: int) -> int:
        """The argument register that holds c * var (allocated on first use; it leaves the
        address registers)."""
        if (var, c) in self.released:
            raise CompileError(f"{c}*{var} used after its register was released")
        if (var, c) not in self.run_args:
            if len(self.run_args) == 16 - ARG0:
                raise CompileError(f"more than {16 - ARG0} run-time argument values")
            r = arg_reg(len(self.run_args))
            if r in self.used_regs:
                r = self._move_arg(r)
            self.free_regs = [(f, lt) for f, lt in self.free_regs if f != r]
            self.run_args.append((var, c))
        k = self.run_args.index((var, c))
        return self.moved.get(k, arg_reg(k))

    def _move_arg(self, r: int) -> int:
        """Argument register r was taken by an address after a release: at the release of a
        register still free (address registers take released ones last) that came before r was
        taken, copy the argument into it, and use that. Not at an earlier release: the register
        held its own argument until its release."""
        k = len(self.run_args)
        q = next((f for f, _ in self.free_regs
                  if f in self.dirty and r not in self.release_at[f][1]), None)
        if q is None:
            raise CompileError(f"out of address registers: R{r} is taken, argument {k} "
                               f"needs it")
        last = self.release_at[q][0]
        i = 0 if last is None else next(j for j, x in enumerate(self.root) if x is last) + 1
        self.root[i:i] = [I.addi(q, r, 0, comment=f"argument {k}"),
                          I.li(r, 0, comment="address register")]
        for f, (x, taken) in self.release_at.items():      # releases there: after the copy
            if x is last:
                self.release_at[f] = (self.root[i + 1], taken)
        self.dirty.discard(q)
        self.late_zero.add(r)
        self.moved[k] = q
        return q

    def release_arg(self, var: RunVar) -> None:
        """The program no longer uses var: its argument registers become address registers,
        taken only when no other register is free (a program that fits without them keeps its
        registers), and zeroed before their first loop. Outside every loop; in the generate
        loop's body (run_words: only c * var alone takes an argument register) it keeps them."""
        if self.loops and self.run_words is not None:
            return
        assert not self.loops, "release_arg inside a loop"
        at = (self.root[-1] if self.root else None, frozenset(self.used_regs))
        for k, (v, c) in enumerate(self.run_args):
            if v is var and (v, c) not in self.released:
                r = self.moved.get(k, arg_reg(k))
                self.released.add((v, c))
                self.dirty.add(r)
                self.release_at[r] = at
                self.free_regs.insert(0, (r, frozenset()))

    def _init_before_loops(self, key, ins: I.Instr) -> None:
        """Put `ins` before the outermost live loop among key's terms (the register must hold
        its run-time value when that loop starts; the loops step and reset it from there), or,
        for c * var alone (run_words: the argument registers all taken), right here."""
        for d, lb in enumerate(self.loops):
            if (lb.loop, dict(key).get(lb.loop)) in key:
                items = self.stack[d]
                items.insert(next(i for i, x in enumerate(items) if x is lb), ins)
                self._set(ins.rd, self.loops[:d])
                return
        if len(key) == 1 and self.run_words is not None:
            self.stack[-1].append(ins)
            self._set(ins.rd, self.loops)
            return
        raise CompileError("a run-time address outside its loops")    # (a bare argument)

    @staticmethod
    def _may_skip(loop: Loop) -> bool:
        return bool(loop.rcount) or not loop.count

    def _set(self, r: int, loops) -> None:
        """Register r's value is set by an instruction inside `loops` (LoopBlocks): note those
        that may run 0 times (set_in)."""
        g = tuple(lb.loop for lb in loops if self._may_skip(lb.loop))
        if g:
            self.set_in[r] = g
        else:
            self.set_in.pop(r, None)

    def _unset_after(self, loop: Loop) -> None:
        """`loop` (one that may run 0 times) has ended: the registers set inside it that the
        compiler counts on (a spare's run-time value, a live run-time key's, a free register's
        0) may hold what they held before it; they are free again, last in line, and zeroed
        when taken. A free register that was the loop's own count is 0 when it ran 0 times."""
        live = frozenset(id(lb.loop) for lb in self.loops)
        run = lambda k: all(isinstance(l, RunVar) for l, _ in k)        # noqa: E731
        for r, g in list(self.set_in.items()):
            if not any(x is loop for x in g):
                continue
            rest = tuple(x for x in g if x is not loop)
            spare = [s for s in self.spare_regs if s[2] == r]
            key = next((k for k, x in self.regs.items() if x == r), None)
            free = [t for f, t in self.free_regs if f == r]
            if spare:
                lt = live | spare[0][1]
                self.spare_regs.remove(spare[0])
            elif key is not None and run(key):
                lt = live
                del self.regs[key]
            elif free and r != loop.rcount and r not in self.unset:
                lt = live | free[0]
                self.free_regs = [(f, t) for f, t in self.free_regs if f != r]
            else:              # a scratch register (unscratch sets it), the loop's count, ...
                if rest:
                    self.set_in[r] = rest
                else:
                    del self.set_in[r]
                continue
            del self.set_in[r]
            self.free_regs.insert(0, (r, lt))
            self.unset.add(r)

    def _spare_for(self, key):
        """A retired register holding a subset of `key` whose missing terms all belong to loops
        begun after it was retired (a run-time term must be there already)."""
        for i, (terms, live_then, r) in enumerate(self.spare_regs):
            if terms <= key and not any(id(l) in live_then or isinstance(l, RunVar)
                                        for l, _ in key - terms):
                del self.spare_regs[i]
                return r
        return None

    def _free_for(self, key):
        """A free register that may hold `key`: freed while none of its loops was live, and for
        a key with a run-time term while none of the live loops was: its init leaves c * var in
        it after its loops, which an earlier use in a live loop's body would start from in the
        loop's next iteration (unless that body zeroes it first: zero_head)."""
        run = any(isinstance(l, RunVar) for l, _ in key)
        live = {id(lb.loop) for lb in self.loops}
        for i in builtins.range(len(self.free_regs) - 1, -1, -1):
            r, live_then = self.free_regs[i]
            if not any(id(l) in live_then for l, _ in key) and \
                    not (run and (live_then & live) - self.zero_head.get(r, set())):
                del self.free_regs[i]
                return r
        return None

    def begin_loop(self, n: int, rcount: int = 0) -> Loop:
        """A hardware loop of n iterations, or of R[rcount] (n = 0) read when it starts."""
        if len(self.loops) >= 4:
            raise CompileError("loop nesting deeper than 4")
        lb = LoopBlock(Loop(n, rcount))
        self.stack[-1].append(lb)
        self.stack.append(lb.items)
        self.loops.append(lb)
        return lb.loop

    def end_loop(self, loop: Loop) -> None:
        lb = self.loops.pop()
        assert lb.loop is loop
        self.stack.pop()
        # Registers tracking this loop step at the end of its body and are reset after it; a
        # register is freed once every loop it depends on has ended (its value is 0 again).
        live = {id(l.loop) for l in self.loops}
        retired = []
        for key, r in list(self.regs.items()):
            terms = dict(key)
            if loop in terms:
                lb.steps.append((r, terms[loop]))
                del self.regs[key]
                retired.append((frozenset((l, c) for l, c in key if l is not loop), r))
        for i, (terms, live_then, r) in enumerate(list(self.spare_regs)):
            t = dict(terms)
            if loop in t:
                lb.steps.append((r, t[loop]))
                self.spare_regs.remove((terms, live_then, r))
                retired.append((frozenset((l, c) for l, c in terms if l is not loop), r))
        # after the reset a retired register holds its outer-loop terms only
        for terms, r in retired:
            if not terms:
                self.free_regs.append((r, frozenset(live)))
            elif all(isinstance(l, RunVar) for l, _ in terms):     # the argument's value
                self.spare_regs.append((terms, frozenset(live), r))
            elif terms not in self.regs:
                self.regs[terms] = r
            else:
                self.spare_regs.append((terms, frozenset(live), r))
        if self._may_skip(loop):
            self._unset_after(loop)
        depth = len(self.loops)
        for d, stat in self.loop_uses:
            if d > depth and not self.stationary_valid(stat):
                raise CompileError("a stationary operand used inside a loop was overwritten "
                                   "inside that loop; quantize it inside the loop instead")
        self.loop_uses = [(d, s) for d, s in self.loop_uses if d <= depth]

    # ---- device-computed values (RLD)
    def scratch(self) -> int:
        """A free register (value 0) for a device-computed value; give it back with unscratch
        once its last use has been emitted."""
        if not self.free_regs:
            raise CompileError("out of registers for a device-computed value")
        r, _ = self.free_regs.pop()      # zeroed again before any later use of it
        if r in self.dirty:              # a released argument's: it still holds the argument
            self.dirty.discard(r)        # (from the start: not zeroed there)
            self.late_zero.add(r)
            self.emit(I.li(r, 0, comment="released argument"))
            self._set(r, self.loops)
        elif r in self.unset:            # set inside a loop that may not have run
            self.unset.discard(r)
            self.emit(I.li(r, 0, comment="register reset"))
            self._set(r, self.loops)
        self.used_regs.add(r)
        self.scratch_loops[r] = [lb.loop for lb in self.loops]
        return r

    def unscratch(self, r: int) -> None:
        """Zero a scratch register (free registers hold 0) and return it to the pool. Zeroed
        inside a loop that may run 0 times begun since it was taken (but the one it counts),
        it keeps its value when that loop does not run."""
        self.emit(I.li(r, 0, comment="scratch free"))
        since = self.scratch_loops.pop(r, [])
        if self.loops and not any(self.loops[-1].loop is x for x in since):
            self.zero_head.setdefault(r, set()).add(id(self.loops[-1].loop))
        new = [lb for lb in self.loops if not any(lb.loop is x for x in since)
               and lb.loop.rcount != r]
        g = self.set_in.get(r, ()) + tuple(lb.loop for lb in new if self._may_skip(lb.loop))
        if g:
            self.set_in[r] = g
        self.free_regs.append((r, frozenset(id(lb.loop) for lb in self.loops)))

    # ---- counters (Counter: a count the program steps where it says)
    def counter(self, name: str) -> Counter:
        """A Counter at 0 in a scratch register, taken here (before the loops it counts in):
        step() adds to it, end_counter() gives the register back (zeroed)."""
        c = Counter(name, self.scratch())
        c.depth, c.derived, c.steps = len(self.loops), {}, 0
        return c

    def _derived(self, k: Counter, v: RunVar, c: int) -> int:
        """The register of k + c * v (an address adds both: an LFM2 attention layer's KV rows at
        the token's position, a conv layer's ring row), stepped with k: it starts at c * v (an
        RLD MUL of v's word under run_words, bounded by v's bound; else a copy of v's argument
        register) before the outermost loop begun since k was taken. Taken on its first use,
        before k's first step, from the registers free since k was taken (one used and freed
        inside those loops would be clobbered)."""
        r = k.derived.get((v, c))
        if r is not None:
            return r
        if k.steps:
            raise CompileError(f"{k.name} + {c}*{v}: first used after {k.name}'s step")
        inner = {id(lb.loop) for lb in self.loops[k.depth:]}
        i = next((i for i in builtins.range(len(self.free_regs) - 1, -1, -1)
                  if not self.free_regs[i][1] & inner and self.free_regs[i][0] not in self.dirty),
                 None)
        if i is None:
            raise CompileError(f"out of registers for {k.name} + {c}*{v}")
        r, _ = self.free_regs.pop(i)
        self.unset.discard(r)
        self.used_regs.add(r)
        what = f"{k.name} + {c}*{v}"
        if self.run_words is None:
            init = I.addi(r, self.arg_reg(v, c), 0, comment=f"{what} (argument)")
        elif v.name not in self.run_words:
            raise CompileError(f"run-time value {v}: no TMEM word (run_words)")
        else:
            bound = (0, int(c) * (int(v.bound) - 1)) if v.bound and c >= 0 else None
            init = I.rld(r, self.run_words[v.name], mul=int(c), bound=bound, comment=what)
        if len(self.loops) > k.depth:
            lb, items = self.loops[k.depth], self.stack[k.depth]
            items.insert(next(j for j, x in enumerate(items) if x is lb), init)
        else:
            self.emit(init)
        k.derived[(v, c)] = r
        return r

    def step(self, c: Counter, by: int) -> None:
        """c += by: ADDI (to c's register and each of c and a run-time value's). The address
        registers that hold an address of its earlier value (spare ones: their loops have
        ended) are free again, zeroed when taken; one of a live loop's could not follow the
        step (the loop resets it to its init)."""
        self._drop(c)
        for r in (c.reg, *c.derived.values()):
            self.emit(I.addi(r, r, int(by), comment=f"{c.name} + {by}"))
        c.steps += 1

    def end_counter(self, c: Counter) -> None:
        """c's last use has been emitted: its registers (and those of addresses of it) free."""
        self._drop(c)
        for r in (c.reg, *c.derived.values()):
            self.unscratch(r)

    def _drop(self, c: Counter) -> None:
        """Before c changes: the spare registers holding an address of it are free (unset)."""
        if any(any(v is c for v, _ in key) for key in self.regs):
            raise CompileError(f"{c.name} stepped while a loop's address adds it")
        live = frozenset(id(lb.loop) for lb in self.loops)
        for s in [s for s in self.spare_regs if any(v is c for v, _ in s[0])]:
            self.spare_regs.remove(s)
            self.free_regs.insert(0, (s[2], s[1] | live))
            self.unset.add(s[2])

    def rld(self, r: int, t: "Tile", raw: bool = False, comment: str = "",
            bound: tuple | None = None) -> None:
        """R[r] = the one-word tile t as an int (RLD: f2i, or its bits with raw); bound: the
        values it may take (isa.rld)."""
        self.check_live(t)
        if t.rows * t.cols != 1:
            raise CompileError(f"rld: {t} is not one word")
        self.emit(I.rld(r, t.base, raw=raw, bound=bound, comment=comment or "rld"))

    def waitw(self, t: "Tile", dram: int, ref: int = 0, cmp: int = I.C_EQ, ra: int = 0,
              rc: int = 0, interval: int = 0, timeout: int = 0, comment: str = "") -> None:
        """Wait until the DRAM word at R[ra] + dram compares (cmp: I.C_EQ / C_NE / C_GE) with
        R[rc] + ref, then the one-word tile t = the word's bits (WAITW: a word the host writes,
        docs/isa.md; rld(raw=True) takes it into a register). The host's words are in region 0
        (isa.py "DRAM address width"); a static word above it (ra 0) takes its region."""
        self.check_live(t)
        if t.rows * t.cols != 1:
            raise CompileError(f"waitw: {t} is not one word")
        g = self.region(dram, 4, "waitw") if not ra else self.region(Affine(dram) + DevVar(
            "a register", ra), 4, "waitw")
        self.emit(I.waitw(dram, t.base, ref, cmp, ra=ra, rc=rc, interval=interval,
                          timeout=timeout, comment=comment or "waitw", reg=g))

    def argmax(self, x: "Tile", base: int = 0, out: "Tile | None" = None,
               rbase: int = 0) -> "Tile":
        """Per row of x its maximum and the index of its first column (+ base, + R[rbase]), as
        fp32 pairs: a [rows, 2] tile (a [2] one for one row), or into `out`."""
        x = self.materialize(x)
        self.check_live(x)
        if out is None:
            out = self.alloc((2,) if x.rows == 1 else (x.rows, 2))
        drs = out.rs if len(out.shape) == 2 else 2
        if out.rows != x.rows or out.cols != 2:
            raise CompileError(f"argmax of {x} into {out}")
        ars = x.rs if len(x.shape) == 2 else 0
        self.emit(I.argmax(out.base, x.base, x.rows, x.cols, drs, ars, base, rd=rbase))
        return out

    # ---- TMEM / ACT RAM allocation
    def alloc(self, shape, rs: int | None = None, spare: int = 0) -> Tile:
        """A new tile; `spare` extra words are reserved right after its last row (the MXU
        writes RMAX row maxima there)."""
        shape = tuple(int(x) for x in shape)
        rs = shape[-1] if rs is None else rs
        n = (shape[0] - 1) * rs + shape[-1] if len(shape) == 2 else shape[0]
        if spare:
            n = max(n, shape[0] * rs + spare) if len(shape) == 2 else n + spare
        base = self._tmem_fit(n)
        t = Tile(self, base, shape, rs)
        t.spare = spare
        self.tmem_regions.append((base, base + n, weakref.ref(t.buf)))
        self.tmem_regions.sort(key=lambda r: r[0])
        self.tmem_peak = max(self.tmem_peak, sum(e - b for b, e, _ in self.tmem_regions))
        return t

    def _tmem_fit(self, n: int) -> int:
        """Next-fit: the first gap of n words at or after the cursor (wrapping once). Regions
        are free once no tile references them; reusing one is always correct (the scoreboard
        orders the accesses), and next-fit keeps reuse far from recent, still-running work."""
        W = self.cfg.TMEM_WORDS - (self.desc_area if self.cfg.TMEM_WORDS <= 1 << 16 else 0)
        for attempt in range(2):
            self.tmem_regions = [r for r in self.tmem_regions if r[2]() is not None]
            if self.fit == "best":          # the smallest gap that holds n words
                best, a = None, 0
                for b, e, _ in self.tmem_regions + [(W, W, None)]:
                    if b - a >= n and (best is None or b - a < best[1]):
                        best = (a, b - a)
                    a = max(a, e)
                if best is not None:
                    self.tmem_next = best[0] + n
                    return best[0]
                gc.collect()
                continue
            for start in (self.tmem_next, 0):
                a = start
                for b, e, _ in self.tmem_regions:
                    if e <= a:
                        continue
                    if b - a >= n:
                        break
                    a = max(a, e)
                if a + n <= W:
                    self.tmem_next = a + n
                    return a
            gc.collect()       # dead tiles in reference cycles still hold their regions
        live = sum(e - b for b, e, _ in self.tmem_regions)
        raise CompileError(f"TMEM exhausted: a tile of {n} words, {live} of {W} in use "
                           f"(TMEM_WORDS {self.cfg.TMEM_WORDS})")

    def act_alloc(self, KB: int) -> tuple[int, object]:
        """KB contiguous ACT RAM blocks not held by a live Stationary, searched round-robin
        from the last allocation (so recently freed blocks, which an MM may still be reading,
        are reused last)."""
        nb = self.cfg.ACT_BLOCKS
        if KB > nb:
            raise CompileError("operand does not fit ACT RAM")

        def free(k):
            ref = self.act_live[k]
            return ref is None or ref() is None

        for i in builtins.range(nb):
            ab = (self.act_next + i) % nb
            if ab + KB <= nb and all(free(k) for k in builtins.range(ab, ab + KB)):
                break
        else:
            raise CompileError("ACT RAM full: too many live quantized operands")
        self.act_next = (ab + KB) % nb
        owner = object()
        for k in builtins.range(ab, ab + KB):
            self.act_owner[k] = owner
            self.act_live[k] = None
        return ab, owner

    def stationary_valid(self, st: Stationary) -> bool:
        for (ab, _), owner in zip(st.chunks, st.owners):
            if any(self.act_owner[k] is not owner for k in builtins.range(ab, ab + st.KB)):
                return False
        return True

    def bump_version(self, buf: int) -> None:
        self.versions[buf] = self.versions.get(buf, 0) + 1

    # ---- VPU lowering
    def materialize(self, x, shape=None) -> Tile:
        if isinstance(x, Tile):
            return x
        if isinstance(x, Bcast):
            if shape is None:
                raise CompileError("cannot materialize a broadcast without a shape")
            out = self.alloc(shape)
            self.vop(I.V_FILL, out, None, x)
            return out
        if isinstance(x, (int, float)):
            out = self.alloc(shape)
            self.vop(I.V_FILL, out, None, float(x))
            return out
        raise CompileError(f"cannot use {x!r} as a tile")

    def vop(self, func: int, dst: Tile, a: Tile | None, b, at=None) -> None:
        """Emit dst = func(a, b) over dst's shape; b may be a Tile, Bcast or float. at: a TMEM
        word offset with loop terms added to dst's address (VOP's dst = R[ra] + w1: rows of a
        buffer the loop walks, the looped attention's outputs, tiled._attention_loop)."""
        self.check_live(dst, a, b, b.t if isinstance(b, Bcast) else None)
        rows, cols = dst.rows, dst.cols
        a_base, ars = (a.base, a.rs if len(a.shape) == 2 else 0) if a is not None else (dst.base, 0)
        if a is not None and (a.rows, a.cols) != (rows, cols):
            raise CompileError(f"shape mismatch {a.shape} vs {dst.shape}")
        bmode, b_base, brs, imm = I.B_SCALAR, 0, 0, 0.0
        if isinstance(b, (int, float)):
            imm = float(b)
        elif isinstance(b, Tile) and b.shape == (1,) and (rows, cols) != (1, 1):
            bmode, b_base, brs = I.B_ROW, b.base, 0          # a TMEM scalar: every row reads T[b]
        elif isinstance(b, Tile):
            if (b.rows, b.cols) != (rows, cols):
                raise CompileError(f"shape mismatch {b.shape} vs {dst.shape}")
            bmode, b_base, brs = I.B_FULL, b.base, (b.rs if len(b.shape) == 2 else 0)
        elif isinstance(b, Bcast):
            v = b.t
            if b.kind == B_ROWVIEW:
                if v.cols != rows:
                    raise CompileError(f"row broadcast of length {v.cols} over {rows} rows")
                bmode, b_base, brs = I.B_ROW, v.base, 1
            else:
                if v.cols != cols:
                    raise CompileError(f"column broadcast of length {v.cols} over {cols} cols")
                bmode, b_base = I.B_COL, v.base
        elif b is not None:
            raise CompileError(f"bad operand {b!r}")
        drs = dst.rs if len(dst.shape) == 2 else 0
        ra, d_base = (0, dst.base) if at is None else self.addr(Affine(dst.base) + Affine.of(at))
        self.emit(I.vop(func, d_base, a_base, b_base, rows, cols, drs, ars, brs, bmode, imm,
                        ra=ra, comment=I.VFUNCS[func]))

    def binop(self, func: int, x, y):
        swap = {I.V_ADD: I.V_ADD, I.V_MUL: I.V_MUL, I.V_MAX: I.V_MAX, I.V_MIN: I.V_MIN,
                I.V_SUB: I.V_RSUB, I.V_RSUB: I.V_SUB}
        if not isinstance(x, Tile) or (x.shape == (1,) and isinstance(y, Tile)
                                       and y.shape != (1,)):
            if isinstance(y, Tile):
                x, y, func = y, x, swap[func]
            else:
                raise CompileError("an elementwise op needs at least one full tile operand")
        self.check_live(x, y)
        out = self.alloc(x.shape)
        self.vop(func, out, x, y)
        self.stack[-1][-1].out_ref = weakref.ref(out)
        out.fresh = True
        return out

    def unop(self, func: int, x, temp: bool = False) -> Tile:
        x = self.materialize(x)
        self.check_live(x)
        if func == I.V_EXP2 and temp and self.fuse_exp2_sub(x):
            out = Tile(self, x.base, x.shape, x.rs, x.buf)
            out.fresh = True
            x.dead = True
            return out
        out = self.alloc(x.shape)
        self.vop(func, out, x, None)
        out.fresh = True
        return out

    # ---- peephole optimizations on the instruction just emitted
    def _last_vop_writing(self, t: Tile):
        body = self.stack[-1]
        if not body or not isinstance(body[-1], I.Instr) or body[-1].op != I.VOP:
            return None
        ins = body[-1]
        if ins.w[0] != t.base or (ins.w[3] & 0xFFFF) != t.rows or (ins.w[3] >> 16) != t.cols:
            return None
        return ins

    def fuse_exp2_sub(self, x: Tile) -> bool:
        """exp2(a - b) where `a - b` was just computed into a fresh temp: rewrite that SUB as
        EXP2SUB in place (the temp now holds the exponential)."""
        ins = self._last_vop_writing(x)
        if ins is None or ((ins.w[5] >> 16) & 0xFF) != I.V_SUB or not getattr(x, "fresh", False):
            return False
        ins.w[5] = (ins.w[5] & ~(0xFF << 16)) | (I.V_EXP2SUB << 16)
        ins.comment = "exp2sub"
        return True

    def retarget(self, value: Tile, dst: Tile) -> bool:
        """`dst.set(value)`: if `value` is a fresh temp written by the last instruction, make
        that instruction write `dst` directly (no copy). The temp is then dead."""
        if not getattr(value, "fresh", False) or value.shape != dst.shape:
            return False
        ins = self._last_vop_writing(value)
        if ins is None or (ins.w[5] >> 16) & 0xFF in I.REDUCE:
            return False
        rows, cols = dst.rows, dst.cols
        drs = dst.rs if len(dst.shape) == 2 else 0
        dset = {dst.base + r * drs + c for r in builtins.range(rows) for c in builtins.range(cols)}
        a, b, bmode = ins.w[1], ins.w[2], (ins.w[5] >> 24) & 3
        ars, brs = ins.w[4] >> 16, ins.w[5] & 0xFFFF
        func = (ins.w[5] >> 16) & 0xFF
        reads = []
        if func != I.V_FILL:
            reads.append({a + r * ars + c for r in builtins.range(rows) for c in builtins.range(cols)})
        if func in I.BINARY and bmode != I.B_SCALAR:
            if bmode == I.B_FULL:
                reads.append({b + r * brs + c for r in builtins.range(rows) for c in builtins.range(cols)})
            elif bmode == I.B_ROW:
                reads.append({b + r * brs for r in builtins.range(rows)})
            else:
                reads.append({b + c for c in builtins.range(cols)})
        exact = {a + r * ars + c for r in builtins.range(rows) for c in builtins.range(cols)} == dset \
            and ars == drs
        for rd in reads:
            if rd & dset and not (exact and rd is reads[0]):
                return False
        ins.w[0] = dst.base
        ins.w[4] = (ins.w[4] & ~0xFFFF) | drs
        value.dead = True
        return True

    def check_live(self, *tiles) -> None:
        for t in tiles:
            if isinstance(t, Tile) and getattr(t, "dead", False):
                raise CompileError("tile was moved into another tile by .set() and is dead")

    def reduce(self, func: int, x: Tile, axis: int, temp: bool = False) -> Tile:
        x = self.materialize(x)
        self.check_live(x)
        if axis not in (-1, len(x.shape) - 1):
            raise CompileError("reductions are along the last axis only")
        if func == I.V_RSUM and temp:
            fused = self.fuse_sum_products(x)
            if fused is not None:
                return fused
        if func == I.V_RMAX:
            fused = self.fuse_mm_rmax(x)
            if fused is not None:
                return fused
        out = self.alloc((x.rows,))
        ars = x.rs if len(x.shape) == 2 else 0
        self.emit(I.vop(func, out.base, x.base, 0, x.rows, x.cols, 1, ars, 0, I.B_FULL, 0.0,
                        comment=I.VFUNCS[func]))
        return out

    # ---- data movement
    def load(self, t: Tensor, out: Tile | None = None) -> Tile:
        if not t.shape:
            raise CompileError("cannot load a scalar")
        # into a tile whose rows have the source's stride (a padded score buffer and its mask
        # tile, attention.Additive): one LD, the pad words between the rows included
        strided = out is not None and len(t.shape) == 2 and t.shape[0] > 1 and \
            t.strides[1] == 1 and out.rs == t.strides[0] > t.shape[1]
        # rows apart in the source go row by row into out's rows, whatever its row stride; so
        # do a source's back-to-back rows into a tile with an odd row stride (ol.empty's: an
        # MXU accumulator's results of a streamed row in distinct TMEM banks)
        by_row = len(t.shape) == 2 and (t.strides[0] != t.shape[1] or (
            out is not None and not strided and out.rs != t.shape[1]))
        if out is None:
            out = self.alloc(t.shape)
        else:
            self.check_live(out)
            if out.shape != tuple(t.shape) or not (out.contiguous or strided or by_row):
                raise CompileError(f"load: out {out} is not a contiguous {t.shape} tile")
            self.bump_version(out.buf)
        if strided:
            n = (t.shape[0] - 1) * out.rs + t.shape[1]
            ra, imm, g = self.dram(t.base, 4 * n, "load rows")
            self.emit(I.ld(imm, out.base, n, ra=ra, comment="load rows", reg=g))
            return out
        if len(t.shape) == 1:
            if t.strides[0] != 1:
                raise CompileError("strided 1-D loads are not supported")
            ra, imm, g = self.dram(t.base, 4 * t.shape[0], "load")
            self.emit(I.ld(imm, out.base, t.shape[0], ra=ra, comment="load", reg=g))
            return out
        if len(t.shape) != 2 or t.strides[1] != 1:
            raise CompileError("loads must be 1-D or row-major 2-D")
        if not by_row:
            n = t.shape[0] * t.shape[1]
            ra, imm, g = self.dram(t.base, 4 * n, "load")
            self.emit(I.ld(imm, out.base, n, ra=ra, comment="load", reg=g))
        else:
            for r in builtins.range(t.shape[0]):
                ra, imm, g = self.dram(t.base + r * 4 * t.strides[0], 4 * t.shape[1], "load row")
                self.emit(I.ld(imm, out.base + r * out.rs, t.shape[1], ra=ra, comment="load row",
                               reg=g))
        return out

    def store(self, t: Tensor, x) -> None:
        x = self.materialize(x, t.shape)
        self.check_live(x)
        if (x.rows, x.cols) != ((t.shape[0] if len(t.shape) == 2 else 1), t.shape[-1]):
            raise CompileError(f"store shape {x.shape} into {t.shape}")
        rows = t.shape[0] if len(t.shape) == 2 else 1
        rstride = t.strides[0] if len(t.shape) == 2 else 0
        if len(t.shape) == 2 and t.strides[1] != 1:
            raise CompileError("stores must be row-major")
        merged = rows > 1 and x.contiguous and rstride == x.cols
        for r in builtins.range(1 if merged else rows):
            base = t.base + r * 4 * rstride
            n = rows * x.cols if merged else x.cols
            ra, imm, g = self.dram(base, 4 * n, "store")
            self.emit(I.st(imm, x.base + r * x.rs, n, ra=ra, comment="store", reg=g))
            if t.host is not None:
                if not Affine.of(t.offset).is_static or not Affine.of(base).is_static:
                    raise CompileError("stores to kernel outputs must use static addresses")
                off = Affine.of(t.offset).static() + r * rstride
                self.stores.append((t.host, off, n))

    def quantize(self, x: Tile, temp: bool = False, free: bool = False,
                 rscale: Tile | None = None, cscale: Tile | None = None) -> Stationary:
        """QACT x's rows into ACT RAM blocks. free: the stationary does not keep x (its TMEM is
        free once the QACTs have read it), only its shape: operands kept in ACT RAM while TMEM
        is reused (a tiled prefill's MLP activations). rscale [rows] / cscale [cols]: quantize
        (x * rscale[:, None]) * cscale[None, :] (QACT RSCALE / CSCALE), the instruction the
        peephole makes of that expression, without its product tiles."""
        x = self.materialize(x)
        self.check_live(x)
        D = self.cfg.D
        if x.cols % D:
            raise CompileError(f"reduction dim {x.cols} must be a multiple of D={D}")
        KB = x.cols // D
        if KB > QACT_KB_MAX:
            raise CompileError(f"QACT of {KB} blocks: at most {QACT_KB_MAX} (its field); quantize "
                               f"the operand in ranges")
        src, rs, cs, rsc = x.base, (x.rs if len(x.shape) == 2 else x.cols), None, None
        fused = self.fuse_cscale(x) if temp and rscale is None and cscale is None else None
        if fused is not None:
            src, rs, cs, rsc = fused
        if rscale is not None or cscale is not None:
            self.check_live(rscale, cscale)
            if (rscale is not None and rscale.shape != (x.rows,)) or \
                    (cscale is not None and cscale.shape != (x.cols,)):
                raise CompileError(f"quantize: scales {rscale and rscale.shape} / "
                                   f"{cscale and cscale.shape} for a {x.shape} tile")
            cs = None if cscale is None else cscale.base
            rsc = None if rscale is None else rscale.base
        chunks, owners = [], []
        # column reuse: a tile of at most MCOLS/2 rows is written twice (no extra cycles), so a
        # 4-bit MM can feed the odd K-blocks to the second half of the columns. A taller tile
        # (up to ACT_ROWS rows) replays the streamed chunks instead: PAIR would not be faster
        pair = self.cfg.PAIR and 2 * x.rows <= self.cfg.MCOLS
        st = Stationary(x, chunks, KB, owners, pair)
        for m0 in builtins.range(0, x.rows, self.cfg.act_rows):
            mc = min(self.cfg.act_rows, x.rows - m0)
            ab, owner = self.act_alloc(KB)
            for k in builtins.range(ab, ab + KB):   # live now: the next chunk must not take it
                self.act_live[k] = weakref.ref(st)
            self.emit(I.qact(src + m0 * rs, mc, ab, KB, rs, cscale=cs,
                             rscale=None if rsc is None else rsc + m0, dup=pair,
                             comment="quantize -> ACT" + (" x col scale" if cs is not None else "")
                             + (" x row scale" if rsc is not None else "")
                             + (" dup" if pair else "")))
            chunks.append((ab, mc))
            owners.append(owner)
        st.loop = self.loops[-1].loop if self.loops else None     # created in this loop body
        if free:
            st.src = _Parts(self, x.rows, x.cols, x.shape)
        return st

    def stationary(self, rows: int, cols: int) -> Stationary:
        """ACT RAM blocks for a stationary operand of `rows` rows (at most ACT_ROWS) and `cols`
        columns, filled in parts of whole blocks by quantize_into (a tiled prefill's attention
        output: a KV head's columns at a time). Each part's QACT gives its blocks a row scale of
        their own (docs/isa.md "QACT"), so a dot over the operand runs an MM per part
        (Stationary.qranges) and adds them up (MM ACC)."""
        D = self.cfg.D
        if cols % D:
            raise CompileError(f"reduction dim {cols} must be a multiple of D={D}")
        if rows > self.cfg.act_rows:
            raise CompileError(f"a stationary of {rows} rows: ACT RAM holds {self.cfg.act_rows}")
        KB = cols // D
        pair = self.cfg.PAIR and 2 * rows <= self.cfg.MCOLS
        st = Stationary(_Parts(self, rows, cols), [], KB, [], pair, qranges=[])
        ab, owner = self.act_alloc(KB)
        for k in builtins.range(ab, ab + KB):
            self.act_live[k] = weakref.ref(st)
        st.chunks.append((ab, rows))
        st.owners.append(owner)
        st.loop = self.loops[-1].loop if self.loops else None
        return st

    def quantize_into(self, st: Stationary, x: Tile, col: int, cscale: Tile | None = None,
                      at=None) -> None:
        """Quantize the rows of x into the blocks of `st` (from stationary()) from column
        `col` on (QACT at the part's first block). cscale [cols]: x * cscale[None, :] (QACT
        CSCALE, as quantize's). at: a TMEM word offset with loop terms added to x's address
        (QACT's src = R[ra] + w1: x's rows in a buffer the loop walks, the looped attention's
        query tiles, tiled._attend_loop); x is then those rows at offset 0."""
        x = self.materialize(x)
        self.check_live(x, cscale)
        D = self.cfg.D
        if not isinstance(st.src, _Parts):
            raise CompileError("quantize_into: a stationary from stationary()")
        if x.rows != st.src.rows or x.cols % D or col % D or col + x.cols > st.KB * D:
            raise CompileError(f"quantize_into: a {x.shape} tile at column {col} of a "
                               f"{st.src.shape} stationary")
        if cscale is not None and cscale.shape != (x.cols,):
            raise CompileError(f"quantize_into: a column scale of {cscale.shape} for {x.shape}")
        if not self.stationary_valid(st):
            raise CompileError("stationary operand was overwritten in ACT RAM")
        (ab, mc), = st.chunks
        rs = x.rs if len(x.shape) == 2 else x.cols
        ra, src = (0, x.base) if at is None else self.addr(Affine(x.base) + Affine.of(at))
        self.emit(I.qact(src, mc, ab + col // D, x.cols // D, rs, dup=st.pair, ra=ra,
                         cscale=None if cscale is None else cscale.base,
                         comment="quantize -> ACT (part)" + (" dup" if st.pair else "")
                         + (" x col scale" if cscale is not None else "")))
        st.add_qrange(col // D, (col + x.cols) // D)

    def fuse_cscale(self, x: Tile):
        """quantize(a * v[None, :]) with the product an unnamed temp just computed: drop the
        multiply and let QACT scale the columns (CSCALE). Returns (src, row stride, scale)."""
        ins = self._last_vop_writing(x)
        if ins is None or not getattr(x, "fresh", False):
            return None
        func, bmode = (ins.w[5] >> 16) & 0xFF, (ins.w[5] >> 24) & 3
        if func != I.V_MUL or bmode != I.B_COL:
            return None
        rows = ins.w[3] & 0xFFFF
        ars = ins.w[4] >> 16
        if rows > 1 and ars == 0:
            return None
        self.stack[-1].pop()
        x.dead = True
        src, srs, rsc = ins.w[1], (ars if rows > 1 else x.cols), None
        # (a * r[:, None]) * g[None, :] where a * r[:, None] is a dead temp: RSCALE as well
        body = self.stack[-1]
        prev = body[-1] if body and isinstance(body[-1], I.Instr) else None
        ref = getattr(prev, "out_ref", None)
        if (prev is not None and prev.op == I.VOP and ref is not None and ref() is None
                and prev.w[0] == src and (prev.w[3] & 0xFFFF) == rows
                and (prev.w[3] >> 16) == x.cols and (rows == 1 or (prev.w[4] & 0xFFFF) == srs)
                and (prev.w[5] >> 16) & 0xFF == I.V_MUL and (prev.w[5] >> 24) & 3 == I.B_ROW
                and (prev.w[5] & 0xFFFF) == 1):
            body.pop()
            p_ars = prev.w[4] >> 16
            src, srs, rsc = prev.w[1], (p_ars if rows > 1 else x.cols), prev.w[2]
        return src, srs, ins.w[2], rsc

    def fuse_sum_products(self, x: Tile):
        """sum(a * b) with the product an unnamed temp just computed: drop the multiply and
        reduce with RDOT over a and b (any broadcast of b), or RSSQ when b is a itself."""
        ins = self._last_vop_writing(x)
        if ins is None or not getattr(x, "fresh", False):
            return None
        func, bmode = (ins.w[5] >> 16) & 0xFF, (ins.w[5] >> 24) & 3
        if func != I.V_MUL:
            return None
        rows, cols = ins.w[3] & 0xFFFF, ins.w[3] >> 16
        ars, brs = ins.w[4] >> 16, ins.w[5] & 0xFFFF
        self.stack[-1].pop()
        x.dead = True
        out = self.alloc((rows,))
        square = bmode == I.B_FULL and ins.w[1] == ins.w[2] and (rows == 1 or ars == brs)
        if square:
            self.emit(I.vop(I.V_RSSQ, out.base, ins.w[1], 0, rows, cols, 1, ars, 0, I.B_FULL,
                            0.0, comment="rssq"))
        else:
            self.emit(I.Instr(I.VOP, rb=ins.rb, rc=ins.rc, w=[
                out.base, ins.w[1], ins.w[2], ins.w[3], 1 | (ars << 16),
                brs | (I.V_RDOT << 16) | (bmode << 24), ins.w[6]], comment="rdot"))
        return out

    def matvec(self, x: Tile, v) -> Tile:
        if not isinstance(v, Tile) or len(v.shape) != 1 or v.cols != x.cols:
            raise CompileError(f"{x.shape} @ v needs a 1-D tile v of length {x.cols}")
        self.check_live(x, v)
        out = self.alloc((x.rows,))
        ars = x.rs if len(x.shape) == 2 else 0
        self.emit(I.vop(I.V_RDOT, out.base, x.base, v.base, x.rows, x.cols, 1, ars, 0, I.B_COL,
                        comment="rdot"))
        out.matvec = self.stack[-1][-1]
        return out

    def retarget_matvec(self, value: Tile, dst: Tile) -> bool:
        """`dst.set(x @ v)` right after the RDOT: it writes dst (1-D) directly."""
        ins = getattr(value, "matvec", None)
        body = self.stack[-1]
        if ins is None or not body or body[-1] is not ins or len(dst.shape) != 1 \
                or dst.shape != value.shape:
            return False
        rows, cols, ars = ins.w[3] & 0xFFFF, ins.w[3] >> 16, ins.w[4] >> 16
        lo, hi = dst.base, dst.base + dst.cols
        if (ins.w[1] < hi and lo < ins.w[1] + (rows - 1) * ars + cols) or \
                (ins.w[2] < hi and lo < ins.w[2] + cols):
            return False
        ins.w[0] = dst.base
        value.dead = True
        return True

    def outer(self, x, y, acc: Tile | None = None, decay: Tile | None = None) -> Tile:
        """x[:, None] * y[None, :] as one MUL (A = y repeated with row stride 0, B = x per row);
        with `acc`, the in-place OUTER acc = acc * decay + x[:, None] * y[None, :]."""
        for t in (x, y):
            if not isinstance(t, Tile) or len(t.shape) != 1:
                raise CompileError("outer(x, y) takes two 1-D tiles")
        self.check_live(x, y, acc, decay)
        rows, cols = x.cols, y.cols
        if acc is None:
            if decay is not None:
                raise CompileError("outer(decay=...) needs acc=")
            out = self.alloc((rows, cols))
            self.emit(I.vop(I.V_MUL, out.base, y.base, x.base, rows, cols, out.rs, 0, 1,
                            I.B_ROW, comment="outer"))
            return out
        if len(acc.shape) != 2 or acc.shape != (rows, cols):
            raise CompileError(f"outer: acc shape {acc.shape} != {(rows, cols)}")
        if cols > I.OUTER_MAX_COLS:
            raise CompileError(f"outer(acc=...) supports up to {I.OUTER_MAX_COLS} columns")
        lo, hi = acc.base, acc.base + (rows - 1) * acc.rs + cols
        if x.base < hi and lo < x.base + rows:
            raise CompileError("outer(acc=...): x must not overlap acc (it is read per row)")
        if decay is None:
            d, dmode = 0, "one"
        elif isinstance(decay, Tile) and decay.shape == (1,):
            d, dmode = decay.base, "scalar"
        elif isinstance(decay, Tile) and decay.shape == (cols,):
            d, dmode = decay.base, "column"
        else:
            raise CompileError(f"outer: decay must be a [1] or [{cols}] tile")
        self.emit(I.outer(acc.base, d, x.base, y.base, rows, cols, acc.rs, 1, dmode,
                          comment=f"outer acc*={dmode}"))
        self.bump_version(acc.buf)
        return acc

    def stream_desc(self, d: I.StreamDesc) -> int:
        """The TMEM address of descriptor d: a slot of the descriptor area (8-word aligned),
        written by FILLs at the program's start (docs/stream.md, 3.2)."""
        words = tuple(d.words())
        if words not in self.descs:
            at = self.desc_at + sum(-(-len(w) // 8) * 8 for w in self.descs)
            if at + len(words) > self.desc_at + self.desc_area:
                raise CompileError("the STREAM descriptor area is full")
            fills = [I.vop(I.V_FILL, at + i, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR,
                           imm=np.uint32(x).view(np.float32), comment=f"stream desc {i}")
                     for i, x in enumerate(words)]
            self.root[self.desc_fills:self.desc_fills] = fills
            self.desc_fills += len(fills)
            self.descs[words] = at
        return self.descs[words]

    def stream(self, d: I.StreamDesc, state: Tensor, vec: Tile | None, x: Tile | None,
               k: Tile | None, o: Tile | None, zero: bool = False, ks: int = 1,
               k_off: int = 0, dst: Tensor | None = None) -> None:
        """STREAM with descriptor d over the fp32 row-major tensor `state` [rows, cols] in
        DRAM, updated in place or written to `dst` (a tensor of its shape at a compile-time
        distance from `state`: both addresses take one register): column slots at vec, row
        scalars at x, constants at k + k_off (K_j at + j * ks), row outputs to o
        (docs/stream.md). `zero`: the state reads as +0."""
        if not self.cfg.STREAM:
            raise CompileError("STREAM needs Config.STREAM (the stream engine)")
        if len(state.shape) != 2 or state.strides != (state.shape[1], 1):
            raise CompileError("stream: the state must be a row-major [rows, cols] tensor")
        if state.shape != (d.rows, d.cols):
            raise CompileError(f"stream: the state is {state.shape}, the descriptor "
                               f"{(d.rows, d.cols)}")
        if Affine.of(state.base).is_static and Affine.of(state.base).const % self.cfg.D:
            raise CompileError("stream: the state must be DRAM-chunk aligned")
        tiles = [t for t in (vec, x, k, o) if t is not None]
        self.check_live(*tiles)
        gap = 0
        if dst is not None:
            g = Affine.of(dst.base) - Affine.of(state.base)
            if dst.shape != state.shape or dst.strides != state.strides or not g.is_static:
                raise CompileError("stream: dst must be the state's shape at a compile-time "
                                   "distance from it")
            gap = g.const
            if gap and (gap % self.cfg.D or abs(gap) < 4 * d.rows * d.cols):
                raise CompileError("stream: dst must be chunk aligned and not overlap the state")
        at = self.stream_desc(d)
        # the source (unless zero: not read) and the destination (the state, or dst at `gap`)
        # share the register ra and the instruction's region: each range rows x cols words
        span = 4 * d.rows * d.cols
        ra, imm, gs = self.dram(state.base, span, "stream state")
        gd = gs if not gap else self.region(Affine.of(state.base) + gap, span, "stream dst")
        g = self.same_region("stream", None if zero else gs, gd)
        (rb, vb), (rc, xb), (rd, kb) = (self.addr(t.base) if t is not None else (0, 0)
                                        for t in (vec, x, k))
        if o is not None and not Affine.of(o.base).is_static:
            raise CompileError("stream: o must be at a static TMEM address")
        if kb + k_off < 0:
            raise CompileError("stream: the constants start below TMEM 0")
        if not 0 < ks < 1 << 16:
            raise CompileError("stream: the constants' stride must be 1..65535")
        self.emit(I.stream(at, imm, imm + gap, vb, xb, kb + k_off,
                           0 if o is None else Affine.of(o.base).const, ks=ks,
                           zero=zero, ra=ra, rb=rb, rc=rc, rd=rd,
                           comment=f"stream {d.rows}x{d.cols}", reg=g))
        if o is not None:
            self.bump_version(o.buf)

    def deltanet_step(self, state: Tensor, qk: Tile, v: Tile, decay: Tile, beta: Tile,
                      o: Tile, zero: bool = False, dst: Tensor | None = None) -> None:
        """DSTEP: one Gated DeltaNet head step on the fp32 state `state` [rows, cols] in DRAM,
        updated in place; qk = [q | k] (2 * cols words), v [rows], decay and beta [1] tiles
        (beta after decay), o [rows] written (docs/isa.md). `zero`: the state starts at +0.
        Without Config.DSTEP, a Config.STREAM machine runs it as STREAM with isa.gdn_desc
        (bit-identical). `dst`: the new state goes there instead and `state` keeps the old one
        (a STREAM, which has a destination; Config.STREAM)."""
        if not (self.cfg.DSTEP or self.cfg.STREAM):
            raise CompileError("deltanet_step needs Config.DSTEP (the DMA's DSTEP) or "
                               "Config.STREAM (the stream engine)")
        if len(state.shape) != 2 or state.strides != (state.shape[1], 1):
            raise CompileError("deltanet_step: the state must be a row-major [rows, cols] tensor")
        rows, cols = state.shape
        if not (0 < rows <= I.DSTEP_MAX_ROWS) or cols % 64 or not (0 < cols <= 256):
            raise CompileError("deltanet_step: rows 1..256, cols 64, 128, 192 or 256")
        for t, n, what in ((qk, 2 * cols, "qk"), (v, rows, "v"), (decay, 1, "decay"),
                           (beta, 1, "beta"), (o, rows, "o")):
            if not isinstance(t, Tile) or len(t.shape) != 1 or t.cols != n:
                raise CompileError(f"deltanet_step: {what} must be a 1-D tile of {n}")
        gs = beta.base - decay.base
        if not 0 < gs < 1 << 16:
            raise CompileError("deltanet_step: beta must follow decay in TMEM")
        self.check_live(qk, v, decay, beta, o)
        if Affine.of(state.base).is_static and Affine.of(state.base).const % self.cfg.D:
            raise CompileError("deltanet_step: the state must be DRAM-chunk aligned")
        if dst is not None and not self.cfg.STREAM:
            raise CompileError("deltanet_step into another state needs Config.STREAM")
        if not self.cfg.DSTEP or dst is not None:
            return self.stream(I.gdn_desc(rows, cols), state, qk, v, decay, o, zero, ks=gs,
                               dst=dst)
        ra, imm, g = self.dram(state.base, 4 * rows * cols, "dstep state")
        self.emit(I.dstep(imm, qk.base, v.base, rows, cols, decay.base, gs, o.base, zero=zero,
                          ra=ra, comment="dstep", reg=g))
        self.bump_version(o.buf)

    def fuse_mm_rmax(self, x: Tile):
        """max(s, axis=1) of a dot output nothing has touched since: set RMAX on that MM and
        return the maxima it writes after the tile's last row."""
        src = getattr(x, "mm_src", None)
        if src is None:
            return None
        ins, version, M, ors = src
        if self.versions.get(x.buf, 0) != version or not any(ins is i for i in self.stack[-1]):
            return None
        ins.flags |= I.F_RMAX
        ins.comment += " +rowmax"
        return Tile(self, x.base + M * ors, (M,), None, x.buf)

    def dot(self, a, w: QTensor, acc: Tile | None = None, out: Tile | None = None,
            temp: bool = False, rowmax: bool = False, acc_scale: Tile | None = None) -> Tile:
        if not isinstance(w, QTensor):
            raise CompileError("dot(a, w): w must be a streamed QTensor descriptor")
        if acc is not None and out is not None:
            raise CompileError("dot: give either acc= (accumulate) or out= (overwrite)")
        if acc_scale is not None:
            if acc is None or w.scale is not None:
                raise CompileError("dot(acc_scale=...) needs acc= and an unscaled stream (V^T)")
            self.check_live(acc_scale)
        st = a if isinstance(a, Stationary) else self.quantize(a, temp)
        if not self.stationary_valid(st):
            raise CompileError("stationary operand was overwritten in ACT RAM")
        # an operand from outside the innermost loop must survive every iteration: keep it (and
        # its blocks) until the loop ends; one created in the same body is rebuilt per iteration
        if self.loops and getattr(st, "loop", None) is not self.loops[-1].loop:
            self.loop_uses.append((len(self.loops), st))
        N, K = w.shape
        if K != st.KB * self.cfg.D:
            raise CompileError(f"dot: K mismatch {K} vs {st.KB * self.cfg.D}")
        one_d = len(st.src.shape) == 1
        M = st.src.rows
        if acc is None and out is None:
            # Layout choice: an odd row stride puts the M results of one streamed row in M
            # distinct TMEM banks, so the MXU writes them back in a single cycle. The spare row
            # after the tile receives the row maxima if the kernel asks for them (RMAX).
            rs = N if (one_d or M == 1 or N % 2) else N + 1
            out = self.alloc((N,) if one_d else (M, N), rs, spare=0 if one_d else M)
        else:
            out = acc if acc is not None else out
            self.check_live(out)
            if (out.rows, out.cols) != (M, N):
                raise CompileError(f"dot: output shape {out.shape} != {(M, N)}")
        ors = out.rs if len(out.shape) == 2 else N
        # the DRAM ranges the MMs read (isasim._mm), both in the MM's region: the weight rows
        # [sa, sa + (N - 1) rs + the row's blocks in whole chunks), the scale rows [ssa,
        # ssa + (N - 1) srs + what a row reads: Sw, then the blocks up to the operand's last)
        D = self.cfg.D
        bpb = D if w.wf == I.W8 else D // 2             # streamed bytes per block
        ra, sa, gw = self.dram(w.data, (N - 1) * w.rs + -(-st.KB * bpb // D) * D, "mm weights")
        rb, ssa, gsc = 0, 0, None
        if w.scale is not None:
            kend = w.k0 + st.KB
            ext = 4 if w.rsc else {I.W8: 4 + kend, I.WNV: 16 + 8 * kend}.get(w.wf, 8 + 4 * kend)
            rb, ssa, gsc = self.dram(w.scale, (N - 1) * w.srs + ext, "mm scales")
        g = self.same_region("mm (its weight rows and scale rows)", gw, gsc)
        if acc_scale is not None:
            if acc_scale.shape != (M,):
                raise CompileError(f"dot: acc_scale shape {acc_scale.shape} != {(M,)}")
            if len(st.chunks) != 1 or M > self.cfg.MCOLS:
                raise CompileError("dot(acc_scale=...) needs M <= MCOLS")
        if rowmax and (len(st.chunks) != 1 or M > self.cfg.MCOLS or len(out.shape) != 2
                       or getattr(out, "spare", 0) < M or out.rs < N):
            raise CompileError("dot(rowmax=True) needs M <= MCOLS and an output tile from "
                               "ol.empty/zeros/full (they reserve the row-max area)")
        unit = w.scale is None
        # PAIR reads a chunk's two blocks' scales at once: 8-byte (nvfp4: 16-byte) aligned rows
        # and an even first block; int8 PAIR (DUAL: two chunks an advance) two adjacent bytes
        al = 16 if w.wf == I.WNV else 8 if w.wf != I.W8 else 4
        pair = st.pair and (w.wf != I.W8 or self.cfg.DUAL) and (unit or w.rsc or (
            w.k0 % 2 == 0 and w.srs % al == 0 and Affine.of(w.scale).const % al == 0
            and all(c * getattr(v, "align", 1) % al == 0
                    for v, c in Affine.of(w.scale).terms.items())))
        # one MM per QACT range of the operand (an MM's rows read one Sa: docs/isa.md "MM"),
        # added up in the output: the first as asked (overwrite, ACC or ASCALE), the rest ACC;
        # RMAX on the last, whose written values are the sums
        parts = st.ranges()
        for i, (k0, k1) in enumerate(parts):
            if (k0 * bpb) % self.cfg.D:
                raise CompileError(f"dot: a QACT range starting at block {k0} of 4-bit weights "
                                   f"(their D-byte chunks hold two blocks): split the operand "
                                   f"at even blocks")
            first, last = i == 0, i == len(parts) - 1
            kp = 0 if unit or w.rsc else w.k0 + k0
            ppair = pair and kp % 2 == 0
            m0 = 0
            for ab, mc in st.chunks:
                ins = I.mm(sa + k0 * bpb, ssa, out.base + m0 * ors, N, k1 - k0, w.rs, ors, mc,
                           ab + k0, w.srs, unit=unit, acc=acc is not None or not first,
                           rmax=rowmax and last, wf=w.wf, pair=ppair, k0=kp, rsc=w.rsc,
                           ascale=acc_scale.base if acc_scale is not None and first else None,
                           ra=ra, rb=rb, reg=g,
                           comment=f"mm {M}x{K} . {N}x{K}^T"
                           + (f" part {k0}:{k1}" if len(parts) > 1 else "")
                           + (" pair" if ppair else "") + (" +rowmax" if rowmax and last else "")
                           + (" acc*=scale" if acc_scale is not None and first else ""))
                self.emit(ins)
                m0 += mc
        self.bump_version(out.buf)
        if (acc is None and len(st.chunks) == 1 and M <= self.cfg.MCOLS and len(out.shape) == 2
                and getattr(out, "spare", 0) >= M and out.rs >= N):
            out.mm_src = (ins, self.versions.get(out.buf, 0), M, ors)
        return out

    def store_quantized(self, x: Tile, dst: Affine, sdst: Affine, drs: int, es: int,
                        row_scale: bool, half: bool = False) -> None:
        self.check_live(x)
        D = self.cfg.D
        if x.cols % D:
            raise CompileError("quantized stores need a multiple of D elements per row")
        # the data rows [dst, dst + (rows - 1) drs + (elements - 1) es + 1) and the scale rows
        # (4 bytes a row with ROW, else Sw and the block offsets: 4 + 4 ceil(KB / 4)) back to
        # back, both in the QST's region (isasim._qst)
        kb = x.cols // D
        ne = x.cols // 2 if half else x.cols
        sw = 4 if row_scale else 4 + 4 * -(-kb // 4)
        rb, dimm, gd = self.dram(dst, (x.rows - 1) * drs + (ne - 1) * es + 1, "qst data")
        rc, simm, gsc = self.dram(sdst, x.rows * sw, "qst scales")
        g = self.same_region("qst (its data rows and scale rows)", gd, gsc)
        rs = x.rs if len(x.shape) == 2 else x.cols
        if not row_scale and x.cols // D > QST_KB_MAX:
            raise CompileError(f"QST of {x.cols // D} blocks without ROW: at most {QST_KB_MAX}")
        self.emit(I.qst(x.base, dimm, simm, x.rows, x.cols // D, rs, drs, es, row=row_scale,
                        half=half, rb=rb, rc=rc, comment="quantized store", reg=g))

    def all_gather(self, x: Tile, S: int) -> Tile:
        x = self.materialize(x)
        self.check_live(x)
        if len(x.shape) == 1:
            out = self.alloc((x.cols * S,))
            self.emit(I.gather(x.base, out.base, 1, x.cols, x.cols, x.cols * S, x.cols,
                               comment="all-gather"))
        else:
            out = self.alloc((x.rows, x.cols * S))
            self.emit(I.gather(x.base, out.base, x.rows, x.cols, x.rs, out.rs, x.cols,
                               comment="all-gather"))
        return out

    # ---- finalize
    def finish(self) -> list[I.Instr]:
        if self.loops:
            raise CompileError("unterminated loop")
        prog = [I.li(r, 0, comment="address register")
                for r in sorted(self.used_regs - self.late_zero)]
        prog += self._flatten(self.root)
        prog.append(I.halt())
        return Program(prog, self.run_args, self.run_seen)

    def _flatten(self, items: list) -> list[I.Instr]:
        out = []
        for it in items:
            if isinstance(it, I.Instr):
                out.append(it)
                continue
            loop = it.loop
            if loop.rcount and it.steps:
                raise CompileError("a loop with a device-computed count cannot step registers")
            body = self._flatten(it.items)
            body += [I.addi(r, r, c, comment=f"{loop} step") for r, c in it.steps]
            if not body or self._ends_inner(it.items, it.steps) or self._ends_loop(body):
                body.append(I.nop("loop end"))
            out.append(I.loop(len(body), loop.count, rcount=loop.rcount,
                              comment=f"{loop} x" + (f"R{loop.rcount}" if loop.rcount
                                                      else f"{loop.count}")))
            out += body
            out += [I.addi(r, r, -c * loop.count, comment=f"{loop} reset") for r, c in it.steps]
        return out

    @staticmethod
    def _ends_inner(items: list, mine: list) -> bool:
        return not mine and bool(items) and isinstance(items[-1], LoopBlock)

    @staticmethod
    def _ends_loop(body: list) -> bool:
        """A LOOP in the body (a guard emitted as instructions, generate.py's) ends on its last
        instruction: a body must not end on an enclosing body's (docs/isa.md, LOOP)."""
        return any(x.op == I.LOOP and i + x.w[0] == len(body) - 1 for i, x in enumerate(body))


# =============================================================================== tracing context
# per thread: the Engine compiles the next token's program on a worker thread while the
# device runs the current one (opentpu/llm/qwen3.py)
_TLS = threading.local()


def _stack() -> list:
    s = getattr(_TLS, "stack", None)
    if s is None:
        s = _TLS.stack = []
    return s


def current() -> Builder:
    st = _stack()
    if not st:
        raise CompileError("openTPU language functions can only be called inside a kernel")
    return st[-1]


@dataclass
class Compiled:
    cfg: Config
    programs: list            # per slice list[Instr]
    stores: list              # per slice store records
    layout: object            # runtime DRAM layout (see runtime.py)


class Kernel:
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__

    def trace(self, cfg: Config, sid: int, bound: dict, fit: str = "next") -> Builder:
        b = Builder(cfg, sid)
        b.S = cfg.S
        b.fit = fit
        st = _stack()
        st.append(b)
        try:
            self.fn(**bound)
        finally:
            st.pop()
        return b


_TRACE_CODE = Kernel.trace.__code__


def trace_fit(kernel: Kernel, cfg: Config, sid: int, bound) -> Builder:
    """kernel.trace(cfg, sid, bound()), and when its tiles do not fit TMEM placed next-fit (the
    free words split into gaps too small: a small TMEM), the trace again with best-fit
    placement. For the programs that have no smaller form to fall back to (a decode step, a
    generate bucket's split form):
    the compiles that probe (a prefill run's rows, a prompt run's, a generate bucket's single
    program) keep next-fit's answer, so a program that fits next-fit is placed as before.
    `bound` makes the arguments (fresh descriptors for the second trace)."""
    try:
        return kernel.trace(cfg, sid, bound())
    except CompileError as e:
        if not str(e).startswith("TMEM exhausted"):
            raise
    return kernel.trace(cfg, sid, bound(), fit="best")


def jit(fn) -> Kernel:
    return Kernel(fn)


def slow_drains(prog, cfg, lanes: int | None = None) -> list:
    """The MMs of `prog` whose drain cannot keep up with their array: [(pc, comment)].

    The array consumes a streamed row's chunks once per group of MCOLS stationary rows (replay:
    ceil(M / MCOLS) groups), KB advances a group (ceil(KB / 2) with PAIR). The drain writes a
    group's results, one a row at the output's row stride (ors), `lanes` a cycle (the MXU's
    epilogue lanes; production's MCOLS) from distinct TMEM banks only: its pick takes rows in
    order up to the first bank conflict (bank = address mod LANES). At a row stride that is a
    multiple of the banks, that is one a cycle. An MM is slow when its drain takes more steps a
    group than its array takes advances: W_down at KB 2 into a [16, 1152] tile loaded from DRAM
    (MiniCPM-V's tower: 40% of its pops). A fresh output and ol.empty / zeros / full give an odd
    row stride, which spreads the rows over the banks."""
    lanes = lanes or cfg.MCOLS
    out = []
    for pc, ins in enumerate(prog):
        if ins.op != I.MM:
            continue
        ors, M, KB = ins.w[5] & 0xFFFF, (ins.w[5] >> 16) & 0xFF, (ins.w[3] >> 16) & 0xFF
        adv = -(-KB // 2) if ins.flags & I.F_PAIR else KB
        steps, j, rows = 0, 0, min(M, cfg.MCOLS)
        while j < rows:
            banks = set()
            while j < rows and len(banks) < lanes and (j * ors) % cfg.LANES not in banks:
                banks.add((j * ors) % cfg.LANES)
                j += 1
            steps += 1
        if steps > adv:
            out.append((pc, ins.comment))
    return out
