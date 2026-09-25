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


# =============================================================================== affine addresses
_loop_ids = itertools.count()


class Loop:
    """Hardware loop induction variable."""

    def __init__(self, count: int):
        self.id = next(_loop_ids)
        self.count = count

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


class Affine:
    def __init__(self, const: int = 0, terms: dict | None = None):
        self.const = int(const)
        self.terms = {l: c for l, c in (terms or {}).items() if c != 0}

    @staticmethod
    def of(x) -> "Affine":
        if isinstance(x, Affine):
            return x
        if isinstance(x, Loop):
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
    """int8 matrix [rows, cols] with fp32 scales per (row, D-block) -- the MXU streamed operand.

    `scale is None` means unit scales (e.g. V^T, whose per-token scale is folded into P).
    `parts`: the matrix is stored as separate column slices of width `pw` (each with its own
    contiguous rows and scales, so a column-slice MM streams contiguous scales); it can then
    only be sliced into whole parts.
    """
    data: Affine
    scale: Affine | None
    shape: tuple
    rs: int            # row stride, bytes
    srs: int           # scale row stride, bytes
    D: int
    parts: tuple | None = None
    pw: int = 0

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
        data = self.data + r0 * self.rs + c0
        scale = None
        if self.scale is not None:
            scale = self.scale + r0 * self.srs + c0.div_exact(self.D) * 4
        return QTensor(data, scale, (nr, nc), self.rs, self.srs, self.D)


class KVDesc:
    """Per-slice view of a KV cache whose heads are dealt round-robin over slices.

    Layout per head: K token-major int8 [cap, d] + scales [cap, d/D]; V^T dim-major int8
    [d, cap]; one V scale per token [cap] (folded into P before P.V). With head_dim < D the
    rows are zero-padded to d = D and dv = head_dim: P.V reads only the first dv rows of V^T.
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
        e = self._h(h)
        return QTensor(Affine.of(e["k"]), Affine.of(e["ks"]), (self.cap, self.d), self.d,
                       4 * (self.d // self.D), self.D)

    def vt(self, h: int) -> QTensor:
        e = self._h(h)
        return QTensor(Affine.of(e["vt"]), None, (self.dv, self.cap), self.cap, 0, self.D)

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
                a, e, _ = key[0].indices(self.cols)
                return Tile(self.b, self.base + a, (e - a,), None, self.buf)
            raise CompileError(f"unsupported 1-D index {key}")
        rk = key[0]
        ck = key[1] if len(key) > 1 else slice(None)
        if isinstance(rk, (int, np.integer)) and isinstance(ck, slice):
            a, e, _ = ck.indices(self.cols)
            return Tile(self.b, self.base + int(rk) * self.rs + a, (e - a,), None, self.buf)
        if isinstance(rk, slice) and isinstance(ck, slice):
            r0, r1, rstep = rk.indices(self.rows)
            c0, c1, cstep = ck.indices(self.cols)
            if rstep != 1 or cstep != 1:
                raise CompileError("strided tile views are not supported")
            return Tile(self.b, self.base + r0 * self.rs + c0, (r1 - r0, c1 - c0), self.rs,
                        self.buf)
        raise CompileError(f"unsupported 2-D index {key}")

    def repeat_rows(self, n: int) -> "Tile":
        """A 1-D tile (or a single row) as an [n, cols] view whose every row is that row: row
        stride 0, for the A operand of a VOP (the outer product u[:, None] * v[None, :])."""
        if self.rows != 1:
            raise CompileError(f"repeat_rows: {self} is not a single row")
        return Tile(self.b, self.base, (n, self.cols), 0, self.buf)

    def row_stride_view(self, r0: int, n: int, step: int) -> "Tile":
        """Rows r0, r0+step, ... (n of them) of a 2-D tile, as a [n, cols] view."""
        if len(self.shape) != 2 or r0 + (n - 1) * step >= self.rows:
            raise CompileError(f"row_stride_view({r0}, {n}, {step}) outside {self}")
        return Tile(self.b, self.base + r0 * self.rs, (n, self.cols), self.rs * step, self.buf)

    def reshape(self, rows: int, cols: int) -> "Tile":
        """The same words as a [rows, cols] tile (contiguous tiles only)."""
        if not self.contiguous or rows * cols != self.rows * self.cols:
            raise CompileError(f"reshape: {self} is not {rows}x{cols} contiguous words")
        return Tile(self.b, self.base, (rows, cols), cols, self.buf)

    # ---- in-place update (loop-carried values)
    def set(self, value) -> "Tile":
        temp = sys.getrefcount(value) <= TEMP_RC_METHOD
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

    # ---- in place: `t += x` is one VOP writing t (no temporary)
    def _inplace(self, func: int, o) -> "Tile":
        self.b.check_live(self, o)
        self.b.vop(func, self, self, o)
        self.b.bump_version(self.buf)
        return self

    def __iadd__(self, o):
        return self._inplace(I.V_ADD, o)

    def __isub__(self, o):
        return self._inplace(I.V_SUB, o)

    def __imul__(self, o):
        return self._inplace(I.V_MUL, o)


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


class Stationary:
    """A tile quantized into ACT RAM blocks [ab, ab+KB) for up to MCOLS rows."""

    def __init__(self, src: Tile, chunks: list, KB: int, owners: list):
        self.src, self.chunks, self.KB, self.owners = src, chunks, KB, owners


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
        self.act_next = 0
        self.act_owner = [None] * cfg.ACT_BLOCKS
        self.act_live = [None] * cfg.ACT_BLOCKS      # weakref to the Stationary holding a block
        self.regs: dict = {}          # live: frozenset(terms) -> register
        # Registers whose inner loops have ended but that still hold outer-loop terms:
        # (terms, loops live when retired, register). One may take a new key that adds terms
        # of loops begun since (those are 0 at the loop start and stepped from then on).
        self.spare_regs: list = []
        self.free_regs = list(range(15, 0, -1))
        self.used_regs: set = set()
        self.versions = weakref.WeakKeyDictionary()
        self.tmem_regions: list = []     # (base, end, weakref to the allocation's _Buf)
        self.tmem_peak = 0
        self.stores: list = []        # (host, element offsets, dram byte address)
        self.loop_uses: list = []     # stationaries used inside loops: (loop depth, stat)

    # ---- emission
    def emit(self, ins: I.Instr) -> None:
        ins.src = _user_frames()
        self.stack[-1].append(ins)

    def addr(self, a) -> tuple[int, int]:
        a = Affine.of(a)
        if a.is_static:
            return 0, a.const
        for l in a.terms:
            if not any(lb.loop is l for lb in self.loops):
                raise CompileError(f"address uses {l} outside its loop")
        key = frozenset(a.terms.items())
        if key not in self.regs:
            r = self._spare_for(key)
            if r is None:
                if not self.free_regs:
                    raise CompileError("out of address registers")
                r = self.free_regs.pop()
            self.regs[key] = r
            self.used_regs.add(r)
        return self.regs[key], a.const

    def _spare_for(self, key):
        """A retired register holding a subset of `key` whose missing terms all belong to loops
        begun after it was retired."""
        for i, (terms, live_then, r) in enumerate(self.spare_regs):
            if terms <= key and not any(id(l) in live_then for l, _ in key - terms):
                del self.spare_regs[i]
                return r
        return None

    def begin_loop(self, n: int) -> Loop:
        if len(self.loops) >= 4:
            raise CompileError("loop nesting deeper than 4")
        lb = LoopBlock(Loop(n))
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
                self.free_regs.append(r)
            elif terms not in self.regs:
                self.regs[terms] = r
            else:
                self.spare_regs.append((terms, frozenset(live), r))
        depth = len(self.loops)
        for d, stat in self.loop_uses:
            if d > depth and not self.stationary_valid(stat):
                raise CompileError("a stationary operand used inside a loop was overwritten "
                                   "inside that loop; quantize it inside the loop instead")
        self.loop_uses = [(d, s) for d, s in self.loop_uses if d <= depth]

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
        W = self.cfg.TMEM_WORDS
        for attempt in range(2):
            self.tmem_regions = [r for r in self.tmem_regions if r[2]() is not None]
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
        raise CompileError("TMEM exhausted")

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

    def vop(self, func: int, dst: Tile, a: Tile | None, b) -> None:
        """Emit dst = func(a, b) over dst's shape; b may be a Tile, Bcast or float."""
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
            if v.cols == 1 and (rows if b.kind == B_ROWVIEW else cols) > 1:
                # one value for the whole tile: the per-row word, with row stride 0
                bmode, b_base, brs = I.B_ROW, v.base, 0
            elif b.kind == B_ROWVIEW:
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
        self.emit(I.vop(func, dst.base, a_base, b_base, rows, cols, drs, ars, brs, bmode, imm,
                        comment=I.VFUNCS[func]))

    def binop(self, func: int, x, y):
        swap = {I.V_ADD: I.V_ADD, I.V_MUL: I.V_MUL, I.V_MAX: I.V_MAX, I.V_MIN: I.V_MIN,
                I.V_SUB: I.V_RSUB, I.V_RSUB: I.V_SUB}
        if not isinstance(x, Tile) or (x.shape == (1,) and isinstance(y, Tile)
                                       and y.shape != (1,)):
            if isinstance(y, Tile):
                x, y, func = y, x, swap[func]
            elif isinstance(x, Bcast) and isinstance(y, Bcast) and {x.kind, y.kind} == {
                    B_ROWVIEW, B_COLVIEW}:
                # outer: u[:, None] (op) v[None, :] reads v as A with row stride 0, u per row
                if x.kind == B_ROWVIEW:
                    x, y, func = y, x, swap[func]
                x = x.t.repeat_rows(y.t.cols)
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
        if out is None:
            out = self.alloc(t.shape)
        else:
            self.check_live(out)
            if out.shape != tuple(t.shape) or not out.contiguous:
                raise CompileError(f"load: out {out} is not a contiguous {t.shape} tile")
            self.bump_version(out.buf)
        if len(t.shape) == 1:
            if t.strides[0] != 1:
                raise CompileError("strided 1-D loads are not supported")
            ra, imm = self.addr(t.base)
            self.emit(I.ld(imm, out.base, t.shape[0], ra=ra, comment="load"))
            return out
        if len(t.shape) != 2 or t.strides[1] != 1:
            raise CompileError("loads must be 1-D or row-major 2-D")
        if t.strides[0] == t.shape[1]:
            ra, imm = self.addr(t.base)
            self.emit(I.ld(imm, out.base, t.shape[0] * t.shape[1], ra=ra, comment="load"))
        else:
            for r in builtins.range(t.shape[0]):
                ra, imm = self.addr(t.base + r * 4 * t.strides[0])
                self.emit(I.ld(imm, out.base + r * out.rs, t.shape[1], ra=ra, comment="load row"))
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
            ra, imm = self.addr(base)
            n = rows * x.cols if merged else x.cols
            self.emit(I.st(imm, x.base + r * x.rs, n, ra=ra, comment="store"))
            if t.host is not None:
                if not Affine.of(t.offset).is_static or not Affine.of(base).is_static:
                    raise CompileError("stores to kernel outputs must use static addresses")
                off = Affine.of(t.offset).static() + r * rstride
                self.stores.append((t.host, off, n))

    def quantize(self, x: Tile, temp: bool = False) -> Stationary:
        x = self.materialize(x)
        self.check_live(x)
        D = self.cfg.D
        if x.cols % D:
            raise CompileError(f"reduction dim {x.cols} must be a multiple of D={D}")
        KB = x.cols // D
        src, rs, cs, rsc = x.base, (x.rs if len(x.shape) == 2 else x.cols), None, None
        fused = self.fuse_cscale(x) if temp else None
        if fused is not None:
            src, rs, cs, rsc = fused
        chunks, owners = [], []
        for m0 in builtins.range(0, x.rows, self.cfg.MCOLS):
            mc = min(self.cfg.MCOLS, x.rows - m0)
            ab, owner = self.act_alloc(KB)
            self.emit(I.qact(src + m0 * rs, mc, ab, KB, rs, cscale=cs,
                             rscale=None if rsc is None else rsc + m0,
                             comment="quantize -> ACT" + (" x col scale" if cs is not None else "")
                             + (" x row scale" if rsc is not None else "")))
            chunks.append((ab, mc))
            owners.append(owner)
        st = Stationary(x, chunks, KB, owners)
        st.loop = self.loops[-1].loop if self.loops else None     # created in this loop body
        for ab, _ in chunks:
            for k in builtins.range(ab, ab + KB):
                self.act_live[k] = weakref.ref(st)
        return st

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
        ra, sa = self.addr(w.data)
        rb, ssa = self.addr(w.scale) if w.scale is not None else (0, 0)
        if acc_scale is not None:
            if acc_scale.shape != (M,):
                raise CompileError(f"dot: acc_scale shape {acc_scale.shape} != {(M,)}")
            if len(st.chunks) != 1:
                raise CompileError("dot(acc_scale=...) needs M <= MCOLS")
        if rowmax and (len(st.chunks) != 1 or len(out.shape) != 2
                       or getattr(out, "spare", 0) < M or out.rs < N):
            raise CompileError("dot(rowmax=True) needs M <= MCOLS and an output tile from "
                               "ol.empty/zeros/full (they reserve the row-max area)")
        m0 = 0
        for ab, mc in st.chunks:
            ins = I.mm(sa, ssa, out.base + m0 * ors, N, st.KB, w.rs, ors, mc, ab, w.srs,
                       unit=w.scale is None, acc=acc is not None, rmax=rowmax,
                       ascale=acc_scale.base if acc_scale is not None else None, ra=ra, rb=rb,
                       comment=f"mm {M}x{K} . {N}x{K}^T" + (" +rowmax" if rowmax else "")
                       + (" acc*=scale" if acc_scale is not None else ""))
            self.emit(ins)
            m0 += mc
        self.bump_version(out.buf)
        if (acc is None and len(st.chunks) == 1 and len(out.shape) == 2
                and getattr(out, "spare", 0) >= M and out.rs >= N):
            out.mm_src = (ins, self.versions.get(out.buf, 0), M, ors)
        return out

    def store_quantized(self, x: Tile, dst: Affine, sdst: Affine, drs: int, es: int,
                        row_scale: bool) -> None:
        D = self.cfg.D
        if x.cols % D:
            raise CompileError("quantized stores need a multiple of D elements per row")
        rb, dimm = self.addr(dst)
        rc, simm = self.addr(sdst)
        rs = x.rs if len(x.shape) == 2 else x.cols
        self.emit(I.qst(x.base, dimm, simm, x.rows, x.cols // D, rs, drs, es, row=row_scale,
                        rb=rb, rc=rc, comment="quantized store"))

    def all_gather(self, x: Tile, S: int) -> Tile:
        x = self.materialize(x)
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
        prog = [I.li(r, 0, comment="address register") for r in sorted(self.used_regs)]
        prog += self._flatten(self.root)
        prog.append(I.halt())
        return prog

    def _flatten(self, items: list) -> list[I.Instr]:
        out = []
        for it in items:
            if isinstance(it, I.Instr):
                out.append(it)
                continue
            loop = it.loop
            body = self._flatten(it.items)
            body += [I.addi(r, r, c, comment=f"{loop} step") for r, c in it.steps]
            if not body or self._ends_inner(it.items, it.steps):
                body.append(I.nop("loop end"))
            out.append(I.loop(len(body), loop.count, comment=f"{loop} x{loop.count}"))
            out += body
            out += [I.addi(r, r, -c * loop.count, comment=f"{loop} reset") for r, c in it.steps]
        return out

    @staticmethod
    def _ends_inner(items: list, mine: list) -> bool:
        return not mine and bool(items) and isinstance(items[-1], LoopBlock)


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

    def trace(self, cfg: Config, sid: int, bound: dict) -> Builder:
        b = Builder(cfg, sid)
        b.S = cfg.S
        st = _stack()
        st.append(b)
        try:
            self.fn(**bound)
        finally:
            st.pop()
        return b


def jit(fn) -> Kernel:
    return Kernel(fn)
