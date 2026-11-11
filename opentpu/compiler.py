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
    (a V^T append at a run-time token stays in its tile: tile_split). raw: under run words
    (Builder.run_words) its word holds the value's bits, an int32 (RLD raw), not an fp32 (one
    past 2^24: a V^T place, batch35's one program per slots)."""

    raw = False

    def __init__(self, name: str, bound: int | None = None, raw: bool = False):
        self.name, self.bound = name, bound
        if raw:
            self.raw = True

    def value(self, values: dict) -> int:
        """The variable's value among the run's values ({name: value})."""
        return values[self.name]

    def comp(self) -> "Comp":
        """Comp(self), one for this variable."""
        c = self.__dict__.get("_comp")
        if c is None:
            c = self._comp = Comp(self)
        return c

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


class Comp(RunVar):
    """bound - 1 - var for a RunVar of a bound (named var's name and "~"): above AW 32 an
    argument register holds a zero-extended word, so the argument of an address's c * var with
    c < 0 is -c * Comp(var), its register c * var plus -c (bound - 1) (Builder._host_arg)."""

    def __init__(self, var: RunVar):
        super().__init__(var.name + "~", var.bound)
        self.var = var

    def value(self, values: dict) -> int:
        return self.bound - 1 - self.var.value(values)


def run_var(name: str, bound: int | None = None) -> RunVar:
    """The RunVar (or Comp, its name ending in "~") of this name and bound."""
    return RunVar(name[:-1], bound).comp() if name.endswith("~") else RunVar(name, bound)


def mul_bound(c: int, bound: int | None, aw: int = 32) -> tuple | None:
    """The bound of an RLD MUL's c * v for v in [0, bound) (isa.rld's, None without one): c *
    [0, bound), a negative c's product mod 2^aw the interval from 2^aw + c (bound - 1) that
    wraps to 0 (hi 2^aw: waits' intervals)."""
    if not bound:
        return None
    lo, hi = sorted((0, int(c) * (int(bound) - 1)))
    return (lo, hi) if c >= 0 else (lo + (1 << aw), 1 << aw)


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


class _Bias(RunVar):
    """An address register's constant at AW > 32 (Builder.addr, docs/compiler.md "Addresses
    above 4 GiB"): a key's term (BIAS, B) says its register holds B plus its other terms, the
    address's constant minus B the offset in the instruction (below 4 GiB, zero-extended)."""


BIAS = _Bias("bias")
BIAS_ALIGN = 1 << 28            # a new base: the address rounded down to 256 MiB
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
        w[arg_reg(k) - ARG0] = (c * v.value(values)) & 0xFFFFFFFF
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
    """fp32 tensor in the slice's DRAM. Addresses in bytes, strides in elements. `esize` 2: a
    DeltaNet state in 16 bits (Config.ST16; docs/isa.md "ST16"), which only DSTEP and STREAM
    (flag ST16) read and write."""
    base: Affine
    shape: tuple
    strides: tuple
    host: object = None            # host argument this view belongs to (for outputs)
    offset: Affine = field(default_factory=Affine)   # element offset inside the host array
    esize: int = 4                 # bytes an element

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
                base, off = base + start * (self.esize * st), off + start * st
                shape.append(size)
                strides.append(st)
            else:
                base, off = base + Affine.of(k) * (self.esize * st), off + Affine.of(k) * st
        return Tensor(base, tuple(shape), tuple(strides), self.host, off, self.esize)


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
    at a loop-dependent column is an address), the last one narrower where the width is not
    whole parts; it can then only be sliced into whole parts.
    `gs`: the columns are in an all-gather's order over gs slices (quant.gather_order: chunk c
    of every slice's columns, then chunk c + 1; kernels.mlp.swiglu_down's W_down, in parts of
    gs chunks: chunk c's down projection one MM).
    `wf`: the element format (isa.W8 int8; W4I / W4F / WNV: 4-bit elements, D/2 bytes per
    D-block, docs/isa.md "Weight formats").
    `tiles`: (tb, tstride): the columns are stored in tiles of tb (each tile rows x tb, row
    stride rs = tb, tiles tstride bytes apart; int8, unscaled: the V^T cache). A column slice
    must stay within one tile.
    `grp`: rows a group (docs/isa.md "Weight groups", MM's groups; 1: rows of one, the natural
    layout): group g's chunk c (K-block c) at data + g rs + c P, P = WG x D, lane j's D bytes at
    j D: int8 (grp = WG) row g grp + j's block, 4-bit (grp = 2 WG) rows g grp + 2j and 2j + 1's,
    a byte their elements i; group g's scale area at scale + g srs (quant.group_rows /
    group_scales). A row slice starts at a group.
    `kt`: the groups in MM KT's pairs (Config.KT, docs/isa.md "K tail"; quant.grouped kt): group
    2p's chunks but the last, the pair's tail chunk (both groups' last half blocks), group 2p +
    1's but the last, rs the pairs' stride; a row slice starts at a pair, and the matrix streams
    its whole K (no column slice: k0 0).
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
    grp: int = 1
    gs: int = 1
    kt: bool = False

    @staticmethod
    def weight(data, scale, n: int, k: int, fmt: str, D: int, grp: int = 1, kt: bool = False,
               **kw) -> "QTensor":
        """The descriptor of an [n, k] matrix in MXU format `fmt` ("int8", "int4", "fp4",
        "nvfp4", opentpu/quant.py quantize_mxu's rows): element rows row_bytes apart, scale rows
        scale_bytes apart; grouped for a machine of WG `grp` (quant.quantize_grouped: groups of
        quant.grp_r(fmt, grp) rows), groups grp_row_bytes and their scale areas
        grp_scale_bytes apart; kt: the groups in MM KT's pairs, kt_pair_bytes apart."""
        from . import quant as Q
        rs = Q.kt_pair_bytes(k, fmt, D, grp) if kt else Q.grp_row_bytes(k, fmt, D, grp)
        return QTensor(Affine.of(data), Affine.of(scale), (n, k), rs,
                       Q.grp_scale_bytes(k, fmt, D, grp), D, wf=Q.mxu_wf(fmt),
                       grp=Q.grp_r(fmt, grp), kt=kt, **kw)

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
        if self.grp > 1:          # a tile's groups rs apart, chunk c (columns c D ..) c P in
            g = self.grp
            try:
                at = Affine.of(r0).div_exact(g) * self.rs + Affine.of(r).div_exact(self.D) * (
                    g * self.D)
            except CompileError:
                raise CompileError(f"a grouped tile's slice starts at a group and a D-block "
                                   f"(rows {r0}, columns {c0})") from None
            return QTensor(self.data + q * tstride + at, None, (nr, nc), self.rs, 0, self.D,
                           grp=g)
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
            if c0 % self.pw or c1 - c0 != min(self.pw, self.shape[1] - c0):
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
        g = self.grp
        if self.kt and not (c0.is_static and c0.static() == 0 and nc == self.shape[1]):
            raise CompileError(f"a KT matrix streams its whole K (docs/isa.md \"K tail\"): no "
                               f"column slice (columns {c0} + {nc} of {self.shape[1]})")
        if g > 1:                 # whole groups: chunk c of the group's rows is g D bytes
            try:
                r0 = r0.div_exact(g)
                rp = r0.div_exact(2) if self.kt else r0
            except CompileError:
                raise CompileError(f"a row slice of a matrix in groups of {g} rows starts at a "
                                   f"group (row {r0}; docs/isa.md \"Weight groups\": WG rows "
                                   f"int8, 2 WG 4-bit)" + (", of KT's pairs at a pair" if self.kt
                                                         else "")) from None
            # a group chunk is a K-block, its lanes' D bytes each (4-bit: two rows a lane); KT:
            # the pair's
            data = self.data + rp * self.rs + c0.div_exact(self.D) * (
                self.D * (g if self.wf == I.W8 else g // 2))
        elif self.wf == I.W8:
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
                       rsc=self.rsc, grp=g, kt=self.kt)


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


def gall_wg(cfg) -> int:
    """The KV caches' groups (KVDesc's wg): WG under GALL (at WG > 1), else 1, rows of one."""
    return cfg.WG if cfg.WG > 1 and getattr(cfg, "GALL", False) else 1


def kv_bytes(k, ks, vt, D: int, wg: int = 1, chash: bool = False, at_k: int = 0,
             at_vt: int = 0) -> tuple:
    """A KV head's K [cap, d] (int8), its scale rows [cap, k_scale_bytes] and V^T [d, cap]
    (int8) as KVDesc stores them: (K, K scales, V^T) bytes, rows of one, or (wg > 1) in groups
    (quant.group_rows / group_scales; V^T a tile at a time), with the channel hash the lanes
    as stored at at_k / at_vt (quant.hash_rows)."""
    from . import quant as Q
    cap, d = k.shape
    k8 = np.ascontiguousarray(k).view(np.uint8)
    ks = np.asarray(ks, np.uint8)
    if wg == 1:
        return k8.reshape(-1), ks.reshape(-1), vt_tiled(vt).view(np.uint8)
    tb = vt_tile(cap)
    v8 = np.ascontiguousarray(vt).view(np.uint8)
    kg = Q.group_rows(k8, wg, D).reshape(-1)
    vg = np.concatenate([Q.group_rows(v8[:, i:i + tb], wg, D).reshape(-1)
                         for i in range(0, cap, tb)])
    if chash:
        kg, vg = Q.hash_rows(kg, at_k, wg, D), Q.hash_rows(vg, at_vt, wg, D)
    return kg, Q.group_scales(ks, "int8", d, wg, D, rsc=d == D).reshape(-1), vg


def kv_rows(kb, ksb, vtb, cap: int, d: int, D: int, wg: int = 1, chash: bool = False,
            at_k: int = 0, at_vt: int = 0) -> tuple:
    """kv_bytes' inverse: (K [cap, d] uint8, scale rows [cap, k_scale_bytes], V^T [d, cap]
    int8) from a head's stored bytes."""
    from . import quant as Q
    kb, ksb, vtb = (np.asarray(x, np.uint8).reshape(-1) for x in (kb, ksb, vtb))
    if wg == 1:
        return kb.reshape(cap, d), ksb.reshape(cap, -1), vt_untiled(vtb.view(np.int8), d, cap)
    if chash:
        kb, vtb = Q.hash_rows(kb, at_k, wg, D), Q.hash_rows(vtb, at_vt, wg, D)
    tb = vt_tile(cap)
    kq = Q.ungroup_rows(kb.reshape(cap // wg, -1), cap, wg, D)
    ks = Q.ungroup_scales(ksb.reshape(cap // wg, -1), "int8", d, cap, wg, D, rsc=d == D)
    vt = np.concatenate([Q.ungroup_rows(t.reshape(d // wg, -1), d, wg, D)
                         for t in vtb.reshape(cap // tb, -1)], axis=1)
    return kq, ks, vt.view(np.int8)


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

    wg > 1 (GALL, docs/weight_groups.md): both in groups of wg rows (docs/isa.md
    "Weight groups"), in the same bytes: K's groups of wg tokens (a group's chunk c its tokens'
    block c, its scale area their Sw and offset bytes: quant.group_rows / group_scales), each V^T
    tile's groups of wg dimensions (chunk c the tile's tokens c D ..); the appends QST GRP.
    """

    def __init__(self, heads: dict, cap: int, d: int, D: int, S: int, sid: int,
                 dv: int | None = None, wg: int = 1):
        self.heads, self.cap, self.d, self.D, self.S, self.sid = heads, cap, d, D, S, sid
        self.dv = d if dv is None else dv
        self.wg = wg
        if wg > 1 and (cap % wg or vt_tile(cap) % D):
            raise CompileError(f"a grouped KV cache of {cap} tokens: whole groups of {wg} and "
                               f"V^T tiles of whole {D}-token blocks")

    def kv_heads(self, n_kv_heads: int) -> int:
        """The cache's KV heads: the model's n_kv_heads, or S of them where it has fewer (S a
        multiple: each of its heads on S / n_kv_heads slices, a copy each, its query heads split
        between them: layouts.head_parallel_attention_weights)."""
        return n_kv_heads if n_kv_heads % self.S == 0 else self.S

    def owned_heads(self, n_kv_heads: int):
        """This slice's heads of kv_heads(n_kv_heads). A head past the model's own (a copy for
        the slices beyond n_kv_heads) counts only where the cache was placed with it: a cache of
        the model's heads alone (runtime's KVCache) leaves those slices without one."""
        return [h for h in builtins.range(self.sid, self.kv_heads(n_kv_heads), self.S)
                if h < n_kv_heads or h in self.heads]

    def _h(self, h: int) -> dict:
        if h not in self.heads:
            raise CompileError(f"KV head {h} does not live on slice {self.sid}")
        return self.heads[h]

    def k(self, h: int) -> QTensor:
        """K [cap, d]: int8 rows, a token's scale row per row (k_scale_bytes): at d = D its
        fp32 scale alone (QST ROW, MM RSC), else an int8 scale row (QST: the token's S and an
        offset per block)."""
        e = self._h(h)
        rsc = self.d == self.D
        if self.wg > 1:
            from . import quant as Q
            return QTensor(Affine.of(e["k"]), Affine.of(e["ks"]), (self.cap, self.d),
                           Q.grp_row_bytes(self.d, "int8", self.D, self.wg),
                           Q.grp_scale_bytes(self.d, "int8", self.D, self.wg, rsc), self.D,
                           rsc=rsc, grp=self.wg)
        return QTensor(Affine.of(e["k"]), Affine.of(e["ks"]), (self.cap, self.d), self.d,
                       k_scale_bytes(self.d, self.D), self.D, rsc=rsc)

    def vt(self, h: int) -> QTensor:
        e = self._h(h)
        tb = vt_tile(self.cap)
        tiles = (tb, self.d * tb) if tb < self.cap else None
        if self.wg > 1:           # a tile's groups: its tb / D chunks of wg x D bytes each
            return QTensor(Affine.of(e["vt"]), None, (self.dv, self.cap), tb * self.wg, 0,
                           self.D, tiles=tiles, grp=self.wg)
        return QTensor(Affine.of(e["vt"]), None, (self.dv, self.cap), tb, 0, self.D,
                       tiles=tiles)

    def vt_column(self, h: int, t) -> Affine:
        """The address of token t's byte in V^T row 0 (the next rows are vt(h).rs apart)."""
        tb = vt_tile(self.cap)
        q, r = tile_split(t, tb)
        return Affine.of(self._h(h)["vt"]) + q * (self.d * tb) + r

    def vt_place(self, h: int, t) -> tuple:
        """(wg > 1) Token t's V^T tile (its address) and column in it."""
        tb = vt_tile(self.cap)
        q, r = tile_split(t, tb)
        return Affine.of(self._h(h)["vt"]) + q * (self.d * tb), Affine.of(r)

    def vscale(self, h: int) -> Tensor:
        e = self._h(h)
        return Tensor(Affine.of(e["vs"]), (self.cap,), (1,))


class _Diverge:
    """Builder.diverge's region."""

    def __init__(self, b):
        self.b = b

    def _live(self) -> set:
        return {(lo, hi) for lo, hi, r in self.b.tmem_regions if r() is not None}

    def __enter__(self):
        self.cursor, self.before = self.b.tmem_next, self._live()
        self.args = len(self.b.run_args)
        return self

    def __exit__(self, kind, *_):
        if kind is not None:
            return False
        gc.collect()                    # (tiles in reference cycles: the pipeline's closures)
        extra = self._live() - self.before
        if extra:
            raise CompileError(f"a slice-dependent region left {len(extra)} TMEM regions "
                               f"allocated ({sorted(extra)[:4]}): allocate them before it")
        if len(self.b.run_args) != self.args:
            raise CompileError(f"a slice-dependent region took the run-time arguments "
                               f"{self.b.run_args[self.args:]}: take them before it "
                               f"(Builder.args_ahead)")
        self.b.tmem_next = self.cursor
        return False


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

    # exp2(a - b, rowsum=True) on a machine with FSUM: the row sums the EXP2SUB writes with the
    # tile (its spare row, as the MXU's row maxima), its sum(axis=1) the take; else None
    # (Builder.fuse_exp2_rowsum)
    rowsum = None

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
        if key == (slice(None), None) and self.cols == 1:
            return Bcast(self, B_ROWVIEW)       # a column: B per row, down its row stride
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

    def rows_at(self, rows: int, cols: int, rs: int) -> "Tile":
        """Its words as a [rows, cols] tile at row stride rs, the rows free to overlap (rs <
        cols): an operand that is only read (kernels/gather.py's one-hot). Contiguous tiles."""
        n = self.rows * self.cols if self.contiguous else 0
        if rows < 1 or cols < 1 or rs < 1 or (rows - 1) * rs + cols > n:
            raise CompileError(f"rows_at({rows}, {cols}, {rs}) outside {self}")
        return Tile(self.b, self.base, (rows, cols), rs, self.buf)

    def reshape(self, rows: int, cols: int) -> "Tile":
        """The same words as a [rows, cols] tile (contiguous tiles only; one row always is)."""
        if not (self.contiguous or self.rows == 1) or rows * cols != self.rows * self.cols:
            raise CompileError(f"reshape: {self} is not {rows}x{cols} contiguous words")
        return Tile(self.b, self.base, (rows, cols), cols, self.buf)

    # ---- in-place update (loop-carried values)
    def set(self, value) -> "Tile":
        if value is self:               # written in place already (add)
            return self
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

    def add(self, o) -> "Tile":
        """self + o into self's words: the ADD `self.set(self + o)` ends with, without the temp
        `self + o` takes until set moves the ADD into self (where TMEM has room for it, the
        next tiles still go where they went after it: Builder.skip). Returns self."""
        self.b.skip(self.rows * self.cols)
        self.b.vop(I.V_ADD, self, self, o)
        self.b.bump_version(self.buf)
        return self

    def mac(self, a, b) -> "Tile":
        """self + a * b into self's words in one pass (VOP MAC, Config.VMAC): `self.set(self +
        a * b)`'s MUL then ADD bit for bit, b a column broadcast (v[None, :]) or a float.
        Returns self."""
        return self.b.mac(self, a, b)

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
    """A 1-D tile broadcast along columns (`v[:, None]`) or rows (`v[None, :]`); a 2-D
    tile's column ([n, 1], `t[r0:r1, c:c + 1][:, None]`) along columns, read down its row
    stride (VOP bmode 1's brs)."""

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
        if isinstance(self.src, _Parts):        # (filled in parts: its shape only)
            src = _Parts(self.src.b, self.src.rows, kb * D)
        else:
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


_INTERNAL = (os.path.join("opentpu", "compiler.py"), os.path.join("opentpu", "language.py"),
             os.path.join("opentpu", "stage.py"))


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

    unroll = False                      # language.range unrolled (a merged stage's stream: stage.py)

    def stage_wait(self, key) -> None:
        """A merged stage's stream: wait for the tile before it's stage_post(key) (stage.py)."""

    def stage_post(self, key) -> None:
        """A merged stage's stream: what the next tile's stage_wait(key) waits for (stage.py)."""

    def __init__(self, cfg: Config, sid: int):
        self.cfg, self.sid = cfg, sid
        self.root: list = []
        self.stack: list = [self.root]
        self.loops: list[LoopBlock] = []
        self.tmem_next = 0
        self.fit = "next"               # TMEM placement: next-fit, or "best" (trace_fit's retry)
        self.homes: dict = {}           # language.load_home's tiles, kept for the whole program
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
        # transient (with run_words): an address of c * var alone takes a scratch register for
        # the instruction that uses it (its RLD MUL there), not an argument register held from
        # the start (a step over slots: each row's values, batch35); pending: those registers,
        # each with the emitted count after its RLD, given back once an instruction followed
        self.transient = False
        self.pending: list = []
        self.emitted = 0
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
        # AW > 32 (docs/compiler.md "Addresses above 4 GiB"): a static address at or above 4 GiB
        # is the window register's base (LI'd where it moves: win_base, None after any loop's
        # start or end) plus an offset; a loop's address there a biased key's register (BIAS)
        self.aw = cfg.AW
        self.win_reg: int | None = None
        self.win_base: int | None = None
        self.end_zero: dict = {}      # loop id -> registers its body zeroes at its end (_take_reg)

    # ---- emission
    def emit(self, ins: I.Instr) -> None:
        ins.src = _user_frames()
        self.stack[-1].append(ins)
        self.emitted += 1

    def _transient(self, v: RunVar, c: int) -> int:
        """A scratch register holding c * v (RLD MUL of v's run word) for the next instruction
        (Builder.transient)."""
        self._drain()
        r = self.scratch()
        n = self.emitted
        for x in I.rld_mul(r, self.run_words[v.name], int(c), raw=v.raw,
                           bound=mul_bound(c, v.bound, self.aw), comment=f"{c}*{v}",
                           aw=self.aw):
            self.emit(x)
        self.emitted = n
        self.pending.append((r, n))
        return r

    def _drain(self) -> None:
        """Give back the transient registers whose instruction has been emitted."""
        if self.pending:
            n0 = self.emitted
            keep = [(r, n) for r, n in self.pending if n == n0]
            for r, n in self.pending:
                if n != n0:
                    self.unscratch(r)
            self.pending, self.emitted = keep, n0       # (the zeroing LIs are none's)

    def addr(self, a) -> tuple[int, int]:
        """An address as (register, offset): the instruction's R[register] + offset. At AW > 32
        the offset is a 32-bit word, zero-extended: the window register or a biased key's below
        an address at or above 4 GiB (docs/compiler.md "Addresses above 4 GiB")."""
        a = Affine.of(a)
        if a.is_static:
            if self.aw == 32 or 0 <= a.const < 1 << 32:
                return 0, a.const
            return self._window(a.const)
        r, c = self._addr(a)
        if self.aw > 32 and not 0 <= c < 1 << 32:
            raise CompileError(f"address {a}: R{r} + {c:#x} cannot be reached at AW {self.aw} "
                               f"(an offset of 32 bits, zero-extended; a device value's base "
                               f"goes into its register)")
        return r, c

    def _addr(self, a: Affine) -> tuple[int, int]:
        self._drain()
        run = [(v, c) for v, c in a.terms.items() if isinstance(v, RunVar)]
        for l in a.terms:
            if not isinstance(l, RunVar) and not any(lb.loop is l for lb in self.loops):
                raise CompileError(f"address uses {l} outside its loop")
        dev = None              # a Counter and a run-time value: the register of their sum
        ka = 0                  # (the argument register's excess: _host_arg)
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
                if words is not None and self.transient and len(a.terms) == 1:
                    if v.name not in words:
                        raise CompileError(f"run-time value {v}: no TMEM word (run_words)")
                    return self._transient(v, c), a.const
                # c * var alone takes an argument register; with run_words, once all of them
                # are taken, an address register loaded from var's word where it is used
                room = self._arg_key(v, c) in self.run_args or len(self.run_args) < 16 - ARG0
                if words is None or (len(a.terms) == 1 and room):
                    arg, ka = self._host_arg(v, c)
                    if len(a.terms) == 1 and (self.aw == 32 or 0 <= a.const - ka < 1 << 32):
                        return arg, a.const - ka    # the argument register itself
        key, const = frozenset(a.terms.items()), a.const
        bias = None             # (a host argument's register is set before the loops)
        host = run and dev is None and words is None and not isinstance(run[0][0], DevVar)
        if self.aw > 32 and not 0 <= const < 1 << 32 and \
                (host or any(isinstance(l, Loop) for l in a.terms)):
            bias = self._bias(key, const)
            key, const = key | {(BIAS, bias)}, const - bias
        if key not in self.regs:
            r = self._spare_for(key)
            if r is None:
                r = self._free_for(key)
                b = bias or 0
                if r is not None and run:          # it starts at the run-time value
                    (v, c), what = run[0], " + ".join(f"{c_}*{v_}" for v_, c_ in run)
                    if dev is not None or words is None or isinstance(v, DevVar):
                        init = [I.addi(r, arg, b - ka, comment=f"{what} (argument)",
                                       aw=self.aw)]
                    elif v.name not in words:
                        raise CompileError(f"run-time value {v}: no TMEM word (run_words)")
                    else:
                        init = I.rld_mul(r, words[v.name], int(c), raw=v.raw,
                                         bound=mul_bound(c, v.bound, self.aw), comment=what,
                                         aw=self.aw)
                        if b:
                            init.append(I.addi(r, r, b, comment="+ base", aw=self.aw))
                    self._init_before_loops(key, init)
                elif r is not None and bias is not None:      # it starts at the base
                    self._init_before_loops(key, [I.li(r, bias, comment="address base",
                                                       aw=self.aw)])
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
        return self.regs[key], const

    def _bias(self, key, const: int) -> int:
        """The base B of a loop's address at or above 4 GiB (or below 0) at AW > 32: one a live
        or spare register of the same terms holds with 0 <= const - B < 2^32, else const rounded
        down to BIAS_ALIGN."""
        fits = lambda b: 0 <= const - b < 1 << 32                         # noqa: E731
        for k in self.regs:
            b = dict(k).get(BIAS)
            if b is not None and fits(b) and k - {(BIAS, b)} == key:
                return b
        for terms, _, _ in self.spare_regs:
            b = dict(terms).get(BIAS)
            if b is not None and fits(b) and terms - {(BIAS, b)} <= key:
                return b
        return const - const % BIAS_ALIGN

    def _window(self, const: int) -> tuple[int, int]:
        """A static address at or above 4 GiB (AW > 32): the window register's base plus an
        offset below 4 GiB. The base moves (an LI here: the address rounded down to BIAS_ALIGN)
        when the address is outside the window, or after a loop's start or end since it was set
        (the body runs again after its end: an LI inside it is the window at the body's start)."""
        if const < 0 or const >= 1 << self.aw:
            raise CompileError(f"address {const:#x} outside the {self.aw}-bit DRAM")
        b = self.win_base
        if b is None or not 0 <= const - b < 1 << 32:
            if self.win_reg is None:
                self.win_reg = self._take_reg("the window register")
            b = self.win_base = const - const % BIAS_ALIGN
            self.emit(I.li(self.win_reg, b, comment="window", aw=self.aw))
        return self.win_reg, const - b

    def _take_reg(self, what: str) -> int:
        """A free register for good (the window register): any value it held before is not
        needed (a released argument's is overwritten here, not zeroed at the start). One freed
        while none of the live loops was, if any: a register freed inside a live loop's body
        holds 0 where that body starts again for the address that used it there, so one set
        here for good is zeroed at the end of those loops' bodies (end_zero)."""
        if not self.free_regs:
            raise CompileError(f"out of registers for {what}")
        live = {id(lb.loop) for lb in self.loops}
        i = next((i for i in builtins.range(len(self.free_regs) - 1, -1, -1)
                  if not self.free_regs[i][1] & live), len(self.free_regs) - 1)
        r, live_then = self.free_regs.pop(i)
        for lid in live_then & live:
            self.end_zero.setdefault(lid, set()).add(r)
        if r in self.dirty:
            self.dirty.discard(r)
            self.late_zero.add(r)
        self.unset.discard(r)
        self.used_regs.add(r)
        return r

    def _arg_key(self, v: RunVar, c: int) -> tuple:
        """The (variable, coefficient) of the argument that serves c * v (_host_arg)."""
        if self.aw == 32 or c >= 0 or self.run_words is not None or isinstance(v, DevVar):
            return v, c
        return v.comp(), -c

    def _host_arg(self, v: RunVar, c: int) -> tuple[int, int]:
        """(register, K): the argument register that serves c * v, holding c * v + K. Above AW
        32 the host's argument words are zero-extended (docs/isa.md "Address bits"), so c * v
        with c < 0 is the argument -c * Comp(v) (v's bound - 1 - v), K = -c (bound - 1); the
        program's own loads of its arguments (run_words: RLD MUL, isa.rld_mul) are AW bits and
        take c * v as it is."""
        var, k = self._arg_key(v, c)
        if var is v:
            return self.arg_reg(v, c), 0
        if not v.bound:
            raise CompileError(f"{c}*{v} at AW {self.aw}: a negative argument needs {v}'s "
                               f"bound (argument words are zero-extended)")
        return self.arg_reg(var, k), k * (v.bound - 1)

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
        self.root[i:i] = [I.addi(q, r, 0, comment=f"argument {k}", aw=self.aw),
                          I.li(r, 0, comment="address register", aw=self.aw)]
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
            if (v is var or getattr(v, "var", None) is var) and (v, c) not in self.released:
                r = self.moved.get(k, arg_reg(k))
                self.released.add((v, c))
                self.dirty.add(r)
                self.release_at[r] = at
                self.free_regs.insert(0, (r, frozenset()))

    def _init_before_loops(self, key, ins) -> None:
        """Put `ins` (an instruction or a list of them) before the outermost live loop among
        key's terms (the register must hold its run-time value when that loop starts; the loops
        step and reset it from there), or, for c * var alone (run_words: the argument registers
        all taken; or a host argument's with a base above 4 GiB, BIAS), right here."""
        ins = [ins] if isinstance(ins, I.Instr) else list(ins)
        for d, lb in enumerate(self.loops):
            if (lb.loop, dict(key).get(lb.loop)) in key:
                items = self.stack[d]
                i = next(i for i, x in enumerate(items) if x is lb)
                items[i:i] = ins
                self._set(ins[-1].rd, self.loops[:d])
                return
        bare = all(isinstance(l, RunVar) for l, _ in key)        # (BIAS is one)
        if bare and (len(key) == 1 and self.run_words is not None or
                     self.run_words is None and BIAS in dict(key)):
            self.stack[-1].extend(ins)
            self._set(ins[-1].rd, self.loops)
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
        # none: a spare that holds a base alone (its loops ended, BIAS), zeroed when taken
        for i, (terms, live_then, r) in enumerate(self.spare_regs):
            if all(l is BIAS for l, _ in terms) and \
                    not any(id(l) in live_then for l, _ in key) and \
                    not (run and (live_then & live)):
                del self.spare_regs[i]
                self.unset.add(r)
                return r
        return None

    def begin_loop(self, n: int, rcount: int = 0) -> Loop:
        """A hardware loop of n iterations, or of R[rcount] (n = 0) read when it starts."""
        self.emitted += 1                   # (a transient register's instruction is before it)
        self._drain()
        if len(self.loops) >= 4:
            raise CompileError("loop nesting deeper than 4")
        lb = LoopBlock(Loop(n, rcount))
        self.stack[-1].append(lb)
        self.stack.append(lb.items)
        self.loops.append(lb)
        self.win_base = None
        return lb.loop

    def end_loop(self, loop: Loop) -> None:
        self.emitted += 1
        self._drain()
        for r in sorted(self.end_zero.pop(id(loop), ())):
            self.emit(I.li(r, 0, comment="register reset (the next pass)"))
        lb = self.loops.pop()
        assert lb.loop is loop
        self.stack.pop()
        self.win_base = None
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
        self._drain()
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

    def borrow(self) -> tuple[int, tuple | None]:
        """A register for the caller to set and step (a loop of a run-time count stepping its own:
        attention.RunSpan), and how to give it back (give_back): a scratch register, or, none
        being free, a spare whose terms another register holds too (that one's copy restores
        it: equal terms, equal values)."""
        self._drain()
        if self.free_regs:
            return self.scratch(), None
        pinned = getattr(self, "pinned", set())
        for i, (terms, live_then, r) in enumerate(self.spare_regs):
            if r in self.unset or r in self.set_in or r in pinned:
                continue
            twin = next((q for k, q in self.regs.items() if k == terms and q not in self.unset),
                        None)
            if twin is None:
                twin = next((q for t, _, q in self.spare_regs
                             if t == terms and q != r and q not in self.unset), None)
            if twin is not None:
                del self.spare_regs[i]
                self.pinned = pinned | {twin}
                self.used_regs.add(r)
                return r, (terms, live_then, twin)
        raise CompileError("out of registers for a device-computed value")

    def give_back(self, r: int, how: tuple | None) -> None:
        """A borrowed register back (borrow): unscratched, or the spare its twin restores."""
        if how is None:
            self.unscratch(r)
            return
        terms, live_then, twin = how
        self.emit(I.addi(r, twin, 0, comment="the spare register back", aw=self.aw))
        self.spare_regs.append((terms, live_then, r))
        self.pinned = self.pinned - {twin}

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
            arg, ka = self._host_arg(v, c)
            init = [I.addi(r, arg, -ka, comment=f"{what} (argument)", aw=self.aw)]
        elif v.name not in self.run_words:
            raise CompileError(f"run-time value {v}: no TMEM word (run_words)")
        else:
            init = I.rld_mul(r, self.run_words[v.name], int(c), raw=v.raw,
                             bound=mul_bound(c, v.bound, self.aw), comment=what, aw=self.aw)
        if len(self.loops) > k.depth:
            lb, items = self.loops[k.depth], self.stack[k.depth]
            j = next(j for j, x in enumerate(items) if x is lb)
            items[j:j] = init
        else:
            for x in init:
                self.emit(x)
        k.derived[(v, c)] = r
        return r

    def step(self, c: Counter, by: int) -> None:
        """c += by: ADDI (to c's register and each of c and a run-time value's). The address
        registers that hold an address of its earlier value (spare ones: their loops have
        ended) are free again, zeroed when taken; one of a live loop's could not follow the
        step (the loop resets it to its init)."""
        self._drop(c)
        for r in (c.reg, *c.derived.values()):
            self.emit(I.addi(r, r, self._wrap(int(by)), comment=f"{c.name} + {by}", aw=self.aw))
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
        docs/isa.md; rld(raw=True) takes it into a register)."""
        self.check_live(t)
        if t.rows * t.cols != 1:
            raise CompileError(f"waitw: {t} is not one word")
        self.emit(I.waitw(dram, t.base, ref, cmp, ra=ra, rc=rc, interval=interval,
                          timeout=timeout, comment=comment or "waitw"))

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
    def alloc(self, shape, rs: int | None = None, spare: int = 0, top: bool = False) -> Tile:
        """A new tile; `spare` extra words are reserved right after its last row (the MXU
        writes RMAX row maxima there). top: the highest free words instead (_tmem_top)."""
        shape = tuple(int(x) for x in shape)
        rs = shape[-1] if rs is None else rs
        n = (shape[0] - 1) * rs + shape[-1] if len(shape) == 2 else shape[0]
        if spare:
            n = max(n, shape[0] * rs + spare) if len(shape) == 2 else n + spare
        base = self._tmem_top(n) if top else self._tmem_fit(n)
        t = Tile(self, base, shape, rs)
        t.spare = spare
        self.tmem_regions.append((base, base + n, weakref.ref(t.buf)))
        self.tmem_regions.sort(key=lambda r: r[0])
        self.tmem_peak = max(self.tmem_peak, sum(e - b for b, e, _ in self.tmem_regions))
        return t

    def free_now(self, t: "Tile") -> None:
        """t's words free now, while the caller still holds t, which is not read again."""
        t.dead = True
        self.tmem_regions = [r for r in self.tmem_regions if r[2]() is not t.buf]

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

    def diverge(self) -> "_Diverge":
        """A slice-dependent region of the program (`with b.diverge():`): its allocations may
        differ by slice, so as it ends the TMEM allocator is put back as it was (the cursor;
        every region allocated in it must be free again, else CompileError), and the slices'
        later allocations -- their GATHERs' destinations among them, which must agree --
        come out as before it. What outlives the region is allocated before it."""
        return _Diverge(self)

    def args_ahead(self, a) -> None:
        """Address a's run-time argument taken now, as its first use takes it (_addr: one
        run-time value of the host's): a slice-dependent region (diverge) that uses a on some
        slices only then leaves every slice's run_args alike."""
        a = Affine.of(a)
        run = [(v, c) for v, c in a.terms.items() if isinstance(v, RunVar)]
        if len(run) != 1 or isinstance(run[0][0], (DevVar, Counter)):
            raise CompileError(f"address {a}: args_ahead takes one run-time value of the host's")
        v, c = run[0]
        room = self._arg_key(v, c) in self.run_args or len(self.run_args) < 16 - ARG0
        if self.run_words is None or (len(a.terms) == 1 and room):
            self._host_arg(v, c)

    def _tmem_top(self, n: int) -> int:
        """The highest gap's last n words, the cursor left as it is: a small tile kept for the
        whole program (attention's variant table) apart from next-fit's turn through TMEM, whose
        gaps it would split (a decode loop's head then finds no gap for its logits' chunks)."""
        W = self.cfg.TMEM_WORDS - (self.desc_area if self.cfg.TMEM_WORDS <= 1 << 16 else 0)
        self.tmem_regions = [r for r in self.tmem_regions if r[2]() is not None]
        e = W
        for rb, re, _ in reversed(self.tmem_regions):
            if e - re >= n:
                return e - n
            e = min(e, rb)
        if e >= n:
            return e - n
        live = sum(re - rb for rb, re, _ in self.tmem_regions)
        raise CompileError(f"TMEM exhausted: a tile of {n} words, {live} of {W} in use "
                           f"(TMEM_WORDS {self.cfg.TMEM_WORDS})")

    def skip(self, n: int) -> None:
        """The next-fit cursor as alloc of n words and the tile's death would leave it, no
        region held (Tile.add's temp that is not allocated); unchanged where n words do not
        fit."""
        try:
            self._tmem_fit(n)
        except CompileError:
            pass

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
            if b.kind == B_ROWVIEW:             # (a 1-D tile, or a 2-D one's column)
                n, brs = (v.rows, v.rs) if len(v.shape) == 2 else (v.cols, 1)
                if n != rows:
                    raise CompileError(f"row broadcast of length {n} over {rows} rows")
                bmode, b_base = I.B_ROW, v.base
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

    def unop(self, func: int, x, temp: bool = False, rowsum: bool = False) -> Tile:
        x = self.materialize(x)
        self.check_live(x)
        if func == I.V_EXP2 and temp and self.fuse_exp2_sub(x):
            out = self.fuse_exp2_rowsum(x) if rowsum else None
            if out is None:
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

    def fuse_exp2_rowsum(self, x: Tile) -> Tile | None:
        """exp2(a - b, rowsum=True), just fused into EXP2SUB (x the temp it writes): on a machine
        with FSUM (Config.FSUM, docs/isa.md VOP) and a shape it takes (2-D, rows <= 256, cols >=
        64, B not a scalar), the EXP2SUB writes a new tile with a spare row instead of x, and
        its flag FSUM the rows' isum_64 into that row (the RSUM's sums of the results, bit for
        bit; the sequencer's hull of the two ranges is the tile's own): the tile, its `rowsum`
        the sums, which the tile's sum(axis=1) takes (reduce: VOP RSUM FSUM where the RSUM
        would be; nothing touches the row before). None where it does not apply (the caller's
        sum is an RSUM)."""
        ins = self._last_vop_writing(x)
        if (not self.cfg.FSUM or ins is None or len(x.shape) != 2 or ins.ra or ins.rd
                or x.rows > I.FSUM_MAX_ROWS or x.cols < I.FSUM_MIN_COLS
                or (ins.w[5] >> 24) & 3 == I.B_SCALAR):
            return None
        out = self.alloc(x.shape, spare=x.rows)
        ins.w[0] = out.base
        ins.w[4] = (ins.w[4] & ~0xFFFF) | out.rs
        ins.w[6] = out.base + x.rows * out.rs
        ins.flags |= I.F_FSUM
        ins.comment = "exp2sub fsum"
        out.rowsum = Tile(self, ins.w[6], (x.rows,), None, out.buf)
        return out

    def retarget(self, value: Tile, dst: Tile) -> bool:
        """`dst.set(value)`: if `value` is a fresh temp written by the last instruction, make
        that instruction write `dst` directly (no copy). The temp is then dead."""
        if not getattr(value, "fresh", False) or value.shape != dst.shape:
            return False
        body = self.stack[-1]
        if body and isinstance(body[-1], I.Instr) and body[-1].op == I.GATHER:
            return self._retarget_gather(body[-1], value, dst)
        ins = self._last_vop_writing(value)
        if ins is None or (ins.w[5] >> 16) & 0xFF in I.REDUCE | {I.V_MAC}:   # (MAC reads dst)
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

    def _retarget_gather(self, ins: I.Instr, value: Tile, dst: Tile) -> bool:
        """retarget's GATHER: `dst.set(all_gather(x))` gathers into dst (every slice's, the
        program being the same) when the temp is the GATHER's whole destination and x's words
        are not in dst."""
        rows, cols, srs, seg = ins.w[2] & 0xFFFF, ins.w[2] >> 16, ins.w[3], ins.w[5]
        if ins.ra or ins.rb or ins.w[1] != value.base or rows != value.rows or seg != cols \
                or value.cols != cols * self.cfg.S:
            return False
        drs = dst.rs if len(dst.shape) == 2 else ins.w[4]
        if ins.w[0] < dst.base + (rows - 1) * drs + dst.cols and \
                dst.base < ins.w[0] + (rows - 1) * srs + cols:
            return False
        ins.w[1], ins.w[4] = dst.base, drs
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
        if func == I.V_RSUM and x.rowsum is not None:     # FSUM's sums: their take
            out, x.rowsum = x.rowsum, None
            self.emit(I.fsum_take(out.base, x.base, x.rows, x.cols, x.rs))
            return out
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
        if t.esize != 4:
            raise CompileError("load: a 16-bit state (ST16) is DSTEP's and STREAM's alone")
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
            ra, imm = self.addr(t.base)
            n = (t.shape[0] - 1) * out.rs + t.shape[1]
            self.emit(I.ld(imm, out.base, n, ra=ra, comment="load rows"))
            return out
        if len(t.shape) == 1:
            if t.strides[0] != 1:
                raise CompileError("strided 1-D loads are not supported")
            ra, imm = self.addr(t.base)
            self.emit(I.ld(imm, out.base, t.shape[0], ra=ra, comment="load"))
            return out
        if len(t.shape) != 2 or t.strides[1] != 1:
            raise CompileError("loads must be 1-D or row-major 2-D")
        if not by_row:
            ra, imm = self.addr(t.base)
            self.emit(I.ld(imm, out.base, t.shape[0] * t.shape[1], ra=ra, comment="load"))
        else:
            for r in builtins.range(t.shape[0]):
                ra, imm = self.addr(t.base + r * 4 * t.strides[0])
                self.emit(I.ld(imm, out.base + r * out.rs, t.shape[1], ra=ra, comment="load row"))
        return out

    def store(self, t: Tensor, x) -> None:
        if t.esize != 4:
            raise CompileError("store: a 16-bit state (ST16) is DSTEP's and STREAM's alone")
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

    def quantize(self, x: Tile, temp: bool = False, free: bool = False,
                 rscale: Tile | None = None, cscale: Tile | None = None,
                 lo: bool = False, hl: bool = False, store=None):
        """QACT x's rows into ACT RAM blocks. free: the stationary does not keep x (its TMEM is
        free once the QACTs have read it), only its shape: operands kept in ACT RAM while TMEM
        is reused (a tiled prefill's MLP activations). rscale [rows] / cscale [cols]: quantize
        (x * rscale[:, None]) * cscale[None, :] (QACT RSCALE / CSCALE), the instruction the
        peephole makes of that expression, without its product tiles. lo: the split's lo part
        (QACT flag LO, Config.SPLITQ). hl: (hi, lo), two stationaries from one QACT a chunk
        (flag HL, Config.QHL: lo in the KB blocks after hi's). store: (base, stride), each row's
        ACT bytes and records also to DRAM, row r at the byte address base + r stride (QACT ST,
        Config.QONCE: the operand act_load loads back, isa.qrow_bytes a row)."""
        if store is not None and (not self.cfg.QONCE or hl):
            raise CompileError("quantize(store=): the machine has no QONCE (QACT flag ST), or hl")
        if store is not None and store[1] < I.qrow_bytes(-(-x.cols // self.cfg.D), self.cfg.D):
            raise CompileError(f"quantize(store=): rows {store[1]} bytes apart, each "
                               f"{I.qrow_bytes(-(-x.cols // self.cfg.D), self.cfg.D)}")
        if lo and not self.cfg.SPLITQ:
            raise CompileError("quantize(lo=True): the machine has no SPLITQ (QACT flag LO)")
        if hl and not (self.cfg.QHL and self.cfg.SPLITQ) or hl and lo:
            raise CompileError("quantize(hl=True): the machine has no QHL (QACT flag HL), or lo")
        x = self.materialize(x)
        self.check_live(x)
        D = self.cfg.D
        if x.cols % D:
            raise CompileError(f"reduction dim {x.cols} must be a multiple of D={D}")
        KB = x.cols // D
        if KB > QACT_KB_MAX:
            raise CompileError(f"QACT of {KB} blocks: at most {QACT_KB_MAX} (its field); quantize "
                               f"the operand in ranges")
        if hl and x.cols > I.QHL_ELEMS:
            raise CompileError(f"quantize(hl=True) of {x.cols} columns: at most {I.QHL_ELEMS}")
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
        pair = self.cfg.PAIR and 2 * x.rows <= self.cfg.MCOLS and not hl
        st = Stationary(x, chunks, KB, owners, pair)
        sl = Stationary(x, [], KB, [], pair) if hl else None      # (hl: lo's, KB blocks on)
        for m0 in builtins.range(0, x.rows, self.cfg.act_rows):
            mc = min(self.cfg.act_rows, x.rows - m0)
            ab, owner = self.act_alloc(2 * KB if hl else KB)
            for k in builtins.range(ab, ab + KB):   # live now: the next chunk must not take it
                self.act_live[k] = weakref.ref(st)
                if hl:
                    self.act_live[k + KB] = weakref.ref(sl)
            sto, rc = None, 0
            if store is not None:                   # (QACT ST: the chunk's rows to DRAM)
                rc, imm = self.addr(Affine.of(store[0]) + m0 * store[1])
                sto = (imm, store[1])
            self.emit(I.qact(src + m0 * rs, mc, ab, KB, rs, cscale=cs,
                             rscale=None if rsc is None else rsc + m0, dup=pair, lo=lo, hl=hl,
                             store=sto, rc=rc,
                             comment="quantize -> ACT" + (" lo" if lo else "")
                             + (" + DRAM" if store is not None else "")
                             + (" hi + lo" if hl else "")
                             + (" x col scale" if cs is not None else "")
                             + (" x row scale" if rsc is not None else "")
                             + (" dup" if pair else "")))
            chunks.append((ab, mc))
            owners.append(owner)
            if hl:
                sl.chunks.append((ab + KB, mc))
                sl.owners.append(owner)
        for s in (st, sl) if hl else (st,):
            s.loop = self.loops[-1].loop if self.loops else None  # created in this loop body
            if free:
                s.src = _Parts(self, x.rows, x.cols, x.shape)
        return (st, sl) if hl else st

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
                      at=None, rscale: Tile | None = None) -> None:
        """Quantize the rows of x into the blocks of `st` (from stationary()) from column
        `col` on (QACT at the part's first block). cscale [cols] / rscale [rows]:
        (x * rscale[:, None]) * cscale[None, :] (QACT CSCALE / RSCALE, as quantize's). at: a
        TMEM word offset with loop terms added to x's address (QACT's src = R[ra] + w1: x's
        rows in a buffer the loop walks, the looped attention's query tiles,
        tiled._attend_loop); x is then those rows at offset 0."""
        x = self.materialize(x)
        self.check_live(x, cscale, rscale)
        D = self.cfg.D
        if not isinstance(st.src, _Parts):
            raise CompileError("quantize_into: a stationary from stationary()")
        if x.rows != st.src.rows or x.cols % D or col % D or col + x.cols > st.KB * D:
            raise CompileError(f"quantize_into: a {x.shape} tile at column {col} of a "
                               f"{st.src.shape} stationary")
        if cscale is not None and cscale.shape != (x.cols,):
            raise CompileError(f"quantize_into: a column scale of {cscale.shape} for {x.shape}")
        if rscale is not None and rscale.shape != (x.rows,):
            raise CompileError(f"quantize_into: a row scale of {rscale.shape} for {x.shape}")
        if not self.stationary_valid(st):
            raise CompileError("stationary operand was overwritten in ACT RAM")
        (ab, mc), = st.chunks
        rs = x.rs if len(x.shape) == 2 else x.cols
        ra, src = (0, x.base) if at is None else self.addr(Affine(x.base) + Affine.of(at))
        self.emit(I.qact(src, mc, ab + col // D, x.cols // D, rs, dup=st.pair, ra=ra,
                         cscale=None if cscale is None else cscale.base,
                         rscale=None if rscale is None else rscale.base,
                         comment="quantize -> ACT (part)" + (" dup" if st.pair else "")
                         + (" x col scale" if cscale is not None else "")
                         + (" x row scale" if rscale is not None else "")))
        st.add_qrange(col // D, (col + x.cols) // D)

    def act_load(self, st: Stationary, row: int, src, rows: int = 1, stride: int = 0) -> None:
        """Rows row .. row + rows - 1 of the stationary `st` (from stationary(), all its blocks)
        from DRAM: what quantize(store=) wrote, its first row at the byte address src, the
        others stride bytes on (LDA, Config.QONCE; with PAIR its rows' copies too). The
        operand's bytes and scales are the stored QACT's: a dot over it gives that QACT's
        operand's results bit for bit."""
        D = self.cfg.D
        if not self.cfg.QONCE:
            raise CompileError("act_load: the machine has no QONCE (LDA)")
        if not isinstance(st.src, _Parts) or len(st.chunks) != 1:
            raise CompileError("act_load: a stationary from stationary()")
        (ab, mc), = st.chunks
        if not 0 <= row < row + rows <= mc:
            raise CompileError(f"act_load: rows {row} .. {row + rows - 1} of a {mc}-row operand")
        n = I.qrow_bytes(st.KB, D)
        if rows > 1 and stride < n:
            raise CompileError(f"act_load: rows {stride} bytes apart, each {n}")
        if not self.stationary_valid(st):
            raise CompileError("stationary operand was overwritten in ACT RAM")
        ra, imm = self.addr(Affine.of(src))
        for r0 in (row, row + mc) if st.pair else (row,):   # (PAIR: the copies QACT DUP writes)
            self.emit(I.lda(imm, rows, ab, st.KB, stride or n, arow=r0, ra=ra,
                            comment="DRAM -> ACT" + (" (dup)" if r0 != row else "")))
        st.add_qrange(0, st.KB)

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

    def mac(self, dst: Tile, a: Tile, b) -> Tile:
        """dst += a * b in place (VOP MAC: dst read and written, b a column broadcast of at most
        I.MAC_MAX_COLS columns or a float): the MUL then ADD of `dst.set(dst + a * b)`, bit for
        bit, in one pass."""
        if not self.cfg.VMAC:
            raise CompileError("mac needs VOP MAC, which this configuration's VPU lacks "
                               "(Config.VMAC off: a target built without it)")
        if not isinstance(a, Tile) or len(dst.shape) != 2:
            raise CompileError(f"mac: dst {dst!r} must be a 2-D tile and a a tile")
        col = isinstance(b, Bcast) and b.kind != B_ROWVIEW
        if not (col or isinstance(b, (int, float))) or (col and dst.cols > I.MAC_MAX_COLS):
            raise CompileError(f"mac: b must be a column broadcast of at most "
                               f"{I.MAC_MAX_COLS} columns (v[None, :]) or a float, not {b!r}")
        self.vop(I.V_MAC, dst, a, float(b) if isinstance(b, (int, float)) else b)
        self.bump_version(dst.buf)
        return dst

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
        if not (self.cfg.OUTER or self.cfg.OUTER1 and decay is None):
            raise CompileError("outer(acc=...) needs VOP OUTER, which this configuration's VPU "
                               "lacks (Config.OUTER off: a target built without it, e.g. "
                               "bcu1525-wg, whose DeltaNet runs deltanet_step on DSTEP; "
                               "Config.OUTER1: decay one only)")
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
        (docs/stream.md). `zero`: the state reads as +0. A 16-bit state or dst (esize 2,
        Config.ST16) takes the flag ST16 on its side, on the hardware's subset
        (isa.stream_hw_cfg): both 16-bit ST16, one of them ST16R / ST16W (dst then apart from
        the state; docs/isa.md "ST16")."""
        if not self.cfg.STREAM:
            raise CompileError("STREAM needs Config.STREAM (the stream engine)")
        h16 = self._st16(state, "stream")
        w16 = h16 if dst is None else self._st16(dst, "stream")
        if h16 != w16 and not self.cfg.ST16S:
            raise CompileError("stream: a 16-bit state into an fp32 one or back needs "
                               "Config.ST16S (STREAM's ST16R / ST16W)")
        if (h16 or w16) and I.stream_hw_cfg(d) is None:
            raise CompileError("stream: a 16-bit state (ST16) needs the hardware's subset")
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
            # (apart: the lower one's bytes; in place only with both ends' element size)
            low = state.esize if gap > 0 else dst.esize
            if (gap or h16 != w16) and (gap % self.cfg.D or abs(gap) < low * d.rows * d.cols):
                raise CompileError("stream: dst must be chunk aligned and not overlap the state")
        at = self.stream_desc(d)
        ra, imm = self.addr(state.base)
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
                           zero=zero, ra=ra, rb=rb, rc=rc, rd=rd, st16r=h16, st16w=w16,
                           comment=f"stream {d.rows}x{d.cols}"))
        if o is not None:
            self.bump_version(o.buf)

    def _st16(self, state: Tensor, what: str) -> bool:
        """A DSTEP's / STREAM's state in 16 bits (esize 2): the flag ST16, on a Config.ST16
        machine (docs/isa.md "ST16")."""
        if state.esize not in (2, 4):
            raise CompileError(f"{what}: a state of {state.esize}-byte elements")
        if state.esize == 2 and not self.cfg.ST16:
            raise CompileError(f"{what}: a 16-bit state needs Config.ST16")
        return state.esize == 2

    def deltanet_step(self, state: Tensor, qk: Tile, v: Tile, decay: Tile, beta: Tile,
                      o: Tile, zero: bool = False, dst: Tensor | None = None) -> None:
        """DSTEP: one Gated DeltaNet head step on the fp32 state `state` [rows, cols] in DRAM,
        updated in place; qk = [q | k] (2 * cols words), v [rows], decay and beta [1] tiles
        (beta after decay), o [rows] written (docs/isa.md). `zero`: the state starts at +0.
        Without Config.DSTEP, a Config.STREAM machine runs it as STREAM with isa.gdn_desc
        (bit-identical). `dst`: the new state goes there instead and `state` keeps the old one
        (a STREAM, which has a destination; Config.STREAM). A 16-bit state (esize 2): flag
        ST16 (Config.ST16)."""
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
        ra, imm = self.addr(state.base)
        self.emit(I.dstep(imm, qk.base, v.base, rows, cols, decay.base, gs, o.base, zero=zero,
                          ra=ra, st16=self._st16(state, "deltanet_step"), comment="dstep"))
        self.bump_version(o.buf)

    def deltanet_step2(self, state: Tensor, qk: Tile, v: Tile, decay: Tile, beta: Tile,
                       o: Tile, zero: bool = False) -> None:
        """Two Gated DeltaNet heads in one DSTEP (Config.DSTEP2: the PAIR flag): `state` [2 h,
        cols] head a's rows, then head b's; qk [q | k] of one key head for both (2 cols) or
        head a's then head b's (4 cols); v and o [2 h]; decay and beta [2] (head a's, b's; beta
        after decay). Without DSTEP2, or off a pair's shape (cols 128, rows a multiple of 16),
        deltanet_step for each head (the same words)."""
        rows, cols = state.shape
        if rows % 2 or not isinstance(qk, Tile) or qk.cols not in (2 * cols, 4 * cols) or \
                any(not isinstance(t, Tile) or len(t.shape) != 1 for t in (v, o, decay, beta)):
            raise CompileError("deltanet_step2: [2 h, cols] state, qk of 2 or 4 cols, 1-D tiles")
        if decay.cols != 2 or beta.cols != 2 or v.cols != rows or o.cols != rows:
            raise CompileError("deltanet_step2: decay and beta [2], v and o [2 h]")
        h, qkb = rows // 2, qk.cols == 4 * cols
        if not (self.cfg.DSTEP and self.cfg.DSTEP2) or cols != I.DSTEP_PAIR_COLS or \
                rows % I.DSTEP_PAIR_ROWS:
            for j in range(2):
                self.deltanet_step(Tensor(Affine.of(state.base) + state.esize * j * h * cols,
                                          (h, cols), (cols, 1), esize=state.esize),
                                   qk[2 * j * cols:2 * (j + 1) * cols] if qkb else qk,
                                   v[j * h:(j + 1) * h], decay[j:j + 1], beta[j:j + 1],
                                   o[j * h:(j + 1) * h], zero)
            return
        if len(state.shape) != 2 or state.strides != (cols, 1) or \
                not (0 < rows <= I.DSTEP_MAX_ROWS):
            raise CompileError(f"deltanet_step2: a row-major state of <= {I.DSTEP_MAX_ROWS} rows")
        gs = beta.base - decay.base
        if not 1 < gs < 1 << 16:
            raise CompileError("deltanet_step2: beta must follow decay in TMEM")
        self.check_live(qk, v, decay, beta, o)
        if Affine.of(state.base).is_static and Affine.of(state.base).const % self.cfg.D:
            raise CompileError("deltanet_step2: the state must be DRAM-chunk aligned")
        ra, imm = self.addr(state.base)
        self.emit(I.dstep(imm, qk.base, v.base, rows, cols, decay.base, gs, o.base, zero=zero,
                          pair=True, qkb=qkb, ra=ra, st16=self._st16(state, "deltanet_step2"),
                          comment="dstep pair"))
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
        if ins.rd and not self.cfg.GALL:         # a grouped matrix's MM takes no RMAX
            return None
        ins.flags |= I.F_RMAX
        ins.comment += " +rowmax"
        return Tile(self, x.base + M * ors, (M,), None, x.buf)

    def dot_split(self, a: tuple, w: tuple, acc: Tile | None = None,
                  out: Tile | None = None) -> Tile:
        """A split product (Config.SPLITQ's operands): a = (hi, lo) stationaries and w = (hi, lo)
        streamed matrices, hi.hi + hi.lo + lo.hi; a = (hi, None): hi.hi + hi.lo (the transposes'
        identity against split rows). With Config.SPLMM one MM where the operands sit as SPL
        takes them (docs/isa.md "Split terms": lo's stationary the block after hi's, QACT HL;
        w's lo rows a constant HLD after hi's, Scratch's layout; one K-block of int8 RSC
        groups); otherwise the dots, hi.hi first (overwrite or acc=), then hi.lo, then lo.hi."""
        (ah, al), (wh, wl) = a, w
        hld = self._spl(ah, al, wh, wl)
        if hld:
            return self.dot(ah, wh, acc=acc, out=out, spl=(al, hld))
        if acc is None:
            acc = self.dot(ah, wh, out=out)
        else:
            self.dot(ah, wh, acc=acc)
        self.dot(ah, wl, acc=acc)
        if al is not None:
            self.dot(al, wh, acc=acc)
        return acc

    def _spl(self, ah, al, wh: QTensor, wl: QTensor) -> int:
        """dot_split's HLD, wl's rows after wh's (0: not one MM SPL)."""
        cfg = self.cfg
        if not (cfg.SPLMM and cfg.GALL and cfg.WG > 1 and isinstance(ah, Stationary)
                and (al is None or isinstance(al, Stationary))
                and isinstance(wh, QTensor) and isinstance(wl, QTensor)):
            return 0
        d = Affine.of(wl.data) - Affine.of(wh.data)
        hld = d.const
        if (d.terms or hld <= 0 or hld % 1024 or hld % (cfg.WG * cfg.D)
                or hld >> 10 >= 1 << (31 - I.MM_HLD)):
            return 0
        if any(x.wf != I.W8 or not x.rsc or x.grp != cfg.WG or x.kt or x.k0 or x.scale is None
               or x.shape != wh.shape or x.rs != wh.rs or x.srs != wh.srs for x in (wh, wl)):
            return 0
        if wh.shape[1] != cfg.D or wh.srs >= 1 << I.MM_HLD:
            return 0
        for st in (ah,) if al is None else (ah, al):
            if st.KB != 1 or st.pair or st.ranges() != [(0, 1)]:
                return 0
        if al is not None and (al.src.rows != ah.src.rows
                               or list(al.chunks) != [(ab + 1, mc) for ab, mc in ah.chunks]):
            return 0
        return hld

    def dot(self, a, w: QTensor, acc: Tile | None = None, out: Tile | None = None,
            temp: bool = False, rowmax: bool = False, acc_scale: Tile | None = None,
            spl: tuple | None = None) -> Tile:
        """(spl: dot_split's one MM, (lo's stationary or None, HLD).)"""
        if not isinstance(w, QTensor):
            raise CompileError("dot(a, w): w must be a streamed QTensor descriptor")
        if acc is not None and out is not None:
            raise CompileError("dot: give either acc= (accumulate) or out= (overwrite)")
        if acc_scale is not None:
            if acc is None or w.scale is not None:
                raise CompileError("dot(acc_scale=...) needs acc= and an unscaled stream (V^T)")
            self.check_live(acc_scale)
        st = a if isinstance(a, Stationary) else self.quantize(a, temp)
        sts = (st,) if spl is None or spl[0] is None else (st, spl[0])
        if not all(self.stationary_valid(x) for x in sts):
            raise CompileError("stationary operand was overwritten in ACT RAM")
        # an operand from outside the innermost loop must survive every iteration: keep it (and
        # its blocks) until the loop ends; one created in the same body is rebuilt per iteration
        for x in sts:
            if self.loops and getattr(x, "loop", None) is not self.loops[-1].loop:
                self.loop_uses.append((len(self.loops), x))
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
            if len(st.chunks) != 1 or M > self.cfg.MCOLS:
                raise CompileError("dot(acc_scale=...) needs M <= MCOLS")
        if rowmax and (len(st.chunks) != 1 or M > self.cfg.MCOLS or len(out.shape) != 2
                       or getattr(out, "spare", 0) < M or out.rs < N):
            raise CompileError("dot(rowmax=True) needs M <= MCOLS and an output tile from "
                               "ol.empty/zeros/full (they reserve the row-max area)")
        g = w.grp
        wg = self.cfg.WG if w.wf == I.W8 or self.cfg.WG == 1 else 2 * self.cfg.WG
        if g > 1 and g != wg:
            raise CompileError(f"dot: a matrix in groups of {g} rows on a machine of WG "
                               f"{self.cfg.WG} (its groups of this format: {wg} rows)")
        if g > 1 and (rowmax or acc_scale is not None) and not self.cfg.GALL:
            raise CompileError("dot: rowmax / acc_scale need a matrix of rows (not grouped)")
        unit = w.scale is None
        # PAIR (rows of one: a group takes a K-block a step at any M) reads a chunk's two
        # blocks' scales at once: 8-byte (nvfp4: 16-byte) aligned rows and an even first block
        al = 16 if w.wf == I.WNV else 8
        pair = st.pair and g == 1 and w.wf != I.W8 and (unit or w.rsc or (
            w.k0 % 2 == 0 and w.srs % al == 0 and Affine.of(w.scale).const % al == 0
            and all(c * getattr(v, "align", 1) % al == 0
                    for v, c in Affine.of(w.scale).terms.items())))
        # one MM per QACT range of the operand (an MM's rows read one Sa: docs/isa.md "MM"),
        # added up in the output: the first as asked (overwrite, ACC or ASCALE), the rest ACC;
        # RMAX on the last, whose written values are the sums
        bpb = self.cfg.D if w.wf == I.W8 else self.cfg.D // 2   # streamed bytes per block
        parts = st.ranges()
        if w.kt and (not self.cfg.KT or g == 1 or w.k0 or
                     [tuple(r) for r in parts] != [(0, st.KB)]):
            raise CompileError(f"dot: a matrix in KT's pairs (docs/isa.md \"K tail\") on a "
                               f"machine with KT, grouped, its whole K one MM (the operand one "
                               f"QACT range: {parts}, k0 {w.k0}, KT {self.cfg.KT}, groups {g})")
        for i, (k0, k1) in enumerate(parts):
            if g == 1 and (k0 * bpb) % self.cfg.D:
                raise CompileError(f"dot: a QACT range starting at block {k0} of 4-bit weights "
                                   f"(their D-byte chunks hold two blocks): split the operand "
                                   f"at even blocks")
            first, last = i == 0, i == len(parts) - 1
            kp = 0 if unit or w.rsc else w.k0 + k0
            ppair = pair and kp % 2 == 0
            m0 = 0
            for ab, mc in st.chunks:
                # block k0's chunk: k0 bpb / D chunks in (a group's: k0 chunks of g bpb bytes)
                ins = I.mm(sa + k0 * bpb * g, ssa, out.base + m0 * ors, N,
                           k1 - k0 if spl is None else 2 + (spl[0] is not None), w.rs, ors,
                           mc, ab + k0, w.srs, unit=unit, acc=acc is not None or not first,
                           rmax=rowmax and last, wf=w.wf, pair=ppair, k0=kp, rsc=w.rsc, grp=g,
                           kt=w.kt, hld=0 if spl is None else spl[1],
                           ascale=acc_scale.base if acc_scale is not None and first else None,
                           ra=ra, rb=rb,
                           comment=f"mm {M}x{K} . {N}x{K}^T"
                           + (f" part {k0}:{k1}" if len(parts) > 1 else "")
                           + (" kt" if w.kt else "")
                           + ("" if spl is None else " split" if spl[0] is not None else " hi + lo")
                           + (" pair" if ppair else "") + (" +rowmax" if rowmax and last else "")
                           + (" acc*=scale" if acc_scale is not None and first else ""))
                self.emit(ins)
                m0 += mc
        self.bump_version(out.buf)
        if (acc is None and spl is None and len(st.chunks) == 1 and M <= self.cfg.MCOLS
                and len(out.shape) == 2 and getattr(out, "spare", 0) >= M and out.rs >= N):
            out.mm_src = (ins, self.versions.get(out.buf, 0), M, ors)
        return out

    def store_quantized(self, x: Tile, dst: Affine, sdst: Affine, drs: int, es: int,
                        row_scale: bool, half: bool = False, row=None, col: bool = False,
                        lo: bool = False, hld: int = 0) -> None:
        """QST of x's rows (row stride drs, element stride es); `row` (GALL): into a grouped
        matrix at dst (QST GRP), its rows row .. (col: its columns, es its group stride; else es
        the scale areas' group stride), `row` an index expression (R[rd] + drs). lo: the
        split's lo bytes and scale (QST flag LO, Config.SPLITQ). hld > 0 (QST HL, Config.QSTHL;
        GRP ROW rows): the hi part and, from the same read, the lo part hld bytes (a multiple of
        1 KiB and of the group chunk P = WG x D) after it, its bytes and scales."""
        if lo and not self.cfg.SPLITQ:
            raise CompileError("store_quantized(lo=True): the machine has no SPLITQ (QST flag LO)")
        if hld and not (self.cfg.QSTHL and self.cfg.SPLITQ and row is not None and row_scale
                        and not (col or half or lo) and x.cols <= I.QHL_ELEMS
                        and hld % 1024 == 0 and hld % (self.cfg.WG * self.cfg.D) == 0):
            raise CompileError(f"store_quantized(hld={hld}): QST HL needs Config.QSTHL, GRP ROW "
                               f"rows (not COL, HALF or LO) of at most {I.QHL_ELEMS} elements "
                               f"and lo's part a multiple of 1 KiB and of P after hi's")
        self.check_live(x)
        D = self.cfg.D
        if x.cols % D:
            raise CompileError("quantized stores need a multiple of D elements per row")
        rb, dimm = self.addr(dst)
        rc, simm = self.addr(sdst)
        rd = 0
        if row is not None:
            rd, drs = self.addr(row)
        rs = x.rs if len(x.shape) == 2 else x.cols
        if not row_scale and x.cols // D > QST_KB_MAX:
            raise CompileError(f"QST of {x.cols // D} blocks without ROW: at most {QST_KB_MAX}")
        self.emit(I.qst(x.base, dimm, simm, x.rows, x.cols // D, rs, drs, es, row=row_scale,
                        half=half, rb=rb, rc=rc, grp=self.cfg.WG if row is not None else 1,
                        col=col, rd=rd, lo=lo, hld=hld,
                        comment="quantized store" + (" lo" if lo else " hi + lo" if hld else "")
                        + (" (grouped)" if row is not None else "")))

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
        out.fresh = True                # dst.set(all_gather(x)): retarget
        return out

    def all_gather_rows(self, x: Tile, S: int, out: Tile | None = None) -> Tile:
        """language.all_gather_rows: one GATHER, slice s's rows at rows s R .. of the result."""
        x = self.materialize(x)
        self.check_live(x)
        R, C = x.rows, x.cols
        fresh = out is None             # (a given destination stays where it is: no retarget)
        if fresh:
            out = self.alloc((S * R, C))
        elif out.shape != (S * R, C):
            raise CompileError(f"all_gather_rows into {out.shape}: {S * R} rows of {C}")
        else:
            self.check_live(out)
        self.emit(I.gather(x.base, out.base, R, C, x.rs, out.rs, R * out.rs,
                           comment="all-gather rows"))
        out.fresh = fresh
        return out

    # ---- finalize
    def finish(self) -> list[I.Instr]:
        if self.loops:
            raise CompileError("unterminated loop")
        self.emitted += 1
        self._drain()
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
            body += [I.addi(r, r, self._wrap(c), comment=f"{loop} step", aw=self.aw)
                     for r, c in it.steps]
            if not body or self._ends_inner(it.items, it.steps) or self._ends_loop(body):
                body.append(I.nop("loop end"))
            out.append(I.loop(len(body), loop.count, rcount=loop.rcount,
                              comment=f"{loop} x" + (f"R{loop.rcount}" if loop.rcount
                                                      else f"{loop.count}")))
            out += body
            out += [I.addi(r, r, self._wrap(-c * loop.count), comment=f"{loop} reset", aw=self.aw)
                    for r, c in it.steps]
        return out

    def _wrap(self, v: int) -> int:
        """An increment as ADDI takes it (isa.wrap: mod 2^AW, registers are AW bits), so a
        reset of a stride times a count of 4 GiB or more encodes."""
        return I.wrap(v, self.aw)

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
    row stride, which spreads the rows over the banks. A grouped matrix's MM (docs/isa.md
    "Weight groups") drains a group's rows, consecutive words, once per activation row, against
    the group's KB advances (a chunk a K-block: 2 WG rows of 4-bit, WG of int8)."""
    lanes = lanes or cfg.MCOLS
    out = []
    for pc, ins in enumerate(prog):
        if ins.op != I.MM:
            continue
        ors, M, KB = ins.w[5] & 0xFFFF, (ins.w[5] >> 16) & 0xFF, (ins.w[3] >> 16) & 0xFF
        adv = -(-KB // 2) if ins.flags & I.F_PAIR and not ins.rd else KB
        if ins.rd:
            M, ors = 1 << ins.rd, 1
        steps, j, rows = 0, 0, M if ins.rd else min(M, cfg.MCOLS)
        while j < rows:
            banks = set()
            while j < rows and len(banks) < lanes and (j * ors) % cfg.LANES not in banks:
                banks.add((j * ors) % cfg.LANES)
                j += 1
            steps += 1
        if steps > adv:
            out.append((pc, ins.comment))
    return out
