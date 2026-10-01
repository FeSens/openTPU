"""Mixture-of-experts FFN with its experts streamed into a DRAM cache: path (a) of
docs/offload.md. The card routes and computes every expert; opentpu.host.offload's server only
moves experts into per-layer LRU slots in the card's DRAM and keeps the directory.

A MoE layer on the device (`moe_ffn`), one token:

1. the router: an MM of the normalized residual -> E logits (and, with a shared expert, its
   gate's logit as row E of the same matrix), and the model's rule (MoESpec.rule);
2. the k best by ARGMAX with knock-out (in order, ties to the first), their global ids
   (the layer's base j * E + index) and weights: LFM2's sigmoid scores, chosen with the
   expert bias, renormalized and scaled; Qwen's softmax of the k largest logits;
3. the fence, WAITW served >= seq (the host has finished every earlier request: none of this
   layer's slots is being replaced, and the one request row is free), then the ids and
   seq + 1 to the mailbox;
4. the directory: each id's present flag;
5. the experts present (a LOOP over the k ids, a LOOP R[present] inside), then the others
   (LOOP R[1 - present]); each takes its slot address from its directory entry by WAITW
   (!= 0: at once for an expert present; for the others once the host has written the entry
   after the expert's DMA) to a TMEM word and RLD (raw) into a register, runs
   kernels.mlp.swiglu_down there and writes its weighted output to its row of a [k, H] tile;
6. the rows summed in the router's order (the result does not depend on what the cache held),
   then a shared expert (a dense SwiGLU of the layer block, times sigmoid of its gate's
   logit), and added to the residual.

It holds one register (the resident decode's run arguments and the layer loops hold the
rest): the per-expert values are the columns of a small tile that each loop rotates, so
expert i is always at column 0 in iteration i.

`ExpertFormat` is one expert's slot: gate and up [F, H] and W_down's column parts, laid out as a
layer block's MLP (lfm2.Image), so the FFN is kernels.mlp.swiglu_down at the slot's address
(a compiler.DevVar: the register RLD sets). One slice (S = 1, the board's).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from .. import quant as Q
from ..compiler import Affine, CompileError, DevVar, QTensor, Tensor, current
from ..host.offload import LINE, ExpertServer, Layout, dram_of
from ..kernels.lib import rmsnorm, sigmoid
from ..kernels.mlp import _chunk, swiglu_down

NEG = -3.0e38                       # knocked out (below every score)


@dataclass(frozen=True)
class MoESpec:
    E: int                 # experts per layer
    k: int                 # experts per token
    ffn: int               # an expert's width
    first: int = 0         # the first MoE layer (the ones before it have a dense MLP)
    rule: str = "sigmoid_bias"       # LFM2: sigmoid scores, the choice on score + expert bias;
                                     # "softmax": Qwen, the k largest logits' softmax
    norm: bool = True                # renormalize the chosen weights (softmax: always)
    scale: float = 1.0               # then multiply them by this
    shared: int = 0                  # a shared expert's width (Qwen: the layer's dense MLP,
                                     # times sigmoid(x . w_gate)), 0: none

    def __post_init__(self):
        if self.rule not in ("sigmoid_bias", "softmax"):
            raise ValueError(f"MoE router rule {self.rule!r}")
        if self.rule == "softmax" and not self.norm:
            raise ValueError("a softmax router over all experts is not supported (norm=False)")


class ExpertFormat:
    """One expert in its slot: W_gate, W_up [F, H] (rows, then their scale words) and W_down
    [H, F] in column parts of `C` (swiglu_down's chunks), each D-byte aligned."""

    def __init__(self, H: int, F: int, D: int, wformat: str):
        rb = lambda k: Q.row_bytes(k, wformat, D)                       # noqa: E731
        self.H, self.F, self.D, self.wformat = H, F, D, wformat
        self.C = _chunk(F, D, D if wformat == "int8" else 2 * D)
        a, align = 0, max(LINE, D)          # the MXU streams whole D-byte chunks

        def take(n):
            nonlocal a
            at, a = a, a + -(-n // align) * align
            return at
        self.wg = (take(F * rb(H)), take(4 * F * (H // D)))
        self.wu = (take(F * rb(H)), take(4 * F * (H // D)))
        self.wd = [(take(H * rb(self.C)), take(4 * H * (self.C // D)))
                   for _ in range(F // self.C)]
        self.nbytes = a

    def pack(self, wg: np.ndarray, wu: np.ndarray, wd: np.ndarray) -> np.ndarray:
        """The slot's bytes for W_gate, W_up [F, H] and W_down [H, F] (float)."""
        out = np.zeros(self.nbytes, np.uint8)

        def put(pair, W):
            q, s = Q.quantize_mxu(W, self.wformat, self.D)
            for at, v in zip(pair, (q, s)):
                v = np.ascontiguousarray(v).view(np.uint8).reshape(-1)
                out[at:at + v.size] = v
        put(self.wg, wg)
        put(self.wu, wu)
        C = self.C
        for j, pair in enumerate(self.wd):
            put(pair, wd[:, j * C:(j + 1) * C])
        return out

    def descs(self, base) -> SimpleNamespace:
        """wg, wu, wd QTensors of the expert whose slot starts at `base` (an address
        expression: a DevVar for a slot found at run time)."""
        H, F, D, C, fm = self.H, self.F, self.D, self.C, self.wformat
        wf, base = Q.mxu_wf(fm), Affine.of(base)

        def q(pair, n, k):
            return QTensor(base + pair[0], base + pair[1], (n, k), Q.row_bytes(k, fm, D),
                           4 * (k // D), D, wf=wf)
        parts = tuple(q(p, H, C) for p in self.wd)
        wd = QTensor(parts[0].data, parts[0].scale, (H, F), Q.row_bytes(C, fm, D),
                     4 * (C // D), D, parts=parts, pw=C, wf=wf)
        return SimpleNamespace(wg=q(self.wg, F, H), wu=q(self.wu, F, H), wd=wd)


def _read_ahead(path) -> None:
    """Ask the kernel to read the pool file into the page cache in the background (Linux), so a
    miss's staging copy reads RAM, not the disk. It moves bytes only: the host's RAM tier."""
    import os
    if hasattr(os, "posix_fadvise"):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
        finally:
            os.close(fd)


def serve(layout: Layout, expert, backend, pool_file=None, warm=True) -> ExpertServer:
    """The host's expert server on the backend's DRAM (slice 0); with `warm`, the slots filled
    with each layer's first experts. expert(g): global expert g's slot bytes. An expert is
    packed when it is first asked for and kept in host RAM, or, with `pool_file`, in that file
    (the page cache, or the SSD tier), which keeps it for later runs: the file is the whole
    pool's size (sparse until packed), and `<pool_file>.packed` marks the experts in it (a
    file of the right size without one is a pool packed whole)."""
    L = layout
    n = L.layers * L.E
    if pool_file is None:
        cache: dict = {}

        def pool(g):
            if g not in cache:
                cache[g] = expert(g).tobytes()
            return cache[g]
    else:
        path = Path(pool_file)
        done = Path(str(path) + ".packed")
        fresh = not path.exists() or path.stat().st_size != n * L.slot_bytes
        if fresh:
            with open(path, "wb") as f:
                f.truncate(n * L.slot_bytes)
        if fresh or not done.exists() or done.stat().st_size != n:
            done.write_bytes(bytes(n) if fresh else bytes([1]) * n)
        arr = np.memmap(path, np.uint8, "r+", shape=(n, L.slot_bytes))
        packed = np.memmap(done, np.uint8, "r+", shape=(n,))
        _read_ahead(path)

        def pool(g):                    # the expert's pages (no copy: the DMA's staging
            if not packed[g]:           # reads them)
                arr[g] = expert(g)
                packed[g] = 1
            return arr[g]
    srv = ExpertServer(dram_of(backend, L), L, pool)
    srv.load([j * L.E + e for j in range(L.layers) for e in range(L.E)] if warm else ())
    return srv


# ---------------------------------------------------------------------------- reference
def route(logits: np.ndarray, bias: np.ndarray | None, mo: MoESpec):
    """The model's choice for one token's router logits [E]: (ids in the device's order, their
    weights). The k largest (ties: the first) of: LFM2, sigmoid scores + bias, the chosen
    scores renormalized (+ 1e-6, as Hugging Face) and scaled; softmax (Qwen), the logits, the
    chosen ones' softmax (Hugging Face's softmax over all, renormalized over the k)."""
    lg = np.asarray(logits, np.float64)
    if mo.rule == "softmax":
        s, sel = lg, lg.copy()
    else:
        s = 1.0 / (1.0 + np.exp(-lg))
        sel = s + bias
    ids = []
    for _ in range(mo.k):
        i = int(np.argmax(sel))
        ids.append(i)
        sel[i] = -np.inf
    w = s[ids]
    if mo.rule == "softmax":
        w = np.exp(w - w[0])
        w = w / w.sum()
    elif mo.norm:
        w = w / (w.sum() + 1e-6)
    return ids, w * mo.scale


# ---------------------------------------------------------------------------- the device
def moe_ffn(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float):
    """x + the MoE FFN of one token (module docstring). lw: the layer's g_post [H], router
    QTensor [E, H] ([E + 1, H] with a shared expert: its gate is row E), ebias Tensor [E]
    (sigmoid_bias), gbase Tensor [1] (j * E as fp32), and with a shared expert its SwiGLU
    wg, wu, wd. dev: the offload words (mbox, served, dir: static DRAM addresses) and `fmt`,
    the ExpertFormat."""
    b = current()
    if ol.num_programs() != 1:
        raise CompileError("moe_ffn runs on one slice")
    E, k, H = mo.E, mo.k, x.cols
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_post), eps))
    lg = ol.dot(xs, lw.router)
    sc, sel = ol.empty((E,)), ol.empty((E,))
    if mo.rule == "softmax":
        sc.set(lg[0, 0:E])
        sel.set(sc)
    else:
        sc.set(sigmoid(lg[0, 0:E]))
        sel.set(sc + ol.load(lw.ebias))
    gate = sigmoid(lg[0, E:E + 1]) if mo.shared else None
    # One register in all (the resident decode's arguments and the layer loops hold the
    # rest): r holds one value at a time (an index, the seq bits, an offset, a count, a slot
    # address, a row). The k experts' values are the columns of pe, a [5, k] tile that every
    # loop over the experts rotates left by one at the end of its body: expert i is at column
    # 0 in iteration i, and after k iterations every row is back in order. (A slot address
    # never passes through the VPU, which would flush it as a denormal: WAITW copies it from
    # the directory to the word `word`, at once for an expert present, and RLD raw takes its
    # bits into r.)
    EP, MISS, OFF, WT, ROW = range(5)               # pe's rows
    pe, tmp, ids, pr = ol.empty((5, k)), ol.empty((5, k)), ol.empty((k,)), ol.empty((2,))
    word = ol.empty((1,))
    wt = pe[WT, :]

    def rotate(rows=slice(0, 5)):
        a, e = rows.start, rows.stop
        if k > 1:
            tmp[a:e, 0:k - 1].set(pe[a:e, 1:k])
        tmp[a:e, k - 1:k].set(pe[a:e, 0:1])
        pe[a:e, :].set(tmp[a:e, :])

    r = b.scratch()
    # the k best: ARGMAX, knock-out; in the router's order, the new one to column k - 1
    lp = b.begin_loop(k)
    b.emit(I.argmax(pr.base, sel.base, 1, E, comment="moe: best"))
    b.emit(I.rld(r, pr.base + 1, comment="its index"))
    b.emit(I.vop(I.V_FILL, sel.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, NEG, ra=r,
                 comment="knock out"))
    rotate(slice(WT, WT + 1))
    b.emit(I.vop(I.V_COPY, wt.base + k - 1, sc.base, 0, 1, 1, 0, 0, 0, rb=r, comment="weight"))
    if k > 1:
        tmp[0, 0:k - 1].set(ids[1:k])
        ids[0:k - 1].set(tmp[0, 0:k - 1])
    b.emit(I.vop(I.V_COPY, ids.base + k - 1, pr.base + 1, 0, 1, 1, 0, 0, 0, comment="id"))
    b.end_loop(lp)
    if mo.rule == "softmax":                            # column 0 holds the largest
        e = ol.exp2((wt - wt[0:1]) * ol.LOG2E)
        wt.set(e * ol.recip(ol.sum(e)))
    elif mo.norm:
        wt.set(wt * ol.recip(ol.sum(wt) + 1e-6))
    if mo.scale != 1.0:
        wt.set(wt * float(mo.scale))
    gid = ol.empty((k,))
    gid.set(ids + ol.load(lw.gbase))                    # global ids: j * E + index
    pe[OFF, :].set(gid * 8.0)                           # their directory entries' offsets
    y = ol.empty((k, H))
    for i in range(k):                                  # y's rows (ol.empty may pad them)
        pe[ROW, i:i + 1].set(float(i * y.rs))
    # the fence, then the request: seq + 1 and the ids
    seq = ol.load(Tensor(Affine(dev.mbox), (1,), (1,)))
    b.rld(r, seq, raw=True, comment="seq (bits)")
    b.waitw(word, dev.served, 0, I.C_GE, rc=r, comment="fence: served >= seq")
    ol.store(Tensor(Affine(dev.mbox + LINE), (k,), (1,)), gid)
    seq.set(seq + 1.0)
    ol.store(Tensor(Affine(dev.mbox), (1,), (1,)), seq)
    # the directory: each entry's present flag
    col = lambda row: pe.base + row * pe.rs                       # noqa: E731 (column 0)
    lp = b.begin_loop(k)
    b.emit(I.rld(r, col(OFF), comment="entry offset"))
    b.emit(I.ld(dev.dir + 4, col(EP), 1, ra=r, comment="entry: present"))
    rotate()
    b.end_loop(lp)
    pe[MISS, :].set(pe[EP, :] * -1.0 + 1.0)
    ex = dev.fmt.descs(DevVar("expert slot", r))

    def expert():
        """The expert whose slot is R[r], weighted, into its row of y."""
        o = swiglu_down(xs, ex.wg, ex.wu, ex.wd)
        b.check_live(o)
        b.emit(I.rld(r, col(ROW), comment="its row"))
        b.emit(I.vop(I.V_MUL, y.base, o.base, col(WT), 1, H, 0, 0, 0, I.B_ROW, ra=r,
                     comment="weighted, to its row"))

    for flag, wait in ((EP, False), (MISS, True)):
        lp = b.begin_loop(k)
        b.emit(I.rld(r, col(flag), comment="missing" if wait else "present"))
        inner = b.begin_loop(0, rcount=r)               # (the count is read here: r is free)
        b.emit(I.rld(r, col(OFF), comment="entry offset"))
        b.waitw(word, dev.dir, 0, I.C_NE, ra=r, comment="its slot" if not wait else
                "wait: its slot")
        b.rld(r, word, raw=True, comment="its slot")
        expert()
        b.end_loop(inner)
        rotate()
        b.end_loop(lp)
    b.unscratch(r)
    acc = y[0:1, :]
    for i in range(1, k):
        acc = acc + y[i:i + 1, :]
    if mo.shared:
        acc = acc + swiglu_down(xs, lw.wg, lw.wu, lw.wd) * gate[:, None]
    return x + acc
