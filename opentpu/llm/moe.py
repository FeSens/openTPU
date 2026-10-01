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
from ..host.offload import LINE, SPLIT, ExpertServer, Layout, PoolFile, dram_of, to_split
from ..kernels.lib import gelu_tanh, rmsnorm, sigmoid, silu
from ..kernels.mlp import _chunk, swiglu_down

NEG = -3.0e38                       # knocked out (below every score)
ACTS = {"silu": silu, "gelu_tanh": gelu_tanh}


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
    act: str = "silu"                # the experts' gate: SiLU, or "gelu_tanh" (Gemma 4)

    def __post_init__(self):
        if self.rule not in ("sigmoid_bias", "softmax"):
            raise ValueError(f"MoE router rule {self.rule!r}")
        if self.act not in ACTS:
            raise ValueError(f"MoE expert activation {self.act!r}")
        if self.rule == "softmax" and not self.norm:
            raise ValueError("a softmax router over all experts is not supported (norm=False)")


class ExpertFormat:
    """One expert in its slot: W_gate, W_up [F, H] (rows, then their scale words) and W_down
    [H, F] in column parts of `C` (swiglu_down's chunks), each D-byte aligned. A width that is
    not a whole number of the format's chunks (Gemma 4's 704) is padded with zero rows of
    W_gate and W_up and zero columns of W_down to `F` (768): exact, act(0) * 0 = 0."""

    def __init__(self, H: int, F: int, D: int, wformat: str):
        rb = lambda k: Q.row_bytes(k, wformat, D)                       # noqa: E731
        q = D if wformat == "int8" else 2 * D
        self.F0, F = F, -(-F // q) * q       # the expert's own width, the slot's
        self.H, self.F, self.D, self.wformat = H, F, D, wformat
        self.C = _chunk(F, D, q)
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
        """The slot's bytes for W_gate, W_up [F0, H] and W_down [H, F0] (float)."""
        out = np.zeros(self.nbytes, np.uint8)
        if self.F != self.F0:                   # the zero padding
            pad = self.F - self.F0
            wg, wu = (np.pad(np.asarray(w, np.float32), ((0, pad), (0, 0))) for w in (wg, wu))
            wd = np.pad(np.asarray(wd, np.float32), ((0, 0), (0, pad)))

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


def gemma_expert(W, p: str, e: int):
    """Gemma 4's routed expert e of the layer at prefix p (`p + "experts.gate_up_proj"` [E, 2F,
    H], gate rows first; `p + "experts.down_proj"` [E, H, F]) as W_gate, W_up [F, H] and W_down
    [H, F] for ExpertFormat.pack, router.per_expert_scale[e] folded into W_down (it multiplies
    the expert's output). pre_feedforward_layernorm_2's gain stays out: moe_ffn applies it to
    the experts' input (lw.g_exp; folded into the columns its outliers, up to 8x the gain's
    rms in the 26B, would coarsen every other column's block). A lazy W with `part` reads the
    one expert."""
    k = p + "experts."
    get = getattr(W, "part", None)
    gu, dn = ((get(k + "gate_up_proj", e), get(k + "down_proj", e)) if get is not None else
              (W[k + "gate_up_proj"][e], W[k + "down_proj"][e]))
    gu, dn = np.asarray(gu, np.float32), np.asarray(dn, np.float32)
    F = gu.shape[0] // 2
    s = np.asarray(W[p + "router.per_expert_scale"], np.float32)[e]
    return gu[:F], gu[F:], dn * s


def gemma_router(W, p: str) -> np.ndarray:
    """Gemma 4's router [E, H] on the residual's unit RMSNorm (moe_ffn's input): router.proj
    with router.scale and H^-0.5 folded into its columns."""
    w = np.asarray(W[p + "router.proj.weight"], np.float32)
    return w * (np.asarray(W[p + "router.scale"], np.float32) * np.float32(w.shape[1] ** -0.5))


def open_pool(layout: Layout, pool_file) -> PoolFile:
    """The expert pool file for `layout`, opened, and its packed experts' read into the page
    cache started (PoolFile.warm: the host's RAM tier; the Engine opens it before it builds the
    image, so the read runs during the build). The file is the whole pool's size (sparse until
    packed), and `<pool_file>.packed` marks the experts in it (a file of the right size without
    one is a pool packed whole). A new file is in the split format
    (opentpu.host.offload.split_order), as `<pool_file>.format` says; a file without it holds
    the slot format. The page cache stays the kernel's to reclaim: nothing is pinned."""
    L = layout
    n = L.layers * L.E
    path = Path(pool_file)
    done = Path(str(path) + ".packed")
    fresh = not path.exists() or path.stat().st_size != n * L.slot_bytes
    if fresh:
        with open(path, "wb") as f:
            f.truncate(n * L.slot_bytes)
    if fresh or not done.exists() or done.stat().st_size != n:
        done.write_bytes(bytes(n) if fresh else bytes([1]) * n)
    fmt = Path(str(path) + ".format")
    if fresh:
        fmt.write_text(SPLIT + "\n")
    pf = PoolFile(path, L.slot_bytes, fmt.exists() and fmt.read_text().strip() == SPLIT)
    pf.arr = np.memmap(path, np.uint8, "r+", shape=(n, L.slot_bytes))
    pf.packed = np.memmap(done, np.uint8, "r+", shape=(n,))
    pf.ids = np.nonzero(np.asarray(pf.packed))[0]
    pf.resident_at_open = pf.resident(pf.ids)
    pf.warm(pf.ids)
    return pf


def serve(layout: Layout, expert, backend, pool_file=None, warm=True,
          policy: str = "lfu") -> ExpertServer:
    """The host's expert server on the backend's DRAM (slice 0); with `warm`, the slots filled
    with each layer's first experts. expert(g): global expert g's slot bytes. An expert is
    packed when it is first asked for and kept in host RAM, or, with `pool_file` (a path, or
    open_pool's PoolFile), in that file (the page cache, or the SSD tier), which keeps it for
    later runs (`server.pool`: the PoolFile). `policy`: the slots' replacement (ExpertServer's;
    least decayed use, docs/offload.md 5.5)."""
    L = layout
    pf = None
    if pool_file is None:
        cache: dict = {}

        def pool(g):
            if g not in cache:
                cache[g] = expert(g).tobytes()
            return cache[g]
    else:
        pf = pool_file if isinstance(pool_file, PoolFile) else open_pool(L, pool_file)

        def pool(g):                    # read from the file (the memmap's writes are in the
            if not pf.packed[g]:        # page cache preadv reads)
                pf.arr[g] = to_split(expert(g)) if pf.split else expert(g)
                pf.packed[g] = 1
            return pf.get(g)
    srv = ExpertServer(dram_of(backend, L), L, pool, policy=policy)
    srv.pool_file = pf
    srv.pool_warm = None if pf is None else pf.warm_t
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
def moe_ffn(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float, beside=None,
            residual: bool = True, y_first: bool = False):
    """x + the MoE FFN of one token (module docstring). lw: the layer's g_post [H], router
    QTensor [E, H] ([E + 1, H] with a shared expert: its gate is row E), ebias Tensor [E]
    (sigmoid_bias), gbase Tensor [1] (j * E as fp32), and with a shared expert its SwiGLU
    wg, wu, wd; with g_exp [H] (Gemma 4: pre_feedforward_layernorm_2's gain) the routed
    experts read their own quantized input, the norm times g_exp, and the router the norm
    (a gain with outliers folded into the experts' columns would coarsen every other column's
    block). dev: the offload words (mbox, served, dir: static DRAM addresses) and `fmt`,
    the ExpertFormat.

    beside(): emits work that needs no expert (Gemma 4's dense MLP) right after the request is
    posted, so that it runs while the host streams the missing experts (docs/offload.md 5.3);
    moe_ffn's register is free during it. residual=False: the FFN's output without x (Gemma 4
    norms the experts' sum before its residual). y_first: the experts' [k, H] outputs take
    their TMEM before the router's input does (Gemma 4 26B-A4B: 22,544 words, which only the
    free space before the step's later tiles holds)."""
    b = current()
    if ol.num_programs() != 1:
        raise CompileError("moe_ffn runs on one slice")
    E, k, H = mo.E, mo.k, x.cols
    y = ol.empty((k, H)) if y_first else None
    if getattr(lw, "g_exp", None) is None:
        xs = xe = ol.quantize(rmsnorm(x, ol.load(lw.g_post), eps))
    else:
        xn = rmsnorm(x, ol.load(lw.g_post), eps)
        xs, xe = ol.quantize(xn), ol.quantize(xn * ol.load(lw.g_exp)[None, :])   # (QACT
        del xn                                                                   # CSCALE)
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
    if y is None:
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
    if beside is not None:                              # while the host streams
        b.unscratch(r)
        beside()
        r = b.scratch()
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
        o = swiglu_down(xe, ex.wg, ex.wu, ex.wd, act=ACTS[mo.act])
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
    return x + acc if residual else acc
