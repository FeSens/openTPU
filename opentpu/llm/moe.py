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
   layer's slots is being replaced, and the one request row and answer are free), then the
   ids and seq + 1 to the mailbox;
4. the directory: each id's present flag;
5. the experts present (a LOOP over the k ids, a LOOP R[present] inside), then the others
   (LOOP R[1 - present]). A present one takes its slot address from its directory entry, a
   missing one from its word of the request's answer (docs/offload.md 10.11), each by WAITW
   (!= 0: at once for an expert present; for the others once the host has written the
   answer) to a TMEM word and RLD (raw) into a register; a missing one then waits for its
   slot's tag word (WAITW != 0: the host writes it in its expert's DMA's last beat). Each
   zeroes its slot's tag (the next nonzero tag there is a later expert's), runs
   kernels.mlp.swiglu_down there and writes its weighted output to its row of a [k, H] tile;
   then the answer's words are zeroed (the host writes the next request's answer only after
   its post);
6. the rows summed in the router's order (the result does not depend on what the cache held),
   then a shared expert (a dense SwiGLU of the layer block, times sigmoid of its gate's
   logit), and added to the residual.

It holds one register (the resident decode's run arguments and the layer loops hold the
rest): the per-expert values are the columns of a small tile that each loop rotates, so
expert i is always at column 0 in iteration i.

With MoESpec.hint the layer also posts a prefetch hint before its mixer (`moe_hint`,
docs/offload.md section 12): its router on the layer's input, the k best as a request whose
ids are offset by layers x E; the host replaces their slots' victims at once and moves the
missing experts in the link's idle time, and the route's own request finds them landed or on
their way.

`moe_ffn_rows` is the layer on R rows at once (layer-major prefill, docs/offload.md section 13):
each row routes as moe_ffn's; one request carries the rows' R x k ids (the count at mbox + 4);
the card finds their union (each id's first place among them) and runs each union expert once
on all R rows, storing each row's unweighted output to its (row, rank) place in a DRAM scratch
of [R k + 1, H] (R k <= 16, the request's one line; the last row a sink for the outputs no row
chose); each row then sums its own k in its router's order, so a row's result is moe_ffn's bit
for bit. `moe_hint_rows` is moe_hint on R rows: a layer-major run's hint for the next layer, its
router on the run's output rows (section 13.9).

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
from ..kernels import mailbox as MB
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
    hint: bool = False               # each layer posts its router's k best on its input as a
                                     # prefetch hint before its mixer (moe_hint; Qwen3.5)

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
    the experts' input before quantizing it (lw.g_exp; docs/offload.md 11.2). A lazy W with
    `part` reads the one expert."""
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


def open_pool(layout: Layout, pool_file, mapped: bool = True) -> PoolFile:
    """The expert pool file for `layout`, opened, and its packed experts' read into the page
    cache started (PoolFile.warm: the host's RAM tier; the Engine opens it before it builds the
    image, so the read runs during the build). The file is the whole pool's size (sparse until
    packed), and `<pool_file>.packed` marks the experts in it (a file of the right size without
    one is a pool packed whole). A new file is in the split format
    (opentpu.host.offload.split_order), as `<pool_file>.format` says; a file without it holds
    the slot format. The page cache stays the kernel's to reclaim: nothing is pinned. mapped:
    PoolFile's (its reads touched through a read-only map: docs/offload.md 10.7)."""
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
    pf = PoolFile(path, L.slot_bytes, fmt.exists() and fmt.read_text().strip() == SPLIT,
                  mapped=mapped)
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
def moe_hint(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float) -> None:
    """Path (a)'s prefetch hint (docs/offload.md section 12), before the layer's mixer: its
    router on the layer's input x through moe_ffn's norm, the k best by the model's rule (no
    weights), posted as a request whose global ids are offset by dev.hint_off (layers x E: the
    host's mark of a hint). The host writes served once it has replaced the hinted experts'
    victims, before their transfers, so the route's fence (moe_ffn) waits for no expert; an
    expert still on its way reads as missing there and is waited for by its answer word and
    its tag. lw and dev as moe_ffn's. One register."""
    b = current()
    E, k = mo.E, mo.k
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_post), eps))
    lg = ol.dot(xs, lw.router)
    del xs
    sel, pr, ids, tmp = ol.empty((E,)), ol.empty((2,)), ol.empty((k,)), ol.empty((k,))
    if mo.rule == "softmax":
        sel.set(lg[0, 0:E])
    else:
        sel.set(sigmoid(lg[0, 0:E]) + ol.load(lw.ebias))
    del lg
    r = b.scratch()
    lp = b.begin_loop(k)                # the k best, as the route picks them
    b.emit(I.argmax(pr.base, sel.base, 1, E, comment="hint: best"))
    b.emit(I.rld(r, pr.base + 1, comment="its index"))
    b.emit(I.vop(I.V_FILL, sel.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, NEG, ra=r,
                 comment="knock out"))
    if k > 1:
        tmp[0:k - 1].set(ids[1:k])
        ids[0:k - 1].set(tmp[0:k - 1])
    b.emit(I.vop(I.V_COPY, ids.base + k - 1, pr.base + 1, 0, 1, 1, 0, 0, 0, comment="id"))
    b.end_loop(lp)
    b.unscratch(r)
    ids.set(ids + ol.load(lw.gbase) + float(dev.hint_off))
    seq = MB.wait_served(dev.mbox)                  # MB.post, with the request's count
    ol.store(Tensor(Affine(dev.mbox + LINE), (k,), (1,)), ids)
    ol.store(Tensor(Affine(dev.mbox + 4), (1,), (1,)), ol.full((1,), float(k)))
    seq.set(seq + 1.0)
    ol.store(Tensor(Affine(dev.mbox), (1,), (1,)), seq)


def moe_hint_rows(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float) -> None:
    """moe_hint on R rows at once (x [R, H], R * k <= LINE / 4: one request row): each row's k
    best by the router of lw's layer, through its norm (moe_ffn's route), posted as one hint
    of the rows' R * k ids (repeats included) and their count. A layer-major prefill run's
    hint for the next layer (docs/offload.md 13.9): layer j's run on its output rows, with
    layer j + 1's lw, ahead of layer j + 1's runs. One register."""
    b = current()
    R, E, k = x.rows, mo.E, mo.k
    N = R * k
    if N > LINE // 4:
        raise CompileError(f"{R} rows of {k} experts: more ids than a request row holds")
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_post), eps))
    lg = ol.dot(xs, lw.router)                          # [R, E] (+ the shared expert's gate)
    del xs
    gid = ol.empty((R, k), dense=True)
    pr, tmp, ids = ol.empty((2,)), ol.empty((k,)), ol.empty((k,))
    off = ol.load(lw.gbase) + float(dev.hint_off)
    r = b.scratch()
    for q in range(R):                                  # each row's k best, as the route's
        sel = ol.empty((E,))
        if mo.rule == "softmax":
            sel.set(lg[q, 0:E])
        else:
            sel.set(sigmoid(lg[q, 0:E]) + ol.load(lw.ebias))
        lp = b.begin_loop(k)
        b.emit(I.argmax(pr.base, sel.base, 1, E, comment="hint: best"))
        b.emit(I.rld(r, pr.base + 1, comment="its index"))
        b.emit(I.vop(I.V_FILL, sel.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, NEG, ra=r,
                     comment="knock out"))
        if k > 1:
            tmp[0:k - 1].set(ids[1:k])
            ids[0:k - 1].set(tmp[0:k - 1])
        b.emit(I.vop(I.V_COPY, ids.base + k - 1, pr.base + 1, 0, 1, 1, 0, 0, 0, comment="id"))
        b.end_loop(lp)
        del sel
        gid[q, :].set(ids + off)
    b.unscratch(r)
    del lg, ids, tmp, off
    seq = MB.wait_served(dev.mbox)                  # MB.post, with the request's count
    ol.store(Tensor(Affine(dev.mbox + LINE), (N,), (1,)), gid.reshape(1, N)[0, :])
    ol.store(Tensor(Affine(dev.mbox + 4), (1,), (1,)), ol.full((1,), float(N)))
    seq.set(seq + 1.0)
    ol.store(Tensor(Affine(dev.mbox), (1,), (1,)), seq)


def slot_of(r: int, word, off: int, dev: SimpleNamespace, missing: bool) -> None:
    """R[r] = the slot of the expert whose offset (TMEM word `off`: its directory entry's, or
    for a missing one its answer word's) the loop has at column 0, its data landed, and the
    slot's tag zeroed. Present: WAITW on its entry (at once). Missing: WAITW on its answer
    word (once the host has written the answer), then on the slot's tag (once the expert's
    DMA has landed: the host writes the tag in its last beat). The tag is zeroed for every
    expert the card uses, so a nonzero tag is always a later expert's (docs/offload.md 10.11).
    word: a one-word TMEM tile."""
    b = current()
    b.emit(I.rld(r, off, comment="answer word" if missing else "entry offset"))
    b.waitw(word, dev.answer if missing else dev.dir, 0, I.C_NE, ra=r,
            comment="wait: its slot (the answer)" if missing else "its slot")
    b.rld(r, word, raw=True, comment="its slot")
    if missing:
        b.waitw(word, dev.tag, 0, I.C_NE, ra=r, comment="wait: its data (the slot's tag)")
    ol.store(Tensor(Affine(dev.tag) + DevVar("expert slot", r), (1,), (1,)), ol.zeros((1,)))


def moe_ffn(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float, beside=None,
            residual: bool = True, y_first: bool = False):
    """x + the MoE FFN of one token (module docstring). lw: the layer's g_post [H], router
    QTensor [E, H] ([E + 1, H] with a shared expert: its gate is row E), ebias Tensor [E]
    (sigmoid_bias), gbase Tensor [1] (j * E as fp32), and with a shared expert its SwiGLU
    wg, wu, wd; with g_exp [H] (Gemma 4: pre_feedforward_layernorm_2's gain) the routed
    experts read their own quantized input, the norm times g_exp, and the router the norm (the
    norm alone, quantized, has its blocks scaled by the residual's outlier channels, which the
    gain zeroes: docs/offload.md 11.2). dev: the offload words (mbox, served, answer, dir:
    static DRAM addresses), `tag` (a slot's tag word, from the slot) and `fmt`, the
    ExpertFormat.

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
    EP, MISS, OFF, WT, ROW, ANS = range(6)          # pe's rows
    pe, tmp, ids, pr = ol.empty((6, k)), ol.empty((6, k)), ol.empty((k,)), ol.empty((2,))
    word = ol.empty((1,))
    wt = pe[WT, :]

    def rotate(rows=slice(0, 6)):
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
    for i in range(k):              # y's rows (ol.empty may pad them), the answer's words
        pe[ROW, i:i + 1].set(float(i * y.rs))
        pe[ANS, i:i + 1].set(float(4 * i))
    # the fence, then the request: seq + 1 and the ids
    seq = ol.load(Tensor(Affine(dev.mbox), (1,), (1,)))
    b.rld(r, seq, raw=True, comment="seq (bits)")
    b.waitw(word, dev.served, 0, I.C_GE, rc=r, comment="fence: served >= seq")
    ol.store(Tensor(Affine(dev.mbox + LINE), (k,), (1,)), gid)
    ol.store(Tensor(Affine(dev.mbox + 4), (1,), (1,)), ol.full((1,), float(k)))  # (its count)
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
    ex = dev.fmt.descs(DevVar("expert slot", r, align=LINE))   # (slots: LINE-aligned)

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
        slot_of(r, word, col(ANS) if wait else col(OFF), dev, wait)
        expert()
        b.end_loop(inner)
        rotate()
        b.end_loop(lp)
    b.unscratch(r)
    ol.store(Tensor(Affine(dev.answer), (k,), (1,)), ol.zeros((k,)))   # (for the next request)
    acc = y[0:1, :]
    for i in range(1, k):
        acc = acc + y[i:i + 1, :]
    if mo.shared:
        acc = acc + swiglu_down(xs, lw.wg, lw.wu, lw.wd) * gate[:, None]
    return x + acc if residual else acc


def moe_ffn_rows(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float, beside=None,
                 residual: bool = True):
    """x + the MoE FFN of R token rows at once (x [R, H], R * k <= LINE / 4: one request row):
    a layer-major prefill run's (docs/offload.md 13). Every row routes as moe_ffn's; one
    request names the rows' R * k ids (repeats included: the host serves each once) and their
    count (the word after seq); each expert of the rows' union runs once, on every row (one MM
    of R rows: with PAIR, R <= 2 cost what one does), and each row that chose it stores its
    output, unweighted, to its place (row, rank) of dev.scratch [R * k + 1, H] (the last row a
    sink for the rows that did not); then each row sums its experts' outputs, weighted, in its
    router's order. So a row's result is moe_ffn's on that row alone, bit for bit, whatever
    the cache held. lw, dev (and dev.scratch), beside and residual as moe_ffn's.

    The union: for each row q, eq[n, j] = 1 - min(1, |id_n - id_q,j|) over every entry n of
    the request (the ids are integers: 1 where equal); its row sums say whether row q chose
    entry n's expert (found) and at which rank (the sum of j eq[n, j]), which gives the byte
    offset of entry n's output for row q in the scratch; an entry whose expert an earlier row
    chose is not computed again. One register, as moe_ffn's."""
    b = current()
    if ol.num_programs() != 1:
        raise CompileError("moe_ffn_rows runs on one slice")
    R, H, E, k = x.rows, x.cols, mo.E, mo.k
    N = R * k
    if N > LINE // 4:
        raise CompileError(f"{R} rows of {k} experts: more ids than a request row holds")
    if getattr(lw, "g_exp", None) is None:
        xs = xe = ol.quantize(rmsnorm(x, ol.load(lw.g_post), eps))
    else:
        xn = rmsnorm(x, ol.load(lw.g_post), eps)
        xs, xe = ol.quantize(xn), ol.quantize(xn * ol.load(lw.g_exp)[None, :])
        del xn
    lg = ol.dot(xs, lw.router)                          # [R, E] (+ the shared expert's gate)
    gid, wts = ol.empty((R, k), dense=True), ol.empty((R, k), dense=True)
    pr, tmp, ids = ol.empty((2,)), ol.empty((k,)), ol.empty((k,))
    gb = ol.load(lw.gbase)
    r = b.scratch()
    for q in range(R):                                  # each row's k best, as moe_ffn's
        sc, sel = ol.empty((E,)), ol.empty((E,))
        if mo.rule == "softmax":
            sc.set(lg[q, 0:E])
            sel.set(sc)
        else:
            sc.set(sigmoid(lg[q, 0:E]))
            sel.set(sc + ol.load(lw.ebias))
        wt = wts[q, :]
        lp = b.begin_loop(k)
        b.emit(I.argmax(pr.base, sel.base, 1, E, comment="moe: best"))
        b.emit(I.rld(r, pr.base + 1, comment="its index"))
        b.emit(I.vop(I.V_FILL, sel.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, NEG, ra=r,
                     comment="knock out"))
        if k > 1:
            tmp[0:k - 1].set(wt[1:k])
            wt[0:k - 1].set(tmp[0:k - 1])
            tmp[0:k - 1].set(ids[1:k])
            ids[0:k - 1].set(tmp[0:k - 1])
        b.emit(I.vop(I.V_COPY, wt.base + k - 1, sc.base, 0, 1, 1, 0, 0, 0, rb=r,
                     comment="weight"))
        b.emit(I.vop(I.V_COPY, ids.base + k - 1, pr.base + 1, 0, 1, 1, 0, 0, 0, comment="id"))
        b.end_loop(lp)
        del sc, sel
        if mo.rule == "softmax":                        # column 0 holds the largest
            e = ol.exp2((wt - wt[0:1]) * ol.LOG2E)
            wt.set(e * ol.recip(ol.sum(e)))
            del e
        elif mo.norm:
            wt.set(wt * ol.recip(ol.sum(wt) + 1e-6))
        if mo.scale != 1.0:
            wt.set(wt * float(mo.scale))
        gid[q, :].set(ids + gb)                         # global ids: j * E + index
    del ids, tmp
    gf = gid.reshape(1, N)[0, :]                        # the request's entries, row by row
    # pe's rows (columns: the entries; every loop over them rotates pe left by one, so entry
    # n is at column 0 in iteration n): present and computed here, missing, the directory
    # offset, and per row q the byte offset of the entry's output for q in the scratch
    EP, MISS, OFF, ANS, TG = 0, 1, 2, 3, 4
    RB, SINK = 4 * H, N * 4 * H
    pe, tpe = ol.empty((TG + R, N)), ol.empty((TG + R, N))
    rank = ol.empty((k,))
    for j in range(k):
        rank[j:j + 1].set(float(j))
    dup = ol.zeros((N,))
    for q in range(R):
        d = ol.empty((N, k))
        d.set(gf[:, None])
        eq = 1.0 - ol.minimum(ol.abs(d - gid[q, :][None, :]), 1.0)
        del d
        found = ol.sum(eq)                              # [N]: row q chose entry n's expert
        at = ol.sum(eq * rank[None, :])                 # its rank in row q
        del eq
        pe[TG + q, :].set(found * ((at + float(q * k)) * float(RB) - float(SINK)) + float(SINK))
        if q + 1 < R:                                   # later rows' entries chosen before
            dup[(q + 1) * k:N].set(dup[(q + 1) * k:N] + found[(q + 1) * k:N])
        del found, at
    pe[OFF, :].set(gf * 8.0)
    for n in range(N):                                  # each entry's word of the answer
        pe[ANS, n:n + 1].set(float(4 * n))

    def rotate():
        tpe[:, 0:N - 1].set(pe[:, 1:N])
        tpe[:, N - 1:N].set(pe[:, 0:1])
        pe[:, :].set(tpe[:, :])

    # the fence, then the request: the ids, their count, seq + 1
    word = ol.empty((1,))
    seq = ol.load(Tensor(Affine(dev.mbox), (1,), (1,)))
    b.rld(r, seq, raw=True, comment="seq (bits)")
    b.waitw(word, dev.served, 0, I.C_GE, rc=r, comment="fence: served >= seq")
    ol.store(Tensor(Affine(dev.mbox + LINE), (N,), (1,)), gf)
    ol.store(Tensor(Affine(dev.mbox + 4), (1,), (1,)), ol.full((1,), float(N)))
    seq.set(seq + 1.0)
    ol.store(Tensor(Affine(dev.mbox), (1,), (1,)), seq)
    if beside is not None:                              # while the host streams
        b.unscratch(r)
        beside()
        r = b.scratch()
    col = lambda row: pe.base + row * pe.rs                       # noqa: E731 (column 0)
    lp = b.begin_loop(N)                                # the directory: present flags
    b.emit(I.rld(r, col(OFF), comment="entry offset"))
    b.emit(I.ld(dev.dir + 4, col(EP), 1, ra=r, comment="entry: present"))
    rotate()
    b.end_loop(lp)
    keep = 1.0 - ol.minimum(dup, 1.0)                   # entries computed here: the union
    pe[MISS, :].set((pe[EP, :] * -1.0 + 1.0) * keep)
    pe[EP, :].set(pe[EP, :] * keep)
    del keep, dup
    ex = dev.fmt.descs(DevVar("expert slot", r, align=LINE))      # (slots: LINE-aligned)

    def expert():
        """The expert whose slot is R[r] on every row; each row's output to its place."""
        o = swiglu_down(xe, ex.wg, ex.wu, ex.wd, act=ACTS[mo.act])
        b.check_live(o)
        for q in range(R):
            b.emit(I.rld(r, col(TG + q), comment="its place for this row"))
            ol.store(Tensor(Affine(dev.scratch) + DevVar("place", r), (H,), (1,)), o[q, :])

    for flag, wait in ((EP, False), (MISS, True)):
        lp = b.begin_loop(N)
        b.emit(I.rld(r, col(flag), comment="missing" if wait else "present"))
        inner = b.begin_loop(0, rcount=r)               # (the count is read here: r is free)
        slot_of(r, word, col(ANS) if wait else col(OFF), dev, wait)
        expert()
        b.end_loop(inner)
        rotate()
        b.end_loop(lp)
    b.unscratch(r)
    ol.store(Tensor(Affine(dev.answer), (N,), (1,)), ol.zeros((N,)))   # (for the next request)
    out = ol.empty((R, H))
    for q in range(R):                                  # each row's sum in its router's order
        acc = None
        for i in range(k):
            y = ol.load(Tensor(Affine(dev.scratch + (q * k + i) * RB), (1, H), (H, 1))) * \
                wts[q, i:i + 1]
            acc = y if acc is None else acc + y
        out[q:q + 1, :].set(acc)
        del acc, y
    if mo.shared:
        g = ol.empty((R,))
        g.column().set(lg[:, E:E + 1])
        gate = sigmoid(g)
        out = out + swiglu_down(xs, lw.wg, lw.wu, lw.wd) * gate[:, None]
    return x + out if residual else out
