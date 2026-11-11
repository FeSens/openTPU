"""A pre-norm ViT encoder on openTPU: SigLIP's blocks (LayerNorm with a bias, linears with a bias,
a GELU MLP, bidirectional multi-head attention over a crop's patches), LLaVA-UHD's 2 x 2 window
attention and 2 x 2 merges (MiniCPM-V's vision tower and projector: docs/minicpmv.md 4.2). No
new op: LayerNorm is RSUM, RSSQ and RSQRT passes, a bias one VOP add.

The tower runs as phases over a crop of N = h w patches whose rows live in DRAM (fp32 [N, C]),
each phase a program the host runs in order, its rows in groups of R (a hardware loop, so a
program's size does not grow with N). The rows are in the card's window order (the host cuts
the patches in it, minicpmv.card_order): each 4 x 4 block of the grid row-major, within it
its 2 x 2 windows row-major, within those their 4 patches. A window is then 4 consecutive
rows, its [4C] merge row those rows' words as they are, and the window merger's rows come out
in the same order one level up (the projector's windows consecutive, its rows the grid's
4 x 4 blocks row-major: HF's order). Attention and the row-wise parts do not see the order.

  vit_embed        X = P Wp^T + bp + E    (P the host's patch rows, E its position-table rows)
  vit_qkv          LayerNorm, then per head q (to the Q scratch) and k, v (appended to the
                   crop's KV scratch); the heads padded to D (zero weight rows: a head of 72
                   is a [., 128] row whose last 56 values are 0, its scores the 72's)
  vit_attend       every head's queries in groups of MCOLS rows against all N keys (no
                   mask: attention._attend_heads with a length), out rows to the O scratch
  vit_window_attend   each 2 x 2 window's 4 queries against its 4 keys (G / 4 windows
                   an MM, a block-diagonal mask, one-block softmax)
  vit_out_mlp      x += O Wo^T + bo; x += MLP(LayerNorm(x)) (the MLP optional: the window
                   merger has none); post: the stored rows LayerNorm'd (the tower's last)
  vit_merge        each 2 x 2 window's four rows as one [4C] row, LayerNorm, the MLP (and the
                   window's mean added: the window merger), to `dst` in window order

The KV scratch is untiled (VitKV: V^T rows N tokens wide), so a row group's appends are at
loop-dependent tokens.

The weights may be pieced (kernels/pieces.py: PQ, at static addresses or at the ones a table
in DRAM holds, the host's staging into expert slots): each phase binds its weights' tables at
its start, and a row group's body gives its piece registers back at its end (row_groups).
"""
from __future__ import annotations

from .. import language as ol
from ..compiler import Affine, CompileError, KVDesc, QTensor, Tensor, current
from . import pieces as PC
from .attention import _attend_heads
from .lib import gelu_tanh


class VitKV(KVDesc):
    """A crop's K / V scratch: KVDesc's layout with V^T untiled (rows `cap` tokens wide), so an
    append at a loop-dependent token is an address (vit_append). In groups (wg > 1, GALL) K's
    groups of wg tokens (KVDesc's) and V^T's rows in groups of wg dimensions, chunk c its
    tokens c D .. (one tile of cap tokens)."""

    def vt(self, h: int) -> QTensor:
        e = self._h(h)
        if self.wg > 1:
            return QTensor(Affine.of(e["vt"]), None, (self.dv, self.cap), self.cap * self.wg, 0,
                           self.D, grp=self.wg)
        return QTensor(Affine.of(e["vt"]), None, (self.dv, self.cap), self.cap, 0, self.D)

    def vt_column(self, h: int, t) -> Affine:
        return Affine.of(self._h(h)["vt"]) + Affine.of(t)

    def vt_place(self, h: int, t) -> tuple:
        return Affine.of(self._h(h)["vt"]), Affine.of(t)


def vit_append(kv: VitKV, h: int, t, k, v) -> None:
    """Append rows k, v ([n, d], d the padded head) at token t (static, loop-dependent or a
    run-time value) of head h: K token-major with its fp32 row scale (QST ROW), V transposed
    with a scale per token (kv_append's quantization); in groups rows t .. of the grouped K and
    columns t .. of the grouped V^T (QST GRP, GRP COL)."""
    b = current()
    t = Affine.of(t)
    kd = kv.k(h)
    e = kv._h(h)
    if kv.wg > 1:
        b.store_quantized(k, kd.data, kd.scale, 0, kd.srs, row_scale=kd.rsc, row=t)
        b.store_quantized(v, Affine.of(e["vt"]), Affine.of(e["vs"]) + t * 4, 0, kv.vt(h).rs,
                          row_scale=True, row=t, col=True)
        return
    b.store_quantized(k, kd.data + t * kd.rs, kd.scale + t * kd.srs, kd.rs, 1, row_scale=kd.rsc)
    b.store_quantized(v, kv.vt_column(h, t), Affine.of(e["vs"]) + t * 4, 1, kv.cap,
                      row_scale=True)


def layernorm_(x, w, b, eps: float, cols: int = 384) -> None:
    """x's rows [R, C] LayerNorm'd in place with gain w and bias b ([C] tiles): mean, then the
    centered variance (two passes, as torch); the elementwise passes `cols` columns at a time
    (their temps [R, cols], not [R, C])."""
    n = 1.0 / x.cols
    mean = ol.sum(x, axis=1) * n
    parts = [(c, min(c + cols, x.cols)) for c in range(0, x.cols, cols)]
    for c0, c1 in parts:
        xv = x[:, c0:c1]
        xv.set(xv - mean[:, None])
    r = ol.rsqrt(ol.sumsq(x) * n + eps)
    for c0, c1 in parts:
        xv = x[:, c0:c1]
        xv.set((xv * r[:, None]) * w[c0:c1][None, :] + b[c0:c1][None, :])


def layernorm_q(x, w, eps: float, inplace: bool = False):
    """The quantized operand of LayerNorm(x) with gain w and NO bias (the caller folds the
    bias into the next linear's: W (y + b) = W y + W b, minicpmv.VisionImage.build): x
    centered on the VPU (in place, or a temp), then QACT's row and column scales give it
    rsqrt(var + eps) and w (QACT RSCALE / CSCALE): no normalized tile in TMEM."""
    n = 1.0 / x.cols
    mean = ol.sum(x, axis=1) * n
    if inplace:
        x.set(x - mean[:, None])
        xc = x
    else:
        xc = x - mean[:, None]
    r = ol.rsqrt(ol.sumsq(xc) * n + eps)
    return ol.quantize(xc, free=True, rscale=r, cscale=w)


def row_groups(n: int, R: int, body) -> None:
    """body(r0, rows) over rows 0 .. n - 1 in groups of R: a hardware loop over the whole groups
    (r0 loop-affine), then the rest at a static r0."""
    full, tail = divmod(n, R)
    if full == 1:
        body(0, R)
        PC.flush()
    elif full > 1:
        for g in ol.range(full):
            body(g * R, R)
            PC.flush()
    if tail:
        body(full * R, tail)
        PC.flush()


def _row_parts(w, r0: int, r1: int) -> list:
    """[r0, r1) of w's rows as slices an MM can stream: one, or a pieced matrix's pieces."""
    return w.row_split(r0, r1) if isinstance(w, PC.PQ) else [(r0, r1)]


def mlp_bias(xs, w1: QTensor, b1: Tensor, w2: QTensor, chunk: int, act=gelu_tanh, acc=None,
             rows: int | None = None):
    """W2 act(W1 x + b1) for x quantized in xs, F in chunks: chunk c's W1 rows stream while the
    VPU works on chunk c - 1, and its activations (quantized on their own) multiply W2's
    columns of the chunk, added into `acc` (a [R, out] tile: the residual) or a new tile.
    b2 is the caller's. A chunk whose W1 rows lie in several pieces (pieces.PQ) is one MM a
    piece, each writing its columns of the chunk's tile (rows: xs's)."""
    F = w1.shape[0]
    if F % chunk:
        raise CompileError(f"MLP width {F} is not a multiple of its chunk {chunk}")
    starts = list(range(0, F, chunk))

    def up(c0):
        parts = _row_parts(w1, c0, c0 + chunk)
        if len(parts) == 1:
            return ol.dot(xs, w1[c0:c0 + chunk, :])
        h = ol.empty([rows, chunk])
        for a, b in parts:
            ol.dot(xs, w1[a:b, :], out=h[:, a - c0:b - c0])
        return h

    y = acc
    nxt = up(starts[0])
    for i, c0 in enumerate(starts):
        h = nxt
        if i + 1 < len(starts):
            nxt = up(starts[i + 1])
        a = act(h + ol.load(b1[c0:c0 + chunk])[None, :])
        if y is None:
            y = ol.dot(a, w2[:, c0:c0 + chunk])
        else:
            ol.dot(a, w2[:, c0:c0 + chunk], acc=y)
    return y


# =============================================================================== the phases
@ol.jit
def vit_embed(v, N: int, R: int):
    """X[r] = P[r] Wp^T + bp + E[r] for the crop's N rows."""
    emb = PC.bind(v.emb)
    bp = ol.load(emb.bp)

    def body(r0, n):
        ps = ol.quantize(ol.load(v.P[r0:r0 + n, :]), free=True)
        x = ol.load(v.E[r0:r0 + n, :])
        for a, c in _row_parts(emb.wp, 0, emb.wp.shape[0]):    # (a piece's output columns)
            ol.dot(ps, emb.wp[a:c, :], acc=x[:, a:c])
        x.set(x + bp[None, :])
        ol.store(v.X[r0:r0 + n, :], x)
    row_groups(N, R, body)


@ol.jit
def vit_qkv(v, lw, src, N: int, R: int, eps: float):
    """LayerNorm (ln1) of src's N rows, then per head q -> v.Q[h] (head-major: one store a
    head), k and v -> the KV scratch. The weights are the checkpoint's [C, C] (a head's dv
    rows; in groups dq rows a head, those past dv zero: minicpmv.VisionImage); each head's
    q, k, v land in the first dv columns of [R, dp] tiles whose other
    columns stay 0 (two heads' sets, alternating): the MXU streams no padding rows, the cache
    and Q get D-wide rows. Each starts as its bias (a fill) and the MM accumulates into it."""
    H, dp, dh, dq = v.heads, v.dp, v.dv, v.dq
    lw = PC.bind(lw)
    w = ol.load(lw.ln1w)
    bs = ol.load(lw.bq), ol.load(lw.bk), ol.load(lw.bv)
    ws = lw.wq, lw.wk, lw.wv
    tiles = [[ol.zeros([R, dp]) for _ in range(3)] for _ in range(2)]

    def body(r0, n):
        x = ol.load(src[r0:r0 + n, :])
        xs = layernorm_q(x, w, eps, inplace=True)
        del x
        for h in range(H):
            q, k, vv = (t[0:n, :] for t in tiles[h % 2])
            for t, wt, b in zip((q, k, vv), ws, bs):
                t[:, 0:dh].set(b[h * dh:(h + 1) * dh][None, :])
                ol.dot(xs, wt[h * dq:h * dq + dh, :], acc=t[:, 0:dh])
            ol.store(v.Q[h][r0:r0 + n, :], q)
            vit_append(v.kv, h, r0, k, vv)
    row_groups(N, R, body)


@ol.jit
def vit_attend(v, N: int, scale: float, block: int):
    """Every head's queries (v.Q) against the N keys of the KV scratch, in groups of MCOLS
    rows, no mask; the outputs (dv of each head) to v.O."""
    H, dv = v.heads, v.dv
    G = ol.mxu_columns()

    def body(r0, g):
        qs = [lambda h=h: ol.load(v.Q[h][r0:r0 + g, :]) for h in range(H)]
        o = ol.empty([g, v.C])          # the group's rows, every head's columns: one store

        def emit(i, acc, l):
            o[:, i * dv:(i + 1) * dv].set(acc * ol.recip(l)[:, None])
        _attend_heads(qs, v.kv, list(range(H)), N, block, scale, emit=emit)
        ol.store(v.O[r0:r0 + g, :], o)
    row_groups(N, G, body)


@ol.jit
def vit_window_attend(v, N: int, G: int, scale: float, heads=None):
    """Each 2 x 2 window's 4 queries against its 4 keys, every head (or `heads`); outputs to
    v.O. The rows are in the card's window order (minicpmv.card_order: window w is rows 4w ..
    4w + 3), so G / 4 windows are G consecutive rows: their scores [G, G] against their G keys
    in one MM, a block-diagonal mask (-inf off the windows) added, the softmax in one pass (one
    block: no running maximum), then P V^T. The MXU streams D-aligned rows of V^T, so P goes
    into a [G, D] tile at the group's offset in its D tokens (zero elsewhere, the band cleared
    after its QACT) and V^T streams from the D-aligned token: the D / G groups of each D tokens
    unrolled, the D-token stretches a hardware loop (the last one whole: the rows past N are
    the scratch's, their windows see only themselves and nothing reads them back)."""
    H, dp, dv = v.heads, v.dp, v.dv
    hs = list(range(H)) if heads is None else list(heads)
    D = ol.block_size()
    if G % 4 or D % G or N % G:
        raise CompileError(f"window attention: groups of {G} rows (4-row windows) of {N} rows, "
                           f"D = {D}")
    mask = None
    if G > 4:
        mask = ol.full([G, G], float("-inf"))
        for i in range(0, G, 4):
            mask[i:i + 4, i:i + 4].set(0.0)
    sc = ol.full([dp], scale)
    pp = ol.zeros([G, D])

    def group(seg, off):
        t0 = seg + off
        band = pp[:, off:off + G]
        out = ol.empty([G, v.C]) if heads is None else None
        for h in hs:
            qs = ol.quantize(ol.load(v.Q[h][t0:t0 + G, :]) * sc[None, :])
            K = v.kv.k(h)[t0:t0 + G, :]
            if mask is None:
                s = ol.dot(qs, K)
            else:
                s = ol.empty([G, G])
                s.set(mask)
                ol.dot(qs, K, acc=s)
            m = ol.max(s, axis=1)
            p = ol.exp2(s - m[:, None])
            l = ol.sum(p, axis=1)
            band.set(p * ol.load(v.kv.vscale(h)[t0:t0 + G])[None, :])
            o = ol.dot(ol.quantize(pp), v.kv.vt(h).padded_cols(0, dv, seg, D, off + G))
            band.set(0.0)
            if out is None:
                ol.store(v.O[t0:t0 + G, h * dv:(h + 1) * dv], o * ol.recip(l)[:, None])
            else:
                out[:, h * dv:(h + 1) * dv].set(o * ol.recip(l)[:, None])
        if out is not None:
            ol.store(v.O[t0:t0 + G, :], out)

    def stretch(seg):
        for off in range(0, D, G):
            group(seg, off)
    n = -(-N // D)                  # the last stretch whole: its rows past N are the scratch's
    if n * D > v.kv.cap:            # (their windows see only themselves; never read back)
        raise CompileError(f"window attention: {N} rows in stretches of {D}, KV scratch "
                           f"{v.kv.cap}")
    if n == 1:
        stretch(0)
    else:
        for a in ol.range(n):
            stretch(a * D)


@ol.jit
def vit_out_mlp(v, lw, dst, N: int, R: int, eps: float, chunk: int, mlp: bool = True,
                post=None, odd: bool = True):
    """dst rows x (the residual, in place): x += O Wo^T + bo, then (mlp) x += W2 gelu(W1
    LayerNorm(x) + b1) + b2; post (w, b): the rows stored LayerNorm'd with it (the tower's
    post_layernorm after its last layer). odd: x at an odd row stride, one LD a row (False: at
    its DRAM stride, one LD; the same rows, the tests' check of that)."""
    lw = PC.bind(lw)
    bo = ol.load(lw.bo)
    if mlp:
        w2 = ol.load(lw.ln2w)
        b2 = ol.load(lw.b2)
    if post is not None:
        pw = ol.load(post[0]), ol.load(post[1])

    def body(r0, n):
        os = ol.quantize(ol.load(v.O[r0:r0 + n, :]), free=True)
        # the residual at an odd row stride (ol.empty's): W_o's and W_down's drains add a
        # streamed row's results into distinct TMEM banks, several a cycle (at the rows' own
        # stride, C a multiple of the banks, they share one: one a cycle, W_down's MMs then
        # take twice their pops)
        x = ol.load(dst[r0:r0 + n, :], out=ol.empty([n, v.C]) if odd else None)
        for a, c in _row_parts(lw.wo, 0, lw.wo.shape[0]):      # (a piece's output columns)
            ol.dot(os, lw.wo[a:c, :], acc=x[:, a:c])
        x.set(x + bo[None, :])
        if mlp:
            zs = layernorm_q(x, w2, eps)
            mlp_bias(zs, lw.w1, lw.b1, lw.w2, chunk, acc=x, rows=n)
            x.set(x + b2[None, :])
        if post is not None:
            layernorm_(x, pw[0], pw[1], eps)
        ol.store(dst[r0:r0 + n, :], x)
    row_groups(N, R, body)


@ol.jit
def vit_merge(v, mw, src, dst, N4: int, R: int, eps: float, chunk: int, mean: bool,
              part: int, act=None):
    """The N4 2 x 2 windows of src's rows ([4 N4, C], the card's window order: a window's rows
    (2i, 2j), (2i, 2j + 1), (2i + 1, 2j), (2i + 1, 2j + 1) consecutive, so its [4C] row is
    those rows' words as they are): LayerNorm (mw.pre_w, pre_b), then W2 act(W1 z + b1) + b2
    (+ the window's mean: the window merger) -> dst[window]. A window's row is never held
    whole: its LayerNorm takes three passes over `part`-wide pieces, the last quantizing each
    piece into the operand."""
    C = v.C
    act = act or gelu_tanh
    K = 4 * C
    if K % part:
        raise CompileError(f"merge pieces of {part} do not split 4C = {K}")
    pieces = list(range(0, K, part))
    mw = PC.bind(mw)
    b2 = ol.load(mw.b2)

    def piece(w0, n, c):
        """Columns c .. c + part of the rows [4C] of windows w0 .. w0 + n."""
        base = src[4 * w0:4 * w0 + 4 * n, :]
        return Tensor(base.base + 4 * c, (n, part), (K, 1))

    def group(w0, n):
        s = ol.zeros([n])
        y = ol.zeros([n, mw.w2.shape[0]])
        for c in pieces:
            t = ol.load(piece(w0, n, c))
            s.set(s + ol.sum(t, axis=1))
            a = c
            while mean and a < c + part:    # the window's 4 rows, summed: t's columns by row
                e = min(c + part, (a // C + 1) * C)
                q = a % C
                y[:, q:q + e - a].set(y[:, q:q + e - a] + t[:, a - c:e - c])
                a = e
        mu = s * (1.0 / K)
        ss = ol.zeros([n])
        for c in pieces:
            t = ol.load(piece(w0, n, c))
            t.set(t - mu[:, None])
            ss.set(ss + ol.sumsq(t))
        r = ol.rsqrt(ss * (1.0 / K) + eps)
        st = ol.stationary(n, K)
        for c in pieces:
            t = ol.load(piece(w0, n, c))
            t.set(((t - mu[:, None]) * r[:, None]) * ol.load(mw.pre_w[c:c + part])[None, :]
                  + ol.load(mw.pre_b[c:c + part])[None, :])
            ol.quantize_into(st, t, c)
        if mean:
            y.set(y * 0.25)
        mlp_bias(st, mw.w1, mw.b1, mw.w2, chunk, act=act, acc=y, rows=n)
        y.set(y + b2[None, :])
        ol.store(dst[w0:w0 + n, :], y)

    row_groups(N4, R, group)
