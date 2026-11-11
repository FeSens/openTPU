"""Tiled prefill (docs/tiled_prefill.md): T prompt rows through every layer in one run, each
weight chunk streamed once for all T rows.

A prefill run of today's kernel (qwen3_rows) holds its R rows' activations in TMEM at once (about
7K words a row on Qwen3-0.6B, so R <= 8, R_max 3-4 on most layouts) and attends row by row: each
(row, KV head) pair streams its KV head's K and V^T on its own. This kernel takes T rows (T <=
ACT_ROWS: one stationary operand, every MM replays each streamed chunk for ceil(T / MCOLS) groups,
docs/isa.md MM) with what TMEM holds made independent of the projections' widths:

- the attention's projections run one KV head at a time (its K, V and query group: W_k, W_v
  and W_q split by rows; the MM results per element are the same, docs/isa.md), so TMEM holds
  one head's [T, (G + 2) d] instead of [T, (n_q + 2 n_kv) d];
- each KV head's queries attend in tiles of P = MCOLS / G rows (P G query rows, one MM), so K
  and V^T are streamed once per P rows, not once per row. The rows of a tile see different
  numbers of tokens: their last blocks take a per-row-group additive mask (RowMasks at
  compile-time positions, RowTiles at run-time ones: each row's own tile of the image's mask
  table), which gives bit for bit the online softmax of each row alone. A query group wider
  than the MXU (G > MCOLS) attends a row at a time in parts of MCOLS heads, as today's;
- the MLP keeps every chunk's activation quantized in ACT RAM (F / D blocks of T rows) and runs
  the down projection by parts of its output columns, each part summing the chunks in the same
  order (the same bits as swiglu_down's accumulation), so TMEM holds [T, H / parts], not [T, H];
- LFM2's convolution blocks run by parts of their channels (the convolution is per channel):
  in_proj's B, C and x of a part, its ring rows and taps, the gated product quantized into
  out_proj's operand; then out_proj by output parts, as W_o.

Each row's arithmetic is today's rows kernels' (qwen3_rows, qwen35_rows, lfm2_rows), so the
logits, the KV cache and the recurrent states are bit for bit those of today's prefill runs
over the same rows (tests/test_tiled.py): an MM's output is one exact integer (the MX MXU,
docs/isa.md MM: replayed or paired alike) and each operand is quantized over today's columns
(one QACT, a row's one Sa: the attention's outputs whole into W_o's, a down chunk's activation,
LFM2's gated product whole). Where the outputs whole do not fit TMEM at a layer run's rows the
range forms (ranges) quantize them a part at a time (a QACT range each, W_o an MM a range): not
today's bits, as close to the fp32 forward. Qwen3 and the Llama-likes on its code
(llama.py: Phi-4-mini, SmolLM3), Qwen3.5 (gated attention and the MLP tiled, its DeltaNet rows
today's: they already stream each weight once a run, pair of heads by pair) and LFM2 / LFM2.5.

Prompt runs (opentpu/llm/prefill.py, docs/prefill.md) take this kernel by default (not with the
engine's tile_runs False or OTPU_TILE=0: prefill.tile_flag): compile_tile_run's programs at a
run-time position of a bucket, their tokens from out[], as compile_prompt_run's. A MoE model's
expert-major layer runs (docs/offload.md 13.11) take it, Qwen3.5-MoE's by default (not with
OTPU_TILE=0), Gemma 4's when asked for (tile_runs True or OTPU_TILE=1: prefill.layer_tile_flag,
LAYER_ROWS; Engine.layer_tile: qwen35_layer_tile / gemma4_layer_tile, T rows of a chunk through
one layer from their records; compile_layer_tile), the mixer's and the dense
MLP's weights once for the T rows, the experts left to the layer's expert run as today.
"""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace

from .. import isa as I
from .. import language as ol
from ..compiler import Affine, CompileError, Tensor, current, trace_fit
from ..kernels.attention import (Additive, RowMasks, RowTiles, TileSeq, _attend_heads, _finish,
                                 _scores)
from ..kernels.lib import sigmoid, silu
from ..kernels.mlp import _chunk
from .qwen3 import (ATTN_BLOCK, ATTN_DEPTH, RunRows, RunWords, _amask, _inputs_rows, _layers,
                    _lm_head_rows, _load_opt, _norm_heads, _padded, _rope_gate,
                    _rope_rows_padded)

QPART = 8       # a query part's [T, heads d] at most TMEM / QPART words (fewer heads, more rows)
SHALLOW = 2     # the attention pipeline's depth of a run over IMEM at ATTN_DEPTH
LOOP = True     # a prompt run's attention in hardware loops over KV heads and packed tiles
LOOPS = (2, 1, 0)   # their levels, tried in order (_first_fit): heads and tiles, tiles, none
PEEL = False    # the last full tile after the loop (no prime of a tile past T: _heads_loop)


def packing(mcols: int, G: int) -> int:
    """Query rows of a KV head's group (G each) that one attention MM takes: MCOLS / G."""
    return max(1, mcols // G)


def tile_seq(p0: int, r0: int, n: int, block: int, G: int) -> TileSeq:
    """The blocks a packed query tile attends over: rows r0 .. r0 + n - 1 of a tile at p0
    (positions p0 + r, each counting tokens 0 .. p0 + r), G query rows each. Blocks every row
    sees whole are `full` (looped); the rest up to the last row's token are listed, each with a
    RowMasks of the tokens each row group counts (none: every row counts the whole block)."""
    first, last = p0 + r0, p0 + r0 + n - 1          # the tile's first and last rows' positions
    full = (first + 1) // block                     # blocks before the first row's own block
    items = []
    for t0 in range(full * block, last + 1, block):
        m = min(block, last + 1 - t0)               # the block's tokens up to the last row's
        keep = [(i * G, (i + 1) * G, max(0, min(m, p0 + r0 + i + 1 - t0))) for i in range(n)]
        items.append((t0, m, None if all(k == m for _, _, k in keep) else RowMasks(keep)))
    return TileSeq(full, items)


def run_seq(pos: RunRows, r0: int, n: int, G: int, cap: int) -> TileSeq:
    """tile_seq at a run-time position: rows r0 .. r0 + n - 1 of the run `pos` (a prompt run's
    RunRows, every row in its bucket), G query rows each. The bucket's whole blocks, then its
    last block (up to the cache's end), each row group with its own row's Additive tile
    (RunRows.bucket_row: the image's mask table at the row's position in the block)."""
    if pos.amask is None:
        raise CompileError("a tiled run at a run-time position needs the image's mask tiles "
                           "(RunRows amask)")
    t0 = pos.t0
    last = min(pos.block, cap - t0)
    tiles = [(i * G, (i + 1) * G, pos.bucket_row(r0 + i).add) for i in range(n)]
    return TileSeq(pos.blocks - 1, [(t0, last, RowTiles(tiles))])


def _norm_quantize(x, g, eps: float):
    """quantize(rmsnorm(x, g)): the QACT RSCALE / CSCALE that the peephole makes of it (the
    same bytes and scales), without tracing the [T, H] products (x * x, x * r, x * r * g)."""
    r = ol.rsqrt(ol.sumsq(x) * (1.0 / x.cols) + eps)
    return ol.quantize(x, free=True, rscale=r, cscale=ol.load(g))


class Res:
    """A tiled run's residual rows x [T, H]: in TMEM, or with a DRAM scratch (`spill`: the I/O
    area's input rows, which a prompt run does not read) stored there while a block's
    attention runs, the TMEM free for its queries and outputs, and loaded back for the
    residual add (fp32 both ways: the same bits) into its own words (x is carried across the
    layers' hardware loops: every iteration's instructions address it there)."""

    def __init__(self, x, spill=None):
        self.x, self.scratch = x, spill

    def spill(self) -> None:
        if self.scratch is not None:
            x = self.x
            self.at = (x.base, x.shape, x.rs)
            ol.store(self.scratch, x)
            self.x = None

    def back(self):
        if self.x is None:
            x = self.x = _alloc_at(*self.at)
            if x.contiguous:
                ol.load(self.scratch, out=x)
            else:                               # (rows padded: a load a row)
                for r in range(x.rows):
                    ol.load(self.scratch[r, :], out=x[r, :])
        return self.x


def _alloc_at(base: int, shape, rs: int):
    """A tile at TMEM word `base` (Res.back: a loop-carried tile back in its own words), which
    no live tile may hold."""
    import gc
    import weakref

    from ..compiler import Tile
    b = current()
    n = (shape[0] - 1) * rs + shape[-1]
    for attempt in range(2):
        b.tmem_regions = [r for r in b.tmem_regions if r[2]() is not None]
        if not any(lo < base + n and base < hi for lo, hi, _ in b.tmem_regions):
            break
        gc.collect()
    else:
        raise CompileError("TMEM: the residual rows' words are taken when they come back")
    t = Tile(b, base, shape, rs)
    t.spare = 0
    b.tmem_regions.append((base, base + n, weakref.ref(t.buf)))
    b.tmem_regions.sort(key=lambda r: r[0])
    return t


def _attention_tile(res: Res, lw, c, s_, pos, T: int, spec, block: int, parts: int,
                    gated: bool = False, depth: int = ATTN_DEPTH, whole: bool = True,
                    oscr=None, loop: int = 0):
    """x += W_o . attention(x) for T rows of sequence 0 at positions pos .. pos + T - 1 (pos an
    int, or a prompt run's RunRows: run-time positions), their K / V appended first, each row
    attending causally over the cache: _attention_rows' arithmetic, one KV head's projections
    at a time, its query group in parts of at most MCOLS heads (a part's [T, heads d] at most
    TMEM / QPART words: Qwen3.5-35B's G = 8) and each part's rows in packed tiles (MCOLS query
    rows). gated (Qwen3.5): each head's outputs times sigmoid(W_gate x) before W_o, as
    _attention_rows' (acc / l) * sigmoid(gate). x: res.x, spilled while the heads attend (Res).
    whole: the outputs [T, n_q d] quantized at once into W_o's operand, as _attention_rows'
    (one QACT: a row's one Sa, one MM a W_o part): each part's stored to `oscr` (the image's
    I/O scratch, _oscr) as it ends and loaded back whole once the heads are done, or without
    oscr gathered in TMEM; else each part's quantized as it ends (quantize_into: a QACT range
    of its own, an MM of W_o's a range, docs/isa.md MM; not the same bits)."""
    kvc = lw.kvs[0]
    d, G, eps = spec.head_dim, spec.n_q // kvc.kv_heads(spec.n_kv), spec.eps
    D = ol.block_size()
    dk = -(-d // D) * D
    run = isinstance(pos, RunRows)
    S = ol.num_programs()
    if S > 1:                                  # the slice's heads in TMEM, gathered before W_o
        whole, oscr, loop = True, None, 0
    xs = _norm_quantize(res.x, lw.g_in, eps)   # T rows: one pass a weight
    qn, kn = _load_opt(lw.qn), _load_opt(lw.kn)
    c, s_ = _rope_gate(lw, c, s_)
    res.spill()                                # (after the tiles that outlive the heads)
    scale = ol.LOG2E / math.sqrt(d)
    heads = list(kvc.owned_heads(spec.n_kv))
    nq = len(heads) * G
    mc = ol.mxu_columns()
    gs = min(G, mc)                            # a part of a KV head's query group: its heads
    while gs % 2 == 0 and T * gs * dk > ol.tmem_words() // QPART:
        gs //= 2                               # (a part's [T, gs dk] queries in TMEM / QPART)
    P = packing(mc, gs)                        # rows of a tile (P gs <= MCOLS query rows)
    tiles = [(r0, min(P, T - r0)) for r0 in range(0, T, P)]
    gparts = [(g0, min(G, g0 + gs)) for g0 in range(0, G, gs)]
    if whole and oscr is None:                 # the outputs, quantized once (W_o's operand)
        o_all, ost = ol.empty([T, nq * d]), None
    else:                                      # W_o's operand, a part's columns at a time
        o_all, ost = None, None if whole else ol.stationary(T, nq * d)
    if oscr is not None:                       # (the scratch as [T, n_q d]: one load back)
        oscr = Tensor(oscr.base, (T, nq * d), (nq * d, 1))
    looped = LOOP and loop and _loopable(pos, T, P, G, gs, kvc, heads, block, loop, whole
                                         and not gated and oscr is not None)
    if looped:                                 # (the outputs to the scratch, as below)
        _attention_loop(lw, xs, qn, kn, c, s_, pos, T, spec, heads, G, gs, P, dk, block,
                        depth, scale, oscr, loop >= 2)
    made, oh = {}, {}                          # (KV head, part) -> its queries, its outputs
    ent, qhs, kvh, seqs = [], [], [], []

    def part(j, g0, g1):
        """KV head j's projections and K / V appends (with its first part), then the queries
        of heads g0 .. g1 of its group [T (g1 - g0), dk], row (r, g) = r (g1 - g0) + g."""
        hh = heads[j]
        if g0 == 0:
            k = ol.dot(xs, lw.wk[j * d:(j + 1) * d, :])
            v = ol.dot(xs, lw.wv[j * d:(j + 1) * d, :])
            kj = _rope_rows_padded(_norm_heads(k, kn, eps), c, s_)
            vj = _padded(v)
            if run:                            # V^T at a run-time position: a row at a time
                for r in range(T):
                    ol.kv_append(kvc, hh, pos.pos + r, kj[r:r + 1, :], vj[r:r + 1, :])
            else:
                ol.kv_append(kvc, hh, pos, kj, vj)
            del k, v, kj, vj
        w = g1 - g0
        q = ol.dot(xs, lw.wq[(j * G + g0) * d:(j * G + g1) * d, :])    # [T, w d]
        Q = ol.empty([T * w, dk], dense=True)
        if dk > d:                                             # heads padded to whole blocks
            Q.set(0.0)
        Qv = Q.reshape(T, w * dk)
        for g in range(w):                                     # head g of every row at once
            _rope_rows_padded(_norm_heads(q[:, g * d:(g + 1) * d], qn, eps), c, s_,
                              out=Qv[:, g * dk:g * dk + d])
        return Q

    def query(j, g0, g1, r0, n):
        def emit():
            if (j, g0) not in made:
                made[(j, g0)] = part(j, g0, g1)
            Q, w = made[(j, g0)], g1 - g0
            if r0 + n == T:
                made.pop((j, g0))
            return Q[r0 * w:(r0 + n) * w, :]
        return emit

    for j in range(len(heads)):
        for g0, g1 in gparts:
            for r0, n in tiles:
                ent.append((j, g0, g1, r0, n))
                qhs.append(query(j, g0, g1, r0, n))
                kvh.append(heads[j])
                seqs.append(run_seq(pos, r0, n, g1 - g0, kvc.cap) if run else
                            tile_seq(pos, r0, n, block, g1 - g0))

    def emit(i, acc, l):
        j, g0, g1, r0, n = ent[i]
        w = g1 - g0
        if o_all is not None:                  # (the part's columns of the outputs)
            o, o0 = o_all, (j * G + g0) * d
        else:
            if (j, g0) not in oh:
                oh[(j, g0)] = ol.empty([T, w * d])
            o, o0 = oh[(j, g0)], 0
        for a in range(n):
            o[r0 + a, o0:o0 + w * d].reshape(w, d).set(
                acc[a * w:(a + 1) * w, :] / l[a * w:(a + 1) * w][:, None])
        if r0 + n == T:                        # the part's rows done
            if gated:                          # a head at a time (TMEM: [T, d] temporaries)
                for g in range(w):
                    h0 = (j * G + g0 + g) * d
                    og = o[:, o0 + g * d:o0 + (g + 1) * d]
                    og.set(og * sigmoid(ol.dot(xs, lw.wgate[h0:h0 + d, :])))
                    del og
            if o_all is None:                  # to its columns of the scratch, or quantized
                h0 = (j * G + g0) * d          # into them; freed
                if whole:
                    ol.store(oscr[0:T, h0:h0 + w * d], oh.pop((j, g0)))
                else:
                    ol.quantize_into(ost, oh.pop((j, g0)), h0)
        del o

    if not looped:
        _attend_heads(qhs, kvc, kvh, seqs, block, scale, depth=depth, emit=emit)
    xs = qn = kn = None                        # (the norm gains too: placed after x's tiles,
    #                                            they split TMEM under the load back)
    if S > 1:                                  # every slice's heads [T, n_q d], one QACT
        ost = ol.quantize(ol.all_gather(o_all), free=True)      # (_o_residual's operand)
        del o_all
        x, hl, sid = res.back(), lw.wo.shape[0], ol.program_id()
        x.set(ol.all_gather(x[:, sid * hl:(sid + 1) * hl] + ol.dot(ost, lw.wo)))
        return
    if whole:                                  # (the scratch's rows back: the stores before)
        if o_all is None:
            o_all = ol.load(oscr[0:T, 0:nq * d])
        ost = ol.quantize(o_all, free=True)
        del o_all
    _add_parts(res.back(), [ost], [lw.wo], parts)              # x += W_o o, by output parts


def _loopable(pos, T: int, P: int, G: int, gs: int, kvc, heads, block: int, level: int,
              out: bool, tspan=None) -> bool:
    """_attention_tile's heads attend in hardware loops (_attention_loop, _heads_loop): a
    prompt run at a run-time position with the image's mask tiles (every packed tile's span the
    bucket's, its last block whole; tspan(0): a tile's span, whole blocks), the outputs whole
    through the I/O scratch (out), whole parts (G % gs), one slice, the KV heads' blocks evenly
    apart (a loop over them: level 2), and loop depth to spare."""
    if not (out and isinstance(pos, RunRows) and pos.amask is not None and G % gs == 0
            and kvc.S == 1):
        return False
    if tspan is None:
        if min(pos.block, kvc.cap - pos.t0) != block:
            return False
    else:
        try:
            if any(n != block for _, n in tspan(0)[0]):
                return False
        except CompileError:
            return False
    hl = level >= 2 and len(heads) > 1
    if hl and _head_step(kvc, heads) is None:
        return False
    return len(current().loops) + hl + (T // P > 1) <= 4


def _span_blocks(sq, block: int, cap: int):
    """(blocks [(t0, n)], masks {t0: mask}) of a packed tile's span (pack_seq's: a TileSeq,
    Blocks, or one row's Bucket with its mask tile)."""
    from ..kernels.attention import Blocks, Bucket
    if isinstance(sq, TileSeq):
        items = list(sq.items)
        return ([(i * block, block) for i in range(sq.full)] + [(t0, n) for t0, n, _ in items],
                {t0: mk for t0, _, mk in items if mk is not None})
    if isinstance(sq, Blocks):
        return [(t0, n) for t0, n, _ in sq.items], {t0: mk for t0, _, mk in sq.items
                                                    if mk is not None}
    if isinstance(sq, Bucket) and sq.add is not None:
        t0 = (sq.blocks - 1) * block
        return [(i * block, block) for i in range(sq.blocks - 1)] + [(t0, min(block, cap - t0))], \
            {t0: sq.add}
    raise CompileError(f"no looped span of {type(sq).__name__}")


def _head_step(kvc, heads):
    """The bytes from one KV head's cache blocks to the next one's (the same for K, its scales,
    V^T and V's scales, and for every head), or None."""
    e = [{k: Affine.of(a) for k, a in kvc._h(h).items()} for h in heads]
    diff = [{k: e[i][k] - e[0][k] for k in e[0]} for i in range(len(e))]
    if any(x.terms for dd in diff for x in dd.values()):
        return None
    step = {x.const for x in diff[1].values()}
    if len(step) != 1:
        return None
    step = step.pop()
    if any(x.const != i * step for i, dd in enumerate(diff) for x in dd.values()):
        return None
    return step


def _attention_loop(lw, xs, qn, kn, c, s_, pos, T: int, spec, heads, G: int, gs: int, P: int,
                    dk: int, block: int, depth: int, scale: float, oscr,
                    hloop: bool = True) -> None:
    """_attention_tile's K / V appends and attention in hardware loops (_heads_loop), the
    outputs to `oscr` ([T, n_q d], the I/O scratch): every KV head's K and V projected and
    appended first (unrolled: their run-time rows), then the heads' queries and attention."""
    d, eps = spec.head_dim, spec.eps
    kvc = lw.kvs[0]
    for j, hh in enumerate(heads):             # every head's K / V rows at their positions
        k = ol.dot(xs, lw.wk[j * d:(j + 1) * d, :])
        v = ol.dot(xs, lw.wv[j * d:(j + 1) * d, :])
        kj = _rope_rows_padded(_norm_heads(k, kn, eps), c, s_)
        vj = _padded(v)
        for r in range(T):
            ol.kv_append(kvc, hh, pos.pos + r, kj[r:r + 1, :], vj[r:r + 1, :])
        del k, v, kj, vj

    def query(hq, Qv):
        """Query heads hq .. hq + gs of every row (W_q's rows: hq a loop's index), normed and
        RoPE'd into Qv[0:T, g dk ..] (head g)."""
        q = ol.dot(xs, lw.wq[hq * d:hq * d + gs * d, :])                # [T, gs d]
        for g in range(gs):                             # head g of every row at once
            _rope_rows_padded(_norm_heads(q[:, g * d:(g + 1) * d], qn, eps), c, s_,
                              out=Qv[0:T, g * dk:g * dk + d])
        del q

    mbase, mstep = pos.amask
    blocks = [(i * block, block) for i in range(pos.blocks - 1)] + [(pos.t0, block)]

    def span(t, n=P):
        """Tile t's blocks (the bucket's) and its masked block's row tiles (its n rows, row t P
        + i: the image's mask tile at tpos + t P + i)."""
        tl = [(i * gs, (i + 1) * gs, Additive(Affine(mbase + mstep * i) + Affine.of(t) * (
            P * mstep) + pos.tpos * mstep, block)) for i in range(n)]
        return blocks, {pos.t0: RowTiles(tl)}

    _heads_loop(kvc, heads, T, G, gs, P, d, dk, block, depth, scale, oscr, hloop, query, span)


def _heads_loop(kvc, heads, T: int, G: int, gs: int, P: int, d: int, dk: int, block: int,
                depth: int, scale: float, oscr, hloop: bool, query, span) -> None:
    """A tiled run's attention after its K / V appends (_attention_loop, _gemma_attention_loop):
    a hardware loop over the KV heads `heads` of kvc (hloop; else each head in turn: fewer
    address registers, a body a head) and in it its query group's parts of gs heads
    (unrolled: query(hq, Qv) projects one into a TMEM buffer), each a hardware loop over its
    packed tiles of P rows (span(t, n): tile t's blocks [(t0, n)], the same for every tile, and
    the masks of its n rows), a partial last one after it (unrolled: T % P rows of its own
    operand, state and score buffers). One packed tile's attention is the loop body
    (docs/tiled_prefill.md 13), where the unrolled form (_attend_heads) has one a KV head, part
    and tile (R x heads x blocks: what fills IMEM past the first buckets): the same
    instructions a tile (_scores / _finish: its blocks, its masked blocks' row tiles; acc / l),
    so the same bits, with their addresses affine in the loops' indices (the head's cache
    blocks, W_q's rows, the tile's queries in TMEM: QACT's src register, its outputs over them:
    VOP's dst, the same register; its mask tiles; then the part's outputs to their columns of
    the scratch). The
    pipeline runs across the tiles of a part: a tile's body ends with the next one's queries
    quantized and its first blocks' scores in flight (in score buffers of their own: every
    iteration addresses the same); the last tile's are a phantom's (the rows past T: the query
    buffer's zeroed tile, the mask tiles of the table's padding, qwen3.AMASK_PAD), never used,
    or with PEEL the last tile runs after the loop. Between parts and heads it drains."""
    rows, ntiles, nh, tail = P * gs, T // P, len(heads), T % P
    blocks, _ = span(0)
    if any(n != block for _, n in blocks):
        raise CompileError("a looped attention's blocks are whole")
    Q = ol.empty([(T + P) * gs, dk], dense=True)       # a part's queries, row (r, g) = r gs + g
    if dk > d:                                          # (heads padded to whole blocks)
        Q.set(0.0)
    else:                                               # (the phantom tile's rows)
        Q[T * gs:(T + P) * gs, :].set(0.0)
    Qv = Q.reshape(T + P, gs * dk)
    sv = ol.full([dk], scale)                           # the softmax scale (QACT CSCALE)
    qs = ol.stationary(rows, dk)                        # a tile's queries, quantized
    dv = kvc.dv
    m_, l_, acc = ol.empty([rows]), ol.empty([rows]), ol.empty([rows, dv])
    nb = len(blocks)
    npb = min(depth - 1, nb)                            # blocks in flight at a tile's start
    PB = [ol.empty([rows, block]) for _ in range(npb)]
    RB = [ol.empty([rows, block]) for _ in range(min(depth, nb - npb))]

    def buf(f):
        return PB[f] if f < npb else RB[(f - npb) % len(RB)]

    if tail:                                            # the partial tile's
        prow = tail * gs
        qs2 = ol.stationary(prow, dk)
        m2, l2, acc2 = ol.empty([prow]), ol.empty([prow]), ol.empty([prow, dv])
        PB2 = [ol.empty([prow, block]) for _ in range(npb)]
        RB2 = [ol.empty([prow, block]) for _ in range(min(depth, nb - npb))]

        def buf2(f):
            return PB2[f] if f < npb else RB2[(f - npb) % len(RB2)]

    def state(kvh, h, t):
        """Tile t's head state (_Head's fields for _scores / _finish)."""
        return SimpleNamespace(G=rows, K=kvh.k(h), VT=kvh.vt(h), VS=kvh.vscale(h), qs=qs,
                               m=m_, l=l_, acc=acc, rowmax={}, masked=None, masks=span(t)[1])

    def quantize(t):
        """Tile t's queries (rows t P gs .. of the part's buffer) into qs, times the scale."""
        at = Affine.of(t) * (rows * dk)
        ol.quantize_into(qs, Q[0:rows, :], 0, cscale=sv,
                         at=None if at.is_static and at.const == 0 else at)

    hloop = hloop and nh > 1
    step = _head_step(kvc, heads) if hloop else 0
    for j in (ol.range(nh) if hloop else range(nh)):
        if hloop:                                       # head j's blocks: head 0's + j steps
            h = heads[0]                                # (kvc's own kind: its K's layout)
            kvh = copy.copy(kvc)
            kvh.heads = {h: {k: Affine.of(a) + j * step for k, a in kvc._h(h).items()}}
        else:
            kvh, h = kvc, heads[j]
        for g0 in range(0, G, gs):
            hq = Affine.of(j) * G + g0                  # the part's first query head
            query(hq, Qv)
            if ntiles:                                  # tile 0's queries and first blocks
                quantize(0)
                st0 = state(kvh, h, 0)
                for f in range(npb):
                    _scores(st0, blocks[f][0], block, PB[f])
                del st0
            def tile(t, primes):
                """Tile t's attention (and with `primes` the next tile's first blocks')."""
                m_.set(-1e30)
                l_.set(0.0)
                acc.set(0.0)
                cur = state(kvh, h, t)
                nxt = state(kvh, h, Affine.of(t) + 1) if primes else None
                _tile_blocks(cur, nxt, blocks, depth, buf, npb,
                             lambda: quantize(Affine.of(t) + 1))
                r = ol.recip(l_)                        # the tile's outputs over its queries
                at = Affine.of(t) * (rows * dk)
                current().vop(I.V_MUL, Q[0:rows, 0:d], acc, r[:, None],
                              at=None if at.is_static and at.const == 0 else at)
                del cur, nxt, r

            nl = ntiles - 1 if PEEL else ntiles         # (PEEL: the last one after the loop)
            for t in (ol.range(nl) if nl > 1 else range(nl)):
                tile(t, PEEL or ntiles > 1)
            if PEEL and ntiles:
                tile(ntiles - 1, False)
            if tail:                                    # the partial tile, unrolled
                q0 = ntiles * rows
                ol.quantize_into(qs2, Q[q0:q0 + prow, :], 0, cscale=sv)
                m2.set(-1e30)
                l2.set(0.0)
                acc2.set(0.0)
                st = SimpleNamespace(G=prow, K=kvh.k(h), VT=kvh.vt(h), VS=kvh.vscale(h),
                                     qs=qs2, m=m2, l=l2, acc=acc2, rowmax={}, masked=None,
                                     masks=span(ntiles, tail)[1])
                for f in range(npb):
                    _scores(st, blocks[f][0], block, PB2[f])
                _tile_blocks(st, None, blocks, depth, buf2, npb, None)
                r = ol.recip(l2)
                Q[q0:q0 + prow, 0:d].set(acc2 * r[:, None])
                del st, r
            c0 = hq * d                                 # the part's outputs to the scratch
            if dk == d:
                ol.store(oscr[0:T, c0:c0 + gs * d], Qv[0:T, :])
            else:
                for r in range(T):
                    for g in range(gs):
                        ol.store(oscr[r:r + 1, c0 + g * d:c0 + (g + 1) * d],
                                 Q[r * gs + g:r * gs + g + 1, 0:d])
    Q = Qv = sv = qs = m_ = l_ = acc = PB = RB = None   # (free; the closures hold their names)
    if tail:
        del qs2, m2, l2, acc2
        PB2 = RB2 = None


def _tile_blocks(cur, nxt, blocks, depth: int, buf, npb: int, quantize_next) -> None:
    """One packed tile's blocks (_attend_heads' pipeline for one entry): each block's scores
    issued depth - 1 blocks ahead, in its buffer (buf(f)); once the tile's are all issued, the
    next tile's queries quantized (nxt: its blocks follow in the stream) and its first npb
    blocks' scores into the buffers this tile's first blocks have left; then the finishes."""
    nb = len(blocks)
    issued, primed, quantized = npb, 0, False
    for b in range(nb):
        ahead = b + depth - 1
        while issued <= min(ahead, nb - 1):
            t0, n = blocks[issued]
            _scores(cur, t0, n, buf(issued))
            issued += 1
        if nxt is not None and issued == nb:
            while primed < npb and nb + primed <= ahead and primed < b:
                if not quantized:
                    quantize_next()
                    quantized = True
                t0, n = blocks[primed]
                _scores(nxt, t0, n, buf(primed))
                primed += 1
        t0, n = blocks[b]
        _finish(cur, buf(b), t0, n)
    if nxt is not None:
        if not quantized:
            quantize_next()
        for f in range(primed, npb):
            t0, n = blocks[f]
            _scores(nxt, t0, n, buf(f))


def _add_parts(x, acts, ws, parts: int, out=None) -> None:
    """x += sum_i acts[i] . ws[i]^T, the sum in the order of the list (dot, then dot(acc=)),
    by `parts` parts of the output columns: TMEM holds [T, H / parts] of the sum at a time,
    each element summed as one [T, H] result would be. out: the sum into that [T, H] tile
    instead (x untouched: Gemma's post-norms need the whole row)."""
    H = x.cols
    hp = H // parts
    for p in range(parts):
        n0 = p * hp
        y = None
        for a, w in zip(acts, ws):
            if y is None:
                y = ol.dot(a, w[n0:n0 + hp, :])
            else:
                ol.dot(a, w[n0:n0 + hp, :], acc=y)
        if out is None:
            x[:, n0:n0 + hp].set(x[:, n0:n0 + hp] + y)
        else:
            out[:, n0:n0 + hp].set(y)
        del y


def down_chunk(wg, wd) -> int:
    """The W_down chunk swiglu_down sums by (its `chunk`, else _chunk's): W_down's stored
    column parts, else _chunk of the slice's F."""
    if wd.parts is not None:
        return wd.pw // wd.gs
    D = ol.block_size()
    return _chunk(wg.shape[0], D, D if wd.wf == 0 else 2 * D)


def _mlp_tile(x, lw, spec, parts: int, sub: int, act=silu, g=None, out=None, xs=None):
    """x + W_down(act(W_gate xn) * (W_up xn)) for T rows (xn: x normed by g, default
    lw.g_post): every chunk's activation quantized once and kept in ACT RAM (the C columns of
    a W_down part: one QACT, as swiglu_down's, of the chunk made `sub` columns at a time into
    a [T, C] tile when sub < C), then the down projection by `parts` parts of its output
    columns, each summing the chunks in swiglu_down's order (the same bits). out: the MLP's
    output into that [T, H] tile instead (_add_parts). xs: the operand, quantized already (x
    then may be None, with out).

    Where the chunks' activations do not all fit ACT RAM beside the operand (Qwen3.5-27B's F
    17408 against ACT_BLOCKS 128 x D 128: the down projection's K split, act_groups), they go
    in groups that do: a group's gate / up MMs, then its down MMs into every part's sum, the
    sums kept in TMEM across the groups (fresh MM outputs: odd row strides, no bank conflict
    on their MM ACC) and added to x after the last; each element sums the chunks in the same
    order (the same bits)."""
    if xs is None:
        xs = _norm_quantize(x, lw.g_post if g is None else g, spec.eps)
    wg, wu, wd = lw.wg, lw.wu, lw.wd
    T = (out if x is None else x).rows
    C = down_chunk(wg, wd)
    groups = act_groups(list(range(0, wg.shape[0], C)), C, xs)
    ys = None                           # (groups: every part's sum so far)
    for gi, starts in enumerate(groups):
        pieces = [(c0, s0) for c0 in starts for s0 in range(0, C, sub)]

        def gate_up(c0, s0):
            a = c0 + s0
            return ol.dot(xs, wg[a:a + sub, :]), ol.dot(xs, wu[a:a + sub, :])

        acts, a = {}, None
        nxt = gate_up(*pieces[0])
        for i, (c0, s0) in enumerate(pieces):
            g, u = nxt
            if i + 1 < len(pieces):
                nxt = gate_up(*pieces[i + 1])                  # streams while the VPU works
            if sub == C:                                       # the chunk's activation
                acts[c0] = ol.quantize(act(g) * u, free=True)
            else:
                if s0 == 0:
                    a = ol.empty([T, C])
                a[:, s0:s0 + sub].set(act(g) * u)
                if s0 + sub == C:
                    acts[c0] = ol.quantize(a, free=True)
                    a = None
            del g, u
        if gi + 1 == len(groups):
            xs = nxt = None             # the operand's ACT blocks are free for the down MMs
        nxt = None
        ws = [wd[:, c0:c0 + C] for c0 in starts]
        if len(groups) == 1:
            _add_parts(out if x is None else x, [acts[c0] for c0 in starts], ws, parts, out)
        else:
            ys = _sum_parts(ys, [acts[c0] for c0 in starts], ws, parts)
        acts = None                     # (the group's ACT blocks: the next group's)
    if ys is not None:
        dst = out if x is None else x
        hp = dst.cols // parts
        for p, y in enumerate(ys):
            n0 = p * hp
            if out is None:
                x[:, n0:n0 + hp].set(x[:, n0:n0 + hp] + y)
            else:
                out[:, n0:n0 + hp].set(y)
    return x


def act_groups(starts: list, C: int, xs) -> list:
    """_mlp_tile's chunks (their first columns) in consecutive groups whose activations ([T,
    C] each: C / D ACT RAM blocks) fit ACT RAM beside the operand xs (live while a group's gate
    and up MMs run): one group where they all do (every model's but Qwen3.5-27B's on one
    slice), else as few as fit, of sizes within one of each other."""
    cfg = current().cfg
    per = max(1, (cfg.ACT_BLOCKS - xs.KB) // (C // cfg.D))
    if per >= len(starts):
        return [starts]
    k = -(-len(starts) // per)
    q, r = divmod(len(starts), k)                       # r groups of q + 1, then q
    groups, i = [], 0
    for g in range(k):
        groups.append(starts[i:i + q + (g < r)])
        i += q + (g < r)
    return groups


def _sum_parts(ys, acts, ws, parts: int) -> list:
    """_add_parts' sums by part over these chunks, onto ys (each part's [T, H / parts] sum so
    far; None: the first group's), in the order of the list."""
    hp = ws[0].shape[0] // parts
    ys = list(ys) if ys is not None else [None] * parts
    for p in range(parts):
        n0 = p * hp
        for a, w in zip(acts, ws):
            if ys[p] is None:
                ys[p] = ol.dot(a, w[n0:n0 + hp, :])
            else:
                ol.dot(a, w[n0:n0 + hp, :], acc=ys[p])
    return ys


def mlp_parts(H: int, D: int, T: int, tmem: int) -> int:
    """The output projections' parts: the fewest (whole D blocks each, dividing H) whose [T, H /
    parts] sum takes at most an eighth of TMEM."""
    for parts in range(1, H // D + 1):
        if H % (parts * D) == 0 and T * (H // parts + 1) <= tmem // 8:
            return parts
    return H // D


def mlp_sub(C: int, D: int, T: int, tmem: int) -> int:
    """Gate / up columns an MM: the most (whole D blocks, dividing the chunk C) whose four
    [T, sub] results in flight (this piece's gate and up, the next one's), with the chunk's
    [T, C] activation when sub < C, take at most a quarter of TMEM."""
    for sub in range(C, D - 1, -D):
        if C % sub == 0 and 4 * T * (sub + 1) + (T * (C + 1) if sub < C else 0) <= tmem // 4:
            return sub
    return D


def conv_parts(n: int, K: int, D: int, T: int, tmem: int) -> int:
    """LFM2's convolution channel parts: the fewest (whole D blocks each, dividing n) whose
    rows (K - 1 + T of B * x, the part's taps, three [T, n / parts] products and the sum in
    flight) take at most a quarter of TMEM (beside the [T, n] gated product)."""
    for parts in range(1, n // D + 1):
        cp = n // parts
        if n % (parts * D) == 0 and (K - 1 + T + K + 4 * T) * (cp + 1) <= tmem // 4:
            return parts
    return n // D


def _ring_store_part(lw, last, ringo: int, K: int, n: int, c0: int) -> None:
    """lfm2._ring_store of the channels c0 .. c0 + last.cols of the state ring (`last` their
    B * x rows of the last K positions): the same rotation, each row's part to its columns."""
    b = current()
    cp = last.cols
    win = ol.empty([2 * K - 1, cp])
    win[0:K, :].set(last)
    win[K:2 * K - 1, :].set(last[0:K - 1, :])
    r = b.scratch()
    b.emit(I.rld(r, ringo, mul=win.rs, comment="the ring's rotation"))
    for y in range(1, 2 * K):
        ra, imm = b.addr(Affine.of(lw.state.base) + 4 * (n * y + c0))
        j = y - 1 if y <= K else y - K - 1
        b.emit(I.st(imm, win.base + j * win.rs, cp, ra=ra, rb=r, comment="ring row part"))
    b.unscratch(r)
    del win


def _conv_tile(x, lw, pos: RunRows, spec, parts: int, cparts: int):
    """x += out_proj(conv(B * x) * C) for T rows at a prompt run's run-time positions (LFM2's
    convolution block, lfm2._conv_rows): by `cparts` parts of the channels, each part's B, x
    and C streamed once for the T rows, its ring rows and taps loaded, the ring rewritten from
    its last K rows (the run's rotation), the gated product to its columns of [T, n]; then
    out_proj's operand quantized at once (one QACT, as _conv_rows') and out_proj by output
    parts. Per channel and row _conv_rows' products and sums in its order: the same bits."""
    from .lfm2 import _ring_rows
    K, T = spec.conv_k, x.rows
    n = lw.taps.shape[1]
    if not isinstance(pos, RunRows):
        raise CompileError("a tiled convolution block runs at a prompt run's position")
    xs = _norm_quantize(x, lw.g_in, spec.eps)
    y = ol.empty([T, n])                                       # the gated product
    cp = n // cparts
    rows = _ring_rows(K, pos)[0]
    for q in range(cparts):
        c0 = q * cp
        taps = ol.load(lw.taps[:, c0:c0 + cp])                 # [K, cp]
        E = ol.empty([K - 1 + T, cp])                          # B * x, positions p-K+1 ..
        for j, row in zip(range(1, K), rows):                  # the ring's rows before the run
            ol.load(lw.state[row, c0:c0 + cp], out=E[K - 1 - j, :])
        bx = E[K - 1:K - 1 + T, :]
        bx.set(ol.dot(xs, lw.win[c0:c0 + cp, :]) *
               ol.dot(xs, lw.win[2 * n + c0:2 * n + c0 + cp, :]))
        _ring_store_part(lw, E[T - 1:T - 1 + K, :], pos.words["ringo"], K, n, c0)
        acc = bx * taps[K - 1, :][None, :]
        for j in range(1, K):
            acc = acc + E[K - 1 - j:K - 1 + T - j, :] * taps[K - 1 - j, :][None, :]
        del E, bx, taps
        y[:, c0:c0 + cp].set(acc * ol.dot(xs, lw.win[n + c0:n + c0 + cp, :]))
        del acc
    xs = None
    yst = ol.quantize(y, free=True)
    del y
    _add_parts(x, [yst], [lw.wout], parts)
    return x


def _oscr(m, on: bool = True):
    """The image's I/O scratch for a tiled run's attention outputs ([ACT_ROWS, n_q d], the
    widest layer's: the image's io "o"), or None (`on` False, the form's oscr: the outputs
    gathered in TMEM)."""
    return getattr(m, "oscr", None) if on else None


def _widths(m, T: int):
    """(parts, the MLP's sub of a layer): the output projections' parts and the gate / up
    piece width (mlp_parts, mlp_sub) of this slice's hidden and F."""
    spec = m.spec
    D, W = ol.block_size(), ol.tmem_words()
    parts = mlp_parts(spec.hidden // ol.num_programs(), D, T, W)
    return parts, lambda lw: mlp_sub(down_chunk(lw.wg, lw.wd), D, T, W)


def _pieces(T: int, most: int) -> list:
    """T rows in the fewest runs of at most `most` rows, as even as they go ((r0, n), the
    longer first: 6 by 4 is 3 + 3, not 4 + 2, whose 2 rows a 4-bit MM would pair at MCOLS 4)."""
    k = -(-T // most)
    q, r = divmod(T, k)
    out, r0 = [], 0
    for i in range(k):
        n = q + (i < r)
        out.append((r0, n))
        r0 += n
    return out


def _rows_of(pos, T: int):
    return pos if isinstance(pos, RunRows) else [(0, pos + r) for r in range(T)]


@ol.jit
def qwen3_tile(m, p0, T: int, logit_rows, block: int = ATTN_BLOCK, spill: bool = False,
               depth: int = ATTN_DEPTH, whole: bool = True, oscr: bool = True, loop: int = 2):
    """T rows at positions p0 .. p0 + T - 1 of sequence 0 (p0 an int: their embedding and RoPE
    rows in the I/O area, as qwen3_rows'; a RunRows: a prompt run's tokens and positions), the
    logits of `logit_rows` (empty: none). spill: x in the I/O area's rows while each layer's
    heads attend (Res). depth: the attention's pipeline (_attend_heads: blocks in flight a
    head; fewer, a smaller program: compile_tile_run's forms). whole: W_o's operand quantized
    at once (_attention_tile's; else a QACT range a part). loop: the attention's loops
    (_attention_loop's level: 2 heads and tiles, 1 tiles, 0 none)."""
    spec = m.spec
    x, c, s_ = _inputs_rows(m, _rows_of(p0, T))
    res = Res(x, m.xr[0:T, :] if spill else None)
    del x
    parts, sub = _widths(m, T)
    for lw in _layers(m):
        _attention_tile(res, lw, c, s_, p0, T, spec, block, parts, depth=depth, whole=whole,
                        oscr=_oscr(m, oscr), loop=loop)
        _mlp_tile(res.x, lw, spec, parts, sub(lw))
    _lm_head_rows(res.x, m, spec, logit_rows)


@ol.jit
def qwen35_tile(m, pos, T: int, logit_rows, block: int = ATTN_BLOCK, hidden: bool = False,
                spill: bool = False, drows: int = 0, depth: int = ATTN_DEPTH,
                whole: bool = True, oscr: bool = True, loop: int = 2):
    """qwen35_rows tiled: T rows of a prompt run (pos a RunRows), its attention layers and
    MLPs tiled, its DeltaNet layers today's rows (_deltanet_rows over the T rows, or in runs
    of `drows` rows one after another: their pair buffers grow with the rows; each weight
    streamed once a run, pair of heads by pair). hidden: MTP's rows (each row's final-normed
    hidden to m.hid, as qwen35_rows'). spill, depth, whole: as qwen3_tile's."""
    from ..kernels.lib import rmsnorm
    from . import qwen35 as Q35
    from .qwen35 import LIN, _deltanet_rows, run_layers
    spec = m.spec
    if not isinstance(pos, RunRows) or pos.lo < spec.conv_k - 1:
        raise CompileError("a tiled Qwen3.5 run is a prompt run from conv_k - 1")
    x, c, s_ = _inputs_rows(m, pos)
    res = Res(x, m.xr[0:T, :] if spill else None)
    del x
    parts, sub = _widths(m, T)
    dr = drows or T

    def layer(li, it, jt=None):
        lw = m.layer(li, it, jt)
        x = res.x
        if lw.kind == LIN:
            for r0, n in _pieces(T, dr):
                xr = x[r0:r0 + n, :]
                xr.set(_deltanet_rows(xr, lw, pos.lo, spec, m.gr[0:n, :], m.on[0:n, :],
                                      xs=_norm_quantize(xr, lw.g_in, spec.eps)))
                del xr
            del x
        else:
            del x
            _attention_tile(res, lw, c, s_, pos, T, spec, block, parts, gated=True,
                            depth=depth, whole=whole, oscr=_oscr(m, oscr))
        _mlp_tile(res.x, lw, spec, parts, sub(lw))

    run_layers(m.plan, layer, m.subs if Q35.NEST else None)   # (the DeltaNet layers' loop)
    x = res.x
    if hidden:
        g = ol.load(m.g_final)
        for r in range(T):
            ol.store(m.hid[r:r + 1, :], rmsnorm(x[r:r + 1, :], g, spec.eps))
        del g
    _lm_head_rows(x, m, spec, logit_rows)


@ol.jit
def lfm2_tile(m, pos, T: int, logit_rows, block: int = ATTN_BLOCK, spill: bool = False,
              depth: int = ATTN_DEPTH, whole: bool = True, oscr: bool = True, loop: int = 2):
    """lfm2_rows tiled: T rows of a prompt run (pos a RunRows), its attention layers, its
    convolution blocks and its MLPs tiled. spill, depth, whole: as qwen3_tile's. A stack the
    image lays out for it (m.stack) runs in one loop over its layers (_lfm2_stack, STACK);
    else the runs of its plan (lfm2.run_layers)."""
    from .lfm2 import CONV, run_layers
    spec = m.spec
    if not isinstance(pos, RunRows):
        raise CompileError("a tiled LFM2 run is a prompt run")
    x, c, s_ = _inputs_rows(m, pos)
    res = Res(x, m.xr[0:T, :] if spill else None)
    del x
    parts, sub = _widths(m, T)
    D, W = ol.block_size(), ol.tmem_words()

    def mixer(lw):
        if lw.kind == CONV:
            n = lw.taps.shape[1]
            _conv_tile(res.x, lw, pos, spec, parts, conv_parts(n, spec.conv_k, D, T, W))
        else:
            _attention_tile(res, lw, c, s_, pos, T, spec, block, parts, depth=depth,
                            whole=whole, oscr=_oscr(m, oscr), loop=loop)

    def mlp(lw):
        _mlp_tile(res.x, lw, spec, parts, sub(lw))

    def layer(li, it):
        lw = m.layer(li, it)
        mixer(lw)
        mlp(lw)

    if STACK and getattr(m, "stack", None) is not None:
        _lfm2_stack(m, mixer, mlp)
    else:
        run_layers(m.plan, layer)
    _lm_head_rows(res.x, m, spec, logit_rows)


STACK = True        # LFM2's tiled runs: a stack laid out for it in one loop (_lfm2_stack)


def _lfm2_stack(m, mixer, mlp) -> None:
    """LFM2's layers in one hardware loop, whatever its kinds' pattern (docs/tiled_prefill.md
    12.3): a layer's kind from a TMEM table (a word per layer and kind, 1.0 where the layer is of
    that kind: FILLs at the start), RLD (bounded: 0 or 1) into the counts of loops of 0 or 1
    iterations (a count register, its word read before each loop and free in its body: the
    attention's loops take the address registers the counts would hold): the kind's mixer,
    then after the MLP (the blocks' common part, at the same offsets in both kinds) the step of
    the layer's block Counter (compiler.Counter: the blocks before it, lfm2.Image.stack) by the
    kind's block size. One copy of each kind's code, where the plan's runs unroll every layer
    of an irregular stack (LFM2-2.6B's 30: 1 or 2 rows a run in the card's IMEM)."""
    import itertools
    from .lfm2 import ATTN, CONV
    b = current()
    kinds = list(m.spec.kinds)
    L, st = len(kinds), m.stack
    tab = ol.zeros((2 * L,))                    # [conv layer? | attention layer?]
    i = 0
    for kd, run in itertools.groupby(kinds):    # a FILL a run of layers of a kind
        n = len(list(run))
        o = i + (0 if kd == CONV else L)
        tab[o:o + n].set(1.0)
        i += n
    first = {kd: kinds.index(kd) for kd in (CONV, ATTN)}
    blk = b.counter("layer block")

    def kind(it, kd):
        """Begin layer it's loop of kind kd: its word of the table into a count register, free
        again in the loop's body (LOOP reads it as it starts)."""
        b.check_live(tab)
        r = b.scratch()
        ra, imm = b.addr(Affine(tab.base + (0 if kd == CONV else L)) + Affine.of(it))
        b.emit(I.rld(r, imm, ra=ra, bound=(0, 1), comment=f"a {kd} layer?"))
        lp = b.begin_loop(0, rcount=r)
        b.unscratch(r)
        return lp

    for it in ol.range(L):
        for kd in (CONV, ATTN):
            lp = kind(it, kd)
            mixer(m.layer(first[kd], at=blk))
            b.end_loop(lp)
        mlp(m.layer(first[CONV], at=blk))      # (the common part: the same offsets)
        for kd in (CONV, ATTN):
            lp = kind(it, kd)
            b.step(blk, st.size[kd])
            b.end_loop(lp)
    b.end_counter(blk)
    tab = None                                  # (free: kind's closure holds the name)


def pack_seq(seqs, G: int, block: int, cap: int):
    """One packed tile's attention span from its rows' own (at run-time positions: Bucket with
    an Additive tile, or Blocks whose masks are Additive tiles; the same blocks for every row
    of a run, the masks their own): every masked block's RowTiles, row group i (G query rows)
    taking row i's tile. One row: its own span."""
    from ..kernels.attention import Additive, Blocks, Bucket
    s0 = seqs[0]
    if len(seqs) == 1:
        return s0
    if isinstance(s0, Bucket):
        if any(s.add is None for s in seqs):
            raise CompileError("a packed tile's masked block needs the mask tiles (Additive)")
        t0 = (s0.blocks - 1) * block
        return TileSeq(s0.blocks - 1, [(t0, min(block, cap - t0), RowTiles(
            [(i * G, (i + 1) * G, s.add) for i, s in enumerate(seqs)]))])
    if isinstance(s0, Blocks):
        items = []
        for k, (t0, n, m0) in enumerate(s0.items):
            ms = [s.items[k][2] for s in seqs]
            if m0 is None:
                items.append((t0, n, None))
                continue
            if not all(isinstance(mk, Additive) for mk in ms):
                raise CompileError("a packed tile's masked blocks need the mask tiles")
            items.append((t0, n, RowTiles([(i * G, (i + 1) * G, mk)
                                           for i, mk in enumerate(ms)])))
        return Blocks(items)
    raise CompileError(f"no packed span of {type(s0).__name__}")


def _rope_rows_g(x, c, s, out):
    """Gemma 4's _rope with one position per row (c, s: [T, nr]): the pairs (i, i + d / 2), i
    < nr, rotated, the others passing through; into `out`. The same products as _rope's."""
    from ..kernels.lib import rope_rows
    d, nr = x.cols, c.cols
    h = d // 2
    if nr == h:
        return rope_rows(x, c, s, out)
    a, b = x[:, 0:nr], x[:, h:h + nr]
    out[:, 0:nr].set(a * c - b * s)
    out[:, h:h + nr].set(b * c + a * s)
    out[:, nr:h].set(x[:, nr:h])
    out[:, h + nr:d].set(x[:, h + nr:d])
    return out




def _gemma_attention_tile(res: Res, lw, m, pos: RunRows, ropes, T: int, block: int,
                          parts: int, depth: int = ATTN_DEPTH, whole: bool = True,
                          oscr=None, loop: int = 0):
    """gemma4._attention tiled for T rows of a prompt run: x += norm(W_o attention(norm(x))).
    Per KV head its K / V (own layers: projected for the T rows, normed, K RoPE'd, appended a
    row at a time to its slot: the sliding ring's or the cache's), then its query group in
    parts of MCOLS heads, each part's [T, heads d] projected, normed and RoPE'd in place and
    its rows attending in packed tiles (P = MCOLS / G rows; a row's span is its own: the
    sliding window's Blocks or the global Bucket, their masks the mask tiles), the outputs
    written over the part's queries, then (whole: _attention's one QACT) stored to `oscr` as
    the part ends and loaded back whole into W_o's operand (without oscr gathered in a [T, n_q
    d] tile), or quantized as the part ends (a QACT range each, as _attention_tile's); then W_o
    by output parts into [T, H] and the post-norm. Each row's arithmetic is _attention's. x:
    res.x, spilled while the heads attend (Res). loop: the heads' and tiles' hardware loops
    (_gemma_attention_loop) at this level where they apply (_attention_tile's)."""
    from ..kernels.lib import rmsnorm
    from .gemma4 import SLIDE, _RowPos, _add_norm, _full_seq, _slide_seq, _slot
    spec = m.spec
    d, nkv, eps = lw.hd, lw.nkv, spec.eps
    G = spec.n_q // nkv
    c0, nr = (0, spec.head_dim // 2) if lw.kind == SLIDE else (spec.head_dim, spec.global_rot)
    c, s_ = ropes[:, c0:c0 + nr], ropes[:, c0 + nr:c0 + 2 * nr]
    kv = lw.kv
    rows = [_RowPos(pos, r) for r in range(T)]
    span = _slide_seq if lw.kind == SLIDE else _full_seq
    rseq = [span(m, p, block) for p in rows]
    xs = _norm_quantize(res.x, lw.g_in, eps)
    qn = ol.load(lw.qn)
    kn = ol.load(lw.kn) if lw.own else None
    res.spill()                                # (after the tiles that outlive the heads)
    mc = ol.mxu_columns()
    gs = min(G, mc)
    while gs % 2 == 0 and T * gs * d > ol.tmem_words() // QPART:
        gs //= 2                               # a part's [T, gs d] queries in an eighth of TMEM
    P = packing(mc, gs)                        # rows of gs heads a tile: K / V streamed G / mc
    #                                            times a row whatever gs (P gs = MCOLS)
    tiles = [(r0, min(P, T - r0)) for r0 in range(0, T, P)]
    gparts = [(g0, min(G, g0 + gs)) for g0 in range(0, G, gs)]
    if whole and oscr is None:                 # the outputs, quantized once (W_o's operand)
        o_all, ost = ol.empty([T, spec.n_q * d]), None
    else:                                      # W_o's operand, a part's columns at a time
        o_all, ost = None, None if whole else ol.stationary(T, spec.n_q * d)
    if oscr is not None:                       # (the scratch as [T, n_q d]: one load back)
        oscr = Tensor(oscr.base, (T, spec.n_q * d), (spec.n_q * d, 1))
    heads = list(range(nkv))

    def tspan(t, n=P):                         # tile t's span (its n rows t P ..: t a loop's
        return _span_blocks(pack_seq([span(m, _RowPos(pos, Affine.of(t) * P + i), block)
                                      for i in range(n)], gs, block, kv.cap), block, kv.cap)

    looped = LOOP and loop and whole and oscr is not None and _loopable(
        pos, T, P, G, gs, kv, heads, block, loop, True, tspan)
    if looped:                                 # (the outputs to the scratch, as below)
        if lw.own:                             # every head's K / V rows at their slots
            for j in heads:
                k = ol.dot(xs, lw.wk[j * d:(j + 1) * d, :])
                v = k if lw.same else ol.dot(xs, lw.wv[j * d:(j + 1) * d, :])
                kr = _rope_rows_g(rmsnorm(k, kn, eps), c, s_, ol.empty([T, d]))
                vr = rmsnorm(v, None, eps)
                for r in range(T):
                    ol.kv_append(kv, j, _slot(m, lw.kind, rows[r]), kr[r:r + 1, :],
                                 vr[r:r + 1, :])
                del k, v, kr, vr

        def lquery(hq, Qv):                    # heads hq .. hq + gs, normed and RoPE'd in place
            q = Qv[0:T, :]
            ol.dot(xs, lw.wq[hq * d:hq * d + gs * d, :], out=q)
            for g in range(gs):
                qg = q[:, g * d:(g + 1) * d]
                _rope_rows_g(rmsnorm(qg, qn, eps), c, s_, qg)
                del qg
            del q

        _heads_loop(kv, heads, T, G, gs, P, d, d, block, depth, ol.LOG2E, oscr, loop >= 2,
                    lquery, tspan)
    made, ent, qhs, kvh, seqs = {}, [], [], [], []

    def part(j, g0, g1):
        """KV head j's K / V appends (with its first part), the queries of heads g0 .. g1 of
        its group: [T, (g1 - g0) d], read as [T (g1 - g0), d] (row (r, g))."""
        if g0 == 0 and lw.own:
            k = ol.dot(xs, lw.wk[j * d:(j + 1) * d, :])
            v = k if lw.same else ol.dot(xs, lw.wv[j * d:(j + 1) * d, :])
            kr = _rope_rows_g(rmsnorm(k, kn, eps), c, s_, ol.empty([T, d]))
            vr = rmsnorm(v, None, eps)
            for r in range(T):
                ol.kv_append(kv, j, _slot(m, lw.kind, rows[r]), kr[r:r + 1, :], vr[r:r + 1, :])
            del k, v, kr, vr
        h0, h1 = j * G + g0, j * G + g1
        q = ol.empty([T, (g1 - g0) * d], dense=True)          # (rows of heads: reshaped)
        ol.dot(xs, lw.wq[h0 * d:h1 * d, :], out=q)
        for g in range(g1 - g0):
            qg = q[:, g * d:(g + 1) * d]
            _rope_rows_g(rmsnorm(qg, qn, eps), c, s_, qg)
        return q

    def query(j, g0, g1, r0, n):
        def emit():
            if (j, g0) not in made:
                made[(j, g0)] = part(j, g0, g1)
            w = g1 - g0
            return made[(j, g0)].reshape(T * w, d)[r0 * w:(r0 + n) * w, :]
        return emit

    for j in range(nkv):
        for g0, g1 in gparts:
            for r0, n in tiles:
                ent.append((j, g0, g1, r0, n))
                qhs.append(query(j, g0, g1, r0, n))
                kvh.append(j)
                seqs.append(pack_seq(rseq[r0:r0 + n], g1 - g0, block, kv.cap))

    def emit(i, acc, l):
        j, g0, g1, r0, n = ent[i]
        w = g1 - g0
        if o_all is not None:                  # the part's columns of the outputs
            h0 = (j * G + g0) * d
            for a in range(n):
                o_all[r0 + a, h0:h0 + w * d].reshape(w, d).set(
                    acc[a * w:(a + 1) * w, :] / l[a * w:(a + 1) * w][:, None])
            if r0 + n == T:
                del made[(j, g0)]              # (its queries)
            return
        o = made[(j, g0)]                      # the outputs over the part's queries
        o.reshape(T * w, d)[r0 * w:(r0 + n) * w, :].set(acc / l[:, None])
        if r0 + n == T:                        # to the scratch, or quantized into W_o's
            h0 = (j * G + g0) * d              # operand
            if whole:
                ol.store(oscr[0:T, h0:h0 + w * d], made.pop((j, g0)))
            else:
                ol.quantize_into(ost, made.pop((j, g0)), h0)
        del o

    if not looped:
        _attend_heads(qhs, kv, kvh, seqs, block, ol.LOG2E, depth=depth, emit=emit)
    xs = qn = kn = None                        # (the norm gains: _attention_tile's)
    if whole:                                  # (the scratch's rows back: the stores before)
        if o_all is None:
            o_all = ol.load(oscr[0:T, 0:spec.n_q * d])
        ost = ol.quantize(o_all, free=True)
        del o_all
    x = res.back()
    y = ol.empty([T, spec.hidden])
    _add_parts(y, [ost], [lw.wo], parts, y)
    ost = None
    _add_norm(x, y, ol.load(lw.g_attn), eps)


@ol.jit
def gemma4_tile(m, pos, T: int, logit_rows, block: int = ATTN_BLOCK, spill: bool = False,
                depth: int = ATTN_DEPTH, whole: bool = True, oscr: bool = True, loop: int = 2):
    """gemma4_step's rows tiled: T rows of a prompt run (pos a RunRows with the image's mask
    tiles), their tokens gathered from out[] and per-layer inputs as gemma4_step's; per layer
    the tiled attention, the dense MLP (GeGLU) tiled into [T, H] for its post-norm, the
    per-layer input block as gemma4._ple. spill: x in the I/O area's input rows while each
    layer's heads attend and W_o runs (Res). depth, whole: as qwen3_tile's."""
    from ..kernels.lib import gelu_tanh
    from .gemma4 import _add_norm, _gathered_out, _ple, _ple_inputs, _pos_rows
    from .qwen3 import fill_logits
    fill_logits(m)
    spec = m.spec
    if not isinstance(pos, RunRows) or pos.amask is None or spec.experts:
        raise CompileError("a tiled Gemma 4 run is a dense model's prompt run with mask tiles")
    ropes = _pos_rows(m, pos, T, masks=False)   # (first: x's gather leaves the next-fit
    x = _gathered_out(m, pos)                   # cursor past its temporaries, TMEM split)
    x.set(x * math.sqrt(spec.hidden))           # (in place: one [T, H] tile)
    if spec.ple_dim:
        _ple_inputs(m, x, None)
    res = Res(x, m.x[0:T, :] if spill else None)
    del x
    parts, sub = _widths(m, T)

    def layer(li, it=None, jt=None):
        lw = m.layer(li, it, jt)
        _gemma_attention_tile(res, lw, m, pos, ropes, T, block, parts, depth, whole,
                              _oscr(m, oscr), loop)
        x = res.x
        y = ol.empty([T, spec.hidden])
        _mlp_tile(x, lw, spec, parts, sub(lw), act=gelu_tanh, g=lw.g_pre, out=y)
        _add_norm(x, y, ol.load(lw.g_ffn), spec.eps, None if spec.ple_dim else ol.load(lw.ls))
        del y
        if spec.ple_dim:
            _ple(x, lw, spec)

    def unit(first, it=None):
        for e0, su, r, nb in m.subs[first]:
            if r == 1:
                for e_ in range(len(su)):
                    layer(first + e0 + e_, it)
                continue
            for jt in ol.range(r):
                for e_ in range(len(su)):
                    layer(first + e0 + e_, it, (jt, len(su), nb))

    for first, _, reps in m.runs:
        if reps == 1:
            unit(first)
            continue
        for it in ol.range(reps):
            unit(first, it)
    if logit_rows:
        _lm_head_rows(res.x, m, spec, list(logit_rows))


EM_GROUP = 2    # rows a group of an expert-major tiled run's per-row record work ([R, H] tiles)


@ol.jit
def gemma4_layer_tile(m, li: int, pos, row, T: int, block: int = ATTN_BLOCK,
                      spill: bool = False, depth: int = ATTN_DEPTH, whole: bool = True,
                      oscr: bool = True):
    """gemma4_layer_run's expert-major run tiled (a MoE model's layer-major prefill,
    docs/offload.md 13.11): the T rows of the chunk from `row` (a run argument; their records'
    X, moe.em_x) at the run-time positions pos .. pos + T - 1 (a RunRows with the image's mask
    tiles, in one attention block) through layer li. The layer before ends first
    (gemma4._em_end, EM_GROUP rows at a time); the attention tiled (_gemma_attention_tile;
    spill: x in the records' X while the heads attend); x to X; the route and the need lines
    (moe.moe_prologue_rows over groups of rows, LINE / 4 ids a line); then the dense MLP tiled
    over the T rows (its weights streamed once), normed (norm_1), to the records' SH. Each
    row's arithmetic is gemma4_layer_run's, so the chunk's records, KV cache and expert runs
    are those of its runs of fewer rows, bit for bit where their 4-bit MMs pair alike."""
    from ..kernels.lib import gelu_tanh, rmsnorm
    from .gemma4 import _em_end, _pos_rows
    from .moe import LINE, em_row, em_x, moe_prologue_rows
    spec, dev = m.spec, m.moe_dev
    if not isinstance(pos, RunRows) or pos.amask is None or not spec.experts or spec.ple_dim:
        raise CompileError("a tiled Gemma 4 layer run is a MoE model's, at a run-time position "
                           "with the image's mask tiles")
    H, k = spec.hidden, spec.top_k
    X = em_x(dev, row, T, H)
    x = ol.load(X)
    ropes = _pos_rows(m, pos, T, masks=False)
    groups = [(r0, min(EM_GROUP, T - r0)) for r0 in range(0, T, EM_GROUP)]
    if li > 0:                                  # (the layer before: its experts' outputs)
        for r0, n in groups:
            _em_end(x[r0:r0 + n, :], m.layer(li - 1), spec, m, row + r0)
    lw = m.layer(li)
    res = Res(x, X if spill else None)
    del x
    parts, sub = _widths(m, T)
    _gemma_attention_tile(res, lw, m, pos, ropes, T, block, parts, depth, whole, _oscr(m, oscr))
    x = res.x
    ol.store(X, x)
    g = max(1, (LINE // 4) // k)                # rows a need line
    for r0 in range(0, T, g):
        moe_prologue_rows(x[r0:r0 + min(g, T - r0), :], lw, spec.moe, dev, spec.eps, row + r0)
    xs = _norm_quantize(x, lw.g_pre, spec.eps)  # the dense MLP: the rows normed (g_pre)
    res.x = x = None
    y = ol.empty([T, H])
    _mlp_tile(None, lw, spec, parts, sub(lw), act=gelu_tanh, out=y, xs=xs)
    xs = None
    gf = ol.load(lw.g_f1)
    for r0, n in groups:                        # SH: norm_1 of the dense MLP's output
        ol.store(Tensor(em_row(dev, row, r0) + 4 * H * (k + 1), (n, H), (dev.em_rec // 4, 1)),
                 rmsnorm(y[r0:r0 + n, :], gf, spec.eps))


@ol.jit
def qwen35_layer_tile(m, li: int, pos, row, T: int, block: int = ATTN_BLOCK,
                      spill: bool = False, drows: int = 0, depth: int = ATTN_DEPTH,
                      whole: bool = True, oscr: bool = True, part: str | None = None):
    """qwen35_layer_run's expert-major run tiled (Qwen3.5-MoE's layer-major prefill,
    docs/offload.md 13.11): the T rows of the chunk from `row` (their records' X) at the
    run-time positions pos .. pos + T - 1 (a RunRows with the image's mask tiles) through layer
    li. The MoE layer before ends first (moe.moe_combine_rows, EM_GROUP rows at a time); a
    DeltaNet layer's rows in runs of `drows` (default T; at most the image's gr / on rows),
    an attention layer tiled (_attention_tile, gated; spill: x in the records' X while the
    heads attend); x to X; a MoE layer's route and shared expert over the T rows, its need
    lines between the shared expert's chunks (moe.moe_prologue_rows), a dense layer's MLP
    tiled. Each row's arithmetic is qwen35_layer_run's.

    part (a DeltaNet layer by chunks, Config.DNCHUNK: DN_PARTS): "dn-prep" the combine, x to
    X and the rows' prep to the image's dnq (qwen35._deltanet_prep); "dn-core" the recurrence
    over the rows, dnq to dno (_deltanet_core_rows); "dn-post" from X and dno the rest of the
    layer (_deltanet_post, then as above)."""
    from .moe import combine_stage, em_x, moe_combine_rows, moe_prologue_rows
    from .qwen35 import LIN, _deltanet_core_rows, _deltanet_post, _deltanet_prep, _deltanet_rows
    spec, dev = m.spec, m.moe_dev
    if not isinstance(pos, RunRows) or pos.amask is None or spec.moe is None:
        raise CompileError("a tiled Qwen3.5 layer run is a MoE model's, at a run-time "
                           "position with the image's mask tiles")
    H, mo, K = spec.hidden, spec.moe, spec.conv_k
    if part is not None and (part not in DN_PARTS or m.layer(li).kind != LIN or m.dnq is None):
        raise CompileError(f"part {part!r} of layer {li}: a DeltaNet layer's {DN_PARTS} on an "
                           "image with the chunked DeltaNet's regions (Config.DNCHUNK)")
    if part == "dn-core":
        _deltanet_core_rows(m.layer(li), spec, m.dnq, m.dno, row, T)
        return
    X = em_x(dev, row, T, H)
    late = part == "dn-post" and ol.current().unroll            # (merged, stage.py: x's words
    x = None if late else ol.load(X)                            # only where the residual is)
    if part != "dn-post" and li > 0 and m.layer(li - 1).moe:    # (the layer before: its
        st = combine_stage(mo, H)                               # experts' outputs)
        for r0 in range(0, T, EM_GROUP):
            xg = x[r0:min(T, r0 + EM_GROUP), :]
            xg.set(moe_combine_rows(xg, mo, dev, row + r0, stage=st))
            del xg
        del st
    lw = m.layer(li)
    parts, sub = _widths(m, T)
    if part == "dn-prep":
        xs = _norm_quantize(x, lw.g_in, spec.eps)
        if ol.current().unroll:                 # (merged, stage.py: x's words free from here)
            if li > 0 and m.layer(li - 1).moe:
                ol.store(X, x)
            ol.current().free_now(x)
            _deltanet_prep(None, lw, spec, m.dnq, row, xs, T)
            return
        _deltanet_prep(x, lw, spec, m.dnq, row, xs)
        if li > 0 and m.layer(li - 1).moe:
            ol.store(X, x)
        return
    if part == "dn-post":
        if late:
            x = _deltanet_post(None, lw, spec, m.dnq, m.dno, row, T=T, X=X)
        else:
            x.set(_deltanet_post(x, lw, spec, m.dnq, m.dno, row))
    elif lw.kind == LIN:
        for r0, n in _pieces(T, drows or T):
            xr = x[r0:r0 + n, :]
            xr.set(_deltanet_rows(xr, lw, K - 1, spec, m.gr[0:n, :], m.on[0:n, :],
                                  xs=_norm_quantize(xr, lw.g_in, spec.eps)))
            del xr
    else:
        c = ol.load(m.cos_t[pos.pos:pos.pos + T, :])
        s_ = ol.load(m.sin_t[pos.pos:pos.pos + T, :])
        res = Res(x, X if spill else None)
        del x
        _attention_tile(res, lw, c, s_, pos, T, spec, block, parts, gated=True, depth=depth,
                        whole=whole, oscr=_oscr(m, oscr))
        x = res.x
        del c, s_
    if not lw.moe:
        _mlp_tile(x, lw, spec, parts, sub(lw))
        ol.store(X, x)
        return
    ol.store(X, x)
    moe_prologue_rows(x, lw, mo, dev, spec.eps, row)


@ol.jit
def tile_run(m, kernel, pos, T: int, kind: str, block: int = ATTN_BLOCK, kw=None):
    """A tiled prompt run (docs/prefill.md): `kernel`'s T rows at the run-time position in the
    generate state's tpos word (pos: a RunRows, toks_at 0), their tokens from out[]; kind "P",
    or "L" (the prompt's last run: its last row's logits). kw: the kernel's own (hidden, spill,
    drows)."""
    with RunWords(m, pos):
        kernel.fn(m, pos, T, [T - 1] if kind == "L" else [], block, **(kw or {}))


KERNELS = {"qwen3": qwen3_tile, "qwen35": qwen35_tile, "lfm2": lfm2_tile,
           "gemma4": gemma4_tile}


def family(img) -> str | None:
    """The image's model family (its module's name) when a tile kernel serves it, else None."""
    name = type(img).__module__.rsplit(".", 1)[-1]
    return name if name in KERNELS else None


def check(img, T: int) -> None:
    """T rows fit the image and the configuration: the I/O area's rows (the logits, Qwen3.5's
    gr / on rows), ACT_ROWS (one stationary of T rows), one slice."""
    cfg = img.cfg
    if T > img.rows:
        raise ValueError(f"{T} rows, the image's I/O area holds {img.rows}")
    if T > cfg.act_rows:
        raise ValueError(f"{T} rows: one MM over them needs ACT_ROWS >= {T} ({cfg.act_rows})")
    if cfg.S != 1:
        raise ValueError("tiled prefill: one slice")


def compile_tile(img, p0: int, T: int, block: int = ATTN_BLOCK, logit_rows=()) -> list:
    """One program per slice: qwen3_tile over the image's I/O rows (img.rows >= T), every MM
    over T rows (the configuration's ACT_ROWS >= T)."""
    check(img, T)
    return [trace_fit(qwen3_tile, img.cfg, s, lambda s=s: {
        "m": img.descriptors(s), "p0": p0, "T": T, "logit_rows": list(logit_rows),
        "block": block}).finish() for s in range(img.cfg.S)]


def compile_tile_run(img, blocks: int, T: int, kind: str, block: int = ATTN_BLOCK,
                     hidden: bool = False, slot: int = 0):
    """tile_run's (programs, run_args): T rows of a prompt at a run-time position of bucket
    `blocks` (from conv_k - 1), as the image's compile_prompt_run's P and L runs. The first
    of the run's forms (variants) that fits TMEM and ACT RAM, its tiles placed next-fit, then
    best-fit (_first_fit), at the attention's pipeline depth (ATTN_DEPTH), or where that is
    over IMEM at SHALLOW (its program grows with the blocks a head's entries leave unrolled,
    up to ~3 x depth a packed tile: past bucket ~3 the bigger tiles); over IMEM at both, the
    first (the caller's IMEM check)."""
    fam = family(img)
    if fam is None:
        raise ValueError(f"no tiled prompt run for {type(img).__module__}")
    if not img.lookup:
        raise ValueError("a prompt run needs lookup tables (lookup=True)")
    if kind not in ("P", "L"):
        raise ValueError(f"tiled prompt run kind {kind!r}")
    spec = img.spec
    if getattr(spec, "moe", None) is not None or getattr(spec, "experts", 0):
        raise ValueError("a MoE model's prompt runs layer by layer")
    if hidden and fam != "qwen35":
        raise ValueError("hidden rows: Qwen3.5's MTP")
    check(img, T)
    lo = max((blocks - 1) * block, getattr(spec, "conv_k", 1) - 1)
    if fam == "gemma4":             # (Gemma 4: its own mask tiles' address and step)
        lk = img.lookup
        if "amask" not in lk:
            raise ValueError("a tiled Gemma 4 run needs the image's mask tiles")
        pos = RunRows(blocks, block, lo, 0, img.cap, T, 0,
                      (lk["amask"], 4 * lk["amask_M"] * (block + 1)))
    else:
        pos = RunRows(blocks, block, lo, img.lookup["zmask"], img.cap, T, 0, _amask(img.lookup))
    desc = (lambda s: img.descriptors(s, slot)) if fam == "qwen35" else img.descriptors

    def trace(kw, fit):
        kw = dict(kw, hidden=True) if hidden else kw
        return [tile_run.trace(img.cfg, s, {
            "m": desc(s), "kernel": KERNELS[fam], "pos": pos, "T": T, "kind": kind,
            "block": block, "kw": kw}, fit=fit) for s in range(img.cfg.S)]

    return _first_fit(trace, variants(fam, T, img.cfg.MCOLS), img.cfg, levels=True)


def _first_fit(trace, forms, cfg, levels: bool = False):
    """(programs, run_args) of trace(form, fit) (its builders, one a slice) for the first of
    the forms that fits TMEM, ACT RAM and IMEM: each form at the attention's pipeline depth,
    then over IMEM at SHALLOW, then the next form, its tiles placed next-fit; then the same with
    best-fit placement (slower: best-fit reuses the words just freed, which work still running
    reads; Qwen3-0.6B's 16 rows in co-simulation, MX, the card's configuration: 94.9K cycles
    a row placed best-fit, 87.4K next-fit); over IMEM at every one, the first (the caller's
    IMEM check: prefill.programs); none fitting, the last TMEM / ACT RAM error (or, where no
    form applied, the range forms' QACT range error). levels: each form's attention at the
    loop levels of LOOPS in turn where the one before runs out of address registers (a form
    that names its level: that one)."""
    err = over = skip = None
    imem = cfg.IMEM_WORDS // 8
    for fit, v in [(fit, v) for fit in ("next", "best") for v in forms]:
        for lv in ([v] if not levels or "loop" in v else [dict(v, loop=x) for x in LOOPS]):
            short = False
            for f in [lv] + ([dict(lv, depth=SHALLOW)] if "depth" not in lv else []):
                try:
                    bs = trace(f, fit)
                except CompileError as e:
                    if "QACT range" in str(e):     # (a range form's part at an odd block of
                        skip = e                   # 4-bit weights: the form does not apply)
                        break
                    short = levels and "registers" in str(e)
                    if not short and not any(w in str(e) for w in ("TMEM", "ACT RAM")):
                        raise
                    err = e
                    break
                progs = [b.finish() for b in bs]
                if max(map(len, progs)) <= imem:
                    return progs, list(bs[0].run_args)
                if over is None:
                    over = progs, list(bs[0].run_args)
            if not short:                          # (the next level: on a register shortage)
                break
    if over is not None:
        return over
    raise err or skip


def variants(fam: str, T: int, mcols: int) -> list:
    """A tiled run's forms, tried in order until one fits TMEM and ACT RAM: as it is; x spilled
    while the heads attend (Res); Qwen3.5's DeltaNet layers in runs of fewer rows (whole passes
    of MCOLS, then fewer: their pair buffers grow with the rows). The attention's outputs
    whole (W_o's operand one QACT: today's bits), through the image's I/O scratch, then
    gathered in TMEM (wholes); a prompt run whose outputs do not fit takes fewer rows
    (prefill._probe_tile)."""
    out = [{}, {"spill": True}]
    if fam == "qwen35":
        d = T
        while d > 1:
            d = d - mcols if d > mcols else d - 1
            out.append({"spill": True, "drows": d})
    return wholes(out)


def wholes(forms: list) -> list:
    """The forms (the attention's outputs whole through the image's I/O scratch: a part
    stored as it ends, loaded back whole), then each with them gathered in TMEM (oscr False:
    no stores, the [T, n_q d] tile beside the heads' work)."""
    return forms + [dict(f, oscr=False) for f in forms]


def ranges(forms: list) -> list:
    """The forms, then those through the scratch with the outputs quantized a part at a time
    instead (whole False: a QACT range a part, W_o an MM a range; not today's bits)."""
    return forms + [dict(f, whole=False) for f in forms if f.get("oscr", True)]


LAYER_KERNELS = {"gemma4": gemma4_layer_tile, "qwen35": qwen35_layer_tile}
# a Qwen3.5 DeltaNet layer by chunks (Config.DNCHUNK): its tiles' parts, each over the chunk's
# rows before the next (qwen35_layer_tile's part; dn_parts)
DN_PARTS = ("dn-prep", "dn-core", "dn-post")
# the families whose tiled layer runs are on by default (prefill.layer_tile_flag: unless
# OTPU_TILE=0) and their rows a run (Engine.tile_rows; at most ACT_ROWS): the most rows whose
# forms are whole (today's bits) on every Qwen3.5-MoE layer kind and bucket of the card's
# configuration (docs/tiled_prefill.md 14); the others opt-in, of the engine's layer_major rows
LAYER_ROWS = {"qwen35": 8}


def layer_family(img) -> str | None:
    """The image's family when a tiled expert-major layer run serves it (LAYER_KERNELS, its
    class's module or a base class's: MiniCPM-V's Image is Qwen3.5's; a MoE image with prefill
    rows, the expert server's words (or a resident pool's records: em_ok) and the mask tiles,
    which a MoE model's image holds when OTPU_TILE=1 as it is built: prefill.layer_tile_flag),
    else None."""
    name = next((n for n in (t.__module__.rsplit(".", 1)[-1] for t in type(img).__mro__)
                 if n in LAYER_KERNELS), None)
    em = img.em_ok() if hasattr(img, "em_ok") else getattr(img, "offload", None) is not None
    ok = name is not None and getattr(img, "prefill_rows", 0) and em and \
        "amask" in (img.lookup or {})
    return name if ok else None


def compile_layer_tile(img, li: int, blocks: int, T: int, block: int = ATTN_BLOCK,
                       slot: int = 0, part: str | None = None):
    """(programs, run_args): layer li's expert-major run tiled (LAYER_KERNELS), T rows at a
    run-time position of bucket `blocks` (from conv_k - 1) and a run-time chunk row ("row"),
    in place of the image's compile_layer_run(li, blocks, R=T, em=True) (Engine.prefill_layers'
    runs of more than one row with tiling on: Engine.layer_tile); the first of its forms that
    fits TMEM, ACT RAM and IMEM (_first_fit). slot: the DeltaNet states' and windows' slot of an
    MTP image (its committed one, mtp.MTPDecoder: a chat's next turn), as compile_layer_run's.
    part: a DeltaNet layer's part by chunks (DN_PARTS, qwen35_layer_tile's)."""
    from ..compiler import RunVar
    fam = layer_family(img)
    if fam is None:
        raise ValueError(f"no tiled layer run for this image ({type(img).__module__}: an "
                         "expert-major MoE image with the mask tiles, OTPU_TILE=1)")
    cfg, lk = img.cfg, img.lookup
    tiles = 0
    if part and ":" in part:            # (dn_merged's run: that many tiles merged, stage.py)
        part, tiles = part.split(":")[0], int(part.split(":")[1])
    if not 0 <= li < img.spec.layers or not 1 <= T <= cfg.act_rows or \
            cfg.S != 1 and img.offload is not None:
        raise ValueError(f"layer {li}, {T} rows: a layer's run of at most ACT_ROWS "
                         f"({cfg.act_rows}) rows (S > 1: a resident pool's)")
    lo = max((blocks - 1) * block, getattr(img.spec, "conv_k", 1) - 1)
    amask = (lk["amask"], 4 * lk["amask_M"] * (block + 1))
    pos = RunRows(blocks, block, lo, lk.get("zmask", 0), img.cap, T, 0, amask)
    row = RunVar("row", img.prefill_rows)

    def trace(kw, fit):
        return [LAYER_KERNELS[fam].trace(cfg, s, {
            "m": img.descriptors(s, slot) if slot else img.descriptors(s), "li": li, "pos": pos,
            "row": row, "T": T, "block": block, **kw}, fit=fit) for s in range(cfg.S)]

    if tiles:
        from ..stage import trace_merged

        def streams(s):
            m = img.descriptors(s, slot) if slot else img.descriptors(s)
            return [lambda j=j: LAYER_KERNELS[fam].fn(m=m, li=li, pos=pos, row=row + j * T, T=T,
                                                      block=block, part=part)
                    for j in range(tiles)]
        err = None
        for fit, N in [(fit, N) for fit in ("next", "best")
                       for N in range(cfg.DNMERGE, 1, -1)]:
            try:
                bs = trace_merged(cfg, streams, N, fit)
            except CompileError as e:
                if not any(w in str(e) for w in ("TMEM", "ACT RAM", "registers")):
                    raise
                err = e
                continue
            return [b.finish() for b in bs], list(bs[0].run_args)
        raise err
    forms = [{"part": part}, {"part": part, "spill": True}] if part else \
        layer_variants(img, fam, T)
    return _first_fit(trace, forms, cfg)


def dn_parts(runs: list, core: tuple | None = None) -> list:
    """A DeltaNet layer's runs of a chunk (Engine.prefill_layers': (key, values, rows)) by
    chunks (Config.DNCHUNK): its tiled runs (keys ending "tile") as DN_PARTS' runs, every
    tile's prep, then every tile's core, then every tile's post; its untiled runs (the rows
    before conv_k - 1, layer 0's gathered rows) first, as they are. Runs that are not in that
    order come back unchanged (the recurrent tiles). core (dn_core's (C, whether it takes a
    start row)): the core tiles of the tiled rows from the first, up to the last tile's end a
    multiple of C rows on, as one run of the chunked core, (("dnc", layer), {"row": the first
    row, "rows": n}), the rest (a chunk's tail) as their core tiles; a first row past 0 only
    when the core takes it (else every core tile)."""
    tiled = [r[0][-1] == "tile" for r in runs]
    if not any(tiled) or tiled.index(True) != len(tiled) - sum(tiled):
        return runs
    k = tiled.index(True)
    part = {p: [(key[:-1] + (p,), vals, rows) for key, vals, rows in runs[k:]] for p in DN_PARTS}
    if core is not None and (runs[k][1]["row"] == 0 or core[1]):
        r0, j = runs[k][1]["row"], 0
        for n, (key, vals, _) in enumerate(runs[k:]):
            if (vals["row"] + key[2] - r0) % core[0] == 0:
                j, e = n + 1, vals["row"] + key[2]
        if j:
            part["dn-core"][:j] = [(("dnc", runs[k][0][0]), {"row": r0, "rows": e - r0}, None)]
    return runs[:k] + [r for p in DN_PARTS for r in part[p]]


STAGE_TILES = 8                     # a merged run's tiles (dn_merged)


def dn_merged(runs: list, N: int) -> list:
    """dn_parts' runs with each STAGE_TILES tiles in a row of dn-prep, and of dn-post, one merged
    run at the first's values (stage.py, Config.DNMERGE; N the streams at a time; 0: unchanged):
    the key's part "dn-prep:8", the others' rows the first's row + j T."""
    if N < 2:
        return runs
    out, i = [], 0
    while i < len(runs):
        key, vals, rows = runs[i]
        grp = runs[i:i + STAGE_TILES]
        if key[-1] in ("dn-prep", "dn-post") and len(grp) == STAGE_TILES and \
                all(k[-1] == key[-1] and k[2] == key[2] and v["row"] == vals["row"] + j * key[2]
                    for j, (k, v, _) in enumerate(grp)):
            out.append((key[:-1] + (f"{key[-1]}:{STAGE_TILES}",), vals, None))
            i += STAGE_TILES
        else:
            out.append(runs[i])
            i += 1
    return out


def core_end(core: tuple | None, r0: int, rows: int) -> int | None:
    """The row a chunk's chunked-core run ends at (dn_parts: from the first tiled row r0 in
    whole chunks of C, of a chunk of `rows` rows), where the tiles must have an end for the core
    to take every whole chunk (a prompt's first chunk tiles from r0 = conv_k - 1, so its tiles'
    ends at the attention blocks' multiples are never r0 + a multiple of C); None: no core."""
    if core is None or (r0 and not core[1]):
        return None
    e = r0 + (rows - r0) // core[0] * core[0]
    return e if e > r0 else None


def dn_core(img) -> tuple | None:
    """The image's chunked DeltaNet core (Image.compile_dn_run, where its
    dn_core_ok): (its chunk's rows, Config.DNCHUNK; whether it takes a start row, the run
    argument "row"), or None (the recurrence's dn-core tiles)."""
    import inspect
    import sys
    ok = getattr(img, "dn_core_ok", None)
    if not getattr(img.cfg, "DNCHUNK", 0) or ok is None or not ok():
        return None
    fn = sys.modules[type(img).__module__].qwen35_dn_run.fn
    return img.cfg.DNCHUNK, "row" in inspect.signature(fn).parameters


def untiled_rows(img) -> int:
    """The most rows of an untiled layer run (the image's compile_layer_run: its family's
    RUN_ROWS, and R k <= LINE / 4 ids in its request): layer 0's runs that gather their rows
    from the host's slot (embed_host) stay untiled."""
    import sys
    from .moe import LINE
    spec = img.spec
    k = spec.moe.k if getattr(spec, "moe", None) is not None else getattr(spec, "top_k", 1)
    return max(1, min(sys.modules[type(img).__module__].RUN_ROWS, (LINE // 4) // max(1, k)))


def layer_variants(img, fam: str, T: int) -> list:
    """A tiled layer run's forms (variants'): Qwen3.5's DeltaNet rows in runs of at most the
    image's gr / on rows (dnrows) and MCOLS (more only replay the MXU's groups), then fewer; the
    run's rows are the engine's (layer_major),
    so then the attention's outputs a part at a time (ranges: the 26B's global layers'
    [8, 16 x 512] outputs alone are TMEM's 64K words)."""
    if fam != "qwen35":
        return ranges(wholes([{}, {"spill": True}]))
    d = min(T, img.dnrows, img.cfg.MCOLS)
    out = [{"drows": d}, {"spill": True, "drows": d}]
    while d > 1:
        d = d - img.cfg.MCOLS if d > img.cfg.MCOLS else d - 1
        out.append({"spill": True, "drows": d})
    return ranges(wholes(out))
