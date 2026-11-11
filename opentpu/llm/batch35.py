"""Batched decode of a Qwen3.5 / Qwen3.5-MoE image (Image(batch=N); the 35B's design:
docs/batch35.md): one run feeds a token to each of B sequences, each in its
own slot (its KV caches, DeltaNet states and convolution windows), every weight streamed once
for all of them.

A step's rows are (slot, position), one per sequence, every position past conv_k - 2 (the
window full: the prompt fed). Per layer:

  DeltaNet   the projections over the B rows (one weight pass); per row the convolution over
             its slot's window, which it stores back shifted by one row, and a DSTEP on its
             slot's states (_deltanet_slots)
  attention  qwen3's rows kernel over the rows' own caches (the layer's kv_slots)
  MLP / MoE  the dense MLP over the B rows; a MoE layer's moe_ffn_rows over groups of rows
             (as many as one request row names: 2 of the 35B's top-8; each expert of a
             group's union once, each row moe_ffn's bit for bit), at S > 1 moe_ffn a row
             (tensor-parallel experts, the layer runs' way) (_moe_rows)

then the LM head over the rows: the rows' logits (qwen3._lm_head_rows), or with a batch area
(opentpu/llm/batch.py: Engine(batch_area=True)) each row's pick on the card, greedy or with its
slot's sampler (batch._sinks), its token read from the area's tin and its pick written to tout:
the host writes the rows' tokens and reads one word a row, as the dense models' batched steps.
With a span (batch.Steps') one program serves its rows for span steps at positions + t, t the
area's step word (batch.step_rows: the attention rows' run-time positions; the DeltaNet's
slots do not depend on the position). Per row every value is the decode step's (qwen35_step
at that row's slot, Image.compile_step(slot=)), bit for bit.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from ..compiler import CompileError, Tensor, current, gall_wg, mul_bound
from ..kernels.attention import SPAN_P
from ..kernels.deltanet import gates
from ..kernels.lib import rmsnorm, softplus
from ..host.offload import LINE
from . import moe as MO
from .lfm2 import run_layers
from . import batch as BA
from .qwen3 import (ATTN_BLOCK, _attention_rows, _embed_word, _inputs_rows, _lm_head_rows,
                    _tokens_arg)
from .qwen35 import (LIN, _dn_eps, _dn_gain, _dn_gz, _dn_norm, _l2n, _l2n_qk, _moe_or_mlp,
                     _out_proj, _residual, _silu_dn)

# _deltanet_slots_ahead (the batched mixer pipelined) for a slice of at most AHEAD_PAIRS head
# pairs (unrolled: the 35B's 4 at S = 4) whose buffers take at most AHEAD_TMEM of TMEM (every
# pair's taps, windows and projections at once: the 35B's at four rows 59,528 words of 131,072
# on x2); else the pairs in their loop, unpipelined
AHEAD_PAIRS = 4
AHEAD_TMEM = 0.5


def _ahead_words(R: int, NP: int, TP: int, K: int, CP: int, dv: int, qkw: int, og: int) -> int:
    """TMEM words of _deltanet_slots_ahead's buffers (a 2-D tile's rows at an odd stride)."""
    return (NP * (TP + R * (K * CP + 1) + R * (2 * dv + 1))       # taps, windows | q k v, z
            + 2 * R * (qkw + 3 * (2 * dv + 1)) + R * (CP + 1)     # QK, V, GZ, O by parity, U
            + 2 * R * (og * dv + 1) + R * (4 * NP + 1))          # ON by parity, the gates


def _gates_rows(a, b, alog, dtb, out, fold: bool) -> None:
    """kernels.deltanet.gates of R rows at once: a, b [R, H] (alog, dtb [H] over every row) ->
    out [R, 2 H], row r its decays then its betas, gates' operations per element in its order
    (each row bit for bit gates' of that row), one VOP chain for the R rows."""
    H = a.cols
    if fold:
        d = softplus(a + dtb[None, :]) * alog[None, :]
        out[:, 0:H].set(ol.exp2(d))
        out[:, H:2 * H].set(ol.recip(ol.exp2(b) + 1.0))
        return
    g = softplus(a + dtb[None, :]) * ol.exp2(alog * ol.LOG2E)[None, :]
    out[:, 0:H].set(ol.exp2(g * -ol.LOG2E))
    out[:, H:2 * H].set(ol.recip(ol.exp2(b * -ol.LOG2E) + 1.0))


def _deltanet_slots_ahead(x, lw, slots, spec):
    """_deltanet_slots pipelined as _deltanet_dstep's ahead, over the R rows (the same words):

        DMA  every pair's windows and taps first (pair 0's first, its windows before its taps
             and the layer's vectors), one load a window into its row of the pair's tile CW[p]
             (row r: its K - 1 window rows, then its q k v projections, which the MM writes
             there)
        MXU  a|b, pair 0's projections (after its windows: one tile), pair 1's after pair 0's
             prep, then each pair's two ahead of its DSTEPs (a pair's convolution waits on its
             MM: one tile, so pair p + 1's window products do not run beside pair p's prep)
        VPU  the R rows' gates into G at once (TMEM: no DRAM scratch), pair 0's prep
        then per pair p (buffers t = p % 2):
        DMA  each row's DSTEP2 (its two heads in one) on its slot's states, back to back
        VPU  prep(p + 1) beside them (convolutions; SiLU, l2 norms and the z gate each one chain
             over the pair's blocks), then post(p)
        MXU  the out_proj flush pair p completes (the last group pair by pair, as
             _deltanet_slots), its ON by the group's parity; the first writes y (no FILL)
        DMA  after the last DSTEP each row's next window, one store: its tile row's last K - 1
             blocks (window rows 1 .. K - 2, then its projections)"""
    eps, K, R = spec.eps, spec.conv_k, x.rows
    dk, dv = spec.lin_dk, spec.lin_dv
    dn = lw.dn
    eo = _dn_eps(dn, spec)
    nl, C = dn.nl, 2 * dk + dv
    og, sh = dn.og, dn.shared
    nb, bw = (4, dk) if sh else (2, C)                  # a pair's blocks (_deltanet_rows')
    CP, RP = nb * bw, nb * bw + 2 * dv
    NP, ng, gp = nl // 2, nl // og, og // 2
    TP, WW = K * CP, (K - 1) * CP
    split = og == 4 and NP > 1                          # the last group pair by pair
    gin = ol.load_home(lw.g_in, "g_in")
    taps = [ol.empty([TP]) for _ in range(NP)]
    CW = [ol.empty([R, K * CP]) for _ in range(NP)]     # row r: its window's K - 1 rows, q k v
    Z = [ol.empty([R, 2 * dv]) for _ in range(NP)]      # a pair's z rows

    def fetch(p):
        for r, s in enumerate(slots):
            ol.load(dn.window(p, s)[0:WW], out=CW[p][r, 0:WW])
        ol.load(dn.cv(p)[0:TP], out=taps[p])

    def project(p):
        ol.dot(xs, dn.wh(p)[0:CP, :], out=CW[p][:, WW:WW + CP])
        ol.dot(xs, dn.wh(p)[CP:RP, :], out=Z[p])                            # z of a, of b

    fetch(0)
    alog, dtb, gs = ol.load(lw.alog), ol.load(lw.dtb), _dn_gain(ol.load(lw.gn), dv)
    xs = ol.quantize(rmsnorm(x, gin, eps))
    del gin
    ab = ol.dot(xs, lw.wab)                             # [R, 2nl]: a, then b, of each head
    project(0)
    for p in range(1, NP):
        fetch(p)
    G = ol.empty([R, 2 * nl])                           # row r: its decays, then its betas
    _gates_rows(ab[:, 0:nl], ab[:, nl:2 * nl], alog, dtb, G, dn.fold)
    del ab, alog, dtb
    qkw = 2 * dk if sh else 4 * dk          # a row's q | k: its key head's, or each head's
    QK = [ol.empty([R, qkw], dense=True) for _ in range(2)]
    V, GZ, O = ([ol.empty([R, 2 * dv]) for _ in range(2)] for _ in range(3))

    def conv(p, a, out):
        """Block a's convolution (the K - 1 positions before each row's, every tap) into out
        [R, bw]: _deltanet_slots' head_in, its terms in that order (the projection's first)."""
        W, tp = CW[p], taps[p].reshape(nb * K, bw)

        def blk(j):                                     # window row j's block a
            return W[:, j * CP + a * bw:j * CP + (a + 1) * bw]
        u = blk(K - 1) * tp[a * K + K - 1, :][None, :]
        for j in range(1, K - 1):
            u = u + blk(K - 1 - j) * tp[a * K + K - 1 - j, :][None, :]
        out.set(u + blk(0) * tp[a * K, :][None, :])

    def prep(p, t):
        """Pair p's q | k, v and z gate rows into buffers t (_deltanet_slots' heads, per element
        their operations: SiLU, the l2 norms (_l2n_qk) and the gate's SiLU each one chain over
        the blocks)."""
        if sh:                                          # the key head's q and k once
            U = ol.empty([R, 2 * dk], dense=True)
            for a in range(4):
                conv(p, a, U[:, a * dk:(a + 1) * dk] if a < 2 else
                     V[t][:, (a - 2) * dv:(a - 1) * dv])
            U.set(_silu_dn(dn, U))
            V[t].set(_silu_dn(dn, V[t]))
            _l2n_qk(dn, U, QK[t], dk)
            del U
        else:
            for a in range(2):
                U = ol.empty([R, C])
                conv(p, a, U)
                U.set(_silu_dn(dn, U))
                q0 = 2 * dk * a
                QK[t][:, q0:q0 + dk].set(_l2n(dn, U[:, 0:dk], dk ** -0.5))
                QK[t][:, q0 + dk:q0 + 2 * dk].set(_l2n(dn, U[:, dk:2 * dk]))
                V[t][:, a * dv:(a + 1) * dv].set(U[:, 2 * dk:C])
                del U
        GZ[t].set(_silu_dn(dn, Z[p]))                   # _dn_gz: SiLU, then the gain
        for a in range(2):
            GZ[t][:, a * dv:(a + 1) * dv].set(GZ[t][:, a * dv:(a + 1) * dv] * gs[None, :])
        taps[p] = Z[p] = None

    def steps(p, t):
        """Each row's DSTEP2 on its slot's two states of pair p."""
        q = 2 * p
        for r, s in enumerate(slots):
            ol.deltanet_step2(dn.states(p, s), QK[t][r, :], V[t][r, :], G[r, q:q + 2],
                              G[r, nl + q:nl + q + 2], O[t][r, :])

    ONg = [ol.empty([R, og * dv]) for _ in range(2)] if gp > 1 else None
    ONp = [ol.empty([R, 2 * dv]) for _ in range(2)] if gp == 1 or split else None
    y = ol.empty([R, spec.hidden // ol.num_programs()])
    fresh = [True]                                      # y not yet written (_out_proj first)

    def post(p, t):
        """Pair p's normed, gated o into its group's ON; the out_proj flush it completes."""
        g, i = divmod(p, gp)
        last = split and g == ng - 1
        dst = ONp[p % 2] if gp == 1 or last else ONg[g % 2][:, 2 * i * dv:(2 * i + 2) * dv]
        for a in range(2):
            dst[:, a * dv:(a + 1) * dv].set(_dn_norm(O[t][:, a * dv:(a + 1) * dv], eo) *
                                            GZ[t][:, a * dv:(a + 1) * dv])
        if gp == 1:
            _out_proj(dn, dst, g, 0, 2, dv, y, first=fresh[0])
        elif last:
            _out_proj(dn, dst, g, 2 * i, 2 * i + 2, dv, y, first=fresh[0])
        elif i == gp - 1:
            _out_proj(dn, ONg[g % 2], g, 0, og, dv, y, first=fresh[0])
        else:
            return
        fresh[0] = False

    prep(0, 0)
    if NP > 1:                                          # (its weights' stream after pair 0's
        project(1)                                      # windows and prep: off their DRAM)
    for p in range(NP):
        t = p % 2
        if p + 2 < NP:
            project(p + 2)
        steps(p, t)
        if p + 1 < NP:
            prep(p + 1, 1 - t)
        post(p, t)
    for p in range(NP):                                 # the rows' next windows
        for r, s in enumerate(slots):
            ol.store(dn.window(p, s)[0:WW], CW[p][r, CP:K * CP])
    del CW, G, QK, V, GZ, O
    return _residual(x, y)


def _deltanet_slots(x, lw, slots, spec, gr, on):
    """x + the DeltaNet mixer of R decode rows, row r sequence slots[r]'s next token: the
    operations of _deltanet_rows (its unpipelined pairs, DSTEP) per row, each row's window and
    states its slot's. The projections stream once for the R rows; the R windows' rows come
    in as tiles W[j] (row r: its window's row j, oldest first), so each block's convolution is
    one chain over the R rows, the terms in _deltanet_rows' order; each row's next window
    (rows 1 .. K - 2 of its own, then its projections) is stored back to its slot. gr [R, 2nl]
    and on [R, og dv]: _deltanet_rows' DRAM scratch. Pipelined (_deltanet_slots_ahead) where
    the slice's pairs unroll and their buffers fit (AHEAD_PAIRS, AHEAD_TMEM)."""
    if not ol.has_dstep():
        raise CompileError("batched DeltaNet rows need DSTEP")
    eps, K, R = spec.eps, spec.conv_k, x.rows
    dk, dv = spec.lin_dk, spec.lin_dv
    dn = lw.dn
    if dn.qkonce:
        raise CompileError("batched DeltaNet rows: not with q and k once a layer (qkonce)")
    eo = _dn_eps(dn, spec)
    nl, C = dn.nl, 2 * dk + dv
    og, sh = dn.og, dn.shared
    nb, bw = (4, dk) if sh else (2, C)                  # a pair's blocks (_deltanet_rows')
    CP, RP = nb * bw, nb * bw + 2 * dv
    NP, ng, gp = nl // 2, nl // og, og // 2
    TP = K * CP
    if NP <= AHEAD_PAIRS and _ahead_words(R, NP, TP, K, CP, dv, 2 * dk if sh else 4 * dk, og) \
            <= AHEAD_TMEM * current().cfg.TMEM_WORDS:
        return _deltanet_slots_ahead(x, lw, slots, spec)
    xs = ol.quantize(rmsnorm(x, ol.load_home(lw.g_in, "g_in"), eps))
    split = og == 4 and NP > 1                          # the last group pair by pair
    ab = ol.dot(xs, lw.wab)                             # [R, 2nl]: a, then b, of each head
    alog, dtb = ol.load(lw.alog), ol.load(lw.dtb)
    for r in range(R):
        dr, br = gates(ab[r, 0:nl], ab[r, nl:2 * nl], alog, dtb, fold=dn.fold)
        ol.store(gr[r, 0:nl], dr)
        ol.store(gr[r, nl:2 * nl], br)
        del dr, br
    del ab, alog, dtb
    gs = _dn_gain(ol.load(lw.gn), dv)
    y = ol.zeros([R, spec.hidden // ol.num_programs()])    # (_out_proj)
    taps = ol.empty([TP])
    XP, Z = ol.empty([R, CP]), ol.empty([R, 2 * dv])    # a pair's q k v rows, its z
    W = [ol.empty([R, CP]) for _ in range(K - 1)]       # the rows' windows, row by row
    ONp = ol.empty([R, 2 * dv])                         # normed, gated o of a pair per row
    GDB = ol.empty([R, 4])                              # decays, then betas, of the pair's heads
    GD, GB = GDB[:, 0:2], GDB[:, 2:4]

    def project(p):
        """Pair p's taps, the rows' windows and projections; the rows' next windows stored."""
        ol.load(dn.cv(p)[0:TP], out=taps)              # rows (block, tap)
        for r, s in enumerate(slots):
            win = dn.window(p, s)
            for j in range(K - 1):
                ol.load(win[j * CP:(j + 1) * CP], out=W[j][r, :])
        ol.dot(xs, dn.wh(p)[0:CP, :], out=XP)
        ol.dot(xs, dn.wh(p)[CP:RP, :], out=Z)                             # z of a, of b
        for r, s in enumerate(slots):
            win = dn.window(p, s)
            for i in range(K - 2):
                ol.store(win[i * CP:(i + 1) * CP], W[i + 1][r, :])
            ol.store(win[(K - 2) * CP:(K - 1) * CP], XP[r, :])

    def head_in(tp, a):
        """Block a's convolution (the K - 1 positions before each row's, every tap) and SiLU ->
        [R, bw]: _deltanet_rows' head_in at p0 >= K - 1, its terms in that order."""
        blk = slice(a * bw, (a + 1) * bw)
        u = XP[:, blk] * tp[a * K + K - 1, :][None, :]
        for j in range(1, K - 1):
            u = u + W[K - 1 - j][:, blk] * tp[a * K + K - 1 - j, :][None, :]
        U = ol.empty([R, bw])
        U.set(u + W[0][:, blk] * tp[a * K, :][None, :])
        del u
        U.set(_silu_dn(dn, U))
        return U

    def steps(p, QK, V, O):
        """Each row's DSTEP on its slot's state of head a of pair p (QK[a] / V[a]: [R, 2dk] /
        [R, dv] of head a)."""
        for a in range(2):
            for r, s in enumerate(slots):
                ol.deltanet_step(dn.state(p, a, s), QK[a][r, :], V[a][r, :], GDB[r, a:a + 1],
                                 GDB[r, 2 + a:3 + a], O[a][r, :])

    def heads(p):
        """Pair p's heads -> ONp (_deltanet_rows' heads / shared_heads with DSTEP)."""
        tp = taps.reshape(nb * K, bw)
        q = dn.index(p)
        for r in range(R):
            ol.load(gr[r, 2 * q:2 * q + 2], out=GD[r, :])
            ol.load(gr[r, nl + 2 * q:nl + 2 * q + 2], out=GB[r, :])
        if sh:                                          # the key head's q and k once
            QK1 = ol.empty([R, 2 * dk])
            for i in range(2):
                U = head_in(tp, i)
                QK1[:, i * dk:(i + 1) * dk].set(_l2n(dn, U, dk ** -0.5 if i == 0 else 1.0))
                del U
            QK = [QK1, QK1]
            V = [head_in(tp, 2 + a) for a in range(2)]
            GZ = [_dn_gz(dn, Z[:, a * dv:(a + 1) * dv], gs) for a in range(2)]
        else:
            QK = [ol.empty([R, 2 * dk]) for _ in range(2)]
            V = [ol.empty([R, dv]) for _ in range(2)]
            GZ = [ol.empty([R, dv]) for _ in range(2)]
            for a in range(2):
                U = head_in(tp, a)
                QK[a][:, 0:dk].set(_l2n(dn, U[:, 0:dk], dk ** -0.5))
                QK[a][:, dk:2 * dk].set(_l2n(dn, U[:, dk:2 * dk]))
                V[a].set(U[:, 2 * dk:C])
                del U
                GZ[a].set(_dn_gz(dn, Z[:, a * dv:(a + 1) * dv], gs))
        O = [ol.empty([R, dv]) for _ in range(2)]
        steps(p, QK, V, O)
        del QK, V
        for a in range(2):
            ONp[:, a * dv:(a + 1) * dv].set(_dn_norm(O[a], eo) * GZ[a])

    def pair(p):
        project(p)
        heads(p)

    def loop(n):
        return ol.range(n) if n > 1 else range(n)

    for g in loop(ng - 1 if split else ng):             # whole groups (_deltanet_rows')
        if gp == 1:
            pair(g)
            _out_proj(dn, ONp, g, 0, 2, dv, y)
            continue
        for q in loop(gp):
            pair((g, q))
            ol.store(on[:, q * 2 * dv:(q + 1) * 2 * dv], ONp)
        _out_proj(dn, ol.load(on), g, 0, og, dv, y)
    if split:                                           # the last group, pair by pair
        for q in range(gp):
            pair((ng - 1, q))
            _out_proj(dn, ONp, ng - 1, 2 * q, 2 * q + 2, dv, y)
    return _residual(x, y)


def _moe_rows(x, lw, m, spec) -> None:
    """x += the layer's MLP or MoE over its rows, in place: the dense MLP, or a MoE's experts
    by groups of rows: moe_ffn_rows over as many rows as one request names (R k <= LINE / 4;
    uniform top-k picks of a few rows rarely share an expert, so groups cost about what one
    union does), a lone row moe_ffn; at S > 1 moe_ffn_tp_rows (tensor-parallel experts: the
    router and the shared expert once over the rows, each row's routed experts in turn). Each
    row's result is moe_ffn's on that row, bit for bit."""
    if not lw.moe or x.rows == 1:
        x.set(_moe_or_mlp(x, lw, m, spec))
        return
    mo, R = spec.moe, x.rows
    if ol.num_programs() > 1:
        x.set(MO.moe_ffn_tp_rows(x, lw, mo, m.moe_dev, spec.eps))
        return
    n = max(1, (LINE // 4) // mo.k)
    for r0 in range(0, R, n):
        g = min(n, R - r0)
        xr = x[r0:r0 + g, :]
        xr.set(MO.moe_ffn(xr, lw, mo, m.moe_dev, spec.eps) if g == 1 else
               MO.moe_ffn_rows(xr, lw, mo, m.moe_dev, spec.eps))


def layer_rows(x, lw, m, rows, c, s_, block: int) -> None:
    """x += one layer over the rows (slot, position), in place: its mixer on each row's slot
    (DeltaNet: _deltanet_slots; attention: the rows' own caches, kv_slots), then its MLP or
    MoE (_moe_rows). c, s_: the rows' RoPE rows."""
    spec, R = m.spec, x.rows
    if lw.kind == LIN:
        x.set(_deltanet_slots(x, lw, [s for s, _ in rows], spec, m.gr[0:R, :], m.on[0:R, :]))
    else:
        lws = SimpleNamespace(**{**vars(lw), "kvs": lw.kv_slots})
        x.set(_attention_rows(x, lws, c, s_, rows, spec, block, gated=True))
    _moe_rows(x, lw, m, spec)


def _inputs_area(m, pos, area):
    """The rows' embeddings from their token ids in the batch area's tin (run-time values, the
    image's tables) and their RoPE rows from the tables at positions pos (batch_step's)."""
    R = len(pos)
    x = ol.empty([R, m.xr.shape[1]], dense=True)
    c = ol.empty([R, m.cosr.shape[1]], dense=True)
    s_ = ol.empty([R, m.sinr.shape[1]], dense=True)
    for r, p in enumerate(pos):
        ol.load(m.cos_t[p:p + 1, :], out=c[r:r + 1, :])
        ol.load(m.sin_t[p:p + 1, :], out=s_[r:r + 1, :])
    tk = ol.load(Tensor(area["tin"], (R,), (1,)))
    for r in range(R):
        x[r:r + 1, :].set(_embed_word(m, tk[r:r + 1]))
    del tk
    return x, c, s_


@ol.jit
def qwen35_batch_step(m, rows, logit_rows=(), block: int = ATTN_BLOCK, tokens=None,
                      kinds=None, area=None, span: int = 0, slots: bool = False,
                      nest: bool = False):
    """One token for each of the rows' sequences (rows[r] = (slot, position), one row a slot,
    positions >= conv_k - 1): their embeddings m.xr and RoPE rows m.cosr / m.sinr (the host's),
    or those of `tokens` from the image's tables -> the logits of `logit_rows` (m.logits). With
    a batch area (batch.alloc's) the tokens come from its tin and each row picks on the card
    (kinds[r]: batch.GREEDY or a Sampling key, batch._sinks) into tout[r]; with a span, for the
    steps t < span at positions + t (t: the area's step word); with slots, at every position:
    each row's position and attention blocks run-time values, its words in the area's rw
    (batch.slot_rows; the rows' positions are not the program's). nest: a unit's DeltaNet
    layers loop inside the unit's loop (m.subs, the prefill's qwen35.NEST: one copy of a
    DeltaNet layer instead of three, the same arithmetic in the same order), for a step
    whose layers do not fit IMEM unrolled (compile_step)."""
    b = current()
    spec, R = m.spec, len(rows)
    sl = [s for s, _ in rows]
    if len(set(sl)) != R:
        raise CompileError(f"rows {rows}: one row a sequence")
    if not slots and any(p < spec.conv_k - 1 for _, p in rows):
        raise CompileError(f"rows {rows}: batched rows from position conv_k - 1")
    if area is not None and (tokens is not None or R > area["R"]):
        raise CompileError("a batch area's step takes its tokens from the area, at most R rows")
    if kinds is not None and BA.FEED in kinds:
        raise CompileError("batched DeltaNet rows pick: a prompt feeds by the prompt runs")
    if (span or slots) and area is None:
        raise CompileError("a span of steps or a step over slots reads its words from a batch "
                           "area")
    if slots and area.get("mask_blocks", 1) < SPAN_P:
        raise CompileError(f"a step over slots masks {SPAN_P} blocks past a row: the mask table of "
                           "an image of several sequences (qwen3._lookup_alloc's mask_blocks)")
    if span:                                    # (batch_step's: the step word, the RLDs below)
        rows = BA.step_rows(rows, span, block, area["zmask"], area["cap"])
        stw = ol.load(Tensor(area["step"], (2,), (1,)))
        at = b.stack[-1][-1]
        b.run_words = {"tpos": stw.base, "fpos": stw.base + 1}
    elif slots:                                 # each row's words (its position, V^T column and
        stw = ol.load(Tensor(area["rw"], (BA.SLOT_WORDS * R,), (1,)))  # attention groups),
        at = b.stack[-1][-1]                                     # the RLDs below
        d = -(-spec.head_dim // ol.block_size()) * ol.block_size()
        rows = BA.slot_rows(rows, stw.base, block, area["zmask"], area["cap"], d,
                            wg=gall_wg(b.cfg))
        b.run_words = {k: w for _, p in rows for k, w in p.words.items()}
        b.transient = True                      # (no argument registers held from the start)
    x, c, s_ = _inputs_rows(m, rows, tokens) if area is None else \
        _inputs_area(m, [p.pos if span or slots else p for _, p in rows], area)

    run_layers(m.plan, lambda li, it, jt=None: layer_rows(x, m.layer(li, it, jt), m, rows, c,
                                                          s_, block),
               m.subs if nest else None)
    if area is None:
        _lm_head_rows(x, m, spec, logit_rows)
        return
    del c, s_
    kinds = [BA.GREEDY] * R if kinds is None else list(kinds)
    BA._sinks(b, m, spec, x, rows, kinds, list(range(R)), area)
    if span or slots:                           # the arguments from the step words, at the start
        body = b.stack[-1]
        i = next(j for j, x_ in enumerate(body) if x_ is at) + 1
        body[i:i] = [x_ for k, (v, c_) in enumerate(b.run_args)
                     for x_ in I.rld_mul(15 - k, b.run_words[v.name], int(c_), raw=v.raw,
                                         bound=mul_bound(c_, v.bound, b.aw) if slots else None,
                                         comment=f"argument {c_}*{v.name}", aw=b.aw)]
        b.run_words, b.transient = None, False
        del stw


def compile_step(image, rows, logit_rows=(), block: int = ATTN_BLOCK, tokens=None,
                 kinds=None, area=None, span: int = 0, slices=None, slots: bool = False,
                 nest: bool | None = None) -> list:
    """One program per slice: qwen35_batch_step over rows [(slot, position)] of an
    Image(batch=N): the logits of `logit_rows` (tokens: their ids, inputs from the image's
    tables, else the host writes the rows' inputs), or with a batch area (Engine.bat) each
    row's pick of kind kinds[r] (default greedy) into its tout word, for span steps with a
    span (batch.compile_step's), or at every position of the rows' slots with slots (each
    row's words in the area's rw: slot_words; the positions are not the program's). slices:
    those slices' programs only (default every one). nest (qwen35_batch_step's): None, the
    layers unrolled where they fit IMEM, else a unit's DeltaNet layers in a loop (the 35B's
    B = 4 on x2 without ICACHE: 5,063 instructions unrolled)."""
    rows = [(int(s), 0 if slots else int(p)) for s, p in rows]
    if not 1 <= len(rows) <= min(image.batch, image.rows):
        raise ValueError(f"{len(rows)} rows: an image of {image.batch} sequences, {image.rows} "
                         "I/O rows")
    kw = {"kinds": kinds, "area": area, "span": span, "slots": slots} if area is not None else \
        {"logit_rows": list(logit_rows), **_tokens_arg(image, tokens, rows)}

    def trace(n):
        return [qwen35_batch_step.trace(image.cfg, s, {"m": image.descriptors(s), "rows": rows,
                                                       "block": block, "nest": n, **kw}).finish()
                for s in (range(image.cfg.S) if slices is None else slices)]
    if nest is not None:
        return trace(nest)
    try:
        progs = trace(False)
        if all(len(p) <= image.cfg.IMEM_WORDS // 8 for p in progs):
            return progs
    except CompileError:
        pass
    return trace(True)


def prefill_slot(eng, tokens, slot: int, logits: bool = True):
    """A prompt into slot `slot` of an Engine(batch=N, layer_major=R): its layer-major runs at
    that slot (Engine.prefill_layers(slot=): the image's descriptors at the slot address its KV
    caches, DeltaNet states and windows) from Engine.poss[slot] on, which advances; the logits
    after its last token (logits). Slot 0: Engine.prefill_layers as it is."""
    if not eng.layer_major:
        raise ValueError("a slot's prompt goes in by layer runs: Engine(layer_major=R)")
    if slot == 0:
        return eng.prefill_layers(tokens, logits)
    K = eng.spec.conv_k
    if getattr(eng.image, "embed_host", False) and not eng.embed_runs and eng.poss[slot] < K - 1:
        raise ValueError(f"slot {slot}: the rows before conv_k - 1 by the decode step of "
                         "sequence 0 (embed_host): Engine(embed_runs=True)")
    p0 = eng.poss[0]                            # (the runs go from Engine.pos)
    eng.poss[0] = eng.poss[slot]
    try:
        out = eng.prefill_layers(tokens, logits, slot=slot)
        eng.poss[slot] = eng.poss[0]
    finally:
        eng.poss[0] = p0
    return out


def slots_fit(eng) -> bool:
    """Whether an engine's batched steps can run over slots (Steps(slots=True): one program for
    every position of its rows' slots): its mask table SPAN_P blocks past the cache (an
    Image(batch > 1)'s) and a cache of whole groups of SPAN_P blocks."""
    a = getattr(eng, "bat", None)
    return a is not None and a.get("mask_blocks", 1) >= SPAN_P and \
        a["cap"] % (SPAN_P * eng.block) == 0


def _worker_step(rows, kinds, area, span=0, s=0, slots=False):
    """The engine's compile worker process: compile_step of slice s over the batch area (over
    slots with slots), assembled (batch._worker_step's)."""
    from .qwen3 import _WORKER, worker_words
    image, block = _WORKER
    return worker_words(image, compile_step(image, rows, block=block, kinds=kinds, area=area,
                                            span=span, slices=[s], slots=slots)[0])


class Steps(BA.Steps):
    """batch.Steps over a Qwen3.5 image's batched steps (qwen35_batch_step): each slot's
    sampling, the steps' groups that fit IMEM, programs that serve a span of steps and their
    successors compiled ahead, as the dense models', B of them (depth: B rows at different
    places in their attention blocks end B spans a block; the 35B's step compiles with its
    wait entries in ~2 s at B = 2 and ~3 s at B = 4, its shortest spans run less). A row
    picks: no feed rows (a slot's prompt goes in by its own runs) and so no mixed runs.

    slots (slots_fit(eng)): one program per (slots, kinds) for every position of the slots
    (compile_step(slots=True)), each row's words (qwen3.SlotPos.values) in the io block's rw:
    compiled once, then live for good; no spans, nothing compiled ahead but other row sets'."""

    worker = staticmethod(_worker_step)

    def __init__(self, eng, span: bool = True, slots: bool = False):
        super().__init__(eng, span)
        self.depth = max(1, eng.batch)
        self.slots = bool(slots)
        if self.slots:
            if not slots_fit(eng):
                raise ValueError(f"steps over slots: a mask table {SPAN_P} blocks past the cache "
                                 f"and a cache of whole groups of them (cap {eng.cap})")
            import functools
            self.worker = functools.partial(_worker_step, slots=True)
            D = eng.cfg.D
            self._d = -(-eng.spec.head_dim // D) * D        # (the V^T rows: SlotPos.values)
            self._wg = gall_wg(eng.cfg)                     # (its KV caches' groups)

    def _key(self, rows, kinds=None):
        if not self.slots:
            return super()._key(rows, kinds)
        kinds = tuple(self.kinds.get(s) for s, _ in rows) if kinds is None else tuple(kinds)
        return tuple((int(s), 0) for s, _ in rows), kinds, 0

    def _serving(self, grp, kinds=None):
        if not self.slots:
            return super()._serving(grp, kinds)
        rows, kinds, _ = self._key(grp, kinds)          # (batch.Steps._take's live key)
        k = (tuple(s for s, _ in rows), kinds, tuple(p for _, p in rows))
        if k not in self._live:
            return None
        self._live.move_to_end(k)
        return self._live[k][1], 0

    def _prefetch(self, key) -> None:
        if self.slots and key is not None and self._serving(key[0], key[1]) is not None:
            return                                      # (live: nothing to compile)
        super()._prefetch(key)

    def _io(self, grp, toks, uni, t):
        io = super()._io(grp, toks, uni, 0 if self.slots else t)
        if not self.slots:
            return io
        from .qwen3 import SlotPos
        return np.concatenate([io] + [SlotPos.values(p, self.eng.block, self.a["cap"], self._d,
                                                     wg=self._wg) for _, p in grp])

    def _trace(self, key) -> list:
        rows, kinds, n = key
        return compile_step(self.eng.image, rows, block=self.eng.block, kinds=kinds,
                            area=self.a, span=n, slots=self.slots)

    def prefetch_group(self, rows, kinds) -> None:
        """Compile ahead the step of rows, some of the batch's (a span from their positions to the
        first one's block end), unless a program serves them already: what they run on once the
        batch's other rows stop (BatchWorker35: the batch less one row). Only while a compile
        worker is left for the batch's own programs (the engine's three worker processes: two of
        these at a time; a thread's: none on the way)."""
        eng = self.eng
        workers = eng._ahead if eng._procs else 1      # (S > 1: the slice workers, one program
        if sum(not f.done() for f in self._next.values()) >= max(1, workers - 1):   # at a time)
            return
        g, k = [tuple(r) for r in rows], tuple(kinds)
        if self._serving(g, k) is None and self._coming(g, k) is None:
            self._prefetch(self._key(g, k))

    def feed(self, slot, tokens):
        raise NotImplementedError("a Qwen3.5 slot's prompt goes in by its own runs, not feed rows")

    def warm(self) -> None:                     # (batch.Steps': the feed programs)
        pass

    def mixed_ready(self, rows, fslot, k):
        return None                             # (no mixed runs: never ready)
