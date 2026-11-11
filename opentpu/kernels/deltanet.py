"""Gated DeltaNet decode step (Qwen3-Next / Qwen3.5 linear attention), one token.

The recurrent state of a head is kept transposed, St[j, i] = S[i, j] with rows j over the value
dimension and columns i over the key dimension, so both contractions are row reductions (RDOT)
and the rank-1 update is row x column (OUTER). Per head:

    kv = St @ k                   RDOT        (on the old state; the decay is applied to kv)
    d  = (v - a * kv) * beta      3 small VOPs
    St = a * St + d k^T           OUTER       (in place: ol.outer(d, k, acc=St, decay=a))
    o  = St @ q                   RDOT

three passes over the fp32 state, which streams through TMEM (LD before, ST after).
"""
from .. import language as ol
from .lib import softplus


def gates(a, b, A_log, dt_bias, out=None, fold: bool = False):
    """The per-head decay a = e^g, g = -e^A_log * softplus(a + dt_bias), and beta = sigmoid(b)
    (all [H] tiles); with `out` [2 H], decay into out[0:H] and beta into out[H:2 H], each by
    its last VOP (no copy). fold: A_log holds -log2(e) e^A_log and b comes -log2(e) times (its
    weights', qwen35's Image.dn_fold), so decay = 2^(softplus(a + dt_bias) A_log) and beta =
    1 / (1 + 2^b): three VOPs fewer."""
    if fold:
        d = softplus(a + dt_bias) * A_log
        if out is None:
            return ol.exp2(d), ol.recip(ol.exp2(b) + 1.0)
        H = a.cols
        out[0:H].set(ol.exp2(d))
        out[H:2 * H].set(ol.recip(ol.exp2(b) + 1.0))
        return out[0:H], out[H:2 * H]
    g = softplus(a + dt_bias) * ol.exp2(A_log * ol.LOG2E)
    if out is None:
        decay = ol.exp2(g * -ol.LOG2E)
        beta = ol.recip(ol.exp2(b * -ol.LOG2E) + 1.0)
        return decay, beta
    H = a.cols
    out[0:H].set(ol.exp2(g * -ol.LOG2E))
    out[H:2 * H].set(ol.recip(ol.exp2(b * -ol.LOG2E) + 1.0))
    return out[0:H], out[H:2 * H]



def gate_logs(a, b, A_log, dt_bias, fold: bool = False):
    """gates() of rows: a, b [n, H] (A_log, dt_bias [H] over every row) -> (lg, beta) [n, H],
    lg the decay's exponent (decay = 2^lg: gates' EXP2 input, so exp2(lg) is gates' decay bit
    for bit) and beta; per element gates' operations (the chunked DeltaNet's inputs: its
    cumulative decays are sums of lg)."""
    sp = softplus(a + dt_bias[None, :])
    if fold:
        return sp * A_log[None, :], ol.recip(ol.exp2(b) + 1.0)
    return (sp * ol.exp2(A_log * ol.LOG2E)[None, :]) * -ol.LOG2E, \
        ol.recip(ol.exp2(b * -ol.LOG2E) + 1.0)

def l2norm_rows(x, scale: float = 1.0, eps: float = 1e-6):
    """x[h] * scale / sqrt(|x[h]|^2 + eps) for every row h of x [H, d]."""
    r = ol.rsqrt(ol.sum(x * x, axis=1) + eps)
    return x * (r * scale if scale != 1.0 else r)[:, None]


def head_step(St, k, v, q, decay, beta, w, o, fused: bool = True, prefetch=None):
    """One head: updates the state tile St [dv, dk] in place, writes o [dv] and returns the
    result of `prefetch()`. k, q: [dk] tiles; v: [dv]; decay, beta: [1] tiles; w: a [dv] work
    tile shared by all heads (kv, then d). `prefetch` loads the next head's state; it is
    called as early as TMEM allows.

    The schedule: a TMEM bank takes one write per cycle and the DMA's writes come first, so a
    state LD (8 writes per cycle) stalls an OUTER (8 per cycle) for as long as they overlap;
    an RDOT writes nothing until its last row and runs beside it. Sharing w makes head h+1's
    first RDOT wait for head h's OUTER, so head h's last RDOT (older) goes first; the LD of
    head h+2, which reuses head h's buffer, then waits for that RDOT and head h's ST, and
    lands during head h+1's first RDOT.

    fused=False is the same step on the ISA without RDOT and OUTER (named temporaries are
    never fused): 7 state passes instead of 3. Every one of them writes TMEM, so a state LD
    could not overlap any of them anyway, and they need two 16K-word temporaries: the states
    are not prefetched (prefetch is not called)."""
    if fused:
        nxt = prefetch() if prefetch else None
        w.set(St @ k)                                    # kv
        w.set((v - w * decay) * beta)                    # d
        ol.outer(w, k, acc=St, decay=decay)
        o.set(St @ q)
        return nxt
    p = St * k[None, :]
    w.set(ol.sum(p, axis=1))
    del p
    w.set((v - w * decay) * beta)
    St.set(St * decay)                    # in place: one 16K-word temporary at a time
    r1 = ol.outer(w, k)
    St.set(St + r1)
    del r1
    p = St * q[None, :]
    o.set(ol.sum(p, axis=1))
    return None


@ol.jit
def gated_deltanet_step(state, q, k, v, a, b, A_log, dt_bias, state_out, out, fused=True):
    """One decode token of a Gated DeltaNet layer's recurrence, all heads.

    state: Input [H, dv, dk] (transposed states St); q, k: Input [H, dk]; v: Input [H, dv];
    a, b: Input [H] (the gate pre-activations); A_log, dt_bias: Input [H].
    state_out: Output [H, dv, dk]; out: Output [H, dv]. q and k are L2-normalized, q scaled
    by dk^-1/2, as in the reference recurrence. Two state buffers: the DMA loads head h+1's
    state while the VPU works on head h (fused; see head_step).
    """
    H, dv, dk = state.shape
    w, o = ol.empty((dv,)), ol.empty((H, dv))
    decay, beta = gates(ol.load(a), ol.load(b), ol.load(A_log), ol.load(dt_bias))
    qn = l2norm_rows(ol.load(q), dk ** -0.5)
    kn = l2norm_rows(ol.load(k))
    vt = ol.load(v)
    St = ol.load(state[0])
    for h in ol.static_range(H):
        nxt = (lambda: ol.load(state[h + 1])) if h + 1 < H else None
        St_next = head_step(St, kn[h], vt[h], qn[h], decay[h:h + 1], beta[h:h + 1], w, o[h],
                            fused, nxt)
        ol.store(state_out[h], St)
        St = St_next if fused or h + 1 == H else ol.load(state[h + 1])
    ol.store(out, o)
