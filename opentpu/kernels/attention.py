"""Decode attention over the int8 KV cache (flash-style online softmax), GQA, head-parallel."""
import math

from .. import language as ol
from .lib import rmsnorm, rope


def _attend(qh, kv, h, seq_len: int, block: int, scale: float | None = None, raw: bool = False,
            depth: int = 3):
    """Flash attention of the G query rows `qh` [G, d] against KV head h (see _attend_heads);
    returns acc / l, or the unnormalized (acc, l) with `raw`."""
    (acc, l), = _attend_heads([qh], kv, [h], seq_len, block, scale, depth)
    if raw:
        return acc, l
    return acc / l[:, None]


class _Head:
    """Online-softmax state of one KV head's query group."""

    def __init__(self, qh, kv, h, scale):
        G, d = qh.shape
        if scale is None:
            self.qs = ol.quantize(qh)                # q stays stationary for every K block
        else:
            self.qs = ol.quantize(qh * ol.full([d], scale)[None, :])
        self.G = G
        self.m = ol.full([G], -1e30)
        self.l = ol.zeros([G])
        self.acc = ol.zeros([G, d])
        self.K, self.VT, self.VS = kv.k(h), kv.vt(h), kv.vscale(h)
        self.buf = {}                                # block index -> score buffer
        self.S = None                                # the hardware loop's score buffers


def _attend_heads(qhs, kv, hs, seq_len: int, block: int, scale: float | None = None,
                  depth: int = 3, ahead: int = 1):
    """Flash attention (online softmax) of several KV heads' query groups, software-pipelined
    FA3-style: while the VPU and quantizer finish block b (softmax, P.V), the MXU is already
    streaming q.K^T of the next depth-1 blocks into other score buffers -- across head
    boundaries too, so the heads' dependency chains overlap. Within a head the hardware loop
    body covers `depth` blocks, so every score buffer has a fixed role.

    qhs: [G, d] query tiles, or functions that emit and return them (they are called
    `ahead` heads early, so the work producing a head's queries -- e.g. its slice of the Q
    projection -- is interleaved with the attention of the heads before it); hs: their KV
    heads. The softmax scale log2(e)/sqrt(d) is either already in the queries or given as
    `scale` (applied by the quantizer when q is loaded into ACT RAM, QACT CSCALE). Returns
    [(acc, l)] per head, unnormalized.
    """
    D = ol.block_size()
    heads: list = []

    def head(i):
        while len(heads) <= min(i, len(hs) - 1):
            j = len(heads)
            qh = qhs[j]() if callable(qhs[j]) else qhs[j]
            heads.append(_Head(qh, kv, hs[j], scale))
        return heads[min(i, len(hs) - 1)]

    def scores(st, t0, n, out):
        """s = q.K^T for tokens [t0, t0+n); the MXU epilogue also writes the row maxima."""
        ol.dot(st.qs, st.K[t0:t0 + n, :], out=out, rowmax=True)
        return out

    def finish(st, s, t0, n):
        """Online-softmax update and acc += P.V for the block whose scores are in `s`."""
        G = st.G
        vs = ol.load(st.VS[t0:t0 + n])
        m_new = ol.maximum(st.m, s.rowmax)
        p = ol.exp2(s - m_new[:, None])                           # fused EXP2SUB
        alpha = ol.exp2(st.m - m_new)
        npad = -(-n // D) * D
        if npad == n:
            pq = ol.quantize(p * vs[None, :])                     # V scales folded (QACT CSCALE)
        else:                                                     # tail block: pad P with zeros
            pp = ol.zeros([G, npad])
            pp[:, :n].set(p * vs[None, :])
            pq = ol.quantize(pp)
        ol.dot(pq, st.VT[:, t0:t0 + npad], acc=st.acc, acc_scale=alpha)  # acc*alpha + P.V
        st.m.set(m_new)                                           # the next block needs m first
        st.l.set(st.l * alpha + ol.sum(p, axis=1))                # off the critical path

    nfull, tail = divmod(seq_len, block)
    blocks = [(i * block, block) for i in range(nfull)] + ([(nfull * block, tail)] if tail else [])
    P = depth
    groups = max(0, (nfull + 1 - P) // P)

    def prime(st):
        """Put the head's first P-1 score blocks in flight."""
        if groups:
            st.S = [ol.empty([st.G, block]) for _ in range(P)]
            for j in range(P - 1):
                scores(st, j * block, block, st.S[j])
        else:
            for f in range(min(P - 1, len(blocks))):
                t0, n = blocks[f]
                st.buf[f] = scores(st, t0, n, ol.empty([st.G, n]))

    head(ahead)
    prime(head(0))
    for i in range(len(hs)):
        st = head(i)
        head(i + ahead + 1)                                       # the next queries, early
        nxt = head(i + 1) if i + 1 < len(hs) else None
        nb = 0
        if groups:
            S = st.S
            for g in ol.range(groups):
                t0 = g * (P * block)
                for p in range(P):
                    scores(st, t0 + (p + P - 1) * block, block, S[(p + P - 1) % P])   # prefetch
                    finish(st, S[p], t0 + p * block, block)
            nb = groups * P
            for j in range(P - 1):
                st.buf[nb + j] = S[j]
        primed = False
        for b in range(nb, len(blocks)):                          # the rest, unrolled
            for f in range(b, min(b + P, len(blocks))):           # keep P-1 blocks in flight
                if f not in st.buf:
                    t0, n = blocks[f]
                    st.buf[f] = scores(st, t0, n, ol.empty([st.G, n]))
            if nxt is not None and not primed and b + P - 1 >= len(blocks):
                prime(nxt)                                        # ... continuing into the next head
                primed = True
            if b + 1 == len(blocks):
                st.qs = None                                      # its ACT RAM blocks are free
            t0, n = blocks[b]
            finish(st, st.buf.pop(b), t0, n)
        if nxt is not None and not primed:
            prime(nxt)
    return [(st.acc, st.l) for st in heads]


@ol.jit
def attention_decode(q, kv, out, n_q_heads, n_kv_heads, seq_len, block):
    """out[h*G+g] = softmax(q K_h^T / sqrt(d)) V_h for the KV heads owned by this slice.

    q: Input [Hq, d]; kv: KVCache (heads round-robin over slices); out: Output [Hq, d].
    """
    G = n_q_heads // n_kv_heads
    scale = ol.LOG2E / math.sqrt(kv.d)
    for h in kv.owned_heads(n_kv_heads):
        acc, l = _attend(ol.load(q[h * G:(h + 1) * G, :]), kv, h, seq_len, block, scale, raw=True)
        inv = ol.recip(l)
        for g in ol.static_range(G):     # row by row: the DMA stores row g while the VPU scales g+1
            ol.store(out[h * G + g:h * G + g + 1, :], acc[g:g + 1, :] * inv[g:g + 1][:, None])


@ol.jit
def attention_layer(h, gamma, wq, wk, wv, wo, cos, sin, kv, out,
                    n_q_heads, n_kv_heads, pos, block, eps):
    """Full decode attention block for one token at position `pos`:

    xn = rmsnorm(h)*gamma; q,k,v projections; RoPE; append k,v to the cache; attention over
    pos+1 tokens; out = h + W_o o. Heads are split over slices: slice s owns KV heads s, s+S, ..
    and their query heads; wq/wk/wv hold the matching head rows per slice (see ref layout).
    h: Input [1, H]; wq: Weight [Hq_loc*d, H] per slice; wk, wv: Weight [Hkv_loc*d, H];
    wo: Weight [H, Hq*d] sharded by rows; cos, sin: Input [d/2]; out: Output [1, H].
    """
    S = ol.num_programs()
    d = kv.d
    G = n_q_heads // n_kv_heads
    x = ol.load(h)
    xs = ol.quantize(rmsnorm(x, ol.load(gamma), eps))
    q = ol.dot(xs, wq)                          # [1, Hkv_loc*G*d]
    k = ol.dot(xs, wk)                          # [1, Hkv_loc*d]
    v = ol.dot(xs, wv)
    c, s_ = ol.load(cos), ol.load(sin)
    scale = ol.LOG2E / math.sqrt(d)
    heads = list(kv.owned_heads(n_kv_heads))
    o_loc = ol.empty([G * len(heads), d])       # this slice's attention outputs, head-major
    for j, hh in enumerate(heads):
        kh = rope(k[:, j * d:(j + 1) * d], c, s_)
        ol.kv_append(kv, hh, pos, kh, v[:, j * d:(j + 1) * d])
        qh = ol.empty([G, d])
        for g in ol.static_range(G):
            r = (j * G + g) * d
            qh[g:g + 1, :].set(rope(q[:, r:r + d], c, s_) * scale)
        o_loc[j * G:(j + 1) * G, :].set(_attend(qh, kv, hh, pos + 1, block))
    # flatten [G*heads_loc, d] -> [1, G*heads_loc*d] view, gather across slices, output proj
    o_row = ol.empty([1, G * len(heads) * d])
    for i in ol.static_range(G * len(heads)):
        o_row[:, i * d:(i + 1) * d].set(o_loc[i:i + 1, :])
    o_all = ol.all_gather(o_row)                # [1, Hq*d], slice-major head order
    y = ol.all_gather(ol.dot(o_all, wo))        # [1, H]
    if ol.program_id() == 0:
        ol.store(out, x + y)
