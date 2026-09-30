"""The decode loop on the card: one device run generates up to N tokens (docs/autodecode.md).

The generate program of an attention bucket wraps the model's resident decode step (its step
kernel at a run-time position, qwen3.RunPos; the step itself is unchanged) in a hardware loop:

    LD the state block (the token, the position's run-time values, the tokens left, stop ids)
    LOOP min(tokens left, block - tpos) times (RLD of the device-computed count):
        each run-time argument c * var: VOP MUL of the variable's state word, RLD into its
            argument register (the registers the host wrote before each run)
        the step: the LM head hands each logits chunk to the sampler (m.lm_sink)
        the sampler's token -> out[p + 1]  (fp32 ids, one word per position)
        stop: the token is a stop id, or the host's stop word is set -> LOOP stop {HALT}
        the state: tok = the token, tpos + 1, ring + 1 mod K, tokens left - 1
    ST the state block, HALT

The host writes the state block, marks out[], starts the run and reads the tokens from out[]
as they appear (Engine.generate_card); it never reads the logits. The ISA pieces are RLD (a
TMEM word to a register), VOP ARGMAX and LOOP with a register count (docs/isa.md).

Every run-time argument must be exact in fp32: c * var has at most 24 significant bits
(check_args), as the card computes it on the VPU.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from .. import fp32 as F
from .. import isa as I
from .. import language as ol
from ..compiler import CompileError, Tensor, current

# ---- the state block (fp32 words in DRAM, one per slice)
STATE_WORDS = 64
S_TOK, S_TPOS, S_RING, S_LEFT = 0, 1, 2, 3       # the run-time variables, the tokens left
S_K, S_ITEMP, S_TOPP = 4, 5, 6                    # sampling: top-k, log2(e) / T, top-p
S_STOP = 8                                        # 8 stop ids (-1: none)
N_STOP = 8
S_HALT = 16                                       # the host's stop word (1.0: stop)
S_PEN, S_PENINV = 17, 18                          # the repetition penalty R and 1 / R
SLOT = {"tok": S_TOK, "tpos": S_TPOS, "ring": S_RING}
OUT_MARK = 0xFFFFFFFF                             # out[] before the token lands (the device
                                                  # stores ids, never this NaN)


PROG_SLOT = 1 << 18        # bytes per bucket program in the chain area (8K instructions)
MODES = 2                  # chain areas: 0 greedy, 1 sampled (Samp)
BLK = 64                   # the sampler's block of logits (block maxima, gathers)
K_MAX = 64                 # the largest top-k the sampler takes


def buckets(cap: int, block: int) -> int:
    return -(-cap // block)


def _vpad(spec) -> int:
    return -(-spec.vocab // BLK) * BLK


def alloc(b, spec, cap: int, block: int) -> dict:
    """DRAM of the generate loop: the state block, out[] (one fp32 id per position), the
    bucket programs a run chains through (one slot per bucket and mode) and their tables
    ([address, instructions] per bucket, raw words, one table per mode); the sampler's
    logits (this slice's rows, -inf after them to a whole block), repetition penalty factors
    (pa, pb: 1 / R and R for the ids in the context, 1 for the others), uniforms (one per
    position, from the host) and constants (iota: 0 .. BLK - 1)."""
    nb = buckets(cap, block)
    V = _vpad(spec)
    return {"state": b.alloc(4 * STATE_WORDS), "out": b.alloc(4 * (cap + 1)),
            "ptab": b.alloc(8 * (nb + 2) * MODES), "progs": b.alloc(PROG_SLOT * nb * MODES),
            "lg": b.alloc(4 * V), "pa": b.alloc(4 * V), "pb": b.alloc(4 * V),
            "uni": b.alloc(4 * (cap + 1)), "iota": b.alloc(4 * BLK), "nb": nb}


def prog_slot(g: dict, blocks: int, mode: int = 0) -> int:
    """The chain area's slot of bucket `blocks` (1, 2, ...) in chain area `mode`."""
    return g["progs"] + PROG_SLOT * (mode * g["nb"] + blocks - 1)


def ptab_addr(g: dict, mode: int = 0) -> int:
    return g["ptab"] + 8 * (g["nb"] + 2) * mode


def build(put, s: int, S: int, spec, cap: int, g: dict) -> None:
    put(s, g["out"], np.full(cap + 1, OUT_MARK, np.uint32).view(np.float32))
    V, v_loc = _vpad(spec), spec.vocab // S
    lg = np.full(V, -np.inf, np.float32)
    put(s, g["lg"], lg[:-(-v_loc // BLK) * BLK])
    put(s, g["pa"], np.ones(V, np.float32))
    put(s, g["pb"], np.ones(V, np.float32))
    put(s, g["iota"], np.arange(BLK, dtype=np.float32))


def desc(g: dict, spec, cap: int) -> SimpleNamespace:
    V = _vpad(spec)
    return SimpleNamespace(state=Tensor(g["state"], (STATE_WORDS,), (1,)),
                           out=Tensor(g["out"], (cap + 1,), (1,)),
                           uni=Tensor(g["uni"], (cap + 1,), (1,)),
                           lg=Tensor(g["lg"], (V,), (1,)), pa=Tensor(g["pa"], (V,), (1,)),
                           pb=Tensor(g["pb"], (V,), (1,)), iota=Tensor(g["iota"], (BLK,), (1,)),
                           addr=g)


def ptab_words(g: dict, progs: dict, mode: int = 0) -> np.ndarray:
    """The chain table of `mode` for {blocks: assembled words}: [address, instructions] of
    bucket k at words 2k, 2k + 1 (absent buckets 0)."""
    t = np.zeros(2 * (max(progs) + 2), np.uint32)
    for k, w in progs.items():
        t[2 * k], t[2 * k + 1] = prog_slot(g, k, mode), len(w) // 8
    return t


# ---- the per-token variables
def rules(spec, block: int) -> dict:
    """How each run-time variable moves from one token to the next: tok is the sampled id;
    the others count (step, modulus). tpos = p mod block (it wraps to 0 as the last token of a
    bucket moves the position into the next one, whose program the run chains to); ring =
    (p + 1) mod K (LFM2's convolution ring, qwen3.RunPos)."""
    return {"tok": None, "tpos": (1, block), "ring": (1, getattr(spec, "conv_k", 1))}


def check_args(run_args, spec, block: int) -> None:
    """c * var must be exact in fp32 for every value var takes (the VPU computes it)."""
    bound = {"tok": spec.vocab, "tpos": block, "ring": getattr(spec, "conv_k", 1)}
    for v, c in run_args:
        if v.name not in bound:
            raise CompileError(f"run-time variable {v.name}: no update rule (generate.rules)")
        c = abs(int(c))
        odd = c >> ((c & -c).bit_length() - 1) if c else 0
        if (bound[v.name] - 1) * odd >= 1 << 24:
            raise CompileError(f"argument {c} * {v.name} is not exact in fp32")


def state_words(spec, tok: int, p: int, n: int, stop_ids, block: int,
                samp: "Sampling | None" = None) -> np.ndarray:
    """The state block for generating up to n tokens after token `tok` at position p."""
    st = np.zeros(STATE_WORDS, np.float32)
    ids = list(stop_ids)[:N_STOP]
    if len(stop_ids) > N_STOP:
        raise ValueError(f"at most {N_STOP} stop ids")
    st[S_TOK], st[S_TPOS] = tok, p % block
    st[S_RING] = (p + 1) % getattr(spec, "conv_k", 1)
    st[S_LEFT] = n
    st[S_STOP:S_STOP + N_STOP] = -1.0
    st[S_STOP:S_STOP + len(ids)] = ids
    if samp is not None:
        st[S_K], st[S_ITEMP], st[S_TOPP] = samp.k, samp.itemp, samp.top_p
        st[S_PEN], st[S_PENINV] = samp.penalty, np.float32(1.0) / np.float32(samp.penalty)
    return st


class Sampling:
    """The sampled decode loop's parameters, as chat.sampler takes them: temperature (0:
    greedy, with the penalty), top_k (1 .. K_MAX), top_p, repetition_penalty (>= 1). kmax,
    penalty: what the program is compiled for (the candidate buffers; the penalty's loads);
    the rest are run-time words of the state block."""

    def __init__(self, temperature: float, top_k: int, top_p: float = 1.0,
                 repetition_penalty: float = 1.0):
        greedy = temperature <= 0
        if not greedy and not 0 < top_k <= K_MAX:
            raise ValueError(f"the sampler on the device takes top_k 1 .. {K_MAX}")
        if repetition_penalty < 1.0:
            raise ValueError("the sampler on the device takes repetition_penalty >= 1")
        self.k = 1 if greedy else int(top_k)
        self.itemp = 1.0 if greedy else math.log2(math.e) / temperature
        self.top_p = 1.0 if greedy else float(top_p)
        self.penalty = float(repetition_penalty)
        self.kmax = -(-self.k // 8) * 8
        self.pen = self.penalty != 1.0

    @property
    def key(self) -> tuple:
        return (self.kmax, self.pen)

    @staticmethod
    def fits(temperature: float, top_k: int, top_p: float, repetition_penalty: float) -> bool:
        return (temperature <= 0 or 0 < top_k <= K_MAX) and repetition_penalty >= 1.0


# ---- the sampler
class Greedy:
    """The LM head's sink: each chunk's ARGMAX while the next chunk streams, then the ARGMAX
    of the chunk maxima picks the chunk and its index (ties: the first id, as np.argmax)."""

    def __init__(self, b, vocab: int, chunk: int):
        self.b, self.chunk = b, chunk
        self.nc = -(-vocab // chunk)
        self.cm = b.alloc((self.nc, 2))              # per chunk (max, id)
        self.seen = 0

    def __call__(self, y, col0: int) -> None:
        """y: the logits of vocabulary rows col0.. (the chunks come in order)."""
        k = self.seen
        if k >= self.nc:
            raise CompileError("the LM head gave more chunks than its rows")
        self.seen += 1
        self.b.argmax(y, base=col0, out=self.cm[k:k + 1, :])

    def token(self):
        b = self.b
        if self.seen != self.nc:
            raise CompileError(f"the LM head gave {self.seen} of {self.nc} chunks")
        best = b.argmax(_col(b, self.cm, 0))          # [max, chunk]
        tok = _pick(b, best[1:2], 2, self.cm.base + 1, "the best chunk's id")
        if b.S > 1:                                   # the slices' vocabulary rows, in order
            ix = ol.all_gather(tok)
            best = b.argmax(ol.all_gather(best[0:1]))
            tok = _pick(b, best[1:2], 1, ix.base, "the best slice's id")
        return tok


NEG = -math.inf
BIG = 2.0 ** 126                 # step(): x * BIG >= 1 for every normal x > 0


def _step(x):
    """1.0 where x > 0, else 0.0 (denormals count as 0: the VPU flushes them)."""
    return ol.minimum(ol.maximum(x * BIG, 0.0), 1.0)


def _fill(b, t, v: float, ra: int = 0, n: int | None = None, comment: str = "") -> I.Instr:
    return I.vop(I.V_FILL, t.base, 0, 0, 1, t.cols if n is None else n, 0, 0, 0, I.B_SCALAR, v,
                 ra=ra, comment=comment or "fill")


def _loop(b, rcount: int, body: list) -> None:
    b.emit(I.loop(len(body), 0, rcount=rcount))
    for ins in body:
        b.emit(ins)


def _select(b, vals, ids, kreg: int, out_v, out_id) -> None:
    """The R[kreg] largest of vals (a 1-D tile; knocked out to -inf as they are taken) in
    descending order into out_v, and their ids (the same positions of `ids`) into out_id:
    ARGMAX, the position by RLD, knock-out, the pair's value and id to the next outputs.
    Ties: the first position."""
    pr = b.alloc((2,))
    ri, rq = b.scratch(), b.scratch()
    _loop(b, kreg, [
        I.argmax(pr.base, vals.base, 1, vals.cols, comment="select"),
        I.rld(rq, pr.base + 1, comment="its position"),
        _fill(b, vals, NEG, ra=rq, n=1, comment="knock out"),
        I.vop(I.V_COPY, out_v.base, pr.base, 0, 1, 1, 0, 0, 0, ra=ri, comment="value"),
        I.vop(I.V_COPY, out_id.base, ids.base, 0, 1, 1, 0, 0, 0, ra=ri, rb=rq, comment="id"),
        I.addi(ri, ri, 1)])
    b.unscratch(ri)
    b.unscratch(rq)


class Sampler:
    """The LM head's sink of the sampled loop (chat.sampler's pick on the device):

    each chunk: the repetition penalty (min(l * pa, l * pb): l / R for l > 0 and l * R for
    l < 0 for the context's ids, as Hugging Face's), the logits to DRAM (lg) and the maxima of
    their blocks of BLK;
    token(): the k blocks with the largest maxima (ARGMAX, knock-out) gathered from lg, the k
    largest logits among them (their order: descending, ties by position), with S > 1 the k
    largest of all slices' (all_gather); p = exp2((l - l0) * log2(e) / T), the cumulative sums
    (RDOT with a triangular matrix), top-p (keep the first i with sum(p[:i]) < P * sum(p)), the
    pick = #{cum <= u * cum[last kept]} for the position's uniform u (at most the last kept),
    its id. after(tok): tok joins the penalty's context."""

    def __init__(self, b, m, g, samp: Sampling, st, pos, consts):
        self.b, self.m, self.g, self.samp, self.st, self.pos = b, m, g, samp, st, pos
        self.U, self.iota = consts
        self.chunk_seen = 0
        v = m.v_loc
        self.o = ol.program_id() * v             # this slice's first vocabulary row
        self.nb = -(-v // BLK)
        self.bm = b.alloc((self.nb,))            # block maxima

    @staticmethod
    def constants(b, g, samp: Sampling):
        """Before the token loop: the triangular [kmax, kmax] ones (U[r, c] = c <= r) from
        iota, and iota."""
        km = samp.kmax
        iota = ol.load(g.iota)
        col = iota[0:km]
        U = b.alloc((km, km))
        # U[r, c] = step(r + 1 - c): A = iota per column (ars 0), B = iota + 1 per row
        r1 = col + 1.0
        b.emit(I.vop(I.V_RSUB, U.base, col.base, r1.base, km, km, km, 0, 1, I.B_ROW,
                     comment="r + 1 - c"))
        U.set(_step(U))
        return U, iota

    def __call__(self, y, col0: int) -> None:
        b, g, m = self.b, self.g, self.m
        self.chunk_seen += 1
        y = y[0, :] if len(y.shape) == 2 else y
        n = y.cols
        c0 = col0 - self.o                       # this slice's row
        if self.samp.pen:
            pa, pb = ol.load(g.pa[col0:col0 + n]), ol.load(g.pb[col0:col0 + n])
            pa.set(y * pa)
            pb.set(y * pb)
            pa.set(ol.minimum(pa, pb))
            y = pa
        ol.store(g.lg[c0:c0 + n], y)
        nf, k0 = n // BLK, c0 // BLK
        if nf:
            self.bm[k0:k0 + nf].set(ol.max(y[0:nf * BLK].reshape(nf, BLK)))
        if n % BLK:
            self.bm[k0 + nf:k0 + nf + 1].set(ol.max(y[nf * BLK:n]))

    def token(self):
        b, g, st, samp = self.b, self.g, self.st, self.samp
        km = samp.kmax
        # the blocks: k1 = min(k, blocks) of them, each block's logits and ids gathered
        cand, ids = b.alloc((km * BLK,)), b.alloc((km * BLK,))
        b.emit(_fill(b, cand, NEG, comment="candidates"))
        kk = st[S_K:S_K + 1]
        k1 = ol.minimum(kk, float(self.nb))
        pr, off, base = b.alloc((2,)), b.alloc((1,)), b.alloc((1,))
        rk, rj, rb_, ro = b.scratch(), b.scratch(), b.scratch(), b.scratch()
        b.rld(rk, k1, comment="blocks")
        _loop(b, rk, [
            I.argmax(pr.base, self.bm.base, 1, self.nb, comment="block"),
            I.rld(rb_, pr.base + 1, comment="its index"),
            _fill(b, self.bm, NEG, ra=rb_, n=1, comment="knock out"),
            I.vop(I.V_MUL, off.base, pr.base + 1, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 4.0 * BLK,
                  comment="block bytes"),
            I.vop(I.V_MUL, base.base, pr.base + 1, 0, 1, 1, 0, 0, 0, I.B_SCALAR, float(BLK),
                  comment="block id"),
            I.vop(I.V_ADD, base.base, base.base, 0, 1, 1, 0, 0, 0, I.B_SCALAR, float(self.o),
                  comment="+ the slice's first row"),
            I.rld(ro, off.base),
            I.ld(g.addr["lg"], cand.base, BLK, ra=ro, rb=rj, comment="the block's logits"),
            I.vop(I.V_ADD, ids.base, self.iota.base, base.base, 1, BLK, 0, 0, 0, I.B_ROW,
                  ra=rj, comment="their ids"),
            I.addi(rj, rj, BLK)])
        for r in (rj, rb_, ro):
            b.unscratch(r)
        b.rld(rk, kk, comment="top-k")
        vals, sid = b.alloc((km,)), b.alloc((km,))
        vals.set(NEG)
        sid.set(0.0)
        _select(b, cand, ids, rk, vals, sid)
        if b.S > 1:                             # the k largest of all slices
            gv, gi = ol.all_gather(vals), ol.all_gather(sid)
            vals.set(NEG)
            _select(b, gv, gi, rk, vals, sid)
        b.unscratch(rk)
        # softmax, the cumulative sums, top-p, the pick
        p = ol.exp2((vals - vals[0:1]) * st[S_ITEMP:S_ITEMP + 1])
        cumx = b.alloc((km + 1,))
        cumx[0:1].set(0.0)
        b.emit(I.vop(I.V_RDOT, cumx.base + 1, self.U.base, p.base, km, km, 1, km, 0, I.B_COL,
                     comment="cumulative sums"))
        cum, prev = cumx[1:km + 1], cumx[0:km]
        pz = cumx[km:km + 1] * st[S_TOPP:S_TOPP + 1]
        kept = _step(pz - prev)
        zk = ol.max(cum * kept)
        u = ol.load(g.uni[self.pos.pos + 1:self.pos.pos + 2])
        t = u * zk
        le = ol.sum(_step(cum - t)) * -1.0 + float(km)
        pick = ol.minimum(le, ol.sum(kept) - 1.0)
        return _pick(b, pick, 1, sid.base, "the sampled id")

    def after(self, tok) -> None:
        """The penalty's context gets tok: pa[tok] = 1 / R, pb[tok] = R."""
        if not self.samp.pen:
            return
        b, g, st = self.b, self.g, self.st
        r = b.scratch()
        b.rld(r, tok * 4.0, comment="the id's bytes")
        b.emit(I.st(g.addr["pa"], st.base + S_PENINV, 1, ra=r, comment="pa[tok] = 1 / R"))
        b.emit(I.st(g.addr["pb"], st.base + S_PEN, 1, ra=r, comment="pb[tok] = R"))
        b.unscratch(r)


def reference_pick(logits, samp: Sampling, context, u, S: int = 1):
    """The id the device's sampler picks from `logits` (the whole vocabulary) with uniform u,
    bit for bit: Sampler's steps in numpy with the ISA's fp32 arithmetic (opentpu.fp32). An
    array of uniforms gives an array of ids."""
    f32 = np.float32
    lg = F.ftz(np.asarray(logits, f32)).copy()
    V = len(lg)
    if samp.pen:
        a, b = np.ones(V, f32), np.ones(V, f32)
        ix = np.unique(np.asarray(list(context), np.int64))
        a[ix], b[ix] = f32(1.0) / f32(samp.penalty), samp.penalty
        lg = F.fmin(F.mul(lg, a), F.mul(lg, b))
    km, k, v = samp.kmax, samp.k, V // S

    def select(vals, ids, n_out, out_v, out_i):
        vals = vals.copy()
        for i in range(n_out):
            q = int(np.argmax(F._key(vals)))
            out_v[i], out_i[i] = F.ftz(vals[q]), ids[q]
            vals[q] = -np.inf

    vals, sid = [], []
    for s in range(S):
        nb = -(-v // BLK)
        x = np.full(nb * BLK, -np.inf, f32)
        x[:v] = lg[s * v:(s + 1) * v]
        bm = F.chain_max(x.reshape(nb, BLK))
        cand, ids = np.full(km * BLK, -np.inf, f32), np.zeros(km * BLK, f32)
        for j in range(min(k, nb)):
            q = int(np.argmax(F._key(bm)))
            bm[q] = -np.inf
            cand[j * BLK:(j + 1) * BLK] = x[q * BLK:(q + 1) * BLK]
            ids[j * BLK:(j + 1) * BLK] = F.add(np.arange(BLK, dtype=f32), f32(q * BLK + s * v))
        vs, si = np.full(km, -np.inf, f32), np.zeros(km, f32)
        select(cand, ids, k, vs, si)
        vals.append(vs)
        sid.append(si)
    vs, si = vals[0], sid[0]
    if S > 1:
        vs = np.full(km, -np.inf, f32)
        select(np.concatenate(vals), np.concatenate(sid), k, vs, si)

    def step(x):
        return F.fmin(F.fmax(F.mul(x, f32(BIG)), f32(0)), f32(1))

    p = F.exp2(F.mul(F.sub(vs, vs[0]), f32(samp.itemp)))
    c = np.arange(km)
    U = step(F.sub(F.add(c.astype(f32), f32(1))[:, None], c.astype(f32)[None, :]))
    cum = F.rdot(U, np.broadcast_to(p, (km, km)))
    prev = np.concatenate([[f32(0)], cum[:-1]]).astype(f32)
    kept = step(F.sub(F.mul(cum[-1], f32(samp.top_p)), prev))
    zk = F.chain_max(F.mul(cum, kept)[None, :])[0]
    uu = np.minimum(np.atleast_1d(np.asarray(u, f32)), f32(1 - 2.0 ** -24))
    t = F.mul(uu, zk)[:, None]

    def total(x):
        return F.interleaved_sum(np.atleast_2d(x), F.RED_PARTIALS)
    le = F.add(F.mul(total(step(F.sub(cum[None, :], t))), f32(-1)), f32(km))
    pick = F.fmin(le, F.sub(total(kept), f32(1)))
    ids = si[pick.astype(np.int64)].astype(np.int64)
    return ids if np.ndim(u) else int(ids[0])


def _col(b, t, c: int):
    """Column c of a [rows, cols] tile as a new contiguous [rows] tile."""
    out = b.alloc((t.rows,))
    b.emit(I.vop(I.V_COPY, out.base, t.base + c, 0, t.rows, 1, 1, t.rs, 0, comment="column"))
    return out


def _pick(b, k, stride: int, base: int, what: str):
    """A new [1] tile holding T[base + stride * k] (k a tile holding a small integer)."""
    if stride != 1:
        k = k * float(stride)
    r = b.scratch()
    b.rld(r, k)
    out = b.alloc((1,))
    b.emit(I.vop(I.V_COPY, out.base, base, 0, 1, 1, 0, 0, 0, rb=r, comment=what))
    b.unscratch(r)
    return out


# ---- the program
def _generate(m, pos, block, step, spec, chain, samp=None):
    b = current()
    g = m.gen
    st = ol.load(g.state)
    consts = Sampler.constants(b, g, samp) if samp is not None else None
    args = b.alloc((8,))               # the run-time arguments' words (c * var)
    left = st[S_LEFT:S_LEFT + 1]
    n = ol.minimum(left, (st[S_TPOS:S_TPOS + 1] * -1.0) + float(block))  # to the bucket's end
    r = b.scratch()
    b.rld(r, n, comment="tokens in this bucket")
    loop = b.begin_loop(0, rcount=r)
    b.unscratch(r)                     # the count is read when the loop starts
    body = b.stack[-1]
    mark = len(body)                   # the argument loads go here (known after the step)
    chunk = min(8192, b.cfg.TMEM_WORDS // 8)       # qwen3.HEAD_CHUNK, _lm_head's chunks
    sink = (Greedy(b, m.v_loc, chunk) if samp is None
            else Sampler(b, m, g, samp, st, pos, consts))
    m.lm_sink = sink
    step.fn(m=m, pos=pos, block=block)
    tok = sink.token()
    ol.store(g.out[pos.pos + 1:pos.pos + 2], tok)
    if samp is not None:
        sink.after(tok)
    # stop: the token is one of the stop ids, or the host asks
    ids = st[S_STOP:S_STOP + N_STOP]
    hit = ol.minimum(ol.abs(ids - tok), 1.0) * -1.0 + 1.0     # 1 where equal (integers)
    halt = ol.load(g.state[S_HALT:S_HALT + 1])
    stop = ol.maximum(ol.max(hit), halt)
    r = b.scratch()
    b.rld(r, stop, comment="stop")
    b.emit(I.loop(1, 0, rcount=r, comment="stop: halt"))
    b.emit(I.halt())
    b.unscratch(r)
    # the next token's state
    rl = rules(spec, block)
    st[S_TOK:S_TOK + 1].set(tok)
    used = {v.name for v, _ in b.run_args}
    for name, slot in SLOT.items():
        rule = rl[name]
        if rule is None or name not in used:
            continue
        stepv, mod = rule
        v = st[slot:slot + 1]
        v.set(v + float(stepv))
        if mod:            # v - mod * [v >= mod], exact for small integers
            w = ol.minimum(ol.maximum(v + float(1 - mod), 0.0), 1.0)
            v.set(v - w * float(mod))
    left.set(left - 1.0)
    # the run-time arguments of this iteration, now that the step has named them all
    check_args(b.run_args, spec, block)
    loads = []
    for k, (v, c) in enumerate(b.run_args):
        loads.append(I.vop(I.V_MUL, args.base + k, st.base + SLOT[v.name], 0, 1, 1, 0, 0, 0,
                           I.B_SCALAR, float(c), comment=f"{c}*{v.name}"))
        loads.append(I.rld(15 - k, args.base + k, comment=f"argument {c}*{v.name}"))
    body[mark:mark] = loads
    b.end_loop(loop)
    ol.store(g.state[:S_HALT], st[:S_HALT])      # not the host's stop word
    if chain:              # tokens left: on to the next bucket's program (chain table)
        more = ol.minimum(left, 1.0)
        tab = b.alloc((2,))
        r, ra, rb = b.scratch(), b.scratch(), b.scratch()
        b.rld(r, more, comment="tokens left")
        b.emit(I.loop(4, 0, rcount=r, comment="chain"))
        b.emit(I.ld(ptab_addr(g.addr, int(samp is not None)) + 8 * (pos.blocks + 1), tab.base,
                    2, comment="next bucket"))
        b.emit(I.rld(ra, tab.base, raw=True))
        b.emit(I.rld(rb, tab.base + 1, raw=True))
        b.emit(I.halt(chain=True, ra=ra, rb=rb))
        for x in (r, ra, rb):
            b.unscratch(x)


def compile_generate(image, kernel, blocks: int, lo: int, block: int, chain: bool = True,
                     samp: Sampling | None = None):
    """The generate program of positions [lo, blocks * block) (compile_decode's bucket):
    the step kernel at a RunPos in the token loop (see the module), greedy or sampled (samp),
    then, with `chain` and tokens left, HALT CHAIN to the next bucket's program (the entry
    blocks + 1 of the mode's table). Returns the programs."""
    from .qwen3 import RunPos
    if not image.lookup:
        raise ValueError("compile_generate needs an image with lookup tables (lookup=True)")
    if not (blocks - 1) * block <= lo < min(blocks * block, image.cap):
        raise ValueError(f"lo {lo} is not in bucket {blocks}")
    rp = RunPos(blocks, block, lo, image.lookup["zmask"], image.cap)
    progs = []
    for s in range(image.cfg.S):
        m = image.descriptors(s)
        progs.append(ol.jit(_generate).trace(image.cfg, s, {"m": m, "pos": rp, "block": block,
                                                            "step": kernel, "chain": chain,
                                                            "spec": image.spec,
                                                            "samp": samp}).finish())
    return progs
