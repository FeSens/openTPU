"""Decode attention over the int8 KV cache (flash-style online softmax), GQA, head-parallel."""
import copy
import math

from .. import isa as I
from .. import language as ol
from ..compiler import Affine, DevVar, Tensor, current
from .lib import rmsnorm, rope


class Bucket:
    """A sequence length known only at run time (one program for every position of a bucket):
    `blocks` blocks of `block` tokens, the last one partly past the sequence. `z` is the
    address (bytes; run-time: it adds -4 * position) of a float row whose entry for token t0 + c
    of a block at t0 is +inf when the token is in the sequence and -inf past it: that block's
    scores are min(s, row), so the tokens past the sequence score -inf and weigh exactly +0 --
    the softmax, its sums and P.V come out bit for bit as with the partial block. With `add`
    (a prompt run's row: Additive) that block's scores are the mask tile plus q.K^T instead.
    The masked block ends at the cache's end: with a capacity that is not a multiple of the
    block it spans cap - t0 tokens (a multiple of D), so it reads nothing past the cache.
    tpos (a decode position's run-time value, qwen3.RunPos: t0 + tpos the token): the masked
    block reads only its first D, 2D, ... tokens, the fewest that hold tokens 0 .. tpos
    (_Variants), so it streams no K row and no V^T column past them: the tokens past tpos
    weigh +0 there, so the shorter block is the same bit for bit."""

    def __init__(self, blocks: int, z, add: "Additive | None" = None, tpos=None):
        self.blocks, self.z, self.add, self.tpos = blocks, z, add, tpos

    def row(self, t0: int, n: int) -> Tensor:
        return Tensor(self.z + 4 * t0, (n,), (1,))


SPAN_P = 2          # RunSpan's blocks a group (2: the 35B's 131,072-token cache is 512 blocks)


class RunSpan:
    """A sequence whose length is known only at run time, its blocks too (one program for every
    position: batch35's step over slots, docs/batch35.md): the
    position's mask row as Bucket's (`z`: -4 * position at run time), and the attention over
    the first `groups` x P blocks, groups the integer in TMEM word `count` (ceil((position +
    1) / (P block)), 1 .. `bound`) read as the loop of the groups starts. Every block is
    masked by its row: min(s, row) is s where the token counts (+inf) and -inf past it, so a
    block that ends at or before the position scores as an unmasked one, and one wholly past it
    leaves the online softmax's state as it was (m stays, alpha is exactly 1, p and P.V are
    +0): bit for bit the exact blocks' (Bucket's). The table needs cap + P block entries, the
    cache a multiple of P blocks (a group is P whole V^T tiles)."""

    def __init__(self, z, count: int, bound: int, P: int = SPAN_P):
        self.z, self.count, self.bound, self.P = z, count, bound, P

    def row(self, t0, n: int) -> Tensor:
        return Tensor(self.z + 4 * t0, (n,), (1,))


class Additive:
    """A masked block's mask as a tile the scores are added to (docs/prefill.md 9): at `addr`
    (bytes; run-time: affine in the position) MCOLS rows of the block's `block` entries and
    one pad word (the score buffer's row stride), -0 where the token counts and -inf past it.
    The block's score buffer is loaded with it, then q.K^T is accumulated into it with the row
    maxima (MM ACC + RMAX): two instructions where min(s, row) takes four (MM, LD, VOP MIN,
    VOP RMAX), bit for bit the same (s + -0 = s, s + -inf = -inf = min(s, -inf)). A masked
    block of n < block tokens (Bucket: the cache's end) takes each row's first n entries,
    loaded row by row."""

    def __init__(self, addr, block: int):
        self.addr, self.block = addr, block

    def tile(self, rows: int, n: int) -> Tensor:
        return Tensor(self.addr, (rows, n), (self.block + 1, 1))


class Blocks:
    """Attention over an explicit list of cache blocks, in order: [(t0, n, mask)], tokens [t0,
    t0 + n) of the cache, mask None (all of them count) or a DRAM row of n floats, +inf where
    the token counts and -inf where not (the block's scores are min(s, row), as Bucket's last
    block), or an Additive tile. A sliding window over a KV ring (Gemma 4: the window's first
    block masked at its start, its last at its end). tpos (a run-time value, as Bucket's): the
    last block's tokens past its first tpos + 1 are masked (a ring's first lap), so it is read
    as Bucket's masked block, its first D, 2D, ... tokens."""

    def __init__(self, items, tpos=None):
        self.items = [(t0, n, m) for t0, n, m in items]
        self.tpos = tpos


class RowMasks:
    """The masked block of query rows packed from several tokens at compile-time positions
    (a tiled prefill run, opentpu/llm/tiled.py): `keep` [(r0, r1, k)], rows r0 .. r1 - 1 of the
    score buffer count the block's first k tokens. The buffer is filled with -0, then -inf
    past each row group's k, and q.K^T is accumulated into it with the row maxima (MM ACC +
    RMAX), as an Additive tile: s + -0 = s, s + -inf = -inf. A row group with k = 0 sees no
    token of the block: its scores are all -inf, so its online softmax state passes the block
    unchanged (m stays, p and the P.V terms are +0, alpha is exactly 1)."""

    def __init__(self, keep):
        self.keep = [(r0, r1, k) for r0, r1, k in keep]

    def fill(self, out, n: int) -> None:
        out.set(-0.0)
        for r0, r1, k in self.keep:
            if k < n:
                out[r0:r1, k:n].set(float("-inf"))


class RowTiles:
    """RowMasks at run-time positions (a tiled prompt run): `tiles` [(r0, r1, add)], rows r0 ..
    r1 - 1 of the score buffer take the Additive tile `add` (their own position's row of the
    image's mask table, as each row's Bucket of a prompt run takes it). Each row group's rows
    are loaded from its tile, then q.K^T is accumulated into the buffer with the row maxima:
    every row's scores bit for bit those of its own Additive block."""

    def __init__(self, tiles):
        self.tiles = [(r0, r1, a) for r0, r1, a in tiles]

    def fill(self, out, n: int) -> None:
        for r0, r1, a in self.tiles:
            ol.load(a.tile(r1 - r0, n), out=out[r0:r1, :])


class TileSeq:
    """Blocks of a packed query tile (tiled prefill): `full` whole blocks every row sees (the
    hardware loop's, as a length's), then `items` [(t0, n, mask)] as Blocks' (a mask: RowMasks,
    or None where every row sees the block whole)."""

    def __init__(self, full: int, items):
        self.full = full
        self.items = [(t0, n, m) for t0, n, m in items]


def _attend(qh, kv, h, seq_len: int, block: int, scale: float | None = None, raw: bool = False,
            depth: int = 3):
    """Flash attention of the G query rows `qh` [G, d] against KV head h (see _attend_heads);
    returns acc / l, or the unnormalized (acc, l) with `raw`."""
    (acc, l), = _attend_heads([qh], kv, [h], seq_len, block, scale, depth)
    if raw:
        return acc, l
    return acc / l[:, None]


class _Variants:
    """A masked block read in sub-blocks of D tokens at a run-time position tpos: variant k
    takes its first n_k = (k + 1) D tokens (the last one all `n` of them) and runs where
    k D <= tpos < (k + 1) D (the last: tpos >= k D), so a run reads the fewest sub-blocks that
    hold tokens 0 .. tpos. Each variant's work goes in a loop of 0 or 1 iterations, its count
    a word of a K-word table (1.0 where the variant runs: from 2^23 + tpos, a FILL of tpos's
    argument register, once a program) read by RLD as the loop starts."""

    def __init__(self, tpos, n: int, D: int):
        self.tpos, self.K = tpos, n // D
        self.ns = [(k + 1) * D for k in range(self.K - 1)] + [n]
        b = current()
        tabs = b.__dict__.setdefault("_variant_counts", {})
        key = (tpos.name, D, self.K)
        self.tab = tabs.get(key)
        if self.tab is None:
            two23 = float(1 << 23)
            w = ol.empty((1,))
            b.emit(I.vop(I.V_FILL, w.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, two23,
                         rd=b.arg_reg(tpos, 1), comment=f"2^23 + {tpos.name}"))
            self.tab = tabs[key] = b.alloc((self.K,), top=True)      # kept: apart
            for k in range(self.K):
                x = w - (two23 + k * D - 1)                        # tpos - k D + 1
                if k < self.K - 1:
                    x = ol.minimum(x, (two23 + (k + 1) * D) - w)   # (k + 1) D - tpos
                self.tab[k:k + 1].set(ol.maximum(ol.minimum(x, 1.0), 0.0))

    def begin(self, k: int):
        """Variant k's loop (0 or 1 iterations): its word into a count register, free again
        in the body (LOOP reads it as it starts)."""
        b = current()
        b.check_live(self.tab)
        r = b.scratch()
        ra, imm = b.addr(Affine(self.tab.base + k))
        b.emit(I.rld(r, imm, ra=ra, bound=(0, 1), variant=(k, self.K) if ra == 0 else None,
                     comment=f"tokens {self.ns[k]}?"))
        lp = b.begin_loop(0, rcount=r)
        b.unscratch(r)
        return lp


def variants_ahead(seq_len, kv, block: int) -> None:
    """The masked block's _Variants table (a decode bucket's: a run-time position) made now, as
    _Head would make it (its key's, once a program): before a slice-dependent region where only
    some slices' heads have that block (_Head's part), so every slice allocates it."""
    if not isinstance(seq_len, Bucket) or seq_len.add is not None or seq_len.tpos is None:
        return
    D = ol.block_size()
    last = min(block, kv.cap - (seq_len.blocks - 1) * block)
    if last > D and last % D == 0:
        _Variants(seq_len.tpos, last, D)


class _Head:
    """Online-softmax state of one KV head's query group; with `sink` (a [G] tile: each query's
    sink logit in the scores' units, GPT-OSS's) the softmax's extra term exp2(sink) starts it:
    m = sink, l = 1, no value. part (p, r): the head's positions split over r slices that hold
    copies of it (a length or a Bucket: _attend_heads' part), this one attending blocks p, p + r,
    p + 2 r, .. alone: boff + k bstride the k-th one's first token."""

    def __init__(self, qh, kv, h, scale, seq_len, block, depth, sink=None, lazy=False,
                 part=None):
        G, d = qh.shape
        self._q = (qh, scale)
        self._qs = None
        if not lazy:
            self.quantize()
        self.G = G
        self.K, self.VT, self.VS = kv.k(h), kv.vt(h), kv.vscale(h)
        if sink is None:
            self.m = ol.full([G], -1e30)
            self.l = ol.zeros([G])
        else:
            self.m = ol.empty([G])
            self.m.set(sink)
            self.l = ol.full([G], 1.0)
        self.acc = ol.zeros([G, self.VT.shape[0]])   # V's head_dim (q and K may be padded)
        self.buf = {}                                # block index -> score buffer
        self.S = None                                # the hardware loop's score buffers
        self.rowmax = {}                             # masked block t0 -> its scores' row maxima
        self.masked = None                           # (t0, Bucket) of the masked block
        self.masks = {}                              # Blocks: t0 -> the block's mask row
        self.var = None                              # the last block's _Variants
        self.run = None                              # a RunSpan: _attend_heads' run loop
        tpos = None
        if isinstance(seq_len, RunSpan):             # groups of masked blocks, a run-time count
            if part is not None or kv.cap % (seq_len.P * block):
                raise ol.CompileError(f"a run-time span: whole heads, a cache of whole groups of "
                                      f"{seq_len.P} blocks ({kv.cap})")
            nfull, self.blocks, self.run = 0, [], seq_len
        elif isinstance(seq_len, Bucket):            # full blocks and a masked last one
            nfull = seq_len.blocks - 1
            last = min(block, kv.cap - nfull * block)    # it ends at the cache's end
            if last <= 0:
                raise ol.CompileError(f"bucket {seq_len.blocks} starts past the cache ({kv.cap})")
            self.blocks = [(i * block, block) for i in range(nfull)] + [(nfull * block, last)]
            self.masked = (nfull * block, seq_len)
            if seq_len.add is None:
                tpos = seq_len.tpos
        elif isinstance(seq_len, Blocks):            # the list, unrolled
            nfull = 0
            self.blocks = [(t0, n) for t0, n, _ in seq_len.items]
            self.masks = {t0: m for t0, _, m in seq_len.items if m is not None}
            if isinstance(self.masks.get(self.blocks[-1][0]), Tensor):
                tpos = seq_len.tpos
        elif isinstance(seq_len, TileSeq):           # whole blocks (looped), then the list
            nfull = seq_len.full
            self.blocks = [(i * block, block) for i in range(nfull)] + \
                [(t0, n) for t0, n, _ in seq_len.items]
            self.masks = {t0: m for t0, _, m in seq_len.items if m is not None}
        else:
            nfull, tail = divmod(seq_len, block)
            self.blocks = [(i * block, block) for i in range(nfull)] + \
                ([(nfull * block, tail)] if tail else [])
        self.boff, self.bstride = 0, block            # the k-th block at boff + k bstride
        if part is not None:
            p, r = part
            if not isinstance(seq_len, (int, Bucket)):
                raise ol.CompileError("a split head's length is a length or a Bucket")
            own = list(range(p, len(self.blocks), r))
            if not own:
                raise ol.CompileError(f"copy {p} of {r} has no block of {len(self.blocks)}")
            if own[-1] != len(self.blocks) - 1:      # the masked (or tail) block is another's
                self.masked, tpos = None, None
            nfull = sum(1 for i in own if i < nfull)
            self.blocks = [self.blocks[i] for i in own]
            self.boff, self.bstride = p * block, r * block
        self.groups = max(0, (nfull + 1 - depth) // depth)
        D = ol.block_size()
        if tpos is not None and self.blocks[-1][1] > D and self.blocks[-1][1] % D == 0:
            self.var = _Variants(tpos, self.blocks[-1][1], D)
        if self.VT.tiles is not None:            # V^T in tiles of tb tokens (KVDesc)
            tb = self.VT.tiles[0]
            if tb % block:
                raise ol.CompileError(f"attention block {block} must divide the V^T tile {tb}")
            if (depth * self.bstride) % tb:      # a loop step must be whole tiles
                self.groups = 0
            if self.run is not None and (self.run.P * block) % tb:
                raise ol.CompileError(f"a run-time span's group of {self.run.P} blocks: whole "
                                      f"V^T tiles of {tb}")

    def quantize(self):
        """The queries into ACT RAM (q stays stationary for every K block), once."""
        if self._q is not None:
            qh, scale = self._q
            self._q = None
            self._qs = ol.quantize(qh if scale is None
                                   else qh * ol.full([qh.shape[1]], scale)[None, :])
        return self._qs

    @property
    def qs(self):
        return self.quantize()

    @qs.setter
    def qs(self, v):
        self._q, self._qs = None, v


def _scores(st, t0, n, out, key=None, qs=None):
    """s = q.K^T for tokens [t0, t0+n) of head state `st` (_Head) into `out`; the MXU epilogue
    also writes the row maxima (a masked block: min with its mask row, then the row maxima of
    that; an Additive mask: the mask tile plus q.K^T, the maxima from the MXU epilogue too).
    key: the masked block's row maxima's (default t0; a variant's (t0, k)); qs: the queries
    (default st.qs)."""
    row = None
    key = t0 if key is None else key
    qs = st.qs if qs is None else qs
    if isinstance(t0, int):
        row = st.masks.get(t0)
        if isinstance(row, Tensor) and row.shape[0] > n:      # a variant's first n tokens
            row = row[0:n]
        if st.masked is not None and st.masked[0] == t0:
            bk = st.masked[1]
            row = bk.row(t0, n) if bk.add is None else bk.add
    if isinstance(row, Additive):                   # the mask tile, then q.K^T added
        ol.load(row.tile(st.G, n), out=out)
        ol.dot(st.qs, st.K[t0:t0 + n, :], acc=out, rowmax=True)
        return out
    if isinstance(row, (RowMasks, RowTiles)):       # built in place, then q.K^T added
        row.fill(out, n)
        ol.dot(st.qs, st.K[t0:t0 + n, :], acc=out, rowmax=True)
        return out
    if row is not None:
        ol.dot(qs, st.K[t0:t0 + n, :], out=out)
        s = ol.minimum(out, ol.load(row)[None, :])
        st.rowmax[key] = ol.max(s, axis=1)
        return s
    ol.dot(qs, st.K[t0:t0 + n, :], out=out, rowmax=True)
    return out


def _finish(st, s, t0, n, key=None, rm=None):
    """Online-softmax update and acc += P.V for the block whose scores are in `s` (key: its row
    maxima's, as _scores'; rm: the maxima's tile itself)."""
    D = ol.block_size()
    G = st.G
    key = t0 if key is None else key
    vs = ol.load(st.VS[t0:t0 + n])
    if rm is None:
        rm = st.rowmax.pop(key) if isinstance(t0, int) and key in st.rowmax else s.rowmax
    m_new = ol.maximum(st.m, rm)
    p = ol.exp2(s - m_new[:, None], rowsum=True)              # fused EXP2SUB (FSUM: + sum(p))
    alpha = ol.exp2(st.m - m_new)
    npad = -(-n // D) * D
    if npad == n:
        pq = ol.quantize(p * vs[None, :])                     # V scales folded (QACT CSCALE)
    else:                                                     # tail block: pad P with zeros
        pp = ol.zeros([G, npad])
        pp[:, :n].set(p * vs[None, :])
        pq = ol.quantize(pp)
    vt = st.VT.padded_cols(0, st.VT.shape[0], t0, npad, n)             # zero-padded P
    ol.dot(pq, vt, acc=st.acc, acc_scale=alpha)                         # acc*alpha + P.V
    st.m.set(m_new)                                           # the next block needs m first
    st.l.set(st.l * alpha + ol.sum(p, axis=1))                # off the critical path


def _attend_heads(qhs, kv, hs, seq_len: int, block: int, scale: float | None = None,
                  depth: int = 3, ahead: int = 1, emit=None, sinks=None, lead: int = 1,
                  lazy: bool = False, lag: int = 0, share: bool = False, part=None,
                  maxima: bool = False):
    """Flash attention (online softmax) of several KV heads' query groups, software-pipelined
    FA3-style: while the VPU and quantizer finish block b (softmax, P.V), the MXU is already
    streaming q.K^T of the next depth-1 blocks into other score buffers -- across head
    boundaries too, so the heads' dependency chains overlap. Within a head the hardware loop
    body covers `depth` blocks, so every score buffer has a fixed role.

    qhs: [G, d] query tiles, or functions that emit and return them (they are called
    `ahead` heads early, so the work producing a head's queries -- e.g. its slice of the Q
    projection -- is interleaved with the attention of the heads before it); hs: their KV
    heads. `kv` and `seq_len` (a length, a Bucket or Blocks) are one cache / length for all
    entries, or one per entry (several tokens or sequences chained through the same pipeline:
    batched decode, prefill). The softmax scale log2(e)/sqrt(d) is either already in the
    queries or given as `scale` (applied by the quantizer when q is loaded into ACT RAM, QACT
    CSCALE). Returns [(acc, l)] per head, unnormalized -- or, with `emit`, calls emit(i, acc,
    l) as soon as entry i is complete and releases its state (TMEM then holds only the entries
    in flight). sinks: per entry a [G] tile of its queries' sink logits in the scores' units
    (log2(e) times GPT-OSS's), or None (_Head). lead: the heads whose first scores are in
    flight before a head's finish (1: the next one's; 2: the next two's, so a late head's
    scores do not queue behind the P.V of the head before it in the MXU's order). lazy: each
    head's queries quantized when its scores first need them, after the pending finishes
    (_Head.quantize), not when they are made: a late head's QACT then does not hold the
    quantizer's P of the heads before it. lag: emit's call for entry i comes after the finish
    of entry i + lag (in the variants' loops, a copy in each), so its instructions find their
    P.V done instead of holding the sequencer's window (0: at once, after the pending work).

    A masked block read in variants (_Head.var: a run-time position's) has each variant's
    scores and finish in its loop of 0 or 1 iterations; the work of consecutive ones (a head's
    finish and the next head's scores) shares the loops, in the order it has without them.
    share: so does the work of consecutive heads' variants (each head has its _Variants, all on
    one table): fewer loops and their RLDs (GPT-OSS's decode; the 35B's q36-4 at r60 is 0.16%
    slower with it, so its default is off).

    part (p, r): every entry's KV head is one of r copies on r slices, which split its positions
    by block: this slice attends blocks p, p + r, .. alone (_Head's part; the caller combines
    the copies' partial states, qwen3._attention's split). maxima: return [(acc, l, m)], each
    entry's running maxima too (the partial state's, without emit).
    """
    P = depth
    kvs = kv if isinstance(kv, (list, tuple)) else [kv] * len(hs)
    lens = seq_len if isinstance(seq_len, (list, tuple)) else [seq_len] * len(hs)
    sks = sinks if isinstance(sinks, (list, tuple)) else [sinks] * len(hs)
    heads: list = []
    pend: list = []                     # [(_Variants, fn(k))]: masked work not emitted yet

    def flush():
        """The pending variants' work, in their loops."""
        if not pend:
            return
        var = pend[0][0]
        b = current()
        for k in range(var.K):
            lp = var.begin(k)
            for _, fn in pend:
                fn(k)
            b.end_loop(lp)
        pend.clear()

    def later(var, fn):
        if pend and (pend[0][0].tab is not var.tab if share else pend[0][0] is not var):
            flush()
        pend.append((var, fn))

    def issue(st, f):
        """Block f's scores into st.buf[f] (the masked block in variants: a dict by k)."""
        t0, n = st.blocks[f]
        if st.var is not None and f == len(st.blocks) - 1:
            if st._q is not None:                                 # (lazy: after the finishes)
                flush()
            bufs, qs, ns = {}, st.qs, st.var.ns
            st.buf[f] = bufs
            later(st.var, lambda k: bufs.__setitem__(
                k, _scores(st, t0, ns[k], ol.empty([st.G, ns[k]]), (t0, k), qs)))
            return
        flush()
        st.buf[f] = _scores(st, t0, n, ol.empty([st.G, n]))

    def finish(st, f):
        t0, n = st.blocks[f]
        s = st.buf.pop(f)
        if isinstance(s, dict):
            ns = st.var.ns
            later(st.var, lambda k: _finish(st, s.pop(k), t0, ns[k], (t0, k)))
            return
        flush()
        _finish(st, s, t0, n)

    lagged: list = []                   # entries finished, emit not yet called (lag)

    def out(j):
        """emit's call for entry j: at once, or in each pending variant's loop (lag)."""
        sj = heads[j]
        heads[j] = None                                           # free its tiles after
        if lag and pend:
            pend.append((pend[0][0], lambda k: emit(j, sj.acc, sj.l)))
            return
        flush()
        emit(j, sj.acc, sj.l)

    def head(i):
        while len(heads) <= min(i, len(hs) - 1):
            flush()
            j = len(heads)
            qh = qhs[j]() if callable(qhs[j]) else qhs[j]
            heads.append(_Head(qh, kvs[j], hs[j], scale, lens[j], block, P, sks[j], lazy,
                               part))
        return heads[min(i, len(hs) - 1)]

    def run(st):
        """A RunSpan head's blocks (st.run): its groups of R blocks in a loop of the span's
        run-time count, each group's scores issued (q.K^T, min with the block's mask row, the
        row maxima), then the R finishes. The loop's count is a run-time value, so it steps
        registers of its own (no reset after it): the head's K, K scales, V^T, V scales and
        mask row at the group's first block, each a device value (DevVar) taken here."""
        flush()
        sp, b = st.run, current()
        R = sp.P
        S = [ol.empty([st.G, block]) for _ in range(R)]
        rm = [ol.empty([st.G]) for _ in range(R)]
        qs = st.qs
        K, VT, VS = st.K, st.VT, st.VS
        g = R * block                                         # a group's tokens
        d = VT.shape[0]
        arrays = [(K.data, K[g:g + block, :].data),
                  (VT.data, VT.padded_cols(0, d, g, block, block).data),
                  (VS.base, VS[g:g + block].base), (sp.z, sp.row(g, block).base)]
        if K.scale is not None:
            arrays.append((K.scale, K[g:g + block, :].scale))
        regs = []
        for a, nxt in arrays:
            ra, imm = b.addr(a)
            r, how = b.borrow()
            b.emit(I.addi(r, ra, imm, comment="the span's first block", aw=b.aw))
            regs.append((r, (Affine.of(nxt) - Affine.of(a)).static(), how))
        dv = [DevVar(f"span R{r}", r) + 0 for r, _, _ in regs]
        st.K = copy.copy(K)
        st.K.data = dv[0]
        if K.scale is not None:
            st.K.scale = dv[4]
        st.VT = copy.copy(VT)
        st.VT.data = dv[1]
        st.VS = Tensor(dv[2], VS.shape, VS.strides)
        z = copy.copy(sp)
        z.z = dv[3]
        c, how = b.borrow()
        b.emit(I.rld(c, sp.count, bound=(1, sp.bound), comment="groups of the span"))
        lp = b.begin_loop(0, rcount=c)
        b.give_back(c, how)                                   # (LOOP read it as it started)
        for j in range(R):
            tj = j * block
            ol.dot(qs, st.K[tj:tj + block, :], out=S[j])
            S[j].set(ol.minimum(S[j], ol.load(z.row(tj, block))[None, :]))
            rm[j].set(ol.max(S[j], axis=1))
        for j in range(R):
            _finish(st, S[j], j * block, block, rm=rm[j])
        for r, step, _ in regs:                               # the next group's
            b.emit(I.addi(r, r, I.wrap(step, b.aw), comment="the span's next group", aw=b.aw))
        b.end_loop(lp)
        for r, _, how in regs:
            b.give_back(r, how)
        st.K, st.VT, st.VS = K, VT, VS
        st.qs = None                                              # its ACT RAM blocks are free
        del S, rm

    def prime(st):
        """Put the head's first P-1 score blocks in flight."""
        if st.run is not None:                                    # (its loop's own)
            return
        blocks = st.blocks
        if st.groups:
            flush()
            st.S = [ol.empty([st.G, block]) for _ in range(P)]
            for j in range(P - 1):
                _scores(st, st.boff + j * st.bstride, block, st.S[j])
        else:
            for f in range(min(P - 1, len(blocks))):
                issue(st, f)

    head(max(ahead, lead - 1))
    for j in range(min(lead, len(hs))):
        prime(head(j))
    for i in range(len(hs)):
        st = head(i)
        head(i + ahead + 1)                                       # the next queries, early
        nxt = head(i + lead) if i + lead < len(hs) else None
        if st.run is not None:
            run(st)
        blocks, groups = st.blocks, st.groups
        nb = 0
        if groups:
            flush()
            S = st.S
            for g in ol.range(groups):
                t0 = g * (P * st.bstride)
                if st.boff:
                    t0 = t0 + st.boff
                for p in range(P):                                    # prefetch, finish
                    _scores(st, t0 + (p + P - 1) * st.bstride, block, S[(p + P - 1) % P])
                    _finish(st, S[p], t0 + p * st.bstride, block)
            nb = groups * P
            for j in range(P - 1):
                st.buf[nb + j] = S[j]
        primed = False
        for b in range(nb, len(blocks)):                          # the rest, unrolled
            for f in range(b, min(b + P, len(blocks))):           # keep P-1 blocks in flight
                if f not in st.buf:
                    issue(st, f)
            if nxt is not None and not primed and b + P - 1 >= len(blocks):
                prime(nxt)                                        # ... continuing into the next head
                primed = True
            if b + 1 == len(blocks):
                st.qs = None                                      # its ACT RAM blocks are free
            finish(st, b)
        if nxt is not None and not primed:
            prime(nxt)
        if emit is not None:
            lagged.append(i)
            while len(lagged) > lag:
                out(lagged.pop(0))
            del st
    while lagged:
        out(lagged.pop(0))
    flush()
    if emit is not None:
        return None
    if maxima:
        return [(st.acc, st.l, st.m) for st in heads]
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
