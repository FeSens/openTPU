"""Speculative decoding with Qwen3.5's MTP drafter, k = 1, on an openTPU backend (docs/mtp.md 9).

Phase 2 drives the loop from the host, one program per run, compiled at its position as
Engine.step's are; phase 3 runs the same kernels per attention bucket at a run-time position
with the loop on the card (docs/mtp.md 2, 3). Each iteration at position p, with the token t
(at p) and the draft d (for p + 1):

    verify   qwen35_rows over (t, d) at p, p + 1 (fork, hidden): logits of both rows, a0 and a1
             their argmax; the final norm's rows to `hid`; the DeltaNet states and conv
             windows after row 0 stay in slot c, those after row 1 go to slot 1 - c
    accept   n = (d == a0); a0 is emitted, and a1 when n = 1 (a stop id ends the loop)
    draft    qwen35_mtp over (hid[0], a0), (hid[1], a1) at p, p + 1: the next draft is row n's
    commit   c ^= n, p += 1 + n, t = a_n

Greedy, the tokens are those of plain greedy decode bit for bit: every emitted token is the
argmax of a logits row the decode step would compute on the committed state (the rows
kernel is bit-identical to decode steps; with PAIR and 4-bit weights where its MMs pair as
the steps' do: 2 rows at MCOLS >= 4, qwen35_rows). Rejected rows leave nothing behind: the
KV caches (the model's and the MTP layer's) are positional and are overwritten before they
are read, and the recurrent state and windows of a rejected row sit in the uncommitted
slot.

The prompt's prefill takes plain prefill's runs (qwen3.fit_chunk: the same rows, so its
MMs pair as plain prefill's do), stores `hid` for every row and runs the MTP layer over the
prompt's rows (h_i, x_(i+1)), which fills its KV cache; its last row (h_(P-1), a0) gives the
first draft.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from .. import isa as I
from .. import language as ol
from ..compiler import CompileError, current
from . import generate as G
from .qwen3 import ATTN_BLOCK, Engine, RunRows, fit_chunk, head_rows_chunk

MTP_ROWS = 4            # the MTP layer's rows per run in the prefill (its TMEM: [R, 2H] input)


@dataclass
class MTPStats:
    """One generation: the prompt, the tokens, and per iteration the drafts accepted (0, 1)."""
    prompt: int = 0
    tokens: list = field(default_factory=list)
    accepted: list = field(default_factory=list)
    runs: list = field(default_factory=list)        # (kind, rows, stats of the backend's run)
    slots: list = field(default_factory=list)       # each verify run's committed slot (parity)
    compile_s: float = 0.0                          # host seconds compiling the programs
    prefill_s: float = 0.0                          # host seconds of the prefill (wall)
    prefill_compile_s: float = 0.0                  # ... compiling its programs
    timed_out: bool = False                         # loop_card: the deadline wrote the stop word

    @property
    def iterations(self) -> int:
        return len(self.accepted)

    @property
    def acceptance(self) -> float:
        return sum(self.accepted) / max(1, len(self.accepted))


def mtp_engine(spec, W, cap: int = 4096, **kw) -> Engine:
    """An Engine whose image holds the MTP drafter and the state slots (Spec.mtp) and the
    resident decode's tables (its runs read their inputs from the image). W must hold the
    checkpoint's mtp.* tensors (load_weights(path, mtp=True))."""
    return Engine(replace(spec, mtp=True), W, cap=cap, resident=True, **kw)


class MTPDecoder:
    """Greedy speculative decoding with the MTP drafter on an Engine made by mtp_engine.
    slot: the DeltaNet states' and windows' committed slot (DeltaNetParts)."""

    def __init__(self, engine: Engine):
        img = engine.image
        if not getattr(img.spec, "mtp", False) or not img.lookup:
            raise ValueError("MTPDecoder needs an Engine made by mtp_engine (Spec.mtp, lookup)")
        if engine.cfg.S != 1:
            raise ValueError("MTP decoding runs on one slice")
        fmts = {img.wformat, img.head_format, *getattr(img, "mf", {}).values()}
        if engine.cfg.PAIR and fmts - {"int8", None} and 2 * 2 > engine.cfg.MCOLS:
            raise ValueError("with PAIR and 4-bit weights a 2-row verify run pairs its MMs as "
                             "the decode steps do only at MCOLS >= 4 (qwen35_rows): its tokens "
                             "would not be plain decode's bit for bit")
        self.eng, self.img = engine, img
        self.slot = 0

    # ---- runs
    def _run(self, progs, kind: str, rows: int, st: MTPStats) -> None:
        r = self.eng.backend.run(progs)
        st.runs.append((kind, rows, r))

    def _logits(self, n: int) -> np.ndarray:
        io, v = self.img.io, self.img.spec.vocab
        return self.eng.backend.read(0, io["logits"], 4 * n * v).view(np.float32).reshape(n, v)

    def _drafts(self, n: int) -> list:
        w = self.eng.backend.read(0, self.img.io["draft"], 4 * n).view(np.float32)
        return [int(x) for x in w]

    def _verify(self, p: int, toks, logit_rows, fork: bool, st: MTPStats, kind: str):
        rows = [(0, p + j) for j in range(len(toks))]
        t0 = time.perf_counter()
        progs = self.img.compile_rows(rows, logit_rows, self.eng.block, tokens=toks,
                                      slot=self.slot, fork=fork, hidden=True)
        st.compile_s += time.perf_counter() - t0
        if fork:
            st.slots.append(self.slot)
        self._run(progs, kind, len(toks), st)

    def _draft(self, p: int, toks, st: MTPStats, kind: str, h0: int = 0) -> list:
        t0 = time.perf_counter()
        progs = self.img.compile_mtp(p, len(toks), toks, self.eng.block, h0=h0)
        st.compile_s += time.perf_counter() - t0
        self._run(progs, kind, len(toks), st)
        return self._drafts(len(toks))

    # ---- generation
    def prefill(self, prompt, st: MTPStats, chunk: int = MTP_ROWS):
        """The prompt in plain prefill's runs (qwen3.fit_chunk, Engine.prefill's: their rows
        and so their MMs' pairing are its), each storing its rows' hidden, then the MTP layer
        over them in runs of up to `chunk` rows; returns (a0, the first draft)."""
        prompt = [int(t) for t in prompt]
        P, p = len(prompt), self.eng.pos
        if p != 0:
            raise ValueError("MTP decoding starts from an empty context (Engine.reset)")
        self.slot = 0
        a0 = draft = None
        fit = self.img.rows
        while p < P:
            left = P - p
            t0 = time.perf_counter()
            n, progs, fit = fit_chunk(self.img, self.eng.block, 0, p, self.img.rows, left, fit,
                                      prompt[p:], hidden=True, slot=self.slot)
            if progs is None:           # one row: the rows kernel (plain prefill's decode step)
                progs = self.img.compile_rows([(0, p)], [0] if n == left else [],
                                              self.eng.block, tokens=prompt[p:p + 1],
                                              slot=self.slot, hidden=True)
            st.compile_s += time.perf_counter() - t0
            self._run(progs, "prefill", n, st)
            nxt = prompt[p + 1:p + n + 1]
            if n == left:
                a0 = int(np.argmax(self._logits(n)[n - 1]))
                nxt = nxt + [a0]
            for j in range(0, n, chunk):
                k = min(chunk, n - j)
                draft = self._draft(p + j, nxt[j:j + k], st, "mtp prefill", h0=j)[k - 1]
            p += n
        self.eng.pos = P
        return a0, draft

    def generate(self, prompt, max_new: int = 32, stop=None, drafts=None) -> MTPStats:
        """Greedy generation from an empty context; stops after a stop id (default the
        model's EOS ids) or max_new tokens, leaving the context as plain decode would (the
        last token not fed: Engine.pos is its position). drafts (tests): drafts(position,
        tokens so far) -> the draft for that position instead of the MTP's."""
        stop = set(self.img.spec.eos if stop is None else stop)
        st = MTPStats(prompt=len(prompt))
        t0 = time.perf_counter()
        a0, d = self.prefill(prompt, st)
        st.prefill_s, st.prefill_compile_s = time.perf_counter() - t0, st.compile_s
        out, p, t = st.tokens, len(prompt), a0
        out.append(a0)
        while len(out) < max_new and out[-1] not in stop and p + 1 < self.img.cap:
            if drafts is not None:
                d = int(drafts(p + 1, out))
            self._verify(p, [t, d], [0, 1], True, st, "verify")
            lg = self._logits(2)
            a = [int(np.argmax(lg[0])), int(np.argmax(lg[1]))]
            n = int(d == a[0])
            st.accepted.append(n)
            # a0, and a1 when the draft holds; cut after a stop id or at max_new, and then the
            # committed state is row 0's (as plain decode's: the last token not fed)
            keep = 1 if a[0] in stop or len(out) + 1 >= max_new else 1 + n
            out += a[:keep]
            if keep - 1 < n or out[-1] in stop or len(out) >= max_new:
                self.slot ^= keep - 1
                p += keep
                break
            nd = self._draft(p, a, st, "mtp")
            self.slot ^= n
            p, t, d = p + 1 + n, a[n], nd[n]
        self.eng.pos = p
        return st

    # ---- the loop on the card
    def _gen_programs(self, b0: int, b1: int, st: MTPStats | None = None,
                      forced: bool = False) -> dict:
        """Buckets b0 .. b1's programs of the MTP loop in the chain area and their table
        entries (each compiled and written once per engine and `forced`); returns {(bucket,
        kind): programs}."""
        eng, img = self.eng, self.img
        a = img.lookup["mtpgen"]
        if eng.__dict__.get("_mtp_forced", forced) != forced:
            eng.__dict__.pop("_mtp_gen", None)      # the chain area holds the other kind
        eng._mtp_forced = forced
        done = eng.__dict__.setdefault("_mtp_gen", {})
        for blk in range(b0, b1 + 1):
            if (blk, V0) in done:
                continue
            t0 = time.perf_counter()
            tab = np.zeros(2 * NK, np.uint32)
            for k in range(NK):
                progs = eng.cached(("mtpgen", blk, k, eng.block, forced),
                                   lambda: (compile_gen(img, blk, k, eng.block, forced), None))[0]
                if progs is None:
                    continue
                if not G.fits(img, progs):
                    raise CompileError(f"MTP loop program {KINDS[k]} of bucket {blk}: "
                                       f"{max(map(len, progs))} instructions, IMEM "
                                       f"{img.cfg.IMEM_WORDS // 8}")
                for s_, prog in enumerate(progs):
                    w = np.asarray(I.assemble(prog), np.uint32)
                    eng.backend.write(s_, prog_addr(a, blk, k), w)
                tab[2 * k], tab[2 * k + 1] = prog_addr(a, blk, k), len(I.assemble(progs[0])) // 8
                done[(blk, k)] = progs
            for s_ in range(img.cfg.S):
                eng.backend.write(s_, ptab_addr(a, blk), tab)
            if st is not None:
                st.compile_s += time.perf_counter() - t0
        return done

    def generate_card(self, prompt, max_new: int = 32, stop=None, on_token=None,
                      drafts=None, deadline: float | None = None) -> MTPStats:
        """Greedy MTP decoding with the loop on the device (docs/mtp.md 10): the prefill as
        generate()'s (its last logits' argmax on the host, as Engine.generate_card's callers
        take it), then one device run that verifies, drafts and chains through the buckets'
        programs until a stop id (default the model's EOS ids), the host's stop word or
        max_new tokens; the tokens are read from out[] as they land (on_token). The context
        is left as plain decode leaves it (the last token not fed: Engine.pos its position,
        the committed state slot read back). st.runs: the run's counters (CYCLES), st.accepted
        the iterations' accepted drafts (their count, from the device's counters). drafts
        (tests): the draft of each position q (drafts[q], q < cap + 2) instead of the MTP's,
        from the first iteration's on. deadline (seconds, a card's run_generate): past it the
        host writes the stop word (st.timed_out; the card halts at its next verify), and 10 s
        later a run still going raises TimeoutError."""
        st = MTPStats(prompt=len(prompt))
        t0 = time.perf_counter()
        a0, d = self.prefill(prompt, st)
        st.prefill_s, st.prefill_compile_s = time.perf_counter() - t0, st.compile_s
        return self.loop_card(a0, d, max_new, stop, on_token, drafts, st, deadline)

    def loop_card(self, a0: int, d: int, max_new: int = 32, stop=None, on_token=None,
                  drafts=None, st: MTPStats | None = None,
                  deadline: float | None = None) -> MTPStats:
        """generate_card after its prefill: a0 (the token at Engine.pos, emitted) and d (the
        draft of the next position) -> the device's run of the MTP loop from the committed
        slot (see generate_card)."""
        eng, img = self.eng, self.img
        spec, block = img.spec, eng.block
        ids = list(spec.eos if stop is None else stop)
        st = MTPStats(prompt=eng.pos) if st is None else st
        st.tokens.append(a0)
        if on_token is not None:
            on_token(a0)
        P = eng.pos
        n = min(max_new - 1, img.cap - P - 2)
        if a0 in ids or n <= 0:
            return st
        b0, b1 = P // block + 1, (P + n) // block + 1
        progs = self._gen_programs(b0, b1, st, drafts is not None)
        g = img.lookup["gen"]
        if drafts is not None:
            dt = np.zeros(img.cap + 2, np.float32)
            dt[:len(drafts)] = np.asarray(drafts, np.float32)[:img.cap + 2]
            eng.backend.write(0, img.lookup["mtpgen"]["dtab"], dt)
            d = int(dt[P + 1])
        words = G.state_words(spec, a0, P, n, ids, block)
        words[S_DRAFT], words[S_PAR] = d, self.slot
        for s_ in range(img.cfg.S):
            eng.backend.write(s_, g["state"], words)
            eng.backend.write(s_, g["out"] + 4 * (P + 1), np.full(n, G.OUT_MARK, np.uint32))
        first = progs[(b0, (E0 if P % block == block - 1 else V0) + self.slot)]
        run = getattr(eng.backend, "run_generate", None)
        if run is not None:
            late = None
            if deadline is not None:
                end = time.perf_counter() + deadline

                def late() -> bool:
                    now = time.perf_counter()
                    if now > end + 10.0:
                        raise TimeoutError(f"the MTP loop runs {deadline + 10:.0f} s, its stop "
                                           f"word unanswered")
                    st.timed_out = st.timed_out or now > end
                    return st.timed_out
            stats, got = run(first, g["out"] + 4 * (P + 1), n, on_token, late, g["state"])
        else:
            stats = eng.backend.run(first)
            w = eng.backend.read(0, g["out"] + 4 * (P + 1), 4 * n).view(np.uint32)
            k = int(np.argmax(w == G.OUT_MARK)) if (w == G.OUT_MARK).any() else n
            got = [int(x) for x in w[:k].view(np.float32)]
            if on_token is not None:
                for t in got:
                    on_token(t)
        st.runs.append(("generate", len(got), stats))
        st.tokens += got
        sw = eng.backend.read(0, g["state"], 4 * G.STATE_WORDS).view(np.float32)
        self.slot = int(sw[S_PAR])
        it, acc = int(sw[S_ITER]), int(sw[S_ACC])
        st.accepted = [1] * acc + [0] * (it - acc)
        eng.pos = P + len(got)
        return st


# ---- the loop on the card (docs/mtp.md 10)
# the state block: generate.py's words, and the iteration's
S_DRAFT, S_A0, S_A1, S_N, S_PAR, S_TPOS0, S_ITER, S_ACC = 20, 21, 22, 23, 24, 25, 26, 27
# a bucket's programs, in its chain table's order: the verify (V) and the bucket's last
# position (E) per parity, the MTP layer over two rows (D) and over one (D1)
KINDS = ("V0", "V1", "E0", "E1", "D", "D1")
NK = len(KINDS)
V0, E0, D2, D1 = 0, 2, 4, 5


def gen_alloc(b, cap: int, block: int = ATTN_BLOCK) -> dict:
    """DRAM of the MTP loop: a program slot per bucket and kind, and the chain table
    ([address, instructions] per bucket and kind)."""
    nb = G.buckets(cap, block)
    return {"progs": b.alloc(G.PROG_SLOT * NK * nb), "ptab": b.alloc(8 * NK * nb), "nb": nb,
            "dtab": b.alloc(4 * (cap + 2))}      # tests: the draft of each position


def prog_addr(a: dict, blocks: int, kind: int) -> int:
    return a["progs"] + G.PROG_SLOT * (NK * (blocks - 1) + kind)


def ptab_addr(a: dict, blocks: int, kind: int = 0) -> int:
    return a["ptab"] + 8 * (NK * (blocks - 1) + kind)


def _flag(x):
    """1.0 where the small integer x is >= 1, else 0.0."""
    return ol.minimum(ol.maximum(x, 0.0), 1.0)


def _hit(st, tok):
    """1.0 when tok is one of the state block's stop ids (integers)."""
    ids = st[G.S_STOP:G.S_STOP + G.N_STOP]
    return ol.max(ol.minimum(ol.abs(ids - tok), 1.0) * -1.0 + 1.0)


def _load_state(b, g):
    """The state block into TMEM, and its LD (the run-time arguments load right after it:
    the compiler puts its prologue before the program's first instruction, so an index into
    the program would not stay put)."""
    st = ol.load(g.state)
    return st, b.stack[-1][-1]


def _args(b, at, words: dict) -> None:
    """The run-time arguments (c * var alone in an address) loaded from their state words
    right after the instruction `at` (the state's LD), now that the program has named them
    all."""
    body = b.stack[-1]
    i = next(j for j, x in enumerate(body) if x is at) + 1
    body[i:i] = [I.rld(15 - k, words[v.name], mul=int(c), comment=f"argument {c}*{v.name}")
                 for k, (v, c) in enumerate(b.run_args)]
    b.run_words = None


def _store_state(g, st) -> None:
    """The state block to DRAM, but the host's stop word and the sampler's."""
    ol.store(g.state[:G.S_HALT], st[:G.S_HALT])
    ol.store(g.state[S_DRAFT:S_ACC + 1], st[S_DRAFT:S_ACC + 1])


def _verify_gen(m, pos, block: int, c: int, R: int):
    """V (R = 2: rows t, d at p, p + 1) or E (R = 1: t at the bucket's last position) of
    parity c: the rows kernel (fork, hidden) with an ARGMAX per row; a0 -> out[p + 1], and
    with the draft accepted (d == a0, tokens left, a0 not a stop id) a1 -> out[p + 2]; the
    iteration's words and the commit; HALT at a stop id, the host's stop word or no tokens
    left, else HALT CHAIN to the bucket's D or D1."""
    from .qwen35 import qwen35_rows
    b = current()
    g, a = m.gen, m.mtpgen
    st, at = _load_state(b, g)
    words = {"tok": st.base + G.S_TOK, "tok1": st.base + S_DRAFT, "tpos": st.base + G.S_TPOS}
    b.run_words = words
    m.lm_sinks = [G.Greedy(b, m.v_loc, head_rows_chunk(R)) for _ in range(R)]
    qwen35_rows.fn(m, pos, R, list(range(R)), block, None, R > 1, True)
    toks = [sk.token() for sk in m.lm_sinks]
    m.lm_sinks = None
    a0, a1 = toks[0], toks[-1]
    left, tpos = st[G.S_LEFT:G.S_LEFT + 1], st[G.S_TPOS:G.S_TPOS + 1]
    stop = _hit(st, a0)
    ol.store(g.out[pos.pos + 1:pos.pos + 2], a0)
    if R > 1:
        d = st[S_DRAFT:S_DRAFT + 1]
        n = (ol.minimum(ol.abs(d - a0), 1.0) * -1.0 + 1.0) * _flag(left - 1.0) * \
            (stop * -1.0 + 1.0)
        r = b.scratch()
        b.rld(r, n, comment="accepted")
        lp = b.begin_loop(0, rcount=r)
        b.unscratch(r)
        ol.store(g.out[pos.pos + 2:pos.pos + 3], a1)
        b.end_loop(lp)
        stop = ol.maximum(stop, _hit(st, a1) * n)
        st[G.S_TOK:G.S_TOK + 1].set(a0 + n * (a1 - a0))
        st[S_PAR:S_PAR + 1].set(n if c == 0 else n * -1.0 + 1.0)
    else:
        n = ol.zeros([1])
        st[G.S_TOK:G.S_TOK + 1].set(a0)
        st[S_PAR:S_PAR + 1].set(float(c))
    st[S_A0:S_A0 + 1].set(a0)
    st[S_A1:S_A1 + 1].set(a1)
    st[S_N:S_N + 1].set(n)
    st[S_TPOS0:S_TPOS0 + 1].set(tpos)
    tpos.set(tpos + n + 1.0)
    left.set(left - n - 1.0)
    st[S_ITER:S_ITER + 1].set(st[S_ITER:S_ITER + 1] + 1.0)
    st[S_ACC:S_ACC + 1].set(st[S_ACC:S_ACC + 1] + n)
    halt = ol.maximum(ol.maximum(stop, ol.load(g.state[G.S_HALT:G.S_HALT + 1])),
                      _flag(left * -1.0 + 1.0))
    _args(b, at, words)
    _store_state(g, st)
    r = b.scratch()
    b.rld(r, halt, comment="stop")
    b.emit(I.loop(1, 0, rcount=r, comment="stop: halt"))
    b.emit(I.halt())
    b.unscratch(r)
    G._chain(b, ptab_addr(a, pos.blocks, D2 if R > 1 else D1), what="the draft")


def _draft_gen(m, pos, block: int, R: int, forced: bool = False):
    """D (R = 2) or D1 (R = 1): the MTP layer over the iteration's rows (hid_r, a_r) at p ..
    (qwen35_mtp), the next draft d = draft[n]; the position moves on (to the next bucket at
    its end), then HALT CHAIN to the next iteration's program: V or E (tpos = block - 1) of
    the committed parity, in this bucket or the next. forced (tests): the draft of the next
    iteration's row 1 (position p + 2 + n) from the host's table (dtab) instead."""
    from .qwen35 import qwen35_mtp
    b = current()
    g, a = m.gen, m.mtpgen
    st, at = _load_state(b, g)
    words = {"tok": st.base + S_A0, "tok1": st.base + S_A1, "tpos": st.base + S_TPOS0}
    b.run_words = words
    drafts = qwen35_mtp.fn(m, pos, R, block)
    _args(b, at, words)
    n = st[S_N:S_N + 1]
    st[S_DRAFT:S_DRAFT + 1].set(drafts[0] if R == 1 else drafts[0] + n * (drafts[1] - drafts[0]))
    if forced:
        r = b.scratch()
        b.rld(r, (st[S_TPOS0:S_TPOS0 + 1] + n) * 4.0, comment="the draft's position")
        b.emit(I.ld(a["dtab"] + 4 * (pos.t0 + 2), st.base + S_DRAFT, 1, ra=r,
                    comment="the host's draft"))
        b.unscratch(r)
    tp = st[G.S_TPOS:G.S_TPOS + 1]                      # the next position's: 1 .. block
    wrap = _flag(tp - float(block - 1))                 # block: the next bucket's first
    tp.set(tp - wrap * float(block))
    end = _flag(tp - float(block - 2))                  # block - 1: the bucket's E
    off = (wrap * float(NK) + end * 2.0 + st[S_PAR:S_PAR + 1]) * 8.0
    _store_state(g, st)
    G._chain(b, ptab_addr(a, pos.blocks), off=off, what="the next iteration")


def compile_gen(image, blocks: int, kind: int, block: int = ATTN_BLOCK,
                forced: bool = False) -> list | None:
    """Program KINDS[kind] of bucket `blocks` of the MTP loop (docs/mtp.md 10), one per slice;
    None when the bucket has no such position (E and D1 past the KV cache's end). forced
    (tests): D and D1 take the drafts from the host's table (_draft_gen)."""
    name = KINDS[kind]
    R = 1 if name in ("E0", "E1", "D1") else 2
    t0 = (blocks - 1) * block
    lo = t0 + block - 1 if R == 1 else max(t0, image.spec.conv_k - 1)
    if lo + R > image.cap or "mtpgen" not in image.lookup:
        if "mtpgen" not in image.lookup:
            raise ValueError("the MTP loop needs an MTP image with lookup tables")
        return None
    rp = RunRows(blocks, block, lo, image.lookup["zmask"], image.cap, R)
    c = int(name[1]) if name[0] in "VE" else 0
    fn, kw = ((_verify_gen, {"c": c, "R": R}) if name[0] in "VE" else
              (_draft_gen, {"R": R, "forced": forced}))
    return [ol.jit(fn).trace(image.cfg, s, {"m": image.descriptors(s, c), "pos": rp,
                                            "block": block, **kw}).finish()
            for s in range(image.cfg.S)]
