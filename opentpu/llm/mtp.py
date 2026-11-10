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

Sampled (the loop on the card, docs/mtp.md 11), a0 and a1 are generate.Sampler's picks with
the uniforms of positions p + 1 and p + 2, and the draft is accepted iff a0 == d: speculative
sampling for a one-hot draft. The tokens are plain sampled decode's for the same uniforms;
row 1's repetition penalty counts d (_penalize_draft).

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
from . import prefill as PF
from ..host.offload import LINE
from ..kernels import mailbox as MB
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
    """Speculative decoding with the MTP drafter on an Engine made by mtp_engine: greedy, and
    sampled with the loop on the card (generate_card(sampling=)). slot: the DeltaNet states'
    and windows' committed slot (DeltaNetParts)."""

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
        mo = getattr(img.spec, "moe", None)
        if mo is not None and (img.rows < 2 or 2 * mo.k > LINE // 4):
            raise ValueError("MTP decoding of a MoE model needs an image of 2 rows and a "
                             "request row for both rows' experts (2 k ids)")
        self.eng, self.img = engine, img
        self.slot = 0
        self.draft = None           # the draft after the token at Engine.pos (prefill, the loop)
        self.pending = None         # (p, a0): a lazy prefill's last row, its draft not run yet
                                    # (the loop's first run drafts it: loop_card; else _settle)
        self.held = {}              # a snapshot's position t -> the residual row of t - 1 (holds)
        self.refill = None          # (t - 1, that row): the next prefill's first K / V row again
        self.split = None           # the loop's V and E in two parts: None when over IMEM (tests:
                                    # True always, False never; _gen_programs)
        # the MTP layer's rows a prefill run (a MoE's: its experts' request names R k ids)
        self.mtp_rows = MTP_ROWS if mo is None else max(1, min(MTP_ROWS, LINE // 4 // mo.k))

    # ---- runs
    def _run(self, progs, kind: str, rows: int, st: MTPStats, tokens=None) -> None:
        """A run; `tokens`: its rows' tokens, whose embedding rows the host writes first when
        the int8 table is on the host (embed_host: a MoE's)."""
        if tokens is not None and getattr(self.img, "embed_host", False):
            self.eng._write_host_rows(tokens)
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
                                      slot=self.slot, fork=fork, hidden=True, nest=False)
        st.compile_s += time.perf_counter() - t0
        if fork:
            st.slots.append(self.slot)
        self._run(progs, kind, len(toks), st, toks)

    def _draft(self, p: int, toks, st: MTPStats, kind: str, h0: int = 0) -> list:
        t0 = time.perf_counter()
        progs = self.img.compile_mtp(p, len(toks), toks, self.eng.block, h0=h0)
        st.compile_s += time.perf_counter() - t0
        self._run(progs, kind, len(toks), st, toks)
        return self._drafts(len(toks))

    # ---- generation
    def prefill(self, tokens, st: MTPStats, chunk: int | None = None, pick=None, on_run=None,
                follow: int | None = None, vis=None, lazy: bool = False):
        """Feed `tokens` from Engine.pos in plain prefill's runs (qwen3.fit_chunk,
        Engine.prefill's: their rows and so their MMs' pairing are its), each storing its rows'
        hidden, then the MTP layer over them, (h_q, x_(q + 1)), in runs of up to `chunk` rows;
        returns (a0, the first draft), also in self.draft. pick(logits) -> a0 (default the
        argmax). From Engine.pos 0 a new context; from a later one, the context this decoder
        left (its prefill, or the loop's run: the MTP layer's KV cache holds every position
        before Engine.pos, the committed slot the states). on_run(tokens): after each prefill
        run (Engine.pos past it); a true return stops the prefill there (-> None, None; a later
        prefill feeds the rest). follow: the token after `tokens` (a part of a prompt, the rest
        fed later): the last row's MTP input, no logits, a0 None. vis: each token's image row
        or -1 (Engine.prefill_layers': MiniCPM-V's images, a layer-by-layer prompt's only; the
        MTP layer's input of an image token is its placeholder's embedding row). lazy (a
        layer-by-layer prompt's, by_layers: the others draft in their last run): the last row's
        draft is left to the loop's first run (loop_card: a D1 before the first verify, served
        in the loop's run), a0 returned at once with no draft (self.pending; a prefill or a
        rollback before any loop settles it: _settle)."""
        self._settle(st)
        toks = [int(t) for t in tokens]
        vis = None if vis is None or max(vis, default=-1) < 0 else [int(v) for v in vis]
        if vis is not None and len(vis) != len(toks):
            raise ValueError(f"vis: {len(vis)} rows for {len(toks)} tokens")
        chunk = self.mtp_rows if chunk is None else chunk
        p0 = p = self.eng.pos
        P = p0 + len(toks)
        if not toks or P > self.img.cap - 1:
            raise ValueError(f"{len(toks)} tokens at position {p0}: the cache holds "
                             f"{self.img.cap}, and the next token needs a position")
        if p0 == 0:
            self.slot = 0
        if vis is not None and not (self.by_layers and P > self.img.spec.conv_k - 1):
            raise ValueError("image rows go in a layer-by-layer prompt (a streamed MoE's)")
        if self.eng.prompt_runs and PF.supported(self.eng) and PF.covers(self.eng, p0, P):
            a0, draft = PF.mtp(self, toks, st, pick, on_run, follow)    # docs/prefill.md
            self.draft = draft
            return a0, draft
        K = self.img.spec.conv_k
        if self.by_layers and P > K - 1:    # a streamed MoE's prompt layer by layer: the rows
            n = max(0, K - 1 - p0)          # before conv_k - 1 in the rows kernel's runs first
            stop = []                       # (their runs reported too: on_run, e.g. Chat.fed)

            def first(part):
                stop.append(bool(on_run(part)))
                return stop[-1]
            if n:
                if vis is not None and max(vis[:n]) >= 0:
                    raise ValueError(f"an image row before position {K - 1}")
                self.prefill(toks[:n], st, chunk, on_run=None if on_run is None else first,
                             follow=toks[n])
            if any(stop):
                self.draft = None
                return None, None
            return self._prefill_layers(toks[n:], st, pick, on_run, follow,
                                        None if vis is None else vis[n:], lazy)
        a0 = draft = None
        fit = self.img.rows
        head = follow is None
        while p < P:
            left, i = P - p, p - p0
            t0 = time.perf_counter()
            n, progs, fit = fit_chunk(self.img, self.eng.block, 0, p, self.img.rows, left, fit,
                                      toks[i:], logits=head, hidden=True, slot=self.slot)
            if progs is None:           # one row: the rows kernel (plain prefill's decode step)
                progs = self.img.compile_rows([(0, p)], [0] if head and n == left else [],
                                              self.eng.block, tokens=toks[i:i + 1],
                                              slot=self.slot, hidden=True)
            st.compile_s += time.perf_counter() - t0
            self._run(progs, "prefill", n, st, toks[i:i + n])
            nxt = toks[i + 1:i + n + 1]
            if n == left and head:
                lg = self._logits(n)[n - 1]
                a0 = int(np.argmax(lg)) if pick is None else int(pick(lg))
                nxt = nxt + [a0]
            elif n == left:
                nxt = nxt + [int(follow)]
            for j in range(0, n, chunk):
                k = min(chunk, n - j)
                draft = self._draft(p + j, nxt[j:j + k], st, "mtp prefill", h0=j)[k - 1]
            p += n
            self.eng.pos = p
            if on_run is not None and on_run(toks[i:i + n]) and p < P:
                self.draft = None
                return None, None
        self.eng.pos = P
        self.draft = draft
        return a0, draft

    @property
    def by_layers(self) -> bool:
        """The prompt goes layer by layer (a streamed MoE's Engine with layer_major and its
        embedding rows from the host, _prefill_layers)."""
        return (getattr(self.img.spec, "moe", None) is not None and bool(self.eng.layer_major)
                and bool(getattr(self.img, "embed_host", False)))

    @property
    def holds(self) -> bool:
        """A snapshot keeps the residual row before it (by_layers: Engine.prefill_layers'
        keep), so a rollback to it takes the context back to the snapshot itself, and the next
        prefill appends the MTP layer's row before it from that row with the new token
        (Engine.prefill_layers' fix) instead of running that position through the model again.
        Without, a rollback goes a token further back (rollback)."""
        return self.by_layers

    def _prefill_layers(self, toks, st: MTPStats, pick, on_run, follow, vis=None,
                        lazy: bool = False):
        """prefill on a streamed MoE whose Engine runs the prompt layer by layer
        (Engine.prefill_layers with the MTP layer's K / V, docs/mtp.md 12), from conv_k - 1
        on: each chunk's rows through the model a layer at a time (the expert slots pooled),
        then the MTP layer's K / V of its rows that have their next token (the prompt's, or
        `follow`); then, the slots back to decode's, the last row's draft: qwen35_mtp over the
        head's hidden row and a0 (its MoE a request in the MTP layer's slots; lazy: left to
        the loop, self.pending). The states in the committed slot. on_run(tokens) once, after
        it. vis: the tokens' image rows. A part before a snapshot (follow) keeps its last
        row's residual (holds); after a rollback to one, the MTP layer's row before it comes
        first (self.refill)."""
        eng = self.eng
        head = follow is None
        P, n0 = eng.pos + len(toks), len(eng.stats)
        kw = {} if vis is None else {"vis": vis}
        if self.refill is not None:
            kw["fix"], self.refill = self.refill, None
        lg = eng.prefill_layers(toks, logits=head, mtp=toks[1:] + [follow], slot=self.slot,
                                keep=not head, **kw)
        st.runs.append(("prefill", len(toks),
                        {"cycles": sum((s or {}).get("cycles", 0) for s in eng.stats[n0:])}))
        if on_run is not None:
            on_run(toks)
        if not head:
            self.draft = None
            return None, None
        a0 = int(np.argmax(lg)) if pick is None else int(pick(lg))
        if lazy:
            self.draft, self.pending = None, (P - 1, a0)
            return a0, None
        draft = self._draft(P - 1, [a0], st, "mtp prefill")[0]
        self.draft = draft
        return a0, draft

    def _settle(self, st: MTPStats | None = None) -> None:
        """A lazy prefill's pending draft run on its own (the 1-row MTP run an eager prefill
        takes), when no loop came to take it: the MTP layer's K / V row at its position and
        self.draft as an eager prefill leaves them."""
        if self.pending is None:
            return
        (q, a0), self.pending = self.pending, None
        self.draft = self._draft(q, [a0], MTPStats() if st is None else st, "mtp prefill")[0]

    def kept_row(self) -> np.ndarray | None:
        """The residual row before Engine.pos, when the last prefill kept it (holds: a part
        before a snapshot, Engine.kept), read back; else None."""
        eng = self.eng
        kept = getattr(eng, "kept", None)
        if not self.holds or kept is None or kept[0] != eng.pos - 1:
            return None
        H = self.img.spec.hidden
        return eng.backend.read(0, self.img.io["xbuf"] + 4 * H * kept[1],
                                4 * H).view(np.float32).copy()

    def snapshot(self) -> bool:
        """Engine.snapshot of the committed slot's states and windows; with the residual row
        before its position, read back (holds: the last prefill kept it, kept_row)."""
        eng = self.eng
        ok = eng.snapshot(self.slot)
        h = self.kept_row() if ok else None
        if h is not None:
            self.held[eng.pos] = h
        at = set(eng._snaps.at.values()) if getattr(eng, "_snaps", None) is not None else set()
        self.held = {t: h for t, h in self.held.items() if t in at}
        return ok

    def rollback(self, k: int) -> int:
        """Engine.rollback for this decoder's context (into the committed slot): the MTP
        layer's row at q took the token at q + 1 (h_q, x_(q + 1)), so a context going back
        because the token at k changes must append the row at k - 1 again with the new token.
        With a snapshot at k that holds the residual row of k - 1 (holds), the context goes
        back to k and the next prefill appends that row from it first (self.refill); else it
        goes back to k - 1 at most, where the next prefill's first row computes that row
        again. Returns the position."""
        if k >= self.eng.pos:
            return self.eng.pos
        h = self.held.get(k)
        t = self.eng.rollback(k if h is not None else max(0, k - 1), self.slot)
        self.draft = self.pending = None    # (the next prefill computes the rows from t on)
        self.refill = (t - 1, h) if h is not None and t == k else None
        self.held = {p: v for p, v in self.held.items() if p <= t}
        if t == 0:
            self.slot = 0
        return t

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
        out, p, t = st.tokens, self.eng.pos, a0
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
                      forced: bool = False, samp: G.Sampling | None = None) -> dict:
        """Buckets b0 .. b1's programs of the MTP loop in the chain area and their table
        entries (each compiled and written once per engine, `forced` and sampler); returns
        {(bucket, kind): programs} (a split kind's first part; its second at (bucket, kind,
        1)). A V or E over IMEM (or any, self.split True) goes in two parts (compile_gen's
        split form, cut at the first of plan_cuts whose parts fit): the first in its slot and
        table entry, the second in the slot's second half, its entry at the first half's end
        (ptab2_addr)."""
        eng, img = self.eng, self.img
        a = img.lookup["mtpgen"]
        mode = (forced, None if samp is None else samp.key)
        if eng.__dict__.get("_mtp_forced", mode) != mode:
            eng.__dict__.pop("_mtp_gen", None)      # the chain area holds another kind
        eng._mtp_forced = mode
        done = eng.__dict__.setdefault("_mtp_gen", {})

        def over(k, progs):
            return CompileError(f"MTP loop program {KINDS[k]} of bucket {blk}: "
                                f"{max(map(len, progs))} instructions, IMEM "
                                f"{img.cfg.IMEM_WORDS // 8}")
        for blk in range(b0, b1 + 1):
            if (blk, V0) in done:
                continue
            t0 = time.perf_counter()
            tab = np.zeros(2 * NK, np.uint32)
            for k in range(NK):
                sk = samp if KINDS[k][0] in "VE" else None      # (D, D1: either mode's)
                what = ("mtpgen", blk, k, eng.block, forced, None if sk is None else sk.key)
                splits = KINDS[k][0] in "VE" and self.split is not False
                progs = None
                if not (splits and self.split):
                    progs = eng.cached(what, lambda: (compile_gen(img, blk, k, eng.block,
                                                                  forced, sk), None))[0]
                    if progs is None:
                        continue
                    if not G.fits(img, progs):
                        if not splits:
                            raise over(k, progs)
                        progs = None
                parts = None
                if progs is None:           # the split form
                    for cut in plan_cuts(img):
                        parts = [eng.cached(what + ("part", j, cut), lambda j=j: (
                            compile_gen(img, blk, k, eng.block, forced, sk, j, cut), None))[0]
                            for j in (0, 1)]
                        if parts[0] is None or (all(G.fits(img, p) for p in parts) and all(
                                4 * len(G.slot_words(eng.backend, p)) <= a["slot"] // 2 - 8
                                for p in parts[0])):    # (room for the second's entry)
                            break
                    else:
                        raise over(k, parts[1] if parts else [[]])
                    if parts[0] is None:
                        continue
                    progs, p2 = parts[0], prog_addr(a, blk, k) + a["slot"] // 2
                    ent = np.array([p2, len(I.assemble(parts[1][0])) // 8], np.uint32)
                    for s_, prog in enumerate(parts[1]):
                        eng.backend.write(s_, p2, G.slot_words(eng.backend, prog))
                        eng.backend.write(s_, ptab2_addr(a, blk, k), ent)
                for s_, prog in enumerate(progs):     # (with its wait entries: chained)
                    eng.backend.write(s_, prog_addr(a, blk, k), G.slot_words(eng.backend, prog))
                tab[2 * k], tab[2 * k + 1] = prog_addr(a, blk, k), len(I.assemble(progs[0])) // 8
                done[(blk, k)] = progs              # (a split kind's first part: the entry)
                if parts is not None:
                    done[(blk, k, 1)] = parts[1]
            for s_ in range(img.cfg.S):
                eng.backend.write(s_, ptab_addr(a, blk), tab)
            if st is not None:
                st.compile_s += time.perf_counter() - t0
        return done

    def generate_card(self, prompt, max_new: int = 32, stop=None, on_token=None,
                      drafts=None, deadline: float | None = None,
                      sampling: G.Sampling | None = None, context=None,
                      rng=None) -> MTPStats:
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
        later a run still going raises TimeoutError.

        sampling (generate.Sampling) with rng (a numpy Generator): sampled, as
        Engine.generate_card(sampling=...) (docs/mtp.md 11): a0 is the sampler's pick from
        the prefill's logits (generate.reference_pick, the device's arithmetic) with the
        generator's first uniform, then the run's tokens with its next n, one per position
        (Engine._generate_inputs); the repetition penalty over `context` (default the prompt)
        and the tokens. The tokens equal the plain sampled loop's for the same uniforms."""
        samp = sampling
        if samp is not None and rng is None:
            raise ValueError("a sampled generate_card needs rng (the uniforms)")
        st = MTPStats(prompt=len(prompt))
        t0 = time.perf_counter()
        ctx = [int(t) for t in (prompt if context is None else context)]

        def pick(lg):
            u = np.minimum(np.float32(rng.random()), np.float32(1 - 2.0 ** -24))
            return G.reference_pick(lg, samp, ctx, u, self.img.cfg.S,
                                    getattr(self.img.spec, "softcap", None))
        a0, d = self.prefill(prompt, st, pick=None if samp is None else pick, lazy=True)
        st.prefill_s, st.prefill_compile_s = time.perf_counter() - t0, st.compile_s
        return self.loop_card(a0, d, max_new, stop, on_token, drafts, st, deadline,
                              samp, ctx + [a0], rng)

    def loop_card(self, a0: int, d: int, max_new: int = 32, stop=None, on_token=None,
                  drafts=None, st: MTPStats | None = None,
                  deadline: float | None = None, samp: G.Sampling | None = None,
                  context=(), rng=None, halt=None) -> MTPStats:
        """generate_card after its prefill: a0 (the token at Engine.pos, emitted) and d (the
        draft of the next position: self.draft after a prefill or a run) -> the device's run
        of the MTP loop from the committed slot (see generate_card); sampled with samp, the
        uniforms from rng and the penalty's context (the ids so far, a0 with them). halt()
        (polled while a card runs): true writes the host's stop word, as the deadline does.
        d None after a lazy prefill (self.pending, the prompt's last row at P - 1 and a0): the
        run starts with that bucket's D1 over (hid[0], a0) at P - 1, as an E's iteration would
        leave it (n = 0), which drafts d, appends the MTP layer's row and chains to the first
        verify; its MoE request is served in the run. No run (a0 a stop id, no tokens left):
        the draft runs on its own (_settle)."""
        eng, img = self.eng, self.img
        spec, block = img.spec, eng.block
        ids = list(spec.eos if stop is None else stop)
        st = MTPStats(prompt=eng.pos) if st is None else st
        st.tokens.append(a0)
        if on_token is not None:
            on_token(a0)
        P = eng.pos
        n = min(max_new - 1, img.cap - P - 2)
        lazy = d is None and self.pending is not None
        if lazy and self.pending != (P - 1, a0):
            raise ValueError(f"the pending draft is {self.pending}'s, not ({P - 1}, {a0})'s")
        if a0 in ids or n <= 0:
            self._settle(st)
            return st
        if d is None and not lazy:
            raise ValueError("loop_card needs the draft (a prefill's or a run's: self.draft)")
        b0, b1 = (P - lazy) // block + 1, (P + n) // block + 1
        progs = self._gen_programs(b0, b1, st, drafts is not None, samp)
        g = img.lookup["gen"]
        if samp is not None:        # u[P + 1 .. P + n], then a stand-in for row 1 of the
            eng._generate_inputs(samp, P, n, context, rng)      # last verify (not emitted)
            for s_ in range(img.cfg.S):
                eng.backend.write(s_, g["uni"] + 4 * (P + n + 1), np.full(1, 0.5, np.float32))
        if drafts is not None:
            dt = np.zeros(img.cap + 2, np.float32)
            dt[:len(drafts)] = np.asarray(drafts, np.float32)[:img.cap + 2]
            eng.backend.write(0, img.lookup["mtpgen"]["dtab"], dt)
            d = int(dt[P + 1])
        words = G.state_words(spec, a0, P, n, ids, block, samp)
        words[S_PAR] = self.slot
        if lazy:                    # the D1 of P - 1's bucket: an E's iteration there (n = 0),
            q = P - 1               # the next position P (tpos block: the next bucket's first)
            words[G.S_TPOS], words[S_TPOS0] = q % block + 1, q % block
            words[S_A0], words[S_A1], words[S_N] = a0, a0, 0
            first = progs[(b0, D1)]
        else:
            words[S_DRAFT] = d
            first = progs[(b0, (E0 if P % block == block - 1 else V0) + self.slot)]
        if getattr(img, "embed_host", False):   # the first program's rows: the V's (the E's
            eng._write_host_rows([a0] if lazy else [a0, d])     # row 0), the D1's a0; then
        for s_ in range(img.cfg.S):                             # each program posts the next
            eng.backend.write(s_, g["state"], words)            # one's
            eng.backend.write(s_, g["out"] + 4 * (P + 1), np.full(n, G.OUT_MARK, np.uint32))
        self.pending = None
        run = getattr(eng.backend, "run_generate", None)
        if run is not None:
            end = None if deadline is None else time.perf_counter() + deadline

            def late() -> bool:
                if end is not None:
                    now = time.perf_counter()
                    if now > end + 10.0:
                        raise TimeoutError(f"the MTP loop runs {deadline + 10:.0f} s, its "
                                           f"stop word unanswered")
                    st.timed_out = st.timed_out or now > end
                return st.timed_out or (halt is not None and bool(halt()))
            stats, got = run(first, g["out"] + 4 * (P + 1), n, on_token,
                             None if end is None and halt is None else late, g["state"])
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
        self.slot, self.draft = int(sw[S_PAR]), int(sw[S_DRAFT])
        it, acc = int(sw[S_ITER]), int(sw[S_ACC])
        st.accepted = [1] * acc + [0] * (it - acc)
        eng.pos = P + len(got)
        return st


# ---- the loop on the card (docs/mtp.md 10)
# the state block: generate.py's words, and the iteration's, which the card stores in its own
# 64-byte beat (words 32..47): not in the host's stop word's (16..31), which the host writes
# during the run (BoardBackend.run_generate: the beat alone, its words read just before)
S_DRAFT, S_A0, S_A1, S_N, S_PAR, S_TPOS0, S_ITER, S_ACC = 32, 33, 34, 35, 36, 37, 38, 39
S_END = 40          # 1.0: the iteration's D halts (a stop id, the host's word, no tokens left)
# a bucket's programs, in its chain table's order: the verify (V) and the bucket's last
# position (E) per parity, the MTP layer over two rows (D) and over one (D1)
KINDS = ("V0", "V1", "E0", "E1", "D", "D1")
NK = len(KINDS)
V0, E0, D2, D1 = 0, 2, 4, 5


def gen_alloc(b, cap: int, spec, block: int = ATTN_BLOCK,
              imem_words: int = 1 << 15, waitc: bool = False) -> dict:
    """DRAM of the MTP loop: a program slot per bucket and kind (with its wait entries on a
    WAITC device: generate.prog_bytes; a split V's or E's second part in its second half, the
    part's chain entry at the first half's end: ptab2_addr), and the chain table ([address,
    instructions] per bucket and kind); the sampled verify's row 1 logits (the generate loop's
    lg holds row 0's: generate.Sampler)."""
    nb, slot = G.buckets(cap, block), G.prog_bytes(imem_words, waitc)
    return {"progs": b.alloc(slot * NK * nb), "ptab": b.alloc(8 * NK * nb), "nb": nb,
            "slot": slot,
            "dtab": b.alloc(4 * (cap + 2)),     # tests: the draft of each position
            "lg": b.alloc(4 * G._vpad(spec))}


def gen_build(put, s: int, S: int, spec, a: dict) -> None:
    """The row 1 logits' padding: -inf after this slice's rows to a whole block (as
    generate.build's lg)."""
    v_loc = spec.vocab // S
    put(s, a["lg"], np.full(-(-v_loc // G.BLK) * G.BLK, -np.inf, np.float32))


def prog_addr(a: dict, blocks: int, kind: int) -> int:
    return a["progs"] + a["slot"] * (NK * (blocks - 1) + kind)


def ptab_addr(a: dict, blocks: int, kind: int = 0) -> int:
    return a["ptab"] + 8 * (NK * (blocks - 1) + kind)


def ptab2_addr(a: dict, blocks: int, kind: int) -> int:
    """A split kind's second part's [address, instructions] (its first part's HALT CHAIN): the
    last 8 bytes of the slot's first half, past the first part's words and wait entries (a
    program of IMEM's instructions with all its entries fills the half: the first part has
    fewer, _gen_programs). No DRAM of its own: an MTP image's layout is the same, split or
    not."""
    return prog_addr(a, blocks, kind) + a["slot"] // 2 - 8


def plan_cuts(image) -> list:
    """The layers at which a split V or E may cut (the first layer of each run of the plan but
    the first: qwen35.plan_span), the most even first (by the runs' count on either side)."""
    firsts = [f for f, _, _ in image.plan[1:]]
    n = len(image.plan)
    return [f for _, f in sorted((abs(2 * (i + 1) - n), f) for i, f in enumerate(firsts))]


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
    body[i:i] = [x for k, (v, c) in enumerate(b.run_args)
                 for x in I.rld_mul(15 - k, words[v.name], int(c),
                                    comment=f"argument {c}*{v.name}", aw=b.aw)]
    b.run_words = None


def _post_rows(m, toks) -> None:
    """With the int8 embedding table on the host (embed_host: a MoE's), the post of the next
    program's tokens (one request of their ids and count: RowServer writes their rows to the
    slot's rows 0 .. n - 1, which the next program's rows gather once served). The verify's
    (a0, a1) for the D (a0 for the D1), the draft's (t, d) for the next V (row 0 the E's)."""
    mbox = getattr(m, "embed_mbox", None)
    if mbox is None:
        return
    ids = ol.empty((len(toks),))
    for r, t in enumerate(toks):
        ids[r:r + 1].set(t)
    MB.post(mbox, ids, count=True)
    del ids


def _store_state(g, st) -> None:
    """The state block to DRAM, but the host's stop word and the sampler's."""
    ol.store(g.state[:G.S_HALT], st[:G.S_HALT])
    ol.store(g.state[S_DRAFT:S_END + 1], st[S_DRAFT:S_END + 1])


def _verify_gen(m, pos, block: int, c: int, R: int, samp: G.Sampling | None = None,
                part: int | None = None, cut: int = 0):
    """V (R = 2: rows t, d at p, p + 1) or E (R = 1: t at the bucket's last position) of
    parity c: the rows kernel (fork, hidden) with an ARGMAX per row, or with samp the sampled
    loop's pick (generate.Sampler: row r with the uniform of position p + 1 + r, row 1's
    logits in the MTP area's lg); a0 -> out[p + 1], and with the draft accepted (d == a0,
    tokens left, a0 not a stop id) a1 -> out[p + 2]; the iteration's words and the commit (the
    penalty's context gets the emitted ids); S_END at a stop id, the host's stop word or no
    tokens left; then HALT CHAIN to the bucket's D or D1 (which halts at S_END). Sampled,
    accepting d iff the pick a0 equals it is speculative sampling for the one-hot draft
    (docs/mtp.md 11.1).

    part 0 / 1 (a V or E over IMEM: compile_gen's split form): part 0 the layers before `cut`
    (the rows' embedding rows gathered), their residual to m.hid (the hidden rows, which part
    1 writes at its end), then HALT CHAIN to part 1 through its entry (ptab2_addr); part 1 the
    layers from `cut` on (the residual from m.hid), the head and the rest as above. The same
    operations: the residual's fp32 rows only pass through DRAM."""
    from .qwen35 import qwen35_rows
    b = current()
    g, a = m.gen, m.mtpgen
    st, at = _load_state(b, g)
    words = {"tok": st.base + G.S_TOK, "tok1": st.base + S_DRAFT, "tpos": st.base + G.S_TPOS}
    b.run_words = words
    span = xs = None
    if part is not None:
        span = (0, cut) if part == 0 else (cut, m.spec.layers)
        xs = m.hid
    if part == 0:
        qwen35_rows.fn(m, pos, R, [], block, None, R > 1, False, span, xs, False)  # (not nested)
        _args(b, at, words)
        G._chain(b, ptab2_addr(a, pos.blocks, (V0 if R > 1 else E0) + c),
                 what="the verify's second part")
        return
    if samp is None:
        m.lm_sinks = [G.Greedy(b, m.v_loc, head_rows_chunk(R)) for _ in range(R)]
    else:
        if b.S != 1:
            raise CompileError("the sampled MTP loop runs on one slice")
        consts = G.Sampler.constants(b, g, samp)
        m.lm_sinks = []
        for r in range(R):          # row 1 takes row 0's penalty factors of each chunk
            m.lm_sinks.append(G.Sampler(b, m, g, samp, st, pos, consts,
                                        lg=a["lg"] if r else None, u_at=pos.pos + 1 + r,
                                        pen_from=m.lm_sinks[0] if r else None))
    qwen35_rows.fn(m, pos, R, list(range(R)), block, None, R > 1, True, span, xs,
                   False)                                               # (not nested)
    if samp is not None and samp.pen and R > 1:
        _penalize_draft(b, g, st, m.lm_sinks[1])
    toks = [sk.token() for sk in m.lm_sinks]
    sinks, m.lm_sinks = m.lm_sinks, None
    a0, a1 = toks[0], toks[-1]
    _post_rows(m, toks)                 # embed_host: the D's rows (a0, a1), D1's (a0)
    left, tpos = st[G.S_LEFT:G.S_LEFT + 1], st[G.S_TPOS:G.S_TPOS + 1]
    stop = _hit(st, a0)
    ol.store(g.out[pos.pos + 1:pos.pos + 2], a0)
    if samp is not None:
        sinks[0].after(a0)
    if R > 1:
        d = st[S_DRAFT:S_DRAFT + 1]
        n = (ol.minimum(ol.abs(d - a0), 1.0) * -1.0 + 1.0) * _flag(left - 1.0) * \
            (stop * -1.0 + 1.0)
        r = b.scratch()
        b.rld(r, n, comment="accepted")
        lp = b.begin_loop(0, rcount=r)
        b.unscratch(r)
        ol.store(g.out[pos.pos + 2:pos.pos + 3], a1)
        if samp is not None:
            sinks[1].after(a1)
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
    st[S_END:S_END + 1].set(ol.maximum(ol.maximum(stop, ol.load(g.state[G.S_HALT:G.S_HALT + 1])),
                                       _flag(left * -1.0 + 1.0)))
    _args(b, at, words)
    _store_state(g, st)
    G._chain(b, ptab_addr(a, pos.blocks, D2 if R > 1 else D1), what="the draft")


def _penalize_draft(b, g, st, sk) -> None:
    """Row 1 follows the draft d, so its repetition penalty counts d as in the context (row 0's
    must not: both rows' chunks took the context's factors pa, pb). Row 1's stored logit x of
    d stays when d is in the context already (pb[d] = R > 1), else (x the raw logit then)
    becomes min(x / R, x * R), as Sampler.after's factors would give; a select by min/max with
    +-2^127, exact for finite logits (x * R / R would round). Then the maximum of d's block
    again (one slice: the row is d)."""
    d = st[S_DRAFT:S_DRAFT + 1]
    t = b.alloc((2,))
    blk = b.alloc((G.BLK,))
    r, rk, rb_ = b.scratch(), b.scratch(), b.scratch()
    b.rld(r, d * 4.0, comment="the draft's bytes")
    b.emit(I.ld(g.addr["pb"], t.base, 1, ra=r, comment="pb[d]"))
    b.emit(I.ld(sk.lg, t.base + 1, 1, ra=r, comment="row 1's logit of d"))
    pb, x = t[0:1], t[1:2]
    z = ol.minimum(x * st[G.S_PENINV:G.S_PENINV + 1], x * st[G.S_PEN:G.S_PEN + 1])
    s = (G._step(pb - 1.0) * 2.0 - 1.0) * 2.0 ** 127        # +2^127: d in the context
    y = ol.maximum(ol.minimum(z, s * -1.0), ol.minimum(x, s))
    b.emit(I.st(sk.lg, y.base, 1, ra=r, comment="penalized"))
    k = d * (1.0 / G.BLK)                                   # d's block (f2i: the floor)
    b.emit(I.rld(rk, k.base, mul=4 * G.BLK, comment="its bytes"))
    b.emit(I.rld(rb_, k.base, comment="its index"))
    b.emit(I.ld(sk.lg, blk.base, G.BLK, ra=rk, comment="d's block"))
    b.emit(I.vop(I.V_RMAX, sk.bm.base, blk.base, 0, 1, G.BLK, 1, G.BLK, 0, ra=rb_,
                 comment="its maximum"))
    for x_ in (r, rk, rb_):
        b.unscratch(x_)


def _draft_gen(m, pos, block: int, R: int, forced: bool = False):
    """D (R = 2) or D1 (R = 1): the MTP layer over the iteration's rows (hid_r, a_r) at p ..
    (qwen35_mtp), the next draft d = draft[n]; the position moves on (to the next bucket at
    its end), then HALT CHAIN to the next iteration's program: V or E (tpos = block - 1) of
    the committed parity, in this bucket or the next; HALT when the verify ended the loop
    (S_END): the MTP layer's KV cache then holds every position before the last token's, and
    S_DRAFT the draft after it, so a later run or prefill continues from there. forced
    (tests): the draft of the next iteration's row 1 (position p + 2 + n) from the host's
    table (dtab) instead."""
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
    _post_rows(m, [st[G.S_TOK:G.S_TOK + 1], st[S_DRAFT:S_DRAFT + 1]])   # the next V's (E's)
    tp = st[G.S_TPOS:G.S_TPOS + 1]                      # the next position's: 1 .. block
    wrap = _flag(tp - float(block - 1))                 # block: the next bucket's first
    tp.set(tp - wrap * float(block))
    end = _flag(tp - float(block - 2))                  # block - 1: the bucket's E
    off = (wrap * float(NK) + end * 2.0 + st[S_PAR:S_PAR + 1]) * 8.0
    _store_state(g, st)
    G._chain(b, ptab_addr(a, pos.blocks), cond=st[S_END:S_END + 1] * -1.0 + 1.0, off=off,
             what="the next iteration")
    b.emit(I.halt())


def compile_gen(image, blocks: int, kind: int, block: int = ATTN_BLOCK,
                forced: bool = False, samp: G.Sampling | None = None,
                part: int | None = None, cut: int = 0) -> list | None:
    """Program KINDS[kind] of bucket `blocks` of the MTP loop (docs/mtp.md 10), one per slice;
    None when the bucket has no such position (E and D1 past the KV cache's end). forced
    (tests): D and D1 take the drafts from the host's table (_draft_gen). samp: V and E
    sample (docs/mtp.md 11), D and D1 are the same either way. part 0 / 1: a V's or E's
    split form, cut at layer `cut` (a run's first: plan_cuts)."""
    name = KINDS[kind]
    if part is not None and (name[0] not in "VE" or cut not in plan_cuts(image)):
        raise ValueError(f"a split {name}: a V or E cut at a run of the plan")
    R = 1 if name in ("E0", "E1", "D1") else 2
    t0 = (blocks - 1) * block
    lo = t0 + block - 1 if R == 1 else max(t0, image.spec.conv_k - 1)
    if lo + R > image.cap or "mtpgen" not in image.lookup:
        if "mtpgen" not in image.lookup:
            raise ValueError("the MTP loop needs an MTP image with lookup tables")
        return None
    rp = RunRows(blocks, block, lo, image.lookup["zmask"], image.cap, R)
    c = int(name[1]) if name[0] in "VE" else 0
    fn, kw = ((_verify_gen, {"c": c, "R": R, "samp": samp, "part": part, "cut": cut})
              if name[0] in "VE" else (_draft_gen, {"R": R, "forced": forced}))
    return [ol.jit(fn).trace(image.cfg, s, {"m": image.descriptors(s, c), "pos": rp,
                                            "block": block, **kw}).finish()
            for s in range(image.cfg.S)]
