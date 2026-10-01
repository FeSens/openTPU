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

from .qwen3 import Engine, fit_chunk

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
