"""Batched decode steps on the device: one run feeds a token to each of several sequences (an
Engine(batch=B) image's KV slots), every weight streamed once for all of them, and picks each
row's next token on the device, greedy or with the row's own sampling (the decode loop's
sampler, opentpu/llm/generate.py), so the host writes the rows' tokens and reads back one
word per row: no logits leave the card and the host computes nothing of the model (the server's
continuous batching, opentpu/serve/scheduler.py).

The batch area (DRAM past the image, Engine(batch_area=True)):

    io        tin[B] the rows' input token ids (fp32 words), uni[B] the rows' uniforms (sampled
              rows; the host's generator of each request), step (the run's step, below): the
              host's, one write per run
    tout[B]   the rows' picked token ids (the device's)
    per slot: st[STATE_WORDS] (the sampler's words: top-k, log2(e) / T, top-p, the penalty R
              and 1 / R), lg[V] (the sampler's logits), pa[V], pb[V] (the repetition
              penalty's factors over the sequence's context, generate.py's)

A step's program (compile_step) takes rows (slot, position) and per row a kind: None (greedy:
ARGMAX) or a Sampling key (kmax, penalty); the tokens come from tin (the embedding rows gathered
or loaded on the device from the image's tables, the RoPE rows from its tables), every layer is
qwen3's rows kernel (each row appends its K / V to its slot's cache and attends over its own
positions), and the LM head hands each chunk to every row's sink. The row's pick goes to
tout[r]; a sampled row with a penalty adds it to its slot's context. Per row the arithmetic is
the decode step's (qwen3_rows: bit-identical to step()), and the sampler is the decode loop's
(generate.reference_pick on the logits with the same uniform).

With a span n, the program serves n steps of its rows: row r is at position p_r + t, t the
run-time step (0 <= t < n) in the io block's step word, which the program loads and multiplies
into its argument and address registers (RLD; qwen3.RunWords' way). Every row's position adds
the same t, so the rows share those registers. n keeps every row in its attention block and
V^T tile (Steps.span), so a program serves ~block / (R + 1) steps on average: the host
compiles one per span, ahead, instead of one per step.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from ..compiler import Affine, CompileError, RunVar, Tensor, current, vt_tile
from . import generate as G

GREEDY = None
LIVE = 64                       # live programs kept (row sets that may run again)
AHEAD = 16                      # programs compiling ahead at once (past it the oldest key's
                                # result is dropped: keys that never came)
PAD_AHEAD = 4                   # a padded group's own program: compiled for this many steps on
FEED = "feed"                   # a row that only feeds its token (a prompt's): no LM head, no pick


def alloc(image, cfg, B: int, base: int) -> dict:
    """The batch area's addresses for B slots from `base` (bytes; page-aligned end)."""
    V = G._vpad(image.spec)
    R = max(B, cfg.MCOLS)                   # rows a run takes (a prompt's feed: MCOLS of a slot)
    lk = image.lookup or {}
    a = {"B": B, "R": R, "base": base, "V": V, "cap": image.cap, "zmask": lk.get("zmask", 0),
         "mask_blocks": lk.get("mask_blocks", 1)}
    o = base

    def take(n):
        nonlocal o
        x = o
        o = -(-(o + n) // 128) * 128
        return x
    io = take(4 * (2 * R + 2 + SLOT_WORDS * B))     # the host's words: one write per run
    a["io"], a["tin"], a["uni"], a["step"] = io, io, io + 4 * R, io + 8 * R
    a["rw"] = io + 4 * (2 * R + 2)          # a step over slots: each row's words (slot_rows)
    a["tout"] = take(4 * R)
    a["st"] = [take(4 * G.STATE_WORDS) for _ in range(B)]
    a["lg"] = [take(4 * V) for _ in range(B)]
    a["pa"] = [take(4 * V) for _ in range(B)]
    a["pb"] = [take(4 * V) for _ in range(B)]
    a["end"] = -(-o // 4096) * 4096
    return a


def build(images, a: dict, spec, cfg) -> None:
    """The area's first contents in the image arrays: penalty factors 1, the logits buffers
    -inf past this slice's rows (as the decode loop's)."""
    V, v_loc = a["V"], spec.vocab // cfg.S
    ones = np.ones(V, np.float32).view(np.uint8)
    lg = np.full(V, -np.inf, np.float32)
    lg = lg[:-(-v_loc // G.BLK) * G.BLK].view(np.uint8)
    for img in images:
        for s in range(a["B"]):
            img[a["pa"][s]:a["pa"][s] + len(ones)] = ones
            img[a["pb"][s]:a["pb"][s] + len(ones)] = ones
            img[a["lg"][s]:a["lg"][s] + len(lg)] = lg


def kind_of(samp) -> tuple | None:
    """A row's compile-time kind: None (greedy, no penalty) or the Sampling's key."""
    return None if samp is None else samp.key


def _mixed(kinds) -> bool:
    """Feed rows beside picking ones (a run with two run-time values)."""
    return any(k == FEED for k in kinds) and any(k != FEED for k in kinds)


def _offsets(rows, base, kinds, n):
    """The run-time value(s) a program compiled at `base` with span n serves rows at: t (all
    rows at base + t, t < n), or with n = (span, feed span) (t, f): the picking rows at
    base + t, the feed rows at base + f; None if it does not."""
    if not isinstance(n, tuple):
        d = {p - b for (_, p), b in zip(rows, base)}
        t = d.pop() if len(d) == 1 else -1
        return t if 0 <= t < n else None
    out = []
    for feed, m in ((False, n[0]), (True, n[1])):
        d = {p - b for (_, p), b, k in zip(rows, base, kinds) if (k == FEED) == feed}
        t = d.pop() if len(d) == 1 else -1
        if not 0 <= t < m:
            return None
        out.append(t)
    return tuple(out)


def step_rows(rows, span, block: int, zmask: int, cap: int, kinds=None) -> list:
    """rows [(slot, position)] at step 0 -> [(slot, qwen3.RunPos)] at position + t, t the
    run-time step below span, one RunVar for every row (each row's bucket: the blocks up to
    its position's, the last one masked at position + t, as RunPos's). span (n, m) with kinds:
    the picking rows' t below n, the feed rows' own run-time value f below m (a mixed run:
    the rows advance at their own rates)."""
    from ..kernels.attention import Bucket
    from .qwen3 import RunPos
    n, m = span if isinstance(span, tuple) else (span, span)
    t = RunVar("tpos", bound=n)
    f = RunVar("fpos", bound=m) if isinstance(span, tuple) else t
    out = []
    for i, (s, p) in enumerate(rows):
        feed = kinds is not None and kinds[i] == FEED
        v, span = (f, m) if feed else (t, n)
        if (p % block) + span > block:
            raise CompileError(f"position {p} + {span} steps crosses an attention block")
        rp = object.__new__(RunPos)
        rp.blocks, rp.block, rp.lo = p // block + 1, block, p
        rp.t0 = (rp.blocks - 1) * block
        rp.tpos, rp.pos = v + (p - rp.t0), v + p
        rp.bucket = Bucket(rp.blocks, Affine(zmask + 4 * (cap - 1 - p)) + v * -4)
        out.append((s, rp))
    return out


SLOT_WORDS = 5          # qwen3.SlotPos.WORDS: a row's words in the io block's rw


def slot_rows(rows, rw: int, block: int, zmask: int, cap: int, d: int, wg: int = 1) -> list:
    """rows [(slot, _)] -> [(slot, qwen3.SlotPos)]: row r at a run-time position, its words at
    TMEM rw + SLOT_WORDS r (from the io block's rw: SlotPos.values, wg the KV caches' groups),
    so one program serves every position of the rows' slots."""
    from .qwen3 import SlotPos
    return [(s, SlotPos(r, rw + SLOT_WORDS * r, block, zmask, cap, d, wg=wg))
            for r, (s, _) in enumerate(rows)]


@ol.jit
def batch_step(m, rows, kinds, block: int, area: dict, span: int = 0):
    from .qwen3 import _attention_rows, _embed_word, _layers, _mlp
    b = current()
    spec, R = m.spec, len(rows)
    if R > area["R"]:
        raise CompileError(f"{R} rows for a run of {area['R']}")
    last = {}
    for (s, p), kind in zip(rows, kinds):      # a slot's rows: consecutive positions, all but
        if s in last and (last[s][0] != p - 1 or last[s][1] != FEED):    # the last feed
            raise CompileError(f"slot {s}: rows at consecutive positions, only its last picks")
        last[s] = (p, kind)
    if span:                                    # the run's step: TMEM words for the RLDs
        rows = step_rows(rows, span, block, area["zmask"], area["cap"], kinds)
        stw = ol.load(Tensor(area["step"], (2,), (1,)))
        at = b.stack[-1][-1]
        b.run_words = {"tpos": stw.base, "fpos": stw.base + 1}
    x = ol.empty([R, m.xr.shape[1]], dense=True)
    c = ol.empty([R, m.cosr.shape[1]], dense=True)
    s_ = ol.empty([R, m.sinr.shape[1]], dense=True)
    for r, (_, p) in enumerate(rows):
        q = p.pos if span else p
        ol.load(m.cos_t[q:q + 1, :], out=c[r:r + 1, :])
        ol.load(m.sin_t[q:q + 1, :], out=s_[r:r + 1, :])
    tk = ol.load(Tensor(area["tin"], (R,), (1,)))
    for r in range(R):
        x[r:r + 1, :].set(_embed_word(m, tk[r:r + 1]))
    del tk
    for lw in _layers(m):
        x.set(_attention_rows(x, lw, c, s_, rows, spec, block))
        x.set(_mlp(x, lw, spec))
    del c, s_
    # the sinks: one per row that picks (the others only fed their tokens)
    pick = [r for r, k in enumerate(kinds) if k != FEED]
    if pick:
        _sinks(b, m, spec, x, rows, kinds, pick, area)
    del x
    if span:                                    # the arguments from the step word, at the start
        body = b.stack[-1]
        i = next(j for j, x_ in enumerate(body) if x_ is at) + 1
        body[i:i] = [x_ for k, (v, c_) in enumerate(b.run_args)
                     for x_ in I.rld_mul(15 - k, b.run_words[v.name], int(c_),
                                         comment=f"argument {c_}*{v.name}", aw=b.aw)]
        b.run_words = None
        del stw


def _sinks(b, m, spec, x, rows, kinds, pick, area) -> None:
    """The LM head for the rows `pick` of x, each chunk to every such row's sink (Greedy, or
    the slot's Sampler with the row's uniform); row r's token to tout[r]. At S > 1 each slice's
    sinks see its vocabulary rows: Greedy gathers the slices' maxima, the Sampler their top k
    (selected again: the k largest of all), so every slice picks the same id."""
    from .qwen3 import head_rows_chunk
    if len(pick) < len(rows):
        xs = ol.empty([len(pick), x.cols], dense=True)
        for i, r in enumerate(pick):
            xs[i:i + 1, :].set(x[r:r + 1, :])
        x = xs
    chunk = area.get("chunk") or head_rows_chunk(len(pick))
    g0 = m.gen
    uni = Tensor(area["uni"], (area["R"],), (1,))
    consts, sts, sinks = {}, [], []
    for r in pick:
        (slot, _), kind = rows[r], kinds[r]
        if kind is GREEDY:
            sinks.append(G.Greedy(b, m.v_loc, chunk))
            sts.append(None)
            continue
        samp = _Key(*kind)
        if kind not in consts:
            consts[kind] = G.Sampler.constants(b, g0, samp)
        st = ol.load(Tensor(area["st"][slot], (G.STATE_WORDS,), (1,)))
        g = SimpleNamespace(addr={"lg": area["lg"][slot], "pa": area["pa"][slot],
                                  "pb": area["pb"][slot]}, uni=uni, iota=g0.iota)
        sinks.append(G.Sampler(b, m, g, samp, st, None, consts[kind], u_at=r))
        sts.append(st)
    _head(x, m, spec, sinks, chunk)
    if len(pick) < len(rows):
        del x
    for r, sk in zip(pick, sinks):
        t = sk.token()
        ol.store(Tensor(area["tout"] + 4 * r, (1,), (1,)), t)
        if kinds[r] is not GREEDY:
            sk.after(t)
        del t


def _head(x, m, spec, sinks, chunk: int) -> None:
    """The final norm and this slice's LM head rows for every row of x, each chunk's logits
    handed to every row's sink (qwen3._lm_head_rows with the chunk given)."""
    from ..kernels.lib import rmsnorm
    sid = ol.program_id()
    xs = ol.quantize(rmsnorm(x, ol.load(m.g_final), spec.eps))
    for c0 in range(0, m.v_loc, chunk):
        n = min(chunk, m.v_loc - c0)
        y = ol.dot(xs, m.head[c0:c0 + n, :])
        for r, sk in enumerate(sinks):
            sk(y[r:r + 1, :], sid * m.v_loc + c0)
        del y


class _Key:
    """A Sampling as its compile-time key: kmax and the penalty (the rest are state words)."""

    def __init__(self, kmax: int, pen: bool):
        self.kmax, self.pen = kmax, pen
        self.key = (kmax, pen)


def compile_step(image, area: dict, rows, kinds, block: int, span: int = 0,
                 slices=None) -> list:
    """One program per slice: the batched step for rows [(slot, position)] of kinds; with a
    span, for the steps t < span at positions + t (the step word: t). slices: those slices'
    programs only (default every one)."""
    rows, kinds = [(int(s), int(p)) for s, p in rows], list(kinds)
    return [batch_step.trace(image.cfg, s, {"m": image.descriptors(s), "rows": rows,
                                            "kinds": kinds, "block": block, "area": area,
                                            "span": span})
            .finish() for s in (range(image.cfg.S) if slices is None else slices)]


def _worker_step(rows, kinds, area, span=0, s=0):
    """The engine's compile worker process: compile_step of slice s, assembled (with its wait
    entries on a WAITC configuration: qwen3.worker_words)."""
    from .qwen3 import _WORKER, worker_words
    image, block = _WORKER
    return worker_words(image, compile_step(image, area, rows, kinds, block, span, [s])[0])


class _Slices:
    """A program's slices compiling in Steps' slice workers: done() when all are, result()
    the slices' words in order (a Future's two calls that Steps makes)."""

    def __init__(self, futs):
        self.futs = futs

    def done(self) -> bool:
        return all(f.done() for f in self.futs)

    def result(self):
        return [f.result() for f in self.futs]


class Steps:
    """An Engine's batched steps (Engine(batch=B, batch_area=True)): each slot's sampling
    (set_sampling), the step (run). A step's rows run in groups whose program fits IMEM (the
    LM head's sinks grow it per row: 4 rows of Qwen3-0.6B with sampling and a penalty fit, 8
    greedy ones do not), found by compiling and kept per (kinds, attention buckets).

    span (default): a program serves its rows for `span(rows)` steps (the run's step in the io
    block), so the same program runs again, loaded in IMEM already, until a row reaches the
    end of its attention block; the next one (each row there) is compiled ahead as soon as a
    program starts serving, on the engine's compile worker or a thread, and with depth d the
    d - 1 after it too (rows at different places in their blocks: a short span's successor
    gets its predecessor's time as well). span=False: one program per step (positions at
    compile time), the next step's compiled while a step runs.

    On a backend that runs assembled words (the card's) every program comes assembled, with
    its wait entries on a WAITC configuration, so a run loads it as it is. At S > 1 (no engine
    compile workers there) the slices compile at once in Steps' own S worker processes (slice
    s's trace, words and entries in worker s): the 35B's step at B = 2 on x2 in about a
    slice's time, ~0.6 s, against 1.8 s for the four in turn."""

    worker = staticmethod(_worker_step)   # (the compile worker's function: a model's own steps,
    #                                       batch35.Steps)

    def __init__(self, eng, span: bool = True):
        if getattr(eng, "bat", None) is None:
            raise ValueError("batched steps need Engine(batch=B > 1, batch_area=True)")
        self.eng, self.a = eng, eng.bat
        self.span_steps = bool(span)
        self.depth = 1                    # span programs compiled ahead of the one serving
        self.kinds: dict = {}             # slot -> its kind
        self.limit = eng.cfg.IMEM_WORDS // 8
        self._fit: dict = {}              # signature (kinds, buckets) -> the program fits IMEM
        self._ready: dict = {}            # key -> programs (compiled, not run yet)
        self._next: dict = {}             # key -> Future of its programs
        # (slots, kinds, positions at step 0) -> (span, programs), the least recently used first
        self._live: OrderedDict = OrderedDict()
        self.live = LIVE                  # (a warm's programs all stay: warm_steps raises it)
        self._pool = None
        self._words = bool(getattr(eng.backend, "runs_words", False))
        self._spool = self._sready = None   # S > 1: the slices' compile worker processes
        if self._words and eng.cfg.S > 1 and eng.pipeline and not eng._procs:
            self._start_slices()
        self.runs = 0
        # where a step's time goes: device cycles, the host's waits for programs (compiles
        # not ready), the whole runs; programs compiled, runs of a program already loaded
        self.timing = {"cycles": 0, "compile_wait_s": 0.0, "compiled_inline": 0, "run_s": 0.0,
                       "programs": 0, "reused": 0, "padded": 0, "write_s": 0.0,
                       "start_s": 0.0, "wait_s": 0.0, "read_s": 0.0}
        self._pad_to: dict = {}           # (slots, kinds) -> the rows its padding waits for

    def set_sampling(self, slot: int, samp, context=()) -> None:
        """Slot `slot` picks greedily (samp None) or with samp (generate.Sampling): its state
        words, and with a penalty its factors over `context` (the ids so far, the token it
        feeds next among them)."""
        eng, a = self.eng, self.a
        self.kinds[slot] = kind_of(samp)
        if samp is None:
            return
        st = G.state_words(eng.spec, 0, 0, 0, [], eng.block, samp)
        for s in range(eng.cfg.S):
            eng.backend.write(s, a["st"][slot], st)
        if samp.pen:
            V = a["V"]
            pa, pb = np.ones(V, np.float32), np.ones(V, np.float32)
            ix = np.unique(np.asarray(list(context), np.int64))
            pa[ix] = np.float32(1.0) / np.float32(samp.penalty)
            pb[ix] = samp.penalty
            for s in range(eng.cfg.S):
                eng.backend.write(s, a["pa"][slot], pa)
                eng.backend.write(s, a["pb"][slot], pb)

    # ---- programs
    def span(self, rows) -> int:
        """The steps one program serves from rows' positions: each row stays in its attention
        block and its V^T tile, below the capacity (0: span off)."""
        if not self.span_steps:
            return 0
        eng = self.eng
        blk, tb = eng.block, vt_tile(eng.cap)
        return max(1, min(min(blk - p % blk, tb - p % tb, eng.cap - p) for _, p in rows))

    def _key(self, rows, kinds=None):
        rows = tuple((int(s), int(p)) for s, p in rows)
        kinds = tuple(self.kinds.get(s) for s, _ in rows) if kinds is None else tuple(kinds)
        if _mixed(kinds) and self.span_steps:   # (the picking rows' span, the feed rows')
            return rows, kinds, (self.span([r for r, k in zip(rows, kinds) if k != FEED]),
                                 self.span([r for r, k in zip(rows, kinds) if k == FEED]))
        return rows, kinds, self.span(rows)

    def _sig(self, key):
        rows, kinds, n = key
        blk = self.eng.block
        return (tuple(sorted(zip(map(repr, kinds), (p // blk for _, p in rows)))),
                2 if isinstance(n, tuple) else bool(n))

    def _start_slices(self) -> None:
        """S compile worker processes (spawned, as the engine's: qwen3._worker_init's image
        of the engine's layout), one slice's program each; ended with this Steps."""
        import multiprocessing as mp
        import weakref
        from concurrent.futures import ProcessPoolExecutor

        from .qwen3 import _worker_init, _worker_ready
        eng = self.eng
        self._spool = ProcessPoolExecutor(
            eng.cfg.S, mp_context=mp.get_context("spawn"), initializer=_worker_init,
            initargs=(eng.spec, eng.cfg, eng.cap, eng.batch, eng.rows, eng.block,
                      eng._image_kw, eng.image.stream_fill))
        self._sready = [self._spool.submit(_worker_ready) for _ in range(eng.cfg.S)]
        weakref.finalize(self, self._spool.shutdown, wait=False, cancel_futures=True)

    def _slices(self, key):
        """key's programs compiling in the slice workers (_Slices), or None (none, or not
        started yet)."""
        if self._spool is None or not all(f.done() for f in self._sready):
            return None
        rows, kinds, n = key
        return _Slices([self._spool.submit(self.worker, rows, kinds, self.a, n, s)
                        for s in range(self.eng.cfg.S)])

    def _trace(self, key) -> list:
        """key's programs, one per slice (traced here)."""
        rows, kinds, n = key
        return compile_step(self.eng.image, self.a, rows, kinds, self.eng.block, n)

    def _compile(self, key):
        progs = self._trace(key)
        if self._words:                 # assembled once: the backend loads the words as they are
            from .qwen3 import worker_words                 # (WAITC: with their entries)
            ws = [worker_words(self.eng.image, p) for p in progs]
            return ws[0] if len(ws) == 1 else ws
        return progs

    def _size(self, progs) -> int:
        if isinstance(progs, np.ndarray):
            return len(progs) // 8
        return len(progs[0]) // 8 if isinstance(progs[0], np.ndarray) else len(progs[0])

    def _get(self, key):
        """key's programs: compiled ahead, or now."""
        if key in self._ready:
            return self._ready.pop(key)
        t0 = time.perf_counter()
        try:
            fut = self._next.pop(key, None)
            if fut is not None:
                return fut.result()
            self.timing["compiled_inline"] += 1
            fut = self._slices(key)
            return self._compile(key) if fut is None else fut.result()
        finally:
            self.timing["compile_wait_s"] += time.perf_counter() - t0

    def _blocks(self, rows, kinds=None) -> list:
        """rows (and kinds) by blocks of feed_rows() slots, in slot order."""
        q = self.feed_rows()
        kinds = [None] * len(rows) if kinds is None else list(kinds)
        out = {}
        for r, k in zip(rows, kinds):
            out.setdefault(r[0] // q, ([], []))
            out[r[0] // q][0].append(r)
            out[r[0] // q][1].append(k)
        return [out[b] for b in sorted(out)]

    def plan_known(self, rows, kinds) -> list[list] | None:
        """plan's groups for rows of kinds when every group's IMEM fit is known (else None),
        compiling nothing."""
        out = []
        for br, bk in self._blocks(rows, kinds):
            g = self._fit_known(br, bk)
            if g is None:
                return None
            out += g
        return out

    def _fit_known(self, rows, kinds):
        fit = self._fit_of(self._sig(self._key(rows, kinds)))
        if fit is None:
            return None
        if fit or len(rows) == 1:
            return [rows] if fit else None
        h = -(-len(rows) // 2)
        a, b = self._fit_known(rows[:h], kinds[:h]), self._fit_known(rows[h:], kinds[h:])
        return None if a is None or b is None else a + b

    def prefetch_rows(self, rows, kinds) -> None:
        """Compile ahead a coming step's programs: rows of kinds (a slot's kind may change by
        then) in plan's groups, or each block's rows while their IMEM fit is unknown (plan
        then takes the compile for its fit check)."""
        groups = self.plan_known(rows, kinds) or [br for br, _ in self._blocks(rows, kinds)]
        k = dict(zip((s for s, _ in rows), kinds))
        for g in groups:
            kg = tuple(k[s] for s, _ in g)
            if self._serving(g, kg) is None:
                self._prefetch(self._key(g, kg))

    def plan(self, rows) -> list[list]:
        """The step's rows in groups whose programs fit IMEM: each block of feed_rows() slots'
        rows (an MXU pass takes that many rows: more in a run cost another pass; and a row
        that leaves changes its block's program only), halved while a program does not fit."""
        out = []
        for br, _ in self._blocks(list(rows)):
            out += self._plan_fit(br)
        return out

    def _fit_of(self, sig):
        """sig's IMEM fit: known, or told by a known one (fewer rows of the same kinds and
        buckets make a smaller program: a subset of a fit fits, a superset of a misfit does
        not), else None."""
        if sig in self._fit:
            return self._fit[sig]
        from collections import Counter
        c = Counter(sig[0])
        for k, fit in self._fit.items():
            if k[1] != sig[1]:
                continue
            ck = Counter(k[0])
            if fit and not c - ck:
                return True
            if not fit and not ck - c:
                return False
        return None

    def _plan_fit(self, rows) -> list[list]:
        rows = list(rows)
        key = self._key(rows)
        sig = self._sig(key)
        if self._fit_of(sig) is not None:
            self._fit.setdefault(sig, self._fit_of(sig))
        if sig not in self._fit:
            if len(self._ready) > 8:
                self._ready.clear()
            progs = self._get(key)
            self._fit[sig] = self._size(progs) <= self.limit
            if self._fit[sig]:
                self._ready[key] = progs
        if self._fit[sig] or len(rows) == 1:
            if not self._fit[sig]:
                raise CompileError(f"one row's step does not fit IMEM ({self.limit})")
            return [rows]
        h = -(-len(rows) // 2)
        return self._plan_fit(rows[:h]) + self._plan_fit(rows[h:])

    def _prefetch(self, key) -> None:
        """Compile key's programs in the background (if not there or on the way already)."""
        eng = self.eng
        if key is None or key in self._ready or key in self._next:
            return
        if self._fit_of(self._sig(key)) is False:
            return
        while len(self._next) > AHEAD:           # (keys that never came)
            self._next.pop(next(iter(self._next))).result()
        rows, kinds, n = key
        if eng._procs and eng._ready is not None and eng._ready.done():
            self._next[key] = eng._pool.submit(self.worker, rows, kinds, self.a, n)
            return
        if not eng.pipeline:
            return
        fut = self._slices(key)
        if fut is not None:
            self._next[key] = fut
            return
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="otpu-batch-compile")
        self._next[key] = self._pool.submit(self._compile, key)

    def _serving(self, grp, kinds=None):
        """(programs, t) of a live program that serves grp at its step t, or None."""
        rows = tuple(grp)
        slots = tuple(s for s, _ in rows)
        kinds = tuple(self.kinds.get(s) for s in slots) if kinds is None else tuple(kinds)
        for k in reversed(list(self._live)):
            if k[0] != slots or k[1] != kinds:
                continue
            n, progs = self._live[k]
            t = _offsets(rows, k[2], kinds, n)
            if t is not None:
                self._live.move_to_end(k)
                return progs, t
        return None

    def _coming(self, grp, kinds):
        """A program compiled or compiling (not live yet) that serves grp at some offset: its
        key, or None (the latest first)."""
        slots = tuple(s for s, _ in grp)
        kinds = tuple(self.kinds.get(s) for s in slots) if kinds is None else tuple(kinds)
        for key in list(self._ready)[::-1] + list(self._next)[::-1]:
            rows, kk, n = key
            if kk == kinds and tuple(s for s, _ in rows) == slots and n and \
                    _offsets(tuple(grp), tuple(p for _, p in rows), kinds, n) is not None:
                return key
        return None

    def _take(self, grp, kinds=None, adv: int = 1):
        """grp's programs and its step word: the live program's, or a new one made live (one
        compiled ahead that serves grp, else grp's own; and its successor compiled ahead:
        each row where the program's span ends, the rows advancing `adv` positions a run)."""
        got = self._serving(grp, kinds)
        if got is not None:
            self.timing["reused"] += 1
            return got
        key = self._coming(grp, kinds)
        if key is None:
            key = self._key(self._canon(grp, kinds), kinds)
        progs = self._get(key)
        self.timing["programs"] += 1
        rows, kinds, n = key
        slots = tuple(s for s, _ in rows)
        k = (slots, kinds, tuple(p for _, p in rows))
        self._live[k] = (n, progs)
        self._live.move_to_end(k)
        while len(self._live) > self.live:
            self._live.popitem(last=False)
        if isinstance(n, tuple):                # (a mixed run's: no successor)
            return progs, _offsets(tuple(grp), tuple(p for _, p in rows), kinds, n)
        if n and (len(grp) == self.feed_rows() or any(k != FEED for k in kinds)):
            nxt = rows                          # (not after a prompt's partial feed run)
            for _ in range(self.depth):         # its successors: each row where a span ends
                nxt = [(s, p + -(-n // adv) * adv) for s, p in nxt]
                if any(p >= self.eng.cap for _, p in nxt):
                    break
                key = self._key(nxt, kinds)
                self._prefetch(key)
                n = key[2]
        return progs, grp[0][1] - rows[0][1]

    def _canon(self, grp, kinds) -> list:
        """The rows a new program for grp is compiled at: a feed run's from its attention
        block's start plus its phase in feed_rows() (so that one program serves every prompt's
        runs on that slot in that block), else grp's own positions."""
        if not self.span_steps or kinds is None or not any(k == FEED for k in kinds):
            return list(grp)
        blk = self.eng.block
        if _mixed(kinds):                       # the feed rows from their block's start
            fp = next(p for (_, p), k in zip(grp, kinds) if k == FEED)
            b, j, out = (fp // blk) * blk, 0, []
            for (s, p), k in zip(grp, kinds):
                out.append((s, p) if k != FEED else (s, b + j))
                j += k == FEED
            return out
        q, p = self.feed_rows(), grp[0][1]
        b = (p // blk) * blk + p % q
        return [(s, b + j) for j, (s, _) in enumerate(grp)]

    def feed_rows(self) -> int:
        """Tokens a feed run takes: the MXU's columns (one pass of every weight)."""
        return min(self.eng.cfg.MCOLS, self.a["R"])

    def feed(self, slot: int, tokens) -> None:
        """Feed a prompt's tokens to `slot` at its position, in runs of up to feed_rows() (from a
        multiple of it on, so that a span program serves an attention block's runs): the KV
        cache only, no LM head."""
        eng, k = self.eng, self.feed_rows()
        tokens = [int(t) for t in tokens]
        i = 0
        while i < len(tokens):
            p = eng.poss[slot]
            n = min(len(tokens) - i, k - p % k)
            if p + n > eng.cap:
                raise RuntimeError("KV cache full")
            grp = [(slot, p + j) for j in range(n)]
            self._go(grp, (FEED,) * n, tokens[i:i + n], None, n)
            i += n

    def warm(self) -> None:
        """Compile in the background each slot's feed program of the first attention block
        (every prompt's first runs; a server's first requests)."""
        q, kinds = self.feed_rows(), (FEED,) * self.feed_rows()
        for s in range(self.a["B"]):
            self._prefetch(self._key([(s, j) for j in range(q)], kinds))

    def prefetch_feed(self, slot: int, p: int, n: int) -> None:
        """Compile ahead the feed run of n tokens (< feed_rows()) at p: a prompt's last run
        (unless a live program serves it)."""
        if 0 < n:
            grp, kinds = [(slot, p + j) for j in range(n)], (FEED,) * n
            if self._serving(grp, kinds) is None:
                self._prefetch(self._key(self._canon(grp, kinds), kinds))

    def _io(self, grp, toks, uni, t) -> np.ndarray:
        """A run's io words: the rows' tokens, their uniforms (or none), the step word(s) t."""
        Rm = self.a["R"]
        io = np.zeros(2 * Rm + 2, np.float32)
        io[:len(grp)] = toks
        if uni is not None:
            io[Rm:Rm + len(grp)] = np.minimum(np.asarray(uni, np.float32),
                                              np.float32(1 - 2.0 ** -24))
        io[2 * Rm], io[2 * Rm + 1] = t if isinstance(t, tuple) else (t, 0)
        return io

    def _go(self, grp, kinds, toks, uni, adv, ahead_key=None):
        """One run of grp (its live program or a new one) with tokens toks, uniforms uni (or
        None); while it runs, ahead_key's programs compile. Returns tout's words."""
        eng, a, tm = self.eng, self.a, self.timing
        progs, t = self._take(grp, kinds, adv)
        c0 = time.perf_counter()
        io = self._io(grp, toks, uni, t)
        for s in range(eng.cfg.S):
            eng.backend.write(s, a["io"], io)
        t0 = time.perf_counter()
        tm["write_s"] += t0 - c0
        start = getattr(eng.backend, "start", None)
        if start is not None:
            start(progs)
            t1 = time.perf_counter()
            self._prefetch(ahead_key)
            st = eng.backend.wait()
            tm["start_s"] += t1 - t0
            tm["wait_s"] += time.perf_counter() - t1
        else:
            st = eng.backend.run(progs)
            self._prefetch(ahead_key)
        self.timing["run_s"] += time.perf_counter() - t0
        self.timing["cycles"] += int((st or {}).get("cycles") or 0)
        eng.stats.append(dict(st or {}, rows=len(grp)))
        self.runs += 1
        for s, _ in grp:
            eng.poss[s] += 1
        if kinds is not None and all(k == FEED for k in kinds):
            return None
        t2 = time.perf_counter()
        out = eng.backend.read(0, a["tout"], 4 * len(grp)).view(np.float32)
        tm["read_s"] += time.perf_counter() - t2
        return out

    # ---- the step
    def _mixed_group(self, rows, fslot: int, k: int):
        eng = self.eng
        rows = [(int(s), int(p)) for s, p in rows]
        grp = rows + [(fslot, eng.poss[fslot] + j) for j in range(k)]
        return grp, tuple(self.kinds.get(s) for s, _ in rows) + (FEED,) * k

    def mixed_ready(self, rows, fslot: int, k: int) -> bool | None:
        """Whether the mixed run of rows and k feed rows of fslot (mixed) has its program at
        hand: True (live, compiled, or compiled ahead and done; with no compile pipeline,
        compiled now), False (compiling: started now for the rows one step on, the feed rows'
        base being their block's), None (it does not fit IMEM)."""
        eng = self.eng
        grp, kinds = self._mixed_group(rows, fslot, k)
        if self._serving(grp, kinds) is not None:
            return True
        key = self._coming(grp, kinds)          # compiled or compiling ahead, serving grp
        if key is None:
            key = self._key(self._canon(grp, kinds), kinds)
            if self._fit_of(self._sig(key)) is False:
                return None
            if eng.pipeline or eng._procs:      # for the rows one step on (it serves this
                one = [(s, p + 1) for s, p in grp[:len(rows)]] + grp[len(rows):]   # span)
                if all(p < eng.cap for _, p in one):
                    self._prefetch(self._key(self._canon(one, kinds), kinds))
                return False
        elif key in self._next and not self._next[key].done():
            return False
        sig = self._sig(key)
        if self._fit_of(sig) is None or key not in self._ready:
            progs = self._get(key)
            self._fit[sig] = self._size(progs) <= self.limit
            if not self._fit[sig]:
                return None
            self._ready[key] = progs
        return self._fit_of(sig) is not False

    def mixed(self, rows, tokens, uniforms, fslot: int, ftoks) -> list[int]:
        """One run of the decoding rows (as run's, one group) and a prompt's feed rows: fslot's
        next len(ftoks) positions, in the MXU pass's spare rows. Its program takes two run-time
        values (the decoding rows advance one a run, the feed rows len(ftoks)) and has its
        feed rows compiled at their attention block's start, so it serves the prompt's runs;
        compiled inline unless mixed_ready said True. Returns the decoding rows' picks."""
        eng = self.eng
        for s, p in rows:
            if p != eng.poss[s] or p >= eng.cap:
                raise ValueError(f"slot {s} is at {eng.poss[s]}, not {p}")
        grp, kinds = self._mixed_group(rows, fslot, len(ftoks))
        uni = None if uniforms is None else list(uniforms) + [0.5] * len(ftoks)
        got = self._go(grp, kinds, list(tokens) + [int(t) for t in ftoks], uni, 1)
        self.timing["mixed"] = self.timing.get("mixed", 0) + 1
        return [int(t) for t in got[:len(rows)]]

    def _pad(self, grp, free):
        """A live program that serves grp and rows of free slots (a request left: its slot's
        row stays, fed a dummy token past its context, its pick dropped) while grp's own
        program is not ready: (its rows, kinds), or None. grp's own program is compiled for
        PAD_AHEAD steps on, once."""
        eng = self.eng
        kinds = tuple(self.kinds.get(s) for s, _ in grp)
        key = self._key(grp, kinds)
        if self._serving(grp, kinds) is not None or key in self._ready or key in self._next:
            return None
        slots = {s for s, _ in grp}
        pos = dict(grp)
        for (lslots, lkinds, base), (n, _) in reversed(list(self._live.items())):
            extra = [s for s in lslots if s not in slots]
            if not extra or not slots <= set(lslots) or any(s not in free for s in extra):
                continue
            lk = dict(zip(lslots, lkinds))
            if any(lk[s] != k for (s, _), k in zip(grp, kinds)):
                continue
            ts = {pos.get(s, eng.poss[s]) - b for s, b in zip(lslots, base)}
            if len(ts) != 1:
                continue
            t = ts.pop()
            if not 0 <= t < n or any(eng.poss[s] >= eng.cap for s in extra):
                continue
            sk = (tuple(sorted(slots)), kinds)
            ahead = [(s, p + PAD_AHEAD) for s, p in grp]
            if self._pad_to.get(sk, (None,))[0] is None or \
                    min(p for _, p in self._pad_to[sk]) <= min(p for _, p in grp):
                if all(p < eng.cap for _, p in ahead):
                    self._prefetch(self._key(ahead, kinds))
                    self._pad_to[sk] = ahead
            return [(s, b + t) for s, b in zip(lslots, base)], tuple(lkinds)
        return None

    def run(self, rows, tokens, uniforms=None, ahead=True, free=()) -> list[int]:
        """Feed tokens[r] to slot rows[r][0] at its position rows[r][1] (the slot's next);
        returns each row's picked token. uniforms: the sampled rows' (one per row). ahead:
        while a group runs, compile the next group's program (span off: after the last, the
        next step's first, each row one position on). free: slots with no request, whose
        rows may pad a live program while a group's own one compiles (_pad)."""
        eng = self.eng
        rows = [(int(s), int(p)) for s, p in rows]
        for s, p in rows:
            if p != eng.poss[s]:
                raise ValueError(f"slot {s} is at {eng.poss[s]}, not {p}")
            if p >= eng.cap:
                raise RuntimeError("KV cache full")
        tok = {s: int(t) for (s, _), t in zip(rows, tokens)}
        uni = None if uniforms is None else {s: u for (s, _), u in zip(rows, uniforms)}
        groups = self.plan(rows)
        nxt_rows = [(s, p + 1) for s, p in rows if p + 1 < eng.cap]
        out = {}
        for gi, grp in enumerate(groups):
            ahead_key = None
            if ahead:
                if gi + 1 < len(groups):
                    if self._serving(groups[gi + 1]) is None:
                        ahead_key = self._key(groups[gi + 1])
                elif nxt_rows and not self.span_steps:
                    k1 = self._key(nxt_rows)
                    if self._fit.get(self._sig(k1)):
                        ahead_key = k1
                    elif self._fit.get(self._sig(k1)) is False:
                        ahead_key = self._key(self.plan(nxt_rows)[0])
            pad = self._pad(grp, set(free)) if free and self.span_steps else None
            if pad is not None:
                ext, lk = pad
                self.timing["padded"] += 1
                got = self._go(ext, lk, [tok.get(s, 0) for s, _ in ext],
                               [uni.get(s, 0.5) if uni else 0.5 for s, _ in ext], 1, ahead_key)
                for (s, _), t_ in zip(ext, got):
                    if s in tok:
                        out[s] = int(t_)
                continue
            got = self._go(grp, None, [tok[s] for s, _ in grp],
                           None if uni is None else [uni[x] for x, _ in grp], 1, ahead_key)
            for (s, _), t_ in zip(grp, got):
                out[s] = int(t_)
        return [out[s] for s, _ in rows]

    def drop(self) -> None:
        """Forget the programs compiled ahead and the live ones."""
        for fut in self._next.values():
            fut.result()
        self._next.clear()
        self._live.clear()
