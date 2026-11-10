"""A prompt's prefill from programs at run-time positions (docs/prefill.md).

The host writes the prompt's tokens to the generate area's out[] (the token at each position),
then runs one program per run of up to R_max rows: P (rows), L (the prompt's last run: its last
row's logits) and, on an MTP image, M (the MTP layer over the run's rows) after each. A program
takes its position from the generate state's tpos word and its tokens from out[], so it serves
every run of its bucket and kind, and the engine keeps it (and the program cache across
processes): the host compiles nothing per prompt. The rows before conv_k - 1 (a new context's
first run) are at compile-time positions, cached the same way.

The runs' split depends only on the first position, the prompt's length, R_max and the attention
block: R_max rows a run, cut where a run would cross a bucket's end (a run-time run's rows stay in
its bucket). R_max, per bucket: the most rows in fit_chunk's sizes (one MXU pass of MCOLS rows,
then whole passes, at most the image's rows; Qwen3.5: one pass) whose L program in that bucket
compiles and fits IMEM (a later bucket's attention makes a longer
program: Phi-4-mini's mix takes 3 rows in bucket 1, 1 in bucket 16). Plain and MTP engines of a
model take the same split, so their logits are the same bit for bit. Where today's prefill
(compile_rows) would fit more rows in a bucket than its prompt run, a prompt that reaches the
bucket takes today's route (covers).

Tiled runs (opentpu/llm/tiled.py, docs/tiled_prefill.md), on by default (off with the engine's
tile_runs False, or OTPU_TILE=0 where the engine does not say: tile_flag): a bucket's P and L
runs are tile_run's programs of up to T rows (_probe_tile: whole passes of MCOLS up to ACT_ROWS
and the image's rows, every run a prompt can ask for in the bucket fitting TMEM and IMEM), each
weight streamed once a run; a bucket where no tile fits, or where today's prompt runs (covering
it) take more rows than the tile (tiled), takes today's prompt runs. MTP's M runs stay today's.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

from .. import progcache as PC
from ..compiler import CompileError
from . import generate as G
from . import tiled as TL
from .qwen3 import RUN_WORDS


def split(p0: int, P: int, R_max, block: int, K: int) -> list[tuple[int, int, str]]:
    """The runs (first position, rows, kind) of the prompt positions [p0, P). R_max: the most
    rows of a run, an int, or R_max(blocks) for bucket `blocks` (0: the first run of a new
    context, before conv_k - 1)."""
    rm = R_max if callable(R_max) else (lambda blocks: R_max)
    runs, p = [], p0
    while p < P:
        if p >= K - 1:                      # a run-time run: its rows in one bucket
            n = min(rm(p // block + 1), P - p, block - p % block)
        else:                               # a new context's first run (R_max(0))
            n = min(rm(0), P - p)
        runs.append((p, n, "L" if p + n == P else "P"))
        p += n
    return runs


def conv_k(eng) -> int:
    """The model's convolution taps (Qwen3.5's DeltaNet, LFM2's): its first conv_k - 1
    positions run at compile-time positions; 1 without."""
    return getattr(eng.spec, "conv_k", 1)


def supported(eng) -> bool:
    """The engine's image runs prompts this way: a dense model's resident image, without
    rows the host writes before each run unless its prompt runs read them (prompt_host_rows:
    Gemma 4's PLE records on the host, written before each run)."""
    img = eng.image
    host = (eng._host_rows is not None and bool(eng._host_rows([0]))) or \
        getattr(eng, "row_server", None) is not None
    return (hasattr(img, "compile_prompt_run") and bool(getattr(img, "lookup", None))
            and getattr(eng.spec, "moe", None) is None and not getattr(eng.spec, "experts", 0)
            and (not host or getattr(img, "prompt_host_rows", False)))


def tile_flag(t) -> bool:
    """Tiled prompt runs asked for: t (an engine's tile_runs), or where it is None OTPU_TILE:
    on unless it is 0 (unset or empty: the default, on; in the reference cache's key as it is
    set: tools/qual/refs.py env_key). Their bits are today's prompt runs' (docs/tiled_prefill.md
    11)."""
    if t is None:
        t = os.environ.get("OTPU_TILE", "").strip() != "0"
    return bool(t)


def layer_tile_flag(t, default: bool = False) -> bool:
    """A MoE model's tiled expert-major layer runs asked for (Engine.layer_tile, with
    expert_major), and so its image's mask tiles as it is built. t (an engine's tile_runs) True,
    or where it is None OTPU_TILE: for a family whose tiles are on by default (`default`:
    Qwen3.5-MoE's, tiled.LAYER_ROWS: their whole forms are today's bits, docs/tiled_prefill.md
    14) on unless it is 0, as the prompt runs' tile_flag; for the others (Gemma 4's: opt-in)
    set and not 0."""
    if t is None:
        v = os.environ.get("OTPU_TILE", "").strip()
        t = v != "0" if default else v not in ("", "0")
    return bool(t)


def tile_scratch(spec, layer_runs: bool, default: bool = False) -> bool:
    """An image holds the tiled runs' I/O scratch (io "o": the attention's outputs a part at a
    time, loaded back whole: tiled.py): a dense model's (its tiled prompt runs, the default);
    a MoE model's only where its family has tiled layer runs (layer_runs) and they are asked for
    as the image is built (layer_tile_flag, the family's `default`), so a MoE image's layout and
    expert slots without them stay as they were."""
    moe = getattr(spec, "moe", None) is not None or bool(getattr(spec, "experts", 0))
    return not moe or (layer_runs and layer_tile_flag(None, default))


def tiling(eng) -> bool:
    """Tiled prompt runs on (opentpu/llm/tiled.py): tile_flag of the engine's tile_runs, for
    an image whose family has a tile kernel, on one slice (tiled.check: a configuration of
    more, the design point's two, takes today's runs)."""
    img = eng.image
    return tile_flag(getattr(eng, "tile_runs", None)) and TL.family(img) is not None \
        and img.cfg.S == 1


def programs(eng, p: int, R: int, kind: str, hidden: bool = False, slot: int = 0,
             tile: bool | None = None):
    """The run's (programs, the state words it reads) (kept by the engine; through its
    program cache): the programs, or on the card (runs_words) the program assembled once, so
    a run of another program loads its words, not an assembly of them (3-5 ms a run; MTP's
    rows and M runs alternate). hidden, slot: MTP's rows (their hidden stored, the states'
    slot); M's take neither. tile: a tiled run (True), today's (False), or the bucket's
    choice (None: tiled)."""
    done = eng.__dict__.setdefault("_prompt_progs", {})
    what, blocks, compile = _what(eng, p, R, kind, hidden, slot, tile)
    with _lock(eng):
        if what not in done:
            done[what] = _prepared(eng, what, blocks, *eng.cached(what, compile))
    return done[what]


def _what(eng, p: int, R: int, kind: str, hidden: bool = False, slot: int = 0,
          tile: bool | None = None):
    """A run's (program cache key, bucket (0: compile-time positions), compile()); tile as
    programs'."""
    img, block = eng.image, eng.block
    if kind == "M":
        hidden, slot = False, 0
    kw = dict(hidden=hidden, slot=slot) if hidden or slot else {}
    if p < conv_k(eng) - 1:
        blocks, kw["p0"] = 0, p
        what = ("prompt", kind, R, "at", p, hidden, slot)
    else:
        blocks = p // block + 1
        if tile is None:
            tile = kind != "M" and tiled(eng, blocks)
        if tile and kind != "M":
            what = ("prompt tile", kind, R, blocks, hidden, slot)
            return what, blocks, lambda: TL.compile_tile_run(img, blocks, R, kind, block, **kw)
        what = ("prompt", kind, R, blocks, hidden, slot)
    return what, blocks, lambda: img.compile_prompt_run(blocks, R, kind, block, **kw)


def _prepared(eng, what, blocks: int, progs, ra):
    """programs()' entry of a compiled run (blocks 0: a compile-time one)."""
    img = eng.image
    if not G.fits(img, progs):
        raise CompileError(f"prompt run {what}: {max(map(len, progs))} instructions, "
                           f"IMEM {img.cfg.IMEM_WORDS // 8}")
    # the words the run reads: its arguments', and those its addresses add with a loop's
    # step (Program.run_seen: a register loaded from the word before the loop, no argument)
    names = {getattr(v, "var", v).name for v, _ in ra or ()} | \
        {v.name for p in progs for v, _ in getattr(p, "run_seen", ())}
    prep = getattr(eng.backend, "prepare", None)
    if getattr(eng.backend, "runs_words", False) and len(progs) == 1:   # (Words: run_args;
        from .qwen3 import worker_words                                 # WAITC: the entries)
        progs = worker_words(img, progs[0], ra or getattr(progs[0], "run_args", ()))
    elif prep is not None:
        prep(progs)
    if blocks:                      # a run-time run: and the words its kernel reads itself
        names |= set(getattr(img, "prompt_words", ()))
    return progs, sorted(names)


def r_max(eng, blocks: int = 1) -> int:
    """The most rows of a run in bucket `blocks`: fit_chunk's sizes (one MXU pass, then whole
    passes) up to the image's rows (Qwen3.5: one pass), the bucket's (plain) L program
    compiled and fitting IMEM (kept by the engine). MTP's runs take the same R_max (their
    programs must fit it), so plain and MTP prefills of a model split alike. With the engine's program cache the answer is kept there
    too (progcache.fact), so a new process does not try the larger R again (the 4B's R = 4,
    traced until TMEM runs out). Tiled runs (tiled): the bucket's tile rows."""
    if tiled(eng, blocks):
        return _tile_fit(eng, blocks)
    return _fit(eng, blocks)[0]


def tiled(eng, blocks: int) -> bool:
    """Bucket `blocks`'s P and L runs tiled: tiling on, a tile fits the bucket, and today's
    prompt runs (which cover it) take fewer rows there. Each run is a pass of the weights, so
    the form with more rows a run streams them less often (release r2: LFM2-2.6B's tile took 1
    row at MCOLS 4 and 2 at 8 where today's runs take 4, and its chat prefill ran at half the
    speed of the untiled one); on a tie today's runs, which were 0.4-1.2% faster at equal rows
    and bytes on every production dense model (docs/tiled_prefill.md 12.4). Where a tile takes
    as many rows as today's runs could (Qwen3-0.6B's 16) nothing more is compiled to know it:
    no production model's today's runs reach those rows."""
    if not (blocks and tiling(eng)):
        return False
    T = _tile_fit(eng, blocks)
    if not T:
        return False
    if T >= _probe_top(eng, blocks):
        return True
    try:
        R, covered = _fit(eng, blocks)
    except CompileError:                # (no prompt run of today's fits the bucket)
        return True
    return not (covered and R >= T)


def _probe_top(eng, blocks: int) -> int:
    """The largest of _probe's run sizes in bucket `blocks` (blocks > 0)."""
    from .qwen3 import _whole_passes
    img, mc = eng.image, eng.image.cfg.MCOLS
    p = max((blocks - 1) * eng.block, conv_k(eng) - 1)
    return _whole_passes(min(img.rows, img.cap - p,
                             mc if hasattr(eng.spec, "mtp") else img.rows), mc)


def covers(eng, p0: int, P: int) -> bool:
    """Prompt runs take the positions [p0, P): in every bucket they touch a run takes today's
    rows (compile_rows, fit_chunk's; Qwen3.5's up to one MXU pass), so they stream no weight
    more often and run no more often (docs/prefill.md 7); else Engine.prefill_chunks and
    MTPDecoder.prefill take today's route for the prompt."""
    P = min(P, eng.cap)                 # (past the cache: the route's own error)
    return all(tiled(eng, b) or _fit(eng, b)[1]
               for b in range(p0 // eng.block + 1, (P - 1) // eng.block + 2))


def _fit(eng, blocks: int):
    """(R_max, covered) of bucket `blocks` (_probe), kept by the engine and the program
    cache."""
    done = eng.__dict__.setdefault("_prompt_rmax", {})
    with _lock(eng):
        if blocks not in done:
            what = ("prompt fit", blocks)
            done[blocks] = tuple(PC.fact(eng.layout, what, lambda: _probe(eng, blocks))
                                 if eng.prog_cache else _probe(eng, blocks))
        return done[blocks]


def _probe(eng, blocks: int) -> list:
    """[R_max, covered] of bucket `blocks`: the largest of fit_chunk's run sizes (up to MCOLS
    rows, then whole passes of MCOLS up to the image's rows; Qwen3.5's, whose MTP runs take
    R_max too, one pass at most) at which every run a prompt can ask for in the bucket fits
    (_kinds: P and L, MTP's rows and M, at every R' <= R: the split cuts a run at the prompt's
    and the bucket's end), then prefer_rows' choice (an odd R's R - 1 where PAIR makes it
    cheaper per row); covered when compile_rows (today's prefill) does not fit the next larger
    size there either, or would take no more rows by the same choice (a prompt program larger
    than today's would run fewer rows). blocks 0: a new context's first run, at compile-time
    positions 0 .. conv_k - 2 (convolutions), sized on its own (LFM2.5-230M's fits 8 rows at a
    run-time position, 4 from position 0; LFM2-2.6B's 4 from 0, not from 1), covered."""
    from .qwen3 import _whole_passes, prefer_rows
    img, block, mc, K = eng.image, eng.block, eng.image.cfg.MCOLS, conv_k(eng)
    ps = range(K - 1) if not blocks else [max((blocks - 1) * block, K - 1)]
    n = _whole_passes(min(img.rows, img.cap - ps[-1],
                          mc if hasattr(eng.spec, "mtp") else img.rows), mc)
    sizes = []
    while n:
        sizes.append(n)
        n = _whole_passes(n - 1, mc)
    kinds = _kinds(eng)
    for i, R in enumerate(sizes):
        if all(_fits(lambda: programs(eng, p, r, k, h, tile=False)) for r in range(R, 0, -1)
               for k, h in kinds for p in ps):
            break
    else:
        raise CompileError(f"no prompt run fits bucket {blocks}")

    def cost(r):                    # a run of r rows without logits, its programs
        what, _, compile = _what(eng, ps[0], r, "P", tile=False)
        return eng.cached(what, compile)[0]
    R = prefer_rows(img, R, cost)
    if i == 0 or not blocks:
        return [R, True]
    up = sizes[i - 1]               # today's next larger size: fits, and would take more rows?
    return [R, not _fits(lambda: _today(eng, ps[0], up)) or
            prefer_rows(img, up, lambda r: _today(eng, ps[0], r, logits=False)) <= R]


def _tile_fit(eng, blocks: int) -> int:
    """The tile rows of bucket `blocks` (_probe_tile; 0: none fits), kept by the engine and
    the program cache."""
    done = eng.__dict__.setdefault("_prompt_tile", {})
    with _lock(eng):
        if blocks not in done:
            what = ("prompt tile fit", blocks)
            done[blocks] = int(PC.fact(eng.layout, what, lambda: _probe_tile(eng, blocks))
                               if eng.prog_cache else _probe_tile(eng, blocks))
        return done[blocks]


def _probe_tile(eng, blocks: int) -> int:
    """The rows T of a tiled run in bucket `blocks` (from conv_k - 1): the largest of the sizes
    up to ACT_ROWS, the image's rows and the bucket's room (whole passes of MCOLS above
    MCOLS rows, as _probe's) at which every run a prompt can ask for in the bucket fits (_kinds
    at every R <= T: tiled P and L, today's M), then an odd T's T - 1 where PAIR makes it
    cheaper a row (prefer_rows' rule: PREFER_MARGIN less MXU time a row); 0 where none fits."""
    from .qwen3 import PREFER_MARGIN, _whole_passes, mxu_time
    img, block, mc = eng.image, eng.block, eng.image.cfg.MCOLS
    p = max((blocks - 1) * block, conv_k(eng) - 1)
    room = min(img.rows, img.cfg.act_rows, img.cap - p, block - p % block)
    if room <= 0:
        return 0
    n, sizes = _whole_passes(room, mc), []
    while n:
        sizes.append(n)
        n = _whole_passes(n - 1, mc)
    kinds = _kinds(eng)
    for T in sizes:
        if all(_fits(lambda: programs(eng, p, r, k, h, tile=k != "M")) for r in range(T, 0, -1)
               for k, h in kinds):
            break
    else:
        return 0
    if T >= 3 and T % 2:
        def cost(r):
            what, _, compile = _what(eng, p, r, "P", tile=True)
            return mxu_time(eng.cached(what, compile)[0]) / r
        if cost(T - 1) <= (1 - PREFER_MARGIN) * cost(T):
            T -= 1
    return T


def _kinds(eng) -> list:
    """The (kind, hidden) of the runs a prompt asks for: P and L; an MTP engine's rows (their
    hidden) and its MTP layer's M."""
    if getattr(eng.spec, "mtp", False):
        return [("L", True), ("P", True), ("M", False)]
    return [("L", False), ("P", False)]


def _fits(compile) -> bool:
    """compile() compiles and fits IMEM; False where TMEM, ACT RAM or IMEM do not hold it."""
    try:
        compile()
    except CompileError as e:
        if not any(w in str(e) for w in ("TMEM", "ACT RAM", "IMEM")):
            raise
        return False
    return True


def _today(eng, p: int, n: int, logits: bool = True):
    """Today's run of n rows at p (compile_rows), the prompt's last with logits; its programs,
    CompileError where they do not fit."""
    img = eng.image
    progs = img.compile_rows([(0, p + j) for j in range(n)], [n - 1] if logits else [],
                             eng.block, **eng._tokens_kw([0] * n))
    if not G.fits(img, progs):
        raise CompileError(f"rows at {p}: {max(map(len, progs))} instructions, IMEM")
    return progs


def _lock(eng):
    """The engine's lock over its prompt programs (warm's thread compiles them too)."""
    return eng.__dict__.setdefault("_prompt_lock", threading.RLock())


def warm(eng) -> None:
    """Bucket 1's prompt programs ahead, on a thread, when the engine starts: R_max, a new
    context's first (compile-time) run, and the bucket's P and L runs, MTP's (hidden rows, the
    MTP layer) on an MTP image, from the program cache or compiled, while the host waits for
    a prompt (a chat). A prompt that comes first takes the lock in turn: none is compiled
    twice. Engines without the pipeline (the simulators' tests) warm nothing."""
    if not (eng.prompt_runs and eng.pipeline and supported(eng)):
        return

    def go():
        try:
            R, K = r_max(eng, 1), conv_k(eng)
            mtp = bool(getattr(eng.spec, "mtp", False))
            todo = [(0, r_max(eng, 0), "P")] if K > 1 else []
            todo += [(K - 1 if K > 1 else 0, r, k) for r in range(R, 0, -1)
                     for k in (("P", "L") if r == R else ("L",))]
            for p, r, k in todo:
                programs(eng, p, r, k, mtp, 0)
                if mtp:
                    programs(eng, p, r, "M")
        except Exception:           # (the prompt meets the same error and reports it)
            pass
    eng._prompt_warm = threading.Thread(target=go, daemon=True, name="otpu-prompt-warm")
    eng._prompt_warm.start()


def write_tokens(eng, p0: int, tokens) -> None:
    """The prompt's tokens to out[p0 ..] (every slice's)."""
    g = eng.image.lookup["gen"]
    t = np.asarray(tokens, np.float32)
    for s in range(eng.cfg.S):
        eng.backend.write(s, g["out"] + 4 * p0, t)


def words(eng, names, p: int, R: int) -> dict:
    """The state words (qwen3.RUN_WORDS) of a run of R rows at p: tpos; LFM2's ring, the
    first row of the last K in its state ring ((p + 1) mod K), and ringo, the rotation of the
    ring after the run (-(p + R) mod K, lfm2._ring_store)."""
    K = conv_k(eng)
    v = {"tpos": p % eng.block, "ring": (p + 1) % K, "ringo": -(p + R) % K}
    return {RUN_WORDS[n]: v[n] for n in names}


def run(eng, entry, p: int, R: int, tokens=()) -> dict:
    """One run at position p (programs' entry): the state words it reads written first, and
    on an image whose prompt runs read host rows (prompt_host_rows) the run's tokens' rows."""
    progs, names = entry
    if getattr(eng.image, "prompt_host_rows", False):
        eng._write_host_rows(tokens)
    if names:
        g, w = eng.image.lookup["gen"], words(eng, names, p, R)
        for a, b in _ranges(sorted(w)):
            x = np.array([w[i] for i in range(a, b)], np.float32)
            for s in range(eng.cfg.S):
                eng.backend.write(s, g["state"] + 4 * a, x)
    st = eng.backend.run(progs)
    st["rows"] = R
    return st


def _ranges(ws):
    """Runs [a, b) of consecutive word indices."""
    out = []
    for i in ws:
        if out and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return out


def logits(eng, R: int) -> np.ndarray:
    """The L run's last row's logits."""
    io, S, v, v_loc = eng.image.io, eng.cfg.S, eng.spec.vocab, eng.image.v_loc
    return np.concatenate([eng.backend.read(s, io["logits"] + 4 * ((R - 1) * v + s * v_loc),
                                            4 * v_loc).view(np.float32) for s in range(S)])


def chunks(eng, tokens, head: bool = True):
    """Engine.prefill_chunks's runs this way: yields (the run's tokens, logits after the
    prompt's last token with the last run, else None; without `head` every run is a P)."""
    tokens = [int(t) for t in tokens]
    p0 = eng.poss[0]
    P = p0 + len(tokens)
    if P > eng.cap:
        raise RuntimeError("KV cache full")
    if not tokens:
        return
    write_tokens(eng, p0, tokens)
    for p, R, kind in split(p0, P, lambda b: r_max(eng, b), eng.block, conv_k(eng)):
        kind = kind if head else "P"
        progs = programs(eng, p, R, kind)
        if kind == "L":
            eng._prefetch(P)            # the first decode step's, while the device runs
        eng.stats.append(run(eng, progs, p, R, tokens[p - p0:p - p0 + R]))
        eng.poss[0] = p + R
        yield tokens[p - p0:p - p0 + R], logits(eng, R) if kind == "L" else None


def mtp(dec, tokens, st, pick=None, on_run=None, follow: int | None = None):
    """MTPDecoder.prefill's runs this way: each run's rows (their hidden stored, the states in
    the decoder's slot), then the MTP layer over them (M: the tokens after the rows, out[P] the
    prompt's last logits' pick a0, or `follow`: a part of a prompt, every run a P). Returns
    (a0, the first draft); on_run as the decoder's."""
    eng = dec.eng
    toks = [int(t) for t in tokens]
    p0 = eng.pos
    P = p0 + len(toks)
    a0 = draft = None
    t0 = time.perf_counter()
    runs = split(p0, P, lambda b: r_max(eng, b), eng.block, conv_k(eng))
    if follow is not None:
        runs = [(p, R, "P") for p, R, _ in runs]
    st.compile_s += time.perf_counter() - t0
    write_tokens(eng, p0, toks if follow is None else toks + [int(follow)])
    for p, R, kind in runs:
        t0 = time.perf_counter()
        rows, mp = programs(eng, p, R, kind, True, dec.slot), programs(eng, p, R, "M")
        st.compile_s += time.perf_counter() - t0
        st.runs.append(("prefill", R, run(eng, rows, p, R)))
        if kind == "L":
            lg = logits(eng, R)
            a0 = int(np.argmax(lg)) if pick is None else int(pick(lg))
            write_tokens(eng, P, [a0])
        st.runs.append(("mtp prefill", R, run(eng, mp, p, R)))
        draft = dec._drafts(R)[R - 1]
        eng.pos = p + R
        if on_run is not None and on_run(toks[p - p0:p - p0 + R]) and p + R < P:
            return None, None
    return a0, draft
