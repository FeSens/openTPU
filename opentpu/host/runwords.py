"""A prompt's layer runs at S > 1 read from the program cache as their words.

At S > 1 an Engine has no compile workers: prefill_layers' runs come from the program cache
(Engine._layer_run through Engine.cached), whose entries are each slice's assembled words. The
cache's reader decodes them into instructions (progcache._load: compiler.Program), and the
board assembles the same instructions again (BoardBackend.start) to load them: on the card host,
for each of the 35B's layer runs of a new bucket, 13.4 ms the decode and 23.2 ms the start (the
x2 chat's 30,870-token prompt, window 4 2026-11-04: 4,989 and 5,047 such runs, 184 s of its
668 s TTFT).

install(eng) gives the engine a _layer_run that reads a run's entry as its words (compiler.Words,
each slice's, with the run's arguments) and hands those to the board: no decode, no assembly;
the wait entries are waits.entries' of the same words and arguments (kept on the same disk, the
same key). A run not in the cache, or one the cache holds another way (a split program, one
slice's words), goes the engine's way (compiled, then written to the cache). Host work only: the
programs are the cache's, word for word.
"""
from __future__ import annotations

import json
import os

import numpy as np


def read(layout, what, S: int) -> tuple | None:
    """The cached program of `what` for this layout as (each of the S slices' words,
    compiler.Words with the run's arguments; the arguments, as progcache.get gives them), or
    None when the cache has no such entry."""
    from opentpu import progcache as PC
    from opentpu.compiler import Words, run_var
    d = PC.cache_dir()
    if d is None:
        return None
    k = PC.key(layout, what)
    f = d / k[:2] / f"{k}.npz"
    try:
        with np.load(f) as z:
            meta = json.loads(bytes(z["meta"]).decode())
            if meta["split"] or meta["kinds"] != ["i"] or meta["S"] != [S]:
                return None
            ra = None if meta["ra"] is None else [(run_var(r[0], *r[2:]), r[1])
                                                  for r in meta["ra"]]
            words = [Words.of(np.asarray(z[f"p0_{s}"], np.uint32), ra) for s in range(S)]
        os.utime(f)                     # (recently used: the cache's eviction order)
    except Exception:                   # (none, or damaged: the engine's way reads or compiles it)
        return None
    PC.stats["disk"] += 1
    return words, ra


def install(eng) -> bool:
    """eng's prompt runs as their cached words (an Engine at S > 1 with the program cache on a
    board that runs words); True when installed."""
    from opentpu import progcache as PC
    if eng.cfg.S == 1 or not eng.prog_cache or PC.cache_dir() is None or \
            not getattr(eng.backend, "runs_words", False):
        return False
    run = eng._layer_run

    def layer_run(key):
        ck = eng._slot_key(key)
        got = eng._layer_runs.get(ck)
        if got is None:
            what = eng._layer_what(key) if hasattr(eng, "_layer_what") else \
                ("layer", key, eng.block, eng.layer_hint, eng.expert_major, eng._layer_slot)
            hit = read(eng.layout, what, eng.cfg.S)     # (the `what` Engine._layer_run asks for)
            if hit is None:
                return run(key)
            got = eng._layer_runs[ck] = (None, hit[1], hit[0])
        return got

    eng._layer_run = layer_run
    return True
