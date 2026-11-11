"""What the server can say about its programs' files: whether the program cache saves, and how far
a long prompt's runs are on disk (docs/dashboard.md, "Program saves").

progcache._store and waits.entries skip a write that would leave under progcache.free_floor() on
the disk, without a word. A server under the floor then compiles and computes a position's
programs and wait entries again on every pass: on the two-card 35B 23-30 s of wait entries a
512-token chunk against 0.04 s from the disk, a chunk's 328 entries past waits.MEM_N's 256.
This module only reads: opentpu/serve is not in progcache.source(), so no program or image key
depends on it."""
from __future__ import annotations

import math
import shutil
import time
from pathlib import Path


def saves() -> dict:
    """The program cache now, cheap enough for each of the monitor's ticks: saving (None without
    a disk cache), the disk's free bytes against the floor, and this process's counts of
    programs compiled and written, wait entries computed (with a compile pool, its reads too)
    and their seconds."""
    from .. import progcache as PC
    from .. import waits as WC
    out = {"dir": None, "saving": None, "free_bytes": None, "floor_bytes": PC.free_floor(),
           "under_floor_bytes": None, "programs_compiled": PC.stats["compile"],
           "programs_written": PC.stats["write"], "entries_computed": WC.stats["computed"],
           "entries_s": round(WC.stats["seconds"], 3), "entries_read": WC.stats["disk"]}
    d = PC.cache_dir()
    if d is None:
        return out
    probe = next((p for p in (Path(d), *Path(d).parents) if p.exists()), None)
    try:
        free = shutil.disk_usage(probe).free
    except (OSError, TypeError):
        return dict(out, dir=str(d))
    return dict(out, dir=str(d), saving=free >= PC.free_floor(), free_bytes=free,
                under_floor_bytes=max(0, PC.free_floor() - free))


def saves_line(s: dict, label: str) -> str | None:
    """The startup's line on saves() (None: no disk cache to say anything of)."""
    if s["free_bytes"] is None:
        return None
    mib = (s["free_bytes"] >> 20, s["floor_bytes"] >> 20)
    if s["saving"]:
        return f"{label}: program saves on ({mib[0]:,} MiB free in {s['dir']}'s disk, floor {mib[1]:,})"
    return (f"{label}: program saves OFF: {s['dir']}'s disk has {mib[0]:,} MiB free, under the "
            f"{mib[1]:,} MiB floor; programs compiled and wait entries computed are not kept")


def _run_keys(eng, b: int) -> list:
    """Two of bucket b's layer runs (warm35x2.layer_keys' last ones): the last layer's R- and
    1-row runs. warm_layers writes a slot's layers in order, so a warm stopped in a bucket
    misses these first. None: an engine whose runs have another shape."""
    R, li = eng.layer_major, eng.spec.layers - 1
    if eng.layer_hint or eng.expert_major:
        return None
    host = bool(getattr(eng.image, "embed_host", False)) and not eng.embed_runs
    flag = R > 1 and not (li == 0 and (R == 1 or host))
    keys = [(li, b, n, flag) for n in dict.fromkeys((R, 1))]
    return [("L",) + k[:3] for k in keys] if eng.layer_loop and R > 1 else keys


def _prompt_whats(eng, b: int, d: Path) -> list | None:
    """Four of bucket b's prompt runs on a dense model's prefill (warm35x2.warm_prompts writes a
    bucket's largest run first, its 1-row runs last). None: the bucket's fit is not on disk, and
    asking prefill for it would compile."""
    from .. import progcache as PC
    from ..llm import prefill as PF
    k = PC.key(eng.layout, ("prompt fit", b))
    if not (d / k[:2] / f"{k}.json").is_file():
        return None
    p = max((b - 1) * eng.block, PF.conv_k(eng) - 1)
    return [PF._what(eng, p, r, kind, hidden)[0] for r in dict.fromkeys((PF.r_max(eng, b), 1))
            for kind, hidden in PF._kinds(eng)]


def _entries_there(eng, what, d: Path) -> bool:
    """The wait entries of a run's programs are on disk (each slice's, as BoardBackend.prepare
    would look them up)."""
    from .. import isa as I
    from .. import progcache as PC
    from .. import waits as WC

    def refuse():
        raise FileNotFoundError(what)
    progs, _ = PC.get(eng.layout, what, refuse)
    if isinstance(progs, tuple):
        return True                     # (a split engine's parts: not a layer run's shape)
    cfg = eng.cfg
    for p in progs:
        k = WC.entries_key(I.assemble(p), cfg.D, cfg.S, getattr(p, "run_args", None), cfg.AW,
                           cfg.WC_WIN)
        if not (d / k[:2] / f"{k}.npz").is_file():
            return False
    return True


def warm_reach(eng, samples: int = 32) -> dict | None:
    """How far a prompt's runs are on disk, for the startup's line: the buckets from the first on
    whose sampled runs (_run_keys, every slot's; _prompt_whats on a dense model's prefill) the
    program cache holds, and on about `samples` of them, the last one too, those programs' wait
    entries when the backend loads any. A sample, not a walk of every file: a report, nothing
    depends on it. None: no prompt runs to look at (neither layer-major nor prefill's prompts, no
    disk cache, another shape of run)."""
    from .. import progcache as PC
    from ..llm import prefill as PF
    from ..llm import qwen3 as Q
    d = PC.cache_dir()
    if not getattr(eng, "prog_cache", False) or d is None:
        return None
    t0 = time.perf_counter()
    B, nb = eng.block, math.ceil(eng.cap / eng.block)
    slots = range(max(1, int(getattr(eng, "batch", 1) or 1)))
    if getattr(eng, "layer_major", 0):
        if _run_keys(eng, 1) is None:
            return None

        def whats(b):
            return [Q.layer_what(k, B, eng.layer_hint, eng.expert_major, s)
                    for k in _run_keys(eng, b) for s in slots]
    elif getattr(eng, "prompt_runs", False) and PF.supported(eng) and not PF.tiling(eng):
        def whats(b):
            return _prompt_whats(eng, b, Path(d))
    else:
        return None
    kept, reach = set(PC._mem), 0
    for b in range(1, nb + 1):
        w = whats(b)
        if w is None or not all(PC.has(eng.layout, x) for x in w):
            break
        reach = b
    checked = 0
    if reach and getattr(eng.backend, "waitc", False):
        step = max(1, reach // samples)
        for b in [*range(step, reach, step), reach]:
            checked += 1
            if not all(_entries_there(eng, x, Path(d)) for x in whats(b)):
                reach = b - 1
                break
    for k in set(PC._mem) - kept:       # (the programs it read to key their entries: not kept)
        PC._mem.pop(k, None)
    return {"tokens": min(reach * B, eng.cap), "of_tokens": eng.cap, "buckets": reach,
            "of_buckets": nb, "block": B, "slots": len(slots), "entries_sampled": checked,
            "seconds": round(time.perf_counter() - t0, 2)}


def warm_line(w: dict | None, label: str) -> str | None:
    """The startup's line on warm_reach()."""
    if w is None:
        return None
    past = "" if w["tokens"] >= w["of_tokens"] else \
        "; a prompt past that compiles its programs and computes its wait entries as it goes"
    return (f"{label}: a prompt's runs are on disk through {w['tokens']:,} of "
            f"{w['of_tokens']:,} tokens (bucket {w['buckets']} of {w['of_buckets']}, "
            f"{w['slots']} slot{'s' if w['slots'] > 1 else ''}, sampled in {w['seconds']:g} s){past}")


def warm_samplers(eng) -> list | None:
    """The samplers (G.Sampling.key: top-k's band of 8, with the penalty or not; None: greedy)
    whose generate programs the program cache holds at the context's first and last bucket, for
    the startup's line and /v1/stats. A request with another compiles them and computes their wait
    entries on its first use in this process, for the buckets its max_tokens reaches (docs/serve.md
    "A request's own sampler"). None: no generate programs to look at."""
    from .. import progcache as PC
    from ..llm import generate as G
    if not getattr(eng, "prog_cache", False) or PC.cache_dir() is None \
            or not getattr(eng, "can_generate", False):
        return None
    chain = bool(getattr(eng.backend, "chains", False))

    def there(key, b):
        lo = max((b - 1) * eng.block, eng._conv_lo)
        return PC.has(eng.layout, ("gen", b, lo, chain, key, eng.gen_debug, eng.gen_split))
    nb = math.ceil(eng.cap / eng.block)
    keys = [None, *((k, p) for k in range(8, G.K_MAX + 1, 8) for p in (True, False))]
    return [k for k in keys if there(k, 1) and there(k, nb)]


def sampler_name(key) -> str:
    """A warm_samplers() key as a request asks for it."""
    if key is None:
        return "greedy"
    return f"top_k {key[0] - 7}-{key[0]} {'with' if key[1] else 'without'} repetition_penalty"


def sampler_line(keys: list | None, label: str) -> str | None:
    """The startup's line on warm_samplers()."""
    if keys is None:
        return None
    have = ", ".join(sampler_name(k) for k in keys) or "none"
    return (f"{label}: samplers on disk: {have}; a request with another compiles its programs on "
            f"its first use in this process (its first token waits)")


def startup(eng, label: str) -> dict | None:
    """The startup's lines, and warm_reach()'s result with warm_samplers()' names for /v1/stats
    and the dashboard. A failure here is a line too: it never stops a server."""
    try:
        w = warm_reach(eng)
        s = warm_samplers(eng)
        for line in (saves_line(saves(), label), warm_line(w, label), sampler_line(s, label)):
            if line:
                print(line, flush=True)
        return w if w is None or s is None else dict(w, samplers=[sampler_name(k) for k in s])
    except Exception as e:
        print(f"{label}: could not look at the program cache: {type(e).__name__}: {e}", flush=True)
        return None
