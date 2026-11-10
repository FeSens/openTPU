"""The decode loop's bucket programs compiled ahead of a chat's or server's first replies.

The decode loop on the card (Engine.generate_card) runs one program per attention bucket
(generate.compile_bucket: the loop over the bucket's positions). The engine compiles a bucket's
programs when a reply first reaches it, and the reply waits: on the card host ~3.6 s a bucket for
Qwen3.5-35B-A3B's (MiniCPM-V's), the compile and the wait entries (session hostr21, 2026-10-14: an
image prompt of 488 tokens put its 32-token reply in buckets 2 and 3, never used before: 7.2 s of
the reply's 14.2 s; docs/minicpmv.md 6). An engine with the program cache keeps them on disk per
tree (the keyed sources), so each tree's first reply in each bucket waited.

start(eng, sampling) compiles them ahead, after the engine's open: one spawned process at
nice 19 builds the engine's image layout (as its compile workers do: the programs address the
same image) and, from bucket 1 to the KV capacity's last, compiles each bucket's programs, greedy
and for `sampling` (the replies' default sampler: its programs are compiled for its top-k's
candidate buffers and its penalty, generate.Sampling.key), into the program cache under the
engine's layout and the `what` its _generate_prog asks for (gen_what); on a card whose sequencer
takes wait entries (WAITC) it computes their entries as BoardBackend.prepare does for programs
read back from the cache (waits.entries, kept on the same disk). The engine's first use of a
bucket finds both there. The engine never waits for this process: a bucket not on disk yet is
compiled in the engine as before (both write the same entry). The process is a daemon, so it
ends with the engine's process; a write cut short leaves nothing (the caches write to a
temporary name, then rename). Compiles are host work, not model math: the card runs the
programs.
"""
from __future__ import annotations

import os


def gen_what(block: int, conv_lo: int, blocks: int, chain: bool, samp, debug: bool,
             split) -> tuple:
    """The program cache's `what` of a bucket's generate programs (Engine._generate_prog's)."""
    lo = max((blocks - 1) * block, conv_lo)
    return ("gen", blocks, lo, chain, None if samp is None else samp.key, debug, split)


def samplings(sampling: dict | None) -> list:
    """The samplers a chat's or server's decode loop compiles for: greedy (None) and, when
    `sampling` (temperature, top_k, top_p, repetition_penalty) is not plain greedy (as
    chat.sampler's pick.greedy: temperature 0 and no penalty) and the card takes it, its
    generate.Sampling."""
    from opentpu.llm import generate as G
    out = [None]
    if sampling and not (sampling.get("temperature", 0) <= 0
                         and sampling.get("repetition_penalty", 1.0) == 1.0):
        kw = {k: sampling[k] for k in ("temperature", "top_k", "top_p", "repetition_penalty")
              if k in sampling}
        if G.Sampling.fits(**{"top_p": 1.0, "repetition_penalty": 1.0, **kw}):
            out.append(G.Sampling(**kw))
    return out


def start(eng, sampling: dict | None = None):
    """The background process compiling eng's generate programs (its multiprocessing Process),
    or None: an engine without the device's decode loop (can_generate), without the program
    cache (prog_cache), of more than one sequence, or no disk cache."""
    from opentpu import progcache as PC
    if not getattr(eng, "can_generate", False) or not getattr(eng, "prog_cache", False) \
            or getattr(eng, "batch", 1) != 1 or PC.cache_dir() is None:
        return None
    import multiprocessing as mp
    be = eng.backend
    p = mp.get_context("spawn").Process(
        target=_main, name="otpu-gen-warm", daemon=True,
        args=(eng.spec, eng.cfg, eng.cap, eng.batch, eng.rows, eng._image_kw,
              eng.image.stream_fill, eng.layout, eng.block, eng._conv_lo,
              bool(getattr(be, "chains", False)), eng.gen_debug, eng.gen_split,
              bool(getattr(be, "waitc", False)), samplings(sampling)))
    p.start()
    return p


def _main(spec, cfg, cap, batch, rows, image_kw: dict, stream_fill: bool, layout, block: int,
          conv_lo: int, chain: bool, debug: bool, split, waitc: bool, samps) -> None:
    from opentpu.llm import generate as G
    os.nice(19)
    image = spec.image(cfg, cap, batch, rows, **image_kw)
    image.stream_fill = stream_fill
    for b in range(1, G.buckets(cap, block) + 1):
        for samp in samps:
            warm(image, layout, block, conv_lo, b, chain, samp, debug, split, waitc)


def warm(image, layout, block: int, conv_lo: int, blocks: int, chain: bool = True, samp=None,
         debug: bool = False, split=None, waitc: bool = False) -> bool:
    """A bucket's generate programs into the program cache's disk, and with waitc their wait
    entries, unless they are there; True when it compiled the programs."""
    from opentpu import isa as I
    from opentpu import progcache as PC
    from opentpu import waits as WC
    from opentpu.llm import generate as G
    w = gen_what(block, conv_lo, blocks, chain, samp, debug, split)
    compiled = not PC.has(layout, w)
    if compiled:
        PC.get(layout, w, lambda: (G.compile_bucket(image, blocks, w[2], block, chain=chain,
                                                    samp=samp, debug=debug, split=split), None))
    if waitc:                       # the programs as the engine reads them back (decoded)
        k = PC.key(layout, w)
        v = PC._load(PC.cache_dir() / k[:2] / f"{k}.npz")
        if v is not None:
            for part in v[0] if isinstance(v[0], tuple) else (v[0],):
                for p in part:
                    WC.entries(I.assemble(p), image.cfg.D, image.cfg.S, prog=p,
                               aw=image.cfg.AW, win=image.cfg.WC_WIN)
    return compiled
