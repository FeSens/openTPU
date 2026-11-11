"""MiniCPM-V's tower programs' wait entries computed ahead of its first images (docs/minicpmv.md 6).

On a card whose sequencer takes wait entries (WAITC: BoardBackend.waitc), a program's first start
computes them (opentpu/waits.py entries: an abstract execution of every instruction its run
takes) and keeps them on the program cache's disk (per tree: the keyed sources). The tower's
programs are per patch count (minicpmv.VisionImage.programs(N)): on the card host a crop of a
count not seen before took ~23 s more than its 5.8 s on the card (session mcpv2: the 13
programs' entries, the attention's 14 s). The processor gives a handful of counts (counts).

start(eng) computes them ahead, after the engine's load: one spawned process at nice 19 builds
the engine's image layout (as its compile workers do: the programs address the same image),
and for each count, the most frequent first, traces the tower's programs and computes their
entries into the disk cache, where the engine's first start of each finds them. A count done
before is skipped: its marker in the program cache (keyed by the engine's layout, the sources
and the OTPU_* environment, PC.key) lists its entries' keys, each still on disk. The process is
a daemon: it ends with the engine's process, a write cut short leaving nothing (waits.entries
writes to a temporary name, then renames). OTPU_VISION_WARM=0: off.
"""
from __future__ import annotations

import json
import os
from collections import Counter

ENV = "OTPU_VISION_WARM"


def counts(vs, lo: int = 200, hi: int = 4000, step: int = 25, nmax: int | None = None) -> list:
    """The crops' patch counts (h x w) the processor gives (minicpmv.crops: the thumbnail and
    the slices), the most frequent first: over images of lo..hi pixels a side (by step) at
    every max_slices from 1 to vs.max_slices; at most nmax (the image's work area)."""
    from opentpu.llm import minicpmv as MV
    n = Counter()
    for H in range(lo, hi + 1, step):
        for W in range(lo, hi + 1, step):
            for ms in range(1, vs.max_slices + 1):
                grid = MV.slice_grid(H, W, ms, vs.scale)
                th, tw = MV.best_resize(H, W, vs.scale, vs.patch, upscale=grid is None)
                n[(th // vs.patch) * (tw // vs.patch)] += 1
                if grid is not None:
                    rh, rw = MV.refine_size(H, W, grid, vs.scale, vs.patch)
                    n[(rh // grid[0] // vs.patch) * (rw // grid[1] // vs.patch)] += 1
    return [N for N, _ in n.most_common() if nmax is None or N <= nmax]


def start(eng):
    """The background process computing the tower's entries for eng (its multiprocessing
    Process), or None: no vision side, a backend that takes no entries, no disk cache, or
    OTPU_VISION_WARM=0."""
    from opentpu import progcache as PC
    if getattr(eng.image, "vision", None) is None or os.environ.get(ENV) == "0" \
            or not getattr(eng.backend, "waitc", False) or PC.cache_dir() is None:
        return None
    import multiprocessing as mp
    p = mp.get_context("spawn").Process(
        target=_main, name="otpu-vision-warm", daemon=True,
        args=(eng.spec, eng.cfg, eng.cap, eng.batch, eng.rows, eng._image_kw, eng.layout))
    p.start()
    return p


def _main(spec, cfg, cap, batch, rows, image_kw: dict, layout) -> None:
    os.nice(19)
    image = spec.image(cfg, cap, batch, rows, **image_kw)
    for N in counts(spec.vision, nmax=image.vision.nmax):
        warm(image, layout, N)


def warm(image, layout, N: int) -> bool:
    """The entries of the tower's programs at N patches into the disk cache, unless its marker
    finds them there; True when it computed them."""
    from opentpu import progcache as PC
    from opentpu import waits as WC
    from opentpu.compiler import Words
    from opentpu.isa import assemble
    cfg, d = image.cfg, PC.cache_dir()
    k = PC.key(layout, ("vision-waits", N))
    f = d / k[:2] / f"{k}.json"
    try:
        keys = json.loads(f.read_text())
        if all((d / e[:2] / f"{e}.npz").is_file() for e in keys):
            return False
    except (OSError, ValueError):
        pass
    P, _ = image.vision.programs(N)
    keys = []
    for p, ra in P.values():
        w = Words.of(assemble(p), ra)
        WC.entries(w, cfg.D, cfg.S, ra, prog=p, aw=cfg.AW, win=cfg.WC_WIN)
        keys.append(WC.entries_key(w, cfg.D, cfg.S, ra, cfg.AW, cfg.WC_WIN))
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".{f.stem}.{os.getpid()}.json")
    tmp.write_text(json.dumps(keys))
    os.replace(tmp, f)
    return True
