"""Quantize models into the image cache (opentpu/qcache.py) before a card session, outside the
card lock, so that the session's tools (tools/qual/perf.py, refs.py card, decode_profile.py,
otpu-chat) build their images from the cache in about a minute instead of quantizing under the
lock (a 4-bit Qwen3.5-4B: ~20 minutes on the card host).

    python3 -m opentpu.host.prebuild [--cfg CFG.pkl] MODEL:WF:HF ...   (HF '-': the head in WF)
    otpu-lock --prebuild MODEL:WF:HF ... -- CMD...      the same at nice 19, then the lock, CMD

tools/qual/qual.sh does it for its 4-bit runs when it is started without the lock. Each run
builds the image the way the card tools do, through Engine with the card's configuration and
those formats (resident: the superset with the lookup tables), and hands it to a backend that
drops it, so every 4-bit matrix they quantize is in the cache. Runs with no 4-bit format are
skipped: int8 is not cached.

--cfg is the card's configuration (tools/qual/refs.py cfg). Default: OTPU_PREBUILD_CFG, else
the production deploy's (~/otpu-build/production/qual/cfg.pkl), else the newest card
configuration that refs.py keeps (~/otpu-build/refcache/configs). The cache's keys are the
matrices' content, so a configuration whose matrices differ (another D or slice count) only
misses. Builds start only while MemAvailable is at least MIN_GB (OTPU_PREBUILD_MIN_GB; a 4-bit
4B builds in up to 17 GB); with less, prebuild says so and leaves the quantizing to the tools.
The models load from the tree on PYTHONPATH, as the session's tools do. Prints each run's
hits, misses and seconds.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

from .. import qcache

CAP = 256               # the KV capacity does not change the matrices (refs.py's)
MIN_GB = 20.0
FORMATS4 = ("fp4", "int4")


class _Drop:
    """A backend that takes the image and keeps nothing."""

    def __init__(self, cfg, images):
        pass


def prebuild(cfg, spec, W, wformat: str, head_format: str | None, cap: int = CAP) -> dict:
    """Build (and drop) the image an Engine with this configuration and these formats builds;
    the cache's counts over it."""
    from ..llm.qwen3 import Engine
    s0 = dict(qcache.stats)
    Engine(spec, W, cap=cap, cfg=cfg, backend=_Drop, wformat=wformat, head_format=head_format,
           resident=True, pipeline=False)
    return {k: qcache.stats[k] - s0[k] for k in s0}


def default_cfg() -> Path | None:
    """The card configuration prebuild uses without --cfg."""
    if os.environ.get("OTPU_PREBUILD_CFG"):
        return Path(os.environ["OTPU_PREBUILD_CFG"])
    home = Path.home() / "otpu-build"
    prod = home / "production/qual/cfg.pkl"
    if prod.is_file():
        return prod
    kept = sorted((home / "refcache/configs").glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    return kept[-1] if kept else None


def is_4bit(run: str) -> bool:
    return any(f in FORMATS4 for f in run.split(":")[1:])


def mem_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="prebuild", description=__doc__.split("\n")[0])
    ap.add_argument("--cfg", help="the card's configuration (tools/qual/refs.py cfg)")
    ap.add_argument("runs", nargs="+", help="MODEL:WF:HF")
    a = ap.parse_args(argv)
    d = qcache.cache_dir()
    if d is None:
        print("prebuild: the image cache is off (OTPU_IMAGE_CACHE=0, or no ~/otpu-build)")
        return 0
    path = Path(a.cfg) if a.cfg else default_cfg()
    if path is None or not path.is_file():
        print(f"prebuild: no card configuration ({path or 'none found'}); give --cfg")
        return 1
    cfg = pickle.loads(path.read_bytes())
    runs = [r for r in a.runs if is_4bit(r)]
    print(f"prebuild: cache {d}, configuration {path}; {' '.join(runs) or 'no 4-bit runs'}",
          flush=True)
    from ..llm import load_spec, model_dir
    from ..llm.qwen3 import load_weights
    need = float(os.environ.get("OTPU_PREBUILD_MIN_GB", MIN_GB))
    for r in runs:
        m, wf, hf = r.split(":")
        free = mem_gb()
        if free is not None and free < need:
            print(f"{r}: not built, MemAvailable {free:.1f} GiB < {need:g} (the tools quantize "
                  "it)", flush=True)
            continue
        t0 = time.time()
        p = model_dir(m)
        n = prebuild(cfg, load_spec(p), load_weights(p), wf, None if hf == "-" else hf)
        skip = f", {n['skip']} not written (disk space)" if n["skip"] else ""
        print(f"{r}: {n['hit']} cached, {n['miss']} quantized, {n['write']} written{skip} "
              f"({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
