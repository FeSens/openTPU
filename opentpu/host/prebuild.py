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

A session that measures host-sensitive performance (wall tok/s, host-bound MoE streaming)
creates the quiet file (~/otpu-build/QUIET, OTPU_QUIET) for its duration, with its pid in it.
Prebuild waits while it exists (up to OTPU_PREBUILD_QUIET_WAIT seconds in all, default an hour;
then it stops, and the tools quantize under the lock), and a run being built when it appears is
stopped at once and built again after it: each run builds in its own process (`--one`). A
quiet file whose pid is gone (a session killed before its trap ran) is ignored.
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

from .. import qcache
from .runstate import pid_alive

CAP = 256               # the KV capacity does not change the matrices (refs.py's)
MIN_GB = 20.0
QUIET_WAIT = 3600.0     # s: the longest a prebuild waits for the quiet file to go
POLL = 10.0             # s between looks at it
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


def sweep_temps(d: Path, age: float = 3600.0) -> int:
    """Remove the cache's temporary files (qcache._store's `.<key>.<pid>.npz`) older than
    `age` seconds: a build killed while writing an entry leaves one, and the cache's size cap
    does not count it. The number removed."""
    n, old = 0, time.time() - age
    for p in d.glob("*/.*.npz"):
        try:
            if p.stat().st_mtime < old:
                p.unlink()
                n += 1
        except FileNotFoundError:
            pass
    return n


def quiet_file() -> Path:
    """While this file exists a session measures host-sensitive performance, and no prebuild
    may load the host (OTPU_QUIET; default ~/otpu-build/QUIET)."""
    return Path(os.environ.get("OTPU_QUIET", Path.home() / "otpu-build/QUIET"))


def quiet() -> bool:
    """The quiet file is there, and the session that made it (the pid in it, when it holds
    one) is alive: a session killed before its trap removed the file does not hold prebuilds
    back."""
    try:
        text = quiet_file().read_text().strip()
    except OSError:
        return False
    return not text.isdigit() or pid_alive(int(text))


def _spawn(cmd: list):
    return subprocess.Popen(cmd)


def _watch(child) -> bool:
    """Wait for a run's build; stop it (False) when the quiet file appears."""
    while child.poll() is None:
        if quiet():
            child.terminate()
            child.wait()
            return False
        time.sleep(POLL)
    return True


def _build_one(cfg, run: str) -> int:
    from ..llm import load_spec, model_dir
    from ..llm.qwen3 import load_weights
    m, wf, hf = run.split(":")
    t0 = time.time()
    p = model_dir(m)
    n = prebuild(cfg, load_spec(p), load_weights(p), wf, None if hf == "-" else hf)
    skip = f", {n['skip']} not written (disk space)" if n["skip"] else ""
    print(f"{run}: {n['hit']} cached, {n['miss']} quantized, {n['write']} written{skip} "
          f"({time.time() - t0:.0f} s)", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="prebuild", description=__doc__.split("\n")[0])
    ap.add_argument("--cfg", help="the card's configuration (tools/qual/refs.py cfg)")
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)  # a run's worker
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
    if a.one:
        return _build_one(pickle.loads(path.read_bytes()), a.runs[0])
    if d.is_dir() and (n := sweep_temps(d)):
        print(f"prebuild: removed {n} temporary files of killed builds", flush=True)
    runs = [r for r in a.runs if is_4bit(r)]
    print(f"prebuild: cache {d}, configuration {path}; {' '.join(runs) or 'no 4-bit runs'}",
          flush=True)
    need = float(os.environ.get("OTPU_PREBUILD_MIN_GB", MIN_GB))
    budget = float(os.environ.get("OTPU_PREBUILD_QUIET_WAIT", QUIET_WAIT))
    waited, told = 0.0, False
    for r in runs:
        while True:
            while quiet():                              # a session measures: not now
                if waited >= budget:
                    print(f"prebuild: {quiet_file()} stayed for {waited:.0f} s; the tools "
                          "quantize the rest under the lock", flush=True)
                    return 0
                if not told:
                    print(f"prebuild: waiting while {quiet_file()} exists (a session "
                          "measures)", flush=True)
                    told = True
                time.sleep(POLL)
                waited += POLL
            told = False
            free = mem_gb()
            if free is not None and free < need:
                print(f"{r}: not built, MemAvailable {free:.1f} GiB < {need:g} (the tools "
                      "quantize it)", flush=True)
                break
            # each run in its own process: stopped at once (its memory freed) when a session
            # goes quiet, and built again after it, from what the cache holds by then
            child = _spawn([sys.executable, "-m", "opentpu.host.prebuild", "--one", "--cfg",
                            str(path), r])
            if _watch(child):
                if child.returncode:
                    print(f"{r}: the build failed (exit {child.returncode}); the tools "
                          "quantize it", flush=True)
                break
            print(f"{r}: stopped, {quiet_file()} appeared; again after it", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
