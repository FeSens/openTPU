"""Quantize models into the image cache (opentpu/qcache.py) before a card session, outside
otpu-lock, so that the session's tools (perf.py, refs.py card, decode_profile, otpu-chat) build
their images from the cache in about a minute instead of quantizing under the lock (a 4-bit
Qwen3.5-4B: ~20 minutes on the card host).

    python3 tools/qual/prebuild.py CFG.pkl MODEL:WF:HF ...      (HF '-' = the LM head in WF)

CFG.pkl is the card's configuration (tools/qual/refs.py cfg; any qualified deploy's
qual/cfg.pkl). Each run builds the image the way the card tools do, through Engine with that
configuration and those formats (resident: the superset with the lookup tables), handing it to
a backend that drops it, so every 4-bit matrix they quantize is in the cache. Run it at
`nice -n 19`, on the card host or anywhere (the cache's keys hold no host: copy the cache
directory). Prints each run's hits, misses and seconds.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opentpu import qcache  # noqa: E402
from opentpu.llm import load_spec, model_dir  # noqa: E402
from opentpu.llm.qwen3 import Engine, load_weights  # noqa: E402

CAP = 256               # the KV capacity does not change the matrices (refs.py's)


class _Drop:
    """A backend that takes the image and keeps nothing."""

    def __init__(self, cfg, images):
        pass


def prebuild(cfg, spec, W, wformat: str, head_format: str | None, cap: int = CAP) -> dict:
    """Build (and drop) the image an Engine with this configuration and these formats builds;
    the cache's counts over it."""
    s0 = dict(qcache.stats)
    Engine(spec, W, cap=cap, cfg=cfg, backend=_Drop, wformat=wformat, head_format=head_format,
           resident=True, pipeline=False)
    return {k: qcache.stats[k] - s0[k] for k in s0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", help="the card's configuration (tools/qual/refs.py cfg)")
    ap.add_argument("runs", nargs="+", help="MODEL:WF:HF")
    a = ap.parse_args()
    cfg = pickle.loads(Path(a.cfg).read_bytes())
    d = qcache.cache_dir()
    if d is None:
        raise SystemExit("the image cache is off (OTPU_IMAGE_CACHE=0, or no ~/otpu-build)")
    print(f"cache {d}")
    for r in a.runs:
        m, wf, hf = r.split(":")
        path = model_dir(m)
        t0 = time.time()
        n = prebuild(cfg, load_spec(path), load_weights(path), wf, None if hf == "-" else hf)
        skip = f", {n['skip']} not written (disk space)" if n["skip"] else ""
        print(f"{r}: {n['hit']} cached, {n['miss']} quantized, {n['write']} written{skip} "
              f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
