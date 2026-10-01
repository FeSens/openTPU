"""A disk cache of the 4-bit weight quantizer's results, so a model's DRAM image builds in a
minute once its matrices have been quantized (a 4-bit Qwen3.5-4B takes ~20 minutes of numpy on
the card host, and every card tool built it again under the card lock).

`quantize_mxu` is `quant.quantize_mxu` through the cache. The images' builds (qwen3, lfm2,
qwen35) quantize every weight matrix with it, so any build of a model and format fills the
cache for every other: perf.py, refs.py card (per-position or resident) and decode_profile
build different layouts (lookup tables, rows, KV capacity) from the same matrices.
`tools/qual/prebuild.py` builds them ahead, outside otpu-lock. (Gemma 4 keeps its own cache of
quantization jobs, gemma4._cached_job.)

The key is the matrix's content: a hash of its fp32 bytes and shape, the format, D, the
quantizer's parameters (the scale search), the source of quant.py and the numpy version. Any
change to the quantizer or the weights misses; the layout and the configuration do not matter.
int8 is not cached (it quantizes about as fast as the matrix hashes), nor matrices under
MIN_ELEMS elements (the tests' tiny models).

The cache: OTPU_IMAGE_CACHE=<dir>, or 0 for none; by default ~/otpu-build/qcache/mxu on a host
with ~/otpu-build (the build and card hosts), else none. Entries are .npz files written to a
temporary name, then renamed. The least recently used ones go when the cache passes CAP_BYTES
(OTPU_IMAGE_CACHE_GB), and nothing is written that would leave the disk under FREE_FLOOR free.
`stats` counts hits, misses, writes and writes skipped for disk space.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np

from . import quant as Q

FORMATS = ("fp4", "int4")       # cached formats (int8 quantizes as fast as it hashes)
MIN_ELEMS = 1 << 20             # smaller matrices are quantized in line
CAP_BYTES = 30 << 30            # the cache's size (OTPU_IMAGE_CACHE_GB overrides)
FREE_FLOOR = 20 << 30           # free disk a write must leave
stats = {"hit": 0, "miss": 0, "write": 0, "skip": 0}
_SRC: str | None = None


def cache_dir() -> Path | None:
    """The cache directory, or None when the cache is off."""
    v = os.environ.get("OTPU_IMAGE_CACHE")
    if v == "0":
        return None
    if v:
        return Path(v)
    home = Path.home() / "otpu-build"
    return home / "qcache" / "mxu" if home.is_dir() else None


def _source() -> str:
    """What decides the quantizer's output besides its arguments: quant.py and numpy."""
    global _SRC
    if _SRC is None:
        _SRC = hashlib.blake2b(Path(Q.__file__).read_bytes() + np.__version__.encode(),
                               digest_size=16).hexdigest()
    return _SRC


def key(W, fmt: str, D: int) -> str:
    """The cache key of quantize_mxu(W, fmt, D)."""
    a = np.ascontiguousarray(W, np.float32)
    h = hashlib.blake2b(digest_size=20)
    h.update(repr((a.shape, fmt, D, "search", _source())).encode())
    h.update(memoryview(a.reshape(-1)).cast("B"))
    return h.hexdigest()


def quantize_mxu(W, fmt: str, D: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """quant.quantize_mxu(W, fmt, D), from the cache when it holds the result."""
    d = cache_dir()
    if d is None or fmt not in FORMATS or np.size(W) < MIN_ELEMS:
        return Q.quantize_mxu(W, fmt, D)
    W = np.ascontiguousarray(W, np.float32)
    k = key(W, fmt, D)
    f = d / k[:2] / f"{k}.npz"
    try:
        with np.load(f) as z:
            q, s = z["q"], z["s"]
        stats["hit"] += 1
        os.utime(f)                     # recently used (the eviction order)
        return q, s
    except FileNotFoundError:
        pass
    except Exception:                   # a damaged entry: quantize again
        f.unlink(missing_ok=True)
    stats["miss"] += 1
    q, s = Q.quantize_mxu(W, fmt, D)
    _store(d, f, q, s)
    return q, s


def _store(d: Path, f: Path, q: np.ndarray, s: np.ndarray) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(f.parent).free - q.nbytes - s.nbytes < FREE_FLOOR:
        stats["skip"] += 1
        return
    tmp = f.with_name(f".{f.stem}.{os.getpid()}.npz")
    np.savez(tmp, q=q, s=s)
    os.replace(tmp, f)
    stats["write"] += 1
    _evict(d)


def _evict(d: Path) -> None:
    """Drop the least recently used entries while the cache is over its size (to 90% of it)."""
    cap = int(float(os.environ.get("OTPU_IMAGE_CACHE_GB", CAP_BYTES / 2**30)) * 2**30)
    ents = []
    for p in d.glob("*/*.npz"):
        if p.name.startswith("."):
            continue
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        ents.append((st.st_mtime, st.st_size, p))
    total = sum(e[1] for e in ents)
    if total <= cap:
        return
    for _, size, p in sorted(ents):
        if total <= 0.9 * cap:
            break
        p.unlink(missing_ok=True)
        total -= size
