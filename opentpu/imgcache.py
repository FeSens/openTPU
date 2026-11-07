"""A disk cache of whole model images: the DRAM images an Engine builds from a checkpoint, and
what the build leaves on its Image for the host (an int8 embedding table kept on the host,
Gemma 4's PLE records), so that a model opened before opens again in seconds (otpu-chat, the
card tools): no weight is converted, hashed or quantized (LFM2-2.6B fp4 on the card host: 20 s
with every matrix in opentpu/qcache.py, 16 minutes with none).

`build(image, W, layout)` is image.build(W) through the cache. The key is everything that
decides the images' bytes: the checkpoint (W.model_dir: the names, sizes, modification times
and safetensors headers of its config.json and weight files; and W.mtp_from's, the checkpoint
an MTP image's drafter comes from when not the model's), the Image's class, the layout
(the Engine's: Spec, Config, KV capacity, batch, prefill rows, attention block, the image
keywords: formats, lookup tables, expert slots and the image's own choices), the OTPU_*
environment (progcache.keyed_env: some layouts read it, e.g. OTPU_FORMATS), the source of every
module of opentpu but opentpu/host, opentpu/serve and progcache.UNKEYED's tools
(progcache.source: the models, kernels and quantizers), and the numpy and torch versions. A W
that is not a checkpoint's (the tests' dicts) is not cached.

What the build leaves on the Image is its attributes (and the entries of its dict attributes)
that the build added or replaced: arrays are stored with the images, the weights themselves
(Gemma 4 keeps W for host_inputs) are put back on a hit; a build that leaves anything else is
not cached (stats "skip"), rather than handed back without it.

An entry is one file: a JSON header, then each slice's nonzero runs of PAGE bytes (the KV
caches and state regions are zeros, and are not stored) and the arrays. A hit reads the runs
into zeroed images. The cache: OTPU_IMG_CACHE=<dir>, or 0 for none; by default the img
directory beside the 4-bit matrices' (qcache.cache_dir(): ~/otpu-build/qcache/img on a host
with ~/otpu-build; none when OTPU_IMAGE_CACHE=0). Entries are written to a temporary name, then
renamed. Entries go when the cache passes CAP_BYTES (OTPU_IMG_CACHE_GB), in the order of their
last use, where each second of build per GB an entry of this tree took (its header's build_s;
src, progcache.source()) counts as KEEP_S of later use: an image that costs minutes to build
(gemma-4-E4B-it's, 287 s for 7.1 GB) outlives one that costs seconds (Qwen3.5-0.8B's, 16 s for
1.8 GB) opened a day later, while another tree's (a former one's after a host code swap: nothing
opens it here again) goes by its last use alone. A write that
would leave under FREE_FLOOR of free disk (OTPU_IMG_CACHE_FREE_GB) first drops the entries it
outranks in that order when they make the room (a former tree's, after a host code swap), else
it is not written and none is dropped.
`stats` counts hits, misses, writes and skipped writes; `last` is the latest build's (how:
"hit", "miss" (built and stored), "skip" (built, not stored) or "off"; seconds).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import time
from pathlib import Path

import numpy as np

from . import progcache as PC
from . import qcache as QC

CAP_BYTES = 16 << 30            # the cache's size (OTPU_IMG_CACHE_GB overrides)
KEEP_S = 3600.0                 # s of later use each build second per GB buys an entry (eviction)
FREE_FLOOR = 20 << 30           # free disk a write must leave (OTPU_IMG_CACHE_FREE_GB)
PAGE = 4096                     # zeros are left out in whole pages
MAGIC = b"OTPUIMG1"
stats = {"hit": 0, "miss": 0, "write": 0, "skip": 0}
last: tuple = ("off", 0.0)
_LIBS: str | None = None


def cache_dir() -> Path | None:
    """The cache directory, or None when the cache is off."""
    v = os.environ.get("OTPU_IMG_CACHE")
    if v == "0":
        return None
    if v:
        return Path(v)
    q = QC.cache_dir()
    if q is None:
        return None
    return q.parent / "img" if os.environ.get("OTPU_IMAGE_CACHE") is None else q / "img"


def checkpoint(model_dir) -> tuple | None:
    """The identity of a checkpoint directory: (name, size, mtime, header hash) of its
    config.json and *.safetensors files, or None (no directory, no weight file)."""
    if model_dir is None:
        return None
    d = Path(model_dir)
    files = sorted(d.glob("*.safetensors"))
    if not files:
        return None
    out = []
    for p in [d / "config.json"] + files:
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        h = ""
        if p.suffix == ".safetensors":      # the tensors' names, dtypes, shapes and offsets
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = hashlib.blake2b(f.read(min(n, 1 << 26)), digest_size=16).hexdigest()
        out.append((p.name, st.st_size, st.st_mtime_ns, h))
    return tuple(out)


def _libs() -> str:
    global _LIBS
    if _LIBS is None:
        from importlib import metadata
        try:
            torch = metadata.version("torch")
        except metadata.PackageNotFoundError:
            torch = None
        _LIBS = f"numpy {np.__version__}, torch {torch}"
    return _LIBS


def key(image, layout, ck) -> str:
    """The cache key of image.build(W) for this layout and checkpoint identity."""
    cls = f"{type(image).__module__}.{type(image).__qualname__}"
    return hashlib.blake2b(repr((cls, layout, ck, PC.keyed_env(), PC.source(), _libs())).encode(),
                           digest_size=20).hexdigest()


def build(image, W, layout) -> list[np.ndarray]:
    """image.build(W), from the cache when it holds it (and then what the build leaves on the
    image put back), else built and stored."""
    global last
    t0 = time.perf_counter()
    d = cache_dir()
    ck = None if d is None else checkpoint(getattr(W, "model_dir", None))
    if ck is not None and getattr(W, "mtp_from", None) is not None:    # (the MTP layer's)
        ck = (ck, checkpoint(W.mtp_from))
    if ck is None:
        images = image.build(W)
        last = ("off", time.perf_counter() - t0)
        return images
    f = d / (k := key(image, layout, ck))[:2] / f"{k}.img"
    got = _load(f, W)
    if got is not None:
        images, left = got
        for path, v in left.items():
            _set(image, path, v)
        stats["hit"] += 1
        last = ("hit", time.perf_counter() - t0)
        return images
    stats["miss"] += 1
    before = _attrs(image, W)
    tb = time.perf_counter()
    images = image.build(W)
    secs = time.perf_counter() - tb
    left = _left(before, _attrs(image, W), W)
    if left is None:
        stats["skip"] += 1
    stored = left is not None and _store(d, f, images, left, secs,
                                         getattr(image, "spans", None))
    last = ("miss" if stored else "skip", time.perf_counter() - t0)
    return images


# ---- what a build leaves on the image
def _attrs(image, W) -> dict:
    """The image's attributes and its dict attributes' entries ("name" and "name.key") as
    objects (a shallow look: a value replaced shows, an array filled in place does not); not
    the weights' entries (W kept on the image, Gemma 4's _W, comes back as itself on a hit: a
    lazy W's items() reads every tensor as fp32, qwen3.LazyWeights, the streamed MoE models')."""
    out = {}
    for n, v in vars(image).items():
        out[n] = v
        if isinstance(v, dict) and v is not W:
            for k, x in v.items():
                if isinstance(k, str):
                    out[f"{n}.{k}"] = x
    return out


def _left(before: dict, after: dict, W) -> dict | None:
    """What the build added or replaced: {path: ("array", v) | ("weights", None) | ("json", v)};
    None when it left something else (such a build is not cached)."""
    out = {}
    for p, v in after.items():
        if p in before and before[p] is v:
            continue
        if "." not in p and isinstance(v, dict) and p in before:
            continue                        # (a dict replaced: its entries are compared)
        if isinstance(v, np.ndarray):
            out[p] = ("array", v)
        elif v is W:
            out[p] = ("weights", None)
        else:
            try:
                if json.loads(json.dumps(v)) != v:
                    return None
            except (TypeError, ValueError):
                return None
            out[p] = ("json", v)
    return out


def _set(image, path: str, v) -> None:
    n, _, k = path.partition(".")
    if k:
        getattr(image, n)[k] = v
    else:
        setattr(image, n, v)


# ---- entries: MAGIC, the header's length (8 bytes), the header (JSON), then each slice's runs
# and the arrays, one after the other
def _runs(img: np.ndarray, spans=None) -> list[tuple[int, int]]:
    """The image's runs of pages with a nonzero byte: [(offset, bytes)]. spans: the byte
    ranges that hold anything (an image laid out over a wide DRAM's regions, Image.spans,
    opentpu/llm/regions.py: tens of GiB of extent, mostly never touched): only those are
    scanned."""
    if spans is not None:
        out = []
        for a, b in spans:
            a, b = a // PAGE * PAGE, min(-(-b // PAGE) * PAGE, len(img))
            out += [(a + o, k) for o, k in _runs(img[a:b])] if b > a else []
        return out
    n = len(img)
    full = n // PAGE * PAGE
    nz = np.zeros(-(-n // PAGE), bool)
    nz[:full // PAGE] = img[:full].view(np.uint64).reshape(-1, PAGE // 8).any(1)
    if full < n:
        nz[-1] = img[full:].any()
    edge = np.diff(np.concatenate([[0], nz.astype(np.int8), [0]]))
    return [(int(a) * PAGE, min(int(b) * PAGE, n) - int(a) * PAGE)
            for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))]


def _bytes(a) -> np.ndarray:
    return np.ascontiguousarray(a).view(np.uint8).reshape(-1)


def _store(d: Path, f: Path, images: list, left: dict, build_s: float = 0.0,
           spans=None) -> bool:
    """Write the entry (True), or not: under the free-disk floor though the entries it outranks
    were dropped (_room), or a write that failed (the disk full, no permission: the image is
    built, not kept). build_s: the build's seconds (what a miss would cost again: the eviction's
    weight). spans: the images' used ranges (_runs)."""
    images = [_bytes(img) for img in images]
    slices = [{"nbytes": len(img), "runs": _runs(img, spans)} for img in images]
    arrays = {p: {"dtype": v.dtype.str, "shape": list(v.shape)}
              for p, (kind, v) in left.items() if kind == "array"}
    meta = {"build_s": round(build_s, 2), "src": PC.source(), "slices": slices, "arrays": arrays,
            "json": {p: v for p, (kind, v) in left.items() if kind == "json"},
            "weights": [p for p, (kind, _) in left.items() if kind == "weights"]}
    size = sum(n for s in slices for _, n in s["runs"]) + \
        sum(v.nbytes for kind, v in left.values() if kind == "array")
    free = float(os.environ.get("OTPU_IMG_CACHE_FREE_GB", FREE_FLOOR / 2**30)) * 2**30
    head = json.dumps(meta).encode()
    tmp = f.with_name(f".{f.stem}.{os.getpid()}.img")
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        short = free + size - shutil.disk_usage(f.parent).free
        if short > 0 and not _room(d, short, _score(time.time(), build_s, size)):
            stats["skip"] += 1
            return False
        with open(tmp, "wb") as out:
            out.write(MAGIC + struct.pack("<Q", len(head)) + head)
            for img, s in zip(images, slices):
                for o, n in s["runs"]:
                    out.write(memoryview(img[o:o + n]))
            for p in arrays:
                out.write(memoryview(_bytes(left[p][1])))
        os.replace(tmp, f)
    except BaseException as e:
        tmp.unlink(missing_ok=True)
        if not isinstance(e, OSError):
            raise
        stats["skip"] += 1
        return False
    stats["write"] += 1
    _evict(d)
    return True


def _load(f: Path, W):
    """(images, what the build left) of an entry, or None (none, or a damaged one: removed)."""
    try:
        with open(f, "rb") as src:
            if src.read(8) != MAGIC:
                raise ValueError("not an image cache entry")
            meta = json.loads(src.read(struct.unpack("<Q", src.read(8))[0]))
            images = []
            for s in meta["slices"]:
                img = np.zeros(s["nbytes"], np.uint8)
                for o, n in s["runs"]:
                    _read(src, img[o:o + n])
                images.append(img)
            left = {}
            for p, a in meta["arrays"].items():
                left[p] = np.empty(a["shape"], np.dtype(a["dtype"]))
                _read(src, left[p].reshape(-1).view(np.uint8))
            if src.read(1):
                raise ValueError("a long image cache entry")
            left.update(meta["json"])
            left.update({p: W for p in meta["weights"]})
    except FileNotFoundError:
        return None
    except Exception:                   # a damaged entry: build again
        f.unlink(missing_ok=True)
        return None
    try:
        os.utime(f)                     # recently used (the eviction order)
    except OSError:
        pass
    return images, left


def _read(src, out: np.ndarray) -> None:
    mv = memoryview(out).cast("B")
    got = 0
    while got < len(mv):
        n = src.readinto(mv[got:])
        if not n:
            raise EOFError("a short image cache entry")
        got += n


def _build_s(p: Path) -> float:
    """An entry's build seconds (its header's build_s) when it is this tree's (its header's src,
    progcache.source()); 0 for another tree's and for an entry without them."""
    try:
        with open(p, "rb") as f:
            if f.read(8) != MAGIC:
                return 0.0
            head = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    except (OSError, ValueError, struct.error):
        return 0.0
    return float(head.get("build_s", 0.0)) if head.get("src") == PC.source() else 0.0


def _score(used: float, build_s: float, size: int) -> float:
    """An entry's place in the eviction order (the lowest goes first): its last use, each build
    second per GB counting as KEEP_S of later use."""
    return used + KEEP_S * build_s / (max(size, PAGE) / 2**30)


def _ranked(d: Path) -> list:
    """The entries as (score, size, path), in the eviction order."""
    ents = []
    for p in d.glob("*/*.img"):
        if p.name.startswith("."):
            continue
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        ents.append((_score(st.st_mtime, _build_s(p), st.st_size), st.st_size, p))
    return sorted(ents)


def _evict(d: Path) -> None:
    """While the cache is over its size, drop entries (to 90% of it) in the eviction order."""
    cap = int(float(os.environ.get("OTPU_IMG_CACHE_GB", CAP_BYTES / 2**30)) * 2**30)
    ents = _ranked(d)
    total = sum(e[1] for e in ents)
    if total <= cap:
        return
    for _, size, p in ents:
        if total <= 0.9 * cap:
            break
        p.unlink(missing_ok=True)
        total -= size


def _room(d: Path, need: int, score: float) -> bool:
    """Free `need` bytes of disk for an entry of this score (True): drop the entries it outranks,
    in the eviction order, until they make the room; none when all of them would not (False)."""
    drop, freed = [], 0
    for s, size, p in _ranked(d):
        if freed >= need or s >= score:
            break
        drop.append(p)
        freed += size
    if freed < need:
        return False
    for p in drop:
        p.unlink(missing_ok=True)
    return True
