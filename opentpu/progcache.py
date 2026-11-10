"""Compiled programs, kept once per process and on disk, so that an Engine (a chat's, a tool's) does
not compile again what an earlier one, in this process or another, compiled for the same image layout:
the generate loop's bucket programs, the resident decode's, MTP's loop programs.

A program is a function of the image's layout and of what is compiled. The layout is the Engine's
Spec, Config, KV capacity, batch, rows, attention block and image keywords (formats, lookup tables,
expert slots, the image's own choices: Engine._image_kw). What is compiled is a tuple naming the
program and its arguments (("gen", blocks, ...)). The key hashes both, the source of the compiler
and the kernels (every module of opentpu but opentpu/host, opentpu/serve and the tools that
decide no program word or image byte (UNKEYED: the profilers, the traces, the reference math, the
RTL simulation), and the host modules whose constants the programs take: HOST_KEYED) and the
OTPU_* environment (some kernels read it, e.g. OTPU_FORMATS, OTPU_MLP_UNROLL_BODIES): any change
misses.

`get(layout, what, compile)` returns the cached value or compile()'s, which it keeps. A value is
(programs, run_args): programs one per slice (a list of Instr, or its assembled words, as the
compile worker processes give them), or a tuple of such lists (a split generate program's parts);
run_args the programs' arguments (RunVar, coefficient) or None. On disk each program is its
assembled words; a program read back is the words decoded, one compiler.Program per slice
whatever the form it was compiled in (Instr.decode: no comments or kernel source frames, which
only the profilers read; they compile their own programs), with run_args (each RunVar's name and
bound) as its own.

An Engine uses the cache with prog_cache (default: enabled(), OTPU_PROG_CACHE set and not 0).
Tests leave it off: they patch kernels' module constants, which the key does not see. The disk
cache: OTPU_PROG_CACHE=<dir>; with 1 (or an engine's prog_cache=True and the variable unset)
~/otpu-build/qcache/prog on a host with ~/otpu-build (the build and card hosts), else none. Entries are .npz files written to a
temporary name, then renamed; the least recently used go past CAP_BYTES (OTPU_PROG_CACHE_GB).
`stats` counts memory hits, disk hits, compiles, disk writes and the writes skipped under the
free-disk floor (free_floor).

`fact(layout, what, compute)` keeps a small JSON value the same way (.json beside the
programs), e.g. the rows a prompt run of a bucket takes (prefill.r_max): what a compile found
out, including that a larger one does not fit, which no program records.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from . import isa as I
from .compiler import Program, run_var

CAP_BYTES = 2 << 30             # the disk cache's size (OTPU_PROG_CACHE_GB overrides)
FREE_FLOOR = 4 << 30            # free disk a write must leave (OTPU_PROG_CACHE_FREE_GB overrides)
# read by the host's tools only: not in the key (otpu-lock sets OTPU_LOCK_HELD and its pid,
# OTPU_LOCK_PID, for every command it runs: in the key, no session found the one before's)
RUNTIME_ENV = ("OTPU_PROG_CACHE", "OTPU_PROG_CACHE_GB", "OTPU_IMAGE_CACHE", "OTPU_IMAGE_CACHE_GB",
               "OTPU_IMG_CACHE", "OTPU_IMG_CACHE_GB", "OTPU_IMG_CACHE_FREE_GB", "OTPU_BUILD_JOBS",
               "OTPU_PROG_CACHE_FREE_GB",
               "OTPU_QUIET", "OTPU_PREBUILD_CFG", "OTPU_PREBUILD_MIN_GB",
               "OTPU_PREBUILD_QUIET_WAIT", "OTPU_LOCK_HELD", "OTPU_LOCK_PID", "OTPU_LOCK_WAIT",
               "OTPU_RUN_DIR", "OTPU_STOP_RUN", "OTPU_SIM_TMP", "OTPU_PARENT",
               "OTPU_VISION_WARM",
               # which device node a run opens (board.default_dev): a card's programs and images
               # are its bitstream's, whichever node it is
               "OTPU_DEV")
# the host modules the compiler, kernels and models import: opentpu/host/offload.py's LINE, TAG,
# RUN, Layout and RowLayout place the mailboxes, slots and tags the programs address
HOST_KEYED = ("host/offload.py",)
# the modules no image build or compile runs (a profile of opentpu.lens, a hardware trace, the
# reference math, the RTL simulation a run takes): an edit to one keeps every cached program and
# image (tests/test_progcache.py: no other module imports one, but runtime's RTL run)
UNKEYED = ("hwtrace.py", "lens.py", "profile.py", "reference.py", "rtlsim.py", "simtmp.py",
           "llm/rtl_backend.py")
stats = {"memory": 0, "disk": 0, "compile": 0, "write": 0, "skip": 0}
_mem: dict = {}
_SRC: str | None = None


def enabled() -> bool:
    """An Engine's default prog_cache."""
    return os.environ.get("OTPU_PROG_CACHE", "0") != "0"


def cache_dir() -> Path | None:
    """The disk cache's directory, or None when it is off."""
    v = os.environ.get("OTPU_PROG_CACHE")
    if v == "0":
        return None
    if v and v != "1":
        return Path(v)
    home = Path.home() / "otpu-build"
    return home / "qcache" / "prog" if home.is_dir() else None


def keyed() -> list[Path]:
    """The modules source() hashes: every module of opentpu but opentpu/host, opentpu/serve (the
    card's drivers, the API server) and UNKEYED (the compiler, the kernels, the models' layouts
    and kernels), and HOST_KEYED."""
    root = Path(__file__).resolve().parent
    rel = {f: f.relative_to(root).as_posix() for f in sorted(root.rglob("*.py"))}
    return [f for f, r in rel.items()
            if (not r.startswith(("host/", "serve/")) and r not in UNKEYED) or r in HOST_KEYED]


def source() -> str:
    """A hash of the keyed() modules."""
    global _SRC
    if _SRC is None:
        root = Path(__file__).resolve().parent
        h = hashlib.blake2b(digest_size=16)
        for f in keyed():
            h.update(f.relative_to(root).as_posix().encode() + b"\0" + f.read_bytes())
        _SRC = h.hexdigest()
    return _SRC


def keyed_env() -> list:
    """The OTPU_* environment a key holds: every variable but RUNTIME_ENV's."""
    return sorted((k, v) for k, v in os.environ.items()
                  if k.startswith("OTPU_") and k not in RUNTIME_ENV)


def free_floor() -> int:
    """The free disk a write must leave, bytes: OTPU_PROG_CACHE_FREE_GB, else FREE_FLOOR. A write
    that would leave less is skipped (stats "skip"): the program is compiled again by the next
    process that wants it."""
    v = os.environ.get("OTPU_PROG_CACHE_FREE_GB")
    return FREE_FLOOR if v is None else int(float(v) * 2**30)


def key(layout, what) -> str:
    return hashlib.blake2b(repr((layout, what, keyed_env(), source())).encode(),
                           digest_size=20).hexdigest()


def get(layout, what, compile, disk: bool = True):
    """The value of `what` for this layout: kept from an earlier call, read from the disk
    cache, or compile() (then kept, and written to the disk cache with disk)."""
    k = key(layout, what)
    if k in _mem:
        stats["memory"] += 1
        return _mem[k]
    d = cache_dir() if disk else None
    f = None if d is None else d / k[:2] / f"{k}.npz"
    v = None if f is None else _load(f)
    if v is not None:
        stats["disk"] += 1
    else:
        stats["compile"] += 1
        v = compile()
        if f is not None and v[0] is not None:
            _store(d, f, v)
    _mem[k] = v
    return v


def fact(layout, what, compute, disk: bool = True):
    """A small JSON value of `what` for this layout, as get() keeps programs: from an earlier
    call, the disk cache, or compute() (then kept, and written with disk)."""
    k = key(layout, what)
    if k in _mem:
        stats["memory"] += 1
        return _mem[k]
    d = cache_dir() if disk else None
    f = None if d is None else d / k[:2] / f"{k}.json"
    try:
        v = json.loads(f.read_text())
        stats["disk"] += 1
    except (AttributeError, OSError, ValueError):   # (no disk cache, no entry, a damaged one)
        stats["compile"] += 1
        v = compute()
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(f.parent).free < free_floor():
                stats["skip"] += 1
            else:
                tmp = f.with_name(f".{f.stem}.{os.getpid()}.json")
                tmp.write_text(json.dumps(v))
                os.replace(tmp, f)
                stats["write"] += 1
    _mem[k] = v
    return v


def has(layout, what) -> bool:
    """get() would not compile: the value is kept here or on disk."""
    k = key(layout, what)
    d = cache_dir()
    return k in _mem or (d is not None and (d / k[:2] / f"{k}.npz").is_file())


def clear() -> None:
    """Forget the programs kept in this process (the disk cache stays)."""
    _mem.clear()


def _parts(progs):
    return (list(progs), True) if isinstance(progs, tuple) else ([progs], False)


def _seen(parts) -> list:
    """The (RunVar, c) the programs' addresses add (Program.run_seen), every slice's, once."""
    out = []
    for part in parts:
        for p in ([part] if isinstance(part, np.ndarray) else part):
            out += [x for x in getattr(p, "run_seen", ()) if x not in out]
    return out


def _store(d: Path, f: Path, v) -> None:
    progs, ra = v
    parts, split = _parts(progs)
    arrs, kinds = {}, []
    for j, part in enumerate(parts):
        if isinstance(part, np.ndarray):            # one slice's words (a compile worker's)
            part = [part]
            kinds.append("w")
        else:
            kinds.append("i")
        for s, p in enumerate(part):
            arrs[f"p{j}_{s}"] = np.asarray(p if isinstance(p, np.ndarray) else I.assemble(p),
                                           np.uint32)
    meta = {"split": split, "kinds": kinds, "S": [len(p) if k == "i" else 1
                                                  for p, k in zip(parts, kinds)],
            "ra": None if ra is None else [[v_.name, int(c), v_.bound] for v_, c in ra],
            "seen": [[v_.name, int(c), v_.bound] for v_, c in _seen(parts)]}
    f.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(f.parent).free - sum(a.nbytes for a in arrs.values()) < free_floor():
        stats["skip"] += 1
        return
    tmp = f.with_name(f".{f.stem}.{os.getpid()}.npz")
    np.savez(tmp, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8), **arrs)
    os.replace(tmp, f)
    stats["write"] += 1
    _evict(d)


def _load(f: Path):
    try:
        with np.load(f) as z:
            meta = json.loads(bytes(z["meta"]).decode())
            parts = []
            for j, (kind, S) in enumerate(zip(meta["kinds"], meta["S"])):
                ws = [z[f"p{j}_{s}"] for s in range(S)]
                parts.append([[I.Instr.decode(w[i:i + 8]) for i in range(0, len(w), 8)]
                              for w in ws])
            ra = None if meta["ra"] is None else [(run_var(r[0], *r[2:]), r[1])
                                                  for r in meta["ra"]]
            seen = [(run_var(r[0], *r[2:]), r[1]) for r in meta.get("seen", ())]
        os.utime(f)                     # recently used (the eviction order)
    except FileNotFoundError:
        return None
    except Exception:                   # a damaged entry: compile again
        f.unlink(missing_ok=True)
        return None
    # each slice's program with its arguments and their bounds (opentpu.waits bounds them), and
    # the run-time values its addresses add (Program.run_seen: the run words it reads)
    parts = [[Program(p, ra or (), seen) for p in part] for part in parts]
    progs = tuple(parts) if meta["split"] else parts[0]
    return progs, ra


def _evict(d: Path) -> None:
    """Drop the least recently used entries while the cache is over its size (to 90% of it)."""
    cap = int(float(os.environ.get("OTPU_PROG_CACHE_GB", CAP_BYTES / 2**30)) * 2**30)
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
