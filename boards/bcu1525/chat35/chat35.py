"""chat35.py: otpu-chat (opentpu.host.chat) with Qwen3.6-35B-A3B resident on one BCU1525 (bcu1525-wg:bhw16), its card
image from the image cache. The chat builds its Engine the way the entry was built: the target's configuration (the
pickle), the model's whole context (CAP, the chain area a ring: generate.RING), the chat's formats for the family
(streamed.FORMATS: fp4 experts, nvfp4 attention, DeltaNet and shared MLP, int8 head), resident, no prompt runs, the
35B whole on the BCU (not the streamed-experts path of a 4 GiB card), its prompt through the decode step a token a
run. The image is built on a host with the RAM for it (CHAT35_BUILD=1: the same Engine on a backend that touches no
hardware, then exit; c35.sh) and copied into the card host's cache, which never builds it (ours has 31 GB of RAM, and
it takes hours): a cache miss is refused.
CHAT35_KEY=1 prints the key this tree and environment give, builds nothing. Each chat keeps gen_warm's arguments in
warm35.pkl beside this file (warm35.py: every bucket's programs into the program cache between chats). Run from the
tree's root with PYTHONPATH=. and the entry's environment (otpu-chat35 sets both); arguments go to the chat.
install() is the setup (otpu-chat's patches) and open_engine() the chat's engine without its chat: otpu-serve35
(serve35.py) serves the same engine."""
import json
import mmap
import os
import pickle
import struct
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
CAP = 262144                    # Qwen3.6-35B-A3B's max_position_embeddings
# (the card: OTPU_DEV, otpu-chat35's, the device of BCU_CARD's card; an image's key does not hold it)
FIXED = ["--backend", "board", "--dev", os.environ.get("OTPU_DEV") or "/dev/xdma0", "--model", "qwen36-35b",
         "--cap", str(CAP),
         "--wformat", "fp4", "--head-format", "int8", "--no-prompt-runs"]


class Null:
    """No hardware: the Engine's backend while the image is built or keyed."""

    def __init__(self, c, images):
        self.cfg = c

    def close(self):
        pass


def _timed(cls, *names):
    """Each method's calls timed to stderr (CHAT35_TIMES): its name, start and seconds."""
    t0 = time.perf_counter()
    for name in names:
        f = getattr(cls, name)

        def wrap(*a, _f=f, _n=name, **kw):
            t = time.perf_counter()
            try:
                return _f(*a, **kw)
            finally:
                d = time.perf_counter() - t
                if d > 0.005 or _n not in ("step", "start", "wait", "prepare", "_layer_run"):
                    print(f"[chat35] {t - t0:9.3f} {_n} {d:.3f} s", file=sys.stderr)
        setattr(cls, name, wrap)


class _LazyWeights:
    """load_weights(path) at the first read of a tensor: an engine whose image comes from the
    cache reads none (its rows from the image's lookup tables), so the chat's load opens neither
    the checkpoint's files (on the HDD) nor torch."""

    def __init__(self, load, path, kw):
        self.model_dir, self.mtp_from = Path(path), None
        self._load, self._w = (lambda: load(path, **kw)), None

    def _real(self):
        if self._w is None:
            t = time.perf_counter()
            self._w = self._load()
            if os.environ.get("CHAT35_TIMES"):
                print(f"[chat35] the checkpoint opened in {time.perf_counter() - t:.3f} s",
                      file=sys.stderr)
        return self._w

    def __getitem__(self, k):
        return self._real()[k]

    def __iter__(self):
        return iter(self._real())

    def __len__(self):
        return len(self._real())

    def __contains__(self, k):
        return k in self._real()

    def __getattr__(self, name):
        return getattr(self._real(), name)


def _entry_meta(f: Path):
    """An image cache entry's header (imgcache._store's) and the offset of its first run."""
    from opentpu import imgcache as IC
    with open(f, "rb") as src:
        if src.read(8) != IC.MAGIC:
            raise ValueError(f"{f}: not an image cache entry")
        n = struct.unpack("<Q", src.read(8))[0]
        return json.loads(src.read(n)), 16 + n


def expand(f: Path, raw: Path) -> dict:
    """The entry f as a raw image (CHAT35_RAW): each slice's bytes at their offsets, its zero pages holes (a sparse
    file), then the arrays the build left, each page-aligned; raw's sidecar (.json) names the entry (key, size) and
    where each part is. Written to a temporary name, then renamed."""
    meta, at = _entry_meta(f)
    slices, base = [], 0
    for s in meta["slices"]:
        slices.append([base, s["nbytes"]])
        base = -(-(base + s["nbytes"]) // 4096) * 4096
    arrays = {}
    for p, a in meta["arrays"].items():
        n = int(np.prod(a["shape"])) * np.dtype(a["dtype"]).itemsize
        arrays[p] = {**a, "at": base}
        base = -(-(base + n) // 4096) * 4096
    side = {"key": f.stem, "entry_bytes": f.stat().st_size, "slices": slices, "arrays": arrays,
            "json": meta["json"], "weights": meta["weights"], "bytes": base}
    raw.parent.mkdir(parents=True, exist_ok=True)
    tmp = raw.with_name(f".{raw.name}.{os.getpid()}")
    buf = memoryview(bytearray(64 << 20))
    t0 = time.time()

    def copy(src, out, n):
        while n:
            k = src.readinto(buf[:min(n, len(buf))])
            if not k:
                raise EOFError(f"{f}: a short entry")
            out.write(buf[:k])
            n -= k
    try:
        with open(f, "rb") as src, open(tmp, "wb") as out:
            out.truncate(base)
            src.seek(at)
            for (b, _), s in zip(slices, meta["slices"]):
                for o, n in s["runs"]:
                    out.seek(b + o)
                    copy(src, out, n)
            for p, a in arrays.items():
                out.seek(a["at"])
                copy(src, out, int(np.prod(a["shape"])) * np.dtype(a["dtype"]).itemsize)
            if src.read(1):
                raise ValueError(f"{f}: a long entry")
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, raw)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    raw.with_suffix(".json").write_text(json.dumps(side))
    print(f"chat35: {f.name} expanded into {raw} in {time.time() - t0:.0f} s", flush=True)
    return side


def data_runs(path: Path, base: int, n: int) -> list:
    """The raw image's data extents in [base, base + n) (SEEK_DATA / SEEK_HOLE), from 0: they hold every non-zero byte
    of the slice there (its holes read as zeros): BoardBackend's images.runs, the scrub leaving them and only they
    written after it (board.keep_ranges)."""
    out, fd = [], os.open(path, os.O_RDONLY)
    try:
        x, end = base, base + n
        while x < end:
            try:
                lo = os.lseek(fd, x, os.SEEK_DATA)
            except OSError:                     # (ENXIO: no data past x)
                break
            if lo >= end:
                break
            hi = min(os.lseek(fd, lo, os.SEEK_HOLE), end)
            out.append((lo - base, hi - base))
            x = hi
    finally:
        os.close(fd)
    return out


RUNS: dict = {}                 # raw_images' data runs a slice, for the backend (backend_with_cfg)


def batch_runs() -> None:
    """--slots N: the batch area among RUNS' runs (it lies in the raw image's holes; batch.build's first contents in
    it, the scrub would leave zeros there), each slice's [base, end) appended when batch.build has written it."""
    from opentpu.llm import batch as BA
    if getattr(BA.build, "adds_runs", False):
        return
    build = BA.build

    def build_and_run(images, a, spec, cfg):
        build(images, a, spec, cfg)
        for r in RUNS.get("runs") or []:
            r.append((a["base"], a["end"]))
    build_and_run.adds_runs = True
    BA.build = build_and_run


def raw_images(f: Path, raw_dir: Path, image, W) -> list:
    """The entry f's images as copy-on-write maps of its raw image in raw_dir (expanded first when it is missing or of
    another entry), and what the build left set on `image` (imgcache.build's hit, without the read: the pages come
    from the page cache, or the disk as the card's write reads them). Copy-on-write: --slots N's batch area
    (opentpu.llm.batch.build: penalty factors 1, -inf logits) is written into the maps, the file left as it is."""
    from opentpu import imgcache as IC
    raw = raw_dir / f"{f.stem}.raw"
    try:
        side = json.loads(raw.with_suffix(".json").read_text())
        ok = side["key"] == f.stem and side["entry_bytes"] == f.stat().st_size and \
            raw.stat().st_size == side["bytes"]
    except (OSError, ValueError, KeyError):
        ok = False
    if not ok:
        print(f"chat35: expanding the image cache entry into {raw} (once)", flush=True)
        side = expand(f, raw)
    images = [np.memmap(raw, np.uint8, "c", offset=b, shape=(n,)) for b, n in side["slices"]]
    if os.environ.get("CHAT35_RUNS", "1") != "0":
        RUNS["runs"] = [data_runs(raw, b, n) for b, n in side["slices"]]
        batch_runs()
    for m in images:
        try:
            m._mmap.madvise(mmap.MADV_SEQUENTIAL)
        except (AttributeError, OSError):
            pass
    left = {p: np.array(np.memmap(raw, np.dtype(a["dtype"]), "r", offset=a["at"], shape=tuple(a["shape"])))
            for p, a in side["arrays"].items()}
    left.update(side["json"])
    left.update({p: W for p in side["weights"]})
    for path, v in left.items():
        IC._set(image, path, v)
    return images


def install():
    """chat35's setup of otpu-chat (opentpu.host.chat), main()'s: the image from the cache (the raw image mapped), the
    configuration the entry was built with, the chat's formats, the weights read lazily, gen_warm's arguments kept, the
    35B resident, a prompt part layer by layer (CHAT35_LM) or a token a run, the layer runs warmed after each turn.
    Returns warm_layers(eng), or None once CHAT35_EXPAND's raw image is written (the caller exits)."""
    from opentpu import imgcache as IC
    import opentpu.host.chat as C
    from opentpu.llm import generate as G
    from opentpu.llm import qwen3 as Q
    build_mode, key_mode = os.environ.get("CHAT35_BUILD") == "1", os.environ.get("CHAT35_KEY") == "1"
    entry = os.environ.get("CHAT35_ENTRY", "")
    raw_dir = os.environ.get("CHAT35_RAW", "")  # the entry as a raw image there, mapped (raw_images)
    cfg = pickle.loads((HERE / "bcu1525-wg-bhw16.pkl").read_bytes())
    hit = {"on": False}
    build, make_backend, load_spec = IC.build, C.make_backend, C.load_spec

    def guarded_build(image, W, layout):
        d = IC.cache_dir()
        ck = None if d is None else IC.checkpoint(getattr(W, "model_dir", None))
        k = None if ck is None else IC.key(image, layout, ck)
        have = k is not None and (d / k[:2] / f"{k}.img").is_file()
        print(f"chat35: image key {k} ({'in the cache' if have else 'not in the cache'}; low area "
              f"{image.low_end / 2**20:.0f} MiB, {image.nbytes / 2**30:.2f} GiB)", flush=True)
        if key_mode:
            os._exit(0)
        if build_mode:
            t0 = time.time()
            build(image, W, layout)
            print(f"chat35: built {IC.last} in {time.time() - t0:.0f} s: {k}", flush=True)
            os._exit(0 if IC.last[0] in ("hit", "miss") else 1)
        if not have or (entry and k != entry):
            raise SystemExit(f"chat35: the image (key {k}) is not in the cache{f' (want {entry})' if entry else ''}: "
                             "it is built on the build host (CHAT35_BUILD=1: c35.sh), never here")
        if raw_dir:
            t0 = time.perf_counter()
            out = raw_images(d / k[:2] / f"{k}.img", Path(raw_dir), image, W)
            IC.stats["hit"] += 1
            IC.last = ("hit", time.perf_counter() - t0)
            hit["on"] = True
            return out
        out = build(image, W, layout)
        if IC.last[0] != "hit":
            raise SystemExit(f"chat35: image cache {IC.last}, not a hit")
        hit["on"] = True
        return out

    def backend_with_cfg(name, spec, cap, dev, model=None, **kw):
        if build_mode or key_mode:
            return Null, cfg
        backend, _ = make_backend(name, spec, cap, dev, model, **kw)

        def with_runs(c, imgs):         # (the raw image's data runs: the scrub leaves them, only they are written)
            if RUNS.get("runs") is not None:
                imgs.runs = RUNS["runs"]
            return backend(c, imgs)
        return with_runs, cfg           # the configuration the entry was built with

    def spec_with_formats(path):
        from opentpu.host.streamed import defaults
        spec = load_spec(path)
        return replace(spec, formats=defaults(spec)[2])     # the chat's mix for the family

    prefill_chunks = Q.Engine.prefill_chunks
    # CHAT35_LM=R: a prompt part of CHAT35_LM_MIN tokens or more layer by layer, R rows a run
    # (Engine(layer_major=R), prefill_layers); shorter parts, and every part without it, through
    # the decode step a token a run
    lm = int(os.environ.get("CHAT35_LM") or 0)
    lm_min = int(os.environ.get("CHAT35_LM_MIN") or 64)
    # CHAT35_LOOP=1 (with CHAT35_LM): a layer's runs in a bucket as one run (Engine.layer_loop, prefill_loop: pf500
    # item 6), the embed runs too
    loop = os.environ.get("CHAT35_LOOP") == "1"
    if lm:
        engine = C.Engine

        def engine_lm(*a, **kw):
            eng = engine(*a, layer_major=lm, **kw)
            eng.layer_loop = loop
            warm_layers(eng)
            return eng
        C.Engine = engine_lm

    def warm_layers(eng, ahead: int = 2) -> None:
        # the layer runs of the context's bucket and the next one's, compiled by the engine's workers while the user
        # reads and types (a part's first runs in a bucket otherwise wait for theirs: ~3.4 s); the runs of the
        # buckets behind it dropped (each bucket's 81 programs stay in the engine's memory)
        b, L, R = eng.pos // eng.block + 1, eng.spec.layers, eng.layer_major
        for d in (eng._layer_runs, eng._layer_next):     # (a loop's key: its bucket third)
            for k in [k for k in d if isinstance(k, tuple) and isinstance(k[1], int)
                      and k[2 if k[0] == "L" else 1] < b]:
                del d[k]
        keys, last = ["head"], min(b + ahead, G.buckets(eng.cap, eng.block) + 1) - 1
        lk = getattr(eng, "layer_loop", False) and R > 1    # (the loops' keys: prefill_loop.loop_key)
        for bb in range(b, last + 1):
            keys += [(-1, bb, 1, True)] + [(li, bb, n, True) for li in range(L) for n in (R, 1)] if not lk else \
                [("L", -1, bb, 1)] + [("L", li, bb, n) for li in range(L) for n in (R, 1)]
        eng._precompile_layers(keys)
        if os.environ.get("CHAT35_TIMES"):      # when the workers finish them
            futs = [eng._layer_next[k] for k in keys if k in eng._layer_next]
            t, left = time.perf_counter(), [len(futs)]

            def done(_f):
                left[0] -= 1
                if not left[0]:
                    print(f"[chat35] warm buckets {b}-{last}: {len(futs)} runs compiled in "
                          f"{time.perf_counter() - t:.3f} s", file=sys.stderr)
            for f in futs:
                f.add_done_callback(done)

    ask, resume = C.Chat.ask, C.Chat.resume

    def ask_warm(self, *a, **kw):
        try:
            return ask(self, *a, **kw)
        finally:
            if self.eng.layer_major:
                warm_layers(self.eng)

    def resume_warm(self, *a, **kw):
        try:
            return resume(self, *a, **kw)
        finally:
            if self.eng.layer_major:
                warm_layers(self.eng)
    C.Chat.ask, C.Chat.resume = ask_warm, resume_warm

    def token_steps(self, tokens, seq=0, chunk=None, **kw):
        tokens = list(tokens)
        if chunk is None and self.spec.moe is not None and (not self.layer_major
                                                             or len(tokens) < lm_min):
            chunk = 1
        return prefill_chunks(self, tokens, seq, chunk, **kw)

    from opentpu.host import gen_warm as GW
    gw_start = GW.start

    def start_saving(eng, sampling=None):
        # gen_warm's arguments kept for warm35.py (every bucket's programs ahead, between chats)
        be = eng.backend
        args = (eng.spec, eng.cfg, eng.cap, eng.batch, eng.rows, eng._image_kw,
                eng.image.stream_fill, eng.layout, eng.block, eng._conv_lo,
                bool(getattr(be, "chains", False)), eng.gen_debug, eng.gen_split,
                bool(getattr(be, "waitc", False)), GW.samplings(sampling))
        try:
            tmp = HERE / f".warm35.{os.getpid()}.pkl"
            tmp.write_bytes(pickle.dumps(args))
            os.replace(tmp, HERE / "warm35.pkl")
        except OSError as e:
            print(f"chat35: warm35.pkl not written: {e}", file=sys.stderr)
        return gw_start(eng, sampling)

    if os.environ.get("CHAT35_EXPAND"):          # the entry named into CHAT35_RAW, then exit
        f = Path(os.environ["CHAT35_EXPAND"])
        expand(f, Path(raw_dir) / f"{f.stem}.raw")
        return None
    IC.build, C.make_backend, C.load_spec = guarded_build, backend_with_cfg, spec_with_formats
    load_weights = C.load_weights
    C.load_weights = lambda path, **kw: _LazyWeights(load_weights, path, kw)
    GW.start = start_saving
    Q.Engine.prefill_chunks = token_steps
    C.ST.streams = lambda spec: False   # the 35B resident: the BCU's DRAM holds it whole
    if os.environ.get("CHAT35_TIMES"):          # a timeline of the decode's host calls (stderr)
        _timed(Q.Engine, "generate_card", "_generate_prog", "_generate_chain", "step",
               "prefill_layers", "_layer_run")
        from opentpu.host import board as BD
        _timed(BD.BoardBackend, "run_generate", "prepare", "start", "wait")
        _timed(BD.BoardBackend, "__init__")
        _timed(BD.Board, "scrub")
        _timed(Q.Engine, "__init__")
        _timed(BD.Board, "info")
        _timed(C, "load_tokenizer", "load_weights", "make_backend", "load_spec")
    return warm_layers


def main():
    if install() is None:
        return 0
    import opentpu.host.chat as C
    build_mode, key_mode = os.environ.get("CHAT35_BUILD") == "1", os.environ.get("CHAT35_KEY") == "1"
    args = FIXED + sys.argv[1:]
    if build_mode or key_mode:
        args += ["--prompt", "-"]
    prof = os.environ.get("CHAT35_PROF")        # a cProfile of the chat into this file (pstats)
    if prof:
        import cProfile
        pr = cProfile.Profile()
        try:
            return pr.runcall(C.main, args)
        finally:
            pr.dump_stats(prof)
    return C.main(args)


def open_engine(argv=()):
    """The engine otpu-chat35 chats with, for otpu-serve35: install(), then otpu-chat's main with FIXED, `argv` and
    --prompt - over a stand-in for its Chat that takes what main opens (the engine, the tokenizer, the sampler, the
    clock) and asks nothing. main starts gen_warm (warm35.pkl kept) and gives otpu-smi the weights as for a chat.
    Returns them (eng, tok, pick, clock_mhz) with warm_layers and the stand-in's class (stub); None after
    CHAT35_EXPAND."""
    import opentpu.host.chat as C
    warm_layers = install()
    if warm_layers is None:
        return None
    got = {}

    class Opened:
        use_mtp = False

        def __init__(self, eng, tok, think, pick, max_new, clock_mhz=0.0):
            got.update(eng=eng, tok=tok, pick=pick, clock_mhz=clock_mhz)

        def attach(self, src):
            raise ValueError("no images: the 35B has no vision side")

        def ask_plain(self, text):
            pass
    chat = C.Chat
    C.Chat = Opened
    try:
        C.main(FIXED + list(argv) + ["--prompt", "-"])
    finally:
        C.Chat = chat
    if "eng" not in got:
        raise SystemExit("chat35: otpu-chat opened no engine")
    return SimpleNamespace(**got, warm_layers=warm_layers, stub=Opened)


if __name__ == "__main__":     # (the Engine's spawned compile workers import this module)
    sys.exit(main())
