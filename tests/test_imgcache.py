"""opentpu/imgcache.py, the disk cache of whole model images: a hit is the build, byte for byte,
with what the build left on the image for the host (an int8 embedding table on the host), and
a MoE's streamed experts served from a hit's weights; the key follows the checkpoint, the layout, the formats and the environment, but not the host's
runtime variables (otpu-lock's pid); entries leave the zero pages out; a build that leaves
something the cache cannot store, a damaged entry, the size cap, the free-disk floor and the
off switches."""
import dataclasses
import json
import os
import struct

import numpy as np
import pytest

from opentpu import imgcache as IC
from opentpu import progcache as PC
from opentpu.llm import load_spec
from opentpu.llm.qwen3 import Engine, Image, device_config, load_weights

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """A tiny Qwen3 checkpoint on disk (config.json, model.safetensors), its LM head untied."""
    from safetensors.numpy import save_file
    d = tmp_path_factory.mktemp("qwen3")
    torch.manual_seed(0)
    hc = transformers.Qwen3Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=128, intermediate_size=512,
                                  vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
                                  tie_word_embeddings=False, max_position_embeddings=4096)
    m = transformers.Qwen3ForCausalLM(hc).float().eval()
    save_file({k: np.array(v.numpy()) for k, v in m.state_dict().items()},
              str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps(hc.to_dict()))
    return d


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A fresh cache, no free-disk floor, the counts from zero."""
    monkeypatch.setenv("OTPU_IMG_CACHE", str(tmp_path / "img"))
    monkeypatch.setenv("OTPU_IMG_CACHE_FREE_GB", "0")
    monkeypatch.setattr(IC, "stats", dict.fromkeys(IC.stats, 0))
    return tmp_path / "img"


class Keep:
    """A backend that keeps a copy of the images it is given."""

    def __init__(self, cfg, images):
        self.images = [np.array(i) for i in images]


def _engine(d, spec=None, W=None, **kw):
    kw = {"cap": 256, "wformat": "fp4", "head_format": "int8", "resident": True, **kw}
    return Engine(spec or load_spec(d), load_weights(d) if W is None else W, backend=Keep, **kw)


def _entries(d):
    return sorted(p for p in d.glob("*/*.img") if not p.name.startswith("."))


def _same(a, b) -> bool:
    """Two values alike: arrays by their bytes, dicts and sequences by their entries."""
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(map(_same, a, b))
    if callable(a):
        return callable(b)
    return a == b


def test_a_hit_is_the_build(ckpt, cache, monkeypatch):
    """The first Engine builds the image and stores it; the second reads it back, byte for
    byte the build's, with every attribute of its Image the build's; with the cache off the
    image is built again, the same."""
    a = _engine(ckpt)
    assert IC.last[0] == "miss" and IC.stats == {"hit": 0, "miss": 1, "write": 1, "skip": 0}
    assert len(_entries(cache)) == 1
    b = _engine(ckpt)
    assert IC.last[0] == "hit" and IC.stats["hit"] == 1
    monkeypatch.setenv("OTPU_IMG_CACHE", "0")
    c = _engine(ckpt)
    assert IC.last[0] == "off"
    for e in (b, c):
        assert all(np.array_equal(x, y) for x, y in zip(a.backend.images, e.backend.images))
        assert _same(vars(a.image), vars(e.image))


def test_what_the_build_leaves_comes_back(ckpt, cache):
    """An int8 embedding table of its own on the host (embed_host: the build quantizes it into
    the image's store, lookup["store"]): a hit puts the store back, and the rows the host
    writes before a run are the built engine's."""
    spec = dataclasses.replace(load_spec(ckpt), embed="int8")
    a = _engine(ckpt, spec, embed_host=True, wformat="int8", head_format=None)
    assert a.image.embed_host and a.image.lookup["store"].shape[0] == 1000
    b = _engine(ckpt, spec, embed_host=True, wformat="int8", head_format=None)
    assert IC.last[0] == "hit"
    assert b.image.lookup["store"] is not a.image.lookup["store"]
    assert _same(vars(a.image), vars(b.image))
    for (x, u), (y, v) in zip(a.image.host_rows([3, 999, 0]), b.image.host_rows([3, 999, 0])):
        assert x == y and np.array_equal(u, v)


def test_prebuild_fills_the_cache_for_the_chat(ckpt, cache):
    """opentpu.host.prebuild builds otpu-chat's image (its default KV capacity, resident, the
    card's configuration and the run's formats): the chat's Engine then reads it from the
    cache."""
    from opentpu.host import prebuild as PB
    spec = load_spec(ckpt)
    cfg = device_config(spec, PB.CAP, lookup=True)
    assert PB.prebuild(cfg, spec, load_weights(ckpt), "fp4", "int8")["image"] == "miss"
    Engine(spec, load_weights(ckpt), cap=2048, cfg=cfg, backend=Keep, wformat="fp4",
           head_format="int8", resident=True, prog_cache=False)
    assert IC.last[0] == "hit"


def test_a_build_leaving_something_else_is_not_cached(ckpt, cache, monkeypatch):
    """A build that leaves on the image what the cache does not store (an object) is not
    cached: built again every time, never handed back without it."""
    build = Image.build

    def odd(self, W):
        self.extra = object()
        return build(self, W)
    monkeypatch.setattr(Image, "build", odd)
    for _ in range(2):
        _engine(ckpt)
        assert IC.last[0] == "skip"
    assert IC.stats["skip"] == 2 and _entries(cache) == []


def test_a_build_keeping_lazy_weights_reads_none_of_them(ckpt, cache, monkeypatch):
    """A build that keeps the weights on the image (Gemma 4's _W) given lazy ones
    (qwen3.LazyWeights, a dict whose items() reads every tensor as fp32: the streamed MoE
    models'): the cache reads none of the tensors and stores none, the entry is written, and a
    hit hands the weights back. (gemma-4-26B-A4B-it's image was never cached on the card host:
    its 597 tensors, read again after each build, made an entry too big for the disk.)"""
    from opentpu.llm.qwen3 import LazyWeights
    build = Image.build

    def keep(self, W):
        self._W = W
        return build(self, W)
    monkeypatch.setattr(Image, "build", keep)
    monkeypatch.setattr(LazyWeights, "items", lambda self: pytest.fail("a tensor read"))
    W = LazyWeights(ckpt)
    _engine(ckpt, W=W)
    assert IC.last[0] == "miss" and len(_entries(cache)) == 1
    b = _engine(ckpt, W=W)
    assert IC.last[0] == "hit" and b.image._W is W


def test_a_moe_streaming_its_experts_from_a_hit(tmp_path, cache):
    """A tiny Qwen3.5-MoE checkpoint with two expert slots a layer (the experts stream from the
    host's server, which reads them from the weights): the hit's image is the build's, and
    its engine, serving from the hit's own weights, gives the built engine's logits bit for
    bit."""
    torch.manual_seed(0)
    kinds = ("linear", "attn")
    hc = transformers.Qwen3_5MoeTextConfig(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
        head_dim=256, vocab_size=1000, layer_types=["full_attention" if k == "attn" else
                                                    "linear_attention" for k in kinds],
        linear_num_key_heads=8, linear_num_value_heads=8, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=True,
        max_position_embeddings=4096, rms_norm_eps=1e-6, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=256, shared_expert_intermediate_size=256,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25})
    from safetensors.numpy import save_file
    m = transformers.Qwen3_5MoeForCausalLM(hc).float().eval()
    d = tmp_path / "moe"
    d.mkdir()
    save_file({k: np.array(v.numpy()) for k, v in m.state_dict().items()},
              str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps(hc.to_dict()))
    spec = load_spec(d)
    cfg = device_config(spec, 256, S=1, experts=2)
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 6)]
    images, logits = [], []
    for how in ("miss", "hit"):
        eng = Engine(spec, load_weights(d), cap=256, cfg=cfg, experts=2, wformat="fp4")
        assert IC.last[0] == how
        images.append(eng.backend.machine.slices[0].dram.copy())
        logits.append(np.array([eng.step(t) for t in toks]))
    assert eng.server.misses > len(toks)
    assert np.array_equal(images[0], images[1])
    assert np.array_equal(logits[0].view(np.uint32), logits[1].view(np.uint32))


def test_weights_without_a_checkpoint_are_not_cached(ckpt, cache):
    """W as a dict of arrays (the tests' models): no identity to key on, built every time."""
    W = load_weights(ckpt)
    _engine(ckpt, W={k: W[k] for k in W})
    assert IC.last[0] == "off" and _entries(cache) == []


def test_key_follows_checkpoint_layout_formats_and_environment(ckpt, cache, monkeypatch):
    """The key changes with the layout (KV capacity), the formats, the configuration, the
    checkpoint's files (a weight file touched) and the OTPU_* environment the layouts read
    (OTPU_FORMATS); not with the host's runtime variables (otpu-lock's OTPU_LOCK_HELD and
    OTPU_LOCK_PID, OTPU_BUILD_JOBS, the caches' own), which leave the program cache's key
    alone too (opentpu/progcache.py: a pid there missed every otpu-lock session)."""
    spec = load_spec(ckpt)
    cfg = device_config(spec, 512, lookup=True)

    def k(cap=256, ck=None, cfg=cfg, **kw):
        img = spec.image(cfg, cap, 1, 1, **kw)
        return IC.key(img, (spec, cfg, cap, kw), ck or IC.checkpoint(ckpt))
    base = k()
    assert k() == base
    assert k(cap=512) != base and k(wformat="fp4") != base
    assert k(cfg=dataclasses.replace(cfg, DRAM_BYTES=cfg.DRAM_BYTES * 2)) != base
    f = ckpt / "model.safetensors"
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    try:
        assert k() != base
    finally:
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert k() == base
    pk = PC.key(("layout",), ("gen", 1))
    for var, val, same in (("OTPU_FORMATS", "gateup@0-1=fp4", False),
                           ("OTPU_LOCK_HELD", "xdma0", True), ("OTPU_LOCK_PID", "4242", True),
                           ("OTPU_BUILD_JOBS", "3", True), ("OTPU_IMG_CACHE_GB", "1", True)):
        monkeypatch.setenv(var, val)
        assert (k() == base) == same, var
        assert (PC.key(("layout",), ("gen", 1)) == pk) == same, var
        monkeypatch.delenv(var)


def test_entries_leave_the_zero_pages_out(ckpt, cache):
    """An entry stores each slice's runs of pages with a nonzero byte (the KV caches are
    zeros): the runs, a partial last page; the tiny model's entry is a fraction of its image."""
    P = IC.PAGE
    img = np.zeros(5 * P + 100, np.uint8)
    img[P + 7] = 1
    img[2 * P] = 2
    img[5 * P + 99] = 3
    assert IC._runs(img) == [(P, 2 * P), (5 * P, 100)]
    assert IC._runs(np.zeros(3 * P, np.uint8)) == []
    a = _engine(ckpt, cap=1024)
    size = _entries(cache)[0].stat().st_size
    nonzero = sum(n for i in a.backend.images for _, n in IC._runs(i))
    assert nonzero < size < nonzero + 64 * 1024
    assert nonzero < 0.7 * sum(len(i) for i in a.backend.images)


def test_damaged_entry_is_built_again(ckpt, cache):
    a = _engine(ckpt)
    f = _entries(cache)[0]
    f.write_bytes(f.read_bytes()[:-1])          # a short entry (a copy cut off)
    b = _engine(ckpt)
    assert IC.last[0] == "miss" and IC.stats["write"] == 2
    assert all(np.array_equal(x, y) for x, y in zip(a.backend.images, b.backend.images))
    _engine(ckpt)
    assert IC.last[0] == "hit"


def test_least_recently_used_go_over_the_cap_and_none_under_the_floor(ckpt, cache,
                                                                     monkeypatch):
    """A hit makes its entry the most recently used; past OTPU_IMG_CACHE_GB the least recently
    used go (to 90% of it); nothing is written that would leave under OTPU_IMG_CACHE_FREE_GB of
    free disk."""
    import time
    _engine(ckpt)
    (f,) = _entries(cache)
    t = time.time() - 1000
    os.utime(f, (t, t))
    _engine(ckpt)
    assert IC.last[0] == "hit" and f.stat().st_mtime > t + 500
    (cache / "ab").mkdir()
    old = []
    for i in range(4):                          # four 1000-byte entries, the oldest first
        e = cache / "ab" / f"{i}.img"
        e.write_bytes(bytes(1000))
        os.utime(e, (t + i, t + i))
        old.append(e)
    f.unlink()
    monkeypatch.setenv("OTPU_IMG_CACHE_GB", str(3500 / 2**30))
    IC._evict(cache)
    assert [e.exists() for e in old] == [False, True, True, True]
    monkeypatch.setenv("OTPU_IMG_CACHE_GB", "16")
    monkeypatch.setenv("OTPU_IMG_CACHE_FREE_GB", str(1 << 30))
    _engine(ckpt)
    assert IC.last[0] == "skip" and IC.stats["skip"] == 1 and len(_entries(cache)) == 3


def test_an_entry_over_the_cap_is_not_written(ckpt, cache, monkeypatch):
    """An image larger than OTPU_IMG_CACHE_GB is not written (the eviction after its write would
    drop it at once, and the next load would build it again believing it cached): a skip, with
    its reason for the host's load line."""
    monkeypatch.setenv("OTPU_IMG_CACHE_GB", str(1000 / 2**30))
    _engine(ckpt)
    assert IC.last[0] == "skip" and IC.stats == {"hit": 0, "miss": 1, "write": 0, "skip": 1}
    assert not _entries(cache) and "OTPU_IMG_CACHE_GB" in IC.why_skipped
    monkeypatch.setenv("OTPU_IMG_CACHE_GB", "16")
    monkeypatch.setenv("OTPU_IMG_CACHE_FREE_GB", str(1 << 30))
    _engine(ckpt)
    assert IC.last[0] == "skip" and "OTPU_IMG_CACHE_FREE_GB" in IC.why_skipped
    monkeypatch.setenv("OTPU_IMG_CACHE_FREE_GB", "0")
    _engine(ckpt)
    assert IC.last[0] == "miss" and len(_entries(cache)) == 1


def _head(**h) -> bytes:
    """An entry's MAGIC and header, page-sized."""
    head = json.dumps(h).encode()
    return (IC.MAGIC + struct.pack("<Q", len(head)) + head).ljust(IC.PAGE, b"\0")


def test_slow_builds_outlive_quick_ones_over_the_cap(ckpt, cache, monkeypatch):
    """An entry's header keeps its build's seconds and its tree (src); over the cap each build
    second per GB of this tree's entries counts as KEEP_S of later use: a slow build used a while
    ago stays while a quick one used since goes, and a stale entry (unused for longer than its
    build buys) goes first. An entry of another tree (a former one's, after a host code swap) or
    without the fields is the least recently used rule's."""
    import time
    _engine(ckpt)
    (f,) = _entries(cache)
    assert IC._build_s(f) > 0
    f.unlink()
    monkeypatch.setattr(IC, "KEEP_S", 2.0**-18)     # a page-sized entry: KEEP_S per build second
    t = time.time() - 10000
    (cache / "ab").mkdir()
    ents = {}
    for name, build_s, src, used in (("none", None, None, t + 1000),
                                     ("stale", 1000, PC.source(), t - 5000),
                                     ("former", 5000, "0" * 32, t),
                                     ("quick", 0, PC.source(), t + 3000),
                                     ("slow", 5000, PC.source(), t)):
        e = cache / "ab" / f"{name}.img"
        e.write_bytes(bytes(IC.PAGE) if build_s is None else _head(build_s=build_s, src=src))
        os.utime(e, (used, used))
        ents[name] = e
    monkeypatch.setenv("OTPU_IMG_CACHE_GB", str(3 * IC.PAGE / 2**30))
    IC._evict(cache)
    assert {n for n, e in ents.items() if e.exists()} == {"quick", "slow"}


def test_a_write_under_the_floor_drops_what_it_outranks(ckpt, cache, monkeypatch):
    """Under the cache's size, a write that would leave under OTPU_IMG_CACHE_FREE_GB of free disk
    drops the entries below it in the eviction order (a former tree's, unused since) until it
    fits, and keeps those above it (a slow build); when dropping all it outranks would not make
    the room, it is not written and none is dropped."""
    import time
    import types
    _engine(ckpt)
    (f,) = _entries(cache)
    n = f.stat().st_size
    f.unlink()
    t = time.time() - 1000
    (cache / "ab").mkdir()

    def entry(name, build_s):
        e = cache / "ab" / f"{name}.img"
        head = bytes(IC.PAGE) if build_s is None else _head(build_s=build_s, src=PC.source())
        e.write_bytes(head.ljust(n, b"\0"))
        os.utime(e, (t, t))
        return e

    old, slow = entry("old", None), entry("slow", 10**9)
    floor = 1 << 30
    disk = floor + 2.5 * n                  # room for two entries over the floor, not three
    monkeypatch.setenv("OTPU_IMG_CACHE_FREE_GB", "1")
    monkeypatch.setattr(IC.shutil, "disk_usage", lambda _: types.SimpleNamespace(
        free=int(disk - sum(p.stat().st_size for p in cache.rglob("*.img")))))
    _engine(ckpt)
    assert IC.last[0] == "miss" and not old.exists() and slow.exists()
    old2 = entry("old2", None)
    monkeypatch.setenv("OTPU_IMG_CACHE_FREE_GB", "2")
    _engine(ckpt, cap=512)
    assert IC.last[0] == "skip" and old2.exists() and len(_entries(cache)) == 3


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes a read-only directory")
def test_a_cache_it_cannot_write_is_no_error(ckpt, cache):
    """A cache directory the process cannot write (another user's, a full disk): its entries
    still hit, and a new image is built and used, not kept."""
    a = _engine(ckpt)
    dirs = [cache] + [p for p in cache.iterdir() if p.is_dir()]
    try:
        for p in dirs:
            p.chmod(0o555)
        b = _engine(ckpt)
        assert IC.last[0] == "hit"
        c = _engine(ckpt, cap=512)
        assert IC.last[0] == "skip" and IC.stats["skip"] == 1
    finally:
        for p in dirs:
            p.chmod(0o755)
    assert all(np.array_equal(x, y) for x, y in zip(a.backend.images, b.backend.images))
    assert len(_entries(cache)) == 1 and c.image.cap == 512


def test_cache_directory_and_off_switches(tmp_path, monkeypatch):
    """OTPU_IMG_CACHE=<dir> or 0; else beside the 4-bit matrices' cache (img under
    OTPU_IMAGE_CACHE's directory, ~/otpu-build/qcache/img by default), off with it."""
    monkeypatch.delenv("OTPU_IMG_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OTPU_IMAGE_CACHE", raising=False)
    assert IC.cache_dir() is None                           # no ~/otpu-build
    (tmp_path / "otpu-build").mkdir()
    assert IC.cache_dir() == tmp_path / "otpu-build/qcache/img"
    monkeypatch.setenv("OTPU_IMAGE_CACHE", str(tmp_path / "q"))
    assert IC.cache_dir() == tmp_path / "q/img"
    monkeypatch.setenv("OTPU_IMAGE_CACHE", "0")
    assert IC.cache_dir() is None
    monkeypatch.setenv("OTPU_IMG_CACHE", str(tmp_path / "i"))
    assert IC.cache_dir() == tmp_path / "i"
    monkeypatch.setenv("OTPU_IMG_CACHE", "0")
    monkeypatch.delenv("OTPU_IMAGE_CACHE")
    assert IC.cache_dir() is None
