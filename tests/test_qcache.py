"""opentpu/qcache.py, the disk cache of the 4-bit quantizer's results, and tools/qual/prebuild.py:
the cached result is the quantizer's, bit for bit; the key follows the content, the format and
D; the size cap, the free-disk floor, a damaged entry, the off switch; and an image prebuilt
for the card's tools comes out of the cache, whole, for every layout a tool builds."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from opentpu import qcache
from opentpu import quant as Q
from opentpu.llm.qwen3 import Engine, Spec, device_config

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A fresh cache for every matrix size; the counts from zero."""
    monkeypatch.setenv("OTPU_IMAGE_CACHE", str(tmp_path))
    monkeypatch.setattr(qcache, "MIN_ELEMS", 0)
    monkeypatch.setattr(qcache, "FREE_FLOOR", 0)
    monkeypatch.setattr(qcache, "stats", dict.fromkeys(qcache.stats, 0))
    return tmp_path


def _w(seed, n=96, k=256):
    return np.random.default_rng(seed).normal(0, 0.05, (n, k)).astype(np.float32)


def _entries(d):
    return sorted(p for p in Path(d).glob("*/*.npz") if not p.name.startswith("."))


def _entry(d, W):
    k = qcache.key(W, "fp4", 128)
    return Path(d) / k[:2] / f"{k}.npz"


@pytest.mark.parametrize("fmt", ["fp4", "int4"])
def test_cached_result_is_the_quantizers(cache, fmt):
    W = _w(0)
    want = Q.quantize_mxu(W, fmt, 128)
    for n in range(2):                          # a miss that writes, then a hit
        q, s = qcache.quantize_mxu(W, fmt, 128)
        assert q.dtype == want[0].dtype and s.dtype == want[1].dtype
        assert np.array_equal(q, want[0]) and np.array_equal(s, want[1])
    assert qcache.stats == {"hit": 1, "miss": 1, "write": 1, "skip": 0}
    assert len(_entries(cache)) == 1


def test_key_follows_content_format_and_d():
    W = _w(1)
    k = qcache.key(W, "fp4", 128)
    assert qcache.key(W.copy(), "fp4", 128) == k
    assert qcache.key(np.asfortranarray(W), "fp4", 128) == k        # the values, not the layout
    assert qcache.key(W.astype(np.float64), "fp4", 128) == k        # the quantizer's fp32 input
    V = W.copy()
    V[5, 7] = np.nextafter(V[5, 7], np.float32(1))
    assert qcache.key(V, "fp4", 128) != k
    assert qcache.key(W, "int4", 128) != k
    assert qcache.key(W, "fp4", 64) != k
    assert qcache.key(W.reshape(192, 128), "fp4", 128) != k         # the shape


def test_off_small_and_int8_are_not_cached(cache, monkeypatch):
    W = _w(2)
    qcache.quantize_mxu(W, "int8", 128)
    monkeypatch.setattr(qcache, "MIN_ELEMS", W.size + 1)
    qcache.quantize_mxu(W, "fp4", 128)
    monkeypatch.setattr(qcache, "MIN_ELEMS", 0)
    monkeypatch.setenv("OTPU_IMAGE_CACHE", "0")
    assert qcache.cache_dir() is None
    q, s = qcache.quantize_mxu(W, "fp4", 128)
    assert np.array_equal(q, Q.quantize_mxu(W, "fp4", 128)[0])
    assert _entries(cache) == [] and qcache.stats["miss"] == 0


def test_no_write_under_the_free_disk_floor(cache, monkeypatch):
    monkeypatch.setattr(qcache, "FREE_FLOOR", 1 << 62)
    qcache.quantize_mxu(_w(3), "fp4", 128)
    assert _entries(cache) == [] and qcache.stats["skip"] == 1


def test_least_recently_used_go_over_the_cap(cache, monkeypatch):
    import os
    import time
    Ws = [_w(10 + i) for i in range(5)]
    for W in Ws[:4]:
        qcache.quantize_mxu(W, "fp4", 128)
    size = _entry(cache, Ws[0]).stat().st_size
    t = time.time() - 100
    for i, W in enumerate(Ws[:4]):              # used in this order, then the first again
        os.utime(_entry(cache, W), (t + i, t + i))
    qcache.quantize_mxu(Ws[0], "fp4", 128)      # a hit: now the most recent
    monkeypatch.setenv("OTPU_IMAGE_CACHE_GB", str(3.5 * size / 2**30))
    qcache.quantize_mxu(Ws[4], "fp4", 128)      # the fifth entry: over the cap of 3.5
    left = set(_entries(cache))
    assert _entry(cache, Ws[1]) not in left and _entry(cache, Ws[2]) not in left  # the oldest
    assert {_entry(cache, Ws[i]) for i in (0, 3, 4)} <= left


def test_damaged_entry_is_quantized_again(cache):
    W = _w(4)
    qcache.quantize_mxu(W, "fp4", 128)
    f = _entries(cache)[0]
    f.write_bytes(b"not an npz")
    q, _ = qcache.quantize_mxu(W, "fp4", 128)
    assert np.array_equal(q, Q.quantize_mxu(W, "fp4", 128)[0])
    assert qcache.stats["miss"] == 2 and qcache.stats["write"] == 2
    with np.load(f) as z:
        assert np.array_equal(z["q"], q)


@pytest.fixture(scope="module")
def tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    hc = transformers.Qwen3Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=128, intermediate_size=512,
                                  vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
                                  tie_word_embeddings=True, max_position_embeddings=4096)
    m = transformers.Qwen3ForCausalLM(hc).float().eval()
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    return W, Spec(256, 2, 4, 2, 128, 512, 1000)


def _prebuild_module():
    sp = importlib.util.spec_from_file_location("prebuild", ROOT / "tools/qual/prebuild.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("head", ["int8", "fp4"])
def test_card_tools_find_the_prebuilt_image(cache, tiny, head, monkeypatch):
    """prebuild.py's image fills the cache with every 4-bit matrix; then the tools' Engines,
    per-position or resident, at another KV capacity and prefill rows, quantize nothing, and
    their images are the uncached build's, byte for byte."""
    W, spec = tiny
    cfg = device_config(spec, 512, lookup=True)         # the card's: room for every layout
    n = _prebuild_module().prebuild(cfg, spec, W, "fp4", head)
    assert n["miss"] > 0 and n["write"] == n["miss"] and n["hit"] == 0
    for kw in ({"cap": 256, "resident": True}, {"cap": 512, "resident": False, "rows": 2}):
        s0 = dict(qcache.stats)
        eng = Engine(spec, W, cfg=cfg, wformat="fp4", head_format=head, **kw)
        got = eng.image.build(W)                # the Engine's build plus this one
        assert qcache.stats["miss"] == s0["miss"]
        assert qcache.stats["hit"] - s0["hit"] == 2 * n["miss"]
        monkeypatch.setenv("OTPU_IMAGE_CACHE", "0")
        want = eng.image.build(W)
        monkeypatch.setenv("OTPU_IMAGE_CACHE", str(cache))
        assert len(got) == len(want)
        assert all(np.array_equal(a, b) for a, b in zip(got, want))
