"""opentpu/qcache.py, the disk cache of the 4-bit quantizer's results, and opentpu/host/prebuild.py:
the cached result is the quantizer's, bit for bit; the key follows the content, the format and
D; the size cap, the free-disk floor, a damaged entry, the off switch; and an image prebuilt
for the card's tools comes out of the cache, whole, for every layout a tool builds."""
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from opentpu import qcache
from opentpu.host import prebuild as PB
from opentpu import quant as Q
from opentpu.llm.qwen3 import Engine, Spec, device_config

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


@pytest.mark.parametrize("head", ["int8", "fp4"])
def test_card_tools_find_the_prebuilt_image(cache, tiny, head, monkeypatch):
    """prebuild's image fills the cache with every 4-bit matrix; then the tools' Engines,
    per-position or resident, at another KV capacity and prefill rows, quantize nothing, and
    their images are the uncached build's, byte for byte."""
    W, spec = tiny
    cfg = device_config(spec, 512, lookup=True)         # the card's: room for every layout
    n = PB.prebuild(cfg, spec, W, "fp4", head)
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


def test_prebuild_main_picks_runs_configuration_and_memory(tmp_path, monkeypatch, capsys):
    """prebuild's command line: only the 4-bit runs; --cfg, else OTPU_PREBUILD_CFG, else the
    production deploy's, else the newest kept card configuration; no build under MIN_GB."""
    assert [PB.is_4bit(r) for r in ("a:int8:-", "a:fp4:int8", "a:int8:fp4", "a:int4:-")] == \
        [False, True, True, True]
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OTPU_PREBUILD_CFG", raising=False)
    assert PB.default_cfg() is None
    kept = tmp_path / "otpu-build/refcache/configs"
    kept.mkdir(parents=True)
    for i, n in enumerate(("old", "new")):
        (kept / f"{n}.pkl").write_bytes(pickle.dumps(i))
        os.utime(kept / f"{n}.pkl", (1000 + i, 1000 + i))
    assert PB.default_cfg() == kept / "new.pkl"
    prod = tmp_path / "otpu-build/production/qual"
    prod.mkdir(parents=True)
    (prod / "cfg.pkl").write_bytes(pickle.dumps(2))
    assert PB.default_cfg() == prod / "cfg.pkl"
    monkeypatch.setenv("OTPU_PREBUILD_CFG", str(kept / "old.pkl"))
    assert PB.default_cfg() == kept / "old.pkl"
    monkeypatch.setenv("OTPU_IMAGE_CACHE", str(tmp_path / "c"))
    monkeypatch.setattr(PB, "mem_gb", lambda: 3.0)
    built = []
    monkeypatch.setattr(PB, "prebuild", lambda *a, **k: built.append(a))
    assert PB.main(["qwen3:int8:-", "qwen3:fp4:int8"]) == 0
    out = capsys.readouterr().out
    assert "configuration " + str(kept / "old.pkl") in out and "; qwen3:fp4:int8" in out
    assert "qwen3:fp4:int8: not built, MemAvailable 3.0 GiB" in out and built == []
    monkeypatch.setenv("OTPU_IMAGE_CACHE", "0")
    assert PB.main(["qwen3:fp4:int8"]) == 0 and "cache is off" in capsys.readouterr().out


def test_otpu_lock_prebuilds_before_the_lock(tmp_path, monkeypatch):
    """otpu-lock --prebuild: opentpu.host.prebuild at nice 19 first, then CMD under the lock."""
    import subprocess
    from opentpu.host import runstate
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(subprocess, "call", lambda cmd, env=None: calls.append((cmd, env)) or 0)
    assert runstate.hold_main(["--dev", "/dev/fake9", "--wait", "0", "--prebuild", "m:fp4:int8",
                               "--prebuild-cfg", "c.pkl", "--", "true", "x"]) == 0
    (pre, _), (cmd, env) = calls
    assert pre[:3] == ["nice", "-n", "19"] and pre[4:] == ["-m", "opentpu.host.prebuild", "--cfg",
                                                           "c.pkl", "m:fp4:int8"]
    assert cmd == ["true", "x"] and env["OTPU_LOCK_HELD"] == "fake9"
    calls.clear()
    assert runstate.hold_main(["--dev", "/dev/fake9", "--", "true"]) == 0
    assert [c for c, _ in calls] == [["true"]]


class _Child:
    """A run's build process for prebuild.main: done after `steps` polls; `during(n)` is called
    at the n-th poll (a session going quiet mid-build)."""
    log: list = []

    def __init__(self, cmd, steps=3, during=None):
        self.run, self.left, self.during, self.n, self.returncode = cmd[-1], steps, during, 0, None
        _Child.log.append(("start", self.run))

    def poll(self):
        self.n += 1
        if self.during:
            self.during(self.n)
        self.left -= 1
        if self.left <= 0 and self.returncode is None:
            self.returncode = 0
            _Child.log.append(("done", self.run))
        return self.returncode

    def terminate(self):
        _Child.log.append(("stop", self.run))

    def wait(self):
        self.returncode = -15


def test_prebuild_keeps_quiet_while_a_session_measures(tmp_path, monkeypatch, capsys):
    """The quiet file (~/otpu-build/QUIET with a live pid: a session measures host-sensitive
    performance): prebuild starts no build while it is there, stops a build when it appears and
    builds that run again after it, ignores a file whose pid is gone, and gives up after
    OTPU_PREBUILD_QUIET_WAIT seconds, leaving the quantizing to the tools."""
    import subprocess
    import threading
    q = tmp_path / "QUIET"
    monkeypatch.setenv("OTPU_QUIET", str(q))
    monkeypatch.setenv("OTPU_IMAGE_CACHE", str(tmp_path / "c"))
    cfg = tmp_path / "cfg.pkl"
    cfg.write_bytes(pickle.dumps(0))
    monkeypatch.setattr(PB, "POLL", 0.01)
    monkeypatch.setattr(PB, "mem_gb", lambda: 64.0)
    runs = ["--cfg", str(cfg), "a:fp4:int8", "b:fp4:int8"]

    def later(f, s=0.2):
        threading.Timer(s, f).start()

    _Child.log = []                             # quiet at the start: no build until it goes
    q.write_text(str(os.getpid()))
    monkeypatch.setattr(PB, "_spawn", lambda cmd: (_Child.log.append(("quiet", q.exists())),
                                                   _Child(cmd))[1])
    later(q.unlink)
    assert PB.main(runs) == 0
    assert _Child.log[0] == ("quiet", False) and ("done", "b:fp4:int8") in _Child.log
    assert "waiting while" in capsys.readouterr().out

    _Child.log = []                             # quiet mid-build: stopped, then again after it

    def go_quiet(n):
        if n == 2 and not _Child.log.count(("stop", "a:fp4:int8")):
            q.write_text(str(os.getpid()))
            later(q.unlink)
    monkeypatch.setattr(PB, "_spawn", lambda cmd: _Child(cmd, during=go_quiet))
    assert PB.main(runs) == 0
    assert _Child.log == [("start", "a:fp4:int8"), ("stop", "a:fp4:int8"),
                          ("start", "a:fp4:int8"), ("done", "a:fp4:int8"),
                          ("start", "b:fp4:int8"), ("done", "b:fp4:int8")]
    assert "a:fp4:int8: stopped" in capsys.readouterr().out

    _Child.log = []                             # a quiet file whose session is gone: ignored
    p = subprocess.Popen(["true"])
    p.wait()
    q.write_text(str(p.pid))
    monkeypatch.setattr(PB, "_spawn", lambda cmd: _Child(cmd))
    assert PB.main(runs) == 0 and len(_Child.log) == 4

    _Child.log = []                             # a quiet session that stays: prebuild gives up
    q.write_text(str(os.getpid()))
    monkeypatch.setenv("OTPU_PREBUILD_QUIET_WAIT", "0.05")
    assert PB.main(runs) == 0 and _Child.log == []
    assert "the tools quantize the rest under the lock" in capsys.readouterr().out
