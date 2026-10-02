"""The program cache (opentpu/progcache.py): an engine with prog_cache compiles a bucket's programs
once per process and image layout, and a later process reads them from the disk cache; the programs
read back run as the compiled ones."""
import numpy as np
import pytest

from opentpu import isa as I
from opentpu import progcache as PC
from opentpu.compiler import RunVar

from test_autodecode import _tiny


def _prog(n, seed):
    r = np.random.default_rng(seed)
    return [I.li(int(r.integers(1, 16)), int(r.integers(0, 1 << 20))) for _ in range(n)] + \
        [I.halt()]


def _words(progs):
    if isinstance(progs, np.ndarray):
        return [progs.tolist()]
    if isinstance(progs, tuple):
        return [w for p in progs for w in _words(p)]
    return [I.assemble(p).tolist() for p in progs]


@pytest.mark.parametrize("value", [
    ([_prog(5, 1)], None),                                  # one slice
    ([_prog(5, 1), _prog(7, 2)], None),                     # two slices
    (([_prog(4, 3)], [_prog(6, 4)]), None),                 # a split program's two parts
    (np.asarray(I.assemble(_prog(9, 5)), np.uint32),        # a compile worker's words and the
     [(RunVar("tok"), 4096), (RunVar("tpos"), -4)])],       # run arguments
    ids=["slice", "slices", "split", "words"])
def test_disk_round_trip(tmp_path, value):
    f = tmp_path / "ab" / "ab.npz"
    PC._store(tmp_path, f, value)
    progs, ra = PC._load(f)                 # a worker's words come back as a slice's list
    assert type(progs) is (list if isinstance(value[0], np.ndarray) else type(value[0]))
    assert _words(progs) == _words(value[0])
    assert (ra is None) == (value[1] is None)
    if ra is not None:
        assert [(v.name, c) for v, c in ra] == [(v.name, c) for v, c in value[1]]


def test_the_key_sees_the_layout_the_environment_and_what(monkeypatch):
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    k = PC.key(("spec", 512), ("gen", 1))
    assert k == PC.key(("spec", 512), ("gen", 1))
    assert k != PC.key(("spec", 1024), ("gen", 1)) != PC.key(("spec", 512), ("gen", 2))
    monkeypatch.setenv("OTPU_FORMATS", "attn=fp4")
    assert PC.key(("spec", 512), ("gen", 1)) != k
    monkeypatch.setenv("OTPU_QUIET", "/tmp/q")          # read by the host's tools only
    assert PC.key(("spec", 512), ("gen", 1)) == PC.key(("spec", 512), ("gen", 1))


def test_engines_share_the_bucket_programs(tmp_path, monkeypatch):
    """A second engine of the layout in this process compiles none of the generate loop's and
    resident decode's bucket programs; after clear() (a new process) one reads them from the
    disk cache; each gives the tokens of an engine without the cache."""
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path))
    PC.clear()
    W, spec = _tiny("qwen3")
    cfg = device_config(spec, 512, rows=PREFILL_ROWS, lookup=True)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 248)]

    def run(cache):
        e = Engine(spec, W, cap=512, cfg=cfg, resident=True, prog_cache=cache)
        t0 = int(np.argmax(e.prefill(toks)))
        got = e.generate_card(t0, 12, stop_ids=[])      # buckets 1 and 2
        return got + [int(np.argmax(e.step(got[-1])))]  # resident decode, bucket 2

    want = run(False)
    s0 = dict(PC.stats)
    assert run(True) == want
    assert PC.stats["compile"] - s0["compile"] == 3     # gen 1, gen 2, decode 2
    assert PC.stats["write"] - s0["write"] == 3
    s1 = dict(PC.stats)
    assert run(True) == want
    assert PC.stats["compile"] == s1["compile"] and PC.stats["memory"] - s1["memory"] == 3
    PC.clear()
    s2 = dict(PC.stats)
    assert run(True) == want
    assert PC.stats["compile"] == s2["compile"] and PC.stats["disk"] - s2["disk"] == 3
    PC.clear()


def test_filling_and_plain_decodes_are_cached_apart(tmp_path, monkeypatch):
    """A streamed decode's programs fill their logits (qwen3.fill_logits), the others not: the
    cache keeps the two apart for one layout and bucket (a card engine and its ISA reference in
    one process), and a filling program read back from the disk cache (no comments) gives the
    fill's gate as compiled."""
    from opentpu.llm.qwen3 import Engine, device_config, fill_gate
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path))
    PC.clear()
    W, spec = _tiny("qwen3")
    cfg = device_config(spec, 256, lookup=True)
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 3)]

    def engine(fill):
        e = Engine(spec, W, cap=256, cfg=cfg, resident=True, prog_cache=True)
        e.image.stream_fill = fill
        return e

    plain, filling = engine(False), engine(True)
    s0 = dict(PC.stats)
    for t in toks:
        assert np.array_equal(plain.step(t).view(np.uint32), filling.step(t).view(np.uint32))
    assert PC.stats["compile"] - s0["compile"] == 2
    gate = fill_gate(filling._decode(0)[0][0])
    with pytest.raises(ValueError):
        fill_gate(plain._decode(0)[0][0])
    PC.clear()
    again = engine(True)
    s1 = dict(PC.stats)
    assert np.array_equal(again.step(toks[0]).view(np.uint32),
                          engine(False).step(toks[0]).view(np.uint32))
    assert PC.stats["disk"] - s1["disk"] == 2 and PC.stats["compile"] == s1["compile"]
    assert fill_gate(again._decode(0)[0][0]) == gate
    PC.clear()


def test_the_mtp_loop_programs_come_from_the_cache(tmp_path, monkeypatch):
    """MTP's loop programs (V, E, D, D1 of a bucket) through the cache: a new process's engine
    reads them from disk and gives the tokens of an engine without the cache."""
    import dataclasses

    from test_mtp import _mtp_weights
    from opentpu.isasim import board_config
    from opentpu.llm.mtp import NK, MTPDecoder, mtp_engine
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path))
    PC.clear()
    W, spec = _tiny("qwen35")
    spec = dataclasses.replace(spec, mtp=True)
    W = _mtp_weights(W, spec)
    cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=True, STREAM=True, PAIR=True)
    prompt = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 20)]

    def run(cache):
        eng = mtp_engine(spec, W, cap=512, cfg=cfg, prog_cache=cache)
        return MTPDecoder(eng).generate_card(prompt, max_new=10).tokens

    want = run(False)
    s0 = dict(PC.stats)
    assert run(True) == want
    assert PC.stats["compile"] - s0["compile"] == NK      # bucket 1's
    PC.clear()
    s1 = dict(PC.stats)
    assert run(True) == want
    assert PC.stats["compile"] == s1["compile"] and PC.stats["disk"] - s1["disk"] == NK
    PC.clear()
