"""The program cache (opentpu/progcache.py): an engine with prog_cache compiles a bucket's programs
once per process and image layout, and a later process reads them from the disk cache; the programs
read back run as the compiled ones."""
import os

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
     [(RunVar("tok"), 4096), (RunVar("tpos", 512), -4)])],  # run arguments (one bounded)
    ids=["slice", "slices", "split", "words"])
def test_disk_round_trip(tmp_path, value):
    f = tmp_path / "ab" / "ab.npz"
    PC._store(tmp_path, f, value)
    progs, ra = PC._load(f)                 # a worker's words come back as a slice's list
    assert type(progs) is (list if isinstance(value[0], np.ndarray) else type(value[0]))
    assert _words(progs) == _words(value[0])
    assert (ra is None) == (value[1] is None)
    if ra is not None:
        assert [(v.name, v.bound, c) for v, c in ra] == [(v.name, v.bound, c) for v, c in value[1]]
    # each slice's program carries the arguments (opentpu.waits bounds them by their variables)
    for part in (progs if isinstance(progs, tuple) else (progs,)):
        for p in part:
            assert [(v.name, v.bound, c) for v, c in p.run_args] == \
                [(v.name, v.bound, c) for v, c in (ra or ())]


def test_a_write_under_the_free_disk_floor_is_skipped_and_counted(tmp_path, monkeypatch):
    """Nothing is written that would leave under the floor (FREE_FLOOR, or
    OTPU_PROG_CACHE_FREE_GB over it); the skip is counted, so a host can say its programs are
    not kept."""
    monkeypatch.setattr(PC, "stats", dict.fromkeys(PC.stats, 0))
    f = tmp_path / "ab" / "ab.npz"
    monkeypatch.setenv("OTPU_PROG_CACHE_FREE_GB", str(1 << 30))
    assert PC.free_floor() == (1 << 30) * 2**30
    PC._store(tmp_path, f, ([_prog(5, 1)], None))
    assert not f.exists() and PC.stats["skip"] == 1 and PC.stats["write"] == 0
    monkeypatch.setenv("OTPU_PROG_CACHE_FREE_GB", "0")
    PC._store(tmp_path, f, ([_prog(5, 1)], None))
    assert f.exists() and PC.stats["write"] == 1
    monkeypatch.delenv("OTPU_PROG_CACHE_FREE_GB")
    assert PC.free_floor() == PC.FREE_FLOOR
    assert "OTPU_PROG_CACHE_FREE_GB" in PC.RUNTIME_ENV      # (not in the key)


def test_the_key_sees_the_layout_the_environment_and_what(monkeypatch):
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    k = PC.key(("spec", 512), ("gen", 1))
    assert k == PC.key(("spec", 512), ("gen", 1))
    assert k != PC.key(("spec", 1024), ("gen", 1)) != PC.key(("spec", 512), ("gen", 2))
    monkeypatch.setenv("OTPU_FORMATS", "attn=fp4")
    assert PC.key(("spec", 512), ("gen", 1)) != k
    k = PC.key(("spec", 512), ("gen", 1))
    for var, val in (("OTPU_QUIET", "/tmp/q"), ("OTPU_BUILD_JOBS", "3"),
                     ("OTPU_IMG_CACHE", "/tmp/i"), ("OTPU_LOCK_HELD", "xdma0"),
                     ("OTPU_LOCK_PID", str(os.getpid())), ("OTPU_VISION_WARM", "0"),
                     ("OTPU_DEV", "/dev/xdma1")):    # (which card's node: the second card's too)
        monkeypatch.setenv(var, val)        # read by the host's tools only (otpu-lock sets its
        assert PC.key(("spec", 512), ("gen", 1)) == k, var  # pid: each session's misses;
                                            # vision_warm's switch: its entries, the engine's)


def test_a_schedule_only_field_parts_the_programs_not_the_images(monkeypatch):
    """Two builds of a target that differ only in SE_R (the streams' engine: Config.SE_OWN, a
    DeltaNet burst a pair) compile different programs, so their program keys differ; their
    images are the same bytes, so the image cache keys them as one (imgcache.PROGRAM_ONLY)."""
    from opentpu import imgcache as IC
    from opentpu import target as T
    monkeypatch.delenv("OTPU_SE_R", raising=False)
    on, off = (T.build_config("bcu1525-x2", SE_R=r) for r in (1, 0))
    assert on.SE_OWN and not off.SE_OWN and T.config("bcu1525-x2").SE_OWN
    assert {f: v for f, v in vars(on).items() if v != vars(off)[f]} == {"SE_OWN": True}
    lay = [("spec", c, 512, 1, 1, 256, ()) for c in (on, off)]
    assert PC.key(lay[0], ("gen", 1)) != PC.key(lay[1], ("gen", 1))
    img = type("Image", (), {})()
    assert IC.key(img, lay[0], None) == IC.key(img, lay[1], None)
    assert IC.key(img, lay[0], None) != IC.key(img, ("spec", on, 1024, 1, 1, 256, ()), None)


def test_the_key_sees_the_host_modules_the_programs_take(monkeypatch):
    """The programs take constants from opentpu/host/offload.py (LINE, TAG, RUN, Layout and
    RowLayout: where the mailboxes, slots and tags are): a change there misses, one in a host
    driver does not; every host module the compiler, kernels and models import is in the key."""
    import ast
    from pathlib import Path
    root = Path(PC.__file__).resolve().parent
    k = PC.key(("spec", 512), ("gen", 1))
    read = Path.read_bytes

    def edited(name):
        return lambda f: read(f) + (b"\n# changed\n" if f == root / "host" / name else b"")

    for name, misses in (("offload.py", True), ("board.py", False)):
        monkeypatch.setattr(PC, "_SRC", None)
        monkeypatch.setattr(Path, "read_bytes", edited(name))
        assert (PC.key(("spec", 512), ("gen", 1)) != k) == misses, name
    keyed = {f.relative_to(root).as_posix() for f in PC.keyed()}
    for f in PC.keyed():                        # (opentpu/serve's are not keyed)
        rel = f.relative_to(root).parts
        if rel[0] == "host":
            continue
        for node in ast.parse(f.read_text()).body:          # the module-level imports
            if not isinstance(node, ast.ImportFrom):
                continue
            mod = (node.module or "").split(".")
            if node.level:
                mod = list(rel[:len(rel) - node.level]) + mod
            elif mod[0] == "opentpu":
                mod = mod[1:]
            if mod and mod[0] == "host":
                for m in ([mod] if len(mod) > 1 else [mod + [a.name] for a in node.names]):
                    assert "/".join(m) + ".py" in keyed, (f, m)


def test_the_key_leaves_out_the_tools_no_build_runs(monkeypatch):
    """An edit to a profiler, trace, reference or RTL simulation module (UNKEYED) keeps the
    cached programs and images; one to the compiler misses. No keyed module imports an UNKEYED
    one, at module level or in a function, but runtime's RTL run (rtlsim: a run, not a build)
    and the ISA simulator's DRAM above 4 GiB (simtmp.dram_file: where its sparse file goes)."""
    import ast
    from pathlib import Path
    root = Path(PC.__file__).resolve().parent
    k = PC.key(("spec", 512), ("gen", 1))
    read = Path.read_bytes

    def edited(name):
        return lambda f: read(f) + (b"\n# changed\n" if f == root / name else b"")

    for name, misses in (("lens.py", False), ("rtlsim.py", False), ("compiler.py", True)):
        monkeypatch.setattr(PC, "_SRC", None)
        monkeypatch.setattr(Path, "read_bytes", edited(name))
        assert (PC.key(("spec", 512), ("gen", 1)) != k) == misses, name
    monkeypatch.undo()
    unkeyed = {u[:-3].replace("/", ".") for u in PC.UNKEYED}
    allowed = {("runtime.py", "rtlsim"), ("isasim.py", "simtmp")}
    for f in PC.keyed():
        rel = f.relative_to(root)
        pkg = list(rel.parts[:-1])
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[1:] for a in node.names if a.name.startswith("opentpu.")]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module.split(".") if node.module else []
                if node.level:
                    mod = pkg[:len(pkg) - node.level + 1] + mod
                elif mod[:1] == ["opentpu"]:
                    mod = mod[1:]
                else:
                    continue
                mods = [mod] + [mod + [a.name] for a in node.names]
            else:
                continue
            for m in mods:
                if ".".join(m) in unkeyed:
                    assert (rel.as_posix(), ".".join(m)) in allowed, (rel.as_posix(), m)


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


def test_the_generate_buckets_warmed_ahead_are_read_back(tmp_path, monkeypatch):
    """opentpu/host/gen_warm.py: its process (spawned, the engine's arguments pickled) compiles
    every bucket's generate programs, greedy and the default sampler's, into the disk cache
    under the engine's layout; a new process's engine then compiles none of them for its
    replies in buckets 1 and 2 (read back from the disk) and gives the tokens of an engine
    without the cache. With wait entries (warm's waitc), the board's computation of a read-back
    program's entries finds them on the disk. A second warm compiles nothing; start() leaves an
    engine without the program cache alone."""
    from types import SimpleNamespace

    from opentpu import waits as WC
    from opentpu.host import gen_warm as GW
    from opentpu.llm import generate as G
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path))
    PC.clear()
    W, spec = _tiny("qwen3")
    cfg = device_config(spec, 512, rows=PREFILL_ROWS, lookup=True)
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 248)]
    sp = dict(temperature=0.7, top_k=20, top_p=0.9, repetition_penalty=1.1)

    def run(e, samp=None):
        t0 = int(np.argmax(e.prefill(toks)))
        return e.generate_card(t0, 12, stop_ids=[], sampling=samp, context=toks + [t0],
                               rng=np.random.default_rng(5))     # buckets 1 and 2

    want = [run(Engine(spec, W, cap=512, cfg=cfg, resident=True, prog_cache=False), s)
            for s in GW.samplings(sp)]
    e = Engine(spec, W, cap=512, cfg=cfg, resident=True, prog_cache=True)
    assert GW.start(SimpleNamespace(can_generate=True, prog_cache=False)) is None
    assert [None if s is None else s.key for s in GW.samplings(sp)] == [None, (24, True)]
    p = GW.start(e, sp)
    p.join(300)
    assert p.exitcode == 0
    chain = bool(getattr(e.backend, "chains", False))
    for b in range(1, G.buckets(512, e.block) + 1):
        for s in GW.samplings(sp):
            assert not GW.warm(e.image, e.layout, e.block, e._conv_lo, b, chain, s,
                               e.gen_debug, e.gen_split)
    PC.clear()
    s0 = dict(PC.stats)
    got = [run(Engine(spec, W, cap=512, cfg=cfg, resident=True, prog_cache=True), s)
           for s in GW.samplings(sp)]
    assert got == want
    assert PC.stats["compile"] == s0["compile"] and PC.stats["disk"] - s0["disk"] == 4
    PC.clear()
    GW.warm(e.image, e.layout, e.block, e._conv_lo, 1, chain, waitc=True)
    WC._mem.clear()
    c0 = WC.stats["computed"]
    progs = PC.get(e.layout, GW.gen_what(e.block, e._conv_lo, 1, chain, None, e.gen_debug,
                                         e.gen_split), None)[0]
    for part in progs if isinstance(progs, tuple) else (progs,):
        for prog in part:
            WC.entries(I.assemble(prog), cfg.D, cfg.S, prog=prog)
    assert WC.stats["computed"] == c0
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
