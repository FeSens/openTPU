"""The layer loop (opentpu/llm/prefill_loop.py): a layer's runs of a prefill chunk in one bucket as
one device run (a hardware loop over the runs, their arguments from the loop's state words)
give the per-run layer-major prefill's logits and DRAM bit for bit, with fewer runs."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from opentpu.llm import prefill_loop as PL  # noqa: E402
from opentpu.llm.qwen3 import Engine, device_config  # noqa: E402

from test_qwen35_moe import E, _tiny, _untied  # noqa: E402


def test_loop_runs_merges_a_layers_runs_in_a_bucket():
    """Consecutive runs of n rows of one layer in one bucket, n rows apart, become one ("L", ...)
    run (tiled runs too), a lone one a loop of one; the host's rows (the embed runs' without
    embed), a compile-time run, an expert run and the head stay as they are."""
    runs = [((0, 1, 2, False), {"tpos": 3, "row": 0}, [5, 6]),
            ((1, 1, 2, True), {"tpos": 3, "row": 0}, None),
            ((1, 1, 2, True), {"tpos": 5, "row": 2}, None),
            ((1, 1, 2, True), {"tpos": 7, "row": 4}, None),
            ((1, 1, 1, True), {"tpos": 9, "row": 6}, None),
            ((1, 2, 2, True), {"tpos": 0, "row": 7}, None),
            ((2, None, 2, True, 0), {"row": 0}, None),
            ((2, 1, 8, True, "tile"), {"tpos": 3, "row": 0}, None),
            ((2, 1, 8, True, "tile"), {"tpos": 11, "row": 8}, None),
            (("x", 2), {"rows": 16, "entries": 64}, None),
            ("head", {"row": 8}, None)]
    out = PL.loop_runs(runs)
    one = {"runs": 1, "tpos": 0, "row": 0}
    assert out == [runs[0],
                   (("L", 1, 1, 2), {"tpos0": 3, "row0": 0, "runs": 3, "tpos": 0, "row": 0},
                    None),
                   (("L", 1, 1, 1), {"tpos0": 9, "row0": 6, **one}, None),
                   (("L", 1, 2, 2), {"tpos0": 0, "row0": 7, **one}, None), runs[6],
                   (("L", 2, 1, 8, "tile"), {"tpos0": 3, "row0": 0, "runs": 2, "tpos": 0,
                                             "row": 0}, None), runs[9], runs[10]]
    emb = [((-1, 1, 1, True), {"tpos": 254 + i, "tok": 7 + i, "ring": 0, "row": i}, [7 + i])
           for i in range(2)] + \
          [((-1, 2, 1, True), {"tpos": i, "tok": 9 + i, "ring": 0, "row": 2 + i}, [9 + i])
           for i in range(3)]
    assert PL.loop_runs(emb) == emb             # (the host's rows: not without embed)
    assert PL.loop_runs(emb, embed=True) == [
        (("L", -1, 1, 1), {"tpos0": 254, "row0": 0, "runs": 2, "tpos": 0, "row": 0}, [7, 8]),
        (("L", -1, 2, 1), {"tpos0": 0, "row0": 2, "runs": 3, "tpos": 0, "row": 0}, [9, 10, 11])]


@pytest.mark.parametrize("R", [2, 4])
def test_layer_loops_are_the_runs_bit_for_bit(R, monkeypatch):
    """A 262-token prompt in chunks of 100 rows (the last crossing an attention block), the 35B's
    DeltaNet layout (pairs sharing q and k, group-major), fp4, every expert resident: with
    layer_loop each layer's runs in a bucket are one run; the logits, every slice's DRAM and
    the next steps equal the per-run path's."""
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 100)
    m, W, spec = _tiny(nk=4)
    spec, W = _untied((m, W, spec))
    spec = replace(spec, pair_loop=True)
    cfg = device_config(spec, 512, rows=1, lookup=True, S=1, experts=E, wformat="fp4", MCOLS=4,
                        PAIR=True, DSTEP=True, STREAM=True)
    a, ref = (Engine(spec, W, cap=512, cfg=cfg, rows=1, resident=True, experts=E,
                     wformat="fp4", layer_major=R, layer_loop=loop) for loop in (True, False))
    toks = [int(t) for t in np.random.default_rng(7).integers(0, 1000, 262)]
    la, lb = a.prefill(toks), ref.prefill(toks)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    loops = [k for k in a._layer_runs if isinstance(k, tuple) and k[0] == "L"]
    assert any(k[3] == R for k in loops) and all(k[3] <= R for k in loops)
    assert len(a.stats) <= len(ref.stats) / 4, (len(a.stats), len(ref.stats))
    assert all(np.array_equal(x.dram, y.dram)
               for x, y in zip(a.backend.machine.slices, ref.backend.machine.slices))
    t = int(np.argmax(la))
    for _ in range(3):
        ga, gb = a.step(t), ref.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))


def test_tiled_expert_major_loops_are_the_runs_bit_for_bit(monkeypatch):
    """Expert-major's path (a resident pool, Image(em=True), the mixer in tiles of 8 rows,
    an expert run a layer): a 270-token prompt in 60-row chunks (the fifth crossing a block)
    with layer_loop, each layer's tiles in a bucket one run and the embed runs one (their tokens
    from out[]), against the same engine without: the logits, DRAM and the steps after bit for
    bit, under a quarter of the runs."""
    import dataclasses
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 60)
    monkeypatch.setenv("OTPU_POOL", "1")
    spec, W = _untied(_tiny(4))
    cfg = device_config(spec, 512, rows=1, lookup=True, S=1, wformat="fp4", MCOLS=4, PAIR=True,
                        DSTEP=True, STREAM=True)
    kw = dict(cap=512, cfg=dataclasses.replace(cfg, ACT_ROWS=16), rows=1, resident=True,
              wformat="fp4", layer_major=2, expert_major=True, tile_rows=8)
    a, ref = Engine(spec, W, layer_loop=True, **kw), Engine(spec, W, **kw)
    assert a.layer_tile and a.server is None
    toks = [int(t) for t in np.random.default_rng(12).integers(0, 1000, 270)]
    la, lb = a.prefill(toks), ref.prefill(toks)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    loops = [k for k in a._layer_runs if isinstance(k, tuple) and k[0] == "L"]
    assert any(k[1] == -1 for k in loops)
    assert any(k[-1] == "tile" and k[3] == 8 for k in loops)
    assert len(a.stats) * 4 < len(ref.stats), (len(a.stats), len(ref.stats))
    o = a.image.lookup["gen"]["out"]                # (DRAM the same but out[]: the tokens)
    x, y = (np.array(e.backend.machine.slices[0].dram) for e in (a, ref))
    assert np.array_equal(x[o:o + 4 * len(toks)].view(np.float32), np.float32(toks))
    x[o:o + 4 * len(toks)] = y[o:o + 4 * len(toks)]
    assert np.array_equal(x, y)
    t = int(np.argmax(la))
    for _ in range(3):
        ga, gb = a.step(t), ref.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))



@pytest.mark.parametrize("S,kinds,shape", [
    (2, ("linear", "attn"), {}),
    (4, ("linear", "attn"), dict(nv=8, H=512, F=512)),
])
def test_tensor_parallel_tiled_loops_are_the_runs_bit_for_bit(monkeypatch, S, kinds, shape):
    """Items 2 + 3 at S slices (test_em_resident's tensor-parallel expert-major engine: the
    layer tiles' heads gathered, the expert run's passes in _moe_ffn_tp's order): with
    layer_loop each slice's tiles in a bucket one loop (a program per slice), against the same
    engine without: the logits, the steps after and every slice's DRAM but out[] bit for bit."""
    from test_moe_s2 import _engine, _tiny as _tiny_s2
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    W, spec = _tiny_s2(kinds, **shape)
    kw = dict(rows=1, resident=True, layer_major=2, expert_major=True, tile_rows=8)
    a = _engine(spec, W, S, monkeypatch, layer_loop=True, **kw)
    ref = _engine(spec, W, S, monkeypatch, **kw)
    assert a.image.fmt.S == S and a.layer_tile and a.server is None
    toks = [int(t) for t in np.random.default_rng(13).integers(0, 1000, 43)]
    la, lb = a.prefill(toks), ref.prefill(toks)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    loops = [k for k in a._layer_runs if isinstance(k, tuple) and k[0] == "L"]
    assert any(k[-1] == "tile" and k[3] == 8 for k in loops), loops
    assert len(a.stats) * 2 < len(ref.stats), (len(a.stats), len(ref.stats))
    o = a.image.lookup["gen"]["out"]
    for s in range(S):
        x, y = (np.array(e.backend.machine.slices[s].dram) for e in (a, ref))
        if any(k[1] == -1 for k in loops):          # (out[]: the embed loop's tokens)
            assert np.array_equal(x[o:o + 4 * len(toks)].view(np.float32), np.float32(toks))
            x[o:o + 4 * len(toks)] = y[o:o + 4 * len(toks)]
        assert np.array_equal(x, y), s
    t = int(np.argmax(la))
    for _ in range(2):
        ga, gb = a.step(t), ref.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))


def test_slot_prompts_loop_bit_for_bit(monkeypatch):
    """batch35's prompts into sequence slots (prefill_slot: prefill_layers at the slot, otpu-serve35
    --slots N) with layer_loop: each slot's logits bit for bit as a single-sequence engine's runs
    one by one, slot 0's context untouched by slot 1's prompt (the steps after: each its own)."""
    from opentpu.llm import batch35 as B35
    from test_batch35 import _cfg, _tiny as _tiny_b
    W, spec = _tiny_b()
    cfg = _cfg(monkeypatch)
    rng = np.random.default_rng(7)
    prompts = [[int(t) for t in rng.integers(0, 1000, n)] for n in (37, 23)]
    eng = Engine(spec, W, cap=256, cfg=cfg, batch=2, resident=True, layer_major=2,
                 batch_area=True, layer_loop=True)
    refs = [Engine(spec, W, cap=256, cfg=cfg, resident=True, layer_major=2) for _ in prompts]
    nxt = {}
    for s, (pr, ref) in enumerate(zip(prompts, refs)):
        got, want = B35.prefill_slot(eng, pr, s), ref.prefill(pr)
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), s
        nxt[s] = int(np.argmax(want))
    assert any(isinstance(k, tuple) and k[0] == "L" for k in eng._layer_runs)
    assert any(isinstance(k, tuple) and k[1:] == ("slot", 1) and k[0][0] == "L"
               for k in eng._layer_runs)
    st = B35.Steps(eng)
    for s in (0, 1):
        st.set_sampling(s, None)
    for _ in range(2):
        out = st.run([(s, eng.poss[s]) for s in (0, 1)], [nxt[s] for s in (0, 1)])
        for r, s in enumerate((0, 1)):
            want = int(np.argmax(refs[s].step(nxt[s])))
            assert out[r] == want, s
            nxt[s] = want

def test_layer_loops_on_two_cards(monkeypatch):
    """x2's S = 4 (bcu1525-x2:st16, the tiny 35B shape of test_chat_x2): a 70-token prompt layer
    by layer (R 2) with layer_loop through CardsTransport (each slice's loop its part of the
    tensor-parallel runs) against the ISA simulator's S = 4 runs one by one: the logits and the
    next step bit for bit, every slice's DRAM the same but out[]."""
    from opentpu.host.board import BoardBackend
    from opentpu.host.cards import CardsTransport, IsaCards
    from opentpu import target as T
    from test_moe_s2 import _tiny as _tiny_s2
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    W, spec = _tiny_s2(("linear", "attn"), nk=8, nv=16, H=512, F=512)
    cfg = T.config(T.load("bcu1525-x2:st16"), S=4, DRAM_BYTES=1 << 26, BHW=0)
    kw = dict(cap=256, cfg=cfg, wformat="fp4", rows=1, resident=True, layer_major=2)
    sh = IsaCards(cfg)
    a = Engine(spec, W, layer_loop=True, **kw,
               backend=lambda c, imgs: BoardBackend(c, imgs, transport=CardsTransport(sh.cards),
                                                    status=False))
    ref = Engine(spec, W, **kw)
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 70)]
    la, lb = (list(e.prefill_chunks(toks, chunk=None))[-1][1] for e in (a, ref))
    assert np.array_equal(np.asarray(la, np.float32).view(np.uint32),
                          np.asarray(lb, np.float32).view(np.uint32))
    assert any(isinstance(k, tuple) and k[0] == "L" and k[1] == -1 for k in a._layer_runs)
    assert a.stats[-1]["runs"] * 8 < ref.stats[-1]["runs"], (a.stats, ref.stats)
    o, n = a.image.lookup["gen"]["out"], a.image.nbytes
    for s in range(4):
        x, y = np.array(sh.dram[s][:n]), np.array(ref.backend.machine.slices[s].dram[:n])
        assert np.array_equal(x[o:o + 4 * len(toks)].view(np.float32), np.float32(toks)), s
        x[o:o + 4 * len(toks)] = y[o:o + 4 * len(toks)]
        assert np.array_equal(x, y), s
    t = int(np.argmax(la))
    assert np.array_equal(a.step(t).view(np.uint32), ref.step(t).view(np.uint32))


@pytest.mark.parametrize("S", [1, 4])
def test_warmed_layer_loops_compile_nothing(S, monkeypatch, tmp_path):
    """warm35x2.warm_layers(loop=True) (warm35.py / warm35x2.py with LB and CHAT35_LOOP=1) puts
    the loops a prompt takes into the program cache: after it, into an empty cache, a prompt
    layer by layer with layer_loop in bucket 1 compiles none, every layer's loop from the disk
    (S = 1: the engine's compile workers read the warmed loops from it, the same words as a
    compile, and compile only the others; S = 4: x2's cards)."""
    import importlib.util
    from pathlib import Path
    from opentpu import progcache as PC
    from opentpu import target as T
    from opentpu.host import gen_warm as GW
    from opentpu.host.board import BoardBackend
    from opentpu.host.cards import CardsTransport, IsaCards
    from test_moe_s2 import _tiny as _tiny_s2
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path / "qprog"))
    d = Path(__file__).resolve().parents[1] / "boards/bcu1525/chat35"
    monkeypatch.syspath_prepend(str(d))
    spec_ = importlib.util.spec_from_file_location("warm35x2", d / "warm35x2.py")
    WX = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(WX)
    W, spec = _tiny_s2(("linear", "attn"), nk=8, nv=16, H=512, F=512)
    cfg = T.config(T.load("bcu1525-x2:st16"), S=S, DRAM_BYTES=1 << 26, BHW=0)
    kw = dict(cap=256, cfg=cfg, wformat="fp4", rows=1, resident=True, layer_major=2,
              layer_loop=True)
    if S > 1:
        sh = IsaCards(cfg)
        kw["backend"] = lambda c, imgs: BoardBackend(c, imgs, transport=CardsTransport(sh.cards),
                                                     status=False)
    eng = Engine(spec, W, **kw)
    assert eng.prog_cache
    be = eng.backend
    args = (eng.spec, eng.cfg, eng.cap, eng.batch, eng.rows, eng._image_kw,
            eng.image.stream_fill, eng.layout, eng.block, eng._conv_lo,
            bool(getattr(be, "chains", False)), eng.gen_debug, eng.gen_split,
            bool(getattr(be, "waitc", False)), GW.samplings(None))
    PC.clear()
    WX.warm_layers(args, [1], 2, loop=True)
    PC.clear()
    if S == 1:                  # (the compile workers, here in this process)
        from concurrent.futures import Future
        from opentpu.llm import qwen3 as Q3
        made, compile = [], Q3.layer_programs

        def counted(image, key, *a):
            made.append(key)
            return compile(image, key, *a)

        class Pool:
            def submit(self, fn, *a):
                f = Future()
                f.set_result(fn(*a))
                return f
        monkeypatch.setattr(Q3, "layer_programs", counted)
        monkeypatch.setattr(Q3, "_WORKER", (eng.image, eng.block))
        procs, pool = eng._procs, eng._pool
        eng._procs, eng._pool = True, Pool()
        keys = WX.layer_keys(spec, 1, 2, loop=True)
        eng._precompile_layers(keys + [("L", 0, 1, 3)])
        assert made == [("L", 0, 1, 3)], made       # (R 3's: not warmed)
        k = keys[3]
        w, ra = eng._layer_next[k].result()         # (the cache's: a compile's words)
        assert np.array_equal(w, Q3.worker_words(eng.image, compile(eng.image, k, eng.block)[0][0]))
        assert [(v.name, c) for v, c in ra] == [(v.name, c) for v, c in w.run_args]
        eng._procs, eng._pool, eng._layer_next = procs, pool, {}
        PC.clear()
    before, disk = PC.stats["compile"], PC.stats["disk"]
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 121)]
    la = list(eng.prefill_chunks(toks))[-1][1]
    assert PC.stats["compile"] == before and PC.stats["disk"] >= disk + spec.layers, PC.stats
    assert any(isinstance(k, tuple) and k[0] == "L" for k in eng._layer_runs)
    if S == 1:                  # (the compile workers in their processes: the same logits)
        from opentpu import isa as I
        from opentpu.llm.qwen3 import IsaBackend

        class Words(IsaBackend):                # (test_qwen35_moe's: the board's words)
            runs_words = True

            def run(self, programs, args=None):
                if isinstance(programs, np.ndarray):
                    programs = [[I.Instr.decode(programs[i:i + 8])
                                 for i in range(0, len(programs), 8)]]
                return super().run(programs, args)
        b = Engine(spec, W, **kw, backend=lambda c, imgs: Words(c, imgs, adopt=True))
        try:
            assert b._procs and b._ready.result(timeout=120) and b.prog_cache
            b._precompile_layers(keys)
            lb = list(b.prefill_chunks(toks))[-1][1]
            assert np.array_equal(np.asarray(la, np.float32).view(np.uint32),
                                  np.asarray(lb, np.float32).view(np.uint32))
            assert b._layer_runs and all(p is None for p, _, _ in b._layer_runs.values())
        finally:
            b._pool.shutdown()
