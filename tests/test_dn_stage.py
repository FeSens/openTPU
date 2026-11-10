"""The merged stage (opentpu/stage.py, Config.DNMERGE): a chunked DeltaNet layer's prep and post tiles
interleaved in one program, against the same chunks tile by tile: logits and decode steps bit for
bit."""
import dataclasses

import numpy as np
import pytest

from opentpu.llm.qwen3 import Engine


def _engines(monkeypatch, S, shape, N, chunk=24, target="bcu1525-wg", tile_rows=8, dnchunk=8,
             dram=1 << 30, **cfgkw):
    import test_moe_s2 as TM
    from opentpu import target as T
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", chunk)
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    W, spec = TM._tiny(("linear", "attn", "linear"), **shape)
    cfg = T.config(target, S=S, DRAM_BYTES=dram, **cfgkw)
    kw = dict(cap=256, rows=1, resident=True, layer_major=2, expert_major=True, tile_rows=tile_rows)
    a = Engine(spec, W, cfg=dataclasses.replace(cfg, DNCHUNK=dnchunk, DNMERGE=N), **kw)
    b = Engine(spec, W, cfg=dataclasses.replace(cfg, DNCHUNK=dnchunk), **kw)
    return a, b


@pytest.mark.parametrize("S,shape,N", [
    (1, dict(nk=4, nv=8), 2),
    (1, dict(nk=4, nv=8), 4),
    (2, dict(nk=2, nv=4), 2),
    (4, dict(nk=4, nv=8, H=512, F=512), 2),
])
def test_merged_stage_is_the_tiles_bits(monkeypatch, S, shape, N):
    from opentpu.llm import tiled as TL
    monkeypatch.setattr(TL, "STAGE_TILES", 2)
    a, b = _engines(monkeypatch, S, shape, N)
    toks = [int(t) for t in np.random.default_rng(15).integers(0, 1000, 43)]
    la, lb = a.prefill(toks), b.prefill(toks)
    keys = [k for k in a._layer_runs if isinstance(k, tuple)]
    assert any(":" in str(k[-1]) and k[-1].startswith("dn-prep") for k in keys), keys
    assert any(":" in str(k[-1]) and k[-1].startswith("dn-post") for k in keys), keys
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    t = int(np.argmax(la))
    for _ in range(2):
        ga, gb = a.step(t), b.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))


@pytest.mark.parametrize("N", [2, 3])
def test_merged_stage_on_b2(monkeypatch, N):
    """B2's arch at four slices, 16 rows a tile, the chunked core (the 35B's form, its slice's head
    shape), three tiles a merged run: the attention blocks' rows cut a chunk's tiles here."""
    from opentpu.llm import tiled as TL
    monkeypatch.setattr(TL, "STAGE_TILES", 3)
    a, b = _engines(monkeypatch, 4, dict(nk=4, nv=8, H=512, F=512), N, chunk=128,
                    target="bcu1525-x2:b2", tile_rows=16, dnchunk=64, BHW=0)
    toks = [int(t) for t in np.random.default_rng(16).integers(0, 1000, 200)]
    la, lb = a.prefill(toks), b.prefill(toks)
    keys = [k for k in a._layer_runs if isinstance(k, tuple)]
    assert any(k[-1] == "dn-prep:3" for k in keys) and any(k[-1] == "dn-post:3" for k in keys), keys
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    t = int(np.argmax(la))
    ga, gb = a.step(t), b.step(t)
    assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))


@pytest.mark.parametrize("S", [1, 4])
def test_merged_stage_rtl(have_verilator, monkeypatch, S):
    """A merged prep run and a merged post run on the RTL at x2:b2's micro-architecture, from the
    DRAM the prefill had before them: the wait entries over two streams' instructions, with each
    stream's own part of TMEM. DRAM and TMEM the ISA's bit for bit."""
    from opentpu import isasim as IS, rtlsim, target as T
    from opentpu.llm import tiled as TL
    monkeypatch.setattr(TL, "STAGE_TILES", 3)
    a, _ = _engines(monkeypatch, S, dict(nk=4, nv=8, H=512, F=512), 2, chunk=128,
                    target="bcu1525-x2:b2", tile_rows=16, dnchunk=64, BHW=0, dram=1 << 28)
    got = {}
    run0 = a.backend.run

    def run(progs, args=None, **kw):
        for k, v in a._layer_runs.items():
            if isinstance(k, tuple) and v[0] is progs and ":" in str(k[-1]):
                got.setdefault(k[-1].split(":")[0],
                               (progs, args, [x.dram.copy() for x in a.backend.machine.slices]))
        return run0(progs, args, **kw)
    a.backend.run = run
    a.prefill([int(t) for t in np.random.default_rng(16).integers(0, 1000, 200)])
    assert set(got) == {"dn-prep", "dn-post"}, set(got)
    ua = {k: v for k, v in T.sim_uarch("bcu1525-x2:b2").items()
          if not k.startswith(("COLL_", "NAT_")) and k != "XRS"}
    for part, (progs, args, drams) in got.items():
        m = IS.Machine(a.cfg, progs, [d.copy() for d in drams], args).run()
        rd, rt, _ = rtlsim.run(a.cfg, progs, drams, uarch=ua, args=args)
        for s in range(S):
            for what, x, y in (("DRAM bytes", rd[s], m.slices[s].dram),
                               ("TMEM words", rt[s], m.slices[s].tmem)):
                bad = np.nonzero(x != y)[0]
                assert len(bad) == 0, f"{part} slice {s}: {len(bad)} {what} differ, first at {bad[:6]}"
