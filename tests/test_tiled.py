"""Tiled prefill (opentpu/llm/tiled.py, docs/tiled_prefill.md): a run of T rows, every weight
chunk streamed once for all of them and the attention's rows packed MCOLS / G to an MM, against
today's prefill runs over the same tokens on the ISA simulator (the MX MXU: an MM's output one
exact integer, PAIR's too): the KV cache, every layer block and the last row's logits bit for
bit, each operand quantized over today's columns (one QACT: W_o's whole, a down chunk's). Tiled
prompt runs (prefill.py's tile mode) of every dense family against today's prompt runs: Qwen3,
the Llama-likes (SmolLM3, Phi-4-mini's shape), LFM2, Qwen3.5, Gemma 4; expert-major layer runs
of the MoE ones. The range forms (W_o's operand a QACT range a part: not today's bits) by the
logits' KL divergence from today's, and RTL = ISA."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

from opentpu import isa as I
from opentpu.isasim import Machine, board_config
from opentpu.llm import prefill as PF
from opentpu.llm import tiled
from opentpu.llm.qwen3 import Engine, Spec, rope_tables

sys.path.insert(0, str(Path(__file__).parent))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    hc = transformers.Qwen3Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=128, intermediate_size=512,
                                  vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
                                  tie_word_embeddings=True, max_position_embeddings=4096)
    m = transformers.Qwen3ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    return m, W, Spec(256, 2, 4, 2, 128, 512, 1000)


def _inputs(eng, toks, p0, n):
    """The I/O area's rows for tokens toks at positions p0 ..: (address, values) pairs."""
    io = eng.image.io
    x = eng.embed[toks]
    cs = [np.stack(a).astype(np.float32)
          for a in zip(*[rope_tables(eng.spec, p0 + r) for r in range(n)])]
    return [(io["x"], x), (io["cos"], cs[0]), (io["sin"], cs[1])]


def _put(dram, parts):
    for addr, v in parts:
        b = np.ascontiguousarray(v, np.float32).view(np.uint8).reshape(-1)
        dram[addr:addr + b.size] = b


def _logits(eng, dram, row):
    v, a = eng.spec.vocab, eng.image.io["logits"]
    return dram[a + 4 * row * v:a + 4 * (row + 1) * v].view(np.float32).copy()


@pytest.mark.parametrize("mcols,wformat,p0,T", [
    (4, "int8", 5, 16),         # 2 rows a packed tile (G = 2), one weight pass for 16 rows
    (4, "fp4", 247, 16),        # 4-bit layers; the tile crosses an attention block (256)
    (2, "int8", 9, 6),          # one row a tile (MCOLS = G), T not a multiple of MCOLS
    (8, "fp4", 120, 14),        # 4 rows a tile, a partial last tile; today's runs of 8 and
])                              # 6 rows (a run of <= MCOLS / 2 4-bit rows would PAIR)
def test_tile_is_todays_runs_bit_for_bit(tiny, mcols, wformat, p0, T):
    """A tile of T rows equals today's prefill runs (compile_rows, MCOLS rows each, its MMs one
    pass of MCOLS rows) over the same tokens: every layer block (weights and KV cache) and the
    last row's logits, bit for bit. The tile's MMs take all T rows (one weight pass, replayed
    groups), its attention P rows a tile with per-row masks."""
    _, W, spec = tiny
    wkw = dict(wformat=wformat, head_format="int8") if wformat != "int8" else {}
    act = max(16, mcols)
    cfg = board_config(MCOLS=mcols, ACT_ROWS=act, DRAM_BYTES=1 << 24,
                       PAIR=wformat != "int8")
    toks = [int(t) for t in np.random.default_rng(p0).integers(0, 1000, p0 + T)]
    eng = Engine(spec, W, cap=512, cfg=cfg, rows=T, **wkw)
    eng.prefill(toks[:p0])                      # the cache the rows attend over
    base = eng.backend.machine.slices[0].dram.copy()
    img = eng.image
    lo, hi = img.layer0, img.head[0]

    tile = tiled.compile_tile(img, p0, T, logit_rows=[T - 1])
    mm = [i for i in tile[0] if i.op == I.MM]
    assert max((i.w[5] >> 16) & 0xFF for i in mm) == T     # the projections: one MM, T rows
    a = base.copy()
    _put(a, _inputs(eng, toks[p0:], p0, T))
    a = Machine(cfg, tile, [a]).run().slices[0].dram

    b = base.copy()                             # today's runs of MCOLS rows
    R = mcols
    for r0 in range(0, T, R):
        n = min(R, T - r0)
        rows = [(0, p0 + r0 + r) for r in range(n)]
        progs = img.compile_rows(rows, [n - 1] if r0 + n == T else [])
        _put(b, _inputs(eng, toks[p0 + r0:p0 + r0 + n], p0 + r0, n))
        b = Machine(cfg, progs, [b]).run().slices[0].dram
        last = n - 1
    assert np.array_equal(a[lo:hi], b[lo:hi])
    la, lb = _logits(eng, a, T - 1), _logits(eng, b, last)
    assert np.isfinite(la).all()
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))


def test_tile_seq_masks_each_row_group_to_its_own_tokens():
    """A packed tile's blocks: whole blocks every row sees first (looped), then up to the last
    row's token with each row group's count; a row past the block's start sees none of it."""
    s = tiled.tile_seq(250, 4, 4, 256, 2)       # rows at positions 254 .. 257, G = 2
    assert s.full == 0
    (t0, n, m0), (t1, n1, m1) = s.items
    assert (t0, n, t1, n1) == (0, 256, 256, 2)
    assert m0.keep == [(0, 2, 255), (2, 4, 256), (4, 6, 256), (6, 8, 256)]
    assert m1.keep == [(0, 2, 0), (2, 4, 0), (4, 6, 1), (6, 8, 2)]
    s = tiled.tile_seq(600, 0, 2, 256, 2)       # positions 600, 601: two whole blocks first
    assert s.full == 2 and [(t, n) for t, n, _ in s.items] == [(512, 90)]
    assert s.items[0][2].keep == [(0, 2, 89), (2, 4, 90)]


@pytest.mark.parametrize("mcols,wformat,act_rows,T", [(4, "fp4", 16, 12), (8, "int8", 16, 12),
                                                     (4, "fp4", 4, 4)])
def test_tile_on_rtl(tiny, have_verilator, mcols, wformat, act_rows, T):
    """A tile across an attention block on the RTL (the board's memory path): 12 rows with an
    ACT RAM of 16 (every projection one MM replayed for ceil(12 / MCOLS) groups; the packed
    attention's masked blocks; the column parts), and 4 rows on the card's own ACT RAM of
    MCOLS rows (no replay): DRAM and TMEM as the ISA simulator's."""
    from opentpu import rtlsim
    _, W, spec = tiny
    wkw = dict(wformat=wformat, head_format="int8") if wformat != "int8" else {}
    cfg = board_config(MCOLS=mcols, ACT_ROWS=act_rows, DRAM_BYTES=1 << 24,
                       PAIR=wformat != "int8")
    p0 = 256 - T // 2
    toks = [int(t) for t in np.random.default_rng(7).integers(0, 1000, p0 + T)]
    eng = Engine(spec, W, cap=512, cfg=cfg, rows=T, **wkw)
    eng.prefill(toks[:p0])
    dram = eng.backend.machine.slices[0].dram.copy()
    _put(dram, _inputs(eng, toks[p0:], p0, T))
    progs = tiled.compile_tile(eng.image, p0, T, logit_rows=[T - 1])
    m = Machine(cfg, progs, [dram.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, progs, [dram.copy()], uarch=rtlsim.BOARD_UARCH, axi=True,
                                 boot=True)
    n = eng.image.nbytes
    assert np.array_equal(drams[0][:n], m.slices[0].dram[:n])
    assert np.array_equal(tmems[0], m.slices[0].tmem)


# ------------------------------------------------------------------ tiled prompt runs
def _family(case, tmp_path_factory):
    """(W, spec, cfg, engine kwargs) of a tiny model of each family (test_prefill's, and
    test_llama's Llama-likes read back from their checkpoints)."""
    if case in ("smollm3", "phi3"):
        import test_llama as TLL
        from opentpu.llm import load_spec
        from opentpu.llm.qwen3 import load_weights
        torch.manual_seed(0)
        m = {"smollm3": TLL._smollm3, "phi3": TLL._phi3}[case]().float().eval()
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "norm" in n:
                    p.copy_(1 + 0.1 * torch.randn_like(p))
        path = tmp_path_factory.mktemp(case)
        m.save_pretrained(path)
        return load_weights(path), load_spec(path), board_config(DRAM_BYTES=1 << 26), {}
    from test_prefill import _model
    return _model(case)


def _prompt_engines(case, mcols, act, R, tmp_path_factory, pair=None):
    """Two engines of the case's model with prompt runs, MCOLS and ACT_ROWS as given and the
    image's rows act: tiled (a) and today's held to R rows a run (b)."""
    W, spec, cfg, kw = _family(case, tmp_path_factory)
    cfg = dataclasses.replace(cfg, MCOLS=mcols, ACT_ROWS=act)
    if pair is not None:
        cfg = dataclasses.replace(cfg, PAIR=pair)
    a = Engine(spec, W, cap=512, cfg=cfg, rows=act, resident=True, prompt_runs=True, **kw)
    a.tile_runs = True
    b = Engine(spec, W, cap=512, cfg=cfg, rows=act, resident=True, prompt_runs=True, **kw)
    b.tile_runs = False
    b._prompt_rmax = {k: (R, True) for k in range(3)}
    return spec, a, b


def _dram_but_scratch(eng, P):
    """Each slice's DRAM but the runs' own words (test_prefill._drams), the I/O area's logits
    rows (an L run writes its last row's: row T - 1 or R - 1), its input rows (a tiled run's
    spilled x: tiled.Res), its attention outputs' scratch (a tiled run's: tiled._oscr) and
    Qwen3.5's per-row scratch gr / on (the last run's rows)."""
    from test_prefill import _drams
    io, n = eng.image.io, eng.image.nbytes
    out = []
    for d in _drams(eng, P):
        d = d[:n].copy()
        d[io["logits"]:io["logits"] + 4 * eng.image.rows * eng.spec.vocab] = 0
        d[io["x"]:io["x"] + 4 * eng.image.rows * eng.spec.hidden] = 0
        o = eng.image.descriptors(0).oscr
        d[io["o"]:io["o"] + 4 * o.shape[0] * o.shape[1]] = 0
        if "gr" in io:
            d[io["gr"]:io["gr"] + 4 * 2 * eng.image.nl * eng.image.grows] = 0
            d[io["on"]:io["on"] + 4 * 4 * eng.spec.lin_dv * eng.image.grows] = 0
        out.append(d)
    return out


@pytest.mark.parametrize("case, mcols, P1, P2, R, T, form", [
    ("qwen3", 4, 250, 20, 8, 16, None),     # 2 rows a packed tile (G = 2), across the bucket's
    ("qwen3", 2, 37, 30, 4, 16, None),      # end; MCOLS = G: a row a tile
    ("qwen3", 8, 37, 30, 8, 16, None),      # 4 rows a tile
    ("smollm3", 4, 37, 30, 8, 16, None),    # no q / k norms, layers without RoPE (G = 4)
    ("phi3", 4, 37, 30, 8, 16, None),       # G = 3, RoPE on 96 of 128 dimensions, LongRoPE
    ("phi3", 8, 37, 30, 8, 16, None),       # 2 rows of 3 heads a tile
    ("lfm2", 4, 250, 20, 8, 16, None),      # convolution blocks by channel parts, 64-wide heads
    ("kh8", 4, 250, 20, 4, 16, None),       # Qwen3.5: gated attention in query parts of
    ("kh16", 4, 37, 30, 4, 16, None),       # 2 heads (QPART), DeltaNet rows; 16 DeltaNet heads
    ("qwen3", 4, 250, 20, 8, 16, {"spill": True}),     # x spilled while the heads attend
    ("lfm2", 4, 37, 30, 8, 16, {"spill": True}),
    ("kh8", 4, 250, 20, 4, 16, {"spill": True, "drows": 5}),   # DeltaNet: runs of 5 rows
    ("kh8", 4, 250, 20, 4, 12, {"spill": True, "drows": 5, "oscr": False}),  # the outputs
])                                          # gathered in TMEM (its [16, n_q d]: 12 rows)
def test_tiled_prompt_runs_are_todays(tmp_path_factory, monkeypatch, case, mcols, P1, P2, R, T,
                                      form):
    """Two prompts (a chat's turns, the second from where the first left) through tiled prompt
    runs of T rows and through today's of R: the last logits and every slice's DRAM (KV caches,
    DeltaNet states and windows, LFM2's rings) bit for bit; today's runs at MCOLS 4 do not pair
    (int8 weights). form: the tiled runs' form (tiled.variants), else the first that fits."""
    if form is not None:
        monkeypatch.setattr(tiled, "variants", lambda fam, T, mc: [form])
    r = np.random.default_rng(P1)
    p1, p2 = ([int(t) for t in r.integers(0, 1000, n)] for n in (P1, P2))
    spec, a, b = _prompt_engines(case, mcols, 16, R, tmp_path_factory)
    got = [a.prefill(p1), a.prefill(p2)]
    assert max(a._prompt_tile.values()) == T     # (a forced form: fewer in a bucket over IMEM)
    assert max(s["rows"] for s in a.stats) == T
    keys = {k[0] for k in a._prompt_progs}
    assert "prompt tile" in keys
    want = [b.prefill(p1), b.prefill(p2)]
    assert max(s["rows"] for s in b.stats) == R
    for x, y in zip(got, want):
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    assert all(np.array_equal(x, y) for x, y in zip(_dram_but_scratch(a, P1 + P2),
                                                    _dram_but_scratch(b, P1 + P2)))


@pytest.mark.parametrize("pair", [False, True], ids=["dstep", "dstep2"])
def test_tiled_27b_shape_prompt_runs_are_the_steps(pair):
    """Qwen3.5-27B's DeltaNet shape (test_qwen35._tiny27: 48 value heads on 16 key heads,
    qkonce) on one BCU slice (bcu1525-wg: DSTEP, WG 8, no PAIR), tiled prompt runs: the
    pipelined pairs' gates in TMEM, the key heads two at a time, with DSTEP2 a pair's heads a
    row in one DSTEP. Two prompts (the second from where the first left) give the logits of
    token-by-token decode and its layers' blocks (KV caches, DeltaNet states and windows) bit
    for bit."""
    import test_qwen35 as TQ
    from opentpu import target as T
    _, W, spec, _ = TQ._tiny27()
    cfg = T.config("bcu1525-wg", DRAM_BYTES=1 << 26, DSTEP2=pair)
    r = np.random.default_rng(5)
    p1, p2 = ([int(t) for t in r.integers(0, 1024, n)] for n in (21, 9))
    a = Engine(spec, W, cap=512, cfg=cfg, rows=16, resident=True, prompt_runs=True)
    a.tile_runs = True
    got = [a.prefill(p1), a.prefill(p2)]
    assert "prompt tile" in {k[0] for k in a._prompt_progs}
    b = Engine(spec, W, cap=512, cfg=cfg, rows=16, resident=True)
    want = [[b.step(t) for t in p][-1] for p in (p1, p2)]
    for x, y in zip(got, want):
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    img = a.image                       # (past the head: the tiled runs' scratch)
    assert np.array_equal(a.backend.machine.slices[0].dram[img.layer0:img.head[0]],
                          b.backend.machine.slices[0].dram[img.layer0:img.head[0]])


@pytest.mark.parametrize("case, mcols, P1, P2, T, levels, peel", [
    ("phi3", 4, 250, 20, 16, {2}, False),   # G = 3: a row a tile; a loop over its 2 KV heads,
    ("qwen3", 4, 250, 20, 7, {2}, False),   # across the bucket's end; G = 2: 2 rows a tile, a
    ("qwen3", 4, 250, 20, 8, {2}, True),    # partial 7th; the last tile peeled (tiled.PEEL)
    ("smollm3", 4, 37, 30, 16, {2}, False),     # layers without RoPE (G = 4)
    ("phi3", 8, 250, 20, 5, {2}, False),    # 2 rows of 3 heads a tile, a partial last one
    ("lfm2", 4, 250, 20, 16, {2}, False),   # 64-wide heads (padded to D), the stack's loops
    ("kh8", 4, 250, 20, 16, set(), False),  # Qwen3.5's gated attention: unrolled (its gates)
])
def test_looped_attention_is_the_unrolled_one(tmp_path_factory, monkeypatch, case, mcols, P1,
                                              P2, T, levels, peel):
    """A tiled prompt run's attention in hardware loops over the KV heads and each part's packed
    tiles (tiled._heads_loop: the loop levels a form's trace takes, `levels`; with `peel` the
    last tile after the loop) against the same runs unrolled (tiled.LOOP off): two prompts at T
    rows a run, the last logits and every slice's DRAM bit for bit, shorter programs, the wait
    counts checked against every run (OTPU_WAITS_CHECK) and every MM's drain keeping up with its
    array."""
    from conftest import assert_drains_keep_up, waits_programs
    monkeypatch.setenv("OTPU_WAITS_CHECK", "1")
    monkeypatch.setattr(tiled, "PEEL", peel)
    W, spec, cfg, kw = _family(case, tmp_path_factory)
    cfg = dataclasses.replace(cfg, MCOLS=mcols, ACT_ROWS=16)
    r = np.random.default_rng(P1)
    p1, p2 = ([int(t) for t in r.integers(0, 1000, n)] for n in (P1, P2))
    seen = set()
    loop = tiled._heads_loop

    def spy(*a, **k):
        seen.add(2 if a[12] and len(a[1]) > 1 else 1)
        return loop(*a, **k)

    def run(on):
        monkeypatch.setattr(tiled, "LOOP", on)
        e = Engine(spec, W, cap=512, cfg=cfg, rows=16, resident=True, prompt_runs=True, **kw)
        e.tile_runs = True
        e._prompt_tile = {b: T for b in (1, 2)}         # (the tile's rows, today's fewer)
        e._prompt_rmax = {b: (1, True) for b in (0, 1, 2)}
        progs = waits_programs(e)
        return e, [e.prefill(p1), e.prefill(p2)], progs

    monkeypatch.setattr(tiled, "_heads_loop", spy)
    a, got, progs = run(True)
    assert seen == levels
    b, want, _ = run(False)
    assert {s["rows"] for s in a.stats} == {s["rows"] for s in b.stats} >= {T}
    for x, y in zip(got, want):
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    assert all(np.array_equal(x, y) for x, y in zip(_dram_but_scratch(a, P1 + P2),
                                                    _dram_but_scratch(b, P1 + P2)))
    n = {e: {k: max(map(len, v[0])) for k, v in e._prompt_progs.items() if k[0] == "prompt tile"}
         for e in (a, b)}
    assert n[a] and n[a].keys() == n[b].keys()
    assert all(n[a][k] < n[b][k] for k in n[a]) if levels else n[a] == n[b]
    assert_drains_keep_up(progs, cfg)


@pytest.mark.parametrize("model, Ts", [("phi4mini", (12,) * 8), ("e4b", (4,) * 7 + (3,))])
def test_looped_tiles_fit_the_cards_imem(model, Ts):
    """At the real dims (layout only, no weights) on the card's configuration (m4: MCOLS 4,
    ACT_ROWS 16, 4K-instruction IMEM; otpu-chat's formats; the image's rows grown to 16) the
    looped attention (tiled._heads_loop) fits a tile of Ts[b - 1] rows in IMEM in bucket b of a
    2K context, P and L, its heads and tiles in loops: Phi-4-mini 12 rows everywhere (16: TMEM;
    unrolled, its whole-model program was 3,666 instructions at 4 rows in bucket 1 and 11,552 in
    bucket 8: 1 to 3 rows on the card), Gemma 4 E4B 4, 3 in bucket 8 (at 4 rows its L program
    is 4,114 instructions there, P 4,044; unrolled it took 1 to 3 rows; at 8 its layers' other
    work is over 4K)."""
    from opentpu.llm import generate as G
    seen = set()
    loop = tiled._heads_loop

    def spy(*a, **k):
        seen.add(a[12] and len(a[1]) > 1)
        return loop(*a, **k)

    if model == "phi4mini":
        spec = Spec(3072, 32, 24, 8, 128, 8192, 200064, eps=1e-5, theta=1e4, bos=199999,
                    eos=(200020, 199999), qk_norm=False, rotary=96, rope_div=(1.0,) * 48,
                    rope_scale=1.1902380714238083, ctx=4096, embed="int8",
                    mix="gateup@4-29=nvfp4,down@4-29=fp4")
        wf = "mix"
    else:
        from opentpu.llm.gemma4 import Spec as SG
        kinds = (("sliding",) * 5 + ("full",)) * 7
        src = tuple(range(24)) + tuple(23 if k == "full" else 22 for k in kinds[24:])
        spec = SG(2560, kinds, src, (10240,) * 42, 8, 2, 256, 512, 262144, 256,
                  fit_formats="head=fp4,gateup@0-11=fp4")
        wf = "int8"
    cfg = board_config(PAIR=True, DSTEP=True, STREAM=True, SEQ_WC=True, DRAM_BYTES=1 << 33)
    assert cfg.MCOLS == 4 and cfg.act_rows == 16 and cfg.IMEM_WORDS // 8 == 4096
    img = spec.image(cfg, 2048, 1, 16, wformat=wf, lookup=True)
    tiled._heads_loop = spy
    try:
        for blocks, T in enumerate(Ts, 1):
            for kind in ("P", "L"):
                progs, _ = tiled.compile_tile_run(img, blocks, T, kind)
                assert G.fits(img, progs), (blocks, kind, max(map(len, progs)))
    finally:
        tiled._heads_loop = loop
    assert seen == {True}


@pytest.mark.parametrize("tile", [True, False], ids=["tile", "prompt"])
def test_qwen35_nested_layers_are_the_unrolled_ones(tmp_path_factory, monkeypatch, tile):
    """Qwen3.5's prefill programs loop a unit's DeltaNet layers inside the unit's loop
    (qwen35.NEST, lfm2.subs: the tiny model's (lin, lin, attn) x 2 runs its two DeltaNet layers
    as a loop of 2, one copy of the layer in the program): two prompts' last logits and every
    slice's DRAM bit for bit as with the layers unrolled, tiled runs and today's prompt runs
    alike, in shorter programs of the same rows."""
    from opentpu.llm import qwen35 as Q35
    r = np.random.default_rng(7)
    p1, p2 = ([int(t) for t in r.integers(0, 1000, n)] for n in (250, 20))
    W, spec, cfg, kw = _family("kh8", tmp_path_factory)
    cfg = dataclasses.replace(cfg, MCOLS=4, ACT_ROWS=16)

    def run():
        e = Engine(spec, W, cap=512, cfg=cfg, rows=16, resident=True, prompt_runs=True, **kw)
        e.tile_runs = tile
        return e, [e.prefill(p1), e.prefill(p2)]

    (a, got) = run()
    monkeypatch.setattr(Q35, "NEST", False)
    (b, want) = run()
    assert [s["rows"] for s in a.stats] == [s["rows"] for s in b.stats]
    assert ("prompt tile" if tile else "prompt") in {k[0] for k in a._prompt_progs}
    for x, y in zip(got, want):
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    assert all(np.array_equal(x, y) for x, y in zip(_dram_but_scratch(a, 270),
                                                    _dram_but_scratch(b, 270)))
    n = {e: {k: max(map(len, v[0])) for k, v in e._prompt_progs.items()} for e in (a, b)}
    assert n[a].keys() == n[b].keys() and all(n[a][k] < n[b][k] for k in n[a])


@pytest.mark.parametrize("dims, T", [("4B", 4), ("0.8B", 12)])
def test_qwen35_tiles_fit_the_cards_imem(dims, T):
    """At the real dims (layout only, no weights) on the card's configuration (m4: MCOLS 4,
    ACT_ROWS 16, 4K-instruction IMEM; otpu-chat's formats: the 4B's mix, the 0.8B's int8; the
    image's rows grown to 16) the nested DeltaNet layers (qwen35.NEST) fit the tile the TMEM
    holds in IMEM at every bucket of a 2K context, P and L: Qwen3.5-4B 4 rows (unrolled, its
    whole-model program was 4,210 instructions at bucket 5, so 2 rows from there), Qwen3.5-0.8B
    12 (unrolled: 4)."""
    from opentpu.llm import generate as G
    from opentpu.llm.qwen35 import Spec as S35
    kinds = ("linear", "linear", "linear", "attn") * (8 if dims == "4B" else 6)
    spec = S35(2560, kinds, 16, 4, 256, 64, 32, 128, 128, 9216, 248320, lin_kheads=16,
               embed="int8", mix="delta=fp4,mlp=nvfp4") if dims == "4B" else \
        S35(1024, kinds, 8, 2, 256, 64, 16, 128, 128, 3584, 248320, lin_kheads=16)
    cfg = board_config(PAIR=True, DSTEP=True, STREAM=True, SEQ_WC=True, DRAM_BYTES=1 << 33)
    assert cfg.MCOLS == 4 and cfg.act_rows == 16 and cfg.IMEM_WORDS // 8 == 4096
    img = spec.image(cfg, 2048, 1, 16, wformat="mix" if dims == "4B" else "int8", lookup=True)
    for blocks in range(1, 9):
        for kind in ("P", "L"):
            progs, _ = tiled.compile_tile_run(img, blocks, T, kind)
            assert G.fits(img, progs), (blocks, kind, max(map(len, progs)))


def test_tiled_mlp_past_act_ram_in_groups_is_todays(monkeypatch):
    """An MLP wider than ACT RAM (the tiny Qwen3.5 with F 2560 at ACT_BLOCKS 18, 2304 columns:
    as Qwen3.5-27B's 17408 against 128 x 128 on one slice): Spec.check takes it (the down MMs'
    chunk, 256 columns), a tile's ten chunks' activations go in two groups of five beside the
    operand (tiled.act_groups, the down MMs' sums held across them) and two prompts' tiled
    runs give today's prompt runs' last logits and every slice's DRAM bit for bit."""
    from test_qwen35 import _tiny_model
    seen, groups = [], tiled.act_groups
    monkeypatch.setattr(tiled, "act_groups",
                        lambda *a: seen.append([len(g) for g in groups(*a)]) or groups(*a))
    _, W, spec = _tiny_model(8, init=0.2, ffn=2560)
    cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=True, STREAM=True, PAIR=True, MCOLS=4,
                       ACT_ROWS=16, ACT_BLOCKS=18)
    assert spec.ffn > cfg.ACT_BLOCKS * cfg.D
    r = np.random.default_rng(37)
    p1, p2 = ([int(t) for t in r.integers(0, 1000, n)] for n in (37, 30))
    a = Engine(spec, W, cap=512, cfg=cfg, rows=16, resident=True, prompt_runs=True)
    a.tile_runs = True
    got = [a.prefill(p1), a.prefill(p2)]
    assert "prompt tile" in {k[0] for k in a._prompt_progs}
    assert seen and all(g == [5, 5] for g in seen)
    b = Engine(spec, W, cap=512, cfg=cfg, rows=16, resident=True, prompt_runs=True)
    b.tile_runs = False
    b._prompt_rmax = {k: (4, True) for k in range(3)}
    want = [b.prefill(p1), b.prefill(p2)]
    for x, y in zip(got, want):
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    assert all(np.array_equal(x, y) for x, y in zip(_dram_but_scratch(a, 67),
                                                    _dram_but_scratch(b, 67)))


@pytest.mark.parametrize("n, per, sizes", [(17, 11, [9, 8]), (10, 8, [5, 5]), (8, 5, [4, 4]),
                                           (10, 3, [3, 3, 2, 2]), (7, 2, [2, 2, 2, 1]),
                                           (5, 5, [5]), (4, 9, [4])])
def test_act_groups_are_as_few_as_fit_and_even(monkeypatch, n, per, sizes):
    """act_groups: n chunks of 2 blocks beside an operand of 4 in ACT RAM of 4 + 2 per blocks:
    one group where all fit, else as few as fit, in order, sizes within one of each other
    (Qwen3.5-27B fp4: 17 chunks of 8 blocks, 11 fit beside its 40: 9 + 8)."""
    from types import SimpleNamespace as NS
    monkeypatch.setattr(tiled, "current", lambda: NS(cfg=NS(ACT_BLOCKS=4 + 2 * per, D=128)))
    starts = list(range(0, 256 * n, 256))
    groups = tiled.act_groups(starts, 256, NS(KB=4))
    assert [len(g) for g in groups] == sizes and sum(groups, []) == starts


def test_lfm2_stack_in_one_loop_is_the_plans_runs(monkeypatch):
    """LFM2's tiled runs take a dense stack of both kinds in one formats group in one loop over
    its layers (tiled._lfm2_stack, tiled.STACK: a TMEM table of the layers' kinds, a loop of 0
    or 1 iterations around each kind's mixer at the kind's Counter, then the MLP of the layer's
    common part): on an irregular stack (its plan: four runs, one a loop), two prompts' last
    logits and every slice's DRAM (KV caches, conv rings) bit for bit as the plan's runs give
    them, the same rows, in shorter programs; the wait counts checked against every run
    (OTPU_WAITS_CHECK) and every MM's drain keeping up with its array."""
    from conftest import assert_drains_keep_up, waits_programs
    from test_prefill import _lfm2
    W, spec, cfg = _lfm2(("conv", "conv", "attn", "conv", "attn", "attn", "conv"))
    cfg = dataclasses.replace(cfg, MCOLS=4, ACT_ROWS=16)
    r = np.random.default_rng(11)
    p1, p2 = ([int(t) for t in r.integers(0, 1000, n)] for n in (250, 20))
    monkeypatch.setenv("OTPU_WAITS_CHECK", "1")

    def run():
        e = Engine(spec, W, cap=512, cfg=cfg, rows=16, resident=True, prompt_runs=True)
        e.tile_runs = True
        progs = waits_programs(e)
        return e, [e.prefill(p1), e.prefill(p2)], progs

    a, got, progs = run()
    assert a.image.stack is not None and len(a.image.plan) == 4      # (c), (c A) x 2, (A), (c)
    monkeypatch.setattr(tiled, "STACK", False)
    b, want, _ = run()
    assert [s["rows"] for s in a.stats] == [s["rows"] for s in b.stats]
    assert "prompt tile" in {k[0] for k in a._prompt_progs}
    for x, y in zip(got, want):
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    assert all(np.array_equal(x, y) for x, y in zip(_dram_but_scratch(a, 270),
                                                    _dram_but_scratch(b, 270)))
    n = {e: {k: max(map(len, v[0])) for k, v in e._prompt_progs.items() if k[0] == "prompt tile"}
         for e in (a, b)}
    assert n[a].keys() == n[b].keys() and all(n[a][k] < n[b][k] for k in n[a])
    assert_drains_keep_up(progs, cfg)


def test_lfm2_tiles_fit_the_cards_imem():
    """LFM2-2.6B's dims and stack (layout only, no weights) on the card's configuration (m4:
    MCOLS 4, ACT_ROWS 16, 4K-instruction IMEM; otpu-chat's formats, the model's mix; the image's
    rows grown to 16): its stack in one loop (tiled.STACK) fits a tile of 4 rows in IMEM at
    every bucket of a 2K context, P and L. The plan's runs unroll its 30 layers: 6,228
    instructions at bucket 1 and 14,110 at bucket 8, so today's prompt runs took it (1 row a run
    from bucket 5)."""
    from opentpu.llm import generate as G
    from opentpu.llm.lfm2 import Spec as SL
    kinds = tuple({"c": "conv", "A": "attn"}[k] for k in "ccAccAcccAcccAcccAcccAccAccAcc")
    spec = SL(2048, kinds, 32, 8, 64, 10752, 65536, mix="conv=nvfp4,gateup=nvfp4,down=fp4")
    cfg = board_config(PAIR=True, DSTEP=True, STREAM=True, SEQ_WC=True, DRAM_BYTES=1 << 33)
    assert cfg.MCOLS == 4 and cfg.act_rows == 16 and cfg.IMEM_WORDS // 8 == 4096
    img = spec.image(cfg, 2048, 1, 16, wformat="mix", lookup=True)
    assert img.stack is not None
    for blocks in range(1, 9):
        for kind in ("P", "L"):
            progs, _ = tiled.compile_tile_run(img, blocks, 4, kind)
            assert G.fits(img, progs), (blocks, kind, max(map(len, progs)))


@pytest.mark.parametrize("P1, P2", [(36, 24), (250, 20)])
def test_tiled_prompt_runs_pair_alike(tmp_path_factory, P1, P2):
    """4-bit Qwen3.5 with PAIR: a tiled run and today's are the same bits whether or not one
    of them pairs (a run of at most MCOLS / 2 rows: the MX MXU's paired MM sums the same exact
    integer, docs/isa.md "Column reuse") -- the tile's rows never pair, today's 2-row runs at
    the bucket's and the prompt's end (250 + 20) do."""
    r = np.random.default_rng(P1)
    p1, p2 = ([int(t) for t in r.integers(0, 1000, n)] for n in (P1, P2))
    _, a, b = _prompt_engines("fp4", 4, 16, 4, tmp_path_factory, pair=True)
    got = [a.prefill(p1), a.prefill(p2)]
    want = [b.prefill(p1), b.prefill(p2)]
    assert any(s["rows"] <= 2 for s in b.stats) == (P1 == 250)
    for x, y in zip(got, want):
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    assert all(np.array_equal(x, y) for x, y in zip(_dram_but_scratch(a, P1 + P2),
                                                    _dram_but_scratch(b, P1 + P2)))


def _dkl(want, got) -> float:
    """KL(softmax(want) || softmax(got)) of two logits vectors, in nats."""
    lw, lg = (np.asarray(v, np.float64) for v in (want, got))
    lw, lg = lw - lw.max(), lg - lg.max()
    lw, lg = lw - np.log(np.exp(lw).sum()), lg - np.log(np.exp(lg).sum())
    return float((np.exp(lw) * (lw - lg)).sum())


def _in_ranges(forms):
    """A form search (tiled.variants, layer_variants) in its range forms only."""
    return lambda *a: [dict(f, whole=False) for f in forms(*a) if f.get("oscr", True)]


@pytest.mark.parametrize("case, mcols, P1, P2, R, T", [
    ("qwen3", 4, 250, 20, 8, 16),           # a part of 2 heads a range (G = 2)
    ("phi3", 4, 37, 30, 8, 16),             # 3 heads (G = 3): ranges at odd blocks (int8)
    ("kh8", 4, 250, 20, 4, 16),             # Qwen3.5: 16 rows
])
def test_tiled_prompt_runs_in_ranges(tmp_path_factory, monkeypatch, case, mcols, P1, P2, R, T):
    """The range forms (tiled.ranges: each part's outputs quantized as it ends, a QACT range
    of its own, W_o an MM a range added up with ACC): their own rows (T), and logits not
    today's prompt runs' bits (a range's Sa is its own row maximum) but as close to the fp32
    forward's (reference_logits) by the KL divergence: within half of today's again (and
    1e-4; the tiny models' own divergence is 0.01-0.02, the ranges' within -15% / +40% of it),
    and from today's under half of today's own divergence."""
    monkeypatch.setattr(tiled, "variants", _in_ranges(tiled.variants))
    r = np.random.default_rng(P1)
    p1, p2 = ([int(t) for t in r.integers(0, 1000, n)] for n in (P1, P2))
    spec, a, b = _prompt_engines(case, mcols, 16, R, tmp_path_factory)
    got = [a.prefill(p1), a.prefill(p2)]
    assert max(a._prompt_tile.values()) == T
    want = [b.prefill(p1), b.prefill(p2)]
    W = _family(case, tmp_path_factory)[0]
    ref = sys.modules[type(a.image).__module__].reference_logits(spec, W, p1 + p2)
    ref = [ref[P1 - 1], ref[-1]]
    kl = [_dkl(y, x) for x, y in zip(got, want)]
    kr = [(_dkl(f, x), _dkl(f, y)) for x, y, f in zip(got, want, ref)]
    print(case, "dKL", kl, "against fp32 (ranges, today's)", kr)
    assert not all(np.array_equal(x, y) for x, y in zip(got, want))
    for k, (kg, kw) in zip(kl, kr):
        assert kg <= 1.5 * kw + 1e-4 and k <= 0.5 * kw


def test_tiled_shallow_pipeline(tmp_path_factory, monkeypatch):
    """A tiled run over IMEM at the attention's depth (ATTN_DEPTH: a head's entries leave up to
    ~3 x depth blocks unrolled) takes the same form at tiled.SHALLOW (over at both: the first,
    for the caller's IMEM check: bucket 8's, 4 blocks' code a packed tile at depth 2 against 8
    at 3); its runs are today's bit for bit (a 1030-token prompt: buckets 1-5, the heads' whole
    blocks looped from bucket 4 at depth 2, none at 3)."""
    W, spec, cfg, kw = _family("qwen3-board", tmp_path_factory)
    cfg = dataclasses.replace(cfg, MCOLS=4, ACT_ROWS=16)
    a, b = (Engine(spec, W, cap=2048, cfg=cfg, rows=16, resident=True, prompt_runs=True, **kw)
            for _ in range(2))
    img, imem = a.image, a.image.cfg
    size = lambda: max(map(len, tiled.compile_tile_run(img, 8, 16, "P")[0]))   # noqa: E731
    whole = tiled.variants          # (the outputs through the I/O scratch: oscr's forms)
    forms = lambda *x: [v for v in whole(*x) if v.get("oscr", True)]          # noqa: E731
    monkeypatch.setattr(tiled, "variants", forms)
    shallow = lambda *x: [dict(v, depth=tiled.SHALLOW) for v in forms(*x)]   # noqa: E731
    n3 = size()
    monkeypatch.setattr(tiled, "variants", shallow)
    n2 = size()
    monkeypatch.setattr(tiled, "variants", forms)
    assert n2 < n3
    monkeypatch.setattr(img, "cfg", dataclasses.replace(imem, IMEM_WORDS=8 * n2))
    assert size() == n2
    monkeypatch.setattr(img, "cfg", dataclasses.replace(imem, IMEM_WORDS=8 * (n2 - 1)))
    assert size() == n3
    monkeypatch.setattr(img, "cfg", imem)
    monkeypatch.setattr(tiled, "variants", shallow)
    a.tile_runs, b.tile_runs = True, False
    b._prompt_rmax = {k: (8, True) for k in range(6)}
    r = np.random.default_rng(5)
    p1, p2 = ([int(t) for t in r.integers(0, 1000, n)] for n in (1030, 20))
    got = [a.prefill(p1), a.prefill(p2)]
    assert set(a._prompt_tile.values()) == {16}
    want = [b.prefill(p1), b.prefill(p2)]
    for x, y in zip(got, want):
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    assert all(np.array_equal(x, y) for x, y in zip(_dram_but_scratch(a, 1050),
                                                    _dram_but_scratch(b, 1050)))


def test_tiled_forms_placed_next_fit_first():
    """The form search (_first_fit) tries every form with its tiles placed next-fit before any
    with best-fit (which fits a fragmented TMEM but reuses words that running work still reads:
    slower), then the first that fits best-fit; over IMEM, the next form."""
    from types import SimpleNamespace

    from opentpu.compiler import CompileError

    class Built:
        run_args = ()

        def __init__(self, n):
            self.n = n

        def finish(self):
            return [0] * self.n

    def search(fits, imem=1 << 17):
        tried = []

        def trace(form, fit):
            tried.append((form["i"], form.get("depth"), fit))
            if fit not in fits[form["i"]]:
                raise CompileError("TMEM exhausted: a tile of 4096 words")
            return [Built(fits[form["i"]][fit])]
        progs, _ = tiled._first_fit(trace, [{"i": i} for i in range(len(fits))],
                                    SimpleNamespace(IMEM_WORDS=imem))
        return len(progs[0]), tried

    # form 0 fits best-fit only, form 1 next-fit: form 1
    n, tried = search([{"best": 10}, {"next": 11, "best": 11}])
    assert n == 11 and tried == [(0, None, "next"), (1, None, "next")]
    # neither fits next-fit: the first that fits best-fit
    n, tried = search([{"best": 10}, {"best": 11}])
    assert n == 10 and tried == [(0, None, "next"), (1, None, "next"), (0, None, "best")]
    # form 1 next-fit over IMEM at both depths: form 0 best-fit, within it
    n, tried = search([{"best": 10}, {"next": 20}], imem=8 * 15)
    assert n == 10 and tried[-1] == (0, None, "best")
    assert (1, tiled.SHALLOW, "next") in tried


@pytest.mark.parametrize("case, mcols, act, p0, T, form", [
    ("qwen3", 4, 4, 300, 4, None),      # the card's ACT RAM (4 rows): a whole block, then the
    ("lfm2", 4, 16, 200, 16, None),     # masked one; 16 rows: replayed MMs, the convolution's
    ("kh8", 4, 16, 300, 16, None),      # channel parts; Qwen3.5: DeltaNet rows (DSTEP), gated
    ("gemma4-1", 4, 16, 600, 16, None),     # Gemma 4: the sliding window's start-masked block;
    ("gemma4-2", 8, 16, 300, 16, None),     # E4B's groups of 4, two rows a tile (the attention
    #                                         looped: tiled._heads_loop, but Qwen3.5's gated one)
    ("kh8", 4, 16, 300, 16, {"spill": True, "drows": 5}),  # x stored and loaded back in place,
    #                                                         DeltaNet in runs of 4 x 4
    ("kh8", 4, 16, 300, 16, "ranges"),  # W_o's operand in QACT ranges: an MM a range (ACC)
    ("kh8", 4, 16, 300, 12, {"spill": True, "drows": 4, "oscr": False}),  # outputs in TMEM
])
def test_tiled_prompt_run_on_rtl(tmp_path_factory, monkeypatch, have_verilator, case, mcols,
                                 act, p0, T, form):
    """A tiled prompt run (an L run, at a run-time position from the state's words, its tokens
    from out[]) on the RTL with the board's memory path: DRAM and TMEM as the ISA simulator's,
    on the card's MXU (MCOLS 4) with its own ACT RAM of 4 rows and with 16, and on 8 columns;
    form: the run's form (tiled.variants; "ranges": its range forms)."""
    from opentpu import rtlsim
    if form == "ranges":
        monkeypatch.setattr(tiled, "variants", _in_ranges(tiled.variants))
    elif form is not None:
        monkeypatch.setattr(tiled, "variants", lambda fam, T, mc: [form])
    if case.startswith("gemma4"):
        (W, spec, cfg), kw, cap = _gemma(tmp_path_factory, int(case[-1])), {}, 1024
    else:
        (W, spec, cfg, kw), cap = _family(case, tmp_path_factory), 512
    cfg = dataclasses.replace(cfg, MCOLS=mcols, ACT_ROWS=act)
    eng = Engine(spec, W, cap=cap, cfg=cfg, rows=act, resident=True, prompt_runs=True, **kw)
    eng.tile_runs = True
    toks = [int(t) for t in np.random.default_rng(p0).integers(0, 1000, p0 + act)]
    eng.prefill(toks[:p0])
    assert PF.r_max(eng, p0 // eng.block + 1) == T
    progs, names = PF.programs(eng, p0, T, "L")
    PF.write_tokens(eng, p0, toks[p0:p0 + T])
    g, w = eng.image.lookup["gen"], PF.words(eng, names, p0, T)
    for a, b in PF._ranges(sorted(w)):
        eng.backend.write(0, g["state"] + 4 * a, np.array([w[i] for i in range(a, b)],
                                                          np.float32))
    dram = eng.backend.machine.slices[0].dram.copy()
    m = Machine(cfg, progs, [dram.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, progs, [dram.copy()], uarch=rtlsim.BOARD_UARCH, axi=True,
                                 boot=True)
    n = eng.image.nbytes
    assert np.array_equal(drams[0][:n], m.slices[0].dram[:n])
    assert np.array_equal(tmems[0], m.slices[0].tmem)


def _gemma(tmp_path_factory, nkv: int):
    """test_gemma4's tiny Gemma 4 (sliding and global layers, KV-shared ones, a double-wide
    MLP, per-layer inputs) with nkv KV heads: 8 query heads on 1 (two parts of 4 on MCOLS 4)
    or on 2 (E4B's: groups of 4, two rows a tile at MCOLS 8)."""
    import test_gemma4 as TG
    tf = TG.transformers            # (the module test_gemma4 builds with: an engine's imports
    orig = tf.Gemma4TextConfig      # can put another in sys.modules)
    try:
        tf.Gemma4TextConfig = lambda **k: orig(**{**k, "num_key_value_heads": nkv})
        _, W, spec = TG._model(tmp_path_factory.mktemp("gemma4"))
    finally:
        tf.Gemma4TextConfig = orig
    assert spec.kvh(0) == nkv
    return W, spec, TG._cfg()


@pytest.mark.parametrize("mcols, nkv, wf, host, P1, P2, R, T, form", [
    (4, 1, "int8", "0", 250, 20, 4, 16, None),  # two parts of 4 heads a row, across the
    (8, 2, "int8", "0", 250, 20, 4, 16, None),  # bucket's end; groups of 4: two rows a tile
    #                                             (the attention looped: 16 rows a bucket)
    (4, 1, "fp4", "1", 37, 30, 8, 16, None),    # 4-bit layers, the PLE table on the host
    (8, 2, "int8", "0", 760, 20, 4, 16, None),  # the sliding ring's wrap (768 slots), packed
    (8, 1, "int8", "0", 250, 20, 4, 8, "small"),    # x spilled; query parts of 2 heads (the
])                                                  # global layers' 1), 4 (8) rows a tile
def test_tiled_gemma4_prompt_runs_are_todays(tmp_path_factory, monkeypatch, mcols, nkv, wf,
                                             host, P1, P2, R, T, form):
    """Gemma 4's tiled prompt runs (T rows, the fewest a bucket takes; the attention in hardware
    loops, tiled._heads_loop) against today's prompt runs of R rows: the last logits, the
    layers' DRAM (weights, KV caches, the sliding rings) and the next step's logits bit for bit.
    form "small": x spilled while the heads attend and the query parts held to TMEM / 64 words
    (tiled.QPART)."""
    monkeypatch.setenv("OTPU_PLE_HOST", host)
    if form == "small":
        monkeypatch.setattr(tiled, "variants", lambda fam, T, mc: [{"spill": True}])
        monkeypatch.setattr(tiled, "QPART", 64)
    W, spec, cfg = _gemma(tmp_path_factory, nkv)
    cfg = dataclasses.replace(cfg, MCOLS=mcols, ACT_ROWS=16)
    kw = dict(wformat=wf, head_format="int8" if wf == "fp4" else None)
    r = np.random.default_rng(P1)
    p1, p2 = ([int(t) for t in r.integers(0, 1000, n)] for n in (P1, P2))
    a = Engine(spec, W, cap=1024, cfg=cfg, rows=16, resident=True, prompt_runs=True, **kw)
    a.tile_runs = True
    got = [a.prefill(p1), a.prefill(p2)]
    assert min(a._prompt_tile.values()) == T
    b = Engine(spec, W, cap=1024, cfg=cfg, rows=16, resident=True, prompt_runs=True, **kw)
    b.tile_runs = False
    b._prompt_rmax = {k: (R, True) for k in range(5)}
    want = [b.prefill(p1), b.prefill(p2)]
    for x, y in zip(got, want):
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    img = a.image

    def layers(e):
        return e.backend.machine.slices[0].dram[img.layer0:img.head[0]]
    assert np.array_equal(layers(a), layers(b))
    assert np.array_equal(a.step(5), b.step(5))
    if form == "small":             # (the global layers' 4 heads a row, 8 at MCOLS 8: parts
        tile = a._prompt_progs      # of 1 and 2 heads, 8 and 4 rows a tile)
        assert any(k[0] == "prompt tile" for k in tile)


@pytest.mark.parametrize("how", ["default", "kwarg", "env"])
def test_engine_tile_runs_rows(tmp_path_factory, monkeypatch, how):
    """Tiled prompt runs by default (OTPU_TILE unset, or Engine(tile_runs=True), or OTPU_TILE=1
    where it says nothing): the image's I/O rows grow to the configuration's ACT_ROWS (an L run
    of T rows writes row T - 1's logits) and a prompt's runs are tiled ones of up to ACT_ROWS
    rows; with Engine(tile_runs=False), or OTPU_TILE=0 where it says nothing, neither (today's
    runs, the same bits)."""
    W, spec, cfg, kw = _family("qwen3", tmp_path_factory)
    cfg = dataclasses.replace(cfg, MCOLS=4, ACT_ROWS=16)
    monkeypatch.delenv("OTPU_TILE", raising=False)
    off = Engine(spec, W, cap=512, cfg=cfg, rows=8, resident=True, prompt_runs=True,
                 tile_runs=False, **kw)
    assert off.rows == 8 and not PF.tiling(off)
    monkeypatch.setenv("OTPU_TILE", "0")
    off0 = Engine(spec, W, cap=512, cfg=cfg, rows=8, resident=True, prompt_runs=True, **kw)
    assert off0.rows == 8 and not PF.tiling(off0)
    monkeypatch.delenv("OTPU_TILE")
    if how == "env":
        monkeypatch.setenv("OTPU_TILE", "1")
    eng = Engine(spec, W, cap=512, cfg=cfg, rows=8, resident=True, prompt_runs=True,
                 **(dict(kw, tile_runs=True) if how == "kwarg" else kw))
    assert eng.rows == eng.image.rows == 16 and PF.tiling(eng)
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 40)]
    lg = eng.prefill(toks)
    assert [s["rows"] for s in eng.stats] == [16, 16, 8]
    want = off.prefill(toks)
    assert np.array_equal(lg.view(np.uint32), want.view(np.uint32))


def test_tile_default_leaves_two_slices_today(monkeypatch):
    """The default reaches one slice only (tiled.check): a design configuration's engine (two
    slices) keeps its rows and today's prompt runs, the tiled default on."""
    import test_prefill as TP
    W, spec, cfg, kw = TP._model("qwen3-design")
    monkeypatch.delenv("OTPU_TILE", raising=False)
    eng = Engine(spec, W, cap=512, cfg=cfg, resident=True, prompt_runs=True, **kw)
    assert cfg.S == 2 and eng.rows == TP.PREFILL_ROWS and not PF.tiling(eng)
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 30)]
    eng.prefill(toks)
    assert eng._prompt_progs and not any(k[0] == "prompt tile" for k in eng._prompt_progs)


@pytest.mark.parametrize("P1", [40, 245])
def test_tiled_mtp_prompt_runs_are_todays(P1):
    """MTP (Qwen3.5's drafter: the tiled P / L runs store their hidden rows, today's M runs
    follow each), the tiled prompt runs against today's (tile_runs False): a prompt, greedy
    decoding with the drafter, then the next turn's prompt from the committed slot (at P1 245
    across the bucket's end): the tokens, the runs' drafts, the DeltaNet states and windows and
    the MTP layer's KV cache bit for bit."""
    import test_prefill as TP
    from opentpu.llm.mtp import MTPDecoder, MTPStats, mtp_engine
    _, W, spec = TP._tiny_model(8, init=0.2)
    W = TP._mtp_weights(W, spec)
    r = np.random.default_rng(P1)
    p1, p2 = [int(t) for t in r.integers(0, 1000, P1)], [int(t) for t in r.integers(0, 1000, 9)]
    a, b = (MTPDecoder(mtp_engine(spec, W, cap=512, cfg=TP.CFG, prompt_runs=True, tile_runs=t))
            for t in (True, False))
    assert PF.tiling(a.eng) and not PF.tiling(b.eng)
    sa, sb = a.generate(p1, max_new=6), b.generate(p1, max_new=6)
    assert sa.tokens == sb.tokens and a.slot == b.slot and a.eng.pos == b.eng.pos
    assert [x[:2] for x in sa.runs] == [x[:2] for x in sb.runs]
    q = [sa.tokens[-1]] + p2
    assert a.prefill(q, MTPStats()) == b.prefill(q, MTPStats())
    n = a.eng.pos
    assert n == b.eng.pos == P1 + 5 + len(q)
    assert all(np.array_equal(x, y) for x, y in
               zip(TP._states(a.eng, spec, a.slot), TP._states(b.eng, spec, b.slot)))
    assert all(np.array_equal(x, y) for x, y in zip(TP._mtp_kv(a.eng, n), TP._mtp_kv(b.eng, n)))
    assert any(k[0] == "prompt tile" for k in a.eng._prompt_progs)


@pytest.fixture(scope="module")
def gmoe(tmp_path_factory):
    """test_gemma4_moe's tiny Gemma 4 MoE: 8 experts 96 wide, top 2, beside the 320-wide MLP;
    sliding layers of 2 KV heads (G = 2), global ones of 1 (G = 4, K = V)."""
    import test_gemma4_moe as TGM
    return TGM._model(tmp_path_factory, "tmoe", enable_moe_block=True, num_experts=8,
                      top_k_experts=2, moe_intermediate_size=96)


def test_tiled_layer_runs_stay_opt_in(gmoe, monkeypatch):
    """The tiled prompt runs' default does not reach a MoE model: with OTPU_TILE unset its
    expert-major layer runs are today's (Engine.layer_tile off) and its image holds neither the
    mask tiles nor the I/O scratch (prefill.tile_scratch: the layout, and so the expert slots,
    as before); OTPU_TILE=1 asks for them."""
    _, W, spec = gmoe
    cfg = board_config(DRAM_BYTES=1 << 26, ACT_ROWS=16)
    kw = dict(cap=1024, cfg=cfg, rows=1, wformat="fp4", experts=6, layer_major=4,
              expert_major=True)
    monkeypatch.delenv("OTPU_TILE", raising=False)
    a = Engine(spec, W, **kw)
    assert not a.layer_tile and "amask" not in a.image.lookup and "o" not in a.image.io
    monkeypatch.setenv("OTPU_TILE", "1")
    b = Engine(spec, W, **kw)
    assert b.layer_tile and "amask" in b.image.lookup and "o" in b.image.io


@pytest.mark.parametrize("wf, T, R", [("int8", 8, 0), ("fp4", 8, 4), ("fp4", 6, 3)])
def test_tiled_expert_major_layer_runs(gmoe, monkeypatch, wf, T, R):
    """Gemma 4 MoE's expert-major layer runs tiled (Engine.layer_tile, OTPU_TILE=1 as the image
    is built: its mask tiles; tiled.compile_layer_tile): runs of T rows (the layer before
    ended a group of rows at a time, the attention in packed tiles, the route and need lines
    a group of LINE / 4 ids at a time, the dense MLP's weights once a run) against expert-major
    runs of R rows (4-bit: none of either pairs at MCOLS 4) or token by token (R = 0, int8):
    the logits, the layers' blocks (KV caches) and the decode steps after, bit for bit; 40
    tokens in chunks of 24 and 16 rows, 6 expert slots a layer."""
    from opentpu.llm import gemma4 as G4
    _, W, spec = gmoe
    monkeypatch.setattr(G4, "PREFILL_CHUNK", 24)
    monkeypatch.setenv("OTPU_TILE", "1")
    cfg = board_config(DRAM_BYTES=1 << 26, ACT_ROWS=16)
    kw = dict(cap=1024, cfg=cfg, rows=1, wformat=wf, experts=6)
    toks = [int(t) for t in np.random.default_rng(13).integers(0, 1000, 40)]
    a = Engine(spec, W, layer_major=T, expert_major=True, **kw)
    b = Engine(spec, W, layer_major=R, expert_major=bool(R), tile_runs=False, **kw)
    assert a.layer_tile and not b.layer_tile and "amask" in b.image.lookup
    la, lb = a.prefill(toks), b.prefill(toks)
    assert any(k[-1] == "tile" for k in a._layer_runs if isinstance(k, tuple))
    assert a.pos == b.pos == len(toks)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    img = a.image
    assert np.array_equal(a.backend.machine.slices[0].dram[img.layer0:img.head[0]],
                          b.backend.machine.slices[0].dram[img.layer0:img.head[0]])
    t = int(np.argmax(la))
    for _ in range(2):
        ga, gb = a.step(t), b.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))


def test_tiled_expert_major_layer_runs_in_ranges(gmoe, monkeypatch):
    """Gemma 4 MoE's tiled layer runs in the range forms (tiled.ranges, the forms a run takes
    where its outputs whole do not fit: gemma-4-26B-A4B's global layers at 8 rows): 8 rows a
    run against expert-major runs of 4 rows (fp4): the logits within a small KL divergence."""
    from opentpu.llm import gemma4 as G4
    _, W, spec = gmoe
    monkeypatch.setattr(G4, "PREFILL_CHUNK", 24)
    monkeypatch.setenv("OTPU_TILE", "1")
    monkeypatch.setattr(tiled, "layer_variants", _in_ranges(tiled.layer_variants))
    cfg = board_config(DRAM_BYTES=1 << 26, ACT_ROWS=16)
    kw = dict(cap=1024, cfg=cfg, rows=1, wformat="fp4", experts=6)
    toks = [int(t) for t in np.random.default_rng(13).integers(0, 1000, 40)]
    a = Engine(spec, W, layer_major=8, expert_major=True, **kw)
    b = Engine(spec, W, layer_major=4, expert_major=True, tile_runs=False, **kw)
    la, lb = a.prefill(toks), b.prefill(toks)
    assert any(k[-1] == "tile" for k in a._layer_runs if isinstance(k, tuple))
    assert not np.array_equal(la, lb)
    print("gemma4 em dKL", _dkl(lb, la))
    assert _dkl(lb, la) < 1e-3


@pytest.mark.parametrize("wf, T, R, pair", [("int8", 8, 0, False), ("fp4", 8, 4, False),
                                             ("fp4", 6, 4, False), ("int8", 8, 0, True)])
def test_tiled_qwen35_expert_major_layer_runs(monkeypatch, wf, T, R, pair):
    """Qwen3.5-MoE's expert-major layer runs tiled (Engine.layer_tile): runs of T rows (the
    MoE layer before ended EM_GROUP rows at a time, the DeltaNet rows in runs of the image's
    4 gr / on rows, the gated attention in tiles, the route, need lines and shared expert a
    group of LINE / 4 ids at a time; layer 0's rows from the host's slot untiled, 4 a run)
    against token by token (R = 0, int8) or expert-major runs of R rows (4-bit with PAIR: no
    run of either of at most 2 rows, which would pair): the logits, the layers' blocks
    (DeltaNet states and windows, KV caches) and the decode steps after, bit for bit; 43
    tokens (the first 3 token by token), chunks of 24 and 16 rows, 6 slots a layer, the
    embedding rows from the host's table. pair: DSTEP2 (a pair's two heads a row in one DSTEP)."""
    import test_qwen35_moe as TQM
    from opentpu.llm import qwen35 as Q35
    from opentpu.llm.qwen3 import device_config
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    monkeypatch.setenv("OTPU_TILE", "1")
    spec, W = TQM._untied(TQM._tiny())
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=6, wformat=wf, MCOLS=4,
                        PAIR=True, DSTEP=True, STREAM=True)
    kw = dict(cap=256, cfg=dataclasses.replace(cfg, ACT_ROWS=16, DSTEP2=pair), rows=1,
              resident=True,
              experts=6, wformat=wf)
    a = Engine(spec, W, layer_major=T, expert_major=True, tile_rows=T, **kw)
    b = Engine(spec, W, tile_runs=False, **(dict(kw, layer_major=R, expert_major=True)
                                            if R else kw))
    assert a.layer_tile and "amask" in b.image.lookup and a.image.embed_host
    toks = [int(t) for t in np.random.default_rng(11).integers(0, 1000, 43)]
    la, lb = a.prefill(toks), b.prefill(toks)
    assert any(k[-1] == "tile" for k in a._layer_runs if isinstance(k, tuple))
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    img = a.image
    assert np.array_equal(a.backend.machine.slices[0].dram[img.layer0:img.head[0]],
                          b.backend.machine.slices[0].dram[img.layer0:img.head[0]])
    t = int(np.argmax(la))
    for _ in range(2):
        ga, gb = a.step(t), b.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))


@pytest.mark.parametrize("fam, act, T, li, form", [
    ("gemma4", 4, 4, 1, None), ("gemma4", 16, 8, 5, None), ("qwen35", 4, 4, 2, None),
    ("qwen35", 16, 8, 3, None), ("gemma4", 16, 8, 5, {"whole": False})])
def test_tiled_expert_major_layer_run_on_rtl(gmoe, monkeypatch, have_verilator, fam, act, T,
                                             li, form):
    """A tiled expert-major layer run on the RTL with the board's memory path (DRAM and TMEM as
    the ISA simulator's), from chunk row 3 (its records as a prompt left them), every need
    served: the tiny Gemma 4 MoE's sliding layer, 4 rows on an MCOLS 4 card's ACT RAM (4 rows),
    and a global one's 8 on ACT_ROWS 16 (and in QACT ranges: form), at position 300 after a
    300-token prompt; the tiny Qwen3.5-MoE's attention layer (4 rows, ACT_ROWS 4, MCOLS 4) and
    the DeltaNet layer after it (its MoE's end; 8 rows on 16, DSTEP, PAIR), at position 200 after
    200 tokens. The ACT RAM's rows are a multiple of MCOLS: 4 rows is the MCOLS 4 card's (the
    target's MCOLS 8 has 16)."""
    from opentpu import rtlsim
    from opentpu.compiler import arg_words
    from opentpu.llm.qwen3 import RunPos
    monkeypatch.setenv("OTPU_TILE", "1")
    if form is not None:
        monkeypatch.setattr(tiled, "layer_variants", lambda img, fam, T: [form])
    mc = min(act, board_config().MCOLS)
    if fam == "gemma4":
        _, W, spec = gmoe
        cfg, p, cap = board_config(DRAM_BYTES=1 << 26, ACT_ROWS=act, MCOLS=mc), 300, 1024
        eng = Engine(spec, W, cap=cap, cfg=cfg, rows=1, wformat="fp4", experts=6,
                     layer_major=T, expert_major=True)
    else:
        import test_qwen35_moe as TQM
        spec, W = TQM._untied(TQM._tiny())
        cfg = board_config(DRAM_BYTES=1 << 26, ACT_ROWS=act, MCOLS=mc, PAIR=True, DSTEP=True,
                           STREAM=True)
        p, cap = 200, 256
        eng = Engine(spec, W, cap=cap, cfg=cfg, rows=1, resident=True, experts=6,
                     wformat="fp4", layer_major=T, expert_major=True)
    toks = [int(t) for t in np.random.default_rng(p).integers(0, 1000, p + 1)]
    eng.prefill(toks[:p])
    img = eng.image
    progs, ra = tiled.compile_layer_tile(img, li, p // 256 + 1, T, 256)
    K = getattr(spec, "conv_k", 1)
    args = arg_words(ra, dict(RunPos.values(toks[p], p, K, 256), row=3))
    dram = eng.backend.machine.slices[0].dram.copy()
    served = img.offload.served
    dram[served:served + 4] = np.frombuffer(np.float32(1e6).tobytes(), np.uint8)
    m = Machine(cfg, progs, [dram.copy()], args=args).run()
    drams, tmems, _ = rtlsim.run(cfg, progs, [dram.copy()], uarch=rtlsim.BOARD_UARCH, axi=True,
                                 boot=True, args=args)
    n = img.nbytes
    assert np.array_equal(drams[0][:n], m.slices[0].dram[:n])
    assert np.array_equal(tmems[0], m.slices[0].tmem)


def test_reference_key_tells_tiled_runs():
    """tools/qual/refs.py: OTPU_TILE is in a reference's key as it is set (env_key: the hashed
    sources read it). Unset, the default (tiled prompt runs), it is not: references computed
    with the default are the card's default's. OTPU_TILE=0 (today's prompt runs) and =1 (a MoE
    model's tiled layer runs too) are keys of their own."""
    import importlib.util
    import os
    root = Path(__file__).resolve().parents[1]
    sp = importlib.util.spec_from_file_location("qual_refs_t", root / "tools/qual/refs.py")
    refs = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(refs)
    old = os.environ.pop("OTPU_TILE", None)
    try:
        assert "OTPU_TILE" not in refs.env_key()
        os.environ["OTPU_TILE"] = "1"
        assert refs.env_key()["OTPU_TILE"] == "1"
        os.environ["OTPU_TILE"] = "0"
        assert refs.env_key()["OTPU_TILE"] == "0"
    finally:
        os.environ.pop("OTPU_TILE", None)
        if old is not None:
            os.environ["OTPU_TILE"] = old
