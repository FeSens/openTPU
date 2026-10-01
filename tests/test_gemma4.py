"""Gemma 4 on openTPU: sliding and global attention with a KV ring and shared KV layers, GeGLU,
per-layer embeddings gathered on the device, against Hugging Face transformers. A tiny random
model always runs; the real model's programs are compiled when its config is in
models/gemma-4-E2B."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from opentpu.compiler import CompileError
from opentpu.isasim import board_config
from opentpu.kernels import gather as GA
from opentpu.llm import gemma4 as G
from opentpu.llm.qwen3 import Engine

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.gemma4")

REAL = Path(__file__).resolve().parent.parent / "models" / "gemma-4-E2B"
REAL_E4B = REAL.parent / "gemma-4-E4B"
# (two own sliding layers and a global one) x 2, a shared sliding one and two shared globals
# (HF forces the last layer global): layers 6..8 read layer 4's (sliding) and 5's (global) K / V.
# The layer loops: (s s f) x 2 with the two sliding layers a loop inside, s, then f x 2
S_, F_ = "sliding_attention", "full_attention"
KINDS = (S_, S_, F_, S_, S_, F_, S_, F_, F_)


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """head_dim 128 (sliding) and 256 (global, 32 rotated pairs), 8 query heads on 1 KV head
    (two parts of 4 on the board's MXU), a double-wide MLP in the shared layers, a 128-wide
    per-layer input. k_norm near the real model's 0.13 (attention is not scaled by 1/sqrt(d):
    with k_norm ~1 the scores are ~10x larger and int8 K dominates the error)."""
    torch.manual_seed(0)
    hc = transformers.Gemma4TextConfig(
        hidden_size=256, num_hidden_layers=len(KINDS), num_attention_heads=8,
        num_key_value_heads=1, head_dim=128, global_head_dim=256, intermediate_size=512,
        vocab_size=1000, vocab_size_per_layer_input=1000, hidden_size_per_layer_input=128,
        layer_types=list(KINDS), num_kv_shared_layers=3, use_double_wide_mlp=True,
        sliding_window=512, final_logit_softcapping=30.0, max_position_embeddings=4096)
    hc._attn_implementation = "eager"
    m = transformers.models.gemma4.modeling_gemma4.Gemma4ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                k = "k_norm" in n
                p.copy_((0.13 if k else 1.0) + (0.01 if k else 0.1) * torch.randn_like(p))
        for n, b in m.named_buffers():
            if n.endswith("layer_scalar"):
                b.copy_(0.5 + torch.rand_like(b))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    d = tmp_path_factory.mktemp("gemma4")
    (d / "config.json").write_text(json.dumps(hc.to_dict()))
    return m, W, G.Spec.from_hf(d)


def _cfg():
    return board_config(DRAM_BYTES=1 << 26)


def test_spec_kv_sources(tiny):
    _, _, spec = tiny
    assert spec.kinds == ("sliding", "sliding", "full") * 2 + ("sliding", "full", "full")
    assert spec.kv_src == (0, 1, 2, 3, 4, 5, 4, 5, 5)
    assert spec.ffn == (512,) * 6 + (1024,) * 3
    assert spec.global_rot == 32
    img = spec.image(_cfg(), 1024)
    assert [(f, len(u), r) for f, u, r in img.runs] == [(0, 3, 2), (6, 1, 1), (7, 1, 2)]
    assert [(e0, len(su), r) for e0, su, r, _ in img.subs[0]] == [(0, 1, 2), (2, 1, 1)]


def test_reference_matches_hf(tiny):
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 32)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(G.softcap(spec, G.reference_logits(spec, W, toks)) - hf).max() < 1e-4


def test_tiny_matches_hf_and_emulation(tiny):
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 24)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    eng = Engine(spec, W, cap=1024, cfg=_cfg())
    dev = np.array([G.softcap(spec, eng.step(t)) for t in toks])
    assert _cos(dev, hf).min() > 0.985          # int8 noise of a random model (emulation 0.991)
    emu = G.softcap(spec, G.emulated_logits(spec, W, toks[:12]))
    assert _cos(dev[:12], emu).min() > 0.99


MIX = "attn@0-2=fp4,down@3-5=fp4,gateup@6-8=fp4,ple@6-8=fp4"


def test_per_layer_formats(tiny):
    """Weight formats per layer and kind (Spec.formats, docs/gemma4_e4b.md): a layer block's
    layout follows its formats, so the loops split where they change. A 4-row prefill equals
    token-by-token decoding bit for bit, and the device follows the emulation of the same
    formats, closer than the all-int8 emulation."""
    _, W, spec = tiny
    mix = replace(spec, formats=MIX)
    img = mix.image(_cfg(), 1024)
    assert [(f, len(u), r) for f, u, r in img.runs] == [(0, 1, 2), (2, 1, 1), (3, 1, 2),
                                                         (5, 1, 1), (6, 1, 1), (7, 1, 2)]
    assert img.lf[:3] == (("fp4", "int8", "int8", "int8"),) * 3
    assert img.lf[3:6] == (("int8", "int8", "fp4", "int8"),) * 3
    assert img.lf[6:] == (("int8", "fp4", "int8", "fp4"),) * 3 and img.pformat == "int8"
    assert img.nbytes < spec.image(_cfg(), 1024).nbytes
    with pytest.raises(ValueError, match="weight format"):
        G.layer_formats(spec, "int8", "mlp@0-2=fp8")
    toks = [int(t) for t in np.random.default_rng(8).integers(0, 1000, 12)]
    a = Engine(mix, W, cap=1024, cfg=_cfg())
    b = Engine(mix, W, cap=1024, cfg=_cfg())
    dev = np.array([G.softcap(spec, b.step(t)) for t in toks])
    assert np.array_equal(G.softcap(spec, a.prefill(toks, chunk=4)), dev[-1])
    emu = G.softcap(spec, G.emulated_logits(mix, W, toks))
    e8 = G.softcap(spec, G.emulated_logits(spec, W, toks))
    assert _cos(dev, emu).min() > 0.99
    assert np.abs(dev - emu).mean() < 0.7 * np.abs(dev - e8).mean()


def test_formats_by_fit(tiny, monkeypatch):
    """The formats follow the fit (Spec.fit_formats, docs/gemma4_e4b.md): an int8 image that
    fits the card beside no PLE choice takes the head and the own-KV layers' down projections
    in fp4 (one loop's format, so no extra loop); an image that fits, or not int8, is as asked.
    The Engine's compile worker gets the choices (Engine._image_kw): its image is the same
    without choosing again."""
    _, W, spec = tiny
    assert spec.fit_formats == "head=fp4,down@0-5=fp4"
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    big = spec.image(_cfg(), 1024, ple_host=True, formats="")
    mix = spec.image(_cfg(), 1024, ple_host=True, formats=spec.fit_formats)
    assert spec.image(_cfg(), 1024).formats == ""
    monkeypatch.setattr(G, "CARD_BYTES", (big.nbytes + mix.nbytes) // 2)
    img = spec.image(_cfg(), 1024)
    assert img.formats == spec.fit_formats and img.head_format == "fp4" and img.ple_host
    assert img.lf[5] == ("int8", "int8", "fp4", "int8") and img.lf[6] == ("int8",) * 4
    assert img.nbytes == mix.nbytes and img.runs == mix.runs and len(img.runs) == len(big.runs)
    assert spec.image(_cfg(), 1024, wformat="fp4").formats == ""
    eng = Engine(spec, W, cap=1024, cfg=_cfg())
    kw = eng._image_kw
    assert kw["formats"] == spec.fit_formats and kw["ple_host"] and kw["head_format"] == "fp4"
    monkeypatch.setattr(G, "CARD_BYTES", 1 << 32)
    again = spec.image(eng.cfg, 1024, 1, eng.rows, **kw)
    assert (again.nbytes, again.lf, again.head_format) == (eng.image.nbytes, eng.image.lf, "fp4")


@pytest.mark.parametrize("ple", ["int8", "fp4"])
def test_records_roundtrip(ple):
    """pack_records / dequant_records: the device's gather values of a packed table."""
    rows = np.random.default_rng(1).standard_normal((5, 896)).astype(np.float32)
    S = GA.record_blocks(7, ple)
    assert S % 2 == 1 or S % 4 == 2
    back = GA.dequant_records(GA.pack_records(rows, ple, 128, S), ple, 128, S)[:, :896]
    assert np.abs(back - rows).max() / np.abs(rows).max() < (0.005 if ple == "int8" else 0.15)


@pytest.mark.parametrize("wf,ple,fm", [("int8", "int8", ""), ("fp4", "fp4", ""),
                                       ("int8", "int8", MIX)], ids=["int8", "fp4", "mix"])
def test_resident_gathers_are_bit_exact(tiny, wf, ple, fm, monkeypatch):
    """Resident decode gathers the embedding and PLE rows on the device and computes its
    attention masks at the run-time position: the logits equal the per-position programs'
    (host-written rows), bit for bit, across the first bucket boundary; also with formats per
    layer (OTPU_FORMATS)."""
    _, W, spec = tiny
    monkeypatch.setenv("OTPU_PLE_FORMAT", ple)
    monkeypatch.setenv("OTPU_FORMATS", fm)
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 300)]
    a = Engine(spec, W, cap=1024, cfg=_cfg(), wformat=wf, resident=True)
    b = Engine(spec, W, cap=1024, cfg=_cfg(), wformat=wf)
    assert a.resident and not b.resident and a.image.ple_format == ple
    a.prefill(toks[:250])
    b.prefill(toks[:250])
    for t in toks[250:]:
        assert np.array_equal(a.step(t), b.step(t))


def test_ring_wrap_resident_is_bit_exact(tiny):
    """Positions 760..775: the sliding window's start and end blocks, across the ring's wrap
    (768 slots), resident against per-position."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 776)]
    a = Engine(spec, W, cap=1024, cfg=_cfg(), resident=True)
    b = Engine(spec, W, cap=1024, cfg=_cfg())
    la, lb = a.prefill(toks[:760]), b.prefill(toks[:760])
    assert np.array_equal(la, lb)
    for t in toks[760:]:
        assert np.array_equal(a.step(t), b.step(t))


def test_chunked_prefill_is_bit_exact(tiny):
    """Prefill runs of several rows give the logits and caches of token-by-token decoding."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 11)]
    a = Engine(spec, W, cap=1024, cfg=_cfg())
    b = Engine(spec, W, cap=1024, cfg=_cfg())
    la = a.prefill(toks, chunk=4)
    lb = b.prefill(toks, chunk=1)
    assert np.array_equal(la, lb)
    img = a.image
    da = a.backend.machine.slices[0].dram[img.layer0:img.head[0]]
    db = b.backend.machine.slices[0].dram[img.layer0:img.head[0]]
    assert np.array_equal(da, db)


def test_device_inputs_are_bit_exact(tiny):
    """A prefill run and a per-position program with their tokens compiled in gather the
    embedding and PLE rows and load the RoPE rows on the device (the host writes nothing):
    logits and caches equal those of the programs with host-written inputs (host_inputs)."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(6).integers(0, 1000, 14)]
    engs = [Engine(spec, W, cap=1024, cfg=_cfg(), wformat="fp4", head_format="int8",
                   resident=True) for _ in range(2)]
    for e in engs:
        e.prefill(toks[:9], chunk=4)
    img, V = engs[0].image, spec.vocab

    def run(e, rows, tokens, device):
        if not device:
            for addr, v in e.image.host_inputs(tokens, [p for _, p in rows]):
                e.backend.write(0, addr, v)
        e.backend.run(e.image.compile_rows(rows, [len(rows) - 1],
                                           tokens=tokens if device else None))
        return e.backend.read(0, img.io["logits"] + 4 * V * (len(rows) - 1), 4 * V)

    for rows, tk in (([(0, 9 + r) for r in range(4)], toks[9:13]), ([(0, 13)], toks[13:])):
        assert np.array_equal(run(engs[0], rows, tk, False), run(engs[1], rows, tk, True))
    lo, hi = img.layer0, img.head[0]
    assert np.array_equal(engs[0].backend.machine.slices[0].dram[lo:hi],
                          engs[1].backend.machine.slices[0].dram[lo:hi])


@pytest.mark.slow
@pytest.mark.parametrize("resident,host", [(False, "0"), (True, "0"), (True, "1")],
                         ids=["per-position", "resident", "resident-ple-host"])
def test_token_on_board_rtl_is_bit_exact(have_verilator, tiny, resident, host, monkeypatch):
    """A token at position 600 (the window's first block masked at its start) on the Verilator
    RTL through the board's memory path (AXI adapter, boot loader) equals the ISA simulator bit
    for bit, logits and the whole DRAM image (caches); resident, with the device's gathers and
    run-time masks; and with the PLE table on the host (the slot the host writes, the fence
    on its mailbox: WAITW holding at its first read)."""
    from opentpu import rtlsim
    from opentpu.llm.rtl_backend import RtlBackend
    _, W, spec = tiny
    monkeypatch.setenv("OTPU_PLE_HOST", host)
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 601)]
    eng = Engine(spec, W, cap=1024, cfg=_cfg(), wformat="fp4", head_format="int8",
                 resident=resident)
    assert eng.resident == resident
    eng.prefill(toks[:-1])
    n, isa = eng.image.nbytes, eng.backend
    rtl = RtlBackend(eng.cfg, [s.dram[:n] for s in isa.machine.slices],
                     uarch=rtlsim.BOARD_UARCH, axi=True, boot=True)
    want = eng.step(toks[-1])
    eng.backend, eng.pos = rtl, eng.pos - 1
    got = eng.step(toks[-1])
    assert np.array_equal(want.view(np.uint32), got.view(np.uint32))
    assert np.array_equal(isa.machine.slices[0].dram[:n], rtl.drams[0][:n])


@pytest.mark.parametrize("resident", [False, True], ids=["per-position", "resident"])
def test_ple_on_host_is_bit_exact(tiny, resident, monkeypatch):
    """The PLE table on the host (docs/gemma4_e4b.md): the image holds a slot that the Engine
    writes the run's records into (Image.host_rows, from the host's store) before each prefill
    run and step, and the gathers read it at the row. Logits and caches equal those of the
    table on the card, bit for bit."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(7).integers(0, 1000, 36)]
    engs = []
    for host in ("0", "1"):
        monkeypatch.setenv("OTPU_PLE_HOST", host)
        engs.append(Engine(spec, W, cap=1024, cfg=_cfg(), wformat="fp4", head_format="int8",
                           resident=resident))
    a, b = engs
    assert b.image.ple_host and not a.image.ple_host and b.resident == resident
    assert b.image.ple_store.shape == (spec.vocab, b.image.ple_rec)
    assert b.image.nbytes < a.image.nbytes - (spec.vocab - b.image.rows) * b.image.ple_rec + 4096
    assert np.array_equal(a.prefill(toks[:30], chunk=4), b.prefill(toks[:30], chunk=4))
    for t in toks[30:]:
        assert np.array_equal(a.step(t), b.step(t))
    lo, hi = a.image.layer0, a.image.head[0]
    assert (lo, hi) == (b.image.layer0, b.image.head[0])
    assert np.array_equal(a.backend.machine.slices[0].dram[lo:hi],
                          b.backend.machine.slices[0].dram[lo:hi])


@pytest.mark.parametrize("split", [None, True], ids=["one", "split"])
def test_generate_with_ple_on_host(tiny, split, monkeypatch):
    """The card's generate loop with the PLE table on the host (docs/gemma4_e4b.md): after
    sampling, each token's id goes to the image's PLE mailbox (offload's format), the host's
    RowServer (here the ISA simulator's WAITW hook; on the card the backend's poll) writes
    its record into slot row 0, and the next token's step waits for that (WAITW served >= seq)
    before its gather. The tokens equal those of the table on the card, greedy across the
    bucket boundary at 256 (one program, or split) and sampled, and a host-written step after
    the loop (its last request served first) gives the same logits."""
    from opentpu.llm import generate as GEN
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 248)]
    engs = []
    for host in ("0", "1"):
        monkeypatch.setenv("OTPU_PLE_HOST", host)
        e = Engine(spec, W, cap=1024, cfg=_cfg(), wformat="fp4", head_format="int8",
                   resident=True)
        e.gen_split = split
        engs.append(e)
    a, b = engs
    assert a.row_server is None and b.row_server is not None and b.can_generate
    b.row_server.history = []
    t0 = int(np.argmax(a.prefill(toks)))
    assert int(np.argmax(b.prefill(toks))) == t0
    ref = a.generate_card(t0, 12, stop_ids=[])
    assert b.generate_card(t0, 12, stop_ids=[]) == ref and len(set(ref)) > 3
    h = b.row_server.history
    assert len(h) >= 11 and h == ref[:len(h)]           # each sampled token, in order
    samp = GEN.Sampling(0.8, 5, 0.9, 1.1)
    ctx = toks + [t0] + ref
    got = [e.generate_card(ref[-1], 8, stop_ids=[], sampling=samp, context=ctx,
                           rng=np.random.default_rng(7)) for e in (a, b)]
    assert got[0] == got[1]
    assert np.array_equal(a.step(got[0][-1]), b.step(got[1][-1]))
    assert b.row_server.seq == len(ref) + len(got[1])


def test_ple_rows_served_during_a_live_cards_runs(tiny, monkeypatch):
    """The host's side as it runs beside a card: BoardBackend polls the RowServer while a run is
    in flight, over a fake card that computes in a thread on the host's DRAM
    (tests/test_lfm2_moe.py's _LiveCard), so the generate loop's PLE fences wait for the host's
    writes during the run (here the host is the faster: they hold at their first read).
    Prefill, resident steps and the generate loop give the ISA simulator's logits and tokens
    bit for bit."""
    from test_lfm2_moe import _LiveCard
    from opentpu.host.board import BoardBackend
    _, W, spec = tiny
    monkeypatch.setenv("OTPU_PLE_HOST", "1")
    cfg = _cfg()
    card = _LiveCard.make(cfg)
    kw = dict(cap=1024, cfg=cfg, wformat="fp4", head_format="int8", resident=True)
    isa = Engine(spec, W, **kw)
    brd = Engine(spec, W, **kw, backend=lambda c, imgs: BoardBackend(c, imgs, transport=card))
    assert brd.can_generate and brd.backend.host is not None and brd.image.ple_host
    toks = [int(t) for t in np.random.default_rng(9).integers(0, 1000, 6)]
    a, b = isa.prefill(toks), brd.prefill(toks)
    assert card.error is None, card.error
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    t0 = int(np.argmax(a))
    for e in (isa, brd):
        e.step(t0)
    got = brd.generate_card(t0, 8, stop_ids=[])
    assert card.error is None, card.error
    assert got == isa.generate_card(t0, 8, stop_ids=[])
    assert brd.row_server.seq >= 7 and isa.row_server.seq >= 7    # served during the run:
                                                                  # else a fence never holds


def test_one_sequence_one_slice(tiny):
    _, W, spec = tiny
    with pytest.raises(ValueError, match="one sequence"):
        Engine(spec, W, cap=256, cfg=_cfg(), batch=2)
    from opentpu.isasim import design_config
    with pytest.raises(ValueError, match="one slice"):
        spec.image(design_config(), 256)


@pytest.mark.skipif(not (REAL / "config.json").exists(), reason="no models/gemma-4-E2B")
def test_real_model_programs_fit():
    """Gemma 4 E2B on the board: fp4 layers, int8 LM head and PLE fit 4 GiB at 4096 tokens;
    the resident decode programs fit IMEM and take 6 argument words."""
    spec = G.Spec.from_hf(REAL)
    img = spec.image(board_config(), 4096, 1, 8, "fp4", "int8", lookup=True)
    assert img.ple_format == "int8" and img.nbytes < 1 << 32
    assert spec.image(board_config(), 4096, 1, 8, lookup=True).formats == ""   # int8 fits
    for blocks in (1, 3, 16):
        progs, ra = img.compile_decode(blocks, (blocks - 1) * 256)
        assert 8 * len(progs[0]) <= board_config().IMEM_WORDS
        assert len(ra) == 6
    with pytest.raises(CompileError, match="TMEM"):
        img.compile_rows([(0, p) for p in range(8)], [7])


@pytest.mark.skipif(not (REAL_E4B / "config.json").exists(), reason="no models/gemma-4-E4B")
def test_e4b_programs_fit():
    """Gemma 4 E4B on the board: with fp4 layers and the int8 head its PLE table does not fit
    beside them in either format, so the image keeps it on the host (int8 records) and fits
    4 GiB at 2048 tokens; the resident decode programs fit IMEM with 5 argument words (the PLE
    gather reads the slot, not the token's row); 4-row prefill runs fit TMEM."""
    spec = G.Spec.from_hf(REAL_E4B)
    img = spec.image(board_config(), 2048, 1, 8, "fp4", "int8", lookup=True)
    assert img.ple_host and img.ple_format == "int8" and img.nbytes < 3 << 30
    for blocks in (1, 8):
        progs, ra = img.compile_decode(blocks, (blocks - 1) * img.block)
        assert 8 * len(progs[0]) <= board_config().IMEM_WORDS
        assert len(ra) == 5
    img.compile_rows([(0, r) for r in range(4)], [3], img.block, tokens=[5, 6, 7, 8])
    # int8 layers do not fit: the formats by fit, two loops, decode and 4-row runs fit
    mix = spec.image(board_config(), 2048, 1, 8, lookup=True)
    assert mix.formats == "head=fp4,down@0-23=fp4" and mix.head_format == "fp4"
    assert mix.ple_host and 3.9 * 2**30 < mix.nbytes < 1 << 32 and len(mix.runs) == 2
    progs, ra = mix.compile_decode(8, 7 * mix.block)
    assert 8 * len(progs[0]) <= board_config().IMEM_WORDS
    mix.compile_rows([(0, r) for r in range(4)], [3], mix.block, tokens=[5, 6, 7, 8])
