"""LFM2 on openTPU: the hybrid decoder (short-conv and GQA attention layers, 64-wide heads
padded to the MXU depth, conv state ring in DRAM) against Hugging Face transformers. A tiny
random model always runs; the real LFM2.5-230M runs when its checkpoint is in
models/LFM2.5-230M."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from opentpu.isasim import board_config
from opentpu.llm import load_spec
from opentpu.llm.lfm2 import Spec, emulated_logits, plan, reference_logits
from opentpu.llm.qwen3 import Engine, load_weights

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

REAL = Path(__file__).resolve().parent.parent / "models" / "LFM2.5-230M"
KINDS = ("conv", "attn", "conv", "attn", "conv")       # a (conv, attn) loop, then a conv


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def _chat_ids(tok, text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                  add_generation_prompt=True, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    hc = transformers.Lfm2Config(
        hidden_size=256, num_hidden_layers=len(KINDS), num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=512, vocab_size=1000, norm_eps=1e-5,
        layer_types=["full_attention" if k == "attn" else "conv" for k in KINDS],
        conv_L_cache=3, conv_bias=False, block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True, max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": 1e6})
    m = transformers.Lfm2ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    return m, W, Spec(256, KINDS, 4, 2, 64, 512, 1000)


def test_from_hf_ffn_as_hf_and_no_rope_scaling(tmp_path):
    """Spec.from_hf sizes the MLP as Hugging Face's Lfm2MLP (with block_auto_adjust_ff_dim:
    2/3, the multiplier, rounded up to block_multiple_of) and refuses a RoPE scaling (YaRN,
    linear: Hugging Face rescales the frequencies)."""
    import json
    from transformers.models.lfm2.modeling_lfm2 import Lfm2MLP

    def config(**kw):
        return transformers.Lfm2Config(
            hidden_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=1000, vocab_size=1000, layer_types=["conv", "full_attention"],
            **kw)

    def spec(hc, **kw):
        (tmp_path / "config.json").write_text(json.dumps(dict(hc.to_dict(), **kw)))
        return Spec.from_hf(tmp_path)
    for kw in (dict(block_auto_adjust_ff_dim=False),
               dict(block_auto_adjust_ff_dim=True, block_ffn_dim_multiplier=1.0),
               dict(block_auto_adjust_ff_dim=True, block_ffn_dim_multiplier=1.5,
                    block_multiple_of=256)):
        hc = config(**kw)
        assert spec(hc).ffn == Lfm2MLP(hc).w1.out_features, kw
    yarn = {"rope_type": "yarn", "factor": 4.0, "rope_theta": 1e6}
    for kw in (dict(rope_parameters=yarn),
               dict(rope_parameters=None, rope_scaling={"type": "linear", "factor": 2.0})):
        with pytest.raises(ValueError, match="RoPE type"):
            spec(config(), **kw)


def test_plan_loops_the_repeated_unit():
    real = ("conv", "conv") + ("attn", "conv") * 6
    assert plan(real) == [(0, ("conv",), 1), (1, ("conv", "attn"), 6), (13, ("conv",), 1)]
    assert plan(("attn",) * 28) == [(0, ("attn",), 28)]
    assert plan(("conv", "attn")) == [(0, ("conv",), 1), (1, ("attn",), 1)]
    assert plan(KINDS) == [(0, ("conv", "attn"), 2), (4, ("conv",), 1)]


def test_tiny_matches_hf(tiny):
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 140)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(reference_logits(spec, W, toks) - hf).max() < 1e-4
    eng = Engine(spec, W, cap=256)
    dev = np.array([eng.step(t) for t in toks])
    assert _cos(dev, hf).min() > 0.998
    # the device follows the quantized math (it differs only in fp32 rounding)
    emu = emulated_logits(spec, W, toks[:12])
    assert _cos(dev[:12], emu).min() > 0.9995


def test_tiny_fp4_follows_emulation(tiny):
    """4-bit (FP4) weights: the device follows the float64 emulation of the same weights."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 12)]
    eng = Engine(spec, W, cap=256, wformat="fp4")
    dev = np.array([eng.step(t) for t in toks])
    assert _cos(dev, emulated_logits(spec, W, toks, wformat="fp4")).min() > 0.9995


def test_tiny_formats_per_kind(tiny, monkeypatch):
    """Weight formats per kind (as test_llama's), the convolutions' in / out projections a
    kind of their own: the device follows the emulation of the same formats; resident decode
    (its own int8 embedding table beside the fp4 head) and a chunked prefill give
    token-by-token decoding's logits bit for bit."""
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    _, W, spec = tiny
    mix = dataclasses.replace(spec, formats="conv=fp4,attn=int4,down=fp4,head=fp4")
    a = Engine(mix, W, cap=256)
    assert a.image.mf == dict(win="fp4", wout="fp4", wq="int4", wk="int4", wv="int4",
                              wo="int4", wg="int8", wu="int8", wd="fp4")
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 12)]
    dev = np.array([a.step(t) for t in toks])
    emu, e8 = emulated_logits(mix, W, toks), emulated_logits(spec, W, toks)
    assert _cos(dev, emu).min() > 0.9995
    assert np.abs(dev - emu).mean() < 0.7 * np.abs(dev - e8).mean()
    r, b = Engine(mix, W, cap=256, resident=True), Engine(mix, W, cap=256)
    assert r.resident
    for t, want in zip(toks[:4], dev):
        assert np.array_equal(r.step(t).view(np.uint32), want.view(np.uint32))
    assert np.array_equal(b.prefill(toks, chunk=4).view(np.uint32), dev[-1].view(np.uint32))


def test_tiny_formats_by_layer_range(tiny, monkeypatch):
    """Weight formats per layer range (formats.py's kind@a-b): a layer block layout per
    formats group, the layers in runs of one unit of (kind, group) keys (plan: a hardware loop
    each, over blocks of its group's size): the device follows the emulation of the same
    formats; resident decode and a chunked prefill give token-by-token decoding's logits bit
    for bit, and decoding goes on from the prefill's KV cache and conv state."""
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    _, W, spec = tiny
    # layer 0's MLP in fp4, then a loop over (attn, conv) of 4-bit mixers: not layer 0's layout
    mix = dataclasses.replace(spec, formats="mlp@0=fp4,conv@1-4=fp4,attn@1-4=int4")
    cfg = board_config(DRAM_BYTES=1 << 26)
    a = Engine(mix, W, cap=256, cfg=cfg)
    img = a.image
    g0, g = ("int8", "int8", "fp4", "fp4"), ("fp4", "int4", "int8", "int8")
    assert img.lf == (g0,) + (g,) * 4
    assert img.plan == [(0, (("conv", g0),), 1), (1, (("attn", g), ("conv", g)), 2)]
    assert img.layouts[g0].size != img.layouts[g].size
    assert [img._off(i).const for i in range(5)] == sorted(img._off(i).const for i in range(5))
    assert mix.image(cfg, 256).nbytes < spec.image(cfg, 256).nbytes
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 12)]
    dev = np.array([a.step(t) for t in toks])
    emu, e8 = emulated_logits(mix, W, toks), emulated_logits(spec, W, toks)
    assert _cos(dev, emu).min() > 0.9995
    assert np.abs(dev - emu).mean() < 0.7 * np.abs(dev - e8).mean()
    r, b = Engine(mix, W, cap=256, cfg=cfg, resident=True), Engine(mix, W, cap=256, cfg=cfg)
    assert r.resident
    for t, want in zip(toks[:4], dev):
        assert np.array_equal(r.step(t).view(np.uint32), want.view(np.uint32))
    assert np.array_equal(b.prefill(toks[:8], chunk=4).view(np.uint32), dev[7].view(np.uint32))
    for t, want in zip(toks[8:], dev[8:]):
        assert np.array_equal(b.step(t).view(np.uint32), want.view(np.uint32))


def test_tiny_reset_reuses_cache_and_conv_state(tiny):
    """After reset, positions 0 and 1 must not read the previous sequence's conv state."""
    _, W, spec = tiny
    eng = Engine(spec, W, cap=128)
    a = [eng.step(t) for t in (5, 6, 7, 8)]
    eng.reset()
    b = [eng.step(t) for t in (5, 6, 7, 8)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_one_sequence_only(tiny):
    _, W, spec = tiny
    with pytest.raises(ValueError, match="one sequence"):
        Engine(spec, W, cap=128, batch=2)


def _layers_dram(eng):
    """The layer blocks (weights, KV cache, conv state) of every slice: all but the I/O area."""
    img = eng.image
    return [s.dram[img.layer0:img.nbytes] for s in eng.backend.machine.slices]


@pytest.mark.parametrize("first,chunk", [(0, 4), (1, 3), (2, 8)])
def test_tiny_chunked_prefill_is_bit_exact(tiny, first, chunk):
    """Prefill in chunks (the convolution over the chunk's rows and the ring, causal attention
    over the cache and the chunk; a chunk may start at position 0, 1 or 2, before the ring is
    full) gives the same logits, KV cache and conv state as token-by-token decode, and
    decoding continues identically."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 23)]
    ref = Engine(spec, W, cap=256)
    want = [ref.step(t) for t in toks]
    eng = Engine(spec, W, cap=256)
    for t in toks[:first]:
        eng.step(t)
    got = eng.prefill(toks[first:21], chunk=chunk)
    assert np.array_equal(got.view(np.uint32), want[20].view(np.uint32)) and eng.pos == 21
    assert eng.stats[first]["rows"] == chunk
    ref21 = Engine(spec, W, cap=256)
    ref21.prefill(toks[:21], chunk=1)
    assert all(np.array_equal(a, b) for a, b in zip(_layers_dram(eng), _layers_dram(ref21)))
    assert all(np.array_equal(eng.step(t), w) for t, w in zip(toks[21:], want[21:]))


@pytest.fixture(scope="module")
def tiny_wide():
    """The tiny model with a wider MLP: 1792, 14 int8 chunks of 128 or 7 4-bit ones of 256."""
    torch.manual_seed(1)
    hc = transformers.Lfm2Config(
        hidden_size=256, num_hidden_layers=len(KINDS), num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=1792, vocab_size=1000, norm_eps=1e-5,
        layer_types=["full_attention" if k == "attn" else "conv" for k in KINDS],
        conv_L_cache=3, conv_bias=False, block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True, max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": 1e6})
    m = transformers.Lfm2ForCausalLM(hc).float().eval()
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    return W, Spec(256, KINDS, 4, 2, 64, 1792, 1000)


@pytest.mark.parametrize("wformat", ["int8", "fp4"])
def test_tiny_mlp_loop_is_bit_exact(tiny_wide, monkeypatch, wformat):
    """The MLP's F chunks as a hardware loop (lfm2.MLP_UNROLL_BODIES: LFM2-2.6B's plan), an
    even and an odd number of chunks: the unrolled programs' logits, KV cache and conv state
    bit for bit, per position, resident and in chunked prefill, in fewer instructions."""
    import opentpu.llm.lfm2 as L
    W, spec = tiny_wide
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 30)]
    cfg = board_config(DRAM_BYTES=1 << 24)
    b = Engine(spec, W, cap=256, cfg=cfg, wformat=wformat)
    monkeypatch.setattr(L, "MLP_UNROLL_BODIES", 0)
    a = Engine(spec, W, cap=256, cfg=cfg, wformat=wformat)
    r = Engine(spec, W, cap=256, cfg=cfg, wformat=wformat, resident=True)
    assert a.image.mlp_loop and r.resident and not b.image.mlp_loop
    n = [len(e.image.compile_step(9)[0]) for e in (a, b)]
    assert n[0] < n[1] - 20, n
    want = b.prefill(toks[:21], chunk=8)
    for e in (a, r):
        assert np.array_equal(e.prefill(toks[:21], chunk=8).view(np.uint32),
                              want.view(np.uint32))
    for t in toks[21:]:
        want = b.step(t)
        for e in (a, r):
            assert np.array_equal(e.step(t).view(np.uint32), want.view(np.uint32))
    assert all(np.array_equal(x, y) for x, y in zip(_layers_dram(a), _layers_dram(b)))


def test_tiny_mlp_loop_on_rtl(tiny_wide, have_verilator, monkeypatch):
    """The looped MLP (its two gate / up buffers reused across iterations) on the Verilator
    RTL: a resident decode step and a 5-row prefill run with its inputs from the image's
    tables, bit-identical to the ISA simulator, DRAM included."""
    import opentpu.llm.lfm2 as L
    from opentpu.llm.rtl_backend import RtlBackend
    monkeypatch.setattr(L, "MLP_UNROLL_BODIES", 0)
    W, spec = tiny_wide
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 12)]
    eng = Engine(spec, W, cap=256, cfg=board_config(DRAM_BYTES=1 << 24), resident=True)
    assert eng.image.mlp_loop and eng.resident and eng.device_inputs
    eng.prefill(toks[:6])
    for run in (toks[6:7], toks[7:12]):
        n = eng.image.nbytes
        rtl = RtlBackend(eng.cfg, [s.dram[:n] for s in eng.backend.machine.slices])
        isa, p = eng.backend, eng.pos
        want = eng.prefill(run, chunk=len(run))
        eng.backend, eng.pos = rtl, p
        got = eng.prefill(run, chunk=len(run))
        eng.backend = isa
        assert np.array_equal(want.view(np.uint32), got.view(np.uint32)), p
        assert np.array_equal(isa.machine.slices[0].dram[:n], rtl.drams[0][:n])


@pytest.mark.parametrize("resident,formats", [(False, ""), (True, ""),
                                             (True, "conv=fp4,attn=int4,down=fp4"),
                                             (True, "mlp@0=fp4,conv@1-4=fp4,attn@1-4=int4")])
def test_tiny_lfm2_on_board_model(tiny, have_verilator, resident, formats):
    """The board model through the host driver, through a full turn of the conv state ring:
    logits bit-identical to the ISA simulator. Resident: from position 2 on one program takes
    the token and position in the ARG registers (CAPS bit25). formats: per-kind weight formats
    (int8, fp4 and int4 MMs in one model), per layer range (a loop over another layout)."""
    from opentpu.host.board import Board, BoardBackend, SimTransport
    _, W, spec = tiny
    spec = dataclasses.replace(spec, formats=formats)
    cfg = board_config(DRAM_BYTES=1 << 23)
    isa = Engine(spec, W, cap=256, cfg=cfg)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5)
    brd = Engine(spec, W, cap=256, cfg=cfg, resident=resident,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    assert brd.resident == resident
    loads, load = [], Board.load_program
    brd.backend.board.load_program = lambda *a: (loads.append(1), load(brd.backend.board, *a))
    for tok in (11, 222, 333, 444, 555):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert brd.stats[-1]["cycles"] > 0
    assert len(loads) == 5          # SimTransport: every run is a fresh simulation (loads again)
    if resident:
        assert sorted(brd._decodes) == [1] and brd.backend._resident[0] is brd._decodes[1][0]


def test_tiny_resident_decode_is_bit_exact(tiny):
    """Resident decode (one program per 256-position attention bucket, the token and position
    as run arguments, the inputs from the image's tables) gives the per-position programs'
    logits bit for bit: from the conv ring's first positions, after chunked prefills, and
    across the bucket boundaries 256 and 512; the KV cache and conv state agree too."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 530)]
    a = Engine(spec, W, cap=768, resident=True)
    b = Engine(spec, W, cap=768)
    assert a.resident and not b.resident
    p = 0
    for stop, run in ((0, 4), (250, 10), (508, 8)):
        if stop > p:
            assert np.array_equal(a.prefill(toks[p:stop]), b.prefill(toks[p:stop]))
            p = stop
        for t in toks[p:p + run]:
            pa = a.pos
            assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32)), pa
        p += run
    assert sorted(a._decodes) == [1, 2, 3] and not b._decodes
    ia, ib = a.image, b.image
    assert (ia.layer0, ia.loc) == (ib.layer0, ib.loc)
    ma, mb = (e.backend.machine.slices[0].dram[ib.layer0:ib.nbytes].copy() for e in (a, b))
    for li, k in enumerate(spec.kinds):     # the state rings' scratch rows (_ring_rows)
        if k == "conv":
            o = ib._off(li).const - ib.layer0 + ib.lofs["conv"]["state"]
            ma[o:o + 4 * ib.h_loc] = mb[o:o + 4 * ib.h_loc] = 0
    assert np.array_equal(ma, mb)


@pytest.mark.parametrize("head", ["int8", "fp4"])
def test_tiny_int8_embedding(tiny, head):
    """Spec.embed "int8" (e.g. LFM2.5-8B-A1B): the token's int8 embedding row is gathered on
    the device from the tied int8 LM head (under a 4-bit head, from an int8 table of the
    image's own), in the resident decode, the per-position programs and the prefill runs: the
    same logits bit for bit, following the float64 emulation, which quantizes the row too."""
    _, W, spec = tiny
    spec = dataclasses.replace(spec, embed="int8")
    cfg = board_config(DRAM_BYTES=1 << 25)
    wf = "int8" if head == "int8" else "fp4"
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 12)]
    a = Engine(spec, W, cap=256, cfg=cfg, resident=True, wformat=wf, head_format=head)
    b = Engine(spec, W, cap=256, cfg=cfg, wformat=wf, head_format=head)
    assert a.resident and b.device_inputs and a.image.lookup["own"] == (head != "int8")
    for t in toks[:3]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32))
    assert np.array_equal(a.prefill(toks[3:9]).view(np.uint32),
                          b.prefill(toks[3:9]).view(np.uint32))
    for t in toks[9:]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32))
    c = Engine(spec, W, cap=256, cfg=cfg, wformat=wf, head_format=head)
    dev = np.array([c.step(t) for t in toks])
    emu = emulated_logits(spec, W, toks, wformat=wf, head_format=head)
    assert _cos(dev, emu).min() > 0.999


@pytest.mark.parametrize("resident", [False, True])
def test_tiny_filling_step_programs_are_transparent(tiny, resident):
    """The step programs that fill their logits first (the card's streamed decode) give the
    plain programs' logits and DRAM bit for bit, per position and resident."""
    from conftest import assert_fill_is_transparent
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(8).integers(0, 1000, 6)]
    assert_fill_is_transparent(lambda: Engine(spec, W, cap=256, resident=resident), toks)


@pytest.mark.parametrize("resident", [False, True])
def test_tiny_streamed_decode_under_compile_contention(tiny, monkeypatch, resident):
    """The card's streamed decode where the programs fill their logits themselves
    (qwen3.fill_logits), at LFM2's shortest run (about 11 ms on the card), with the next
    programs compiling on a thread and another one holding the GIL while the run goes on
    (switches every 0.5 ms), the fill landing anywhere before the LM head's first piece: the
    logits are the ISA simulator's bit for bit, token after token, with pieces taken during
    the runs and probes held back until the fill. (mark-in-run, the host's in-run marking it
    replaces, lost this race on the card: docs/host.md.)"""
    import sys
    import threading
    import time

    from conftest import IsaCard
    from opentpu.host import regs as R
    from opentpu.host.board import BoardBackend, sim_config
    from opentpu.llm import qwen3 as Q
    monkeypatch.setattr(Q, "HEAD_CHUNK", 128)          # pieces of 128 logits: 8 in the vocab
    _, W, spec = tiny
    cfg = sim_config(spec, 256, lookup=resident)
    rng = np.random.default_rng(7)
    toks = [int(t) for t in rng.integers(0, 1000, 12)]
    ref = Engine(spec, W, cap=256, cfg=cfg, resident=resident)
    want = [ref.step(t) for t in toks]

    class Card(IsaCard):
        computing = False                               # the simulation is not the run
        held = 0                                        # ICOUNT reads before the fill

        def reg_write(self, off, val):
            self.computing = True
            try:
                super().reg_write(off, val)
            finally:
                self.computing = False

        def reg_read(self, off):
            v = super().reg_read(off)
            self.held += off == R.R_ICOUNT and v == 0
            return v

    # the fill lands anywhere before the first piece (0.4 of the run); every third run waits
    # for the host to look (IsaCard's anchor), its fill just before the first piece, so the
    # host's probes find it still on the way however late the host gets there
    card = Card(cfg, None, 4 * 128, run_s=0.011, gen=True, args=True,
                fill_at=lambda r: 0.39 if r % 3 == 1 else rng.uniform(0.01, 0.39),
                anchor=lambda r: r % 3 == 1)
    eng = Engine(spec, W, cap=256, cfg=cfg, pipeline="thread", resident=resident,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=card, model="tiny"))
    card.late = (eng.image.io["logits"], 4 * spec.vocab)
    assert eng.image.stream_fill and eng.resident == resident
    stop = threading.Event()

    def hog():
        x = 0
        while not stop.is_set():
            if card.computing:
                time.sleep(1e-3)
            x += 1
    switch = sys.getswitchinterval()
    sys.setswitchinterval(5e-4)
    hogs = [threading.Thread(target=hog, daemon=True)]
    for h in hogs:
        h.start()
    during = 0
    try:
        for t, w in zip(toks, want):
            got = eng.step(t)
            assert np.array_equal(w.view(np.uint32), got.view(np.uint32)), eng.pos
            during += eng.backend.last_stream["during"]
    finally:
        stop.set()
        for h in hogs:
            h.join()
        sys.setswitchinterval(switch)
        eng.backend.close()
    assert during > 0 and card.held > 0                 # pieces during the runs; the gate held


def test_tiny_resident_decode_on_rtl(tiny, have_verilator):
    """The resident decode program on the Verilator RTL (its arguments preset R8..R15) across
    the bucket boundary: positions 255 (bucket 1, the masked block's last column) and 256
    (bucket 2) bit-identical to the ISA simulator, DRAM included."""
    from opentpu.llm.rtl_backend import RtlBackend
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 257)]
    eng = Engine(spec, W, cap=512, resident=True)
    eng.prefill(toks[:255])
    for t in toks[255:257]:
        n = eng.image.nbytes
        rtl = RtlBackend(eng.cfg, [s.dram[:n] for s in eng.backend.machine.slices])
        isa = eng.backend
        want = eng.step(t)
        eng.backend, eng.pos = rtl, eng.pos - 1
        got = eng.step(t)
        eng.backend = isa
        assert np.array_equal(want.view(np.uint32), got.view(np.uint32)), eng.pos - 1
        assert np.array_equal(isa.machine.slices[0].dram[:n], rtl.drams[0][:n])
    assert sorted(eng._decodes) == [1, 2]


@pytest.mark.skipif(not REAL.exists(), reason="models/LFM2.5-230M not downloaded")
def test_lfm2_5_230m_program_fits_board_imem():
    """The hardware loop keeps the longest program (last position of a 4K context) in IMEM."""
    spec = load_spec(REAL)
    img = spec.image(board_config(), 4096)
    assert len(img.compile_step(4095)[0]) <= board_config().IMEM_WORDS // 8


@pytest.mark.skipif(not REAL.exists(), reason="models/LFM2.5-230M not downloaded")
def test_lfm2_5_230m_resident_programs_fit_a_4k_context():
    """The resident decode programs of a 4K context (16 attention buckets, 4-bit layers on the
    board's configuration) fit IMEM and the registers: 6 arguments from R15 down, address
    registers from R1 up."""
    from opentpu import isa as I
    from opentpu.llm.qwen3 import device_config
    spec, kw = load_spec(REAL), dict(wformat="fp4", head_format="int8")
    cfg = device_config(spec, 4096, lookup=True, MCOLS=2, PAIR=True, **kw)
    img = spec.image(cfg, 4096, lookup=True, **kw)
    for blocks in range(1, 17):
        progs, ra = img.compile_decode(blocks, max(2, (blocks - 1) * 256))
        assert len(progs[0]) <= cfg.IMEM_WORDS // 8 and len(ra) == 6
        written = {i.rd for i in progs[0] if i.op in (I.LI, I.ADDI)}
        assert written <= set(range(1, 16 - len(ra)))       # the arguments stay


@pytest.mark.skipif(not REAL.exists(), reason="models/LFM2.5-230M not downloaded")
def test_lfm2_5_230m_greedy_matches_hf():
    tok = transformers.AutoTokenizer.from_pretrained(REAL)
    ids = _chat_ids(tok, "What is the capital of France? Answer in one sentence.")
    hf = transformers.AutoModelForCausalLM.from_pretrained(REAL, dtype=torch.float32).eval()
    with torch.no_grad():
        want = hf.generate(torch.tensor([ids]), max_new_tokens=8, do_sample=False,
                           repetition_penalty=1.0)[0, len(ids):]    # the config's is 1.05
    eng = Engine(load_spec(REAL), load_weights(REAL), cap=256)
    got = eng.generate(ids, max_new=8)
    assert got == want.tolist()[:len(got)] and len(got) >= 7
    assert tok.decode(got).startswith("The capital of France is Paris.")


@pytest.mark.skipif(not REAL.exists(), reason="models/LFM2.5-230M not downloaded")
def test_lfm2_5_230m_fp4_int8_head_greedy():
    """LFM2.5-230M with 4-bit layer weights and an int8 LM head on the board's MCOLS=2 with
    column reuse (every layer MM a full-rate PAIR MM), on the ISA simulator: the greedy answer
    is still right, and every generated token is the argmax of the float64 emulation of the
    same weights."""
    from opentpu.llm.qwen3 import device_config
    tok = transformers.AutoTokenizer.from_pretrained(REAL)
    ids = _chat_ids(tok, "What is the capital of France? Answer in one sentence.")
    spec, W = load_spec(REAL), load_weights(REAL)
    cfg = device_config(spec, 256, wformat="fp4", head_format="int8", MCOLS=2, PAIR=True)
    eng = Engine(spec, W, cap=256, cfg=cfg, wformat="fp4", head_format="int8")
    got = eng.generate(ids, max_new=8)
    assert "Paris" in tok.decode(got)
    emu = emulated_logits(spec, W, ids + got[:-1], wformat="fp4", head_format="int8")
    assert emu[len(ids) - 1:].argmax(-1).tolist() == got


@pytest.mark.skipif(not REAL.exists(), reason="models/LFM2.5-230M not downloaded")
@pytest.mark.parametrize("wformat,head_format,pair,resident", [
    ("int8", None, False, False), ("fp4", "int8", True, False), ("fp4", "int8", True, True)])
def test_lfm2_5_230m_token_on_rtl_is_bit_exact(have_verilator, wformat, head_format, pair,
                                               resident):
    """Feed part of a prompt on the ISA simulator, then run the next token on the Verilator RTL
    and on the ISA simulator from the same DRAM state: weights, KV cache, conv state and
    logits must agree bit for bit (int8; 4-bit layers with an int8 LM head on the board's
    MCOLS=2 with column reuse; that with the resident decode program, whose logits equal the
    per-position program's)."""
    from opentpu.llm.qwen3 import device_config
    from opentpu.llm.rtl_backend import RtlBackend
    spec = load_spec(REAL)
    lk = {"lookup": True} if resident else {}
    cfg = device_config(spec, 256, wformat=wformat, head_format=head_format, MCOLS=2,
                        PAIR=True, **lk) if pair else None
    W = load_weights(REAL)
    eng = Engine(spec, W, cap=256, cfg=cfg, wformat=wformat, head_format=head_format,
                 resident=resident)
    assert eng.resident == resident
    prompt = [1, 6, 6423, 708, 3493, 856, 779, 5706, 803, 4481]
    for t in prompt[:-1]:
        eng.step(t)
    n = eng.image.nbytes
    rtl = RtlBackend(eng.cfg, [s.dram[:n] for s in eng.backend.machine.slices])
    isa = eng.backend
    want = eng.step(prompt[-1])
    eng.backend, eng.pos = rtl, eng.pos - 1
    got = eng.step(prompt[-1])
    assert np.array_equal(want.view(np.uint32), got.view(np.uint32))
    for s in range(eng.cfg.S):
        assert np.array_equal(isa.machine.slices[s].dram[:n], rtl.drams[s][:n])
    if resident:                            # the per-position programs' logits
        ref = Engine(spec, W, cap=256, cfg=eng.cfg, wformat=wformat, head_format=head_format)
        for t in prompt[:-1]:
            ref.step(t)
        assert np.array_equal(ref.step(prompt[-1]).view(np.uint32), want.view(np.uint32))


def test_tiny_vt_tiles_bit_exact(tiny, monkeypatch):
    """A 512-token cache holds V^T in tiles of 256 tokens (compiler.KVDesc): a prefill chunk
    across the tile edge, then decode past it, give the same logits bit for bit as the plain
    [d, cap] layout."""
    import opentpu.compiler as C
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 264)]

    def run():
        eng = Engine(spec, W, cap=512)
        eng.step(toks[0])
        out = [eng.prefill(toks[1:257], chunk=8)]          # rows 249..256 cross the edge
        out += [eng.step(t) for t in toks[257:]]
        return out
    assert C.vt_tile(512) == 256
    tiled = run()
    monkeypatch.setattr(C, "VT_TILE", 1 << 30)
    plain = run()
    assert all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(tiled, plain))


def test_tiny_image_from_the_quantization_cache(tiny, tmp_path, monkeypatch):
    """Image.build through the 4-bit quantization cache (opentpu/qcache.py) gives the uncached
    image, byte for byte, and a second build quantizes nothing."""
    from opentpu import qcache
    _, W, spec = tiny
    eng = Engine(spec, W, cap=256, wformat="fp4", head_format="int8")
    want = eng.image.build(W)
    monkeypatch.setenv("OTPU_IMAGE_CACHE", str(tmp_path))
    monkeypatch.setattr(qcache, "MIN_ELEMS", 0)
    monkeypatch.setattr(qcache, "FREE_FLOOR", 0)
    monkeypatch.setattr(qcache, "stats", dict.fromkeys(qcache.stats, 0))
    first = eng.image.build(W)
    n = dict(qcache.stats)
    again = eng.image.build(W)
    assert n["miss"] > 0 and n["write"] == n["miss"]
    assert qcache.stats["miss"] == n["miss"] and qcache.stats["hit"] == 2 * n["hit"] + n["miss"]
    for got in (first, again):
        assert all(np.array_equal(a, b) for a, b in zip(got, want))
