"""Qwen3.5 on openTPU: the hybrid decoder (Gated DeltaNet with its fp32 state streamed through
TMEM, gated attention with 256-wide heads and partial RoPE) against Hugging Face transformers.
A tiny random model always runs; the real Qwen3.5-0.8B runs when its checkpoint is in
models/Qwen3.5-0.8B."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from opentpu.isasim import board_config
from opentpu.llm import load_spec
from opentpu.llm.lfm2 import plan
from opentpu.llm.qwen3 import ATTN_BLOCK, Engine, load_weights
from opentpu.llm.qwen35 import Spec, emulated_logits, reference_logits

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

REAL = Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-0.8B"
KINDS = ("linear", "linear", "attn", "linear", "linear", "attn")   # a (lin, lin, attn) loop


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def _chat_ids(tok, text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                  enable_thinking=False, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def _tiny_model(nk: int, nv: int = 8):
    """A random tiny Qwen3.5 (HF model, weights, Spec) with nv DeltaNet value heads, nk key
    heads."""
    torch.manual_seed(0)
    hc = transformers.Qwen3_5TextConfig(
        hidden_size=256, num_hidden_layers=len(KINDS), num_attention_heads=8,
        num_key_value_heads=2, head_dim=256, intermediate_size=512, vocab_size=1000,
        layer_types=["full_attention" if k == "attn" else "linear_attention" for k in KINDS],
        linear_num_key_heads=nk, linear_num_value_heads=nv, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=True,
        max_position_embeddings=4096, rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25})
    m = transformers.Qwen3_5ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:     # zero-centered norms, but the DeltaNet output norm is plain
                p.copy_((1.0 if n.endswith("linear_attn.norm.weight") else 0.0)
                        + 0.1 * torch.randn_like(p))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    return m, W, Spec(256, KINDS, 8, 2, 256, 64, nv, 128, 128, 512, 1000, lin_kheads=nk)


@pytest.fixture(scope="module", params=[8, 4], ids=["kh8", "kh4"])
def tiny(request):
    """8 DeltaNet heads (two pairs per slice at S=2, four at S=1: the head loop runs), and a
    query group of 4 heads (split in two on the board's 2-column MXU). kh4: 4 key heads for
    the 8 value heads (each key head's q and k serve two, as Qwen3.5-4B's 16 for 32)."""
    return _tiny_model(request.param)


def test_plan_loops_the_repeated_unit():
    assert plan(("linear", "linear", "linear", "attn") * 6) == [
        (0, ("linear", "linear", "linear", "attn"), 6)]
    assert plan(KINDS) == [(0, ("linear", "linear", "attn"), 2)]


@pytest.mark.parametrize("config", ["design", "board"])
def test_tiny_matches_hf(tiny, config):
    """design: 2 slices, 8 MXU columns; board: 1 slice, 2 columns (query groups split)."""
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 48)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(reference_logits(spec, W, toks) - hf).max() < 1e-4
    cfg = board_config(DRAM_BYTES=1 << 24) if config == "board" else None
    eng = Engine(spec, W, cap=256, cfg=cfg)
    dev = np.array([eng.step(t) for t in toks])
    assert _cos(dev, hf).min() > 0.998
    # the device follows the quantized math (it differs in fp32 rounding and in rounding the
    # weights to int8: the emulation divides by the scale, the device multiplies by 127/amax)
    emu = emulated_logits(spec, W, toks[:12])
    assert _cos(dev[:12], emu).min() > 0.999


@pytest.mark.parametrize("config", ["design", "board"])
def test_tiny_fp4_follows_emulation(tiny, config):
    """4-bit (FP4) weights: the device follows the float64 emulation of the same weights."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 12)]
    cfg = board_config(DRAM_BYTES=1 << 24) if config == "board" else None
    eng = Engine(spec, W, cap=256, cfg=cfg, wformat="fp4")
    dev = np.array([eng.step(t) for t in toks])
    assert _cos(dev, emulated_logits(spec, W, toks, wformat="fp4")).min() > 0.999


def test_tiny_reset_clears_state_and_conv_ring(tiny):
    """After reset, position 0 must not read the previous sequence's DeltaNet state or
    convolution rows."""
    _, W, spec = tiny
    eng = Engine(spec, W, cap=128)
    a = [eng.step(t) for t in (5, 6, 7, 8, 9)]
    eng.reset()
    b = [eng.step(t) for t in (5, 6, 7, 8, 9)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_embedding_format_by_table_size(tiny, tmp_path):
    """Spec.from_hf: an fp32 embedding table over EMBED_F32_MAX (2 GiB: Qwen3.5-4B's is 2.4)
    is int8; the 0.8B's (0.9 GiB) and the 2B's (1.9) stay fp32."""
    m = tiny[0]
    for hidden, embed in ((1024, "f32"), (2048, "f32"), (2560, "int8")):
        m.config.to_json_file(tmp_path / "config.json")
        c = json.loads((tmp_path / "config.json").read_text())
        c.update(hidden_size=hidden, vocab_size=248320)
        (tmp_path / "config.json").write_text(json.dumps(c))
        assert Spec.from_hf(tmp_path).embed == embed, hidden


@pytest.mark.parametrize("head", ["int8", "fp4"])
def test_tiny_int8_embedding(tiny, head):
    """Spec.embed "int8": the token's int8 embedding row is gathered on the device from the
    tied int8 LM head (under a 4-bit head, from an int8 table of the image's own), in the
    resident decode, the per-position programs and the prefill runs alike: the same logits
    bit for bit, following the float64 emulation, which quantizes the row too."""
    _, W, spec = tiny
    spec = dataclasses.replace(spec, embed="int8")
    cfg = board_config(DRAM_BYTES=1 << 25, PAIR=True)
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


def test_one_sequence_only(tiny):
    _, W, spec = tiny
    with pytest.raises(ValueError, match="one sequence"):
        Engine(spec, W, cap=128, batch=2)


def _layers_dram(eng):
    """The layer blocks (weights, KV cache, conv ring, DeltaNet state): all but the I/O area."""
    img = eng.image
    return [s.dram[img.layer0:img.nbytes] for s in eng.backend.machine.slices]


@pytest.mark.parametrize("config,first,chunk,dstep", [("design", 0, 5, False),
                                                      ("design", 2, 8, False),
                                                      ("board", 1, 4, False),
                                                      ("design", 0, 5, True),
                                                      ("board", 2, 6, True)])
def test_tiny_chunked_prefill_is_bit_exact(tiny, config, first, chunk, dstep):
    """Prefill in chunks (the DeltaNet recurrence row after row on a state loaded once per
    chunk, the convolution over the chunk and the ring from positions 0, 1 or 2 on, gated
    row attention; board: query groups split over the 2-column MXU) gives the same logits,
    KV cache, conv ring and DeltaNet state as token-by-token decode. dstep: the chunk's rows
    each run a DSTEP (the reference decodes on the VPU path)."""
    import dataclasses
    from opentpu.isasim import design_config
    _, W, spec = tiny
    cfg = board_config(DRAM_BYTES=1 << 24) if config == "board" else None
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 19)]
    ref = Engine(spec, W, cap=256, cfg=cfg)
    want = [ref.step(t) for t in toks]
    ecfg = dataclasses.replace(cfg or design_config(), DSTEP=True) if dstep else cfg
    eng = Engine(spec, W, cap=256, cfg=ecfg)
    for t in toks[:first]:
        eng.step(t)
    got = eng.prefill(toks[first:17], chunk=chunk)
    assert np.array_equal(got.view(np.uint32), want[16].view(np.uint32)) and eng.pos == 17
    assert eng.stats[first]["rows"] == chunk
    ref17 = Engine(spec, W, cap=256, cfg=cfg)
    ref17.prefill(toks[:17], chunk=1)
    assert all(np.array_equal(a, b) for a, b in zip(_layers_dram(eng), _layers_dram(ref17)))
    assert all(np.array_equal(eng.step(t), w) for t, w in zip(toks[17:], want[17:]))


@pytest.mark.parametrize("config", ["design", "board"])
@pytest.mark.parametrize("unit", ["dstep", "stream"])
def test_tiny_dstep_is_bit_exact(tiny, config, unit):
    """DSTEP (the DMA streams each DeltaNet head's state through its datapath) gives the same
    logits, DeltaNet state, conv ring and KV cache as the VPU's RDOT / OUTER passes, word for
    word, from position 0 on; chunked prefill (the VPU path) continues from its state. unit
    "stream": a stream-engine machine without DSTEP runs each step as STREAM (isa.gdn_desc)."""
    import dataclasses
    from opentpu.isasim import design_config
    _, W, spec = tiny
    base = board_config(DRAM_BYTES=1 << 24) if config == "board" else design_config()
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 9)]
    ref = Engine(spec, W, cap=256, cfg=base)
    kw = {"DSTEP": True} if unit == "dstep" else {"STREAM": True}
    eng = Engine(spec, W, cap=256, cfg=dataclasses.replace(base, **kw))
    assert any(i.op == (0x12 if unit == "dstep" else 0x13)
               for i in eng.image.compile_step(1)[0])
    for t in toks[:6]:
        assert np.array_equal(ref.step(t).view(np.uint32), eng.step(t).view(np.uint32))
    assert all(np.array_equal(a, b) for a, b in zip(_layers_dram(eng), _layers_dram(ref)))
    a, b = ref.prefill(toks[6:], chunk=3), eng.prefill(toks[6:], chunk=3)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.parametrize("dstep", [False, True])
def test_tiny_resident_decode_is_bit_exact(tiny, dstep):
    """Resident decode (one program per 256-position attention bucket, the token and position
    as run arguments, the embedding and RoPE rows from the image's tables, the conv ring read
    through its mirrored rows at a run-time offset) gives the per-position programs' logits bit
    for bit: from the ring's first positions (0 .. 2 run per-position programs), after a
    chunked prefill and across the bucket boundary at 256; the layer blocks (KV cache, conv
    ring, DeltaNet state) agree too, but for the rings' scratch rows."""
    _, W, spec = tiny
    cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=dstep, PAIR=True)
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 262)]
    a = Engine(spec, W, cap=512, cfg=cfg, resident=True)
    b = Engine(spec, W, cap=512, cfg=cfg)
    assert a.resident and not b.resident
    p = 0
    for stop, run in ((0, 6), (252, 10)):
        if stop > p:
            assert np.array_equal(a.prefill(toks[p:stop]), b.prefill(toks[p:stop]))
            p = stop
        for t in toks[p:p + run]:
            pa = a.pos
            assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32)), pa
        p += run
    assert sorted(a._decodes) == [1, 2] and not b._decodes
    ia, ib = a.image, b.image
    assert (ia.layer0, ia.LS) == (ib.layer0, ib.LS)
    ma, mb = (e.backend.machine.slices[0].dram[ib.layer0:ib.layer0 + len(spec.kinds) * ib.LS]
              .copy() for e in (a, b))
    TP = 2 * spec.conv_k * ib.C
    for li, k in enumerate(spec.kinds):     # row 0 of each pair's ring is scratch (_ring)
        if k == "linear":
            for q in range(ib.nl // 2):
                o = li * ib.LS + ib.cv_offset(q) + 4 * TP
                ma[o:o + 8 * ib.C] = mb[o:o + 8 * ib.C] = 0
    assert np.array_equal(ma, mb)


@pytest.fixture(scope="module", params=[(16, 8), (18, 9)], ids=["h16", "h18"])
def tiny_pairs(request):
    """More DeltaNet heads on one slice, so the decode's pair loop runs: 16 value heads for 8
    key heads (8 pairs, as Qwen3.5-35B-A3B's 32 for 16), and 18 for 9 (9 pairs, out_proj
    groups of 2 heads)."""
    nv, nk = request.param
    return _tiny_model(nk, nv)[1:]


def _pair_parts(eng, li, q):
    """Pair q's taps and window, and its two states, in layer li (either layout)."""
    im, d = eng.image, eng.backend.machine.slices[0].dram
    o = im.layer0 + li * im.LS
    n = 4 * 2 * eng.spec.lin_dv * eng.spec.lin_dk
    st = o + (im.pair_offset(q) + im.pofs["state"] if im.grouped else
              im.lofs["linear"]["state"] + q * n)
    return d[o + im.cv_offset(q):o + im.cv_offset(q) + 4 * im.CVW].copy(), d[st:st + n].copy()


@pytest.mark.parametrize("dstep", [False, True])
def test_group_major_is_bit_exact(tiny_pairs, dstep):
    """The group-major DeltaNet layout (Spec.pair_loop) gives the array-major one's logits,
    taps, windows and states bit for bit: chunked prefill (a loop over the groups, each over
    its pairs), per-position decode (the pair loop), resident decode (the pair loop at a
    run-time position: a shorter program); with and without DSTEP."""
    W, spec = tiny_pairs
    cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=dstep, PAIR=True)
    a, b = (Engine(dataclasses.replace(spec, pair_loop=g), W, cap=512, cfg=cfg, resident=True)
            for g in (False, True))
    assert not a.image.grouped and b.image.grouped
    ra, rb = (e.image.compile_decode(1, 3)[0][0] for e in (a, b))
    assert len(rb) < len(ra)
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 12)]
    assert np.array_equal(a.prefill(toks[:5], chunk=3).view(np.uint32),
                          b.prefill(toks[:5], chunk=3).view(np.uint32))
    for t in toks[5:]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32)), a.pos
    assert sorted(a._decodes) == sorted(b._decodes) == [1]
    for li, k in enumerate(spec.kinds):
        if k == "linear":
            for q in range(spec.lin_heads // 2):
                assert all(np.array_equal(x, y) for x, y in
                           zip(_pair_parts(a, li, q), _pair_parts(b, li, q))), (li, q)


@pytest.mark.parametrize("dstep,resident", [(False, False), (True, False), (True, True),
                                            ("stream", False)])
def test_tiny_qwen35_on_board_model(tiny, have_verilator, dstep, resident):
    """The board model through the host driver, through a full turn of the convolution window:
    logits bit-identical to the ISA simulator (with and without DSTEP; resident: from position
    3 on the resident decode program, its token and position as run arguments; "stream": each
    DeltaNet head step as a STREAM on the stream engine instead of DSTEP)."""
    from opentpu.host.board import BoardBackend, SimTransport
    _, W, spec = tiny
    cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=dstep is True, STREAM=dstep == "stream")
    isa = Engine(spec, W, cap=256, cfg=cfg)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5)
    brd = Engine(spec, W, cap=256, cfg=cfg, resident=resident,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    assert brd.resident == resident
    for tok in (11, 222, 333, 444, 555, 666):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert brd.stats[-1]["cycles"] > 0
    if resident:
        assert sorted(brd._decodes) == [1] and brd.backend._resident[0] is brd._decodes[1][0]


@pytest.mark.parametrize("dstep", [False, True])
def test_group_major_on_board_model(tiny_pairs, have_verilator, dstep):
    """Group-major DeltaNet blocks on the board model through the host driver, resident from
    position 3 on (the pair loop at a run-time position): logits bit-identical to the ISA
    simulator's array-major per-position programs."""
    from opentpu.host.board import BoardBackend, SimTransport
    W, spec = tiny_pairs
    cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=dstep)
    isa = Engine(dataclasses.replace(spec, pair_loop=False), W, cap=256, cfg=cfg)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5)
    brd = Engine(dataclasses.replace(spec, pair_loop=True), W, cap=256, cfg=cfg, resident=True,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    assert brd.resident and brd.image.grouped and not isa.image.grouped
    for tok in (11, 222, 333, 444, 555, 666):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert sorted(brd._decodes) == [1]


def test_4b_resident_decode_fits_imem():
    """Qwen3.5-4B's dims on one slice (16 pairs of DeltaNet heads: group-major), fp4 with an
    int8 LM head: the resident decode, the pair loop at a run-time position beside the
    attention's run-time KV addresses, compiles and fits IMEM at every bucket of a 4K context
    (the gates in the head groups' blocks: the pair loop takes one address register). A
    layout-only image, no weights."""
    kinds = ("linear", "linear", "linear", "attn") * 8
    spec = Spec(2560, kinds, 16, 4, 256, 64, 32, 128, 128, 9216, 248320, lin_kheads=16)
    cfg = board_config(DRAM_BYTES=1 << 33)      # the fp32 lookup table alone is 2.4 GiB
    img = spec.image(cfg, 4096, wformat="fp4", head_format="int8", lookup=True)
    assert img.grouped
    for b in (1, 2, 4, 8, 16):
        (prog,), _ = img.compile_decode(b, (b - 1) * ATTN_BLOCK + 3)
        assert len(prog) <= cfg.IMEM_WORDS // 8, b


def test_tiny_qwen35_on_a_board_without_dstep(tiny, have_verilator):
    """A bitstream built with DSTEP=0 (the DMA's DSTEP datapath left out) reports CAPS bit6 = 0;
    the host configuration then has no DSTEP and the model runs on VOPs, bit-identical."""
    from opentpu.host.board import Board, BoardBackend, SimTransport, device_config
    _, W, spec = tiny
    tr = SimTransport(ch_bytes=(1 << 24) // 2, stall=20, seed=5, params={"DSTEP": 0})
    info = Board(tr).info()
    assert not info["caps"]["dstep"] and info["caps"]["pair"]
    cfg = dataclasses.replace(device_config(info), DRAM_BYTES=1 << 24)
    assert not cfg.DSTEP
    isa = Engine(spec, W, cap=256, cfg=cfg)
    brd = Engine(spec, W, cap=256, cfg=cfg,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    for tok in (11, 222, 333):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.skipif(not REAL.exists(), reason="models/Qwen3.5-0.8B not downloaded")
def test_qwen35_0_8b_program_fits_board_imem():
    """The hardware loops keep the longest program (last position of a 4K context) in IMEM,
    with the query groups split for 2 MXU columns (the default board) or not (4)."""
    spec = load_spec(REAL)
    for mcols in (2, 4):
        cfg = board_config(MCOLS=mcols)
        assert len(spec.image(cfg, 4096).compile_step(4095)[0]) <= cfg.IMEM_WORDS // 8


@pytest.mark.skipif(not REAL.exists(), reason="models/Qwen3.5-0.8B not downloaded")
def test_qwen35_0_8b_greedy_matches_hf():
    tok = transformers.AutoTokenizer.from_pretrained(REAL)
    ids = _chat_ids(tok, "What is the capital of France? Answer in one sentence.")
    hf = transformers.AutoModelForCausalLM.from_pretrained(REAL, dtype=torch.float32).eval()
    with torch.no_grad():
        want = hf.generate(torch.tensor([ids]), max_new_tokens=8, do_sample=False)[0, len(ids):]
    eng = Engine(load_spec(REAL), load_weights(REAL), cap=256)
    got = eng.generate(ids, max_new=8)
    assert got == want.tolist()[:len(got)] and len(got) >= 7
    assert tok.decode(got).startswith("The capital of France is Paris")


@pytest.mark.skipif(not REAL.exists(), reason="models/Qwen3.5-0.8B not downloaded")
@pytest.mark.parametrize("dstep", [False, True])
def test_qwen35_0_8b_token_on_rtl_is_bit_exact(have_verilator, dstep):
    """Feed part of a prompt on the ISA simulator (board configuration), then run the next
    token on the Verilator RTL and on the ISA simulator from the same DRAM state: weights, KV
    cache, convolution ring, DeltaNet state and logits must agree bit for bit (with and
    without DSTEP)."""
    from opentpu.llm.rtl_backend import RtlBackend
    spec = load_spec(REAL)
    eng = Engine(spec, load_weights(REAL), cap=256,
                 cfg=board_config(DRAM_BYTES=1 << 30, DSTEP=dstep))
    prompt = [760, 6511, 314, 9338, 369]            # "The capital of France is"
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


def test_tiny_prefill_rows_dstep_on_rtl(tiny, have_verilator):
    """A 6-row prefill run with DSTEP (a DSTEP per row and head) on the RTL's board memory
    path: DRAM equals the ISA simulator's, and the logits equal token-by-token decode's."""
    from opentpu import rtlsim
    from opentpu.isasim import Machine
    from opentpu.llm.qwen3 import rope_tables
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 9)]
    want = Engine(spec, W, cap=256, cfg=board_config(DRAM_BYTES=1 << 24)).prefill(toks, chunk=1)
    cfg = board_config(DRAM_BYTES=1 << 24, DSTEP=True)
    eng = Engine(spec, W, cap=256, cfg=cfg)
    eng.prefill(toks[:3])                         # a state, ring and cache to continue from
    rows = [(0, 3 + j) for j in range(6)]
    progs = eng.image.compile_rows(rows, [5])
    assert sum(i.op == 0x12 for i in progs[0]) >= 6
    io = eng.image.io
    cs = [np.stack(a).astype(np.float32) for a in zip(*[rope_tables(spec, p) for _, p in rows])]
    dram = eng.backend.machine.slices[0].dram.copy()
    for k, v in (("x", eng.embed[toks[3:]]), ("cos", cs[0]), ("sin", cs[1])):
        b = np.ascontiguousarray(v, np.float32).view(np.uint8).reshape(-1)
        dram[io[k]:io[k] + b.size] = b
    m = Machine(cfg, progs, [dram.copy()]).run()
    drams, _, _ = rtlsim.run(cfg, progs, [dram.copy()], uarch=rtlsim.BOARD_UARCH, axi=True,
                             boot=True)
    n = eng.image.nbytes
    assert np.array_equal(drams[0][:n], m.slices[0].dram[:n])
    v = spec.vocab
    assert np.array_equal(drams[0][io["logits"] + 4 * 5 * v:io["logits"] + 4 * 6 * v]
                          .view(np.float32), want)
