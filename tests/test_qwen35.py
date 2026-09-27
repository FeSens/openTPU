"""Qwen3.5 on openTPU: the hybrid decoder (Gated DeltaNet with its fp32 state streamed through
TMEM, gated attention with 256-wide heads and partial RoPE) against Hugging Face transformers.
A tiny random model always runs; the real Qwen3.5-0.8B runs when its checkpoint is in
models/Qwen3.5-0.8B."""
from pathlib import Path

import numpy as np
import pytest

from opentpu.isasim import board_config
from opentpu.llm import load_spec
from opentpu.llm.lfm2 import plan
from opentpu.llm.qwen3 import Engine, load_weights
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


@pytest.fixture(scope="module")
def tiny():
    """8 DeltaNet heads (two pairs per slice at S=2, four at S=1: the head loop runs), and a
    query group of 4 heads (split in two on the board's 2-column MXU)."""
    torch.manual_seed(0)
    hc = transformers.Qwen3_5TextConfig(
        hidden_size=256, num_hidden_layers=len(KINDS), num_attention_heads=8,
        num_key_value_heads=2, head_dim=256, intermediate_size=512, vocab_size=1000,
        layer_types=["full_attention" if k == "attn" else "linear_attention" for k in KINDS],
        linear_num_key_heads=8, linear_num_value_heads=8, linear_key_head_dim=128,
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
    return m, W, Spec(256, KINDS, 8, 2, 256, 64, 8, 128, 128, 512, 1000)


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


def test_one_sequence_only(tiny):
    _, W, spec = tiny
    with pytest.raises(ValueError, match="one sequence"):
        Engine(spec, W, cap=128, batch=2)


def _layers_dram(eng):
    """The layer blocks (weights, KV cache, conv ring, DeltaNet state): all but the I/O area."""
    img = eng.image
    return [s.dram[img.layer0:img.nbytes] for s in eng.backend.machine.slices]


@pytest.mark.parametrize("config,first,chunk", [("design", 0, 5), ("design", 2, 8),
                                                ("board", 1, 4)])
def test_tiny_chunked_prefill_is_bit_exact(tiny, config, first, chunk):
    """Prefill in chunks (the DeltaNet recurrence row after row on a state loaded once per
    chunk, the convolution over the chunk and the ring from positions 0, 1 or 2 on, gated
    row attention; board: query groups split over the 2-column MXU) gives the same logits,
    KV cache, conv ring and DeltaNet state as token-by-token decode."""
    _, W, spec = tiny
    cfg = board_config(DRAM_BYTES=1 << 24) if config == "board" else None
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 19)]
    ref = Engine(spec, W, cap=256, cfg=cfg)
    want = [ref.step(t) for t in toks]
    eng = Engine(spec, W, cap=256, cfg=cfg)
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
def test_tiny_dstep_is_bit_exact(tiny, config):
    """DSTEP (the DMA streams each DeltaNet head's state through its datapath) gives the same
    logits, DeltaNet state, conv ring and KV cache as the VPU's RDOT / OUTER passes, word for
    word, from position 0 on; chunked prefill (the VPU path) continues from its state."""
    import dataclasses
    from opentpu.isasim import design_config
    _, W, spec = tiny
    base = board_config(DRAM_BYTES=1 << 24) if config == "board" else design_config()
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 9)]
    ref = Engine(spec, W, cap=256, cfg=base)
    eng = Engine(spec, W, cap=256, cfg=dataclasses.replace(base, DSTEP=True))
    assert any(i.op == 0x12 for i in eng.image.compile_step(1)[0])
    for t in toks[:6]:
        assert np.array_equal(ref.step(t).view(np.uint32), eng.step(t).view(np.uint32))
    assert all(np.array_equal(a, b) for a, b in zip(_layers_dram(eng), _layers_dram(ref)))
    a, b = ref.prefill(toks[6:], chunk=3), eng.prefill(toks[6:], chunk=3)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.parametrize("dstep", [False, True])
def test_tiny_qwen35_on_board_model(tiny, have_verilator, dstep):
    """The board model through the host driver, through a full turn of the convolution ring:
    logits bit-identical to the ISA simulator (with and without DSTEP)."""
    from opentpu.host.board import BoardBackend, SimTransport
    _, W, spec = tiny
    cfg = board_config(DRAM_BYTES=1 << 24, DSTEP=dstep)
    isa = Engine(spec, W, cap=256, cfg=cfg)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5)
    brd = Engine(spec, W, cap=256, cfg=cfg,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    for tok in (11, 222, 333, 444, 555):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert brd.stats[-1]["cycles"] > 0


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
