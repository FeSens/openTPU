"""LFM2 on openTPU: the hybrid decoder (short-conv and GQA attention layers, 64-wide heads
padded to the MXU depth, conv state ring in DRAM) against Hugging Face transformers. A tiny
random model always runs; the real LFM2.5-230M runs when its checkpoint is in
models/LFM2.5-230M."""
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


def test_tiny_reset_reuses_cache_and_conv_state(tiny):
    """After reset, positions 0 and 1 must not read the previous sequence's conv state."""
    _, W, spec = tiny
    eng = Engine(spec, W, cap=128)
    a = [eng.step(t) for t in (5, 6, 7, 8)]
    eng.reset()
    b = [eng.step(t) for t in (5, 6, 7, 8)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_one_token_per_run_only(tiny):
    _, W, spec = tiny
    with pytest.raises(ValueError, match="one token per device run"):
        Engine(spec, W, cap=128, batch=2)


def test_tiny_lfm2_on_board_model(tiny, have_verilator):
    """The board model through the host driver, through a full turn of the conv state ring:
    logits bit-identical to the ISA simulator."""
    from opentpu.host.board import BoardBackend, SimTransport
    _, W, spec = tiny
    cfg = board_config(DRAM_BYTES=1 << 23)
    isa = Engine(spec, W, cap=256, cfg=cfg)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5)
    brd = Engine(spec, W, cap=256, cfg=cfg,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    for tok in (11, 222, 333, 444):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert brd.stats[-1]["cycles"] > 0


@pytest.mark.skipif(not REAL.exists(), reason="models/LFM2.5-230M not downloaded")
def test_lfm2_5_230m_program_fits_board_imem():
    """The hardware loop keeps the longest program (last position of a 4K context) in IMEM."""
    spec = load_spec(REAL)
    img = spec.image(board_config(), 4096)
    assert len(img.compile_step(4095)[0]) <= board_config().IMEM_WORDS // 8


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
def test_lfm2_5_230m_token_on_rtl_is_bit_exact(have_verilator):
    """Feed part of a prompt on the ISA simulator, then run the next token on the Verilator RTL
    and on the ISA simulator from the same DRAM state: weights, KV cache, conv state and
    logits must agree bit for bit."""
    from opentpu.llm.rtl_backend import RtlBackend
    eng = Engine(load_spec(REAL), load_weights(REAL), cap=256)
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
