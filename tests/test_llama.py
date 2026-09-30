"""The Llama-likes on openTPU (llama.py, on Qwen3's code): SmolLM3 (no q/k norms, layers without
RoPE) and Phi-3 / Phi-4-mini (fused qkv / gate_up projections, partial RoPE, LongRoPE) against
Hugging Face transformers. Tiny random models, saved as checkpoints and read back through
load_spec / load_weights, always run."""
import dataclasses

import numpy as np
import pytest

from opentpu.isasim import board_config
from opentpu.llm import load_spec
from opentpu.llm.qwen3 import Engine, emulated_logits, load_weights, reference_logits

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

NOPE = [1, 1, 1, 0, 1, 1, 1, 0]                 # SmolLM3's no_rope_layers: 0 = no RoPE


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def _smollm3():
    return transformers.SmolLM3ForCausalLM(transformers.SmolLM3Config(
        hidden_size=256, num_hidden_layers=len(NOPE), num_attention_heads=8,
        num_key_value_heads=2, head_dim=128, intermediate_size=512, vocab_size=1000,
        rms_norm_eps=1e-6, rope_parameters={"rope_type": "default", "rope_theta": 5e6},
        no_rope_layers=NOPE, tie_word_embeddings=True, max_position_embeddings=4096,
        bos_token_id=1, eos_token_id=2, pad_token_id=0))


def _phi3():
    """Phi-4-mini's attention in small: query groups of 3, RoPE on 96 of 128 dimensions,
    LongRoPE with short factors that are not all 1 (Phi-4-mini's are) and an attention factor
    of sqrt(1 + ln 4 / ln 4096)."""
    short = [1.0 + 0.05 * i for i in range(48)]
    return transformers.Phi3ForCausalLM(transformers.Phi3Config(
        hidden_size=768, num_hidden_layers=2, num_attention_heads=6, num_key_value_heads=2,
        intermediate_size=512, vocab_size=1000, rms_norm_eps=1e-5, rope_theta=1e4,
        partial_rotary_factor=0.75, max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        rope_scaling={"type": "longrope", "short_factor": short,
                      "long_factor": [4 * f for f in short]},
        tie_word_embeddings=True, bos_token_id=1, eos_token_id=2, pad_token_id=0))


@pytest.fixture(scope="module", params=["smollm3", "phi3"])
def tiny(request, tmp_path_factory):
    torch.manual_seed(0)
    m = {"smollm3": _smollm3, "phi3": _phi3}[request.param]().float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
    path = tmp_path_factory.mktemp(request.param)
    m.save_pretrained(path)                     # the loaders read it back (Phi-3: fused)
    return request.param, m, load_spec(path), load_weights(path)


def test_specs(tiny):
    name, _, spec, W = tiny
    assert not spec.qk_norm
    if name == "smollm3":
        assert spec.nope == (3, 7) and spec.rope_dim == 128 and not spec.rope_div
    else:
        assert spec.nope == () and spec.rope_dim == 96 and spec.ctx == 4096
        assert spec.rope_div[1] == 1.05 and abs(spec.rope_scale - (1 + 1 / 6) ** 0.5) < 1e-12
        # the fused projections, split by rows
        qkv = W["model.layers.0.self_attn.qkv_proj.weight"]
        assert np.array_equal(W["model.layers.0.self_attn.k_proj.weight"], qkv[768:1024])
        gu = W["model.layers.1.mlp.gate_up_proj.weight"]
        assert np.array_equal(W["model.layers.1.mlp.up_proj.weight"], gu[512:])


@pytest.mark.parametrize("config", ["design", "board"])
def test_tiny_matches_hf(tiny, config):
    _, m, spec, W = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 140)]  # > one KV block
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(reference_logits(spec, W, toks) - hf).max() < 1e-4
    cfg = board_config(DRAM_BYTES=1 << 26) if config == "board" else None
    eng = Engine(spec, W, cap=256, cfg=cfg)
    dev = np.array([eng.step(t) for t in toks])
    assert _cos(dev, hf).min() > 0.998
    emu = emulated_logits(spec, W, toks[:12])
    assert _cos(dev[:12], emu).min() > 0.9995


def test_tiny_fp4_follows_emulation(tiny):
    _, _, spec, W = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 12)]
    eng = Engine(spec, W, cap=256, wformat="fp4", head_format="int8")
    dev = np.array([eng.step(t) for t in toks])
    emu = emulated_logits(spec, W, toks, wformat="fp4", head_format="int8")
    assert _cos(dev, emu).min() > 0.9995


def test_tiny_resident_decode_is_bit_exact(tiny):
    """Resident decode (run-time position and token) gives the per-position programs' logits
    bit for bit, across the bucket boundary 256, after a chunked prefill; one hardware loop
    runs every layer, with and without RoPE."""
    _, _, spec, W = tiny
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 262)]
    cfg = board_config(DRAM_BYTES=1 << 26)
    a = Engine(spec, W, cap=512, cfg=cfg, resident=True)
    b = Engine(spec, W, cap=512, cfg=cfg)
    assert a.resident
    for t in toks[:3]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32))
    assert np.array_equal(a.prefill(toks[3:252]), b.prefill(toks[3:252]))
    for t in toks[252:]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32))
    assert sorted(a._decodes) == [1, 2]


@pytest.mark.parametrize("first,chunk", [(0, 5), (3, 8)])
def test_tiny_chunked_prefill_is_bit_exact(tiny, first, chunk):
    _, _, spec, W = tiny
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 21)]
    a, b = Engine(spec, W, cap=128), Engine(spec, W, cap=128)
    for t in toks[:first]:
        a.step(t)
        b.step(t)
    la = a.prefill(toks[first:], chunk=chunk)
    lb = None
    for t in toks[first:]:
        lb = b.step(t)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    lo, hi = a.image.layer0, a.image.nbytes          # the layer blocks: weights and KV cache
    ma, mb = a.backend.machine.slices, b.backend.machine.slices
    assert all(np.array_equal(x.dram[lo:hi], y.dram[lo:hi]) for x, y in zip(ma, mb))


def test_nope_is_one_loop_body(tiny):
    """The rope gate keeps one layer body for every layer: the program does not grow with
    the number of layers without RoPE."""
    name, _, spec, _ = tiny
    if name != "smollm3":
        pytest.skip("no layers without RoPE")
    cfg = board_config(DRAM_BYTES=1 << 26)
    n = [len(dataclasses.replace(spec, nope=nope).image(cfg, 256).compile_step(9)[0])
         for nope in ((3,), (1, 3, 5, 7))]
    assert n[0] == n[1]
