"""Qwen3.5-MoE on openTPU with its experts streamed into DRAM slots (docs/offload.md, path (a)):
the card routes (softmax of the k largest logits) and computes every expert and the gated
shared expert; the host's server moves the missing experts. A tiny random model against
Hugging Face transformers and the quantized emulation; a small cache gives the same logits
bit for bit as one holding every expert."""
import numpy as np
import pytest

from opentpu.llm.moe import MoESpec
from opentpu.llm.qwen3 import Engine, device_config
from opentpu.llm.qwen35 import Spec, emulated_logits, reference_logits

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

KINDS = ("linear", "linear", "attn", "linear", "linear", "attn")
E, K, F = 8, 2, 256


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def _tiny():
    torch.manual_seed(0)
    hc = transformers.Qwen3_5MoeTextConfig(
        hidden_size=256, num_hidden_layers=len(KINDS), num_attention_heads=8,
        num_key_value_heads=2, head_dim=256, vocab_size=1000,
        layer_types=["full_attention" if k == "attn" else "linear_attention" for k in KINDS],
        linear_num_key_heads=8, linear_num_value_heads=8, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=True,
        max_position_embeddings=4096, rms_norm_eps=1e-6, num_experts=E,
        num_experts_per_tok=K, moe_intermediate_size=F, shared_expert_intermediate_size=F,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25})
    m = transformers.Qwen3_5MoeForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:     # zero-centered norms, but the DeltaNet output norm is plain
                p.copy_((1.0 if n.endswith("linear_attn.norm.weight") else 0.0)
                        + 0.1 * torch.randn_like(p))
            elif "experts" in n or "gate" in n:
                p.copy_(0.05 * torch.randn_like(p))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    spec = Spec(256, KINDS, 8, 2, 256, 64, 8, 128, 128, F, 1000,
                moe=MoESpec(E=E, k=K, ffn=F, rule="softmax", shared=F))
    return m, W, spec


@pytest.fixture(scope="module")
def tiny():
    return _tiny()


def test_from_hf_gives_a_moe_the_int8_embedding(tiny, tmp_path):
    """A Qwen3.5-MoE checkpoint's Spec gathers its embedding rows on the device in int8 at any
    vocabulary size (its DRAM beside the layers is expert slots)."""
    tiny[0].config.save_pretrained(tmp_path)
    s = Spec.from_hf(tmp_path)
    assert s.moe is not None and s.embed == "int8"


def _engine(spec, W, experts=None, **kw):
    cfg = device_config(spec, 256, S=1, experts=experts, **kw)
    return Engine(spec, W, cap=256, cfg=cfg, experts=experts, **kw)


def test_reference_matches_hf(tiny):
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 24)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(reference_logits(spec, W, toks) - hf).max() < 1e-4


def test_device_follows_hf_and_routes_on_the_card(tiny):
    """The logits follow HF and the emulation; the card's choices in the first layer, whose
    input no routing has touched, are the emulation's wherever its k-th and (k+1)-th logits
    are apart by more than the quantization's reach."""
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 12)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    eng = _engine(spec, W)
    eng.server.history = []
    dev = np.array([eng.step(t) for t in toks])
    c = _cos(dev, hf)
    assert np.median(c) > 0.997 and (c > 0.99).mean() >= 0.75
    routes = []
    emu = emulated_logits(spec, W, toks, routes=routes)
    assert np.median(_cos(dev, emu)) > 0.999
    L = len(KINDS)
    assert eng.server.seq == len(toks) * L - 1                # the last waits for a next fence
    card = eng.server.history
    checked = 0
    for t, layer, ids, margin in routes:
        if layer == 0 and margin > 1e-2:
            assert card[t * L] == ids, (t, card[t * L], ids, margin)
            checked += 1
    assert checked >= len(toks) - 3
    assert eng.server.misses == 0


@pytest.mark.parametrize("wformat", ["int8", "fp4"])
def test_small_cache_is_bit_exact(tiny, wformat):
    """k slots per layer: nearly every request misses and streams; the logits do not move."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 8)]
    full = _engine(spec, W, wformat=wformat)
    ref = np.array([full.step(t) for t in toks])
    small = _engine(spec, W, experts=K, wformat=wformat)
    got = np.array([small.step(t) for t in toks])
    assert small.server.misses > len(toks)
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))


def test_the_card_generates_with_streamed_experts(tiny):
    """The decode loop on the card (resident decode) with k slots per layer gives the host's
    resident loop token for token while the experts stream."""
    _, W, spec = tiny
    cfg = device_config(spec, 512, rows=1, lookup=True, S=1, experts=K)
    a, b = (Engine(spec, W, cap=512, cfg=cfg, rows=1, resident=True, experts=K)
            for _ in range(2))
    assert a.can_generate
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 16)]
    t0 = int(np.argmax(a.prefill(toks)))
    assert int(np.argmax(b.prefill(toks))) == t0
    ref, t = [], t0
    for _ in range(10):
        t = int(np.argmax(b.step(t)))
        ref.append(t)
    misses = a.server.misses
    assert a.generate_card(t0, 10, stop_ids=[]) == ref
    assert a.server.misses > misses
