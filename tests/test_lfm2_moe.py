"""LFM2-MoE on openTPU with its experts streamed into DRAM slots (docs/offload.md, path (a)):
the card routes and computes every expert, the host's server moves the missing ones. A tiny
random model against Hugging Face transformers and the quantized emulation; a small cache
gives the same logits bit for bit as one holding every expert."""
import numpy as np
import pytest

from opentpu.llm.lfm2 import Spec, emulated_logits, reference_logits
from opentpu.llm.qwen3 import Engine, device_config

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

KINDS = ("conv", "attn", "conv", "attn", "conv")
E, K, F = 8, 2, 256


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def _tiny():
    torch.manual_seed(0)
    hc = transformers.Lfm2MoeConfig(
        hidden_size=256, num_hidden_layers=len(KINDS), num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=512, moe_intermediate_size=F, num_experts=E,
        num_experts_per_tok=K, num_dense_layers=1, vocab_size=1000, norm_eps=1e-5,
        layer_types=["full_attention" if k == "attn" else "conv" for k in KINDS],
        conv_L_cache=3, conv_bias=False, tie_word_embeddings=True,
        max_position_embeddings=4096, rope_parameters={"rope_type": "default", "rope_theta": 1e6})
    m = transformers.Lfm2MoeForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif "experts" in n or "gate" in n:
                p.copy_(0.05 * torch.randn_like(p))
        for mod in m.modules():
            if hasattr(mod, "expert_bias"):
                mod.expert_bias.copy_(0.02 * torch.randn_like(mod.expert_bias))
    W = {}
    for k, v in m.state_dict().items():
        v = v.float().numpy()
        if k.endswith("experts.gate_up_proj"):          # the checkpoint's per-expert names
            p = k[:-len("gate_up_proj")]
            for e in range(E):
                W[f"{p}{e}.w1.weight"], W[f"{p}{e}.w3.weight"] = v[e, :F], v[e, F:]
        elif k.endswith("experts.down_proj"):
            p = k[:-len("down_proj")]
            for e in range(E):
                W[f"{p}{e}.w2.weight"] = v[e]
        else:
            W[k] = v
    from opentpu.llm.moe import MoESpec
    spec = Spec(256, KINDS, 4, 2, 64, 512, 1000, moe=MoESpec(E=E, k=K, ffn=F, first=1))
    return m, W, spec


@pytest.fixture(scope="module")
def tiny():
    return _tiny()


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
    """The logits follow HF but for a token whose routing turns on a near-tie (the card's
    quantized router against bf16's); the card's choices in the first MoE layer, whose input
    no routing has touched, are the emulation's wherever its k-th and (k+1)-th scores are apart
    by more than the quantization's reach."""
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
    emulated_logits(spec, W, toks, routes=routes)
    moe_layers = len(KINDS) - 1
    assert eng.server.seq == len(toks) * moe_layers - 1       # the last waits for a next fence
    card = eng.server.history
    checked = 0
    for t, layer, ids, margin in routes:
        if layer == 1 and margin > 1e-3 and t * moe_layers < len(card):
            assert card[t * moe_layers] == ids, (t, card[t * moe_layers], ids, margin)
            checked += 1
    assert checked >= len(toks) - 2
    assert eng.server.misses == 0


@pytest.mark.parametrize("wformat", ["int8", "fp4"])
def test_small_cache_is_bit_exact(tiny, wformat):
    """k slots per layer: nearly every request misses and streams; the logits do not move."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 10)]
    full = _engine(spec, W, wformat=wformat)
    ref = np.array([full.step(t) for t in toks])
    small = _engine(spec, W, experts=K, wformat=wformat)
    got = np.array([small.step(t) for t in toks])
    assert small.server.misses > len(toks)
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))


def test_pool_file_is_the_same_pool(tiny, tmp_path):
    """The pool packed once into a file (the page cache or the SSD tier) serves the same bytes;
    a second engine reuses the file."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 6)]
    ref = np.array([_engine(spec, W, experts=K).step(t) for t in [toks[0]]])
    f = tmp_path / "pool.bin"
    for _ in range(2):
        eng = Engine(spec, W, cap=256, cfg=device_config(spec, 256, S=1, experts=K),
                     experts=K, pool_file=f)
        got = np.array([eng.step(t) for t in [toks[0]]])
        assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))
    assert f.stat().st_size == (len(KINDS) - 1) * E * eng.image.offload.slot_bytes


@pytest.mark.parametrize("embed", ["fp32", "gather"])
def test_the_card_generates_with_streamed_experts(tiny, embed):
    """The decode loop on the card (autodecode's generate program, resident decode) with k
    slots per layer: the experts stream between the tokens it picks, and it gives the host's
    resident loop token for token (gather: the embedding row gathered from the tied head)."""
    _, W, spec = tiny
    cfg = device_config(spec, 512, rows=1, lookup="gather" if embed == "gather" else True,
                        S=1, experts=K)
    a, b = (Engine(spec, W, cap=512, cfg=cfg, rows=1, resident=True, experts=K, embed=embed)
            for _ in range(2))
    assert a.can_generate
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 20)]
    t0 = int(np.argmax(a.prefill(toks)))
    assert int(np.argmax(b.prefill(toks))) == t0
    ref, t = [], t0
    for _ in range(12):
        t = int(np.argmax(b.step(t)))
        ref.append(t)
    misses = a.server.misses
    assert a.generate_card(t0, 12, stop_ids=[]) == ref
    assert a.server.misses > misses
