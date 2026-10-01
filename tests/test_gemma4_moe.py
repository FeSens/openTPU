"""Gemma 4 26B-A4B's parts of opentpu/llm/gemma4.py against Hugging Face transformers, on tiny
random models: global layers with their own KV head count and K = V (attention_k_eq_v), no
per-layer inputs (PLE), a dense MLP width that is not a multiple of 2D (the image pads it),
and the MoE block beside the MLP (enable_moe_block)."""
import json

import numpy as np
import pytest

import opentpu.llm.gemma4 as G

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.gemma4")

S_, F_ = "sliding_attention", "full_attention"
KINDS = (S_, S_, F_, S_, S_, F_)


def _model(tmp_path_factory, name, **kw):
    """head_dim 128 (sliding) and 256 (global, 32 rotated pairs), 4 query heads on 2 KV heads
    (sliding) and on 1 (global, K = V), the MLP 320 wide, no PLE; norms and the router's scales
    random, k_norm near the real model's 0.13 (test_gemma4's tiny)."""
    torch.manual_seed(0)
    hc = transformers.Gemma4TextConfig(
        hidden_size=256, num_hidden_layers=len(KINDS), num_attention_heads=4,
        num_key_value_heads=2, num_global_key_value_heads=1, attention_k_eq_v=True,
        head_dim=128, global_head_dim=256, intermediate_size=320, vocab_size=1000,
        hidden_size_per_layer_input=0, layer_types=list(KINDS), num_kv_shared_layers=0,
        sliding_window=512, final_logit_softcapping=30.0, max_position_embeddings=4096, **kw)
    hc._attn_implementation = "eager"
    m = transformers.models.gemma4.modeling_gemma4.Gemma4ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n or n.endswith("router.scale"):
                k = "k_norm" in n
                p.copy_((0.13 if k else 1.0) + (0.01 if k else 0.1) * torch.randn_like(p))
            elif n.endswith("per_expert_scale"):
                p.copy_(0.5 + torch.rand_like(p))
            elif "experts" in n or "router" in n:
                p.copy_(0.05 * torch.randn_like(p))
        for n, b in m.named_buffers():
            if n.endswith("layer_scalar"):
                b.copy_(0.5 + torch.rand_like(b))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    d = tmp_path_factory.mktemp(name)
    (d / "config.json").write_text(json.dumps(hc.to_dict()))
    return m, W, G.Spec.from_hf(d)


@pytest.fixture(scope="module")
def kv(tmp_path_factory):
    """No MoE: the attention, PLE and MLP-width parts alone."""
    return _model(tmp_path_factory, "kv")


@pytest.fixture(scope="module")
def moe(tmp_path_factory):
    """8 experts 96 wide, top 2, beside the 320-wide MLP."""
    return _model(tmp_path_factory, "moe", enable_moe_block=True, num_experts=8,
                  top_k_experts=2, moe_intermediate_size=96)


def test_spec(kv, moe):
    _, _, s = kv
    assert (s.n_kv, s.n_kv_global, s.k_eq_v, s.ple_dim, s.experts) == (2, 1, True, 0, 0)
    assert [s.kvh(i) for i in range(s.layers)] == [2, 2, 1, 2, 2, 1]
    assert [s.kv_same(i) for i in range(s.layers)] == [False, False, True] * 2
    assert s.kv_src == tuple(range(6)) and s.ffn == (320,) * 6
    _, _, s = moe
    assert (s.experts, s.top_k, s.expert_ffn) == (8, 2, 96)


@pytest.mark.parametrize("name", ["kv", "moe"])
def test_reference_matches_hf(name, request):
    m, W, spec = request.getfixturevalue(name)
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 32)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(G.softcap(spec, G.reference_logits(spec, W, toks)) - hf).max() < 1e-4


def _cfg():
    from opentpu.isasim import board_config
    return board_config(DRAM_BYTES=1 << 26)


def _engine(kv, **kw):
    from opentpu.llm.qwen3 import Engine
    _, W, spec = kv
    return Engine(spec, W, cap=1024, cfg=_cfg(), **kw)


def _cos(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


@pytest.mark.parametrize("wf", ["int8", "fp4"])
def test_image_pads_the_mlp(kv, wf):
    """The 320-wide MLP in blocks of 384 (int8 down) or 512 (4-bit) rows / columns, the extra
    gate / up rows zero; no PLE areas, no V projection in the global (K = V) layers, 2 KV heads
    in the sliding layers' caches and 1 in the global layers'."""
    _, W, spec = kv
    e = _engine(kv, wformat=wf)
    img, F = e.image, {"int8": 384, "fp4": 512}[wf]
    assert img.wproj is None and img.ple_rec == 0 and not img.ple_host
    for i in range(spec.layers):
        o = img.bofs[img._key(i)]
        assert o["mats"]["wg"][0] == F and len(o["wd"]) * img.dchunk[320, wf] == F
        assert ("wv" in o["mats"]) == (spec.kinds[i] == G.SLIDE) and len(o["kv"]) == spec.kvh(i)
        assert "wpg" not in o["mats"] and "g_ple" not in o
    o, base = img.bofs[img._key(0)], img._off(0).static()
    rb = {"int8": 256, "fp4": 128}[wf]
    dram = e.backend.machine.slices[0].dram
    assert not dram[base + o["wg"][0] + 320 * rb:base + o["wg"][0] + F * rb].any()


@pytest.mark.parametrize("wf", ["int8", "fp4"])
def test_device_matches_hf_and_emulation(kv, wf):
    """The device against the emulation (its quantization points) in both formats, and int8
    against HF (fp4 on random weights is far from it: cosine ~0.7)."""
    m, W, spec = kv
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 24)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    e = _engine(kv, wformat=wf)
    dev = np.array([G.softcap(spec, e.step(t)) for t in toks])
    emu = G.softcap(spec, G.emulated_logits(spec, W, toks[:12], wformat=wf))
    assert _cos(dev[:12], emu).min() > 0.99
    if wf == "int8":
        assert _cos(dev, hf).min() > 0.985


@pytest.mark.parametrize("wf", ["int8", "fp4"])
def test_resident_is_bit_exact(kv, wf):
    """Resident decode (embedding gathered on the device, run-time masks) against the
    per-position programs, across the first bucket boundary."""
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 300)]
    a, b = _engine(kv, wformat=wf, resident=True), _engine(kv, wformat=wf)
    assert a.resident and not b.resident
    assert np.array_equal(a.prefill(toks[:250]), b.prefill(toks[:250]))
    for t in toks[250:]:
        assert np.array_equal(a.step(t), b.step(t))


def test_chunked_prefill_is_bit_exact(kv):
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 11)]
    a, b = _engine(kv), _engine(kv)
    assert np.array_equal(a.prefill(toks, chunk=4), b.prefill(toks, chunk=1))
    img = a.image
    assert np.array_equal(a.backend.machine.slices[0].dram[img.layer0:img.head[0]],
                          b.backend.machine.slices[0].dram[img.layer0:img.head[0]])


def test_moe_image_is_not_yet_on_the_device(moe):
    with pytest.raises(NotImplementedError):
        _engine(moe)
