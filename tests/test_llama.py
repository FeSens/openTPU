"""The Llama-likes on openTPU (llama.py, on Qwen3's code): SmolLM3 (no q/k norms, layers without
RoPE) and Phi-3 / Phi-4-mini (fused qkv / gate_up projections, partial RoPE, LongRoPE) against
Hugging Face transformers. Tiny random models, saved as checkpoints and read back through
load_spec / load_weights, always run."""
import dataclasses

import numpy as np
import pytest

from opentpu import language as ol
from opentpu.isasim import board_config
from opentpu.kernels.gather import gather_row, onehot
from opentpu.llm import load_spec
from opentpu.llm.qwen3 import Engine, emulated_logits, gathered_rows, load_weights, reference_logits
from opentpu.runtime import Input, Output, Weight, launch

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


@ol.jit
def _gather(table, oh, out, row):
    ol.store(out, gather_row(ol.quantize(ol.load(oh)), table, row, "int8").reshape(1, out.shape[1]))


@pytest.mark.parametrize("K", [256, 768])
def test_embedding_rows_are_the_gathers(K):
    """qwen3.gathered_rows (the host's reference of an int8 embedding) is what the device's
    gather (kernels.gather.gather_row) reads from an int8 table, bit for bit (rows with a zero
    block), within an ulp of q * s."""
    rng = np.random.default_rng(K)
    W = rng.normal(0, 0.05, (40, K)).astype(np.float32)
    W[7, :128] = 0
    oh = onehot(128, 4, "int8")
    for r in (0, 7, 39):
        got = launch(_gather, board_config(DRAM_BYTES=1 << 22), table=Weight(W, 0),
                     oh=Input(oh), out=Output((1, K)), row=r).outputs["out"]
        want = gathered_rows(W[r], 128)
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), r
        assert np.abs(got - W[r]).max() <= np.abs(W[r]).reshape(-1, 128).max(1).max() / 254


def test_specs(tiny):
    name, _, spec, W = tiny
    assert not spec.qk_norm and spec.embed == "int8"
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
    runs every layer, with and without RoPE, and the token's int8 embedding row is gathered
    from the tied int8 LM head (the per-position programs gather it too)."""
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


def test_tiny_resident_own_int8_table(tiny):
    """A 4-bit LM head cannot serve the gather: the image holds its own int8 embedding
    table, with the same values."""
    _, _, spec, W = tiny
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 6)]
    cfg = board_config(DRAM_BYTES=1 << 26)
    a = Engine(spec, W, cap=256, cfg=cfg, resident=True, wformat="fp4", head_format="fp4")
    b = Engine(spec, W, cap=256, cfg=cfg, wformat="fp4", head_format="fp4")
    assert a.resident and a.image.lookup["own"]
    for t in toks:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32))


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


def test_tiny_formats_per_kind(tiny, monkeypatch):
    """Weight formats per kind (Spec.formats, opentpu/llm/formats.py) over a 4-bit image: each
    kind's projections in their format, one layer layout; the device follows the emulation of
    the same formats, closer than the all-fp4 emulation; resident decode and a chunked prefill
    give token-by-token decoding's logits bit for bit. A range that splits a kind across the
    layers is refused (one layout)."""
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    _, _, spec, W = tiny
    mix = dataclasses.replace(spec, formats="attn=int8,gateup=fp4,down=int4,head=int8")
    cfg = board_config(DRAM_BYTES=1 << 26)
    a = Engine(mix, W, cap=256, cfg=cfg, wformat="fp4")
    assert a.image.mf == dict(wq="int8", wk="int8", wv="int8", wo="int8", wg="fp4", wu="fp4",
                              wd="int4") and a.image.head_format == "int8"
    assert mix.image(cfg, 256).nbytes < spec.image(cfg, 256).nbytes
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 12)]
    dev = np.array([a.step(t) for t in toks])
    emu = emulated_logits(mix, W, toks, wformat="fp4")
    e4 = emulated_logits(spec, W, toks, wformat="fp4", head_format="int8")
    assert _cos(dev, emu).min() > 0.9995
    assert np.abs(dev - emu).mean() < 0.7 * np.abs(dev - e4).mean()
    r = Engine(mix, W, cap=256, cfg=cfg, wformat="fp4", resident=True)
    b = Engine(mix, W, cap=256, cfg=cfg, wformat="fp4")
    assert r.resident
    for t, want in zip(toks[:4], dev):
        assert np.array_equal(r.step(t).view(np.uint32), want.view(np.uint32))
    assert np.array_equal(b.prefill(toks, chunk=4).view(np.uint32), dev[-1].view(np.uint32))
    with pytest.raises(ValueError, match="weight formats"):
        spec.image(cfg, 256, formats="mlp@0=fp4")


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
