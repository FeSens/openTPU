"""Gemma 4 on openTPU: sliding and global attention with a KV ring and shared KV layers, GeGLU,
per-layer embeddings gathered on the device, against Hugging Face transformers. A tiny random
model always runs; the real model's programs are compiled when its config is in
models/gemma-4-E2B."""
import json
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


@pytest.mark.parametrize("ple", ["int8", "fp4"])
def test_records_roundtrip(ple):
    """pack_records / dequant_records: the device's gather values of a packed table."""
    rows = np.random.default_rng(1).standard_normal((5, 896)).astype(np.float32)
    S = GA.record_blocks(7, ple)
    assert S % 2 == 1 or S % 4 == 2
    back = GA.dequant_records(GA.pack_records(rows, ple, 128, S), ple, 128, S)[:, :896]
    assert np.abs(back - rows).max() / np.abs(rows).max() < (0.005 if ple == "int8" else 0.15)


@pytest.mark.parametrize("wf,ple", [("int8", "int8"), ("fp4", "fp4")])
def test_resident_gathers_are_bit_exact(tiny, wf, ple, monkeypatch):
    """Resident decode gathers the embedding and PLE rows on the device and computes its
    attention masks at the run-time position: the logits equal the per-position programs'
    (host-written rows), bit for bit, across the first bucket boundary."""
    _, W, spec = tiny
    monkeypatch.setenv("OTPU_PLE_FORMAT", ple)
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


@pytest.mark.slow
@pytest.mark.parametrize("resident", [False, True], ids=["per-position", "resident"])
def test_token_on_board_rtl_is_bit_exact(have_verilator, tiny, resident):
    """A token at position 600 (the window's first block masked at its start) on the Verilator
    RTL through the board's memory path (AXI adapter, boot loader) equals the ISA simulator bit
    for bit, logits and the whole DRAM image (caches); resident, with the device's gathers and
    run-time masks."""
    from opentpu import rtlsim
    from opentpu.llm.rtl_backend import RtlBackend
    _, W, spec = tiny
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
    for blocks in (1, 3, 16):
        progs, ra = img.compile_decode(blocks, (blocks - 1) * 256)
        assert 8 * len(progs[0]) <= board_config().IMEM_WORDS
        assert len(ra) == 6
    with pytest.raises(CompileError, match="TMEM"):
        img.compile_rows([(0, p) for p in range(8)], [7])
