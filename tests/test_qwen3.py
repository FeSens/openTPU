"""Qwen3 on openTPU: the full decoder (layer loop, QK-norm, GQA, RoPE, tied LM head) against
Hugging Face transformers. A tiny random model always runs; the real Qwen3-0.6B runs when its
checkpoint is in models/Qwen3-0.6B."""
from pathlib import Path

import numpy as np
import pytest

from opentpu.llm.qwen3 import Engine, Spec, emulated_logits, load_weights, reference_logits

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

REAL = Path(__file__).resolve().parent.parent / "models" / "Qwen3-0.6B"


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    hc = transformers.Qwen3Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=128, intermediate_size=512,
                                  vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
                                  tie_word_embeddings=True, max_position_embeddings=4096)
    m = transformers.Qwen3ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    return m, W, Spec(256, 2, 4, 2, 128, 512, 1000)


def test_tiny_matches_hf(tiny):
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 140)]  # > one KV block
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(reference_logits(spec, W, toks) - hf).max() < 1e-4
    eng = Engine(spec, W, cap=256)
    dev = np.array([eng.step(t) for t in toks])
    assert _cos(dev, hf).min() > 0.998
    # the device follows the quantized math (it differs only in fp32 rounding)
    emu = emulated_logits(spec, W, toks[:12])
    assert _cos(dev[:12], emu).min() > 0.9995


def test_tiny_reset_reuses_cache(tiny):
    _, W, spec = tiny
    eng = Engine(spec, W, cap=128)
    a = [eng.step(t) for t in (5, 6, 7)]
    eng.reset()
    b = [eng.step(t) for t in (5, 6, 7)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.skipif(not REAL.exists(), reason="models/Qwen3-0.6B not downloaded")
def test_qwen3_0_6b_greedy_matches_hf():
    tok = transformers.AutoTokenizer.from_pretrained(REAL)
    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                  tokenize=True)
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    hf = transformers.AutoModelForCausalLM.from_pretrained(REAL, dtype=torch.float32).eval()
    with torch.no_grad():
        want = hf.generate(torch.tensor([ids]), max_new_tokens=8, do_sample=False)[0, len(ids):]
    eng = Engine(Spec.from_hf(REAL), load_weights(REAL), cap=256)
    got = eng.generate(ids, max_new=8)
    assert got == want.tolist()[:len(got)] and len(got) >= 7
    assert tok.decode(got).startswith("The capital of France is Paris.")


@pytest.mark.skipif(not REAL.exists(), reason="models/Qwen3-0.6B not downloaded")
def test_qwen3_0_6b_token_on_rtl_is_bit_exact():
    """Feed part of a prompt on the ISA simulator, then run the next token on the Verilator RTL
    and on the ISA simulator from the same DRAM state: weights, KV cache and logits must agree
    bit for bit."""
    from opentpu.llm.rtl_backend import RtlBackend
    eng = Engine(Spec.from_hf(REAL), load_weights(REAL), cap=256)
    prompt = [151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30]
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
