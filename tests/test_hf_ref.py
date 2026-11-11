"""tools/qual/hf_ref.py: Hugging Face's logits with the decoder layers streamed from the
checkpoint (built on the meta device, each layer read before it runs and dropped after) are
from_pretrained's bit for bit, for a tiny Qwen3.5 and a tiny Qwen3.5-MoE, in fp32 and bf16, from
save_pretrained's checkpoint (a MoE's experts one by one) and from a multimodal one's names with
the experts fused (Qwen3.6-35B-A3B's); and for a tiny GPT-OSS whose experts are MXFP4
(GPT-OSS-20B's checkpoint)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("linear", "linear", "attn", "linear")
IDS = [3, 17, 256, 999, 42, 7, 512, 100, 64]


def _hf_ref():
    spec = importlib.util.spec_from_file_location("hf_ref", ROOT / "tools/qual/hf_ref.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _model(moe: bool):
    torch.manual_seed(1)
    kw = dict(hidden_size=256, num_hidden_layers=len(KINDS), num_attention_heads=8,
              num_key_value_heads=2, head_dim=256, vocab_size=1000,
              layer_types=["full_attention" if k == "attn" else "linear_attention" for k in KINDS],
              linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=128,
              linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=False,
              max_position_embeddings=4096, rms_norm_eps=1e-6, initializer_range=0.2,
              rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                               "partial_rotary_factor": 0.25})
    if moe:
        hc = transformers.Qwen3_5MoeTextConfig(num_experts=8, num_experts_per_tok=2,
                                               moe_intermediate_size=128,
                                               shared_expert_intermediate_size=128, **kw)
        m = transformers.Qwen3_5MoeForCausalLM(hc)
    else:
        m = transformers.Qwen3_5ForCausalLM(transformers.Qwen3_5TextConfig(intermediate_size=512,
                                                                           **kw))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_((1.0 if n.endswith("linear_attn.norm.weight") else 0.0)
                        + 0.1 * torch.randn_like(p))
    return m.float().eval()


def _multimodal(src: Path, d: Path) -> Path:
    """src's checkpoint (save_pretrained's: a MoE's experts one by one) as Qwen3.6-35B-A3B's
    stores it: the text model under model.language_model., each layer's experts fused, gate_up_proj
    (E, 2F, H: the gate's rows, then the up projection's) and down_proj (E, H, F)."""
    import re

    from safetensors.torch import load_file, save_file
    W = load_file(str(src / "model.safetensors"))
    out, ex = {}, {}
    for k, v in W.items():
        m = re.fullmatch(r"(.*mlp\.experts)\.(\d+)\.(\w+)_proj\.weight", k)
        if m:
            ex.setdefault(m[1], {}).setdefault(m[3], {})[int(m[2])] = v
        else:
            out[k] = v
    for base, p in ex.items():
        n = len(p["down"])
        out[base + ".gate_up_proj"] = torch.stack([torch.cat([p["gate"][e], p["up"][e]])
                                                   for e in range(n)])
        out[base + ".down_proj"] = torch.stack([p["down"][e] for e in range(n)])
    out = {k.replace("model.", "model.language_model.", 1) if k.startswith("model.") else k:
           v.contiguous() for k, v in out.items()}
    d.mkdir(parents=True)
    (d / "config.json").write_text((src / "config.json").read_text())
    save_file(out, str(d / "model.safetensors"))
    return d


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_streamed_layers_give_from_pretraineds_logits(tmp_path, moe, dtype):
    H = _hf_ref()
    d = tmp_path / "ckpt"
    _model(moe).save_pretrained(d)
    td = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    ref = transformers.AutoModelForCausalLM.from_pretrained(d, dtype=td).eval()
    with torch.no_grad():
        want = ref(input_ids=torch.tensor([IDS]), use_cache=False).logits[0].float().numpy()
    dirs = [d, _multimodal(d, tmp_path / "fused")]
    for c in dirs:
        s = H.Stream(c, dtype)
        got = s.logits(IDS, len(IDS))
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), c.name
        assert all(p.is_meta for p in s.m.model.layers.parameters())   # dropped after the pass
        assert len(s.layer_s) == len(KINDS)
        assert s.ck.bytes == sum(p.numel() * 4 for p in ref.parameters())   # all read once


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_mxfp4_experts_are_dequantized_as_read(tmp_path, dtype):
    """GPT-OSS-20B's checkpoint: the experts MXFP4 (NAME_blocks, NAME_scales) and a
    quantization_config in its config (dropped: the model's own parameters on the meta device).
    The streamed logits are from_pretrained's of the same model with its experts' dequantized
    values stored dense, bit for bit (the values exact in bf16; from_pretrained of the MXFP4
    checkpoint itself without a GPU dequantizes them to bf16 in either dtype)."""
    import json

    from safetensors.torch import save_file
    from test_gptoss import _mxfp4_model, _tiny
    H = _hf_ref()
    m, W, _ = _tiny()
    W = _mxfp4_model(m, W)                      # m: given the MXFP4 values
    m.save_pretrained(tmp_path / "dense")
    d = tmp_path / "mx"
    d.mkdir()
    c = json.loads((tmp_path / "dense" / "config.json").read_text())
    (d / "config.json").write_text(json.dumps({**c, "quantization_config":
                                               {"quant_method": "mxfp4"}}))
    t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in W.items()}
    assert t["model.layers.0.mlp.experts.gate_up_proj_blocks"].dtype == torch.uint8
    save_file(t, str(d / "model.safetensors"))
    td = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    ref = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "dense", dtype=td).eval()
    ids = [int(i) for i in np.random.default_rng(5).integers(0, 1000, 24)]   # past the window
    with torch.no_grad():
        want = ref(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float().numpy()
    s = H.Stream(d, dtype)
    assert not hasattr(s.m.config, "quantization_config")
    got = s.logits(ids, len(ids))
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
    assert all(p.is_meta for p in s.m.model.layers.parameters())
    assert s.ck.bytes == sum(v.numel() * v.element_size() for v in t.values())
