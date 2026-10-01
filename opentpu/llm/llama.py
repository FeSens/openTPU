"""Llama-like decoders on openTPU: SmolLM3 (e.g. SmolLM3-3B) and Phi-3 / Phi-4-mini.

They are Qwen3 (qwen3.py) without the RMSNorm on each q and k head, so they run on Qwen3's
Spec, DRAM image, kernels and Engine; this module only reads their configs (spec_from_hf).
What differs from Qwen3, all in qwen3.py and off for Qwen3:
  qk_norm   False: no q_norm / k_norm
  nope      SmolLM3's layers without RoPE (config no_rope_layers[i] == 0, every 4th layer):
            one hardware loop still runs every layer, each with a rope gate (qwen3._rope_gate)
  rotary    partial RoPE (Phi-3 partial_rotary_factor 0.75: 96 of 128 dimensions; the others
            pass through, as Qwen3.5's)
  rope_div, rope_scale
            LongRoPE (Phi-3 rope_scaling type longrope): the short factors divide the
            frequencies, the attention factor scales cos and sin; the KV capacity stays within
            original_max_position_embeddings (ctx), where Hugging Face uses the short factors
  embed     "int8": the embedding rows are int8 per 128 (their vocabularies, 128K and 200K,
            make an fp32 table 1-2.4 GB): the device gathers them from the tied int8 LM head
            (kernels.gather.gather_row)
Phi-3's fused qkv_proj and gate_up_proj are read as separate q/k/v and gate/up projections
(qwen3.Weights).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import formats as FM
from .qwen3 import Spec

MODEL_TYPES = ("llama", "smollm3", "phi3")


def spec_from_hf(model_dir) -> Spec:
    """The qwen3.Spec of a SmolLM3 / Phi-3 / Llama checkpoint, as Hugging Face's modeling code
    reads its config.json (the RoPE parameters as transformers' standardize_rope_params)."""
    c = json.loads((Path(model_dir) / "config.json").read_text())
    mt = c.get("model_type")
    if mt not in MODEL_TYPES:
        raise ValueError(f"{model_dir}: model type {mt!r} is not a Llama-like one {MODEL_TYPES}")
    if c.get("attention_bias") or c.get("mlp_bias"):
        raise ValueError(f"{model_dir}: attention / MLP biases are not supported")
    H, nq = c["hidden_size"], c["num_attention_heads"]
    d = c.get("head_dim") or H // nq
    L = c["num_hidden_layers"]
    rp = dict(c.get("rope_scaling") or {})
    rp.update(c.get("rope_parameters") or {})
    theta = rp.get("rope_theta", c.get("rope_theta", 10000.0))
    rotary = int(d * rp.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0)))
    kind = rp.get("rope_type", rp.get("type", "default"))
    extra: dict = {}
    if kind == "longrope":
        orig = rp.get("original_max_position_embeddings",
                      c.get("original_max_position_embeddings"))
        factor = rp.get("factor") or c["max_position_embeddings"] / orig
        af = rp.get("attention_factor")
        if af is None:
            af = 1.0 if factor <= 1.0 else math.sqrt(1 + math.log(factor) / math.log(orig))
        extra = dict(rope_div=tuple(float(f) for f in rp["short_factor"]), rope_scale=af,
                     ctx=orig)
        if len(extra["rope_div"]) != rotary // 2:
            raise ValueError(f"{len(extra['rope_div'])} short factors for {rotary // 2} "
                             f"frequencies")
    elif kind != "default":
        raise ValueError(f"{model_dir}: RoPE type {kind!r} is not supported")
    if mt == "smollm3":
        flags = c.get("no_rope_layers")
        if flags is None:                               # SmolLM3Config's default
            n = c.get("no_rope_layer_interval", 4)
            flags = [int((i + 1) % n != 0) for i in range(L)]
        extra["nope"] = tuple(i for i, f in enumerate(flags) if not f)
        if c.get("use_sliding_window"):
            raise ValueError(f"{model_dir}: sliding-window attention is not supported")
    eos = c.get("eos_token_id")
    g = Path(model_dir) / "generation_config.json"
    if g.exists():
        eos = json.loads(g.read_text()).get("eos_token_id", eos)
    return Spec(hidden=H, layers=L, n_q=nq, n_kv=c["num_key_value_heads"], head_dim=d,
                ffn=c["intermediate_size"], vocab=c["vocab_size"], eps=c["rms_norm_eps"],
                theta=theta, tied=c.get("tie_word_embeddings", False),
                bos=c.get("bos_token_id"), eos=tuple(eos) if isinstance(eos, list) else (eos,),
                qk_norm=False, rotary=0 if rotary == d else rotary, embed="int8",
                mix=FM.mix_for(c), **extra)
