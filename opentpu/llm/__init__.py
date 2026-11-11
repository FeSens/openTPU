"""Running real LLMs on openTPU: Qwen3 (qwen3.py), LFM2 (lfm2.py), Qwen3.5 (qwen35.py), Gemma 4
(gemma4.py), GPT-OSS (gptoss.py) and the Llama-likes SmolLM3 and Phi-3 / Phi-4-mini (llama.py,
on Qwen3's code)."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# short names for the checkpoints the tools know, downloaded into models/<dir>; the last four
# are mixture-of-experts models bigger than the card's DRAM (Qwen3.6-35B-A3B is Qwen3.5-35B-A3B's
# Spec), their experts streamed from a pool file (opentpu/host/streamed.py), as is MiniCPM-V
# 4.7's language model (Qwen3.5-35B-A3B's architecture: docs/minicpmv.md). Gemma 4's are the
# instruct checkpoints (-it): the base ones (gemma-4-E2B, ...) have no chat template and continue
# a chat's text instead of answering
MODELS = {"qwen3": "Qwen3-0.6B", "lfm2": "LFM2.5-230M", "qwen35": "Qwen3.5-0.8B",
          "lfm2-2.6b": "LFM2-2.6B", "smollm3": "SmolLM3-3B", "phi4-mini": "Phi-4-mini-instruct",
          "qwen35-2b": "Qwen3.5-2B", "qwen35-4b": "Qwen3.5-4B", "qwen35-27b": "Qwen3.5-27B",
          "qwen38-27b": "Qwen3.8-27B",
          "gemma4": "gemma-4-E2B-it",
          "gemma4-e4b": "gemma-4-E4B-it", "lfm2-8b": "LFM2.5-8B-A1B",
          "qwen36-35b": "Qwen3.6-35B-A3B",
          "qwen35-35b": "Qwen3.5-35B-A3B", "gemma4-26b": "gemma-4-26B-A4B-it",
          "minicpmv-35b": "MiniCPM-V-4.7-35B-A3B", "gpt-oss-20b": "gpt-oss-20b"}


def model_dir(name) -> Path:
    """A short name from MODELS (models/<dir> of the repository) or a checkpoint directory."""
    return ROOT / "models" / MODELS[name] if name in MODELS else Path(name)


def rope_parameters(c: dict, model_dir, types=("default",)) -> dict:
    """A config's RoPE parameters as transformers reads them (rope_scaling, then
    rope_parameters over it). A RoPE type outside `types` is refused: Hugging Face's YaRN,
    linear or dynamic scaling change the frequencies or cos / sin, the outputs with them."""
    rp = dict(c.get("rope_scaling") or {})
    rp.update(c.get("rope_parameters") or {})
    kind = rp.get("rope_type", rp.get("type", "default"))
    if kind not in types:
        raise ValueError(f"{model_dir}: RoPE type {kind!r} is not supported")
    return rp


def yarn(rp: dict, dim: int, theta: float) -> tuple:
    """Hugging Face's YaRN (rope_type "yarn": rope_parameters' factor, original_max_position_
    embeddings, beta_fast / beta_slow, truncate, attention_factor or mscale / mscale_all_dim)
    as qwen3.rope_tables' (rope_div, rope_scale): frequency i divided by 1 / (1 - r_i + r_i /
    factor), r_i the ramp from the correction range's low to high dimension (r 0: the
    frequency as it is, 1: divided by the factor), and both tables times the attention
    factor. In float64 (Hugging Face's frequencies are fp32)."""
    factor = rp["factor"]
    orig = rp["original_max_position_embeddings"]
    att = rp.get("attention_factor")

    def mscale(s, m=1.0):
        return 1.0 if s <= 1 else 0.1 * m * math.log(s) + 1.0
    if att is None:
        m, ma = rp.get("mscale"), rp.get("mscale_all_dim")
        att = mscale(factor, m) / mscale(factor, ma) if m and ma else mscale(factor)

    def corr(rot):
        return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(theta))
    lo, hi = corr(rp.get("beta_fast") or 32), corr(rp.get("beta_slow") or 1)
    if rp.get("truncate", True):
        lo, hi = math.floor(lo), math.ceil(hi)
    lo, hi = max(lo, 0), min(hi, dim - 1)
    if lo == hi:
        hi += 0.001
    r = [min(1.0, max(0.0, (i - lo) / (hi - lo))) for i in range(dim // 2)]
    return tuple(1.0 / (1.0 - x + x / factor) for x in r), float(att)


def load_spec(path):
    """The Spec of a Hugging Face checkpoint directory, chosen by its config's model_type."""
    t = json.loads((Path(path) / "config.json").read_text()).get("model_type")
    if t == "qwen3":
        from .qwen3 import Spec
    elif t in ("lfm2", "lfm2_moe"):
        from .lfm2 import Spec
    elif t in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"):
        from .qwen35 import Spec
    elif t == "minicpmv4_7":            # (qwen35's Spec with the vision side)
        from .minicpmv import Spec
    elif t in ("gemma4", "gemma4_text"):
        from .gemma4 import Spec
    elif t == "gpt_oss":
        from .gptoss import Spec
    else:
        from .llama import MODEL_TYPES, spec_from_hf
        if t in MODEL_TYPES:
            return spec_from_hf(path)
        raise ValueError(f"{path}: model type {t!r} is not supported (qwen3, lfm2, qwen3_5 and "
                         f"their MoE variants, gemma4, minicpmv4_7, gpt_oss, "
                         f"{', '.join(MODEL_TYPES)})")
    return Spec.from_hf(path)
