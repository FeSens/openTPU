"""Running real LLMs on openTPU: Qwen3 (qwen3.py) and LFM2 (lfm2.py)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# short names for the checkpoints the tools know, downloaded into models/<dir>
MODELS = {"qwen3": "Qwen3-0.6B", "lfm2": "LFM2.5-230M"}


def model_dir(name) -> Path:
    """A short name from MODELS (models/<dir> of the repository) or a checkpoint directory."""
    return ROOT / "models" / MODELS[name] if name in MODELS else Path(name)


def load_spec(path):
    """The Spec of a Hugging Face checkpoint directory, chosen by its config's model_type."""
    t = json.loads((Path(path) / "config.json").read_text()).get("model_type")
    if t == "qwen3":
        from .qwen3 import Spec
    elif t == "lfm2":
        from .lfm2 import Spec
    else:
        raise ValueError(f"{path}: model type {t!r} is not supported (qwen3, lfm2)")
    return Spec.from_hf(path)
