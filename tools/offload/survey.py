"""Survey of MoE (and too-big dense) checkpoints for expert offloading (docs/offload.md).

    python3 tools/offload/survey.py [--json out.json] [repo ...]

Reads each checkpoint's safetensors headers over HTTP (two range requests per shard, no weights
downloaded) and the config, sorts every text-decoder tensor into experts, shared experts,
dense (attention, norms, routers, dense MLPs), embedding, LM head and per-layer embeddings (PLE),
and prints per model:

- the expert shape: routed experts per MoE layer, top-k, shared experts, one expert's parameters
  and its bytes in our 4-bit format (fp4 with two-level scales, 4.25 bits per weight);
- total and active parameters (active: dense + shared + top-k experts per MoE layer + LM head);
- the bytes a decode token streams on the card: dense + shared + top-k experts in fp4, the LM
  head in int8 (8.25 bits), the embedding row not counted (one row is looked up);
- the footprint in fp4 / int8 head: the part that must stay on the card (dense, shared, head,
  embedding table in int8 when untied) and the expert pool (what an expert cache holds).

Vision, audio and multi-token-prediction weights are left out: the card decodes text.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

HF = "https://huggingface.co"
FP4_BITS = 4.25          # fp4 elements + one 32-bit two-level scale word per 128 (docs/quant.md)
INT8_BITS = 8.25         # int8 + an fp32 scale per 128

MODELS = [
    "google/gemma-4-26B-A4B",
    "google/gemma-4-E4B",
    "LiquidAI/LFM2-8B-A1B",
    "LiquidAI/LFM2.5-8B-A1B",
    "LiquidAI/LFM2-24B-A2B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen/Qwen3.5-122B-A10B",
    "allenai/OLMoE-1B-7B-0125",
    "ibm-granite/granite-3.1-3b-a800m-instruct",
    "ibm-granite/granite-4.0-h-tiny",
    "ibm-granite/granite-4.0-h-small",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-V2-Lite",
    "moonshotai/Moonlight-16B-A3B-Instruct",
    "PowerInfer/SmallThinker-21BA3B-Instruct",
    "PowerInfer/SmallThinker-4BA0.6B-Instruct",
    "microsoft/Phi-mini-MoE-instruct",
    "mistralai/Mixtral-8x7B-v0.1",
]

SKIP = re.compile(r"(^|\.)(visual|vision_tower|vision_model|audio_tower|audio_model|embed_vision|"
                  r"embed_audio|multi_modal_projector|mtp|vision|audio)(\.|_)")
EXPERT = re.compile(r"(\.experts\.|\.experts_|block_sparse_moe\.(input|output)_linear)")
SHARED = re.compile(r"(shared_expert\.|shared_experts\.|shared_mlp\.)")
LAYER = re.compile(r"layers\.(\d+)\.")
EXPERT_ID = re.compile(r"experts\.(\d+)\.")


def _get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def header(repo, fname):
    url = f"{HF}/{repo}/resolve/main/{fname}"
    n = struct.unpack("<Q", _get(url, (0, 7)))[0]
    h = json.loads(_get(url, (8, 8 + n - 1)))
    h.pop("__metadata__", None)
    return h


def text_config(cfg):
    for k in ("text_config", "llm_config", "language_config"):
        if isinstance(cfg.get(k), dict):
            return cfg[k]
    return cfg


def first(c, *keys, default=None):
    for k in keys:
        if c.get(k) is not None:
            return c[k]
    return default


def numel(t):
    n = 1
    for d in t["shape"]:
        n *= d
    return n


def survey(repo):
    info = json.loads(_get(f"{HF}/api/models/{repo}"))
    files = [s["rfilename"] for s in info["siblings"]
             if s["rfilename"].endswith(".safetensors") and "/" not in s["rfilename"]
             and not s["rfilename"].startswith("consolidated")]
    cfg = json.loads(_get(f"{HF}/{repo}/resolve/main/config.json"))
    tc = text_config(cfg)
    E = first(tc, "num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts",
              "moe_num_primary_experts")
    k = first(tc, "num_experts_per_tok", "top_k_experts", "moe_topk", "num_experts_per_token",
              "moe_top_k", "moe_num_active_primary_experts")
    groups = defaultdict(int)
    with ThreadPoolExecutor(8) as ex:
        headers = list(ex.map(lambda f: header(repo, f), files))
    moe_layers, expert_params = set(), defaultdict(int)   # per (layer): params of all experts
    for h in headers:
        for name, t in h.items():
            if SKIP.search(name):
                continue
            n = numel(t)
            if name.endswith("_blocks"):          # MXFP4 (gpt-oss): two elements per byte
                n *= 2
            elif name.endswith("_scales"):        # its E8M0 scales: part of the 4-bit format
                continue
            m = LAYER.search(name)
            if EXPERT.search(name) and not SHARED.search(name):
                layer = int(m.group(1))
                moe_layers.add(layer)
                expert_params[layer] += n
                groups["experts"] += n
            elif SHARED.search(name) and "shared_expert_gate" not in name:
                groups["shared"] += n
            elif "per_layer" in name and ("embed" in name):
                groups["ple"] += n               # Gemma per-layer embeddings (a row per token)
            elif "embed_tokens" in name or name.endswith("wte.weight") or "word_embeddings" in name:
                groups["embed"] += n
            elif name.startswith("lm_head") or ".lm_head" in name or name == "output.weight":
                groups["head"] += n
            else:
                groups["dense"] += n
    L = first(tc, "num_hidden_layers", "n_layers")
    tied = groups["head"] == 0                    # no LM head tensor: the embedding is the head
    nm = len(moe_layers)
    per_expert = (groups["experts"] / (nm * E)) if nm and E else 0
    head = groups["head"] or groups["embed"]
    active_exp = (k or 0) * per_expert * nm
    total = groups["dense"] + groups["shared"] + groups["experts"] + groups["embed"] \
        + groups["head"] + groups["ple"]
    active = groups["dense"] + groups["shared"] + active_exp + head
    b4 = FP4_BITS / 8
    b8 = INT8_BITS / 8
    return {
        "repo": repo, "arch": first(tc, "model_type", default=cfg.get("model_type")),
        "layers": L, "moe_layers": nm, "experts": E, "top_k": k,
        "shared": groups["shared"], "hidden": first(tc, "hidden_size", "d_model"),
        "expert_ffn": first(tc, "moe_intermediate_size", "expert_intermediate_size",
                            "intermediate_size"),
        "vocab": first(tc, "vocab_size"), "tied": tied,
        "per_expert": per_expert, "experts_total": groups["experts"],
        "dense": groups["dense"], "embed": groups["embed"], "head": groups["head"],
        "ple": groups["ple"], "total": total, "active": active,
        # bytes: fp4 blocks, int8 head; the embedding table in int8 on the card when untied
        "expert_bytes": per_expert * b4,
        "tok_expert_bytes": active_exp * b4,
        "tok_bytes": (groups["dense"] + groups["shared"] + active_exp) * b4 + head * b8,
        "resident_bytes": (groups["dense"] + groups["shared"]) * b4 + head * b8
        + (0 if tied else groups["embed"] * b8),
        "pool_bytes": groups["experts"] * b4,
        "ple_bytes": groups["ple"] * b4,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repos", nargs="*", default=MODELS)
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = []
    for r in a.repos:
        try:
            rows.append(survey(r))
        except Exception as e:  # noqa: BLE001 - a missing / gated repo is reported, not fatal
            print(f"{r}: {type(e).__name__}: {e}", file=sys.stderr)
    G = 1e9
    print(f"{'model':42} {'L':>3} {'moe':>3} {'E':>4} {'k':>2} {'exp MB':>6} {'total B':>7} "
          f"{'active B':>8} {'tok MB':>7} {'exp/tok MB':>10} {'resident GB':>11} {'pool GB':>7}")
    for d in rows:
        print(f"{d['repo']:42} {d['layers']:>3} {d['moe_layers']:>3} {d['experts'] or 0:>4} "
              f"{d['top_k'] or 0:>2} {d['expert_bytes'] / 1e6:6.2f} {d['total'] / G:7.2f} "
              f"{d['active'] / G:8.2f} {d['tok_bytes'] / 1e6:7.0f} "
              f"{d['tok_expert_bytes'] / 1e6:10.0f} {d['resident_bytes'] / G:11.2f} "
              f"{d['pool_bytes'] / G:7.2f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
