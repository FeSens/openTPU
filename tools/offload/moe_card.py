"""LFM2.5-8B-A1B end to end on the ISA simulator with the card's 4 GiB of DRAM (docs/offload.md,
phase 2): the card routes and computes every expert, its experts stream into per-layer slots
through the host's server (opentpu.host.offload), and the decode loop runs on the card
(autodecode's generate program). Greedy tokens against Hugging Face's.

    python3 tools/offload/moe_card.py MODEL --hf out.json       # HF's greedy tokens (bf16, CPU)
    python3 tools/offload/moe_card.py MODEL --check out.json    # the simulator's, compared

The HF run and the simulator run are separate processes: each needs most of a 32 GB host.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PROMPT = "What is the capital of France? Answer in one sentence."


def hf_greedy(model: str, n: int, max_memory: str | None) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                  add_generation_prompt=True, tokenize=True)
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    kw = {}
    if max_memory:                              # the rest offloaded to disk (accelerate)
        kw = dict(device_map="auto", max_memory={"cpu": max_memory},
                  offload_folder=str(Path(model) / ".." / ".offload-moe-card"))
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, **kw).eval()
    t = time.time()
    with torch.no_grad():
        out = m.generate(torch.tensor([ids]), max_new_tokens=n, do_sample=False)
    new = [int(x) for x in out[0, len(ids):]]
    return dict(model=model, prompt=PROMPT, ids=ids, tokens=new, text=tok.decode(new),
                seconds=round(time.time() - t, 1))


def card(model: str, ref: dict, n: int, experts: int, cap: int, pool: str | None) -> dict:
    from opentpu.isasim import board_config
    from opentpu.llm.lfm2 import Spec
    from opentpu.llm.qwen3 import Engine, LazyWeights
    spec = Spec.from_hf(model)
    W = LazyWeights(model)
    t = time.time()
    eng = Engine(spec, W, cap=cap, cfg=board_config(), rows=1, wformat="fp4",
                 head_format="int8", resident=True, experts=experts, pool_file=pool)
    load_s = time.time() - t
    srv = eng.server
    ids = ref["ids"]
    t = time.time()
    lg = eng.prefill(ids)
    t0 = int(np.argmax(lg))
    prefill_s = time.time() - t
    miss0 = srv.misses
    t = time.time()
    got = [t0] + eng.generate_card(t0, n - 1, stop_ids=[])
    gen_s = time.time() - t
    L = eng.image.offload
    return dict(tokens=got, match=got == ref["tokens"][:len(got)],
                first_diff=next((i for i, (a, b) in enumerate(zip(got, ref["tokens"])) if a != b),
                                None),
                experts_per_layer=experts, slots=L.layers * experts, pool=L.layers * L.E,
                image_mib=round(eng.image.nbytes / 2**20), slot_mb=round(L.slot_bytes / 1e6, 2),
                requests=srv.seq, hits=srv.hits, misses=srv.misses,
                misses_per_token_decode=round((srv.misses - miss0) / max(1, n - 1), 2),
                load_s=round(load_s), prefill_s=round(prefill_s), generate_s=round(gen_s))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--hf", help="write HF's greedy tokens here")
    ap.add_argument("--check", help="HF's tokens (--hf's output) to compare the card's with")
    ap.add_argument("-n", type=int, default=16, help="tokens to generate")
    ap.add_argument("--experts", type=int, default=20, help="expert slots per MoE layer")
    ap.add_argument("--cap", type=int, default=4096, help="KV capacity")
    ap.add_argument("--pool", help="the expert pool file (packed once, reused)")
    ap.add_argument("--max-memory", help="HF: host RAM for weights, the rest to disk (e.g. 10GiB)")
    ap.add_argument("--out", help="the card's result as JSON")
    a = ap.parse_args()
    if a.hf:
        r = hf_greedy(a.model, a.n, a.max_memory)
        Path(a.hf).write_text(json.dumps(r, indent=1))
        print(json.dumps(r))
        return
    ref = json.loads(Path(a.check).read_text())
    r = card(a.model, ref, a.n, a.experts, a.cap, a.pool)
    print(json.dumps(r))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
