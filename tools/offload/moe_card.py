"""A MoE model (LFM2.5-8B-A1B, Qwen3.5-35B-A3B) end to end on the ISA simulator with the card's
4 GiB of DRAM (docs/offload.md, phase 2): the card routes and computes every expert, its
experts stream into per-layer slots through the host's server (opentpu.host.offload), and the
decode loop runs on the card (autodecode's generate program). Greedy tokens against Hugging
Face's.

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


def hf_greedy(model: str, n: int, max_memory: str | None, prompt: str = PROMPT) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, tokenize=True)
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    kw = {}
    if max_memory:                              # the rest offloaded to disk (accelerate)
        kw = dict(device_map="auto", max_memory={"cpu": max_memory},
                  offload_folder=str(Path(model) / ".." / ".offload-moe-card"))
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, **kw).eval()
    t = time.time()
    with torch.no_grad():
        out = m.generate(torch.tensor([ids]), max_new_tokens=n, min_new_tokens=n,
                         do_sample=False, output_scores=True, return_dict_in_generate=True)
    new = [int(x) for x in out.sequences[0, len(ids):]]  # n tokens: past an end of turn too
    top = []                                    # per step HF's 8 best (id, logit): a card's
    for sc in out.scores:                       # different pick, by how much it lost
        v, i = torch.topk(sc[0].float(), 8)
        top.append([[int(a), round(float(b), 4)] for a, b in zip(i, v)])
    return dict(model=model, prompt=prompt, ids=ids, tokens=new, text=tok.decode(new),
                top=top, seconds=round(time.time() - t, 1))


def fit_experts(spec, cfg, cap: int, **kw) -> int:
    """The expert slots per MoE layer that fill the card's DRAM beside the rest of the image."""
    from dataclasses import replace
    probe = spec.image(replace(cfg, DRAM_BYTES=1 << 40), cap, rows=1, experts=spec.moe.k, **kw)
    L = probe.offload
    return min(spec.moe.E, (cfg.DRAM_BYTES - L.slots[0][0]) // (L.layers * L.slot_bytes))


def card(model: str, ref: dict, n: int, experts: int, cap: int, pool: str | None,
         host_loop: bool = False, embed: str | None = None, trace: str | None = None) -> dict:
    from dataclasses import replace
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import Engine, LazyWeights
    spec = load_spec(model)                     # a MoE's Spec.embed: "int8" (from_hf)
    if embed is not None:
        spec = replace(spec, embed=embed)
    W = LazyWeights(model)
    if not experts:
        experts = fit_experts(spec, board_config(), cap, wformat="fp4", head_format="int8",
                              lookup=True)
    t = time.time()
    eng = Engine(spec, W, cap=cap, cfg=board_config(), rows=1, wformat="fp4",
                 head_format="int8", resident=True, experts=experts, pool_file=pool)
    load_s = time.time() - t
    srv = eng.server
    srv.history, per_req = [], []               # each request's ids and misses
    serve = srv.serve

    def counted(ids_):
        m0 = srv.misses
        serve(ids_)
        per_req.append(srv.misses - m0)
    srv.serve = counted
    ids = ref["ids"]
    t = time.time()
    lg = eng.prefill(ids if host_loop else ids[:-1])
    prefill_s = time.time() - t
    t = time.time()
    top = []                    # host loop: the device's 8 best (id, logit) per step

    def best(v):
        i = np.argsort(-v, kind="stable")[:8]
        top.append([[int(j), round(float(v[j]), 4)] for j in i])
        return int(i[0])
    if host_loop:               # the resident decode programs, the argmax on the host
        got = [best(lg)]
        for _ in range(n - 1):
            got.append(best(eng.step(got[-1])))
    else:                       # the prompt's last token fed by the card's loop: every pick
        got = eng.generate_card(ids[-1], n, stop_ids=[])        # on the card
    gen_s = time.time() - t
    L = eng.image.offload
    J, E = L.layers, L.E
    T = len(per_req) // J                       # whole tokens (the last request may be unread)
    mpt = np.array(per_req[:T * J]).reshape(T, J).sum(1)       # misses per token
    dec = mpt[len(ids):]
    if trace:                                   # the card's routes, as router_trace.py's
        req = np.array(srv.history[:T * J]).reshape(T, J, -1)
        first = spec.moe.first
        z = {f"L{first + j}_idx": (req[:, j] - j * E).astype(np.int16) for j in range(J)}
        z.update({f"L{first + j}_ok": np.float64(1.0) for j in range(J)})
        meta = dict(model=model, moe_layers=[first + j for j in range(J)], experts=E,
                    k=spec.moe.k, source="moe_card: the card's routes")
        np.savez(trace, meta=json.dumps(meta), **z)
    diff = next((i for i, (a, b) in enumerate(zip(got, ref["tokens"])) if a != b), None)
    at = None
    if diff is not None and "top" in ref:       # HF's 8 best at the first different pick
        at = dict(hf=ref["tokens"][diff], card=got[diff], hf_top=ref["top"][diff],
                  card_top=top[diff] if top else None)
    return dict(tokens=got, match=got == ref["tokens"][:len(got)], first_diff=diff,
                at_first_diff=at,
                experts_per_layer=experts, slots=L.layers * experts, pool=L.layers * L.E,
                image_mib=round(eng.image.nbytes / 2**20), slot_mb=round(L.slot_bytes / 1e6, 2),
                requests=srv.seq, hits=srv.hits, misses=srv.misses,
                misses_per_token_decode=round(float(dec.mean()), 2) if len(dec) else None,
                expert_uses_per_token=J * spec.moe.k,
                misses_per_token_decode_2nd_half=round(float(dec[len(dec) // 2:].mean()), 2)
                if len(dec) else None,
                misses_per_token=mpt.tolist(),
                load_s=round(load_s), prefill_s=round(prefill_s), generate_s=round(gen_s),
                loop="host" if host_loop else "card", embed=spec.embed, top=top or None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--hf", help="write HF's greedy tokens here")
    ap.add_argument("--check", help="HF's tokens (--hf's output) to compare the card's with")
    ap.add_argument("-n", type=int, default=16, help="tokens to generate")
    ap.add_argument("--prompt", default=PROMPT, help="HF: the user turn")
    ap.add_argument("--experts", type=int, default=0,
                    help="expert slots per MoE layer (0: as many as fit)")
    ap.add_argument("--cap", type=int, default=4096, help="KV capacity")
    ap.add_argument("--pool", help="the expert pool file (each expert packed once, reused)")
    ap.add_argument("--max-memory", help="HF: host RAM for weights, the rest to disk (e.g. 10GiB)")
    ap.add_argument("--out", help="the card's result as JSON")
    ap.add_argument("--embed", choices=["int8", "f32"], default=None,
                    help="Spec.embed (default the model's: int8 for a MoE, the rows gathered on "
                         "the device from the int8 head or table), or an fp32 table (1 GiB for "
                         "LFM2.5-8B-A1B)")
    ap.add_argument("--trace", help="save the card's routes as a router trace (cachesim.py)")
    ap.add_argument("--host-loop", action="store_true",
                    help="decode with the resident step programs and the argmax on the host, "
                         "the device's 8 best logits per step in the result (a diagnostic: "
                         "where a run parts from HF's)")
    a = ap.parse_args()
    if a.hf:
        r = hf_greedy(a.model, a.n, a.max_memory, a.prompt)
        Path(a.hf).write_text(json.dumps(r, indent=1))
        print(json.dumps(r))
        return
    ref = json.loads(Path(a.check).read_text())
    r = card(a.model, ref, a.n, a.experts, a.cap, a.pool, a.host_loop, a.embed, a.trace)
    print(json.dumps(r))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
