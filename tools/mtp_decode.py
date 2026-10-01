#!/usr/bin/env python3
"""MTP phase 2 (docs/mtp.md 9): greedy speculative decoding with Qwen3.5's MTP drafter on the ISA
simulator, against plain greedy decode on it, for a few of phase 0's prompts.

Per prompt (its chat template, thinking off) both engines generate up to --tokens tokens: plain
greedy decode (Engine.generate, resident decode) and the MTP loop (opentpu/llm/mtp.py). The tokens
must be equal. The JSON records, per prompt, the tokens, every iteration's acceptance (0, 1) and
the device runs by kind; --cycles turns them into the k = 1 loop's speedup in cycles from the
co-simulated runs (tools/perf_qwen.py: the decode step, --mtp verify, --mtp draft):

    speedup = (tokens - 1) x decode step / (verify runs x verify + MTP runs x draft)

(the first token comes from the prefill in both).

    OTPU_MCOLS=4 OTPU_PAIR=1 OTPU_DSTEP=1 OTPU_STREAM=1 tools/mtp_decode.py \\
        --model models/Qwen3.5-0.8B --prompts 0,3,7 --tokens 48 --out q35-0.8b.json
    tools/mtp_decode.py --summary q35-0.8b.json --cycles STEP,VERIFY,DRAFT
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def run(a) -> None:
    from transformers import AutoTokenizer
    from mtp_accept import PROMPTS
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec, model_dir
    from opentpu.llm.mtp import MTPDecoder, mtp_engine
    from opentpu.llm.qwen3 import Engine, load_weights

    path = model_dir(a.model)
    spec = load_spec(path)
    W = load_weights(path, mtp=True)
    tok = AutoTokenizer.from_pretrained(path)
    wkw = dict(wformat=a.wformat, head_format=a.head_format)
    need = replace(spec, mtp=True).image(board_config(DRAM_BYTES=1 << 40), a.cap, lookup=True,
                                         rows=8, **wkw).nbytes
    cfg = board_config(DRAM_BYTES=1 << max(20, (need - 1).bit_length()))
    print(f"{path.name}: DRAM {cfg.DRAM_BYTES >> 20} MiB, MCOLS {cfg.MCOLS} PAIR {cfg.PAIR} "
          f"DSTEP {cfg.DSTEP} STREAM {cfg.STREAM}", flush=True)
    res = {"model": path.name, "wformat": a.wformat, "head_format": a.head_format,
           "cfg": {k: getattr(cfg, k) for k in ("MCOLS", "PAIR", "DSTEP", "STREAM")},
           "prompts": []}
    prompts = []
    for i in [int(x) for x in a.prompts.split(",")]:
        kind, text = PROMPTS[i]
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, enable_thinking=False,
                                      tokenize=True)
        prompts.append((i, kind, list(ids["input_ids"] if hasattr(ids, "keys") else ids)))
    # plain greedy decode first, then the MTP loop: one engine (one DRAM image) at a time
    t0 = time.time()
    eng = Engine(spec, W, cap=a.cap, cfg=cfg, resident=True, **wkw)
    print(f"plain engine built in {time.time() - t0:.0f} s", flush=True)
    wants = []
    for i, kind, ids in prompts:
        t0 = time.time()
        eng.reset()
        wants.append((eng.generate(ids, max_new=a.tokens), round(time.time() - t0, 1)))
        print(f"prompt {i}: plain greedy in {wants[-1][1]} s", flush=True)
    del eng
    t0 = time.time()
    eng = mtp_engine(spec, W, cap=a.cap, cfg=cfg, **wkw)
    print(f"MTP engine built in {time.time() - t0:.0f} s", flush=True)
    for (i, kind, ids), (want, tw) in zip(prompts, wants):
        t0 = time.time()
        eng.reset()
        st = MTPDecoder(eng).generate(ids, max_new=a.tokens)
        runs = {}
        for k, _, _ in st.runs:
            runs[k] = runs.get(k, 0) + 1
        r = {"index": i, "kind": kind, "prompt": len(ids), "equal": st.tokens == want,
             "tokens": st.tokens, "plain": want, "accepted": st.accepted, "runs": runs,
             "seconds": [tw, round(time.time() - t0, 1)]}
        res["prompts"].append(r)
        print(f"prompt {i} ({kind}, {len(ids)} tokens): equal {r['equal']}, {len(st.tokens)} "
              f"tokens in {st.iterations} iterations, acceptance {st.acceptance:.2f}, runs "
              f"{runs}, {r['seconds']} s; {tok.decode(st.tokens)[:80]!r}", flush=True)
        if a.out:
            Path(a.out).write_text(json.dumps(res))


def summary(files, cycles) -> None:
    step, verify, draft = cycles if cycles else (None, None, None)
    for f in files:
        r = json.loads(Path(f).read_text())
        tot_t = tot_v = tot_m = 0
        for p in r["prompts"]:
            n, v, m = len(p["tokens"]), p["runs"].get("verify", 0), p["runs"].get("mtp", 0)
            tot_t, tot_v, tot_m = tot_t + n - 1, tot_v + v, tot_m + m
            acc = sum(p["accepted"]) / max(1, len(p["accepted"]))
            s = f"{r['model']:14s} prompt {p['index']} {p['kind']:8s} equal {p['equal']} " \
                f"tokens {n:3d} iterations {len(p['accepted']):3d} acceptance {acc:.2f}"
            if step:
                s += f" speedup {(n - 1) * step / (v * verify + m * draft):.3f}x"
            print(s)
        if step:
            sp = tot_t * step / (tot_v * verify + tot_m * draft)
            print(f"{r['model']:14s} all: {tot_t} tokens after the first, {tot_v} verify and "
                  f"{tot_m} MTP runs: speedup {sp:.3f}x (c_2 {verify / step:.3f}, c_draft "
                  f"{draft / step:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen35")
    ap.add_argument("--prompts", default="0,3,7", help="indices into mtp_accept.PROMPTS")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--wformat", default="fp4", choices=["int8", "int4", "fp4"])
    ap.add_argument("--head-format", default="int8", choices=["int8", "int4", "fp4"])
    ap.add_argument("--out")
    ap.add_argument("--summary", nargs="+")
    ap.add_argument("--cycles", help="decode step, verify and draft cycles (perf_qwen)")
    a = ap.parse_args()
    if a.summary:
        summary(a.summary, [float(x) for x in a.cycles.split(",")] if a.cycles else None)
        return
    run(a)


if __name__ == "__main__":
    main()
