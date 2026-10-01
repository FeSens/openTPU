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

--card CFG.pkl runs both on the card (under otpu-lock; CFG from tools/qual/refs.py cfg, e.g.
~/otpu-build/production/qual/cfg.pkl): plain greedy decode is the production loop (prefill,
then Engine.generate_card: the tokens picked and fed back on the card), the MTP loop is
phase 2's host-driven one (a verify and a draft program compiled per iteration on the host).
The JSON then holds each one's device cycles and wall seconds, the host's compile seconds and
each verify run's slot parity; --summary prints tok/s. --want FILE (an ISA simulator run's
JSON) checks the tokens against it too. --prebuild builds both images into the image cache
(opentpu/qcache.py) without the card, outside the lock, waiting while a session holds the
host's quiet file.
"""
from __future__ import annotations

import argparse
import json
import pickle
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
    if a.card:
        cfg = pickle.loads(Path(a.card).read_bytes())
    else:
        need = replace(spec, mtp=True).image(board_config(DRAM_BYTES=1 << 40), a.cap,
                                             lookup=True, rows=8, **wkw).nbytes
        cfg = board_config(DRAM_BYTES=1 << max(20, (need - 1).bit_length()))
    print(f"{path.name}: DRAM {cfg.DRAM_BYTES >> 20} MiB, MCOLS {cfg.MCOLS} PAIR {cfg.PAIR} "
          f"DSTEP {cfg.DSTEP} STREAM {cfg.STREAM}" + (" (card)" if a.card else ""), flush=True)
    if a.prebuild:
        prebuild(cfg, spec, W, a)
        return
    res = {"model": path.name, "wformat": a.wformat, "head_format": a.head_format,
           "cfg": {k: getattr(cfg, k) for k in ("MCOLS", "PAIR", "DSTEP", "STREAM")},
           "card": bool(a.card), "prompts": []}
    want_sim = {}
    if a.want:
        sim = json.loads(Path(a.want).read_text())["prompts"]
        want_sim = {p["index"]: p["tokens"] for p in sim}
    prompts = []
    for i in [int(x) for x in a.prompts.split(",")]:
        kind, text = PROMPTS[i]
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, enable_thinking=False,
                                      tokenize=True)
        prompts.append((i, kind, list(ids["input_ids"] if hasattr(ids, "keys") else ids)))
    # plain greedy decode first, then the MTP loop: one engine (one DRAM image) at a time
    t0 = time.time()
    eng = Engine(spec, W, cap=a.cap, cfg=cfg, resident=True, **wkw, **_backend(a, path))
    print(f"plain engine built in {time.time() - t0:.0f} s", flush=True)
    wants = []
    for i, kind, ids in prompts:
        t0 = time.time()
        eng.reset()
        if a.card:                      # the production loop: picked and fed on the card
            got, pl = _plain_card(eng, spec, ids, a.tokens)
        else:
            got, pl = eng.generate(ids, max_new=a.tokens), {}
        pl["seconds"] = round(time.time() - t0, 1)
        wants.append((got, pl))
        print(f"prompt {i}: plain greedy {pl}", flush=True)
    _close(eng)
    del eng
    t0 = time.time()
    eng = mtp_engine(spec, W, cap=a.cap, cfg=cfg, **wkw, **_backend(a, path))
    print(f"MTP engine built in {time.time() - t0:.0f} s", flush=True)
    for (i, kind, ids), (want, pl) in zip(prompts, wants):
        t0 = time.time()
        eng.reset()
        st = MTPDecoder(eng).generate(ids, max_new=a.tokens)
        dt = time.time() - t0
        runs, cyc = {}, {}
        for k, _, s_ in st.runs:
            runs[k] = runs.get(k, 0) + 1
            cyc[k] = cyc.get(k, 0) + int(s_.get("cycles", 0) if isinstance(s_, dict) else 0)
        r = {"index": i, "kind": kind, "prompt": len(ids), "equal": st.tokens == want,
             "tokens": st.tokens, "plain": want, "accepted": st.accepted, "runs": runs,
             "seconds": [pl["seconds"], round(dt, 1)]}
        if i in want_sim:
            r["equal_sim"] = st.tokens == want_sim[i] and want == want_sim[i]
        if a.card:
            r["plain_card"] = pl
            r["mtp_card"] = {"cycles": cyc, "wall": round(dt, 3),
                             "prefill_wall": round(st.prefill_s, 3),
                             "compile": round(st.compile_s, 3),
                             "prefill_compile": round(st.prefill_compile_s, 3),
                             "slots": [st.slots.count(0), st.slots.count(1)],
                             "core_khz": eng.backend.info["core_khz"]}
        res["prompts"].append(r)
        print(f"prompt {i} ({kind}, {len(ids)} tokens): equal {r['equal']}"
              + (f" (ISA simulator {r['equal_sim']})" if "equal_sim" in r else "")
              + f", {len(st.tokens)} tokens in {st.iterations} iterations, acceptance "
              f"{st.acceptance:.2f}, runs {runs}, {r['seconds']} s"
              + (f", {r['mtp_card']}" if a.card else "")
              + f"; {tok.decode(st.tokens)[:80]!r}", flush=True)
        if a.out:
            Path(a.out).write_text(json.dumps(res))
    _close(eng)


def _backend(a, path) -> dict:
    """--card: the Engine's backend, the card (a transport per engine; BoardBackend holds the
    device lock until it closes)."""
    if not a.card:
        return {}
    from opentpu.host.board import BoardBackend, XdmaTransport
    tr = XdmaTransport(a.dev)
    return {"backend": lambda c, imgs: BoardBackend(c, imgs, transport=tr, model=path.name)}


def _close(eng) -> None:
    close = getattr(eng.backend, "close", None)
    if close is not None:
        getattr(eng, "_drain", lambda: None)()
        close()


def _plain_card(eng, spec, ids, n):
    """Plain greedy decode on the card as the production loop runs it: the prefill, then
    Engine.generate_card (each token picked and fed back on the card); the tokens, and the
    decode's device cycles and wall seconds (the first token is the prefill's)."""
    import numpy as np
    t0 = time.perf_counter()
    got = [int(np.argmax(eng.prefill(ids)))]
    t1 = time.perf_counter()
    k = len(eng.stats)
    if got[0] not in spec.eos and n > 1:
        got += eng.generate_card(got[0], n - 1)
    t2 = time.perf_counter()
    return got, {"cycles": int(sum(s["cycles"] for s in eng.stats[k:])), "runs": len(eng.stats) - k,
                 "prefill_wall": round(t1 - t0, 3), "wall": round(t2 - t1, 3),
                 "core_khz": eng.backend.info["core_khz"]}


def prebuild(cfg, spec, W, a) -> None:
    """Both engines' images into the image cache (the card's configuration, no card), after
    any session holding the host's quiet file is done (opentpu/host/prebuild.py)."""
    from opentpu import qcache
    from opentpu.host import prebuild as pb
    t0 = time.time()
    while pb.quiet():
        if time.time() - t0 > a.quiet_wait:
            print(f"prebuild: {pb.quiet_file()} still held after {a.quiet_wait:.0f} s: not built")
            return
        time.sleep(pb.POLL)
    for name, sp in (("plain", spec), ("MTP", replace(spec, mtp=True))):
        t1 = time.time()
        n = pb.prebuild(cfg, sp, W, a.wformat, a.head_format, cap=a.cap)
        print(f"prebuild {name}: {n} ({time.time() - t1:.0f} s), cache {qcache.cache_dir()}",
              flush=True)


def card_summary(r) -> None:
    """A --card run's tok/s: device (the runs' cycles at the core clock) and wall, decode
    after the prefill's token, plain (the card's loop) against MTP (host-driven)."""
    tot = {"n": 0, "pc": 0, "pw": 0.0, "mc": 0, "mw": 0.0, "comp": 0.0, "acc": [],
           "slots": [0, 0], "v": [0, 0], "d": [0, 0]}
    for p in r["prompts"]:
        pl, mt = p["plain_card"], p["mtp_card"]
        hz = 1e3 * mt["core_khz"]
        n = len(p["tokens"]) - 1
        mc = mt["cycles"].get("verify", 0) + mt["cycles"].get("mtp", 0)
        mw = mt["wall"] - mt["prefill_wall"]
        comp = mt["compile"] - mt["prefill_compile"]
        tot["n"] += n
        tot["pc"] += pl["cycles"]
        tot["pw"] += pl["wall"]
        tot["mc"] += mc
        tot["mw"] += mw
        tot["comp"] += comp
        tot["acc"] += p["accepted"]
        tot["slots"] = [a + b for a, b in zip(tot["slots"], mt["slots"])]
        for k, key in (("v", "verify"), ("d", "mtp")):
            tot[k] = [tot[k][0] + mt["cycles"].get(key, 0), tot[k][1] + p["runs"].get(key, 0)]
        print(f"{r['model']:14s} prompt {p['index']} {p['kind']:8s} equal {p['equal']}"
              + (f" sim {p['equal_sim']}" if "equal_sim" in p else "")
              + f" tokens {n + 1:3d} acceptance "
              f"{sum(p['accepted']) / max(1, len(p['accepted'])):.2f}"
              f" | device tok/s plain {n * hz / pl['cycles']:.2f} MTP {n * hz / mc:.2f} "
              f"({pl['cycles'] / mc:.3f}x) | wall tok/s plain {n / pl['wall']:.2f} MTP "
              f"{n / mw:.2f} (host compile {comp:.2f} of {mw:.2f} s) | verify parities "
              f"{mt['slots']}")
    hz = 1e3 * r["prompts"][0]["mtp_card"]["core_khz"]
    n = tot["n"]
    print(f"{r['model']:14s} all: {n} tokens after the first, acceptance "
          f"{sum(tot['acc']) / max(1, len(tot['acc'])):.2f}; device tok/s plain "
          f"{n * hz / tot['pc']:.2f} MTP {n * hz / tot['mc']:.2f} ({tot['pc'] / tot['mc']:.3f}x); "
          f"wall tok/s plain {n / tot['pw']:.2f} MTP {n / tot['mw']:.2f} (host compile "
          f"{tot['comp']:.1f} of {tot['mw']:.1f} s; without it "
          f"{n / (tot['mw'] - tot['comp']):.2f}); verify parities {tot['slots']}")
    step = tot["pc"] / n                # the card loop's cycles per token
    print(f"{r['model']:14s} per run: decode step {step:,.0f} cycles (the card's loop), verify "
          f"{tot['v'][0] / tot['v'][1]:,.0f} (c_2 {tot['v'][0] / tot['v'][1] / step:.3f}), draft "
          f"{tot['d'][0] / tot['d'][1]:,.0f} (c_draft {tot['d'][0] / tot['d'][1] / step:.3f})")


def summary(files, cycles) -> None:
    step, verify, draft = cycles if cycles else (None, None, None)
    for f in files:
        r = json.loads(Path(f).read_text())
        if r.get("card"):
            card_summary(r)
            continue
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
    ap.add_argument("--card", help="run on the card: its configuration (refs.py cfg pickle)")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--want", help="an ISA simulator run's JSON: the tokens to equal too")
    ap.add_argument("--prebuild", action="store_true",
                    help="build both images into the image cache, no card (with --card CFG)")
    ap.add_argument("--quiet-wait", type=float, default=3600.0)
    ap.add_argument("--summary", nargs="+")
    ap.add_argument("--cycles", help="decode step, verify and draft cycles (perf_qwen)")
    a = ap.parse_args()
    if a.summary:
        summary(a.summary, [float(x) for x in a.cycles.split(",")] if a.cycles else None)
        return
    run(a)


if __name__ == "__main__":
    main()
