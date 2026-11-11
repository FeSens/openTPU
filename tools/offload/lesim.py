#!/usr/bin/env python3
"""Lookahead expert scheduling (LES, OpenBMB's llama.cpp-omni) on path (a): how well the next
layer's experts can be predicted, and what prefetching them would buy on the card (docs/offload.md
15). Offline, on offload's router traces (tools/offload/router_trace.py; ~/otpu-build/offload/traces
on the build host: <model>.<text>.npz, four texts of 2048 tokens), each text warmed by the other three's
profile (cachesim.py's rule), the server's per-layer slots by decayed use (lfu_layer).

  recall MODEL SLOTS MB TRACE...   the predictions against the routes and against the server's
      misses. Per predictor, the share of a layer's true top-k its k' best name (k' = 8, 12, 16),
      the share of the server's misses they name, and cachesim.prefetch's capped prefetch (before
      each request the first n of the predictor's first `top` ids not in a slot take one, the
      victim of least decayed use outside the prediction, no use counted): misses a token, sent,
      useful, MB early and MB wasted.
  time q35|g26 TRACE... [--slots S] [--modes ...] [--gates 1,0] [--set K=V]   the decode's
      timeline per MoE request, with and without prefetch for the next layer:
        seen  = post + DET
        crit  = HEAD + REC x misses, after any part in flight (the link)
        next post = max(crit end + F, post + W0 + WM x CT x MB x misses)
                    + CT x MB for each prefetched expert landing while the card computes
        a token: the layers' requests, then C0 (the LM head, sampling)
      F, W0: the card's own time after a request's critical end, and a request's with no miss
      (card_fit.py w on session 16's A' runs, docs/offload.md 10.13); HEAD, REC: session 16's
      seen-to-crit fit; CT: H2C writes during the card's compute, 0.17 s a GB (13.8); PART: a
      whole expert's idle part (13.12, pfv2). C0 makes the base match the card's token at its
      miss rate (35B: session 16's 102.85 misses, 190 ms). Prefetch (modes): the predictor's
      first `top` ids, the first n not in a slot take one (least decayed use outside the
      prediction, no use counted) and go on the link after the request's demand, from when the
      prediction is known: oracle (the next layer's true misses), prev_r and prev_in at this
      layer's request (posted with it), pre at this layer's output (its critical end + TAIL, or
      post + W0 - MIX with no miss). gate 1: a part starts only if it lands before the next post
      is due (else it stays queued: a plain miss if named); 0: a request waits for the part in
      flight. A request finds a landed prefetch a hit, one in flight finished (its rest, no
      head), a queued one a miss; unnamed queued ones are withdrawn. Not modelled: the
      predictor's own card time (about 50 us a layer: the next router's int8 MM and the top-k)
      and any host cost of a hint.

Predictors (router_trace.py's, each the layer's router through its norm on an earlier state):
pre (the layer's input: one mixer ahead), prev_r (the previous layer's router input: one layer
ahead, Fate's cross-layer gate), prev_in (the previous layer's input: one layer and one mixer
ahead; the card has it as early as prev_r).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cachesim import decayed, freq, load, prefetch, simulate, top_set, victim  # noqa: E402

# ms (MB an expert), Gen2 x8, the card's A' host (docs/offload.md 10.13). The 35B: session 16's fit
# (F, W0, HEAD 1.43 s over 4649 requests, REC 9.89 s over 13,165 misses), MX experts of 1.711 MB;
# the 26B: session 16's F, W0 and W(m) (W(m) - W0 ~ the demand's own write contention, WM 1), its
# record scaled to the nvfp4 experts' 3.81 MB, whole idle parts 1.68-1.73 ms at 3.45 MB (pfv2)
MODELS = {
    "q35": dict(DET=0.045, HEAD=0.308, REC=0.751, F=1.705, W0=2.47, MIX=1.30, TAIL=0.40,
                PART=0.88, CT=0.17, MB=1.711, C0=29.0, WM=0.0, slots=41),
    "g26": dict(DET=0.045, HEAD=0.33, REC=1.75, F=2.956, W0=5.70, MIX=2.6, TAIL=0.5,
                PART=1.90, CT=0.17, MB=3.81, C0=35.0, WM=1.0, slots=16),
}
PREDICTORS = ("pre", "prev_r", "prev_in")


def warm_set(traces, i, C):
    """Text i's warm slots: the other texts' profile (cachesim's rule)."""
    t = traces[i]
    T, L, k = t["req"].shape
    E = t["E"]
    return top_set(sum(freq(o["req"], L * E) for j, o in enumerate(traces) if j != i), C)


def recall(traces, spl: int, mb: float) -> dict:
    """The recall tables (module docstring), means over the texts."""
    agg: dict = {}
    E = traces[0]["E"]
    for i, t in enumerate(traces):
        req = t["req"]
        T, L, k = req.shape
        warm = warm_set(traces, i, spl * L)
        miss_tok, miss_at = simulate(req, spl * L, "lfu_layer", warm)
        agg.setdefault("base", []).append(miss_tok.mean())
        for name in PREDICTORS:
            p = t["preds"].get(name)
            if p is None:
                continue
            for kk in (8, 12, 16):
                if p[1].shape[1] < kk:
                    continue
                rec = np.mean([np.mean([len(set(p[j][s, :kk].tolist()) & set(req[s, j].tolist())) / k
                                        for s in range(T)]) for j in range(L) if p[j] is not None])
                hit = tot = 0
                for (s, j), m in miss_at.items():
                    if p[j] is not None:
                        got = set(p[j][s, :kk].tolist())
                        hit += sum(e in got for e in m)
                        tot += len(m)
                agg.setdefault(f"{name} topk@{kk}", []).append(rec)
                agg.setdefault(f"{name} miss@{kk}", []).append(hit / tot if tot else 1.0)
            for top, n in ((4, 1), (4, 2), (4, 4), (8, 1), (8, 2), (8, 4), (8, 8)):
                agg.setdefault(f"{name} pf {top} {n}", []).append(
                    prefetch(req, p, E, spl * L, warm, n, top))
    return {key: np.mean(np.array(v, float), 0).tolist() for key, v in agg.items()}


def run(req, pred, slots: int, warm, E: int, p: dict, mode: str = "none", top: int = 8,
        n: int = 0, gate: bool = True, half: float = 32.0):
    """One text's decode under the timeline model (module docstring). Returns ms a token,
    misses, prefetches sent (slots taken), useful (named while in their slot) and landed, each a
    token."""
    T, L, k = req.shape
    cs = [OrderedDict() for _ in range(L)]
    for e in warm:
        if len(cs[e // E]) < slots:
            cs[e // E][e] = None
    uses = [dict.fromkeys(c, 0.0) for c in cs]
    cs = [OrderedDict.fromkeys(reversed(c)) for c in cs]
    pf = [dict() for _ in range(L)]         # per layer, prefetched expert -> [q|f|l, landing]
    ct = p["CT"] * p["MB"]
    t = link = 0.0
    miss = sent = useful = landed = 0
    for tok in range(T):
        for j in range(L):
            post = t
            seen = post + p["DET"]
            ids = [int(e) for e in req[tok, j]]
            decayed(uses[j], ids, (tok + 1) / half)
            c, q = cs[j], pf[j]
            plain, rest = 0, 0.0
            for e in ids:
                if e in c:
                    c.move_to_end(e)
                    st = q.pop(e, None)
                    if st is None:
                        continue
                    if st[0] == "l" or (st[0] == "f" and st[1] <= seen):
                        useful += 1                 # landed before the request
                    elif st[0] == "f":              # on its way: its rest now
                        rest += st[1] - seen
                        useful += 1
                        miss += 1
                    else:                           # queued, not started: a miss
                        plain += 1
                        miss += 1
                    continue
                plain += 1
                miss += 1
                if len(c) >= slots:
                    v = victim(c, ids, uses[j])
                    del c[v]
                    q.pop(v, None)
                c[e] = None
            for e in list(q):                       # the unnamed: queued withdrawn, the rest stay
                if q.pop(e)[0] == "q" and e in c:
                    del c[e]
            start = max(seen, link)
            crit = (start + p["HEAD"] + p["REC"] * plain + rest if plain else
                    start + rest if rest else None)
            link = crit if crit is not None else max(link, seen)
            nxt = post + p["W0"] + p["WM"] * ct * (plain + (rest > 0))
            if crit is not None:
                nxt = max(nxt, crit + p["F"])
            if mode != "none" and n and j + 1 < L:
                jn, cn = j + 1, cs[j + 1]
                if mode == "oracle":
                    hint = [int(e) for e in req[tok, jn]]
                    cand = [e for e in hint if e not in cn][:n]
                else:
                    hint = [int(e) for e in pred[jn][tok][:k]]
                    cand = [int(e) for e in pred[jn][tok][:top] if int(e) not in cn][:n]
                if mode == "pre":                   # layer j's output, then its post
                    known = (crit + p["TAIL"] if crit is not None else
                             post + p["W0"] - p["MIX"]) + p["DET"]
                else:                               # with layer j's request
                    known = seen
                for e in cand:
                    if len(cn) >= slots:
                        v = victim(cn, hint, uses[jn])
                        del cn[v]
                        pf[jn].pop(v, None)
                    cn[e] = None
                    pf[jn][e] = ["q", 0.0]
                sent += len(cand)
                tl, n_land = max(link, known), 0
                for e in cand:
                    end = tl + p["PART"]
                    if gate and end > nxt + (n_land + 1) * ct:
                        break
                    pf[jn][e] = ["f", end]
                    tl, n_land = end, n_land + 1
                landed += n_land
                link = max(link, tl)
                nxt += n_land * ct
            t = nxt
        t += p["C0"]
    return t / T, miss / T, sent / T, useful / T, landed / T


def timeline(traces, p: dict, slots: int, variants) -> dict:
    """run() per text and variant (mode, top, n, gate); the means over the texts."""
    out: dict = {}
    E = traces[0]["E"]
    for i, t in enumerate(traces):
        warm = warm_set(traces, i, slots * t["req"].shape[1])
        for v in variants:
            mode, top, n, gate = v
            pred = t["preds"].get(mode) if mode in PREDICTORS else None
            out.setdefault(v, []).append(run(t["req"], pred, slots, warm, E, p, mode, top, n, gate))
    return {v: np.mean(np.array(r), 0).tolist() for v, r in out.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recall")
    r.add_argument("model")
    r.add_argument("slots", type=int, help="slots a MoE layer")
    r.add_argument("mb", type=float, help="MB an expert")
    r.add_argument("traces", nargs="+")
    r.add_argument("--json")
    s = sub.add_parser("time")
    s.add_argument("model", choices=sorted(MODELS))
    s.add_argument("traces", nargs="+")
    s.add_argument("--slots", type=int, help="slots a MoE layer (default the card's)")
    s.add_argument("--modes", default="oracle,pre,prev_in,prev_r")
    s.add_argument("--tops", default="2,4,8")
    s.add_argument("--ns", default="1,2,3,4")
    s.add_argument("--gates", default="1,0")
    s.add_argument("--set", action="append", default=[], metavar="K=V", help="a model constant")
    s.add_argument("--json")
    a = ap.parse_args(argv)
    traces = [load(x) for x in a.traces]
    if a.cmd == "recall":
        res = recall(traces, a.slots, a.mb)
        base = res["base"]
        print(f"{a.model}: {a.slots} slots a layer; base {base:.2f} misses a token "
              f"({base * a.mb:.1f} MB)")
        print("predictor  true top-k by its 8/12/16 best    the server's misses by its 8/12/16 best")
        for name in PREDICTORS:
            f = " ".join(f"{res[f'{name} topk@{kk}']:.3f}" for kk in (8, 12, 16) if f"{name} topk@{kk}" in res)
            g = " ".join(f"{res[f'{name} miss@{kk}']:.3f}" for kk in (8, 12, 16) if f"{name} miss@{kk}" in res)
            print(f"  {name:8s} {f:30s} {g}")
        print("prefetch: predictor top n -> misses a token (change), sent, useful, precision, MB early, MB wasted")
        for key, v in res.items():
            if " pf " in key:
                m, snt, use = v
                print(f"  {key.replace(' pf', ''):14s} -> {m:6.2f} ({(m / base - 1) * 100:+5.1f}%) "
                      f"{snt:6.2f} {use:6.2f} {use / snt if snt else 0:.2f} {use * a.mb:6.1f} "
                      f"{(snt - use) * a.mb:6.1f}")
        if a.json:
            json.dump(res, open(a.json, "w"), indent=1)
        return 0
    p = dict(MODELS[a.model])
    for kv in a.set:
        key, val = kv.split("=")
        if key not in p:
            raise SystemExit(f"no model constant {key!r}: {', '.join(p)}")
        p[key] = float(val)
    slots = a.slots or int(p["slots"])
    variants = [("none", 0, 0, True)]
    for mode in a.modes.split(","):
        for top in ((8,) if mode == "oracle" else [int(x) for x in a.tops.split(",")]):
            for n in [int(x) for x in a.ns.split(",")]:
                if n <= top:
                    variants += [(mode, top, n, g == "1") for g in a.gates.split(",")]
    res = timeline(traces, p, slots, variants)
    b = res[variants[0]]
    print(f"{a.model}, {slots} slots a layer, {', '.join(f'{k} {v:g}' for k, v in p.items() if k != 'slots')}")
    print(f"base: {b[0]:.1f} ms a token ({1000 / b[0]:.2f} tok/s), {b[1]:.1f} misses a token")
    print("mode     top  n gate -> ms/token tok/s   gain   misses  sent useful landed precision")
    for v, x in res.items():
        print(f"{v[0]:8s} {v[1]:3d} {v[2]:2d} {'g' if v[3] else 'u'}    -> {x[0]:7.1f} {1000 / x[0]:5.2f} "
              f"{(b[0] / x[0] - 1) * 100:+6.1f}%  {x[1]:6.1f} {x[2]:5.1f} {x[3]:6.1f} {x[4]:6.1f}  "
              f"{x[3] / x[2] if x[2] else 0:.2f}")
    if a.json:
        json.dump(dict(p=p, slots=slots, res=[dict(v=list(v), r=x) for v, x in res.items()]),
                  open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
