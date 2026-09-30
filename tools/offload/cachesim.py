"""Expert-cache hit rates on router traces (tools/offload/router_trace.py; docs/offload.md).

    python3 tools/offload/cachesim.py TRACE.npz [TRACE.npz ...] [--frac 0.1,0.2,...]
                                      [--slots N,...] [--json out.json]
                                      [--survey survey.json [--repo NAME]] [--host-frac F,...]

The traces of one model on several texts. Each is replayed in decode order (token by token,
layer by layer; a layer asks for its top-k experts at once) against a cache of C expert slots
shared by all layers (every expert of a model has the same size), for each policy:

- `static`: the C experts most often picked in the other texts (a profile), never replaced;
- `lru`: least recently used, replaced on a miss;
- `lru_layer`: the same with the slots split evenly over the layers (a token sweeps the layers in
  order, a loop that a global LRU smaller than the sweep thrashes on);
- `lfu`: least often used (counts so far, the profile's counts as a prior), ties by recency;
- `opt`: Belady's optimum (evict the expert needed furthest in the future): an upper bound.

Every policy starts from the static set (the host fills the cache from a profile at load).
Per policy and size: the hit rate, and the misses per token (mean, p50, p90, max): a token's
time depends on its misses. Then, for the per-layer LRU cache, the share of its misses that a
prediction would have named early enough to prefetch (`pre`: the layer's router on the layer's
input, one mixer ahead; `prev_r` / `prev_in`: one layer ahead), taking the prediction's k or 2k
best.

With `--survey` (tools/offload/survey.py --json): the card's cache size (4 GiB less the model's
on-card part and a reserve) joins the sizes, and each replay becomes tokens per second under the
strategies of docs/offload.md section 4 (`tok_times`). `--host-frac`: the SSD tier, a host RAM
holding that fraction of the pool (global LRU) over the SSD.
"""
from __future__ import annotations

import argparse
import heapq
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    moe = meta["moe_layers"]
    E = meta["experts"] or int(max(z[f"L{l}_idx"].max() for l in moe)) + 1
    # requests[t, j, :]: the global expert ids (j-th MoE layer) token t asks for
    req = np.stack([z[f"L{l}_idx"].astype(np.int64) + j * E for j, l in enumerate(moe)], 1)
    preds = {}
    for name in ("pre", "prev_r", "prev_in"):
        if all(f"L{l}_{name}" in z for l in moe[1:]):
            p = [z[f"L{l}_{name}"].astype(np.int64) + j * E if f"L{l}_{name}" in z else None
                 for j, l in enumerate(moe)]
            preds[name] = p
    ok = min(float(z[f"L{l}_ok"]) for l in moe)
    return dict(meta=meta, E=E, req=req, preds=preds, ok=ok)


def freq(req, n):
    return np.bincount(req.reshape(-1), minlength=n)


def top_set(counts, C):
    return [int(i) for i in np.argsort(-counts, kind="stable")[:C]]


def simulate(req, C, policy, warm, prior=None):
    """Replay; returns misses per token and the per-(token, layer) miss sets."""
    T, L, k = req.shape
    miss_tok = np.zeros(T, np.int64)
    miss_at = {}
    if policy == "static":
        s = set(warm)
        for t in range(T):
            for j in range(L):
                m = [e for e in req[t, j] if e not in s]
                miss_tok[t] += len(m)
                if m:
                    miss_at[(t, j)] = m
        return miss_tok, miss_at
    if policy == "lru":
        c = OrderedDict((e, None) for e in warm)
        for t in range(T):
            for j in range(L):
                cur = req[t, j]
                m = []
                for e in cur:
                    e = int(e)
                    if e in c:
                        c.move_to_end(e)
                    else:
                        m.append(e)
                for e in m:
                    if len(c) >= C:
                        c.popitem(last=False)
                    c[e] = None
                miss_tok[t] += len(m)
                if m:
                    miss_at[(t, j)] = m
        return miss_tok, miss_at
    if policy == "lru_layer":
        E = int(req.max()) // L + 1 if prior is None else len(prior) // L
        caps = [C // L + (j < C % L) for j in range(L)]
        cs = [OrderedDict() for _ in range(L)]
        for e in warm:                                    # the profile's best of each layer
            j = e // E
            if len(cs[j]) < caps[j]:
                cs[j][e] = None
        for t in range(T):
            for j in range(L):
                c = cs[j]
                m = []
                for e in req[t, j]:
                    e = int(e)
                    if e in c:
                        c.move_to_end(e)
                    else:
                        m.append(e)
                for e in m:
                    if len(c) >= max(caps[j], k):
                        c.popitem(last=False)
                    c[e] = None
                miss_tok[t] += len(m)
                if m:
                    miss_at[(t, j)] = m
        return miss_tok, miss_at
    if policy == "lfu":
        cnt = dict(enumerate(prior.tolist())) if prior is not None else {}
        cache, heap, step = {}, [], 0
        for e in warm:
            cache[e] = (cnt.get(e, 0), 0)
            heapq.heappush(heap, (cnt.get(e, 0), 0, e))
        for t in range(T):
            for j in range(L):
                step += 1
                cur = set(int(e) for e in req[t, j])
                m = [e for e in cur if e not in cache]
                for e in cur:
                    cnt[e] = cnt.get(e, 0) + 1
                for e in m:
                    while len(cache) >= C:
                        f, s, v = heapq.heappop(heap)
                        if v in cache and cache[v] == (f, s) and v not in cur:
                            del cache[v]
                for e in cur:
                    cache[e] = (cnt[e], step)
                    heapq.heappush(heap, (cnt[e], step, e))
                miss_tok[t] += len(m)
                if m:
                    miss_at[(t, j)] = m
        return miss_tok, miss_at
    if policy == "opt":
        flat = req.reshape(-1)
        n = flat.size
        nxt = np.full(n, n + 1, np.int64)
        last = {}
        for i in range(n - 1, -1, -1):
            e = int(flat[i])
            nxt[i] = last.get(e, n + 1)
            last[e] = i
        first = last                                      # first use of each expert
        cache = {e: first.get(e, n + 1) for e in warm}
        heap = [(-v, e) for e, v in cache.items()]
        heapq.heapify(heap)
        i = 0
        for t in range(T):
            for j in range(L):
                idx = range(i, i + k)
                cur = set(int(flat[q]) for q in idx)
                m = [e for e in cur if e not in cache]
                for q in idx:                             # the layer's own experts stay
                    cache[int(flat[q])] = int(nxt[q])
                    heapq.heappush(heap, (-int(nxt[q]), int(flat[q])))
                keep = []
                while len(cache) > C:
                    v, e = heapq.heappop(heap)
                    if e in cache and cache[e] == -v:
                        if e in cur:
                            keep.append((v, e))
                        else:
                            del cache[e]
                for x in keep:
                    heapq.heappush(heap, x)
                i += k
                miss_tok[t] += len(m)
                if m:
                    miss_at[(t, j)] = m
        return miss_tok, miss_at
    raise ValueError(policy)


def coverage(miss_at, preds, width):
    """Share of the misses a prediction names (its `width` best)."""
    out = {}
    for name, p in preds.items():
        hit = tot = 0
        for (t, j), m in miss_at.items():
            if p[j] is None:
                continue
            s = set(p[j][t, :width].tolist())
            hit += sum(e in s for e in m)
            tot += len(m)
        out[name] = hit / tot if tot else 1.0
    return out


def accuracy(req, preds, k):
    """Mean share of a layer's top-k that the prediction's top-k names."""
    out = {}
    for name, p in preds.items():
        acc = [np.mean([len(set(p[j][t, :k]) & set(req[t, j])) / k for t in range(req.shape[0])])
               for j in range(req.shape[1]) if p[j] is not None]
        out[name] = float(np.mean(acc))
    return out


def tok_times(req, miss_at, preds, hw, by):
    """Tokens per second under each strategy (docs/offload.md section 4), from the per-layer
    misses of one replay. `by`: expert bytes `x`, dense bytes per MoE layer `d`, LM head bytes
    `head`; `hw`: the rates and costs (main's options)."""
    T, L, k = req.shape
    x, d, head = by["x"], by["d"], by["head"]
    Bd, Bh, sync, call = hw["dram"], hw["host"], hw["sync"], hw["call"]
    miss = np.zeros((T, L), np.int64)
    for (t, j), m in miss_at.items():
        miss[t, j] = len(m)
    hit_rate = 1 - miss.sum() / (T * L * k)
    td = d / Bd                                       # a layer's dense part on the card
    tk = k * x / Bd                                   # its k experts on the card
    out = {"resident": np.full(T, L * (td + tk) + head / Bd),
           "host_only": np.full(T, (L * (d + k * x) + head) / Bh)}
    for pc in hw["pcie"]:
        tx = x / pc + call                            # one expert over PCIe
        out[f"stream@{pc / 1e9:.1f}"] = (L * (td + tk + sync) + head / Bd) + miss.sum(1) * tx
        for name, win in (("pre", td), ("prev_r", td + tk)):
            if name not in preds:
                continue
            for w in (k, 2 * k):
                tt = np.full(T, L * (td + tk + sync) + head / Bd)
                for (t, j), m in miss_at.items():
                    p = preds[name][j]
                    if p is None:
                        tt[t] += len(m) * tx
                        continue
                    P = set(p[t, :w].tolist())
                    cov = sum(e in P for e in m)
                    wasted = (w - len(P & set(req[t, j].tolist()))) * (1 - hit_rate)
                    tt[t] += max(0.0, (cov + wasted) * tx - win) + (len(m) - cov) * tx
                out[f"prefetch-{name}-{w}@{pc / 1e9:.1f}"] = tt
    # the card computes its hits while the host computes the misses from host RAM; a layer with
    # misses takes a second halt (the host's outputs must land before the card combines them)
    hyb = np.array([sum(td + sync + max((k - miss[t, j]) * x / Bd,
                                        (sync + miss[t, j] * x / Bh) if miss[t, j] else 0.0)
                        for j in range(L)) + head / Bd for t in range(T)])
    out["hybrid"] = hyb
    res = {n: float(T / v.sum()) for n, v in out.items()}
    # PCIe share a cache that inserts every miss needs (the hybrid's inserts run beside it)
    res["hybrid_link"] = float(miss.sum() * x / hyb.sum() / min(hw["pcie"]))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--frac", default="0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.75,0.9")
    ap.add_argument("--slots", default="", help="extra cache sizes in expert slots")
    ap.add_argument("--policies", default="static,lru,lru_layer,lfu,opt")
    ap.add_argument("--json")
    ap.add_argument("--survey", help="tools/offload/survey.py --json output: bytes, card slots")
    ap.add_argument("--repo", help="the model's row in --survey (default: the trace's model)")
    ap.add_argument("--card-gib", type=float, default=4.0)
    ap.add_argument("--reserve-gb", type=float, default=0.3,
                    help="card DRAM kept for KV / state, I/O, programs")
    ap.add_argument("--dram-gbs", type=float, default=14.1, help="card DRAM while decoding")
    ap.add_argument("--pcie-gbs", default="1.3,2.6", help="host -> card (Gen1 measured, Gen2)")
    ap.add_argument("--host-gbs", type=float, default=12.0,
                    help="host fp4 expert kernel (tools/offload/hostkern.c: 12-16 on opentpu)")
    ap.add_argument("--sync-us", type=float, default=60.0, help="one halt, host step, restart")
    ap.add_argument("--call-us", type=float, default=30.0, help="per DMA call")
    ap.add_argument("--timing-policy", default="lru_layer")
    ap.add_argument("--host-frac", default="",
                    help="host RAM caches of these pool fractions over an SSD (section 8)")
    ap.add_argument("--ssd-gbs", type=float, default=0.51, help="SSD reads, O_DIRECT 4 MiB")
    a = ap.parse_args()
    tr = [load(p) for p in a.traces]
    E, L = tr[0]["E"], tr[0]["req"].shape[1]
    k = tr[0]["req"].shape[2]
    n = E * L
    extra = {int(s) for s in a.slots.split(",") if s}
    by = card = None
    if a.survey:
        rs = json.loads(Path(a.survey).read_text())
        name = a.repo or Path(tr[0]["meta"]["model"]).name
        d = next(r for r in rs if r["repo"] == name or r["repo"].split("/")[-1] == name)
        head = (d["head"] or d["embed"]) * 8.25 / 8
        by = dict(x=d["expert_bytes"], head=head,
                  d=(d["tok_bytes"] - d["tok_expert_bytes"] - head) / L)
        card = int((a.card_gib * 2**30 - d["resident_bytes"] - a.reserve_gb * 1e9)
                   // d["expert_bytes"])
        card = max(k, min(card, n))
        extra.add(card)
        print(f"  survey {d['repo']}: expert {d['expert_bytes'] / 1e6:.2f} MB, dense "
              f"{by['d'] / 1e6:.1f} MB / layer, head {head / 1e6:.0f} MB; card cache {card} slots "
              f"({card / n:.2f} of the pool) in {a.card_gib} GiB")
    hw = dict(dram=a.dram_gbs * 1e9, host=a.host_gbs * 1e9, sync=a.sync_us * 1e-6,
              call=a.call_us * 1e-6, pcie=[float(v) * 1e9 for v in a.pcie_gbs.split(",")])
    sizes = sorted({max(k, int(round(float(f) * n))) for f in a.frac.split(",") if f} | extra)
    rows = []
    print(f"{tr[0]['meta']['model']}: {L} MoE layers x {E} experts, top-{k}; "
          f"router check {min(x['ok'] for x in tr):.4f}")
    for i, x in enumerate(tr):
        acc = accuracy(x["req"], x["preds"], k)
        print(f"  {Path(x['meta']['text']).name}: {x['req'].shape[0]} tokens, nll "
              f"{x['meta']['nll_last128']:.2f}; prediction accuracy (top-{k} overlap): "
              + ", ".join(f"{p} {v:.3f}" for p, v in acc.items()))
        others = [y["req"] for j, y in enumerate(tr) if j != i] or [x["req"]]
        prof = sum(freq(r, n) for r in others)
        for C in sizes:
            warm = top_set(prof, C)
            warm_all = top_set(prof, n)
            for pol in a.policies.split(","):
                mt, ma = simulate(x["req"], C, pol, warm_all if pol == "lru_layer" else warm,
                                  prof)
                T = x["req"].shape[0]
                r = dict(text=Path(x["meta"]["text"]).name, slots=C, frac=C / n, policy=pol,
                         hit=1 - mt.sum() / (T * L * k), miss_mean=float(mt.mean()),
                         miss_p50=float(np.percentile(mt, 50)),
                         miss_p90=float(np.percentile(mt, 90)), miss_max=int(mt.max()))
                if pol in ("lru", "lru_layer"):
                    r["cover_k"] = coverage(ma, x["preds"], k)
                    r["cover_2k"] = coverage(ma, x["preds"], 2 * k)
                if by:
                    r["tok_s"] = tok_times(x["req"], ma, x["preds"], hw, by)
                rows.append(r)
    print(f"{'slots':>6} {'frac':>5} " + " ".join(f"{p:>13}" for p in a.policies.split(","))
          + "   (hit rate, misses/token mean; mean over texts)")
    for C in sizes:
        cells = []
        for pol in a.policies.split(","):
            rs = [r for r in rows if r["slots"] == C and r["policy"] == pol]
            cells.append(f"{np.mean([r['hit'] for r in rs]):.3f} "
                         f"{np.mean([r['miss_mean'] for r in rs]):6.1f}")
        lru = [r for r in rows if r["slots"] == C and r["policy"] == "lru_layer"]
        cov = ""
        if lru and lru[0].get("cover_k"):
            cov = "  lru_layer misses predicted (k / 2k): " + ", ".join(
                f"{p} {np.mean([r['cover_k'][p] for r in lru]):.2f}/"
                f"{np.mean([r['cover_2k'][p] for r in lru]):.2f}" for p in lru[0]["cover_k"])
        print(f"{C:>6} {C / n:5.2f} " + " ".join(f"{c:>13}" for c in cells) + cov)
    if by:
        names = [m for m in rows[0]["tok_s"] if m != "hybrid_link"]
        print(f"tok/s ({a.timing_policy} cache; mean over texts; pcie GB/s after @; hybrid/static: "
              f"the host computes every miss of a fixed cache; link: the PCIe share the "
              f"{a.timing_policy} inserts need beside the hybrid):")
        print(f"{'slots':>6}  " + "  ".join(names) + "  hybrid/static  link")
        for C in sizes:
            rs = [r for r in rows if r["slots"] == C and r["policy"] == a.timing_policy]
            st = [r for r in rows if r["slots"] == C and r["policy"] == "static"]
            print(f"{C:>6}  " + "  ".join(f"{np.mean([r['tok_s'][m] for r in rs]):{len(m)}.2f}"
                                          for m in names)
                  + f"  {np.mean([r['tok_s']['hybrid'] for r in st]):13.2f}"
                  + f"  {np.mean([r['tok_s']['hybrid_link'] for r in rs]):4.2f}"
                  + ("   <- card" if C == card else ""))
    ssd = []
    if by and a.host_frac:
        print(f"SSD tier: the host RAM holds a fraction of the pool (global LRU, warm from the "
              f"profile), misses read from the SSD at {a.ssd_gbs} GB/s on the token's path; the "
              f"card cache ({card} slots, {a.timing_policy}) and the hybrid as above")
        print(f"{'host frac':>9} {'host hit':>8} {'SSD reads/token':>15} {'SSD ms/token':>12} "
              f"{'tok/s':>6}")
        for hf in [float(v) for v in a.host_frac.split(",")]:
            Ch = max(card, int(round(hf * n)))
            hits, reads, tps = [], [], []
            for i, x in enumerate(tr):
                others = [y["req"] for j, y in enumerate(tr) if j != i] or [x["req"]]
                prof = sum(freq(r, n) for r in others)
                mt, _ = simulate(x["req"], Ch, "lru", top_set(prof, Ch), prof)
                T = x["req"].shape[0]
                hits.append(1 - mt.sum() / (T * L * k))
                reads.append(mt.mean())
                hy = next(r for r in rows if r["text"] == Path(x["meta"]["text"]).name
                          and r["slots"] == card and r["policy"] == a.timing_policy)["tok_s"]
                t = 1 / hy["hybrid"] + mt.mean() * by["x"] / (a.ssd_gbs * 1e9)
                tps.append(1 / t)
            ssd.append(dict(host_frac=hf, host_slots=Ch, host_hit=float(np.mean(hits)),
                            ssd_reads=float(np.mean(reads)), tok_s=float(np.mean(tps))))
            print(f"{hf:9.2f} {np.mean(hits):8.3f} {np.mean(reads):15.1f} "
                  f"{np.mean(reads) * by['x'] / (a.ssd_gbs * 1e6):12.1f} {np.mean(tps):6.2f}")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(model=tr[0]["meta"]["model"], E=E, layers=L, k=k,
                                                ssd=ssd,
                                                card_slots=card, bytes=by, hw=hw,
                                                accuracy={Path(x["meta"]["text"]).name:
                                                          accuracy(x["req"], x["preds"], k)
                                                          for x in tr},
                                                rows=rows), indent=1))


if __name__ == "__main__":
    main()
