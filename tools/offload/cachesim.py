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
on-card part and a reserve) joins the sizes, and each replay becomes tokens per second under
path (a) of docs/offload.md section 4, the card computing everything and the host only moving
experts (`linksim`), with and without prefetch from the predictions. `--host-frac`: the SSD
tier, a host RAM holding that fraction of the pool (LRU) over the SSD.
"""
from __future__ import annotations

import argparse
import heapq
import json
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    meta.setdefault("text", str(path))          # a card's routes (moe_card.py --trace): no
    moe = meta["moe_layers"]                    # text, no predictions
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


def resident_time(L, k, by, hw):
    """Seconds per token if every expert fit on the card: the bound."""
    return (L * (by["d_pre"] + by["d_post"] + k * by["x"]) + by["head"]) / hw["dram"]


def linksim(req, preds, caps, warm, hw, by, pred=None, width=None, host=None, overlap="each"):
    """Path (a), docs/offload.md section 4: the card computes everything and the experts it
    lacks stream over PCIe into per-layer LRU slots, one token and layer at a time.

    Per MoE layer the card runs its mixer and router (`d_pre` bytes), posts the k expert ids
    (the host sees them `req` seconds later and DMAs the missing ones one after another,
    `x / pcie + call` each), runs what needs no expert (`d_post`: Gemma 4's dense MLP, a shared
    expert), then the k experts: `overlap` "all" waits for the last missing expert (plus
    `done`: the flag the host sets and WAITW sees) and computes the k after it; "each" computes
    the experts it has first and each missing one as soon as it lands (plus `done`), while the
    rest stream, each transfer's DRAM writes charged to the card's time. A prediction (`pred`: "pre", the layer's own router
    before its mixer; "prev_r", the next layer's router right after this one's; its `width`
    best) posts prefetches that take a slot at once (an LRU victim) and use the link only when
    no demand transfer needs it: a demand preempts them at the next `chunk` bytes (each chunk a
    DMA call), and each prefetched byte, landing while the card computes, costs DRAM time.
    `host`: (slots, profile order) of a host-RAM LRU over the pool, whose misses are read from
    the SSD (`ssd` bytes/s, its own queue) before they cross the link. Returns tok/s and
    per-token counts: demand transfers, completed prefetches, prefetches wasted (evicted or
    overwritten unused), SSD reads, and the link's busy share."""
    T, L, k = req.shape
    x, Bd, Bp = by["x"], hw["dram"], hw["pcie_one"]
    tx = x / Bp + hw["call"]                           # a demand transfer: one DMA call
    chunk = hw.get("chunk", 512 << 10)
    tx_pre = x / Bp + -(-int(x) // chunk) * hw["call"]  # a prefetch, in chunks
    t_chunk = chunk / Bp + hw["call"]
    t_pre, t_post, t_exp = by["d_pre"] / Bd, by["d_post"] / Bd, x / Bd
    caches = [OrderedDict() for _ in range(L)]        # expert -> arrival time, or None: queued
    for j, ws in enumerate(warm):
        for e in ws[:caps[j]]:
            caches[j][e] = -1.0
    hc = None
    if host is not None:
        hs, order = host
        hc = OrderedDict((e, None) for e in order[:hs])
    unused = set()                                     # prefetched, not used yet
    pending = deque()                                  # [ready time, layer, e, time left]
    st = dict(free=0.0, ssd=0.0, busy=0.0)
    cnt = dict(demand=0, prefetched=0, wasted=0, ssd=0)

    def ssd_ready(e, when):
        """When expert e is in host RAM (after an SSD read if the host's cache misses)."""
        if hc is None:
            return when
        if e in hc:
            hc.move_to_end(e)
            return when
        cnt["ssd"] += 1
        st["ssd"] = max(st["ssd"], when) + x / hw["ssd"]
        if len(hc) >= hs:
            hc.popitem(last=False)
        hc[e] = None
        return st["ssd"]

    def evict_into(j, e, v, protect):
        c = caches[j]
        if len(c) >= max(caps[j], k):
            victim = next(q for q in c if q not in protect)
            if c[victim] is None:                      # a queued prefetch: drop it
                for it in pending:
                    if it[1] == j and it[2] == victim:
                        it[3] = -1.0
            if (j, victim) in unused:
                unused.discard((j, victim))
                cnt["wasted"] += 1
            del c[victim]
        c[e] = v

    def advance(now):
        """Prefetches use the link's idle time up to `now` (in order, preemptible)."""
        t = st["free"]
        while pending and t < now:
            it = pending[0]
            if it[3] < 0 or caches[it[1]].get(it[2], 0.0) is not None:
                pending.popleft()                      # dropped, or taken as a demand
                continue
            s0 = max(t, it[0])
            if s0 >= now:
                break
            run = min(it[3], now - s0)
            it[3] -= run
            st["busy"] += run
            t = s0 + run
            if it[3] <= 1e-12:
                pending.popleft()
                caches[it[1]][it[2]] = t
                unused.add((it[1], it[2]))
                cnt["prefetched"] += 1
        st["free"] = max(st["free"], t)

    def post(j, ids, when):
        for e in ids:
            e = int(e)
            if e not in caches[j]:
                evict_into(j, e, None, set())
                pending.append([ssd_ready(e, when), j, e, tx_pre])

    t = 0.0
    debt = 0
    for tok in range(T):
        for j in range(L):
            if pred == "pre" and preds[pred][j] is not None:
                post(j, preds[pred][j][tok, :width], t + hw["req"])
            tr = t + t_pre + debt * t_exp               # the router is done
            want = [int(e) for e in req[tok, j]]
            td = tr + hw["req"]                         # the host sees the request
            pre0 = cnt["prefetched"]
            advance(td)
            need = tr
            dem, late = [], []
            for e in want:
                c = caches[j]
                if e in c and c[e] is not None:
                    c.move_to_end(e)
                    need = max(need, c[e])
                    if c[e] > tr:
                        late.append(c[e])
                    unused.discard((j, e))
                else:
                    left = tx
                    if e in c:                          # queued or part-sent: finish it now
                        for it in pending:
                            if it[1] == j and it[2] == e and it[3] >= 0:
                                left, it[3] = min(tx, it[3] + hw["call"]), -1.0
                        c.move_to_end(e)
                    else:
                        evict_into(j, e, None, set(want))
                    dem.append((e, left))
            if dem:
                s0 = max(st["free"], td)
                if pending and pending[0][3] >= 0 and pending[0][0] < td:
                    s0 += min(t_chunk, pending[0][3])  # the prefetch chunk under way ends
                for e, left in dem:
                    s0 = max(s0, ssd_ready(e, td))
                    s0 += left
                    st["busy"] += left
                    caches[j][e] = s0
                    late.append(s0)
                    cnt["demand"] += 1
                need = max(need, s0)
                st["free"] = s0
            if pred == "prev_r" and j + 1 < L and preds[pred][j + 1] is not None:
                post(j + 1, preds[pred][j + 1][tok, :width], td)
            if overlap == "each":
                t = tr + t_post + (k - len(late) + len(dem)) * t_exp
                for a_ in sorted(late):
                    t = max(t, a_ + hw["done"]) + t_exp
            else:
                ready = tr + t_post
                if need > tr:
                    ready = max(ready, need + hw["done"])
                t = ready + k * t_exp
            advance(t)
            debt = cnt["prefetched"] - pre0             # their DRAM writes, on the next layer
        t += by["head"] / Bd
    return dict(tok_s=T / t, link=st["busy"] / t, **{q: v / T for q, v in cnt.items()})


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
    ap.add_argument("--call-us", type=float, default=30.0, help="per DMA call")
    ap.add_argument("--req-us", type=float, default=30.0,
                    help="the card posts expert ids -> the host starts their DMA")
    ap.add_argument("--done-us", type=float, default=15.0,
                    help="a DMA ends -> the card's WAITW sees the host's flag")
    ap.add_argument("--post-mb", type=float, default=0.0,
                    help="a MoE layer's work after its router that needs no expert (MB)")
    ap.add_argument("--stream-slots", default="",
                    help="cache sizes to run the stream model at (default: the card's)")
    ap.add_argument("--host-frac", default="",
                    help="host RAM caches of these pool fractions over an SSD (section 8)")
    ap.add_argument("--ssd-gbs", type=float, default=0.51, help="SSD reads, O_DIRECT 4 MiB")
    ap.add_argument("--overlap", default="each", choices=("each", "all"),
                    help="compute each missing expert when it lands, or all after the last")
    ap.add_argument("--expert-bits", type=float, default=4.25,
                    help="bits per expert weight (4.25: fp4 blocks; 8.25: int8)")
    ap.add_argument("--head-bits", type=float, default=8.25,
                    help="bits per LM head weight (8.25: int8; 4.25: fp4)")
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
        head8 = (d["head"] or d["embed"]) * 8.25 / 8         # the survey's: int8
        head = head8 * a.head_bits / 8.25
        dl = (d["tok_bytes"] - d["tok_expert_bytes"] - head8) / L
        xb = d["expert_bytes"] * a.expert_bits / 4.25
        by = dict(x=xb, head=head, d_pre=dl - a.post_mb * 1e6,
                  d_post=a.post_mb * 1e6)
        card = int((a.card_gib * 2**30 - d["resident_bytes"] + head8 - head
                    - a.reserve_gb * 1e9) // xb)
        card = max(k, min(card, n))
        extra.add(card)
        print(f"  survey {d['repo']}: expert {xb / 1e6:.2f} MB, dense "
              f"{dl / 1e6:.1f} MB / layer ({a.post_mb} after the router), head "
              f"{head / 1e6:.0f} MB; card cache {card} slots "
              f"({card / n:.2f} of the pool) in {a.card_gib} GiB")
    hw = dict(dram=a.dram_gbs * 1e9, call=a.call_us * 1e-6, req=a.req_us * 1e-6,
              done=a.done_us * 1e-6, ssd=a.ssd_gbs * 1e9,
              pcie=[float(v) * 1e9 for v in a.pcie_gbs.split(",")])
    sizes = sorted({max(k, int(round(float(f) * n))) for f in a.frac.split(",") if f} | extra)
    rows = []
    print(f"{tr[0]['meta']['model']}: {L} MoE layers x {E} experts, top-{k}; "
          f"router check {min(x['ok'] for x in tr):.4f}")
    for i, x in enumerate(tr):
        acc = accuracy(x["req"], x["preds"], k)
        print(f"  {Path(x['meta']['text']).name}: {x['req'].shape[0]} tokens, nll "
              f"{x['meta'].get('nll_last128', float('nan')):.2f}; prediction accuracy (top-{k} "
              "overlap): "
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
    stream, ssd = [], []
    if by:
        cfgs = [("stream", None, None)] + [(f"{p}-{w}", p, w) for p in ("pre", "prev_r")
                                           for w in (k, 2 * k) if p in tr[0]["preds"]]
        ssizes = [int(v) for v in a.stream_slots.split(",") if v] or [card]
        hosts = [None] + [float(v) for v in a.host_frac.split(",") if v]
        print(f"path (a), the card computes everything, overlap {a.overlap} (tok/s, mean over "
              f"texts; resident bound "
              f"{1 / resident_time(L, k, by, hw):.2f}); per config: tok/s (demand transfers / "
              f"prefetches / wasted prefetches per token, the link's busy share)")
        for C in ssizes:
            caps = [C // L + (j < C % L) for j in range(L)]
            for hf in hosts:
                for pc in hw["pcie"]:
                    line = []
                    for name, pr, w in cfgs:
                        res = []
                        for i, x in enumerate(tr):
                            others = [y["req"] for j, y in enumerate(tr) if j != i] or [x["req"]]
                            prof = sum(freq(r, n) for r in others)
                            warm = [[int(j * E + e) for e in np.argsort(-prof[j * E:(j + 1) * E],
                                                                        kind="stable")]
                                    for j in range(L)]
                            host = None
                            if hf is not None:
                                hs = max(C, int(round(hf * n)))
                                host = (hs, top_set(prof, n))
                            res.append(linksim(x["req"], x["preds"], caps, warm,
                                               dict(hw, pcie_one=pc), by, pr, w, host,
                                               a.overlap))
                        m = {q: float(np.mean([r_[q] for r_ in res])) for q in res[0]}
                        row = dict(slots=C, host_frac=hf, pcie=pc, config=name, **m)
                        (ssd if hf is not None else stream).append(row)
                        line.append(f"{name} {m['tok_s']:.2f} ({m['demand']:.1f}/"
                                    f"{m['prefetched']:.1f}/{m['wasted']:.1f}, "
                                    f"{m['link']:.2f})")
                    tag = f"host RAM {hf:.0%} of the pool, " if hf is not None else ""
                    print(f"  {C} slots, {tag}PCIe {pc / 1e9:.1f} GB/s: " + "; ".join(line))
    if a.json:
        Path(a.json).write_text(json.dumps(dict(model=tr[0]["meta"]["model"], E=E, layers=L, k=k,
                                                stream=stream, ssd=ssd, overlap=a.overlap,
                                                card_slots=card, bytes=by, hw=hw,
                                                accuracy={Path(x["meta"]["text"]).name:
                                                          accuracy(x["req"], x["preds"], k)
                                                          for x in tr},
                                                rows=rows), indent=1))


if __name__ == "__main__":
    main()
