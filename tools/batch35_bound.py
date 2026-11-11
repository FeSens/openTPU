#!/usr/bin/env python3
"""Batched decode of Qwen3.6-35B-A3B: unit bounds at B sequences a step (docs/batch35.md).

Built from pf500's ISA ledger (tools/pf500.py isa): one row of each layer kind (pf500's R = 1
layer runs, the decode's work for one sequence) split into
  - shared bytes: the layer's dense weights (projections, shared expert, router, norms), read
    once a step whatever B is;
  - routed bytes: one expert (the pool's ebytes a slice) per expert of the rows' union;
  - per-row bytes: the DeltaNet state, the KV cache (linear in the position), activations;
  - per-row unit cycles: MXU, VPU, SE (DSTEP), Q, each row's work (decode's MMs stream every
    weight once a row, so the MXU does not batch; a row's k experts stay k with the rows
    gathered per expert, expert-major);
  - COLL (x2): each collective's latency once a step, its words a row.
The head (int8, the full vocabulary, a slice's share) adds its bytes once a step and its MMs a
row. A layer's bound is its busiest unit (DRAM at pf500's 88 %); the step is the layers' bounds
(30 DeltaNet + 10 attention) plus the head's. No stalls and no overlap loss: unit bounds, not the
card. The B = 1 row is checked against the card's decode (4.32 Mcycles a token at S = 1).

--isa: the batched step's own layers measured instead (opentpu/llm/batch35.py's layer_rows on
B rows of pf500's cut at batch B, each row in its slot, uniform routing as pf500's): per layer
kind its units on the ISA ledger, the step 30 DeltaNet + 10 attention layers plus the head
(the model above), each layer at its busiest unit (DRAM at 88 %).

Usage: PYTHONPATH=. python3 tools/batch35_bound.py [--target T,...] [--B 1,2,4,8,16]
           [--pos 1024,32768] [--json out.jsonl] [--isa]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pf500 as P  # noqa: E402

FIT_POS = (1024, 32768, 131072)     # attention's per-row terms: a + b pos, fit on these
ROW_KINDS = ("kv", "dn_state", "act")
UNITS = ("MXU", "VPU", "SE", "Q")
MXU_MACS = 2048                     # a slice's MACs a cycle a decode row (pf500: the DeltaNet
                                    # projections' 33.7 M MACs in 16,448 cycles)
CARD_B1 = {"bcu1525-wg:bhw16": 4.32e6}   # the card's decode, cycles a token (emb-card, 11-04)


def measure(tg, kind: str, pos: int) -> dict:
    """pf500's R = 1 run of layer LAYER[kind] at pos, uniform routing (seed 1)."""
    img = P.image(tg, P.MODEL, pos)
    _, progs, args = P.compile_runs(img, kind, 1, pos)[0]
    ids = P.route_ids(1, img.spec.moe.E, img.spec.moe.k)
    words, live = P.route_words(img, P.LAYER[kind], ids)
    r = P.run_isa(tg, progs, args, words, live)
    r["kinds"] = P.kinds_of(r["phases"])
    r["ngather"] = sum(d.get("collectives", 0) for d in r["phases"].values())
    return r


def full_image(tg):
    """The whole model's image (every layer, the full vocabulary) as the chat builds it."""
    from opentpu.host.streamed import FORMATS
    from opentpu.llm import load_spec
    wf, hf, fm = FORMATS["qwen35"]
    return load_spec(P.MODEL).image(tg.cfg, 4096, 1, 1, wformat=wf, head_format=hf, formats=fm,
                                    lookup=True)


def layer_model(tg, kind: str, ebytes: int, k: int) -> dict:
    """{shared, row_bytes(pos), units(pos), coll_lat, coll_row} of one layer kind, a slice."""
    pts = FIT_POS if kind == "attn" else FIT_POS[:1]
    ms = {p: measure(tg, kind, p) for p in pts}
    m0 = ms[pts[0]]
    shared = m0["kinds"]["weights"] - k * ebytes
    lat = m0["ngather"] * tg.gather_lat()
    out = {"shared": shared, "coll_lat": lat, "coll_row": m0["total"]["COLL"] - lat,
           "pts": {p: {"row": sum(m["kinds"][x] for x in ROW_KINDS),
                       **{u: m["total"][u] for u in UNITS}} for p, m in ms.items()}}
    if len(pts) > 1:      # least squares a + b pos on each term
        A = np.array([[1.0, p] for p in pts])
        for key in ("row", *UNITS):
            y = np.array([out["pts"][p][key] for p in pts], float)
            out[key] = tuple(np.linalg.lstsq(A, y, rcond=None)[0])
    else:
        for key in ("row", *UNITS):
            out[key] = (float(out["pts"][pts[0]][key]), 0.0)
    return out


def cut_image(tg, pos: int, B: int):
    """pf500's cut (its first CUT layers, CUT_VOCAB) with B sequence slots (Image(batch=B)), KV
    capacity the next 4096 past pos."""
    from opentpu.host.streamed import FORMATS
    from opentpu.llm import load_spec
    cap = 4096 * (pos // 4096 + 1)
    spec = load_spec(P.MODEL)
    s = dataclasses.replace(spec, kinds=spec.kinds[:P.CUT], vocab=P.CUT_VOCAB)
    wf, hf, fm = FORMATS["qwen35"]
    return s.image(tg.cfg, cap, B, B, wformat=wf, head_format=hf, formats=fm, lookup=True)


def _layer_prog():
    from opentpu import language as ol
    from opentpu.compiler import Tensor
    from opentpu.llm import batch35 as B35

    @ol.jit
    def batch_layer(m, rows, li: int, xa: int, block: int):
        """Layer li over the rows (slot, position) (batch35.layer_rows), x [R, H] from and back
        to xa (pf500's xbuf: route_words' one-hot rows)."""
        R, H = len(rows), m.spec.hidden
        xt = Tensor(xa, (R, H), (H, 1))
        x = ol.empty([R, H], dense=True)
        ol.load(xt, out=x)
        c = ol.empty([R, m.cosr.shape[1]], dense=True)
        s_ = ol.empty([R, m.sinr.shape[1]], dense=True)
        for r, (_, p) in enumerate(rows):
            ol.load(m.cos_t[p:p + 1, :], out=c[r:r + 1, :])
            ol.load(m.sin_t[p:p + 1, :], out=s_[r:r + 1, :])
        B35.layer_rows(x, m.layer(li), m, rows, c, s_, block)
        ol.store(xt, x)
    return batch_layer


def batch_case(tg, kind: str, B: int, pos: int, seed: int = 1):
    """(image, programs, words, live, ids): layer LAYER[kind] of pf500's cut as the batched step
    runs it, B rows, row r in slot r at pos, uniform routing (pf500's ids, seed)."""
    from opentpu.llm.qwen3 import ATTN_BLOCK
    img = cut_image(tg, pos, B)
    li, rows = P.LAYER[kind], [(s, pos) for s in range(B)]
    prog = _layer_prog()
    progs = [prog.trace(tg.cfg, s, {"m": img.descriptors(s), "rows": rows, "li": li,
                                    "xa": img.io["xbuf"], "block": ATTN_BLOCK}).finish()
             for s in range(tg.cfg.S)]
    ids = P.route_ids(B, img.spec.moe.E, img.spec.moe.k, seed)
    words, live = P.route_words(img, li, ids)
    return img, progs, words, live, ids


def measure_batch(tg, kind: str, B: int, pos: int, seed: int = 1, sb=None,
                  dump: str = "") -> dict:
    """batch_case's run on the ISA; with sb (pf500.SimBoard) on the board model too: its cycles,
    busy unions, timeline (timeline) and pf500.run_diag's waits beside the ISA's ledger; dump: a
    directory, the trace's records to DUMP/KIND-B-POS.jsonl (pf500.dump_recs)."""
    img, progs, words, live, ids = batch_case(tg, kind, B, pos, seed)
    keep: list = []
    r = P.run_isa(tg, progs, [], words, live, keep)
    r["kinds"] = P.kinds_of(r["phases"])
    r["experts"] = int(np.unique(ids).size)
    if sb is not None:
        keep[0].R_rows = B
        prof: list = []
        r["sim"] = P.run_sim(tg, sb, img, progs, [], words, live, keep[0], prof)
        r["sim"]["timeline"] = timeline(prof[0], progs)
        r["sim"]["diag"] = P.run_diag(prof[0], progs)
        if dump:
            P.dump_recs(f"{dump}/{kind}-{B}-{pos}.jsonl", prof[0], progs)
    return r


# the trace's instruction classes (opentpu.isa ops), the step's work: the weight / state / KV
# loads, the stream engine (DSTEP, STREAM), the MXU, the VPU, the quantizer, the collectives
CLASSES = {"LD": ("LD",), "ST": ("ST",), "SE": ("DSTEP", "STREAM"), "MM": ("MM",),
           "VOP": ("VOP",), "Q": ("QACT", "QST"), "COLL": ("GATHER", "BAR")}
COMPUTE = ("SE", "MM", "VOP", "Q")


def timeline(p, progs, s: int = 0, bins: int = 40) -> dict:
    """Slice s's cycles by what was in flight: each class's busy union (CLASSES), then the
    cycles with a load in flight and no compute (SE, MXU, VPU, Q), compute and no load, both,
    neither; each pair of classes' cycles both busy (the MXU streams its weights itself: MM&SE,
    MM&VOP are the rows' work beside the weight stream), any class busy; per class its busy share
    in each of `bins` equal spans of the run (where the rows' compute sits against the weight
    stream)."""
    from opentpu import isa as I
    ops = {getattr(I, o): c for c, os_ in CLASSES.items() for o in os_}
    recs = [r for r in p.slice_recs(s) if r.start >= 0 and r.end > r.start]
    n = max([r.end for r in recs] + [1])
    on = {c: np.zeros(n, bool) for c in CLASSES}
    for r in recs:
        c = ops.get(r.op)
        if c is not None:
            on[c][r.start:r.end] = True
    comp = np.logical_or.reduce([on[c] for c in COMPUTE])
    ld = on["LD"]
    edges = np.linspace(0, n, bins + 1).astype(int)
    return {"cycles": n, "busy": {c: int(v.sum()) for c, v in on.items()},
            "ld_only": int((ld & ~comp).sum()), "compute_only": int((comp & ~ld).sum()),
            "both": int((ld & comp).sum()), "neither": int((~ld & ~comp).sum()),
            "compute_only_by": {c: int((on[c] & ~ld).sum()) for c in COMPUTE},
            "pairs": {f"{a}&{b}": int((on[a] & on[b]).sum()) for i, a in enumerate(CLASSES)
                      for b in list(CLASSES)[i + 1:]},
            "any": int(np.logical_or.reduce(list(on.values())).sum()),
            "bins": {c: [round(float(v[a:b].mean()), 2) if b > a else 0.0
                         for a, b in zip(edges[:-1], edges[1:])] for c, v in on.items()}}


def isa_step(tg, head: dict, B: int, pos: int, n: dict, sb=None, dump: str = "") -> dict:
    """The batched step from its layers measured on the ISA (measure_batch) plus the head
    (the model's): each layer at its busiest unit, DRAM at 88 %. With sb (pf500.SimBoard) the
    layers on the board model too: the step at their cycles (sim_Mcyc_step, the head the
    model's bound), each layer's timeline."""
    tot = {u: 0.0 for u in (*UNITS, "COLL", "DRAM88")}
    bound, per, sim = 0.0, {}, 0.0
    for kind, cnt in n.items():
        m = measure_batch(tg, kind, B, pos, sb=sb, dump=dump)
        c = {u: m["total"][u] for u in (*UNITS, "COLL")}
        c["DRAM88"] = m["total"]["dram88"]
        for u, v in c.items():
            tot[u] += cnt * v
        bound += cnt * max(c.values())
        per[kind] = {"bound": round(max(c.values())), "binding": max(c, key=c.get),
                     "experts": m["experts"], "program": m["program"],
                     **{u: round(v) for u, v in c.items()}}
        if sb is not None:
            per[kind]["sim"] = m["sim"]
            sim += cnt * m["sim"]["cycles"]
    h = {"MXU": B * head["mxu"], "VPU": B * head["vpu"],
         "DRAM88": head["bytes"] / tg.dram_bpc / P.DRAM_EFF}
    for u, v in h.items():
        tot[u] += v
    bound += max(h.values())
    out = {"B": B, "pos": pos, "isa": True, "Mcyc_step": round(bound / 1e6, 3),
           "tok_s_user": round(P.MHZ * 1e6 / bound, 1),
           "tok_s_total": round(B * P.MHZ * 1e6 / bound, 1),
           "busy": {u: round(100 * v / bound) for u, v in tot.items()},
           "binding": max(tot, key=tot.get), "layers": per}
    if sb is not None:
        sim += max(h.values())
        out |= {"sim_Mcyc_step": round(sim / 1e6, 3),
                "sim_tok_s_total": round(B * P.MHZ * 1e6 / sim, 1),
                "sim_ok": all(per[k]["sim"].get("ok") for k in per)}
    return out


def union(B: int, E: int, k: int) -> float:
    """Expected distinct experts of B rows' top-k picks, uniform routing (real routing is
    skewed: fewer)."""
    return E * (1.0 - (1.0 - k / E) ** B)


def step(tg, lm: dict, head: dict, B: int, pos: int, E: int, k: int, ebytes: int,
         n: dict) -> dict:
    """One step of B rows at position pos: per-unit cycles a step (a slice), the bound."""
    bpc, U = tg.dram_bpc, union(B, E, k)
    tot = {u: 0.0 for u in (*UNITS, "COLL", "DRAM88")}
    bound, nbytes = 0.0, 0.0
    for kind, cnt in n.items():
        m = lm[kind]
        f = lambda key: m[key][0] + m[key][1] * pos     # noqa: E731
        c = {u: B * f(u) for u in UNITS}
        c["COLL"] = m["coll_lat"] + B * m["coll_row"]
        c["DRAM88"] = (m["shared"] + U * ebytes + B * f("row")) / bpc / P.DRAM_EFF
        nbytes += cnt * (m["shared"] + U * ebytes + B * f("row"))
        for u, v in c.items():
            tot[u] += cnt * v
        bound += cnt * max(c.values())
    h = {"MXU": B * head["mxu"], "VPU": B * head["vpu"], "DRAM88": head["bytes"] / bpc / P.DRAM_EFF}
    for u, v in h.items():
        tot[u] += v
    bound += max(h.values())
    nbytes += head["bytes"]
    cyc = bound
    return {"B": B, "pos": pos, "union": round(U, 1), "Mcyc_step": round(cyc / 1e6, 3),
            "tok_s_user": round(P.MHZ * 1e6 / cyc, 1),
            "tok_s_total": round(B * P.MHZ * 1e6 / cyc, 1),
            "busy": {u: round(100 * v / cyc) for u, v in tot.items()},
            "bytes_step": int(nbytes), "bpc": bpc,
            "binding": max(tot, key=tot.get)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", default="bcu1525-wg:bhw16,bcu1525-x2:st16")
    ap.add_argument("--B", default="1,2,4,8,16")
    ap.add_argument("--pos", default="1024,8192,32768")
    ap.add_argument("--json", default="")
    ap.add_argument("--isa", action="store_true",
                    help="the batched step's layers measured on the ISA (measure_batch)")
    ap.add_argument("--sim", action="store_true",
                    help="--isa, and each layer on the target's board model (a "
                         "Verilator build, then a run a layer)")
    ap.add_argument("--kinds", default="dn,attn", help="--isa: the layer kinds measured")
    ap.add_argument("--dump", default="", help="--sim: each run's trace records to DIR")
    a = ap.parse_args(argv)
    a.isa = a.isa or a.sim
    rows = []
    for tname in a.target.split(","):
        tg = P.target(tname)
        im = full_image(tg)
        spec, cfg = im.spec, tg.cfg
        E, k, H = spec.moe.E, spec.moe.k, spec.hidden
        ebytes = im.pool.ebytes
        n = {"dn": spec.kinds.count(spec.kinds[0]), "attn": len(spec.kinds) - spec.kinds.count(
            spec.kinds[0])}
        from opentpu import quant as Q
        hb = sum(Q.grp_bytes(im.v_loc, H, im.head_format, cfg.D, cfg.WG))
        # the head's MMs at the MXU's decode rate, its sampling one ARGMAX pass over the slice's
        # logits (generate.Greedy: LANES words a cycle)
        head = {"bytes": hb, "mxu": im.v_loc * H / MXU_MACS, "vpu": im.v_loc / cfg.LANES}
        if a.isa:
            sb = None
            nk = {x: n[x] for x in a.kinds.split(",")}
            if a.sim:                           # (one build: DRAM past the largest image)
                Bs = [int(x) for x in a.B.split(",")]
                img = cut_image(tg, max(int(x) for x in a.pos.split(",")), max(Bs))
                sb = P.SimBoard(tg, P.prog_at(img), P.K7, "", 65536)
            for pos in [int(x) for x in a.pos.split(",")]:
                for B in [int(x) for x in a.B.split(",")]:
                    r = {"target": tname, **isa_step(tg, head, B, pos, nk, sb, a.dump)}
                    print(json.dumps(r), flush=True)
                    rows.append(r)
            continue
        lm = {kind: layer_model(tg, kind, ebytes, k) for kind in ("dn", "attn")}
        info = {"target": tname, "S": cfg.S, "ebytes": ebytes, "head_bytes": hb, "n": n,
                "shared": {x: lm[x]["shared"] for x in lm},
                "row1024": {x: lm[x]["row"][0] + 1024 * lm[x]["row"][1] for x in lm},
                "attn_row_per_pos": lm["attn"]["row"][1], "attn_vpu_per_pos": lm["attn"]["VPU"][1],
                "coll": {x: (lm[x]["coll_lat"], lm[x]["coll_row"]) for x in lm}}
        print(json.dumps(info), flush=True)
        rows.append(info)
        for pos in [int(x) for x in a.pos.split(",")]:
            for B in [int(x) for x in a.B.split(",")]:
                r = {"target": tname, **step(tg, lm, head, B, pos, E, k, ebytes, n)}
                if B == 1 and tname in CARD_B1 and pos == 1024:
                    r["card_ratio"] = round(CARD_B1[tname] / (r["Mcyc_step"] * 1e6), 3)
                print(json.dumps(r), flush=True)
                rows.append(r)
    if a.json:
        Path(a.json).write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
