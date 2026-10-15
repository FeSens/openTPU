"""Batched decode steps (opentpu/llm/batch.py) against the card's decode loop: device cycles and
wall time per run, per row count.

    python tools/batch_bench.py --model qwen3 --backend board [--slots 8] [--context 64]

Each slot gets a prompt of `context` tokens; then, for R = 1, 2, 4, 8 rows (greedy and sampled),
`reps` steps; and slot 0's decode loop (generate_card) for the same number of tokens. Prints per
case the device Mcycles per run and per token, the run's wall time, and the host's share (wall
minus device time). --trace: then one decode step and one step of each traced row count with
the card's trace buffer (its first records): per instruction class the busy cycles, and Lens
profiles (--out's directory, <out>.<case>.otpuprof.json; otpu-lens open).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--backend", default="board", choices=["board", "isa"])
    ap.add_argument("--dev", help="the card's device prefix (default: /dev/otpu0 if present, "
                    "else /dev/xdma0)")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--context", type=int, default=64)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--trace", default="", help="row counts of traced steps, e.g. 1,4")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    from opentpu.host import chat as CH
    from opentpu.llm import batch as BA
    from opentpu.llm import generate as G
    from opentpu.llm import load_spec, model_dir
    from opentpu.llm.qwen3 import Engine, load_weights
    path = model_dir(a.model)
    spec = load_spec(path)
    backend, cfg = CH.make_backend(a.backend, spec, a.cap, a.dev, path.name, lookup=True)
    if cfg is None:
        from dataclasses import replace

        from opentpu.llm.qwen3 import PREFILL_ROWS, device_config
        c = device_config(spec, a.cap, batch=a.slots, rows=PREFILL_ROWS, lookup=True, S=1)
        cfg = replace(c, DRAM_BYTES=2 * c.DRAM_BYTES)
    eng = Engine(spec, load_weights(path), cap=a.cap, cfg=cfg, backend=backend, resident=True,
                 prompt_runs=True, batch=a.slots, batch_area=True)
    khz = (getattr(eng.backend, "info", None) or {}).get("core_khz") or 0
    r = np.random.default_rng(0)
    nxt = []
    st = BA.Steps(eng)
    res = []
    pre = {"prompt_runs": [0, 0.0, 0], "rows": [0, 0.0, 0], "feed": [0, 0.0, 0]}
    for s in range(a.slots):                    # slot 0: the prompt runs; 1: the rows kernel
        p = [int(x) for x in r.integers(1000, 20000, a.context)]  # (a program per run); the
        k = len(eng.stats)                                         # others: feed runs + a step
        t0 = time.perf_counter()
        if s < 2:
            nxt.append(int(np.argmax(eng.prefill(p, seq=s))))
        else:
            st.set_sampling(s, None)
            st.feed(s, p[:-1])
            nxt.append(st.run([(s, eng.poss[s])], [p[-1]])[0])
        x = pre["prompt_runs" if s == 0 else "rows" if s == 1 else "feed"]
        x[0] += len(p)
        x[1] += time.perf_counter() - t0
        x[2] += sum((y or {}).get("cycles") or 0 for y in eng.stats[k:])
    for name, (n, w, cyc) in pre.items():
        row = {"case": f"prefill {name}", "tokens": n, "tok_s": round(n / w, 1) if w else None,
               "device_s": round(cyc / (khz * 1e3), 3) if khz else None, "wall_s": round(w, 3)}
        res.append(row)
        print(json.dumps(row), flush=True)

    def case(name, fn):
        """fn() -> (runs, tokens)."""
        k = len(eng.stats)
        t0 = time.perf_counter()
        runs, toks = fn()
        wall = time.perf_counter() - t0
        cyc = [x.get("cycles") or 0 for x in eng.stats[k:]]
        dev = sum(cyc) / (khz * 1e3) if khz else 0.0
        row = {"case": name, "runs": runs, "tokens": toks,
               "Mcyc_run": round(np.mean(cyc) / 1e6, 3) if cyc else None,
               "Mcyc_tok": round(sum(cyc) / 1e6 / max(1, toks), 3),
               "wall_ms_run": round(1e3 * wall / max(1, runs), 2),
               "device_ms_run": round(1e3 * dev / max(1, runs), 2) if khz else None,
               "host_ms_run": round(1e3 * (wall - dev) / max(1, runs), 2) if khz else None,
               "tok_s": round(toks / wall, 2)}
        res.append(row)
        print(json.dumps(row), flush=True)

    # slot 0's decode loop
    def loop():
        got = eng.generate_card(nxt[0], a.reps * 4)
        nxt[0] = got[-1]
        return 1, len(got)
    case("generate_card slot 0", loop)
    for kind in ("greedy", "sampled"):
        samp = None if kind == "greedy" else G.Sampling(0.7, 20, 0.8, 1.0)
        for s in range(a.slots):
            st.set_sampling(s, samp, [nxt[s]])
        for R in (1, 2, 4, 8):
            if R > a.slots:
                continue

            def steps(R=R, samp=samp):
                n0 = st.runs
                for _ in range(a.reps):
                    rows = [(s, eng.poss[s]) for s in range(R)]
                    u = [0.5] * R if samp is not None else None
                    out = st.run(rows, [nxt[s] for s in range(R)], u)
                    for s in range(R):
                        nxt[s] = out[s]
                return st.runs - n0, a.reps * R
            p0 = dict(st.timing)
            case(f"step {kind} R={R}", steps)
            groups = len(st.plan([(s, eng.poss[s]) for s in range(R)]))
            res[-1]["groups"] = groups
            res[-1]["programs"] = st.timing["programs"] - p0["programs"]
            res[-1]["compile_wait_ms"] = round(1e3 * (st.timing["compile_wait_s"] -
                                                      p0["compile_wait_s"]), 1)
            nr = max(1, res[-1]["runs"])
            res[-1]["phase_ms_run"] = {k[:-2]: round(1e3 * (st.timing[k] - p0[k]) / nr, 2)
                                       for k in ("write_s", "start_s", "wait_s", "read_s")}
            print(f"   (per run: {res[-1]['phase_ms_run']})", flush=True)
            print(f"   (groups per step: {groups}, programs {res[-1]['programs']}, compile "
                  f"wait {res[-1]['compile_wait_ms']} ms)", flush=True)
    print("timing:", st.timing, flush=True)
    prof = {}
    if a.trace:
        try:
            prof = traced(eng, st, nxt, [int(x) for x in a.trace.split(",") if x], khz, a.out)
        except Exception:
            import traceback
            traceback.print_exc()
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"cases": res, "timing": st.timing, "context": a.context,
                       "traced": prof}, f, indent=1)


def traced(eng, st, nxt, counts, khz, out) -> dict:
    """One decode step (slot 0) and one greedy step per row count with the trace buffer on:
    per instruction class busy cycles (of the records the buffer kept)."""
    import collections

    from opentpu import profile as PR
    from opentpu.host import hwlens as HL
    be = eng.backend
    if getattr(be, "info", {}).get("regmap", 0) < 2 or not be.info["caps"].get("trace"):
        print("trace: no trace buffer on this backend", flush=True)
        return {}
    res = {}

    def one(name, fn, instrs=None):
        be.trace = {"keep": "first"}
        try:
            fn()
        finally:
            be.trace = None
        programs, stats = be.last
        if instrs is not None:                  # (the run loaded assembled words)
            programs = instrs()
        p = PR.parse(HL.records_to_trace(stats["trace"]["records"], programs), eng.cfg,
                     programs, name)
        last = p.cycles
        for r in p.recs:                    # the buffer filled up: close what is open
            if r.start >= 0 and r.end < 0:
                r.end = last
        agg = collections.defaultdict(lambda: [0, 0])
        for r in p.slice_recs(0):
            if r.start >= 0:
                agg[r.name][0] += 1
                agg[r.name][1] += r.end - r.start
        top = sorted(agg.items(), key=lambda kv: -kv[1][1])[:14]
        busy = p.unit_busy(0)
        row = {"cycles": int(stats["cycles"]), "traced_to": int(last),
               "records": int(len(stats["trace"]["records"])),
               "unit_busy_pct": {k: round(100 * v / max(1, last), 1) for k, v in busy.items()},
               "classes": {k: {"n": n, "busy": b} for k, (n, b) in top}}
        res[name] = row
        print(f"trace {name}: {row['cycles']} cycles, traced to {last}; units "
              f"{row['unit_busy_pct']}", flush=True)
        for k, (n, b) in top:
            print(f"   {k:40s} n={n:6d} busy={b:9d} ({100 * b / max(1, last):.1f}%)", flush=True)
        if out:
            import json as J
            d = HL.hw_profile(name, eng.cfg, programs, stats, khz or None)
            with open(f"{out}.{name.replace(' ', '_')}.otpuprof.json", "w") as f:
                J.dump(d, f)

    def step0():
        lg = eng.step(nxt[0])
        nxt[0] = int(np.argmax(lg))
    one("decode step", step0)
    from opentpu.llm import batch as BA
    for R in counts:
        def stepR(R=R):
            st.run([(s, eng.poss[s]) for s in range(R)], [nxt[s] for s in range(R)])

        def instrs(R=R):
            slots = tuple(range(R))
            kinds = tuple(st.kinds.get(s) for s in slots)
            base, n = next((k[2], v[0]) for k, v in reversed(list(st._live.items()))
                           if k[:2] == (slots, kinds))  # (the program the traced run took)
            return BA.compile_step(eng.image, st.a, list(zip(slots, base)), list(kinds),
                                   eng.block, n)
        for s in range(R):
            st.set_sampling(s, None)
        stepR()                                         # (compiled, then traced)
        one(f"step greedy R={R}", stepR, instrs)
    return res


if __name__ == "__main__":
    sys.exit(main())
