"""Prefill vs decode cycles per token on the RTL at the board configuration (simulated).

    python3 tools/perf_prefill.py [--model qwen3|lfm2|qwen35|DIR] [--prompts 16,64,256]
                                  [--bw 100] [--lat 30] [--cache FILE] [--validate]

A prompt of P tokens from an empty cache, fed as the Engine does it: token by token with the
decode kernel (P steps; before chunked prefill) or in chunked prefill runs (Engine._chunk
picks each run's rows: PREFILL_ROWS, fewer where TMEM or IMEM do not hold them; only the last
run computes logits, for its last row). Reported: device cycles per prompt token, both ways,
the speedup, and TTFT at an assumed 100 MHz (device time only, no host time).

A full-model run takes minutes of RTL simulation per token, so the cost is built from proxies
with the real layer shapes and the full LM head (the timing does not depend on the data): the
first 1, 2, ... layers of the model (enough to contain one layer of each kind). A layer's cost
is the difference between consecutive prefixes, measured at a few positions and interpolated
linearly in the position (contexts below 256 are one attention block); the full model is the
first prefix plus the cost of every further layer by kind. --validate also runs the whole
model for one prefill run and one decode step and prints the extrapolation error.

Configuration: board_config() (OTPU_MCOLS / OTPU_LANES select the MXU columns and VPU lanes),
rtlsim.BOARD_UARCH, the AXI memory path with the program booted from DRAM, `lat` cycles of AXI
latency, `bw` percent of the peak DRAM bandwidth (one D-byte chunk per cycle).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu import rtlsim  # noqa: E402
from opentpu.isasim import board_config  # noqa: E402
from opentpu.llm import MODELS, load_spec, model_dir  # noqa: E402
from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, load_weights, rope_tables  # noqa: E402

SAMPLES = (0, 120, 240)          # positions where a layer's cost is measured


def _kinds(spec) -> tuple:
    return spec.kinds if hasattr(spec, "kinds") else ("attn",) * spec.layers


def _prefix(spec, n: int):
    """The model cut to its first n layers."""
    if hasattr(spec, "kinds"):
        return dataclasses.replace(spec, kinds=spec.kinds[:n])
    return dataclasses.replace(spec, layers=n)


def _proxies(kinds) -> tuple[int, dict]:
    """The prefix lengths to measure and, per layer kind, (n, n + 1): the kind's cost is
    c(n + 1) - c(n)."""
    diff = {}
    for i in range(1, len(kinds)):
        diff.setdefault(kinds[i], (i, i + 1))
    return sorted({1} | {n for p in diff.values() for n in p}), diff


class Bench:
    def __init__(self, model: str, bw: int, lat: int, cache: Path | None):
        self.path = model_dir(model)
        self.spec = load_spec(self.path)
        self.W = load_weights(self.path)
        self.bw, self.lat, self.cache = bw, lat, cache
        self.cap = 512
        self.results = json.loads(cache.read_text()) if cache and cache.exists() else {}
        self.images = {}

    def image(self, n: int):
        if n not in self.images:
            spec = _prefix(self.spec, n)
            probe = spec.image(board_config(DRAM_BYTES=1 << 40), self.cap, 1, PREFILL_ROWS)
            cfg = board_config(DRAM_BYTES=1 << max(20, (probe.nbytes - 1).bit_length()))
            img = spec.image(cfg, self.cap, 1, PREFILL_ROWS)
            dram = img.build(self.W)[0]
            x = np.asarray(self.W["model.embed_tokens.weight"][791:791 + PREFILL_ROWS],
                           np.float32)
            b = np.ascontiguousarray(x).view(np.uint8).reshape(-1)
            dram[img.io["x"]:img.io["x"] + b.size] = b
            self.images[n] = (img, dram)
        return self.images[n]

    def cycles(self, n: int, kind: str, p0: int, rows: int = 1, head: bool = True) -> int:
        """Cycles of one run of the n-layer prefix: a decode step at p0 ("step") or `rows`
        prompt rows at p0 .. ("rows"; logits for the last one with `head`)."""
        key = f"{self.path.name}|{os.environ.get('OTPU_MCOLS', '2')}|{self.bw}|" \
              f"{self.lat}|{n}|{kind}|{p0}|{rows}|{head}"
        if key in self.results:
            return self.results[key]
        img, dram = self.image(n)
        if kind == "step":
            progs = img.compile_step(p0)
        else:
            progs = img.compile_rows([(0, p0 + j) for j in range(rows)],
                                     [rows - 1] if head else [])
        dram = dram.copy()
        cs = [rope_tables(img.spec, p0 + j) for j in range(rows)]
        for k, v in (("cos", np.stack([c for c, _ in cs])), ("sin", np.stack([s for _, s in cs]))):
            b = np.ascontiguousarray(v, np.float32).view(np.uint8).reshape(-1)
            dram[img.io[k]:img.io[k] + b.size] = b
        cfg = img.cfg
        if 8 * len(progs[0]) > cfg.IMEM_WORDS:      # a proxy of the long program: time it
            cfg = dataclasses.replace(cfg, IMEM_WORDS=1 << (8 * len(progs[0]) - 1).bit_length())
        t = time.time()
        with tempfile.TemporaryDirectory(prefix="otpu_prefill_") as d:
            _, _, st = rtlsim.run(cfg, progs, [dram], uarch=rtlsim.BOARD_UARCH, axi=True,
                                  boot=True, stall=0, bw=self.bw, lat=self.lat, keep=Path(d),
                                  max_cycles=1 << 40)
        print(f"  {n} layers {kind} p0={p0} rows={rows} head={head}: {st['cycles']} cycles "
              f"({time.time() - t:.0f} s)", flush=True)
        self.results[key] = int(st["cycles"])
        if self.cache:
            self.cache.write_text(json.dumps(self.results, indent=1))
        return self.results[key]

    def model_cost(self, kind: str, p0: int, rows: int, head: bool) -> float:
        """The full model's cycles for one run, from the prefixes (layer costs interpolated
        between the SAMPLES positions for full-size runs)."""
        kinds = _kinds(self.spec)
        ns, diff = _proxies(kinds)

        def c(n):
            if kind == "rows" and (rows, head) != (self.R, False):
                return self.cycles(n, kind, p0, rows, head)     # a run of its own: measured
            lo = max(s for s in SAMPLES if s <= p0) if p0 < SAMPLES[-1] else SAMPLES[-2]
            hi = SAMPLES[SAMPLES.index(lo) + 1]
            a, b = self.cycles(n, kind, lo, rows, head), self.cycles(n, kind, hi, rows, head)
            return a + (b - a) * (p0 - lo) / (hi - lo)

        total = c(1)
        for k in kinds[1:]:
            i, j = diff[k]
            total += c(j) - c(i)
        return total

    def plan(self, P: int) -> list:
        """The Engine's prefill runs for a P-token prompt: [(p0, rows)]."""
        img = self.spec.image(board_config(), self.cap, 1, PREFILL_ROWS)
        eng = Engine.__new__(Engine)
        eng.image, eng.cfg, eng.block, eng.backend = img, img.cfg, 256, None
        eng._fit_rows = PREFILL_ROWS
        out, p0 = [], 0
        while p0 < P:
            n, _ = eng._chunk(0, p0, PREFILL_ROWS, P - p0)
            out.append((p0, n))
            p0 += n
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3",
                    help=f"{' or '.join(MODELS)} (models/<name>), or a checkpoint directory")
    ap.add_argument("--prompts", default="16,64,256")
    ap.add_argument("--bw", type=int, default=100)
    ap.add_argument("--lat", type=int, default=30)
    ap.add_argument("--cache", type=Path, help="keep RTL results between invocations")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    bench = Bench(a.model, a.bw, a.lat, a.cache)
    prompts = [int(x) for x in a.prompts.split(",")]
    plans = {P: bench.plan(P) for P in prompts}
    ns = [n for pl in plans.values() for _, n in pl]
    bench.R = max(set(ns), key=ns.count)       # the usual run: interpolated; others measured
    print(f"{bench.path.name}: {len(_kinds(bench.spec))} layers, prefill rows {bench.R} "
          f"(MCOLS={board_config().MCOLS}), bw {a.bw}%, lat {a.lat}; simulated RTL cycles")
    rows = []
    for P in prompts:
        dec = sum(bench.model_cost("step", p, 1, True) for p in range(P))
        pre = 0.0
        for i, (p0, n) in enumerate(plans[P]):
            last = i + 1 == len(plans[P])
            pre += bench.model_cost("step", p0, 1, True) if n == 1 else \
                bench.model_cost("rows", p0, n, last)
        rows.append((P, dec, pre, plans[P]))
    print(f"\n{'prompt':>6} {'runs':>5} {'decode cyc/tok':>15} {'prefill cyc/tok':>16} "
          f"{'speedup':>8} {'TTFT before':>12} {'TTFT after':>11}  (100 MHz, device only)")
    for P, dec, pre, pl in rows:
        print(f"{P:6d} {len(pl):5d} {dec / P:15,.0f} {pre / P:16,.0f} {dec / pre:7.2f}x "
              f"{dec / 1e8:11.2f}s {pre / 1e8:10.2f}s")
    if a.validate:
        n = len(_kinds(bench.spec))
        for kind, p0, r, head in (("rows", SAMPLES[1], bench.R, False), ("step", SAMPLES[1], 1,
                                                                          True)):
            want = bench.model_cost(kind, p0, r, head)
            got = bench.cycles(n, kind, p0, r, head)
            print(f"validate {kind} p0={p0} rows={r}: full model {got}, extrapolated "
                  f"{want:.0f} ({100 * (want - got) / got:+.3f}%)")


if __name__ == "__main__":
    main()
