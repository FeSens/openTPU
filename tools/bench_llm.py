"""LLM benchmark on the RTL at the board configuration: prefill, batched decode and TTFT.

    python3 tools/bench_llm.py [--bw 80,100] [--batches 1,2,4,8] [--ctx 128,512,1024]
                               [--prompts 32,128,512] [--chunk 8] [--mcols 2] [--lanes 8]
                               [--validate] [--jobs 6] [--json out.json]

Model: Qwen3-0.6B shapes (models/Qwen3-0.6B/config.json if present). Timing on this machine is
data-independent (no data-dependent latencies or skips), so the model image holds random
weights and an empty KV cache. Every number comes from the Verilator RTL of the board
configuration (opentpu.isasim.board_config(), rtlsim.BOARD_UARCH, the AXI memory path with the
program booted from DRAM) at `bw` percent of the peak DRAM bandwidth (one D-byte chunk per
cycle), `lat` cycles of latency, no random stalls.

Method (full 28-layer RTL runs take minutes per token):
  layer(ctx)   per-layer cycles = c(2 layers) - c(1 layer), with the LM head skipped
  head(M)      LM head cycles for M rows = c(1 layer + head) - c(1 layer)
  fixed        c(1 layer) - layer: program boot, I/O loads, pipeline fill/drain
  step         fixed + 28 * layer + head
  b=1 decode uses the decode kernel qwen3_step (always with its head: c(1) and c(2) with head);
  b>1 decode uses qwen3_rows with b sequences at the same context; prefill of P tokens runs
  chunks of `chunk` rows (qwen3_rows, one sequence) at chunk starts 0, C, 2C, ...; their
  per-layer cost is measured at a few chunk starts and interpolated (piecewise linear in the
  start position); only the last chunk runs the LM head (one row). TTFT = prefill with the
  head, i.e. the time until the first generated token's logits are in DRAM (host time
  excluded). --validate runs the full 28-layer model once for b=1 and b=2 and compares.

Roofline per step (at `bw`): DRAM = (weights + LM head + KV cache read) / (D * bw/100) bytes
per cycle, compute = MACs / (MCOLS * D) per cycle (weights: rows x parameters; attention:
2 x n_q x head_dim x context per row per layer); the bound is the larger, "achieved" is
bound / measured. "MXU-stream" is this MXU's own bound: an MM holds MCOLS rows and streams
its weights from DRAM, so R rows stream every weight ceil(R / MCOLS) times (prefill: per
chunk).

MCOLS (MXU stationary rows, <= LANES) and LANES are configuration overrides for MXU studies.
Programs that exceed the board's 4K-instruction IMEM (large batch x long context: attention is
unrolled per row, KV head and block) are run with a larger IMEM and flagged "imem".
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu import rtlsim  # noqa: E402
from opentpu.isasim import board_config  # noqa: E402
from opentpu.llm.qwen3 import Image, Spec, rope_tables  # noqa: E402

MODEL = ROOT / "models" / "Qwen3-0.6B"
QWEN3_0_6B = Spec(hidden=1024, layers=28, n_q=16, n_kv=8, head_dim=128, ffn=3072, vocab=151936)
F_MHZ = 100


def model_spec() -> Spec:
    return Spec.from_hf(MODEL) if (MODEL / "config.json").exists() else QWEN3_0_6B


def random_weights(spec: Spec, seed: int = 0) -> dict:
    """HF-named random weights of `spec` (float32; timing does not depend on values)."""
    rng = np.random.default_rng(seed)
    H, d, F_ = spec.hidden, spec.head_dim, spec.ffn

    def r(*shape):
        return (rng.standard_normal(shape, np.float32) * 0.02).astype(np.float32)

    W = {"model.embed_tokens.weight": r(spec.vocab, H),
         "model.norm.weight": np.ones(H, np.float32)}
    for i in range(spec.layers):
        p = f"model.layers.{i}."
        W.update({p + "input_layernorm.weight": np.ones(H, np.float32),
                  p + "post_attention_layernorm.weight": np.ones(H, np.float32),
                  p + "self_attn.q_norm.weight": np.ones(d, np.float32),
                  p + "self_attn.k_norm.weight": np.ones(d, np.float32),
                  p + "self_attn.q_proj.weight": r(spec.n_q * d, H),
                  p + "self_attn.k_proj.weight": r(spec.n_kv * d, H),
                  p + "self_attn.v_proj.weight": r(spec.n_kv * d, H),
                  p + "self_attn.o_proj.weight": r(H, spec.n_q * d),
                  p + "mlp.gate_proj.weight": r(F_, H),
                  p + "mlp.up_proj.weight": r(F_, H),
                  p + "mlp.down_proj.weight": r(H, F_)})
    return W


# ------------------------------------------------------------------------------ roofline
def roofline(spec: Spec, cfg, rows: list, head_rows: int, bw: int) -> dict:
    """Ideal cycles of one device run over `rows` = [(seq, pos)] (all layers) plus an LM head
    over `head_rows` rows: DRAM-bound (every weight byte once, each sequence's KV cache once)
    vs compute-bound (MCOLS x D MACs per cycle)."""
    D, H, d = cfg.D, spec.hidden, spec.head_dim
    mats = [(spec.n_q * d, H), (spec.n_kv * d, H), (spec.n_kv * d, H), (H, spec.n_q * d),
            (spec.ffn, H), (spec.ffn, H), (H, spec.ffn)]
    params = sum(n * k for n, k in mats)
    wbytes = sum(n * k + 4 * n * (k // D) for n, k in mats)
    hbytes = spec.vocab * H + 4 * spec.vocab * (H // D) if head_rows else 0
    ctx = {}
    for sq, p in rows:                      # tokens of each sequence's cache read this run
        ctx[sq] = max(ctx.get(sq, 0), p + 1)
    kv_tok = spec.n_kv * (2 * d + 4 * (d // D) + 4)
    kvbytes = sum(ctx.values()) * kv_tok
    dram = (spec.layers * (wbytes + kvbytes) + hbytes) / (D * bw / 100)
    macs = spec.layers * len(rows) * params + head_rows * spec.vocab * H + \
        spec.layers * sum(2 * spec.n_q * d * (p + 1) for _, p in rows)
    comp = macs / (cfg.MCOLS * D)
    # this MXU: each MM holds MCOLS rows and streams its weights from DRAM, so R rows stream
    # every weight ceil(R / MCOLS) times
    passes = -(-len(rows) // cfg.MCOLS)
    stream = (spec.layers * (passes * wbytes + kvbytes) +
              -(-head_rows // cfg.MCOLS) * hbytes) / (D * bw / 100)
    return {"dram": dram, "compute": comp, "bound": max(dram, comp),
            "kind": "DRAM" if dram >= comp else "compute", "stream": max(stream, comp)}


# ------------------------------------------------------------------------------ RTL runs
class Bench:
    def __init__(self, spec: Spec, cfg_kw: dict, cap: int, batch: int, rows: int, bw: list,
                 lat: int, jobs: int, verbose: bool = True):
        self.spec, self.cfg_kw, self.cap, self.batch, self.rows = spec, cfg_kw, cap, batch, rows
        self.bws, self.lat, self.jobs, self.verbose = bw, lat, jobs, verbose
        self.images = {}        # layers -> (Image, dram image)
        self.progs = {}
        self.cache = None       # JSON file of RTL results (reused across invocations)
        self.W = None
        self.full_cap = 256     # the full-model (validation) image: two sequences / rows

    def cfg(self, layers: int, imem: int | None = None):
        spec = dataclasses.replace(self.spec, layers=layers)
        batch, rows = (self.batch, self.rows) if layers < self.spec.layers else (2, 2)
        cap = self.cap if layers < self.spec.layers else self.full_cap
        probe = Image(spec, board_config(DRAM_BYTES=1 << 40, **self.cfg_kw), cap, batch, rows)
        kw = dict(self.cfg_kw)
        if imem:
            kw["IMEM_WORDS"] = imem
        return board_config(DRAM_BYTES=1 << max(20, (probe.nbytes - 1).bit_length()), **kw), \
            spec, cap, batch, rows

    def image(self, layers: int):
        if layers not in self.images:
            cfg, spec, cap, batch, rows = self.cfg(layers)
            if self.W is None:
                self.W = random_weights(dataclasses.replace(self.spec, layers=2))
            W = self.W
            if layers > 2:
                W = random_weights(spec)
            img = Image(spec, cfg, cap, batch, rows)
            dram = img.build(W)[0]
            x = np.asarray(W["model.embed_tokens.weight"][791:791 + rows], np.float32)
            b = np.ascontiguousarray(x).view(np.uint8).reshape(-1)
            dram[img.io["x"]:img.io["x"] + b.size] = b
            self.images[layers] = (img, dram)
        return self.images[layers]

    def programs(self, layers: int, kind: str, rows: list, head: bool):
        img, dram = self.image(layers)
        key = (layers, kind, rows, head)
        if key in self.progs:
            return img, None, self.progs[key]
        if kind == "step":
            progs = img.compile_step(rows[0][1])
        else:
            progs = img.compile_rows(rows, [len(rows) - 1] if head == "last" else
                                     (list(range(len(rows))) if head else []))
        self.progs[key] = progs
        return img, None, progs

    def dram(self, layers: int, rows: list):
        """The model image with this run's RoPE tables in the I/O area."""
        img, dram = self.image(layers)
        dram = dram.copy()
        cs = [rope_tables(img.spec, p) for _, p in rows]
        for key, v in (("cos", np.stack([c for c, _ in cs])), ("sin", np.stack([s for _, s in cs]))):
            b = np.ascontiguousarray(v, np.float32).view(np.uint8).reshape(-1)
            dram[img.io[key]:img.io[key] + b.size] = b
        return dram

    def run(self, job):
        """job = (layers, kind, rows, head, bw) -> cycles, instructions, imem flag."""
        layers, kind, rows, head, bw = job
        img, _, progs = self.programs(layers, kind, rows, head)
        dram = self.dram(layers, rows)
        n = len(progs[0])
        imem = None
        if 8 * n > img.cfg.IMEM_WORDS:
            imem = 1 << (8 * n - 1).bit_length()
        cfg = self.cfg(layers, imem)[0]
        tmp = Path(tempfile.mkdtemp(prefix="otpu_bench_"))
        t = time.time()
        try:
            _, _, st = rtlsim.run(cfg, progs, [dram], uarch=rtlsim.BOARD_UARCH, axi=True,
                                  boot=True, stall=0, bw=bw, lat=self.lat, keep=tmp,
                                  max_cycles=1 << 40)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if self.verbose:
            print(f"  L={layers} {kind} rows={len(rows)} pos={rows[0][1]}..{rows[-1][1]} "
                  f"head={head} bw={bw}: {st['cycles']} cycles, {n} instr"
                  f"{' (imem ' + str(imem // 8) + ')' if imem else ''} [{time.time() - t:.0f}s]",
                  flush=True)
        return {"cycles": st["cycles"], "instr": n, "imem": bool(imem)}

    def prebuild(self, jobs):
        """Build every Verilator model the jobs need (serially: builds are not thread-safe)."""
        seen = set()
        for layers, kind, rows, head, bw in jobs:
            img, _, progs = self.programs(layers, kind, rows, head)
            n = len(progs[0])
            imem = 1 << (8 * n - 1).bit_length() if 8 * n > img.cfg.IMEM_WORDS else None
            cfg = self.cfg(layers, imem)[0]
            at = -(-img.nbytes // cfg.D) * cfg.D
            grow = at + 32 * n > cfg.DRAM_BYTES
            key = (cfg, grow)
            if key not in seen:
                seen.add(key)
                rtlsim.build_top(cfg, 20, rtlsim.BOARD_UARCH, True,
                                 2 * cfg.DRAM_BYTES if grow else None)

    def key(self, job) -> str:
        layers = job[0]
        return repr((job, sorted(self.cfg_kw.items()), self.lat, self.cfg(layers)[2:],
                     rtlsim.BOARD_UARCH))

    def run_all(self, jobs):
        """Run the jobs (in parallel); results already in the cache file are reused."""
        jobs = list(dict.fromkeys(jobs))
        cache = {}
        if self.cache and Path(self.cache).exists():
            cache = json.loads(Path(self.cache).read_text())
        todo = [j for j in jobs if self.key(j) not in cache]
        if todo:
            for L in sorted({j[0] for j in todo}):
                self.image(L)
            self.prebuild(todo)
            with ThreadPoolExecutor(self.jobs) as ex:
                for j, r in zip(todo, ex.map(self.run, todo)):
                    cache[self.key(j)] = r
            if self.cache:
                Path(self.cache).write_text(json.dumps(cache))
        return {j: cache[self.key(j)] for j in jobs}


def _rows_key(rows):
    return tuple(tuple(r) for r in rows)


def plan(a, spec):
    """The RTL jobs: (layers, kind, rows, head, bw)."""
    jobs = []
    C = a.chunk
    for bw in a.bw:
        for ctx in a.ctx:
            for b in a.batches:
                if b == 1:
                    for L in (1, 2):
                        jobs.append((L, "step", ((0, ctx),), True, bw))
                else:
                    rows = tuple((s, ctx) for s in range(b))
                    for L in (1, 2):
                        jobs.append((L, "rows", rows, False, bw))
        ctx0 = a.ctx[0]
        for b in sorted(set(a.batches) | {1}):
            rows = tuple((s, ctx0) for s in range(b))
            jobs.append((1, "rows", rows, True, bw))
            jobs.append((1, "rows", rows, False, bw))
        for p0 in prefill_samples(a.prompts, C):
            rows = tuple((0, p0 + i) for i in range(C))
            for L in (1, 2):
                jobs.append((L, "rows", rows, False, bw))
    return jobs


def prefill_samples(prompts, C):
    """Chunk start positions at which the prefill chunk cost is measured: the first chunk and
    the last chunk of every prompt (and the chunk before each 256-token attention block
    boundary in between)."""
    last = max(-(-P // C) * C - C for P in prompts)
    pts = {0}
    for P in prompts:
        pts.add(-(-P // C) * C - C)
    b = 256
    while b <= last:
        pts.add((b // C) * C - C)
        pts.add((b // C) * C)
        b += 256
    return sorted(p for p in pts if 0 <= p <= last)


def analyse(a, spec, res):
    """Extrapolate the proxy runs to the full model; returns the result tables."""
    L = spec.layers
    C = a.chunk
    cfg = board_config(**cfg_overrides(a))
    out = {"decode": [], "prefill": [], "head": {}, "config": {
        "MCOLS": cfg.MCOLS, "LANES": cfg.LANES, "D": cfg.D, "chunk": C, "lat": a.lat,
        "f_MHz": F_MHZ, "uarch": rtlsim.BOARD_UARCH}}
    for bw in a.bw:
        ctx0 = a.ctx[0]
        head = {}
        for b in sorted(set(a.batches) | {1}):
            rows = tuple((s, ctx0) for s in range(b))
            head[b] = res[(1, "rows", rows, True, bw)]["cycles"] - \
                res[(1, "rows", rows, False, bw)]["cycles"]
        out["head"][bw] = head
        for ctx in a.ctx:
            for b in a.batches:
                if b == 1:
                    c1 = res[(1, "step", ((0, ctx),), True, bw)]
                    c2 = res[(2, "step", ((0, ctx),), True, bw)]
                    layer = c2["cycles"] - c1["cycles"]
                    fixed = c1["cycles"] - layer
                    step = c1["cycles"] + (L - 1) * layer
                else:
                    rows = tuple((s, ctx) for s in range(b))
                    c1 = res[(1, "rows", rows, False, bw)]
                    c2 = res[(2, "rows", rows, False, bw)]
                    layer = c2["cycles"] - c1["cycles"]
                    fixed = c1["cycles"] - layer
                    step = fixed + L * layer + head[b]
                rl = roofline(spec, cfg, [(s, ctx) for s in range(b)], b, bw)
                out["decode"].append({
                    "bw": bw, "b": b, "ctx": ctx, "layer": layer, "fixed": fixed,
                    "step": step, "tok_s": b * F_MHZ * 1e6 / step,
                    "ms_per_step": step / (F_MHZ * 1e3), "roof": rl["bound"],
                    "roof_kind": rl["kind"], "roof_tok_s": b * F_MHZ * 1e6 / rl["bound"],
                    "achieved": rl["bound"] / step, "stream": rl["stream"],
                    "stream_tok_s": b * F_MHZ * 1e6 / rl["stream"],
                    "achieved_stream": rl["stream"] / step, "imem": c1["imem"] or c2["imem"]})
        samples = prefill_samples(a.prompts, C)
        per = {}
        for p0 in samples:
            rows = tuple((0, p0 + i) for i in range(C))
            c1 = res[(1, "rows", rows, False, bw)]
            c2 = res[(2, "rows", rows, False, bw)]
            layer = c2["cycles"] - c1["cycles"]
            per[p0] = (c1["cycles"] - layer + L * layer, c1["imem"] or c2["imem"])
        xs = np.array(samples, float)
        ys = np.array([per[p][0] for p in samples], float)
        for P in a.prompts:
            starts = list(range(0, P, C))
            # a short last chunk costs about as much as a full one (same weight streams)
            tot = float(sum(np.interp(p0, xs, ys) for p0 in starts)) + head[1]
            # the chunked schedule's own bound: weights streamed once per chunk
            rl_c = sum(roofline(spec, cfg, [(0, p0 + i) for i in range(min(C, P - p0))],
                                1 if p0 == starts[-1] else 0, bw)["stream"] for p0 in starts)
            # ideal prefill: the whole prompt in one pass (weights once) vs compute
            ideal = roofline(spec, cfg, [(0, i) for i in range(P)], 1, bw)
            out["prefill"].append({
                "bw": bw, "P": P, "chunks": len(starts), "cycles": tot,
                "tok_s": P * F_MHZ * 1e6 / tot, "ttft_ms": tot / (F_MHZ * 1e3),
                "roof": ideal["bound"], "roof_kind": ideal["kind"],
                "roof_tok_s": P * F_MHZ * 1e6 / ideal["bound"],
                "stream": rl_c, "stream_tok_s": P * F_MHZ * 1e6 / rl_c,
                "achieved": ideal["bound"] / tot, "achieved_stream": rl_c / tot,
                "imem": any(per[p][1] for p in samples if p < P)})
    return out


def cfg_overrides(a) -> dict:
    kw = {}
    if a.mcols:
        kw["MCOLS"] = a.mcols
    if a.lanes:
        kw["LANES"] = a.lanes
    mc, ln = kw.get("MCOLS", board_config().MCOLS), kw.get("LANES", board_config().LANES)
    if mc > ln:
        raise SystemExit(f"MCOLS {mc} must be <= LANES {ln}")
    return kw


def report(out) -> str:
    c = out["config"]
    s = [f"Qwen3-0.6B on the RTL, board config MCOLS={c['MCOLS']} LANES={c['LANES']} "
         f"D={c['D']}, {c['f_MHz']} MHz, AXI lat {c['lat']}, prefill chunk {c['chunk']}", "",
         "Decode (tokens/s over all sequences; roofline = max(DRAM, compute))",
         "| bw | b | ctx | cycles/step | ms/step | tok/s | roofline tok/s | bound | achieved "
         "| MXU-stream bound tok/s | achieved |",
         "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for r in out["decode"]:
        s.append(f"| {r['bw']}% | {r['b']} | {r['ctx']} | {r['step']:,.0f} | "
                 f"{r['ms_per_step']:.1f} | {r['tok_s']:.2f} | {r['roof_tok_s']:.2f} | "
                 f"{r['roof_kind']} | {100 * r['achieved']:.0f}% | {r['stream_tok_s']:.2f} | "
                 f"{100 * r['achieved_stream']:.0f}%{' (imem)' if r['imem'] else ''} |")
    s += ["", "Prefill (P prompt tokens from an empty cache; TTFT = prefill incl. the LM head "
          "for the last token)",
          "| bw | P | chunks | cycles | tok/s | TTFT ms | roofline tok/s | bound | achieved "
          "| MXU-stream bound tok/s | achieved |",
          "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for r in out["prefill"]:
        s.append(f"| {r['bw']}% | {r['P']} | {r['chunks']} | {r['cycles']:,.0f} | "
                 f"{r['tok_s']:.1f} | {r['ttft_ms']:.0f} | {r['roof_tok_s']:.1f} | "
                 f"{r['roof_kind']} | {100 * r['achieved']:.0f}% | {r['stream_tok_s']:.1f} | "
                 f"{100 * r['achieved_stream']:.0f}%{' (imem)' if r['imem'] else ''} |")
    s += ["", "LM head cycles by rows: " + "; ".join(
        f"bw {bw}%: " + ", ".join(f"M={m}: {v:,}" for m, v in h.items())
        for bw, h in out["head"].items())]
    for v in out.get("validate", []):
        s.append(f"Validation (full {v['layers']}-layer b={v['b']} decode, ctx {v['ctx']}, bw "
                 f"{v['bw']}%): measured {v['measured']:,} cycles, extrapolated "
                 f"{v['predicted']:,.0f} ({100 * (v['predicted'] / v['measured'] - 1):+.2f}%)")
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ints = lambda s: [int(x) for x in s.split(",")]  # noqa: E731
    ap.add_argument("--bw", type=ints, default=[80, 100])
    ap.add_argument("--lat", type=int, default=30)
    ap.add_argument("--batches", type=ints, default=[1, 2, 4, 8])
    ap.add_argument("--ctx", type=ints, default=[128, 512, 1024])
    ap.add_argument("--prompts", type=ints, default=[32, 128, 512])
    ap.add_argument("--chunk", type=int, default=8, help="prefill rows per device run (<= 8)")
    ap.add_argument("--mcols", type=int, default=None, help="override MCOLS (<= LANES)")
    ap.add_argument("--lanes", type=int, default=None, help="override LANES")
    ap.add_argument("--validate", action="store_true",
                    help="also run the full model once (b=1 decode at the first ctx, first bw)")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--json", help="write the results here")
    ap.add_argument("--cache", help="RTL result cache (JSON), reused when the RTL/kernels "
                    "are unchanged -- delete it after changing either")
    a = ap.parse_args()
    spec = model_spec()
    kw = cfg_overrides(a)
    rows = max(max(a.batches), a.chunk)
    cap = -(-(max(a.ctx + [max(a.prompts)]) + rows + 1) // 128) * 128
    bench = Bench(spec, kw, cap, max(a.batches), rows, a.bw, a.lat, a.jobs)
    bench.full_cap = -(-(a.ctx[0] + 2) // 128) * 128
    bench.cache = a.cache
    jobs = plan(a, spec)
    print(f"{len(jobs)} RTL runs", flush=True)
    t = time.time()
    res = bench.run_all(jobs)
    out = analyse(a, spec, res)
    if a.validate:
        bw, ctx = a.bw[0], a.ctx[0]
        vb = [b for b in (1, 2) if b in a.batches]
        jobs = {b: (spec.layers, "step", ((0, ctx),), True, bw) if b == 1 else
                (spec.layers, "rows", ((0, ctx), (1, ctx)), True, bw) for b in vb}
        m = bench.run_all(list(jobs.values()))
        out["validate"] = [{"layers": spec.layers, "b": b, "ctx": ctx, "bw": bw,
                            "measured": m[j]["cycles"],
                            "predicted": next(r["step"] for r in out["decode"] if r["bw"] == bw
                                              and r["b"] == b and r["ctx"] == ctx)}
                           for b, j in jobs.items()]
    print(f"[{time.time() - t:.0f}s]\n")
    print(report(out))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
