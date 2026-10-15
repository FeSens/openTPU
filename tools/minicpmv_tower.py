"""MiniCPM-V's vision tower timed (docs/minicpmv.md 6): where an image's seconds go.

    python3 tools/minicpmv_tower.py time MODEL [--backend board|isa] [--cfg PKL] [--pool POOL]
        [--cap 2048] [--vision-staged | --vision-resident] [--no-warm]
        --image PATH[:SLICES]... [--reps 2] [--settle S]
        [--out JSON]
    python3 tools/minicpmv_tower.py cosim MODEL --cfg PKL [--size 28x36] [--phases a,b]
        [--profile] [--out JSON]

time: the engine as chatmoe.py opens it (streamed.engine, the family's formats), then per image
the host's preprocessing (minicpmv.prepare: decode, resize, slices, patch rows), and each crop
through the tower as VisionImage.encode runs it, every step timed: the programs' build (trace,
assemble: once per patch count), the host rows (patch and position-table rows) and their writes,
each run's start (program load, arguments) and wait (the expert server's host hook while the
card runs) against its device time (CYCLES / CORE_KHZ), by phase, and the projector rows' read;
staged (the Spec's default, or --vision-staged; --vision-resident: resident), the pieces'
placement (Vision.stage) and the slots' release. Then the image through the engine's own path
(minicpmv.Vision.encode, its programs cached) as a whole, `reps` times. --no-warm: without the
engine's tower wait entries computed ahead (opentpu/host/vision_warm.py). --settle: seconds to wait after the load (the engine's compiles in the
background) before the first image.

cosim: each phase's program once on the RTL with the card's LiteDRAM controllers (DDR3-1066,
the core at 133.33 MHz; tools/moe_prefill_cosim.py's harness), the configuration of --cfg, one
crop of --size patches, the weights zero (timing does not depend on the data) at their resident
addresses: its cycles, and the crop's total over the runs (VisionImage.programs' order, an
encoder layer's phases at layer 0 for every layer). --profile: each phase's cycles by function
with the VPU's and the MXU's busy cycles (moe_prefill_cosim.print_profile, on stderr), and its
units' busy cycles, DRAM ports and roofline (opentpu/profile.py; the JSON's "profile"), the
activity windows' sums (MXU pops, TMEM arbitration lost, port waits, MXU starved / blocked) and
the MMs by shape with the MXU's stall counters.

JSON: per crop the phases' runs, wall and device seconds (time), or cycles (cosim).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MHZ = 133.33


def _engine(a):
    from opentpu.host import chat as C
    from opentpu.host import streamed as ST
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec, model_dir
    path = model_dir(a.model)
    spec = load_spec(path)
    wformat, head, formats = ST.defaults(spec)
    be, cfg = C.make_backend(a.backend, spec, a.cap, a.dev, path.name)
    if a.cfg is not None:
        want = pickle.loads(Path(a.cfg).read_bytes())
        if cfg is not None and cfg != want:
            raise SystemExit(f"the device's configuration is not {a.cfg}'s")
        cfg = want
    t = time.time()
    eng = ST.engine(path, cfg or board_config(), be, a.cap, a.pool or ST.DEFAULT,
                    wformat=wformat, head_format=head, formats=formats,
                    vision_staged=True if a.vision_staged else False if a.vision_resident
                    else None, vision_warm=not a.no_warm)
    return eng, time.time() - t


def _crop(eng, vis, P, size, cache: dict, rec: dict) -> None:
    """One crop through the tower as VisionImage.encode runs it, each step timed into rec."""
    from opentpu import isa as I
    from opentpu.compiler import Words, arg_words
    v, be = eng.image.vision, eng.backend
    pc = time.perf_counter
    khz = (getattr(be, "info", None) or {}).get("core_khz")
    h, w = size
    N = h * w
    t = pc()
    if N not in cache:
        P_, runs = v.programs(N)
        t1 = pc()
        if getattr(be, "runs_words", False):
            P_ = {k: (Words.of(I.assemble(p), ra), ra) for k, (p, ra) in P_.items()}
        else:
            P_ = {k: ([p], ra) for k, (p, ra) in P_.items()}
        cache[N] = (P_, runs)
        rec["trace_s"], rec["assemble_s"] = t1 - t, pc() - t1
    P_, runs = cache[N]
    t = pc()
    p, e = v.host_rows(P, size)
    rec["host_rows_s"] = pc() - t
    t = pc()
    be.write(0, v.work["P"], p)
    be.write(0, v.work["E"], e)
    rec["write_s"], rec["write_bytes"] = pc() - t, p.nbytes + e.nbytes
    start = getattr(be, "start", None)
    ph = rec["phases"] = {}
    for name, vals in runs:
        prog, ra = P_[name]
        if "vis_row" in vals:
            vals = {"vis_row": 0}
        args = arg_words(ra, vals)
        t0 = pc()
        if start is None:
            st = be.run(prog, args=args)
            t1 = t0
        else:
            start(prog, args=args)
            t1 = pc()
            st = be.wait()
        t2 = pc()
        x = ph.setdefault(name, {"runs": 0, "start_s": 0.0, "wait_s": 0.0, "device_s": 0.0,
                                 "cycles": 0})
        cyc = (st or {}).get("cycles") or 0
        x["runs"] += 1
        x["start_s"] += t1 - t0
        x["wait_s"] += t2 - t1
        x["cycles"] += int(cyc)
        x["device_s"] += cyc / (khz * 1e3) if khz else 0.0
    t = pc()
    be.read(0, v.work["vis"], 4 * (N // 16) * v.vs.out)
    rec["read_s"] = pc() - t
    tot = {k: sum(x[k] for x in ph.values()) for k in ("runs", "start_s", "wait_s", "device_s",
                                                       "cycles")}
    rec.update({"runs": tot["runs"], "start_s": tot["start_s"], "wait_s": tot["wait_s"],
                "device_s": tot["device_s"], "cycles": tot["cycles"]})


def time_(a) -> dict:
    from opentpu.llm import minicpmv as MV
    eng, load_s = _engine(a)
    v = eng.image.vision
    from opentpu.host import streamed as ST
    out = {"model": Path(a.model).name, "backend": a.backend, "staged": not v.resident,
           "load_s": round(load_s, 1), "experts": ST.describe(eng, eng.pool_path),
           "images": []}
    if a.settle:
        time.sleep(a.settle)
    vis = MV.Vision(eng)
    cache: dict = {}
    pc = time.perf_counter
    for spec_ in a.image:
        path, _, sl = spec_.partition(":")
        t = pc()
        im = MV.prepare(path, eng.spec.vision, int(sl) if sl else None)
        r = {"image": path, "max_slices": int(sl) if sl else None, "sizes": im.sizes,
             "tokens": im.tokens, "prepare_s": pc() - t, "crops": []}
        t = pc()
        if not v.resident:
            vis.stage()
            r["stage_s"] = pc() - t
        try:
            for P, size in zip(im.patches, im.sizes):
                rec = {"size": list(size)}
                t = pc()
                _crop(eng, vis, P, size, cache, rec)
                rec["wall_s"] = pc() - t
                r["crops"].append(rec)
        finally:
            if not v.resident:
                t = pc()
                eng.server.release()
                r["release_s"] = pc() - t
        r["encode_s"] = []                  # the engine's own path, its programs warm
        for _ in range(a.reps):
            t = pc()
            vis.encode([im])
            r["encode_s"].append(round(pc() - t, 3))
        out["images"].append(r)
        _print(r)
    return out


def _print(r: dict) -> None:
    print(f"{Path(r['image']).name} (max_slices {r['max_slices']}): {len(r['sizes'])} crops "
          f"{r['sizes']}, prepare {r['prepare_s']:.2f} s"
          + (f", stage {r['stage_s']:.2f} s" if "stage_s" in r else ""), flush=True)
    for c in r["crops"]:
        print(f"  crop {c['size']}: wall {c['wall_s']:.2f} s = build "
              f"{c.get('trace_s', 0) + c.get('assemble_s', 0):.2f} + rows {c['host_rows_s']:.2f} "
              f"+ write {c['write_s']:.3f} ({c['write_bytes'] / 1e6:.1f} MB) + {c['runs']} runs "
              f"(start {c['start_s']:.2f}, wait {c['wait_s']:.2f}; device {c['device_s']:.3f} s, "
              f"{c['cycles']} cycles) + read {c['read_s']:.3f}", flush=True)
        for k, x in c["phases"].items():
            print(f"    {k:8s} x{x['runs']:3d}: start {x['start_s']:.3f} wait {x['wait_s']:.3f} "
                  f"device {x['device_s']:.3f} s", flush=True)
    print(f"  Vision.encode (programs warm): {r['encode_s']} s", flush=True)


def _windows(pf) -> dict:
    """The slice's activity windows summed (opentpu/profile.py's P and Q lines): MXU pops (mx),
    TMEM arbitration lost by the MXU drain, QUANT, the VPU, the collective (fm fq fv fc), port B
    / A requests waiting (bs as), the MXU starved (no chunk) or blocked (chunks, no pop)."""
    b = pf.buckets[0]
    return {k: int(sum(b.get(k, []))) for k in ("mx", "fm", "fq", "fv", "fc", "bs", "as",
                                                "ms", "mb")}


def _mm_shapes(pf, cfg) -> dict:
    """The MMs by shape: count, span (cycles since the MM before completed: the spans add up
    to the MXU's time from its first start), pops (nominal: chunks x replay groups, ceil(M /
    MCOLS)), and the MXU's counters over the spans (starved, back-pressured, the drain frozen
    out of TMEM, the issuer denied)."""
    from opentpu import isa as I
    out, prev = {}, None
    for r in sorted((r for r in pf.slice_recs(0) if r.op == I.MM and r.end >= 0),
                    key=lambda r: r.end):
        M = (pf.programs[0][r.pc].w[5] >> 16) & 0xFF
        a = out.setdefault(r.detail, {"count": 0, "span": 0, "pops": 0, "starve": 0, "bp": 0,
                                      "frz": 0, "deny": 0})
        a["count"] += 1
        a["span"] += r.end - (r.start if prev is None else max(prev, r.start))
        a["pops"] += r.work * -(-M // cfg.MCOLS)
        for k in ("starve", "bp", "frz", "deny"):
            a[k] += r.stats.get(k, 0)
        prev = r.end
    return out


def cosim(a) -> dict:
    sys.path.insert(0, str(ROOT / "tools"))
    import moe_prefill_cosim as MC
    from opentpu.compiler import arg_words
    from opentpu.llm import minicpmv as MV
    from opentpu.llm.qwen3 import _Bump
    vs = MV.VisionSpec.from_hf(a.model)
    cfg = replace(pickle.loads(Path(a.cfg).read_bytes()), DRAM_BYTES=MC.DRAM)
    h, w = (int(x) for x in a.size.split("x"))
    N = h * w
    vi = MV.VisionImage(vs, cfg, _Bump(), nmax=N, out_rows=N // 16, fmt=a.fmt)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    t = np.asarray(vi.resident_addresses(), np.uint32).view(np.uint8)
    dram[vi.table:vi.table + t.size] = t
    P_, runs = vi.programs(N)
    want = a.phases.split(",") if a.phases else list(P_)
    seen, prof = {}, {}
    for name, vals in runs:
        if name in want and name not in seen:
            prog, ra = P_[name]
            vals = {"vis_row": 0} if "vis_row" in vals else vals
            t0 = time.time()
            _, st = MC._rtl(cfg, [prog], dram, args=arg_words(ra, vals), profile=a.profile)
            seen[name] = st["cycles"]
            print(f"{name:8s} {st['cycles']:11d} cycles {st['cycles'] / MHZ / 1e3:8.2f} ms "
                  f"(RTL {time.time() - t0:.0f} s)", flush=True)
            if a.profile:               # the units' busy cycles and the ports' bound
                from opentpu.profile import parse
                pf = parse(st["trace"], cfg, [prog], name)
                pf.cycles = st["cycles"]
                rl = pf.roofline()
                cls = pf.by_class(0)
                gaps = pf.mxu_gaps(0)
                blame = sorted(gaps["blame"].items(), key=lambda x: -x[1])
                prof[name] = {"busy": pf.unit_busy(0), "roofline": rl["per_slice"][0],
                              "ports": {k: int(v) for k, v in pf.ports[0].items()},
                              "classes": cls, "mxu_gaps": {
                                  "prologue": gaps["prologue"], "epilogue": gaps["epilogue"],
                                  "gaps": sum(b - a for a, b, _ in gaps["gaps"]),
                                  "blame": [[k[0], k[1], v] for k, v in blame[:20]]}}
                print(pf.summary(), f"\n  roofline {rl['per_slice'][0]}", flush=True)
                for k, x in sorted(cls.items(), key=lambda x: -x[1]["busy"])[:12]:
                    print(f"  {k:14s} x{x['count']:7d} busy {x['busy']:10d} work {x['work']:10d} "
                          f"wait dep {x['wait_dep']:10d} unit {x['wait_unit']:10d}", flush=True)
                print(f"  MXU gaps {prof[name]['mxu_gaps']['gaps']} cycles (prologue "
                      f"{gaps['prologue']}, epilogue {gaps['epilogue']}); blame: "
                      + ", ".join(f"{k[0]}@{k[1]} {v}" for k, v in blame[:8]), flush=True)
                prof[name]["windows"] = win = _windows(pf)
                prof[name]["mm_shapes"] = shapes = _mm_shapes(pf, cfg)
                print("  windows: " + "  ".join(f"{k} {v} ({100 * v / pf.cycles:.1f}%)"
                                                for k, v in win.items()), flush=True)
                for k, x in sorted(shapes.items(), key=lambda x: -x[1]["span"]):
                    print(f"  MM {k:34s} x{x['count']:5d} span {x['span']:9d} pops "
                          f"{x['pops']:9d} starve {x['starve']:8d} bp {x['bp']:8d} frz "
                          f"{x['frz']:8d} deny {x['deny']:7d}", flush=True)
    count: dict = {}
    for name, _ in runs:
        count[name] = count.get(name, 0) + 1
    tot = sum(seen[k] * count[k] for k in seen)
    out = {"size": [h, w], "fmt": a.fmt, "cycles": seen, "runs": count, "profile": prof,
           "crop_cycles": tot if len(seen) == len(count) else None,
           "crop_ms": round(tot / MHZ / 1e3, 1) if len(seen) == len(count) else None}
    print(f"crop {h}x{w}: " + (f"{out['crop_ms']} ms ({tot} cycles)" if out["crop_ms"] else
                               f"phases {sorted(seen)} of {sorted(count)}"), flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("time", "cosim"))
    ap.add_argument("model")
    ap.add_argument("--backend", choices=("isa", "board"), default="board")
    ap.add_argument("--dev", help="the card's device prefix (default: /dev/otpu0 if present, "
                    "else /dev/xdma0)")
    ap.add_argument("--cfg", help="the card's configuration, pickled (tools/qual/refs.py cfg)")
    ap.add_argument("--pool", help="the expert pool (default: the checkpoint's)")
    ap.add_argument("--cap", type=int, default=2048)
    vis = ap.add_mutually_exclusive_group()
    vis.add_argument("--vision-staged", action="store_true",
                     help="the tower's weights staged through the expert slots (the Spec's "
                          "default)")
    vis.add_argument("--vision-resident", action="store_true",
                     help="the tower's weights resident in the image")
    ap.add_argument("--no-warm", action="store_true",
                    help="no wait entries computed ahead (opentpu/host/vision_warm.py)")
    ap.add_argument("--image", action="append", default=[], metavar="PATH[:SLICES]")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--settle", type=float, default=0.0)
    ap.add_argument("--size", default="28x36", help="cosim: the crop's patches, h x w")
    ap.add_argument("--fmt", default="int8", help="cosim: the tower's formats")
    ap.add_argument("--phases", help="cosim: these phases only (default all)")
    ap.add_argument("--profile", action="store_true",
                    help="cosim: each phase's cycles by function, the VPU's and the MXU's")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.mode == "cosim" and not a.cfg:
        ap.error("cosim needs --cfg")
    r = time_(a) if a.mode == "time" else cosim(a)
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
