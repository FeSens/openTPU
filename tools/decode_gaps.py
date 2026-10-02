"""Where a decode token's cycles go when the weights are not streaming: one token of the whole
model on the co-simulated controller (tools/perf_ddr.py's point: --mem ldc, the board's
configuration from the environment), traced (+trace, P / Q windows of --bucket cycles), then
every gap between two MMs (the MXU not streaming) blamed on the instruction that finished last
before the next MM was released (opentpu/profile.py mxu_gaps), with the DRAM traffic of the gap
(the P windows' port counters) and the program's region (the layer's part, from the pc).

    python3 tools/decode_gaps.py IMGDIR --model qwen3 [--format fp4] [--mhz 133.33]
                                 [--bucket 16] [--drop PC-PC,...] [--drop-match REGEX]
                                 [--uarch K=V,...] [--hoist [--check]] [--json out.json]

--drop / --drop-match replace instructions with NOPs (the program's length and its loop stay):
the token's cycles without them, a bound for making that work free or hidden. The data come out
wrong; only the cycles count. --uarch changes the board's timing-only knobs (WIN, WPB,
FIFO_DEPTH, ...). --hoist moves attention's score MMs ahead of the previous PV MM; --check runs
the ISA simulator on both programs first. The image is built once in IMGDIR (perf_ddr.py's).
OTPU_LDC_BREAK=1 adds the controller's cycle breakdown (+ldc_break, sim/verilator/otpu_ldc_mem.sv:
each channel's BRK line, printed and in the JSON as "brk"). Results: docs/litedram.md section 11,
"The core's own gaps".
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from opentpu import isa as I  # noqa: E402
from opentpu import profile as PR  # noqa: E402
from opentpu import rtlsim  # noqa: E402
from opentpu.isasim import board_config  # noqa: E402
from opentpu.llm import load_spec, model_dir  # noqa: E402
from opentpu.llm.qwen3 import ATTN_BLOCK  # noqa: E402
from perf_ddr import MAX_CYCLES, Grid, LazyW  # noqa: E402
from perf_prefill import _kinds, _prefix  # noqa: E402


def programs(spec, n, wf, head, pos, cap, size):
    """The decode programs perf_ddr.Grid.image assembles into the image (compiled again)."""
    sp = _prefix(spec, n)
    img = sp.image(board_config(DRAM_BYTES=size), cap, wformat=wf, head_format=head, lookup=True)
    K = getattr(img.spec, "conv_k", 1)
    b = pos // ATTN_BLOCK + 1
    progs, _ = img.compile_decode(b, max((b - 1) * ATTN_BLOCK, K - 1), ATTN_BLOCK)
    return progs


def parse_drop(s: str) -> list[int]:
    out = []
    for part in filter(None, (s or "").split(",")):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def hoist_scores(prog) -> list:
    """Attention's score MMs ahead of the previous PV MM: each run of MMs with '+rowmax' (a
    head's scores against the cached K blocks) moves up to just before the nearest earlier MM
    with 'acc*=sc' (the previous head's softmax-weighted V), when no other score run lies
    between, so the in-order MXU streams the next head's K while the VPU runs the softmax.
    The order is only valid where nothing in between conflicts: --check compares the ISA
    simulator's results."""
    p = list(prog)
    i = 0
    while i < len(p):
        if p[i].op == I.MM and "+rowmax" in (p[i].comment or ""):
            j = i
            while j < len(p) and p[j].op == I.MM and "+rowmax" in (p[j].comment or ""):
                j += 1
            k = i - 1
            while k >= 0 and not (p[k].op == I.MM and ("acc*=sc" in (p[k].comment or "") or
                                                       "+rowmax" in (p[k].comment or ""))):
                k -= 1
            if k >= 0 and "acc*=sc" in (p[k].comment or "") and p[k].op != I.LOOP:
                p[k:j] = p[i:j] + p[k:i]
            i = j
        else:
            i += 1
    return p


def check(meta, prog, new, cfg) -> str:
    """The ISA simulator on the image with each program: the same DRAM and TMEM, or why not."""
    from opentpu.isasim import Machine
    img = np.fromfile(meta["bin"], ">u4", count=meta["at"] // 4).astype("<u4").view(np.uint8)
    out = []
    for q in (prog, new):
        dram = np.zeros(meta["DRAM_BYTES"], np.uint8)
        dram[:len(img)] = img
        m = Machine(cfg, [q], [dram], args=meta["args"]).run()
        out.append((m.slices[0].dram, m.slices[0].tmem))
    if not np.array_equal(out[0][1], out[1][1]):
        return f"TMEM differs at {np.nonzero(out[0][1] != out[1][1])[0][:4]}"
    if not np.array_equal(out[0][0], out[1][0]):
        return f"DRAM differs at {np.nonzero(out[0][0] != out[1][0])[0][:4]}"
    return "same"


def simulate(meta, prog, mhz, ddr, bucket, trace_p: Path, changed: bool,
             uarch: dict | None = None) -> int:
    size = meta["DRAM_BYTES"]
    cfg = dataclasses.replace(board_config(DRAM_BYTES=size), IMEM_WORDS=meta["IMEM_WORDS"])
    exe = rtlsim.build_top(cfg, 20, {**rtlsim.BOARD_UARCH, **(uarch or {})}, True,
                           2 * size if meta["grow"] else None, ddr)
    with tempfile.TemporaryDirectory(prefix="otpu_gaps_", dir=trace_p.parent) as d:
        d = Path(d)
        if changed:       # the image with the program's words replaced (same length)
            words = np.asarray(I.assemble(prog), "<u4")
            shutil.copyfile(meta["bin"], d / "dram_0.bin")
            with open(d / "dram_0.bin", "r+b") as f:
                f.seek(meta["at"])
                words.astype(">u4").tofile(f)
            rtlsim._write_hex(d / "prog_0.hex", words)
        else:
            (d / "dram_0.bin").symlink_to(meta["bin"])
            (d / "prog_0.hex").symlink_to(meta["hex"])
        (d / "dram_out_0.bin").symlink_to("/dev/null")
        args = [str(exe), f"+dir={d}", f"+max_cycles={MAX_CYCLES}", "+trace", f"+bucket={bucket}"]
        if os.environ.get("OTPU_LDC_BREAK"):
            args += ["+ldc_break"]     # the controller's cycle breakdown (otpu_ldc_mem.sv)
        args += rtlsim.ldc_plusargs(ddr, mhz)
        args += [f"+arg{k}={int(v) & 0xFFFFFFFF}" for k, v in enumerate(meta["args"])]
        args += ["+axi_stall=0", f"+axi_seed={rtlsim.MEMORY['SEED']}", "+axi_bw=100",
                 f"+axi_lat={round(0.3 * mhz)}"]
        args += ["+boot", f"+boot_addr={meta['at']}", f"+boot_n={meta['boot_n']}"]
        with open(trace_p, "w") as f:
            r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, timeout=7200)
    tail = trace_p.read_bytes()[-4000:].decode(errors="replace")
    m = re.search(r"RESULT cycles=(\d+) halted=(\d+) error=(\d+)", tail)
    if r.returncode or not m or m.group(2) != "1" or m.group(3) != "0":
        raise RuntimeError(f"RTL simulation failed:\n{tail[-2000:]}")
    return int(m.group(1))


def region_names(prog) -> list[str]:
    """Each pc's region: the layer's parts as the weight MMs split them (an MM of one row of
    activations against a matrix from DRAM: 'mm 1xK . NxK^T'); within the attention block
    (between the q/k/v and the o projections) the parts are the attention's own."""
    names, cur, k = [], "prologue", 0
    for pc, ins in enumerate(prog):
        c = ins.comment or ""
        if ins.op == I.MM and c.startswith("mm 1x"):
            k += 1
            cur = f"wmm{k}"
        names.append(cur)
    return names


def analyse(p: PR.Profile, prog, cycles: int, bucket: int) -> dict:
    gaps = p.mxu_gaps(0)
    recs = {r.idx: r for r in p.slice_recs(0)}
    b = p.buckets[0]
    c = np.asarray(b.get("c", []), np.int64)
    act = sum(np.asarray(b.get(k, [0] * len(c)), np.int64) for k in ("bm", "bd", "am", "aq"))
    win_idle = act == 0
    names = region_names(prog)
    by_pc: dict = {}
    for s, e, who in gaps["gaps"]:
        r = recs.get(who)
        pc = r.pc if r else -1
        lo, hi = np.searchsorted(c, s), np.searchsorted(c, e)
        idle = int(win_idle[lo:hi].sum()) * bucket
        a = by_pc.setdefault(pc, {"gaps": 0, "cycles": 0, "idle_windows": 0,
                                  "op": r.name if r else "?", "comment": r.comment if r else "",
                                  "region": names[pc] if 0 <= pc < len(names) else "?"})
        a["gaps"] += 1
        a["cycles"] += e - s
        a["idle_windows"] += idle
    by_region: dict = {}
    for pc, a in by_pc.items():
        x = by_region.setdefault(a["region"], {"cycles": 0, "idle_windows": 0, "gaps": 0})
        for k in ("cycles", "idle_windows", "gaps"):
            x[k] += a[k]
    q = {k: int(np.sum(b.get(k, [0]))) for k in ("bs", "as", "ms", "mb", "fm", "fq", "fv", "fc")}
    return {"cycles": cycles, "prologue": gaps["prologue"], "epilogue": gaps["epilogue"],
            "gap_cycles": sum(e - s for s, e, _ in gaps["gaps"]), "n_gaps": len(gaps["gaps"]),
            "idle_window_cycles": int(win_idle.sum()) * bucket, "windows": len(c),
            "unit_busy": p.unit_busy(0), "stalls": q,
            "by_pc": {str(k): v for k, v in sorted(by_pc.items())}, "by_region": by_region}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("imgdir", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--format", default="fp4", choices=["int8", "int4", "fp4"])
    ap.add_argument("--head", default="int8", choices=["int8", "int4", "fp4"])
    ap.add_argument("--mhz", type=float, default=133.33)
    ap.add_argument("--ddr", type=int, default=1066)
    ap.add_argument("--pos", type=int, default=544)
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--bucket", type=int, default=16)
    ap.add_argument("--drop", default="", help="pcs to replace with NOPs: PC or PC-PC, comma list")
    ap.add_argument("--drop-match", default="", help="also every instruction whose 'OP comment' "
                    "matches this regular expression (e.g. '^VOP exp2sub')")
    ap.add_argument("--hoist", action="store_true", help="attention's score MMs ahead of the "
                    "previous PV MM (hoist_scores)")
    ap.add_argument("--check", action="store_true", help="with --hoist: the ISA simulator's "
                    "results of both programs compared first")
    ap.add_argument("--uarch", default="", help="the board's micro-architecture knobs changed "
                    "(timing only; rtlsim.BOARD_UARCH): K=V, comma list, e.g. WIN=32,VPU_CL=4")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--keep-trace", type=Path)
    a = ap.parse_args(argv)
    spec = load_spec(model_dir(a.model))
    n = len(_kinds(spec))
    a.imgdir.mkdir(parents=True, exist_ok=True)
    g = Grid(a.imgdir / "unused.json", a.imgdir, "ldc", a.pos, a.cap, a.head)
    meta = g.image(a.model, spec, LazyW(model_dir(a.model)), n, a.format)    # built once
    prog = programs(spec, n, a.format, a.head, a.pos, a.cap, meta["DRAM_BYTES"])[0]
    words = np.asarray(I.assemble(prog), "<u4")
    assert len(words) == 8 * meta["boot_n"], "the program differs from the image's"
    drop = parse_drop(a.drop)
    if a.drop_match:
        rx = re.compile(a.drop_match)
        drop += [pc for pc, ins in enumerate(prog)
                 if rx.search(f"{PR.OPNAMES.get(ins.op, ins.op)} {ins.comment}")]
    t = time.time()
    trace_p = a.keep_trace or a.imgdir / f"trace_{os.getpid()}.txt"
    uarch = {k: int(v) for k, v in (x.split("=") for x in filter(None, a.uarch.split(",")))}
    run_prog = [I.nop(f"dropped {ins.comment}") if pc in drop else ins
                for pc, ins in enumerate(prog)]
    cfg = dataclasses.replace(board_config(DRAM_BYTES=meta["DRAM_BYTES"]),
                              IMEM_WORDS=meta["IMEM_WORDS"])
    if a.hoist:
        run_prog = hoist_scores(run_prog)
        moved = sum(x is not y for x, y in zip(run_prog, prog))
        print(f"hoist: {moved} instructions moved", flush=True)
        if a.check:
            print(f"hoist check (ISA simulator): {check(meta, prog, run_prog, cfg)}", flush=True)
    cycles = simulate(meta, run_prog, a.mhz, a.ddr, a.bucket, trace_p,
                      bool(drop) or a.hoist, uarch)
    text = trace_p.read_text()
    p = PR.parse(text, cfg, [run_prog], a.model)
    brk = {m.group(1): {k: [int(x) for x in v.split("/")] if "/" in v else int(v)
                        for k, v in re.findall(r"(\w+)=([-\d/]+)", m.group(2))}
           for m in re.finditer(r"^BRK (ch\d) (.*)$", text, re.M)}
    if not a.keep_trace:
        trace_p.unlink()
    res = analyse(p, run_prog, cycles, a.bucket)
    res.update(model=a.model, format=a.format, mhz=a.mhz, drop=a.drop, uarch=a.uarch,
               drop_match=a.drop_match, hoist=a.hoist, brk=brk,
               sim_s=time.time() - t)
    what = ((" drop " + a.drop if a.drop else "") +
            (f" drop /{a.drop_match}/ ({len(drop)} pcs)" if a.drop_match else "") +
            (" hoist" if a.hoist else "") + (" uarch " + a.uarch if a.uarch else ""))
    print(f"{a.model} {a.format} {a.mhz} MHz{what}: {cycles} cycles; "
          f"MXU gaps {res['gap_cycles']} in {res['n_gaps']} "
          f"({100 * res['gap_cycles'] / cycles:.2f}%), "
          f"no port traffic in {res['idle_window_cycles']} "
          f"({100 * res['idle_window_cycles'] / cycles:.2f}%; {a.bucket}-cycle windows)")
    for ch, v in brk.items():
        print(f"  BRK {ch} " + " ".join(f"{k}={'/'.join(map(str, x)) if isinstance(x, list) else x}"
                                       for k, x in v.items()))
    for k, v in sorted(res["by_region"].items(), key=lambda kv: -kv[1]["cycles"])[:20]:
        print(f"  {k:10s} gaps {v['gaps']:5d}  cycles {v['cycles']:8d} "
              f"({100 * v['cycles'] / cycles:5.2f}%)  idle {v['idle_windows']:8d}")
    top = sorted(res["by_pc"].items(), key=lambda kv: -kv[1]["cycles"])[:30]
    for pc, v in top:
        print(f"  pc {pc:>4} {v['region']:8s} {v['op']:14s} {v['comment'][:40]:40s} "
              f"gaps {v['gaps']:4d} cycles {v['cycles']:7d} idle {v['idle_windows']:7d}")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
