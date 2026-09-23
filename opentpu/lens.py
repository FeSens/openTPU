"""Lens: the openTPU profiler report.

Turns RTL traces (see profile.py) into one self-contained HTML page per set of workloads:
headline metrics and a verdict, a zoomable per-unit timeline with DRAM / MXU / TMEM-arbitration
strips, a roofline chart, unit utilisation, bottleneck notes, and cycles attributed to the
kernel source lines that issued the instructions.

    from opentpu.lens import write_report
    write_report([profile(...), profile(...)], "build/lens.html")

or run tools/lens.py.
"""
from __future__ import annotations

import json
import linecache
import os
from pathlib import Path

from . import isa as I
from . import rtlsim
from .profile import UNITS, Profile

CLOCK_MHZ = 200          # Kintex-7 target clock
TEMPLATE = Path(__file__).with_name("lens_template.html")
ROOT = Path(__file__).resolve().parent.parent


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def notes(p: Profile) -> list[dict]:
    """Bottleneck analysis in plain sentences, most important first."""
    out = []
    rl = p.roofline()
    eff = rl["efficiency"]
    cyc = p.cycles
    b = p.ports[0] if p.ports and p.ports[0] else {}
    portb = (b.get("bmxu", 0) + b.get("bdma", 0)) / cyc if cyc else 0
    gaps = p.mxu_gaps(0)
    if eff >= 0.95:
        out.append({"level": "good", "text":
                    f"At the roofline: {_pct(eff)} of the DRAM-bound minimum. DRAM port B was "
                    f"busy {_pct(portb)} of the run on slice 0; the kernel is memory-bound, as a "
                    f"decode kernel should be."})
    elif eff >= 0.85:
        out.append({"level": "warn", "text":
                    f"Near the roofline: {_pct(eff)} of the DRAM-bound minimum "
                    f"({cyc - rl['bound']} cycles above it)."})
    else:
        out.append({"level": "bad", "text":
                    f"Below the roofline: {_pct(eff)} of the DRAM-bound minimum. DRAM port B "
                    f"idled {_pct(1 - portb)} of the run."})
    lost = cyc - rl["bound"]
    if lost > 0:
        parts = []
        if gaps["prologue"]:
            parts.append(f"{gaps['prologue']} before the MXU could consume its first stream")
        gsum = sum(e - s_ for s_, e, _ in gaps["gaps"])
        if gsum:
            parts.append(f"{gsum} in MXU gaps mid-run")
        if gaps["epilogue"]:
            parts.append(f"{gaps['epilogue']} after the last matmul")
        if parts:
            out.append({"level": "info", "text": "Cycles off the stream (slice 0): "
                        + ", ".join(parts) + "."})
    by_loc = {}
    for (name, pc), g in gaps["blame"].items():
        if pc < 0:
            continue
        key = (name, _loc(p.programs[0][pc]))
        by_loc[key] = by_loc.get(key, 0) + g
    for (name, loc), g in sorted(by_loc.items(), key=lambda kv: -kv[1])[:3]:
        if g >= 0.01 * cyc:
            out.append({"level": "warn", "text":
                        f"The MXU waited {g} cycles ({_pct(g / cyc)}) for {name} at {loc}."})
    lose = _losses(p, 0)
    for u, n in sorted(lose.items(), key=lambda kv: -kv[1]):
        if n > 0.02 * cyc:
            out.append({"level": "warn", "text":
                        f"The {u} lost {n} cycles ({_pct(n / cyc)}) to TMEM bank arbitration: "
                        f"another unit held the banks it needed."})
    busy = p.unit_busy(0)
    if busy.get("VPU", 0) > 0.85 * cyc and eff < 0.95:
        out.append({"level": "bad", "text":
                    "The VPU is busy almost all the time: the kernel is vector-bound. Fuse "
                    "elementwise work into the MXU/quantizer epilogues or widen the VPU (LANES)."})
    if b and b.get("bdma", 0) > 0.1 * cyc:
        out.append({"level": "info", "text":
                    f"The DMA used {_pct(b['bdma'] / cyc)} of DRAM port B for loads and "
                    f"stores; that traffic counts toward the roofline."})
    return out


def _loc(ins) -> str:
    if not ins.src:
        return "(compiler)"
    f, line, fn = ins.src[0]
    return f"{os.path.basename(f)}:{line} in {fn}"


def _losses(p: Profile, s: int) -> dict:
    b = p.buckets[s] if getattr(p, "buckets", None) else {}
    return {"MXU drain": sum(b.get("fm", [])), "QUANT": sum(b.get("fq", [])),
            "VPU": sum(b.get("fv", [])), "COLL": sum(b.get("fc", []))}


def to_data(p: Profile, uarch: dict | None = None) -> dict:
    """Everything the report page needs, as plain JSON."""
    cfg = p.cfg
    ua = dict(rtlsim.UARCH)
    ua.update(uarch or {})
    rl = p.roofline()
    sources, src_index = [], {}

    def src_id(ins):
        if not ins.src:
            return -1
        key = ins.src[0]              # group by the innermost line; keep its first call chain
        if key not in src_index:
            f, line, fn = ins.src[0]
            src_index[key] = len(sources)
            sources.append({"file": _rel(f), "line": line, "func": fn,
                            "text": linecache.getline(f, line).strip(),
                            "chain": [f"{os.path.basename(a)}:{b} {c}" for a, b, c in ins.src]})
        return src_index[key]

    instrs = []
    for r in p.recs:
        ins = p.programs[r.slice][r.pc]
        instrs.append([r.slice, r.idx, r.pc, r.unit, r.name, r.detail, r.dispatch, r.ready,
                       r.release, r.start, r.end, src_id(ins), r.stats, r.work])
    slices = []
    for s in range(cfg.S):
        slices.append({"ports": p.ports[s], "busy": p.unit_busy(s), "lose": _losses(p, s),
                       "buckets": p.buckets[s] if getattr(p, "buckets", None) else {}})
    macs = p.macs()
    bytes_b = sum(x["portb"] for x in rl["per_slice"]) * cfg.D
    return {
        "name": p.name, "cycles": p.cycles, "clock_mhz": CLOCK_MHZ,
        "config": {"S": cfg.S, "D": cfg.D, "MCOLS": cfg.MCOLS, "LANES": cfg.LANES,
                   "ACT_BLOCKS": cfg.ACT_BLOCKS, **ua},
        "roofline": {"bound": rl["bound"], "efficiency": rl["efficiency"],
                     "per_slice": rl["per_slice"]},
        "macs": macs, "bytes": bytes_b,
        "peak_macs": cfg.D * cfg.MCOLS * cfg.S, "peak_bytes": cfg.D * cfg.S,
        "units": UNITS, "slices": slices, "instrs": instrs, "sources": sources,
        "notes": notes(p), "mxu_gaps": {k: v for k, v in p.mxu_gaps(0).items() if k != "blame"},
    }


def render(profiles: list[Profile], uarch: dict | None = None, standalone: bool = True,
           title: str = "openTPU Lens") -> str:
    data = [to_data(p, uarch) for p in profiles]
    html = TEMPLATE.read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("/*__DATA__*/[]", payload).replace("__TITLE__", title)
    if standalone:
        html = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, "
                "viewport-fit=cover\">\n</head>\n<body>\n" + html + "\n</body>\n</html>\n")
    return html


def write_report(profiles: list[Profile], path, uarch: dict | None = None,
                 standalone: bool = True, title: str = "openTPU Lens") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(profiles, uarch, standalone, title))
    return path
