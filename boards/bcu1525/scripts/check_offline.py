#!/usr/bin/env python3
"""Offline checks of the BCU1525 flow (no Vivado needed):
  - every Tcl script (vivado/, constraints/) is complete (balanced braces / quotes: tclsh `info complete`);
per DIMM variant (create_project.tcl DIMM: udimm, the x8 UDIMMs' core litedram_udimm/ and pins
ddr4_c<n>.udimm.pins.xdc; rdimm, the x4 RDIMMs' litedram/ and ddr4_c<n>.pins.xdc):
  - no package pin is placed twice across the variant's XDCs, and each port is placed once;
  - each DDR4 channel's XDC ports are the LiteDRAM core's pads (ports.txt: ddram_* ->
    ddr4_c<n>_*) plus the board-only ones (par, cs_hi[2:0], cke1, odt1, which the top ties);
  - the committed core matches its core.json's sha256s;
  - with the top present (rtl/boards/bcu1525/otpu_fpga_top_bcu.sv), every XDC port exists on it,
    with the variant's defines (OTPU_BCU_RDIMM for rdimm).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOARD = HERE.parent
ROOT = BOARD.parent.parent
TOP = ROOT / "rtl/boards/bcu1525/otpu_fpga_top_bcu.sv"
BOARD_ONLY = {"par": 1, "cs_hi": 3, "cke1": 1, "odt1": 1}
# variant -> (core directory, a channel's pin file, the top's defines)
VARIANTS = {"udimm": ("litedram_udimm", "ddr4_c{ch}.udimm.pins.xdc", set()),
            "rdimm": ("litedram", "ddr4_c{ch}.pins.xdc", {"OTPU_BCU_RDIMM"})}


def preprocess(src: str, defines: set[str]) -> str:
    """The top's `ifdef / `ifndef / `else / `endif and its object-like `defines (the DIMM pads'
    widths), enough to read its port list."""
    out, stack, macros = [], [], {}
    for line in src.splitlines():
        t = line.strip()
        m = re.match(r"`(ifdef|ifndef)\s+(\w+)", t)
        if m:
            stack.append((m[2] in defines | set(macros)) == (m[1] == "ifdef"))
            continue
        if t.startswith("`else"):
            stack[-1] = not stack[-1]
            continue
        if t.startswith("`endif"):
            stack.pop()
            continue
        if not all(stack):
            continue
        m = re.match(r"`define\s+(\w+)\s+(\S+)$", t)
        if m:
            macros[m[1]] = m[2]
            continue
        out.append(re.sub(r"`(\w+)", lambda k: macros.get(k[1], k[0]), line))
    return "\n".join(out)


def xdc_ports(path: Path) -> dict[str, str]:
    """port bit -> package pin"""
    out = {}
    for pin, braced, bare in re.findall(r"PACKAGE_PIN (\S+) \[get_ports (?:\{([^}]+)\}|(\S+?))\]",
                                        path.read_text()):
        out[braced or bare] = pin
    return out


def widths(bits) -> dict[str, int]:
    w: dict[str, int] = {}
    for b in bits:
        m = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", b)
        w[m[1]] = max(w.get(m[1], 0), int(m[2]) + 1 if m[2] else 1)
    return w


def check_variant(v: str, ld: Path, ddr_xdcs: list[Path], defines: set[str]) -> list[str]:
    bad = []
    pins: dict[str, str] = {}
    allports: dict[str, str] = {}
    for x in [BOARD / "constraints/bcu1525_pins.xdc"] + ddr_xdcs:
        for port, pin in xdc_ports(x).items():
            if pin in pins:
                bad.append(f"{v}: {pin}: {pins[pin]} and {port}")
            if port in allports:
                bad.append(f"{v}: {port} placed twice")
            pins[pin] = port
            allports[port] = pin
    print(f"{v} xdc: {len(allports)} port bits on {len(pins)} pins")
    core = {}
    for line in (ld / "ports.txt").read_text().splitlines():
        m = re.match(r"(input|output|inout)\s+(?:wire|reg)\s*(?:\[(\d+):(\d+)\])?\s*(\w+)", line)
        if m and m[4].startswith("ddram_"):
            core[m[4][len("ddram_"):]] = int(m[2]) - int(m[3]) + 1 if m[2] else 1
    for ch in range(4):
        got = widths(p[len(f"ddr4_c{ch}_"):] for p in allports if p.startswith(f"ddr4_c{ch}_"))
        want = dict(core, **BOARD_ONLY)
        if got != want:
            bad.append(f"{v} ddr4_c{ch}: XDC {sorted(set(got.items()) ^ set(want.items()))} differ from the core's pads")
    print(f"{v} ddr4: the core's {len(core)} pads + {len(BOARD_ONLY)} board-only, per channel")
    meta = json.loads((ld / "core.json").read_text())
    for f, h in meta["sha256"].items():
        if hashlib.sha256((ld / f).read_bytes()).hexdigest() != h:
            bad.append(f"{ld.name}/{f}: sha256 differs from core.json")
    print(f"{v} core.json: {len(meta['sha256'])} files")
    if TOP.exists():
        src = preprocess(TOP.read_text(), defines)
        hdr = src[src.index("module otpu_fpga_top_bcu"):src.index(");", src.index("module otpu_fpga_top_bcu"))]
        top = {m[4]: (eval(m[2]) - int(m[3]) + 1 if m[2] else 1) for m in
               re.finditer(r"(input|output|inout)\s+(?:logic|wire)\s*(?:\[([\d+-]+):(\d+)\])?\s*(\w+)", hdr)}
        for p, w in widths(allports).items():
            if top.get(p) != w:
                bad.append(f"{v}: {p}[{w}]: not a port of the top (top: {top.get(p)})")
        missing = [p for p in top if p.startswith("ddr4_") and p not in widths(allports)]
        if missing:
            bad.append(f"{v}: the top's {missing} are not placed")
    else:
        print(f"top: {TOP.relative_to(ROOT)} not written yet (docs/bcu1525.md section 8)")
    return bad


def main() -> int:
    bad = []
    for tcl in sorted((BOARD / "vivado").glob("*.tcl")) + sorted((BOARD / "constraints").glob("*.tcl")):
        r = subprocess.run(["tclsh"], input=f"set f [open {{{tcl}}}]; set s [read $f]; "
                           f"puts [info complete $s]", capture_output=True, text=True)
        ok = r.stdout.strip() == "1"
        print(f"tcl {tcl.name}: {'ok' if ok else 'INCOMPLETE ' + r.stderr}")
        if not ok:
            bad.append(tcl.name)
    for v, (ld, chfile, defines) in VARIANTS.items():
        bad += check_variant(v, BOARD / ld, [BOARD / "constraints" / chfile.format(ch=ch)
                                             for ch in range(4)], defines)
    for b in bad:
        print("FAIL", b)
    print("offline checks:", "FAIL" if bad else "ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
