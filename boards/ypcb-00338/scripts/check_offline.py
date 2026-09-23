#!/usr/bin/env python3
"""Offline checks of the Vivado flow (no Vivado needed):
  - every Tcl script is complete (balanced braces/quotes, via tclsh `info complete`);
  - every port an XDC constrains exists on otpu_fpga_top, with a valid bit index;
  - every top-level port bit has a location (except the GT lanes and the refclk N side, which
    the XDMA IP / the GT reference-clock buffer place);
  - no package pin is assigned twice, and every pin exists in the xc7k480t-ffg1156 IOB map.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOARD = HERE.parent
ROOT = BOARD.parent.parent
TOP = ROOT / "rtl/boards/ypcb-00338/otpu_fpga_top.sv"
XDCS = [BOARD / "constraints/otpu_top.xdc", BOARD / "constraints/otpu_ddr3_pins.xdc"]
GT_PINS = {"J8"}          # MGTREFCLK: not an IOB


def top_ports() -> dict[str, int]:
    src = TOP.read_text()
    hdr = src[src.index("module otpu_fpga_top"):src.index(");")]
    ports = {}
    for m in re.finditer(r"(input|output|inout)\s+(?:logic|wire)\s*(\[(\d+):(\d+)\])?\s*(\w+)", hdr):
        ports[m[5]] = int(m[3]) - int(m[4]) + 1 if m[2] else 0   # 0: scalar
    return ports


def main() -> int:
    bad = []
    # ---- Tcl
    for tcl in sorted((BOARD / "vivado").glob("*.tcl")):
        r = subprocess.run(["tclsh"], input=f"set f [open {{{tcl}}}]; set s [read $f]; "
                           f"puts [info complete $s]", capture_output=True, text=True)
        ok = r.stdout.strip() == "1"
        print(f"tcl {tcl.name}: {'ok' if ok else 'INCOMPLETE ' + r.stderr}")
        if not ok:
            bad.append(tcl.name)
    # ---- XDC vs ports
    ports = top_ports()
    sites = {l.split()[0] for l in (HERE / "xc7k480t_ffg1156_iob.txt").read_text().split("\n") if l}
    placed, pads = set(), {}
    for x in XDCS:
        for n, line in enumerate(x.read_text().split("\n"), 1):
            if line.lstrip().startswith("#"):
                continue
            for m in re.finditer(r"get_ports \{?([\w]+)(?:\[(\d+|\*)\])?\}?", line):
                name, idx = m[1], m[2]
                if name not in ports:
                    bad.append(f"{x.name}:{n}: no port {name}")
                    continue
                if idx not in (None, "*") and not (0 <= int(idx) < max(ports[name], 1)):
                    bad.append(f"{x.name}:{n}: {name}[{idx}] out of range")
                pm = re.search(r"PACKAGE_PIN (\w+)", line)
                if pm:
                    bit = f"{name}[{idx}]" if idx is not None else name
                    placed.add(bit)
                    if pm[1] in pads and pads[pm[1]] != bit:
                        bad.append(f"pin {pm[1]} on both {pads[pm[1]]} and {bit}")
                    pads[pm[1]] = bit
                    if pm[1] not in sites and pm[1] not in GT_PINS:
                        bad.append(f"{x.name}:{n}: pin {pm[1]} is not a user IO of the part")
    for name, w in ports.items():
        if name.startswith("pcie_mgt_") or name == "pcie_refclk_clk_n":
            continue
        bits = [name] if w == 0 else [f"{name}[{i}]" for i in range(w)]
        for b in bits:
            if b not in placed:
                bad.append(f"port {b} has no PACKAGE_PIN")
    print(f"xdc: {len(ports)} top ports, {len(pads)} placed pins")
    for b in bad:
        print("  ERROR", b)
    print("offline checks:", "FAILED" if bad else "ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
