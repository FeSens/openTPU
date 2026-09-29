#!/usr/bin/env python3
"""Offline checks of the Vivado flow (no Vivado needed):
  - every Tcl script is complete (balanced braces/quotes, via tclsh `info complete`);
  - every proc_sys_reset in bd.tcl and bd_native.tcl has its derived reset polarity checked;
  - every port an XDC constrains exists on the top, with a valid bit index;
  - every top-level port bit has a location (except the GT lanes and the refclk N side, which
    the XDMA IP / the GT reference-clock buffer place);
  - no package pin is assigned twice, and every pin exists in the xc7k480t-ffg1156 IOB map.
With --mem mig_native the top is otpu_fpga_top_mn (the MIG build's XDCs). With --mem litedram
it is otpu_fpga_top_ld (otpu_top_ld.xdc and the LiteDRAM core's XDC, whose pins are LOCs), and
also:
  - otpu_top_ld.xdc has every board line of otpu_top.xdc (all but the MIG's and cal_s1's);
  - the lint stub sim/otpu_litedram_stub.sv has the core's port list (litedram/otpu_litedram.v).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOARD = HERE.parent
ROOT = BOARD.parent.parent
TOPS = {"mig": ROOT / "rtl/boards/ypcb-00338/otpu_fpga_top.sv",
        "mig_native": ROOT / "rtl/boards/ypcb-00338/otpu_fpga_top_mn.sv",
        "litedram": ROOT / "rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv"}
XDCS = {"mig": [BOARD / "constraints/otpu_top.xdc", BOARD / "constraints/otpu_ddr3_pins.xdc"],
        "litedram": [BOARD / "constraints/otpu_top_ld.xdc", BOARD / "litedram/otpu_litedram.xdc"]}
XDCS["mig_native"] = XDCS["mig"]
CORE = BOARD / "litedram/otpu_litedram.v"
CORE_STUB = BOARD / "sim/otpu_litedram_stub.sv"
GT_PINS = {"J8"}          # MGTREFCLK: not an IOB


def top_ports(top: Path) -> dict[str, int]:
    src = top.read_text()
    i = src.index("module " + top.stem)
    hdr = src[i:src.index(");", i)]
    ports = {}
    for m in re.finditer(r"(input|output|inout)\s+(?:logic|wire)\s*(\[(\d+):(\d+)\])?\s*(\w+)", hdr):
        ports[m[5]] = int(m[3]) - int(m[4]) + 1 if m[2] else 0   # 0: scalar
    return ports


def module_header(path: Path, name: str) -> list[str]:
    """A module's port list, one normalized line per port."""
    src = path.read_text()
    i = src.index(f"module {name} (")
    return [" ".join(l.split()) for l in src[i:src.index(");", i)].split("\n")[1:] if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="offline checks of the Vivado flow")
    ap.add_argument("--mem", choices=("mig", "mig_native", "litedram"), default="mig")
    mem = ap.parse_args().mem
    bad = []
    # ---- Tcl
    for tcl in sorted((BOARD / "vivado").glob("*.tcl")) + sorted((BOARD / "constraints").glob("*.tcl")):
        r = subprocess.run(["tclsh"], input=f"set f [open {{{tcl}}}]; set s [read $f]; "
                           f"puts [info complete $s]", capture_output=True, text=True)
        ok = r.stdout.strip() == "1"
        print(f"tcl {tcl.name}: {'ok' if ok else 'INCOMPLETE ' + r.stderr}")
        if not ok:
            bad.append(tcl.name)
    # ---- reset polarity: C_EXT_RESET_HIGH is read-only, derived from the driving pin's POLARITY
    # at validation. A wrong one (e.g. active-high on a *_n source) holds the design in reset, so
    # bd.tcl checks every proc_sys_reset's derived value after validate_bd_design.
    for bdf in ("bd.tcl", "bd_native.tcl"):
        bd = (BOARD / "vivado" / bdf).read_text()
        chk = re.search(r"foreach \{cell want\} \{([^}]*)\}", bd)
        checked = set(chk[1].split()[0::2]) if chk else set()
        for m in re.finditer(r"proc_sys_reset\] (\w+)(\$ch)?", bd):
            names = {m[1] + c for c in "01"} if m[2] else {m[1]}
            for n in sorted(names - checked):
                bad.append(f"{bdf}: proc_sys_reset {n} has no post-validation polarity check")
        if "C_EXT_RESET_HIGH {" in bd:
            bad.append(f"{bdf}: C_EXT_RESET_HIGH is read-only in IP integrator; set the source POLARITY")
    # ---- the LiteDRAM build's copies: the MIG build's board constraints, the core's port list
    if mem == "litedram":
        ld = set((BOARD / "constraints/otpu_top_ld.xdc").read_text().split("\n"))
        for n, line in enumerate((BOARD / "constraints/otpu_top.xdc").read_text().split("\n"), 1):
            if line.strip() and not line.lstrip().startswith("#") and "mig_" not in line \
                    and "cal_s1" not in line and line not in ld:
                bad.append(f"otpu_top.xdc:{n}: not in otpu_top_ld.xdc: {line}")
        if module_header(CORE, "otpu_litedram") != module_header(CORE_STUB, "otpu_litedram"):
            bad.append(f"{CORE_STUB.name}: port list differs from {CORE.name}'s (copy it over)")
    # ---- XDC vs ports
    ports = top_ports(TOPS[mem])
    sites = {l.split()[0] for l in (HERE / "xc7k480t_ffg1156_iob.txt").read_text().split("\n") if l}
    placed, pads = set(), {}
    for x in XDCS[mem]:
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
                pm = re.search(r"(?:PACKAGE_PIN|LOC) (\w+)", line)
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
