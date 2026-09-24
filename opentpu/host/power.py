"""Power: Vivado's report_power -> power.json, and otpu-smi's estimate from it.

The card cannot measure its power, so otpu-smi estimates it (docs/observability.md):

    P = fixed_w + sum over units u of units[u] x utilization(util[u])

where units[u] is the dynamic power report_power attributes to the unit's instances (at the
vectorless default activity, read as "the unit busy") and fixed_w is everything else: device
static power, the PCIe/XDMA block, the MIG controllers, clocking, the control block -- counted
in full whatever the load. It is a model, not a measurement: good for relative numbers (idle vs
decoding) and the order of magnitude.

power.json (written by build.tcl through `python3 -m opentpu.host.power`):

    {"format": "openTPU-power", "version": 1, "source": "power.rpt", "tool": "Vivado ...",
     "total_w": 5.32, "dynamic_w": 4.87, "static_w": 0.45,        # the report's summary
     "junction_c": 32.5, "confidence": "Low",
     "components": {"Clocks": 0.5, "Slice Logic": 0.3, ...},     # On-Chip Components (W)
     "units": {"MXU": 1.6, "VPU": 0.4, ...},                     # dynamic W per unit
     "util": {"MXU": "MXU_BUSY", ...},                           # the counter that scales it
     "fixed_w": 3.1,                                             # total_w - sum(units)
     "hierarchy": [[depth, "name", W], ...]}                     # By Hierarchy, in order

Units are found by instance name under u_board (UNIT_INSTANCES; the outermost match counts, so
nothing is counted twice). "DRAM" (the AXI DRAM adapter; the MIG is fixed) scales with DRAM
beats per cycle.

    python3 -m opentpu.host.power build/vivado/reports/power.xml -o build/vivado/reports/power.json
    python3 -m opentpu.host.power build/vivado/reports/power.rpt     # prints the JSON
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

FORMAT, VERSION = "openTPU-power", 1
# instance name (rtl/top/otpu_slice.sv, rtl/boards/ypcb-00338/otpu_board.sv) -> unit
UNIT_INSTANCES = {"u_mxu": "MXU", "u_act": "MXU", "u_vpu": "VPU", "u_quant": "QNT",
                  "u_dma": "DMA", "u_tmem": "TMEM", "u_seq": "SEQ", "u_mem": "DRAM"}
BOARD_INSTANCE = "u_board"          # otpu_board in otpu_fpga_top: the accelerator
# unit -> the free-running counter whose utilization scales its dynamic power ("DRAM": beats
# per cycle, from DRAM_RD + DRAM_WR)
UNIT_UTIL = {"MXU": "MXU_BUSY", "VPU": "VPU_BUSY", "QNT": "QNT_BUSY", "DMA": "DMA_BUSY",
             "TMEM": "RUNNING", "SEQ": "RUNNING", "DRAM": "DRAM"}
SUMMARY = {"Total On-Chip Power (W)": "total_w", "Dynamic (W)": "dynamic_w",
           "Device Static (W)": "static_w", "Junction Temperature (C)": "junction_c",
           "Confidence Level": "confidence"}


def _num(s: str):
    try:
        return float(s.strip().rstrip("*"))
    except ValueError:
        return None


# ------------------------------------------------------------------------------ parsers
def _tables_text(text: str) -> dict[str, list[list[str]]]:
    """Section title -> rows (cells, un-stripped) of the tables in a text report."""
    out: dict[str, list[list[str]]] = {}
    sec = ""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^\d+(\.\d+)*\.?\s+(\S.*?)\s*$", line)
        if m and i + 1 < len(lines) and re.match(r"^-{3,}\s*$", lines[i + 1]):
            sec = m.group(2)
            continue
        if line.startswith("|"):
            out.setdefault(sec, []).append(line.strip().strip("|").split("|"))
    return out


def _tables_xml(text: str) -> dict[str, list[list[str]]]:
    """Same from `report_power -format xml`: <section title=..> ... <tablerow> with
    <tableheader>/<tablecell contents=..> children. Nested tablerows (a hierarchy) get their
    depth as leading spaces, like the text report."""
    root = ET.fromstring(text)
    out: dict[str, list[list[str]]] = {}

    def rows(el, depth, sec):
        for ch in el:
            tag = ch.tag.lower()
            if tag == "section":
                rows(ch, 0, ch.get("title", sec))
            elif tag == "tablerow":
                cells = [c.get("contents", c.text or "") for c in ch
                         if c.tag.lower() in ("tablecell", "tableheader")]
                if cells:
                    cells[0] = " " * (1 + 2 * depth) + cells[0].lstrip() \
                        if depth else " " + cells[0]
                    out.setdefault(sec, []).append(cells)
                rows(ch, depth + 1, sec)
            else:
                rows(ch, depth, sec)
    rows(root, 0, "")
    return out


def parse_report(text: str, source: str = "") -> dict:
    """report_power output (text, or -format xml) -> the power.json dict."""
    tables = _tables_xml(text) if text.lstrip().startswith("<") else _tables_text(text)
    d = {"format": FORMAT, "version": VERSION, "source": source, "tool": None,
         "total_w": None, "dynamic_w": None, "static_w": None, "junction_c": None,
         "confidence": None, "components": {}, "units": {}, "util": {}, "fixed_w": None,
         "hierarchy": []}
    m = re.search(r"Tool Version\s*:\s*(.+)", text) or re.search(r'Vivado v[\d.]+', text)
    if m:
        d["tool"] = (m.group(1) if m.groups() else m.group(0)).strip()
    for sec, rs in tables.items():
        low = sec.lower()
        for cells in rs:
            if len(cells) < 2:
                continue
            k, v = cells[0].strip(), cells[1].strip()
            if k in SUMMARY and d[SUMMARY[k]] is None:
                d[SUMMARY[k]] = v if SUMMARY[k] == "confidence" else _num(v)
            elif ("on-chip components" in low and k not in ("On-Chip", "Total")
                  and _num(v) is not None):
                d["components"][k] = _num(v)
            elif "hierarchy" in low and k != "Name" and _num(v) is not None:
                raw = cells[0]
                depth = max(0, (len(raw) - len(raw.lstrip()) - 1) // 2)
                d["hierarchy"].append([depth, k, _num(v)])
    if d["total_w"] is None:
        raise ValueError(f"{source or 'report'}: no 'Total On-Chip Power' -- not a report_power "
                         "output?")
    # units: the outermost instance with a known name on each branch, inside the accelerator
    # (u_board, when the report reaches it: IP cores such as the XDMA or the MIG never count)
    scope = any(n == BOARD_INSTANCE for _, n, _ in d["hierarchy"])
    path: list[str] = []
    claimed = None                          # depth of the unit subtree we are inside
    for depth, name, w in d["hierarchy"]:
        del path[depth:]
        path.append(name)
        if claimed is not None and depth <= claimed:
            claimed = None
        u = UNIT_INSTANCES.get(name)
        if u and claimed is None and (not scope or BOARD_INSTANCE in path[:-1]):
            d["units"][u] = round(d["units"].get(u, 0.0) + w, 6)
            claimed = depth
    d["util"] = {u: UNIT_UTIL[u] for u in d["units"]}
    d["fixed_w"] = round(d["total_w"] - sum(d["units"].values()), 6)
    return d


def load(path) -> dict | None:
    """power.json, or None when absent / unreadable / another format."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return d if d.get("format") == FORMAT and int(d.get("version", 0)) <= VERSION else None


# ------------------------------------------------------------------------------ the estimate
def estimate(pj: dict, util: dict) -> dict:
    """power.json + utilizations {counter: fraction, "DRAM": beats per cycle} -> {"w", "fixed_w",
    "units": {unit: W}}. A unit whose counter is missing counts at full power."""
    units = {u: w * min(1.0, max(0.0, util.get(pj["util"].get(u, ""), 1.0)))
             for u, w in pj["units"].items()}
    return {"w": round(pj["fixed_w"] + sum(units.values()), 3), "fixed_w": pj["fixed_w"],
            "max_w": pj["total_w"], "units": {u: round(w, 3) for u, w in units.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m opentpu.host.power",
                                 description="Vivado report_power (text or XML) -> power.json")
    ap.add_argument("report")
    ap.add_argument("-o", "--out", help="write here (default: stdout)")
    a = ap.parse_args(argv)
    d = parse_report(Path(a.report).read_text(), Path(a.report).name)
    s = json.dumps(d, indent=1)
    if a.out:
        Path(a.out).write_text(s + "\n")
        print(f"wrote {a.out}: {d['total_w']} W total, units "
              + ", ".join(f"{u} {w:.3f} W" for u, w in d["units"].items()))
    else:
        print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
