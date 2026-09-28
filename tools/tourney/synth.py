"""Component synthesis evaluators (EVAL=yosys | vivado | vivado-remote) and the Vivado report
parsers (component OOC and full-board builds).

Each returns a dict: lut, lutram (LUTs used as memory / shift registers), ff, dsp, bram36,
bram18, logic_ns, fmax (MHz), backend, log (path). fmax is estimated from the logic-only
arrival for yosys (accept.est_fmax) and measured post-route for vivado.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .accept import est_fmax

PART = "xc7k480tffg1156-2"
YOSYS = os.environ.get("YOSYS", "yosys")
VIVADO = os.environ.get("VIVADO", str(Path.home() / "bonetto/vivado-docker/vivado"))

# LUTs taken by each LUT-RAM / shift-register primitive (7-series SLICEM)
LUTRAM_LUTS = {"RAM32M": 4, "RAM32M16": 8, "RAM64M": 4, "RAM64M8": 8, "RAM32X1D": 2,
               "RAM64X1D": 2, "RAM128X1D": 4, "RAM256X1S": 4, "RAM32X1S": 1, "RAM64X1S": 1,
               "RAM128X1S": 2, "SRL16E": 1, "SRLC32E": 1, "SRLC16E": 1}


def parse_yosys_stat(text: str) -> dict:
    """Cell counts from the first module listing of `stat` (the flattened top)."""
    first = text.split("=== design hierarchy ===")[0]
    cells: dict[str, int] = {}
    for m in re.finditer(r"^\s+(\d+)\s+([A-Za-z_][\w$]*)\s*$", first, re.M):
        cells[m.group(2)] = cells.get(m.group(2), 0) + int(m.group(1))
    return {
        "lut": sum(v for k, v in cells.items() if re.fullmatch(r"LUT[1-6]", k)),
        "lutram": sum(LUTRAM_LUTS[k] * v for k, v in cells.items() if k in LUTRAM_LUTS),
        "ff": sum(v for k, v in cells.items() if re.fullmatch(r"FD[A-Z]*", k)),
        "dsp": cells.get("DSP48E1", 0),
        "bram36": cells.get("RAMB36E1", 0),
        "bram18": cells.get("RAMB18E1", 0),
        "carry4": cells.get("CARRY4", 0),
    }


def parse_yosys_sta(text: str) -> float | None:
    m = re.search(r"Latest arrival time in '[^']*' is (\d+)", text)
    return int(m.group(1)) / 1000.0 if m else None


def yosys(root: Path, top: str, sources: list[str], params: dict, out: Path,
          timeout: int = 3600) -> dict:
    """Out-of-context Yosys synthesis of `top` (synth_xilinx, flattened, abc9) plus the
    logic-only `sta` with the Xilinx cell timing (specify) models."""
    out.mkdir(parents=True, exist_ok=True)
    srcs = " ".join(str(root / s) for s in sources)
    gens = " ".join(f"-G {k}={v}" for k, v in params.items())
    stat, sta, log = out / "stat.txt", out / "sta.txt", out / "yosys.log"
    script = (f"read_slang {srcs} --top {top} {gens} -DSYNTHESIS --allow-use-before-declare; "
              f"synth_xilinx -family xc7 -flatten -abc9 -noiopad; "
              f"read_verilog -lib -specify +/xilinx/cells_sim.v +/xilinx/cells_xtra.v; "
              f"tee -q -o {stat} stat; tee -q -o {sta} sta")
    with log.open("w") as f:
        r = subprocess.run([YOSYS, "-m", "slang", "-p", script], stdout=f,
                           stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode != 0 or not stat.exists():
        raise RuntimeError(f"yosys failed for {top}: {log.read_text()[-1500:]}")
    m = parse_yosys_stat(stat.read_text())
    m["logic_ns"] = parse_yosys_sta(sta.read_text()) if sta.exists() else None
    m["fmax"] = est_fmax(m["logic_ns"])
    m["backend"] = "yosys"
    m["log"] = str(log)
    return m


# ------------------------------------------------------------------------------ vivado
# UNTESTED: Vivado is not installed yet (the vivado-docker volume is empty). The flow is the
# standard out-of-context one; check the report parsing against a real run before trusting it.
def vivado_tcl(root: Path, top: str, sources: list[str], params: dict, out: Path,
               period_ns: float = 10.0, clock: str = "clk") -> str:
    files = " ".join(f"{{{root / s}}}" for s in sources)
    gens = " ".join(f"-generic {k}={v}" for k, v in params.items())
    return f"""
create_project -in_memory -part {PART}
set_property source_mgmt_mode None [current_project]
read_verilog -sv {files}
synth_design -top {top} -part {PART} -mode out_of_context -flatten_hierarchy rebuilt \\
  -verilog_define SYNTHESIS {gens}
create_clock -period {period_ns} -name {clock} [get_ports {clock}]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file {{{out / 'util.txt'}}}
report_timing_summary -max_paths 10 -file {{{out / 'timing.txt'}}}
set p [lindex [get_timing_paths -setup -max_paths 1 -nworst 1] 0]
puts "OTPU_WNS [get_property SLACK $p]"
puts "OTPU_PERIOD {period_ns}"
"""


def parse_vivado_util(text: str) -> dict:
    """Utilization from `report_utilization` (full board or out-of-context component). The
    first matching row wins: the summary tables come before the per-primitive breakdown."""
    def row(*names):
        for name in names:
            m = re.search(rf"^\|\s*{re.escape(name)}\s*\|\s*([\d.]+)\s*\|", text, re.M)
            if m:
                return float(m.group(1))
        return 0.0
    return {"lut": row("LUT as Logic"), "lutram": row("LUT as Memory"),
            "ff": row("Slice Registers", "Register as Flip Flop"),
            "dsp": row("DSPs"), "bram36": row("RAMB36/FIFO*", "RAMB36/FIFO"),
            "bram18": row("RAMB18")}


COLLISION = re.compile(r"Synth 8-6430\]")
CONGESTION = re.compile(r"Route 35-447\]")


def collisions(log: str) -> int:
    """Block RAMs that synthesis gave a read-address register where the RTL reads
    asynchronously (their read-after-write differs from the simulation; the build stops)."""
    return len(COLLISION.findall(log))


def congested(log: str) -> bool:
    """The router warned that congestion kept it from routing all nets."""
    return bool(CONGESTION.search(log))


def parse_ooc(log: str, util: str, period_ns: float) -> dict:
    """An OOC run's vivado.log (OTPU_WNS / OTPU_WHS lines) and util.txt."""
    res = parse_vivado_util(util)
    w = re.search(r"OTPU_WNS\s+(-?[\d.]+)", log)
    h = re.search(r"OTPU_WHS\s+(-?[\d.]+)", log)
    res["period"] = period_ns
    res["wns"] = float(w.group(1)) if w else None
    res["whs"] = float(h.group(1)) if h else None
    res["fmax"] = 1000.0 / (period_ns - res["wns"]) if w else None
    res["logic_ns"] = period_ns - res["wns"] if w else None
    res["collisions"] = collisions(log)
    res["congested"] = congested(log)
    res["backend"] = "vivado-remote"
    return res


def parse_summary(text: str) -> dict:
    """reports/SUMMARY.txt of a board build: overall WNS / WHS and each clock's period and
    slack."""
    out: dict = {"wns": None, "whs": None, "clocks": {}}
    m = re.search(r"^WNS\s+(-?[\d.]+) ns\s+WHS\s+(-?[\d.]+) ns", text, re.M)
    if m:
        out["wns"], out["whs"] = float(m.group(1)), float(m.group(2))
    for m in re.finditer(r"^(\S+)\s+period\s+([\d.]+) ns\s+slack\s+(-?[\d.]+) ns", text, re.M):
        out["clocks"][m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def parse_full(summary: str, logs: str, util: str) -> dict:
    """A full board build: the core clock's period and slack (-> fmax), the design's WNS / WHS,
    the collision and congestion warnings from the logs, and the utilization."""
    s = parse_summary(summary)
    res = parse_vivado_util(util) if util.strip() else {}
    core = next(((k, v) for k, v in s["clocks"].items() if k.startswith("core_clk")), None)
    res["wns_design"], res["whs"] = s["wns"], s["whs"]
    if core:
        period, slack = core[1]
        res.update(period=period, wns=slack, fmax=1000.0 / (period - slack),
                   logic_ns=period - slack, core_clock=core[0])
    else:
        res.update(period=None, wns=None, fmax=None, logic_ns=None)
    res["collisions"] = collisions(logs)
    res["congested"] = congested(logs)
    res["backend"] = "vivado-remote-full"
    return res


def worst_paths(timing: str, n: int = 30) -> str:
    """The first n paths of a `report_timing` report, one line each: slack, source ->
    destination (register names), clock of the destination."""
    out = []
    for blk in re.split(r"\n(?=Slack \()", timing):
        m = re.match(r"Slack \((\w+)\)\s*:\s*(-?[\d.]+)ns", blk)
        if not m:
            continue
        src = re.search(r"Source:\s+(\S+)", blk)
        dst = re.search(r"Destination:\s+(\S+)", blk)
        lv = re.search(r"Logic Levels:\s+(\d+)", blk)
        dp = re.search(r"Data Path Delay:\s+([\d.]+)ns\s+\(logic ([\d.]+)ns.*?route ([\d.]+)ns", blk)
        extra = (f" levels {lv.group(1)}" if lv else "") + (
            f", data {dp.group(1)} ns (logic {dp.group(2)}, route {dp.group(3)})" if dp else "")
        out.append(f"{m.group(2):>7} ns {m.group(1):<8} {src.group(1) if src else '?'} -> "
                   f"{dst.group(1) if dst else '?'}{extra}")
        if len(out) >= n:
            break
    return "\n".join(out)


def vivado(root: Path, top: str, sources: list[str], params: dict, out: Path,
           timeout: int = 4 * 3600, period_ns: float = 10.0) -> dict:
    """Out-of-context synth + place + route in Vivado (via the vivado-docker wrapper, which
    mounts the current directory: run from the worktree root)."""
    out.mkdir(parents=True, exist_ok=True)
    tcl = out / "ooc.tcl"
    tcl.write_text(vivado_tcl(root, top, sources, params, out, period_ns))
    log = out / "vivado.log"
    with log.open("w") as f:
        r = subprocess.run([VIVADO, "-mode", "batch", "-nojournal", "-nolog", "-source", str(tcl)],
                           cwd=root, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    txt = log.read_text()
    m = re.search(r"OTPU_WNS\s+(-?[\d.]+)", txt)
    if r.returncode != 0 or not m or not (out / "util.txt").exists():
        raise RuntimeError(f"vivado failed for {top}: {txt[-1500:]}")
    wns = float(m.group(1))
    res = parse_vivado_util((out / "util.txt").read_text())
    res["logic_ns"] = period_ns - wns           # post-route critical path (incl. routing)
    res["fmax"] = 1000.0 / (period_ns - wns)
    res["backend"] = "vivado"
    res["log"] = str(log)
    return res


def run(backend: str, root: Path, top: str, sources: list[str], params: dict, out: Path) -> dict:
    if backend == "yosys":
        return yosys(root, top, sources, params, out)
    if backend == "vivado":
        return vivado(root, top, sources, params, out)
    raise ValueError(f"unknown EVAL backend {backend!r}")
