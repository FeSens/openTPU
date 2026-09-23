"""Component synthesis evaluators (EVAL=yosys | vivado).

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
    def row(name):
        m = re.search(rf"^\|\s*{re.escape(name)}\s*\|\s*([\d.]+)\s*\|", text, re.M)
        return float(m.group(1)) if m else 0.0
    return {"lut": row("LUT as Logic"), "lutram": row("LUT as Memory"),
            "ff": row("Slice Registers") or row("Register as Flip Flop"),
            "dsp": row("DSPs"), "bram36": row("RAMB36/FIFO*") or row("RAMB36/FIFO"),
            "bram18": row("RAMB18")}


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
