"""Synthesize the board build's components (and the whole otpu_board) with yosys, in parallel,
and print area and an fmax estimate per component.

    python3 tools/synth/run.py [component ...]     (default: all)

fmax estimate: logic-only arrival from yosys `sta` (Xilinx cell delays, no routing), scaled for
routing and clocking: T = 1.6 * logic + 0.5 ns (clk-to-q + setup). This is a rough estimate for a
moderately full Kintex-7 -2; Vivado's post-route timing is the reference.
"""
from __future__ import annotations

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RTL = ROOT / "rtl"
OUT = ROOT / "build" / "synth_board"
BASE = ["vpu/otpu_fp.sv", "vpu/otpu_fpipe.sv", "top/otpu_pkg.sv"]
ALL = ["mem/otpu_tmem.sv", "mem/otpu_axi_dram.sv", "mem/otpu_actram.sv", "seq/otpu_seq.sv",
       "dma/otpu_dma.sv", "mxu/otpu_mxu.sv", "vpu/otpu_quant.sv", "vpu/otpu_vpu.sv",
       "top/otpu_coll.sv", "top/otpu_slice.sv", "boards/ypcb-00338/otpu_ctrl.sv",
       "boards/ypcb-00338/otpu_board.sv"]

# board parameters (rtl/boards/ypcb-00338/otpu_board.sv, opentpu.isasim.board_config)
COMPONENTS = {
    "otpu_seq": (["seq/otpu_seq.sv"], dict(IMEM_WORDS=32768, S=1, D=128, WIN=16)),
    "otpu_dma": (["dma/otpu_dma.sv"], dict(D=128, LANES=8)),
    "otpu_mxu": (["mxu/otpu_mxu.sv"], dict(D=128, MCOLS=2, DEPTH=128, LANES=8)),
    "otpu_quant": (["vpu/otpu_quant.sv"], dict(D=128, LANES=8)),
    "otpu_vpu": (["vpu/otpu_vpu.sv"], dict(LANES=8)),
    "otpu_tmem": (["mem/otpu_tmem.sv"], dict(WORDS=65536, LANES=8, NRP=8, NWP=4, WPB=1)),
    "otpu_actram": (["mem/otpu_actram.sv"], dict(D=128, MCOLS=2, BLOCKS=128, LANES=8)),
    "otpu_coll": (["top/otpu_coll.sv"], dict(S=1, LANES=8)),
    "otpu_axi_dram": (["mem/otpu_axi_dram.sv"], dict(D=128)),
    "otpu_ctrl": (["boards/ypcb-00338/otpu_ctrl.sv"], dict(D=128, MCOLS=2, LANES=8)),
    "otpu_board": (ALL, {}),
}


def synth(name: str) -> dict:
    files, params = COMPONENTS[name]
    srcs = [str(RTL / f) for f in BASE + [f for f in files if f not in BASE]]
    gp = " ".join(f"-G {k}={v}" for k, v in params.items())
    r = subprocess.run(["bash", str(ROOT / "tools/synth/sta.sh"), str(OUT), name, gp] + srcs,
                       capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    d = {"name": name, "raw": lines[-1] if lines else f"{name} failed"}
    d.update(re.findall(r"(\w+)=(\S+)", d["raw"]))
    return d


def main():
    names = sys.argv[1:] or list(COMPONENTS)
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(synth, names))
    print(f"{'component':<15}{'LUT':>8}{'FF':>8}{'DSP':>6}{'BRAM36':>8}{'LUTRAM':>8}"
          f"{'logic ns':>10}{'est fmax':>11}")
    for d in res:
        if not d.get("arrival", "?").isdigit():
            print(d["raw"])
            continue
        ns = int(d["arrival"]) / 1000
        b36 = int(d["bram36"]) + int(d["bram18"]) / 2
        f = 1000 / (1.6 * ns + 0.5)
        print(f"{d['name']:<15}{d['luts']:>8}{d['ff']:>8}{d['dsp']:>6}{b36:>8.1f}{d['lutram']:>8}"
              f"{ns:>10.2f}{f:>7.0f} MHz", flush=True)


if __name__ == "__main__":
    main()
