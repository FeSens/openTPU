#!/usr/bin/env python3
"""The host's H2C writes against the card's own traffic, co-simulated (docs/offload.md 13.19, 13.8,
15.3: the write contention, CT). A program of the card's runs on the RTL with the card's LiteDRAM
controllers (tools/moe_prefill_cosim.py's setup: DDR3-1066, the core at 133.33 MHz, the board's
configuration, every expert present), once alone and once beside a load on XDMA's side of each
channel's bridge (sim/verilator/otpu_ldc_mem.sv's +h2c_*: INCR bursts at a rate, optionally in
parts with idle gaps, into the channels' top, which no program uses). Its record: the run's cycles,
the bytes the controllers took from the load, the contention per GB written while the card computes
(CT, s a GB: the run's extra time over those bytes), and channel 0's cycle breakdown (+ldc_break:
data, refresh, multiplexer, rows, crossbar, idle) against the run alone, which splits the extra
time into the writes' own column commands (the bandwidth they take) and the rest (turnarounds,
rows, the crossbar).

    python3 tools/litedram/h2c_contention.py MODEL_DIR --work layer:1:2,layer:4:2,layer:1:1,em:131
        --rates 0.9,2,3 [--burst 256] [--on 524288 --off-us 0.29]

Works: layer:PREFIX:R, the layer-major prefill's run of the prefix's last layer on R rows
(compile_layer_run; R = 1 is a decode step's layer, as the card runs it in a prompt); em:ROWS,
expert-major's expert run of the first MoE layer on ROWS rows (compile_expert_run, the prompts'
144 experts of the 35B). Rates in GB/s over both channels; --on BYTES --off-us US: BYTES at the
rate, then US idle, as the server's idle parts (13.8: 512 KiB parts at about 2.2 GB/s and 0.29 ms
between them). One JSON line per run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import moe_prefill_cosim as C  # noqa: E402
from opentpu import rtlsim  # noqa: E402
from opentpu.compiler import arg_words  # noqa: E402
from opentpu.llm import moe as MO  # noqa: E402
from opentpu.llm.qwen3 import RunPos  # noqa: E402
from opentpu.profile import ddr3_plusargs  # noqa: E402

XCLK = 1.25                     # the simulation's XDMA clock against the core's (otpu_ldc_mem)
BRK = ("data", "ref", "mux", "rows", "xbar", "idle")


def work_setup(kind, M, spec, work: str, cfg, cap: int, pos: int):
    """(name, programs, run arguments, DRAM) of one work."""
    w = work.split(":")
    if w[0] == "layer":
        prefix, R = w[1], int(w[2])
        s, img = C.model_image(kind, M, spec, prefix, "moe", 1, cfg, cap, lookup=True)
        block = getattr(img, "block", 256)
        progs, ra = img.compile_layer_run(s.layers - 1, pos // block + 1, block, R=R,
                                          embedded=R > 1)
        dram = np.zeros(img.nbytes, np.uint8)
        C.timing_hack(img, dram)
        return progs, arg_words(ra, dict(RunPos.values(791, pos, 1, block), row=3)), dram
    if w[0] == "em":
        rows = int(w[1])
        s, img = C.model_image(kind, M, spec, "1", "moe", 1, cfg, cap)
        mo = s.moe
        ids = C.em_ids(rows, mo.k, mo.E, C.EM_USED[kind], 0)
        need = len(MO.em_slots(img.offload,
                               rows * MO.em_record(s.hidden, mo.k, MO.em_qd(img.cfg)))) \
            + C.em_passes(ids, mo.E)[1]
        n = max(mo.k, -(-need // len(img.offload.slots)))
        s, img = C.model_image(kind, M, spec, "1", "moe", 1, cfg, cap, experts=n)
        progs, ra = img.compile_expert_run(s.layers - 1, None)
        dram, _ = C.em_dram(img, s.layers - 1, ids, None)
        return progs, arg_words(ra, {"rows": rows, "entries": rows * mo.k}), dram
    raise SystemExit(f"--work {work}: layer:PREFIX:R or em:ROWS")


def h2c_plusargs(rate: float, burst: int, on: int, off_us: float) -> list[str]:
    """The load's simulator arguments: `rate` GB/s over both channels (each its half)."""
    if rate <= 0:
        return []
    xhz = XCLK * C.MHZ * 1e6
    plus = [f"+h2c_bpk={round(1000 * rate / 2 * 1e9 / xhz)}", f"+h2c_burst={burst}"]
    if on:
        plus += [f"+h2c_on={on // 2}", f"+h2c_off={round(off_us * 1e-6 * xhz)}"]
    return plus


def brk0(st) -> dict:
    """Channel 0's breakdown (controller cycles by kind)."""
    for line in st.get("brk", []):
        if line.startswith("BRK ch0"):
            return {k: int(re.search(rf"\b{k}=(\d+)", line).group(1)) for k in BRK}
    return {}


def run(cfg, progs, args, dram, plus: list[str]) -> dict:
    os.environ["OTPU_SIM_PLUSARGS"] = ",".join(["+ldc_break"] + plus)
    sim = rtlsim.ldc_plusargs(1066, C.MHZ) + ddr3_plusargs(3200 / 3, C.MHZ)[2:]
    t = time.time()
    _, _, st = rtlsim.run(cfg, progs, [dram.copy()], uarch=rtlsim.BOARD_UARCH, axi=True,
                          boot=True, stall=0, bw=100, lat=round(0.3 * C.MHZ),
                          max_cycles=1 << 40, ldc=1066, args=args,
                          plusargs=["+axi_dram=1", "+axi_map=1"] + sim)
    h = st.get("h2c") or []
    return dict(cycles=st["cycles"], ms=round(st["cycles"] / (C.MHZ * 1e3), 4),
                h2c_bytes=sum(x["bytes"] for x in h), h2c_bursts=sum(x["bursts"] for x in h),
                brk=brk0(st), sim_s=round(time.time() - t))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--work", default="layer:1:2,layer:4:2,layer:1:1,em:131")
    ap.add_argument("--rates", default="0.9,2,3", help="GB/s over both channels")
    ap.add_argument("--burst", type=int, default=256, help="bytes an AXI burst (64-4096)")
    ap.add_argument("--on", type=int, default=0, help="bytes a part (both channels; 0: steady)")
    ap.add_argument("--off-us", type=float, default=0.0, help="idle us after each part")
    ap.add_argument("--pos", type=int, default=256)
    ap.add_argument("--cap", type=int, default=1024)
    a = ap.parse_args(argv)
    kind, M, spec = C.load_model(a.model)
    cfg = C.cfg_of()
    for work in a.work.split(","):
        progs, args, dram = work_setup(kind, M, spec, work, cfg, a.cap, a.pos)
        base = run(cfg, progs, args, dram, [])
        print(json.dumps(dict(work=work, rate=0, **base)), flush=True)
        for rate in [float(x) for x in a.rates.split(",")]:
            r = run(cfg, progs, args, dram, h2c_plusargs(rate, a.burst, a.on, a.off_us))
            dt = (r["cycles"] - base["cycles"]) / (C.MHZ * 1e6)
            gb = r["h2c_bytes"] / 1e9
            d = {k: r["brk"].get(k, 0) - base["brk"].get(k, 0) for k in BRK}
            print(json.dumps(dict(
                work=work, rate=rate, burst=a.burst, on=a.on, off_us=a.off_us, **r,
                slower=round(r["cycles"] / base["cycles"] - 1, 4),
                ct_s_gb=round(dt / gb, 4) if gb else None,
                landed_gbps=round(gb / (r["cycles"] / (C.MHZ * 1e6)), 3),
                brk_delta=d,
                # the writes' own column commands on channel 0 (64 bytes each, half the bytes)
                write_beats_ch0=r["h2c_bytes"] // 2 // 64)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
