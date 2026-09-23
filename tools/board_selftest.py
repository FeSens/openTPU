"""Staged bring-up of the openTPU card (or of its Verilator model with --sim).

    python3 tools/board_selftest.py                  # the card, /dev/xdma0
    python3 tools/board_selftest.py --sim            # the board model (no hardware)
    python3 tools/board_selftest.py --qwen models/Qwen3-0.6B   # also chat-level check

Stages stop at the first failure, with a hint. Each builds on the previous one:
  1 link       the control registers answer (ID register)
  2 config     the bitstream's D / MCOLS / LANES match opentpu.isasim.board_config()
  3 calib      both DDR3 controllers report calibration done
  4 regs       SCRATCH register write / read
  5 addr       walking address bits on each channel (raw channel addresses)
  6 pattern    random data through the 64-byte channel interleave, unaligned edges, the top
               of each channel (logical addresses near 4 GiB)
  7 bandwidth  host <-> card DMA rate
  8 kernel     a program using every unit, DRAM compared with the ISA simulator bit for bit
  9 qwen       (with --qwen) greedy decoding on the card equals the ISA simulator, token for
               token, and the answer to "What is the capital of France?"
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from host.board import (CH_BYTES, ID_OTPU, R_ID, R_SCRATCH, R_STATUS, ST_CALIB0,  # noqa: E402
                        ST_CALIB1, Board, BoardBackend, SimTransport, XdmaTransport, sim_config)
from host.checks import address_lines, bandwidth, pattern_test, run_demo  # noqa: E402
from opentpu.isasim import board_config  # noqa: E402

HINTS = {
    "link": "Is the card enumerated (lspci -d 10ee:), the XDMA driver loaded (lsmod | grep "
            "xdma) and /dev/xdma0_user present? Did the host reboot after programming the "
            "bitstream (or rescan the PCIe bus)? See docs/host.md.",
    "config": "The bitstream was built with other parameters than board_config(): rebuild, or "
              "run with a matching opentpu configuration.",
    "calib": "A DDR3 controller did not calibrate: check the MIG pinout / clocking in the "
             "bitstream (docs/board.md) and the memory voltage; STATUS bit5 = channel 0, "
             "bit6 = channel 1.",
    "regs": "Register writes do not stick: the AXI-Lite path (XDMA BAR0 -> otpu_ctrl) is "
            "broken, or the core clock / reset is not running.",
    "addr": "An address line of that channel is stuck or aliased: DDR3 pinout / MIG address "
            "width, or the interconnect address map (channel 1 must be at 0x8000_0000).",
    "pattern": "Data errors: if only one channel fails, suspect its byte lanes / calibration; "
               "if errors follow the 64-byte interleave, suspect the host mapping (host/board.py) "
               "against rtl/mem/otpu_axi_dram.sv.",
    "bandwidth": "DMA is slow or failed: check the PCIe link width and speed (lspci -vv, "
                 "LnkSta should be 5GT/s x8).",
    "kernel": "The accelerator computed something different from the ISA simulator: run the "
              "same program on the RTL model (tests/test_board.py) and compare the counters.",
    "qwen": "Kernels pass but the model differs: compare per-token logits against "
            "IsaBackend with opentpu.llm.qwen3.Engine; check that the image fits the DRAM.",
}


class Runner:
    def __init__(self):
        self.failed = None

    def stage(self, name: str, fn):
        if self.failed:
            return None
        t = time.time()
        try:
            ok, msg = fn()
        except Exception as e:                            # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
            traceback.print_exc(limit=3)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<10} {msg}  ({time.time() - t:.1f}s)",
              flush=True)
        if not ok:
            self.failed = name
            print(f"         hint: {HINTS[name]}")
        return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", action="store_true", help="the Verilator board model")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--qwen", help="model directory for the Qwen3 stage")
    ap.add_argument("--tokens", type=int, default=8, help="Qwen3 tokens to generate")
    ap.add_argument("--bw-mib", type=int, default=512, help="bandwidth test size (MiB)")
    a = ap.parse_args()

    if a.sim:
        ch_bytes = 1 << 22
        t = SimTransport(ch_bytes=ch_bytes)
        cfg = board_config(DRAM_BYTES=2 * ch_bytes)
    else:
        ch_bytes = CH_BYTES
        t = XdmaTransport(a.dev)
        cfg = board_config()
    print(f"openTPU self-test on {'the board model' if a.sim else a.dev}")
    r = Runner()
    board = Board(t, check=False)
    top = 2 * ch_bytes                                    # logical DRAM size

    def link():
        v = t.reg_read(R_ID)
        return v == ID_OTPU, f"ID {v:#010x}" + ("" if v == ID_OTPU else f", want {ID_OTPU:#x}")

    def config():
        i = board.info()
        want = (cfg.D, cfg.MCOLS, cfg.LANES)
        got = (i["D"], i["MCOLS"], i["LANES"])
        return got == want, f"D={got[0]} MCOLS={got[1]} LANES={got[2]}"

    def calib():
        deadline = time.time() + (0 if a.sim else 5)
        while True:
            st = t.reg_read(R_STATUS)
            c0, c1 = bool(st & ST_CALIB0), bool(st & ST_CALIB1)
            if (c0 and c1) or time.time() > deadline:
                return c0 and c1, f"channel 0 {'ok' if c0 else 'NOT calibrated'}, " \
                                  f"channel 1 {'ok' if c1 else 'NOT calibrated'}"
            time.sleep(0.1)

    def regs():
        vals = [0x0, 0xFFFFFFFF, 0xA5A5_5A5A, 0x1234_5678]
        got = []
        for v in vals:
            t.reg_write(R_SCRATCH, v)
            got.append(t.reg_read(R_SCRATCH))
        return got == vals, "SCRATCH " + ("ok" if got == vals else f"read {got}")

    def addr():
        msgs = []
        for c in (0, 1):
            ok, m = address_lines(t, c, ch_bytes)
            if not ok:
                return False, m
            msgs.append(m)
        return True, "; ".join(msgs)

    def pattern():
        m = 1 << 20                                       # disjoint regions
        regions = [(0, m), (m + 12345, 1000), (2 * m + 60, 70), (top // 2 - 4096, 8192),
                   (top - 2 * m, m), (top - 100, 100)]
        return pattern_test(board, regions)

    def bw():
        if a.sim:
            return True, "skipped on the model"
        w, rd = bandwidth(t, a.bw_mib << 20)
        ok = w > 0.5 and rd > 0.5
        return ok, f"host->card {w:.2f} GB/s, card->host {rd:.2f} GB/s"

    def kernel():
        ok, msg, st = run_demo(board, cfg)
        return ok, msg + f" (b_reads={st['b_reads']}, a_writes={st['a_writes']})"

    def qwen():
        from opentpu.llm.qwen3 import Engine, Spec, load_weights
        from transformers import AutoTokenizer
        spec = Spec.from_hf(a.qwen)
        W = load_weights(a.qwen)
        tok = AutoTokenizer.from_pretrained(a.qwen)
        msgs = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                      tokenize=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        cap = 256
        rcfg = sim_config(spec, cap)                      # same layout, DRAM sized to the model
        dev = Engine(spec, W, cap=cap, cfg=rcfg if a.sim else cfg,
                     backend=lambda c, imgs: BoardBackend(c, imgs, transport=t))
        ref = Engine(spec, W, cap=cap, cfg=rcfg)
        t0 = time.time()
        got = dev.generate(ids, max_new=a.tokens)
        dt = time.time() - t0
        want = ref.generate(ids, max_new=a.tokens)
        text = tok.decode(got, skip_special_tokens=True)
        cyc = np.mean([s["cycles"] for s in dev.stats])
        ok = got == want
        return ok, (f"{text!r}; {len(dev.stats)} tokens, {cyc / 1e6:.2f} Mcycles/token, "
                    f"{len(dev.stats) / dt:.2f} tok/s wall" +
                    ("" if ok else f"; ISA simulator says {tok.decode(want)!r}"))

    r.stage("link", link)
    r.stage("config", config)
    r.stage("calib", calib)
    r.stage("regs", regs)
    r.stage("addr", addr)
    r.stage("pattern", pattern)
    r.stage("bandwidth", bw)
    r.stage("kernel", kernel)
    if a.qwen:
        r.stage("qwen", qwen)
    print("ALL PASS" if not r.failed else f"stopped at stage '{r.failed}'")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
