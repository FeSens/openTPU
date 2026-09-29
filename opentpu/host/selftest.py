"""otpu-selftest: staged bring-up of the openTPU card (or of its Verilator model with --sim).

    otpu-selftest                         # the card, /dev/xdma0
    otpu-selftest --sim                   # the board model (no hardware)
    otpu-selftest --model qwen3           # also a model-level check (lfm2, qwen35 or a directory)

Stages stop at the first failure, with a hint. Each builds on the previous one:
  1 link       the control registers answer (ID register)
  2 config     the bitstream's VERSION (D / MCOLS / LANES) gives the host configuration
               (opentpu.host.board.device_config); OTPU_MCOLS / OTPU_LANES, when set, must agree
  3 calib      both DDR3 controllers report calibration done (a LiteDRAM bitstream's, CAPS
               bit27, calibrated by the host first: memcal, as Board does when it opens)
     scrub      zeros over the whole DRAM once per configuration (Board.scrub): with ECC, a read
               of a beat never written since power-up hangs
  4 regs       SCRATCH register write / read
  5 addr       walking address bits and random patterns on each channel (raw channel
               addresses: bottom, middle, top)
  6 pattern    random data through the 64-byte channel interleave, unaligned edges, the top
               of DRAM (logical addresses near 4 GiB); sub-beat host updates on each channel
               (merged into whole beats on the host: XdmaTransport.mem_write)
  7 bandwidth  host <-> card DMA rate
  8 kernel     a program using every unit, and one of partial DRAM writes from the
               accelerator (QST bytes, short stores), compared with the ISA simulator bit for bit
  9 vops       RDOT / OUTER / LOG2 (the VPU functions of Qwen3.5's DeltaNet layers) against
               the ISA simulator; a bitstream without them passes with a note, unless --model
               names Qwen3.5. Then otpu-diag's reduction and RDOT / OUTER / LOG2 programs
               (opchecks.py groups vpu-reduce, vpu-new) back to back, each result read right
               after its program: a bitstream that has the functions must pass them all (a
               125 MHz build computed the vops program right but returned the previous
               program's RDOT result)
 10 stream     (a bitstream with the stream engine, CAPS bit26) STREAM in every mode of the
               board's subset and a DSTEP against the ISA simulator
 11 model      (with --model) greedy decoding of Qwen3, LFM2 or Qwen3.5 on the card equals the
               ISA simulator, token for token, and the answer to "What is the capital of France?"
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

from opentpu.host import memcal
from opentpu.host.runstate import busy_exits
from opentpu.host.board import (CH_BYTES, ID_OTPU, R_ID, R_SCRATCH, R_STATUS, ST_CALIB0,
                                ST_CALIB1, Board, SimTransport, XdmaTransport, device_config)
from opentpu.host.checks import (address_lines, bandwidth, channel_patterns, masked_program,
                                 model_check, partial_writes, pattern_test, run_demo,
                                 stream_check, vops_check)

HINTS = {
    "link": "Is the card enumerated (lspci -d 10ee:), the XDMA driver loaded (lsmod | grep "
            "xdma) and /dev/xdma0_user present? Did the host reboot after programming the "
            "bitstream (or rescan the PCIe bus)? See docs/host.md.",
    "config": "The host follows the bitstream's VERSION register: unset OTPU_MCOLS / "
              "OTPU_LANES, or load the bitstream built for them (docs/board.md, which bitstream "
              "to load). D must be 128.",
    "calib": "A DDR3 controller did not calibrate. MIG bitstreams: check the MIG pinout / "
             "clocking in the bitstream (docs/board.md) and the memory voltage. LiteDRAM "
             "bitstreams (CAPS bit27): the host calibrates them (otpu-memcal cal --force "
             "redoes it, otpu-memcal shows the last result; docs/litedram.md). STATUS bit5 = "
             "channel 0, bit6 = channel 1.",
    "scrub": "Writing the DRAM failed or timed out: DMA host->card is broken (dmesg: XDMA "
             "errors); try the driver in poll mode (sudo otpu-setup --poll).",
    "regs": "Register writes do not stick: the AXI-Lite path (XDMA BAR0 -> otpu_ctrl) is "
            "broken, or the core clock / reset is not running.",
    "addr": "An address line of that channel is stuck or aliased: DDR3 pinout / MIG address "
            "width, or the interconnect address map (channel 1 must be at 0x8000_0000).",
    "pattern": "Data errors: if only one channel fails, suspect its byte lanes / calibration; "
               "if errors follow the 64-byte interleave, suspect the host mapping "
               "(opentpu/host/board.py) "
               "against rtl/mem/otpu_axi_dram.sv.",
    "bandwidth": "DMA is slow or failed: check the PCIe link width and speed (lspci -vv, "
                 "LnkSta should be 2.5GT/s x8).",
    "kernel": "The accelerator computed something different from the ISA simulator: run the "
              "same program on the RTL model (tests/test_board.py) and compare the counters.",
    "vops": "On register map 3 or later the VPU computes RDOT / OUTER / LOG2 wrong: run "
            "otpu-diag (the vpu-new group) and the same programs on the board model "
            "(tests/test_board.py). Older maps: Qwen3.5 needs them, which bitstreams built "
            "before commit ddec900 "
            "lack (they run the program but compute other values): load a bitstream with them "
            "(docs/board.md, which bitstream to load), or run Qwen3 / LFM2. If the vops "
            "program passes and only the op checks fail, the bitstream itself misbehaves "
            "(e.g. a core clock without enough timing margin): load a slower build and run "
            "otpu-diag.",
    "stream": "The stream engine computed a STREAM or DSTEP differently from the ISA "
              "simulator: run tests/test_stream.py -k rtl and the stream stage on the board "
              "model (tests/test_board.py::test_stream_on_board_model).",
    "model": "Kernels pass but the model differs: compare per-token logits against "
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


@busy_exits
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="otpu-selftest", description=__doc__.split("\n")[0])
    ap.add_argument("--sim", action="store_true", help="the Verilator board model")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--model",
                    help="qwen3, lfm2, qwen35 or a checkpoint directory: the model stage")
    ap.add_argument("--tokens", type=int, default=8, help="tokens to generate in the model stage")
    ap.add_argument("--wformat", default="int8", choices=["int8", "fp4", "int4"],
                    help="weight format of the model stage's layers")
    ap.add_argument("--head-format", default=None, choices=["int8", "fp4", "int4"],
                    help="weight format of the model stage's LM head (default: --wformat)")
    ap.add_argument("--bw-mib", type=int, default=512, help="bandwidth test size (MiB)")
    a = ap.parse_args(argv)

    if a.sim:
        ch_bytes = 1 << 22
        t = SimTransport(ch_bytes=ch_bytes)
    else:
        ch_bytes = CH_BYTES
        t = XdmaTransport(a.dev)
    cfg = None                                            # from the bitstream (stage config)
    print(f"openTPU self-test on {'the board model' if a.sim else a.dev}")
    r = Runner()
    board = Board(t, check=False)
    top = 2 * ch_bytes                                    # logical DRAM size

    def link():
        v = t.reg_read(R_ID)
        return v == ID_OTPU, f"ID {v:#010x}" + ("" if v == ID_OTPU else f", want {ID_OTPU:#x}")

    def config():
        nonlocal cfg
        i = board.info()
        cfg = device_config(i, DRAM_BYTES=2 * ch_bytes)
        msg = f"D={i['D']} MCOLS={i['MCOLS']} LANES={i['LANES']}, register map {i['regmap']}"
        if i["core_khz"]:
            msg += f", core {i['core_khz'] / 1e3:g} MHz"
        if i["ddr_mts"]:
            msg += f", DDR3-{i['ddr_mts']}"
        if i["build_id"] is not None:
            msg += f", build {i['build_id']:08x}"
        return True, msg

    def calib():
        # a host-calibrated bitstream (LiteDRAM): the channels not calibrated since configuration
        # first (the Board above opens without check, so it has not; a ddrcal.CalError fails
        # the stage)
        how = ""
        if memcal.hostcal(t):
            t0 = time.time()
            res = memcal.ensure(t, log=lambda m: print(f"         {m}", flush=True))
            how = (f" (host calibration: channel {', '.join(map(str, res['channels']))} in "
                   f"{time.time() - t0:.1f}s)" if res else " (host calibration: done before)")
        deadline = time.time() + (0 if a.sim else 5)
        while True:
            st = t.reg_read(R_STATUS)
            c0, c1 = bool(st & ST_CALIB0), bool(st & ST_CALIB1)
            if (c0 and c1) or time.time() > deadline:
                return c0 and c1, f"channel 0 {'ok' if c0 else 'NOT calibrated'}, " \
                                  f"channel 1 {'ok' if c1 else 'NOT calibrated'}{how}"
            time.sleep(0.1)

    def scrub():
        t0 = time.time()
        if board.scrub():
            return True, f"4 GiB written in {time.time() - t0:.1f}s (ECC check bits initialized)"
        return True, "already done since configuration" if not a.sim else "no ECC on the model"

    def regs():
        vals = [0x0, 0xFFFFFFFF, 0xA5A5_5A5A, 0x1234_5678]
        got, was = [], t.reg_read(R_SCRATCH)
        for v in vals:
            t.reg_write(R_SCRATCH, v)
            got.append(t.reg_read(R_SCRATCH))
        t.reg_write(R_SCRATCH, was)                       # Board.scrub's marker
        return got == vals, "SCRATCH " + ("ok" if got == vals else f"read {got}")

    def addr():
        msgs = []
        for c in (0, 1):
            for check in (address_lines, channel_patterns):
                ok, m = check(t, c, ch_bytes)
                if not ok:
                    return False, m
            msgs.append(f"channel {c} ok")
        return True, "; ".join(msgs)

    def pattern():
        m = 1 << 20                                       # disjoint regions
        regions = [(0, m), (m + 12345, 1000), (2 * m + 60, 70), (top // 2 - 4096, 8192),
                   (top - 2 * m, m), (top - 100, 100)]
        ok, msg = pattern_test(board, regions)
        for c in (0, 1):
            if ok:
                ok, m2 = partial_writes(t, c)
                msg += "; " + m2
        return ok, msg

    def bw():
        if a.sim:
            return True, "skipped on the model"
        w, rd = bandwidth(t, a.bw_mib << 20)
        ok = w > 0.5 and rd > 0.5
        return ok, f"host->card {w:.2f} GB/s, card->host {rd:.2f} GB/s"

    def kernel():
        ok, msg, st = run_demo(board, cfg)
        if not ok:
            return ok, "all units: " + msg
        ok2, msg2, st2 = run_demo(board, cfg, masked_program())
        return ok2, (f"all units: {msg} (b_reads={st['b_reads']}, a_writes={st['a_writes']}); "
                     f"partial writes: {msg2} (b_writes={st2['b_writes']}, "
                     f"a_writes={st2['a_writes']})")

    def vops():
        from opentpu.llm import load_spec, model_dir
        qwen35 = bool(a.model) and \
            type(load_spec(model_dir(a.model))).__module__.endswith(".qwen35")
        return vops_check(board, cfg, need=qwen35)

    def stream():
        return stream_check(board, cfg)

    def model():
        return model_check(t, cfg, a.model, a.tokens, a.sim, a.wformat, a.head_format)

    r.stage("link", link)
    r.stage("config", config)
    r.stage("calib", calib)
    r.stage("scrub", scrub)
    r.stage("regs", regs)
    r.stage("addr", addr)
    r.stage("pattern", pattern)
    r.stage("bandwidth", bw)
    r.stage("kernel", kernel)
    r.stage("vops", vops)
    r.stage("stream", stream)
    if a.model:
        r.stage("model", model)
    print("ALL PASS" if not r.failed else f"stopped at stage '{r.failed}'")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
