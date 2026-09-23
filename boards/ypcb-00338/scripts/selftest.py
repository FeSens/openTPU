#!/usr/bin/env python3
"""Bring-up self-test of the openTPU card (run on the Linux host with the XDMA driver loaded).

  python3 boards/ypcb-00338/scripts/selftest.py [--dev /dev/xdma0] [--sim]

Steps (each stops the test on failure, with a hint):
  1. control registers: ID, version (D / MCOLS / LANES), scratch register write/read-back
  2. DDR3 calibration of both channels (STATUS.CALIB0/1)
  3. DRAM through PCIe: random patterns on each channel directly, then through the logical
     64-byte interleave (host/board.py), at several offsets including the top of each channel
  4. the accelerator: a small LD / VOP / ST program, loaded from DRAM into IMEM and run,
     against the bit-exact ISA simulator (exercises the loader, both DRAM ports and TMEM)
--sim runs steps 1, 3 and 4 against the Verilator board model instead of the card.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from host.board import (Board, SimTransport, XdmaTransport, CH_BYTES, R_SCRATCH,  # noqa: E402
                        ST_CALIB0, ST_CALIB1, R_STATUS)
from opentpu import isa as I  # noqa: E402
from opentpu.isasim import Machine, board_config  # noqa: E402


def step(msg):
    print(f"-- {msg}", flush=True)


def fail(msg, hint):
    print(f"FAIL: {msg}\n  hint: {hint}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--sim", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    t = SimTransport(ch_bytes=1 << 22) if a.sim else XdmaTransport(a.dev)
    ch_bytes = len(t.ch[0]) if a.sim else CH_BYTES

    step("1. control registers")
    b = Board(t)
    info = b.info()
    print("   ", info)
    t.reg_write(R_SCRATCH, 0xA5A5_1234)
    if not a.sim and t.reg_read(R_SCRATCH) != 0xA5A5_1234:
        fail("scratch register did not read back", "AXI-Lite path (XDMA BAR0 -> sc_ctl -> core)")

    if not a.sim:
        step("2. DDR3 calibration")
        for _ in range(50):
            st = t.reg_read(R_STATUS)
            if st & ST_CALIB0 and st & ST_CALIB1:
                break
            time.sleep(0.1)
        else:
            fail(f"STATUS {st:#x}: calibration incomplete (CALIB0={bool(st & ST_CALIB0)} "
                 f"CALIB1={bool(st & ST_CALIB1)})",
                 "MIG pinout / VREF / termination, see docs/board.md 'DDR3 calibration'")

    step("3. DRAM through PCIe")
    for ch in (0, 1):
        for off in (0, 4096, ch_bytes // 2, ch_bytes - 65536):
            d = rng.integers(0, 256, 65536, dtype=np.uint8)
            t.mem_write(ch, off, d)
            if not np.array_equal(t.mem_read(ch, off, len(d)), d):
                fail(f"channel {ch} offset {off:#x}: read-back mismatch",
                     "MIG / SmartConnect address map (MIG1 at 0x8000_0000)")
    for addr in (0, 64, 4096 + 8, 1 << 20):
        d = rng.integers(0, 256, 100_003, dtype=np.uint8)
        b.write(addr, d)
        if not np.array_equal(b.read(addr, len(d)), d):
            fail(f"logical address {addr:#x}: interleaved read-back mismatch", "host/board.py map")
    print("    ok")

    step("4. accelerator vs the ISA simulator")
    cfg = board_config(DRAM_BYTES=2 * ch_bytes)
    n = 1000
    x = rng.standard_normal(n).astype(np.float32)
    prog = [I.ld(0, 0, n), I.vop(I.V_MUL, 2048, 0, 0, 1, n, n, n, n, I.B_SCALAR, 1.5),
            I.vop(I.V_EXP2, 4096, 2048, 0, 1, n, n, n, n, I.B_SCALAR, 0.0),
            I.st(1 << 16, 4096, n), I.halt()]
    img = np.zeros(1 << 17, np.uint8)
    img[:4 * n] = x.view(np.uint8)
    b.write(0, img)
    m = Machine(cfg, [prog], [np.concatenate([img, np.zeros(cfg.DRAM_BYTES - len(img),
                                                            np.uint8)])]).run()
    want = m.slices[0].dram[1 << 16:(1 << 16) + 4 * n]
    b.load_program(1 << 20, np.asarray(I.assemble(prog), np.uint32))
    stats = b.run(timeout=10)
    got = b.read(1 << 16, 4 * n)
    if not np.array_equal(got, want):
        bad = np.nonzero(got.view(np.uint32) != want.view(np.uint32))[0]
        fail(f"{len(bad)} of {n} results differ (first at {bad[0]})",
             "run the same program on the RTL simulator (host SimTransport) to localize")
    print(f"    ok: {stats}")
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
