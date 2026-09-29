#!/usr/bin/env python3
"""LiteDRAM controller efficiency on the native port (migen simulation, DFI stub PHY).

One DDR3 channel of the YPCB-00338 (MT41K256M8 x8, 8 data bytes: 64-bit), 1:4 rate, so one
native beat = one BL8 burst = 64 bytes = one sys cycle at the peak. The PHY is LiteDRAM's
behavioural model (fixed read/write latency, no I/O); the controller, bank machines, refresher
and crossbar are LiteDRAM's own RTL (2024.12), so the command scheduling is real.

The PHY is a DFI stub with the s7ddrphy read latency (no memory array: the timing is all in the
controller). MT41K256M8 geometry: 1024 columns, so one row is 8 KB of data per channel, as on
the card.

    python3 bench_native.py [--mhz 133.333] [--beats 8192] [--scen read,write,...] [--trfc160]

Setup (not part of the repo's environment): a Python 3.11 venv with migen, litex, litedram and
litex-boards from git (`uv pip install git+https://github.com/m-labs/migen
git+https://github.com/enjoy-digital/litex git+https://github.com/enjoy-digital/litedram
git+https://github.com/litex-hub/litex-boards`). migen's simulator (fhdl/simplify.py,
MemoryToArray) then fails on LiteX's write-only memory ports (read_capable=False, dat_r None):
skip the read statement when port.dat_r is None. Results: docs/litedram.md section 4.
"""
import argparse
import random
import time

from migen import *
from litex.gen.sim import run_simulation

from litedram.modules import MT41K256M8, _SpeedgradeTimings


class MT41K256M8_tRFC160(MT41K256M8):
    """The board's 2 Gb parts: tRFC 160 ns (LiteDRAM's MT41K256M8 uses 128 nCK, 240 ns at 1066;
    the MIG project uses 160 ns)."""
    speedgrade_timings = {"default": _SpeedgradeTimings(tRP=13.75, tRCD=13.75, tWR=15,
                                                        tRFC=(None, 160), tFAW=(None, 40), tRAS=35)}
from litedram.phy.model import get_sdram_phy_settings
from litedram.phy.dfi import Interface as DFIInterface
from litedram.core import LiteDRAMCore
from litedram.core.controller import ControllerSettings


class StubPHY(Module):
    """DFI sink with the s7ddrphy latencies: read data valid read_latency cycles after the
    read enable. The controller enforces the DRAM timings; the PHY only sets latencies."""
    def __init__(self, module, settings):
        self.settings = settings
        g = module.geom_settings
        self.dfi = DFIInterface(g.addressbits, g.bankbits, settings.nranks,
                                settings.dfi_databits, settings.nphases)
        for ph in self.dfi.phases:
            d = ph.rddata_en
            for _ in range(settings.read_latency):
                n = Signal()
                self.sync += n.eq(d)
                d = n
            self.comb += ph.rddata_valid.eq(d)


class DUT(Module):
    def __init__(self, clk_freq, nports=1, trfc160=False, **ctrl):
        module = (MT41K256M8_tRFC160 if trfc160 else MT41K256M8)(clk_freq, "1:4")
        phy_settings = get_sdram_phy_settings("DDR3", 64, clk_freq)
        self.submodules.phy = StubPHY(module, phy_settings)
        self.submodules.core = LiteDRAMCore(self.phy, module.geom_settings,
                                            module.timing_settings, clk_freq,
                                            controller_settings=ControllerSettings(**ctrl))
        self.ports = [self.core.crossbar.get_port() for _ in range(nports)]
        self.module = module


def driver(port, ops, stats, name):
    """ops: list of (we, addr). Issues commands back to back; writes feed wdata as the
    controller asks for it; counts read beats returned."""
    pending_w = [0]

    def cmd_gen():
        for we, addr in ops:
            yield port.cmd.valid.eq(1)
            yield port.cmd.we.eq(we)
            yield port.cmd.addr.eq(addr)
            yield
            while not (yield port.cmd.ready):
                yield
            if we:
                pending_w[0] += 1
        yield port.cmd.valid.eq(0)
        stats[name + "_cmd_done"] = True

    def wdata_gen():
        yield port.wdata.we.eq(2**len(port.wdata.we) - 1)
        n = 0
        while n < sum(1 for we, _ in ops if we):
            yield port.wdata.valid.eq(1)
            yield port.wdata.data.eq(n)
            yield
            if (yield port.wdata.ready):
                n += 1
        yield port.wdata.valid.eq(0)
        stats[name + "_w"] = n

    def rdata_gen():
        yield port.rdata.ready.eq(1)
        n, want = 0, sum(1 for we, _ in ops if not we)
        while n < want:
            yield
            if (yield port.rdata.valid):
                n += 1
        stats[name + "_r"] = n

    return [cmd_gen(), wdata_gen(), rdata_gen()]


def clock(stats, dut, total_beats):
    def gen():
        # wait for the refresher's first cycles to settle, then count until all data moved
        cyc = 0
        start = None
        while True:
            yield
            cyc += 1
            if start is None:
                for p in dut.ports:
                    if (yield p.cmd.valid):
                        start = cyc
            if all(k in stats for k in stats.get("_expect", [])):
                stats["cycles"] = cyc - (start or 0)
                return
    return gen()


def run(scen, mhz, beats, trfc160=False, **ctrl):
    clk = mhz * 1e6
    nports = 2 if scen.startswith("2p") else 1
    dut = DUT(clk, nports, trfc160, **ctrl)
    ops = []
    rng = random.Random(1)
    if scen == "read":
        ops = [[(0, a) for a in range(beats)]]
    elif scen == "write":
        ops = [[(1, a) for a in range(beats)]]
    elif scen.startswith("mix"):
        # 32-beat read runs; every K-th run a 32-beat write run elsewhere (ST / DSTEP style)
        k = int(scen[3:])
        o, ra, wa = [], 0, 1 << 20
        while len(o) < beats:
            for i in range(k):
                o += [(0, ra + j) for j in range(32)]; ra += 32
            o += [(1, wa + j) for j in range(32)]; wa += 32
        ops = [o[:beats]]
    elif scen.startswith("2pmix"):
        # port 0: sequential reads; port 1: 32-beat write runs spread through, 1 per K reads
        k = int(scen[5:])
        r = [(0, a) for a in range(beats)]
        w = [(1, (1 << 20) + a) for a in range(beats // (32 * k) * 32)]
        ops = [r, w]
    elif scen == "2prow":
        # one sequential read stream split over two ports by 8 KB row (128 beats): port 0 takes
        # the even rows, port 1 the odd ones, so one port opens the next bank while the other
        # still reads (a single port holds its bank until its queue drains: the crossbar lock)
        r0, r1 = [], []
        for a in range(beats):
            (r0 if (a // 128) % 2 == 0 else r1).append((0, a))
        ops = [r0, r1]
    elif scen == "rand32":
        # 32-beat reads at random 2 KB-aligned places (row misses)
        o = []
        while len(o) < beats:
            base = rng.randrange(0, 1 << 17) & ~31
            o += [(0, base + j) for j in range(32)]
        ops = [o[:beats]]
    stats = {}
    expect = []
    gens = []
    for i, (p, o) in enumerate(zip(dut.ports, ops)):
        nm = f"p{i}"
        gens += driver(p, o, stats, nm)
        if any(we for we, _ in o):
            expect.append(nm + "_w")
        if any(not we for we, _ in o):
            expect.append(nm + "_r")
    stats["_expect"] = expect
    gens.append(clock(stats, dut, beats))
    t = time.time()
    run_simulation(dut, gens)
    moved = sum(len(o) for o in ops)
    eff = moved / stats["cycles"]
    tim = dut.module.timing_settings
    return dict(scen=scen, mhz=mhz, beats=moved, cycles=stats["cycles"], eff=eff,
                gbs=eff * 64 * clk / 1e9, sim_s=time.time() - t,
                tREFI=tim.tREFI, tRFC=tim.tRFC, tRCD=tim.tRCD, tRP=tim.tRP)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhz", type=float, default=133.333)
    ap.add_argument("--beats", type=int, default=8192)
    ap.add_argument("--scen", default="read,write,mix8,2pmix8,rand32")
    ap.add_argument("--cmd-buffer-depth", type=int, default=8)
    ap.add_argument("--read-time", type=int, default=32)
    ap.add_argument("--write-time", type=int, default=16)
    ap.add_argument("--trfc160", action="store_true", help="tRFC 160 ns (the board's 2 Gb parts)")
    a = ap.parse_args()
    for s in a.scen.split(","):
        r = run(s, a.mhz, a.beats, a.trfc160, cmd_buffer_depth=a.cmd_buffer_depth,
                read_time=a.read_time, write_time=a.write_time)
        print("{scen:8s} {mhz:.3f} MHz  {beats:6d} beats  {cycles:7d} cycles  eff {eff:.3f}  "
              "{gbs:.2f} GB/s/channel  (tREFI {tREFI} tRFC {tRFC} tRCD {tRCD} tRP {tRP}; "
              "sim {sim_s:.0f} s)".format(**r), flush=True)
