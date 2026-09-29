#!/usr/bin/env python3
"""LiteDRAM controller + PHY on the YPCB-00338's DDR3 channel 0, for an area / timing run.

The real pins (litex-boards' ypcb_00338_1p1 platform), all 9 byte lanes (72 bits, no DM pins),
A7DDRPHY (the DDR3 banks are HR: no ODELAYE2), 1:4, sys = DDR clock / 4. No CPU: the PHY and
controller CSRs sit on a CSR bus behind JTAGBone, standing in for the host driving calibration
through BAR0. One native port carries a traffic load (LFSR write data, the read data XOR-folded
to a pin), so Vivado keeps the whole 576-bit datapath.

    python3 ooc_ypcb.py --sys-mhz 133.333 --out build_133                 # controller + PHY
    python3 ooc_ypcb.py --sys-mhz 133.333 --modules 8 --real --out real64  # + CDC ports, XDMA AXI
    python3 ooc_ypcb.py --sys-mhz 133.333 --ecc --out real72ecc            # + ECC frontends

Writes build/gateware/ld_eval.{v,xdc,tcl}; run the Tcl with Vivado on a build host (docs/litedram.md
section 4). Needs LiteX, LiteDRAM and litex-boards from git (see bench_native.py).
"""
import argparse
from functools import reduce
from operator import xor
import sys
from pathlib import Path

from migen import *
from litex.gen import LiteXModule
from litex.soc.cores.clock import S7PLL, S7MMCM, S7IDELAYCTRL
from litex.soc.interconnect.axi import AXIInterface

from litedram.common import LiteDRAMNativePort
from litedram.frontend.adapter import LiteDRAMNativePortCDC, LiteDRAMNativePortConverter
from litedram.frontend.axi import LiteDRAMAXI2Native
from litedram.frontend.ecc import LiteDRAMNativePortECC
from litex.soc.integration.soc_core import SoCCore
from litex.soc.integration.builder import Builder

from litedram.modules import MT41K256M8
from litedram.phy import s7ddrphy

from litex_boards.platforms import ypcb_00338_1p1 as ypcb  # litex-boards (git, 2025 and later)


class CRG(LiteXModule):
    def __init__(self, platform, f, real=False):
        self.rst = Signal()
        self.cd_sys = ClockDomain()
        self.cd_sys4x = ClockDomain()
        self.cd_sys4x_dqs = ClockDomain()
        self.cd_idelay = ClockDomain()
        self.pll = pll = S7PLL(speedgrade=-2)  # VCO 1600 MHz: 133.33, 533.33 and 200 MHz
        self.comb += pll.reset.eq(~platform.request("rst_n") | self.rst)
        pll.register_clkin(platform.request("clk200"), 200e6)
        pll.create_clkout(self.cd_sys, f)
        pll.create_clkout(self.cd_sys4x, 4 * f)
        pll.create_clkout(self.cd_sys4x_dqs, 4 * f, phase=90)
        pll.create_clkout(self.cd_idelay, 200e6)
        platform.add_false_path_constraints(self.cd_sys.clk, pll.clkin)
        self.idelayctrl = S7IDELAYCTRL(self.cd_idelay)
        if real:
            # stand-ins for the accelerator's core clock and XDMA's AXI clock (asynchronous to sys)
            self.cd_core = ClockDomain()
            self.cd_xdma = ClockDomain()
            self.mmcm = mmcm = S7MMCM(speedgrade=-2)
            clk200_1 = platform.request("clk200_1")
            platform.add_period_constraint(clk200_1, 5.0)
            mmcm.register_clkin(clk200_1, 200e6)
            mmcm.create_clkout(self.cd_core, 125e6)
            mmcm.create_clkout(self.cd_xdma, 100e6)
            platform.add_false_path_constraints(self.cd_sys.clk, self.cd_core.clk, self.cd_xdma.clk)


class Traffic(LiteXModule):
    """Keeps a native port busy: reads and writes from an LFSR, read data folded to one bit."""
    def __init__(self, port, led):
        lfsr = Signal(32, reset=1)
        self.sync += lfsr.eq(Cat(lfsr[1:], lfsr[0] ^ lfsr[2] ^ lfsr[6] ^ lfsr[7]))
        addr = Signal(len(port.cmd.addr))
        self.sync += If(port.cmd.valid & port.cmd.ready, addr.eq(addr + 1))
        w = len(port.wdata.data)
        acc = Signal(w)
        self.comb += [
            port.cmd.valid.eq(1),
            port.cmd.we.eq(lfsr[3] & lfsr[9]),
            port.cmd.addr.eq(addr ^ Cat(Replicate(0, 5), lfsr[16:24])),
            port.wdata.valid.eq(1),
            port.wdata.we.eq(2**len(port.wdata.we) - 1),
            port.wdata.data.eq(Replicate(lfsr, (w + 31) // 32)[:w]),
            port.rdata.ready.eq(1),
        ]
        self.sync += If(port.rdata.valid, acc.eq(acc ^ port.rdata.data))
        bits = [acc[i] for i in range(w)]
        while len(bits) > 1:  # balanced XOR tree (a linear chain overflows migen's recursion)
            bits = [bits[i] ^ bits[i + 1] if i + 1 < len(bits) else bits[i]
                    for i in range(0, len(bits), 2)]
        self.sync += led.eq(bits[0])


class AXITraffic(LiteXModule):
    """Keeps an AXI slave busy with 16-beat INCR bursts (LFSR data), read data folded to a pin."""
    def __init__(self, axi, led):
        lfsr = Signal(32, reset=1)
        self.sync += lfsr.eq(Cat(lfsr[1:], lfsr[0] ^ lfsr[2] ^ lfsr[6] ^ lfsr[7]))
        wa, ra, beat = Signal(20), Signal(20), Signal(4)
        self.sync += [
            If(axi.aw.valid & axi.aw.ready, wa.eq(wa + 1)),
            If(axi.ar.valid & axi.ar.ready, ra.eq(ra + 1)),
            If(axi.w.valid & axi.w.ready, beat.eq(beat + 1)),
        ]
        w = len(axi.r.data)
        acc = Signal(w)
        self.comb += [
            axi.aw.valid.eq(1), axi.aw.addr.eq(Cat(Replicate(0, 8), wa)), axi.aw.len.eq(15),
            axi.aw.size.eq(4), axi.aw.burst.eq(1),
            axi.w.valid.eq(1), axi.w.data.eq(Replicate(lfsr, w // 32)),
            axi.w.strb.eq(Cat(lfsr[0:4], Replicate(1, w // 8 - 4))), axi.w.last.eq(beat == 15),
            axi.b.ready.eq(1),
            axi.ar.valid.eq(1), axi.ar.addr.eq(Cat(Replicate(0, 8), ra)), axi.ar.len.eq(15),
            axi.ar.size.eq(4), axi.ar.burst.eq(1),
            axi.r.ready.eq(1),
        ]
        self.sync += If(axi.r.valid, acc.eq(acc ^ axi.r.data))
        bits = [acc[i] for i in range(w)]
        while len(bits) > 1:
            bits = [bits[i] ^ bits[i + 1] if i + 1 < len(bits) else bits[i]
                    for i in range(0, len(bits), 2)]
        self.sync += led.eq(bits[0])


class Top(SoCCore):
    def __init__(self, f, modules=9, real=False, ecc=False):
        platform = ypcb.Platform()
        self.crg = CRG(platform, f, real)
        SoCCore.__init__(self, platform, f, ident="litedram-eval", cpu_type=None,
                         integrated_rom_size=0, integrated_sram_size=0,
                         with_uart=False, with_timer=False, with_jtagbone=True)
        pads = platform.request("ddram", 0)
        if modules < 9:
            from litedram.common import PHYPadsReducer
            pads = PHYPadsReducer(pads, list(range(modules)))
        self.ddrphy = s7ddrphy.A7DDRPHY(pads, memtype="DDR3", nphases=4, sys_clk_freq=f)
        self.add_sdram("sdram", phy=self.ddrphy, module=MT41K256M8(f, "1:4"),
                       with_soc_interconnect=False)
        if not real:
            self.traffic = Traffic(self.sdram.crossbar.get_port(), platform.request("user_led", 0))
            return
        # What would replace the SmartConnect + MIG AXI front end on one channel: port B (the
        # DMA's chunks) and port A (the scale reads) as native ports in the core clock domain
        # (async FIFOs), and XDMA's 128-bit AXI master (host loads) through LiteDRAM's AXI
        # frontend with its read-modify-write for partial strobes (the board has no DM pins).
        xbar = self.sdram.crossbar
        def user_port(cd):
            if not ecc:
                return xbar.get_port(clock_domain=cd)
            xp = xbar.get_port()                                    # 576 bits, sys
            up = LiteDRAMNativePort("both", xp.address_width, 512, "sys")
            self.submodules += LiteDRAMNativePortECC(up, xp)
            cp = LiteDRAMNativePort("both", xp.address_width, 512, cd)
            self.submodules += LiteDRAMNativePortCDC(cp, up)
            return cp
        pb, pa = user_port("core"), user_port("core")
        self.tb = ClockDomainsRenamer("core")(Traffic(pb, platform.request("user_led", 0)))
        self.ta = ClockDomainsRenamer("core")(Traffic(pa, platform.request("user_led", 1)))
        px = user_port("xdma")
        p128 = LiteDRAMNativePort("both", px.address_width + 2, 128, "xdma")
        self.submodules += ClockDomainsRenamer("xdma")(LiteDRAMNativePortConverter(p128, px))
        axi = AXIInterface(data_width=128, address_width=32, id_width=4, clock_domain="xdma")
        self.submodules += ClockDomainsRenamer("xdma")(
            LiteDRAMAXI2Native(axi, p128, with_read_modify_write=True))
        self.tx = ClockDomainsRenamer("xdma")(AXITraffic(axi, platform.request("user_led", 2)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys-mhz", type=float, default=133.333)
    ap.add_argument("--modules", type=int, default=9)
    ap.add_argument("--out", default="build")
    ap.add_argument("--real", action="store_true",
                    help="two native ports in a core clock domain + XDMA's AXI port (with RMW)")
    ap.add_argument("--ecc", action="store_true", help="--real with LiteDRAM's ECC frontends")
    a = ap.parse_args()
    soc = Top(a.sys_mhz * 1e6, a.modules, a.real or a.ecc, a.ecc)
    Builder(soc, output_dir=a.out, compile_software=False, compile_gateware=False).build(
        build_name="ld_eval")
