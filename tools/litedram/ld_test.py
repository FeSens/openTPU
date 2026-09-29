#!/usr/bin/env python3
"""LiteDRAM test image for the YPCB-00338: both DDR3 channels at their full 72 bits behind
LiteDRAM, calibrated by the host over PCIe (docs/litedram.md, "Test image").

    python3 ld_test.py --sys-mhz 133.333 --out build_ldtest2   # DDR3-1066, channels 0 and 1
    python3 ld_test.py --channels 0 --out build_ldtest         # channel 0 only (the first image)
    python3 ld_test.py --sys-mhz 100 --out build_ldtest800      # DDR3-800

Generates the gateware (Verilog, XDC, Vivado Tcl) and, for the host (tools/litedram/ld_host.py):
csr.csv and sdram_init.py (the DDR3 init sequence and PHY settings). Then run Vivado on a build
host in <out>/gateware: vivado -mode batch -source ld_test.tcl.

The design: XDMA (Gen1 x8, the production image's settings, device 10ee:7028 so the host's XDMA
driver binds, subsystem 4C44 so the openTPU tools do not take it for an openTPU image) whose
AXI-Lite master (BAR0, 1 MB) reaches a CPU-less LiteX SoC's CSR bus at BAR0 offset 0. The SoC,
per channel: A7DDRPHY on the channel's 9 byte lanes (the banks are HR: no ODELAY, so no write
leveling; write latency by bitslip, read by IDELAY taps and bitslip), LiteDRAM's controller
(MT41K256M8, 2 Gb parts, tRFC 160 ns as the MIG project), and a BIST generator and checker on the
full 576-bit port (memtest and bandwidth, started by the host). Each channel's DQS output clock
comes from an MMCM output with fine phase shift (1/56 of the VCO period per step) under host
control, so the host can scan each channel's write DQS phase: without write leveling this is the
write margin. One MMCM shifts all of its fine-phase outputs together, so channel 1's DQS clock
comes from a second MMCM (same input, same VCO; its static offset to the first does not matter,
the host scans a whole tCK). Channel 0 keeps the names of the one-channel image (ddrphy, sdram,
bist, phase); channel 1's CSRs are ddrphy1, sdram1, bist1, phase1. Also: the XADC (die
temperature and supplies: xadc_*) and the two I2C buses as production's I2C_CTRL / I2C_IN
(i2c_ctrl / i2c_in, the same bits), so the host reads the board's LM73 with opentpu.host.i2c.
XDMA's DMA master is answered by a stub (OKAY, zeros): the test image moves no data over DMA.

Clocks: the 50 MHz oscillator (AA28; the board's reset pin R28 is not wired, and the 200 MHz inputs
are not used by the production design) -> MMCM (integer, for the fine phase shift): sys, sys4x,
sys4x_dqs (channel 0); a second MMCM: sys4x1_dqs (channel 1; its DQ use sys4x, as channel 0's);
a PLL: the 200 MHz IDELAYCTRL reference.
"""
import argparse
import json
import sys
from pathlib import Path

from migen import *
from litex.gen import LiteXModule
from litex.build.generic_platform import Pins, IOStandard, Subsignal, Misc
from litex.soc.cores.clock import S7MMCM, S7PLL, S7IDELAYCTRL
from litex.soc.cores.xadc import S7SystemMonitor
from litex.soc.interconnect import wishbone
from litex.soc.interconnect.axi import (AXILiteInterface, AXILiteClockDomainCrossing,
                                        AXILite2Wishbone)
from litex.soc.interconnect.csr import CSRStatus, CSRStorage, AutoCSR
from litex.soc.integration.soc_core import SoCCore
from litex.soc.integration.builder import Builder

from litedram.modules import MT41K256M8, _SpeedgradeTimings
from litedram.phy import s7ddrphy
from litedram.init import get_sdram_phy_py_header

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ypcb_platform as ypcb     # litex-boards' ypcb_00338_1p1 platform (pins checked against ours)


class MT41K256M8_tRFC160(MT41K256M8):
    """The board's 2 Gb parts with tRFC 160 ns (LiteDRAM's MT41K256M8 table: 128 nCK)."""
    speedgrade_timings = {"default": _SpeedgradeTimings(tRP=13.75, tRCD=13.75, tWR=15,
                                                        tRFC=(None, 160), tFAW=(None, 40),
                                                        tRAS=35)}


class CRG(LiteXModule):
    """clk50: the 50 MHz clock after its BUFG (production's top shares it with the core clock's
    MMCM), or None: the pad, buffered here (the test image)."""
    def __init__(self, platform, f, dqs_phase, two=False, clk50=None):
        self.rst = Signal()
        self.cd_sys = ClockDomain()
        self.cd_sys4x = ClockDomain()
        self.cd_sys4x_dqs = ClockDomain()
        self.cd_idelay = ClockDomain()
        if clk50 is None:
            clk50_pad = platform.request("clk50")
            clk50 = Signal()
            self.specials += Instance("BUFG", i_I=clk50_pad, o_O=clk50)   # an MMCM and a PLL
        # MMCM, integer multiply / divide (fine phase shift needs it): 50 / 3 * 64 = 1066.67 MHz
        # VCO for DDR3-1066 (133.33 / 533.33), 50 * 16 = 800 for DDR3-800 (100 / 400)
        self.mmcm = mmcm = S7MMCM(speedgrade=-2, fractional=False)
        mmcm.register_clkin(clk50, 50e6)
        mmcm.create_clkout(self.cd_sys, f)                            # CLKOUT0
        mmcm.create_clkout(self.cd_sys4x, 4 * f)                      # CLKOUT1
        mmcm.create_clkout(self.cd_sys4x_dqs, 4 * f, phase=dqs_phase)  # CLKOUT2 (fine PS)
        mmcm.params["p_CLKOUT2_USE_FINE_PS"] = "TRUE"
        mmcm.expose_dps("sys", with_csr=False)          # driven by DQSPhase
        platform.add_false_path_constraints(self.cd_sys.clk, mmcm.clkin)
        if two:
            # channel 1: its DQ on sys4x (sys4x1 is the same clock under the name the PHY
            # derives its DQS clock's from), its DQS on the second MMCM's fine-phase output. The
            # first MMCM (so sys) is held in reset until the second has locked: the PHYs come out
            # of reset with every clock running.
            self.cd_sys4x1 = ClockDomain()
            self.cd_sys4x1_dqs = ClockDomain()
            self.comb += [self.cd_sys4x1.clk.eq(self.cd_sys4x.clk),
                          self.cd_sys4x1.rst.eq(self.cd_sys4x.rst)]
            self.mmcm1 = mmcm1 = S7MMCM(speedgrade=-2, fractional=False)
            self.comb += mmcm1.reset.eq(self.rst)
            mmcm1.register_clkin(clk50, 50e6)
            mmcm1.create_clkout(self.cd_sys4x1_dqs, 4 * f, phase=dqs_phase)   # CLKOUT0 (fine PS)
            mmcm1.params["p_CLKOUT0_USE_FINE_PS"] = "TRUE"
            mmcm1.expose_dps("sys", with_csr=False)
            self.comb += mmcm.reset.eq(self.rst | ~mmcm1.locked)
        else:
            self.comb += mmcm.reset.eq(self.rst)
        # IDELAYCTRL reference: a PLL of its own (200 MHz is no integer divide of 1066.67)
        self.pll = pll = S7PLL(speedgrade=-2)
        self.comb += pll.reset.eq(self.rst)
        pll.register_clkin(clk50, 50e6)
        pll.create_clkout(self.cd_idelay, 200e6)
        self.idelayctrl = S7IDELAYCTRL(self.cd_idelay)


class DQSPhase(LiteXModule):
    """The write DQS clock's phase: each write of dqs_shift moves it one fine step (1/56 of the
    VCO period: 16.7 ps at 1066.67 MHz) later (bit0 = 1) or earlier (0), once the MMCM has done
    the previous one (dqs_busy). dqs_steps: the steps since configuration (signed)."""
    def __init__(self, mmcm):
        self.dqs_shift = CSRStorage(1, description="Write: one step; 1 = later, 0 = earlier.")
        self.dqs_busy = CSRStatus(1)
        self.dqs_steps = CSRStatus(32)
        busy = Signal()
        steps = Signal((32, True))
        self.sync += [
            mmcm.psen.eq(0),
            If(self.dqs_shift.re & ~busy,
                mmcm.psen.eq(1), mmcm.psincdec.eq(self.dqs_shift.storage[0]), busy.eq(1),
                If(self.dqs_shift.storage[0], steps.eq(steps + 1)).Else(steps.eq(steps - 1)),
            ).Elif(mmcm.psdone, busy.eq(0)),
        ]
        self.comb += [self.dqs_busy.status.eq(busy), self.dqs_steps.status.eq(steps)]


XDMA_IN = ("awready", "wready", "bid", "bresp", "bvalid", "arready", "rid", "rdata", "rresp",
           "rlast", "rvalid")                       # XDMA's m_axi inputs


class AXIStub(Module):
    """Answers XDMA's DMA master: writes are taken and acknowledged (OKAY), reads return zeros."""
    def __init__(self, m):
        wbusy, rbusy = Signal(), Signal()
        bid, rid, rlen = Signal(4), Signal(4), Signal(8)
        self.sync += [
            If(m["awvalid"] & ~wbusy, wbusy.eq(1), bid.eq(m["awid"])),
            If(m["bvalid"] & m["bready"], wbusy.eq(0)),
            If(m["arvalid"] & ~rbusy, rbusy.eq(1), rid.eq(m["arid"]), rlen.eq(m["arlen"])),
            If(m["rvalid"] & m["rready"], rlen.eq(rlen - 1), If(rlen == 0, rbusy.eq(0))),
        ]
        wdone = Signal()
        self.sync += [
            If(m["awvalid"] & ~wbusy, wdone.eq(0)),
            If(m["wvalid"] & m["wready"] & m["wlast"], wdone.eq(1)),
            If(m["bvalid"] & m["bready"], wdone.eq(0)),
        ]
        self.comb += [
            m["awready"].eq(~wbusy), m["wready"].eq(wbusy & ~wdone),
            m["bvalid"].eq(wbusy & wdone), m["bid"].eq(bid), m["bresp"].eq(0),
            m["arready"].eq(~rbusy), m["rvalid"].eq(rbusy), m["rid"].eq(rid),
            m["rdata"].eq(0), m["rresp"].eq(0), m["rlast"].eq(rlen == 0),
        ]


class BIST(LiteXModule):
    """Memtest and bandwidth on a native port of any width (LiteDRAM's BIST wants a power-of-two
    width; this port is 576 bits). The host sets base / length (beats), mode and seed and pulses
    start; the engine issues `length` commands to consecutive beats, back to back, and counts the
    cycles from start to the last data (ticks). Write: the data of beat i is the generator's word
    i. Read: every returned beat is compared with the same word (reads return in order), and per
    module (byte lane) the beats with a wrong bit are counted and the wrong bits OR'ed.
    Data (mode bit 1): 0 = a 64-bit xorshift stream from the seed, one step per beat, spread over
    the 72-bit groups rotated per group (every lane sees random data); 1 = the beat address and its
    complement XOR the seed in every group (finds aliased address bits)."""
    def __init__(self, port, modules=9):
        dw, aw = len(port.wdata.data), len(port.cmd.addr)
        ngroups = dw // (8 * modules)                 # 72-bit groups (DQ beats): 8
        self.start = CSRStorage(1, description="Write 1: start (mode / base / length as set).")
        self.mode = CSRStorage(2, description="bit0: 1 = read and check, 0 = write; bit1: data.")
        self.base = CSRStorage(aw)
        self.length = CSRStorage(aw + 1)
        self.seed = CSRStorage(64, reset=0x0123456789ABCDEF)
        self.done = CSRStatus(1)
        self.ticks = CSRStatus(32)
        self.beats = CSRStatus(aw + 1, description="Data beats seen (written or read).")
        self.errors = CSRStatus(32, description="Read beats with any wrong bit.")
        self.lane_errors = [CSRStatus(32, name=f"lane{m}_errors") for m in range(modules)]
        self.lane_bits = CSRStatus(8 * modules, description="Wrong bits seen, OR'ed per lane (bit 8m+b).")
        for m, c in enumerate(self.lane_errors):
            setattr(self, f"lane{m}_errors", c)

        read = self.mode.storage[0]
        addrmode = self.mode.storage[1]
        running, ncmd, ndat, ticks = Signal(), Signal(aw + 1), Signal(aw + 1), Signal(32)
        cmd_addr = Signal(aw)
        # data generator, stepped per data beat
        st = Signal(64)
        def xorshift(x):
            a = x ^ (x << 13)[:64]
            b = a ^ (a >> 7)
            return b ^ (b << 17)[:64]
        dat_addr = Signal(aw)
        exp = Signal(dw)
        groups = []
        for g in range(ngroups):
            gw = 8 * modules
            rot = (st << (7 * g % 64)) | (st >> (64 - 7 * g % 64)) if g else st
            rnd = Cat(rot[:64], rot[:gw - 64] ^ rot[64 - (gw - 64):64])
            pat = Cat(dat_addr, ~dat_addr, dat_addr)[:gw] ^ Cat(self.seed.storage, self.seed.storage)[:gw]
            groups.append(Mux(addrmode, pat, rnd))
        self.comb += exp.eq(Cat(*groups))
        wd_hs = port.wdata.valid & port.wdata.ready
        rd_hs = port.rdata.valid & port.rdata.ready
        step = Mux(read, rd_hs, wd_hs)
        self.comb += [
            port.cmd.valid.eq(running & (ncmd != self.length.storage)),
            port.cmd.we.eq(~read),
            port.cmd.addr.eq(cmd_addr),
            port.wdata.valid.eq(running & ~read & (ndat != ncmd)),
            port.wdata.we.eq(2**len(port.wdata.we) - 1),
            port.wdata.data.eq(exp),
            port.rdata.ready.eq(1),
            self.done.status.eq(~running),
            self.ticks.status.eq(ticks),
            self.beats.status.eq(ndat),
        ]
        start = Signal()
        self.comb += start.eq(self.start.re & self.start.storage[0])
        lane_err = Signal(modules)
        lane_bits = Signal(8 * modules)
        for m in range(modules):
            bits = Cat(*[(port.rdata.data ^ exp)[g * 8 * modules + 8 * m: g * 8 * modules + 8 * m + 8]
                         for g in range(ngroups)])
            self.comb += lane_err[m].eq(bits != 0)
            self.comb += lane_bits[8 * m:8 * m + 8].eq(
                reduce_or8(bits, ngroups))
        self.sync += [
            If(start,
                running.eq(1), ncmd.eq(0), ndat.eq(0), ticks.eq(0),
                cmd_addr.eq(self.base.storage), dat_addr.eq(self.base.storage),
                st.eq(self.seed.storage),
                self.errors.status.eq(0), self.lane_bits.status.eq(0),
                *[c.status.eq(0) for c in self.lane_errors],
            ).Elif(running,
                ticks.eq(ticks + 1),
                If(port.cmd.valid & port.cmd.ready, ncmd.eq(ncmd + 1), cmd_addr.eq(cmd_addr + 1)),
                If(step,
                    ndat.eq(ndat + 1), dat_addr.eq(dat_addr + 1), st.eq(xorshift(st)),
                    If(ndat + 1 == self.length.storage, running.eq(0)),
                ),
                If(rd_hs & read,
                    If(lane_err != 0, self.errors.status.eq(self.errors.status + 1)),
                    self.lane_bits.status.eq(self.lane_bits.status | lane_bits),
                    *[If(lane_err[m], c.status.eq(c.status + 1)) for m, c in enumerate(self.lane_errors)],
                ),
            ),
        ]


def reduce_or8(bits, n):
    """OR of n consecutive bytes of `bits`."""
    r = bits[0:8]
    for i in range(1, n):
        r = r | bits[8 * i:8 * i + 8]
    return r


class I2CPins(LiteXModule):
    """The card's two I2C buses as production's I2C_CTRL / I2C_IN (otpu_ctrl, opentpu/host/i2c.py):
    i2c_ctrl bit 2b drives bus b's SCL low, bit 2b+1 its SDA (1 = low, 0 = released); i2c_in reads
    the levels the same way, plus the LM73's ALERT (active low) at bit 4."""
    def __init__(self, platform):
        self.ctrl = CSRStorage(4)                        # i2c_ctrl
        self.levels = CSRStatus(5, name="in")            # i2c_in
        pads = [platform.request(n) for n in ("lm73_scl", "lm73_sda", "smb_scl", "smb_sda")]
        lvl = Signal(4)
        for i, p in enumerate(pads):
            self.specials += Instance("IOBUF", io_IO=p, i_I=0, i_T=~self.ctrl.storage[i],
                                      o_O=lvl[i])
        s0, s1 = Signal(5), Signal(5)
        self.sync += [s0.eq(Cat(lvl, platform.request("lm73_alert_n"))), s1.eq(s0)]
        self.comb += self.levels.status.eq(s1)


# as constraints/otpu_top.xdc
I2C_IO = [(n, 0, Pins(p), IOStandard("LVCMOS18"), Misc("PULLUP=TRUE"),
           *([Misc("DRIVE=4"), Misc("SLEW=SLOW")] if n != "lm73_alert_n" else []))
          for n, p in [("lm73_scl", "N24"), ("lm73_sda", "N25"), ("lm73_alert_n", "P25"),
                       ("smb_scl", "R26"), ("smb_sda", "R27")]]


class LDTest(SoCCore):
    mem_map = {"csr": 0x0000_0000}          # CSRs at BAR0 offset 0

    def __init__(self, f, dqs_phase=90, xdma_tcl=None, channels=(0, 1)):
        platform = ypcb.Platform()
        platform.add_extension(I2C_IO)
        self.crg = CRG(platform, f, dqs_phase, two=1 in channels)
        SoCCore.__init__(self, platform, f, ident="openTPU LiteDRAM test image", cpu_type=None,
                         integrated_rom_size=0, integrated_sram_size=0, with_uart=False,
                         with_timer=False, csr_data_width=32)
        self.channels = channels
        self.xadc = S7SystemMonitor()
        self.i2c = I2CPins(platform)

        # ---- the DDR3 channels, 72 bits each
        # CL / CWL as the MIG project (LiteDRAM's table would take 533.33 MHz for DDR3-1333's bin)
        cl, cwl = (7, 6) if f > 101e6 else (6, 5)
        for ch in channels:
            sfx = "" if ch == 0 else str(ch)
            phy = s7ddrphy.A7DDRPHY(platform.request("ddram", ch), memtype="DDR3", nphases=4,
                                    sys_clk_freq=f, iodelay_clk_freq=200e6, cl=cl, cwl=cwl,
                                    write_latency_calibration=True, ddr_clk="sys4x" + sfx)
            setattr(self, "ddrphy" + sfx, phy)
            self.add_sdram("sdram" + sfx, phy=phy, module=MT41K256M8_tRFC160(f, "1:4"),
                           with_soc_interconnect=False)
            setattr(self, "bist" + sfx, BIST(getattr(self, "sdram" + sfx).crossbar.get_port(), modules=9))
            setattr(self, "phase" + sfx, DQSPhase(self.crg.mmcm if ch == 0 else self.crg.mmcm1))

        # ---- PCIe: XDMA as an RTL IP; BAR0 (AXI-Lite, axi_aclk) -> CSR bus (sys)
        self.cd_xdma = ClockDomain()
        pcie = platform.request("pcie_x8")
        refclk = Signal()
        self.specials += Instance("IBUFDS_GTE2", i_CEB=0, i_I=pcie.clk_p, i_IB=pcie.clk_n,
                                  o_O=refclk)
        platform.add_period_constraint(pcie.clk_p, 10.0)
        axil_x = AXILiteInterface(data_width=32, address_width=32, clock_domain="xdma")
        axil_s = AXILiteInterface(data_width=32, address_width=32, clock_domain="sys")
        m = {n: Signal(w, name="xdma_m_axi_" + n) for n, w in [
            ("awid", 4), ("awaddr", 64), ("awlen", 8), ("awsize", 3), ("awburst", 2),
            ("awprot", 3), ("awvalid", 1), ("awready", 1), ("awlock", 1), ("awcache", 4),
            ("wdata", 128), ("wstrb", 16), ("wlast", 1), ("wvalid", 1), ("wready", 1),
            ("bid", 4), ("bresp", 2), ("bvalid", 1), ("bready", 1),
            ("arid", 4), ("araddr", 64), ("arlen", 8), ("arsize", 3), ("arburst", 2),
            ("arprot", 3), ("arvalid", 1), ("arready", 1), ("arlock", 1), ("arcache", 4),
            ("rid", 4), ("rdata", 128), ("rresp", 2), ("rlast", 1), ("rvalid", 1),
            ("rready", 1)]}
        axi_aresetn, lnk_up = Signal(), Signal()
        unused = [Signal(n) for n in (1, 1, 3, 32, 3, 3, 1)]
        self.specials += Instance("xdma_0",
            i_sys_clk=refclk, i_sys_rst_n=pcie.rst_n,
            o_user_lnk_up=lnk_up,
            o_pci_exp_txp=pcie.tx_p, o_pci_exp_txn=pcie.tx_n,
            i_pci_exp_rxp=pcie.rx_p, i_pci_exp_rxn=pcie.rx_n,
            o_axi_aclk=ClockSignal("xdma"), o_axi_aresetn=axi_aresetn,
            i_usr_irq_req=0, o_usr_irq_ack=unused[0], o_msi_enable=unused[1],
            o_msi_vector_width=unused[2],
            **{("i_m_axi_" if n in XDMA_IN else "o_m_axi_") + n: s for n, s in m.items()},
            o_m_axil_awaddr=axil_x.aw.addr, o_m_axil_awprot=unused[4],
            o_m_axil_awvalid=axil_x.aw.valid, i_m_axil_awready=axil_x.aw.ready,
            o_m_axil_wdata=axil_x.w.data, o_m_axil_wstrb=axil_x.w.strb,
            o_m_axil_wvalid=axil_x.w.valid, i_m_axil_wready=axil_x.w.ready,
            i_m_axil_bvalid=axil_x.b.valid, i_m_axil_bresp=axil_x.b.resp,
            o_m_axil_bready=axil_x.b.ready,
            o_m_axil_araddr=axil_x.ar.addr, o_m_axil_arprot=unused[5],
            o_m_axil_arvalid=axil_x.ar.valid, i_m_axil_arready=axil_x.ar.ready,
            i_m_axil_rdata=axil_x.r.data, i_m_axil_rresp=axil_x.r.resp,
            i_m_axil_rvalid=axil_x.r.valid, o_m_axil_rready=axil_x.r.ready,
            i_cfg_mgmt_addr=0, i_cfg_mgmt_write=0, i_cfg_mgmt_write_data=0,
            i_cfg_mgmt_byte_enable=0, i_cfg_mgmt_read=0, o_cfg_mgmt_read_data=unused[3],
            o_cfg_mgmt_read_write_done=unused[6], i_cfg_mgmt_type1_cfg_reg_access=0)
        self.comb += ResetSignal("xdma").eq(~axi_aresetn)
        self.submodules += ClockDomainsRenamer("xdma")(AXIStub(m))
        self.axil_cdc = AXILiteClockDomainCrossing(axil_x, axil_s, cd_from="xdma", cd_to="sys")
        # BAR0 offsets map one to one onto the CSR bus (1 MB window; the CSRs use the first 64 KB)
        wb = wishbone.Interface(data_width=32, address_width=32, addressing="word")
        self.axil2wb = AXILite2Wishbone(axil_s, wb)
        self.bus.add_master(name="pcie", master=wb)

        # link LED: green = PCIe link up, yellow = sys MMCM locked, red = blink (sys alive)
        blink = Signal(27)
        self.sync += blink.eq(blink + 1)
        self.comb += [platform.request("user_led", 0).eq(blink[26]),
                      platform.request("user_led", 1).eq(lnk_up),
                      platform.request("user_led", 2).eq(self.crg.mmcm.locked)]

        # XDMA IP (created in the Vivado run) and the GT channel LOCs of the production design
        # (constraints/otpu_top.xdc), in an XDC read LATE so they override the IP's own
        # XDMA is created and synthesized with the design. After synthesis, before placement,
        # its GT channels move to the card's lanes (lane i on GTXE2_CHANNEL_X0Y(23 - i), as
        # constraints/otpu_top.xdc: the IP's own XDC puts them one quad lower) -- all eight
        # cleared first, since a lane's new site may be another lane's old one -- and the BAR0
        # crossing's clocks are declared asynchronous (the AXI-Lite CDC's FIFOs cross them).
        gt = "[get_cells -hier -filter {NAME =~ *pipe_lane[%d].gt_wrapper_i/gtx_channel.gtxe2_channel_i}]"
        pre_synth = [
            f"source {{{xdma_tcl}}}",
            "file mkdir ip",
            "otpu_xdma_ip [pwd]/ip",
            "generate_target all [get_ips xdma_0]",
            "synth_ip [get_ips xdma_0] -force",     # flagged in project mode, but links the IP
        ]
        pre_place = ["reset_property LOC " + gt % i for i in range(8)] + \
            ["set_property LOC GTXE2_CHANNEL_X0Y%d " % (23 - i) + gt % i for i in range(8)] + [
            "set_clock_groups -asynchronous -group [get_clocks -of_objects [get_pins xdma_0/axi_aclk]]"
            " -group [get_clocks -of_objects [get_nets sys_clk]]",
            "report_property [lindex " + gt % 0 + " 0] LOC",
        ]
        # (LiteX formats these commands: braces doubled)
        esc = lambda cs: [c.replace("{", "{{").replace("}", "}}") for c in cs]
        platform.toolchain.pre_synthesis_commands += esc(pre_synth)
        platform.toolchain.pre_placement_commands += esc(pre_place)
        platform.add_platform_command("set_property PULLUP true [get_ports {{pcie_x8_rst_n}}]")
        platform.add_platform_command("set_false_path -from [get_ports {{pcie_x8_rst_n}}]")
        # configuration as the production image (BPI x16 flash, 1.8 V configuration banks)
        for cmd in ["set_property CFGBVS GND [current_design]",
                    "set_property CONFIG_VOLTAGE 1.8 [current_design]",
                    "set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]",
                    "set_property BITSTREAM.CONFIG.UNUSEDPIN PULLNONE [current_design]",
                    "set_property CONFIG_MODE BPI16 [current_design]"]:
            platform.add_platform_command(cmd)


def reset_value(x):
    """A PHY setting that is a CSR-backed Signal: its reset value."""
    return int(x.reset.value) if isinstance(x, Signal) else int(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sys-mhz", type=float, default=133.333)
    ap.add_argument("--dqs-phase", type=float, default=90.0,
                    help="static sys4x_dqs phase, degrees (the host shifts it from there)")
    ap.add_argument("--channels", default="0,1", help="DDR3 channels: 0,1 (default), 0 or 1")
    ap.add_argument("--out", default="build_ldtest2")
    a = ap.parse_args()
    channels = tuple(int(c) for c in a.channels.split(","))
    out = Path(a.out).resolve()
    (out / "gateware").mkdir(parents=True, exist_ok=True)
    xdma_tcl = out / "gateware" / "xdma_ip.tcl"
    xdma_tcl.write_text((HERE / "xdma_ip.tcl").read_text())
    soc = LDTest(a.sys_mhz * 1e6, a.dqs_phase, xdma_tcl="xdma_ip.tcl", channels=channels)
    b = Builder(soc, output_dir=str(out), compile_software=False, compile_gateware=False,
                csr_csv=str(out / "csr.csv"))
    b.build(build_name="ld_test", vivado_place_directive="Explore",
            vivado_post_place_phys_opt_directive="AggressiveExplore",
            vivado_route_directive="Explore",
            vivado_post_route_phys_opt_directive="AggressiveExplore")
    # the host's calibration inputs
    ps = getattr(soc, "ddrphy" + ("" if channels[0] == 0 else str(channels[0]))).settings
    ctl = getattr(soc, "sdram" + ("" if channels[0] == 0 else str(channels[0])))
    hdr = get_sdram_phy_py_header(ps, ctl.controller.settings.timing)
    info = {"sys_hz": a.sys_mhz * 1e6, "nphases": ps.nphases, "rdphase": reset_value(ps.rdphase),
            "wrphase": reset_value(ps.wrphase), "databits": ps.databits, "dfi_databits": ps.dfi_databits,
            "modules": ps.databits // 8, "delays": 32, "bitslips": 8, "cl": ps.cl, "cwl": ps.cwl,
            "read_latency": ps.read_latency, "write_latency": ps.write_latency,
            "vco_hz": soc.crg.mmcm.compute_config()["vco"], "dqs_phase": a.dqs_phase,
            "channels": list(channels)}
    if 1 in channels:       # one fine step is 1/56 of the VCO period: both MMCMs must agree
        assert soc.crg.mmcm1.compute_config()["vco"] == info["vco_hz"], "MMCM VCOs differ"
    (out / "sdram_init.py").write_text(hdr + "\nphy = " + json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
