#!/usr/bin/env python3
"""The production LiteDRAM core for the YPCB-00338: both DDR3 channels (72 bits, A7DDRPHY at
DDR3-1066) with one native user port each, generated as one Verilog module, otpu_litedram, for
the board top (docs/litedram.md, section 7). It replaces the two MIGs.

    python3 gen_core.py --out build_core          # -> build_core/otpu_litedram.v, .xdc, csr.csv,
                                                  #    sdram_init.py, ports.txt

The module (every port is a plain Verilog port; the DDR3 pads are the only I/O):

    clk50g              in   the 50 MHz clock after its BUFG (shared with the core clock's MMCM)
    rst                 in   asynchronous, high: the clocks and both channels
    sys_clk, sys_rst    out  the controller clock (DDR clock / 4: 133.33 MHz) and its reset: the
                             native ports' clock (otpu_mem_ch's controller side)
    ctl_clk, ctl_rst    in   the CSR port's clock and reset (XDMA's axi_aclk / BAR0)
    ctl_*               AXI-Lite slave, 16-bit byte address, 32-bit data: the CSRs (calibration,
                        ECC counters, the ready bits), csr.csv offsets; crosses into sys_clk
    c0_* / c1_*         each channel's native port in sys_clk, 512-bit data through LiteDRAM's
                        ECC (SECDED per 64-bit word): cmd_valid/ready/we/addr[24:0] (64-byte beat
                        index in the channel), wdata_valid/ready/data[511:0]/we[63:0],
                        rdata_valid/ready/data[511:0]. Write data is taken when the controller
                        wants it (wdata_ready), whatever wdata_valid says, so a write command may
                        go only with its data already valid; partial beats are an error (no DM
                        pins, and ECC words are whole): the user side reads, merges and writes
                        whole beats. Reads return in command order; rdata_ready is ignored
                        (there is no backpressure).
    c0_ready, c1_ready  out  the channel is calibrated: set by the host (cal_ready, cal1_ready) at
                        the end of opentpu.host.ddrcal.calibrate_channel, cleared by rst

Per channel, as the test image (tools/litedram/ld_test.py, whose CRG, BIST and DQS phase control
this reuses): the DQS clock on an MMCM output with fine phase shift, driven by the host
(phase*_dqs_*), and a BIST on a second crossbar port (bist*_*) for calibration's traffic check.
Channel 0's CSR names are bare (ddrphy_, sdram_, bist_, phase_, ecc_, cal_), channel 1's carry
a 1. The ECC counts corrected and uncorrectable words (ecc_sec_errors, ecc_ded_errors).

Clocks: clk50 -> MMCM (sys, sys4x, channel 0's DQS), a second MMCM (channel 1's DQS), a PLL (the
200 MHz IDELAYCTRL reference). The XDC has the DDR3 pins and I/O standards (LiteX's platform,
checked against the MIG project's pins; the top's DDR3 ports keep the names ddram0_*, ddram1_*),
the banks' internal VREF and LiteX's reset-synchronizer paths; the top adds the clock relations
(ctl_clk and sys_clk are asynchronous: the CSR port crosses them through FIFOs).
"""
import argparse
import json
import sys
from pathlib import Path

from migen import *
from litex.build.generic_platform import Pins, Subsignal
from litex.soc.interconnect import wishbone
from litex.soc.interconnect.axi import (AXILiteInterface, AXILiteClockDomainCrossing,
                                        AXILite2Wishbone)
from litex.soc.interconnect.csr import AutoCSR, CSRStorage
from litex.soc.integration.soc_core import SoCCore
from litex.soc.integration.builder import Builder

from litedram.common import LiteDRAMNativePort
from litedram.frontend.ecc import LiteDRAMNativePortECC
from litedram.phy import s7ddrphy
from litedram.init import get_sdram_phy_py_header

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ypcb_platform as ypcb                                    # noqa: E402
from ld_test import (CRG, WLCRG, DQSPhase, BIST, MT41K256M8_tRFC160, WriteClocks,  # noqa: E402
                     reset_value)
from wl7ddrphy import WL7DDRPHY                                 # noqa: E402

AW, DW = 25, 512            # beat address, user data (+ 64 ECC bits on the DRAM side)
USER = [("clk50g", 0, Pins(1)), ("rst", 0, Pins(1)), ("sys_clk", 0, Pins(1)),
        ("sys_rst", 0, Pins(1)), ("ctl_clk", 0, Pins(1)), ("ctl_rst", 0, Pins(1)),
        ("ctl", 0,
         Subsignal("awvalid", Pins(1)), Subsignal("awready", Pins(1)), Subsignal("awaddr", Pins(16)),
         Subsignal("wvalid", Pins(1)), Subsignal("wready", Pins(1)), Subsignal("wdata", Pins(32)),
         Subsignal("wstrb", Pins(4)), Subsignal("bvalid", Pins(1)), Subsignal("bready", Pins(1)),
         Subsignal("bresp", Pins(2)), Subsignal("arvalid", Pins(1)), Subsignal("arready", Pins(1)),
         Subsignal("araddr", Pins(16)), Subsignal("rvalid", Pins(1)), Subsignal("rready", Pins(1)),
         Subsignal("rdata", Pins(32)), Subsignal("rresp", Pins(2)))]
for _c in (0, 1):
    USER += [(f"c{_c}", 0,
              Subsignal("cmd_valid", Pins(1)), Subsignal("cmd_ready", Pins(1)),
              Subsignal("cmd_we", Pins(1)), Subsignal("cmd_addr", Pins(AW)),
              Subsignal("wdata_valid", Pins(1)), Subsignal("wdata_ready", Pins(1)),
              Subsignal("wdata_data", Pins(DW)), Subsignal("wdata_we", Pins(DW // 8)),
              Subsignal("rdata_valid", Pins(1)), Subsignal("rdata_ready", Pins(1)),
              Subsignal("rdata_data", Pins(DW)), Subsignal("ready", Pins(1)))]
USER_NAMES = {u[0] for u in USER}


class Cal(Module, AutoCSR):
    """cal_ready / cal1_ready: the host sets it once the channel is calibrated (the accelerator
    waits for it)."""
    def __init__(self):
        self.ready = CSRStorage(1, description="1: calibrated, the controller has the PHY.")


class OTPULiteDRAM(SoCCore):
    mem_map = {"csr": 0x0000_0000}

    def __init__(self, f=133.333e6, dqs_phase=90, bist=True, phy="a7", groups=None):
        """phy "wl": WL7DDRPHY with each channel's WriteClocks MMCM (docs/litedram.md section 8;
        groups[ch]: its lanes' write clock groups)."""
        platform = ypcb.Platform()
        platform.add_extension(USER)
        clk50 = platform.request("clk50g")
        self.crg = CRG(platform, f, dqs_phase, two=True, clk50=clk50) if phy == "a7" else \
            WLCRG(platform, f, clk50=clk50)
        self.comb += self.crg.rst.eq(platform.request("rst"))
        SoCCore.__init__(self, platform, f, ident="openTPU LiteDRAM", cpu_type=None,
                         integrated_rom_size=0, integrated_sram_size=0, with_uart=False,
                         with_timer=False, csr_data_width=32, with_ctrl=True)
        self.comb += [platform.request("sys_clk").eq(ClockSignal("sys")),
                      platform.request("sys_rst").eq(ResetSignal("sys"))]

        cl, cwl = (7, 6) if f > 101e6 else (6, 5)
        for ch in (0, 1):
            sfx = "" if ch == 0 else str(ch)
            if phy == "wl":
                wc = WriteClocks(ch, f)
                setattr(self, "wclk" + sfx, wc)
                p = ClockDomainsRenamer(wc.domains)(WL7DDRPHY(
                    platform.request("ddram", ch), groups=groups[ch], sys_clk_freq=f,
                    iodelay_clk_freq=200e6, cl=cl, cwl=cwl))
                for c in wc.constraints(hier=True):
                    platform.add_platform_command(c.replace("{", "{{").replace("}", "}}"))
            else:
                p = s7ddrphy.A7DDRPHY(platform.request("ddram", ch), memtype="DDR3", nphases=4,
                                      sys_clk_freq=f, iodelay_clk_freq=200e6, cl=cl, cwl=cwl,
                                      write_latency_calibration=True, ddr_clk="sys4x" + sfx)
            setattr(self, "ddrphy" + sfx, p)
            self.add_sdram("sdram" + sfx, phy=p, module=MT41K256M8_tRFC160(f, "1:4"),
                           with_soc_interconnect=False)
            core = getattr(self, "sdram" + sfx)
            setattr(self, "phase" + sfx, DQSPhase(getattr(self, "wclk" + sfx) if phy == "wl" else
                                                  self.crg.mmcm if ch == 0 else self.crg.mmcm1))
            # the user port: 512 bits through the ECC (576 on the crossbar)
            raw = core.crossbar.get_port()
            assert raw.address_width == AW and raw.data_width == DW + 64, (raw.address_width, raw.data_width)
            user = LiteDRAMNativePort("both", AW, DW)
            ecc = LiteDRAMNativePortECC(user, raw, with_error_injection=False)
            setattr(self, "ecc" + sfx, ecc)
            pads = platform.request(f"c{ch}")
            self.comb += [
                user.cmd.valid.eq(pads.cmd_valid), pads.cmd_ready.eq(user.cmd.ready),
                user.cmd.we.eq(pads.cmd_we), user.cmd.addr.eq(pads.cmd_addr),
                user.cmd.last.eq(1),
                user.wdata.valid.eq(pads.wdata_valid), pads.wdata_ready.eq(user.wdata.ready),
                user.wdata.data.eq(pads.wdata_data), user.wdata.we.eq(pads.wdata_we),
                pads.rdata_valid.eq(user.rdata.valid), user.rdata.ready.eq(pads.rdata_ready),
                pads.rdata_data.eq(user.rdata.data),
            ]
            cal = Cal()
            setattr(self, "cal" + sfx, cal)
            self.comb += pads.ready.eq(cal.ready.storage)
            if bist:
                setattr(self, "bist" + sfx, BIST(core.crossbar.get_port(), modules=9))

        # CSRs: AXI-Lite (ctl_clk) -> sys -> the CSR bus, BAR0 offsets relative to the window
        self.cd_ctl = ClockDomain()
        self.comb += [self.cd_ctl.clk.eq(platform.request("ctl_clk")),
                      self.cd_ctl.rst.eq(platform.request("ctl_rst"))]
        axil_c = AXILiteInterface(data_width=32, address_width=32, clock_domain="ctl")
        axil_s = AXILiteInterface(data_width=32, address_width=32, clock_domain="sys")
        p = platform.request("ctl")
        self.comb += [
            axil_c.aw.valid.eq(p.awvalid), p.awready.eq(axil_c.aw.ready), axil_c.aw.addr.eq(p.awaddr),
            axil_c.w.valid.eq(p.wvalid), p.wready.eq(axil_c.w.ready), axil_c.w.data.eq(p.wdata),
            axil_c.w.strb.eq(p.wstrb),
            p.bvalid.eq(axil_c.b.valid), axil_c.b.ready.eq(p.bready), p.bresp.eq(axil_c.b.resp),
            axil_c.ar.valid.eq(p.arvalid), p.arready.eq(axil_c.ar.ready), axil_c.ar.addr.eq(p.araddr),
            p.rvalid.eq(axil_c.r.valid), axil_c.r.ready.eq(p.rready), p.rdata.eq(axil_c.r.data),
            p.rresp.eq(axil_c.r.resp),
        ]
        self.axil_cdc = AXILiteClockDomainCrossing(axil_c, axil_s, cd_from="ctl", cd_to="sys")
        wb = wishbone.Interface(data_width=32, address_width=32, addressing="word")
        self.axil2wb = AXILite2Wishbone(axil_s, wb)
        self.bus.add_master(name="ctl", master=wb)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sys-mhz", type=float, default=133.333)
    ap.add_argument("--no-bist", action="store_true", help="no BIST (DFII-only calibration)")
    ap.add_argument("--out", default="build_core")
    ap.add_argument("--phy", default="a7", choices=["a7", "wl"],
                    help="a7: A7DDRPHY; wl: WL7DDRPHY, write leveling by clock groups (section 8)")
    ap.add_argument("--groups0", default="0,0,0,0,1,1,1,1,1", help="wl: channel 0's lane groups")
    ap.add_argument("--groups1", default="0,0,1,0,1,1,1,1,0", help="wl: channel 1's lane groups")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    groups = {0: [int(x) for x in a.groups0.split(",")], 1: [int(x) for x in a.groups1.split(",")]}
    soc = OTPULiteDRAM(a.sys_mhz * 1e6, bist=not a.no_bist, phy=a.phy, groups=groups)
    b = Builder(soc, output_dir=str(out), compile_software=False, compile_gateware=False,
                csr_csv=str(out / "csr.csv"))
    b.build(build_name="otpu_litedram", run=False)
    gw = out / "gateware"
    # the XDC: the DDR3 pads' blocks and the platform commands, not the user ports (no pins)
    keep, block = [], []
    for line in (gw / "otpu_litedram.xdc").read_text().splitlines():
        if line.startswith("# ") and ":" in line:
            block = [line]
            name = line[2:].split(":")[0]
            skip = name in USER_NAMES
            continue
        if block and not line.strip():
            if not skip:
                keep += block + [""]
            block = []
            continue
        if block:
            block.append(line)
        elif "get_nets sys_clk" not in line:     # the CRG's own, by a net name the top renames
            keep.append(line)
    (out / "otpu_litedram.xdc").write_text("\n".join(keep) + "\n")
    (out / "otpu_litedram.v").write_text((gw / "otpu_litedram.v").read_text())
    ps = soc.ddrphy.settings
    hdr = get_sdram_phy_py_header(ps, soc.sdram.controller.settings.timing)
    info = {"sys_hz": a.sys_mhz * 1e6, "nphases": ps.nphases, "rdphase": reset_value(ps.rdphase),
            "wrphase": reset_value(ps.wrphase), "databits": ps.databits,
            "dfi_databits": ps.dfi_databits, "modules": ps.databits // 8, "delays": 32,
            "bitslips": 8, "cl": ps.cl, "cwl": ps.cwl, "read_latency": ps.read_latency,
            "write_latency": ps.write_latency, "vco_hz": soc.crg.mmcm.compute_config()["vco"],
            "dqs_phase": 90.0, "channels": [0, 1], "phy": a.phy}
    if a.phy == "wl":
        info.update(vco_hz=8 * a.sys_mhz * 1e6, groups=groups, group1_deg={0: 0.0, 1: 0.0},
                    ps_moves="ck")
    else:
        assert soc.crg.mmcm1.compute_config()["vco"] == info["vco_hz"], "MMCM VCOs differ"
    (out / "sdram_init.py").write_text(hdr + "\nphy = " + json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
