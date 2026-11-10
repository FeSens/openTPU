#!/usr/bin/env python3
"""The BCU1525's LiteDRAM core: one DDR4 channel (LiteDRAM's UltraScale+ PHY USPDDRPHY) with two
native user ports, as one Verilog module, otpu_ld4, that the board top instantiates once per DIMM
channel (docs/bcu1525.md section 4). --dimm picks the DIMM:

    rdimm   the card's original modules: a 16 GB x4 RDIMM (MTA18ASF2G72PZ), 72 bits with ECC
            (boards/bcu1525/litedram/)
    udimm   the lab card's (2026-10-19): an 8 GB x8 UDIMM (GOODRAM GR2400D464L17S/8G, one
            rank of eight 8 Gb x8 chips), 64 bits, no ECC, a data mask per byte
            (boards/bcu1525/litedram_udimm/)

    PATH=~/eda/rvshim:~/eda/litex-venv/bin:$PATH \
      python tools/litedram/gen_core_ddr4.py --cpu [--dimm udimm] --out build_ld4
        # -> build_ld4/otpu_ld4.v, csr.csv, sdram_init.py, ports.txt (the BIOS needs meson from
        #    the venv and the riscv64-unknown-elf-* names: ~/eda/rvshim)

One core per channel, not one for all four as the YPCB's (gen_core.py: both DDR3 channels in one
module): the BCU1525's channels sit in three SLRs (C0 SLR0, C1 / C2 SLR1, C3 SLR2) with a 300 MHz
clock each, so each core is its channel's: its own MMCM from its own clock, its own IDELAYCTRL,
calibrated on its own, placed by its banks; the module is generated once and is the same for every
channel (the pins come from the top's ports: boards/bcu1525/constraints/ddr4_c<n>.pins.xdc).

The module's ports (the DDR4 pads are the only I/O):

    clk300              in   the channel's 300 MHz clock, on a global buffer (the top's IBUFDS and
                             BUFG: the top may use one of them for the core clock too)
    rst                 in   asynchronous, high: the clocks and the channel
    sys_clk, sys_rst    out  the controller clock (the DDR4 clock / 4: --sys-mhz) and its reset:
                             the native ports' clock (otpu_mem_ch's controller side)
    ctl_clk, ctl_rst    in   the CSR port's clock and reset (XDMA's axi_aclk)
    ctl_*               AXI-Lite slave, 16-bit byte address, 32-bit data: the CSRs (csr.csv),
                        crossing into sys_clk; as the YPCB core's. The address is the offset in
                        the CSR region: csr.csv's addresses less its base (0xF000_0000 with
                        --cpu, whose BIOS ROM is at 0; 0 without), the host's window
    c0_* / c0b_*        the two native ports (the even banks', the odd banks': otpu_mem_ch puts a
                        beat on the port of its bank's parity), in sys_clk, 512-bit data through
                        LiteDRAM's ECC (SECDED per 64-bit word over the 72-bit DIMM; the UDIMM's
                        straight, through the same registers: ecc_ports.NativePortsPlain):
                        cmd_valid/ready/we/addr[27:0] (the 64-byte beat in the channel: 16 GiB;
                        the UDIMM's addr[26:0], 8 GiB),
                        wdata_valid/ready/data[511:0]/we[63:0], rdata_valid/ready/data[511:0];
                        the YPCB core's contract (gen_core.py): whole beats only (ECC), write
                        data taken when the controller wants it, reads in command order, no read
                        backpressure.
    c0_ready            out  calibrated (cal_ready, set by the calibration: the host or the CPU)
    c0_ded              out  the ECC has counted an uncorrectable word since the last ecc_clear
                             (the UDIMM's: 0)
    ddram_*             the DDR4 pads: a[13:0], act_n, ba[1:0], bg[1:0], ras_n, cas_n, we_n (A16,
                        A15, A14), cke, odt, cs_n, reset_n, clk_p, clk_n, dq[71:0], dqs_p[17:0],
                        dqs_n[17:0] (x4: dqs[2i], dqs[2i+1] strobe byte i's nibbles); the UDIMM's
                        dq[63:0], dqs_p[7:0], dqs_n[7:0] (x8: dqs[i] strobes byte i) and dm[7:0]
                        (DM_n, on the pins an RDIMM has its upper nibbles' strobes: DQS9..16_t)

Calibration. USPDDRPHY is calibrated in software (LiteX's liblitedram sdram_init(): the RDIMM's
register (RCD) control words, the mode registers, write leveling, read leveling over the IDELAYs
and bitslips, write latency): --cpu adds a VexRiscv with LiteX's BIOS, which runs it at reset and
prints its log on the UART pads (uart_tx / uart_rx; one channel's core drives the board's USB
UART), then hands the PHY to the controller; the core's `ready` is then cal_ready OR the DFII's
hardware control (the BIOS leaves DFII_CONTROL_SEL set after its init, passing or not). The host
calibrates a UDIMM core's DIMM through the CSRs (opentpu/host/ddr4.py's calibrate_dimm, after the
BIOS: `otpu-ddr4 calibrate`), and sets cal_ready; without --cpu that is the only calibration.

DDR4-2400 (--sys-mhz 300) is the DIMMs' rate; 200 (DDR4-1600, the lowest JEDEC DDR4 speed bin)
is the first bring-up's default: LiteDRAM's controller at 300 MHz on the fabric is unproven here.
USPDDRPHY uses the ISERDESE3s in component mode, whose CLK (sys4x) takes 1.6 ns at least on the -2
part: at 200 its 800 MHz fails that pulse width check by 0.350 ns (every BCU1525 build), and the
UDIMM core is generated at 150 (DDR4-1200, sys4x 600 MHz; LiteX's own XCU1525 target runs 125).

The UDIMM. Its chips' maker is not known (its SPD is not read: no SPD bus on the FPGA's I2C), so
its timings are JEDEC's DDR4-2400R bin (17-17-17: tAA = tRCD = tRP = 14.16 ns) on MT40A1G8's
geometry, and the CAS latency comes from that tAA (CL 12 at DDR4-1600, where LiteDRAM's default
11 is the faster 13.75 ns bin's). LiteDRAM's DDR4 init enables the data mask (MR5 A10) whatever
the PHY: the UDIMM's DM_n pins are the PHY's dm (driven high, unmasked, for the whole beats the
ports write), not left floating. No RCD control words (is_rdimm off).
"""
import argparse
import json
import math
import pprint
import re
import sys
from pathlib import Path

from migen import *
from litex.build.generic_platform import Pins, Subsignal, IOStandard
from litex.build.xilinx import XilinxUSPPlatform
from litex.gen import LiteXModule
from litex.soc.cores.clock import USPMMCM, USPIDELAYCTRL
from litex.soc.interconnect import wishbone
from litex.soc.interconnect.axi import (AXILiteInterface, AXILiteClockDomainCrossing,
                                        AXILite2Wishbone)
from litex.soc.integration.soc_core import SoCCore
from litex.soc.integration.builder import Builder

from litedram.common import LiteDRAMNativePort
from litedram.core.controller import ControllerSettings
from litedram.modules import MTA18ASF2G72PZ, MT40A1G8, _SpeedgradeTimings
from litedram.phy import usddrphy
from litedram.init import get_sdram_phy_py_header

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ecc_ports import NativePortsECC, NativePortsPlain          # noqa: E402
from calcpu import Cal                                          # noqa: E402
from ld_test import BIST, reset_value                           # noqa: E402

PART = "xcvu9p-fsgd2104-2l-e"
DW = 512                    # user data (+ 64 ECC bits on the RDIMM's DRAM side)


class GR2400D464L17S(MT40A1G8):
    """GOODRAM GR2400D464L17S/8G: DDR4-2400 CL17, 8 GB, one rank of eight 8 Gb x8 chips (MT40A1G8's
    geometry and refresh), no ECC; JEDEC's DDR4-2400R timings (its chips' own are not known)."""
    TAA = 14.16
    speedgrade_timings = {
        "2400": _SpeedgradeTimings(tRP=14.16, tRCD=14.16, tWR=15, tRFC=MT40A1G8.trfc, tFAW=(20, 25),
                                   tRAS=32),
    }
    speedgrade_timings["default"] = speedgrade_timings["2400"]


# per DIMM: the module, the data and strobe pads, the beat address bits (64-byte beats), ECC
DIMMS = {
    "rdimm": dict(module=MTA18ASF2G72PZ, rdimm=True, dq=72, dqs=18, dm=0, aw=28, ecc=True,
                  ident="openTPU LiteDRAM DDR4"),
    "udimm": dict(module=GR2400D464L17S, rdimm=False, dq=64, dqs=8, dm=8, aw=27, ecc=False,
                  ident="openTPU LiteDRAM DDR4 UDIMM"),
}

# The pads: names only (the core is a module; the top's XDC places them). LiteX wants a pin per
# bit, so each gets a placeholder that the generated XDC drops.
def _pins(n, tag):
    return Pins(" ".join(f"{tag}{i}" for i in range(n)))


def io(dimm):
    d = DIMMS[dimm]
    dm = (Subsignal("dm", _pins(d["dm"], "DM")),) if d["dm"] else ()
    return [
        ("ddram", 0,
         Subsignal("a", _pins(14, "A")), Subsignal("act_n", _pins(1, "ACT")),
         Subsignal("ba", _pins(2, "BA")), Subsignal("bg", _pins(2, "BG")),
         Subsignal("ras_n", _pins(1, "RAS")), Subsignal("cas_n", _pins(1, "CAS")),
         Subsignal("we_n", _pins(1, "WE")), Subsignal("cke", _pins(1, "CKE")),
         Subsignal("clk_n", _pins(1, "CN")), Subsignal("clk_p", _pins(1, "CP")),
         Subsignal("cs_n", _pins(1, "CS")), Subsignal("dq", _pins(d["dq"], "DQ")),
         Subsignal("dqs_n", _pins(d["dqs"], "DQSN")), Subsignal("dqs_p", _pins(d["dqs"], "DQSP")),
         *dm,
         Subsignal("odt", _pins(1, "ODT")), Subsignal("reset_n", _pins(1, "RST"))),
        ("serial", 0, Subsignal("tx", _pins(1, "TX")), Subsignal("rx", _pins(1, "RX"))),
    ]


def user_io(aw):
    user = [("clk300", 0, Pins(1)), ("rst", 0, Pins(1)), ("sys_clk", 0, Pins(1)),
            ("sys_rst", 0, Pins(1)), ("ctl_clk", 0, Pins(1)), ("ctl_rst", 0, Pins(1)),
            ("ctl", 0,
             Subsignal("awvalid", Pins(1)), Subsignal("awready", Pins(1)),
             Subsignal("awaddr", Pins(16)), Subsignal("wvalid", Pins(1)),
             Subsignal("wready", Pins(1)), Subsignal("wdata", Pins(32)),
             Subsignal("wstrb", Pins(4)), Subsignal("bvalid", Pins(1)), Subsignal("bready", Pins(1)),
             Subsignal("bresp", Pins(2)), Subsignal("arvalid", Pins(1)),
             Subsignal("arready", Pins(1)), Subsignal("araddr", Pins(16)),
             Subsignal("rvalid", Pins(1)), Subsignal("rready", Pins(1)),
             Subsignal("rdata", Pins(32)), Subsignal("rresp", Pins(2)))]
    for u in ("", "b"):
        user += [(f"c0{u}", 0,
                  Subsignal("cmd_valid", Pins(1)), Subsignal("cmd_ready", Pins(1)),
                  Subsignal("cmd_we", Pins(1)), Subsignal("cmd_addr", Pins(aw)),
                  Subsignal("wdata_valid", Pins(1)), Subsignal("wdata_ready", Pins(1)),
                  Subsignal("wdata_data", Pins(DW)), Subsignal("wdata_we", Pins(DW // 8)),
                  Subsignal("rdata_valid", Pins(1)), Subsignal("rdata_ready", Pins(1)),
                  Subsignal("rdata_data", Pins(DW)))
                 + ((Subsignal("ready", Pins(1)), Subsignal("ded", Pins(1))) if u == "" else ())]
    return user


class Platform(XilinxUSPPlatform):
    def __init__(self, dimm):
        XilinxUSPPlatform.__init__(self, PART, io(dimm), toolchain="vivado")


class CRG(LiteXModule):
    """LiteX's sqrl_xcu1525 CRG: the channel's 300 MHz clock -> an MMCM: sys4x (the PHY's
    serializers) on a BUFGCE, sys = sys4x / 4 on a BUFGCE_DIV (phase aligned), the IDELAYCTRL's
    reference `fref` (>= 300 MHz for USPDDRPHY; one the MMCM's VCO also makes 4 x sys from:
    idelay_ref())."""
    def __init__(self, platform, f, fref):
        self.rst = Signal()
        self.cd_sys = ClockDomain()
        self.cd_sys4x = ClockDomain()
        self.cd_pll4x = ClockDomain()
        self.cd_idelay = ClockDomain()
        self.pll = pll = USPMMCM(speedgrade=-2)
        self.comb += pll.reset.eq(self.rst)
        pll.register_clkin(platform.request("clk300"), 300e6)
        pll.create_clkout(self.cd_pll4x, f * 4, buf=None, with_reset=False)
        pll.create_clkout(self.cd_idelay, fref)
        platform.add_false_path_constraints(self.cd_sys.clk, pll.clkin)
        self.specials += [
            Instance("BUFGCE_DIV", p_BUFGCE_DIVIDE=4, i_CE=1, i_I=self.cd_pll4x.clk,
                     o_O=self.cd_sys.clk),
            Instance("BUFGCE", i_CE=1, i_I=self.cd_pll4x.clk, o_O=self.cd_sys4x.clk),
        ]
        self.idelayctrl = USPIDELAYCTRL(cd_ref=self.cd_idelay, cd_sys=self.cd_sys)
        # LiteX sets the IDELAYCTRL's SIM_DEVICE to ULTRASCALE on UltraScale+ too: Vivado changes it
        # at link_design, but its IODELAY_GROUP replicas keep ULTRASCALE (write_bitstream's DRC
        # ADEF-911 refused all of them, 2026-10-19)
        for i in self.idelayctrl._fragment.specials:
            if isinstance(i, Instance) and i.of == "IDELAYCTRL":
                for it in i.items:
                    if isinstance(it, Instance.Parameter) and it.name == "SIM_DEVICE":
                        it.value = "ULTRASCALE_PLUS"


def idelay_ref(f):
    """The IDELAYCTRL reference: the first of 400, 600, 500, 300 MHz that divides a VCO in the
    UltraScale+ MMCM's range (800-1600 MHz) together with 4 x sys."""
    for fref in (400e6, 600e6, 500e6, 300e6):
        for vco in range(800, 1601):
            if (vco * 1e6) % (4 * f) == 0 and (vco * 1e6) % fref == 0:
                return fref
    raise ValueError(f"no IDELAYCTRL reference for sys {f / 1e6} MHz")


class OTPULiteDRAM4(SoCCore):
    mem_map = {"csr": 0x0000_0000} | {}

    def __init__(self, f=200e6, bist=True, cpu=False, dimm="rdimm"):
        d = DIMMS[dimm]
        aw = d["aw"]
        platform = Platform(dimm)
        platform.add_extension(user_io(aw))
        fref = idelay_ref(f)
        self.crg = CRG(platform, f, fref)
        self.comb += self.crg.rst.eq(platform.request("rst"))
        kw = dict(cpu_type=None, integrated_rom_size=0, integrated_sram_size=0, with_uart=False,
                  with_timer=False)
        if cpu:
            # LiteX's BIOS on a VexRiscv: sdram_init() at reset, its log on the UART
            kw = dict(cpu_type="vexriscv", cpu_variant="minimal", integrated_rom_size=0x10000,
                      integrated_sram_size=0x2000, with_uart=True, uart_name="serial",
                      with_timer=True)
            self.mem_map = {"rom": 0x0000_0000, "sram": 0x1000_0000, "main_ram": 0x4000_0000,
                            "csr": 0xF000_0000}
        SoCCore.__init__(self, platform, f, ident=d["ident"], csr_data_width=32,
                         with_ctrl=True, **kw)
        self.comb += [platform.request("sys_clk").eq(ClockSignal("sys")),
                      platform.request("sys_rst").eq(ResetSignal("sys"))]

        # the CAS latency from the module's tAA where it has one (the UDIMM), LiteDRAM's default
        # for the DDR4 clock otherwise
        taa = getattr(d["module"], "TAA", None)
        cl = None if taa is None else math.ceil(round(taa * 4 * f / 1e9, 3))
        self.ddrphy = p = usddrphy.USPDDRPHY(platform.request("ddram"), memtype="DDR4",
                                             sys_clk_freq=f, iodelay_clk_freq=fref, cl=cl,
                                             is_rdimm=d["rdimm"])
        # the CPU's view of the memory (the BIOS's memtest) only with --cpu; the accelerator's
        # ports are the crossbar's own
        self.add_sdram("sdram", phy=p, module=d["module"](f, "1:4"),
                       with_soc_interconnect=cpu, l2_cache_size=0,
                       size=0x4000_0000 if cpu else None,     # the BIOS's window: 1 GiB
                       controller_settings=ControllerSettings())
        core = self.sdram
        users, raws = [], []
        for u in ("", "b"):
            raw = core.crossbar.get_port()
            assert raw.address_width == aw and raw.data_width == DW + 64 * d["ecc"], \
                (raw.address_width, raw.data_width)
            user = LiteDRAMNativePort("both", aw, DW)
            users.append(user)
            raws.append(raw)
            upads = platform.request(f"c0{u}")
            self.comb += [
                user.cmd.valid.eq(upads.cmd_valid), upads.cmd_ready.eq(user.cmd.ready),
                user.cmd.we.eq(upads.cmd_we), user.cmd.addr.eq(upads.cmd_addr),
                user.cmd.last.eq(1),
                user.wdata.valid.eq(upads.wdata_valid), upads.wdata_ready.eq(user.wdata.ready),
                user.wdata.data.eq(upads.wdata_data), user.wdata.we.eq(upads.wdata_we),
                upads.rdata_valid.eq(user.rdata.valid), user.rdata.ready.eq(upads.rdata_ready),
                upads.rdata_data.eq(user.rdata.data),
            ]
            if u == "":
                pads = upads
        if d["ecc"]:
            self.ecc = NativePortsECC(users, raws)
            self.sync += pads.ded.eq(self.ecc.ded_errors.status != 0)
        else:
            self.ecc = NativePortsPlain(users, raws)
            self.comb += pads.ded.eq(0)
        self.cal = Cal()
        hw = Signal()
        if cpu:   # the BIOS hands the PHY to the controller after a passing init
            self.comb += hw.eq(core.dfii._control.storage[0])
        self.comb += pads.ready.eq(self.cal.ready.storage | hw)
        if bist:
            self.bist = BIST(core.crossbar.get_port(), modules=d["dq"] // 8)

        # CSRs: AXI-Lite (ctl_clk) -> sys -> the CSR bus, as gen_core.py
        self.cd_ctl = ClockDomain()
        self.comb += [self.cd_ctl.clk.eq(platform.request("ctl_clk")),
                      self.cd_ctl.rst.eq(platform.request("ctl_rst"))]
        axil_c = AXILiteInterface(data_width=32, address_width=32, clock_domain="ctl")
        axil_s = AXILiteInterface(data_width=32, address_width=32, clock_domain="sys")
        c = platform.request("ctl")
        # the host's 16-bit offset into the CSR region (with --cpu the ROM is at 0)
        csr_base = self.mem_map["csr"]
        self.comb += [
            axil_c.aw.valid.eq(c.awvalid), c.awready.eq(axil_c.aw.ready),
            axil_c.aw.addr.eq(c.awaddr | csr_base),
            axil_c.w.valid.eq(c.wvalid), c.wready.eq(axil_c.w.ready), axil_c.w.data.eq(c.wdata),
            axil_c.w.strb.eq(c.wstrb),
            c.bvalid.eq(axil_c.b.valid), axil_c.b.ready.eq(c.bready), c.bresp.eq(axil_c.b.resp),
            axil_c.ar.valid.eq(c.arvalid), c.arready.eq(axil_c.ar.ready),
            axil_c.ar.addr.eq(c.araddr | csr_base),
            c.rvalid.eq(axil_c.r.valid), axil_c.r.ready.eq(c.rready), c.rdata.eq(axil_c.r.data),
            c.rresp.eq(axil_c.r.resp),
        ]
        self.axil_cdc = AXILiteClockDomainCrossing(axil_c, axil_s, cd_from="ctl", cd_to="sys")
        wb = wishbone.Interface(data_width=32, address_width=32, addressing="word")
        self.axil2wb = AXILite2Wishbone(axil_s, wb)
        self.bus.add_master(name="ctl", master=wb)


def sdram_init(soc, sys_mhz):
    """sdram_init.py: the init sequence and the PHY settings (for a host-side calibration)."""
    ps = soc.ddrphy.settings
    hdr = get_sdram_phy_py_header(ps, soc.sdram.controller.settings.timing)
    info = {"sys_hz": sys_mhz * 1e6, "nphases": ps.nphases, "rdphase": reset_value(ps.rdphase),
            "wrphase": reset_value(ps.wrphase), "databits": ps.databits,
            "dfi_databits": ps.dfi_databits, "modules": ps.databits // 8,
            "delays": ps.delays, "bitslips": ps.bitslips, "cl": ps.cl, "cwl": ps.cwl,
            "read_latency": ps.read_latency, "write_latency": ps.write_latency,
            "rdimm": bool(getattr(ps, "is_rdimm", False)), "phy": "uspddrphy", "memtype": "DDR4",
            "part": PART, "idelay_ref_hz": idelay_ref(sys_mhz * 1e6)}
    # Python, not JSON (the host exec()s it: "rdimm" is a bool)
    return info, hdr + "\nphy = " + pprint.pformat(info, sort_dicts=False) + "\n"


def append_sources(verilog, sources):
    """The core as one Verilog file: the platform's other sources (the CPU) appended, their modules
    renamed otpu_ld4_* (the YPCB core's calibration CPU is otpu_selfcal_*: both may be in a tree)."""
    for src in sources:
        text = Path(src).read_text()
        names = re.findall(r"^module\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.M)
        if not names:
            continue
        pat = r"\b(" + "|".join(names) + r")\b(?=\s+[A-Za-z_\\])"
        text = re.sub(r"^module\s+(" + "|".join(names) + r")\b",
                      lambda m: "module otpu_ld4_" + m.group(1), text, flags=re.M)
        text = re.sub(pat, lambda m: "otpu_ld4_" + m.group(1), text)
        verilog = re.sub(pat, lambda m: "otpu_ld4_" + m.group(1), verilog)
        verilog += f"\n// {Path(src).name} (modules renamed otpu_ld4_*)\n" + text
    return verilog


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sys-mhz", type=float, default=200.0,
                    help="controller clock: the DDR4 rate / 8 (200: DDR4-1600, 300: DDR4-2400)")
    ap.add_argument("--no-bist", action="store_true")
    ap.add_argument("--cpu", action="store_true",
                    help="a VexRiscv with LiteX's BIOS calibrates the channel at reset (UART log)")
    ap.add_argument("--dimm", choices=sorted(DIMMS), default="rdimm",
                    help="rdimm: the original 16 GB x4 ECC RDIMMs; udimm: 8 GB x8 UDIMMs, no ECC")
    ap.add_argument("--out", default="build_ld4")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    soc = OTPULiteDRAM4(a.sys_mhz * 1e6, bist=not a.no_bist, cpu=a.cpu, dimm=a.dimm)
    info, init_py = sdram_init(soc, a.sys_mhz)
    b = Builder(soc, output_dir=str(out), compile_software=a.cpu, compile_gateware=False,
                csr_csv=str(out / "csr.csv"), bios_console="lite")
    b.build(build_name="otpu_ld4", run=False)
    gw = out / "gateware"
    # the XDC: LiteX's platform commands (false paths, the PHY's), not the pads' placeholders
    # (the top's ddr4_c<n>.pins.xdc places them)
    keep = [l for l in (gw / "otpu_ld4.xdc").read_text().splitlines()
            if "set_property LOC" not in l and "IOSTANDARD" not in l and not l.startswith("#")]
    (out / "otpu_ld4.xdc").write_text("\n".join(l for l in keep if l.strip()) + "\n")
    verilog = (gw / "otpu_ld4.v").read_text()
    if a.cpu:
        verilog = append_sources(verilog, [src for src, *_ in soc.platform.sources
                                           if Path(src).name != "otpu_ld4.v"])
    (out / "otpu_ld4.v").write_text(verilog)
    # the memories the Verilog $readmemh's by file name (project files: create_project.tcl): the
    # identifier ROM; with --cpu the BIOS ROM and the CPU's SRAM
    for init in sorted(gw.glob("otpu_ld4_*.init")):
        (out / init.name).write_text(init.read_text())
    (out / "sdram_init.py").write_text(init_py)
    ports = [l.strip() for l in (gw / "otpu_ld4.v").read_text().split(");")[0].splitlines()
             if l.strip().startswith(("input", "output", "inout"))]
    (out / "ports.txt").write_text("\n".join(ports) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
