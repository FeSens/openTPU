#!/usr/bin/env python3
"""The YPCB-00338's PCIe core: LitePCIe (Gen2 x8, 128-bit, 250 MHz) in place of Xilinx's XDMA,
generated as one Verilog module, otpu_litepcie, for the board top (docs/litepcie.md).

    python3 gen_core.py --out build_core      # -> build_core/otpu_litepcie.v, pcie_s7.tcl,
                                              #    otpu_litepcie.xdc, csr.csv, csr.json, csr.h

The PCIe hard block (PCIE_2_1 and its GTX lanes) stays Xilinx's: LitePCIe's S7PCIEPHY drives it
through the 7 Series Integrated Block wrapper (pcie_7x, the IP pcie_s7.tcl creates; no license),
with the pipe clocks from LitePCIe's own MMCM. Everything above the hard block is LitePCIe's: the
TLP layer, the BAR0 master, the scatter-gather DMA and MSI.

The module (every port a plain Verilog port; the PCIe pads are the only I/O):

    pcie_*              the edge connector: refclk (clk_p/n, 100 MHz), PERST# (rst_n), 8 lanes
    clk, rst            out  the PCIe user clock (250 MHz) and its reset (link down, PERST#): the
                        clock of every other port (the top's xclk, as XDMA's axi_aclk was)
    link_up             out  the link is up (LTSSM L0)
    mmap_axi_lite_*     AXI4-Lite master, 32-bit: BAR0 0x00000-0x7FFFF, the user window (the
                        accelerator's control registers, the LiteDRAM CSRs, the XADC: the block
                        design's SmartConnect). The window is what /dev/otpu0 lets a process mmap
    h2c_cmd_*           out  host->card commands {addr, len, flags}: write the next len bytes of
                        the H2C stream at card byte address addr (otpu_pcie_mm)
    h2c_dat_*           out  the H2C stream: LitePCIe's DMA reader, 16 bytes a beat, byte k of a
                        beat at data[8k+7:8k] (host memory order)
    c2h_cmd_*           out  card->host commands {addr, len, flags}: read len bytes at addr into
                        the C2H stream
    c2h_dat_*           in   the C2H stream: LitePCIe's DMA writer
    mm_h2c_done         in   a pulse per finished H2C command (its last write response is in)
    mm_status[31:0]     in   otpu_pcie_mm's state for the driver (mm_status CSR, the record)
    mm_err              in   otpu_pcie_mm saw an error (sticky): an MSI
    mm_rst              out  the driver's soft reset of otpu_pcie_mm (the command queues reset
                        with it)

BAR0 (1 MiB): 0x00000-0x7FFFF the user window above; 0x80000- LiteX's CSR space (LitePCIe's DMA
tables, PHY status, MSI, and the otpu_dma block below), reachable only by the driver (it programs
DMA engines that can write anywhere in host memory). csr.csv / csr.h have the addresses.

otpu_dma (the CSRs the driver uses on top of LitePCIe's DMA):
    id, caps            0x4f544c50 ("OTLP"); the reader / writer table depth and the endpoint's
                        read requests in flight
    h2c_addr, h2c_len   an H2C command: writing h2c_len ([23:0] bytes, [31:24] flags) queues
                        {h2c_addr, len, flags}; c2h_addr, c2h_len the same for C2H
    queue               the command queues' levels and overflow flags
    h2c_count           finished H2C commands (mm_h2c_done pulses), mod 2^32
    c2h_count           finished C2H transfers: DMA writer descriptors with IRQ enabled whose last
                        write left the DMA (the driver enables it on a transfer's last), mod 2^32
    wb_lo, wb_hi, wb_ctrl  the completion record: when enabled, every change of h2c_count or
                        c2h_count is written to host memory at {wb_hi, wb_lo} as 16 bytes
                        {h2c_count, c2h_count, mm_status, seq}, one memory write behind the
                        transfer's own (PCIe keeps posted writes in order, so a record that shows
                        a C2H transfer done comes after all of its data), then an MSI (vector bit 0)
    mm_ctrl, mm_status  otpu_pcie_mm's soft reset ([0]) and state

Checked by tools/litepcie/sim_core.py (the core's DMA and otpu_dma against LitePCIe's host model)
and reproduced by tools/litepcie/check_core.sh.
"""
import argparse
import json
import os
import re
import shutil
import sys

from migen import *
from migen.genlib.cdc import MultiReg

from litex.gen import *
from litex.build.generic_platform import Pins, Subsignal
from litex.soc.interconnect import stream, wishbone
from litex.soc.interconnect.axi import AXILiteInterface, Wishbone2AXILite
from litex.soc.interconnect.csr import CSRStorage, CSRStatus, CSRField
from litex.soc.integration.soc import SoCRegion
from litex.soc.integration.soc_core import SoCMini
from litex.soc.integration.builder import Builder

from litepcie.common import dma_layout
from litepcie.core import LitePCIeEndpoint, LitePCIeMSI
from litepcie.frontend.dma import LitePCIeDMA
from litepcie.frontend.wishbone import LitePCIeWishboneMaster

OTLP_ID = 0x4F544C50          # "OTLP"
OTLP_VERSION = 1
BAR0_SIZE = 0x100000          # 1 MiB, XDMA's BAR0
USER_SIZE = 0x80000           # the user window (R_DDR4 + 4 x DDR4_STRIDE)
CSR_BASE = 0x80000
TABLE_DEPTH = 256             # descriptors per DMA table (1 MiB of 4 KiB pages; LUTRAM)
PENDING = 16                  # read requests in flight (16 x 512 B = 8 KiB)
CMD_DEPTH = 64                # commands per queue
MSI_DELAY = 32                # cycles a record's MSI waits behind its write (see core_logic)
LUTRAM_DEPTH = 64             # memories this deep or less: distributed RAM (see lutram)
# PCI identity: Xilinx's vendor ID as before, the device ID and subsystem new so that neither the
# xdma driver (device 7028, subsystem 10ee:4f54 by udev rule) nor this one takes the other's card.
# pcie_7x validates Device_ID as a long written in hex (Vivado 2026.1 refuses 4F54): hex digits
# 0-9 only. 5054 is in neither the xdma driver's table nor LitePCIe's
DEVICE_ID = 0x5054
SUBSYS_ID = 0x4C50            # "LP"
REVISION = 0x02

cmd_layout = [("addr", 32), ("len", 24), ("flags", 8)]


class CmdQueue(LiteXModule):
    """A command queue the driver fills through two CSRs: addr, then len (whose write queues the
    command). overflow: a command written into a full queue (dropped), sticky until reset."""
    def __init__(self, depth=CMD_DEPTH):
        self.addr = CSRStorage(32, description="Card byte address of the next command.")
        self.len = CSRStorage(32, fields=[
            CSRField("len", size=24, description="Bytes (a multiple of 16)."),
            CSRField("flags", size=8, offset=24, description="Flags (otpu_pcie_mm)."),
        ], description="Writing it queues {addr, len, flags}.")
        self.source = stream.Endpoint(cmd_layout)
        self.reset = Signal()
        self.level = Signal(bits_for(depth))
        self.overflow = Signal()

        fifo = ResetInserter()(stream.SyncFIFO(cmd_layout, depth, buffered=True))
        self.fifo = fifo
        self.comb += [
            fifo.reset.eq(self.reset),
            fifo.sink.valid.eq(self.len.wr_stb),
            fifo.sink.addr.eq(self.addr.storage),
            fifo.sink.len.eq(self.len.fields.len),
            fifo.sink.flags.eq(self.len.fields.flags),
            fifo.source.connect(self.source),
            self.level.eq(fifo.level),
        ]
        self.sync += If(self.reset, self.overflow.eq(0)).Elif(
            self.len.wr_stb & ~fifo.sink.ready, self.overflow.eq(1))


class OTPUDMA(LiteXModule):
    """The driver's side of the card's DMA (see the module docstring): the command queues, the
    completion counters and record, otpu_pcie_mm's control and status."""
    def __init__(self, endpoint, c2h_event, address_width=64):
        self.id = CSRStatus(32, reset=OTLP_ID, description="0x4f544c50, \"OTLP\".")
        self.caps = CSRStatus(fields=[
            CSRField("version", size=8, reset=OTLP_VERSION),
            CSRField("table_log2", size=8, offset=8, reset=log2_int(TABLE_DEPTH)),
            CSRField("pending", size=8, offset=16, reset=PENDING),
            CSRField("cmd_log2", size=8, offset=24, reset=log2_int(CMD_DEPTH)),
        ])
        self.h2c = CmdQueue()
        self.c2h = CmdQueue()
        self.queue = CSRStatus(fields=[
            CSRField("h2c_level", size=8), CSRField("c2h_level", size=8, offset=8),
            CSRField("h2c_overflow", size=1, offset=16), CSRField("c2h_overflow", size=1, offset=17),
        ])
        self.h2c_count = CSRStatus(32, description="Finished H2C commands, mod 2^32.")
        self.c2h_count = CSRStatus(32, description="Finished C2H transfers, mod 2^32.")
        self.wb_lo = CSRStorage(32, description="Completion record's host address, bits 31:0.")
        self.wb_hi = CSRStorage(32, description="Completion record's host address, bits 63:32.")
        self.wb_ctrl = CSRStorage(fields=[CSRField("enable", size=1)])
        self.mm_ctrl = CSRStorage(fields=[CSRField("reset", size=1,
            description="Hold otpu_pcie_mm and the command queues in reset.")])
        self.mm_status = CSRStatus(32, description="otpu_pcie_mm's state.")

        # otpu_pcie_mm's ports
        self.h2c_cmd = self.h2c.source
        self.c2h_cmd = self.c2h.source
        self.h2c_done = Signal()
        self.mm_status_in = Signal(32)
        self.mm_err = Signal()
        self.mm_rst = Signal()
        self.irq = Signal()         # a record written (MSI vector bit 0)

        # # #

        self.comb += [
            self.mm_rst.eq(self.mm_ctrl.fields.reset),
            self.h2c.reset.eq(self.mm_rst),
            self.c2h.reset.eq(self.mm_rst),
            self.queue.fields.h2c_level.eq(self.h2c.level),
            self.queue.fields.c2h_level.eq(self.c2h.level),
            self.queue.fields.h2c_overflow.eq(self.h2c.overflow),
            self.queue.fields.c2h_overflow.eq(self.c2h.overflow),
            self.mm_status.status.eq(self.mm_status_in),
        ]

        # completion counters (never reset but by the link: the driver works with differences)
        h2c_cnt = Signal(32)
        c2h_cnt = Signal(32)
        self.sync += [
            If(self.h2c_done, h2c_cnt.eq(h2c_cnt + 1)),
            If(c2h_event, c2h_cnt.eq(c2h_cnt + 1)),
        ]
        self.comb += [self.h2c_count.status.eq(h2c_cnt), self.c2h_count.status.eq(c2h_cnt)]

        # the completion record: one 4-dword memory write per change of the counts, through its
        # own crossbar master port. A C2H transfer's count steps when its last write is accepted
        # by the DMA writer's port, so the record's write enters the crossbar after it, and the
        # TLP layer keeps requests in order from there
        port = endpoint.crossbar.get_master_port(write_only=True)
        dwords = len(port.source.dat) // 32
        assert dwords == 4, "the record is one 128-bit beat"
        sh = Signal(32)             # the counts the last record carried
        sc = Signal(32)
        wh = Signal(32)             # the record being written
        wc = Signal(32)
        ws = Signal(32)
        seq = Signal(32)
        adr = Signal(address_width)
        enable = self.wb_ctrl.fields.enable
        self.fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            If(enable & ((h2c_cnt != sh) | (c2h_cnt != sc)),
                NextValue(wh, h2c_cnt),
                NextValue(wc, c2h_cnt),
                NextValue(ws, self.mm_status_in),
                NextValue(adr, Cat(self.wb_lo.storage, self.wb_hi.storage)[:address_width]),
                NextState("WRITE"),
            )
        )
        self.comb += [
            port.source.channel.eq(port.channel),
            port.source.first.eq(1),
            port.source.last.eq(1),
            port.source.we.eq(1),
            port.source.req_id.eq(endpoint.phy.id),
            port.source.tag.eq(0),
            port.source.len.eq(dwords),
            port.source.adr.eq(adr),
            port.source.dat.eq(Cat(wh, wc, ws, seq)),
        ]
        fsm.act("WRITE",
            port.source.valid.eq(1),
            If(port.source.ready,
                NextValue(sh, wh),
                NextValue(sc, wc),
                NextValue(seq, seq + 1),
                self.irq.eq(1),
                NextState("IDLE"),
            )
        )
        # with the record off, an MSI per change all the same (the driver then reads the counts)
        self.sync += If(~enable, sh.eq(h2c_cnt), sc.eq(c2h_cnt))
        h2c_d = Signal()
        c2h_d = Signal()
        self.sync += [h2c_d.eq(self.h2c_done), c2h_d.eq(c2h_event)]
        self.irq_off = Signal()
        self.comb += self.irq_off.eq(~enable & (h2c_d | c2h_d))


def pcie_ios(nlanes):
    return [
        ("pcie", 0,
            Subsignal("rst_n", Pins(1)),
            Subsignal("clk_p", Pins(1)),
            Subsignal("clk_n", Pins(1)),
            Subsignal("rx_p", Pins(nlanes)),
            Subsignal("rx_n", Pins(nlanes)),
            Subsignal("tx_p", Pins(nlanes)),
            Subsignal("tx_n", Pins(nlanes)),
        ),
    ]


def user_ios(dw):
    def cmd(name, out):
        return (name, 0, Subsignal("valid", Pins(1)), Subsignal("ready", Pins(1)),
                Subsignal("addr", Pins(32)), Subsignal("len", Pins(24)), Subsignal("flags", Pins(8)))

    def dat(name):
        return (name, 0, Subsignal("valid", Pins(1)), Subsignal("ready", Pins(1)),
                Subsignal("data", Pins(dw)), Subsignal("last", Pins(1)))
    return [
        ("clk", 0, Pins(1)), ("rst", 0, Pins(1)), ("link_up", 0, Pins(1)),
        cmd("h2c_cmd", True), dat("h2c_dat"), cmd("c2h_cmd", True), dat("c2h_dat"),
        ("mm", 0, Subsignal("h2c_done", Pins(1)), Subsignal("status", Pins(32)),
                  Subsignal("err", Pins(1)), Subsignal("rst", Pins(1))),
    ]


def core_logic(soc, phy, platform=None, sys_clk_freq=250e6):
    """Everything above the PHY but BAR0's master (shared with sim_core.py, which puts LitePCIe's
    host model in the PHY's place): the endpoint, the DMA, otpu_dma, MSI. Returns the endpoint and
    otpu_pcie_mm's side: the command and data streams, each through a register slice (both ways:
    otpu_pcie_mm sits apart from the PCIe block, by the memory port), and otpu_dma."""
    soc.pcie_endpoint = endpoint = LitePCIeEndpoint(phy,
        endianness=phy.endianness, address_width=64, max_pending_requests=PENDING)
    soc.pcie_dma0 = dma = LitePCIeDMA(phy, endpoint, with_table=True, table_depth=TABLE_DEPTH,
        address_width=64, data_width=phy.data_width)
    soc.otpu_dma = od = OTPUDMA(endpoint, c2h_event=dma.writer.irq)
    # the record's MSI MSI_DELAY cycles behind its write: the write still crosses the TLP layer
    # and the PHY when its pulse comes, and the hard block's MSI could otherwise reach the host
    # first (the driver's waiter would find the old record and sleep again)
    soc.pcie_msi = msi = LitePCIeMSI(width=32)
    rec_irq = Signal(MSI_DELAY)
    soc.sync += rec_irq.eq(Cat(od.irq | od.irq_off, rec_irq[:-1]))
    soc.comb += [
        msi.irqs[0].eq(rec_irq[-1]),
        msi.irqs[1].eq(od.mm_err),
        msi.source.connect(phy.msi),
    ]
    user = {}
    for name, src, layout in (("h2c_cmd", od.h2c_cmd, cmd_layout), ("c2h_cmd", od.c2h_cmd, cmd_layout),
                              ("h2c_dat", dma.source, dma_layout(phy.data_width))):
        buf = stream.Buffer(layout, pipe_valid=True, pipe_ready=True)
        setattr(soc, name + "_buf", buf)
        soc.comb += src.connect(buf.sink)
        user[name] = buf.source
    buf = stream.Buffer(dma_layout(phy.data_width), pipe_valid=True, pipe_ready=True)
    soc.c2h_dat_buf = buf
    soc.comb += buf.source.connect(dma.sink)
    user["c2h_dat"] = buf.sink
    return endpoint, user, od


def bar0_logic(soc, endpoint, phy):
    """BAR0's master (shared with sim_core.py): wishbone, the user window to an AXI-Lite master
    (returned), LiteX's CSRs above it."""
    soc.pcie_wb = LitePCIeWishboneMaster(endpoint,
        qword_aligned=getattr(phy, "qword_aligned", False))
    soc.bus.add_master(name="pcie", master=soc.pcie_wb.wishbone)
    wb = wishbone.Interface(data_width=32)
    soc.bus.add_slave(name="user", slave=wb,
        region=SoCRegion(origin=0, size=USER_SIZE, cached=False))
    axi = AXILiteInterface(data_width=32, address_width=32)
    soc.wb2axi = Wishbone2AXILite(wb, axi)
    return axi


class OTPULitePCIe(SoCMini):
    SoCMini.mem_map["csr"] = CSR_BASE

    def add_csr_bridge(self, name="csr", origin=None, with_register=False):
        # registered: BAR0's write decode (the TLP header to the CSR storages) in two cycles
        SoCMini.add_csr_bridge(self, name=name, origin=origin, with_register=True)

    def __init__(self, platform, nlanes=8):
        from litepcie.phy.s7pciephy import S7PCIEPHY
        sys_clk_freq = 250e6
        platform.add_extension(pcie_ios(nlanes))
        platform.add_extension(user_ios(128))
        SoCMini.__init__(self, platform, clk_freq=sys_clk_freq, csr_data_width=32,
            csr_ordering="big", ident="openTPU LitePCIe", ident_version=True)

        # clock: the PCIe user clock (userclk2, 250 MHz at Gen2 x8 / 128 bits), out to the top
        self.cd_sys = ClockDomain()
        clk = platform.request("clk")
        rst = platform.request("rst")
        self.comb += [
            self.cd_sys.clk.eq(ClockSignal("pcie")), self.cd_sys.rst.eq(ResetSignal("pcie")),
            clk.eq(ClockSignal("pcie")), rst.eq(ResetSignal("pcie")),
        ]

        # PHY: Gen2 x8, 128 bits, 1 MiB BAR0, one MSI vector. Its streams in its own clock domain
        # (cd "pcie", which is sys): another would put asynchronous FIFOs on both datapaths
        self.pcie_phy = phy = S7PCIEPHY(platform, platform.request("pcie"),
            data_width=128, pcie_data_width=128, bar0_size=BAR0_SIZE, cd="pcie")
        phy.update_config({
            "Device_ID": f"{DEVICE_ID:04X}",
            "Subsystem_Vendor_ID": "10EE",
            "Subsystem_ID": f"{SUBSYS_ID:04X}",
            "Revision_ID": f"{REVISION:02X}",
            "Class_Code_Base": "12", "Class_Code_Sub": "00", "Class_Code_Interface": "00",
        })
        self.comb += platform.request("link_up").eq(phy._link_status.fields.status)

        endpoint, user, od = core_logic(self, phy, platform, sys_clk_freq)
        axi = bar0_logic(self, endpoint, phy)
        platform.add_extension(axi.get_ios("mmap_axi_lite"))
        self.comb += axi.connect_to_pads(platform.request("mmap_axi_lite"), mode="master")

        # otpu_pcie_mm
        h2c_cmd = platform.request("h2c_cmd")
        c2h_cmd = platform.request("c2h_cmd")
        h2c_dat = platform.request("h2c_dat")
        c2h_dat = platform.request("c2h_dat")
        mm = platform.request("mm")
        for cmd, src in ((h2c_cmd, user["h2c_cmd"]), (c2h_cmd, user["c2h_cmd"])):
            self.comb += [cmd.valid.eq(src.valid), src.ready.eq(cmd.ready),
                          cmd.addr.eq(src.addr), cmd.len.eq(src.len), cmd.flags.eq(src.flags)]
        h2c, c2h = user["h2c_dat"], user["c2h_dat"]
        self.comb += [
            # the reader's stream (the driver disables last on its descriptors: no last)
            h2c_dat.valid.eq(h2c.valid),
            h2c.ready.eq(h2c_dat.ready),
            h2c_dat.data.eq(h2c.data),
            h2c_dat.last.eq(h2c.last),
            c2h.valid.eq(c2h_dat.valid),
            c2h_dat.ready.eq(c2h.ready),
            c2h.data.eq(c2h_dat.data),
            c2h.last.eq(c2h_dat.last),
            od.h2c_done.eq(mm.h2c_done),
            od.mm_status_in.eq(mm.status),
            od.mm_err.eq(mm.err),
            mm.rst.eq(od.mm_rst),
        ]


def lutram(v):
    """Distributed RAM for the core's small memories (LUTRAM_DEPTH words or fewer: the endpoint's
    completion buffers, its tag and request queues, the reader's request queue, the command
    queues). Vivado puts some in block RAM, whose 1.8 ns clock-to-out the 250 MHz paths after them
    cannot take; a LUTRAM's synchronous read is a slice flip-flop."""
    def attr(m):
        return ('(* ram_style = "distributed" *)\n' if int(m[2]) + 1 <= LUTRAM_DEPTH else "") + m[0]
    return re.sub(r"^(reg \[\d+:0\] storage(?:_\d+)?)\[0:(\d+)\];", attr, v, flags=re.M)


def ip_tcl(platform):
    """The pcie_7x IP's creation (LitePCIe's pre-synthesis commands, without synth_ip: the
    project synthesizes it out of context) and the GTX common's LOC reset."""
    # LiteX's Vivado flow formats each command (doubled braces escape single ones)
    cmds = [c.format(build_name="otpu_litepcie") for c in platform.toolchain.pre_synthesis_commands]
    cmds = [c for c in cmds if not c.startswith("synth_ip")]
    return "\n".join(cmds) + "\n"


def place_tcl(platform):
    """LitePCIe's pre-placement commands (the IP's GTX common LOC reset): the board's
    constraints place the lanes, and the commons follow them."""
    cmds = [c.format(build_name="otpu_litepcie") for c in platform.toolchain.pre_placement_commands]
    return "\n".join(cmds) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="build_core")
    ap.add_argument("--lanes", type=int, default=8)
    a = ap.parse_args()
    from litex.build.xilinx import XilinxPlatform
    platform = XilinxPlatform("xc7k480tffg1156-2", io=[], toolchain="vivado")
    soc = OTPULitePCIe(platform, nlanes=a.lanes)
    gw = os.path.join(a.out, "gateware")
    builder = Builder(soc, output_dir=a.out, gateware_dir=gw, compile_gateware=False,
                      csr_csv=os.path.join(a.out, "csr.csv"),
                      csr_json=os.path.join(a.out, "csr.json"))
    builder.build(build_name="otpu_litepcie", run=False)
    with open(os.path.join(gw, "otpu_litepcie.v")) as f:
        v = lutram(f.read())
    with open(os.path.join(a.out, "otpu_litepcie.v"), "w") as f:
        f.write(v)
    # the identifier ROM ("openTPU LitePCIe"), which the Verilog $readmemh's by file name
    shutil.copy(os.path.join(gw, "otpu_litepcie_mem.init"), os.path.join(a.out, "otpu_litepcie_mem.init"))
    with open(os.path.join(a.out, "pcie_s7_place.tcl"), "w") as f:
        f.write("# LitePCIe's pre-placement commands, generated by tools/litepcie/gen_core.py\n"
                + place_tcl(platform))
    with open(os.path.join(a.out, "pcie_s7.tcl"), "w") as f:
        f.write("# The PCIe hard block's wrapper (LitePCIe S7PCIEPHY's IP), generated by "
                "tools/litepcie/gen_core.py\n" + ip_tcl(platform))
    # LiteX's design constraints (the pipe clock mux's select, its synchronizers), less the
    # refclk's clock (the board XDC names it on its own port)
    xdc = open(os.path.join(gw, "otpu_litepcie.xdc")).read()
    keep = [ln for ln in xdc.splitlines()
            if ln.startswith("set_false_path") or ln.startswith("set_max_delay")]
    with open(os.path.join(a.out, "otpu_litepcie.xdc"), "w") as f:
        f.write("# otpu_litepcie's constraints (LiteX's), generated by tools/litepcie/gen_core.py\n"
                + "\n".join(keep) + "\n")
    # the driver's header: the CSR addresses it uses, as BAR0 offsets
    csr = json.load(open(os.path.join(a.out, "csr.json")))
    regs = csr["csr_registers"]
    lines = ["/* otpu_litepcie's CSRs (BAR0 offsets), generated by tools/litepcie/gen_core.py */",
             "#ifndef OTPU_LITEPCIE_CSR_H", "#define OTPU_LITEPCIE_CSR_H"]
    for name in sorted(regs):
        r = regs[name]
        lines.append(f"#define CSR_{name.upper()}_ADDR 0x{r['addr']:05x}L  /* {r['size']} word(s), {r['type']} */")
    for k, v in (("OTLP_ID", OTLP_ID), ("OTLP_VERSION", OTLP_VERSION), ("OTLP_USER_SIZE", USER_SIZE),
                 ("OTLP_CSR_BASE", CSR_BASE), ("OTLP_TABLE_DEPTH", TABLE_DEPTH),
                 ("OTLP_CMD_DEPTH", CMD_DEPTH), ("OTLP_DEVICE_ID", DEVICE_ID),
                 ("OTLP_SUBSYS_ID", SUBSYS_ID)):
        lines.append(f"#define {k} 0x{v:x}")
    lines.append("#endif")
    with open(os.path.join(a.out, "csr.h"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"core in {a.out}: otpu_litepcie.v, otpu_litepcie_mem.init, pcie_s7.tcl, pcie_s7_place.tcl, "
          "otpu_litepcie.xdc, csr.csv, csr.json, csr.h")


if __name__ == "__main__":
    main()
