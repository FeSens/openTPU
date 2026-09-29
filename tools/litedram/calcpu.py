"""The LiteDRAM core's own calibration CPU (docs/litedram.md section 10): a VexRiscv (LiteX's
"minimal" variant: RV32I, no caches) that calibrates the core's DDR3 channels at reset, so the card
needs no host for it. Its firmware (tools/litedram/selfcal_fw) is opentpu/host/ddrcal.py's
calibrate_channel in C, run on each channel; it ends by raising the channel's cal_ready as the
host would. The host keeps an override (selfcal_hold) and reads the result (opentpu/host/selfcal.py).

The CPU's address space (byte addresses):

    0x0000_0000   the firmware memory, MEM_BYTES (instructions on one port, data on the other: a
                  true dual-port BRAM whose initial content is the image; the CPU restarts in place.
                  While the CPU is held, the data port is the host's: selfcal_mem_adr /
                  selfcal_mem_dat write a word, selfcal_mem_rdat reads it, so a new firmware goes
                  in without a new bitstream, opentpu/host/selfcal.py load())
    0x2000_0000   the result mailbox, 256 words (the CPU writes, the host reads them through
                  selfcal_mbox_adr / selfcal_mbox_dat)
    0x2000_0800   the sys cycle counter, 64 bits (reading the low word latches the high word)
    0xF000_0000   the SoC bus: CSR byte address a at 0xF000_0000 + a (the addresses of csr.csv)
    elsewhere     reads 0, writes are dropped

The window onto the SoC bus is a second bus master beside the host's (the SoC's round-robin
arbiter); after each of its accesses it leaves the bus free for a cycle, so a waiting host access
goes next. selfcal_hold stops the CPU between two of its SoC bus accesses (an access under way
finishes first; `held` in selfcal_status says when it has) and keeps it in reset; releasing it
restarts the firmware from the top.
"""
import re
import sys
from pathlib import Path

from migen import *
from litex.gen import LiteXModule
from litex.soc.cores.cpu.vexriscv import VexRiscv
from litex.soc.integration.builder import Builder
from litex.soc.interconnect import wishbone
from litex.soc.interconnect.csr import AutoCSR, CSRStatus, CSRStorage, CSRField

HERE = Path(__file__).resolve().parent

MAGIC = 0x5CA1
LOCATION = 16           # CSR location (0x8000): after the core's 16 blocks, the same in every image
MEM_BYTES = 16384
MBOX_WORDS = 256


class SelfCal(LiteXModule):
    autocsr_exclude = {"mem"}           # the firmware memory is the CPU's alone, not a CSR memory

    def __init__(self, platform, firmware=None, mem_bytes=MEM_BYTES):
        """firmware: the image as 32-bit words (selfcal_fw/fw.py words()), None for an empty
        memory (a first pass that only needs the CSR map)."""
        self.hold = CSRStorage(1, description="1: the CPU held in reset (from its next SoC bus "
                                              "access on); 0: it restarts its calibration.")
        self.config = CSRStorage(fields=[
            CSRField("channels", size=2, reset=0b11, description="Channels to calibrate."),
            CSRField("reserved", size=6),
            CSRField("stride", size=8, reset=1, description="CK phase scan stride, fine steps."),
        ], description="Read by the firmware when it starts.")
        self.status = CSRStatus(fields=[
            CSRField("held", size=1, description="The CPU is held (selfcal_hold)."),
            CSRField("mem_log2", size=4, description="The firmware memory's size: 2**mem_log2 words."),
            CSRField("reserved", size=11),
            CSRField("magic", size=16, reset=MAGIC, description=f"{MAGIC:#x}: the core has the CPU."),
        ])
        self.state = CSRStorage(32, description="Written by the firmware: 2 bits per channel (0 "
                                "idle, 1 running, 2 ok, 3 failed), bit 7 done, errors in 15:8 / 23:16.")
        self.mbox_adr = CSRStorage(8)
        self.mbox_dat = CSRStatus(32, description="Mailbox word selfcal_mbox_adr.")
        self.mem_adr = CSRStorage(16, description="Firmware memory word for selfcal_mem_dat / _rdat.")
        self.mem_dat = CSRStorage(32, description="A write stores the word at selfcal_mem_adr while "
                                  "the CPU is held (dropped otherwise).")
        self.mem_rdat = CSRStatus(32, description="The firmware memory's word at selfcal_mem_adr "
                                  "while the CPU is held.")
        self.bus = bus = wishbone.Interface(data_width=32, address_width=32, addressing="word")

        # # #

        self.cpu = cpu = VexRiscv(platform, variant="minimal")
        cpu.set_reset_address(0)
        ibus, dbus = cpu.ibus, cpu.dbus
        words = mem_bytes // 4
        aw = log2_int(words)
        mem = Memory(32, words, init=list(firmware) if firmware is not None else None,
                     name="selfcal_mem")
        ip = mem.get_port()
        # READ_FIRST: a registered read (LiteX writes a byte-writable WRITE_FIRST port as an
        # asynchronous read of a registered address, which synthesis maps to a block RAM whose
        # collisions differ from the RTL: Synth 8-6430, which build.tcl stops on). The CPU never
        # reads the word it writes in the same cycle, so the mode is not seen.
        dp = mem.get_port(write_capable=True, we_granularity=8, mode=READ_FIRST)
        mbox = Memory(32, MBOX_WORDS, name="selfcal_mbox")
        mw = mbox.get_port(write_capable=True)
        mr = mbox.get_port()
        self.specials += mem, ip, dp, mbox, mw, mr
        self.mem = mem

        # instructions: the memory alone. The port reads ahead: once a fetch is answered it reads
        # the next word, so a fetch of that word is answered in its first cycle (a jump costs one)
        fadr, fvalid, hit = Signal(aw), Signal(), Signal()
        self.comb += [
            hit.eq(fvalid & (fadr == ibus.adr[:aw])),
            ibus.ack.eq(ibus.cyc & ibus.stb & hit),
            ip.adr.eq(Mux(ibus.cyc & ibus.stb & hit, ibus.adr[:aw] + 1, ibus.adr[:aw])),
            ibus.dat_r.eq(ip.dat_r),
        ]
        self.sync += [fadr.eq(ip.adr), fvalid.eq(1)]

        # data: decoded on byte address bits 31:28 (word address bits 29:26)
        region = dbus.adr[26:30]
        req = dbus.cyc & dbus.stb
        to_mem, to_loc, to_soc = region == 0x0, region == 0x2, region == 0xF
        loc = dbus.adr[:10]                                   # word within the local region
        to_mbox, to_timer = to_loc & (loc[8:10] == 0), to_loc & (loc[9:10] == 1)
        local_ack, loc_dat = Signal(), Signal(32)
        self.sync += local_ack.eq(req & ~to_soc & ~local_ack)
        held = Signal()
        self.comb += [
            # the data port: the CPU's, or the host's while the CPU is held (a firmware upload)
            If(held,
                dp.adr.eq(self.mem_adr.storage[:aw]), dp.dat_w.eq(self.mem_dat.storage),
                [dp.we[i].eq(self.mem_dat.re) for i in range(4)],
            ).Else(
                dp.adr.eq(dbus.adr[:aw]), dp.dat_w.eq(dbus.dat_w),
                [dp.we[i].eq(req & dbus.we & dbus.sel[i] & to_mem) for i in range(4)],
            ),
            self.mem_rdat.status.eq(dp.dat_r),
            self.status.fields.mem_log2.eq(aw),
            mw.adr.eq(loc[:8]), mw.dat_w.eq(dbus.dat_w), mw.we.eq(req & dbus.we & to_mbox),
            mr.adr.eq(self.mbox_adr.storage), self.mbox_dat.status.eq(mr.dat_r),
        ]
        cycles, hi = Signal(64), Signal(32)
        self.sync += [
            cycles.eq(cycles + 1),
            loc_dat.eq(0),
            If(req & ~dbus.we & to_timer & ~local_ack,
                If(loc[0], loc_dat.eq(hi)).Else(loc_dat.eq(cycles[:32]), hi.eq(cycles[32:]))),
        ]

        # the SoC bus: one access at a time, a free cycle after each; hold between accesses
        started, gap = Signal(), Signal()
        open_ = (~self.hold.storage | started) & ~gap
        self.comb += [
            bus.adr.eq(dbus.adr[:26]), bus.dat_w.eq(dbus.dat_w), bus.sel.eq(dbus.sel),
            bus.we.eq(dbus.we), bus.cti.eq(0), bus.bte.eq(0),
            bus.cyc.eq(dbus.cyc & to_soc & open_), bus.stb.eq(dbus.stb & to_soc & open_),
            dbus.dat_r.eq(Mux(to_soc, bus.dat_r, Mux(to_mem, dp.dat_r, loc_dat))),
            dbus.ack.eq(Mux(to_soc, bus.ack | bus.err, local_ack)),   # the CPU ignores err
            dbus.err.eq(0),
            ibus.err.eq(0),
            held.eq(self.hold.storage & ~started),
            cpu.reset.eq(held),
            self.status.fields.held.eq(held),
        ]
        self.sync += [
            If(bus.ack | bus.err, started.eq(0)).Elif(bus.cyc & bus.stb, started.eq(1)),
            gap.eq(bus.ack | bus.err),
        ]

    def set_firmware(self, firmware):
        """The memory's initial content (before the Verilog is written)."""
        assert len(firmware) == self.mem.depth
        self.mem.init = list(firmware)


def add(soc, platform, firmware=None, mem_bytes=MEM_BYTES):
    """The CPU in a CPU-less LiteX SoC (gen_core.py, ld_test.py): its CSRs pinned at LOCATION, its
    window a master on the SoC bus."""
    soc.selfcal = SelfCal(platform, firmware, mem_bytes)
    soc.csr.add("selfcal", n=LOCATION)
    soc.bus.add_master(name="selfcal", master=soc.selfcal.bus)
    return soc.selfcal


class Cal(Module, AutoCSR):
    """cal_ready / cal1_ready: set once the channel is calibrated, by the core's CPU or the host
    (the accelerator waits for it)."""
    def __init__(self):
        self.ready = CSRStorage(1, description="1: calibrated, the controller has the PHY.")


class FirmwareBuilder(Builder):
    """LiteX's Builder with one step between the CSR map and the Verilog: `hook` (the calibration
    CPU's firmware is compiled against the map and put in its memory)."""
    def __init__(self, soc, hook=None, **kw):
        super().__init__(soc, **kw)
        self.hook = hook

    def _generate_csr_map(self):
        super()._generate_csr_map()
        if self.hook:
            self.hook()


def build_firmware(soc, out, fw_id=0):
    """The firmware for this core (csr.csv and sdram_init.py in `out`), into its memory."""
    sys.path.insert(0, str(HERE / "selfcal_fw"))
    import fw
    mem_bytes = soc.selfcal.mem.depth * 4
    data = fw.target(out, out / "selfcal_fw", fw_id=fw_id, mem_bytes=mem_bytes)
    soc.selfcal.set_firmware(fw.words(data, mem_bytes))
    return len(data)


def one_file(verilog, gateware, sources):
    """The core as one Verilog file: the calibration CPU's memory initial values inlined (LiteX
    writes them as .init files for $readmemh; the core's other .init files stay as they are) and
    the platform's other Verilog sources (the calibration CPU) appended, their modules renamed
    otpu_selfcal_*."""
    def inline(m):
        if "selfcal" not in m.group(1):
            return m.group(0)
        vals = (gateware / m.group(1)).read_text().split()
        return "\n".join(f"\t{m.group(2)}[{i}] = 'h{v};" for i, v in enumerate(vals))
    verilog = re.sub(r'\$readmemh\("([^"]+)", ([A-Za-z0-9_]+)\);', inline, verilog)
    for src in sources:
        text = Path(src).read_text()
        names = re.findall(r"^module\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.M)
        rename = lambda t: re.sub(r"\b(" + "|".join(names) + r")\b(?=\s+[A-Za-z_\\])",
                                  lambda m: "otpu_selfcal_" + m.group(1), t)
        # module declarations and instantiations (a name followed by an instance name)
        text = re.sub(r"^module\s+(" + "|".join(names) + r")\b",
                      lambda m: "module otpu_selfcal_" + m.group(1), text, flags=re.M)
        text = rename(text)
        verilog = rename(verilog)
        verilog += f"\n// {Path(src).name} (modules renamed otpu_selfcal_*)\n" + text
    return verilog
