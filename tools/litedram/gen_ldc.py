#!/usr/bin/env python3
"""One channel of the production LiteDRAM controller as a simulation model: LiteDRAM's own
controller (bank machines, multiplexer, refresher) and crossbar, from the commit the production
core pins (boards/ypcb-00338/litedram/core.json), with gen_core.py's settings -- the MT41K256M8
geometry with tRFC 160 ns, 1:4, the core's ControllerSettings (LiteDRAM's defaults but
ctl_settings.py's refresh postponing and read / write times), the two user ports (the even and
the odd banks', otpu_mem_ch's split) through LiteDRAMNativePortECC's encoders and one decoder
(ecc_ports.py, as the core) and a third, idle crossbar port (the BIST's) -- behind a DFI stub PHY
with WL7DDRPHY's latencies and no memory. So the command scheduling is the card's; the data is
not modelled (reads return the ECC decode of zeros). sim/verilator/otpu_ldc_mem.sv puts it behind
the board's bridge (otpu_mem_ch) with the data held in the model.

    python3 gen_ldc.py OUT.v [--name otpu_ldc_ch] [--cmd-buffer-depth 8] [--no-refresh]
                      [--postponing 2] [--read-time 256] [--write-time 128] [--ports 2]
                      [--no-lock]

DDR3-1066 only (the controller at 133.33 MHz, CL 7 / CWL 6, the latencies WL7DDRPHY derives
from them, the data-sheet timings in that clock): the board's DDR3 never runs faster, the rate
its HR banks are specified for, so the generator has no other rate. The options are for
experiments: the controller's command buffer depth, no refresh, refresh postponing, the
multiplexer's read and write times (the defaults: the core's, ctl_settings.py), the number
of user ports (the one-port core's: --ports 1, its signals unprefixed; more: p<i>_*), and the
crossbar without its lock (a master's commands in one bank at a time; without it read data may
come back out of order, so it is a timing bound only).

Ports (sys clock): sys_clk, sys_rst and per user port p<i>_cmd_valid/ready/we/addr[24:0] (the
64-byte beat in the channel), p<i>_wdata_valid/ready/data[511:0]/we[63:0],
p<i>_rdata_valid/ready/data[511:0] (ready tied high, as the board ties it).

Setup: bench_native.py's venv, with migen, litex and litedram at core.json's commits. The
committed model (rtlsim.MEMORY's LDC): python3 gen_ldc.py sim/verilator/otpu_ldc_ch.v
"""
import argparse
import sys
from pathlib import Path

from migen import Module, Signal
from migen.fhdl.verilog import convert

from litedram.modules import MT41K256M8, _SpeedgradeTimings
from litedram.common import PhySettings, LiteDRAMNativePort, get_sys_latency, get_sys_phase
from litedram.phy.dfi import Interface as DFIInterface
from litedram.core import LiteDRAMCore
from litedram.core.controller import ControllerSettings
import litedram.core.crossbar as xbar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ecc_ports import NativePortsECC                            # noqa: E402
from dfii_q import registered_injector                         # noqa: E402
from ctl_settings import CONTROLLER                             # noqa: E402


class MT41K256M8_tRFC160(MT41K256M8):
    """The board's 2 Gb parts with tRFC 160 ns (as tools/litedram/ld_test.py's)."""
    speedgrade_timings = {"default": _SpeedgradeTimings(tRP=13.75, tRCD=13.75, tWR=15,
                                                        tRFC=(None, 160), tFAW=(None, 40),
                                                        tRAS=35)}


# DDR3-1066, the board's only rate: the controller clock (MHz) and CL / CWL
MTS, SYS_MHZ, CL, CWL = 1066, 133.333, 7, 6


class StubPHY(Module):
    """DFI sink: read data valid read_latency cycles after the read enable (the controller
    enforces every DRAM timing; the PHY only sets the latencies)."""
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


class Channel(Module):
    def __init__(self, cmd_buffer_depth=8, refresh=True,
                 postponing=CONTROLLER["refresh_postponing"], nports=1,
                 read_time=CONTROLLER["read_time"], write_time=CONTROLLER["write_time"]):
        clk_freq = SYS_MHZ * 1e6
        module = MT41K256M8_tRFC160(clk_freq, "1:4")
        # WL7DDRPHY's latencies (tools/litedram/wl7ddrphy.py): read 8, write 1
        cl, cwl = CL, CWL
        cl_sys, cwl_sys = get_sys_latency(4, cl), get_sys_latency(4, cwl)
        phase_cycles, rdphase = divmod(get_sys_phase(4, cl_sys, cl) + 1, 4)
        cl_sys -= phase_cycles
        self.lat = dict(cl=cl, cwl=cwl, rdphase=rdphase, wrphase=get_sys_phase(4, cwl_sys, cwl),
                        read_latency=cl_sys + 6, write_latency=cwl_sys - 1)
        ps = PhySettings(phytype="A7DDRPHY", memtype="DDR3", databits=72, strobes=9,
                         dfi_databits=144, nranks=1, nphases=4, cmd_latency=0, cmd_delay=None,
                         write_latency_calibration=True, read_leveling=True, delays=32,
                         bitslips=8, with_dm=False, **self.lat)
        self.submodules.phy = StubPHY(module, ps)
        cs = ControllerSettings(cmd_buffer_depth=cmd_buffer_depth, with_refresh=refresh,
                                refresh_postponing=postponing, read_time=read_time,
                                write_time=write_time)
        with registered_injector():         # as the production core (dfii_q.py)
            self.submodules.core = core = LiteDRAMCore(self.phy, module.geom_settings,
                                                       module.timing_settings, clk_freq,
                                                       controller_settings=cs)
        self.users, self.ios, raws = [], set(), []
        for _ in range(nports):
            raw = core.crossbar.get_port()
            assert raw.address_width == 25 and raw.data_width == 576
            user = LiteDRAMNativePort("both", 25, 512)
            raws.append(raw)
            self.comb += user.cmd.last.eq(1)
            self.users.append(user)
            self.ios |= {user.cmd.valid, user.cmd.ready, user.cmd.we, user.cmd.addr,
                         user.wdata.valid, user.wdata.ready, user.wdata.data, user.wdata.we,
                         user.rdata.valid, user.rdata.ready, user.rdata.data}
        self.submodules += NativePortsECC(self.users, raws)
        bist = core.crossbar.get_port()
        self.comb += [bist.cmd.valid.eq(0), bist.wdata.valid.eq(0), bist.rdata.ready.eq(1)]
        self.module = module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--name", default="otpu_ldc_ch")
    ap.add_argument("--cmd-buffer-depth", type=int, default=8)
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--postponing", type=int, default=CONTROLLER["refresh_postponing"])
    ap.add_argument("--read-time", type=int, default=CONTROLLER["read_time"])
    ap.add_argument("--write-time", type=int, default=CONTROLLER["write_time"])
    ap.add_argument("--ports", type=int, default=2)
    ap.add_argument("--no-lock", action="store_true")
    a = ap.parse_args()
    if a.no_lock:
        orig = xbar.LiteDRAMCrossbar.do_finalize

        def do_finalize(self):
            for nb in range(self.controller.nbanks):     # the crossbar sees locks never set
                getattr(self.controller, "bank" + str(nb)).lock = Signal()
            orig(self)
        xbar.LiteDRAMCrossbar.do_finalize = do_finalize
    ch = Channel(a.cmd_buffer_depth, not a.no_refresh, a.postponing, a.ports, a.read_time,
                 a.write_time)
    for i, u in enumerate(ch.users):
        pre = f"p{i}_" if a.ports > 1 else ""
        for sig, n in ((u.cmd.valid, "cmd_valid"), (u.cmd.ready, "cmd_ready"),
                       (u.cmd.we, "cmd_we"), (u.cmd.addr, "cmd_addr"),
                       (u.wdata.valid, "wdata_valid"), (u.wdata.ready, "wdata_ready"),
                       (u.wdata.data, "wdata_data"), (u.wdata.we, "wdata_we"),
                       (u.rdata.valid, "rdata_valid"), (u.rdata.ready, "rdata_ready"),
                       (u.rdata.data, "rdata_data")):
            sig.name_override = pre + n
    t = ch.module.timing_settings
    hdr = (f"// generated by tools/litedram/gen_ldc.py: {a.name}, DDR3-{MTS} "
           f"({SYS_MHZ} MHz; {', '.join(f'{k} {v}' for k, v in ch.lat.items())}), "
           f"cmd_buffer_depth {a.cmd_buffer_depth}, refresh {not a.no_refresh} (postponing "
           f"{a.postponing}), read_time {a.read_time}, write_time {a.write_time}, {a.ports} user "
           f"port(s), lock {not a.no_lock}; tREFI {t.tREFI} "
           f"tRFC {t.tRFC} tRP {t.tRP} tRCD {t.tRCD} tRAS {t.tRAS} tWR {t.tWR} tWTR {t.tWTR} "
           f"tCCD {t.tCCD} tRRD {t.tRRD} tFAW {t.tFAW}\n")
    open(a.out, "w").write(hdr + str(convert(ch, ios=ch.ios, name=a.name)))
    print(hdr.strip())


if __name__ == "__main__":
    main()
