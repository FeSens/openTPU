"""The core's CSR bus with a register stage per group of banks (gen_core.py).

LiteX's CSR bus is one bus: the bridge's adr / re / we / dat_w registers drive every bank, and
every bank's read data comes back on one OR. In the production core the two channels' banks sit
at the die's two ends (each by its DDR3 banks' I/O), so the bridge's registers drive about 140
loads each, over 200 rows: 7 ns of route at 0 levels in the 812bb01 build, the core's worst
paths at 133.33 MHz (interface1_dat_w -> phaseinjector*_storage, +0.077 ns).

CSRGroups puts a register stage in front of each group of banks (channel 0's, channel 1's, the
rest: group_of), so each group's copy is placed by its banks; the read data comes back through a
register per group. Wishbone2CSRWait is LiteX's registered Wishbone2CSR acking WAIT cycles later,
for the two registers: an access takes 5 cycles instead of 3. Writes land in the banks one cycle
after the bridge presents them, reads return the same data; the CSR map is unchanged.
tools/litedram/csr_pipe_check.py checks it against LiteX's bridge and bus in migen's simulator.
"""
from functools import reduce
from operator import or_

from migen import Module, Signal
from migen.genlib.fsm import NextState
from litex.soc.interconnect import csr_bus, wishbone

# the banks of channel 0 (channel 1's carry a 1: gen_core.py's CSR names)
CHANNEL_BANKS = {"bist", "cal", "ddrphy", "ecc", "phase", "sdram", "wclk"}


def group_of(name):
    """A bank's group: its channel's (ch0, ch1), or the rest's (ctrl, identifier_mem, selfcal)."""
    if name in CHANNEL_BANKS:
        return "ch0"
    if name.endswith("1") and name[:-1] in CHANNEL_BANKS:
        return "ch1"
    return "misc"


class Wishbone2CSRWait(wishbone.Wishbone2CSR):
    """LiteX's registered Wishbone2CSR (whatever `register` says: the core's is registered) with
    its ack WAIT cycles later (the bus behind it: CSRGroups, a register on the way out and one on
    the way back)."""
    WAIT = 2

    def __init__(self, bus_wishbone=None, bus_csr=None, register=True):
        super().__init__(bus_wishbone, bus_csr, register=True)
        nxt = [s for s in self.fsm.actions["WRITE-READ"] if isinstance(s, NextState)]
        assert len(nxt) == 1 and nxt[0].state == "ACK", nxt
        nxt[0].state = "WAIT0"
        for i in range(self.WAIT):
            self.fsm.act(f"WAIT{i}", NextState(f"WAIT{i + 1}" if i + 1 < self.WAIT else "ACK"))


class CSRGroups(Module):
    """The CSR bus from one master to groups of slaves: per group, the master's adr / re / we /
    dat_w a cycle late and the slaves' read data (their OR) a cycle late."""
    def __init__(self, master, groups):
        rds = []
        for slaves in groups:
            bus = csr_bus.Interface.like(master)
            rd = Signal(len(master.dat_r))
            self.sync += [
                bus.adr.eq(master.adr), bus.re.eq(master.re), bus.we.eq(master.we),
                bus.dat_w.eq(master.dat_w),
                rd.eq(bus.dat_r),
            ]
            self.comb += bus.connect(*slaves)
            rds.append(rd)
        self.comb += master.dat_r.eq(reduce(or_, rds))


class PipelinedCSR:
    """A SoC mixin: the CSR bridge Wishbone2CSRWait, the CSR bus CSRGroups by group_of."""
    def _finalize_bus(self):
        stock, wishbone.Wishbone2CSR = wishbone.Wishbone2CSR, Wishbone2CSRWait
        try:
            super()._finalize_bus()
        finally:
            wishbone.Wishbone2CSR = stock

    def _finalize_csr(self):
        soc = self

        def interconnect(masters, slaves):
            assert len(masters) == 1, "one CSR master (the bridge)"
            ba = soc.csr_bankarray
            names = {id(rmap.bus): name for name, _, _, rmap in ba.banks}
            names.update({id(mmap.bus): name for name, _, _, mmap in ba.srams})
            groups = {}
            for s in slaves:
                groups.setdefault(group_of(names[id(s)]), []).append(s)
            soc.csr_groups = {g: [n for n in names.values() if group_of(n) == g] for g in groups}
            return CSRGroups(masters[0], [groups[g] for g in sorted(groups)])

        stock, csr_bus.InterconnectShared = csr_bus.InterconnectShared, interconnect
        try:
            super()._finalize_csr()
        finally:
            csr_bus.InterconnectShared = stock
