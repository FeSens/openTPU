"""The software-injected DFI command a sys cycle after its CSR write (production core,
gen_core.py; docs/litedram.md section 12).

LiteDRAM's PhaseInjector drives a phase's cs_n / ras_n / cas_n / we_n and wrdata_en / rddata_en
straight from its command_issue CSR's write strobe, which the CSR bus decodes from its address
register in the same cycle. From there the DFIInjector's mux and the PHY's register on sys's
falling edge (WL7DDRPHY's nreg) leave half a cycle, and the PHY's serializer another half less
the crossing's 1.0 ns: in the 110ec6d full build at 133.33 MHz those four pads' registers sat
between the CSR bus at the die's centre (3-4 LUT levels, +0.33..+0.6 ns) and their pads
(+0.220..+0.29 ns), the write clocks' worst paths. This PhaseInjector registers the strobe, so the
injected command leaves from a flip-flop: the same phase, a sys cycle later. Its address, bank,
write data and command fields are CSR storages written by earlier accesses (an access takes 5
sys cycles), so they hold; the command, its write data enable and its read data enable move
together, and so do commands on different phases. The command still goes out within its own
CSR access; the calibration (the host's ddrcal, the selfcal firmware) reads rddata many
accesses later. The controller's commands (the injector's sel) do not pass here.

    with registered_injector(): ...      # LiteDRAMCores built inside use this PhaseInjector

Derived from LiteDRAM's litedram/dfii.py (BSD-2-Clause): Copyright (c) 2015 Sebastien
Bourdeauducq, (c) 2016-2019 Florent Kermarrec.
"""
from contextlib import contextmanager

from migen import If, Module, Replicate, Signal

import litedram.dfii
from litex.soc.interconnect.csr import AutoCSR, CSR, CSRField, CSRStatus, CSRStorage


class PhaseInjector(Module, AutoCSR):
    """litedram.dfii.PhaseInjector (its name, so the netlist's names stay, and its CSRs), the
    command from the command_issue strobe registered (DELAY registers; dfii_q_check.py's
    mutations set 0 and 2)."""
    DELAY = 1

    def __init__(self, phase):
        self._command = CSRStorage(fields=[
            CSRField("cs", size=1, description="DFI chip select bus"),
            CSRField("we", size=1, description="DFI write enable bus"),
            CSRField("cas", size=1, description="DFI column address strobe bus"),
            CSRField("ras", size=1, description="DFI row address strobe bus"),
            CSRField("wren", size=1, description="DFI write data enable bus"),
            CSRField("rden", size=1, description="DFI read data enable bus"),
            CSRField("cs_top", size=1, description="DFI chip select bus for top half only"),
            CSRField("cs_bottom", size=1, description="DFI chip select bus for bottom half only"),
        ], description="Control DFI signals on a single phase")
        self._command_issue = CSR()
        self._address = CSRStorage(len(phase.address), reset_less=True,
                                   description="DFI address bus")
        self._baddress = CSRStorage(len(phase.bank), reset_less=True,
                                    description="DFI bank address bus")
        self._wrdata = CSRStorage(len(phase.wrdata), reset_less=True,
                                  description="DFI write data bus")
        self._rddata = CSRStatus(len(phase.rddata), description="DFI read data bus")

        # # #

        issue = self._command_issue.wr_stb
        for _ in range(self.DELAY):
            q = Signal()
            self.sync += q.eq(issue)
            issue = q
        self.comb += [
            If(issue,
                If(self._command.fields.cs_top,
                    phase.cs_n.eq(2),
                ).Else(
                    If(self._command.fields.cs_bottom,
                        phase.cs_n.eq(1),
                    ).Else(
                        phase.cs_n.eq(Replicate(~self._command.fields.cs, len(phase.cs_n))),
                    ),
                ),
                phase.we_n.eq(~self._command.fields.we),
                phase.cas_n.eq(~self._command.fields.cas),
                phase.ras_n.eq(~self._command.fields.ras)
            ).Else(
                phase.cs_n.eq(Replicate(1, len(phase.cs_n))),
                phase.we_n.eq(1),
                phase.cas_n.eq(1),
                phase.ras_n.eq(1)
            ),
            phase.address.eq(self._address.storage),
            phase.bank.eq(self._baddress.storage),
            phase.wrdata_en.eq(issue & self._command.fields.wren),
            phase.rddata_en.eq(issue & self._command.fields.rden),
            phase.wrdata.eq(self._wrdata.storage),
            phase.wrdata_mask.eq(0)
        ]
        self.sync += If(phase.rddata_valid, self._rddata.status.eq(phase.rddata))


@contextmanager
def registered_injector():
    """DFIInjectors built inside take this PhaseInjector for their phases."""
    stock = litedram.dfii.PhaseInjector
    litedram.dfii.PhaseInjector = PhaseInjector
    try:
        yield
    finally:
        litedram.dfii.PhaseInjector = stock
