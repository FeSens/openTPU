"""A refresher that refreshes in the controller's idle time (docs/litedram.md section 11, "Refresh
in the idle time"), for the simulation model (gen_ldc.py --idle-refresh) and, once it earns its
place, the core (gen_core.py).

LiteDRAM's Refresher refreshes on a timer: every tREFI (with refresh_postponing N, every N tREFI,
N refreshes back to back), whatever the ports are doing, so each refresh blocks the reads it falls
on. DDR3 lets the controller postpone up to 8 refreshes and pull up to 8 in ahead of time (JEDEC
JESD79-3: at most 8 owed and at most 8 ahead at any time). This one keeps that balance in a
credit, the tREFI ticks minus the refreshes done, between -ahead and +behind:

- idle: when no bank machine has a request (Multiplexer wires `idle`: no request at a bank
  machine and none queued in it) for min_idle cycles in a row and the credit is above -ahead, it
  refreshes once (precharge all, tRP, refresh, tRFC: tRP + tRFC cycles with the banks closed);
- forced: when the credit reaches behind, it refreshes as LiteDRAM's does (waits for the bank
  machines, then refreshes burst times back to back, each lowering the credit).

ZQCS (once a second) follows a refresh as in LiteDRAM's. The FSM keeps LiteDRAM's states and the
module's name (`refresher`), so the netlist's names and the simulation's probes stay.

Derived from LiteDRAM's litedram/core/refresher.py at core.json's commit (BSD-2-Clause):
Copyright (c) 2016-2019 Florent Kermarrec, (c) 2015 Sebastien Bourdeauducq.
"""
from migen import *

from litex.soc.interconnect import stream

from litedram.common import cmd_request_rw_layout
from litedram.core.refresher import RefreshExecuter, RefreshTimer, ZQCSExecuter


class IdleRefresher(Module):
    def __init__(self, settings, clk_freq, zqcs_freq=1e0, postponing=1, ahead=8, behind=8,
                 burst=2, min_idle=4):
        assert 0 <= ahead <= 8 and 1 <= behind <= 8 and 1 <= burst <= behind
        abits = settings.geom.addressbits
        babits = settings.geom.bankbits + log2_int(settings.phy.nranks)
        self.cmd = cmd = stream.Endpoint(cmd_request_rw_layout(a=abits, ba=babits))
        self.idle = Signal()          # no bank machine has a request (the Multiplexer's)

        # # #

        trp, trfc = settings.timing.tRP, settings.timing.tRFC
        if settings.timing.tREFI < 100:
            raise ValueError("Clk/tREFI is ratio too low , please increase Clk frequency or disable Refresh.")
        self.submodules.timer = timer = RefreshTimer(settings.timing.tREFI)
        self.comb += timer.wait.eq(~timer.done)

        # the credit, biased by ahead: 0 = ahead refreshes done early, ahead + behind = behind owed
        credit = Signal(max=ahead + behind + 1, reset=ahead)
        dec = Signal()                # a refresh done this cycle
        self.sync += If(timer.done & ~dec, credit.eq(credit + 1)
                        ).Elif(dec & ~timer.done, credit.eq(credit - 1))

        idle_n = Signal(max=min_idle + 1)
        self.sync += If(self.idle, If(idle_n != min_idle, idle_n.eq(idle_n + 1))).Else(idle_n.eq(0))
        forced = Signal()
        self.comb += forced.eq(credit >= ahead + behind)
        want_idle = Signal()
        self.comb += want_idle.eq(self.idle & (idle_n == min_idle) & (credit != 0))

        executer = RefreshExecuter(cmd, trp, trfc)
        self.submodules += executer
        left = Signal(max=burst + 1)  # refreshes left in this sequence
        self.comb += dec.eq(executer.done)

        wants_zqcs = Signal()
        zq = settings.timing.tZQCS is not None
        if zq:
            self.submodules.zqcs_timer = zqcs_timer = RefreshTimer(int(clk_freq / zqcs_freq))
            self.comb += wants_zqcs.eq(zqcs_timer.done)
            self.submodules.zqs_executer = zqcs_executer = ZQCSExecuter(cmd, trp, settings.timing.tZQCS)
            self.comb += zqcs_timer.wait.eq(~zqcs_executer.done)

        self.submodules.fsm = fsm = FSM()
        fsm.act("IDLE",
            If(settings.with_refresh,
                If(forced,
                    NextValue(left, burst),
                    NextState("WAIT-BANK-MACHINES")
                ).Elif(want_idle,
                    NextValue(left, 1),
                    NextState("WAIT-BANK-MACHINES")
                )
            )
        )
        fsm.act("WAIT-BANK-MACHINES",
            cmd.valid.eq(1),
            If(cmd.ready,
                executer.start.eq(1),
                NextState("DO-REFRESH")
            )
        )
        finish = [cmd.valid.eq(0), cmd.last.eq(1), NextState("IDLE")]
        if zq:
            finish = [If(wants_zqcs, zqcs_executer.start.eq(1), NextState("DO-ZQCS")
                         ).Else(*finish)]
        fsm.act("DO-REFRESH",
            cmd.valid.eq(1),
            If(executer.done,
                If(left != 1,       # the next refresh of a forced burst, back to back
                    executer.start.eq(1),
                    NextValue(left, left - 1)
                ).Else(*finish)
            )
        )
        if zq:
            fsm.act("DO-ZQCS",
                cmd.valid.eq(1),
                If(zqcs_executer.done,
                    cmd.valid.eq(0),
                    cmd.last.eq(1),
                    NextState("IDLE")
                )
            )


def idle_refresher(ahead=8, behind=8, burst=2, min_idle=4):
    """ControllerSettings(refresh_cls=...) for IdleRefresher with these options (postponing
    is ignored: behind and burst replace it)."""
    class Configured(IdleRefresher):
        def __init__(self, settings, clk_freq, zqcs_freq=1e0, postponing=1):
            IdleRefresher.__init__(self, settings, clk_freq, zqcs_freq, postponing,
                                   ahead=ahead, behind=behind, burst=burst, min_idle=min_idle)
    Configured.__name__ = "Refresher"
    return Configured
