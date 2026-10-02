"""LiteDRAM's multiplexer with three scheduling options (docs/litedram.md section 11, "The chooser
and the turnarounds"), for the production core (gen_core.py) and its simulation model (gen_ldc.py);
ctl_settings.py's MULTIPLEXER sets them (the core's: FASTMUX, all three; all off: LiteDRAM's
multiplexer, byte for byte):

- rtw: the read-to-write turnaround as a command spacing in controller cycles, counted from the
  last read the multiplexer issued (a tXXDController, like tCCD's and tWTR's). The multiplexer
  goes from READ straight to WRITE and holds the writes there until the spacing has passed; its
  bank machines' activates and precharges go on meanwhile. None: LiteDRAM's turnaround, an RTW
  state of read_latency - 1 cycles in which nothing is issued (9 cycles from the last read to the
  first write at our read latency of 8; its TODO: "actual limit is around (cl+1)/nphases"). The
  PHY needs 3 (the docs: its output enables switch a whole sys cycle before the write data).
- direct_wtr: the write-to-read turnaround the same way: from WRITE straight to READ, the reads
  held there by LiteDRAM's own tWTR counter (tWTR + CWL's cycles + tCCD from the last write: 5),
  instead of a WTR state that waits for it and adds a transition (6 cycles).
- same_cycle: the command choosers grant a valid request in the cycle it is valid. LiteDRAM's
  _CommandChooser keeps its round-robin grant in a register and moves it only when the granted
  request is accepted or not valid, so a grant on a bank with nothing to issue in the current
  state (a read in WRITE, a bank waiting on its own timers) costs a cycle with no command. Here
  the grant is the first valid request from a round-robin pointer on (a priority encoder in front
  of the command mux, in LiteDRAM's sys domain), and the pointer moves past the grant when it is
  accepted.

    with tuned_multiplexer(rtw=3, same_cycle=True, direct_wtr=True): ...   # cores built inside

Derived from LiteDRAM's litedram/core/multiplexer.py at core.json's commit (BSD-2-Clause):
Copyright (c) 2015 Sebastien Bourdeauducq, (c) 2016-2019 Florent Kermarrec, (c) 2018 John Sully.
"""
import math
from contextlib import contextmanager
from functools import reduce
from operator import or_, and_

from migen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import AutoCSR

import litedram.core.controller
from litedram.common import *
from litedram.core.bandwidth import Bandwidth
from litedram.core.multiplexer import (_CommandChooser, _Steerer, STEER_NOP, STEER_CMD,
                                       STEER_REQ, STEER_REFRESH)


class SameCycleChooser(Module):
    """_CommandChooser's interface and request filter; the grant is combinational: the first valid
    request from the pointer on, in round-robin order."""
    def __init__(self, requests):
        self.want_reads = Signal()
        self.want_writes = Signal()
        self.want_cmds = Signal()
        self.want_activates = Signal()

        a = len(requests[0].a)
        ba = len(requests[0].ba)

        # cas/ras/we are 0 when valid is inactive
        self.cmd = cmd = stream.Endpoint(cmd_request_rw_layout(a, ba))

        # # #

        n = len(requests)

        valids = Signal(n)
        for i, request in enumerate(requests):
            is_act_cmd = request.ras & ~request.cas & ~request.we
            command = request.is_cmd & self.want_cmds & (~is_act_cmd | self.want_activates)
            read = request.is_read == self.want_reads
            write = request.is_write == self.want_writes
            self.comb += valids[i].eq(request.valid & (command | (read & write)))

        ptr = Signal(max=max(2, n))
        grant = Signal(max=max(2, n))
        cases = {}
        for p in range(n):
            sw = [grant.eq(p)]
            for j in reversed(range(p, p + n)):
                sw = [If(valids[j % n], grant.eq(j % n)).Else(*sw)]
            cases[p] = sw
        self.comb += Case(ptr, cases)
        self.grant = grant

        choices = Array(valids[i] for i in range(n))
        self.comb += cmd.valid.eq(choices[grant])

        for name in ["a", "ba", "is_read", "is_write", "is_cmd"]:
            choices = Array(getattr(req, name) for req in requests)
            self.comb += getattr(cmd, name).eq(choices[grant])

        for name in ["cas", "ras", "we"]:
            # we should only assert those signals when valid is 1
            choices = Array(getattr(req, name) for req in requests)
            self.comb += If(cmd.valid, getattr(cmd, name).eq(choices[grant]))

        for i, request in enumerate(requests):
            self.comb += If(cmd.valid & cmd.ready & (grant == i), request.ready.eq(1))

        # past the accepted request; a valid grant not yet accepted stays
        self.sync += If(cmd.valid & cmd.ready,
                        If(grant == n - 1, ptr.eq(0)).Else(ptr.eq(grant + 1))
                     ).Elif(cmd.valid, ptr.eq(grant))

    # helpers
    def accept(self):
        return self.cmd.valid & self.cmd.ready

    def activate(self):
        return self.cmd.ras & ~self.cmd.cas & ~self.cmd.we

    def write(self):
        return self.cmd.is_write

    def read(self):
        return self.cmd.is_read


class Multiplexer(Module, AutoCSR):
    """litedram.core.multiplexer.Multiplexer (its name, so the netlist's names stay) with rtw,
    same_cycle and direct_wtr (the module doc)."""
    def __init__(self,
            settings,
            bank_machines,
            refresher,
            dfi,
            interface,
            rtw=None,
            same_cycle=False,
            direct_wtr=False):
        assert(settings.phy.nphases == len(dfi.phases))
        assert rtw is None or rtw >= 1

        ras_allowed = Signal(reset=1)
        cas_allowed = Signal(reset=1)

        # Read/Write Cmd/Dat phases ----------------------------------------------------------------
        nphases = settings.phy.nphases
        rdphase = settings.phy.rdphase
        wrphase = settings.phy.wrphase
        if isinstance(rdphase, Signal):
            rdcmdphase = Signal.like(rdphase)
            self.comb += rdcmdphase.eq(rdphase - 1) # Implicit %nphases.
        else:
            rdcmdphase = (rdphase - 1)%nphases
        if isinstance(rdphase, Signal):
            wrcmdphase = Signal.like(wrphase)
            self.comb += wrcmdphase.eq(wrphase - 1) # Implicit %nphases.
        else:
            wrcmdphase = (wrphase - 1)%nphases

        # Command choosing -------------------------------------------------------------------------
        requests = [bm.cmd for bm in bank_machines]
        chooser = SameCycleChooser if same_cycle else _CommandChooser
        self.submodules.choose_cmd = choose_cmd = chooser(requests)
        self.submodules.choose_req = choose_req = chooser(requests)
        if settings.phy.nphases == 1:
            # When only 1 phase, use choose_req for all requests
            choose_cmd = choose_req
            self.comb += choose_req.want_cmds.eq(1)
            self.comb += choose_req.want_activates.eq(ras_allowed)

        # Command steering -------------------------------------------------------------------------
        nop = Record(cmd_request_layout(settings.geom.addressbits,
                                        log2_int(len(bank_machines))))
        # nop must be 1st
        commands = [nop, choose_cmd.cmd, choose_req.cmd, refresher.cmd]
        steerer = _Steerer(commands, dfi)
        self.submodules += steerer
        if hasattr(refresher, "idle"):        # idlerefresh.py's: no bank machine has a request
            self.comb += refresher.idle.eq(~reduce(or_, [bm.req.valid | bm.req.lock
                                                         for bm in bank_machines]))

        # tRRD timing (Row to Row delay) -----------------------------------------------------------
        self.submodules.trrdcon = trrdcon = tXXDController(settings.timing.tRRD)
        self.comb += trrdcon.valid.eq(choose_cmd.accept() & choose_cmd.activate())

        # tFAW timing (Four Activate Window) -------------------------------------------------------
        self.submodules.tfawcon = tfawcon = tFAWController(settings.timing.tFAW)
        self.comb += tfawcon.valid.eq(choose_cmd.accept() & choose_cmd.activate())

        # RAS control ------------------------------------------------------------------------------
        self.comb += ras_allowed.eq(trrdcon.ready & tfawcon.ready)

        # tCCD timing (Column to Column delay) -----------------------------------------------------
        self.submodules.tccdcon = tccdcon = tXXDController(settings.timing.tCCD)
        self.comb += tccdcon.valid.eq(choose_req.accept() &
                                      (choose_req.write() | choose_req.read()))

        # CAS control ------------------------------------------------------------------------------
        self.comb += cas_allowed.eq(tccdcon.ready)

        # tWTR timing (Write to Read delay) --------------------------------------------------------
        write_latency = math.ceil(settings.phy.cwl / settings.phy.nphases)
        self.submodules.twtrcon = twtrcon = tXXDController(
            settings.timing.tWTR + write_latency +
            # tCCD must be added since tWTR begins after the transfer is complete
            settings.timing.tCCD if settings.timing.tCCD is not None else 0)
        self.comb += twtrcon.valid.eq(choose_req.accept() & choose_req.write())

        # tRTW timing (Read to Write delay; rtw) ---------------------------------------------------
        wr_allowed = cas_allowed                # the writes' CAS control (LiteDRAM's: cas_allowed)
        if rtw is not None:
            self.submodules.trtwcon = trtwcon = tXXDController(rtw)
            self.comb += trtwcon.valid.eq(choose_req.accept() & choose_req.read())
            wr_allowed = cas_allowed & trtwcon.ready

        # tWTR in READ (direct_wtr) --------------------------------------------------------------
        rd_allowed = cas_allowed                # the reads' CAS control (LiteDRAM's: cas_allowed)
        if direct_wtr:
            rd_allowed = cas_allowed & twtrcon.ready

        # Read/write turnaround --------------------------------------------------------------------
        read_available = Signal()
        write_available = Signal()
        reads = [req.valid & req.is_read for req in requests]
        writes = [req.valid & req.is_write for req in requests]
        self.comb += [
            read_available.eq(reduce(or_, reads)),
            write_available.eq(reduce(or_, writes))
        ]

        # Anti Starvation --------------------------------------------------------------------------

        def anti_starvation(timeout):
            en = Signal()
            max_time = Signal()
            if timeout:
                t = timeout - 1
                time = Signal(max=t+1)
                self.comb += max_time.eq(time == 0)
                self.sync += If(~en,
                        time.eq(t)
                    ).Elif(~max_time,
                        time.eq(time - 1)
                    )
            else:
                self.comb += max_time.eq(0)
            return en, max_time

        read_time_en,   max_read_time = anti_starvation(settings.read_time)
        write_time_en, max_write_time = anti_starvation(settings.write_time)

        # Refresh ----------------------------------------------------------------------------------
        self.comb += [bm.refresh_req.eq(refresher.cmd.valid) for bm in bank_machines]
        go_to_refresh = Signal()
        bm_refresh_gnts = [bm.refresh_gnt for bm in bank_machines]
        self.comb += go_to_refresh.eq(reduce(and_, bm_refresh_gnts))

        # Datapath ---------------------------------------------------------------------------------
        all_rddata = [p.rddata for p in dfi.phases]
        all_wrdata = [p.wrdata for p in dfi.phases]
        all_wrdata_mask = [p.wrdata_mask for p in dfi.phases]
        self.comb += [
            interface.rdata.eq(Cat(*all_rddata)),
            Cat(*all_wrdata).eq(interface.wdata),
            Cat(*all_wrdata_mask).eq(~interface.wdata_we)
        ]

        def steerer_sel(steerer, access):
            assert access in ["read", "write"]
            r = []
            for i in range(nphases):
                r.append(steerer.sel[i].eq(STEER_NOP))
                if access == "read":
                    r.append(If(i == rdphase,    steerer.sel[i].eq(STEER_REQ)))
                    r.append(If(i == rdcmdphase, steerer.sel[i].eq(STEER_CMD)))
                if access == "write":
                    r.append(If(i == wrphase,    steerer.sel[i].eq(STEER_REQ)))
                    r.append(If(i == wrcmdphase, steerer.sel[i].eq(STEER_CMD)))
            return r

        # Control FSM ------------------------------------------------------------------------------
        to_write = "RTW" if rtw is None else "WRITE"
        self.submodules.fsm = fsm = FSM()
        fsm.act("READ",
            read_time_en.eq(1),
            choose_req.want_reads.eq(1),
            If(settings.phy.nphases == 1,
                choose_req.cmd.ready.eq(rd_allowed & (~choose_req.activate() | ras_allowed))
            ).Else(
                choose_cmd.want_activates.eq(ras_allowed),
                choose_cmd.cmd.ready.eq(~choose_cmd.activate() | ras_allowed),
                choose_req.cmd.ready.eq(rd_allowed)
            ),
            steerer_sel(steerer, access="read"),
            If(write_available,
                # TODO: switch only after several cycles of ~read_available?
                If(~read_available | max_read_time,
                    NextState(to_write)
                )
            ),
            If(go_to_refresh,
                NextState("REFRESH")
            )
        )
        fsm.act("WRITE",
            write_time_en.eq(1),
            choose_req.want_writes.eq(1),
            If(settings.phy.nphases == 1,
                choose_req.cmd.ready.eq(wr_allowed & (~choose_req.activate() | ras_allowed))
            ).Else(
                choose_cmd.want_activates.eq(ras_allowed),
                choose_cmd.cmd.ready.eq(~choose_cmd.activate() | ras_allowed),
                choose_req.cmd.ready.eq(wr_allowed),
            ),
            steerer_sel(steerer, access="write"),
            If(read_available,
                If(~write_available | max_write_time,
                    NextState("READ" if direct_wtr else "WTR")
                )
            ),
            If(go_to_refresh,
                NextState("REFRESH")
            )
        )
        fsm.act("REFRESH",
            steerer.sel[0].eq(STEER_REFRESH),
            refresher.cmd.ready.eq(1),
            If(refresher.cmd.last,
                NextState("READ")
            )
        )
        if not direct_wtr:
            fsm.act("WTR",
                If(twtrcon.ready,
                    NextState("READ")
                )
            )
        if rtw is None:
            # TODO: reduce this, actual limit is around (cl+1)/nphases
            fsm.delayed_enter("RTW", "WRITE", settings.phy.read_latency-1)

        if settings.with_bandwidth:
            data_width = settings.phy.dfi_databits*settings.phy.nphases
            self.submodules.bandwidth = Bandwidth(self.choose_req.cmd, data_width)


@contextmanager
def tuned_multiplexer(rtw=None, same_cycle=False, direct_wtr=False):
    """LiteDRAMControllers built inside take this Multiplexer with these options."""
    stock = litedram.core.controller.Multiplexer

    class Tuned(Multiplexer):
        def __init__(self, *args, **kw):
            Multiplexer.__init__(self, *args, rtw=rtw, same_cycle=same_cycle,
                                 direct_wtr=direct_wtr, **kw)
    Tuned.__name__ = "Multiplexer"
    litedram.core.controller.Multiplexer = Tuned
    try:
        yield
    finally:
        litedram.core.controller.Multiplexer = stock
