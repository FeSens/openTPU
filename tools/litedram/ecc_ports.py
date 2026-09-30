"""LiteDRAMNativePortECC for the native ports of one crossbar, with one read decoder.

LiteDRAMCrossbar gives every master the controller's read data (master.rdata.data =
controller.rdata; only the valids are per master), so the ports' read paths decode the same bus:
one decoder (LiteDRAMNativePortECCR) serves them all, and each port keeps the register behind it.
Each port keeps its own encoder (its write data is its own) with its register. The encoding, the
latencies (one register on each path) and the CSRs (enable, clear, sec_errors, ded_errors: every
port's reads counted) are one LiteDRAMNativePortECC's, so a core with two ports per channel has
the one-port core's CSR map (gen_core.py, gen_ldc.py).
"""
from functools import reduce
from operator import or_

from migen import Module, If
from litex.soc.interconnect.csr import AutoCSR, CSR, CSRStatus, CSRStorage
from litex.soc.interconnect.stream import Buffer, BufferizeEndpoints, DIR_SOURCE

from litedram.common import rdata_description
from litedram.frontend.ecc import LiteDRAMNativePortECCR, LiteDRAMNativePortECCW


class NativePortsECC(Module, AutoCSR):
    def __init__(self, users, raws, burst_cycles=8):
        dw_from, dw_to = users[0].data_width, raws[0].data_width
        self.enable     = CSRStorage(reset=1)
        self.clear      = CSR()
        self.sec_errors = CSRStatus(32)
        self.ded_errors = CSRStatus(32)

        # commands straight through; write data through each port's encoder and register
        for user, raw in zip(users, raws):
            self.comb += user.cmd.connect(raw.cmd)
            w = BufferizeEndpoints({"source": DIR_SOURCE})(
                LiteDRAMNativePortECCW(dw_from, dw_to, burst_cycles))
            self.submodules += w
            self.comb += [user.wdata.connect(w.sink), w.source.connect(raw.wdata)]

        # read data: one decoder on the crossbar's read bus, a register per port
        r = LiteDRAMNativePortECCR(dw_from, dw_to, burst_cycles)
        self.submodules += r
        self.comb += [
            r.enable.eq(self.enable.storage),
            r.sink.valid.eq(reduce(or_, [raw.rdata.valid for raw in raws])),
            r.sink.data.eq(raws[0].rdata.data),
            r.source.ready.eq(1),
        ]
        for user, raw in zip(users, raws):
            buf = Buffer(rdata_description(dw_from))
            self.submodules += buf
            self.comb += [
                buf.sink.valid.eq(raw.rdata.valid), raw.rdata.ready.eq(buf.sink.ready),
                buf.sink.first.eq(raw.rdata.first), buf.sink.last.eq(raw.rdata.last),
                buf.sink.data.eq(r.source.data),
                buf.source.connect(user.rdata),
            ]

        # error counts (LiteDRAMNativePortECC's), over every port's reads
        sec_errors, ded_errors = self.sec_errors.status, self.ded_errors.status
        self.sync += [
            If(self.clear.wr_stb,
                sec_errors.eq(0),
                ded_errors.eq(0),
            ).Else(
                If((sec_errors != (2**len(sec_errors) - 1)) & (r.sec != 0),
                    sec_errors.eq(sec_errors + 1)),
                If((ded_errors != (2**len(ded_errors) - 1)) & (r.ded != 0),
                    ded_errors.eq(ded_errors + 1)),
            )
        ]
