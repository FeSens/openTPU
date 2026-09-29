"""WL7DDRPHY: LiteDRAM's A7DDRPHY (litedram/phy/s7ddrphy.py, DDR3, 1:4, no ODELAY) with write
leveling by clock groups (docs/litedram.md, section 8).

The stock PHY serializes DQ and DM on sys4x and DQS on sys4x_dqs (sys4x + 90 deg), all with
CLKDIV = sys, and shifts sys4x_dqs: DQS moves alone, against CK (tDQSS) but also against its own
DQ, so a lane's write window is the DQ-DQS eye cut by its tDQSS crossing (section 7). Here the
channel's MMCM shifts the other side: CK, the commands and the read capture move together
(fine phase shift), while each byte lane's write side (DQ, DM, DQS) stays on its group's static
clocks, DQS 90 deg after its DQ (the eye centre). So DQ and DQS move together against CK, and the
phase that works is limited only by the lanes' tDQSS crossings. (An MMCM's fine-phase outputs
must share their sub-VCO phase fraction, so DQ at 0 and DQS at 90 deg cannot both shift.)

    sys4x_ck, sys_ck    CK, commands, read ISERDES (CLK, CLKDIV): shifted by the host
    sys_w               CLKDIV of every write OSERDES (static, sys's phase)
    sys4x_w<g>          group g's DQ / DM (static, plus the group's offset)
    sys4x_w<g>_dqs      group g's DQS (sys4x_w<g> + 90 deg)

The PHY's logic runs in sys (the controller's clock). The data to and from the shifted clocks
(command serializer inputs, read deserializer outputs, their resets) go through registers on
sys's falling edge (FDREs with IS_C_INVERTED): sys_ck may sit up to half a tCK either side of sys and the
serializers still see about 2.8 ns of setup and hold (the build constrains the crossing with that
uncertainty). With the phase at 0 the timing is the stock PHY's, cycle for cycle. The tristate
controls (T1, OSERDES TQ in BUF mode, not clocked) stay as the stock PHY's.

Every domain name above except sys is the PHY's own; the SoC maps them per channel with
ClockDomainsRenamer. The CSRs and settings are A7DDRPHY's (phytype A7DDRPHY).

Derived from LiteDRAM (BSD-2-Clause): Copyright (c) 2015-2020 Florent Kermarrec, (c) 2015 Sebastien
Bourdeauducq, (c) 2021 Antmicro.
"""
from functools import reduce
from operator import or_

import math

from migen import *

from litex.soc.interconnect.csr import *

from litedram.common import *
from litedram.phy.dfi import *


class WL7DDRPHY(Module, AutoCSR):
    def __init__(self, pads, groups, sys_clk_freq, iodelay_clk_freq=200e6, cl=None, cwl=None):
        """groups[i]: byte lane i's write clock group (0 or 1)."""
        memtype, nphases = "DDR3", 4
        pads = PHYPadsCombiner(pads)
        tck = 2 / (2 * nphases * sys_clk_freq)
        addressbits = len(pads.a)
        bankbits = len(pads.ba)
        nranks = 1 if not hasattr(pads, "cs_n") else len(pads.cs_n)
        databits = len(pads.dq)
        strobes = len(pads.dqs_p)
        assert databits == 8 * strobes and len(groups) == strobes and set(groups) <= {0, 1}
        self.groups = list(groups)

        # Parameters (A7DDRPHY's) -------------------------------------------------------------------
        half_sys8x_taps = math.floor(tck / (4 * {200e6: 78e-12, 300e6: 52e-12}[iodelay_clk_freq]))
        cl = get_default_cl(memtype, tck) if cl is None else cl
        cwl = get_default_cwl(memtype, tck) if cwl is None else cwl
        cl_sys_latency = get_sys_latency(nphases, cl)
        cwl_sys_latency = get_sys_latency(nphases, cwl)
        rdphase = get_sys_phase(nphases, cl_sys_latency, cl)
        wrphase = get_sys_phase(nphases, cwl_sys_latency, cwl)
        # no ODELAY: read data one memory-clock phase into the ISERDESE2 word (as A7DDRPHY)
        phase_cycles, rdphase = divmod(rdphase + 1, nphases)
        cl_sys_latency -= phase_cycles

        # Registers (A7DDRPHY's) --------------------------------------------------------------------
        self._rst = CSRStorage()
        self._dly_sel = CSRStorage(strobes)
        self._half_sys8x_taps = CSRStorage(5, reset=half_sys8x_taps)
        self._wlevel_en = CSRStorage()
        self._wlevel_strobe = CSR()
        self._rdly_dq_rst = CSR()
        self._rdly_dq_inc = CSR()
        self._rdly_dq_bitslip_rst = CSR()
        self._rdly_dq_bitslip = CSR()
        self._wdly_dq_bitslip_rst = CSR()
        self._wdly_dq_bitslip = CSR()
        self._rdphase = CSRStorage(int(math.log2(nphases)), reset=rdphase)
        self._wrphase = CSRStorage(int(math.log2(nphases)), reset=wrphase)

        rdly_dq_rst = self._rdly_dq_rst.wr_stb
        rdly_dq_inc = self._rdly_dq_inc.wr_stb
        rdly_dq_bitslip_rst = self._rdly_dq_bitslip_rst.wr_stb
        rdly_dq_bitslip = self._rdly_dq_bitslip.wr_stb
        wlevel_strobe = self._wlevel_strobe.wr_stb
        wdly_dq_bitslip_rst = self._wdly_dq_bitslip_rst.wr_stb
        wdly_dq_bitslip = self._wdly_dq_bitslip.wr_stb

        self.settings = PhySettings(
            phytype="A7DDRPHY", memtype=memtype, databits=databits, strobes=strobes,
            dfi_databits=2 * databits, nranks=nranks, nphases=nphases,
            rdphase=self._rdphase.storage, wrphase=self._wrphase.storage, cl=cl, cwl=cwl,
            read_latency=cl_sys_latency + 6, write_latency=cwl_sys_latency - 1,
            cmd_latency=0, cmd_delay=None, write_leveling=False, write_dq_dqs_training=False,
            write_latency_calibration=True, read_leveling=True, delays=32, bitslips=8,
            with_dm=hasattr(pads, "dm"))

        self.dfi = dfi = Interface(addressbits, bankbits, nranks, 2 * databits, nphases)

        # # #

        def oserdes(d, clk, clkdiv, rst, o, t1=None, tq=None):
            kw = dict(i_TCE=1, i_T1=t1, o_TQ=tq) if t1 is not None else {}
            return Instance("OSERDESE2",
                p_SERDES_MODE="MASTER", p_DATA_WIDTH=2 * nphases, p_TRISTATE_WIDTH=1,
                p_DATA_RATE_OQ="DDR", p_DATA_RATE_TQ="BUF",
                i_RST=rst, i_CLK=clk, i_CLKDIV=clkdiv,
                **{f"i_D{n + 1}": d[n] for n in range(8)},
                i_OCE=1, o_OQ=o, **kw)

        def nreg(x):
            """x through a register on sys's falling edge, to or from the shifted sys_ck: FDREs
            with the clock inverted in the slice (a migen falling-edge domain would put a LUT in
            the clock path: 3 ns of skew in the first build)."""
            r = Signal(len(x))
            for i in range(len(x)):
                self.specials += Instance("FDRE", p_INIT=0, p_IS_C_INVERTED=1,
                                          i_C=ClockSignal("sys"), i_CE=1, i_R=0, i_D=x[i], o_Q=r[i])
            return r

        sys_rst = ResetSignal("sys") | self._rst.storage
        ck_rst = nreg(sys_rst)                  # the command serializers' reset (one bank)
        rd_rst = [nreg(sys_rst) for _ in range(strobes)]    # the read deserializers', per lane

        # Clock ---------------------------------------------------------------------------------
        for i in range(len(pads.clk_p)):
            clk_o = Signal()
            self.specials += oserdes([(0b10101010 >> n) & 1 for n in range(8)],
                                     ClockSignal("sys4x_ck"), ClockSignal("sys_ck"), ck_rst, clk_o)
            self.specials += Instance("OBUFDS", i_I=clk_o, o_O=pads.clk_p[i], o_OB=pads.clk_n[i])

        # Commands ------------------------------------------------------------------------------
        pads_ba = Signal(bankbits)
        commands = {"reset_n": "reset_n", "cs_n": "cs_n", "a": "address", pads_ba: "bank",
                    "ras_n": "ras_n", "cas_n": "cas_n", "we_n": "we_n", "cke": "cke", "odt": "odt"}
        for pad_name, dfi_name in commands.items():
            pad = pad_name if isinstance(pad_name, Signal) else getattr(pads, pad_name, None)
            if pad is None:
                assert pad_name in ("reset_n", "cs_n", "cke", "odt"), pad_name
                continue
            for i in range(len(pad)):
                self.specials += oserdes(nreg(Cat(*[getattr(dfi.phases[n // 2], dfi_name)[i]
                                                    for n in range(8)])),
                                         ClockSignal("sys4x_ck"), ClockSignal("sys_ck"), ck_rst, pad[i])
        self.comb += pads.ba.eq(pads_ba)

        # DQS -----------------------------------------------------------------------------------
        dqs_oe = Signal()
        dqs_preamble = Signal()
        dqs_postamble = Signal()
        dqs_oe_delay = TappedDelayLine(ntaps=2)
        dqs_pattern = DQSPattern(wlevel_en=self._wlevel_en.storage, wlevel_strobe=wlevel_strobe,
                                 register=True)
        self.submodules += dqs_oe_delay, dqs_pattern
        self.comb += dqs_oe_delay.input.eq(dqs_preamble | dqs_oe | dqs_postamble)
        for i in range(strobes):
            g = groups[i]
            dqs_o = Signal()
            dqs_t = Signal()
            dqs_bitslip = BitSlip(8, i=dqs_pattern.o,
                                  rst=(self._dly_sel.storage[i] & wdly_dq_bitslip_rst) | self._rst.storage,
                                  slp=self._dly_sel.storage[i] & wdly_dq_bitslip, cycles=1)
            self.submodules += dqs_bitslip
            self.specials += oserdes(dqs_bitslip.o, ClockSignal(f"sys4x_w{g}_dqs"),
                                     ClockSignal("sys_w"), sys_rst, dqs_o,
                                     t1=~dqs_oe_delay.output, tq=dqs_t)
            self.specials += Instance("IOBUFDS", i_T=dqs_t, i_I=dqs_o,
                                      io_IO=pads.dqs_p[i], io_IOB=pads.dqs_n[i])

        # DM ------------------------------------------------------------------------------------
        if hasattr(pads, "dm"):
            for i in range(databits // 8):
                g = groups[i]
                dm_i = Cat(*[dfi.phases[n // 2].wrdata_mask[n % 2 * databits // 8 + i] for n in range(8)])
                dm_o_bitslip = BitSlip(8, i=dm_i,
                                       rst=(self._dly_sel.storage[i] & wdly_dq_bitslip_rst) | self._rst.storage,
                                       slp=self._dly_sel.storage[i] & wdly_dq_bitslip, cycles=1)
                self.submodules += dm_o_bitslip
                self.specials += oserdes(dm_o_bitslip.o, ClockSignal(f"sys4x_w{g}"),
                                         ClockSignal("sys_w"), sys_rst, pads.dm[i])

        # DQ ------------------------------------------------------------------------------------
        dq_oe = Signal()
        dq_oe_delay = TappedDelayLine(ntaps=2)
        self.submodules += dq_oe_delay
        self.comb += dq_oe_delay.input.eq(dqs_preamble | dq_oe | dqs_postamble)
        for i in range(databits):
            lane = i // 8
            g = groups[lane]
            dq_o = Signal()
            dq_i = Signal()
            dq_i_delayed = Signal()
            dq_t = Signal()
            dq_o_bitslip = BitSlip(8,
                i=Cat(*[dfi.phases[n // 2].wrdata[n % 2 * databits + i] for n in range(8)]),
                rst=(self._dly_sel.storage[lane] & wdly_dq_bitslip_rst) | self._rst.storage,
                slp=self._dly_sel.storage[lane] & wdly_dq_bitslip, cycles=1)
            self.submodules += dq_o_bitslip
            self.specials += oserdes(dq_o_bitslip.o, ClockSignal(f"sys4x_w{g}"),
                                     ClockSignal("sys_w"), sys_rst, dq_o,
                                     t1=~dq_oe_delay.output, tq=dq_t)
            dq_q = Signal(8)
            dq_i_bitslip = BitSlip(8, i=nreg(dq_q),
                rst=(self._dly_sel.storage[lane] & rdly_dq_bitslip_rst) | self._rst.storage,
                slp=self._dly_sel.storage[lane] & rdly_dq_bitslip, cycles=1)
            self.submodules += dq_i_bitslip
            self.specials += Instance("ISERDESE2",
                p_SERDES_MODE="MASTER", p_INTERFACE_TYPE="NETWORKING", p_DATA_WIDTH=2 * nphases,
                p_DATA_RATE="DDR", p_NUM_CE=1, p_IOBDELAY="IFD",
                i_RST=rd_rst[lane], i_CLK=ClockSignal("sys4x_ck"), i_CLKB=~ClockSignal("sys4x_ck"),
                i_CLKDIV=ClockSignal("sys_ck"), i_BITSLIP=0, i_CE1=1, i_DDLY=dq_i_delayed,
                **{f"o_Q{n + 1}": dq_q[8 - 1 - n] for n in range(8)})
            for n in range(8):
                self.comb += dfi.phases[n // 2].rddata[n % 2 * databits + i].eq(dq_i_bitslip.o[n])
            self.specials += Instance("IDELAYE2",
                p_SIGNAL_PATTERN="DATA", p_DELAY_SRC="IDATAIN", p_CINVCTRL_SEL="FALSE",
                p_HIGH_PERFORMANCE_MODE="TRUE", p_REFCLK_FREQUENCY=iodelay_clk_freq / 1e6,
                p_PIPE_SEL="FALSE", p_IDELAY_TYPE="VARIABLE", p_IDELAY_VALUE=0,
                i_C=ClockSignal("sys"),
                i_LD=(self._dly_sel.storage[lane] & rdly_dq_rst) | self._rst.storage,
                i_LDPIPEEN=0, i_CE=self._dly_sel.storage[lane] & rdly_dq_inc, i_INC=1,
                i_IDATAIN=dq_i, o_DATAOUT=dq_i_delayed)
            self.specials += Instance("IOBUF", i_I=dq_o, o_O=dq_i, i_T=dq_t, io_IO=pads.dq[i])

        # Read Control Path (A7DDRPHY's) --------------------------------------------------------
        rddata_en = TappedDelayLine(
            signal=reduce(or_, [dfi.phases[i].rddata_en for i in range(nphases)]),
            ntaps=self.settings.read_latency)
        self.submodules += rddata_en
        self.comb += [phase.rddata_valid.eq(rddata_en.output | self._wlevel_en.storage)
                      for phase in dfi.phases]

        # Write Control Path (A7DDRPHY's) -------------------------------------------------------
        wrtap = cwl_sys_latency - 1
        wrdata_en = TappedDelayLine(
            signal=reduce(or_, [dfi.phases[i].wrdata_en for i in range(nphases)]),
            ntaps=wrtap + 2)
        self.submodules += wrdata_en
        self.comb += dq_oe.eq(wrdata_en.taps[wrtap])
        self.comb += If(self._wlevel_en.storage, dqs_oe.eq(1)).Else(dqs_oe.eq(dq_oe))
        self.comb += dqs_preamble.eq(wrdata_en.taps[wrtap - 1] & ~wrdata_en.taps[wrtap + 0])
        self.comb += dqs_postamble.eq(wrdata_en.taps[wrtap + 1] & ~wrdata_en.taps[wrtap + 0])
