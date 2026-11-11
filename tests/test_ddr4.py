"""The DDR4 bring-up tool (opentpu/host/ddr4.py) and the BCU1525's LiteDRAM core's CSR port: the
host's CSR map against the core's, the DIMM of each channel against the board top's, and the
tool on a fake card whose DIMM windows answer as the cores' CSRs."""
import re
from pathlib import Path

import pytest

from opentpu.host import ddr4, ddrcal
from opentpu.host import regs as R
from opentpu.host.fake import FakeTransport
from test_board import _bcu_cfg

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "boards/bcu1525/litedram"
CORE_UDIMM = ROOT / "boards/bcu1525/litedram_udimm"


def test_host_csr_map_is_the_cores():
    """The host's copies of the core's csr.csv / sdram_init.py are the board's; its offsets are in
    the 64 KiB window (the addresses less 0xF000_0000, where the core's CPU has the CSRs); the
    core adds that base to the host's 16-bit offset (otpu_ld4.v; without it the host read the
    BIOS ROM at 0) and decodes the CSR region there."""
    for f in ("csr.csv", "sdram_init.py"):
        assert (ddr4.DATA / f).read_bytes() == (CORE / f).read_bytes(), f
    regs, bases = ddr4.csr_map()
    assert bases["bist"] == 0 and bases["identifier_mem"] == 0x2800
    assert regs["bist_start"] == (0, 1) and regs["bist_seed"] == (0x10, 2)
    assert regs["ecc_sec_errors"][0] == bases["ecc"] + 8
    assert max(a + 4 * w for a, w in regs.values()) <= R.DDR4_STRIDE
    v = (CORE / "otpu_ld4.v").read_text()
    base = min(int(b, 16) for b in re.findall(r"^csr_base,\w+,(0x[0-9a-f]+)",
                                              (CORE / "csr.csv").read_text(), re.M))
    assert base == 0xF000_0000
    for ch in ("aw", "ar"):
        assert f"assign axil_c_{ch}_payload_addr = (ctl_{ch}addr | 32'hf0000000);" in v
    assert "(shared_adr[29:14] == 16'hf000)" in v          # the CSR region, word address


def test_host_csr_map_is_the_udimm_cores():
    """The UDIMM core's (litedram_udimm/, no ECC): the host's copies are the board's; its blocks
    are where the RDIMM core's are (identifier_mem above all: data_of reads the identifier with
    either map), its ECC CSRs are there (NativePortsPlain's: enable 0), its BIST has eight lanes,
    its CSR port the same base; ports 27-bit (8 GiB), 64 data bits, eight strobes and masks, at
    DDR4-1200."""
    for f in ("csr.csv", "sdram_init.py"):
        assert (ddr4.DATA_UDIMM / f).read_bytes() == (CORE_UDIMM / f).read_bytes(), f
    regs, bases = ddr4.csr_map(ddr4.DATA_UDIMM / "csr.csv")
    assert bases == ddr4.csr_map()[1]
    assert {"ecc_enable", "ecc_clear", "ecc_sec_errors", "ecc_ded_errors"} <= set(regs)
    assert "bist_lane7_errors" in regs and "bist_lane8_errors" not in regs
    assert max(a + 4 * w for a, w in regs.values()) <= R.DDR4_STRIDE
    v = (CORE_UDIMM / "otpu_ld4.v").read_text()
    for ch in ("aw", "ar"):
        assert f"assign axil_c_{ch}_payload_addr = (ctl_{ch}addr | 32'hf0000000);" in v
    ports = (CORE_UDIMM / "ports.txt").read_text()
    for p in ("[26:0] c0_cmd_addr", "[63:0] ddram_dq", "[7:0] ddram_dqs_p", "[7:0] ddram_dm"):
        assert p in ports, p
    ns = {}
    exec((ddr4.DATA_UDIMM / "sdram_init.py").read_text(), ns)
    # DDR4-1200 (sys 150 MHz: the ISERDESE3s' CLK limit, docs/bcu1525.md section 14), CL 9
    assert not ns["phy"]["rdimm"] and ns["phy"]["databits"] == 64
    assert ns["phy"]["sys_hz"] == 150e6 and ns["phy"]["cl"] == ns["phy"]["cwl"] == 9


def test_channel_dimms_are_the_tops():
    """CHANNEL_DIMM is otpu_fpga_top_bcu's dimm() and bcu1525_top.tcl's channel -> DIMM list."""
    top = (ROOT / "rtl/boards/bcu1525/otpu_fpga_top_bcu.sv").read_text()
    expr = re.search(r"function automatic int dimm\(int j\);\s*return (.+?);", top).group(1)
    parts = re.split(r"\s*[?:]\s*", expr)            # c0 ? v0 : c1 ? v1 : default

    def dimm(j):
        for c, v in zip(parts[0:-1:2], parts[1::2]):
            if eval(c, {"j": j}):
                return eval(v, {"j": j})
        return eval(parts[-1], {"j": j})
    assert tuple(dimm(j) for j in range(4)) == ddr4.CHANNEL_DIMM
    tcl = (ROOT / "boards/bcu1525/constraints/bcu1525_top.tcl").read_text()
    pairs = re.search(r"set dimm_of \{([\d ]+)\}", tcl).group(1).split()
    assert tuple(int(d) for d in pairs[1::2]) == ddr4.CHANNEL_DIMM


class _Core:
    """A LiteDRAM DDR4 core's CSRs, as the host sees them (csr.csv's): the identifier, the
    scratch register, cal_ready, DFII control, the ECC counters, a BIST done at once (one beat
    a cycle) whose read check finds `bist_errors` wrong beats in lane 4."""
    def __init__(self, hw=True, sec=0, ded=0, bist_errors=0, udimm=False):
        self.regs, bases = ddr4.csr_map((ddr4.DATA_UDIMM if udimm else ddr4.DATA) / "csr.csv")
        self.by_off = {a + 4 * i: (n, i, w) for n, (a, w) in self.regs.items() for i in range(w)}
        self.ident = bases["identifier_mem"]
        self.name = b"openTPU LiteDRAM DDR4" + (b" UDIMM" if udimm else b"")
        self.v = {"ctrl_scratch": ddr4.SCRATCH_RESET, "cal_ready": 0,
                  "sdram_dfii_control": 0xF if hw else 0xE, "ecc_enable": 0 if udimm else 1,
                  "ecc_sec_errors": sec, "ecc_ded_errors": ded, "bist_done": 0}
        self.bist_errors, self.writes = bist_errors, []

    def read(self, off):
        if self.ident <= off < self.ident + 4 * 64:
            s = self.name
            i = (off - self.ident) // 4
            return s[i] if i < len(s) else 0
        n, i, w = self.by_off[off]
        return (self.v.get(n, 0) >> (32 * (w - 1 - i))) & 0xFFFFFFFF

    def write(self, off, val):
        n, i, w = self.by_off[off]
        self.writes.append(n)
        sh = 32 * (w - 1 - i)
        self.v[n] = (self.v.get(n, 0) & ~(0xFFFFFFFF << sh)) | (val << sh)
        if n == "ecc_clear" and val & 1:
            self.v["ecc_sec_errors"] = self.v["ecc_ded_errors"] = 0
        if n == "bist_start" and val & 1:
            beats = self.v["bist_length"]
            self.v.update(bist_done=1, bist_ticks=beats, bist_beats=beats, bist_errors=0,
                          bist_lane4_errors=0, bist_lane_bits=0)
            if self.v["bist_mode"] & 1 and self.bist_errors:
                self.v.update(bist_errors=self.bist_errors, bist_lane4_errors=self.bist_errors,
                              bist_lane_bits=1 << 33)


class _DDR4Card(FakeTransport):
    """The fake card with the BCU1525's CFG_* registers and a LiteDRAM core per DIMM window."""
    ecc = True

    def __init__(self, cores, **kw):
        super().__init__(devname=None, regmap=4, cfg=_bcu_cfg(), ch_bytes=1 << 16, **kw)
        self.cores = cores

    def _core(self, off):
        d, o = divmod(off - R.R_DDR4, R.DDR4_STRIDE)
        return (self.cores[d], o) if off >= R.R_DDR4 else (None, off)

    def reg_read(self, off):
        c, o = self._core(off)
        return c.read(o) if c else super().reg_read(off)

    def reg_write(self, off, val):
        c, o = self._core(off)
        return c.write(o, val) if c else super().reg_write(off, val)


def test_ddr4_status_ecc_and_bist_on_a_fake_card(capsys):
    """status reads each DIMM's window (the identifier, the scratch register written and
    restored, calibrated by the BIOS's hand-over, the ECC counters) and fails on an
    uncalibrated DIMM; ecc --clear clears the counters (Board.scrub does too); a BIST with
    wrong beats fails and clears the scrub marker. Not on the YPCB's bitstream, nor on the
    board model (no `ecc`)."""
    cores = [_Core(), _Core(sec=3), _Core(hw=False), _Core(bist_errors=5)]
    t = _DDR4Card(cores)
    assert ddr4.present(t) and ddr4.dimms(t) == [0, 1, 3, 2]
    s = ddr4.status(t, 1)
    assert s["ident"] == "openTPU LiteDRAM DDR4" and s["scratch_ok"]
    assert s["scratch"] == ddr4.SCRATCH_RESET and cores[1].v["ctrl_scratch"] == ddr4.SCRATCH_RESET
    assert s["calibrated"] and s["dfii_hw"] and not s["cal_ready"] and s["ecc_sec"] == 3
    assert not ddr4.status(t, 2)["calibrated"]
    assert ddr4.main(["status"], open_transport=lambda: t) == 1
    out = capsys.readouterr().out
    assert "DIMM C3 (channel 2, slice 1): 'openTPU LiteDRAM DDR4'" in out
    assert "DIMM C2 (channel 3, slice 1)" in out and "NOT calibrated" in out
    assert ddr4.main(["status", "--dimm", "0", "--dimm", "1"], open_transport=lambda: t) == 0
    assert ddr4.ecc(t) == {0: (0, 0), 1: (3, 0), 3: (0, 0), 2: (0, 0)}
    assert ddr4.main(["ecc", "--clear"], open_transport=lambda: t) == 0
    assert ddr4.ecc(t)[1] == (0, 0)

    t.reg_write(R.R_SCRATCH, R.DRAM_INIT)
    r = ddr4.bist(t, 0, gib=1 / 1024)
    assert [x["data"] for x in r] == ["random", "address"] and all(x["ok"] for x in r)
    assert r[0]["beats"] == (1 << 20) // 64 and t.reg_read(R.R_SCRATCH) == 0
    assert r[0]["read_gbs"] == pytest.approx(200e6 * 64 / 1e9, rel=1e-3)   # a beat a cycle
    assert ddr4.main(["bist", "--dimm", "3", "--gib", "0.001"], open_transport=lambda: t) == 1
    assert "errors 5 (per lane [0, 0, 0, 0, 5, 0, 0, 0, 0], bits 0x200000000)" in \
        capsys.readouterr().out
    assert ddr4.main(["bist", "--dimm", "5"], open_transport=lambda: t) == 2

    ypcb = FakeTransport(devname=None, regmap=4)
    ypcb.ecc = True
    assert not ddr4.present(ypcb) and ddr4.ecc(ypcb) is None
    assert not ddr4.present(FakeTransport(devname=None, regmap=4, cfg=_bcu_cfg()))


def test_ddr4_tool_on_udimm_cores(capsys):
    """A card with the UDIMM cores (no ECC): the tool takes each DIMM's map by its identifier,
    status passes with the ECC off, ecc reads zeros, the BIST reports eight lanes."""
    cores = [_Core(udimm=True), _Core(udimm=True), _Core(udimm=True), _Core(udimm=True, bist_errors=2)]
    t = _DDR4Card(cores)
    assert all(ddr4.data_of(t, d) == ddr4.DATA_UDIMM for d in range(4))
    s = ddr4.status(t, 0)
    assert s["ident"] == "openTPU LiteDRAM DDR4 UDIMM" and s["scratch_ok"] and s["calibrated"]
    assert not s["ecc_enable"]
    assert ddr4.main(["status"], open_transport=lambda: t) == 0
    assert "ECC off: 0 corrected, 0 uncorrectable" in capsys.readouterr().out
    assert ddr4.ecc(t, clear=True) == {0: (0, 0), 1: (0, 0), 3: (0, 0), 2: (0, 0)}
    r = ddr4.bist(t, 1, gib=1 / 1024)
    assert all(x["ok"] and len(x["lanes"]) == 8 for x in r)
    assert ddr4.main(["bist", "--dimm", "3", "--gib", "0.001"], open_transport=lambda: t) == 1
    assert "errors 2 (per lane [0, 0, 0, 0, 2, 0, 0, 0], bits 0x200000000)" in \
        capsys.readouterr().out
    assert ddr4.data_of(_DDR4Card([_Core()] * 4), 0) == ddr4.DATA


class _UspPhy:
    """A UDIMM core's DFII and USPDDRPHY leveling CSRs by name (eight lanes, 512-tap delays). Write
    leveling (MR1 A7 and wlevel_en) feeds back CK's level at each lane's DQS: its count + tCK / 4
    taps, CK OFF[m] taps ahead. A lane writes right with its DQ delay its DQS's, the DQS within
    tCK / 8 of a CK rising edge and its write bitslip WB[m] less 2 a tCK of DQS delay (the burst
    lands a tCK later an edge later); it reads right at read bitslip RB[m], taps LO[m]..HI[m];
    a `broken` lane never does (DIMM C1's lane 5). The DQS counters have no reset (they start
    where the BIOS left them), and no delay moves with EN_VTC on."""
    TCK = 356
    STEPS = {"ddrphy_wdly_dqs_inc": ("dqs", 512), "ddrphy_wdly_dq_rst": ("dq", 0),
             "ddrphy_wdly_dq_inc": ("dq", 512), "ddrphy_wdly_dq_bitslip_rst": ("wb", 0),
             "ddrphy_wdly_dq_bitslip": ("wb", 8), "ddrphy_rdly_dq_rst": ("rd", 0),
             "ddrphy_rdly_dq_inc": ("rd", 512), "ddrphy_rdly_dq_bitslip_rst": ("rb", 0),
             "ddrphy_rdly_dq_bitslip": ("rb", 8)}

    def __init__(self, broken=()):
        self.regs, _ = ddr4.csr_map(ddr4.DATA_UDIMM / "csr.csv")
        # lane 2 has two edges, its last a tCK too late for any write bitslip; lane 4's one
        # edge is its second tCK's
        self.OFF = [0, 200, 237, 120, 300, 50, 10, 150]
        self.WB = [2, 4, 2, 6, 4, 2, 2, 2]
        self.RB = [0, 7, 0, 1, 0, 0, 7, 0]
        self.LO = [10, 0, 140, 60, 200, 0, 30, 90]
        self.HI = [150, 80, 300, 170, 290, 156, 120, 200]
        self.broken = set(broken)
        self.dqs = [221, 300, 0, 17, 400, 5, 0, 330]       # the BIOS's
        self.dq, self.wb, self.rb, self.rd = [0] * 8, [0] * 8, [0] * 8, [0] * 8
        self.sel, self.wr, self.mr, self.mr6 = 0, {}, {}, []
        self.v = {"sdram_dfii_control": 0xF, "ddrphy_en_vtc": 1, "cal_ready": 0,
                  "ddrphy_cdly_value": 53}

    def edge(self, m):
        """Lane m's DQS against CK: (taps after a CK rising edge, the nearest edge's index)."""
        x = self.dqs[m] + self.TCK // 4 + self.OFF[m]
        return x % self.TCK, (x + self.TCK // 8) // self.TCK

    def good(self, m):
        p, k = self.edge(m)
        return (m not in self.broken and self.dq[m] == self.dqs[m]
                and min(p, self.TCK - p) <= self.TCK // 8 and self.wb[m] == self.WB[m] - 2 * k
                and self.rb[m] == self.RB[m] and self.LO[m] <= self.rd[m] <= self.HI[m])

    def w(self, n, v):
        self.v[n] = v
        lanes = [m for m in range(8) if self.sel >> m & 1]
        if n == "ddrphy_dly_sel":
            self.sel = v
        elif n in self.STEPS:
            assert not self.v["ddrphy_en_vtc"], f"{n} with EN_VTC on"
            arr, wrap = self.STEPS[n]
            a = getattr(self, arr)
            for m in lanes:
                a[m] = (a[m] + 1) % wrap if wrap else 0
        elif n == "ddrphy_wlevel_strobe":
            on = self.v.get("ddrphy_wlevel_en") and self.mr.get(1, 0) & ddr4.MR1_WL
            self.v["sdram_dfii_pi0_rddata"] = sum(
                0xFF << (64 + 8 * m) for m in range(8) if on and self.edge(m)[0] < self.TCK // 2)
        elif n.startswith("sdram_dfii_pi") and n.endswith("_command_issue"):
            ph = n[len("sdram_dfii_pi")]
            c = self.v[f"sdram_dfii_pi{ph}_command"]
            if c & 0xF == 0xF:                                  # mode register set
                ba, a = self.v[f"sdram_dfii_pi{ph}_baddress"], self.v[f"sdram_dfii_pi{ph}_address"]
                self.mr[ba] = a
                if ba == 6:
                    self.mr6.append(a)
            if c & ddrcal.Dram.WRDATA:
                self.wr = {p: self.v.get(f"sdram_dfii_pi{p}_wrdata", 0) for p in range(4)}
            if c & ddrcal.Dram.RDDATA:
                bad = sum((0xFF << 8 * m) | (0xFF << (64 + 8 * m))
                          for m in range(8) if not self.good(m))
                for p in range(4):
                    self.v[f"sdram_dfii_pi{p}_rddata"] = self.wr.get(p, 0) ^ bad

    def r(self, n):
        if n == "ddrphy_wdly_dqs_inc_count":
            return self.dqs[(self.sel & -self.sel).bit_length() - 1]
        if n == "ddrphy_half_sys8x_taps":                     # module 0's DQS CNTVALUEOUT
            return (self.dqs[0] + self.TCK // 4) % 512
        n_beats = self.v.get("bist_length", 0)
        bad = [n_beats if self.v.get("bist_mode", 0) & 1 and not self.good(m) else 0
               for m in range(8)]
        if n == "bist_done":
            return 1
        if n in ("bist_beats", "bist_ticks"):
            return n_beats
        if n == "bist_errors":
            return max(bad)
        if n.startswith("bist_lane") and n.endswith("_errors"):
            return bad[int(n[len("bist_lane"):-len("_errors")])]
        return self.v.get(n, 0)


def test_ddr4_calibrate_dimm_levels_a_udimm():
    """The host's calibration on the PHY model: each lane's DQS at its scan's last edge (lane 2 at
    its earlier one: no write bitslip reaches its last), the DQ with it, the write bitslips the
    edges need, the read taps in the centre of each lane's window, then the controller's (DFII
    hardware control), EN_VTC on, write leveling off (MR1 the init's), cal_ready left 0."""
    phy = _UspPhy()
    log = []
    r = ddr4.calibrate_dimm(phy, seeds=(42,), log=log.append)
    assert r["ok"], "\n".join(log)
    assert r["tck_taps"] == 356
    assert r["edges"][2] == [30, 386] and r["edges"][4] == [323]
    assert r["wdelay"] == [267, 67, 30, 147, 323, 217, 257, 117] == phy.dqs == phy.dq
    assert r["write_latency"] == [0, 2, 0, 4, 0, 0, 0, 0] == phy.wb
    assert [x["bitslip"] for x in r["read"]] == phy.RB == phy.rb
    assert [x["tap"] for x in r["read"]] == [(lo + hi + 1) // 2 for lo, hi in zip(phy.LO, phy.HI)]
    assert all(phy.good(m) for m in range(8))
    assert phy.v["sdram_dfii_control"] == ddrcal.Dram.CTL_SEL and phy.v["ddrphy_en_vtc"] == 1
    assert phy.v["ddrphy_wlevel_en"] == 0 and phy.mr[1] == 0x301 and phy.v["cal_ready"] == 0
    assert any("m2: no write latency at the last edge: edge 30" in x for x in log)
    assert phy.mr6 == []

    phy = _UspPhy()
    ddr4.UspDram(phy, ddr4.DATA_UDIMM).set_vrefdq(ddr4.vref_value(70))
    assert phy.mr6 == [15 | ddr4.MR6_TRAIN, 15 | ddr4.MR6_TRAIN, 15]     # 69.75 %

    phy = _UspPhy(broken={5})
    r = ddr4.calibrate_dimm(phy, seeds=(42,), log=log.append)
    assert not r["ok"] and r["write_latency"][5] == -1 and r["read"][5]["taps"] == 0
    assert phy.v["sdram_dfii_control"] == 0xE and phy.v["ddrphy_en_vtc"] == 1   # software control

    with pytest.raises(ddrcal.CalError, match="RDIMM"):
        ddr4.calibrate_dimm(_UspPhy(), ddr4.DATA)


def test_ddr4_repair_lanes():
    """A lane whose DQS sits 3/16 tCK late (a write the BIOS's settings leave wrong, as DIMM C1's
    lane 6 on 2026-10-26): repair_lanes moves its write delay an eighth of tCK down (the first
    try), finds its read bitslip and a tap in its window from the BIST grids, then centres the
    write delay in the lane's write eye (the model's: the DQS within tCK / 8 of CK's edge, where
    the calibration put it), keeps every other lane as it was, and turns EN_VTC on again. A lane
    that no write change mends (`broken`) goes back to its write settings, not ok."""
    phy = _UspPhy()
    assert ddr4.calibrate_dimm(phy, seeds=(42,), log=lambda s: None)["ok"]
    others = {m: (phy.dqs[m], phy.wb[m], phy.rb[m], phy.rd[m]) for m in range(8) if m != 6}
    edge = phy.dqs[6]
    phy.v["ddrphy_en_vtc"] = 0
    d = ddr4.UspDram(phy, ddr4.DATA_UDIMM)
    dqs0 = phy.dqs[6] + 3 * phy.TCK // 16
    d.wd = [phy.dqs[m] for m in range(8)]
    d.set_wdelay(6, dqs0)
    d.set_rbitslip(6, 3)
    d.set_rdelay(6, 0)
    phy.v["ddrphy_en_vtc"] = 1
    assert not phy.good(6)
    log = []
    r = ddr4.repair_lanes(phy, [6], phy.wb, log=log.append)[6]
    assert r["ok"], "\n".join(log)
    assert r["tries"][0]["wdelay"] == dqs0 - phy.TCK // 8 and len(r["tries"]) == 1
    assert r["wdelay"] == phy.dqs[6] == phy.dq[6] and abs(r["wdelay"] - edge) <= 16
    assert 64 <= r["write_taps"] <= 2 * (phy.TCK // 8) + 16
    assert r["bitslip"] == phy.RB[6] and phy.LO[6] <= r["tap"] <= phy.HI[6] and phy.good(6)
    assert all(phy.good(m) for m in range(8)) and phy.v["ddrphy_en_vtc"] == 1
    assert {m: (phy.dqs[m], phy.wb[m], phy.rb[m], phy.rd[m]) for m in others} == others

    phy = _UspPhy(broken={5})
    assert not ddr4.calibrate_dimm(phy, seeds=(42,), log=lambda s: None)["ok"]
    wd5, wb5 = phy.dqs[5], phy.wb[5]
    r = ddr4.repair_lanes(phy, [5], phy.wb, log=lambda s: None)[5]
    assert not r["ok"] and len(r["tries"]) == len(ddr4.REPAIR_TRIES)
    assert (phy.dqs[5], phy.dq[5], phy.wb[5]) == (wd5, wd5, wb5) and phy.v["ddrphy_en_vtc"] == 1


def test_ddr4_centre_writes():
    """centre_writes: a lane whose DQS sits inside its write eye but near its late edge goes back
    to the eye's centre (the model's: CK's edge), the other lanes' delays scanned and left at
    their centres too, every lane still right, EN_VTC on."""
    phy = _UspPhy()
    assert ddr4.calibrate_dimm(phy, seeds=(42,), log=lambda s: None)["ok"]
    edges = list(phy.dqs)
    phy.v["ddrphy_en_vtc"] = 0
    d = ddr4.UspDram(phy, ddr4.DATA_UDIMM)
    d.wd = list(phy.dqs)
    d.set_wdelay(3, edges[3] + phy.TCK // 10)
    phy.v["ddrphy_en_vtc"] = 1
    assert phy.good(3)
    r = ddr4.centre_writes(phy, log=lambda s: None)
    assert r[3]["wdelay0"] == edges[3] + phy.TCK // 10 and abs(r[3]["wdelay"] - edges[3]) <= 16
    assert all(abs(phy.dqs[m] - edges[m]) <= 16 and phy.dq[m] == phy.dqs[m] for m in range(8))
    assert all(phy.good(m) for m in range(8)) and phy.v["ddrphy_en_vtc"] == 1


def test_ddr4_calibrate_command(monkeypatch, capsys):
    """otpu-ddr4 calibrate: per DIMM the calibration, then its BIST; cal_ready 1 and status
    "calibrated (host)" when both pass; a failing calibration (no BIST run) or BIST leaves the
    DIMM in software control: NOT calibrated, exit 1. SCRATCH's DRAM_INIT cleared."""
    cores = [_Core(udimm=True), _Core(udimm=True), _Core(udimm=True), _Core(udimm=True, bist_errors=4)]
    t = _DDR4Card(cores)
    seen = []

    def fake(c, data, vref=None, seeds=ddrcal.SEEDS, log=print):
        d = [i for i, x in enumerate(cores) if c.base == R.R_DDR4 + i * R.DDR4_STRIDE][0]
        seen.append((d, data, vref))
        c.w("cal_ready", 0)
        ok = d != 1
        c.w("sdram_dfii_control", 1 if ok else 0xE)
        return {"ok": ok}
    monkeypatch.setattr(ddr4, "calibrate_dimm", fake)
    t.reg_write(R.R_SCRATCH, R.DRAM_INIT)
    assert ddr4.main(["calibrate", "--dimm", "0", "--vref", "72", "--gib", "0.001"],
                     open_transport=lambda: t) == 0
    assert seen == [(0, ddr4.DATA_UDIMM, 72.0)] and cores[0].v["cal_ready"] == 1
    assert t.reg_read(R.R_SCRATCH) == 0
    assert "DIMM C0: calibration passed" in capsys.readouterr().out
    assert "calibrated (host)" in ddr4.describe(ddr4.status(t, 0), 0)
    assert "calibrated (BIOS, unchecked)" in ddr4.describe(ddr4.status(t, 2), 3)
    assert ddr4.main(["calibrate", "--dimm", "1", "--dimm", "3", "--gib", "0.001"],
                     open_transport=lambda: t) == 1
    out = capsys.readouterr().out
    assert "DIMM C1: calibration FAILED" in out and "DIMM C3: calibration FAILED" in out
    assert "BIST random: errors 4 (per lane [0, 0, 0, 0, 4, 0, 0, 0]" in out
    for d in (1, 3):
        s = ddr4.status(t, d)
        assert not s["cal_ready"] and not s["calibrated"], d
    assert ddr4.main(["status"], open_transport=lambda: t) == 1
    assert "DIMM C1 (channel 1, slice 0): 'openTPU LiteDRAM DDR4 UDIMM', scratch ok" in \
        capsys.readouterr().out


def test_selftest_checks_every_channel_of_a_two_slice_card(monkeypatch, capsys):
    """otpu-selftest's address stage walks every slice's channels (the BCU1525's four), not
    only the YPCB's two (the config stage is left out: its DRAM size is the card's, not the
    fake's 1 MiB channels)."""
    from opentpu.host import selftest

    def some(self, name, fn, stage=selftest.Runner.stage):
        return stage(self, name, fn) if name in ("link", "regs", "addr") else None
    monkeypatch.setattr(selftest.Runner, "stage", some)
    monkeypatch.setattr(selftest, "CH_BYTES", 1 << 20)
    monkeypatch.setattr(selftest, "XdmaTransport",
                        lambda dev: FakeTransport(devname="fakebcu", regmap=4, cfg=_bcu_cfg(),
                                                  ch_bytes=1 << 20))
    selftest.main([])
    line = [s for s in capsys.readouterr().out.splitlines() if "addr" in s][0]
    assert "[PASS]" in line and "channel 3 ok" in line, line
