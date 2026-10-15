"""The DDR4 bring-up tool (opentpu/host/ddr4.py) and the BCU1525's LiteDRAM core's CSR port: the
host's CSR map against the core's, the DIMM of each channel against the board top's, and the
tool on a fake card whose DIMM windows answer as the cores' CSRs."""
import re
from pathlib import Path

import pytest

from opentpu.host import ddr4
from opentpu.host import regs as R
from opentpu.host.fake import FakeTransport
from test_board import _bcu_cfg

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "boards/bcu1525/litedram"


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
    def __init__(self, hw=True, sec=0, ded=0, bist_errors=0):
        self.regs, bases = ddr4.csr_map()
        self.by_off = {a + 4 * i: (n, i, w) for n, (a, w) in self.regs.items() for i in range(w)}
        self.ident = bases["identifier_mem"]
        self.v = {"ctrl_scratch": ddr4.SCRATCH_RESET, "cal_ready": 0,
                  "sdram_dfii_control": 0xF if hw else 0xE, "ecc_enable": 1,
                  "ecc_sec_errors": sec, "ecc_ded_errors": ded, "bist_done": 0}
        self.bist_errors, self.writes = bist_errors, []

    def read(self, off):
        if self.ident <= off < self.ident + 4 * 64:
            s = b"openTPU LiteDRAM DDR4"
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


def test_selftest_checks_every_channel_of_a_two_slice_card(monkeypatch, capsys):
    """otpu-selftest's address stage walks every slice's channels (the BCU1525's four), not
    only the YPCB's two (the config stage is left out: its DRAM size is the card's, not the
    fake's 1 MiB channels)."""
    from opentpu.host import selftest

    def some(self, name, fn, stage=selftest.Runner.stage):
        return stage(self, name, fn) if name in ("link", "regs", "addr") else None
    monkeypatch.setattr(selftest.Runner, "stage", some)
    monkeypatch.setattr(selftest, "CH_BYTES", 1 << 20)
    monkeypatch.setattr(selftest, "card_transport",
                        lambda dev: FakeTransport(devname="fakebcu", regmap=4, cfg=_bcu_cfg(),
                                                  ch_bytes=1 << 20))
    selftest.main([])
    line = [s for s in capsys.readouterr().out.splitlines() if "addr" in s][0]
    assert "[PASS]" in line and "channel 3 ok" in line, line
