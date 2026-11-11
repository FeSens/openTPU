"""The BCU1525's thermal governor: otpu_bcu_gov's own testbench (sim/verilator/tb_bcu_gov.sv), the CSR gate
in front of a DIMM it turns off (otpu_axil_off, tb_axil_off.sv), the DMA gate (otpu_axi_off,
tb_axi_off.sv), and
the host's side of it (opentpu/host/gov.py) against a model of its registers."""
import subprocess

import pytest

from opentpu.host import gov as G


def test_tb_bcu_gov():
    from opentpu import rtlsim
    exe = rtlsim.build("tb_bcu_gov", [rtlsim.RTL / "boards/bcu1525/otpu_bcu_gov.sv",
                                      rtlsim.TB / "tb_bcu_gov.sv"])
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    assert "PASS" in r.stdout and "FAIL" not in r.stdout, r.stdout[-2000:]


def test_codes():
    """UltraScale+'s transfer function (UG580; regs.temp_c's), both ways, and the build's limits."""
    assert G.code_of(80.0) == 724 and G.code_of(55.0) == 674 and G.code_of(65.0) == 694
    assert G.code_of(70.0) == 704
    assert abs(G.celsius(724) - 79.87) < 0.01 and abs(G.celsius(674) - 55.00) < 0.01
    for c in (25.0, 49.3, 79.9):
        assert abs(G.celsius(G.code_of(c)) - c) <= 0.25


class FakeGov:
    """otpu_bcu_gov's registers, enough for the host's side: GO's conditions, the causes; with
    `reg` (quarter degrees) a HAS_REG build's regulator."""
    def __init__(self, code=660, reg=None):
        self.code, self.run, self.ctrl, self.cause, self.kicks = code, 0, 0x100, 0, 0
        self.trip, self.resume, self.hb = 724, 674, 2000
        self.reg, self.reg_age = reg, 4
        self.reg_trip, self.reg_resume, self.reg_stale = 312, 300, 10000
        self.temp_stale, self.temp_age = 0, 0                   # 0: a build before TEMP_AGE
        self.writes = []

    def reg_cool(self):
        return self.reg is None or self.reg_age < self.reg_stale and self.reg <= self.reg_resume

    def reg_read(self, a):
        o = a - G.R_GOV
        if o == G.ID:
            return G.GOV_ID
        if o == G.STATUS:
            cool = self.code <= self.resume
            st = (self.run | (cool << 2) | (1 << 3) | (1 << 4) | (self.cause << 16)
                  | ((self.ctrl & 0xF) << 8 if self.run else 0))
            if self.reg is not None:
                st |= G.ST_HAS_REG | (self.reg_age < self.reg_stale) << 13 | self.reg_cool() << 14
            return st
        if o == G.CTRL:
            return self.ctrl
        if o == G.TEMP:
            return self.code | (self.code << 16)
        if self.reg is not None and o == G.REG:
            return (0x8000 << 16) | (self.reg & 0xFFFF)
        return {G.TRIP: self.trip, G.RESUME: self.resume, G.HB_MS: self.hb, G.LIMITS:
                (704 << 16) | 724, G.REG_TRIP: self.reg_trip, G.REG_RESUME: self.reg_resume,
                G.REG_STALE: self.reg_stale, G.REG_AGE: self.reg_age,
                G.REG_LIMITS: (300 << 16) | 312, G.TEMP_STALE: self.temp_stale,
                G.TEMP_AGE: self.temp_age}.get(o, 0)

    def reg_write(self, a, v):
        o = a - G.R_GOV
        self.writes.append((o, v))
        if o == G.CTRL:
            self.ctrl = v
        elif o == G.KICK:
            self.kicks += 1
        elif o == G.CMD:
            if v & G.CMD_SAFE and self.run:
                self.run, self.cause = 0, self.cause | 0x8
            elif v & G.CMD_GO and self.code <= self.resume and self.reg_cool():
                self.run = 1
            if v & G.CMD_CLEAR:
                self.cause = 0


def test_host_side():
    t = FakeGov()
    g = G.Gov(t)
    assert g.present()
    s = g.status()
    assert not s["run"] and s["temp_c"] == pytest.approx(G.celsius(660))
    assert g.go(dimms=(1,), acc=False)
    assert t.ctrl == 0x102 and t.kicks == 1
    assert g.status()["dimms_on"] == [1]
    g.safe()
    s = g.status()
    assert not s["run"] and s["causes"] == ["host SAFE"]
    t.code = 700                                         # above RESUME: GO refused
    assert not g.go(dimms=(0, 1, 2, 3))
    with pytest.raises(ValueError):
        g.go(dimms=(4,))


def test_host_side_regulator():
    """A HAS_REG governor: the regulator's reading, age and limits in status() / limits(), GO
    refused above REG_RESUME and with a stale reading, set_limits' quarter degrees; a governor
    without it reports the regulator as no condition."""
    t = FakeGov(reg=290)                                     # 72.5 C
    g = G.Gov(t)
    s = g.status()
    assert s["has_reg"] and s["reg_ok"] and s["reg_cool"] and s["reg_c"] == 72.5
    assert s["reg_max_c"] is None and s["reg_age_ms"] == 4
    lim = g.limits()
    assert (lim["reg_trip_c"], lim["reg_resume_c"], lim["reg_stale_ms"]) == (78.0, 75.0, 10000)
    assert (lim["reg_trip_max_c"], lim["reg_resume_max_c"]) == (78.0, 75.0)
    t.reg = 304                                              # 76 C: above REG_RESUME
    assert not g.status()["reg_cool"] and not g.go(dimms=(0,))
    t.reg, t.reg_age = 290, 10000                            # a reading REG_STALE old
    s = g.status()
    assert not s["reg_ok"] and not s["reg_cool"] and not g.go(dimms=(0,))
    t.reg_age = 4
    assert g.go(dimms=(0,))
    g.set_limits(reg_trip_c=76.5, reg_resume_c=-1.25, reg_stale_ms=2000)
    assert (G.REG_TRIP, 306) in t.writes and (G.REG_RESUME, 0xFFFB) in t.writes
    assert (G.REG_STALE, 2000) in t.writes and G.quarter_c(0xFFFB) == -1.25
    assert G.CAUSES[5:7] == ("regulator hot", "regulator unread")
    s = G.Gov(FakeGov()).status()
    assert not s["has_reg"] and s["reg_cool"] and "reg_c" not in s
    assert "reg_trip_c" not in G.Gov(FakeGov()).limits()


def test_guard_reads_the_watch(tmp_path, monkeypatch):
    """The guard on a bridge with the watch: its readings from the registers (no BMC session,
    no byte on the line), the reading's age counted; a watch without readings is an error (the
    regulator unread); a bridge without the watch: the session."""
    from opentpu.host import bmc as B
    from opentpu.host import regs as R
    from opentpu.host.fake import FakeTransport
    from opentpu.host.fake_bmc import FakeBmc
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path / "otpu"))     # (the session's lock)
    t = FakeTransport(devname="fakewatch", regmap=R.CAPS2_SINCE, bmc=FakeBmc(), ddr_mts=1066)
    t.bmc.watch_age = 1500
    g = G.Guard(t)
    g.bmc_step(100.0)
    assert g.reg["reg_ctl_c"] == 52.0 and (g.reg["reg_p0_c"], g.reg["reg_p1_c"]) == (45.0, 47.0)
    assert g.reg["pin_w"] == pytest.approx(24.8, abs=0.02) and g.reg["vout0"] is None
    assert g.reg_at == pytest.approx(98.5) and g.reg_max() == 52.0
    assert t.bmc.sent == bytearray() and not t.bmc.requests
    t.bmc.watch_age = B.AGE_NONE
    g.bmc_step(105.0)
    assert g.reg_at == pytest.approx(98.5) and g.errors[-1].startswith("bmc: the watch has no")
    with pytest.raises(B.BmcError, match="no reading"):
        g.read_first(timeout_s=0.3)                         # none within the time: raises
    t.bmc.watch_age = 40
    assert g.read_first()["reg_ctl_c"] == 52.0
    t1 = FakeTransport(devname="fakewatch1", regmap=R.CAPS2_SINCE, bmc=FakeBmc(watch=False),
                       ddr_mts=1066)
    g1 = G.Guard(t1)
    g1.bmc_step(10.0)
    assert g1.reg["vout0"] == pytest.approx(0.85, abs=1e-3) and g1.reg_at == 10.0
    assert t1.bmc.requests


def test_tb_axil_off():
    from opentpu import rtlsim
    exe = rtlsim.build("tb_axil_off", [rtlsim.RTL / "shell/otpu_ctrl.sv",
                                       rtlsim.RTL / "boards/bcu1525/otpu_axil_off.sv",
                                       rtlsim.TB / "tb_axil_off.sv"])
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    assert "PASS" in r.stdout and "FAIL" not in r.stdout, r.stdout[-2000:]


def test_tb_axi_off():
    from opentpu import rtlsim
    exe = rtlsim.build("tb_axi_off", [rtlsim.RTL / "boards/bcu1525/otpu_axi_off.sv",
                                      rtlsim.TB / "tb_axi_off.sv"])
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
    assert "PASS" in r.stdout and "FAIL" not in r.stdout, r.stdout[-2000:]


class FakeCard(FakeGov):
    """FakeGov with the JTAG transport's SYSMON (mon) and boot, and a switch to stop answering."""
    def __init__(self, code=660):
        super().__init__(code)
        self.jtag_c, self.boots, self.dead = 50.0, 0, False

    def reg_read(self, a):
        if self.dead:
            raise OSError("JTAG: no answer")
        return super().reg_read(a)

    def reg_write(self, a, v):
        if self.dead:
            raise OSError("JTAG: no answer")
        super().reg_write(a, v)

    def mon(self):
        return {"t": self.jtag_c, "vccint": 0.85}

    def boot(self):
        self.boots += 1
        self.run = 0


def _guard(t, tmp_path, **kw):
    g = G.Guard(t, csv=tmp_path / "guard.csv", **kw)
    g.reg = {"reg_ctl_c": 50.0, "reg_p0_c": 45.0, "reg_p1_c": 45.0}
    g.read_bmc = lambda: dict(g.reg)
    return g


def test_guard_trips(tmp_path):
    t = FakeCard()
    g = _guard(t, tmp_path)
    assert g.g.go(dimms=(0,))
    nr = nb = 0.0
    g.kick_step(0.0)
    nr, nb = g.step(0.0, nr, nb)
    assert t.run and t.kicks == 2 and g.tripped is None
    t.jtag_c = 81.2                                          # the die over JTAG: SAFE
    nr, nb = g.step(2.0, nr, nb)
    assert not t.run and g.tripped.startswith("die 81.2") and t.boots == 0
    t.jtag_c = 50.0
    assert g.g.go(dimms=(0,))
    g.reg["reg_p1_c"] = 78.5                                 # a regulator page: SAFE
    nr, nb = g.step(6.0, nr, nb)
    assert not t.run and g.tripped == "regulator 78.5 C"
    g.reg["reg_p1_c"] = 45.0
    assert g.g.go(dimms=(0,))
    t.jtag_c = 83.0                                          # the die at 83 C: the flash's image
    nr, nb = g.step(12.0, nr, nb)
    assert t.boots == 1 and g.booted
    rows = (tmp_path / "guard.csv").read_text().splitlines()
    assert rows[0].startswith("time,run") and any("SAFE die" in r for r in rows) \
        and any("BOOT" in r for r in rows)


def test_guard_contact_and_bmc(tmp_path):
    t = FakeCard()
    g = _guard(t, tmp_path, lost_n=3)
    g.read_bmc = lambda: (_ for _ in ()).throw(OSError("BMC: no reply"))
    g.reg, g.reg_at = {}, None
    assert g.g.go(dimms=(0,))
    g.step(0.0, 0.0, 100.0)                                  # no regulator reading yet: SAFE
    assert not t.run and g.tripped == "the regulator unread"
    t.dead = True                                            # the governor stops answering
    for k in range(3):
        g.kick_step(1.0 + k)
    assert t.boots == 1 and g.booted
    g.stop()                                                 # booted: no SAFE attempted
    assert not [e for e in g.errors if e.startswith("SAFE at stop")]


def test_guard_kicks_need_readings(tmp_path):
    """Kicks go on through a slow reading, and stop once the readings have stalled."""
    t = FakeCard()
    g = _guard(t, tmp_path, stale_s=15.0)
    assert g.g.go(dimms=(0,))
    g.step(0.0, 0.0, 0.0)
    for k in range(20):                                      # 10 s inside one slow reading
        g.kick_step(0.5 * k)
    assert t.kicks == 21 and t.run
    g.kick_step(16.0)                                        # no pass for 16 s: no kick
    assert t.kicks == 21 and g.errors[-1].startswith("kick held")
    g.step(16.5, 100.0, 100.0)
    g.kick_step(17.0)
    assert t.kicks == 22


def test_guard_errors_are_bounded(tmp_path):
    """A reader stalled for hours: the kicks stay held, and `errors` keeps its last
    ERRORS_KEPT lines (it grew by one every kick_s)."""
    t = FakeCard()
    g = _guard(t, tmp_path, stale_s=15.0)
    g.step(0.0, 0.0, 0.0)
    for k in range(3 * g.ERRORS_KEPT):
        g.kick_step(16.0 + 0.5 * k)
    assert len(g.errors) == g.ERRORS_KEPT and t.kicks == 0
    assert g.errors[-1] == f"kick held: no reading for {16.0 + 0.5 * (3 * g.ERRORS_KEPT - 1):.0f} s"


def test_guard_threads(tmp_path):
    """start / stop with the three threads (the real clock); a BMC reading slower than
    `stale_s` holds back neither the die's readings nor the kicks (2026-10-26: one took 15 s)."""
    import time
    t = FakeCard()
    g = _guard(t, tmp_path, kick_s=0.05, read_s=0.05, bmc_s=0.05, stale_s=0.5)
    slow = dict(g.reg)
    g.read_bmc = lambda: (time.sleep(1.0), dict(slow))[1]
    g.reg_at = g.clock()
    assert g.g.go(dimms=(0,))
    g.start()
    time.sleep(1.5)
    assert t.run and t.kicks >= 10 and not [e for e in g.errors if e.startswith("kick held")]
    g.stop()
    assert not any(th.is_alive() for th in (g.thread, g.kthread, g.bthread))
    assert not t.run                                         # SAFE at stop


def test_guard_bmc_in_priority(tmp_path):
    """The regulator reading runs in the transport's priority section (JtagAxi's: the DRAM
    transfers wait for it), and without one where the transport has none."""
    import contextlib
    t = FakeCard()
    seen = []

    @contextlib.contextmanager
    def priority():
        seen.append("open")
        yield
        seen.append("closed")
    t.priority = priority
    g = _guard(t, tmp_path)
    g.read_bmc = lambda: seen.append("read") or {"reg_ctl_c": 60.0}
    g.bmc_step(5.0)
    assert seen == ["open", "read", "closed"] and g.reg_at == 5.0
    del t.priority
    g.bmc_step(10.0)
    assert seen[-1] == "read" and g.reg_at == 10.0


def test_host_side_temp_age():
    """A build with the staleness check: status() has the code's age and the governor's verdict
    (TEMP_AGE bit31, TEMP_STALE_MS from TEMP_STALE); one before it reads 0 there: None."""
    t = FakeGov()
    g = G.Gov(t)
    s = g.status()
    assert (s["temp_stale_ms"], s["temp_age_ms"], s["temp_stale"]) == (None, None, False)
    t.temp_stale, t.temp_age = 60000, 1234
    s = g.status()
    assert (s["temp_stale_ms"], s["temp_age_ms"], s["temp_stale"]) == (60000, 1234, False)
    t.temp_age, t.cause = G.TEMP_AGE_STALE | 61000, 0x80
    s = g.status()
    assert s["temp_age_ms"] == 61000 and s["temp_stale"]
    assert s["causes"] == ["temperature stale"] and G.CAUSES[7] == "temperature stale"
