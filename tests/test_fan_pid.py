"""boards/bcu1525/scripts/fan_pid.py: the fan's PI loop, its overrides and holds, the 80 C
ceiling, the governor's readings and the BMC watch's power through a register file, the status
file, which Vivados it steps aside for, the gains file, and the relay autotune on a simulated
card."""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import math
import os
import random
import threading
import time
import types
from collections import deque
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# the host's two cards (fan_pid's defaults: BCU_CARDn_*, read when the module loads)
os.environ.update(BCU_ENV=os.devnull, BCU_CARD0_SERIAL="SERIALCRD0", BCU_CARD0_BDF="0000:0a:00.0",
                  BCU_CARD0_NAME="CRD0", BCU_CARD1_SERIAL="SERIALCRD1", BCU_CARD1_BDF="0000:0b:00.0",
                  BCU_CARD1_NAME="CRD1")
os.environ.pop("OTPU_FAN_DIR", None)
spec = importlib.util.spec_from_file_location("fan_pid", ROOT / "boards/bcu1525/scripts/fan_pid.py")
FP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FP)
G, B = FP.G, FP.B
L11 = FP.host("i2c").to_linear11


def pid(**kw):
    return FP.Pid(**{"target": 60.0, "kp": 4.0, "ki": 0.04, "kd": 0.0, "lo": 60.0, "hi": 100.0,
                     **kw})


def test_below_the_target_the_fan_rests_at_its_lowest():
    p = pid()
    assert [p.step(t, 2.0) for t in (40, 50, 59.5)] == [60.0, 60.0, 60.0]
    assert p.i == 0.0


def test_above_the_target_the_duty_rises_and_saturates():
    p = pid()
    assert p.step(65.0, 0.0) == pytest.approx(80.0)                 # 60 + 4 x 5
    assert p.step(65.0, 2.0) > 80.0                                 # the integral grows
    assert p.step(80.0, 2.0) == 100.0


def test_no_wind_up_while_pinned_at_full():
    p = pid()
    for _ in range(500):
        p.step(75.0, 2.0)                                           # 1000 s at full duty
    i = p.i
    assert i < 1.0                                                  # P alone already pins it
    assert p.step(58.0, 2.0) < 70.0                                 # back below: off full at once


def test_steady_state_above_the_target_is_carried_by_the_integral():
    p = pid()
    for _ in range(600):
        d = p.step(61.0, 2.0)
    assert p.i == pytest.approx(40.0 - 4.0, abs=0.1)                # held at hi - lo - P
    assert d == pytest.approx(100.0)


def test_the_derivative_acts_on_the_rise():
    p = pid(kd=10.0, smooth=1.0)
    p.step(60.0, 0.0)
    assert p.step(61.0, 1.0) == pytest.approx(60 + 4 + 0.04 + 10.0)


class Fan:
    def __init__(self):
        self.duties, self.timeouts = [], []

    def set(self, d):
        self.duties.append(d)
        return {"duty": float(d)}

    def set_timeout(self, s):
        self.timeouts.append(s)


class Src:
    name = "pcie"

    def __init__(self, rs):
        self.rs = list(rs)

    def read(self):
        return self.rs.pop(0)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def loop(rs, **kw):
    kw = {"filter_s": 0, **kw}                          # the readings as they are, unless asked
    a = FP.args_of([f"--{k.replace('_', '-')}={v}" for k, v in kw.items()])
    c = Clock()
    lp = FP.Loop(Fan(), Src(rs), pid(), a, out=io.StringIO(), clock=c)
    return lp, c


def r(die, reg=None):
    return {"die_c": die, "reg_c": reg, "state": "SAFE"}


def test_overrides_set_full_duty():
    lp, c = loop([r(50.0), r(76.0), r(50.0, reg=75.5)])
    rows = []
    for _ in range(3):
        rows.append(lp.step())
        c.t += 2.0
    assert lp.fan.duties == [60, 100, 100]
    assert rows[1]["why"].startswith("die") and rows[2]["why"].startswith("regulator")


def test_without_a_reading_the_duty_holds_then_goes_full():
    lp, c = loop([r(65.0), None, None, None], blind_s=10)
    for dt in (0.0, 4.0, 4.0, 4.0):
        c.t += dt
        lp.step()
    assert lp.fan.duties == [80, 80, 80, 100]                       # 12 s blind >= 10


def test_pcie_source_reads_the_governor(tmp_path):
    regs = bytearray(G.R_GOV + 0x1000)

    def put(off, v):
        regs[G.R_GOV + off:G.R_GOV + off + 4] = v.to_bytes(4, "little")
    put(G.ID, G.GOV_ID)
    put(G.STATUS, G.ST_TEMP_VALID | G.ST_HAS_REG | G.ST_RUN | 0b0101 << 8 | G.ST_ACC_ON)
    put(G.TEMP, G.code_of(61.0) | G.code_of(70.0) << 16)
    put(G.REG, 66 * 4)
    put(G.REG_AGE, 500)
    f = tmp_path / "xdma0_user"
    f.write_bytes(bytes(regs))
    got = FP.PcieSource(str(tmp_path / "xdma0")).read()
    assert got["die_c"] == pytest.approx(61.0, abs=0.5)
    assert got["reg_c"] == 66.0
    assert got["state"] == "RUN C0 C2 acc"
    put(G.REG_AGE, 20000)                                           # a stale regulator reading
    f.write_bytes(bytes(regs))
    assert FP.PcieSource(str(tmp_path / "xdma0")).read()["reg_c"] is None
    put(G.ID, 0xFFFFFFFF)                                           # the link down
    f.write_bytes(bytes(regs))
    assert FP.PcieSource(str(tmp_path / "xdma0")).read() is None
    assert FP.PcieSource(str(tmp_path / "none")).read() is None


def test_pcie_source_reads_the_sysmon_register_only_without_the_governors(tmp_path):
    # an AXI read of the SYSMON's DRP registers can freeze the governor's temperature bus
    # (the wizard's drp_arbiter race), so the register is read only when the governor's is gone
    regs = bytearray(FP.SM_TEMP + 0x1000 & ~0xFFF)

    def put(off, v):
        regs[off:off + 4] = v.to_bytes(4, "little")
    put(G.R_GOV + G.ID, G.GOV_ID)
    put(G.R_GOV + G.STATUS, G.ST_TEMP_VALID)
    put(G.R_GOV + G.TEMP, 656 | 656 << 16)
    put(FP.SM_TEMP, G.code_of(76.0) << 6)
    f = tmp_path / "xdma0_user"
    f.write_bytes(bytes(regs))
    got = FP.PcieSource(str(tmp_path / "xdma0")).read()
    assert got["die_c"] == pytest.approx(46.05, abs=0.01) and got["sysmon_c"] is None
    assert got["state"] == "SAFE"
    put(G.R_GOV + G.STATUS, 0)                                      # the governor's not valid
    put(FP.SM_TEMP, G.code_of(70.0) << 6)
    f.write_bytes(bytes(regs))
    got = FP.PcieSource(str(tmp_path / "xdma0")).read()
    assert got["die_c"] == pytest.approx(70.0, abs=0.5) and got["gov_c"] is None
    for v in (0, 0xFFFFFFFF):                                       # and no SYSMON there
        put(FP.SM_TEMP, v)
        f.write_bytes(bytes(regs))
        assert FP.PcieSource(str(tmp_path / "xdma0")).read() is None


def test_pcie_source_takes_the_governors_own_stale_verdict(tmp_path):
    """A governor with TEMP_AGE (TEMP_STALE non-zero) says itself that its die stopped: TEMP_VALID
    clear and TEMP_AGE's bit31. Then the first reading is "gov frozen" and the SYSMON's register
    the die, without --stale-s of history; under the limit the register stays unread."""
    regs = bytearray(FP.SM_TEMP + 0x1000 & ~0xFFF)

    def put(off, v):
        regs[off:off + 4] = v.to_bytes(4, "little")
    put(G.R_GOV + G.ID, G.GOV_ID)
    put(G.R_GOV + G.TEMP, 656 | 656 << 16)
    put(G.R_GOV + G.TEMP_STALE, 60000)
    put(G.R_GOV + G.STATUS, G.ST_TEMP_VALID)
    put(G.R_GOV + G.TEMP_AGE, 59000)
    put(FP.SM_TEMP, G.code_of(76.0) << 6)
    f = tmp_path / "xdma0_user"
    f.write_bytes(bytes(regs))
    got = FP.PcieSource(str(tmp_path / "xdma0")).read()
    assert got["die_c"] == pytest.approx(46.05, abs=0.01) and got["sysmon_c"] is None
    put(G.R_GOV + G.STATUS, 0)
    put(G.R_GOV + G.TEMP_AGE, G.TEMP_AGE_STALE | 61000)
    f.write_bytes(bytes(regs))
    got = FP.PcieSource(str(tmp_path / "xdma0")).read()
    assert "gov frozen" in got["state"] and got["gov_c"] is None
    assert got["die_c"] == pytest.approx(76.0, abs=0.5)


class Watched(FP.Bar):
    """fan_pid's transport to bmc.watch, the addresses it reads kept."""
    reads: list = []

    def reg_read(self, a):
        Watched.reads.append(a)
        return super().reg_read(a)


def bar_with_watch(tmp_path, size=FP.SM_TEMP + 0x1000 & ~0xFFF, status=None, ident=B.ID_BMC2,
                   pin=(2.25, 400), vin=(12.0, 300)):
    """A BAR0 file: the governor in SAFE with C0 and C2 on and HAS_REG, the regulator at 40 C;
    the BMC2 bridge's watch with READ_PIN and READ_VIN as (value, age ms), None: no reading."""
    regs = bytearray(size)

    def put(off, v):
        regs[off:off + 4] = v.to_bytes(4, "little")
    put(G.R_GOV + G.ID, G.GOV_ID)
    put(G.R_GOV + G.STATUS, G.ST_TEMP_VALID | G.ST_HAS_REG | 0b0101 << 8 if status is None
        else status)
    put(G.R_GOV + G.TEMP, G.code_of(41.0) | G.code_of(41.0) << 16)
    put(G.R_GOV + G.REG, 40 * 4)
    put(G.R_GOV + G.REG_AGE, 500)
    put(G.R_GOV + G.REG_STALE, 10000)
    put(B.R.R_BMC + B.ID, ident)
    for off, rd in ((B.W_PIN, pin), (B.W_VIN, vin)):
        put(B.R.R_BMC + off, B.AGE_NONE << 16 if rd is None else rd[1] << 16 | L11(rd[0]))
    put(B.R.R_BMC + B.W_FW, 0x10003)
    (tmp_path / "xdma0_user").write_bytes(bytes(regs))
    return str(tmp_path / "xdma0")


def test_pcie_source_reads_the_bmc_watchs_power(tmp_path, monkeypatch):
    """VCCINT's regulator input (READ_PIN) and VIN from the BMC2 bridge's watch, with the power's
    age; through bmc.watch over the same mapping, which reads the bridge's ID and the watch's
    registers alone (never STATUS, whose read clears its flags, nor RX, whose read pops a byte)."""
    monkeypatch.setattr(FP, "Bar", Watched)
    Watched.reads = []
    got = FP.PcieSource(bar_with_watch(tmp_path)).read()
    assert got["power"] == {"vccint_in_w": 2.25, "vin_v": 12.0, "age_s": 0.4, "stale": False,
                            "firmware": 0x10003, "watch_stopped": False}
    assert (got["mode"], got["dimms"], got["acc"], got["reg_c"]) == ("SAFE", [0, 2], False, 40.0)
    assert (got["gov_frozen"], got["sysmon_frozen"], got["gov_age_s"]) == (False, False, None)
    watch = {B.R.R_BMC + o for o in (B.ID, B.W_HOT, B.W_P0, B.W_P1, B.W_CTL, B.W_PIN, B.W_VIN,
                                     B.W_FW, B.W_CNT)}
    assert Watched.reads and set(Watched.reads) <= watch
    assert B.R.R_BMC + B.STATUS not in Watched.reads and B.R.R_BMC + B.RX not in Watched.reads
    p = FP.PcieSource(bar_with_watch(tmp_path, pin=(2.5, 12000))).read()["power"]
    assert (p["vccint_in_w"], p["age_s"], p["stale"]) == (2.5, 12.0, True)  # REG_STALE old
    p = FP.PcieSource(bar_with_watch(tmp_path, pin=None)).read()["power"]
    assert (p["vccint_in_w"], p["age_s"], p["stale"], p["vin_v"]) == (None, None, None, 12.0)
    # no watch: a BMC1 bridge, a governor without HAS_REG (nothing read at 0x20000), a BAR
    # that stops short of the bridge
    assert FP.PcieSource(bar_with_watch(tmp_path, ident=B.ID_BMC)).read()["power"] is None
    Watched.reads = []
    got = FP.PcieSource(bar_with_watch(tmp_path, status=G.ST_TEMP_VALID)).read()
    assert got["power"] is None and Watched.reads == []
    got = FP.PcieSource(bar_with_watch(tmp_path, size=G.R_GOV + 0x1000)).read()
    assert got["power"] is None and got["die_c"] == pytest.approx(41.0, abs=0.5)


class Regs:
    """A BAR0 register file with the governor and the SYSMON, rewritten for each reading."""

    def __init__(self, tmp_path):
        self.b = bytearray(FP.SM_TEMP + 0x1000 & ~0xFFF)
        self.f = tmp_path / "xdma0_user"
        self.put(G.R_GOV + G.ID, G.GOV_ID)
        self.put(G.R_GOV + G.STATUS, G.ST_TEMP_VALID)

    def put(self, off, v):
        self.b[off:off + 4] = v.to_bytes(4, "little")

    def temps(self, gov, sm):
        self.put(G.R_GOV + G.TEMP, gov | gov << 16)
        self.put(FP.SM_TEMP, sm)
        self.f.write_bytes(bytes(self.b))


def test_a_reading_that_stops_moving_leaves_the_die_and_both_set_full_duty(tmp_path):
    rg, c = Regs(tmp_path), Clock()
    src = FP.PcieSource(str(tmp_path / "xdma0"), stale_s=60, clock=c)
    lp = FP.Loop(Fan(), src, pid(), FP.args_of(["--filter-s", "0"]), out=io.StringIO(),
                 clock=c)
    code = G.code_of(50.0)
    for i in range(31):                         # the governor stuck, the SYSMON's moving
        c.t = 2.0 * i
        rg.temps(656, (code + i % 2) << 6)
        row = lp.step()
        if i < 30:                              # live till --stale-s: the register unread
            assert row["sysmon_c"] is None and row["die_c"] == pytest.approx(46.05, abs=0.01)
    assert "gov frozen" in row["state"] and row["gov_c"] is None
    assert row["die_c"] == pytest.approx(50.0, abs=0.6) and row["duty"] < 100
    for i in range(31, 62):                     # the SYSMON's stuck too: lost, 100 % at once
        c.t = 2.0 * i
        rg.temps(656, code << 6)
        row = lp.step()
        if row["die_c"] is None:
            break
    assert row["die_c"] is None and row["duty"] == 100.0
    assert row["why"] == "die readings frozen 60 s" and "sysmon frozen" in row["state"]
    assert c.t - 2.0 * 30 == 60.0               # --stale-s after its last change
    c.t += 2.0                                  # moving again: back under the loop
    rg.temps(656, (code + 3) << 6)
    row = lp.step()
    assert row["die_c"] == pytest.approx(51.5, abs=0.6) and row["why"] == ""


def test_auto_goes_to_jtag_when_the_governor_readings_freeze():
    lost = {"die_c": None, "reg_c": None, "lost": "die readings frozen 60 s", "state": "SAFE"}
    pc = types.SimpleNamespace(read=lambda: lost)
    jt = {"r": None}
    a = FP.AutoSource(pc, lambda: types.SimpleNamespace(read=lambda: jt["r"], note="",
                                                        close=lambda: None), after_s=0.0)
    assert a.read() is lost and a.name == "pcie"    # jtag starting: the lost reading, 100 %
    jt["r"] = r(52.0)
    assert a.read()["die_c"] == 52.0 and a.name == "jtag"


def test_the_failsafe_defaults():
    a = FP.args_of([])
    assert a.blind_s == 60.0 and a.stale_s == 60.0


def test_steps_aside_only_for_other_vivados():
    me = 100
    ps = [(200, "vivado", me, 200),        # our reader's wrapper, its own session
          (201, "vivado", 200, 200),       # its Vivado below it
          (202, "hw_server", me, 202),
          (300, "vivado", 1, 300)]         # a bring-up's
    own = FP.own_sessions(ps, me)
    assert FP.other_vivados(ps, own) == [300]
    assert FP.other_vivados(ps[:3], own) == []


def test_auto_takes_pcie_and_falls_back_to_jtag(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(FP.time, "monotonic", clock)
    pcie = Src([r(50.0), None, None, r(52.0)])
    made = []

    def make():
        j = types.SimpleNamespace(read=lambda: r(51.0), note="", closed=False)
        j.close = lambda: setattr(j, "closed", True)
        made.append(j)
        return j
    a = FP.AutoSource(pcie, make, after_s=5.0)
    assert a.read()["die_c"] == 50.0 and a.name == "pcie"
    assert a.read() is None and not made                            # not yet 5 s
    clock.t = 6.0
    assert a.read()["die_c"] == 51.0 and a.name == "jtag"
    assert a.read()["die_c"] == 52.0 and a.name == "pcie" and made[0].closed


def test_the_loop_follows_die_and_regulator_each_by_its_own_target():
    lp, c = loop([r(50.0, reg=74.0), r(65.0, reg=70.0), r(65.0), r(55.0, reg=69.0)],
                 reg_target=70, log_s=0)                            # 10 C above the die's 60
    rows = []
    for _ in range(4):
        rows.append(lp.step())
        c.t = 0.0                                                   # dt 0: P alone
        lp.prev_t = None
    assert lp.fan.duties == [76, 80, 80, 60]    # vreg 4 over, die 5, die 5, vreg 1 under
    assert [x["lead"] for x in rows] == [True, False, False, True]
    assert lp.max_c == 74.0                                         # the hotter reading's
    lines = lp.out.getvalue().splitlines()
    assert "err  +4.0 (vreg)" in lines[0] and "(vreg)" not in lines[1]


def test_the_tuning_follows_the_die_alone(tmp_path):
    a = tuned_args(tmp_path, "--filter-s", "0")
    lp = FP.Loop(Fan(), Src([r(44.0, reg=70.0), r(46.5, reg=50.0)]),
                 FP.Pid(a.target, a.kp, a.ki, a.kd, a.lo, a.hi), a, out=io.StringIO(),
                 clock=Clock())
    assert [lp.step()["why"], lp.step()["why"]] == ["", ""]
    assert lp.fan.duties == [55, 80]            # the regulator 15 C over its 55 does not count
    a = FP.args_of([])
    lp = FP.Loop(Fan(), Src([r(44.0, reg=70.0)]), FP.Pid(45.0, 4.0, 0.04, 0.0, 55.0, 100.0), a,
                 out=io.StringIO(), clock=Clock())
    assert lp.step()["lead"] and lp.fan.duties == [100]             # outside the tuning it does


def test_defaults_45_c_and_55_percent(tmp_path):
    a = FP.args_of(["--gains-file", str(tmp_path / "none.json")])
    assert (a.target, a.reg_target, a.lo, a.hi, a.hot_c, a.reg_hot_c) == \
        (45.0, 55.0, 55.0, 100.0, 75.0, 75.0)
    assert FP.args_of(["--target", "50"]).reg_target == 60.0       # 10 C above the die's
    assert FP.args_of(["--target", "50", "--reg-target", "52"]).reg_target == 52.0
    assert (a.kp, a.ki, a.kd) == (None, None, None)                 # resolved after parsing
    FP.resolve_gains(a, err=io.StringIO())
    p = FP.Pid(a.target, a.kp, a.ki, a.kd, a.lo, a.hi)
    assert p.step(40.0, 2.0) == 55.0                                # cool: the fan stands still
    assert p.step(46.0, 0.0) == pytest.approx(59.0)                 # 55 + 4 x 1


def test_tuning_defaults_and_checks():
    a = FP.args_of([])
    assert a.gains_file == "~/.config/otpu/fan_pid.json"
    assert (a.filter_s, a.tune, a.tune_hi, a.tune_lo, a.tune_hyst, a.tune_cycles, a.tune_rule,
            a.tune_stall_s, a.tune_max_s, a.tune_dry_run, a.tune_exit) == \
        (10.0, False, 80.0, 55.0, 1.0, 4, "tl-pi", 900.0, 3600.0, False, False)
    assert FP.args_of(["--lo", "50"]).tune_lo == 50.0               # --tune-lo follows --lo
    for bad in (["--tune-lo", "80"], ["--tune-hi", "101"], ["--tune-cycles", "1"],
                ["--filter-s", "-1"],
                ["--reg-target", "75"],                             # at --reg-hot-c
                ["--tune", "--target", "74.6", "--reg-target", "60"]):  # 75.1 above --hot-c
        with pytest.raises(SystemExit):
            FP.args_of(bad)


# --- the 80 C ceiling --------------------------------------------------------------------------

def test_80_c_is_the_ceiling_whatever_the_flags(capsys):
    lp, c = loop([r(61.0), r(80.0), r(50.0, reg=80.0)], hot_c=90, reg_hot_c=85)
    assert (lp.a.hot_c, lp.a.reg_hot_c) == (80.0, 80.0)            # clamped
    assert "ceiling" in capsys.readouterr().err
    rows = []
    for _ in range(3):
        rows.append(lp.step())
        c.t += 2.0
    assert lp.fan.duties == [64, 100, 100]
    assert (rows[1]["why"], rows[2]["why"]) == ("die >= 80 C", "regulator >= 80 C")
    lp, c = loop([r(80.0), r(50.0, reg=80.0)])
    lp.a.hot_c = lp.a.reg_hot_c = 95.0                              # past args_of's clamp
    assert [lp.step()["why"], lp.step()["why"]] == ["die >= 80 C", "regulator >= 80 C"]
    assert lp.fan.duties == [100, 100]


def tuned_args(tmp_path, *argv):
    a = FP.args_of(["--tune", "--gains-file", str(tmp_path / "fan_pid.json"), *argv])
    FP.resolve_gains(a, err=io.StringIO())
    return a


@pytest.mark.parametrize("hot", [r(80.0), r(50.0, reg=80.0)], ids=["die", "regulator"])
def test_80_c_aborts_the_tuning_and_keeps_the_gains(tmp_path, hot):
    a = tuned_args(tmp_path, "--hot-c", "90", "--reg-hot-c", "90", "--filter-s", "0")
    p0, c = FP.Pid(a.target, a.kp, a.ki, a.kd, a.lo, a.hi), Clock()
    lp = FP.Loop(Fan(), Src([r(79.5), hot, r(46.0)]), p0, a, out=io.StringIO(), clock=c)
    rows = []
    for _ in range(3):
        rows.append(lp.step())
        c.t += 2.0
    assert lp.fan.duties == [80, 100, 59]          # the relay's hi, the ceiling, the PID again
    assert rows[0]["tune"].startswith("relay hi") and rows[2]["tune"] == ""
    assert lp.tuner is None and lp.pid is p0 and (p0.kp, p0.ki, p0.kd) == (4.0, 0.04, 0.0)
    out = lp.out.getvalue()
    assert "tuning aborted" in out and ">= 80 C" in out and "nothing saved" in out
    assert not (tmp_path / "fan_pid.json").exists()


def test_the_blind_timeout_aborts_the_tuning(tmp_path):
    a = tuned_args(tmp_path, "--blind-s", "10", "--filter-s", "0")
    p0, c = FP.Pid(a.target, a.kp, a.ki, a.kd, a.lo, a.hi), Clock()
    lp = FP.Loop(Fan(), Src([r(50.0), None, None, None]), p0, a, out=io.StringIO(), clock=c)
    for dt in (0.0, 4.0, 4.0, 4.0):
        c.t += dt
        lp.step()
    assert lp.fan.duties == [80, 80, 80, 100] and lp.tuner is None and lp.pid is p0
    assert "tuning aborted: no reading for 12 s" in lp.out.getvalue()
    assert not (tmp_path / "fan_pid.json").exists()


# --- the gains file ----------------------------------------------------------------------------

def gains(tmp_path, *argv, text=None):
    f = tmp_path / "fan_pid.json"
    if isinstance(text, bytes):
        f.write_bytes(text)
    elif text is not None:
        f.write_text(text)
    a = FP.args_of(["--gains-file", str(f), *argv])
    err = io.StringIO()
    said = FP.resolve_gains(a, err=err)
    return (a.kp, a.ki, a.kd), said, err.getvalue()


def test_a_flag_beats_the_file_and_the_file_the_built_ins(tmp_path):
    f = tmp_path / "fan_pid.json"
    text = json.dumps({"kp": 2.5, "ki": 0.01, "kd": 3, "rule": "tl-pi", "ku": 8.0})
    assert gains(tmp_path, text=text) == ((2.5, 0.01, 3.0), f"kp 2.5 ki 0.01 kd 3 ({f})", "")
    g, said, _ = gains(tmp_path, "--kp", "7", text=text)
    assert g == (7.0, 0.01, 3.0)
    assert said == f"kp 7 ki 0.01 kd 3 (kp flag, ki {f}, kd {f})"
    g, said, err = gains(tmp_path, "--kp", "7", "--ki", "0.1", "--kd", "0", text="{broken")
    assert g == (7.0, 0.1, 0.0) and said.endswith("(flag)") and not err  # the file not read


def test_a_missing_or_corrupt_file_gives_the_built_ins_and_a_warning(tmp_path):
    g, said, err = gains(tmp_path)
    assert g == (4.0, 0.04, 0.0) and said.endswith("(built-in)") and "no gains file" in err
    for text in ("{not json", "[1, 2]", '{"kp": "4", "ki": 0.1, "kd": 0}',
                 '{"kp": NaN, "ki": 0.1, "kd": 0}', '{"kp": -1, "ki": 0.1, "kd": 0}',
                 '{"kp": true, "ki": 0.1, "kd": 0}', '{"kp": 1}', "", b"\xff\xfe"):
        g, said, err = gains(tmp_path, "--kd", "2", text=text)
        assert g == (4.0, 0.04, 2.0), text
        assert said.endswith("(kp built-in, ki built-in, kd flag)") and "warning" in err, text


def test_the_gains_file_is_written_whole_or_not_at_all(tmp_path, monkeypatch):
    f = tmp_path / "new/dir/fan_pid.json"
    FP.save_gains(f, {"kp": 2.0, "ki": 0.01, "kd": 0.0, "rule": "tl-pi"})
    assert os.listdir(f.parent) == ["fan_pid.json"]                 # no temporary file left
    assert FP.load_gains(f) == {"kp": 2.0, "ki": 0.01, "kd": 0.0}
    FP.save_gains(f, {"kp": 3.0, "ki": 0.02, "kd": 0.0})            # over an old one
    assert FP.load_gains(f)["kp"] == 3.0 and os.listdir(f.parent) == ["fan_pid.json"]

    def fail(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(FP.os, "replace", fail)
    with pytest.raises(OSError):
        FP.save_gains(f, {"kp": 9.0, "ki": 9.0, "kd": 9.0})
    assert os.listdir(f.parent) == ["fan_pid.json"] and FP.load_gains(f)["kp"] == 3.0


def test_the_loop_acts_on_the_filtered_temperature_and_the_overrides_on_the_raw():
    lp, c = loop([r(60.0), r(72.0), r(80.0), r(60.0)], filter_s=10, log_s=0)
    rows = []
    for _ in range(4):
        rows.append(lp.step())
        c.t += 2.0
    # alpha = 2 / (10 + 2): 60, 60 + 12 / 6, 62 + 18 / 6, 65 - 5 / 6
    assert [round(x["filt_c"], 2) for x in rows] == [60.0, 62.0, 65.0, 64.17]
    # P and I on 62 (68), the raw 80 at --hot-c (75) though 65 filtered, P and I on 64.17
    assert lp.fan.duties == [60, 68, 100, 78] and rows[2]["why"] == "die >= 75 C"
    assert "filt  62.0 C" in lp.out.getvalue().splitlines()[1]
    lp, c = loop([r(60.0), None, r(70.0)], filter_s=10)
    lp.step()
    c.t += 20.0
    lp.step()
    c.t += 2.0                                          # 22 s since the last reading: 22 / 32
    assert lp.step()["filt_c"] == pytest.approx(60.0 + 10.0 * 22 / 32)


def test_the_csv_has_the_tuning_column_and_an_old_file_keeps_its_columns(tmp_path):
    for n in (10, 11, 12):          # before the tuning column, the filter's, the power's
        old = tmp_path / f"old{n}.csv"
        old.write_text(",".join(FP.CSV_COLS[:n]) + "\n")
        lp, _ = loop([r(50.0)], csv=str(old))
        lp.step()
        lp.csv_f.close()
        assert [len(x) for x in csv.reader(old.open())] == [n, n]
    new = tmp_path / "new.csv"
    lp, _ = loop([r(50.0)], csv=str(new))
    lp.step()
    lp.csv_f.close()
    rows = list(csv.reader(new.open()))
    assert rows[0] == FP.CSV_COLS and len(rows[1]) == len(FP.CSV_COLS)
    assert dict(zip(rows[0], rows[1]))["filt_c"] == "50.0"


# --- the relay autotune ------------------------------------------------------------------------

def square(tune, periods, now=0.0, amp=1.0):
    """A square wave about the tuner's target, +-amp C, one reading a second: above it for the
    first half of each period, below for the second."""
    for per in periods:
        for k in range(per):
            tune.step(tune.target + (amp if k < per / 2 else -amp), now)
            now += 1.0
    return now


def test_the_relay_measures_the_cycle_and_does_not_count_the_first():
    t = FP.RelayTune(45.0, 0.5, 55.0, 80.0, cycles=4)
    now = square(t, [80] * 4)
    assert t.step(46.0, now) == 80.0 and not t.done     # 4 cycles done, the first not counted
    assert t.state() == "relay hi, cycle 3/4, Tu 80 s, a 1.00 C"
    now = square(t, [80], now)
    t.step(46.0, now)                                   # the 5th cycle done
    g = t.result
    ku = 4 * 12.5 / (math.pi * math.sqrt(1.0 - 0.25))   # d = (80 - 55) / 2, a = 1, h = 0.5
    assert (g["tu_s"], g["amplitude_c"], g["duty"]) == (80.0, 1.0, 67.5)
    assert g["ku"] == pytest.approx(ku, rel=1e-3)
    assert g["kp"] == pytest.approx(ku / 3.2, rel=1e-3) and g["kd"] == 0.0
    assert g["ki"] == pytest.approx(ku / 3.2 / (2.2 * 80), rel=1e-3)


def test_cycles_that_do_not_agree_end_at_the_time_limit():
    t = FP.RelayTune(45.0, 0.5, 55.0, 80.0, cycles=4, max_s=1000.0)
    square(t, [60, 100] * 10)
    assert t.result is None and t.error == "no 4 agreeing cycles in 1000 s"
    t = FP.RelayTune(45.0, 0.5, 55.0, 80.0, cycles=3)
    square(t, [80, 80, 80], amp=1.0)
    square(t, [80, 80], now=240.0, amp=3.0)              # peak-to-peaks 2 and 6: no agreement
    assert not t.done and "cycle 1/3" in t.state()


class Plant:
    """A card on its fan, first order with a dead time: C dT/dt = heat - cool(duty) (T - amb),
    the cooling rising with the duty from 60 % (below it the fan stands still and k0 is left),
    the readings dead_s late, with uniform noise of +-noise and in steps of q (the card's die:
    0.5 C steps, 1 to 1.5 C apart from one reading to the next). It is the loop's fan and its
    source both."""
    name = "pcie"

    def __init__(self, clock, heat=15.0, amb=30.0, c=120.0, k0=0.2, k1=0.08, dead_s=20.0,
                 t0=40.0, dt=0.5, noise=0.0, q=0.0, seed=0):
        self.clock, self.heat, self.amb, self.c, self.k0, self.k1 = clock, heat, amb, c, k0, k1
        self.dead_s, self.dt, self.T, self.duty = dead_s, dt, t0, 55.0
        self.seen = deque([t0] * (int(round(dead_s / dt)) + 1))
        self.noise, self.q, self.rng = noise, q, random.Random(seed)

    def cool(self, duty):
        return self.k0 + self.k1 * max(duty - 60.0, 0.0)

    def set(self, d):
        self.duty = d
        return {"duty": float(d)}

    def set_timeout(self, s):
        pass

    def read(self):
        t = self.seen[0] + (self.rng.uniform(-self.noise, self.noise) if self.noise else 0.0)
        return {"die_c": round(t / self.q) * self.q if self.q else t, "reg_c": None, "state": ""}

    def run(self, s):
        for _ in range(int(round(s / self.dt))):
            self.T += (self.heat - self.cool(self.duty) * (self.T - self.amb)) / self.c * self.dt
            self.seen.append(self.T)
            self.seen.popleft()
        self.clock.t += s

    def ultimate(self, target, filter_s=0.0, lead=0.0):
        """Ku (% per C) and Tu (s) of the plant linearized at the target, behind the loop's
        filter: K e^-Ls / ((tau s + 1)(tf s + 1)) with cool* = heat / (target - amb),
        K = k1 (target - amb) / cool*, tau = C / cool*, tf = filter_s. A relay with hysteresis
        h cycles where the phase is -pi + asin(h / a), not -pi (`lead`), and reads
        1 / (|G| cos(lead)) there."""
        cool = self.heat / (target - self.amb)
        k, tau = self.k1 * (target - self.amb) / cool, self.c / cool
        lo, hi = 1e-4, 10.0
        for _ in range(100):
            w = (lo + hi) / 2
            ph = w * self.dead_s + math.atan(w * tau) + math.atan(w * filter_s)
            lo, hi = (w, hi) if ph < math.pi - lead else (lo, w)
        g = k / math.sqrt(1 + (w * tau) ** 2) / math.sqrt(1 + (w * filter_s) ** 2)
        return 1 / (g * math.cos(lead)), 2 * math.pi / w


def tune_loop(tmp_path, *argv, **plant):
    c = Clock()
    p = Plant(c, **plant)
    a = tuned_args(tmp_path, *argv)
    return FP.Loop(p, p, FP.Pid(a.target, a.kp, a.ki, a.kd, a.lo, a.hi), a, out=io.StringIO(),
                   clock=c), p


def drive(lp, p, until, s):
    """2 s steps for s seconds or until until(lp); the plant's temperatures."""
    temps, end = [], p.clock.t + s
    while p.clock.t < end and not until(lp):
        lp.step()
        p.run(2.0)
        temps.append(p.T)
    return temps


@pytest.mark.parametrize("rule", sorted(FP.RelayTune.RULES))
def test_tuning_a_simulated_card_saves_gains_that_hold_the_target(tmp_path, rule):
    f, log = tmp_path / "fan_pid.json", tmp_path / "fan.csv"
    lp, p = tune_loop(tmp_path, "--tune-rule", rule, "--csv", str(log))
    temps = drive(lp, p, lambda lp: lp.tuner is None, 3600)
    assert lp.tuner is None and lp.exit is None and p.clock.t < 1200
    g = json.loads(f.read_text())
    ku, tu = p.ultimate(45.0, lp.a.filter_s, math.asin(g["hyst_c"] / g["amplitude_c"]))
    assert 0.5 * ku < g["ku"] < 2 * ku and 0.5 * tu < g["tu_s"] < 2 * tu   # 9.1 % per C, 132 s
    assert g["amplitude_c"] > 2 * g["hyst_c"]
    fk, fi, fd = FP.RelayTune.RULES[rule]
    assert g["kp"] == pytest.approx(fk * g["ku"], rel=1e-3)
    assert g["ki"] == pytest.approx(g["kp"] / (fi * g["tu_s"]), rel=1e-3)
    assert g["kd"] == pytest.approx(g["kp"] * fd * g["tu_s"], rel=1e-3)
    assert g["kp"] > 0 and g["ki"] > 0 and all(math.isfinite(g[k]) for k in FP.GAINS)
    assert (g["rule"], g["target_c"], g["hyst_c"], g["relay_lo"], g["relay_hi"], g["cycles"]) \
        == (rule, 45.0, 1.0, 55.0, 80.0, 4)
    assert {"tu_s", "amplitude_c", "duty", "time", "host"} <= set(g)
    assert (lp.pid.kp, lp.pid.ki, lp.pid.kd) == (g["kp"], g["ki"], g["kd"])
    assert FP.load_gains(f) == {k: g[k] for k in FP.GAINS}          # the next start's gains
    out = lp.out.getvalue()
    assert "tune: relay" in out and f"saved to {f}" in out and "going on with the new gains" in out
    tunes = [x["tune"] for x in csv.DictReader(log.open())]
    assert tunes[0].startswith("relay ") and "cycle 4/4" in tunes[-1]
    temps += drive(lp, p, lambda lp: False, 1500)                   # then the PID holds it
    assert all(abs(T - 45.0) < 0.5 for T in temps[-250:])
    assert max(temps) < 50.0


def test_tune_exit_and_dry_run(tmp_path, monkeypatch):
    f = tmp_path / "fan_pid.json"
    lp, p = tune_loop(tmp_path, "--tune-exit")
    drive(lp, p, lambda lp: lp.exit is not None, 3600)
    assert lp.exit == 0 and f.exists() and "going on" not in lp.out.getvalue()
    f.unlink()
    lp, p = tune_loop(tmp_path, "--tune-dry-run")
    drive(lp, p, lambda lp: lp.tuner is None, 3600)
    assert lp.exit is None and not f.exists() and lp.pid.kp != 4.0
    assert "not saved (--tune-dry-run)" in lp.out.getvalue()

    def fail(path, g):
        raise OSError("read-only")
    monkeypatch.setattr(FP, "save_gains", fail)
    lp, p = tune_loop(tmp_path, "--tune-exit")
    drive(lp, p, lambda lp: lp.exit is not None, 3600)
    assert lp.exit == 1 and "NOT saved (read-only)" in lp.out.getvalue()


def tune_noisy(tmp_path, *argv, noise=1.5, q=0.5, seed=0):
    """A tuning to its end, by default on the card's noise (0.5 C steps, +-1.5 C); the tuner and
    the times its relay switched."""
    lp, p = tune_loop(tmp_path, "--tune-dry-run", *argv, noise=noise, q=q, seed=seed)
    t, high, switched = lp.tuner, False, []
    while lp.tuner is not None and p.clock.t < 3600:
        now = p.clock.t
        lp.step()
        if t.high != high:
            switched.append(now)
            high = t.high
        p.run(2.0)
    return t, switched


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_on_the_cards_noise_the_tuning_converges_without_chatter(tmp_path, seed):
    clean, _ = tune_noisy(tmp_path, noise=0.0, q=0.0)
    t, switched = tune_noisy(tmp_path, seed=seed)
    assert t.result is not None, t.error
    assert min(b - a for a, b in zip(switched, switched[1:])) >= 10.0
    assert t.result["ku"] == pytest.approx(clean.result["ku"], rel=0.25)
    assert t.result["tu_s"] == pytest.approx(clean.result["tu_s"], rel=0.25)


def test_the_filter_alone_stops_the_chatter_at_the_old_hysteresis(tmp_path):
    gap = lambda sw: min(b - a for a, b in zip(sw, sw[1:]))        # noqa: E731
    _, raw = tune_noisy(tmp_path, "--filter-s", "0", "--tune-hyst", "0.5", "--tune-max-s", "600")
    _, filtered = tune_noisy(tmp_path, "--tune-hyst", "0.5")
    assert gap(raw) <= 4.0                      # 54229950's defaults: the card's chatter
    assert gap(filtered) >= 10.0


@pytest.mark.parametrize("heat, t0, hint", [(2.0, 35.0, "load the card or lower --target"),
                                            (30.0, 50.0, "raise --tune-hi or --target")],
                         ids=["below", "above"])
def test_a_stall_aborts_the_tuning_and_keeps_the_gains(tmp_path, heat, t0, hint):
    lp, p = tune_loop(tmp_path, heat=heat, t0=t0)
    p0 = lp.pid
    drive(lp, p, lambda lp: lp.tuner is None, 3600)
    assert lp.tuner is None and lp.pid is p0 and 900 <= p.clock.t < 910
    out = lp.out.getvalue()
    assert "tuning aborted: no crossing in 900 s" in out and hint in out and "nothing saved" in out
    assert not (tmp_path / "fan_pid.json").exists()


# --- the status file ---------------------------------------------------------------------------

def status_loop(tmp_path, src, *argv):
    a = FP.args_of(["--filter-s", "0", "--target", "60", "--log-s", "0", *argv])
    st = tmp_path / "bcu-fan/status.json"
    return FP.Loop(Fan(), src, pid(), a, out=io.StringIO(), clock=Clock(), status=st), st


def test_the_status_file_after_each_step(tmp_path):
    """Each step's status as JSON: the time, the source, the governor's state, the die's readings,
    the regulator's, the power with its label, the fan's duty and why, this process; None where
    a reading is missing."""
    src = FP.PcieSource(bar_with_watch(tmp_path))
    lp, st = status_loop(tmp_path, src, "--csv", str(tmp_path / "fan.csv"))
    t0 = time.time()
    lp.step()
    d = json.loads(st.read_text())
    assert t0 - 1 <= d["time"] <= time.time() + 1 and d["interval_s"] == 2.0
    assert abs(datetime.fromisoformat(d["iso"]).timestamp() - d["time"]) < 1
    assert d["source"] == "pcie"
    assert d["state"] == {"mode": "SAFE", "dimms": [0, 2], "acc": False, "text": "SAFE C0 C2"}
    assert d["die_c"] == pytest.approx(41.0, abs=0.5) and d["gov_c"] == d["die_c"]
    assert (d["sysmon_c"], d["gov_frozen"], d["sysmon_frozen"], d["reg_c"]) == \
        (None, False, False, 40.0)
    assert d["power"] == {"vccint_in_w": 2.25, "vin_v": 12.0, "age_s": 0.4, "stale": False,
                          "label": "VCCINT regulator input (READ_PIN), not the board total"}
    assert d["fan"] == {"duty": 60.0, "why": None, "mode": "pid", "target_c": 60.0,
                        "filt_c": d["die_c"], "lead": "die",
                        "gains": {"kp": lp.pid.kp, "ki": lp.pid.ki, "kd": lp.pid.kd},
                        "i": round(lp.pid.i, 2)}
    assert d["fan_pid"]["pid"] == os.getpid() and d["fan_pid"]["version"] == FP.VERSION
    assert "VCCINT in   2.2 W" in lp.out.getvalue()                     # the live line
    lp.csv_f.close()
    row = dict(zip(*list(csv.reader((tmp_path / "fan.csv").open()))))
    assert FP.CSV_COLS[-2:] == ["pin_w", "vin"] and (row["pin_w"], row["vin"]) == ("2.25", "12.0")


def test_the_status_file_without_readings(tmp_path):
    """The jtag source's die alone, then no reading at all: the rest None."""
    lp, st = status_loop(tmp_path, Src([{"die_c": 52.0, "reg_c": None, "state": ""}, None]))
    lp.step()
    d = json.loads(st.read_text())
    assert d["state"] == {"mode": None, "dimms": None, "acc": None, "text": None}
    assert d["die_c"] == 52.0 and d["gov_c"] is d["sysmon_c"] is d["reg_c"] is None
    assert d["power"]["vccint_in_w"] is d["power"]["vin_v"] is d["power"]["age_s"] is None
    assert "  --   W" in lp.out.getvalue()
    lp.step()
    d = json.loads(st.read_text())
    assert d["die_c"] is None and d["fan"]["why"] == "no reading: holding"
    assert d["fan"]["filt_c"] is None and d["fan"]["lead"] is None


def test_the_status_file_is_replaced_whole(tmp_path, monkeypatch):
    """A rename over the file: a reader beside the loop parses every version it opens; a failed
    write leaves the last one, no temporary file, says so once and the loop goes on."""
    rg = Regs(tmp_path)
    rg.temps(G.code_of(50.0), 0)
    lp, st = status_loop(tmp_path, FP.PcieSource(str(tmp_path / "xdma0")))
    lp.step()
    stop, seen, bad = threading.Event(), [], []

    def reader():
        while not stop.is_set():
            try:
                seen.append(json.loads(st.read_text())["time"])
            except ValueError as e:
                bad.append(e)
    th = threading.Thread(target=reader)
    th.start()
    for _ in range(300):
        lp.step()
    stop.set()
    th.join()
    assert seen and not bad
    assert os.listdir(st.parent) == ["status.json"]
    last = st.read_text()

    def fail(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(FP.os, "replace", fail)
    for _ in range(3):
        lp.step()
    assert st.read_text() == last and os.listdir(st.parent) == ["status.json"]
    assert lp.fan.duties[-1] == 60 and lp.out.getvalue().count("disk full") == 1
    monkeypatch.undo()
    time.sleep(0.005)                           # (the file's time is in ms: a fast host steps twice in one)
    lp.step()
    assert "written again" in lp.out.getvalue() and st.read_text() != last


def test_the_status_files_default_and_none(tmp_path):
    assert FP.args_of([]).status == "~/otpu-build/bcu-fan/status.json"
    assert FP.args_of(["--status", ""]).status == ""
    lp, _ = loop([r(50.0)])
    assert lp.status is None
    lp.step()                                       # tests and --status '': nothing written


# --- two cards, two fans -----------------------------------------------------------------------

def test_the_xdma_device_is_found_by_the_cards_pci_function(tmp_path):
    """Both cards on the bus: /dev/xdma0 can be either, so each card's device comes from its PCI
    function (the driver's node under it), looked up at each reading."""
    def bind(bdf, node):
        d = tmp_path / "bus/pci/devices" / bdf / "xdma"
        d.mkdir(parents=True)
        for k in ("user", "control", "h2c_0"):
            (d / f"{node}_{k}").mkdir()
    assert FP.xdma_of(FP.PCI, tmp_path) is None                     # off the bus / not bound
    bind(FP.PCI2, "xdma0")
    bind(FP.PCI, "xdma1")
    assert FP.xdma_of(FP.PCI, tmp_path) == "/dev/xdma1"
    assert FP.xdma_of(FP.PCI2, tmp_path) == "/dev/xdma0"
    src = FP.PcieSource(None, pci=FP.PCI, sys_root=tmp_path)
    assert src.path == "/dev/xdma1_user"
    assert FP.PcieSource(None, pci="0000:09:00.0", sys_root=tmp_path).read() is None
    assert FP.PcieSource("/dev/xdma7").path == "/dev/xdma7_user"    # --dev as given


def test_the_pcie_source_reads_its_own_cards_device(tmp_path, monkeypatch):
    rg = Regs(tmp_path)
    rg.temps(G.code_of(55.0), 0)
    d = tmp_path / "sys/bus/pci/devices" / FP.PCI2 / "xdma"
    d.mkdir(parents=True)
    (d / "xdma0_user").mkdir()
    monkeypatch.setattr(FP, "xdma_of", lambda bdf, root: str(tmp_path / "xdma0")
                        if bdf == FP.PCI2 else None)
    assert FP.PcieSource(None, pci=FP.PCI2).read()["die_c"] == pytest.approx(55.0, abs=0.5)
    assert FP.PcieSource(None, pci=FP.PCI).read() is None           # card 0 not on the bus


class Ctl:
    """The two-fan controller as fanctl's FanController: fan 2's duty is duty2."""

    def __init__(self, fans=2):
        self.d, self.fans, self.cmds = {1: 60.0, 2: 60.0}, fans, []

    def status(self):
        s = {"duty": self.d[1], "rpm": 0}
        return {**s, "duty2": self.d[2]} if self.fans == 2 else s

    def set(self, duty, fan=1):
        self.cmds.append((fan, duty))
        self.d[fan] = float(duty)
        return self.status()


def test_each_fan_takes_its_own_command():
    c = Ctl()
    p1, p2 = FP.FanPort(c, 1), FP.FanPort(c, 2)
    assert p1.set(70)["duty"] == 70.0 and p2.set(85)["duty"] == 85.0
    assert c.cmds == [(1, 70), (2, 85)] and c.d == {1: 70.0, 2: 85.0}


def test_fan_2_needs_the_two_fan_fanctl_and_firmware():
    two = types.SimpleNamespace(FanController=Ctl, __file__="fanctl.py")
    assert FP.has_fan2(two, Ctl()) == ""

    class One:
        def set(self, duty):
            pass
    one = types.SimpleNamespace(FanController=One, __file__="/home/x/fanctl.py")
    assert FP.has_fan2(one, Ctl()) == "/home/x/fanctl.py is the one-fan fanctl"
    assert FP.has_fan2(two, Ctl(fans=1)) == "the controller's firmware has one fan"


def test_fan_2s_flags():
    a = FP.args_of(["--target", "60", "--csv", "/x/fan.csv"])
    assert (a.serial2, a.pci2, a.dev2, a.target2, a.reg_target2) == \
        ("SERIALCRD1", "0000:0b:00.0", None, 60.0, 70.0)
    assert (a.serial, a.pci, a.dev) == ("SERIALCRD0", "0000:0a:00.0", None)
    assert a.status2 == "~/otpu-build/bcu-fan/status2.json" and a.csv2 == "/x/fan2.csv"
    assert not a.no_fan2 and FP.args_of(["--no-fan2"]).no_fan2
    a = FP.args_of(["--target", "60", "--target2", "50", "--reg-target", "72"])
    assert (a.target2, a.reg_target2, a.csv2) == (50.0, 62.0, None)
    with pytest.raises(SystemExit):
        FP.args_of(["--target2", "70"])                             # its vreg 80 >= 75


def test_two_loops_each_by_its_own_card(tmp_path):
    """One loop per card: each its own reading, target, regulator target, overrides, status file
    and CSV, and its lines named; a lost card sets its own fan full, not the other's."""
    c, clock = Ctl(), Clock()
    a = FP.args_of(["--filter-s", "0", "--target", "60", "--target2", "50", "--log-s", "0",
                    "--csv", str(tmp_path / "fan.csv"), "--blind-s", "4"])
    out = io.StringIO()
    s1, s2 = tmp_path / "status.json", tmp_path / "status2.json"
    l1 = FP.Loop(FP.FanPort(c, 1), Src([r(65.0), r(65.0), r(65.0)]), pid(), a, out=out,
                 clock=clock, status=s1, card="CRD0 fan 1")
    l2 = FP.Loop(FP.FanPort(c, 2), Src([r(55.0, reg=58.0), None, None]),
                 pid(target=50.0), a, out=out, clock=clock, status=s2, card="CRD1 fan 2",
                 csv_path=a.csv2, tune=False, reg_target=a.reg_target2)
    for _ in range(3):
        l1.prev_t = l2.prev_t = None                                # dt 0: P alone
        l1.step()
        l2.step()
        clock.t += 2.0
    assert c.cmds == [(1, 80), (2, 80), (1, 80), (2, 80), (1, 80), (2, 100)]
    d1, d2 = json.loads(s1.read_text()), json.loads(s2.read_text())
    assert (d1["card"], d1["fan_no"], d1["fan"]["target_c"]) == ("CRD0 fan 1", 1, 60.0)
    assert (d2["card"], d2["fan_no"], d2["fan"]["target_c"]) == ("CRD1 fan 2", 2, 50.0)
    assert d2["fan"]["duty"] == 100.0 and d2["fan"]["why"] == "no reading for 4 s"
    lines = out.getvalue().splitlines()
    assert any("CRD0 fan 1  die  65.0" in x for x in lines)
    assert any("CRD1 fan 2  die  55.0" in x for x in lines)
    l1.csv_f.close()
    l2.csv_f.close()
    assert len((tmp_path / "fan.csv").read_text().splitlines()) == 4    # the header and 3 rows
    assert len((tmp_path / "fan2.csv").read_text().splitlines()) == 4


def fake_fanctl(ctl):
    class FC:
        def __init__(self, port=None):
            self.t = []

        def __enter__(self):
            return self

        def __exit__(self, *e):
            pass

        def set_timeout(self, s):
            ctl.cmds.append(("timeout", s))

        def status(self):
            return ctl.status()

        def set(self, duty, fan=1):
            return ctl.set(duty, fan)
    return types.SimpleNamespace(FanController=FC, FanError=RuntimeError, __file__="fanctl.py")


def run_main(monkeypatch, tmp_path, ctl, steps, *argv):
    """main() with a fake controller and sources that read 60 C; KeyboardInterrupt (or `steps`'
    exception) after `steps` loop steps."""
    monkeypatch.setattr(FP, "load", lambda name, path: fake_fanctl(ctl) if name == "fanctl"
                        else pytest.fail(name))
    monkeypatch.setattr(FP.Path, "home", lambda: tmp_path)
    n = {"k": 0}

    class S:
        name = "pcie"

        def __init__(self, dev, stale_s, pci=None):
            self.pci = pci

        def read(self):
            n["k"] += 1
            if n["k"] > steps[0]:
                raise steps[1]
            return r(60.0)
    monkeypatch.setattr(FP, "PcieSource", S)
    monkeypatch.setattr(FP.time, "sleep", lambda s: None)
    return FP.main(["--source", "pcie", "--status", "", "--status2", "", "--gains-file",
                    str(tmp_path / "g.json"), "--target", "60", *argv])


def test_main_drives_both_fans_and_a_clean_stop_turns_the_watchdog_off(monkeypatch, tmp_path,
                                                                        capsys):
    c = Ctl()
    assert run_main(monkeypatch, tmp_path, c, (4, KeyboardInterrupt())) == 0
    assert c.cmds[0] == ("timeout", 15) and c.cmds[-1] == ("timeout", 0)
    assert [x[0] for x in c.cmds[1:-1]] == [1, 2, 1, 2]             # both fans each step
    out = capsys.readouterr().out
    assert "CRD0 fan 1: target die 60 C" in out and "CRD1 fan 2: target die 60 C" in out
    assert "fan 1 left at 55 %, fan 2 left at 55 %, watchdog off" in out


def test_main_on_an_error_sets_both_fans_full_and_leaves_the_watchdog_on(monkeypatch, tmp_path,
                                                                          capsys):
    c = Ctl()
    assert run_main(monkeypatch, tmp_path, c, (3, OSError("bus gone"))) == 1
    assert ("timeout", 0) not in c.cmds and c.d == {1: 100.0, 2: 100.0}
    err = capsys.readouterr().err
    assert "the fans set to 100 %, the watchdog left on" in err and "bus gone" in err


def test_main_without_fan_2_drives_fan_1_alone(monkeypatch, tmp_path, capsys):
    c = Ctl(fans=1)
    assert run_main(monkeypatch, tmp_path, c, (2, KeyboardInterrupt())) == 0
    assert [x[0] for x in c.cmds[1:-1]] == [1, 1]
    out = capsys.readouterr().out
    assert "WARNING: fan 2 not driven (the controller's firmware has one fan)" in out
    assert "CRD0 fan 1:" not in out and "fan 1 left at 55 %, watchdog off" in out
    c = Ctl()
    assert run_main(monkeypatch, tmp_path, c, (2, KeyboardInterrupt()), "--no-fan2") == 0
    assert [x[0] for x in c.cmds[1:-1]] == [1, 1] and "WARNING" not in capsys.readouterr().out


# --- the gains while it runs -------------------------------------------------------------------

def test_the_gains_set_while_running(tmp_path):
    """gains_live.json beside the status file: this fan's gains at the next step, the integral
    kept, said once in a line a laptop's guard does not read as a reading. A bad file: said once,
    the gains stay, the loop goes on; gone: the gains stay."""
    src = Src([{"die_c": 65.0, "reg_c": None}] * 16)
    lp, st = status_loop(tmp_path, src)
    live = st.parent / FP.LIVE_GAINS
    lp.pid.ki = 0.1
    kp0 = lp.pid.kp
    lp.card = "CRD0 fan 1"
    lp.step()
    assert lp.pid.kp == kp0                                 # no file: as started
    lp.clock.t += 2
    lp.step()
    i0 = lp.pid.i
    assert i0 > 0
    live.write_text(json.dumps({"fan1": {"kp": 2, "ki": 0.5, "kd": 2}}))
    lp.clock.t += 2
    lp.step()
    assert (lp.pid.kp, lp.pid.ki, lp.pid.kd) == (2.0, 0.5, 2.0)
    assert lp.pid.i == pytest.approx(i0 + 0.5 * 5.0 * 2.0)  # on from where it was, at the new ki
    d = json.loads(st.read_text())
    assert d["fan"]["gains"] == {"kp": 2.0, "ki": 0.5, "kd": 2.0}
    assert d["fan"]["i"] == round(lp.pid.i, 2)
    said = [x for x in lp.out.getvalue().splitlines() if "gains of fan" in x]
    n = len(said)
    assert n == 1
    lp.clock.t += 2
    lp.step()
    assert len([x for x in lp.out.getvalue().splitlines() if "gains of fan" in x]) == n

    for bad in ('{"fan1": {"kp": 101, "ki": 0, "kd": 0}}', '{"fan1": {"kp": NaN, "ki": 0, "kd": 0}}',
                '{"fan1": {"kp": 1' + "0" * 400 + ', "ki": 0, "kd": 0}}',
                '{"fan1": {"kp": true, "ki": 0, "kd": 0}}', '{"fan1": {"kp": 1}}', '{"fan1": 3}',
                "not json"):
        live.write_text(bad)
        lp.clock.t += 2
        lp.step()
        assert (lp.pid.kp, lp.pid.ki, lp.pid.kd) == (2.0, 0.5, 2.0), bad
    assert "the gains stay" in lp.out.getvalue()
    # the fan guard greps the pane for these names and takes the last two lines as the readings
    assert all("die" in x for x in lp.out.getvalue().splitlines()
               if "CRD0 fan 1" in x or "CRD1 fan 2" in x)
    live.unlink()
    lp.clock.t += 2
    lp.step()
    assert (lp.pid.kp, lp.pid.ki, lp.pid.kd) == (2.0, 0.5, 2.0)


def test_the_gains_are_each_fans_own(tmp_path):
    """fan 2 takes its own entry; another fan's leaves it alone. Without a status file, or
    while tuning, the loop reads none."""
    src = Src([{"die_c": 50.0, "reg_c": None}] * 3)
    lp, st = status_loop(tmp_path, src)
    lp.fan.n = 2
    kp0 = lp.pid.kp
    (st.parent).mkdir(parents=True, exist_ok=True)
    (st.parent / FP.LIVE_GAINS).write_text(json.dumps({"fan1": {"kp": 9, "ki": 1, "kd": 1}}))
    lp.step()
    assert lp.pid.kp == kp0
    (st.parent / FP.LIVE_GAINS).write_text(json.dumps({"fan2": {"kp": 9, "ki": 1, "kd": 1}}))
    lp.clock.t += 2
    lp.step()
    assert lp.pid.kp == 9.0


def test_the_target_set_while_running(tmp_path):
    """target_c moves the die's target, the status file says so, the integral stays and the
    regulator keeps its own target; no target_c leaves it; outside 40-70 C the whole entry is
    refused."""
    src = Src([{"die_c": 65.0, "reg_c": 72.0}] * 12)
    lp, st = status_loop(tmp_path, src, "--reg-target", "70")
    live = st.parent / FP.LIVE_GAINS
    g = {"kp": 1, "ki": 0.1, "kd": 0}
    lp.step()
    assert lp.pid.target == 60.0
    lp.clock.t += 2
    lp.step()
    i0 = lp.pid.i
    live.write_text(json.dumps({"fan1": dict(g, target_c=66)}))
    lp.clock.t += 2
    row = lp.step()
    assert lp.pid.target == 66.0 and json.loads(st.read_text())["fan"]["target_c"] == 66.0
    assert lp.reg_target == 70.0
    assert row["ctl_c"] == 68.0                 # the regulator's 2 C above its own, in the die's terms
    assert lp.pid.i == pytest.approx(i0 + 0.1 * (68.0 - 66.0) * 2)
    assert "target 66 C" in lp.out.getvalue()
    live.write_text(json.dumps({"fan1": dict(g, kp=2)}))
    lp.clock.t += 2
    lp.step()
    assert (lp.pid.kp, lp.pid.target) == (2.0, 66.0)
    for bad in (39.9, 70.1, "65", True, None):
        live.write_text(json.dumps({"fan1": dict(g, kp=3, target_c=bad)}))
        lp.clock.t += 2
        lp.step()
        assert (lp.pid.kp, lp.pid.target) == (2.0, 66.0), bad


def test_the_limits_are_the_servers(tmp_path):
    from opentpu.serve import fans as FN
    assert FP.GAIN_MAX == FN.GAIN_MAX and FP.LIVE_GAINS == FN.LIVE_GAINS
    assert FP.TARGET_RANGE == FN.TARGET_RANGE


def test_card_env_reads_the_wrappers_file_as_the_shell_does(tmp_path, monkeypatch):
    """BCU_ENV is a shell file the wrappers source (boards/bcu1525/chat35/README.md's example
    has a comment after each value): card_env takes the value the shell would, without the
    comment, a quoted value whole, and the environment's first."""
    f = tmp_path / "bcu.env"
    f.write_text("# this host's cards\n"
                 "BCU_CARD0_SERIAL=ABC123        # the card's JTAG cable serial\n"
                 "export BCU_CARD0_BDF=0000:0b:00.0  # its PCI function\n"
                 "BCU_CARD0_NAME='left card'   # quoted\n"
                 'BCU_CARD1_NAME="a#b"\n'
                 "BCU_CARD1_SERIAL=X#1\n")
    monkeypatch.setenv("BCU_ENV", str(f))
    for k in ("SERIAL", "BDF", "NAME"):
        monkeypatch.delenv(f"BCU_CARD0_{k}", raising=False)
        monkeypatch.delenv(f"BCU_CARD1_{k}", raising=False)
    assert FP.card_env(0, "SERIAL") == "ABC123"
    assert FP.card_env(0, "BDF") == "0000:0b:00.0"
    assert FP.card_env(0, "NAME", "card0") == "left card"
    assert FP.card_env(1, "NAME", "card1") == "a#b"
    assert FP.card_env(1, "SERIAL") == "X#1"            # (a # inside a word is no comment)
    assert FP.card_env(1, "BDF", "none") == "none"
    monkeypatch.setenv("BCU_CARD0_SERIAL", "ENV")
    assert FP.card_env(0, "SERIAL") == "ENV"
