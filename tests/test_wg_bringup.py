"""boards/bcu1525/scripts/wg_bringup.py's host side on a fake card: SAFE as configured, RESUME
for the GOs, the trips, the DIMMs then the accelerator, the guard's transport while JTAG is busy,
the tools' processes under the guard."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
spec = importlib.util.spec_from_file_location("wg_bringup",
                                              ROOT / "boards/bcu1525/scripts/wg_bringup.py")
W = importlib.util.module_from_spec(spec)
_target = os.environ.get("OTPU_TARGET")
spec.loader.exec_module(W)          # (it sets OTPU_TARGET for its children; the other tests' isn't)
if _target is None:
    os.environ.pop("OTPU_TARGET", None)
else:
    os.environ["OTPU_TARGET"] = _target
G, R = W.G, W.R
from test_bcu_gov import FakeGov                                # noqa: E402


@pytest.fixture(autouse=True)
def _wg_target(monkeypatch):
    """The target wg_bringup's import gives its own runs, for this file's tests only."""
    monkeypatch.setenv("OTPU_TARGET", os.environ.get("OTPU_TARGET") or "bcu1525-wg")


class FakeWg(FakeGov):
    """FakeGov in front of the WG top: the control registers fenced until the accelerator runs,
    which needs every DIMM on and calibrated (`calibrates`: the DIMMs that do); TRIP and RESUME
    written with the build's clamps (724, 704), RUN ended on the die at TRIP."""

    def __init__(self, calibrates=(0, 1, 2, 3), code=600):
        super().__init__(code=code)
        self.calibrates = set(calibrates)

    def reg_write(self, a, v):
        o = a - G.R_GOV
        if o == G.TRIP:
            self.trip = min(v & 0x3FF, 724)
        elif o == G.RESUME:
            self.resume = min(v & 0x3FF, 704)
        elif o == G.HB_MS:
            self.hb = v
        super().reg_write(a, v)

    def close(self):
        pass

    def acc_up(self):
        return bool(self.run and self.ctrl & G.CTRL_ACC and self.ctrl & 0xF == 0xF
                    and self.calibrates == {0, 1, 2, 3})

    def reg_read(self, a):
        if a < G.R_GOV:
            if not self.acc_up():
                return W.FENCED
            return {R.R_ID: R.ID_OTPU, R.R_STATUS: 0x3}.get(a, 0)
        if a - G.R_GOV == G.STATUS:
            if self.run and self.code >= self.trip:
                self.run, self.cause = 0, self.cause | 1                # "hot"
            return super().reg_read(a) | (G.ST_ACC_ON if self.run and self.ctrl & G.CTRL_ACC
                                          else 0)
        return super().reg_read(a)

    def reg_read_many(self, offs):
        return [self.reg_read(o) for o in offs]


class Guard:
    tripped, booted = None, False


def args(**kw):
    return types.SimpleNamespace(**{"cal_s": 0.0, "acc_s": 0.3, "ltx": None, "out": Path("."),
                                    **kw})


def test_safe_after_configuration():
    t = FakeWg()
    g = G.Gov(t)
    assert W.safe0(t, g)["ok"]
    t.run, t.ctrl = 1, 0xF | G.CTRL_ACC
    assert not W.safe0(t, g)["ok"]


def test_all_four_calibrated_start_the_accelerator():
    t = FakeWg()
    g, res = G.Gov(t), {}
    W.dimms_up(t, g, Guard(), W.Card(t), args(), res)
    assert res["acc"]["ok"] and res["acc"]["id"] == hex(R.ID_OTPU)
    assert res["dimms_on"]["dimms"] == [0, 1, 2, 3]


def test_a_dimm_that_does_not_calibrate_stops_the_bring_up():
    t = FakeWg(calibrates=(0, 1, 3))
    res = {}
    with pytest.raises(SystemExit, match="did not calibrate"):
        W.dimms_up(t, G.Gov(t), Guard(), W.Card(t), args(), res)
    assert res["acc"] == {**res["acc"], "ok": False, "id": hex(W.FENCED)}


def test_the_mig_report_holds_off_the_guards_jtag_reading(tmp_path):
    t = FakeWg()
    seen = []

    class Jtag:
        def mon(self):
            return {"t": 50.0, "vccint": 0.85}

        def boot(self):
            pass

        def migs(self, ltx, out):
            try:
                card.mon()
            except OSError as e:
                seen.append(str(e))
            return {"ddr4_0": {"CAL_STATUS": "PASS"}}
    card = W.Card(t, Jtag())
    res = {}
    W.dimms_up(t, G.Gov(card), Guard(), card, args(ltx=tmp_path / "x.ltx", out=tmp_path), res)
    assert seen and "busy" in seen[0]
    assert card.mon()["t"] == 50.0                                  # free again after it
    assert res["migs"] == {"ddr4_0": {"CAL_STATUS": "PASS"}}


def test_the_die_without_the_sysmons_drp_registers():
    """up's die: the governor's code and the SYSMON over JTAG, no BAR0 read of the wizard's DRP
    registers (one can stop its temperature bus, docs/bcu1525_sysmon.md;
    fullimage_jtag's JTAG-to-AXI reads are the same path). A JTAG reading that fails is noted."""
    t, seen = FakeWg(), []
    rd = t.reg_read
    t.reg_read = lambda a: seen.append(a) or rd(a)

    class Jtag:
        def __init__(self):
            self.fail = False

        def mon(self):
            if self.fail:
                raise OSError("JTAG busy")
            return {"t": 41.5, "vccint": 0.85, "vccaux": 1.8, "vccbram": 0.85}

        def boot(self):
            pass
    jt = Jtag()
    card = W.Card(t, jt)
    d = G.die_readings(G.Gov(card), card.mon)
    assert d["gov_c"] == pytest.approx(G.celsius(600), abs=0.01) and d["gov_valid"]
    assert d["jtag_c"] == 41.5 and d["vccint"] == 0.85 and d["age_ms"] is None
    jt.fail = True
    d = G.die_readings(G.Gov(card), card.mon)
    assert "jtag_c" not in d and d["jtag_error"] == "JTAG busy"
    assert G.die_readings(G.Gov(W.Card(t)))["gov_c"] == d["gov_c"]          # --no-jtag
    assert seen and not any(G.R_SYSMON <= a < G.R_SYSMON + 0x10000 for a in seen)
    for f in ("wg_bringup.py", "fullimage_jtag.py"):
        src = (ROOT / "boards/bcu1525/scripts" / f).read_text()
        assert "0x30000" not in src and "R_SYSMON" not in src and "G.sysmon(" not in src, f


def test_resume_inside_the_governors_clamp():
    """up's RESUME: --resume (65 C), at most the build's RESUME_MAX (704, 69.9 C) and TRIP less
    20 codes (10 C); asked, before, after and the ceiling for wg.json."""
    t = FakeWg()
    g = G.Gov(t)
    r = W.set_resume(g, 65.0, g.limits())
    assert t.resume == G.code_of(65.0) and r["set_c"] == pytest.approx(65.0, abs=0.25)
    assert r["asked_c"] == 65.0 and r["was_c"] == pytest.approx(55.0, abs=0.25)
    assert r["max_c"] == round(G.celsius(704), 2)
    r = W.set_resume(g, 69.99, g.limits())                         # past RESUME_MAX
    assert t.resume == 704 and r["set_c"] == r["max_c"]
    t.trip = G.code_of(70.0)                                        # a lower TRIP: 20 codes under
    r = W.set_resume(g, 65.0, g.limits())
    assert t.resume == G.code_of(70.0) - 20 and r["set_c"] == r["max_c"] < 60.1
    assert W.RESUME_C == 65.0
    for bad in ("70.5", "85"):
        with pytest.raises(SystemExit):
            W.main(["up", "--out", "x", "--resume", bad])


class TripGuard:
    """gov.Guard for trips() on a FakeWg: stop(safe=False) ends the kicks, so the governor drops
    RUN (no heartbeat); tripped once a GO would wait on RESUME (the die above it), so go_cool
    fails at once instead of retrying for 30 minutes."""

    def __init__(self, t, csv=None):
        self.t = getattr(t, "pcie", t)                               # (up's Card)
        self.csv, self.booted, self.reg, self.errors, self.stops = csv, False, {}, [], []

    @property
    def tripped(self):
        return f"GO would wait: the die {self.t.code} above RESUME {self.t.resume}" \
            if self.t.code > self.t.resume else None

    def read_first(self):
        pass

    def start(self):
        return self

    def stop(self, safe=True):
        self.stops.append(safe)
        if not safe and self.t.run:
            self.t.run, self.t.cause = 0, self.t.cause | 4              # "no heartbeat"


def test_resume_survives_the_trips(tmp_path):
    """The idle die at 60 C, fan_pid's target, over the governor's reset RESUME (55 C): RESUME at
    65 C before the first GO; the thermal trip takes TRIP under the die (and under RESUME), then
    TRIP back at 80 C (TRIP_MAX) and RESUME at 65 C again, so the heartbeat trip's GO comes at
    once."""
    t = FakeWg(code=G.code_of(60.0))
    g = G.Gov(t)
    with pytest.raises(W.Tripped, match="GO would wait"):          # 55 C: the 11:47 hang
        W.trips(g, lambda csv: TripGuard(t, csv), tmp_path)
    t.run = t.cause = 0
    r = W.set_resume(g, 65.0, g.limits())
    res, guard = W.trips(g, lambda csv: TripGuard(t, csv), tmp_path, r["set_c"])
    assert res["thermal"]["causes"] == ["hot"] and res["heartbeat"]["causes"] == ["no heartbeat"]
    assert t.trip == 724 and t.resume == G.code_of(65.0)
    assert res["thermal"]["trip_c"] == round(G.celsius(724), 2)
    assert res["thermal"]["resume_c"] == r["set_c"]
    assert guard.csv == tmp_path / "thermal2.csv" and not t.run
    trips = [v for o, v in t.writes if o == G.TRIP]
    assert len(trips) == 2 and trips[0] < G.code_of(65.0)          # the trip: under RESUME too


def test_up_sets_resume_before_the_first_go(tmp_path, monkeypatch):
    """up on a fake card through PCIe, the die at 60 C: RESUME at --resume before the trips and
    the DIMMs (each GO taken at once), in wg.json."""
    t = FakeWg(code=G.code_of(60.0))
    seen = []
    wr = t.reg_write

    def write(a, v):                                                # RESUME before any GO
        if a - G.R_GOV == G.CMD and v & G.CMD_GO:
            seen.append(t.resume)
        wr(a, v)
    t.reg_write = write
    monkeypatch.setattr(W.signal, "signal", lambda *a: None)
    monkeypatch.setattr(W.G, "Guard", TripGuard)
    import opentpu.host.board as board
    monkeypatch.setattr(board, "XdmaTransport", lambda dev, dma=False: t)
    rc = W.main(["up", "--out", str(tmp_path), "--no-jtag", "--mem", "none", "--no-selftest",
                 "--cal-s", "0", "--acc-s", "0.3", "--resume", "64"])
    d = json.loads((tmp_path / "wg.json").read_text())
    assert rc == 0, d.get("tripped")
    assert d["resume"]["asked_c"] == 64.0 and d["resume"]["set_c"] == pytest.approx(64.0, abs=0.25)
    assert d["resume"]["was_c"] == pytest.approx(55.0, abs=0.25)
    assert d["trips"]["thermal"]["resume_c"] == d["resume"]["set_c"] and d["acc"]["ok"]
    assert len(seen) == 3 and set(seen) == {G.code_of(64.0)}       # two trips' GOs, the DIMMs'


def test_up_closes_the_card_when_jtag_does_not_open(tmp_path, monkeypatch):
    """up with JTAG: the cable's session failing to open leaves no PCIe transport open."""
    t = FakeWg()
    closed = []
    t.close = lambda: closed.append(1)

    def no_jtag(*a, **kw):
        raise OSError("no hw_server target")
    monkeypatch.setattr(W.signal, "signal", lambda *a: None)
    import opentpu.host.board as board
    import opentpu.host.jtagaxi as jtagaxi
    monkeypatch.setattr(board, "XdmaTransport", lambda dev, dma=False: t)
    monkeypatch.setattr(jtagaxi, "JtagAxi", no_jtag)
    with pytest.raises(OSError, match="no hw_server target"):
        W.main(["up", "--out", str(tmp_path), "--serial", "S", "--mem", "none", "--no-selftest"])
    assert closed == [1]


def test_no_jtag_no_mon_or_boot():
    c = W.Card(FakeWg())
    assert not hasattr(c, "mon") and not hasattr(c, "boot")


def test_go_cool_retries_until_the_governor_takes_it():
    t = FakeWg()
    t.code = t.resume + 5                                           # above RESUME: refused
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) == 3:
            t.code = t.resume - 2
    W.go_cool(G.Gov(t), Guard(), (), sleep=sleep)
    assert t.run == 1 and len(sleeps) == 3


def test_check_raises_once_safe():
    t = FakeWg()
    g = G.Gov(t)
    g.go(dimms=())
    W.check(g, Guard())
    g.safe()
    with pytest.raises(W.Tripped, match="SAFE"):
        W.check(g, Guard())


def test_child_runs_and_is_stopped_on_a_trip(tmp_path):
    t = FakeWg()
    g = G.Gov(t)
    g.go(dimms=())
    guard = Guard()
    assert W.child("ok", ["timeit", "-n", "1", "pass"], tmp_path, g, guard, 60) == 0
    assert "loop" in (tmp_path / "ok.log").read_text()
    threading.Timer(1.0, g.safe).start()
    with pytest.raises(W.Tripped):
        W.child("long", ["timeit", "-n", "1", "import time; time.sleep(60)"], tmp_path, g, guard,
                120)


def test_a_run_is_a_module_of_the_repository(tmp_path):
    t = FakeWg()
    g = G.Gov(t)
    g.go(dimms=())
    argv = W.shlex.split("tools.q35_split --help --points 'a b'")  # (as --run splits it)
    assert W.child("run0", argv, tmp_path, g, Guard(), 60) == 0
    assert "--peak-gbs" in (tmp_path / "run0.log").read_text()
    assert argv[-1] == "a b"
