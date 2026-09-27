"""The bit-banged I2C master (opentpu/host/i2c.py) against open-drain slave models
(opentpu/host/fake_i2c.py) on FakeTransport: transactions, clock stretching, arbitration,
bus recovery, the read-only guarantees, LM73 / PMBus decoding, and otpu-i2c, otpu-diag and
otpu-smi on a card with I2C devices. The RTL side (I2C_CTRL / I2C_IN on the board model) is
test_observability.py::test_i2c_pins.
"""
import json
import time

import pytest

from opentpu.host import fake_i2c as F
from opentpu.host import i2c as I
from opentpu.host import regs as R
from opentpu.host.fake import FakeTransport

FAST = {"half": 0, "idle": 0, "busy_timeout": 0.01}


@pytest.fixture(autouse=True)
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path / "otpu"))
    return tmp_path / "otpu"


def card(sensor=(), smbus=(), devname="fakei2c"):
    return FakeTransport(devname=devname, i2c=[F.OpenDrainBus(sensor), F.OpenDrainBus(smbus)])


def all_targets(t):
    return [x for b in t.i2c for x in b.targets if isinstance(x, F.Target)]


# ------------------------------------------------------------------------------ decoding
def test_linear_formats():
    assert I.linear11(0x1E << 11 | 500) == 125.0          # exponent -2
    assert I.linear11(0x1F << 11 | 0x7FF) == -0.5          # mantissa -1, exponent -1
    assert I.linear11(0x0000) == 0.0
    for x in (0.0, 1.0, 12.0, 0.95, 20.25, -3.5, 1500.0, 0.001):
        assert I.linear11(I.to_linear11(x)) == pytest.approx(x, rel=2e-3, abs=1e-3)
    assert I.linear16(409, 0x17) == pytest.approx(409 / 512)        # VOUT_MODE exponent -9
    assert I.linear16(409, 0x40 | 0x17) is None                     # direct mode
    assert I.lm73_temp(0x14A0) == 41.25 and I.lm73_temp(0xF600) == -20.0


# ------------------------------------------------------------------------------ the master
def test_transactions_on_the_card_buses():
    t = FakeTransport(devname=None, i2c=F.card_buses())
    assert t.reg_read(R.R_CAPS) & R.CAP_I2C
    s, m = I.Bus(t, "sensor", **FAST), I.Bus(t, "smbus", **FAST)
    assert s.scan() == [0x49] and m.scan() == [0x50, 0x60]
    assert s.scan(quick_write=True) == [0x49]
    assert s.read(0x49, 0x07, 2) == b"\x01\x90"
    assert m.read_block(0x60, 0x99) == b"ACME"
    assert m.read_byte(0x60, 0x98) == 0x22
    assert m.read_word(0x60, 0x8C) == I.to_linear11(20.0)
    with pytest.raises(I.Nack, match="command 0x80"):
        m.read_byte(0x60, 0x80)                  # an unsupported command is refused
    with pytest.raises(I.Nack, match="no ACK"):
        m.read_byte(0x62, 0x98)
    assert m.read(0x50, 0x10, 3) == bytes([0x10 ^ 0x5A, 0x11 ^ 0x5A, 0x12 ^ 0x5A])
    assert all(not x.writes for x in all_targets(t))
    assert t.regs[R.R_I2C_CTRL] == 0 and s.levels() == (True, True)   # released after each call


def test_other_bus_bits_are_kept():
    t = card([F.lm73()])
    t.regs[R.R_I2C_CTRL] = 0
    b0 = I.Bus(t, 0, **FAST)
    with b0:
        b0._other = 0b1000                                  # bus 1's SDA, as another user left it
        b0._drive(True, False)
        assert t.regs[R.R_I2C_CTRL] == 0b1001


def test_clock_stretching():
    t = card(smbus=[F.pmbus(stretch=40)])
    b = I.Bus(t, "smbus", **FAST)
    assert b.read_block(0x60, 0x9A) == b"PMB1000"
    t.i2c[1].hold[0] = True                                 # SCL never comes back
    with pytest.raises(I.StuckBus, match="SCL"):
        b.read_byte(0x60, 0x98)
    b2 = I.Bus(t, "smbus", **{**FAST, "stretch": 0.002})
    t.i2c[1].hold[0] = False
    with b2:
        b2._start()
        b2._tx(0x60 << 1)
        t.i2c[1].hold[0] = True
        with pytest.raises(I.StuckBus, match="held low"):
            b2._tx(0x98)
    t.i2c[1].hold[0] = False


def test_arbitration_loss_backs_off_and_retries():
    g = F.Glitch(times=2)
    t = card(smbus=[F.pmbus(), g])
    b = I.Bus(t, "smbus", **FAST)
    assert b.read_byte(0x60, 0x98) == 0x22 and g.fired == 2
    t = card(smbus=[F.pmbus(), F.Glitch(times=99)])
    with pytest.raises(I.BusBusy, match="arbitration"):
        I.Bus(t, "smbus", **FAST).read_byte(0x60, 0x98)


def test_busy_and_stuck_lines():
    t = card([F.lm73()])
    t.i2c[0].hold[0] = True
    with pytest.raises(I.StuckBus, match="SCL low"):
        I.Bus(t, 0, **FAST).scan(0x48, 0x4A)
    t.i2c[0].hold = [False, True]                           # SDA held by a fault: recovery fails
    with pytest.raises(I.StuckBus, match="9 recovery clocks"):
        I.Bus(t, 0, **FAST).probe(0x49)


def test_recovery_of_a_slave_left_mid_byte():
    lm = F.lm73(temp_c=0.0)                                 # the register reads 0x0000: SDA low
    t = card([lm])
    b = I.Bus(t, 0, **FAST)
    with b:                                                 # a master that died mid-read
        b._start()
        assert b._tx(0x49 << 1 | 1)
        b._rx_bit()
    assert b.levels() == (True, False)                      # the LM73 still drives a 0
    assert I.lm73_identify(b, 0x49) == {"kind": "lm73", "id": 0x0190, "temp_c": 0.0}
    assert b.recover()


def test_the_lock_serializes_processes(run_dir):
    t = card([F.lm73()])
    a = I.Bus(t, 0, **FAST)
    b = I.Bus(FakeTransport(devname="fakei2c", i2c=t.i2c), 0, lock_timeout=0.1, **FAST)
    with a:
        t0 = time.monotonic()
        with pytest.raises(I.BusBusy, match="lock"):
            b.probe(0x49)
        assert time.monotonic() - t0 >= 0.1
    assert b.probe(0x49)
    assert (run_dir / "fakei2c.i2c.lock").exists()


# ------------------------------------------------------------------------------ safety
def test_read_only_guarantees():
    t = FakeTransport(devname=None, i2c=F.card_buses())
    m = I.Bus(t, "smbus", **FAST)
    for cmd in sorted(I.SEND_BYTE):
        with pytest.raises(ValueError, match="sent alone"):
            m.read_byte(0x60, cmd)
    for addr in (0x30, 0x36, 0x37, 0x0C, 0x61):
        with pytest.raises(ValueError, match="no command codes"):
            m.read_byte(addr, 0x00)
    d = I.discover(t, **FAST)
    I.measured(t, d, **FAST)
    for x in all_targets(t):
        assert not x.writes                                 # never a data byte
    ee = t.i2c[1].targets[1]
    assert ee.addr == 0x50 and not ee.pointers              # identification skips EEPROMs
    pm = t.i2c[1].targets[0]
    assert set(pm.pointers) <= set(I.PMBUS.values())


def test_identification_skips_expanders_and_muxes():
    ex, mux = F.Target(0x20, {0: b"\x00"}), F.Target(0x70, {0: b"\x00"})
    t = card(smbus=[ex, mux])
    d = I.discover(t, ["smbus"], **FAST)["buses"]["smbus"]["devices"]
    assert [(x["addr"], x["kind"]) for x in d] == [(0x20, "I/O expander range"),
                                                   (0x70, "I2C mux range")]
    assert not ex.pointers and not mux.pointers


# ------------------------------------------------------------------------------ discovery
def test_discovery_identifies_devices():
    ina = F.Target(0x40, {0xFE: b"\x54\x49", 0xFF: b"\x22\x60", 0x01: b"\x00\x64",
                          0x02: b"\x25\x80"})
    odd = F.Target(0x58, {0x98: b"\x77", 0x99: b"\x02\x01\x02"})
    t = card([F.lm73(0x4C, temp_c=-5.5)], [ina, odd, F.pmbus(0x62, pin=None)])
    d = I.discover(t, **FAST)
    sensor, smbus = (d["buses"][n]["devices"] for n in ("sensor", "smbus"))
    assert sensor == [{"addr": 0x4C, "kind": "lm73", "id": 0x0190, "temp_c": -5.5}]
    assert smbus[0] == {"addr": 0x40, "kind": "ina", "model": "INA226", "shunt_uv": 250.0,
                        "vbus": pytest.approx(12.0)}
    assert smbus[1]["kind"] == "pmbus?" and "telemetry" not in smbus[1]
    p = smbus[2]
    assert p["kind"] == "pmbus" and p["mfr_id"] == "ACME" and p["vout_mode"] == 0x17
    tel = p["telemetry"]
    assert tel["pin"] is None and tel["vin"] == 12.0 and tel["iout"] == 20.0
    assert tel["vout"] == pytest.approx(0.95, abs=2e-3) and tel["temp_c"] == 55.0
    assert tel["power"] == {"w": pytest.approx(19.0, abs=0.02), "how": "POUT"}
    assert I.power_devices(d) == [("smbus", 0x62, 0x17)]


def test_measured_power_uses_the_cache(run_dir):
    t = card([F.lm73()], [F.pmbus(0x60, pin=21.5), F.pmbus(0x62, pin=8.0)])
    m = I.measured(t, **FAST)
    assert m["w"] == pytest.approx(29.5) and m["how"] == "PIN" and m["board_temp_c"] == 41.25
    assert [r["addr"] for r in m["rails"]] == [0x60, 0x62]
    assert json.loads((run_dir / "fakei2c.i2c.json").read_text())["version"] == 1
    t.i2c[1].targets.append(F.pmbus(0x64))                  # not in the cache: not read
    assert I.measured(t, **FAST)["w"] == pytest.approx(29.5)
    t2 = card([F.lm73()], devname="fakei2c2")               # no PMBus: no measured power
    assert I.measured(t2, **FAST) == {"w": None, "how": None, "rails": [],
                                     "board_temp_c": 41.25}


# ------------------------------------------------------------------------------ tools
def test_otpu_i2c(capsys):
    t = FakeTransport(devname=None, i2c=F.card_buses())
    run = lambda *a: I.main(["--khz", "1e9", *a], open_transport=lambda dev: t)  # noqa: E731
    assert run("scan") == 0
    out = capsys.readouterr().out
    assert "0x49  LM73 temperature sensor  41.25 C" in out and "0x60  PMBus ACME PMB1000" in out
    assert run("read", "sensor", "0x49", "7", "-n", "2") == 0
    assert capsys.readouterr().out.strip() == "01 90"
    assert run("--json", "pmbus-dump", "smbus", "0x60") == 0
    d = json.loads(capsys.readouterr().out)
    assert d["telemetry"]["power"] == {"w": 21.5, "how": "PIN"}
    assert run("read", "smbus", "0x60", "0x11") == 1
    assert "sent alone" in capsys.readouterr().err
    assert I.main(["scan"], open_transport=lambda dev: FakeTransport(devname=None)) == 1
    assert "no I2C pins" in capsys.readouterr().err


def test_diag_i2c_section(tmp_path, capsys):
    from opentpu.host import diag
    t = card([F.lm73()], [F.pmbus()], devname="fake7")
    out = tmp_path / "diag.json"
    assert diag.main(["--only", "i2c", "--json", str(out)], open_transport=lambda dev: t) == 0
    st = {r["name"]: (r["status"], r["msg"]) for r in json.loads(out.read_text())["rows"]}
    assert st["I2C pins"][0] == "PASS"
    assert st["LM73 bus (N24 / N25) scan"] == ("PASS", "1 device: 0x49 LM73 41.2 C")
    assert st["PCIe SMBus (R26 / R27) scan"][1] == "1 device: 0x60 PMBus ACME PMB1000 21.50 W (PIN)"
    assert st["power monitors (PMBus)"][1].startswith("1 PMBus device reports power")
    # no I2C pins: the section is skipped
    assert diag.main(["--only", "i2c", "--json", str(out)],
                     open_transport=lambda dev: FakeTransport(devname="fake8")) == 0
    st = {r["name"]: r["status"] for r in json.loads(out.read_text())["rows"]}
    assert st["I2C pins"] == "SKIP" and st["power monitors (PMBus)"] == "SKIP"
    # a stuck SCL fails with a hint
    t = card([F.lm73()], devname="fake9")
    t.i2c[0].hold[0] = True
    assert diag.main(["--only", "i2c", "--json", str(out)], open_transport=lambda dev: t) == 1
    rep = json.loads(out.read_text())
    assert any("pull-up" in h for h in rep["hints"])


def test_smi_measured_power(tmp_path, capsys):
    from opentpu.host import smi
    t = card([F.lm73()], [F.pmbus()], devname="fake6")
    none = str(tmp_path / "none.json")
    assert smi.main(["--json", "--dev", "/dev/fake6", "--power-json", none, "-i", "0"],
                    open_transport=lambda dev: t) == 0
    (d,) = json.loads(capsys.readouterr().out)
    assert d["measured"]["w"] == 21.5 and d["measured"]["board_temp_c"] == 41.25
    smi.main(["--dev", "/dev/fake6", "--power-json", none, "-i", "0"],
             open_transport=lambda dev: t)
    tab = capsys.readouterr().out
    assert "Power 21.5W measured" in tab and "Board 41 °C" in tab
    smi.main(["--no-i2c", "--dev", "/dev/fake6", "--power-json", none, "-i", "0"],
             open_transport=lambda dev: t)
    tab = capsys.readouterr().out
    assert "measured" not in tab and "Power n/a" in tab
