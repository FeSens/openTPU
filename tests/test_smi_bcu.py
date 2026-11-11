"""otpu-smi on the BCU1525 (fake.FakeBcu: the WG image's BAR0): the board's rows from registers
alone, in SAFE with the accelerator fenced and in RUN with it on; VCCINT's regulator input power
from the BMC watch, labelled as what it is; nothing written at the governor, the bridge or the
SYSMON, no byte on the BMC's line, and the bridge's RX and STATUS (reads with side effects) never
read; the SYSMON's DRP registers (whose AXI reads can stop the governor's die) read only with
--sysmon or once the governor's die is gone; the other cards' tables without any of it."""
import csv
import json

import pytest

from opentpu.host import bmc as B
from opentpu.host import gov as G
from opentpu.host import regs as R
from opentpu.host import smi
from opentpu.host.fake import FakeBcu, FakeTransport


@pytest.fixture(autouse=True)
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path / "otpu"))
    B._known.clear()


class Reads:
    """A transport's register reads and writes, recorded."""

    def __init__(self, t):
        self.t, self.reads, self.writes = t, [], []

    def reg_read(self, off):
        self.reads.append(off)
        return self.t.reg_read(off)

    def reg_read_many(self, offs):
        return [self.reg_read(o) for o in offs]

    def reg_write(self, off, val):
        self.writes.append((off, val))
        self.t.reg_write(off, val)

    def __getattr__(self, k):
        return getattr(self.t, k)


def side_effect_reads(rec: Reads) -> list[int]:
    """The bridge's reads that take an RX byte or clear STATUS flags (it decodes [5:2])."""
    return [o for o in rec.reads if R.R_BMC <= o < R.R_BMC + 0x10000 and o & 0x3C in (B.RX,
                                                                                    B.STATUS)]


def drp_reads(rec: Reads) -> list[int]:
    """The SYSMON's DRP registers read (R_SYSMON + 0x400 and up: each read risks the wizard's
    temperature bus)."""
    return [o for o in rec.reads if G.R_SYSMON + 0x400 <= o < G.R_SYSMON + 0x10000]


def flat(tab: str) -> str:
    return " ".join(tab.replace("│", "").split())


def box(tab: str) -> None:
    assert all(len(r) == smi.W for r in tab.splitlines()[1:]), tab


def test_sysmon_and_bridge_decode_the_cards_registers():
    """The values a card read on 2026-10-29 (f150m15, SAFE): the SYSMON's die, VCCINT and flag, the
    bridge's divider at 150 MHz, the watch's LINEAR11 words."""
    vals = {G.R_SYSMON + G.SM_TEMP: 0xB320, G.R_SYSMON + G.SM_VCCINT: 0x44E0,
            G.R_SYSMON + G.SM_VCCAUX: 0x9BBE, G.R_SYSMON + G.SM_VBRAM: 0x48A3,
            G.R_SYSMON + G.SM_MAX_TEMP: 0xB52A, G.R_SYSMON + G.SM_FLAG: 0x201,
            G.R_SYSMON + G.SM_TEMP_UP: 0xB52A, G.R_SYSMON + G.SM_TEMP_LO: 0xB00A,
            R.R_BMC + B.ID: B.ID_BMC2, R.R_BMC + B.DIV: 0x516, R.R_BMC + B.W_HOT: 0x01300117,
            R.R_BMC + B.W_P0: 0x0130EA0A, R.R_BMC + B.W_P1: 0x0127EA17,
            R.R_BMC + B.W_CTL: 0x0124EA2E, R.R_BMC + B.W_PIN: 0x0121C2DD,
            R.R_BMC + B.W_VIN: 0x011ED303, R.R_BMC + B.W_FW: 0x10003, R.R_BMC + B.W_CNT: 0xFFFF}

    class Card:
        def reg_read(self, off):
            return vals.get(off, 0)

        def reg_read_many(self, offs):
            return [self.reg_read(o) for o in offs]
    sm = G.sysmon(Card())
    assert sm["die_c"] == pytest.approx(75.89, abs=0.01) and sm["max_c"] == pytest.approx(79.87,
                                                                                        abs=0.01)
    assert sm["vccint"] == pytest.approx(0.807, abs=1e-3) and sm["vbram"] == pytest.approx(0.851,
                                                                                         abs=1e-3)
    assert sm["alarm"] and not sm["ot"] and sm["alarm_lo_c"] == pytest.approx(69.92, abs=0.01)
    br = smi.bmc_bridge(Card())
    assert br["name"] == "BMC2" and br["core_mhz"] == pytest.approx(149.99, abs=0.01)
    w = br["watch"]
    assert (w["p0_c"], w["p1_c"], w["ctl_c"]) == (65.25, 66.875, 69.75)
    assert w["pin"] == pytest.approx(2.863, abs=1e-3) and w["vin"] == pytest.approx(12.047,
                                                                                    abs=1e-3)
    p = smi.watch_power(br)
    assert p["label"] == "VCCINT regulator input" and p["age_ms"] == 0x121 and p["vin"] == w["vin"]
    rec = Reads(Card())
    one = G.sysmon(rec, full=False)                     # the die alone: one DRP read
    assert rec.reads == [G.R_SYSMON + G.SM_TEMP] and one["die_c"] == sm["die_c"]
    assert one["vccint"] is None and one["flag"] is None and not one["full"]


def test_safe_the_board_rows(tmp_path, capsys):
    """SAFE as on the card: the accelerator fenced, the SYSMON's die above the governor's (its
    temperature bus), the alarm standing. Every row from registers; nothing written; the
    SYSMON's DRP registers only with --sysmon."""
    fk = FakeBcu(die_c=75.9, gov_die_c=46.05, max_c=79.87, alarm=True, run_ms=15103,
                 trips=(1, 1))
    t = Reads(fk)
    none = str(tmp_path / "none.json")
    args = ["--dev", "/dev/fakebcu0", "--power-json", none, "-i", "0"]
    assert smi.main(args, open_transport=lambda dev: t) == 0
    tab = capsys.readouterr().out
    box(tab)
    assert "n/a (accelerator off)" in tab and "core ~100.0 MHz" in tab and "BMC fw 0x10004" in tab
    assert "accelerator off: its registers fenced" in tab
    assert "SAFE (host SAFE)  trips 1 thermal, 1 heartbeat  last RUN 15.1 s" in tab
    assert "Thermal    die 46.0 °C (governor)  trip 79.9  resume 55.0" in tab
    assert "24.8 W VCCINT regulator input (not board total)  VIN 12.09 V  0.0 s old" in tab
    assert "pages 45.0 / 47.0 °C  controller 52.0 °C  trip 78.0  resume 75.0" in tab
    assert "Rails" not in tab and "! SYSMON temperature alarm: GO refused until it clears" in tab
    assert "Process" in tab and "No running process" in tab
    assert fk.writes == [] and t.writes == [] and side_effect_reads(t) == [] and drp_reads(t) == []
    smi.main(args + ["--sysmon"], open_transport=lambda dev: t)    # the SYSMON too, on request
    tab = capsys.readouterr().out
    box(tab)
    assert "die 75.9 °C SYSMON (max 79.9)  gov 46.0 °C  trip 79.9  resume 55.0" in tab
    assert "VCCINT 0.807 V  VCCAUX 1.822 V  VBRAM 0.850 V" in tab
    assert "! SYSMON temperature alarm (set at 79.9, clears at 69.9 °C): GO refused" in tab
    assert "the governor's die reads 29.8 °C below the SYSMON's: its bus may have stopped" \
        in flat(tab)
    assert len(drp_reads(t)) == 13 and t.writes == []
    assert fk.bmc.sent == bytearray() and not fk.bmc.holds
    smi.main(["--json"] + args, open_transport=lambda dev: t)
    (d,) = json.loads(capsys.readouterr().out)
    assert not d["ok"] and d["id"] == smi.FENCED and d["bcu"]["gov"]["causes"] == ["host SAFE"]
    assert d["bcu"]["bmc"]["name"] == "BMC2" and d["bcu"]["gov"]["alarm"]
    assert d["bcu"]["sysmon"] is None and d["bcu"]["gov_frozen"] is None
    assert d["measured"]["label"] == "VCCINT regulator input" and d["measured"]["source"] == "watch"
    csvp = tmp_path / "s.csv"
    smi.main(args + ["--csv", str(csvp)], open_transport=lambda dev: t)
    head = csvp.read_text().splitlines()[0].split(",")
    assert head[:len(smi.CSV_FIELDS) - 7] == smi.CSV_FIELDS[:-7]       # the old ones, in order
    assert head[-7:] == ["power_age_s", "vin", "die_sysmon_c", "vreg_c", "governor", "die_gov_c",
                         "gov_frozen"]
    (row,) = csv.DictReader(csvp.read_text().splitlines())
    assert row["ok"] == "0" and float(row["power_w"]) == pytest.approx(24.8, abs=0.02)
    assert row["power_how"] == "VCCINT regulator input" and row["governor"] == "SAFE"
    assert row["die_sysmon_c"] == "" and row["vreg_c"] == "52" and row["gov_frozen"] == "0"
    assert float(row["vin"]) == pytest.approx(12.09, abs=0.01) and row["power_age_s"] == "0.003"
    assert float(row["die_gov_c"]) == pytest.approx(46.05, abs=0.3)
    smi.main(args + ["--sysmon", "--csv", str(csvp)], open_transport=lambda dev: t)
    (row,) = csv.DictReader(csvp.read_text().splitlines())
    assert float(row["die_sysmon_c"]) == pytest.approx(75.9, abs=0.3)
    assert t.writes == [] and fk.bmc.sent == bytearray()


def test_run_the_accelerator_and_the_board(tmp_path, capsys):
    """RUN with the accelerator on: the bitstream, its CFG, the DIMMs the governor runs, the
    SYSMON's die, the board's rows and the counters; DRAM GB/s in bcu1525-wg's units (a DRAM_RD
    count is half a 128-byte beat, a DRAM_WR count a whole one). The one write is SNAP."""
    fk = FakeBcu(run=True, kick_ms=300, run_ms=754_000)
    t = Reads(fk)
    d = smi.query(t, "/dev/fakebcu0", sleep=lambda s: None, power_json=str(tmp_path / "n.json"))
    assert d["ok"] and d["bcu"]["target"] == "bcu1525-wg" and d["bcu"]["cfg"]["WG"] == 8
    tab = smi.table([d])
    box(tab)
    # the fake's CFG is the target's: 32 GB at AW 36, 4 GiB channels (Phase A; 512 MiB before)
    assert "Config     bcu1525-wg  S=1  8 ch x 4096 MiB  beat 128 B  WG 8  CFG_HASH" in tab
    assert "DDR4-2400 C0 C1 C2 C3 on" in tab and "Die 62.0 °C" in tab and "150 MHz" in tab
    assert "RUN 12m34s  DIMMs C0 C1 C2 C3  accelerator on  heartbeat 0.3 s ago" in tab
    assert "24.8 W VCCINT regulator input (not board total)" in tab and "GO refused" not in tab
    assert "State      Idle" in tab and "Utilization" in tab and "Power 24.8W" not in tab
    sec = 1_000_000 / 150e6
    assert d["dram_rd_gbs"] == pytest.approx(int(0.70 * 1e6) * 64 / sec / 1e9)
    assert d["dram_wr_gbs"] == pytest.approx(int(0.02 * 1e6) * 128 / sec / 1e9)
    assert {o for o, _ in t.writes} == {R.R_SNAP} and fk.writes == []
    assert side_effect_reads(t) == [] and fk.bmc.sent == bytearray() and drp_reads(t) == []


def test_no_bmc_and_the_watch_without_a_reading(tmp_path):
    none = str(tmp_path / "none.json")
    fk = FakeBcu()
    t = Reads(fk)
    d = smi.query(t, "/dev/fakebcu0", sleep=lambda s: None, power_json=none, bmc=False)
    assert d["measured"] is None and d["bcu"]["bmc"] is None
    assert not [o for o in t.reads if R.R_BMC <= o < R.R_BMC + 0x10000]   # the bridge untouched
    assert "Power      n/a (--no-bmc)" in smi.table([d])
    fk.bmc.watch_age = B.AGE_NONE                       # no reading yet (or 65.5 s old)
    d = smi.query(fk, "/dev/fakebcu0", sleep=lambda s: None, power_json=none)
    assert d["measured"] is None and "Power      n/a (no reading yet)" in smi.table([d])
    fk.bmc.watch_age = 12_000                           # past the governor's REG_STALE
    tab = smi.table([smi.query(fk, "/dev/fakebcu0", sleep=lambda s: None, power_json=none)])
    assert "VIN 12.09 V  12 s STALE" in tab
    box(tab)
    fk2 = FakeBcu(bmc=__import__("opentpu.host.fake_bmc", fromlist=["FakeBmc"]).FakeBmc(
        firmware=0x20001))
    tab = smi.table([smi.query(fk2, "/dev/fakebcu0", sleep=lambda s: None, power_json=none)])
    assert "n/a (the watch stopped)" in tab and "! the BMC watch stopped: firmware 0x20001" in tab


def test_gov_frozen_and_no_sysmon(tmp_path):
    """The governor's die gone: the same for gov.FROZEN_S over the queries sharing a track, or
    not valid. Then (only then, without --sysmon) the SYSMON's die is read, one register, and is
    the die; with neither, the die is n/a. Without a SYSMON (its registers all ones) nothing of
    it is shown."""
    none = str(tmp_path / "none.json")

    def q(fk, **kw):
        t = Reads(fk)
        return smi.query(t, "/dev/fakebcu0", sleep=lambda s: None, power_json=none, **kw), t
    fk = FakeBcu(die_c=40.1, gov_die_c=46.05)           # a card at 09:48: the bus stopped at 46
    track = {"c": G.celsius(G.code_of(46.05)), "since": smi._clock() - 61}
    d, t = q(fk, track=track)
    tab = smi.table([d])
    box(tab)
    assert d["bcu"]["gov_frozen"] == "frozen 61 s" and drp_reads(t) == [G.R_SYSMON + G.SM_TEMP]
    assert "die 40.1 °C SYSMON  gov 46.0 °C FROZEN 61 s  trip 79.9  resume 55.0" in tab
    assert "gov frozen: the governor's die (the wizard's temperature bus) has read 46.0 °C for " \
        "61 s; the die is the SYSMON's register" in flat(tab) and "Rails" not in tab
    c, src = smi.die_now(d["bcu"])
    assert c == pytest.approx(40.1, abs=0.3) and src == "SYSMON"
    assert smi.csv_row(d)["gov_frozen"] == "1"
    d, t = q(fk, track={"c": 50.0, "since": smi._clock() - 600})   # it moved: a new start
    assert d["bcu"]["gov_frozen"] is None and d["bcu"]["gov_same_s"] < 1 and drp_reads(t) == []
    d, t = q(FakeBcu(gov_valid=False))
    tab = smi.table([d])
    assert d["bcu"]["gov_frozen"] == "not valid" and "gov n/a NOT VALID" in tab
    assert "die 62.0 °C SYSMON" in tab and "is not valid; the die is the SYSMON's" in flat(tab)
    d, t = q(FakeBcu(sysmon=False, gov_die_c=46.05, alarm=True), track=dict(track))
    tab = smi.table([d])
    box(tab)
    assert "die n/a (no SYSMON)  gov 46.0 °C FROZEN" in tab and smi.die_now(d["bcu"])[0] is None
    assert "the SYSMON reads nothing either: the die is unknown" in flat(tab)
    assert "! SYSMON temperature alarm: GO refused" in tab
    d, t = q(FakeBcu(sysmon=False, alarm=True), sysmon=True)
    sm = d["bcu"]["sysmon"]
    assert not sm["valid"] and sm["die_c"] is None and sm["vccint"] is None and not sm["alarm"]
    assert "Rails" not in smi.table([d])
    assert G.sysmon_c(0xB320) == pytest.approx(75.89, abs=0.01)
    assert [G.sysmon_c(v) for v in (0, 0xFFFFFFFF, smi.FENCED, 0xFFC0)] == [None] * 4
    d, t = q(FakeBcu(run=True, gov_valid=False, sysmon=False))
    assert "Die n/a" in smi.table([d]) and smi.csv_row(d)["die_sysmon_c"] == ""


def test_the_governors_own_temp_age(tmp_path):
    """A build with TEMP_AGE: smi takes the governor's count (no -l history), the SYSMON's die
    read only once the governor says stale; a build before it falls back to the samples."""
    none = str(tmp_path / "none.json")

    def q(fk):
        t = Reads(fk)
        return smi.query(t, "/dev/fakebcu0", sleep=lambda s: None, power_json=none), t
    d, t = q(FakeBcu(die_c=40.1, gov_die_c=46.05, temp_age_ms=5000))
    assert d["bcu"]["gov_same_s"] == 5.0 and d["bcu"]["gov_frozen"] is None
    assert d["bcu"]["sysmon"] is None and drp_reads(t) == []
    assert "die 46.0 °C (governor)" in smi.table([d])
    d, t = q(FakeBcu(die_c=40.1, gov_die_c=46.05, temp_age_ms=61000))
    tab = smi.table([d])
    assert d["bcu"]["gov_frozen"] == "frozen 61 s" and drp_reads(t) == [G.R_SYSMON + G.SM_TEMP]
    assert "die 40.1 °C SYSMON  gov 46.0 °C FROZEN 61 s  trip 79.9  resume 55.0" in tab
    assert "has read 46.0 °C for 61 s" in flat(tab) and smi.csv_row(d)["gov_frozen"] == "1"
    d, t = q(FakeBcu(die_c=40.1, gov_die_c=46.05))          # before TEMP_AGE: no history here
    assert d["bcu"]["gov_same_s"] is None and d["bcu"]["gov_frozen"] is None


def test_loop_finds_the_frozen_die(tmp_path, capsys, monkeypatch):
    """-l: the governor's die the same sample after sample is gone at gov.FROZEN_S; the SYSMON's
    die read from then on (one register a sample), never before."""
    fk = FakeBcu(die_c=40.1, gov_die_c=46.05)
    t = Reads(fk)
    now = [1000.0]

    def clock():
        now[0] += 25.0
        return now[0]
    monkeypatch.setattr(smi, "_clock", clock)
    monkeypatch.setattr(smi.time, "sleep", lambda s: None)
    assert smi.main(["--dev", "/dev/fakebcu0", "-l", "1", "--samples", "4", "-i", "0",
                     "--power-json", str(tmp_path / "n.json")], open_transport=lambda dev: t) == 0
    out = capsys.readouterr().out
    assert out.count("FROZEN") == 1 and "gov 46.0 °C FROZEN 75 s" in out
    assert out.count("die 46.0 °C (governor)") == 3 and drp_reads(t) == [G.R_SYSMON + G.SM_TEMP]
    assert t.writes == []


def test_loop_trend(tmp_path, capsys, monkeypatch):
    """-l: the power's and the die's trend once there are two samples."""
    fk = FakeBcu()
    monkeypatch.setattr(smi.time, "sleep", lambda s: None)
    assert smi.main(["--dev", "/dev/fakebcu0", "-l", "1", "--samples", "3", "-i", "0",
                     "--power-json", str(tmp_path / "n.json")], open_transport=lambda dev: fk) == 0
    out = capsys.readouterr().out
    assert out.count("Trend      VCCINT in ▄▄") == 2 and "24.8-24.8 W" in out
    assert "die ▄▄▄ 62.0-62.0 °C" in out and fk.writes == []


def test_other_cards_unchanged(tmp_path):
    """A YPCB-00338 bitstream: no governor read (its 0x10000 is LiteDRAM's CSRs), no board rows,
    the CSV's new cells empty."""
    t = Reads(FakeTransport(devname="fake7", regmap=4))
    d = smi.query(t, "/dev/fake7", sleep=lambda s: None, power_json=str(tmp_path / "n.json"))
    assert "bcu" not in d and not [o for o in t.reads if o >= G.R_GOV]
    tab = smi.table([d])
    assert "Governor" not in tab and "Config" not in tab and "Temp " in tab and "ch0 ok" in tab
    row = smi.csv_row(d)
    assert [row[k] for k in smi.CSV_FIELDS[-7:]] == [""] * 7
    # a card that answers neither as openTPU nor as the BCU's fence: the Link row alone
    class Dead:
        def reg_read(self, off):
            return 0xFFFFFFFF
    d = smi.query(Dead(), "/dev/fake8", sleep=lambda s: None)
    assert d["link"].startswith("down") and "bcu" not in d


# ---------------------------------------------------------------- the two-card machine's link
def _x2_cards(monkeypatch, **kw):
    from opentpu import target as T
    from opentpu.host.cards import IsaCards
    monkeypatch.setenv("OTPU_TARGET", "bcu1525-x2")
    return IsaCards(T.config("bcu1525-x2", DRAM_BYTES=1 << 20), **kw).cards


def test_each_card_of_the_two_card_machine_shows_its_link(monkeypatch):
    """otpu-smi on either card of bcu1525-x2 (IsaCards' fake cards: CAPS2 XCARD, the X2 window):
    before a host has set the cards, "not set" and both links down; after the host's set-up
    (X2_CARD, the link reset), card 0 of 2 and card 1 of 2, each port up with the peer its hello
    named; a sticky error and its CRC count show. The monitor writes nothing."""
    cards = _x2_cards(monkeypatch)
    views = [Reads(k) for k in cards]
    ds = [smi.query(v, f"/dev/fakex2{c}", sleep=lambda s: None, i2c=False)
          for c, v in enumerate(views)]
    for d in ds:
        assert d["ok"] and d["x2"]["ncards"] == 1 and not any(p["up"] for p in d["x2"]["ports"])
    assert "not set yet (a bring-up sets them)" in smi.table(ds) and smi.table(ds).count("DOWN (no hello)") == 4
    for c, k in enumerate(cards):               # the host's set-up (CardsTransport.set_cards')
        k.reg_write(R.R_X2_CARD, c | 2 << 8)
    for k in cards:
        k.reg_write(R.R_X2_CTRL, R.X2_LRST)
    cards[1].port_errors[0] = R.X2_CRC
    cards[1].crc_errors[0] = 3
    ds = [smi.query(v, f"/dev/fakex2{c}", sleep=lambda s: None, i2c=False)
          for c, v in enumerate(views)]
    assert [(d["x2"]["card"], d["x2"]["ncards"]) for d in ds] == [(0, 2), (1, 2)]
    assert ds[0]["x2"]["ports"][0]["peer"] == {"card": 1, "port": 0}
    assert ds[1]["x2"]["ports"][1]["peer"] == {"card": 0, "port": 1}
    assert all(p["up"] for d in ds for p in d["x2"]["ports"])
    tab = smi.table(ds)
    assert "card 0 of 2" in tab and "card 1 of 2" in tab, tab
    assert "QSFP0 up (card 0 port 0) ! CRC, 3 CRC errors" in tab, tab
    assert "QSFP1 up (card 0 port 1)" in tab and "QSFP0 up (card 1 port 0)" in tab, tab
    for v in views:                             # SNAP alone (the counters' latch)
        assert {off for off, _ in v.writes} <= {R.R_SNAP}, v.writes
    json.dumps(smi._jsonable(ds))


def test_a_single_card_shows_no_link_rows():
    d = smi.query(FakeTransport(), "/dev/fake0", sleep=lambda s: None, i2c=False)
    assert "x2" not in d and "QSFP" not in smi.table([d])
