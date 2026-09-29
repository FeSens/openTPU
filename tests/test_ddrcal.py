"""opentpu.host.ddrcal (LiteDRAM DDR3 calibration from the host) against its simulated PHY, and
the test image's host tool (tools/litedram/ld_host.py) on two simulated channels. The CSR map and
init sequence are the two-channel test image's (tools/litedram/ld_test.py)."""
import subprocess
import sys
from pathlib import Path

from opentpu.host import ddrcal as C

DATA = Path(__file__).parent / "data" / "litedram"
ROOT = Path(__file__).resolve().parents[1]


def test_calibrate_channel_on_the_simulated_phy():
    fake = C.FakeCsr(DATA)
    res = C.calibrate_channel(fake, DATA, stride=4, log=lambda *_: None)
    assert res["traffic_checked"]
    assert all(fake.good(m) for m in range(fake.nm))            # the BIST passes where it runs
    assert res["write_latency"] == fake.WB
    assert [r["bitslip"] for r in res["read"]] == fake.RB
    steps = res["dqs_steps"] % 112
    assert all(fake.WLO[m] <= steps <= fake.WHI[m] for m in range(fake.nm))


def test_channel_1_names_reach_the_second_channel():
    board = C.FakeBoard(DATA)
    res = C.calibrate_channel(C.Chan(board, 1), DATA, stride=4, log=lambda *_: None)
    assert res["write_latency"] == board.ch[1].WB and board.ch[0].steps == 0
    assert C.Chan(board, 1).name("ddrphy_dly_sel") == "ddrphy1_dly_sel"
    assert C.Chan(board, 1).name("xadc_temperature") == "xadc_temperature"


def test_no_common_phase_raises():
    fake = C.FakeCsr(DATA)
    fake.WLO[0], fake.WHI[0] = 0, 10          # lane 0 writes only at 0..10,
    fake.WLO[1], fake.WHI[1] = 60, 70         # lane 1 only at 60..70
    try:
        C.calibrate_channel(fake, DATA, stride=2, log=lambda *_: None)
    except C.CalError:
        return
    raise AssertionError("no CalError")


def test_margins_wrap_around_the_tck():
    period, nm = 112, 2
    table = {k: [10, 10] if (k >= 100 or k <= 8) else [10, 0] for k in range(period + 1)}
    per, common, pick, run = C.margins(table, nm, period=period)
    assert run == list(range(100, 112)) + list(range(0, 9)) and pick == run[len(run) // 2]
    assert C.offsets([(k - pick) % period for k in run], period) == (-10, 10)


def test_write_clock_groups_on_the_simulated_phy():
    # a WL7DDRPHY channel whose two groups write in disjoint phase ranges: no common phase with
    # group 1 on group 0, a 25-step one with group 1 six eighths (42 steps) ahead
    groups = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    fake = C.FakeCsr(DATA, groups=groups)
    for m in range(fake.nm):
        fake.WLO[m], fake.WHI[m] = (20 + m, 60 - m) if groups[m] == 0 else (62 + m, 102 - m)
    d = C.Dram(fake, DATA)
    dqs = C.DqsPhase(fake, d.phy["vco_hz"])
    w = C.WriteClocks(fake)
    assert w.check() == {3: 0, 4: 4, 5: 0, 6: 4}
    res = C.calibrate_groups(d, dqs, w, groups, 112, stride=1, csr=fake, mib=1, log=lambda *_: None)
    assert res["offset_eighths"] == 6 and w.group1() == 6 and w.check()[6] == 10
    assert res["group_runs"] == {0: 35, 1: 25} and len(res["run"]) == 25
    wl, rl, err = d.calibrate(verbose=False)
    assert not any(err) and all(fake.good(m) for m in range(fake.nm))


def test_calibrate_channel_write_clock_groups():
    # a WL7DDRPHY channel (the WL image's settings: phy "wl", groups per channel) through the
    # production entry point
    wl = DATA.parent / "litedram_wl"
    board = C.FakeBoard(wl)
    f = board.ch[1]
    for m in range(f.nm):
        f.WLO[m], f.WHI[m] = (20 + m, 60 - m) if f.groups[m] == 0 else (62 + m, 102 - m)
    res = C.calibrate_channel(C.Chan(board, 1), wl, stride=2, log=lambda *_: None)
    # the WL image's phase shift moves CK: group 1's DQ goes 70 steps later (= 42 earlier)
    assert f.ck and res["group1_eighths"] == 10 and board.ch[0].steps == 0
    assert all(f.good(m) for m in range(f.nm))


def test_word_csr_most_significant_word_first():
    mem = {}
    regs = {"a": (0x10, 1), "seed": (0x20, 2)}
    c = C.WordCsr(regs, lambda o: mem.get(o, 0), mem.__setitem__, base=0x10000)
    c.w("seed", 0x0123456789ABCDEF)
    assert mem == {0x10020: 0x01234567, 0x10024: 0x89ABCDEF} and c.r("seed") == 0x0123456789ABCDEF


def test_ld_host_selftest():
    r = subprocess.run([sys.executable, str(ROOT / "tools/litedram/ld_host.py"), str(DATA), "selftest"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode in (0, None) and "selftest: PASS" in r.stdout, r.stdout[-2000:] + r.stderr


def test_ld_host_selftest_write_clock_groups(tmp_path):
    # the WL7DDRPHY image's CSR map and settings (ld_test.py --phy wl): the temperature run picks
    # each channel's group 1 offset first
    for f in ("csr.csv", "sdram_init.py"):
        (tmp_path / f).write_text((DATA.parent / "litedram_wl" / f).read_text())
    r = subprocess.run([sys.executable, str(ROOT / "tools/litedram/ld_host.py"), str(tmp_path),
                        "selftest"], capture_output=True, text=True, timeout=300)
    assert r.returncode in (0, None) and "selftest: PASS" in r.stdout, r.stdout[-2000:] + r.stderr
    assert "group 1 offset" in r.stdout


# ------------------------------------------------------------------------------ otpu-memcal
from opentpu.host import memcal, regs as R          # noqa: E402


class FakeCard:
    """BAR0 of a host-calibrated bitstream: REGMAP 3, CAPS bit27 (unless `hostcal` is off),
    STATUS CALIB0/1 from the two simulated channels' cal_ready, and their CSRs (the production
    core's map, opentpu/host/litedram/csr.csv) in the window at R_MEMCAL."""
    devname = None

    def __init__(self, hostcal=True):
        self.b = C.FakeBoard(memcal.DATA)
        self.caps = R.CAP_HOSTCAL if hostcal else 0
        self.words = {off + 4 * i: (name, i, n) for name, (off, n) in self.b.regs.items()
                      for i in range(n)}
        self.pending, self.csr_ops = {}, 0

    def reg_read(self, off):
        if off == R.R_ID:
            return R.ID_OTPU
        if off == R.R_REGMAP:
            return 3
        if off == R.R_CAPS:
            return self.caps
        if off == R.R_STATUS:
            return (R.ST_CALIB0 if self.b.r("cal_ready") else 0) | \
                   (R.ST_CALIB1 if self.b.r("cal1_ready") else 0)
        name, i, n = self.words[off - R.R_MEMCAL]
        self.csr_ops += 1
        return (self.b.r(name) >> (32 * (n - 1 - i))) & 0xFFFFFFFF

    def reg_write(self, off, v):
        name, i, n = self.words[off - R.R_MEMCAL]
        self.csr_ops += 1
        acc = (self.pending.pop(name, 0) << 32) | v          # most significant word first
        if i == n - 1:
            self.b.w(name, acc)
        else:
            self.pending[name] = acc


def test_memcal_calibrates_each_channel_once(monkeypatch, tmp_path):
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path))
    card = FakeCard()
    res = memcal.ensure(card, stride=4, log=lambda *_: None)
    assert sorted(res["channels"]) == [0, 1]
    assert card.reg_read(R.R_STATUS) & (R.ST_CALIB0 | R.ST_CALIB1) == R.ST_CALIB0 | R.ST_CALIB1
    for ch, f in enumerate(card.b.ch):
        assert all(f.good(m) for m in range(f.nm)) and res["channels"][ch]["write_latency"] == f.WB
    assert memcal.ensure(card, log=lambda *_: None) is None            # calibrated: nothing to do
    assert memcal.last(card)["channels"]["1"]["traffic_checked"]


def test_memcal_leaves_mig_bitstreams_alone():
    card = FakeCard(hostcal=False)
    assert memcal.ensure(card, log=lambda *_: None) is None and card.csr_ops == 0


def test_board_open_calibrates(monkeypatch, tmp_path):
    from opentpu.host.board import Board
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path))
    ensure = memcal.ensure
    monkeypatch.setattr(memcal, "ensure", lambda t, **k: ensure(t, stride=8, **k))
    card = FakeCard()
    Board(card, lock=False).close()
    assert card.reg_read(R.R_STATUS) & (R.ST_CALIB0 | R.ST_CALIB1) == R.ST_CALIB0 | R.ST_CALIB1
    ops = card.csr_ops
    Board(card, lock=False).close()                  # calibrated: no CSR access
    assert card.csr_ops == ops
