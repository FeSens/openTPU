"""opentpu.host.ddrcal (LiteDRAM DDR3 calibration from the host) against its simulated PHY, and
the test image's host tool (tools/litedram/ld_host.py) on two simulated channels. The CSR map and
init sequence are the two-channel test image's (tools/litedram/ld_test.py)."""
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_a_dqs_phase_move_that_does_not_get_there_times_out():
    """A phase that does not follow its shifts, or stays busy: the move ends with a
    TimeoutError (it looped for ever, inside Board() with the card's lock held)."""
    class Stuck:
        def __init__(self, busy):
            self.busy = busy

        def r(self, name):
            return int(self.busy) if name == "phase_dqs_busy" else 0

        def w(self, name, v):
            pass
    for busy in (False, True):
        with pytest.raises(TimeoutError, match="DQS phase at 0 steps"):
            C.DqsPhase(Stuck(busy), 1e9).move(5, timeout=0.2)
    fake = C.FakeCsr(DATA)
    C.DqsPhase(fake, 1e9).move(-7)
    assert fake.steps == -7


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
    # a WL7DDRPHY channel: group 1 stays on group 0 (its DRP offset breaks its serializers), so
    # the phase is the centre of the run common to every lane; groups that write in disjoint
    # phase ranges have none
    groups = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    fake = C.FakeCsr(DATA, groups=groups)
    for m in range(fake.nm):
        fake.WLO[m], fake.WHI[m] = (20 + m, 60 - m) if groups[m] == 0 else (40 + m, 80 - m)
    d = C.Dram(fake, DATA)
    dqs = C.DqsPhase(fake, d.phy["vco_hz"])
    w = C.WriteClocks(fake)
    assert w.check() == {3: 0, 4: 4, 5: 0, 6: 4}
    res = C.calibrate_groups(d, dqs, w, groups, 112, stride=1, csr=fake, mib=1, log=lambda *_: None)
    assert res["offset_eighths"] == 0 and w.group1() == 0
    assert res["group_runs"] == {0: 35, 1: 25} and len(res["run"]) == 10      # 48..57
    wl, rl, err = d.calibrate(verbose=False)
    assert not any(err) and all(fake.good(m) for m in range(fake.nm))
    for m in range(fake.nm):
        fake.WLO[m], fake.WHI[m] = (20 + m, 60 - m) if groups[m] == 0 else (62 + m, 102 - m)
    with pytest.raises(C.CalError) as e:
        C.calibrate_groups(d, dqs, w, groups, 112, stride=1, csr=fake, mib=1, log=lambda *_: None)
    assert e.value.scan                     # the table, for the pass map


def test_calibrate_channel_write_clock_groups():
    # a WL7DDRPHY channel (the WL image's settings: phy "wl", groups per channel) through the
    # production entry point: one common phase, group 1 left at offset 0
    wl = DATA.parent / "litedram_wl"
    board = C.FakeBoard(wl)
    f = board.ch[1]
    for m in range(f.nm):
        f.WLO[m], f.WHI[m] = (20 + m, 60 - m) if f.groups[m] == 0 else (40 + m, 80 - m)
    res = C.calibrate_channel(C.Chan(board, 1), wl, stride=2, log=lambda *_: None)
    assert f.ck and res["group1_eighths"] == 0 and board.ch[0].steps == 0
    assert all(f.good(m) for m in range(f.nm))


def test_per_bit_read_framing():
    # ldtest3d, channel 1: single DQ bits read a CLK off their lane (dq27 at the lane's read
    # bitslip + 2; dq64 and dq67 at -2 against the other six). With ddrphy_dly_sel_bits (the WL
    # image's map) each takes its own bitslip and the lanes calibrate; without it they cannot
    wl = DATA.parent / "litedram_wl"
    for per_bit in (True, False):
        fake = C.FakeCsr(wl)
        if not per_bit:
            fake.regs = {k: v for k, v in fake.regs.items() if "dly_sel_bits" not in k}
        fake.RB[3], fake.RB[8] = 4, 5
        fake.BOFF = {(3, 3): 2, (8, 0): -2, (8, 3): -2}
        for m in range(fake.nm):
            fake.WLO[m], fake.WHI[m] = 0, 111
        d = C.Dram(fake, wl)
        assert d.per_bit == per_bit
        wl_, rl, err = d.calibrate(verbose=False)
        if per_bit:
            assert not any(err) and all(fake.good(m) for m in range(fake.nm)), (wl_, rl, err)
            assert d.boff[3] == [0, 0, 0, 2, 0, 0, 0, 0] and d.boff[8] == [-2, 0, 0, -2, 0, 0, 0, 0]
            assert fake.v["ddrphy_dly_sel_bits"] == 0xFF
        else:
            assert err[3] and err[8] and not any(e for m, e in enumerate(err) if m not in (3, 8))


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
    # the WL7DDRPHY image's CSR map and settings (ld_test.py --phy wl): the temperature run scans
    # each channel's common CK phase first
    for f in ("csr.csv", "sdram_init.py"):
        (tmp_path / f).write_text((DATA.parent / "litedram_wl" / f).read_text())
    r = subprocess.run([sys.executable, str(ROOT / "tools/litedram/ld_host.py"), str(tmp_path),
                        "selftest"], capture_output=True, text=True, timeout=300)
    assert r.returncode in (0, None) and "selftest: PASS" in r.stdout, r.stdout[-2000:] + r.stderr
    assert "all lanes:" in r.stdout


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


class FusedCard:
    """A LiteDRAM card for otpu-selftest: FakeTransport's registers, with CAPS bit27 and STATUS
    CALIB0/1 and the CSR window of FakeCard's two simulated channels."""

    def __init__(self):
        from opentpu.host.fake import FakeTransport
        self.f, self.cal = FakeTransport(devname=None), FakeCard()
        self.devname = None

    def __getattr__(self, name):
        return getattr(self.f, name)

    def reg_read(self, off):
        if off == R.R_CAPS:
            return self.f.reg_read(off) | R.CAP_HOSTCAL
        if off == R.R_STATUS:
            return self.f.reg_read(off) & ~(R.ST_CALIB0 | R.ST_CALIB1) | self.cal.reg_read(off)
        if R.R_MEMCAL <= off < R.R_MEMCAL + 0x10000:
            return self.cal.reg_read(off)
        return self.f.reg_read(off)

    def reg_write(self, off, v):
        if R.R_MEMCAL <= off < R.R_MEMCAL + 0x10000:
            return self.cal.reg_write(off, v)
        return self.f.reg_write(off, v)

    def reg_read_many(self, offs):
        return [self.reg_read(o) for o in offs]


def _selftest_to_calib(monkeypatch, tmp_path, card):
    """otpu-selftest on `card`, stopped after its calib stage (the scrub fails)."""
    from opentpu.host import selftest
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(selftest, "XdmaTransport", lambda dev: card)
    monkeypatch.setattr(selftest.Board, "scrub", lambda self: 1 / 0)
    assert selftest.main([]) == 1


def test_selftest_calibrates_a_host_calibrated_card(monkeypatch, tmp_path, capsys):
    """otpu-selftest opens its Board without check (its link stage reads the ID itself), so
    the calib stage calibrates a LiteDRAM bitstream's channels (memcal.ensure) before it reads
    STATUS; a second run finds them calibrated."""
    ensure = memcal.ensure
    monkeypatch.setattr(memcal, "ensure", lambda t, **k: ensure(t, stride=8, **k))
    card = FusedCard()
    for how in ("host calibration: channel 0, 1 in", "host calibration: done before"):
        _selftest_to_calib(monkeypatch, tmp_path, card)
        out = capsys.readouterr().out
        assert "[PASS] calib      channel 0 ok, channel 1 ok" in out and how in out, out
        assert "stopped at stage 'scrub'" in out


def test_selftest_calib_fails_on_a_calibration_error(monkeypatch, tmp_path, capsys):
    def fail(t, **k):
        raise C.CalError("channel 1: no read window")
    monkeypatch.setattr(memcal, "ensure", fail)
    _selftest_to_calib(monkeypatch, tmp_path, FusedCard())
    out = capsys.readouterr().out
    assert "[FAIL] calib      CalError: channel 1: no read window" in out
    assert "otpu-memcal cal --force" in out and "stopped at stage 'calib'" in out
