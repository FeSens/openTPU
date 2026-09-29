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
