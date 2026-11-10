"""The x2 link (docs/x2_link.md): the co-sim cable model otpu_link_sim
(sim/verilator/tb_x2_link_sim.sv: a hub-side agent at each end sending random messages both
ways, the receivers' m_tready withheld) and the link port's RTL (rtl/link/)."""
import re

import pytest

from opentpu import rtlsim

PKG = rtlsim.RTL / "link/otpu_link_pkg.sv"
SIM_SRC = [PKG, rtlsim.TB / "otpu_link_sim.sv", rtlsim.TB / "tb_x2_link_sim.sv"]
@pytest.fixture(autouse=True)
def _bounded_builds(monkeypatch):
    # One build at a time; two compiler jobs keep this four-end bench below the
    # simulation host's memory budget. The flags remain in rtlsim's content-addressed build key.
    flags = list(rtlsim.VFLAGS)
    flags[flags.index("-j") + 1] = "2"
    monkeypatch.setattr(rtlsim, "VFLAGS", flags)


DONE = re.compile(r"TBLS DONE L=(\d+) msgs=(\d+) words=(\d+),(\d+) got=(\d+),(\d+) "
                  r"rx_err=(\d+),(\d+) inflight_max=(\d+),(\d+) lat=(-?\d+)\.\.(-?\d+) "
                  r"abort=(-?\d+) cycles=(\d+)")


def _sim(*plus, L=40, rate=591):
    exe = rtlsim.build("tb_x2_link_sim", SIM_SRC, {"L": L, "RATE_Q8": rate})
    r = rtlsim.run_sim([str(exe), *plus], timeout=1800)
    out = r.stdout + r.stderr
    assert "TBLS PASS" in out and r.returncode == 0, out[-3000:]
    return DONE.search(out).groups()


@pytest.mark.parametrize("L", [1, 40, 240])
@pytest.mark.parametrize("seed,stall,gap", [(1, 0, 0), (2, 30, 10), (3, 70, 40)])
def test_sim_messages_both_ways(have_verilator, L, seed, stall, gap):
    """1,000 random messages each way (LEN 0-32), the receivers withholding m_tready stall
    percent of the cycles and the senders idle gap percent: every message whole, in order, as
    sent; no rx_err; the words in flight within the credits."""
    g = _sim(f"+seed={seed}", f"+stall={stall}", f"+gap={gap}", "+msgs=1000", L=L)
    assert int(g[4]) == int(g[5]) == 1000 and g[6:8] == ("0", "0"), g


@pytest.mark.parametrize("L", [1, 40, 240])
def test_sim_latency(have_verilator, L):
    """Idle messages: on the far end's m_* L cycles after the header is taken (L 1: the least)."""
    g = _sim("+lat", "+msgs=20", L=L)
    assert int(g[10]) == int(g[11]) == L, g


@pytest.mark.parametrize("L", [1, 40, 240])
def test_sim_back_pressure(have_verilator, L):
    """b holds m_tready low for 20,000 cycles mid-stream: a's sender stops on the credits (its
    words in flight reach the RX FIFO's credits, not beyond), and everything arrives after."""
    g = _sim("+seed=4", "+msgs=2000", "+len=32", "+hold=20000", L=L)
    assert int(g[4]) == 2000 and int(g[8]) >= 2000 - 8 - 34, g


def test_sim_rate_cap(have_verilator):
    """RATE_Q8 128 (half a word a cycle): 1,000 messages of 32 words take about twice the words'
    cycles (the trailer counted); 591 (11.1 GB/s) never binds at a word a cycle."""
    slow = _sim("+msgs=1000", "+len=32", L=40, rate=128)
    fast = _sim("+msgs=1000", "+len=32", L=40)
    words = 1000 * 33
    assert 2 * words * 0.95 < int(slow[13]) < 2 * words * 1.15, slow
    assert int(fast[13]) < words * 1.1, fast


@pytest.mark.parametrize("n", [5, 6])
def test_sim_flip_and_drop(have_verilator, n):
    """+lnk0a_flip: a's n-th message reaches b with its first payload bit flipped and one rx_err
    (LEN 0: dropped, rx_err); +lnk0a_drop: a's n-th message never arrives, no rx_err."""
    g = _sim("+seed=5", "+msgs=50", f"+lnk0a_flip={n}")
    assert g[7] == "1", g
    g = _sim("+seed=5", "+msgs=50", f"+lnk0a_drop={n}")
    assert int(g[4]) == 49 and g[7] == "0", g


def test_sim_abort_and_down(have_verilator):
    """a's abort reaches b's abort_in L cycles later; the link dropping raises rx_err at both
    ends and lowers up."""
    g = _sim("+msgs=50", "+abort=300")
    assert int(g[12]) == 340, g
    _sim("+msgs=400", "+lnk0_down=700")


LINK = [PKG] + [rtlsim.RTL / f"link/{m}.sv" for m in
                ("otpu_link_cdc", "otpu_link_lane", "otpu_link_port", "otpu_link")]
XL_SRC = LINK + [rtlsim.TB / f"{m}.sv" for m in
                 ("otpu_link_sim", "otpu_link_chan", "otpu_link_pair", "tb_x2_link")]
XL = re.compile(r"TBXL DONE MODEL=(\d+) msgs=(\d+) got=(\d+),(\d+),(\d+),(\d+) lost=(\d+) "
                r"bad=(\d+) rx_err=(\d+),(\d+),(\d+),(\d+) lat=(-?\d+)\.\.(-?\d+) "
                r"abort=(-?\d+) up_at=(\d+) cycles=(\d+)")


def _xl(*plus, model=0, L=40, digest=False):
    exe = rtlsim.build("tb_x2_link", XL_SRC, {"MODEL": model, "L": L})
    r = rtlsim.run_sim([str(exe), *plus], timeout=1800)
    out = r.stdout + r.stderr
    assert "TBXL PASS" in out and r.returncode == 0, out[-3000:]
    g = XL.search(out).groups()
    return (g, re.search(r"TBXL DIGEST (.*)", out)[1]) if digest else g


@pytest.mark.parametrize("model", [0, 1])
@pytest.mark.parametrize("seed,stall,gap", [(1, 0, 0), (2, 30, 10), (3, 70, 40)])
def test_link_messages_four_ends(have_verilator, model, seed, stall, gap):
    """Two cards' link ports joined by two cables (RTL: otpu_link + the Aurora stand-in; model:
    otpu_link_sim), 300 random messages from each of the four ends: every message whole and in
    order, headers (TYPE, LEN, ARG, SRC, ROW, COL) and payloads as sent, no rx_err; every end up
    and (RTL) its hello register naming the far card and port."""
    g = _xl(f"+seed={seed}", f"+stall={stall}", f"+gap={gap}", "+msgs=300", model=model)
    assert all(int(x) == 300 for x in g[2:6]) and g[8:12] == ("0",) * 4, g


def test_link_back_pressure(have_verilator):
    """The receivers hold m_tready low 30,000 core cycles mid-stream (64 KB RX FIFOs fill, the
    senders stop on credits): nothing lost, no rx_err."""
    g = _xl("+seed=4", "+msgs=1500", "+len=32", "+hold=30000")
    assert all(int(x) == 1500 for x in g[2:6]) and g[8:12] == ("0",) * 4, g


def test_link_latency(have_verilator):
    """Idle messages, header taken to header on the far m_*, in core cycles: the RTL (Aurora +
    cable stand-in 58 user_clk cycles, s6's 144 ns) against the model's L 40."""
    rtl = _xl("+lat", "+msgs=20")
    mdl = _xl("+lat", "+msgs=20", model=1)
    print(f"latency core cycles: RTL {rtl[12]}..{rtl[13]}, model {mdl[12]}..{mdl[13]}")
    assert 25 <= int(rtl[12]) <= int(rtl[13]) <= 45, rtl
    assert int(mdl[12]) == int(mdl[13]) == 40, mdl


@pytest.mark.parametrize("word", [40, 41, 300, 301, 302, 777])
def test_link_corrupted_word(have_verilator, word):
    """Bit 0 of the n-th wire word card a's port 0 sends flips on the cable (a header: CHK fails,
    the message is lost; a payload word: delivered altered, the trailer's CRC fails; a trailer
    or a control word: dropped or caught): an rx_err at b.p0, at most one message lost or
    altered, every other end clean."""
    g = _xl("+seed=7", "+msgs=200", "+tolerate", f"+chan0a_flip={word}")
    assert int(g[10]) >= 1 and int(g[6]) + int(g[7]) <= 1, g


def test_link_abort_and_down(have_verilator):
    """a.p0's abort reaches b.p0's abort_in; cable 0 dropping raises rx_err and lowers up at
    both its ends."""
    g = _xl("+msgs=100", "+abort=3000")
    assert int(g[14]) > 0, g
    _xl("+msgs=400", "+chan0_down=4000")


@pytest.mark.parametrize("seed,stall,gap", [(11, 0, 0), (12, 35, 10), (13, 80, 40)])
def test_link_swap_identical_messages(have_verilator, seed, stall, gap):
    """Identical deterministic messages, checked word for word at all four ends, have the
    same full-stream CRC in RTL and model despite independent timing and back pressure."""
    args = [f"+seed={seed}", f"+stall={stall}", f"+gap={gap}", "+msgs=1000"]
    rtl, rd = _xl(*args, digest=True)
    mdl, md = _xl(*args, model=1, digest=True)
    assert rtl[2:6] == mdl[2:6] == ("1000",) * 4
    assert rd == md


def _ctrl(*plus):
    src = LINK + [rtlsim.TB / f"{m}.sv" for m in
                  ("otpu_link_chan", "otpu_link_pair", "tb_x2_link_ctrl")]
    exe = rtlsim.build("tb_x2_link_ctrl", src)
    r = rtlsim.run_sim([str(exe), *plus], timeout=180)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "TBXC PASS" in out, out[-5000:]
    return out


def test_link_registers_ping_abort_reset(have_verilator):
    _ctrl("+case=0")


def test_link_hello_agreement_and_changes(have_verilator):
    _ctrl("+case=1")


@pytest.mark.parametrize("src", range(4))
@pytest.mark.parametrize("kind,fault", [(0, "hflip"), (1, "pflip"), (2, "tflip"),
                                       (3, "drop_hdr"), (4, "drop_payload"), (5, "drop_trl"),
                                       (6, "hflip"), (7, "chk_flip"), (8, "len_bad")])
def test_link_directed_fault_and_resync(have_verilator, src, kind, fault):
    _ctrl("+case=2", f"+src={src}", f"+kind={kind}",
          f"+chan{src % 2}{'a' if src < 2 else 'b'}_{fault}=0")


@pytest.mark.parametrize("mode,fault", [(3, "down"), (4, "lane_down")])
def test_link_channel_and_lane_loss(have_verilator, mode, fault):
    _ctrl(f"+case={mode}", f"+chan0_{fault}=4000")


def test_link_soft_error_counter(have_verilator):
    _ctrl("+case=5", "+chan0_soft=4000:17")


def test_link_long_tx_ready_stall(have_verilator):
    _xl("+msgs=500", "+stall=40", "+chan0_stall=4000:5000", "+chan1_stall=5000:3000")


@pytest.mark.parametrize("stall", [0, 75])
@pytest.mark.parametrize("pa,pb", [(0, 0), (1, 1), (0, 1), (1, 0)])
def test_link_pair_exact_model_ports(have_verilator, stall, pa, pb):
    # The pre-existing two-agent model bench, unchanged signal connections and scoreboard.
    src = LINK + [rtlsim.TB / f"{m}.sv" for m in
                  ("otpu_link_sim", "otpu_link_chan", "otpu_link_pair", "tb_x2_link_sim")]
    exe = rtlsim.build("tb_x2_link_sim", src, {"PAIR": 1, "PA": pa, "PB": pb})
    r = rtlsim.run_sim([str(exe), "+msgs=1000", f"+stall={stall}"], timeout=180)
    assert r.returncode == 0 and "TBLS PASS" in r.stdout, (r.stdout + r.stderr)[-4000:]


def test_link_credit_total_rollover(have_verilator):
    _xl("+msgs=3000", "+len=32", "+stall=50", "+hold=40000")


def test_link_crc_reference_vectors(have_verilator):
    _ctrl("+case=6")


def test_link_generic_lint(have_verilator):
    import subprocess
    r = subprocess.run(["verilator", "--lint-only", "-Wall", "--top-module", "otpu_link",
                        *map(str, LINK)], capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr


def test_link_phy_lint(have_verilator):
    import subprocess
    root = rtlsim.ROOT / "boards/bcu1525"
    phy = root / "link"
    sources = [phy / "sim/otpu_link_phy_stubs.sv",
               rtlsim.RTL / "link/otpu_link_cdc.sv", phy / "otpu_link_gtcommon_bcu.v",
               phy / "otpu_link_aurora_bcu.sv", phy / "otpu_link_phy_bcu.sv"]
    r = subprocess.run(["verilator", "--lint-only", "-Wall", "--top-module", "otpu_link_phy_bcu",
                        *map(str, sources)], capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
