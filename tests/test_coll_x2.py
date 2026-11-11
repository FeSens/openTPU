"""The collective across two cards (sim/verilator/tb_coll_x2.sv: on each card otpu_coll behind
otpu_coll_hub at NC 2, two slices XRS register stages away, the hub's two links each through its
crossing, otpu_coll_lx, to a cable joined to the other card's: qsfp-link's otpu_link_sim, or
sim/verilator/otpu_link_stub.sv for random back-pressure and a header's collective number
changed): GATHERs and BARs across the cards, each card walking the list on its own (up to a few
microseconds apart, a collective ahead, one card's run starting a million cycles late), GATHERs
larger than a link's RX FIFO, back-pressure on the links and on the slices' grants, a link
latency L of 0, 40 and 240 core cycles, the links straight or crossed, FOLLOW's tags; errors
(a flipped word, a dropped message, a link down, ABORT, a header of another collective): the
cards abort and run every collective to its end on their local writes; a GATHER's cycles by
size and L."""
import re

import pytest

from opentpu import rtlsim

SRC = [rtlsim.RTL / "top/otpu_pkg.sv", rtlsim.RTL / "link/otpu_link_pkg.sv",
       rtlsim.RTL / "top/otpu_coll.sv", rtlsim.RTL / "shell/otpu_board.sv",
       rtlsim.TB / "otpu_link_stub.sv", rtlsim.TB / "otpu_link_sim.sv",
       rtlsim.TB / "tb_coll_x2.sv"]
DONE = re.compile(r"TBCX DONE XRS=(\d+) ops=(\d+) gathers=(\d+) bars=(\d+) "
                  r"writes=(\d+)/(\d+) cycles=(\d+)")
STATUS = re.compile(r"TBCX STATUS card=(\d) xst=([0-9a-f]{2}) seq=(\d+) drain=(\d+)")
FOLLOW = re.compile(r"TBCX FOLLOW ops=(\d+) tagged=(\d+)/(\d+)")
SIZE = re.compile(r"TBCX SIZE XRS=(\d+) words=(\d+) first=(\d+) last=(\d+)")
UP = 0x60                                   # X2_HUB: both links up, nothing else
BIG = 1 << 18                               # TMEM words for +big's GATHERs


def _run(*plus, xrs: int = 2, cross: int = 0, lsim: int = 1, tw: int = 1 << 16) -> str:
    exe = rtlsim.build("tb_coll_x2", SRC, {"XRS": xrs, "CROSS": cross, "LSIM": lsim, "TW": tw})
    r = rtlsim.run_sim([str(exe), *plus], timeout=1800)
    out = r.stdout + r.stderr
    assert "TBCX DONE" in out and r.returncode == 0, out[-3000:]
    return out


def _lat(lat: int) -> list:
    return [f"+lnk0_L={lat}", f"+lnk1_L={lat}"]


def _status(out: str) -> dict:
    return {int(c): int(x, 16) for c, x, _, _ in STATUS.findall(out)}


def _drain(out: str) -> dict:
    return {int(c): int(d) for c, _, _, d in STATUS.findall(out)}


def _seq(out: str) -> dict:
    return {int(c): int(q) for c, _, q, _ in STATUS.findall(out)}


@pytest.mark.parametrize("lat", [0, 40, 240])
@pytest.mark.parametrize("seed,stall,cross", [(1, 0, 0), (2, 30, 1), (3, 70, 0)])
def test_gathers_and_bars_across_cards(have_verilator, lat, seed, stall, cross):
    """200 collectives (a fifth BARs) on both cards, card 1 up to 200 cycles behind each time,
    the slices' grants withheld at random: every slice's writes of each piece as the
    reference's, in order, every piece in before the ack, every destination right, FOLLOW's
    tagged writes confined to the local half in order, both cards' X2_SEQ the list's length, and
    neither card's status flags anything."""
    out = _run(f"+seed={seed}", f"+stall={stall}", *_lat(lat), "+ops=200", "+burst=8",
               cross=cross)
    g = DONE.search(out).groups()
    assert int(g[2]) > 120 and int(g[3]) > 20 and int(g[4]) > 1000, out[-500:]
    assert int(g[4]) == int(g[5]), out[-500:]                # both cards wrote every piece
    assert _status(out) == {0: UP, 1: UP}, out[-500:]
    assert _seq(out) == {0: 200, 1: 200}, out[-500:]
    f = [int(x) for x in FOLLOW.search(out).groups()]
    assert f[0] > 0 and f[1] > 0 and f[2] > 0, out[-500:]


@pytest.mark.parametrize("lat", [40, 240])
def test_link_backpressure(have_verilator, lat):
    """The stub's cable: the senders' tready withheld half the cycles at random, the slices'
    grants 30 percent: the collectives stay right."""
    out = _run("+seed=7", "+stall=30", "+xstall=50", f"+xlat={lat}", "+ops=150", lsim=0)
    assert _status(out) == {0: UP, 1: UP}, out[-500:]


def test_start_skew_of_microseconds(have_verilator):
    """Card 1 starts each collective up to 1,500 cycles (10 us at 150 MHz) after card 0, so
    card 0's words wait in card 1's buffers: the collectives stay right (XRS 3)."""
    out = _run("+seed=9", "+stall=20", *_lat(40), "+ops=60", "+cskew=1500", xrs=3)
    assert _status(out) == {0: UP, 1: UP}, out[-500:]


@pytest.mark.parametrize("c1wait", [0, 100000])
def test_gather_larger_than_the_rx_fifo(have_verilator, c1wait):
    """GATHERs of 2,560 words a piece (80 KB a link each way, the RX FIFO 64 KB), both
    directions at once, card 1 starting each 0 or 100,000 cycles after card 0: a card's TX
    stalls on the link's credits while its hub keeps writing the other card's words (TBCX
    DRAIN), and every collective completes."""
    out = _run("+seed=11", "+stall=10", *_lat(40), "+big", "+ops=4", "+gap=0", "+cskew=0",
               f"+c1wait={c1wait}", tw=BIG)
    assert _status(out) == {0: UP, 1: UP}, out[-500:]
    d = _drain(out)
    assert max(d.values()) > 1000, d
    if c1wait != 0:
        assert d[0] > 1000, d


def test_one_card_a_collective_ahead(have_verilator):
    """Back to back large GATHERs, card 1 waiting 20,000 cycles before each: card 0 finishes
    collective k and fills card 1's buffers with k + 1 while card 1 has not issued it, so card
    0's TX stalls on credits; card 0's hub keeps taking card 1's k + 1 words once they come
    (drain independence)."""
    out = _run("+seed=12", "+stall=0", *_lat(40), "+big", "+ops=4", "+gap=0", "+cskew=0",
               "+c1wait=20000", tw=BIG)
    assert _status(out) == {0: UP, 1: UP}, out[-500:]
    assert _drain(out)[0] > 1000, out[-500:]
    ahead = re.search(r"TBCX AHEAD cycles=(\d+)/(\d+)", out)
    assert int(ahead[1]) > 1000, out[-500:]


def test_card_starting_a_million_cycles_late(have_verilator):
    """Card 1's run starts 1,000,000 cycles after card 0's (its links up all along): card 0's
    first large GATHER fills card 1's RX FIFOs and waits; then both run the list."""
    out = _run("+seed=13", "+stall=10", *_lat(40), "+big", "+ops=3", "+gap=0", "+cskew=0",
               "+c1late=1000000", "+wd=67108864", tw=BIG)
    assert _status(out) == {0: UP, 1: UP}, out[-500:]
    assert int(DONE.search(out).group(7)) > 1000000


def test_flipped_word_aborts_both_cards(have_verilator):
    """A payload bit flipped on link 0, card 0 to card 1 (message 5): card 1's link check
    (bit3) aborts it (bit0), its ABORT aborts card 0 (bits 0 and 4), and both cards walk the
    whole list on their local writes (the word's data is not checked: it is wrong)."""
    out = _run("+seed=4", *_lat(40), "+ops=40", "+lnk0a_flip=5")
    st = _status(out)
    assert st[1] & 0x09 == 0x09 and st[0] & 0x11 == 0x11, st
    assert not st[0] & 0x0E, st


def test_dropped_message_aborts_both_cards(have_verilator):
    """Link 1 drops a message, card 1 to card 0 (message 4): card 0's collective waits for it
    until its watchdog (3,000 cycles) aborts it (bits 0 and 1), ABORT stops card 1 (bit 4),
    and both cards walk the rest of the list on their local writes."""
    out = _run("+seed=5", *_lat(40), "+ops=40", "+lnk1b_drop=4", "+wd=3000")
    st = _status(out)
    assert st[0] & 0x03 == 0x03 and st[1] & 0x11 == 0x11, st
    assert not (st[0] | st[1]) & 0x0C, st


def test_link_down_aborts_both_cards(have_verilator):
    """Link 0 drops at cycle 3,000: both ends' rx_err abort both cards (bits 0 and 3)."""
    out = _run("+seed=6", *_lat(40), "+ops=40", "+lnk0_down=3000")
    st = _status(out)
    assert st[0] & 0x09 == 0x09 and st[1] & 0x09 == 0x09, st


def test_abort_from_the_other_card(have_verilator):
    """ABORT from card 1's end of link 1 at cycle 2,000 (as if its hub aborted): card 0 aborts
    (bits 0 and 4) and sends ABORT on, which aborts card 1 (bits 0 and 4)."""
    out = _run("+seed=8", *_lat(40), "+ops=40", "+lnk1b_abort=2000")
    st = _status(out)
    assert st[0] & 0x11 == 0x11 and st[1] & 0x11 == 0x11, st
    assert not (st[0] | st[1]) & 0x0E, st


def test_header_of_another_collective_aborts(have_verilator):
    """A header whose collective number is wrong (the stub's link 0, card 0 to card 1, message
    6): card 1 aborts on it (bits 0 and 2), ABORT stops card 0 (bit 4), and both walk the rest."""
    out = _run("+seed=6", "+xlat=40", "+ops=40", "+xelink=0", "+xedir=0", "+xemsg=6",
               "+xekind=seq", "+coll_xerr_ok", lsim=0)
    st = _status(out)
    assert st[1] & 0x05 == 0x05 and st[0] & 0x11 == 0x11, st


def test_gather_cycles_by_size_and_latency(have_verilator):
    """One-row GATHERs of 16, 256 and 4,096 words a slice (four slices, two cards) at L 1, 40
    and 240 (otpu_link_sim takes L 0 as 1): a small GATHER takes L more, a large one hides it
    (each card writes four pieces, a word a cycle: about 4 x words / 8 cycles)."""
    res = {}
    for lat in (0, 40, 240):
        out = _run("+sizes", "+reps=4", "+stall=0", "+skew=0", "+cskew=0", "+gap=0",
                   *_lat(lat))
        res[lat] = {int(g[1]): int(g[3]) for g in SIZE.findall(out)}
    print("TBCX cycles a GATHER (last of 4) by L:", res)
    for lat in (40, 240):
        assert lat - 1 <= res[lat][16] - res[0][16] <= lat + 40, res
    assert res[0][4096] < 4 * 4096 // 8 + 200, res
    assert res[240][4096] - res[0][4096] < 120, res


def test_link_reset_while_idle_then_traffic(have_verilator):
    """Both cards stop halfway through the list, LRST for 20 cycles, then start again:
    traffic remains correct, status clears and sequence restarts at zero."""
    out = _run("+seed=17", *_lat(40), "+ops=20", "+lrst_idle")
    assert _status(out) == {0: UP, 1: UP}, out[-500:]
    assert _seq(out) == {0: 10, 1: 10}, out[-500:]


@pytest.mark.parametrize("nc", [1, 2])
def test_x2_control_register_window(have_verilator, nc):
    """The fixed host window, link CSR forwarding, LRST width, ERROR, concurrent AXI-Lite
    channels and the NC=1 capability / zero-window behavior."""
    exe = rtlsim.build("tb_x2_ctrl", [rtlsim.RTL / "shell/otpu_ctrl.sv",
                                     rtlsim.TB / "tb_x2_ctrl.sv"], {"NC": nc})
    r = rtlsim.run_sim([str(exe)], timeout=60)
    assert r.returncode == 0 and "TBXCTRL PASS" in r.stdout, r.stdout + r.stderr


def test_board_aborted_collectives_reach_halt(have_verilator):
    """The full two-slice board with no peer: watchdog on the first BAR aborts, subsequent
    GATHER / BAR finish locally, HALT is reached with STATUS ERROR and COLL_STATUS set.
    RUN=0 preserves the sequence and abort; idle LRST clears both."""
    from opentpu import isa as I
    from opentpu.host.board import Board, SimTransport
    t = SimTransport(ch_bytes=1 << 20, target="bcu1525", native=True,
                     params={"NCARDS": 2, "PLATFORM": 7, "COLL_PIPE": 1},
                     plusargs=["+max_cycles=200000"])
    b = Board(t, check=False)
    prog = [I.bar(), I.gather(0, 1024, 1, 16, 16, 16, 16), I.bar(), I.halt()]
    b.load(0, [prog, prog])
    for a, v in [(0x300, 0x200), (0x304, 4), (0x364, 256)]:
        t.reg_write(a, v)
    t.wait_cycles(80)
    t.reg_write(0x08, 1)
    t.poll(0x0C, 1, 1)
    reads = [t.queue_read(a) for a in (0x0C, 0x360, 0x30C, 0x5C, 0x98)]
    t.reg_write(0x08, 0)
    after_run = t.queue_read(0x30C)
    t.reg_write(0x304, 4)
    t.wait_cycles(80)
    after_link = [t.queue_read(a) for a in (0x360, 0x30C)]
    t.flush()
    status, coll, seq, caps, platform = [t.results[i] for i in reads]
    assert status & 3 == 3 and coll & 3 == 3
    assert seq == 3 and caps & 0x8000 and platform == 7
    assert t.results[after_run] == 3
    assert [t.results[i] for i in after_link] == [0, 0]


def test_two_boards_host_script_smoke(have_verilator, tmp_path, monkeypatch):
    """tb_x2: the BCU wrapper, bridges and DIMM pairs on two boards; @card script lines and
    32 global channel files as X2SimTransport expects. Four distinct sources gathered and
    stored by every slice, first RUN 4,096 cycles early. Storage sizes reduced for the smoke."""
    # Four full slices plus the DIMM models make large C++ translation units. Use two
    # compiler workers and split them, with optimization off (RTL cycle counts unchanged),
    # so the compiler stays within the shared bench's 15 GB budget.
    flags = list(rtlsim.VFLAGS)
    flags[flags.index("-j") + 1] = "2"
    for key, value in (("--output-groups", "8"), ("--output-split", "10000"),
                       ("--output-split-cfuncs", "5000")):
        if key in flags:
            flags[flags.index(key) + 1] = value
        else:
            flags += [key, value]
    monkeypatch.setattr(rtlsim, "VFLAGS", flags)
    monkeypatch.setenv("MAKEFLAGS", "OPT_FAST=-O0 OPT_SLOW=-O0")
    monkeypatch.setenv("CCACHE_DIR", str(tmp_path / "ccache"))
    monkeypatch.setenv("CCACHE_DISABLE", "1")
    import numpy as np
    from opentpu import isa as I
    from opentpu.host.board import SimTransport, join, split
    t = SimTransport(ch_bytes=1 << 16, target="bcu1525-s2", bcuwg=True,
                     params={"WG": 8, "MCOLS": 8, "ACT_ROWS": 16, "ACT_BLOCKS": 8,
                             "TMEM_WORDS": 8192, "IMEM_WORDS": 256, "FIFO_DEPTH": 32,
                             "SEQ_WC": 0, "ICACHE": 0, "TRACE_DEPTH": 0,
                             "PLATFORM": 7, "NAT_XPOST": 1})
    _, srcs, params = t.sim_build()
    srcs = [s for s in srcs if s.name != "tb_bcuwg.sv"]
    srcs += [rtlsim.RTL / "link/otpu_link_pkg.sv", rtlsim.TB / "otpu_link_sim.sv",
             rtlsim.TB / "tb_x2.sv"]
    exe = rtlsim.build("tb_x2", srcs, params | {"NCARDS": 2})
    prog = [I.ld(0x1000, 0, 32), I.gather(0, 1024, 1, 32, 32, 32, 32), I.bar(),
            I.st(0x2000, 1024, 128), I.halt()]
    words = np.asarray(I.assemble(prog), np.uint32).view(np.uint8)
    sources = [np.arange(32, dtype=np.uint32) + 0x1000 * g for g in range(4)]
    for g in range(4):
        img = np.zeros(8 << 16, np.uint8)
        img[:len(words)] = words
        img[0x1000:0x1080] = sources[g].view(np.uint8)
        for c, off, data in split(0, img, True, 8, 128):
            assert off == 0
            (tmp_path / f"ch{8*g+c}.bin").write_bytes(data.view("<u4").astype(">u4").tobytes())
            (tmp_path / f"ch{8*g+c}_hi.txt").write_text("")
    script = ["@0 W 300 200", "@1 W 300 201", "@0 W 304 4", "@1 W 304 4", "C 80"]
    for c in range(2):
        script += [f"@{c} P 310 80 80", f"@{c} P 320 80 80", f"@{c} R 314",
                   f"@{c} W 364 2000", f"@{c} W 10 0", f"@{c} W 14 {len(prog):x}", f"@{c} W 8 2",
                   f"@{c} P c 4 0"]
    script += ["@0 W 8 1", "C 1000", "@1 W 8 1"]
    for c in range(2):
        script += [f"@{c} P c 1 1", f"@{c} P c 8 8",
                   *[f"@{c} R {a:x}" for a in (0xC, 0x360, 0x30C, 0x20)]]
    (tmp_path / "host.txt").write_text("\n".join(script) + "\n")
    r = rtlsim.run_sim([str(exe), f"+dir={tmp_path}", "+max_cycles=20000", "+link_lat=40", "+x2_debug"],
                       timeout=1800)
    out = r.stdout + r.stderr
    progress_file = tmp_path / "x2_progress.log"
    progress = progress_file.read_text() if progress_file.exists() else ""
    assert r.returncode == 0 and "DONE cycles=" in out, out[-4000:] + "\n" + progress[-6000:]
    vals = [int(v, 16) for _, v in re.findall(r"^REG ([0-9a-f]+) ([0-9a-f]+)$", out, re.M)]
    assert vals[:2] == [0xA5002001, 0xA5002000], vals
    for st, coll, seq, ic in (vals[2:6], vals[6:10]):
        assert st & 3 == 1 and coll == UP and seq == 2 and ic == len(prog), vals
    want = np.concatenate(sources).view(np.uint8)
    for g in range(4):
        parts = [np.fromfile(tmp_path / f"ch{8*g+c}_out.bin", np.uint8) for c in range(8)]
        img = join(parts, 0, True, 128)
        assert np.array_equal(img[0x2000:0x2200], want), g


def test_watchdog_counts_idle_cycles_not_total_cycles(have_verilator):
    """A 256-cycle watchdog on large GATHERs taking tens of thousands of cycles: traffic
    continues to make progress, so neither card aborts."""
    out = _run("+seed=21", *_lat(40), "+big", "+ops=2", "+gap=0", "+cskew=0",
               "+stall=10", "+wd=256", tw=BIG)
    assert _status(out) == {0: UP, 1: UP}, out[-500:]
    assert int(DONE.search(out).group(7)) > 10000


def test_run_reset_while_idle_preserves_link_sequence(have_verilator):
    """Both runs stop halfway and resume without LRST: FIFO / credit state and X2_SEQ
    survive RUN=0, the subsequent collectives still use matching sequence numbers."""
    out = _run("+seed=22", *_lat(40), "+ops=20", "+run_idle")
    assert _status(out) == {0: UP, 1: UP}, out[-500:]
    assert _seq(out) == {0: 20, 1: 20}, out[-500:]
