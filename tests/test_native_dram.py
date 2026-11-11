"""otpu_native_dram alone (sim/verilator/tb_ndram.sv): random traffic on ports A, SW, B and W
through the native memory model, against a reference memory, at the cards' channels (two of
64-byte beats), the WG targets' (eight of 128: port W's 1 KiB wide chunks, its beats lane by lane) and
two in between; and port W's rate on its own. Also through otpu_xstg's register stages between
the traffic and the adapter (tb_ndram XSTG: otpu_board NAT_XSTG)."""
import re

import pytest

from opentpu import rtlsim

SRC = [rtlsim.RTL / "tech/generic/otpu_tech_ram.sv", rtlsim.RTL / "mem/otpu_native_dram.sv",
       rtlsim.RTL / "mem/otpu_xstg.sv", rtlsim.TB / "otpu_native_mem.sv", rtlsim.TB / "tb_ndram.sv"]
DONE = re.compile(r"TBN DONE cycles=(\d+) b_rd=(\d+) b_wr=(\d+) a_rd=(\d+) a_wr=(\d+) sw=(\d+) "
                  r"w_rd=(\d+)")
SWQ = re.compile(r"sw_fill=(\d+) sw_beats=(\d+) sw_masked=(\d+)")


def _run(nch: int, cb: int, *plus, params: dict | None = None) -> str:
    exe = rtlsim.build("tb_ndram", SRC, {"NCH": nch, "CB": cb} | (params or {}))
    r = rtlsim.run_sim([str(exe), *plus], timeout=1800)
    out = r.stdout + r.stderr
    assert "TBN DONE" in out, out[-3000:]
    return out


@pytest.mark.parametrize("nch,cb", [(2, 64), (8, 128), (4, 64), (2, 128)])
@pytest.mark.parametrize("seed,stall", [(1, 0), (2, 30)])
def test_random_traffic_on_the_adapter_rtl(have_verilator, nch, cb, seed, stall):
    """Epochs of reads and writes on every port (B word-masked chunks, A and SW byte-enabled
    words, SW in runs and strides, W wide chunks), each port's reads in order with their data
    as issued, rd_idle never with a read outstanding, every write in memory once wr_idle; the
    next epoch writes what the last one read and reads what it wrote (tb_ndram)."""
    out = _run(nch, cb, f"+seed={seed}", f"+axi_seed={seed}", f"+axi_stall={stall}",
               "+epochs=20", "+ops=2000")
    cyc, brd, bwr, ard, awr, sw, wrd = map(int, DONE.search(out).groups())
    assert min(brd, bwr, ard, awr, sw) > 1000, out[-500:]
    assert (wrd > 1000) == (nch * cb > 128) and (wrd == 0) == (nch * cb == 128)
    fill, beats, masked = map(int, SWQ.search(out).groups())
    assert fill > 100 and masked == 0, out[-500:]          # (SW_FILL: partial beats read first)


@pytest.mark.parametrize("nch,cb", [(8, 128), (2, 64)])
@pytest.mark.parametrize("seed,stall", [(1, 0), (2, 30)])
def test_random_traffic_without_fill_reads_rtl(have_verilator, nch, cb, seed, stall):
    """SW_FILL 0 (a memory that takes byte masks: the WG targets' memory): the same random traffic, a
    partial SW beat written once with its mask, no fill read, the memory the reference's."""
    out = _run(nch, cb, f"+seed={seed}", f"+axi_seed={seed}", f"+axi_stall={stall}",
               "+epochs=20", "+ops=2000", params={"SW_FILL": 0})
    assert min(map(int, DONE.search(out).groups()[1:6])) > 1000, out[-500:]
    fill, beats, masked = map(int, SWQ.search(out).groups())
    assert fill == 0 and masked > 100 and beats >= masked, out[-500:]


@pytest.mark.parametrize("seed,stall", [(1, 0), (2, 30)])
def test_random_traffic_at_the_wg_runs_rtl(have_verilator, seed, stall):
    """The WG targets' channels with its A runs (targets/bcu1525-x2.toml NAT_APF / NAT_AD: shorter runs, a
    shallower A FIFO) under the same random traffic."""
    from opentpu import target as T
    u = T.load("bcu1525-x2").uarch
    out = _run(8, 128, f"+seed={seed}", f"+axi_seed={seed}", f"+axi_stall={stall}", "+epochs=20",
               "+ops=2000", params={"APF": u["NAT_APF"], "AD": u["NAT_AD"]})
    assert min(map(int, DONE.search(out).groups()[1:])) > 1000, out[-500:]


@pytest.mark.parametrize("seed,stall,bw,swg", [(1, 0, 100, 0), (2, 30, 100, 0), (3, 0, 15, 0),
                                              (4, 30, 15, 0), (5, 0, 100, 1), (6, 30, 15, 1)])
def test_random_traffic_with_posted_write_data_rtl(have_verilator, seed, stall, bw, swg):
    """WPOST (the WG targets': B's and SW's write data posted to the channel at the take, into the
    model's write-data FIFO of 32 beats with its free count, as the bridges' u_wd): the same random
    traffic, every write going out in its data's order on its channel and never into a full FIFO
    (the adapter's checks), the memory the reference's; at a channel rate of 15% of a command a
    cycle the FIFOs fill and the takes wait on their room. Port A reads only (the slice's). swg:
    one SW gather beat for every channel and one posted beat a cycle (SWG; the adapter checks the
    one beat, and that the B stage waits exactly when an SW beat goes in on any channel)."""
    out = _run(8, 128, f"+seed={seed}", f"+axi_seed={seed}", f"+axi_stall={stall}",
               f"+axi_bw={bw}", "+epochs=20", "+ops=2000", "+busy=90",
               params={"SW_FILL": 0, "WPOST": 1} | ({"SWG": 1} if swg else {}))
    cyc, brd, bwr, ard, awr, sw, wrd = map(int, DONE.search(out).groups())
    assert min(brd, bwr, ard, sw, wrd) > 1000 and awr == 0, out[-500:]
    waits, bwaits = map(int, re.search(r"TBN WPOST room_waits=(\d+) b_waits=(\d+)", out).groups())
    assert bwaits > 0 and (waits > 1000 if bw < 100 else True), out[-500:]


@pytest.mark.parametrize("seed,stall,bw,swg", [(1, 0, 100, 0), (2, 30, 15, 0), (3, 60, 25, 0),
                                              (4, 30, 15, 1)])
def test_random_traffic_with_posted_w_beats_rtl(have_verilator, seed, stall, bw, swg):
    """XPOST ((a) v2, the WG targets': port W's beats posted into the MXU's chunk FIFO by the channels,
    as the bridges will): the same random traffic with WPOST, every W read's beats on the model's
    posted port (p_we, the read's entry, its data, each lane in its reads' order), none through
    the adapter, and rd_idle covering them (from the adapter's synchronized counts); a third of
    the W reads one beat (w_one: a natural MM's chunk, on its channel's lane alone)."""
    out = _run(8, 128, f"+seed={seed}", f"+axi_seed={seed}", f"+axi_stall={stall}",
               f"+axi_bw={bw}", "+epochs=12", "+ops=2000", "+busy=90",
               params={"SW_FILL": 0, "WPOST": 1, "XPOST": 1} | ({"SWG": 1} if swg else {}))
    cyc, brd, bwr, ard, awr, sw, wrd = map(int, DONE.search(out).groups())
    assert min(brd, bwr, ard, sw, wrd) > 1000 and awr == 0, out[-500:]
    beats, one = map(int, re.search(r"TBN XPOST beats=(\d+) one_beat_reads=(\d+)", out).groups())
    assert beats == 8 * (wrd - one) + one and one > wrd // 5, out[-500:]


@pytest.mark.parametrize("mut", ["OTPU_MUT_XSLOT", "OTPU_MUT_XIDLE"])
def test_posted_w_beat_mutants_are_caught_rtl(have_verilator, monkeypatch, mut):
    """XPOST's mutants fail tb_ndram without the adapter's own checks (OTPU_WO_UNCHECKED): a beat
    posted into the next entry (OTPU_MUT_XSLOT: the testbench's entry check) and the W reads in
    flight not counted (OTPU_MUT_XIDLE: rd_idle with posted beats outstanding)."""
    monkeypatch.setattr(rtlsim, "VFLAGS",
                        rtlsim.VFLAGS + [f"+define+{d}" for d in (mut, "OTPU_WO_UNCHECKED")])
    exe = rtlsim.build("tb_ndram", SRC, {"NCH": 8, "CB": 128, "SW_FILL": 0, "WPOST": 1,
                                         "XPOST": 1})
    r = rtlsim.run_sim([str(exe), "+seed=3", "+axi_seed=3", "+axi_stall=30", "+axi_bw=25",
                        "+epochs=6", "+ops=1500", "+busy=90"], timeout=1200)
    out = r.stdout + r.stderr
    assert "TBN DONE" not in out and ("entry" in out or "rd_idle with reads" in out), out[-800:]


@pytest.mark.parametrize("xstg", [1, 2, 3])
@pytest.mark.parametrize("nch,cb,wg", [(2, 64, False), (8, 128, True)])
@pytest.mark.parametrize("seed,stall", [(1, 0), (2, 30)])
def test_random_traffic_through_stages_rtl(have_verilator, xstg, nch, cb, wg, seed, stall):
    """otpu_xstg's stages (xstg each way: the queues on the adapter's side, the slice's rdy their
    room for 2 xstg + 1 more, the idles back with the queues' and held after each send) under
    the same random traffic: every read in order with its data, rd_idle never with a read
    outstanding (the testbench's every cycle), the memory the reference's once wr_idle; at the
    cards' channels, and at the WG targets' with WPOST and XPOST (the posts and their counts staged:
    rd_idle covering the posted beats)."""
    params = {"XSTG": xstg} | ({"SW_FILL": 0, "WPOST": 1, "XPOST": 1} if wg else {})
    out = _run(nch, cb, f"+seed={seed}", f"+axi_seed={seed}", f"+axi_stall={stall}",
               "+epochs=12", "+ops=2000", "+busy=90", params=params)
    cyc, brd, bwr, ard, awr, sw, wrd = map(int, DONE.search(out).groups())
    assert min(brd, bwr, ard, sw) > 1000 and (wrd > 1000) == wg, out[-500:]
    if wg:
        m = re.search(r"TBN XPOST beats=(\d+) one_beat_reads=(\d+)", out)
        beats, one = map(int, m.groups())
        assert beats == 8 * (wrd - one) + one, out[-500:]


@pytest.mark.parametrize("xstg,xstgp,swg", [(3, 4, 0), (2, 5, 0), (0, 3, 0), (3, 4, 1)])
@pytest.mark.parametrize("seed,stall", [(1, 0), (2, 30)])
def test_posts_through_their_own_stages_rtl(have_verilator, xstg, xstgp, swg, seed, stall):
    """The posted W beats and their counts through stages of their own (tb_ndram XSTG_P:
    otpu_board NAT_XSTG_P, p_slot crossing as one copy), more than the ports' or with the ports
    wired, also with one SW gather beat and one posted beat a cycle (SWG, the WG targets'): every lane's
    posted beat into its entry, in order, and rd_idle covering them."""
    out = _run(8, 128, f"+seed={seed}", f"+axi_seed={seed}", f"+axi_stall={stall}",
               "+epochs=12", "+ops=2000", "+busy=90",
               params={"XSTG": xstg, "XSTG_P": xstgp, "SW_FILL": 0, "WPOST": 1, "XPOST": 1,
                       "SWG": swg})
    cyc, brd, bwr, ard, awr, sw, wrd = map(int, DONE.search(out).groups())
    assert min(brd, bwr, ard, sw, wrd) > 1000, out[-500:]
    m = re.search(r"TBN XPOST beats=(\d+) one_beat_reads=(\d+)", out)
    beats, one = map(int, m.groups())
    assert beats == 8 * (wrd - one) + one, out[-500:]


@pytest.mark.parametrize("mut", ["OTPU_MUT_XHIST", "OTPU_MUT_XROOM", "OTPU_MUT_XHEAD"])
def test_stage_mutants_are_caught_rtl(have_verilator, monkeypatch, mut):
    """otpu_xstg's mutants fail tb_ndram without the queue's own check (OTPU_WO_UNCHECKED): the
    idles without the slice's sends in flight (OTPU_MUT_XHIST: rd_idle with reads outstanding,
    or a write not yet in memory), the rdy one request short of the room (OTPU_MUT_XROOM: a
    queue overrun, its requests lost or reordered), and port A's and W's heads in flip-flops
    without the write through behind a pop (OTPU_MUT_XHEAD: a stale request at the head)."""
    monkeypatch.setattr(rtlsim, "VFLAGS",
                        rtlsim.VFLAGS + [f"+define+{d}" for d in (mut, "OTPU_WO_UNCHECKED")])
    exe = rtlsim.build("tb_ndram", SRC, {"NCH": 8, "CB": 128, "SW_FILL": 0, "WPOST": 1,
                                         "XPOST": 1, "XSTG": 3})
    r = rtlsim.run_sim([str(exe), "+seed=3", "+axi_seed=3", "+axi_stall=60", "+axi_bw=25",
                        "+epochs=6", "+ops=1500", "+busy=90"], timeout=1200)
    out = r.stdout + r.stderr
    assert "TBN DONE" not in out and "tb_ndram" in out, out[-800:]


@pytest.mark.parametrize("xstg,stream", [(2, "w"), (3, "w"), (3, "b")])
def test_stages_keep_a_chunk_a_cycle_rtl(have_verilator, xstg, stream):
    """Through xstg stages each way port W back to back still takes a wide chunk nearly every
    cycle, and port B a chunk (the queue's 2 xstg + 2 entries: its room for 2 xstg + 1 more with
    one in it each cycle), at channels that take a beat a cycle."""
    plus = ["+axi_stall=0", "+seed=3"] + (["+wstream=6000"] if stream == "w" else
                                          ["+bstream=3000"])
    out = _run(8, 128, *plus, params={"XSTG": xstg})
    m = re.search(r"TBN [WB]STREAM reads=(\d+) cycles=(\d+)", out)
    reads, cycles = map(int, m.groups())
    assert reads / cycles >= 0.9, out[-500:]


@pytest.mark.parametrize("bpc,floor,cred", [(0, 0.9, 0), (99752, 0.9 * 99752 / 128000, 0),
                                            (99752, 0.9 * 99752 / 128000, 128)])
def test_port_w_streams_a_wide_chunk_a_cycle_rtl(have_verilator, bpc, floor, cred):
    """The WG targets' port W alone (NCH 8, CB 128): back to back it takes a 1 KiB wide chunk nearly
    every cycle when the channels take a beat a cycle (bpc 0: no limit), and the channels'
    share when they sustain 99.75 bytes a cycle (13.3 GB/s at 133.33 MHz: opentpu.target
    mem_model) with the probe's 268 ns of latency (36 cycles), also with the MXU's credits (at
    most `cred` reads not yet in every lane: its chunk FIFO's entries)."""
    plus = ["+wstream=6000", "+axi_stall=0", "+seed=3", f"+wcred={cred}"]
    if bpc:
        plus += [f"+nat_bpc={bpc}", "+axi_lat=36"]
    out = _run(8, 128, *plus)
    m = re.search(r"TBN WSTREAM reads=(\d+) cycles=(\d+)", out)
    reads, cycles = map(int, m.groups())
    assert reads / cycles >= floor, out[-500:]


# otpu_top's board memory path at the WG targets' channels (otpu_native_dram NCH 8, CB 128 in front of the
# native memory model at the target's channel rate and latency, opentpu.target.mem_model at
# 133.33 MHz): test_rtl's hazard and random programs, booted from DRAM, bit-exact against the ISA
# simulator; also without the SW queue's fill reads (NAT_FILL 0: QST's partial beats masked),
# with the write data posted at the take (NAT_WPOST 1: the model's write-data FIFO, 32 beats),
# with one SW gather beat and one posted beat a cycle for every channel (post 2: NAT_SWG 1), and
# through register stages between the slice and its adapter (NAT_XSTG, at the WG targets' NAT_FILL 0).
# Random 12 with the fill reads: an ST's last bytes and a QST's row in the two 64-byte halves of
# one 128-byte channel beat; the QST's scoreboard footprint takes the adapter's fill beat (otpu_pkg
# beats(), otpu_slice QB 7), or its fill read writes the beat back over the ST's bytes.
@pytest.mark.parametrize("prog,seed,stall,fill,post,xstg", [
    ("hazard", 0, 0, 1, 0, 0), ("hazard", 1, 50, 1, 0, 0), ("random", 2, 40, 1, 0, 0),
    ("random", 3, 70, 1, 0, 0), ("hazard", 4, 50, 0, 0, 0), ("random", 5, 40, 0, 0, 0),
    ("hazard", 6, 50, 0, 1, 0), ("random", 7, 40, 0, 1, 0), ("hazard", 8, 50, 0, 2, 0),
    ("random", 9, 40, 0, 2, 0), ("hazard", 10, 50, 0, 1, 2), ("random", 11, 40, 0, 1, 2),
    ("random", 12, 70, 0, 1, 3), ("hazard", 13, 50, 0, 2, 3), ("random", 14, 40, 0, 2, 3),
    ("random", 12, 70, 1, 0, 0)])
def test_wg_channels_on_the_top_rtl(have_verilator, prog, seed, stall, fill, post, xstg):
    import numpy as np
    from opentpu import Config
    from opentpu.isasim import Machine
    from opentpu.target import mem_model
    from test_rtl import _hazard_program, _images, _random_program
    rng = np.random.default_rng(6200 + seed)
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    p = (_hazard_program if prog == "hazard" else _random_program)(rng, cfg)
    imgs = _images(rng, 1)
    m = Machine(cfg, [p], [i.copy() for i in imgs]).run()
    drams, tmems, st = rtlsim.run(cfg, [p], [i.copy() for i in imgs], axi=True, boot=True,
                                  stall=stall, seed=seed + 1,
                                  uarch={"NCH": 8, "CB": 128} | ({} if fill else {"NAT_FILL": 0})
                                  | ({"NAT_WPOST": 1} if post else {})
                                  | ({"NAT_SWG": 1} if post == 2 else {})
                                  | ({"NAT_XSTG": xstg} if xstg else {}),
                                  plusargs=mem_model("bcu1525-x2", 133.33))
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert len(st["native"]) == 8 and sum(n["b_rd"] for n in st["native"]) > 0
    if not fill:                                    # (no fill read; QST's beats masked)
        assert sum(n["sw_rd"] for n in st["native"]) == 0


@pytest.mark.parametrize("seed,stall,bw", [(1, 0, 100), (2, 30, 15), (3, 60, 25)])
def test_b_fifo_in_block_ram_at_the_wg_rtl(have_verilator, seed, stall, bw):
    """RB_BLOCK (NAT_RBB's slices: the B FIFO's 32 beats in block RAM read ahead, as
    the cards' RD >= 128) at the WG targets' adapter (WPOST, SWG, XPOST, its A runs): the same random
    traffic, B's reads in order with their data, the memory the reference's."""
    from opentpu import target as T
    u = T.load("bcu1525-x2").uarch
    out = _run(8, 128, f"+seed={seed}", f"+axi_seed={seed}", f"+axi_stall={stall}",
               f"+axi_bw={bw}", "+epochs=12", "+ops=2000", "+busy=90",
               params={"SW_FILL": 0, "WPOST": 1, "XPOST": 1, "SWG": 1, "APF": u["NAT_APF"],
                       "AD": u["NAT_AD"], "RB_BLOCK": 1})
    cyc, brd, bwr, ard, awr, sw, wrd = map(int, DONE.search(out).groups())
    assert min(brd, bwr, ard, sw, wrd) > 1000 and awr == 0, out[-500:]


@pytest.mark.parametrize("rbb", [0, 1])
def test_port_b_rate_with_the_b_fifo_in_block_ram_rtl(have_verilator, rbb):
    """RB_BLOCK keeps port B's peak: back to back at the channel model's rate, nearly a chunk a
    cycle with its FIFO in block RAM as in LUT RAM."""
    plus = ["+bstream=3000", "+wstream=0", "+axi_stall=0", "+seed=3", "+nat_bpc=99752", "+axi_lat=36"]
    out = _run(8, 128, *plus, params={"RB_BLOCK": rbb})
    m = re.search(r"TBN BSTREAM reads=(\d+) cycles=(\d+) per_cycle=\S+ w_reads=(\d+)", out)
    reads, cycles, _ = map(int, m.groups())
    print(f"RBB {rbb}: {reads} reads in {cycles} cycles, {reads / cycles:.3f} a cycle")
    assert reads / cycles >= 0.95, out[-500:]


# an adapter of posted channels (NAT_*: posted write data, one gather beat, posted W
# beats, A's runs of 8 in a FIFO of 16, B's reads 16 a channel, port A's beat whole, the stages to
# the slice) and a fast channel model (76 bytes a cycle a channel, 47 cycles of latency)
PADA = {"SW_FILL": 0, "WPOST": 1, "XPOST": 1, "SWG": 1, "APF": 8, "AD": 16, "RD": 16, "AB": 128,
       "XSTG": 3, "XSTG_P": 4}
M76 = ["+nat_bpc=76000", "+axi_lat=47"]


@pytest.mark.parametrize("seed,stall,bw,rhb,whb,rbb,xstg", [
    (1, 0, 100, 1, 1, 0, 3), (2, 30, 15, 1, 1, 0, 3), (3, 60, 25, 1, 1, 0, 3), (4, 30, 15, 1, 1, 1, 3),
    (5, 0, 100, 1, 0, 0, 3), (6, 30, 25, 0, 1, 0, 3), (7, 60, 15, 1, 1, 1, 0), (8, 30, 25, 1, 1, 0, 0)])
def test_random_traffic_with_half_beats_rtl(have_verilator, seed, stall, bw, rhb, whb, rbb, xstg):
    """RHB / WHB (the WG targets' R1 / R2: the channels' read data and posted write data in half beats,
    the model's as the bridges') at the WG targets' adapter: the same random traffic on every port, every
    read in order with its data (B's chunk joined from its halves, A's beat whole from alo and its
    head, the run beats a read drops skipped), the memory the reference's (each posted beat's two
    halves); read or write halves alone, B's FIFO in LUT RAM and in block RAM (RB_BLOCK), with
    the slice's stages and without."""
    p = PADA | {"RHB": rhb, "WHB": whb, "RB_BLOCK": rbb, "XSTG": xstg, "XSTG_P": 4 if xstg else 0}
    out = _run(8, 128, f"+seed={seed}", f"+axi_seed={seed}", f"+axi_stall={stall}",
               f"+axi_bw={bw}", "+epochs=12", "+ops=2000", "+busy=90", params=p)
    cyc, brd, bwr, ard, awr, sw, wrd = map(int, DONE.search(out).groups())
    assert min(brd, bwr, ard, sw, wrd) > 1000 and awr == 0, out[-500:]
    beats, one = map(int, re.search(r"TBN XPOST beats=(\d+) one_beat_reads=(\d+)", out).groups())
    assert beats == 8 * (wrd - one) + one, out[-500:]


@pytest.mark.parametrize("mut", ["OTPU_MUT_WGI", "OTPU_MUT_ASKIP", "OTPU_MUT_BOV"])
def test_half_beat_mutants_are_caught_rtl(have_verilator, monkeypatch, mut):
    """The half beats' mutants fail tb_ndram without the adapter's own checks (OTPU_WO_UNCHECKED):
    an SW take in the cycle a high half goes onto the bus (OTPU_MUT_WGI: the beat goes in under
    it, the memory's halves out of step), a dropped run's skip a half short (OTPU_MUT_ASKIP:
    port A's beats from the wrong halves after it) and port B's next read taking its low half
    into b_lo on the same channel as the chunk going out (OTPU_MUT_BOV: the chunk's high half
    taken for the next read's low half)."""
    monkeypatch.setattr(rtlsim, "VFLAGS",
                        rtlsim.VFLAGS + [f"+define+{d}" for d in (mut, "OTPU_WO_UNCHECKED")])
    exe = rtlsim.build("tb_ndram", SRC, {"NCH": 8, "CB": 128} | PADA | {"RHB": 1, "WHB": 1})
    r = rtlsim.run_sim([str(exe), "+seed=3", "+axi_seed=3", "+axi_stall=30", "+axi_bw=25",
                        "+epochs=6", "+ops=1500", "+busy=90"], timeout=1200)
    out = r.stdout + r.stderr
    assert "TBN DONE" not in out and "tb_ndram" in out, out[-800:]


def _rate(out: str, port: str) -> float:
    return float(re.search(rf"TBN {port}STREAM reads=\d+ cycles=\d+ per_cycle=(\S+)", out).group(1))


@pytest.mark.parametrize("rhb,whb", [(0, 0), (1, 0), (1, 1)], ids=["whole", "rhb", "rhb_whb"])
def test_port_rates_with_half_beats_rtl(have_verilator, rhb, whb):
    """The posted adapter (PADA) at the fast channel model (M76), whole beats, RHB (the targets') and RHB /
    WHB: port A back to back (the scale stream, each read the next 128-byte beat: consecutive
    channels) beside a backlogged W stream, as under a grouped fp4 MM, keeps its rate with RHB (each
    channel's alo: a beat out every cycle from different channels; 0.873 a cycle against 0.880
    whole, W beside it 0.763 / 0.765; with WHB 0.873, W 0.764). Port B back to back: on consecutive
    chunks (hashed over the channels, bseq 1) nearly a chunk a cycle with RHB (b_lo takes the next
    read's low half as a chunk goes out, b_ov: 0.964 against 0.979, beside W 0.956 / 0.970); on one
    channel (bseq 2) the channel's rate as whole beats (two halves through one read port; 0.280 /
    0.286 at NAT_RD 16: the reads in flight a channel); at random chunks (bseq 0) the one-channel
    pairs' share (0.864 / 0.975; 0.859 with WHB)."""
    p = PADA | {"RHB": rhb, "WHB": whb}
    a = _run(8, 128, "+astream=3000", "+wstream=6000", "+axi_stall=0", "+seed=3", *M76, params=p)
    ar, wr = _rate(a, "A"), _rate(a, "W")
    br = {q: _rate(_run(8, 128, "+bstream=3000", "+axi_stall=0", "+seed=3", f"+bseq={q}", *M76,
                        params=p), "B") for q in (1, 2, 0)}
    bw = _run(8, 128, "+bstream=3000", "+wstream=6000", "+axi_stall=0", "+seed=3", "+bseq=1", *M76,
              params=p)
    print(f"RHB {rhb} WHB {whb}: A {ar:.3f} a cycle beside W {wr:.3f}; B consecutive {br[1]:.3f}, one "
          f"channel {br[2]:.3f}, random {br[0]:.3f}, consecutive beside W {_rate(bw, 'B'):.3f} "
          f"(W {_rate(bw, 'W'):.3f})")
    assert ar >= 0.86 and wr >= 0.74, a[-500:]
    assert br[1] >= (0.9 if rhb else 0.95) and _rate(bw, "B") >= 0.9, (br, bw[-500:])
    assert br[2] >= 0.27 and br[0] >= (0.85 if rhb else 0.95), br


@pytest.mark.parametrize("wstream", [0, 6000])
def test_port_b_goes_before_w_rtl(have_verilator, wstream):
    """Port B (NCH 8, CB 128) back to back at a fast channel model (99.75 bytes a cycle a channel,
    36 cycles) alone, and beside a backlogged W stream: B goes before W on every channel (the
    port plan's I1 as amended: the weight stream must not starve a DSTEP's state on B), so B
    keeps nearly a chunk a cycle (its peak: B's reads in flight, RD = 256 / NCH a channel, cover
    the latency), and W takes the channels' rest meanwhile (8 x 0.78 beats a cycle, less B's)."""
    plus = ["+bstream=3000", f"+wstream={wstream}", "+axi_stall=0", "+seed=3", "+nat_bpc=99752",
            "+axi_lat=36"]
    out = _run(8, 128, *plus)
    m = re.search(r"TBN BSTREAM reads=(\d+) cycles=(\d+) per_cycle=\S+ w_reads=(\d+)", out)
    reads, cycles, wr = map(int, m.groups())
    assert reads / cycles >= 0.95, out[-500:]
    if wstream:
        assert wr / cycles >= 0.5, out[-500:]
