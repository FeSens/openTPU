"""The BCU1525 WG build's DIMM pair (rtl/boards/bcu1525/otpu_mig_pair.sv: two channel bridges' AXI3
sides onto one DIMM's 512-bit AXI4 port): its testbench (sim/verilator/tb_migpair.sv) over random
traffic, the decode's read stream at the port's full rate, and mutations it must catch; at CL 32
(the 32 GB card: 4 GiB a channel) and 33 (64 GB) with the channels' top address bits in use. With
DM 1 the board model's DIMM (sim/verilator/otpu_dimm_model.sv) is the port: its sparse store above
the image, both its timings, and its images in and out (ch<c>.bin / ch<c>_hi.txt). With HM 1 the
DIMM's hmux (rtl/boards/bcu1525/otpu_mig_hmux.sv: the pair and the host's DMA onto the DIMM's one
port) sits between them, a host master beside the bridges: random host bursts of 1 .. 64 beats with
strobes, byte-exact, W in AW order across the two, and the stream still a beat a cycle. NC 4 (the
two-slice board, bcu1525-s2: four channels a DIMM, the host's ID at bit 0) the same at CL 31 and 32."""
import random
import subprocess

import numpy as np
import pytest

from opentpu import rtlsim

SRC = [rtlsim.RTL / "shell/otpu_mem_ch.sv", rtlsim.RTL / "shell/otpu_afifo.sv",
       rtlsim.TB / "otpu_dimm_model.sv"]
PAIR = rtlsim.RTL / "boards/bcu1525/otpu_mig_pair.sv"
TB = rtlsim.TB / "tb_migpair.sv"
HMUX = rtlsim.RTL / "boards/bcu1525/otpu_mig_hmux.sv"
HM_SRC = SRC + [rtlsim.RTL / "mem/otpu_axi_rs.sv"]   # (otpu_rs)


def _run(exe, args) -> str:
    return subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=600).stdout


@pytest.mark.parametrize("args", [
    [],
    ["+seed=2", "+wpct=70", "+ppct=60"],                 # mostly writes, many partial beats
    ["+seed=3", "+blocks=4", "+wpct=50"],                # few blocks: the hazard rule at work
    ["+seed=4", "+st=50", "+rbw=50", "+jit=200"],        # a slow, stalling port
    ["+seed=5", "+gapw=60", "+ot=2"],                    # W with gaps, little in flight
    ["+seed=6", "+blocks=2", "+wpct=50", "+ot=32", "+lat=5", "+jit=0"],
])
def test_tb_migpair(args):
    out = _run(rtlsim.build("tb_migpair", SRC + [PAIR, TB]), args)
    assert "PASS" in out and "FAIL" not in out, out[-2000:]


@pytest.mark.parametrize("cl", [29, 32, 33])
@pytest.mark.parametrize("args", [
    ["+hi"],
    ["+hi", "+seed=7", "+blocks=32", "+wpct=50", "+ppct=50"],   # 2 blocks a top pattern: hazards
])
def test_tb_migpair_high_bits(cl, args):
    """Blocks spread over the channel's top four address bits (CL 32: bits 28-31; 33: 29-32): a map
    that dropped or folded them would alias blocks (a read's bytes) or miss the DIMM address the
    testbench computes for each byte."""
    out = _run(rtlsim.build("tb_migpair", SRC + [PAIR, TB], {"CL": cl}), args)
    assert "PASS" in out and "FAIL" not in out, out[-2000:]


def test_migpair_stream():
    """The decode's stream (reads only, 16-beat bursts block after block on both channels): the
    DIMM's R channel busy every cycle at 150 cycles of latency (500 ns at 300 MHz)."""
    out = _run(rtlsim.build("tb_migpair", SRC + [PAIR, TB]),
               ["+seq", "+n=4000", "+blocks=4096", "+lat=150"])
    assert "PASS" in out, out[-2000:]
    assert float(out.split("rrate ")[1].split()[0]) >= 0.99, out[-500:]


@pytest.mark.parametrize("old,new", [
    ("return {a[CL-1:12], s, a[11:0]};", "return {a[CL-1:13], s, a[12:0]};"),    # the map
    ("assign rdata[LW-1:0] = hi ? d[2*LW-1:LW] : d[LW-1:0];",                    # R's halves
     "assign rdata[LW-1:0] = hi ? d[LW-1:0] : d[2*LW-1:LW];"),
    ("assign s_rdata[s][LW * l +: LW] = q[LW-1:0];",                             # R's lanes
     "assign s_rdata[s][LW * (RL - 1 - l) +: LW] = q[LW-1:0];"),
    ("assign s_rlast[s]  = rx[2] && hi[0];", "assign s_rlast[s]  = rx[2];"),      # RLAST's half
    ("assign m_wdata  = {w_h[255:0], hd};", "assign m_wdata  = {hd, w_h[255:0]};"),
    ("assign m_wstrb  = {w_h[287:256], hs};", "assign m_wstrb  = {64{1'b1}};"),    # strobes
    ("m_arlen  <= 8'(s_arlen[ar_pick] >> 1);", "m_arlen  <= 8'(s_arlen[ar_pick]);"),
])
def test_migpair_mutations(tmp_path, old, new):
    src = PAIR.read_text()
    assert src.count(old) == 1, old
    mut = tmp_path / "otpu_mig_pair.sv"
    mut.write_text(src.replace(old, new))
    out = _run(rtlsim.build("tb_migpair", SRC + [mut, TB]), ["+n=300"])
    assert "PASS" not in out, (new, out[-500:])


@pytest.mark.parametrize("old,new", [
    ("return {a[CL-1:12], s, a[11:0]};", "return {a[CL-2:12], s, a[11:0]};"),    # the top bit lost
    ("return {a[CL-1:12], s, a[11:0]};", "return {a[CL-1] ^ a[CL-2], a[CL-2:12], s, a[11:0]};"),
])
def test_migpair_high_bit_mutations(tmp_path, old, new):
    """At CL 32 the map must carry the channel's bits 28-31 (+hi puts blocks there)."""
    src = PAIR.read_text()
    assert src.count(old) == 1, old
    mut = tmp_path / "otpu_mig_pair.sv"
    mut.write_text(src.replace(old, new))
    out = _run(rtlsim.build("tb_migpair", SRC + [mut, TB], {"CL": 32}), ["+n=300", "+hi"])
    assert "PASS" not in out, (new, out[-500:])


DMB = 64                                    # the model's dense beats a channel (2 KiB)


# ---- the bank-half window (otpu_mig_pair BHW): a channel's first 2^BHW bytes swap the MIG map's
# bank-high bit (BHB) with the window's top bit

def _slice_map(x: int, cl: int, nc: int, bhw: int, bhb: int = 16) -> tuple[int, int]:
    """A slice's byte x: (its DIMM, the DIMM's byte address) as the bridges and the pairs give it
    (8 channels of 2^cl bytes, 128 bytes a channel in each 1 KiB, nc channels a DIMM)."""
    c, o = (x >> 7) & 7, ((x >> 10) << 7) | (x & 127)
    lnc = nc.bit_length() - 1
    d = (o >> 12 << (12 + lnc)) | ((c % nc) << 12) | (o & 0xFFF)
    if bhw and o >> bhw == 0:
        hi, lo = (d >> bhb) & 1, (d >> (bhw - 1 + lnc)) & 1
        d ^= (hi ^ lo) * ((1 << bhb) | (1 << (bhw - 1 + lnc)))
    return c // nc, d


@pytest.mark.parametrize("cl,nc,bhw", [
    (32, 2, 30),        # the 32 GB card (8 GB DIMMs, two channels each): the slice's first 8 GiB
    (31, 2, 29),        # a 16 GB card (4 GB DIMMs): its first 4 GiB
    (31, 4, 30),        # a 16 GB slice of bcu1525-s2 (8 GB DIMMs, four channels each): 8 GiB
])
def test_bhw_map_is_a_bijection(cl, nc, bhw):
    """The window's map is a bijection of the slice's bytes onto its DIMMs' (it is its own inverse
    on the DIMM's address, and the select bits are not swapped): over every pattern of the bits it
    reads or moves (the select bits all clear or one of them set, the two swapped bits, the
    channel, the channel's low bytes) with the others random, and over all of a scaled-down
    slice's bytes, where every DIMM address is hit once."""
    rng = random.Random(cl * 10 + nc)
    lnc = nc.bit_length() - 1
    top = [None] + list(range(bhw, cl))         # (the select bits: all clear, or bit t of o set)
    seen = {}
    for _ in range(4000):
        o = rng.getrandbits(cl)
        t = rng.choice(top)
        o &= (1 << bhw) - 1
        if t is not None:
            o |= 1 << t
        for b in (16 - lnc, bhw - 1):           # (the swapped bits, in the channel's byte)
            o = (o & ~(1 << b)) | (rng.getrandbits(1) << b)
        c = rng.randrange(8)
        x = ((o >> 7) << 10) | (c << 7) | (o & 127)
        dm, d = _slice_map(x, cl, nc, bhw)
        _, d0 = _slice_map(x, cl, nc, 0)
        assert d < 1 << (cl + lnc) and dm == c // nc
        if (o >> bhw) == 0:                     # (inside: the bank-high bit the window's top)
            assert (d >> 16) & 1 == (o >> (bhw - 1)) & 1
            assert d ^ d0 in (0, (1 << 16) | (1 << (bhw - 1 + lnc)))
        else:
            assert d == d0
        assert seen.setdefault((dm, d), x) == x
    # scaled down: 2^(cl' + 3) bytes, the window and the bank-high bit at their places' ratios
    small_cl, small_bhw, small_bhb = 16, 14, 12 + lnc
    hits = set()
    for x in range(1 << (small_cl + 3)):
        hits.add(_slice_map(x, small_cl, nc, small_bhw, small_bhb))
    assert len(hits) == 1 << (small_cl + 3)


BHW = {"CL": 32, "BHW": 30}                 # the 32 GB card's window: a channel's first 1 GiB


@pytest.mark.parametrize("dm", [0, 1])
@pytest.mark.parametrize("args", [
    ["+hi"],
    ["+hi", "+seed=11", "+blocks=32", "+wpct=50", "+ppct=50"],
])
def test_tb_migpair_bhw(dm, args):
    """The window at CL 32 (+hi: blocks over the channel's bits 28-31, those with 31:30 clear in
    it, half of them with its top bit 29 set): the pair's map is the swap (DM 0: every byte at the
    DIMM address the testbench computes), the board model's DIMM takes its inverse (DM 1: the
    channels' bytes in its images where the window's absence would put them)."""
    p = {**BHW, **({"DM": 1, "DMB": DMB} if dm else {})}
    out = _run(rtlsim.build("tb_migpair", SRC + [PAIR, TB], p), args + ["+dimm_mc=1"] * dm)
    assert "PASS" in out and "FAIL" not in out and "ERROR" not in out, out[-2000:]


def test_migpair_bhw_stream():
    """The decode's stream inside the window: still a beat a cycle."""
    out = _run(rtlsim.build("tb_migpair", SRC + [PAIR, TB], BHW),
               ["+seq", "+n=4000", "+blocks=4096", "+lat=150"])
    assert "PASS" in out, out[-2000:]
    assert float(out.split("rrate ")[1].split()[0]) >= 0.99, out[-500:]


@pytest.mark.parametrize("dm,path,old,new", [
    (0, PAIR, "e[HB] = d[WB-1+LNC];", ""),                     # the pair: the bank bit kept
    (0, PAIR, "e[WB-1+LNC] = d[HB];", ""),                     # the pair: a bit lost (not a bijection)
    (0, PAIR, "d[CL+LNC-1:WB+LNC] == '0", "d[CL+LNC-1:WB+LNC+1] == '0"),   # the window twice as large
    (1, rtlsim.TB / "otpu_dimm_model.sv", "d = unbhw(a);", "d = a;"),   # the model: no inverse
])
def test_migpair_bhw_mutations(tmp_path, dm, path, old, new):
    src = path.read_text()
    assert src.count(old) == 1, old
    mut = tmp_path / path.name
    mut.write_text(src.replace(old, new))
    srcs = [mut if f == path else f for f in SRC + [PAIR, TB]]
    p = {**BHW, **({"DM": 1, "DMB": DMB} if dm else {})}
    out = _run(rtlsim.build("tb_migpair", srcs, p), ["+n=300", "+hi"])
    assert "PASS" not in out, (new, out[-500:])


def _dm(cl: int):
    return rtlsim.build("tb_migpair", SRC + [PAIR, TB], {"CL": cl, "DM": 1, "DMB": DMB})


@pytest.mark.parametrize("cl", [29, 32, 33])
@pytest.mark.parametrize("args", [
    ["+hi"],
    ["+hi", "+dimm_mc=1"],                                   # the MIG's timing
    ["+hi", "+seed=9", "+blocks=32", "+wpct=60", "+ppct=60", "+dimm_mc=1", "+mc_strict=1"],
    ["+hi", "+seed=10", "+dimm_jit=40", "+dimm_st=20", "+dimm_ooo=50"],
])
def test_dimm_model_sparse(cl, args):
    """The board model's DIMM behind the pair, blocks over the channels' top address bits: all but
    the low blocks above its dense beats, every read checked and every byte at the end."""
    out = _run(_dm(cl), args)
    assert "PASS" in out and "FAIL" not in out and "ERROR" not in out, out[-2000:]
    assert int(out.split("model: ")[1].split()[0]) > 0, out[-500:]


def _beats(path) -> dict[int, int]:
    return {int(b, 16): int(d, 16) for b, d in (ln.split() for ln in path.read_text().splitlines())}


def test_dimm_model_images(tmp_path):
    """ch<c>.bin and ch<c>_hi.txt in, ch<c>_out.bin and ch<c>_hi_out.txt out (no traffic): the same
    beats, and a beat's bytes in the same order in both (byte j at bits 8 j .. 8 j + 7)."""
    rng = random.Random(1)
    v = rng.getrandbits(256)
    top = (1 << (32 - 5)) - 1                         # CL 32: the channel's last 32-byte beat
    hi = [{DMB: v, 12345: rng.getrandbits(256), top: rng.getrandbits(256)},
          {DMB + 1: rng.getrandbits(256), 1 << 20: 0}]
    lo = [bytearray(rng.randbytes(32 * DMB)), bytearray(rng.randbytes(32 * DMB))]
    lo[0][32 * (DMB - 1):] = v.to_bytes(32, "little")  # the last dense beat = the first sparse
    for c in range(2):
        # ch<c>.bin: 32-bit big-endian words of the beats' little-endian words ($fread)
        (tmp_path / f"ch{c}.bin").write_bytes(
            np.frombuffer(bytes(lo[c]), "<u4").astype(">u4").tobytes())
        (tmp_path / f"ch{c}_hi.txt").write_text("".join(f"{b:x} {d:064x}\n" for b, d in hi[c].items()))
    out = _run(_dm(32), ["+n=0", f"+dir={tmp_path}"])
    assert "PASS" in out and "ERROR" not in out, out[-2000:]
    for c in range(2):
        assert (tmp_path / f"ch{c}_out.bin").read_bytes() == bytes(lo[c]), c
        assert _beats(tmp_path / f"ch{c}_hi_out.txt") == hi[c], c


@pytest.mark.parametrize("beat", [DMB - 1, 1 << 27])
def test_dimm_model_hi_range(tmp_path, beat):
    """A ch<c>_hi.txt beat in the dense image or past the channel (CL 32: 2^27 beats) is an error."""
    (tmp_path / "ch0_hi.txt").write_text(f"{beat:x} {1:064x}\n")
    out = _run(_dm(32), ["+n=0", f"+dir={tmp_path}"])
    assert "ERROR dimm 0: ch0_hi.txt" in out and "PASS" not in out, out[-1000:]


@pytest.mark.parametrize("cl, dmb", [(32, 0), (13, 1 << 9)])
def test_dimm_model_dense_range(cl, dmb):
    """The dense image (BEATS) is 1 .. a channel's beats (CL 13: 2^8), else a fatal at time 0 that
    names it, not Verilator's "Size of range is [0]": bcu1525-s2's expert run at BHW 33 asked
    for its 16 GiB a slice dense, and tb_bcuwg's WORDS (an int) wrapped to 0."""
    exe = rtlsim.build("tb_migpair", SRC + [PAIR, TB], {"CL": cl, "DM": 1, "DMB": dmb})
    r = subprocess.run([str(exe), "+n=0"], capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    assert f"otpu_dimm_model: BEATS {dmb}:" in out and "PASS" not in out, out[-1000:]
    assert r.returncode != 0


def _hm(params: dict | None = None, hmux=HMUX):
    return rtlsim.build("tb_migpair", HM_SRC + [hmux, PAIR, TB], {"HM": 1, **(params or {})})


@pytest.mark.parametrize("args", [
    [],
    ["+seed=2", "+wpct=70", "+ppct=60", "+hwpct=80", "+hppct=70"],   # mostly writes: W's order
    ["+seed=3", "+hmaxb=1", "+hot=16"],                              # single-beat host bursts
    ["+seed=4", "+st=50", "+rbw=50", "+jit=200", "+hst=50"],         # a slow port, the host slow
    ["+seed=5", "+gapw=60", "+hgapw=60", "+ot=2", "+hot=2"],         # W with gaps on both sides
    ["+seed=6", "+blocks=2", "+hblk=1", "+ot=32", "+lat=5", "+jit=0"],   # hazards on both sides
    ["+seed=7", "+n=0", "+hn=3000"],                                 # the host alone
])
def test_hmux(args):
    """The host's random reads and writes (1 .. 64 beats, strobes) in its region of the DIMM beside
    both bridges' random traffic: every read checked as it returns, every byte at the end."""
    out = _run(_hm(), args)
    assert "PASS" in out and "FAIL" not in out, out[-2000:]
    assert "host: reads" in out, out[-500:]


@pytest.mark.parametrize("cl", [29, 32])
def test_hmux_high_bits(cl):
    out = _run(_hm({"CL": cl}), ["+hi", "+seed=8", "+blocks=32", "+wpct=50"])
    assert "PASS" in out and "FAIL" not in out, out[-2000:]


@pytest.mark.parametrize("args", [
    ["+hi", "+hmaxb=16", "+dimm_mc=1"],
    ["+hi", "+hmaxb=16", "+seed=11", "+dimm_jit=40", "+dimm_st=20", "+dimm_ooo=50"],
])
def test_hmux_dimm_model(args):
    """The board model's DIMM behind the hmux (its bursts of 16 beats at most): the MIG's timing,
    and reads out of order across IDs (the host's and the bridges')."""
    out = _run(_hm({"DM": 1, "DMB": DMB}), args)
    assert "PASS" in out and "FAIL" not in out and "ERROR" not in out, out[-2000:]


def test_hmux_stream():
    """The decode's stream through the hmux, the host idle: the DIMM's R channel still busy every
    cycle at 150 cycles of latency (its register slices take a beat a cycle)."""
    out = _run(_hm(), ["+seq", "+n=4000", "+blocks=4096", "+lat=150", "+hn=0"])
    assert "PASS" in out, out[-2000:]
    assert float(out.split("rrate ")[1].split()[0]) >= 0.99, out[-500:]


@pytest.mark.parametrize("old,new", [
    ("m_awaddr <= aw_h ? h_awaddr : p_awaddr;", "m_awaddr <= p_awaddr;"),         # AW's payload
    (".rready(w_v && w_r && w_l)", ".rready(w_v && w_r)"),                         # W's order
    ("s_data(wo_h ? {h_wlast, h_wstrb, h_wdata}", "s_data(wo_h ? {h_wlast, {64{1'b1}}, h_wdata}"),
    ("assign r_r      = r_id[HB] ? h_rready : p_rready;", "assign r_r      = p_rready;"),
    ("assign p_rvalid = r_v && !r_id[HB];", "assign p_rvalid = r_v && !r_id[IDW-1];"),
    ("assign m_bready = b_h ? h_bready : p_bready;", "assign m_bready = p_bready;"),
    ("HID = IDW'(1) << HB;", "HID = IDW'(1) << (IDW - 1);"),                     # a bridge's ID
])
def test_hmux_mutations(tmp_path, old, new):
    src = HMUX.read_text()
    assert src.count(old) == 1, old
    mut = tmp_path / "otpu_mig_hmux.sv"
    mut.write_text(src.replace(old, new))
    out = _run(_hm(hmux=mut), ["+n=300", "+hn=300", "+maxc=200000"])
    assert "PASS" not in out, (new, out[-500:])


# ---- NC 4: four channels a DIMM (bcu1525-s2; CL 31 there: 2 GiB a channel)
NC4 = {"NC": 4}


@pytest.mark.parametrize("args", [
    [],
    ["+seed=2", "+wpct=70", "+ppct=60"],
    ["+seed=3", "+blocks=4", "+wpct=50"],
    ["+seed=4", "+st=50", "+rbw=50", "+jit=200"],
    ["+seed=6", "+blocks=2", "+wpct=50", "+ot=32", "+lat=5", "+jit=0"],
])
def test_tb_migpair_nc4(args):
    out = _run(rtlsim.build("tb_migpair", SRC + [PAIR, TB], NC4), args)
    assert "PASS" in out and "FAIL" not in out, out[-2000:]


@pytest.mark.parametrize("cl", [31, 32])
def test_tb_migpair_nc4_high_bits(cl):
    out = _run(rtlsim.build("tb_migpair", SRC + [PAIR, TB], {**NC4, "CL": cl}),
               ["+hi", "+seed=7", "+blocks=32", "+wpct=50", "+ppct=50"])
    assert "PASS" in out and "FAIL" not in out, out[-2000:]


def test_migpair_nc4_stream():
    """Four channels' decode streams: the DIMM's R channel still busy every cycle."""
    out = _run(rtlsim.build("tb_migpair", SRC + [PAIR, TB], NC4),
               ["+seq", "+n=2000", "+blocks=4096", "+lat=150"])
    assert "PASS" in out, out[-2000:]
    assert float(out.split("rrate ")[1].split()[0]) >= 0.99, out[-500:]


@pytest.mark.parametrize("old,new", [
    ("return {a[CL-1:12], s, a[11:0]};", "return {a[CL-1:12], s[0], s[LNC-1], a[11:0]};"),  # the map
    ("assign r_s = m_rid[IDW-1 -: LNC];", "assign r_s = LNC'(m_rid[IDW-1]);"),      # R's bridge
    ("assign b_s = m_bid[IDW-1 -: LNC];", "assign b_s = LNC'(m_bid[IDW-1]);"),      # B's bridge
    ("if (req[k]) rr = k;", "if (req[k] && k < 2) rr = k;"),                         # two starved
])
def test_migpair_nc4_mutations(tmp_path, old, new):
    src = PAIR.read_text()
    assert src.count(old) == 1, old
    mut = tmp_path / "otpu_mig_pair.sv"
    mut.write_text(src.replace(old, new))
    out = _run(rtlsim.build("tb_migpair", SRC + [mut, TB], NC4), ["+n=300", "+maxc=200000"])
    assert "PASS" not in out, (new, out[-500:])


@pytest.mark.parametrize("args", [
    [],
    ["+seed=6", "+blocks=2", "+ot=32", "+lat=5", "+jit=0", "+rbw=100"],
    ["+seq", "+n=1500", "+blocks=4096", "+lat=150"],
])
@pytest.mark.parametrize("nc,rd", [(2, 32), (4, 32), (4, 4)])
def test_migpair_rs(args, nc, rd):
    """RS (the top's PRS at S 2): R through a register a bridge, its FIFO's fill counted from the
    accept. Every byte back (the tb's traffic follows the timing, so not RS 0's run beat for beat;
    RD 4 fills the FIFOs, so RREADY's early drop is taken), never a beat into a full FIFO (the
    pair's $error), and the sequential stream at RD 32 still takes R at >= 0.99 a cycle (as RS 0,
    test_migpair_stream)."""
    p = {"NC": nc, "RD": rd}
    out = _run(rtlsim.build("tb_migpair", SRC + [PAIR, TB], {**p, "RS": 1}), args)
    assert "PASS" in out and "FAIL" not in out and "%Error" not in out, out[-2000:]
    assert "R stage into a full FIFO" not in out, out[-2000:]
    if "+seq" in args and rd == 32:
        assert float(out.split("rrate ")[1].split()[0]) >= 0.99, out[-500:]


@pytest.mark.parametrize("args", [[], ["+seed=6", "+blocks=2", "+hblk=1", "+ot=32", "+lat=5", "+jit=0"]])
def test_hmux_nc4_rs(args):
    """The hmux beside the pair at NC 4 with all of the top's S 2 copies and the R stage (RCP 8,
    WCP 8, WOC 4, RS): every byte of both masters back, the host's reads too."""
    out = _run(_hm({**NC4, "CL": 31, "RCP": 8, "WCP": 8, "WOC": 4, "RS": 1}), args)
    assert "PASS" in out and "FAIL" not in out and "%Error" not in out, out[-2000:]
    assert "host: reads" in out, out[-500:]


@pytest.mark.parametrize("old,new", [
    ("assign rf_wr[s] = !nf;", "assign rf_wr[s] = wr[0];"),           # RREADY not a beat early
    ("nf <= int'(n) + int'(acc) - int'(pop) == RD;",                   # the credits one too many
     "nf <= int'(n) + int'(acc) - int'(pop) == RD + 1;"),
    ("assign pop = rv[0] && hi[0] && s_rready[s];",                    # a credit back per half
     "assign pop = rv[0] && s_rready[s];"),
])
def test_migpair_rs_mutations(tmp_path, old, new):
    """RS at NC 4, RD 4: the stage's credits are what keep a beat out of a full FIFO; RREADY the
    FIFO's not-full (as RS 0), a credit too many or one back per half each lose a beat or stall."""
    src = PAIR.read_text()
    assert src.count(old) == 1, old
    mut = tmp_path / "otpu_mig_pair.sv"
    mut.write_text(src.replace(old, new))
    out = _run(rtlsim.build("tb_migpair", SRC + [mut, TB], {"NC": 4, "RD": 4, "RS": 1}),
               ["+n=300", "+maxc=200000", "+rbw=100", "+lat=5", "+jit=0"])
    assert "PASS" not in out or "R stage into a full FIFO" in out, (new, out[-500:])


@pytest.mark.parametrize("old,new", [
    ("assign w_h[g * WG +: WG] = c[wo_c[g]];",                          # a copy's bits: wrong bridge
     "assign w_h[g * WG +: WG] = c[wo_c[g] ^ LNC'(g == 2)];"),
    (".wdata({WOC{aw_pick}})", ".wdata({{WOC-1{aw_last}}, aw_pick})"),  # copies 1-3 not the order
])
def test_migpair_woc_mutations(tmp_path, old, new):
    """WOC 4 at NC 4: each copy of u_wo's head picks its own bits of the W beat (the first run
    passes; a copy picking another bridge's, or holding another value, fails it)."""
    src = PAIR.read_text()
    assert src.count(old) == 1, old
    p = {**NC4, "WOC": 4}
    out = _run(rtlsim.build("tb_migpair", SRC + [PAIR, TB], p), ["+n=300", "+maxc=200000"])
    assert "PASS" in out and "FAIL" not in out, out[-2000:]
    mut = tmp_path / "otpu_mig_pair.sv"
    mut.write_text(src.replace(old, new))
    out = _run(rtlsim.build("tb_migpair", SRC + [mut, TB], p), ["+n=300", "+maxc=200000"])
    assert "PASS" not in out, (new, out[-500:])


@pytest.mark.parametrize("args", [
    ["+hi"],
    ["+hi", "+seed=9", "+blocks=32", "+wpct=60", "+ppct=60", "+dimm_mc=1", "+mc_strict=1"],
])
def test_dimm_model_nc4(args):
    """The board model's DIMM at four channels (CL 31: bcu1525-s2's) behind the pair."""
    out = _run(rtlsim.build("tb_migpair", SRC + [PAIR, TB], {**NC4, "CL": 31, "DM": 1, "DMB": DMB}),
               args)
    assert "PASS" in out and "FAIL" not in out and "ERROR" not in out, out[-2000:]
    assert int(out.split("model: ")[1].split()[0]) > 0, out[-500:]


def test_dimm_model_actlog(tmp_path):
    """+mc_actlog: a line for every ACT the MIG model counts (NC 4 with the window: the channel
    bytes before its swap; the model may open a row or two after its report), its cause one of
    C / P / R, a precharge's row a new one, and the timing the same as without it."""
    p = {**NC4, "CL": 31, "BHW": 30, "DM": 1, "DMB": DMB}
    exe = rtlsim.build("tb_migpair", SRC + [PAIR, TB], p)
    args = ["+hi", "+seed=9", "+blocks=32", "+wpct=60", "+dimm_mc=1", "+mc_strict=1"]
    out = _run(exe, args + [f"+mc_actlog={tmp_path}/a-"])
    assert "PASS" in out and "FAIL" not in out and "ERROR" not in out, out[-2000:]
    n = int(out.split("model: ")[1].split()[0])
    lines = (tmp_path / "a-d0.txt").read_text().splitlines()
    assert lines[0].startswith("# dimm 0 NC 4 DAW 33 BHW 30") and n > 0, lines[:2]
    assert n <= len(lines) - 1 <= n + 2, (n, len(lines))    # (the model runs on past its report)
    for ln in lines[1:]:
        cyc, g, b, row, old, why, we, ch, off, owe, och, ooff = ln.split()
        assert why in "CPR" and int(we) in (0, 1) and 0 <= int(ch) < 4 and 0 <= int(off, 16) < 1 << 31
        if why == "P":
            assert row != old
    base = _run(exe, args)
    assert base.split("model: ")[1].split("\n")[0] == out.split("model: ")[1].split("\n")[0]


@pytest.mark.parametrize("args", [
    [],
    ["+seed=6", "+blocks=2", "+hblk=1", "+ot=32", "+lat=5", "+jit=0"],
])
@pytest.mark.parametrize("rcp", [0, 8])
def test_hmux_nc4(args, rcp):
    """The hmux at HB 0 (the host's ID {0, 0, 0, 1}: the pair's four channels take the top two
    bits) beside four bridges; RCP 8 and WCP 8 (the top's at S 2: the W and R rings' read selects
    and their entries' enables in copies, otpu_rs RCP / WCP) with the pair's W order's head in 4
    (WOC 4, the top's at S 2) the same run, beat for beat."""
    out = _run(_hm({**NC4, "CL": 31, **({"RCP": rcp, "WCP": rcp, "WOC": 4} if rcp else {})}),
               args)
    assert "PASS" in out and "FAIL" not in out, out[-2000:]
    assert "host: reads" in out, out[-500:]
    if rcp:
        base = _run(_hm({**NC4, "CL": 31}), args)
        assert [ln for ln in out.splitlines() if not ln.startswith("- ")] == \
            [ln for ln in base.splitlines() if not ln.startswith("- ")]


@pytest.mark.parametrize("dm", [0, 1])
@pytest.mark.parametrize("args", [
    ["+hi"],
    ["+hi", "+seed=12", "+blocks=32", "+wpct=50", "+ppct=50"],
])
def test_tb_migpair_nc4_bhw(dm, args):
    """bcu1525-s2's window (CL 31, BHW 30: a channel's first 1 GiB, the slice's first 8 GiB; +hi:
    blocks over the channel's bits 27-30, those with bit 30 clear in it, half of them with its top
    bit 29 set): at four channels the window's top bit sits at the DIMM's bit 31 and the swap
    still holds (DM 0: the testbench's map; DM 1: the board model's DIMM undoes it)."""
    p = {**NC4, "CL": 31, "BHW": 30, **({"DM": 1, "DMB": DMB} if dm else {})}
    out = _run(rtlsim.build("tb_migpair", SRC + [PAIR, TB], p), args + ["+dimm_mc=1"] * dm)
    assert "PASS" in out and "FAIL" not in out and "ERROR" not in out, out[-2000:]


def test_migpair_nc4_bhw_mutation(tmp_path):
    """At four channels the window's top bit is the DIMM's WB - 1 + LNC: a map that puts it at
    NC 2's place (WB) fails."""
    src = PAIR.read_text()
    old = "e[WB-1+LNC] = d[HB];"
    assert src.count(old) == 1
    mut = tmp_path / "otpu_mig_pair.sv"
    mut.write_text(src.replace(old, "e[WB] = d[HB];").replace("e[HB] = d[WB-1+LNC];",
                                                              "e[HB] = d[WB];"))
    out = _run(rtlsim.build("tb_migpair", SRC + [mut, TB], {**NC4, "CL": 31, "BHW": 30}),
               ["+n=300", "+hi"])
    assert "PASS" not in out, out[-500:]
