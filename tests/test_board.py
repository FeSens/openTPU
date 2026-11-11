"""The host driver (opentpu/host/board.py) and the board model (sim/verilator/tb_board.sv).

The address-map tests are pure Python. The others drive the Verilator model of the board --
control registers, program loader, slice, DRAM adapter, two-channel AXI memory -- through the
same register and memory protocol the PCIe driver uses, and require the ISA simulator's
results bit for bit.
"""
import re

import numpy as np
import pytest

from opentpu import rtlsim
from opentpu.host import regs as R
from opentpu.host.board import BASE, BEAT, Board, BoardBackend, SimTransport, join, split
from opentpu.host.checks import (address_lines, channel_patterns, masked_program, partial_writes,
                                 pattern_test, run_demo, vops_program)
from opentpu.host.opchecks import diag_image, op_checks
from conftest import sim_board_config
from opentpu.isasim import board_config


# ------------------------------------------------------------------------------ address map
class MemTransport:
    """Two channel memories in RAM and the identity registers (CAPS: CHASH or not): exercises
    Board.read/write only."""

    def __init__(self, ch_bytes=1 << 16, chash=False):
        self.ch = [np.zeros(ch_bytes, np.uint8) for _ in range(2)]
        self.regs = {R.R_VERSION: 128 << 16 | 2 << 8 | 8, R.R_REGMAP: 3,
                     R.R_CAPS: R.CAP_CHASH if chash else 0}

    def reg_read_many(self, offs):
        return [self.regs.get(o, 0) for o in offs]

    def mem_write(self, c, off, data):
        self.ch[c][off:off + len(data)] = data

    def mem_read(self, c, off, n):
        return self.ch[c][off:off + n].copy()


def test_split_matches_the_rtl_interleave():
    """Logical beat b lives on channel b % 2 at offset (b // 2) * 64 (otpu_native_dram.sv)."""
    data = np.arange(8 * BEAT, dtype=np.uint32).astype(np.uint8)
    parts = split(256, data)
    for c, off, part in parts:
        assert off == 128
        for k in range(len(part) // BEAT):
            b = 256 // BEAT + 2 * k + c                   # logical beat
            assert np.array_equal(part[k * BEAT:(k + 1) * BEAT],
                                  data[(b - 4) * BEAT:(b - 3) * BEAT])
    assert np.array_equal(join([p for _, _, p in parts]), data)
    assert BASE == (0, 0x8000_0000)


def test_split_hashed_interleave():
    """CHASH: logical beat b of chunk m = b // 2 is on channel (b % 2) ^ parity(m), at channel
    offset m * 64 (otpu_native_dram.sv), so a column at a power-of-two chunk stride is spread
    over both channels."""
    data = np.arange(64 * BEAT, dtype=np.uint32).astype(np.uint8)
    parts = split(512, data, chash=True)
    for c, off, part in parts:
        assert off == 256
        for k in range(len(part) // BEAT):
            m = 512 // (2 * BEAT) + k
            b = 2 * m + (c ^ (bin(m).count("1") & 1))
            assert np.array_equal(part[k * BEAT:(k + 1) * BEAT],
                                  data[(b - 8) * BEAT:(b - 7) * BEAT])
    assert np.array_equal(join([p for _, _, p in parts], 512, chash=True), data)
    # beat 0 of every 4th chunk (a 512-byte stride) from an aligned start: rows 2j and 2j + 1
    # on different channels, so the column is spread evenly
    chans = [bin(m).count("1") & 1 for m in range(0, 256, 4)]
    assert all(x != y for x, y in zip(chans[0::2], chans[1::2]))


@pytest.mark.parametrize("chash", [False, True])
@pytest.mark.parametrize("addr,n", [(0, 128), (4, 8), (60, 8), (100, 300), (127, 1), (1000, 2000)])
def test_unaligned_read_modify_write(addr, n, chash):
    rng = np.random.default_rng(addr + n)
    t = MemTransport(chash=chash)
    b = Board(t, check=False)
    ref = rng.integers(0, 256, 1 << 14).astype(np.uint8)
    b.write(0, ref)
    new = rng.integers(0, 256, n).astype(np.uint8)
    b.write(addr, new)
    ref[addr:addr + n] = new
    assert np.array_equal(b.read(0, len(ref)), ref)
    assert np.array_equal(b.read(addr, n), new)
    # the channels hold the interleave
    for beat in range(len(ref) // BEAT):
        c, off = beat % 2 ^ (chash and bin(beat // 2).count("1") & 1), (beat // 2) * BEAT
        assert np.array_equal(t.ch[c][off:off + BEAT], ref[beat * BEAT:(beat + 1) * BEAT])


@pytest.mark.parametrize("chash", [False, True])
@pytest.mark.parametrize("addr,n", [(0, 1 << 16), (4096 + 64, 20000), (300, 9000)])
def test_writes_during_a_run_move_run_h2c_per_call(addr, n, chash):
    """While a run is in flight (Board.start .. wait) a transport with run_h2c (XdmaTransport:
    XDMA's H2C engine laps its read buffer on a longer call that meets the card's traffic,
    docs/host.md) gets Board.write's bytes in calls of at most run_h2c per channel, the same
    bytes in the same places; outside a run, one call per channel."""
    class T(MemTransport):
        run_h2c = 4096

        def __init__(self):
            super().__init__(ch_bytes=1 << 17, chash=chash)
            self.calls = []

        def mem_write(self, c, off, data):
            self.calls.append(len(data))
            super().mem_write(c, off, data)

        def reg_write(self, off, val):
            pass

        def reg_read(self, off):                # (start() checks that the card took RUN)
            return R.ST_RUN | R.ST_CALIB0 | R.ST_CALIB1 if off == R.R_STATUS else 0

    rng = np.random.default_rng(n)
    t = T()
    b = Board(t, check=False)
    ref = rng.integers(0, 256, 1 << 17).astype(np.uint8)
    b.write(0, ref)
    assert t.calls == [1 << 16, 1 << 16]
    b.start()
    assert b.in_run
    t.calls = []
    new = rng.integers(0, 256, n).astype(np.uint8)
    b.write(addr, new)
    ref[addr:addr + n] = new
    assert max(t.calls) <= 4096 and sum(t.calls) >= n
    b.in_run = False                        # (wait() clears it once HALTED is seen)
    assert np.array_equal(b.read(0, len(ref)), ref)


def test_channel_map_from_the_cfg_registers():
    """Register map 4: channel j = 2s + c (slice s's channel c) at j x the bytes a slice
    addresses per channel; the YPCB-00338's two are BASE, as before map 4."""
    from opentpu.host.board import channel_map
    ypcb = {"cfg": {"S": 1, "channels": 2, "channel_bytes": 1 << 31}}
    assert channel_map(ypcb) == BASE == channel_map({"cfg": None}) == channel_map(None)
    bcu = {"cfg": {"S": 2, "channels": 2, "channel_bytes": 1 << 31}}
    assert channel_map(bcu) == (0, 1 << 31, 2 << 31, 3 << 31)


def _ref_channel(addr: int, nch: int, cb: int, chash: bool) -> tuple[int, int]:
    """The port plan's I1, bit by bit: logical beat b = addr / CB, wide chunk m = b / NCH,
    channel (b mod NCH) XOR h(m), h the XOR fold of m's bits into log2(NCH) bits (bit i of m
    into bit i mod log2 NCH), channel offset m x CB + addr mod CB."""
    b, cl = addr // cb, (nch - 1).bit_length()
    m = b // nch
    h = 0
    for i in range(m.bit_length() if chash else 0):
        h ^= (m >> i & 1) << (i % cl)
    return b % nch ^ h, m * cb + addr % cb


@pytest.mark.parametrize("chash", [False, True])
@pytest.mark.parametrize("nch,cb", [(2, 64), (8, 128), (4, 64)])
def test_split_and_beat_at_follow_the_channel_map(nch, cb, chash):
    """split / join / beat_at at NCH channels of CB-byte beats (the cards' 2 x 64, the WG targets'
    8 x 128): every beat where the map puts it, join split's inverse."""
    from opentpu.host.board import beat_at
    rng = np.random.default_rng(nch + cb)
    P = nch * cb
    a0 = P * int(rng.integers(0, 4096))
    data = rng.integers(0, 256, 64 * P).astype(np.uint8)
    parts = split(a0, data, chash, nch, cb)
    assert [c for c, _, _ in parts] == list(range(nch)) and {o for _, o, _ in parts} == {a0 // nch}
    for x in rng.integers(0, len(data), 300):
        x = int(x)
        c, off = _ref_channel(a0 + x, nch, cb, chash)
        assert beat_at(a0 + x, chash, nch, cb) == (c, off)
        assert parts[c][2][off - a0 // nch] == data[x]
    assert np.array_equal(join([p for _, _, p in parts], a0, chash, cb), data)


def _wg_cfg(name: str = "bcu1525-x2") -> dict:
    """The CFG_* registers of bcu1525-x2's machine (targets/bcu1525-x2.toml): 4 slices, 8 channels of
    2 GiB each in 128-byte beats; `name` a variant's (bcu1525-s2: 2 slices)."""
    from opentpu.target import arch_hash, build_config, load
    t = load(name)
    a, u, p = t.arch, t.uarch, t.platform
    return dict(S=a["S"], TMEM_WORDS=a["TMEM_WORDS"], IMEM_WORDS=a["IMEM_WORDS"],
                ACT_BLOCKS=a["ACT_BLOCKS"], DRAM_BYTES=a["DRAM_BYTES"],
                channels=p["channels_per_slice"], channel_bytes=p["channel_bytes"],
                beat_bytes=p["beat_bytes"], MXU_IMPL=u["MXU_IMPL"], TMEM_IMPL=u["TMEM_IMPL"],
                VPU_CL=u["VPU_CL"], ULANES=u["ULANES"], SEQ_WC=1, platform=t.platform_id,
                hash=arch_hash(build_config(t)))


@pytest.mark.parametrize("name", ["bcu1525-x2", "bcu1525-s2"])
@pytest.mark.parametrize("chash", [False, True])
def test_wg_random_round_trip_on_a_fake_card(chash, name):
    """The WG targets' map (CFG_DRAM: 8 channels of 128-byte beats a slice, channel j = 8s + c at
    j x 2 GiB): Board.write / read of random unaligned ranges on every slice read back what
    was written, each byte in its channel where the map (I1, _ref_channel) puts it; nothing of
    another slice's channels written. Also the S 2 variant's (bcu1525-s2, named as the transport's
    target: check_target): slice s on channels 8s .. 8s + 7, 16 channels."""
    from opentpu.host.board import channel_map
    from opentpu.host.fake import FakeTransport
    from opentpu.target import load
    f = FakeTransport(devname=None, regmap=4, cfg=_wg_cfg(name), ch_bytes=1 << 16, chash=chash)
    f.want_target = load(name)
    S = f.want_target.arch["S"]
    b = Board(f, lock=False)
    i = b.info()
    assert (b.slices, b.nch, b.beat, len(f.ch)) == (S, 8, 128, 8 * S)
    cb = f.want_target.platform["channel_bytes"]
    assert cb == 1 << 31 and channel_map(i) == tuple(j * cb for j in range(8 * S))
    assert i["target"].variant == name.partition(":")[2]
    rng = np.random.default_rng(7 + chash)
    ref = [np.zeros(8 << 16, np.uint8) for _ in range(S)]
    for _ in range(200):
        s = int(rng.integers(0, S))
        n = int(rng.integers(1, 6000))
        a = int(rng.integers(0, len(ref[s]) - n))
        d = rng.integers(0, 256, n).astype(np.uint8)
        before = [c.copy() for k, c in enumerate(f.ch) if k // 8 != s]
        b.write(a, d, s)
        ref[s][a:a + n] = d
        assert all(np.array_equal(x, c) for x, c in
                   zip(before, (c for k, c in enumerate(f.ch) if k // 8 != s)))
        m = int(rng.integers(1, 6000))
        r = int(rng.integers(0, len(ref[s]) - m))
        assert np.array_equal(b.read(r, m, s), ref[s][r:r + m])
    for s in range(S):
        assert np.array_equal(b.read(0, len(ref[s]), s), ref[s])
        for x in rng.integers(0, len(ref[s]), 500):
            c, off = _ref_channel(int(x), 8, 128, chash)
            assert f.ch[8 * s + c][off] == ref[s][int(x)]


@pytest.mark.parametrize("chash", [False, True])
def test_pipelined_write_hashes_piece_by_piece(chash, monkeypatch):
    """A write of many pipelined pieces (a threaded transport; PIPE cut to 8 beats: pieces of 4
    wide chunks, PIPE / 2 bytes a channel) on the WG targets' map puts each byte in its channel where
    the map puts it (_ref_channel): with CHASH each piece's channel order comes from its own
    address, gathered straight into the channels' buffers (gather): no copy of the span or of a
    piece (hash_swap)."""
    from opentpu.host import board as B
    from opentpu.host.fake import FakeTransport
    monkeypatch.setattr(B, "PIPE", 8 * 128)
    f = FakeTransport(devname=None, regmap=4, cfg=_wg_cfg("bcu1525-x2"), ch_bytes=1 << 16,
                      chash=chash)
    f.threaded, f.bases = True, B.BASE
    b = Board(f, lock=False)
    assert (b.nch, b.beat, b.chash) == (8, 128, chash)
    copies, calls, mw = [], [], f.mem_write
    hs = B.hash_swap
    monkeypatch.setattr(B, "hash_swap", lambda a, v: (copies.append(v.nbytes), hs(a, v))[1])
    f.mem_write = lambda ch, off, dd: calls.append(ch) or mw(ch, off, dd)
    rng = np.random.default_rng(11)
    a, n = 3 * 1024 + 77, 40 * 1024 + 300                # unaligned both ends, ten pieces
    d = rng.integers(0, 256, n).astype(np.uint8)
    b.write(a, d)
    assert copies == [] and calls.count(0) >= 10
    for x in range(a, a + n, 7):
        c, off = _ref_channel(x, 8, 128, chash)
        assert f.ch[c][off] == d[x - a]
    assert np.array_equal(b.read(a, n), d)


def test_dram_layout_of_an_image_without_kv_bytes():
    """The runner status's DRAM layout: a model Image's KV capacity and the part filled; GPT-OSS's
    Image has no kv_bytes (BoardBackend.attach on a card raised AttributeError): none counted."""
    from types import SimpleNamespace as NS
    from opentpu.host.board import dram_layout
    cfg = NS(DRAM_BYTES=1 << 30, IMEM_WORDS=4096, SEQ_WC=False)
    d = dram_layout(cfg, 1 << 28, 1 << 28, NS(cap=512, kv_bytes=1 << 20, batch=2), [256, 1024])
    assert (d["kv_capacity"], d["kv_used"], d["weights"]) == \
        (2 << 20, 3 << 19, (1 << 28) - (2 << 20))
    d = dram_layout(cfg, 1 << 28, 1 << 28, NS(cap=512, batch=1), [100])
    assert (d["kv_capacity"], d["kv_used"], d["weights"]) == (0, 0, 1 << 28)


@pytest.mark.parametrize("chash", [False, True])
def test_wg_waitw_tag_and_channel_checks_on_a_fake_card(chash):
    """The bring-up checks on the WG targets' channels (8 of 128-byte beats a slice): pattern_test,
    address_lines, partial_writes and channel_patterns pass at the beat; waitw_tag splits each
    record into its 8 channel runs (CHASH's permutation) and writes the tag's channel's run
    last, ending in the tag beat. The fake computes nothing, so the card's side (its copy of the
    sampled 64-byte pieces after the WAITW) is played by Board.wait: they read back the new
    data where Board.read's map puts it, the last ones the 2 x WT_SAMPLES / 8 chunks before
    the tag on every channel."""
    from opentpu.host import checks as C
    from opentpu.host.fake import FakeTransport
    f = FakeTransport(devname=None, regmap=4, cfg={**_wg_cfg(), "hash": 0, "SEQ_WC": 0},
                      ch_bytes=1 << 20, chash=chash)
    b = Board(f, lock=False)
    assert (b.nch, b.beat, b.chash) == (8, 128, chash)
    ok, msg = pattern_test(b, [(0, 4096), (4096 + 60, 1000), ((8 << 20) - 777, 777)])
    assert ok, msg
    for c in (0, 7, 13, 31):
        assert address_lines(f, c, 1 << 20, beat=128)[0]
        assert channel_patterns(f, c, 1 << 20)[0]
        assert partial_writes(f, c, base=1 << 17, beat=128)[0]
    sizes, base, rounds = (1 << 16, 3 << 15), 0x40000, 4
    seen, calls, mw, r = [], [], f.mem_write, [0]
    f.mem_write = lambda ch, off, d: calls.append((ch, off, len(d))) or mw(ch, off, d)

    def start(*a, **kw):
        calls.clear()

    def wait(*a, **kw):                     # the card: its WAITW saw the tag, then its copy
        p = C.waitw_tag_round(r[0], 5, sizes, base, 8, 128)
        ct, off = b.beat_at(p["tag"])
        assert [x[0] for x in calls].count(ct) == 1 and calls[-1][0] == ct
        assert calls[-1][1] + calls[-1][2] == off + 128          # the run ends in the tag beat
        assert sorted(x[0] for x in calls) == list(range(8))
        for i, k in enumerate(p["beats"]):
            b.write(C.W_RES + 64 * i, b.read(p["addr"] + 64 * k, 64))
        seen.append(p["tag"])
        r[0] += 1
        return {"cycles": 1}
    b.start, b.wait = start, wait
    ok, msg = C.waitw_tag(b, rounds=rounds, sizes=sizes, base=base)
    assert ok and len(seen) == rounds, msg
    assert {b.beat_at(a)[0] for a in seen} != {0} or not chash


def _bcu_cfg() -> dict:
    """The CFG_* registers of the BCU1525's build (targets/bcu1525.toml)."""
    from opentpu.target import arch_hash, build_config, load
    t = load("bcu1525")
    a, u, p = t.arch, t.uarch, t.platform
    return dict(S=a["S"], TMEM_WORDS=a["TMEM_WORDS"], IMEM_WORDS=a["IMEM_WORDS"],
                ACT_BLOCKS=a["ACT_BLOCKS"], DRAM_BYTES=a["DRAM_BYTES"],
                channels=p["channels_per_slice"], channel_bytes=p["channel_bytes"],
                MXU_IMPL=u["MXU_IMPL"], TMEM_IMPL=u["TMEM_IMPL"], VPU_CL=u["VPU_CL"],
                ULANES=u["ULANES"], platform=t.platform_id, hash=arch_hash(build_config(t)))


def test_two_slices_on_a_fake_card():
    """A two-slice bitstream (the BCU1525's CFG_* registers): slice s's DRAM on the card's
    channels 2s and 2s + 1, the XDMA channel map from them; one program loads into slice 0
    and a HALT into slice 1, a list gives each slice its own (the shorter padded with NOPs to
    the shared PROG_N); a run reports every slice's ICOUNT."""
    from opentpu import isa as I
    from opentpu.host.board import XdmaTransport, channel_map
    from opentpu.host.fake import FakeTransport
    f = FakeTransport(devname=None, regmap=4, cfg=_bcu_cfg(), ch_bytes=1 << 16)
    b = Board(f, lock=False)
    i = b.info()
    assert b.slices == 2 and len(f.ch) == 4
    assert channel_map(i) == (0, 1 << 31, 2 << 31, 3 << 31)
    assert XdmaTransport.bases == BASE
    x = np.arange(1000, dtype=np.uint32).view(np.uint8)
    b.write(256, x, 1)
    assert not f.ch[0].any() and not f.ch[1].any() and f.ch[2].any() and f.ch[3].any()
    assert np.array_equal(b.read(256, len(x), 1), x)
    y = np.arange(37, dtype=np.uint8)
    b.write(100, y)
    assert np.array_equal(b.read(100, 37), y) and np.array_equal(b.read(256, len(x), 1), x)
    w = np.asarray(I.assemble([I.nop(), I.nop(), I.halt()]), np.uint32)
    b.load_program(4096, w)
    halt = np.asarray(I.assemble([I.halt()]), np.uint32)
    assert f.regs[R.R_PROG_N] == 3
    assert np.array_equal(b.read(4096, 4 * len(w)).view(np.uint32), w)
    assert np.array_equal(b.read(4096, 4 * len(w), 1).view(np.uint32),
                          np.concatenate([halt, np.zeros(len(w) - len(halt), np.uint32)]))
    assert b.run()["instructions"] == [3, 3]
    b.load_program(8192, [halt, w])
    assert np.array_equal(b.read(8192, 4 * len(w), 1).view(np.uint32), w)
    assert np.array_equal(b.read(8192, 4 * len(halt)).view(np.uint32), halt)
    with pytest.raises(ValueError, match="3 programs for a bitstream of 2 slices"):
        b.load_program(8192, [w, w, w])
    with pytest.raises(ValueError, match="PROG_ADDR is a byte address in the slice's first 4"):
        b.load_program(1 << 32, w)              # (an Image.high's past its weights: wrapped)


# ------------------------------------------------------------------------------ board model
CFG = sim_board_config(DRAM_BYTES=1 << 22)


def test_program_on_board_model(have_verilator):
    """LD/ST (aligned and not), simple and composite VOPs, QACT, MM, QST on the board model."""
    t = SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=30, seed=3)
    b = Board(t)
    assert b.info()["calibrated"]
    ok, msg, st = run_demo(b, CFG)
    assert ok, msg
    assert st["b_reads"] > 0 and st["a_writes"] > 0 and st["cycles"] > 0


@pytest.mark.parametrize("seed", [3, 4])
def test_program_on_native_board_model(have_verilator, seed):
    """The same on the native memory model alone (MEM_NATIVE=1: otpu_native_dram in front of
    otpu_native_mem), whatever OTPU_NATIVE says; and the partial writes."""
    b = Board(SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=30, seed=seed, native=True))
    ok, msg, st = run_demo(b, CFG)
    assert ok, msg
    assert st["b_reads"] > 0 and st["a_writes"] > 0 and st["cycles"] > 0
    ok, msg, st = run_demo(b, CFG, masked_program())
    assert ok, msg
    assert st["a_writes"] > 500 and st["b_writes"] > 20


def test_program_on_channel_board_model(have_verilator):
    """The same on the build's channels (MEM_NATIVE=2: otpu_native_dram, then per channel
    otpu_mem_ch in front of a model of LiteDRAM's native port in its own clock, partial beats
    read-modified-written in otpu_mem_ch), whatever OTPU_NATIVE says."""
    b = Board(SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=30, seed=3, native="ld"))
    ok, msg, st = run_demo(b, CFG)
    assert ok, msg
    assert st["b_reads"] > 0 and st["a_writes"] > 0 and st["cycles"] > 0
    ok, msg, st = run_demo(b, CFG, masked_program())
    assert ok, msg
    assert st["a_writes"] > 500 and st["b_writes"] > 20


def test_dram_turnarounds_on_channel_board_model(have_verilator):
    """tools/qual/turnaround.py (the qual's read / write turnaround check) on the build's channels:
    two runs of its program, then the stored tiles and the MMs' results equal the ISA
    simulator's (the board model has no ECC counters)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "qual"))
    from turnaround import turnarounds
    b = Board(SimTransport(ch_bytes=4 << 20, stall=30, seed=3, native="ld"))
    rows = dict(turnarounds(b, mb=0.25, runs=2))
    assert rows["data"][0], rows["data"][1]
    assert rows["ECC"] == (True, "no ECC on the board model")


def test_partial_dram_writes_on_board_model(have_verilator):
    """QST byte writes and short word-masked stores (read-modify-writes in the board's memory
    controllers, which have no DDR3 data-mask pins)."""
    b = Board(SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=40, seed=9))
    ok, msg, st = run_demo(b, CFG, masked_program())
    assert ok, msg
    assert st["a_writes"] > 500 and st["b_writes"] > 20


def test_vops_on_board_model(have_verilator):
    """RDOT / OUTER / LOG2 (the selftest's vops stage) on the board model."""
    b = Board(SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=20, seed=5))
    ok, msg, _ = run_demo(b, CFG, vops_program())
    assert ok, msg


def test_without_outer_on_board_model(have_verilator):
    """A board built without VOP OUTER (otpu_board OUTER 0, the WG targets'): CAPS2 bit1, the host's
    Config without OUTER (device_config, the CFG_HASH checked with it), and the selftest's vops
    stage (the vops program and the reduction / RDOT / LOG2 op checks, without OUTER) and stream
    stage (a DSTEP: the WG targets' DeltaNet) bit for bit."""
    import dataclasses
    from opentpu.host.board import device_config
    from opentpu.host.checks import stream_program, vops_check
    b = Board(SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=20, seed=5, params={"OUTER": 0}))
    i = b.info()
    assert i["caps2"]["outer"] is False and not device_config(i).OUTER
    cfg = dataclasses.replace(CFG, OUTER=False)
    ok, msg = vops_check(b, cfg)
    assert ok, msg
    ok, msg, _ = run_demo(b, dataclasses.replace(cfg, DSTEP=True, STREAM=True), stream_program())
    assert ok, msg


def test_stream_on_board_model(have_verilator):
    """STREAM in every mode of the board's subset and a DSTEP (the selftest's stream stage) on
    the board model."""
    import dataclasses
    from opentpu.host.checks import stream_program
    b = Board(SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=20, seed=6))
    ok, msg, _ = run_demo(b, dataclasses.replace(CFG, DSTEP=True, STREAM=True), stream_program())
    assert ok, msg


_OPS = op_checks(CFG, gen=True, waitw=True, gather=rtlsim.gathers(CFG.S))


@pytest.mark.parametrize("group,name,prog", _OPS, ids=[name for _, name, _ in _OPS])
def test_op_check_on_board_model(have_verilator, group, name, prog):
    """otpu-diag's per-instruction programs (opentpu/host/opchecks.py), bit for bit; GATHER where
    the simulated collective unit has it, as otpu-diag takes it from CAPS2 (the card's: BAR
    alone)."""
    ok, msg, _ = run_demo(Board(SimTransport(ch_bytes=CFG.DRAM_BYTES // 2)), CFG, prog,
                          diag_image(CFG))
    assert ok, msg


def test_mm_sw_line_alone_on_board_model(have_verilator):
    """An MM that starts a program (a HALT CHAIN's: the loader waited for every read, and ACT RAM
    and TMEM are kept) and whose first scale row has its Sw alone in a 32-byte line (the rows 28
    bytes into one): its first port A read, that line, goes out with nothing else in flight, so
    for a cycle the slice's only read is the one in its port A register. tb_board stops if the
    board's read idle or the slice's loader start misses it (otpu_slice); bit for bit against
    the ISA simulator (the demo's scale rows, moved there)."""
    from opentpu import isa as I
    from opentpu.host.checks import DATA, OUT, SC, W8
    from opentpu.host.opchecks import CHAIN_AT
    M, res, at, ssa = CFG.MCOLS, 8192, CHAIN_AT + 0x1000, SC + 0x100 + 28
    mm = [I.mm(W8, ssa, res, 16, 2, 256, 16, M, 0, 8), I.st(OUT, res, 16 * M), I.halt()]
    w = np.asarray(I.assemble(mm), np.uint32)
    if CFG.SEQ_WC:
        from opentpu import waits as W
        w = W.loadable(w, W.entries(w, CFG.D, CFG.S, prog=mm), CFG.D)
    img = diag_image(CFG)
    img[ssa:ssa + 16 * 8] = img[SC:SC + 16 * 8]
    img[at:at + 4 * len(w)] = w.view(np.uint8)
    prog = [I.ld(DATA, 0, 4096), I.qact(0, M, 0, 2, 256), I.li(1, at), I.li(2, len(mm)),
            I.halt(chain=True, ra=1, rb=2)]
    ok, msg, _ = run_demo(Board(SimTransport(ch_bytes=CFG.DRAM_BYTES // 2)), CFG, prog, img)
    assert ok, msg


def test_porta_beats_dropped_at_a_waitw_on_board_model(have_verilator):
    """An MM, then a WAITW whose word holds at once (ordered after the MM by the word it writes
    into the MM's output): its end drops port A's held beats (otpu_native_dram a_flush; the
    MM's scale reads leave the reused beat and a run behind). tb_board stops if a beat outlives
    the flush; bit for bit against the ISA simulator."""
    from opentpu import isa as I
    from opentpu.host.checks import DATA, OUT, SC, W8
    from opentpu.host.opchecks import CHAIN_AT
    M, res, flag = CFG.MCOLS, 8192, CHAIN_AT + 0x2000
    img = diag_image(CFG)
    img[flag:flag + 4] = np.array([1], "<u4").view(np.uint8)
    prog = [I.ld(DATA, 0, 4096), I.qact(0, M, 0, 2, 256),
            I.mm(W8, SC, res, 16, 2, 256, 16, M, 0, 8),
            I.waitw(flag, res + 16 * M - 1, 1, I.C_EQ), I.st(OUT, res, 16 * M), I.halt()]
    ok, msg, _ = run_demo(Board(SimTransport(ch_bytes=CFG.DRAM_BYTES // 2)), CFG, prog, img)
    assert ok, msg


def test_waitw_host_check_on_board_model(have_verilator):
    """qual.sh's WAITW phase (checks.waitw_host, waitw_timeout) through the host driver: the board
    model runs a script, so the host's data and flag are there before the start (the writes
    during the run: tests/test_waitw_rtl.py); a WAITW timeout stops with ERROR, the next run
    halts normally."""
    from opentpu.host.checks import waitw_host, waitw_timeout
    b = Board(SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=20, seed=4))
    ok, msg = waitw_host(b, rounds=5, sizes=(16, 17, 1000))
    assert ok and "5 rounds" in msg, msg
    ok, msg = waitw_timeout(b)
    assert ok and msg.startswith("ERROR and WAIT_TO at the timeout"), msg


def test_pattern_and_address_lines_on_board_model(have_verilator):
    t = SimTransport(ch_bytes=1 << 21)
    b = Board(t, check=False)
    ok, msg = pattern_test(b, [(0, 4096), (4096 + 60, 1000), ((1 << 22) - 777, 777)])
    assert ok, msg
    for c in (0, 1):
        for check in (address_lines, channel_patterns):
            ok, msg = check(t, c, 1 << 21)
            assert ok, msg
        ok, msg = partial_writes(t, c)
        assert ok, msg


# ------------------------------------------------------------------------------ stops and resets
@pytest.mark.parametrize("native", [True, "ld"])
@pytest.mark.parametrize("after", [600, 1300, 2500])
def test_run_dropped_then_load_and_run_at_once(have_verilator, native, after):
    """CTRL = 0 in the middle of a run that streams reads and writes (600 and 1300 cycles in:
    during its first LD's reads; 2500: later), then at once (without the wait for WR_IDLE,
    docs/observability.md "Stopping a run") a LOAD of another program and its RUN: the loader
    takes none of the dropped run's read data still in flight (it starts once the memory
    adapter's reads are all back), the run starts once its writes are answered, and the second
    program's results and instruction count are the ISA simulator's. Before, a drop during the
    reads (600, 1300) left the loader writing their late data into IMEM: the program hung."""
    import dataclasses
    from opentpu import isa as I
    from opentpu.host.checks import PROG_AT, ZERO_AT, demo_image, demo_program
    from opentpu.isasim import Machine
    a_out = 0x300000                                    # the dropped run's stores (not compared)
    prog_a = [I.loop(2, 1 << 20), I.ld(0, 0, 16384), I.st(a_out, 0, 16384), I.halt()]
    prog_b = [I.ld(ZERO_AT, 0, CFG.TMEM_WORDS)] + demo_program()
    img = demo_image()
    ref = np.zeros(CFG.DRAM_BYTES, np.uint8)
    ref[:len(img)] = img
    sl = Machine(dataclasses.replace(CFG, DRAM_BYTES=len(ref)), [prog_b], [ref]).run().slices[0]
    t = SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=50, seed=after, native=native,
                     plusargs=["+max_cycles=4000000"])
    b = Board(t, check=False)
    b.write(0, img)
    words = [np.asarray(I.assemble(p), np.uint32) for p in (prog_a, prog_b)]
    at = [PROG_AT, PROG_AT + 0x10000]
    for a, w in zip(at, words):
        if b.waitc:                                     # WAITC: the entries after the words
            from opentpu import waits as W
            w = W.loadable(w, W.entries(w, CFG.D, CFG.S), CFG.D)
        b.write(a, w.view(np.uint8))

    def load(a, w):
        t.reg_write(R.R_CTRL, 0)
        t.reg_write(R.R_PROG_ADDR, a)
        t.reg_write(R.R_PROG_N, len(w) // 8)
        t.reg_write(R.R_CTRL, R.CTRL_LOAD)
        t.poll(R.R_STATUS, R.ST_LOADING, 0)
    load(at[0], words[0])
    t.reg_write(R.R_CTRL, R.CTRL_CLEAR)
    t.reg_write(R.R_CTRL, R.CTRL_RUN)
    t.wait_cycles(after)
    load(at[1], words[1])                               # (its CTRL = 0 drops the run)
    st = b.run(timeout=10.0)
    assert st["instructions"][0] == sl.icount
    assert np.array_equal(b.read(0, a_out), sl.dram[:a_out])


def test_axil_iso_resets_apart(have_verilator):
    """otpu_axil_iso (otpu_ctrl.sv: in front of the control registers and the LiteDRAM core's CSR
    port) under random master and slave resets (sim/verilator/tb_axil_iso.sv): no response the
    master did not ask for, every access answered (SLVERR after a slave reset), the slave never
    left with half a write. Without it (BYPASS, the ports as they were) the same traffic fails:
    a response to an access from before the master's reset reaches the master after it."""
    import subprocess
    from opentpu import rtlsim
    for bypass in (0, 1):
        exe = rtlsim.build("tb_axil_iso", [rtlsim.RTL / "shell/otpu_ctrl.sv",
                                           rtlsim.TB / "tb_axil_iso.sv"],
                           {"BYPASS": bypass})
        for seed in (1, 2):
            r = subprocess.run([str(exe), f"+seed={seed}", "+mrst=300", "+slvrst=60",
                                "+cycles=400000"], capture_output=True, text=True, timeout=120)
            if bypass:
                assert "FAIL" in r.stdout and "did not ask for" in r.stdout, r.stdout[-2000:]
            else:
                assert "PASS" in r.stdout, r.stdout[-2000:]


def test_uncalibrated_channel_refuses_run_and_load(have_verilator):
    """A channel not calibrated (STATUS CALIB1 clear, +calib): CTRL takes neither RUN nor LOAD
    (nothing would answer their memory requests); TEMP is valid all the same (the XADC does not
    wait for the DDR3)."""
    t = SimTransport(ch_bytes=1 << 20, plusargs=["+calib=1"])
    t.reg_write(R.R_PROG_ADDR, 0)
    t.reg_write(R.R_PROG_N, 4)
    t.reg_write(R.R_CTRL, R.CTRL_LOAD)
    t.wait_cycles(20)
    i = [t.queue_read(R.R_STATUS)]
    t.reg_write(R.R_CTRL, R.CTRL_RUN)
    t.wait_cycles(20)
    i += [t.queue_read(R.R_STATUS), t.queue_read(R.R_ICOUNT), t.queue_read(R.R_TEMP)]
    t.flush()
    st0, st1, ic, temp = (t.results[k] for k in i)
    for st in (st0, st1):
        assert st & (R.ST_CALIB0 | R.ST_CALIB1) == R.ST_CALIB0
        assert not st & (R.ST_LOADING | R.ST_RUN)
    assert ic == 0 and temp & R.TEMP_VALID


@pytest.mark.parametrize("ded", [0, 2])
def test_ecc_ded_in_status(have_verilator, ded):
    """STATUS ECC_DED: a channel's uncorrectable-ECC count is not zero (+ded, the LiteDRAM core's
    c<n>_ded); CLEAR does not clear it. otpu-selftest's ecc stage fails on it."""
    from opentpu.host import selftest
    t = SimTransport(ch_bytes=1 << 20, plusargs=[f"+ded={ded}"])
    t.reg_write(R.R_CTRL, R.CTRL_CLEAR)
    t.reg_write(R.R_CTRL, 0)
    assert bool(t.reg_read(R.R_STATUS) & R.ST_ECC_DED) == bool(ded)
    ok, msg = selftest.ecc_check(t)
    assert ok == (not ded) and msg.startswith("ECC_DED set" if ded else "no uncorrectable word")


def test_stream_without_the_stream_engine_is_an_error(have_verilator):
    """On a bitstream without the stream engine (DSTEP = 0, CAPS bit6 clear) a DSTEP or STREAM
    stops the run with ERROR, an illegal instruction (the sequencer had sent it to a unit that
    is not there, and the run hung)."""
    import dataclasses
    from opentpu.host.checks import stream_program
    b = Board(SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, params={"DSTEP": 0},
                           plusargs=["+max_cycles=400000"]))
    with pytest.raises(RuntimeError, match="illegal instruction"):
        run_demo(b, dataclasses.replace(CFG, DSTEP=True, STREAM=True), stream_program())


# ------------------------------------------------------------------------------ Qwen3
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.mark.parametrize("formats", ["", "attn=fp4,down=int4,head=fp4",
                                     "attn@0=fp4,mlp@1=fp4"])
def test_tiny_qwen3_on_board_model(have_verilator, formats):
    """Logits bit-identical to the ISA simulator; with per-kind weight formats too (int8,
    fp4 and int4 MMs in one layer), and per layer (a run, a layer block layout, each)."""
    import dataclasses
    from opentpu.llm.qwen3 import Engine, Spec
    torch.manual_seed(0)
    hc = transformers.Qwen3Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=128, intermediate_size=512,
                                  vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
                                  tie_word_embeddings=True, max_position_embeddings=4096)
    mdl = transformers.Qwen3ForCausalLM(hc).float().eval()
    W = {k: v.float().numpy() for k, v in mdl.state_dict().items()}
    spec = dataclasses.replace(Spec(256, 2, 4, 2, 128, 512, 1000), formats=formats)
    cfg = sim_board_config(DRAM_BYTES=1 << 23)
    isa = Engine(spec, W, cap=256, cfg=cfg)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5)
    brd = Engine(spec, W, cap=256, cfg=cfg,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    for tok in (11, 222, 333):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert brd.stats[-1]["cycles"] > 0


@pytest.mark.parametrize("nreg", [0, 0b1010])
def test_tiny_qwen3_two_slices_on_board_model(have_verilator, nreg):
    """The BCU1525's board (otpu_board S = 2: two slices, each on its two channels, one
    control block, the collective unit between them; SimTransport(target="bcu1525")) runs a
    tiny Qwen3 split over two slices: logits bit-identical to the ISA simulator at S = 2, and
    each slice's DRAM, read back through the host's channel map, the ISA simulator's. nreg: the
    channels that cross an SLR on the board (1 and 3) through two otpu_native_reg each, as
    otpu_bcu_sys builds them."""
    from opentpu.llm.qwen3 import Engine, Spec
    torch.manual_seed(0)
    hc = transformers.Qwen3Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=128, intermediate_size=512,
                                  vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
                                  tie_word_embeddings=True, max_position_embeddings=4096)
    mdl = transformers.Qwen3ForCausalLM(hc).float().eval()
    W = {k: v.float().numpy() for k, v in mdl.state_dict().items()}
    spec = Spec(256, 2, 4, 2, 128, 512, 1000)
    cfg = sim_board_config(DRAM_BYTES=1 << 23, S=2)
    isa = Engine(spec, W, cap=256, cfg=cfg)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5, target="bcu1525",
                      params={"NREG": nreg})
    brd = Engine(spec, W, cap=256, cfg=cfg,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    assert brd.backend.info["cfg"]["S"] == 2 and brd.backend.info["cfg"]["platform"] == 2
    for tok in (11, 222):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
        st = brd.stats[-1]
        assert st["cycles"] > 0 and st["instructions"] == isa.stats[-1]["instructions"]
    n = isa.image.nbytes
    for s in range(2):
        assert np.array_equal(brd.backend.read(s, 0, n), isa.backend.machine.slices[s].dram[:n])


# BAR / GATHER across the BCU1525's SLR crossings (otpu_board g_xs: otpu_coll_hub, a port per
# slice): TMEM words, DRAM bytes
_CA, _CR, _CG, _CG2, _CG3, _CG4 = 0, 4096, 8192, 10240, 12288, 14336
_CDATA, _CACC, _COUT, _CPROG = 0x10000, 0x20000, 0x100000, 0x300000
_COUTS = ((0, _CG, 400), (0x1000, _CG2, 160), (0x2000, _CG3, 512), (0x3000, _CG4, 512),
          (0x4000, _CR, 1024))


def _collective_program(s: int) -> list:
    """Slice s's program: BARs back to back, a BAR while the other slice is still busy (slice 1
    first, then slice 0: a loop of VOPs on its accumulator), BARs right after and right before
    GATHERs, two GATHERs back to back (the second of the lagging accumulators), the results to
    DRAM."""
    from opentpu import isa as I

    def lag(who):
        return [I.loop(1, 120), I.vop(I.V_ADD, _CR, _CR, _CA, 4, 256, 256, 256, 256)] \
            if s == who else []
    return [I.ld(_CDATA, _CA, 2048), I.ld(_CACC, _CR, 1024),
            I.bar(), I.bar(), I.bar(),
            *lag(1), I.bar(),
            I.gather(_CA, _CG, 3, 50, 64, 64, 192), I.bar(),
            *lag(0), I.bar(), I.gather(_CA + 512, _CG2, 2, 37, 40, 40, 80),
            I.bar(), I.gather(_CR, _CG3, 4, 64, 256, 64, 256),
            I.gather(_CR + 8, _CG4, 4, 64, 256, 64, 256), I.bar(), I.bar(),
            *[I.st(_COUT + o, t, n) for o, t, n in _COUTS],
            I.halt()]


@pytest.mark.parametrize("nreg,seed,coll", [(0b1010, 5, {}), (0, 9, {}),
                                             (0b1010, 5, {"COLL_PIPE": 1}),
                                             (0, 9, {"COLL_PIPE": 1, "XRS": 3})])
def test_bar_and_gather_across_slrs_on_board_model(have_verilator, nreg, seed, coll):
    """The BCU1525's board (S = 2, the board's SLR crossings, the collective through
    otpu_coll_hub and a port per slice; nreg: the channels' crossings, as the board's 1 and 3,
    or none) runs BARs back to back, a BAR with one slice far behind the other (each in turn),
    BARs right after and right before GATHERs and GATHERs back to back, under two memory stall
    patterns (seed): the results and both slices' instruction counts are the ISA simulator's.
    coll: otpu_board's COLL_PIPE (the hub's reads in flight) and XRS (a stage more each way).
    (One run per simulation: each flush of the board model starts it from reset.)"""
    from opentpu import isa as I
    from opentpu.isasim import Machine
    cfg = board_config(DRAM_BYTES=1 << 23, S=2)
    progs = [_collective_program(s) for s in range(2)]
    rng = np.random.default_rng(11)
    ins = [(rng.standard_normal(2048).astype(np.float32), rng.standard_normal(1024)
            .astype(np.float32)) for _ in range(2)]
    drams = []
    for x, y in ins:
        d = np.zeros(cfg.DRAM_BYTES, np.uint8)
        d[_CDATA:_CDATA + x.nbytes] = x.view(np.uint8)
        d[_CACC:_CACC + y.nbytes] = y.view(np.uint8)
        drams.append(d)
    m = Machine(cfg, progs, drams).run()
    b = Board(SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=seed, target="bcu1525",
                           params={"NREG": nreg} | coll))
    assert b.slices == 2
    for s, (x, y) in enumerate(ins):
        b.write(_CDATA, x.view(np.uint8), s)
        b.write(_CACC, y.view(np.uint8), s)
    words = [np.asarray(I.assemble(p), np.uint32) for p in progs]
    if b.waitc:                                         # WAITC: each slice's program its entries
        from opentpu import waits as W
        b.load_program(_CPROG, words, [W.entries(w, cfg.D, cfg.S) for w in words])
    else:
        b.load_program(_CPROG, words)
    st = b.run()
    assert st["instructions"] == [sl.icount for sl in m.slices]
    for s in range(2):
        for o, _, n in _COUTS:
            got = b.read(_COUT + o, 4 * n, s)
            assert np.array_equal(got, m.slices[s].dram[_COUT + o:_COUT + o + 4 * n]), (s, hex(o))


@pytest.mark.parametrize("pipe", [0, 1])
def test_run_cut_mid_gather_on_board_model(have_verilator, pipe):
    """A run cut short mid-GATHER on the BCU1525's board (S = 2; COLL_PIPE 0 or 1): CTRL = 0 holds
    the core in reset with the collective's reads and writes in flight through the crossings,
    and the control master's reset (ctl_mrst, tb_board +mrst_c) comes during the cut run; then
    the program runs again whole. The cut run had not halted (6,000 cycles: its eight GATHERs of
    4,096 words a slice take ~8,400 at COLL_PIPE 1, ~74,000 at 0; ICOUNT counts the instructions
    issued, all of them by then), and the second run's results and instruction counts are the ISA
    simulator's: the hub, the ports and their crossings start it clean."""
    from opentpu import isa as I
    from opentpu.isasim import Machine
    cfg = board_config(DRAM_BYTES=1 << 23, S=2)
    n_g = 8
    progs = [[I.ld(_CDATA, _CA, 4096), *[I.gather(_CA, _CG, 1, 4096, 4096, 4096, 4096)
                                          for _ in range(n_g)],
              I.st(_COUT, _CG, 8192), I.halt()] for _ in range(2)]
    rng = np.random.default_rng(13)
    xs = [rng.standard_normal(4096).astype(np.float32) for _ in range(2)]
    drams = []
    for x in xs:
        d = np.zeros(cfg.DRAM_BYTES, np.uint8)
        d[_CDATA:_CDATA + x.nbytes] = x.view(np.uint8)
        drams.append(d)
    m = Machine(cfg, progs, drams).run()
    b = Board(SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=7, target="bcu1525",
                           params={"COLL_PIPE": pipe}, plusargs=["+mrst_c=3000"]))
    for s, x in enumerate(xs):
        b.write(_CDATA, x.view(np.uint8), s)
    words = [np.asarray(I.assemble(p), np.uint32) for p in progs]
    if b.waitc:                                         # WAITC: each slice's program its entries
        from opentpu import waits as W
        b.load_program(_CPROG, words, [W.entries(w, cfg.D, cfg.S) for w in words])
    else:
        b.load_program(_CPROG, words)
    t = b.t
    t.reg_write(R.R_CTRL, R.CTRL_CLEAR)
    t.reg_write(R.R_CTRL, R.CTRL_RUN)
    t.wait_cycles(6000)                                 # (ctl_mrst at 3,000 of them)
    ic = t.queue_read(R.R_STATUS)
    t.reg_write(R.R_CTRL, 0)
    t.poll(R.R_STATUS, R.ST_WR_IDLE, R.ST_WR_IDLE)
    t.wait_cycles(100)
    first: list[int] = []                               # the reads of run()'s first flush: the
    flush = t.flush                                     # simulation with the cut run in it
    def flush1():                                       # (each flush's reads its own)
        flush()
        if not first:
            first.extend(t.results)
    t.flush = flush1
    st = b.run()
    assert first[ic] & R.ST_RUN and not first[ic] & R.ST_HALTED, hex(first[ic])  # cut mid-run
    assert st["instructions"] == [sl.icount for sl in m.slices]
    for s in range(2):
        got = b.read(_COUT, 4 * 8192, s)
        assert np.array_equal(got, m.slices[s].dram[_COUT:_COUT + 4 * 8192]), s


@pytest.mark.parametrize("pipe", [0, 1])
def test_collective_reads_held_behind_dma_reads_on_board_model(have_verilator, pipe):
    """test_rtl's held collective on the BCU1525's board (S = 2: the collective through
    otpu_coll_hub and a port per slice) with release r2's TMEM (the collective the DMA's guest):
    each port asks its held read from its second cycle (TMEM's input stage, otpu_coll_port hq;
    COLL_PIPE 1: from its read FIFO's head, two cycles after it came at the earliest) and holds
    it while the guest rule blocks it (an ST beside the GATHERs); the results and the
    instruction counts are the ISA simulator's."""
    from opentpu import isa as I
    from opentpu.isasim import Machine
    from test_rtl import R2_TMEM, held_collective_program
    cfg = board_config(DRAM_BYTES=1 << 23, S=2)
    out = 0x100000
    progs = [held_collective_program(out) for _ in range(2)]
    rng = np.random.default_rng(12)
    xs = [rng.standard_normal(4096).astype(np.float32) for _ in range(2)]
    drams = []
    for x in xs:
        d = np.zeros(cfg.DRAM_BYTES, np.uint8)
        d[:x.nbytes] = x.view(np.uint8)
        drams.append(d)
    m = Machine(cfg, progs, drams).run()
    b = Board(SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=3, target="bcu1525",
                           params=dict(R2_TMEM) | ({"COLL_PIPE": 1} if pipe else {})))
    assert b.slices == 2
    for s, x in enumerate(xs):
        b.write(0, x.view(np.uint8), s)
    words = [np.asarray(I.assemble(p), np.uint32) for p in progs]
    if b.waitc:                                         # WAITC: each slice's program its entries
        from opentpu import waits as W
        b.load_program(_CPROG, words, [W.entries(w, cfg.D, cfg.S) for w in words])
    else:
        b.load_program(_CPROG, words)
    st = b.run()
    held = {int(a): int(c) for a, c in re.findall(r"TMEMSTAT sid=(\d+) cyc=\d+ lose=\d+,\d+,\d+,(\d+)",
                                                  b.t.out)}    # the run's simulation
    assert sorted(held) == [0, 1] and held[0] > 0, held  # (slice 0's segment is read first)
    assert st["instructions"] == [sl.icount for sl in m.slices]
    for s in range(2):
        for o, n in ((0, 4096), (0x8000, 512), (0x9000, 160)):
            got = b.read(out + o, 4 * n, s)
            assert np.array_equal(got, m.slices[s].dram[out + o:out + o + 4 * n]), (s, hex(o))


def test_tiny_qwen3_generate_on_board_model(have_verilator):
    """The generate loop through the board (CAPS bit30, BoardBackend.run_generate): from the
    DRAM state of a 248-token prefill on the ISA simulator, 12 tokens across the attention
    bucket boundary at 256 in one start (HALT CHAIN on the board's loader), token for token
    and the whole DRAM of the ISA simulator's run."""
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, Spec
    torch.manual_seed(0)
    hc = transformers.Qwen3Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=128, intermediate_size=512,
                                  vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
                                  tie_word_embeddings=True, max_position_embeddings=4096,
                                  initializer_range=0.2)
    mdl = transformers.Qwen3ForCausalLM(hc).float().eval()
    W = {k: v.float().numpy() for k, v in mdl.state_dict().items()}
    spec = Spec(256, 2, 4, 2, 128, 512, 1000)
    need = spec.image(sim_board_config(DRAM_BYTES=1 << 40), 512, rows=PREFILL_ROWS,
                      lookup=True).nbytes
    size = 1 << (need - 1).bit_length()
    cfg = sim_board_config(DRAM_BYTES=size)
    eng = Engine(spec, W, cap=512, cfg=cfg, resident=True)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 248)]
    t0 = int(np.argmax(eng.prefill(toks)))
    isa, pos, n = eng.backend, eng.pos, eng.image.nbytes
    tr = SimTransport(ch_bytes=size // 2, stall=20, seed=5)
    brd = BoardBackend(cfg, [isa.machine.slices[0].dram[:n]], transport=tr)   # the prefill's
    assert brd.generates and brd.chains
    if brd.waitc:       # WAITC: the ISA's chain slots as the board's loader writes them (the
        isa.slot_words = brd.slot_words     # entries after the words), so the DRAMs compare whole
    want = eng.generate_card(t0, 12, stop_ids=[])
    eng.backend, eng.pos, eng._chained = brd, pos, {}
    seen = []
    got = eng.generate_card(t0, 12, stop_ids=[], on_token=seen.append)
    assert got == want == seen and len(set(got)) >= 3
    assert eng.stats[-1]["cycles"] > 0
    assert np.array_equal(brd.read(0, 0, n), isa.machine.slices[0].dram[:n])


def _fake_bcu(monkeypatch, **kw):
    """A fake card of the bcu1525-wg's registers (AW 36: 8 channels of 4 GiB, 32 GiB), its
    CFG_HASH 0 (unchecked), channels of 1 GiB in RAM."""
    from opentpu.host import fake as F
    monkeypatch.setenv("OTPU_TARGET", "bcu1525-wg")
    c = board_config()
    cf = dict(F._target_cfg(c.D, c.MCOLS, c.LANES, False, False, False, c.act_rows), hash=0,
              **kw)
    return F.FakeTransport(ch_bytes=1 << 30, regmap=4, cfg=cf)


def _info(t) -> dict:
    """The card's info from a Board closed after it (the device lock is the next Board's: a
    later test's fake card of the same name is refused while one is held)."""
    b = Board(t)
    try:
        return b.info()
    finally:
        b.close()


def test_an_image_of_the_bank_half_window_skips_the_window_past_its_mixers(monkeypatch):
    """qwen3.Images with skip (qwen35 Image.mix: the DeltaNet mixers' weights from the bank-half
    window's upper half, the other weights past the window): BoardBackend writes [0, end),
    [high, skip's start) and [skip's end, the image's end), not the zeros between."""
    from opentpu.host.board import device_config
    from opentpu.llm.qwen3 import HIGH, Images
    t = _fake_bcu(monkeypatch)
    cfg = device_config(_info(t), DRAM_BYTES=1 << 33)
    img = np.zeros(HIGH + (3 << 20), np.uint8)
    lo, mx, hi = 4096, HIGH + (1 << 20), HIGH + (2 << 20)
    img[:lo], img[HIGH:mx], img[hi:] = 7, 5, 9
    im = Images([img], lo, HIGH, (mx, hi))
    assert im.spans() == [(0, lo), (HIGH, mx), (hi, len(img))]
    img[mx:hi] = 3                                  # (a backend never writes them)
    be = BoardBackend(cfg, im, transport=t, status=False)
    try:
        assert set(be.board.read(0, lo)) == {7} and set(be.board.read(HIGH, 1 << 20)) == {5}
        assert set(be.board.read(hi, 1 << 20)) == {9} and set(be.board.read(mx, 1 << 20)) == {0}
    finally:
        be.board.close()


def test_an_image_split_at_4gib_has_the_program_area_below(monkeypatch):
    """qwen3.Images with high (the weights from 4 GiB on: AW 36): BoardBackend puts the program
    area right after the low part's end, below 4 GiB, and writes [0, end) and [4 GiB, the
    image's end), not the image's zeros between them."""
    from opentpu.host.board import device_config
    from opentpu.llm.qwen3 import HIGH, Images
    t = _fake_bcu(monkeypatch)
    info = _info(t)
    assert (info["cfg"]["channels"], info["cfg"]["channel_bytes"]) == (8, 1 << 32)
    cfg = device_config(info, DRAM_BYTES=1 << 33)
    assert cfg.AW == 36
    img = np.zeros(HIGH + (1 << 20), np.uint8)
    img[:4096], img[HIGH - 4096:HIGH], img[HIGH:HIGH + 4096] = 7, 5, 9
    be = BoardBackend(cfg, Images([img], 8192, HIGH), transport=t, status=False)
    try:
        assert be.prog_at == 8192 and be.image_bytes == len(img)
        assert set(be.board.read(0, 4096)) == {7} and set(be.board.read(HIGH, 4096)) == {9}
        assert set(be.board.read(HIGH - 4096, 4096)) == {0}            # (not written)
    finally:
        be.close()
    with pytest.raises(MemoryError, match="program area"):        # (it closes its Board)
        BoardBackend(cfg, Images([img], HIGH - 4096, HIGH), transport=t, status=False)


def test_a_card_with_more_dram_than_its_addresses_is_refused(monkeypatch):
    """CFG_DRAM's 32 GiB on a platform this tree has not, on a host whose target addresses 4
    GiB (AW 32): ConfigMismatch naming the target to select, not programs that wrap at 4 GiB."""
    from opentpu.host.board import ConfigMismatch, device_config
    info = _info(_fake_bcu(monkeypatch, platform=0x7E))
    monkeypatch.setenv("OTPU_TARGET", "ypcb-00338")
    with pytest.raises(ConfigMismatch, match="addresses are 32 bits"):
        device_config(info)


def test_the_host_takes_a_cards_address_bits_from_its_hash(monkeypatch):
    """No register says AW; the CFG_HASH holds it above 32 (arch_hash). device_aw: a bcu1525-wg
    build of this tree (its hash at AW 36) is AW 36 on a host of any target (here the
    YPCB-00338's: the host follows the bitstream), one from before AW (its hash at 32, 4 GiB
    of DRAM) AW 32, and its programs compile at that (device_config's AW); a hash of neither is
    refused (check_hash)."""
    from opentpu.host.board import ConfigMismatch, _reg_config, device_aw, device_config
    from opentpu.target import arch_hash
    info = _info(_fake_bcu(monkeypatch))
    monkeypatch.setenv("OTPU_TARGET", "ypcb-00338")
    info["cfg"]["hash"] = arch_hash(_reg_config(info, 36))     # (a build of this tree's)
    assert device_aw(info) == 36 and device_config(info).AW == 36
    old = dict(info, cfg=dict(info["cfg"], DRAM_BYTES=1 << 32))
    old["cfg"]["hash"] = arch_hash(_reg_config(old, 32))
    assert device_aw(old) == 32 and device_config(old).AW == 32
    bad = dict(info, cfg=dict(info["cfg"], hash=info["cfg"]["hash"] ^ 1))
    assert device_aw(bad) == 36
    with pytest.raises(ConfigMismatch, match="CFG_HASH"):
        device_config(bad)


def test_the_4gib_bcuwg_bitstreams_hash_d17ac192_is_aw_32(monkeypatch):
    """f150r62 (tree 5acda1da) and f150cc (bcu-ccb e5948979), the BCU1525's builds from before
    AW (512 MiB channels, WG_SET "SE_R=2 DMA_DEPTH=512"), carry CFG_HASH d17ac192: make_wg.py's
    ARCH_HASH at both trees and at bcu-wg 658dd6cd, with or without the WG_SET. With their
    registers (make_wg.py's otpu_board parameters there) a host of bcu1525-wg (AW 36) or of
    another target takes them at AW 32: check_hash passes and the programs compile at 32."""
    from opentpu.host import fake as F
    from opentpu.host.board import _reg_config, check_hash, device_aw, device_config
    from opentpu.target import arch_hash
    regs = dict(S=1, TMEM_WORDS=131072, IMEM_WORDS=32768, ACT_BLOCKS=128, DRAM_BYTES=1 << 32,
                channels=8, channel_bytes=1 << 29, beat_bytes=128, MXU_IMPL=2, TMEM_IMPL=3,
                VPU_CL=2, ULANES=8, SEQ_WC=1, WG=8, platform=5, hash=0xD17AC192)
    for host in ("bcu1525-wg", "ypcb-00338"):
        monkeypatch.setenv("OTPU_TARGET", host)
        info = _info(F.FakeTransport(ch_bytes=1 << 20, regmap=5, D=128, MCOLS=8, LANES=8,
                                     act_rows=16, dstep=True, stream=True, chash=True,
                                     seq_wc=True, outer=False, dstep2=True, se_own=True,
                                     cfg=regs))
        assert info["cfg"]["hash"] == 0xD17AC192 and info["cfg"]["WG"] == 8
        assert device_aw(info) == 32
        check_hash(info)
        assert device_config(info).AW == 32
        assert arch_hash(_reg_config(info, 36)) != 0xD17AC192


def test_sim_transport_at_caw_holds_the_beats_past_its_image_sparsely(monkeypatch):
    """SimTransport(caw=25) (tb_bcuwg with 4 GiB a channel): a channel's bytes past its dense
    image in 32-byte beats, a partial beat's other bytes kept and an all-zero beat dropped, read
    back across the boundary; handed to the DIMM models as ch<c>_hi.txt ("<beat> <32 bytes,
    byte 0 last>", ascending) and taken back from ch<c>_hi_out.txt; refused past 2^(CAW + 7)
    bytes, and without bcuwg."""
    from pathlib import Path
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="caw 25"):
        SimTransport(ch_bytes=1 << 12, target="bcu1525-wg", caw=25)
    t = SimTransport(ch_bytes=1 << 12, target="bcu1525-wg", bcuwg=True, caw=25)
    r = np.random.default_rng(7)
    a = r.integers(1, 256, 100, dtype=np.uint8)
    t.mem_write(3, 4096 - 40, a)                    # 40 bytes dense, 60 sparse: beats 128, 129
    assert sorted(t.hi[3]) == [128, 129]
    assert np.array_equal(t.mem_read(3, 4096 - 40, 100), a)
    assert not t.mem_read(3, 4096 + 60, 4).any()    # (beat 129's other bytes)
    far = (1 << 32) - 64
    b = np.concatenate([np.zeros(32, np.uint8), r.integers(1, 256, 32, dtype=np.uint8)])
    t.mem_write(7, far, b)                          # (its zero beat stays out)
    assert sorted(t.hi[7]) == [far // 32 + 1]
    t.mem_write(3, 4096 + 2, np.full(3, 0xA5, np.uint8))
    a[42:45] = 0xA5
    assert np.array_equal(t.mem_read(3, 4096 - 40, 100), a)
    t.mem_write(3, 4096 + 32, np.zeros(32, np.uint8))   # (beat 129 all zero: dropped)
    assert sorted(t.hi[3]) == [128]
    with pytest.raises(ValueError, match="past its 2"):
        t.mem_write(0, (1 << 32) - 16, np.ones(32, np.uint8))
    want = {3: f"80 {a[40:72][::-1].tobytes().hex()}\n",
            7: f"{far // 32 + 1:x} {b[32:][::-1].tobytes().hex()}\n"}
    seen = {}

    def run_sim(args):                              # the DIMM models: a beat stored up there
        d = Path(next(x for x in args if x.startswith("+dir="))[5:])
        for c in range(8):
            np.fromfile(d / f"ch{c}.bin", ">u4").astype("<u4").tofile(d / f"ch{c}_out.bin")
            seen[c] = (d / f"ch{c}_hi.txt").read_text()
            (d / f"ch{c}_hi_out.txt").write_text(
                seen[c] + (f"7ffffff {'00' * 31}01\n{1 << 20:x} {'0' * 64}\n" if c == 5 else ""))
        return SimpleNamespace(stdout="REG 0 1\nDONE cycles=9\n", stderr="")
    monkeypatch.setattr(SimTransport, "sim_build", lambda self: ("tb_bcuwg", [], {}))
    monkeypatch.setattr(rtlsim, "build", lambda *a: "exe")
    monkeypatch.setattr(rtlsim, "run_sim", run_sim)
    t.reg_read(0)
    assert seen == {c: want.get(c, "") for c in range(8)}
    assert sorted(t.hi[5]) == [(1 << 27) - 1]       # (its zero beat left out)
    assert np.array_equal(t.mem_read(5, (1 << 32) - 32, 32), [1] + [0] * 31)
    assert np.array_equal(t.mem_read(3, 4096 - 40, 72), a[:72])


def test_sim_transport_at_caw_keeps_tb_bcuwgs_words_an_int(monkeypatch):
    """SimTransport(caw)'s dense image is DENSE bytes a channel at most, the bytes past it in the
    sparse store: bcu1525-s2's expert run at BHW 33 asked for its whole 16 GiB a slice dense (2
    GiB a channel), tb_bcuwg's WORDS (an int) wrapped to 0 and otpu_dimm_model's arrays had 0
    beats. A smaller image is as asked."""
    monkeypatch.setattr(SimTransport, "_wrap", staticmethod(lambda s, p: (s, p)))
    t = SimTransport(ch_bytes=1 << 31, target="bcu1525-s2", bcuwg=True)
    assert t.caw == 24 and len(t.ch[0]) == SimTransport.DENSE == 1 << 29
    assert t.sim_build()[2]["WORDS"] == 1 << 30
    a = np.arange(1, 65, dtype=np.uint8)
    t.mem_write(9, (1 << 29) - 32, a)               # a beat dense, a beat sparse
    assert sorted(t.hi[9]) == [1 << 24]
    assert np.array_equal(t.mem_read(9, (1 << 29) - 32, 64), a)
    t = SimTransport(ch_bytes=1 << 24, target="bcu1525-s2", bcuwg=True)
    assert len(t.ch[0]) == 1 << 24 and t.sim_build()[2]["WORDS"] == 1 << 25


TARGET_FLAGS = ("WPB", "ST16", "ST16S", "QUANT_SPLIT", "QUANT_HL", "QUANT_QSTHL")


def _x2_target(name):
    from dataclasses import replace
    from opentpu import target as T
    base, _, hl = name.partition("+")              # (+QSTHL: B2's QST HL, an arch override)
    t = T.load(base)
    return replace(t, arch=dict(t.arch, QSTHL=True)) if hl else t


@pytest.mark.parametrize("env", [None, "ypcb-00338", "bcu1525-x2", "bcu1525-x2:wmx"])
@pytest.mark.parametrize("name", ["bcu1525-x2:wmx", "bcu1525-x2:wmx+QSTHL", "bcu1525-x2:st16",
                                  "bcu1525-x2"])
def test_x2_board_models_build_the_targets_tmem_writes_and_quantizer(monkeypatch, name, env):
    """An x2 board model's wrapper (OTPU_BOARD: otpu_board's parameters) has TMEM's writes a bank
    (WPB), DSTEP's / the streams' 16-bit state (ST16, ST16S) and the quantizer's flags (QUANT_SPLIT,
    QUANT_HL, QUANT_QSTHL) as its target's card builds them, whatever OTPU_TARGET selects:
    sim_build took them from the environment's board_config (the YPCB by default) and never
    passed WPB, so x2:wmx's board models ran at WPB 1 without QACT's LO / HL (otpu_board's
    defaults; docs/pf500.md)."""
    from dataclasses import replace
    from opentpu import target as T
    from opentpu.host.cards import X2SimTransport
    if env is None:
        monkeypatch.delenv("OTPU_TARGET", raising=False)
    else:
        monkeypatch.setenv("OTPU_TARGET", env)
    t = _x2_target(name)
    sim = X2SimTransport(target=t, ch_bytes=1 << 16)
    _, srcs, _ = sim.sim_build()
    defs = next(s for s in srcs if str(s).rsplit("/", 1)[-1].startswith("otpu_board_")).read_text()
    rp = T.rtl_params(replace(t, arch=dict(t.arch, S=sim.spc)))
    want = {k: rp[k] for k in TARGET_FLAGS if rp.get(k) and not (k == "WPB" and rp[k] == 1)}
    assert {k: v for k, v in re.findall(r"\.(\w+)\((\d+)\)", defs)
            if k in TARGET_FLAGS and v != "0"} == {k: str(v) for k, v in want.items()}
    if name.startswith("bcu1525-x2:wmx"):          # (x2:wmx: the pumped TMEM's second write)
        assert want["WPB"] == 2 and {"QUANT_SPLIT", "QUANT_HL", "ST16S"} <= set(want)
        assert ("QUANT_QSTHL" in want) == name.endswith("+QSTHL")
    else:
        assert set(want) <= {"ST16"}


def test_a_board_model_takes_no_flags_from_otpu_target(monkeypatch):
    """A one-slice board model of the YPCB builds none of x2:wmx's flags when OTPU_TARGET selects
    x2:wmx (the environment's board_config is the bitstream's sizes only, not its options)."""
    monkeypatch.setenv("OTPU_TARGET", "bcu1525-x2:wmx")
    p = SimTransport(ch_bytes=1 << 16, target="ypcb-00338").sim_build()[2]
    assert not set(TARGET_FLAGS) & set(p)


def test_a_bcuwg_board_model_is_the_targets_build_at_its_caw(monkeypatch):
    """tb_bcuwg's channels are the target's by default (bcu1525-wg: CAW 25, 4 GiB a channel) and
    the model's CFG_HASH is the target's build's at the DRAM its channels give (CFG_DRAM). On
    a merged tree the tiny models' board-model runs at tb_bcuwg's CAW 22 (512 MiB a
    channel) were refused: CFG_HASH 4b35ec47 (the target's 32 GiB), the registers' 58863819."""
    from dataclasses import replace
    from opentpu import target as T
    monkeypatch.setattr(SimTransport, "_wrap", staticmethod(lambda s, p: (s, p)))
    t = T.load("bcu1525-wg")
    rp = T.rtl_params(t)
    p = {k: rp[k] for k in ("MCOLS", "LANES", "TMEM_WORDS", "SEQ_WC", "OUTER", "DSTEP2")}
    p |= {"WG": t.arch["WG"], "ACT_ROWS": t.arch["ACT_ROWS"]}
    build = T.build_config(t, **p, DSTEP=1, CHASH=1, SE_R=t.uarch.get("SE_R", 0))
    seen, ic = {}, set()
    for caw in (None, 22, 25, 26):
        tr = SimTransport(ch_bytes=1 << 12, target=t, bcuwg=True, caw=caw, params=p)
        q = tr.sim_build()[2]
        seen[caw] = tr.caw, q["CAW"], q["ARCH_HASH"]
        ic.add(tr.icache)
    build = replace(build, ICACHE=ic.pop())     # (the instruction cache where it builds one: OTPU_ICACHE)
    h = {c: "32'h%08x" % T.arch_hash(replace(build, DRAM_BYTES=8 << c + 7)) for c in (22, 25, 26)}
    assert seen == {None: (25, 25, h[25]), 22: (22, 22, h[22]), 25: (25, 25, h[25]),
                    26: (26, 26, h[26])}
    assert build.DRAM_BYTES == 8 << 32 and len(set(h.values())) == 3


def test_run_clock_is_there_while_the_host_serves_a_run():
    """BoardBackend.run_clock, the expert server's halt-aware hold (docs/offload.md 13.12): the
    host hook polled during a run sees the run's start and the shortest time of its program
    length's earlier runs (None on the first). It returned None at every poll on the card in
    pfv2 (holds 0): wait() cleared the running programs before it served the run."""
    from opentpu import isa as I
    from opentpu.host.fake import FakeTransport
    t = FakeTransport(ch_bytes=1 << 21, run_s=0.01, cycles=1_000_000, core_khz=100_000)
    be = BoardBackend(sim_board_config(DRAM_BYTES=1 << 22), [np.zeros(4096, np.uint8)],
                      transport=t)
    seen = []
    be.host = lambda: (seen.append(be.run_clock()), 0)[1]
    prog = [[I.Instr(I.HALT)]]
    for run in range(3):
        seen.clear()
        be.start(prog)
        be.wait()
        assert seen and be.run_clock() is None      # (after the run: none)
        if run == 0:
            assert set(seen) == {None}
        else:                                       # (CYCLES / CORE_KHZ, times the ratio)
            assert all(c is not None and 0.009 < c[1] < 0.011 for c in seen)



@pytest.mark.parametrize("name", ["bcu1525-x2", "bcu1525-s2"])
def test_the_generic_board_models_hash_is_its_registers(name):
    """tb_board built for a target of 16 GiB slices behind the bank-half window (SimTransport
    without bcuwg: the projection of the slices on one board) reports otpu_board's 4 GiB
    (DRAM_LOG2 32) and no window (the DIMM pairs', tb_bcuwg's), so its CFG_HASH is that
    configuration's: the target's own DRAM and BHW in it made check_hash refuse every such
    model. Parameters only, nothing simulated."""
    from dataclasses import replace
    from opentpu import target as T
    t = T.load(name)
    assert (t.arch["BHW"], t.arch["DRAM_BYTES"]) == (33, 16 << 30)
    p = SimTransport(ch_bytes=1 << 19, target=name, params={"MCOLS": 8, "WG": 8}).sim_build()[2]
    assert "DRAM_LOG2" not in p and "BHW" not in p and p["AW"] == 36
    bc = T.build_config(t, **{k: p.get(k, d) for k, d in (
        ("MCOLS", None), ("LANES", None), ("ACT_ROWS", None), ("TMEM_WORDS", None),
        ("IMEM_WORDS", None), ("DSTEP", 1), ("SEQ_WC", 0), ("WG", 1), ("CHASH", 1), ("OUTER", 1),
        ("DSTEP2", 0), ("SE_R", 0), ("KT", 0))})
    assert int(p["ARCH_HASH"][4:], 16) == T.arch_hash(replace(bc, BHW=0, DRAM_BYTES=1 << 32))
    assert int(p["ARCH_HASH"][4:], 16) != T.arch_hash(bc)


# ------------------------------------------------------------------------------ the WG targets' channels
# The WG targets' board (targets/bcu1525-x2.toml: otpu_board S = 4, each slice on 8 channels of 128-byte
# beats through otpu_native_dram NCH 8, CB 128, the native memory model with the target's
# channel rate and latency: SimTransport). The arch is the session's board configuration at S 4
# (WG 1: weight groups are the MXU's).
def _wg_transport(dram_bytes: int, seed: int = 5, stall: int = 20) -> SimTransport:
    return SimTransport(ch_bytes=dram_bytes // 8, stall=stall, seed=seed, target="bcu1525-x2")


def test_wg_map_round_trip_on_board_model(have_verilator):
    """Host and RTL agree on the WG targets' channel map byte for byte: the host writes random ranges
    (unaligned starts and lengths) into every slice's DRAM through its map (the channels'
    physical images), each slice's program copies some of them with LD / ST through the RTL's
    map (otpu_native_dram), and the host reads the copies and the rest back through its map; the
    channel images hold each byte where the map (I1) puts it."""
    from opentpu import isa as I
    from opentpu.isasim import Machine
    dram = 1 << 22
    b = Board(_wg_transport(dram))
    assert (b.slices, b.nch, b.beat, b.chash) == (4, 8, 128, True)
    rng = np.random.default_rng(11)
    ref = [np.zeros(dram, np.uint8) for _ in range(4)]
    for s in range(4):
        for _ in range(40):
            n = int(rng.integers(1, 9000))
            a = int(rng.integers(0, dram // 2 - n))
            d = rng.integers(0, 256, n).astype(np.uint8)
            b.write(a, d, s)
            ref[s][a:a + n] = d
    progs, copies = [], []
    for s in range(4):
        cp, prog = [], []
        for k in range(12):                          # word-aligned, any length of words
            n = int(rng.integers(1, 1500))
            src = 4 * int(rng.integers(0, (dram // 2 - 4 * n) // 4))
            dst = dram // 2 + 4 * int(rng.integers(0, (dram // 4 - 4 * n) // 4))
            t = 2048 * k
            prog += [I.ld(src, t, n), I.st(dst, t, n)]
            cp.append((src, dst, n))
        progs.append(prog + [I.halt()])
        copies.append(cp)
    m = Machine(sim_board_config(DRAM_BYTES=dram, S=4), progs, [r.copy() for r in ref]).run()
    b.load(dram - (1 << 16), progs)
    st = b.run()
    assert st["instructions"] == [sl.icount for sl in m.slices]
    for s in range(4):
        for src, dst, n in copies[s]:
            ref[s][dst:dst + 4 * n] = ref[s][src:src + 4 * n]
        assert np.array_equal(m.slices[s].dram[:dram // 2 + dram // 4], ref[s][:3 * dram // 4])
    for s in range(4):
        for src, dst, n in copies[s]:
            assert np.array_equal(b.read(dst, 4 * n, s), ref[s][dst:dst + 4 * n]), (s, hex(dst))
        assert np.array_equal(b.read(0, dram // 2, s), ref[s][:dram // 2]), s
        for x in rng.integers(0, dram - (1 << 16), 400):
            c, off = _ref_channel(int(x), 8, 128, True)
            assert b.t.ch[8 * s + c][off] == ref[s][int(x)], (s, int(x))
    # the copies' and the program's beats went over every channel of every slice
    assert all(c.any() for c in b.t.ch)


def _tiny_qwen3_s4():
    """A tiny Qwen3 that maps onto 4 slices (hidden 512 = S x D, 4 KV heads, 2 per query
    group)."""
    from opentpu.llm.qwen3 import Spec
    torch.manual_seed(0)
    hc = transformers.Qwen3Config(hidden_size=512, num_hidden_layers=2, num_attention_heads=8,
                                  num_key_value_heads=4, head_dim=128, intermediate_size=1024,
                                  vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
                                  tie_word_embeddings=True, max_position_embeddings=4096)
    mdl = transformers.Qwen3ForCausalLM(hc).float().eval()
    return {k: v.float().numpy() for k, v in mdl.state_dict().items()}, \
        Spec(512, 2, 8, 4, 128, 1024, 1000)


def _tiny_qwen35_s4():
    """A tiny Qwen3.5 that maps onto 4 slices: DeltaNet and attention layers, 8 DeltaNet heads
    (a pair a slice), 4 KV heads, hidden 512."""
    from opentpu.llm.qwen35 import Spec
    kinds = ("linear", "linear", "attn")
    torch.manual_seed(0)
    hc = transformers.Qwen3_5TextConfig(
        hidden_size=512, num_hidden_layers=len(kinds), num_attention_heads=8,
        num_key_value_heads=4, head_dim=256, intermediate_size=1024, vocab_size=1000,
        layer_types=["full_attention" if k == "attn" else "linear_attention" for k in kinds],
        linear_num_key_heads=8, linear_num_value_heads=8, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=True,
        max_position_embeddings=4096, rms_norm_eps=1e-6, initializer_range=0.2,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                         "partial_rotary_factor": 0.25})
    m = transformers.Qwen3_5ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:     # zero-centered norms, but the DeltaNet output norm is plain
                p.copy_((1.0 if n.endswith("linear_attn.norm.weight") else 0.0)
                        + 0.1 * torch.randn_like(p))
    return {k: v.float().numpy() for k, v in m.state_dict().items()}, \
        Spec(512, kinds, 8, 4, 256, 64, 8, 128, 128, 1024, 1000)


@pytest.mark.parametrize("model", ["qwen3", "qwen35"])
def test_tiny_models_four_slices_wg_on_board_model(have_verilator, model):
    """A tiny Qwen3 and a tiny Qwen3.5 split over bcu1525-x2's 4 slices, each on 8 channels of
    128-byte beats (NCH 8, CB 128, WG 1): logits bit-identical to the ISA simulator at S = 4,
    every slice's instruction count the simulator's (Qwen3.5 through a turn of its convolution
    window), and each slice's DRAM, read back through the host's channel map, the
    simulator's."""
    from opentpu.llm.qwen3 import Engine
    W, spec = _tiny_qwen3_s4() if model == "qwen3" else _tiny_qwen35_s4()
    need = spec.image(sim_board_config(DRAM_BYTES=1 << 40, S=4), 256).nbytes
    cfg = sim_board_config(DRAM_BYTES=1 << max(22, (need - 1).bit_length() + 1), S=4)
    isa = Engine(spec, W, cap=256, cfg=cfg)
    tr = _wg_transport(cfg.DRAM_BYTES)
    brd = Engine(spec, W, cap=256, cfg=cfg,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    c = brd.backend.info["cfg"]
    from opentpu import target as T
    assert (c["S"], c["platform"], c["channels"], c["beat_bytes"]) == \
        (4, T.load("bcu1525-x2").platform_id, 8, 128)
    for tok in (11, 222) if model == "qwen3" else (11, 222, 333, 444, 555):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
        st = brd.stats[-1]
        assert st["cycles"] > 0 and st["instructions"] == isa.stats[-1]["instructions"]
    n = isa.image.nbytes
    for s in range(4):
        assert np.array_equal(brd.backend.read(s, 0, n), isa.backend.machine.slices[s].dram[:n])
