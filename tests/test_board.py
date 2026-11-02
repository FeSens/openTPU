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
    from opentpu.target import current
    # (DUAL: four channels, CALIB0 channels 0 and 2)
    t = SimTransport(ch_bytes=1 << 20,
                     plusargs=["+calib=" + ("5" if current().arch.get("DUAL") else "1")])
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


@pytest.mark.parametrize("nreg,seed", [(0b1010, 5), (0, 9)])
def test_bar_and_gather_across_slrs_on_board_model(have_verilator, nreg, seed):
    """The BCU1525's board (S = 2, the board's SLR crossings, the collective through
    otpu_coll_hub and a port per slice; nreg: the channels' crossings, as the board's 1 and 3,
    or none) runs BARs back to back, a BAR with one slice far behind the other (each in turn),
    BARs right after and right before GATHERs and GATHERs back to back, under two memory stall
    patterns (seed): the results and both slices' instruction counts are the ISA simulator's.
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
                           params={"NREG": nreg}))
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


def test_collective_reads_held_behind_dma_reads_on_board_model(have_verilator):
    """test_rtl's held collective on the BCU1525's board (S = 2: the collective through
    otpu_coll_hub and a port per slice) with release r2's TMEM (the collective the DMA's guest):
    each port asks its held read from its second cycle (TMEM's input stage, otpu_coll_port hq)
    and holds it while the guest rule blocks it (an ST beside the GATHERs); the results and the
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
                           params=dict(R2_TMEM)))
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

