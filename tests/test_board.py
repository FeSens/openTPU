"""The host driver (opentpu/host/board.py) and the board model (sim/verilator/tb_board.sv).

The address-map tests are pure Python. The others drive the Verilator model of the board --
control registers, program loader, slice, DRAM adapter, two-channel AXI memory -- through the
same register and memory protocol the PCIe driver uses, and require the ISA simulator's
results bit for bit.
"""
import numpy as np
import pytest

from opentpu.host import regs as R
from opentpu.host.board import BASE, BEAT, Board, BoardBackend, SimTransport, join, split
from opentpu.host.checks import (address_lines, channel_patterns, masked_program, partial_writes,
                                 pattern_test, run_demo, vops_program)
from opentpu.host.opchecks import diag_image, op_checks
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


# ------------------------------------------------------------------------------ board model
CFG = board_config(DRAM_BYTES=1 << 22)


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


@pytest.mark.parametrize("group,name,prog", op_checks(CFG, gen=True, waitw=True),
                         ids=[name for _, name, _ in op_checks(CFG, gen=True, waitw=True)])
def test_op_check_on_board_model(have_verilator, group, name, prog):
    """otpu-diag's per-instruction programs (opentpu/host/opchecks.py), bit for bit."""
    ok, msg, _ = run_demo(Board(SimTransport(ch_bytes=CFG.DRAM_BYTES // 2)), CFG, prog,
                          diag_image())
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
@pytest.mark.parametrize("after", [600, 2500])
def test_run_dropped_then_load_and_run_at_once(have_verilator, native, after):
    """CTRL = 0 in the middle of a run that streams reads and writes, then at once (without the
    wait for WR_IDLE, docs/observability.md "Stopping a run") a LOAD of another program and its
    RUN: the loader takes none of the dropped run's read data still in flight (it starts once the
    memory adapter's reads are all back), the run starts once its writes are answered, and the
    second program's results and instruction count are the ISA simulator's. Before, the loader
    wrote the dropped run's late read data into IMEM as the new program."""
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
    cfg = board_config(DRAM_BYTES=1 << 23)
    isa = Engine(spec, W, cap=256, cfg=cfg)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5)
    brd = Engine(spec, W, cap=256, cfg=cfg,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    for tok in (11, 222, 333):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert brd.stats[-1]["cycles"] > 0


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
    need = spec.image(board_config(DRAM_BYTES=1 << 40), 512, rows=PREFILL_ROWS,
                      lookup=True).nbytes
    size = 1 << (need - 1).bit_length()
    cfg = board_config(DRAM_BYTES=size)
    eng = Engine(spec, W, cap=512, cfg=cfg, resident=True)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 248)]
    t0 = int(np.argmax(eng.prefill(toks)))
    isa, pos, n = eng.backend, eng.pos, eng.image.nbytes
    tr = SimTransport(ch_bytes=size // 2, stall=20, seed=5)
    brd = BoardBackend(cfg, [isa.machine.slices[0].dram[:n]], transport=tr)   # the prefill's
    assert brd.generates and brd.chains
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
    be = BoardBackend(board_config(DRAM_BYTES=1 << 22), [np.zeros(4096, np.uint8)],
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

