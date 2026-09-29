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
    """Logical beat b lives on channel b % 2 at offset (b // 2) * 64 (otpu_axi_dram.sv)."""
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
    offset m * 64 (otpu_axi_dram.sv), so a column at a power-of-two chunk stride is spread
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
    """The same on the board model's native memory path (MEM_NATIVE: otpu_native_dram in front of
    the native memory model), whatever OTPU_NATIVE says; and the partial writes."""
    b = Board(SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=30, seed=seed, native=True))
    ok, msg, st = run_demo(b, CFG)
    assert ok, msg
    assert st["b_reads"] > 0 and st["a_writes"] > 0 and st["cycles"] > 0
    ok, msg, st = run_demo(b, CFG, masked_program())
    assert ok, msg
    assert st["a_writes"] > 500 and st["b_writes"] > 20


@pytest.mark.parametrize("ctrl", ["mig", "ld"])
def test_program_on_channel_board_model(have_verilator, ctrl):
    """The same on the native-channel builds' channels (MEM_NATIVE=3 / 2: otpu_native_dram, then
    per channel otpu_mem_ch in front of a model of the controller in its own clock: the MIG's
    native interface through otpu_mig_native, partial beats as wr_bytes; or LiteDRAM's native
    port, partial beats read-modified-written in otpu_mem_ch), whatever OTPU_NATIVE says."""
    b = Board(SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=30, seed=3, native=ctrl))
    ok, msg, st = run_demo(b, CFG)
    assert ok, msg
    assert st["b_reads"] > 0 and st["a_writes"] > 0 and st["cycles"] > 0
    ok, msg, st = run_demo(b, CFG, masked_program())
    assert ok, msg
    assert st["a_writes"] > 500 and st["b_writes"] > 20


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


@pytest.mark.parametrize("group,name,prog", op_checks(CFG),
                         ids=[name for _, name, _ in op_checks(CFG)])
def test_op_check_on_board_model(have_verilator, group, name, prog):
    """otpu-diag's per-instruction programs (opentpu/host/opchecks.py), bit for bit."""
    ok, msg, _ = run_demo(Board(SimTransport(ch_bytes=CFG.DRAM_BYTES // 2)), CFG, prog,
                          diag_image())
    assert ok, msg


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


# ------------------------------------------------------------------------------ Qwen3
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_tiny_qwen3_on_board_model(have_verilator):
    from opentpu.llm.qwen3 import Engine, Spec
    torch.manual_seed(0)
    hc = transformers.Qwen3Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=128, intermediate_size=512,
                                  vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
                                  tie_word_embeddings=True, max_position_embeddings=4096)
    mdl = transformers.Qwen3ForCausalLM(hc).float().eval()
    W = {k: v.float().numpy() for k, v in mdl.state_dict().items()}
    spec = Spec(256, 2, 4, 2, 128, 512, 1000)
    cfg = board_config(DRAM_BYTES=1 << 23)
    isa = Engine(spec, W, cap=256, cfg=cfg)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5)
    brd = Engine(spec, W, cap=256, cfg=cfg,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    for tok in (11, 222, 333):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert brd.stats[-1]["cycles"] > 0
