"""ISA encoding and simulator semantics (directed tests)."""
import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I
from opentpu.isasim import Machine, SimError


def f(x):
    return np.asarray(x, np.float32)


def run1(prog, dram=None, cfg=Config(S=1)):
    return Machine(cfg, [prog], [dram]).run().slices[0]


def test_encode_decode_roundtrip():
    ins = I.mm(0x100, 0x200, 7, 33, 4, 128, 64, 5, 3, 16, unit=True, acc=True, ra=1, rb=2, rc=3)
    words = ins.encode()
    back = I.Instr.decode(words)
    assert back.encode() == words
    assert back.op == I.MM and back.flags == I.F_UNIT | I.F_ACC and (back.ra, back.rb, back.rc) == (1, 2, 3)


def test_nested_loops_and_registers():
    # sum 1.0 into T[0] 3*4 times via nested loops, and walk an address register
    prog = [
        I.vop(I.V_FILL, 0, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 0.0),
        I.li(1, 100),
        I.loop(5, 3),                       # outer: 3 iterations, body = next 5 instrs
        I.loop(2, 4),                       #   inner: 4 iterations
        I.vop(I.V_ADD, 0, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 1.0),
        I.addi(1, 1, 1),                    #   R1 += 1 per inner iteration
        I.addi(2, 2, 10),                   # R2 += 10 per outer iteration
        I.nop(),
        I.vop(I.V_FILL, 1, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 5.0, rb=0),
        I.halt(),
    ]
    s = run1(prog)
    assert s.tget([0])[0] == 12.0
    assert s.R[1] == 112 and s.R[2] == 30
    assert s.tget([1])[0] == 5.0


def test_loop_zero_count_skips_body():
    prog = [I.loop(1, 0), I.vop(I.V_FILL, 0, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 9.0), I.halt()]
    assert run1(prog).tget([0])[0] == 0.0


@pytest.mark.parametrize("count", [0, 1])
def test_nested_bodies_ending_together_are_refused_whatever_the_count(count):
    """docs/isa.md LOOP: a body must not end on an enclosing body's last instruction. At
    count 0 the inner LOOP would jump past the outer body's end and skip its back edge."""
    prog = [I.li(1, count), I.loop(3, 2), I.vop(I.V_ADD, 0, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 1.0),
            I.loop(1, 0, rcount=1), I.nop(), I.halt()]
    with pytest.raises(SimError, match="same instruction"):
        run1(prog)


def test_vop_broadcast_modes_and_reductions():
    dram = np.zeros(4096, np.uint8)
    a = f(np.arange(12).reshape(3, 4) - 5)
    dram[:48] = a.view(np.uint8).reshape(-1)
    dram[64:76] = f([10, 20, 30]).view(np.uint8)          # per-row operand
    dram[128:144] = f([1, 2, 3, 4]).view(np.uint8)        # per-column operand
    prog = [I.ld(0, 0, 12), I.ld(64, 16, 3), I.ld(128, 24, 4),
            I.vop(I.V_ADD, 32, 0, 16, 3, 4, 4, 4, 1, I.B_ROW),
            I.vop(I.V_MUL, 48, 0, 24, 3, 4, 4, 4, 0, I.B_COL),
            I.vop(I.V_RSUM, 64, 0, 0, 3, 4, 1, 4, 0),
            I.vop(I.V_RMAX, 68, 0, 0, 3, 4, 1, 4, 0),
            I.halt()]
    s = run1(prog, dram)
    assert np.array_equal(s.tget(np.arange(32, 44)).reshape(3, 4), a + f([10, 20, 30])[:, None])
    assert np.array_equal(s.tget(np.arange(48, 60)).reshape(3, 4), a * f([1, 2, 3, 4])[None, :])
    assert np.array_equal(s.tget(np.arange(64, 67)), a.sum(1))
    assert np.array_equal(s.tget(np.arange(68, 71)), a.max(1))


def test_vop_hazard_is_detected():
    prog = [I.vop(I.V_ADD, 1, 0, 0, 1, 4, 0, 0, 0, I.B_SCALAR, 1.0), I.halt()]
    with pytest.raises(SimError, match="hazard"):
        run1(prog)


def test_mm_matches_integer_math():
    """MM, docs/isa.md: one exact integer per output, each group's products shifted by
    W - dw - da (the groups past the window left out), then (i2f(acc) Sw) (Sa 2^-W), bit for
    bit (int8: W 16, a dw byte per block, QACT's groups of D/4)."""
    cfg = Config(S=1, D=32)
    rng = np.random.default_rng(0)
    x = f(rng.standard_normal((3, 64)) * np.repeat(np.exp2(rng.integers(-14, 1, (3, 8))), 8, 1))
    w = rng.integers(-127, 128, (5, 64)).astype(np.int8)
    sw = f(rng.uniform(0.01, 0.1, 5))
    dw = rng.integers(0, 12, (5, 2)).astype(np.uint8)
    srows = np.zeros((5, 8), np.uint8)                       # Sw, then a dw byte per block
    srows[:, :4] = sw.view(np.uint8).reshape(5, 4)
    srows[:, 4:6] = dw
    dram = np.zeros(8192, np.uint8)
    dram[:768] = x.view(np.uint8).reshape(-1)
    dram[1024:1024 + 320] = w.view(np.uint8).reshape(-1)
    dram[2048:2048 + 40] = srows.reshape(-1)
    prog = [I.ld(0, 0, 192), I.qact(0, 3, 0, 2, 64), I.mm(1024, 2048, 256, 5, 2, 64, 5, 3, 0, 8),
            I.halt()]
    s = run1(prog, dram, cfg)
    y = s.tget(256 + np.arange(15)).reshape(3, 5)
    q, sa, da = F.quantize_mx(x, 8)
    acc = np.zeros((3, 5), np.int64)
    dropped = 0
    for k in range(2):
        for b in range(4):
            g = slice(32 * k + 8 * b, 32 * k + 8 * b + 8)
            c = q[:, g].astype(np.int64) @ w[:, g].astype(np.int64).T          # [3, 5]
            sh = 16 - dw[None, :, k].astype(np.int64) - da[:, 4 * k + b, None].astype(np.int64)
            acc += np.where(sh >= 0, c << np.maximum(sh, 0), 0)
            dropped += int((sh < 0).sum())
    want = F.mul(F.mul(acc.astype(np.float64).astype(np.float32), sw[None, :]),
                 F.mul(sa[:, None], f(2.0 ** -16)))
    assert dropped and np.array_equal(y, want)


def test_gather_two_slices():
    cfg = Config(S=2)
    progs = []
    for s in range(2):
        progs.append([I.vop(I.V_FILL, 0, 0, 0, 2, 3, 3, 0, 0, I.B_SCALAR, float(s + 1)),
                      I.gather(0, 100, 2, 3, 3, 6, 3), I.halt()])
    m = Machine(cfg, progs, [None, None]).run()
    for sl in m.slices:
        got = sl.tget(100 + np.arange(12)).reshape(2, 6)
        assert np.array_equal(got, np.array([[1, 1, 1, 2, 2, 2]] * 2, np.float32))


def test_field_widths_raise_value_errors():
    """A field that does not fit its bits is a ValueError (an assert would let python -O
    truncate the word)."""
    for make in (lambda: I.vop(I.V_ADD, 0, 0, 0, 1, 1 << 16, 0, 0, 0),
                 lambda: I.vop(I.V_ADD, 0, 0, 0, 1, 4, 1 << 16, 0, 0),
                 lambda: I.mm(0, 0, 0, 0, 1, 32, 1, 1, 0, 4),
                 lambda: I.qact(0, 256, 0, 1, 32),
                 lambda: I.qst(0, 0, 0, 1, 0, 32, 32, 1),
                 lambda: I.gather(0, 0, 1 << 16, 1, 1, 1, 1),
                 lambda: I.dstep(0, 0, 0, 1 << 16, 64, 0, 1, 0),
                 lambda: I.Instr(I.NOP, ra=16).encode(),
                 lambda: I.Instr(I.NOP, flags=256).encode()):
        with pytest.raises(ValueError):
            make()


def test_immediates_at_aw():
    """docs/isa.md "Address bits": LI / ADDI's {w2, w1}: w1 alone at AW 32 (the words as
    before), the high word in w2 above it; any other immediate one 32-bit word, never
    truncated."""
    assert I.li(1, 0x1234).w == [0x1234, 0, 0, 0, 0, 0, 0]
    assert I.addi(1, 2, -0x40).w == [0xFFFFFFC0, 0, 0, 0, 0, 0, 0]
    assert I.li(1, 0x7_0000_1000, aw=36).w[:2] == [0x1000, 7]
    assert I.addi(1, 2, -0x40, aw=36).w[:2] == [0xFFFFFFC0, 0xFFFFFFFF]
    assert I.addi(1, 2, 0x40, aw=36).w[:2] == [0x40, 0]
    assert I.imm64(-1, 36) == [0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, 0, 0]
    for make in (lambda: I.li(1, 1 << 32),                    # 33 bits at AW 32
                 lambda: I.li(1, 1 << 36, aw=36),             # past AW
                 lambda: I.ld(1 << 32, 0, 1)):                # an offset of 33 bits
        with pytest.raises(ValueError):
            make()


def test_rld_mul_by_a_negative_constant_at_aw():
    """isa.rld_mul: at AW 32 one RLD MUL (the product's low 32 bits); above it the word is
    zero-extended, so a negative multiplier is LI'd into R[rd] first and the RLD multiplies by
    R[rd] + 0."""
    one = I.rld_mul(3, 0x40, -1, ra=1)
    assert len(one) == 1 and one[0].w[1] == 0xFFFFFFFF and one[0].flags & I.F_MUL
    two = I.rld_mul(3, 0x40, -1, ra=1, aw=36)
    assert [x.op for x in two] == [I.LI, I.RLD] and two[0].w[:2] == [0xFFFFFFFF, 0xFFFFFFFF]
    assert (two[1].rb, two[1].ra, two[1].w[1]) == (3, 1, 0)
    assert len(I.rld_mul(3, 0x40, 9344, aw=36)) == 1
    with pytest.raises(ValueError):
        I.rld_mul(3, 0x40, -1, ra=3, aw=36)


def test_addresses_above_4gib():
    """docs/isa.md "Address bits" in the simulator at AW 36: LI / ADDI's {w2, w1}, a DRAM address
    R[x] + w zero-extended (no wrap at 4 GiB), RLD MUL's AW-bit product, the TMEM index the low
    32 bits; DRAM a sparse file (only the pages written)."""
    cfg = Config(S=1, AW=36, DRAM_BYTES=1 << 34, TMEM_WORDS=1 << 10)
    hi = 0x3_0000_0040
    prog = [I.vop(I.V_FILL, 0, 0, 0, 1, 4, 0, 0, 0, I.B_SCALAR, 2.5),
            I.li(1, hi, aw=36),
            I.st(0, 0, 4, ra=1),                          # T[0..3] -> DRAM 0x3_0000_0040
            I.addi(2, 1, -0x40, aw=36),                   # 0x3_0000_0000
            I.ld(0x44, 8, 2, ra=2),                       # its words 1, 2 -> T[8..9]
            I.li(3, 0x1_0000_0000, aw=36),
            I.ld(0xFFFFFFF0, 16, 1, ra=3),               # 0x1_FFFF_FFF0: no wrap
            I.vop(I.V_FILL, 32, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 3.0),
            I.rld(4, 32, mul=0, rb=3),                    # 3 x 2^32
            *I.rld_mul(5, 32, -1, aw=36),                 # 3 x -1 at AW: 2^36 - 3
            I.addi(6, 5, 4, aw=36),                       # 1
            I.Instr(I.ADDI, rd=7, ra=0, w=[5, 9, 0, 0, 0, 0, 0]),
            I.halt()]
    dram = np.zeros(1 << 20, np.uint8)
    dram[0x1_0000:0x1_0004] = 0xAB                        # an image's word below 4 GiB
    s = Machine(cfg, [prog], [dram]).run().slices[0]
    assert isinstance(s.dram, np.memmap) and s.dram.shape == (1 << 34,)
    assert s.m32[hi // 4:hi // 4 + 4].view(np.float32).tolist() == [2.5] * 4
    assert s.tget([8, 9]).tolist() == [2.5, 2.5]
    assert s.R[1:8] == [hi, 0x3_0000_0000, 1 << 32, 3 << 32, (1 << 36) - 3, 1, 9 << 32 | 5]
    assert s.dram[0x1_0000] == 0xAB and s.dram[0x1_0004] == 0
    # at AW 32 the same words wrap at 4 GiB, and ADDI ignores w2
    s32 = run1([I.li(1, 0xFFFF_FFF0), I.addi(2, 1, 0x20),
                I.Instr(I.ADDI, rd=3, ra=0, w=[5, 9, 0, 0, 0, 0, 0]), I.halt()])
    assert s32.R[1:4] == [0xFFFF_FFF0, 0x10, 5]


def test_halt_chain_below_4gib():
    """HALT CHAIN's program is read below 4 GiB (the board's chain loader, docs/isa.md)."""
    cfg = Config(S=1, AW=36, DRAM_BYTES=1 << 34)
    prog = [I.li(1, 1 << 32, aw=36), I.li(2, 1), I.Instr(I.HALT, ra=1, rb=2, flags=I.F_CHAIN)]
    with pytest.raises(SimError, match="HALT CHAIN"):
        Machine(cfg, [prog], [None]).run()


def test_qst_data_and_scale_overlap_is_refused():
    """docs/isa.md QST: the data and scale ranges of one QST must not overlap."""
    cfg = Config(S=1, D=32)
    fill = I.vop(I.V_FILL, 0, 0, 0, 1, 32, 0, 0, 0, I.B_SCALAR, 3.0)
    _, sc = F.quantize(np.full((1, 32), 3.0, np.float32))
    s = run1([fill, I.qst(0, 256, 512, 1, 1, 32, 32, 1), I.halt()], cfg=cfg)
    assert (s.dram[256:288] == 127).all() and s.m32[128] == sc.view(np.uint32)[0]
    # a byte column (es 64, V^T's) with its scale word between two of its bytes
    s = run1([fill, I.qst(0, 256, 256 + 5 * 64 + 4, 1, 1, 32, 32, 64, row=True), I.halt()],
             cfg=cfg)
    assert (s.dram[256:256 + 32 * 64:64] == 127).all()
    with pytest.raises(SimError, match="overlap"):
        run1([fill, I.qst(0, 256, 280, 1, 1, 32, 32, 1), I.halt()], cfg=cfg)
    with pytest.raises(SimError, match="overlap"):      # on one of the column's bytes
        run1([fill, I.qst(0, 256, 256 + 5 * 64, 1, 1, 32, 32, 64, row=True), I.halt()],
             cfg=cfg)


def test_gather_overlap_and_the_resolved_destination():
    """GATHER: the slices agree on dst = R[rb] + w2 (not on w2), and a slice's source must not
    overlap the destination (docs/isa.md)."""
    cfg = Config(S=2)

    def run(dst, rb_val, src=0):
        progs = [[I.li(1, rb_val[s]), I.vop(I.V_FILL, 0, 0, 0, 2, 3, 3, 0, 0, I.B_SCALAR,
                                            float(s + 1)),
                  I.gather(src, dst[s], 2, 3, 3, 6, 3, rb=1), I.halt()] for s in range(2)]
        return Machine(cfg, progs, [None, None]).run()

    m = run([100, 90], [0, 10])                          # R1 + w2 = 100 on both
    for sl in m.slices:
        assert np.array_equal(sl.tget(100 + np.arange(12)).reshape(2, 6),
                              np.array([[1, 1, 1, 2, 2, 2]] * 2, np.float32))
    with pytest.raises(SimError, match="disagree"):
        run([100, 100], [0, 10])                         # the same w2, different addresses
    with pytest.raises(SimError, match="overlap"):
        run([4, 4], [0, 0])                              # [4, 16) over the source [0, 6)


def test_gather_slices_must_name_the_same_resolved_dst():
    # dst is R[rb] + w2: equal immediates through different registers are different addresses
    # (the collective unit writes every slice's TMEM at slice 0's); equal addresses through
    # different registers are the same GATHER
    cfg = Config(S=2)
    fill = I.vop(I.V_FILL, 0, 0, 0, 1, 4, 4, 0, 0, I.B_SCALAR, 1.0)
    progs = [[fill, I.gather(0, 100, 1, 4, 4, 4, 4), I.halt()],
             [I.li(1, 8), fill, I.gather(0, 100, 1, 4, 4, 4, 4, rb=1), I.halt()]]
    with pytest.raises(SimError, match="disagree"):
        Machine(cfg, progs, [None, None]).run()
    progs[1][2] = I.gather(0, 92, 1, 4, 4, 4, 4, rb=1)
    m = Machine(cfg, progs, [None, None]).run()
    for sl in m.slices:
        assert np.array_equal(sl.tget(100 + np.arange(8)), np.ones(8, np.float32))


def test_mm_needs_n_and_kb():
    # N = 0 or KB = 0 is not an MM (docs/isa.md): the RTL's MXU skips it, which wrote nothing
    # where the simulator wrote +0 (KB = 0) or failed on RMAX's empty rows (N = 0)
    for n, kb in ((1, 0), (0, 1)):
        ins = I.Instr(I.MM, flags=I.F_UNIT | I.F_RMAX,
                      w=I._w(0, 0, 0, n | kb << 16, 32, 1 | 1 << 16))
        with pytest.raises(SimError, match="MM: N and KB"):
            run1([ins, I.halt()])
