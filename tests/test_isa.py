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
    cfg = Config(S=1, D=32)
    rng = np.random.default_rng(0)
    x = f(rng.standard_normal((3, 64)))
    w = rng.integers(-127, 128, (5, 64)).astype(np.int8)
    ws = f(rng.uniform(0.01, 0.1, (5, 2)))
    dram = np.zeros(8192, np.uint8)
    dram[:768] = x.view(np.uint8).reshape(-1)
    dram[1024:1024 + 320] = w.view(np.uint8).reshape(-1)
    dram[2048:2048 + 40] = ws.view(np.uint8).reshape(-1)
    prog = [I.ld(0, 0, 192), I.qact(0, 3, 0, 2, 64), I.mm(1024, 2048, 256, 5, 2, 64, 5, 3, 0, 8),
            I.halt()]
    s = run1(prog, dram, cfg)
    y = s.tget(256 + np.arange(15)).reshape(3, 5)
    q, sx = F.quantize(x.reshape(3, 2, 32), axis=2)
    exact = np.einsum("jki,nki,nk,jk->jn", q.astype(np.float64), w.reshape(5, 2, 32),
                      ws.astype(np.float64), sx.astype(np.float64))
    assert np.allclose(y, exact, rtol=1e-5, atol=1e-5)


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
