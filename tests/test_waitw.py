"""WAITW (docs/isa.md): the ISA simulator's execution and its host hook."""
import numpy as np
import pytest

from opentpu import isa as I
from opentpu.isasim import Config, Machine, SimError

FLAG = 0x1000


def _machine(prog, words=None, S=1):
    cfg = Config(S=S, DRAM_BYTES=1 << 16)
    drams = []
    for _ in range(S):
        d = np.zeros(cfg.DRAM_BYTES, np.uint8)
        for a, v in (words or {}).items():
            d.view(np.uint32)[a // 4] = v
        drams.append(d)
    return Machine(cfg, [list(prog) + [I.halt()] for _ in range(S)], drams)


def _word(m, a, v, s=0):
    m.slices[s].m32[a // 4] = v


def test_holds_at_once_without_the_host():
    m = _machine([I.waitw(FLAG, 3, 7, I.C_EQ)], {FLAG: 7})
    m.host = lambda mach: pytest.fail("the host was called")
    m.run()
    assert m.slices[0].tmem[3] == 7 and m.slices[0].stalls == 0


def test_waits_for_the_host_then_writes_the_word_to_tmem():
    calls = []

    def host(mach):
        calls.append(1)
        _word(mach, FLAG + 4, 0x12345678)
    # R1 = 4 (DRAM address register), R3 = 2 (TMEM), R2 = 0: NE 0 on FLAG + R1, to T[5 + R3];
    # then RLD raw takes the word's bits (a denormal as fp32) into R6
    m = _machine([I.li(1, 4), I.li(3, 2), I.waitw(FLAG, 5, 0, I.C_NE, ra=1, rb=3, rc=2),
                  I.rld(6, 7, raw=True)])
    m.host = host
    m.run()
    assert calls == [1]
    sl = m.slices[0]
    assert sl.tmem[7] == 0x12345678 and sl.R[6] == 0x12345678 and sl.stalls == 1


def test_ge_is_a_signed_difference_and_the_mask_applies():
    assert I.waitw_holds(5, 3, I.C_GE) and I.waitw_holds(3, 3, I.C_GE)
    assert not I.waitw_holds(2, 3, I.C_GE)
    assert I.waitw_holds(0x00000001, 0xFFFFFFFF, I.C_GE)      # wrapped: 1 - (-1) = 2
    assert not I.waitw_holds(0x7FFFFFFF, 0x80000001, I.C_GE)  # difference -2
    assert I.waitw_holds(0xAB00, 0xAB00, I.C_EQ, mask=0xFF00)
    assert I.waitw_holds(0xAB12, 0xAB00, I.C_EQ, mask=0xFF00)
    # the reference is R[rc] + w3: served >= seq - 1 with seq = 10 in R4
    m = _machine([I.li(4, 10), I.waitw(FLAG, 6, -1, I.C_GE, rc=4)], {FLAG: 9})
    m.run()
    assert m.slices[0].tmem[6] == 9


def test_no_host_or_no_progress_is_the_timeout():
    m = _machine([I.waitw(FLAG, 1, 1, I.C_EQ)])
    with pytest.raises(SimError, match="WAITW"):
        m.run()
    m = _machine([I.waitw(FLAG, 1, 1, I.C_EQ)])
    m.host = lambda mach: None
    with pytest.raises(SimError, match="timeout"):
        m.run()


def test_the_host_acts_only_when_no_slice_can_run():
    """Slice 0 waits on its flag; slice 1 runs a longer program first: the host is called once
    both have stopped (slice 1 at the BAR), then both pass the BAR."""
    seen = []

    def host(mach):
        seen.append((mach.slices[0].polling is not None, mach.slices[1].waiting is not None))
        _word(mach, FLAG, 1, s=0)
    cfg = Config(S=2, DRAM_BYTES=1 << 16)
    p0 = [I.waitw(FLAG, 1, 1, I.C_EQ), I.rld(1, 1, raw=True), I.bar(), I.halt()]
    p1 = [I.li(2, 1), I.li(2, 2), I.li(2, 3), I.bar(), I.halt()]
    m = Machine(cfg, [p0, p1], [None, None])
    m.host = host
    m.run()
    assert seen == [(True, True)]
    assert m.slices[0].R[1] == 1 and m.slices[1].R[2] == 3


def test_encoding_round_trips():
    ins = I.waitw(0x40, 0x80, 0xFFFFFFFF, I.C_GE, ra=3, rb=4, rc=5, mask=0xFF, interval=64,
                  timeout=1000)
    d = I.Instr.decode(ins.encode())
    assert (d.op, d.ra, d.rb, d.rc, d.rd, d.flags) == (I.WAITW, 3, 4, 5, 0, I.C_GE)
    assert d.w[:6] == [0x40, 0x80, 0xFFFFFFFF, 0xFF, 64, 1000]
    with pytest.raises(ValueError):
        I.waitw(0, 0, 0, 3)
