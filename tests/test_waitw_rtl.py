"""WAITW on the RTL (otpu_dma, docs/isa.md "WAITW"): the DMA polls a DRAM word until its
condition holds, then writes it to TMEM. The host's writes during the run are the simulation
DRAM's pokes (rtlsim.run(pokes=)) and, on the ISA simulator, Machine.host."""
import numpy as np
import pytest

from opentpu import Config, isa as I
from opentpu.isasim import Machine, SimError

FLAG, DATA, OUT = 0x1000, 0x2000, 0x8000          # DRAM bytes


def _dram(cfg, words=()):
    d = np.zeros(cfg.DRAM_BYTES, np.uint8)
    for a, v in words:
        d[a:a + 4] = np.array([v & 0xFFFFFFFF], "<u4").view(np.uint8)
    return d


def _both(prog, cfg, dram, pokes=(), axi=None):
    """prog on the ISA simulator (the pokes written by its host hook when it waits) and on the
    RTL (the pokes at their cycles): the same DRAM and TMEM. Returns the RTL's stats."""
    from opentpu import rtlsim
    m = Machine(cfg, [prog], [dram.copy()])

    def host(mach):
        for _, a, v in pokes:
            mach.slices[0].m32[a // 4] = np.uint32(v & 0xFFFFFFFF)
    m.host = host
    m.run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [dram.copy()], axi=axi,
                                  pokes={0: list(pokes)} if pokes else None)
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    return {**st, "dram": drams[0]}


def test_waitw_holds_at_once(have_verilator):
    """The condition holds on the first read: EQ, NE and GE (a signed difference across the
    wrap), a mask, register-relative fields; RLD RAW takes the word, which addresses a store."""
    cfg = Config(S=1, DRAM_BYTES=1 << 20)
    dram = _dram(cfg, [(FLAG, 0x40), (FLAG + 4, 0x80000010), (FLAG + 8, 0x1234ABCD),
                       (FLAG + 64, 0x300)])
    prog = [I.waitw(FLAG, 16, 0x40, I.C_EQ),
            I.rld(1, 16, raw=True), I.st(OUT, 0, 8, ra=1),                   # at OUT + 0x40
            I.li(2, 4), I.li(3, 0x7FFFFF00),                               # 0x80000010 - 0x7FFFFF10
            I.waitw(FLAG, 17, 0x10, I.C_GE, ra=2, rc=3),                   # = 0x100 >= 0
            I.waitw(FLAG + 8, 19, 0xABCD, I.C_EQ, mask=0xFFFF),
            I.waitw(FLAG + 64, 20, 0, I.C_NE),
            I.rld(5, 20, raw=True), I.st(OUT, 8, 8, ra=5), I.halt()]
    _both(prog, cfg, dram)


def test_waitw_waits_for_the_host(have_verilator):
    """The word changes during the run: the DMA polls (interval 40) until the host's flag, and
    the LD after it reads the data the host wrote before the flag."""
    cfg = Config(S=1, DRAM_BYTES=1 << 20)
    dram = _dram(cfg, [(DATA + 4 * k, k) for k in range(16)])
    prog = [I.waitw(FLAG, 32, 0x40, I.C_EQ, interval=40),
            I.ld(DATA, 0, 16), I.st(OUT, 0, 16),
            I.rld(1, 32, raw=True), I.st(OUT, 0, 4, ra=1), I.halt()]
    pokes = [(3000, DATA + 4 * k, 1000 + k) for k in range(16)] + [(3100, FLAG, 0x40)]
    st = _both(prog, cfg, dram, pokes)
    assert st["cycles"] > 3100


def test_waitw_times_out(have_verilator):
    """A WAITW that never holds stops the slice with an error at its timeout; on the ISA
    simulator, with no host to change the word, it is the SimError."""
    from opentpu import rtlsim
    cfg = Config(S=1, DRAM_BYTES=1 << 20)
    prog = [I.waitw(FLAG, 0, 1, I.C_EQ, interval=10, timeout=2000), I.halt()]
    with pytest.raises(SimError, match="WAITW"):
        Machine(cfg, [prog], [_dram(cfg)]).run()
    with pytest.raises(RuntimeError, match="error=1"):
        rtlsim.run(cfg, [prog], [_dram(cfg)], max_cycles=100000)


def test_waitw_after_a_store_on_the_board_memory_path(have_verilator):
    """On the board's configuration and memory path: an older store to the word lands before
    WAITW's first read (EQ holds at once), and a younger store waits for it."""
    from opentpu.isasim import board_config
    cfg = board_config(DRAM_BYTES=1 << 22)
    prog = [I.vop(I.V_FILL, 0, 0, 0, 1, 16, 16, 16, 0, I.B_SCALAR, 5.0),
            I.st(FLAG, 0, 16),
            I.waitw(FLAG, 64, I.f32bits(5.0), I.C_EQ),
            I.vop(I.V_FILL, 0, 0, 0, 1, 16, 16, 16, 0, I.B_SCALAR, 6.0),
            I.st(FLAG, 0, 16), I.rld(1, 64, raw=True), I.addi(1, 1, -I.f32bits(5.0)),
            I.st(OUT, 64, 1, ra=1), I.halt()]
    _both(prog, cfg, _dram(cfg))


def test_the_waitw_op_checks_on_the_isa_simulator():
    """otpu-diag's waitw group (CAPS bit31): each program changes DRAM (its stores land)."""
    from opentpu.host.opchecks import diag_image, op_checks
    from opentpu.isasim import board_config
    cfg = board_config(DRAM_BYTES=1 << 23)
    img = diag_image()
    checks = [(n, p) for g, n, p in op_checks(cfg, waitw=True) if g == "waitw"]
    assert len(checks) == 2 and not [g for g, _, _ in op_checks(cfg, gen=True) if g == "waitw"]
    for name, prog in checks:
        m = Machine(cfg, [prog], [img.copy()]).run()
        assert (m.slices[0].dram[:len(img)] != img).sum() >= 60, name


def test_the_waitw_op_checks_on_the_board_config_rtl(have_verilator):
    from opentpu.host.checks import ZERO_AT
    from opentpu.host.opchecks import diag_image, op_checks
    from opentpu.isasim import board_config
    cfg = board_config(DRAM_BYTES=1 << 23)
    img = diag_image()
    for name, prog in [(n, p) for g, n, p in op_checks(cfg, waitw=True) if g == "waitw"]:
        dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
        dram[:len(img)] = img
        _both([I.ld(ZERO_AT, 0, cfg.TMEM_WORDS)] + prog, cfg, dram)


@pytest.mark.parametrize("r", range(4))
def test_the_card_check_waits_for_the_host(have_verilator, r):
    """checks.waitw_host's program (qual.sh's WAITW phase) on the board's configuration: EQ, NE,
    GE and the masked EQ; the host writes the new data, then the flag, during the run (pokes)."""
    from opentpu.host.checks import W_RES, W_TOK, waitw_host_program, waitw_round
    from opentpu.isasim import board_config
    cfg = board_config(DRAM_BYTES=1 << 22)
    p = waitw_round(r, sizes=(16, 17))
    old, new = np.arange(p["n"]) + 100, np.arange(p["n"]) + 7000
    dram = _dram(cfg, [(p["data"] + 4 * k, int(v)) for k, v in enumerate(old)]
                 + [(p["flag"], p["pre"])])
    pokes = [(4000, p["data"] + 4 * k, int(v)) for k, v in enumerate(new)] \
        + [(4100, p["flag"], p["word"])]
    st = _both(waitw_host_program(p, timeout=1 << 20), cfg, dram, pokes)
    assert st["cycles"] > 4100
    got = st["dram"].view("<u4")
    assert list(got[W_RES // 4:W_RES // 4 + p["n"]]) == list(new)
    assert got[W_TOK // 4] == p["word"]
