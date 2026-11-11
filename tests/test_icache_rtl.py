"""otpu_seq's instruction cache (ICACHE, docs/icache.md) on the RTL, bit for bit against the ISA
simulator: the program stays in DRAM and the slice fetches its lines of 32 instructions (and,
with WAITC, their wait entries) on port B as it runs them. Small caches (2 or 4 lines) make the
loops' bodies evict each other; a cache the program fits in fills each of its lines once. Every
test requires the same DRAM, TMEM and instruction counts in every slice as the ISA simulator."""
import numpy as np
import pytest

from opentpu import Config, isa as I
from opentpu import rtlsim
from opentpu.isasim import Machine

from test_rtl import DATA, SCRATCH, _images, _random_program

WAITC = {"SEQ_WC": 1, "WIN": 16}


def _ua(lines: int, pf: int = 2, **kw) -> dict:
    return {"ICACHE": 1, "IC_LINES": lines, "IC_PF": pf, **kw}


def _lines(prog: list) -> int:
    return -(-len(prog) // 32)


def _check(cfg, progs, imgs, uarch, args=None, **kw):
    """progs on the ISA simulator and on the RTL at `uarch` (rtlsim.run's keywords `kw`): the
    same DRAM, TMEM and instruction counts in every slice; the RTL's stats."""
    m = Machine(cfg, progs, [i.copy() for i in imgs], args=args).run()
    drams, tmems, st = rtlsim.run(cfg, progs, [i.copy() for i in imgs], uarch=uarch, args=args,
                                  **kw)
    for s in range(cfg.S):
        assert np.array_equal(drams[s], m.slices[s].dram), f"slice {s} DRAM"
        assert np.array_equal(tmems[s], m.slices[s].tmem), f"slice {s} TMEM"
        assert st["instructions"][s] == m.slices[s].icount, f"slice {s} instructions"
    assert len(st["icache"]) == cfg.S, "the slices' ICACHE lines (otpu_seq's final)"
    return m, st


def _loops(pad_o: int, pad_i: int, outer: int, inner: int, reg_count: bool = False,
           out: int = SCRATCH) -> list:
    """Two nested loops whose bodies span lines: the outer one scales TMEM's first 16 words by
    1.5 (the inner loop's stores read them: the scoreboard, or WAITC's entries, order them) and
    pads with pad_o NOPs; the inner one (its count R4's with reg_count) stores 16 words a time
    at advancing DRAM and TMEM offsets, pad_i NOPs between; the TMEM offset starts over after
    it. Every iteration's store lands at a DRAM word of its own, so a skipped, repeated or
    misfetched instruction shows."""
    body_i = [I.st(out, 0, 16, ra=1, rb=2)] + [I.nop()] * pad_i + [I.addi(1, 1, 64),
                                                                    I.addi(2, 2, 16)]
    body_o = ([I.vop(I.V_MUL, 0, 0, 0, 1, 16, 0, 0, 0, I.B_SCALAR, 1.5)] + [I.nop()] * pad_o
              + [I.loop(len(body_i), 0 if reg_count else inner, 4 if reg_count else 0)]
              + body_i + [I.li(2, 0)])
    return ([I.ld(DATA, 0, 4096), I.li(1, 0), I.li(2, 0), I.li(4, inner),
             I.loop(len(body_o), outer)] + body_o
            + [I.st(out + 0x10000, 0, 64), I.halt()])


@pytest.mark.parametrize("pad_o,pad_i,outer,inner,lines,pf,reg", [
    (40, 50, 3, 4, 16, 2, False),        # fits: each line filled once
    (40, 50, 3, 4, 2, 2, False),         # two lines: the loops' bodies evict each other
    (20, 70, 4, 3, 4, 4, False),         # four ahead
    (60, 1, 3, 5, 2, 2, False),          # an inner loop inside a line (popped by the walker)
    (35, 33, 2, 6, 2, 2, True),          # the inner count from a register
], ids=["fits", "thrash", "pf4", "short-inner", "reg-count"])
def test_icache_nested_loops(have_verilator, pad_o, pad_i, outer, inner, lines, pf, reg):
    cfg = Config(S=1)
    prog = _loops(pad_o, pad_i, outer, inner, reg)
    imgs = _images(np.random.default_rng(31), 1)
    _, st = _check(cfg, [prog], imgs, _ua(lines, pf))
    ic, n = st["icache"][0], _lines(prog)
    assert ic["fills"] == ic["demand"] + ic["prefetch"]
    if lines >= n:                      # every line once (nothing evicted), all of them reached
        assert ic["fills"] == n, ic
        assert ic["prefetch"] > 0, ic   # the walker ran ahead of the demand
    else:
        assert ic["fills"] > n, ic


@pytest.mark.parametrize("D", [32, 64])
def test_icache_narrow_rows(have_verilator, D):
    """D 32 and 64: a line of 32 and 16 chunks (one and two instructions a row)."""
    cfg = Config(S=1, D=D)
    prog = _loops(30, 40, 3, 3)
    imgs = _images(np.random.default_rng(32), 1)
    _check(cfg, [prog], imgs, _ua(2))


@pytest.mark.parametrize("lines", [2, 16])
def test_icache_wait_entries(have_verilator, lines):
    """WAITC: each line's wait entries come with it (a fill's entry chunk, then its four
    instruction chunks: the entries' rows written from the staging register)."""
    cfg = Config(S=1)
    prog = _loops(40, 50, 3, 4)
    imgs = _images(np.random.default_rng(33), 1)
    _check(cfg, [prog], imgs, _ua(lines, **WAITC))


@pytest.mark.parametrize("seed,stall", [(0, 30), (1, 60)])
def test_icache_board_memory_path(have_verilator, seed, stall):
    """The board's memory path (the native adapter's port B, stalls): the fills' chunks come back
    in order among the DMA's, the slice tells them apart by its reads' order."""
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    rng = np.random.default_rng(4400 + seed)
    prog = _random_program(rng, cfg, n_ops=120)
    imgs = _images(rng, 1)
    _check(cfg, [prog], imgs, _ua(2), axi=True, stall=stall, seed=seed)


@pytest.mark.parametrize("seed,lines,pf", [(0, 2, 2), (1, 4, 2), (2, 2, 4), (3, 16, 2)])
def test_icache_fuzz_single_slice(have_verilator, seed, lines, pf):
    rng = np.random.default_rng(4500 + seed)
    cfg = Config(S=1)
    prog = _random_program(rng, cfg, n_ops=150)
    imgs = _images(rng, 1)
    _check(cfg, [prog], imgs, _ua(lines, pf))


@pytest.mark.parametrize("waitc", [False, True])
def test_icache_halt_chain(have_verilator, waitc):
    """HALT CHAIN: the loader drops the first program's lines and the second one runs from its
    own DRAM address (its lines at the same indices as the first's: a stale line would run the
    first program's instructions). With WAITC the second program's entries follow it in DRAM, as
    the host lays them out."""
    from opentpu import waits as W
    cfg = Config(S=1)
    second = _loops(30, 40, 2, 3, out=SCRATCH + 0x20000)
    w = np.asarray(I.assemble(second), np.uint32)
    if waitc:
        w = W.loadable(w, W.entries(w, cfg.D, cfg.S, prog=second), cfg.D)
    at = 0x80000
    imgs = _images(np.random.default_rng(34), 1)
    imgs[0][at:at + 4 * len(w)] = w.view(np.uint8)
    first = ([I.ld(DATA, 0, 4096), I.vop(I.V_MUL, 0, 0, 0, 1, 16, 0, 0, 0, I.B_SCALAR, 0.5)]
             + [I.nop()] * 70 + [I.st(SCRATCH, 0, 16), I.li(1, at), I.li(2, len(second)),
                                 I.halt(chain=True, ra=1, rb=2)])
    _check(cfg, [first], imgs, _ua(2, **(WAITC if waitc else {})))


@pytest.mark.parametrize("reload", [False, True])
def test_icache_second_run(have_verilator, reload):
    """A program run twice in one simulation (the memory path kept, as on the card): the second
    run starts with the first one's lines in the cache (RUN again), or none (LOAD again drops
    them)."""
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    prog = _loops(40, 50, 3, 4)
    imgs = _images(np.random.default_rng(35), 1)
    m1 = Machine(cfg, [prog], [imgs[0].copy()]).run()
    m2 = Machine(cfg, [prog], [m1.slices[0].dram.copy()])
    m2.slices[0].tmem[:] = m1.slices[0].tmem
    m2.run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [imgs[0].copy()], axi=True, stall=30, seed=7,
                                  uarch=_ua(16), reload=reload, again=[{}])
    assert np.array_equal(tmems[0], m2.slices[0].tmem)
    assert np.array_equal(drams[0], m2.slices[0].dram)
    n = _lines(prog)
    assert st["icache"][0]["fills"] == (2 * n if reload else n), st["icache"]


def _coll(pad: int, S: int) -> list:
    """BAR and GATHER after `pad` NOPs (the slices' pads differ: their collectives sit in other
    lines and wait for other fills), then nested loops and the gathered rows stored."""
    return ([I.ld(DATA, 0, 4096)] + [I.nop()] * pad
            + [I.bar(), I.gather(0, 8192, 1, 64, 64, 64 * S, 64)] + [I.nop()] * (80 - pad)
            + _loops(30, 40, 2, 3)[1:-1]
            + [I.gather(16, 9216, 2, 32, 32, 32 * S, 32), I.st(SCRATCH + 0x30000, 8192, 64 * S),
               I.st(SCRATCH + 0x31000, 9216, 64 * S), I.halt()])


@pytest.mark.parametrize("pads", [(5, 70), (70, 5)])
def test_icache_two_slices_bar_gather(have_verilator, pads):
    cfg = Config(S=2)
    progs = [_coll(p, 2) for p in pads]
    imgs = _images(np.random.default_rng(36), 2)
    _check(cfg, progs, imgs, _ua(2))


@pytest.mark.parametrize("seed", range(2))
def test_icache_fuzz_two_slices(have_verilator, seed):
    rng = np.random.default_rng(4600 + seed)
    cfg = Config(S=2)
    prog = _random_program(rng, cfg, n_ops=120)
    imgs = _images(rng, 2)
    _check(cfg, [prog, prog], imgs, _ua(2))
