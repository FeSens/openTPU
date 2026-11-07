"""DUAL (port B chunk pairs, otpu_mxu DUAL): int8 MM PAIR takes two chunks an advance, the even
block in columns j < M and the odd one in columns M..2M-1 (ACT rows j + M, QACT DUP). The RTL
against the bit-exact ISA simulator: the same DRAM and TMEM."""
import numpy as np
import pytest

from opentpu import Config, isa as I
from opentpu.isasim import Machine
from opentpu import rtlsim

from test_rtl import (DATA, INT8, MX_BASE, MX_PITCH, _hazard_program, _images, _mx_scales,
                      _random_program)


def _cfg(mcols=4, rows=None, D=32):
    return Config(S=1, D=D, MCOLS=mcols, ACT_ROWS=rows or mcols, PAIR=True, DUAL=True)


def _same(cfg, prog, img, **kw):
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [img.copy()], **kw)
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    return st


@pytest.mark.parametrize("mcols,M", [(4, 1), (4, 2), (8, 3)])
@pytest.mark.parametrize("KB", [1, 3, 4])
@pytest.mark.parametrize("off", [0, 1])
def test_int8_pair(have_verilator, mcols, M, KB, off):
    """An int8 PAIR MM: ceil(KB/2) advances a row (the last of an odd KB one chunk), its rows
    at an even or an odd chunk (a pair across the chunk parity), 2M columns read."""
    cfg = _cfg(mcols)
    D, N = cfg.D, 20
    srs = KB * D
    rs = KB * D + D * off
    prog = [I.ld(DATA, 0, 4096), I.qact(0, M, 0, KB, srs, dup=True),
            I.mm(INT8 + D * off, MX_BASE[I.W8], 4096, N, KB, rs, N, M, 0, MX_PITCH[I.W8],
                 wf=I.W8, pair=True),
            I.halt()]
    st = _same(cfg, prog, _images(np.random.default_rng(KB + 10 * M), 1)[0])
    a = st["act_reads"][0]
    adv = -(-KB // 2)
    assert a["pop"] == N * adv
    assert a["any"] == (1 << (2 * M)) - 1


@pytest.mark.parametrize("seed,axi", [(s, False) for s in range(6)] + [(4, True), (5, True),
                                                                        (6, True), (7, True)])
def test_fuzz_int8_pair(have_verilator, seed, axi):
    """Random QACT DUP + MM sequences: int8 PAIR (unit, RSC or MX scales at an even k0; ACC,
    RMAX, ASCALE), mixed with plain int8 and 4-bit PAIR MMs on the same operands. axi: the
    board's memory path, otpu_native_dram2 on a four-channel model with random stalls and
    latency (D = 128)."""
    rng = np.random.default_rng(seed)
    cfg = _cfg(4 if seed % 2 == 0 else 8, D=32 if seed < 4 else 128)
    D = cfg.D
    prog = [I.ld(DATA, 0, 4096)]
    out = 4096
    for _ in range(8):
        M = int(rng.integers(1, cfg.MCOLS // 2 + 1))
        KB = int(rng.integers(1, 7 if D == 32 else 3))
        srs = KB * D + int(rng.integers(0, 4))
        M = min(M, 4096 // srs)
        prog.append(I.qact(int(rng.integers(0, 4096 - M * srs)), M, 0, KB, srs,
                           row=bool(rng.integers(2)), dup=True))
        for _ in range(int(rng.integers(1, 4))):
            wf = I.W8 if rng.random() < 0.7 else I.W4F
            pair = rng.random() < 0.8
            N = int(rng.integers(1, 24))
            rb = KB * D if wf == I.W8 else -(-KB // 2) * D
            rs = rb + D * int(rng.integers(0, 3))
            sa = INT8 + D * int(rng.integers(0, (32768 - N * rs) // D + 1))
            unit = bool(rng.integers(3) == 0)
            ssa, srs_s, k0, rsc = _mx_scales(rng, wf, N, KB, pair, unit)
            # ACC (and ASCALE) into the first output's words, whatever an earlier MM left there
            acc = bool(rng.integers(2)) and out > 4096
            asc = None
            if unit and acc and rng.integers(2):
                asc = int(rng.integers(0, 4096 - M))
            prog.append(I.mm(sa, ssa, 4096 if acc else out, N, KB, rs, N, M, 0, srs_s, unit=unit,
                             acc=acc, rmax=bool(rng.integers(2)), ascale=asc, wf=wf,
                             pair=pair, k0=k0, rsc=rsc))
            if not acc:
                out += M * N + M + 8
    prog.append(I.halt())
    _same(cfg, prog, _images(rng, 1)[0], **({"axi": True, "stall": 20, "seed": seed} if axi else {}))


@pytest.mark.parametrize("seed,hazard", [(0, False), (1, False), (2, False), (0, True),
                                         (1, True), (2, True)])
def test_native_dram2_mixed(have_verilator, seed, hazard):
    """Every port through otpu_native_dram2's two adapters (D = 128, four channels with random
    stalls): the fuzzers' DMA loads and stores (port B reads and writes), QST (SW), MM scale
    lines (A) and chunks (B), and the scoreboard's cross-port hazards."""
    rng = np.random.default_rng(5000 + seed)
    cfg = Config(S=1, D=128, ACT_BLOCKS=16, PAIR=True, DUAL=True)
    prog = (_hazard_program if hazard else _random_program)(rng, cfg)
    _same(cfg, prog, _images(rng, 1)[0], axi=True, stall=20, seed=seed)
