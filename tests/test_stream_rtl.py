"""STREAM on the RTL (the DMA moving the stream through the stream engine, docs/stream.md) bit for
bit against the ISA simulator: random descriptors of the hardware subset (every dmode, every
gate, A on slot 1 or 2, Q on or off, the zero flag), written by FILL VOPs right before the
STREAM, with LDs into its vectors before it and VOPs reading its outputs and rewriting its
inputs after it (RAW / WAR through the scoreboard), register-relative addresses, and DSTEPs and
VOPs interleaved (SE's arbitration)."""
import numpy as np
import pytest

from opentpu import fp32 as F
from opentpu import isa as I
from opentpu import rtlsim
from opentpu.isasim import Config, Machine

DPROGS = [I.D_DELTA, I.D_DELTA1, I.D_SCALE, I.D_DOT]
GATES = ["const", "col", "one"]


def f(x):
    return np.asarray(x, np.float32)


def _special(rng, n):
    """Normals with +-0, denormals (flushed) and a few huge values; infinities rarely."""
    x = rng.standard_normal(n).astype(np.float32)
    u = rng.random(n)
    x[u < 0.04] = 0.0
    x[(u >= 0.04) & (u < 0.06)] = -0.0
    x[(u >= 0.06) & (u < 0.08)] = f(1e-40) * np.sign(x[(u >= 0.06) & (u < 0.08)])
    x[(u >= 0.08) & (u < 0.09)] = f(3e38) * np.sign(x[(u >= 0.08) & (u < 0.09)])
    x[(u >= 0.09) & (u < 0.0905)] = np.inf * np.sign(x[(u >= 0.09) & (u < 0.0905)])
    return x


def _fill_words(addr, words):
    """FILL VOPs writing descriptor words (float-safe: FILL writes them exactly)."""
    return [I.vop(I.V_FILL, addr + j, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR,
                  float(np.uint32(w).view(np.float32))) for j, w in enumerate(words)]


def _stream_rtl_program(rng, n_steps=10, combos=None):
    NDATA = 4096
    prog = [I.ld(0, 0, NDATA)]
    state, o, st = 0x10000, 9000, 0xE0000
    heads = []
    for _ in range(3):
        rows = int(rng.choice([1, 7, 64, 128, 129, 256]))
        cols = int(rng.choice([64, 128, 192, 256]))
        heads.append((state, rows, cols))
        state += -(-rows * cols * 4 // 128) * 128 + 128 * int(rng.integers(0, 3))
    desc = 60000
    combos = list(combos or [])
    for i in range(n_steps):
        d, rows, cols = heads[int(rng.integers(len(heads)))] if i else heads[0]
        if combos and i % 3 != 2:
            dp, gate = combos.pop(0)
        else:
            dp, gate = DPROGS[int(rng.integers(4))], GATES[int(rng.integers(3))]
        a_idx = int(rng.choice([1, 2]))
        q = bool(rng.integers(4) != 0)
        ks = int(rng.integers(1, 4))
        dsc = I.state_desc(rows, cols, dp, a_idx, gate, q)
        assert I.stream_hw_cfg(dsc) is not None
        vec, x, k = 5000 + 16 * i, 7000 + 4 * i, 8000 + 8 * i
        if i % 3 == 2:
            # DSTEP beside the STREAMs: the same hardware, decoded from its fields
            prog += [I.ld(4 * int(rng.integers(0, NDATA - 2 * cols)), vec, 2 * cols),
                     I.ld(4 * int(rng.integers(0, NDATA - rows)), x, rows),
                     I.vop(I.V_FILL, k, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR,
                           float(rng.uniform(.2, 1))),
                     I.vop(I.V_FILL, k + ks, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR,
                           float(rng.uniform(0, 1)))]
            prog.append(I.dstep(d, vec, x, rows, cols, k, ks, o, zero=bool(rng.integers(4) == 0)))
        else:
            dw = desc + 32 * (i % 4)
            prog += _fill_words(dw, dsc.words())
            prog += [I.ld(4 * int(rng.integers(0, NDATA - 4 * cols)), vec, 4 * cols),
                     I.ld(4 * int(rng.integers(0, NDATA - rows)), x, rows),
                     I.vop(I.V_FILL, k, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR,
                           float(rng.uniform(.2, 1))),
                     I.vop(I.V_FILL, k + ks, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR,
                           float(rng.uniform(0, 1)))]
            if gate == "col":           # a positive per-column gate in slot 3
                prog.append(I.vop(I.V_ABS, vec + 3 * cols, vec + 3 * cols, 0, 1, cols, 0, 0, 0))
            zero = bool(rng.integers(5) == 0)
            if rng.integers(2):
                # register-relative: src/dst (ra), vec (rb), x (rc), k (rd)
                offs = [128 * int(rng.integers(0, 4)), int(rng.integers(0, 8)),
                        int(rng.integers(0, 8)), int(rng.integers(0, 8))]
                prog += [I.li(1, offs[0]), I.li(2, offs[1]), I.li(3, offs[2]), I.li(4, offs[3])]
                prog.append(I.stream(dw, d - offs[0], d - offs[0], vec - offs[1], x - offs[2],
                                     k - offs[3], o, ks=ks, zero=zero, ra=1, rb=2, rc=3, rd=4))
            else:
                prog.append(I.stream(dw, d, d, vec, x, k, o, ks=ks, zero=zero))
        # the outputs read right after (RAW), an input rewritten (WAR), an ST of o
        prog += [I.vop(I.V_MUL, o + 300, o, o, 1, rows, 0, rows, rows, I.B_FULL),
                 I.vop(I.V_FILL, vec, 0, 0, 1, 2 * cols, 0, 0, 0, I.B_SCALAR, 3.0)]
        if rng.integers(2):
            prog.append(I.st(st, o, rows))
            st += 4 * rows + 64
        o += 600
    prog.append(I.halt())
    return prog


@pytest.mark.parametrize("uarch,seed", [({}, 0), ("board", 1), ("board", 2)])
def test_stream_rtl_bit_exact(have_verilator, uarch, seed):
    cfg = Config(S=1, D=128, DRAM_BYTES=1 << 20, DSTEP=True, STREAM=True)
    uarch = dict(rtlsim.BOARD_UARCH) if uarch == "board" else uarch
    r = np.random.default_rng(1700 + seed)
    combos = [(dp, g) for dp in DPROGS for g in GATES] if seed == 0 else None
    prog = _stream_rtl_program(r, n_steps=18 if combos else 10, combos=combos)
    img = np.zeros(1 << 20, np.uint8)
    img[:4 * 4096] = (_special(r, 4096) if seed else f(r.standard_normal(4096))).view(np.uint8)
    img[0x10000:0x10000 + 4 * 65536] = f(0.3 * r.standard_normal(65536)).view(np.uint8)
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], uarch=uarch)
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"
    bad = np.nonzero(drams[0] != m.slices[0].dram)[0]
    assert len(bad) == 0, f"{len(bad)} DRAM bytes differ, first at {bad[:8]}"
