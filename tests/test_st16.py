"""ST16 (docs/isa.md "ST16"): DSTEP's and STREAM's DRAM state in 16 bits (Config.ST16: e6m9 or
bf16), fp32 in the step. The formats' conversions, the ISA simulator's steps against the fp32
steps on the expanded state, the compiler's flags and refusals, the scoreboard's footprint, and
the RTL (otpu_dma ST16) against the ISA simulator bit for bit: DSTEP single and PAIR, STREAM in
place, into another state (the MTP fork's) and with one side alone 16-bit (ST16R / ST16W: the
chunked DeltaNet's conversions, ol.state_copy), at SE_R 0, 1 and 2."""
import dataclasses

import numpy as np
import pytest

from opentpu import Config, isa as I, rtlsim, waits
from opentpu.compiler import CompileError
from opentpu.isasim import (Machine, SimError, bf16_f32, bf16_rne, e6m9_f32, e6m9_rne,
                            st16_f32, st16_pack)

from test_vops import _dstep_case, _dstep_pair_case, _dstep_pair_rtl_program, _run_dram


def f(x):
    return np.asarray(x, np.float32)


def _wide(rng, n):
    """Values over e6m9's range and past it: magnitudes 2^-36 .. 2^36, +-0, fp32 denormals,
    the smallest normal 2^-30 and its neighbours, the largest finite (2 - 2^-9) 2^31 and the
    values rounding to 2^32, huge ones, +-inf."""
    x = (rng.standard_normal(n) * np.exp2(rng.uniform(-36, 36, n))).astype(np.float32)
    u = rng.random(n)
    sg = np.where(rng.random(n) < 0.5, -1, 1).astype(np.float32)
    for lo, hi, v in ((0, .03, 0.0), (.03, .05, 1e-40), (.05, .07, 2.0 ** -30),
                      (.07, .08, 2.0 ** -30 * (1 - 2 ** -12)), (.08, .09, 2.0 ** -30 * (1 + 2 ** -9)),
                      (.09, .10, 2.0 ** 31 * (2 - 2 ** -9)), (.10, .11, 2.0 ** 31 * (2 - 2 ** -10)),
                      (.11, .12, 3e38), (.12, .125, np.inf)):
        m = (u >= lo) & (u < hi)
        x[m] = f(v) * sg[m]
    return x


def test_e6m9_values():
    """e6m9 (6 exponent bits, bias 31, 9 mantissa bits): round to nearest even, flush below
    2^-30 to a signed zero, saturate past (2 - 2^-9) 2^31 (or rounding up to 2^32) to it, an
    infinity one and a NaN a NaN; expanding is exact; st16_pack / st16_f32 pick the format."""
    x = f([1.0, -1.0, 1 + 2 ** -10, 1 + 3 * 2 ** -10, 1 + 2 ** -10 + 2 ** -20, 2.0 ** -30,
           2.0 ** -30 * (1 - 2 ** -12), 2.0 ** -31, 1e-40, 2.0 ** 31 * (2 - 2 ** -9),
           2.0 ** 31 * (2 - 2 ** -10), 2.0 ** 32, 3e38, np.inf, -np.inf, 0.0, -0.0, -2.0 ** 40])
    assert e6m9_rne(x).tolist() == [0x3E00, 0xBE00, 0x3E00, 0x3E02, 0x3E01, 0x0200, 0, 0, 0,
                                    0x7DFF, 0x7DFF, 0x7DFF, 0x7DFF, 0x7E00, 0xFE00, 0, 0x8000,
                                    0xFDFF]
    assert e6m9_rne(np.array([0x7FA00001], np.uint32).view(np.float32))[0] == 0x7F00
    assert np.isnan(e6m9_f32(np.array([0x7F00], np.uint16)))[0]
    h = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    fin = ((h >> 9) & 0x3F) != 0x3F
    assert np.array_equal(e6m9_rne(e6m9_f32(h))[fin], np.where(((h >> 9) & 0x3F) == 0, h & 0x8000,
                                                                h)[fin])
    rng = np.random.default_rng(16)
    x = _wide(rng, 1 << 16)
    r = e6m9_f32(e6m9_rne(x)).astype(np.float64)
    ok = np.isfinite(x) & (np.abs(x) >= 2.0 ** -30) & (np.abs(x) <= 2.0 ** 31 * (2 - 2 ** -9))
    assert np.all(np.abs(r[ok] - x[ok]) <= np.abs(x[ok]) * 2.0 ** -10)
    for fmt, pk, ex in (("e6m9", e6m9_rne, e6m9_f32), ("bf16", bf16_rne, bf16_f32)):
        assert np.array_equal(st16_pack(x, fmt), pk(x))
        assert np.array_equal(st16_f32(pk(x), fmt), ex(pk(x)), equal_nan=True)


def test_e6m9_is_the_studys_format():
    """The ISA simulator's e6m9 is the dKL study's emulation (tools/dn_state.py store: the
    format the gate measured) value for value."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import dn_state
    x = _wide(np.random.default_rng(17), 1 << 16)
    x = x[np.isfinite(x)]
    assert np.array_equal(dn_state.store(x, "e6m9"), e6m9_f32(e6m9_rne(x)).astype(np.float64))


def _cfg(fmt, **kw):
    return Config(**{**dict(S=1, D=128, DRAM_BYTES=1 << 20, DSTEP=True, DSTEP2=True, STREAM=True,
                            ST16=fmt, ST16S=bool(fmt)), **kw})


@pytest.mark.parametrize("fmt", ["e6m9", "bf16"])
def test_dstep_st16_is_the_fp32_step_rounded(fmt):
    """DSTEP with ST16 is the fp32 DSTEP on the state its 16-bit values expand to, its new state
    rounded (st16_pack) and its o the fp32 step's; the pair's two heads the same, head b's state
    rows after head a's; nothing past the state's 2 rows cols bytes written."""
    cfg = _cfg(fmt)
    for rows, cols, pair, sp in ((128, 128, False, False), (7, 192, False, True),
                                 (64, 128, True, False), (32, 128, True, True)):
        rng = np.random.default_rng(rows + cols)
        dram, tm = _dstep_case(rng, rows, cols, sp) if not pair else \
            _dstep_pair_case(rng, rows // 2, sp)
        n = rows * cols
        S = dram[0x1000:0x1000 + 4 * n].view(np.float32)
        h16 = dram.copy()
        h16[0x1000:0x1000 + 2 * n] = st16_pack(S, fmt).view(np.uint8)
        f32 = dram.copy()
        f32[0x1000:0x1000 + 4 * n] = st16_f32(st16_pack(S, fmt), fmt).view(np.uint8)
        a = _run_dram([I.dstep(0x1000, 100, 700, rows, cols, 1000, 2, 4000, pair=pair),
                       I.halt()], f32, tm, cfg)
        b = _run_dram([I.dstep(0x1000, 100, 700, rows, cols, 1000, 2, 4000, pair=pair,
                               st16=True), I.halt()], h16, tm, cfg)
        want = st16_pack(a.dram[0x1000:0x1000 + 4 * n].view(np.float32), fmt)
        assert np.array_equal(b.dram[0x1000:0x1000 + 2 * n].view(np.uint16), want)
        assert np.array_equal(b.dram[0x1000 + 2 * n:0x1000 + 4 * n],
                              h16[0x1000 + 2 * n:0x1000 + 4 * n])
        assert np.array_equal(a.tmem[4000:4000 + rows], b.tmem[4000:4000 + rows])


@pytest.mark.parametrize("fmt,dst", [("e6m9", 0x1000), ("e6m9", 0x40000), ("bf16", 0x40000)])
def test_stream_st16_is_the_fp32_stream_rounded(fmt, dst):
    """STREAM with ST16 (the fork's: Gated DeltaNet's descriptor, into the state or another) is
    the fp32 STREAM on the expanded state, its result rounded; the same as DSTEP's ST16 step;
    the source kept when it writes another state."""
    cfg = _cfg(fmt)
    rows, cols = 64, 128
    rng = np.random.default_rng(5)
    dram, tm = _dstep_case(rng, rows, cols, True)
    tm = dict(tm)
    d = I.gdn_desc(rows, cols)
    tm[30000] = np.array(d.words(), np.uint32).view(np.float32)
    n = rows * cols
    S = dram[0x1000:0x1000 + 4 * n].view(np.float32)
    h16 = dram.copy()
    h16[0x1000:0x1000 + 2 * n] = st16_pack(S, fmt).view(np.uint8)
    f32 = dram.copy()
    f32[0x1000:0x1000 + 4 * n] = st16_f32(st16_pack(S, fmt), fmt).view(np.uint8)
    a = _run_dram([I.stream(30000, 0x1000, dst, 100, 700, 1000, 4000, ks=2), I.halt()], f32, tm,
                  cfg)
    b = _run_dram([I.stream(30000, 0x1000, dst, 100, 700, 1000, 4000, ks=2, st16=True),
                   I.halt()], h16, tm, cfg)
    c = _run_dram([I.dstep(0x1000, 100, 700, rows, cols, 1000, 2, 4000, st16=True), I.halt()],
                  h16, tm, cfg)
    want = st16_pack(a.dram[dst:dst + 4 * n].view(np.float32), fmt)
    assert np.array_equal(b.dram[dst:dst + 2 * n].view(np.uint16), want)
    assert np.array_equal(b.dram[dst:dst + 2 * n], c.dram[0x1000:0x1000 + 2 * n])
    assert np.array_equal(b.tmem[4000:4000 + rows], c.tmem[4000:4000 + rows])
    assert np.array_equal(a.tmem[4000:4000 + rows], b.tmem[4000:4000 + rows])
    if dst != 0x1000:
        assert np.array_equal(b.dram[0x1000:0x1000 + 2 * n], h16[0x1000:0x1000 + 2 * n])


@pytest.mark.parametrize("fmt,side,adj", [("e6m9", "r", False), ("e6m9", "w", True),
                                          ("bf16", "r", True), ("bf16", "w", False)])
def test_stream_st16_one_side(fmt, side, adj):
    """STREAM with ST16R (a 16-bit source, an fp32 destination) is the fp32 STREAM on the
    expanded state, unrounded; with ST16W (fp32 source, 16-bit destination) the fp32 STREAM's
    result rounded. The destination right after the source (adj) or apart; the source kept."""
    cfg = _cfg(fmt)
    rows, cols = 64, 128
    rng = np.random.default_rng(15 + adj)
    dram, tm = _dstep_case(rng, rows, cols, True)
    tm = dict(tm)
    d = I.gdn_desc(rows, cols)
    tm[30000] = np.array(d.words(), np.uint32).view(np.float32)
    n = rows * cols
    S = dram[0x1000:0x1000 + 4 * n].view(np.float32)
    f32 = dram.copy()
    if side == "r":
        f32[0x1000:0x1000 + 4 * n] = st16_f32(st16_pack(S, fmt), fmt).view(np.uint8)
        src = dram.copy()
        src[0x1000:0x1000 + 2 * n] = st16_pack(S, fmt).view(np.uint8)
    else:
        src = dram
    dst = 0x1000 + (2 if side == "r" else 4) * n if adj else 0x40000
    a = _run_dram([I.stream(30000, 0x1000, 0x40000, 100, 700, 1000, 4000, ks=2), I.halt()], f32,
                  tm, cfg)
    b = _run_dram([I.stream(30000, 0x1000, dst, 100, 700, 1000, 4000, ks=2,
                            **{f"st16{side}": True}), I.halt()], src, tm, cfg)
    y = a.dram[0x40000:0x40000 + 4 * n]
    if side == "r":
        assert np.array_equal(b.dram[dst:dst + 4 * n], y)
    else:
        assert np.array_equal(b.dram[dst:dst + 2 * n].view(np.uint16),
                              st16_pack(y.view(np.float32), fmt))
    assert np.array_equal(a.tmem[4000:4000 + rows], b.tmem[4000:4000 + rows])
    nb = (2 if side == "r" else 4) * n
    assert np.array_equal(b.dram[0x1000:0x1000 + nb], src[0x1000:0x1000 + nb])
    st = I.stream(30000, 0x1000, dst, 100, 700, 1000, 4000, ks=2, **{f"st16{side}": True})
    assert st.flags == (I.F_SST16R if side == "r" else I.F_SST16W)


def test_st16_one_side_refusals():
    """ST16 on one side: in place (the encoder), a destination overlapping the source (the
    ISA simulator, by each side's bytes), a TMEM end, a machine without ST16S (CAPS2 bit18),
    and the compiler's dst overlapping the state or in place, are refused; both sides ST16R +
    ST16W encode as ST16. ST16S: in the CFG_HASH and the RTL parameters only when on, with
    ST16."""
    rows, cols = 16, 64
    n = rows * cols
    dram, tm = _dstep_case(np.random.default_rng(1), rows, cols)
    tm = {**tm, 30000: np.array(I.gdn_desc(rows, cols).words(), np.uint32).view(np.float32)}
    with pytest.raises(ValueError, match="apart"):
        I.stream(30000, 0x1000, 0x1000, 100, 700, 1000, 4000, ks=2, st16r=True)
    with pytest.raises(ValueError, match="ST16"):
        I.stream(30000, 0x1000, 0, 100, 700, 1000, 4000, ks=2, st16w=True, dst_t=True)
    assert I.stream(30000, 0x1000, 0x1000, 100, 700, 1000, 4000, ks=2, st16r=True,
                    st16w=True).flags == I.F_SST16
    # the 16-bit source's 2 n bytes, then the fp32 one's 4 n: the last chunk of each overlaps
    for side, dst in (("r", 0x1000 + 2 * n - 128), ("w", 0x1000 + 4 * n - 128),
                      ("w", 0x1000 - 2 * n + 128), ("r", 0x1000 - 4 * n + 128)):
        st = I.stream(30000, 0x1000, dst, 100, 700, 1000, 4000, ks=2, **{f"st16{side}": True})
        with pytest.raises(SimError, match="overlap"):
            _run_dram([st, I.halt()], dram, tm, _cfg("e6m9"))
    for side, dst in (("r", 0x1000 + 2 * n), ("w", 0x1000 - 2 * n), ("r", 0x1000 - 4 * n)):
        st = I.stream(30000, 0x1000, dst, 100, 700, 1000, 4000, ks=2, **{f"st16{side}": True})
        _run_dram([st, I.halt()], dram, tm, _cfg("e6m9"))
        with pytest.raises(SimError, match="ST16S"):
            _run_dram([st, I.halt()], dram, tm, _cfg("e6m9", ST16S=False))
    from opentpu import target as T
    from opentpu.host import regs as R
    assert R.CAP2_ST16S == 1 << 18 and R.caps2(R.CAP2_ST16S)["st16s"]
    assert not R.caps2(R.CAP2_SPLITQ)["st16s"] and not R.caps2(None)["st16s"]
    t = T.load("bcu1525-x2:st16")                # (ST16 without ST16S; wmx has both)
    on, off = T.build_config(t, ST16S=1), T.build_config(t)
    assert on.ST16S and not off.ST16S and not T.build_config(t, ST16S=1, ST16=0).ST16S
    assert T.arch_hash(T.build_config(t, ST16S=0)) == T.arch_hash(off) != T.arch_hash(on)
    assert "ST16S" not in T.rtl_params(t)

    from opentpu import language as ol
    from opentpu.compiler import Affine, Kernel, Tensor

    def prog(body):
        return [i for i in Kernel(body).trace(_cfg("e6m9"), 0, {}).root
                if isinstance(i, I.Instr)]

    a = Tensor(Affine(0x10000), (rows, cols), (cols, 1))
    for at, es in ((0x10000, 2), (0x10000 + 4 * n - 128, 2), (0x10000 - 2 * n + 128, 2)):
        with pytest.raises(CompileError, match="overlap"):
            prog(lambda: ol.state_copy(a, Tensor(Affine(at), (rows, cols), (cols, 1), esize=es)))
    for at in (0x10000 + 4 * n, 0x10000 - 2 * n):
        p = prog(lambda: ol.state_copy(a, Tensor(Affine(at), (rows, cols), (cols, 1), esize=2)))
        assert [i.flags for i in p if i.op == I.STREAM] == [I.F_SST16W]
    with pytest.raises(CompileError, match="ST16S"):
        [i for i in Kernel(lambda: ol.state_copy(a, Tensor(Affine(0x40000), (rows, cols),
                                                           (cols, 1), esize=2))).trace(
            _cfg("e6m9", ST16S=False), 0, {}).root]
    with pytest.raises(CompileError, match="shape"):
        prog(lambda: ol.state_copy(a, Tensor(Affine(0x40000), (rows, 128), (128, 1), esize=2)))


@pytest.mark.parametrize("fmt", ["e6m9", "bf16"])
def test_state_copy_converts(fmt):
    """ol.state_copy fp32 -> 16 bits (one STREAM, ST16W) rounds as ST16 stores (st16_pack),
    16 bits -> fp32 (ST16R) widens exactly, fp32 -> fp32 copies: -0 to +0 and fp32 subnormals
    flushed on the way; the source kept."""
    from opentpu import language as ol
    from opentpu.compiler import Affine, Kernel, Tensor
    cfg = _cfg(fmt)
    rows, cols = 129, 192
    n = rows * cols
    a32, b32, c32 = (Tensor(Affine(at), (rows, cols), (cols, 1)) for at in
                     (0x10000, 0x60000, 0x90000))
    h = Tensor(Affine(0x40000), (rows, cols), (cols, 1), esize=2)

    def body():
        ol.state_copy(a32, h)
        ol.state_copy(h, b32)
        ol.state_copy(a32, c32)
    prog = [i for i in Kernel(body).trace(cfg, 0, {}).root if isinstance(i, I.Instr)]
    st = [i for i in prog if i.op == I.STREAM]
    assert [i.flags for i in st] == [I.F_SST16W, I.F_SST16R, 0]
    S = _wide(np.random.default_rng(23), n)
    dram = np.zeros(1 << 20, np.uint8)
    dram[0x10000:0x10000 + 4 * n] = S.view(np.uint8)
    m = Machine(cfg, [prog + [I.halt()]], [dram]).run().slices[0]

    def flush(x):
        return np.where(np.abs(x) < f(2.0 ** -126), f(0), x).astype(np.float32)
    h16 = st16_pack(flush(S), fmt)
    assert np.array_equal(m.dram[0x40000:0x40000 + 2 * n].view(np.uint16), h16)
    assert np.array_equal(m.dram[0x60000:0x60000 + 4 * n].view(np.uint32),
                          flush(st16_f32(h16, fmt)).view(np.uint32))
    assert np.array_equal(m.dram[0x90000:0x90000 + 4 * n].view(np.uint32),
                          flush(S).view(np.uint32))
    assert np.array_equal(m.dram[0x10000:0x10000 + 4 * n], dram[0x10000:0x10000 + 4 * n])


def test_st16_refusals():
    """The flag on a configuration without ST16, STREAM's ST16 on a TMEM end or without a
    destination, are refused (the builder and the ISA simulator)."""
    dram, tm = _dstep_case(np.random.default_rng(1), 16, 64)
    with pytest.raises(SimError, match="ST16"):
        _run_dram([I.dstep(0x1000, 100, 700, 16, 64, 1000, 2, 4000, st16=True), I.halt()],
                  dram, tm, _cfg(""))
    tm = {**tm, 30000: np.array(I.gdn_desc(16, 64).words(), np.uint32).view(np.float32)}
    with pytest.raises(SimError, match="ST16"):
        _run_dram([I.stream(30000, 0x1000, 0x1000, 100, 700, 1000, 4000, ks=2, st16=True),
                   I.halt()], dram, tm, _cfg(""))
    with pytest.raises(ValueError, match="ST16"):
        I.stream(30000, 0, 0x1000, 100, 700, 1000, 4000, ks=2, st16=True, src_t=True)
    with pytest.raises(ValueError, match="ST16"):
        I.stream(30000, 0x1000, 0, 100, 700, 1000, 4000, ks=2, st16=True, nodst=True)
    bad = dataclasses.replace(I.stream(30000, 0x1000, 0, 100, 700, 1000, 4000, ks=2),
                              flags=I.F_SST16 | I.F_DST_T)
    with pytest.raises(SimError, match="ST16"):
        _run_dram([bad, I.halt()], dram, tm, _cfg("e6m9"))


def test_st16_footprint_is_half():
    """The scoreboard's DSTEP write range (otpu_pkg footprint, opentpu.waits): 2 rows cols bytes
    with ST16, 4 without."""
    for pair in (False, True):
        a = I.dstep(0x1000, 100, 700, 64, 128, 1000, 2, 4000, pair=pair)
        b = I.dstep(0x1000, 100, 700, 64, 128, 1000, 2, 4000, pair=pair, st16=True)
        fa, fb = (waits.footprint(x, [(0, 0)] * 16) for x in (a, b))
        assert (waits.SP_DRAM, 0x1000, 0x1000 + 4 * 64 * 128) in fa.wr
        assert (waits.SP_DRAM, 0x1000, 0x1000 + 2 * 64 * 128) in fb.wr


def test_compiler_st16_flags_and_refusals():
    """A 16-bit state (Tensor esize 2): deltanet_step, deltanet_step2 (a pair, and per head off
    its shape) and a STREAM into another state (dst) take the flag; LD / ST of it, a state
    descriptor off the hardware's subset and a machine without ST16 are refused."""
    from opentpu import language as ol
    from opentpu.compiler import Affine, Kernel, Tensor

    def prog(cfg, body):
        return [i for i in Kernel(body).trace(cfg, 0, {}).root if isinstance(i, I.Instr)]

    st = Tensor(Affine(0x1000), (128, 128), (128, 1), esize=2)

    def steps():
        qk, v, g, o = ol.empty([512]), ol.empty([128]), ol.empty([4]), ol.empty([128])
        ol.deltanet_step(st[0:64], qk[0:256], v[0:64], g[0:1], g[2:3], o[0:64])
        ol.deltanet_step2(st, qk, v, g[0:2], g[2:4], o)
        ol.deltanet_step(st[0:64], qk[0:256], v[0:64], g[0:1], g[2:3], o[0:64],
                         dst=Tensor(Affine(0x20000), (64, 128), (128, 1), esize=2))
    p = prog(_cfg("e6m9"), steps)
    ds = [i for i in p if i.op in (I.DSTEP, I.STREAM)]
    assert [i.op for i in ds] == [I.DSTEP, I.DSTEP, I.STREAM]
    assert ds[0].flags & I.F_DST16 and ds[1].flags & I.F_DST16 and ds[1].flags & I.F_DPAIR
    assert ds[2].flags & I.F_SST16 and ds[2].w[2] - ds[2].w[1] == 0x20000 - 0x1000
    p = prog(_cfg("e6m9", DSTEP2=False), steps)       # (the pair a head at a time)
    hs = [i for i in p if i.op == I.DSTEP]
    assert len(hs) == 3 and all(i.flags & I.F_DST16 for i in hs)
    assert hs[2].w[0] - hs[1].w[0] == 2 * 64 * 128
    with pytest.raises(CompileError, match="ST16"):
        prog(_cfg(""), steps)
    with pytest.raises(CompileError, match="16-bit"):
        prog(_cfg("e6m9"), lambda: ol.load(st[0:1]))
    with pytest.raises(CompileError, match="16-bit"):
        prog(_cfg("e6m9"), lambda: ol.store(st[0, :], ol.zeros([128])))


# ------------------------------------------------------------------------ RTL vs ISA simulator
def _st16_rtl_program(rng, n_steps=12):
    """_dstep_pair_rtl_program's DSTEPs (pairs, single heads of 7 / 129 rows and 64..256
    columns, the zero flag, a state stepped twice in a row) with most states in 16 bits (the
    pairs of 16 and 256 rows and the single head; the other pair's at random), and STREAMs
    (Gated DeltaNet's and other descriptors of the hardware's subset) on the single head's
    state after its DSTEPs (the first one's always, then at random): in place or into a state
    of their own (the fork's) that DSTEPs then step; then a round trip through an fp32 copy of
    the state, ST16R into it (the same descriptor) and ST16W back (state_copy's: mode scale,
    gate one; the copy in DRAM 1 to 2 MiB)."""
    prog = _dstep_pair_rtl_program(rng, n_steps)
    h16, out, descs = {}, [], {}
    fork, f32 = 0x80000, 0x100000
    for ins in prog:
        if ins.op == I.DSTEP:
            d = ins.w[0]
            h16.setdefault(d, len(h16) != 1 or bool(rng.integers(2)))
            if h16[d]:
                ins = dataclasses.replace(ins, flags=ins.flags | I.F_DST16)
            out.append(ins)
            rows, cols = ins.w[3] & 0xFFFF, ins.w[3] >> 16
            if h16[d] and not ins.flags & I.F_DPAIR and (not descs or rng.integers(2)):
                # a STREAM on the same state (the vectors DSTEP read), then into the fork's
                desc = I.gdn_desc(rows, cols) if rng.integers(2) else I.state_desc(
                    rows, cols, I.D_DELTA1, 1, "one", bool(rng.integers(2)))
                at = descs.setdefault(tuple(desc.words()), 60000 + 16 * len(descs))
                dst = d if rng.integers(2) else fork
                args = (ins.w[1], ins.w[2], ins.w[4], ins.w[5] + 400)
                out.append(I.stream(at, d, dst, *args, ks=ins.w[6], st16=True))
                if dst == fork:             # the fork's state stepped by a DSTEP
                    out.append(dataclasses.replace(ins, w=(fork,) + tuple(ins.w[1:]),
                                                   flags=I.F_DST16))
                    fork += -(-rows * cols * 2 // 128) * 128
                back = I.state_desc(rows, cols, I.D_SCALE, 1, "one", False)
                ab = descs.setdefault(tuple(back.words()), 60000 + 16 * len(descs))
                out += [I.stream(at, d, f32, *args, ks=ins.w[6], st16r=True),   # 16 -> fp32
                        I.stream(ab, f32, d, *args, ks=ins.w[6], st16w=True)]   # -> 16
        else:
            out.append(ins)
    fills = [I.vop(I.V_FILL, at + j, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR,
                   imm=np.uint32(w).view(np.float32))
             for words, at in descs.items() for j, w in enumerate(words)]
    assert any(i.flags & I.F_DST16 and i.flags & I.F_DPAIR for i in out if i.op == I.DSTEP)
    assert any(i.op == I.STREAM and i.flags & I.F_SST16R for i in out)
    return fills + out


def _wide_state(rng, n):
    """A state's fp32 values for the 16-bit tests: e6m9's range and past it (_wide) or
    normals."""
    return _wide(rng, n) if rng.integers(2) else f(0.3 * rng.standard_normal(n))


@pytest.mark.parametrize("fmt,uarch,se_r,seed", [("e6m9", {}, 0, 0), ("e6m9", "board", 0, 1),
                                                 ("e6m9", "board", 1, 2), ("e6m9", {}, 2, 3),
                                                 ("e6m9", "board", 2, 4), ("bf16", "board", 0, 5),
                                                 ("bf16", "board", 2, 6)])
def test_st16_rtl_bit_exact(have_verilator, fmt, uarch, se_r, seed):
    """otpu_dma ST16 (half the chunks each way, the 16-bit values expanded as fed, the updated
    ones rounded into the chunks written: SR 1 in the gather in pairs, SR 2 into the row
    groups' chunk registers at the fp32 positions) against the ISA simulator, TMEM and DRAM bit
    for bit: DSTEP single and PAIR, STREAM in place and into another state, with fp32 states
    beside them; states over the format's range and past it (saturation, flush, inf)."""
    cfg = _cfg(fmt, OUTER=se_r == 0, DRAM_BYTES=1 << 21)
    uarch = dict(rtlsim.BOARD_UARCH) if uarch == "board" else dict(uarch)
    uarch |= {"SE_R": se_r} if se_r else {}
    r = np.random.default_rng(1600 + seed)
    prog = _st16_rtl_program(r)
    img = np.zeros(1 << 21, np.uint8)
    img[:4 * 4096] = (_wide(r, 4096) if seed % 2 else f(r.standard_normal(4096))).view(np.uint8)
    img[0x10000:0x10000 + 4 * 65536] = _wide_state(r, 65536).view(np.uint8)
    img[0x80000:0x80000 + 4 * 65536] = _wide_state(r, 65536).view(np.uint8)
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], uarch=uarch)
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"
    bad = np.nonzero(drams[0] != m.slices[0].dram)[0]
    assert len(bad) == 0, f"{len(bad)} DRAM bytes differ, first at {bad[:8]}"


@pytest.mark.parametrize("grouped", [False, True])
def test_tiny_qwen35_st16_image(grouped):
    """A Qwen3.5 image on a Config.ST16 machine keeps its DeltaNet states in 16 bits (Image.st16,
    Spec.state16 None): half their bytes in the state regions (the snapshots', chat-KV's) and
    kv_bytes, every DSTEP with flag ST16, the logits within the format's rounding of the fp32
    states' (not equal), the snapshots' state format named; Spec.state16 False keeps fp32
    states there."""
    pytest.importorskip("torch")
    from opentpu.isasim import board_config
    from opentpu.llm import snapshot as SN
    from opentpu.llm.qwen3 import Engine
    from test_qwen35 import _tiny_model
    _, W, spec = _tiny_model(8)
    spec = dataclasses.replace(spec, pair_loop=grouped)
    c32 = board_config(DRAM_BYTES=1 << 25, DSTEP=True, STREAM=True, PAIR=True)
    c16 = dataclasses.replace(c32, ST16="e6m9")
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 12)]
    engs = [Engine(s, W, cap=256, cfg=c) for s, c in
            ((spec, c32), (spec, c16), (dataclasses.replace(spec, state16=False), c16))]
    assert [e.image.st16 for e in engs] == [False, True, False]
    assert [SN.state_format(e.image) for e in engs] == ["", "e6m9", ""]
    dn = lambda img: sum(n for (_, n), t in zip(img.state_regions(0), _kinds(img)) if t)  # noqa
    assert 2 * dn(engs[1].image) == dn(engs[0].image) == dn(engs[2].image)
    dv, dk, nl = spec.lin_dv, spec.lin_dk, engs[0].image.nl
    nlin = spec.kinds.count("linear")
    assert engs[0].image.kv_bytes - engs[1].image.kv_bytes == nlin * 2 * nl * dv * dk
    fl = [i.flags for e in engs for p in e.image.compile_step(5) for i in p if i.op == I.DSTEP]
    n = len(fl) // 3
    assert n and [bool(f & I.F_DST16) for f in fl] == [False] * n + [True] * n + [False] * n
    out = [np.stack([e.step(t) for t in toks]) for e in engs]
    assert np.array_equal(out[0], out[2]) and not np.array_equal(out[0], out[1])
    cos = np.sum(out[0] * out[1], 1) / np.linalg.norm(out[0], axis=1) / np.linalg.norm(out[1],
                                                                                        axis=1)
    assert cos.min() > 0.999


def _kinds(img):
    """Per state region (Image.state_regions' order, a slice's): True for a DeltaNet state,
    False for a convolution window."""
    out = []
    for kind in img.spec.kinds:
        if kind != "linear":
            continue
        pairs = range(img.nl // 2)
        out += [x for _ in pairs for x in (False, True, True)] if img.grouped else \
            [False for _ in pairs] + [True for _ in pairs for _ in (0, 1)] + \
            ([False] if img.qkonce else [])
    return out
