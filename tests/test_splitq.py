"""SPLITQ (Config.SPLITQ, QACT / QST flag LO; docs/isa.md "Quantization") on the ISA simulator:
the split's lo part q8lo(x * (inv * 2^d)) with the scale S * 2^-8 and the hi part's offsets, so
an operand is two int8 halves (chunked DeltaNet's matmuls,
docs/deltanet_chunk.md). The RTL against it: test_quant_rtl's test_splitq_rtl and test_fp's vectors."""
import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I, quant as Q
from opentpu.host import regs as R
from opentpu.isasim import Machine, SimError

X, WA, OUT = 0, 0x40000, 0x8000


def _put(dram, at, a):
    b = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
    dram[at:at + b.size] = b


def _rows(rng, rows, n, g, spread=3.0):
    """Rows whose groups of g span a range of magnitudes (offsets d 0 to ~8), one zero group."""
    x = (rng.standard_normal((rows, n)) *
         np.exp(rng.uniform(-spread, spread, (rows, n // g, 1))).repeat(g, 2).reshape(rows, n))
    x = x.astype(np.float32)
    x[0, g:2 * g] = 0
    return x


def test_lo_refused_without_splitq():
    """Flag LO on a machine without SPLITQ is refused; the encoders set QACT bit4, QST bit7."""
    cfg = Config(S=1, D=32, MCOLS=4, DRAM_BYTES=1 << 20)
    qa, qs = I.qact(0, 2, 0, 1, 32, lo=True), I.qst(0, 0x40000, 0x48000, 2, 1, 32, 32, 1, lo=True)
    assert qa.flags == I.F_QLO == 0x10 and qs.flags == I.F_SLO == 0x80
    for ins in (qa, qs):
        with pytest.raises(SimError, match="SPLITQ"):
            Machine(cfg, [[ins, I.halt()]], [np.zeros(cfg.DRAM_BYTES, np.uint8)]).run()


@pytest.mark.parametrize("row", [False, True])
def test_qact_lo(row):
    """QACT hi into blocks 0.., QACT LO of the same rows into blocks KB..: the LO blocks hold
    q8lo with S * 2^-8 and hi's offsets (fp32.quantize_mx(lo=True)); identity MMs over the hi
    blocks and then (ACC) the LO blocks give x to S 2^-d 2^-8, where hi alone is off by up to
    S 2^-d / 2."""
    cfg = Config(S=1, D=32, MCOLS=8, DRAM_BYTES=1 << 20, SPLITQ=True)
    D, Rr, KB = cfg.D, 6, 3
    rng = np.random.default_rng(3 + row)
    x = _rows(rng, Rr, KB * D, D // 4)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    _put(dram, WA, np.eye(D, dtype=np.int8))
    prog = [I.ld(X, 0, x.size), I.qact(0, Rr, 0, KB, KB * D, row=row),
            I.qact(0, Rr, KB, KB, KB * D, row=row, lo=True)]
    for k in range(KB):
        o = OUT + k * Rr * D
        prog += [I.mm(WA, 0, o, D, 1, D, D, Rr, k, 0, unit=True),
                 I.mm(WA, 0, o + KB * Rr * D, D, 1, D, D, Rr, k, 0, unit=True),
                 I.mm(WA, 0, o + KB * Rr * D, D, 1, D, D, Rr, KB + k, 0, unit=True, acc=True)]
    m = Machine(cfg, [prog + [I.halt()]], [dram]).run()
    sl = m.slices[0]
    q, s, d = F.quantize_mx(x, D // 4, row)
    lo, s8, d8 = F.quantize_mx(x, D // 4, row, lo=True)
    assert np.array_equal(sl.act[:Rr, :KB * D], q)
    assert np.array_equal(sl.act[:Rr, KB * D:2 * KB * D], lo)
    assert np.array_equal(sl.ascale[:Rr, KB:2 * KB], np.repeat(s8[:, None], KB, 1))
    assert np.array_equal(s8, F.mul(s, F.f32(2.0 ** -8))) and np.array_equal(d8, d)
    assert np.array_equal(sl.adoff[:Rr, KB:2 * KB], sl.adoff[:Rr, :KB])
    hi = sl.tget(OUT + np.arange(KB * Rr * D)).reshape(KB, Rr, D).transpose(1, 0, 2)
    both = sl.tget(OUT + KB * Rr * D + np.arange(KB * Rr * D)).reshape(KB, Rr, D)
    both = both.transpose(1, 0, 2)
    unit = (s[:, None] * np.exp2(-d.astype(np.float64))).repeat(D // 4, 1).reshape(Rr, KB, D)
    xr = x.reshape(Rr, KB, D).astype(np.float64)
    e_hi = np.abs(hi - xr) / unit
    e_both = np.abs(both - xr) / unit
    assert e_hi.max() <= 0.5 + 1e-5 and e_hi.max() > 0.4
    assert e_both.max() <= 2.0 ** -8 + 1e-5, e_both.max()


@pytest.mark.parametrize("row,half", [(False, False), (True, False), (True, True)])
def test_qst_lo(row, half):
    """QST LO writes q8lo's bytes and the scale row of S * 2^-8 (then, not ROW, hi's d bytes):
    the same addresses and layout as QST's."""
    cfg = Config(S=1, D=32, MCOLS=4, DRAM_BYTES=1 << 20, SPLITQ=True)
    D, Rr, KB = cfg.D, 5, 3
    rng = np.random.default_rng(7 + row + 2 * half)
    x = _rows(rng, Rr, KB * D, D)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    drs = KB * D + 4
    prog = [I.ld(X, 0, x.size),
            I.qst(0, 0x40000, 0x48000, Rr, KB, KB * D, drs, 1, row=row, half=half),
            I.qst(0, 0x50000, 0x58000, Rr, KB, KB * D, drs, 1, row=row, half=half, lo=True),
            I.halt()]
    dm = Machine(cfg, [prog], [dram]).run().slices[0].dram
    q, s, d = F.quantize_mx(x, D, row)
    lo, s8, _ = F.quantize_mx(x, D, row, lo=True)
    ne = KB * D // 2 if half else KB * D
    sw = 4 if row else 4 + 4 * -(-KB // 4)
    for r in range(Rr):
        assert np.array_equal(dm[0x40000 + r * drs:][:ne].view(np.int8), q[r, :ne])
        assert np.array_equal(dm[0x50000 + r * drs:][:ne].view(np.int8), lo[r, :ne])
        hs, ls = dm[0x48000 + r * sw:][:sw], dm[0x58000 + r * sw:][:sw]
        assert hs[:4].view(np.float32)[0] == s[r] and ls[:4].view(np.float32)[0] == s8[r]
        assert np.array_equal(hs[4:], ls[4:])
    assert np.array_equal(s8, F.mul(s, F.f32(2.0 ** -8)))


def test_splitq_caps2():
    """CAPS2 bit17 is SPLITQ (clear on every bitstream before it)."""
    assert R.CAP2_SPLITQ == 1 << 17
    assert R.caps2(R.CAP2_SPLITQ)["splitq"] and not R.caps2(R.CAP2_FSUM)["splitq"]
    assert not R.caps2(None)["splitq"]


def test_splitq_target():
    """[arch] SPLITQ: in the CFG_HASH and the RTL parameters only when on (every target keeps
    its hash), the build knob SPLITQ."""
    from opentpu import target as T
    t = T.load("bcu1525-x2")
    base = T.config(t)
    on = T.build_config(t, SPLITQ=1)
    assert not base.SPLITQ and on.SPLITQ
    assert T.arch_hash(T.build_config(t, SPLITQ=0)) == T.arch_hash(base) != T.arch_hash(on)
    assert "QUANT_SPLIT" not in T.rtl_params(t)


def test_compiler_split_dot():
    """ol.quantize(x, lo=True) (QACT flag LO) and a dot over it accumulated onto the hi dot's
    give x's product with int8 weights to ~2^-9 of the hi quantum (~100x the int8 dot's
    accuracy against the same weights); the compiler refuses lo without SPLITQ, and
    store_quantized(lo=True) emits QST's flag LO."""
    from opentpu import language as ol
    from opentpu.compiler import CompileError
    from opentpu.runtime import Input, Output, Weight, compile_kernel, launch

    @ol.jit
    def k(x, w, hi, both):
        t = ol.load(x)
        y = ol.dot(ol.quantize(t), w)
        ol.store(hi, y)
        ol.dot(ol.quantize(t, lo=True), w, acc=y)
        ol.store(both, y)

    rng = np.random.default_rng(11)
    x = rng.standard_normal((4, 128)).astype(np.float32)
    w = rng.integers(-127, 128, (16, 128)).astype(np.float32)
    w[:, 0] = 127                                                   # exact in int8 (amax 127)
    args = dict(x=Input(x), w=Weight(w), hi=Output((4, 16)), both=Output((4, 16)))
    on = Config(SPLITQ=True)
    comp, _ = compile_kernel(k, on, **args)
    qa = [p for p in comp.programs[0] if p.op == I.QACT]
    assert [bool(p.flags & I.F_QLO) for p in qa] == [False, True]
    r = launch(k, on, **args).outputs
    ref = x.astype(np.float64) @ w.T.astype(np.float64)
    e_hi, e_both = (np.abs(r[n] - ref).max() / np.abs(ref).max() for n in ("hi", "both"))
    assert e_both < e_hi / 50, (e_hi, e_both)
    with pytest.raises(CompileError, match="SPLITQ"):
        compile_kernel(k, Config(), **args)


def test_hl_refused_without_qhl():
    """QACT flag HL (bit5) needs QHL (and SPLITQ); the encoder refuses HL with LO or DUP; HL
    past ACT RAM or over QHL_ELEMS a row is refused."""
    assert I.qact(0, 2, 0, 1, 32, hl=True).flags == I.F_QHL == 0x20
    for kw in ({"lo": True}, {"dup": True}):
        with pytest.raises(ValueError):
            I.qact(0, 2, 0, 1, 32, hl=True, **kw)
    ins = I.qact(0, 2, 0, 1, 32, hl=True)
    for cfg in (Config(S=1, D=32, MCOLS=4, DRAM_BYTES=1 << 20, SPLITQ=True),
                Config(S=1, D=32, MCOLS=4, DRAM_BYTES=1 << 20, QHL=True)):
        with pytest.raises(SimError, match="QHL"):
            Machine(cfg, [[ins, I.halt()]], [np.zeros(cfg.DRAM_BYTES, np.uint8)]).run()
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=255, DRAM_BYTES=1 << 20, SPLITQ=True, QHL=True)
    for bad in (I.qact(0, 2, cfg.ACT_BLOCKS - 1, 1, 128, hl=True),
                I.qact(0, 1, 0, I.QHL_ELEMS // 128 + 1, 128, hl=True)):
        with pytest.raises(SimError, match="QACT HL"):
            Machine(cfg, [[bad, I.halt()]], [np.zeros(cfg.DRAM_BYTES, np.uint8)]).run()


@pytest.mark.parametrize("row,scales", [(False, False), (True, False), (False, True)])
def test_qact_hl(row, scales):
    """QACT HL is QACT and QACT LO of the same rows: q into blocks ab .., lo into ab + KB ..,
    their scales and offsets (RSCALE and CSCALE as the two QACTs take them)."""
    cfg = Config(S=1, D=32, MCOLS=8, DRAM_BYTES=1 << 20, SPLITQ=True, QHL=True)
    D, Rr, KB = cfg.D, 6, 3
    rng = np.random.default_rng(11 + row + 2 * scales)
    x = _rows(rng, Rr, KB * D, D // 4)
    cs = np.exp(rng.uniform(-1, 1, KB * D)).astype(np.float32)
    rsv = np.exp(rng.uniform(-1, 1, Rr)).astype(np.float32)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, np.concatenate([x.reshape(-1), cs, rsv]))
    kw = dict(row=row, cscale=Rr * KB * D if scales else None,
              rscale=Rr * KB * D + KB * D if scales else None)
    n = x.size + cs.size + rsv.size
    a = Machine(cfg, [[I.ld(X, 0, n), I.qact(0, Rr, 4, KB, KB * D, **kw),
                       I.qact(0, Rr, 4 + KB, KB, KB * D, lo=True, **kw), I.halt()]],
                [dram.copy()]).run().slices[0]
    b = Machine(cfg, [[I.ld(X, 0, n), I.qact(0, Rr, 4, KB, KB * D, hl=True, **kw), I.halt()]],
                [dram]).run().slices[0]
    for f in ("act", "ascale", "adoff"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f


def test_quantize_hl_compiler():
    """ol.quantize(x, hl=True): (hi, lo) from one QACT a chunk of rows, the same split dot as
    quantize(x) and quantize(x, lo=True) bit for bit; the wait entries cover lo's blocks."""
    import dataclasses
    from opentpu import language as ol, waits
    from opentpu import target as T
    from opentpu.compiler import Affine, QTensor, Tensor, current
    cfg = dataclasses.replace(T.config("bcu1525-x2:wmx"), S=1, DRAM_BYTES=1 << 22, QHL=True)
    D, Rr, wg = cfg.D, 40, cfg.WG
    rng = np.random.default_rng(5)
    x = _rows(rng, Rr, D, D // 4)
    w = rng.standard_normal((D, D)).astype(np.float32)
    rs, srs = Q.grp_row_bytes(D, "int8", D, wg), Q.grp_scale_bytes(D, "int8", D, wg, True)
    WQ = 0x100000
    wq = QTensor(Affine(WQ), Affine(WQ + (D // wg) * rs), (D, D), rs, srs, D, rsc=True, grp=wg)
    progs = []
    for hl in (False, True):
        @ol.jit
        def kern():
            wt = ol.load(Tensor(Affine(WA), (D, D), (D, 1)))
            current().store_quantized(wt, wq.data, wq.scale, 0, wq.srs, row_scale=True,
                                      row=Affine(0))
            del wt
            t = ol.load(Tensor(Affine(X), (Rr, D), (D, 1)))
            a = ol.quantize(t, hl=True) if hl else (ol.quantize(t), ol.quantize(t, lo=True))
            o = ol.dot(a[0], wq)
            ol.dot(a[1], wq, acc=o)
            ol.store(Tensor(Affine(OUT), (Rr, D), (D, 1)), o)
        progs.append(kern.trace(cfg, 0, {}).finish())
    assert sum(i.op == I.QACT for i in progs[1]) * 2 == sum(i.op == I.QACT for i in progs[0])
    outs = []
    for p in progs:
        dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
        _put(dram, X, x)
        _put(dram, WA, w)
        wc = waits.wait_counts(p, D, 1, aw=cfg.AW, win=cfg.WC_WIN)
        m = Machine(cfg, [p], [dram]).run()
        waits.check_isasim(m, [wc])
        outs.append(m.slices[0].dram[OUT:OUT + 4 * Rr * D].copy())
    assert np.array_equal(outs[0], outs[1])


def test_qhl_caps2_target():
    """CAPS2 bit21 is QHL; [arch] QHL in the CFG_HASH and the RTL parameters only when on, the
    build knob QHL (with SPLITQ)."""
    from opentpu import target as T
    assert R.CAP2_QHL == 1 << 21
    assert R.caps2(R.CAP2_QHL)["qhl"] and not R.caps2(R.CAP2_SPLITQ)["qhl"]
    assert not R.caps2(None)["qhl"]
    t = T.load("bcu1525-x2")
    base = T.config(t)
    on = T.build_config(t, SPLITQ=1, QHL=1)
    assert not base.QHL and on.QHL and not T.build_config(t, QHL=1).QHL
    assert T.arch_hash(T.build_config(t, SPLITQ=1, QHL=0)) != T.arch_hash(on)
    assert "QUANT_HL" not in T.rtl_params(t)


@pytest.mark.parametrize("chash", [False, True])
def test_qst_hl(chash):
    """QST HL (w5 bit31 with HLD in w5[30:20] KiB, QSTHL): a GRP ROW QST's hi part and, from the
    same read, its lo part HLD KiB after it (bytes and scales), bit for bit a QST and a QST LO;
    srs from w5[19:0]; the rows from R[rd] + w6."""
    D, R, KB, Rr = 32, 4, 3, 9
    cfg = Config(S=1, D=D, MCOLS=4, WG=R, GALL=True, CHASH=chash, DRAM_BYTES=1 << 20,
                 SPLITQ=True, QSTHL=True)
    x = _rows(np.random.default_rng(11 + chash), Rr, KB * D, D)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    es, hld = 4 * R, 17 << 10
    two = [I.li(3, 2),
           I.qst(0, 0x40000, 0x48000, Rr, KB, KB * D, 1, es, row=True, grp=R, rd=3),
           I.qst(0, 0x40000 + hld, 0x48000 + hld, Rr, KB, KB * D, 1, es, row=True, grp=R, rd=3,
                 lo=True)]
    one = [I.li(3, 2),
           I.qst(0, 0x40000, 0x48000, Rr, KB, KB * D, 1, es, row=True, grp=R, rd=3, hld=hld)]
    assert one[1].w[4] == KB * D | I.QST_HL | 17 << I.QST_HLD
    d2, d1 = (Machine(cfg, [[I.ld(X, 0, x.size)] + p + [I.halt()]], [dram.copy()])
              .run().slices[0].dram for p in (two, one))
    assert np.array_equal(d1, d2)
    assert d1[0x40000 + hld:0x48000 + hld + 0x1000].any()


def test_qst_hl_refused():
    """QST HL needs QSTHL and SPLITQ; the encoder refuses HL off GRP ROW rows, with COL, HALF or
    LO, an offset not a KiB multiple, srs past 20 bits; the simulator rows over QHL_ELEMS."""
    D, R = 32, 4
    kw = dict(row=True, grp=R, hld=1 << 10)
    ins = I.qst(0, 0x40000, 0x48000, 4, 1, D, 0, 4 * R, **kw)
    for cfg in (Config(S=1, D=D, MCOLS=4, WG=R, GALL=True, DRAM_BYTES=1 << 20, SPLITQ=True),
                Config(S=1, D=D, MCOLS=4, WG=R, GALL=True, DRAM_BYTES=1 << 20, QSTHL=True)):
        with pytest.raises(SimError, match="QSTHL"):
            Machine(cfg, [[ins, I.halt()]], [np.zeros(cfg.DRAM_BYTES, np.uint8)]).run()
    for bad in ({"row": False}, {"grp": 1}, {"col": True}, {"lo": True}, {"hld": 1000}):
        with pytest.raises(ValueError):
            I.qst(0, 0x40000, 0x48000, 4, 1, D, 0, 4 * R, **(kw | bad))
    with pytest.raises(ValueError):
        I.qst(0, 0x40000, 0x48000, 4, 1, 1 << 20, 0, 4 * R, **kw)
    cfg = Config(S=1, D=D, MCOLS=4, WG=R, GALL=True, DRAM_BYTES=1 << 22, SPLITQ=True, QSTHL=True)
    big = I.qst(0, 0x40000, 0x80000, 1, I.QHL_ELEMS // D + 1, 0, 0, 4 * R, **kw)
    with pytest.raises(SimError, match="QST HL"):
        Machine(cfg, [[big, I.halt()]], [np.zeros(cfg.DRAM_BYTES, np.uint8)]).run()


def test_qsthl_caps2_target():
    """CAPS2 bit22 is QSTHL; [arch] QSTHL in the CFG_HASH and the RTL parameters only when on,
    the build knob QSTHL (with SPLITQ and the uarch's QUANT_QSTB)."""
    from opentpu import target as T
    assert R.CAP2_QSTHL == 1 << 22
    assert R.caps2(R.CAP2_QSTHL)["qsthl"] and not R.caps2(R.CAP2_QHL)["qsthl"]
    assert not R.caps2(None)["qsthl"]
    t = T.load("bcu1525-x2")
    assert not T.config(t).QSTHL and "QUANT_QSTHL" not in T.rtl_params(t)
    assert not T.build_config(t, SPLITQ=1, QSTHL=1).QSTHL      # (no QUANT_QSTB in its uarch)
    import dataclasses
    w = T.load("bcu1525-x2:wmx")
    w = dataclasses.replace(w, uarch=w.uarch | {"QUANT_QSTB": 1})
    on = T.build_config(w, QSTHL=1)
    assert on.QSTHL and not T.build_config(w, SPLITQ=0, QSTHL=1).QSTHL
    assert T.arch_hash(T.build_config(w, QSTHL=0)) == T.arch_hash(T.config(w)) != T.arch_hash(on)
