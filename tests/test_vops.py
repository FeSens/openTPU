"""The linear-recurrence VOPs (docs/isa.md): RDOT, OUTER and LOG2.

ISA simulator vs float64 references, the LOG2 error bound, and RTL vs ISA simulator bit for bit
(random tiles with zeros, negatives, denormals and infinities)."""
from dataclasses import replace

import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I, rtlsim, waits
from opentpu.isasim import Machine, SimError

rng = np.random.default_rng(35)


def f(x):
    return np.asarray(x, np.float32)


def run1(prog, tmem_init, cfg=Config(S=1)):
    """Run `prog` with TMEM preloaded (word address -> fp32 array)."""
    m = Machine(cfg, [prog], [None])
    for addr, v in tmem_init.items():
        m.slices[0].tput(addr + np.arange(v.size), f(v).reshape(-1))
    return m.run().slices[0]


def ulp_err(y, ref):
    """|y - ref| in units of the fp32 ulp of ref (ref in float64)."""
    r32 = ref.astype(np.float32)
    ulp = np.where(r32 == 0, 2.0 ** -149, np.spacing(np.abs(r32)).astype(np.float64))
    return np.abs(y.astype(np.float64) - ref) / ulp


# ------------------------------------------------------------------------------------ RDOT
@pytest.mark.parametrize("bmode", [I.B_FULL, I.B_ROW, I.B_COL, I.B_SCALAR])
def test_rdot_all_bmodes(bmode):
    rows, cols = 5, 150
    A = f(rng.standard_normal((rows, cols)))
    Bt = f(rng.standard_normal((rows, cols)))
    prog = [I.vop(I.V_RDOT, 60000, 0, 20000, rows, cols, 1, cols, cols, bmode, imm=0.75),
            I.halt()]
    s = run1(prog, {0: A, 20000: Bt})
    got = s.tget(60000 + np.arange(rows))
    Bf = {I.B_FULL: Bt, I.B_ROW: np.repeat(Bt.reshape(-1)[:rows * cols:cols][:, None], cols, 1),
          I.B_COL: np.repeat(Bt.reshape(-1)[None, :cols], rows, 0),
          I.B_SCALAR: np.full((rows, cols), 0.75, np.float32)}[bmode]
    assert np.array_equal(got, F.rdot(A, Bf))
    want = (A.astype(np.float64) * Bf).sum(1)
    assert np.max(np.abs(got - want)) < 1e-4 * np.abs(A.astype(np.float64) * Bf).sum(1).max()


def test_rssq_is_rdot_of_a_with_itself():
    A = f(rng.standard_normal((3, 200)))
    prog = [I.vop(I.V_RSSQ, 60000, 0, 0, 3, 200, 1, 200, 0),
            I.vop(I.V_RDOT, 60010, 0, 0, 3, 200, 1, 200, 200),
            I.halt()]
    s = run1(prog, {0: A})
    assert np.array_equal(s.tmem[60000:60003], s.tmem[60010:60013])


# ------------------------------------------------------------------------------------ OUTER
@pytest.mark.parametrize("dmode", ["scalar", "column", "one"])
def test_outer_decay_plus_rank1(dmode):
    rows, cols, drs = 7, 130, 131
    S = f(rng.standard_normal((rows, drs)))
    x, y = f(rng.standard_normal(rows)), f(rng.standard_normal(cols))
    d = f(rng.uniform(0.5, 1.0, 1 if dmode == "scalar" else cols))
    prog = [I.outer(1000, 30000, 31000, 32000, rows, cols, drs, 1, dmode), I.halt()]
    s = run1(prog, {1000: S, 30000: d, 31000: x, 32000: y})
    if dmode == "one":
        d = np.ones(cols, np.float32)
    got = s.tget(1000 + np.arange(rows)[:, None] * drs + np.arange(cols)[None, :])
    assert np.array_equal(got, F.outer(S[:, :cols], d[None, :], x[:, None], y[None, :]))
    want = S[:, :cols].astype(np.float64) * d[None, :] + np.outer(x, y)
    assert np.max(np.abs(got - want)) < 1e-6 * np.abs(want).max()
    # the padding column of every row is untouched
    assert np.array_equal(s.tget(1000 + np.arange(rows) * drs + cols), S[:, cols])


def test_outer_column_vectors_are_read_before_any_write():
    """Cv and Dv may overlap dst: they are read (buffered) before the first write."""
    rows, cols = 4, 16
    S = f(rng.standard_normal((rows, cols)))
    x = f(rng.standard_normal(rows))
    # C is row 2 of S itself, D is row 3 of S itself
    prog = [I.outer(0, 3 * cols, 500, 2 * cols, rows, cols, cols, 1, "column"), I.halt()]
    s = run1(prog, {0: S, 500: x})
    got = s.tget(np.arange(rows * cols)).reshape(rows, cols)
    assert np.array_equal(got, F.outer(S, S[3][None, :], x[:, None], S[2][None, :]))


def test_outer_register_relative_column_vector():
    """The column vector's address is R[rd] + w7 (as every VOP's w7 is resolved)."""
    S = f(rng.standard_normal((2, 8)))
    x, y = f([1.0, 2.0]), f(rng.standard_normal(8))
    prog = [I.li(5, 900), I.outer(0, 800, 700, 100, 2, 8, 8, 1, rd=5), I.halt()]
    s = run1(prog, {0: S, 800: f([0.5]), 700: x, 1000: y})
    assert np.array_equal(s.tget(np.arange(16)).reshape(2, 8),
                          F.outer(S, f(0.5), x[:, None], y[None, :]))


def test_outer_b_overlapping_an_earlier_row_is_a_hazard():
    prog = [I.outer(0, 800, 8, 700, 2, 8, 8, 1), I.halt()]         # B(1) = T[8]: row 1
    with pytest.raises(SimError, match="hazard"):
        run1(prog, {})
    prog = [I.outer(0, 800, 0, 700, 2, 8, 8, 1), I.halt()]         # B(1) = T[1]: row 0
    with pytest.raises(SimError, match="hazard"):
        run1(prog, {})


# ------------------------------------------------------------------------------------ LOG2
def test_log2_error_bound_all_mantissas():
    """Every fp32 in [0.5, 4) (all mantissas, both sides of the sqrt(2) split, and e = -1..1):
    within 2.5 ulp of the correctly rounded result; plus random values over every exponent."""
    x = F.from_bits(np.arange(0x3F000000, 0x40800000, dtype=np.uint32))
    e = ulp_err(F.log2(x), np.log2(x.astype(np.float64)))
    assert e.max() < 2.5, e.max()
    b = rng.integers(0x00800000, 0x7F800000, 1 << 20, dtype=np.uint32)
    x = F.from_bits(b)
    e = ulp_err(F.log2(x), np.log2(x.astype(np.float64)))
    assert e.max() < 2.5, e.max()


def test_log2_special_values():
    x = f([0.0, -0.0, 1e-40, -1e-40, -1.0, -np.inf, np.inf, np.nan, 1.0, 2.0 ** -126,
           2.0 ** 127, 3.4e38])
    y = F.log2(x)
    assert np.all(np.isneginf(y[:4]))
    assert np.all(F.bits(y[4:6]) == 0x7FC00000) and F.bits(y[7]) == 0x7FC00000
    assert y[6] == np.inf and y[8] == 0.0 and F.bits(y[8]) == 0
    assert y[9] == -126.0 and y[10] == 127.0
    assert abs(y[11] - np.log2(3.4e38)) < 1e-5


def test_softplus_via_log2():
    """softplus(x) = max(x, 0) + ln2 * log2(1 + exp2(-|x| log2e)) with the ISA's exp2/log2:
    relative error below 1 (where softplus > 1), absolute below it (1 + e^x rounds for x << 0)."""
    x = f(np.linspace(-30, 30, 20001))
    ln2, log2e = f(np.log(2.0)), f(1 / np.log(2.0))
    sp = F.add(F.fmax(x, f(0)), F.mul(ln2, F.log2(F.add(f(1), F.exp2(F.mul(F.fabs(x), -log2e))))))
    want = np.logaddexp(0, x.astype(np.float64))
    err = np.abs(sp - want) / np.maximum(want, 1.0)
    assert err.max() < 1e-6, err.max()


def test_log2_vop():
    x = f(np.exp(rng.uniform(-60, 60, (3, 50))))
    s = run1([I.vop(I.V_LOG2, 5000, 0, 0, 3, 50, 50, 50, 0), I.halt()], {0: x})
    assert np.array_equal(s.tget(5000 + np.arange(150)).reshape(3, 50), F.log2(x))


# ------------------------------------------------------------------------------ language
from opentpu import language as ol  # noqa: E402
from opentpu import reference as ref  # noqa: E402
from opentpu.kernels.deltanet import gated_deltanet_step  # noqa: E402
from opentpu.runtime import Input, Output, compile_kernel, launch  # noqa: E402


def _vops(prog):
    return [(i.w[5] >> 16) & 0xFF for i in prog if i.op == I.VOP]


def test_language_fusions():
    @ol.jit
    def k(x, v, w, s, out, out2, out3, out4, out5):
        X, V, W, sc = ol.load(x), ol.load(v), ol.load(w), ol.load(s)
        ol.store(out, ol.sum(X * V[None, :], axis=1))       # RDOT, B per column
        t = ol.empty((6,))
        t.set(X @ V)                                        # RDOT straight into t
        ol.store(out5, t)
        ol.store(out2, ol.sum(X * X, axis=1))               # RSSQ, as before
        ol.outer(W, V, acc=X, decay=sc)                     # OUTER, scalar decay
        ol.outer(W, V, acc=X, decay=V)                      # OUTER, per-column decay
        ol.outer(W, V, acc=X)                               # OUTER, no decay
        ol.store(out3, X + ol.outer(W, V))                  # a plain rank-1 tile: MUL, ADD
        ol.store(out4, ol.log2(ol.abs(V)))
    x = f(rng.standard_normal((6, 40)))
    v, w, s = f(rng.standard_normal(40)), f(rng.standard_normal(6)), f([0.9])
    args = dict(x=Input(x), v=Input(v), w=Input(w), s=Input(s), out=Output((6,)),
                out2=Output((6,)), out3=Output((6, 40)), out4=Output((40,)), out5=Output((6,)))
    comp, _ = compile_kernel(k, Config(S=1), **args)
    assert _vops(comp.programs[0]) == [I.V_RDOT, I.V_RDOT, I.V_RSSQ, I.V_OUTER, I.V_OUTER,
                                       I.V_OUTER, I.V_MUL, I.V_ADD, I.V_ABS, I.V_LOG2]
    r = launch(k, Config(S=1), **args).outputs
    X = x.astype(np.float64)
    assert np.allclose(r["out"], X @ v, rtol=1e-5, atol=1e-5)
    assert np.array_equal(r["out5"], r["out"])
    assert np.allclose(r["out2"], (X * X).sum(1), rtol=1e-5)
    X = X * 0.9 + np.outer(w, v)
    X = X * v[None, :] + np.outer(w, v)
    X = X + 2 * np.outer(w, v)
    assert np.allclose(r["out3"], X, rtol=1e-5, atol=1e-5)
    assert np.allclose(r["out4"], np.log2(np.abs(v)), rtol=1e-6, atol=1e-6)


def test_named_product_is_not_fused():
    @ol.jit
    def k(x, v, out):
        X, V = ol.load(x), ol.load(v)
        p = X * V[None, :]
        ol.store(out, ol.sum(p, axis=1))
    args = dict(x=Input(f(rng.standard_normal((4, 16)))), v=Input(f(rng.standard_normal(16))),
                out=Output((4,)))
    comp, _ = compile_kernel(k, Config(S=1), **args)
    assert _vops(comp.programs[0]) == [I.V_MUL, I.V_RSUM]


def test_outer_errors():
    @ol.jit
    def k(x, v):
        X, V = ol.load(x), ol.load(v)
        ol.outer(X[0], V, acc=X)
    with pytest.raises(ol.CompileError, match="overlap"):
        compile_kernel(k, Config(S=1), x=Input(f(np.ones((4, 4)))), v=Input(f(np.ones(4))))


def test_without_outer():
    """A VPU built without VOP OUTER (Config.OUTER off: targets/bcu1525-x2.toml, whose DeltaNet runs
    DSTEP): the compiler refuses outer(acc=...) and the ISA simulator an OUTER, naming it; a
    plain rank-1 tile (MUL) compiles, and so does the DeltaNet step's ISA form (fused=False)."""
    cfg = Config(S=1, OUTER=False)

    @ol.jit
    def k(x, v, w, out):
        X, V, W = ol.load(x), ol.load(v), ol.load(w)
        ol.outer(W, V, acc=X)
        ol.store(out, X)

    @ol.jit
    def plain(x, v, w, out):
        ol.store(out, ol.load(x) + ol.outer(ol.load(w), ol.load(v)))
    args = dict(x=Input(f(np.ones((4, 4)))), v=Input(f(np.ones(4))), w=Input(f(np.ones(4))),
                out=Output((4, 4)))
    with pytest.raises(ol.CompileError, match="needs VOP OUTER"):
        compile_kernel(k, cfg, **args)
    assert I.V_OUTER in _vops(compile_kernel(k, Config(S=1), **args)[0].programs[0])
    comp, _ = compile_kernel(plain, cfg, **args)
    assert I.V_OUTER not in _vops(comp.programs[0])
    with pytest.raises(SimError, match="has no OUTER"):
        run1([I.outer(0, 800, 700, 100, 2, 8, 8, 1), I.halt()], {}, cfg)
    args, (Sw, ow) = deltanet_args(np.random.default_rng(7))
    r = launch(gated_deltanet_step, cfg, **args, fused=False).outputs
    assert np.max(np.abs(r["state_out"].transpose(0, 2, 1) - Sw)) < 1e-5 * np.abs(Sw).max()
    assert np.max(np.abs(r["out"] - ow)) < 1e-5 * np.abs(ow).max()


def test_outer1_decay_one_alone():
    """Config.OUTER1 (OUTER off, VMAC: VOP OUTER of decay one alone, otpu_vpu OUTER1): the
    compiler takes outer(acc=...) without a decay and refuses one with a decay tile, the ISA
    simulator runs an OUTER of decay one (bit for bit the full OUTER's) and refuses a decay word,
    naming OUTER1."""
    cfg = Config(S=1, OUTER=False, VMAC=True, OUTER1=True)

    @ol.jit
    def one(x, v, w, out):
        X, V, W = ol.load(x), ol.load(v), ol.load(w)
        ol.outer(W, V, acc=X)
        ol.store(out, X)

    @ol.jit
    def dec(x, v, w, a, out):
        X, V, W = ol.load(x), ol.load(v), ol.load(w)
        ol.outer(W, V, acc=X, decay=ol.load(a))
        ol.store(out, X)
    rng = np.random.default_rng(5)
    args = dict(x=Input(f(rng.standard_normal((4, 4)))), v=Input(f(rng.standard_normal(4))),
                w=Input(f(rng.standard_normal(4))), out=Output((4, 4)))
    comp, _ = compile_kernel(one, cfg, **args)
    assert I.V_OUTER in _vops(comp.programs[0])
    got = launch(one, cfg, **args).outputs["out"]
    want = launch(one, Config(S=1), **args).outputs["out"]
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
    with pytest.raises(ol.CompileError, match="OUTER1: decay one only"):
        compile_kernel(dec, cfg, **args, a=Input(f(np.ones(1))))
    run1([I.outer(0, 800, 700, 100, 2, 8, 8, 1, "one"), I.halt()], {}, cfg)
    for dmode in ("scalar", "column"):
        with pytest.raises(SimError, match="OUTER1: decay one only"):
            run1([I.outer(0, 800, 700, 100, 2, 8, 8, 1, dmode), I.halt()], {}, cfg)


def deltanet_args(rng, H=2, dk=128, dv=128, scale=0.3):
    S = f(scale * rng.standard_normal((H, dk, dv)))
    q, k = f(rng.standard_normal((H, dk))), f(rng.standard_normal((H, dk)))
    v = f(rng.standard_normal((H, dv)))
    a, b = f(rng.standard_normal(H)), f(rng.standard_normal(H))
    A_log, dt_bias = f(rng.uniform(-1, 1, H)), f(rng.uniform(-1, 1, H))
    args = dict(state=Input(S.transpose(0, 2, 1)), q=Input(q), k=Input(k), v=Input(v),
                a=Input(a), b=Input(b), A_log=Input(A_log), dt_bias=Input(dt_bias),
                state_out=Output((H, dv, dk)), out=Output((H, dv)))
    return args, ref.gated_deltanet_step(S, q, k, v, a, b, A_log, dt_bias)


@pytest.mark.parametrize("fused", [True, False])
def test_gated_deltanet_step_isa(fused):
    args, (Sw, ow) = deltanet_args(np.random.default_rng(7))
    comp, _ = compile_kernel(gated_deltanet_step, Config(S=1), **args, fused=fused)
    big = [i for i in comp.programs[0] if i.op == I.VOP and (i.w[3] & 0xFFFF) == 128
           and (i.w[3] >> 16) == 128]
    assert len(big) == 2 * (3 if fused else 7)                # state passes per head
    r = launch(gated_deltanet_step, Config(S=1), **args, fused=fused).outputs
    St = r["state_out"].transpose(0, 2, 1)
    assert np.max(np.abs(St - Sw)) < 1e-5 * np.abs(Sw).max()
    assert np.max(np.abs(r["out"] - ow)) < 1e-5 * np.abs(ow).max()


# ------------------------------------------------------------------------------------ DSTEP
def _dstep_case(rng, rows, cols, special=False):
    """A state in DRAM at 0x1000 and the per-head vectors in TMEM: q|k at 100, v at 700, e and
    beta at 1000 and 1002 (the decay/beta layout of qwen35's gate tile)."""
    St = (_special(rng, rows * cols) if special else
          f(0.3 * rng.standard_normal(rows * cols))).reshape(rows, cols)
    qk = _special(rng, 2 * cols) if special else f(rng.standard_normal(2 * cols))
    v = _special(rng, rows) if special else f(rng.standard_normal(rows))
    eb = f([rng.uniform(0.2, 1.0), 0.0, rng.uniform(0.0, 1.0)])
    dram = np.zeros(1 << 20, np.uint8)
    dram[0x1000:0x1000 + St.nbytes] = St.view(np.uint8).reshape(-1)
    return dram, {100: qk, 700: v, 1000: eb}


def _run_dram(prog, dram, tmem_init, cfg):
    m = Machine(cfg, [prog], [dram])
    for addr, v in tmem_init.items():
        m.slices[0].tput(addr + np.arange(v.size), f(v).reshape(-1))
    return m.run().slices[0]


def _dstep_as_vops(rows, cols):
    """The sequence DSTEP replaces (qwen35 before it): LD, RDOT, MUL, SUB, MUL, OUTER, RDOT, ST."""
    St, w, t, s2 = 20000, 3000, 3300, 3600
    return [I.ld(0x1000, St, rows * cols),
            I.vop(I.V_RDOT, w, St, 100 + cols, rows, cols, 1, cols, 0, I.B_COL),
            I.vop(I.V_MUL, t, w, 1000, 1, rows, 0, rows, 0, I.B_ROW),
            I.vop(I.V_SUB, s2, 700, t, 1, rows, 0, rows, rows, I.B_FULL),
            I.vop(I.V_MUL, w, s2, 1002, 1, rows, 0, rows, 0, I.B_ROW),
            I.outer(St, 1000, w, 100 + cols, rows, cols, cols, 1, "scalar"),
            I.vop(I.V_RDOT, 4000, St, 100, rows, cols, 1, cols, 0, I.B_COL),
            I.st(0x1000, St, rows * cols), I.halt()]


@pytest.mark.parametrize("rows,cols,special", [(128, 128, False), (64, 128, True),
                                               (7, 64, True), (256, 64, False), (32, 256, True),
                                               (3, 192, True)])
def test_dstep_is_the_vop_sequence(rows, cols, special):
    """DSTEP computes exactly what RDOT, MUL, SUB, MUL, OUTER, RDOT compute, bit for bit."""
    cfg = Config(S=1, D=128, DRAM_BYTES=1 << 20)
    dram, tm = _dstep_case(np.random.default_rng(rows * 1000 + cols), rows, cols, special)
    a = _run_dram(_dstep_as_vops(rows, cols), dram, tm, cfg)
    b = _run_dram([I.dstep(0x1000, 100, 700, rows, cols, 1000, 2, 4000), I.halt()], dram, tm,
                  cfg)
    n = rows * cols * 4
    assert np.array_equal(a.dram[0x1000:0x1000 + n], b.dram[0x1000:0x1000 + n])
    assert np.array_equal(a.tmem[4000:4000 + rows], b.tmem[4000:4000 + rows])


def test_dstep_float64():
    """Against the recurrence in float64: S = e S + beta (v - e S k) k^T, o = S q."""
    rows, cols = 128, 128
    cfg = Config(S=1, D=128, DRAM_BYTES=1 << 20)
    dram, tm = _dstep_case(np.random.default_rng(3), rows, cols)
    s = _run_dram([I.dstep(0x1000, 100, 700, rows, cols, 1000, 2, 4000), I.halt()], dram, tm,
                  cfg)
    S = dram[0x1000:0x1000 + rows * cols * 4].view(np.float32).reshape(rows, cols).astype(float)
    q, k = tm[100][:cols].astype(float), tm[100][cols:].astype(float)
    v, e, beta = tm[700].astype(float), float(tm[1000][0]), float(tm[1000][2])
    d = beta * (v - e * (S @ k))
    S = e * S + np.outer(d, k)
    got = s.dram[0x1000:0x1000 + rows * cols * 4].view(np.float32).reshape(rows, cols)
    assert np.max(np.abs(got - S)) < 1e-5 * np.abs(S).max()
    o = s.tget(4000 + np.arange(rows))
    assert np.max(np.abs(o - S @ q)) < 1e-5 * np.abs(S @ q).max()


def test_dstep_errors():
    cfg = Config(S=1, D=128, DRAM_BYTES=1 << 20)
    for ins in (I.dstep(0x1000, 100, 700, 4, 96, 1000, 2, 4000),     # cols not a multiple of 64
                I.dstep(0x1040, 100, 700, 4, 64, 1000, 2, 4000),     # state not chunk aligned
                I.dstep(0x1000, 100, 700, 257, 64, 1000, 2, 4000)):  # more than 256 rows
        with pytest.raises(SimError, match="DSTEP"):
            _run_dram([ins, I.halt()], None, {}, cfg)


def _dstep_pair_case(rng, h, special=False):
    """Two heads' states (head b's rows after head a's) at 0x1000, q | k of two key heads at 100
    (head a's, then head b's), v at 700 (2 h rows), e_a e_b beta_a beta_b at 1000."""
    cols = I.DSTEP_PAIR_COLS
    St = (_special(rng, 2 * h * cols) if special else f(0.3 * rng.standard_normal(2 * h * cols)))
    qk = _special(rng, 4 * cols) if special else f(rng.standard_normal(4 * cols))
    v = _special(rng, 2 * h) if special else f(rng.standard_normal(2 * h))
    eb = f([rng.uniform(0.2, 1.0), rng.uniform(0.2, 1.0), rng.uniform(0, 1), rng.uniform(0, 1)])
    dram = np.zeros(1 << 20, np.uint8)
    dram[0x1000:0x1000 + St.nbytes] = St.view(np.uint8).reshape(-1)
    return dram, {100: qk, 700: v, 1000: eb}


@pytest.mark.parametrize("h,qkb,special,zero", [(128, True, False, False), (128, False, True, False),
                                                (64, True, True, True), (8, False, False, False)])
def test_dstep_pair_is_two_dsteps(h, qkb, special, zero):
    """DSTEP's pair (Config.DSTEP2) is head a's DSTEP and head b's, bit for bit: head b's rows
    after head a's, its q | k after head a's (QKB) or head a's, its e and beta a word on."""
    cols = I.DSTEP_PAIR_COLS
    cfg = Config(S=1, D=128, DRAM_BYTES=1 << 20, DSTEP=True, DSTEP2=True)
    dram, tm = _dstep_pair_case(np.random.default_rng(h + 10 * qkb), h, special)
    b0 = 0x1000 + 4 * h * cols
    two = [I.dstep(0x1000, 100, 700, h, cols, 1000, 2, 4000, zero=zero),
           I.dstep(b0, 100 + 2 * cols if qkb else 100, 700 + h, h, cols, 1001, 2, 4000 + h,
                   zero=zero), I.halt()]
    one = [I.dstep(0x1000, 100, 700, 2 * h, cols, 1000, 2, 4000, zero=zero, pair=True, qkb=qkb),
           I.halt()]
    a, b = _run_dram(two, dram, tm, cfg), _run_dram(one, dram, tm, cfg)
    n = 2 * h * cols * 4
    assert np.array_equal(a.dram[0x1000:0x1000 + n], b.dram[0x1000:0x1000 + n])
    assert np.array_equal(a.tmem[4000:4000 + 2 * h], b.tmem[4000:4000 + 2 * h])
    fp = waits.footprint(one[0], [(0, 0)] * 16)      # (the sequencer's: q | k, e .. beta_b)
    assert (waits.SP_TMEM, 100, 100 + (4 if qkb else 2) * cols) in fp.rd
    assert (waits.SP_TMEM, 1000, 1004) in fp.rd


def test_dstep_pair_errors():
    """A pair without Config.DSTEP2, of rows not a multiple of 16 or of cols other than 128,
    and QKB alone, are refused (the instruction builder and the ISA simulator)."""
    cfg = Config(S=1, D=128, DRAM_BYTES=1 << 20, DSTEP=True)
    with pytest.raises(SimError, match="DSTEP2"):
        _run_dram([I.dstep(0x1000, 100, 700, 16, 128, 1000, 2, 4000, pair=True), I.halt()],
                  None, {}, cfg)
    for kw in (dict(rows=8, cols=128, pair=True), dict(rows=16, cols=64, pair=True),
               dict(rows=16, cols=128, qkb=True)):
        with pytest.raises(ValueError, match="DSTEP"):
            I.dstep(0x1000, 100, 700, kw.pop("rows"), kw.pop("cols"), 1000, 2, 4000, **kw)


# ------------------------------------------------------------------------ RTL vs ISA simulator
def _special(rng, n):
    """Normals with +-0, denormals (flushed), +-inf and a few huge values."""
    x = rng.standard_normal(n).astype(np.float32)
    u = rng.random(n)
    x[u < 0.04] = 0.0
    x[(u >= 0.04) & (u < 0.06)] = -0.0
    x[(u >= 0.06) & (u < 0.08)] = f(1e-40) * np.sign(x[(u >= 0.06) & (u < 0.08)])
    x[(u >= 0.08) & (u < 0.09)] = np.inf * np.sign(x[(u >= 0.08) & (u < 0.09)])
    x[(u >= 0.09) & (u < 0.10)] = f(3e38) * np.sign(x[(u >= 0.09) & (u < 0.10)])
    return x


def _vops_program(rng, cfg, n_ops=45):
    """Random RDOT / OUTER / LOG2 instructions mixed with older VOPs of other latencies (so
    elementwise instructions overlap in the lanes), LDs into OUTER's column vectors right
    before it and VOPs rewriting them right after (the scoreboard's C and D ranges); no OUTER
    where cfg has none (Config.OUTER)."""
    NDATA = 8192
    prog = [I.ld(0, 0, NDATA)]
    nxt = NDATA

    def fresh(n):
        nonlocal nxt
        a = nxt
        nxt += n + int(rng.integers(0, 3))
        return a

    def src(n):
        return int(rng.integers(0, NDATA - n))

    scratch = 4 * NDATA
    for _ in range(n_ops):
        kind = rng.choice(["rdot", "outer", "outer", "log2", "old", "ld_outer"] if cfg.OUTER
                          else ["rdot", "log2", "old"])
        if kind == "rdot":                   # buffered row sums (<= 256 rows, drs 1) or not
            rows, cols = int(rng.integers(1, 6)), int(rng.integers(1, 300))
            if rng.integers(4) == 0:
                rows, cols = int(rng.integers(250, 262)), int(rng.integers(1, 20))
            ars, brs = cols + int(rng.integers(0, 3)), int(rng.integers(0, cols + 2))
            drs = 1 if rng.integers(3) else 2
            if rows * max(ars, brs) + cols >= NDATA:
                ars = brs = cols
            prog.append(I.vop(I.V_RDOT, fresh(rows * drs), src(rows * ars), src(rows * brs + cols),
                              rows, cols, drs, ars, brs, int(rng.integers(0, 4)),
                              imm=float(rng.standard_normal())))
        elif kind == "log2":
            rows, cols = int(rng.integers(1, 4)), int(rng.integers(1, 60))
            prog.append(I.vop(I.V_LOG2, fresh(rows * cols), src(rows * cols), 0, rows, cols,
                              cols, cols, 0))
        elif kind == "old":
            rows, cols = int(rng.integers(1, 4)), int(rng.integers(1, 40))
            func = int(rng.choice([I.V_ADD, I.V_MUL, I.V_EXP2, I.V_RECIP, I.V_RSQRT, I.V_MAX,
                                   I.V_RSSQ, I.V_RSUM]))
            if func in I.REDUCE:
                prog.append(I.vop(func, fresh(rows), src(rows * cols), 0, rows, cols, 1, cols, 0))
            else:
                prog.append(I.vop(func, fresh(rows * cols), src(rows * cols), src(cols), rows,
                                  cols, cols, cols, 0, I.B_COL))
        else:
            rows, cols = int(rng.integers(1, 7)), int(rng.integers(1, 257))
            drs = cols + int(rng.integers(0, 2))
            dst = fresh(rows * drs)
            prog.append(I.vop(I.V_COPY, dst, src(rows * cols), 0, rows, cols, drs, cols, 0))
            dmode = str(rng.choice(["scalar", "column", "one"]))
            c, d = src(cols), src(cols)
            if kind == "ld_outer":            # C and D arrive by DMA just before the OUTER
                c, d = fresh(cols), fresh(cols)
                prog += [I.ld(4 * int(rng.integers(0, NDATA - cols)), c, cols),
                         I.ld(4 * int(rng.integers(0, NDATA - cols)), d, cols)]
            elif rng.integers(3) == 0:        # C inside dst: read before the first write
                c = dst + int(rng.integers(0, rows)) * drs
            b, brs = src(rows * 2), int(rng.integers(0, 2))
            if rng.integers(2):
                base = int(rng.integers(0, c + 1))
                prog += [I.li(7, base), I.outer(dst, d, b, c - base, rows, cols, drs, brs, dmode,
                                                rd=7)]
            else:
                prog.append(I.outer(dst, d, b, c, rows, cols, drs, brs, dmode))
            if kind == "ld_outer":            # overwrite C and D right away (WAR)
                prog += [I.vop(I.V_FILL, c, 0, 0, 1, cols, 0, 0, 0, I.B_SCALAR, 1.5),
                         I.ld(0, d, cols)]
            if rng.integers(2):
                prog.append(I.st(scratch, dst, rows * drs))
                scratch += 4 * rows * drs + 64
        assert nxt < cfg.TMEM_WORDS and scratch < cfg.DRAM_BYTES
    prog.append(I.halt())
    return prog


@pytest.mark.parametrize("lanes,uarch,seed", [(8, {}, 0), (8, {}, 1), (8, "board4", 2),
                                              (16, {}, 3), (4, {}, 4)])
def test_vops_rtl_bit_exact(have_verilator, lanes, uarch, seed):
    cfg = Config(S=1, LANES=lanes, MCOLS=min(8, lanes))
    if uarch == "board4":                     # the board's TMEM/window, 4 composite lanes
        uarch = dict(rtlsim.BOARD_UARCH, VPU_CL=4)
    r = np.random.default_rng(500 + seed)
    prog = _vops_program(r, cfg)
    img = np.zeros(1 << 20, np.uint8)
    img[:4 * 8192] = _special(r, 8192).view(np.uint8)
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], uarch=uarch)
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"
    assert np.array_equal(drams[0], m.slices[0].dram)


@pytest.mark.parametrize("lanes,uarch,seed", [(8, {}, 5), (8, "board4", 6), (16, {}, 7),
                                              (4, {}, 8)])
def test_vops_without_outer_rtl_bit_exact(have_verilator, lanes, uarch, seed):
    """A VPU built without OUTER (otpu_vpu OUTER 0, Config.OUTER off: slot 0 a*b + c, no column
    buffers): RDOT, LOG2 and the older VOPs (test_vops_rtl_bit_exact's, without OUTER), then
    EXP2SUB, MIN, RSUB, ABS, FILL and COPY (test_direct_vops_rtl_bit_exact's), RTL against the
    ISA simulator bit for bit."""
    cfg = Config(S=1, LANES=lanes, MCOLS=min(8, lanes), OUTER=False)
    if uarch == "board4":
        uarch = dict(rtlsim.BOARD_UARCH, VPU_CL=4)
    r = np.random.default_rng(500 + seed)
    prog = _vops_program(r, cfg)
    assert I.V_OUTER not in _vops(prog)
    img = np.zeros(1 << 20, np.uint8)
    img[:4 * 8192] = _special(r, 8192).view(np.uint8)
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], uarch=uarch)
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"
    assert np.array_equal(drams[0], m.slices[0].dram)
    prog = _direct_program(r)
    img[4 * 8192:4 * 12288] = f(r.standard_normal(4096) * np.float32(60)).view(np.uint8)
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], uarch=uarch)
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"


DIRECT = [I.V_EXP2SUB, I.V_MIN, I.V_RSUB, I.V_ABS, I.V_FILL, I.V_COPY]


def _direct_program(rng, n_ops=48):
    """EXP2SUB, MIN, RSUB, ABS, FILL and COPY on their own and mixed with ADD, EXP2 and RECIP
    (other latencies, so elementwise instructions overlap in the lanes): every B mode, row
    strides, odd shapes, the B_ROW word and the immediate from the special values. The data
    is the special values (words 0..8191) and values spread over EXP2's range and past it
    (8192..12287: exponents near -126 and 128, the clamps)."""
    NDATA = 12288
    prog = [I.ld(0, 0, NDATA)]
    nxt = NDATA
    ops = []

    def fresh(n):
        nonlocal nxt
        a = nxt
        nxt += n + int(rng.integers(0, 3))
        return a

    def src(n):
        return int(rng.integers(0, NDATA - n))

    for i in range(n_ops):
        func = DIRECT[i % len(DIRECT)] if i < 2 * len(DIRECT) else \
            int(rng.choice(DIRECT + [I.V_ADD, I.V_EXP2, I.V_RECIP]))
        rows, cols = int(rng.integers(1, 6)), int(rng.integers(1, 300))
        ars, drs = cols + int(rng.integers(0, 3)), cols + int(rng.integers(0, 2))
        bmode = int(rng.integers(0, 4)) if func in I.READS_B else I.B_FULL
        brs, b = 0, 0
        if bmode == I.B_FULL:
            brs = cols + int(rng.integers(0, 3))
            b = src(rows * brs)
        elif bmode == I.B_ROW:
            brs = int(rng.integers(0, 3))
            b = src(rows * brs + 1)
        elif bmode == I.B_COL:
            b = src(cols)
        imm = float(rng.choice(_special(rng, 64)))
        if func == I.V_EXP2SUB and rng.integers(2):
            imm = float(rng.uniform(-140, 140))
        a = src(rows * ars) if func != I.V_FILL else 0
        prog.append(I.vop(func, fresh(rows * drs), a, b, rows, cols, drs, ars, brs, bmode,
                          imm=imm))
        ops.append(func)
    prog.append(I.halt())
    assert set(DIRECT) <= set(ops)
    return prog


@pytest.mark.parametrize("lanes,uarch,seed", [(8, {}, 0), (8, "board", 1), (8, "board4", 2),
                                              (16, {}, 3), (4, {}, 4)])
def test_direct_vops_rtl_bit_exact(have_verilator, lanes, uarch, seed):
    """EXP2SUB, MIN, RSUB, ABS, FILL and COPY, RTL against the ISA simulator bit for bit."""
    cfg = Config(S=1, LANES=lanes, MCOLS=min(8, lanes))
    if uarch == "board":
        uarch = dict(rtlsim.BOARD_UARCH)
    elif uarch == "board4":
        uarch = dict(rtlsim.BOARD_UARCH, VPU_CL=4)
    r = np.random.default_rng(700 + seed)
    prog = _direct_program(r)
    img = np.zeros(1 << 20, np.uint8)
    img[:4 * 8192] = _special(r, 8192).view(np.uint8)
    wide = f(r.standard_normal(4096) * np.float32(60)) * np.where(r.random(4096) < 0.1, 2, 1)
    img[4 * 8192:4 * 12288] = f(wide).view(np.uint8)
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], uarch=uarch)
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"


@pytest.mark.parametrize("uarch,seed", [({}, 0), ("board", 1)])
def test_many_small_composites_rtl_bit_exact(have_verilator, uarch, seed):
    """Runs of independent tiny composite VOPs (1..3 elements, often one row of one column, as
    RMSNorm's rsqrt), so many are in flight in the VPU at once (one may start every other
    cycle, each ~66 cycles long, ~100 with COMP8's RSQRT/LOG2: more than 15, and at the
    board's WIN = 16 all 16 window entries), in rising and falling latency order, RTL
    against the ISA simulator bit for bit."""
    cfg = Config(S=1)
    uarch = dict(rtlsim.BOARD_UARCH) if uarch == "board" else uarch
    r = np.random.default_rng(900 + seed)
    img = np.zeros(1 << 20, np.uint8)
    img[:4 * 4096] = _special(r, 4096).view(np.uint8)
    prog, dst = [I.ld(0, 0, 4096)], 5000
    runs = [[I.V_RECIP] * 20, [I.V_EXP2] * 20, [I.V_RSQRT] * 40, [I.V_LOG2] * 40,
            [I.V_RECIP, I.V_EXP2, I.V_EXP2SUB, I.V_RSQRT, I.V_LOG2] * 8,
            [I.V_LOG2, I.V_RSQRT, I.V_EXP2SUB, I.V_EXP2, I.V_RECIP] * 8]
    for run in runs:
        for fn in run:
            rows, cols = (1, 1) if r.integers(3) else (int(r.integers(1, 3)), int(r.integers(1, 3)))
            a = int(r.integers(0, 4000))
            bm = I.B_SCALAR if fn == I.V_EXP2SUB else I.B_FULL
            prog.append(I.vop(fn, dst, a, int(r.integers(0, 4000)), rows, cols, cols, cols, cols,
                              bm, float(r.uniform(-3, 3))))
            dst += rows * cols + int(r.integers(0, 2))
        # independent of the run (no dependency holds them back): a reduction and an ADD may
        # start only once every composite has left the lanes
        prog.append(I.vop(I.V_RSUM, dst, int(r.integers(0, 3000)), 0, 4, 200, 1, 200, 0))
        prog.append(I.vop(I.V_ADD, dst + 8, int(r.integers(0, 3000)), int(r.integers(0, 3000)),
                          2, 40, 40, 40, 40))
        dst += 100
    prog.append(I.halt())
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], uarch=uarch)
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"


def _dstep_rtl_program(rng, n_steps=6):
    """DSTEPs on a few DRAM states (odd rows, 64..256 columns, the zero flag) with their inputs
    LDed into TMEM just before (RAW), VOPs reading o right after, the same state stepped twice
    in a row (the DRAM RAW), an input overwritten right after the DSTEP (WAR) and an ST of TMEM
    between them."""
    NDATA = 4096
    prog = [I.ld(0, 0, NDATA)]
    state, o, st = 0x10000, 8000, 0xE0000
    heads = []
    for _ in range(3):
        rows = int(rng.choice([1, 7, 64, 128, 129, 256]))
        cols = int(rng.choice([64, 128, 192, 256]))
        heads.append((state, rows, cols))
        state += -(-rows * cols * 4 // 128) * 128 + 128 * int(rng.integers(0, 3))
    for i in range(n_steps):
        d, rows, cols = heads[int(rng.integers(len(heads)))] if i else heads[0]
        qk, v, g = 5000 + 8 * i, 6000 + 2 * i, 7000 + 4 * i
        gs = int(rng.integers(1, 3))
        prog += [I.ld(4 * int(rng.integers(0, NDATA - 2 * cols)), qk, 2 * cols),
                 I.ld(4 * int(rng.integers(0, NDATA - rows)), v, rows),
                 I.vop(I.V_FILL, g, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, float(rng.uniform(.2, 1))),
                 I.vop(I.V_FILL, g + gs, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR,
                       float(rng.uniform(0, 1)))]
        prog.append(I.dstep(d, qk, v, rows, cols, g, gs, o, zero=bool(rng.integers(4) == 0)))
        prog += [I.vop(I.V_MUL, o + 300, o, o, 1, rows, 0, rows, rows, I.B_FULL),
                 I.vop(I.V_FILL, qk, 0, 0, 1, 2 * cols, 0, 0, 0, I.B_SCALAR, 3.0)]
        if rng.integers(2):
            prog.append(I.st(st, o, rows))
            st += 4 * rows + 64
        o += 600
    prog.append(I.halt())
    return prog


@pytest.mark.parametrize("uarch,seed", [({}, 0), ("board", 1), ("board", 2)])
def test_dstep_rtl_bit_exact(have_verilator, uarch, seed):
    cfg = Config(S=1, D=128, DRAM_BYTES=1 << 20, DSTEP=True)
    uarch = dict(rtlsim.BOARD_UARCH) if uarch == "board" else uarch
    r = np.random.default_rng(900 + seed)
    prog = _dstep_rtl_program(r)
    img = np.zeros(1 << 20, np.uint8)
    img[:4 * 4096] = (_special(r, 4096) if seed else f(r.standard_normal(4096))).view(np.uint8)
    img[0x10000:0x10000 + 4 * 65536] = f(0.3 * r.standard_normal(65536)).view(np.uint8)
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], uarch=uarch)
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"
    bad = np.nonzero(drams[0] != m.slices[0].dram)[0]
    assert len(bad) == 0, f"{len(bad)} DRAM bytes differ, first at {bad[:8]}"


def _dstep_pair_rtl_program(rng, n_steps=8):
    """DSTEP pairs (Config.DSTEP2: 16..256 rows, QKB both ways, beta 1 to 3 words after the
    decay or 12, as the WG targets' bursts lay a slice's gates out, the zero flag) and single DSTEPs
    (64..256 columns) on a few DRAM states, inputs LDed just before, o read right after, an
    input overwritten right after (WAR)."""
    NDATA = 4096
    prog = [I.ld(0, 0, NDATA)]
    state, o = 0x10000, 8000
    heads = []
    for k in range(4):
        pair = k < 3
        rows = [16, int(rng.choice([64, 128])), 256, int(rng.choice([7, 129]))][k]
        cols = I.DSTEP_PAIR_COLS if pair else int(rng.choice([64, 192, 256]))
        heads.append((state, rows, cols, pair))
        state += -(-rows * cols * 4 // 128) * 128 + 128 * int(rng.integers(0, 3))
    for i in range(n_steps):
        d, rows, cols, pair = heads[i] if i < len(heads) else heads[int(rng.integers(len(heads)))]
        qkb = pair and bool(rng.integers(2))
        nq = (4 if qkb else 2) * cols
        qk, v, g = 5000 + 8 * i, 6000 + 2 * i, 7000 + 8 * i
        gs = int(rng.choice([1, 2, 3, 12]))
        prog += [I.ld(4 * int(rng.integers(0, NDATA - nq)), qk, nq),
                 I.ld(4 * int(rng.integers(0, NDATA - rows)), v, rows)]
        for w, lo, hi in ((g, .2, 1), (g + 1, .2, 1), (g + gs, 0, 1), (g + gs + 1, 0, 1)):
            prog.append(I.vop(I.V_FILL, w, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR,
                              float(rng.uniform(lo, hi))))
        prog.append(I.dstep(d, qk, v, rows, cols, g, gs, o, zero=bool(rng.integers(4) == 0),
                            pair=pair, qkb=qkb))
        prog += [I.vop(I.V_MUL, o + 300, o, o, 1, rows, 0, rows, rows, I.B_FULL),
                 I.vop(I.V_FILL, qk, 0, 0, 1, nq, 0, 0, 0, I.B_SCALAR, 3.0)]
        o += 600
    prog.append(I.halt())
    return prog


@pytest.mark.parametrize("uarch,se_r,seed,outer", [({}, 0, 0, True), ("board", 0, 1, True),
                                                   ({}, 2, 2, True), ("board", 0, 3, True),
                                                   ("board", 1, 4, False)])
def test_dstep_pair_rtl_bit_exact(have_verilator, uarch, se_r, seed, outer):
    """DSTEP's pair on the RTL (otpu_dma and otpu_se_tail at DSTEP2 1: the stream on the VPU's
    engine, or at SE_R 1 / 2 an engine of its own with 1 / 2 row groups; outer off: the WG targets' VPU
    without VOP OUTER) against the ISA simulator, TMEM and DRAM bit for bit, single DSTEPs
    between them."""
    cfg = Config(S=1, D=128, DRAM_BYTES=1 << 20, DSTEP=True, DSTEP2=True, OUTER=outer)
    uarch = dict(rtlsim.BOARD_UARCH) if uarch == "board" else dict(uarch)
    uarch |= {"SE_R": se_r} if se_r else {}
    r = np.random.default_rng(950 + seed)
    prog = _dstep_pair_rtl_program(r)
    assert any(i.flags & I.F_DPAIR for i in prog) and any(
        i.op == I.DSTEP and not i.flags & I.F_DPAIR for i in prog)
    img = np.zeros(1 << 20, np.uint8)
    img[:4 * 4096] = (_special(r, 4096) if seed % 2 else f(r.standard_normal(4096))).view(np.uint8)
    img[0x10000:0x10000 + 4 * 65536] = f(0.3 * r.standard_normal(65536)).view(np.uint8)
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], uarch=uarch)
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"
    bad = np.nonzero(drams[0] != m.slices[0].dram)[0]
    assert len(bad) == 0, f"{len(bad)} DRAM bytes differ, first at {bad[:8]}"


# ------------------------------------------------------------------------------------ FSUM
FSUM_COLS = [64, 65, 71, 72, 100, 127, 128, 129, 192, 200, 256, 257, 320]


def _fsum_program(rng, n_ops=30):
    """EXP2SUBs with their row sums (FSUM: rows 1..256, cols 64 and up, every B mode but the
    scalar, sums at w7 or w7 + R[rd]) among other VOPs (reductions and composites of other
    latencies), and their takes (RSUM FSUM): between an FSUM and its take, often elementwise VOPs
    (they run during the sums' fold and flush), a reduction (it waits for the sums) or the next
    FSUM (two pending); after the take, often a VOP reading the sums at once (the sequencer waits
    for the take) and an RSUM of the results (RSUM's path, into a range of its own). Returns the
    program and the (sums, rsum, rows) of the pairs."""
    NDATA = 8192
    prog, pairs = [I.ld(0, 0, NDATA)], []
    nxt, big = NDATA, True

    def fresh(n):
        nonlocal nxt
        a = nxt
        nxt += n + int(rng.integers(0, 3))
        return a

    def src(n):
        return int(rng.integers(0, NDATA - n))

    def other(funcs):
        rows, cols = int(rng.integers(1, 4)), int(rng.integers(1, 90))
        func = int(rng.choice(funcs))
        if func in I.REDUCE:
            return I.vop(func, fresh(rows), src(rows * cols), src(rows * cols), rows, cols, 1,
                         cols, cols)
        return I.vop(func, fresh(rows * cols), src(rows * cols), src(cols), rows, cols, cols,
                     cols, 0, I.B_COL)

    def fsum():
        nonlocal big
        rows, cols = int(rng.integers(1, 7)), int(rng.choice(FSUM_COLS))
        ars = cols + int(rng.integers(0, 3))
        if big:
            big, rows, cols, ars = False, 256, int(rng.choice([64, 72])), int(rng.integers(0, 30))
        elif rng.integers(4) == 0:
            rows, cols, ars = int(rng.integers(9, 60)), int(rng.choice(FSUM_COLS[:5])), int(
                rng.integers(0, 90))
        bmode = int(rng.choice([I.B_FULL, I.B_ROW, I.B_COL]))
        brs = cols if bmode == I.B_FULL else 1 if bmode == I.B_ROW else 0
        if bmode == I.B_FULL and rows * cols + cols >= NDATA:
            brs = int(rng.integers(0, 30))
        b = src(cols) if bmode == I.B_COL else src(rows * brs + cols)
        drs = cols + int(rng.integers(0, 2))
        dst, sums = fresh(rows * drs), fresh(rows)
        a = src(rows * ars + cols)
        if rng.integers(3) == 0:
            base = int(rng.integers(0, sums + 1))
            prog.extend([I.li(5, base), I.exp2sub_sum(dst, a, b, sums - base, rows, cols, drs,
                                                      ars, brs, bmode, rd=5)])
        else:
            prog.append(I.exp2sub_sum(dst, a, b, sums, rows, cols, drs, ars, brs, bmode))
        return dst, sums, rows, cols, drs

    def take(dst, sums, rows, cols, drs):
        prog.append(I.fsum_take(sums, dst, rows, cols, drs))
        if rng.integers(2):                 # the sums read at once
            prog.append(I.vop(I.V_ADD, fresh(rows), sums, sums, rows, 1, 1, 1, 1))
        rs = fresh(rows)
        prog.append(I.vop(I.V_RSUM, rs, dst, 0, rows, cols, 1, drs, 0))
        pairs.append((sums, rs, rows))

    for _ in range(n_ops):
        if rng.integers(3) == 0:
            prog.append(other([I.V_ADD, I.V_EXP2, I.V_RECIP, I.V_RSQRT, I.V_RSUM, I.V_RDOT]))
            continue
        fs = [fsum()]
        k = int(rng.integers(6))
        if k == 0:                          # the next FSUM before this one's take
            fs.append(fsum())
        elif k == 1:                        # a reduction: it waits for the sums
            prog.append(other([I.V_RSUM, I.V_RDOT, I.V_RMAX]))
        else:                               # elementwise VOPs during the fold and the flush
            for _ in range(k - 2):
                prog.append(other([I.V_ADD, I.V_MUL, I.V_EXP2, I.V_RECIP]))
        for f in fs:
            take(*f)
        assert nxt < (1 << 16)
    prog.append(I.halt())
    return prog, pairs


def _fsum_img(rng):
    """The special values (words 0..4095) and values over EXP2's range (4096..8191)."""
    img = np.zeros(1 << 20, np.uint8)
    img[:4 * 4096] = _special(rng, 4096).view(np.uint8)
    img[4 * 4096:4 * 8192] = f(rng.standard_normal(4096) * np.float32(40)).view(np.uint8)
    return img


def test_fsum_is_rsum_of_the_results():
    """EXP2SUB's row sums (FSUM) are RSUM's of its results, bit for bit, and the results are
    EXP2SUB's without the flag."""
    r = np.random.default_rng(1900)
    cfg = Config(S=1, FSUM=True)
    for _ in range(3):
        prog, pairs = _fsum_program(r)
        img = _fsum_img(r)
        m = Machine(cfg, [prog], [img.copy()]).run().slices[0]
        for sums, rs, rows in pairs:
            assert np.array_equal(m.tmem[sums:sums + rows], m.tmem[rs:rs + rows])
        plain = [replace(i, flags=0) if i.flags & I.F_FSUM else i for i in prog]
        p = Machine(cfg, [plain], [img.copy()]).run().slices[0]
        for sums, rs, rows in pairs:
            assert np.array_equal(p.tmem[rs:rs + rows], m.tmem[rs:rs + rows])


def test_fsum_errors():
    a, b = f(rng.standard_normal(64 * 4)), f(rng.standard_normal(64))
    ok = I.exp2sub_sum(4096, 0, 1024, 8192, 4, 64, 64, 64, 0, I.B_COL)
    tk = I.fsum_take(8192, 4096, 4, 64, 64)
    with pytest.raises(SimError, match="no fused row sums"):
        run1([ok, tk, I.halt()], {0: a, 1024: b})
    cfg = Config(S=1, FSUM=True)
    run1([ok, tk, I.halt()], {0: a, 1024: b}, cfg)
    for kw in ({"cols": 63}, {"rows": 257}, {"bmode": I.B_SCALAR}, {"rows": 0}):
        args = dict(dst=4096, a=0, b=1024, sums=8192, rows=4, cols=64, drs=64, ars=64, brs=0,
                    bmode=I.B_COL) | kw
        with pytest.raises(ValueError):
            I.exp2sub_sum(**args)
    narrow = replace(ok, w=list(ok.w))
    narrow.w[3] = 4 | (32 << 16)                # cols 32
    for bad in (replace(I.vop(I.V_EXP2, 4096, 0, 0, 4, 64, 64, 64, 0), flags=I.F_FSUM), narrow):
        with pytest.raises(SimError, match="FSUM"):
            run1([bad, tk, I.halt()], {0: a, 1024: b}, cfg)
    over = I.exp2sub_sum(4096, 0, 1024, 4096 + 64, 4, 64, 64, 64, 0, I.B_COL)
    with pytest.raises(SimError, match="overlap"):
        run1([over, tk, I.halt()], {0: a, 1024: b}, cfg)
    # every FSUM's sums taken, in order, at their words; nothing to take with no FSUM
    with pytest.raises(SimError, match="not taken"):
        run1([ok, I.halt()], {0: a, 1024: b}, cfg)
    with pytest.raises(SimError, match="no EXP2SUB FSUM"):
        run1([tk, I.halt()], {0: a, 1024: b}, cfg)
    with pytest.raises(SimError, match="takes 4 rows at 8193"):
        run1([ok, I.fsum_take(8193, 4096, 4, 64, 64), I.halt()], {0: a, 1024: b}, cfg)
    # the sums' words are undefined until the take: NaN to a read before it, the sums after
    rd = I.vop(I.V_ADD, 9000, 8192, 8192, 4, 1, 1, 1, 1)
    t = run1([ok, rd, tk, I.halt()], {0: a, 1024: b}, cfg).tmem.view(np.float32)
    assert np.isnan(t[9000:9004]).all() and not np.isnan(t[8192:8196]).any()


@pytest.mark.parametrize("uarch,seed", [({}, 0), ("board", 1), ("cns5", 2), ({"VPU_CNS": 5}, 3)])
def test_fsum_rtl_bit_exact(have_verilator, uarch, seed):
    """FSUM on the RTL (otpu_vpu FSUM, through the sequencer: the sums' footprint) against the
    ISA simulator, TMEM bit for bit; on the board's uarch at CNS 3 and 5."""
    cfg = Config(S=1, FSUM=True)
    uarch = (dict(rtlsim.BOARD_UARCH) if uarch == "board" else
             dict(rtlsim.BOARD_UARCH, VPU_CNS=5) if uarch == "cns5" else dict(uarch))
    r = np.random.default_rng(1950 + seed)
    prog, pairs = _fsum_program(r)
    img = _fsum_img(r)
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], uarch=uarch)
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"
    for sums, rs, rows in pairs:
        assert np.array_equal(tmems[0][sums:sums + rows], tmems[0][rs:rs + rows])


def test_fsum_footprint_has_the_sums():
    """The scoreboard's footprint of an EXP2SUB with FSUM (otpu_pkg footprint, opentpu.waits):
    the results' range and the sums' (w7 + R[rd], a word a row); without the flag, the results'
    alone (w7 is the immediate there). Its take's is RSUM's."""
    R = [(0, 0)] * 16
    R[3] = (100, 100)
    ins = I.exp2sub_sum(4096, 0, 1024, 9000, 5, 64, 70, 64, 0, I.B_COL, rd=3)
    fp = waits.footprint(ins, R)
    assert (waits.SP_TMEM, 4096, 4096 + 4 * 70 + 64) in fp.wr
    assert (waits.SP_TMEM, 9100, 9105) in fp.wr and len(fp.wr) == 2
    plain = waits.footprint(replace(ins, flags=0), R)
    assert plain.wr == ((waits.SP_TMEM, 4096, 4096 + 4 * 70 + 64),)
    # its take's: RSUM's of the results into the sums (the sequencer's fence for them)
    tk = I.fsum_take(9100, 4096, 5, 64, 70)
    fp = waits.footprint(tk, R)
    assert fp == waits.footprint(replace(tk, flags=0), R)
    assert fp.wr == ((waits.SP_TMEM, 9100, 9105),) and \
        fp.rd == ((waits.SP_TMEM, 4096, 4096 + 4 * 70 + 64),)


# ------------------------------------------------------------------------------------ VSIG
LOG2E = float(1.0 / np.log(2.0))


def _with_nans(x, rng, frac=0.01):
    """x with a few NaNs: the canonical one, payloads, the sign set."""
    x = x.copy()
    u = x.view(np.uint32)
    k = rng.random(x.size) < frac
    u[k] = rng.choice(np.array([0x7FC00000, 0x7F800001, 0xFFC00000, 0x7FFFFFFF], np.uint32),
                      int(k.sum()))
    return x


def _sig_chain(fn, dst, a, b, rows, cols, drs, ars, brs, bmode, k, tmp):
    """The VOPs SIGMOID (SILU) is bit for bit: MUL, EXP2, ADD 1.0, RECIP (then MUL by A), in
    dense temps from tmp, the result at dst."""
    n = rows * cols
    t1, t2, t3, t4 = tmp, tmp + n, tmp + 2 * n, tmp + 3 * n
    prog = [I.vop(I.V_MUL, t1, a, b, rows, cols, cols, ars, brs, bmode, imm=k),
            I.vop(I.V_EXP2, t2, t1, 0, rows, cols, cols, cols, 0),
            I.vop(I.V_ADD, t3, t2, 0, rows, cols, cols, cols, 0, I.B_SCALAR, imm=1.0)]
    if fn == I.V_SIGMOID:
        return prog + [I.vop(I.V_RECIP, dst, t3, 0, rows, cols, drs, cols, 0)]
    return prog + [I.vop(I.V_RECIP, t4, t3, 0, rows, cols, cols, cols, 0),
                   I.vop(I.V_MUL, dst, a, t4, rows, cols, drs, ars, cols, I.B_FULL)]


@pytest.mark.parametrize("fn", [I.V_SIGMOID, I.V_SILU])
@pytest.mark.parametrize("bmode", [I.B_FULL, I.B_ROW, I.B_COL, I.B_SCALAR])
def test_sigmoid_silu_are_their_vop_chains(fn, bmode):
    """VOP SIGMOID / SILU (Config.VSIG) against the chain of VOPs they replace, TMEM bit for bit:
    every B mode, B -log2(e), 1.0 and others, data with zeros, denormals, infinities, huge values
    and NaNs; the results also near the logistic function / SiLU."""
    r = np.random.default_rng(2100 + 4 * fn + bmode)
    cfg = Config(S=1, VSIG=True)
    rows, cols, ars, brs = 5, 150, 153, 151
    A = _with_nans(np.concatenate([_special(r, rows * ars), f(r.standard_normal(4096) * 60)]), r)
    for k in (-LOG2E, 1.0, -2.0 * LOG2E / 30.0, 0.0, float(np.inf)):
        prog = [I.vop(fn, 40000, 0, 20000, rows, cols, cols + 1, ars, brs, bmode, imm=k)]
        prog += _sig_chain(fn, 50000, 0, 20000, rows, cols, cols + 1, ars, brs, bmode, k, 60000)
        s = run1(prog + [I.halt()], {0: A, 20000: A[::-1][:rows * brs + cols]}, cfg)
        got = s.tmem[40000:40000 + rows * (cols + 1)]
        assert np.array_equal(got, s.tmem[50000:50000 + rows * (cols + 1)]), k
    x = f(np.linspace(-20, 20, 4001))
    s = run1([I.vop(fn, 10000, 0, 0, 1, 4001, 0, 4001, 0, I.B_SCALAR, imm=-LOG2E), I.halt()],
             {0: x}, cfg)
    sg = 1 / (1 + np.exp(-x.astype(np.float64)))
    want = sg if fn == I.V_SIGMOID else x * sg
    got = s.tget(10000 + np.arange(4001))
    assert np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-3)) < 1e-5


def test_sigmoid_silu_at_k1_skip_the_mul():
    """SIGMOID and SILU with B = 1.0 are the chains without their MUL (lib.silu_l2's x / (1 +
    2^x), gelu_tanh's exponent): EXP2's first step is x * 1 + -0 already. Bit for bit over random
    words (every exponent, NaNs, denormals) and the special values."""
    r = np.random.default_rng(2200)
    x = np.concatenate([r.integers(0, 1 << 32, 1 << 20, dtype=np.uint64).astype(np.uint32),
                        np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000,
                                  0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001,
                                  0x3F800000, 0x43000000, 0xC3000000, 0xC2FC0000], np.uint32)])
    x = x.view(np.float32)
    one = f(1.0)
    sg = F.recip(F.add(F.exp2(x), one))
    assert np.array_equal(F.sigmoid(x, one).view(np.uint32), sg.view(np.uint32))
    assert np.array_equal(F.silu(x, one).view(np.uint32), F.mul(x, sg).view(np.uint32))


def test_vsig_errors():
    """Without VSIG the ISA simulator refuses SIGMOID and SILU, naming them; the compiler emits
    the chains there (lib.sigmoid / silu: no SIGMOID or SILU in the program)."""
    for fn in (I.V_SIGMOID, I.V_SILU):
        with pytest.raises(SimError, match="SIGMOID / SILU"):
            run1([I.vop(fn, 5000, 0, 0, 2, 8, 8, 8, 0, I.B_SCALAR, imm=-LOG2E), I.halt()],
                 {0: f(np.ones(16))})
    comp, _ = compile_kernel(_lib_kernel, Config(S=1), **_lib_args(np.random.default_rng(0)))
    assert not {I.V_SIGMOID, I.V_SILU} & set(_vops(comp.programs[0]))


from opentpu.kernels import lib as L  # noqa: E402


@ol.jit
def _lib_kernel(x, v, o1, o2, o3, o4, o5, o6, o7):
    X, V = ol.load(x), ol.load(v)
    ol.store(o1, L.sigmoid(X))
    ol.store(o2, L.silu(X))
    ol.store(o3, L.silu_l2(X))
    ol.store(o4, L.softcap(X, 30.0))
    ol.store(o5, L.gelu_tanh(X))
    ol.store(o6, L.silu(V) * V)                         # 1-D, then a product (MLP's gate)
    t = ol.empty(X.shape)
    t.set(L.silu(X) * X)                                # retargeted into t
    t.set(L.sigmoid(t))                                 # in place: dst = A
    ol.store(o7, t)


def _lib_args(r):
    x = _with_nans(np.concatenate([_special(r, 6 * 40 - 40), f(r.standard_normal(40) * 30)]), r)
    return dict(x=Input(x.reshape(6, 40)), v=Input(f(r.standard_normal(100) * 4)),
                **{f"o{i}": Output((6, 40)) for i in (1, 2, 3, 4, 5, 7)}, o6=Output((100,)))


def test_lib_vsig_bit_exact():
    """lib.sigmoid, silu, silu_l2, softcap and gelu_tanh on a machine with VSIG: one SIGMOID or
    SILU each in place of the chain (the products and sums around it as before), the outputs bit
    for bit those without it (specials and NaNs among the inputs), in place and retargeted."""
    args = _lib_args(np.random.default_rng(1))
    cfg, vcfg = Config(S=1), Config(S=1, VSIG=True)
    plain = _vops(compile_kernel(_lib_kernel, cfg, **args)[0].programs[0])
    vs = _vops(compile_kernel(_lib_kernel, vcfg, **args)[0].programs[0])
    assert vs.count(I.V_SIGMOID) == 4 and vs.count(I.V_SILU) == 4, vs
    assert I.V_EXP2 not in vs and I.V_RECIP not in vs
    # sigmoid 4 -> 1, silu 5 -> 1, silu_l2 4 -> 1, softcap 6 -> 3, gelu_tanh 8 -> 6, the two
    # silu's 5 -> 1 and t's sigmoid 4 -> 1
    assert len(plain) - len(vs) == 3 + 4 + 3 + 3 + 2 + 4 + 4 + 3, (len(plain), len(vs))
    a = launch(_lib_kernel, cfg, **args).outputs
    b = launch(_lib_kernel, vcfg, **args).outputs
    for k in a:
        assert np.array_equal(a[k].view(np.uint32), b[k].view(np.uint32)), k


VSIG_COLS = [1, 3, 7, 8, 9, 40, 64, 100, 255, 256, 300]


def _vsig_program(rng, n_ops=40):
    """SIGMOID and SILU (every B mode; B -log2(e), 1.0 and others; now and then in place, dst =
    A) among the other composites (EXP2, EXP2SUB, RECIP, RSQRT, LOG2), elementwise VOPs and sums,
    so the composite loop holds SIG chunks beside the others' passes, and VOPs reading a SIG
    result right after it."""
    NDATA = 8192
    prog = [I.ld(0, 0, NDATA)]
    nxt = NDATA

    def fresh(n):
        nonlocal nxt
        a = nxt
        nxt += n + int(rng.integers(0, 3))
        return a

    def src(n):
        return int(rng.integers(0, NDATA - n))

    for _ in range(n_ops):
        kind = rng.choice(["sig", "sig", "sig", "comp", "elem", "red"])
        rows, cols = int(rng.integers(1, 5)), int(rng.choice(VSIG_COLS))
        if kind == "sig":
            fn = int(rng.choice([I.V_SIGMOID, I.V_SILU]))
            bmode = int(rng.integers(0, 4))
            k = float(rng.choice([-LOG2E, -LOG2E, 1.0, -0.0667, 2.5]))
            ars, brs = cols + int(rng.integers(0, 3)), cols + int(rng.integers(0, 2))
            b = src(rows * brs + cols)
            if rng.integers(5) == 0:            # in place
                a = fresh(rows * ars)
                prog.append(I.vop(I.V_COPY, a, src(rows * ars), 0, rows, cols, ars, ars, 0))
                dst, drs = a, ars
            else:
                a, drs = src(rows * ars), cols + int(rng.integers(0, 2))
                dst = fresh(rows * drs)
            prog.append(I.vop(fn, dst, a, b, rows, cols, drs, ars, brs, bmode, imm=k))
            if rng.integers(3) == 0:            # read at once (RAW on the SIG's result)
                prog.append(I.vop(I.V_ADD, fresh(rows * cols), dst, 0, rows, cols, cols, drs, 0,
                                  I.B_SCALAR, imm=0.5))
        elif kind == "comp":
            fn = int(rng.choice([I.V_EXP2, I.V_EXP2SUB, I.V_RECIP, I.V_RSQRT, I.V_LOG2]))
            prog.append(I.vop(fn, fresh(rows * cols), src(rows * cols), src(cols), rows, cols,
                              cols, cols, 0, I.B_COL if fn == I.V_EXP2SUB else I.B_FULL))
        elif kind == "elem":
            fn = int(rng.choice([I.V_ADD, I.V_MUL, I.V_MAX, I.V_COPY]))
            prog.append(I.vop(fn, fresh(rows * cols), src(rows * cols), src(cols), rows, cols,
                              cols, cols, 0, I.B_COL))
        else:
            fn = int(rng.choice([I.V_RSUM, I.V_RSSQ, I.V_RMAX]))
            prog.append(I.vop(fn, fresh(rows), src(rows * cols), 0, rows, cols, 1, cols, 0))
        assert nxt < 60000
    prog.append(I.halt())
    return prog


@pytest.mark.parametrize("uarch,seed", [({"VPU_CNS": 5}, 0), ("cns5", 1), ("cns5", 2),
                                        ({"VPU_CNS": 5}, 3), ({"VPU_CNS": 3}, 4), ("cns3", 5),
                                        ({"VPU_CNS": 3}, 6)])
def test_vsig_rtl_bit_exact(have_verilator, uarch, seed):
    """SIGMOID and SILU on the RTL (otpu_se_comp SIG at VPU_CNS 5 in four passes, at 3 in six,
    through the sequencer) against the ISA simulator, TMEM bit for bit; the defaults' and the
    board's uarch."""
    cfg = Config(S=1, VSIG=True)
    if isinstance(uarch, str):
        uarch = dict(rtlsim.BOARD_UARCH, VPU_CNS=int(uarch[3:]))
    uarch = dict(uarch)
    r = np.random.default_rng(2300 + seed)
    prog = _vsig_program(r)
    img = np.zeros(1 << 20, np.uint8)
    x = np.concatenate([_special(r, 6144), f(r.standard_normal(2048) * 60)])
    img[:4 * 8192] = _with_nans(x[r.permutation(8192)], r).view(np.uint8)
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], uarch=uarch)
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"


# ------------------------------------------------------------------------------------ MAC
VMAC = Config(S=1, VMAC=True)


def _mac_data(r, n):
    """The special values with NaNs among them (the products' and sums' NaN inputs)."""
    x = _special(r, n)
    x[r.random(n) < 0.02] = np.nan
    return x


@pytest.mark.parametrize("bmode,imm", [(I.B_COL, 0.0), (I.B_SCALAR, 0.75), (I.B_SCALAR, -0.0),
                                       (I.B_SCALAR, np.inf), (I.B_SCALAR, 1e-40),
                                       (I.B_SCALAR, np.nan), (I.B_SCALAR, 3e38)])
def test_mac_is_mul_then_add(bmode, imm):
    """VOP MAC (Config.VMAC): dst = add(dst, mul(A, B)) is the MUL into a temp and the ADD of dst
    and it, bit for bit, on +-0, denormals (flushed), +-inf, NaN and huge values (overflow, inf
    - inf), B per column (256 columns: the buffer's) or the immediate."""
    r = np.random.default_rng(2100)
    rows, cols, drs, ars = 6, 256 if bmode == I.B_COL else 100, 259, 257
    D, A, Bv = _mac_data(r, rows * drs), _mac_data(r, rows * ars), _mac_data(r, cols)
    Bv[:4] = [np.nan, -np.inf, 1e-40, -0.0]
    init = {0: D, 4000: A, 8000: Bv}
    mac = [I.mac(0, 4000, 8000, rows, cols, drs, ars, bmode, imm), I.halt()]
    two = [I.vop(I.V_MUL, 20000, 4000, 8000, rows, cols, cols, ars, 0, bmode, imm),
           I.vop(I.V_ADD, 0, 0, 20000, rows, cols, drs, drs, cols), I.halt()]
    got, want = run1(mac, init, VMAC), run1(two, init, VMAC)
    idx = np.arange(rows)[:, None] * drs + np.arange(cols)[None, :]
    assert np.array_equal(got.tmem[idx], want.tmem[idx])
    assert np.array_equal(got.tmem[:20000], want.tmem[:20000])        # padding untouched
    B = Bv[None, :cols] if bmode == I.B_COL else f(imm)
    assert np.array_equal(got.tmem[idx], F.bits(F.add(D[idx], F.mul(A[np.arange(rows)[:, None] * ars
                                                                      + np.arange(cols)], B))))
    assert np.isnan(got.tget(idx)).any() and (got.tmem[idx] == 0x7FC00000).any()


def test_mac_reads_b_before_any_write_and_a_in_place():
    """B per column may overlap dst (the fill reads it into the column buffer first); A may be
    dst itself (each element read before its own write: dst += dst * B); an A row that an
    earlier row wrote is a hazard."""
    r = np.random.default_rng(2101)
    rows, cols = 5, 40
    D, A = _mac_data(r, rows * cols).reshape(rows, cols), _mac_data(r, rows * cols)
    s = run1([I.mac(0, 1000, 2 * cols, rows, cols, cols, cols), I.halt()], {0: D, 1000: A}, VMAC)
    want = F.add(D, F.mul(A.reshape(rows, cols), D[2][None, :]))
    assert np.array_equal(s.tmem[:rows * cols], F.bits(want).reshape(-1))
    s = run1([I.mac(0, 0, 1000, rows, cols, cols, cols), I.halt()], {0: D, 1000: A}, VMAC)
    want = F.add(D, F.mul(D, A[None, :cols]))
    assert np.array_equal(s.tmem[:rows * cols], F.bits(want).reshape(-1))
    s = run1([I.mac(0, cols, 1000, rows - 1, cols, cols, cols), I.halt()], {0: D, 1000: A}, VMAC)
    want = F.add(D[:-1], F.mul(D[1:], A[None, :cols]))                # A one row ahead: fine
    assert np.array_equal(s.tmem[:(rows - 1) * cols], F.bits(want).reshape(-1))
    with pytest.raises(SimError, match="hazard"):                     # A one row behind
        run1([I.mac(cols, 0, 1000, rows - 1, cols, cols, cols), I.halt()], {0: D}, VMAC)


def test_mac_errors():
    """Without Config.VMAC the simulator refuses a MAC, naming it; B neither per column (cols
    <= 256) nor a scalar, and flags, are refused by isa.mac and the simulator."""
    ok = I.mac(4096, 0, 1024, 4, 64, 64, 64)
    with pytest.raises(SimError, match="has no MAC"):
        run1([ok, I.halt()], {})
    run1([ok, I.halt()], {}, VMAC)
    run1([I.mac(4096, 0, 1024, 2, 256, 256, 256), I.halt()], {}, VMAC)
    run1([I.mac(10000, 0, 0, 2, 4000, 4000, 4000, I.B_SCALAR, 2.0), I.halt()], {}, VMAC)
    for kw in ({"bmode": I.B_FULL}, {"bmode": I.B_ROW}, {"cols": 257}):
        args = dict(dst=4096, a=0, b=1024, rows=4, cols=64, drs=300, ars=300) | kw
        with pytest.raises(ValueError, match="MAC"):
            I.mac(**args)
    for bad in (I.vop(I.V_MAC, 4096, 0, 1024, 4, 64, 64, 64, 64, I.B_FULL),
                I.vop(I.V_MAC, 4096, 0, 1024, 4, 64, 64, 64, 1, I.B_ROW),
                I.vop(I.V_MAC, 4096, 0, 1024, 2, 257, 300, 300, 0, I.B_COL),
                replace(ok, flags=I.F_DSCALAR)):
        with pytest.raises(SimError, match="MAC"):
            run1([bad, I.halt()], {}, VMAC)


def test_mac_footprint_reads_and_writes_dst():
    """The scoreboard's footprint of a MAC (otpu_pkg footprint, opentpu.waits): A, B's columns
    (none for a scalar) and dst read, dst written."""
    R = [(0, 0)] * 16
    fp = waits.footprint(I.mac(4096, 0, 1024, 5, 64, 70, 80), R)
    dst = (waits.SP_TMEM, 4096, 4096 + 4 * 70 + 64)
    assert fp.wr == (dst,)
    assert set(fp.rd) == {(waits.SP_TMEM, 0, 4 * 80 + 64), (waits.SP_TMEM, 1024, 1088), dst}
    fp = waits.footprint(I.mac(4096, 0, 0, 5, 64, 70, 80, I.B_SCALAR, 0.5), R)
    assert fp.wr == (dst,) and set(fp.rd) == {(waits.SP_TMEM, 0, 4 * 80 + 64), dst}


def test_language_mac():
    """ol.mac (VOP MAC with Config.VMAC): acc += a * v[None, :] and acc += a * 0.5 in one pass
    each, the words of acc.set(acc + a * v[None, :]) (a MUL and an ADD); without VMAC the
    compiler refuses it, naming it."""
    @ol.jit
    def k(x, a, v, out):
        X, A, V = ol.load(x), ol.load(a), ol.load(v)
        ol.mac(X, A, V[None, :])
        ol.mac(X[1:4, :], A[0:3, :], 0.5)
        ol.store(out, X)

    @ol.jit
    def chain(x, a, v, out):
        X, A, V = ol.load(x), ol.load(a), ol.load(v)
        X.set(X + A * V[None, :])
        X[1:4, :].set(X[1:4, :] + A[0:3, :] * 0.5)
        ol.store(out, X)
    r = np.random.default_rng(2102)
    args = dict(x=Input(_mac_data(r, 6 * 40).reshape(6, 40)),
                a=Input(_mac_data(r, 6 * 40).reshape(6, 40)), v=Input(_mac_data(r, 40)),
                out=Output((6, 40)))
    comp, _ = compile_kernel(k, VMAC, **args)
    assert _vops(comp.programs[0]) == [I.V_MAC, I.V_MAC]
    assert I.V_MAC not in _vops(compile_kernel(chain, VMAC, **args)[0].programs[0])
    got = launch(k, VMAC, **args).outputs["out"]
    want = launch(chain, Config(S=1), **args).outputs["out"]
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
    with pytest.raises(ol.CompileError, match="needs VOP MAC"):
        compile_kernel(k, Config(S=1), **args)


def _mac_program(rng, cfg, n_ops=48):
    """MACs (B per column, at most 256 columns, or a scalar) among other VOPs of other latencies:
    chains of MACs on one dst (the convolution's taps: dst read after the last MAC's write),
    A in place (dst) or one row ahead, B in dst (read by the fill before the first write) or
    loaded by DMA right before the MAC and overwritten right after (the fill's WAR), dst read
    and A rewritten right after it; the data the special values with NaNs."""
    NDATA = 8192
    prog = [I.ld(0, 0, NDATA)]
    nxt = NDATA
    scratch = 4 * NDATA

    def fresh(n):
        nonlocal nxt
        a = nxt
        nxt += n + int(rng.integers(0, 3))
        return a

    def src(n):
        return int(rng.integers(0, NDATA - n))

    kinds = ["mac", "mac", "chain", "inplace", "ld_mac", "old"] + (["outer"] if cfg.OUTER else [])
    for _ in range(n_ops):
        kind = str(rng.choice(kinds))
        rows, cols = int(rng.integers(1, 7)), int(rng.choice([1, 7, 8, 9, 40, 128, 200, 255, 256]))
        drs, ars = cols + int(rng.integers(0, 3)), cols + int(rng.integers(0, 3))
        if kind == "old":
            func = int(rng.choice([I.V_ADD, I.V_MUL, I.V_EXP2, I.V_RECIP, I.V_RSUM, I.V_COPY]))
            rows, cols = int(rng.integers(1, 4)), int(rng.integers(1, 60))
            if func in I.REDUCE:
                prog.append(I.vop(func, fresh(rows), src(rows * cols), 0, rows, cols, 1, cols, 0))
            else:
                prog.append(I.vop(func, fresh(rows * cols), src(rows * cols), src(cols), rows,
                                  cols, cols, cols, 0, I.B_COL))
            continue
        if kind == "outer":
            prog.append(I.outer(fresh(rows * drs), src(cols), src(rows), src(cols), rows, cols,
                                drs, 1, str(rng.choice(["scalar", "column", "one"]))))
            continue
        dst = fresh(rows * drs)
        prog.append(I.vop(I.V_COPY, dst, src(rows * cols), 0, rows, cols, drs, cols, 0))
        for _ in range(int(rng.integers(2, 5)) if kind == "chain" else 1):
            sc = rng.integers(4) == 0
            bm = I.B_SCALAR if sc else I.B_COL
            imm = float(rng.choice(_mac_data(rng, 32)))
            b, a, ar = src(cols), src(rows * ars), ars
            if kind == "inplace":
                if rows > 1 and rng.integers(2):        # A one row ahead of dst
                    a, ar, rows_m = dst + drs, drs, rows - 1
                else:
                    a, ar, rows_m = dst, drs, rows
            else:
                rows_m = rows
            if not sc and rng.integers(3) == 0:         # B in dst: read before the first write
                b = dst + int(rng.integers(0, rows)) * drs
            if kind == "ld_mac" and not sc:             # B arrives by DMA right before
                b = fresh(cols)
                prog.append(I.ld(4 * int(rng.integers(0, NDATA - cols)), b, cols))
            prog.append(I.mac(dst, a, b, rows_m, cols, drs, ar, bm, imm))
            if kind == "ld_mac" and not sc:             # ... and is overwritten right after
                prog.append(I.vop(I.V_FILL, b, 0, 0, 1, cols, 0, 0, 0, I.B_SCALAR, 1.5))
        if rng.integers(2):                             # dst read right after (RAW)
            prog.append(I.vop(I.V_ADD, fresh(rows * cols), dst, src(cols), rows, cols, cols, drs,
                              0, I.B_COL))
        if rng.integers(2):
            prog.append(I.st(scratch, dst, rows * drs))
            scratch += 4 * rows * drs + 64
        assert nxt < cfg.TMEM_WORDS and scratch < cfg.DRAM_BYTES
    prog.append(I.halt())
    assert I.V_MAC in _vops(prog)
    return prog


@pytest.mark.parametrize("lanes,uarch,outer,seed", [(8, {}, True, 0), (8, "board", False, 1),
                                                    (8, "cns5", False, 2), (8, "board", True, 3),
                                                    (16, {}, False, 4), (4, {}, True, 5)])
def test_mac_rtl_bit_exact(have_verilator, lanes, uarch, outer, seed):
    """MAC on the RTL (otpu_vpu MAC, through the sequencer: dst read and written, B's columns)
    against the ISA simulator, TMEM and DRAM bit for bit: with and without OUTER (the C buffer
    MAC's alone), the board's uarch at CNS 3 and 5, the long lanes (LANES 16 and 4)."""
    cfg = Config(S=1, LANES=lanes, MCOLS=min(8, lanes), OUTER=outer, VMAC=True)
    uarch = (dict(rtlsim.BOARD_UARCH) if uarch == "board" else
             dict(rtlsim.BOARD_UARCH, VPU_CNS=5) if uarch == "cns5" else dict(uarch))
    r = np.random.default_rng(2200 + seed)
    prog = _mac_program(r, cfg)
    img = np.zeros(1 << 20, np.uint8)
    img[:4 * 8192] = _mac_data(r, 8192).view(np.uint8)
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], uarch=uarch)
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"
    assert np.array_equal(drams[0], m.slices[0].dram)
