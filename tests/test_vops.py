"""The linear-recurrence VOPs (docs/isa.md): RDOT, OUTER and LOG2.

ISA simulator vs float64 references, the LOG2 error bound, and RTL vs ISA simulator bit for bit
(random tiles with zeros, negatives, denormals and infinities)."""
import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I, rtlsim
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
    before it and VOPs rewriting them right after (the scoreboard's C and D ranges)."""
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
        kind = rng.choice(["rdot", "outer", "outer", "log2", "old", "ld_outer"])
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
