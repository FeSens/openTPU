"""STREAM (docs/stream.md): the stream engine's instruction at the ISA level. DSTEP is STREAM with
isa.gdn_desc; every linear-recurrence family of section 5.2 matches its VOP fallback bit for
bit; RMSNorm and attention's reductions as TMEM streams match the VOPs; descriptors are
float-safe and isa.stream_hw_cfg picks the board's subset."""
import dataclasses

import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I
from opentpu import language as ol
from opentpu.isasim import Machine, SimError
from opentpu.kernels.lib import rmsnorm
from opentpu.runtime import Input, Output, compile_kernel, launch

from test_vops import _dstep_as_vops, _dstep_case, _run_dram, _special

CFG = Config(S=1, D=128, DRAM_BYTES=1 << 20)
DESC = 60000                          # where the ISA-level tests put a descriptor


def f(x):
    return np.asarray(x, np.float32)


def _with_desc(tm, d, at=DESC):
    tm = dict(tm)
    tm[at] = np.array(d.words(), np.uint32).view(np.float32)
    return tm


# ------------------------------------------------------------------------ DSTEP = STREAM(gdn)
@pytest.mark.parametrize("rows,cols,special,zero", [(128, 128, False, False),
                                                    (64, 128, True, False), (7, 64, True, False),
                                                    (256, 64, False, False),
                                                    (32, 256, True, False), (3, 192, True, False),
                                                    (128, 128, False, True), (5, 64, True, True)])
def test_stream_gdn_is_dstep_and_the_vops(rows, cols, special, zero):
    """STREAM with gdn_desc = DSTEP = RDOT, MUL, SUB, MUL, OUTER, RDOT, bit for bit (ZERO:
    the state reads as +0)."""
    dram, tm = _dstep_case(np.random.default_rng(rows * 1000 + cols), rows, cols, special)
    a = _run_dram([I.dstep(0x1000, 100, 700, rows, cols, 1000, 2, 4000, zero=zero), I.halt()],
                  dram, tm, CFG)
    b = _run_dram([I.stream(DESC, 0x1000, 0x1000, 100, 700, 1000, 4000, ks=2, zero=zero),
                   I.halt()], dram, _with_desc(tm, I.gdn_desc(rows, cols)), CFG)
    n = rows * cols * 4
    assert np.array_equal(a.dram[0x1000:0x1000 + n], b.dram[0x1000:0x1000 + n])
    assert np.array_equal(a.tmem[4000:4000 + rows], b.tmem[4000:4000 + rows])
    if not zero:
        c = _run_dram(_dstep_as_vops(rows, cols), dram, tm, CFG)
        assert np.array_equal(c.dram[0x1000:0x1000 + n], b.dram[0x1000:0x1000 + n])
        assert np.array_equal(c.tmem[4000:4000 + rows], b.tmem[4000:4000 + rows])


def test_stream_registers_relative():
    """src/dst + R[ra], vec + R[rb], x + R[rc], k + R[rd]."""
    rows, cols = 16, 64
    dram, tm = _dstep_case(np.random.default_rng(5), rows, cols)
    a = _run_dram([I.dstep(0x1000, 100, 700, rows, cols, 1000, 2, 4000), I.halt()], dram, tm,
                  CFG)
    prog = [I.li(1, 0x800), I.li(2, 50), I.li(3, 300), I.li(4, 999),
            I.stream(DESC, 0x800, 0x800, 50, 400, 1, 4000, ks=2, ra=1, rb=2, rc=3, rd=4),
            I.halt()]
    b = _run_dram(prog, dram, _with_desc(tm, I.gdn_desc(rows, cols)), CFG)
    n = rows * cols * 4
    assert np.array_equal(a.dram[0x1000:0x1000 + n], b.dram[0x1000:0x1000 + n])
    assert np.array_equal(a.tmem[4000:4000 + rows], b.tmem[4000:4000 + rows])


# ------------------------------------------------------------------------ descriptors
def _rand_desc(rng):
    ops = tuple((int(rng.integers(0, 11)), int(rng.integers(0, 8)), int(rng.integers(0, 16)),
                 int(rng.integers(0, 16))) for _ in range(int(rng.integers(0, 16))))
    return I.StreamDesc(
        rows=int(rng.integers(1, 4096)), cols=int(rng.integers(1, 4096)),
        a_en=bool(rng.integers(2)), a_op=int(rng.integers(4)), a_idx=int(rng.integers(4)),
        u_mode=int(rng.integers(4)), g_src=int(rng.integers(4)), g_idx=int(rng.integers(8)),
        b_src=int(rng.integers(3)), b_idx=int(rng.integers(8)), d_reg=int(rng.integers(8)),
        q_en=bool(rng.integers(2)), q_idx=int(rng.integers(4)), q_self=bool(rng.integers(2)),
        out_p2=bool(rng.integers(2)), o_reg_en=bool(rng.integers(2)), o_reg=int(rng.integers(8)),
        rinit=int(rng.integers(16)), rsave=int(rng.integers(16)), srs=int(rng.integers(4096)),
        drs=int(rng.integers(4096)), ops=ops)


def test_descriptor_round_trip_and_float_safe():
    """Every descriptor word is a normal fp32 (exponent field 0x80): FILL writes it exactly
    (FTZ changes nothing), and decode(words()) gives the descriptor back."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        d = _rand_desc(rng)
        w = np.array(d.words(), np.uint32)
        assert np.all((w >> 23) & 0xFF == 0x80)
        assert np.array_equal(F.ftz(w.view(np.float32)).view(np.uint32), w)
        assert I.StreamDesc.decode(w) == d
    d = I.gdn_desc(128, 128)
    w = d.words()
    prog = [I.vop(I.V_FILL, 500 + i, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR,
                  imm=np.uint32(x).view(np.float32)) for i, x in enumerate(w)] + [I.halt()]
    s = Machine(CFG, [prog], [None]).run().slices[0]
    assert list(s.tmem[500:500 + len(w)]) == w
    with pytest.raises(ValueError, match="float-safe"):
        I.StreamDesc.decode([0x3F800000] + w[1:])


def test_stream_bad_descriptor_is_an_error():
    rows, cols = 4, 64
    dram, tm = _dstep_case(np.random.default_rng(1), rows, cols)
    tm = _with_desc(tm, I.gdn_desc(rows, cols))
    with pytest.raises(SimError, match="chunk aligned"):
        _run_dram([I.stream(DESC, 0x1040, 0x1040, 100, 700, 1000, 4000, ks=2), I.halt()], dram,
                  tm, CFG)
    tm[DESC + 1] = f([1.0])                                    # not float-safe
    with pytest.raises(SimError, match="STREAM"):
        _run_dram([I.stream(DESC, 0x1000, 0x1000, 100, 700, 1000, 4000, ks=2), I.halt()], dram,
                  tm, CFG)


def test_stream_hw_cfg():
    """The board's subset (docs/stream.md, 4.4): the state step's four scalar programs, three
    gates and two A vectors, DRAM in place, rows <= 256, cols 64..256; nothing else."""
    g = I.stream_hw_cfg(I.gdn_desc(128, 128))
    assert g == {"ns": 16, "rows": 128, "a_en": True, "a_sel": 0, "dmode": 0, "g_src": 0,
                 "q_en": True}
    for dp, dm, uses_a in ((I.D_DELTA, 0, True), (I.D_DELTA1, 1, True),
                           (I.D_SCALE, 2, False), (I.D_DOT, 3, True)):
        for gate, gs in (("const", 0), ("col", 1), ("one", 2)):
            for a_idx in (1, 2):
                for q in (True, False):
                    d = I.state_desc(256, 64, dp, a_idx, gate, q)
                    h = I.stream_hw_cfg(d)
                    assert h == {"ns": 8, "rows": 256, "a_en": uses_a,
                                 "a_sel": int(uses_a and a_idx == 2), "dmode": dm, "g_src": gs,
                                 "q_en": q}
    d = I.gdn_desc(128, 128)
    rep = dataclasses.replace
    for bad, fl in ((d, I.F_SRC_T), (d, I.F_DST_T), (d, I.F_NODST), (rep(d, rows=257), 0),
                    (rep(d, cols=96), 0), (rep(d, cols=320), 0), (rep(d, q_self=True), 0),
                    (rep(d, rinit=1), 0), (rep(d, rsave=2), 0), (rep(d, out_p2=True), 0),
                    (rep(d, srs=4), 0), (rep(d, u_mode=I.U_ADD), 0),
                    (rep(d, b_src=I.B_CONST), 0), (rep(d, b_idx=2), 0), (rep(d, d_reg=1), 0),
                    (rep(d, g_src=I.G_SLOT, g_idx=2), 0), (rep(d, g_src=I.G_REG), 0),
                    (rep(d, q_idx=1), 0), (rep(d, a_idx=3), 0), (rep(d, a_en=False), 0),
                    (rep(d, a_op=I.A_SELF), 0), (rep(d, ops=d.ops[:2]), 0),
                    (rep(d, ops=((I.SC_ADD, 2, I.OP_X, I.OP_K0 + 1),)), 0)):
        assert I.stream_hw_cfg(bad, fl) is None, bad


# ------------------------------------------------------------------------ the families
@ol.jit
def _family(state, vec, x, kk, state_out, out, mode="delta", a_slot=1, gate="const",
            with_o=True, zero=False, no_k0=False):
    """The state step of one head: the state is copied to state_out, stepped there in place."""
    rows, cols = state.shape
    t = ol.load(state)
    ol.store(state_out, t)
    del t
    V, X, K = ol.load(vec), ol.load(x), ol.load(kk)
    o = ol.empty([rows]) if with_o else None
    ol.state_step(state_out, V, X, None if no_k0 else K[0:1], K[1:2], o, mode=mode,
                  a_slot=a_slot, gate=gate, zero=zero)
    ol.store(out, o if with_o else ol.zeros([rows]))


FAMILIES = {                  # name: (mode, a_slot, gate), docs/stream.md 5.2
    "gated_deltanet": ("delta", 1, "const"),
    "deltanet": ("delta1", 1, "one"),
    "kda": ("delta1", 2, "col"),
    "gla": ("scale", 1, "col"),
    "retnet": ("scale", 1, "const"),
    "mamba2": ("scale", 1, "const"),
    "rwkv7_pass1": ("dot", 2, "col"),
    "rwkv7_pass2": ("scale", 1, "one"),
}


def _family_case(rng, rows, cols, name):
    S = f(0.3 * rng.standard_normal((rows, cols)))
    vec = f(rng.standard_normal(4 * cols))
    vec[3 * cols:] = f(rng.uniform(0.3, 1.0, cols))           # a decay column
    if name == "kda":
        vec[2 * cols:3 * cols] = F.mul(vec[3 * cols:], vec[cols:2 * cols])   # alpha * k
    x = f(rng.standard_normal(rows))
    kk = f([rng.uniform(0.3, 1.0), rng.uniform(0.1, 1.0) if name != "rwkv7_pass1" else -1.0])
    return S, vec, x, kk


def _family_f64(S, vec, x, kk, mode, a_slot, gate):
    S, vec, x = S.astype(float), vec.astype(float), x.astype(float)
    cols = S.shape[1]
    slot = [vec[i * cols:(i + 1) * cols] for i in range(4)]
    k0, k1 = float(kk[0]), float(kk[1])
    A = S @ slot[a_slot]
    d = {"delta": (x - A * k0) * k1, "delta1": (x - A) * k1, "scale": x * k1,
         "dot": A * k1}[mode]
    G = {"const": k0, "col": slot[3][None, :], "one": 1.0}[gate]
    S = S * G + np.outer(d, slot[1])
    return S, S @ slot[0]


@pytest.mark.parametrize("name", list(FAMILIES))
@pytest.mark.parametrize("rows,cols,with_o", [(128, 128, True), (7, 64, False)])
def test_family_stream_is_its_vop_fallback(name, rows, cols, with_o):
    """Each family's step as one STREAM (Config.STREAM) = its VOP fallback, bit for bit, and
    close to the float64 recurrence."""
    mode, a_slot, gate = FAMILIES[name]
    rng = np.random.default_rng(abs(hash(name)) % 1000 + rows)
    S, vec, x, kk = _family_case(rng, rows, cols, name)
    no_k0 = mode != "delta" and gate != "const"
    args = dict(state=Input(S), vec=Input(vec), x=Input(x), kk=Input(kk),
                state_out=Output((rows, cols)), out=Output((rows,)), mode=mode, a_slot=a_slot,
                gate=gate, with_o=with_o, no_k0=no_k0)
    scfg = dataclasses.replace(CFG, STREAM=True)
    comp, _ = compile_kernel(_family, scfg, **args)
    assert sum(i.op == I.STREAM for i in comp.programs[0]) == 1
    comp, _ = compile_kernel(_family, CFG, **args)
    assert not any(i.op == I.STREAM for i in comp.programs[0])
    a = launch(_family, scfg, **args).outputs
    b = launch(_family, CFG, **args).outputs
    assert np.array_equal(a["state_out"].view(np.uint32), b["state_out"].view(np.uint32))
    if with_o:
        assert np.array_equal(a["out"].view(np.uint32), b["out"].view(np.uint32))
    Sw, ow = _family_f64(S, vec, x, kk, mode, a_slot, gate)
    assert np.max(np.abs(a["state_out"] - Sw)) < 1e-5 * np.abs(Sw).max()
    if with_o:
        assert np.max(np.abs(a["out"] - ow)) < 1e-5 * np.abs(ow).max()


def test_family_zero_state():
    """zero: the state reads as +0 on both paths."""
    rows, cols = 16, 64
    S, vec, x, kk = _family_case(np.random.default_rng(9), rows, cols, "gla")
    args = dict(state=Input(S), vec=Input(vec), x=Input(x), kk=Input(kk),
                state_out=Output((rows, cols)), out=Output((rows,)), mode="scale", gate="col",
                zero=True)
    a = launch(_family, dataclasses.replace(CFG, STREAM=True), **args).outputs
    b = launch(_family, CFG, **args).outputs
    assert np.array_equal(a["state_out"].view(np.uint32), b["state_out"].view(np.uint32))
    assert np.array_equal(a["out"].view(np.uint32), b["out"].view(np.uint32))


def test_state_step_descriptors_are_filled_at_the_start():
    """The compiler writes each distinct descriptor once, by FILLs before anything else, in
    the reserved words at the top of TMEM."""
    rows, cols = 16, 64
    S, vec, x, kk = _family_case(np.random.default_rng(2), rows, cols, "retnet")
    args = dict(state=Input(S), vec=Input(vec), x=Input(x), kk=Input(kk),
                state_out=Output((rows, cols)), out=Output((rows,)), mode="scale")
    comp, _ = compile_kernel(_family, dataclasses.replace(CFG, STREAM=True), **args)
    prog = comp.programs[0]
    st = next(i for i in prog if i.op == I.STREAM)
    d = I.state_desc(rows, cols, I.D_SCALE, 1, "const", True)
    n = d.nwords()
    assert st.w[0] == CFG.TMEM_WORDS - I.STREAM_DESC_AREA | 1 << 16
    fills = [i for i in prog[:n + 4] if i.op == I.VOP and (i.w[5] >> 16) & 0xFF == I.V_FILL
             and i.w[0] >= st.w[0] & 0xFFFF]
    assert [i.w[6] for i in fills] == d.words()


# ------------------------------------------------------------------------ TMEM streams
@ol.jit
def _rmsnorm_k(x, gamma, y, eps=1e-6):
    ol.store(y, rmsnorm(ol.load(x), ol.load(gamma), eps))


@pytest.mark.parametrize("M,H", [(4, 256), (3, 1024)])
def test_rmsnorm_as_one_tmem_stream(M, H):
    """RMSNorm in one pass (ISA level; not in the v1 hardware): A = RSSQ (S . S), r2 =
    rsqrt(A * (1/H) + eps), Y = (S * r2) * gamma -- lib.rmsnorm's VOPs, bit for bit."""
    rng = np.random.default_rng(M * H)
    xv, gv, eps = f(rng.standard_normal((M, H))), f(rng.standard_normal(H)), 1e-6
    want = launch(_rmsnorm_k, Config(S=1), x=Input(xv), gamma=Input(gv), y=Output((M, H)),
                  eps=eps).outputs["y"]
    d = I.StreamDesc(M, H, a_en=True, a_op=I.A_SELF, u_mode=I.U_MUL, g_src=I.G_REG, g_idx=2,
                     out_p2=True, q_idx=0,
                     ops=((I.SC_MUL, 0, I.OP_A, I.OP_K0), (I.SC_ADD, 1, 0, I.OP_K0 + 1),
                          (I.SC_RSQRT, 2, 1, 0)))
    X0, G0, K0, Y0 = 0, 8000, 12000, 20000
    tm = _with_desc({X0: xv, G0: gv, K0: f([1.0 / H, eps])}, d)
    prog = [I.stream(DESC, X0, Y0, G0, 0, K0, 0, src_t=True, dst_t=True), I.halt()]
    s = _run_dram(prog, None, tm, CFG)
    got = s.tget(Y0 + np.arange(M * H)).reshape(M, H)
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))


def test_attention_reductions_as_tmem_streams():
    """Attention decode's two reductions (ISA level; decode attention stays on the MXU):
    scores s = (K[t] . q) * scale and o = V^T[j] . p as streams, against RDOT + MUL."""
    rng = np.random.default_rng(4)
    T, d = 40, 128
    K, q, Vt = f(rng.standard_normal((T, d))), f(rng.standard_normal(d)), \
        f(rng.standard_normal((d, T)))
    p = f(rng.uniform(0, 1, T))
    scale = float(f(d ** -0.5))
    K0, Q0, V0, P0, C0 = 0, 6000, 7000, 13000, 14000
    tm = {K0: K, Q0: q, V0: Vt, P0: p, C0: f([scale])}
    vops = [I.vop(I.V_RDOT, 20000, K0, Q0, T, d, 1, d, 0, I.B_COL),
            I.vop(I.V_MUL, 20000, 20000, 0, 1, T, 0, T, 0, I.B_SCALAR, imm=scale),
            I.vop(I.V_RDOT, 21000, V0, P0, d, T, 1, T, 0, I.B_COL), I.halt()]
    a = _run_dram(vops, None, tm, CFG)
    ds = I.StreamDesc(T, d, a_en=True, a_op=I.A_SLOT, a_idx=0, o_reg_en=True, o_reg=0,
                      ops=((I.SC_MUL, 0, I.OP_A, I.OP_K0),))
    dv = I.StreamDesc(d, T, q_en=True, q_idx=0)                # O = S . p (Y = S)
    tm = _with_desc(_with_desc(tm, ds), dv, DESC + 32)
    prog = [I.stream(DESC, K0, 0, Q0, 0, C0, 20000, src_t=True, nodst=True),
            I.stream(DESC + 32, V0, 0, P0, 0, 0, 21000, src_t=True, nodst=True), I.halt()]
    b = _run_dram(prog, None, tm, CFG)
    for at, n in ((20000, T), (21000, d)):
        assert np.array_equal(a.tmem[at:at + n], b.tmem[at:at + n])


def test_carried_registers_and_saved_constants():
    """Registers carry from row to row: r4 = max(r4, rowmax) from K0 (rinit), O = r4, saved
    back to K0 (rsave); r5 = r5 + S[r] . slot 0 (a running dot)."""
    rng = np.random.default_rng(6)
    R, C = 12, 64
    S, v = _special(rng, R * C).reshape(R, C), f(rng.standard_normal(C))
    S[np.isnan(S)] = 0
    S[np.isinf(S)] = 0
    k0 = f([-3.0])
    tm0 = {0: S, 5000: v, 6000: k0}
    dmax = I.StreamDesc(R, C, a_en=True, a_op=I.A_MAX, o_reg_en=True, o_reg=4, rinit=1,
                        rsave=1, ops=((I.SC_MAX, 4, 4, I.OP_A),))
    dsum = I.StreamDesc(R, C, a_en=True, a_op=I.A_SLOT, a_idx=0, o_reg_en=True, o_reg=5,
                        rsave=2, ops=((I.SC_ADD, 5, 5, I.OP_A),))
    tm = _with_desc(_with_desc(tm0, dmax), dsum, DESC + 32)
    prog = [I.stream(DESC, 0, 0, 5000, 0, 6000, 7000, src_t=True, nodst=True),
            I.stream(DESC + 32, 0, 0, 5000, 0, 6000, 8000, ks=4, src_t=True, nodst=True),
            I.halt()]
    s = _run_dram(prog, None, tm, CFG)
    m, want = k0, []
    for r in range(R):
        m = F.fmax(m, F.chain_max(S[r:r + 1]))
        want.append(m[0])
    assert np.array_equal(s.tget(7000 + np.arange(R)), f(want))
    assert s.tget(6000) == f(want[-1])
    acc, want = f([0.0]), []
    for r in range(R):
        acc = F.add(acc, F.rdot(S[r:r + 1], v[None, :]))
        want.append(acc[0])
    assert np.array_equal(s.tget(8000 + np.arange(R)).view(np.uint32), f(want).view(np.uint32))
    assert s.tget(6004).view(np.uint32) == f(want[-1]).view(np.uint32)


def test_rows_read_earlier_rows_writes():
    """A TMEM stream whose output row r is input row r + 1: rows run in order, each reading
    what the row before wrote (Y = S * K0: row r + 1 = S0 * K0^(r+1), rounded each time)."""
    rng = np.random.default_rng(8)
    R, C = 6, 64
    S0, g = f(rng.standard_normal(C)), f([0.75])
    d = I.StreamDesc(R, C, u_mode=I.U_MUL, g_src=I.G_CONST, g_idx=0)
    tm = _with_desc({0: S0, 9000: g}, d)
    s = _run_dram([I.stream(DESC, 0, C, 0, 0, 9000, 0, src_t=True, dst_t=True), I.halt()],
                  None, tm, CFG)
    y = S0
    for r in range(1, R + 1):
        y = F.mul(y, g)
        assert np.array_equal(s.tget(r * C + np.arange(C)).view(np.uint32), y.view(np.uint32))


def test_every_vop_family_maps_to_a_descriptor():
    """docs/stream.md 5.1 at the ISA level: ADD/SUB/RSUB/MUL with a column, RSUM, RSSQ, RMAX
    and OUTER as streams equal the VOPs."""
    rng = np.random.default_rng(10)
    R, C = 9, 64
    A, B = f(rng.standard_normal((R, C))), f(rng.standard_normal(C))
    x, dcol = f(rng.standard_normal(R)), f(rng.uniform(0.2, 1, C))
    tm = {0: A, 1000: B, 1000 + C: dcol, 2000: x, 3000: f([1.0, -1.0, 0.0, 0.0])}
    cases = [   # (VOP, descriptor, stream args(vec, x, k), writes rows (Y) or O)
        (I.vop(I.V_ADD, 5000, 0, 1000, R, C, C, C, 0, I.B_COL),
         I.StreamDesc(R, C, u_mode=I.U_ADD, b_src=I.B_SLOT, b_idx=0, d_reg=0,
                      ops=((I.SC_MOV, 0, I.OP_ONE, 0),)), "Y"),
        (I.vop(I.V_SUB, 5000, 0, 1000, R, C, C, C, 0, I.B_COL),
         I.StreamDesc(R, C, u_mode=I.U_ADD, b_src=I.B_SLOT, b_idx=0, d_reg=0,
                      ops=((I.SC_MOV, 0, I.OP_K0 + 1, 0),)), "Y"),
        (I.vop(I.V_RSUB, 5000, 0, 1000, R, C, C, C, 0, I.B_COL),
         I.StreamDesc(R, C, u_mode=I.U_FMMA, g_src=I.G_CONST, g_idx=1, b_src=I.B_SLOT,
                      b_idx=0, d_reg=0, ops=((I.SC_MOV, 0, I.OP_ONE, 0),)), "Y"),
        (I.vop(I.V_MUL, 5000, 0, 1000, R, C, C, C, 0, I.B_COL),
         I.StreamDesc(R, C, u_mode=I.U_MUL, g_src=I.G_SLOT, g_idx=0), "Y"),
        (I.vop(I.V_ADD, 5000, 0, 2000, R, C, C, C, 1, I.B_ROW),
         I.StreamDesc(R, C, u_mode=I.U_ADD, b_src=I.B_CONST, b_idx=0, d_reg=0,
                      ops=((I.SC_MOV, 0, I.OP_X, 0),)), "Y"),
        (I.vop(I.V_RSUM, 5000, 0, 0, R, C, 1, C, 0),
         I.StreamDesc(R, C, a_en=True, a_op=I.A_CONST, a_idx=0, o_reg_en=True,
                      ops=((I.SC_MOV, 0, I.OP_A, 0),)), "O"),
        (I.vop(I.V_RSSQ, 5000, 0, 0, R, C, 1, C, 0),
         I.StreamDesc(R, C, a_en=True, a_op=I.A_SELF, o_reg_en=True,
                      ops=((I.SC_MOV, 0, I.OP_A, 0),)), "O"),
        (I.vop(I.V_RMAX, 5000, 0, 0, R, C, 1, C, 0),
         I.StreamDesc(R, C, a_en=True, a_op=I.A_MAX, o_reg_en=True,
                      ops=((I.SC_MOV, 0, I.OP_A, 0),)), "O"),
    ]
    for vop, d, kind in cases:
        a = _run_dram([vop, I.halt()], None, tm, CFG)
        if kind == "Y":
            st = I.stream(DESC, 0, 5000, 1000, 2000, 3000, 0, src_t=True, dst_t=True)
            n = R * C
        else:
            st = I.stream(DESC, 0, 0, 1000, 2000, 3000, 5000, src_t=True, nodst=True)
            n = R
        b = _run_dram([st, I.halt()], None, _with_desc(tm, d), CFG)
        assert np.array_equal(a.tmem[5000:5000 + n], b.tmem[5000:5000 + n]), vop
    # OUTER with a column decay: T[dst] = T[dst] * Dv + x[r] * Cv, in place
    a = _run_dram([I.outer(0, 1000 + C, 2000, 1000, R, C, C, 1, "column"), I.halt()], None, tm,
                  CFG)
    d = I.StreamDesc(R, C, u_mode=I.U_FMMA, g_src=I.G_SLOT, g_idx=1, b_src=I.B_SLOT, b_idx=0,
                     d_reg=0, ops=((I.SC_MOV, 0, I.OP_X, 0),))
    b = _run_dram([I.stream(DESC, 0, 0, 1000, 2000, 3000, 0, src_t=True, dst_t=True),
                   I.halt()], None, _with_desc(tm, d), CFG)
    assert np.array_equal(a.tmem[:R * C], b.tmem[:R * C])
