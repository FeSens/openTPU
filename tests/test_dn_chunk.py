"""The chunked DeltaNet core (opentpu/kernels/dn_chunk.py) on the ISA simulator at
bcu1525-x2:wmx's arch (one slice): a pass's o and final states against the float64 recurrence
from the same initial states, to dn_chunk -o's accuracy (all split but M1)."""
import dataclasses

import numpy as np
import pytest

from opentpu import isasim as IS
from opentpu import language as ol
from opentpu import target as T
from opentpu.compiler import Affine, RunVar, Tensor, arg_words
from opentpu.kernels import dn_chunk as DC

D = 128
DNQ, DNO, STATE, SCR = 0x100000, 0x800000, 0xA00000, 0xC00000


def _cfg():
    return dataclasses.replace(T.config("bcu1525-x2:wmx"), S=1, DRAM_BYTES=1 << 25)


def _data(rng, npairs, R, zero_state=False):
    nl = 2 * npairs

    def l2n(x):
        return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)

    q = (l2n(rng.standard_normal((npairs, R, D))) * D ** -0.5).astype(np.float32)
    k = l2n(rng.standard_normal((npairs, R, D))).astype(np.float32)
    v = (rng.standard_normal((nl, R, D)) * 0.5).astype(np.float32)
    lg = (-np.exp(rng.uniform(-5, 0.5, (nl, R)))).astype(np.float32)        # log2 decays
    beta = (1 / (1 + np.exp(-rng.standard_normal((nl, R))))).astype(np.float32)
    st = np.zeros((nl, D, D), np.float32) if zero_state else \
        (rng.standard_normal((nl, D, D)) * 0.05).astype(np.float32)
    st = IS.st16_f32(IS.st16_pack(st, "e6m9"), "e6m9").astype(np.float32)    # (the slot's)
    return q, k, v, lg, beta, st


def _ref(q, k, v, lg, beta, st):
    """The recurrence in float64 (DSTEP's order): o [R, nl, dv] and the states St [nl, dv, dk]."""
    nl, R = v.shape[0], v.shape[1]
    S = np.swapaxes(st.astype(np.float64), 1, 2).copy()        # S [nl, dk, dv]
    o = np.zeros((R, nl, D))
    for t in range(R):
        for j in range(nl):
            kj, qj = k[j // 2, t].astype(np.float64), q[j // 2, t].astype(np.float64)
            Sj = S[j] * 2.0 ** float(lg[j, t])
            delta = (v[j, t] - kj @ Sj) * float(beta[j, t])
            Sj = Sj + np.outer(kj, delta)
            o[t, j] = qj @ Sj
            S[j] = Sj
    return o, np.swapaxes(S, 1, 2)


def _run(cfg, q, k, v, lg, beta, st, R, C, split1=False, st16=True, r0=None, rtl=None,
         between=None, info=None, fit="next"):
    """The kernel on one slice: o [R, nl, d] and the states [nl, dv, dk] after the pass (r0: the
    run argument "row", the pass's first row). rtl: a micro-architecture to run it on the RTL too
    (rtlsim), whose DRAM and TMEM must be the ISA's bit for bit. between: chunked's hook (None:
    the argument not given). info: a dict given the trace's TMEM peak and the ISA run's TMEM.
    fit: the trace's TMEM placement (Kernel.trace)."""
    npairs, Rmax = q.shape[0], q.shape[1]
    nl = 2 * npairs
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    at, parts = DNQ, {}
    z = np.zeros((nl, Rmax, D), np.float32)
    for name, a in (("q", q), ("k", k), ("v", v), ("z", z), ("lg", lg), ("b", beta)):
        b = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
        dram[at:at + b.size] = b
        strides, s = [], 1
        for n in reversed(a.shape):
            strides.append(s)
            s *= n
        parts[name] = Tensor(Affine(at), a.shape, tuple(reversed(strides)))
        at += b.size
    es = 2 if st16 else 4
    h = (IS.st16_pack(st.reshape(nl * D, D), "e6m9") if st16 else st).view(np.uint8).reshape(-1)
    dram[STATE:STATE + h.size] = h
    out = Tensor(Affine(DNO), (Rmax, nl * D), (nl * D, 1))
    rows = RunVar("rows", Rmax + 1)
    row = None if r0 is None else RunVar("row", Rmax)
    scr = DC.Scratch(SCR, C, D, D, cfg.WG)
    assert scr.end <= cfg.DRAM_BYTES

    def states(p):
        return Tensor(Affine(STATE + p * 2 * D * D * es), (2 * D, D), (D, 1), esize=es)

    hook = {} if between is None else {"between": between}

    @ol.jit
    def kern():
        DC.chunked(parts["q"], parts["k"], parts["v"], parts["lg"], parts["b"], out, states,
                   scr, rows, C, split1, row, **hook)

    bb = kern.trace(cfg, 0, {}, fit=fit)
    prog = bb.finish()
    args = arg_words(prog.run_args, {"rows": R, "row": r0 or 0})
    m = IS.Machine(cfg, [prog], [dram.copy()], args).run()
    dm = m.slices[0].dram
    if rtl is not None:
        from opentpu import rtlsim
        drams, tmems, _ = rtlsim.run(cfg, [prog], [dram], uarch=rtl, args=args)
        for what, a, b in (("DRAM bytes", drams[0], dm), ("TMEM words", tmems[0], m.slices[0].tmem)):
            bad = np.nonzero(a != b)[0]
            assert len(bad) == 0, f"{len(bad)} {what} differ, first at {bad[:6]}"
    if info is not None:
        info.update(tmem_peak=bb.tmem_peak, tmem=m.slices[0].tmem)
    o = dm[DNO:DNO + 4 * Rmax * nl * D].view(np.float32).reshape(Rmax, nl, D)
    sh = dm[STATE:STATE + es * nl * D * D]
    sf = IS.st16_f32(sh.view(np.uint16), "e6m9") if st16 else sh.view(np.float32)
    return o, sf.reshape(nl, D, D), prog


def _rel(x, ref):
    return float(np.abs(x - ref).max() / np.abs(ref).max())


def _emu(q, k, v, lg, beta, st, mode):
    """dn_chunk's numpy chunked recurrence in the mode (tools/dn_chunk.py), from the states st."""
    import math
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    DCH = pytest.importorskip("dn_chunk")
    qq, kk = (np.repeat(np.swapaxes(x, 0, 1), 2, 1) for x in (q, k))
    o, S = DCH.recur_chunk(qq, kk, np.swapaxes(v, 0, 1), lg.T * math.log(2.0), beta.T, mode,
                           np.swapaxes(st.astype(np.float64), 1, 2))
    return o, np.swapaxes(S, 1, 2)


@pytest.mark.parametrize("zero_state", [True, False])
def test_split_vs_recurrence(zero_state):
    """Every matmul split (split1: dn_chunk h12345678-o), one pair, two chunks of 64: o against
    the float64 recurrence within the split's ~1e-4, the states (e6m9 at the pass's end) within
    e6m9's rounding (2^-10)."""
    rng = np.random.default_rng(5 + zero_state)
    q, k, v, lg, beta, st = _data(rng, 1, 128, zero_state)
    o, sf, _ = _run(_cfg(), q, k, v, lg, beta, st, 128, 64, split1=True)
    o_ref, s_ref = _ref(q, k, v, lg, beta, st)
    eo, es = _rel(o, o_ref), _rel(sf, s_ref)
    assert eo < 2.5e-4 and es < 1.5e-3, (eo, es)


def test_fp32_states():
    """fp32 state slots (no ST16): the states' own error is the split's, ~1e-4."""
    rng = np.random.default_rng(9)
    q, k, v, lg, beta, st = _data(rng, 1, 128)
    o, sf, _ = _run(_cfg(), q, k, v, lg, beta, st, 128, 64, split1=True, st16=False)
    o_ref, s_ref = _ref(q, k, v, lg, beta, st)
    eo, es = _rel(o, o_ref), _rel(sf, s_ref)
    assert eo < 2.5e-4 and es < 3e-4, (eo, es)


@pytest.mark.parametrize("C", [32, 64])
def test_q1_vs_emulation(C):
    """The kernel's pick (M1 one int8 pass: dn_chunk c64-q1h2345678-o; chunks of 32 too) against
    dn_chunk's numpy emulation of it: within ~1e-4, while both are ~1e-3 from the recurrence (M1's
    int8). (The kernel's transposes take both heads' rows, a scale a row over the two.)"""
    rng = np.random.default_rng(13)
    q, k, v, lg, beta, st = _data(rng, 1, 128)
    o, sf, _ = _run(_cfg(), q, k, v, lg, beta, st, 128, C)
    o_ref, _ = _ref(q, k, v, lg, beta, st)
    o_emu, _ = _emu(q, k, v, lg, beta, st, f"c{C}-q1h2345678-o")
    assert _rel(o, o_emu) < 2.5e-4 and _rel(o, o_ref) < 3e-3, (_rel(o, o_emu), _rel(o, o_ref))


def test_pairs_and_rows():
    """Two pairs (key head p, value heads 2p, 2p + 1; their states and output columns), the same
    program for a pass of 1 and of 3 chunks (rows a run-time value): each pass's o and states
    against the recurrence over its rows, the rows past it not written."""
    rng = np.random.default_rng(17)
    q, k, v, lg, beta, st = _data(rng, 2, 192)
    progs = []
    for R in (64, 192):
        o, sf, prog = _run(_cfg(), q, k, v, lg, beta, st, R, 64, split1=True)
        progs.append(prog)
        o_ref, s_ref = _ref(q[:, :R], k[:, :R], v[:, :R], lg[:, :R], beta[:, :R], st)
        eo, es = _rel(o[:R], o_ref), _rel(sf, s_ref)
        assert eo < 2.5e-4 and es < 1.5e-3, (R, eo, es)
        assert not o[R:].any()
    words = [[(i.op, i.ra, i.rb, i.rc, i.rd, i.flags, tuple(i.w)) for i in p] for p in progs]
    assert words[0] == words[1]


def test_start_row():
    """The run argument "row" (a prompt's first chunk from row 3: its rows 0 .. 2 ran as layer
    runs): rows r0 .. r0 + R - 1 from the states given, o into the same rows, the others not
    written; one program for r0 3 and 7."""
    rng = np.random.default_rng(19)
    q, k, v, lg, beta, st = _data(rng, 2, 192)
    progs = []
    for r0 in (3, 7):
        R = 128
        o, sf, prog = _run(_cfg(), q, k, v, lg, beta, st, R, 64, split1=True, r0=r0)
        progs.append(prog)
        sl = slice(r0, r0 + R)
        o_ref, s_ref = _ref(q[:, sl], k[:, sl], v[:, sl], lg[:, sl], beta[:, sl], st)
        eo, es = _rel(o[sl], o_ref), _rel(sf, s_ref)
        assert eo < 2.5e-4 and es < 1.5e-3, (r0, eo, es)
        assert not o[:r0].any() and not o[r0 + R:].any()
    words = [[(i.op, i.ra, i.rb, i.rc, i.rd, i.flags, tuple(i.w)) for i in p] for p in progs]
    assert words[0] == words[1]


def test_waits():
    """The pass's wait entries (opentpu.waits, x2's AW 36 and window of 32) hold every conflict
    of an ISA run of three chunks (waits.check_isasim)."""
    from opentpu import waits as W
    cfg = _cfg()
    rng = np.random.default_rng(19)
    q, k, v, lg, beta, st = _data(rng, 2, 192)
    _, _, prog = _run(cfg, q, k, v, lg, beta, st, 192, 64)
    w = W.wait_counts(prog, cfg.D, 1, aw=cfg.AW, win=cfg.WC_WIN)
    m = IS.Machine(cfg, [prog], [np.zeros(cfg.DRAM_BYTES, np.uint8)],
                   arg_words(prog.run_args, {"rows": 192}))
    assert W.check_isasim(m, [w]) > 1000


def test_qhl_one_qact():
    """Config.QHL (QACT flag HL): every split stationary's hi and lo from one QACT, o and the
    states bit for bit those of two QACTs (q, then LO), one QACT fewer a split operand; the
    wait entries hold every conflict of the run (lo's blocks after hi's)."""
    from opentpu import isa as I
    from opentpu import waits as W
    rng = np.random.default_rng(29)
    q, k, v, lg, beta, st = _data(rng, 2, 128)
    off = dataclasses.replace(_cfg(), QHL=False)
    on = dataclasses.replace(off, QHL=True)
    assert on.SPLITQ and on.SEQ_WC
    o0, s0, p0 = _run(off, q, k, v, lg, beta, st, 128, 64)
    o1, s1, p1 = _run(on, q, k, v, lg, beta, st, 128, 64)
    assert np.array_equal(o0.view(np.uint32), o1.view(np.uint32))
    assert np.array_equal(s0.view(np.uint32), s1.view(np.uint32))

    def qacts(p):
        qa = [i for i in p if i.op == I.QACT]
        return len(qa), sum(bool(i.flags & I.F_QHL) for i in qa), sum(bool(i.flags & I.F_QLO)
                                                                       for i in qa)
    n0, h0, l0 = qacts(p0)
    n1, h1, l1 = qacts(p1)
    assert h0 == 0 and l0 > 0 and l1 == 0 and h1 == l0 and n1 == n0 - l0, (qacts(p0), qacts(p1))
    w = W.wait_counts(p1, on.D, 1, aw=on.AW, win=on.WC_WIN)
    m = IS.Machine(on, [p1], [np.zeros(on.DRAM_BYTES, np.uint8)],
                   arg_words(p1.run_args, {"rows": 128}))
    assert W.check_isasim(m, [w]) > 1000


def test_qsthl_one_qst():
    """Config.QSTHL (QST HL): every streamed operand's hi and lo rows from one QST (lo's bytes and
    scales in the scratch's lo matrix, the same KiB multiple after hi's), o and the states bit
    for bit those of a QST and a QST LO, one QST fewer a split operand; the wait entries hold
    every conflict of the run (lo's part included)."""
    from opentpu import isa as I
    from opentpu import waits as W
    rng = np.random.default_rng(31)
    q, k, v, lg, beta, st = _data(rng, 2, 128)
    off = dataclasses.replace(_cfg(), QSTHL=False)
    on = dataclasses.replace(off, QSTHL=True)
    assert on.SPLITQ and on.SEQ_WC and on.WG > 1
    o0, s0, p0 = _run(off, q, k, v, lg, beta, st, 128, 64)
    o1, s1, p1 = _run(on, q, k, v, lg, beta, st, 128, 64)
    assert np.array_equal(o0.view(np.uint32), o1.view(np.uint32))
    assert np.array_equal(s0.view(np.uint32), s1.view(np.uint32))

    def qsts(p):
        qs = [i for i in p if i.op == I.QST]
        return len(qs), sum(bool(i.w[4] & I.QST_HL) for i in qs), sum(bool(i.flags & I.F_SLO)
                                                                      for i in qs)
    n0, h0, l0 = qsts(p0)
    n1, h1, l1 = qsts(p1)
    assert h0 == 0 and l0 > 0 and l1 == 0 and h1 == l0 and n1 == n0 - l0, (qsts(p0), qsts(p1))
    w = W.wait_counts(p1, on.D, 1, aw=on.AW, win=on.WC_WIN)
    m = IS.Machine(on, [p1], [np.zeros(on.DRAM_BYTES, np.uint8)],
                   arg_words(p1.run_args, {"rows": 128}))
    assert W.check_isasim(m, [w]) > 1000


def test_splmm_one_mm():
    """Config.SPLMM (MM SPL): a split product's three terms (and the transposes' hi + lo pairs) as
    one MM each, the lo terms' weights QST HL's lo part HLD after hi's: about 2.5x fewer MMs, o
    and the states within an e6m9 ulp of the three MMs' (one epilogue rounding, the lo terms'
    drop at da >= 9) and as near the float64 recurrence; the wait entries hold every conflict of
    the run (the weights' lo part and the lo ACT block in the MM's footprint)."""
    from opentpu import isa as I
    from opentpu import waits as W
    rng = np.random.default_rng(29)
    q, k, v, lg, beta, st = _data(rng, 2, 128)
    off = dataclasses.replace(_cfg(), QHL=True, QSTHL=True)
    on = dataclasses.replace(off, SPLMM=True)
    o0, s0, p0 = _run(off, q, k, v, lg, beta, st, 128, 64)
    o1, s1, p1 = _run(on, q, k, v, lg, beta, st, 128, 64)
    mm0 = [i for i in p0 if i.op == I.MM]
    mm1 = [i for i in p1 if i.op == I.MM]
    spl = [i for i in mm1 if i.w[6] & I.MM_SPL]
    kb3 = sum((i.w[3] >> 16 & 0xFF) == 3 for i in spl)
    assert not any(i.w[6] & I.MM_SPL for i in mm0)
    assert len(mm1) == len(mm0) - 2 * kb3 - (len(spl) - kb3) and kb3 and len(spl) > kb3, \
        (len(mm0), len(mm1), len(spl), kb3)
    o_ref, s_ref = _ref(q, k, v, lg, beta, st)
    e0, e1 = _rel(o0, o_ref), _rel(o1, o_ref)
    assert _rel(o1, o0) < 2e-4 and _rel(s1, s0) < 1.5e-3, (_rel(o1, o0), _rel(s1, s0))
    assert e1 < 1.05 * e0 and _rel(s1, s_ref) < 1.05 * _rel(s0, s_ref), (e0, e1)
    w = W.wait_counts(p1, on.D, 1, aw=on.AW, win=on.WC_WIN)
    m = IS.Machine(on, [p1], [np.zeros(on.DRAM_BYTES, np.uint8)],
                   arg_words(p1.run_args, {"rows": 128}))
    assert W.check_isasim(m, [w]) > 1000


def test_splmm_refused_without_it():
    """An MM SPL on a machine without SPLMM is refused (the bit is the encoding's, not a plain
    srs's)."""
    from opentpu import isa as I
    cfg = dataclasses.replace(_cfg(), QHL=True, QSTHL=True, SPLMM=True)
    rng = np.random.default_rng(29)
    q, k, v, lg, beta, st = _data(rng, 1, 64)
    _, _, prog = _run(cfg, q, k, v, lg, beta, st, 64, 64)
    assert any(i.op == I.MM and i.w[6] & I.MM_SPL for i in prog)
    m = IS.Machine(dataclasses.replace(cfg, SPLMM=False), [prog],
                   [np.zeros(cfg.DRAM_BYTES, np.uint8)], arg_words(prog.run_args, {"rows": 64}))
    with pytest.raises(IS.SimError, match="SPL"):
        m.run()


@pytest.mark.parametrize("flag", ["OUTER", "OUTER1"])
def test_outer_t_loop(flag):
    """Config.OUTER (VOP OUTER, decay one) or Config.OUTER1 (decay one alone, on MAC's datapath:
    wmx's VPU, OUTER off): T = (I - NL)^-1's rank-one steps one OUTER each, tv += NL[s + 1.., s] x
    T[s, ..s], in place of a MUL into a temp and an ADD: o and the states bit for bit the MUL /
    ADD loop's, C - 1 OUTERs for each head of each pair (two pairs here), one VOP fewer a step;
    the wait entries hold every conflict of the run."""
    from opentpu import isa as I
    from opentpu import waits as W
    rng = np.random.default_rng(41)
    q, k, v, lg, beta, st = _data(rng, 2, 128)
    off = dataclasses.replace(_cfg(), OUTER=False)
    on = dataclasses.replace(off, **{flag: True})
    o0, s0, p0 = _run(off, q, k, v, lg, beta, st, 128, 64)
    o1, s1, p1 = _run(on, q, k, v, lg, beta, st, 128, 64)
    assert np.array_equal(o0.view(np.uint32), o1.view(np.uint32))
    assert np.array_equal(s0.view(np.uint32), s1.view(np.uint32))
    nout = sum(i.op == I.VOP and (i.w[5] >> 16) & 0xFF == I.V_OUTER for i in p1)
    assert nout == 2 * 2 * 63 and len(p1) == len(p0) - nout, (nout, len(p0), len(p1))
    w = W.wait_counts(p1, on.D, 1, aw=on.AW, win=on.WC_WIN)
    m = IS.Machine(on, [p1], [np.zeros(on.DRAM_BYTES, np.uint8)],
                   arg_words(p1.run_args, {"rows": 128}))
    assert W.check_isasim(m, [w]) > 1000


def _words(prog):
    return [(i.op, i.ra, i.rb, i.rc, i.rd, i.flags, tuple(i.w)) for i in prog]


def test_between_none():
    """chunked's between=None, and a hook that emits nothing: the program without the argument,
    instruction for instruction."""
    rng = np.random.default_rng(41)
    q, k, v, lg, beta, st = _data(rng, 2, 128)
    calls = []
    progs = [_run(_cfg(), q, k, v, lg, beta, st, 128, 64, between=b)[2]
             for b in (None, calls.append)]
    assert calls == [0, 1]
    assert _words(progs[0]) == _words(progs[1])


def test_between_hook():
    """A hook between the pairs (a pipelined stage's shape), traced best-fit (the fallback of
    compiler.trace_fit): a tile of 30,720 words its call's own (filled), 8,192 kept live across
    the pairs (a copy of the first's rows each call), its own tiles only: the core's o and states
    bit for bit those without it (next-fit), the kept tile's words the last call's, the TMEM
    peak with the hook live under TMEM's words. (Next-fit places the core's tiles so that it
    has no room for a tile kept across a pair: 54,719 words free at the pair boundary, in gaps
    of 16,384 at most, and the chunk's 16K temporaries need them.)"""
    cfg = _cfg()
    rng = np.random.default_rng(43)
    q, k, v, lg, beta, st = _data(rng, 2, 128)
    kept, calls = [], []

    def between(p):
        calls.append(p)
        if not kept:
            kept.append(ol.empty([64, D], dense=True))  # 8,192 words: live across the pairs
        big = ol.empty([240, D], dense=True)            # 30,720 words: this call's
        big.set(float(p + 1))
        kept[0].set(big[0:64, :])
        del big

    i0, i1 = {}, {}
    o0, s0, _ = _run(cfg, q, k, v, lg, beta, st, 128, 64, info=i0)
    o1, s1, _ = _run(cfg, q, k, v, lg, beta, st, 128, 64, between=between, info=i1, fit="best")
    assert calls == [0, 1]
    assert np.array_equal(o0.view(np.uint32), o1.view(np.uint32))
    assert np.array_equal(s0.view(np.uint32), s1.view(np.uint32))
    t = kept[0]
    w = i1["tmem"][t.base:t.base + 64 * D].view(np.float32)
    assert (w == 2.0).all()
    print(f"TMEM peak {i0['tmem_peak']} -> {i1['tmem_peak']} with the hook (of {cfg.TMEM_WORDS})")
    assert i0["tmem_peak"] + 8192 <= i1["tmem_peak"] <= cfg.TMEM_WORDS


@pytest.mark.parametrize("qhl,qsthl,outer,spl", [(False, False, False, False),
                                                 (True, False, False, False),
                                                 (True, True, False, False),
                                                 (True, True, True, False),
                                                 (True, True, True, True)])
def test_dn_chunk_rtl(have_verilator, qhl, qsthl, outer, spl):
    """The kernel on the RTL at bcu1525-x2:wmx's slice (its core's micro-architecture: TMEM's one
    pumped copy, the quantizer at PW 2 with SWW 4, the VPU's composites at VPU_CNS 5, the
    instruction cache, WAITC at the window of 32; the behavioural DRAM at its 8 channels of
    128-byte beats, port A's WG 8 width), one pair, two chunks of 64 (the states carried), QHL
    off and on (QACT flag HL: a split stationary's hi and lo from one QACT), and QSTHL (QST HL,
    QUANT_QSTB's row buffer: a streamed operand's hi and lo rows from one QST), and OUTER1 (the
    VPU, without OUTER, with VOP OUTER decay one on MAC's datapath: T's rank-one steps), and
    SPLMM (otpu_mxu SPL: a split's terms as one MM): its DRAM (o, the e6m9 states) and TMEM bit
    for bit the ISA's."""
    from opentpu import isa as I
    ua = {k: v for k, v in T.sim_uarch("bcu1525-x2:wmx").items()
          if not k.startswith(("COLL_", "NAT_")) and k != "XRS"}
    if qsthl:
        ua |= {"QUANT_QSTB": 1}
    cfg = dataclasses.replace(_cfg(), QHL=qhl, QSTHL=qsthl, OUTER1=outer, SPLMM=spl)
    rng = np.random.default_rng(37)
    q, k, v, lg, beta, st = _data(rng, 1, 128)
    o, sf, prog = _run(cfg, q, k, v, lg, beta, st, 128, 64, rtl=ua)
    assert any(i.op == I.QACT and i.flags & I.F_QHL for i in prog) == qhl
    assert any(i.op == I.QST and i.w[4] & I.QST_HL for i in prog) == qsthl
    assert any(i.op == I.VOP and (i.w[5] >> 16) & 0xFF == I.V_OUTER for i in prog) == outer
    assert any(i.op == I.MM and i.w[6] & I.MM_SPL for i in prog) == spl
    o_ref, s_ref = _ref(q, k, v, lg, beta, st)
    assert _rel(o, o_ref) < 3e-3 and _rel(sf, s_ref) < 1.5e-3, (_rel(o, o_ref), _rel(sf, s_ref))


@pytest.mark.parametrize("S, r0", [(1, 0), (4, 3)])
def test_image_dn_run(monkeypatch, S, r0):
    """Image.compile_dn_run on a Qwen3.5-MoE image in the 35B's head shape (4 key heads, 8 value
    heads, d 128: at S = 1 the 35B's slice at S = 4, four pairs) on bcu1525-x2:wmx with
    Config.DNCHUNK 64: io["dns"] beside dnq / dno, the run over 128 of the dnq rows a slice holds
    from row r0 (a prompt's first chunk: 3) and the states in its layer's slots
    (DeltaNetParts.states), against the recurrence."""
    from opentpu.llm import qwen35 as Q35
    from opentpu.llm.moe import MoESpec
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    H = 256 * max(S // 2, 1)                      # hidden, the MLPs' widths: multiples of S D
    spec = Q35.Spec(H, ("linear", "attn"), 8, 2, 256, 64, 8, 128, 128, H, 1000,
                    lin_kheads=4, moe=MoESpec(E=8, k=2, ffn=H, rule="softmax", shared=H))
    cfg = dataclasses.replace(T.config("bcu1525-x2:wmx", S=S, DRAM_BYTES=1 << 28), DNCHUNK=64)
    Rp, R = 192, 128
    img = Q35.Image(spec, cfg, 256, prefill_rows=Rp, em=True, lookup=True)
    assert {"dnq", "dno", "dns"} <= set(img.io) and img.dn_core_ok()
    progs, run_args = img.compile_dn_run(0)
    rng = np.random.default_rng(23)
    drams, data = [], []
    for s in range(S):
        dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
        m = img.descriptors(s)
        q = m.dnq
        npairs = q.nk
        x = _data(rng, npairs, Rp)
        for t, a in zip((q.q, q.k, q.v, q.lg, q.b), x[:5]):
            b = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            at = Affine.of(t.base).static()
            dram[at:at + b.size] = b
        dn = m.layer(0).dn
        for p in range(npairs):
            t = dn.states(p)
            assert t.esize == 2
            h = IS.st16_pack(x[5][2 * p:2 * p + 2].reshape(2 * D, D), "e6m9")
            at = Affine.of(t.base).static()
            dram[at:at + h.nbytes] = h.view(np.uint8).reshape(-1)
        drams.append(dram)
        data.append(x)
    mach = IS.Machine(cfg, progs, drams, arg_words(run_args, {"row": r0, "rows": R})).run()
    for s in range(S):
        m, dm, x = img.descriptors(s), mach.slices[s].dram, data[s]
        nl = 2 * m.dnq.nk
        at = Affine.of(m.dno.base).static()
        o = dm[at:at + 4 * Rp * nl * D].view(np.float32).reshape(Rp, nl, D)
        dn = m.layer(0).dn
        st = np.stack([IS.st16_f32(dm[Affine.of(dn.states(p).base).static():][:4 * D * D]
                                   .view(np.uint16), "e6m9").reshape(2, D, D)
                       for p in range(nl // 2)]).reshape(nl, D, D)
        sl = slice(r0, r0 + R)
        o_ref, s_ref = _ref(*(a[:, sl] for a in x[:5]), x[5])
        eo, es = _rel(o[sl], o_ref), _rel(st, s_ref)
        assert eo < 3e-3 and es < 2.5e-3, (s, eo, es)       # (q1, e6m9: ~1.5e-3)
        assert not o[:r0].any() and not o[r0 + R:].any()
