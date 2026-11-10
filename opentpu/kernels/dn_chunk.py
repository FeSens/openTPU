"""Chunked Gated DeltaNet on the MXU (docs/deltanet_chunk.md):
a pass's rows of every head by chunks of C rows, the recurrence's eight matmuls per chunk and
head on the MXU, the in-chunk solve on the VPU, the state in fp32 in TMEM for the whole pass.

Per pair p of heads a, b sharing key head p (Qwen3.5-MoE's: value head j's key head j // 2), the
chunk's q, k [C, d] and each head's v [C, d], log2 decays lg and betas [C] (the image's dnq,
docs/deltanet_chunk.md), gc = cumsum(lg), Gam[r, s] = 2^(gc_r - gc_s), St = S^T
[dv, dk] the head's state (the slot's layout):
    M4  A  = (q k^T) Gam, lower with the diagonal             (once a pair)
    M1  NL = -(beta k) k^T Gam, strictly lower                (one int8 pass)
        T  = (I - NL)^-1                                       (VPU: a column of NL at a time)
    M2  U  = T diag(beta) V                                    (V^T streamed)
    M3  W  = -T diag(beta 2^gc) K                              (K^T streamed)
    M5  U += W S                                               (St's rows streamed)
    M6  O  = (2^gc q) S
    M7  O += A U                                               (U^T streamed)
    M8  St = 2^gl St + U^T diag(2^(gl - gc)) K                 (gl = gc's last; into St)
dn_chunk's -o form (tools/dn_chunk.py): every streamed operand is QST'd a row at a time (ROW:
one fp32 scale a row) into the scratch, X^T (K^T, V^T, U^T) by an identity MM from X's rows,
and every matmul but M1 in three passes of split operands (hi.hi + hi.lo + lo.hi: QACT / QST
flag LO, Config.SPLITQ). With Config.QHL a stationary's two halves come from one QACT (flag HL:
one read of the tile, lo's blocks after hi's), and with Config.QSTHL a streamed operand's from one
QST (HL: lo's bytes and scales in the scratch's lo matrix, the same offset after hi's).

A token contraction's operands are D wide and C <= D / 2: the pair's two heads share them, head
a's tokens in columns 0 .. C - 1 and head b's in C .. 2 C - 1 (T, A; V^T and U^T both heads' in
one tile, one transpose and one QST each; K^T in both halves). A head's QACT takes its half by
its column scale (CSCALE: zero on the other head's columns; A's by a 0/1 mask). The state's
rows (M5's, M6's streamed side) a head each.

The state: the slot's (16-bit with Config.ST16: STREAM's ST16R / ST16W copies, Config.ST16S) to
an fp32 copy in the scratch, into TMEM, back at the pass's end.
"""
from __future__ import annotations

import builtins

from .. import isa as I
from .. import language as ol
from .. import quant as Q
from ..compiler import (B_ROWVIEW, Affine, Bcast, CompileError, QTensor, RunVar, Tensor, Tile,
                        current)

ALIGN = 1024                    # the scratch's matrices: whole group chunks (WG x D bytes)


class Chunks(RunVar):
    """A pass's chunks, rows // C of the run's `rows` (the chunk loop's count register)."""

    def __init__(self, rows: RunVar, C: int):
        super().__init__(f"{rows.name}//{C}", None if rows.bound is None else rows.bound // C + 1)
        self.rows, self.C = rows, C

    def value(self, values: dict) -> int:
        n = self.rows.value(values)
        if n % self.C:
            raise ValueError(f"{self.rows.name} {n}: not a multiple of the chunk's {self.C} rows")
        return n // self.C


class Scratch:
    """The core's DRAM scratch at `base` (a slice's: Image's io["dns"]): the streamed operands
    (int8 in groups, a fp32 scale a row: QST ROW, MM RSC), hi and lo parts each: K's rows (C)
    and K^T's (d); the pair's V and U rows (2 C: head a's, head b's) and their V^T, U^T (d);
    each head's St rows (d); then the pair's states in fp32 ([2 d, d]: the 16-bit slot's
    copy)."""

    MATS = (("kq", "C"), ("kt", "d"), ("vq", "2C"), ("vt", "d"), ("uq", "2C"), ("ut", "d"),
            ("sta", "d"), ("stb", "d"))

    def __init__(self, base: int, C: int, d: int, D: int, wg: int):
        self.C, self.d, self.D, self.wg = C, d, D, wg
        rs, srs = Q.grp_row_bytes(D, "int8", D, wg), Q.grp_scale_bytes(D, "int8", D, wg, True)
        at, self.m = int(base), {}
        for name, rows in self.MATS:
            n = {"C": C, "d": d, "2C": 2 * C}[rows]
            for part in ("hi", "lo"):
                data = at
                at += (n // wg) * rs
                scale = at
                at = -(-(at + (n // wg) * srs) // ALIGN) * ALIGN
                self.m[name, part] = QTensor(Affine(data), Affine(scale), (n, D), rs, srs, D,
                                             rsc=True, grp=wg)
        self.stage = Tensor(Affine(at), (2 * d, d), (d, 1))
        self.end = at + 4 * 2 * d * d

    @staticmethod
    def nbytes(C: int, d: int, D: int, wg: int) -> int:
        return Scratch(0, C, d, D, wg).end


def check(C: int, d: int, cfg) -> None:
    D = cfg.D
    if d != D:
        raise CompileError(f"the chunked DeltaNet: heads of D = {D} (dk = dv = {d})")
    if not cfg.SPLITQ or cfg.WG < 2 or not cfg.GALL:
        raise CompileError("the chunked DeltaNet needs SPLITQ and every MM in groups (GALL)")
    if C % cfg.WG or not cfg.WG <= C <= D // 2 or C & (C - 1):
        raise CompileError(f"chunks of {C} rows: a power of two, whole groups, at most D / 2")


def _at(t: Tensor, c) -> Tensor:
    """t moved by counter c's bytes."""
    return Tensor(Affine.of(t.base) + Affine.of(c), t.shape, t.strides, t.host, t.offset,
                  t.esize)


def _qst(x: Tile, m: QTensor, lo: bool = False, hld: int = 0) -> None:
    """x's rows into the grouped matrix m from row 0, quantized a row (QST GRP ROW)."""
    current().store_quantized(x, m.data, m.scale, 0, m.srs, row_scale=True, row=Affine(0),
                              lo=lo, hld=hld)


def _qst2(x: Tile, hi: QTensor, lo: QTensor) -> None:
    """x's rows split into hi and lo: one QST HL where the machine has it (Config.QSTHL; lo's
    bytes and scales the same KiB and group chunk multiple after hi's, as Scratch lays them
    out), else a QST and a QST LO."""
    dd, ds = Affine.of(lo.data) - Affine.of(hi.data), Affine.of(lo.scale) - Affine.of(hi.scale)
    cfg = current().cfg
    if (cfg.QSTHL and not dd.terms and not ds.terms and dd.const == ds.const and dd.const > 0
            and dd.const % 1024 == 0 and dd.const % (cfg.WG * cfg.D) == 0):
        _qst(x, hi, hld=dd.const)
    else:
        _qst(x, hi)
        _qst(x, lo, lo=True)


def _vop(func: int, dst: Tile, a: Tile | None, b=None) -> None:
    bb = current()
    bb.vop(func, dst, a, b)
    bb.bump_version(dst.buf)


def _split(a, w_hi: QTensor, w_lo: QTensor, out: Tile | None = None, acc: Tile | None = None):
    """a's split (hi, lo stationaries) times w's (hi, lo): hi.hi + hi.lo + lo.hi (one MM SPL
    with Config.SPLMM: Builder.dot_split)."""
    return current().dot_split(a, (w_hi, w_lo), acc=acc, out=out)


def _q2(x: Tile, **kw) -> tuple:
    if current().cfg.QHL:                       # one QACT HL (Config.QHL): the same bytes
        return ol.quantize(x, hl=True, **kw)
    return ol.quantize(x, **kw), ol.quantize(x, lo=True, **kw)


def _cumsum(x: Tile) -> Tile:
    """x's rows' prefix sums ([r, n], n a power of two): Hillis-Steele between x and a tile of
    its shape (an ADD and a COPY a step). Returns the tile that holds them."""
    n = x.cols
    a, t = x, ol.empty(x.shape, dense=True)
    k = 1
    while k < n:
        _vop(I.V_ADD, t[:, k:n], a[:, k:n], a[:, 0:n - k])
        _vop(I.V_COPY, t[:, 0:k], a[:, 0:k])
        a, t = t, a
        k *= 2
    return a


def _iota(n: int) -> Tile:
    """0, 1, .. n - 1 (n a power of two)."""
    x = ol.full([1, n], 1.0)
    c = _cumsum(x)
    out = ol.empty([n])
    out.set(c[0, :] - 1.0)
    return out


def chunked(q: Tensor, k: Tensor, v: Tensor, lg: Tensor, beta: Tensor, out: Tensor,
            states, scr: Scratch, rows: RunVar, C: int, split1: bool = False,
            row: RunVar | None = None, between=None) -> None:
    """The slice's pairs over a pass of `rows` rows (a run-time value, a multiple of C) from row
    `row` (a run-time value; None: 0): q, k [np, Rmax, d] (np pairs), v [2 np, Rmax, d], lg,
    beta [2 np, Rmax] (log2 decays, betas), out [Rmax, 2 np d] (row-major: head j's chunk [C, d]
    at columns j d), rows row .. row + rows - 1 of each;
    states(p): pair p's two states [2 d, d] (St each, head a's rows then b's), 16-bit (ST16) or
    fp32, read at the pass's start and written at its end. The states stay in TMEM in fp32
    over the pass (head-outer, chunks inner). split1: M1 split as well (dn_chunk h1).
    between: a callable emitting code between the pairs, between(p) once a pair, after pair p's
    states are written back (their slot's copy included) and before pair p + 1's load, the last
    pair's too (a pipelined stage's work at the pair boundaries). The
    core's tiles live there are the pass's shared ones (the identity, the masks, XT, T, A, the
    column scales, St: 76,353 words at D 128, C 64; its peak inside a chunk 97,216); a tile the
    hook keeps across a pair adds to that peak. Next-fit placement leaves the core no room for
    one (the chunk's 16K temporaries take the gaps): trace a program with a hook best-fit
    (compiler.trace_fit's fallback), where it keeps up to 16,384 words beside 30,720 of its
    own call's (test_dn_chunk.test_between_hook). None: the program as without it."""
    b = current()
    cfg = b.cfg
    D, np_, d = cfg.D, q.shape[0], q.shape[2]
    check(C, d, cfg)
    if v.shape[0] != 2 * np_ or out.shape[1] != 2 * np_ * d:
        raise CompileError("the chunked DeltaNet: two value heads a key head")
    # the prologue: the identity (the transposes' stationary), the masks, the zero-padded tiles
    io = _iota(D)
    X = ol.empty([D, D])
    X.set(io[None, :])                                  # X[r, c] = c
    X.set(X - io[:, None])                              # c - r
    X.set(ol.abs(X))
    X.set(1.0 - X)
    X.set(ol.maximum(X, 0.0))                           # (c == r)
    eye = ol.quantize(X, free=True)
    del X
    Y = ol.empty([C, C], dense=True)
    Y.set(io[0:C][:, None])
    Y.set(Y - io[0:C][None, :])                         # r - s
    Ms = ol.empty([C, C], dense=True)                   # s < r
    Ms.set(ol.maximum(Y, 0.0))
    Ms.set(ol.minimum(Ms, 1.0))
    Ml = ol.empty([C, C], dense=True)                   # s <= r
    Ml.set(Y + 1.0)
    Ml.set(ol.maximum(Ml, 0.0))
    Ml.set(ol.minimum(Ml, 1.0))
    del Y
    mk = [ol.empty([D]), ol.empty([D])]                 # head a's columns, head b's (A's QACT)
    mk[0].set(ol.minimum(ol.maximum(float(C) - io, 0.0), 1.0))
    mk[1].set(ol.minimum(ol.maximum(float(2 * C) - io, 0.0), 1.0))
    mk[1].set(mk[1] - mk[0])
    del io
    XT = ol.zeros([D, D])                               # X^T [d, 2 C]: head a's, head b's
    T, A = ol.zeros([C, D]), ol.zeros([C, D])           # head a's columns 0 .., head b's C ..
    cs = [[ol.zeros([D]) for _ in range(3)] for _ in (0, 1)]   # a head's csb, csg, csd
    St = [ol.empty([d, d]), ol.empty([d, d])]
    m = scr.m
    nch = Chunks(rows, C)
    for p in builtins.range(np_):
        sts = states(p)
        src = scr.stage if sts.esize != 4 else sts
        if src is not sts:
            ol.state_copy(sts, src)
        for a in (0, 1):
            t = ol.load(src[a * d:(a + 1) * d, :])
            St[a].set(t)
            del t
            _qst2(St[a], m[f"st{'ab'[a]}", "hi"], m[f"st{'ab'[a]}", "lo"])
        cq, cg, co = b.counter("dn rows"), b.counter("dn gate rows"), b.counter("dn out rows")
        at = [Affine.of(c) if row is None else Affine.of(c) + row * (4 * t.strides[i])
              for c, t, i in ((cq, q, 1), (cq, v, 1), (cg, lg, 1), (cg, beta, 1), (co, out, 0))]
        lp = b.begin_loop(0, rcount=b.arg_reg(nch, 1))
        _chunk(p, q, k, v, lg, beta, out, m, St, eye, Ms, Ml, mk, XT, T, A, cs, C, d, at,
               split1)
        b.step(cq, 4 * C * d)
        b.step(cg, 4 * C)
        b.step(co, 4 * C * out.strides[0])
        b.end_loop(lp)
        for c in (cq, cg, co):
            b.end_counter(c)
        for a in (0, 1):
            t = ol.empty([d, d], dense=True)
            t.set(St[a])
            ol.store(src[a * d:(a + 1) * d, :], t)
            del t
        if src is not sts:
            ol.state_copy(src, sts)
        if between is not None:
            between(p)
    b.release_arg(nch)


def _transpose(eye: Tile, x: tuple, out: Tile) -> None:
    """out = X^T from X's split rows x (hi, lo): the identity's MMs (one MM SPL with
    Config.SPLMM)."""
    current().dot_split((eye, None), x, out=out)


def _chunk(p, q, k, v, lg, beta, out, m, St, eye, Ms, Ml, mk, XT, T, A, cs, C, d, at,
           split1: bool) -> None:
    """Pair p's chunk at the counters' rows (at: q / k's, v's, lg's, beta's and out's offsets)."""
    aq, av, ag, ab, ao = at
    def mat(name, n=None):
        return tuple(m[name, part] if n is None else m[name, part][0:n] for part in ("hi", "lo"))

    def qst(x, name):
        _qst2(x, m[name, "hi"], m[name, "lo"])

    Qt = ol.load(_at(q[p, 0:C, :], aq))
    Kt = ol.load(_at(k[p, 0:C, :], aq))
    G = ol.load(_at(lg[2 * p:2 * p + 2, 0:C], ag), out=ol.empty([2, C], dense=True))
    B = ol.load(_at(beta[2 * p:2 * p + 2, 0:C], ab), out=ol.empty([2, C], dense=True))
    # K's rows (M1's and M4's streamed side) and K^T's in both halves (M3's and M8's)
    qst(Kt, "kq")
    kq = mat("kq", C)
    _transpose(eye, kq, XT[:, 0:C])
    _transpose(eye, kq, XT[:, C:2 * C])
    qst(XT, "kt")
    kt = mat("kt")
    # M4: q k^T, split
    QK = _split(_q2(Qt), *kq)
    # the heads' decays (gc = cumsum(lg)); q and k quantized for both heads' M6 and M1 (2^gc,
    # -beta in RSCALE) while they are here
    GC = _cumsum(G)
    egcs = [ol.exp2(GC[a, :]) for a in (0, 1)]
    nbs = [B[a, :] * -1.0 for a in (0, 1)]
    q6 = [_q2(Qt, rscale=egcs[a], free=True) for a in (0, 1)]
    k1 = [_q2(Kt, rscale=nbs[a], free=True) if split1 else
          (ol.quantize(Kt, rscale=nbs[a], free=True), None) for a in (0, 1)]
    del Qt, Kt
    # the heads' V rows, V^T's (M2's streamed side): one transpose
    for a in (0, 1):
        Vt = ol.load(_at(v[2 * p + a, 0:C, :], av))
        _qst2(Vt, m["vq", "hi"][a * C:(a + 1) * C], m["vq", "lo"][a * C:(a + 1) * C])
        del Vt
    _transpose(eye, mat("vq", 2 * C), XT[:, 0:2 * C])
    qst(XT, "vt")
    vt = mat("vt")
    for a in (0, 1):
        c0, c1 = a * C, (a + 1) * C
        gc, bt = GC[a, :], B[a, :]
        csb, csg = cs[a][0], cs[a][1]
        csb[c0:c1].set(bt)
        csg[c0:c1].set(nbs[a] * egcs[a])
        # Gam: 2^(gc_r - gc_s) masked (MIN against the 0/1 masks: the upper triangle's inf -> 0)
        E = ol.empty([C, C])
        E.set(gc[:, None])
        E.set(ol.exp2(E - gc[None, :]))
        E.set(ol.minimum(E, Ml))
        A[:, c0:c1].set(QK * E)
        E.set(ol.minimum(E, Ms))
        # M1: NL = -(beta k) k^T Gam (one int8 pass; split1: split)
        if split1:
            NL = _split(k1[a], *kq)
        else:
            NL = ol.dot(k1[a][0], kq[0])
        k1[a] = None
        NL.set(NL * E)
        del E
        # T = (I - NL)^-1 = (I + L)^-1: from I, rows s + 1 .. += NL[r, s] T[s, :] for each s
        T[:, c0:c1].set(Ml - Ms)
        if current().cfg.OUTER or current().cfg.OUTER1:
            # (VOP OUTER, decay one: tv += NL[s + 1.., s] (per row) x T[s, ..s] (the column
            # vector), the MUL then ADD below in one pass; OUTER1: on MAC's datapath)
            bb = current()
            for s in builtins.range(C - 1):
                n = C - 1 - s
                tv = T[s + 1:C, c0:c0 + s + 1]
                bb.emit(I.outer(tv.base, 0, NL.base + (s + 1) * NL.rs + s,
                                T.base + s * T.rs + c0, n, s + 1, T.rs, NL.rs, "one",
                                comment="T += NL[:, s] T[s, :]"))
                bb.bump_version(T.buf)
        else:
            W = ol.empty([C, C])
            for s in builtins.range(C - 1):
                n = C - 1 - s
                row = Tile(T.b, T.base + s * T.rs + c0, (n, s + 1), 0, T.buf)  # T[s, ..s] a row
                _vop(I.V_MUL, W[0:n, 0:s + 1], row, NL[s + 1:C, s:s + 1][:, None])
                tv = T[s + 1:C, c0:c0 + s + 1]
                _vop(I.V_ADD, tv, tv, W[0:n, 0:s + 1])
            del W
        del NL
        # M2: U = T diag(beta) V; M3: W = -T diag(beta 2^gc) K; M5: U += W S
        U = _split(_q2(T, cscale=csb), *vt)
        Wn = _split(_q2(T, cscale=csg), *kt)
        wq = _q2(Wn, free=True)
        del Wn
        _split(wq, *mat(f"st{'ab'[a]}"), acc=U)
        del wq
        # U's rows (M7's streamed side and M8's stationary, by the transpose)
        _qst2(U, m["uq", "hi"][c0:c1], m["uq", "lo"][c0:c1])
        del U
    del QK
    _transpose(eye, mat("uq", 2 * C), XT[:, 0:2 * C])
    qst(XT, "ut")
    ut = mat("ut")
    for a in (0, 1):
        c0, c1 = a * C, (a + 1) * C
        h, j = "ab"[a], 2 * p + a
        sq = mat(f"st{h}")
        # M6: O = (2^gc q) S; M7: O += A U
        Oa = _split(q6[a], *sq)
        q6[a] = None
        _split(_q2(A, cscale=mk[a]), *ut, acc=Oa)
        ol.store(_at(out[0:C, j * d:(j + 1) * d], ao), Oa)
        del Oa
        # M8: St = 2^gl St + U^T diag(2^(gl - gc)) K; its rows for the next chunk's M5 / M6
        gc, gl = GC[a, :], GC[a, C - 1:C]
        csd = cs[a][2]
        csd[c0:c1].set(ol.exp2(gl - gc))
        S = St[a]
        S.set(S * ol.exp2(gl))
        _split(_q2(XT, cscale=csd), *kt, acc=S)
        qst(S, f"st{h}")
