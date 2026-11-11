"""The MX column and the quantizer on the RTL against the ISA simulator (docs/isa.md "MM",
"Weight formats", "Quantization", "QACT", "QST"; docs/mx_mxu.md): MMs in every weight format
(int8x with -128, int4x, fp4x6, NVFP4) with UNIT, RSC, PAIR, k0, ACC, RMAX and ASCALE, the
scale words' corners (Sw negative, signed zeros, denormals, the largest finite, inf, NaN), groups
at the shift window's edge (sh = 0 kept, sh = -1 dropped), at MCOLS 4 and 8 and 1, 2 or 4
epilogue lanes; QACT and QST in every mode (ROW, CSCALE, RSCALE, DUP, HALF, element strides,
unaligned rows, KB not a multiple of 4, zero, inf and denormal rows, a chunk holding two QACT
groups at D = 32 with 16 lanes), and the quantizer's rounding ties (q8 rounds x * (inv * 2^d)
half to even), read back through MMs with identity weights. Each runs the same program on both
and compares DRAM and TMEM bit for bit."""
import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I, quant as Q, rtlsim
from opentpu.isasim import Machine

from test_mx import _scale_rows

F32_CORNERS = [0xBF400000, 0x00000000, 0x80000000, 0x00000001, 0x007FFFFF, 0x00800000,
               0x7F7FFFFF, 0x7F800000, 0x7FC00000]


def _both(cfg, prog, dram, uarch=None):
    m = Machine(cfg, [prog], [dram.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [dram.copy()], uarch=uarch)
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, (f"{len(bad)} TMEM words differ, first {bad[:6]}: RTL "
                           f"{[hex(int(v)) for v in tmems[0][bad[:6]]]} ISA "
                           f"{[hex(int(v)) for v in m.slices[0].tmem[bad[:6]]]}")
    bad = np.nonzero(drams[0] != m.slices[0].dram)[0]
    assert len(bad) == 0, (f"{len(bad)} DRAM bytes differ, first at {bad[:6]}: RTL "
                           f"{drams[0][bad[:6]]} ISA {m.slices[0].dram[bad[:6]]}")
    return m


def _put(dram, at, a):
    b = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
    dram[at:at + b.size] = b


X, WA, SA, OUT, SC, AL = 0, 0x40000, 0x80000, 0x8000, 0xA0000, 0xB0000


def _xrows(rng, rows, n, g, spread=6.0):
    """Rows of n elements whose groups of g span a wide range of magnitudes (offsets d from 0 to
    beyond the shift window), with a zero group."""
    x = (rng.standard_normal((rows, n)) *
         np.exp(rng.uniform(-spread, spread, (rows, n // g, 1))).repeat(g, 2).reshape(rows, n))
    x = x.astype(np.float32)
    x[0, g:2 * g] = 0
    return x


# wf, D, MCOLS, pair, rsc, k0, E
MM_CASES = [
    (I.W8, 32, 4, False, False, 0, 0), (I.W8, 128, 4, False, False, 3, 1),
    (I.W8, 32, 8, False, False, 1, 4), (I.W8, 128, 4, False, True, 0, 2),
    (I.W4I, 32, 4, False, False, 1, 0), (I.W4I, 128, 4, True, False, 2, 2),
    (I.W4F, 128, 4, False, False, 0, 1), (I.W4F, 32, 8, True, False, 0, 4),
    (I.W4F, 128, 4, False, True, 0, 0), (I.WNV, 128, 4, False, False, 1, 2),
    (I.WNV, 128, 8, True, False, 2, 1), (I.WNV, 32, 4, False, False, 0, 0),
]


@pytest.mark.parametrize("wf,D,mcols,pair,rsc,k0,E", MM_CASES)
def test_mm_mx_rtl(have_verilator, wf, D, mcols, pair, rsc, k0, E):
    """QACT, then MMs over its rows: with scales (rows 0 and 1 of each block at the window's
    edge for ACT row 0, sh = 0 and sh = -1; random offsets elsewhere, many past the window),
    UNIT, ACC onto the first, RMAX, ASCALE; Sw corners in rows 16.. . KB = 3 (PAIR's last chunk
    holds one block)."""
    cfg = Config(S=1, D=D, MCOLS=mcols, PAIR=pair, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(100 * wf + D + mcols + 3 * pair + 5 * rsc + k0)
    KB, M = 3, (mcols // 2 if pair else mcols)
    N = 16 + (0 if rsc else len(F32_CORNERS))
    W = I.MX_W[wf]
    x = _xrows(rng, M, KB * D, D // 4)
    KBr = k0 + KB
    if wf == I.W8:
        wel = rng.integers(-128, 128, (N, KB * D))
        wel[:, ::7] = -128
        data = wel.astype(np.int8).view(np.uint8)
    else:
        data = Q.pack4(rng.integers(0, 16, (N, KB * D)))
    rs = -(-data.shape[1] // D) * D
    rows, *_ = _scale_rows(rng, wf, N, KBr, rsc, True)
    rows = rows.copy()
    sw = rows[:, :4].copy().view(np.uint32)
    if not rsc:
        sw[16:, 0] = F32_CORNERS
        # rows 0 / 1: every block's group 0 at sh = 0 / -1 for ACT row 0's offsets
        _, _, d = F.quantize_mx(x, D // 4)
        for n, off in ((0, 0), (1, 1)):
            for k in range(KB):
                da = int(d[0, 4 * k])
                kk = k0 + k
                if wf == I.W8:
                    rows[n, 4 + kk] = max(0, W - da + off)
                elif wf == I.WNV:
                    rows[n, 16 + 8 * kk] = (rows[n, 16 + 8 * kk] & 7) | (min(15, da + 2 - off) << 3)
                else:
                    rows[n, 8 + 4 * kk] = max(0, W - da + off)
    rows[:, :4] = sw.view(np.uint8)
    srs = rows.shape[1]
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    for n in range(N):
        dram[WA + n * rs:WA + n * rs + data.shape[1]] = data[n]
    _put(dram, SA, rows)
    alpha = rng.uniform(-1.5, 1.5, M).astype(np.float32)
    _put(dram, AL, alpha)
    mm = dict(sa=WA, n=N, kb=KB, rs=rs, ors=N + 1, m=M, ab=1, srs=srs, wf=wf, pair=pair, k0=k0,
              rsc=rsc)
    prog = [I.ld(X, 0, x.size), I.ld(AL, 0x7000, M), I.qact(0, M, 1, KB, KB * D, dup=pair),
            I.mm(ssa=SA, out=OUT, **mm),
            I.mm(ssa=SA, out=OUT + 4096, **mm, unit=True, rmax=True),
            I.mm(ssa=SA, out=OUT, **mm, acc=True, rmax=True),
            I.mm(ssa=0, out=OUT + 4096, **mm, unit=True, acc=True, ascale=0x7000),
            I.halt()]
    m = _both(cfg, prog, dram, uarch={"MXU_E": E})
    if not rsc:                                    # the window's edge was reached, both ways
        s = m.slices[0]
        da = s.adoff[0, 1:1 + KB, 0].astype(int)
        if wf == I.WNV:
            E4 = rows[:2, 16 + 8 * (k0 + np.arange(KB))] >> 3
            sh = np.maximum(E4, 1) - 2 - da
        else:
            col = 4 + k0 + np.arange(KB) if wf == I.W8 else 8 + 4 * (k0 + np.arange(KB))
            sh = W - rows[:2, col].astype(int) - da
        assert (sh[0] == 0).any() and (sh[1] == -1).any(), sh


@pytest.mark.parametrize("mcols,rows,wf,D", [(4, 8, I.W8, 32), (2, 6, I.W4F, 32),
                                             (4, 12, I.WNV, 32), (4, 16, I.W4F, 32),
                                             (4, 16, I.WNV, 128)])
def test_mm_mx_replay_rtl(have_verilator, mcols, rows, wf, D):
    """M > MCOLS: each streamed chunk and its scale entry serve every group of MCOLS rows (up to
    ACT_ROWS 16, the production build's, at the card's D = 128)."""
    cfg = Config(S=1, D=D, MCOLS=mcols, ACT_ROWS=rows, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(300 + rows)
    KB, N = 4, 9
    x = _xrows(rng, rows, KB * D, D // 4, 3.0)
    data = (rng.integers(-128, 128, (N, KB * D)).astype(np.int8).view(np.uint8) if wf == I.W8
            else Q.pack4(rng.integers(0, 16, (N, KB * D))))
    rs = -(-data.shape[1] // D) * D
    srows, *_ = _scale_rows(rng, wf, N, KB + 1, False, False)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    for n in range(N):
        dram[WA + n * rs:WA + n * rs + data.shape[1]] = data[n]
    _put(dram, SA, srows)
    prog = [I.ld(X, 0, x.size), I.qact(0, rows, 0, KB, KB * D),
            I.mm(WA, SA, OUT, N, KB, rs, N + 3, rows, 0, srows.shape[1], wf=wf, k0=1),
            I.mm(WA, SA, OUT, N, KB, rs, N + 3, rows, 0, srows.shape[1], wf=wf, acc=True),
            I.halt()]
    _both(cfg, prog, dram)


def _qrows(rng, R, n, g):
    """Rows for the quantizer: wide-ranging groups (and a zero group), then a zero row, a row
    with an infinity, a row of denormals and normals (denormals flush), a NaN row."""
    x = _xrows(rng, R, n, g)
    x[1] = 0
    x[2, 3] = np.inf
    x[3, :] = F.from_bits(rng.integers(1, 0x800000, n).astype(np.uint32))
    x[3, ::5] *= 0
    x[3, ::5] += 1e-30
    x[4, 7] = np.nan
    return x


@pytest.mark.parametrize("D,lanes,xd,r1,pw", [
    (32, 8, None, 0, 1), (32, 16, None, 0, 1), (128, 8, None, 0, 1), (128, 16, None, 0, 1),
    (32, 8, 0, 0, 1), (32, 8, 32, 0, 1), (128, 8, 128, 0, 1), (128, 8, None, 1, 1),
    (32, 8, 0, 1, 1), (32, 8, 32, 1, 1), (128, 8, None, 1, 4), (128, 8, None, 0, 2),
    (32, 8, 32, 0, 4), (32, 8, 0, 0, 2), (32, 16, None, 0, 2)])
def test_quantizer_modes_rtl(have_verilator, D, lanes, xd, r1, pw):
    """QACT (ROW or not, CSCALE, RSCALE, DUP) read back by identity MMs, UNIT (y = q * Sa *
    2^-da; groups past the window drop) and int8x; QST (ROW or not, HALF, element strides 1 to
    3, rows not word aligned, KB 1 to 6) into DRAM with its scale rows. xd: the quantizer's row
    buffer (QUANT_XD; default 1024 chunks: every QACT here goes through it), 0 (none: two passes
    over TMEM), or one row and a third of the next (the reads wait for free entries). r1:
    RSCALE's row factor read once a row (QUANT_R1). pw: a buffered pass 1's chunks a cycle
    (QUANT_PW; without the buffer, a chunk's bytes in its lane group of the wide write)."""
    cfg = Config(S=1, D=D, MCOLS=4, LANES=lanes, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(D + lanes)
    R, KB = 6, 3
    x = _qrows(rng, R, 2 * KB * D, D // 4)
    cs = rng.uniform(0.25, 4.0, 2 * KB * D).astype(np.float32)
    rsf = rng.uniform(0.25, 4.0, R).astype(np.float32)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    _put(dram, X + x.nbytes, cs)
    _put(dram, X + x.nbytes + cs.nbytes, rsf)
    _put(dram, WA, np.eye(D, dtype=np.int8))
    int8x = np.zeros((D, 8), np.uint8)
    int8x[:, :4] = np.array([1.0], np.float32).view(np.uint8)   # Sw 1, dw 0
    _put(dram, SC, int8x)
    T = 2 * KB * D * R
    prog = [I.ld(X, 0, x.size + cs.size + rsf.size)]
    out = 0x2000                                   # after the rows and factors
    # (src row, ROW, CSCALE, RSCALE, DUP): rows 0..3 (a zero group, a zero row, an infinity,
    # denormals), rows 2..5 (a NaN)
    for r0, row, csf, rscf, dup in [(0, False, False, False, False), (0, True, False, False, False),
                                    (0, False, True, True, False), (2, True, True, False, False),
                                    (0, False, False, True, True), (2, False, False, False, False)]:
        rows = 2 if dup else 4
        prog.append(I.qact(r0 * 2 * KB * D, rows, 0, 2 * KB, 2 * KB * D, row=row, dup=dup,
                           cscale=T if csf else None,
                           rscale=T + 2 * KB * D + r0 if rscf else None))
        for k in range(2 * KB):
            prog.append(I.mm(WA, 0, out, D, 1, D, D, 2 * rows if dup else rows, k, 0,
                             unit=True))
            prog.append(I.mm(WA, SC, out + 4 * D, D, 1, D, D, rows, k, 8))
            out += 8 * D
    assert out < cfg.TMEM_WORDS
    sd = 0x50000
    for row, half, es, off, kb in [(False, False, 1, 0, KB), (True, False, 1, 0, 2 * KB),
                                   (True, True, 1, 0, 2 * KB), (False, False, 2, 1, 5),
                                   (False, False, 3, 2, 1), (True, False, 1, 3, 2),
                                   (False, False, 1, 4, 6)]:
        n = R
        drs = kb * D * es + 4
        prog.append(I.qst(0, sd + off, sd + 0x6000, n, kb, 2 * KB * D, drs, es, row=row,
                          half=half))
        sd += 0x8000
    prog.append(I.halt())
    _both(cfg, prog, dram, uarch=({} if xd is None else {"QUANT_XD": xd}) |
                                ({"QUANT_R1": 1} if r1 else {}) |
                                ({"QUANT_PW": pw} if pw > 1 else {}))


def _norm_qact_cycles(cs: bool, uarch: dict) -> int:
    """A norm's QACT at targets/bcu1525-x2.toml's one-copy TMEM (1 x 5120 elements: 40 blocks at D
    128, RSCALE, CSCALE when `cs`, DUP): its cycles, start to end, at `uarch` over the TMEM's."""
    import re
    from opentpu import target as T
    ua = {k: v for k, v in T.sim_uarch("bcu1525-x2").items()
          if k.startswith("TMEM_") or k in ("RPB", "WPB")}
    cfg = Config(S=1, D=128, MCOLS=8, DRAM_BYTES=1 << 20)
    n = 40 * cfg.D
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, np.full(2 * n + 8, 1.0, np.float32))
    prog = [I.ld(X, 0, 2 * n + 8), I.qact(0, 1, 0, 40, n, cscale=n if cs else None,
                                           rscale=2 * n, dup=True), I.halt()]
    _, _, st = rtlsim.run(cfg, [prog], [dram], trace=True, uarch=ua | uarch)
    S = {s: int(c) for c, s in re.findall(r"T0 S c=(\d+) s=(\d+)", st["trace"])}
    E = {s: int(c) for c, s in re.findall(r"T0 E c=(\d+) s=(\d+)", st["trace"])}
    return E["1"] - S["1"]


def test_rscale_once_a_row_rtl(have_verilator):
    """QUANT_R1 at the WG targets' one-copy TMEM: a norm's QACT (CSCALE and RSCALE) takes the
    CSCALE-only cycles once RSCALE's row factor is read once a row; read with every chunk, the
    reads of Q, Q2 and Q3 do not fit the banks in a cycle (1,653 cycles against 1,333;
    docs/weight_groups.md). The values are test_quantizer_modes_rtl's."""
    alone, both = _norm_qact_cycles(False, {}), _norm_qact_cycles(True, {})
    once = _norm_qact_cycles(True, {"QUANT_R1": 1})
    assert once <= alone + 4 and both > alone + 200, (alone, both, once)


def test_wide_pass1_rtl(have_verilator):
    """QUANT_PW at the WG targets' TMEM: a norm's QACT (one row, as in decode) pops its pass 1 from the
    row buffer PW chunks a cycle: its 640 pass-1 cycles at LANES 8 become 640 / PW (pass 0, the
    scale and the pipeline unchanged; docs/weight_groups.md). The values are
    test_quantizer_modes_rtl's."""
    c = {pw: _norm_qact_cycles(True, {"QUANT_R1": 1} | ({"QUANT_PW": pw} if pw > 1 else {}))
         for pw in (1, 2, 4)}
    assert abs(c[1] - c[2] - 320) <= 4 and abs(c[1] - c[4] - 480) <= 4, c


def _ties(amax, rng):
    """Values x with F.mul(x, inv) exactly k + 0.5 (inv = 127 * recip(amax), the quantizer's)
    for the k where such an x <= amax exists, their 1-ulp neighbours, and +-0, shuffled."""
    inv = F.mul(F.f32(127), F.recip(F.f32(amax)))
    out = []
    for k in range(-128, 128):
        t = F.f32(k + 0.5)
        x0 = F.f32(np.float64(t) / np.float64(inv))
        for d in range(-4, 5):
            x = F.from_bits(np.uint32(int(F.bits(x0)) + d))
            if F.mul(x, inv) == t:
                out += [x, F.from_bits(F.bits(x) + np.uint32(1)),
                        F.from_bits(F.bits(x) - np.uint32(1))]
                break
    out = np.array(out + [F.f32(v) for v in (0.0, -0.0)], np.float32)
    out = out[np.abs(out) <= amax]
    return out[rng.permutation(len(out))]


def _lo_ties(amax, rng, n=240):
    """_ties at the split's lo byte: values x with F.mul(x, inv) exactly (k + 0.5) / 256 for n
    k spread over |p| < 127.5 (and at the saturation edges 127 + 127.5 / 256, 127.5 +- 2^-9),
    their 1-ulp neighbours, then q8's ties (_ties)."""
    inv = F.mul(F.f32(127), F.recip(F.f32(amax)))
    ks = np.concatenate([rng.choice(np.arange(-127 * 256 - 128, 127 * 256 + 128), n,
                                    replace=False), [127 * 256 + 127, 127 * 256 + 128,
                                                     -127 * 256 - 129, 127, -129]])
    out = []
    for k in ks:
        t = F.f32((int(k) + 0.5) / 256)
        x0 = F.f32(np.float64(t) / np.float64(inv))
        for d in range(-4, 5):
            x = F.from_bits(np.uint32(int(F.bits(x0)) + d))
            if F.mul(x, inv) == t:
                out += [x, F.from_bits(F.bits(x) + np.uint32(1)),
                        F.from_bits(F.bits(x) - np.uint32(1))]
                break
    out = np.concatenate([np.array(out, np.float32), _ties(amax, rng)])
    out = out[np.abs(out) <= amax]
    return out[rng.permutation(len(out))]


def _tie_rows(D, g, rng, nrows=8, KB=6, ties=_ties):
    """Rows of KB blocks with groups of g: the row's amax A at element 0, each group its offset
    d (0 to 5; element 0 of the group is +-A 2^-d, so the group's d is d) and the ties of A
    (`ties`: q8's, or _lo_ties) scaled by 2^-d elsewhere: x * (inv * 2^d) is a tie of A's.
    Returns x and its tie count (q8's, with the groups' offsets)."""
    x = np.zeros((nrows, KB * D), np.float32)
    for r in range(nrows):
        A = F.f32([1.0, 2.0, 0.75, 127.0][r] if r < 4 else rng.uniform(0.5, 3.0))
        t = ties(A, rng)
        for i in range(0, KB * D, g):
            d = 0 if i == 0 else int(rng.integers(0, 6))
            blk = np.resize(np.roll(t, -i), g) * np.float32(2.0 ** -d)
            blk[0] = A * np.float32(2.0 ** -d) * (1 if (r + i // g) % 2 else -1)
            x[r, i:i + g] = blk
    q_, s_, d = F.quantize_mx(x, g)
    inv = F.mul(F.f32(127), F.recip(np.abs(x).max(1)))
    ig = F.mul(inv[:, None], np.ldexp(F.f32(1), d.astype(np.int32)).astype(np.float32))
    p = F.mul(x.reshape(nrows, -1, g), ig[:, :, None])
    return x, int((p - np.floor(p) == 0.5).sum())


@pytest.mark.parametrize("D,row", [(32, False), (32, True), (128, False), (128, True)])
def test_quantizer_ties_qst_rtl(have_verilator, D, row):
    """QST (D-block offsets, or ROW) on exact ties of x * (inv * 2^d) and their neighbours: the
    bytes and scale rows land in DRAM."""
    cfg = Config(S=1, D=D, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(D + row)
    x, ties = _tie_rows(D, D, rng)
    assert ties > 100
    R, KB = x.shape[0], x.shape[1] // D
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    prog = [I.ld(X, 0, x.size),
            I.qst(0, 0x40000, 0x48000, R, KB, KB * D, KB * D, 1, row=row),
            I.qst(0, 0x50000, 0x58000, R, KB, KB * D, 2 * KB * D, 2, row=row),
            I.halt()]
    _both(cfg, prog, dram)


@pytest.mark.parametrize("row", [False, True])
def test_quantizer_ties_qact_rtl(have_verilator, row):
    """QACT (sub-block offsets, or ROW) on the same ties, read back by MMs with identity int8
    weights and UNIT: T[out + j, n] = q[j][n] * Sa[j] * 2^-da for each block."""
    cfg = Config(S=1, D=32, MCOLS=8, DRAM_BYTES=1 << 20)
    D = cfg.D
    rng = np.random.default_rng(5 + row)
    x, ties = _tie_rows(D, D // 4, rng)
    assert ties > 100
    R, KB = x.shape[0], x.shape[1] // D
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    _put(dram, WA, np.eye(D, dtype=np.int8))
    prog = [I.ld(X, 0, x.size), I.qact(0, R, 0, KB, KB * D, row=row)]
    for k in range(KB):
        prog.append(I.mm(WA, 0, OUT + k * R * D, D, 1, D, D, R, k, 0, unit=True))
    prog.append(I.halt())
    m = _both(cfg, prog, dram)
    got = m.slices[0].tget(OUT + np.arange(KB * R * D)).reshape(KB, R, D)
    q, s, d = F.quantize_mx(x, D // 4, row)
    q = q.reshape(R, KB, D).astype(np.int64)
    d = np.repeat(d.reshape(R, KB, 4), D // 4, 2).astype(np.int64)
    acc = np.where(d <= 16, q << np.maximum(16 - d, 0), 0)
    want = F.mul(F.i2f(acc), F.mul(s, F.f32(2.0 ** -16))[:, None, None])
    assert np.array_equal(got.view(np.uint32), want.transpose(1, 0, 2).view(np.uint32))


@pytest.mark.parametrize("D,lanes,xd,pw", [(32, 8, None, 1), (128, 8, None, 2), (32, 8, 0, 1),
                                           (32, 16, None, 2), (128, 16, 0, 1)])
def test_splitq_rtl(have_verilator, D, lanes, xd, pw):
    """SPLITQ (otpu_quant SPLIT): QACT and QST with flag LO, the split's lo bytes and the scale
    S * 2^-8, in test_quantizer_modes_rtl's modes (ROW, CSCALE, RSCALE, DUP; HALF, element
    strides, unaligned rows) over its rows (a zero group and row, an infinity, denormals, a NaN)
    and rows of the lo byte's rounding ties (and q8's, _lo_ties), through the row buffer or not
    (xd), PW 1 or 2; a QACT and a QST without the flag after them (hi again: lom follows the
    instruction). QACT read back by identity MMs (UNIT and int8x), QST in DRAM."""
    cfg = Config(S=1, D=D, MCOLS=4, LANES=lanes, DRAM_BYTES=1 << 20, SPLITQ=True)
    rng = np.random.default_rng(D + lanes + 3 * pw)
    R, KB = 6, 3
    x = _qrows(rng, R, 2 * KB * D, D // 4)
    xt, _ = _tie_rows(D, D // 4, rng, nrows=4, KB=2 * KB, ties=_lo_ties)
    p = []
    for r in range(4):                             # the lo ties land (256 p at k + 1/2)
        q_, s_, d_ = F.quantize_mx(xt[r:r + 1], D // 4)
        inv = F.mul(F.f32(127), F.recip(np.abs(xt[r]).max()))
        ig = F.mul(inv, np.ldexp(F.f32(1), d_.astype(np.int32)).astype(np.float32))
        p.append(F.mul(xt[r].reshape(-1, D // 4), ig.reshape(-1, 1)).astype(np.float64) * 256)
    assert sum(int((pp - np.floor(pp) == 0.5).sum()) for pp in p) > 40
    x = np.concatenate([x, xt])
    R = x.shape[0]
    cs = rng.uniform(0.25, 4.0, 2 * KB * D).astype(np.float32)
    rsf = rng.uniform(0.25, 4.0, R).astype(np.float32)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    _put(dram, X + x.nbytes, cs)
    _put(dram, X + x.nbytes + cs.nbytes, rsf)
    _put(dram, WA, np.eye(D, dtype=np.int8))
    int8x = np.zeros((D, 8), np.uint8)
    int8x[:, :4] = np.array([1.0], np.float32).view(np.uint8)   # Sw 1, dw 0
    _put(dram, SC, int8x)
    T = 2 * KB * D * R
    prog = [I.ld(X, 0, x.size + cs.size + rsf.size)]
    out = T + 2 * KB * D + R + 64
    for r0, row, csf, rscf, dup, lo in [(0, False, False, False, False, True),
                                        (0, True, False, False, False, True),
                                        (0, False, True, True, False, True),
                                        (2, True, True, False, False, True),
                                        (0, False, False, True, True, True),
                                        (6, False, False, False, False, True),
                                        (6, True, False, False, False, True),
                                        (2, False, False, False, False, False)]:
        rows = 2 if dup else 4
        prog.append(I.qact(r0 * 2 * KB * D, rows, 0, 2 * KB, 2 * KB * D, row=row, dup=dup,
                           cscale=T if csf else None,
                           rscale=T + 2 * KB * D + r0 if rscf else None, lo=lo))
        for k in range(2 * KB):
            prog.append(I.mm(WA, 0, out, D, 1, D, D, 2 * rows if dup else rows, k, 0,
                             unit=True))
            prog.append(I.mm(WA, SC, out + 4 * D, D, 1, D, D, rows, k, 8))
            out += 8 * D
    assert out < cfg.TMEM_WORDS
    sd = 0x50000
    for row, half, es, off, kb, lo in [(False, False, 1, 0, KB, True), (True, False, 1, 0, 2 * KB, True),
                                       (True, True, 1, 0, 2 * KB, True), (False, False, 2, 1, 5, True),
                                       (False, False, 3, 2, 1, True), (True, False, 1, 3, 2, True),
                                       (False, False, 1, 4, 6, True), (False, False, 1, 0, KB, False)]:
        drs = kb * D * es + 4
        prog.append(I.qst(0, sd + off, sd + 0x6000, R, kb, 2 * KB * D, drs, es, row=row,
                          half=half, lo=lo))
        sd += 0x8000
    assert sd < cfg.DRAM_BYTES
    prog.append(I.halt())
    _both(cfg, prog, dram, uarch=({} if xd is None else {"QUANT_XD": xd}) |
                                ({"QUANT_PW": pw} if pw > 1 else {}))


@pytest.mark.parametrize("D,lanes,pw", [(32, 8, 1), (128, 8, 2), (32, 16, 2), (128, 16, 1)])
def test_qhl_rtl(have_verilator, D, lanes, pw):
    """QHL (otpu_quant QHL): QACT flag HL, q into blocks ab .. ab + KB - 1 and lo into ab + KB ..
    ab + 2 KB - 1 from one read (each chunk popped twice from the row buffer), in ROW / not,
    CSCALE, RSCALE, over test_splitq_rtl's rows (a zero group and row, an infinity, denormals, a
    NaN, the lo byte's ties) and 8 rows at once (the 4 row banks reused), PW 1 or 2, LANES 8 or
    16 (a chunk of two groups at D 32); then a QACT and a QACT LO (the flag follows
    the instruction). Every block read back by identity MMs (UNIT and int8x: the records' S and
    S * 2^-8), with the compiler's wait entries (SEQ_WC: they cover lo's blocks)."""
    cfg, prog, dram, ua = _qhl_prog(D, lanes, pw)
    _both(cfg, prog, dram, uarch=ua)


@pytest.mark.parametrize("seed,wmx", [(1, False), (2, True)])
def test_qhl_power_up_rtl(have_verilator, monkeypatch, seed, wmx):
    """test_qhl_rtl's program at bcu1525-x2's quantizer (D 128, LANES 8, PW 2; wmx: R1 and its
    TMEM, the one pumped copy) from a random power-up (OTPU_X_INITIAL=unique, as test_rtl's
    test_random_power_up): HL's state (the pop's phase, the lo flag beside the pass-1 pipeline
    and the lanes') holds nothing past the reset. DRAM is the ISA's, and TMEM at every word the
    program writes (those an ISA run from TMEM all ones agrees on)."""
    from opentpu import target as T
    monkeypatch.setenv("OTPU_X_INITIAL", "unique")
    monkeypatch.setenv("OTPU_X_SEED", str(seed))
    cfg, prog, dram, ua = _qhl_prog(128, 8, 2)
    if wmx:
        ua |= {k: v for k, v in T.sim_uarch("bcu1525-x2:wmx").items()
               if k.startswith("TMEM_") or k in ("RPB", "WPB", "QUANT_R1")}
    m = Machine(cfg, [prog], [dram.copy()]).run()
    m1 = Machine(cfg, [prog], [dram.copy()])
    m1.slices[0].tmem[:] = 0xFFFFFFFF
    m1.run()
    assert np.array_equal(m1.slices[0].dram, m.slices[0].dram)
    wr = m.slices[0].tmem == m1.slices[0].tmem
    drams, tmems, _ = rtlsim.run(cfg, [prog], [dram.copy()], uarch=ua)
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0][wr], m.slices[0].tmem[wr])


def _qhl_prog(D, lanes, pw):
    """test_qhl_rtl's run: (cfg, program, DRAM, uarch)."""
    cfg = Config(S=1, D=D, MCOLS=4, ACT_ROWS=8, LANES=lanes, TMEM_WORDS=1 << 17,
                 DRAM_BYTES=1 << 20, SPLITQ=True, QHL=True, SEQ_WC=True)
    rng = np.random.default_rng(D + lanes + 5 * pw)
    KB = 4
    x = _qrows(rng, 8, KB * D, D // 4)
    xt, _ = _tie_rows(D, D // 4, rng, nrows=4, KB=KB, ties=_lo_ties)
    x = np.concatenate([x, xt])
    R = x.shape[0]
    cs = rng.uniform(0.25, 4.0, KB * D).astype(np.float32)
    rsf = rng.uniform(0.25, 4.0, R).astype(np.float32)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    _put(dram, X + x.nbytes, cs)
    _put(dram, X + x.nbytes + cs.nbytes, rsf)
    _put(dram, WA, np.eye(D, dtype=np.int8))
    int8x = np.zeros((D, 8), np.uint8)
    int8x[:, :4] = np.array([1.0], np.float32).view(np.uint8)   # Sw 1, dw 0
    _put(dram, SC, int8x)
    T = KB * D * R
    prog = [I.ld(X, 0, x.size + cs.size + rsf.size)]
    out = T + KB * D + R + 64
    for r0, rows, row, csf, rscf, hl, lo in [(0, 4, False, False, False, True, False),
                                             (0, 8, False, True, True, True, False),
                                             (4, 8, True, False, False, True, False),
                                             (8, 4, False, False, True, True, False),
                                             (2, 4, True, True, False, True, False),
                                             (0, 4, False, False, False, False, False),
                                             (8, 4, False, False, False, False, True)]:
        prog.append(I.qact(r0 * KB * D, rows, 0, KB, KB * D, row=row,
                           cscale=T if csf else None, rscale=T + KB * D + r0 if rscf else None,
                           lo=lo, hl=hl))
        for k in range(2 * KB if hl else KB):
            prog.append(I.mm(WA, 0, out, D, 1, D, D, rows, k, 0, unit=True))
            prog.append(I.mm(WA, SC, out + rows * D, D, 1, D, D, rows, k, 8))
            out += 2 * rows * D
    assert out < cfg.TMEM_WORDS
    prog.append(I.halt())
    return cfg, prog, dram, {"SEQ_WC": 1, "WIN": 16} | ({"QUANT_PW": pw} if pw > 1 else {})


def test_qhl_cycles_rtl(have_verilator):
    """QACT HL at bcu1525-x2:wmx's quantizer and TMEM (PW 2, R1; ACT_ROWS 16): a split operand's
    16 rows of one block (the chunked DeltaNet's stationaries) in about one QACT's cycles, not two
    (QACT then QACT LO): its pass 1 pops each chunk twice at PW 2, as fast as pass 0 reads, and
    the rows' passes overlap; the bytes are test_qhl_rtl's."""
    import re
    from opentpu import target as T
    ua = {k: v for k, v in T.sim_uarch("bcu1525-x2:wmx").items()
          if k.startswith("TMEM_") or k in ("RPB", "WPB", "QUANT_PW", "QUANT_R1")}
    ua |= {"SEQ_WC": 1, "WIN": 16}
    cfg = Config(S=1, D=128, MCOLS=8, ACT_ROWS=16, LANES=8, DRAM_BYTES=1 << 20, SPLITQ=True,
                 QHL=True, SEQ_WC=True)
    rng = np.random.default_rng(31)
    x = _qrows(rng, 16, cfg.D, cfg.D // 4)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    prog = [I.ld(X, 0, x.size), I.qact(0, 16, 0, 1, cfg.D, hl=True),
            I.qact(0, 16, 2, 1, cfg.D), I.qact(0, 16, 3, 1, cfg.D, lo=True), I.halt()]
    _, _, st = rtlsim.run(cfg, [prog], [dram], trace=True, uarch=ua)
    S = {s: int(c) for c, s in re.findall(r"T0 S c=(\d+) s=(\d+)", st["trace"])}
    E = {s: int(c) for c, s in re.findall(r"T0 E c=(\d+) s=(\d+)", st["trace"])}
    hl, two = E["1"] - S["1"], E["3"] - S["2"]
    print(f"QACT HL {hl} cycles, QACT + QACT LO {two}")
    assert hl < 0.6 * two, (hl, two)


def _qst_cycles(cfg, x, uarch, row=False):
    """One QST of `x`'s rows (rows mode, 16-byte aligned rows: QUANT_SWW's 16 bytes a cycle;
    row: ROW mode): its cycles, start to end, on the RTL at `uarch`."""
    import re
    R, n = x.shape
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    prog = [I.ld(X, 0, x.size), I.qst(0, 0x40000, 0x48000, R, n // cfg.D, n, n, 1, row=row),
            I.halt()]
    _, _, st = rtlsim.run(cfg, [prog], [dram], trace=True, uarch=uarch)
    S = {s: int(c) for c, s in re.findall(r"T0 S c=(\d+) s=(\d+)", st["trace"])}
    E = {s: int(c) for c, s in re.findall(r"T0 E c=(\d+) s=(\d+)", st["trace"])}
    return E["1"] - S["1"]


@pytest.mark.parametrize("D,lanes,pw,axi", [(32, 8, 2, False), (128, 8, 2, False),
                                            (128, 8, 2, True), (32, 16, 1, False)])
def test_qst_sww_rtl(have_verilator, D, lanes, pw, axi):
    """QUANT_SWW 4 (otpu_quant SWW, the SW port 16 bytes wide): QST's pass 1 in rows mode with
    element stride 1 writes 16 contiguous bytes a cycle where its row is 16-byte aligned (lanes 8
    .. 15 through the Q2 port at LANES 8; LANES 16: the Q port alone), and a word or a byte with
    its enables elsewhere: test_quantizer_modes_rtl's rows (a zero group and row, an infinity,
    denormals, a NaN) in rows aligned or not (rows of one QST at both), ROW, HALF, strides 2 and
    3, the scale rows, SPLITQ's flag LO, a CSCALE QACT after them (Q2 back to the factors), on the
    behavioural DRAM and on the native adapter (its SW gather over 16-byte writes, stalls): DRAM
    and TMEM the ISA simulator's bit for bit; and at bcu1525-x2's TMEM (one copy, two sub-banks:
    Q's 8 words and Q2's in different ones) a QST of aligned rows in under 0.6 x the cycles of
    SWW 1."""
    cfg = Config(S=1, D=D, MCOLS=4, LANES=lanes, DRAM_BYTES=1 << 20, SPLITQ=True)
    rng = np.random.default_rng(D + lanes + 7 * axi)
    R, KB = 6, 3
    x = _qrows(rng, R, 2 * KB * D, D // 4)
    cs = rng.uniform(0.25, 4.0, 2 * KB * D).astype(np.float32)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    _put(dram, X + x.nbytes, cs)
    _put(dram, WA, np.eye(D, dtype=np.int8))
    T = 2 * KB * D * R
    prog = [I.ld(X, 0, x.size + cs.size)]
    sd = 0x50000
    # (ROW, HALF, es, dst offset, KB, row stride pad, LO): aligned rows (pad 0, 16), rows aligned
    # one in four (pad 12 at offset 4, pad 4), strides 2 and 3, unaligned rows
    for row, half, es, off, kb, pad, lo in [(False, False, 1, 0, KB, 0, False),
                                            (True, False, 1, 0, 2 * KB, 16, False),
                                            (True, True, 1, 0, 2 * KB, 0, True),
                                            (False, False, 1, 4, 5, 12, False),
                                            (False, False, 1, 0, 2 * KB, 4, True),
                                            (False, False, 2, 1, 5, 4, False),
                                            (False, False, 3, 2, 1, 4, True),
                                            (True, False, 1, 3, 2, 4, False),
                                            (False, False, 1, 16, 1, 16, True)]:
        drs = kb * D * es + pad
        prog.append(I.qst(0, sd + off, sd + 0x6000, R, kb, 2 * KB * D, drs, es, row=row,
                          half=half, lo=lo))
        sd += 0x8000
    assert sd < cfg.DRAM_BYTES
    out = T + 2 * KB * D + 64
    prog.append(I.qact(0, 4, 0, 2 * KB, 2 * KB * D, cscale=T))
    for k in range(2 * KB):
        prog.append(I.mm(WA, 0, out, D, 1, D, D, 4, k, 0, unit=True))
        out += 4 * D
    assert out < cfg.TMEM_WORDS
    prog.append(I.halt())
    ua = {"QUANT_SWW": 4} | ({"QUANT_PW": pw} if pw > 1 else {})
    m = Machine(cfg, [prog], [dram.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [dram.copy()], uarch=ua,
                                 **(dict(axi=True, stall=30) if axi else {}))
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first {bad[:6]}"
    bad = np.nonzero(drams[0] != m.slices[0].dram)[0]
    assert len(bad) == 0, (f"{len(bad)} DRAM bytes differ, first at {bad[:6]}: RTL "
                           f"{drams[0][bad[:6]]} ISA {m.slices[0].dram[bad[:6]]}")
    if D == 128 and not axi:
        from opentpu import target as Tg
        tm = {k: v for k, v in Tg.sim_uarch("bcu1525-x2:wmx").items()
              if k.startswith("TMEM_") or k in ("RPB", "WPB")}
        xc = rng.standard_normal((8, 8 * D)).astype(np.float32)
        c1 = _qst_cycles(cfg, xc, tm | {"QUANT_PW": pw})
        c4 = _qst_cycles(cfg, xc, tm | ua)
        assert c4 < 0.6 * c1, (c1, c4)


def _qstb_prog(D, lanes, rng, R=9, KB=3):
    """test_qstb_rtl's program: QSTs of R rows (past the quantizer's NB = 4 row banks) that
    QUANT_QSTB buffers (ROW, element stride 1, dst and row stride 16-byte aligned, not HALF; LO
    too) between ones it does not (HALF, an unaligned dst, rows mode without ROW, stride 2), then
    a CSCALE QACT read back through identity MMs (the row buffer and Q2 back to QACT's)."""
    cfg = Config(S=1, D=D, MCOLS=4, LANES=lanes, DRAM_BYTES=1 << 20, SPLITQ=True)
    x = _qrows(rng, R, 2 * KB * D, D // 4)
    cs = rng.uniform(0.25, 4.0, 2 * KB * D).astype(np.float32)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    _put(dram, X + x.nbytes, cs)
    _put(dram, WA, np.eye(D, dtype=np.int8))
    T = 2 * KB * D * R
    prog = [I.ld(X, 0, x.size + cs.size)]
    sd = 0x50000
    # (ROW, HALF, es, dst offset, KB, row stride pad, LO)
    for row, half, es, off, kb, pad, lo in [(True, False, 1, 0, KB, 0, False),
                                            (True, False, 1, 16, 2 * KB, 16, True),
                                            (True, False, 1, 0, 1, 0, False),
                                            (True, True, 1, 0, 2 * KB, 0, False),
                                            (True, False, 1, 4, KB, 12, False),
                                            (False, False, 1, 0, KB, 0, False),
                                            (True, False, 2, 0, KB, 0, False),
                                            (True, False, 1, 0, 2 * KB, 32, True)]:
        drs = kb * D * es + pad
        prog.append(I.qst(0, sd + off, sd + 0x6000, R, kb, 2 * KB * D, drs, es, row=row,
                          half=half, lo=lo))
        sd += 0x8000
    assert sd < cfg.DRAM_BYTES
    out = T + 2 * KB * D + 64
    prog.append(I.qact(0, 4, 0, 2 * KB, 2 * KB * D, cscale=T))
    for k in range(2 * KB):
        prog.append(I.mm(WA, 0, out, D, 1, D, D, 4, k, 0, unit=True))
        out += 4 * D
    assert out < cfg.TMEM_WORDS
    prog.append(I.halt())
    return cfg, prog, dram


@pytest.mark.parametrize("D,lanes,pw,xd,axi", [(32, 8, 2, None, False), (128, 8, 2, None, False),
                                               (128, 8, 2, None, True), (32, 16, 1, None, False),
                                               (32, 8, 2, 32, False), (128, 8, 2, 64, True)])
def test_qstb_rtl(have_verilator, D, lanes, pw, xd, axi):
    """QUANT_QSTB (otpu_quant QSTB, with QUANT_SWW 4 at 16 writer lanes): QST rows in ROW mode
    through the row buffer, each row's pass 1 popped from it (16 bytes a cycle) and its scale word
    written first, while the next rows' pass 0 run; the other QSTs as before (_qstb_prog). DRAM
    and TMEM the ISA simulator's bit for bit, on the behavioural DRAM and the native adapter
    (stalls); xd: a row buffer of 32 chunks (D 32: rows of 12 and 24 chunks, the 24 one row in it
    at a time: the credits) or 64 (D 128: rows of 16 and 48, and of 96 past it: QACT's two
    passes, QST unbuffered)."""
    cfg, prog, dram = _qstb_prog(D, lanes, np.random.default_rng(D + lanes + 3 * axi))
    ua = {"QUANT_SWW": 4, "QUANT_QSTB": 1} | ({"QUANT_PW": pw} if pw > 1 else {}) | \
        ({"QUANT_XD": xd} if xd else {})
    m = Machine(cfg, [prog], [dram.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [dram.copy()], uarch=ua,
                                 **(dict(axi=True, stall=30) if axi else {}))
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first {bad[:6]}"
    bad = np.nonzero(drams[0] != m.slices[0].dram)[0]
    assert len(bad) == 0, (f"{len(bad)} DRAM bytes differ, first at {bad[:6]}: RTL "
                           f"{drams[0][bad[:6]]} ISA {m.slices[0].dram[bad[:6]]}")


@pytest.mark.parametrize("seed", [1, 2])
def test_qstb_power_up_rtl(have_verilator, monkeypatch, seed):
    """test_qstb_rtl's program at bcu1525-x2:wmx's quantizer and TMEM (D 128, LANES 8, PW 2,
    SWW 4, R1, the one pumped copy) from a random power-up (OTPU_X_INITIAL=unique): QSTB's state
    (the scale word's flag beside the pops, the pops' address line) holds nothing past the reset.
    DRAM is the ISA's, and TMEM at every word the program writes."""
    from opentpu import target as T
    monkeypatch.setenv("OTPU_X_INITIAL", "unique")
    monkeypatch.setenv("OTPU_X_SEED", str(seed))
    cfg, prog, dram = _qstb_prog(128, 8, np.random.default_rng(40 + seed))
    ua = {k: v for k, v in T.sim_uarch("bcu1525-x2:wmx").items()
          if k.startswith("TMEM_") or k in ("RPB", "WPB", "QUANT_R1", "QUANT_PW", "QUANT_SWW")}
    ua |= {"QUANT_QSTB": 1}
    m = Machine(cfg, [prog], [dram.copy()]).run()
    m1 = Machine(cfg, [prog], [dram.copy()])
    m1.slices[0].tmem[:] = 0xFFFFFFFF
    m1.run()
    wr = m.slices[0].tmem == m1.slices[0].tmem
    drams, tmems, _ = rtlsim.run(cfg, [prog], [dram.copy()], uarch=ua)
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0][wr], m.slices[0].tmem[wr])


def test_qstb_cycles_rtl(have_verilator):
    """QUANT_QSTB at bcu1525-x2:wmx's quantizer and TMEM: a QST of 64 rows of 128 (ROW, aligned)
    takes about pass 0's 16 cycles a row (each row's pass 1 under the next rows' pass 0), against
    ~67 unbuffered (each row waiting for its scale before its pass 1)."""
    from opentpu import target as T
    tm = {k: v for k, v in T.sim_uarch("bcu1525-x2:wmx").items()
          if k.startswith("TMEM_") or k in ("RPB", "WPB", "QUANT_R1", "QUANT_PW", "QUANT_SWW")}
    cfg = Config(S=1, D=128, MCOLS=8, LANES=8, DRAM_BYTES=1 << 20, SPLITQ=True)
    x = np.random.default_rng(9).standard_normal((64, 128)).astype(np.float32)
    c0 = _qst_cycles(cfg, x, tm, row=True)
    c1 = _qst_cycles(cfg, x, tm | {"QUANT_QSTB": 1}, row=True)
    print(f"QST 64 x 128 ROW: {c0} cycles unbuffered ({c0 / 64:.1f} a row), {c1} with QSTB "
          f"({c1 / 64:.1f} a row)")
    assert c1 < 20 * 64 and c1 < 0.4 * c0, (c0, c1)

