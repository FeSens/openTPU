"""SE v2's composite functions (rtl/vpu/otpu_se_comp.sv) on their own: EXP2, EXP2SUB, RECIP,
RSQRT and LOG2 microcoded over NS generic multiply-add stages per lane
(sim/verilator/tb_se_comp.sv), bit for bit against opentpu/fp32.py. Directed special values
(+-0, denormals, +-inf, NaN, the range limits of each function) and random words over every
exponent; chunks issued as the VPU issues them (hold, the latency rule), with random gaps and
random enable stalls; the stage units inside the module and outside it (EXT, built as SE's
slot 0, U and Q)."""
import numpy as np
import pytest

from opentpu import fp32 as F
from opentpu import rtlsim

V_EXP2, V_RECIP, V_RSQRT, V_EXP2SUB, V_LOG2 = 9, 10, 11, 14, 15
FUNCS = [V_EXP2, V_EXP2SUB, V_RECIP, V_RSQRT, V_LOG2]

EDGE = F.f32([0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 2.5, -2.5, 127.5, -127.5, 126.9, 3e38, -3e38,
              1.2e-38, -1.2e-38, 1e-45, -1e-45, 1e-40, -1e-40, 2.0 ** -126, -(2.0 ** -126),
              1 - 2 ** -24, 1 + 2 ** -23, 65504.0, -126.0, -126.5, -125.99, -127.0, 127.99,
              128.0, -128.0, 129.0, 0.4999999, 0.5000001, np.inf, -np.inf, 2.0 ** 126,
              -(2.0 ** 126), 2.0 ** 125.9, 1.7e38, 2.0 ** 127, 3.4028235e38, -3.4028235e38,
              1e-30, -1e-30, 2.0 ** -23, -(2.0 ** -23), 0.75, -0.75, 1.4142135, 0.70710677])
NANS = F.from_bits(np.array([0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFBFFFFF], np.uint32))


def _words(rng, n):
    """Random fp32 over many exponents, fully random words, and the edge values."""
    mant = rng.integers(0, 1 << 23, n, dtype=np.uint32)
    exp = rng.integers(1, 255, n, dtype=np.uint32)
    sign = rng.integers(0, 2, n, dtype=np.uint32)
    a = F.from_bits((sign << 31) | (exp << 23) | mant)
    b = F.from_bits(rng.integers(0, 1 << 32, n, dtype=np.int64).astype(np.uint32))
    c = F.from_bits((sign << 31) | (rng.integers(100, 160, n, dtype=np.uint32) << 23) | mant)
    return np.concatenate([a, b, c, EDGE, NANS]).astype(np.float32)


def _operands(rng, f, n):
    """x (and y for EXP2SUB) for n columns of function f."""
    w = _words(rng, n)
    pick = lambda k: rng.choice(w, k)
    if f in (V_EXP2, V_EXP2SUB):
        u = rng.random(n)
        x = np.where(u < 0.5, F.f32(rng.uniform(-140, 135, n)), pick(n)).astype(np.float32)
        x[u < 0.1] = rng.choice(EDGE, int((u < 0.1).sum()))
        y = np.zeros(n, np.float32)
        if f == V_EXP2SUB:
            y = np.where(rng.random(n) < 0.6, F.f32(rng.uniform(-80, 80, n)),
                         pick(n)).astype(np.float32)
            # softmax-like: x - max(x) <= 0, near the -126 limit and exact ties
            m = rng.random(n) < 0.3
            with np.errstate(invalid="ignore", over="ignore"):
                y[m] = x[m] + F.f32(rng.uniform(0, 140, int(m.sum())))
            t = rng.random(n) < 0.05
            y[t] = x[t]
        return x, y
    x = pick(n)
    if f == V_LOG2:
        near = F.from_bits(np.uint32(0x3F3504F3) + rng.integers(-8, 8, n).astype(np.uint32)
                           + (rng.integers(-3, 4, n).astype(np.uint32) << np.uint32(23)))
        x = np.where(rng.random(n) < 0.15, near, x).astype(np.float32)
    return x, np.zeros(n, np.float32)


def reference(f, x, y):
    """fp32.py's function (NaN inputs too: fp32.py follows the RTL's definitions, e.g.
    recip(+-NaN) is a zero with its sign, rsqrt(-NaN) is +0, exp2(NaN) is +inf)."""
    return F.f32(_reference(f, x, y))


def _reference(f, x, y):
    return {V_EXP2: lambda: F.exp2(x), V_EXP2SUB: lambda: F.exp2(F.sub(x, y)),
            V_RECIP: lambda: F.recip(x), V_RSQRT: lambda: F.rsqrt(x),
            V_LOG2: lambda: F.log2(x)}[f]()


def _program(rng, n_instr, lanes, fixed=None):
    """Instructions (runs of chunks of one function), as a list of (func, x, y) chunks."""
    chunks = []
    for _ in range(n_instr):
        f = fixed if fixed is not None else int(rng.choice(FUNCS))
        k = int(rng.choice([1, 2, 3, 7, 24, 25, 26, 40, 90]))
        x, y = _operands(rng, f, k * lanes)
        for i in range(k):
            chunks.append((f, x[i * lanes:(i + 1) * lanes], y[i * lanes:(i + 1) * lanes]))
    return chunks


def run_comp(chunks, tmp_path, lanes=8, ns=3, ext=0, ha=2, ipct=100, epct=0, seed=1):
    lines = [f"{len(chunks):x} {ipct:x} {epct:x} {seed:x}"]
    for f, x, y in chunks:
        lines.append(" ".join([f"{f:x}"] + [f"{int(w):08x}" for w in F.bits(x)] +
                              [f"{int(w):08x}" for w in F.bits(y)]))
    fin, fout = tmp_path / "in.hex", tmp_path / "out.txt"
    fin.write_text("\n".join(lines) + "\n")
    R = rtlsim.RTL
    exe = rtlsim.build("tb_se_comp", [R / "vpu/otpu_fp.sv", R / "vpu/otpu_fpipe.sv",
                                      R / "top/otpu_pkg.sv", R / "vpu/otpu_se_comp.sv",
                                      rtlsim.TB / "tb_se_comp.sv"],
                       {"LANES": lanes, "NS": ns, "EXT": ext, "HA": ha})
    r = rtlsim.run_sim([exe, f"+in={fin}", f"+out={fout}"], timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    got, cyc = {}, None
    for ln in fout.read_text().split("\n"):
        t = ln.split()
        if not t:
            continue
        if t[0] == "E":
            cyc, nref = int(t[1]), int(t[2])
            continue
        got[int(t[0])] = np.array([int(w, 16) for w in t[1:]], np.uint32)
    assert len(got) == len(chunks) and cyc is not None
    assert nref == 0, r.stdout[-3000:]
    return got, cyc


def check(chunks, got):
    bad = []
    for i, (f, x, y) in enumerate(chunks):
        exp = F.bits(reference(f, x, y))
        for l in np.nonzero(got[i] != exp)[0]:
            bad.append(f"chunk {i} f={f} lane {l}: x={F.bits(x)[l]:08x} y={F.bits(y)[l]:08x} "
                       f"exp={exp[l]:08x} got={got[i][l]:08x}")
    assert not bad, f"{len(bad)} mismatches:\n" + "\n".join(bad[:20])


@pytest.mark.parametrize("f", FUNCS)
def test_se_comp_each_function(f, tmp_path):
    """Every function alone, back to back at full rate: edge values and random words."""
    rng = np.random.default_rng(f)
    chunks = []
    x = np.concatenate([EDGE, NANS, -EDGE]).astype(np.float32)
    for y0 in (EDGE, np.zeros_like(EDGE)):
        xs, ys = np.concatenate([x, x]), np.concatenate([np.resize(y0, len(x)), x[::-1]])
        n = len(xs) // 8 * 8
        for i in range(0, n, 8):
            chunks.append((f, xs[i:i + 8], ys[i:i + 8]))
    chunks += _program(rng, 6, 8, fixed=f)
    got, _ = run_comp(chunks, tmp_path)
    check(chunks, got)


def test_se_comp_range_reduction(tmp_path):
    """EXP2's floor and i2f and LOG2's i2f(e) on their whole domains: every integer and half
    integer in [-130, 130], its neighbours one ulp up and down, a fine grid, and LOG2 on
    every exponent with the mantissa split's edges."""
    i = np.arange(-130, 131).astype(np.float32)
    up = np.nextafter(i, np.float32(np.inf)).astype(np.float32)
    dn = np.nextafter(i, np.float32(-np.inf)).astype(np.float32)
    grid = F.f32(np.arange(-130, 130, 1 / 64))
    x = np.concatenate([i, i + 0.5, up, dn, grid, -EDGE, EDGE]).astype(np.float32)
    ex = np.arange(0, 256, dtype=np.uint32) << np.uint32(23)
    lg = np.concatenate([F.from_bits(ex | m) for m in (0, 1, 0x3504F2, 0x3504F3, 0x7FFFFF)])
    lg = np.concatenate([lg, -lg]).astype(np.float32)
    chunks = []
    for f, v in [(V_EXP2, x), (V_EXP2SUB, x), (V_LOG2, lg)]:
        v = np.resize(v, -(-len(v) // 8) * 8).astype(np.float32)
        y = F.f32(np.resize([0.0, -0.0, 0.5, -1.0], len(v)))
        for k in range(0, len(v), 8):
            chunks.append((f, v[k:k + 8], y[k:k + 8]))
    got, _ = run_comp(chunks, tmp_path)
    check(chunks, got)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_se_comp_mixed_stalls(seed, tmp_path):
    """Random programs of all functions, random gaps and enable stalls (the latency rule and
    hold decide when a chunk enters)."""
    rng = np.random.default_rng(100 + seed)
    chunks = _program(rng, 24, 8)
    got, _ = run_comp(chunks, tmp_path, ipct=int(rng.choice([100, 60])), epct=25, seed=seed)
    check(chunks, got)


@pytest.mark.parametrize("ns,ext,ha", [(3, 1, 2), (2, 0, 2), (4, 0, 1), (4, 1, 3), (1, 0, 2)])
def test_se_comp_configs(ns, ext, ha, tmp_path):
    """Other stage counts and hold leads, and the stage units outside the module (EXT)."""
    rng = np.random.default_rng(10 * ns + ext)
    chunks = _program(rng, 16, 8)
    got, _ = run_comp(chunks, tmp_path, ns=ns, ext=ext, ha=ha, ipct=80, epct=20, seed=ns)
    check(chunks, got)


def test_se_comp_throughput(tmp_path):
    """Columns per cycle at full rate, 8 lanes, NS = 3: EXP2/EXP2SUB 8/3, RECIP 4, RSQRT and
    LOG2 2 (a pass of T = 25 cycles per NS ops)."""
    rng = np.random.default_rng(7)
    rates = {}
    for f, p in [(V_EXP2, 3), (V_EXP2SUB, 3), (V_RECIP, 2), (V_RSQRT, 4), (V_LOG2, 4)]:
        n = 25 * 40
        x, y = _operands(rng, f, n * 8)
        chunks = [(f, x[i * 8:(i + 1) * 8], y[i * 8:(i + 1) * 8]) for i in range(n)]
        got, cyc = run_comp(chunks, tmp_path)
        check(chunks, got)
        rates[f] = n * 8 / cyc
        assert abs(rates[f] - 8 / p) < 0.05 * 8 / p, rates
