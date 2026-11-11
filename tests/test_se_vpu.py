"""The stream engine's front (rtl/vpu/otpu_vpu.sv with HAS_SE): the VPU with its tail, driven
like the DMA drives it (sim/verilator/tb_se_vpu.sv), with independent VOPs on a TMEM model
around the streams. Streams: bit for bit against their op lists (tests/test_se_tail.py's
reference, every dmode x gate, A on slot 1 or 2, Q on or off, specials, random pe stalls), so
dot A on the VPU's slot-0 partial loop and u_vt is DSTEP's. VOPs (every function, the
reductions sharing slot 0's loop and u_vt with the streams): the TMEM against the ISA
simulator. Some VOPs are queued or in flight when a stream asks; the testbench fails on a TMEM
access, a VOP start or an unfinished VOP in stream mode, or a stream output outside it. FSUM
(otpu_vpu FSUM, Config.FSUM): EXP2SUBs with their row sums among them (rows 1..256, cols 64 and
up) and their takes 0..5 VOPs later, at CNS 3 and 5. VSIG (otpu_vpu VSIG, Config.VSIG): SIGMOID
and SILU among the elementwise VOPs at CNS 5, and at CNS 3 (six passes) on B2's VPU (FSUM and
OUTER off, MAC on) in every B mode. MAC (otpu_vpu MAC, Config.VMAC): MACs among
them (B per column or a scalar) at CNS 3 and 5, WBUF 0 and 1. OUTER1 (otpu_vpu OUTER1,
Config.OUTER1: a VPU without OUTER, OUTER with decay one on MAC's datapath): OUTERs of decay
one among wmx's VOPs; one with a decay word stops the simulation."""
import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I, rtlsim
from opentpu.isasim import Machine
from test_se_tail import L, _check, _fills, _special, _stream

TW = 1 << 16
NDATA = 8192
ELEM = [I.V_ADD, I.V_SUB, I.V_RSUB, I.V_MUL, I.V_MAX, I.V_MIN, I.V_COPY, I.V_EXP2, I.V_RECIP,
        I.V_RSQRT, I.V_ABS, I.V_FILL, I.V_EXP2SUB, I.V_LOG2]
RED = [I.V_RSUM, I.V_RSSQ, I.V_RDOT, I.V_RMAX]


FSUM_COLS = [64, 65, 71, 72, 100, 127, 128, 129, 191, 192, 193, 200, 255, 256, 257, 320]
NFSUM, NSIG, NMAC, NOUT1 = 8, 4, 8, 8   # the coverage _vops places itself (the tests assert it)


def _vops(rng, n, fsum=False, sig=False, mac=False, outer1=False, outer=True, sig_modes=False):
    """n independent VOPs: sources in [0, NDATA), each result in a fresh range (OUTER updates
    its own). fsum: a third of them EXP2SUBs with their row sums (I.exp2sub_sum) in a fresh range
    of their own, most of a few rows, some of 20-60 and one of 256 (A's rows overlapping), each
    taken (I.fsum_take) after 0..5 of the VOPs after it, in order (the next FSUMs' among them).
    sig: SIGMOID and SILU among the elementwise functions (a quarter of them).
    mac: a quarter of them MACs (I.mac), B per column (up to 256 columns, a third of them a row of
    dst: read by the fill before the first write) or a scalar, A in place (dst) for some.
    outer1: OUTERs of decay one only (a VPU with OUTER1), at least NOUT1 of them. outer False:
    no OUTERs (a VPU without OUTER or OUTER1).
    sig_modes: at least 2 NSIG SIGMOID / SILU, each function's taking the B modes in turn (every
    mode for both).
    At least NFSUM EXP2SUBs with their sums (fsum), NSIG SIGMOID / SILU (sig, half each) and NMAC
    MACs (mac): what a seed's draws leave out the last positions place, so the coverage the tests
    assert does not hang on the draws (a seed whose draws have it keeps them)."""
    out, nxt = [], NDATA
    nsig = NSIG * (1 + sig_modes) // 2 * sig
    need = {"fsum": NFSUM * fsum, I.V_SIGMOID: nsig, I.V_SILU: nsig,
            "mac": NMAC * mac, "outer": NOUT1 * outer1}   # the coverage still to place
    smode = {I.V_SIGMOID: 0, I.V_SILU: 0}       # sig_modes: each function's next B mode
    assert sum(need.values()) <= n
    big = fsum
    takes = []                                  # [VOPs left, the take], oldest first

    def fresh(k):
        nonlocal nxt
        a = nxt
        nxt += k + int(rng.integers(0, 3))
        return a

    def src(k):
        return int(rng.integers(0, NDATA - k))

    for i in range(n):
        for t in takes:
            t[0] -= 1
        while takes and takes[0][0] < 0:
            out.append(takes.pop(0)[1])
        kind = rng.choice(["elem", "elem", "red"] + ["outer"] * outer + ["fsum"] * (2 * fsum)
                          + ["mac"] * (2 * mac))
        short = [k for k, v in need.items() if v > 0]
        f = short[0] if short and sum(max(v, 0) for v in need.values()) >= n - i else None
        if f is not None:                       # placed: a kind, or an elementwise function
            kind = f if isinstance(f, str) else "elem"
        if kind in ("fsum", "mac", "outer"):
            need[str(kind)] -= 1
        rows, cols = int(rng.integers(1, 5)), int(rng.integers(1, 160))
        if kind == "fsum":
            cols = int(rng.choice(FSUM_COLS))
            ars = cols + int(rng.integers(0, 3))
            if big:                             # the row-sum buffer's 256 rows
                big, rows, cols, ars = False, 256, 64, int(rng.integers(0, 30))
            elif rng.integers(4) == 0:
                rows, cols = int(rng.integers(20, 61)), int(rng.choice(FSUM_COLS[:5]))
                ars = int(rng.integers(0, 90))
            bmode = int(rng.choice([I.B_FULL, I.B_ROW, I.B_COL]))
            brs = cols if bmode == I.B_FULL else 1 if bmode == I.B_ROW else 0
            if bmode == I.B_FULL and rows * cols + cols >= NDATA:
                brs = int(rng.integers(0, 30))
            b = src(cols) if bmode == I.B_COL else src(rows * brs + cols)
            drs = cols + int(rng.integers(0, 2))
            dst, sums = fresh(rows * drs), fresh(rows)
            out.append(I.exp2sub_sum(dst, src(rows * ars + cols), b, sums, rows, cols, drs, ars,
                                     brs, bmode))
            takes.append([int(rng.integers(0, 6)), I.fsum_take(sums, dst, rows, cols, drs)])
            continue
        if kind == "mac":
            cols = int(rng.choice([1, 8, 9, 64, 129, 200, 255, 256]))
            drs, ars = cols + int(rng.integers(0, 2)), cols + int(rng.integers(0, 3))
            dst = fresh(rows * drs)
            bm = I.B_SCALAR if rng.integers(4) == 0 else I.B_COL
            b = dst + int(rng.integers(0, rows)) * drs if rng.integers(3) == 0 else src(cols)
            a = src(rows * ars)
            if rng.integers(4) == 0:                # A in place
                a, ars = dst, drs
            out.append(I.mac(dst, a, b, rows, cols, drs, ars, bm,
                             imm=float(rng.choice(_special(rng, 16)))))
            continue
        if kind == "outer":
            drs = cols + int(rng.integers(0, 2))
            out.append(I.outer(fresh(rows * drs), src(cols), src(rows), src(cols), rows, cols, drs,
                               1, "one" if outer1 else str(rng.choice(["scalar", "column", "one"]))))
            continue
        func = int(rng.choice((ELEM + [I.V_SIGMOID, I.V_SILU] * 2 if sig else ELEM)
                              if kind == "elem" else RED))
        if f is not None and not isinstance(f, str):
            func = f
        if func in need:
            need[func] -= 1
        if func == I.V_RDOT and rng.integers(3) == 0:   # buffered row sums, many rows
            rows, cols = int(rng.integers(20, 70)), int(rng.integers(1, 80))
        ars = cols + int(rng.integers(0, 3))
        bmode = int(rng.integers(0, 4)) if func in I.READS_B else I.B_FULL
        if sig_modes and func in smode:
            bmode, smode[func] = smode[func] % 4, smode[func] + 1
        brs, b = 0, 0
        if bmode == I.B_FULL:
            brs = cols
            b = src(rows * brs)
        elif bmode == I.B_ROW:
            brs = 1
            b = src(rows + 1)
        elif bmode == I.B_COL:
            b = src(cols)
        drs = 1 if func in RED else cols + int(rng.integers(0, 2))
        dst = fresh(rows * drs if func not in RED else rows)
        out.append(I.vop(func, dst, src(rows * ars) if func != I.V_FILL else 0, b, rows, cols,
                         drs, ars, brs, bmode, imm=float(rng.choice(_special(rng, 16)))))
    out += [t for _, t in takes]
    assert nxt < TW
    return out


def run_front(streams, vops, rels, tm0, rng, tmp_path, wbuf, cns=3, fsum=False, sig=False,
              mac=False, outer1=False, outer=True, stops=None):
    """A 64-column stream is presented as the DMA presents it (ns = 16, pad64: each row's 8
    segments, then 8 of +0 whose Y is dropped). stops: the simulation must stop with it instead
    (a check of otpu_vpu's), and nothing is returned."""
    lines = [f"{len(vops):x}"]
    for ins in vops:
        lines.append(" ".join(f"{v:x}" for v in [ins.op, ins.flags] + list(ins.w)))
    for st, (nrel, dly) in zip(streams, rels):
        rows, cols = st["S"].shape
        fills = _fills(st, rng)
        pad = cols == 64
        S = np.concatenate([st["S"], np.zeros_like(st["S"])], 1) if pad else st["S"]
        segs = F.bits(S).reshape(-1, L)
        pct = int(rng.choice([100, 70, 40]))
        drain = 3 if rng.integers(2) else 1        # pe after the last output or not
        lines.append(" ".join(f"{v:x}" for v in [drain, S.shape[1] // L, rows, int(st["a_en"]),
                                                  int(st["a_sel"]), st["dmode"], st["g_src"],
                                                  int(st["q_en"]), len(fills), len(segs), pct,
                                                  nrel, dly, int(pad)]))
        lines += [f"{fk:x} {fi:x} " + " ".join(f"{int(w):08x}" for w in d) for fk, fi, d in fills]
        lines += [" ".join(f"{int(w):08x}" for w in s) for s in segs]
    lines.append("0")
    fin, fout = tmp_path / "in.hex", tmp_path / "out.txt"
    ftm, fdump = tmp_path / "tmem.hex", tmp_path / "dump.hex"
    fin.write_text("\n".join(lines) + "\n")
    ftm.write_text("\n".join(f"{int(w):08x}" for w in tm0) + "\n")
    R = rtlsim.RTL
    exe = rtlsim.build("tb_se_vpu", [R / "vpu/otpu_fp.sv", R / "vpu/otpu_fpipe.sv",
                                     R / "top/otpu_pkg.sv", R / "vpu/otpu_vtree.sv",
                                     R / "tech/generic/otpu_tech_ram.sv",
                                     R / "vpu/otpu_se_comp.sv", R / "vpu/otpu_se_tail.sv",
                                     R / "vpu/otpu_vpu.sv", rtlsim.TB / "tb_se_vpu.sv"],
                        {"WBUF": int(wbuf), "CNS": cns, "FSUM": int(fsum), "VSIG": int(sig)}
                        | ({"MAC": 1} if mac else {})
                        | ({"OUTER": 0, "OUTER1": 1} if outer1 else {})
                        | ({"OUTER": 0} if not outer else {}))
    r = rtlsim.run_sim([exe, f"+in={fin}", f"+out={fout}", f"+tmem={ftm}", f"+dump={fdump}"],
                       timeout=900)
    if stops is not None:
        log = r.stdout + r.stderr
        assert r.returncode != 0 and stops in log, log[-3000:]
        return None
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out, cur, done = [], {"Y": [], "O": []}, False
    for ln in fout.read_text().split("\n"):
        t = ln.split()
        if not t:
            continue
        if t[0] == "E":
            out.append(cur)
            cur = {"Y": [], "O": []}
        elif t[0] == "T":
            done = True
        else:
            cur[t[0]].append([int(v, 16) for v in t[1:]])
    assert done and len(out) == len(streams)
    for st, o in zip(streams, out):
        if st["S"].shape[1] == 64:                  # the padded segments' Y
            o["Y"] = [y for i, y in enumerate(o["Y"]) if i % 16 < 8]
    tm = np.array([int(v, 16) for v in fdump.read_text().split()
                   if not v.startswith(("//", "@"))], np.uint32)
    return out, tm


@pytest.mark.parametrize("wbuf,seed", [(1, 0), (1, 1), (0, 2), (1, 3), (1, 4), (0, 5), (1, 6),
                                       (1, 7), (0, 8), (1, 9)])
def test_se_front_rtl_bit_exact(have_verilator, tmp_path, wbuf, seed):
    rng = np.random.default_rng(1700 + seed)
    special = seed != 0
    streams = [_stream(rng, special, dm, gs) for dm in range(4) for gs in range(3)]
    streams += [_stream(rng, special) for _ in range(3)]
    vops = _vops(rng, 8 * len(streams))
    # before stream i: VOPs up to 8 * i + 0..8 released, ss_req 0..30 cycles later (often 1:
    # the first VOP's start then comes with ss_req, a cycle after rdy)
    rels = [(min(len(vops), 8 * i + int(rng.integers(0, 9))),
             int(rng.choice([0, 1, 1, 2, 3, int(rng.integers(0, 31))])))
            for i in range(len(streams))]
    tm0 = F.bits(_special(rng, TW))
    outs, tm = run_front(streams, vops, rels, tm0, rng, tmp_path, wbuf)
    _check(streams, outs)
    m = Machine(Config(S=1), [vops + [I.halt()]], [None])
    m.slices[0].tput(np.arange(TW), tm0.view(np.float32))
    ref = m.run().slices[0].tmem
    bad = np.nonzero(tm[:TW] != ref[:TW])[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"


@pytest.mark.parametrize("wbuf,cns,seed", [(1, 3, 0), (0, 3, 1), (1, 5, 2), (0, 5, 3), (1, 5, 4)])
def test_se_front_fsum_rtl_bit_exact(have_verilator, tmp_path, wbuf, cns, seed):
    """FSUM: EXP2SUBs with their row sums among the VOPs (the reductions and RDOT's buffered row
    sums share u_vt and the buffer with them) and a few streams asking while their sums drain;
    the TMEM against the ISA simulator with Config.FSUM, every word (the sums: RSUM's of the
    results, bit for bit)."""
    rng = np.random.default_rng(1800 + seed)
    streams = [_stream(rng, True) for _ in range(4)]
    vops = _vops(rng, 48, fsum=True)
    fs = [v for v in vops if v.flags & I.F_FSUM and (v.w[5] >> 16) & 0xFF == I.V_EXP2SUB]
    assert len(fs) >= NFSUM
    rels = [(min(len(vops), 12 * i + int(rng.integers(0, 13))),
             int(rng.choice([0, 1, 2, int(rng.integers(0, 80))]))) for i in range(len(streams))]
    tm0 = F.bits(_special(rng, TW))
    outs, tm = run_front(streams, vops, rels, tm0, rng, tmp_path, wbuf, cns, fsum=True)
    _check(streams, outs)
    m = Machine(Config(S=1, FSUM=True), [vops + [I.halt()]], [None])
    m.slices[0].tput(np.arange(TW), tm0.view(np.float32))
    ref = m.run().slices[0].tmem
    bad = np.nonzero(tm[:TW] != ref[:TW])[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"
    for v in fs:                                # the sums are RSUM's of the results
        rows, cols, drs = v.w[3] & 0xFFFF, v.w[3] >> 16, v.w[4] & 0xFFFF
        y = ref[v.w[0] + np.arange(rows)[:, None] * drs + np.arange(cols)].view(np.float32)
        want = F.bits(F.interleaved_sum(y, F.RED_PARTIALS))
        assert np.array_equal(ref[v.w[6] + np.arange(rows)], want)


def test_se_front_fsum_on_a_vpu_without_it_stops(have_verilator, tmp_path):
    """EXP2SUBs with their row sums (and their takes) on a VPU built without FSUM (tb_se_vpu's
    FSUM 0, test_se_front_rtl_bit_exact's build) stop the simulation at otpu_vpu's check instead
    of running as EXP2SUBs without their sums."""
    rng = np.random.default_rng(1850)
    vops = _vops(rng, 16, fsum=True)
    run_front([], vops, [], F.bits(_special(rng, TW)), rng, tmp_path, 1,
              stops="VOP EXP2SUB / RSUM with FSUM on a VPU built without it (FSUM 0)")


@pytest.mark.parametrize("wbuf,seed", [(1, 0), (0, 3), (1, 4)])
def test_se_front_vsig_rtl_bit_exact(have_verilator, tmp_path, wbuf, seed):
    """VSIG at CNS 5 with FSUM (bcu1525-x2:wmx's VPU): SIGMOID and SILU (four passes of the
    composite loop, SILU's A from its ring) among the VOPs, EXP2SUBs with their row sums and
    streams asking while they run; the TMEM against the ISA simulator, every word."""
    rng = np.random.default_rng(1900 + seed)
    streams = [_stream(rng, True) for _ in range(4)]
    vops = _vops(rng, 48, fsum=True, sig=True)
    fs = [(v.w[5] >> 16) & 0xFF for v in vops if v.op == I.VOP]
    assert sum(f in (I.V_SIGMOID, I.V_SILU) for f in fs) >= NSIG
    rels = [(min(len(vops), 12 * i + int(rng.integers(0, 13))),
             int(rng.choice([0, 1, 2, int(rng.integers(0, 80))]))) for i in range(len(streams))]
    tm0 = F.bits(_special(rng, TW))
    outs, tm = run_front(streams, vops, rels, tm0, rng, tmp_path, wbuf, 5, fsum=True, sig=True)
    _check(streams, outs)
    m = Machine(Config(S=1, FSUM=True, VSIG=True), [vops + [I.halt()]], [None])
    m.slices[0].tput(np.arange(TW), tm0.view(np.float32))
    ref = m.run().slices[0].tmem
    bad = np.nonzero(tm[:TW] != ref[:TW])[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"


@pytest.mark.parametrize("wbuf,seed", [(1, 0), (0, 1), (1, 2)])
def test_se_front_vsig3_rtl_bit_exact(have_verilator, tmp_path, wbuf, seed):
    """VSIG at CNS 3 on B2's VPU (bcu1525-x2's B2: FSUM and OUTER off, MAC on): SIGMOID and SILU in
    six passes of the composite loop (EXP2's finish and 2^i into the loop register at pass 2's
    end, RECIP's setup on RR at pass 3, its finish at pass 5, SILU's A from its ring), each in
    every B mode, among the VOPs, MACs and streams asking while they run; the TMEM against the
    ISA simulator, every word."""
    rng = np.random.default_rng(1950 + seed)
    streams = [_stream(rng, True) for _ in range(4)]
    vops = _vops(rng, 48, sig=True, mac=True, outer=False, sig_modes=True)
    fb = [((v.w[5] >> 16) & 0xFF, v.w[5] >> 24) for v in vops if v.op == I.VOP]
    assert {(f, b) for f, b in fb if f in (I.V_SIGMOID, I.V_SILU)} == \
        {(f, b) for f in (I.V_SIGMOID, I.V_SILU) for b in range(4)}
    assert all(f != I.V_OUTER for f, _ in fb) and sum(f == I.V_MAC for f, _ in fb) >= NMAC
    rels = [(min(len(vops), 12 * i + int(rng.integers(0, 13))),
             int(rng.choice([0, 1, 2, int(rng.integers(0, 80))]))) for i in range(len(streams))]
    tm0 = F.bits(_special(rng, TW))
    outs, tm = run_front(streams, vops, rels, tm0, rng, tmp_path, wbuf, 3, sig=True, mac=True,
                         outer=False)
    _check(streams, outs)
    m = Machine(Config(S=1, VSIG=True, VMAC=True, OUTER=False), [vops + [I.halt()]], [None])
    m.slices[0].tput(np.arange(TW), tm0.view(np.float32))
    ref = m.run().slices[0].tmem
    bad = np.nonzero(tm[:TW] != ref[:TW])[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"


@pytest.mark.parametrize("wbuf,cns,seed", [(1, 5, 0), (0, 5, 1), (1, 3, 2), (0, 3, 3)])
def test_se_front_mac_rtl_bit_exact(have_verilator, tmp_path, wbuf, cns, seed):
    """MAC: MACs among the VOPs (B per column through the column buffer's fill, or a scalar; dst
    on port B) with FSUM's EXP2SUBs and takes, and a few streams asking around them, at CNS 3 and
    5, WBUF 0 and 1; the TMEM against the ISA simulator with Config.VMAC, every word."""
    rng = np.random.default_rng(1900 + seed)
    streams = [_stream(rng, True) for _ in range(4)]
    vops = _vops(rng, 48, fsum=True, mac=True)
    assert sum((v.w[5] >> 16) & 0xFF == I.V_MAC for v in vops) >= NMAC
    rels = [(min(len(vops), 12 * i + int(rng.integers(0, 13))),
             int(rng.choice([0, 1, 2, int(rng.integers(0, 80))]))) for i in range(len(streams))]
    tm0 = F.bits(_special(rng, TW))
    outs, tm = run_front(streams, vops, rels, tm0, rng, tmp_path, wbuf, cns, fsum=True, mac=True)
    _check(streams, outs)
    m = Machine(Config(S=1, FSUM=True, VMAC=True), [vops + [I.halt()]], [None])
    m.slices[0].tput(np.arange(TW), tm0.view(np.float32))
    ref = m.run().slices[0].tmem
    bad = np.nonzero(tm[:TW] != ref[:TW])[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"


@pytest.mark.parametrize("wbuf,seed", [(1, 0), (0, 1)])
def test_se_front_wmx_rtl_bit_exact(have_verilator, tmp_path, wbuf, seed):
    """bcu1525-x2:wmx's VPU, every composite it adds at once (CNS 5, FSUM, VSIG and MAC): SIGMOID
    and SILU, MACs (B per column through the column buffer's fill, or a scalar), EXP2SUBs with
    their row sums and their takes, streams asking around them; the TMEM against the ISA
    simulator, every word."""
    rng = np.random.default_rng(2000 + seed)
    streams = [_stream(rng, True) for _ in range(4)]
    vops = _vops(rng, 64, fsum=True, sig=True, mac=True)
    fs = [(v.w[5] >> 16) & 0xFF for v in vops if v.op == I.VOP]
    assert sum(f in (I.V_SIGMOID, I.V_SILU) for f in fs) >= NSIG and fs.count(I.V_MAC) >= NMAC
    rels = [(min(len(vops), 16 * i + int(rng.integers(0, 17))),
             int(rng.choice([0, 1, 2, int(rng.integers(0, 80))]))) for i in range(len(streams))]
    tm0 = F.bits(_special(rng, TW))
    outs, tm = run_front(streams, vops, rels, tm0, rng, tmp_path, wbuf, 5, fsum=True, sig=True,
                         mac=True)
    _check(streams, outs)
    m = Machine(Config(S=1, FSUM=True, VSIG=True, VMAC=True), [vops + [I.halt()]], [None])
    m.slices[0].tput(np.arange(TW), tm0.view(np.float32))
    ref = m.run().slices[0].tmem
    bad = np.nonzero(tm[:TW] != ref[:TW])[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"


@pytest.mark.parametrize("wbuf,seed", [(1, 0), (0, 1)])
def test_se_front_outer1_rtl_bit_exact(have_verilator, tmp_path, wbuf, seed):
    """OUTER1 on wmx's VPU (CNS 5, FSUM, VSIG, MAC; OUTER 0): OUTERs of decay one on MAC's
    datapath (C through the column buffer's fill, B(r) on port B, slot 0's a*b + c) among
    SIGMOID / SILU, MACs, EXP2SUBs with their row sums and streams asking around them; the TMEM
    against the ISA simulator with Config.OUTER1, every word."""
    rng = np.random.default_rng(2100 + seed)
    streams = [_stream(rng, True) for _ in range(4)]
    vops = _vops(rng, 64, fsum=True, sig=True, mac=True, outer1=True)
    fs = [(v.w[5] >> 16) & 0xFF for v in vops if v.op == I.VOP]
    assert fs.count(I.V_OUTER) >= NOUT1 and fs.count(I.V_MAC) >= NMAC
    assert all(v.flags & I.F_DONE for v in vops if (v.w[5] >> 16) & 0xFF == I.V_OUTER)
    rels = [(min(len(vops), 16 * i + int(rng.integers(0, 17))),
             int(rng.choice([0, 1, 2, int(rng.integers(0, 80))]))) for i in range(len(streams))]
    tm0 = F.bits(_special(rng, TW))
    outs, tm = run_front(streams, vops, rels, tm0, rng, tmp_path, wbuf, 5, fsum=True, sig=True,
                         mac=True, outer1=True)
    _check(streams, outs)
    m = Machine(Config(S=1, FSUM=True, VSIG=True, VMAC=True, OUTER=False, OUTER1=True),
                [vops + [I.halt()]], [None])
    m.slices[0].tput(np.arange(TW), tm0.view(np.float32))
    ref = m.run().slices[0].tmem
    bad = np.nonzero(tm[:TW] != ref[:TW])[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"


def test_se_front_outer1_decay_word_stops(have_verilator, tmp_path):
    """An OUTER with a decay word (scalar or per column) on a VPU with OUTER1 alone stops the
    simulation at otpu_vpu's check instead of running as decay one."""
    rng = np.random.default_rng(2150)
    vops = [I.outer(NDATA, 0, 100, 200, 2, 16, 16, 1, "one"),
            I.outer(NDATA + 64, 0, 100, 200, 2, 16, 16, 1, "scalar")]
    run_front([], vops, [], F.bits(_special(rng, TW)), rng, tmp_path, 1, 5, fsum=True,
              sig=True, mac=True, outer1=True,
              stops="VOP OUTER on a VPU built without it (OUTER 0; OUTER1: decay one only)")
