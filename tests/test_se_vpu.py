"""The stream engine's front (rtl/vpu/otpu_vpu.sv with HAS_SE): the VPU with its tail, driven
like the DMA drives it (sim/verilator/tb_se_vpu.sv), with independent VOPs on a TMEM model
around the streams. Streams: bit for bit against their op lists (tests/test_se_tail.py's
reference, every dmode x gate, A on slot 1 or 2, Q on or off, specials, random pe stalls), so
dot A on the VPU's slot-0 partial loop and u_vt is DSTEP's. VOPs (every function, the
reductions sharing slot 0's loop and u_vt with the streams): the TMEM against the ISA
simulator. Some VOPs are queued or in flight when a stream asks; the testbench fails on a TMEM
access, a VOP start or an unfinished VOP in stream mode, or a stream output outside it."""
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


def _vops(rng, n):
    """n independent VOPs: sources in [0, NDATA), each result in a fresh range (OUTER updates
    its own)."""
    out, nxt = [], NDATA

    def fresh(k):
        nonlocal nxt
        a = nxt
        nxt += k + int(rng.integers(0, 3))
        return a

    def src(k):
        return int(rng.integers(0, NDATA - k))

    for _ in range(n):
        kind = rng.choice(["elem", "elem", "red", "outer"])
        rows, cols = int(rng.integers(1, 5)), int(rng.integers(1, 160))
        if kind == "outer":
            drs = cols + int(rng.integers(0, 2))
            out.append(I.outer(fresh(rows * drs), src(cols), src(rows), src(cols), rows, cols, drs,
                               1, str(rng.choice(["scalar", "column", "one"]))))
            continue
        func = int(rng.choice(ELEM if kind == "elem" else RED))
        if func == I.V_RDOT and rng.integers(3) == 0:   # buffered row sums, many rows
            rows, cols = int(rng.integers(20, 70)), int(rng.integers(1, 80))
        ars = cols + int(rng.integers(0, 3))
        bmode = int(rng.integers(0, 4)) if func in I.READS_B else I.B_FULL
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
    assert nxt < TW
    return out


def run_front(streams, vops, rels, tm0, rng, tmp_path, wbuf, one_tree=False, comp8=False):
    """With one_tree, a 64-column stream is presented as the DMA presents it (ns = 16, pad64:
    each row's 8 segments, then 8 of +0 whose Y is dropped)."""
    lines = [f"{len(vops):x}"]
    for ins in vops:
        lines.append(" ".join(f"{v:x}" for v in [ins.op, ins.flags] + list(ins.w)))
    for st, (nrel, dly) in zip(streams, rels):
        rows, cols = st["S"].shape
        fills = _fills(st, rng)
        pad = one_tree and cols == 64
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
                                     R / "vpu/otpu_se_comp.sv", R / "vpu/otpu_se_tail.sv",
                                     R / "vpu/otpu_vpu.sv", rtlsim.TB / "tb_se_vpu.sv"],
                        {"WBUF": int(wbuf), "ONE_TREE": int(one_tree), "COMP8": int(comp8)})
    r = rtlsim.run_sim([exe, f"+in={fin}", f"+out={fout}", f"+tmem={ftm}", f"+dump={fdump}"],
                       timeout=900)
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
        if one_tree and st["S"].shape[1] == 64:     # the padded segments' Y
            o["Y"] = [y for i, y in enumerate(o["Y"]) if i % 16 < 8]
    tm = np.array([int(v, 16) for v in fdump.read_text().split()
                   if not v.startswith(("//", "@"))], np.uint32)
    return out, tm


@pytest.mark.parametrize("wbuf,seed,one_tree,comp8", [
    (1, 0, 0, 0), (1, 1, 0, 0), (0, 2, 0, 0),            # v1
    (1, 3, 1, 0), (1, 4, 1, 0), (0, 5, 1, 0),            # ONE_TREE
    (1, 6, 1, 1), (1, 7, 1, 1), (0, 8, 1, 1), (1, 9, 0, 1)])   # v2 (and COMP8 alone)
def test_se_front_rtl_bit_exact(have_verilator, tmp_path, wbuf, seed, one_tree, comp8):
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
    outs, tm = run_front(streams, vops, rels, tm0, rng, tmp_path, wbuf, one_tree, comp8)
    _check(streams, outs)
    m = Machine(Config(S=1), [vops + [I.halt()]], [None])
    m.slices[0].tput(np.arange(TW), tm0.view(np.float32))
    ref = m.run().slices[0].tmem
    bad = np.nonzero(tm[:TW] != ref[:TW])[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"
