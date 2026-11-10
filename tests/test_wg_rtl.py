"""Weight groups on the RTL (docs/weight_groups.md): otpu_mxu at WG 8 (the WG targets' MCOLS 8 x D 128,
port W's 1 KiB chunks) runs every MM in groups (GALL, section 12: int8 groups of 8 rows, 4-bit
of 16, a chunk one K-block; no rows of one, no PAIR) to the ISA simulator's TMEM bit for bit:
every weight format, scaled / UNIT / RSC, ACC, k0 (odd too), N not a multiple of a group, odd
KB, KB 1, M past the columns (replay, up to ACT_ROWS), RMAX and ASCALE (a result row one
group's rows against one activation row), on the behavioural DRAM and on the native adapter
with stalls (port W below A and SW); the images' group chunks with their lanes in the channel
hash's order (CHASH, the WG targets': port W's lane c is column c's) and, without it, in order. MM's K
tail (KT, docs/isa.md "K tail"): a pair of groups sharing its tail chunk, as the ISA's."""
import re
import subprocess

import numpy as np
import pytest

from opentpu import Config, isa as I, quant as Q, rtlsim
from opentpu.isasim import Machine

from test_kt import _act as _kt_act, _operand as _kt_operand
from test_wg import FMT, _act, _matrix, _place, tiny3  # noqa: F401 (tiny3: the fixture)

D, WG = 128, 8
UARCH = {"NCH": 8, "CB": 128}       # a slice's 8 channels of 128-byte beats: P = 1 KiB


def _cfg(**kw):
    return Config(**{**dict(S=1, D=D, MCOLS=8, WG=WG, ACT_BLOCKS=16, ACT_ROWS=16, GALL=True,
                            DRAM_BYTES=1 << 23, CHASH=True), **kw})


def _up(a, m):
    return -(-a // m) * m


def _extreme_act(rng, M, KB, D):
    """Activations of one magnitude (QACT quantizes them to +-127) in sign patterns by row: all
    +, all -, alternating, alternating by 16 elements (a chain's group)."""
    i = np.arange(KB * D)
    pats = [np.ones_like(i), -np.ones_like(i), 1 - 2 * (i % 2), 1 - 2 * ((i // 16) % 2)]
    return np.stack([pats[m % 4] for m in range(M)]).astype(np.float32)


def _extreme_matrix(rng, wf, N, KB, KBr, rsc, D):
    """4-bit rows of the largest codes (int4 7 / -8; E2M1 +-6, the PEs' +-12), int8 rows of 127
    and -113 (the packed PE's nibbles h 7 / -8 and l 15 both), in sign patterns: the packed pairs
    (2j, 2j+1) all +/- in every combination, then alternating ones; scales as _matrix's."""
    _, rows = _matrix(rng, wf, N, KB, KBr, rsc, D)
    pos, neg = (127, 0x8F) if wf == I.W8 else (7, 8) if wf == I.W4I else (7, 15)
    i = np.arange(KB * D)
    pats = [np.full_like(i, pos), np.full_like(i, neg), np.where(i % 2, neg, pos),
            np.where((i // 16) % 2, neg, pos)]
    order = [0, 0, 0, 1, 1, 0, 1, 1, 2, 2, 2, 3, 3, 2, 3, 3]
    data = np.zeros((N, Q.row_bytes(KB * D, FMT[wf], D)), np.uint8)
    codes = np.stack([pats[order[r % 16]] for r in range(N)])
    if wf == I.W8:
        data[:, :KB * D] = codes
    else:
        data[:, :KB * D // 2] = Q.pack4(codes)
    return data, rows


def _program(rng, cases, act=_act, matrix=_matrix, cfg=None):
    """One program over `cases` (wf, N, KB, M, k0, unit, rsc, post, acc): per case its
    activations (act) loaded and quantized, its grouped MM (quant.grp_r rows a group; ACC: twice,
    the second accumulating) over a matrix (matrix), the outputs in TMEM rows of N + 3 words;
    post "rmax": the MMs with RMAX (the row maxima after the rows), "ascale": the second MM with
    ASCALE (the old outputs times the rows' factors, M words loaded before the outputs; needs
    UNIT and ACC), "both": the two. Returns (program, DRAM image)."""
    cfg = cfg or _cfg()
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    prog = []
    xa, wa, sa, out = 0, 1 << 20, 6 << 20, 1 << 14
    for wf, N, KB, M, k0, unit, rsc, post, acc in cases:
        fmt = FMT[wf]
        x = act(rng, M, KB, D)
        data, rows = matrix(rng, wf, N, KB, k0 + KB, rsc, D)
        data = Q.group_rows(data, WG, D, fmt, KB * D)
        rows = Q.group_scales(rows, fmt, (k0 + KB) * D, WG, D, rsc)
        _place(dram, xa, x)
        _place(dram, wa, Q.hash_rows(data, wa, WG, D) if cfg.CHASH else data)
        _place(dram, sa, rows)
        ascale = None
        if post in ("ascale", "both"):          # (the factors, signs and magnitudes mixed)
            assert unit and acc, "ASCALE needs UNIT and ACC"
            al = (rng.uniform(0.25, 2.0, M) * rng.choice([-1, 1], M)).astype(np.float32)
            _place(dram, xa + _up(x.nbytes, 1024), al)
            prog.append(I.ld(xa + _up(x.nbytes, 1024), out, M))
            ascale, out = out, out + M
            xa += 1024
        ors = N + 3
        mm = dict(sa=wa, ssa=sa, out=out, n=N, kb=KB, rs=data.shape[1], ors=ors, m=M, ab=0,
                  srs=rows.shape[1], unit=unit, wf=wf, k0=k0, rsc=rsc, grp=Q.grp_r(fmt, WG),
                  rmax=post in ("rmax", "both"))
        prog += [I.ld(xa, 0, x.size), I.qact(0, M, 0, KB, KB * D), I.mm(**mm)]
        if acc:
            prog.append(I.mm(**mm, acc=True, ascale=ascale))
        xa += _up(x.nbytes, 1024)
        wa += _up(data.size, 1024)
        sa += _up(rows.size, 1024)
        out += (M + 1) * ors
        assert xa < 1 << 20 and wa < 6 << 20 and sa < cfg.DRAM_BYTES and out < cfg.TMEM_WORDS
    return prog + [I.halt()], dram


def _check(cases, seed, gen={}, chash=True, cfg=None, program=_program, **kw):
    cfg = cfg or _cfg(CHASH=chash)
    prog, dram = program(np.random.default_rng(seed), cases, cfg=cfg, **gen)
    m = Machine(cfg, [prog], [dram.copy()]).run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [dram.copy()], uarch=UARCH | kw.pop("uarch", {}),
                                  **kw)
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"
    outs = m.slices[0].tmem[1 << 14:]
    assert np.count_nonzero(outs) > len(cases) * 4          # (the MMs wrote their outputs)
    if kw.get("axi"):                                       # (on port W)
        assert sum(n["w_rd"] for n in st["native"]) > 0
    return st


def _cases(wf):
    """A format's cases: plain, ACC, k0, N 1 / 8 / 12 / 13 / 20, KB 1 / 2 / 3 / 5, M 1 / 3 / 9
    (replay), UNIT, RSC (not NVFP4); 4-bit (groups of 16) also odd k0 with N 33 (a last group of
    one row), M 16 (ACT_ROWS) with N 24 (a last group of 8: its second result row empty), KB 1
    over whole groups (a row's last advance a cycle apart from the next's)."""
    w4 = wf != I.W8
    c = [(wf, 8, 2, 1, 0, False, False, "", False),
         (wf, 12, 3, 1, 2, False, False, "", True),
         (wf, 20, 5, 3, 0, False, False, "", False),
         (wf, 1, 1, 9, 0, False, False, "", False),
         (wf, 20, 2, 9, 2, False, False, "", True),
         (wf, 13, 3, 2, 0, True, False, "", False)]
    if wf != I.WNV:
        c.append((wf, 12, 3, 3, 0, False, True, "", False))
    if w4:
        c += [(wf, 33, 3, 2, 1, False, False, "", False),
              (wf, 24, 5, 16, 0, False, False, "", True),
              (wf, 32, 1, 3, 1, False, False, "", False)]
    return c


def _post_cases(wf):
    """A format's RMAX and ASCALE cases (M <= MCOLS; a drain step one group's rows of one
    activation row, the row's max over its lanes and its factor the row's): M 1 / 3 / 5 / 8, N
    within a group, past it, a last group of one row (4-bit: 33; int8: 9), ACC, k0, KB 1, the
    two together; scaled and UNIT."""
    n1 = 33 if wf != I.W8 else 9
    return [(wf, 20, 3, 3, 0, False, False, "rmax", False),
            (wf, n1, 2, 8, 1, False, False, "rmax", True),
            (wf, 5, 1, 1, 0, True, False, "rmax", False),
            (wf, 12, 3, 2, 0, True, False, "ascale", True),
            (wf, n1, 2, 8, 2, True, False, "ascale", True),
            (wf, 24, 5, 5, 0, True, False, "both", True)]


@pytest.mark.parametrize("wf", [I.W8, I.W4I, I.W4F, I.WNV], ids=["int8", "int4", "fp4", "nvfp4"])
def test_grouped_mm_rtl(have_verilator, wf):
    _check(_cases(wf), seed=wf)


@pytest.mark.parametrize("wf", [I.W8, I.W4F, I.WNV], ids=["int8", "fp4", "nvfp4"])
def test_grouped_rmax_ascale_rtl(have_verilator, wf):
    """RMAX and ASCALE on grouped MMs (GALL: attention's scores and its output's rescale): the
    row maxima and the factors by activation row, as the ISA's."""
    _check(_post_cases(wf), seed=40 + wf)


@pytest.mark.parametrize("wf", [I.W8, I.W4I, I.W4F, I.WNV], ids=["int8", "int4", "fp4", "nvfp4"])
def test_packed_extremes_rtl(have_verilator, wf):
    """The packed PEs (docs/weight_groups.md: GRP at a 4-bit format, a lane's rows 2j and
    2j + 1; int8 in either layout, w = 16 h + l) at their bound: whole chains of the largest
    activations (+-127) and codes (int4 -8 / 7, E2M1 +-12; int8 127 / -113, h 7 / -8 and l 15),
    so a chain's two sums reach +-16 x 127 x 12 = +-24,384 (int4 and int8's h 16,256, int8's l
    30,480) in every sign combination, the low one negative under a positive high one included:
    the split stays exact."""
    cases = [(wf, 16, 5, 1, 0, False, False, "", False),
             (wf, 16, 4, 2, 2, False, False, "", True),
             (wf, 24, 3, 4, 0, False, False, "", False),
             (wf, 32, 3, 9, 1, False, False, "", False),
             (wf, 16, 2, 3, 0, True, False, "", False),
             (wf, 24, 3, 8, 0, False, False, "rmax", False)]     # (ties: equal rows' maxima)
    _check(cases, seed=20 + wf, gen=dict(act=_extreme_act, matrix=_extreme_matrix))


def test_rtl_at_groups_is_gall():
    """The RTL at WG > 1 runs every MM in groups: a configuration without GALL is refused before
    a build (the ISA simulator's natural MMs at WG > 1 have no hardware)."""
    with pytest.raises(ValueError, match="without GALL"):
        rtlsim.top_params(_cfg(GALL=False))
    assert rtlsim.top_params(_cfg())["WG"] == WG and "WG" not in rtlsim.top_params(
        _cfg(WG=1, GALL=False, CHASH=False))


@pytest.mark.parametrize("stall,chash", [(0, True), (30, True), (30, False)])
def test_grouped_mm_on_the_native_adapter_rtl(have_verilator, stall, chash):
    """The board's memory path (otpu_native_dram's port W, 8 x 128-byte channels, random
    stalls), its channels hashed (the images' rows in the hash's order) or not."""
    cases = _cases(I.W8)[:3] + _cases(I.W4F)[:3] + _cases(I.WNV)[:2]
    _check(cases, seed=11 + stall, chash=chash, axi=True, stall=stall)


# the WG targets' chunk FIFO: 2,048 entries, the lanes in UltraRAM (MXU_FIFO_STYLE)
UFIFO = {"FIFO_DEPTH": 2048, "MXU_FIFO_STYLE": '"ultra"'}

# the pumped array (otpu_mxu PUMP, [uarch] MXU_PUMP): each column's DSPs on the 2x clock (clk_f),
# a pass two activation rows (2p and 2p + 1; an odd M's last pass one), its result FIFO rows
# pushed in the unpumped passes' order
PUMP = {"MXU_PUMP": 1}


@pytest.mark.parametrize("stall,xqb,fifo", [(0, 0, {}), (30, 0, {}), (30, 2, {}), (0, 0, UFIFO),
                                            (30, 2, UFIFO)], ids=["0-0", "30-0", "30-2",
                                                                  "0-0-u", "30-2-u"])
def test_posted_w_beats_rtl(have_verilator, stall, xqb, fifo):
    """Port W's beats posted by the channels into the MXU's chunk FIFO lanes ((a) v2, NAT_XPOST:
    otpu_native_dram and otpu_mxu XPOST, the native model posting whole beats at the core clock;
    docs/native_channels.md), with random stalls: grouped MMs of every format, ACC, k0, replay,
    RMAX, ASCALE, to the ISA simulator's TMEM bit for bit. xqb 2: each beat in four quarter
    writes, as the WG targets' (otpu_top NAT_XQB: otpu_mxu QB 2, option A's quarter RAMs and their
    remainder's field writes, p_q one-hot). fifo UFIFO (-u): the WG targets' 2,048 entries, the lanes
    in UltraRAM, each beat staged in LUT RAM and copied in on the core clock (otpu_mxu g_xp g_u)."""
    ua = {"NAT_XPOST": 1} | ({"NAT_XQB": xqb} if xqb else {}) | fifo
    top = rtlsim.top_params(_cfg(), 8, UARCH | ua, True)
    assert top["NAT_XPOST"] == 1 and top.get("NAT_XQB", 0) == xqb   # (kept: AXI 1, WG 8)
    cases = _cases(I.W8)[:3] + _cases(I.W4F)[:3] + _cases(I.WNV)[:2]
    cases[2:2] = _post_cases(I.W8)[3:4] + _post_cases(I.W4F)[1:2]
    _check(cases, seed=31 + stall, axi=True, stall=stall, uarch=ua)


def test_posted_ultra_counts_landed_beats_rtl(have_verilator, tmp_path, monkeypatch):
    """The UltraRAM lanes' entries complete on the beats landed in the lanes (lu, counted at the
    UltraRAM's write, v1), not on those the copy took from the staging RAM a cycle before (tk):
    otpu_mxu with the count moved to the take stops on its check (an entry complete before a
    lane's beat landed), at the WG targets' chunk FIFO and quarter posts."""
    src = (rtlsim.RTL / "mxu/otpu_mxu.sv").read_text()
    a = "if (v1[c]) lu[c] <= lu[c] + 1'b1;"
    assert src.count(a) == 1
    mut = tmp_path / "otpu_mxu.sv"
    mut.write_text(src.replace(a, "if (tk[c]) lu[c] <= lu[c] + 1'b1;"))
    monkeypatch.setattr(rtlsim, "RTL_SOURCES", [str(mut) if s == "mxu/otpu_mxu.sv" else s
                                                for s in rtlsim.RTL_SOURCES])
    with pytest.raises(RuntimeError, match="complete before lane"):
        _check(_cases(I.W8)[:2], seed=41, axi=True, stall=0,
               uarch={"NAT_XPOST": 1, "NAT_XQB": 2} | UFIFO)


# a width warning on an instance's port connection (Verilator: "Input port connection 'p_slot'
# expects 800 bits on the pin connection, but ... generates 160 bits")
PORT_WIDTH = re.compile(r"%Warning-WIDTH\w*: (\S+) .*port connection '(\w+)' expects (\d+) bits"
                        r".*generates (\d+) bits")


@pytest.mark.parametrize("top", ["tb_board", "tb_top", "tb_bcuwg"])
def test_posted_port_widths_lint(have_verilator, top):
    """The posted configuration's testbenches linted with WIDTHEXPAND / WIDTHTRUNC on: rtlsim
    builds with both off (VFLAGS), so a port connected at the wrong shape builds and runs zero
    extended or cut (766101d5: tb_board's one-copy pslot, 160 bits, into otpu_board's p_slot
    copies, 800; otpu_mxu's copies check stopped the board model). tb_board and tb_bcuwg: the
    board model's builds (SimTransport.sim_build) of bcu1525-s2 at NAT_XPOST, tb_board the board
    alone and tb_bcuwg the BCU1525's system around it (otpu_bcu_wg_sys, its bridges posting quarters);
    tb_top: S 2 with NAT_XQB 2 (the quarters' one-hot p_q); all with the WG targets' chunk FIFO (2,048
    deep, its lanes in UltraRAM). Any width warning naming a port connection fails."""
    fifo = {"FIFO_DEPTH": 2048, "MXU_FIFO_STYLE": '"ultra"'}
    if top == "tb_top":                     # rtlsim.build_top's sources at AXI 1 (no LDC)
        srcs = [rtlsim.RTL / s for s in rtlsim.RTL_SOURCES]
        srcs += [rtlsim.TB / "otpu_native_mem.sv", rtlsim.TB / "tb_top.sv"]
        p = rtlsim.top_params(_cfg(S=2), 8, UARCH | fifo | {"NAT_XPOST": 1, "NAT_XQB": 2}, True)
        assert p["NAT_XPOST"] == 1 and p["NAT_XQB"] == 2
    else:
        from opentpu.host.board import SimTransport
        wrap = top == "tb_bcuwg"
        bp = {"MCOLS": 8, "WG": WG, "NAT_XPOST": 1}
        st = SimTransport(ch_bytes=1 << 12, target="bcu1525-s2", bcuwg=wrap, params=bp)
        t, srcs, p = st.sim_build()
        assert t == top and (wrap or (p["NAT_XPOST"], p["FIFO_DEPTH"]) == (1, 2048))
    cmd = ["verilator", "--lint-only", "-Wno-fatal", "-Wwarn-WIDTHEXPAND", "-Wwarn-WIDTHTRUNC",
           "--top-module", top, *[f"-G{k}={v}" for k, v in p.items()], *map(str, srcs)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    bad = [m.group(0) for m in PORT_WIDTH.finditer(r.stderr)]
    assert not bad, "\n".join(bad)


# QST GRP's cases (col, KB, ROW, n0, rows, HALF): rows mode (a K cache's appends: n0 the first
# row, w6 + R[rd]) one row and several, across groups, KB past a group word of d bytes, HALF;
# COL (V^T: a token's d = KB x D elements as column n) one column and 30, across a column chunk
QST_GRP = [(False, 1, True, 0, 1, False), (False, 2, False, 13, 1, False),
           (False, 3, False, 5, 11, False), (False, 1, True, 30, 9, False),
           (False, 2, True, 6, 3, True), (False, 5, False, 3, 2, False),
           (True, 1, True, 0, 1, False), (True, 2, True, 37, 1, True),
           (True, 2, True, 21, 30, False), (True, 1, True, 126, 3, False)]


def _qst_program(rng, cases, cfg):
    """One program of QST GRPs (QST_GRP's cases) over a DRAM field of 0xA5 (the bytes they do
    not write), each case's rows at its own TMEM rows and matrix: rows mode a matrix of n0 +
    rows rows in groups (its scale areas Q.grp_scale_bytes apart), COL one of d rows by 256
    columns (the group stride es 256 / D chunks); the first row R[1] + w6. A case's seventh
    field, if any: QST flag LO (SPLITQ), rows mode."""
    P, cap = WG * D, 256
    dram = np.full(cfg.DRAM_BYTES, 0xA5, np.uint8)
    prog, xa, ta, wa, sa = [], 0, 0, 1 << 20, 6 << 20
    for col, KB, row, n0, rows, half, *lo in cases:
        x = rng.standard_normal((rows, KB * D)).astype(np.float32) * rng.uniform(0.01, 4.0)
        _place(dram, xa, x)
        prog += [I.ld(xa, ta, x.size), I.li(1, n0 - 3)]
        if col:
            es = cap // D * P
            prog.append(I.qst(ta, wa, sa + 4 * n0, rows, KB, KB * D, 3, es, row=True,
                              half=half, grp=WG, col=True, rd=1))
            wn, sn = KB * D // WG * es, 4 * cap
        else:
            es = Q.grp_scale_bytes(KB * D, "int8", D, WG, rsc=row)
            prog.append(I.qst(ta, wa, sa, rows, KB, KB * D, 3, es, row=row, half=half,
                              grp=WG, rd=1, lo=bool(lo and lo[0])))
            ng = -(-(n0 + rows) // WG)
            wn, sn = ng * KB * P, ng * es
        xa += _up(x.nbytes, 1024)
        ta += x.size
        wa += _up(wn, P)
        sa += _up(sn, 1024)
        assert xa < 1 << 20 and wa < 6 << 20 and sa < cfg.DRAM_BYTES and ta < cfg.TMEM_WORDS
    return prog + [I.halt()], dram


@pytest.mark.parametrize("axi,chash,sww", [(False, True, 1), (False, False, 1), (True, True, 1),
                                           (False, True, 4), (True, True, 4)])
def test_qst_grp_rtl(have_verilator, axi, chash, sww):
    """QST GRP (otpu_quant at WG 8, otpu_seq's R[rd] + w6): every byte of DRAM as the ISA
    simulator's, rows mode and COL, the channel hash's lane order or none, on the behavioural
    DRAM and on the native adapter (its byte-masked port A writes, stalls); sww 4: QUANT_SWW's
    16 bytes a cycle in rows mode (QUANT_PW 2: the writer's 16 lanes)."""
    cfg = _cfg(CHASH=chash)
    prog, dram = _qst_program(np.random.default_rng(5), QST_GRP, cfg)
    m = Machine(cfg, [prog], [dram.copy()]).run()
    ua = UARCH | ({"QUANT_SWW": sww, "QUANT_PW": 2} if sww > 1 else {})
    drams, _, _ = rtlsim.run(cfg, [prog], [dram.copy()], uarch=ua,
                             **(dict(axi=True, stall=30) if axi else {}))
    bad = np.nonzero(drams[0] != m.slices[0].dram)[0]
    assert len(bad) == 0, f"{len(bad)} DRAM bytes differ, first at {bad[:8]}"
    assert np.count_nonzero(m.slices[0].dram[1 << 20:] != 0xA5) > 1000   # (they wrote)


# QUANT_QSTB's GRP cases (col, KB, ROW, n0, rows, HALF, LO): rows mode in ROW mode (the chunked
# DeltaNet's operands, hi then lo) one row and many, from a group's start and inside one, across
# groups and their scale areas, KB 1 to 3; among them ones it leaves unbuffered (not ROW, HALF,
# COL)
QSTB_GRP = [(False, 1, True, 0, 1, False, False), (False, 1, True, 0, 20, False, False),
            (False, 1, True, 0, 20, False, True), (False, 2, True, 13, 11, False, False),
            (False, 2, False, 13, 4, False, False), (False, 3, True, 5, 9, False, True),
            (False, 2, True, 6, 3, True, False), (True, 2, True, 21, 30, False, False),
            (False, 1, True, 30, 9, False, False)]


@pytest.mark.parametrize("axi,chash", [(False, True), (True, True), (False, False)])
def test_qstb_grp_rtl(have_verilator, axi, chash):
    """QUANT_QSTB on QST GRP (QUANT_SWW 4, QUANT_PW 2): a ROW rows-mode QST's rows through the row
    buffer, each row's S word into its group's area and then its bytes at its lane of each D
    block's group chunk (the channel hash's lane order or none), the pops' addresses stepping as
    the reader's would; QSTB_GRP's cases, every byte of DRAM the ISA simulator's, on the
    behavioural DRAM and the native adapter (stalls)."""
    cfg = _cfg(CHASH=chash, SPLITQ=True)
    prog, dram = _qst_program(np.random.default_rng(11), QSTB_GRP, cfg)
    m = Machine(cfg, [prog], [dram.copy()]).run()
    ua = UARCH | {"QUANT_SWW": 4, "QUANT_PW": 2, "QUANT_QSTB": 1}
    drams, _, _ = rtlsim.run(cfg, [prog], [dram.copy()], uarch=ua,
                             **(dict(axi=True, stall=30) if axi else {}))
    bad = np.nonzero(drams[0] != m.slices[0].dram)[0]
    assert len(bad) == 0, f"{len(bad)} DRAM bytes differ, first at {bad[:8]}"
    assert np.count_nonzero(m.slices[0].dram[1 << 20:] != 0xA5) > 1000   # (they wrote)


# QUANT_QSTHL's cases (KB, n0, rows, extra): GRP ROW rows hi and lo (the chunked DeltaNet's
# operands) one row and many, from a group's start and inside one, across groups and their scale
# areas, KB 1 to 3; lo's part `extra` KiB past the end of hi's (its chunks' folds other lanes
# than hi's under the channel hash); extra None: a QST GRP of hi alone between them
QSTHL_GRP = [(1, 0, 1, 0), (1, 0, 20, 1), (2, 13, 11, 3), (2, 6, 3, None), (3, 5, 9, 0),
             (1, 30, 9, 2)]


def _qsthl_program(rng, cases, cfg, hl=True):
    """QSTHL_GRP's program (as _qst_program's rows mode, ROW): each case's hi part and its lo part
    hld bytes after it (data and scales), from one QST HL (hl) or a QST and a QST LO."""
    P = WG * D
    dram = np.full(cfg.DRAM_BYTES, 0xA5, np.uint8)
    prog, xa, ta, wa, sa = [], 0, 0, 1 << 20, 6 << 20
    for KB, n0, rows, extra in cases:
        x = rng.standard_normal((rows, KB * D)).astype(np.float32) * rng.uniform(0.01, 4.0)
        _place(dram, xa, x)
        prog += [I.ld(xa, ta, x.size), I.li(1, n0 - 3)]
        es = Q.grp_scale_bytes(KB * D, "int8", D, WG, rsc=True)
        ng = -(-(n0 + rows) // WG)
        wn, sn = ng * KB * P, ng * es
        hld = 0 if extra is None else _up(max(wn, sn), max(P, 1024)) + extra * 1024
        a = (ta, wa, sa, rows, KB, KB * D, 3, es)
        if hld and hl:
            prog.append(I.qst(*a, row=True, grp=WG, rd=1, hld=hld))
        else:
            prog.append(I.qst(*a, row=True, grp=WG, rd=1))
            if hld:
                prog.append(I.qst(ta, wa + hld, sa + hld, *a[3:], row=True, grp=WG, rd=1,
                                  lo=True))
        xa += _up(x.nbytes, 1024)
        ta += x.size
        wa += _up(hld + wn, P)
        sa += _up(hld + sn, 1024)
        assert xa < 1 << 20 and wa < 6 << 20 and sa < cfg.DRAM_BYTES and ta < cfg.TMEM_WORDS
    return prog + [I.halt()], dram


@pytest.mark.parametrize("axi,chash,seed", [(False, True, 0), (True, True, 0),
                                            (False, False, 0), (False, True, 3)])
def test_qsthl_grp_rtl(have_verilator, monkeypatch, axi, chash, seed):
    """QUANT_QSTHL (otpu_quant QSTHL, with QUANT_QSTB, SWW 4, PW 2): QST HL's GRP ROW rows, each
    row's hi and lo scale words, then each chunk popped twice (hi's bytes, then lo's HLD after
    them in their own chunk's lanes), QSTHL_GRP's cases among a QST GRP of hi alone: every byte of
    DRAM the ISA simulator's, and the ISA's QST HL a QST and a QST LO; on the behavioural DRAM and
    the native adapter (stalls), the channel hash or none; seed: from a random power-up
    (OTPU_X_INITIAL=unique: sbh, the lo pops' chunk glb, HLD hold nothing past the reset)."""
    if seed:
        monkeypatch.setenv("OTPU_X_INITIAL", "unique")
        monkeypatch.setenv("OTPU_X_SEED", str(seed))
    cfg = _cfg(CHASH=chash, SPLITQ=True, QSTHL=True, SEQ_WC=True)
    prog, dram = _qsthl_program(np.random.default_rng(13), QSTHL_GRP, cfg)
    prog0, _ = _qsthl_program(np.random.default_rng(13), QSTHL_GRP, cfg, hl=False)
    assert sum(i.op == I.QST and bool(i.w[4] & I.QST_HL) for i in prog) == 5
    m = Machine(cfg, [prog], [dram.copy()]).run()
    m0 = Machine(cfg, [prog0], [dram.copy()]).run()
    assert np.array_equal(m.slices[0].dram, m0.slices[0].dram)
    ua = UARCH | {"QUANT_SWW": 4, "QUANT_PW": 2, "QUANT_QSTB": 1}
    drams, _, _ = rtlsim.run(cfg, [prog], [dram.copy()], uarch=ua,
                             **(dict(axi=True, stall=30) if axi else {}))
    bad = np.nonzero(drams[0] != m.slices[0].dram)[0]
    assert len(bad) == 0, f"{len(bad)} DRAM bytes differ, first at {bad[:8]}"
    assert np.count_nonzero(m.slices[0].dram[1 << 20:] != 0xA5) > 2000   # (they wrote)


def test_qsthl_cycles_rtl(have_verilator):
    """QST HL at bcu1525-x2:wmx's quantizer and TMEM (PW 2, SWW 4, R1, QSTB): 64 GRP ROW rows of
    128, hi and lo, in about 18 cycles a row (two scale words, each chunk popped twice) against
    ~35 for a QST and a QST LO (two reads of the rows)."""
    from opentpu import target as T
    ua = {k: v for k, v in T.sim_uarch("bcu1525-x2:wmx").items()
          if k.startswith("TMEM_") or k in ("RPB", "WPB", "QUANT_R1", "QUANT_PW", "QUANT_SWW")}
    ua |= UARCH | {"QUANT_QSTB": 1}
    cfg = _cfg(SPLITQ=True, QSTHL=True)
    x = np.random.default_rng(9).standard_normal((64, D)).astype(np.float32)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _place(dram, 0, x)
    es = Q.grp_scale_bytes(D, "int8", D, WG, rsc=True)
    hld = 64 << 10
    a = (0, 1 << 20, 6 << 20, 64, 1, D, 0, es)
    cyc = {}
    for hl in (True, False):
        qs = ([I.qst(*a, row=True, grp=WG, hld=hld)] if hl else
              [I.qst(*a, row=True, grp=WG),
               I.qst(0, (1 << 20) + hld, (6 << 20) + hld, *a[3:], row=True, grp=WG, lo=True)])
        prog = [I.ld(0, 0, x.size)] + qs + [I.halt()]
        _, _, st = rtlsim.run(cfg, [prog], [dram.copy()], trace=True, uarch=ua)
        S = {s: int(c) for c, s in re.findall(r"T0 S c=(\d+) s=(\d+)", st["trace"])}
        E = {s: int(c) for c, s in re.findall(r"T0 E c=(\d+) s=(\d+)", st["trace"])}
        cyc[hl] = E[str(len(qs))] - S["1"]
    print(f"QST GRP 64 x 128 ROW hi + lo: {cyc[True]} cycles with QST HL "
          f"({cyc[True] / 64:.1f} a row), {cyc[False]} as QST + QST LO "
          f"({cyc[False] / 64:.1f} a row)")
    assert cyc[True] < 22 * 64 and cyc[True] < 0.65 * cyc[False], cyc


@pytest.mark.parametrize("model,wf,S,ua", [
    ("qwen3", "int8", 1, {}), ("qwen35", "nvfp4", 2, {}), ("qwen3", "fp4", 2, {}),
    ("smollm3", "int8", 2, {}), ("phi3", "fp4", 1, {}),
    ("qwen3", "fp4", 2, {"NAT_XPOST": 1, "NAT_XQB": 2}),
    ("qwen3", "fp4", 2, {"NAT_XPOST": 1, "NAT_XQB": 2, **UFIFO}),
    pytest.param("qwen35", "nvfp4", 2, {"NAT_XPOST": 1, "NAT_XQB": 2, **UFIFO, "MXU_QD": 4},
                 id="qwen35-nvfp4-2-xpost-ufifo-qd4"),
    pytest.param("qwen3", "int8", 1, PUMP, id="qwen3-int8-1-pump"),
    pytest.param("qwen35", "nvfp4", 2, {"NAT_XPOST": 1, "NAT_XQB": 2, **UFIFO, **PUMP},
                 id="qwen35-nvfp4-2-xpost-ufifo-pump")])
def test_tiny_models_gall_on_tb_top_rtl(have_verilator, tiny3, tmp_path, model, wf, S, ua):
    """GALL's programs on tb_top (otpu_top at WG 8: the WG targets' MCOLS 8 x D 128 on 8 channels of
    128-byte beats): every MM in groups, the KV cache's appends QST GRP (K's rows, V^T's
    columns), attention's MMs with RMAX and ASCALE. A tiny Qwen3 (test_wg's), Qwen3.5, SmolLM3
    or Phi-3 (test_llama's: layers without RoPE; RoPE on 96 of 128, LongRoPE) fed 5 tokens on
    the ISA simulator, then a 4-row prefill (rows 5 .. 8: across a group of the cache) and a
    decode step on the RTL from the same DRAM: the logits and every slice's DRAM the ISA
    simulator's bit for bit. ua: the WG targets' memory path, the native channels posting port W's beats
    in quarters (NAT_XPOST, NAT_XQB 2), into option A's block RAM lanes or the WG targets' UltraRAM ones
    (UFIFO); qd4 with the MXU's command queue 4 deep (MXU_QD); pump with the pumped array
    (MXU_PUMP)."""
    from opentpu.isasim import design_config
    from opentpu.llm.qwen3 import Engine
    from opentpu.llm.rtl_backend import RtlBackend
    if model == "qwen3":
        W, spec = tiny3
    elif model in ("smollm3", "phi3"):
        pytest.importorskip("transformers")
        from test_llama import _tiny
        _, _, spec, W = _tiny(model, tmp_path)
    else:
        pytest.importorskip("transformers")
        from test_qwen35 import _tiny_model
        _, W, spec = _tiny_model(4, 8)
    cfg = design_config(S=S, DRAM_BYTES=1 << 25, MCOLS=8, WG=WG, CHASH=True, GALL=True,
                        PAIR=False)
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 10)]
    eng = Engine(spec, W, cap=256, cfg=cfg, wformat=wf, rows=4)
    for t in toks[:5]:
        eng.step(t)
    isa, n = eng.backend, eng.image.nbytes
    rtl = RtlBackend(cfg, [s.dram[:n] for s in isa.machine.slices], uarch=UARCH | ua,
                     **({"axi": True} if ua else {}))
    want = [eng.prefill(toks[5:9], chunk=4), eng.step(toks[9])]
    eng.backend, eng.pos = rtl, 5
    got = [eng.prefill(toks[5:9], chunk=4), eng.step(toks[9])]
    for a, b in zip(want, got):
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    for s in range(S):
        assert np.array_equal(isa.machine.slices[s].dram[:n], rtl.drams[s][:n]), s


def _kt_program(rng, cases, cfg):
    """One program over MM KT's `cases` (wf, N, KB, M, unit, rsc, acc): per case activations and
    a matrix whose K ends in the middle of block KB-1 (test_kt's: its upper half zero), the
    groups in KT's pairs (quant.kt_rows / kt_scales), loaded and quantized, the MM with rs bit 0
    (ACC: twice, the second accumulating), the outputs in TMEM rows of N + 3 words."""
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    prog, xa, wa, sa, out = [], 0, 1 << 20, 6 << 20, 1 << 14
    for wf, N, KB, M, unit, rsc, acc in cases:
        fmt = FMT[wf]
        x = _kt_act(rng, M, KB, D)
        data, rows = _kt_operand(rng, wf, N, KB, D)
        d = Q.kt_rows(Q.group_rows(data, WG, D, fmt, KB * D), WG, D)
        s = Q.kt_scales(Q.group_scales(rows, fmt, KB * D, WG, D, rsc), fmt, KB, WG, rsc)
        _place(dram, xa, x)
        _place(dram, wa, Q.hash_rows(d, wa, WG, D) if cfg.CHASH else d)
        _place(dram, sa, s)
        ors = N + 3
        mm = dict(sa=wa, ssa=sa, out=out, n=N, kb=KB, rs=d.shape[1], ors=ors, m=M, ab=0,
                  srs=s.shape[1], unit=unit, wf=wf, rsc=rsc, grp=Q.grp_r(fmt, WG), kt=True)
        prog += [I.ld(xa, 0, x.size), I.qact(0, M, 0, KB, KB * D), I.mm(**mm)]
        if acc:
            prog.append(I.mm(**mm, acc=True))
        xa += _up(x.nbytes, 1024)
        wa += _up(d.size, 1024)
        sa += _up(s.size, 1024)
        out += (M + 1) * ors
        assert xa < 1 << 20 and wa < 6 << 20 and sa < cfg.DRAM_BYTES and out < cfg.TMEM_WORDS
    return prog + [I.halt()], dram


def _kt_cfg(**kw):
    return _cfg(KT=True, ACT_BLOCKS=32, **kw)


def _kt_cases(wf):
    """A format's KT cases (R rows a group): one group (its tail chunk freed alone), a pair at
    KB 1 (the tail chunk alone) with ACC, three groups (an odd count's last alone) and four,
    replayed (M 9, 3) and UNIT, a last group of one row, KB 23 (GPT-OSS's K 2880) over five groups
    and replayed over a pair with ACC, RSC with M 16 (ACT_ROWS) at KB 1."""
    R = Q.grp_r(FMT[wf], WG)
    return [(wf, R, 2, 1, False, False, False),
            (wf, 2 * R, 1, 1, False, False, True),
            (wf, 3 * R - 5, 2, 9, False, False, False),
            (wf, 4 * R, 3, 3, True, False, False),
            (wf, R + 1, 5, 2, False, False, False),
            (wf, 5 * R - 1, 23, 1, False, False, False),
            (wf, 2 * R, 23, 2, False, False, True),
            (wf, 3 * R, 1, 16, False, True, False)]


@pytest.mark.parametrize("wf", [I.W8, I.W4I, I.W4F], ids=["int8", "int4", "fp4"])
def test_kt_mm_rtl(have_verilator, wf):
    """MM's K tail (otpu_mxu KT): the odd groups' first advance on the even ones' tail chunk
    (kept in the FIFO past their last pass), its ACT block's halves and sub-block offsets
    swapped, the blocks after it rotated, to the ISA simulator's TMEM bit for bit."""
    _check(_kt_cases(wf), seed=50 + wf, cfg=_kt_cfg(), program=_kt_program)


def test_kt_mm_posted_rtl(have_verilator):
    """KT on the BCU1525's memory path: port W's beats posted in quarters into the UltraRAM
    lanes (2,048 entries, WASK: an entry's reads trail its pop, occz), the native channels with
    random stalls, the MXU's command queue 4 deep."""
    ua = {"NAT_XPOST": 1, "NAT_XQB": 2, **UFIFO, "MXU_QD": 4}
    cases = _kt_cases(I.W8)[1:6] + _kt_cases(I.W4F)[:4] + _kt_cases(I.W4F)[6:]
    _check(cases, seed=57, cfg=_kt_cfg(), program=_kt_program, axi=True, stall=30, uarch=ua)


def test_kt_mm_shallow_fifo_rtl(have_verilator):
    """KT at an 8-entry chunk FIFO without the channel hash: the scale FIFO holds a pair's tail
    entries twice (two a chunk at KB 1), so the issuer waits on its own count (socc); the kept
    tail chunks across the FIFO's wrap, replayed and not."""
    cases = [(I.W8, 12 * 8, 1, 3, False, False, False),
             (I.W8, 9 * 8 - 3, 2, 1, False, False, True),
             (I.W4F, 7 * 16, 1, 1, False, False, False),
             (I.W4F, 6 * 16, 3, 2, True, False, False),
             (I.W4I, 5 * 16 - 7, 8, 1, False, False, False)]
    _check(cases, seed=58, cfg=_kt_cfg(CHASH=False), program=_kt_program,
           uarch={"FIFO_DEPTH": 8})


def _pump_cases(wf):
    """_cases plus passes back to back: KB 1 and 2 at M 8 and 7 (pg's four result FIFO rows a
    pass, an int8 pass's two: the passes' first advances wait on the last ones' rows), M 2 (one
    pass, no replay)."""
    return _cases(wf) + [(wf, 40, 1, 8, 0, False, False, "", False),
                         (wf, 24, 2, 7, 1, False, False, "", True),
                         (wf, 17, 1, 2, 0, True, False, "", False)]


@pytest.mark.parametrize("wf", [I.W8, I.W4I, I.W4F, I.WNV], ids=["int8", "int4", "fp4", "nvfp4"])
def test_pumped_mm_rtl(have_verilator, wf):
    """MXU_PUMP: grouped MMs of every format (M 1 .. 16: two rows a pass, an odd M's last pass
    of one, replay; KB 1 passes back to back) to the ISA simulator's TMEM bit for bit."""
    _check(_pump_cases(wf), seed=60 + wf, uarch=PUMP)


@pytest.mark.parametrize("wf", [I.W8, I.W4F, I.WNV], ids=["int8", "fp4", "nvfp4"])
def test_pumped_rmax_ascale_rtl(have_verilator, wf):
    """MXU_PUMP: RMAX and ASCALE by activation row, the pumped passes' rows drained as the
    unpumped ones'."""
    _check(_post_cases(wf), seed=70 + wf, uarch=PUMP)


@pytest.mark.parametrize("wf", [I.W8, I.W4I, I.W4F], ids=["int8", "int4", "fp4"])
def test_pumped_packed_extremes_rtl(have_verilator, wf):
    """MXU_PUMP at the packed PEs' bound (test_packed_extremes_rtl's activations and codes):
    both rows' sums interleaved on the cascade split exactly."""
    cases = [(wf, 16, 5, 2, 0, False, False, "", False),
             (wf, 24, 3, 4, 0, False, False, "", False),
             (wf, 32, 3, 9, 1, False, False, "", False),
             (wf, 16, 2, 3, 0, True, False, "", False)]
    _check(cases, seed=80 + wf, gen=dict(act=_extreme_act, matrix=_extreme_matrix), uarch=PUMP)


@pytest.mark.parametrize("target", ["bcu1525-x2:wmx", "bcu1525-x2"], ids=["wpb2", "wpb1"])
def test_pumped_one_tmem_rtl(have_verilator, target):
    """MXU_PUMP on the BCU1525's one-copy TMEM (TMEM_IMPL 3: :wmx's two writes a bank, the base's
    one): at WG > 1 the MXU's write port is a rotator (otpu_slice GEN_W), each of its writes a run
    from lane 0 that TMEM's simulation checks; passes back to back, a 4-bit pass's four rows, ACC,
    RMAX and ASCALE."""
    from opentpu import target as T
    ua = {k: v for k, v in T.sim_uarch(target).items()
          if k.startswith("TMEM_") or k in ("RPB", "WPB")}
    cases = _pump_cases(I.W8)[-3:] + _pump_cases(I.W4F)[-3:] + _post_cases(I.W8)[3:]
    _check(cases, seed=90, uarch=ua | PUMP)


def test_pumped_posted_rtl(have_verilator):
    """MXU_PUMP on the BCU1525's lanes: port W's beats posted in quarters into the UltraRAM lanes
    read stage by stage (WASK at the pumped columns' stage times, ceil(k / 2)), random stalls,
    the command queue 4 deep."""
    ua = {"NAT_XPOST": 1, "NAT_XQB": 2, **UFIFO, "MXU_QD": 4} | PUMP
    cases = _pump_cases(I.W8)[:3] + _pump_cases(I.W4F)[:3] + _pump_cases(I.WNV)[:2]
    cases += _pump_cases(I.W4F)[-3:] + _post_cases(I.W8)[3:4]
    _check(cases, seed=88, axi=True, stall=30, uarch=ua)


def test_pumped_kt_rtl(have_verilator):
    """MXU_PUMP with MM's K tail (KT): both rows' tail blocks swapped at column 0's input, on the
    posted UltraRAM lanes."""
    ua = {"NAT_XPOST": 1, **UFIFO} | PUMP
    cases = _kt_cases(I.W8) + _kt_cases(I.W4F)
    _check(cases, seed=89, cfg=_kt_cfg(), program=_kt_program, axi=True, uarch=ua)


def test_grouped_mm_unhashed_rtl(have_verilator):
    """Without the channel hash (CHASH off: a group chunk's rows in order, h = 0) on the
    behavioural DRAM (otpu_dram's port W lanes in order)."""
    _check(_cases(I.W4F)[:4] + _cases(I.W8)[:2], seed=13, chash=False)


# Qwen3.5-27B's decode MMs on one of bcu1525-x2's 4 slices (hidden 5120, FFN 17408, 48 DeltaNet value
# heads of 128, 16 key heads), as the decode quantizes their inputs: (K, ACT block, the rows N of
# the MMs that read it): in_proj_qkv and in_proj_z on x, out_proj on the heads' outputs, gate and
# up on x, down in two column parts (the ACT RAM's 128 blocks)
QWEN35_27B_S4 = [(5120, 0, (2560, 1536)), (1536, 40, (1280,)), (5120, 52, (4352, 4352)),
                 (8704, 0, (1280,)), (8704, 60, (1280,))]


def decode_mm_rate(wf=I.W8, layer=QWEN35_27B_S4, mhz=133.33, depth=None):
    """A layer's decode MMs (M = 1, GRP; 4-bit the packed advance) on tb_top's board
    memory path with the WG targets' channel model at `mhz` (opentpu.target.mem_model), at the WG targets'
    micro-architecture (targets/bcu1525-x2.toml: MXU_E 8, the one-copy TMEM, its FIFO depth unless
    `depth`; the scoreboard, not WAITC: the program has no WAITs): (P-chunks streamed, cycles,
    the channel model's chunk rate, the bytes the
    program needs: weights, their scale areas, the inputs, and the cycles the MMs stream: from
    each MM's start, or its input's QACT's end, to its end, as the trace's (the rest is the
    quantizer's, which the MMs wait for). The weights are zeros: the timing is the stream's."""
    from opentpu import target as T
    R = Q.grp_r(FMT[wf], WG)
    need = sum(-(-n // R) * Q.grp_row_bytes(k, FMT[wf], D, WG) for k, _, ns in layer for n in ns)
    cfg = _cfg(DRAM_BYTES=1 << max(23, (2 * need).bit_length()), ACT_BLOCKS=128)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    prog, wa, sa, out, chunks, nbytes = [], 1 << 20, cfg.DRAM_BYTES // 2, 1 << 15, 0, 0
    for k, ab, ns in layer:
        KB = k // D
        rs = Q.grp_row_bytes(k, FMT[wf], D, WG)
        srs = Q.grp_scale_bytes(k, FMT[wf], D, WG)
        prog += [I.ld(0, 0, KB * D), I.qact(0, 1, ab, KB, KB * D)]
        nbytes += KB * D * 4
        for n in ns:
            prog.append(I.mm(wa, sa, out, n, KB, rs, n, 1, ab, srs, wf=wf, grp=R))
            G = -(-n // R)
            chunks += G * (rs // (WG * D))
            nbytes += G * (rs + srs)
            wa += _up(G * rs, 1 << 12)
            sa += _up(G * srs, 1 << 12)
            out += n
    assert wa < cfg.DRAM_BYTES // 2 and sa < cfg.DRAM_BYTES and out < cfg.TMEM_WORDS
    plus = T.mem_model("bcu1525-x2", mhz)
    ua = {k: v for k, v in T.sim_uarch("bcu1525-x2").items()           # (XPOST: its own tests')
          if k not in ("SEQ_WC", "NAT_XPOST")}
    _, _, st = rtlsim.run(cfg, [prog + [I.halt()]], [dram],
                          uarch=ua | ({"FIFO_DEPTH": depth} if depth else {}),
                          axi=True, stall=0, plusargs=plus, trace=True)
    bpc = int(plus[0].split("=")[1]) / 1000            # a channel's bytes per cycle
    return (chunks, st["cycles"], 8 * bpc / (WG * D), nbytes,
            _streaming(cfg, prog + [I.halt()], st["trace"]))


def _streaming(cfg, prog, trace):
    """The cycles in which some MM streams (the union of [max(its start, its input's QACT's end),
    its end]), from slice 0's +trace lines (opentpu.profile; the program has no loops)."""
    from opentpu.profile import parse
    rec = {r.pc: r for r in parse(trace, cfg, [prog]).slice_recs(0)}
    spans, q = [], 0
    for k, ins in enumerate(prog):
        if ins.op == I.QACT:
            q = rec[k].end
        elif ins.op == I.MM:
            spans.append((max(rec[k].start, q), rec[k].end))
    n, last = 0, 0
    for a, b in sorted(spans):
        a = max(a, last)
        if b > a:
            n, last = n + b - a, b
    return n


@pytest.mark.parametrize("wf", [I.W8, I.W4F, I.WNV], ids=["int8", "fp4", "nvfp4"])
def test_decode_mm_rate_rtl(have_verilator, wf):
    """Acceptance (docs/weight_groups.md): while a Qwen3.5-27B layer's decode MMs (GRP; 4-bit
    packed, groups of 16 rows) stream, they read the bytes they need (weights, scales, inputs) at
    >= 0.9 of what the WG targets' channel model delivers, and their P-chunks at >= 0.9 of its chunk rate
    (int8, fp4; NVFP4's scales are 12.5% more bytes). The cycles the MMs wait for their inputs'
    QACTs are the quantizer's (printed: the whole layer's rate)."""
    chunks, cycles, port, nbytes, busy = decode_mm_rate(wf)
    rate, srate = chunks / cycles, chunks / busy
    used, sused = (nbytes / c / (port * WG * D) for c in (cycles, busy))
    print(f"WGRATE {FMT[wf]} chunks={chunks} cycles={cycles} streaming={busy} rate={rate:.3f} "
          f"port={port:.3f} ratio={rate / min(1.0, port):.3f} streaming_ratio="
          f"{srate / min(1.0, port):.3f} bytes={nbytes} used={used:.3f} streaming_used={sused:.3f}")
    assert sused >= 0.9 * min(1.0, port) / port
    if wf != I.WNV:         # (NVFP4's chunks are at most 1 / 1.125 of the bytes)
        assert srate >= 0.9 * min(1.0, port)
