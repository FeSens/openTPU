"""36-bit DRAM addresses on the RTL (AW 36: the BCU1525's 32 / 64 GB;
docs/bcu1525_wg.md section 8). The registers hold AW bits, a DRAM
address is R[r] + w mod 2^AW (w zero-extended), LI R[rd] = {w2, w1} and ADDI R[rd] = R[ra] +
{w2, w1} mod 2^AW, RLD MUL's product mod 2^AW; at AW 32 the w2 of LI and ADDI is ignored.

On the core's simulation (tb_top, the behavioural DRAM: AXI 0) at AW 36 the words above the dense
image live in otpu_dram's sparse store (dram_hi_<s>.txt in, dram_hi_out_<s>.txt out), so an
address that loses a bit above 31 reads or writes another word, visibly. On the BCU1525 at WG 8's
board model (tb_bcuwg: otpu_bcu_wg_sys, the bridges and the DIMMs' pairs in front of the DIMM
models) at CAW 25 (32 GiB: a channel 4 GiB) and 26 (64 GiB) the beats above the dense image are
the DIMM models' sparse ones (ch<c>_hi.txt in, ch<c>_hi_out.txt out).

Two kinds of program: hand-encoded ones (stores and loads at addresses that differ only in bits
32 .. L - 1, the carry from w into bit 32, the wrap at 2^AW, LI's w2 above AW, ADDI by -2^32 and
by small negative {w2, w1}, RLD MUL's products through every partial product and past 2^AW, RLD
writing the register's high bits) checked against the addresses they must land at; and the
fuzzers' and units' programs relocated by H (a multiple of 2^32: every DRAM address H higher,
every TMEM address and count the same, _relocate) against the ISA simulator's run of the program
itself (32-bit addresses) -- TMEM, and the DRAM at H, the images of a grouped matrix in the
channel hash's order at their new address (the fold over the address's bits above 31)."""
import copy

import numpy as np
import pytest

from opentpu import Config, isa as I, quant as Q, rtlsim
from opentpu.isasim import Machine

M32 = 0xFFFFFFFF
AW = 36


def _li(rd, v):
    """LI R[rd] = {w2, w1} = v (64 bits, two's complement)."""
    v &= (1 << 64) - 1
    return I.Instr(I.LI, rd=rd, w=[v & M32, v >> 32, 0, 0, 0, 0, 0])


def _addi(rd, ra, v):
    """ADDI R[rd] = R[ra] + {w2, w1}, {w2, w1} = v (64 bits, two's complement)."""
    v &= (1 << 64) - 1
    return I.Instr(I.ADDI, rd=rd, ra=ra, w=[v & M32, v >> 32, 0, 0, 0, 0, 0])


def _fill(t, v, n=16):
    return I.vop(I.V_FILL, t, 0, 0, 1, n, n, n, 0, I.B_SCALAR, v)


def _fv(v, n=16):
    return np.full(n, I.f32bits(v), np.uint32)


# ---------------------------------------------------------------------- the hand-encoded program
# low addresses (below the dense images: tb_top's 1 MiB, the board's 8 MiB)
B, P, X, OUT, OUT2 = 0x10000, 0x20000, 0x30000, 0x50000, 0x58000
TF, TL, TP, TX, TM = 0, 1024, 2048, 3072, 4096         # TMEM


def _rld_cases(L):
    """(x, raw, R[rb] (None: R0), w2): RLD's word, its multiplier R[rb] + w2 at AW bits, and
    the product mod 2^AW, a 64-byte aligned address below 2^L. A 32 x 32 product past 32 bits;
    f2i's value times a multiplier past 2^32 (the product past 2^36); R[rb]'s high bits
    alone; every partial product (x >= 2^20, m[31:16] and m[35:32] not 0); R[rb] + w2 carrying
    into bit 32; at L 36 products with bit 35 (m = 2^35 - 1 + 2^35: -x mod 2^36)."""
    c = [(0x12345678, True, None, 0x40),
         (I.f32bits(4100.0), False, 1 << 32, 0x40),
         (0x00400010, True, 0x5_0000_0000, 0x400),
         (0x6F036740, True, 0x1_0000_0000, 0x6B0D549B),
         (0x2217BE80, True, 0x5_FFFF_FFFF, 0x4A23D597)]
    if L > 35:
        c += [(0x9ABCDEF0, True, 0x3_0001_0000, 0x4),
              (0xFFFFFFC0, True, 0xF_FFFF_FFFF, 0)]
    return c


def _rld_product(x, raw, rb, w2):
    v = x if raw else I.f2i(x)
    m = ((rb or 0) + w2) % (1 << AW)
    return v * m % (1 << AW)


def _hand(L):
    """The hand-encoded program for a DRAM of 2^L bytes (L <= AW): (program, the words loaded
    before it {byte address: uint32 words}, the words it must store {byte address: words})."""
    K = 1 << (L - 32)
    prog, pre, want = [], {}, {}
    # the write path: K stores at B | k << 32, then each read back (LD) and stored low
    for k in range(K):
        prog += [_fill(TF + 16 * k, 1.0 + k), _li(1, k << 32), I.st(B, TF + 16 * k, 16, ra=1)]
        want[B | k << 32] = _fv(1.0 + k)
    for k in range(K):
        prog += [_li(2, k << 32), I.ld(B, TL + 16 * k, 16, ra=2)]
    prog.append(I.st(OUT, TL, 16 * K))
    want[OUT] = np.concatenate([_fv(1.0 + k) for k in range(K)])
    # the read path: K words loaded before the run at P | k << 32, read and stored low
    for k in range(K):
        pre[P | k << 32] = (np.arange(16, dtype=np.uint32) * 0x111 + 0xA000 + (k << 24)).astype(
            np.uint32)
        prog += [_li(3, (k << 32) | 0x100), I.ld(P - 0x100, TP + 16 * k, 16, ra=3)]
    prog.append(I.st(OUT2, TP, 16 * K))
    want[OUT2] = np.concatenate([pre[P | k << 32] for k in range(K)])
    marks = iter(range(100, 200))

    def mark_st(dram, ra, at):
        m = float(next(marks))
        t = TM + 16 * (int(m) - 100)
        want[at] = _fv(m)
        return [_fill(t, m), I.st(dram, t, 16, ra=ra)]
    # w + R[ra] carrying into bit 32; wrapping at 2^AW; LI's w2 above AW
    prog += [_li(4, 0x1_FFFF_F000)] + mark_st(0x2000, 4, 0x2_0000_1000)
    prog += [_li(5, (1 << AW) - 0x1000)] + mark_st(0x3000, 5, 0x2000)
    prog += [_li(6, ((1 << (AW - 32)) | 3) << 32 | 0x4000)] + mark_st(0, 6, 0x3_0000_4000)
    # ADDI: -2^32, -0x200, from R0 (an LI), a carry into bit 32, +2^33
    prog += [_li(7, 0x5_0000_0100)]
    prog += [_addi(8, 7, -(1 << 32))] + mark_st(0x5000, 8, 0x4_0000_5100)
    prog += [_addi(9, 7, -0x200)] + mark_st(0x10, 9, 0x4_FFFF_FF10)
    prog += [_addi(10, 0, 0x6_0000_7000)] + mark_st(0, 10, 0x6_0000_7000)
    prog += [_li(11, 0xFFFFFF00), _addi(11, 11, 0x200)] + mark_st(0x8000, 11, 0x1_0000_8100)
    prog += [_addi(12, 7, (1 << 33) + 0x40)] + mark_st(0, 12, 0x7_0000_0140)
    # RLD MUL: each product the address of a store
    xs = _rld_cases(L)
    pre[X] = np.array([x for x, *_ in xs] + [0x01234540], np.uint32)
    prog.append(I.ld(X, TX, len(xs) + 1))
    for i, (x, raw, rb, w2) in enumerate(xs):
        at = _rld_product(x, raw, rb, w2)
        assert at % 64 == 0 and at < 1 << L, hex(at)
        if rb is not None:
            prog.append(_li(15, rb))
        prog.append(I.rld(13, TX + i, raw=raw, mul=w2, rb=15 if rb is not None else 0))
        prog += mark_st(0, 13, at)
    # RLD without MUL writes all of the register: its old high bits go
    prog += [_li(14, 0xF << 32), I.rld(14, TX + len(xs), raw=True)] + mark_st(0, 14, 0x01234540)
    prog.append(I.halt())
    assert len(set(want)) == len(want)
    return prog, pre, want


def _aliases(pre, want):
    """The low 32 bits of each store above 4 GiB where the program puts nothing (where a store
    that lost its high bits would land)."""
    spans = [(a, a + 4 * len(w)) for a, w in list(pre.items()) + list(want.items())]
    return [a & M32 for a, w in want.items() if a >> 32 and not any(
        x0 < (a & M32) + 4 * len(w) and (a & M32) < x1 for x0, x1 in spans)]


def test_hand_program_addresses_are_distinct():
    """(No simulator.) The stores and the words loaded before the run do not overlap, all below
    2^L; the low 32 bits of every store above 4 GiB are free but the write and read paths' own
    (B | k << 32 all alias B, P | k << 32 P: the point of them)."""
    for L in (35, 36):
        prog, pre, want = _hand(L)
        spans = sorted((a, a + 4 * len(w)) for a, w in list(want.items()) + list(pre.items()))
        assert all(b0 >= a1 for (_, a1), (b0, _) in zip(spans, spans[1:])), "overlap"
        assert all(a1 <= 1 << L for _, a1 in spans)
        high = [a for a in want if a >> 32 and a & M32 not in (B, P)]
        assert len(_aliases(pre, want)) == len(high) >= 12


# ---------------------------------------------------------------------------- tb_top at AW 36
def _hi_text(words: dict) -> str:
    return "".join(f"{a:x} {int(v) & M32:08x}\n" for a, v in sorted(words.items()))


def _read_hi(path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            a, v = line.split()
            out[int(a, 16)] = int(v, 16)
    return out


def _top(cfg, prog, dense, hi: dict, tmp, uarch=None, **kw):
    """prog on tb_top at AW 36 (the behavioural DRAM): the dense image `dense` and the sparse
    words `hi` {word address: word} (every one at or above the dense image). Returns (the dense
    image after, TMEM, the sparse words after {word address: word}, stats)."""
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "dram_hi_0.txt").write_text(_hi_text(hi))
    drams, tmems, st = rtlsim.run(cfg, [prog], [dense], keep=tmp, axi=False,
                                  uarch={**(uarch or {}), "AW": AW}, **kw)
    return drams[0], tmems[0], _read_hi(tmp / "dram_hi_out_0.txt"), st


def _split(cfg, words: dict) -> tuple[dict, dict]:
    """{byte address: uint32 words} -> (dense {word address: word}, sparse {word address: word})."""
    lo, hi = {}, {}
    for a, w in words.items():
        assert a % 4 == 0
        for j, v in enumerate(np.asarray(w, np.uint32).tolist()):
            wa = a // 4 + j
            (lo if wa < cfg.DRAM_BYTES // 4 else hi)[wa] = v
    return lo, hi


WAITC = {"SEQ_WC": 1, "WIN": 16}


@pytest.mark.parametrize("ua", [{}, WAITC], ids=["scoreboard", "waitc"])
def test_hand_program_at_aw36_rtl(have_verilator, tmp_path, ua):
    """The hand-encoded program on tb_top at AW 36: every store where it must land (dense or
    sparse), nothing else written, the loads above 4 GiB reading the words loaded there. The
    footprint scoreboard (an instruction with an address past 4 GiB conflicts with every other,
    otpu_seq's Q stage) and WAITC (the compiler's wait entries; the BCU1525's sequencer)."""
    cfg = Config(S=1, DRAM_BYTES=1 << 20)
    prog, pre, want = _hand(AW)
    lo_pre, hi_pre = _split(cfg, pre)
    img = np.zeros(cfg.DRAM_BYTES // 4, np.uint32)
    for a, v in lo_pre.items():
        img[a] = v
    dense, _, hi, _ = _top(cfg, prog, img.view(np.uint8), hi_pre, tmp_path, ua)
    lo_want, hi_want = _split(cfg, want)
    exp = img.copy()
    for a, v in lo_want.items():
        exp[a] = v
    got = dense.view("<u4")
    bad = np.flatnonzero(got != exp)
    assert len(bad) == 0, [(hex(4 * int(a)), hex(int(got[a])), hex(int(exp[a]))) for a in bad[:8]]
    assert hi == {**hi_pre, **hi_want}, (
        sorted(hex(4 * a) for a in set(hi) ^ set(hi_pre) ^ set(hi_want))[:8],
        [(hex(4 * a), hex(hi.get(a, 0)), hex(v)) for a, v in hi_want.items() if hi.get(a) != v][:8])


def test_w2_is_ignored_at_aw32_rtl(have_verilator):
    """At AW 32 (every target but the BCU1525) LI and ADDI ignore w2, an address and RLD MUL's
    product wrap at 2^32 (the 1 MiB behavioural DRAM then wraps them at its size)."""
    cfg = Config(S=1, DRAM_BYTES=1 << 20)
    prog = [I.ld(X, TX, 1), _fill(0, 1.0), _fill(16, 2.0), _fill(32, 3.0), _fill(48, 4.0),
            _li(1, 5 << 32 | 0x100), I.st(0x1000, 0, 16, ra=1),
            _addi(2, 1, -(1 << 32) + 0x40), I.st(0x2000, 16, 16, ra=2),
            _li(3, 0xFFFFFF00), I.st(0x200, 32, 16, ra=3),
            I.rld(4, TX, raw=True, mul=0x40), I.st(0, 48, 16, ra=4), I.halt()]
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)
    img[X:X + 4] = np.array([0x12345678], "<u4").view(np.uint8)
    drams, _, _ = rtlsim.run(cfg, [prog], [img], axi=False)
    got = drams[0].view("<u4")
    for at, v in ((0x1100, 1.0), (0x2140, 2.0), (0x100, 3.0), (0x8D159E00 % (1 << 20), 4.0)):
        assert (got[at // 4:at // 4 + 16] == _fv(v)).all(), hex(at)


# --------------------------------------------------------------------------------- relocation
RH = 14                             # the register of the relocated DRAM fields named from R0


def _relocate(prog, H):
    """prog with every DRAM address H higher (H a multiple of 2^32, below 2^AW): LI, and ADDI
    from R0, take w2 = H >> 32 (every register holds its value + H: the low 32 bits, a TMEM
    address, a count, LOOP's trip, the same), and a DRAM field named from R0 (LD / ST / WAITW /
    DSTEP w1, STREAM w2 and w3, MM w1 and w2, QST w2 and w3) from R[RH] = H instead. Programs
    with RLD (a register's value from TMEM) or HALT CHAIN are not relocatable, nor an ADDI whose
    w1 is negative (its carry would move the high bits)."""
    assert H % (1 << 32) == 0 and 0 < H < 1 << AW
    named = {r for x in prog for r in (x.ra, x.rb, x.rc, x.rd)}
    assert RH not in named, f"R{RH} is the program's"
    fields = {I.LD: "a", I.ST: "a", I.WAITW: "a", I.DSTEP: "a", I.STREAM: "a", I.MM: "ab",
              I.QST: "bc"}
    bases = {getattr(x, f"r{f}") for x in prog for f in fields.get(x.op, "")} - {0}
    out = [_li(RH, H)]
    for x in prog:
        x = copy.deepcopy(x)
        assert x.op != I.RLD and not (x.op == I.HALT and x.flags & I.F_CHAIN)
        if x.op in (I.LI, I.ADDI) and x.rd in bases:
            assert x.w[0] < 1 << 31, "a DRAM base register's negative w1 (its carry moves H)"
        if x.op == I.LI or (x.op == I.ADDI and x.ra == 0):
            x.w = [x.w[0], H >> 32] + list(x.w[2:])
        elif x.op == I.ADDI:
            assert x.w[0] < 1 << 31 and x.w[1] == 0, "an ADDI of a negative w1"
        for f in fields.get(x.op, ""):
            if getattr(x, f"r{f}") == 0:
                setattr(x, f"r{f}", RH)
        out.append(x)
    return out


def _rehash(img, lo, hi, H, wg, D):
    """The grouped matrices' bytes [lo, hi) of image `img` (in the channel hash's order at their
    address) as they are at H + their address."""
    img[lo:hi] = Q.hash_rows(Q.hash_rows(img[lo:hi], lo, wg, D), lo + H, wg, D)


def _relocated_on_top(cfg, prog, img, H, tmp, uarch=None, grouped=()):
    """prog on the ISA simulator from `img`, and relocated by H (_relocate) on tb_top at AW 36
    from `img` at H (sparse) under its complement (dense, so a lost high bit reads other data):
    the same TMEM, and DRAM at H the simulator's (grouped: (lo, hi) ranges of grouped matrices,
    _rehash), nothing written elsewhere."""
    img = np.asarray(img, np.uint8)
    m = Machine(cfg, [prog], [img.copy()]).run()
    at = img.copy()
    for lo, hi in grouped:
        _rehash(at, lo, hi, H, cfg.WG, cfg.D)
    w = at.view("<u4")
    nz = np.flatnonzero(w)
    h4 = H // 4
    hi_pre = dict(zip((nz + h4).tolist(), w[nz].tolist()))
    scr = img ^ np.uint8(0xFF)
    dense, tmem, hi, st = _top(cfg, _relocate(prog, H), scr, hi_pre, tmp, uarch)
    assert st["instructions"][0] == m.slices[0].icount + 1
    bad = np.flatnonzero(tmem != m.slices[0].tmem)
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"
    assert np.array_equal(dense, scr), "a store landed below 4 GiB"
    stray = [a for a in hi if not h4 <= a < h4 + len(w)]
    assert not stray, f"stores outside the relocated image: {[hex(4 * a) for a in stray[:8]]}"
    got = np.zeros(len(w), np.uint32)
    ks = np.array(list(hi), np.int64)
    if len(ks):
        got[ks - h4] = np.array(list(hi.values()), np.uint32)
    exp = m.slices[0].dram.copy()
    for lo, hi_ in grouped:
        _rehash(exp, lo, hi_, H, cfg.WG, cfg.D)
    bad = np.flatnonzero(got != exp.view("<u4"))
    assert len(bad) == 0, f"{len(bad)} DRAM words at H differ, first at {[hex(4 * int(a)) for a in bad[:8]]}"
    return st


@pytest.mark.parametrize("seed,H,ua", [(0, 5 << 32, {}), (1, 0xB << 32, {}),
                                       (2, 0xF << 32, WAITC)], ids=["5-sb", "B-sb", "F-waitc"])
def test_fuzz_relocated_above_4gib_rtl(have_verilator, tmp_path, seed, H, ua):
    """The fuzzer's programs (VOPs, QACT + MM of every weight format, QST, LD / ST, LOOPs,
    GATHER) with their DRAM at 20, 44 and 60 GiB."""
    from test_rtl import _images, _random_program
    rng = np.random.default_rng(3600 + seed)
    cfg = Config(S=1)
    prog = _random_program(rng, cfg)
    assert any(x.op == I.MM for x in prog) and any(x.op == I.QST for x in prog)
    _relocated_on_top(cfg, prog, _images(rng, 1)[0], H, tmp_path, ua)


@pytest.mark.parametrize("H", [3 << 32, 0x9 << 32])
def test_stream_relocated_above_4gib_rtl(have_verilator, tmp_path, H):
    """STREAM (src and dst from R[ra]) and DSTEP with their states above 4 GiB (12 and 36 GiB),
    the stream engine's DMA reading and writing them."""
    from test_stream_rtl import _special, _stream_rtl_program, f
    cfg = Config(S=1, D=128, DRAM_BYTES=1 << 20, DSTEP=True, STREAM=True)
    r = np.random.default_rng(3700 + (H >> 32))
    prog = _stream_rtl_program(r, n_steps=10)
    assert any(x.op == I.STREAM and x.ra for x in prog) and any(x.op == I.DSTEP for x in prog)
    img = np.zeros(1 << 20, np.uint8)
    img[:4 * 4096] = _special(r, 4096).view(np.uint8)
    img[0x10000:0x10000 + 4 * 65536] = f(0.3 * r.standard_normal(65536)).view(np.uint8)
    _relocated_on_top(cfg, prog, img, H, tmp_path)


@pytest.mark.parametrize("H,ua", [(5 << 32, {}), (0xE << 32, WAITC)], ids=["5-sb", "E-waitc"])
def test_grouped_mm_relocated_above_4gib_rtl(have_verilator, tmp_path, H, ua):
    """Grouped MMs (WG 8, port W's 1 KiB chunks in the channel hash's lane order: the fold of
    the chunk index's bits above 31 too) of every format, ACC, RMAX and ASCALE, streaming their
    weights and scales from 20 and 56 GiB."""
    from test_wg_rtl import UARCH, _cases, _cfg, _post_cases, _program
    cases = _cases(I.W8)[:3] + _cases(I.W4F)[:3] + _cases(I.WNV)[:2]
    cases[2:2] = _post_cases(I.W8)[3:4] + _post_cases(I.W4F)[1:2]
    cfg = _cfg()
    prog, img = _program(np.random.default_rng(37 + (H >> 32)), cases, cfg=cfg)
    _relocated_on_top(cfg, prog, img, H, tmp_path, UARCH | ua, grouped=[(1 << 20, 6 << 20)])


def test_qst_grp_relocated_above_4gib_rtl(have_verilator, tmp_path):
    """QST GRP (rows and COL) writing a grouped int8 matrix and its scales at 28 GiB: the lanes
    of each chunk in the hash's order at the new address."""
    from test_wg_rtl import QST_GRP, UARCH, _cfg, _qst_program
    cfg = _cfg()
    prog, img = _qst_program(np.random.default_rng(5), QST_GRP, cfg)
    _relocated_on_top(cfg, prog, img, 7 << 32, tmp_path, UARCH, grouped=[(1 << 20, 6 << 20)])


def test_waitc_check_takes_dram_ranges_at_aw_bits_rtl(have_verilator, tmp_path, monkeypatch):
    """otpu_seq's simulation check of WAITC's entries (every conflict on the window has its
    wait) at AW bits: grouped MMs streaming their weights from 4 GiB, an ST just before the first
    writing 4 GiB below that MM's weights (the same low 32 bits: no conflict), under the waits
    pass's entries at AW 36 (no wait between them), run as the ISA simulator does (before, the
    check stopped it: the 32-bit ranges alias); the ST at the weights themselves (a conflict: the
    pass gives the MM a wait for it) under the same entries stops the run."""
    from opentpu import waits as W
    from test_wg_rtl import UARCH, _cases, _cfg, _program
    H, n = 1 << 32, 64
    cfg = _cfg()
    prog, img = _program(np.random.default_rng(41), _cases(I.W8)[:2], cfg=cfg)
    mi = next(k for k, x in enumerate(prog) if x.op == I.MM)
    r0 = [(0, 0)] * 16
    wlo = next(lo for sp, lo, _ in W.footprint(prog[mi], r0, cfg.D).rd if sp == W.SP_DRAM)
    ts, far = cfg.TMEM_WORDS - n, cfg.DRAM_BYTES - 4 * n   # the ST's TMEM; its place on the ISA
    for x in prog:
        fp = W.footprint(x, r0, cfg.D)
        for sp, lo, hi in fp.rd + fp.wr:
            assert not (sp == W.SP_TMEM and lo < ts + n and ts < hi), "TMEM's top is the program's"
            assert not (sp == W.SP_DRAM and lo < far + 4 * n and far < hi), "so are the far words"
    assert not img[far:].any()

    def with_st(p, k, dst, ra):
        return p[:k] + [I.st(dst, ts, n, ra=ra)] + p[k:]

    rel = _relocate(prog, H)
    ww = W.wait_counts(with_st(rel, mi + 1, wlo, 0), cfg.D, cfg.S, aw=AW)
    assert W.counts(ww[mi + 2])[0] == 0 and W.counts(ww[mi + 2])[5] == 0, "the MM waits on the DMA"
    wc = W.wait_counts(with_st(rel, mi + 1, wlo, RH), cfg.D, cfg.S, aw=AW)
    assert W.counts(wc[mi + 2])[0] != 0 and W.counts(wc[mi + 2])[5] != 0
    sync = tmp_path / "sync.txt"
    W.sync_file(ww, sync)
    monkeypatch.setenv("OTPU_WAITS", "0")
    m = Machine(cfg, [with_st(prog, mi, far, 0)], [img.copy()]).run()
    at = img.copy()
    _rehash(at, 1 << 20, 6 << 20, H, cfg.WG, cfg.D)
    w = at.view("<u4")
    nz = np.flatnonzero(w)
    hi_pre = dict(zip((nz + H // 4).tolist(), w[nz].tolist()))
    scr = img ^ np.uint8(0xFF)
    ua = UARCH | WAITC
    dense, tmem, hi, st = _top(cfg, with_st(rel, mi + 1, wlo, 0), scr, hi_pre, tmp_path / "alias",
                               ua, plusargs=[f"+hz_sync_s0={sync}"])
    assert st["instructions"][0] == m.slices[0].icount + 1
    bad = np.flatnonzero(tmem != m.slices[0].tmem)
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"
    exp = scr.copy()
    exp[wlo:wlo + 4 * n] = 0                                # (TMEM's top: zeros)
    assert np.array_equal(dense, exp) and hi == hi_pre
    with pytest.raises(RuntimeError, match=r"WAITC: pc \d+'s wait \d+ \(k 0\) misses slot"):
        _top(cfg, with_st(rel, mi + 1, wlo, RH), scr, hi_pre, tmp_path / "conflict", ua,
             plusargs=[f"+hz_sync_s0={sync}"])


# --------------------------------------------------------------------------- tb_bcuwg at AW 36
TARGET = "bcu1525-wg"
DENSE = 1 << 23                     # the board model's dense image (every board test's: one build)
BPROG = 0x7C0000                    # the programs' place (below 4 GiB)


def _hi_transport():
    from opentpu.host.board import SimTransport
    from opentpu import simtmp

    class HiTransport(SimTransport):
        """SimTransport (bcuwg) with tb_bcuwg at CAW and the channels' beats past the dense image
        held here sparsely ({32-byte beat: 32 bytes}), handed to the DIMM models as ch<c>_hi.txt
        and read back from ch<c>_hi_out.txt (otpu_dimm_model)."""

        def __init__(self, caw, **kw):
            super().__init__(**kw)
            self.caw = caw
            self.hi = [{} for _ in self.ch]

        def sim_build(self):
            top, srcs, p = super().sim_build()
            return top, srcs, {**p, "CAW": self.caw}

        def mem_write(self, ch, off, data):
            self.flush()
            data = np.ascontiguousarray(data, np.uint8).reshape(-1)
            k = min(len(data), max(0, len(self.ch[ch]) - off))
            self.ch[ch][off:off + k] = data[:k]
            x, end, hi = off + k, off + len(data), self.hi[ch]
            while x < end:
                b, j = divmod(x, 32)
                e = min(end, 32 * b + 32)
                v = hi.get(b)
                part = data[x - off:e - off]
                if v is None and not part.any():
                    x = e
                    continue
                v = np.zeros(32, np.uint8) if v is None else v.copy()
                v[j:j + e - x] = part
                hi[b] = v
                x = e

        def mem_read(self, ch, off, n, out=None):
            self.flush()
            r = np.zeros(n, np.uint8)
            k = min(n, max(0, len(self.ch[ch]) - off))
            r[:k] = self.ch[ch][off:off + k]
            for b, v in self.hi[ch].items():
                lo, hi = max(32 * b, off + k), min(32 * b + 32, off + n)
                if lo < hi:
                    r[lo - off:hi - off] = v[lo - 32 * b:hi - 32 * b]
            if out is None:
                return r
            out[:] = r
            return out

        def flush(self):
            if not self.script:
                return
            top, srcs, p = self.sim_build()
            exe = rtlsim.build(top, srcs, p)
            with simtmp.tempdir("otpu_board_") as d:
                for c in range(len(self.ch)):
                    self.ch[c].view("<u4").astype(">u4").tofile(d / f"ch{c}.bin")
                    (d / f"ch{c}_hi.txt").write_text("".join(
                        f"{b:x} {int.from_bytes(v.tobytes(), 'little'):064x}\n"
                        for b, v in sorted(self.hi[c].items())))
                (d / "host.txt").write_text("\n".join(self.script) + "\n")
                self.script, self.nreads = [], 0
                r = rtlsim.run_sim([str(exe), f"+dir={d}", f"+axi_stall={self.stall}",
                                    f"+axi_seed={self.seed}", *self.plusargs])
                out = self.out = r.stdout + r.stderr
                if "DONE" not in out:
                    raise RuntimeError(f"board simulation failed:\n{out[-3000:]}")
                self.results = []
                for line in out.splitlines():
                    if line.startswith("REG "):
                        _, a, v = line.split()
                        self.results.append(int(v, 16))
                        self.regs_seen[int(a, 16)] = int(v, 16)
                    elif line.startswith("DONE"):
                        self.cycles += int(line.split("=")[1])
                for c in range(len(self.ch)):
                    self.ch[c] = np.fromfile(d / f"ch{c}_out.bin", np.uint8)
                    self.hi[c] = {}
                    for line in (d / f"ch{c}_hi_out.txt").read_text().splitlines():
                        if line.strip():
                            b, v = line.split()
                            self.hi[c][int(b, 16)] = np.frombuffer(
                                int(v, 16).to_bytes(32, "little"), np.uint8).copy()
    return HiTransport


def _board(caw):
    """A Board on tb_bcuwg at CAW (25: 32 GiB, 26: 64 GiB) with otpu_board at AW 36, the target's
    core (test_bcuwg_board's parameters)."""
    from opentpu import target as T
    from opentpu.host.board import Board
    t = T.load(TARGET)
    rp = T.rtl_params(t)
    p = {k: rp[k] for k in ("MCOLS", "LANES", "TMEM_WORDS", "SEQ_WC", "OUTER", "DSTEP2")}
    p |= {"WG": t.arch["WG"], "ACT_ROWS": t.arch["ACT_ROWS"], "NAT_XPOST": 1, "AW": AW}
    tr = _hi_transport()(caw, ch_bytes=DENSE // 8, stall=20, seed=5, target=t, bcuwg=True,
                         params=p)
    b = Board(tr)
    c = b.info()["cfg"]
    assert (c["channels"], c["channel_bytes"]) == (8, 1 << (caw + 7)), c
    assert f".AW({AW})" in tr.sim_build()[1][0].read_text()
    return b, tr, T.config(t, DRAM_BYTES=DENSE)


@pytest.fixture(scope="module", params=[25, 26], ids=["32GB", "64GB"])
def bcu(request):
    import shutil
    if shutil.which("verilator") is None:
        pytest.skip("verilator not installed")
    return (request.param, *_board(request.param))


def test_hand_program_on_bcuwg_board_model(bcu):
    """The hand-encoded program on the board model at 32 / 64 GiB (bits 32 .. 34, and 35 at 64
    GiB): every store where it must land, the low 32 bits of each one above 4 GiB untouched."""
    caw, b, tr, cfg = bcu
    L = caw + 10
    prog, pre, want = _hand(L)
    for a, w in pre.items():
        b.write(a, np.asarray(w, "<u4").view(np.uint8))
    b.load(BPROG, prog)
    st = b.run()
    assert st["instructions"][0] == len(prog)
    for a, w in want.items():
        got = b.read(a, 4 * len(w)).view("<u4")
        assert np.array_equal(got, w), (hex(a), [hex(int(v)) for v in got[:4]])
    for a in _aliases(pre, want):
        assert not b.read(a, 64).any(), f"{a:#x} written (a store's low 32 bits)"


def _relocated_on_board(b, cfg, prog, src, H, grouped=(), out=None):
    """prog on the ISA simulator from image `src` (zeros up to the DRAM's size), and relocated by
    H on the board model from it at H under its complement below 4 GiB: as many instructions,
    DRAM at H the simulator's (grouped: _rehash), nothing written below the program's place."""
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)
    img[:len(src)] = src
    m = Machine(cfg, [prog], [img.copy()]).run()
    at = img.copy()
    for lo, hi in grouped:
        _rehash(at, lo, hi, H, cfg.WG, cfg.D)
    scr = img ^ np.uint8(0xFF)
    b.write(0, scr)
    b.write(H, at)
    rp = _relocate(prog, H)
    b.load(BPROG, rp)
    st = b.run()
    assert st["instructions"][0] == m.slices[0].icount + 1
    exp = m.slices[0].dram.copy()
    for lo, hi in grouped:
        _rehash(exp, lo, hi, H, cfg.WG, cfg.D)
    got = b.read(H, len(img))
    bad = np.flatnonzero(got != exp)
    assert len(bad) == 0, f"{len(bad)} DRAM bytes at H differ, first at {[hex(int(a)) for a in bad[:8]]}"
    assert np.array_equal(b.read(0, BPROG), scr[:BPROG]), "a store landed below 4 GiB"
    if out is not None:
        assert np.count_nonzero(exp[out[0]:out[1]]) > (out[1] - out[0]) // 8   # (it wrote)
    return st


def test_grouped_mm_relocated_on_bcuwg_board_model(bcu):
    """Grouped MMs (int8, fp4, NVFP4; ACC, RMAX, ASCALE) streaming their weights and scales
    from above 4 GiB (20 GiB; at 64 GiB 44 GiB, bit 35) through port W, the bridges and the
    DIMMs, their outputs stored there."""
    from test_wg_rtl import _cases, _post_cases, _program
    caw, b, tr, cfg = bcu
    H = (5 if caw == 25 else 0xB) << 32
    cases = _cases(I.W8)[:2] + _cases(I.W4F)[:2] + _cases(I.WNV)[:1]
    cases[2:2] = _post_cases(I.W8)[3:4] + _post_cases(I.W4F)[1:2]
    prog, img = _program(np.random.default_rng(41), cases, cfg=cfg)
    m = Machine(cfg, [prog], [img.copy()]).run()
    o, used = 7 << 20, int(np.flatnonzero(m.slices[0].tmem[1 << 14:]).max()) + 1
    prog = prog[:-1] + [I.st(o, 1 << 14, used), I.halt()]        # (the outputs into DRAM)
    _relocated_on_board(b, cfg, prog, img, H, grouped=[(1 << 20, 6 << 20)],
                        out=(o, o + 4 * used))


def test_stream_relocated_on_bcuwg_board_model(bcu):
    """STREAM and DSTEP with their states above 4 GiB (12 GiB; at 64 GiB 40 GiB) on the board
    model."""
    from test_stream_rtl import _special, _stream_rtl_program, f
    caw, b, tr, cfg = bcu
    H = (3 if caw == 25 else 0xA) << 32
    r = np.random.default_rng(3800 + caw)
    prog = _stream_rtl_program(r, n_steps=8)
    img = np.zeros(1 << 20, np.uint8)
    img[:4 * 4096] = _special(r, 4096).view(np.uint8)
    img[0x10000:0x10000 + 4 * 65536] = f(0.3 * r.standard_normal(65536)).view(np.uint8)
    _relocated_on_board(b, cfg, prog, img, H)


def test_beyond_the_dram_stops_the_board_model(bcu):
    """At 32 GiB (DRAM_LOG2 35 < AW 36) a store at bit 35 is a beat beyond the channels': the
    simulation stops (otpu_board g_chk); at 64 GiB the whole of 2^AW is DRAM."""
    caw, b, tr, cfg = bcu
    if caw != 25:
        pytest.skip("64 GiB: every address is DRAM")
    prog = [_fill(0, 7.0), _li(1, 8 << 32), I.st(0x1000, 0, 16, ra=1), I.halt()]
    b.load(BPROG, prog)
    with pytest.raises(RuntimeError, match="beyond its 2\\^32 bytes"):
        b.run()
