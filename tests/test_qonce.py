"""Quantize-once (Config.QONCE, CAPS2 bit24; docs/dma_ldq.md): QACT
flag ST stores each row's ACT bytes and records to DRAM, LDA loads them back into ACT RAM
unchanged. The ISA simulator, the compiler's quantize(store=) / act_load, the wait entries'
footprints, and the expert-major expert run that loads each entry's row instead of norming and
quantizing it again: QONCE on against off, every logit bit equal, at S 1, 2 and 4."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from opentpu import Config, isa as I, quant as Q, waits
from opentpu.host import regs as R
from opentpu.isasim import Machine, SimError

X, ST, OUT = 0, 0x40000, 0x80000


def _put(dram, at, a):
    b = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
    dram[at:at + b.size] = b


def _rows(rng, rows, n, g, spread=3.0):
    x = (rng.standard_normal((rows, n)) *
         np.exp(rng.uniform(-spread, spread, (rows, n // g, 1))).repeat(g, 2).reshape(rows, n))
    x = x.astype(np.float32)
    x[0, g:2 * g] = 0
    return x


def test_encodings():
    """LDA is opcode 0x14 (a DMA instruction); QACT flag ST is bit6, its DRAM row base in w3
    (R[rc]), the source row stride then in w6 and the DRAM row stride in w7; a row is KB x D
    bytes and KB records of QREC bytes."""
    assert I.LDA == 0x14 and I.OPNAMES[I.LDA] == "LDA" and waits.UNIT[I.LDA] == waits.U_DMA
    assert I.F_QST == 0x40 and I.QREC == 4 + Q.NSUB == 8
    assert I.qrow_bytes(16, 128) == 2176
    q = I.qact(0x100, 3, 5, 2, 64, store=(0x2000, 512), rc=7, rscale=9)
    assert q.flags == I.F_QST | I.F_RSCALE and q.rc == 7
    assert q.w[:7] == [0x100, 3 | 5 << 8 | 2 << 16, 0x2000, 0, 9, 64, 512]
    assert I.qact(0x100, 3, 5, 2, 64).w[2] == 64
    with pytest.raises(ValueError):
        I.qact(0, 2, 0, 1, 32, hl=True, store=(0, 64))
    a = I.lda(0x3000, 4, 6, 2, 640, arow=8, ra=3)
    assert a.op == I.LDA and a.ra == 3 and a.w[:4] == [0x3000, 4 | 6 << 8 | 2 << 16, 640, 8]


def test_refused_without_qonce():
    cfg = Config(S=1, D=32, MCOLS=4, DRAM_BYTES=1 << 20)
    for ins in (I.qact(0, 2, 0, 1, 32, store=(ST, 64)), I.lda(ST, 1, 0, 1, 0)):
        with pytest.raises(SimError, match="QONCE"):
            Machine(cfg, [[ins, I.halt()]], [np.zeros(cfg.DRAM_BYTES, np.uint8)]).run()


@pytest.mark.parametrize("row,scales,dup", [(False, False, False), (True, False, False),
                                            (False, True, False), (False, True, True)])
def test_qact_st_then_lda_is_the_qact(row, scales, dup):
    """QACT ST writes, per row, the KB x D bytes ACT RAM got and KB records (S, the NSUB offsets);
    LDA of those rows into other ACT rows and blocks gives ACT RAM the QACT's bytes, scales and
    offsets exactly (row by row from a register base, or all rows one LDA)."""
    cfg = Config(S=1, D=32, MCOLS=16, ACT_ROWS=16, DRAM_BYTES=1 << 20, QONCE=True)
    D, Rr, KB, ab, drs = cfg.D, 6, 3, 4, 512
    rng = np.random.default_rng(3 + row + 2 * scales)
    x = _rows(rng, Rr, KB * D, D // 4)
    cs = np.exp(rng.uniform(-1, 1, KB * D)).astype(np.float32)
    rsv = np.exp(rng.uniform(-1, 1, Rr)).astype(np.float32)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, np.concatenate([x.reshape(-1), cs, rsv]))
    n = x.size + cs.size + rsv.size
    kw = dict(row=row, dup=dup, cscale=Rr * KB * D if scales else None,
              rscale=Rr * KB * D + KB * D if scales else None)
    prog = [I.ld(X, 0, n), I.li(5, ST - 0x100), I.qact(0, Rr, ab, KB, KB * D, store=(0x100, drs),
                                                         rc=5, **kw)]
    prog += [I.lda(ST, Rr, ab + KB, KB, drs)]                       # all rows, one LDA
    prog += [I.lda(ST + r * drs, 1, ab + 2 * KB, KB, 0, arow=Rr + r) for r in range(Rr)]
    m = Machine(cfg, [prog + [I.halt()]], [dram]).run()
    s = m.slices[0]
    a0 = slice(ab * D, (ab + KB) * D)
    a1, a2 = slice((ab + KB) * D, (ab + 2 * KB) * D), slice((ab + 2 * KB) * D, (ab + 3 * KB) * D)
    assert np.array_equal(s.act[:Rr, a0], s.act[:Rr, a1])
    assert np.array_equal(s.act[:Rr, a0], s.act[Rr:2 * Rr, a2])
    for f in ("ascale", "adoff"):
        v = getattr(s, f)
        assert np.array_equal(v[:Rr, ab:ab + KB], v[:Rr, ab + KB:ab + 2 * KB]), f
        assert np.array_equal(v[:Rr, ab:ab + KB], v[Rr:2 * Rr, ab + 2 * KB:ab + 3 * KB]), f
    if dup:                                     # (DUP's copies are the stored rows too)
        assert np.array_equal(s.act[Rr:2 * Rr, a0], s.act[:Rr, a0])
    blk = s.dram[ST:ST + Rr * drs].reshape(Rr, drs)[:, :I.qrow_bytes(KB, D)]
    assert np.array_equal(blk[:, :KB * D].view(np.int8), s.act[:Rr, a0])
    rec = blk[:, KB * D:].reshape(Rr, KB, I.QREC)
    assert np.array_equal(np.ascontiguousarray(rec[:, :, :4]).view(np.float32)[..., 0],
                          s.ascale[:Rr, ab:ab + KB])
    assert np.array_equal(rec[:, :, 4:], s.adoff[:Rr, ab:ab + KB])
    assert not s.dram[ST + I.qrow_bytes(KB, D):ST + drs].any()


def test_lda_bounds():
    cfg = Config(S=1, D=32, MCOLS=8, ACT_ROWS=16, DRAM_BYTES=1 << 20, QONCE=True)
    for bad in (I.lda(ST, 2, 0, 1, 0, arow=15), I.lda(ST, 1, cfg.ACT_BLOCKS - 1, 2, 0),
                I.lda(ST, 2, 0, 2, 64),                     # (rows overlapping: 80 bytes each)
                I.lda(ST + 32, 1, 0, 1, 0), I.lda(ST, 2, 0, 1, 96)):   # (not QALIGN multiples)
        with pytest.raises(SimError, match="LDA"):
            Machine(cfg, [[bad, I.halt()]], [np.zeros(cfg.DRAM_BYTES, np.uint8)]).run()


def test_footprints():
    """The wait entries' footprints (otpu_pkg footprint() QX): QACT ST reads its source at srs
    (w6) and writes its rows' bytes and records to DRAM at R[rc] + w3, w7 apart; LDA reads the
    DRAM rows (R[ra] + w1, w3 apart) and writes ACT RAM's blocks ab .. ab + KB - 1."""
    Rg = [(0, 0)] * 16
    Rg[5] = (0x1000, 0x1000)
    q = I.qact(0x100, 3, 5, 2, 64, store=(0x40, 4096), rc=5)
    f = waits.footprint(q, Rg, D=128)
    n = I.qrow_bytes(2, 128)
    assert (waits.SP_TMEM, 0x100, 0x100 + 2 * 64 + 256) in f.rd
    assert (waits.SP_DRAM, 0x1040, 0x1040 + 2 * 4096 + n) in f.wr
    assert (waits.SP_ACT, 5, 7) in f.wr
    a = I.lda(0x80, 3, 6, 2, 4096, arow=1, ra=5)
    f = waits.footprint(a, Rg, D=128)
    assert f.rd == ((waits.SP_DRAM, 0x1080, 0x1080 + 2 * 4096 + n),)
    assert f.wr == ((waits.SP_ACT, 6, 8),)


def test_qonce_caps2_target():
    """CAPS2 bit24 is QONCE; [arch] QONCE in the CFG_HASH only when on, the build knob QONCE (with
    SEQ_WC)."""
    from opentpu import target as T
    assert R.CAP2_QONCE == 1 << 24
    assert R.caps2(R.CAP2_QONCE)["qonce"] and not R.caps2(None)["qonce"]
    t = T.load("bcu1525-x2")
    base, on = T.config(t), T.build_config(t, QONCE=1)
    assert not base.QONCE and on.QONCE and not T.build_config(t, QONCE=1, SEQ_WC=0).QONCE
    assert T.arch_hash(base) == T.arch_hash(dataclasses.replace(base, QONCE=False))
    assert T.arch_hash(on) != T.arch_hash(base)


def test_qonce_target_rtl_params(tmp_path):
    """[arch] QONCE on a target: the RTL parameter QONCE (otpu_slice, otpu_board) and the
    simulators' with it, only where the quantizer's SW port is as wide as its ACT writes (4 x
    QUANT_SWW = ULANES x QUANT_PW: x2:wmx's SWW 4, PW 2, LANES 8) and with the wait entries."""
    from opentpu import rtlsim
    from opentpu import target as T
    ROOT = Path(__file__).resolve().parents[1]
    p = tmp_path / "bcu1525-x2.toml"
    txt = (ROOT / "targets/bcu1525-x2.toml").read_text()
    p.write_text(txt.replace("[variants.wmx]\n", "[variants.wmx]\nQONCE = true\n", 1)
                 + "\n[variants.qbad]\nQONCE = true\n")
    T.load.cache_clear()
    try:
        t = T.load(f"{p}:wmx")
        assert T.rtl_params(t)["QONCE"] == 1 and T.sim_uarch(t)["QONCE"] == 1
        assert T.config(t).QONCE and "QONCE" not in T.rtl_params(T.load(str(p)))
        assert rtlsim.top_params(T.config(t))["QONCE"] == 1
        with pytest.raises(T.TargetError, match="QONCE needs the quantizer's SW port"):
            T.load(f"{p}:qbad")
    finally:
        T.load.cache_clear()


def test_quantize_store_and_act_load_compiler():
    """ol.quantize(x, store=) then ol.act_load of its rows (one at a time, reversed, into a
    stationary of their own): the same dot bit for bit as quantize(x)'s; the wait entries cover
    the stores and loads."""
    from opentpu import language as ol
    from opentpu import target as T
    from opentpu.compiler import Affine, QTensor, Tensor, current
    cfg = dataclasses.replace(T.config("bcu1525-x2:wmx"), S=1, DRAM_BYTES=1 << 22, QONCE=True)
    D, Rr, wg = cfg.D, 12, cfg.WG
    K = 2 * D
    rng = np.random.default_rng(5)
    x = _rows(rng, Rr, K, D // 4)
    w = rng.standard_normal((D, K)).astype(np.float32)
    rs, srs = Q.grp_row_bytes(K, "int8", D, wg), Q.grp_scale_bytes(K, "int8", D, wg, True)
    WA, WQ, drs = 0x40000, 0x100000, 1024
    wq = QTensor(Affine(WQ), Affine(WQ + (D // wg) * rs), (D, K), rs, srs, D, rsc=True, grp=wg)
    progs = []
    for once in (False, True):
        @ol.jit
        def kern():
            wt = ol.load(Tensor(Affine(WA), (D, K), (K, 1)))
            current().store_quantized(wt, wq.data, wq.scale, 0, wq.srs, row_scale=True,
                                      row=Affine(0))
            del wt
            t = ol.load(Tensor(Affine(X), (Rr, K), (K, 1)))
            if once:
                ol.quantize(t, store=(Affine(ST), drs))
                a = ol.stationary(Rr, K)
                for r in reversed(range(Rr)):
                    ol.act_load(a, r, Affine(ST + r * drs))
            else:
                a = ol.quantize(t)
            ol.store(Tensor(Affine(OUT), (Rr, D), (D, 1)), ol.dot(a, wq))
        progs.append(kern.trace(cfg, 0, {}).finish())
    assert sum(i.op == I.LDA for i in progs[1]) == Rr
    assert sum(i.op == I.QACT and bool(i.flags & I.F_QST) for i in progs[1]) == 1
    outs = []
    for p in progs:
        dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
        _put(dram, X, x)
        _put(dram, WA, w)
        wc = waits.wait_counts(p, D, 1, aw=cfg.AW, win=cfg.WC_WIN)
        m = Machine(cfg, [p], [dram]).run()
        waits.check_isasim(m, [wc])
        outs.append(m.slices[0].dram[OUT:OUT + 4 * Rr * D].copy())
    assert np.array_equal(outs[0], outs[1])


@pytest.mark.parametrize("S,shape,rows,pipe", [
    (1, {}, 8, 2),
    (1, {}, 8, 0),
    (2, {}, 16, 2),
    (2, {}, 8, 1),
    (4, dict(nv=8, H=512, F=512), 8, 2),
])
def test_expert_run_quantized_once_is_bit_exact(monkeypatch, S, shape, rows, pipe):
    """The expert-major prefill on x2:wmx's machine at S slices (a resident pool, tiles of `rows`
    rows; EM_PIPE `pipe`: 2 the deep schedule, 1 pipelined, 0 a pass at a time) with QONCE: the
    prologue's router QACT stores each row to its record's em_q (QACT ST) and every expert-run
    entry loads it into ACT RAM (LDA) in place of its X and 1 / rms and a QACT. Against the same
    machine without QONCE: the prefill's logits and the decode steps after, every bit; the wait
    entries checked against the runs (OTPU_WAITS_CHECK), the deep schedule used at EM_PIPE 2."""
    from test_moe_s2 import _tiny
    from opentpu import isasim as SIM
    from opentpu import target as T
    from opentpu.llm import moe as MO
    from opentpu.llm import qwen35 as Q35
    from opentpu.llm.qwen3 import Engine
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    monkeypatch.setattr(MO, "EM_PIPE", pipe)
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    monkeypatch.setenv("OTPU_WAITS_CHECK", "1")
    used, deep = [], MO._em_deep

    def spy(u):
        used.append(deep(u))
        return used[-1]
    monkeypatch.setattr(MO, "_em_deep", spy)
    n = {"st": 0, "lda": 0}
    qrows, lda = SIM.Slice._qrows_out, SIM.Slice._lda

    def spy_st(self, *a):
        n["st"] += 1
        return qrows(self, *a)

    def spy_lda(self, ins):
        n["lda"] += 1
        return lda(self, ins)
    monkeypatch.setattr(SIM.Slice, "_qrows_out", spy_st)
    monkeypatch.setattr(SIM.Slice, "_lda", spy_lda)
    W, spec = _tiny(("linear", "attn"), **shape)
    toks = [int(t) for t in np.random.default_rng(13).integers(0, 1000, 43)]
    out = []
    for once in (False, True):
        cfg = T.config("bcu1525-x2:wmx", S=S, DRAM_BYTES=1 << 30, BHW=0, IMEM_WORDS=1 << 17,
                       QONCE=once)
        e = Engine(spec, W, expert_major=True, tile_rows=rows, cap=256, cfg=cfg, rows=1,
                   resident=True, layer_major=2)
        assert e.expert_major and e.image.pool is not None
        logits = e.prefill(toks)
        assert (n["st"] > 0 and n["lda"] > 0) == once, n
        t, steps = int(np.argmax(logits)), []
        for _ in range(2):
            steps.append(e.step(t))
            t = int(np.argmax(steps[-1]))
        out.append([logits] + steps)
    assert all(used) and bool(used) == (pipe == 2), used
    for a, b in zip(*out):
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


WA, SC, QB = 0x40000, 0x48000, 0x50040        # (QB: 64- and not 128-byte aligned)


def _qonce_prog(D, lanes, rng, hold=False):
    """test_qonce_rtl's run: (cfg, program, DRAM, index of each pass's all-rows LDA and of the
    QACT beside it). Per pass: a QACT ST of R rows (ROW, CSCALE, RSCALE, DUP, LO; KB 1 .. 4; row
    strides 64-byte multiples, odd ones too, from a register base), LDA of all its rows into
    other blocks while a QACT of other rows writes blocks of its own (the ACT RAM's two writers),
    LDA of each row alone into reversed ACT rows (w4), LDA of all rows at ACT row R (rows R ..
    2R - 1 of the QACT's own blocks), every block read back by identity MMs (UNIT: the bytes;
    int8x: the records' S and offsets). `hold`: a last QACT ST of 16 rows of 8 blocks (128
    records: the record FIFO full under pass 1's byte writes) loaded back the same way."""
    cfg = Config(S=1, D=D, MCOLS=4, ACT_ROWS=16, LANES=lanes, TMEM_WORDS=1 << 17,
                 DRAM_BYTES=1 << 20, SPLITQ=True, QONCE=True, SEQ_WC=True)
    R, KM = 8, 8 if hold else 4
    x = _rows(rng, 2 * R, KM * D, D // 4)
    cs = np.exp(rng.uniform(-1, 1, KM * D)).astype(np.float32)
    rsf = np.exp(rng.uniform(-1, 1, 2 * R)).astype(np.float32)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, np.concatenate([x.reshape(-1), cs, rsf]))
    T, n = x.size, x.size + cs.size + rsf.size
    _put(dram, WA, np.eye(D, dtype=np.int8))
    int8x = np.zeros((D, 8), np.uint8)
    int8x[:, :4] = np.array([1.0], np.float32).view(np.uint8)   # Sw 1, dw 0
    _put(dram, SC, int8x)
    prog = [I.ld(X, 0, n), I.li(5, QB - 0x40)]
    out, srs, sd, pairs = n + 64, KM * D, QB, []

    def mms(ab, kb, m):
        nonlocal out
        for k in range(kb):
            prog.append(I.mm(WA, 0, out, D, 1, D, D, m, ab + k, 0, unit=True))
            prog.append(I.mm(WA, SC, out + m * D, D, 1, D, D, m, ab + k, 8))
            out += 2 * m * D

    def lda_back(kb, drs, rows):
        prog.append(I.li(6, sd))
        pairs.append(len(prog))
        prog.append(I.lda(0, rows, kb, kb, drs, ra=6))             # all rows, other blocks
        prog.append(I.qact(rows * srs, rows, 2 * kb, kb, srs))     # (beside it: the quantizer)
        for r in range(rows):                                      # each row, reversed rows
            prog.append(I.lda(r * drs, 1, 3 * kb, kb, 0, arow=rows - 1 - r, ra=6))
        if rows <= 8:
            prog.append(I.lda(sd, rows, 0, kb, drs, arow=rows))    # rows R .. 2R - 1
        mms(0, kb, min(2 * rows, 16))
        for b in (kb, 2 * kb, 3 * kb):
            mms(b, kb, rows)

    # (ROW, CSCALE, RSCALE, DUP, LO, KB, row stride pad, rows: DUP's at most MCOLS / 2)
    for row, csf, rscf, dup, lo, kb, pad, rows in [
            (False, False, False, False, False, 3, 0, R),
            (True, True, False, False, False, 4, 64, R),
            (False, False, True, True, False, 1, 0, cfg.MCOLS // 2),
            (False, True, True, False, True, 2, 64, R)]:
        drs = -(-I.qrow_bytes(kb, D) // 64) * 64 + pad
        prog.append(I.qact(0, rows, 0, kb, srs, row=row, cscale=T if csf else None,
                           rscale=T + cs.size if rscf else None, dup=dup, lo=lo,
                           store=(sd - (QB - 0x40), drs), rc=5))
        lda_back(kb, drs, rows)
        sd += -(-(rows * drs + 64) // 128) * 128
    if hold:
        kb, rows = 8, 2 * R
        drs = I.qrow_bytes(kb, D) + 64
        prog.append(I.qact(0, rows, 0, kb, srs, store=(sd - (QB - 0x40), drs), rc=5))
        prog.append(I.li(6, sd))
        prog.append(I.lda(0, rows, kb, kb, drs, ra=6))
        mms(0, kb, rows)
        mms(kb, kb, rows)
        sd += rows * drs
    assert sd < cfg.DRAM_BYTES and out < cfg.TMEM_WORDS
    prog.append(I.halt())
    return cfg, prog, dram, pairs


def _spans(trace):
    """Each instruction's start and end cycles by its pc, from the sequencer's trace (rtlsim
    trace=True: "T0 D" dispatches a pc into a window slot, "T0 S" / "T0 E" name the slot; a
    cycle's E lines free slots before its D lines fill them, and S follows D)."""
    import re
    pc_of, S, E = {}, {}, {}
    ev = [(int(c), "EDS".index(k), k, int(s), pc) for k, c, s, pc in
          re.findall(r"^T0 ([DSE]) c=(\d+) s=(\d+)(?: pc=(\d+))?", trace, re.M)]
    for c, _, k, s, pc in sorted(ev, key=lambda e: e[:2]):
        if k == "D":
            pc_of[s] = int(pc)
        elif s in pc_of:
            (S if k == "S" else E).setdefault(pc_of[s], c)
    return S, E


def _qonce_ua(pw, ldq=1, wmx=False):
    from opentpu import target as T
    ua = {"SEQ_WC": 1, "WIN": 16, "QUANT_SWW": 4} | ({"QUANT_PW": pw} if pw > 1 else {}) \
        | ({"DMA_LDQ": ldq} if ldq > 1 else {})
    if wmx:
        ua |= {k: v for k, v in T.sim_uarch("bcu1525-x2:wmx").items()
               if k.startswith("TMEM_") or k in ("RPB", "WPB", "QUANT_R1", "DMA_LDQ")}
    return ua


@pytest.mark.parametrize("D,lanes,pw,axi,ldq", [(32, 8, 2, False, 1), (128, 8, 2, False, 1),
                                                (128, 8, 2, False, 2), (128, 8, 2, True, 2),
                                                (32, 16, 1, False, 2)])
def test_qonce_rtl(have_verilator, D, lanes, pw, axi, ldq):
    """QONCE on the RTL (otpu_quant QONCE: QACT ST's bytes and records through the SW port;
    otpu_dma LDA: the rows back into ACT RAM through its port, the quantizer's writes first) at
    QUANT_SWW 4 (4 x SWW = LANES x QUANT_PW): _qonce_prog's passes (and at PW 1, LANES 16 the
    record FIFO's hold), on the behavioural DRAM and on the native adapter (stalls), with one
    DMA command in flight or two (DMA_LDQ 2: x2:wmx's; an LDA never under another): DRAM and
    TMEM the ISA simulator's bit for bit, with the sequencer's wait entries (SEQ_WC: QACT ST's
    DRAM rows and LDA's footprints). At D 128 on the behavioural DRAM the first pass's all-rows
    LDA runs beside the QACT after it (their cycles overlap: the ACT RAM's two writers; later
    passes' QACTs wait on the MMs before them, which read their blocks)."""
    from opentpu import rtlsim
    cfg, prog, dram, pairs = _qonce_prog(D, lanes,
                                         np.random.default_rng(D + lanes + 7 * axi + 3 * ldq),
                                         hold=pw == 1)
    m = Machine(cfg, [prog], [dram.copy()]).run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [dram.copy()], uarch=_qonce_ua(pw, ldq),
                                  trace=D == 128 and not axi,
                                  **(dict(axi=True, stall=30) if axi else {}))
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first {bad[:6]}"
    bad = np.nonzero(drams[0] != m.slices[0].dram)[0]
    assert len(bad) == 0, (f"{len(bad)} DRAM bytes differ, first at {bad[:6]}: RTL "
                           f"{drams[0][bad[:6]]} ISA {m.slices[0].dram[bad[:6]]}")
    if D == 128 and not axi:
        S, E = _spans(st["trace"])
        i = pairs[0]
        assert S[i + 1] < E[i] and S[i] < E[i + 1], (S[i], E[i], S[i + 1], E[i + 1])


@pytest.mark.parametrize("seed", [1, 2])
def test_qonce_power_up_rtl(have_verilator, monkeypatch, seed):
    """_qonce_prog at bcu1525-x2:wmx's quantizer and TMEM (D 128, LANES 8, PW 2, SWW 4, R1, the
    one pumped copy, DMA_LDQ 2) from a random power-up (OTPU_X_INITIAL=unique): the record
    FIFO's pointers, the DMA's ACT queue and LDA's row state hold nothing past the reset. DRAM is the ISA's, and
    TMEM at every word the program writes."""
    from opentpu import rtlsim
    monkeypatch.setenv("OTPU_X_INITIAL", "unique")
    monkeypatch.setenv("OTPU_X_SEED", str(seed))
    cfg, prog, dram, _ = _qonce_prog(128, 8, np.random.default_rng(50 + seed))
    m = Machine(cfg, [prog], [dram.copy()]).run()
    m1 = Machine(cfg, [prog], [dram.copy()])
    m1.slices[0].tmem[:] = 0xFFFFFFFF
    m1.run()
    wr = m.slices[0].tmem == m1.slices[0].tmem
    drams, tmems, _ = rtlsim.run(cfg, [prog], [dram.copy()], uarch=_qonce_ua(2, wmx=True))
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0][wr], m.slices[0].tmem[wr])
