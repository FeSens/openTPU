"""tools/dataflow.py's ledger of one MM: the weight and scale bytes it streams from DRAM and its
port B chunks are the operand's layout (quant.row_bytes / grp_row_bytes, the Sw part and the
blocks' entries), for every weight format, at R = 1 and in weight groups, odd KB too (a group's
chunk is a K-block of its rows, docs/isa.md "Weight groups": no half-empty last chunk)."""
import dataclasses
import sys
from pathlib import Path

import pytest

from opentpu import Config, isa as I, quant as Q, target as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dataflow as DF  # noqa: E402

WA, SA, OUT = 1 << 16, 3 << 18, 1 << 12
FMT = {I.W8: "int8", I.W4I: "int4", I.W4F: "fp4", I.WNV: "nvfp4"}


@pytest.mark.parametrize("KB", [2, 3])
@pytest.mark.parametrize("wg", [1, 8])
@pytest.mark.parametrize("wf", [I.W8, I.W4I, I.W4F, I.WNV])
def test_mm_streams_its_layout(wf, wg, KB):
    D, N, M = 32, 37, 2
    fmt, R = FMT[wf], Q.grp_r(FMT[wf], wg)
    cfg = Config(S=1, D=D, MCOLS=8, ACT_BLOCKS=16, ACT_ROWS=16, PAIR=True, WG=wg,
                 DRAM_BYTES=1 << 21)
    rs = Q.grp_row_bytes(KB * D, fmt, D, wg)
    srs = Q.grp_scale_bytes(KB * D, fmt, D, wg) if R > 1 else Q.scale_bytes(KB * D, fmt, D)
    prog = [I.qact(0, M, 1, KB, KB * D),
            I.mm(WA, SA, OUT, N, KB, rs, N, M, 1, srs, wf=wf, grp=R), I.halt()]
    t = DF.ledger(cfg, [prog])["total"]
    G = -(-N // R)
    if R == 1:
        wb = N * Q.row_bytes(KB * D, fmt, D)
    else:
        wb = G * rs
        assert wb == G * R * KB * (D if wf == I.W8 else D // 2)
    assert t["dram_rd"]["weights"] == wb
    assert t["portb"] * D == wb
    assert t["dram_rd"]["scales"] == G * (Q.swp(fmt, R) + KB * R * Q.SCALE_DATA[fmt])
    assert t["mxu_adv"] == (N * KB if R == 1 else G * M * KB)


@pytest.mark.parametrize("tname", [None, "bcu1525-x2:b2"])
@pytest.mark.parametrize("KB", [2, 3])
def test_mm_spl_streams_its_parts(KB, tname):
    """MM SPL (SPLMM, docs/isa.md "Split terms"): srs w6[19:0] under MM_SPL and HLD (the fast MM
    and the ledger strip them, as isasim._mm; with the whole w6 the fast MM's scale reads ran out
    of DRAM, pf500 sim on x2:b2 11-05); a group streams a chunk from each part as isasim._mm
    reads them, the hi at sa + g rs and the lo HLD after it, and takes KB advances a pass (KB 3:
    hi twice): the fast MM's ledger (a zero image) the same as with isasim._mm's products."""
    N, M = 37, 2
    if tname:
        cfg = dataclasses.replace(T.config(tname), S=1, DRAM_BYTES=1 << 24)
    else:
        cfg = Config(S=1, D=32, MCOLS=8, ACT_BLOCKS=16, ACT_ROWS=16, WG=8, SPLMM=True,
                     DRAM_BYTES=1 << 21)
    D, R, hld = cfg.D, cfg.WG, 16 << 10
    prog = [I.qact(0, M, 1, KB - 1, (KB - 1) * D),
            I.mm(WA, SA, OUT, N, KB, R * D, N, M, 1, 4 * R, rsc=True, grp=R, hld=hld), I.halt()]
    assert prog[1].w[6] & I.MM_SPL
    t, tf = (DF.ledger(cfg, [prog], fast=f)["total"] for f in (True, False))
    G = -(-N // R)
    for k in ("dram_rd", "portb", "porta", "mxu_adv"):
        assert t[k] == tf[k], k
    assert t["dram_rd"]["weights"] == t["portb"] * D == G * 2 * R * D
    assert t["dram_rd"]["scales"] == G * R * 4
    assert t["mxu_adv"] == G * M * KB


@pytest.mark.parametrize("row,half,off,pad,buffered", [
    (True, False, 0, 0, True), (True, False, 16, 32, True), (True, True, 0, 0, False),
    (False, False, 0, 0, False), (True, False, 4, 0, False), (True, False, 0, 4, False)])
def test_qstb_reads_qst_rows_once(monkeypatch, row, half, off, pad, buffered):
    """QUANT_QSTB (LedgerSlice.QSTB, otpu_quant QSTB): a ROW QST whose rows are 16-byte aligned
    (dst and row stride, element stride 1) goes through the row buffer and reads its rows from
    TMEM once, Q's chunk cycles half the two passes'; HALF, without ROW, or a dst or row stride
    off the 16 bytes: twice, as without QSTB."""
    D, R, KB = 128, 8, 2
    cfg = Config(S=1, D=D, MCOLS=4, LANES=8, DRAM_BYTES=1 << 20, SPLITQ=True)
    prog = [I.qst(0, WA + off, SA, R, KB, KB * D, KB * D + pad, 1, row=row, half=half),
            I.halt()]
    two = R * KB * D // cfg.LANES * 2
    q = {}
    for qstb in (False, True):
        monkeypatch.setattr(DF.LedgerSlice, "QSTB", qstb)
        q[qstb] = DF.ledger(cfg, [prog])["total"]["tmem_rd_chunks"]["Q"]
    assert q[False] == two
    assert q[True] == (two // 2 if buffered else two)



def test_qst_hl_one_read(monkeypatch):
    """QST HL (QSTHL, through the row buffer): a GRP ROW QST's hi and lo parts from one read,
    Q's chunk cycles a buffered QST's (pass 1's 2 KB D / QSW = pass 0's KB D / LANES at SWW 4),
    half a QST and a QST LO's with QSTB; its DRAM writes both parts'."""
    D, R, KB, rows = 128, 8, 2, 16
    cfg = Config(S=1, D=D, MCOLS=4, LANES=8, WG=R, GALL=True, DRAM_BYTES=1 << 22, SPLITQ=True,
                 QSTHL=True)
    hld = 64 << 10
    kw = dict(row=True, grp=R)
    two = [I.qst(0, WA, SA, rows, KB, KB * D, 0, 4 * R, **kw),
           I.qst(0, WA + hld, SA + hld, rows, KB, KB * D, 0, 4 * R, lo=True, **kw), I.halt()]
    one = [I.qst(0, WA, SA, rows, KB, KB * D, 0, 4 * R, hld=hld, **kw), I.halt()]
    monkeypatch.setattr(DF.LedgerSlice, "QSTB", True)
    t2, t1 = (DF.ledger(cfg, [p])["total"] for p in (two, one))
    assert t1["tmem_rd_chunks"]["Q"] * 2 == t2["tmem_rd_chunks"]["Q"] == 2 * rows * KB * D // 8
    assert t1["dram_wr"]["qst"] == t2["dram_wr"]["qst"]