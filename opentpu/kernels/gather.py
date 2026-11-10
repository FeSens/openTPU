"""Rows of a quantized table (int8 or 4-bit, an MXU format: opentpu/quant.py) gathered and
dequantized on the device, at a row index known only at run time (the token id): an embedding
straight from a tied LM head, or Gemma 4's per-layer embeddings, with no fp32 copy of the table
and no host work per token.

The MXU does it. A table row is streamed as a matrix of its D-blocks (one block, or for 4-bit
one D-byte chunk of two blocks, per streamed row), against a stationary operand of one-hot
rows: row j of one-hot block k is 1 at element M k + j (M = MCOLS), so one MM gives element
M k + j of every block, dequantized by the MXU's own arithmetic (the one-hot quantizes to 127
with Sa = f32(1/127), every offset 0); D / M MMs give every element (for 4-bit, twice as many,
one for each block of a chunk: the one-hot blocks are interleaved with zero blocks, [Z, O_0, Z,
O_1, ..., O_{D/M-1}, Z], and [O_k, Z] reads a chunk's first block, [Z, O_k] its second). The
operand is the constant `onehot(D, M, fmt)` from DRAM, quantized once (operand; where it does
not fit ACT RAM or TMEM whole, each MM's blocks on their own).

A streamed row's scales are its own, so a table keeps them per streamed row, its gather scales
(quant.gather_scales, next to the scale rows an MM over the table reads, or alone): int8 an
fp32 per block, Sw 2^-dw (MM RSC); 4-bit a scale row per chunk (Sw and the chunk's two blocks'
offsets and multipliers). An element is then rne(rne(i2f(127 w m 2^sh) Sw) (f32(1/127) 2^-W)),
sh = W - dw (MM's window; a block with sh < 0 gives 0).

Under GALL at WG > 1 (every MM in groups, compiler.gall_wg) a row's blocks are streamed as a
grouped matrix [K / D, D], its rows the blocks: int8 in groups of WG blocks, one chunk a group,
the row's bytes as they are, its rows padded to whole WG x D chunks and its gather scales to
whole areas of WG words (MM RSC's group areas); 4-bit in groups of 2 WG blocks, one chunk a
group, lane l blocks 2l and 2l + 1 (quant.group_rows), each block's scale entry its chunk's Sw
and its own word (the group's scale area, quant.group_scales). `table_bytes`, `pack_table`;
records the same (`pack_records` wg). With the channel hash each chunk's lanes are permuted by
its address as a grouped matrix's. A block is then one streamed row of one block, int8's
shape, so every format's gathers take int8's one-hot operand (onehot wg) and its D / M MMs
(4-bit half its natural 2 D / M), and the values are the natural gathers', bit for bit.

  gather_row     a row of a row-major table (the LM head): the MMs write the elements
                 transposed, [D, blocks], and one VOP per block copies a column to the row
  gather_record  a row of a table stored for gathering (`pack_records`): the elements are
                 placed in the blocks so that the MMs write the row in order, no copies
                 (Gemma 4's PLE table: 70 blocks, 32 MMs)

dequant_row / dequant_records give the same values on the host, bit for bit (the host-written
inputs of chunked prefill equal the device's gathers).
"""
from __future__ import annotations

import numpy as np

from .. import fp32 as F
from .. import isa as I
from .. import language as ol
from .. import qcache as QC
from .. import quant as Q
from ..compiler import Affine, QTensor


# =============================================================================== the operand
def onehot(D: int, M: int, fmt: str, wg: int = 1) -> np.ndarray:
    """The one-hot stationary operand [M, nb * D] fp32: int8, D / M blocks, row j of block k 1
    at element M k + j; 4-bit, the same blocks interleaved with zero blocks, Z O_0 Z O_1 ...
    O_{D/M-1} Z (O_k is block 2k + 1); with the gathers in groups (wg > 1) int8's for every
    format (a streamed row is one block)."""
    n = D // M
    four = fmt != "int8" and wg == 1
    nb = 2 * n + 1 if four else n
    a = np.zeros((M, nb * D), np.float32)
    for k in range(n):
        b = 2 * k + 1 if four else k
        for j in range(M):
            a[j, b * D + M * k + j] = 1.0
    return a


def onehot_blocks(D: int, M: int, fmt: str, wg: int = 1) -> int:
    return onehot(D, M, fmt, wg).shape[1] // D


class _Parts:
    """The one-hot operand where it does not fit whole (D / M blocks, 2 D / M + 1 for 4-bit,
    against ACT_BLOCKS; or its tile, M x the blocks, against TMEM: a small configuration):
    each MM's blocks loaded and quantized on their own (blocks(k, kb): M rows of kb blocks
    from DRAM, a QACT, the operand of one MM)."""

    def __init__(self, desc):
        self.desc, self.rows = desc, desc.shape[0]

    @property
    def src(self):                  # (oh.src.rows: M, as a quantized operand's)
        return self

    def blocks(self, k: int, kb: int = 1):
        D = ol.block_size()
        return ol.quantize(ol.load(self.desc[:, k * D:(k + kb) * D]))


def operand(desc):
    """The gathers' one-hot operand (oh) from its DRAM descriptor (the image's onehot(D, M,
    fmt)): loaded and quantized into ACT RAM once where the configuration holds it whole (its
    blocks in ACT RAM, its tile in at most 3/4 of TMEM, leaving the gather's own tiles room:
    the card's largest, 4-bit at D = 128 and 4 columns, is 51%), else _Parts. The choice is
    the configuration's, not the trace's: a whole operand that finds TMEM full fails the
    compile as it always did (the generate loop's split form, a prefill run's fewer rows)."""
    M, cols = desc.shape
    D = ol.block_size()
    n = cols // D
    if n <= ol.act_blocks() and 4 * M * cols <= 3 * ol.tmem_words():
        if n == D // M:
            return _made(M, D, n)
        return ol.quantize(ol.load(desc))
    return _Parts(desc)


def _made(M: int, D: int, n: int):
    """int8's operand (n = D / M one-hot blocks) made in TMEM, nothing loaded: row j is 1 at
    columns p k + j (p = D + M, k < n), so read at a row stride of p - 1 (row j, column c is
    word j (p - 1) + c) it is a run of words whose ones are at the multiples of p, n + M - 1 of
    them: one FILL of zeros, one FILL of ones down a column at stride p, and the QACT of the
    whole operand from rows that overlap (the bytes and scales of the image's onehot loaded:
    the card's 3,128 words written against 16,384 moved)."""
    p = D + M
    buf = ol.empty([n + M - 1, p], dense=True)
    buf.set(0.0)
    buf[:, 0:1].set(1.0)
    return ol.quantize(buf.rows_at(M, n * D, p - 1))


def table_bytes(k: int, fmt: str, D: int, wg: int = 1) -> tuple:
    """(row stride, gather scale stride) of a table [., k] stored for gathering on a machine
    whose MMs are in groups of wg lanes (language.gall_wg): row_bytes and gather_scale_bytes
    (wg = 1); int8 those padded to whole wg x D chunks and to areas of wg fp32 words; 4-bit the
    row's blocks in groups of 2 wg, a wg x D chunk and a scale area (_block_area) each."""
    rb, sb = Q.row_bytes(k, fmt, D), Q.gather_scale_bytes(k, fmt, D)
    if wg == 1:
        return rb, sb
    if fmt == "int8":
        return -(-rb // (wg * D)) * wg * D, -(-sb // (4 * wg)) * 4 * wg
    G = -(-(k // D) // Q.grp_r(fmt, wg))
    return G * wg * D, G * _block_area(fmt, D, wg)


def _block_area(fmt: str, D: int, wg: int) -> int:
    """The scale area of a group of blocks streamed one a row (wg > 1): int8 its wg fp32 Sw
    (MM RSC), 4-bit grp_scale_bytes of one block (the 2 wg Sw words, then their words)."""
    return 4 * wg if fmt == "int8" else Q.grp_scale_bytes(D, fmt, D, wg)


def _group_blocks(codes: np.ndarray, sw: np.ndarray, word: np.ndarray, fmt: str, D: int,
                  wg: int) -> tuple:
    """4-bit blocks as the rows of a grouped matrix (groups of 2 wg blocks): codes [n, nb, D/2]
    (a block's bytes, quant.pack4's), sw [n, nb, 4] (its Sw's bytes), word [n, nb, sb] (its scale
    data, quant.SCALE_DATA) -> (data [n, G wg D], scale areas [n, G _block_area]), G the groups
    of each of the n rows, the padding blocks 0."""
    n, nb = codes.shape[:2]
    R = Q.grp_r(fmt, wg)
    G = -(-nb // R)
    br = np.zeros((n, G * R, D), np.uint8)
    br[:, :nb, :D // 2] = codes
    h, sb = Q.swp(fmt, 1), Q.SCALE_DATA[fmt]
    sr = np.zeros((n, G * R, Q.scale_bytes(D, fmt, D)), np.uint8)
    sr[:, :nb, :4] = sw
    sr[:, :nb, h:h + sb] = word
    return (Q.group_rows(br.reshape(-1, D), wg, D, fmt, D).reshape(n, -1),
            Q.group_scales(sr.reshape(n * G * R, -1), fmt, D, wg, D).reshape(n, -1))


def table(data, gscale, n: int, k: int, fmt: str, D: int, wg: int = 1) -> QTensor:
    """The descriptor of a row-major table [n, k] in MXU format fmt to gather from: its element
    rows (quant.quantize_mxu's) at `data` and their gather scales (quant.gather_scales) at
    `gscale`, strides table_bytes(k, fmt, D, wg)."""
    rs, srs = table_bytes(k, fmt, D, wg)
    return QTensor(Affine.of(data), Affine.of(gscale), (n, k), rs, srs, D, wf=Q.mxu_wf(fmt))


def pack_table(q: np.ndarray, gs: np.ndarray, fmt: str, D: int, wg: int = 1,
               at: int | None = None) -> tuple:
    """A table's element rows [n, row_bytes] and gather scales (quant.gather_scales) as `table`
    reads them at wg: (data [n, rs], scales [n, srs]) uint8, zeros past each row (4-bit at wg >
    1: each row's blocks grouped, _group_blocks); `at`: the data's DRAM address on a machine
    with the channel hash (Config.CHASH), each chunk's lanes as stored there (quant.hash_rows)."""
    q, gs = np.asarray(q).view(np.uint8), np.asarray(gs).view(np.uint8)
    n, k = len(q), q.shape[1] if fmt == "int8" else 2 * q.shape[1]
    rs, srs = table_bytes(k, fmt, D, wg)
    if wg > 1 and fmt != "int8":
        nb, cs = k // D, _chunk_srs(fmt, D)
        g = gs.reshape(n, -1, cs)
        b = np.arange(nb)
        h, sb = Q.swp(fmt, 1), Q.SCALE_DATA[fmt]
        word = g[:, (b // 2)[:, None], h + (b % 2)[:, None] * sb + np.arange(sb)[None, :]]
        d, s = _group_blocks(q[:, :nb * D // 2].reshape(n, nb, D // 2), g[:, b // 2, :4], word,
                             fmt, D, wg)
    else:
        d, s = np.zeros((n, rs), np.uint8), np.zeros((n, srs), np.uint8)
        d[:, :q.shape[1]], s[:, :gs.reshape(n, -1).shape[1]] = q, gs.reshape(n, -1)
    if wg > 1 and at is not None:
        d = Q.hash_rows(d, at, wg, D)
    return d, s


def _mms(oh, qv: QTensor, out, fmt: str, half: int = 0):
    """The one-hot MMs of a row viewed as blocks (int8: qv [nb, D]; 4-bit: chunks, qv [nc,
    2D]): element M k + j of streamed row n into out(k)[j, n] (4-bit: the chunks' first blocks
    into out(k, 0), their second into out(k, 1))."""
    D, M = qv.D, oh.src.rows
    for k in range(D // M):
        if fmt == "int8":
            ol.dot(oh.blocks(k), qv, out=out(k, 0))
        else:
            ol.dot(oh.blocks(2 * k + 1, 2), qv, out=out(k, 0))
            ol.dot(oh.blocks(2 * k, 2), qv, out=out(k, 1))


def _blocks(r: QTensor, nb: int, fmt: str, wg: int) -> QTensor:
    """A gathered row's nb blocks as the streamed matrix [nb, D], a block a row (int8; any format
    in groups at wg > 1): at wg 1 int8's D bytes and an fp32 Sw (MM RSC) a block, in groups
    (table_bytes, pack_records) a chunk and a scale area (_block_area) per group."""
    D = r.D
    if fmt == "int8":
        return QTensor(r.data, r.scale, (nb, D), wg * D, 4 * wg, D, wf=I.W8, rsc=True, grp=wg)
    return QTensor(r.data, r.scale, (nb, D), wg * D, _block_area(fmt, D, wg), D, wf=r.wf,
                   grp=Q.grp_r(fmt, wg))


def _chunk_srs(fmt: str, D: int) -> int:
    """A 4-bit chunk's gather scale row: its bytes and its stride."""
    return Q.gather_scale_bytes(2 * D, fmt, D)


# =============================================================================== kernels
def gather_row(oh, tab: QTensor, row, fmt: str):
    """Row `row` (an int or a run-time value) of a row-major table [N, K] (int8 or 4-bit, its
    gather scales: `table`), dequantized: an fp32 [K] tile. oh: quantize(onehot(D, M, fmt)) (a
    live stationary operand)."""
    D, K = tab.D, tab.shape[1]
    nb = K // D
    M = oh.src.rows
    wg = ol.gall_wg()
    if wg > 1 and (tab.rs % (wg * D) or tab.srs % _block_area(fmt, D, wg)):
        raise ValueError(f"a {fmt} table of strides {tab.rs}, {tab.srs}: gathers in groups of "
                         f"{wg} read rows of whole {wg} x D chunks and scale areas of "
                         f"{_block_area(fmt, D, wg)} bytes (table_bytes)")
    r = tab[row:row + 1, :] if tab.shape[0] > 1 else tab
    if fmt == "int8" or wg > 1:
        T = ol.empty([D, nb])                       # T[p, b]: element p of block b
        _mms(oh, _blocks(r, nb, fmt, wg), lambda k, h: T[M * k:M * k + M, :], "int8")
        Ts = [(T, b) for b in range(nb)]
    else:
        nc = -(-nb // 2)
        T0, T1 = ol.empty([D, nc]), ol.empty([D, nc])
        qv = QTensor(r.data, r.scale, (nc, 2 * D), D, _chunk_srs(fmt, D), D, wf=tab.wf)
        _mms(oh, qv, lambda k, h: (T0, T1)[h][M * k:M * k + M, :], fmt)
        Ts = [((T0, T1)[b % 2], b // 2) for b in range(nb)]
    out = ol.empty([K])
    for b, (T, c) in enumerate(Ts):                 # the transpose, a column per VOP
        out[b * D:(b + 1) * D].column().set(T[:, c:c + 1])
    return out


def gather_record(oh, tab: QTensor, row, fmt: str, S: int):
    """Row `row` of a table packed by pack_records (S blocks per record; tab.shape[1] = S * D
    elements, tab.scale the records' gather scales), dequantized in order: an fp32 [S * D]
    tile (the first n are the row)."""
    D, M = tab.D, oh.src.rows
    wg = ol.gall_wg()
    out = ol.empty([S * D])
    v = out.reshape(D, S)                           # v[p, c] = element p S + c
    r = tab[row:row + 1, :] if tab.shape[0] > 1 else tab
    if fmt == "int8" or wg > 1:                     # (in groups: the blocks in order, a row each)
        _mms(oh, _blocks(r, S, fmt, wg), lambda k, h: v[M * k:M * k + M, :], "int8")
    else:
        nc = S // 2
        qv = QTensor(r.data, r.scale, (nc, 2 * D), D, _chunk_srs(fmt, D), D, wf=tab.wf)
        _mms(oh, qv, lambda k, h: v[M * k:M * k + M, h * nc:(h + 1) * nc], fmt)
    return out


# =============================================================================== host side
def record_blocks(n: int, fmt: str) -> int:
    """S, the blocks of a record of n elements: at least n / D (the caller rounds n up to
    blocks), with the MM's output row stride S safe for the TMEM banks: M consecutive rows S
    apart fall in distinct banks when S is odd or S = 2 mod 4 and M <= LANES / 2 (every
    configuration: LANES >= 8, M <= 4 on the board, 8 in the design point). 4-bit records hold
    S / 2 chunks: S = 2 mod 4."""
    s = n
    if fmt == "int8":
        while not (s % 2 == 1 or s % 4 == 2):
            s += 1
    else:
        s += s % 2
        if s % 4 == 0:
            s += 2
    return s


def record_data(S: int, fmt: str, D: int, wg: int = 1) -> int:
    """Bytes of a record's elements (its gather scales follow them); in groups (wg > 1) the S
    blocks' groups, a wg x D chunk each."""
    if wg > 1:
        return -(-S // Q.grp_r(fmt, wg)) * wg * D
    return S * D if fmt == "int8" else S * D // 2


def _record_scales(S: int, fmt: str, D: int, wg: int = 1) -> int:
    """Bytes of a record's gather scales: int8 an fp32 per block, 4-bit a scale row per chunk;
    in groups a scale area per group (_block_area)."""
    if wg > 1:
        return -(-S // Q.grp_r(fmt, wg)) * _block_area(fmt, D, wg)
    return 4 * S if fmt == "int8" else S // 2 * _chunk_srs(fmt, D)


def record_bytes(S: int, fmt: str, D: int, wg: int = 1) -> int:
    """Bytes of one record: the S blocks' data, then their gather scales, padded to D bytes (in
    groups to whole wg x D chunks: every record's data at a chunk)."""
    u = D if wg == 1 else wg * D
    return -(-(record_data(S, fmt, D, wg) + _record_scales(S, fmt, D, wg)) // u) * u


def record_table(base, n: int, S: int, fmt: str, D: int, wg: int = 1) -> QTensor:
    """The descriptor of n records of S blocks at `base` (pack_records' layout at wg) to gather
    from."""
    rec = record_bytes(S, fmt, D, wg)
    return QTensor(Affine.of(base), Affine.of(base) + record_data(S, fmt, D, wg), (n, S * D),
                   rec, rec, D, wf=Q.mxu_wf(fmt))


def pack_records(rows: np.ndarray, fmt: str, D: int, S: int, wg: int = 1) -> np.ndarray:
    """Records [T, record_bytes] (uint8) of fp32 rows [T, n], n <= S D: element i of a row is
    element i // S of block i % S (int8: block c is the c-th D bytes; 4-bit: blocks c and S/2
    + c are the two halves of chunk c), zeros past n; the gather scales follow the data. Each
    block is quantized on its own (int8: a row of one block, its scale Sw), a 4-bit chunk's two
    blocks as one row (one Sw, their offsets and multipliers: the chunk's scale row). In groups
    (wg > 1) the same quantization, the blocks in order as a grouped matrix's rows (int8: the
    same bytes, padded to whole groups; 4-bit _group_blocks, a block's Sw its chunk's); with
    the channel hash the caller permutes each record's lanes by its address (hash_records)."""
    T, n = rows.shape
    x = np.zeros((T, S * D), np.float32)
    x[:, :n] = rows
    blk = x.reshape(T, D, S).transpose(0, 2, 1).reshape(T * S, D)     # [T*S, D]: block c
    rec = np.zeros((T, record_bytes(S, fmt, D, wg)), np.uint8)
    nd = record_data(S, fmt, D, wg)
    if fmt == "int8":
        q, s = QC.quantize_rows(blk, fmt, D)                          # [T*S, D], [T*S, 8]
        rec[:, :S * D] = q.reshape(T, S * D)
        rec[:, nd:nd + 4 * S] = Q.gather_scales(s, fmt, D, D).view(np.uint8).reshape(T, 4 * S)
        return rec
    nc = S // 2
    pair = blk.reshape(T, 2, nc, D).transpose(0, 2, 1, 3).reshape(T * nc, 2 * D)  # chunk rows
    b, s = QC.quantize_rows(pair, fmt, D)                             # [T*nc, D], [T*nc, srs]
    cs = _chunk_srs(fmt, D)
    g = Q.gather_scales(s, fmt, 2 * D, D).reshape(T, nc, cs)
    if wg > 1:                      # block c: half c // nc of chunk c % nc
        c = np.arange(S)
        h, sb = Q.swp(fmt, 1), Q.SCALE_DATA[fmt]
        hb = (c // nc)[:, None]
        codes = b.reshape(T, nc, D)[:, (c % nc)[:, None], hb * (D // 2) + np.arange(D // 2)]
        word = g[:, (c % nc)[:, None], h + hb * sb + np.arange(sb)[None, :]]
        d, sc = _group_blocks(codes, g[:, c % nc, :4], word, fmt, D, wg)
        rec[:, :nd], rec[:, nd:nd + sc.shape[1]] = d, sc
        return rec
    rec[:, :nd] = b.reshape(T, nd)
    rec[:, nd:nd + nc * cs] = g.reshape(T, nc * cs)
    return rec


def hash_records(recs: np.ndarray, at: int, S: int, fmt: str, D: int, wg: int) -> np.ndarray:
    """Records [T, record_bytes] (pack_records' at wg) as a machine with the channel hash stores
    them from DRAM address `at` on: each record's data lanes permuted by its chunks' addresses
    (quant.hash_rows), its scales as they are; at wg = 1 the records as given."""
    if wg == 1:
        return recs
    rec, nd = record_bytes(S, fmt, D, wg), record_data(S, fmt, D, wg)
    out = np.array(recs, np.uint8).reshape(-1, rec)
    for t in range(len(out)):
        out[t, :nd] = Q.hash_rows(out[t, :nd], at + t * rec, wg, D)
    return out


def _onehot_q(D: int):
    """The quantized one-hot: (q, Sa) as QACT makes them (every offset 0)."""
    x = np.zeros((1, D), np.float32)
    x[0, 0] = 1.0
    q, s, _ = F.quantize_mx(x, D // Q.NSUB)
    return int(q[0, 0]), np.float32(s[0])


def dequant_blocks(w, Sw, dw, m, fmt: str) -> np.ndarray:
    """The one-hot MMs' values of blocks [N, D]: their integer elements w [N, D] (int8, or the
    4-bit codes decoded) and per block its Sw [N] and per group (G of them) dw and m [N, G].
    fp32 [N, D], as the MXU computes: rne(rne(i2f(127 w (m << sh)) Sw) (Sa 2^-W)), sh = W - dw
    (sh < 0: 0)."""
    q1, s1 = _onehot_q(np.asarray(w).shape[1])
    W = I.MX_W[Q.mxu_wf(fmt)]
    w = np.asarray(w, np.int64)
    N, D = w.shape
    G = np.asarray(dw).shape[1]
    sh = W - np.asarray(dw, np.int64)
    B = np.where(sh >= 0, np.asarray(m, np.int64) << np.maximum(sh, 0), 0)
    acc = q1 * w * np.repeat(B, D // G, axis=1)
    return F.mul(F.mul(F.i2f(acc), np.asarray(Sw, np.float32)[:, None]),
                 F.mul(s1, np.float32(2.0 ** -W)))


def dequant_gathered(rows: np.ndarray, gsc: np.ndarray, fmt: str, K: int,
                     D: int = 128) -> np.ndarray:
    """The gathered values of table rows [N, K] (fp32): their element rows (row_bytes) and
    gather scales (quant.gather_scales: int8 fp32 [N, KB], 4-bit uint8 [N, chunks, srs])."""
    rows = np.asarray(rows, np.uint8)
    N, KB = rows.shape[0], K // D
    w = Q.elements(rows, fmt, K).reshape(N * KB, D)
    if fmt == "int8":
        Sw = np.asarray(gsc).reshape(N, -1)[:, :KB].view(np.float32) \
            if np.asarray(gsc).dtype == np.uint8 else np.asarray(gsc, np.float32)
        Sw = Sw.reshape(N * KB)
        zero = np.zeros((N * KB, Q.NSUB), np.int64)
        return dequant_blocks(w, Sw, zero, zero + 1, fmt).reshape(N, K)
    g = np.asarray(gsc, np.uint8).reshape(N, -1, _chunk_srs(fmt, D))   # [N, nc, srs]
    nc = g.shape[1]
    Sw, dw, m = Q.unpack_scales(g.reshape(N * nc, -1), fmt, 2)        # [N*nc], [N*nc, 2, G]
    Sw = np.repeat(Sw, 2)[:N * nc * 2].reshape(N, 2 * nc)[:, :KB].reshape(-1)
    dw = dw.reshape(N, 2 * nc, -1)[:, :KB].reshape(N * KB, -1)
    m = m.reshape(N, 2 * nc, -1)[:, :KB].reshape(N * KB, -1)
    return dequant_blocks(w, Sw, dw, m, fmt).reshape(N, K)


def dequant_row(qrow: np.ndarray, gsrow: np.ndarray, fmt: str, D: int, K: int | None = None
                ) -> np.ndarray:
    """gather_row's values of one table row: its MXU bytes (quant.quantize_mxu's row) and its
    gather scales (quant.gather_scales' row)."""
    qrow = np.asarray(qrow, np.uint8)
    if K is None:
        K = len(qrow) if fmt == "int8" else 2 * len(qrow)
    return dequant_gathered(qrow[None], np.asarray(gsrow)[None], fmt, K, D)[0]


def gathered(a: np.ndarray, fmt: str, D: int = 128) -> np.ndarray:
    """The values gather_row gives for fp32 rows [n, K] of a table in MXU format fmt
    (quantize_mxu, gather_scales): [n, K] fp32, bit for bit."""
    a = np.atleast_2d(np.asarray(a, np.float32))
    q, s = Q.quantize_mxu(a, fmt, D)
    return dequant_gathered(q, Q.gather_scales(s, fmt, a.shape[1], D), fmt, a.shape[1], D)


def dequant_records(rec: np.ndarray, fmt: str, D: int, S: int, wg: int = 1) -> np.ndarray:
    """gather_record's values of records [T, record_bytes] (pack_records' at wg, unhashed):
    fp32 [T, S * D]."""
    T = rec.shape[0]
    nd = record_data(S, fmt, D, wg)
    if fmt == "int8":
        q = rec[:, :S * D].reshape(T * S, D)
        s = rec[:, nd:nd + 4 * S].copy().view(np.float32).reshape(T * S, 1)
        v = dequant_gathered(q, s, fmt, D, D).reshape(T, S, D)
    elif wg > 1:                    # the grouped blocks, a block a row
        G = -(-S // Q.grp_r(fmt, wg))
        w = Q.elements(Q.ungroup_rows(rec[:, :nd].reshape(T * G, wg * D), T * G * Q.grp_r(
            fmt, wg), wg, D, fmt), fmt, D)
        sr = Q.ungroup_scales(rec[:, nd:nd + G * _block_area(fmt, D, wg)].reshape(T * G, -1),
                              fmt, D, T * G * Q.grp_r(fmt, wg), wg, D)
        Sw, dw, m = Q.unpack_scales(sr, fmt, 1)
        v = dequant_blocks(w, Sw, dw[:, 0], m[:, 0], fmt).reshape(T, -1, D)[:, :S]
    else:
        nc = S // 2
        cs = _chunk_srs(fmt, D)
        b = rec[:, :nd].reshape(T * nc, D)                               # a chunk row each
        g = rec[:, nd:nd + nc * cs].reshape(T * nc, 1, cs)
        v = dequant_gathered(b, g, fmt, 2 * D, D).reshape(T, nc, 2, D)
        v = v.transpose(0, 2, 1, 3).reshape(T, S, D)                    # block c, S/2 + c
    return np.ascontiguousarray(v.transpose(0, 2, 1)).reshape(T, S * D)
