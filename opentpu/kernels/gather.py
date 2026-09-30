"""Rows of a quantized table (int8 or 4-bit, block-scaled: an MXU operand in DRAM) gathered and
dequantized on the device, at a row index known only at run time (the token id): an embedding
straight from a tied int8 LM head, or Gemma 4's per-layer embeddings, with no fp32 copy of the
table and no host work per token.

The MXU does it. A table row is streamed as a matrix of its D-blocks (one block, or for 4-bit
one D-byte chunk of two blocks, per streamed row), against a stationary operand of one-hot
rows: row j of one-hot block k is 1 at element M k + j (M = MCOLS), so one MM gives element
M k + j of every block, dequantized by the MXU's own arithmetic, t = (i2f(127 w) * ws) * (1 /
127) (the one-hot quantizes to 127 with scale f32(1/127)); D / M MMs give every element (for
4-bit, twice as many, one for each block of a chunk: the one-hot blocks are interleaved with
zero blocks, [Z, O_0, Z, O_1, ..., O_{D/M-1}, Z], and [O_k, Z] reads a chunk's first block,
[Z, O_k] its second). The operand is the constant `onehot(D, M, fmt)` from DRAM, quantized once.

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
from .. import quant as Q
from ..compiler import QTensor


# =============================================================================== the operand
def onehot(D: int, M: int, fmt: str) -> np.ndarray:
    """The one-hot stationary operand [M, nb * D] fp32: int8, D / M blocks, row j of block k 1
    at element M k + j; 4-bit, the same blocks interleaved with zero blocks, Z O_0 Z O_1 ...
    O_{D/M-1} Z (O_k is block 2k + 1)."""
    n = D // M
    four = fmt != "int8"
    nb = 2 * n + 1 if four else n
    a = np.zeros((M, nb * D), np.float32)
    for k in range(n):
        b = 2 * k + 1 if four else k
        for j in range(M):
            a[j, b * D + M * k + j] = 1.0
    return a


def onehot_blocks(D: int, M: int, fmt: str) -> int:
    return onehot(D, M, fmt).shape[1] // D


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


def _view(table: QTensor, row, fmt: str, n: int, rs_scale: int) -> QTensor:
    """Row `row` of `table` as the MXU streams its blocks: n blocks of D (int8) or n chunks of
    two blocks (4-bit), each a streamed row D bytes apart, with its scale words."""
    D = table.D
    r = table[row:row + 1, :] if table.shape[0] > 1 else table
    if fmt == "int8":
        return QTensor(r.data, r.scale, (n, D), D, 4, D, wf=I.W8)
    return QTensor(r.data, r.scale, (n, 2 * D), D, 8, D, wf=table.wf)


# =============================================================================== kernels
def gather_row(oh, table: QTensor, row, fmt: str):
    """Row `row` (an int or a run-time value) of a row-major table [N, K] of MXU rows (int8 or
    4-bit, qwen3's LM head layout), dequantized: an fp32 [K] tile. oh: quantize(onehot(D, M,
    fmt)) (a live stationary operand)."""
    D, K = table.D, table.shape[1]
    nb = K // D
    M = oh.src.rows
    if fmt == "int8":
        T = ol.empty([D, nb])                       # T[p, b]: element p of block b
        _mms(oh, _view(table, row, fmt, nb, 4), lambda k, h: T[M * k:M * k + M, :], fmt)
        Ts = [(T, b) for b in range(nb)]
    else:
        nc = -(-nb // 2)
        T0, T1 = ol.empty([D, nc]), ol.empty([D, nc])
        _mms(oh, _view(table, row, fmt, nc, 8),
             lambda k, h: (T0, T1)[h][M * k:M * k + M, :], fmt)
        Ts = [((T0, T1)[b % 2], b // 2) for b in range(nb)]
    out = ol.empty([K])
    for b, (T, c) in enumerate(Ts):                 # the transpose, a column per VOP
        out[b * D:(b + 1) * D].column().set(T[:, c:c + 1])
    return out


def gather_record(oh, table: QTensor, row, fmt: str, S: int):
    """Row `row` of a table packed by pack_records (S blocks per record; table.shape[1] =
    S * D elements), dequantized in order: an fp32 [S * D] tile (the first n are the row)."""
    D, M = table.D, oh.src.rows
    out = ol.empty([S * D])
    v = out.reshape(D, S)                           # v[p, c] = element p S + c
    if fmt == "int8":
        _mms(oh, _view(table, row, fmt, S, 4), lambda k, h: v[M * k:M * k + M, :], fmt)
    else:
        nc = S // 2
        _mms(oh, _view(table, row, fmt, nc, 8),
             lambda k, h: v[M * k:M * k + M, h * nc:(h + 1) * nc], fmt)
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


def record_bytes(S: int, fmt: str, D: int) -> int:
    """Bytes of one record: the S blocks' data, then their scale words, padded to D bytes."""
    data = S * D if fmt == "int8" else S * D // 2
    return data + -(-4 * S // D) * D


def pack_records(rows: np.ndarray, fmt: str, D: int, S: int) -> np.ndarray:
    """Records [T, record_bytes] (uint8) of fp32 rows [T, n], n <= S D: element i of a row is
    element i // S of block i % S (int8: block c is the c-th D bytes; 4-bit: blocks c and S/2
    + c are the two halves of chunk c), zeros past n; the scale words follow the data (block
    order; 4-bit: chunk order, two words per chunk)."""
    T, n = rows.shape
    x = np.zeros((T, S * D), np.float32)
    x[:, :n] = rows
    blk = x.reshape(T, D, S).transpose(0, 2, 1).reshape(T * S, D)     # [T*S, D]: block c
    rec = np.zeros((T, record_bytes(S, fmt, D)), np.uint8)
    if fmt == "int8":
        q, s = Q.quantize_mxu(blk, fmt, D)                            # [T*S, D], [T*S, 1]
        rec[:, :S * D] = q.reshape(T, S * D)
        rec[:, S * D:S * D + 4 * S] = s.reshape(T, S).view(np.uint8).reshape(T, 4 * S)
        return rec
    b, w = Q.quantize_mxu(blk, fmt, D)                                # [T*S, D/2 padded to D]
    b = b[:, :D // 2].reshape(T, 2, S // 2, D // 2)                   # (half, chunk)
    rec[:, :S * D // 2] = b.transpose(0, 2, 1, 3).reshape(T, S * D // 2)
    w = w.reshape(T, 2, S // 2).transpose(0, 2, 1)                     # chunk, then half
    rec[:, S * D // 2:S * D // 2 + 4 * S] = np.ascontiguousarray(w).view(np.uint8).reshape(
        T, 4 * S)
    return rec


def _onehot_q(D: int):
    """The quantized one-hot: (q, s) as QACT makes them."""
    x = np.zeros(D, np.float32)
    x[0] = 1.0
    q, s = F.quantize(x)
    return int(q[0]), np.float32(s)


def dequant_blocks(q, ws, fmt: str, D: int) -> np.ndarray:
    """The one-hot MMs' values of blocks [N, D]: int8 q (int8) with fp32 scales ws [N]; 4-bit
    q packed [N, D/2] with scale words ws [N] (uint32). fp32 [N, D], as the MXU computes."""
    q1, s1 = _onehot_q(D)
    if fmt == "int8":
        isum = q1 * np.asarray(q, np.int64)
        scale = np.asarray(ws, np.float32)[:, None]
    else:
        w = Q.DEC[fmt][Q.unpack4(np.asarray(q, np.uint8))].astype(np.int64)      # [N, D]
        words = np.asarray(ws, np.uint32)
        mb = np.stack([(words >> np.uint32(16 + 4 * b)) & np.uint32(15)
                       for b in range(Q.NSUB)], -1).astype(np.int64)             # [N, NSUB]
        isum = q1 * w * np.repeat(mb, D // Q.NSUB, axis=1)
        scale = F.ftz((words << np.uint32(16)).view(np.float32))[:, None]
    return F.mul(F.mul(F.i2f(isum), scale), s1)


def dequant_row(qrow: np.ndarray, srow: np.ndarray, fmt: str, D: int) -> np.ndarray:
    """gather_row's values of one table row: its MXU bytes (quant.quantize_mxu's row) and
    scale words (fp32 bits for int8)."""
    if fmt == "int8":
        q = np.asarray(qrow).view(np.int8).reshape(-1, D)
        return dequant_blocks(q, np.asarray(srow).view(np.float32), fmt, D).reshape(-1)
    K = 4 * len(srow) * D // 4
    q = np.asarray(qrow, np.uint8)[:K // 2].reshape(-1, D // 2)
    return dequant_blocks(q, np.asarray(srow).view(np.uint32), fmt, D).reshape(-1)


def dequant_records(rec: np.ndarray, fmt: str, D: int, S: int) -> np.ndarray:
    """gather_record's values of records [T, record_bytes]: fp32 [T, S * D]."""
    T = rec.shape[0]
    if fmt == "int8":
        q = rec[:, :S * D].view(np.int8).reshape(T * S, D)
        s = rec[:, S * D:S * D + 4 * S].copy().view(np.float32).reshape(T * S)
        v = dequant_blocks(q, s, fmt, D).reshape(T, S, D)
    else:
        nc = S // 2
        b = rec[:, :S * D // 2].reshape(T, nc, 2, D // 2).transpose(0, 2, 1, 3)
        w = rec[:, S * D // 2:S * D // 2 + 4 * S].copy().view(np.uint32).reshape(T, nc, 2)
        v = dequant_blocks(b.reshape(T * S, D // 2), w.transpose(0, 2, 1).reshape(T * S),
                           fmt, D).reshape(T, S, D)
    return np.ascontiguousarray(v.transpose(0, 2, 1)).reshape(T, S * D)
