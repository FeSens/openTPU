"""Weights stored in pieces of at most a slot's bytes (docs/minicpmv.md 4.2): a matrix's rows,
or its column parts, split into pieces, each piece its element rows then their scale rows (a
part's: its rows, then its scale rows, the parts one after the other), every block D-byte
aligned. A piece's address is static (the pieces laid out in the image) or read at run time
from a table of piece addresses in DRAM (the host stages pieces into expert slots and writes
their addresses there): `PQ` slices as a QTensor and gives the slice's QTensor at its piece's
address. With a table, a run loads its table words once (`bind`), and a slice's piece address
goes into a register by RLD (raw: the table holds the addresses' bits) before the slice's
instructions; a few pieces stay in registers (REGS), and every piece's register is given back
at the end of a row group's body (`flush`). The values a run computes are the same either way:
only where its MMs read their weights changes.

    lay = plan(n, k, fmt, D, unit, max_bytes)       # rows in pieces of whole `unit`s
    lay = plan(n, k, fmt, D, pw, max_bytes, parts=True)   # column parts of pw, whole parts
    pack(lay, q, s)                                 # each piece's bytes (quantize_mxu's rows)
    PQ(lay, bases=[...])  /  PQ(lay, table=Table(dram, n, bound), first=i)

In groups (wg > 1: GALL's machine, compiler.gall_wg) each block is a grouped matrix (docs/isa.md
"Weight groups": quant.grouped's data, then its scale areas), rows in whole groups, every block
P = wg x D aligned; on a machine with the channel hash a piece's data lanes are permuted by its
address (hash_piece, where it is placed: the resident pieces at build, staged ones by the host).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .. import quant as Q
from ..compiler import Affine, CompileError, DevVar, QTensor, Tensor, current

ALIGN = 128            # a block's alignment in a piece (D bytes: the MXU streams whole chunks)
REGS = 4               # piece addresses a body keeps in registers at once


def _up(n: int, a: int = ALIGN) -> int:
    return -(-n // a) * a


@dataclass(frozen=True)
class Layout:
    """A matrix [n, k] in format fmt, stored in pieces: groups[i] = (u0, u1), rows u0 .. u1 - 1
    (pw 0) or column parts u0 .. u1 - 1 of pw columns each; wg > 1: every block in groups for
    a machine of wg lanes."""
    n: int
    k: int
    fmt: str
    D: int
    pw: int
    groups: tuple
    wg: int = 1

    @property
    def rb(self) -> int:
        return Q.row_bytes(self.pw or self.k, self.fmt, self.D)

    @property
    def sb(self) -> int:
        return Q.scale_bytes(self.pw or self.k, self.fmt, self.D)

    @property
    def unit(self) -> int:
        """A block's alignment: D bytes (ALIGN), in groups a group chunk's P = wg x D."""
        return ALIGN if self.wg == 1 else max(ALIGN, self.wg * self.D)

    def block(self, rows: int) -> tuple:
        """(data bytes, scale bytes) of a block of `rows` rows, each aligned (in groups: its
        grouped data and scale areas, quant.grp_bytes)."""
        if self.wg == 1:
            return _up(rows * self.rb), _up(rows * self.sb)
        d, s = Q.grp_bytes(rows, self.pw or self.k, self.fmt, self.D, self.wg)
        return _up(d, self.unit), _up(s, self.unit)

    def piece_bytes(self, i: int) -> int:
        u0, u1 = self.groups[i]
        if self.pw:
            return (u1 - u0) * sum(self.block(self.n))
        return sum(self.block(u1 - u0))

    def offsets(self, i: int, j: int = 0) -> tuple:
        """(data, scale) offsets in piece i: of its rows (pw 0), or of its j-th part."""
        if self.pw:
            d, s = self.block(self.n)
            return j * (d + s), j * (d + s) + d
        u0, u1 = self.groups[i]
        return 0, self.block(u1 - u0)[0]

    def piece_of(self, u: int) -> int:
        for i, (u0, u1) in enumerate(self.groups):
            if u0 <= u < u1:
                return i
        raise CompileError(f"unit {u} is in no piece")

    @property
    def nbytes(self) -> int:
        return sum(self.piece_bytes(i) for i in range(len(self.groups)))

    def data_regions(self, i: int) -> list:
        """Piece i's element bytes: [(offset, bytes)], a block each (the scales after them)."""
        if self.pw:
            u0, u1 = self.groups[i]
            return [(self.offsets(i, j)[0], self.block(self.n)[0]) for j in range(u1 - u0)]
        u0, u1 = self.groups[i]
        return [(0, self.block(u1 - u0)[0])]


def plan(n: int, k: int, fmt: str, D: int, unit: int, max_bytes: int,
         parts: bool = False, wg: int = 1) -> Layout:
    """Pieces of at most max_bytes: rows in whole `unit`s (the kernel's slices: a head, an MLP
    chunk), or (parts) column parts of `unit` columns, whole parts a piece; wg > 1: in groups
    (a row unit then whole groups of quant.grp_r rows)."""
    if parts:
        if k % unit:
            raise ValueError(f"{k} columns in parts of {unit}")
        lay = Layout(n, k, fmt, D, unit, (), wg)
        per = max_bytes // sum(lay.block(n))
        m = k // unit
    else:
        if n % unit:
            raise ValueError(f"{n} rows in units of {unit}")
        if unit % Q.grp_r(fmt, wg):
            raise ValueError(f"rows in units of {unit}: whole groups of {Q.grp_r(fmt, wg)} rows "
                             f"({fmt} at WG {wg})")
        lay = Layout(n, k, fmt, D, 0, (), wg)
        per = 0
        while per * unit < n and sum(lay.block((per + 1) * unit)) <= max_bytes:
            per += 1
        m = n // unit
    if per < 1:
        raise ValueError(f"a unit of {unit} ({n} x {k} {fmt}) does not fit {max_bytes} bytes")
    groups = tuple((a, min(a + per, m)) for a in range(0, m, per))
    if not parts:
        groups = tuple((a * unit, b * unit) for a, b in groups)
    return Layout(n, k, fmt, D, lay.pw, groups, wg)


def pack(lay: Layout, q: np.ndarray, s: np.ndarray) -> list:
    """Each piece's bytes (uint8) from the whole matrix's quantized element and scale rows
    (quant.quantize_mxu of [n, k]): a column part's scale rows are the rows' Sw and its blocks'
    (quant.column_parts), so the values are the whole matrix's. In groups each block grouped
    (quant.grouped), unhashed (hash_piece)."""
    q, s = np.asarray(q, np.uint8), np.asarray(s, np.uint8)
    out = []
    if lay.pw:
        cols = Q.column_parts(q, s, lay.fmt, lay.k, lay.pw, lay.D)
    for i, (u0, u1) in enumerate(lay.groups):
        p = np.zeros(lay.piece_bytes(i), np.uint8)
        blocks = [cols[j] for j in range(u0, u1)] if lay.pw else [(q[u0:u1], s[u0:u1])]
        for j, (qq, ss) in enumerate(blocks):
            d, c = lay.offsets(i, j)
            if lay.wg > 1:
                qq, ss = Q.grouped(qq, ss, lay.fmt, lay.pw or lay.k, lay.wg, lay.D)
            qq, ss = qq.reshape(-1), ss.reshape(-1)
            p[d:d + qq.size] = qq
            p[c:c + ss.size] = ss
        out.append(p)
    return out


def hash_piece(lay: Layout, i: int, p: np.ndarray, at: int) -> np.ndarray:
    """Piece i's bytes (pack's) as a machine with the channel hash stores them at DRAM address
    `at`: each block's data lanes by their addresses (quant.hash_rows), its scales as they are;
    at wg = 1 as given."""
    if lay.wg == 1:
        return p
    out = np.array(p, np.uint8)
    for o, n in lay.data_regions(i):
        out[o:o + n] = Q.hash_rows(out[o:o + n], at + o, lay.wg, lay.D)
    return out


@dataclass
class Table:
    """A run's table of piece addresses in DRAM: `dram` a Tensor of n words (an address may hold
    a run-time layer index); bound (lo, hi): the addresses a piece may be at (RLD's bound)."""
    dram: Tensor
    n: int
    bound: tuple
    tile: object = None


class _Regs:
    """A builder's piece registers: (table tile, index) -> register, the last REGS used."""

    def __init__(self):
        self.held: OrderedDict = OrderedDict()

    def get(self, b, table: Table, idx: int) -> int:
        key = (id(table.tile), idx)
        if key in self.held:
            self.held.move_to_end(key)
            return self.held[key]
        if len(self.held) >= REGS:
            _, r = self.held.popitem(last=False)
            b.unscratch(r)
        r = b.scratch()
        b.rld(r, table.tile[idx:idx + 1], raw=True, bound=table.bound,
              comment=f"piece {idx}'s address")
        self.held[key] = r
        return r

    def flush(self, b) -> None:
        for r in self.held.values():
            b.unscratch(r)
        self.held.clear()


def _regs(b) -> _Regs:
    r = getattr(b, "_piece_regs", None)
    if r is None:
        r = b._piece_regs = _Regs()
    return r


def flush() -> None:
    """Give back the piece registers (a row group's body ends: the next one loads again)."""
    b = current()
    r = getattr(b, "_piece_regs", None)
    if r is not None:
        r.flush(b)


class PQ:
    """A pieced matrix (Layout) whose piece i is at bases[i] (static), or at the address in
    word first + i of `table` (bind it in the run first). Slices as a QTensor: rows within one
    piece (row_split cuts a range at the pieces), or one whole column part."""

    def __init__(self, lay: Layout, bases=None, table: Table | None = None, first: int = 0):
        if (bases is None) == (table is None):
            raise ValueError("a PQ has static bases or a table")
        self.lay, self.bases, self.table, self.first = lay, bases, table, first
        self.shape = (lay.n, lay.k)

    def _base(self, i: int) -> Affine:
        if self.bases is not None:
            return Affine.of(self.bases[i])
        if self.table.tile is None:
            raise CompileError("a pieced matrix's table is not loaded in this run (bind)")
        b = current()
        r = _regs(b).get(b, self.table, self.first + i)
        return Affine(0) + DevVar(f"piece {self.first + i}", r, align=self.lay.unit)

    def row_split(self, r0: int, r1: int) -> list:
        """[r0, r1) cut at the pieces: [(a, b)], each within one piece."""
        out, a = [], r0
        while a < r1:
            u0, u1 = self.lay.groups[self.lay.piece_of(a)]
            out.append((a, min(r1, u1)))
            a = min(r1, u1)
        return out

    def __getitem__(self, key) -> QTensor:
        if not isinstance(key, tuple):
            key = (key, slice(None))
        rk, ck = key
        lay = self.lay
        r0 = 0 if rk.start is None else rk.start
        r1 = lay.n if rk.stop is None else rk.stop
        if lay.pw:
            c0 = 0 if ck.start is None else ck.start
            c1 = lay.k if ck.stop is None else ck.stop
            if c0 % lay.pw or c1 - c0 != lay.pw:
                raise CompileError(f"stored in column parts of {lay.pw}: slice one part")
            part = c0 // lay.pw
            i = lay.piece_of(part)
            d, s = lay.offsets(i, part - lay.groups[i][0])
            base = self._base(i)
            q = QTensor.weight(base + d, base + s, lay.n, lay.pw, lay.fmt, lay.D, grp=lay.wg)
            return q[r0:r1, :]
        if not (isinstance(r0, int) and isinstance(r1, int)):
            raise CompileError("a pieced matrix's rows are sliced at static rows")
        i = lay.piece_of(r0)
        u0, u1 = lay.groups[i]
        if r1 > u1:
            raise CompileError(f"rows {r0} .. {r1} cross a piece ({u0} .. {u1}): row_split")
        d, s = lay.offsets(i)
        base = self._base(i)
        q = QTensor.weight(base + d, base + s, u1 - u0, lay.k, lay.fmt, lay.D, grp=lay.wg)
        return q[r0 - u0:r1 - u0, ck]


def bind(ns):
    """A namespace's PQs with their tables loaded into TMEM (one load a table, at the run's
    start, outside its loops); other attributes as they are."""
    from types import SimpleNamespace
    from .. import language as ol
    tiles = {}
    out = {}
    for k, v in vars(ns).items():
        if isinstance(v, PQ) and v.table is not None:
            t = v.table
            if id(t) not in tiles:
                tiles[id(t)] = Table(t.dram, t.n, t.bound, ol.load(t.dram))
            v = PQ(v.lay, table=tiles[id(t)], first=v.first)
        out[k] = v
    return SimpleNamespace(**out)
