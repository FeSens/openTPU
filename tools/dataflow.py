"""Data-movement ledger (docs/dataflow.md): the bytes a program moves at every level of a slice,
per instruction class and per producer -> consumer, from a run of the ISA simulator.

Levels:
- DRAM: reads and writes by kind (weights and scales an MM streams, the KV cache an attention MM
  streams, LD / ST, QST's quantized stores with the beats it read-merge-writes, DSTEP / STREAM
  state, WAITW polls, HALT CHAIN program loads);
- TMEM: words read per read port (the slice's: MXU, DMA, Q, Q2, Q3, VA, VB, COLL, plus SE for the
  stream engine) and written per write port (DMA, MXU, VPU, COLL), and the ports' chunk cycles
  (LANES words a cycle);
- ACT RAM: QACT's bytes and scale records written; MM's reads (each advance reads its block for
  every stationary row: logical M x D bytes, physical MCOLS x D bytes per replay group);
- IMEM: 32 bytes per executed instruction.

Producer -> consumer: every TMEM word carries the class of the instruction that wrote it; a read
adds its words to (writer, reader). When a word is overwritten (or the run ends) its life is
closed into (writer, first reader, reads: 0 / 1 / 2+): a word written by one unit and read once
by another is a hand-off through TMEM (a round trip a fused path would not make). DRAM beats
written by the program (ST, QST, DSTEP, STREAM) carry their writer the same way, so an I/O
scratch stored and loaded back, the KV cache and the DeltaNet state show as DRAM flows.

The run is the ISA simulator's (bit-exact control flow). On a zero DRAM image (the default: the
timing and the movement do not depend on the data, only config.json is read) MM skips its
products (`fast`): every product is 0 there, so its outputs are the ones the full MM computes.

    python3 tools/dataflow.py MODEL KIND N POS [--mcols M] [--layers L] [--wformat F]
        [--head-format H] [--set K=V,...] [--json FILE]
                                                 KIND: decode, rows, batch (tools/vpu_hack.py);
                                                 --set: Config fields over the board's (S=4, ...)
    python3 tools/dataflow.py moe MODEL --prefix P [--pos POS]    a MoE decode row, every expert
        resident (tools/moe_prefill_cosim.py's timing hack)

API: ledger(cfg, programs, drams=None, args=None, fast=True, poke=None) -> dict (`poke(slice)`
writes into the slice's DRAM before the run: a batch area, a MoE directory).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu import fp32 as F  # noqa: E402
from opentpu import isa as I  # noqa: E402
from opentpu import isasim as SIM  # noqa: E402
from opentpu import quant as Q  # noqa: E402

BEAT = 64                     # DRAM beat (otpu_native_dram): a partial write is read-merge-written
IBYTES = 32                   # bytes per instruction (8 words)
UNITS = {I.LD: "DMA", I.ST: "DMA", I.WAITW: "DMA", I.LDA: "DMA", I.DSTEP: "SE", I.STREAM: "SE",
         I.MM: "MXU",
         I.QACT: "QUANT", I.QST: "QUANT", I.VOP: "VPU", I.RLD: "COLL", I.GATHER: "COLL"}
RPORTS = ("MXU", "DMA", "Q", "Q2", "Q3", "VA", "VB", "COLL", "SE")
WPORTS = ("DMA", "MXU", "VPU", "COLL")
COMPOSITE = {I.V_EXP2, I.V_EXP2SUB, I.V_RECIP, I.V_RSQRT, I.V_LOG2, I.V_SIGMOID, I.V_SILU}
SUMS = {I.V_RSUM, I.V_RSSQ, I.V_RDOT}
PASSES = {I.V_EXP2: 3, I.V_EXP2SUB: 3, I.V_RECIP: 2, I.V_RSQRT: 4, I.V_LOG2: 4,
          I.V_SIGMOID: 6, I.V_SILU: 6}      # (VSIG: six passes at CNS 3, four at 5)


def vfunc(ins) -> int:
    return (ins.w[5] >> 16) & 0xFF


def opkey(ins) -> str:
    """The op; a VOP's function and, for one that reads B, B's mode (':F' full, ':R' per row,
    ':C' per column, ':S' scalar): a norm's x * r is ':R', its ss * (1 / n) ':S'."""
    if ins.op == I.VOP:
        f = vfunc(ins)
        m = ":" + "FRCS"[(ins.w[5] >> 24) & 3] if f in I.READS_B else ""
        return "VOP." + I.VFUNCS.get(f, str(f)) + m
    return I.OPNAMES.get(ins.op, f"op{ins.op:#x}")


def site(ins) -> str:
    """Where the compiler emitted it: the innermost kernel frame and its two callers."""
    fr = [(Path(f).name, fn) for f, _, fn in ins.src] if ins.src else []
    if not fr:
        return "?"
    return f"{fr[0][0][:-3]}.{fr[0][1]}" + "".join(f"<{fn}" for _, fn in fr[1:3])


def vpu_cycles(ins, lanes: int) -> int:
    """tools/vpu_hack.issue_cycles: the VPU's / stream engine's issue cycles as built."""
    if ins.op == I.DSTEP:
        rows, cols = ins.w[3] & 0xFFFF, ins.w[3] >> 16
        return -(-rows * cols // 8)
    if ins.op != I.VOP:
        return 0
    rows, cols = ins.w[3] & 0xFFFF, ins.w[3] >> 16
    f = vfunc(ins)
    ch = -(-cols // lanes)
    if f in SUMS:
        rl = 64 // lanes
        return rows * -(-ch // rl) * rl
    if f == I.V_OUTER or (f == I.V_MAC and (ins.w[5] >> 24) & 3 == I.B_COL):   # (+ the fill)
        return max(2, ch) + rows * ch
    return rows * ch * PASSES.get(f, 1)


def _beats(addrs) -> np.ndarray:
    a = np.asarray(addrs, np.int64).reshape(-1)
    return np.unique(a // BEAT) if a.size else a


def _span_beats(starts, nbytes: int) -> np.ndarray:
    """The beats of runs of nbytes from each start."""
    s = np.asarray(starts, np.int64).reshape(-1)
    if not s.size or nbytes <= 0:
        return np.zeros(0, np.int64)
    first, last = s // BEAT, (s + nbytes - 1) // BEAT
    n = int((last - first).max()) + 1
    b = first[:, None] + np.arange(n)[None, :]
    return np.unique(b[b <= last[:, None]])


class Acc:
    """The counters of one instruction class (or of the whole run)."""

    def __init__(self):
        self.n = 0
        self.rd = defaultdict(int)       # TMEM words read per port
        self.rdc = defaultdict(int)      # TMEM port chunk cycles (reads)
        self.wr = defaultdict(int)       # TMEM words written per port
        self.wrc = defaultdict(int)
        self.dram_rd = defaultdict(int)  # bytes per kind
        self.dram_wr = defaultdict(int)
        self.act_wr = 0                  # ACT RAM bytes (int8 + 8-byte scale records)
        self.act_rd = 0                  # logical: each advance's block for each row read
        self.act_rd_phys = 0             # physical: MCOLS rows a replay group
        self.act_rd_gated = 0            # ... with the read enables gated to the MM's columns
        self.imem = 0
        self.vpu = 0                     # VPU / stream engine issue cycles
        self.mxu_adv = 0                 # MXU advances (chunk cycles of the array)
        self.portb = 0                   # D-byte chunks on port B
        self.porta = 0                   # port A requests: 32-byte scale lines (GRP: beats)

    def add(self, o: "Acc") -> None:
        self.n += o.n
        for k in ("rd", "rdc", "wr", "wrc", "dram_rd", "dram_wr"):
            d = getattr(self, k)
            for kk, v in getattr(o, k).items():
                d[kk] += v
        for k in ("act_wr", "act_rd", "act_rd_phys", "act_rd_gated", "imem", "vpu", "mxu_adv",
                  "portb", "porta"):
            setattr(self, k, getattr(self, k) + getattr(o, k))

    def as_dict(self) -> dict:
        return {"n": self.n, "tmem_rd": dict(self.rd), "tmem_rd_chunks": dict(self.rdc),
                "tmem_wr": dict(self.wr), "tmem_wr_chunks": dict(self.wrc),
                "dram_rd": dict(self.dram_rd), "dram_wr": dict(self.dram_wr),
                "act_wr": self.act_wr, "act_rd": self.act_rd, "act_rd_phys": self.act_rd_phys,
                "act_rd_gated": self.act_rd_gated,
                "imem": self.imem, "vpu_cycles": self.vpu, "mxu_adv": self.mxu_adv,
                "portb": self.portb, "porta": self.porta}


class LedgerSlice(SIM.Slice):
    """isasim.Slice with the counters (module docstring)."""

    def __init__(self, cfg, sid, program, dram, fast=True):
        super().__init__(cfg, sid, program, dram)
        self.fast = fast
        self.cls: dict = {}                       # (opkey, site) -> id
        self.names: list = []
        self.acc: list = []                       # per class id: Acc
        T = cfg.TMEM_WORDS
        self.t_w = np.full(T, -1, np.int32)       # TMEM word -> its writer's class (-1: none)
        self.t_n = np.zeros(T, np.uint8)          # reads since the write (saturating at 2)
        self.t_f = np.full(T, -1, np.int32)       # its first reader's class
        self.uses = defaultdict(int)              # (writer, first reader, reads) -> words
        self.flows = defaultdict(int)             # (writer, reader) -> TMEM words read
        self.d_w = np.full(cfg.DRAM_BYTES // BEAT, -1, np.int32)   # DRAM beat -> writer class
        self.dflows = defaultdict(int)            # (writer, reader) -> bytes read
        self.prog_bytes = 0                       # HALT CHAIN program loads
        self.cur = -1

    # ------------------------------------------------------------------ classes
    def klass(self, ins) -> int:
        k = (opkey(ins), site(ins))
        c = self.cls.get(k)
        if c is None:
            c = self.cls[k] = len(self.names)
            self.names.append(k)
            self.acc.append(Acc())
        return c

    # ------------------------------------------------------------------ TMEM bookkeeping
    def _rd(self, a: Acc, port: str, idx, chunks: int | None = None) -> np.ndarray:
        idx = np.asarray(idx, np.int64).reshape(-1)
        if not idx.size:
            return idx
        u = np.unique(idx)
        a.rd[port] += u.size
        a.rdc[port] += chunks if chunks is not None else -(-u.size // self.cfg.LANES)
        return u

    def _uses_read(self, u: np.ndarray) -> None:
        if not u.size:
            return
        u = u[(u >= 0) & (u < self.cfg.TMEM_WORDS)]
        w = self.t_w[u]
        ws, cnt = np.unique(w, return_counts=True)
        for x, c in zip(ws.tolist(), cnt.tolist()):
            self.flows[(x, self.cur)] += c
        first = self.t_n[u] == 0
        self.t_f[u[first]] = self.cur
        self.t_n[u] = np.minimum(self.t_n[u].astype(np.int32) + 1, 2).astype(np.uint8)

    def _wr(self, a: Acc, port: str, idx, chunks: int | None = None) -> np.ndarray:
        idx = np.asarray(idx, np.int64).reshape(-1)
        if not idx.size:
            return idx
        u = np.unique(idx)
        a.wr[port] += u.size
        a.wrc[port] += chunks if chunks is not None else -(-u.size // self.cfg.LANES)
        return u

    def _close(self, u: np.ndarray) -> None:
        """The lives of words u end (overwritten): into uses."""
        if not u.size:
            return
        live = self.t_w[u] >= 0
        if live.any():
            key = np.stack([self.t_w[u][live], self.t_f[u][live],
                            self.t_n[u][live].astype(np.int32)], 1)
            ks, cnt = np.unique(key, axis=0, return_counts=True)
            for k, c in zip(ks.tolist(), cnt.tolist()):
                self.uses[tuple(k)] += c

    def _uses_write(self, u: np.ndarray) -> None:
        if not u.size:
            return
        self._close(u)
        self.t_w[u], self.t_n[u], self.t_f[u] = self.cur, 0, -1

    def finish(self) -> None:
        """Close every live word (the run ended): reads so far, with a flag in the key."""
        u = np.flatnonzero(self.t_w >= 0)
        if u.size:
            key = np.stack([self.t_w[u], self.t_f[u], self.t_n[u].astype(np.int32) + 10], 1)
            ks, cnt = np.unique(key, axis=0, return_counts=True)
            for k, c in zip(ks.tolist(), cnt.tolist()):
                self.uses[tuple(k)] += c

    # ------------------------------------------------------------------ DRAM bookkeeping
    def _dram_read(self, a: Acc, kind: str, nbytes: int, beats=None) -> None:
        a.dram_rd[kind] += int(nbytes)
        if beats is not None and len(beats):
            b = beats[beats < self.d_w.size]
            w = self.d_w[b]
            ws, cnt = np.unique(w[w >= 0], return_counts=True)
            for x, c in zip(ws.tolist(), cnt.tolist()):
                self.dflows[(int(x), self.cur)] += c * BEAT

    def _dram_write(self, a: Acc, kind: str, nbytes: int, beats=None) -> None:
        a.dram_wr[kind] += int(nbytes)
        if beats is not None and len(beats):
            self.d_w[beats[beats < self.d_w.size]] = self.cur

    # ------------------------------------------------------------------ execution
    def step(self) -> None:
        if self.pc < len(self.prog):
            ins = self.prog[self.pc]
            c = self.klass(ins)
            a = self.acc[c]
            a.imem += IBYTES
            if ins.op == I.WAITW:          # a poll: one DRAM word, the TMEM word if it holds
                self.cur = c
                a.n += 1
                self._dram_read(a, "waitw", 4)
                if self.holds(ins):
                    t = (self.reg(ins.rb) + ins.w[1]) & 0xFFFFFFFF
                    u = self._wr(a, "DMA", [t])
                    self._uses_write(u)
            elif ins.op == I.HALT and ins.flags & I.F_CHAIN:
                n = self.reg(ins.rb)
                self.prog_bytes += IBYTES * n
                a.dram_rd["prog"] += IBYTES * n
        super().step()

    def execute(self, ins) -> None:
        op = ins.op
        if op in (I.NOP, I.LI, I.ADDI):
            return super().execute(ins)
        c = self.klass(ins)
        self.cur = c
        a = self.acc[c]
        a.n += 1
        h = getattr(self, "_a_" + I.OPNAMES.get(op, "x").lower(), None)
        if h is not None:
            post = h(ins, a)
        else:
            post = None
        if op == I.MM and self.fast:
            self._mm_fast(ins)
        elif op == I.STREAM:
            self._stream_traced(ins, a)
        else:
            super().execute(ins)
        if post is not None:
            post()

    # ---- per-op accounting: reads first (with the old tags), writes after the instruction
    def _a_rld(self, ins, a):
        t = (self.reg(ins.ra) + ins.w[0]) & 0xFFFFFFFF
        self._uses_read(self._rd(a, "COLL", [t]))

    def _a_ld(self, ins, a):
        d = (self.reg(ins.ra) + ins.w[0]) & 0xFFFFFFFF
        t = (self.reg(ins.rb) + ins.w[1]) & 0xFFFFFFFF
        n = ins.w[2]
        kv = self.names[self.cur][1].startswith("attention.")   # V's per-token scales, masks
        self._dram_read(a, "ld.kv" if kv else "ld", 4 * n, _span_beats([d], 4 * n))
        a.portb += -(-4 * n // self.cfg.D)
        u = self._wr(a, "DMA", t + np.arange(n))
        return lambda: self._uses_write(u)

    def _a_st(self, ins, a):
        d = (self.reg(ins.ra) + ins.w[0]) & 0xFFFFFFFF
        t = (self.reg(ins.rb) + ins.w[1]) & 0xFFFFFFFF
        n = ins.w[2]
        self._uses_read(self._rd(a, "DMA", t + np.arange(n)))
        a.portb += -(-4 * n // self.cfg.D)
        self._dram_write(a, "st", 4 * n, _span_beats([d], 4 * n))

    def _a_mm(self, ins, a):
        cfg, w = self.cfg, ins.w
        D = cfg.D
        sa = (self.reg(ins.ra) + w[0]) & 0xFFFFFFFF
        ssa = (self.reg(ins.rb) + w[1]) & 0xFFFFFFFF
        out = (self.reg(ins.rc) + w[2]) & 0xFFFFFFFF
        N, KB, k0 = w[3] & 0xFFFF, (w[3] >> 16) & 0xFF, (w[3] >> 24) & 0xFF
        rs = w[4]
        ors, M = w[5] & 0xFFFF, (w[5] >> 16) & 0xFF
        srs = w[6]
        hld = 0
        if srs & I.MM_SPL:          # SPL (SPLMM, isasim._mm): KB terms from two parts of a chunk a
            hld = (srs >> I.MM_HLD & (1 << 31 - I.MM_HLD) - 1) << 10    # group, hi at sa + g rs,
            srs &= (1 << I.MM_HLD) - 1                                  # lo HLD after it
        fl = ins.flags
        wf = (fl >> I.WF_SHIFT) & 3
        pair, unit, rsc = bool(fl & I.F_PAIR), bool(fl & I.F_UNIT), bool(fl & I.F_RSC)
        bpb = D if wf == I.W8 else D // 2
        chunks = -(-KB * bpb // D)                    # D-byte chunks a streamed row
        adv = -(-KB // 2) if pair else KB             # advances a row
        groups = -(-M // cfg.MCOLS)
        # weight groups (docs/isa.md): Rg rows a group, whole groups streamed, each group's rows
        # chunk-interleaved at sa + g * rs; an advance takes a P-chunk (Rg rows x D bytes) for
        # one of the M ACT rows, 4-bit with PAIR both blocks of a chunk (phase 2a's packed PE,
        # docs/weight_groups.md), without it one a block (otpu_mxu's cmd_KBa). A group's
        # chunk c is its rows' K-block c (P = Rg x bpb bytes; otpu_mxu's pg: a 4-bit chunk is a
        # block), so a group streams KB chunks whatever KB's parity, where a row of one streams
        # whole D-byte chunks (4-bit: two blocks each, an odd KB's last half of one)
        Rg = 1 << ins.rd
        G = -(-N // Rg)
        gch = chunks if Rg == 1 else 2 if hld else KB  # chunks a row (Rg 1) or a group (SPL: its parts')
        gb = gch * (D if Rg == 1 else Rg * bpb)       # their bytes
        gadv = -(-KB // 2) if pair and wf != I.W8 else KB
        a.mxu_adv += N * adv * groups if Rg == 1 else G * M * gadv
        a.portb += G * Rg * chunks if Rg == 1 else G * gb // D
        kv = self.names[self.cur][1].startswith("attention.")   # q.K^T, P.V: the KV cache
        kind = "kv" if kv else "weights"
        gs = sa + np.arange(G, dtype=np.int64) * rs
        self._dram_read(a, kind, G * gb, _span_beats(np.concatenate([gs, gs + hld]), gb // 2)
                        if hld else _span_beats(gs, gb))
        if not unit:
            # a row's scale bytes: its Sw (rows of one: the scale row's header, 4 / 8 / 8 / 16
            # bytes; in groups the Sw words' part, SWP = 4 R, NVFP4 8 R) and its blocks' entries
            per = 4 if rsc else ({I.W8: 4 + KB, I.W4I: 8 + 4 * KB, I.W4F: 8 + 4 * KB,
                                  I.WNV: 16 + 8 * KB} if Rg == 1 else
                                 {I.W8: 4 + KB, I.W4I: 4 + 4 * KB, I.W4F: 4 + 4 * KB,
                                  I.WNV: 8 + 8 * KB})[wf]
            self._dram_read(a, "kvscale" if kv else "scales", G * Rg * per)
            if Rg == 1:
                a.porta += N * (1 if rsc else adv)
            else:
                # docs/weight_groups.md: a request (port A's 128-byte beat) a group's step, RSC's a chunk;
                # the group's Sw line in its first step's beat, or (k0's entries in another beat)
                # a request of its own before it
                fmt = {I.W8: "int8", I.W4I: "int4", I.W4F: "fp4", I.WNV: "nvfp4"}[wf]
                so = Q.swp(fmt, Rg) + k0 * Rg * Q.SCALE_DATA[fmt]
                g0 = ssa + np.arange(G, dtype=np.int64) * srs
                sws = 0 if rsc else int(np.count_nonzero(g0 // 128 != (g0 + so) // 128))
                a.porta += G * (gch if rsc else gadv) + sws
        a.act_rd += N * KB * M * D + N * KB * M * 8       # bytes and records, per output row
        a.act_rd_phys += N * adv * groups * cfg.MCOLS * (D + 8)
        # otpu_mxu's act_ren: a command of one group reads only its M columns (PAIR: 2M)
        cols = min(cfg.MCOLS, 2 * M if pair else M) if groups == 1 else groups * cfg.MCOLS
        a.act_rd_gated += N * adv * cols * (D + 8)
        idx = out + np.arange(M)[:, None] * ors + np.arange(N)[None, :]
        E = int(os.environ.get("OTPU_MXU_E", 0)) or cfg.MCOLS
        dch = N * -(-M // E)
        if fl & I.F_ASCALE:
            self._uses_read(np.union1d(self._rd(a, "MXU", idx, dch),
                                       self._rd(a, "MXU", ssa + np.arange(M), 1)))
        elif fl & I.F_ACC:
            self._uses_read(self._rd(a, "MXU", idx, dch))
        widx = idx.reshape(-1)
        if fl & I.F_RMAX:
            widx = np.concatenate([widx, out + M * ors + np.arange(M)])
        u = self._wr(a, "MXU", widx, dch + (1 if fl & I.F_RMAX else 0))
        return lambda: self._uses_write(u)

    def _quant_reads(self, a, src, rows, KB, srs, cs, rsc, passes):
        D = self.cfg.D
        idx = src + np.arange(rows)[:, None] * srs + np.arange(KB * D)[None, :]
        n = rows * -(-KB * D // self.cfg.LANES)
        u = self._rd(a, "Q", idx, n * passes)
        a.rd["Q"] += u.size * (passes - 1)
        us = [u]
        if cs is not None:
            us.append(self._rd(a, "Q2", cs + np.arange(KB * D), n))
        if rsc is not None:
            us.append(self._rd(a, "Q3", rsc + np.arange(rows), rows))
        self._uses_read(np.unique(np.concatenate(us)))

    XD = 1024                         # otpu_quant's row buffer: chunks of LANES words
    QSTB = False                      # otpu_quant QSTB (QUANT_QSTB): QST rows through it, ...
    QSW = 16                          # ... their pass 1 at 4 x QUANT_SWW bytes a cycle

    def _qst_buffered(self, ins, dst: int, KB: int, drs: int, es: int) -> bool:
        """otpu_quant's QSTB rows: ROW, not HALF, through the row buffer, pass 1 QSW bytes a
        cycle (GRP rows; else element stride 1, dst and the row stride QSW aligned): read
        once."""
        f = ins.flags
        if not self.QSTB or not f & I.F_ROW or f & I.F_HALF or \
                KB * self.cfg.D > self.XD * self.cfg.LANES:
            return False
        if f & I.F_QGRP:
            return not f & I.F_QCOL
        return es == 1 and dst % self.QSW == 0 and drs % self.QSW == 0

    def _a_qact(self, ins, a):
        cfg, w = self.cfg, ins.w
        src = (self.reg(ins.ra) + w[0]) & 0xFFFFFFFF
        rows, KB = w[1] & 0xFF, w[1] >> 16
        dup = bool(ins.flags & I.F_DUP)
        cs = (w[3] & 0xFFFFFFFF) if ins.flags & I.F_CSCALE else None
        rsc = (w[4] & 0xFFFFFFFF) if ins.flags & I.F_RSCALE else None
        passes = 1 if KB * cfg.D <= self.XD * cfg.LANES else 2
        st = ins.flags & I.F_QST                # (ST: srs w6; its rows to DRAM at R[rc] + w3)
        self._quant_reads(a, src, rows, KB, w[5] if st else w[2], cs, rsc, passes)
        a.act_wr += (2 if dup else 1) * rows * KB * (cfg.D + 8)
        if st:
            d, nb = self.da(ins.rc, w[2]), I.qrow_bytes(KB, cfg.D)
            self._dram_write(a, "qact_st", rows * nb,
                             _span_beats([d + r * w[6] for r in range(rows)], nb))

    def _a_lda(self, ins, a):
        cfg, w = self.cfg, ins.w
        rows, KB = w[1] & 0xFF, w[1] >> 16
        d, nb = self.da(ins.ra, w[0]), I.qrow_bytes(KB, cfg.D)
        self._dram_read(a, "lda", rows * nb, _span_beats([d + r * w[2] for r in range(rows)], nb))
        a.portb += rows * -(-nb // cfg.D)
        a.act_wr += rows * KB * (cfg.D + 8)

    def _a_qst(self, ins, a):
        cfg, w = self.cfg, ins.w
        src = (self.reg(ins.ra) + w[0]) & 0xFFFFFFFF
        dst = (self.reg(ins.rb) + w[1]) & 0xFFFFFFFF
        sdst = (self.reg(ins.rc) + w[2]) & 0xFFFFFFFF
        rows, KB = w[3] & 0xFFFF, w[3] >> 16
        srs, drs, es = w[4], w[5], w[6]
        hld = 0
        if srs & I.QST_HL:          # HL (QSTHL, through the row buffer): one read, pass 1 pops
            hld = (srs >> I.QST_HLD & 0x7FF) << 10      # each chunk twice (2 KB D / QSW: pass
            srs &= (1 << I.QST_HLD) - 1                 # 0's KB D / LANES at QSW = 2 LANES)
        row_mode, half = bool(ins.flags & I.F_ROW), bool(ins.flags & I.F_HALF)
        passes = 1 if hld or self._qst_buffered(ins, dst, KB, drs, es) else 2
        self._quant_reads(a, src, rows, KB, srs, None, None, passes)
        ne = KB * cfg.D // 2 if half else KB * cfg.D
        baddr = dst + np.arange(rows)[:, None] * drs + np.arange(ne)[None, :] * es
        sb = 4 if row_mode else 4 + 4 * -(-KB // 4)
        saddr = sdst + np.arange(rows)[:, None] * sb + np.arange(sb)[None, :]
        allb = np.concatenate([baddr.reshape(-1), saddr.reshape(-1)])
        if hld:                                         # (lo's bytes and scales, HLD after)
            allb = np.concatenate([allb, allb + hld])
        beats, cnt = np.unique(allb // BEAT, return_counts=True)
        part = int((cnt < BEAT).sum())                # beats written in part: read, merge, write
        self._dram_write(a, "qst", allb.size, beats)
        self._dram_write(a, "qst_beats", beats.size * BEAT)
        self._dram_read(a, "qst_rmw", part * BEAT)

    def _a_vop(self, ins, a):
        cfg, w = self.cfg, ins.w
        L = cfg.LANES
        dst = (self.reg(ins.ra) + w[0]) & 0xFFFFFFFF
        av = (self.reg(ins.rb) + w[1]) & 0xFFFFFFFF
        bv = (self.reg(ins.rc) + w[2]) & 0xFFFFFFFF
        rows, cols = w[3] & 0xFFFF, w[3] >> 16
        drs, ars = w[4] & 0xFFFF, w[4] >> 16
        brs, func, bmode = w[5] & 0xFFFF, (w[5] >> 16) & 0xFF, (w[5] >> 24) & 3
        imm = (self.reg(ins.rd) + w[6]) & 0xFFFFFFFF
        a.vpu += vpu_cycles(ins, L)
        r, c = np.arange(rows)[:, None], np.arange(cols)[None, :]
        ch = rows * -(-cols // L)
        us = []
        if func == I.V_OUTER:
            didx = dst + r * drs + c
            us.append(self._rd(a, "VA", didx, ch))
            us.append(self._rd(a, "VB", bv + np.arange(rows) * brs, rows))
            us.append(self._rd(a, "VB", imm + np.arange(cols), -(-cols // L)))
            if not ins.flags & I.F_DONE:
                dv = [av] if ins.flags & I.F_DSCALAR else av + np.arange(cols)
                us.append(self._rd(a, "VA", dv, 1))
            self._uses_read(np.unique(np.concatenate(us)))
            u = self._wr(a, "VPU", didx, ch)
            return lambda: self._uses_write(u)
        if func == I.V_MAC:                     # dst on port B (in place); B's columns filled
            didx = dst + r * drs + c
            us.append(self._rd(a, "VA", av + r * ars + c, ch))
            us.append(self._rd(a, "VB", didx, ch))
            if bmode == I.B_COL:
                us.append(self._rd(a, "VB", bv + np.arange(cols), -(-cols // L)))
            self._uses_read(np.unique(np.concatenate(us)))
            u = self._wr(a, "VPU", didx, ch)
            return lambda: self._uses_write(u)
        if func != I.V_FILL:
            us.append(self._rd(a, "VA", av + r * ars + c, ch))
        if func in I.READS_B and bmode != I.B_SCALAR:
            bidx = {I.B_FULL: bv + r * brs + c, I.B_ROW: bv + np.arange(rows) * brs,
                    I.B_COL: bv + c}[bmode]
            us.append(self._rd(a, "VB", bidx, ch if bmode != I.B_ROW else rows))
        if us:
            self._uses_read(np.unique(np.concatenate(us)))
        if func == I.V_ARGMAX:
            widx = dst + np.arange(rows)[:, None] * drs + np.arange(2)[None, :]
            u = self._wr(a, "VPU", widx, rows)
        elif func in I.REDUCE:
            u = self._wr(a, "VPU", dst + np.arange(rows) * drs, rows)
        else:
            u = self._wr(a, "VPU", dst + r * drs + c, ch)
        return lambda: self._uses_write(u)

    def _a_dstep(self, ins, a):
        w = ins.w
        d = (self.reg(ins.ra) + w[0]) & 0xFFFFFFFF
        qk = (self.reg(ins.rb) + w[1]) & 0xFFFFFFFF
        v = (self.reg(ins.rc) + w[2]) & 0xFFFFFFFF
        rows, cols = w[3] & 0xFFFF, w[3] >> 16
        g, o, gs = w[4], w[5], w[6]
        a.vpu += vpu_cycles(ins, self.cfg.LANES)
        nb = rows * cols * 4
        beats = _span_beats([d], nb)
        if not ins.flags & I.F_DZERO:
            self._dram_read(a, "state", nb, beats)
            a.portb += -(-nb // self.cfg.D)
        a.portb += -(-nb // self.cfg.D)
        us = [self._rd(a, "SE", qk + np.arange(2 * cols)), self._rd(a, "SE", v + np.arange(rows)),
              self._rd(a, "SE", [g, g + gs])]
        self._uses_read(np.unique(np.concatenate(us)))
        u = self._wr(a, "VPU", o + np.arange(rows))

        def post():
            self._dram_write(a, "state", nb, beats)
            self._uses_write(u)
        return post

    def _a_gather(self, ins, a):
        w = ins.w
        src = (self.reg(ins.ra) + w[0]) & 0xFFFFFFFF
        rows, cols, srs = w[2] & 0xFFFF, w[2] >> 16, w[3]
        idx = src + np.arange(rows)[:, None] * srs + np.arange(cols)[None, :]
        self._uses_read(self._rd(a, "COLL", idx))

    # ---- STREAM: the simulator's own TMEM accesses (SE), the DRAM stream from the descriptor
    def _stream_traced(self, ins, a):
        reads, writes = [], []
        tget, tput = self.tget, self.tput
        m32 = self.m32
        self.tget = lambda idx: (reads.append(np.asarray(idx, np.int64).reshape(-1)), tget(idx))[1]
        self.tput = lambda idx, v: (writes.append(np.asarray(idx, np.int64).reshape(-1)),
                                    tput(idx, v))[1]
        try:
            SIM.Slice._stream(self, ins)
        finally:
            del self.tget, self.tput
        import re
        m = re.search(r"stream (\d+)x(\d+)", ins.comment or "")
        rows, cols = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        a.vpu += -(-rows * cols // 8)
        fl = ins.flags
        nb = rows * cols * 4
        if not fl & I.F_SZERO and not fl & I.F_SRC_T:
            self._dram_read(a, "state", nb)
            a.portb += -(-nb // self.cfg.D)
        if not fl & I.F_NODST and not fl & I.F_DST_T:
            self._dram_write(a, "state", nb)
            a.portb += -(-nb // self.cfg.D)
        if reads:
            self._uses_read(self._rd(a, "SE", np.concatenate(reads)))
        if writes:
            self._uses_write(self._wr(a, "VPU", np.concatenate(writes)))

    # ---- MM on a zero image: no products (module docstring)
    def _mm_fast(self, ins) -> None:
        cfg, w = self.cfg, ins.w
        out = (self.reg(ins.rc) + w[2]) & 0xFFFFFFFF
        ssa = (self.reg(ins.rb) + w[1]) & 0xFFFFFFFF
        N, KB, k0 = w[3] & 0xFFFF, (w[3] >> 16) & 0xFF, (w[3] >> 24) & 0xFF
        ors, M, ab = w[5] & 0xFFFF, (w[5] >> 16) & 0xFF, w[5] >> 24
        srs = w[6] & (1 << I.MM_HLD) - 1 if w[6] & I.MM_SPL else w[6]     # (SPL: HLD above srs)
        unit, rsc = bool(ins.flags & I.F_UNIT), bool(ins.flags & I.F_RSC)
        wf = (ins.flags >> I.WF_SHIFT) & 3
        sw, _, _ = self._mx_scales(wf, ssa, srs, N, KB, k0, unit, rsc, 1 << ins.rd)
        W = I.MX_W[wf]
        acc = np.zeros((M, N), np.int64)
        y = F.mul(F.mul(F.i2f(acc), sw[None, :]),
                  F.mul(self.ascale[:M, ab][:, None], np.float32(2.0 ** -W)))
        idx = out + np.arange(M)[:, None] * ors + np.arange(N)[None, :]
        if ins.flags & I.F_ASCALE:
            alpha = self.tget(ssa + np.arange(M))[:, None]
            y = F.add(F.mul(self.tget(idx), alpha), y)
        elif ins.flags & I.F_ACC:
            y = F.add(self.tget(idx), y)
        self.tput(idx, y)
        if ins.flags & I.F_RMAX:
            self.tput(out + M * ors + np.arange(M), F.chain_max(y))


# ---------------------------------------------------------------------------- the fusion map
# role -> (where it belongs, what it is). a: the MXU's output path (the epilogue lanes, at drain
# rate); a*: the same with a numerics or weight-layout change; b: QUANT's input path; c: stays in
# a (small) vector or scalar unit; se: the stream engine; x: a layout copy the compiler drops.
FUSION = {
    "residual": ("a", "x + y: MM ACC into the residual"),
    "norm.stat": ("a", "RMSNorm statistics: RSSQ on the residual MM's drain, rsqrt in QACT"),
    "norm.apply": ("a", "RMSNorm scaling outside QACT: QACT RSCALE / CSCALE"),
    "postnorm": ("c", "x + norm(y) (Gemma's sandwich norms): y's row statistic before the add"),
    "qknorm": ("a*", "per-head q / k RMSNorm: sum of squares on drain, r into QACT / QST"),
    "rope": ("a*", "RoPE on drain: paired weight rows, cos / sin operand"),
    "pad": ("x", "head padding copies: the MM writes the padded tile"),
    "glu": ("a", "SiLU / GELU (g) * u on drain: interleaved gate / up rows"),
    "gate": ("a", "sigmoid gate on its MM's drain"),
    "attn.out": ("a", "acc / l (x gate) on the last P.V MM's drain"),
    "softmax.p": ("b", "p = exp2(s - m) and its row sum: QUANT's input path (P's QACT)"),
    "softmax.row": ("c", "online-softmax row scalars: m, alpha, l"),
    "softmax.init": ("c", "online-softmax state init (fills)"),
    "mask": ("b", "masked block: min with the mask row / row max"),
    "conv": ("a*", "short causal conv on drain: taps and the previous rows"),
    "dn.front": ("a*", "DeltaNet q / k SiLU and L2 norm on drain"),
    "dn.gates": ("c", "DeltaNet per-head gates (scalars)"),
    "dn.post": ("b", "DeltaNet gated RMSNorm of o: into W_out's QACT"),
    "dn.state": ("se", "DeltaNet state step (DSTEP / STREAM)"),
    "ple": ("a", "Gemma 4 per-layer inputs: GELU and product on drain"),
    "sample": ("c", "argmax / sampling over the logits"),
    "moe": ("c", "MoE routing, top-k and combine"),
    "misc": ("c", "copies, fills, layout"),
}


def role(op: str, site_: str) -> str | None:
    """A VOP / DSTEP / STREAM class's role in the layer (FUSION); None for the other units."""
    if op in ("DSTEP", "STREAM"):
        return "dn.state"
    if not op.startswith("VOP"):
        return None
    f = op[4:]
    bm = f.partition(":")[2]                 # B's mode: F, R, C, S ('' : B not read)
    names = site_.split("<")
    fn, chain = names[0], set(n.split(".")[-1] for n in names)
    if "moe" in fn or any(n.startswith("moe") for n in chain):
        return "moe"
    if fn == "lib.rmsnorm":
        if "_norm_heads" in chain:
            return "qknorm"
        if "post" in chain:
            return "dn.post"
        if "_add_norm" in chain:
            return "postnorm"
        # rssq, then ss * (1 / n), + eps, rsqrt (the statistics); x * r, * gamma (the scaling)
        return "norm.apply" if f.startswith("mul") and bm in ("F", "R", "C") else "norm.stat"
    if fn == "gemma4._add_norm":
        return "postnorm" if "_ple" not in chain else "ple"
    if fn == "deltanet.l2norm_rows":
        return "dn.front"
    if fn in ("deltanet.gates", "lib.softplus"):
        return "dn.gates"
    if fn in ("lib.rope", "lib.rope_rows", "gemma4._rope", "qwen3._rope_gate"):
        return "rope"
    if fn in ("qwen3._rope_padded", "qwen3._rope_rows_padded", "qwen3._padded"):
        return "pad"
    # the tile kernel's (opentpu/llm/tiled.py): its norm statistics, residual adds, acc / l and
    # SwiGLU have their own sites
    if fn == "tiled._norm_quantize":
        return "norm.stat"
    if fn == "tiled._add_parts":
        return "residual"
    if fn == "tiled.emit":
        return "attn.out"
    if fn == "tiled._mlp_tile":
        return "glu"
    if fn in ("lib.sigmoid", "lib.silu", "lib.gelu_tanh") and "_mlp_tile" in chain:
        return "glu"
    if fn == "attention.fill":
        return "mask"
    if fn in ("lib.sigmoid", "lib.silu", "lib.gelu_tanh"):
        if chain & {"swiglu_down", "_dense", "_swiglu_down_loop", "down"}:
            return "glu"
        if "_ple" in chain or "_ple_inputs" in chain:
            return "ple"
        if chain & {"qk", "head_in", "prep"} and "gatez" not in chain:
            return "dn.front"
        if chain & {"gatez", "post"} or "head" in chain:
            return "dn.post"
        if chain & {"_attention", "_attention_rows"}:
            return "gate"
        return "misc"
    if fn in ("mlp.swiglu_down", "mlp._swiglu_down_loop", "mlp.down"):
        return "glu"
    if fn in ("qwen35.conv", "lfm2._conv", "lfm2._conv_rows", "qwen35.head_in"):
        return "conv"
    if fn == "qwen35.post":
        return "dn.post"
    if fn in ("gemma4._ple", "gemma4._ple_inputs"):
        return "ple"
    if fn == "attention.finish":
        # p = exp2(s - m_new[:, None]) (B per row), its row sum, a tail block's padded P (fill,
        # p * vs per column); alpha = exp2(m - m_new), m, l (vectors of a value per row)
        if f in ("exp2sub:R", "rsum", "mul:C") or f.startswith("fill"):
            return "softmax.p"
        return "softmax.row"
    if fn == "attention.__init__":
        return "softmax.init"
    if fn == "attention.scores":
        return "mask"
    if fn in ("qwen3.emit", "gemma4.emit"):
        return "attn.out"
    if fn in ("qwen3._attention", "qwen3._attention_rows", "gemma4._attention"):
        return "residual" if f.startswith("add") else "attn.out"
    if fn in ("qwen3._mlp", "qwen35._deltanet_dstep", "qwen35._deltanet_rows", "gemma4._mlp",
              "gemma4._dense", "lfm2._conv_layer"):
        return "residual" if f.startswith("add") else "misc"
    if fn.startswith("generate.") or fn.startswith("batch._sinks"):
        return "sample"
    return "misc"


class LedgerMachine(SIM.Machine):
    def __init__(self, cfg, programs, drams, args=None, fast=True):
        assert len(programs) == cfg.S
        SIM._fit_imem(cfg, programs)
        self.cfg = cfg
        self.slices = [LedgerSlice(cfg, s, programs[s], None if drams is None else drams[s],
                                   fast) for s in range(cfg.S)]
        self.args = args
        self.chains = 0
        self.host = None
        for s in self.slices:
            s.R = SIM._regs(args)


def _name(names, c) -> str:
    if c < 0:
        return "init"
    return f"{names[c][0]}@{names[c][1]}"


def ledger(cfg, programs, drams=None, args=None, fast=True, poke=None, host=None) -> dict:
    """Run the programs on the ISA simulator with the counters; the ledger as a dict (module
    docstring). drams None: zero images (the slices' DRAM as allocated)."""
    m = LedgerMachine(cfg, programs, drams, args, fast)
    if poke is not None:
        for s in m.slices:
            poke(s)
    m.host = host
    t = time.time()
    m.run(max_steps=1 << 40)
    s = m.slices[0]
    s.finish()
    names = s.names
    tot = Acc()
    for a in s.acc:
        tot.add(a)
    loaded = sum(len(p) for p in programs) * IBYTES
    return {
        "sim_s": round(time.time() - t, 1),
        "instructions": s.icount,
        "programs_bytes": loaded,
        "chained_bytes": s.prog_bytes,
        "total": tot.as_dict(),
        "classes": [{"op": n[0], "site": n[1], **a.as_dict()} for n, a in zip(names, s.acc)],
        "flows": [[_name(names, w), _name(names, r), v] for (w, r), v in s.flows.items()],
        "uses": [[_name(names, w), _name(names, f), n, v] for (w, f, n), v in s.uses.items()],
        "dflows": [[_name(names, w), _name(names, r), v] for (w, r), v in s.dflows.items()],
    }


# ---------------------------------------------------------------------------- cases
def card_env() -> None:
    """The card's features (the co-sims' OTPU_PAIR / DSTEP / STREAM)."""
    for k in ("OTPU_PAIR", "OTPU_DSTEP", "OTPU_STREAM"):
        os.environ.setdefault(k, "1")


def compile_tile(model, T, pos, mcols=None, wformat="fp4", head="int8", layers=None,
                 dram_bytes=1 << 32):
    """A tiled prompt run of T rows from pos (opentpu/llm/tiled.compile_tile, as
    tools/perf_tiled.py --kernel tile builds it: ACT_ROWS max(the board's, MCOLS, T), the
    card's TMEM); vpu_hack.compile_case's full tuple."""
    sys.path.insert(0, str(ROOT / "tools"))
    import vpu_hack as VH
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec, model_dir, tiled
    os.environ.setdefault("OTPU_TILE", "1")          # the image's mask tiles (Gemma)
    spec = VH._prefix(load_spec(model_dir(model)), layers)
    mk = {"MCOLS": mcols} if mcols else {}
    c0 = board_config(DRAM_BYTES=dram_bytes, **mk)
    cfg = board_config(DRAM_BYTES=dram_bytes, ACT_ROWS=max(c0.act_rows, c0.MCOLS, T), **mk)
    cap = 256 * ((pos + T - 1) // 256 + 1)
    img = spec.image(cfg, cap, rows=T, wformat=wformat, head_format=head)
    return cfg, tiled.compile_tile(img, pos, T, 256), img, None, spec


def compile_any(model, kind, n, pos, mcols=None, wformat="fp4", head="int8", layers=None,
                dram_bytes=1 << 32, sets=None):
    """vpu_hack.compile_case's full tuple; kind "tile": compile_tile."""
    if kind == "tile":
        return compile_tile(model, n, pos, mcols, wformat, head, layers, dram_bytes)
    sys.path.insert(0, str(ROOT / "tools"))
    import vpu_hack as VH
    return VH.compile_case(model, kind, n, pos, mcols, wformat, head, layers=layers,
                           dram_bytes=dram_bytes, full=True, sets=sets)


def case(model, kind, n, pos, mcols=None, layers=None, wformat="fp4", head="int8", rows_kw=None,
         sets=None):
    """(cfg, programs, poke, info) of a dense case (tools/vpu_hack.compile_case; kind "tile":
    a tiled prompt run); sets: Config fields over the board's."""
    card_env()
    cfg, progs, img, area, spec = compile_any(model, kind, n, pos, mcols, wformat, head, layers,
                                              sets=sets)
    end = area["end"] if area else img.nbytes
    # the run's DRAM is the image's; a program over IMEM (8 batched rows) runs as one here (the
    # card splits the step into groups of rows that fit, each streaming the weights again)
    rcfg = dataclasses.replace(cfg, DRAM_BYTES=-(-end // (1 << 20)) * (1 << 20),
                               IMEM_WORDS=max(cfg.IMEM_WORDS, 8 * max(len(p) for p in progs)))
    poke = None
    if area:
        from opentpu.llm import batch as BA

        def poke(s):
            BA.build([s.dram], area, spec, cfg)
    info = dict(model=str(model).rstrip("/").split("/")[-1], kind=kind, n=n, pos=pos,
                mcols=cfg.MCOLS, layers=layers, wformat=wformat, head=head,
                image_bytes=int(img.nbytes), act_rows=cfg.act_rows)
    return rcfg, progs, poke, info


def moe_case(model, prefix, pos, R=1):
    """A MoE decode row (moe_prefill_cosim rows --variant moe): every expert resident."""
    card_env()
    sys.path.insert(0, str(ROOT / "tools"))
    import moe_prefill_cosim as MP
    kind, M, spec = MP.load_model(model)
    cfg = MP.cfg_of()
    cfg = dataclasses.replace(cfg, DRAM_BYTES=1 << 32)
    s, img = MP.model_image(kind, M, spec, prefix, "moe", R, cfg, 4096)
    progs = img.compile_step(pos)
    rcfg = dataclasses.replace(cfg, DRAM_BYTES=-(-img.nbytes // (1 << 20)) * (1 << 20))

    def poke(sl):
        MP.timing_hack(img, sl.dram)
    info = dict(model=Path(model).name, kind="moe", n=R, pos=pos, prefix=prefix, mcols=cfg.MCOLS,
                image_bytes=int(img.nbytes))
    return rcfg, progs, poke, info


# ---------------------------------------------------------------------------- upper bounds
# Timing-only transforms (docs/dataflow.md, upper bounds): an instruction whose work a fused path
# would do becomes a "span" stand-in that keeps its dependencies but not its work: a VOP becomes
# a two-element ADD whose first and last elements are the original's first and last words of
# each operand and of dst (the sequencer's footprints are ranges, so every producer it waited
# for and every consumer that waited for it still wait), at one chunk per row; a QACT becomes a
# one-row, one-block QACT at the same place (quantize-on-drain: the quantizer's pass is hidden
# under the producing MM's drain). As tools/vpu_hack.py's transforms, the results are wrong
# afterwards; only the cycles mean anything.
FUSE_MODES = {
    "A": {"a", "x"},                         # the epilogue as the ISA could take it now
    "AS": {"a", "a*", "x"},                  # + the numerics / layout changes
    "ASB": {"a", "a*", "x", "b"},            # + softmax's P on QUANT's input path
    "C": {"a", "a*", "x", "b", "c"},         # every VOP (the VPU free, dependencies kept)
}
# stages by role (docs/dataflow.md 7.3): T, what today's ISA takes with compiler changes alone
# (the residual as MM ACC, o / l as W_o's RSCALE, the gain as CSCALE, the pad copies); GE, the
# quantizer's two composite modes (GLU: act(g) x u, the gates, PLE; EXP + SUM: the softmax
# numerators with the mask). A mode's parts may also name roles (FUSION) one by one.
STAGE_ROLES = {
    "T": {"residual", "attn.out", "norm.apply", "pad"},
    "GE": {"glu", "gate", "ple", "softmax.p", "mask"},
}


def _span(base, rows, rs, cols) -> tuple[int, int]:
    return base, max(0, (rows - 1) * rs) + max(0, cols - 1)


def span_vop(ins) -> I.Instr:
    """The VOP as a two-element ADD over its operands' and dst's first and last words."""
    w = list(ins.w)
    rows, cols = w[3] & 0xFFFF, w[3] >> 16
    drs, ars = w[4] & 0xFFFF, w[4] >> 16
    brs, func, bmode = w[5] & 0xFFFF, (w[5] >> 16) & 0xFF, (w[5] >> 24) & 3
    red = func in I.REDUCE
    dspan = max(0, (rows - 1) * drs) + (1 if func == I.V_ARGMAX else 0 if red else max(0, cols - 1))
    if func == I.V_OUTER:                    # in place: A is dst
        aspan, a_rb, a_w2 = dspan, ins.ra, w[0]
    elif func == I.V_FILL:
        aspan, a_rb, a_w2 = dspan, ins.ra, w[0]
    else:
        aspan, a_rb, a_w2 = max(0, (rows - 1) * ars) + max(0, cols - 1), ins.rb, w[1]
    reads_b = func in I.READS_B and bmode != I.B_SCALAR or func == I.V_OUTER
    if reads_b:
        bspan = {I.B_FULL: max(0, (rows - 1) * brs) + max(0, cols - 1),
                 I.B_ROW: max(0, (rows - 1) * brs), I.B_COL: max(0, cols - 1)}.get(bmode, 0)
        if func == I.V_OUTER:
            bspan = max(0, (rows - 1) * brs)
    spans = [dspan, aspan] + ([bspan] if reads_b else [])
    if max(spans) > 0xFFFF:
        return ins                           # a span over the stride fields: keep it as is
    nw = [w[0], a_w2, w[2] if reads_b else 0, 2 | (1 << 16), dspan | (aspan << 16),
          (bspan if reads_b else 0) | (I.V_ADD << 16) | ((I.B_FULL if reads_b else I.B_SCALAR) << 24),
          0]
    return dataclasses.replace(ins, rb=a_rb, rc=ins.rc if reads_b else 0, rd=0, flags=0, w=nw,
                               comment=f"span {ins.comment}")


def span_qact(ins) -> I.Instr:
    w = list(ins.w)
    ab = (w[1] >> 8) & 0xFF
    w[1] = 1 | (ab << 8) | (1 << 16)
    return dataclasses.replace(ins, flags=ins.flags & ~(I.F_DUP | I.F_CSCALE | I.F_RSCALE), w=w,
                               comment=f"span {ins.comment}")


def fuse_apply(prog, mode: str, keep=None):
    """prog with mode's roles spanned (FUSE_MODES' classes, STAGE_ROLES' or FUSION's roles,
    ','-joined; ',Q' adds QACT's spans); returns (program, counts by role)."""
    parts = mode.split(",")
    verd, roles = set(), set()
    for p in parts:
        if p in FUSE_MODES:
            verd |= FUSE_MODES[p]
        elif p in STAGE_ROLES:
            roles |= STAGE_ROLES[p]
        elif p in FUSION:
            roles.add(p)
        elif p and p != "Q":
            raise ValueError(f"fuse mode {mode!r}: unknown part {p!r}")
    q = "Q" in parts
    out, n = [], defaultdict(int)
    for ins in prog:
        if keep is not None and keep(ins):
            out.append(ins)
            continue
        if ins.op == I.VOP:
            rl = role(opkey(ins), site(ins))
            if FUSION[rl][0] in verd or rl in roles:
                out.append(span_vop(ins))
                n[rl] += 1
                continue
        elif ins.op == I.QACT and q:
            out.append(span_qact(ins))
            n["QACT"] += 1
            continue
        out.append(ins)
    return out, dict(n)


MHZ = 133.33


def cosim(cfg, progs, dram, args=None) -> dict:
    """The RTL co-sim of tools/vpu_hack.cosim_case: the board's micro-architecture, LiteDRAM's
    controller at DDR3-1066, the core at 133.33 MHz."""
    from opentpu import rtlsim
    from opentpu.profile import ddr3_plusargs
    plus = rtlsim.ldc_plusargs(1066, MHZ) + ddr3_plusargs(3200 / 3, MHZ)[2:]
    t = time.time()
    kw = {"args": args} if args is not None else {}
    # the epilogue lanes: the card's E at MCOLS 4, E = MCOLS / 2 above (OTPU_MXU_E overrides)
    uarch = dict(rtlsim.BOARD_UARCH)
    if "MXU_E" not in uarch:                 # a tree before the MX column: no lanes to set
        pass
    elif os.environ.get("OTPU_MXU_E"):
        uarch["MXU_E"] = int(os.environ["OTPU_MXU_E"])
    elif cfg.MCOLS >= 8:
        uarch["MXU_E"] = cfg.MCOLS // 2
    _, _, st = rtlsim.run(cfg, progs, [dram], uarch=uarch, axi=True, boot=True,
                          stall=0, bw=100, lat=round(0.3 * MHZ), max_cycles=1 << 40, ldc=1066,
                          plusargs=["+axi_dram=1", "+axi_map=1"] + plus, **kw)
    rec = {"cycles": int(st["cycles"]), "sim_s": round(time.time() - t),
           "E": uarch.get("MXU_E", 0)}
    if st.get("act_reads"):                  # the ACT RAM's column reads (otpu_mxu xa_*)
        rec["act"] = st["act_reads"][0]
    return rec


def cosim_cmd(o) -> None:
    """dataflow.py cosim: a case co-simulated under each mode (one JSON line each)."""
    card_env()
    sys.path.insert(0, str(ROOT / "tools"))
    import vpu_hack as VH
    keep, args = None, None
    if o.kind in ("moe", "moelayer"):
        import moe_prefill_cosim as MP
        kind, M, spec = MP.load_model(o.model)
        cfg = dataclasses.replace(MP.cfg_of(), DRAM_BYTES=MP.DRAM)
        if o.mcols:
            cfg = dataclasses.replace(cfg, MCOLS=o.mcols)
        if o.kind == "moe":                  # a decode row through the prefix (with the head)
            s, img = MP.model_image(kind, M, spec, o.layers, "moe", 1, cfg, 4096)
            progs0 = img.compile_step(o.pos)
        else:                                # the layer-major prefill's program of the last layer
            from opentpu.compiler import arg_words
            from opentpu.llm.qwen3 import RunPos
            s, img = MP.model_image(kind, M, spec, o.layers, "moe", 1, cfg, 4096, lookup=True)
            li, block = s.layers - 1, getattr(img, "block", 256)
            progs0, ra = img.compile_layer_run(li, o.pos // block + 1, block, R=o.n,
                                               embedded=o.n > 1)
            args = arg_words(ra, dict(RunPos.values(791, o.pos, 1, block), row=3))
        keep = VH.moe_keep
        end, area, spec_ = img.nbytes, None, s
    else:
        cfg, progs0, img, area, spec_ = compile_any(o.model, o.kind, o.n, o.pos, o.mcols,
                                                    o.wformat, o.head_format, o.layers,
                                                    VH.COSIM_DRAM)
        end = area["end"] if area else img.nbytes
    for mode in o.modes.split(":"):
        rec = dict(model=Path(o.model).name, kind=o.kind, n=o.n, pos=o.pos, mcols=cfg.MCOLS,
                   layers=o.layers, mode=mode or "base", E=os.environ.get("OTPU_MXU_E", ""))
        if mode == "novop":
            progs = [VH.apply(p, "novop", keep=keep)[0] for p in progs0]
        elif mode:
            res = [fuse_apply(p, mode, keep) for p in progs0]
            progs = [p for p, _ in res]
            rec["spanned"] = res[0][1]
        else:
            progs = progs0
        dram = np.zeros(end, np.uint8)
        if area:
            from opentpu.llm import batch as BA
            BA.build([dram], area, spec_, cfg)
        if o.kind in ("moe", "moelayer"):
            MP.timing_hack(img, dram)
        try:
            rec.update(cosim(cfg, progs, dram, args))
        except Exception as e:                  # a run that fails: say so, go on
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        print(json.dumps(rec), flush=True)


# ---------------------------------------------------------------------------- the report
def per(r) -> int:
    """The divisor that makes a run's totals per token (decode), per row (rows) or per user."""
    c = r["case"]
    return 1 if c["kind"] in ("decode", "moe") else c["n"]


def unit_of(name: str) -> str:
    """A flow end's unit ('VOP.mul@lib.rope<...' -> 'VPU:rope')."""
    if name == "init":
        return "init"
    op, _, st = name.partition("@")
    rl = role(op, st)
    if rl is not None and op.startswith("VOP"):
        return f"VPU:{rl}"
    base = op.split(".")[0]
    return {"MM": "MXU", "QACT": "QACT", "QST": "QST", "LD": "LD", "ST": "ST", "DSTEP": "SE",
            "STREAM": "SE", "RLD": "COLL", "GATHER": "COLL", "WAITW": "DMA"}.get(base, base)


def dram_rows(r) -> dict:
    """DRAM bytes of a run by item (reads and writes), per token / row / user."""
    t, k = r["total"], per(r)
    rd, wr = t["dram_rd"], t["dram_wr"]
    out = {"weights": rd.get("weights", 0), "scales": rd.get("scales", 0),
           "kv": rd.get("kv", 0) + rd.get("kvscale", 0) + rd.get("ld.kv", 0),
           "kv append": wr.get("qst_beats", 0) + rd.get("qst_rmw", 0),
           "state": rd.get("state", 0) + wr.get("state", 0),
           "ld/st": rd.get("ld", 0) + wr.get("st", 0),
           "other": rd.get("waitw", 0) + rd.get("prog", 0)}
    out = {kk: v / k for kk, v in out.items()}
    out["total"] = sum(out.values())
    out["kv append useful"] = wr.get("qst", 0) / k
    return out


def roles(r) -> dict:
    """Per role (FUSION): VOP instructions, VPU / stream-engine issue cycles, TMEM words the
    role's instructions read and write, per token / row / user."""
    k = per(r)
    out = {}
    for c in r["classes"]:
        rl = role(c["op"], c["site"])
        if rl is None:
            continue
        a = out.setdefault(rl, {"n": 0, "cycles": 0, "comp": 0, "rd": 0, "wr": 0})
        a["n"] += c["n"] / k
        a["cycles"] += c["vpu_cycles"] / k
        if c["op"].split(":")[0] in ("VOP.exp2", "VOP.exp2sub", "VOP.recip", "VOP.rsqrt",
                                     "VOP.log2"):
            a["comp"] += c["vpu_cycles"] / k      # the composites' (se_comp's) share
        a["rd"] += sum(c["tmem_rd"].values()) / k
        a["wr"] += sum(c["tmem_wr"].values()) / k
    return out


def fused_ports(r, mode: str) -> dict:
    """TMEM words per read / write port per token / row / user after a fusion mode
    (FUSE_MODES, ',Q' quantize-on-drain), estimated from the ledger: a fused role's VPU / SE
    reads and VPU writes go; the MXU's read-modify-write takes the residual's x (its A reads)
    and RoPE's tables (half its B reads: a pair shares cos and sin); a GLU writes one word for
    its two drained ones; with Q the QACTs whose source a fused role or the MXU wrote no longer
    read TMEM (nor does that source get written, when QACT was its only reader)."""
    k = per(r)
    t = r["total"]
    rd = {p: v / k for p, v in t["tmem_rd"].items()}
    wr = {p: v / k for p, v in t["tmem_wr"].items()}
    parts = mode.split(",")
    verd = set()
    for p in parts:
        if p and p != "Q":
            verd |= FUSE_MODES[p]
    fused = {rl for rl, (v, _) in FUSION.items() if v in verd}
    for c in r["classes"]:
        rl = role(c["op"], c["site"])
        if rl not in fused:
            continue
        for p, v in c["tmem_rd"].items():
            rd[p] = rd.get(p, 0) - v / k
        for p, v in c["tmem_wr"].items():
            wr[p] = wr.get(p, 0) - v / k
        if rl == "residual":
            rd["MXU"] = rd.get("MXU", 0) + c["tmem_rd"].get("VA", 0) / k
        elif rl == "rope":
            rd["MXU"] = rd.get("MXU", 0) + c["tmem_rd"].get("VB", 0) / k / 2
        elif rl == "conv":
            rd["MXU"] = rd.get("MXU", 0) + c["tmem_rd"].get("VB", 0) / k
    if "glu" in fused:                       # gate and up: one word written for two drained
        g = sum(c["tmem_wr"].get("MXU", 0) for c in r["classes"]
                if c["op"] == "MM" and "gate_up" in c["site"]) / k
        wr["MXU"] = wr.get("MXU", 0) - g / 2
    if "Q" in parts:                         # QACT's data reads of drained / fused rows
        src = {"MXU"} | {f"VPU:{rl}" for rl in fused}
        q = sum(v for (w, d), v in flows(r, single=True).items()
                if d == "QACT" and w in src)
        rd["Q"] = rd.get("Q", 0) - q
        # what QACT alone read is not written: the MXU's drained rows, and the fused roles'
        # results (the drain's after fusion; their VPU writes went above)
        wr["MXU"] = wr.get("MXU", 0) - sum(
            v for (w, d), v in flows(r, single=True).items() if d == "QACT" and w in src)
    return {"rd": {p: max(0.0, v) for p, v in rd.items()},
            "wr": {p: max(0.0, v) for p, v in wr.items()}}


def flows(r, single: bool = False) -> dict:
    """TMEM words (writer unit -> reader unit), per token / row / user. single: the words read
    exactly once, by that reader (hand-offs)."""
    k, out = per(r), defaultdict(float)
    if single:
        for w, f, n, v in r["uses"]:
            if n in (1, 11):
                out[(unit_of(w), unit_of(f))] += v / k
    else:
        for w, rd, v in r["flows"]:
            out[(unit_of(w), unit_of(rd))] += v / k
    return dict(out)


def label(r) -> str:
    c = r["case"]
    lay = f" L{c['layers']}" if c.get("layers") else ""
    return f"{c['model']} {c['kind']}{c['n']} @{c['pos']}{lay}"


def load(paths) -> list:
    """Ledger JSONs from paths or glob patterns, in a stable order."""
    import glob
    files = []
    for p in paths:
        files += sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
    return [json.loads(Path(f).read_text()) for f in files]


def report(paths) -> str:
    rs = load(paths)
    L = []
    MB = 1e6
    L.append("DRAM, MB per token / row / user (simulated, ISA ledger)\n")
    keys = ["weights", "scales", "kv", "kv append", "state", "ld/st", "other", "total"]
    L.append("| run | " + " | ".join(keys) + " | KV share |")
    L.append("|---|" + "---:|" * (len(keys) + 1))
    for r in rs:
        d = dram_rows(r)
        L.append(f"| {label(r)} | " + " | ".join(f"{d[k] / MB:.3f}" for k in keys) +
                 f" | {100 * (d['kv'] + d['kv append']) / d['total']:.1f}% |")
    L.append("\nTMEM, KB per token / row / user: reads by port, writes by port; ACT RAM, IMEM\n")
    rp, wp = list(RPORTS), list(WPORTS)
    L.append("| run | " + " | ".join(f"r {p}" for p in rp) + " | " +
             " | ".join(f"w {p}" for p in wp) + " | TMEM | ACT w | ACT r | ACT r phys | IMEM |")
    L.append("|---|" + "---:|" * (len(rp) + len(wp) + 6))
    for r in rs:
        t, k = r["total"], per(r)
        rd = [4 * t["tmem_rd"].get(p, 0) / k / 1e3 for p in rp]
        wr = [4 * t["tmem_wr"].get(p, 0) / k / 1e3 for p in wp]
        L.append(f"| {label(r)} | " + " | ".join(f"{x:.1f}" for x in rd + wr) +
                 f" | {sum(rd) + sum(wr):.1f} | {t['act_wr'] / k / 1e3:.1f} | "
                 f"{t['act_rd'] / k / 1e3:.0f} | {t['act_rd_phys'] / k / 1e3:.0f}"
                 f"{' (gated %.0f)' % (t['act_rd_gated'] / k / 1e3) if 'act_rd_gated' in t else ''} | "
                 f"{t['imem'] / k / 1e3:.1f} |")
    L.append("\nVPU roles: issue cycles and TMEM words (read + written), per token / row / user\n")
    allr = sorted({rl for r in rs for rl in roles(r)}, key=lambda x: list(FUSION).index(x))
    L.append("| run | " + " | ".join(f"{x} ({FUSION[x][0]})" for x in allr) + " |")
    L.append("|---|" + "---:|" * len(allr))
    for r in rs:
        ro = roles(r)
        cyc = sum(v["cycles"] for v in ro.values()) or 1
        L.append(f"| {label(r)} cycles (composites) | " + " | ".join(
            f"{ro[x]['cycles']:.0f} ({ro[x]['comp']:.0f}) {100 * ro[x]['cycles'] / cyc:.0f}%"
            if x in ro else "" for x in allr) + " |")
        L.append(f"| {label(r)} words | " + " | ".join(
            f"{ro[x]['rd'] + ro[x]['wr']:.0f}" if x in ro else "" for x in allr) + " |")
    L.append("\nTMEM producer -> consumer, words per token / row / user (all reads; in brackets "
             "the words read exactly once: hand-offs)\n")
    coarse = lambda u: "VPU" if u.startswith("VPU") else u          # noqa: E731
    pairs = defaultdict(float)
    per_run = []
    for r in rs:
        fa, fs = defaultdict(float), defaultdict(float)
        for (w, d), v in flows(r).items():
            fa[(coarse(w), coarse(d))] += v
        for (w, d), v in flows(r, single=True).items():
            fs[(coarse(w), coarse(d))] += v
        per_run.append((r, fa, fs))
        for k, v in fa.items():
            pairs[k] += v
    top = [k for k, _ in sorted(pairs.items(), key=lambda kv: -kv[1])[:12]]
    L.append("| run | " + " | ".join(f"{w}->{d}" for w, d in top) + " |")
    L.append("|---|" + "---:|" * len(top))
    for r, fa, fs in per_run:
        L.append(f"| {label(r)} | " + " | ".join(
            f"{fa.get(k, 0):.0f} ({fs.get(k, 0):.0f})" for k in top) + " |")
    return "\n".join(L)


# ---------------------------------------------------------------------------- KV options
# docs/dataflow.md 4.8: the KV cache's bytes and the attention MMs' advances today and under the
# layout / format options, estimated from decode / batch ledger runs (MCOLS 4). The attention
# layers' (q heads, KV heads, head dim) of the card models (Gemma 4 E2B: its sliding layers';
# its 512-wide global K rows are whole chunks either way).
KV_HEADS = {"Qwen3-0.6B": (16, 8, 128), "SmolLM3-3B": (16, 4, 128),
            "Phi-4-mini-instruct": (24, 8, 128), "Qwen3.5-0.8B": (8, 2, 256),
            "Qwen3.5-2B": (8, 2, 256), "Qwen3.5-4B": (16, 4, 256), "LFM2.5-230M": (16, 8, 64),
            "LFM2-2.6B": (32, 8, 64), "gemma-4-E2B-it": (8, 1, 256),
            "gemma-4-E4B-it": (8, 2, 256)}


def kv_parts(r) -> dict:
    """A run's KV bytes (K values and scales, V values, V's per-token scales by LD), appends
    (beats written + read-merges; useful bytes; QSTs), attention advances (K, V), all DRAM."""
    p = dict(kv_k=0, ks=0, kv_v=0, vs=0, app=0, app_use=0, app_n=0, adv_k=0, adv_v=0)
    for c in r["classes"]:
        st, op, rd, wr = c["site"], c["op"], c["dram_rd"], c["dram_wr"]
        if op.startswith("MM") and st.startswith("attention.scores"):
            p["kv_k"] += rd.get("kv", 0)
            p["ks"] += rd.get("kvscale", 0)
            p["adv_k"] += c["mxu_adv"]
        elif op.startswith("MM") and st.startswith("attention.finish"):
            p["kv_v"] += rd.get("kv", 0)
            p["adv_v"] += c["mxu_adv"]
        elif op.startswith("LD") and st.startswith("attention."):
            p["vs"] += rd.get("ld.kv", 0)
        if op.startswith("QST"):
            p["app"] += wr.get("qst_beats", 0) + rd.get("qst_rmw", 0)
            p["app_use"] += wr.get("qst", 0)
            p["app_n"] += c["n"]
    t = r["total"]
    p["all"] = sum(t["dram_rd"].values()) + sum(t["dram_wr"].values()) - t["dram_wr"].get("qst", 0)
    return p


def kv_options(heads, p, mcols: int, D: int = 128) -> dict:
    """{option: (KV read bytes, append bytes, K advances, V advances, q.K^T outputs an
    advance)} at MCOLS mcols. A K row takes whole D-byte chunks unless N-pair (two rows a chunk,
    2G <= MCOLS) packs half-chunk rows; fp4x6 K: d/2 value bytes, its scale row Sw + a word a
    block; fp4 V^T (E2M1, UNIT, the per-token scale in P's CSCALE): half the bytes, PAIR halves
    its advances (2G <= MCOLS); V rows: the append writes whole beats (+ a scale beat's merge)."""
    q, kv, d = heads
    g = q // kv
    pair = 2 * g <= mcols
    G4, G = -(-g // 4), -(-g // mcols)            # replay groups (the runs': MCOLS 4)
    p = dict(p, adv_k=p["adv_k"] * G / G4, adv_v=p["adv_v"] * G / G4)
    rows_k = p["adv_k"] / max(1, d // D) / G      # K rows streamed

    def kbytes(rb):                               # a K row of rb bytes: stream bytes, advances
        per = rb if rb >= D else (D / 2 if (pair and rb <= D / 2) else D)
        return rows_k * per, G * rows_k * per / D

    kb = max(1, d // D)
    k8, ak8 = kbytes(d)
    k4, ak4 = kbytes(d // 2)
    if d >= D:                                    # whole-chunk rows
        k8, k4 = p["kv_k"], p["kv_k"] / 2
    if d // 2 >= D and not pair:                  # 4-bit whole-chunk rows: two advances a chunk
        ak4 = 2 * ak4
    v8, vs = p["kv_v"], p["vs"]
    av4 = p["adv_v"] / 2 if pair else p["adv_v"]
    app_rows = p["app_use"] + 128 * p["app_n"]

    def opa(ak):
        return round(rows_k * g / ak, 2)
    o = {"today": (p["kv_k"] + p["ks"] + v8 + vs, p["app"], p["adv_k"], p["adv_v"],
                   opa(p["adv_k"]))}
    o["N-pair int8 K"] = (k8 + p["ks"] + v8 + vs, p["app"], ak8, p["adv_v"], opa(ak8))
    o["V rows"] = (o["today"][0], app_rows, p["adv_k"], p["adv_v"], opa(p["adv_k"]))
    o["fp4 V"] = (o["today"][0] - v8 / 2, p["app"], p["adv_k"], av4, opa(p["adv_k"]))
    # 6-bit (int6 / MXFP6; a proposal): 3/4 of the values' bytes, rows packed in the stream (d
    # 128: 96 bytes a row), scales as today's (MXFP6's E8M0 a 32-block: d / 32 bytes a row);
    # whole-chunk rows (d >= D) scale the run's K bytes, as fp4 K's do
    k6 = p["kv_k"] * 0.75 if d >= D else rows_k * 0.75 * d
    o["6b V"] = (o["today"][0] - v8 / 4, p["app"], p["adv_k"], p["adv_v"], opa(p["adv_k"]))
    o["6b K"] = (o["today"][0] - p["kv_k"] + k6, p["app"], p["adv_k"], p["adv_v"],
                 opa(p["adv_k"]))
    o["6b K+V"] = (o["6b K"][0] - v8 / 4, p["app"], p["adv_k"], p["adv_v"], opa(p["adv_k"]))
    o["fp4 K"] = (k4 + p["ks"] * (2 + kb) + v8 + vs, p["app"], ak4, p["adv_v"], opa(ak4))
    o["fp4 K+V"] = (o["fp4 K"][0] - v8 / 2, p["app"], ak4, av4, opa(ak4))
    o["fp4 K+V, V rows"] = (o["fp4 K+V"][0], app_rows - p["app_use"] / 2, ak4, av4, opa(ak4))
    return o


def kv_report(paths) -> str:
    """dataflow.py kv JSONs...: the KV options of the decode / batch runs."""
    L = ["| model | pos | B | G | option | KV MB / user (MCOLS 8) | change of the step, MCOLS 8 | "
         "MCOLS 4 | attention advances K / user (MCOLS 8) | q.K^T outputs an advance |",
         "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in load(paths):
        c = r["case"]
        heads = c.get("heads") or KV_HEADS.get(c["model"])
        if heads is None or c["kind"] not in ("decode", "batch"):
            continue
        p = kv_parts(r)
        o4, o8 = kv_options(heads, p, 4), kv_options(heads, p, 8)
        B = c["n"] if c["kind"] == "batch" else 1
        t8, t4 = sum(o8["today"][:2]), sum(o4["today"][:2])
        for name in o8:
            b8, b4 = sum(o8[name][:2]), sum(o4[name][:2])
            L.append(f"| {c['model']} | {c['pos']} | {B} | {heads[0] // heads[1]} | {name} | "
                     f"{b8 / B / 1e6:.1f} | {(b8 - t8) / p['all']:+.1%} | "
                     f"{(b4 - t4) / p['all']:+.1%} | {(o8[name][2] + o8[name][3]) / B / 1e3:.0f} | "
                     f"{o8[name][4]} |")
    return "\n".join(L)


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    ap = argparse.ArgumentParser(prog="dataflow.py")
    if a and a[0] == "report":
        print(report(a[1:]))
        return 0
    if a and a[0] == "kv":
        print(kv_report(a[1:]))
        return 0
    if a and a[0] == "cosim":
        ap.add_argument("cmd")
        for k in ("model", "kind"):
            ap.add_argument(k)
        ap.add_argument("n", type=int)
        ap.add_argument("pos", type=int)
        ap.add_argument("--mcols", type=int)
        ap.add_argument("--layers")
        ap.add_argument("--wformat", default="fp4")
        ap.add_argument("--head-format", default="int8")
        ap.add_argument("--modes", default=":A:AS:ASB:C",
                        help="':'-separated: '' (as compiled), novop, or FUSE_MODES with ',Q'")
        cosim_cmd(ap.parse_args(a))
        return 0
    if a and a[0] == "moe":
        ap.add_argument("cmd")
        ap.add_argument("model")
        ap.add_argument("--prefix", required=True)
        ap.add_argument("--pos", type=int, default=544)
        ap.add_argument("--json")
        o = ap.parse_args(a)
        cfg, progs, poke, info = moe_case(o.model, o.prefix, o.pos)
    else:
        for k in ("model", "kind"):
            ap.add_argument(k)
        ap.add_argument("n", type=int)
        ap.add_argument("pos", type=int)
        ap.add_argument("--mcols", type=int)
        ap.add_argument("--layers")
        ap.add_argument("--wformat", default="fp4")
        ap.add_argument("--head-format", default="int8")
        ap.add_argument("--set", default="", help="K=V,... Config fields over the board's")
        ap.add_argument("--json")
        o = ap.parse_args(a)
        sets = {k: int(v) for k, v in (kv.split("=") for kv in o.set.split(",") if kv)}
        cfg, progs, poke, info = case(o.model, o.kind, o.n, o.pos, o.mcols, o.layers, o.wformat,
                                      o.head_format, sets=sets)
        info["sets"] = sets
    info["instructions_static"] = max(len(p) for p in progs)
    r = ledger(cfg, progs, poke=poke)
    r["case"] = info
    if o.json:
        Path(o.json).write_text(json.dumps(r))
    t = r["total"]
    print(json.dumps({**info, "sim_s": r["sim_s"], "instructions": r["instructions"],
                      "dram_rd": t["dram_rd"], "dram_wr": t["dram_wr"],
                      "tmem_rd": t["tmem_rd"], "tmem_wr": t["tmem_wr"], "act_wr": t["act_wr"],
                      "act_rd": t["act_rd"], "imem": t["imem"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
