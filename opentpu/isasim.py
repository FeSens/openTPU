"""Bit-exact functional simulator of the openTPU ISA (docs/isa.md).

It is the golden model for the RTL: after running the same program on the same DRAM images, the
TMEM and DRAM contents of every slice must be identical bit for bit.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import fp32 as F
from . import isa as I
from . import quant as Q


@dataclass(frozen=True)
class Config:
    S: int = 1               # slices
    D: int = 32              # MXU depth == quantization block (int8 elements)
    MCOLS: int = 8           # MXU columns == max stationary rows
    ACT_BLOCKS: int = 64     # ACT RAM depth in blocks
    TMEM_WORDS: int = 1 << 16
    DRAM_BYTES: int = 1 << 20
    IMEM_WORDS: int = 1 << 16   # 8 words per instruction
    LANES: int = 8           # VPU lanes == TMEM banks (timing only; results do not depend on it)
    PAIR: bool = False       # MM PAIR / QACT DUP: 4-bit MMs of M <= MCOLS/2 rows at full rate
    DSTEP: bool = False      # the DMA runs DSTEP (Gated DeltaNet head steps on DRAM state)
    ACT_ROWS: int = 0        # ACT RAM rows == max stationary rows of one MM (0: MCOLS); more
    #                          than MCOLS: the MXU replays each streamed chunk (docs/isa.md, MM)

    @property
    def act_rows(self) -> int:
        return self.ACT_ROWS or self.MCOLS


def design_config(**kw) -> Config:
    """The Kintex-7 design point: 2 slices, 128-deep MXU (x 8 columns), 16 VPU lanes / TMEM
    banks, 64 ACT RAM blocks (K <= 8192). Override any field with keyword arguments."""
    base = dict(S=2, D=128, MCOLS=8, ACT_BLOCKS=64, LANES=16, DRAM_BYTES=1 << 24)
    base.update(kw)
    return Config(**base)


def board_config(**kw) -> Config:
    """The configuration built for the YPCB-00338 board (xc7k480t, 2 x DDR3, PCIe): one slice,
    128-deep MXU x 2 columns (Qwen3 query groups are 2 rows), 8 VPU lanes / TMEM banks,
    64K-word TMEM, 128 ACT RAM blocks (K <= 16384), 4K-instruction IMEM, 4 GiB DRAM.
    OTPU_MCOLS in the environment selects the MXU column count (default 2; make -C
    boards/ypcb-00338 bit MCOLS=4), OTPU_LANES the VPU lanes / TMEM banks (default 8; bit
    LANES=16; timing only, the programs do not change), OTPU_PAIR=1 column reuse (MM PAIR /
    QACT DUP), OTPU_DSTEP=1 the DMA's DSTEP, OTPU_ACT_ROWS the ACT RAM rows (default MCOLS;
    more: the MXU replays each weight chunk for MCOLS rows at a time). They configure the
    simulators and the board model; on the card, opentpu.host.board.device_config takes them
    from the bitstream."""
    base = dict(S=1, D=128, MCOLS=int(os.environ.get("OTPU_MCOLS", 2)), ACT_BLOCKS=128,
                LANES=int(os.environ.get("OTPU_LANES", 8)),
                ACT_ROWS=int(os.environ.get("OTPU_ACT_ROWS", 0)), TMEM_WORDS=1 << 16,
                IMEM_WORDS=1 << 15, DRAM_BYTES=1 << 32,
                PAIR=bool(int(os.environ.get("OTPU_PAIR", 0))),
                DSTEP=bool(int(os.environ.get("OTPU_DSTEP", 0))))
    base.update(kw)
    return Config(**base)


class SimError(RuntimeError):
    pass


class Slice:
    def __init__(self, cfg: Config, sid: int, program: list[I.Instr], dram: np.ndarray | None):
        self.cfg, self.sid = cfg, sid
        self.prog = program
        self.dram = np.zeros(cfg.DRAM_BYTES, dtype=np.uint8)
        if dram is not None:
            self.dram[: len(dram)] = dram
        self.tmem = np.zeros(cfg.TMEM_WORDS, dtype=np.uint32)
        self.act = np.zeros((cfg.act_rows, cfg.ACT_BLOCKS * cfg.D), dtype=np.int8)
        self.ascale = np.zeros((cfg.act_rows, cfg.ACT_BLOCKS), dtype=np.float32)
        self.R = [0] * 16
        self.pc = 0
        self.stack: list[list[int]] = []
        self.halted = False
        self.waiting: I.Instr | None = None   # blocked on a collective
        self.icount = 0

    # ---------------------------------------------------------------- memory helpers
    @property
    def m32(self) -> np.ndarray:
        return self.dram.view(np.uint32)

    def reg(self, r: int) -> int:
        return 0 if r == 0 else self.R[r]

    def _widx(self, byte_addr: np.ndarray | int) -> np.ndarray:
        a = np.asarray(byte_addr, dtype=np.int64)
        if np.any(a % 4) or np.any(a < 0) or np.any(a + 4 > self.cfg.DRAM_BYTES):
            raise SimError(f"slice {self.sid}: bad DRAM word address")
        return a // 4

    def _tidx(self, idx: np.ndarray) -> np.ndarray:
        if np.any(idx < 0) or np.any(idx >= self.cfg.TMEM_WORDS):
            raise SimError(f"slice {self.sid}: TMEM index out of range")
        return idx

    def tget(self, idx) -> np.ndarray:
        return self.tmem[self._tidx(np.asarray(idx, dtype=np.int64))].view(np.float32)

    def tput(self, idx, vals) -> None:
        self.tmem[self._tidx(np.asarray(idx, dtype=np.int64))] = F.f32(vals).view(np.uint32)

    # ---------------------------------------------------------------- execution
    def step(self) -> None:
        """Execute one instruction (or mark the slice as waiting on a collective)."""
        if self.pc >= len(self.prog):
            raise SimError(f"slice {self.sid}: pc {self.pc} past end of program")
        ins = self.prog[self.pc]
        op = ins.op
        self.icount += 1
        if op == I.HALT:
            self.halted = True
            return
        if op in (I.BAR, I.GATHER):
            self.waiting = ins
            return
        if op == I.LOOP:
            count = (self.reg(ins.ra) + ins.w[1]) & 0xFFFFFFFF
            body = ins.w[0]
            if body < 1:
                raise SimError("LOOP with empty body")
            if count == 0:
                self.pc += 1 + body
                return
            if len(self.stack) >= 4:
                raise SimError("loop stack overflow")
            end = self.pc + body
            if any(e[1] == end for e in self.stack):
                raise SimError("nested loop bodies end on the same instruction")
            self.stack.append([self.pc + 1, end, count])
            self.pc += 1
            return
        self.execute(ins)
        self.advance()

    def advance(self) -> None:
        pc = self.pc
        if self.stack and self.stack[-1][1] == pc:
            top = self.stack[-1]
            if top[2] > 1:
                top[2] -= 1
                self.pc = top[0]
                return
            self.stack.pop()
        self.pc = pc + 1

    def execute(self, ins: I.Instr) -> None:
        op, w = ins.op, ins.w
        if op == I.NOP:
            return
        if op == I.LI:
            if ins.rd:
                self.R[ins.rd] = w[0]
            return
        if op == I.ADDI:
            if ins.rd:
                self.R[ins.rd] = (self.reg(ins.ra) + w[0]) & 0xFFFFFFFF
            return
        if op == I.LD:
            d = (self.reg(ins.ra) + w[0]) & 0xFFFFFFFF
            t = (self.reg(ins.rb) + w[1]) & 0xFFFFFFFF
            n = w[2]
            wi = self._widx(d + 4 * np.arange(n))
            self.tmem[self._tidx(t + np.arange(n))] = self.m32[wi]
            return
        if op == I.ST:
            d = (self.reg(ins.ra) + w[0]) & 0xFFFFFFFF
            t = (self.reg(ins.rb) + w[1]) & 0xFFFFFFFF
            n = w[2]
            wi = self._widx(d + 4 * np.arange(n))
            self.m32[wi] = self.tmem[self._tidx(t + np.arange(n))]
            return
        if op == I.DSTEP:
            return self._dstep(ins)
        if op == I.MM:
            return self._mm(ins)
        if op == I.QACT:
            return self._qact(ins)
        if op == I.QST:
            return self._qst(ins)
        if op == I.VOP:
            return self._vop(ins)
        raise SimError(f"slice {self.sid}: bad opcode {op:#x} at pc {self.pc}")

    def _dstep(self, ins: I.Instr) -> None:
        """DSTEP: every input (q, k, v, e, beta and each state row) is read before the row's
        results are written; o is written last."""
        w = ins.w
        d = (self.reg(ins.ra) + w[0]) & 0xFFFFFFFF
        qk = (self.reg(ins.rb) + w[1]) & 0xFFFFFFFF
        v = (self.reg(ins.rc) + w[2]) & 0xFFFFFFFF
        rows, cols = w[3] & 0xFFFF, w[3] >> 16
        g, o, gs = w[4], w[5], w[6]
        if not (0 < rows <= I.DSTEP_MAX_ROWS) or cols % 64 or not (0 < cols <= 256):
            raise SimError("DSTEP: rows must be 1..256, cols 64, 128, 192 or 256")
        if d % self.cfg.D:
            raise SimError("DSTEP: the state must be DRAM-chunk aligned")
        q = self.tget(qk + np.arange(cols))[None, :]
        k = self.tget(qk + cols + np.arange(cols))[None, :]
        vv = self.tget(v + np.arange(rows))
        e = self.tget(np.array([g]))
        beta = self.tget(np.array([g + gs]))
        wi = self._widx(d + 4 * np.arange(rows * cols))
        if ins.flags & I.F_DZERO:
            S = np.zeros((rows, cols), np.float32)
        else:
            S = self.m32[wi].view(np.float32).reshape(rows, cols)
        kv = F.rdot(S, k)
        dd = F.mul(F.sub(vv, F.mul(kv, e)), beta)
        S = F.outer(S, e[None, :], dd[:, None], k)
        oo = F.rdot(S, q)
        self.m32[wi] = F.f32(S).reshape(-1).view(np.uint32)
        self.tput(o + np.arange(rows), oo)

    # ---------------------------------------------------------------- MXU
    def _mm(self, ins: I.Instr) -> None:
        cfg, w = self.cfg, ins.w
        D = cfg.D
        sa = (self.reg(ins.ra) + w[0]) & 0xFFFFFFFF
        ssa = (self.reg(ins.rb) + w[1]) & 0xFFFFFFFF
        out = (self.reg(ins.rc) + w[2]) & 0xFFFFFFFF
        N, KB = w[3] & 0xFFFF, w[3] >> 16
        rs = w[4]
        ors, M, ab = w[5] & 0xFFFF, (w[5] >> 16) & 0xFF, w[5] >> 24
        srs = w[6]
        unit, accf = bool(ins.flags & I.F_UNIT), bool(ins.flags & I.F_ACC)
        pair = bool(ins.flags & I.F_PAIR)
        R = 2 * M if pair else M                          # ACT rows read
        if not (0 < R <= cfg.act_rows) or ab + KB > cfg.ACT_BLOCKS:
            raise SimError("MM: M or ACT RAM range out of bounds")
        if M > cfg.MCOLS and ins.flags & (I.F_RMAX | I.F_ASCALE):
            raise SimError("MM: RMAX and ASCALE need M <= MCOLS")
        if pair and R > cfg.MCOLS:
            raise SimError("MM: PAIR needs 2*M <= MCOLS")
        wf = (ins.flags >> I.WF_SHIFT) & 3
        if wf not in (I.W8, I.W4I, I.W4F):
            raise SimError("MM: bad weight format")
        if pair and wf == I.W8:
            raise SimError("MM: PAIR needs 4-bit weights")
        if pair and not unit and (ssa % 8 or srs % 8):
            raise SimError("MM: PAIR reads a chunk's two scale words at once: ssa and srs must "
                           "be multiples of 8")
        bpb = D if wf == I.W8 else D // 2                 # streamed bytes per K-block
        if sa % D or rs % D or sa + (N - 1) * rs + -(-KB * bpb // D) * D > cfg.DRAM_BYTES:
            raise SimError("MM: streamed rows must be D-byte aligned and in range")
        wv = np.lib.stride_tricks.as_strided(self.dram[sa:], (N, KB, bpb),
                                             (rs, bpb, 1))               # [N, KB, bpb], no copy
        if unit:
            wsw = None
        else:
            wsi = self._widx(ssa + np.arange(N)[:, None] * srs + 4 * np.arange(KB)[None, :])
            wsw = self.m32[wsi]
        act = self.act[:R, ab * D:(ab + KB) * D].reshape(R, KB, D).astype(np.float64)
        if wf == I.W8:
            ws = np.ones((N, KB), np.float32) if unit else wsw.view(np.float32)
            # exact int32 block dot products (|sum| <= D * 127 * 128), via float64 BLAS
            isum = np.einsum("jki,nki->jnk", act, wv.view(np.int8).astype(np.float64),
                             optimize=True).astype(np.int64)               # [M, N, KB]
        else:
            # 4-bit: integer elements, four sub-block sums scaled by their multipliers m_b
            # (exact: |isum| <= 15 * D * 127 * 12 < 2^24), then the block's bf16 scale
            w = Q.DEC["int4" if wf == I.W4I else "fp4"][Q.unpack4(wv)].astype(np.float64)
            if unit:
                ws = np.ones((N, KB), np.float32)
                mb = np.ones((N, KB, Q.NSUB))
            else:
                ws = F.ftz((wsw << np.uint32(16)).view(np.float32))
                mb = np.stack([(wsw >> np.uint32(16 + 4 * b)) & np.uint32(15)
                               for b in range(Q.NSUB)], -1).astype(np.float64)
            sub = np.einsum("jkbi,nkbi->jnkb", act.reshape(R, KB, Q.NSUB, D // Q.NSUB),
                            w.reshape(N, KB, Q.NSUB, D // Q.NSUB), optimize=True)
            isum = np.einsum("jnkb,nkb->jnk", sub, mb).astype(np.int64)   # [R, N, KB]
        t = F.mul(F.i2f(isum), ws[None, :, :])                            # [R, N, KB]
        t = F.mul(t, self.ascale[:R, ab:ab + KB][:, None, :])
        if pair:
            # column reuse: row j's even blocks from ACT row j, its odd blocks from ACT row
            # j + M; the two terms of a chunk are added (+0 for a missing odd block) first
            ev, od = t[:M, :, 0::2], t[M:, :, 1::2]
            if od.shape[2] < ev.shape[2]:
                od = np.concatenate([od, np.zeros((M, N, 1), np.float32)], axis=2)
            t = F.add(ev, od)
        acc = F.interleaved_sum(t.reshape(M * N, t.shape[2]), F.MM_PARTIALS).reshape(M, N)
        idx = out + np.arange(M)[:, None] * ors + np.arange(N)[None, :]
        if ins.flags & I.F_ASCALE:                 # y = old * alpha[j] + acc
            if not (unit and accf):
                raise SimError("MM: ASCALE needs UNIT and ACC")
            alpha = self.tget(ssa + np.arange(M))[:, None]
            acc = F.add(F.mul(self.tget(idx), alpha), acc)
        elif accf:
            acc = F.add(self.tget(idx), acc)
        self.tput(idx, acc)
        if ins.flags & I.F_RMAX:                   # row max of the written values, in n order
            mx = F.chain_max(acc)
            self.tput(out + M * ors + np.arange(M), mx)

    # ---------------------------------------------------------------- quantizer
    def _quant_groups(self, src, rows, KB, srs, row_mode, cscale=None, rscale=None):
        D = self.cfg.D
        idx = src + np.arange(rows)[:, None] * srs + np.arange(KB * D)[None, :]
        x = self.tget(idx)                                                  # [rows, KB*D]
        if rscale is not None:                     # QACT RSCALE: x * T[rscale + r]
            x = F.mul(x, self.tget(rscale + np.arange(rows))[:, None])
        if cscale is not None:                     # QACT CSCALE: x * T[cscale + c]
            x = F.mul(x, self.tget(cscale + np.arange(KB * D))[None, :])
        if row_mode:
            q, s = F.quantize(x, axis=1)                                    # s: [rows]
            s = np.repeat(s[:, None], KB, axis=1)
        else:
            q, s = F.quantize(x.reshape(rows, KB, D), axis=2)               # s: [rows, KB]
            q = q.reshape(rows, KB * D)
        return q, s.astype(np.float32)

    def _qact(self, ins: I.Instr) -> None:
        cfg, w = self.cfg, ins.w
        src = (self.reg(ins.ra) + w[0]) & 0xFFFFFFFF
        rows, ab, KB = w[1] & 0xFF, (w[1] >> 8) & 0xFF, w[1] >> 16
        srs = w[2]
        dup = bool(ins.flags & I.F_DUP)
        if (2 * rows > cfg.MCOLS if dup else rows > cfg.act_rows) or ab + KB > cfg.ACT_BLOCKS:
            raise SimError("QACT out of ACT RAM bounds")
        cs = (w[3] & 0xFFFFFFFF) if ins.flags & I.F_CSCALE else None
        rsc = (w[4] & 0xFFFFFFFF) if ins.flags & I.F_RSCALE else None
        q, s = self._quant_groups(src, rows, KB, srs, bool(ins.flags & I.F_ROW), cs, rsc)
        for r0 in ((0, rows) if dup else (0,)):     # DUP: rows r and r + rows, same bytes
            self.act[r0:r0 + rows, ab * cfg.D:(ab + KB) * cfg.D] = q
            self.ascale[r0:r0 + rows, ab:ab + KB] = s

    def _qst(self, ins: I.Instr) -> None:
        cfg, w = self.cfg, ins.w
        src = (self.reg(ins.ra) + w[0]) & 0xFFFFFFFF
        dst = (self.reg(ins.rb) + w[1]) & 0xFFFFFFFF
        sdst = (self.reg(ins.rc) + w[2]) & 0xFFFFFFFF
        rows, KB = w[3] & 0xFFFF, w[3] >> 16
        srs, drs, es = w[4], w[5], w[6]
        row_mode = bool(ins.flags & I.F_ROW)
        half = bool(ins.flags & I.F_HALF)
        if half and not row_mode:
            raise SimError("QST: HALF needs ROW mode")
        q, s = self._quant_groups(src, rows, KB, srs, row_mode)
        ne = KB * cfg.D // 2 if half else KB * cfg.D   # HALF: the row's first half only
        baddr = dst + np.arange(rows)[:, None] * drs + np.arange(ne)[None, :] * es
        if np.any(baddr < 0) or np.any(baddr >= cfg.DRAM_BYTES):
            raise SimError("QST: byte address out of range")
        self.dram[baddr] = q[:, :ne].view(np.uint8)
        if row_mode:
            self.m32[self._widx(sdst + 4 * np.arange(rows))] = s[:, 0].view(np.uint32)
        else:
            sa = sdst + 4 * (np.arange(rows)[:, None] * KB + np.arange(KB)[None, :])
            self.m32[self._widx(sa)] = s.view(np.uint32)

    # ---------------------------------------------------------------- VPU
    def _vop(self, ins: I.Instr) -> None:
        w = ins.w
        dst = (self.reg(ins.ra) + w[0]) & 0xFFFFFFFF
        a = (self.reg(ins.rb) + w[1]) & 0xFFFFFFFF
        b = (self.reg(ins.rc) + w[2]) & 0xFFFFFFFF
        rows, cols = w[3] & 0xFFFF, w[3] >> 16
        drs, ars = w[4] & 0xFFFF, w[4] >> 16
        brs, func, bmode = w[5] & 0xFFFF, (w[5] >> 16) & 0xFF, (w[5] >> 24) & 3
        w7 = (self.reg(ins.rd) + w[6]) & 0xFFFFFFFF
        imm = np.array([w7], dtype=np.uint32).view(np.float32)[0]
        r = np.arange(rows)[:, None]
        c = np.arange(cols)[None, :]
        if func == I.V_OUTER:
            return self._outer(ins, dst, a, b, w7, rows, cols, drs, brs, bmode)
        aidx = a + r * ars + c
        A = self.tget(aidx) if func != I.V_FILL else None
        bidx = None
        if func in I.READS_B:
            if bmode == I.B_FULL:
                bidx = b + r * brs + c
            elif bmode == I.B_ROW:
                bidx = np.broadcast_to(b + r * brs, (rows, cols))
            elif bmode == I.B_COL:
                bidx = np.broadcast_to(b + c, (rows, cols))
            B = self.tget(bidx) if bidx is not None else np.full((rows, cols), imm, np.float32)
        if func in I.REDUCE:
            didx = dst + np.arange(rows) * drs
            reads = [aidx.reshape(-1)] + ([bidx.reshape(-1)] if bidx is not None else [])
            self._check_hazard(np.repeat(didx, cols), reads, reduce_cols=cols)
            if func == I.V_RSUM:
                acc = F.interleaved_sum(A, F.RED_PARTIALS)
            elif func == I.V_RSSQ:
                acc = F.rdot(A, A)
            elif func == I.V_RDOT:
                acc = F.rdot(A, B)
            else:
                acc = F.chain_max(A)
            self.tput(didx, acc)
            return
        didx = dst + r * drs + c
        reads = ([aidx.reshape(-1)] if func != I.V_FILL else []) + \
            ([bidx.reshape(-1)] if bidx is not None else [])
        self._check_hazard(didx.reshape(-1), reads)
        ops = {I.V_ADD: lambda: F.add(A, B), I.V_SUB: lambda: F.sub(A, B),
               I.V_RSUB: lambda: F.sub(B, A), I.V_MUL: lambda: F.mul(A, B),
               I.V_MAX: lambda: F.fmax(A, B), I.V_MIN: lambda: F.fmin(A, B),
               I.V_COPY: lambda: F.ftz(A), I.V_EXP2: lambda: F.exp2(A),
               I.V_RECIP: lambda: F.recip(A), I.V_RSQRT: lambda: F.rsqrt(A),
               I.V_ABS: lambda: F.fabs(A), I.V_FILL: lambda: F.ftz(B),
               I.V_EXP2SUB: lambda: F.exp2(F.sub(A, B)), I.V_LOG2: lambda: F.log2(A)}
        if func not in ops:
            raise SimError(f"VOP: bad func {func}")
        self.tput(didx, ops[func]())

    def _outer(self, ins, dst, d, b, cv, rows, cols, drs, brs, bmode) -> None:
        """dst = dst * Dv + B(r) * Cv(c). Cv and Dv are read before anything is written (the
        RTL buffers them), so they may overlap dst; B(r) is read with every element."""
        if bmode != I.B_ROW or cols > I.OUTER_MAX_COLS:
            raise SimError("VOP OUTER: bmode must be ROW and cols <= 256")
        r = np.arange(rows)[:, None]
        c = np.arange(cols)[None, :]
        C = self.tget(cv + c)
        if ins.flags & I.F_DONE:
            Dv = np.ones((1, cols), np.float32)
        else:
            Dv = self.tget(np.broadcast_to(d if ins.flags & I.F_DSCALAR else d + c, (1, cols)))
        didx = dst + r * drs + c
        bidx = np.broadcast_to(b + r * brs, (rows, cols))
        self._check_hazard(didx.reshape(-1), [didx.reshape(-1), bidx.reshape(-1)])
        self.tput(didx, F.outer(self.tget(didx), Dv, self.tget(bidx), C))

    def _check_hazard(self, writes: np.ndarray, reads: list, reduce_cols: int = 0) -> None:
        """The RTL processes elements in order; a later element must not read an earlier write."""
        n = len(writes)
        pos = np.arange(n)
        if reduce_cols:     # a reduction writes its row after reading the whole row
            pos = (pos // reduce_cols + 1) * reduce_cols - 1
        writes = np.asarray(writes, np.int64)
        order = np.argsort(writes, kind="stable")
        uw, first = np.unique(writes[order], return_index=True)
        first_pos = pos[order][first]                  # earliest write position per address
        for rd in reads:
            rd = np.asarray(rd, np.int64)
            loc = np.minimum(np.searchsorted(uw, rd), len(uw) - 1)
            hit = uw[loc] == rd
            late = hit & (first_pos[loc] < np.arange(len(rd)))
            if late.any():
                v = int(rd[np.argmax(late)])
                raise SimError(f"slice {self.sid}: VOP read-after-write hazard at TMEM {v}")


ARG0 = 8                    # the run's arguments are R8..R15 at the start (docs/isa.md)


def _regs(args) -> list[int]:
    args = list(args or [])
    if len(args) > 8:
        raise ValueError("at most 8 run arguments")
    R = [0] * 16
    for k, v in enumerate(args):
        R[ARG0 + k] = int(v) & 0xFFFFFFFF
    return R


class Machine:
    """args: the run's arguments (up to 8 words): R8..R15 start with them (0 without)."""

    def __init__(self, cfg: Config, programs: list[list[I.Instr]], drams: list[np.ndarray | None],
                 args=None):
        assert len(programs) == cfg.S and len(drams) == cfg.S
        self.cfg = cfg
        self.slices = [Slice(cfg, s, programs[s], drams[s]) for s in range(cfg.S)]
        for s in self.slices:
            s.R = _regs(args)

    def load(self, programs: list[list[I.Instr]], args=None) -> "Machine":
        """Start new programs on the same machine: DRAM, TMEM and ACT RAM are kept (as on the
        board, where the host writes a new program image between launches)."""
        assert len(programs) == self.cfg.S
        for s, p in zip(self.slices, programs):
            s.prog, s.R, s.pc, s.stack = p, _regs(args), 0, []
            s.halted, s.waiting, s.icount = False, None, 0
        return self

    def run(self, max_steps: int = 10_000_000) -> "Machine":
        steps = 0
        while not all(s.halted for s in self.slices):
            progressed = False
            for s in self.slices:
                while not s.halted and s.waiting is None:
                    s.step()
                    steps += 1
                    progressed = True
                    if steps > max_steps:
                        raise SimError("step limit exceeded")
            live = [s for s in self.slices if not s.halted]
            if live and all(s.waiting is not None for s in live):
                if len(live) != len(self.slices):
                    raise SimError("collective with a halted slice")
                ops = {s.waiting.op for s in live}
                if len(ops) != 1:
                    raise SimError("slices disagree on the collective instruction")
                if ops == {I.GATHER}:
                    self._gather()
                for s in live:
                    s.waiting = None
                    s.advance()
                progressed = True
            if not progressed:
                raise SimError("deadlock")
        return self

    def _gather(self) -> None:
        sl = self.slices
        ws = [s.waiting.w for s in sl]
        key = [(w[1], w[2], w[4], w[5]) for w in ws]
        if any(k != key[0] for k in key):
            raise SimError("GATHER: slices disagree on dst/rows/cols/drs/seg")
        vals = []
        for s in sl:
            w = s.waiting.w
            src = (s.reg(s.waiting.ra) + w[0]) & 0xFFFFFFFF
            rows, cols, srs = w[2] & 0xFFFF, w[2] >> 16, w[3]
            idx = src + np.arange(rows)[:, None] * srs + np.arange(cols)[None, :]
            vals.append(s.tmem[s._tidx(idx)].copy())
        for s in sl:
            w = s.waiting.w
            dst = (s.reg(s.waiting.rb) + w[1]) & 0xFFFFFFFF
            rows, cols, drs, seg = w[2] & 0xFFFF, w[2] >> 16, w[4], w[5]
            for k, v in enumerate(vals):
                idx = dst + k * seg + np.arange(rows)[:, None] * drs + np.arange(cols)[None, :]
                s.tmem[s._tidx(idx)] = v
