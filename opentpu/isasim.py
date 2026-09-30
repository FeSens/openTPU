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
    STREAM: bool = False     # the stream engine runs STREAM's hardware subset (docs/stream.md)
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
    128-deep MXU x 4 columns, 8 VPU lanes / TMEM banks, 64K-word TMEM, 128 ACT RAM blocks
    (K <= 16384), 4K-instruction IMEM, 4 GiB DRAM. OTPU_MCOLS in the environment selects the MXU
    column count (default 4, as make -C boards/ypcb-00338 bit builds; 2: bit MCOLS=2), OTPU_LANES the VPU lanes / TMEM banks (default 8; bit
    LANES=16; timing only, the programs do not change), OTPU_PAIR=1 column reuse (MM PAIR /
    QACT DUP), OTPU_DSTEP=1 the DMA's DSTEP, OTPU_STREAM=1 the stream engine, OTPU_ACT_ROWS the ACT RAM rows (default MCOLS;
    more: the MXU replays each weight chunk for MCOLS rows at a time). They configure the
    simulators and the board model; on the card, opentpu.host.board.device_config takes them
    from the bitstream."""
    base = dict(S=1, D=128, MCOLS=int(os.environ.get("OTPU_MCOLS", 4)), ACT_BLOCKS=128,
                LANES=int(os.environ.get("OTPU_LANES", 8)),
                ACT_ROWS=int(os.environ.get("OTPU_ACT_ROWS", 0)), TMEM_WORDS=1 << 16,
                IMEM_WORDS=1 << 15, DRAM_BYTES=1 << 32,
                PAIR=bool(int(os.environ.get("OTPU_PAIR", 0))),
                DSTEP=bool(int(os.environ.get("OTPU_DSTEP", 0))),
                STREAM=bool(int(os.environ.get("OTPU_STREAM", 0))))
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
        self.chain: tuple | None = None       # HALT CHAIN: (DRAM byte address, instructions)
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
            if ins.flags & I.F_CHAIN:
                self.chain = (self.reg(ins.ra), self.reg(ins.rb))
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
        if op == I.RLD:
            v = int(self.tmem[self._tidx(np.int64((self.reg(ins.ra) + w[0]) & 0xFFFFFFFF))])
            if ins.rd:
                self.R[ins.rd] = v if ins.flags & I.F_RAW else I.f2i(v)
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
        if op == I.STREAM:
            return self._stream(ins)
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

    def _stream(self, ins: I.Instr) -> None:
        """STREAM (docs/stream.md, 3): the descriptor, the column slots, the row scalars and the
        constants are read first; row r of the stream is written after row r is read (a later
        row reads what an earlier row wrote); O and the saved registers are written last."""
        w, fl, D = ins.w, ins.flags, self.cfg.D
        at, ks = w[0] & 0xFFFF, w[0] >> 16
        n = min(I.STREAM_DESC_FIXED + I.STREAM_MAX_OPS, self.cfg.TMEM_WORDS - at)
        try:
            d = I.StreamDesc.decode(self.tmem[self._tidx(at + np.arange(max(n, 0)))])
        except (ValueError, IndexError) as e:
            raise SimError(f"STREAM: {e}") from None
        src = (self.reg(ins.ra) + w[1]) & 0xFFFFFFFF
        dst = (self.reg(ins.ra) + w[2]) & 0xFFFFFFFF
        vec = (self.reg(ins.rb) + w[3]) & 0xFFFFFFFF
        xa = (self.reg(ins.rc) + w[4]) & 0xFFFFFFFF
        ka = (self.reg(ins.rd) + w[5]) & 0xFFFFFFFF
        out = w[6]
        rows, cols = d.rows, d.cols
        if not rows or not cols:
            raise SimError("STREAM: rows and cols must be nonzero")
        if d.q_en and d.o_reg_en:
            raise SimError("STREAM: O is either Q's dot or a register")
        if any(op > I.SC_EXP2SUB for op, _, _, _ in d.ops):
            raise SimError("STREAM: bad scalar op")
        uses_g = d.u_mode in (I.U_FMMA, I.U_MUL)
        uses_b = d.u_mode in (I.U_FMMA, I.U_ADD)
        unary = (I.SC_MOV, I.SC_RSQRT, I.SC_RECIP, I.SC_EXP2, I.SC_LOG2)
        opnds = [c for op, _, a, b in d.ops for c in ((a,) if op in unary else (a, b))]
        slots, consts = set(), {j for j in range(4) if d.rinit >> j & 1}
        consts |= {c - I.OP_K0 for c in opnds if I.OP_K0 <= c < I.OP_K0 + 4}
        for on, src_, idx, s_slot, s_const in (
                (d.a_en, d.a_op, d.a_idx, I.A_SLOT, I.A_CONST),
                (uses_g, d.g_src, d.g_idx, I.G_SLOT, I.G_CONST),
                (uses_b, d.b_src, d.b_idx, I.B_SLOT, I.B_CONST)):
            if on and src_ in (s_slot, s_const):
                if idx >= I.STREAM_SLOTS:
                    raise SimError("STREAM: slot or constant index out of range")
                (slots if src_ == s_slot else consts).add(idx)
        if (d.q_en and not d.q_self) or d.out_p2:
            slots.add(d.q_idx)
        # ---- the inputs, before anything is written
        C = {i: self.tget(vec + i * cols + np.arange(cols))[None, :] for i in sorted(slots)}
        K = {j: self.tget(np.array([ka + j * ks])) for j in sorted(consts)}
        X = self.tget(xa + np.arange(rows)) if I.OP_X in opnds else None
        in_t, out_t = bool(fl & I.F_SRC_T), bool(fl & I.F_DST_T)
        zero, nodst = bool(fl & I.F_SZERO), bool(fl & I.F_NODST)
        r, c = np.arange(rows)[:, None], np.arange(cols)[None, :]
        sidx = src + r * (d.srs or cols) + c if in_t else src + 4 * (r * cols + c)
        didx = dst + r * (d.drs or cols) + c if out_t else dst + 4 * (r * cols + c)
        if (not in_t and not zero and src % D) or (not out_t and not nodst and dst % D):
            raise SimError("STREAM: a DRAM stream must be DRAM-chunk aligned")
        if not in_t and not zero:
            sidx = self._widx(sidx)
        if not out_t and not nodst:
            didx = self._widx(didx)
        # rows that read what an earlier row wrote run one at a time
        same = in_t == out_t and not zero and not nodst
        seq = same and not np.array_equal(sidx, didx) and \
            np.intersect1d(sidx, didx).size > 0
        regs = [np.zeros(1, np.float32)] * 8
        for j in range(4):
            if d.rinit >> j & 1:
                regs[4 + j] = K[j]
        written, carried = set(), False
        for op, dst_, a, b in d.ops:
            carried |= any(x < 8 and x not in written and x in {o[1] for o in d.ops}
                           for x in ((a,) if op in unary else (a, b)))
            written.add(dst_)
        one = np.ones(1, np.float32)

        def value(code, A, Xr):
            return (regs[code] if code < 8 else A if code == I.OP_A else Xr if code == I.OP_X
                    else K[code - I.OP_K0] if code < I.OP_ZERO
                    else np.zeros(1, np.float32) if code == I.OP_ZERO else one)

        def scalar(A, Xr):
            for op, dst_, a, b in d.ops:
                x, y = value(a, A, Xr), value(b, A, Xr)
                regs[dst_] = {
                    I.SC_ADD: F.add, I.SC_SUB: F.sub, I.SC_MUL: F.mul, I.SC_MAX: F.fmax,
                    I.SC_MIN: F.fmin, I.SC_MOV: lambda p, q: F.ftz(p),
                    I.SC_RSQRT: lambda p, q: F.rsqrt(p), I.SC_RECIP: lambda p, q: F.recip(p),
                    I.SC_EXP2: lambda p, q: F.exp2(p), I.SC_LOG2: lambda p, q: F.log2(p),
                    I.SC_EXP2SUB: lambda p, q: F.exp2(F.sub(p, q))}[op](x, y)
            return [regs[d.d_reg], regs[d.g_idx & 7], regs[d.b_idx & 7], regs[d.o_reg]]

        def block(S, lo, hi):
            n = hi - lo
            if not d.a_en:
                A = np.zeros(n, np.float32)
            elif d.a_op == I.A_MAX:
                A = F.chain_max(S)
            else:
                opA = {I.A_SLOT: lambda: C.get(d.a_idx), I.A_SELF: lambda: S,
                       I.A_CONST: lambda: K.get(d.a_idx, one)[None, :]}[d.a_op]()
                A = F.rdot(S, opA)
            Xb = X[lo:hi] if X is not None else None
            if carried:
                per = [scalar(A[i:i + 1], Xb[i:i + 1] if Xb is not None else None)
                       for i in range(n)]
                Dv, Gr, Br, Or = (np.concatenate([p[k] for p in per]) for k in range(4))
            else:
                Dv, Gr, Br, Or = (np.broadcast_to(v, (n,)).astype(np.float32)
                                  for v in scalar(A, Xb))
            G = {I.G_REG: lambda: Gr[:, None], I.G_SLOT: lambda: C.get(d.g_idx),
                 I.G_CONST: lambda: K.get(d.g_idx, one)[None, :],
                 I.G_ONE: lambda: one[None, :]}[d.g_src]() if uses_g else None
            B = {I.B_SLOT: lambda: C.get(d.b_idx), I.B_CONST: lambda: K.get(d.b_idx, one)[None, :],
                 I.B_REG: lambda: Br[:, None]}.get(d.b_src, lambda: None)() if uses_b else None
            if uses_b and B is None:
                raise SimError("STREAM: bad B source")
            Y = {I.U_PASS: lambda: F.ftz(S), I.U_FMMA: lambda: F.outer(S, G, Dv[:, None], B),
                 I.U_MUL: lambda: F.mul(S, G),
                 I.U_ADD: lambda: F.add(S, F.mul(Dv[:, None], B))}[d.u_mode]()
            O = (F.rdot(Y, Y if d.q_self else C[d.q_idx]) if d.q_en
                 else Or if d.o_reg_en else None)
            return (F.mul(Y, C[d.q_idx]) if d.out_p2 else Y), O

        def load(lo, hi):
            if zero:
                return np.zeros((hi - lo, cols), np.float32)
            if in_t:
                return self.tget(sidx[lo:hi])
            return self.m32[sidx[lo:hi]].view(np.float32)

        def store(lo, hi, Y):
            if nodst:
                return
            if out_t:
                self.tput(didx[lo:hi], Y)
            else:
                self.m32[didx[lo:hi]] = F.f32(Y).view(np.uint32)

        Os = []
        for lo, hi in ([(i, i + 1) for i in range(rows)] if seq else [(0, rows)]):
            Y, O = block(load(lo, hi), lo, hi)
            store(lo, hi, Y)
            Os.append(O)
        if d.q_en or d.o_reg_en:
            self.tput(out + np.arange(rows), np.concatenate(Os))
        for j in range(4):
            if d.rsave >> j & 1:
                self.tput(np.array([ka + j * ks]), regs[4 + j][-1:])

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
        if func == I.V_ARGMAX:
            # per row the maximum and the index of its first column (+ the integer base w7)
            if rows > 1 and drs < 2:
                raise SimError("VOP ARGMAX: rows > 1 needs drs >= 2 (a pair per row)")
            didx = (dst + np.arange(rows)[:, None] * drs + np.arange(2)[None, :]).reshape(-1)
            ends = np.repeat((np.arange(rows) + 1) * cols - 1, 2)   # after the row's reads
            self._check_hazard(didx, [aidx.reshape(-1)], wpos=ends)
            c = np.argmax(F._key(A), axis=1)
            base = w7 - (1 << 32) if w7 >> 31 else w7
            self.tput(didx, np.stack([F.chain_max(A), F.i2f(c + base)], axis=1).reshape(-1))
            return
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

    def _check_hazard(self, writes: np.ndarray, reads: list, reduce_cols: int = 0,
                      wpos=None) -> None:
        """The RTL processes elements in order; a later element must not read an earlier write.
        wpos: each write's position in the read order (default: the element's own)."""
        n = len(writes)
        pos = np.arange(n) if wpos is None else np.asarray(wpos, np.int64)
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
        self.args = args
        self.chains = 0                  # HALT CHAIN restarts so far
        for s in self.slices:
            s.R = _regs(args)

    def load(self, programs: list[list[I.Instr]], args=None) -> "Machine":
        """Start new programs on the same machine: DRAM, TMEM and ACT RAM are kept (as on the
        board, where the host writes a new program image between launches)."""
        assert len(programs) == self.cfg.S
        self.args = args
        for s, p in zip(self.slices, programs):
            s.prog, s.R, s.pc, s.stack = p, _regs(args), 0, []
            s.halted, s.waiting, s.icount, s.chain = False, None, 0, None
        return self

    def _chain(self, s: Slice) -> None:
        """HALT CHAIN (docs/isa.md): the board loads the next program from the slice's DRAM
        into IMEM and starts it with the run's arguments; TMEM, ACT RAM and DRAM are kept."""
        addr, n = s.chain
        if addr % self.cfg.D or not 0 < n <= self.cfg.IMEM_WORDS // 8:
            raise SimError(f"slice {s.sid}: HALT CHAIN to {addr:#x}, {n} instructions")
        words = s.m32[s._widx(addr + 4 * np.arange(8 * n))]
        s.prog = [I.Instr.decode(words[8 * k:8 * k + 8]) for k in range(n)]
        s.R, s.pc, s.stack, s.halted, s.waiting = _regs(self.args), 0, [], False, None
        s.chain = None
        self.chains += 1

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
                    if s.halted and s.chain is not None:
                        self._chain(s)
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
