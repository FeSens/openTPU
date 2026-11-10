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
    SEQ_WC: bool = False     # EXPERIMENTAL: the sequencer takes each program's wait entries
    #                          (WAITC, docs/isa.md "Wait counts"): the host writes them after
    #                          every program (opentpu.waits), the chain slots hold them. The
    #                          programs do not change; the ISA simulator ignores it
    WG: int = 1              # weight groups: an MM with GRP streams its operand in groups of WG
    #                          rows, a group's chunk c (WG x D bytes) a step (docs/isa.md "Weight
    #                          groups"); 1: no GRP, today's machine
    CHASH: bool = False      # the memory's channel hash (CAPS bit7) at WG > 1: each group
    #                          chunk's rows are stored permuted by its address (docs/isa.md
    #                          "Weight groups", quant.hash_rows); off: in order (h = 0)
    GALL: bool = False       # EXPERIMENTAL (docs/weight_groups.md, N): at WG > 1
    #                          every MM in groups (no rows of one): the KV caches in groups,
    #                          their appends QST GRP, grouped MMs with RMAX and ASCALE; the
    #                          RTL's at WG > 1 (a target's and the board's Config: WG > 1)
    OUTER: bool = True       # the VPU runs VOP OUTER; off (CAPS2 bit1): a VPU built without it
    #                          (slot 0 a*b + c, no column buffers: the WG targets', whose DeltaNet runs
    #                          DSTEP), on which the compiler and this simulator refuse OUTER
    DSTEP2: bool = False     # DSTEP's pair (flag PAIR, CAPS2 bit2): a pair's two heads in one
    #                          stream (the WG targets'); off, the compiler and this simulator refuse it
    SE_OWN: bool = False     # the streams on an engine of their own beside the VPU (CAPS2 bit3,
    #                          a target's [uarch] SE_R > 0 with DSTEP): the VPU runs VOPs while a
    #                          state steps, so the compiler gives each DeltaNet pair a burst of
    #                          its own (qwen35._bursts). The programs' order only: the same
    #                          results, and the same image (opentpu/imgcache.py PROGRAM_ONLY)
    KT: bool = False         # MM's K tail (rs bit 0 under GRP, CAPS2 bit5; docs/isa.md "K tail"):
    #                          a pair of groups shares the chunk of their last K-block's halves;
    #                          off, the compiler pads that block and this simulator refuses KT
    AW: int = 32             # DRAM address bits (docs/isa.md "Address bits"): the registers and
    #                          every DRAM address R[x] + w are AW bits (TMEM indices and the
    #                          32-bit fields the low 32), LI / ADDI take {w2, w1}, RLD MUL's
    #                          product is AW bits; 32: today's machine (w2 ignored), the BCU 36
    ICACHE: bool = False     # IMEM is an instruction cache over the program in DRAM (otpu_seq
    #                          ICACHE, CAPS2 bit12; docs/icache.md): a program runs from where
    #                          the host wrote it, IMEM_WORDS its longest (the DRAM's, not IMEM's:
    #                          the 4096-instruction cap of the others lifts). The programs and
    #                          the ISA simulator do not change
    BHW: int = 0             # the memory's bank-half window: the slice's first 2^BHW bytes,
    #                          the BCU1525's DIMMs' bank-high bit its top bit (otpu_mig_pair BHW,
    #                          a channel's bytes: BHW - log2 of the slice's channels; 0: none):
    #                          the images' layout only (what a decode writes in its lower half,
    #                          the DeltaNet mixers' weights in its upper, every other weight
    #                          above it: no bank of the writes' is a mixer weight's)
    WC_WIN: int = 16         # SEQ_WC: the sequencer's window the wait entries assume (16 or 32;
    #                          a target's SEQ_WC with [uarch] WIN 32: 32, opentpu.waits): at 32
    #                          an in-order unit's count saturates at 15. The entries only: the
    #                          programs and results do not change
    ST16: str = ""           # DSTEP's (STREAM's) DRAM state in 16 bits with its flag ST16
    #                          (CAPS2 bits 14:13, otpu_dma ST16; docs/isa.md "ST16"): "e6m9" or
    #                          "bf16", the values rounded to nearest even as they are written
    #                          (st16_pack), fp32 in the step; "": none, the flag refused
    FSUM: bool = False       # VOP EXP2SUB's fused row sums (flag FSUM, CAPS2 bit16; docs/isa.md
    #                          VOP): T[imm + r] = isum_64 of row r's results, written by its take
    #                          (an RSUM with the flag), so a softmax's sum needs no RSUM pass;
    #                          off, the compiler emits the RSUM and this simulator refuses the flag
    SPLITQ: bool = False     # QACT / QST flag LO (CAPS2 bit17, otpu_quant SPLIT; docs/isa.md
    #                          "Quantization"): the split's lo part, q8lo(x') with the scale
    #                          S * 2^-8, so hi + lo / 256 is x' to 2^-9 (int8 x int8 splits for
    #                          chunked DeltaNet's matmuls); off, the flag refused
    QHL: bool = False        # with SPLITQ, QACT flag HL (CAPS2 bit21, otpu_quant HL; docs/isa.md
    #                          "Quantization"): q into blocks ab .. and lo into ab + KB .. from
    #                          one read (a split operand's two QACTs in one's cycles); off, the
    #                          flag refused
    QSTHL: bool = False      # with SPLITQ, QST HL (w5 bit31; CAPS2 bit22, otpu_quant QSTHL; docs/
    #                          isa.md QST): a GRP ROW QST's hi and lo parts from one read, lo's
    #                          bytes and scales HLD KiB after hi's; off, the bit refused
    ST16S: bool = False      # with ST16, STREAM's flags ST16R / ST16W (CAPS2 bit18, otpu_dma
    #                          ST16S; docs/isa.md "ST16"): one end alone 16-bit, the other fp32
    #                          (a chunked prefill's state conversions); off, the flags refused
    VSIG: bool = False       # VOP SIGMOID and SILU (CAPS2 bit19; docs/isa.md VOP): recip(exp2(A*B)
    #                          + 1) and A times it in one VOP (the VPU's composites at VPU_CNS 5,
    #                          four passes), bit for bit the VOPs MUL, EXP2, ADD 1.0, RECIP (MUL);
    #                          off, the compiler emits those and this simulator refuses them
    VMAC: bool = False       # VOP MAC (CAPS2 bit20; docs/isa.md VOP): dst += A * B in place, B per
    #                          column (a column buffer, cols <= 256) or a scalar, the MUL then
    #                          ADD bit for bit in one pass (the DeltaNet's convolution); off, the
    #                          compiler emits the MUL and ADD and this simulator refuses it
    OUTER1: bool = False     # with OUTER off and VMAC, VOP OUTER of decay one (flag DONE; CAPS2
    #                          bit23, otpu_vpu OUTER1: MAC's datapath, docs/isa.md VOP OUTER),
    #                          the chunked DeltaNet core's rank-one steps; an OUTER with a decay
    #                          word still refused, by the compiler and by this simulator
    QONCE: bool = False      # quantize-once (CAPS2 bit24; docs/isa.md "Quantize once"): QACT flag
    #                          ST (bit6) also writes each row's ACT bytes and records to DRAM, and
    #                          LDA (DMA, opcode 0x14) loads such rows into ACT RAM unchanged (the
    #                          expert run's rows quantized once, in the MoE prologue); off, the
    #                          flag and the opcode refused
    SPLMM: bool = False      # MM SPL (w7 bit31; CAPS2 bit25, otpu_mxu SPL; docs/isa.md "Split
    #                          terms"): a split's three terms (KB 3) or a pair's two (KB 2) as one
    #                          MM, the weights' lo part HLD KiB after hi, the lo terms' offsets
    #                          +8 (one epilogue: the chunked DeltaNet core's MMs); off, the bit
    #                          refused
    SEXACT: bool = False     # a compile choice, not the hardware's (never in the CFG_HASH): at
    #                          S > 1 Qwen3.5's images S-exact (qwen35.s_exact_on: S = 1's
    #                          DeltaNet head groups and dense MLP chunks over the slices, so the
    #                          programs give S = 1's results bit for bit); OTPU_SEXACT=0 / 1
    #                          overrides it. A target's default: bcu1525-s2's on, the others' off
    KVSPLIT: int = 0         # a compile choice too (never in the CFG_HASH): decode at a bucket of
    #                          KVSPLIT or more attention blocks splits a KV head copied on r = S /
    #                          n_kv slices (n_kv < S) by block over its copies, each attending
    #                          its blocks for the r query groups, the partial softmax states
    #                          combined (qwen3._attention; reads of the cache halved at r = 2,
    #                          its appends on every copy as before). 0: never. OTPU_KVSPLIT=N
    #                          overrides it. A target's default: bcu1525-x2's on
    DNMERGE: int = 0         # a compile choice too (never in the CFG_HASH): a chunked DeltaNet
    #                          layer's prep and post tiles merged, this many streams at a time
    #                          (tiled.dn_merged, opentpu/stage.py); 0: tile by tile
    DNCHUNK: int = 0         # a compile choice too (never in the CFG_HASH): a Qwen3.5 prefill
    #                          chunk's DeltaNet layers by row blocks, chunks of DNCHUNK rows
    #                          (docs/deltanet_chunk.md: prep tiles to the
    #                          image's dnq, the chunked core, post tiles). 0: the recurrent rows

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
    """The configuration of the target the environment selects (opentpu.target.current():
    OTPU_TARGET, default the YPCB-00338 card, targets/ypcb-00338.toml: one slice, 128-deep MXU
    x 4 columns, 8 VPU lanes / TMEM banks, 64K-word TMEM, 128 ACT RAM blocks (K <= 16384),
    4K-instruction IMEM, 4 GiB DRAM). The sizes come from the target, and so do WG, OUTER, DSTEP2,
    KT, ICACHE (IMEM an instruction cache: IMEM_WORDS the longest program), FSUM (EXP2SUB's
    fused row sums), VSIG (VOP SIGMOID and SILU), VMAC (VOP MAC), OUTER1 (VOP OUTER decay one),
    (at WG > 1) CHASH and GALL (the RTL at WG > 1 runs every MM in groups) and, with DSTEP, SE_OWN
    (the target's SE_R > 0; OTPU_SE_R, the simulators' engine, overrides it). OTPU_MCOLS in the
    environment selects the MXU column count (default the target's: 4, as make -C
    boards/ypcb-00338 bit builds; 2: bit MCOLS=2), OTPU_LANES the VPU lanes / TMEM banks (default
    the target's 8; bit LANES=16; timing only, the programs do not change), OTPU_ACT_ROWS the ACT
    RAM rows (default MCOLS; more: the MXU replays each weight chunk for MCOLS rows at a time).
    The optional features stay off unless the environment turns them on, as before targets (the
    target states the hardware's truth: opentpu.target.config): OTPU_PAIR=1 column reuse (MM
    PAIR / QACT DUP), OTPU_DSTEP=1 the DMA's DSTEP, OTPU_STREAM=1 the stream engine,
    OTPU_SEQ_WC=1 the sequencer's wait entries (SEQ_WC, EXPERIMENTAL). They configure the
    simulators and the board model; on the card, opentpu.host.board.device_config takes them
    from the bitstream."""
    from .target import current
    t, env = current(), os.environ
    a = t.arch
    base = dict(S=a["S"], D=a["D"], MCOLS=int(env.get("OTPU_MCOLS", a["MCOLS"])),
                ACT_BLOCKS=a["ACT_BLOCKS"], LANES=int(env.get("OTPU_LANES", a["LANES"])),
                ACT_ROWS=int(env.get("OTPU_ACT_ROWS", a["ACT_ROWS"])),
                TMEM_WORDS=a["TMEM_WORDS"], IMEM_WORDS=a["IMEM_WORDS"],
                DRAM_BYTES=a["DRAM_BYTES"],
                PAIR=bool(int(env.get("OTPU_PAIR", 0))),
                DSTEP=bool(int(env.get("OTPU_DSTEP", 0))),
                STREAM=bool(int(env.get("OTPU_STREAM", 0))),
                SEQ_WC=bool(int(env.get("OTPU_SEQ_WC", 0))), WG=a["WG"], OUTER=a["OUTER"],
                DSTEP2=a["DSTEP2"], KT=a["KT"], AW=a["AW"], BHW=a["BHW"], ICACHE=a["ICACHE"],
                ST16=a["ST16"], FSUM=a["FSUM"], ST16S=a["ST16S"], VSIG=a["VSIG"],
                VMAC=a["VMAC"], OUTER1=a["OUTER1"])
    base.update(kw)
    base.setdefault("CHASH", base["WG"] > 1 and bool(t.uarch["CHASH"]))
    base.setdefault("GALL", base["WG"] > 1)
    se_r = int(env.get("OTPU_SE_R") or t.uarch.get("SE_R", 0))
    base.setdefault("SE_OWN", se_own(se_r, base["DSTEP"], base["LANES"], base["D"]))
    base.setdefault("WC_WIN", 32 if base["SEQ_WC"] and t.uarch["WIN"] > 16 else 16)
    return Config(**base)


def se_own(se_r: int, dstep: bool, lanes: int, d: int) -> bool:
    """Config.SE_OWN of a build with SE_R and DSTEP: otpu_slice's SE_OWN (an engine of its own
    at SE_R > 0 where the slice has the stream engine: DSTEP, 8 lanes, D >= 32)."""
    return se_r > 0 and dstep and lanes == 8 and d >= 32


class SimError(RuntimeError):
    pass


DRAM_FILE_MIN = 1 << 32     # a slice's DRAM above it (AW > 32) is a sparse file: dram_zeros


def bf16_rne(x) -> np.ndarray:
    """fp32 values as bf16 bits (uint16), as a DSTEP with BF16 stores its state (otpu_dma
    bf16_rne): round to nearest even; a subnormal to its signed zero (the VPU's flush); an
    infinity stays one, a NaN a NaN with its quiet bit set."""
    b = np.ascontiguousarray(x, np.float32).view(np.uint32)
    e = (b >> 23) & 0xFF
    r = (b + np.uint32(0x7FFF) + ((b >> 16) & 1)) >> 16
    nan = (b >> 16) | (((b & 0x7FFFFF) != 0).astype(np.uint32) << 6)
    return np.where(e == 0, (b >> 16) & 0x8000, np.where(e == 0xFF, nan, r)).astype(np.uint16)


def bf16_f32(h) -> np.ndarray:
    """bf16 bits (uint16) as the fp32 values they are (the low half +0: exact)."""
    return (np.asarray(h, np.uint16).astype(np.uint32) << 16).view(np.float32)


def e6m9_rne(x) -> np.ndarray:
    """fp32 values as e6m9 bits (uint16: sign, 6 exponent bits with bias 31, 9 mantissa bits),
    as a DSTEP with ST16 "e6m9" stores its state (otpu_dma st16_pack): the mantissa rounded to
    nearest even; below 2^-30 (the smallest normal, fp32 exponent < 97) a signed zero, no
    subnormals; a finite magnitude past the largest ((2 - 2^-9) 2^31) or rounding up to 2^32
    saturates to it (0x7DFF); an infinity stays one (exponent 63, mantissa 0), a NaN a NaN
    (mantissa 0x100)."""
    b = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.int64)
    sg, e, m = (b >> 16) & 0x8000, (b >> 23) & 0xFF, b & 0x7FFFFF
    r = ((e - 96) << 9 | m >> 14) + ((m >> 13) & 1 & (((m & 0x1FFF) != 0) | (m >> 14)))
    r = np.where(r >> 9 >= 63, 0x7DFF, r)
    out = np.where(e == 0xFF, 0x7E00 | (m != 0) << 8,
                   np.where(e < 97, 0, np.where(e > 158, 0x7DFF, r)))
    return (sg | out).astype(np.uint16)


def e6m9_f32(h) -> np.ndarray:
    """e6m9 bits (uint16) as the fp32 values they are (exact): exponent 0 a signed zero, 63
    an infinity or NaN (the mantissa's high bits), else 2^(e - 31) (1 + m / 512)."""
    h = np.asarray(h, np.uint16).astype(np.uint32)
    e, rest = (h >> 9) & 0x3F, (h & 0x1FF) << 14
    b = np.where(e == 0, 0, np.where(e == 63, 0xFF << 23 | rest, (e + 96) << 23 | rest))
    return ((h & 0x8000) << 16 | b).astype(np.uint32).view(np.float32)


def st16_pack(x, fmt: str) -> np.ndarray:
    """fp32 values as the 16-bit format's bits (Config.ST16: "e6m9" or "bf16")."""
    return {"e6m9": e6m9_rne, "bf16": bf16_rne}[fmt](x)


def st16_f32(h, fmt: str) -> np.ndarray:
    """The 16-bit format's bits as the fp32 values they are (Config.ST16)."""
    return {"e6m9": e6m9_f32, "bf16": bf16_f32}[fmt](h)


def dram_zeros(nbytes: int) -> np.ndarray:
    """A slice's DRAM, nbytes of zeros allocated as they are written: up to DRAM_FILE_MIN
    np.zeros (anonymous pages, a page's memory the first time it is written); above it
    simtmp.dram_file's sparse file, so a 32 GiB slice holds only the pages written, and those
    in the page cache the kernel can write back and drop."""
    if nbytes <= DRAM_FILE_MIN:
        return np.zeros(nbytes, dtype=np.uint8)
    from . import simtmp
    return simtmp.dram_file(nbytes)


def dram_copy(dst: np.ndarray, src: np.ndarray, chunk: int = 1 << 26) -> None:
    """dst[:len(src)] = src, dst zeros (dram_zeros): into a file the chunks of src that are all
    zeros are skipped, so an image's empty DRAM costs neither disk nor page cache."""
    n = len(src)
    if len(dst) <= DRAM_FILE_MIN:
        dst[:n] = src
        return
    for a in range(0, n, chunk):
        blk = src[a:a + chunk]
        if blk.any():
            dst[a:a + len(blk)] = blk


class Slice:
    def __init__(self, cfg: Config, sid: int, program: list[I.Instr], dram: np.ndarray | None):
        self.cfg, self.sid = cfg, sid
        self.prog = program
        self.dram = dram_zeros(cfg.DRAM_BYTES)
        self.am = (1 << cfg.AW) - 1             # a DRAM address's bits (docs/isa.md "Address bits")
        if dram is not None:
            dram_copy(self.dram, dram)
        self.tmem = np.zeros(cfg.TMEM_WORDS, dtype=np.uint32)
        self.act = np.zeros((cfg.act_rows, cfg.ACT_BLOCKS * cfg.D), dtype=np.int8)
        # the scale record of each (row, block): the row's Sa and the sub-blocks' offsets da
        self.ascale = np.zeros((cfg.act_rows, cfg.ACT_BLOCKS), dtype=np.float32)
        self.adoff = np.zeros((cfg.act_rows, cfg.ACT_BLOCKS, Q.NSUB), dtype=np.uint8)
        self.R = [0] * 16
        self.pc = 0
        self.stack: list[list[int]] = []
        self.halted = False
        self.chain: tuple | None = None       # HALT CHAIN: (DRAM byte address, instructions)
        self.waiting: I.Instr | None = None   # blocked on a collective
        self.polling: I.Instr | None = None   # a WAITW whose condition does not hold yet
        self.stalls = 0                       # WAITWs that had to wait (for the host)
        self.icount = 0
        # FSUM: the EXP2SUBs' row sums not taken yet, oldest first: (their TMEM indices, values)
        self.fsums: list[tuple] = []
        # the writes' end with the bank-half window (Config.BHW; qwen3.IsaBackend sets it for an
        # image with its weights out of the low area): a write at or past it raises
        self.wlim: int | None = None

    # ---------------------------------------------------------------- memory helpers
    @property
    def m32(self) -> np.ndarray:
        return self.dram.view(np.uint32)

    def reg(self, r: int) -> int:
        return 0 if r == 0 else self.R[r]

    def da(self, r: int, w: int) -> int:
        """R[r] + w as a DRAM address: AW bits, w zero-extended."""
        return (self.reg(r) + w) & self.am

    def ta(self, r: int, w: int) -> int:
        """R[r] + w as a TMEM index or a 32-bit field: the low 32 bits."""
        return (self.reg(r) + w) & 0xFFFFFFFF

    def imm(self, w) -> int:
        """LI's and ADDI's immediate: {w2, w1} above AW 32, w1 at it (w2 ignored)."""
        return w[0] | w[1] << 32 if self.cfg.AW > 32 else w[0]

    def _widx(self, byte_addr: np.ndarray | int) -> np.ndarray:
        a = np.asarray(byte_addr, dtype=np.int64)
        if np.any(a % 4) or np.any(a < 0) or np.any(a + 4 > self.cfg.DRAM_BYTES):
            raise SimError(f"slice {self.sid}: bad DRAM word address")
        return a // 4

    def _wguard(self, byte_addr, what: str) -> None:
        """A write's bytes (or their words' byte addresses) below wlim (the low area's end)."""
        if self.wlim is not None and np.size(byte_addr) and int(np.max(byte_addr)) >= self.wlim:
            raise SimError(f"slice {self.sid}: {what} at pc {self.pc} writes byte "
                           f"{int(np.max(byte_addr)):#x}, past the low area's end {self.wlim:#x} "
                           f"(the bank-half window: Config.BHW)")

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
            if self.fsums:
                raise SimError(f"slice {self.sid}: HALT with {len(self.fsums)} FSUM row sums not "
                               f"taken (an RSUM FSUM after each EXP2SUB FSUM)")
            self.halted = True
            if ins.flags & I.F_CHAIN:
                self.chain = (self.reg(ins.ra), self.reg(ins.rb))
            return
        if op in (I.BAR, I.GATHER):
            self.waiting = ins
            return
        if op == I.WAITW:
            if self.poll(ins):
                self.advance()
            else:
                self.polling = ins
                self.stalls += 1
            return
        if op == I.LOOP:
            count = self.ta(ins.ra, ins.w[1])
            body = ins.w[0]
            if body < 1:
                raise SimError("LOOP with empty body")
            # whatever the count: one of 0 would jump past the enclosing body's end (and skip
            # its back edge)
            end = self.pc + body
            if any(e[1] == end for e in self.stack):
                raise SimError("nested loop bodies end on the same instruction")
            if count == 0:
                self.pc += 1 + body
                return
            if len(self.stack) >= 4:
                raise SimError("loop stack overflow")
            self.stack.append([self.pc + 1, end, count])
            self.pc += 1
            return
        self.execute(ins)
        self.advance()

    def holds(self, ins: I.Instr) -> bool:
        """Whether WAITW ins's condition holds on its DRAM word now (no effect: poll takes it)."""
        v = int(self.m32[self._widx(np.int64(self.da(ins.ra, ins.w[0])))])
        return I.waitw_holds(v, self.reg(ins.rc) + ins.w[2], ins.flags & 3, ins.w[3])

    def poll(self, ins: I.Instr) -> bool:
        """WAITW's DRAM read: if its condition holds, T[R[rb] + w2] = the word's bits and True."""
        v = int(self.m32[self._widx(np.int64(self.da(ins.ra, ins.w[0])))])
        if not I.waitw_holds(v, self.reg(ins.rc) + ins.w[2], ins.flags & 3, ins.w[3]):
            return False
        self.tmem[self._tidx(np.int64(self.ta(ins.rb, ins.w[1])))] = v
        return True

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
                self.R[ins.rd] = self.imm(w) & self.am
            return
        if op == I.ADDI:
            if ins.rd:
                self.R[ins.rd] = (self.reg(ins.ra) + self.imm(w)) & self.am
            return
        if op == I.RLD:
            v = int(self.tmem[self._tidx(np.int64(self.ta(ins.ra, w[0])))])
            v = v if ins.flags & I.F_RAW else I.f2i(v)
            if ins.flags & I.F_MUL:
                v = (v * self.da(ins.rb, w[1])) & self.am     # the word zero-extended
            lo, hi = w[2] | w[4] << 32, w[3] | w[5] << 32     # (hi past 2^AW: it wraps)
            if ins.flags & I.F_BOUND and not (lo <= v <= hi or lo <= v + self.am + 1 <= hi):
                raise SimError(f"RLD at pc {self.pc}: {v} outside its bound [{lo}, {hi}] "
                               f"({ins.comment})")
            if ins.rd:
                self.R[ins.rd] = v
            return
        if op == I.LD:
            d, t = self.da(ins.ra, w[0]), self.ta(ins.rb, w[1])
            n = w[2]
            wi = self._widx(d + 4 * np.arange(n))
            self.tmem[self._tidx(t + np.arange(n))] = self.m32[wi]
            return
        if op == I.ST:
            d, t = self.da(ins.ra, w[0]), self.ta(ins.rb, w[1])
            n = w[2]
            wi = self._widx(d + 4 * np.arange(n))
            self._wguard(4 * wi, "ST")
            self.m32[wi] = self.tmem[self._tidx(t + np.arange(n))]
            return
        if op == I.DSTEP:
            return self._dstep(ins)
        if op == I.STREAM:
            return self._stream(ins)
        if op == I.LDA:
            return self._lda(ins)
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
        results are written; o is written last. PAIR: rows r >= rows / 2 are head b's, its q | k
        2 cols on with QKB, its e and beta one word on. ST16: the state is 16-bit in DRAM (the
        configuration's format, 2 rows cols bytes, column c's bits at byte 2 c of its row), read
        exactly and written rounded (st16_pack) after o is computed from the fp32 values."""
        w = ins.w
        d, qk, v = self.da(ins.ra, w[0]), self.ta(ins.rb, w[1]), self.ta(ins.rc, w[2])
        rows, cols = w[3] & 0xFFFF, w[3] >> 16
        g, o, gs = w[4], w[5], w[6]
        pair = bool(ins.flags & I.F_DPAIR)
        if not (0 < rows <= I.DSTEP_MAX_ROWS) or cols % 64 or not (0 < cols <= 256):
            raise SimError("DSTEP: rows must be 1..256, cols 64, 128, 192 or 256")
        if d % self.cfg.D:
            raise SimError("DSTEP: the state must be DRAM-chunk aligned")
        if pair and not self.cfg.DSTEP2:
            raise SimError("DSTEP pair: this configuration has no DSTEP2 (Config.DSTEP2)")
        if pair and (rows % I.DSTEP_PAIR_ROWS or cols != I.DSTEP_PAIR_COLS):
            raise SimError(f"DSTEP pair: rows must be a multiple of {I.DSTEP_PAIR_ROWS}, cols "
                           f"{I.DSTEP_PAIR_COLS}")
        if ins.flags & I.F_DQKB and not pair:
            raise SimError("DSTEP: QKB without PAIR")
        heads = 2 if pair else 1
        h = rows // heads
        qb = qk + (2 * cols if ins.flags & I.F_DQKB else 0)
        q = np.concatenate([np.repeat(self.tget(a + np.arange(cols))[None, :], h, 0)
                            for a in (qk, qb)[:heads]])
        k = np.concatenate([np.repeat(self.tget(a + cols + np.arange(cols))[None, :], h, 0)
                            for a in (qk, qb)[:heads]])
        vv = self.tget(v + np.arange(rows))
        e = np.repeat(self.tget(g + np.arange(heads)), h)
        beta = np.repeat(self.tget(g + gs + np.arange(heads)), h)
        h16 = bool(ins.flags & I.F_DST16)
        if h16 and not self.cfg.ST16:
            raise SimError("DSTEP ST16: this configuration has no ST16 (Config.ST16)")
        if h16 and 2 * cols % self.cfg.D:
            raise SimError("DSTEP ST16: a row's 16-bit values must be whole DRAM chunks")
        wi = self._widx(d + 4 * np.arange(rows * cols // (2 if h16 else 1)))
        if ins.flags & I.F_DZERO:
            S = np.zeros((rows, cols), np.float32)
        elif h16:
            S = st16_f32(self.m32[wi].view(np.uint16), self.cfg.ST16).reshape(rows, cols)
        else:
            S = self.m32[wi].view(np.float32).reshape(rows, cols)
        kv = F.rdot(S, k)
        dd = F.mul(F.sub(vv, F.mul(kv, e)), beta)
        S = F.outer(S, e[:, None], dd[:, None], k)
        oo = F.rdot(S, q)
        self._wguard(4 * wi, "DSTEP")
        self.m32[wi] = (st16_pack(F.f32(S).reshape(-1), self.cfg.ST16) if h16 else
                        F.f32(S).reshape(-1)).view(np.uint32)
        self.tput(o + np.arange(rows), oo)

    def _stream(self, ins: I.Instr) -> None:
        """STREAM (docs/stream.md, 3): the descriptor, the column slots, the row scalars and the
        constants are read first; row r of the stream is written after row r is read (a later
        row reads what an earlier row wrote); O and the saved registers are written last. ST16:
        the DRAM stream's values 16-bit (as DSTEP's state: 2 cols bytes a row), expanded as
        read and rounded as written; ST16R (ST16W) the source's (the destination's) alone, the
        other end fp32, the two not overlapping."""
        w, fl, D = ins.w, ins.flags, self.cfg.D
        at, ks = w[0] & 0xFFFF, w[0] >> 16
        n = min(I.STREAM_DESC_FIXED + I.STREAM_MAX_OPS, self.cfg.TMEM_WORDS - at)
        try:
            d = I.StreamDesc.decode(self.tmem[self._tidx(at + np.arange(max(n, 0)))])
        except (ValueError, IndexError) as e:
            raise SimError(f"STREAM: {e}") from None
        in_t, out_t = bool(fl & I.F_SRC_T), bool(fl & I.F_DST_T)
        h16r = bool(fl & (I.F_SST16 | I.F_SST16R))         # the source 16-bit
        h16w = bool(fl & (I.F_SST16 | I.F_SST16W))         # the destination 16-bit
        if (h16r or h16w) and not self.cfg.ST16:
            raise SimError("STREAM ST16: this configuration has no ST16 (Config.ST16)")
        if h16r != h16w and not self.cfg.ST16S:
            raise SimError("STREAM ST16R / ST16W: this configuration has no ST16S")
        if (h16r or h16w) and (in_t or out_t or fl & I.F_NODST):
            raise SimError("STREAM ST16: both ends must be in DRAM")
        src = (self.ta if in_t else self.da)(ins.ra, w[1])     # TMEM 32 bits, DRAM AW
        dst = (self.ta if out_t else self.da)(ins.ra, w[2])
        vec, xa, ka = self.ta(ins.rb, w[3]), self.ta(ins.rc, w[4]), self.ta(ins.rd, w[5])
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
        zero, nodst = bool(fl & I.F_SZERO), bool(fl & I.F_NODST)
        r, c = np.arange(rows)[:, None], np.arange(cols)[None, :]
        if (h16r or h16w) and 2 * cols % D:
            raise SimError("STREAM ST16: a row's 16-bit values must be whole DRAM chunks")
        c2 = np.arange(cols // 2)[None, :]          # (the words of a row's 16-bit pairs)
        if h16r:
            sidx = src + 2 * (r * cols + 2 * c2)
        else:
            sidx = src + r * (d.srs or cols) + c if in_t else src + 4 * (r * cols + c)
        if h16w:
            didx = dst + 2 * (r * cols + 2 * c2)
        else:
            didx = dst + r * (d.drs or cols) + c if out_t else dst + 4 * (r * cols + c)
        if (not in_t and not zero and src % D) or (not out_t and not nodst and dst % D):
            raise SimError("STREAM: a DRAM stream must be DRAM-chunk aligned")
        if not in_t and not zero:
            sidx = self._widx(sidx)
        if not out_t and not nodst:
            didx = self._widx(didx)
        if h16r != h16w and not zero and np.intersect1d(sidx, didx).size:
            raise SimError("STREAM ST16 on one side: the source and destination overlap")
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
                x = value(a, A, Xr)
                y = x if op in unary else value(b, A, Xr)     # a unary op has no b
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
            if h16r:
                return st16_f32(self.m32[sidx[lo:hi]].view(np.uint16), self.cfg.ST16)
            return self.m32[sidx[lo:hi]].view(np.float32)

        def store(lo, hi, Y):
            if nodst:
                return
            if out_t:
                self.tput(didx[lo:hi], Y)
            else:
                self._wguard(4 * didx[lo:hi], "STREAM")
                self.m32[didx[lo:hi]] = (st16_pack(F.f32(Y), self.cfg.ST16) if h16w else
                                         F.f32(Y)).view(np.uint32)

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
    def _mx_scales(self, wf, ssa, srs, N, KB, k0, unit, rsc, R=1):
        """The weights' scales (docs/isa.md "Weight formats", "Weight groups"): Sw [N] fp32, and
        per (row, block, group) the offset dw and the multiplier m [N, KB, G] (G = 8 groups of
        D/8 for NVFP4, else the 4 sub-blocks of D/4). Row n is row j = n % R of the scale area
        at ssa + (n // R) * srs: its Sw the word at 4 j, block k's data at swp + (k R + j) sb
        (R = 1: the row's scale row)."""
        G = 8 if wf == I.WNV else Q.NSUB
        if unit:
            return (np.ones(N, np.float32), np.zeros((N, KB, G), np.int64),
                    np.ones((N, KB, G), np.int64))
        fmt = {I.W8: "int8", I.W4I: "int4", I.W4F: "fp4", I.WNV: "nvfp4"}[wf]
        p = Q.swp(fmt, R, rsc)
        if ssa % p or srs % p:
            raise SimError(f"MM: ssa and srs must be multiples of {p}")
        n = np.arange(N)
        area, j = ssa + (n // R) * srs, n % R
        sw = self.m32[self._widx(area + 4 * j)].view(np.float32)
        if rsc:
            return sw, np.zeros((N, KB, G), np.int64), np.ones((N, KB, G), np.int64)
        sb = Q.SCALE_DATA[fmt]
        ba = (area + p + sb * j)[:, None] + (k0 + np.arange(KB))[None, :] * (R * sb)  # [N, KB]
        if wf == I.W8:                    # one dw byte per block
            if ba.max() >= self.cfg.DRAM_BYTES:
                raise SimError("MM: scale rows out of range")
            dw = np.repeat(self.dram[ba].astype(np.int64)[:, :, None], G, axis=2)
            return sw, dw, np.ones((N, KB, G), np.int64)
        if wf in (I.W4I, I.W4F):          # a word per block
            word = self.m32[self._widx(ba)].astype(np.int64)
            dw = np.repeat((word & 0xFF)[:, :, None], G, axis=2)
            m = np.stack([(word >> (8 + 6 * b)) & 63 for b in range(G)], -1)
            return sw, dw, m
        # NVFP4: eight E4M3 bytes per block
        ba = ba[:, :, None] + np.arange(8)[None, None, :]
        if ba.max() >= self.cfg.DRAM_BYTES:
            raise SimError("MM: scale rows out of range")
        e4 = self.dram[ba].astype(np.int64)
        if np.any(e4 & 0x80):
            raise SimError("MM: an NVFP4 scale with the sign bit set")
        E, Mt = (e4 >> 3) & 15, e4 & 7
        return sw, 15 - np.maximum(E, 1), np.where(E > 0, 8 + Mt, Mt)

    MM_CHUNK = 1 << 22                    # (row, output, block, group) terms per numpy pass

    def _mm(self, ins: I.Instr) -> None:
        cfg, w = self.cfg, ins.w
        D = cfg.D
        sa, ssa, out = self.da(ins.ra, w[0]), self.da(ins.rb, w[1]), self.ta(ins.rc, w[2])
        N, KB, k0 = w[3] & 0xFFFF, (w[3] >> 16) & 0xFF, (w[3] >> 24) & 0xFF
        rs = w[4]
        kt = bool(ins.rd) and bool(rs & 1)                # K tail (docs/isa.md "K tail")
        rs -= kt
        ors, M, ab = w[5] & 0xFFFF, (w[5] >> 16) & 0xFF, w[5] >> 24
        srs = w[6]
        spl = bool(srs & I.MM_SPL)                        # split terms (docs/isa.md "Split terms")
        hld = (srs >> I.MM_HLD & (1 << 31 - I.MM_HLD) - 1) << 10 if spl else 0
        srs = srs & (1 << I.MM_HLD) - 1 if spl else srs
        unit, accf = bool(ins.flags & I.F_UNIT), bool(ins.flags & I.F_ACC)
        pair, rsc = bool(ins.flags & I.F_PAIR), bool(ins.flags & I.F_RSC)
        Rg = 1 << ins.rd                                  # rows a group (docs/isa.md)
        R = 2 * M if pair else M                          # ACT rows read
        nab = KB - 1 if spl else KB                       # ACT blocks read
        if not N or not KB:
            raise SimError("MM: N and KB must be at least 1")
        if spl:
            if not cfg.SPLMM:
                raise SimError("MM SPL: this configuration has no SPLMM (Config.SPLMM)")
            if (KB not in (2, 3) or Rg == 1 or kt or k0 or not rsc or unit or pair or not hld
                    or (ins.flags >> I.WF_SHIFT) & 3 != I.W8
                    or ins.flags & (I.F_RMAX | I.F_ASCALE)):
                raise SimError("MM SPL: KB 2 or 3, groups, int8 RSC, k0 0, HLD > 0 and not "
                               "UNIT, PAIR, KT, RMAX or ASCALE")
        if not (0 < R <= cfg.act_rows) or ab + nab > cfg.ACT_BLOCKS:
            raise SimError("MM: M or ACT RAM range out of bounds")
        if M > cfg.MCOLS and ins.flags & (I.F_RMAX | I.F_ASCALE):
            raise SimError("MM: RMAX and ASCALE need M <= MCOLS")
        if pair and R > cfg.MCOLS:
            raise SimError("MM: PAIR needs 2*M <= MCOLS")
        wf = (ins.flags >> I.WF_SHIFT) & 3
        if pair and wf == I.W8:
            raise SimError("MM: PAIR needs 4-bit weights")
        if pair and k0 % 2:
            raise SimError("MM: PAIR needs an even k0")
        if rsc and wf == I.WNV:
            raise SimError("MM: RSC does not take NVFP4")
        if ins.flags & I.F_ASCALE and not (unit and accf):
            raise SimError("MM: ASCALE needs UNIT and ACC")
        fmt = {I.W8: "int8", I.W4I: "int4", I.W4F: "fp4", I.WNV: "nvfp4"}[wf]
        if Rg > 1 and Rg != Q.grp_r(fmt, cfg.WG):
            raise SimError(f"MM: {fmt} groups of {Rg} rows, the machine's are "
                           f"{Q.grp_r(fmt, cfg.WG)} (WG {cfg.WG}: WG rows, 2 WG of a 4-bit format)")
        if Rg > 1 and ins.flags & (I.F_PAIR | (0 if cfg.GALL else I.F_RMAX | I.F_ASCALE)):
            raise SimError("MM: groups take neither RMAX, ASCALE (but GALL's) nor PAIR")
        if cfg.GALL and cfg.WG > 1 and Rg == 1:
            raise SimError("MM: rows of one at WG > 1 under GALL (every MM in groups)")
        bpb = D if wf == I.W8 else D // 2                 # streamed bytes per K-block
        L = Rg // 2 if Rg > 1 and wf != I.W8 else Rg      # a group chunk's lanes (WG; 1: rows)
        P = L * D                                         # a group chunk's bytes
        nch = 1 if spl else KB if Rg > 1 else -(-KB * bpb // D)   # chunks a group (a row; SPL: a part)
        if Rg > 1 and (sa % P or rs % P or hld % P):
            raise SimError(f"MM: grouped sa, rs and HLD must be multiples of WG x D = {P}")
        ng = -(-N // Rg)                                  # groups (rows of one: rows)
        end = sa + hld + (ng - 1) * rs + nch * P
        if kt:
            # groups 2p and 2p + 1 at sa + p rs: 2p's chunks 0 .. KB-2, the tail chunk T_p (lane
            # bytes 0 .. D/2-1 2p's half block, D/2 .. D-1 2p + 1's), 2p + 1's chunks 0 .. KB-2;
            # 2p + 1 takes T_p first against the tail block's halves swapped
            if not cfg.KT:
                raise SimError("MM: KT (rs bit 0) on a machine without it (Config.KT)")
            if k0 or wf == I.WNV:
                raise SimError("MM: KT needs k0 0 and an int8, int4 or fp4 operand")
            if rs < (2 * KB - 1) * P:
                raise SimError(f"MM: KT's pair stride {rs} is under (2 KB - 1) P")
            if np.any(self.act[:R, (ab + KB - 1) * D + D // 2:(ab + KB) * D]):
                raise SimError("MM: KT needs the last ACT block's upper half zero")
            end = sa + (ng - 1) // 2 * rs + (KB if ng % 2 else 2 * KB - 1) * P
        if sa % D or rs % D or end > cfg.DRAM_BYTES:
            raise SimError("MM: streamed rows must be D-byte aligned and in range")

        def rows(n0, n1, sa=sa, KB=KB):
            """The elements [n1 - n0, KB, D] of rows n0 .. n1 - 1 (n0 a multiple of Rg): chunk c
            of group g at sa + g rs + c P, lane l's D bytes at l D (CHASH: (l ^ h) D, h the
            chunk's fold); int8 (and rows of one) lane l row g Rg + l's chunk c, 4-bit blocks
            2c, 2c + 1 a row's chunk's halves; 4-bit groups lane l rows g Rg + 2l (the low
            nibbles) and g Rg + 2l + 1 (the high), element i of block c in byte i. (SPL: a
            part's one block, sa its own.)"""
            g0, g1 = n0 // Rg, -(-n1 // Rg)
            gs = np.arange(g0, g1)
            if kt:                                      # a group's chunks from its first (KT)
                base = sa + gs // 2 * rs + gs % 2 * (KB - 1) * P
                ch = np.empty((g1 - g0, nch, L, D), np.uint8)
                for e in (0, 1):
                    f = g0 + (e - g0) % 2
                    if f < g1:
                        ch[f - g0::2] = np.lib.stride_tricks.as_strided(
                            self.dram[base[f - g0]:], (len(range(f, g1, 2)), nch, L, D),
                            (rs, P, D, 1))
            else:
                base = sa + gs * rs
                ch = np.lib.stride_tricks.as_strided(
                    self.dram[sa + g0 * rs:], (g1 - g0, nch, L, D), (rs, P, D, 1))  # [g, c, l, i]
            if cfg.CHASH and L > 1:                     # lane l of each chunk at l ^ h
                g, c = np.arange(g1 - g0)[:, None, None], np.arange(nch)[None, :, None]
                h = Q.chunk_fold(base[:, None, None] // P + c, L)               # [g, c, 1]
                ch = ch[g, c, np.arange(L)[None, None, :] ^ h]
            if L < Rg:
                e = np.stack([ch & 15, ch >> 4], 3).transpose(0, 2, 3, 1, 4)    # [g, l, 2, c, i]
                return Q.DEC["int4" if wf == I.W4I else "fp4"][
                    e.reshape((g1 - g0) * Rg, nch, D)[:n1 - n0, :KB]]
            wv = ch.transpose(0, 2, 1, 3).reshape((g1 - g0) * L, nch * D // bpb, bpb)
            wv = wv[:n1 - n0, :KB]                                   # [n, KB, bpb]
            if wf == I.W8:
                return wv.view(np.int8)
            return Q.DEC["int4" if wf == I.W4I else "fp4"][Q.unpack4(wv)]
        sw, dw, mw = self._mx_scales(wf, ssa, srs, N, KB, k0, unit, rsc, Rg)
        G = dw.shape[2]
        # the ACT row of each (output row, block): PAIR takes the odd blocks from row j + M.
        # SPL: advance k's ACT block ab + kk[k] (KB 3: hi, lo, hi; KB 2: hi, hi), its offsets
        # + ex[k] (8 on the lo terms'); Sa block ab's (hi's: lo's is not read)
        kk = np.array([0, 1, 0] if KB == 3 else [0, 0]) if spl else np.arange(KB)
        ex = np.array([0, 8, 8] if KB == 3 else [0, 8]) if spl else np.zeros(KB, np.int64)
        rsel = np.arange(M)[:, None] + np.where(pair & (kk % 2 == 1), M, 0)[None, :]   # [M, KB]
        sa_rows = self.ascale[:R, ab:ab + (1 if spl else KB)]
        if np.any(sa_rows.view(np.uint32) != sa_rows[np.arange(R) % M][:, :1].view(np.uint32)):
            raise SimError("MM: the ACT rows' blocks hold different Sa (more than one QACT)")
        act = self.act[:R, ab * D:(ab + nab) * D].reshape(R, nab, D)
        act = act[rsel, kk[None, :]].astype(np.float64)                 # [M, KB, D]
        da = self.adoff[:R, ab:ab + nab][rsel, kk[None, :]].astype(np.int64)  # [M, KB, 4]
        da = np.repeat(da + ex[None, :, None], G // Q.NSUB, axis=2)      # [M, KB, G]
        if kt:                     # an odd group's advances: the tail block, halves swapped, first
            act_o = np.concatenate([np.roll(act[:, KB - 1:], D // 2, -1), act[:, :KB - 1]], 1)
            da_o = np.concatenate([da[:, KB - 1:, [2, 3, 0, 1]], da[:, :KB - 1]], 1)
        W = I.MX_W[wf]
        acc = np.zeros((M, N), np.int64)
        step = -(-max(1, self.MM_CHUNK // max(1, M * KB * G)) // Rg) * Rg
        for n0 in range(0, N, step):
            n1 = min(N, n0 + step)
            # exact group sums (|c| < 2^19, float64 BLAS), B = m << sh (sh < 0: the group drops)
            if spl:                    # SPL: hi's block (twice in a triple), then lo's
                wh, wl = rows(n0, n1, sa, 1), rows(n0, n1, sa + hld, 1)
                wr = np.concatenate([wh] * (KB - 1) + [wl], 1)
            else:
                wr = rows(n0, n1)
            wr = wr.reshape(n1 - n0, KB, G, D // G).astype(np.float64)
            c = np.einsum("jkgi,nkgi->jnkg", act.reshape(M, KB, G, D // G), wr,
                          optimize=True).astype(np.int64)                  # [M, n, KB, G]
            sh = W - dw[None, n0:n1] - da[:, None]                             # [M, n, KB, G]
            if kt:
                odd = (np.arange(n0, n1) // Rg % 2 == 1)[None, :, None, None]
                c = np.where(odd, np.einsum("jkgi,nkgi->jnkg", act_o.reshape(M, KB, G, D // G),
                                            wr, optimize=True).astype(np.int64), c)
                sh = np.where(odd, W - dw[None, n0:n1] - da_o[:, None], sh)
            B = np.where(sh >= 0, mw[None, n0:n1] << np.maximum(sh, 0), 0)
            acc[:, n0:n1] = (B * c).sum(axis=(2, 3))
        y = F.mul(F.mul(F.i2f(acc), sw[None, :]),
                  F.mul(self.ascale[:M, ab][:, None], np.float32(2.0 ** -W)))
        idx = out + np.arange(M)[:, None] * ors + np.arange(N)[None, :]
        if ins.flags & I.F_ASCALE:                 # y = old * alpha[j] + y
            alpha = self.tget(self.ta(ins.rb, w[1]) + np.arange(M))[:, None]   # ssa in TMEM
            y = F.add(F.mul(self.tget(idx), alpha), y)
        elif accf:
            y = F.add(self.tget(idx), y)
        self.tput(idx, y)
        if ins.flags & I.F_RMAX:                   # row max of the written values, in n order
            mx = F.chain_max(y)
            self.tput(out + M * ors + np.arange(M), mx)

    # ---------------------------------------------------------------- quantizer
    def _quant_rows(self, src, rows, KB, srs, g, row_mode, cscale=None, rscale=None, lo=False):
        """docs/isa.md "Quantization" of rows rows x KB*D at src (groups of g elements); lo: the
        split's lo part (QACT F_QLO / QST F_SLO, SPLITQ)."""
        if lo and not self.cfg.SPLITQ:
            raise SimError("QACT/QST LO needs SPLITQ")
        D = self.cfg.D
        idx = src + np.arange(rows)[:, None] * srs + np.arange(KB * D)[None, :]
        x = self.tget(idx)                                                  # [rows, KB*D]
        if rscale is not None:                     # QACT RSCALE: x * T[rscale + r]
            x = F.mul(x, self.tget(rscale + np.arange(rows))[:, None])
        if cscale is not None:                     # QACT CSCALE: x * T[cscale + c]
            x = F.mul(x, self.tget(cscale + np.arange(KB * D))[None, :])
        return F.quantize_mx(x, g, row_mode, lo)

    def _qact(self, ins: I.Instr) -> None:
        cfg, w = self.cfg, ins.w
        src = self.ta(ins.ra, w[0])
        rows, ab, KB = w[1] & 0xFF, (w[1] >> 8) & 0xFF, w[1] >> 16
        srs = w[2]
        dup = bool(ins.flags & I.F_DUP)
        if (2 * rows > cfg.MCOLS if dup else rows > cfg.act_rows) or ab + KB > cfg.ACT_BLOCKS:
            raise SimError("QACT out of ACT RAM bounds")
        cs = (w[3] & 0xFFFFFFFF) if ins.flags & I.F_CSCALE else None
        rsc = (w[4] & 0xFFFFFFFF) if ins.flags & I.F_RSCALE else None
        hl = bool(ins.flags & I.F_QHL)
        st = bool(ins.flags & I.F_QST)
        if st and not cfg.QONCE:
            raise SimError("QACT ST: this configuration has no QONCE (Config.QONCE)")
        if st and hl:
            raise SimError("QACT ST with HL")
        if st:                                     # (ST: the source row stride is w6)
            srs = w[5]
        if hl and (not cfg.QHL or not cfg.SPLITQ):
            raise SimError("QACT HL: this configuration has no QHL (Config.QHL)")
        if hl and (dup or ins.flags & I.F_QLO or ab + 2 * KB > cfg.ACT_BLOCKS
                   or KB * cfg.D > I.QHL_ELEMS):
            raise SimError("QACT HL: with DUP or LO, past ACT RAM, or rows over QHL_ELEMS")
        parts = ((ab, False), (ab + KB, True)) if hl else ((ab, bool(ins.flags & I.F_QLO)),)
        for a0, lo in parts:
            q, s, d = self._quant_rows(src, rows, KB, srs, cfg.D // Q.NSUB,
                                       bool(ins.flags & I.F_ROW), cs, rsc, lo)
            for r0 in ((0, rows) if dup else (0,)):  # DUP: rows r and r + rows, same bytes
                self.act[r0:r0 + rows, a0 * cfg.D:(a0 + KB) * cfg.D] = q
                self.ascale[r0:r0 + rows, a0:a0 + KB] = s[:, None]
                self.adoff[r0:r0 + rows, a0:a0 + KB] = d.reshape(rows, KB, Q.NSUB)
            if st:
                self._qrows_out(self.da(ins.rc, w[2]), w[6], rows, KB, a0, "QACT ST")

    def _qrow_addrs(self, base: int, drs: int, rows: int, KB: int, what: str) -> np.ndarray:
        """QACT ST's and LDA's byte addresses [rows, qrow_bytes]: row r at base + r drs (a DRAM
        line apart at least, so that rows do not overlap; both multiples of QALIGN), in range."""
        n = I.qrow_bytes(KB, self.cfg.D)
        if base % I.QALIGN or (rows > 1 and drs % I.QALIGN):
            raise SimError(f"{what}: rows at {base:#x}, {drs} bytes apart: not multiples of "
                           f"{I.QALIGN}")
        if Q.NSUB + 4 != I.QREC:
            raise SimError(f"{what}: records of {I.QREC} bytes need NSUB {I.QREC - 4}")
        if rows > 1 and drs < n:
            raise SimError(f"{what}: rows {drs} bytes apart, each {n}")
        a = base + np.arange(rows, dtype=np.int64)[:, None] * drs + np.arange(n)[None, :]
        if np.any(a >= self.cfg.DRAM_BYTES):
            raise SimError(f"slice {self.sid}: {what} at pc {self.pc}: byte address out of range")
        return a

    def _qrows_out(self, base: int, drs: int, rows: int, KB: int, a0: int, what: str) -> None:
        """ACT rows 0 .. rows - 1 at blocks a0 .. a0 + KB - 1 to DRAM (QACT ST): a row's KB x D
        bytes, then a block's record a block (its fp32 S, its NSUB offsets)."""
        D = self.cfg.D
        a = self._qrow_addrs(base, drs, rows, KB, what)
        rec = np.empty((rows, KB, I.QREC), np.uint8)
        rec[:, :, :4] = np.ascontiguousarray(self.ascale[:rows, a0:a0 + KB]).view(np.uint8) \
            .reshape(rows, KB, 4)
        rec[:, :, 4:] = self.adoff[:rows, a0:a0 + KB]
        self._wguard(a, what)
        self.dram[a] = np.concatenate(
            [self.act[:rows, a0 * D:(a0 + KB) * D].view(np.uint8), rec.reshape(rows, -1)], axis=1)

    def _lda(self, ins: I.Instr) -> None:
        """LDA: QACT ST's rows from DRAM into ACT rows arow .. at blocks ab .. ab + KB - 1, the
        bytes, scales and offsets unchanged."""
        cfg, w = self.cfg, ins.w
        if not cfg.QONCE:
            raise SimError("LDA: this configuration has no QONCE (Config.QONCE)")
        rows, ab, KB = w[1] & 0xFF, (w[1] >> 8) & 0xFF, w[1] >> 16
        arow, D = w[3], cfg.D
        if not rows or not KB or arow + rows > cfg.act_rows or ab + KB > cfg.ACT_BLOCKS:
            raise SimError("LDA out of ACT RAM bounds")
        x = self.dram[self._qrow_addrs(self.da(ins.ra, w[0]), w[2], rows, KB, "LDA")]
        rec = np.ascontiguousarray(x[:, KB * D:]).reshape(rows, KB, I.QREC)
        r = slice(arow, arow + rows)
        self.act[r, ab * D:(ab + KB) * D] = x[:, :KB * D].view(np.int8)
        self.ascale[r, ab:ab + KB] = np.ascontiguousarray(rec[:, :, :4]).view(np.float32) \
            .reshape(rows, KB)
        self.adoff[r, ab:ab + KB] = rec[:, :, 4:]

    def _qst(self, ins: I.Instr, dlo: int = 0) -> None:
        cfg, w = self.cfg, ins.w
        if w[4] & I.QST_HL:                        # HL: QST, then QST LO HLD KiB after it
            hld = (w[4] >> I.QST_HLD & (1 << 31 - I.QST_HLD) - 1) << 10
            fl, KB = ins.flags, w[3] >> 16
            if not (cfg.QSTHL and cfg.SPLITQ):
                raise SimError("QST HL: this configuration has no QSTHL (Config.QSTHL)")
            if (not fl & I.F_QGRP or fl & (I.F_QCOL | I.F_HALF | I.F_SLO) or not fl & I.F_ROW
                    or not hld or KB * cfg.D > I.QHL_ELEMS):
                raise SimError("QST HL: GRP ROW rows only (not COL, HALF or LO), HLD > 0 and "
                               "rows of at most QHL_ELEMS")
            ww = list(w)
            ww[4] = w[4] & (1 << I.QST_HLD) - 1
            self._qst(I.Instr(ins.op, ins.ra, ins.rb, ins.rc, ins.rd, fl, ww))
            self._qst(I.Instr(ins.op, ins.ra, ins.rb, ins.rc, ins.rd, fl | I.F_SLO, ww), hld)
            return
        src = self.ta(ins.ra, w[0])
        dst = (self.da(ins.rb, w[1]) + dlo) & self.am         # (dlo: HL's lo part, HLD after)
        sdst = (self.da(ins.rc, w[2]) + dlo) & self.am
        rows, KB = w[3] & 0xFFFF, w[3] >> 16
        srs, drs, es = w[4], w[5], w[6]
        row_mode = bool(ins.flags & I.F_ROW)
        half = bool(ins.flags & I.F_HALF)
        if half and not row_mode:
            raise SimError("QST: HALF needs ROW mode")
        q, s, d = self._quant_rows(src, rows, KB, srs, cfg.D, row_mode,
                                   lo=bool(ins.flags & I.F_SLO))
        ne = KB * cfg.D // 2 if half else KB * cfg.D   # HALF: the row's first half only
        if ins.flags & I.F_QGRP:
            return self._qst_grp(ins, dst, sdst, rows, KB, ne, es, q, s, d)
        baddr = dst + np.arange(rows)[:, None] * drs + np.arange(ne)[None, :] * es
        if np.any(baddr < 0) or np.any(baddr >= cfg.DRAM_BYTES):
            raise SimError("QST: byte address out of range")
        if row_mode:                               # the RSC scale row: S
            sbytes = np.zeros((rows, 4), np.uint8)
            sbytes[:, :4] = s[:, None].view(np.uint8)
        else:                                      # the int8 scale row: S, then the bytes d_k
            sbytes = np.zeros((rows, 4 + 4 * -(-KB // 4)), np.uint8)
            sbytes[:, :4] = s[:, None].view(np.uint8)
            sbytes[:, 4:4 + KB] = d
        sa = sdst + np.arange(rows)[:, None] * sbytes.shape[1] + np.arange(sbytes.shape[1])
        if sdst % 4 or np.any(sa >= cfg.DRAM_BYTES):
            raise SimError("QST: bad scale address")
        if _overlap(baddr, sa.reshape(-1)):
            raise SimError("QST: the data and scale ranges overlap")
        self._wguard(baddr, "QST")
        self._wguard(sa, "QST")
        self.dram[baddr] = q[:, :ne].view(np.uint8)
        self.dram[sa] = sbytes

    def _qst_grp(self, ins, dst, sdst, rows, KB, ne, es, q, s, d) -> None:
        """QST GRP (docs/isa.md QST): rows n = R[rd] + w6 + r of a grouped int8 matrix at dst
        (groups of R = 2^LG = WG rows, group g's chunk c at dst + g gs + c P, lane l's D bytes at
        (l ^ h) D, h the chunk's fold with CHASH: quant.group_rows / hash_rows). Rows: element e
        of row n in chunk e / D of group n / R, lane n % R, byte e % D, gs = KB P; its scales
        in the group's area at sdst + g es: S at 4 l, block k's offset byte at 4 R + k R + l
        (ROW: S alone; quant.group_scales). COL: the rows are the matrix's columns, element e
        its row e (group e / R, lane e % R), column n in chunk n / D at byte n % D, gs = es;
        the scales S at sdst + 4 r (ROW)."""
        cfg, w, D = self.cfg, ins.w, self.cfg.D
        R = 1 << ((ins.flags >> I.QST_LG) & 7)
        P = R * D
        if cfg.WG == 1 or not cfg.GALL:
            raise SimError("QST: GRP needs WG > 1 and GALL")
        if R != cfg.WG:
            raise SimError(f"QST: GRP in groups of {R} rows, the machine's are WG {cfg.WG}")
        col = bool(ins.flags & I.F_QCOL)
        if dst % P or es % (P if col else 4) or sdst % 4:
            raise SimError(f"QST: GRP needs dst a multiple of P = {P} and an aligned es")
        n = (self.reg(ins.rd) + w[5] + np.arange(rows)) & 0xFFFFFFFF          # [rows]
        e = np.arange(ne)
        if col:
            A = dst + (e // R)[None, :] * es + (n // D)[:, None] * P            # [rows, ne]
            lane, off = (e % R)[None, :], (n % D)[:, None]
        else:
            A = dst + (n // R)[:, None] * (KB * P) + (e // D)[None, :] * P
            lane, off = (n % R)[:, None], (e % D)[None, :]
        h = Q.chunk_fold(A // P, R) if cfg.CHASH else 0
        baddr = A + (lane ^ h) * D + off
        if col:
            sa = sdst + 4 * np.arange(rows)[:, None] + np.arange(4)[None, :]
            sb = s[:, None].view(np.uint8)
        else:
            area = (sdst + (n // R) * es)[:, None]
            sa = area + 4 * (n % R)[:, None] + np.arange(4)[None, :]
            sb = s[:, None].view(np.uint8)
            if not ins.flags & I.F_ROW:
                sa = np.concatenate([sa, area + 4 * R + np.arange(KB)[None, :] * R +
                                     (n % R)[:, None]], axis=1)
                sb = np.concatenate([sb, d.reshape(rows, KB).astype(np.uint8)], axis=1)
        if np.any(baddr >= cfg.DRAM_BYTES) or np.any(sa >= cfg.DRAM_BYTES):
            raise SimError("QST: byte address out of range")
        if _overlap(baddr, sa.reshape(-1)) or np.unique(baddr).size != baddr.size:
            raise SimError("QST: GRP's data and scale bytes overlap")
        self._wguard(baddr, "QST")
        self._wguard(sa, "QST")
        self.dram[baddr] = q[:, :ne].view(np.uint8)
        self.dram[sa] = sb

    # ---------------------------------------------------------------- VPU
    def _vop(self, ins: I.Instr) -> None:
        w = ins.w
        dst, a, b = self.ta(ins.ra, w[0]), self.ta(ins.rb, w[1]), self.ta(ins.rc, w[2])
        rows, cols = w[3] & 0xFFFF, w[3] >> 16
        drs, ars = w[4] & 0xFFFF, w[4] >> 16
        brs, func, bmode = w[5] & 0xFFFF, (w[5] >> 16) & 0xFF, (w[5] >> 24) & 3
        w7 = self.ta(ins.rd, w[6])
        imm = np.array([w7], dtype=np.uint32).view(np.float32)[0]
        r = np.arange(rows)[:, None]
        c = np.arange(cols)[None, :]
        if ins.flags & I.F_FSUM:
            self._fsum_check(func, bmode, rows, cols)
        if func == I.V_OUTER:
            if not (self.cfg.OUTER or self.cfg.OUTER1 and ins.flags & I.F_DONE):
                raise SimError("VOP OUTER: this configuration's VPU has no OUTER (Config.OUTER "
                               "off: a target built without it; OUTER1: decay one only)")
            return self._outer(ins, dst, a, b, w7, rows, cols, drs, brs, bmode)
        if func == I.V_MAC:
            if not self.cfg.VMAC:
                raise SimError("VOP MAC: this configuration's VPU has no MAC (Config.VMAC off: a "
                               "target built without it)")
            return self._mac(ins, dst, a, b, imm, rows, cols, drs, ars, bmode)
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
            ci = ((c + base) & 0xFFFFFFFF).astype(np.uint32).view(np.int32)   # int32, as i2f's
            self.tput(didx, np.stack([F.chain_max(A), F.i2f(ci)], axis=1).reshape(-1))
            return
        if func in I.REDUCE:
            didx = dst + np.arange(rows) * drs
            if ins.flags & I.F_FSUM:            # the take: the oldest FSUM's sums, written now
                if not self.fsums:
                    raise SimError("VOP RSUM FSUM: no EXP2SUB FSUM's row sums to take")
                sidx, sums = self.fsums.pop(0)
                if not np.array_equal(didx, sidx):
                    raise SimError(f"VOP RSUM FSUM: takes {rows} rows at {dst} (drs {drs}), the "
                                   f"EXP2SUB's sums are {len(sidx)} rows at {sidx[0]}")
                self.tput(didx, sums)
                return
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
               I.V_EXP2SUB: lambda: F.exp2(F.sub(A, B)), I.V_LOG2: lambda: F.log2(A),
               I.V_SIGMOID: lambda: F.sigmoid(A, B), I.V_SILU: lambda: F.silu(A, B)}
        if func not in ops:
            raise SimError(f"VOP: bad func {func}")
        if func in (I.V_SIGMOID, I.V_SILU) and not self.cfg.VSIG:
            raise SimError("VOP SIGMOID / SILU: this configuration's VPU has neither (Config.VSIG "
                           "off: a target built without them)")
        y = ops[func]()
        self.tput(didx, y)
        if ins.flags & I.F_FSUM:            # the row sums, for the take (the RTL's flush)
            sidx = w7 + np.arange(rows)
            if _overlap(sidx, didx):
                raise SimError("VOP EXP2SUB FSUM: the row sums overlap the results")
            self.fsums.append((sidx, F.interleaved_sum(y, F.RED_PARTIALS)))
            # their words are undefined until the take (the RTL writes them at any time
            # between): a NaN here, so a program reading them before it computes NaNs
            self.tput(sidx, np.full(rows, np.nan, np.float32))

    def _fsum_check(self, func, bmode, rows, cols) -> None:
        if not self.cfg.FSUM:
            raise SimError("VOP FSUM: this configuration's VPU has no fused row sums (Config.FSUM "
                           "off: a target built without it)")
        if func == I.V_RSUM:
            return
        if func != I.V_EXP2SUB or bmode == I.B_SCALAR:
            raise SimError(f"VOP FSUM: EXP2SUB (or RSUM, its take) only, B not a scalar (func "
                           f"{func}, bmode {bmode})")
        if rows > I.FSUM_MAX_ROWS or cols < I.FSUM_MIN_COLS:
            raise SimError(f"VOP EXP2SUB FSUM: rows {rows} (<= {I.FSUM_MAX_ROWS}), cols {cols} "
                           f"(>= {I.FSUM_MIN_COLS})")

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

    def _mac(self, ins, dst, a, b, imm, rows, cols, drs, ars, bmode) -> None:
        """dst = add(dst, mul(A, B)) in place, B per column (read before anything is written:
        the RTL's column buffer, so it may overlap dst) or the scalar imm; dst and A are read
        with every element."""
        if bmode not in (I.B_COL, I.B_SCALAR) or (bmode == I.B_COL and cols > I.MAC_MAX_COLS):
            raise SimError(f"VOP MAC: bmode {bmode}, cols {cols} (B per column with cols <= "
                           f"{I.MAC_MAX_COLS}, or a scalar)")
        if ins.flags:
            raise SimError(f"VOP MAC: flags {ins.flags:#x} (none)")
        r = np.arange(rows)[:, None]
        c = np.arange(cols)[None, :]
        B = self.tget(np.broadcast_to(b + c, (1, cols))) if bmode == I.B_COL else \
            np.full((1, cols), imm, np.float32)
        didx, aidx = dst + r * drs + c, a + r * ars + c
        self._check_hazard(didx.reshape(-1), [didx.reshape(-1), aidx.reshape(-1)])
        self.tput(didx, F.add(self.tget(didx), F.mul(self.tget(aidx), B)))

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


def _overlap(a, b) -> bool:
    """Whether two sets of addresses meet (their spans first: most never do)."""
    a, b = np.asarray(a, np.int64).reshape(-1), np.asarray(b, np.int64).reshape(-1)
    if a.max() < b.min() or b.max() < a.min():
        return False
    return np.intersect1d(a, b).size > 0


ARG0 = 8                    # the run's arguments are R8..R15 at the start (docs/isa.md)


def _regs(args) -> list[int]:
    args = list(args or [])
    if len(args) > 8:
        raise ValueError("at most 8 run arguments")
    R = [0] * 16
    for k, v in enumerate(args):
        R[ARG0 + k] = int(v) & 0xFFFFFFFF
    return R


def _fit_imem(cfg: Config, programs) -> None:
    """The board's loader takes at most IMEM's instructions (IMEM_WORDS / 8; ICACHE: the
    longest program's): a longer program cannot run on a device of this configuration."""
    for sid, p in enumerate(programs):
        if len(p) > cfg.IMEM_WORDS // 8:
            raise SimError(f"slice {sid}: a program of {len(p)} instructions, "
                           f"{'the longest' if cfg.ICACHE else 'IMEM holds'} "
                           f"{cfg.IMEM_WORDS // 8} (IMEM_WORDS {cfg.IMEM_WORDS})")


class Machine:
    """args: the run's arguments (up to 8 words): R8..R15 start with them (0 without)."""

    def __init__(self, cfg: Config, programs: list[list[I.Instr]], drams: list[np.ndarray | None],
                 args=None):
        assert len(programs) == cfg.S and len(drams) == cfg.S
        _fit_imem(cfg, programs)
        self.cfg = cfg
        self.slices = [Slice(cfg, s, programs[s], drams[s]) for s in range(cfg.S)]
        self.args = args
        self.collectives = 0             # issued GATHER / BAR since LRST, mod 65536 (X2_SEQ)
        self.chains = 0                  # HALT CHAIN restarts so far
        self.host = None                 # WAITW's host: host(machine) when every slice waits
        for s in self.slices:
            s.R = _regs(args)

    def load(self, programs: list[list[I.Instr]], args=None) -> "Machine":
        """Start new programs on the same machine: DRAM, TMEM and ACT RAM are kept (as on the
        board, where the host writes a new program image between launches). X2_SEQ survives
        RUN resets; only reset_link (the host's LRST) starts its sequence again."""
        assert len(programs) == self.cfg.S
        _fit_imem(self.cfg, programs)
        self.args = args
        for s, p in zip(self.slices, programs):
            s.prog, s.R, s.pc, s.stack = p, _regs(args), 0, []
            s.halted, s.waiting, s.icount, s.chain = False, None, 0, None
            s.polling, s.stalls = None, 0
        return self

    def reset_link(self) -> None:
        """LRST on a stopped machine: restart the collective sequence, keeping its memories."""
        self.collectives = 0

    def _chain(self, s: Slice) -> None:
        """HALT CHAIN (docs/isa.md): the board loads the next program from the slice's DRAM
        into IMEM and starts it with the run's arguments; TMEM, ACT RAM and DRAM are kept."""
        addr, n = s.chain
        if addr % self.cfg.D or addr >> 32 or not 0 < n <= self.cfg.IMEM_WORDS // 8:
            raise SimError(f"slice {s.sid}: HALT CHAIN to {addr:#x}, {n} instructions")
        words = s.m32[s._widx(addr + 4 * np.arange(8 * n))]
        s.prog = [I.Instr.decode(words[8 * k:8 * k + 8]) for k in range(n)]
        s.R, s.pc, s.stack, s.halted, s.waiting = _regs(self.args), 0, [], False, None
        s.chain, s.polling = None, None
        self.chains += 1

    def run(self, max_steps: int = 10_000_000) -> "Machine":
        steps = 0
        # OTPU_WAITS_CHECK=1: each program's wait counts (opentpu.waits, otpu_seq WAITC) checked
        # against this run (no effect on the result: the counts only order the units)
        chk = {} if os.environ.get("OTPU_WAITS_CHECK") else None
        while not all(s.halted for s in self.slices):
            progressed = False
            for s in self.slices:
                while not s.halted and s.waiting is None and s.polling is None:
                    if chk is not None:
                        self._waits_issue(chk, s)
                    s.step()
                    steps += 1
                    progressed = True
                    if steps > max_steps:
                        raise SimError("step limit exceeded")
                    if s.halted and s.chain is not None:
                        self._chain(s)
                        if chk is not None:
                            chk.pop(s.sid, None)
            live = [s for s in self.slices if not s.halted]
            if live and all(s.waiting is not None for s in live):
                if len(live) != len(self.slices):
                    raise SimError("collective with a halted slice")
                ops = {s.waiting.op for s in live}
                if len(ops) != 1:
                    raise SimError("slices disagree on the collective instruction")
                self.collectives = (self.collectives + 1) & 0xFFFF
                if ops == {I.GATHER}:
                    self._gather()
                for s in live:
                    s.waiting = None
                    s.advance()
                progressed = True
            polling = [s for s in self.slices if s.polling is not None]
            if polling and not progressed:
                # every slice that can run is waiting: the host (its WAITW hook) acts now, as
                # it would while the card polls; nothing to act on is the WAITW timeout
                if self.host is not None:
                    self.host(self)
                for s in polling:
                    if s.poll(s.polling):
                        s.polling = None
                        s.advance()
                        progressed = True
                if not progressed:
                    ins = polling[0].polling
                    raise SimError(f"slice {polling[0].sid}: WAITW at pc {polling[0].pc} "
                                   f"never holds ({ins}; host {self.host!r}): the timeout")
            if not progressed:
                raise SimError("deadlock")
        return self

    def _waits_issue(self, chk: dict, s: Slice) -> None:
        from . import waits as W
        c = chk.get(s.sid)
        if c is None or c.prog is not s.prog:
            cfg = self.cfg
            c = chk[s.sid] = W.Checker(s.prog, W.wait_counts(s.prog, cfg.D, cfg.S, aw=cfg.AW,
                                                             win=cfg.WC_WIN),
                                       cfg.D, cfg.S, cfg.AW, cfg.WC_WIN)
        if 0 <= s.pc < len(s.prog) and s.prog[s.pc].op in W.UNIT:
            c.issue(s.pc, [s.reg(r) for r in range(16)])

    def _gather(self) -> None:
        sl = self.slices
        key = []                 # dst resolved: the RTL writes every slice at slice 0's
        for s in sl:
            w = s.waiting.w
            key.append((s.ta(s.waiting.rb, w[1]), w[2], w[4], w[5]))
        if any(k != key[0] for k in key):
            raise SimError("GATHER: slices disagree on dst/rows/cols/drs/seg")
        vals = []
        for s in sl:
            w = s.waiting.w
            src, dst = s.ta(s.waiting.ra, w[0]), s.ta(s.waiting.rb, w[1])
            rows, cols, srs, drs, seg = w[2] & 0xFFFF, w[2] >> 16, w[3], w[4], w[5]
            idx = src + np.arange(rows)[:, None] * srs + np.arange(cols)[None, :]
            didx = dst + np.arange(len(sl))[:, None, None] * seg + \
                np.arange(rows)[None, :, None] * drs + np.arange(cols)[None, None, :]
            if _overlap(idx, didx):
                raise SimError(f"slice {s.sid}: GATHER: the source and destination overlap")
            vals.append(s.tmem[s._tidx(idx)].copy())
        for s in sl:
            w = s.waiting.w
            dst = s.ta(s.waiting.rb, w[1])
            rows, cols, drs, seg = w[2] & 0xFFFF, w[2] >> 16, w[4], w[5]
            for k, v in enumerate(vals):
                idx = dst + k * seg + np.arange(rows)[:, None] * drs + np.arange(cols)[None, :]
                s.tmem[s._tidx(idx)] = v
