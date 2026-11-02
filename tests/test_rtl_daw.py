"""Wide DRAM addresses on the RTL (DAW 33 .. 36; isa.py "DRAM address width", docs/wide_dram_hbm.md):
a DRAM byte address is {REG, (R[x] + imm) mod 2^32}, REG the instruction's 4 GiB region (rd[3:0] of
LD, ST, DSTEP, WAITW, MM, QST; flags[7:4] of STREAM), every DRAM operand of the instruction in it.
The RTL is built with -DOTPU_DAW=N (rtlsim: from cfg.DAW) and runs against the ISA simulator's
2^DAW-byte DRAM (lazy), bit for bit: the windows the RTL's memory models hold (a window per region,
rtlsim.WIDE_WINDOW) and TMEM.

- footprints: otpu_pkg footprint() (sim/verilator/tb_fp_daw.sv) against opentpu.waits.footprint,
  the scoreboard's ranges with their DAW-bit ends, for random instructions in every region;
- regions: random programs (the fuzzers' of test_rtl.py, and int8 / 4-bit PAIR MMs with DUAL)
  whose DRAM instructions each take region 0, 1 or the top one, every region holding its own
  data at the same offsets (they must not alias); on the behavioural DRAM and on the board's
  memory path (otpu_native_dram2, the four-channel model with random stalls);
- the ranges at a region's end (the window at the region's top: the last byte is 0xFFFFFFFF);
- STREAM and DSTEP in regions (test_stream_rtl.py's programs);
- the scoreboard: a write in region 1 does not hold a read of region 0 at the same offset, and
  holds one of region 1 (the cycles, and opentpu.waits' conflict());
- WAITW: all of DRAM, every region, is its write range (a younger MM of another region waits)."""
import dataclasses
import re

import numpy as np
import pytest

from opentpu import isa as I, rtlsim, waits as W
from opentpu.isasim import Config, Machine

from test_rtl import DATA, INT8, MX_BASE, MX_PITCH, _hazard_program, _images, _mx_scales, \
    _random_program
from test_stream_rtl import _special, _stream_rtl_program

G = 1 << 32                          # a region
WIN = rtlsim.WIDE_WINDOW             # the RTL's window per region (1 MiB)
DAWS = [33, 34, 36]
DRAM_OPS = (I.LD, I.ST, I.DSTEP, I.WAITW, I.MM, I.QST, I.STREAM)


def top(daw: int) -> int:
    """The top region of DAW."""
    return (1 << (daw - 32)) - 1


def wcfg(daw: int, **kw) -> Config:
    """One shape for every RTL run here (one build per DAW and memory path): D 128, DUAL (int8
    PAIR), the streams; the ISA simulator's DRAM all 2^DAW bytes."""
    base = dict(S=1, D=128, MCOLS=8, ACT_BLOCKS=64, PAIR=True, DUAL=True, DSTEP=True,
                STREAM=True, DAW=daw, DRAM_BYTES=1 << daw)
    return Config(**{**base, **kw})


def reloc(ins: I.Instr, reg: int, base: int = 0) -> I.Instr:
    """ins with its DRAM operands in region `reg`, `base` bytes up in it (the 32-bit fields mod
    2^32): LD ST DSTEP WAITW w1; MM sa and its DRAM scale rows (not UNIT or ASCALE); QST dst and
    sdst; STREAM its DRAM src / dst."""
    if ins.op not in DRAM_OPS:
        return ins
    w = list(ins.w)
    if ins.op == I.MM:
        idx = [0] + ([] if ins.flags & (I.F_UNIT | I.F_ASCALE) else [1])
    elif ins.op == I.QST:
        idx = [1, 2]
    elif ins.op == I.STREAM:
        idx = [i for i, f in ((1, I.F_SRC_T | I.F_SZERO), (2, I.F_DST_T | I.F_NODST))
               if not ins.flags & f]
    else:
        idx = [0]
    for i in idx:
        w[i] = (w[i] + base) & 0xFFFFFFFF
    if ins.op == I.STREAM:
        return dataclasses.replace(ins, w=w, flags=(ins.flags & 15) | reg << I.REG_FLAGS)
    return dataclasses.replace(ins, w=w, rd=reg)


def regionize(rng, prog, regs, base=None) -> list:
    """Every DRAM instruction of prog in a region drawn from regs (base: {region: base})."""
    return [reloc(x, r, (base or {}).get(r, 0))
            for x, r in zip(prog, (int(rng.choice(regs)) for _ in prog))]


def run_both(cfg, prog, regions: dict, pokes=(), **kw):
    """prog on the ISA simulator and on the RTL. regions: {region: (image bytes <= WIN, base)},
    the region's data from `base` (a multiple of WIN) in the region; the others start as zero.
    pokes: [(cycle, 2^DAW-byte address, word)], the host's writes (the simulator: when it waits).
    Every region's window (at its base), TMEM and the instruction count equal. Returns
    (machine, RTL stats)."""
    daw = cfg.DAW
    prog = list(prog)
    m = Machine(cfg, [prog], [None])
    d = m.slices[0].dram
    for r, (img, b) in regions.items():
        d[(r << 32) + b:(r << 32) + b + len(img)] = img
    if pokes:
        def host(mach):
            for _, a, v in pokes:
                mach.slices[0].m32[a // 4] = np.uint32(v & 0xFFFFFFFF)
        m.host = host
    m.run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [{r: img for r, (img, _) in regions.items()}],
                                  window=WIN, pokes={0: list(pokes)} if pokes else None, **kw)
    bases = {r: b for r, (_, b) in regions.items()}
    want = rtlsim.wide_from_flat(m.slices[0].dram, daw, WIN, bases)
    assert drams[0].shape == want.shape
    for r in range(1 << (daw - 32)):
        bad = np.nonzero(drams[0][r] != want[r])[0]
        assert len(bad) == 0, (f"region {r} (+{bases.get(r, 0):#x}): {len(bad)} DRAM bytes differ, "
                               f"first at {bad[:8]}")
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, f"{len(bad)} TMEM words differ, first at {bad[:8]}"
    assert st["instructions"] == [m.slices[0].icount]
    return m, st


# ------------------------------------------------------------------------------------ footprints
def _fp_instrs(rng, daw: int, n: int = 400) -> list:
    """Random instructions of every DRAM op (and some others) in every region, their DRAM
    offsets often near the region's end (where the ranges clamp to its last byte)."""
    nreg = 1 << (daw - 32)

    def off(align=4):
        k = int(rng.integers(4))
        if k == 0:
            a = G - int(rng.integers(1, 1 << 16))
        elif k == 1:
            a = int(rng.integers(0, 1 << 20))
        else:
            a = int(rng.integers(0, G))
        return a & ~(align - 1)

    def t(n_=60000):
        return int(rng.integers(0, n_))

    out = []
    for _ in range(n):
        r = int(rng.integers(nreg))
        k = int(rng.integers(10))
        if k == 0:
            out.append(I.ld(off(), t(), int(rng.integers(1, 1 << 14)), reg=r))
        elif k == 1:
            out.append(I.st(off(), t(), int(rng.integers(1, 1 << 14)), reg=r))
        elif k == 2:
            out.append(I.dstep(off(), t(), t(), int(rng.integers(1, 257)),
                               int(rng.choice([64, 128, 256])), t(), int(rng.integers(1, 4)), t(),
                               zero=bool(rng.integers(2)), reg=r))
        elif k == 3:
            out.append(I.waitw(off(), t(), 5, I.C_EQ, reg=r))
        elif k in (4, 5):
            wf = int(rng.integers(4))
            unit = bool(rng.integers(2))
            acc = bool(rng.integers(2))
            asc = t() if unit and acc and rng.integers(2) else None
            pair = bool(rng.integers(2)) and wf != I.WNV
            out.append(I.mm(off(128), off(), t(), int(rng.integers(1, 300)),
                            int(rng.integers(1, 20)), 128 * int(rng.integers(1, 40)),
                            int(rng.integers(0, 400)), int(rng.integers(1, 9)),
                            int(rng.integers(0, 40)), 4 * int(rng.integers(1, 200)), unit=unit,
                            acc=acc, rmax=bool(rng.integers(2)), ascale=asc, wf=wf, pair=pair,
                            k0=2 * int(rng.integers(0, 8)),
                            rsc=bool(rng.integers(2)) and wf != I.WNV, reg=r))
        elif k == 6:
            row = bool(rng.integers(2))
            out.append(I.qst(t(), off(), off(), int(rng.integers(1, 64)), int(rng.integers(1, 9)),
                             128, int(rng.integers(1, 5000)), int(rng.integers(1, 5)), row=row,
                             half=row and bool(rng.integers(2)), reg=r))
        elif k == 7:
            out.append(I.stream(int(rng.integers(0, 60000)), off(128), off(128), t(), t(), t(),
                                t(), ks=int(rng.integers(1, 4)), zero=bool(rng.integers(3) == 0),
                                reg=r))
        elif k == 8:
            out.append(I.vop(I.V_ADD, t(), t(), t(), int(rng.integers(1, 9)),
                             int(rng.integers(1, 300)), 300, 300, 300))
        else:
            out.append(I.qact(t(), int(rng.integers(1, 9)), int(rng.integers(0, 32)),
                              int(rng.integers(1, 20)), 128 * int(rng.integers(1, 20))))
    return out


def _rtl_footprints(daw: int, prog: list, tmp_path) -> list:
    """otpu_pkg footprint() of each instruction (its fields resolved with every register 0)."""
    src = [rtlsim.RTL / "top/otpu_pkg.sv", rtlsim.TB / "tb_fp_daw.sv"]
    exe = rtlsim.build("tb_fp_daw", src, {"D": 128, "S": 1}, rtlsim.daw_defines(daw))
    f = tmp_path / f"cmds_{daw}.txt"
    f.write_text("".join(
        f"{x.op:x} {x.flags:x} {x.region():x} " + " ".join(f"{v & 0xFFFFFFFF:x}" for v in x.w) + "\n"
        for x in prog))
    r = rtlsim.run_sim([exe, f"+cmds={f}"], timeout=600)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("FP all=")]
    assert f"FP DONE DAW={daw}" in r.stdout and len(lines) == len(prog), r.stdout[-2000:]
    out = []
    for ln in lines:
        fl = re.findall(r"([RW])\d=(\d+),(\d+),([0-9a-f]+),([0-9a-f]+)", ln)
        rd = sorted((int(sp), int(lo, 16), int(hi, 16)) for k, v, sp, lo, hi in fl
                    if k == "R" and v == "1")
        wr = sorted((int(sp), int(lo, 16), int(hi, 16)) for k, v, sp, lo, hi in fl
                    if k == "W" and v == "1")
        out.append((ln.startswith("FP all=1"), rd, wr))
    return out


@pytest.mark.parametrize("daw", [32] + DAWS + [35])
def test_footprints_match_waits(have_verilator, daw, tmp_path):
    """The scoreboard's ranges (otpu_pkg footprint, DAW-bit ends {REG, 32 bits}) are
    opentpu.waits.footprint's, for every op in every region; WAITW's one range is all of DRAM
    (the RTL's 2^DAW - 1, waits' 2^MAX_DAW - 1: the same against any range of DAW bits)."""
    prog = _fp_instrs(np.random.default_rng(900 + daw), daw)
    got = _rtl_footprints(daw, prog, tmp_path)
    R0 = [(0, 0)] * 16
    allw = (1 << daw) - 1
    for i, (x, (a, rd, wr)) in enumerate(zip(prog, got)):
        f = W.footprint(x, R0, D=128, S=1)
        if x.op == I.WAITW:          # (the whole DRAM: DAW bits here, MAX_DAW in waits)
            wr = [(sp, lo, (1 << I.MAX_DAW) - 1 if (sp, lo, hi) == (W.SP_DRAM, 0, allw) else hi)
                  for sp, lo, hi in wr]
        assert (a, rd, wr) == (f.all, sorted(f.rd), sorted(f.wr)), f"#{i} {x}"


def test_scoreboard_regions_waits_model():
    """opentpu.waits (the scoreboard's model): ranges of different regions never conflict, the
    same offsets in one region do, and WAITW conflicts with every region's."""
    st = I.st(0x4000, 0, 1024, reg=1)
    for daw in DAWS:
        R0 = [(0, 0)] * 16
        fp = lambda x: W.footprint(x, R0)
        assert W.conflict(fp(st), fp(I.ld(0x4000, 5000, 16, reg=1)))
        assert not W.conflict(fp(st), fp(I.ld(0x4000, 5000, 16, reg=0)))
        if top(daw) != 1:
            assert not W.conflict(fp(st), fp(I.ld(0x4000, 5000, 16, reg=top(daw))))
        for r in (0, 1, top(daw)):
            assert W.conflict(fp(I.waitw(0x10, 6000, 1, I.C_EQ, reg=0)),
                              fp(I.ld(G - 64, 5000, 16, reg=r)))


# ------------------------------------------------------------------------------------ regions
def _pair_mms(rng, cfg, n=6) -> list:
    """QACT DUP + int8 PAIR (DUAL: two chunks an advance), 4-bit PAIR and plain int8 MMs on the
    fuzzers' image layout (test_dual_rtl.py's mix)."""
    D = cfg.D
    prog = [I.ld(DATA, 0, 4096)]
    out = 4096
    for _ in range(n):
        M = int(rng.integers(1, cfg.MCOLS // 2 + 1))
        KB = int(rng.integers(1, 4))
        srs = KB * D + int(rng.integers(0, 4))
        M = min(M, 4096 // srs)
        prog.append(I.qact(int(rng.integers(0, 4096 - M * srs)), M, 0, KB, srs, dup=True))
        for _ in range(int(rng.integers(1, 3))):
            wf = I.W8 if rng.random() < 0.7 else I.W4F
            pair = rng.random() < 0.8
            N = int(rng.integers(1, 24))
            rb = KB * D if wf == I.W8 else -(-KB // 2) * D
            rs = rb + D * int(rng.integers(0, 3))
            sa = INT8 + D * int(rng.integers(0, (32768 - N * rs) // D + 1))
            unit = bool(rng.integers(3) == 0)
            ssa, srs_s, k0, rsc = _mx_scales(rng, wf, N, KB, pair, unit)
            prog.append(I.mm(sa, ssa, out, N, KB, rs, N, M, 0, srs_s, unit=unit,
                             rmax=bool(rng.integers(2)), wf=wf, pair=pair, k0=k0, rsc=rsc))
            out += M * N + M + 8
    return prog


def _region_images(rng, daw, regs, base=None) -> dict:
    """The fuzzers' image layout (test_rtl._images), different data in each region."""
    return {r: (_images(rng, 1)[0], (base or {}).get(r, 0)) for r in regs}


@pytest.mark.parametrize("daw", DAWS)
@pytest.mark.parametrize("kind,seed", [("random", 0), ("random", 1), ("hazard", 0), ("pair", 0)])
def test_regions(have_verilator, daw, kind, seed):
    """LD, ST, QST and MM (int8, 4-bit PAIR, int8 PAIR with DUAL), each in region 0, 1 or the top
    one, the regions holding different data at the same offsets."""
    rng = np.random.default_rng(3300 + 100 * daw + 10 * seed + len(kind))
    cfg = wcfg(daw)
    regs = sorted({0, 1, top(daw)})
    body = {"random": _random_program, "hazard": _hazard_program}.get(kind)
    prog = body(rng, cfg) if body else _pair_mms(rng, cfg) + [I.halt()]
    run_both(cfg, regionize(rng, prog, regs), _region_images(rng, daw, regs))


@pytest.mark.parametrize("daw", DAWS)
@pytest.mark.parametrize("seed,hazard", [(0, False), (1, True), (2, False)])
def test_regions_native_dual(have_verilator, daw, seed, hazard):
    """The same on the board's memory path: otpu_native_dram2's two adapters on the four-channel
    model (BAW DAW - 8: 25, 26, 28) with random stalls; the int8 PAIRs' chunk pairs."""
    rng = np.random.default_rng(3400 + 100 * daw + seed)
    cfg = wcfg(daw)
    regs = sorted({0, 1, top(daw)})
    prog = (_hazard_program if hazard else _random_program)(rng, cfg)
    prog = prog[:-1] + _pair_mms(rng, cfg, 3) + [I.halt()]
    run_both(cfg, regionize(rng, prog, regs), _region_images(rng, daw, regs),
             axi=True, stall=20, seed=seed)


def _end_program(D: int) -> list:
    """Accesses that end at the window's last byte (WIN): with the window at a region's top, at
    the region's (0xFFFFFFFF): an ST of the last chunks, an MM reading them back (the MXU waits for
    the ST: their ranges clamp to the region's end), a QST writing the last rows, LDs of the last
    words, a STREAM / DSTEP whose state ends there."""
    rows, cols = 64, 128
    state = WIN - 4 * rows * cols
    return [I.ld(0, 0, 4096), I.qact(0, 2, 0, 2, 256),
            I.st(WIN - 4 * 8192, 0, 8192),                               # the last 32 KiB
            I.mm(WIN - 64 * 256, 0, 20000, 64, 2, 256, 64, 2, 0, 0, unit=True),
            I.ld(WIN - 4 * 1000, 30000, 1000),
            I.qst(0, WIN - 3 * 2 * D * 2, WIN - 3 * 2 * D * 2 - 64, 3, 2, 2 * D, 2 * D * 2, 2),
            I.ld(WIN - 4 * 600, 32000, 600),
            I.ld(0, 40000, 2 * cols), I.ld(0, 41000, rows),
            I.vop(I.V_FILL, 42000, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 0.9),
            I.vop(I.V_FILL, 42001, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 0.5),
            I.dstep(state, 40000, 41000, rows, cols, 42000, 1, 43000),
            I.ld(WIN - 4 * rows * cols, 45000, rows * cols),
            I.halt()]


@pytest.mark.parametrize("daw", DAWS)
@pytest.mark.parametrize("axi", [False, True])
def test_region_end(have_verilator, daw, axi):
    """The windows at their regions' tops: every range ends at 0xFFFFFFFF in the region (the
    address counters' last word; the scoreboard's clamp), in region 1 and the top region (DAW 33:
    0 and 1), the regions' data different."""
    rng = np.random.default_rng(3500 + daw)
    cfg = wcfg(daw)
    regs = [0, 1] if daw == 33 else [1, top(daw)]
    base = {r: G - WIN for r in regs}
    imgs = {}
    for r in regs:
        img = np.zeros(WIN, np.uint8)
        img[:4 * 4096] = np.float32(rng.standard_normal(4096)).view(np.uint8)
        img[WIN - (1 << 17):] = (0.3 * rng.standard_normal(1 << 15)).astype(np.float32).view(np.uint8)
        imgs[r] = (img, base[r])
    prog = _end_program(cfg.D)
    for r in regs:                 # the program once in each region, its DRAM there
        run_both(cfg, [reloc(x, r, base[r]) for x in prog], imgs,
                 **({"axi": True, "stall": 20, "seed": 3} if axi else {}))
    # and the instructions alternating between the two regions
    run_both(cfg, [reloc(x, regs[i % 2], base[regs[i % 2]]) for i, x in enumerate(prog)], imgs)


@pytest.mark.parametrize("daw", DAWS)
def test_streams_in_regions(have_verilator, daw):
    """STREAM and DSTEP (test_stream_rtl.py's program: every dmode and gate, in place) with their
    DRAM state in region 0, 1 or the top one (flags[7:4], rd), each region's state different."""
    r = np.random.default_rng(3600 + daw)
    cfg = wcfg(daw)
    regs = sorted({0, 1, top(daw)})
    prog = _stream_rtl_program(r, n_steps=12)
    imgs = {}
    for g in regs:
        img = np.zeros(WIN, np.uint8)
        img[:4 * 4096] = _special(r, 4096).view(np.uint8)
        img[0x10000:0x10000 + 4 * 65536] = (0.3 * r.standard_normal(65536)).astype(
            np.float32).view(np.uint8)
        imgs[g] = (img, 0)
    run_both(cfg, regionize(r, prog, regs), imgs)


# ------------------------------------------------------------------------------------ scoreboard
@pytest.mark.parametrize("daw", DAWS)
def test_scoreboard_regions(have_verilator, daw):
    """A long ST in region 1, then an MM reading the same offset: in region 0 the MXU streams
    beside the ST (no conflict), in region 1 it waits for it (the cycles); both bit-exact."""
    cfg = wcfg(daw)
    X = 0x20000
    rng = np.random.default_rng(3700 + daw)
    imgs = _region_images(rng, daw, [0, 1])
    cyc = {}
    for r in (0, 1):
        prog = [I.ld(DATA, 0, 4096), I.qact(0, 2, 0, 2, 256),
                I.st(X, 0, 32768, reg=1),                                   # 1,024 chunks
                I.mm(X, 0, 40000, 256, 2, 256, 256, 2, 0, 0, unit=True, reg=r),
                I.halt()]
        R0 = [(0, 0)] * 16
        assert W.conflict(W.footprint(prog[2], R0), W.footprint(prog[3], R0)) == (r == 1)
        _, st = run_both(cfg, prog, imgs)
        cyc[r] = st["cycles"]
    assert cyc[1] > cyc[0] + 300, cyc


@pytest.mark.parametrize("daw", DAWS)
def test_waitw_all_regions(have_verilator, daw):
    """WAITW writes all of DRAM: an MM of the top region after a WAITW on a region-0 flag waits
    for it, and reads the weights the host wrote before the flag (a range of the flag's region
    alone would let the MXU stream them early)."""
    cfg = wcfg(daw)
    rng = np.random.default_rng(3800 + daw)
    t = top(daw)
    imgs = _region_images(rng, daw, [0, t])
    FLAG, X = 0x1000, 0x30000
    # (the behavioural DRAM takes up to 64 pokes: otpu_dram NPOKE)
    pk = [(2500 + k, (t << 32) + X + 4 * k, int(rng.integers(0, 1 << 32))) for k in range(32)]
    pk.append((2600, FLAG, 7))
    prog = [I.ld(DATA, 0, 4096), I.qact(0, 2, 0, 2, 256),
            I.waitw(FLAG, 30000, 7, I.C_EQ, interval=20, reg=0),
            I.mm(X, 0, 40000, 8, 2, 256, 8, 2, 0, 0, unit=True, reg=t),
            I.halt()]
    _, st = run_both(cfg, prog, imgs, pokes=pk)
    assert st["cycles"] > 2600


# ------------------------------------------------------------------------------------ HBM map
@pytest.mark.parametrize("pcw,pcb,baw,aaw,base", [(22, 3, 25, 33, 3 << 31), (23, 3, 26, 34, 3 << 32),
                                                  (22, 4, 26, 33, 1 << 32), (23, 4, 27, 34, 1 << 33)])
def test_hbm_amap_pcw(have_verilator, tmp_path, pcw, pcb, baw, aaw, base):
    """otpu_hbm_port's beat map (otpu_hbm_amap) at both pseudo-channel sizes: PCW 22 (256 MiB, the
    VU33P / VU35P's 4-high stacks, today's) and 23 (512 MiB, a VU47P's 8-high: 16 GB, AXI 34 bits,
    CH_BASE past 2^33). Each of the channel's 2^(PCW + PCB) beats has its own 64 bytes in the
    channel's span at CH_BASE: the pseudo-channel (address bits PCW+6 up) the beat's low PCB bits
    XOR the next PCB, the rest of the beat the offset in it. (No HBM-port bench in this tree: the
    port itself is only elaborated, by Verilator's lint and the builds.)"""
    rng = np.random.default_rng(pcw * 10 + pcb)
    n = 1 << (pcw + pcb)
    beats = sorted({0, 1, n - 1, n // 2, (1 << pcb) - 1, 1 << pcb, 1 << pcw}
                   | {int(x) for x in rng.integers(0, n, 3000)})
    f = tmp_path / "beats.txt"
    f.write_text("".join(f"{b:x}\n" for b in beats))
    src = [rtlsim.RTL / "boards/sqrl-fk33/otpu_hbm_port.sv", rtlsim.TB / "tb_hbm_amap.sv"]
    exe = rtlsim.build("tb_hbm_amap", src, {"AAW": aaw, "CH_BASE": f"{aaw}'h{base:x}", "PCW": pcw,
                                            "PCB": pcb, "BAW": baw})
    r = rtlsim.run_sim([exe, f"+beats={f}"], timeout=600)
    got = {int(b, 16): int(a, 16) for b, a in re.findall(r"AMAP ([0-9a-f]+) ([0-9a-f]+)", r.stdout)}
    assert "AMAP DONE" in r.stdout and len(got) == len(beats), r.stdout[-2000:]
    m = (1 << pcb) - 1
    for b in beats:
        pc = (b & m) ^ ((b >> pcb) & m)
        want = base | pc << (pcw + 6) | ((b >> pcb) & ((1 << pcw) - 1)) << 6
        assert got[b] == want, f"beat {b:#x}: {got[b]:#x}, want {want:#x}"
        assert base <= got[b] < base + (n << 6) and got[b] < 1 << aaw
    assert len(set(got.values())) == len(beats)
