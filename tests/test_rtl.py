"""RTL (Verilator) vs the bit-exact ISA simulator: kernels, the design configuration, and fuzzing.

Every test runs the same program on the same DRAM images through both and requires identical
DRAM and TMEM contents in every slice.
"""
import subprocess

import numpy as np
import pytest

from opentpu import Config, isa as I
from opentpu.isasim import Machine, SimError, Slice
from opentpu.kernels import attention_decode, attention_layer, mlp
from opentpu.runtime import compile_kernel, run
from opentpu import rtlsim

from conftest import assert_same_state, rel
from test_kernels import attn_args, layer_args, mlp_args


def both(kernel, cfg, **args):
    comp, imgs = compile_kernel(kernel, cfg, **args)
    ri = run(comp, [i.copy() for i in imgs], "isa")
    rr = run(comp, [i.copy() for i in imgs], "rtl")
    assert_same_state(ri, rr)
    return ri, rr


@pytest.mark.parametrize("S", [1, 2, 4])
def test_mlp_rtl(have_verilator, S):
    args, want = mlp_args(np.random.default_rng(S), M=11)
    ri, rr = both(mlp, Config(S=S), **args)
    assert rel(rr.outputs["out"], want) < 0.02


@pytest.mark.parametrize("S,Hq,Hkv,T", [(1, 8, 2, 100), (2, 8, 2, 100), (4, 8, 4, 45)])
def test_attention_decode_rtl(have_verilator, S, Hq, Hkv, T):
    args, want = attn_args(np.random.default_rng(T), Hq=Hq, Hkv=Hkv, T=T)
    ri, rr = both(attention_decode, Config(S=S), **args)
    assert rel(rr.outputs["out"], want) < 0.03


@pytest.mark.parametrize("S", [1, 2])
def test_attention_layer_rtl(have_verilator, S):
    args, (want, knew, vnew) = layer_args(np.random.default_rng(11), S, pos=40)
    ri, rr = both(attention_layer, Config(S=S), **args)
    assert rel(rr.outputs["out"], want) < 0.01
    K, V = rr.kv("kv")
    assert rel(K[:, 40], knew) < 0.02 and rel(V[:, 40], vnew) < 0.02


def test_design_size_block_128_rtl(have_verilator):
    args, want = attn_args(np.random.default_rng(5), Hq=6, Hkv=1, d=128, T=200, cap=256,
                           block=128)
    ri, rr = both(attention_decode, Config(S=1, D=128, ACT_BLOCKS=16), **args)
    assert rel(rr.outputs["out"], want) < 0.03


# ------------------------------------------------------------------------------------ fuzzing
DATA, INT8, SCALES, SCRATCH = 0, 16384, 49152, 65536


def _images(rng, S):
    imgs = []
    for _ in range(S):
        img = np.zeros(1 << 20, np.uint8)
        x = rng.standard_normal(4096).astype(np.float32)
        x[rng.random(4096) < 0.05] = 0.0
        img[DATA:DATA + 16384] = x.view(np.uint8)
        img[INT8:INT8 + 32768] = rng.integers(-127, 128, 32768).astype(np.int8).view(np.uint8)
        img[SCALES:SCALES + 4096] = rng.uniform(0.01, 0.1, 1024).astype(np.float32).view(np.uint8)
        imgs.append(img)
    return imgs


def _random_program(rng, cfg: Config, n_ops=40):
    """A valid, hazard-free random program. The same program runs on every slice (so the
    collectives line up); the slices differ by their DRAM contents."""
    D = cfg.D
    prog = [I.ld(DATA, 0, 4096)]
    src = [(0, 4096)]                 # TMEM regions holding finite "input" data
    nxt = 4096                        # fresh TMEM destinations (never read by VOPs again)
    mm_outs = []
    scratch = SCRATCH

    def fresh(n):
        nonlocal nxt
        a = nxt
        nxt += n
        return a

    def pick_src(n):
        cands = [r for r in src if r[1] >= n]
        base, size = cands[rng.integers(len(cands))]
        return base + int(rng.integers(0, size - n + 1))

    for _ in range(n_ops):
        kind = rng.choice(["vop", "vop", "vop", "qact_mm", "qst", "ldst", "loop", "gather"])
        if kind == "vop":
            rows, cols = int(rng.integers(1, 6)), int(rng.integers(1, 40))
            func = int(rng.choice([I.V_ADD, I.V_SUB, I.V_RSUB, I.V_MUL, I.V_MAX, I.V_MIN,
                                   I.V_COPY, I.V_EXP2, I.V_RECIP, I.V_RSQRT, I.V_ABS, I.V_FILL,
                                   I.V_RSUM, I.V_RMAX, I.V_RSSQ]))
            bmode = int(rng.integers(0, 4))
            ars = cols + int(rng.integers(0, 3))
            a = pick_src(rows * ars)
            brs = int(rng.integers(1, cols + 2))
            b = pick_src(rows * brs + cols)
            imm = float(rng.standard_normal())
            if func in (I.V_RSUM, I.V_RMAX, I.V_RSSQ):
                dst, drs = fresh(rows), 1
            else:
                drs = cols + int(rng.integers(0, 2))
                dst = fresh(rows * drs)
            prog.append(I.vop(func, dst, a, b, rows, cols, drs, ars, brs, bmode, imm))
        elif kind == "qact_mm":
            M, KB = int(rng.integers(1, cfg.MCOLS + 1)), int(rng.integers(1, 5))
            ab = int(rng.integers(0, cfg.ACT_BLOCKS - KB + 1))
            srs = KB * D + int(rng.integers(0, 4))
            cs = pick_src(KB * D) if rng.integers(3) == 0 else None
            rsc = pick_src(M) if rng.integers(3) == 0 else None
            prog.append(I.qact(pick_src(M * srs), M, ab, KB, srs, row=bool(rng.integers(2)),
                               cscale=cs, rscale=rsc))
            N = int(rng.integers(1, 24))
            rs = KB * D + D * int(rng.integers(0, 2))
            sa = INT8 + D * int(rng.integers(0, (32768 - N * rs) // D + 1))   # D aligned
            srs_s = 4 * KB + 4 * int(rng.integers(0, 2))
            ssa = SCALES + 4 * int(rng.integers(0, (4096 - N * srs_s) // 4))
            unit = bool(rng.integers(2))
            reuse = [o for o in mm_outs if o[1:] == (M, N)]
            if reuse and rng.integers(2):
                out, acc = reuse[0][0], True
            else:
                out, acc = fresh(M * N + M), False
                mm_outs.append((out, M, N))
            asc = pick_src(M) if (unit and acc and rng.integers(2)) else None
            prog.append(I.mm(sa, ssa, out, N, KB, rs, N, M, ab, srs_s, unit=unit, acc=acc,
                             rmax=bool(rng.integers(2)), ascale=asc))
            src.append((out, M * N))
        elif kind == "qst":
            rows, KB = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            es = int(rng.integers(1, 5))
            drs = KB * D * es + int(rng.integers(0, 8))
            srs = KB * D
            sdst = (scratch + rows * drs + 64 + 3) // 4 * 4
            prog.append(I.qst(pick_src(rows * srs), scratch, sdst, rows, KB,
                              srs, drs, es, row=bool(rng.integers(2))))
            scratch = sdst + 4 * rows * KB + 64
            scratch = (scratch + 3) // 4 * 4
        elif kind == "ldst":
            n = int(rng.integers(1, 64))
            prog.append(I.st(scratch, pick_src(n), n))
            dst = fresh(n)
            prog.append(I.ld(scratch, dst, n))
            src.append((dst, n))
            scratch += 4 * n + 64
        elif kind == "loop":
            k, n = int(rng.integers(1, 5)), int(rng.integers(1, 20))
            base = fresh(k * n)
            prog += [I.li(3, base), I.loop(2, k),
                     I.vop(I.V_MUL, 0, pick_src(n), 0, 1, n, 0, 0, 0, I.B_SCALAR, 0.5, ra=3),
                     I.addi(3, 3, n)]
        else:
            rows, cols = int(rng.integers(1, 4)), int(rng.integers(1, 20))
            dst = fresh(rows * cols * cfg.S)
            prog.append(I.gather(pick_src(rows * cols), dst, rows, cols, cols, cols * cfg.S, cols))
        assert nxt < cfg.TMEM_WORDS and scratch < (1 << 20)
    prog.append(I.halt())
    return prog


@pytest.mark.parametrize("seed", range(8))
def test_fuzz_single_slice(have_verilator, seed):
    rng = np.random.default_rng(1000 + seed)
    cfg = Config(S=1)
    prog = _random_program(rng, cfg)
    imgs = _images(rng, 1)
    m = Machine(cfg, [prog], [i.copy() for i in imgs]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [i.copy() for i in imgs])
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0], m.slices[0].tmem)


@pytest.mark.parametrize("seed", range(4))
def test_fuzz_two_slices(have_verilator, seed):
    rng = np.random.default_rng(2000 + seed)
    cfg = Config(S=2)
    prog = _random_program(rng, cfg)
    imgs = _images(rng, 2)
    m = Machine(cfg, [prog, prog], [i.copy() for i in imgs]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog, prog], [i.copy() for i in imgs])
    for s in range(2):
        assert np.array_equal(drams[s], m.slices[s].dram), f"slice {s} DRAM"
        assert np.array_equal(tmems[s], m.slices[s].tmem), f"slice {s} TMEM"


@pytest.mark.parametrize("lanes", [4, 16])
def test_lane_count_does_not_change_results(have_verilator, lanes):
    """LANES only changes timing: the same kernels give the same bits at any lane count."""
    args, want = attn_args(np.random.default_rng(3), Hq=8, Hkv=2, T=70)
    ri, rr = both(attention_decode, Config(S=2, LANES=lanes, MCOLS=min(8, lanes)), **args)
    assert rel(rr.outputs["out"], want) < 0.03
    args, want = mlp_args(np.random.default_rng(4), M=3)
    ri, rr = both(mlp, Config(S=2, LANES=lanes, MCOLS=min(8, lanes)), **args)
    assert rel(rr.outputs["out"], want) < 0.02


# ------------------------------------------------------------------ scoreboard stress fuzzing
def _hazard_program(rng, cfg: Config, n_ops=60):
    """Random programs whose instructions read and write a small shared pool of TMEM, ACT RAM
    and DRAM addresses, so that RAW, WAR and WAW hazards between concurrently running units are
    everywhere. Only NaN-free-by-construction ops are used (ADD/SUB/MUL/COPY/ABS/FILL/RSUM)."""
    D = cfg.D
    POOL = 1536 * max(1, D // 32)                 # TMEM words everybody fights over
    SCR = SCRATCH                                 # DRAM bytes everybody fights over
    prog = [I.ld(DATA, 0, 4096)]

    def region(n):
        return int(rng.integers(0, POOL - n))

    for _ in range(n_ops):
        kind = rng.choice(["vop", "vop", "vop", "mm", "qst", "ld", "st", "gather"])
        if kind == "vop":
            rows, cols = int(rng.integers(1, 5)), int(rng.integers(1, 40))
            func = int(rng.choice([I.V_ADD, I.V_SUB, I.V_MUL, I.V_COPY, I.V_ABS, I.V_FILL,
                                   I.V_RSUM, I.V_RSSQ]))
            bmode = int(rng.integers(0, 4))
            ars = cols + int(rng.integers(0, 3))
            brs = int(rng.integers(1, cols + 2))
            a = region(rows * ars)
            b = region(rows * brs + cols)
            if func in (I.V_RSUM, I.V_RSSQ):
                drs = 1
                dst = region(rows)
            else:
                drs = cols + int(rng.integers(0, 2))
                dst = region(rows * drs)
            imm = float(rng.uniform(0.3, 0.9))
            if func == I.V_MUL and bmode != I.B_SCALAR:
                bmode = I.B_SCALAR                # keep magnitudes bounded
            ins = I.vop(func, dst, a, b, rows, cols, drs, ars, brs, bmode, imm)
            # skip VOPs that read their own earlier writes (an ISA-level error, see isasim)
            try:
                probe = Slice(cfg, 0, [ins], None)
                probe._vop(ins)
            except SimError:
                continue
            prog.append(ins)
        elif kind == "mm":
            M, KB = int(rng.integers(1, cfg.MCOLS + 1)), int(rng.integers(1, 4))
            ab = int(rng.integers(0, 8))
            srs = KB * D + int(rng.integers(0, 3))
            cs = region(KB * D) if rng.integers(3) == 0 else None
            rsc = region(M) if rng.integers(3) == 0 else None
            prog.append(I.qact(region(M * srs), M, ab, KB, srs, row=bool(rng.integers(2)),
                               cscale=cs, rscale=rsc))
            N = int(rng.integers(1, 20))
            rs = KB * D
            sa = SCR + D * int(rng.integers(0, 4096 // D))                   # D aligned
            ssa = SCALES + 4 * int(rng.integers(0, 256))
            ors = N + int(rng.integers(0, 2))
            out = region(M * ors + M)
            unit, acc = bool(rng.integers(2)), bool(rng.integers(2))
            asc = region(M) if (unit and acc and rng.integers(2)) else None
            prog.append(I.mm(sa, ssa, out, N, KB, rs, ors, M, int(rng.integers(0, 8)), 4 * KB,
                             unit=unit, acc=acc, rmax=bool(rng.integers(2)), ascale=asc))
        elif kind == "qst":
            rows, KB = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            es = int(rng.integers(1, 3))
            drs = KB * D * es
            dst = SCR + 4 * int(rng.integers(0, 2048))
            span = (rows - 1) * drs + (KB * D - 1) * es + 1
            while True:                           # data and scales of one QST must not overlap
                sdst = SCR + 4 * int(rng.integers(0, 2048))
                if sdst + 4 * rows * KB <= dst or sdst >= dst + span:
                    break
            prog.append(I.qst(region(rows * KB * D), dst, sdst, rows, KB, KB * D, drs, es,
                              row=bool(rng.integers(2))))
        elif kind == "ld":
            n = int(rng.integers(1, 80))
            prog.append(I.ld(DATA + 4 * int(rng.integers(0, 2048)), region(n), n))
        elif kind == "st":
            n = int(rng.integers(1, 80))
            prog.append(I.st(SCR + 4 * int(rng.integers(0, 2048)), region(n), n))
        else:
            rows, cols = int(rng.integers(1, 3)), int(rng.integers(1, 30))
            n = rows * cols
            src = region(n)
            while True:                           # source and destination must not overlap
                dst = region(n * cfg.S)
                if dst + n * cfg.S <= src or dst >= src + n:
                    break
            prog.append(I.gather(src, dst, rows, cols, cols, cols * cfg.S, cols))
    prog.append(I.halt())
    return prog


@pytest.mark.parametrize("seed", range(12))
def test_scoreboard_stress_single_slice(have_verilator, seed):
    rng = np.random.default_rng(3000 + seed)
    cfg = Config(S=1)
    prog = _hazard_program(rng, cfg)
    imgs = _images(rng, 1)
    m = Machine(cfg, [prog], [i.copy() for i in imgs]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [i.copy() for i in imgs])
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)


@pytest.mark.parametrize("seed", range(4))
def test_scoreboard_stress_two_slices(have_verilator, seed):
    rng = np.random.default_rng(4000 + seed)
    cfg = Config(S=2)
    prog = _hazard_program(rng, cfg)
    imgs = _images(rng, 2)
    m = Machine(cfg, [prog, prog], [i.copy() for i in imgs]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog, prog], [i.copy() for i in imgs])
    for s in range(2):
        assert np.array_equal(tmems[s], m.slices[s].tmem), f"slice {s} TMEM"
        assert np.array_equal(drams[s], m.slices[s].dram), f"slice {s} DRAM"


# ------------------------------------------------------------------ the board's memory path
# The AXI adapter (two interleaved DDR3 channels) in front of an AXI memory model that stalls
# every handshake and delays every response at random, with the program booted from DRAM by
# the slice's loader: results must stay bit-identical to the ISA simulator.
@pytest.mark.parametrize("seed,stall", [(0, 0), (1, 30), (2, 60), (3, 30), (4, 80), (5, 50)])
def test_board_memory_path_stress(have_verilator, seed, stall):
    rng = np.random.default_rng(5000 + seed)
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    prog = _hazard_program(rng, cfg)
    imgs = _images(rng, 1)
    m = Machine(cfg, [prog], [i.copy() for i in imgs]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [i.copy() for i in imgs], axi=True, boot=True,
                                 stall=stall, seed=seed + 1)
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)


@pytest.mark.parametrize("seed,stall", [(0, 40), (1, 70)])
def test_board_memory_path_fuzz(have_verilator, seed, stall):
    rng = np.random.default_rng(6000 + seed)
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    prog = _random_program(rng, cfg)
    imgs = _images(rng, 1)
    m = Machine(cfg, [prog], [i.copy() for i in imgs]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [i.copy() for i in imgs], axi=True, boot=True,
                                 stall=stall, seed=seed + 7)
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0], m.slices[0].tmem)


def _dma_program(rng, cfg: Config, n_ops=70):
    """LD / ST only: every alignment within a chunk, lengths around the segment and chunk sizes
    and past the DMA's chunk buffer (32 chunks), stores that share chunks, and loads of what
    was just stored."""
    CW, W = cfg.D // 4, min(cfg.D // 4, cfg.LANES)
    lens = [1, 2, W - 1, W, W + 1, CW - 1, CW, CW + 1, 2 * CW + 3, 40 * CW + 5]
    SRC, DST = 0, 1 << 18                           # DRAM bytes
    prog = []
    for _ in range(n_ops):
        n = int(rng.choice(lens)) if rng.integers(3) else int(rng.integers(1, 6 * CW))
        n = max(n, 1)
        t = int(rng.integers(0, (1 << 14) - n))
        kind = rng.choice(["ld", "st", "st_run", "st_ld"])
        if kind == "ld":
            prog.append(I.ld(SRC + 4 * int(rng.integers(0, 1 << 15)), t, n))
        elif kind == "st":
            prog.append(I.st(DST + 4 * int(rng.integers(0, 1 << 15)), t, n))
        elif kind == "st_run":                      # short adjacent stores: shared chunks
            d = DST + 4 * int(rng.integers(0, 1 << 15))
            for _ in range(int(rng.integers(2, 6))):
                k = int(rng.integers(1, W + 2))
                prog.append(I.st(d, int(rng.integers(0, (1 << 14) - k)), k))
                d += 4 * k
        else:                                       # store, then load it back elsewhere
            d = DST + 4 * int(rng.integers(0, 1 << 15))
            prog.append(I.st(d, t, n))
            prog.append(I.ld(d + 4 * int(rng.integers(0, 3)), (1 << 14) + t, n))
    prog.append(I.halt())
    return prog


# The DMA moves whole chunks on DRAM port B and one segment per cycle on TMEM: partial and
# misaligned first / last segments and chunks, long transfers that wrap its chunk buffer, and
# (AXI) random backpressure and response delays.
@pytest.mark.parametrize("D,lanes,axi,stall", [(32, 8, False, 0), (128, 4, False, 0),
                                               (128, 16, False, 0), (128, 8, True, 0),
                                               (128, 8, True, 50), (128, 8, True, 85)])
def test_dma_alignment_stress(have_verilator, D, lanes, axi, stall):
    rng = np.random.default_rng(7000 + D + lanes + stall)
    cfg = Config(S=1, D=D, LANES=lanes, MCOLS=min(8, lanes), ACT_BLOCKS=16)
    prog = _dma_program(rng, cfg)
    img = rng.integers(0, 256, 1 << 20, dtype=np.uint8)
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], axi=axi, boot=axi, stall=stall,
                                 seed=stall + 3, uarch=rtlsim.BOARD_UARCH if axi else None)
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)


def test_tmem_random_traffic(have_verilator):
    """TMEM alone against a reference model; most reads hit the previous cycle's writes, which
    are still in TMEM's registered write stage (the bypass)."""
    exe = rtlsim.build("tb_tmem", [rtlsim.RTL / "mem/otpu_tmem.sv", rtlsim.TB / "tb_tmem.sv"])
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
