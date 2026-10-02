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
DATA, INT8, SCALES, SCALES4, SCRATCH = 0, 16384, 49152, 57344, 65536


def _images(rng, S):
    imgs = []
    for _ in range(S):
        img = np.zeros(1 << 20, np.uint8)
        x = rng.standard_normal(4096).astype(np.float32)
        x[rng.random(4096) < 0.05] = 0.0
        img[DATA:DATA + 16384] = x.view(np.uint8)
        img[INT8:INT8 + 32768] = rng.integers(-127, 128, 32768).astype(np.int8).view(np.uint8)
        img[SCALES:SCALES + 4096] = rng.uniform(0.01, 0.1, 1024).astype(np.float32).view(np.uint8)
        img[SCALES4:SCALES4 + 4096] = _scale_words(1024).view(np.uint8)
        imgs.append(img)
    return imgs


def _scale_words(n, seed=0):
    """4-bit MM scale words: a bf16 scale in [2^-10, 2^-3) and four multipliers in 0..15
    (docs/isa.md, "Weight formats"); a separate generator, so that the rest of the fuzzers'
    random streams do not depend on it."""
    r = np.random.default_rng(seed)
    s = (r.uniform(2.0 ** -10, 2.0 ** -3, n).astype(np.float32).view(np.uint32) >> 16)
    m = r.integers(0, 16, (n, 4)).astype(np.uint32)
    return (s | (m << (16 + 4 * np.arange(4, dtype=np.uint32))).sum(1)).astype(np.uint32)


def _wf(rng):
    """A random MM weight format: int8 half the time, else int4 or E2M1."""
    return [I.W8, I.W4I, I.W4F][int(rng.choice([0, 0, 1, 2]))]


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
            M, KB = int(rng.integers(1, cfg.act_rows + 1)), int(rng.integers(1, 5))
            ab = int(rng.integers(0, cfg.ACT_BLOCKS - KB + 1))
            srs = KB * D + int(rng.integers(0, 4))
            M = min(M, 4096 // srs)               # D = 128: the rows fit the input region
            cs = pick_src(KB * D) if rng.integers(3) == 0 else None
            rsc = pick_src(M) if rng.integers(3) == 0 else None
            dup = 2 * M <= cfg.MCOLS and rng.integers(3) == 0
            prog.append(I.qact(pick_src(M * srs), M, ab, KB, srs, row=bool(rng.integers(2)),
                               cscale=cs, rscale=rsc, dup=dup))
            if dup and rng.integers(2):
                M *= 2                            # the MM reads the copies too
            N = int(rng.integers(1, 24))
            wf = _wf(rng)
            rb = KB * D if wf == I.W8 else -(-KB // 2) * D           # 4-bit: two blocks a chunk
            rs = rb + D * int(rng.integers(0, 2))
            sa = INT8 + D * int(rng.integers(0, (32768 - N * rs) // D + 1))   # D aligned
            # PAIR (column reuse): its scale pairs are 8-byte aligned; rows M..2M-1 may hold
            # a DUP copy or whatever an earlier QACT left
            pair = wf != I.W8 and 2 * M <= cfg.MCOLS and rng.integers(2 if dup else 4) == 0
            al = 8 if pair else 4
            srs_s = al * -(-4 * KB // al) + al * int(rng.integers(0, 2))
            ssa = (SCALES if wf == I.W8 else SCALES4) + \
                al * int(rng.integers(0, (4096 - N * srs_s) // al))
            unit = bool(rng.integers(2))
            reuse = [o for o in mm_outs if o[1:] == (M, N)]
            if reuse and rng.integers(2):
                out, acc = reuse[0][0], True
            else:
                out, acc = fresh(M * N + M), False
                mm_outs.append((out, M, N))
            small = M <= cfg.MCOLS                  # RMAX and ASCALE need M <= MCOLS
            asc = pick_src(M) if (small and unit and acc and rng.integers(2)) else None
            prog.append(I.mm(sa, ssa, out, N, KB, rs, N, M, ab, srs_s, unit=unit, acc=acc,
                             rmax=small and bool(rng.integers(2)), ascale=asc, wf=wf, pair=pair))
            src.append((out, M * N))
        elif kind == "qst":
            rows, KB = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            es = int(rng.integers(1, 5))
            drs = KB * D * es + int(rng.integers(0, 8))
            srs = KB * D
            sdst = (scratch + rows * drs + 64 + 3) // 4 * 4
            row = bool(rng.integers(2))
            prog.append(I.qst(pick_src(rows * srs), scratch, sdst, rows, KB,
                              srs, drs, es, row=row, half=row and bool(rng.integers(2))))
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


@pytest.mark.parametrize("mcols,rows,seed", [(2, 8, 0), (2, 8, 1), (4, 8, 2), (2, 6, 3),
                                              (4, 16, 4)])
def test_fuzz_mm_replay(have_verilator, mcols, rows, seed):
    """ACT RAM rows beyond the MXU columns (ACT_ROWS): an MM of M > MCOLS rows replays each
    streamed chunk for every group of MCOLS rows; the results are the ISA's."""
    rng = np.random.default_rng(3000 + seed)
    cfg = Config(S=1, MCOLS=mcols, ACT_ROWS=rows)
    prog = _random_program(rng, cfg, n_ops=60)
    assert any(i.op == I.MM and (i.w[5] >> 16) & 0xFF > mcols for i in prog)
    imgs = _images(rng, 1)
    m = Machine(cfg, [prog], [i.copy() for i in imgs]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [i.copy() for i in imgs])
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0], m.slices[0].tmem)


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
            dup = 2 * M <= cfg.MCOLS and rng.integers(3) == 0
            prog.append(I.qact(region(M * srs), M, ab, KB, srs, row=bool(rng.integers(2)),
                               cscale=cs, rscale=rsc, dup=dup))
            if dup and rng.integers(2):
                M *= 2                            # the MM reads the copies too
            N = int(rng.integers(1, 20))
            wf = _wf(rng)
            rs = KB * D if wf == I.W8 else -(-KB // 2) * D
            sa = SCR + D * int(rng.integers(0, 4096 // D))                   # D aligned
            pair = wf != I.W8 and 2 * M <= cfg.MCOLS and rng.integers(2 if dup else 4) == 0
            al = 8 if pair else 4                 # PAIR: 8-byte aligned scale pairs
            ssa = (SCALES if wf == I.W8 else SCALES4) + al * int(rng.integers(0, 1024 // al))
            ors = N + int(rng.integers(0, 2))
            out = region(M * ors + M)
            unit, acc = bool(rng.integers(2)), bool(rng.integers(2))
            asc = region(M) if (unit and acc and rng.integers(2)) else None
            prog.append(I.mm(sa, ssa, out, N, KB, rs, ors, M, int(rng.integers(0, 8)),
                             al * -(-4 * KB // al), unit=unit, acc=acc, rmax=bool(rng.integers(2)),
                             ascale=asc, wf=wf, pair=pair))
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
            row = bool(rng.integers(2))
            prog.append(I.qst(region(rows * KB * D), dst, sdst, rows, KB, KB * D, drs, es,
                              row=row, half=row and bool(rng.integers(2))))
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


# The native memory path (otpu_native_dram, one command per beat, in front of the native memory
# model: commands and write data taken independently, read data with jitter and no backpressure,
# n_wdone late): the hazard and random programs, bit-exact; the DDR3 model with long latencies
# for some.
@pytest.mark.parametrize("prog,seed,stall,lat", [("hazard", 0, 0, 20), ("hazard", 1, 50, 20),
                                                 ("hazard", 2, 80, 120), ("random", 3, 40, 20),
                                                 ("random", 4, 70, 400)])
def test_native_memory_path(have_verilator, prog, seed, stall, lat):
    rng = np.random.default_rng(6100 + seed)
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    p = (_hazard_program if prog == "hazard" else _random_program)(rng, cfg)
    imgs = _images(rng, 1)
    m = Machine(cfg, [p], [i.copy() for i in imgs]).run()
    drams, tmems, st = rtlsim.run(cfg, [p], [i.copy() for i in imgs], axi=True, boot=True,
                                  stall=stall, seed=seed + 1, lat=lat,
                                  plusargs=["+axi_dram=1"] if seed % 2 else [])
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert sum(n["part_sw"] for n in st["native"]) == 0, st["native"]


# A streamed step's first instructions (qwen3.fill_logits): FILL keeps -inf on the RTL too, the
# stores cover the region exactly, and the load-back and RLD complete, on the native memory
# path with stalls; a load of the region after them reads the fill.
@pytest.mark.parametrize("stall", [0, 60])
def test_logits_fill_rtl(have_verilator, stall):
    from types import SimpleNamespace
    from opentpu import language as ol
    from opentpu.host.board import FILL_SENTINEL
    from opentpu.llm.qwen3 import _tdesc, fill_logits
    cfg = Config(S=1, D=128, ACT_BLOCKS=16, DRAM_BYTES=1 << 21)
    V, at = 20000, 1 << 19

    @ol.jit
    def kern(m):
        fill_logits(m)
        ol.store(_tdesc(at + 4 * V, (64,)), ol.load(_tdesc(at + 4 * (V - 64), (64,))))
    m = SimpleNamespace(fill=True, v_loc=V, logits=_tdesc(at, (1, V)))
    prog = kern.trace(cfg, 0, {"m": m}).finish()
    img = np.random.default_rng(stall).integers(0, 256, cfg.DRAM_BYTES, np.uint8)
    ref = Machine(cfg, [prog], [img.copy()]).run()
    w = ref.slices[0].dram[at:at + 4 * V + 512].view(np.uint32)
    assert (w[:V + 64] == FILL_SENTINEL).all() and not (w[V + 64:] == FILL_SENTINEL).any()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], axi=True, boot=True, stall=stall,
                                 seed=stall + 3)
    assert np.array_equal(drams[0], ref.slices[0].dram)
    assert np.array_equal(tmems[0], ref.slices[0].tmem)


# The card's channels (rtlsim's LDC: the board's bridge otpu_mem_ch and LiteDRAM's own controller,
# generated with the production core's settings, sim/verilator/otpu_ldc_mem.sv). The controller
# alone, on one channel's sequential reads or writes on one port (the BIST's pattern), moves what
# the card's BIST measures: with memeff's refresh postponing 2 and LiteDRAM's multiplexer, 90.4%
# of peak reading, 89.8% writing on the card (build f8c6c950, docs/litedram.md section 11),
# 90.4% / 89.9% here. The core before it: 91.0% / 90.1% on the card, 90.9% / 90.1% here; behind
# one port's crossbar lock a burst of refreshes costs more than single ones (on two ports, the
# path decode and XDMA take, it costs less: section 11). The figures follow ctl_settings.py's
# MULTIPLEXER: with fastmux (FASTMUX) the model moves 91.65% / 91.49%, with its fallback
# (FASTMUX_SAFE) 90.38% / 90.17%; the first fastmux build's qual checks the card's BIST against
# them (until then they are the model's own).
BIST = {"stock": (0.904, 0.898), "FASTMUX": (0.9165, 0.9149), "FASTMUX_SAFE": (0.9038, 0.9017)}


def _bist_figures():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ctl_settings", rtlsim.ROOT / "tools" / "litedram" / "ctl_settings.py")
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    names = [k for k in ("FASTMUX", "FASTMUX_SAFE") if cs.MULTIPLEXER == getattr(cs, k)]
    stock = cs.MULTIPLEXER == dict(rtw=None, same_cycle=False, direct_wtr=False)
    assert names or stock, f"no BIST figures for MULTIPLEXER {cs.MULTIPLEXER}"
    return BIST[names[0] if names else "stock"]


@pytest.mark.parametrize("we", [0, 1], ids=["read", "write"])
def test_ldc_sequential_is_the_card_bist(have_verilator, tmp_path, we):
    import re
    card = _bist_figures()[we]
    exe = rtlsim.build("tb_ldc_replay", [rtlsim.TB / "otpu_ldc_ch.v",
                                         rtlsim.TB / "tb_ldc_replay.sv"])
    trace = tmp_path / "seq.txt"
    trace.write_text("".join(f"{m} 0 {we} {m}\n" for m in range(1 << 16)))
    out = subprocess.run([str(exe), f"+trace={trace}"], capture_output=True, text=True,
                         timeout=600).stdout
    eff = float(re.search(r"REPLAY ch0 cycles=\d+ .* beats/cycle=([\d.]+)", out).group(1))
    assert abs(eff - card) < 0.005, out


# The whole memory path on the card's channels, the core as fast as, slower and faster than the
# controller: the hazard, random and DMA programs, bit-exact.
@pytest.mark.parametrize("prog,seed,ldc,mhz", [("hazard", 0, 1066, 133.33),
                                               ("random", 1, 1066, 100),
                                               ("dma", 2, 1066, 200)])
def test_ldc_memory_path(have_verilator, prog, seed, ldc, mhz):
    rng = np.random.default_rng(6200 + seed)
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    p = {"hazard": _hazard_program, "random": _random_program, "dma": _dma_program}[prog](rng, cfg)
    img = rng.integers(0, 256, 1 << 20, dtype=np.uint8) if prog == "dma" else _images(rng, 1)[0]
    m = Machine(cfg, [p], [img.copy()]).run()
    drams, tmems, st = rtlsim.run(cfg, [p], [img.copy()], axi=True, boot=True, ldc=ldc,
                                  uarch=rtlsim.BOARD_UARCH, plusargs=rtlsim.ldc_plusargs(ldc, mhz))
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert all(r > 0 for r, _ in st["axi_reads"]), st["axi_reads"]


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


# Port B streams on the board's memory path: long loads that start and end mid-chunk and cross
# 4 KB channel pages, a store read back at once, under backpressure: results bit-exact.
@pytest.mark.parametrize("stall", [0, 30, 50])
def test_axi_read_bursts(have_verilator, stall):
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    CW, PAGE = cfg.D // 4, 2 * 4096                 # a 4 KB page on each channel
    loads = [(PAGE - 3 * cfg.D, 40 * CW + 5), (3 * PAGE + 5 * cfg.D + 28, 70 * CW + 3),
             (5 * PAGE - cfg.D, 2 * CW), (7 * PAGE + 12, 1)]
    prog, t = [], 0
    for a, n in loads:
        prog.append(I.ld(a, t, n))
        t += n
    prog += [I.st(9 * PAGE - 4 * cfg.D + 8, 100, 9 * CW), I.ld(9 * PAGE - 4 * cfg.D, t, 12 * CW),
             I.halt()]
    img = np.random.default_rng(8000).integers(0, 256, 1 << 20, dtype=np.uint8)
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], axi=True, boot=True, stall=stall,
                                 seed=stall + 11, uarch=rtlsim.BOARD_UARCH)
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)


# QST stores go out as single bytes (one byte-enabled word per cycle). The adapter gathers an
# SW beat until another beat is written or it has been idle, so a contiguous store (a K row)
# costs one write per 64-byte beat; a beat left partial (a transposed V column: one byte per
# beat) is read and written whole. No write reaches the memory with a partial byte mask (the
# channel would read-modify-write it). Results bit-exact under random stalls.
@pytest.mark.parametrize("stall", [0, 40])
def test_axi_sw_write_gather(have_verilator, stall):
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    D = cfg.D
    rng = np.random.default_rng(8200 + stall)
    img = rng.integers(0, 256, 1 << 20, dtype=np.uint8)
    img[:4 * 3 * D * 4] = rng.standard_normal(4 * 3 * D).astype(np.float32).view(np.uint8)
    prog = [I.ld(0, 0, 4 * 3 * D),                                        # rows to quantize
            I.qst(0, 0x40000, 0x48000, 2, 2, 2 * D, 2 * D, 1),            # contiguous, 2 rows
            I.qst(2 * 2 * D, 0x50000 + 3, 0x70000, 1, 1, D, 1, 512, row=True),  # strided
            I.qst(0, 0x60010, 0x68004, 1, 1, D, D, 1),                    # unaligned start
            I.ld(0x40000, 1024, 4 * D), I.halt()]
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [img.copy()], axi=True, boot=True, stall=stall,
                                  seed=3, uarch=rtlsim.BOARD_UARCH, plusargs=["+axi_dram=1"])
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    # the adapter reads every partial beat (the strided store's 128 bytes, the scales, the
    # unaligned store's first and last beats) and writes it whole: no controller RMW
    assert sum(d["rmw_a"] for d in st["axi_detail"]) == 0, st["axi_detail"]


# The same beat left partial twice in one QST (row 0 writes the even bytes, row 1 the odd ones)
# and again by the next QST: each read of the beat must see the writes queued before it.
@pytest.mark.parametrize("stall,seed", [(0, 1), (30, 2), (60, 3), (80, 4)])
def test_axi_sw_partial_beat_order(have_verilator, stall, seed):
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    D = cfg.D
    rng = np.random.default_rng(8300 + seed)
    img = rng.integers(0, 256, 1 << 20, dtype=np.uint8)
    img[:4 * 4 * D * 4] = rng.standard_normal(4 * 4 * D).astype(np.float32).view(np.uint8)
    prog = [I.ld(0, 0, 4 * 4 * D),
            I.qst(0, 0x40000, 0x48000, 2, 1, D, 1, 2),                   # interleaved rows
            I.qst(2 * D, 0x40100, 0x48040, 2, 1, D, 3, 2),               # overlaps the first
            I.qst(0, 0x50001, 0x58000, 4, 1, D, 2, 64, row=True),         # strided, 4 rows
            I.ld(0x40000, 1024, 3 * D), I.ld(0x50000, 2048, 64 * D), I.halt()]
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [img.copy()], axi=True, boot=True, stall=stall,
                                  seed=seed, lat=400, uarch=rtlsim.BOARD_UARCH, plusargs=["+axi_dram=1"])
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert sum(d["rmw_a"] for d in st["axi_detail"]) == 0, st["axi_detail"]


# Random QSTs (strides 1..130 bytes, so beats are left partial or whole, and later stores revisit
# the beats of earlier ones) into a 32 KB region, interleaved with loads of it, under random
# stalls and long write latencies: every read sees every store before it.
@pytest.mark.parametrize("seed", range(8))
def test_axi_sw_rmw_fuzz(have_verilator, seed):
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    D = cfg.D
    rng = np.random.default_rng(8400 + seed)
    img = rng.integers(0, 256, 1 << 20, dtype=np.uint8)
    img[:4 * 8 * D * 4] = rng.standard_normal(8 * 4 * D).astype(np.float32).view(np.uint8)
    REG, SZ = 0x40000, 0x8000                         # the stores' 32 KB region
    prog, t = [I.ld(0, 0, 8 * 4 * D)], 8 * 4 * D
    for _ in range(14):
        if rng.integers(3):
            rows = int(rng.integers(1, 3))
            es = int(rng.choice([1, 1, 2, 3, 64, 65, 130]))
            drs = int(rng.integers(1, 3 * D))
            span = (rows - 1) * drs + (D - 1) * es + 1
            dst = REG + int(rng.integers(0, SZ - span))
            sdst = REG + SZ + 4 * int(rng.integers(0, 256))       # scales: their own area
            row = bool(rng.integers(2))
            prog.append(I.qst(4 * D * int(rng.integers(0, 6)), dst, sdst, rows, 1, D, drs, es,
                              row=row, half=row and bool(rng.integers(2))))
        else:
            n = 4 * int(rng.integers(1, 64))
            prog.append(I.ld(REG + 4 * int(rng.integers(0, SZ // 4 - n)), t, n))
            t += n
    prog += [I.ld(REG, t, 4096), I.halt()]                   # (and the DRAM image compares)
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [img.copy()], axi=True, boot=True,
                                  stall=int(rng.integers(0, 70)), seed=seed,
                                  lat=int(rng.choice([20, 120, 400])), uarch=rtlsim.BOARD_UARCH,
                                  plusargs=["+axi_dram=1"])
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert sum(d["rmw_a"] for d in st["axi_detail"]) == 0, st["axi_detail"]


# A transposed V append (8 KV heads x 128 values, each value to its own beat: element stride =
# the cache capacity) on the calibrated DDR3 model: every beat is a read-modify-write, and the
# SW queue (WQD beats per channel) keeps enough of them in flight that the appends cost about
# the channel's reads. All the beats of one token fall on one channel (the rows are chunk
# aligned).
def test_axi_vt_append_throughput(have_verilator):
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    D, CAP, H = cfg.D, 256, 8
    rng = np.random.default_rng(8500)
    img = rng.integers(0, 256, 1 << 20, dtype=np.uint8)
    img[:4 * H * D] = rng.standard_normal(H * D).astype(np.float32).view(np.uint8)
    plus = ["+axi_dram=1", "+axi_map=1", "+axi_tpc=16", "+axi_tpu=15",
                     "+axi_trp=3", "+axi_trcd=3", "+axi_tras=5", "+axi_trc=7", "+axi_trfc=22",
                     "+axi_trefi=1040", "+axi_trmw=29"]
    cyc = []
    for n in (1, 4):
        prog = [I.ld(0, 0, 4 * H * D)]
        prog += [I.qst(0, 0x20000 + t, 0xE0000 + 64 * t, H, 1, D, D * CAP, CAP) for t in range(n)]
        prog.append(I.halt())
        m = Machine(cfg, [prog], [img.copy()]).run()
        drams, _, st = rtlsim.run(cfg, [prog], [img.copy()], axi=True, boot=True, lat=38,
                                  uarch=rtlsim.BOARD_UARCH, plusargs=plus)
        assert np.array_equal(drams[0], m.slices[0].dram)
        assert sum(d["rmw_a"] for d in st["axi_detail"]) == 0, st["axi_detail"]
        cyc.append(st["cycles"])
    # 1024 read-modify-writes on one channel: ~4.1K cycles; with a 16-deep SW queue this was
    # ~7.7K
    per = (cyc[1] - cyc[0]) / 3
    assert per < 5300, cyc


# The MXU's scale stream (port A, one word per chunk) goes out as runs of beats (a read reuses
# the beat of the previous one); a QST between the MMs rewrites some scales (so a run fetched
# before it must not be used after it), under random stalls: bit-exact, and far fewer A reads
# than scale beats.
@pytest.mark.parametrize("stall,seed", [(0, 1), (40, 2), (70, 3)])
def test_axi_scale_runs(have_verilator, stall, seed):
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    D = cfg.D
    rng = np.random.default_rng(8500 + seed)
    img = rng.integers(0, 256, 1 << 20, dtype=np.uint8)
    SC, W = 0x20000, 0x40000                       # scales (fp32), weights (int8)
    img[SC:SC + 4 * 4096] = (rng.random(4096, dtype=np.float32) + 0.5).view(np.uint8)
    img[:4 * 2 * 4 * D * 4] = rng.standard_normal(2 * 4 * D * 4).astype(np.float32).view(np.uint8)
    N, KB = 300, 4
    prog = [I.ld(0, 0, 2 * 4 * D * 4), I.qact(0, 2, 0, KB, 4 * D),
            I.mm(W, SC, 4096, N, KB, KB * D, N + 2, 2, 0, 4 * KB),
            # a new scale right after the first MM's (a run fetched then covers it), and an MM
            # whose scale stream continues there
            I.qst(4096, 0x30000, SC + 4 * KB * N + 8, 1, 1, D, D, 1, row=True),  # after MM 1
            I.mm(W, SC + 4 * KB * N, 16384, 40, KB, KB * D, 42, 2, 0, 4 * KB),
            I.mm(W, SC, 8192, N, KB, KB * D, N + 2, 2, 0, 4 * KB),
            I.mm(W + KB * D, SC + 4 * 3, 12288, N // 2, KB, KB * D, N, 2, 0, 4 * KB + 4),
            I.halt()]
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [img.copy()], axi=True, boot=True, stall=stall,
                                  seed=seed, uarch=rtlsim.BOARD_UARCH, plusargs=["+axi_dram=1"])
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)
    ar_a = sum(d["ar_a"] for d in st["axi_detail"])
    scale_beats = (2 * N + N // 2 + 40) * KB * 4 // 64
    assert ar_a < scale_beats // 3, (ar_a, scale_beats)


# The DMA's chunk writes (an ST; a DSTEP's state write-back) beside the MXU's scale stream: writes
# to other addresses leave the scale runs alone, a write into scales that a run may hold drops
# it (and every run fetched while the writes were outstanding), under random stalls: bit-exact,
# and still far fewer A reads than scale beats (a B write once dropped every run).
@pytest.mark.parametrize("stall,seed", [(0, 1), (40, 2), (70, 3)])
def test_axi_scale_runs_beside_dma_writes(have_verilator, stall, seed):
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    D = cfg.D
    rng = np.random.default_rng(8600 + seed)
    img = rng.integers(0, 256, 1 << 20, dtype=np.uint8)
    SC, W, OUT = 0x20000, 0x40000, 0x80000         # scales (fp32), weights (int8), ST target
    img[SC:SC + 4 * 4096] = (rng.random(4096, dtype=np.float32) + 0.5).view(np.uint8)
    img[:4 * 2 * 4 * D * 4] = rng.standard_normal(2 * 4 * D * 4).astype(np.float32).view(np.uint8)
    T = 24576                                      # TMEM words the STs write from
    N, KB = 300, 4
    mm = lambda out: I.mm(W, SC, out, N, KB, KB * D, N + 2, 2, 0, 4 * KB)    # noqa: E731
    prog = [I.ld(0, 0, 2 * 4 * D * 4), I.qact(0, 2, 0, KB, 4 * D),
            I.ld(0x60000, T, 2048), mm(4096)]
    prog += [I.st(OUT + 8192 * j, T, 2048) for j in range(6)]    # beside the MMs: elsewhere
    prog += [mm(8192), I.st(SC + 4096, T, 64),                    # into the scales mm 3 reads
             mm(12288), I.st(OUT + 0x10000, T, 2048), mm(16384), I.halt()]
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [img.copy()], axi=True, boot=True, stall=stall,
                                  seed=seed, uarch=rtlsim.BOARD_UARCH, plusargs=["+axi_dram=1"])
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)
    ar_a = sum(d["ar_a"] for d in st["axi_detail"])
    scale_beats = 4 * N * KB * 4 // 64
    assert ar_a < scale_beats // 3, (ar_a, scale_beats)


# The host's writes reach DRAM through its own master (XDMA), which the adapter does not see:
# port A's reused beat and runs must not outlive the points where the host may have written
# (otpu_native_dram a_flush: between runs, a program load, a WAITW that held). The 35B's
# back-to-back embed runs took the previous run's scale beat over the host's rewrite. Each case:
# an MM whose scales are in one beat, the host rewriting the beat the next MM reads first (the
# reused beat, or the next channel beat of the run the first MM's miss fetched), then that MM;
# bit-exact against the ISA simulator, which has no such state.
_PA_SC, _PA_W, _PA_FLAG = 0x20000, 0x40000, 0x60000


def _pa_beat(b: int, rest: int = 0) -> int:
    """Byte address of logical beat b (+ rest bytes)."""
    return 64 * b + rest


def _pa_next(b: int) -> int:
    """The logical beat after b in its channel's run (otpu_native_dram's map, CHASH: logical beat
    b = 2 m + ((b % 2) ^ parity(m)) is channel beat m of channel b % 2 ^ parity(m))."""
    par = lambda m: bin(m).count("1") & 1                  # noqa: E731
    m, c = b // 2, (b % 2) ^ par(b // 2)
    return 2 * (m + 1) + (c ^ par(m + 1))


def _pa_setup(cfg, rng):
    D, KB = cfg.D, 4
    img = rng.integers(0, 256, 1 << 20, dtype=np.uint8)
    img[:4 * 2 * KB * D] = rng.standard_normal(2 * KB * D).astype(np.float32).view(np.uint8)
    img[_PA_SC:_PA_SC + 4 * 4096] = (rng.random(4096, dtype=np.float32) + 0.5).view(np.uint8)
    img[_PA_FLAG:_PA_FLAG + 64] = 0
    head = [I.ld(0, 0, 2 * KB * D), I.qact(0, 2, 0, KB, KB * D)]
    # 2 rows x KB scales (8 words, 32 bytes) at R8 + sc: one beat
    mm = lambda sc, out: I.mm(_PA_W, sc, out, 2, KB, KB * D, 4, 2, 0, 4 * KB, rb=8)  # noqa: E731
    return img, head, mm


def _pa_new_scales(rng, at):
    v = (rng.random(8, dtype=np.float32) * 3 + 4).view(np.uint32)
    return [(at + 4 * k, int(x)) for k, x in enumerate(v)]


@pytest.mark.parametrize("case", ["reuse", "run"])
@pytest.mark.parametrize("reload", [False, True])
def test_porta_host_writes_between_runs(have_verilator, case, reload):
    """Two runs of one program in one simulation (the memory path not reset between them, as on
    the card), the host rewriting scales between them: run 2's first A read is the beat run 1
    read last (reuse), or the next channel beat of run 1's run (run: R8 moves the MM's scales
    there); with the program loaded again or not."""
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    rng = np.random.default_rng(8800 + 2 * (case == "run") + reload)
    img, head, mm = _pa_setup(cfg, rng)
    b0 = _PA_SC // 64 + 3
    b1 = b0 if case == "reuse" else _pa_next(b0)
    prog = head + [mm(_pa_beat(b0), 8192), I.st(0x70000, 8192, 4 * 128), I.halt()]
    args1, args2 = [0], [_pa_beat(b1) - _pa_beat(b0)]
    pokes = _pa_new_scales(rng, _pa_beat(b1))
    m1 = Machine(cfg, [prog], [img.copy()], args=args1).run()
    d1 = m1.slices[0].dram.copy()
    for a, v in pokes:
        d1[a:a + 4] = np.array([v], "<u4").view(np.uint8)
    m2 = Machine(cfg, [prog], [d1], args=args2)
    m2.slices[0].tmem[:] = m1.slices[0].tmem
    m2.run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [img.copy()], axi=True, boot=True, stall=30,
                                  seed=5, uarch=rtlsim.BOARD_UARCH, args=args1, reload=reload,
                                  again=[{"args": args2, "pokes": {0: pokes}}])
    assert np.array_equal(tmems[0], m2.slices[0].tmem)
    assert np.array_equal(drams[0], m2.slices[0].dram)


@pytest.mark.parametrize("case", ["reuse", "run"])
def test_porta_host_writes_before_a_waitw(have_verilator, case):
    """In one run: an MM, a WAITW on a flag the host sets after rewriting scales, an MM whose
    first A read is the rewritten beat (the one the first MM read last, or the next channel beat
    of its run); the WAITW's footprint orders the MM after it, and its end drops port A's beats."""
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    rng = np.random.default_rng(8810 + (case == "run"))
    img, head, mm = _pa_setup(cfg, rng)
    b0 = _PA_SC // 64 + 5
    b1 = b0 if case == "reuse" else _pa_next(b0)
    prog = head + [mm(_pa_beat(b0), 8192), I.waitw(_PA_FLAG, 4000, 1, I.C_EQ, interval=40),
                   mm(_pa_beat(b1), 12288), I.st(0x70000, 8192, 4 * 128),
                   I.st(0x71000, 12288, 4 * 128), I.halt()]
    pokes = [(20000, a, v) for a, v in _pa_new_scales(rng, _pa_beat(b1))] + \
        [(20010, _PA_FLAG, 1)]
    m = Machine(cfg, [prog], [img.copy()], args=[0])

    def host(mach):
        for _, a, v in pokes:
            mach.slices[0].m32[a // 4] = np.uint32(v)
    m.host = host
    m.run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [img.copy()], axi=True, boot=True, stall=30,
                                  seed=7, uarch=rtlsim.BOARD_UARCH, args=[0],
                                  pokes={0: pokes})
    assert st["cycles"] > 20010
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)


@pytest.mark.parametrize("stall,seed", [(0, 1), (40, 2), (70, 3), (40, 4), (30, 5)])
def test_axi_write_bursts(have_verilator, stall, seed):
    """Port B writes (a DMA ST's chunk runs): STs of whole and partial chunks at various offsets
    and lengths (masked first / last beats, a run across a 4 KB page), each read back by an LD
    that depends on it, beside an MM's weight and scale stream, under random stalls: bit-exact
    with the ISA simulator, one write command per beat."""
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    D = cfg.D
    rng = np.random.default_rng(8700 + seed)
    img = rng.integers(0, 256, 1 << 20, dtype=np.uint8)
    SC, W = 0x20000, 0x40000
    img[SC:SC + 4 * 4096] = (rng.random(4096, dtype=np.float32) + 0.5).view(np.uint8)
    img[:4 * 2 * 4 * D * 4] = rng.standard_normal(2 * 4 * D * 4).astype(np.float32).view(np.uint8)
    T = 24576
    N, KB = 200, 4
    prog = [I.ld(0, 0, 2 * 4 * D * 4), I.qact(0, 2, 0, KB, 4 * D), I.ld(0x60000, T, 4096),
            I.mm(W, SC, 4096, N, KB, KB * D, N + 2, 2, 0, 4 * KB)]
    # (byte address, words): whole runs, unaligned starts and ends, a run across 4 KB pages
    for j, (addr, n) in enumerate([(0x80000, 2048), (0x84004, 1000), (0x88f00, 700),
                                   (0x8c0f4, 3), (0x90000, 4096)]):
        prog += [I.st(addr, T + 7 * j, n), I.ld(addr, 32768 + 5000 * j, n)]
    prog += [I.halt()]
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [img.copy()], axi=True, boot=True, stall=stall,
                                  seed=seed, uarch=rtlsim.BOARD_UARCH, plusargs=["+axi_dram=1"])
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)
    aw = sum(a for a, _ in st["axi_writes"])
    beats = sum(b for _, b in st["axi_writes"])
    assert beats >= (2048 + 1000 + 700 + 3 + 4096) // 16
    assert aw == beats


def test_tmem_random_traffic(have_verilator):
    """TMEM alone against a reference model; most reads hit the previous cycle's writes, which
    are still in TMEM's registered write stage (the bypass)."""
    exe = rtlsim.build("tb_tmem", [rtlsim.RTL / "mem/otpu_tmem.sv", rtlsim.TB / "tb_tmem.sv"])
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]


# ------------------------------------------------------------------ 4-bit weights
@pytest.mark.parametrize("fmt,S,D,pair", [("int4", 2, 32, False), ("fp4", 2, 32, False),
                                          ("fp4", 1, 128, False), ("int4", 1, 128, False),
                                          ("fp4", 1, 128, True), ("int4", 1, 128, True)])
def test_mlp_4bit_rtl(have_verilator, fmt, S, D, pair):
    """The MLP with 4-bit weights (real quantizer output, odd and even block counts per row).
    pair: one row at the board's MCOLS=2 with column reuse (QACT DUP, MM PAIR); H = 384 gives
    gate / up an odd block count, whose dense scale rows (12 bytes) are not 8-byte aligned, so
    those two stay half rate and only W_down runs PAIR (odd-KB PAIR is in the fuzzers)."""
    H, Fd = (384 if pair else 256, 512) if D == 128 else (96, 256)
    args, want = mlp_args(np.random.default_rng(9), M=1 if pair else 3, H=H, Fd=Fd)
    for k in ("w_gate", "w_up", "w_down"):
        args[k].fmt = fmt
    cfg = Config(S=S, D=D, ACT_BLOCKS=16, MCOLS=2, PAIR=True) if pair else \
        Config(S=S, D=D, ACT_BLOCKS=16)
    if pair:
        mms = [p for p in compile_kernel(mlp, cfg, **args)[0].programs[0] if p.op == I.MM]
        assert any(p.flags & I.F_PAIR for p in mms)
    ri, rr = both(mlp, cfg, **args)
    assert rel(rr.outputs["out"], want) < 0.1


@pytest.mark.parametrize("boot", [False, True])
def test_run_arguments_start_in_r8_to_r15(have_verilator, boot):
    """The run's arguments are R8..R15 at the start (docs/isa.md "Arguments"): addresses and a
    loop count taken from them, the same on the RTL as on the ISA simulator; R1..R7 start at 0."""
    from opentpu.isasim import board_config
    cfg = board_config(DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(4)
    dram = rng.integers(0, 256, 1 << 16, dtype=np.uint8)
    prog = [I.loop(3, 1, rcount=10),                    # R10 + 1 times
            I.ld(0, 0, 32, ra=8),                       # 32 words from R8
            I.st(0, 0, 32, ra=9),                       # to R9, then both move on
            I.addi(9, 9, 128),
            I.st(0, 0, 32, ra=1),                       # R1 = 0: to address 0
            I.halt()]
    args = [1024, 8192, 2, 0, 0, 0, 0, 0xFFFFFFF0]
    m = Machine(cfg, [prog], [dram.copy()], args=args).run()
    want = m.slices[0].dram[:1 << 16]
    assert np.array_equal(want[8192:8192 + 3 * 128],
                          np.tile(dram[1024:1024 + 128], 3))   # 3 iterations
    got = rtlsim.run(cfg, [prog], [dram.copy()], args=args, boot=boot)[0][0][:1 << 16]
    assert np.array_equal(want, got)


def test_simulator_child_dies_with_its_parent(tmp_path):
    """rtlsim.run_sim: a simulator whose Python parent is killed (SIGKILL: no finally, no
    atexit) is killed within ~1 s by its watchdog shell, instead of running on under launchd;
    a timeout kills its process group."""
    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path
    from opentpu import rtlsim
    with pytest.raises(subprocess.TimeoutExpired):
        rtlsim.run_sim(["sleep", "30"], timeout=0.3)
    pidf = tmp_path / "pid"
    code = (f"import os, sys; sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r});"
            "from opentpu import rtlsim; "
            f"rtlsim.run_sim(['sh', '-c', 'echo $$ > {pidf}; exec sleep 60'])")
    parent = subprocess.Popen([sys.executable, "-c", code])
    for _ in range(100):
        if pidf.exists() and pidf.read_text().strip():
            break
        time.sleep(0.05)
    child = int(pidf.read_text())
    os.kill(parent.pid, signal.SIGKILL)
    parent.wait()
    for _ in range(60):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    os.kill(child, signal.SIGKILL)
    raise AssertionError("the simulator outlived its parent")
