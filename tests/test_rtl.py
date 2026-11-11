"""RTL (Verilator) vs the bit-exact ISA simulator: kernels, the design configuration, and fuzzing.

Every test runs the same program on the same DRAM images through both and requires identical
DRAM and TMEM contents in every slice.
"""
import re
import subprocess

import numpy as np
import pytest

from opentpu import Config, isa as I
from opentpu.isasim import Machine, SimError, Slice
from opentpu.kernels import attention_decode, attention_layer, mlp
from opentpu.runtime import compile_kernel, run
from opentpu import rtlsim
from opentpu import target as _target

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


@pytest.mark.parametrize("S,d,T,block", [(1, 32, 100, 64), (2, 32, 200, 64), (1, 128, 300, 128)])
def test_attention_decode_fsum_rtl(have_verilator, S, d, T, block):
    """FSUM (Config.FSUM, otpu_vpu FSUM): attention's sum(p) from its EXP2SUBs on the RTL, the
    ISA simulator's state bit for bit, and its output that of the machine without FSUM."""
    big = dict(D=128, ACT_BLOCKS=16) if d == 128 else {}
    args, want = attn_args(np.random.default_rng(T), Hq=4, Hkv=2, d=d, T=T, cap=-(-T // 256) * 256,
                           block=block)
    cfg = Config(S=S, FSUM=True, **big)
    comp, _ = compile_kernel(attention_decode, cfg, **args)
    assert any(i.flags & I.F_FSUM for p in comp.programs for i in p if i.op == I.VOP)
    ri, rr = both(attention_decode, cfg, **args)
    pc, pim = compile_kernel(attention_decode, Config(S=S, **big), **args)
    plain = run(pc, pim, "isa")
    assert np.array_equal(rr.outputs["out"], plain.outputs["out"])
    assert rel(rr.outputs["out"], want) < 0.03


# ------------------------------------------------------------------------------------ fuzzing
DATA, INT8, SCALES, SCALES4, SCRATCH = 0, 16384, 49152, 57344, 65536
# MX scale rows (docs/isa.md "Weight formats") at a fixed pitch, so that an MM may start at any
# row and any block k0: int8x (Sw, a dw byte a block), int4x / fp4x6 (Sw, a reserved word, a
# word a block), NVFP4 (Sw, three reserved words, eight E4M3 bytes a block)
SC8, SC4, SCNV = 0xC0000, 0xD0000, 0xE0000
MX_PITCH = {I.W8: 64, I.W4I: 64, I.W4F: 64, I.WNV: 128}
MX_BASE = {I.W8: SC8, I.W4I: SC4, I.W4F: SC4, I.WNV: SCNV}
MX_BLOCKS = {I.W8: 60, I.W4I: 14, I.W4F: 14, I.WNV: 14}       # blocks a row holds


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
        for wf in (I.W8, I.W4I, I.WNV):
            rows = _mx_scale_rows(wf, 0x10000 // MX_PITCH[wf])
            img[MX_BASE[wf]:MX_BASE[wf] + rows.size] = rows.reshape(-1)
        imgs.append(img)
    return imgs


def _mx_scale_rows(wf, n, seed=0):
    """n MX scale rows of format wf at MX_PITCH: Sw of either sign in [0.01, 0.1), offsets dw
    from 0 to past the shift window (most small), random multipliers / E4M3 scales; a separate
    generator, as _scale_words."""
    r = np.random.default_rng(seed + 10 * wf)
    P, KBr = MX_PITCH[wf], MX_BLOCKS[wf]
    rows = np.zeros((n, P), np.uint8)
    sw = (r.uniform(0.01, 0.1, n) * r.choice([-1, 1], n)).astype(np.float32)
    rows[:, :4] = sw[:, None].view(np.uint8)
    dw = np.where(r.random((n, KBr)) < 0.8, r.integers(0, 6, (n, KBr)),
                  r.integers(0, 20, (n, KBr)))
    if wf == I.W8:
        rows[:, 4:4 + KBr] = dw
    elif wf == I.WNV:
        rows[:, 16:16 + 8 * KBr] = r.integers(0, 0x80, (n, 8 * KBr))
    else:
        m = r.integers(0, 64, (n, KBr, 4))
        word = (dw | sum(m[:, :, b] << (8 + 6 * b) for b in range(4))).astype(np.uint32)
        rows[:, 8:8 + 4 * KBr] = word.view(np.uint8).reshape(n, -1)
    return rows


def _mx_scales(rng, wf, N, KB, pair, unit):
    """An MM's scale fields (ssa, srs, k0, rsc): RSC rows (Sw alone, the fp32 SCALES) a quarter
    of the time when it may, else MX rows at a random row and block k0 (PAIR: even)."""
    if not unit and wf != I.WNV and rng.integers(4) == 0:
        srs = 4 * int(rng.integers(1, 3))
        return SCALES + 4 * int(rng.integers(0, (4096 - N * srs) // 4)), srs, 0, True
    P = MX_PITCH[wf]
    srs = P * int(rng.integers(1, 3))
    ssa = MX_BASE[wf] + P * int(rng.integers(0, (0x10000 - N * srs) // P))
    k0 = int(rng.integers(0, MX_BLOCKS[wf] - KB - 1))
    if pair:
        k0 &= ~1
    return ssa, srs, k0, False


def _qst_scale_bytes(rows, KB, row):
    """QST's scale rows: ROW the word S a row, else S and ceil(KB/4) words of d bytes."""
    return 4 * rows if row else rows * (4 + 4 * -(-KB // 4))


def _scale_words(n, seed=0):
    """4-bit MM scale words: a bf16 scale in [2^-10, 2^-3) and four multipliers in 0..15
    (docs/isa.md, "Weight formats"); a separate generator, so that the rest of the fuzzers'
    random streams do not depend on it."""
    r = np.random.default_rng(seed)
    s = (r.uniform(2.0 ** -10, 2.0 ** -3, n).astype(np.float32).view(np.uint32) >> 16)
    m = r.integers(0, 16, (n, 4)).astype(np.uint32)
    return (s | (m << (16 + 4 * np.arange(4, dtype=np.uint32))).sum(1)).astype(np.uint32)


def _wf(rng):
    """A random MM weight format: int8x half the time, else int4x, fp4x6 or NVFP4."""
    return [I.W8, I.W4I, I.W4F, I.WNV][int(rng.choice([0, 0, 0, 1, 2, 3]))]


def _kinds(kinds: list, cfg: Config, gather: bool | None) -> list:
    """The fuzzers' instruction kinds, without GATHER where the simulated collective unit has
    none (gather None: rtlsim.gathers, the simulation's uarch; the board's at S = 1 is BAR-only,
    as the card builds it)."""
    if not (rtlsim.gathers(cfg.S) if gather is None else gather):
        kinds = [k for k in kinds if k != "gather"]
    return kinds


def _random_program(rng, cfg: Config, n_ops=40, t0: int = 0, gather: bool | None = None,
                    kinds: list | None = None, acc_p: float | None = None):
    """A valid, hazard-free random program. The same program runs on every slice (so the
    collectives line up); the slices differ by their DRAM contents. t0: its TMEM words from
    there (every TMEM address the program names is t0 or more). gather: GATHERs among its
    instructions (None: where the simulation's collective unit runs them, _kinds). kinds: the
    instruction kinds to draw from (None: the default mix); acc_p: an MM's chance to accumulate
    into an earlier MM's output (ACC: the MXU reads TMEM), taking that output's M and N (None:
    half the MMs whose shape an earlier output has, drawn as before)."""
    D = cfg.D
    kinds = _kinds(kinds or ["vop", "vop", "vop", "qact_mm", "qst", "ldst", "loop", "gather"], cfg,
                   gather)
    prog = [I.ld(DATA, t0, 4096)]
    src = [(t0, 4096)]                # TMEM regions holding finite "input" data
    nxt = t0 + 4096                   # fresh TMEM destinations (never read by VOPs again)
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
        kind = rng.choice(kinds)
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
            want = None                           # acc_p: the earlier output it accumulates into
            if acc_p is not None and mm_outs and rng.random() < acc_p:
                want = mm_outs[int(rng.integers(len(mm_outs)))]
                M = want[1]
                KB = min(KB, max(1, (4096 // M - 3) // D))
            ab = int(rng.integers(0, cfg.ACT_BLOCKS - KB + 1))
            srs = KB * D + int(rng.integers(0, 4))
            M = min(M, 4096 // srs)               # D = 128: the rows fit the input region
            assert want is None or M == want[1]
            cs = pick_src(KB * D) if rng.integers(3) == 0 else None
            rsc = pick_src(M) if rng.integers(3) == 0 else None
            dup = 2 * M <= cfg.MCOLS and rng.integers(3) == 0 and want is None
            prog.append(I.qact(pick_src(M * srs), M, ab, KB, srs, row=bool(rng.integers(2)),
                               cscale=cs, rscale=rsc, dup=dup))
            N = int(rng.integers(1, 24)) if want is None else want[2]
            wf = _wf(rng)
            # with DUP the MM reads the copies too (2M rows), or pairs them (PAIR, 4-bit: rows
            # M..2M-1 must be row j's copies, one Sa a row)
            pair = dup and wf != I.W8 and rng.integers(2) == 0
            if dup and not pair and rng.integers(2):
                M *= 2
            rb = KB * D if wf == I.W8 else -(-KB // 2) * D           # 4-bit: two blocks a chunk
            rs = rb + D * int(rng.integers(0, 2))
            sa = INT8 + D * int(rng.integers(0, (32768 - N * rs) // D + 1))   # D aligned
            unit = bool(rng.integers(2))
            ssa, srs_s, k0, rsc_ = _mx_scales(rng, wf, N, KB, pair, unit)
            reuse = [o for o in mm_outs if o[1:] == (M, N)]
            if want is not None:
                out, acc = want[0], True
            elif reuse and rng.integers(2):
                out, acc = reuse[0][0], True
            else:
                out, acc = fresh(M * N + M), False
                mm_outs.append((out, M, N))
            small = M <= cfg.MCOLS                  # RMAX and ASCALE need M <= MCOLS
            asc = pick_src(M) if (small and unit and acc and rng.integers(2)) else None
            prog.append(I.mm(sa, ssa, out, N, KB, rs, N, M, ab, srs_s, unit=unit, acc=acc,
                             rmax=small and bool(rng.integers(2)), ascale=asc, wf=wf, pair=pair,
                             k0=k0, rsc=rsc_))
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
            scratch = sdst + _qst_scale_bytes(rows, KB, row) + 64
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
        assert nxt < cfg.TMEM_WORDS and scratch < SC8
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


@pytest.mark.parametrize("mcols,rows,seed", [(2, 8, 0), (2, 8, 1), (4, 8, 5), (2, 6, 3),
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


@pytest.mark.parametrize("qd,mcols,rows,seed", [(3, 2, 8, 0), (4, 2, 8, 1), (4, 4, 16, 4),
                                                 (4, 2, 6, 3), (7, 4, 8, 5)])
def test_fuzz_mm_queue_depth_qd(have_verilator, qd, mcols, rows, seed):
    """The MXU's command queue QD deep (otpu_mxu QD; bcu1525-x2's MXU_QD 4): the commands behind
    h and n stream their chunks from their accept and reach the pipeline through n.
    test_fuzz_mm_replay's programs (MMs with replays, VOPs, loads and gathers between them):
    the results the ISA's."""
    rng = np.random.default_rng(3000 + seed)
    cfg = Config(S=1, MCOLS=mcols, ACT_ROWS=rows)
    prog = _random_program(rng, cfg, n_ops=60)
    imgs = _images(rng, 1)
    m = Machine(cfg, [prog], [i.copy() for i in imgs]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [i.copy() for i in imgs], uarch={"MXU_QD": qd})
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0], m.slices[0].tmem)


# the single pumped TMEM copy as release r2 builds it (targets/ypcb-00338.toml: r2pgRs)
R2_TMEM = {"TMEM_IMPL": 3, "TMEM_NSB": 2, "TMEM_SBP": 3, "TMEM_RPL": 1, "TMEM_TDP": 1,
           "TMEM_GST": 1, "TMEM_PMP": 1}
# and the slice's arbiter its replays run under (otpu_slice ARB_MASK: RPB every read lane, the
# card's RPB 0 as rtlsim builds it, and WPB 1), whatever the selected target's TMEM
# (nangate45-small's banked one: RPB 1); a board top sets its own (otpu_board RPB)
R2_ARB = {"RPB": _target.NRP * _target.SIM_LANES_MAX, "WPB": 1}


@pytest.mark.parametrize("mcols,rows,seed", [(2, 8, 1), (2, 6, 3)])
def test_one_copy_replays_make_progress(have_verilator, mcols, rows, seed):
    """The single pumped TMEM copy (TMEM_IMPL 3 with replays and guests, on the board's uarch)
    on the two fuzz_mm_replay programs it never finished (docs/tmem_nocopy.md 5.3): a collective
    read waited behind QUANT 3's request while QUANT held for a replay, the DMA's asker asked for
    the collective's words every cycle, and the replay, on that RAM side, lost to it for good
    (QUANT 2's; VPU B's too in [2-8-1]). The results are the ISA's, and they come within a
    million cycles (about 3,000 with the fix). The collective's read is a GATHER's: the unit
    with GATHER (COLL_GATHER 1, every S > 1 build's; the card's one slice builds it BAR-only)."""
    rng = np.random.default_rng(3000 + seed)
    cfg = Config(S=1, MCOLS=mcols, ACT_ROWS=rows)
    prog = _random_program(rng, cfg, n_ops=60, gather=True)
    imgs = _images(rng, 1)
    m = Machine(cfg, [prog], [i.copy() for i in imgs]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [i.copy() for i in imgs],
                                 uarch={**rtlsim.BOARD_UARCH, **R2_TMEM, **R2_ARB, "COLL_GATHER": 1},
                                 max_cycles=1_000_000)
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0], m.slices[0].tmem)


@pytest.mark.parametrize("mcols,rows,seed", [(8, 16, 0), (2, 8, 1)])
def test_one_copy_tmem_of_128k_words(have_verilator, mcols, rows, seed):
    """Release r2's TMEM: 128K words (targets/ypcb-00338.toml) in the single pumped copy (R2_TMEM:
    128 RAMB36), a random program's TMEM data across the first 64K words' end and past it (the
    sequencer's TAW, the banks' rows, the replays' addresses); the results are the ISA's. Its
    GATHERs' addresses too: the unit with GATHER (COLL_GATHER 1, as the progress test's)."""
    rng = np.random.default_rng(4000 + seed)
    cfg = Config(S=1, MCOLS=mcols, ACT_ROWS=rows, TMEM_WORDS=1 << 17)
    prog = _random_program(rng, cfg, n_ops=60, t0=(1 << 16) - 2048, gather=True)
    imgs = _images(rng, 1)
    m = Machine(cfg, [prog], [i.copy() for i in imgs]).run()
    assert m.slices[0].tmem[1 << 16:].any()                 # (words past 64K written)
    drams, tmems, _ = rtlsim.run(cfg, [prog], [i.copy() for i in imgs],
                                 uarch={**rtlsim.BOARD_UARCH, **R2_TMEM, **R2_ARB, "COLL_GATHER": 1},
                                 max_cycles=1_000_000)
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0], m.slices[0].tmem)


@pytest.mark.parametrize("card,seed", [(0, 1), (0, 2), (1, 1), (1, 2), (1, 3)])
def test_random_power_up(have_verilator, monkeypatch, card, seed):
    """Every variable no `initial` block sets powers up at a random value (OTPU_X_INITIAL=unique:
    Verilator's --x-initial unique, seeded by OTPU_X_SEED; docs/asic.md "Power-up state"), as on
    an ASIC, not at 0 as on an FPGA: no power-up value outlives the reset, on the simulation's
    micro-architecture and on the card's one TMEM copy with replays and guests (card). A random
    program's DRAM is the ISA's, and so is its TMEM at every word it writes: a RAM's words are
    not reset (the writes of the cycles before the reset holds land in some), and the ISA starts
    TMEM at 0. The words the program writes are those an ISA run from TMEM all ones agrees on,
    and the program reads no other (its DRAM is the same)."""
    monkeypatch.setenv("OTPU_X_INITIAL", "unique")
    monkeypatch.setenv("OTPU_X_SEED", str(seed))
    rng = np.random.default_rng(5000 + seed)
    uarch = {**rtlsim.BOARD_UARCH, **R2_TMEM, **R2_ARB, "COLL_GATHER": 1} if card else None
    # (the card's WAITC at WIN 32, a target's under OTPU_TARGET: the wait entries for a window
    # of 32, Config.WC_WIN, as the target's Config has them)
    ua = {**rtlsim.UARCH, **(uarch or {})}
    cfg = Config(S=1, MCOLS=2, ACT_ROWS=8,
                 WC_WIN=32 if ua.get("SEQ_WC") and ua.get("WIN", 16) > 16 else 16)
    spc = (cfg.D // 4) // min(cfg.D // 4, cfg.LANES)
    if card and ua.get("SE_R", 0) > spc:     # (SE_R, timing only: at most otpu_dma's segments a
        uarch["SE_R"] = spc                  # chunk, 1 at this D)
    prog = _random_program(rng, cfg, n_ops=60, gather=bool(card))
    imgs = _images(rng, 1)
    m = Machine(cfg, [prog], [i.copy() for i in imgs]).run()
    m1 = Machine(cfg, [prog], [i.copy() for i in imgs])
    m1.slices[0].tmem[:] = 0xFFFFFFFF
    m1.run()
    assert np.array_equal(m1.slices[0].dram, m.slices[0].dram)
    wr = m.slices[0].tmem == m1.slices[0].tmem
    drams, tmems, _ = rtlsim.run(cfg, [prog], [i.copy() for i in imgs], uarch=uarch,
                                 max_cycles=1_000_000)
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0][wr], m.slices[0].tmem[wr])


@pytest.mark.parametrize("seed", range(6))
def test_mxu_reads_on_the_card_tmem(have_verilator, seed, tmp_path, monkeypatch):
    """The MXU's TMEM reads (ACC, ASCALE) on the m8 card's TMEM (R2_TMEM at 128K words, MCOLS 8:
    the MXU's TQ, its read by bank from its flip-flops, otpu_tmem PBR) beside the VPU's, the
    quantizer's and the DMA's reads: programs mostly of MMs that accumulate into earlier outputs,
    so the MXU's reads lose RAMs and replay (in 5 of the 6 seeds). The results are the ISA's, and
    the MXU read (TMEMSTAT's rd, port 0)."""
    log = tmp_path / "tmemstat.log"
    monkeypatch.setenv("OTPU_TMEMSTAT_LOG", str(log))
    rng = np.random.default_rng(9000 + seed)
    cfg = Config(S=1, MCOLS=8, ACT_ROWS=16, TMEM_WORDS=1 << 17)
    prog = _random_program(rng, cfg, n_ops=60, gather=False, acc_p=0.9,
                           kinds=["vop", "vop", "qact_mm", "qact_mm", "qact_mm", "qst", "ldst"])
    assert sum(i.op == I.MM and bool(i.flags & I.F_ACC) for i in prog) >= 4
    imgs = _images(rng, 1)
    m = Machine(cfg, [prog], [i.copy() for i in imgs]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [i.copy() for i in imgs],
                                 uarch={**rtlsim.BOARD_UARCH, **R2_TMEM, **R2_ARB},
                                 max_cycles=1_000_000)
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)
    st = re.search(r"TMEMSTAT .* rd= ?(\d+),[\d,]+ rpl= ?(\d+),", log.read_text())
    assert st and int(st[1]) > 0, log.read_text()


# WPB 2 on the pumped one copy (otpu_tmem IMPL 3 PMP 1: a bank's second write on its RAMs' port B
# at the odd edge; otpu_slice ARB_MASK: a unit is granted unless two units before it write a bank
# it writes)
R2_ARB2 = {**R2_ARB, "WPB": 2}


@pytest.mark.parametrize("seed", range(4))
def test_one_copy_two_writes_a_bank(have_verilator, seed):
    """The pumped one copy at WPB 2 (R2_TMEM, R2_ARB2: the bcu1525-x2:wmx TMEM) on programs of
    MMs that accumulate, VOPs, QACT / MM pairs, QSTs, LDs / STs and GATHERs, every unit writing
    beside the others (the MXU's drain, the VPU's results, the DMA's loads and the collective):
    the results are the ISA's at WPB 1 and 2 (the cycles: test_one_copy_two_writers)."""
    rng = np.random.default_rng(9100 + seed)
    cfg = Config(S=1, MCOLS=8, ACT_ROWS=16, TMEM_WORDS=1 << 17)
    prog = _random_program(rng, cfg, n_ops=60, gather=True, acc_p=0.5,
                           kinds=["vop", "vop", "vop", "qact_mm", "qact_mm", "qst", "ldst",
                                  "gather"])
    imgs = _images(rng, 1)
    m = Machine(cfg, [prog], [i.copy() for i in imgs]).run()
    for w, arb in ((1, R2_ARB), (2, R2_ARB2)):
        ua = {**rtlsim.BOARD_UARCH, **R2_TMEM, **arb, "COLL_GATHER": 1}
        drams, tmems, _ = rtlsim.run(cfg, [prog], [i.copy() for i in imgs], uarch=ua,
                                      max_cycles=1_000_000)
        assert np.array_equal(tmems[0], m.slices[0].tmem), w
        assert np.array_equal(drams[0], m.slices[0].dram), w


def test_one_copy_two_writers(have_verilator):
    """Two units writing TMEM at once, independent: LDs (the DMA) beside VPU FILLs. At WPB 1 a bank
    takes one of them a cycle (the slice's mask arbiter holds the other); at WPB 2 both land, in
    fewer cycles. The results are the ISA's."""
    rng = np.random.default_rng(9200)
    cfg = Config(S=1, MCOLS=8, TMEM_WORDS=1 << 17)
    prog = [I.ld(DATA, 0, 4096),
            I.vop(I.V_FILL, 8192, 0, 0, 64, 256, 256, 0, 0, I.B_SCALAR, 2.5),
            I.ld(DATA, 4096, 4096),
            I.vop(I.V_FILL, 24576, 0, 0, 64, 256, 256, 0, 0, I.B_SCALAR, -1.5),
            I.halt()]
    imgs = _images(rng, 1)
    m = Machine(cfg, [prog], [i.copy() for i in imgs]).run()
    cyc = {}
    for w, arb in ((1, R2_ARB), (2, R2_ARB2)):
        drams, tmems, st = rtlsim.run(cfg, [prog], [i.copy() for i in imgs],
                                      uarch={**rtlsim.BOARD_UARCH, **R2_TMEM, **arb},
                                      max_cycles=1_000_000)
        assert np.array_equal(tmems[0], m.slices[0].tmem), w
        cyc[w] = st["cycles"]
    assert cyc[2] < cyc[1], cyc


@pytest.mark.parametrize("mcols,rows,seed", [(2, 8, 1), (2, 6, 3)])
def test_one_copy_two_writes_replays_make_progress(have_verilator, mcols, rows, seed):
    """test_one_copy_replays_make_progress's programs at WPB 2 (the replays beside a bank's two
    writes): the ISA's results within a million cycles."""
    rng = np.random.default_rng(3000 + seed)
    cfg = Config(S=1, MCOLS=mcols, ACT_ROWS=rows)
    prog = _random_program(rng, cfg, n_ops=60, gather=True)
    imgs = _images(rng, 1)
    m = Machine(cfg, [prog], [i.copy() for i in imgs]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [i.copy() for i in imgs],
                                 uarch={**rtlsim.BOARD_UARCH, **R2_TMEM, **R2_ARB2, "COLL_GATHER": 1},
                                 max_cycles=1_000_000)
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0], m.slices[0].tmem)


def held_collective_program(out: int = SCRATCH) -> list:
    """Every slice: an ST of 4,096 TMEM words (the DMA asking TMEM for its words nearly every
    cycle) and, with no dependency on it, two GATHERs of other words back to back, then the
    gathered words to DRAM (at out + 0x8000 and out + 0x9000)."""
    return [I.ld(DATA, 0, 4096),
            I.st(out, 0, 4096),
            I.gather(1024, 8192, 4, 64, 256, 128, 64),
            I.gather(3000, 10240, 2, 37, 40, 80, 40),
            I.bar(),
            I.st(out + 0x8000, 8192, 512), I.st(out + 0x9000, 10240, 160),
            I.halt()]


@pytest.mark.parametrize("seed", [0, 1])
def test_collective_reads_held_behind_dma_reads(have_verilator, seed, tmp_path, monkeypatch):
    """Two slices on release r2's TMEM (R2_TMEM: the collective the DMA's guest, GST). Each
    GATHER's read is presented a cycle after it forms (TMEM's input stage: otpu_coll's request
    register, RQ) and held there while the guest rule blocks it, with no replay (coll_blk: the
    DMA asks, here for an ST of 4,096 words running beside the GATHERs: slice 0's collective
    loses cycles, TMEMSTAT). The hold must not stall the DMA or lose a word, and the results are
    the ISA's (docs/tmem_nocopy.md 7.1)."""
    log = tmp_path / "tmemstat.log"
    monkeypatch.setenv("OTPU_TMEMSTAT_LOG", str(log))
    rng = np.random.default_rng(7000 + seed)
    cfg = Config(S=2)
    progs = [held_collective_program() for _ in range(2)]
    imgs = _images(rng, 2)
    m = Machine(cfg, progs, [i.copy() for i in imgs]).run()
    drams, tmems, _ = rtlsim.run(cfg, progs, [i.copy() for i in imgs],
                                 uarch={**rtlsim.BOARD_UARCH, **R2_TMEM, **R2_ARB},
                                 max_cycles=1_000_000)
    for s in range(2):
        assert np.array_equal(tmems[s], m.slices[s].tmem), f"slice {s} TMEM"
        assert np.array_equal(drams[s], m.slices[s].dram), f"slice {s} DRAM"
    held = {int(a): int(b) for a, b in
            re.findall(r"TMEMSTAT sid=(\d+) cyc=\d+ lose=\d+,\d+,\d+,(\d+)", log.read_text())}
    # (slice 0's collective port: the GATHERs read slice 0's segment first, while its ST runs)
    assert sorted(held) == [0, 1] and held[0] > 0, held


@pytest.mark.parametrize("mcols,rows", [(4, 8), (8, 16)])
@pytest.mark.parametrize("M,wf,pair", [(1, I.W8, False), (3, I.W8, False), (1, I.W4F, True),
                                       (2, I.WNV, True), (4, I.W4I, False), (6, I.W4F, False),
                                       (9, I.W8, False)])
def test_act_reads_follow_the_mm_rows(have_verilator, mcols, rows, M, wf, pair):
    """The ACT RAM reads only the columns of an MM's output rows (otpu_mxu act_ren): M of the
    MCOLS columns (PAIR: 2M), every column with replay (M > MCOLS); the results are the ISA's.
    Before the gating every column read every cycle (cyc * MCOLS)."""
    if (pair and 2 * M > mcols) or M > rows:
        pytest.skip("PAIR needs 2M <= MCOLS; M <= ACT_ROWS")
    cfg = Config(S=1, MCOLS=mcols, ACT_ROWS=rows)
    D, KB, N = cfg.D, 3, 20
    srs = KB * D
    rs = KB * D if wf == I.W8 else -(-KB // 2) * D
    prog = [I.ld(DATA, 0, 4096), I.qact(0, M, 0, KB, srs, dup=pair),
            I.mm(INT8, MX_BASE[wf], 4096, N, KB, rs, N, M, 0, MX_PITCH[wf], wf=wf, pair=pair),
            I.halt()]
    img = _images(np.random.default_rng(7), 1)[0]
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [img.copy()])
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    a = st["act_reads"][0]
    G = -(-M // mcols)
    cols = mcols if G > 1 else min(mcols, 2 * M if pair else M)
    adv = -(-KB // 2) if pair else KB
    last = (M - (G - 1) * mcols) * (2 if pair else 1)        # the last group's columns
    assert a["cols"] == mcols
    assert a["any"] == (1 << cols) - 1                       # no other column ever read
    assert a["pop"] == N * adv * G
    assert a["use"] == N * adv * ((G - 1) * mcols + last)
    assert a["rd"] % cols == 0 and a["use"] <= a["rd"] <= a["cyc"] * cols
    if cols < mcols:
        assert a["rd"] < a["cyc"] * mcols


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
def _hazard_program(rng, cfg: Config, n_ops=60, gather: bool | None = None):
    """Random programs whose instructions read and write a small shared pool of TMEM, ACT RAM
    and DRAM addresses, so that RAW, WAR and WAW hazards between concurrently running units are
    everywhere. Only NaN-free-by-construction ops are used (ADD/SUB/MUL/COPY/ABS/FILL/RSUM).
    gather: as _random_program's."""
    D = cfg.D
    kinds = _kinds(["vop", "vop", "vop", "mm", "qst", "ld", "st", "gather"], cfg, gather)
    POOL = 1536 * max(1, D // 32)                 # TMEM words everybody fights over
    SCR = SCRATCH                                 # DRAM bytes everybody fights over
    prog = [I.ld(DATA, 0, 4096)]

    def region(n):
        return int(rng.integers(0, POOL - n))

    for _ in range(n_ops):
        kind = rng.choice(kinds)
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
            N = int(rng.integers(1, 20))
            wf = _wf(rng)
            # the MM reads this QACT's blocks (one Sa a row): its copies too, or PAIR
            pair = dup and wf != I.W8 and rng.integers(2) == 0
            if dup and not pair and rng.integers(2):
                M *= 2
            rs = KB * D if wf == I.W8 else -(-KB // 2) * D
            sa = SCR + D * int(rng.integers(0, 4096 // D))                   # D aligned
            ors = N + int(rng.integers(0, 2))
            out = region(M * ors + M)
            unit, acc = bool(rng.integers(2)), bool(rng.integers(2))
            asc = region(M) if (unit and acc and rng.integers(2)) else None
            ssa, srs_s, k0, rsc_ = _mx_scales(rng, wf, N, KB, pair, unit)
            prog.append(I.mm(sa, ssa, out, N, KB, rs, ors, M, ab, srs_s, unit=unit, acc=acc,
                             rmax=bool(rng.integers(2)), ascale=asc, wf=wf, pair=pair, k0=k0,
                             rsc=rsc_))
        elif kind == "qst":
            rows, KB = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            es = int(rng.integers(1, 3))
            drs = KB * D * es
            dst = SCR + 4 * int(rng.integers(0, 2048))
            span = (rows - 1) * drs + (KB * D - 1) * es + 1
            row = bool(rng.integers(2))
            while True:                           # data and scales of one QST must not overlap
                sdst = SCR + 4 * int(rng.integers(0, 2048))
                if sdst + _qst_scale_bytes(rows, KB, row) <= dst or sdst >= dst + span:
                    break
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


# A DRAM range ending exactly at 2^32 (the board's DRAM is 4 GiB) wrapped its end to 0 and
# overlapped nothing. An ST into the last 256 bytes waits for a long VOP; a younger LD of those
# words (ready at once on the DMA) or a WAITW on the last word (all of DRAM) must wait for it.
# The RTL's 1 MiB DRAM is the 4 GiB space's top MiB (it decodes the low address bits); the ISA
# simulator has the whole space (sparse).
@pytest.mark.parametrize("young", ["ld", "waitw"])
def test_dram_range_ending_at_4gib(have_verilator, young):
    TOP, END = (1 << 32) - (1 << 20), 1 << 32
    rng = np.random.default_rng(3100)
    # the simulated 1 MiB DRAM is the top MiB of 4 GiB through its low address bits; the image
    # leaves its last 64 KiB free for a booted program (BOOT, ICACHE: rtlsim puts it after the
    # image), so the simulated DRAM is not doubled and the top MiB still lands on the image
    img = rng.integers(0, 256, (1 << 20) - (1 << 16), dtype=np.uint8)
    img[:4 * 4096] = rng.standard_normal(4096).astype(np.float32).view(np.uint8)
    prog = [I.ld(TOP, 0, 4096),
            I.vop(I.V_FILL, 8192, 0, 0, 64, 256, 256, 0, 0, I.B_SCALAR, 2.5),   # ~2K cycles
            I.st(END - 256, 8192 + 64 * 256 - 64, 64)]
    if young == "ld":
        prog.append(I.ld(END - 256, 30000, 64))
    else:
        prog.append(I.waitw(END - 4, 30000, I.f32bits(2.5), I.C_EQ, timeout=100_000))
    prog.append(I.halt())
    assert len(img) + 2 * 32 * len(prog) <= Config(S=1).DRAM_BYTES       # (wait entries too)
    m = Machine(Config(S=1, DRAM_BYTES=1 << 32), [prog], [None])
    m.slices[0].dram[TOP:TOP + len(img)] = img
    m.run()
    drams, tmems, _ = rtlsim.run(Config(S=1), [prog], [img.copy()])
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram[TOP:])


# Only the fields docs/isa.md writes R[x] + w are register-relative; the counts, shapes and
# strides beside them are immediates. The sequencer added R[rb] to every w2 and R[rc] to every
# w3, so LD/ST's n, QACT's (rows, ab, KB) and srs and GATHER's (rows, cols) moved with registers
# the instructions do not name (the assembler encodes 0 there; these set them).
@pytest.mark.parametrize("op", ["ld", "st", "qact", "gather"])
def test_count_fields_are_immediates(have_verilator, op):
    cfg = Config(S=2)
    D = cfg.D
    rng = np.random.default_rng(3200)
    imgs = _images(rng, 2)
    prog = [I.li(3, 5), I.li(4, 0x100), I.ld(DATA, 0, 4096)]
    if op == "ld":
        prog.append(I.Instr(I.LD, rc=3, w=I._w(DATA + 4096, 5000, 64)))
    elif op == "st":
        prog.append(I.Instr(I.ST, rc=3, w=I._w(SCRATCH, 0, 64)))
    elif op == "qact":
        # rows 2, ab 0, KB 2, srs 2D (R4 would make ab 1, R3 move row 1); the MM reads them
        prog += [I.Instr(I.QACT, rb=4, rc=3, w=I._w(0, 2 | 2 << 16, 2 * D)),
                 I.mm(INT8, SCALES, 5000, 16, 2, 2 * D, 16, 2, 0, 8)]
    else:
        prog.append(I.Instr(I.GATHER, rc=3, w=I._w(0, 5000, 2 | 16 << 16, 32, 16, 64)))
    prog.append(I.halt())
    m = Machine(cfg, [prog, prog], [i.copy() for i in imgs]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog, prog], [i.copy() for i in imgs])
    for s in range(2):
        assert np.array_equal(tmems[s], m.slices[s].tmem), f"slice {s} TMEM"
        assert np.array_equal(drams[s], m.slices[s].dram), f"slice {s} DRAM"


# GATHER's dst is R[rb] + w2 and every slice names the same one (otpu_coll writes at slice 0's),
# here through different registers (the odd slices' programs one instruction longer: with WAITC
# and boot, each slice's wait entries after the longest program, as the host loads them); slices
# that name different ones stop the simulation, as they raise in the ISA simulator.
# hub: otpu_top COLL_HUB (otpu_board's collective at S > 1; 2: at COLL_PIPE, its reads in flight)
@pytest.mark.parametrize("S,hub", [(2, 0), (4, 0), (2, 1), (4, 1), (2, 2), (4, 2)])
def test_gather_dst_through_registers(have_verilator, S, hub):
    cfg = Config(S=S)
    ua = {"COLL_HUB": 1, "COLL_PIPE": int(hub == 2)} if hub else None
    rng = np.random.default_rng(3300)
    imgs = _images(rng, S)
    progs = [[I.ld(DATA, 0, 4096), I.gather(0, 5000, 2, 16, 32, 16, 64), I.halt()] if s % 2 == 0
             else [I.li(6, 40), I.ld(DATA, 0, 4096), I.gather(0, 4960, 2, 16, 32, 16, 64, rb=6),
                   I.halt()] for s in range(S)]
    m = Machine(cfg, progs, [i.copy() for i in imgs]).run()
    drams, tmems, _ = rtlsim.run(cfg, progs, [i.copy() for i in imgs], uarch=ua)
    for s in range(S):
        assert np.array_equal(tmems[s], m.slices[s].tmem), f"slice {s} TMEM"
    progs[1][0] = I.li(6, 48)
    with pytest.raises(SimError, match="disagree"):
        Machine(cfg, progs, [i.copy() for i in imgs]).run()
    with pytest.raises(RuntimeError, match="otpu_coll"):
        rtlsim.run(cfg, progs, [i.copy() for i in imgs], uarch=ua)


@pytest.mark.parametrize("pipe", [0, 1])
@pytest.mark.parametrize("S,r2", [(2, False), (2, True), (3, False), (4, True)])
def test_gathers_through_the_hub(have_verilator, S, r2, pipe):
    """otpu_top COLL_HUB: otpu_board's collective at S > 1 (otpu_coll_hub beside the unit, an
    otpu_coll_port a slice, two register stages each way; S 3 the slices' count not a power of
    two; pipe: COLL_PIPE, the hub's reads in flight). GATHERs of random shapes (rows 1..4, cols 1..300 and 1, 8, 9: whole, partial and
    single lanes; each slice its own source and row stride), back to back and between BARs,
    with an ST of 4,096 TMEM words beside them (r2: release r2's TMEM, the collective the DMA's
    guest, held reads); every slice's TMEM and DRAM are the ISA's."""
    rng = np.random.default_rng(3401 + 10 * S)
    cfg = Config(S=S)
    shapes = [(int(rng.integers(1, 5)), int(rng.integers(1, 301))) for _ in range(6)]
    shapes += [(1, 1), (3, 8), (2, 9)]
    progs: list = [[I.ld(DATA, 0, 4096), I.st(SCRATCH, 0, 4096)] for _ in range(S)]
    dst = 8192
    for i, (rows, cols) in enumerate(shapes):
        drs = cols + int(rng.integers(0, 5))
        seg = rows * drs + int(rng.integers(0, 9))
        for s in range(S):
            srs = cols + int(rng.integers(0, 17))
            src = int(rng.integers(0, 4096 - rows * srs))
            progs[s].append(I.gather(src, dst, rows, cols, srs, drs, seg))
            if i % 3 == 2:
                progs[s].append(I.bar())
        dst += S * seg
    assert dst <= cfg.TMEM_WORDS
    for p in progs:
        p.append(I.halt())
    imgs = _images(rng, S)
    m = Machine(cfg, progs, [i.copy() for i in imgs]).run()
    ua = {"COLL_HUB": 1, "COLL_PIPE": pipe} | ({**R2_TMEM, **R2_ARB} if r2 else {})
    drams, tmems, _ = rtlsim.run(cfg, progs, [i.copy() for i in imgs], uarch=ua,
                                 max_cycles=2_000_000)
    for s in range(S):
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
    # GATHERs where the board's collective unit runs them (its one slice's: BAR alone, _kinds),
    # whichever uarch the session simulates
    kw = {} if prog == "dma" else {"gather": rtlsim.gathers(cfg.S, rtlsim.BOARD_UARCH)}
    p = {"hazard": _hazard_program, "random": _random_program, "dma": _dma_program}[prog](rng, cfg, **kw)
    img = rng.integers(0, 256, 1 << 20, dtype=np.uint8) if prog == "dma" else _images(rng, 1)[0]
    m = Machine(cfg, [p], [img.copy()]).run()
    drams, tmems, st = rtlsim.run(cfg, [p], [img.copy()], axi=True, boot=True, ldc=ldc,
                                  uarch=rtlsim.BOARD_UARCH, plusargs=rtlsim.ldc_plusargs(ldc, mhz))
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert all(r > 0 for r, _ in st["axi_reads"]), st["axi_reads"]


H2C_SPAN = 32 << 10             # each channel's bytes the host load writes: the image's top 2x


def _stream_program(n_ld: int, words: int = 8192, region: int = 768 << 10):
    """Loads alone, one after another through `region` bytes of DRAM, each `words` words into
    TMEM's two halves in turn: the core reading as fast as its DMA moves (decode's weights)."""
    prog = [I.ld((k * 4 * words) % region, (k % 2) * words, words) for k in range(n_ld)]
    return prog + [I.halt()]


def _h2c_cosim(rate: float, mhz: float = 400, n_ld: int = 48):
    """The stream program on the card's channels at `mhz`, alone and beside a host load of `rate`
    GB/s over both channels (otpu_ldc_mem's +h2c_*: 256-byte bursts on XDMA's side of each bridge,
    into the H2C_SPAN bytes below each channel's half of the image, the image's top, which the
    program never touches; zeros over 0x5A). The program goes above the image (rtlsim's boot: the
    simulated DRAM doubles). Returns (ISA DRAM and TMEM, then per run: DRAM, TMEM, stats)."""
    rng = np.random.default_rng(6300)
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    p = _stream_program(n_ld)
    img = rng.integers(0, 256, cfg.DRAM_BYTES, dtype=np.uint8)
    img[-2 * H2C_SPAN:] = 0x5A
    m = Machine(cfg, [p], [img.copy()]).run()
    xhz = 1.25 * mhz * 1e6                          # otpu_ldc_mem's xclk: 1.25x the core's
    load = [f"+h2c_bpk={round(1000 * rate / 2 * 1e9 / xhz)}", "+h2c_burst=256",
            f"+h2c_span={H2C_SPAN}", f"+h2c_base={cfg.DRAM_BYTES // 2 - H2C_SPAN}"]
    out = []
    for plus in ([], load):
        drams, tmems, st = rtlsim.run(cfg, [p], [img.copy()], axi=True, boot=True, ldc=1066,
                                      uarch=rtlsim.BOARD_UARCH,
                                      plusargs=rtlsim.ldc_plusargs(1066, mhz) + plus)
        out.append((drams[0], tmems[0], st))
    return (m.slices[0].dram, m.slices[0].tmem), out


# XDMA's writes beside the core's on the card's channels (docs/offload.md 13.19, the write
# contention): the core's results stay bit-exact, every host burst lands (the image's top ends all
# zeros) and is answered (but those in flight at the end), and otpu_mem_ch's batches (XT, XB: a
# host turn carries a batch, so the controller turns its bus and changes rows once a batch) keep
# the run's extra time per GB written under 0.015 s: 0.006 here (1.3% slower), against 0.034
# without them (XT 0: 7.2% slower).
def test_ldc_host_writes(have_verilator):
    (dram, tmem), ((d0, t0, s0), (d1, t1, s1)) = _h2c_cosim(2.0)
    want = dram.copy()
    want[-2 * H2C_SPAN:] = 0
    assert np.array_equal(t0, tmem) and np.array_equal(d0, dram)
    assert np.array_equal(t1, tmem) and np.array_equal(d1, want)
    h = s1["h2c"]
    assert len(h) == 2 and all(0 <= x["bursts"] - x["bresp"] <= 4 and x["bytes"] >= H2C_SPAN
                               and x["bytes"] == 256 * x["bursts"] for x in h), h
    ct = (s1["cycles"] - s0["cycles"]) / 400e6 / (sum(x["bytes"] for x in h) / 1e9)
    assert ct < 0.015, (ct, s0["cycles"], s1["cycles"], h)


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


# A partial SW beat is read, merged and written whole (above), so a DMA write to the beat's other
# bytes between that read and the write would be lost. A QST's footprint was its byte hull: the
# words of its first and last beats outside it (and of its scales' beats) were free for an ST.
# The footprint now covers its beats' words, so the scoreboard orders the two either way round.
# Each QST writes bytes 32..159 of a 256-byte block (two partial beats) and scale word 1 of a
# beat; STs write words 8..15 of the block's third beat and word 0 of the scale's beat. The LD
# of the QST's row starts it; an LD on the DMA (a read latency, 400 cycles) delays the STs until
# the QST's fill reads are out.
@pytest.mark.parametrize("order", ["qst_first", "st_first"])
@pytest.mark.parametrize("stall,seed", [(0, 1), (30, 2), (60, 3)])
def test_axi_sw_partial_beat_beside_dma_writes(have_verilator, order, stall, seed):
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    D = cfg.D
    rng = np.random.default_rng(8350 + seed)
    img = rng.integers(0, 256, 1 << 20, dtype=np.uint8)
    REG, SC, T, N = 0x40000, 0x48000, 1024, 12
    img[:4 * D * N] = rng.standard_normal(D * N).astype(np.float32).view(np.uint8)
    prog = [I.ld(0x10000, T, 9 * N)]
    for i in range(N):
        blk, sc, src = REG + 256 * i, SC + 64 * i, 2048 + D * i
        qst = I.qst(src, blk + 32, sc + 4, 1, 1, D, D, 1)
        sts = [I.ld(0x60000 + 512 * i, 4096 + 128 * i, int(rng.integers(1, 128))),
               I.st(blk + 160, T + 9 * i, 8), I.st(sc, T + 9 * i + 8, 1)]
        prog.append(I.ld(4 * D * i, src, D))
        prog += [qst] + sts if order == "qst_first" else sts + [qst]
    prog += [I.ld(REG, 8192, 64 * N), I.ld(SC, 8192 + 64 * N, 16 * N), I.halt()]
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [img.copy()], axi=True, boot=True, stall=stall,
                                  seed=seed, lat=400, uarch=rtlsim.BOARD_UARCH,
                                  plusargs=["+axi_dram=1"])
    bad = np.nonzero(drams[0] != m.slices[0].dram)[0]
    assert len(bad) == 0, f"{len(bad)} DRAM bytes differ, first at {[hex(a) for a in bad[:8]]}"
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert sum(d["rmw_a"] for d in st["axi_detail"]) == 0, st["axi_detail"]


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


@pytest.mark.parametrize("impl,nsb,sbp,sbh,rpl,tdp,gst,pmp", [
    (0, 4, 2, 0, 0, 0, 0, 0), (1, 4, 2, 0, 0, 0, 0, 0), (3, 4, 2, 0, 0, 0, 0, 0), (3, 2, 4, 0, 0, 0, 0, 0),
    (3, 8, 2, 0, 0, 0, 0, 0), (3, 4, 2, 2, 0, 0, 0, 0), (3, 4, 2, 0, 1, 0, 0, 0), (3, 4, 2, 2, 1, 0, 0, 0),
    (3, 2, 2, 1, 1, 0, 0, 0), (3, 4, 2, 0, 0, 1, 0, 0), (3, 4, 2, 0, 1, 1, 0, 0), (3, 4, 2, 2, 1, 1, 0, 0),
    (3, 8, 2, 0, 1, 1, 0, 0), (3, 4, 2, 0, 1, 0, 1, 0), (3, 2, 2, 0, 1, 0, 1, 0), (3, 8, 2, 2, 1, 0, 1, 0),
    (3, 4, 2, 0, 1, 1, 0, 1), (3, 2, 2, 0, 1, 1, 0, 2), (3, 1, 2, 0, 1, 1, 0, 1), (3, 1, 2, 0, 1, 1, 0, 2),
    (3, 2, 2, 0, 1, 1, 1, 1)])
def test_tmem_random_traffic(have_verilator, impl, nsb, sbp, sbh, rpl, tdp, gst, pmp):
    """TMEM alone against a reference model; most reads hit the previous cycle's writes, which
    are still in TMEM's registered write stage (the bypass). impl 1: the banked TMEM (copies of
    1R1W bank RAMs shared by ports, one granted read per copy and bank). impl 3 (EXPERIMENTAL,
    docs/tmem_nocopy.md): one copy, nsb RAMs a bank with the sub-bank at word-address bit sbp
    (interleaved groups, or regions; sbh hashed), a port read only while no port before it takes
    one of its RAMs for another word, or (rpl) every read granted and a lost one replayed while
    its group holds; tdp: true dual-port RAMs, one port on their port A with the writes; gst: a
    port the guest of the first (no asker of its own, the host's data); pmp: the RAMs pumped at
    2x (1, a second clock) or its model (2), a write taking no read port; nsb 1: one RAM a bank."""
    exe = rtlsim.build("tb_tmem", [rtlsim.RTL / "mem/otpu_tmem.sv",
                                   rtlsim.RTL / "tech/generic/otpu_tech_ram.sv",
                                   rtlsim.RTL / "exp/tmem_pump/otpu_tmem_pump.sv",
                                   rtlsim.TB / "tb_tmem.sv"],
                       {"IMPL": impl, **({"NSB": nsb, "SBP": sbp, "SBH": sbh, "RPL": rpl, "TDP": tdp,
                                          "GST": gst, "PMP": pmp}
                                         if impl == 3 else {})})
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]


@pytest.mark.parametrize("impl,nsb,sbp,tdp", [(0, 4, 2, 0), (1, 4, 2, 0), (3, 4, 2, 0), (3, 2, 2, 1)])
def test_tmem_random_traffic_input_stage(have_verilator, impl, nsb, sbp, tdp):
    """test_tmem_random_traffic with TMEM's input stage (IQ, release r2: the granted writes
    registered before the selection): a write is seen two cycles after its own, the reads of the
    words still in the write stage take its bypass as before (no replays: the testbench keeps a
    write off its own cycle's reads only)."""
    exe = rtlsim.build("tb_tmem", [rtlsim.RTL / "mem/otpu_tmem.sv",
                                   rtlsim.RTL / "tech/generic/otpu_tech_ram.sv",
                                   rtlsim.RTL / "exp/tmem_pump/otpu_tmem_pump.sv",
                                   rtlsim.TB / "tb_tmem.sv"],
                       {"IMPL": impl, "IQ": 1,
                        **({"NSB": nsb, "SBP": sbp, "TDP": tdp} if impl == 3 else {})})
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]


@pytest.mark.parametrize("port", [None, 4])
def test_tmem_dump_is_the_ram(have_verilator, port):
    """The TMEM dump that assert_same_state compares is the memory's contents, not the writes
    asked of it: a write that lands with bit 0 flipped (+tmem_fault: the registered write stage
    into the block RAMs) shows in the dump, though no instruction reads the word again; in one
    copy only (port 4's: VPU B holds a copy of its own in both TMEMs, a copy per port and the
    banked one), the dump's check of the copies stops the simulation. Before, the dump was a
    shadow of the write requests and saw neither."""
    if port is not None and rtlsim.UARCH.get("TMEM_IMPL") == 3:
        pytest.skip("one TMEM copy (TMEM_IMPL 3): no port has a copy of its own to differ")
    cfg = Config(S=1)
    dram = np.random.default_rng(7).integers(0, 256, 1 << 16, dtype=np.uint8)
    prog = [I.ld(0, 64, 32), I.halt()]          # TMEM 64..95, never read again
    want = Machine(cfg, [prog], [dram.copy()]).run().slices[0].tmem
    w = 64 + 5
    plus = [f"+tmem_fault=0,{w}"] + ([] if port is None else [f"+tmem_fault_port={port}"])
    if port is None:
        tmem = rtlsim.run(cfg, [prog], [dram.copy()], plusargs=plus)[1][0]
        assert list(np.nonzero(tmem != want)[0]) == [w] and tmem[w] == want[w] ^ 1
    else:
        with pytest.raises(RuntimeError, match=f"TMEM0: copy {port}'s word {w} is"):
            rtlsim.run(cfg, [prog], [dram.copy()], plusargs=plus)
    tmem = rtlsim.run(cfg, [prog], [dram.copy()], plusargs=["+tmem_fault=1,69"])[1][0]
    assert np.array_equal(tmem, want)           # (another slice's fault)


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


@pytest.mark.parametrize("board", [False, True])
def test_additive_mask_block_rtl(have_verilator, board):
    """A prompt run's masked attention block (attention.Additive, docs/prefill.md 9): the score
    tile loaded with a mask (-0 where a token counts, -inf past it), then MM ACC + RMAX of
    q.K^T (int8 K with token scales) into it, beside today's way (MM, the +inf / -inf row, VOP
    MIN by column, VOP RMAX). The RTL's ACC keeps -inf + s = -inf and its RMAX takes the max
    after the add, as the ISA simulator; on the ISA the two tiles and their maxima are the same
    bits. Rows 1-4, the row's position 0-255 in the block (-1: every token masked, a sliding
    window's first block at its last position), K blocks 1 and 2."""
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import amask_table
    cfg = board_config(DRAM_BYTES=1 << 22, PAIR=True)
    D, N = cfg.D, 256
    Q, K, KS, ZA, ZM = 0, 0x10000, 0x40000, 0x50000, 0x160000
    rng = np.random.default_rng(7)
    img = np.zeros(1 << 22, np.uint8)
    img[K:K + N * 2 * D] = rng.integers(-127, 128, N * 2 * D).astype(np.int8).view(np.uint8)
    img[KS:KS + 8 * N] = rng.uniform(0.01, 0.1, 2 * N).astype(np.float32).view(np.uint8)
    tab = amask_table(N, cfg.MCOLS)                        # [position, MCOLS, N + 1]
    tab = np.concatenate([np.full((1, cfg.MCOLS, N + 1), -np.inf, np.float32), tab])
    tab[0, :, N] = -0.0                                    # (position -1: all masked)
    img[ZA:ZA + tab.nbytes] = tab.view(np.uint8).ravel()
    img[ZM:ZM + 8 * N] = np.concatenate([np.full(N, np.inf, np.float32),
                                         np.full(N, -np.inf, np.float32)]).view(np.uint8)
    prog, cases, t, qa = [], [], 0, Q
    for M, tpos, KB in [(1, 0, 1), (2, 255, 2), (3, 77, 1), (4, 1, 2), (4, 0, 1), (4, 255, 1),
                        (3, 254, 2), (4, -1, 1)]:
        M = min(M, cfg.MCOLS)                              # a score tile's rows (the table's)
        q = (rng.standard_normal((M, KB * D)) * 2).astype(np.float32)
        img[qa:qa + q.nbytes] = q.view(np.uint8).ravel()
        tq = t
        ta = tq + M * KB * D                               # the additive tile (rows of N + 1)
        tb = ta + M * (N + 1) + M                          # today's tile, its row, its maxima
        tz = tb + M * N
        tx = tz + N
        t = tx + M
        prog += [I.ld(qa, tq, M * KB * D), I.qact(tq, M, 0, KB, KB * D, row=True),
                 I.ld(ZA + tab[0].nbytes * (tpos + 1), ta, (M - 1) * (N + 1) + N),
                 I.mm(K, KS, ta, N, KB, KB * D, N + 1, M, 0, 4 * KB, acc=True, rmax=True),
                 I.mm(K, KS, tb, N, KB, KB * D, N, M, 0, 4 * KB),
                 I.ld(ZM + 4 * (N - 1 - tpos), tz, N),
                 I.vop(I.V_MIN, tb, tb, tz, M, N, N, N, 0, I.B_COL),
                 I.vop(I.V_RMAX, tx, tb, 0, M, N, 1, N, 0)]
        cases.append((M, tpos, ta, tb, tx))
        qa += q.nbytes + 64
    prog.append(I.halt())
    m = Machine(cfg, [prog], [img.copy()]).run()
    tm = m.slices[0].tmem
    for M, tpos, ta, tb, tx in cases:
        a = np.stack([tm[ta + j * (N + 1):ta + j * (N + 1) + N] for j in range(M)])
        b = tm[tb:tb + M * N].reshape(M, N)
        assert np.array_equal(a, b), (M, tpos)            # the same bits, -0 included
        amax = tm[ta + M * (N + 1):ta + M * (N + 1) + M]
        assert np.array_equal(amax, tm[tx:tx + M]), (M, tpos)
        f = a.view(np.float32)
        assert np.isneginf(f[:, tpos + 1:]).all() and np.isfinite(f[:, :tpos + 1]).all()
    kw = dict(uarch=rtlsim.BOARD_UARCH, axi=True, boot=True) if board else {}
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], **kw)
    assert np.array_equal(np.asarray(tmems[0]).view(np.uint32), tm)
    assert np.array_equal(drams[0][:len(m.slices[0].dram)], m.slices[0].dram)
