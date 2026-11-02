"""Wide DRAM (Config.DAW 33..36, docs/wide_dram_hbm.md) in the compiler, the layouts and the host.

A DRAM byte address is {REG, (R[x] + imm) mod 2^32}: REG is the instruction's 4 GiB region and
every DRAM operand of the instruction lies in it; no instruction's DRAM range crosses a region
boundary (isa.py "DRAM address width"). The compiler (Builder.region / dram / same_region) sets
REG from each operand's address and its range, and refuses what breaks the rules; the layouts
keep a matrix with its scale rows and a cache with its scale rows in one region, start a tensor
that would straddle a boundary at the next region and split the layer runs (hardware loops) at
4 GiB (opentpu/llm/regions.py, runtime.place); the host follows the bitstream's DRAM.

* DAW 32 is unchanged: no instruction has a region, and a wide configuration whose tensors sit
  below 4 GiB compiles the same words;
* relocation: the same kernels and tiny models with their tensors (layer blocks) above 4 GiB at
  DAW 33, 34 and 36 give the DAW 32 placement's outputs bit for bit on the ISA simulator (its
  DRAM is np.zeros: lazy), from programs whose words differ in REG alone where nothing moved
  across a boundary;
* the rules: what straddles moves to the next region, a run of layers is split at a region
  boundary, an instruction whose operands would span two regions (or whose range crosses one)
  is refused;
* the host: device_config's DAW, check_fits, programs below 4 GiB, logical addresses past
  4 GiB through the channels, the board backend's program area and spans, and the whole stack
  over the Board protocol with an ISA simulator behind it."""
import dataclasses

import numpy as np
import pytest

from opentpu import Config, isa as I
from opentpu import language as ol
from opentpu import target as T
from opentpu.compiler import (Affine, CompileError, DevVar, QTensor, RunVar, Tensor, current,
                              region_of)
from opentpu.host import regs as R
from opentpu.host.board import (Board, BoardBackend, ConfigMismatch, check_fits, check_hash,
                                channel_map, device_config, device_daw, dram_layout, swapped)
from opentpu.host.fake import FakeTransport
from opentpu.isasim import Machine
from opentpu.kernels import attention_decode, attention_layer, mlp
from opentpu.llm import regions as RG
from opentpu.llm import snapshot as SN
from opentpu.runtime import Input, Output, Weight, _Dram, compile_kernel, launch, place, run

from test_kernels import attn_args, layer_args, mlp_args

G = 1 << 32                                     # a region
DAWS = [33, 34, 36]


def wide(cfg: Config, daw: int) -> Config:
    return dataclasses.replace(cfg, DAW=daw, DRAM_BYTES=1 << daw)


def bits(x):
    return np.asarray(x, np.float32).view(np.uint32)


def unreg(prog) -> list:
    """A program's words without REG (rd of the REG_RD ops, flags[7:4] of STREAM)."""
    out = []
    for ins in prog:
        rd, fl = ins.rd_flags()
        out.append(I.Instr(ins.op, ins.ra, ins.rb, ins.rc, rd, fl, list(ins.w)))
    return I.assemble(out)


def regions_used(progs) -> set:
    return {ins.region() for p in progs for ins in p if ins.op in I.REG_RD + (I.STREAM,)}


# ------------------------------------------------------------------------------ DAW 32 unchanged
def test_dram32_programs_have_no_region_and_a_low_wide_placement_the_same_words():
    """At DAW 32 no instruction carries a region; a DAW 33..36 configuration whose tensors lie
    below 4 GiB compiles word for word the same programs (REG 0 everywhere)."""
    for S in (1, 2):
        c32 = Config(S=S)
        for kernel, mk in ((mlp, mlp_args), (attention_decode, attn_args)):
            args, _ = mk(np.random.default_rng(0))
            comp, _ = compile_kernel(kernel, c32, **args)
            assert regions_used(comp.programs) <= {0}
            for daw in DAWS:
                args, _ = mk(np.random.default_rng(0))
                comp2, _ = compile_kernel(kernel, wide(c32, daw), **args)
                for a, b in zip(comp.programs, comp2.programs):
                    assert np.array_equal(I.assemble(a), I.assemble(b))


def test_dram32_model_programs_are_those_of_a_low_wide_placement(models):
    """A tiny Qwen3.5's decode, step, rows and generate programs at DAW 32 and at DAW 34 with
    a DRAM of 16 GiB (the image fits region 0: laid out as at DAW 32): the same words."""
    W, spec = models("qwen35")
    lo = spec.image(card(32, 1 << 26), 384, 1, 8, lookup=True)
    hi = spec.image(card(34, 1 << 34), 384, 1, 8, lookup=True)
    assert hi.prog_at is None and hi.nbytes == lo.nbytes and hi.loc == lo.loc
    for f in (lambda m: m.compile_decode(2, 259)[0], lambda m: m.compile_step(7),
              lambda m: m.compile_rows([(0, 3), (0, 4), (0, 5)], [2]),
              lambda m: m.compile_generate(1, 3)):
        for a, b in zip(f(lo), f(hi)):
            assert np.array_equal(I.assemble(a), I.assemble(b))


# ------------------------------------------------------------------------------ kernels relocated
def _relocated(kernel, mk, cfg, daw, regions, **kw):
    """The kernel's outputs and programs at DAW 32 (low) and at `daw` with its arguments in
    `regions` (place's): both runs' outputs, and the programs."""
    args, _ = mk(np.random.default_rng(0), **kw)
    low = launch(kernel, cfg, **args)
    args, _ = mk(np.random.default_rng(0), **kw)
    comp, imgs = compile_kernel(kernel, wide(cfg, daw), regions=regions, **args)
    got = run(comp, imgs)
    for n in low.outputs:
        assert np.array_equal(bits(low.outputs[n]), bits(got.outputs[n])), n
    return low, got


@pytest.mark.parametrize("daw", DAWS)
@pytest.mark.parametrize("fmt", ["int8", "int4", "fp4", "nvfp4"])
def test_mlp_above_4gib(daw, fmt):
    """Every argument of the MLP in the top region (a weight's rows and its scale rows there):
    the outputs and the programs of the low placement, but for REG."""
    def mk(rng):
        args, want = mlp_args(rng)
        for k in ("w_gate", "w_up", "w_down"):
            args[k] = Weight(args[k].array, 0, fmt=fmt)
        return args, want
    top = (1 << (daw - 32)) - 1
    low, got = _relocated(mlp, mk, Config(S=1, D=128), daw, top)
    assert regions_used(got.compiled.programs) == {top}
    for a, b in zip(low.compiled.programs, got.compiled.programs):
        assert np.array_equal(I.assemble(a), unreg(b))


@pytest.mark.parametrize("daw", DAWS)
@pytest.mark.parametrize("S", [1, 2])
def test_attention_decode_with_its_kv_above_4gib(daw, S):
    """The KV cache (K, its scale rows, V^T tiles, V scales: one region) in a high region, the
    queries and outputs in another: the outputs of the low placement."""
    top = (1 << (daw - 32)) - 1
    _, got = _relocated(attention_decode, attn_args, Config(S=S), daw,
                        {"kv": top, "q": 1, "out": 0}, T=300, cap=512, block=128)
    assert regions_used(got.compiled.programs) >= {0, top}


@pytest.mark.parametrize("daw", [33, 36])
def test_attention_layer_appends_its_kv_above_4gib(daw):
    """A decode attention layer (QST appends of K and of V^T with their scale rows) with every
    argument in region 1, the KV cache in the top one: the outputs and the KV cache of the low
    placement."""
    S = 2
    cfg = Config(S=S)

    def mk(rng):
        return layer_args(rng, S)
    top = (1 << (daw - 32)) - 1
    low, got = _relocated(attention_layer, mk, cfg, daw,
                          {k: (top if k == "kv" else 1) for k in mk(np.random.default_rng(0))[0]})
    for a, b in zip(low.kv("kv"), got.kv("kv")):
        assert np.array_equal(bits(a), bits(b))
    qst = [i for p in got.compiled.programs for i in p if i.op == I.QST]
    assert qst and all(i.region() == top for i in qst)


@ol.jit
def _loop_matmul(x, w, out, n: int, steps: int):
    """A hardware loop over row blocks of w: the MM's address register walks its rows (the
    blocks' products summed)."""
    xs = ol.quantize(ol.load(x))
    acc = ol.zeros((1, n))
    for i in ol.range(steps):
        ol.dot(xs, w[i * n:(i + 1) * n, :], acc=acc)
    ol.store(out, acc)


@pytest.mark.parametrize("daw", DAWS)
def test_a_matmul_in_a_hardware_loop_above_4gib(daw):
    def mk(rng):
        K, n, steps = 256, 16, 6
        return dict(x=Input(rng.standard_normal((1, K))),
                    w=Weight(rng.standard_normal((n * steps, K)), None, fmt="fp4"),
                    out=Output((1, n)), n=n, steps=steps), None
    top = (1 << (daw - 32)) - 1
    _, got = _relocated(_loop_matmul, mk, Config(S=1), daw, {"w": top, "x": 1})
    mm = [i for i in got.compiled.programs[0] if i.op == I.MM]
    assert mm and all(i.region() == top for i in mm)


@ol.jit
def _dstep_kernel(state, qk, v, eb, state_out, chk, out, in_place: bool = True,
                  zero: bool = False):
    rows, cols = state.shape
    QK, V, E = ol.load(qk), ol.load(v), ol.load(eb)
    o = ol.empty([rows])
    if in_place:
        ol.store(state_out, ol.load(state))
        ol.deltanet_step(state_out, QK, V, E[0:1], E[1:2], o, zero)
    else:                                          # STREAM: the new state to another tensor
        ol.deltanet_step(state, QK, V, E[0:1], E[1:2], o, zero, dst=state_out)
    ol.store(chk, ol.load(state_out))
    ol.store(out, o)


def _dstep_args(rng, in_place=True, zero=False):
    rows, cols = 16, 64
    return dict(state=Input(0.3 * rng.standard_normal((rows, cols))),
                qk=Input(rng.standard_normal(2 * cols)), v=Input(rng.standard_normal(rows)),
                eb=Input([0.7, 0.4]), state_out=Input(np.zeros((rows, cols))),
                chk=Output((rows, cols)), out=Output((rows,)), in_place=in_place,
                zero=zero), None


@pytest.mark.parametrize("daw", [33, 36])
@pytest.mark.parametrize("zero", [False, True], ids=["", "zero"])
@pytest.mark.parametrize("flags,in_place", [(dict(DSTEP=True), True), (dict(STREAM=True), True),
                                            (dict(STREAM=True), False)],
                         ids=["dstep", "stream", "stream-dst"])
def test_a_deltanet_step_on_a_state_above_4gib(daw, flags, in_place, zero):
    """DSTEP and STREAM (in place, and into another state of the same region) on states in a
    high region: the outputs of the low placement; the step's REG is the states' region."""
    top = (1 << (daw - 32)) - 1
    cfg = Config(S=1, **flags)
    _, got = _relocated(_dstep_kernel, _dstep_args, cfg, daw,
                        {"state": top, "state_out": top, "chk": 1}, in_place=in_place,
                        zero=zero)
    st = next(i for i in got.compiled.programs[0] if i.op in (I.STREAM, I.DSTEP))
    assert st.region() == top


# ------------------------------------------------------------------------------ the rules
def _trace(fn, cfg):
    return ol.jit(fn).trace(cfg, 0, {}).finish()


C33 = Config(S=1, DAW=33, DRAM_BYTES=1 << 33)
C36 = Config(S=1, DAW=36, DRAM_BYTES=1 << 36, STREAM=True)


def test_region_of():
    assert region_of(0) == 0 and region_of(G - 4, 4) == 0 and region_of(5 * G + 7, 100) == 5
    with pytest.raises(CompileError, match="crosses a 4 GiB boundary"):
        region_of(G - 4, 8)


def test_static_ranges_take_their_region_and_may_not_cross_one():
    """A load's REG is its range's region, the immediate its 32-bit offset (the words of the
    same load 4 GiB lower); a range across a boundary or past 2^DAW is refused."""
    def ld(a, n):
        def k():
            ol.load(Tensor(Affine(a), (n,), (1,)))
        return _trace(k, C36)
    p = ld(7 * G + 0x1000, 64)
    ins = next(i for i in p if i.op == I.LD)
    assert ins.region() == 7 and ins.w[0] == 0x1000
    assert ld(7 * G - 256, 64)[0].region() == 6
    with pytest.raises(CompileError, match="crosses a 4 GiB boundary"):
        ld(7 * G - 128, 64)
    with pytest.raises(CompileError, match="leaves the DRAM"):
        ld((1 << 36) - 128, 64)


def test_a_loop_may_not_walk_across_a_region():
    """A hardware loop's register walks [base, base + (count - 1) * stride + span): it must stay
    in one region (the run planners split loops there)."""
    def mk(base, steps):
        def k():
            for i in ol.range(steps):
                ol.store(Tensor(Affine(base) + i * 4096, (256,), (1,)), ol.zeros((256,)))
        return k
    p = _trace(mk(3 * G - 8 * 4096, 8), C36)
    assert {i.region() for i in p if i.op == I.ST} == {2}
    with pytest.raises(CompileError, match="crosses a 4 GiB boundary"):
        _trace(mk(3 * G - 8 * 4096, 9), C36)


def test_unbounded_and_device_values_address_region_0_only():
    """A run-time value without a bound, or a device value (an absolute address in a register),
    has no known range: region 0 only (as at DAW 32); with a bound it is checked."""
    def k_run(base, var):
        def k():
            ol.load(Tensor(Affine(base) + var * 512, (128,), (1,)))
        return k
    _trace(k_run(0x1000, RunVar("tok")), C33)
    with pytest.raises(CompileError, match="tok .no bound. addresses region 0 only"):
        _trace(k_run(G + 0x1000, RunVar("tok")), C33)
    p = _trace(k_run(G + 0x1000, RunVar("tpos", bound=256)), C33)
    assert next(i for i in p if i.op == I.LD).region() == 1
    with pytest.raises(CompileError, match="crosses"):
        _trace(k_run(2 * G - 512 * 100, RunVar("tpos", bound=256)), C36)

    def k_dev(base):
        def k():
            b = current()
            r = b.scratch()
            ol.load(Tensor(Affine(base) + DevVar("slot", r), (128,), (1,)))
            b.unscratch(r)
        return k
    _trace(k_dev(64), C33)
    with pytest.raises(CompileError, match="a device value"):
        _trace(k_dev(G + 64), C33)


def test_an_mm_takes_its_weights_and_scale_rows_from_one_region():
    """MM's REG covers the weight rows and the DRAM scale rows: a matrix whose scale rows are in
    another region is refused, as are rows or scale rows that cross a boundary."""
    D = C36.D

    def mm(data, scale, n=8, k=256):
        def f():
            x = ol.quantize(ol.full((1, k), 1.0))
            ol.dot(x, QTensor.weight(data, scale, n, k, "int8", D))
        return _trace(f, C36)
    p = mm(5 * G + 0x8000, 5 * G + 0x20000)
    assert {i.region() for i in p if i.op == I.MM} == {5}
    with pytest.raises(CompileError, match="regions \\[5, 6\\]"):
        mm(5 * G + 0x8000, 6 * G + 0x20000)
    with pytest.raises(CompileError, match="mm weights.*crosses"):
        mm(6 * G - 1024, 5 * G)
    with pytest.raises(CompileError, match="mm scales.*crosses"):
        mm(5 * G, 6 * G - 64)


def test_a_qst_and_a_stream_take_both_their_ranges_from_one_region():
    def qst(d, s):
        def f():
            current().store_quantized(ol.full((2, 128), 1.0), Affine(d), Affine(s), 128, 1,
                                      row_scale=True)
        return _trace(f, C36)
    assert next(i for i in qst(9 * G, 9 * G + 4096) if i.op == I.QST).region() == 9
    with pytest.raises(CompileError, match="qst .*regions \\[9, 10\\]"):
        qst(9 * G, 10 * G)

    def stream(a, b, zero=False):
        def f():
            current().stream(I.StreamDesc(8, 64), Tensor(Affine(a), (8, 64), (64, 1)), None, None,
                             None, None, zero=zero, dst=Tensor(Affine(b), (8, 64), (64, 1)))
        return _trace(f, C36)
    st = next(i for i in stream(3 * G + 0x1000, 3 * G + 0x4000) if i.op == I.STREAM)
    assert st.region() == 3 and st.w[1] == 0x1000 and st.w[2] == 0x4000
    with pytest.raises(CompileError, match="stream: its DRAM operands lie in regions \\[3, 4\\]"):
        stream(3 * G + 0x1000, 4 * G + 0x4000)


def test_the_host_arguments_never_straddle_and_stay_with_their_scale_rows():
    """runtime._Dram: an allocation that would cross a region boundary starts at the next region
    (after the one before otherwise); a group (a Weight's rows and scale rows, a KV head's four
    regions) lies in one region; at DAW 32 the addresses are as ever."""
    d = _Dram(1, 1 << 34, wide=True)
    a = d.alloc(G - 4096)
    assert a == 0
    b, c = d.alloc_group([8192, 4096])
    assert (b, c) == (G, G + 8192)
    assert d.alloc(100, 3) == 3 * G and d.alloc(100) == G + 12288
    with pytest.raises(MemoryError, match="region 3 is full"):
        d.alloc(G, 3)
    n = _Dram(1, 1 << 24)
    assert n.alloc_group([1000, 10]) == [0, 1024] and n.alloc(5) == 1152
    cfg = Config(S=1, DAW=34, DRAM_BYTES=1 << 34)
    args = dict(pad=Input(np.zeros(G // 4 - 64, np.float32)),
                w=Weight(np.ones((64, 256), np.float32)))
    _, lay, _ = place(cfg, args)
    da, sa = lay.params["w"][1]
    assert da == G and region_of(da) == region_of(sa) == 1
    with pytest.raises(ValueError, match="DAW > 32"):
        place(Config(S=1), args, regions=1)


def test_the_snapshot_copies_split_at_a_region_boundary():
    """snapshot.copy_programs: LD / ST pairs that each stay in one region, REG set; merged()
    does not join regions across a boundary."""
    cfg = Config(S=1, DAW=34, DRAM_BYTES=1 << 34)
    n = 3 * 16384 * 4
    progs = SN.copy_programs([(2 * G - 2 * 16384 * 4 + 64, 0x10000, n)], cfg)
    body = [i for p in progs for i in p[0] if i.op in (I.LD, I.ST)]
    lds = [i for i in body if i.op == I.LD]
    assert [i.region() for i in lds] == [1, 1, 2, 2] and sum(i.w[2] for i in lds) == n // 4
    assert lds[1].w[2] == (G - lds[1].w[0]) // 4             # (up to the boundary)
    assert all(i.region() == 0 for i in body if i.op == I.ST)
    assert SN.merged([(G - 64, 64), (G, 64)]) == [(G - 64, 64), (G, 64)]
    assert SN.merged([(G - 128, 64), (G - 64, 64)]) == [(G - 128, 128)]


# ------------------------------------------------------------------------------ layouts (Regions)
def test_regions_place_runs_whole_and_split_what_fits_no_region():
    """Default layout: each run whole in the first high region with room, else region 0; one that
    alone exceeds a region is split at units, filling the region with the most room first."""
    rg = RG.Regions(0x10000, G // 2, 1 << 34)
    sz = {"a": 1 << 30, "b": 3 << 29}
    runs, bases = rg.place([(0, ("a",), 3), (3, ("b",), 2), (5, ("a", "a"), 1)], sz.get)
    assert runs == [(0, ("a",), 3), (3, ("b",), 2), (5, ("a", "a"), 1)]
    assert bases == [G, 2 * G, 3 * G]
    for (first, unit, reps), b in zip(runs, bases):
        n = reps * sum(sz[k] for k in unit)
        assert region_of(b, n) == b // G
    rg = RG.Regions(0x10000, G // 2, 1 << 34)
    runs, bases = rg.place([(0, ("a",), 10)], sz.get)          # 10 GiB: fits no region whole
    assert sum(r for _, _, r in runs) == 10 and len(runs) == 3
    assert [b // G for b in bases] == [1, 2, 3]
    with pytest.raises(MemoryError, match="fit no DRAM region"):
        RG.Regions(0x10000, G // 2, 1 << 33).place([(0, ("a",), 9)], sz.get)
    rg = RG.Regions(0x10000, 0x10000, 1 << 34, high=2 * G - (3 << 30), whole=False)
    runs, bases = rg.place([(0, ("a",), 5)], sz.get)          # in address order, split at 8 GiB
    assert runs == [(0, ("a",), 3), (3, ("a",), 2)] and bases == [2 * G - (3 << 30), 2 * G]


def test_an_image_that_fits_region_0_is_the_dram32_layout_and_a_big_one_goes_high(models):
    """qwen35.Image at DAW 33..36: a tiny model is laid out as at DAW 32; Qwen3.5-4B's dims with
    an fp32 embedding table (layout only, no weights) put their layer run whole in region 1
    (one loop: the resident decode has DAW 32's instruction counts), the rest in region 0 with
    the program area."""
    from opentpu.llm.qwen3 import ATTN_BLOCK
    from opentpu.llm.qwen35 import Spec
    spec = Spec(2560, ("linear", "linear", "linear", "attn") * 8, 16, 4, 256, 64, 32, 128, 128,
                9216, 248320, lin_kheads=16, embed="f32")
    counts = {}
    for daw in (32, 33, 36):
        cfg = card(daw, 1 << 33)
        img = spec.image(cfg, 4096, wformat="fp4", head_format="int8", lookup=True)
        assert img.grouped and len(img.plan) == 1
        counts[daw] = [len(img.compile_decode(b, (b - 1) * ATTN_BLOCK + 3)[0][0])
                       for b in (1, 16)]
        if daw > 32:
            assert img.loc[0][0] == G and img.nbytes > G and img.prog_at < G
            assert img.spans[0][0] == 0 and img.spans[0][1] <= img.prog_at
            assert max(img.head[1], img.lookup["zmask"], img.lookup["embed"]) < img.prog_at
        else:
            assert img.prog_at is None
    assert counts[33] == counts[32] == counts[36]


def test_what_is_not_placed_over_regions_stays_in_region_0():
    """An image of a family not laid out over regions (LFM2), or a Qwen3.5 with MTP, must fit
    region 0 with the program area at DAW > 32 (MemoryError), and takes no relocation switch."""
    from opentpu.llm.lfm2 import Spec as SL
    from opentpu.llm.qwen35 import Spec as S35
    big = SL(4096, ("conv", "attn") * 10, 32, 8, 128, 16384, 200000)      # ~5.4 GiB
    with pytest.raises(MemoryError, match="region 0"):
        big.image(card(33, 1 << 33), 4096, wformat="int8")
    k35 = ("linear", "linear", "attn", "linear", "linear", "attn")
    mtp = S35(256, k35, 8, 2, 256, 64, 8, 128, 128, 512, 1000, mtp=True)
    with pytest.raises(ValueError, match="MTP layer"):
        mtp.image(card(33, 1 << 33), 384, 1, 2, high_base=G)
    with pytest.raises(ValueError, match="DAW > 32"):
        S35(256, k35, 8, 2, 256, 64, 8, 128, 128, 512, 1000).image(card(32, 1 << 26), 384,
                                                                    high_base=G)


# ------------------------------------------------------------------------------ models relocated
def card(daw: int, dram: int) -> Config:
    """The FK33-like card's configuration (one slice, the DSTEP and the stream engine)."""
    return Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, TMEM_WORDS=1 << 16, IMEM_WORDS=1 << 15,
                  LANES=8, PAIR=True, DSTEP=True, STREAM=True, DRAM_BYTES=dram, DAW=daw)


CAP = 384
LOW = card(32, 1 << 26)


@pytest.fixture(scope="module")
def models():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from test_autodecode import _tiny
    got = {}

    def get(name, tied=True):
        if name not in got:
            got[name] = _tiny(name)
        W, spec = got[name]
        if not tied:
            W = dict(W, **{"lm_head.weight": np.array(W["model.embed_tokens.weight"],
                                                      np.float32) * np.float32(1.25)})
            spec = dataclasses.replace(spec, tied=False)
        return W, spec
    return get


def _play(spec, W, cfg, toks, backend="isa", **kw):
    """A resident engine's prompt (prompt runs), a step and the decode loop on the device: the
    logits and the tokens."""
    from opentpu.llm.qwen3 import Engine
    e = Engine(spec, W, cap=CAP, cfg=cfg, resident=True, prompt_runs=True, backend=backend, **kw)
    la = e.prefill(toks[:24])
    t0 = int(np.argmax(la))
    steps = [e.step(t0)]
    gen = e.generate_card(t0, 6, stop_ids=[])
    return e, [la] + steps, gen


@pytest.fixture(scope="module")
def reference(models):
    out = {}

    def get(name, tied=True):
        if (name, tied) not in out:
            W, spec = models(name, tied)
            toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 40)]
            e, logits, gen = _play(spec, W, LOW, toks)
            assert e.image.prog_at is None
            out[(name, tied)] = (spec, W, toks, logits, gen)
        return out[(name, tied)]
    return get


def _same(want, got):
    assert len(want) == len(got)
    for a, b in zip(want, got):
        assert np.array_equal(bits(a), bits(b))


def _runs(img):
    return getattr(img, "plan", None) or img.runs


def _run_regions(img):
    """The region of each run (its blocks one after the other from the run's base)."""
    out = []
    for first, _, reps in _runs(img):
        base, us = img.loc[first][:2]
        out.append(region_of(base, reps * us))
    return out


@pytest.mark.parametrize("daw,high", [(33, G), (34, 3 * G), (36, 9 * G + (1 << 20))])
@pytest.mark.parametrize("name", ["qwen3", "qwen35"])
def test_the_layers_above_4gib_give_the_low_placements_logits_and_tokens(reference, name, daw,
                                                                         high):
    """The relocation test: every layer block from `high` on (weights, KV caches, DeltaNet
    states), the same tiny random model at DAW 33, 34 and 36: the prompt's logits, a step's and
    the decode loop's tokens of the DAW 32 placement, bit for bit."""
    spec, W, toks, logits, gen = reference(name)
    e, got, g = _play(spec, W, card(daw, 1 << daw), toks, high_base=high)
    img = e.image
    assert img.prog_at < G < img.nbytes and set(_run_regions(img)) == {high // G}
    assert isinstance(e.backend.machine.slices[0].dram, np.ndarray)
    _same(logits, got)
    assert g == gen


@pytest.mark.parametrize("name", ["qwen3", "qwen35"])
def test_runs_split_at_a_region_boundary(reference, models, name):
    """low_limit: the first unit in region 0, the rest from 4 GiB (a run split in two loops);
    and a high_base one unit short of 8 GiB's boundary at DAW 34 (a run split across regions 1
    and 2): the same logits and tokens."""
    spec, W, toks, logits, gen = reference(name)
    probe = spec.image(card(33, 1 << 33), CAP, 1, 8, lookup=True)
    first, unit, _ = _runs(probe)[0]
    us = probe.loc[first][1]
    e, got, g = _play(spec, W, card(33, 1 << 33), toks, low_limit=probe.layer0 + us + 4096)
    assert set(_run_regions(e.image)) == {0, 1} and len(_runs(e.image)) > len(_runs(probe))
    _same(logits, got)
    assert g == gen
    e, got, g = _play(spec, W, card(34, 1 << 34), toks, high_base=2 * G - us)
    assert set(_run_regions(e.image)) == {1, 2}
    _same(logits, got)
    assert g == gen


@pytest.mark.parametrize("name", ["qwen3", "qwen35"])
def test_an_untied_lm_head_above_4gib(reference, name):
    """An untied model's head rows and scale rows (one region) with the layers: the same logits
    and tokens (a tied head is the embedding table's rows, gathered by a device value: region
    0)."""
    spec, W, toks, logits, gen = reference(name, tied=False)
    e, got, g = _play(spec, W, card(34, 1 << 34), toks, high_base=2 * G)
    assert region_of(e.image.head[0]) == region_of(e.image.head[1]) == 2
    _same(logits, got)
    assert g == gen


def test_snapshots_of_deltanet_states_above_4gib(models):
    """Engine.snapshot / rollback: the DeltaNet states and windows in a high region are copied
    to the snapshot area in region 0 (past the image's region-0 areas) and back; the steps after
    a rollback are those before it, and the high placement's the low one's."""
    from opentpu.llm.qwen3 import Engine
    W, spec = models("qwen35")
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 40)]

    def play(cfg, **kw):
        e = Engine(spec, W, cap=CAP, cfg=cfg, resident=True, snapshots=2, **kw)
        assert e._snaps is not None and e._snaps.base is not None
        e.prefill(toks[:16])
        e.snapshot()
        out = [e.step(toks[16 + i]) for i in range(4)]
        e.rollback(16)
        return e, out + [e.step(toks[16 + i]) for i in range(4)]
    _, want = play(LOW)
    e, got = play(card(33, 1 << 33), high_base=G)
    assert e._snaps.base == e.image.prog_at < G
    assert all(a >= G for a, _ in e.image.state_regions(0))
    _same(want, got)
    _same(got[:4], got[4:])


# ------------------------------------------------------------------------------ the host
def regs_of(cfg: Config, channels: int = 2) -> dict:
    return dict(S=cfg.S, TMEM_WORDS=cfg.TMEM_WORDS, IMEM_WORDS=cfg.IMEM_WORDS,
                ACT_BLOCKS=cfg.ACT_BLOCKS, DRAM_BYTES=cfg.DRAM_BYTES, channels=channels,
                channel_bytes=cfg.DRAM_BYTES // channels, MXU_IMPL=2, TMEM_IMPL=0, VPU_CL=2,
                ULANES=8, SEQ_WC=0, platform=6,
                hash=T.arch_hash(dataclasses.replace(cfg, DUAL=channels == 4)))


def fake(cfg: Config, channels: int = 2, **kw) -> FakeTransport:
    """A fake card with one slice of `cfg`: its CFG registers, its DRAM as `channels` channels
    (np.zeros: lazy)."""
    return FakeTransport(ch_bytes=cfg.DRAM_BYTES // channels, devname=None, regmap=5, D=cfg.D,
                         MCOLS=cfg.MCOLS, LANES=cfg.LANES, act_rows=cfg.MCOLS, pair=cfg.PAIR,
                         dstep=cfg.DSTEP, stream=cfg.STREAM, cfg=regs_of(cfg, channels), **kw)


class WideIsaCard(FakeTransport):
    """A fake card that computes over all of a wide slice's DRAM: RUN runs the loaded program on
    the ISA simulator over the slice's logical DRAM (np.zeros, lazy; the simulator adopts it).
    The Board protocol's channel writes and reads are translated to it (channel c's beats at
    chunk m are logical beat 2 m + c, or c ^ parity(m) with CHASH)."""

    def __init__(self, cfg: Config, chash: bool = False, **kw):
        super().__init__(ch_bytes=1, devname=None, regmap=5, D=cfg.D, MCOLS=cfg.MCOLS,
                         LANES=cfg.LANES, act_rows=cfg.MCOLS, pair=cfg.PAIR, dstep=cfg.DSTEP,
                         stream=cfg.STREAM, args=True, gen=True, chash=chash, cfg=regs_of(cfg),
                         **kw)
        self.cfg = cfg
        self.dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
        self.beats = self.dram.reshape(-1, 64)
        self.written = 0

    def _logical(self, ch: int, off: int, n: int):
        b0, b1 = off // 64, (off + n - 1) // 64
        m = np.arange(b0, b1 + 1, dtype=np.int64)
        par = swapped(b0 * 128, len(m)).astype(np.int64) if self.chash else 0
        return 2 * m + ((ch % 2) ^ par), off - b0 * 64

    def mem_write(self, ch, off, data):
        data = np.asarray(data, np.uint8).reshape(-1)
        self.written += len(data)
        if len(data):
            rows, o = self._logical(ch, off, len(data))
            blk = self.beats[rows]
            blk.reshape(-1)[o:o + len(data)] = data
            self.beats[rows] = blk

    def mem_read(self, ch, off, n, out=None):
        rows, o = self._logical(ch, off, n)
        got = self.beats[rows].reshape(-1)[o:o + n]
        if out is None:
            return got.copy()
        out[:] = got
        return out

    def reg_write(self, off, val):
        rising = off == R.R_CTRL and val & R.CTRL_RUN and not self.regs[R.R_CTRL] & R.CTRL_RUN
        super().reg_write(off, val)
        if rising:
            a, n = self.regs[R.R_PROG_ADDR], self.regs[R.R_PROG_N]
            words = self.dram[a:a + 32 * n].view(np.uint32).reshape(n, 8)
            args = [self.regs.get(R.R_ARG0 + 4 * k, 0) for k in range(8)]
            Machine(self.cfg, [[I.Instr.decode(w) for w in words]], [self.dram], args=args,
                    adopt=True).run(max_steps=1 << 40)


@pytest.mark.parametrize("daw", [32] + DAWS)
def test_the_device_config_follows_the_dram_width(daw):
    """DAW from CFG_DRAM's DRAM_LOG2 (32 at the least); the CFG_HASH (DAW in it above 32) is
    checked against the host's hash of what the registers report."""
    cfg = card(daw, 1 << max(daw, 32))
    info = Board(fake(cfg)).info()
    assert device_daw(info["cfg"]) == daw
    got = device_config(info)
    assert got == cfg and info["cfg"]["hash"] == T.arch_hash(cfg)
    check_hash(info)
    if daw > 32:
        assert T.arch_hash(cfg) != T.arch_hash(dataclasses.replace(cfg, DAW=32))
        bad = dict(regs_of(cfg), hash=T.arch_hash(dataclasses.replace(cfg, DAW=32)))
        t = FakeTransport(ch_bytes=1, devname=None, regmap=5, D=128, MCOLS=4, LANES=8,
                          act_rows=4, pair=True, dstep=True, stream=True, cfg=bad)
        with pytest.raises(ConfigMismatch, match="CFG_HASH"):
            device_config(Board(t).info())


def test_programs_of_a_wider_dram_do_not_run_on_a_narrower_device():
    """check_fits: DAW 32 programs run on a wide device (REG 0); a DAW 34 configuration's
    regions do not on a DAW 33 device, nor on a DAW 32 one."""
    w33 = Board(fake(card(33, 1 << 33))).info()
    check_fits(device_config(w33), w33)
    check_fits(card(32, 1 << 32), w33)
    with pytest.raises(ConfigMismatch, match="DAW 34"):
        check_fits(card(34, 1 << 33), w33)
    n32 = Board(fake(card(32, 1 << 32))).info()
    with pytest.raises(ConfigMismatch, match="DAW 33"):
        check_fits(card(33, 1 << 32), n32)


@pytest.mark.parametrize("daw,channels,chash", [(33, 2, False), (33, 2, True), (36, 2, True),
                                                (34, 4, False), (34, 4, True)])
def test_logical_addresses_past_4gib_through_the_channels(daw, channels, chash):
    """Board.write / read at logical addresses past 4 GiB (across each boundary, unaligned, to
    the top): they read back; the DMA's channel map follows CFG_DRAM's channel bytes."""
    cfg = card(daw, 1 << daw)
    t = fake(cfg, channels, chash=chash)
    b = Board(t)
    info = b.info()
    cb = cfg.DRAM_BYTES // channels
    assert channel_map(info) == tuple(j * cb for j in range(channels))
    top = cfg.DRAM_BYTES
    rng = np.random.default_rng(7)
    regions = [(G - 1000, 3000), (G + 12345, 777), ((1 << 30) + G + 64, 4096), (top - 9000, 300),
               (top - 128, 128), (0x1234, 99), (top - G // 2 - 77, 5000)]
    data = [rng.integers(0, 256, n).astype(np.uint8) for _, n in regions]
    for (a, _), d in zip(regions, data):
        b.write(a, d)
    for (a, n), d in zip(regions, data):
        assert np.array_equal(b.read(a, n), d), hex(a)


def test_a_program_is_loaded_below_4gib_only():
    """PROG_ADDR is 32 bits: a program at or across 4 GiB is refused, the registers kept."""
    t = fake(card(33, 1 << 33))
    b = Board(t)
    prog = [I.halt()] * 3
    b.load(0x10000, prog)
    for at in (G, G + 0x10000, G - 32):
        with pytest.raises(ValueError, match="not below 4 GiB"):
            b.load(at, prog)
        assert t.regs[R.R_PROG_ADDR] == 0x10000 and t.regs[R.R_PROG_N] == 3
    b.load(G - 4096, prog)
    assert t.regs[R.R_PROG_ADDR] == G - 4096


@pytest.mark.parametrize("chash", [False, True])
def test_a_program_below_4gib_computes_on_every_region(chash):
    """The Board protocol over a DAW 34 slice: LD / ST between regions 3, 0 and 1 from a program
    in region 0, what was written at logical addresses."""
    cfg = card(34, 1 << 34)
    c = WideIsaCard(cfg, chash=chash)
    b = Board(c)
    rng = np.random.default_rng(3)
    n = 128
    src3, dst0, src0, dst1 = 3 * G + 0x12380, 0x23400, 0x40080, G + (3 << 20) + 0x180
    a1, a2 = (rng.integers(0, 256, 4 * n, dtype=np.uint8) for _ in range(2))
    b.write(src3, a1)
    b.write(src0, a2)

    def ld(a, t):
        return I.ld(a, t, n, reg=a // G)

    def st(a, t):
        return I.st(a, t, n, reg=a // G)
    b.load(0x100000, [ld(src3, 0), st(dst0, 0), ld(src0, n), st(dst1, n), I.halt()])
    b.run(timeout=10.0)
    assert np.array_equal(b.read(dst0, 4 * n), a1)
    assert np.array_equal(b.read(dst1, 4 * n), a2)


def test_the_board_backend_puts_the_program_area_in_region_0_and_writes_the_spans():
    """BoardBackend: an image laid out over regions (ImageArrays) has its program area at its
    prog_at, and only its spans are written (not the gap below 4 GiB); a plain list's program
    area follows it; no room in region 0: MemoryError."""
    cfg = card(33, 1 << 33)
    t = fake(cfg)
    sentinel = np.full(4096, 0xAB, np.uint8)
    Board(t).write(G - 8192, sentinel)                          # in the gap
    img = np.zeros(G + 8192, np.uint8)
    img[:4096] = 7
    img[G:G + 8192] = 9
    be = BoardBackend(cfg, RG.ImageArrays([img], 0x3000, [(0, 0x3000), (G, G + 8192)]),
                      transport=t, status=False)
    assert be.prog_at == 0x3000
    assert np.array_equal(be.read(0, G + 4096, 4096), np.full(4096, 9, np.uint8))
    assert np.array_equal(be.read(0, 0, 4096), np.full(4096, 7, np.uint8))
    assert np.array_equal(be.read(0, G - 8192, 4096), sentinel)
    d = dram_layout(cfg, be.image_bytes, be.prog_at)
    assert d["image"] == be.image_bytes - (G - 0x3000)
    area = RG.prog_area(cfg)
    with pytest.raises(MemoryError, match="program area"):
        BoardBackend(cfg, RG.ImageArrays([np.zeros(4096, np.uint8)], G - area + 4096),
                     transport=fake(cfg), status=False)
    be2 = BoardBackend(cfg, [np.zeros(5000, np.uint8)], transport=fake(cfg), status=False)
    assert be2.prog_at == 8192


@pytest.mark.parametrize("chash", [False, True])
def test_the_board_protocol_over_a_wide_slice_gives_the_low_placements_logits(reference, chash):
    """The whole stack: a tiny Qwen3.5 with its layer blocks in region 1 of a DAW 33 card whose
    brain is the ISA simulator: the image's spans written through the Board's channel map, the
    programs loaded at the image's prog_at, the resident decode, prompt runs and decode loop:
    the DAW 32 placement's logits and tokens."""
    spec, W, toks, logits, gen = reference("qwen35")
    cfg = card(33, 1 << 33)
    c = WideIsaCard(cfg, chash=chash)
    assert device_config(Board(c).info()) == cfg

    def backend(cf, images):
        return BoardBackend(cf, images, transport=c, status=False)
    e, got, g = _play(spec, W, cfg, toks, backend=backend, high_base=G)
    assert e.backend.prog_at == e.image.prog_at < G < e.image.nbytes
    assert c.written < 2 * e.image.nbytes - G               # (the gap is not written)
    _same(logits, got)
    assert g == gen


def test_batch_steps_with_the_kv_slots_above_4gib(models):
    """The batched decode steps (opentpu/llm/batch.py) over four sequences' KV slots in the layer
    blocks of region 1, the batch area in region 0 past the image's region-0 areas: each
    sequence's tokens are the DAW 32 placement's."""
    from opentpu.llm import batch as BA
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    W, spec = models("qwen3")
    cap = 512
    base = device_config(spec, cap, batch=4, rows=PREFILL_ROWS, lookup=True, S=1)
    rng = np.random.default_rng(0)
    prompts = {s: [int(x) for x in rng.integers(0, 1000, n)] for s, n in ((0, 21), (1, 9), (3, 30))}
    res = {}
    for tag, cfg, kw in (("low", base, {}),
                         ("high", dataclasses.replace(base, DRAM_BYTES=1 << 33, DAW=33),
                          {"high_base": G})):
        eng = Engine(spec, W, cap=cap, cfg=cfg, batch=4, resident=True, prompt_runs=True,
                     batch_area=True, **kw)
        assert eng.bat is not None
        if tag == "high":
            assert eng.image.nbytes > G and eng.bat["base"] == eng.image.prog_at
            assert eng.bat["end"] + RG.prog_area(cfg) <= G
            assert set(_run_regions(eng.image)) == {1}
        steps, got = BA.Steps(eng), {}
        for s, p in prompts.items():
            got[s] = [int(np.argmax(eng.prefill(p, seq=s)))]
            steps.set_sampling(s, None)
        for _ in range(5):
            rows = [(s, eng.poss[s]) for s in prompts]
            for (s, _), t in zip(rows, steps.run(rows, [got[s][-1] for s, _ in rows])):
                got[s].append(t)
        res[tag] = got
    assert res["high"] == res["low"]


def test_diag_tests_a_wide_cards_channels_and_each_4gib_boundary(tmp_path, monkeypatch):
    """otpu-diag's registers and memory sections on a DAW 33 card (two 4 GiB channels): the
    CFG registers are sane, the address bits walk the 4 GiB channels, and the interleave
    patterns reach both sides of 4 GiB and the top of 8 GiB."""
    import json
    from opentpu.host import diag
    for k in ("OTPU_MCOLS", "OTPU_LANES", "OTPU_PAIR", "OTPU_DSTEP", "OTPU_ACT_ROWS"):
        monkeypatch.delenv(k, raising=False)
    out = tmp_path / "diag.json"
    t = fake(card(33, 1 << 33))
    rc = diag.main(["--only", "regs,mem", "--bw-mib", "1", "--json", str(out)],
                   open_transport=lambda dev: t)
    rep = json.loads(out.read_text())
    bad = [(r["name"], r["msg"]) for r in rep["rows"] if r["status"] == "FAIL"]
    assert rc == 0 and not bad, bad
    rows = {r["name"]: r for r in rep["rows"]}
    assert "8 regions" in rows["interleave patterns"]["msg"]
