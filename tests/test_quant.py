"""Weight formats (opentpu/quant.py): the MX MXU's packers and scale rows, its MM on the ISA
simulator against the float64 emulation (docs/isa.md "MM", docs/mx_mxu.md), column parts and
slices, the gathers' tables, and the checkpoint imports (OCP MXFP4, NVIDIA NVFP4)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from opentpu import Config, isa as I, quant as Q
from opentpu import language as ol
from opentpu import reference as ref
from opentpu.compiler import Affine, Tensor, current
from opentpu.isasim import Machine
from opentpu.kernels import gather as GA
from opentpu.kernels import mlp
from opentpu.runtime import Input, Output, Weight, launch

from conftest import rel
from test_kernels import mlp_args

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------------- the study formats
def test_e2m1_and_e4m3_rounding():
    # ties go to the even code (the even mantissa bit), values past the top saturate
    got = Q.round_to_grid(np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, 100.0]), Q.E2M1)
    assert got.tolist() == [0, 2, 2, 4, 4, 6, 6, 7, 7]
    assert Q.e4m3(np.array([1.0625, 1.1875, 500.0, 2.0 ** -9])).tolist() == \
        [1.0, 1.25, 448.0, 2.0 ** -9]
    assert len(Q.E4M3) == 127 and Q.E4M3[-1] == 448.0


@pytest.mark.parametrize("fmt,g,bits", [("int8", 128, 8.25), ("int4", 32, 4.5), ("mxfp4", 32, 4.25),
                                        ("nvfp4", 16, 4.5), ("int4k", 32, 4.25)])
def test_formats_bits_and_error(fmt, g, bits):
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((64, 256)) * 0.02).astype(np.float32)
    q = Q.quantize(W, fmt)
    assert q.bits == bits and q.g == g
    assert Q.rel_err(W, q.deq) < (1e-4 if fmt == "int8" else 0.02)


def test_mxfp4_elements_are_e2m1_times_power_of_two():
    W = (np.random.default_rng(1).standard_normal((8, 64))).astype(np.float32)
    q = Q.quantize(W, "mxfp4")
    s = q.scales                                                # [8, 2] powers of two
    assert np.all(np.log2(s) == np.round(np.log2(s)))
    x = np.abs(q.deq.reshape(8, 2, 32) / s[..., None])
    assert np.isin(x, Q.E2M1).all()


# ------------------------------------------------------------------------- the MXU's formats
def _spread(rng, N, K, s=3.0, g=32):
    """Rows with log-normal scales per g elements (offsets up to the window and past it)."""
    return (rng.standard_normal((N, K)) * np.exp(s * rng.standard_normal((N, K // g, 1)))
            .repeat(g, 2).reshape(N, K) * 0.02).astype(np.float32)


def _mx_formats():
    s = importlib.util.spec_from_file_location("mx_formats", ROOT / "tools/exp/mx_formats.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_quantizers_are_the_studys():
    """int8x, fp4x6 and nvfp4 (per-row) are tools/exp/mx_formats.py's w_int8x, w_fp4x(a, 6)
    and w_nvfp4(a, per_row=True), the formats docs/mx_mxu.md 3.3 measured: the same values
    (to fp32 rounding: the device's are fp32 products) and the same offsets per 16 elements
    (an all-zero group's never counts)."""
    MXF = _mx_formats()
    rng = np.random.default_rng(0)
    W = _spread(rng, 64, 512)
    W[3, 100:132] = 0
    W[5, :128] = 0
    for fmt, want in (("int8", MXF.w_int8x(W)), ("fp4", MXF.w_fp4x(W, 6)),
                      ("nvfp4", MXF.w_nvfp4(W, True))):
        rows, srows = Q.quantize_mxu(W, fmt, 128)
        deq = Q.dequantize_mxu(rows, srows, fmt, 512, 128)
        assert np.abs(deq - want.deq).max() <= 1e-6 * np.abs(want.deq).max(), fmt
        live = np.abs(W).reshape(64, 32, 16).max(-1) > 0
        dw = Q.group_offsets(srows, fmt, 512, 128)
        assert np.array_equal(dw[live], np.asarray(want.dw)[live]), fmt
        assert Q.mxu_wf(fmt) == I.WFORMATS[fmt] and I.MX_W[I.WFORMATS[fmt]] == want.W


@pytest.mark.parametrize("fmt", Q.MXU_FORMATS)
def test_layout_and_error(fmt):
    """Element rows and scale rows (docs/isa.md "MM"): int8 a byte an element, Sw then a dw
    byte per block (4-byte words); int4 / fp4 a nibble an element, Sw, a reserved word, then a
    word per block (dw, four 6-bit multipliers, the m search's); nvfp4 Sw, three reserved
    words, then eight E4M3 bytes per block (every one's sign clear). The relative error is the
    format's."""
    rng = np.random.default_rng(2)
    W = _spread(rng, 16, 640, 1.0)
    W[3, 128:256] = 0                                           # an all-zero block
    rows, srows = Q.quantize_mxu(W, fmt, 128)
    K, KB = 640, 5
    assert rows.shape == (16, Q.row_bytes(K, fmt, 128)) and rows.dtype == np.uint8
    assert srows.shape == (16, Q.scale_bytes(K, fmt, 128))
    assert Q.scale_bytes(K, fmt, 128) == {"int8": 4 + 4 * 2, "int4": 8 + 8 * 3,
                                          "fp4": 8 + 8 * 3, "nvfp4": 16 + 16 * 3}[fmt]
    Sw, dw, m = Q.unpack_scales(srows, fmt, KB)
    assert np.array_equal(Q.pack_scales(fmt, Sw, dw=dw[..., 0].astype(np.uint8),
                                        m=m.astype(np.uint8),
                                        e4=srows[:, 16:16 + 8 * KB].reshape(16, KB, 8)
                                        if fmt == "nvfp4" else None), srows)
    if fmt in ("int4", "fp4"):
        assert m.min() >= 1 and m.max() <= Q.MMAX
    if fmt == "nvfp4":
        assert not np.any(srows[:, 16:] & 0x80)
    deq = Q.dequantize_mxu(rows, srows, fmt, K, 128)
    assert np.all(deq[3, 128:256] == 0)
    assert Q.rel_err(W, deq) < {"int8": 0.006, "int4": 0.12, "fp4": 0.1, "nvfp4": 0.09}[fmt]
    if fmt != "int8":
        assert np.array_equal(Q.pack4(Q.unpack4(rows[:, :K // 2])), rows[:, :K // 2])


@pytest.mark.parametrize("fmt", Q.MXU_FORMATS)
def test_quantize_mxu_in_row_pieces_is_the_same(fmt, monkeypatch):
    """quantize_mxu quantizes a tall matrix QUANT_ROWS rows at a time (a big vocabulary's LM
    head): the same bytes and scales as all rows at once."""
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((1000, 512)) * np.exp(rng.standard_normal((1000, 1)))).astype(
        np.float32)
    W[3, :128] = 0
    whole = Q.quantize_mxu(W, fmt)
    monkeypatch.setattr(Q, "QUANT_ROWS", 96)
    pieces = Q.quantize_mxu(W, fmt)
    assert all(np.array_equal(a, b) and a.dtype == b.dtype for a, b in zip(whole, pieces))


@pytest.mark.parametrize("fmt", Q.MXU_FORMATS)
def test_column_parts_are_the_whole_rows(fmt):
    """A matrix stored in column parts (W_down's chunks) keeps its rows' quantization: each
    part's scale rows hold the rows' Sw and its blocks' offsets (and multipliers), so the parts
    stand for the whole rows' columns exactly, as a column slice's k0 reads them."""
    rng = np.random.default_rng(3)
    W = _spread(rng, 8, 1024)
    rows, srows = Q.quantize_mxu(W, fmt, 128)
    whole = Q.dequantize_mxu(rows, srows, fmt, 1024, 128)
    for C in (256, 512):
        parts = Q.column_parts(rows, srows, fmt, 1024, C, 128)
        assert len(parts) == 1024 // C
        for j, (pr, ps) in enumerate(parts):
            assert pr.shape[1] == Q.row_bytes(C, fmt, 128)
            assert ps.shape[1] == Q.scale_bytes(C, fmt, 128)
            assert np.array_equal(Q.dequantize_mxu(pr, ps, fmt, C, 128),
                                  whole[:, j * C:(j + 1) * C])
            assert np.array_equal(Q.part_scales(srows, fmt, j * C // 128, C // 128), ps)


@pytest.mark.parametrize("fmt", Q.MXU_FORMATS)
def test_gathered_parts_are_the_rows_in_gather_order(fmt):
    """W_down on S slices in the all-gather's order (quant.gathered_parts): part c holds chunk c
    of every slice's F part, the whole rows' values (each row's Sw, the blocks' offsets and
    multipliers moved with their columns); S = 1 the column parts of C."""
    rng = np.random.default_rng(4)
    K = 2048
    W = _spread(rng, 8, K)
    rows, srows = Q.quantize_mxu(W, fmt, 128)
    whole = Q.dequantize_mxu(rows, srows, fmt, K, 128)
    for S, C in ((1, 256), (2, 256), (4, 256), (2, 512)):
        order = Q.gather_order(K, C, S)
        assert sorted(order) == list(range(K))
        assert list(order[:2 * C:C]) == [0, K // S if S > 1 else C]
        parts = Q.gathered_parts(rows, srows, fmt, K, C, S, 128)
        assert len(parts) == K // (S * C)
        for j, (pr, ps) in enumerate(parts):
            assert pr.shape[1] == Q.row_bytes(S * C, fmt, 128)
            assert ps.shape[1] == Q.scale_bytes(S * C, fmt, 128)
            assert np.array_equal(Q.dequantize_mxu(pr, ps, fmt, S * C, 128),
                                  whole[:, order[j * S * C:(j + 1) * S * C]])
        if S == 1:
            assert all(np.array_equal(a, b) for p, q in
                       zip(parts, Q.column_parts(rows, srows, fmt, K, C, 128))
                       for a, b in zip(p, q))


def test_fake_act_gathered_is_one_range_a_chunk():
    """A down projection's input on S slices (quant.fake_act_gathered) is quantized as one QACT
    over chunk c of every slice's F part: fake_act of those columns, values and offsets in
    place; S = 1 fake_act's ranges of C."""
    rng = np.random.default_rng(5)
    x = _spread(rng, 3, 2048).astype(np.float64)
    a = Q.fake_act_gathered(x, 128, 256, 1)
    b = Q.fake_act(x, 128, 256)
    assert np.array_equal(a, b) and np.array_equal(a.da, b.da)
    for S in (2, 4):
        order = Q.gather_order(2048, 256, S)
        a = Q.fake_act_gathered(x, 128, 256, S)
        b = Q.fake_act(x[:, order], 128, S * 256)
        assert np.array_equal(np.asarray(a)[:, order], b)
        assert np.array_equal(a.da[:, order[::16] // 16], b.da)
        assert not np.array_equal(a, Q.fake_act(x, 128, 256))


# ------------------------------------------------------------------------- MM on the ISA
@ol.jit
def k_dot(x, w, out):
    ol.store(out, ol.dot(ol.load(x), w))


@ol.jit
def k_dot_cols(x, w, out, c0, c1):
    ol.store(out, ol.dot(ol.load(x)[:, c0:c1], w[:, c0:c1]))


def _x(rng, M, K, s):
    return (rng.standard_normal((M, K)) * np.exp(s * rng.standard_normal((M, K // 8, 1)))
            .repeat(8, 2).reshape(M, K)).astype(np.float32)


@pytest.mark.parametrize("fmt", Q.MXU_FORMATS)
@pytest.mark.parametrize("D", [32, 128])
def test_mm_follows_the_emulation(fmt, D):
    """One MM per format (activations and weights with offsets wide enough for the window to
    drop groups): the ISA simulator's sums are the float64 emulation's (quant.fake_act, the
    weights dequantized, mx_matmul's drops) to fp32 rounding, and the drops matter (without
    them the emulation is 100x as far off); a column slice of the weights (MM k0) the same."""
    rng = np.random.default_rng(D)
    M, N, K = 3, 24, 8 * D
    x, W = _x(rng, M, K, 6.0), _spread(rng, N, K, 6.0, 8)
    cfg = Config(S=1, D=D, MCOLS=4)
    got = launch(k_dot, cfg, x=Input(x), w=Weight(W, fmt=fmt), out=Output((M, N))).outputs["out"]
    wd, xa = Q.fake_w(W, fmt, D), Q.fake_act(x, D)
    want = xa @ wd.T
    plain = np.asarray(xa).view(np.ndarray) @ np.asarray(wd).view(np.ndarray).T
    assert np.abs(got - want).max() < 1e-6 * np.abs(want).max()
    assert np.abs(got - plain).max() > 100 * np.abs(got - want).max()
    c0, c1 = 2 * D, 4 * D
    got = launch(k_dot_cols, cfg, x=Input(x), w=Weight(W, fmt=fmt), out=Output((M, N)),
                 c0=c0, c1=c1).outputs["out"]
    srows = Q.quantize_mxu(W, fmt, D)[1]
    want = Q.mx_matmul(Q.fake_act(x[:, c0:c1], D), np.asarray(wd)[:, c0:c1],
                       Q.group_offsets(Q.part_scales(srows, fmt, 2, 2), fmt, c1 - c0, D),
                       I.MX_W[I.WFORMATS[fmt]], D)
    assert np.abs(got - want).max() < 1e-6 * np.abs(want).max()


@pytest.mark.parametrize("fmt", ["int8", "fp4", "nvfp4"])
def test_mm_runs_a_pass_per_qact_range(fmt):
    """A stationary operand quantized again by two QACTs (blocks [0, 2) and [2, 4): two row
    scales, as compiler.Stationary.qranges records them) is multiplied by one MM per range, the
    second accumulating into the first's output (MM ACC); one MM over both ranges would read
    two scales of a row (the ISA simulator refuses it). The result is the emulation's with the
    two ranges quantized on their own."""
    D, M, N, K = 128, 2, 16, 512
    rng = np.random.default_rng(5)
    x, W = _x(rng, M, K, 2.0), _spread(rng, N, K, 2.0, 8)
    x[:, 256:] *= 300.0                         # the ranges' scales far apart

    @ol.jit
    def k(x, w, out):
        b = current()
        xt = ol.load(x)
        st = b.quantize(xt)                     # blocks 0..3, then 0..1 and 2..3 on their own
        (ab, mc), = st.chunks
        b.emit(I.qact(xt.base, mc, ab, 2, xt.rs, comment="first range"))
        b.emit(I.qact(xt.base + 2 * D, mc, ab + 2, 2, xt.rs, comment="second range"))
        st.add_qrange(0, 2)
        st.add_qrange(2, 4)
        ol.store(out, ol.dot(st, w))

    cfg = Config(S=1, D=D, MCOLS=4)
    r = launch(k, cfg, x=Input(x), w=Weight(W, fmt=fmt), out=Output((M, N)))
    want = Q.fake_act(x, D, [256, 256]) @ Q.fake_w(W, fmt, D).T
    assert np.abs(r.outputs["out"] - want).max() < 1e-6 * np.abs(want).max()
    mms = [p for p in r.compiled.programs[0] if p.op == I.MM]
    assert len(mms) == 2 and not mms[0].flags & I.F_ACC and mms[1].flags & I.F_ACC


@pytest.mark.parametrize("fmt", Q.FOUR)
def test_mlp_kernel_4bit(fmt):
    """The MLP kernel with 4-bit weights, against float64 math on the dequantized 4-bit weights
    (what remains is the int8 activations' error, as with int8 weights)."""
    args, _ = mlp_args(np.random.default_rng(3), M=4, H=256, Fd=512)
    deq = {}
    for k in ("w_gate", "w_up", "w_down"):
        args[k].fmt = fmt
        deq[k] = np.asarray(Q.fake_w(args[k].array, fmt, 32)).view(np.ndarray)
    want = ref.mlp(args["h"].array, args["gamma"].array, deq["w_gate"], deq["w_up"],
                   deq["w_down"], 1e-6)
    got = launch(mlp, Config(S=2), **args).outputs["out"]
    assert rel(got, want) < 0.01


# ------------------------------------------------------------------------- gathers
def _gather_run(D, M, fmt, rows, record, wg=1):
    """(gathered, the host's values) of rows [N, K] from a table (record: packed records) on the
    ISA simulator; wg > 1: at WG wg under GALL with the channel hash (the gathers in groups)."""
    cfg = Config(S=1, D=D, MCOLS=M, ACT_BLOCKS=64, TMEM_WORDS=1 << 16, DRAM_BYTES=1 << 22,
                 **({"WG": wg, "CHASH": True, "GALL": True} if wg > 1 else {}))
    N, K = rows.shape
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)
    oh = GA.onehot(D, M, "int8" if fmt == "int8" else "fp4", wg)
    OH, TAB, OUT = 0, 1 << 18, 1 << 21
    img[OH:OH + oh.nbytes] = oh.view(np.uint8).reshape(-1)
    if record:
        S = GA.record_blocks(-(-K // D), fmt)
        rec = GA.pack_records(rows, fmt, D, S, wg)
        put = GA.hash_records(rec, TAB, S, fmt, D, wg)
        img[TAB:TAB + put.size] = put.reshape(-1)
        tab = GA.record_table(TAB, N, S, fmt, D, wg)
        want, width = GA.dequant_records(GA.pack_records(rows, fmt, D, S), fmt, D, S)[:, :K], S * D
        assert np.array_equal(GA.dequant_records(rec, fmt, D, S, wg)[:, :K].view(np.uint32),
                              want.view(np.uint32))
    else:
        q, s = Q.quantize_mxu(rows, fmt, D)
        g = Q.gather_scales(s, fmt, K, D)
        d, gs = GA.pack_table(q, g, fmt, D, wg, TAB) if wg > 1 else (q, g)
        img[TAB:TAB + d.size] = np.ascontiguousarray(d).view(np.uint8).reshape(-1)
        GS = -(-(TAB + d.size) // 4096) * 4096
        img[GS:GS + gs.nbytes] = np.ascontiguousarray(gs).view(np.uint8).reshape(-1)
        tab = GA.table(TAB, GS, N, K, fmt, D, wg)
        want, width = GA.dequant_gathered(q, g, fmt, K, D), K
    out = Tensor(Affine(OUT), (N, width), (width, 1))

    @ol.jit
    def k(ohd, out):
        op = GA.operand(ohd)
        for r in range(N):
            v = GA.gather_record(op, tab, r, fmt, S) if record else \
                GA.gather_row(op, tab, r, fmt)
            ol.store(out[r:r + 1, :], v.reshape(1, width))

    prog = k.trace(cfg, 0, {"ohd": Tensor(Affine(OH), oh.shape, (oh.shape[1], 1)),
                            "out": out}).finish()
    s = Machine(cfg, [prog], [img]).run().slices[0]
    return s.dram[OUT:OUT + 4 * N * width].view(np.float32).reshape(N, width)[:, :K], want


@pytest.mark.parametrize("fmt", Q.MXU_FORMATS)
@pytest.mark.parametrize("record", [False, True], ids=["row", "record"])
@pytest.mark.parametrize("D", [32, 128])
def test_gathers_are_the_hosts_bit_for_bit(fmt, record, D):
    """kernels.gather's one-hot MMs on the ISA simulator: a table row (its element row and
    gather scales: quant.gather_scales) or a packed record, dequantized bit for bit as the
    host's dequant_gathered / dequant_records says (the embedding rows and PLE records the
    references take), with no VPU dequantization: the MM's scale arithmetic does it."""
    rng = np.random.default_rng(D)
    K = 6 * D
    rows = _spread(rng, 5, K, 3.0, 8) * 50
    rows[1, :D] = 0
    got, want = _gather_run(D, 4, fmt, rows, record)
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
    assert rel(got, rows) < {"int8": 0.02, "int4": 0.15, "fp4": 0.12, "nvfp4": 0.12}[fmt]


@pytest.mark.parametrize("fmt", Q.MXU_FORMATS)
@pytest.mark.parametrize("record", [False, True], ids=["row", "record"])
@pytest.mark.parametrize("D,KB", [(32, 6), (128, 6), (32, 18)])
def test_gathers_in_groups_are_the_natural_ones(fmt, record, D, KB):
    """Under GALL at WG 8 with the channel hash (docs/weight_groups.md) a table row or a
    record is streamed as a grouped matrix of its blocks (int8 groups of 8, 4-bit of 16: lane l
    blocks 2l and 2l + 1), every format with int8's one-hot operand: the natural gathers'
    values bit for bit (dequant_gathered / dequant_records of the natural layout), and
    dequant_records reads the grouped records the same. KB 18: several groups, a partial one."""
    from opentpu import isasim
    rng = np.random.default_rng(D + KB)
    rows = _spread(rng, 5, KB * D, 3.0, 8) * 50
    rows[1, :D] = 0
    mm = isasim.Slice._mm
    rds = []
    isasim.Slice._mm = lambda s, ins: (rds.append(ins.rd), mm(s, ins))[1]
    try:
        got, want = _gather_run(D, 8, fmt, rows, record, wg=8)
    finally:
        isasim.Slice._mm = mm
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
    assert set(rds) == {3 if fmt == "int8" else 4}        # every MM in groups (LG 3 / 4)


# ------------------------------------------------------------------------- checkpoint imports
def _e2m1(codes):
    v = Q.E2M1[codes & 7]
    return np.where(codes & 8, -v, v)


def test_import_mxfp4_exact():
    """OCP MXFP4 (E2M1 per element, E8M0 per 32) comes in exactly: as fp4 (fp4x6, m = 2^(e -
    the block's smallest e)) where each 128-block's exponents span at most 5 (all-zero
    sub-blocks have any), as nvfp4 (mantissa-0 E4M3 bytes, subnormals at the bottom) where a
    block spans more but no row more than 17 octaves; past that NotExact. The values are the
    checkpoint's, and the ISA simulator's MM with them is the float64 product."""
    rng = np.random.default_rng(7)
    N, K = 6, 512
    codes = rng.integers(0, 16, (N, K)).astype(np.uint8)
    codes[2, 64:96] &= 8                                        # a zero sub-block (signs only)
    base = rng.integers(110, 140, (N, 1))
    e = base + rng.integers(0, 6, (N, K // 32))                 # a block spans <= 5
    e[2, 2] = 0                                                 # (the zero sub-block's)
    want = _e2m1(codes) * np.exp2(np.repeat(e, 32, 1) - 127.0)
    rows, srows, fmt = Q.import_mxfp4(codes, e)
    assert fmt == "fp4"
    assert np.array_equal(Q.dequantize_mxu(rows, srows, "fp4", K), want)
    e2 = e.copy()
    e2[:, 1] = base[:, 0] + 17                                  # spans 6..17 in a block
    want2 = _e2m1(codes) * np.exp2(np.repeat(e2, 32, 1) - 127.0)
    rows2, srows2, fmt2 = Q.import_mxfp4(codes, e2)
    assert fmt2 == "nvfp4"
    assert np.array_equal(Q.dequantize_mxu(rows2, srows2, "nvfp4", K), want2)
    e3 = e.copy()
    e3[0, 5] = base[0, 0] + 23                                  # a row spans 18 or more
    with pytest.raises(Q.NotExact, match="span"):
        Q.import_mxfp4(codes, e3)
    with pytest.raises(Q.NotExact, match="NaN"):
        Q.import_mxfp4(codes, np.full_like(e, 0xFF))
    # the MM: fp4's sums are the exact products of the imported values (int8 activations with
    # a 127 in every group: offsets 0, nothing dropped); nvfp4's drop what its window does
    # (17 octaves of scales against W = 13): the emulation's sums
    x = rng.integers(-127, 128, (2, K)).astype(np.float32)
    x[:, ::32] = 127
    cfg = Config(S=1, D=128, MCOLS=4)
    for r, s, f, w in ((rows, srows, fmt, want), (rows2, srows2, fmt2, want2)):
        img = np.zeros(1 << 20, np.uint8)
        img[:x.nbytes] = x.view(np.uint8).reshape(-1)
        WA, SA = 1 << 16, 1 << 17
        img[WA:WA + r.size] = r.reshape(-1)
        img[SA:SA + s.size] = s.reshape(-1)
        prog = [I.ld(0, 0, x.size), I.qact(0, 2, 0, K // 128, K),
                I.mm(WA, SA, 4096, N, K // 128, r.shape[1], N, 2, 0, s.shape[1],
                     wf=Q.mxu_wf(f)), I.halt()]
        m = Machine(cfg, [prog], [img]).run().slices[0]
        got = m.tget(4096 + np.arange(2 * N)).reshape(2, N)
        if f == "fp4":
            exact = x.astype(np.float64) @ w.T
        else:
            exact = Q.mx_matmul(Q.fake_act(x, 128), w, Q.group_offsets(s, f, K),
                                I.MX_W[Q.mxu_wf(f)])
        assert np.abs(got - exact).max() <= 2e-7 * np.abs(exact).max()


def test_import_nvfp4_exact():
    """NVFP4 (E2M1, an E4M3 scale per 16, an fp32 per tensor) comes in exactly, each row's
    scales re-based so that its largest has dw 0: subnormal scales renormalize, a largest of
    mantissa 7 is the MXU's 0x7F (15 2^0; NVFP4's NaN code is not a scale here), and a
    subnormal 17 octaves below a row's 448 is an MXU subnormal again (E4M3's own range fits:
    every valid checkpoint comes in). NotExact: a negative or NaN scale."""
    rng = np.random.default_rng(8)
    N, K = 5, 256
    codes = rng.integers(0, 16, (N, K)).astype(np.uint8)
    s = rng.integers(1, 127, (N, K // 16)).astype(np.uint8)    # positive, not NaN
    s[0, :] = rng.integers(1, 8, K // 16)                       # a row of subnormals
    s[1, 0] = 0x77                                              # 480/2: mantissa 7
    s[2, 3] = 0                                                 # a zero scale
    ts = np.float32(0.0123)
    want = _e2m1(codes) * np.repeat(Q.E4M3[s], 16, 1) * np.float64(ts)
    rows, srows = Q.import_nvfp4(codes, s, ts)
    got = Q.dequantize_mxu(rows, srows, "nvfp4", K)
    assert np.array_equal(got, want)
    assert not np.any(srows[:, 16:] & 0x80)
    for bad, what in ((0x80 | 0x30, "negative"), (0x7F, "NaN")):
        b = s.copy()
        b[3, 1] = bad
        with pytest.raises(Q.NotExact, match=what):
            Q.import_nvfp4(codes, b, ts)
    far = s.copy()
    far[4, :] = 0x60                                            # 32
    far[4, 0] = 0x7E                                            # 448 = 1.75 2^8 at the top
    far[4, 1] = 0x01                                            # 2^-9: 17 octaves down
    far[4, 2] = 0x03                                            # 3 2^-9
    rows, srows = Q.import_nvfp4(codes, far, ts)
    assert np.array_equal(Q.dequantize_mxu(rows, srows, "nvfp4", K),
                          _e2m1(codes) * np.repeat(Q.E4M3[far], 16, 1) * np.float64(ts))
    assert srows[4, 17] == 0x01 and srows[4, 18] == 0x03       # MXU subnormals
