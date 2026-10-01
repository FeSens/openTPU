"""Weight formats (opentpu/quant.py) and the 4-bit MM of the ISA simulator (docs/isa.md)."""
import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I, quant as Q
from opentpu import reference as ref
from opentpu.isasim import Machine
from opentpu.kernels import mlp
from opentpu.runtime import launch

from conftest import rel
from test_kernels import mlp_args


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


@pytest.mark.parametrize("fmt", ["int4", "fp4"])
def test_pack_roundtrip(fmt):
    rng = np.random.default_rng(2)
    W = (rng.standard_normal((16, 512)) * 0.05).astype(np.float32)
    W[3, 100:132] = 0                                           # an all-zero sub-block
    data, words, deq = Q.quantize_w4(W, fmt)
    assert data.shape == (16, 256) and words.shape == (16, 4)
    assert np.array_equal(Q.dequantize_w4(data, words, fmt), deq)
    m = (words[..., None] >> (16 + 4 * np.arange(4))) & 15
    assert m.min() >= 1 and m.max() <= 15
    assert Q.rel_err(W, deq) < 0.012
    codes = Q.unpack4(data)
    assert np.array_equal(Q.pack4(codes), data)


def _mm4(fmt, unit=False, acc=False, KB=3, N=5, M=3, D=32):
    """One 4-bit MM on the ISA simulator; returns (result [M, N], float64 expectation)."""
    cfg = Config(S=1, D=D)
    rng = np.random.default_rng(KB + N)
    x = rng.standard_normal((M, KB * D)).astype(np.float32)
    W = (rng.standard_normal((N, KB * D)) * 0.1).astype(np.float32)
    rows, words = Q.quantize_mxu(W, fmt, D)
    rs = rows.shape[1]
    dram = np.zeros(1 << 16, np.uint8)
    dram[:x.nbytes] = x.view(np.uint8).reshape(-1)
    WA, SA, OUT = 8192, 16384, 1024
    dram[WA:WA + rows.size] = rows.reshape(-1)
    dram[SA:SA + words.nbytes] = words.view(np.uint8).reshape(-1)
    init = np.full((M, N), 0.5, np.float32)
    prog = [I.ld(0, 0, x.size), I.qact(0, M, 0, KB, KB * D),
            I.vop(I.V_FILL, OUT, 0, 0, M, N, N, 0, 0, I.B_SCALAR, 0.5),
            I.mm(WA, SA, OUT, N, KB, rs, N, M, 0, 4 * KB, unit=unit, acc=acc,
                 wf=Q.mxu_wf(fmt)),
            I.halt()]
    s = Machine(cfg, [prog], [dram]).run().slices[0]
    got = s.tget(OUT + np.arange(M * N)).reshape(M, N)
    q, sx = F.quantize(x.reshape(M, KB, D), axis=2)
    xa = (q.astype(np.float64) * sx[..., None]).reshape(M, KB * D)
    if unit:
        codes = Q.DEC[fmt][Q.unpack4(rows)][:, :KB * D].astype(np.float64)
        want = xa @ codes.T
    else:
        want = xa @ Q.dequantize_w4(rows[:, :KB * D // 2], words, fmt, D).T
    return got, want + (init if acc else 0)


@pytest.mark.parametrize("fmt", ["int4", "fp4"])
@pytest.mark.parametrize("KB", [1, 2, 3])
def test_mm_4bit_matches_dequantized_math(fmt, KB):
    got, want = _mm4(fmt, KB=KB)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("fmt", ["int4", "fp4"])
def test_mm_4bit_unit_and_acc(fmt):
    got, want = _mm4(fmt, unit=True, acc=True)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("fmt", ["int4", "fp4"])
def test_mlp_kernel_4bit(fmt):
    """The MLP kernel with 4-bit weights, against float64 math on the dequantized 4-bit weights
    (what remains is the int8 activations' error, as with int8 weights)."""
    args, _ = mlp_args(np.random.default_rng(3), M=4, H=256, Fd=512)
    deq = {}
    for k in ("w_gate", "w_up", "w_down"):
        args[k].fmt = fmt
        deq[k] = Q.quantize_w4(args[k].array, fmt, 32)[2]
    want = ref.mlp(args["h"].array, args["gamma"].array, deq["w_gate"], deq["w_up"],
                   deq["w_down"], 1e-6)
    got = launch(mlp, Config(S=2), **args).outputs["out"]
    assert rel(got, want) < 0.01


@pytest.mark.parametrize("fmt", ["int8", "int4", "fp4"])
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
