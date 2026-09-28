"""Performance on the RTL: lower bounds on how close MLP and attention decode get to the DRAM roofline.

The roofline of a program is the number of transfers it needs on the slice's DRAM burst port
(one D-byte chunk per cycle; the MXU consumes chunks at the same rate) -- see profile.py. These
tests pin the achieved fraction so that an RTL or compiler change cannot silently lose it.

The simulated DRAM delivers one D-byte chunk per cycle, the MXU's peak rate; the FPGA's DDR3
delivers well under half of that, so on the board the (deeply pipelined) vector and quantizer
work has correspondingly more room behind the weight stream.
"""
import numpy as np
import pytest

from opentpu.isasim import design_config
from opentpu.kernels import attention_decode, attention_layer, mlp
from opentpu.profile import profile

from test_kernels import attn_args, layer_args, mlp_args


def _eff(p):
    return p.roofline()["efficiency"]


def test_mlp_decode_at_roofline(have_verilator):
    a, _ = mlp_args(np.random.default_rng(0), M=1, H=1024, Fd=2048)
    p = profile(mlp, design_config(), **a)
    # 0.94, was 0.945 (0.95 before that): the timing registers added for the board (VPU/quantizer
    # inputs and writes, MXU combine operands, then the 120 MHz image's WBUF and the MXU row
    # register before the result FIFO) add a fixed few cycles per instruction -- 94.4% on this
    # small kernel; a Qwen3 token changes by +0.02%
    assert _eff(p) > 0.94, p.summary()


def test_mlp_small_batch_near_roofline(have_verilator):
    a, _ = mlp_args(np.random.default_rng(0), M=4, H=1024, Fd=2048)
    p = profile(mlp, design_config(), **a)
    # 0.915, was 0.92: the MXU tree is two stages longer for 4-bit weights (a register after the
    # weight decode, the sub-block multipliers), +64 cycles here (92.1% -> 91.9%); a two-layer
    # Qwen3 token does not change (-20 cycles of 1.9M)
    assert _eff(p) > 0.915, p.summary()


@pytest.mark.parametrize("Hq,Hkv,T", [(16, 4, 1024), (6, 1, 2048)])
def test_attention_decode_at_roofline(have_verilator, Hq, Hkv, T):
    a, _ = attn_args(np.random.default_rng(1), Hq=Hq, Hkv=Hkv, d=128, T=T, cap=T, block=128)
    p = profile(attention_decode, design_config(), **a)
    # exp2 runs on LANES/4 composite lanes (the split VPU trades these exp2-heavy shapes --
    # 4 and 6 query rows per KV head -- for 70K LUTs; Qwen3's 2 rows per head barely notice)
    assert _eff(p) > 0.60, p.summary()


def test_attention_layer_near_roofline(have_verilator):
    cfg = design_config()
    a, _ = layer_args(np.random.default_rng(2), cfg.S, H=1024, Hq=16, Hkv=4, d=128, pos=511,
                      cap=640)
    a["block"] = 128
    p = profile(attention_layer, cfg, **a)
    assert _eff(p) > 0.88, p.summary()


def test_dram_efficiency_definition():
    from opentpu.profile import ddr3_peak, ddr3_plusargs, dram_efficiency
    assert ddr3_peak(3200 / 3) == pytest.approx(17.0667e9, rel=1e-4)
    # a token that moves 128 B per 100 MHz cycle at DDR3-1066 uses 75% of the DDR3 peak
    assert dram_efficiency(128 * 1000, 1000, 3200 / 3, 100) == pytest.approx(0.75)
    a = dict(x[len("+axi_"):].split("=") for x in ddr3_plusargs(3200 / 3, 100)[2:])
    # the timings the model was calibrated with (DDR3-1066, 100 MHz core: ui_clk 133 MHz)
    assert {k: int(v) for k, v in a.items()} == dict(
        tpc=10000, tpu=7500, trp=3, trcd=3, tras=5, trc=7, trfc=22, trefi=1040, trmw=23, tgap=170, wgap=200, tturn=4)
    a = dict(x[len("+axi_"):].split("=") for x in ddr3_plusargs(800, 100)[2:])
    assert (int(a["tras"]), int(a["trc"]), int(a["trfc"]), int(a["trefi"])) == (4, 6, 16, 780)
