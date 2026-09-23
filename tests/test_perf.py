"""Performance on the RTL: MLP and attention decode must run at the DRAM roofline.

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
    assert _eff(p) > 0.95, p.summary()


def test_mlp_small_batch_near_roofline(have_verilator):
    a, _ = mlp_args(np.random.default_rng(0), M=4, H=1024, Fd=2048)
    p = profile(mlp, design_config(), **a)
    assert _eff(p) > 0.92, p.summary()


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
