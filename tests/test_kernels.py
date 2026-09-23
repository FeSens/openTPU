"""Kernels on the bit-exact ISA simulator vs float64 math (int8 block quantization tolerance)."""
import numpy as np
import pytest

from opentpu import Config
from opentpu import reference as ref
from opentpu.kernels import attention_decode, attention_layer, mlp
from opentpu.kernels.layouts import head_parallel_attention_weights
from opentpu.runtime import Input, KVCache, Output, Weight, launch

from conftest import rel


def mlp_args(rng, M=4, H=128, Fd=256):
    h = rng.standard_normal((M, H)).astype(np.float32)
    gamma = (1 + 0.1 * rng.standard_normal(H)).astype(np.float32)
    wg = (rng.standard_normal((Fd, H)) / np.sqrt(H)).astype(np.float32)
    wu = (rng.standard_normal((Fd, H)) / np.sqrt(H)).astype(np.float32)
    wd = (rng.standard_normal((H, Fd)) / np.sqrt(Fd)).astype(np.float32)
    args = dict(h=Input(h), gamma=Input(gamma), w_gate=Weight(wg, 0), w_up=Weight(wu, 0),
                w_down=Weight(wd, 0), out=Output((M, H)), eps=1e-6)
    return args, ref.mlp(h, gamma, wg, wu, wd, 1e-6)


def attn_args(rng, Hq=8, Hkv=2, d=64, T=100, cap=128, block=32):
    q = rng.standard_normal((Hq, d)).astype(np.float32)
    k = rng.standard_normal((Hkv, T, d)).astype(np.float32)
    v = rng.standard_normal((Hkv, T, d)).astype(np.float32)
    args = dict(q=Input(q), kv=KVCache(k, v, cap), out=Output((Hq, d)), n_q_heads=Hq,
                n_kv_heads=Hkv, seq_len=T, block=block)
    return args, ref.attention(q, k, v, Hkv)


def layer_args(rng, S, H=128, Hq=4, Hkv=2, d=64, pos=40, cap=64):
    h = rng.standard_normal((1, H)).astype(np.float32)
    gamma = (1 + 0.1 * rng.standard_normal(H)).astype(np.float32)
    wq = (rng.standard_normal((Hq * d, H)) / np.sqrt(H)).astype(np.float32)
    wk = (rng.standard_normal((Hkv * d, H)) / np.sqrt(H)).astype(np.float32)
    wv = (rng.standard_normal((Hkv * d, H)) / np.sqrt(H)).astype(np.float32)
    wo = (rng.standard_normal((H, Hq * d)) / np.sqrt(Hq * d)).astype(np.float32)
    theta = 10000.0 ** (-np.arange(d // 2) / (d // 2))
    cos, sin = np.cos(pos * theta).astype(np.float32), np.sin(pos * theta).astype(np.float32)
    kc = rng.standard_normal((Hkv, pos, d)).astype(np.float32)
    vc = rng.standard_normal((Hkv, pos, d)).astype(np.float32)
    want = ref.attention_layer(h, gamma, wq, wk, wv, wo, cos, sin, kc, vc, Hq, Hkv, pos, 1e-6)
    pq, pk, pv, po = head_parallel_attention_weights(wq, wk, wv, wo, Hq, Hkv, d, S)
    args = dict(h=Input(h), gamma=Input(gamma), wq=Weight(pq, 0), wk=Weight(pk, 0),
                wv=Weight(pv, 0), wo=Weight(po, 0), cos=Input(cos), sin=Input(sin),
                kv=KVCache(kc, vc, cap), out=Output((1, H)), n_q_heads=Hq, n_kv_heads=Hkv,
                pos=pos, block=32, eps=1e-6)
    return args, want


@pytest.mark.parametrize("S", [1, 2, 4])
@pytest.mark.parametrize("M", [1, 4, 11])
def test_mlp(S, M):
    args, want = mlp_args(np.random.default_rng(M), M=M)
    got = launch(mlp, Config(S=S), **args).outputs["out"]
    assert rel(got, want) < 0.02


@pytest.mark.parametrize("S", [1, 2])
@pytest.mark.parametrize("T,block", [(1, 32), (31, 32), (32, 32), (100, 32), (100, 64)])
def test_attention_decode(S, T, block):
    args, want = attn_args(np.random.default_rng(T), T=T, block=block)
    got = launch(attention_decode, Config(S=S), **args).outputs["out"]
    assert rel(got, want) < 0.03


@pytest.mark.parametrize("S,Hq,Hkv", [(1, 2, 1), (1, 4, 4), (2, 4, 4), (4, 8, 4)])
def test_attention_decode_head_layouts(S, Hq, Hkv):
    args, want = attn_args(np.random.default_rng(7), Hq=Hq, Hkv=Hkv, T=70)
    got = launch(attention_decode, Config(S=S), **args).outputs["out"]
    assert rel(got, want) < 0.03


@pytest.mark.parametrize("S", [1, 2])
@pytest.mark.parametrize("pos", [0, 40, 63])
def test_attention_layer(S, pos):
    args, (want, knew, vnew) = layer_args(np.random.default_rng(pos), S, pos=pos)
    r = launch(attention_layer, Config(S=S), **args)
    assert rel(r.outputs["out"], want) < 0.01
    K, V = r.kv("kv")
    assert rel(K[:, pos], knew) < 0.02 and rel(V[:, pos], vnew) < 0.02


def test_design_size_block_128():
    """The design configuration: D = 128 (= head dim), 128-token blocks."""
    args, want = attn_args(np.random.default_rng(5), Hq=6, Hkv=1, d=128, T=300, cap=384,
                           block=128)
    got = launch(attention_decode, Config(S=1, D=128, ACT_BLOCKS=16), **args).outputs["out"]
    assert rel(got, want) < 0.03
