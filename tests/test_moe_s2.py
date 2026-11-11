"""MoE at S > 1: tensor-parallel experts (opentpu/llm/moe.py, "At S > 1"; moe._moe_ffn_tp), and
DeltaNet's out_proj and the dense MLP in S = 1's QACT ranges and adds (qwen35.s_exact,
qwen35.mlp_s_exact: Config.SEXACT, bcu1525-s2's default and off elsewhere; OTPU_SEXACT=0 / 1
overrides it).

Each slice holds its rows of every expert (W_gate's and W_up's F / S, W_down's H / S output
columns at the whole K) in a resident pool, and its rows of the router. The activations each
W_down reads are gathered before it, so its QACT ranges, MMs and adds are S = 1's.

The exactness contract:
- the MoE block at S = 2 equals S = 1 bit for bit on the same input;
- so a whole model whose dense part is S-exact gives S = 1's logits and tokens bit for bit at
  S = 2. Attention is S-exact (o_proj gathers the whole o, its QACT S = 1's). DeltaNet is with
  s_exact (S-exact on): S = 1's groups of 4 heads interleaved over the slices, so the
  gathers deliver them in S = 1's order, and the last group's pairs gathered side by side (an
  MM's K-sum is exact and its ACC one fp32 add, so equal ranges in equal order give equal
  bits). The dense MLP is with mlp_s_exact (S-exact on: slice s holds part s of each S = 1
  chunk, so a step's gather is S = 1's chunk in order), or when it has one S = 1 chunk (the
  tiny models' 256); Qwen3.5-MoE's MLP is the shared expert, S = 1's in moe._moe_ffn_tp;
- a model whose dense part is not S-exact (S-exact off: every target's but bcu1525-s2's)
  follows the S = 2 emulation (qwen35.emulated_logits at S, its MoE layers S = 1's).
  The 35B's reference: its S = 1 references at S = 2 when it is S-exact, else the S = 2
  emulation plus a dKL check against HF on a few prompts.
"""
import tomllib
from dataclasses import replace

import numpy as np
import pytest

from opentpu import quant as Q
from opentpu.llm import moe as MO
from opentpu.llm.moe import MoESpec

E, K, F = 8, 2, 256


@pytest.mark.parametrize("H,F_,wf,grp,kw", [
    (2048, 512, "fp4", 8, {}),                                    # the 35B's experts
    (256, 256, "int8", 1, {}),
    (2944, 2880, "fp4", 8, dict(bias=True, rows=2880, tight=True)),   # GPT-OSS's
])
def test_a_slices_part_is_the_slots_rows(H, F_, wf, grp, kw):
    """ExpertFormat S = 2: slice s's part holds the S = 1 slot's rows of slice s, byte for byte
    (rows are quantized on their own and a part starts at a whole group), W_down's column
    parts too, and the biases' entries; each part is half the slot (and the biases'
    alignment)."""
    rng = np.random.default_rng(0)
    one = MO.ExpertFormat(H, F_, 128, wf, grp, **kw)
    two = MO.ExpertFormat(H, F_, 128, wf, grp, S=2, **kw)
    wg, wu = rng.standard_normal((2, F_, H)).astype(np.float32)
    wd = rng.standard_normal((one.Hd, F_)).astype(np.float32)
    bias = tuple(rng.standard_normal(n) for n in (F_, F_, one.Hd)) if kw.get("bias") else None
    a = one.pack(wg, wu, wd, bias)
    assert one.nbytes <= 2 * two.nbytes <= one.nbytes + 3 * 1024
    mats = [(one.wg, two.wg, one.Fs, two.Fs, H), (one.wu, two.wu, one.Fs, two.Fs, H)] + \
        [(x, y, one.Hs, two.Hs, w) for x, y, w in zip(one.wd, two.wd, one.cw)]
    for s in range(2):
        b = two.pack(wg, wu, wd, bias, part=s)
        for p1, p2, n1, n2, k in mats:
            d1, s1 = Q.grp_bytes(n1, k, wf, 128, grp)
            d2, s2 = Q.grp_bytes(n2, k, wf, 128, grp)
            assert (d1, s1) == (2 * d2, 2 * s2)
            assert np.array_equal(a[p1[0] + s * d2:p1[0] + (s + 1) * d2], b[p2[0]:p2[0] + d2])
            assert np.array_equal(a[p1[1] + s * s2:p1[1] + (s + 1) * s2], b[p2[1]:p2[1] + s2])
        if bias is not None:
            for o1, o2, n in ((one.bg, two.bg, two.Fs), (one.bu, two.bu, two.Fs),
                              (one.bd, two.bd, two.Hs)):
                assert np.array_equal(a[o1 + 4 * s * n:o1 + 4 * (s + 1) * n], b[o2:o2 + 4 * n])


def test_a_part_needs_whole_groups():
    with pytest.raises(ValueError, match="whole groups"):
        MO.ExpertFormat(272, 256, 128, "fp4", 8, S=2)    # (W_down: 272 rows, 17 groups)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny(kinds, nk=4, nv=4, H=256, F=F, seed=0):
    """A tiny Qwen3.5-MoE's W and spec (_tiny_model)."""
    return _tiny_model(kinds, nk, nv, H, F, seed)[1:]


def _tiny_model(kinds, nk=4, nv=4, H=256, F=F, seed=0):
    """(float model, W, spec): a tiny Qwen3.5-MoE (E 8, k 2, hidden H, experts and shared expert
    F wide); nv 4: its DeltaNet S-exact (module docstring), nv 8 not."""
    from opentpu.llm.qwen35 import Spec
    torch.manual_seed(seed)
    hc = transformers.Qwen3_5MoeTextConfig(
        hidden_size=H, num_hidden_layers=len(kinds), num_attention_heads=8,
        num_key_value_heads=2, head_dim=256, vocab_size=1000,
        layer_types=["full_attention" if k == "attn" else "linear_attention" for k in kinds],
        linear_num_key_heads=nk, linear_num_value_heads=nv, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=True,
        max_position_embeddings=4096, rms_norm_eps=1e-6, num_experts=E,
        num_experts_per_tok=K, moe_intermediate_size=F, shared_expert_intermediate_size=F,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25})
    m = transformers.Qwen3_5MoeForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:     # zero-centered norms, but the DeltaNet output norm is plain
                p.copy_((1.0 if n.endswith("linear_attn.norm.weight") else 0.0)
                        + 0.1 * torch.randn_like(p))
            elif "experts" in n or "gate" in n:
                p.copy_(0.05 * torch.randn_like(p))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    spec = Spec(H, tuple(kinds), 8, 2, 256, 64, nv, 128, 128, F, 1000, lin_kheads=nk,
                moe=MoESpec(E=E, k=K, ffn=F, rule="softmax", shared=F))
    return m, W, spec


def _engine(spec, W, S, monkeypatch, **kw):
    """bcu1525-wg's Config at S slices (the board's two-slice target's shape: WG 8, MCOLS 8),
    the experts in a resident pool below 4 GiB (OTPU_POOL: the S = 1 image's too)."""
    from opentpu import target as T
    from opentpu.llm.qwen3 import Engine
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    cfg = T.config("bcu1525-wg", S=S, DRAM_BYTES=1 << 30)
    return Engine(spec, W, cap=256, cfg=cfg, **kw)


@pytest.mark.parametrize("kinds,wformat", [
    (("attn",), "fp4"),                             # the block on attention's (S-exact) output
    (("linear", "attn", "linear", "attn"), "fp4"),
    (("linear", "attn"), "int8"),
])
def test_tensor_parallel_experts_are_s1_bit_for_bit(kinds, wformat, monkeypatch):
    """S = 2 gives S = 1's logits bit for bit on a model whose dense part is S-exact (module
    docstring): every MoE block's input is S = 1's, so its output is too. One attention layer:
    the block on its input; the router's rows split (E + 1 = 9 rows: 8 a slice, zero rows past
    the shared gate), each slice's expert parts in its pool."""
    W, spec = _tiny(kinds)
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 5)]
    got = {}
    for S in (1, 2):
        eng = _engine(spec, W, S, monkeypatch, wformat=wformat)
        im = eng.image
        assert im.pool is not None and eng.server is None
        assert im.fmt.S == S and im.er == (E + 1 if S == 1 else 8)
        got[S] = np.array([eng.step(t) for t in toks])
    assert np.array_equal(got[1].view(np.uint32), got[2].view(np.uint32))


def test_fewer_kv_heads_than_slices_are_s1_bit_for_bit(monkeypatch):
    """S = 4 on a model of 2 KV heads (the 35B's: KVDesc.kv_heads, each KV head on two slices,
    a copy of its K and V each, its query group split between them; hidden 512, experts 512
    wide: S x D): S = 1's logits bit for bit (attention is S-exact, the MoE blocks S = 1's)."""
    W, spec = _tiny(("attn", "attn"), nv=8, H=512, F=512)     # (nv: 2S DeltaNet heads, unused)
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 5)]
    got = {}
    for S in (1, 4):
        eng = _engine(spec, W, S, monkeypatch)
        assert eng.image.nkv_loc == (2 if S == 1 else 1)
        got[S] = np.array([eng.step(t) for t in toks])
    assert np.array_equal(got[1].view(np.uint32), got[4].view(np.uint32))


def test_35b_dims_with_the_chats_formats_are_s1_bit_for_bit(monkeypatch):
    """The 35B's layers at S = 4 as the two-card chat runs them (bcu1525-x2:st16): hidden
    2048, 16 query heads over 2 KV heads, DeltaNet's 32 heads over 16 key heads (s_exact),
    top-8 of 16 experts (the 35B's 256 cut) with its shared expert, the chat's formats (attn,
    delta and mlp nvfp4 over fp4, the head and the embedding int8, untied): a prompt layer by
    layer (layer_major 2), decode steps and the generate loop give S = 1's logits and tokens
    bit for bit."""
    from opentpu import target as T
    from opentpu.llm.qwen3 import Engine
    from opentpu.llm.qwen35 import Spec, s_exact
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    H, F_, E_, K_ = 2048, 512, 16, 8
    torch.manual_seed(0)
    hc = transformers.Qwen3_5MoeTextConfig(
        hidden_size=H, num_hidden_layers=2, num_attention_heads=16, num_key_value_heads=2,
        head_dim=256, vocab_size=1000, layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=16, linear_num_value_heads=32, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=False,
        max_position_embeddings=4096, rms_norm_eps=1e-6, num_experts=E_,
        num_experts_per_tok=K_, moe_intermediate_size=F_, shared_expert_intermediate_size=F_,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25})
    m = transformers.Qwen3_5MoeForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_((1.0 if n.endswith("linear_attn.norm.weight") else 0.0)
                        + 0.1 * torch.randn_like(p))
            elif "experts" in n or "gate" in n:
                p.copy_(0.05 * torch.randn_like(p))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    spec = Spec(H, ("linear", "attn"), 16, 2, 256, 64, 32, 128, 128, F_, 1000, lin_kheads=16,
                tied=False, embed="int8", formats="attn=nvfp4,delta=nvfp4,mlp=nvfp4",
                moe=MoESpec(E=E_, k=K_, ffn=F_, rule="softmax", norm=True, shared=F_))
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 8)]
    lg, out = {}, {}
    for S in (1, 4):
        cfg = T.config(T.load("bcu1525-x2:st16"), S=S, DRAM_BYTES=1 << 28, BHW=0)
        eng = Engine(spec, W, cap=512, cfg=cfg, wformat="fp4", head_format="int8", rows=1,
                     resident=True, layer_major=2)
        assert s_exact(spec, S, cfg) == (S == 4)
        got = [list(eng.prefill_chunks(toks, chunk=None))[-1][1]]
        for _ in range(2):
            got.append(eng.step(int(np.argmax(got[-1]))))
        lg[S] = np.array(got, np.float32)
        out[S] = eng.generate_card(int(np.argmax(got[-1])), 4, stop_ids=[])
    assert np.array_equal(lg[1].view(np.uint32), lg[4].view(np.uint32))
    assert out[1] == out[4] and len(out[4]) == 4


def test_tensor_parallel_experts_generate_s1s_tokens(monkeypatch):
    """The resident decode loop and the generate loop on the device at S = 2 give S = 1's
    logits and tokens."""
    W, spec = _tiny(("linear", "attn"))
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 6)]
    lg, out = {}, {}
    for S in (1, 2):
        eng = _engine(spec, W, S, monkeypatch, rows=1, resident=True)
        assert eng.resident and eng.can_generate
        lg[S] = np.array([eng.step(t) for t in toks])
        out[S] = eng.generate_card(int(np.argmax(lg[S][-1])), 6, stop_ids=[])
    assert np.array_equal(lg[1].view(np.uint32), lg[2].view(np.uint32))
    assert out[1] == out[2] and len(out[2]) == 6


def test_s_exact_deltanet_makes_nv8_s1s_bit_for_bit(monkeypatch):
    """nv 8: DeltaNet's out_proj takes S = 1's groups (heads 0-3, 4-7) at S = 2 only with
    s_exact (slice 0 the first group, slice 1 the second, in pairs): S = 1's logits bit for
    bit. Without it (S-exact off: a slice's heads, each slice's last group in pairs) the
    logits follow the emulation at S = 2, whose MoE layers are S = 1's (the shared expert's
    down in S = 1's chunks)."""
    from opentpu.llm.qwen35 import emulated_logits
    W, spec = _tiny(("linear", "linear", "attn", "linear", "linear", "attn"), nk=8, nv=8)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 6)]
    got = {}
    monkeypatch.setenv("OTPU_SEXACT", "1")
    for S in (1, 2):
        eng = _engine(spec, W, S, monkeypatch)
        assert eng.image.sexact == (S == 2)
        got[S] = np.array([eng.step(t) for t in toks])
    assert np.array_equal(got[1].view(np.uint32), got[2].view(np.uint32))
    monkeypatch.delenv("OTPU_SEXACT")
    eng = _engine(spec, W, 2, monkeypatch)
    assert not eng.image.sexact
    dev = np.array([eng.step(t) for t in toks])
    assert not np.array_equal(dev.view(np.uint32), got[2].view(np.uint32))
    emu = emulated_logits(spec, W, toks, S=2)
    c = (dev * emu).sum(-1) / np.linalg.norm(dev, axis=-1) / np.linalg.norm(emu, axis=-1)
    assert c.min() > 0.999


def test_s_exact_is_bcu1525_s2s_default(monkeypatch):
    """Config.SEXACT: bcu1525-s2's default (its toml; bcu1525-x2's, two cards of its), off for
    every other target (bcu1525-x2's S = 2 variant included), never in the CFG_HASH (a compile
    choice); OTPU_SEXACT=0 / 1 overrides it. On bcu1525-s2's own Config the nv 8 model's S = 2 logits are S = 1's bit for
    bit with no environment, and OTPU_SEXACT=0 gives S > 1's own layout."""
    from opentpu import target as T
    from opentpu.llm.qwen3 import Engine
    from opentpu.llm.qwen35 import s_exact_on
    monkeypatch.delenv("OTPU_SEXACT", raising=False)
    s2 = T.config("bcu1525-s2")             # (its 16 GiB a slice: the bank-half window, BHW 33)
    assert s2.SEXACT and s_exact_on(s2) and not s_exact_on()
    for t in T.names():                 # every target file and its variants
        with open(T.TARGETS / f"{t}.toml", "rb") as f:
            vs = tomllib.load(f).get("variants", {})
        for v in [t] + [f"{t}:{n}" for n in vs]:
            assert T.config(v).SEXACT == (t in ("bcu1525-s2", "bcu1525-x2")), v
    assert T.arch_hash(s2) == T.arch_hash(replace(s2, SEXACT=False))
    monkeypatch.setenv("OTPU_SEXACT", "0")
    assert not s_exact_on(s2)
    monkeypatch.setenv("OTPU_SEXACT", "1")
    assert s_exact_on(T.config("bcu1525-s2"))
    monkeypatch.setenv("OTPU_SEXACT", "yes")
    with pytest.raises(ValueError):
        s_exact_on(s2)
    monkeypatch.delenv("OTPU_SEXACT")
    W, spec = _tiny(("linear", "linear", "attn", "linear", "linear", "attn"), nk=8, nv=8)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 6)]
    one = _engine(spec, W, 1, monkeypatch)
    got = np.array([one.step(t) for t in toks])
    two = Engine(spec, W, cap=256, cfg=s2)
    assert two.cfg.S == 2 and two.image.sexact
    dev = np.array([two.step(t) for t in toks])
    assert np.array_equal(got.view(np.uint32), dev.view(np.uint32))
    monkeypatch.setenv("OTPU_SEXACT", "0")
    assert not Engine(spec, W, cap=256, cfg=s2).image.sexact


def test_a_pool_past_the_slice_spills_into_the_window_in_one_block(monkeypatch):
    """The bank-half window at S = 2 (bcu1525-s2: BHW 33, 16 GiB a slice): a resident pool that
    does not fit past 8 GiB with the other weights (the 35B's 8.07 + 0.35 GiB) starts at the
    highest place that fits, in the window's upper half after the DeltaNet mixers, still one
    block (pool.base + g ebytes), every other weight from its end to the slice's top; the
    window's bytes between the mixers and the pool unwritten (wskip). Here a slice of half the
    tiny model's high part: the logits are S = 1's and the unspilled image's bit for bit, every
    write below 4 GiB; a slice too small for even the spill is refused."""
    from opentpu import target as T
    from opentpu.llm.qwen3 import Engine
    W, spec = _tiny(("linear", "attn"))
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 5)]
    one = _engine(spec, W, 1, monkeypatch)
    want = np.array([one.step(t) for t in toks])
    monkeypatch.delenv("OTPU_POOL")
    s2, G8 = T.config("bcu1525-s2"), 1 << 33
    got = {}
    for name, dram in (("whole", s2.DRAM_BYTES), ("spill", None)):
        if dram is None:            # (half the high part past 8 GiB: the pool spills)
            dram = G8 + (got["whole"][1].nbytes - G8) // 2
        eng = Engine(spec, W, cap=256, cfg=replace(s2, DRAM_BYTES=dram))
        im, pb = eng.image, eng.image.pool.experts * eng.image.pool.ebytes
        assert im.mix and im.wlo == G8 // 2 and im.low_end < G8 // 2 < im.wskip[0]
        if name == "whole":
            assert im.wskip[1] == G8 == im.layer0 < im.pool.base
        else:
            assert im.wskip == (got["whole"][1].wskip[0], im.pool.base)
            assert im.wskip[0] < im.pool.base < G8 < im.pool.base + pb <= im.layer0
            assert im.nbytes <= dram and im.nbytes - im.layer0 == \
                got["whole"][1].pool.base - G8          # (the other weights, after the pool)
        assert all(s.wlim == G8 // 2 for s in eng.backend.machine.slices)
        got[name] = (np.array([eng.step(t) for t in toks]), im)
    for name in got:
        assert np.array_equal(want.view(np.uint32), got[name][0].view(np.uint32)), name
    with pytest.raises(MemoryError, match="past the DeltaNet mixers"):
        Engine(spec, W, cap=256, cfg=replace(s2, DRAM_BYTES=got["whole"][1].wskip[0] + (1 << 20)))


def _dense(kinds, nv, ffn=256):
    """A tiny dense Qwen3.5 (MLP 256 wide: one S = 1 chunk, S-exact; 512: two)."""
    from opentpu.llm.qwen35 import Spec
    torch.manual_seed(0)
    hc = transformers.Qwen3_5TextConfig(
        hidden_size=256, num_hidden_layers=len(kinds), num_attention_heads=8,
        num_key_value_heads=2, head_dim=256, intermediate_size=ffn, vocab_size=1000,
        layer_types=["full_attention" if k == "attn" else "linear_attention" for k in kinds],
        linear_num_key_heads=nv, linear_num_value_heads=nv, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=True,
        max_position_embeddings=4096, rms_norm_eps=1e-6, initializer_range=0.2,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25})
    m = transformers.Qwen3_5ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_((1.0 if n.endswith("linear_attn.norm.weight") else 0.0)
                        + 0.1 * torch.randn_like(p))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    return W, Spec(256, tuple(kinds), 8, 2, 256, 64, nv, 128, 128, ffn, 1000, lin_kheads=nv)


@pytest.mark.parametrize("nv,dstep,pair_loop", [(8, True, None), (8, False, None),
                                                (8, True, True), (16, True, None)])
def test_s_exact_deltanet_dense_is_s1s_bit_for_bit(nv, dstep, pair_loop, monkeypatch):
    """A dense model's DeltaNet layers with s_exact: the DSTEP decode and the VPU's, the
    pairs as a loop (group-major blocks), and 16 heads (4 groups: slice 0 groups 0 and 2,
    slice 1 groups 1 and 3, the last in pairs) give S = 1's logits bit for bit at S = 2."""
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import Engine
    monkeypatch.setenv("OTPU_SEXACT", "1")
    W, spec = _dense(("linear", "attn", "linear"), nv)
    spec = replace(spec, pair_loop=pair_loop)
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 5)]
    got = {}
    for S in (1, 2):
        cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=dstep, PAIR=True, S=S)
        eng = Engine(spec, W, cap=256, cfg=cfg)
        assert eng.image.sexact == (S == 2)
        if pair_loop:
            assert eng.image.grouped
        got[S] = np.array([eng.step(t) for t in toks])
    assert np.array_equal(got[1].view(np.uint32), got[2].view(np.uint32))


@pytest.mark.parametrize("wf", ["int8", "fp4"])
def test_s_exact_dense_mlp_is_s1s_bit_for_bit(wf, monkeypatch):
    """ffn 512: S = 1's W_down in two chunks of 256 (int8 and fp4: down_chunk), S = 2's F / S
    in one chunk of 256, its all-gather [M, 512] one QACT: not S = 1's. With mlp_s_exact
    (S-exact on) slice s holds rows 128 s .. and 256 + 128 s .. of W_gate and W_up, its
    chunk 128, so each step's gather is S = 1's chunk of 256 in order: S = 1's logits bit for
    bit at S = 2 (DeltaNet's 8 heads with s_exact). Without the flag they differ."""
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import Engine
    W, spec = _dense(("linear", "attn", "linear"), 8, ffn=512)
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 5)]

    def run(S):
        cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=True, PAIR=True, S=S)
        eng = Engine(spec, W, cap=256, cfg=cfg, wformat=wf)
        lay = eng.image.layouts[eng.image.lf[0]]
        return np.array([eng.step(t) for t in toks]), lay
    monkeypatch.setenv("OTPU_SEXACT", "1")
    one, _ = run(1)
    two, lay = run(2)
    assert lay.msx and (lay.dchunk, lay.gs) == (128, 2)
    assert np.array_equal(one.view(np.uint32), two.view(np.uint32))
    monkeypatch.delenv("OTPU_SEXACT")
    off, lay = run(2)
    assert not lay.msx and (lay.dchunk, lay.gs) == (256, 2)
    assert not np.array_equal(one.view(np.uint32), off.view(np.uint32))


def test_tensor_parallel_experts_need_every_expert_resident(monkeypatch):
    """At S > 1 the experts are in a resident pool: an image of fewer slots (the expert server's
    path) is refused."""
    W, spec = _tiny(("attn",))
    with pytest.raises(ValueError, match="every expert resident"):
        _engine(spec, W, 2, monkeypatch, experts=K)


def test_tensor_parallel_experts_in_groups_are_bit_for_bit(monkeypatch):
    """dev.groups (OTPU_MOE_GROUPS): the routed experts' activations gathered in two groups (the
    first group's down MMs after the second's gate and up MMs) change no bit."""
    W, spec = _tiny(("attn",))
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 4)]
    got = {}
    for g in ("1", "2"):
        monkeypatch.setenv("OTPU_MOE_GROUPS", g)
        eng = _engine(spec, W, 2, monkeypatch)
        assert eng.image.moe_groups == int(g)
        got[g] = np.array([eng.step(t) for t in toks])
    assert np.array_equal(got["1"].view(np.uint32), got["2"].view(np.uint32))


def _goss(tmp_path, layers=2):
    """A tiny GPT-OSS (test_gptoss's: hidden 320, padded to 384 as 2880 to 2944; 8 q / 2 KV
    heads of 64; E 8, k 4, F 320; window 16): its weights and Spec."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from opentpu.llm.gptoss import Spec as GSpec
    torch.manual_seed(0)
    hc = transformers.GptOssConfig(
        hidden_size=320, intermediate_size=320, num_hidden_layers=layers, num_attention_heads=8,
        num_key_value_heads=2, head_dim=64, num_local_experts=8, num_experts_per_tok=4,
        vocab_size=1000, sliding_window=16, rms_norm_eps=1e-5, tie_word_embeddings=False,
        max_position_embeddings=131072, attn_implementation="eager")
    m = transformers.GptOssForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif "sinks" in n:
                p.copy_(torch.randn_like(p))
            elif n.endswith("bias"):
                p.copy_(0.1 * torch.randn_like(p))
            else:
                p.copy_((0.08 if "experts" in n else 0.05) * torch.randn_like(p))
    hc.save_pretrained(tmp_path)
    return {k: v.float().numpy() for k, v in m.state_dict().items()}, GSpec.from_hf(tmp_path)


@pytest.mark.parametrize("wformat,formats,kt", [("fp4", None, False), ("int8", None, False),
                                                ("fp4", "attn=fp4", False), ("fp4", None, True)])
def test_gptoss_tensor_parallel_is_s1_bit_for_bit(tmp_path, monkeypatch, wformat, formats, kt):
    """GPT-OSS at S = 2 (gptoss module docstring): a slice's KV head with its query group, the
    heads' outputs gathered before W_o's rows of the slice, x's 160 columns a slice gathered
    into its 320 (its padding to 384 zero), the experts tight (rows 160 of 320 a slice, moe.
    _moe_ffn_tp's two groups of two) and the router in whole groups: the resident decode's
    logits past the sliding window (16) and the generate loop's tokens are S = 1's bit for bit,
    with the attention projections int8 (Spec.formats) and fp4, and with MM KT's pairs (Config.KT
    at WG 8: the K tails of q / k / v, the router, the head, the experts' W_gate and W_up at K
    320 of 384 and W_down's last part at 64 of 128, as at S = 1)."""
    from opentpu.llm.qwen3 import Engine, device_config
    W, spec = _goss(tmp_path)
    monkeypatch.setenv("OTPU_POOL", "1")
    if formats:
        monkeypatch.setenv("OTPU_FORMATS", formats)
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 20)]
    lg, out = {}, {}
    kw = dict(WG=8, CHASH=True, GALL=True, KT=True) if kt else {}
    for S in (1, 2):
        cfg = device_config(spec, 256, S=S, rows=1, wformat=wformat, lookup=True, MCOLS=8, **kw)
        eng = Engine(spec, W, cap=256, cfg=cfg, rows=1, wformat=wformat, resident=True)
        assert eng.resident and eng.can_generate and eng.image.pool is not None
        assert eng.image.mf["wq"] == (formats or "attn=int8").split("=")[1]
        assert eng.image.fmt.kts == [kt, kt, False, kt] and eng.image.kt["router"] == kt
        lg[S] = np.array([eng.step(t) for t in toks])
        out[S] = eng.generate_card(int(np.argmax(lg[S][-1])), 6, stop_ids=[])
    assert np.array_equal(lg[1].view(np.uint32), lg[2].view(np.uint32))
    assert out[1] == out[2] and len(out[2]) == 6
