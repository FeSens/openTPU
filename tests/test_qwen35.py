"""Qwen3.5 on openTPU: the hybrid decoder (Gated DeltaNet with its fp32 state streamed through
TMEM, gated attention with 256-wide heads and partial RoPE) against Hugging Face transformers.
A tiny random model always runs; the real Qwen3.5-0.8B runs when its checkpoint is in
models/Qwen3.5-0.8B."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from opentpu import target as T
from opentpu.isasim import board_config, design_config
from opentpu.llm import load_spec
from opentpu.llm.lfm2 import plan, subs
from opentpu.llm.qwen3 import ATTN_BLOCK, Engine, load_weights
from opentpu.llm.qwen35 import Spec, emulated_logits, reference_logits

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

REAL = Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-0.8B"
KINDS = ("linear", "linear", "attn", "linear", "linear", "attn")   # a (lin, lin, attn) loop


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def _chat_ids(tok, text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                  enable_thinking=False, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def _tiny_model(nk: int, nv: int = 8, init: float = 0.02, ffn: int = 512, hidden: int = 256):
    """A random tiny Qwen3.5 (HF model, weights, Spec) with nv DeltaNet value heads, nk key
    heads, hidden wide and an MLP ffn wide. init: the initializer range (0.2: tokens that vary;
    at 0.02 the tied head mostly gives back the input token)."""
    torch.manual_seed(0)
    hc = transformers.Qwen3_5TextConfig(
        hidden_size=hidden, num_hidden_layers=len(KINDS), num_attention_heads=8,
        num_key_value_heads=2, head_dim=256, intermediate_size=ffn, vocab_size=1000,
        layer_types=["full_attention" if k == "attn" else "linear_attention" for k in KINDS],
        linear_num_key_heads=nk, linear_num_value_heads=nv, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=True,
        max_position_embeddings=4096, rms_norm_eps=1e-6, initializer_range=init,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25})
    m = transformers.Qwen3_5ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:     # zero-centered norms, but the DeltaNet output norm is plain
                p.copy_((1.0 if n.endswith("linear_attn.norm.weight") else 0.0)
                        + 0.1 * torch.randn_like(p))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    return m, W, Spec(hidden, KINDS, 8, 2, 256, 64, nv, 128, 128, ffn, 1000, lin_kheads=nk)


@pytest.fixture(scope="module", params=[8, 4], ids=["kh8", "kh4"])
def tiny(request):
    """8 DeltaNet heads (two pairs per slice at S=2, four at S=1: the head loop runs), and a
    query group of 4 heads (split in two on the board's 2-column MXU). kh4: 4 key heads for
    the 8 value heads (each key head's q and k serve two, as Qwen3.5-4B's 16 for 32)."""
    return _tiny_model(request.param)


def test_from_hf_refuses_a_rope_scaling(tmp_path):
    """Spec.from_hf refuses a RoPE type other than the default (mRoPE's sections are 1-D RoPE
    for text): Hugging Face's YaRN rescales the frequencies and cos / sin."""
    c = transformers.Qwen3_5TextConfig(
        hidden_size=256, num_hidden_layers=len(KINDS), num_attention_heads=8,
        num_key_value_heads=2, head_dim=256, intermediate_size=512, vocab_size=1000,
        layer_types=["full_attention" if k == "attn" else "linear_attention" for k in KINDS],
        linear_num_key_heads=8, linear_num_value_heads=8, linear_key_head_dim=128,
        linear_value_head_dim=128).to_dict()
    for rope, ok in (({"rope_type": "default", "rope_theta": 1e7, "mrope_interleaved": True,
                       "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25}, True),
                     ({"rope_type": "yarn", "factor": 4.0, "rope_theta": 1e7}, False)):
        (tmp_path / "config.json").write_text(json.dumps(dict(c, rope_parameters=rope)))
        if ok:
            assert Spec.from_hf(tmp_path).theta == 1e7
        else:
            with pytest.raises(ValueError, match="RoPE type"):
                Spec.from_hf(tmp_path)


def test_plan_loops_the_repeated_unit():
    assert plan(("linear", "linear", "linear", "attn") * 6) == [
        (0, ("linear", "linear", "linear", "attn"), 6)]
    assert plan(KINDS) == [(0, ("linear", "linear", "attn"), 2)]


def test_subs_loop_the_units_repeated_part():
    """lfm2.subs: a unit's DeltaNet layers before its attention layer are a loop of their own
    inside the unit's (the prefill programs' NEST), its part's bytes the stride."""
    size = {"linear": 3, "attn": 5}.get
    assert subs(plan(("linear", "linear", "linear", "attn") * 8), size) == {
        0: [(0, ("linear",), 3, 3), (3, ("attn",), 1, 5)]}
    assert subs(plan(KINDS), size) == {0: [(0, ("linear",), 2, 3), (2, ("attn",), 1, 5)]}
    assert subs(plan(("attn", "linear")), size) == {0: [(0, ("attn",), 1, 5)],
                                                    1: [(0, ("linear",), 1, 3)]}


@pytest.mark.parametrize("config", ["design", "board"])
def test_tiny_matches_hf(tiny, config):
    """design: 2 slices, 8 MXU columns; board: 1 slice, 2 columns (query groups split)."""
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 48)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(reference_logits(spec, W, toks) - hf).max() < 1e-4
    cfg = board_config(DRAM_BYTES=1 << 24) if config == "board" else None
    eng = Engine(spec, W, cap=256, cfg=cfg)
    dev = np.array([eng.step(t) for t in toks])
    assert _cos(dev, hf).min() > 0.998
    # the device follows the quantized math (it differs in fp32 rounding and in rounding the
    # weights to int8: the emulation divides by the scale, the device multiplies by 127/amax)
    emu = emulated_logits(spec, W, toks[:12], S=eng.cfg.S)
    assert _cos(dev[:12], emu).min() > 0.999


@pytest.mark.parametrize("config", ["design", "board"])
def test_tiny_fp4_follows_emulation(tiny, config):
    """4-bit (FP4) weights: the device follows the float64 emulation of the same weights."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 12)]
    cfg = board_config(DRAM_BYTES=1 << 24) if config == "board" else None
    eng = Engine(spec, W, cap=256, cfg=cfg, wformat="fp4")
    dev = np.array([eng.step(t) for t in toks])
    assert _cos(dev, emulated_logits(spec, W, toks, wformat="fp4", S=eng.cfg.S)).min() > 0.999


@pytest.mark.parametrize("grouped", [False, True], ids=["pairs", "groups"])
def test_tiny_formats_per_kind(tiny, monkeypatch, grouped):
    """Weight formats per kind (as test_llama's), the DeltaNet projections (in_proj_qkv / z,
    a / b, out_proj) a kind of their own, in either DeltaNet layout: the device follows the
    emulation of the same formats; resident decode and a chunked prefill give token-by-token
    decoding's logits bit for bit."""
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    _, W, spec = tiny
    spec = dataclasses.replace(spec, pair_loop=grouped)
    mix = dataclasses.replace(spec, formats="delta=fp4,attn=int4,gateup=fp4,head=int8")
    cfg = board_config(DRAM_BYTES=1 << 25)
    a = Engine(mix, W, cap=256, cfg=cfg)
    assert a.image.grouped == grouped
    assert a.image.mf == dict(wh="fp4", wab="fp4", wout="fp4", wq="int4", wgate="int4",
                              wk="int4", wv="int4", wo="int4", wg="fp4", wu="fp4", wd="int8")
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 12)]
    dev = np.array([a.step(t) for t in toks])
    emu, e8 = emulated_logits(mix, W, toks), emulated_logits(spec, W, toks)
    assert _cos(dev, emu).min() > 0.999
    assert np.abs(dev - emu).mean() < 0.7 * np.abs(dev - e8).mean()
    r, b = Engine(mix, W, cap=256, cfg=cfg, resident=True), Engine(mix, W, cap=256, cfg=cfg)
    assert r.resident
    for t, want in zip(toks[:4], dev):
        assert np.array_equal(r.step(t).view(np.uint32), want.view(np.uint32))
    assert np.array_equal(b.prefill(toks, chunk=4).view(np.uint32), dev[-1].view(np.uint32))


@pytest.mark.parametrize("grouped", [False, True], ids=["pairs", "groups"])
def test_tiny_formats_by_layer_range(tiny, monkeypatch, grouped):
    """Weight formats per layer range (formats.py's kind@a-b), in either DeltaNet layout: a
    layer block layout per formats group, the layers in runs of one unit of (kind, group) keys
    (plan: a hardware loop each, over blocks of its group's size): the device follows the
    emulation of the same formats; resident decode and a chunked prefill give token-by-token
    decoding's logits bit for bit, and decoding goes on from the prefill's KV cache, windows
    and states."""
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    _, W, spec = tiny
    spec = dataclasses.replace(spec, pair_loop=grouped)
    # the first half's MLP in fp4, the second half's mixers 4-bit: a loop in each layout
    mix = dataclasses.replace(spec, formats="mlp@0-2=fp4,delta@3-5=fp4,attn@3-5=int4")
    cfg = board_config(DRAM_BYTES=1 << 25)
    a = Engine(mix, W, cap=256, cfg=cfg)
    img = a.image
    g0, g = ("int8", "int8", "fp4", "fp4"), ("fp4", "int4", "int8", "int8")
    assert img.grouped == grouped and img.lf == (g0,) * 3 + (g,) * 3
    assert img.plan == [(0, (("linear", g0),), 2), (2, (("attn", g0),), 1),
                        (3, (("linear", g),), 2), (5, (("attn", g),), 1)]
    assert img.layouts[g0].size != img.layouts[g].size
    assert [img._off(i).const for i in range(6)] == sorted(img._off(i).const for i in range(6))
    assert mix.image(cfg, 256).nbytes < spec.image(cfg, 256).nbytes
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 12)]
    dev = np.array([a.step(t) for t in toks])
    emu, e8 = emulated_logits(mix, W, toks), emulated_logits(spec, W, toks)
    assert _cos(dev, emu).min() > 0.999
    assert np.abs(dev - emu).mean() < 0.7 * np.abs(dev - e8).mean()
    r, b = Engine(mix, W, cap=256, cfg=cfg, resident=True), Engine(mix, W, cap=256, cfg=cfg)
    assert r.resident
    for t, want in zip(toks[:4], dev):
        assert np.array_equal(r.step(t).view(np.uint32), want.view(np.uint32))
    assert np.array_equal(b.prefill(toks[:8], chunk=4).view(np.uint32), dev[7].view(np.uint32))
    for t, want in zip(toks[8:], dev[8:]):
        assert np.array_equal(b.step(t).view(np.uint32), want.view(np.uint32))


def test_tiny_reset_clears_state_and_conv_ring(tiny):
    """After reset, position 0 must not read the previous sequence's DeltaNet state or
    convolution rows."""
    _, W, spec = tiny
    eng = Engine(spec, W, cap=128)
    a = [eng.step(t) for t in (5, 6, 7, 8, 9)]
    eng.reset()
    b = [eng.step(t) for t in (5, 6, 7, 8, 9)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_embedding_format_by_table_size(tiny, tmp_path):
    """Spec.from_hf: an fp32 embedding table over EMBED_F32_MAX (2 GiB: Qwen3.5-4B's is 2.4)
    is int8; the 0.8B's (0.9 GiB) and the 2B's (1.9) stay fp32."""
    m = tiny[0]
    for hidden, embed in ((1024, "f32"), (2048, "f32"), (2560, "int8")):
        m.config.to_json_file(tmp_path / "config.json")
        c = json.loads((tmp_path / "config.json").read_text())
        c.update(hidden_size=hidden, vocab_size=248320)
        (tmp_path / "config.json").write_text(json.dumps(c))
        assert Spec.from_hf(tmp_path).embed == embed, hidden


@pytest.mark.parametrize("wformat", ["int8", "fp4"])
def test_its_programs_drain_at_the_arrays_rate(tiny, wformat):
    """Every MM the dense Qwen3.5 runs (DeltaNet and attention layers: the chunked prefill,
    the prompt runs, the step and resident decode programs, the decode loop) drains at its
    array's rate (compiler.slow_drains): where the array takes fewer cycles a column than its
    rows, their results are in distinct TMEM banks."""
    from conftest import assert_drains_keep_up, waits_programs
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 40)]
    cfg = board_config(DRAM_BYTES=1 << 25, PAIR=True)
    n = 0
    for kw in ({}, {"prompt_runs": True}):
        e = Engine(spec, W, cap=256, cfg=cfg, resident=True, wformat=wformat, **kw)
        progs = waits_programs(e)
        e.prefill(toks[:30])
        e.step(toks[30])
        e.generate_card(toks[31], 6, stop_ids=[])
        assert len(progs) >= 3
        n += assert_drains_keep_up(progs, e.cfg)
    assert n > 100


@pytest.mark.parametrize("head", ["int8", "fp4"])
def test_tiny_int8_embedding(tiny, head):
    """Spec.embed "int8": the token's int8 embedding row is gathered on the device from the
    tied int8 LM head (under a 4-bit head, from an int8 table of the image's own), in the
    resident decode, the per-position programs and the prefill runs alike: the same logits
    bit for bit, following the float64 emulation, which quantizes the row too."""
    _, W, spec = tiny
    spec = dataclasses.replace(spec, embed="int8")
    cfg = board_config(DRAM_BYTES=1 << 25, PAIR=True)
    wf = "int8" if head == "int8" else "fp4"
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 12)]
    a = Engine(spec, W, cap=256, cfg=cfg, resident=True, wformat=wf, head_format=head)
    b = Engine(spec, W, cap=256, cfg=cfg, wformat=wf, head_format=head)
    assert a.resident and b.device_inputs and a.image.lookup["own"] == (head != "int8")
    for t in toks[:3]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32))
    assert np.array_equal(a.prefill(toks[3:9]).view(np.uint32),
                          b.prefill(toks[3:9]).view(np.uint32))
    for t in toks[9:]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32))
    c = Engine(spec, W, cap=256, cfg=cfg, wformat=wf, head_format=head)
    dev = np.array([c.step(t) for t in toks])
    emu = emulated_logits(spec, W, toks, wformat=wf, head_format=head)
    assert _cos(dev, emu).min() > 0.999


def _layers_dram(eng):
    """The layer blocks (weights, KV cache, conv ring, DeltaNet state): all but the I/O area."""
    img = eng.image
    return [s.dram[img.layer0:img.nbytes] for s in eng.backend.machine.slices]


@pytest.mark.parametrize("config,first,chunk,dstep", [("design", 0, 5, False),
                                                      ("design", 2, 8, False),
                                                      ("board", 1, 4, False),
                                                      ("design", 0, 5, True),
                                                      ("board", 2, 6, True)])
def test_tiny_chunked_prefill_is_bit_exact(tiny, config, first, chunk, dstep):
    """Prefill in chunks (the DeltaNet recurrence row after row on a state loaded once per
    chunk, the convolution over the chunk and the ring from positions 0, 1 or 2 on, gated
    row attention; board: query groups split over the 2-column MXU) gives the same logits,
    KV cache, conv ring and DeltaNet state as token-by-token decode. dstep: the chunk's rows
    each run a DSTEP (the reference decodes on the VPU path)."""
    import dataclasses
    from opentpu.isasim import design_config
    _, W, spec = tiny
    cfg = board_config(DRAM_BYTES=1 << 24) if config == "board" else None
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 19)]
    ref = Engine(spec, W, cap=256, cfg=cfg)
    want = [ref.step(t) for t in toks]
    ecfg = dataclasses.replace(cfg or design_config(), DSTEP=True) if dstep else cfg
    eng = Engine(spec, W, cap=256, cfg=ecfg)
    for t in toks[:first]:
        eng.step(t)
    got = eng.prefill(toks[first:17], chunk=chunk)
    assert np.array_equal(got.view(np.uint32), want[16].view(np.uint32)) and eng.pos == 17
    assert eng.stats[first]["rows"] == chunk
    ref17 = Engine(spec, W, cap=256, cfg=cfg)
    ref17.prefill(toks[:17], chunk=1)
    assert all(np.array_equal(a, b) for a, b in zip(_layers_dram(eng), _layers_dram(ref17)))
    assert all(np.array_equal(eng.step(t), w) for t, w in zip(toks[17:], want[17:]))


@pytest.mark.parametrize("config,first,chunk,dstep", [("design", 0, 5, False),
                                                      ("board", 2, 6, False),
                                                      ("board", 1, 4, True)])
def test_tiny_prefill_vmac_is_bit_exact(tiny, monkeypatch, config, first, chunk, dstep):
    """VOP MAC (Config.VMAC): the chunked prefill's DeltaNet convolution (_deltanet_rows'
    head_in: a MUL of the last tap, then a MAC a tap in the same order and row groups) gives the
    logits, KV cache, conv rings and DeltaNet states of the MUL and ADD chain without it, bit for
    bit, and the next steps' logits."""
    from opentpu import compiler as C
    _, W, spec = tiny
    base = board_config(DRAM_BYTES=1 << 24) if config == "board" else design_config()
    base = dataclasses.replace(base, DSTEP=dstep)
    macs = []
    real = C.Builder.mac
    monkeypatch.setattr(C.Builder, "mac", lambda b, *a: macs.append(b.cfg.VMAC) or real(b, *a))
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 15)]
    out = []
    for vmac in (False, True):
        eng = Engine(spec, W, cap=256, cfg=dataclasses.replace(base, VMAC=vmac), prog_cache=False)
        for t in toks[:first]:
            eng.step(t)
        logits = eng.prefill(toks[first:first + chunk], chunk=chunk)
        out.append((logits, _layers_dram(eng), [eng.step(t) for t in toks[first + chunk:]]))
    assert macs and all(macs)
    (l0, d0, s0), (l1, d1, s1) = out
    assert np.array_equal(l0.view(np.uint32), l1.view(np.uint32))
    assert all(np.array_equal(a, b) for a, b in zip(d0, d1))
    assert all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(s0, s1))


@pytest.mark.parametrize("nk,grouped", [(16, False), (8, False), (8, True)],
                         ids=["kh16", "kh8", "kh8-grouped"])
def test_dstep_rows_pipeline_is_bit_exact(nk, grouped):
    """16 DeltaNet value heads on one slice (8 pairs, out_proj groups of 4) with DSTEP: the
    chunked prefill's pairs run software pipelined (_deltanet_rows' _rows_pipelined, its
    hardware loop over the middle pairs; 16 key heads, or 8 with q and k shared, array- or
    group-major) and give the logits, KV cache, conv rings and DeltaNet states of
    token-by-token decode, chunks of 2 and 6 from positions 0 and 5 (group-major: the taps,
    windows and states of every pair, then the next tokens' logits)."""
    import dataclasses
    _, W, spec = _tiny_model(nk, 16)
    spec = dataclasses.replace(spec, pair_loop=grouped)
    cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=True, PAIR=True)
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 13)]
    ref = Engine(spec, W, cap=256, cfg=cfg)
    assert ref.image.grouped == grouped
    want = [ref.step(t) for t in toks]
    eng = Engine(spec, W, cap=256, cfg=cfg)
    got = eng.prefill(toks[:2], chunk=2)
    assert np.array_equal(got.view(np.uint32), want[1].view(np.uint32))
    for t in toks[2:5]:
        eng.step(t)
    got = eng.prefill(toks[5:11], chunk=6)
    assert np.array_equal(got.view(np.uint32), want[10].view(np.uint32)) and eng.pos == 11
    assert eng.stats[0]["rows"] == 2 and eng.stats[4]["rows"] == 6
    ref11 = Engine(spec, W, cap=256, cfg=cfg)
    for t in toks[:11]:
        ref11.step(t)
    if grouped:     # (the group blocks also hold the decode's gates of its last token)
        for li, k in enumerate(spec.kinds):
            if k == "linear":
                for q in range(spec.lin_heads // 2):
                    assert all(np.array_equal(x, y) for x, y in
                               zip(_pair_parts(eng, li, q), _pair_parts(ref11, li, q))), (li, q)
    else:
        assert all(np.array_equal(a, b) for a, b in zip(_layers_dram(eng), _layers_dram(ref11)))
    assert all(np.array_equal(eng.step(t), w) for t, w in zip(toks[11:], want[11:]))


@pytest.mark.parametrize("wf", ["fp4", "int4"])
def test_prefill_pair_sum_order(wf):
    """PAIR with 4-bit weights: a decode step's MMs pair (column reuse), and so do a prefill
    run's of at most MCOLS / 2 rows -- those runs are bit-identical to the steps. A wider
    run's MMs do not pair, and a paired MM adds each even K-block to the odd one before the
    partial sums: the same products, summed in another order (qwen35_rows). An activation's
    int8 rounding downstream can turn that last-bit difference into a quantization step: the
    logits stay within the pinned tolerance (this prompt: ~1e-2 of the largest with fp4,
    ~1e-7 with int4; 1 - cos ~1e-4 at most over the prompts tried)."""
    _, W, spec = _tiny_model(8)
    cfg = board_config(DRAM_BYTES=1 << 25, PAIR=True, MCOLS=4)
    kw = dict(wformat=wf, head_format="int8")
    toks = [int(t) for t in np.random.default_rng(17).integers(0, 1000, 17)]
    ref = Engine(spec, W, cap=256, cfg=cfg, **kw)
    for t in toks:
        want = ref.step(t)
    paired = Engine(spec, W, cap=256, cfg=cfg, **kw).prefill(toks, chunk=2)
    assert np.array_equal(paired.view(np.uint32), want.view(np.uint32))
    wide = Engine(spec, W, cap=256, cfg=cfg, **kw).prefill(toks)        # runs of 8 rows
    cos = float(wide @ want / np.linalg.norm(wide) / np.linalg.norm(want))
    assert 1 - cos < 1e-3 and np.abs(wide - want).max() < 5e-2 * np.abs(want).max()
    assert np.argmax(wide) == np.argmax(want)


@pytest.mark.parametrize("config", ["design", "board"])
@pytest.mark.parametrize("unit", ["dstep", "stream"])
def test_tiny_dstep_is_bit_exact(tiny, config, unit):
    """DSTEP (the DMA streams each DeltaNet head's state through its datapath) gives the same
    logits, DeltaNet state, conv ring and KV cache as the VPU's RDOT / OUTER passes, word for
    word, from position 0 on; chunked prefill (the VPU path) continues from its state. unit
    "stream": a stream-engine machine without DSTEP runs each step as STREAM (isa.gdn_desc)."""
    import dataclasses
    from opentpu.isasim import design_config
    _, W, spec = tiny
    base = board_config(DRAM_BYTES=1 << 24) if config == "board" else design_config()
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 9)]
    ref = Engine(spec, W, cap=256, cfg=base)
    kw = {"DSTEP": True} if unit == "dstep" else {"STREAM": True}
    eng = Engine(spec, W, cap=256, cfg=dataclasses.replace(base, **kw))
    assert any(i.op == (0x12 if unit == "dstep" else 0x13)
               for i in eng.image.compile_step(1)[0])
    for t in toks[:6]:
        assert np.array_equal(ref.step(t).view(np.uint32), eng.step(t).view(np.uint32))
    assert all(np.array_equal(a, b) for a, b in zip(_layers_dram(eng), _layers_dram(ref)))
    a, b = ref.prefill(toks[6:], chunk=3), eng.prefill(toks[6:], chunk=3)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.parametrize("resident", [False, True])
def test_tiny_filling_step_programs_are_transparent(tiny, resident):
    """The step programs that fill their logits first (the card's streamed decode) give the
    plain programs' logits and DRAM bit for bit, per position and resident."""
    from conftest import assert_fill_is_transparent
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(8).integers(0, 1000, 6)]
    assert_fill_is_transparent(lambda: Engine(spec, W, cap=256, resident=resident), toks)


@pytest.mark.parametrize("dstep", [False, True])
def test_tiny_resident_decode_is_bit_exact(tiny, dstep):
    """Resident decode (one program per 256-position attention bucket, the token and position
    as run arguments, the embedding and RoPE rows from the image's tables, the conv ring read
    through its mirrored rows at a run-time offset) gives the per-position programs' logits bit
    for bit: from the ring's first positions (0 .. 2 run per-position programs), after a
    chunked prefill and across the bucket boundary at 256; the layer blocks (KV cache, conv
    ring, DeltaNet state) agree too, but for the rings' scratch rows."""
    _, W, spec = tiny
    cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=dstep, PAIR=True)
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 262)]
    a = Engine(spec, W, cap=512, cfg=cfg, resident=True)
    b = Engine(spec, W, cap=512, cfg=cfg)
    assert a.resident and not b.resident
    p = 0
    for stop, run in ((0, 6), (252, 10)):
        if stop > p:
            assert np.array_equal(a.prefill(toks[p:stop]), b.prefill(toks[p:stop]))
            p = stop
        for t in toks[p:p + run]:
            pa = a.pos
            assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32)), pa
        p += run
    assert sorted(a._decodes) == [1, 2] and not b._decodes
    ia, ib = a.image, b.image
    assert (ia.layer0, ia.loc, ia.head) == (ib.layer0, ib.loc, ib.head)
    ma, mb = (e.backend.machine.slices[0].dram[ib.layer0:ib.head[0]].copy() for e in (a, b))
    TP = spec.conv_k * ib.CP
    for li, k in enumerate(spec.kinds):     # row 0 of each pair's ring is scratch (_ring)
        if k == "linear":
            for q in range(ib.nl // 2):
                o = ib._off(li).const - ib.layer0 + ib.cv_offset(q) + 4 * TP
                ma[o:o + 4 * ib.CP] = mb[o:o + 4 * ib.CP] = 0
    assert np.array_equal(ma, mb)


@pytest.fixture(scope="module", params=[(16, 8), (18, 9)], ids=["h16", "h18"])
def tiny_pairs(request):
    """More DeltaNet heads on one slice, so the decode's pair loop runs: 16 value heads for 8
    key heads (8 pairs, as Qwen3.5-35B-A3B's 32 for 16), and 18 for 9 (9 pairs, out_proj
    groups of 2 heads)."""
    nv, nk = request.param
    return _tiny_model(nk, nv)[1:]


@pytest.mark.parametrize("grouped", [False, True], ids=["array", "group"])
def test_tiny_image_built_in_workers(tiny_pairs, grouped, monkeypatch):
    """The image with its matrices quantized in worker processes (opentpu.qcache.Batch: every
    matrix there, in jobs of a few rows) is the one-process build's, byte for byte, in both
    DeltaNet layouts (the group-major one quantizes each pair's rows on its own)."""
    from opentpu import qcache
    W, spec = tiny_pairs
    cfg = board_config(DRAM_BYTES=1 << 25, PAIR=True)
    eng = Engine(dataclasses.replace(spec, pair_loop=grouped), W, cap=256, cfg=cfg,
                 wformat="fp4", head_format="int8", resident=True)
    assert eng.image.grouped == grouped
    monkeypatch.setenv("OTPU_BUILD_JOBS", "1")
    want = eng.image.build(W)
    monkeypatch.setenv("OTPU_BUILD_JOBS", "2")
    monkeypatch.setattr(qcache, "INLINE", 0)
    monkeypatch.setattr(qcache, "JOB", 3 * 256)
    got = eng.image.build(W)
    assert all(np.array_equal(a, b) for a, b in zip(got, want))


def _pair_parts(eng, li, q):
    """Pair q's taps and window, and its two states, in layer li (either layout)."""
    im, d = eng.image, eng.backend.machine.slices[0].dram
    o = im._off(li).const
    n = 4 * 2 * eng.spec.lin_dv * eng.spec.lin_dk
    st = o + (im.pair_offset(q) + im.pofs["state"] if im.grouped else
              im.lofs["linear"]["state"] + q * n)
    return d[o + im.cv_offset(q):o + im.cv_offset(q) + 4 * im.CVW].copy(), d[st:st + n].copy()


@pytest.mark.parametrize("dstep", [False, True, "pair"])
def test_group_major_is_bit_exact(tiny_pairs, dstep):
    """The group-major DeltaNet layout (Spec.pair_loop) gives the array-major one's logits,
    taps, windows and states bit for bit: chunked prefill (a loop over the groups, each over
    its pairs), per-position decode (the pair loop), resident decode (the pair loop at a
    run-time position: a shorter program); with and without DSTEP, and with a shared pair in
    one DSTEP (Config.DSTEP2: in the pair loop too)."""
    W, spec = tiny_pairs
    cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=bool(dstep), DSTEP2=dstep == "pair", PAIR=True)
    a, b = (Engine(dataclasses.replace(spec, pair_loop=g), W, cap=512, cfg=cfg, resident=True)
            for g in (False, True))
    assert not a.image.grouped and b.image.grouped
    ra, rb = (e.image.compile_decode(1, 3)[0][0] for e in (a, b))
    assert len(rb) < len(ra)
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 12)]
    assert np.array_equal(a.prefill(toks[:5], chunk=3).view(np.uint32),
                          b.prefill(toks[:5], chunk=3).view(np.uint32))
    for t in toks[5:]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32)), a.pos
    assert sorted(a._decodes) == sorted(b._decodes) == [1]
    for li, k in enumerate(spec.kinds):
        if k == "linear":
            for q in range(spec.lin_heads // 2):
                assert all(np.array_equal(x, y) for x, y in
                           zip(_pair_parts(a, li, q), _pair_parts(b, li, q))), (li, q)


@pytest.mark.parametrize("step", ["vop", "dstep", "stream"])
def test_shared_qk_is_bit_exact(tiny, step):
    """A pair of value heads of one key head (kh4: 4 key heads for 8 value heads) projects,
    convolves and normalizes its q and k once (Spec.qk_share): the logits and states of the
    repeated q and k rows, bit for bit, in chunked prefill, per-position and resident decode,
    with the state step on VOPs, on DSTEP or on STREAM; a pair's rows are 2 dk fewer."""
    _, W, spec = tiny
    if spec.lin_heads // spec.lin_nk % 2:
        pytest.skip("one value head per key head: nothing to share")
    cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=step == "dstep", STREAM=step == "stream",
                       PAIR=True)
    a, b = (Engine(dataclasses.replace(spec, qk_share=g), W, cap=512, cfg=cfg, resident=True)
            for g in (False, True))
    assert not a.image.shared and b.image.shared
    assert b.image.RP == a.image.RP - 2 * spec.lin_dk and b.image.CP == a.image.CP - 2 * spec.lin_dk
    toks = [int(t) for t in np.random.default_rng(6).integers(0, 1000, 12)]
    assert np.array_equal(a.prefill(toks[:5], chunk=3).view(np.uint32),
                          b.prefill(toks[:5], chunk=3).view(np.uint32))
    for t in toks[5:]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32)), a.pos
    for li, k in enumerate(spec.kinds):
        if k == "linear":
            for q in range(a.image.nl // 2):
                assert np.array_equal(_pair_parts(a, li, q)[1], _pair_parts(b, li, q)[1])


@pytest.fixture(scope="module")
def tiny_rk3():
    """12 DeltaNet value heads for 4 key heads (3 a key head, as Qwen3.5-27B's 48 for 16): on
    one slice 6 pairs, 2 of them straddling two key heads; on two slices 3 pairs and 2 key
    heads each."""
    return _tiny_model(4, 12)


@pytest.mark.parametrize("config,step", [("board", "vop"), ("board", "dstep"),
                                         ("board", "stream"), ("design", "vop")])
def test_qk_once_is_bit_exact(tiny_rk3, config, step):
    """3 value heads a key head: a slice's key heads' q and k projected, convolved and normed
    once a layer (Spec.qk_once), its pairs' rows v and z only, gives the repeated q and k
    rows' logits and states bit for bit, in chunked prefill (the pipelined DSTEP rows too),
    per-position and resident decode, with the state step on VOPs, DSTEP or STREAM; the
    image holds 2 dk rows a value head fewer, plus 2 dk a key head."""
    _, W, spec = tiny_rk3
    cfg = (board_config(DRAM_BYTES=1 << 25, DSTEP=step == "dstep", STREAM=step == "stream",
                        PAIR=True) if config == "board" else design_config(DRAM_BYTES=1 << 25))
    a, b = (Engine(dataclasses.replace(spec, qk_once=g), W, cap=512, cfg=cfg,
                   resident=config == "board") for g in (False, True))
    assert not a.image.qkonce and b.image.qkonce
    assert dataclasses.replace(spec, qk_once=None).image(cfg, 512).qkonce      # (the default)
    nl, nk, dk = a.image.nl, b.image.nkl, spec.lin_dk
    assert nl // 2 * (a.image.RP - b.image.RP) == 4 * dk * (nl // 2) and b.image.QC == 2 * dk * nk
    toks = [int(t) for t in np.random.default_rng(6).integers(0, 1000, 12)]
    assert np.array_equal(a.prefill(toks[:5], chunk=3).view(np.uint32),
                          b.prefill(toks[:5], chunk=3).view(np.uint32))
    for t in toks[5:]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32)), a.pos
    for li, k in enumerate(spec.kinds):
        if k == "linear":
            for q in range(nl // 2):
                assert np.array_equal(_pair_parts(a, li, q)[1], _pair_parts(b, li, q)[1])


def test_qk_once_matches_hf(tiny_rk3):
    """The tiny 3-to-1 model with q and k once a layer against Hugging Face, decoding (design
    configuration: 2 slices of 2 key heads)."""
    m, W, spec = tiny_rk3
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 24)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(reference_logits(spec, W, toks) - hf).max() < 1e-4
    eng = Engine(spec, W, cap=256)
    assert eng.image.qkonce and eng.cfg.S == 2
    dev = np.array([eng.step(t) for t in toks])
    assert _cos(dev, hf).min() > 0.998


@pytest.fixture(scope="module")
def tiny27():
    """Qwen3.5-27B's ratios in a tiny model (_tiny27)."""
    return _tiny27()


def _tiny27():
    """Qwen3.5-27B's ratios in a tiny model: 48 DeltaNet value heads for 16 key heads, 8 query
    heads for 4 KV heads, an untied head, the int8 embedding table; the WG targets' configuration
    (targets/bcu1525-x2.toml: 4 slices, 8 MXU columns, weight groups at WG 8, DSTEP)."""
    torch.manual_seed(0)
    kinds = ("linear", "linear", "linear", "attn")
    hc = transformers.Qwen3_5TextConfig(
        hidden_size=512, num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4,
        head_dim=256, intermediate_size=1024, vocab_size=1024,
        layer_types=["full_attention" if k == "attn" else "linear_attention" for k in kinds],
        linear_num_key_heads=16, linear_num_value_heads=48, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=False,
        max_position_embeddings=4096, rms_norm_eps=1e-6, initializer_range=0.02,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25})
    m = transformers.Qwen3_5ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_((1.0 if n.endswith("linear_attn.norm.weight") else 0.0)
                        + 0.1 * torch.randn_like(p))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    spec = Spec(512, kinds, 8, 4, 256, 64, 48, 128, 128, 1024, 1024, tied=False, lin_kheads=16,
                embed="int8")
    return m, W, spec, T.config("bcu1525-x2", DRAM_BYTES=1 << 26, BHW=0)


def test_tiny_27b_shape_on_four_slices(tiny27):
    """Qwen3.5-27B's ratios on the WG targets' configuration: 12 DeltaNet value heads and 4 key heads
    a slice (q and k once a layer, out_proj groups of 4), one KV head a slice, an untied head
    and the int8 embedding table split by columns over the slices (each gathers its 128 of 512
    columns, then all-gathers): against Hugging Face; bit for bit the logits of the
    host-written embedding slot's whole rows, of the resident decode and of a chunked
    prefill."""
    m, W, spec, cfg = tiny27
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1024, 8)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    a = Engine(spec, W, cap=256, cfg=cfg)
    b = Engine(spec, W, cap=256, cfg=cfg, embed_host=True)
    c = Engine(spec, W, cap=256, cfg=cfg, resident=True)
    assert a.image.qkonce and a.image.og == 4 and a.image.lookup["split"] == 4
    assert b.image.embed_host and c.resident
    dev = []
    for t in toks:
        x, y, z = a.step(t), b.step(t), c.step(t)
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32)), a.pos
        assert np.array_equal(x.view(np.uint32), z.view(np.uint32)), a.pos
        dev.append(x)
    assert _cos(np.array(dev), hf).min() > 0.995
    d = Engine(spec, W, cap=256, cfg=cfg)
    assert np.array_equal(d.prefill(toks[:5], chunk=3).view(np.uint32), dev[4].view(np.uint32))


@pytest.mark.parametrize("mcols,wf", [(8, "fp4"), (1, "int8")])
def test_tiny_27b_shape_at_wg8_is_wg1s(tiny27, monkeypatch, mcols, wf):
    """The tiny 27B shape's image grouped at WG 8 (the WG targets' weight groups; q and k once,
    the split embedding table natural): WG 1's logits bit for bit, per position, resident and
    in a 3-row chunked prefill; WG 8's MMs over the weights grouped (rd 3 int8, 4 fp4). GALL off
    (the ISA's rows of one beside the groups: the attention's, the table's); the WG targets' GALL:
    test_wg's test_tiny_27b_shape_gall_is_natural."""
    from opentpu import isasim
    _, W, spec, cfg = tiny27
    rds, mm = [], isasim.Slice._mm
    monkeypatch.setattr(isasim.Slice, "_mm", lambda s, ins: (rds.append(ins.rd), mm(s, ins))[1])
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1024, 8)]
    out = []
    for wg in (1, 8):
        c = dataclasses.replace(cfg, MCOLS=mcols, WG=wg, GALL=False)
        rds.clear()
        a = Engine(spec, W, cap=256, cfg=c, wformat=wf)
        b = Engine(spec, W, cap=256, cfg=c, wformat=wf, resident=True)
        assert a.image.qkonce and a.image.lookup["split"] == 4 and b.resident
        got = [np.array([e.step(t) for t in toks]) for e in (a, b)]
        got.append(Engine(spec, W, cap=256, cfg=c, wformat=wf).prefill(toks[:5], chunk=3))
        lg = {0} if wg == 1 else {0, 3 if wf == "int8" else 4}
        assert lg <= set(rds) <= (lg | {3}), set(rds)          # (an int8 head's LG 3)
        out.append([g.view(np.uint32) for g in got])
    for x, y in zip(*out):
        assert np.array_equal(x, y)
    assert np.array_equal(out[0][0], out[0][1]) and np.array_equal(out[0][0][4], out[0][2])


def test_tiny_27b_shape_cv_ahead_is_bit_exact(tiny27):
    """The WG targets' DSTEP decode with every pair's taps, window and gates in TMEM before the steps
    and the windows stored after them (Spec.cv_ahead, on at WG 8), key head 0's q and k first:
    the logits and every layer block (weights, KV, windows, states) bit for bit those of the
    per-pair loads and stores, per position, resident too."""
    _, W, spec, cfg = tiny27
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1024, 6)]
    engs = [Engine(dataclasses.replace(spec, cv_ahead=on), W, cap=256, cfg=cfg, resident=res)
            for on in (False, True) for res in (False, True)]
    assert [e.image.cv_ahead for e in engs] == [False, False, True, True]
    for t in toks:
        got = [e.step(t).view(np.uint32) for e in engs]
        assert all(np.array_equal(got[0], g) for g in got[1:]), engs[0].pos
    assert all(np.array_equal(a, b) for e in engs[1:]
               for a, b in zip(_layers_dram(engs[0]), _layers_dram(e)))


def test_tiny_27b_shape_dstep_pair_is_bit_exact(tiny27):
    """The WG targets' DSTEP decode with a pair's two heads in one DSTEP (Config.DSTEP2, flag PAIR; its
    pairs' key heads the same and the next: QKB both ways): the logits and every layer block bit
    for bit those of a DSTEP a head, per position, resident too."""
    from opentpu import isa as I
    _, W, spec, cfg = tiny27
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1024, 6)]
    engs = [Engine(spec, W, cap=256, cfg=dataclasses.replace(cfg, DSTEP2=on), resident=res)
            for on in (False, True) for res in (False, True)]
    fl = [i.flags for p in engs[2].image.compile_step(5) for i in p if i.op == I.DSTEP]
    assert fl and all(f & I.F_DPAIR for f in fl) and {f & I.F_DQKB for f in fl} == {0, I.F_DQKB}
    assert not any(i.flags & I.F_DPAIR for p in engs[0].image.compile_step(5) for i in p
                   if i.op == I.DSTEP)
    for t in toks:
        got = [e.step(t).view(np.uint32) for e in engs]
        assert all(np.array_equal(got[0], g) for g in got[1:]), engs[0].pos
    assert all(np.array_equal(a, b) for e in engs[1:]
               for a, b in zip(_layers_dram(engs[0]), _layers_dram(e)))


@pytest.mark.parametrize("target", ["bcu1525-wg", "bcu1525-x2", "bcu1525-wg+st16"])
@pytest.mark.parametrize("pair", [False, True], ids=["dstep", "dstep2"])
def test_tiny_27b_shape_rows_pipeline_is_bit_exact(tiny27, target, pair):
    """The 27B's DeltaNet shape (48 value heads on 16 key heads: qkonce) in prompt rows with
    DSTEP, pipelined (_rows_pipelined): the gates in TMEM, the key heads a group of two at a time
    (one slice: 24 pairs, 8 groups; the WG targets: 6 pairs a slice, 2 groups) and, with DSTEP2, a pair's
    two heads a row in one DSTEP (QKB both ways). Chunks of 2 and 6 rows from positions 0 and 5
    give the logits and every layer block of token-by-token decode, bit for bit; +st16: with
    the states in 16 bits (Config.ST16 e6m9, bhw16's; every DSTEP's flag ST16)."""
    from opentpu import isa as I
    _, W, spec, _ = tiny27
    t, _, st = target.partition("+")
    cfg = T.config(t, DRAM_BYTES=1 << 26, BHW=0, DSTEP2=pair, ST16="e6m9" if st else "")
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1024, 13)]
    ref = Engine(spec, W, cap=256, cfg=cfg)
    want = [ref.step(t) for t in toks]
    eng = Engine(spec, W, cap=256, cfg=cfg)
    fl = [i.flags for p in eng.image.compile_rows([(0, 5 + j) for j in range(6)], [5])
          for i in p if i.op == I.DSTEP]
    assert fl and all(bool(f & I.F_DPAIR) == pair for f in fl)
    assert all(bool(f & I.F_DST16) == bool(cfg.ST16) for f in fl) and eng.image.st16 == bool(cfg.ST16)
    assert not pair or {f & I.F_DQKB for f in fl} == {0, I.F_DQKB}
    got = eng.prefill(toks[:2], chunk=2)
    assert np.array_equal(got.view(np.uint32), want[1].view(np.uint32))
    for t in toks[2:5]:
        eng.step(t)
    got = eng.prefill(toks[5:11], chunk=6)
    assert np.array_equal(got.view(np.uint32), want[10].view(np.uint32)) and eng.pos == 11
    ref11 = Engine(spec, W, cap=256, cfg=cfg)
    for t in toks[:11]:
        ref11.step(t)
    assert all(np.array_equal(a, b) for a, b in zip(_layers_dram(eng), _layers_dram(ref11)))
    assert all(np.array_equal(eng.step(t), w) for t, w in zip(toks[11:], want[11:]))


def test_tiny_27b_shape_bursts_a_pair_beside_is_bit_exact(tiny27):
    """The WG targets' DSTEP decode with the streams on an engine of their own (Config.SE_OWN, the
    target's SE_R 1; CAPS2 bit3): a DeltaNet burst a pair (qwen35._bursts) instead of 1, 2, 2, 1.
    Other programs, the same image bytes, and the logits and every layer block bit for bit those
    of 1, 2, 2, 1, per position, resident too."""
    from opentpu import isa as I
    from opentpu.llm.qwen35 import _bursts
    _, W, spec, cfg = tiny27
    assert cfg.SE_OWN and _bursts(6, True) == [[q] for q in range(6)]
    assert _bursts(6) == [[0], [1, 2], [3, 4], [5]]
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1024, 6)]
    engs = [Engine(spec, W, cap=256, cfg=dataclasses.replace(cfg, SE_OWN=on), resident=res)
            for on in (False, True) for res in (False, True)]
    assert all(np.array_equal(a, b) for e in engs[1:]
               for a, b in zip(_layers_dram(engs[0]), _layers_dram(e)))
    w = [I.assemble(p).tolist() for e in (engs[0], engs[2]) for p in e.image.compile_step(5)]
    assert w[:cfg.S] != w[cfg.S:]
    for t in toks:
        got = [e.step(t).view(np.uint32) for e in engs]
        assert all(np.array_equal(got[0], g) for g in got[1:]), engs[0].pos
    assert all(np.array_equal(a, b) for e in engs[1:]
               for a, b in zip(_layers_dram(engs[0]), _layers_dram(e)))


@pytest.mark.parametrize("model", ["kh4", "27b"])
def test_dn_fold_matches_hf_and_its_paths_agree(tiny27, model):
    """Spec.dn_fold: the DeltaNet's scale constants in its weights (SiLU's and beta's -log2(e)
    in the taps and z's and b's rows, -log2(e) e^A_log, q's dk^-1/2 in the output norm's eps:
    qwen35._silu_dn, _l2n, _dn_eps; Image.build's constants: the taps, z's and b's Sw, A_log's
    slot, the output norm's gain). Against Hugging
    Face as the unfolded image; every path bit for bit the others' (the VPU's recurrence and
    DSTEP on kh4, the WG targets' bursts on the 27B shape, the resident decode, a 3-row chunked
    prefill); a decode step only MULs and EXP2s fewer, the unfolded image's unchanged (Spec
    .dn_fold off: the programs as before)."""
    from opentpu import isa as I
    if model == "27b":
        m, W, spec, cfg = tiny27
        cfgs = [cfg]
    else:
        m, W, spec = _tiny_model(4)
        cfgs = [board_config(DRAM_BYTES=1 << 25, DSTEP=d) for d in (False, True)]
    fs, fw = dataclasses.replace(spec, dn_fold=True), W
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 6)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    got = []
    for cfg in cfgs:
        u = Engine(spec, W, cap=256, cfg=cfg)
        engs = [Engine(fs, fw, cap=256, cfg=cfg, resident=r) for r in (False, True)]
        assert u.image.dn_fold is False and engs[0].image.dn_fold is True
        counts = []
        for e in (u, engs[0]):
            n = {}
            for i in e.image.compile_step(5)[0]:
                k = (i.op, (i.w[5] >> 16) & 0xFF if i.op == I.VOP else 0)
                n[k] = n.get(k, 0) + 1
            counts.append(n)
        less = {k for k in counts[0] | counts[1] if counts[0].get(k) != counts[1].get(k)}
        assert less == {(I.VOP, I.V_MUL), (I.VOP, I.V_EXP2)}, less
        assert all(counts[1][k] < counts[0][k] for k in less)
        du = np.array([u.step(t) for t in toks])
        df = [np.array([e.step(t) for t in toks]) for e in engs]
        pf = Engine(fs, fw, cap=256, cfg=cfg).prefill(toks[:5], chunk=3)
        assert np.array_equal(df[0].view(np.uint32), df[1].view(np.uint32))
        assert np.array_equal(pf.view(np.uint32), df[0][4].view(np.uint32))
        assert _cos(df[0], hf).min() > _cos(du, hf).min() - 2e-4 > 0.995
        got.append(df[0].view(np.uint32))
    assert all(np.array_equal(got[0], g) for g in got[1:])


@pytest.mark.parametrize("fold", [False, True])
def test_tiny_vsig_is_bit_exact(fold):
    """Config.VSIG (VOP SIGMOID / SILU, the wmx VPU's): the DeltaNet's SiLUs (folded: silu_l2's
    SILU with B = 1), the gated attention's sigmoid and the MLP's SiLU are one VOP each; decode
    steps, the resident decode and a 3-row chunked prefill give the logits without it bit for
    bit, the step program with fewer VOPs."""
    from opentpu import isa as I
    _, W, spec = _tiny_model(4)
    spec = dataclasses.replace(spec, dn_fold=fold)
    base = board_config(DRAM_BYTES=1 << 25, DSTEP=True)
    vcfg = dataclasses.replace(base, VSIG=True)
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 6)]
    u, v = Engine(spec, W, cap=256, cfg=base), Engine(spec, W, cap=256, cfg=vcfg)
    fn = [[(i.w[5] >> 16) & 0xFF for i in e.image.compile_step(5)[0] if i.op == I.VOP]
          for e in (u, v)]
    assert not {I.V_SIGMOID, I.V_SILU} & set(fn[0])
    assert fn[1].count(I.V_SIGMOID) > 0 and fn[1].count(I.V_SILU) > 0
    assert len(fn[1]) < len(fn[0])
    du = np.array([u.step(t) for t in toks]).view(np.uint32)
    dv = np.array([v.step(t) for t in toks]).view(np.uint32)
    assert np.array_equal(du, dv)
    r = Engine(spec, W, cap=256, cfg=vcfg, resident=True)
    assert np.array_equal(np.array([r.step(t) for t in toks]).view(np.uint32), du)
    pf = Engine(spec, W, cap=256, cfg=vcfg).prefill(toks[:5], chunk=3)
    assert np.array_equal(pf.view(np.uint32), du[4])


def test_tiny_vsig_on_board_model(have_verilator):
    """Config.VSIG on the board model (the default target's VPU_CNS 3: SIGMOID and SILU in six
    passes of otpu_se_comp's loop, VSIG through otpu_board and otpu_slice): the DeltaNet's SiLUs
    (folded), the gated attention's sigmoid and the MLP's SiLU as one VOP each give the ISA
    simulator's logits bit for bit, resident from position 3 on."""
    from opentpu.host.board import BoardBackend, SimTransport
    _, W, spec = _tiny_model(4)
    spec = dataclasses.replace(spec, dn_fold=True)
    cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=True, VSIG=True)
    isa = Engine(spec, W, cap=256, cfg=cfg)
    # (the model's build: the environment's board_config and the target's flags, so VPU_SIG here)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5, params={"VPU_SIG": 1})
    brd = Engine(spec, W, cap=256, cfg=cfg, resident=True,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    for tok in (11, 222, 333, 444, 555):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.parametrize("dstep,resident,formats", [
    (False, False, ""), (True, False, ""), (True, True, ""), ("stream", False, ""),
    (True, True, "delta=fp4,attn=int4,gateup=fp4"),
    (True, True, "mlp@0-2=fp4,delta@3-5=fp4,attn@3-5=int4")])
def test_tiny_qwen35_on_board_model(tiny, have_verilator, dstep, resident, formats):
    """The board model through the host driver, through a full turn of the convolution window:
    logits bit-identical to the ISA simulator (with and without DSTEP; resident: from position
    3 on the resident decode program, its token and position as run arguments; "stream": each
    DeltaNet head step as a STREAM on the stream engine instead of DSTEP; formats: per-kind
    weight formats, int8, fp4 and int4 MMs in one model, and per layer range: a loop in each
    of two layouts)."""
    from opentpu.host.board import BoardBackend, SimTransport
    _, W, spec = tiny
    spec = dataclasses.replace(spec, formats=formats)
    cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=dstep is True, STREAM=dstep == "stream")
    isa = Engine(spec, W, cap=256, cfg=cfg)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5)
    brd = Engine(spec, W, cap=256, cfg=cfg, resident=resident,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    assert brd.resident == resident
    for tok in (11, 222, 333, 444, 555, 666):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert brd.stats[-1]["cycles"] > 0
    if resident:
        assert sorted(brd._decodes) == [1] and brd.backend._resident[0] is brd._decodes[1][0]


@pytest.mark.parametrize("dstep", [False, True])
def test_group_major_on_board_model(tiny_pairs, have_verilator, dstep):
    """Group-major DeltaNet blocks on the board model through the host driver, resident from
    position 3 on (the pair loop at a run-time position): logits bit-identical to the ISA
    simulator's array-major per-position programs."""
    from opentpu.host.board import BoardBackend, SimTransport
    W, spec = tiny_pairs
    cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=dstep)
    isa = Engine(dataclasses.replace(spec, pair_loop=False), W, cap=256, cfg=cfg)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5)
    brd = Engine(dataclasses.replace(spec, pair_loop=True), W, cap=256, cfg=cfg, resident=True,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    assert brd.resident and brd.image.grouped and not isa.image.grouped
    for tok in (11, 222, 333, 444, 555, 666):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert sorted(brd._decodes) == [1]


@pytest.mark.parametrize("embed", ["f32", "int8"])
def test_4b_resident_decode_fits_imem(embed):
    """Qwen3.5-4B's dims on one slice (16 pairs of DeltaNet heads: group-major), fp4 with an
    int8 LM head: the resident decode, the pair loop at a run-time position beside the
    attention's run-time KV addresses, compiles and fits IMEM at every bucket of a 4K context
    (the gates in the head groups' blocks: the pair loop takes one address register), with the
    fp32 embedding table and with the int8 one gathered on the device (the 4B's: two more
    arguments, the token's registers given back after the gather). A layout-only image, no
    weights. The fp32 table (2.4 GiB) takes the image past 4 GiB: at AW 32 its addresses do
    not fit an LD's word (isa._w: they wrapped before AW); at AW 36 (the weights from 4 GiB,
    the states below: two registers a layer loop) it fits to bucket 4 and from bucket 8 is a
    register short (docs/compiler.md "Addresses above 4 GiB": open; since the masked block's
    sub-block read, 66f658be, a device-computed value's runs out before an address's)."""
    from opentpu.compiler import CompileError
    kinds = ("linear", "linear", "linear", "attn") * 8
    spec = Spec(2560, kinds, 16, 4, 256, 64, 32, 128, 128, 9216, 248320, lin_kheads=16,
                embed=embed)
    cfg = board_config(DRAM_BYTES=1 << 33)
    img = spec.image(cfg, 4096, wformat="fp4", head_format="int8", lookup=True)
    assert img.grouped
    buckets = (1, 2, 4, 8, 16)
    if embed == "f32":
        with pytest.raises(ValueError, match="does not fit 32 bits"):
            img.compile_decode(1, 3)
        cfg = board_config(DRAM_BYTES=1 << 33, AW=36)
        img = spec.image(cfg, 4096, wformat="fp4", head_format="int8", lookup=True)
        assert img.grouped and img.high
        buckets = (1, 2, 4)
        with pytest.raises(CompileError, match="out of (address )?registers"):
            img.compile_decode(16, 15 * ATTN_BLOCK + 3)
    for b in buckets:
        (prog,), _ = img.compile_decode(b, (b - 1) * ATTN_BLOCK + 3)
        assert len(prog) <= cfg.IMEM_WORDS // 8, b


def test_4b_on_one_slice_with_its_own_stream_engine_decodes_unrolled():
    """bcu1525-wg's arch (one slice, the streams on an engine of their own: Config.SE_OWN): Qwen3.5-
    4B's 16 pairs of DeltaNet heads are array-major (qwen35.pair_loop: PAIR_LOOP_OWN), so its DSTEP
    decode loads every pair's taps and window ahead (cv_ahead), and the resident decode and the
    decode loop on the device fit IMEM at every bucket of a 4K context. The same arch without
    SE_OWN keeps the group-major blocks, and the image cache keys the two layouts apart (SE_OWN
    is a program-only field). A layout-only image, no weights."""
    from opentpu import imgcache as IC
    from opentpu.llm.qwen35 import PAIR_LOOP_OWN, pair_loop
    kinds = ("linear", "linear", "linear", "attn") * 8
    spec = Spec(2560, kinds, 16, 4, 256, 64, 32, 128, 128, 9216, 248320, lin_kheads=16,
                embed="int8")
    cfg = T.config("bcu1525-wg", DRAM_BYTES=1 << 33)
    assert cfg.SE_OWN and cfg.S == 1 and pair_loop(cfg) == PAIR_LOOP_OWN == 16
    img = spec.image(cfg, 4096, wformat="fp4", head_format="int8", lookup=True)
    assert not img.grouped and img.cv_ahead
    for b in (1, 2, 4, 8, 16):
        (prog,), _ = img.compile_decode(b, (b - 1) * ATTN_BLOCK + 3)
        assert len(prog) <= cfg.IMEM_WORDS // 8, b
    gen = img.compile_generate(8, 7 * ATTN_BLOCK)
    assert all(len(p) <= cfg.IMEM_WORDS // 8 for p in (gen[0] if isinstance(gen, tuple) else gen))
    vpu = dataclasses.replace(cfg, SE_OWN=False)
    img0 = spec.image(vpu, 4096, wformat="fp4", head_format="int8", lookup=True)
    assert img0.grouped
    assert IC.key(img, (spec, cfg, 4096), "ck") != IC.key(img0, (spec, vpu, 4096), "ck")


def test_27b_on_one_slice_projects_its_key_heads_once():
    """bcu1525-wg's arch: the 27B's 24 pairs of DeltaNet heads (48 value heads, 16 key heads: 3
    a key head) decode unrolled (qwen35.pair_loop: PAIR_LOOP_QK, past PAIR_LOOP_OWN's 16) with
    each key head's q and k projected once (qk_once), the taps and windows not ahead (cv_ahead:
    with qk_once its tiles exhausted TMEM). Its decode at buckets 1 and 16, its tile runs of 4
    rows (the MLP's K split) and its prompt runs of 2 at bucket 16 fit IMEM, the weights from 4
    GiB on. A layout-only image, no weights."""
    from opentpu.llm import tiled as TL
    from opentpu.llm.qwen35 import PAIR_LOOP_OWN, PAIR_LOOP_QK, pair_loop
    kinds = ("linear", "linear", "linear", "attn") * 16
    spec = Spec(5120, kinds, 24, 4, 256, 64, 48, 128, 128, 17408, 248320, lin_kheads=16,
                embed="int8")
    cfg = T.config("bcu1525-wg")
    assert pair_loop(cfg, qk=True) == PAIR_LOOP_QK == 24 > PAIR_LOOP_OWN
    img = spec.image(cfg, 4096, 1, 16, wformat="fp4", head_format="int8", lookup=True)
    assert img.high and not img.grouped and img.qkonce and not img.cv_ahead and not img.cv_pre
    n = cfg.IMEM_WORDS // 8
    for b in (1, 16):
        (prog,), _ = img.compile_decode(b, (b - 1) * ATTN_BLOCK + 3)
        assert len(prog) <= n, b
    for kind in ("P", "L"):
        progs, _ = TL.compile_tile_run(img, 16, 4, kind)
        assert max(map(len, progs)) <= n, kind
        progs, _ = img.compile_prompt_run(16, 2, kind)
        assert max(map(len, progs)) <= n, kind
    four = T.config("bcu1525-x2")                     # (4 slices: 6 pairs a slice, as before)
    img4 = spec.image(four, 4096, 1, 16, wformat="fp4", head_format="int8", lookup=True)
    assert not img4.grouped and img4.qkonce and img4.cv_ahead


@pytest.mark.parametrize("S", [1, 2])
def test_cv_pre_is_bit_exact(tiny, S):
    """The DSTEP decode at a run-time position with each DeltaNet layer's taps and windows loaded
    beside the MLP before it, and its windows stored beside its own MLP (Spec.cv_pre: qwen35
    ._CvPipe; the tiny model's (lin, lin, attn) loop, whose unit's second DeltaNet layer loads
    the next iteration's first): the logits and every layer block bit for bit those of the
    loads and stores in the mixer, resident decode after a chunked prefill. The program is the
    mixer's one plus the first run's loads, and the mixer holds no LD of them."""
    from opentpu import isa as I
    W, spec = tiny[1:]
    cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=True, PAIR=True, S=S)
    a, b = (Engine(dataclasses.replace(spec, cv_ahead=True, cv_pre=on), W, cap=512, cfg=cfg,
                   resident=True) for on in (False, True))
    assert not a.image.cv_pre and b.image.cv_pre and a.image.cv_ahead and not a.image.grouped
    ra, rb = (e.image.compile_decode(1, 3)[0][0] for e in (a, b))
    npairs = spec.lin_heads // S // 2
    assert len(rb) == len(ra) + npairs
    lds = [i for i in rb if i.op == I.LD]
    assert len(lds) == len([i for i in ra if i.op == I.LD]) + npairs
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 10)]
    assert np.array_equal(a.prefill(toks[:4], chunk=2).view(np.uint32),
                          b.prefill(toks[:4], chunk=2).view(np.uint32))
    for t in toks[4:]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32)), a.pos
    assert sorted(a._decodes) == sorted(b._decodes) == [1]
    assert all(np.array_equal(x, y) for x, y in zip(_layers_dram(a), _layers_dram(b)))


def _tiny16():
    """16 DeltaNet value heads, an LM head of its own (1.25 x the embedding: a tied tiny head
    mostly gives back its input token, which would hide a broken loop) and the int8 embedding,
    the taps and windows a layer ahead (cv_pre)."""
    _, W, spec = _tiny_model(8, nv=16, init=0.2)
    W = dict(W, **{"lm_head.weight": W["model.embed_tokens.weight"] * np.float32(1.25)})
    return dataclasses.replace(spec, tied=False, embed="int8", cv_ahead=True, cv_pre=True), W


def _run_dram(eng):
    """Each slice's DRAM a run may write: all of it, or below HIGH (Image.high's low_end: the
    slices refuse writes past the low area, wlim)."""
    end = eng.image.low_end if getattr(eng.image, "high", False) else None
    return [s.dram[:end] for s in eng.backend.machine.slices]


def _token_loop(p):
    """(the generate program's token LOOP, its body's end): the first LOOP, nothing before it
    loops."""
    from opentpu import isa as I
    lo = next(i for i, x in enumerate(p) if x.op == I.LOOP)
    return lo, lo + 1 + p[lo].w[0]


@pytest.mark.parametrize("target,split", [("bcu1525-wg", None), ("bcu1525-s2", None),
                                          ("bcu1525-wg", True)])
def test_generate_ahead_gives_the_steps_tokens(target, split, monkeypatch):
    """The generate loop's ahead (generate._generate, qwen35._loop_ahead) on the BCU's arch (8
    MXU columns: the gather's one-hot operand made in TMEM; DSTEP; cv_pre), one slice and two:
    the operand built once before the token loop and held in ACT RAM, and the first DeltaNet
    layer's taps and windows loaded before the loop for its first token and in each token's LM
    head for the next. Against the loop without it, the body loses the operand's two FILLs and
    QACT and keeps its loads, the pipe's now between the head's operand and its MM. Two runs'
    tokens (the second continuing the first) are the host's resident steps', and after them the
    DRAM is the steps' bit for bit but for their logits and the loop's state block and out[],
    and the loop's without the ahead bit for bit (bcu1525-s2: 16 GiB a slice, an Image.high's
    low area): each run's last token loaded the taps and windows for no token, and nothing was
    written from them. split: the split form (two programs a token, no loop) has no ahead, and
    gives the same tokens."""
    from opentpu import isa as I
    from opentpu.llm import generate as G
    from opentpu.llm import qwen35 as Q35
    spec, W = _tiny16()
    cfg = T.config(target) if target == "bcu1525-s2" else T.config(target, DRAM_BYTES=1 << 27)
    a, b, c = (Engine(spec, W, cap=512, cfg=cfg, resident=True) for _ in range(3))
    a.gen_split = split
    assert a.image.cv_pre and a.can_generate
    np_ = spec.lin_heads // cfg.S // 2
    if not split:
        on = a.image.compile_generate(1, 3, chain=False)
        with monkeypatch.context() as mp:
            mp.delattr(Q35.qwen35_step, "ahead")
            off = a.image.compile_generate(1, 3, chain=False)
        for p, q in zip(on, off):
            (l1, e1), (l0, e0) = _token_loop(p), _token_loop(q)
            assert [x.op for x in p[:l1]].count(I.QACT) == 1 and \
                [x.op for x in q[:l0]].count(I.QACT) == 0
            assert [x.op for x in p[:l1]].count(I.LD) == [x.op for x in q[:l0]].count(I.LD) + np_
            body1, body0 = p[l1 + 1:e1], q[l0 + 1:e0]
            assert len(body1) == len(body0) - 3 and len(p) - e1 == len(q) - e0
            ops = lambda body, op: sum(x.op == op for x in body)
            assert ops(body1, I.LD) == ops(body0, I.LD) and ops(body1, I.QACT) == \
                ops(body0, I.QACT) - 1 and ops(body1, I.VOP) == ops(body0, I.VOP) - 2
            head = max(i for i, x in enumerate(body1) if x.op == I.MM)   # (one chunk: 1000)
            xs = max(i for i in range(head) if body1[i].op == I.QACT)   # (the final norm's)
            assert sum(x.op == I.LD for x in body1[xs:head]) == np_
            assert any(x.op == I.LD for x in body0[:body0.index(next(
                x for x in body0 if x.op == I.MM))])            # (the pipe's prime, each token)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 20)]
    t0 = int(np.argmax(a.prefill(toks)))
    assert int(np.argmax(b.prefill(toks))) == t0
    ref, t = [], t0
    for _ in range(16):
        t = int(np.argmax(b.step(t)))
        ref.append(t)
    assert len(set(ref)) > 8                        # the tokens vary
    assert a.generate_card(t0, 10, stop_ids=[]) == ref[:10]
    assert a.generate_card(ref[9], 6, stop_ids=[]) == ref[10:16]
    assert a.pos == b.pos == 20 + 16
    assert all(isinstance(p, tuple) == bool(split) for p in a._gens.values())
    g, lg = a.image.lookup["gen"], a.image.io["logits"]
    ours = [(lg, lg + 4 * spec.vocab)] + ([(g["state"], a.image.nbytes)] if split else [
        (g["state"], g["state"] + 4 * G.STATE_WORDS), (g["out"], g["out"] + 4 * a.cap)])
    for x, y in zip(_run_dram(a), _run_dram(b)):    # (the steps' logits; the loop's state and
        d = np.flatnonzero(x != y)                  # out[], the split form's generate area)
        assert all(any(u <= i < v for u, v in ours) for i in d)
    if split:
        return
    with monkeypatch.context() as mp:
        mp.delattr(Q35.qwen35_step, "ahead")
        c.prefill(toks)
        assert c.generate_card(t0, 10, stop_ids=[]) + c.generate_card(ref[9], 6, stop_ids=[]) \
            == ref[:16]
    assert all(np.array_equal(x, y) for x, y in zip(_run_dram(a), _run_dram(c)))


@pytest.mark.parametrize("S", [1, 2])
def test_shared_dstep_pair_is_bit_exact(tiny, S):
    """A pair of value heads on one key head (kh4, as Qwen3.5-4B's 32 on 16) in one DSTEP with
    Config.DSTEP2 (qwen35's dstep_sh; pair 0 a DSTEP a head, head b's prep beside head a's
    step): the logits and every layer block bit for bit those of a DSTEP a head, per position
    and resident (cv_ahead; resident with cv_pre). kh8's pairs are on two key heads: a DSTEP
    a head either way."""
    from opentpu import isa as I
    W, spec = tiny[1:]
    cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=True, PAIR=True, S=S)
    engs = [Engine(dataclasses.replace(spec, cv_ahead=True), W, cap=512,
                   cfg=dataclasses.replace(cfg, DSTEP2=on), resident=res)
            for on in (False, True) for res in (False, True)]
    np_, sh = spec.lin_heads // S // 2, spec.lin_heads == 2 * spec.lin_kheads
    ndn = {}                            # (the programs' DeltaNet layers: the loop's body)
    for k, e in enumerate(engs):
        prog = e.image.compile_decode(1, 3)[0][0] if e.resident else e.image.compile_step(3)[0]
        ds = [i for i in prog if i.op == I.DSTEP]
        pairs = sum(bool(i.flags & I.F_DPAIR) for i in ds)
        n = ndn.setdefault(e.resident, len(ds) // (2 * np_))
        assert pairs == (n * (np_ - 1) if sh and k >= 2 else 0)
        assert len(ds) == n * 2 * np_ - pairs and n > 0
    toks = [int(t) for t in np.random.default_rng(6).integers(0, 1000, 6)]
    for t in toks:
        got = [e.step(t).view(np.uint32) for e in engs]
        assert all(np.array_equal(got[0], g) for g in got[1:]), engs[0].pos
    assert all(np.array_equal(a, b) for k in (0, 1)     # (per position, resident: other images)
               for a, b in zip(_layers_dram(engs[k]), _layers_dram(engs[k + 2])))


def test_27b_fits_four_slices():
    """Qwen3.5-27B's dims on bcu1525-x2's four slices (48 DeltaNet value heads for 16 key
    heads, 12 a slice: 6 pairs, array-major, q and k once a layer for its 4 key heads; DSTEP: on
    the VPU path a head pair's states and the all-reduce's 4 x 5120 words exceed TMEM; 128 ACT
    RAM blocks: the widest MLP operand is 4 down chunks of 256, not F; W_down in parts of 1024,
    the all-gather's order, a scale row each where 4 parts of 256 had 4), fp4 layers with an int8
    LM head, the KV cache at 4096 and the int8 embedding table split by columns over the
    slices. On the target as it is (targets/bcu1525-x2.toml: 16 GiB a slice, 36-bit addresses,
    the bank-half window) the image fits a slice with its DeltaNet mixers' weights in the
    window's upper half and the other weights past it, and holds 3.9 GiB: what the same arch
    lays out compactly in a slice of 4 GiB with 32-bit addresses and no window, where the
    parts sum to the image's end. There the repeated q and k rows would cost 1024 rows a layer
    and slice more than q and k once, and the image would pass 4 GiB. The resident decode fits
    IMEM at every bucket, at 1 column and WG 1 too. A layout-only image, no weights."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import image_fit
    from opentpu.llm.qwen3 import HIGH
    kinds = ("linear", "linear", "linear", "attn") * 16
    spec = Spec(5120, kinds, 24, 4, 256, 64, 48, 128, 128, 17408, 248320, tied=False,
                lin_kheads=16, embed="int8")
    real = T.config("bcu1525-x2")
    assert (real.S, real.MCOLS, real.WG, real.DSTEP) == (4, 8, 8, True)
    assert (real.AW, real.BHW, real.DRAM_BYTES) == (36, 33, 16 << 30)
    cfg = T.config("bcu1525-x2", AW=32, BHW=0, DRAM_BYTES=1 << 32)    # the compact layout
    img = spec.image(cfg, 4096, wformat="fp4", head_format="int8", lookup=True)
    assert not img.grouped and not img.shared and img.qkonce and img.lookup["split"] == 4
    assert img.nbytes <= cfg.DRAM_BYTES and not img.embed_host and not img.high
    p = image_fit.parts(img)
    assert sum(p.values()) == img.nbytes and p["qk-rep"] == 0
    assert p["embed"] == 248320 * (1280 + 40)         # a slice's columns, their gather scales
    wide = spec.image(real, 4096, wformat="fp4", head_format="int8", lookup=True)
    assert wide.high and wide.mix and (wide.wlo, wide.layer0) == (HIGH, 2 * HIGH)
    assert wide.low_end < HIGH < wide.wskip[0] < wide.wskip[1] == 2 * HIGH < wide.nbytes
    assert wide.nbytes <= real.DRAM_BYTES and wide.qkonce and not wide.embed_host
    assert image_fit.parts(wide) == p                 # the window moves the parts, adds none
    assert image_fit.used_bytes(wide) == img.nbytes < wide.nbytes
    big = dataclasses.replace(cfg, DRAM_BYTES=1 << 40)
    rep = dataclasses.replace(spec, qk_once=False).image(big, 4096, wformat="fp4",
                                                         head_format="int8", lookup=True)
    pr = image_fit.parts(rep)        # the repeated q / k rows: 2048 a layer and slice
    r = 2560 + 164                   # a row's fp4 bytes and 1/16 of its group's scale area
    assert pr["qk-rep"] == 48 * 2048 * r and rep.nbytes > cfg.DRAM_BYTES
    assert rep.nbytes - img.nbytes > pr["qk-rep"] - 48 * 1024 * r
    for c in (dataclasses.replace(cfg, MCOLS=1, WG=1), cfg, real):
        im = spec.image(c, 4096, wformat="fp4", head_format="int8", lookup=True)
        p1 = image_fit.parts(im)
        assert sum(p1.values()) == image_fit.used_bytes(im) and im.nbytes <= c.DRAM_BYTES
        assert p1["io"] > 0 and (im.high or image_fit.used_bytes(im) == im.nbytes)
        for b in (1, 8, 16):
            progs, _ = im.compile_decode(b, (b - 1) * ATTN_BLOCK + 3)
            assert len(progs) == 4 and max(len(q) for q in progs) <= c.IMEM_WORDS // 8, b


@pytest.mark.parametrize("fold", [False, True])
def test_27b_layer_loop_rtl_has_no_loop_carried_waits(have_verilator, fold):
    """Qwen3.5-27B's dims, its first 4 layers (DeltaNet x 3, attention) and a 2,048-word
    vocabulary, decoding at 128 on the WG targets' Config and RTL
    (targets/bcu1525-x2.toml's micro-architecture and channel model at 133.33 MHz, zero images),
    unfolded and with Spec.dn_fold: the layer loop runs the 3 DeltaNet layers, and no TMEM
    writer of its body waits on the previous iteration's instructions (Profile.loop_waits).
    With a plain load of the input norm's gain the folded program's g_in LD, at the body's head,
    waited 4,132 cycles at iterations 2 and 3 on the MLP's residual add at the body's end (the
    down output placed in the gain's words; docs/se_engine.md); load_home keeps the
    gain's tile for the program. A layout-only image, no weights."""
    from opentpu import isa as I, rtlsim
    from opentpu.profile import parse
    spec = Spec(5120, ("linear",) * 3 + ("attn",), 24, 4, 256, 64, 48, 128, 128, 17408, 2048,
                tied=False, lin_kheads=16, dn_fold=fold)
    n = spec.image(T.config("bcu1525-x2", BHW=0), 256, wformat="fp4", head_format="int8").nbytes
    cfg = T.config("bcu1525-x2", BHW=0, DRAM_BYTES=1 << max(24, (n - 1).bit_length() + 1))
    img = spec.image(cfg, 256, wformat="fp4", head_format="int8")
    progs = img.compile_step(128)
    assert [ins.w[1] for ins in progs[0] if ins.op == I.LOOP] == [3]
    _, _, st = rtlsim.run(cfg, progs, [np.zeros(img.nbytes, np.uint8)] * cfg.S,
                          uarch=T.sim_uarch("bcu1525-x2"), axi=True, boot=True, stall=0,
                          max_cycles=1 << 30, trace=True,
                          plusargs=T.mem_model("bcu1525-x2", 133.33))
    w = parse(st["trace"], cfg, progs).loop_waits(0)
    assert w["unresolved"] == 0 and [x["iterations"] for x in w["loops"]] == [3]
    assert w["wait_all"] == 0, w["loops"]

def test_tiny_qwen35_on_a_board_without_dstep(tiny, have_verilator):
    """A bitstream built with DSTEP=0 (the DMA's DSTEP datapath left out) reports CAPS bit6 = 0;
    the host configuration then has no DSTEP and the model runs on VOPs, bit-identical."""
    from opentpu.host.board import Board, BoardBackend, SimTransport, device_config
    _, W, spec = tiny
    tr = SimTransport(ch_bytes=(1 << 24) // 2, stall=20, seed=5, params={"DSTEP": 0})
    info = Board(tr).info()
    assert not info["caps"]["dstep"] and info["caps"]["pair"]
    cfg = dataclasses.replace(device_config(info), DRAM_BYTES=1 << 24)
    assert not cfg.DSTEP
    isa = Engine(spec, W, cap=256, cfg=cfg)
    brd = Engine(spec, W, cap=256, cfg=cfg,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    for tok in (11, 222, 333):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.skipif(not REAL.exists(), reason="models/Qwen3.5-0.8B not downloaded")
def test_qwen35_0_8b_program_fits_board_imem():
    """The hardware loops keep the longest program (last position of a 4K context) in IMEM,
    with the query groups split for 2 MXU columns (the default board) or not (4)."""
    spec = load_spec(REAL)
    for mcols in (2, 4):
        cfg = board_config(MCOLS=mcols)
        assert len(spec.image(cfg, 4096).compile_step(4095)[0]) <= cfg.IMEM_WORDS // 8


@pytest.mark.skipif(not REAL.exists(), reason="models/Qwen3.5-0.8B not downloaded")
def test_qwen35_0_8b_greedy_matches_hf():
    tok = transformers.AutoTokenizer.from_pretrained(REAL)
    ids = _chat_ids(tok, "What is the capital of France? Answer in one sentence.")
    hf = transformers.AutoModelForCausalLM.from_pretrained(REAL, dtype=torch.float32).eval()
    with torch.no_grad():
        want = hf.generate(torch.tensor([ids]), max_new_tokens=8, do_sample=False)[0, len(ids):]
    eng = Engine(load_spec(REAL), load_weights(REAL), cap=256)
    got = eng.generate(ids, max_new=8)
    assert got == want.tolist()[:len(got)] and len(got) >= 7
    assert tok.decode(got).startswith("The capital of France is Paris")


@pytest.mark.skipif(not REAL.exists(), reason="models/Qwen3.5-0.8B not downloaded")
@pytest.mark.parametrize("dstep", [False, True])
def test_qwen35_0_8b_token_on_rtl_is_bit_exact(have_verilator, dstep):
    """Feed part of a prompt on the ISA simulator (board configuration), then run the next
    token on the Verilator RTL and on the ISA simulator from the same DRAM state: weights, KV
    cache, convolution ring, DeltaNet state and logits must agree bit for bit (with and
    without DSTEP)."""
    from opentpu.llm.rtl_backend import RtlBackend
    spec = load_spec(REAL)
    eng = Engine(spec, load_weights(REAL), cap=256,
                 cfg=board_config(DRAM_BYTES=1 << 30, DSTEP=dstep))
    prompt = [760, 6511, 314, 9338, 369]            # "The capital of France is"
    for t in prompt[:-1]:
        eng.step(t)
    n = eng.image.nbytes
    rtl = RtlBackend(eng.cfg, [s.dram[:n] for s in eng.backend.machine.slices])
    isa = eng.backend
    want = eng.step(prompt[-1])
    eng.backend, eng.pos = rtl, eng.pos - 1
    got = eng.step(prompt[-1])
    assert np.array_equal(want.view(np.uint32), got.view(np.uint32))
    for s in range(eng.cfg.S):
        assert np.array_equal(isa.machine.slices[s].dram[:n], rtl.drams[s][:n])


def test_tiny_vt_tiles_bit_exact(tiny, monkeypatch):
    """A 512-token cache holds V^T in tiles of 256 tokens (compiler.KVDesc): a prefill chunk
    across the tile edge, then decode past it, give the same logits bit for bit as the plain
    [d, cap] layout."""
    import opentpu.compiler as C
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 264)]

    def run():
        eng = Engine(spec, W, cap=512)
        eng.step(toks[0])
        out = [eng.prefill(toks[1:257], chunk=8)]          # rows 249..256 cross the edge
        out += [eng.step(t) for t in toks[257:]]
        return out
    assert C.vt_tile(512) == 256
    tiled = run()
    monkeypatch.setattr(C, "VT_TILE", 1 << 30)
    plain = run()
    assert all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(tiled, plain))


def test_tiny_prefill_rows_dstep_on_rtl(tiny, have_verilator):
    """A 6-row prefill run with DSTEP (a DSTEP per row and head) on the RTL's board memory
    path: DRAM equals the ISA simulator's, and the logits equal token-by-token decode's."""
    _rows_dstep_on_rtl(*tiny[1:])


def test_prefill_rows_pipeline_on_rtl(have_verilator):
    """The same with 16 DeltaNet heads (8 pairs): _rows_pipelined's hardware loop over the
    middle pairs on the RTL."""
    _rows_dstep_on_rtl(*_tiny_model(16, 16)[1:], dram=1 << 25)


def test_tiny_prefill_rows_st16_on_rtl(tiny, have_verilator):
    """The same with the DeltaNet states in 16 bits (Config.ST16 e6m9: each row's DSTEPs with
    flag ST16, otpu_dma ST16); the logits those of token-by-token DSTEP decode on 16-bit
    states."""
    _rows_dstep_on_rtl(*tiny[1:], st16="e6m9")


def _rows_dstep_on_rtl(W, spec, dram=1 << 24, st16=""):
    from opentpu import isa as I
    from opentpu import rtlsim
    from opentpu.isasim import Machine
    from opentpu.llm.qwen3 import rope_tables
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 9)]
    cfg = board_config(DRAM_BYTES=dram, DSTEP=True, STREAM=bool(st16), ST16=st16)
    want = Engine(spec, W, cap=256, cfg=cfg if st16 else board_config(DRAM_BYTES=dram)
                  ).prefill(toks, chunk=1)
    eng = Engine(spec, W, cap=256, cfg=cfg)
    eng.prefill(toks[:3])                         # a state, ring and cache to continue from
    rows = [(0, 3 + j) for j in range(6)]
    progs = eng.image.compile_rows(rows, [5])
    assert sum(i.op == 0x12 for i in progs[0]) >= 6
    assert all(bool(i.flags & I.F_DST16) == bool(st16) for i in progs[0] if i.op == I.DSTEP)
    io = eng.image.io
    cs = [np.stack(a).astype(np.float32) for a in zip(*[rope_tables(spec, p) for _, p in rows])]
    dram = eng.backend.machine.slices[0].dram.copy()
    for k, v in (("x", eng.embed[toks[3:]]), ("cos", cs[0]), ("sin", cs[1])):
        b = np.ascontiguousarray(v, np.float32).view(np.uint8).reshape(-1)
        dram[io[k]:io[k] + b.size] = b
    m = Machine(cfg, progs, [dram.copy()]).run()
    drams, _, _ = rtlsim.run(cfg, progs, [dram.copy()], uarch=rtlsim.BOARD_UARCH, axi=True,
                             boot=True)
    n = eng.image.nbytes
    assert np.array_equal(drams[0][:n], m.slices[0].dram[:n])
    v = spec.vocab
    assert np.array_equal(drams[0][io["logits"] + 4 * 5 * v:io["logits"] + 4 * 6 * v]
                          .view(np.float32), want)


def test_tiny_image_from_the_quantization_cache(tiny, tmp_path, monkeypatch):
    """Image.build through the 4-bit quantization cache (opentpu/qcache.py) gives the uncached
    image, byte for byte, and a second build quantizes nothing."""
    from opentpu import qcache
    _, W, spec = tiny
    eng = Engine(spec, W, cap=256, wformat="fp4", head_format="int8")
    want = eng.image.build(W)
    monkeypatch.setenv("OTPU_IMAGE_CACHE", str(tmp_path))
    monkeypatch.setattr(qcache, "MIN_ELEMS", 0)
    monkeypatch.setattr(qcache, "FREE_FLOOR", 0)
    monkeypatch.setattr(qcache, "stats", dict.fromkeys(qcache.stats, 0))
    first = eng.image.build(W)
    n = dict(qcache.stats)
    again = eng.image.build(W)
    assert n["miss"] > 0 and n["write"] == n["miss"]
    assert qcache.stats["miss"] == n["miss"] and qcache.stats["hit"] == 2 * n["hit"] + n["miss"]
    for got in (first, again):
        assert all(np.array_equal(a, b) for a, b in zip(got, want))
