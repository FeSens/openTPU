"""Gemma 4 26B-A4B's parts of opentpu/llm/gemma4.py against Hugging Face transformers, on tiny
random models: global layers with their own KV head count and K = V (attention_k_eq_v), no
per-layer inputs (PLE), a dense MLP width that is not a multiple of 2D (the image pads it),
and the MoE block beside the MLP (enable_moe_block)."""
import json

import numpy as np
import pytest

import opentpu.llm.gemma4 as G
from opentpu.llm.qwen3 import Engine

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.gemma4")

S_, F_ = "sliding_attention", "full_attention"
KINDS = (S_, S_, F_, S_, S_, F_)


def _model(tmp_path_factory, name, **kw):
    """head_dim 128 (sliding) and 256 (global, 32 rotated pairs), 4 query heads on 2 KV heads
    (sliding) and on 1 (global, K = V), the MLP 320 wide, no PLE; norms and the router's scales
    random, k_norm near the real model's 0.13 (test_gemma4's tiny)."""
    torch.manual_seed(0)
    hc = transformers.Gemma4TextConfig(
        hidden_size=256, num_hidden_layers=len(KINDS), num_attention_heads=4,
        num_key_value_heads=2, num_global_key_value_heads=1, attention_k_eq_v=True,
        head_dim=128, global_head_dim=256, intermediate_size=320, vocab_size=1000,
        hidden_size_per_layer_input=0, layer_types=list(KINDS), num_kv_shared_layers=0,
        sliding_window=512, final_logit_softcapping=30.0, max_position_embeddings=4096, **kw)
    hc._attn_implementation = "eager"
    m = transformers.models.gemma4.modeling_gemma4.Gemma4ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n or n.endswith("router.scale"):
                k = "k_norm" in n
                p.copy_((0.13 if k else 1.0) + (0.01 if k else 0.1) * torch.randn_like(p))
            elif n.endswith("per_expert_scale"):
                p.copy_(0.5 + torch.rand_like(p))
            elif "experts" in n or "router" in n:   # (a router decisive enough that the
                p.copy_((0.5 if "router.proj" in n else 0.05) * torch.randn_like(p))  # int8 one
                # mostly picks the same experts: its logits ~1 apart, not ~0.1)
        for n, b in m.named_buffers():
            if n.endswith("layer_scalar"):
                b.copy_(0.5 + torch.rand_like(b))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    d = tmp_path_factory.mktemp(name)
    (d / "config.json").write_text(json.dumps(hc.to_dict()))
    return m, W, G.Spec.from_hf(d)


@pytest.fixture(scope="module")
def kv(tmp_path_factory):
    """No MoE: the attention, PLE and MLP-width parts alone."""
    return _model(tmp_path_factory, "kv")


@pytest.fixture(scope="module")
def moe(tmp_path_factory):
    """8 experts 96 wide, top 2, beside the 320-wide MLP."""
    return _model(tmp_path_factory, "moe", enable_moe_block=True, num_experts=8,
                  top_k_experts=2, moe_intermediate_size=96)


def test_spec(kv, moe):
    _, _, s = kv
    assert (s.n_kv, s.n_kv_global, s.k_eq_v, s.ple_dim, s.experts) == (2, 1, True, 0, 0)
    assert [s.kvh(i) for i in range(s.layers)] == [2, 2, 1, 2, 2, 1]
    assert [s.kv_same(i) for i in range(s.layers)] == [False, False, True] * 2
    assert s.kv_src == tuple(range(6)) and s.ffn == (320,) * 6
    _, _, s = moe
    assert (s.experts, s.top_k, s.expert_ffn) == (8, 2, 96)


@pytest.mark.parametrize("name", ["kv", "moe"])
def test_reference_matches_hf(name, request):
    m, W, spec = request.getfixturevalue(name)
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 32)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(G.softcap(spec, G.reference_logits(spec, W, toks)) - hf).max() < 1e-4


def _cfg():
    from opentpu.isasim import board_config
    return board_config(DRAM_BYTES=1 << 26)


def _engine(kv, **kw):
    _, W, spec = kv
    return Engine(spec, W, cap=1024, cfg=_cfg(), **kw)


def _cos(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


@pytest.mark.parametrize("wf", ["int8", "fp4"])
def test_image_pads_the_mlp(kv, wf):
    """The 320-wide MLP in blocks of 384 (int8 down) or 512 (4-bit) rows / columns, the extra
    gate / up rows zero; no PLE areas, no V projection in the global (K = V) layers, 2 KV heads
    in the sliding layers' caches and 1 in the global layers'."""
    _, W, spec = kv
    e = _engine(kv, wformat=wf)
    img, F = e.image, {"int8": 384, "fp4": 512}[wf]
    assert img.wproj is None and img.ple_rec == 0 and not img.ple_host
    for i in range(spec.layers):
        o = img.bofs[img._key(i)]
        assert o["mats"]["wg"][0] == F and len(o["wd"]) * img.dchunk[320, wf] == F
        assert ("wv" in o["mats"]) == (spec.kinds[i] == G.SLIDE) and len(o["kv"]) == spec.kvh(i)
        assert "wpg" not in o["mats"] and "g_ple" not in o
    o, base = img.bofs[img._key(0)], img._off(0).static()
    rb = {"int8": 256, "fp4": 128}[wf]
    dram = e.backend.machine.slices[0].dram
    assert not dram[base + o["wg"][0] + 320 * rb:base + o["wg"][0] + F * rb].any()


@pytest.mark.parametrize("wf", ["int8", "fp4"])
def test_device_matches_hf_and_emulation(kv, wf):
    """The device against the emulation (its quantization points) in both formats, and int8
    against HF (fp4 on random weights is far from it: cosine ~0.7)."""
    m, W, spec = kv
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 24)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    e = _engine(kv, wformat=wf)
    dev = np.array([G.softcap(spec, e.step(t)) for t in toks])
    emu = G.softcap(spec, G.emulated_logits(spec, W, toks[:12], wformat=wf))
    assert _cos(dev[:12], emu).min() > 0.99
    if wf == "int8":
        assert _cos(dev, hf).min() > 0.985


@pytest.mark.parametrize("wf", ["int8", "fp4"])
def test_resident_is_bit_exact(kv, wf):
    """Resident decode (embedding gathered on the device, run-time masks) against the
    per-position programs, across the first bucket boundary."""
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 300)]
    a, b = _engine(kv, wformat=wf, resident=True), _engine(kv, wformat=wf)
    assert a.resident and not b.resident
    assert np.array_equal(a.prefill(toks[:250]), b.prefill(toks[:250]))
    for t in toks[250:]:
        assert np.array_equal(a.step(t), b.step(t))


def test_sliding_k_rows_at_their_own_stride(kv):
    """Spec.k_rows (from_hf: MoE models): the sliding layers' K rows at a row and its scales
    (256 bytes here, not the RoPE row's 768): the same logits bit for bit, resident and
    per-position, past the ring's wrap, and a smaller image."""
    _, W, spec = kv
    toks = [int(t) for t in np.random.default_rng(7).integers(0, 1000, 700)]
    sk = G.replace(spec, k_rows=True)
    a = Engine(sk, W, cap=1024, cfg=_cfg(), resident=True)
    b = Engine(spec, W, cap=1024, cfg=_cfg(), resident=True)
    c = Engine(sk, W, cap=1024, cfg=_cfg())
    assert a.image.ks[G.SLIDE] == 256 and b.image.ks[G.SLIDE] == a.image.ps == 768
    assert a.image.nbytes < b.image.nbytes
    la, lb, lc = (e.prefill(toks[:680]) for e in (a, b, c))
    assert np.array_equal(la, lb) and np.array_equal(la, lc)
    for t in toks[680:]:
        ra = a.step(t)
        assert np.array_equal(ra, b.step(t)) and np.array_equal(ra, c.step(t))


def test_chunked_prefill_is_bit_exact(kv):
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 11)]
    a, b = _engine(kv), _engine(kv)
    assert np.array_equal(a.prefill(toks, chunk=4), b.prefill(toks, chunk=1))
    img = a.image
    assert np.array_equal(a.backend.machine.slices[0].dram[img.layer0:img.head[0]],
                          b.backend.machine.slices[0].dram[img.layer0:img.head[0]])


K = 2                   # the tiny MoE's top k: the smallest expert cache


def _moe_engine(moe, **kw):
    _, W, spec = moe
    return Engine(spec, W, cap=1024, cfg=_cfg(), rows=1, **kw)


@pytest.mark.parametrize("wf", ["int8", "fp4"])
def test_moe_device_follows_the_emulation(moe, wf):
    """The MoE block on the device (moe.moe_ffn: the router and every expert on the card, the
    dense MLP emitted beside the request, the two norms combined) follows the emulation of the
    same quantization given the card's choice of experts (the emulation's own differs where
    the k-th and the next logit are near a tie: the device's attention differs from the
    emulation's by ~1e-3 from the second position on, as in every Gemma 4 model): the first
    token to the VPU's EXP2 / RECIP (the routing weights' softmax), the others to cosine >
    0.99. The card's experts in the first layer,
    whose input no routing has touched, are the emulation's wherever its k-th logit is clear
    of the next; int8 follows HF."""
    m, W, spec = moe
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 12)]
    eng = _moe_engine(moe, wformat=wf)
    assert (eng.image.fmt.F0, eng.image.fmt.F) == (96, 128 if wf == "int8" else 256)
    eng.server.history = []
    dev = np.array([eng.step(t) for t in toks])
    eng.server.poll()                   # (the last layer's request, served at a next fence)
    L, E, card = spec.layers, spec.experts, eng.server.history
    assert len(card) == len(toks) * L and eng.server.misses == 0
    assert all(g // E == j % L for j, r in enumerate(card) for g in r)
    routes = []
    G.emulated_logits(spec, W, toks, wformat=wf, routes=routes)
    clear = [(t, ids) for t, layer, ids, margin in routes if layer == 0 and margin > 0.05]
    assert len(clear) >= len(toks) // 2
    assert all(sorted(card[t * L]) == sorted(ids) for t, ids in clear)
    emu = G.emulated_logits(spec, W, toks, wformat=wf, routing={
        (j // L, j % L): [g % E for g in r] for j, r in enumerate(card)})
    c = _cos(dev, emu)
    assert c[0] > 1 - 1e-4 and c.min() > 0.99, c
    if wf == "int8":
        with torch.no_grad():
            hf = m(torch.tensor([toks])).logits[0].numpy()
        assert np.median(_cos(G.softcap(spec, dev), hf)) > 0.98


@pytest.mark.parametrize("wf", ["int8", "fp4"])
def test_moe_small_cache_is_bit_exact(moe, wf):
    """k slots per layer: nearly every request misses and streams (while the dense MLP runs
    beside it); the logits do not move."""
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 10)]
    full = _moe_engine(moe, wformat=wf)
    ref = np.array([full.step(t) for t in toks])
    small = _moe_engine(moe, wformat=wf, experts=K)
    got = np.array([small.step(t) for t in toks])
    assert small.server.misses > len(toks)
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))


def test_moe_resident_and_the_card_loop(moe):
    """Resident decode (the token's rows gathered on the device) gives the per-position
    programs' logits bit for bit, with k slots per layer; the decode loop on the card (the
    generate program) gives the host's resident loop token for token while the experts stream
    between the tokens it picks."""
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 20)]
    a = _moe_engine(moe, experts=K, resident=True)
    b = _moe_engine(moe, experts=K)
    assert a.resident and a.can_generate
    la, lb = a.prefill(toks), b.prefill(toks)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    c = _moe_engine(moe, experts=K, resident=True)
    c.prefill(toks)
    ref, t = [], int(np.argmax(la))
    t0 = t
    for _ in range(8):
        t = int(np.argmax(a.step(t)))
        ref.append(t)
    misses = c.server.misses
    assert c.generate_card(t0, 8, stop_ids=[]) == ref
    assert c.server.misses > misses


def test_moe_lfu_policy_is_bit_exact(moe):
    """The slots replaced by least decayed use (moe.serve's default, ExpertServer policy "lfu")
    or by LRU, 4 slots of 8 per layer: other misses, the full cache's logits bit for bit."""
    toks = [int(t) for t in np.random.default_rng(6).integers(0, 1000, 16)]
    full = _moe_engine(moe)
    ref = np.array([full.step(t) for t in toks])
    misses = {}
    for policy in ("lru", "lfu"):
        eng = _moe_engine(moe, experts=4)
        assert eng.server.policy == "lfu"
        eng.server.policy = policy
        got = np.array([eng.step(t) for t in toks])
        assert np.array_equal(got.view(np.uint32), ref.view(np.uint32)), policy
        misses[policy] = eng.server.misses
    assert misses["lfu"] > len(toks) and misses["lfu"] != misses["lru"]


def test_moe_live_card_streams_from_a_split_pool(moe, tmp_path):
    """The host's side of path (a) as it runs beside a card (tests/test_lfm2_moe.py's
    _LiveCard: a fake card computing in a thread over the host's DRAM, with CHASH's channel
    map), k slots per layer: the experts come from a pool file in the split format, read into
    the channel runs and written by BoardDram's DMA thread while the card's MoE layers wait.
    Resident steps and the card's generate loop give the ISA simulator's logits and tokens bit
    for bit, with misses served during the runs."""
    from test_lfm2_moe import _LiveCard
    from opentpu.host.board import BoardBackend
    from opentpu.host.offload import BoardDram
    _, W, spec = moe
    cfg = _cfg()
    card = _LiveCard.make(cfg, chash=True)
    card.threaded = True
    kw = dict(cap=1024, cfg=cfg, rows=1, resident=True, experts=K)
    isa = Engine(spec, W, **kw)
    brd = Engine(spec, W, **kw, backend=lambda c, imgs: BoardBackend(c, imgs, transport=card),
                 pool_file=tmp_path / "pool.bin")
    assert brd.backend.board.chash and brd.can_generate and brd.backend.host is not None
    assert isinstance(brd.server.mem, BoardDram) and brd.server.pool_file.split
    toks = [int(t) for t in np.random.default_rng(8).integers(0, 1000, 6)]
    for t in toks:
        a, b = isa.step(t), brd.step(t)
        assert card.error is None, card.error
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), brd.pos
    assert brd.server.misses > len(toks) and card.waits > 0
    t0, misses, waits = int(np.argmax(a)), brd.server.misses, card.waits
    got = brd.generate_card(t0, 8, stop_ids=[])
    assert card.error is None, card.error
    assert got == isa.generate_card(t0, 8, stop_ids=[])
    assert brd.server.misses > misses and card.waits > waits      # served during the loop
    assert brd.server.misses == isa.server.misses
    assert 0 < brd.server.mem.direct <= brd.server.misses and brd.server.pool_warm


def test_moe_formats(moe):
    """experts=fmt sets the experts' format (one slot size for every layer); a range is
    refused; the router stays int8."""
    _, W, spec = moe
    assert G.expert_format(spec, "int8", "experts=fp4") == "fp4"
    assert G.expert_format(spec, "fp4", "") == "fp4"
    with pytest.raises(ValueError, match="per layer range"):
        G.expert_format(spec, "int8", "experts@0-2=fp4")
    img = spec.image(_cfg(), 1024, rows=1, formats="experts=fp4", experts=K)
    assert img.efmt == "fp4" and img.lf[0] == ("int8",) * 4
    assert img.offload.slots[0][1] == K


def _quant_eval():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "tools" / "gemma4_quant_eval.py"
    s = importlib.util.spec_from_file_location("gemma4_quant_eval", path)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


def test_quant_eval_emulation(kv, moe):
    """tools/gemma4_quant_eval.py's batched emulation of these models: with float weights it is
    reference_logits (the MoE's folds exact); without the MoE, int8 and fp4 equal
    emulated_logits (up to rounding ties); its MoE block in int8 (weights and activations) is
    within the dense MLP's error of the reference block (~2%)."""
    from opentpu.llm.qwen3 import _fake_q, _fake_w
    Q = _quant_eval()
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 20)]
    for m in (kv, moe):
        _, W, spec = m
        ref = G.reference_logits(spec, W, toks)
        assert np.abs(Q.emulate(spec, W, toks, wformat="none") - ref).max() < 1e-3
    _, W, spec = kv
    for wf in ("int8", "fp4"):
        a = Q.emulate(spec, W, toks, wformat=wf)
        b = G.emulated_logits(spec, W, toks, wformat=wf)
        assert np.abs(a - b).max() < 1e-3 * np.abs(b).max()
    _, W, spec = moe
    x = np.random.default_rng(5).standard_normal((64, 256)).astype(np.float32) * 3
    p = "model.layers.3."
    ref = G._moe(spec, W, p, x)

    def wq(n, fmt="int8", a=None):
        a = np.asarray(W[n] if a is None else a, np.float32)
        return _fake_w(np.pad(a, ((0, 0), (0, -a.shape[1] % 128))), 128, fmt)

    fq = lambda v, d=128: _fake_q(np.asarray(v, np.float64), d)                 # noqa: E731
    padq = lambda v, n: fq(np.pad(v, ((0, 0), (0, n - v.shape[1]))))           # noqa: E731
    out = Q._moe(spec, W, p, x, wq, fq, padq, False)
    assert (np.linalg.norm(out - ref, axis=1) / np.linalg.norm(ref, axis=1)).max() < 0.04


def test_moe_compile_worker_builds_the_image(moe):
    """The compile worker process (the card's backend compiles ahead in it) builds the
    engine's image, its K expert slots per layer included, and the same programs."""
    from opentpu import isa as I
    from opentpu.llm import qwen3 as Q
    _, W, spec = moe
    eng = _moe_engine(moe, experts=K, resident=True)
    Q._worker_init(spec, eng.cfg, eng.cap, eng.batch, eng.rows, eng.block, eng._image_kw)
    try:
        img = Q._WORKER[0]
        assert img.nbytes == eng.image.nbytes and img.offload == eng.image.offload
        words, _ = Q._worker_decode(1, 2)
        assert np.array_equal(words, I.assemble(eng.image.compile_decode(1, 2)[0][0]))
    finally:
        Q._WORKER = None
