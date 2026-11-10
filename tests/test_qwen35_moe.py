"""Qwen3.5-MoE on openTPU with its experts streamed into DRAM slots (docs/offload.md, path (a)):
the card routes (softmax of the k largest logits) and computes every expert and the gated
shared expert; the host's server moves the missing experts. A tiny random model against
Hugging Face transformers and the quantized emulation; a small cache gives the same logits
bit for bit as one holding every expert."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from opentpu.compiler import Affine
from opentpu.llm.moe import MoESpec
from opentpu.llm.qwen3 import Engine, device_config
from opentpu.llm.qwen35 import Spec, emulated_logits, reference_logits

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

KINDS = ("linear", "linear", "attn", "linear", "linear", "attn")
E, K, F = 8, 2, 256


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def _tiny(nk=8, nv=8):
    """nk DeltaNet key heads (8: one per value head; 4: a pair shares its q and k, as the
    35B's), nv value heads (16: a real model's register pressure in the device programs)."""
    torch.manual_seed(0)
    hc = transformers.Qwen3_5MoeTextConfig(
        hidden_size=256, num_hidden_layers=len(KINDS), num_attention_heads=8,
        num_key_value_heads=2, head_dim=256, vocab_size=1000,
        layer_types=["full_attention" if k == "attn" else "linear_attention" for k in KINDS],
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
    spec = Spec(256, KINDS, 8, 2, 256, 64, nv, 128, 128, F, 1000, lin_kheads=nk,
                moe=MoESpec(E=E, k=K, ffn=F, rule="softmax", shared=F))
    return m, W, spec


@pytest.fixture(scope="module")
def tiny():
    return _tiny()


def test_from_hf_gives_a_moe_the_int8_embedding(tiny, tmp_path):
    """A Qwen3.5-MoE checkpoint's Spec gathers its embedding rows on the device in int8 at any
    vocabulary size (its DRAM beside the layers is expert slots)."""
    tiny[0].config.save_pretrained(tmp_path)
    s = Spec.from_hf(tmp_path)
    assert s.moe is not None and s.embed == "int8"


def test_hints_are_off_by_default(tiny, tmp_path):
    """The router's prefetch hints stay off unless asked for (card session 5: 4.02 against 4.23
    tok/s without them, docs/offload.md 12.5): from_hf's programs are those of hint=False, not
    hint=True's."""
    from opentpu import isa as I
    tiny[0].config.save_pretrained(tmp_path)
    s = Spec.from_hf(tmp_path)
    assert not s.moe.hint
    cfg = device_config(s, 256, rows=1, lookup=True, S=1, experts=K)

    def program(spec):
        img = spec.image(cfg, 256, 1, 1, lookup=True, experts=K)
        return np.asarray(I.assemble(img.compile_step(5)[0])).tobytes()
    off, on = (program(replace(s, moe=replace(s.moe, hint=h))) for h in (False, True))
    assert program(s) == off != on


def _engine(spec, W, experts=None, **kw):
    from opentpu.llm.qwen3 import PREFILL_ROWS
    cfg = device_config(spec, 256, S=1, rows=PREFILL_ROWS, experts=experts, **kw)   # (the
    return Engine(spec, W, cap=256, cfg=cfg, experts=experts, **kw)     # Engine's rows)


def test_reference_matches_hf(tiny):
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 24)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(reference_logits(spec, W, toks) - hf).max() < 1e-4


def _hf_routed(m, toks, card):
    """HF's logits with every layer's experts the card's (card: the server's history, one
    request of global ids per token and layer; the last token's last layer, whose request waits
    for a next fence, keeps HF's own), weighted by HF's softmax over them."""
    L = len(KINDS)

    def hook(layer):
        def route(mod, inp, out):
            lg, _, idx = out
            idx = idx.clone()
            for t in range(len(toks)):
                if t * L + layer < len(card):
                    idx[t] = torch.tensor([g - layer * E for g in card[t * L + layer]])
            p = torch.softmax(lg, dim=-1, dtype=torch.float).gather(-1, idx)
            return lg, (p / p.sum(-1, keepdim=True)).to(lg.dtype), idx
        return route
    hs = [m.model.layers[i].mlp.gate.register_forward_hook(hook(i)) for i in range(L)]
    try:
        with torch.no_grad():
            return m(torch.tensor([toks])).logits[0].numpy()
    finally:
        for h in hs:
            h.remove()


def test_device_follows_hf_and_routes_on_the_card(tiny):
    """The logits follow HF routed as the card routed (every step but the last) and the
    emulation; the card's choices in the first layer, whose input no routing has touched, are
    the emulation's wherever its k-th and (k+1)-th logits are apart by more than the
    quantization's reach. HF takes the card's experts: a choice within the quantization's reach
    goes either way, and one unlike HF's takes the logits far from HF's own (cos 0.08 - 0.97;
    6 of 16 token sequences failed a test against HF's own routing, on main too)."""
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 12)]
    eng = _engine(spec, W)
    eng.server.history = []
    dev = np.array([eng.step(t) for t in toks])
    c = _cos(dev, _hf_routed(m, toks, eng.server.history))[:-1]
    assert c.min() > 0.99 and np.median(c) > 0.997
    routes = []
    emu = emulated_logits(spec, W, toks, routes=routes)
    assert np.median(_cos(dev, emu)) > 0.999
    L = len(KINDS)
    assert eng.server.seq == len(toks) * L - 1                # the last waits for a next fence
    card = eng.server.history
    checked = 0
    for t, layer, ids, margin in routes:
        if layer == 0 and margin > 1e-2:
            assert card[t * L] == ids, (t, card[t * L], ids, margin)
            checked += 1
    assert checked >= len(toks) - 3
    assert eng.server.misses == 0


@pytest.mark.parametrize("wformat,formats", [("int8", ""), ("fp4", ""), ("nvfp4", ""),
                                             ("fp4", "attn=nvfp4,delta=nvfp4,mlp=nvfp4")])
def test_small_cache_is_bit_exact(tiny, wformat, formats):
    """k slots per layer: nearly every request misses and streams; the logits do not move. The
    35B's formats on the card too (nvfp4 layers and shared expert, fp4 experts:
    streamed.FORMATS)."""
    _, W, spec = tiny
    spec = replace(spec, formats=formats)
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 8)]
    full = _engine(spec, W, wformat=wformat)
    assert full.image.fmt.wformat == wformat
    assert full.image.lf[0] == ("nvfp4" if formats else wformat,) * 4
    ref = np.array([full.step(t) for t in toks])
    small = _engine(spec, W, experts=K, wformat=wformat)
    got = np.array([small.step(t) for t in toks])
    assert small.server.misses > len(toks)
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))


@pytest.mark.parametrize("resident", [False, True])
def test_a_resident_pool_above_4gib_is_the_streamed_path_bit_for_bit(tiny, resident,
                                                                     monkeypatch):
    """At AW 36 (bcu1525-wg) with the weights above 4 GiB and every expert present, the pool
    is resident: expert g at the pool's base + g x its bytes (ADDI(RLD MUL(id, bytes), base)),
    no directory, no server. Its logits are the low image's (the offload path, every expert
    in a slot) bit for bit, per position and in the resident decode loop."""
    from opentpu import target as T
    _, W, spec = tiny
    cfg = T.config("bcu1525-wg", DRAM_BYTES=1 << 33)
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 6)]
    got = {}
    for high in ("0", "1"):
        monkeypatch.setenv("OTPU_HIGH", high)
        eng = Engine(spec, W, cap=256, cfg=cfg, resident=resident)
        im = eng.image
        assert im.high == (high == "1") and (im.pool is None) == (high == "0")
        assert (eng.server is None) == (high == "1")
        if im.high:
            assert im.layer0 == 1 << 32 and im.pool.base > 1 << 32 and im.low_end < 1 << 32
            assert im.pool.experts == len(KINDS) * E
        got[high] = np.array([eng.step(t) for t in toks])
    assert np.array_equal(got["0"].view(np.uint32), got["1"].view(np.uint32))


@pytest.mark.parametrize("resident,grouped", [(False, False), (True, True)])
def test_the_bank_half_window_is_the_low_image_bit_for_bit(tiny, resident, grouped,
                                                            monkeypatch):
    """bcu1525-wg:bhw (Config.BHW 33: the slice's first 8 GiB the bank-half window): the image is
    high whatever OTPU_HIGH says (Image.mix), its low area below 4 GiB (what a decode writes:
    the states, windows, KV caches, group-major the pairs' gates), every DeltaNet layer's
    mixers' weights in the window's upper half from 4 GiB (mloc; the mixer matrices' scale rows
    in the states' block below 4 GiB, Image.mix's slow), every other weight and the
    pool from 8 GiB on, the window past the mixers unwritten (Images.skip). The ISA simulator's
    guard (every write below 4 GiB) holds and the logits are the low image's bit for bit, per
    position and in the resident decode loop, array- and group-major."""
    from opentpu import target as T
    _, W, spec = tiny
    spec = replace(spec, pair_loop=grouped)
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 6)]
    monkeypatch.setenv("OTPU_HIGH", "0")
    got = {}
    for name, cfg in (("low", T.config("bcu1525-wg", DRAM_BYTES=1 << 33)),
                      ("bhw", T.config("bcu1525-wg:bhw", DRAM_BYTES=1 << 34))):
        eng = Engine(spec, W, cap=256, cfg=cfg, resident=resident)
        im = eng.image
        assert im.grouped == grouped and im.mix == (name == "bhw")
        if im.mix:
            lin = [i for i, k in enumerate(KINDS) if k == "linear"]
            mx = [im._moff(i).const for i in lin]
            assert im.low_end < 1 << 32 == min(mx) < max(mx) < im.wskip[0] < im.wskip[1] == \
                1 << 33 == im.layer0 < im.pool.base and im.wlo == 1 << 32
            assert all(s.wlim == 1 << 32 for s in eng.backend.machine.slices)
            # the mixer matrices' scale rows in the states' block below 4 GiB, apart from
            # their data's banks in the upper half (the layouts' slow)
            m = im.descriptors(0, 0)
            for i in lin:
                slow = im.layouts[im.keys[i][1]].slow
                assert "wab" in slow and ("wh" in slow) == (not grouped)
                for n in slow:
                    q = getattr(m.layer(i), n)
                    assert Affine.of(q.scale).static() < 1 << 32 <= Affine.of(q.data).static()
        got[name] = np.array([eng.step(t) for t in toks])
    assert np.array_equal(got["low"].view(np.uint32), got["bhw"].view(np.uint32))


def test_a_sum_from_minus_zero_is_the_rows_sum_bit_for_bit():
    """moe_ffn's resident loop adds each expert's weighted output into a row filled with -0
    (the offload path sums y's rows, y0 + y1 + ...): on the ISA simulator -0 + y0 is y0's bits,
    so the sum is the rows' bit for bit, a column of -0s included (from +0 it would be +0)."""
    from opentpu import Config
    from opentpu import language as ol
    from opentpu.runtime import Input, Output, launch

    @ol.jit
    def k(y, s, z):
        a, b = ol.full((1, 128), -0.0), ol.zeros((1, 128))
        for i in ol.static_range(4):
            r = ol.load(y[i:i + 1, :])
            a.set(a + r)
            b.set(b + r)
        ol.store(s, a)
        ol.store(z, b)

    rng = np.random.default_rng(5)
    y = rng.choice(np.float32([-0.0, 0.0, 1.5, -1.5, 2.0 ** -126, -2.0 ** -126, 1e30, -1e30]),
                   (4, 128))
    y[:, 0], y[:, 1] = -0.0, [-0.0, 0.0, -0.0, -0.0]
    r = launch(k, Config(), y=Input(y.astype(np.float32)), s=Output((1, 128)),
               z=Output((1, 128)))
    want = y[0]
    for i in range(1, 4):
        want = (want + y[i]).astype(np.float32)
    got = r.outputs["s"].astype(np.float32).reshape(-1)
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
    assert np.signbit(got[0]) and not np.signbit(r.outputs["z"].reshape(-1)[0])


@pytest.mark.parametrize("resident", [False, True])
def test_a_resident_pool_below_4gib_is_the_streamed_path_bit_for_bit(tiny, resident,
                                                                     monkeypatch):
    """OTPU_POOL=1: the resident pool with the weights below 4 GiB (the board model's runs of a
    cut take the full model's resident path; its DIMM model holds no image above 4 GiB): expert
    g at the pool's base + g x its bytes, below HIGH, no directory, no server. Its logits are
    the offload path's bit for bit, per position and in the resident decode loop; without the
    knob the image is the offload path's, as before."""
    from opentpu import target as T
    _, W, spec = tiny
    cfg = T.config("bcu1525-wg", DRAM_BYTES=1 << 33)
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 6)]
    monkeypatch.setenv("OTPU_HIGH", "0")
    got = {}
    for pool in ("", "1"):
        monkeypatch.setenv("OTPU_POOL", pool)
        eng = Engine(spec, W, cap=256, cfg=cfg, resident=resident)
        im = eng.image
        assert not im.high and (im.pool is None) == (pool == "") == (eng.server is not None)
        if pool:
            assert im.pool.base + im.pool.experts * im.pool.ebytes <= im.nbytes < 1 << 32
            assert im.pool.experts == len(KINDS) * E
        got[pool] = np.array([eng.step(t) for t in toks])
    assert np.array_equal(got[""].view(np.uint32), got["1"].view(np.uint32))


def test_a_resident_pool_streams_the_shared_experts_mms_while_it_routes():
    """With a resident pool the shared expert's gate and up MMs (they need only the normed
    input) are issued right after the router's MM, before its ARGMAX loop, so DRAM streams them
    while the VPU routes (moe.moe_ffn / moe_ffn_rows: swiglu_down's each(0)); with the server's
    slots (low) they come after the request, as before. In the decode step and in layer runs of
    1 and 4 rows. (The logits stay the low image's bit for bit: the test above.)"""
    from opentpu import isa as I
    from opentpu import target as T
    spec = Spec(256, KINDS, 8, 2, 256, 64, 8, 128, 128, F, 1000, lin_kheads=8,
                moe=MoESpec(E=E, k=K, ffn=F, rule="softmax", shared=F))
    cfg = T.config("bcu1525-wg", DRAM_BYTES=1 << 33)

    def between(prog):
        """MMs between each routing's router MM (E + 1 rows) and its first ARGMAX."""
        out, n = [], None
        for x in prog:
            if x.op == I.MM:
                n = 0 if f". {E + 1}x" in x.comment else None if n is None else n + 1
            elif x.op == I.VOP and x.comment == "moe: best" and n is not None:
                out.append(n)
                n = None
        return out
    for high, want in ((True, 2), (False, 0)):      # (F 256 at fp4: one chunk, its gate and up)
        im = spec.image(cfg, 256, 1, 1, wformat="fp4", lookup=True, prefill_rows=8, high=high)
        assert (im.pool is not None) == high
        for progs in (im.compile_step(5), im.compile_layer_run(1, 1, R=1)[0],
                      im.compile_layer_run(3, 1, R=4)[0]):
            got = between(progs[0])
            assert got and set(got) == {want}, (high, got)


def test_a_resident_pool_queues_the_windows_beside_its_experts():
    """The DSTEP decode's taps and windows a layer ahead (qwen35._CvPipe) in a MoE layer: with a
    resident pool the layer's window stores and the next DeltaNet layer's loads are queued in
    parts, one after each of the first routed experts' MMs (moe_ffn's spread: those stream from
    the pool and the DMA is idle; queued whole before them they filled the sequencer's window),
    none before the first, so the router's gain load no longer waits for them; with the server's
    slots (low) they are queued before the FFN, as before. (The logits stay the low image's bit
    for bit: test_a_resident_pool_above_4gib_is_the_streamed_path_bit_for_bit.)"""
    from opentpu import isa as I
    from opentpu import target as T
    spec = Spec(256, KINDS, 8, 2, 256, 64, 8, 128, 128, F, 1000, lin_kheads=8,
                moe=MoESpec(E=E, k=K, ffn=F, rule="softmax", shared=F))
    cfg = T.config("bcu1525-wg", DRAM_BYTES=1 << 33)
    for high in (True, False):
        im = spec.image(cfg, 256, 1, 1, wformat="fp4", lookup=True, high=high)
        assert im.cv_pre and (im.pool is not None) == high
        (prog,), _ = im.compile_decode(1, 3)
        fns = [[f[2] for f in x.src] for x in prog]
        pipe = [i for i, f in enumerate(fns) if "_emit" in f or "share" in f]
        groups = [g for g in np.split(pipe, np.flatnonzero(np.diff(pipe) > 1) + 1)]
        # (the loop's unit: two DeltaNet layers; with the pool a part after each of their K
        # routed experts)
        assert len(groups) == 2 * (K if high else 1), (high, len(groups))
        mms = [i for i, x in enumerate(prog) if x.op == I.MM]
        assert {prog[i].op for i in pipe} == {I.ST, I.LD}
        for g in groups:
            before = fns[max(i for i in mms if i < g[0])]
            after = fns[min(i for i in mms if i > g[-1])]
            if high:                    # (a routed expert's MM: none before the first)
                assert "expert" in before
            else:                       # (out_proj's last MM, the router's)
                assert "_out_proj" in before and "moe_ffn" in after and "swiglu_down" not in after


def test_a_resident_pools_minus_zero_fill_runs_under_the_routers_mm():
    """moe_ffn's resident loop sums the experts into a row filled with -0. The FILL is emitted
    before the router's MM (the VPU waits for the logits there): after the routing it sat in
    the VPU's order before the first expert's place, and behind the pipe's LDs beside the
    experts (qwen35._CvPipe) on TMEM's write port, where the DMA comes first."""
    from opentpu import isa as I
    from opentpu import target as T
    spec = Spec(256, KINDS, 8, 2, 256, 64, 8, 128, 128, F, 1000, lin_kheads=8,
                moe=MoESpec(E=E, k=K, ffn=F, rule="softmax", shared=F))
    cfg = T.config("bcu1525-wg", DRAM_BYTES=1 << 33)
    im = spec.image(cfg, 256, 1, 1, wformat="fp4", lookup=True, high=True)
    assert im.pool is not None
    (prog,), _ = im.compile_decode(1, 3)
    fns = [[f[2] for f in x.src] for x in prog]
    fills = [i for i, x in enumerate(prog) if x.op == I.VOP and "moe_ffn" in fns[i]
             and (x.w[5] >> 16) & 0xFF == I.V_FILL and x.w[6] == 0x80000000]
    mms = [i for i, x in enumerate(prog) if x.op == I.MM]
    routers = [i for i in mms if fns[i][0] == "moe_ffn"]
    assert fills and len(fills) == len(routers)
    for f, r in zip(fills, routers):    # (no MM, ST or LD between the FILL and the router's MM)
        assert f < r and not any(prog[i].op in (I.MM, I.ST, I.LD) for i in range(f + 1, r))


def test_a_resident_experts_place_is_bounded_by_its_pool():
    """The resident pool's RLD MUL (id x expert bytes: past 32 bits above 4 GiB) carries the
    pool's bound ({w5, w3} .. {w6, w4}: isa.rld), so the waits pass takes the expert's place as
    one in the pool, above the run-time parts: no wait is stricter than without the bound (as
    before: any address), every expert MM waits for less (none of the DMA's older work), and so
    do the stores after them (the next layer's DSTEPs, STs and QSTs had waited for the expert
    MMs). In the decode step and a layer run of 4 rows."""
    from opentpu import isa as I
    from opentpu import target as T
    from opentpu import waits as WT
    spec = Spec(256, KINDS, 8, 2, 256, 64, 8, 128, 128, F, 1000, lin_kheads=8,
                moe=MoESpec(E=E, k=K, ffn=F, rule="softmax", shared=F))
    cfg = T.config("bcu1525-wg", DRAM_BYTES=1 << 33)
    im = spec.image(cfg, 256, 1, 1, wformat="fp4", lookup=True, prefill_rows=8, high=True)
    P = im.pool

    def looser(x, y):               # (a count 0: no wait; k: for the k-th last of its unit)
        return all(cx == 0 or 0 < cy <= cx for cx, cy in zip(WT.counts(x), WT.counts(y))) \
            and not WT.vmask(x) & ~WT.vmask(y)
    for prog, per in ((im.compile_step(5)[0], 2), (im.compile_layer_run(3, 1, R=4)[0][0], 1)):
        pcs = [pc for pc, x in enumerate(prog) if x.op == I.RLD and x.flags & I.F_MUL]
        assert pcs and all(prog[pc].flags & I.F_BOUND for pc in pcs)
        assert {(prog[pc].w[2] | prog[pc].w[4] << 32, prog[pc].w[3] | prog[pc].w[5] << 32)
                for pc in pcs} == {(0, (P.experts - 1) * P.ebytes)} and P.base >= 1 << 32
        loose = [replace(x, flags=x.flags & ~I.F_BOUND) if pc in pcs else x
                 for pc, x in enumerate(prog)]
        a, b = (WT.wait_counts(p, cfg.D, 1, run_args=prog.run_args, aw=cfg.AW)
                for p in (prog, loose))
        assert all(looser(x, y) for x, y in zip(a, b))
        less = [prog[pc].op for pc in range(len(prog)) if a[pc] != b[pc]]
        # (gate, up, down at least, in each expert loop: moe_ffn reads the first expert's
        # place before its loop and the next one's at the end of its body, moe_ffn_rows
        # each in its body)
        assert len(pcs) % per == 0 and less.count(I.MM) >= 3 * len(pcs) // per, less
        assert {I.ST, I.MM} <= set(less), less


def test_a_resident_pool_quantizes_its_experts_in_the_batchs_workers(tiny, monkeypatch):
    """The resident pool's experts' matrices go through a qcache.Batch, in worker processes (in
    one process an expert of Qwen3.5-35B-A3B's shapes took 0.68 s on a laptop, 116 minutes for
    its 10,240; in a Batch of 4, 0.16 s): every expert's bytes in DRAM are its pack()'s (the
    server's), with the channel hash at its address, as before; the matrices quantized in two
    workers (INLINE 0, the cache off)."""
    from opentpu import qcache as QC
    from opentpu import target as T
    from opentpu.quant import hash_rows
    _, W, spec = tiny
    monkeypatch.setenv("OTPU_HIGH", "1")
    monkeypatch.setenv("OTPU_IMAGE_CACHE", "0")
    monkeypatch.setenv("OTPU_BUILD_JOBS", "2")
    monkeypatch.setattr(QC, "INLINE", 0)
    sent = []
    submit = QC._Rows.__init__

    def rows(self, n, done, f):
        sent.append(n)
        submit(self, n, done, f)
    monkeypatch.setattr(QC._Rows, "__init__", rows)
    eng = Engine(spec, W, cap=256, cfg=T.config("bcu1525-wg", DRAM_BYTES=1 << 33))
    im, dram = eng.image, eng.backend.machine.slices[0].dram
    P, (wg, D, own) = im.pool, im.slot_lanes
    assert P.experts == len(KINDS) * E and len(sent) >= 3 * P.experts
    for g in range(P.experts):
        at, want = P.base + g * P.ebytes, im.expert(W, g)
        for o, n in own:
            want[o:o + n] = hash_rows(want[o:o + n], at + o, wg, D)
        assert np.array_equal(dram[at:at + P.ebytes], want), g


def _untied(tiny):
    """The tiny model with an LM head of its own: its int8 embedding table is the image's own
    (a tied int8 head's rows are gathered from the head)."""
    _, W, spec = tiny
    W = dict(W, **{"lm_head.weight": W["model.embed_tokens.weight"] * np.float32(1.25)})
    return replace(spec, tied=False, embed="int8"), W


def test_the_embedding_table_on_the_host(tiny):
    """A MoE model's int8 embedding table of its own stays on the host (embed_host, the
    default; the card's DRAM is expert slots): the card holds a slot of one row, which the host
    writes before each run, and in the card's generate loop the host's row server after each
    sampled token's post. Prefill, the generate loop and the resident steps give the table on
    the card's logits and tokens bit for bit (k slots per layer). A tied int8 head's rows stay
    the gather's."""
    spec, W = _untied(tiny)
    cfg = device_config(spec, 512, rows=1, lookup=True, S=1, experts=K, embed_host=False)
    a, b = (Engine(spec, W, cap=512, cfg=cfg, rows=1, resident=True, experts=K, **kw)
            for kw in ({}, {"embed_host": False}))
    assert a.image.embed_host and a.row_server is not None and not b.image.embed_host
    assert b.image.nbytes - a.image.nbytes >= spec.vocab * spec.hidden
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 12)]
    la, lb = a.prefill(toks), b.prefill(toks)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    t0 = int(np.argmax(la))
    got = a.generate_card(t0, 8, stop_ids=[])
    assert got == b.generate_card(t0, 8, stop_ids=[]) and a.row_server.seq >= 7
    for t in got[-3:]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32))
    _, W1, s1 = tiny
    tied = Engine(replace(s1, embed="int8"), W1, cap=512, cfg=cfg, rows=1, resident=True,
                  experts=K)
    assert not tied.image.embed_host and tied.row_server is None


def test_a_resident_pool_keeps_the_table_on_the_card(tiny, monkeypatch):
    """With every expert resident (bcu1525-wg's pool after the head above HIGH, the 35B's image)
    the int8 embedding table of its own stays on the card by default, as a dense model's: no
    row server, the generate loop gathers each token's row itself. Prefill's logits, the
    generate loop's tokens and the resident steps' logits are the table on the host's bit for
    bit."""
    from opentpu import target as T
    spec, W = _untied(tiny)
    monkeypatch.setenv("OTPU_HIGH", "1")            # (the 35B's: by fit)
    cfg = T.config("bcu1525-wg", DRAM_BYTES=1 << 33)
    a, b = (Engine(spec, W, cap=512, cfg=cfg, rows=1, resident=True, **kw)
            for kw in ({}, {"embed_host": True}))
    assert a.image.high and a.image.pool is not None and a.image.offload is None
    assert a.server is None
    assert not a.image.embed_host and a.row_server is None and "mbox" not in a.image.lookup
    assert b.image.embed_host and b.row_server is not None
    assert a.image.low_end - b.image.low_end >= spec.vocab * spec.hidden     # (below HIGH)
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 12)]
    la, lb = a.prefill(toks), b.prefill(toks)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    t0 = int(np.argmax(la))
    got = a.generate_card(t0, 8, stop_ids=[])
    assert got == b.generate_card(t0, 8, stop_ids=[]) and b.row_server.seq >= 7
    assert len(set(got)) > 2                        # (the tokens vary)
    for t in got[-3:]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32))


def test_the_live_card_asks_the_host_for_experts_and_embedding_rows(tiny, tmp_path):
    """The host's two servers beside a card that computes while they work
    (tests/test_lfm2_moe.py's _LiveCard, CHASH's map): the experts from a split-format pool
    and the embedding rows (embed_host) through one BoardDram, its DMA thread writing while the
    card waits. Resident steps and the card's generate loop give the ISA simulator's logits
    and tokens bit for bit, every expert and row request served during the runs."""
    from test_lfm2_moe import _LiveCard, _wait_first
    from opentpu.host.board import BoardBackend
    from opentpu.host.offload import BoardDram
    from opentpu.isasim import board_config
    spec, W = _untied(tiny)
    cfg = board_config(DRAM_BYTES=1 << 25)
    card = _LiveCard.make(cfg, chash=True)
    card.threaded = True
    kw = dict(cap=256, cfg=cfg, rows=1, resident=True, experts=K)
    isa = Engine(spec, W, **kw)
    brd = Engine(spec, W, **kw, backend=lambda c, imgs: BoardBackend(c, imgs, transport=card),
                 pool_file=tmp_path / "pool.bin")
    assert brd.image.embed_host and isinstance(brd.server.mem, BoardDram)
    assert brd.row_server.mem is brd.server.mem
    again = _wait_first(card, brd.server)   # (card.waits whatever the host's speed)
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 6)]
    for t in toks:
        a, b = isa.step(t), brd.step(t)
        assert card.error is None, card.error
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), brd.pos
    t0, waits = int(np.argmax(a)), card.waits
    again()
    got = brd.generate_card(t0, 8, stop_ids=[])
    assert card.error is None, card.error
    assert got == isa.generate_card(t0, 8, stop_ids=[]) and card.waits > waits
    assert brd.row_server.seq >= 7 and isa.row_server.seq >= 7     # (the last token's post:
                                                                    # served when the host looks)
    assert brd.server.misses == isa.server.misses > len(toks)


def test_the_card_generates_with_streamed_experts(tiny):
    """The decode loop on the card (resident decode) with k slots per layer gives the host's
    resident loop token for token while the experts stream."""
    _, W, spec = tiny
    cfg = device_config(spec, 512, rows=1, lookup=True, S=1, experts=K)
    a, b = (Engine(spec, W, cap=512, cfg=cfg, rows=1, resident=True, experts=K)
            for _ in range(2))
    assert a.can_generate
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 16)]
    t0 = int(np.argmax(a.prefill(toks)))
    assert int(np.argmax(b.prefill(toks))) == t0
    ref, t = [], t0
    for _ in range(10):
        t = int(np.argmax(b.step(t)))
        ref.append(t)
    misses = a.server.misses
    assert a.generate_card(t0, 10, stop_ids=[]) == ref
    assert a.server.misses > misses


def test_moe_on_board_model(tiny, have_verilator):
    """The MoE block on the RTL (WAITW) through the host driver, as test_lfm2_moe.py's: every
    expert resident and `served` past any request, so each layer's fence and directory waits
    hold at their first read. Per-position programs, then the resident decode, then the card's
    generate loop (the DeltaNet layers, the routed experts and the gated shared expert): the ISA
    simulator's logits and tokens bit for bit."""
    from conftest import sim_board_config
    from opentpu.host.board import Board, BoardBackend, SimTransport
    _, W, spec = tiny
    cfg = sim_board_config(DRAM_BYTES=1 << 26)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5)
    assert Board(tr).info()["caps"].get("waitw"), "the board model has no WAITW (CAPS bit31)"
    isa = Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True)
    brd = Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    assert brd.resident and brd.can_generate and brd.server.misses == 0
    L = brd.image.offload
    brd.backend.write(0, L.served, np.array([3e38], np.float32))
    for tok in (11, 222, 333, 444):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), brd.pos
    t0 = int(np.argmax(a))
    assert brd.generate_card(t0, 4, stop_ids=[]) == isa.generate_card(t0, 4, stop_ids=[])
    assert isa.server.misses == 0 and brd.stats[-1]["cycles"] > 0


@pytest.mark.parametrize("embed_host", [True, False])
def test_generate_ahead_gives_the_steps_tokens(monkeypatch, embed_host):
    """The generate loop's ahead (qwen35._loop_ahead: the gather's operand held in ACT RAM, the
    first DeltaNet layer's taps and windows loaded in each token's LM head for the next) on a
    Qwen3.6-style MoE of 16 DeltaNet value heads, a key head a pair (the 35B's), an LM head of
    its own and the int8 table on the host (embed_host: the row server feeds the loop) or on
    the card (the loop's gather), on bcu1525-wg's arch (cv_pre, every expert resident): two
    runs' tokens are the host's resident steps', and the loop's DRAM after them is the loop's
    without the ahead bit for bit (the last token's loads write nothing)."""
    from opentpu import isa as I
    from opentpu import target as T
    from opentpu.llm import qwen35 as Q35
    _, W, spec = _tiny(nk=8, nv=16)
    W = dict(W, **{"lm_head.weight": W["model.embed_tokens.weight"] * np.float32(1.25)})
    spec = replace(spec, tied=False, embed="int8", cv_ahead=True, cv_pre=True)
    cfg = T.config("bcu1525-wg", DRAM_BYTES=1 << 27)
    a, b, c = (Engine(spec, W, cap=512, cfg=cfg, rows=1, resident=True, embed_host=embed_host)
               for _ in range(3))
    assert a.image.cv_pre and a.image.embed_host == embed_host and a.can_generate
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 12)]
    t0 = int(np.argmax(a.prefill(toks)))
    assert int(np.argmax(b.prefill(toks))) == t0
    ref, t = [], t0
    for _ in range(12):
        t = int(np.argmax(b.step(t)))
        ref.append(t)
    assert len(set(ref)) > 6                        # the tokens vary
    assert a.generate_card(t0, 8, stop_ids=[]) + a.generate_card(ref[7], 4, stop_ids=[]) == ref
    for progs in a._gens.values():  # (the ahead's: the operand's QACT and the pipe's loads)
        lo = next(i for i, x in enumerate(progs[0]) if x.op == I.LOOP)
        ops = [x.op for x in progs[0][:lo]]
        assert ops.count(I.QACT) == 1 and ops.count(I.LD) == 1 + spec.lin_heads // 2
    with monkeypatch.context() as mp:
        mp.delattr(Q35.qwen35_step, "ahead")
        c.prefill(toks)
        assert c.generate_card(t0, 8, stop_ids=[]) + c.generate_card(ref[7], 4, stop_ids=[]) \
            == ref
    assert all(np.array_equal(x.dram, y.dram)
               for x, y in zip(a.backend.machine.slices, c.backend.machine.slices))


@pytest.mark.parametrize("wformat,R,b,real,embed_runs", [
    ("int8", 1, False, False, False), ("fp4", 2, True, False, False),
    ("fp4", 1, True, True, False), ("fp4", 2, True, True, False), ("fp4", 1, True, True, True)])
def test_layer_major_prefill_is_bit_exact(tiny, wformat, R, b, real, embed_runs, monkeypatch):
    """The prompt layer by layer (Engine.prefill_layers, docs/offload.md 13): every row of a
    chunk through a layer before the next, in runs of R rows at their run-time position and
    chunk row (R > 1: _deltanet_rows, _attention_rows at run-time rows, moe.moe_ffn_rows),
    gives token-by-token prefill's logits, DeltaNet states and windows and KV cache bit for bit
    (all of layer 0 to the head but a group-major layer's gates, a one-row run's scratch that
    R rows keep in m.gr): 262 tokens in chunks of 100 rows, the last crossing an attention block
    (runs split at its end), 2k slots per layer, the table on the host (each embedding row the
    host's); then the next decode steps. b: build B's PAIR,
    DSTEP and STREAM (the state steps by DSTEP). real: the 35B's DeltaNet layout, its pairs
    sharing q and k (4 key heads) and group-major (Spec.pair_loop: a hardware loop over the
    pairs, as past PAIR_LOOP). The rows before conv_k - 1 token by token and layer 0's runs
    gathering their rows from the host's slot (no embed runs: the card's port A, docs/offload.md
    13.6; int8: the prompt in two calls, the first of 2 tokens), or with embed_runs the embed
    runs and the rows before conv_k - 1 at compile-time positions."""
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 100)
    spec, W = _untied(_tiny(nk=4) if real else tiny)
    if real:
        spec = replace(spec, pair_loop=True)
    kw = dict(MCOLS=4, PAIR=True, DSTEP=True, STREAM=True) if b else {}
    cfg = device_config(spec, 512, rows=1, lookup=True, S=1, experts=2 * K, wformat=wformat,
                        **kw)
    a, ref = (Engine(spec, W, cap=512, cfg=cfg, rows=1, resident=True, experts=2 * K,
                     wformat=wformat, **lm)
              for lm in ({"layer_major": R, "embed_runs": embed_runs}, {}))
    assert a.image.embed_host and a.image.prefill_rows == 100
    assert (a.image.grouped, a.image.shared) == (real, real)
    toks = [int(t) for t in np.random.default_rng(7).integers(0, 1000, 262)]
    if wformat == "int8":
        a.prefill(toks[:2])
        assert a.pos == 2 and not a._layer_runs
        la = a.prefill(toks[2:])
    else:
        la = a.prefill(toks)
    lb = ref.prefill(toks)
    static = [k for k in a._layer_runs if k != "head" and (k[0] < 0 or k[1] is None)]
    assert bool(static) == embed_runs           # (embed runs, compile-time positions)
    assert a.pos == ref.pos == len(toks)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    img = a.image
    # all of it but a group-major layer's gates (DeltaNetParts._gates, each head group's): dead
    # after every run -- _deltanet / _deltanet_dstep store every pair's (store_gates) before
    # they load one (gates), in the same program, and nothing else reads them -- and R rows
    # keep theirs in m.gr, so the last one-row run's stay there. Exactly those bytes, from the
    # image's descriptors; the states, windows and KV cache compared whole
    dram = [e.backend.machine.slices[0].dram[img.layer0:img.head[0]].copy() for e in (a, ref)]
    for li in range(spec.layers):
        dn = getattr(img.descriptors(0).layer(li), "dn", None)
        for g in range(dn.nl // dn.og if dn is not None and dn.grouped else 0):
            e = dn._gates(g)
            o = Affine.of(e.base).const - img.layer0
            for d in dram:
                d[o:o + 4 * e.shape[0] * e.shape[1]] = 0
    assert np.array_equal(*dram)
    assert a.server.misses > 0
    t = int(np.argmax(la))
    for _ in range(3):
        ga, gb = a.step(t), ref.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))



@pytest.mark.parametrize("expert_major", [False, True])
def test_formats_by_layer_range_stream_bit_exact(tiny, expert_major, monkeypatch):
    """Formats by layer range on the streamed MoE, as MiniCPM-V's mix on the card
    (streamed.FORMATS, docs/minicpmv.md 5.1: the first DeltaNet layers and the shared expert
    int8, the other layers nvfp4, fp4 experts): each layer's projections in its range's format,
    and the layer-major prefill (expert-major and not) gives token by token's logits, then its
    decode steps', bit for bit, on build B's PAIR, DSTEP and STREAM with 6 slots a layer."""
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    spec, W = _untied(tiny)
    spec = replace(spec, formats="delta@0-2=int8,attn=nvfp4,delta=nvfp4,mlp=int8")
    kw = dict(MCOLS=4, PAIR=True, DSTEP=True, STREAM=True)
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=6, wformat="fp4", **kw)
    a, ref = (Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True, experts=6,
                     wformat="fp4", **x)
              for x in ({"layer_major": 2, "expert_major": expert_major}, {}))
    assert a.image.lf[0] == ("int8", "nvfp4", "int8", "int8")
    assert a.image.lf[3] == ("nvfp4", "nvfp4", "int8", "int8")
    assert a.image.fmt.wformat == "fp4"
    toks = [int(t) for t in np.random.default_rng(13).integers(0, 1000, 40)]
    la, lb = a.prefill(toks), ref.prefill(toks)
    assert a.server.misses > 0
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    t = int(np.argmax(la))
    for _ in range(2):
        ga, gb = a.step(t), ref.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))


@pytest.mark.parametrize("ready,hint", [(True, False), (False, False), (True, True)])
def test_layer_major_runs_compile_in_the_worker_processes(tiny, ready, hint):
    """Engine._precompile_layers: with the process pipeline (a backend running assembled words,
    as the board does) prefill_layers' programs all come from the worker processes
    (layer_programs in the worker's image, assembled), queued in the order they run; here the
    ISA simulator runs them (the words decoded), and the logits and the decode steps after
    equal token by token's bit for bit. With the workers not up yet (ready False: a prompt
    right after the engine) each run compiles in line and its queued compile is dropped.
    hint: the workers' programs post the next layer's hints (layer_ahead "hint")."""
    from opentpu import isa as I
    from opentpu.llm.qwen3 import IsaBackend

    class Words(IsaBackend):                    # the ISA simulator taking the board's words
        runs_words = True

        def run(self, programs, args=None):
            if isinstance(programs, np.ndarray):
                programs = [[I.Instr.decode(programs[i:i + 8])
                             for i in range(0, len(programs), 8)]]
            return super().run(programs, args)

    spec, W = _untied(tiny)
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=2 * K)
    a = Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True, experts=2 * K, layer_major=2,
               backend=lambda c, imgs: Words(c, imgs, adopt=True),
               **({"layer_ahead": "hint"} if hint else {}))
    ref = Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True, experts=2 * K)
    try:
        assert a._procs and a._ready.result(timeout=120)
        if not ready:                           # the workers still starting: no compile
            from concurrent.futures import Future   # done, none ready

            class Starting:
                def __init__(self, pool):
                    self.pool, self.futures = pool, []

                def submit(self, *args):
                    self.futures.append(Future())
                    return self.futures[-1]

                def shutdown(self):
                    self.pool.shutdown()
            a._pool, a._ready = Starting(a._pool), Future()
        toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 21)]
        la, lb = a.prefill(toks), ref.prefill(toks)
        assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
        assert a._layer_runs and not a._layer_next
        assert (a.server.hints > 0) == hint
        assert all((progs is None) == ready and isinstance(w, np.ndarray)  # (None: the
                   for progs, _, w in a._layer_runs.values())             # worker's)
        if not ready:
            assert a._pool.futures and all(f.cancelled() for f in a._pool.futures)
        for t in (5, 6):
            assert np.array_equal(a.step(t).view(np.uint32), ref.step(t).view(np.uint32))
    finally:
        a._pool.shutdown()



def test_layer_major_prompts_as_prefill_chunks(tiny):
    """Engine.prefill_chunks of a layer-major engine (sequence 0): the prompt in one part,
    prefill_layers' logits, its runs' stats one entry (a chat's prefill cycles: Chat.ask reads
    the last); with logits=False (a chat's part up to its snapshot) prefill_layers runs no head
    and the part yields None, and the prompt continued from there gives the logits of the whole
    prompt in one call, also from a part inside the rows before conv_k - 1 (token by token)."""
    from opentpu.isasim import board_config
    spec, W = _untied(tiny)
    cfg = board_config(DRAM_BYTES=1 << 26)
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 11)]

    def engine():
        return Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True, experts=K,
                      layer_major=2)
    a, b = engine(), engine()
    parts = list(a.prefill_chunks(toks))
    want = b.prefill(toks)
    assert len(parts) == 1 and parts[0][0] == toks
    assert a.stats == [{"cycles": sum(s.get("cycles") or 0 for s in b.stats),   # (its runs'
                        "runs": len(b.stats), "rows": len(toks)}] and len(b.stats) > 6  # in one)
    assert list(a.prefill_chunks([])) == [] and a.pos == len(toks)
    assert np.array_equal(parts[0][1].view(np.uint32), want.view(np.uint32))
    c = engine()
    assert list(c.prefill_chunks(toks[:6], logits=False)) == [(toks[:6], None)]
    assert "head" not in c._layer_runs and c.pos == 6
    (_, got), = c.prefill_chunks(toks[6:])
    assert "head" in c._layer_runs
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
    for t in toks[:2]:
        assert np.array_equal(c.step(t).view(np.uint32), b.step(t).view(np.uint32))
    e = engine()
    assert spec.conv_k > 2 and e.prefill_layers(toks[:2], logits=False) is None
    assert np.array_equal(e.prefill_layers(toks[2:]).view(np.uint32), want.view(np.uint32))


def test_layer_ahead_sends_the_next_layers_experts(tiny, monkeypatch):
    """Engine(layer_ahead=orders): prefill_layers turns the server's layer ahead on
    (begin_prefill(ahead=True)) and gives it each MoE layer's experts in the order's, as global
    ids, between runs (ExpertServer.ahead_layer): the first layer's after begin_prefill, the
    next layer's before each layer's first run, the first layer's again before the last layer
    of a chunk another chunk follows (17 rows in chunks of 12 and 5). With the link's idle time
    between runs (polls until none sends: the ISA simulator polls only while a run waits) the
    queued experts land: fewer misses than without, the logits and the decode steps after
    token by token's bit for bit."""
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 12)
    spec, W = _untied(tiny)
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=2 * K)
    order = [[(e * 3 + j) % E for e in range(6)] for j in range(len(KINDS))]
    a, b, ref = (Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True, experts=2 * K, **kw)
                 for kw in ({"layer_major": 2, "layer_ahead": order}, {"layer_major": 2}, {}))
    srv, calls = a.server, []
    ahead, begin, run = srv.ahead_layer, srv.begin_prefill, a.backend.run

    def ahead_layer(j, ids):
        calls.append((j, list(ids)))
        ahead(j, ids)

    def idle_then_run(*args, **kw):             # the link's idle time before each run
        while srv.poll():
            pass
        return run(*args, **kw)
    srv.ahead_layer, a.backend.run = ahead_layer, idle_then_run
    srv.begin_prefill = lambda **kw: (calls.append(("begin", kw)), begin(**kw))
    toks = [int(t) for t in np.random.default_rng(9).integers(0, 1000, 21)]
    m0 = a.server.misses, b.server.misses
    la, lb, lr = a.prefill(toks), b.prefill(toks), ref.prefill(toks)
    assert a.image.prefill_rows == 12 and a.pos == len(toks)
    assert calls[0] == ("begin", {"ahead": True})
    assert [j for j, _ in calls[1:]] == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    assert all(ids == [j * E + e for e in order[j]] for j, ids in calls[1:])
    assert srv.aheads == 12 and srv.landed > 0
    assert np.array_equal(la.view(np.uint32), lr.view(np.uint32))
    assert np.array_equal(lb.view(np.uint32), lr.view(np.uint32))
    assert a.server.misses - m0[0] < b.server.misses - m0[1]
    for t in (5, 6):
        assert np.array_equal(a.step(t).view(np.uint32), ref.step(t).view(np.uint32))
    a.layer_ahead, got = True, []                # True: each layer's experts in index order
    a._send_ahead(lambda j, ids: got.append((j, ids)), 2)
    assert got == [(2, list(range(2 * E, 3 * E)))]


def _no_mixers(W):
    """W with every mixer's output projection zero: a layer's MoE block reads the layer's input
    (x + 0), so its route is the hint the layer before posts."""
    return {k: (np.zeros_like(v) if k.endswith(("self_attn.o_proj.weight",
                                                "linear_attn.out_proj.weight")) else v)
            for k, v in W.items()}


def _record_hints(eng):
    """eng's server's hints (each one's global ids, a list) and the requests named after
    begin_prefill (ExpertServer.history after its call: it serves the last token step's
    request first, settle)."""
    srv, hints, mark = eng.server, [], []
    hint, begin = srv.hint, srv.begin_prefill
    srv.hint = lambda ids: (hints.append(list(ids)), hint(ids))
    srv.begin_prefill = lambda **kw: (begin(**kw), mark.append(len(srv.history)))
    srv.history = []
    return hints, lambda: srv.history[mark[0]:]


def test_layer_runs_hint_the_next_layer_and_change_no_logit(tiny):
    """Engine(layer_ahead="hint"): each layer run but the last layer's ends with the next
    layer's hint (moe.moe_hint_rows: that layer's router on the run's output rows, one line of
    R k ids), served as the next run waits on its fence; begin_prefill gets AHEAD_PART and
    every ahead_layer call names no expert. The logits and the decode steps after equal token
    by token's bit for bit. With the mixers' output projections zero a layer's route reads the
    layer's input, so each hint is the next layer's request for the same rows, ids in their
    order: 20 rows, the first 3 token by token (conv_k - 1), then runs of 2 and one of 1."""
    from opentpu.llm.qwen3 import AHEAD_PART
    spec, W = _untied(tiny)
    L = len(KINDS)
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=2 * K)
    a, ref = (Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True, experts=2 * K, **kw)
              for kw in ({"layer_major": 2, "layer_ahead": "hint"}, {}))
    calls, ahead, begin = [], a.server.ahead_layer, a.server.begin_prefill
    a.server.ahead_layer = lambda j, ids: (calls.append(list(ids)), ahead(j, ids))
    a.server.begin_prefill = lambda **kw: (calls.append(kw), begin(**kw))
    toks = [int(t) for t in np.random.default_rng(9).integers(0, 1000, 20)]
    la, lr = a.prefill(toks), ref.prefill(toks)
    assert calls[0] == {"ahead": True, "part": AHEAD_PART} and calls[1:] == [[]] * L
    runs = 9                                    # (rows 3..19 in runs of 2, the last of 1)
    assert a.server.hints == (L - 1) * runs
    assert np.array_equal(la.view(np.uint32), lr.view(np.uint32))
    for t in (5, 6):
        assert np.array_equal(a.step(t).view(np.uint32), ref.step(t).view(np.uint32))
    z = Engine(spec, _no_mixers(W), cap=256, cfg=cfg, rows=1, resident=True, experts=2 * K,
               layer_major=2, layer_ahead="hint")
    hints, requests = _record_hints(z)
    z.prefill(toks)
    got = requests()
    assert len(got) == L * runs and len(hints) == (L - 1) * runs
    assert hints == got[runs:] and any(len(h) > K for h in hints)  # (two rows' ids)


def test_layer_hints_queue_the_next_layer_and_its_experts_land_early(tiny):
    """The server's side of layer_ahead "hint" (ExpertServer.hint with ahead_layer's queue on
    the hinted layer: its ids to the end of the queue, docs/offload.md 13.4): with the link's
    idle time before each run (polls until none sends: the ISA simulator polls only while a
    run waits) the hinted experts land before the next layer's runs ask for them, so fewer
    misses than layer-major runs without the layer ahead; the logits and the decode steps
    after equal theirs bit for bit."""
    spec, W = _untied(tiny)
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=2 * K)
    a, b = (Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True, experts=2 * K,
                   layer_major=2, **kw) for kw in ({"layer_ahead": "hint"}, {}))
    srv, run = a.server, a.backend.run

    def idle_then_run(*args, **kw):
        while srv.poll():
            pass
        return run(*args, **kw)
    a.backend.run = idle_then_run
    toks = [int(t) for t in np.random.default_rng(9).integers(0, 1000, 20)]
    m0 = a.server.misses, b.server.misses
    la, lb = a.prefill(toks), b.prefill(toks)
    assert srv.hinted_ahead > 0 and srv.landed > 0 and not srv.queue
    assert a.server.misses - m0[0] < b.server.misses - m0[1]
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    for t in (5, 6):
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32))


def test_layer_ahead_on_a_live_card_with_idle_parts_v2(tiny, tmp_path):
    """Layer-major prefill with the layer ahead from the runs' hints (layer_ahead "hint") beside
    a card that computes while the host serves (_LiveCard, CHASH, a split pool through one
    BoardDram), its idle parts read ahead, one call each, and held back near a run's expected
    end (docs/offload.md 13.12): ahead_layer's experts take their slots as they are staged,
    each ahead_layer call and request drops a staged part. The ISA simulator's token-by-token
    logits and the decode step after them, bit for bit."""
    from test_lfm2_moe import _LiveCard
    from opentpu.host.board import BoardBackend
    from opentpu.host.offload import BoardDram
    from opentpu.isasim import board_config
    spec, W = _untied(tiny)
    cfg = board_config(DRAM_BYTES=1 << 25)
    card = _LiveCard.make(cfg, chash=True)
    card.threaded = True
    kw = dict(cap=256, cfg=cfg, rows=1, resident=True, experts=2 * K)
    isa = Engine(spec, W, **kw, embed_host=False)
    brd = Engine(spec, W, **kw, layer_major=2, layer_ahead="hint",
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=card),
                 pool_file=tmp_path / "pool.bin")
    s = brd.server
    assert isinstance(s.mem, BoardDram)
    s.read_ahead = s.halt_aware = True
    toks = [int(t) for t in np.random.default_rng(11).integers(0, 1000, 13)]
    la, lb = isa.prefill(toks), brd.prefill(toks)
    assert card.error is None, card.error
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    assert s.hinted_ahead > 0 and s.mem.staged > 0 and not s.mem._held
    assert s.holds > 0          # (run_clock seen during the runs: 0 on the card in pfv2, fixed)
    t = int(np.argmax(la))
    a, b = isa.step(t), brd.step(t)
    assert card.error is None, card.error
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


def _hinted(spec):
    return replace(spec, moe=replace(spec.moe, hint=True))


def test_hints_move_experts_early_and_change_no_logit(tiny):
    """MoESpec.hint: each layer's router on its input, before the mixer, posts its k best (the
    host gives them slots at once, by decayed use). k + 1 slots per layer: per-token steps and
    the card's generate loop give the logits and tokens of the same engine without hints bit
    for bit; every MoE layer posts one hint per token, and the hinted experts the route then
    names are sent when it asks (the ISA simulator calls the host only when the card waits:
    no idle polls)."""
    _, W, spec = tiny
    L = len(KINDS)
    cfg = device_config(spec, 512, rows=1, lookup=True, S=1, experts=K + 1)
    a, b = (Engine(s, W, cap=512, cfg=cfg, rows=1, resident=True, experts=K + 1)
            for s in (_hinted(spec), spec))
    assert a.server.policy == "lfu"
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 8)]
    for t in toks:
        x, y = a.step(t), b.step(t)
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32)), a.pos
    assert a.server.hints == len(toks) * L and b.server.hints == 0
    assert a.server.promoted > 0 and a.server.prefetched == 0
    t0 = int(np.argmax(x))
    assert a.generate_card(t0, 8, stop_ids=[]) == b.generate_card(t0, 8, stop_ids=[])
    assert a.server.hints >= (len(toks) + 7) * L


def test_hinted_experts_on_their_way_wait_on_their_tags_beat_by_beat(tiny):
    """docs/offload.md 10.11 with hints over an adversarial link (tests/beat_link.py: the
    server's writes land a 64-byte beat at a time, the machine running as soon as one of its
    WAITWs holds): a route that names an expert still on its way finds its slot in the answer
    and waits for its tag (the rest of it is sent then); per-token steps and the card's
    generate loop give the logits and tokens of the engine without hints bit for bit."""
    from beat_link import BeatLink
    _, W, spec = tiny
    cfg = device_config(spec, 512, rows=1, lookup=True, S=1, experts=K + 1)
    a, b = (Engine(s, W, cap=512, cfg=cfg, rows=1, resident=True, experts=K + 1)
            for s in (_hinted(spec), spec))
    link = BeatLink(a)
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 6)]
    for t in toks:
        x, y = a.step(t), b.step(t)
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32)), a.pos
    assert a.server.promoted > 0 and link.held > 0
    t0 = int(np.argmax(x))
    assert a.generate_card(t0, 6, stop_ids=[]) == b.generate_card(t0, 6, stop_ids=[])


@pytest.mark.parametrize("caps,parts", [((0, 0), "v1"), ((1, 1), "v1"), ((0, 0), "v2")])
def test_the_live_card_takes_hinted_experts_on_the_links_idle_time(tiny, tmp_path, caps, parts):
    """Hints beside a card that computes while the host works (tests/test_lfm2_moe.py's
    _LiveCard, CHASH's map, a split-format pool), with the embedding table on the host: one
    BoardDram for the experts and the rows. The host sends the hinted experts in parts (64 KiB
    here) while the card runs its mixers, the rest of one the route names at once. Resident
    steps and the card's generate loop give the ISA simulator's logits and tokens with the
    table on the card and no hints bit for bit, with hinted experts landed before the route
    asked; uncapped, and with each hint's best expert only (hint_n, hint_top: 12.7); v2:
    the idle parts read ahead, one call each, and held back near a run's expected end (13.12)."""
    from test_lfm2_moe import _LiveCard
    from opentpu.host.board import BoardBackend
    from opentpu.host.offload import RUN, BoardDram
    from opentpu.isasim import board_config
    spec, W = _untied(tiny)
    cfg = board_config(DRAM_BYTES=1 << 25)
    card = _LiveCard.make(cfg, chash=True)
    card.threaded = True
    kw = dict(cap=256, cfg=cfg, rows=1, resident=True, experts=K + 1)
    isa = Engine(spec, W, **kw, embed_host=False)
    brd = Engine(_hinted(spec), W, **kw,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=card),
                 pool_file=tmp_path / "pool.bin")
    assert brd.image.embed_host and not isa.image.embed_host
    assert isinstance(brd.server.mem, BoardDram) and brd.row_server.mem is brd.server.mem
    assert brd.server.L.slot_bytes > 16 * RUN
    brd.server.part = 16 * RUN
    brd.server.hint_n, brd.server.hint_top = caps
    brd.server.read_ahead = brd.server.halt_aware = parts == "v2"
    toks = [int(t) for t in np.random.default_rng(6).integers(0, 1000, 6)]
    for t in toks:
        a, b = isa.step(t), brd.step(t)
        assert card.error is None, card.error
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), brd.pos
    t0 = int(np.argmax(a))
    got = brd.generate_card(t0, 8, stop_ids=[])
    assert card.error is None, card.error
    assert got == isa.generate_card(t0, 8, stop_ids=[])
    s = brd.server
    assert s.hints >= 13 * len(KINDS) and s.prefetched > 0 and s.mem.direct > 0
    assert (s.mem.staged > 0) == (parts == "v2")
    assert s.misses < isa.server.misses and brd.row_server.seq >= 7


def test_moe_card_streams_as_the_resident_run(tiny, tmp_path, monkeypatch):
    """tools/offload/moe_card.py end to end on the ISA simulator (a tiny Qwen3.5-MoE checkpoint,
    the card's sessions' tool): with 2 slots a layer, from a split pool file with --hint-trace
    and with --legacy-serve, its tokens and prefill logits' sha are the run's with every expert
    in a slot. Its wrappers of ExpertServer's methods (serve, the memory's) follow their
    signatures, so a change there fails here, not on a card (the reference run of
    2026-10-02)."""
    import importlib.util
    import json
    import sys
    from pathlib import Path

    from safetensors.numpy import save_file
    m, W, _ = tiny
    d = tmp_path / "tiny"
    m.config.save_pretrained(d)                     # (the checkpoint's fused expert tensors)
    save_file({k: np.ascontiguousarray(v) for k, v in W.items()}, str(d / "model.safetensors"))
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"ids": [11, 222, 333, 444, 55, 66], "tokens": [0] * 6}))
    tool = Path(__file__).resolve().parents[1] / "tools" / "offload" / "moe_card.py"
    spec = importlib.util.spec_from_file_location("moe_card", tool)
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)

    def run(name, *opts):
        out = tmp_path / f"{name}.json"
        monkeypatch.setattr(sys, "argv", ["moe_card", str(d), "--check", str(ref), "-n", "6",
                                          "--out", str(out), *opts])
        mc.main()
        r = json.loads(out.read_text())
        return r["tokens"], r["prefill_logits_sha"], r["misses"], r
    want = run("resident", "--experts", "0")
    got = run("split", "--experts", "2", "--pool", str(tmp_path / "pool.bin"),
              "--hint-trace", str(tmp_path / "trace.json"))
    old = run("legacy", "--experts", "2", "--legacy-serve")
    capped = run("capped", "--experts", "3", "--hints", "on", "--hint-drop", "--hint-n", "1",
                 "--hint-top", "1", "--idle-parts", "ra")   # (docs/offload.md 12.7, 13.12)
    assert want[2] == 0 and got[2] > 0 and old[2] > 0
    assert got[:2] == old[:2] == want[:2] == capped[:2]
    h = capped[3]["hints"]
    assert (h["n"], h["top"], h["drop"]) == (1, 1, True) and h["served"] > 0
    assert capped[3]["idle_parts"]["mode"] == "ra" and got[3]["idle_parts"]["mode"] == "v2"
    assert json.loads((tmp_path / "trace.json").read_text())       # (the decode's timeline)


@pytest.mark.parametrize("wformat", ["fp4", "nvfp4"])
def test_pack_pool_tool_packs_the_images_experts(tiny, tmp_path, monkeypatch, wformat):
    """tools/offload/pack_pool.py: init then two workers pack every expert of a checkpoint as the
    image's split slot bytes, each marked packed; send | recv into a second pool, on the host of
    the checkpoint without its experts (strip_experts.py), gives the same file. In fp4 and in
    nvfp4 (LFM2.5-8B-A1B's experts, streamed.FORMATS), whose key names the format."""
    import io
    import json
    import runpy
    import sys
    from pathlib import Path

    from safetensors.numpy import save_file

    from opentpu.host.offload import to_split
    from opentpu.llm.qwen3 import LazyWeights

    m = tmp_path / "model"                          # HF's tensor names, as LazyWeights reads them
    tiny[0].config.save_pretrained(m)
    save_file({k: np.ascontiguousarray(v) for k, v in tiny[1].items()},
              str(m / "model.safetensors"))
    tools = Path(__file__).resolve().parents[1] / "tools" / "offload"
    tool = str(tools / "pack_pool.py")

    def run(*args, stdin=None, stdout=None):
        monkeypatch.setattr(sys, "argv", ["pack_pool.py", *map(str, args)])
        if stdin is not None:
            monkeypatch.setattr(sys, "stdin", stdin)
        if stdout is not None:
            monkeypatch.setattr(sys, "stdout", stdout)
        runpy.run_path(tool, run_name="__main__")
        monkeypatch.undo()

    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    wf = ("--wformat", wformat)
    for p in (a, b):
        run(m, p, "init", *wf)
    run(m, a, "pack", 0, 2, *wf)
    run(m, a, "pack", 1, 2, *wf)
    img = runpy.run_path(tool)["image"](str(m), wformat)
    assert img.fmt.wformat == wformat
    assert json.loads(Path(str(a) + ".key").read_text())["format"] == wformat
    L = img.offload
    n, slot = L.layers * L.E, L.slot_bytes
    assert np.fromfile(str(a) + ".packed", np.uint8).tolist() == [1] * n
    W, pool = LazyWeights(m), np.fromfile(a, np.uint8).reshape(n, slot)
    for g in range(n):
        assert np.array_equal(pool[g], np.asarray(to_split(img.expert(W, g))).view(np.uint8)), g
    ids = tmp_path / "ids.txt"
    ids.write_text("\n".join(map(str, range(n - 1, -1, -1))))
    out = tmp_path / "stream"
    with open(out, "wb") as f:
        run(m, "-", "send", ids, *wf, stdout=f)
    stripped = tmp_path / "stripped"
    monkeypatch.setattr(sys, "argv", ["strip_experts.py", str(m), str(stripped)])
    runpy.run_path(str(tools / "strip_experts.py"), run_name="__main__")
    monkeypatch.undo()
    from safetensors import safe_open
    with safe_open(str(stripped / "model.safetensors"), "np") as h:
        assert set(h.keys()) == {k for k in tiny[1] if ".experts." not in k}
    with open(out, "rb") as f:
        run(stripped, b, "recv", *wf, stdin=f)
    assert a.read_bytes() == b.read_bytes()
    assert Path(str(b) + ".packed").read_bytes() == bytes([1] * n)


def test_pool_key_refuses_another_images_pool(tiny, tmp_path, monkeypatch, capsys):
    """An expert pool's key (`<pool>.key`, moe.pool_key: the quantizer's code, the format, D,
    the layout, the checkpoint's config.json), written with a new pool (pack_pool init, or the
    Engine's), is checked whenever the pool is opened: a pool of the same size packed in
    another format (int4 against fp4), by another quantizer or from another checkpoint is
    refused with how to repack it, and so is a pool of the MXU before MX (a key 1, or no key
    at all: packed before keys) until pack_pool stamp writes its key (after `sums`); info tells
    int8 from 4-bit by the key, or without one by the slot size. send puts the key first in its
    stream, and recv refuses a stream packed for another image."""
    import json
    import runpy
    import sys

    from safetensors.numpy import save_file

    from opentpu.llm.qwen3 import LazyWeights

    m = tmp_path / "model"
    tiny[0].config.save_pretrained(m)
    save_file({k: np.ascontiguousarray(v) for k, v in tiny[1].items()},
              str(m / "model.safetensors"))
    tool = str(Path(__file__).resolve().parents[1] / "tools" / "offload" / "pack_pool.py")

    def run(*args, stdin=None, stdout=None):
        monkeypatch.setattr(sys, "argv", ["pack_pool.py", *map(str, args)])
        if stdin is not None:
            monkeypatch.setattr(sys, "stdin", stdin)
        if stdout is not None:
            monkeypatch.setattr(sys, "stdout", stdout)
        try:
            runpy.run_path(tool, run_name="__main__")
        finally:
            monkeypatch.undo()

    spec = Spec.from_hf(m)
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 3)]

    def engine(pool, wformat="fp4"):
        cfg = device_config(spec, 256, S=1, experts=K, wformat=wformat)
        eng = Engine(spec, LazyWeights(m), cap=256, cfg=cfg, experts=K, wformat=wformat,
                     pool_file=pool)
        return np.array([eng.step(t) for t in toks])

    a = tmp_path / "a.bin"
    run(m, a, "init")
    capsys.readouterr()
    kf = Path(str(a) + ".key")
    key = json.loads(kf.read_text())
    assert (key["format"], key["D"], key["model"]) == ("fp4", 128, "model")
    want = engine(a)
    assert "warning" not in capsys.readouterr().err
    good = kf.read_text()
    for field, edit, wformat in (("format", None, "int4"),
                                 ("quant", dict(key, quant="0" * 32), "fp4"),
                                 ("layout", dict(key, layout={**key["layout"], "C": 128}), "fp4")):
        if edit is not None:
            kf.write_text(json.dumps(edit))
        with pytest.raises(ValueError, match=f"{field} differ.*Repack it: remove .*pack_pool"):
            engine(a, wformat)
        kf.write_text(good)
    cj = m / "config.json"
    conf = cj.read_text()
    cj.write_text(conf.replace("{", '{"_another": 1,', 1))      # another checkpoint's config
    with pytest.raises(ValueError, match="checkpoint differ"):
        engine(a)
    cj.write_text(conf)
    old = {k: v for k, v in key.items() if k != "numerics"}     # a key 1 pool: before MX
    kf.write_text(json.dumps(dict(old, key=1)))
    with pytest.raises(ValueError, match="key, numerics differ.*Repack it"):
        engine(a)
    kf.unlink()                                         # a pool packed before keys
    with pytest.raises(ValueError, match="has no key file.*before MX.*stamp"):
        engine(a)
    run(m, a, "info")
    assert out_line(capsys) == "4-bit (fp4 or int4: the same size)"
    run(m, a, "stamp")
    assert json.loads(kf.read_text()) == key
    with pytest.raises(SystemExit, match="has a key already: this one"):
        run(m, a, "stamp")
    assert np.array_equal(engine(a).view(np.uint32), want.view(np.uint32))
    assert "warning" not in capsys.readouterr().err
    with pytest.raises(SystemExit, match="not this model's pool in this format"):
        run(m, a, "stamp", "--wformat", "int8")
    b = tmp_path / "b.bin"                              # int8: info by its key, then its size
    run(m, b, "init", "--wformat", "int8")
    capsys.readouterr()
    run(m, b, "info", "--wformat", "int8")
    out = capsys.readouterr().out
    assert "key: int8, D 128" in out and "the same key" in out
    Path(str(b) + ".key").unlink()
    run(m, b, "info")
    assert out_line(capsys) == "int8"
    run(m, a, "info")
    assert "key: fp4" in capsys.readouterr().out
    stream, c = tmp_path / "stream", tmp_path / "c.bin"      # send | recv: the key goes first
    ids = tmp_path / "ids.txt"
    ids.write_text("0\n5\n")
    with open(stream, "wb") as f:
        run(m, "-", "send", ids, stdout=f)
    run(m, c, "init", "--wformat", "int4")              # the same size, another format
    with open(stream, "rb") as f, pytest.raises(SystemExit, match="format differ"):
        run(m, c, "recv", "--wformat", "int4", stdin=f)
    assert not np.fromfile(str(c) + ".packed", np.uint8).any()


def out_line(capsys):
    """The format info printed for a pool without a key (its last line's, after the colon)."""
    return capsys.readouterr().out.strip().splitlines()[-1].split(" bytes: ", 1)[1]


@pytest.mark.parametrize("ahead", ["index", "hint"])
def test_moe_card_writes_the_prompts_timeline(tiny, tmp_path, monkeypatch, ahead):
    """tools/offload/moe_card.py --layer-major 2 --layer-ahead index --prefill-trace (ISA): the
    prompt's timeline holds each run (the embed runs, the layer runs by key, the head last) and
    each request with its ids and misses, as many as the result's prefill_time and
    prefill_requests count; the layer ahead called once a MoE layer. With --layer-ahead hint
    (--ahead-part 2048 KiB) each layer run but the last layer's posts a hint, an "h" event."""
    import json
    import pickle
    import runpy
    import sys
    from pathlib import Path

    from safetensors.numpy import save_file

    m = tmp_path / "model"
    tiny[0].config.save_pretrained(m)
    save_file({k: np.ascontiguousarray(v) for k, v in tiny[1].items()},
              str(m / "model.safetensors"))
    spec = tiny[2]
    cfg = device_config(replace(spec, embed="int8"), 256, rows=1, lookup=True, S=1,
                        experts=2 * K)
    (tmp_path / "cfg.pkl").write_bytes(pickle.dumps(cfg))
    ids = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 7)]
    (tmp_path / "ref.json").write_text(json.dumps({"ids": ids, "tokens": []}))
    tr, out = tmp_path / "tr.json", tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [
        "moe_card.py", str(m), "--check", str(tmp_path / "ref.json"), "-n", "1", "--cap", "256",
        "--cfg", str(tmp_path / "cfg.pkl"), "--experts", str(2 * K), "--layer-major", "2",
        "--layer-ahead", ahead, "--prefill-trace", str(tr), "--out", str(out)]
        + (["--ahead-part", "2048"] if ahead == "hint" else []))
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "offload" / "moe_card.py"),
                   run_name="__main__")
    r, t = json.loads(out.read_text()), json.loads(tr.read_text())
    keys = [x[0] for x in t["runs"]]
    pt = r["prefill_time"]
    assert keys[-1] == "head" and pt["runs"] == len(keys) == pt["layer_runs"]
    li = [k[0] for k in keys[:-1]]
    assert li.count(-1) == len(ids) - 1                         # the embed runs, a token each
    assert sorted(set(li) - {-1}) == list(range(len(KINDS)))
    assert all(x[1] <= x[2] for x in t["runs"])                 # started, then done
    assert len(t["requests"]) == len(t["misses"]) == pt["requests"] == r["prefill_requests"]
    assert sum(t["misses"]) == r["prefill_misses"]
    assert r["layer_ahead"] == ahead and pt["ahead"]["calls"] == len(KINDS)    # (one chunk)
    hints = [e for e in t["events"] if e[2] == "h"]
    assert all(e[2] in "dh" for e in t["events"])
    assert len(hints) == pt["ahead"]["hints"] == (len(KINDS) - 1) * li.count(0) * (ahead == "hint")
    assert r["ahead_part"] == (2 << 20 if ahead == "hint" else None)


def test_moe_card_runs_the_expert_major_prefill(tiny, tmp_path, monkeypatch):
    """tools/offload/moe_card.py --layer-major 2 --expert-major --ahead-part 2048 --prefill-trace
    (ISA; docs/offload.md 13.13; --expert-major: these slots leave the default no room, 13.15):
    the prompt's timeline holds the embed runs, each MoE layer's runs then its expert run (key
    ("x", layer)), the head last; no request in the prompt, a need line a layer run ("n"
    events), every need landed during the expert runs (none drained), each in one idle-poll part
    ("p" events: --ahead-part is more than a slot). The decode's misses a token are counted from
    its first step after the prompt's (none in the prompt's requests is not a token by token
    prompt)."""
    import json
    import pickle
    import runpy
    import sys
    from pathlib import Path

    from safetensors.numpy import save_file

    m = tmp_path / "model"
    tiny[0].config.save_pretrained(m)
    save_file({k: np.ascontiguousarray(v) for k, v in tiny[1].items()},
              str(m / "model.safetensors"))
    spec = tiny[2]
    cfg = device_config(replace(spec, embed="int8"), 256, rows=1, lookup=True, S=1,
                        experts=2 * K)
    (tmp_path / "cfg.pkl").write_bytes(pickle.dumps(cfg))
    ids = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 7)]
    (tmp_path / "ref.json").write_text(json.dumps({"ids": ids, "tokens": []}))
    tr, out = tmp_path / "tr.json", tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [
        "moe_card.py", str(m), "--check", str(tmp_path / "ref.json"), "-n", "2", "--cap", "256",
        "--cfg", str(tmp_path / "cfg.pkl"), "--experts", str(2 * K), "--layer-major", "2",
        "--expert-major", "--ahead-part", "2048", "--prefill-trace", str(tr), "--out", str(out)])
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "offload" / "moe_card.py"),
                   run_name="__main__")
    r, t = json.loads(out.read_text()), json.loads(tr.read_text())
    keys = [x[0] for x in t["runs"]]
    pt = r["prefill_time"]
    assert keys[-1] == "head" and pt["runs"] == len(keys) == pt["layer_runs"]
    li = [k[0] for k in keys[:-1]]
    assert li.count(-1) == len(ids) - 1                         # the embed runs, a token each
    assert [k[1] for k in keys if k[0] == "x"] == list(range(len(KINDS)))   # (one chunk)
    runs = li.count(0)                                          # a layer's runs (2 rows each)
    for j in range(len(KINDS)):                                 # its expert run right after
        assert keys[max(i for i, k in enumerate(keys) if k[0] == j) + 1] == ["x", j]
    assert not t["requests"] and pt["requests"] == 0 and r["prefill_requests"] == 0
    assert r["misses_per_token_decode"] is not None and len(r["misses_per_token"]) == 2
    assert len(r["misses_per_request_decode"]) == len(KINDS)    # (the second step's)
    assert r["expert_major"] and r["ahead_part"] == 2 << 20 and pt["ahead"] is None
    assert all(e[2] in "np" for e in t["events"])              # need lines, idle parts
    nd = pt["need"]
    assert nd["need_lines"] == len(KINDS) * runs == sum(e[2] == "n" for e in t["events"])
    assert nd["needs_landed"] == nd["needs_queued"] > 0 and nd["drained"] == 0
    assert sum(e[2] == "p" for e in t["events"]) == nd["needs_landed"]    # (a part a need)


def _scratch(srv, monkeypatch):
    """The ExpertServer's begin_prefill calls, each one's scratch as it returned and the idle
    polls' part: (base, slots, part) (an expert-major prefill's, docs/offload.md 13.14: the
    slot area's first slots, set aside until end_prefill)."""
    got, begin = [], srv.begin_prefill

    def begin_prefill(**kw):
        base = begin(**kw)
        got.append((base, list(srv.scratch_slots), srv.part))
        return base
    monkeypatch.setattr(srv, "begin_prefill", begin_prefill)
    return got


def test_an_expert_runs_timeout_names_where_its_needs_stand(tiny, monkeypatch):
    """Engine(expert_major=True) with a server that sends no need: the first expert run stops
    at its WAITW's timeout (the ISA's 'never holds', the card's 'a WAITW timed out'), and the
    error names the run's layer and the server's need_report (docs/offload.md 13.14): the
    layer's experts without an entry, each still queued."""
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    spec, W = _untied(tiny)
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=6)
    a = Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True, experts=6, layer_major=2,
               expert_major=True)
    monkeypatch.setattr(a.server, "_next_need", lambda: False)
    toks = [int(t) for t in np.random.default_rng(12).integers(0, 1000, 12)]
    want = (r"expert run of layer 0: .*never holds.* \(layer 0: \d+ experts needed, [1-9]\d* "
            r"without an entry; \d+ \(expert \d+\): queued")
    with pytest.raises(RuntimeError, match=want):
        a.prefill(toks)


@pytest.mark.parametrize("kw,wf,C,S", [
    (dict(MCOLS=4, PAIR=True), "fp4", 4, 2),        # the card: 2 rows at the stream's rate, 4
    (dict(MCOLS=4), "fp4", 4, 0),                   # at twice its MXU time; a short pass of 2
    (dict(MCOLS=4, PAIR=True), "int8", 4, 0),
    (dict(MCOLS=4, PAIR=True, ACT_ROWS=2), "fp4", 2, 0),
    (dict(MCOLS=8, PAIR=True), "fp4", 8, 4),
    (dict(MCOLS=16, ACT_ROWS=64, PAIR=True), "fp4", 16, 8),
    (dict(MCOLS=16, ACT_ROWS=64, PAIR=True), "int8", 32, 16)])
def test_em_pass_rows_from_the_configuration(kw, wf, C, S):
    """moe.em_pass_rows: an expert-major pass's entries C from the configuration alone (docs/
    offload.md 13.18), the least time an entry (an expert's stream a pass, or the MXU's time
    when longer), the fewest on a tie; em_short_rows: the most below C at a lower MXU cost (an
    expert's last pass when it has that few left)."""
    from opentpu.isasim import Config
    from opentpu.llm import moe as MO
    cfg = Config(**kw)
    assert MO.em_pass_rows(cfg, wf) == C and MO.em_short_rows(cfg, wf, C) == S
    assert MO.em_pass_order(cfg, wf)[0] == C and len(MO.em_pass_order(cfg, wf)) == cfg.act_rows


def test_em_scratch_keeps_the_expert_runs_experts_past_the_records():
    """moe.em_scratch: the image's prefill rows' records at any first chunk when the slots past
    them are the expert runs' (moe.em_expert_slots: the pipelined passes' slot bound) and leave
    the needs room (3 E), so the server never lands a run's expert in a slot its stores may
    write; else the chunk's rows' (and the runs' bound every slot)."""
    from types import SimpleNamespace

    from opentpu.llm import moe as MO
    sl = [5000 + 1000 * i for i in range(40)]
    L = SimpleNamespace(pitch=1000, E=1, all_slots=lambda: sl)
    assert MO.em_expert_slots(L, 512 * 70, (5000, 45000)) == (41000, 45000)    # 36 + 4 slots
    assert MO.em_scratch(L, 7, 512, 70) == 512 * 70
    assert MO.em_slots(L, MO.em_scratch(L, 7, 512, 70))[-1] < 41000
    L2 = SimpleNamespace(pitch=1000, E=2, all_slots=lambda: sl)                 # 4 < 3 E
    assert MO.em_expert_slots(L2, 512 * 70, (5000, 45000)) is None
    assert MO.em_scratch(L2, 7, 512, 70) == 7 * 70


@pytest.mark.parametrize("R,skew,wformat,b,C", [
    (2, False, "int8", False, None), (1, False, "int8", False, None),
    (2, True, "fp4", True, None), (2, False, "fp4", True, None),
    (2, True, "fp4", True, 2), (2, False, "fp4", True, 3), (2, False, "int8", False, 1)])
def test_expert_major_prefill_is_bit_exact(tiny, R, skew, wformat, b, C, monkeypatch):
    """Expert-major MoE in the layer-major prefill (Engine(expert_major=True), docs/offload.md
    13.11): each layer's runs route their rows, post the experts they chose as need lines and
    run the shared expert; then one expert run a layer (the bucketing into passes of up to C
    entries of one expert, each pass waiting on its directory entry); the next layer's runs and
    the head sum the outputs. The logits, the DeltaNet states and windows and the KV cache,
    and the decode steps after, equal token by token's bit for bit: 40 tokens, the first 3
    token by token (conv_k - 1), then chunks of 24 and 13 rows (odd counts), 6 slots a layer
    behind the ExpertServer (each need landing while an expert run waits on its entry, none
    left for end_prefill; the scratch the first slots), the embedding rows from the host's
    table. skew: a router that knows expert 0 alone (its other rows zero), so experts 0 / 1 / 2
    take every row: long runs of passes of one expert. b: build B's PAIR, DSTEP and STREAM (a
    pass's rows one MM's). C: the passes' entries (Engine.em_pass; None: the configuration's,
    docs/offload.md 13.18: 4 with b's MCOLS 4 and PAIR, a short pass of 2; 8 for int8 at the
    default MCOLS 8): each row's math the same, only grouped otherwise, so every C equals token
    by token's. Today's runs of R rows (tile_runs False; the tiles, Qwen3.5-MoE's default:
    tests/test_em_tile.py)."""
    from opentpu.llm import qwen35 as Q35
    from opentpu.llm.moe import em_pass_rows, em_record, em_slots
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    spec, W = _untied(tiny)
    if skew:
        W = dict(W)
        for i in range(len(KINDS)):
            n = f"model.layers.{i}.mlp.gate.weight"
            W[n] = np.concatenate([W[n][:1], np.zeros_like(W[n][1:])])
    kw = dict(MCOLS=4, PAIR=True, DSTEP=True, STREAM=True) if b else {}
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=6, wformat=wformat, **kw)
    a, ref = (Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True, experts=6,
                     wformat=wformat, **kw)
              for kw in ({"layer_major": R, "expert_major": True, "tile_runs": False}, {}))
    assert em_pass_rows(cfg, wformat) == (4 if b else 8)
    a.em_pass = C
    sc = _scratch(a.server, monkeypatch)
    toks = [int(t) for t in np.random.default_rng(11).integers(0, 1000, 40)]
    la, lb = a.prefill(toks), ref.prefill(toks)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    img, srv = a.image, a.server
    L = srv.L
    assert img.embed_host and a.row_server is not None     # (the 35B's --embed-table host)
    assert sc == [(L.slots[0][0], em_slots(L, 24 * em_record(spec.hidden, K)), a.ahead_part)]
    assert len(sc[0][1]) == -(-24 * em_record(spec.hidden, K) // L.pitch)
    runs = sum(-(-n // R) for n in (24, 13))            # (the two chunks)
    assert srv.need_lines == len(KINDS) * runs
    assert srv.needs_landed == srv.needs_queued > 0 and srv.drained == 0
    assert not srv.scratch_slots and not srv.expert_major
    assert {k for k in a._layer_runs if k[0] == "x"} == \
        {("x", li) + ((C,) if C else ()) for li in range(len(KINDS))}
    dram = [e.backend.machine.slices[0].dram[img.layer0:img.head[0]] for e in (a, ref)]
    assert np.array_equal(*dram)
    t = int(np.argmax(la))
    for _ in range(2):
        ga, gb = a.step(t), ref.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))


@pytest.mark.parametrize("wformat, b", [("int8", False), ("fp4", True)])
def test_waits_memo_is_exact_on_its_programs(tiny, wformat, b, monkeypatch):
    """The waits pass's memoized entries (opentpu/waits.py _MemoPass) are the exact pass's on
    every program the streamed Qwen3.5-MoE runs from 6 slots a layer: expert-major's layer runs
    (DeltaNet and attention) and expert runs, the embedding rows from the host's table, the
    step and resident decode programs, the decode loop's (GEN, RLD; the expert requests' WAITW);
    b: build B's PAIR, DSTEP and STREAM. Their MMs drain at their array's rate
    (compiler.slow_drains)."""
    from conftest import assert_drains_keep_up, assert_waits_exact, waits_programs
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    spec, W = _untied(tiny)
    kw = dict(MCOLS=4, PAIR=True, DSTEP=True, STREAM=True) if b else {}
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=6, wformat=wformat, **kw)
    e = Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True, experts=6, wformat=wformat,
               layer_major=2, expert_major=True)
    progs = waits_programs(e)
    toks = [int(t) for t in np.random.default_rng(11).integers(0, 1000, 40)]
    e.prefill(toks[:30])
    e.step(toks[30])
    e.generate_card(toks[31], 6, stop_ids=[])
    assert any(k[0] == "x" for k in e._layer_runs)
    assert len(progs) >= 4 and assert_waits_exact(progs, e.cfg) > 1000
    assert assert_drains_keep_up(progs, e.cfg) > 0
