"""Qwen3.5-MoE on openTPU with its experts streamed into DRAM slots (docs/offload.md, path (a)):
the card routes (softmax of the k largest logits) and computes every expert and the gated
shared expert; the host's server moves the missing experts. A tiny random model against
Hugging Face transformers and the quantized emulation; a small cache gives the same logits
bit for bit as one holding every expert."""
from dataclasses import replace

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


def _tiny(nk=8):
    """nk DeltaNet key heads (8: one per value head; 4: a pair shares its q and k, as the
    35B's)."""
    torch.manual_seed(0)
    hc = transformers.Qwen3_5MoeTextConfig(
        hidden_size=256, num_hidden_layers=len(KINDS), num_attention_heads=8,
        num_key_value_heads=2, head_dim=256, vocab_size=1000,
        layer_types=["full_attention" if k == "attn" else "linear_attention" for k in KINDS],
        linear_num_key_heads=nk, linear_num_value_heads=8, linear_key_head_dim=128,
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
    spec = Spec(256, KINDS, 8, 2, 256, 64, 8, 128, 128, F, 1000, lin_kheads=nk,
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
    cfg = device_config(spec, 256, S=1, experts=experts, **kw)
    return Engine(spec, W, cap=256, cfg=cfg, experts=experts, **kw)


def test_reference_matches_hf(tiny):
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 24)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(reference_logits(spec, W, toks) - hf).max() < 1e-4


def test_device_follows_hf_and_routes_on_the_card(tiny):
    """The logits follow HF and the emulation; the card's choices in the first layer, whose
    input no routing has touched, are the emulation's wherever its k-th and (k+1)-th logits
    are apart by more than the quantization's reach."""
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 12)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    eng = _engine(spec, W)
    eng.server.history = []
    dev = np.array([eng.step(t) for t in toks])
    c = _cos(dev, hf)
    assert np.median(c) > 0.997 and (c > 0.99).mean() >= 0.75
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


@pytest.mark.parametrize("wformat", ["int8", "fp4"])
def test_small_cache_is_bit_exact(tiny, wformat):
    """k slots per layer: nearly every request misses and streams; the logits do not move."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 8)]
    full = _engine(spec, W, wformat=wformat)
    ref = np.array([full.step(t) for t in toks])
    small = _engine(spec, W, experts=K, wformat=wformat)
    got = np.array([small.step(t) for t in toks])
    assert small.server.misses > len(toks)
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))


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


def test_the_live_card_asks_the_host_for_experts_and_embedding_rows(tiny, tmp_path):
    """The host's two servers beside a card that computes while they work
    (tests/test_lfm2_moe.py's _LiveCard, CHASH's map): the experts from a split-format pool
    and the embedding rows (embed_host) through one BoardDram, its DMA thread writing while the
    card waits. Resident steps and the card's generate loop give the ISA simulator's logits
    and tokens bit for bit, every expert and row request served during the runs."""
    from test_lfm2_moe import _LiveCard
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
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 6)]
    for t in toks:
        a, b = isa.step(t), brd.step(t)
        assert card.error is None, card.error
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), brd.pos
    t0, waits = int(np.argmax(a)), card.waits
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
    dram = [e.backend.machine.slices[0].dram[img.layer0:img.head[0]].copy() for e in (a, ref)]
    for li in range(spec.layers):       # (group-major: but the pairs' gates, a one-row run's
        dn = getattr(img.descriptors(0).layer(li), "dn", None)  # scratch that R rows keep in
        for g in range(dn.nl // dn.og if dn is not None and dn.grouped else 0):    # m.gr)
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


def test_the_live_card_takes_hinted_experts_on_the_links_idle_time(tiny, tmp_path):
    """Hints beside a card that computes while the host works (tests/test_lfm2_moe.py's
    _LiveCard, CHASH's map, a split-format pool), with the embedding table on the host: one
    BoardDram for the experts and the rows. The host sends the hinted experts in parts (64 KiB
    here) while the card runs its mixers, the rest of one the route names at once. Resident
    steps and the card's generate loop give the ISA simulator's logits and tokens with the
    table on the card and no hints bit for bit, with hinted experts landed before the route
    asked."""
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
    assert s.misses < isa.server.misses and brd.row_server.seq >= 7


def test_pack_pool_tool_packs_the_images_experts(tiny, tmp_path, monkeypatch):
    """tools/offload/pack_pool.py: init then two workers pack every expert of a checkpoint as the
    image's split slot bytes, each marked packed; send | recv into a second pool, on the host of
    the checkpoint without its experts (strip_experts.py), gives the same file."""
    import io
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
    for p in (a, b):
        run(m, p, "init")
    run(m, a, "pack", 0, 2)
    run(m, a, "pack", 1, 2)
    img = runpy.run_path(tool)["image"](str(m), "fp4")
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
        run(m, "-", "send", ids, stdout=f)
    stripped = tmp_path / "stripped"
    monkeypatch.setattr(sys, "argv", ["strip_experts.py", str(m), str(stripped)])
    runpy.run_path(str(tools / "strip_experts.py"), run_name="__main__")
    monkeypatch.undo()
    from safetensors import safe_open
    with safe_open(str(stripped / "model.safetensors"), "np") as h:
        assert set(h.keys()) == {k for k in tiny[1] if ".experts." not in k}
    with open(out, "rb") as f:
        run(stripped, b, "recv", stdin=f)
    assert a.read_bytes() == b.read_bytes()
    assert Path(str(b) + ".packed").read_bytes() == bytes([1] * n)
