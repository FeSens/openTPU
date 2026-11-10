"""Batched decode of a Qwen3.5-MoE image (opentpu/llm/batch35.py, Image(batch=N);
docs/batch35.md): each sequence in its own slot (KV caches, DeltaNet states
and windows), a step's rows one per sequence. A tiny random model with the 35B's shape where
it matters (16 DeltaNet value heads, a pair's two sharing a key head; a gated shared expert;
an untied head: a tied tiny model echoes its input): two prompts fed into slots 0 and 1, then
batched steps give each row's logits bit for bit as a single-sequence engine's."""
from dataclasses import replace

import numpy as np
import pytest

from opentpu.compiler import Affine
from opentpu.llm.moe import MoESpec
from opentpu.llm.qwen3 import Engine
from opentpu.llm.qwen35 import Spec

from test_cards import x2_verilator  # noqa: F401 (the RTL tests' Verilator flags)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

KINDS = ("linear", "linear", "attn", "linear", "linear", "attn")
E, K, F = 8, 2, 256


def _tiny(nk=8, nv=16, E=E, K=K, hf=False, H=256, f=F, kinds=KINDS):
    """nv DeltaNet value heads, nk key heads (16 / 8: a pair shares its q and k, as the 35B's
    32 / 16), E experts of which each token picks K, an untied LM head; hidden H, experts and
    shared expert f wide (x2's S = 4: 512, S x D). hf: the HF config too (a checkpoint's)."""
    torch.manual_seed(0)
    hc = transformers.Qwen3_5MoeTextConfig(
        hidden_size=H, num_hidden_layers=len(kinds), num_attention_heads=8,
        num_key_value_heads=2, head_dim=256, vocab_size=1000,
        layer_types=["full_attention" if k == "attn" else "linear_attention" for k in kinds],
        linear_num_key_heads=nk, linear_num_value_heads=nv, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=False,
        max_position_embeddings=4096, rms_norm_eps=1e-6, num_experts=E,
        num_experts_per_tok=K, moe_intermediate_size=f, shared_expert_intermediate_size=f,
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
    spec = Spec(H, tuple(kinds), 8, 2, 256, 64, nv, 128, 128, f, 1000, lin_kheads=nk,
                moe=MoESpec(E=E, k=K, ffn=f, rule="softmax", shared=f), tied=False)
    return (W, spec, hc) if hf else (W, spec)


@pytest.fixture(scope="module")
def tiny():
    return _tiny()


def _cfg(monkeypatch, S=1):
    """bcu1525-wg (DSTEP, weight groups) at S slices, the image below 4 GiB with every expert
    resident (no expert server; S > 1: tensor-parallel experts)."""
    from opentpu import target as T
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    return T.config("bcu1525-wg", S=S, DRAM_BYTES=1 << 33)


def _at(kv) -> dict:
    """A KV cache's arrays' addresses, head by head."""
    return {h: {k: Affine.of(v).static() for k, v in e.items()} for h, e in kv.heads.items()}


def _slot_step(eng, s: int, pos: int, tok: int) -> np.ndarray:
    """Sequence s's token at pos through the decode step at its slot (compile_step(slot=))."""
    kw = {"tok": int(tok)} if eng.device_inputs else {}
    progs = eng.image.compile_step(pos, eng.block, slot=s, **kw)
    return eng._run_rows([(s, pos)], [tok], [0], progs)[0]


def test_slots_layout(tiny, monkeypatch):
    """Image(batch=N): N DeltaNet state and window slots and N KV caches a layer, the low area
    growing by N - 1 sequences' worth; batch 1 lays out as before; slot views see their own
    caches (kv, kvs) and the batched view every sequence's (kv_slots)."""
    W, spec = tiny
    cfg = _cfg(monkeypatch)
    one, two = (spec.image(cfg, 256, b, 2) for b in (1, 2))
    assert one.slots == 1 and two.slots == 2 and two.batch == 2
    assert two.kv_bytes == 2 * one.kv_bytes
    li = KINDS.index("attn")
    a0, a1 = (two.descriptors(0, s).layer(li) for s in (0, 1))
    assert _at(a0.kv_slots[1]) == _at(a1.kv) and a1.kvs == [a1.kv]
    assert _at(a0.kv) != _at(a1.kv) and len(a0.kv_slots) == 2
    dn0, dn1 = (two.descriptors(0, s).layer(0).dn for s in (0, 1))
    assert dn1.state(0, 0).base != dn0.state(0, 0).base
    assert dn1.window(0).base != dn0.window(0).base
    with pytest.raises(ValueError):
        two.descriptors(0, 2)
    with pytest.raises(ValueError):                 # (MTP's two slots are one sequence's)
        replace(spec, mtp=True).image(cfg, 256, 2, 2)


def _batched_vs_single(spec, W, cfg, sizes, steps=3, seed=3):
    """len(sizes) prompts into slots 0, 1, .. of an Image(batch=len(sizes)), each fed token by
    token through the decode step at its slot; then `steps` batched steps (every slot's row;
    the second step with the rows the other way round), each row's greedy token fed back:
    every row's logits are a single-sequence engine's bit for bit, and so are the prompts'
    (the slot views are the image's sequence 0's). Returns the tokens seen."""
    from opentpu.llm import batch35 as B35
    rng = np.random.default_rng(seed)
    prompts = [[int(t) for t in rng.integers(0, 1000, n)] for n in sizes]
    B = len(sizes)
    eng = Engine(spec, W, cap=256, cfg=cfg, batch=B)
    assert eng.image.batch == B and eng.image.pool is not None
    refs = [Engine(spec, W, cap=256, cfg=cfg) for _ in prompts]
    toks, pos = [], []
    for s, (pr, ref) in enumerate(zip(prompts, refs)):
        for p, t in enumerate(pr):
            got, want = _slot_step(eng, s, p, t), ref.step(t)
            assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), (s, p)
        toks.append(int(np.argmax(want)))
        pos.append(len(pr))
    seen = set()
    for t in range(steps):
        order = list(range(B))[::-1] if t == 1 else list(range(B))
        rows = [(s, pos[s]) for s in order]
        tk = [toks[s] for s in order]
        kw = {"tokens": tk} if eng.device_inputs else {}
        lg = eng._run_rows(rows, tk, list(range(B)), B35.compile_step(
            eng.image, rows, list(range(B)), eng.block, **kw))
        for r, s in enumerate(order):
            want = refs[s].step(toks[s])
            assert np.array_equal(lg[r].view(np.uint32), want.view(np.uint32)), (t, s, pos[s])
            toks[s] = int(np.argmax(want))
        seen.update(toks)
        pos = [p + 1 for p in pos]
    return seen


@pytest.mark.parametrize("S", [1, 2])
def test_batched_steps_are_single_sequence_steps(tiny, monkeypatch, S):
    """Two prompts (7 and 4 tokens), then 3 batched steps (_batched_vs_single): each row's
    logits a single-sequence engine's bit for bit. S = 2: tensor-parallel experts, each row's
    MoE moe_ffn's (the layer runs' way at S > 1)."""
    W, spec = tiny
    seen = _batched_vs_single(spec, W, _cfg(monkeypatch, S), (7, 4))
    assert len(seen) > 1                        # (an untied head: no echo)



X2 = "bcu1525-x2:st16"


def _x2(monkeypatch, S: int = 4):
    """bcu1525-x2's Config (both BCU1525s as one machine: four slices in tensor parallel, SEXACT)
    with every expert resident below 4 GiB (test_chat_x2's)."""
    from opentpu import target as T
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    return T.config(T.load(X2), S=S, DRAM_BYTES=1 << 26, BHW=0)


@pytest.fixture(scope="module")
def tiny_x2():
    """_tiny at x2's width (hidden, experts and shared expert S x D = 512 at S = 4)."""
    return _tiny(H=512, f=512, kinds=("linear", "attn"))


def _cards(cfg):
    """The two cards' transport over the ISA simulator (IsaCards), as an Engine's backend."""
    from opentpu.host.board import BoardBackend
    from opentpu.host.cards import CardsTransport, IsaCards
    sh = IsaCards(cfg)
    return lambda c, imgs: BoardBackend(c, imgs, transport=CardsTransport(sh.cards), status=False)


def _ahead_calls(monkeypatch) -> list:
    """Count batch35._deltanet_slots_ahead's calls (the pipelined batched DeltaNet mixer)."""
    from opentpu.llm import batch35 as B35
    calls, f = [], B35._deltanet_slots_ahead
    monkeypatch.setattr(B35, "_deltanet_slots_ahead", lambda *a: calls.append(1) or f(*a))
    return calls


def test_batched_steps_on_x2(tiny_x2, monkeypatch):
    """x2 (S = 4): two prompts, then 3 batched steps (_batched_vs_single), each row's logits a
    single-sequence engine's bit for bit on the same machine; the DeltaNet mixer pipelined
    (_deltanet_slots_ahead: the slice's 2 pairs unrolled, a row's two heads in one DSTEP2)."""
    W, spec = tiny_x2
    cfg = _x2(monkeypatch)
    assert cfg.S == 4 and cfg.SEXACT and cfg.DSTEP2
    calls = _ahead_calls(monkeypatch)
    assert len(_batched_vs_single(spec, W, cfg, (7, 4))) > 1
    assert calls


def test_batched_steps_with_the_pairs_looped(tiny, monkeypatch):
    """S = 2 (4 pairs a slice) with batch35.AHEAD_PAIRS 0: the batched DeltaNet mixer's pairs
    in their loop, unpipelined (S = 1's 8 pairs' way), not _deltanet_slots_ahead (S = 2's
    default: test_batched_steps_are_single_sequence_steps): each row's logits a single-sequence
    engine's bit for bit either way."""
    from opentpu.llm import batch35 as B35
    W, spec = tiny
    monkeypatch.setattr(B35, "AHEAD_PAIRS", 0)
    calls = _ahead_calls(monkeypatch)
    assert len(_batched_vs_single(spec, W, _cfg(monkeypatch, 2), (7, 4))) > 1
    assert not calls



def test_batched_steps_ahead_unshared(monkeypatch):
    """S = 2 (4 pairs a slice), _deltanet_slots_ahead with each value head its own key head
    (nk = nv = 16: a pair's two blocks of q k v): each row's logits a single-sequence engine's
    bit for bit."""
    W, spec = _tiny(nk=16)
    calls = _ahead_calls(monkeypatch)
    assert len(_batched_vs_single(spec, W, _cfg(monkeypatch, 2), (7, 4))) > 1
    assert calls


@pytest.mark.parametrize("cards", [False, True])
def test_slots_on_x2_pick_on_the_cards(tiny_x2, monkeypatch, cards):
    """x2 (S = 4), on the ISA simulator's machine or on the two cards (CardsTransport over
    IsaCards): each prompt into its slot by layer runs (batch35.prefill_slot), its logits a
    single-sequence engine's; then batched steps through batch35.Steps, each row's greedy pick
    on the cards (the four slices' ARGMAX gathered) a single-sequence engine's decode."""
    from opentpu.llm import batch35 as B35
    W, spec = tiny_x2
    cfg = _x2(monkeypatch)
    rng = np.random.default_rng(6)
    prompts = [[int(t) for t in rng.integers(0, 1000, n)] for n in (9, 5)]
    kw = dict(cap=256, cfg=cfg, resident=True, layer_major=2)
    eng = Engine(spec, W, batch=2, batch_area=True, **kw,
                 **({"backend": _cards(cfg)} if cards else {}))
    refs = [Engine(spec, W, **kw) for _ in prompts]
    st, nxt = B35.Steps(eng), {}
    for s, (pr, ref) in enumerate(zip(prompts, refs)):
        got, want = B35.prefill_slot(eng, pr, s), ref.prefill(pr)
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), s
        nxt[s] = int(np.argmax(want))
        st.set_sampling(s, None)
    seen = set()
    for i in range(5):
        out = st.run([(s, eng.poss[s]) for s in (0, 1)], [nxt[s] for s in (0, 1)])
        for r, s in enumerate((0, 1)):
            want = int(np.argmax(refs[s].step(nxt[s])))
            assert out[r] == want, (i, s)
            nxt[s] = want
        seen.update(nxt.values())
    assert len(seen) > 1 and st.timing["reused"] > 0, st.timing


@pytest.mark.parametrize("cards", [False, True])
def test_step_over_slots_on_x2(tiny_x2, monkeypatch, cards):
    """x2 (S = 4): one program for every position of the rows' slots (batch35.Steps(slots=True):
    compile_step(slots=True), each row's position, V^T place and attention groups run-time
    words, every block masked: attention.RunSpan), on the ISA simulator's machine or on the two
    cards. Two prompts into their slots, the slots moved on to 504 and 250 (the single-sequence
    engines' too), then 14 steps across the 256-token attention block (slot 1) and the
    512-token group (slot 0), slot 1 alone in two of them: each row's greedy pick a
    single-sequence engine's decode, from two programs (the pair's, slot 1's), each compiled
    once."""
    from opentpu.llm import batch35 as B35
    W, spec = tiny_x2
    cfg = _x2(monkeypatch)
    rng = np.random.default_rng(6)
    prompts = [[int(t) for t in rng.integers(0, 1000, n)] for n in (9, 5)]
    kw = dict(cap=1024, cfg=cfg, resident=True, layer_major=2)
    eng = Engine(spec, W, batch=2, batch_area=True, **kw,
                 **({"backend": _cards(cfg)} if cards else {}))
    refs = [Engine(spec, W, **kw) for _ in prompts]
    nxt = {}
    for s, (pr, ref) in enumerate(zip(prompts, refs)):
        B35.prefill_slot(eng, pr, s)
        nxt[s] = int(np.argmax(ref.prefill(pr)))
    for s, p in ((0, 504), (1, 250)):
        eng.poss[s] = refs[s].pos = p
    assert B35.slots_fit(eng)
    st = B35.Steps(eng, slots=True)
    for s in (0, 1):
        st.set_sampling(s, None)
    seen = set()
    for i in range(14):
        sl = (1,) if i in (6, 7) else (0, 1)
        out = st.run([(s, eng.poss[s]) for s in sl], [nxt[s] for s in sl])
        for r, s in enumerate(sl):
            want = int(np.argmax(refs[s].step(nxt[s])))
            assert out[r] == want, (i, s)
            nxt[s] = want
        seen.update(nxt.values())
    assert len(seen) > 1 and eng.poss == [516, 264]
    assert st.timing["programs"] == 2 and st.timing["reused"] == 12, st.timing


X2N = ("linear", "linear", "attn", "linear", "linear", "attn")


@pytest.fixture(scope="module")
def tiny_x2n():
    """_tiny at x2's width with two DeltaNet layers before each attention layer (the unit's
    repeated part: qwen35_batch_step's nest, the 35B's three)."""
    return _tiny(H=512, f=512, kinds=X2N)


def _four_rows_cfg(spec, W, variant, kw):
    """x2's Config for the four-rows tests: st16's IMEM between the tiny B = 4 step's nested and
    unrolled sizes (the 35B's B = 4 there: 2,632 nested, 5,063 unrolled, over its 4,096), so
    that compile_step loops a unit's DeltaNet layers; wmx's as it is (ICACHE: unrolled). Checks
    the step compile_step chooses; returns the Config."""
    from opentpu import target as T
    from opentpu.llm import batch35 as B35
    cfg = T.config(T.load(f"bcu1525-x2:{variant}"), S=4, DRAM_BYTES=1 << 26, BHW=0)
    rows4 = [(s, 0) for s in range(4)]

    def sizes(cfg, *nests):                  # None: compile_step's choice
        eng = Engine(spec, W, batch=4, batch_area=True, cfg=cfg, **kw)
        return [max(map(len, B35.compile_step(eng.image, rows4, block=eng.block, kinds=[None] * 4,
                                              area=eng.bat, slots=True, nest=f))) for f in nests]
    if variant == "st16":                   # the unrolled step past IMEM, the nested one in it
        u, n = sizes(cfg, False, True)
        assert n < u, (n, u)
        cfg = replace(cfg, IMEM_WORDS=8 * (n + u) // 2 // 64 * 64)
        assert sizes(cfg, None) == [n] and n <= cfg.IMEM_WORDS // 8 < u
    else:
        assert cfg.ICACHE and sizes(cfg, None)[0] <= cfg.IMEM_WORDS // 8
    return cfg


def _four_rows(spec, W, cfg, backend, kw, starts, sets, steps):
    """Four prompts into their slots on backend, the slots moved on to starts, then `steps`
    steps over slots (sets: step -> its rows' slots, else all four): each row's greedy pick a
    single-sequence engine's decode on the ISA machine; one program a row set, none halved.
    Returns each step's (rows' slots, cycles: the backend's)."""
    from opentpu.llm import batch35 as B35
    eng = Engine(spec, W, batch=4, batch_area=True, cfg=cfg, backend=backend, **kw)
    rng = np.random.default_rng(7)
    prompts = [[int(t) for t in rng.integers(0, 1000, n)] for n in (9, 5, 7, 6)]
    refs = [Engine(spec, W, cfg=cfg, **kw) for _ in prompts]
    nxt = {}
    for s, (pr, ref) in enumerate(zip(prompts, refs)):
        B35.prefill_slot(eng, pr, s)
        nxt[s] = int(np.argmax(ref.prefill(pr)))
    for s, p in enumerate(starts):
        eng.poss[s] = refs[s].pos = p
    st = B35.Steps(eng, slots=True)
    for s in range(4):
        st.set_sampling(s, None)
    seen, end, cycles = set(), list(starts), []
    for i in range(steps):
        sl = sets.get(i, (0, 1, 2, 3))
        c0 = st.timing["cycles"]
        out = st.run([(s, eng.poss[s]) for s in sl], [nxt[s] for s in sl])
        cycles.append((sl, st.timing["cycles"] - c0))
        for r, s in enumerate(sl):
            want = int(np.argmax(refs[s].step(nxt[s])))
            assert out[r] == want, (i, s)
            nxt[s] = want
            end[s] += 1
        seen.update(nxt.values())
    k = len({sets.get(i, (0, 1, 2, 3)) for i in range(steps)})
    assert len(seen) > 1 and eng.poss == end, (eng.poss, end)
    assert st.timing["programs"] == k and st.timing["reused"] == steps - k, st.timing
    return cycles


@pytest.mark.parametrize("variant", ["st16", "wmx"])
def test_four_rows_over_slots_on_x2(tiny_x2n, monkeypatch, variant):
    """B = 4 over slots on x2's two cards (CardsTransport over IsaCards): four prompts into
    their slots, the slots moved on to positions in each of the cache's four V^T tiles (a row's
    V^T append at its tile's column of the tiles as one grouped matrix, SlotPos's raw word: the
    data address the cache's alone, the row's value in the row operand), three of them across
    a tile's end, then 10 steps of the four rows and of three subsets: each row's greedy pick a
    single-sequence engine's decode, one program a row set (four), none halved. wmx (ICACHE):
    the step unrolled. st16 on an IMEM between the step's nested and unrolled sizes
    (_four_rows_cfg): compile_step loops a unit's DeltaNet layers."""
    W, spec = tiny_x2n
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    kw = dict(cap=1024, resident=True, layer_major=2)
    cfg = _four_rows_cfg(spec, W, variant, kw)
    _four_rows(spec, W, cfg, _cards(cfg), kw, (250, 504, 762, 10),     # tiles 0, 1, 2 into 1, 2, 3
               {4: (1, 3), 7: (2,), 8: (0, 1, 2)}, 10)


@pytest.mark.rtl
@pytest.mark.parametrize("variant", ["st16", "wmx"])
def test_four_rows_over_slots_on_x2_rtl(tiny_x2n, monkeypatch, x2_verilator, variant):
    """test_four_rows_over_slots_on_x2 on the two-board co-simulation (X2SimTransport: both
    cards' RTL at the variant's build, their DIMM paths and the link model): the batched rows on
    the RTL, each row's single-sequence reference on the ISA machine; 5 steps, three rows across
    a tile's end (254, 510 and 766 on), row sets 0-3, (1, 3) and (0, 1, 3)."""
    import json
    from opentpu import target as T
    from opentpu.host.board import BoardBackend
    from opentpu.host.cards import CardsTransport, X2SimTransport
    W, spec = tiny_x2n
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    kw = dict(cap=1024, resident=True, layer_major=2)
    cfg = _four_rows_cfg(spec, W, variant, kw)
    sim = X2SimTransport(target=T.load(f"bcu1525-x2:{variant}"), ch_bytes=cfg.DRAM_BYTES // 8,
                         plusargs=["+cp=3333", "+dimm_mc=1", "+mc_ba=15", "+mc_rw=62",
                                   "+mc_strict=1", "+mc_xbsw=16", "+mc_xrtw=8",
                                   "+mc_xwtr=8", "+mc_xgq=12", "+max_cycles=400000000"])
    ct = CardsTransport(sim.cards)
    cycles = _four_rows(spec, W, cfg, lambda c, imgs: BoardBackend(c, imgs, transport=ct, status=False),
                        kw, (254, 510, 766, 10), {1: (1, 3), 3: (0, 1, 3)}, 5)
    print("X2_RTL_FOUR_ROWS " + json.dumps({"variant": variant, "steps": [
        {"rows": list(sl), "cycles": c} for sl, c in cycles]}), flush=True)


def test_the_35bs_four_rows_fit_x2(monkeypatch):
    """The 35B's step over slots at B = 4 is one program on both x2 variants: wmx's (ICACHE)
    unrolled, st16's with a unit's DeltaNet layers in a loop (unrolled it is 5,063 of IMEM's
    4,096 instructions); each row's V^T append in the cache's data register and its row
    operand, so the four rows' addresses fit the registers (out of them at B = 4 with the
    tile's term in each row's data register)."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1] / "tools"))
    import pf500 as P
    from opentpu import target as T
    from opentpu.host.streamed import FORMATS
    from opentpu.llm import batch as BA
    from opentpu.llm import batch35 as B35
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import image_end
    monkeypatch.setenv("OTPU_HIGH", "0")
    spec = load_spec(P.MODEL)
    wf, hf, fm = FORMATS["qwen35"]
    for variant, nested in (("st16", True), ("wmx", False)):
        cfg = T.config(T.load(f"bcu1525-x2:{variant}"))
        img = spec.image(cfg, 4096, 4, 4, wformat=wf, head_format=hf, formats=fm, lookup=True)
        end = image_end(img) if getattr(img, "high", False) else img.nbytes
        bat = BA.alloc(img, cfg, 4, -(-end // 4096) * 4096)
        rows = [(s, 0) for s in range(4)]
        n = {f: len(B35.compile_step(img, rows, block=256, kinds=[None] * 4, area=bat,
                                     slots=True, slices=[0], nest=f)[0]) for f in (False, True)}
        auto = len(B35.compile_step(img, rows, block=256, kinds=[None] * 4, area=bat,
                                    slots=True, slices=[0])[0])
        imem = cfg.IMEM_WORDS // 8
        print(f"x2:{variant} B = 4: unrolled {n[False]}, nested {n[True]}, IMEM {imem}")
        assert auto == n[nested] <= imem and (n[False] > imem) == nested, (variant, n, imem)


def test_warm35x2_warms_every_slots_layer_runs(tiny_x2, monkeypatch, tmp_path):
    """warm35x2's layer runs on an image of two sequences (otpu-serve35x2 --slots 2's): after
    warm_layers into an empty program cache, a prompt into each slot layer by layer
    (batch35.prefill_slot: the runs at its slot) compiles none."""
    import importlib.util
    from pathlib import Path
    from opentpu import progcache as PC
    from opentpu.host import gen_warm as GW
    from opentpu.llm import batch35 as B35
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path / "qprog"))
    d = Path(__file__).resolve().parents[1] / "boards/bcu1525/chat35"
    monkeypatch.syspath_prepend(str(d))
    spec_ = importlib.util.spec_from_file_location("warm35x2", d / "warm35x2.py")
    WX = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(WX)
    W, spec = tiny_x2
    cfg = _x2(monkeypatch)
    eng = Engine(spec, W, cap=256, cfg=cfg, batch=2, resident=True, layer_major=2, batch_area=True,
                 backend=_cards(cfg))
    assert eng.prog_cache and eng.batch == 2
    be = eng.backend
    args = (eng.spec, eng.cfg, eng.cap, eng.batch, eng.rows, eng._image_kw,
            eng.image.stream_fill, eng.layout, eng.block, eng._conv_lo,
            bool(getattr(be, "chains", False)), eng.gen_debug, eng.gen_split,
            bool(getattr(be, "waitc", False)), GW.samplings(None))
    PC.clear()
    WX.warm_layers(args, [1], 2)
    PC.clear()
    before, disk = PC.stats["compile"], PC.stats["disk"]
    rng = np.random.default_rng(9)
    for s, n in ((0, 30), (1, 23)):
        B35.prefill_slot(eng, [int(t) for t in rng.integers(0, 1000, n)], s)
    assert PC.stats["compile"] == before and PC.stats["disk"] >= disk + 4 * spec.layers, PC.stats


@pytest.mark.parametrize("cards", [False, True])
def test_sampled_rows_on_x2(tiny_x2, monkeypatch, cards):
    """x2 (S = 4), on the ISA simulator's machine or on the two cards: both slots sampled on the
    device (slot 0 production's T 0.7, top_k 20, top_p 0.8; slot 1 with a repetition penalty),
    each slice's top k gathered and selected again (generate.Sampler at S > 1): every pick is
    S = 1's sampler's (generate.reference_pick, S = 1) on an S = 1 engine's logits with the
    row's uniform, bit for bit; the rows swap order on one step."""
    from opentpu.llm import batch35 as B35
    from opentpu.llm import generate as G
    W, spec = tiny_x2
    cfg, one = _x2(monkeypatch), _x2(monkeypatch, S=1)
    rng = np.random.default_rng(7)
    prompts = [[int(t) for t in rng.integers(0, 1000, n)] for n in (8, 6)]
    kw = dict(cap=256, resident=True, layer_major=2)
    eng = Engine(spec, W, batch=2, batch_area=True, cfg=cfg, **kw,
                 **({"backend": _cards(cfg)} if cards else {}))
    refs = [Engine(spec, W, cfg=one, **kw) for _ in prompts]
    samps = {0: G.Sampling(0.7, 20, 0.8, 1.0), 1: G.Sampling(0.8, 20, 0.9, 1.3)}
    st, nxt, ctx = B35.Steps(eng), {}, {}
    for s, (pr, ref) in enumerate(zip(prompts, refs)):
        B35.prefill_slot(eng, pr, s)
        nxt[s] = int(np.argmax(ref.prefill(pr)))
        ctx[s] = list(pr) + [nxt[s]]
        st.set_sampling(s, samps[s], ctx[s])
    u = np.minimum(np.random.default_rng(8).random((5, 2)), 1 - 2.0 ** -24).astype(np.float32)
    seen, sampled = set(), 0
    for i in range(5):
        order = [1, 0] if i == 2 else [0, 1]
        out = st.run([(s, eng.poss[s]) for s in order], [nxt[s] for s in order], uniforms=u[i])
        for r, s in enumerate(order):
            lg = refs[s].step(nxt[s])
            want = G.reference_pick(lg, samps[s], ctx[s], u[i][r], 1, None)
            assert out[r] == want, (i, s)
            sampled += want != int(np.argmax(lg))
            nxt[s] = want
            ctx[s].append(want)
        seen.update(nxt.values())
    assert len(seen) > 1 and sampled > 0          # (picks other than the argmax)


def test_steps_compile_the_slices_in_workers_on_x2(tiny_x2, monkeypatch):
    """x2 (S = 4, the two cards over IsaCards): batch35.Steps compiles each program's four
    slices at once in its own four worker processes (no engine compile workers at S > 1), each
    slice's words with its wait entries, so a run loads them as they are; the span after the
    one serving comes from the workers, nothing compiled inline for it."""
    import time
    from opentpu.llm import batch as BA
    from opentpu.llm import batch35 as B35
    W, spec = tiny_x2
    cfg = _x2(monkeypatch)
    eng = Engine(spec, W, cap=1024, cfg=cfg, batch=2, resident=True, batch_area=True,
                 backend=_cards(cfg))
    st = B35.Steps(eng)
    assert eng.pipeline and not eng._procs and st._spool is not None
    t0 = time.time()
    while not all(f.done() for f in st._sready):
        assert time.time() - t0 < 300, "the slice workers did not start"
        time.sleep(0.2)
    for s, p in ((0, 250), (1, 10)):
        st.set_sampling(s, None)
        eng.poss[s] = p
    st.run([(0, 250), (1, 10)], [5, 6])             # span 6: slot 0 to its block's end
    nxt = {k[0]: f for k, f in st._next.items()}
    assert ((0, 256), (1, 16)) in nxt, list(nxt)
    fut = nxt[((0, 256), (1, 16))]
    assert isinstance(fut, BA._Slices)
    ws = fut.result()
    assert len(ws) == 4 and all(isinstance(w, np.ndarray) for w in ws)
    if cfg.SEQ_WC:
        assert all(w.waits is not None for w in ws)
    inline = st.timing["compiled_inline"]
    for i in range(1, 8):                           # into the next span's program
        st.run([(0, 250 + i), (1, 10 + i)], [5, 6])
    assert st.timing["compiled_inline"] == inline, st.timing


@pytest.mark.parametrize("S", [1, 2])
def test_moe_rows_in_request_sized_groups(monkeypatch, S):
    """Top-8 of 16 experts (the 35B's k: one request row names 2 rows' picks), 3 rows' MoE
    (batch35._moe_rows): at S = 1 moe_ffn_rows over rows 0-1 and moe_ffn on row 2; at S = 2
    moe_ffn_tp_rows (the router and the shared expert once over the 3 rows, rows 1 and 2's
    experts in two groups); every row's logits a single-sequence engine's bit for bit."""
    W, spec = _tiny(E=16, K=8)
    seen = _batched_vs_single(spec, W, _cfg(monkeypatch, S=S), (5, 4, 6), steps=2)
    assert len(seen) > 1


def _area_step(eng, rows, toks, kinds=None, uni=None) -> list[int]:
    """One batched step through the batch area: the rows' tokens (and uniforms) in its io
    words, each row's pick read from tout."""
    from opentpu.llm import batch35 as B35
    a = eng.bat
    io = np.zeros(2 * a["R"] + 2, np.float32)
    io[:len(rows)] = toks
    if uni is not None:
        io[a["R"]:a["R"] + len(rows)] = np.minimum(uni, np.float32(1 - 2.0 ** -24))
    for s in range(eng.cfg.S):
        eng.backend.write(s, a["io"], io)
    eng.backend.run(B35.compile_step(eng.image, rows, block=eng.block, kinds=kinds, area=a))
    return [int(t) for t in eng.backend.read(0, a["tout"], 4 * len(rows)).view(np.float32)]


@pytest.mark.parametrize("pen,steps", [(1.0, False), (1.3, False), (1.3, True)])
def test_area_rows_pick_on_the_card(tiny, monkeypatch, pen, steps):
    """Engine(batch=2, batch_area=True) with the lookup tables: the rows' tokens from the
    area's tin (embedded from the image's table), each row's pick on the card into tout: slot
    0 greedy (a single-sequence engine's argmax), slot 1 sampled (top-k, top-p, temperature,
    a repetition penalty: generate.reference_pick on that engine's logits with the row's
    uniform), bit for bit; the rows swap order on one step. steps: through batch35.Steps (a
    program serves its rows for a span of steps, the run's step in the area's step word)."""
    from opentpu.llm import batch as BA
    from opentpu.llm import batch35 as B35
    from opentpu.llm import generate as G
    W, spec = tiny
    cfg = _cfg(monkeypatch)
    rng = np.random.default_rng(4)
    prompts = [[int(t) for t in rng.integers(0, 1000, n)] for n in (6, 5)]
    eng = Engine(spec, W, cap=256, cfg=cfg, batch=2, resident=True, batch_area=True)
    assert eng.bat is not None and eng.device_inputs
    refs = [Engine(spec, W, cap=256, cfg=cfg, resident=True) for _ in prompts]
    samps = {0: None, 1: G.Sampling(0.8, 20, 0.9, pen)}
    st = B35.Steps(eng)
    nxt, pos, ctx = {}, {}, {}
    for s, (pr, ref) in enumerate(zip(prompts, refs)):
        for p, t in enumerate(pr):
            got, want = _slot_step(eng, s, p, t), ref.step(t)
            assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), (s, p)
        nxt[s], pos[s] = int(np.argmax(want)), len(pr)
        eng.poss[s] = len(pr)
        ctx[s] = list(pr) + [nxt[s]]
        st.set_sampling(s, samps[s], ctx[s])
    u = np.random.default_rng(5).random((4, 2)).astype(np.float32)
    seen = set()
    for i in range(4):
        order = [1, 0] if i == 2 else [0, 1]
        rows = [(s, pos[s]) for s in order]
        if steps:
            out = st.run(rows, [nxt[s] for s in order], uniforms=u[i])
        else:
            out = _area_step(eng, rows, [nxt[s] for s in order],
                             [BA.kind_of(samps[s]) for s in order], u[i])
        for r, s in enumerate(order):
            lg = refs[s].step(nxt[s])
            want = int(np.argmax(lg)) if samps[s] is None else G.reference_pick(
                lg, samps[s], ctx[s], np.minimum(u[i][r], np.float32(1 - 2.0 ** -24)), 1, None)
            assert out[r] == want, (i, s)
            nxt[s], pos[s] = out[r], pos[s] + 1
            ctx[s].append(out[r])
        seen.update(nxt.values())
    assert len(seen) > 1
    if steps:                                   # (one program served several steps)
        assert st.timing["reused"] > 0, st.timing


def test_steps_compile_spans_ahead(tiny, monkeypatch):
    """batch35.Steps compiles B span programs ahead of the one serving (depth B, on the
    engine's pipeline): rows at different places in their attention blocks end spans of
    different lengths, each successor's rows where the span before it ends."""
    from opentpu.llm import batch35 as B35
    W, spec = tiny
    cfg = _cfg(monkeypatch)
    eng = Engine(spec, W, cap=1024, cfg=cfg, batch=2, resident=True, batch_area=True,
                 pipeline=True)
    st = B35.Steps(eng)
    assert st.depth == 2 and eng.block == 256
    for s, p in ((0, 10), (1, 200)):
        st.set_sampling(s, None)
        eng.poss[s] = p
    st.run([(0, 10), (1, 200)], [5, 6])
    ahead = {k[0]: k[2] for k in list(st._next) + list(st._ready)}
    # spans: 56 (slot 1 to its block's end), then 190 (slot 0's), then 66
    assert ahead == {((0, 66), (1, 256)): 190, ((0, 256), (1, 446)): 66}, ahead


def test_steps_compile_a_row_alone_ahead(tiny, monkeypatch):
    """prefetch_group (BatchWorker35's, two requests decoding): once the batch's own programs
    are compiled, a row's step alone from its position to its block's end, once; the row then
    runs alone (the other request ended) on it, nothing compiled inline."""
    from opentpu.llm import batch35 as B35
    W, spec = tiny
    cfg = _cfg(monkeypatch)
    eng = Engine(spec, W, cap=1024, cfg=cfg, batch=2, resident=True, batch_area=True,
                 pipeline=True)
    st = B35.Steps(eng)
    for s, p in ((0, 10), (1, 200)):
        st.set_sampling(s, None)
        eng.poss[s] = p
    st.run([(0, 10), (1, 200)], [5, 6])
    for f in list(st._next.values()):
        f.result()
    st.prefetch_group([(1, 201)], [st.kinds[1]])
    st.prefetch_group([(1, 201)], [st.kinds[1]])
    alone = [k for k in st._next if len(k[0]) == 1]
    assert [(k[0], k[2]) for k in alone] == [(((1, 201),), 55)], alone
    inline = st.timing["compiled_inline"]
    eng.poss[1] = 201
    st.run([(1, 201)], [7])
    st.run([(1, 202)], [8])
    assert st.timing["compiled_inline"] == inline, st.timing



def test_steps_compile_a_pair_ahead_at_three(tiny, monkeypatch):
    """prefetch_group at B = 3 (BatchWorker35's: the batch less one row): a pair's step from its
    positions to the first one's block end, compiled ahead; the pair then runs on it (the third
    request ended), nothing compiled inline."""
    from opentpu.llm import batch35 as B35
    W, spec = tiny
    cfg = _cfg(monkeypatch)
    eng = Engine(spec, W, cap=1024, cfg=cfg, batch=3, resident=True, batch_area=True,
                 pipeline=True)
    st = B35.Steps(eng)
    for s, p in ((0, 10), (1, 200), (2, 40)):
        st.set_sampling(s, None)
        eng.poss[s] = p
    st.run([(0, 10), (1, 200), (2, 40)], [5, 6, 7])
    for f in list(st._next.values()):
        f.result()
    pair = [(0, 11), (1, 201)]
    st.prefetch_group(pair, [st.kinds[0], st.kinds[1]])
    assert [(k[0], k[2]) for k in st._next if len(k[0]) == 2] == [(tuple(pair), 55)], st._next
    inline = st.timing["compiled_inline"]
    eng.poss[0], eng.poss[1] = 11, 201
    st.run(pair, [8, 9])
    st.run([(0, 12), (1, 202)], [1, 2])
    assert st.timing["compiled_inline"] == inline, st.timing

def test_prompts_into_slots_by_layer_runs(tiny, monkeypatch):
    """Each prompt into its slot by the layer-major runs at that slot (batch35.prefill_slot):
    the logits bit for bit as a single-sequence engine's layer-major prompt (slot 0 there);
    slot 0's context untouched by slot 1's prompt; then batched greedy steps through
    batch35.Steps pick each sequence's own decode."""
    from opentpu.llm import batch35 as B35
    W, spec = tiny
    cfg = _cfg(monkeypatch)
    rng = np.random.default_rng(6)
    prompts = [[int(t) for t in rng.integers(0, 1000, n)] for n in (9, 5)]
    eng = Engine(spec, W, cap=256, cfg=cfg, batch=2, resident=True, layer_major=2,
                 batch_area=True)
    refs = [Engine(spec, W, cap=256, cfg=cfg, resident=True, layer_major=2) for _ in prompts]
    st, nxt = B35.Steps(eng), {}
    for s, (pr, ref) in enumerate(zip(prompts, refs)):
        got, want = B35.prefill_slot(eng, pr, s), ref.prefill(pr)
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), s
        assert eng.poss[s] == len(pr)
        nxt[s] = int(np.argmax(want))
        st.set_sampling(s, None)
    assert eng.poss == [len(p) for p in prompts]
    seen = set()
    for i in range(4):
        rows = [(s, eng.poss[s]) for s in (0, 1)]
        out = st.run(rows, [nxt[s] for s in (0, 1)])
        for r, s in enumerate((0, 1)):
            want = int(np.argmax(refs[s].step(nxt[s])))
            assert out[r] == want, (i, s)
            nxt[s] = want
        seen.update(nxt.values())
    assert len(seen) > 1
