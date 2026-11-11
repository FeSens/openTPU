"""Speculative decoding with the MTP drafter on a streamed Qwen3.5-MoE (opentpu/llm/mtp.py,
docs/mtp.md 12, docs/offload.md 15.6) on the ISA simulator: the verify run's two rows go through
each MoE layer as one request for their experts' union (moe.moe_ffn_rows), the MTP layer's MoE
is one more layer of the expert server's (its experts in the pool after the model's), and the
greedy tokens equal plain greedy decode's bit for bit whatever the drafts, with 2k expert slots
a layer (misses on every layer). A tiny random Qwen3.5-MoE (tests/test_qwen35_moe.py) with random
mtp.* weights in the checkpoint's form (its experts a tensor each)."""
import dataclasses
import os

import numpy as np
import pytest

from opentpu.isasim import board_config
from opentpu.llm.mtp import MTPDecoder, mtp_engine
from opentpu.llm.qwen3 import Engine

from test_qwen35_moe import K, _tiny, _untied

SLOTS = 2 * K               # a layer's slots: a 2-row request's union fits (k of E = 8 each)

pytest.importorskip("torch")

CFG = board_config(DRAM_BYTES=1 << 26, DSTEP=True, STREAM=True, PAIR=True)   # (an int8 MTP
#   image with the default MoE tiles' mask tiles and I/O scratch: 32 MiB, docs/tiled_prefill.md 14)


def _moe_mtp_weights(W, spec, seed=1):
    """Random mtp.* tensors for the tiny MoE (zero-centred norms; the MoE as the 35B's
    checkpoint has it: the router mlp.gate, the shared expert and its gate, each routed
    expert's gate / up / down a tensor of its own)."""
    r = np.random.default_rng(seed)
    H, d, mo = spec.hidden, spec.head_dim, spec.moe

    def lin(o, i, s=1.0):
        return (s * r.uniform(-1, 1, (o, i)) / np.sqrt(i)).astype(np.float32)

    def norm(n):
        return (0.1 * r.standard_normal(n)).astype(np.float32)
    a, m = "mtp.layers.0.self_attn.", "mtp.layers.0.mlp."
    out = {"mtp.pre_fc_norm_embedding.weight": norm(H), "mtp.pre_fc_norm_hidden.weight": norm(H),
           "mtp.fc.weight": lin(H, 2 * H), "mtp.norm.weight": norm(H),
           "mtp.layers.0.input_layernorm.weight": norm(H),
           "mtp.layers.0.post_attention_layernorm.weight": norm(H),
           a + "q_proj.weight": lin(2 * spec.n_q * d, H), a + "k_proj.weight": lin(spec.n_kv * d, H),
           a + "v_proj.weight": lin(spec.n_kv * d, H), a + "o_proj.weight": lin(H, spec.n_q * d),
           a + "q_norm.weight": norm(d), a + "k_norm.weight": norm(d),
           m + "gate.weight": lin(mo.E, H, 4.0), m + "shared_expert_gate.weight": lin(1, H),
           m + "shared_expert.gate_proj.weight": lin(mo.shared, H),
           m + "shared_expert.up_proj.weight": lin(mo.shared, H),
           m + "shared_expert.down_proj.weight": lin(H, mo.shared)}
    for e in range(mo.E):
        out.update({f"{m}experts.{e}.gate_proj.weight": lin(mo.ffn, H),
                    f"{m}experts.{e}.up_proj.weight": lin(mo.ffn, H),
                    f"{m}experts.{e}.down_proj.weight": lin(H, mo.ffn)})
    return {**W, **out}


@pytest.fixture(scope="module")
def tiny_moe_mtp():
    """(weights, spec): the tiny MoE with an LM head of its own (its int8 embedding table on
    the host, as the 35B's) and random mtp.* weights."""
    spec, W = _untied(_tiny())
    return _moe_mtp_weights(W, spec), spec


FP4 = dict(wformat="fp4", head_format="int8")


def _plain(spec, W, prompt, N, cfg=CFG, **kw):
    """Plain greedy decode's tokens (resident decode, the host loop, 2k slots a layer)."""
    return Engine(spec, W, cap=512, cfg=cfg, rows=1, resident=True, experts=SLOTS,
                  **kw).generate(prompt, max_new=N)


@pytest.mark.parametrize("drafter, fmt", [("mtp", "int8"), ("right", "int8"), ("wrong", "int8"),
                                          ("mixed", "int8"), ("mixed", "fp4"), ("mtp", "fp4")])
def test_moe_mtp_greedy_is_plain_greedy(tiny_moe_mtp, drafter, fmt):
    """Tokens equal plain greedy decode's for any drafts (the MTP's, all right, all wrong, right
    but every third) with 2k slots a layer: the verify's two rows request their experts' union
    in each MoE layer, the MTP layer its own (one more layer of the server's), and every layer
    misses on the way. The prompt's 9 tokens prefill in runs of 2 rows. fp4: 4-bit weights,
    whose 2-row MMs pair (PAIR, MCOLS 4) as the decode steps' do."""
    W, spec = tiny_moe_mtp
    kw = FP4 if fmt == "fp4" else {}
    prompt = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 9)]
    N = 16
    want = _plain(spec, W, prompt, N, **kw)
    P = len(prompt)

    def right(q, out):
        return want[q - P] if q - P < len(want) else 0
    drafts = {"mtp": None, "right": right,
              "wrong": lambda q, out: (right(q, out) + 1) % 1000,
              "mixed": lambda q, out: right(q, out) if q % 3 else (right(q, out) + 7) % 1000}
    eng = mtp_engine(spec, W, cap=512, cfg=CFG, rows=2, experts=SLOTS, **kw)
    assert eng.image.offload.layers == spec.layers + 1
    dec = MTPDecoder(eng)
    m0 = eng.server.misses
    st = dec.generate(prompt, max_new=N, drafts=drafts[drafter])
    assert st.tokens == want
    assert eng.server.misses > m0
    if drafter == "right":
        assert all(st.accepted[:-1]) and st.iterations <= N // 2 + 1
    if drafter == "wrong":
        assert not any(st.accepted)
    E = spec.moe.E
    if drafter == "mtp":                    # the MTP layer's experts went through the server
        assert any(g // E == spec.layers for g in eng.server.lru[spec.layers])


def test_the_mtp_layers_experts_pack_from_its_tensors(tiny_moe_mtp):
    """The expert server's layer `layers` is the MTP layer's MoE: its slots after the model's,
    its experts packed from the checkpoint's mtp.layers.0.mlp.experts.E.* tensors."""
    W, spec = tiny_moe_mtp
    eng = mtp_engine(spec, W, cap=512, cfg=CFG, rows=2, experts=SLOTS)
    img, E = eng.image, spec.moe.E
    g = spec.layers * E + 3
    a = img.expert(W, g)
    W2 = dict(W)
    W2["mtp.layers.0.mlp.experts.3.up_proj.weight"] = W["mtp.layers.0.mlp.experts.4.up_proj.weight"]
    assert not np.array_equal(a, img.expert(W2, g))
    assert np.array_equal(img.expert(W, 5), img.expert(W2, 5))
    plain = Engine(spec, W, cap=512, cfg=CFG, rows=1, resident=True, experts=SLOTS).image
    assert img.offload.layers == plain.offload.layers + 1
    assert dataclasses.replace(spec).mtp is False


@pytest.mark.parametrize("P, drafter, fmt, N", [
    (250, "right", "int8", 16),     # V at 254 accepts: the wrap; the last token alone
    (251, "right", "int8", 15),     # 253 -> 255: E, then D1 and the next bucket's V
    (249, "mixed", "int8", 15),     # E at the other parity
    (250, "wrong", "int8", 12),
    (249, "right", "fp4", 15),
    (250, "mtp", "int8", 15)])
def test_moe_mtp_loop_on_the_device_is_plain_greedy(tiny_moe_mtp, P, drafter, fmt, N):
    """The MTP loop on the device (V, E, D, D1 chained through the buckets' programs) on the
    streamed MoE gives plain greedy decode's tokens across a bucket's end, whatever the drafts,
    and leaves the committed DeltaNet states as plain decode does. The embedding rows come from
    the host: the host writes the first V's (a0, d), then each program posts the next one's
    tokens as one request (V: a0, a1 for the D; D: t, d for the next V; RowServer's rows 0, 1
    of the slot), and the expert server serves the MoE layers' misses meanwhile."""
    from test_mtp import _states
    W, spec = tiny_moe_mtp
    kw = FP4 if fmt == "fp4" else {}
    prompt = [int(t) for t in np.random.default_rng(P).integers(0, 1000, P)]
    want = _plain(spec, W, prompt, N, **kw)
    right = np.zeros(514, np.float32)
    right[P:P + len(want)] = want
    drafts = {"right": right, "wrong": (right + 1) % 1000, "mtp": None,
              "mixed": np.where(np.arange(514) % 3, right, (right + 7) % 1000)}[drafter]
    eng = mtp_engine(spec, W, cap=512, cfg=CFG, rows=2, experts=SLOTS, **kw)
    dec = MTPDecoder(eng)
    m0 = eng.server.misses
    eng.row_server.history = []
    st = dec.generate_card(prompt, max_new=N, stop=[], drafts=drafts)
    assert st.tokens == want
    assert eng.server.misses > m0
    assert len(eng.row_server.history) >= 2 * st.iterations - 2     # both rows, each program
    if drafter == "right":
        assert sum(st.accepted) >= len(want) // 2 - 2
    if drafter == "wrong":
        assert not any(st.accepted)
    ref = Engine(spec, W, cap=512, cfg=CFG, rows=1, resident=True, experts=SLOTS, **kw)
    ref.prefill(prompt)
    for t in want[:-1]:
        ref.step(t)
    assert eng.pos == ref.pos == P + len(want) - 1
    assert all(np.array_equal(a, b) for a, b in
               zip(_states(eng, spec, dec.slot), _states(ref, spec, 0)))


@pytest.mark.parametrize("P, drafter, N", [(249, "mixed", 15), (250, "mtp", 12)])
def test_moe_mtp_loop_in_two_parts_is_plain_greedy(tiny_moe_mtp, P, drafter, N):
    """A V or E over IMEM goes in two parts (MTPDecoder._gen_programs; MiniCPM-V's formats give
    its layers two runs, and its V 6173 instructions against the card's 4096): part 0 the
    layers before a run's first (plan_cuts), the rows' residual to DRAM, HALT CHAIN to part 1
    (the rest, the head, the token). Layers 0-2 with int8 DeltaNet weights, 3-5 fp4 (two runs),
    every V and E split (split True): the loop's tokens are plain greedy decode's of the same
    formats across a bucket's end (an E), whatever the drafts, and the committed states plain
    decode's. The cut: layer 3, the runs' middle."""
    from opentpu.llm.mtp import plan_cuts
    from test_mtp import _states
    W, spec = tiny_moe_mtp
    spec = dataclasses.replace(spec, formats="delta@0-2=int8")
    prompt = [int(t) for t in np.random.default_rng(P).integers(0, 1000, P)]
    want = _plain(spec, W, prompt, N, **FP4)
    right = np.zeros(514, np.float32)
    right[P:P + len(want)] = want
    drafts = {"mtp": None, "mixed": np.where(np.arange(514) % 3, right, (right + 7) % 1000)}[
        drafter]
    eng = mtp_engine(spec, W, cap=512, cfg=CFG, rows=2, experts=SLOTS, **FP4)
    assert plan_cuts(eng.image)[0] == 3            # (runs: layers 0-1, 2, 3-4, 5)
    dec = MTPDecoder(eng)
    dec.split = True
    st = dec.generate_card(prompt, max_new=N, stop=[], drafts=drafts)
    assert st.tokens == want
    ve = {k for k in eng._mtp_gen if len(k) == 2 and k[1] < 4}    # (V0, V1, E0, E1)
    assert ve and {k[:2] for k in eng._mtp_gen if len(k) == 3} == ve    # each in two parts
    ref = Engine(spec, W, cap=512, cfg=CFG, rows=1, resident=True, experts=SLOTS, **FP4)
    ref.prefill(prompt)
    for t in want[:-1]:
        ref.step(t)
    assert eng.pos == ref.pos == P + len(want) - 1
    assert all(np.array_equal(a, b) for a, b in
               zip(_states(eng, spec, dec.slot), _states(ref, spec, 0)))


def test_a_loop_program_that_fits_is_not_split():
    """The split V / E (docs/mtp.md 10.3) changes neither an MTP image's layout nor a program
    of a model whose V and E fit IMEM: Qwen3.5-35B-A3B's loop programs of bucket 1 (its
    formats, the card's configuration, its 38 slots a layer) are byte for byte those of the
    tree before the split (3af05a43) without the image's mask tiles (OTPU_TILE=0). With them
    (Qwen3.5-MoE's tiles on by default, em-tile 2bb7bb0b) the image is 1,331,200 bytes longer
    and the areas the programs address move: the same instructions, only address words
    differing (by 262,144, 1,331,200 and 1,331,328 bytes), their own sum; the tiles' DeltaNet
    rows (Image.dnrows: 8 of the image's gr / on rows for 4) move them by 9,216 and 12,288 bytes
    more. A change of the programs themselves updates the sums here: ol.load_home (9aa097d8,
    the DeltaNet input norm's gain in a TMEM home) moved TMEM words only (6261e98d / d36072c1),
    the DN tail's two VOPs (d2de6822) added 3 instructions to each of the four DeltaNet
    programs (1191e8cb / 843cd971), the gathers' one-hot operand made in TMEM (gather._made)
    two FILLs for its LD, one instruction more in each of the six (5806e9c6 / 2d75e3f3), the
    loop's address registers bounded from the state block's tpos (compiler.mul_bound: the same
    instructions with their bounds; 038ffe6a / 996357c1)."""
    import hashlib
    from pathlib import Path
    from opentpu import isa as I
    from opentpu.host import streamed as ST
    from opentpu.llm import load_spec
    from opentpu.llm import mtp as MT
    d = Path(__file__).resolve().parents[1] / "models" / "Qwen3.5-35B-A3B"
    if not (d / "config.json").exists():
        pytest.skip("no models/Qwen3.5-35B-A3B")
    cfg = board_config(PAIR=True, DSTEP=True, STREAM=True)
    spec = dataclasses.replace(load_spec(d), mtp=True)
    wf, hf, fm = ST.defaults(spec)
    probe = ST.probe(spec, cfg, 2048, rows=2, wformat=wf, head_format=hf, formats=fm,
                     lookup=True)
    n = ST.fit_experts(spec, cfg, 2048, probe)
    img = spec.image(cfg, 2048, 1, 2, wf, hf, lookup=True, experts=n, formats=fm)
    h = hashlib.sha256()
    for k in range(MT.NK):
        progs = MT.compile_gen(img, 1, k, 256)
        assert all(len(p) <= cfg.IMEM_WORDS // 8 for p in progs)
        for p in progs:
            h.update(np.asarray(I.assemble(p), "<u4").tobytes())
    assert n == 38
    untiled = os.environ.get("OTPU_TILE", "").strip() == "0"
    assert h.hexdigest() == ("996357c19260b7cddf92962e04a58d66019d7650de8febaeac49affa206d352e"
                             if untiled else
                             "038ffe6a034291bdcc5931d86b2a7d1284cb4f9c54f31824ccf32f900abc9d09")


@pytest.mark.parametrize("P, drafter, sp", [
    (250, "right", (0.8, 5, 0.9, 1.1)),     # penalty: row 1's counts the draft
    (251, "mixed", (1.5, 20, 1.0, 1.0)),    # E at 255; no penalty, top-k 20
    (249, "mtp", (0.8, 5, 0.9, 1.1))])
def test_moe_mtp_sampled_loop_is_plain_sampling(tiny_moe_mtp, P, drafter, sp):
    """The sampled MTP loop on the device on the streamed MoE gives plain sampled decode's
    tokens for the same uniforms whatever the drafts (the verify samples a0 with p + 1's
    uniform and accepts iff a0 is the draft, a1 with p + 2's), and leaves the committed
    states as plain decode does."""
    from opentpu.llm import generate as G
    from test_mtp import _states
    W, spec = tiny_moe_mtp
    samp, N, seed = G.Sampling(*sp), 15, 5
    prompt = [int(t) for t in np.random.default_rng(P).integers(0, 1000, P)]
    ref = Engine(spec, W, cap=512, cfg=CFG, rows=1, resident=True, experts=SLOTS)
    u = np.minimum(np.random.default_rng(seed).random(N).astype(np.float32),
                   np.float32(1 - 2.0 ** -24))
    c, want, lg = list(prompt), [], ref.prefill(prompt)
    for i in range(N):
        want.append(G.reference_pick(lg, samp, c, u[i]))
        c.append(want[-1])
        if i < N - 1:
            lg = ref.step(want[-1])
    assert len(set(want)) > len(want) // 2
    right = np.zeros(514, np.float32)
    right[P:P + len(want)] = want
    drafts = {"right": right, "mtp": None,
              "mixed": np.where(np.arange(514) % 3, right, (right + 7) % 1000)}[drafter]
    eng = mtp_engine(spec, W, cap=512, cfg=CFG, rows=2, experts=SLOTS)
    dec = MTPDecoder(eng)
    st = dec.generate_card(prompt, max_new=N, stop=[], drafts=drafts, sampling=samp,
                           rng=np.random.default_rng(seed))
    assert st.tokens == want
    if drafter == "right":
        assert sum(st.accepted) >= len(want) // 2 - 2
    assert eng.pos == ref.pos == P + len(want) - 1
    assert all(np.array_equal(a, b) for a, b in
               zip(_states(eng, spec, dec.slot), _states(ref, spec, 0)))


def _layered(spec, W, em, cfg=CFG, **kw):
    """An MTP engine whose prompt goes layer by layer (runs of 2 rows), expert-major or not,
    6 slots a layer (expert-major's scratch beside 2 k)."""
    return mtp_engine(spec, W, cap=512, cfg=cfg, rows=2, experts=6, layer_major=2,
                      expert_major=em, **kw)


@pytest.mark.parametrize("em, fmt, drafter", [(False, "int8", "mtp"), (True, "int8", "mtp"),
                                              (True, "fp4", "mixed"), (False, "int8", "wrong")])
def test_moe_mtp_layer_major_prefill(tiny_moe_mtp, em, fmt, drafter, monkeypatch):
    """A streamed MoE's prompt layer by layer under MTP (Engine.prefill_layers with the MTP
    layer's K / V, docs/mtp.md 12): the rows before conv_k - 1 in the rows kernel's runs, then
    chunks of 24 rows through the model a layer at a time (expert-major or not), after each
    chunk's last layer runs of up to 4 rows append the MTP layer's K / V (no queries, no MoE),
    and the last row drafts after the head (its hidden row the head's): the loop's tokens equal
    plain greedy decode's. int8: the MTP layer's KV cache and the first draft equal those of
    the 2-row runs' prefill (the MTP layer over every row) bit for bit."""
    from opentpu.llm import qwen35 as Q35
    from opentpu.llm.mtp import MTPStats
    from test_mtp import _mtp_kv
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    W, spec = tiny_moe_mtp
    kw = FP4 if fmt == "fp4" else {}
    prompt = [int(t) for t in np.random.default_rng(11).integers(0, 1000, 40)]
    P, N = len(prompt), 12
    want = _plain(spec, W, prompt, N, **kw)
    right = np.zeros(514, np.float32)
    right[P:P + len(want)] = want
    drafts = {"mtp": None, "wrong": (right + 1) % 1000,
              "mixed": np.where(np.arange(514) % 3, right, (right + 7) % 1000)}[drafter]
    eng = _layered(spec, W, em, **kw)
    dec = MTPDecoder(eng)
    assert dec.by_layers and eng.image.prefill_rows == 24 and eng.expert_major == em
    st = dec.generate_card(prompt, max_new=N, stop=[], drafts=drafts)
    assert st.tokens == want
    m = sorted(k for k in eng._layer_runs if k[0] == "m")
    assert m and max(k[2] for k in m) == 4      # (the host's slot rows)
    if drafter == "wrong":
        assert not any(st.accepted)
    if fmt == "int8" and drafter == "mtp":
        a, b = MTPDecoder(_layered(spec, W, em)), MTPDecoder(mtp_engine(
            spec, W, cap=512, cfg=CFG, rows=2, experts=6))
        assert not b.by_layers
        ra, rb = a.prefill(prompt, MTPStats()), b.prefill(prompt, MTPStats())
        assert ra == rb and ra[0] == want[0]
        assert all(np.array_equal(x, y) for x, y in zip(_mtp_kv(a.eng, P), _mtp_kv(b.eng, P)))


@pytest.mark.parametrize("P, em, drafter", [(40, True, "mtp"), (256, False, "mtp"),
                                           (255, False, "mixed")])
def test_moe_mtp_loop_drafts_the_prompts_last_row_first(tiny_moe_mtp, P, em, drafter,
                                                        monkeypatch):
    """A lazy layer-by-layer prefill (MTPDecoder.prefill(lazy=True), otpu-chat's) returns a0
    with no draft run, and the loop's first run is the D1 of P - 1's bucket over (hid[0], a0):
    the same draft, the MTP layer's K / V row at P - 1 and so the same iterations, tokens,
    accepted drafts, last draft, committed slot and MTP KV cache as the eager prefill's draft
    run and a loop from the verify, bit for bit. P 256: the D1 at the first bucket's last
    position chains to the next bucket's V; P 255: to its own bucket's E. A loop that does not
    run (max_new 1) drafts on its own (_settle), as the eager prefill does."""
    from opentpu.llm import qwen35 as Q35
    from opentpu.llm.mtp import MTPStats
    from test_mtp import _mtp_kv
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    W, spec = tiny_moe_mtp
    prompt = [int(t) for t in np.random.default_rng(P).integers(0, 1000, P)]
    N = 12
    want = _plain(spec, W, prompt, N)
    right = np.zeros(514, np.float32)
    right[P:P + len(want)] = want
    drafts = None if drafter == "mtp" else np.where(np.arange(514) % 3, right, (right + 7) % 1000)
    got, drafted = [], []
    for lazy in (False, True):
        dec = MTPDecoder(_layered(spec, W, em))
        st = MTPStats()
        a0, d = dec.prefill(prompt, st, lazy=lazy)
        assert (d is None) == lazy and dec.pending == ((P - 1, a0) if lazy else None)
        drafted.append([k for k, _, _ in st.runs].count("mtp prefill"))
        out = dec.loop_card(a0, d, max_new=N, stop=[], drafts=drafts)
        assert dec.pending is None and out.tokens == want
        got.append((out.accepted, dec.draft, dec.slot, dec.eng.pos, _mtp_kv(dec.eng, dec.eng.pos)))
    assert drafted[0] == drafted[1] + 1         # (the rows before conv_k - 1 draft as before)
    (acc, d0, s0, p0, kv0), (acc1, d1, s1, p1, kv1) = got
    assert (acc, d0, s0, p0) == (acc1, d1, s1, p1)
    assert all(np.array_equal(x, y) for x, y in zip(kv0, kv1))
    if P == 40:
        dec = MTPDecoder(_layered(spec, W, em))
        a0, _ = dec.prefill(prompt, MTPStats(), lazy=True)
        assert dec.loop_card(a0, None, max_new=1, stop=[]).tokens == [a0] == want[:1]
        eager = MTPDecoder(_layered(spec, W, em))
        assert eager.prefill(prompt, MTPStats()) == (a0, dec.draft) and dec.pending is None
        assert all(np.array_equal(x, y) for x, y in zip(_mtp_kv(dec.eng, P),
                                                        _mtp_kv(eager.eng, P)))


@pytest.mark.parametrize("em", [False, True])
def test_moe_mtp_rollback_to_a_held_snapshot_is_a_rerun(tiny_moe_mtp, em, monkeypatch):
    """A rollback to a snapshot whose next token changed (a chat's next turn): the snapshot
    after a part (follow: the token after it) holds the part's last row's residual (the K / V
    run of the expert-major prompt keeps it in its chunk row), the rollback goes back to the
    snapshot itself, and the next prefill first appends the MTP layer's row before it from
    that row and the new token (Engine.prefill_layers' fix), then feeds the rest. The MTP
    layer's KV cache, the committed DeltaNet states, a0 and the first draft equal a fresh
    decoder's over the new context bit for bit: the row as a run of that position through
    the model would append it."""
    from opentpu.llm import qwen35 as Q35
    from opentpu.llm.mtp import MTPStats
    from test_mtp import _mtp_kv, _states
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    W, spec = tiny_moe_mtp
    r = np.random.default_rng(5)
    ids1 = [int(t) for t in r.integers(0, 1000, 40)]
    at = 30
    ids2 = ids1[:at] + [int(t) for t in r.integers(0, 1000, 13)]
    assert ids2[at] != ids1[at]
    dec = MTPDecoder(_layered(spec, W, em, snapshots=1))
    assert dec.holds
    dec.prefill(ids1[:at], MTPStats(), follow=ids1[at])
    assert dec.snapshot() and at in dec.held
    dec.prefill(ids1[at:], MTPStats())
    assert dec.rollback(at) == at == dec.eng.pos and dec.refill[0] == at - 1
    n0 = len(dec.eng.stats)
    got = dec.prefill(ids2[at:], MTPStats())
    assert dec.refill is None and any(k[0] == "m" and k[3:] == (("xbuf",) if em else ())
                                      and k[2] == 1 for k in dec.eng._layer_runs)
    ref = MTPDecoder(_layered(spec, W, em))
    assert got == ref.prefill(ids2, MTPStats())
    L = len(ids2)
    assert dec.eng.pos == ref.eng.pos == L and len(dec.eng.stats) > n0
    assert all(np.array_equal(x, y) for x, y in zip(_mtp_kv(dec.eng, L), _mtp_kv(ref.eng, L)))
    assert all(np.array_equal(x, y) for x, y in zip(_states(dec.eng, spec, dec.slot),
                                                    _states(ref.eng, spec, ref.slot)))


def test_moe_mtp_layer_major_continues_from_slot_1(tiny_moe_mtp, monkeypatch):
    """A chat's next turn on the layer-major path: the first turn's loop (drafts all right)
    leaves the states committed in slot 1, and the next prompt's layer runs read and write that
    slot (the runs of slot 1, compiled apart). The tokens equal one fresh decoder's over the
    whole conversation, and plain greedy decode's."""
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    W, spec = tiny_moe_mtp
    r = np.random.default_rng(3)
    p1, p2 = [int(t) for t in r.integers(0, 1000, 30)], [int(t) for t in r.integers(0, 1000, 9)]
    w1 = _plain(spec, W, p1, 12)
    right = np.zeros(514, np.float32)
    right[len(p1):len(p1) + 12] = w1
    dec = MTPDecoder(_layered(spec, W, True))
    t1 = dec.generate_card(p1, max_new=12, stop=[], drafts=right).tokens
    assert t1 == w1 and dec.slot == 1
    ctx = p1 + t1 + p2
    st = dec.generate_card([t1[-1]] + p2, max_new=10, stop=[])
    assert any(k[1:] == ("slot", 1) for k in dec.eng._layer_runs if len(k) == 3)
    assert any(k[0][-1] == "tile" for k in dec.eng._layer_runs     # (the turn's tiled runs,
               if len(k) == 3 and k[1:] == ("slot", 1))            # on slot 1)
    fresh = MTPDecoder(_layered(spec, W, True)).generate_card(ctx, max_new=10, stop=[])
    assert st.tokens == fresh.tokens == _plain(spec, W, ctx, 10)


def test_streamed_mtp_engine_from_a_checkpoint_and_an_extended_pool(tiny_moe_mtp, tmp_path,
                                                                      monkeypatch):
    """otpu-chat --mtp on a streamed MoE (streamed.engine(mtp=True)): the checkpoint's mtp.*
    tensors (LazyWeights(mtp=True)) and the pool pack_pool.py extend makes from the model's own
    (its experts and marks copied, the MTP layer's after them not packed: the run packs each it
    asks for from the checkpoint). The loop's tokens equal plain greedy decode's from the
    model's pool. MiniCPM-V 4.7's case: a checkpoint without mtp.* is refused, and takes them
    from another (mtp_from), whose identity the pool's key holds: the pool extended for the
    model's own MTP layer is refused, one extended with --mtp-from serves it. Its expert-major
    engine runs its layer runs tiled, as the plain one does."""
    import json
    import runpy
    import sys
    from pathlib import Path

    from safetensors.numpy import save_file

    from opentpu.host import streamed as ST
    from opentpu.host.offload import to_split
    from opentpu.llm.qwen3 import LazyWeights
    from test_qwen35_moe import _tiny

    W, spec = tiny_moe_mtp
    hf = _tiny()[0].config
    hf.tie_word_embeddings = False

    def checkpoint(d, names):
        d.mkdir()
        hf.save_pretrained(d)
        save_file({k: np.ascontiguousarray(W[k]) for k in names}, str(d / "model.safetensors"))
        return d
    m = checkpoint(tmp_path / "model", list(W))
    m2 = checkpoint(tmp_path / "nomtp", [k for k in W if not k.startswith("mtp.")])
    tool = str(Path(__file__).resolve().parents[1] / "tools" / "offload" / "pack_pool.py")

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["pack_pool.py", *map(str, args)])
        try:
            runpy.run_path(tool, run_name="__main__")
        finally:
            monkeypatch.undo()

    a, b, c = (tmp_path / f"{x}.bin" for x in "abc")
    run(m, a, "init")
    run(m, a, "pack", 0, 1)
    run(m, b, "extend", a)
    v = tmp_path / "v.bin"                  # a pool with entries after its experts (vision)
    v.write_bytes(a.read_bytes() + bytes(range(256)) * (2 * a.stat().st_size // 48 // 256))
    Path(str(v) + ".packed").write_bytes(Path(str(a) + ".packed").read_bytes() + bytes([1, 1]))
    for x in (".format", ".key"):
        Path(str(v) + x).write_text(Path(str(a) + x).read_text())
    run(m2, c, "extend", v, "--mtp-from", m)
    nE = spec.layers * spec.moe.E
    pa, pb = (np.fromfile(str(p) + ".packed", np.uint8) for p in (a, b))
    assert pa.tolist() == [1] * nE and pb.tolist() == [1] * nE + [0] * spec.moe.E
    slot = a.stat().st_size // nE
    assert b.read_bytes()[:nE * slot] == c.read_bytes()[:nE * slot] == a.read_bytes()
    assert b.stat().st_size == c.stat().st_size == (nE + spec.moe.E) * slot
    assert Path(str(c) + ".packed").read_bytes() == pb.tobytes()
    kb, kc = (json.loads(Path(str(p) + ".key").read_text()) for p in (b, c))
    assert kb["layout"]["layers"] == spec.layers + 1 and "mtp_from" not in kb
    assert kc["mtp_from"] and {k: v for k, v in kc.items() if k not in ("mtp_from", "model")} \
        == {k: v for k, v in kb.items() if k != "model"}

    prompt = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 9)]
    N = 12
    kw = dict(cap=512, experts=SLOTS, wformat="fp4", head_format="int8")     # (the pools')
    want = ST.engine(m, CFG, pool=a, **kw).generate(prompt, max_new=N)
    eng = ST.engine(m, CFG, pool=b, mtp=True, **kw)
    assert eng.image.rows == 2 and eng.image.offload.layers == spec.layers + 1
    em = dict(kw, expert_major=True, layer_major=2)
    assert ST.engine(m, CFG, pool=a, **em).layer_tile
    assert ST.engine(m, CFG, pool=b, mtp=True, **em).layer_tile
    dec = MTPDecoder(eng)
    assert dec.generate_card(prompt, max_new=N, stop=[]).tokens == want
    got = np.fromfile(str(b) + ".packed", np.uint8)[nE:]
    assert got.any()                        # the MTP layer's experts it asked for, packed
    Wm, img = LazyWeights(m, mtp=True), eng.image
    pool = np.fromfile(b, np.uint8).reshape(-1, slot)
    for e in np.nonzero(got)[0]:
        assert np.array_equal(pool[nE + e], np.asarray(to_split(img.expert(Wm, nE + e)))
                              .view(np.uint8)), e
    with pytest.raises(ValueError, match="no mtp.. tensors"):
        ST.engine(m2, CFG, pool=c, mtp=True, **kw)
    with pytest.raises(ST.PoolError, match="mtp_from differ"):
        ST.engine(m2, CFG, pool=b, mtp=True, mtp_from=m, **kw)
    eng = ST.engine(m2, CFG, pool=c, mtp=True, mtp_from=m, **kw)
    assert MTPDecoder(eng).generate_card(prompt, max_new=N, stop=[]).tokens == want


def test_moe_mtp_chat_turns_roll_back_to_the_reply(tiny_moe_mtp, monkeypatch):
    """otpu-chat's turns with MTP on the streamed MoE (Chat over an MTP engine whose prompt goes
    layer by layer): from the second turn on the context rolls back to the last reply's start
    (the snapshot the turn took there, with the residual row before it: the MTP layer's row
    before it appended again from that row and the new token, MTPDecoder.holds) and feeds the
    reply and the new message, not the whole conversation (restarted False). Chat.fed holds every token fed, the first
    prompt's conv_k - 1 rows of the rows kernel too (without them the next turn found no common
    part and started over). Each turn's reply equals plain greedy decode's of its rendering."""
    from opentpu.host.chat import Chat, sampler
    from opentpu.llm import qwen35 as Q35
    from test_chatkv import MESSAGES, _QwenTok
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    W, spec = tiny_moe_mtp
    tok = _QwenTok()
    chat = Chat(_layered(spec, W, False, snapshots=1), tok, False, sampler(0, 0, 1.0, None), 6)
    assert chat.use_mtp and chat.mtp.by_layers
    start = None
    for i, msg in enumerate(MESSAGES[:3]):
        reply, turn = chat.ask(msg)
        ids = tok.apply_chat_template(chat.history[:-1], True, False, True)
        assert not turn.restarted, i
        assert chat.fed[:len(ids)] == ids, i
        assert turn.reused == (start if i else 0), i
        want = Engine(spec, W, cap=512, cfg=CFG, rows=1, resident=True,
                      experts=SLOTS).generate(ids, max_new=6)
        assert chat._reply == [x for x in want if x not in spec.eos], i
        start = len(tok.apply_chat_template(chat.history, False, False, True)) - \
            len(chat.history[-1]["content"]) - len(f"{tok.E}\n")


def test_chatmoe_traces_each_turns_host_phases(tmp_path, monkeypatch, capsys):
    """tools/qual/chatmoe.py --trace's host phases (_Phases) on a tiny MoE of 16 DeltaNet heads
    with an LM head of its own and its MTP layer, from a checkpoint and pools as the card's
    (pack_pool.py init, pack, extend), three turns that roll back to the reply's start (a
    Qwen3-shaped template): each turn's phases from its start to its first token, a0 last, the
    calls' as `calls` (each with its end_prefill), the first rows of the MTP decoder's prompt
    in a prefill of their own with their runs, a later turn's rollback before its calls and
    the snapshot (MTP's with the held row's read) between them. Host bookkeeping: the traced
    run's tokens are the untraced one's (--check), plain decode's and MTP's the same, and the
    trace keeps its calls as before."""
    import json
    import pickle
    import runpy
    import sys
    from pathlib import Path

    from safetensors.numpy import save_file

    from test_chatkv import MESSAGES, _QwenTok
    root = Path(__file__).resolve().parents[1]
    m, W, spec = _tiny(nv=16)
    spec, W = _untied((m, W, spec))
    W = _moe_mtp_weights(W, spec)
    m.config.tie_word_embeddings = False
    d = tmp_path / "model"
    m.config.save_pretrained(d)
    save_file({k: np.ascontiguousarray(v) for k, v in W.items()}, str(d / "model.safetensors"))

    def pack(*args):
        monkeypatch.setattr(sys, "argv", ["pack_pool.py", *map(str, args)])
        try:
            runpy.run_path(str(root / "tools" / "offload" / "pack_pool.py"), run_name="__main__")
        finally:
            monkeypatch.undo()
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    pack(d, a, "init")
    pack(d, a, "pack", 0, 1)
    pack(d, b, "extend", a)
    cfg = tmp_path / "cfg.pkl"
    cfg.write_bytes(pickle.dumps(board_config(DRAM_BYTES=1 << 26, DSTEP=True, STREAM=True,
                                              PAIR=True)))

    class Tok(_QwenTok):                    # (chatmoe's _Pinned passes the template a date)
        def apply_chat_template(self, history, add_generation_prompt=True,
                                enable_thinking=False, tokenize=True, **kw):
            return super().apply_chat_template(history, add_generation_prompt,
                                               enable_thinking, tokenize)
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda p: Tok())
    sys.path.insert(0, str(root / "tools" / "qual"))
    try:
        import chatmoe
    finally:
        sys.path.remove(str(root / "tools" / "qual"))
    argv = [str(d), "--cfg", str(cfg), "--cap", "512", "--max-new", "6", "--em-rows", "8"]
    argv += [x for msg in MESSAGES[:3] for x in ("--turn", msg)]

    def run(*args):
        assert chatmoe.main(argv + [str(x) for x in args]) == 0
        return capsys.readouterr().out
    run("--pool", a, "--trace", tmp_path / "ta.json", "--out", tmp_path / "a.json")
    run("--pool", b, "--mtp", "--out", tmp_path / "b.json")
    out = run("--pool", b, "--mtp", "--trace", tmp_path / "tb.json", "--out", tmp_path / "bt.json",
              "--check", tmp_path / "b.json")
    assert "[PASS]" in out and out.count("    phases: ") == 3
    ra, rb, rt = (json.loads((tmp_path / f"{x}.json").read_text()) for x in ("a", "b", "bt"))
    assert rb["mtp"] and [t["tokens"] for t in ra["turns"]] == [t["tokens"] for t in rb["turns"]]
    assert all(not t["restarted"] and t["reused"] for t in rb["turns"][1:])
    assert any(c["em"] for t in rb["turns"] for c in t["calls"])
    for x, r in (("ta", ra), ("tb", rt)):
        tr = json.loads((tmp_path / f"{x}.json").read_text())
        assert tr["run_fields"] == ["key", "start", "end", "cycles"] and \
            tr["event_fields"] == ["seen", "done", "kind", "layer or expert", "misses or bytes"]
        assert [(c["rows"], c["em"]) for c in tr["calls"]] == \
            [(c["rows"], c["em"]) for t in r["turns"] for c in t["calls"]]
        assert tr["phase_fields"] == ["phase", "start", "end", "cycles"]
        assert [t["calls"] for t in tr["turns"]] == \
            [[k - len(t["calls"]), k] for k, t in
             zip(np.cumsum([len(t["calls"]) for t in r["turns"]]).tolist(), r["turns"])]
        for i, (t, rt) in enumerate(zip(tr["turns"], r["turns"])):
            ph = t["phases"]
            assert t["ttft_s"] == pytest.approx(rt["ttft_s"], abs=1e-3), (x, i)
            assert {p[0] for p in ph} <= set(chatmoe.PHASES), (x, i)
            assert all(0 <= p[1] <= p[2] for p in ph), (x, i)
            a0 = [p for p in ph if p[0] == "a0"]
            assert len(a0) == 1 and a0[0][2] <= t["ttft_s"] + 1e-3, (x, i)
            assert all(p[1] <= a0[0][1] for p in ph), (x, i)        # (none starts after a0)
            calls = [p for p in ph if p[0] == "prefill_layers"]
            assert len(calls) == len(rt["calls"]), (x, i)
            for p, c in zip(calls, rt["calls"]):        # (a call's s is inside its span)
                assert p[2] - p[1] >= c["s"] - 1e-3, (x, i)
                assert sum(q[0] == "end_prefill" and p[1] <= q[1] <= q[2] <= p[2]
                           for q in ph) == 1, (x, i)
            runs = [p for p in ph if p[0] == "run"]
            assert not any(p[1] < q[1] < p[2] for p in calls for q in runs), (x, i)
            if i:                       # (the rollback first, the snapshot between the calls)
                rb_ = [p for p in ph if p[0] == "rollback"]
                assert len(rb_) == 1 and rb_[0][2] <= calls[0][1], (x, i)
            snap = [p for p in ph if p[0] == "snapshot"]
            assert len(snap) == 1 and calls[0][2] <= snap[0][1] <= snap[0][2] <= calls[-1][1]
            held = [p for p in ph if p[0] == "mtp snapshot"]
            assert len(held) == (x == "tb"), (x, i)
            assert not held or held[0][1] <= snap[0][1] <= snap[0][2] <= held[0][2], (x, i)
            nest = [p for p in ph if p[0] == "mtp prefill"]
            if x == "ta":
                assert not nest, i
            else:                       # (the part before the snapshot, the rest; the first
                inner = [p for p in nest if any(q is not p and q[1] <= p[1] <= p[2] <= q[2]
                                                for q in nest)]     # turn's first rows: a
                assert len(nest) == 2 + (i == 0) and len(inner) == (i == 0), i  # prefill and
                for p in inner:                                             # runs of their own
                    assert p[2] <= calls[0][1], i
                    assert sum(p[1] <= q[1] <= q[2] <= p[2] for q in runs) >= 2, i
