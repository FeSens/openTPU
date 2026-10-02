"""Speculative decoding with Qwen3.5's MTP drafter (opentpu/llm/mtp.py, docs/mtp.md 9) on the ISA
simulator: greedy tokens equal plain greedy decode's bit for bit whatever the drafts (the MTP's,
all right, all wrong, mixed), the committed DeltaNet states and windows equal plain decode's, and
the MTP program follows its numpy reference. A tiny random Qwen3.5 with random mtp.* weights;
int8 weights, and fp4 with PAIR at the board's MCOLS 4 (the 2-row verify's MMs pair as the
decode steps' do, qwen35_rows)."""
import dataclasses

import numpy as np
import pytest

from opentpu import isa as I
from opentpu.compiler import Affine
from opentpu.isasim import board_config
from opentpu.llm import generate as G
from opentpu.llm.mtp import MTPDecoder, MTPStats, mtp_engine
from opentpu.llm.qwen3 import Engine
from opentpu.llm.qwen35 import mtp_reference, qwen35_step, reference_logits

from test_qwen35 import _tiny_model

pytest.importorskip("torch")

CFG = board_config(DRAM_BYTES=1 << 25, DSTEP=True, STREAM=True, PAIR=True)


def _mtp_weights(W, spec, seed=1):
    """Random mtp.* tensors for a tiny model (zero-centred norms, as the checkpoint's)."""
    r = np.random.default_rng(seed)
    H, d, F = spec.hidden, spec.head_dim, spec.ffn

    def lin(o, i):
        return (r.uniform(-1, 1, (o, i)) / np.sqrt(i)).astype(np.float32)

    def norm(n):
        return (0.1 * r.standard_normal(n)).astype(np.float32)
    a = "mtp.layers.0.self_attn."
    m = {"mtp.pre_fc_norm_embedding.weight": norm(H), "mtp.pre_fc_norm_hidden.weight": norm(H),
         "mtp.fc.weight": lin(H, 2 * H), "mtp.norm.weight": norm(H),
         "mtp.layers.0.input_layernorm.weight": norm(H),
         "mtp.layers.0.post_attention_layernorm.weight": norm(H),
         a + "q_proj.weight": lin(2 * spec.n_q * d, H), a + "k_proj.weight": lin(spec.n_kv * d, H),
         a + "v_proj.weight": lin(spec.n_kv * d, H), a + "o_proj.weight": lin(H, spec.n_q * d),
         a + "q_norm.weight": norm(d), a + "k_norm.weight": norm(d),
         "mtp.layers.0.mlp.gate_proj.weight": lin(F, H),
         "mtp.layers.0.mlp.up_proj.weight": lin(F, H),
         "mtp.layers.0.mlp.down_proj.weight": lin(H, F)}
    return {**W, **m}


FP4 = dict(wformat="fp4", head_format="int8")


@pytest.fixture(scope="module", params=[(8, False, False), (4, False, False), (8, True, False),
                                        (8, False, True)],
                ids=["kh8", "kh4-shared", "kh8-grouped", "kh8-fp4"])
def tiny_mtp(request):
    """(weights, spec, formats)."""
    nk, grouped, fp4 = request.param
    _, W, spec = _tiny_model(nk)
    return _mtp_weights(W, spec), dataclasses.replace(spec, pair_loop=grouped), FP4 if fp4 else {}


def _layers(eng):
    return eng.backend.machine.slices[0].dram


def _states(eng, spec, slot):
    """Every DeltaNet layer's states and windows in `slot` (bytes)."""
    img, dram = eng.image, _layers(eng)
    m = img.descriptors(0, slot)
    out = []
    for li, k in enumerate(spec.kinds):
        if k != "linear":
            continue
        dn = m.layer(li).dn
        for q in range(img.nl // 2):
            for t in [dn.state(q, 0), dn.state(q, 1), dn.window(q)]:
                n, b = 4 * int(np.prod(t.shape)), Affine.of(t.base).static()
                out.append(dram[b:b + n].copy())
    return out


@pytest.mark.parametrize("drafter", ["mtp", "right", "wrong", "mixed"])
def test_mtp_greedy_is_plain_greedy(tiny_mtp, drafter):
    """Tokens equal plain greedy decode's for any drafts: the MTP's (random weights: mostly
    rejected), all right (every iteration accepts and flips the state slot), all wrong, and
    right but every third; then the committed states and windows equal plain decode's word
    for word, and a decode step from the committed slot gives plain decode's next logits.
    The prompt's 13 tokens prefill in runs of 8, 4 and 1 rows (the MTP layer over 4 + 4)."""
    W, spec, kw = tiny_mtp
    prompt = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 13)]
    N = 20
    ref = Engine(spec, W, cap=256, cfg=CFG, resident=True, **kw)
    want = ref.generate(prompt, max_new=N)
    P = len(prompt)

    def right(q, out):
        return want[q - P] if q - P < len(want) else 0
    drafts = {"mtp": None, "right": right,
              "wrong": lambda q, out: (right(q, out) + 1) % 1000,
              "mixed": lambda q, out: right(q, out) if q % 3 else (right(q, out) + 7) % 1000}
    eng = mtp_engine(spec, W, cap=256, cfg=CFG, **kw)
    dec = MTPDecoder(eng)
    st = dec.generate(prompt, max_new=N, drafts=drafts[drafter])
    assert st.tokens == want
    if drafter == "right":
        assert all(st.accepted[:-1]) and st.iterations <= N // 2 + 1
    if drafter == "wrong":
        assert not any(st.accepted)
    # the same context as plain greedy decode feeds it: the prompt, then the tokens but the
    # last one by one (decode steps)
    ref = Engine(spec, W, cap=256, cfg=CFG, resident=True, **kw)
    ref.prefill(prompt)
    # the prompt in plain prefill's runs (their MMs pair as plain prefill's do)
    assert [r for k, r, _ in st.runs if k == "prefill"] == [x.get("rows", 1) for x in ref.stats]
    for t in want[:-1]:
        ref.step(t)
    assert eng.pos == ref.pos == P + N - 1
    assert all(np.array_equal(a, b) for a, b in
               zip(_states(eng, spec, dec.slot), _states(ref, spec, 0)))
    t, pos = want[-1], eng.pos
    lg = []
    for e, slot in ((eng, dec.slot), (ref, 0)):
        prog = qwen35_step.trace(e.cfg, 0, {"m": e.image.descriptors(0, slot), "pos": pos,
                                            "block": e.block, "tok": t}).finish()
        e.backend.run([prog])
        io = e.image.io
        lg.append(e.backend.read(0, io["logits"], 4 * spec.vocab).view(np.uint32))
    assert np.array_equal(lg[0], lg[1])


def test_mtp_program_follows_its_reference():
    """The MTP program over a 4-row prompt chunk (its hidden rows from the rows kernel's
    `hidden`): the draft head's logits against mtp_reference on the fp32 model's hidden rows,
    and its draft ids against their argmax."""
    _, W, spec = _tiny_model(8)
    W = _mtp_weights(W, spec)
    R = 4
    toks = [int(t) for t in np.random.default_rng(6).integers(0, 1000, R + 1)]
    eng = mtp_engine(spec, W, cap=256, cfg=CFG)
    img = eng.image
    eng.backend.run(img.compile_rows([(0, j) for j in range(R)], [], tokens=toks[:R],
                                     hidden=True))
    hid = eng.backend.read(0, img.io["hid"], 4 * R * spec.hidden).view(np.float32)
    _, href = reference_logits(spec, W, toks[:R], hidden=True)
    hid = hid.reshape(R, spec.hidden)
    cos = (hid * href).sum(1) / np.linalg.norm(hid, axis=1) / np.linalg.norm(href, axis=1)
    assert cos.min() > 0.999, cos
    eng.backend.run(img.compile_mtp(0, R, toks[1:R + 1], keep=True))
    got = eng.backend.read(0, img.io["logits"], 4 * R * spec.vocab).view(np.float32)
    got = got.reshape(R, spec.vocab)[:, :img.nd]
    want = mtp_reference(spec, W, href, toks[1:R + 1])
    cos = (got * want).sum(1) / np.linalg.norm(got, axis=1) / np.linalg.norm(want, axis=1)
    assert cos.min() > 0.99, cos
    ids = eng.backend.read(0, img.io["draft"], 4 * R).view(np.float32).astype(int)
    assert list(ids) == list(np.argmax(got, axis=1))
    assert np.mean(ids == np.argmax(want, axis=1)) >= 0.75


def test_mtp_verify_and_draft_on_rtl(have_verilator):
    """A forked verify run (row 1's DeltaNet steps a STREAM into the other state slot, dst !=
    src) and an MTP run on the RTL's board memory path: DRAM equals the ISA simulator's."""
    from opentpu import rtlsim
    from opentpu.isasim import Machine
    _, W, spec = _tiny_model(8)
    W = _mtp_weights(W, spec)
    toks = [int(t) for t in np.random.default_rng(7).integers(0, 1000, 8)]
    eng = mtp_engine(spec, W, cap=256, cfg=CFG)
    dec = MTPDecoder(eng)
    a0, d = dec.prefill(toks[:6], MTPStats())
    img, n = eng.image, eng.image.nbytes
    verify = img.compile_rows([(0, 6), (0, 7)], [0, 1], tokens=[a0, d], fork=True, hidden=True)
    assert any(i.op == I.STREAM and i.w[1] != i.w[2] for i in verify[0])
    for progs in (verify, img.compile_mtp(6, 2, [a0, d])):
        dram = eng.backend.machine.slices[0].dram[:n].copy()
        m = Machine(CFG, progs, [dram.copy()]).run()
        drams, _, _ = rtlsim.run(CFG, progs, [dram.copy()], uarch=rtlsim.BOARD_UARCH, axi=True,
                                 boot=True)
        assert np.array_equal(drams[0][:n], m.slices[0].dram[:n])
        eng.backend.machine.slices[0].dram[:n] = m.slices[0].dram[:n]


def test_mtp_needs_paired_verify():
    """With PAIR and 4-bit weights at MCOLS 2 a 2-row run's MMs do not pair as the decode
    steps' do (qwen35_rows): the decoder refuses rather than give other tokens."""
    _, W, spec = _tiny_model(8)
    cfg = board_config(DRAM_BYTES=1 << 25, DSTEP=True, STREAM=True, PAIR=True, MCOLS=2)
    eng = mtp_engine(spec, _mtp_weights(W, spec), cap=256, cfg=cfg, **FP4)
    with pytest.raises(ValueError, match="MCOLS >= 4"):
        MTPDecoder(eng)


def _plain(spec, W, prompt, N, kw, cfg=CFG):
    """Plain greedy decode's tokens (resident decode, the host loop)."""
    return Engine(spec, W, cap=512, cfg=cfg, resident=True, **kw).generate(prompt, max_new=N)


@pytest.mark.parametrize("P, drafter, fmt, N, stop", [
    (250, "right", "int8", 16, False),     # V at 254 accepts: the wrap; the last token alone
    (251, "right", "int8", 15, False),     # 253 -> 255: E, then D1 and the next bucket's V
    (249, "mixed", "int8", 15, False),     # E at the other parity
    (250, "wrong", "int8", 12, False),
    (249, "right", "fp4", 15, False),
    (250, "mtp", "emb8", 15, False),       # int8 embedding: 10 run-time values in D
    (252, "right", "int8", 15, True),      # a stop id: the second of an accepted pair
    (30, "right", "kh16", 6, False)])      # 16 DeltaNet heads: V's arguments move
def test_mtp_loop_on_the_device_is_plain_greedy(P, drafter, fmt, N, stop):
    """The MTP loop on the device (docs/mtp.md 10: V, E, D, D1 chained through the buckets'
    programs, ISA simulator) gives plain greedy decode's tokens across the end of the first
    attention bucket, whatever the drafts (the device's MTP, or the host's table: all right,
    every third wrong, all wrong), and leaves the context as plain decode does: the committed
    DeltaNet states and windows equal, word for word, those of the prompt and the tokens but
    the last fed one by one. kh16: the 0.8B's and 2B's 16 DeltaNet heads, whose addresses
    take an argument register of the verify's, so its last arguments move into the released
    registers of the row tokens', each at its own release (Builder._move_arg)."""
    _, W, spec = _tiny_model(16, 16) if fmt == "kh16" else _tiny_model(8)
    cfg = board_config(DRAM_BYTES=1 << 26, DSTEP=True, STREAM=True, PAIR=True) \
        if fmt == "kh16" else CFG
    if fmt == "emb8":
        spec = dataclasses.replace(spec, embed="int8")
    W = _mtp_weights(W, spec)
    kw = FP4 if fmt == "fp4" else {}
    prompt = [int(t) for t in np.random.default_rng(P).integers(0, 1000, P)]
    want = _plain(spec, W, prompt, N, kw, cfg)
    ids = None
    if stop:                                # the first token of a second kind stops it
        k = next((i for i, t in enumerate(want) if t != want[0]), len(want) - 1)
        ids = [want[k]]
        want = want[:k + 1]
    right = np.zeros(514, np.float32)
    right[P:P + len(want)] = want
    drafts = {"right": right, "wrong": (right + 1) % 1000, "mtp": None,
              "mixed": np.where(np.arange(514) % 3, right, (right + 7) % 1000)}[drafter]
    eng = mtp_engine(spec, W, cap=512, cfg=cfg, **kw)
    dec = MTPDecoder(eng)
    st = dec.generate_card(prompt, max_new=N, stop=ids, drafts=drafts)
    assert st.tokens == want
    if drafter == "right" and not stop:
        assert sum(st.accepted) >= len(want) // 2 - 2
    if drafter == "wrong":
        assert not any(st.accepted)
    ref = Engine(spec, W, cap=512, cfg=cfg, resident=True, **kw)
    ref.prefill(prompt)
    for t in want[:-1]:
        ref.step(t)
    assert eng.pos == ref.pos == P + len(want) - 1
    assert all(np.array_equal(a, b) for a, b in
               zip(_states(eng, spec, dec.slot), _states(ref, spec, 0)))


# (the tiny model's initializer range, the sampling): at 0.02 the tied head mostly gives back
# the input token, so the penalty of row 1's draft decides picks; at 0.2 the tokens vary
SAMPLINGS = {"pen": (0.02, (0.8, 5, 0.9, 1.5)), "greedy-pen": (0.02, (0.0, 0, 1.0, 3.0)),
             "free": (0.2, (1.5, 20, 1.0, 1.0)), "varied-pen": (0.2, (0.8, 5, 0.9, 1.1))}


def _plain_sampled(spec, W, prompt, N, samp, seed, cfg, stop=()):
    """Plain decode's sampled tokens (the device sampler's numpy model, generate.reference_pick,
    on resident decode's logits) with the uniforms of default_rng(seed), as the device takes
    them (fp32, below 1), that engine (the prompt and the tokens but the last fed), and the
    number of picks that the penalty of the token before decides (new to the context: a
    verify's row 1 must count its draft)."""
    ref = Engine(spec, W, cap=512, cfg=cfg, resident=True)
    u = np.minimum(np.random.default_rng(seed).random(N).astype(np.float32),
                   np.float32(1 - 2.0 ** -24))
    c, out, lg, hits = list(prompt), [], ref.prefill(prompt), 0
    for i in range(N):
        out.append(G.reference_pick(lg, samp, c, u[i]))
        if i and c[-1] not in c[:-1]:
            hits += out[-1] != G.reference_pick(lg, samp, c[:-1], u[i])
        c.append(out[-1])
        if out[-1] in stop or i == N - 1:
            return out, ref, hits
        lg = ref.step(out[-1])


@pytest.mark.parametrize("P, drafter, how, N, stop", [
    (250, "right", "pen", 16, False),       # across the bucket's end, accepted pairs
    (251, "mixed", "pen", 15, False),       # E at 255, then D1 and the next bucket's V
    (249, "mtp", "pen", 15, False),
    (250, "right", "greedy-pen", 16, False),
    (250, "mixed", "free", 16, False),      # no penalty: top-k 20, no top-p
    (250, "right", "varied-pen", 16, False),
    (252, "right", "pen", 15, True),        # a stop id
    (30, "right", "kh16", 6, False)])       # 16 DeltaNet heads (the 0.8B's and 2B's)
def test_mtp_sampled_loop_is_plain_sampling(P, drafter, how, N, stop):
    """The sampled MTP loop on the device (docs/mtp.md 11) gives plain sampled decode's tokens
    for the same uniforms, whatever the drafts: the verify's row 0 samples a0 with position p +
    1's uniform and accepts the draft iff a0 equals it, row 1 samples a1 with p + 2's, its
    repetition penalty counting the draft (new to the context, which decides picks here, or
    not); the context is left as plain decode leaves it."""
    kh16 = how == "kh16"
    init, sp = SAMPLINGS["pen" if kh16 else how]
    _, W, spec = _tiny_model(16, 16, init=init) if kh16 else _tiny_model(8, init=init)
    cfg = board_config(DRAM_BYTES=1 << 26, DSTEP=True, STREAM=True, PAIR=True) if kh16 else CFG
    W = _mtp_weights(W, spec)
    samp = G.Sampling(*sp)
    prompt = [int(t) for t in np.random.default_rng(P).integers(0, 1000, P)]
    want, ref, hits = _plain_sampled(spec, W, prompt, N, samp, 5, cfg)
    ids = []
    if stop:                                # the first token of a second kind stops it
        k = next((i for i, t in enumerate(want) if t != want[0]), len(want) - 1)
        ids = [want[k]]
        want, ref, _ = _plain_sampled(spec, W, prompt, N, samp, 5, cfg, ids)
        assert len(want) == k + 1
    assert len(set(want)) > len(want) // 2
    if how in ("pen", "greedy-pen") and not stop:
        assert hits >= 3
    if how == "pen" and not stop:           # drafts in the context already, too
        assert any(t in want[:i] for i, t in enumerate(want))
    right = np.zeros(514, np.float32)
    right[P:P + len(want)] = want
    drafts = {"right": right, "mtp": None,
              "mixed": np.where(np.arange(514) % 3, right, (right + 7) % 1000)}[drafter]
    eng = mtp_engine(spec, W, cap=512, cfg=cfg)
    dec = MTPDecoder(eng)
    st = dec.generate_card(prompt, max_new=N, stop=ids, drafts=drafts, sampling=samp,
                           rng=np.random.default_rng(5))
    assert st.tokens == want
    if drafter == "right" and not stop:
        assert sum(st.accepted) >= len(want) // 2 - 2
    assert eng.pos == ref.pos == P + len(want) - 1
    assert all(np.array_equal(a, b) for a, b in
               zip(_states(eng, spec, dec.slot), _states(ref, spec, 0)))


def _mtp_kv(eng, n):
    """The MTP layer's KV cache at positions [0, n): K, its scales and V's, per head."""
    kv = eng.image.descriptors(0, 0).mtp.layer.kv
    d, D, rd = kv.d, kv.D, eng.backend.read
    return [x for _, r in sorted(kv.heads.items())
            for x in (rd(0, Affine.of(r["k"]).const, n * d),
                      rd(0, Affine.of(r["ks"]).const, n * 4 * (d // D)),
                      rd(0, Affine.of(r["vs"]).const, 4 * n))]


@pytest.mark.parametrize("sampled", [False, True])
def test_mtp_continues_its_context(sampled):
    """A chat's next turn: after the loop's run, a prefill from Engine.pos (the last token,
    then the new ones) and a second run continue the context. The loop's last D (it runs
    before the HALT) fills the MTP layer's KV cache at the position before the last token, so
    the cache, the drafts and the tokens equal those of one fresh decoder over the whole
    conversation, bit for bit (without it, that position's K differs)."""
    _, W, spec = _tiny_model(8, init=0.2)
    W = _mtp_weights(W, spec)
    r = np.random.default_rng(3)
    p1, p2 = [int(t) for t in r.integers(0, 1000, 40)], [int(t) for t in r.integers(0, 1000, 9)]
    samp = G.Sampling(0.8, 5, 0.9, 1.1) if sampled else None

    def kw(seed):
        return {} if samp is None else dict(sampling=samp, rng=np.random.default_rng(seed))
    dec = MTPDecoder(mtp_engine(spec, W, cap=512, cfg=CFG))
    t1 = dec.generate_card(p1, max_new=12, stop=[], **kw(1)).tokens
    ctx = p1 + t1 + p2
    st = dec.generate_card([t1[-1]] + p2, max_new=12, stop=[], context=ctx, **kw(2))
    fresh = MTPDecoder(mtp_engine(spec, W, cap=512, cfg=CFG))
    want = fresh.generate_card(ctx, max_new=12, stop=[], **kw(2))
    assert st.tokens == want.tokens and len(set(st.tokens)) > 6
    assert dec.eng.pos == fresh.eng.pos == len(ctx) + 11 and dec.draft == fresh.draft
    assert all(np.array_equal(a, b) for a, b in
               zip(_mtp_kv(dec.eng, dec.eng.pos), _mtp_kv(fresh.eng, fresh.eng.pos)))


def test_mtp_loop_on_rtl(have_verilator):
    """The MTP loop on the Verilator RTL (the board's memory path): from the same DRAM state (a
    251-token prefill on the ISA simulator), one run of the loop across the first bucket's end (V, E, D, D1 and the next
    bucket's V through HALT CHAIN, both parities, drafts from the host's table, every third
    wrong): the tokens and the image's DRAM equal the ISA simulator's."""
    from opentpu import rtlsim
    from opentpu.llm.rtl_backend import RtlBackend
    _, W, spec = _tiny_model(8)
    W = _mtp_weights(W, spec)
    P, N = 251, 9
    prompt = [int(t) for t in np.random.default_rng(P).integers(0, 1000, P)]
    want = _plain(spec, W, prompt, N, {})
    right = np.zeros(514, np.float32)
    right[P:P + N] = want
    drafts = np.where(np.arange(514) % 3, right, (right + 7) % 1000)
    eng = mtp_engine(spec, W, cap=512, cfg=CFG)
    dec = MTPDecoder(eng)
    a0, d = dec.prefill(prompt, MTPStats())
    isa, n = eng.backend, eng.image.nbytes
    rtl = RtlBackend(eng.cfg, [isa.machine.slices[0].dram[:n]], uarch=rtlsim.BOARD_UARCH,
                     axi=True, boot=True)
    st = dec.loop_card(a0, d, N, drafts=drafts)
    assert st.tokens == want and 0 < sum(st.accepted) < st.iterations
    eng.backend, eng.pos, dec.slot = rtl, P, 0
    eng.__dict__.pop("_mtp_gen")            # the chain area: written again, to the RTL's DRAM
    got = dec.loop_card(a0, d, N, drafts=drafts)
    assert got.tokens == want
    assert np.array_equal(isa.machine.slices[0].dram[:n], rtl.drams[0][:n])


def test_mtp_sampled_loop_on_rtl(have_verilator):
    """The sampled MTP loop on the Verilator RTL: from the same DRAM state (a 251-token
    prefill on the ISA simulator, a0 the sampler's pick), one run across the first bucket's end
    with the repetition penalty (row 1's draft counted), drafts from the host's table, every
    third wrong: the tokens and the image's DRAM equal the ISA simulator's."""
    from opentpu import rtlsim
    from opentpu.llm.rtl_backend import RtlBackend
    _, W, spec = _tiny_model(8)
    W = _mtp_weights(W, spec)
    P, N = 251, 9
    samp = G.Sampling(*SAMPLINGS["pen"][1])
    prompt = [int(t) for t in np.random.default_rng(P).integers(0, 1000, P)]
    want, _, _ = _plain_sampled(spec, W, prompt, N, samp, 5, CFG)
    right = np.zeros(514, np.float32)
    right[P:P + N] = want
    drafts = np.where(np.arange(514) % 3, right, (right + 7) % 1000)
    eng = mtp_engine(spec, W, cap=512, cfg=CFG)
    dec = MTPDecoder(eng)
    u0 = np.minimum(np.float32(np.random.default_rng(5).random()), np.float32(1 - 2.0 ** -24))
    a0, d = dec.prefill(prompt, MTPStats(), pick=lambda lg: G.reference_pick(lg, samp, prompt, u0))
    isa, n = eng.backend, eng.image.nbytes
    rtl = RtlBackend(eng.cfg, [isa.machine.slices[0].dram[:n]], uarch=rtlsim.BOARD_UARCH,
                     axi=True, boot=True)

    def rng():                              # the uniforms after a0's
        r = np.random.default_rng(5)
        r.random()
        return r
    kw = dict(drafts=drafts, samp=samp, context=prompt + [a0])
    st = dec.loop_card(a0, d, N, rng=rng(), **kw)
    assert st.tokens == want and 0 < sum(st.accepted) < st.iterations
    eng.backend, eng.pos, dec.slot = rtl, P, 0
    eng.__dict__.pop("_mtp_gen")            # the chain area: written again, to the RTL's DRAM
    got = dec.loop_card(a0, d, N, rng=rng(), **kw)
    assert got.tokens == want
    assert np.array_equal(isa.machine.slices[0].dram[:n], rtl.drams[0][:n])


class _Chars:
    """One token per character, both ways: a reply's text gives back its ids, so the next
    turn's template starts with the fed tokens (the context continues)."""

    def apply_chat_template(self, history, add_generation_prompt, enable_thinking, tokenize):
        return [ord(c) for m in history for c in m["content"]]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(map(chr, ids))


@pytest.mark.parametrize("sampled", [False, True])
def test_chat_with_mtp(sampled):
    """otpu-chat --mtp: Chat on an MTP engine decodes with MTPDecoder (its prefill, the host
    picking the first token, then the loop on the device), turn after turn. The replies, what
    is fed and the context equal those of Chat on a plain engine with the same sampler and
    seed: a reply cut at max_new resumes from its last token (the draft after it kept), and
    the next turn's prefill continues the context."""
    from opentpu.host.chat import Chat, sampler
    _, W, spec = _tiny_model(8, init=0.2)
    W = _mtp_weights(W, spec)
    args = (0.8, 5, 0.9, 3, 1.1) if sampled else (0, 0, 1.0, None)
    plain, mtp = (Chat(e, _Chars(), False, sampler(*args), 10) for e in (
        Engine(spec, W, cap=512, cfg=CFG, resident=True), mtp_engine(spec, W, cap=512, cfg=CFG)))
    assert mtp.use_mtp and not plain.use_mtp and plain.on_card
    got = {}
    for c in (plain, mtp):
        shown = []
        r1, t1 = c.ask("hello there", lambda d, turn: shown.append(d))
        r2, t2 = c.resume()
        r3, t3 = c.ask("and then?")
        assert t1.end == "max_new" and not t3.restarted and r1 == "".join(shown)
        got[c is mtp] = (r1, r2, r3, list(c.fed), c.eng.pos, [t.gen_tokens for t in (t1, t2, t3)])
        if c is mtp:
            assert t1.mtp_iters and "MTP acceptance" in t3.line()
    assert got[True] == got[False]
    assert len(set(got[True][2])) > 4
