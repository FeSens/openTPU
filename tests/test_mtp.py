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
        dn = m.layer(li, "linear").dn
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


def _plain(spec, W, prompt, N, kw):
    """Plain greedy decode's tokens (resident decode, the host loop)."""
    return Engine(spec, W, cap=512, cfg=CFG, resident=True, **kw).generate(prompt, max_new=N)


@pytest.mark.parametrize("P, drafter, fmt, N, stop", [
    (250, "right", "int8", 16, False),     # V at 254 accepts: the wrap; the last token alone
    (251, "right", "int8", 15, False),     # 253 -> 255: E, then D1 and the next bucket's V
    (249, "mixed", "int8", 15, False),     # E at the other parity
    (250, "wrong", "int8", 12, False),
    (249, "right", "fp4", 15, False),
    (250, "mtp", "emb8", 15, False),       # int8 embedding: 10 run-time values in D
    (252, "right", "int8", 15, True)])     # a stop id: the second of an accepted pair
def test_mtp_loop_on_the_device_is_plain_greedy(P, drafter, fmt, N, stop):
    """The MTP loop on the device (docs/mtp.md 10: V, E, D, D1 chained through the buckets'
    programs, ISA simulator) gives plain greedy decode's tokens across the end of the first
    attention bucket, whatever the drafts (the device's MTP, or the host's table: all right,
    every third wrong, all wrong), and leaves the context as plain decode does: the committed
    DeltaNet states and windows equal, word for word, those of the prompt and the tokens but
    the last fed one by one."""
    _, W, spec = _tiny_model(8)
    if fmt == "emb8":
        spec = dataclasses.replace(spec, embed="int8")
    W = _mtp_weights(W, spec)
    kw = FP4 if fmt == "fp4" else {}
    prompt = [int(t) for t in np.random.default_rng(P).integers(0, 1000, P)]
    want = _plain(spec, W, prompt, N, kw)
    ids = None
    if stop:                                # the first token of a second kind stops it
        k = next((i for i, t in enumerate(want) if t != want[0]), len(want) - 1)
        ids = [want[k]]
        want = want[:k + 1]
    right = np.zeros(514, np.float32)
    right[P:P + len(want)] = want
    drafts = {"right": right, "wrong": (right + 1) % 1000, "mtp": None,
              "mixed": np.where(np.arange(514) % 3, right, (right + 7) % 1000)}[drafter]
    eng = mtp_engine(spec, W, cap=512, cfg=CFG, **kw)
    dec = MTPDecoder(eng)
    st = dec.generate_card(prompt, max_new=N, stop=ids, drafts=drafts)
    assert st.tokens == want
    if drafter == "right" and not stop:
        assert sum(st.accepted) >= len(want) // 2 - 2
    if drafter == "wrong":
        assert not any(st.accepted)
    ref = Engine(spec, W, cap=512, cfg=CFG, resident=True, **kw)
    ref.prefill(prompt)
    for t in want[:-1]:
        ref.step(t)
    assert eng.pos == ref.pos == P + len(want) - 1
    assert all(np.array_equal(a, b) for a, b in
               zip(_states(eng, spec, dec.slot), _states(ref, spec, 0)))


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
