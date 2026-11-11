"""Rolling a context back (opentpu/llm/snapshot.py: Engine.rollback, Engine.snapshot) and
otpu-chat's turns on it (opentpu/host/chat.py, Chat): on the ISA simulator, a context that
goes back to a position and is fed the rest from there gives the logits, the next greedy
tokens and the state of a fresh engine fed the same tokens, bit for bit. The KV caches go back
to any position (Qwen3, the Llama-likes); the recurrent and windowed state comes from a
snapshot taken on the way (Qwen3.5's DeltaNet states and windows, plain and MTP; LFM2's
convolution rings; Gemma 4's sliding rings past their slack), copied on the device (LD / ST
programs) or on the host. A chat whose template renders the last reply otherwise than it was
fed (Qwen3's and Qwen3.5's: the thinking block) rolls back to the reply's start every turn
instead of feeding the whole conversation again. int8 weights: a row's arithmetic is the decode
kernel's in any split of the prompt into runs."""
import dataclasses

import numpy as np
import pytest

from opentpu import isa as I
from opentpu.host.chat import Chat, sampler
from opentpu.isasim import Machine, board_config
from opentpu.llm import snapshot as SN
from opentpu.llm.mtp import MTPDecoder, MTPStats, mtp_engine
from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config

from test_mtp import CFG, _mtp_kv, _mtp_weights, _states
from test_qwen35 import _tiny_model

torch = pytest.importorskip("torch")


def _bits(x):
    return np.asarray(x, np.float32).view(np.uint32)


def _common(a, b) -> int:
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


@pytest.fixture(scope="module")
def smollm3(tmp_path_factory):
    """test_llama's tiny SmolLM3 (no q / k norms, layers without RoPE, the int8 embedding)."""
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import load_weights
    from test_llama import _smollm3
    torch.manual_seed(0)
    m = _smollm3().float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
    path = tmp_path_factory.mktemp("smollm3")
    m.save_pretrained(path)
    return load_weights(path), load_spec(path)


def _model(case, request=None):
    """(weights, spec, cfg) of a tiny model."""
    from test_autodecode import _tiny
    if case in ("qwen3", "lfm2"):
        W, spec = _tiny(case)
        return W, spec, device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=1)
    if case == "smollm3":
        W, spec = request.getfixturevalue("smollm3")
        return W, spec, device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=1)
    _, W, spec = _tiny_model(8, init=0.2)
    if case == "qwen35-grouped":        # group-major DeltaNet blocks (the 4B's)
        spec = dataclasses.replace(spec, pair_loop=True)
    if case == "qwen35-design":         # two slices
        return W, spec, device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=2)
    if case == "qwen35-st16":           # the DeltaNet states in 16 bits (Config.ST16)
        return W, spec, dataclasses.replace(CFG, ST16="e6m9")
    return W, spec, CFG


def _turns(make, turns, seed=0, check_state=None):
    """A chat's turns at the engine level: each (keep, new, at, gen) rolls the context back to
    its first `keep` tokens (Engine.rollback: to t <= keep), feeds the rest of the turn's
    rendering, the context's tokens t .. keep and `new` random ones, in two parts with a
    snapshot at position `at` between, then `gen` greedy decode steps (the last token not fed,
    as a reply's). After each prefill the logits, and after the steps the tokens, equal a fresh
    engine's (make()) fed the turn's rendering; check_state(a, b) compares their state.
    Returns the positions each turn rolled back to."""
    r = np.random.default_rng(seed)
    a, ctx, got = make(snapshots=1), [], []
    for keep, new, at, gen in turns:
        t = a.rollback(keep) if ctx else 0
        got.append(t)
        ids = ctx[:keep] + [int(x) for x in r.integers(0, 1000, new)]
        assert t < at < len(ids)
        for _, lg in a.prefill_chunks(ids[t:at], logits=False):
            assert lg is None
        assert a.pos == at and a.snapshot() == a.stateful
        for _, lg in a.prefill_chunks(ids[at:]):
            pass
        b = make()
        want = b.prefill(ids)
        assert a.pos == b.pos == len(ids)
        assert np.array_equal(_bits(lg), _bits(want)), (keep, t)
        if check_state is not None:
            check_state(a, b)
        out = [int(np.argmax(lg))]
        for _ in range(gen - 1):
            x = a.step(out[-1])
            assert np.array_equal(_bits(x), _bits(b.step(out[-1])))
            out.append(int(np.argmax(x)))
        ctx = ids + out[:-1]
        assert a.pos == len(ctx)
    return got


# Three turns: 30 tokens with a snapshot at 21; back to 21 (the snapshot's), 7 new with a
# snapshot at 26; back to 30, after the snapshot at 26: a model with state goes to 26.
TURNS = [(0, 30, 21, 6), (21, 7, 26, 5), (30, 12, 35, 4)]


@pytest.mark.parametrize("case, resident", [
    ("qwen3", False), ("qwen3", True), ("smollm3", True), ("lfm2", False), ("lfm2", True),
    ("qwen35", False), ("qwen35", True), ("qwen35-grouped", True), ("qwen35-design", False),
    ("qwen35-st16", False), ("qwen35-st16", True)])
def test_rollback_then_feed_is_a_fresh_prefill(case, resident, request):
    """A context rolled back (to the common prefix, or a stateful model's snapshot at or
    before it) and fed the rest gives a fresh engine's logits, next greedy tokens and state
    (Qwen3.5: every DeltaNet state and window, every slice) bit for bit, turn after turn;
    resident: the prompt runs (programs at run-time positions) and resident decode."""
    W, spec, cfg = _model(case, request)

    def make(snapshots=0):
        return Engine(spec, W, cap=512, cfg=cfg, resident=resident, prompt_runs=resident,
                      snapshots=snapshots)
    state = (lambda a, b: _same_states(a, b)) if case.startswith("qwen35") else None
    got = _turns(make, TURNS, check_state=state)
    stateful = case.startswith(("lfm2", "qwen35"))
    assert got == [0, 21, 26 if stateful else 30]


def _same_states(a, b):
    """Every slice's state regions (DeltaNet states and windows) equal."""
    regs = SN.merged(a.image.state_regions(0))
    for s in range(a.cfg.S):
        da, db = a.backend.machine.slices[s].dram, b.backend.machine.slices[s].dram
        assert all(np.array_equal(da[x:x + n], db[x:x + n]) for x, n in regs), s


def test_rollback_needs_the_snapshot():
    """The rollback's snapshot is what makes it exact: with the restore left out (the DeltaNet
    states and windows those of the old context) the logits differ; with no snapshot at or
    before the position (snapshots=0, or one after it) a stateful engine starts over (0) and
    an attention-only one goes back to the position itself."""
    W, spec, cfg = _model("qwen35")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SN.Snapshots, "restore", lambda self, eng, j, slot=0: {})
        with pytest.raises(AssertionError, match=r"\(21, 21\)"):
            _turns(lambda snapshots=0: Engine(spec, W, cap=512, cfg=cfg, snapshots=snapshots),
                   TURNS[:2])
    a = Engine(spec, W, cap=512, cfg=cfg)
    a.prefill(list(range(1, 30)))
    assert a.stateful and not a.snapshot() and a.rollback(20) == 0 and a.pos == 0
    a = Engine(spec, W, cap=512, cfg=cfg, snapshots=1)
    a.prefill(list(range(1, 25)))
    a.snapshot()
    a.prefill(list(range(1, 6)))
    assert a.rollback(20) == 0                  # the snapshot (24) is after 20: none before
    Wq, sq, cq = _model("qwen3")
    q = Engine(sq, Wq, cap=512, cfg=cq, snapshots=1)
    q.prefill(list(range(1, 30)))
    assert not q.stateful and not q.snapshot() and q.rollback(20) == 20 and q.rollback(25) == 20


def test_host_snapshots_where_the_area_does_not_fit(monkeypatch):
    """Where the device area does not fit the DRAM the snapshots go to the host (the backend's
    read and write): the same rollbacks, bit for bit; the device's image did not grow."""
    W, spec, cfg = _model("qwen35")
    dev = Engine(spec, W, cap=512, cfg=cfg, snapshots=1)
    assert dev._snaps.base is not None and dev.snapshot_bytes == dev._snaps.nbytes
    monkeypatch.setattr(SN.Snapshots, "area", staticmethod(lambda image, cfg, n: None))

    def make(snapshots=0):
        return Engine(spec, W, cap=512, cfg=cfg, snapshots=snapshots)
    a = make(1)
    assert a._snaps.base is None and a.snapshot_bytes == 0
    assert _turns(make, TURNS[:2], check_state=_same_states) == [0, 21]


def test_copy_programs_fit_imem_and_copy():
    """copy_programs: chunks of CHUNK words in two TMEM buffers, programs of whole LD / ST pairs
    that fit IMEM, every byte of every region copied (ISA simulator); merged joins adjacent
    regions and refuses overlapping or unaligned ones."""
    cfg = board_config(DRAM_BYTES=1 << 22, IMEM_WORDS=8 * 8)    # 7 instructions a program
    pairs = [(0, 1 << 21, 4 * 4 * SN.CHUNK + 8), (1 << 20, (1 << 21) + (1 << 20), 12)]
    progs = SN.copy_programs(pairs, cfg)
    assert [len(p[0]) for p in progs] == [7, 7] and all(p[0][-1].op == I.HALT for p in progs)
    ops = [i for p in progs for i in p[0][:-1]]
    assert [i.op for i in ops] == [I.LD, I.ST] * 6
    assert [i.w[1] for i in ops[::2]] == [0, SN.CHUNK] * 3
    assert [i.w[2] for i in ops[::2]] == [SN.CHUNK] * 4 + [2, 3]
    dram = np.random.default_rng(0).integers(0, 256, 1 << 22, np.uint8)
    m = Machine(cfg, [[]], [dram.copy()])
    for p in progs:
        m.load(p).run()
    d = m.slices[0].dram
    for src, dst, n in pairs:
        assert np.array_equal(d[dst:dst + n], dram[src:src + n])
    assert SN.merged([(0, 8), (8, 4), (16, 4)]) == [(0, 12), (16, 4)]
    assert SN.merged([(8, 4), (0, 8)]) == [(8, 4), (0, 8)]     # the order kept
    with pytest.raises(ValueError):
        SN.merged([(4, 8), (0, 8)])
    with pytest.raises(ValueError):
        SN.merged([(2, 8)])


# --------------------------------------------------------------------------------- Gemma 4
@pytest.fixture(scope="module")
def gemma(tmp_path_factory):
    """test_gemma4's tiny model with a 256-token window: a ring of 512 slots, slack 257."""
    pytest.importorskip("transformers.models.gemma4")
    from test_gemma4 import _model as g_model
    _, W, spec = g_model(tmp_path_factory.mktemp("gemma4"))
    return W, dataclasses.replace(spec, window=256)


def test_gemma4_rollback_inside_and_beyond_the_slack(gemma):
    """Gemma 4's sliding rings: 530 tokens (the ring of 512 wrapped), a snapshot at 250. Back
    by rollback_slack (257) the window is still in the ring: no restore, a fresh engine's
    logits; one further (258) the window's oldest slot holds a later position: the rollback
    goes to the snapshot (250), and without it (a slack one larger) the logits differ. Each
    case from the same context (the simulator's DRAM kept and put back)."""
    from test_gemma4 import _cfg
    W, spec = gemma

    def make(snapshots=0):
        return Engine(spec, W, cap=1024, cfg=_cfg(), resident=True, prompt_runs=True,
                      snapshots=snapshots)
    a = make(1)
    img = a.image
    assert img.ring == 512 and img.rollback_slack == 257 and a.stateful
    r = np.random.default_rng(3)
    ctx, new = ([int(x) for x in r.integers(0, 1000, n)] for n in (530, 9))
    for _ in a.prefill_chunks(ctx[:250], logits=False):
        pass
    assert a.snapshot() and a.prefill(ctx[250:]) is not None and a.pos == 530
    dram, at = a.backend.machine.slices[0].dram.copy(), dict(a._snaps.at)
    want = {}
    for keep, t, slack in [(273, 273, None), (272, 250, None), (272, 272, 258)]:
        a.backend.machine.slices[0].dram[:] = dram
        a.pos, a._snaps.at = 530, dict(at)
        with pytest.MonkeyPatch.context() as mp:
            if slack:
                mp.setattr(type(img), "rollback_slack", property(lambda self: slack))
            assert a.rollback(keep) == t
        ids = ctx[:keep] + new
        lg = a.prefill(ids[t:])
        if keep not in want:
            want[keep] = make().prefill(ids)
        assert np.array_equal(_bits(lg), _bits(want[keep])) == (slack is None), (keep, t)


# --------------------------------------------------------------------------------- MTP
@pytest.mark.parametrize("where", ["device", "host", "device-st16"])
def test_mtp_rollback_is_plain_greedy(where, monkeypatch):
    """MTPDecoder.rollback / snapshot (the committed slot; one token before the change, so that
    the MTP layer's row there takes the new token) with its prefill in two parts (follow: the
    token after the first, the MTP layer's input there) and the loop on the device, its drafts
    all right (each iteration commits the other slot: a turn's restore goes to the slot its
    snapshot did not come from): each turn's first token and draft, the MTP layer's KV cache
    and the committed states and windows equal a fresh MTP prefill's of the turn's rendering,
    and its tokens plain greedy decode's; the snapshots on the device and on the host;
    device-st16: the states in 16 bits (Config.ST16 e6m9)."""
    if where == "host":
        monkeypatch.setattr(SN.Snapshots, "area", staticmethod(lambda image, cfg, n: None))
    CFG = dataclasses.replace(globals()["CFG"], ST16="e6m9" if where == "device-st16" else "")
    where = where.partition("-")[0]
    _, W, spec = _tiny_model(8, init=0.2)
    W = _mtp_weights(W, spec)
    eng = mtp_engine(spec, W, cap=512, cfg=CFG, snapshots=1)
    assert (eng._snaps.base is None) == (where == "host")
    dec = MTPDecoder(eng)
    r = np.random.default_rng(2)
    ctx, snap, slots = [], None, []
    for keep, new, at, gen in [(0, 30, 20, 8), (21, 7, 26, 9), (28, 9, 33, 6)]:
        if ctx:
            assert dec.rollback(keep) == snap
            slots[-1] += (dec.slot,)
        t = snap if ctx else 0
        ids = ctx[:keep] + [int(x) for x in r.integers(0, 1000, new)]
        st = MTPStats()
        dec.prefill(ids[t:at], st, follow=ids[at])
        assert eng.pos == at and dec.snapshot()
        slots.append((dec.slot,))
        a0, d = dec.prefill(ids[at:], st)
        fresh = MTPDecoder(mtp_engine(spec, W, cap=512, cfg=CFG))
        assert fresh.prefill(ids, MTPStats()) == (a0, d)
        assert all(np.array_equal(x, y) for x, y in zip(_mtp_kv(eng, len(ids)),
                                                        _mtp_kv(fresh.eng, len(ids))))
        assert all(np.array_equal(x, y) for x, y in zip(_states(eng, spec, dec.slot),
                                                        _states(fresh.eng, spec, fresh.slot)))
        want = Engine(spec, W, cap=512, cfg=CFG, resident=True).generate(ids, max_new=gen)
        drafts = [0] * (eng.cap + 2)
        drafts[len(ids):len(ids) + gen] = want
        out = dec.loop_card(a0, d, gen, drafts=drafts).tokens
        assert out == want
        ctx, snap = ids + out[:-1], at
        assert eng.pos == len(ctx)
    assert any(a != b for a, b in slots[:-1])       # a restore into the other slot


# --------------------------------------------------------------------------------- Chat
class _QwenTok:
    """One token per character (its code point), and a template shaped as Qwen3's: each
    message between a header and an end mark; the generation prompt opens the assistant's
    turn and, thinking off, adds an empty thinking block there, which a past reply does not
    have (its thinking block, if any, is dropped): the reply's start is where a turn's
    rendering and the next one's part."""
    S, E, T0, T1 = "\x01", "\x02", "\x03", "\x04"

    def apply_chat_template(self, history, add_generation_prompt, enable_thinking, tokenize):
        out = ""
        for m in history:
            c = m["content"]
            if m["role"] == "assistant" and self.T1 in c:
                c = c.split(self.T1)[-1].lstrip("\n")
            out += f"{self.S}{m['role']}\n{c}{self.E}\n"
        if add_generation_prompt:
            out += f"{self.S}assistant\n" + ("" if enable_thinking else
                                              f"{self.T0}\n\n{self.T1}\n\n")
        return [ord(c) for c in out]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(map(chr, ids))


def _recording(rec):
    """A greedy chat.sampler whose picks from logits (each turn's first token) record them."""
    base = sampler(0, 0, 1.0, None)

    def pick(logits, context=()):
        rec.append(np.array(logits, np.float32))
        return base(logits, context)
    for k in ("stream", "warm", "greedy", "params", "softcap", "rng"):
        setattr(pick, k, getattr(base, k))
    return pick


def _greedy(eng, lg, n):
    """Plain greedy decode of n tokens from the logits `lg` of a prefill (an EOS ends it)."""
    out = []
    while len(out) < n:
        out.append(int(np.argmax(lg)))
        if out[-1] in eng.spec.eos:
            break
        lg = eng.step(out[-1])
    return out


MESSAGES = ["hello there", "and then?", "tell me more about it", "why", "ok, thanks!"]


@pytest.mark.parametrize("case, card, think", [
    ("qwen3", False, False), ("qwen3", True, False), ("lfm2", True, False),
    ("qwen35", False, False), ("qwen35", True, False), ("qwen35", True, True),
    ("qwen35-mtp", True, False)])
def test_chat_turns_roll_back_to_the_reply(case, card, think):
    """Five turns of a chat whose template drops the generation prompt's thinking block from
    past replies: from the second turn on the context rolls back to where the turn's rendering
    and what was fed part, the last reply's start (Qwen3: the KV cache's write position;
    Qwen3.5, LFM2: the snapshot each turn took there; MTP a token before) and feeds the reply
    and the new message, not the conversation (restarted False); each turn's logits (its first
    token's pick) and reply equal a fresh engine's on the turn's rendering, bit for bit. card:
    the decode loop on the device (resident); think: thinking on (no block to drop: a turn
    continues the context unless the reply held the template's end of a thinking block)."""
    mtp = case.endswith("-mtp")
    W, spec, cfg = _model(case.removesuffix("-mtp"))
    if mtp:
        W = _mtp_weights(W, spec)

    def make(snapshots=0, plain=False):
        if mtp and not plain:
            return mtp_engine(spec, W, cap=512, cfg=cfg, snapshots=snapshots)
        return Engine(spec, W, cap=512, cfg=cfg, resident=card, prompt_runs=card,
                      snapshots=snapshots)
    rec, tok = [], _QwenTok()
    chat = Chat(make(1), tok, think, _recording(rec), 6)
    assert chat.on_card == card and chat.use_mtp == mtp
    stateful, start, rollbacks = chat.eng.stateful, None, 0
    for i, msg in enumerate(MESSAGES):
        fed = list(chat.fed)
        reply, turn = chat.ask(msg)
        ids = tok.apply_chat_template(chat.history[:-1], True, think, True)
        k = _common(ids, fed)
        rollbacks += k < len(fed)
        assert not turn.restarted, i
        reused = turn.reused if i else 0
        assert reused == (k if k == len(fed) or not stateful else start - mtp), (i, k, len(fed))
        assert turn.prefill_tokens == len(ids) - reused
        ref = make(plain=True)
        want = ref.prefill(ids)
        assert np.array_equal(_bits(rec[-1]), _bits(want)), i
        assert chat._reply == [x for x in _greedy(ref, want, 6) if x not in spec.eos], i
        start = len(tok.apply_chat_template(chat.history[:-1], False, think, True)) + \
            len(f"{tok.S}assistant\n")
    assert len(rec) == len(MESSAGES) and (rollbacks == 4 if not think else rollbacks < 4)


@pytest.mark.parametrize("case", ["qwen3", "qwen35"])
def test_chat_thinking_on_drops_the_reply_thinking(case):
    """Thinking on: each reply holds a thinking block's end (scripted picks), which the next
    turn's rendering drops with the thinking before it: the context rolls back to the reply's
    start (Qwen3.5: the snapshot there) and feeds the answer and the new message; each turn's
    logits equal a fresh engine's on its rendering, bit for bit."""
    W, spec, cfg = _model(case)
    tok = _QwenTok()
    reply = [ord(c) for c in f"ab{tok.T1}\ncd"] + [min(spec.eos)]
    firsts, k = [], [0]

    def pick(logits, context=()):
        if k[0] == 0:
            firsts.append(np.array(logits, np.float32))
        t, k[0] = reply[k[0]], (k[0] + 1) % len(reply)
        return t
    chat = Chat(Engine(spec, W, cap=512, cfg=cfg, snapshots=1), tok, True, pick, 16)
    assert not chat.on_card
    start = None
    for i, msg in enumerate(MESSAGES[:4]):
        fed = list(chat.fed)
        _, turn = chat.ask(msg)
        assert turn.end == "eos" and chat._reply == reply[:-1]
        ids = tok.apply_chat_template(chat.history[:-1], True, True, True)
        assert not turn.restarted, i
        if i:
            assert _common(ids, fed) == start < len(fed) and turn.reused == start, i
        assert turn.prefill_tokens == len(ids) - (start if i else 0)
        want = Engine(spec, W, cap=512, cfg=cfg).prefill(ids)
        assert np.array_equal(_bits(firsts[-1]), _bits(want)), i
        start = len(ids)


def test_chat_stops_continues_and_resets():
    """A turn stopped inside its prefill (before the reply's start: no snapshot there) is
    continued by the next turn from where it stopped; /continue after max_new resumes the
    reply, and the turn after rolls back to that reply's start; /reset starts over. Every
    turn's logits a fresh engine's."""
    W, spec, cfg = _model("qwen35")
    rec, tok = [], _QwenTok()
    chat = Chat(Engine(spec, W, cap=512, cfg=cfg, snapshots=1), tok, False, _recording(rec), 4)

    def check(i):
        ids = tok.apply_chat_template(chat.history[:-1], True, False, True)
        want = Engine(spec, W, cap=512, cfg=cfg).prefill(ids)
        assert np.array_equal(_bits(rec[-1]), _bits(want)), i
    chat.ask("hello there")
    check(0)
    runs = [0]

    def stop():
        runs[0] += 1
        return runs[0] == 2
    reply, t = chat.ask("a much longer message, which takes several prefill runs", stop=stop)
    assert t.end == "stopped" and reply == "" and t.reused and t.prefill_tokens < 20
    fed = len(chat.fed)
    chat.ask("go on")
    check(1)
    assert chat.last.reused == fed and not chat.last.restarted
    reply, t = chat.resume()
    assert t.prefill_tokens == 0 and len(chat._reply) == 8
    chat.ask("and?")
    check(2)
    assert not chat.last.restarted and chat.last.reused < len(chat.fed) - 8
    chat.reset()
    assert chat.eng.pos == 0 and not chat.eng._snaps.at
    chat.ask("hello there")
    check(3)
    assert chat.last.reused == 0 and not chat.last.restarted


@pytest.mark.parametrize("snapshots", [0, 1])
def test_chat_restarts_without_a_snapshot(snapshots):
    """Qwen3.5 without snapshots (Engine(snapshots=0)): from the second turn on, the rendering
    parts from what was fed at the last reply's start, and no snapshot is there: the turn starts
    over (Turn.restarted, counted in Session.restarts) and feeds its whole rendering. With one,
    no turn does, and the session counts the positions kept. Every turn's logits a fresh
    engine's."""
    W, spec, cfg = _model("qwen35")
    rec, tok = [], _QwenTok()
    chat = Chat(Engine(spec, W, cap=512, cfg=cfg, snapshots=snapshots), tok, False,
                _recording(rec), 4)
    reused = 0
    for i, msg in enumerate(MESSAGES[:3]):
        _, turn = chat.ask(msg)
        ids = tok.apply_chat_template(chat.history[:-1], True, False, True)
        assert turn.restarted == (i > 0 and not snapshots), i
        assert turn.prefill_tokens == len(ids) - turn.reused
        assert (turn.reused == 0) == (i == 0 or not snapshots), i
        reused += turn.reused
        want = Engine(spec, W, cap=512, cfg=cfg).prefill(ids)
        assert np.array_equal(_bits(rec[-1]), _bits(want)), i
    assert chat.session.restarts == (0 if snapshots else 2) and chat.session.reused == reused


class _GemmaTok:
    """Gemma 4's instruct tokenizer and chat template in miniature. One token per character:
    ASCII at 200 + its code, any other id i as the character U+4E00 + i (every reply decodes
    and encodes back); the special tokens at Gemma's ids, which decode drops unless told to
    keep them, as the real one does; plain text without <bos> (the instruct tokenizer adds
    none). The template, as gemma-4-*-it's: <bos>, each message <|turn>role\\n...<turn|>\\n (a
    reply with its thinking channel, <|channel>...<channel|>, dropped and the rest trimmed),
    thinking on a system turn with <|think|> first; the generation prompt <|turn>model\\n, and
    with empty_thought (26B-A4B-it's) an empty thinking channel after it, thinking off."""
    BOS, THINK, SOC, EOC, SOT, EOT = 2, 98, 100, 101, 105, 106
    SPECIAL = {1, BOS, THINK, SOC, EOC, SOT, EOT}

    def __init__(self, empty_thought=False):
        self.empty_thought = empty_thought

    @staticmethod
    def ch(i):
        return chr(i - 200) if 200 <= i < 328 else chr(0x4E00 + i)

    def ids(self, text):
        return [ord(c) + 200 if ord(c) < 128 else ord(c) - 0x4E00 for c in text]

    def __call__(self, text):
        return {"input_ids": self.ids(text)}

    def apply_chat_template(self, history, add_generation_prompt, enable_thinking, tokenize):
        c = self.ch
        out = c(self.BOS)
        if enable_thinking:
            out += f"{c(self.SOT)}system\n{c(self.THINK)}\n{c(self.EOT)}\n"
        for m in history:
            role, text = "model" if m["role"] == "assistant" else m["role"], m["content"]
            if role == "model":         # strip_thinking
                text = "".join(p.split(c(self.SOC))[0] for p in text.split(c(self.EOC)))
            out += f"{c(self.SOT)}{role}\n{text.strip()}{c(self.EOT)}\n"
        if add_generation_prompt:
            out += f"{c(self.SOT)}model\n"
            if self.empty_thought and not enable_thinking:
                out += f"{c(self.SOC)}thought\n{c(self.EOC)}"
        return self.ids(out)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.ch(i) for i in ids if not (skip_special_tokens and i in self.SPECIAL))


@pytest.mark.parametrize("case, think, reply, answer", [
    ("e2b", False, "ab cd", "ab cd"), ("e2b", False, "ab cd\n\n\n", "ab cd"),
    ("e2b", True, "\x01thought\nhm\x02ab", "ab"), ("26b", False, "ab cd", "ab cd")])
def test_gemma4_chat_turns(gemma, monkeypatch, case, think, reply, answer):
    """A Gemma 4 instruct chat's turns (_GemmaTok's template: the E2B's, or the 26B-A4B's whose
    generation prompt opens the reply with an empty thinking channel), each reply scripted
    (\\x01, \\x02: the thinking channel's ends). The history renders each reply as the template
    gets it with its special tokens: its thinking channel dropped, the rest trimmed (with the
    channel's ends dropped from the reply's text, the thinking stayed in the context as text).
    A reply rendered as fed is continued (E2B); one rendered otherwise (whitespace trimmed, the
    thinking dropped, the 26B's empty channel) rolls the sliding rings back to the snapshot at
    the reply's start (rollback_slack patched to 2 for that: the long replies past the real
    slack are test_gemma4_rollback_inside_and_beyond_the_slack's), and feeds the reply and the
    new message. Every turn's logits a fresh engine's on its rendering, bit for bit."""
    from test_gemma4 import _cfg
    W, spec = gemma
    tok = _GemmaTok(empty_thought=case == "26b")
    script = tok.ids(reply.replace("\x01", tok.ch(tok.SOC)).replace("\x02", tok.ch(tok.EOC)))
    script += [tok.EOT]
    head = len(tok.ids(tok.ch(tok.SOT) + "model\n"))        # the reply's header
    firsts, k = [], [0]

    def pick(logits, context=()):
        if k[0] == 0:
            firsts.append(np.array(logits, np.float32))
        t, k[0] = script[k[0]], (k[0] + 1) % len(script)
        return t

    def make(snapshots=0):
        return Engine(spec, W, cap=1024, cfg=_cfg(), resident=True, prompt_runs=True,
                      snapshots=snapshots)
    chat = Chat(make(1), tok, think, pick, 16)
    assert chat.eng.stateful and not chat.on_card
    monkeypatch.setattr(type(chat.eng.image), "rollback_slack", property(lambda self: 2))
    want, start = [], None
    for i, msg in enumerate(["hello", "and?", "ok"]):
        fed, s0 = list(chat.fed), len(chat.eng.snap_stats)
        _, turn = chat.ask(msg)
        assert turn.end == "eos" and chat._reply == script[:-1] and not turn.restarted, i
        want.append({"role": "user", "content": msg})
        ids = tok.apply_chat_template(want, True, think, True)
        assert tok.apply_chat_template(chat.history[:-1], True, think, True) == ids, i
        restores = [s["pos"] for s in chat.eng.snap_stats[s0:] if s["op"] == "restore"]
        if i and reply == answer and case == "e2b":     # rendered as fed: continued
            assert turn.reused == len(fed) and not restores, i
        elif i:
            assert _common(ids, fed) < len(fed) and turn.reused == start == restores[0], i
        assert turn.prefill_tokens == len(ids) - turn.reused
        assert np.array_equal(_bits(firsts[-1]), _bits(make().prefill(ids))), i
        want.append({"role": "assistant", "content": answer})
        start = len(tok.apply_chat_template(want[:-1], False, think, True)) + head


def test_chatkv_tool_on_gemma4(tmp_path, capsys):
    """tools/qual/chatkv.py, the card check, dry run on the ISA simulator with a Gemma 4
    checkpoint (test_gemma4.checkpoint: its template renders a reply as fed): the recording
    sampler takes the model's soft cap (Chat refused the tool's before: "the sampler's softcap
    None is not the model's 30.0"), the turns continue the context, and each restart's logits
    and reply equal the turn's: CHATKV PASS."""
    import importlib.util
    from pathlib import Path

    from test_gemma4 import checkpoint
    d = checkpoint(tmp_path / "gemma4")
    s = importlib.util.spec_from_file_location(
        "chatkv", Path(__file__).resolve().parent.parent / "tools" / "qual" / "chatkv.py")
    CK = importlib.util.module_from_spec(s)
    s.loader.exec_module(CK)
    assert CK.main(["--model", str(d), "--backend", "isa", "--turns", "2", "--max-new", "3"]) == 0
    out = capsys.readouterr().out
    assert "stateful True" in out and "reused 0" in out and "CHATKV PASS" in out


@pytest.mark.parametrize("name", ["gemma4", "gemma4-26b"])
def test_gemma4_instruct_templates(name):
    """The instruct checkpoints' own tokenizer and template (models/gemma-4-E2B-it,
    gemma-4-26B-A4B-it: chat_template.jinja), with Chat's rendering, three turns: plain text
    gets no <bos> (gemma4.text_ids adds it); a reply as the model writes it (its tokens fed,
    its text kept with its special tokens) is continued, thinking on or off, except behind the
    26B's generation prompt, thinking off, which opens the reply with an empty thinking channel
    (4 tokens: the reply's start, reply_start, is before it): back to the reply's start; a
    reply's own thinking channel is dropped: back to the reply's start too."""
    import types

    from opentpu.llm import gemma4 as G4
    from opentpu.llm import model_dir
    path = model_dir(name)
    if not (path / "chat_template.jinja").exists():
        pytest.skip(f"no models/{path.name}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    assert tok("Paris")["input_ids"][:1] != [2] and G4.text_ids(tok, "Paris")[:1] == [2]
    eng = types.SimpleNamespace(spec=G4.Spec.from_hf(path), stateful=True)
    for think in (False, True):
        for reply in ("Paris is the capital of France.",
                      "<|channel>thought\nThe user asks for a capital.<channel|>Paris."):
            pick = sampler(0, 0, 1.0, None, softcap=eng.spec.softcap)
            chat, empty = Chat(eng, tok, think, pick, 16), 0
            for i, msg in enumerate(["What is the capital of France?", "And of Italy?", "Ok"]):
                chat.history.append({"role": "user", "content": msg})
                ids = chat._template()
                k, at = _common(ids, chat.fed), chat.reply_start(ids)
                if i == 0:
                    empty = len(ids) - at
                    assert empty == (4 if name == "gemma4-26b" and not think else 0)
                elif reply.startswith("<|channel>") or empty:
                    assert k == start < len(chat.fed), (think, reply, i)
                else:
                    assert k == len(chat.fed), (think, reply, i)
                start = at
                out = list(tok(reply)["input_ids"])
                chat.fed = ids + out            # (the reply's last token is fed by the next turn)
                chat.history.append({"role": "assistant",
                                     "content": tok.decode(out, skip_special_tokens=False)})


# --------------------------------------------------------------------------------- RTL
@pytest.mark.parametrize("case", ["qwen35", "lfm2"])
def test_rollback_on_board_model(have_verilator, case):
    """The board model through the host driver (the Verilator RTL behind SimTransport): a chat's
    turns at the engine level, the prefill in two parts with the snapshot's copy program between
    (LD / ST through TMEM into the area after the image), decode steps, a rollback whose restore
    program copies it back, the next prefill: every logits row the ISA simulator's bit for bit,
    and then the state regions and the snapshot area its bytes."""
    from opentpu.host.board import BoardBackend, SimTransport
    W, spec, _ = _model(case)
    cfg = board_config(DRAM_BYTES=1 << 25)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5)
    isa = Engine(spec, W, cap=256, cfg=cfg, snapshots=1)
    brd = Engine(spec, W, cap=256, cfg=cfg, snapshots=1,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    assert brd._snaps.base is not None and brd.backend.prog_at >= brd._snaps.end()
    r = np.random.default_rng(7)
    p, new = ([int(x) for x in r.integers(0, 1000, n)] for n in (14, 4))
    out = {}
    for e in (isa, brd):
        for _, lg in e.prefill_chunks(p[:9], logits=False):
            pass
        assert e.snapshot()
        for _, lg in e.prefill_chunks(p[9:]):
            pass
        got = [lg] + [e.step(t) for t in (5, 6)]
        assert e.rollback(11) == 9
        out[e] = got + [e.prefill(p[9:11] + new), e.step(7)]
    assert [s["op"] for s in brd.snap_stats] == ["snapshot", "restore"]
    assert all(s["cycles"] > 0 for s in brd.snap_stats)
    for x, y in zip(out[isa], out[brd]):
        assert np.array_equal(_bits(x), _bits(y))
    sn = brd._snaps
    regs = SN.merged(brd.image.state_regions(0)) + [(sn.base, sn.n * sn.nbytes)]
    d = isa.backend.machine.slices[0].dram
    for a, n in regs:
        assert np.array_equal(brd.backend.read(0, a, n), d[a:a + n]), hex(a)
