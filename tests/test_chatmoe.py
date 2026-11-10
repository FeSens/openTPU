"""otpu-chat with a mixture-of-experts model bigger than the card's DRAM (opentpu/host/streamed.py,
docs/offload.md 14): the experts stream from a pool file into slots while every run is in
flight, the prompt goes layer by layer where the model has layer runs (Qwen3.5-MoE, Gemma 4
MoE) and token by token else (LFM2-MoE). Tiny random MoEs of the three families on the ISA
simulator with 2 slots a layer: a chat turn gives tools/offload/moe_card.py's tokens, a later
turn continues the context bit for bit, the pool is checked before the image is built."""
import json
import pickle
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opentpu.host import streamed as ST
from opentpu.host.chat import Chat, Session, Turn, sampler
from opentpu.host.offload import warm_report
from opentpu.isasim import board_config

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
CFG = board_config(DRAM_BYTES=1 << 26)      # 64 MiB: the tiny images and 2 slots a layer
K = 2                                       # slots a layer (the tiny models' top k)
N = 6                                       # tokens a reply


class Tok:
    """A tokenizer whose text is its ids ("12 7 400": a reply's text gives its ids back), its
    chat template the messages' ids one after another."""

    def apply_chat_template(self, history, add_generation_prompt=True, enable_thinking=False,
                            tokenize=True, **kw):
        return [int(x) for m in history for x in m["content"].split()]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(f" {int(i)}" for i in ids)


def _save(d: Path, config: dict, W: dict) -> Path:
    """A checkpoint directory with a chat template (otpu-chat refuses one without; Tok renders
    the turns)."""
    from safetensors.numpy import save_file
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config))
    (d / "chat_template.jinja").write_text("{% for m in messages %}{{ m.content }} {% endfor %}")
    save_file({k: np.ascontiguousarray(v) for k, v in W.items()}, str(d / "model.safetensors"))
    return d


def _checkpoint(fam: str, d: Path) -> Path:
    """The family's tiny MoE (its tests' fixtures) as a checkpoint directory: Qwen3.5-MoE with
    an LM head of its own (the 35B's: its int8 embedding table on the host), LFM2-MoE (tied, its
    experts by name), Gemma 4 MoE (8 experts 96 wide beside the MLP)."""
    if fam == "qwen35":
        from test_qwen35_moe import _tiny
        m, W, _ = _tiny()
        c = m.config.to_dict()
        c["tie_word_embeddings"] = False
        W = dict(W, **{"lm_head.weight": W["model.embed_tokens.weight"] * np.float32(1.25)})
    elif fam == "lfm2":
        from test_lfm2_moe import _tiny
        m, W, _ = _tiny()
        c = m.config.to_dict()
    else:
        pytest.importorskip("transformers.models.gemma4")
        from test_gemma4_moe import _model

        class Factory:
            @staticmethod
            def mktemp(name):
                (d.parent / f"{name}-spec").mkdir(parents=True, exist_ok=True)
                return d.parent / f"{name}-spec"
        m, W, _ = _model(Factory, "moe", enable_moe_block=True, num_experts=8,
                         top_k_experts=2, moe_intermediate_size=96)
        c = m.config.to_dict()
    return _save(d, c, W)


@pytest.fixture(scope="module", params=["qwen35", "lfm2", "gemma4"])
def ckpt(request, tmp_path_factory):
    fam = request.param
    return fam, _checkpoint(fam, tmp_path_factory.mktemp(fam) / "model")


@pytest.fixture
def formats(ckpt, monkeypatch):
    """The family's weight formats on the card (streamed.FORMATS), as moe_card takes them: the
    formats string through OTPU_FORMATS (Qwen3.5's nvfp4 layers beside fp4 experts, Gemma 4's
    nvfp4 experts beside int8 layers); LFM2 nvfp4 throughout."""
    fam, _ = ckpt
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    wf, hf, f = ST.FORMATS[fam]
    if f:
        monkeypatch.setenv("OTPU_FORMATS", f)
    return wf, hf


def _moe_card():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "moe_card", ROOT / "tools" / "offload" / "moe_card.py")
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    return mc


def _greedy_chat(eng) -> Chat:
    return Chat(eng, Tok(), False,
                sampler(0, 0, 1.0, None, softcap=getattr(eng.spec, "softcap", None)), N)


def _until_eos(tokens, eos):
    return tokens[:next((i for i, t in enumerate(tokens) if t in eos), len(tokens))]


def test_streamed_defaults_are_the_cards_runs():
    """--wformat auto for a streamed MoE: each family's formats on the card (docs/offload.md
    14.2: the MX formats' per-kind choices; MiniCPM-V's its own, docs/minicpmv.md 5.1); the
    pool's default place is the checkpoint's directory, named by the experts' format."""
    from opentpu.llm import gemma4, lfm2, minicpmv, qwen35
    assert ST.defaults(object.__new__(qwen35.Spec)) == \
        ("fp4", "int8", "attn=nvfp4,delta=nvfp4,mlp=nvfp4")
    assert ST.defaults(object.__new__(lfm2.Spec)) == ("nvfp4", "int8", None)
    assert ST.defaults(object.__new__(gemma4.Spec)) == ("int8", "fp4", "experts=nvfp4")
    assert ST.defaults(object.__new__(minicpmv.Spec)) == \
        ("fp4", "int8", "delta@0-19=int8,attn=nvfp4,delta=nvfp4,mlp=int8")
    assert ST.em_rows_of(object.__new__(minicpmv.Spec)) == ST.EM_ROWS["qwen35"]
    assert ST.default_pool("models/Qwen3.5-35B-A3B", "fp4") == \
        Path("models/Qwen3.5-35B-A3B/pool-mx1-fp4.bin")      # (beside the MXU before MX's
    assert ST.default_pool("models/LFM2.5-8B-A1B", "nvfp4") == \
        Path("models/LFM2.5-8B-A1B/pool-mx1-nvfp4.bin")      # pool-fp4.bin, key 1)


def test_expert_major_is_the_layer_major_prompts_default(ckpt, formats):
    """streamed.engine's prompt (docs/offload.md 13.15): expert-major wherever it runs layer by
    layer with pooled slots, no layer ahead and room for its needs (Qwen3.5 and Gemma 4 with the
    DRAM filled: 6 layers of 8 slots less a 20-slot scratch, 28 >= 3 x 8), its needs in
    whole-expert idle parts (NEED_PART), for the calls of the family's EM_ROWS new rows or more
    (13.16: em_rows; expert_major=True or em_rows=0: every call); not with 2 slots a layer
    (12 - 20 < 24: the guarded layers' experts could hold every slot), each layer's own slots,
    the layer ahead, expert_major=False or a prompt token by token (LFM2's, or layer_major=0),
    whose idle parts stay Engine's."""
    from opentpu.llm.qwen3 import AHEAD_PART
    fam, d = ckpt
    wf, hf = formats

    def em(experts=0, **kw):
        e = ST.engine(d, CFG, "isa", 256, None, experts=experts, wformat=wf, head_format=hf,
                      **kw)
        return e.expert_major, e.ahead_part, getattr(e, "em_rows", 0)
    off = (False, AHEAD_PART, 0)
    assert em() == (off if fam == "lfm2" else (True, ST.NEED_PART, ST.EM_ROWS[fam]))
    assert em(K) == em(expert_major=False) == em(K, em_rows=3) == off
    if fam != "lfm2":
        assert em(layer_major=0) == em(pooled=False) == em(layer_ahead=True) == off
        assert em(expert_major=True) == em(em_rows=0) == (True, ST.NEED_PART, 0)
        assert em(em_rows=9) == em(expert_major=True, em_rows=9) == (True, ST.NEED_PART, 9)


def test_a_streamed_engine_above_4gib_holds_every_expert(tmp_path, monkeypatch):
    """streamed.engine at AW 36 (bcu1525-wg) with the DRAM room for every expert and the
    weights from 4 GiB on (OTPU_HIGH=1): the image's resident pool, no pool file, no server,
    no expert-major runs; its layer-major prompt and decode steps give the low image's
    (OTPU_HIGH=0: the server's slots, every expert) logits bit for bit."""
    from opentpu import target as T
    d = _checkpoint("qwen35", tmp_path / "model")
    cfg = T.config("bcu1525-wg", DRAM_BYTES=1 << 33)
    ids = [int(t) for t in np.random.default_rng(6).integers(10, 1000, 7)]
    got = {}
    for high in ("1", "0"):
        monkeypatch.setenv("OTPU_HIGH", high)
        eng = ST.engine(d, cfg, "isa", 256, None, wformat="fp4", head_format="int8")
        if high == "1":
            assert eng.image.pool is not None and eng.server is None and not eng.expert_major
            assert eng.layer_major and eng.pool_path is None
            assert ST.describe(eng).startswith("every one of 48 experts resident from 4 GiB")
        else:
            assert eng.image.pool is None and eng.server is not None
        got[high] = np.vstack([eng.prefill(ids)[None]] + [eng.step(t)[None] for t in (5, 6)])
    assert np.array_equal(got["1"].view(np.uint32), got["0"].view(np.uint32))


@pytest.mark.parametrize("fam", ["qwen35", "gemma4"])
def test_expert_major_rows_by_family(fam):
    """EM_ROWS (docs/offload.md 13.16): the least new rows a call of the family's prompt runs
    expert-major by default; a family without layer runs has none (0: every call)."""
    from opentpu.llm import gemma4, lfm2, qwen35
    spec = object.__new__({"qwen35": qwen35.Spec, "gemma4": gemma4.Spec}[fam])
    assert ST.em_rows_of(spec) == ST.EM_ROWS[fam] > 0
    assert ST.em_rows_of(object.__new__(lfm2.Spec)) == 0


def test_expert_major_from_its_rows_a_call(ckpt, formats):
    """The expert-major prompt only in the prefill_layers calls of at least em_rows new rows
    (streamed.gate_expert_major, docs/offload.md 13.16), layer-major below: with em_rows 6,
    calls of 6, 5 and 6 rows run expert-major (need lines), layer-major (requests, no need
    line) and expert-major again, each mode with its own compiled runs (their keys are the
    same), and each call's logits are those of an engine with every expert resident fed the
    same tokens token by token, bit for bit. The engine stays expert-major between calls;
    em_last and em_calls tell what ran."""
    fam, d = ckpt
    wf, hf = formats
    if fam == "lfm2":
        pytest.skip("LFM2's prompt runs token by token")
    eng = ST.engine(d, CFG, "isa", 256, None, wformat=wf, head_format=hf, em_rows=6)
    ref = ST.engine(d, CFG, "isa", 256, None, wformat=wf, head_format=hf, layer_major=0)
    assert eng.expert_major and eng.em_rows == 6 and eng.em_last is None
    srv, rng = eng.server, np.random.default_rng(6)
    for n, em in ((6, True), (5, False), (6, True)):
        ids = [int(t) for t in rng.integers(10, 1000, n)]
        lines, seq = srv.need_lines, srv.seq
        lg = eng.prefill(ids)
        assert np.array_equal(lg.view(np.uint32), ref.prefill(ids).view(np.uint32)), (n, em)
        assert eng.em_last == (n, em) and eng.expert_major
        assert (srv.need_lines > lines) == em and srv.seq > seq
    assert eng.em_calls == [1, 2] and ref.server.misses == 0
    assert any(k[0] == "x" for k in eng._layer_runs if isinstance(k, tuple))


def test_expert_major_fits_room_for_three_layers_beside_the_scratch():
    """expert_major_fits: the pooled slots less the largest chunk's scratch (prefill rows, a
    record each, in slot pitches) hold the three guarded layers' experts, 3 E, or not."""
    from types import SimpleNamespace as NS

    from opentpu.llm.moe import em_record
    spec = NS(hidden=64, moe=NS(k=2))
    pitch = 1000
    sc = -(-16 * em_record(64, 2) // pitch)             # (16 rows' records: the scratch)

    def img(layers):
        return NS(prefill_rows=16, offload=NS(layers=layers, E=8, pitch=pitch))
    assert ST.expert_major_fits(spec, img(1), 24 + sc)          # (just: 3 E beside it)
    assert not ST.expert_major_fits(spec, img(1), 23 + sc)
    assert ST.expert_major_fits(spec, img(6), 8) and not ST.expert_major_fits(spec, img(6), 2)


def test_a_chat_turn_is_moe_cards_run(ckpt, formats, tmp_path):
    """A greedy chat turn on the streamed engine (the chat's template, its prefill, the first
    token picked from the prefill's logits, then the card's decode loop) gives
    tools/offload/moe_card.py's tokens on the same prompt (its prompt but the last token, then
    the loop from it), up to an end of turn, with 2 slots a layer and the same pool file: Qwen3.5
    and Gemma 4 layer by layer (R = 2), LFM2 token by token. The pool moe_card left half packed
    beside a whole checkpoint serves the chat, which packs the rest. The turn counts the
    prompt's and the decode's misses."""
    fam, d = ckpt
    wf, hf = formats
    lm = 0 if fam == "lfm2" else 2
    ids = [int(t) for t in np.random.default_rng(1).integers(10, 1000, 9)]
    (tmp_path / "cfg.pkl").write_bytes(pickle.dumps(CFG))
    pool = tmp_path / "pool.bin"
    ref = _moe_card().card(str(d), {"ids": ids, "tokens": []}, N, K, 256, str(pool),
                           cfg_file=str(tmp_path / "cfg.pkl"), layer_major=lm, wformat=wf,
                           head_format=hf)
    assert ref["misses"] > 0 and ref["layer_major"] == lm
    packed = np.fromfile(str(pool) + ".packed", np.uint8)
    assert 0 < packed.sum() <= len(packed)
    eng = ST.engine(d, CFG, "isa", 256, pool, experts=K, wformat=wf, head_format=hf)
    assert eng.layer_major == lm and eng.image.offload.slots[0][1] == K
    pw = eng.server.pool_warm               # the open's read, or moe.rewarm's after the build
    assert pw is eng.server.pool_file.warm_t
    pw.join(60)
    r = warm_report(pw)
    assert r["end"] and r["error"] is None and pw.bytes == pw.total
    assert r["resident"] in (None, 1.0)
    chat = _greedy_chat(eng)
    reply, turn = chat.ask(" ".join(map(str, ids)))
    want = _until_eos(ref["tokens"], eng.spec.eos)
    assert chat._reply == want and len(want) > 1
    assert reply == Tok().decode(want)
    assert turn.experts and turn.prefill_misses > 0 and turn.expert_hits + turn.expert_misses
    assert turn.streamed == turn.expert_misses * eng.image.offload.slot_bytes
    assert "experts" in turn.line() and "misses in the prompt" in turn.line()


@pytest.mark.parametrize("slots", [K, 0])
def test_the_next_turn_continues_the_context(ckpt, formats, slots, monkeypatch):
    """A second turn feeds only what the conversation added (the last reply's last token and
    the new message) on the streamed engine: layer by layer from its position for Qwen3.5 and
    Gemma 4 (prefill_layers; a turn's part up to a snapshot inside it without the head), token
    by token for LFM2. Its reply is that of an engine with every expert resident fed the whole
    conversation at once, bit for bit. With the DRAM filled (slots 0) the prompts are
    expert-major where they have 7 rows or more (em_rows 7, 13.16: the first turn's 7, not
    the second's 6), with 2 slots a layer never (no room, 13.15)."""
    fam, d = ckpt
    wf, hf = formats
    if fam == "lfm2" and not slots:
        pytest.skip("LFM2's prompt runs token by token")
    eng = ST.engine(d, CFG, "isa", 256, None, experts=slots, wformat=wf, head_format=hf,
                    em_rows=7)
    assert eng.expert_major == (not slots)
    calls = []
    layers = eng.prefill_layers
    monkeypatch.setattr(eng, "prefill_layers", lambda t, logits=True: calls.append(
        (len(t), logits)) or layers(t, logits))
    chat = _greedy_chat(eng)
    rng = np.random.default_rng(2)
    u1, u2 = (" ".join(str(t) for t in rng.integers(10, 1000, n)) for n in (7, 5))
    chat.ask(u1)
    fed, c1 = len(chat.fed), len(calls)
    _, t2 = chat.ask(u2)
    ids2 = Tok().apply_chat_template(chat.history[:3])
    assert chat.fed[:len(ids2)] == ids2 and not t2.restarted
    assert t2.prefill_tokens == len(ids2) - fed and t2.context == eng.pos == len(chat.fed)
    for part, n in ((calls[:c1], 7), (calls[c1:], len(ids2) - fed)):    # (each turn's parts:
        assert fam == "lfm2" and not part or sum(k for k, _ in part) == n   # the head in the last)
        assert [lg for _, lg in part] == [False] * (len(part) - 1) + [True] * bool(part)
    assert (len(calls) == 0) == (fam == "lfm2")
    ref = ST.engine(d, CFG, "isa", 256, None, wformat=wf, head_format=hf, layer_major=0)
    assert ref.image.offload.slots[0][1] == ref.image.offload.E      # (every expert resident)
    t0 = int(np.argmax(ref.prefill(ids2)))
    want = _until_eos([t0] + ref.generate_card(t0, N - 1, stop_ids=[]), eng.spec.eos)
    assert chat._reply == want
    assert (eng.server.misses > 0 or not slots) and ref.server.misses == 0
    assert slots or eng.server.need_lines > 0 and eng.em_calls == [1, 1]


def _recording(rec, softcap=None):
    """A greedy chat.sampler (with the model's soft cap) whose picks from logits (each turn's
    first token) record them (test_chatkv's, for any model)."""
    base = sampler(0, 0, 1.0, None, softcap=softcap)

    def pick(logits, context=()):
        rec.append(np.array(logits, np.float32))
        return base(logits, context)
    for k in ("stream", "warm", "greedy", "params", "softcap", "rng"):
        setattr(pick, k, getattr(base, k))
    return pick


@pytest.mark.parametrize("slots", [K, 0])
@pytest.mark.parametrize("where", ["host", "device"])
def test_a_streamed_chat_rolls_back_to_the_reply(ckpt, formats, where, slots, monkeypatch):
    """otpu-chat's turns keep the context on a streamed MoE too (snapshots=1 through
    streamed.engine). With a template shaped as Qwen3's (test_chatkv's: a past reply renders
    without the generation prompt's thinking block) each later turn rolls back to the last
    reply's start (Qwen3.5's DeltaNet states, LFM2's convolution rings and Gemma 4's sliding
    rings from the snapshot the turn before took there) and feeds the reply and the new
    message: reused > 0, not restarted. Each turn's logits (its first token's pick) equal a
    fresh streamed engine's prefill of the turn's rendering, bit for bit, and its reply that
    engine's greedy tokens: Qwen3.5 and Gemma 4 layer by layer (the part up to the snapshot
    without the head), LFM2 token by token. The snapshots on the host (where: the slots fill
    the card's DRAM, so the device area does not fit) or on the device. The tiny Gemma 4's
    sliding windows hold its whole context (no state to keep: its KV caches go back). With the
    DRAM filled (slots 0) the prompts are expert-major from 7 rows a call (em_rows 7, 13.16):
    each Qwen3.5 turn's part up to the snapshot, not the 6 rows after it (the generation
    prompt's thinking block), each Gemma 4 turn (one part)."""
    from opentpu.llm import snapshot as SN
    from test_chatkv import MESSAGES, _bits, _greedy, _QwenTok
    fam, d = ckpt
    wf, hf = formats
    if fam == "gemma4" and where == "device":
        pytest.skip("the tiny Gemma 4 keeps no state (where: no snapshots)")
    if fam == "lfm2" and not slots:
        pytest.skip("LFM2's prompt runs token by token")
    if where == "host":
        monkeypatch.setattr(SN.Snapshots, "area", staticmethod(lambda image, cfg, n: None))

    def make(snapshots=0):
        return ST.engine(d, CFG, "isa", 256, None, experts=slots, wformat=wf, head_format=hf,
                         snapshots=snapshots, em_rows=7)
    eng = make(1)
    assert eng.expert_major == (not slots)
    rec, tok = [], _QwenTok()
    chat = Chat(eng, tok, False, _recording(rec, getattr(eng.spec, "softcap", None)), N)
    stateful = fam != "gemma4"
    assert chat.on_card and eng.stateful == stateful
    assert (eng._snaps.base is None) == (where == "host") if stateful else eng._snaps is None
    for i, msg in enumerate(MESSAGES[:3]):
        _, turn = chat.ask(msg)
        ids = tok.apply_chat_template(chat.history[:-1], True, False, True)
        assert not turn.restarted and (turn.reused > 0) == (i > 0), (i, turn.reused)
        assert turn.prefill_tokens == len(ids) - turn.reused and turn.context == len(chat.fed)
        ref = make()
        want = ref.prefill(ids)
        assert np.array_equal(_bits(rec[-1]), _bits(want)), i
        assert chat._reply == [x for x in _greedy(ref, want, N) if x not in eng.spec.eos], i
    assert eng.layer_major == (0 if fam == "lfm2" else 2)
    assert eng.server.misses > 0 or not slots
    assert slots or eng.em_calls == ([3, 3] if stateful else [0, 3])
    ops = [s["op"] for s in eng.snap_stats]             # (a snapshot each turn, restored by
    assert ops == (["snapshot", "restore"] * 3)[:5] if stateful else not ops    # the next)


@pytest.mark.parametrize("top_k", [80, 20])
def test_a_sampled_reply_streams_its_experts(ckpt, formats, top_k):
    """A sampled reply: with a top-k the card's loop does not take (--top-k 80) the host picks
    each token from a step's logits, else (20) the card's loop samples. The server serves those steps as it does the loop. The reply is that of the same
    sampler and seed on an engine with every expert resident, and the turn counts the steps'
    hits, misses and bytes."""
    fam, d = ckpt
    wf, hf = formats

    def chat(eng):
        pick = sampler(0.8, top_k, 1.0, 7, softcap=getattr(eng.spec, "softcap", None))
        return Chat(eng, Tok(), False, pick, N)
    eng = ST.engine(d, CFG, "isa", 256, None, experts=K, wformat=wf, head_format=hf)
    ref = ST.engine(d, CFG, "isa", 256, None, wformat=wf, head_format=hf)
    a, b = chat(eng), chat(ref)
    assert a.on_card == (top_k <= 64)
    u = " ".join(str(t) for t in np.random.default_rng(5).integers(10, 1000, 7))
    _, t = a.ask(u)
    b.ask(u)
    assert a._reply == b._reply and len(a._reply) > 1
    assert t.expert_misses > 0 and t.expert_hits + t.expert_misses >= (len(a._reply) - 1) * K
    assert t.streamed == t.expert_misses * eng.image.offload.slot_bytes
    assert ref.expert_major != (fam == "lfm2")      # (filled: room for expert-major, 13.15)
    assert ref.server.misses == 0 or ref.expert_major    # (its scratch's slots come back empty)


def _pack_pool(*args, monkeypatch):
    argv = sys.argv
    monkeypatch.setattr(sys, "argv", ["pack_pool.py", *map(str, args)])
    try:
        runpy.run_path(str(ROOT / "tools" / "offload" / "pack_pool.py"), run_name="__main__")
    finally:
        monkeypatch.setattr(sys, "argv", argv)


def test_the_pool_is_checked_before_the_image(tmp_path, monkeypatch):
    """streamed.engine refuses a pool that cannot serve the image before it builds it, each
    with what to do: none at the path (pack_pool's init and pack), one of another size
    (another model or format: open_pool would have truncated it), another image's key, or one
    not whole beside a checkpoint without its experts (strip_experts.py's). Whole, it serves
    that checkpoint with the full one's tokens; its default place is the checkpoint's
    directory, and a link to it there is followed to its .packed, .format and .key."""
    from opentpu.llm.qwen3 import LazyWeights
    d = _checkpoint("qwen35", tmp_path / "model")
    monkeypatch.delenv("OTPU_FORMATS", raising=False)

    def engine(pool, W=None):
        return ST.engine(d, CFG, "isa", 256, pool, experts=K, W=W)
    with pytest.raises(ST.PoolError, match=r"no expert pool at .*pool-mx1-fp4.bin: pack it .*"
                                           r"pack_pool.py .* init .* pack I N"):
        engine(ST.DEFAULT)
    bad = tmp_path / "bad.bin"
    bad.write_bytes(bytes(1000))
    with pytest.raises(ST.PoolError, match="1000 bytes, not this image's pool"):
        engine(bad)
    assert bad.stat().st_size == 1000
    pool = d / "pool-mx1-fp4.bin"
    _pack_pool(d, pool, "init", monkeypatch=monkeypatch)
    kf = Path(str(pool) + ".key")
    key = kf.read_text()
    kf.write_text(json.dumps(dict(json.loads(key), quant="0" * 32)))
    with pytest.raises(ST.PoolError, match="quant differ.*Repack it"):
        engine(ST.DEFAULT)
    kf.write_text(key)
    W = LazyWeights(d)
    stripped = {k: W[k] for k in W.keys() if ".experts." not in k}
    with pytest.raises(ST.PoolError, match=r"holds 0 of 48 experts and .* has none"):
        engine(ST.DEFAULT, stripped)
    _pack_pool(d, pool, "pack", 0, 1, monkeypatch=monkeypatch)
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 5)]
    a, b = engine(ST.DEFAULT, stripped), engine(ST.DEFAULT)
    assert a.pool_path == b.pool_path == pool
    for t in toks:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32))
    link = tmp_path / "link.bin"
    link.symlink_to(pool)
    assert engine(link).pool_path == pool.resolve()


def test_pack_pool_sums_tell_a_current_pool(tmp_path, monkeypatch, capsys):
    """pack_pool.py sums: the sha256 of a pool's records and of the experts this tree packs
    from the checkpoint now are the same lists for a current pool (N ids spread over it, or a
    file of ids), and differ where a record is not this tree's."""
    d = _checkpoint("qwen35", tmp_path / "model")
    pool = tmp_path / "pool.bin"
    _pack_pool(d, pool, "init", monkeypatch=monkeypatch)
    _pack_pool(d, pool, "pack", 0, 1, monkeypatch=monkeypatch)
    capsys.readouterr()

    def sums(p, ids):
        _pack_pool(d, p, "sums", ids, monkeypatch=monkeypatch)
        return capsys.readouterr().out.split("\n")[:-1]
    want = sums("-", 5)
    assert [ln.split()[0] for ln in want] == [str(i * 48 // 5) for i in range(5)]
    assert sums(pool, 5) == want
    (tmp_path / "ids").write_text("3\n40\n")
    assert sums(pool, tmp_path / "ids") == sums("-", tmp_path / "ids")
    with open(pool, "r+b") as f:                        # one record of another tree
        f.seek(9 * (pool.stat().st_size // 48) + 100)
        f.write(b"\xff")
    got = sums(pool, 5)
    assert [a == b for a, b in zip(got, want)] == [True, False, True, True, True]


def test_otpu_chat_runs_a_streamed_moe(tmp_path, monkeypatch, capfd):
    """otpu-chat --model <a MoE checkpoint> --prompt: the experts from the checkpoint's pool
    (packed by pack_pool.py), the slots and the pool named at load, the turn's line with the
    slots' hits, MB streamed a token and the prompt's misses; without a pool it stops with how
    to pack one."""
    from opentpu.host import chat as C
    d = _checkpoint("qwen35", tmp_path / "model")
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda p: Tok())
    monkeypatch.setattr("opentpu.isasim.board_config", lambda **kw: CFG)
    plain = C.Chat.ask_plain                    # (its stream: sys.stdout when it is called)
    monkeypatch.setattr(C.Chat, "ask_plain", lambda self, text: plain(self, text, sys.stdout))
    argv = ["--model", str(d), "--prompt", "11 22 33 44 55", "--greedy", "--max-new", "4",
            "--experts", str(K)]
    with pytest.raises(SystemExit, match="otpu-chat: no expert pool at .*pool-mx1-fp4.bin"):
        C.main(argv)
    pool = d / "pool-mx1-fp4.bin"
    _pack_pool(d, pool, "init", monkeypatch=monkeypatch)
    _pack_pool(d, pool, "pack", 0, 1, monkeypatch=monkeypatch)
    capfd.readouterr()
    C.main(argv)
    out = capfd.readouterr().out
    assert f"experts: {6 * K} slots of 48 experts (25%)" in out and str(pool) in out
    assert "weights: fp4 + attn=nvfp4,delta=nvfp4,mlp=nvfp4, head int8" in out
    assert "% hits" in out and "MB/token" in out and "misses in the prompt" in out
    with pytest.raises(SystemExit, match="runs on the ISA simulator or the card"):
        C.main(argv + ["--backend", "board-sim"])


def test_otpu_chat_clamps_the_models_top_k(tmp_path, monkeypatch, capfd):
    """otpu-chat on a checkpoint whose generation_config.json samples from the top 80 (as
    LFM2.5-8B-A1B's): top-k 64, the card's decode loop picks the tokens, and a line at startup
    says so; --top-k 80 keeps 80 (the host samples) without the line. The streamed engine keeps
    a snapshot for the next turn's rollback (LFM2's convolution rings)."""
    from opentpu.host import chat as C
    d = _checkpoint("lfm2", tmp_path / "model")
    (d / "generation_config.json").write_text(json.dumps(
        dict(do_sample=True, temperature=0.2, top_k=80, repetition_penalty=1.05)))
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda p: Tok())
    monkeypatch.setattr("opentpu.isasim.board_config", lambda **kw: CFG)
    plain, on_card = C.Chat.ask_plain, []
    monkeypatch.setattr(C.Chat, "ask_plain", lambda self, text: (
        on_card.append((self.on_card, self.pick.params["top_k"], self.eng._snaps is not None)),
        plain(self, text, sys.stdout)))
    pool = d / "pool-mx1-nvfp4.bin"             # (LFM2's experts are nvfp4: streamed.FORMATS)
    _pack_pool(d, pool, "init", "--wformat", "nvfp4", monkeypatch=monkeypatch)
    _pack_pool(d, pool, "pack", 0, 1, "--wformat", "nvfp4", monkeypatch=monkeypatch)
    argv = ["--model", str(d), "--prompt", "11 22 33", "--max-new", "3", "--experts", str(K)]
    capfd.readouterr()
    C.main(argv)
    out = capfd.readouterr().out
    assert "otpu-chat: top-k 80 (the model's default) clamped to 64" in out
    C.main(argv + ["--top-k", "80"])
    assert "clamped" not in capfd.readouterr().out
    assert on_card == [(True, 64, True), (False, 80, True)]


def test_the_interface_shows_the_streaming():
    """A streamed MoE's turn in the status line (the decode's slot hits and MB a token), the
    /stats panel and inline stats (with the prompt's misses) and the plain line."""
    pytest.importorskip("textual")
    from opentpu.host.chat_tui import stats_inline, stats_markup, status_line, welcome
    t = Turn(clock_mhz=133.33, cap=2048, prefill_tokens=20, prefill_s=2.0, gen_tokens=5,
             decode_s=1.0, decode_steps=4, decode_cycles=4_000_000, context=25, experts=True,
             expert_hits=970, expert_misses=30, streamed=50_100_000, prefill_misses=812)
    assert t.expert_hit_rate == pytest.approx(0.97)
    assert t.streamed_mb_per_token == pytest.approx(12.525)
    assert "; experts 97.0% hits, 12.5 MB/token, 812 misses in the prompt" in t.line()
    assert "experts" not in Turn(decode_steps=4).line()
    chat = SimpleNamespace(eng=SimpleNamespace(pos=25, cap=2048, backend=None), clock_mhz=133.33,
                           session=Session(turns=1, tokens_in=20, tokens_out=5, decode_s=1.0,
                                           decode_tokens=4), think=False)
    meta = {"model": "Qwen3.5-35B-A3B", "backend": "board", "device": "/dev/xdma0",
            "short": "board 133 MHz", "experts": "1680 slots of 10240 experts (16%)"}
    assert "97% hit · 12.5 MB/tok" in str(status_line(meta, chat, t))
    panel = stats_markup(meta, chat, t)
    assert "experts  1680 slots of 10240 experts" in panel
    assert "experts  97.0% hits · 12.5 MB/token · 812 misses in the prompt" in panel
    assert "97.0% hits" in stats_inline(meta, chat, t)
    assert "1680 slots" in str(welcome(meta))


def test_the_scripted_chat_checks_a_run_against_its_reference(tmp_path, monkeypatch, capsys):
    """tools/qual/chatmoe.py (the card check's driver): two greedy turns on the ISA simulator
    with the card's configuration as JSON (each turn's tokens and numbers); a second run
    --check'ed against the first passes turn by turn, and a reference with another token fails
    (exit 1, the turn and the token named). Each turn lists its calls (rows, mode, the server's
    counts, the host's time on requests and parts); without --em-rows the tool does not pass
    em_rows (it runs on production's tree, whose engine has none)."""
    d = _checkpoint("qwen35", tmp_path / "model")
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    pool = ST.default_pool(d, "fp4")
    _pack_pool(d, pool, "init", monkeypatch=monkeypatch)
    _pack_pool(d, pool, "pack", 0, 1, monkeypatch=monkeypatch)
    (tmp_path / "cfg.pkl").write_bytes(pickle.dumps(CFG))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda p: Tok())
    sys.path.insert(0, str(ROOT / "tools" / "qual"))
    try:
        import chatmoe
    finally:
        sys.path.remove(str(ROOT / "tools" / "qual"))
    argv = [str(d), "--cfg", str(tmp_path / "cfg.pkl"), "--experts", str(K), "--max-new", "5",
            "--turn", "11 22 33 44 55 66", "--turn", "77 88 99"]
    engine = ST.engine

    def production_engine(*a, **kw):        # (a tree without em_rows: the tool runs on it)
        assert "em_rows" not in kw
        return engine(*a, **kw)
    monkeypatch.setattr(ST, "engine", production_engine)
    assert chatmoe.main(argv + ["--out", str(tmp_path / "ref.json")]) == 0
    monkeypatch.setattr(ST, "engine", engine)
    ref = json.loads((tmp_path / "ref.json").read_text())
    assert len(ref["turns"]) == 2 and ref["layer_major"] == 2 and ref["on_card"]
    t1, t2 = ref["turns"]
    assert t1["tokens"] and t1["prompt_tokens"] == 6 and t1["prefill_misses"] > 0
    assert t2["fed_before"] == 6 + len(t1["tokens"]) - (t1["end"] != "eos")
    assert t1["fed"][:6] == [11, 22, 33, 44, 55, 66]   # (the conversation's ids: a turn's
    assert t2["fed"][:len(t1["fed"])] == t1["fed"]       # prompt for moe_card)
    assert t2["reused"] == t2["fed_before"] and not t2["restarted"]     # (the context kept)
    assert chatmoe.main(argv + ["--check", str(tmp_path / "ref.json")]) == 0
    assert "[PASS]" in capsys.readouterr().out
    ref["turns"][1]["tokens"][0] += 1
    (tmp_path / "bad.json").write_text(json.dumps(ref))
    assert chatmoe.main(argv + ["--check", str(tmp_path / "bad.json")]) == 1
    out = capsys.readouterr().out
    assert "[FAIL]" in out and "turn 2 from token 0" in out
    # each turn's prompt calls: with 2 slots a layer layer-major (no room for expert-major)
    assert [(c["rows"], c["em"]) for c in t1["calls"]] == [(6, False)]
    assert t1["calls"][0]["misses"] == t1["prefill_misses"] and not t1["calls"][0]["need_lines"]
    assert t1["calls"][0]["requests"] > 0 and t1["calls"][0]["serve_s"] > 0    # (the host's
    assert t1["calls"][0]["parts_s"] == 0                                       # time on them)
    assert sum(c["rows"] for c in t2["calls"]) == t2["prompt_tokens"]
    # the pool's page cache around each call and turn (mincore: a fraction of its experts),
    # the reads by where they came from, a turn read from the disk marked cold
    assert ref["pool_gb"] > 0 and 0 <= ref["pool_at_load"] <= 1
    for t in (t1, t2):
        assert 0 <= t["pool_before"] <= 1 and 0 <= t["pool_after"] <= 1
        assert t["cold"] == (bool(t["reads"].get("disk")) or t["pool_before"] < 0.95)
        assert all(c["pool_before"] is None and c["reads"] for c in t["calls"])  # (no --pool-calls)
    assert 0 < sum(v[0] for c in t1["calls"] for v in c["reads"].values()) <= \
        sum(v[0] for v in t1["reads"].values())
    # the DRAM filled, expert-major from 6 rows a call (--em-rows 6): the first turn's call,
    # not the second's 4 rows; the same tokens as the 2-slot run's; --trace: each call's
    # server timeline (need lines, then requests)
    full = [str(d), "--cfg", str(tmp_path / "cfg.pkl"), "--max-new", "5", "--em-rows", "6",
            "--turn", "11 22 33 44 55 66", "--turn", "77 88 99",
            "--trace", str(tmp_path / "trace.json"), "--out", str(tmp_path / "em.json")]
    assert chatmoe.main(full + ["--check", str(tmp_path / "ref.json"), "--pool-calls"]) == 0
    r = json.loads((tmp_path / "em.json").read_text())
    assert all(0 <= c[k] <= 1 for t in r["turns"] for c in t["calls"]
               for k in ("pool_before", "pool_after"))       # (--pool-calls: each call's too)
    assert r["expert_major"] and r["em_rows"] == 6
    c1, c2 = (t["calls"] for t in r["turns"])
    assert [(c["rows"], c["em"]) for c in c1] == [(6, True)] and c1[0]["need_lines"] > 0
    assert c2 and not any(c["em"] or c["need_lines"] for c in c2)
    assert "call: 6 rows expert-major" in capsys.readouterr().out
    tr = json.loads((tmp_path / "trace.json").read_text())["calls"]
    assert [(c["rows"], c["em"]) for c in tr] == [(c["rows"], c["em"]) for c in c1 + c2]
    assert "n" in {e[2] for e in tr[0]["events"]} and "d" in {e[2] for e in tr[1]["events"]}


def test_the_phases_line_sums_a_turns_phases():
    """chatmoe.py's phases line (_Phases.line): each phase's seconds by name (a count where it
    ran more than once), and the rest of the TTFT outside the union of the phases' spans, those
    that hold others (MTP's prefill and snapshot) left out: their parts are phases of their own."""
    sys.path.insert(0, str(ROOT / "tools" / "qual"))
    try:
        import chatmoe
    finally:
        sys.path.remove(str(ROOT / "tools" / "qual"))
    ph = [["template", 0.0, 0.1, None], ["rollback", 0.1, 0.15, None],
          ["prefill_layers", 0.3, 0.8, None], ["end_prefill", 0.7, 0.75, None],
          ["snapshot", 0.82, 0.84, None], ["mtp snapshot", 0.8, 0.85, None],
          ["prefill_layers", 0.9, 1.4, None], ["run", 1.45, 1.5, 1234],
          ["a0", 1.55, 1.6, None], ["mtp prefill", 0.2, 1.6, None]]
    assert chatmoe._Phases.line(ph, 2.0) == (
        "template 0.100 s, rollback 0.050 s, mtp prefill 1.400 s, prefill_layers 1.000 s (2), "
        "end_prefill 0.050 s, mtp snapshot 0.050 s, snapshot 0.020 s, a0 0.050 s, run 0.050 s; "
        "the rest of TTFT 0.730 s")
