"""The 35B chat on both BCU1525s (boards/bcu1525/chat35/chat35x2.py): the S = 4 machine of two cards
(bcu1525-x2:st16, CardsTransport) under otpu-chat's Engine, on the ISA simulator behind the cards'
registers (IsaCards): a tiny Qwen3.5-MoE's decode logits and the generate loop's tokens bit for
bit against the ISA simulator's one machine (and S = 1's: SEXACT), a chat's turns and their
rollback, the host's stop word on every slice (the generate loop gathers the slices' words, so
the slices stop at the same token), otpu-chat's board backend on "/dev/a+/dev/b", and chat35x2's
turn after a run the link failed."""
import numpy as np
import pytest

from opentpu import target as T
from opentpu.host.board import BoardBackend
from opentpu.host.cards import CardsTransport, IsaCards

TARGET = "bcu1525-x2:st16"

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _cfg(S: int = 4):
    return T.config(T.load(TARGET), S=S, DRAM_BYTES=1 << 26, BHW=0)


def _cards_engine(Engine, spec, W, cfg, **kw):
    sh = IsaCards(cfg)
    eng = Engine(spec, W, cap=256, cfg=cfg, **kw,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=CardsTransport(sh.cards),
                                                      status=False))
    return eng, sh


@pytest.fixture(scope="module")
def tiny35():
    """A tiny Qwen3.5-MoE shaped as the 35B at S = 4: 2 KV heads (each on two slices), DeltaNet
    heads 16 (S = 1's groups of 4 a slice: S-exact), hidden and experts S x D wide."""
    from test_moe_s2 import _tiny
    return _tiny(("linear", "attn"), nk=8, nv=16, H=512, F=512)


def test_tiny_qwen35_moe_on_two_cards(tiny35, monkeypatch):
    """Decode logits and the generate loop's tokens through CardsTransport bit for bit against the
    ISA simulator's S = 4 machine, and its logits S = 1's (SEXACT, bcu1525-x2's default); each
    card's DRAM its own slices' images."""
    from opentpu.llm.qwen3 import Engine
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    W, spec = tiny35
    cfg = _cfg()
    assert cfg.S == 4 and cfg.SEXACT
    kw = dict(wformat="fp4", rows=1, resident=True)
    isa = Engine(spec, W, cap=256, cfg=cfg, **kw)
    brd, sh = _cards_engine(Engine, spec, W, cfg, **kw)
    one = Engine(spec, W, cap=256, cfg=_cfg(1), **kw)
    assert brd.backend.info["cfg"]["S"] == 4 and brd.can_generate and isa.can_generate
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 4)]
    for tok in toks:
        a, b, c = isa.step(tok), brd.step(tok), one.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), isa.pos
        assert np.array_equal(a.view(np.uint32), c.view(np.uint32)), isa.pos
    n = isa.image.nbytes
    for s in range(4):
        assert np.array_equal(sh.dram[s][:n], isa.backend.machine.slices[s].dram[:n]), s
    tok = int(np.argmax(a))
    want = isa.generate_card(tok, 6, stop_ids=[])
    assert brd.generate_card(tok, 6, stop_ids=[]) == want and len(want) == 6


def test_one_slices_stop_word_stops_every_slice(tiny35, monkeypatch):
    """The host's stop word on one slice only (its writes to the four slices land one after
    another, two cards apart): every slice halts after the same token, the run's first (the loop
    gathers the slices' words), the cards' completion checks clean. generate_card goes on with a
    run a token (a halt that is no stop id is a chain's reach): the tokens are the ISA
    simulator's."""
    from opentpu.llm import generate as G
    from opentpu.llm.qwen3 import Engine
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    W, spec = tiny35
    cfg = _cfg()
    kw = dict(wformat="fp4", rows=1, resident=True)
    isa = Engine(spec, W, cap=256, cfg=cfg, **kw)
    brd, sh = _cards_engine(Engine, spec, W, cfg, **kw)
    tok = int(np.argmax(brd.step(17)))
    isa.step(17)
    want = isa.generate_card(tok, 5, stop_ids=[])
    run, runs = BoardBackend.run_generate, []

    def one_slice(self, programs, out, n, on_token=None, stop=None, state=None):
        self.board.write(state + 4 * G.S_HALT, np.ones(1, np.float32), 3)
        st, got = run(self, programs, out, n, on_token, stop, state)
        self.board.t.check_run()
        runs.append((n, len(got)))
        return st, got
    monkeypatch.setattr(BoardBackend, "run_generate", one_slice)
    got = brd.generate_card(tok, 5, stop_ids=[])
    assert got == want, (got, want)
    assert runs and all(k == 1 for _, k in runs) and runs[0][0] > 1, runs


def test_a_chat_on_two_cards_rolls_back(tiny35, monkeypatch):
    """otpu-chat's turns on the two cards (token by token, the 35B chat's prompt path at S > 1):
    each turn's first logits and its greedy reply those of the ISA simulator's S = 4 machine fed
    the same turns, the second turn reusing the first's context."""
    from opentpu.host.chat import Chat, sampler
    from opentpu.llm.qwen3 import Engine
    from test_chatmoe import Tok
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    W, spec = tiny35
    cfg = _cfg()
    kw = dict(wformat="fp4", rows=1, resident=True, snapshots=1)
    replies, firsts = {}, {}
    for name in ("isa", "cards"):
        if name == "isa":
            eng = Engine(spec, W, cap=256, cfg=cfg, **kw)
        else:
            eng, _ = _cards_engine(Engine, spec, W, cfg, **kw)
        rec = []
        base = sampler(0, 0, 1.0, None)

        def pick(logits, context=(), _rec=rec, _base=base):
            _rec.append(np.array(logits, np.float32).view(np.uint32))
            return _base(logits, context)
        for k in ("stream", "warm", "greedy", "params", "softcap", "rng"):
            setattr(pick, k, getattr(base, k))
        chat = Chat(eng, Tok(), False, pick, 6)
        out = []
        for msg in ("11 22 33 44", "55 66"):
            _, turn = chat.ask(msg)
            out.append((list(chat._reply), turn.context, turn.reused > 0, turn.restarted))
        replies[name], firsts[name] = out, rec
    assert replies["cards"] == replies["isa"]
    assert all(np.array_equal(a, b) for a, b in zip(firsts["cards"], firsts["isa"]))
    assert len(firsts["cards"]) == 2


def test_otpu_chats_board_backend_opens_both_cards(monkeypatch):
    """make_backend("board", dev="/dev/a+/dev/b"): the cards as one transport (open_transport),
    the machine's configuration from their registers (S = 4)."""
    from opentpu.host import board as BD
    from opentpu.host import chat as C
    cfg = _cfg()
    sh = IsaCards(cfg)
    seen = []

    def opened(dev, dma=True, target=None):
        seen.append((dev, target))
        return CardsTransport(sh.cards)         # (the tiny DRAM is not the target's)
    monkeypatch.setattr(BD, "open_transport", opened)
    monkeypatch.setenv("OTPU_TARGET", TARGET)
    backend, got = C.make_backend("board", None, 256, "/dev/a+/dev/b")
    assert seen == [("/dev/a+/dev/b", TARGET)]
    assert got.S == 4 and callable(backend)


def test_a_turn_whose_run_fails_on_the_link_is_asked_again(tiny35, monkeypatch):
    """chat35x2's turn after a run failed on the link (the second turn's third run raising
    cards.LinkError in its completion check, after it wrote the state): the link reset, the
    context back to the first turn's snapshot, the turn asked again; the replies the ISA
    simulator's."""
    import importlib.util
    from pathlib import Path
    from opentpu.host.cards import LinkError
    from opentpu.host.chat import Chat, sampler
    from opentpu.llm.qwen3 import Engine
    from test_chatmoe import Tok
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    p = Path(__file__).resolve().parents[1] / "boards/bcu1525/chat35/chat35x2.py"
    spec_ = importlib.util.spec_from_file_location("chat35x2", p)
    X2 = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(X2)
    W, spec = tiny35
    cfg = _cfg()
    kw = dict(wformat="fp4", rows=1, resident=True, snapshots=1)
    replies = {}
    for name in ("isa", "cards"):
        if name == "isa":
            eng, ask = Engine(spec, W, cap=256, cfg=cfg, **kw), Chat.ask
        else:
            eng, _ = _cards_engine(Engine, spec, W, cfg, **kw)
            ask = X2.recovering(Chat.ask)
        chat = Chat(eng, Tok(), False, sampler(0, 0, 1.0, None), 6)
        out = [ask(chat, "11 22 33 44")[0]]
        if name == "cards":
            wait, calls, recovers = BoardBackend.wait, [0], []
            t = eng.backend.board.t
            recover = t.recover

            def failing(self, *a, **k):
                st = wait(self, *a, **k)
                calls[0] += 1
                if calls[0] == 3:
                    raise LinkError("a test's failed completion check")
                return st
            monkeypatch.setattr(BoardBackend, "wait", failing)
            monkeypatch.setattr(t, "recover", lambda *a: (recovers.append(1), recover(*a)))
        out.append(ask(chat, "55 66")[0])
        if name == "cards":
            assert calls[0] > 3 and recovers == [1], (calls, recovers)
        replies[name] = out
    assert replies["cards"] == replies["isa"]


def test_both_cards_scrubbed_and_written_side_by_side(tiny35, monkeypatch):
    """On a threaded transport of two cards with ECC (the BCU1525s'), the scrub's zeros and the
    image's write go to the cards at once (each card's on a thread of its own, the second
    through a Board sharing the transport): each slice's DRAM the ISA simulator's image, and the
    decode logits its."""
    import threading
    from opentpu.host import board as BD
    from opentpu.llm.qwen3 import Engine
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    W, spec = tiny35
    cfg = _cfg()
    kw = dict(wformat="fp4", rows=1, resident=True)
    isa = Engine(spec, W, cap=256, cfg=cfg, **kw)
    sh = IsaCards(cfg)
    for c in sh.cards:
        c.threaded = c.ecc = True
    writers = {}
    mem_write = type(sh.cards[0]).mem_write

    def seen(self, ch, off, data):
        writers.setdefault(sh.cards.index(self), set()).add(threading.current_thread().name)
        return mem_write(self, ch, off, data)
    monkeypatch.setattr(type(sh.cards[0]), "mem_write", seen)
    scrubs, shared = [], []
    scrub, sharing = BD.Board.scrub, BD.Board.sharing
    monkeypatch.setattr(BD.Board, "scrub", lambda self, *a, **k: scrubs.append(
        scrub(self, *a, **k)) or scrubs[-1])
    monkeypatch.setattr(BD.Board, "sharing", lambda self: shared.append(1) or sharing(self))
    brd = Engine(spec, W, cap=256, cfg=cfg, **kw,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=CardsTransport(sh.cards),
                                                      status=False))
    assert scrubs == [True] and shared == [1] and brd.backend.board.t.threaded
    assert all(any(n.startswith("otpu-card") for n in writers[c]) for c in (0, 1)), writers
    n = isa.image.nbytes
    for s in range(4):
        assert np.array_equal(sh.dram[s][:n], isa.backend.machine.slices[s].dram[:n]), s
    tok = 123
    a, b = isa.step(tok), brd.step(tok)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


def test_layer_major_prompt_on_two_cards(tiny35, monkeypatch):
    """A prompt layer by layer at S = 4 (Engine(layer_major=2): each slice's layer run its part
    of the tensor-parallel step's, a MoE row at a time) through CardsTransport: the cards' DRAM
    the ISA simulator's after the same runs, and the logits and the next step's bit for bit
    those of the token steps on the ISA simulator at S = 4 and at S = 1."""
    from opentpu.llm.qwen3 import Engine
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    W, spec = tiny35
    kw = dict(wformat="fp4", rows=1, resident=True)
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 70)]
    engs, lgs, sh = {}, {}, None
    for name in ("cards", "isa_lm", "isa", "one"):
        cfg = _cfg(1 if name == "one" else 4)
        lm = 2 if name in ("cards", "isa_lm") else 0
        if name == "cards":
            engs[name], sh = _cards_engine(Engine, spec, W, cfg, layer_major=lm, **kw)
        else:
            engs[name] = Engine(spec, W, cap=256, cfg=cfg, layer_major=lm, **kw)
        lgs[name] = list(engs[name].prefill_chunks(toks, chunk=None if lm else 1))[-1][1]
    isa = engs["isa_lm"]
    n = isa.image.nbytes
    for s in range(4):
        assert np.array_equal(sh.dram[s][:n], isa.backend.machine.slices[s].dram[:n]), s
    out = {name: [np.asarray(v, np.float32).view(np.uint32)
                  for v in (lg, engs[name].step(int(np.argmax(lg))))] for name, lg in lgs.items()}
    for name in ("isa_lm", "isa", "one"):
        assert all(np.array_equal(a, b) for a, b in zip(out["cards"], out[name])), name


def test_warm35x2_layer_runs_are_the_engines(tiny35, monkeypatch, tmp_path):
    """warm35x2's layer-major runs (its keys, the program cache's `what`) are those the engine's
    prompt takes at S = 4: after warm_layers into an empty program cache, a prompt layer by
    layer in bucket 1 compiles none (gen_warm.start's arguments as chat35x2 keeps
    them)."""
    import importlib.util
    from pathlib import Path
    from opentpu import progcache as PC
    from opentpu.host import gen_warm as GW
    from opentpu.llm.qwen3 import Engine
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path / "qprog"))
    d = Path(__file__).resolve().parents[1] / "boards/bcu1525/chat35"
    monkeypatch.syspath_prepend(str(d))
    spec_ = importlib.util.spec_from_file_location("warm35x2", d / "warm35x2.py")
    WX = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(WX)
    W, spec = tiny35
    eng, _ = _cards_engine(Engine, spec, W, _cfg(), layer_major=2, wformat="fp4", rows=1,
                           resident=True)
    assert eng.prog_cache
    be = eng.backend
    args = (eng.spec, eng.cfg, eng.cap, eng.batch, eng.rows, eng._image_kw,
            eng.image.stream_fill, eng.layout, eng.block, eng._conv_lo,
            bool(getattr(be, "chains", False)), eng.gen_debug, eng.gen_split,
            bool(getattr(be, "waitc", False)), GW.samplings(None))
    PC.clear()
    WX.warm_layers(args, [1], 2)
    PC.clear()
    before, disk = PC.stats["compile"], PC.stats["disk"]
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 120)]
    list(eng.prefill_chunks(toks))
    assert PC.stats["compile"] == before and PC.stats["disk"] >= disk + 2 * spec.layers, PC.stats


def test_prompt_runs_read_as_words_are_the_engines(tiny35, monkeypatch, tmp_path):
    """opentpu.host.runwords at S = 4: a prompt layer by layer whose runs the program cache holds
    goes to the cards as the cache's words (no decode, no assembly: every layer run's programs
    None, the words compiler.Words), its logits and the next step's bit for bit those of an
    engine that read the same entries the program cache's way; a run the cache does not hold is
    compiled the engine's way."""
    from opentpu import progcache as PC
    from opentpu.compiler import Words
    from opentpu.host import runwords as RW
    from opentpu.llm.qwen3 import Engine
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path / "qprog"))
    W, spec = tiny35
    kw = dict(layer_major=2, wformat="fp4", rows=1, resident=True)
    toks = [int(t) for t in np.random.default_rng(6).integers(0, 1000, 90)]
    out = {}
    for name in ("compile", "cache", "words"):
        PC.clear()
        eng, _ = _cards_engine(Engine, spec, W, _cfg(), **kw)
        if name == "words":
            assert RW.install(eng)
        before = dict(PC.stats)
        lg = list(eng.prefill_chunks(toks, chunk=None))[-1][1]
        out[name] = [np.asarray(v, np.float32).view(np.uint32)
                     for v in (lg, eng.step(int(np.argmax(lg))))]
        runs = list(eng._layer_runs.values())
        if name == "compile":
            assert PC.stats["compile"] > before["compile"]
        else:
            assert PC.stats["compile"] == before["compile"], (name, PC.stats)
        if name == "words":
            assert runs and all(p is None and len(w) == 4 and all(isinstance(x, Words) for x in w)
                                for p, _, w in runs)
        else:
            assert all(p is not None and w is None for p, _, w in runs)
    for name in ("cache", "words"):
        assert all(np.array_equal(a, b) for a, b in zip(out["compile"], out[name])), name
    eng, _ = _cards_engine(Engine, spec, W, _cfg(), **kw)
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path / "empty"))
    assert RW.install(eng)
    lg = list(eng.prefill_chunks(toks, chunk=None))[-1][1]
    assert np.array_equal(np.asarray(lg, np.float32).view(np.uint32), out["compile"][0])
    assert all(p is not None for p, _, _ in eng._layer_runs.values())


def test_ahead_keeps_the_cards_warm_arguments(tiny35, monkeypatch):
    """chat35x2's CHAT35_AHEAD (no card): an Engine on its Ahead backend and the image's
    stand-ins keeps the gen_warm arguments (warm35x2.pkl's) the cards' engine does, the chat's
    sampler's too, so warm35x2.py warms what the chat on the cards then takes."""
    import importlib.util
    import pickle
    from pathlib import Path
    from opentpu import imgcache as IC
    from opentpu.host import gen_warm as GW
    from opentpu.llm.qwen3 import Engine
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    p = Path(__file__).resolve().parents[1] / "boards/bcu1525/chat35/chat35x2.py"
    spec_ = importlib.util.spec_from_file_location("chat35x2", p)
    X2 = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(X2)
    W, spec = tiny35
    cfg = _cfg()
    kw = dict(layer_major=2, wformat="fp4", rows=1, resident=True, snapshots=2)
    cards, _ = _cards_engine(Engine, spec, W, cfg, **kw)
    monkeypatch.setattr(IC, "build", lambda image, W, layout: X2.ahead_images(cfg))
    ahead = Engine(spec, W, cap=256, cfg=cfg, backend=X2.Ahead, **kw)
    samp = dict(temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.0)

    def args(eng):
        be = eng.backend
        return (eng.spec, eng.cfg, eng.cap, eng.batch, eng.rows, eng._image_kw,
                eng.image.stream_fill, eng.layout, eng.block, eng._conv_lo,
                bool(getattr(be, "chains", False)), eng.gen_debug, eng.gen_split,
                bool(getattr(be, "waitc", False)), GW.samplings(samp))
    a, c = args(ahead), args(cards)
    assert ahead.can_generate and ahead.resident and cards.can_generate
    assert c[10] and c[13] and len(c[14]) == 2
    assert pickle.dumps(a) == pickle.dumps(c)


def test_layer_experts_reads_a_layers_fused_tensors_once(tiny35):
    """chat35x2's LayerExperts (the image build's W at S = 4): every expert's matrices are W's,
    and each layer's fused expert tensors are read from W once, not once an expert a slice."""
    import importlib.util
    from pathlib import Path
    from opentpu.llm import qwen35 as Q35
    p = Path(__file__).resolve().parents[1] / "boards/bcu1525/chat35/chat35x2.py"
    spec_ = importlib.util.spec_from_file_location("chat35x2", p)
    X2 = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(X2)
    W, spec = tiny35
    reads = []

    class Counted(dict):
        def __getitem__(self, k):
            reads.append(k)
            return dict.__getitem__(self, k)
    L = X2.LayerExperts(Counted(W))
    E = spec.moe.E
    for g in range(spec.layers * E):
        pre = f"model.layers.{g // E}."
        for _ in range(4):
            for x, y in zip(Q35._expert(L, pre, g % E), Q35._expert(W, pre, g % E)):
                np.testing.assert_array_equal(x, y)
    fused = [k for k in reads if k.endswith(X2.LayerExperts.FUSED)]
    assert len(fused) == 2 * spec.layers == len(set(fused))


def test_warm35_decode_steps_are_every_slices(tiny35, monkeypatch, tmp_path):
    """warm35's resident decode step at S = 4 is the engine's: every slice's programs (the
    engine compiles them itself at S > 1, no compile workers), not _worker_decode's words of
    slice 0, which the cards ran with HALT on slices 1-3 (x2a, 11-03: the first prompt token step
    with its position as arguments hung). After warm35.run of the step's bucket into an empty
    program cache the engine's decode step compiles nothing and assembles as its own compile;
    a one-slice program started on four slices raises (the cache gives it back as a list of one)."""
    import importlib.util
    import os
    from pathlib import Path
    from opentpu import isa as I
    from opentpu import progcache as PC
    from opentpu.host import gen_warm as GW
    from opentpu.llm.qwen3 import Engine
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path / "qprog"))
    monkeypatch.setattr(os, "nice", lambda n: 0)
    d = Path(__file__).resolve().parents[1] / "boards/bcu1525/chat35"
    spec_ = importlib.util.spec_from_file_location("warm35", d / "warm35.py")
    W5 = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(W5)
    W, spec = tiny35
    eng, _ = _cards_engine(Engine, spec, W, _cfg(), layer_major=2, wformat="fp4", rows=1,
                           resident=True)
    assert eng.prog_cache and eng.resident
    be = eng.backend
    args = (eng.spec, eng.cfg, eng.cap, eng.batch, eng.rows, eng._image_kw,
            eng.image.stream_fill, eng.layout, eng.block, eng._conv_lo,
            bool(getattr(be, "chains", False)), eng.gen_debug, eng.gen_split,
            bool(getattr(be, "waitc", False)), GW.samplings(None))
    pos = eng._conv_lo
    b = pos // eng.block + 1
    PC.clear()
    W5.run(0, 1, args, lo=b, hi=b)
    PC.clear()
    before = PC.stats["compile"]
    progs, ra = eng._decode(pos)
    assert PC.stats["compile"] == before, PC.stats
    assert not isinstance(progs, np.ndarray) and len(progs) == eng.cfg.S
    own, ra_own = eng.image.compile_decode(b, max((b - 1) * eng.block, eng._conv_lo), eng.block)
    assert repr(ra) == repr(ra_own)
    for p, q in zip(progs, own):
        np.testing.assert_array_equal(np.asarray(I.assemble(p)), np.asarray(I.assemble(q)))
    for one in (np.asarray(I.assemble(own[0]), np.uint32), own[:1]):
        with pytest.raises(ValueError, match="slice"):
            be.start(one)


def test_chain_slots_load_without_follow_on_two_cards(tiny35, monkeypatch):
    """A chained bucket's program (its slot: BoardBackend.slot_words, which HALT CHAIN loads)
    carries the wait entries load_program gives a machine of two cards: FOLLOW cleared
    (Board.cross_card_entries). x2a, 11-04: the 35B's generate loop was bit-exact up to its
    first chain into bucket 2, then read the gathered rows before the other card's pieces
    landed. The ISA simulator has no such race: the entries are checked, not the tokens."""
    from opentpu import isa as I
    from opentpu import waits as W
    from opentpu.llm.qwen3 import Engine
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    W_, spec = tiny35
    eng, _ = _cards_engine(Engine, spec, W_, _cfg(), wformat="fp4", rows=1, resident=True)
    be, cfg = eng.backend, eng.cfg
    assert be.chains and be.waitc and be.board.t.ncards == 2
    progs = eng._generate_prog(1)
    seen = 0
    for pj in progs if isinstance(progs, tuple) else (progs,):
        for prog in pj:
            w = np.asarray(I.assemble(prog), np.uint32)
            e = W.entries(w, cfg.D, cfg.S, prog=prog, aw=cfg.AW, win=cfg.WC_WIN)
            seen += int(np.count_nonzero(e & np.uint64(W.FOLLOW)))
            slot = be.slot_words(prog)
            got = slot[-2 * len(e):].view("<u8")
            np.testing.assert_array_equal(got, e & ~np.uint64(W.FOLLOW))
            np.testing.assert_array_equal(slot[:len(w)], w)
    assert seen, "no FOLLOW in the generate program's entries: the check checks nothing"


def test_the_model_entry(monkeypatch):
    """CHAT35_MODEL: unset the 35B as it was (no prompt runs, layer by layer); lfm2-2.6b its whole
    128,000 positions and prompt runs in every bucket, prefill.covers' other route (it compiles on
    the chat's thread at S > 1) never taken."""
    import importlib
    import sys
    from pathlib import Path
    from opentpu.llm import prefill as PF
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "boards" / "bcu1525" / "chat35"))
    monkeypatch.delenv("CHAT35_MODEL", raising=False)
    import chat35x2 as X2
    X2 = importlib.reload(X2)
    assert X2.MODEL == "qwen36-35b" and X2.FIXED[X2.FIXED.index("--model") + 1] == "qwen36-35b"
    assert "--no-prompt-runs" in X2.FIXED and X2.FIXED[X2.FIXED.index("--cap") + 1] == str(X2.CAP)
    monkeypatch.setenv("CHAT35_MODEL", "lfm2-2.6b")
    X2 = importlib.reload(X2)
    assert "--no-prompt-runs" not in X2.FIXED and X2.FIXED[X2.FIXED.index("--cap") + 1] == "128000"
    monkeypatch.setattr(PF, "covers", lambda *a: False)
    X2.dense_prompts()
    assert PF.covers(None, 0, 128000)
    monkeypatch.delenv("CHAT35_MODEL")
    importlib.reload(X2)
