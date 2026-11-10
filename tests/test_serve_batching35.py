"""The Qwen3.5-MoE batching worker (opentpu/serve/batching35.py) on the ISA simulator: the
tiny 16-head untied Qwen3.5-MoE of test_batch35 in an Engine(batch=2, batch_area=True,
layer_major=2). Requests submitted together run in their slots (their prompts by layer runs
at the slot, their decode steps batched, each token picked on the device): every request's
greedy tokens equal a fresh single-sequence engine's; more requests than slots wait for one;
a request continuing a finished conversation feeds only what is new, from its slot or from
the KV store (pages and DeltaNet state restored into another slot); a client that leaves while
its prompt is fed frees the slot and a retry resumes; how far the layer runs' files reach."""
from dataclasses import replace

import numpy as np
import pytest

from opentpu.llm.qwen3 import Engine
from opentpu.serve import batching35 as BW35
from opentpu.serve import scheduler as SC

from test_batch35 import _cfg, _tiny
from test_serve_scheduler import _Tok, _prompts

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

CAP = 256


@pytest.fixture(scope="module")
def tiny():
    return _tiny()


def _ref(spec, W, cfg, prompt, n):
    """Greedy tokens of a fresh single-sequence engine (its prompt layer by layer)."""
    e = Engine(spec, W, cap=CAP, cfg=cfg, resident=True, layer_major=2)
    lg, out = e.prefill(prompt), []
    while True:
        out.append(int(np.argmax(lg)))
        if len(out) == n:
            return out
        lg = e.step(out[-1])


def test_requests_batched_in_slots_are_their_own_runs(tiny, monkeypatch):
    W, spec = tiny
    cfg = _cfg(monkeypatch)
    eng = Engine(spec, W, cap=CAP, cfg=cfg, batch=2, resident=True, layer_major=2,
                 batch_area=True)
    assert BW35.batches(eng)
    w = BW35.BatchWorker35(eng, _Tok(), prefill_budget=4).start()
    try:
        ps = _prompts(11, [9, 6, 11])               # three requests, two slots
        with w.cv:
            reqs = [w.submit(SC.Request(p, max_new=5, stop_ids=frozenset())) for p in ps]
        for r in reqs:
            r.wait(timeout=900)
        for p, r in zip(ps, reqs):
            assert r.finish == "length" and r.error is None, r.error
            assert r.out == _ref(spec, W, cfg, p, 5), r.rid
        st = w.usage()
        assert st["steps"] > 0 and st["step_rows"] > st["steps"]      # rows batched
        assert {r.slot for r in reqs} == {0, 1}
        # the first conversation's next turn: its slot holds it, or the store does (its slot
        # taken by the third request: saved, then restored with its DeltaNet state)
        for i in (0, 2):
            nxt = ps[i] + reqs[i].out + _prompts(12 + i, [4])[0]
            r = w.submit(SC.Request(nxt, max_new=4, stop_ids=frozenset())).wait(900)
            assert r.error is None, r.error
            assert r.cached == len(ps[i]) + len(reqs[i].out) - 1, (i, r.cached)
            assert r.out == _ref(spec, W, cfg, nxt, 4), i
        assert w.stats["restores"] >= 1 and w.stats["saves"] >= 1, w.stats
    finally:
        w.stop()



def test_requests_in_slots_on_two_cards(monkeypatch):
    """x2 (S = 4: both BCU1525s as one machine, CardsTransport over IsaCards): three requests in
    two slots, each one's greedy tokens a single-sequence engine's (layer-major prompt) on the
    same machine; the next turns from the slot and from the KV store (a slot's pages and
    DeltaNet state saved and written back through the staging area on both cards)."""
    from test_batch35 import _cards, _x2
    W, spec = _tiny(H=512, f=512, kinds=("linear", "attn"))
    cfg = _x2(monkeypatch)

    def ref(prompt, n):
        e = Engine(spec, W, cap=CAP, cfg=cfg, resident=True, layer_major=2)
        lg, out = e.prefill(prompt), []
        while True:
            out.append(int(np.argmax(lg)))
            if len(out) == n:
                return out
            lg = e.step(out[-1])
    eng = Engine(spec, W, cap=CAP, cfg=cfg, batch=2, resident=True, layer_major=2,
                 batch_area=True, backend=_cards(cfg))
    w = BW35.BatchWorker35(eng, _Tok(), prefill_budget=4).start()
    try:
        ps = _prompts(11, [9, 6, 11])
        with w.cv:
            reqs = [w.submit(SC.Request(p, max_new=5, stop_ids=frozenset())) for p in ps]
        for r in reqs:
            r.wait(timeout=900)
        for p, r in zip(ps, reqs):
            assert r.error is None and r.out == ref(p, 5), (r.rid, r.error)
        for i in (0, 2):
            nxt = ps[i] + reqs[i].out + _prompts(12 + i, [4])[0]
            r = w.submit(SC.Request(nxt, max_new=4, stop_ids=frozenset())).wait(900)
            assert r.error is None and r.cached == len(ps[i]) + len(reqs[i].out) - 1, (i, r.error)
            assert r.out == ref(nxt, 4), i
        assert w.stats["restores"] >= 1 and w.stats["saves"] >= 1, w.stats
        assert w.stats["step_rows"] > w.stats["steps"], w.stats       # (rows batched)
    finally:
        w.stop()


@pytest.mark.parametrize("cap", [CAP, 1024])
def test_sampled_requests_on_two_cards(monkeypatch, cap):
    """A seeded request with production's sampling (T 0.7, top_k 20, top_p 0.8) gives the same
    tokens on one slice (x2's target at S = 1) and on x2's two cards (S = 4: each slice's top k
    gathered), alone and with another sampled request (a repetition penalty) batched beside it:
    each row's pick on the cards with its own uniforms. At cap 1024 the steps run over slots
    (batch35.slots_fit: one program per slots and kinds)."""
    from test_batch35 import _cards, _x2
    W, spec = _tiny(H=512, f=512, kinds=("linear", "attn"))
    samp = SC.Sampling(0.7, 20, 0.8, 1.0, seed=11)
    p, q = _prompts(9, [10, 7])
    kw = dict(cap=cap, batch=2, resident=True, layer_major=2, batch_area=True)

    def run(cfg, backend=None, other=False):
        eng = Engine(spec, W, cfg=cfg, **kw, **({"backend": backend} if backend else {}))
        w = BW35.BatchWorker35(eng, _Tok(), prefill_budget=4).start()
        assert w.steps.slots == (cap > CAP)
        try:
            with w.cv:
                if other:
                    o = w.submit(SC.Request(q, max_new=9, stop_ids=frozenset(),
                                            sampling=SC.Sampling(0.9, 20, 0.9, 1.3, seed=3)))
                r = w.submit(SC.Request(p, max_new=8, stop_ids=frozenset(), sampling=samp))
            r.wait(900)
            if other:
                o.wait(900)
                assert o.error is None and o.finish == "length", o.error
                assert w.stats["step_rows"] > w.stats["steps"], w.stats  # (rows batched)
            assert r.error is None and r.finish == "length", r.error
            return r.out
        finally:
            w.stop()
    one = run(_x2(monkeypatch, S=1))
    cfg = _x2(monkeypatch)
    assert run(cfg, _cards(cfg)) == one
    assert run(cfg, _cards(cfg), other=True) == one


def test_warm_steps_leaves_nothing_to_compile_on_two_cards(monkeypatch):
    """BatchWorker35.warm_steps (api.run's, before the server listens) on x2's two cards over
    slots: with the defaults' sampling, every slot set's program for greedy and sampled rows (2
    one-row programs a slot and 4 pairs: 8) live before start(), each compiled once, with
    batch.AHEAD 3 and LIVE 4 below the 8 (the card's B = 4 warm, 11-05: 80 programs against 16
    and 64, 63 compiled twice and 16 evicted before the first request). A greedy request and a
    short sampled one with that sampling, submitted together with a prefill budget of 4 tokens
    an iteration (the short prompt fed beside the first one's steps, then stepping with it): no
    program compiled and no compile waited for from start() on, each request's tokens a cold
    worker's."""
    from opentpu.llm import batch as BA
    from test_batch35 import _cards, _x2
    monkeypatch.setattr(BA, "AHEAD", 3)
    monkeypatch.setattr(BA, "LIVE", 4)
    W, spec = _tiny(H=512, f=512, kinds=("linear", "attn"))
    p, q = _prompts(9, [10, 7])
    cfg = _x2(monkeypatch)
    kw = dict(cap=1024, batch=2, resident=True, layer_major=2, batch_area=True)

    def run(warm):
        eng = Engine(spec, W, cfg=cfg, backend=_cards(cfg), **kw)
        w = BW35.BatchWorker35(eng, _Tok(), prefill_budget=4)
        assert w.steps.slots
        n = w.warm_steps([SC.Sampling(0.7, 20, 0.8, 1.0)]) if warm else 0
        tm = w.steps.timing
        made, waited = tm["programs"], tm["compile_wait_s"]
        assert not warm or (tm["compiled_inline"] == 0 and len(w.steps._live) == n), tm
        w.start()
        try:
            with w.cv:
                rs = [w.submit(SC.Request(p, max_new=8, stop_ids=frozenset())),
                      w.submit(SC.Request(q, max_new=9, stop_ids=frozenset(),
                                          sampling=SC.Sampling(0.7, 20, 0.8, 1.0, seed=5)))]
            for r in rs:
                r.wait(900)
            assert all(r.error is None and r.finish == "length" for r in rs), [r.error for r in rs]
            assert w.stats["step_rows"] > w.stats["steps"], w.stats         # (rows batched)
            return ([r.out for r in rs], n, made, tm["programs"] - made,
                    tm["compile_wait_s"] - waited)
        finally:
            w.stop()
    want, _, _, cold, _ = run(False)
    got, n, made, more, wait = run(True)
    assert n == 8 and made == 8 and cold > 0, (n, made, cold)
    assert more == 0 and wait == 0.0, (more, wait)
    assert got == want, (got, want)


def test_rows_ahead_take_the_prompts_this_budget_ends():
    """_rows_at (the rows _ahead compiles for before the iteration's prompt runs): the decoding
    requests, and the prompts whose runs this iteration's budget ends, at their first step's row
    (the prompt's last token: len(prompt) - 1); not a prompt still feeding after it; k
    iterations on, each a row further, a request at its max_tokens gone."""
    from types import SimpleNamespace as NS

    def req(slot, fed, feed, out=0):
        return NS(finish=None, next=1, slot=slot, fed=[1] * fed, _feed=[1] * feed, _cuts=[],
                  out=[1] * out, max_new=8, _bset=True)
    a, b, c = req(0, 20, 0, out=7), req(1, 3, 100), req(2, 0, 600)
    w = BW35.BatchWorker35.__new__(BW35.BatchWorker35)
    w.running, w.prefilling, w.budget = [c, a, b], [b, c], 512
    w.steps = NS(kinds={0: "g", 1: "g", 2: "g"})
    assert w._rows_at(0) == ([(0, 20), (1, 103)], ["g", "g"])
    assert w._rows_at(1) == ([(1, 104)], ["g"])


def test_decoding_compile_the_batch_less_one_ahead():
    """_ahead with two requests decoding: each one's step alone compiled ahead (the other may
    end on a stop token), not slot 0's while nothing waits (alone it runs the card's decode
    loop); with three decoding, each pair's."""
    from types import SimpleNamespace as NS

    def req(slot, fed):
        return NS(finish=None, next=1, slot=slot, fed=[1] * fed, _feed=[], out=[], max_new=99,
                  _bset=True)
    alone, rows = [], []
    w = BW35.BatchWorker35.__new__(BW35.BatchWorker35)
    w.steps = NS(kinds={0: "g", 1: "g", 2: "g"},
                 prefetch_group=lambda r, k: alone.append(r[0] if len(r) == 1 else tuple(r)),
                 prefetch_rows=lambda r, k: rows.append(r))
    w.eng = NS(cap=1024, can_generate=True)
    w.running, w.prefilling, w.waiting, w.budget = [req(0, 20), req(1, 300)], [], [], 512
    w._ahead()
    assert alone == [(1, 300)] and rows[0] == [(0, 20), (1, 300)], (alone, rows)
    alone.clear()
    w.waiting = [NS()]
    w._ahead()
    assert alone == [(1, 300), (0, 20)], alone
    alone.clear()
    w.running.append(req(2, 40))
    w._ahead()
    assert alone == [((1, 300), (2, 40)), ((0, 20), (2, 40)), ((0, 20), (1, 300))], alone


def test_short_prompt_is_an_error(tiny, monkeypatch):
    """A prompt shorter than conv_k: its last token's step row would sit before the windows
    are full; the request ends with an error, the worker goes on."""
    W, spec = tiny
    cfg = _cfg(monkeypatch)
    eng = Engine(spec, W, cap=CAP, cfg=cfg, batch=2, resident=True, layer_major=2,
                 batch_area=True)
    w = BW35.BatchWorker35(eng, _Tok()).start()
    try:
        r = w.submit(SC.Request([5, 6], max_new=3, stop_ids=frozenset())).wait(900)
        assert r.finish == "error" and "2 tokens" in (r.error or ""), r.error
        p = _prompts(13, [7])[0]
        r2 = w.submit(SC.Request(p, max_new=3, stop_ids=frozenset())).wait(900)
        assert r2.error is None and r2.out == _ref(spec, W, cfg, p, 3)
    finally:
        w.stop()


def test_streamed_engine_with_slots(tmp_path, monkeypatch):
    """otpu-chat --slots 2's engine (streamed.engine(batch=2, batch_area=True), otpu-serve35
    --slots 2's) from a checkpoint: the probe that sizes it holds both slots, its experts are a
    resident pool as on the BCU1525, and the worker's requests are a single-sequence streamed
    engine's of the same checkpoint."""
    from safetensors.numpy import save_file

    from opentpu.host import streamed as ST
    W, spec, hc = _tiny(hf=True)
    d = tmp_path / "model"
    d.mkdir()
    hc.save_pretrained(d)
    save_file({k: np.ascontiguousarray(v) for k, v in W.items()}, str(d / "model.safetensors"))
    cfg = _cfg(monkeypatch)
    kw = dict(cap=CAP, pool=None, wformat="int8", head_format="int8", layer_major=2)
    img = ST.probe(spec, cfg, CAP, batch=2, wformat="int8", head_format="int8", lookup=True)
    assert img.batch == 2 and img.slots == 2
    eng = ST.engine(d, cfg, batch=2, batch_area=True, **kw)
    assert eng.image.batch == 2 and eng.bat is not None and eng.server is None
    assert BW35.batches(eng)
    ref = ST.engine(d, cfg, **kw)
    ps = _prompts(21, [8, 10])
    want = []
    for p in ps:
        ref.reset()
        lg, out = ref.prefill(p), []
        while len(out) < 3:
            out.append(int(np.argmax(lg)))
            lg = ref.step(out[-1])
        want.append(out)
    w = BW35.BatchWorker35(eng, _Tok()).start()
    try:
        with w.cv:
            reqs = [w.submit(SC.Request(p, max_new=3, stop_ids=frozenset())) for p in ps]
        for r, t in zip(reqs, want):
            r.wait(timeout=900)
            assert r.error is None and r.out == t, (r.error, r.out, t)
    finally:
        w.stop()


def test_resident_chat_engine_with_slots(tmp_path, monkeypatch):
    """otpu-chat --slots 2 on a resident Qwen3.5-MoE (chat35's: no expert streaming, the
    BCU1525's DRAM holds every expert; otpu-serve35 --slots 2's engine): an Engine(batch=2,
    batch_area=True) without snapshots that the batching worker takes; without --slots the
    chat's one-sequence engine with its snapshots. --slots on a model with no MoE: refused."""
    from safetensors.numpy import save_file

    import opentpu.host.chat as C
    W, spec, hc = _tiny(hf=True)
    d = tmp_path / "model"
    d.mkdir()
    hc.save_pretrained(d)
    save_file({k: np.ascontiguousarray(v) for k, v in W.items()}, str(d / "model.safetensors"))
    cfg = _cfg(monkeypatch)
    monkeypatch.setattr(C.ST, "streams", lambda spec: False)       # (chat35's)
    monkeypatch.setattr(C, "make_backend", lambda *a, **kw: ("isa", cfg))
    monkeypatch.setattr(C, "load_tokenizer", lambda path: (lambda: None))
    got = []

    class Opened:                       # (chat35.open_engine's: the engine main opens)
        use_mtp = False

        def __init__(self, eng, *a, **kw):
            got.append(eng)

        def ask_plain(self, text):
            pass
    monkeypatch.setattr(C, "Chat", Opened)
    argv = ["--model", str(d), "--raw", "--backend", "isa", "--cap", str(CAP), "--prompt", "-",
            "--no-prompt-runs", "--no-prog-cache", "--wformat", "int8", "--head-format", "int8"]
    C.main(argv + ["--slots", "2"])
    eng = got[-1]
    assert eng.image.batch == 2 and eng.bat is not None and eng._snaps is None
    assert BW35.batches(eng)
    C.main(argv)
    assert got[-1].image.batch == 1 and got[-1].bat is None and got[-1]._snaps is not None
    monkeypatch.setattr(C, "load_spec", lambda path: replace(spec, moe=None))
    with pytest.raises(SystemExit, match="--slots takes a Qwen3.5 MoE model"):
        C.main(argv + ["--slots", "2"])


@pytest.mark.parametrize("slots", [2, 4])
def test_batch_check_against_a_slotted_server(tiny, monkeypatch, capsys, tmp_path, slots):
    """serve35_batch_check.py (otpu-serve35 --slots N's card check) against otpu-serve over a
    BatchWorker35 on the tiny model (shorter prompts): ALL PASS, each reply beside the others
    its reply alone's word for word, the batched steps' timing from /v1/stats. At 4 slots
    (the B=4 window's) all four prompts and the sampled rows at once, and --chats 4: four
    three-turn conversations at once, every turn its reply alone's."""
    import json
    import sys
    import threading
    from pathlib import Path

    from opentpu.serve import api as API
    from test_serve_api import _Tok as ApiTok
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "boards" / "bcu1525" / "chat35"))
    import serve35_batch_check as BC
    import serve35_check as SK
    W, spec = tiny
    cfg = _cfg(monkeypatch)
    eng = Engine(spec, W, cap=CAP, cfg=cfg, batch=slots, resident=True, layer_major=2,
                 batch_area=True)
    tok = ApiTok()
    w = BW35.BatchWorker35(eng, tok).start()
    srv = API.Server(w, tok, spec, "qwen36-35b", dict(temperature=0.0, top_k=20, top_p=1.0,
                                                       repetition_penalty=1.0))
    httpd = API.serve(srv, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(BC, "PROMPTS", {"A": "planets in order", "B": "fibonacci in python",
                                        "C": "why is the sky blue", "D": "rivers of europe"})
    monkeypatch.setattr(BC, "CHATS", [["{paste}\nmost?", "e2?", "sum?"], ["bakery", "two more", "pick"],
                                      ["primes", "91?", "check"], ["water", "sun?", "sum up"]])
    monkeypatch.setattr(SK, "FAILS", [])
    monkeypatch.setattr(BC, "FAILS", SK.FAILS)
    chats = ["--chats", "4", "--chat-tokens", "3", "--paste-lines", "2"] if slots == 4 else []
    monkeypatch.setattr(sys, "argv", ["serve35_batch_check.py", "--url",
                                      f"http://127.0.0.1:{httpd.server_address[1]}",
                                      "--max-tokens", "4", "--replies", str(tmp_path / "r.json")] + chats)
    try:
        assert BC.main() == 0, SK.FAILS
        out = capsys.readouterr().out
        assert "D beside A+B+C: its reply alone's" in out, out
        reps = json.loads((tmp_path / "r.json").read_text())
        assert set(reps) == {"greedy", "sampled", "chats"} and set(reps["greedy"]) == set(BC.PROMPTS)
        assert all(reps["greedy"].values()) and len(reps["chats"]) == (4 if slots == 4 else 0)
        seeds = "A+B+C+D" if slots == 4 else "A+B"
        assert f"{seeds} sampled steps:" in out.replace(" + ", "+"), out
        if slots == 4:
            assert "chat 3 turn 2 beside 3 other chats: its reply alone's" in out, out
    finally:
        httpd.shutdown()
        httpd.server_close()
        w.stop()


def _http_server(w, tok, spec):
    import threading

    from opentpu.serve import api as API
    srv = API.Server(w, tok, spec, "qwen36-35b", dict(temperature=0.0, top_k=20, top_p=1.0,
                                                       repetition_penalty=1.0))
    httpd = API.serve(srv, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _generate(port, prompt, n, stream=False):
    """A client's socket with SGLang's /generate of `prompt`'s token ids sent."""
    import json
    import socket
    s = socket.create_connection(("127.0.0.1", port), timeout=120)
    b = json.dumps({"input_ids": prompt, "stream": stream,
                    "sampling_params": {"max_new_tokens": n, "temperature": 0.0, "ignore_eos": True}}).encode()
    s.sendall(f"POST /generate HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              f"Content-Length: {len(b)}\r\n\r\n".encode() + b)
    return s


def test_an_abandoned_prompt_frees_its_slot_and_a_retry_resumes(tiny, monkeypatch):
    """A client that closes while its prompt is fed (the live server's path: BatchWorker35 behind
    the API): the request is cancelled while the part running goes on, no further part is fed, the
    slot is free and holds what was fed, the same prompt again feeds only the rest in that slot
    and its tokens are a fresh engine's; a client that half-closes gets its reply whole, with the
    same tokens."""
    import http.client
    import json
    import socket
    import threading
    import time

    from opentpu.serve import api as API
    from test_serve_api import _Tok as ApiTok
    W, spec = tiny
    cfg = _cfg(monkeypatch)
    eng = Engine(spec, W, cap=CAP, cfg=cfg, batch=2, resident=True, layer_major=2,
                 batch_area=True)
    tok = ApiTok()
    budget, n = 4, 5
    w = BW35.BatchWorker35(eng, tok, prefill_budget=budget).start()
    httpd = _http_server(w, tok, spec)
    port = httpd.server_address[1]
    monkeypatch.setattr(API, "CLIENT_POLL_S", 0.05)
    reqs, parts, real, real_submit = [], [], BW35.B35.prefill_slot, w.submit
    hold_at, seen = {"part": 2}, threading.Event()
    cancel_s = []

    def submit(r):
        reqs.append(r)
        return real_submit(r)

    def prefill_slot(e, tokens, slot, logits=True):
        out = real(e, tokens, slot, logits)
        parts.append((slot, [int(t) for t in tokens]))
        if len(parts) == hold_at["part"]:       # (the part a client leaves in: a long one)
            seen.set()
            t0 = time.monotonic()
            while not reqs[-1].cancelled.is_set() and time.monotonic() - t0 < 30:
                time.sleep(0.005)
            cancel_s.append(time.monotonic() - t0)
        return out
    monkeypatch.setattr(w, "submit", submit)
    monkeypatch.setattr(BW35.B35, "prefill_slot", prefill_slot)
    try:
        p, p2 = _prompts(21, [24, 21])
        s = _generate(port, p, n, stream=True)
        assert seen.wait(120)
        s.close()
        a = reqs[0]
        assert a.wait(60).finish == "abort" and a.cancelled.is_set()
        assert cancel_s[0] < 5, cancel_s            # (noticed in polls, not at the prompt's end)
        assert len(parts) == 2, parts               # (no part after the one it left in)
        fed = [t for _, ts in parts for t in ts]
        assert fed == p[:2 * budget] and a.fed == fed
        slot = parts[0][0]
        assert w.slot_req[slot] is None and w.slot_dev[slot] == fed
        assert w.stats["aborted"] == 1, w.stats
        # the same prompt again (opencode's retry): only the rest, in that slot
        del parts[:]
        hold_at["part"] = None
        s = _generate(port, p, n)
        resp = http.client.HTTPResponse(s)
        resp.begin()
        assert resp.status == 200
        got = json.loads(resp.read())
        s.close()
        b = reqs[1]
        assert b.error is None and b.cached == len(fed), (b.error, b.cached)
        assert {x for x, _ in parts} == {slot}
        assert [t for _, ts in parts for t in ts] == p[len(fed):-1]
        ref = _ref(spec, W, cfg, p, n)
        assert b.out == ref, (b.out, ref)
        assert got["meta_info"]["completion_tokens"] == n
        # a client that half-closes after sending: not cancelled, its reply whole
        s = _generate(port, p2, n, stream=True)
        s.shutdown(socket.SHUT_WR)
        resp = http.client.HTTPResponse(s)
        resp.begin()
        data = resp.read().decode()
        s.close()
        c = reqs[2]
        assert c.error is None and not c.cancelled.is_set() and c.finish == "length", c.finish
        assert c.out == _ref(spec, W, cfg, p2, n)
        assert json.loads([ln for ln in data.split("\n") if ln.startswith("data: {")][-1][6:])[
            "meta_info"]["completion_tokens"] == n
    finally:
        httpd.shutdown()
        httpd.server_close()
        w.stop()


def test_warm_reach_reads_what_the_engine_and_the_warm_write(tiny, monkeypatch, tmp_path, capsys):
    """progstate.warm_reach against real files: a bucket counts once each slot holds its runs (the
    engine's own, and warm35x2's by its keys); with wait entries on the board's path the sample
    wants those too; the startup's lines say it, and a failure there stops nothing."""
    import sys
    from pathlib import Path

    from opentpu import isa as I
    from opentpu import progcache as PC
    from opentpu import waits as WC
    from opentpu.llm import batch35 as B35
    from opentpu.llm import qwen3 as Q
    from opentpu.serve import progstate as PS
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "boards" / "bcu1525" / "chat35"))
    import warm35x2 as X
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path))
    monkeypatch.setattr(PC, "FREE_FLOOR", 0)
    monkeypatch.setattr(PC, "_mem", {})
    W, spec = tiny
    cfg = _cfg(monkeypatch)
    cap = 3 * 256                               # (three buckets)
    eng = Engine(spec, W, cap=cap, cfg=cfg, batch=2, resident=True, layer_major=2,
                 batch_area=True)
    B, K = eng.block, spec.conv_k
    assert B == 256
    assert eng.prog_cache and PS.warm_reach(eng)["tokens"] == 0
    p = _prompts(31, [K + 2])[0]                # (the rows before conv_k - 1, then 2 + 1 rows)
    B35.prefill_slot(eng, p, 0, logits=False)
    assert PS.warm_reach(eng)["buckets"] == 0   # (slot 1's runs are not there)
    B35.prefill_slot(eng, p, 1, logits=False)
    r = PS.warm_reach(eng)
    assert r["buckets"] == 1 and r["tokens"] == B and r["slots"] == 2, r
    assert r["of_tokens"] == cap and r["of_buckets"] == 3, r

    def warm(b):
        out = []
        for s in range(2):
            for key in X.layer_keys(spec, b, 2):
                what = Q.layer_what(key, B, slot=s)
                progs, _ = PC.get(eng.layout, what,
                                  lambda key=key, s=s: Q.layer_programs(eng.image, key, B, slot=s))
                out.append(progs)
        return out
    warm(2)
    assert PS.warm_reach(eng)["buckets"] == 2
    PS.startup(eng, "t")
    out = capsys.readouterr().out
    assert f"through {2 * B:,} of {cap:,} tokens (bucket 2 of 3" in out and "past that" in out, out
    assert "program saves on" in out, out
    # the board's path: the sample wants the programs' wait entries too
    monkeypatch.setattr(eng.backend, "waitc", True, raising=False)
    assert PS.warm_reach(eng)["buckets"] == 0
    for b in (1, 2):
        for key in PS._run_keys(eng, b):
            for s in range(2):
                progs, _ = PC.get(eng.layout, Q.layer_what(key, B, slot=s),
                                  lambda: pytest.fail("the file is there"))
                for q in progs:
                    WC.entries(I.assemble(q), cfg.D, cfg.S, prog=q, aw=cfg.AW, win=cfg.WC_WIN)
    r = PS.warm_reach(eng)
    assert r["buckets"] == 2 and r["entries_sampled"] >= 1, r
    # under the floor the line says so; a failure is a line
    monkeypatch.setattr(PC, "FREE_FLOOR", 1 << 62)
    PS.startup(eng, "t")
    assert "program saves OFF" in capsys.readouterr().out
    monkeypatch.setattr(PS, "warm_reach", lambda e: 1 / 0)
    assert PS.startup(eng, "t") is None
    assert "could not look at the program cache: ZeroDivisionError" in capsys.readouterr().out


def test_warm_samplers_reads_what_the_engine_compiles(tiny, monkeypatch, tmp_path, capsys):
    """progstate.warm_samplers against real files: a sampler counts once the engine's programs for
    the context's first and last bucket are on disk; the startup's line names it and warm_reach's
    result carries it for /v1/stats."""
    from opentpu import progcache as PC
    from opentpu.llm import generate as G
    from opentpu.serve import progstate as PS
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path))
    monkeypatch.setattr(PC, "FREE_FLOOR", 0)
    monkeypatch.setattr(PC, "_mem", {})
    W, spec = tiny
    eng = Engine(spec, W, cap=2 * 256, cfg=_cfg(monkeypatch), resident=True)
    monkeypatch.setattr(type(eng), "can_generate", True)
    samp = G.Sampling(0.1, 50, 1.0, 1.05)
    assert samp.key == (56, True) and PS.warm_samplers(eng) == []
    eng._generate_prog(1, samp)
    assert PS.warm_samplers(eng) == []          # (the last bucket's are not there)
    eng._generate_prog(2, samp)
    assert PS.warm_samplers(eng) == [(56, True)]
    for b in (1, 2):
        eng._generate_prog(b)
    assert PS.warm_samplers(eng) == [None, (56, True)]
    assert PS.sampler_name(None) == "greedy"
    assert PS.sampler_name((8, False)) == "top_k 1-8 without repetition_penalty"
    monkeypatch.setattr(PS, "warm_reach", lambda e: {"tokens": 512})
    monkeypatch.setattr(PS, "warm_line", lambda w, label: None)
    r = PS.startup(eng, "t")
    out = capsys.readouterr().out
    assert "samplers on disk: greedy, top_k 49-56 with repetition_penalty; a request" in out, out
    assert r == {"tokens": 512, "samplers": ["greedy", "top_k 49-56 with repetition_penalty"]}, r
    monkeypatch.setattr(type(eng), "can_generate", False)
    assert PS.warm_samplers(eng) is None and PS.sampler_line(None, "t") is None


def test_a_step_the_link_fails_resumes_on_two_cards(monkeypatch):
    """x2's two cards (CardsTransport over IsaCards): a batched step whose run fails on the link
    (cards.LinkError in its completion check, after its writes) while two requests decode: the
    link reset once (CardsTransport.recover), both requests resumed from their prompts and their
    output so far (BatchWorker._link_recovered: the slots' contexts fed again by layer runs), no
    error; each request's tokens a run's without the failure."""
    from opentpu.host.board import BoardBackend
    from opentpu.host.cards import LinkError
    from test_batch35 import _cards, _x2
    W, spec = _tiny(H=512, f=512, kinds=("linear", "attn"))
    p, q = _prompts(9, [10, 7])
    kw = dict(cap=CAP, batch=2, resident=True, layer_major=2, batch_area=True)
    cfg = _x2(monkeypatch)
    wait = BoardBackend.wait

    def run(fail):
        eng = Engine(spec, W, cfg=cfg, backend=_cards(cfg), **kw)
        w = BW35.BatchWorker35(eng, _Tok(), prefill_budget=4)
        t, fired, recovers = eng.backend.board.t, [], []
        recover = t.recover

        def failing(self, *a, **k):
            st = wait(self, *a, **k)
            if fail and not fired and w.stats["steps"] >= 2:
                fired.append(1)
                raise LinkError("a test's failed completion check")
            return st
        monkeypatch.setattr(BoardBackend, "wait", failing)
        monkeypatch.setattr(t, "recover", lambda *a: (recovers.append(1), recover(*a))[1])
        w.start()
        try:
            with w.cv:
                rs = [w.submit(SC.Request(x, max_new=8, stop_ids=frozenset())) for x in (p, q)]
            for r in rs:
                r.wait(900)
            assert all(r.error is None and r.finish == "length" for r in rs), [r.error for r in rs]
            return [r.out for r in rs], recovers, dict(w.stats)
        finally:
            w.stop()
    want, _, _ = run(False)
    got, recovers, st = run(True)
    assert got == want and recovers == [1], (got, want, recovers)
    assert st.get("link_recoveries") == 1 and st["errors"] == 0, st


def test_a_turn_the_link_fails_resumes_on_two_cards(monkeypatch):
    """x2's two cards, one sequence (scheduler.Worker: otpu-serve35x2 without --slots): two
    requests taking turns (quantum 2: each one's context saved to the store as the other runs),
    a decode run that fails on the link (cards.LinkError in its completion check) in one's turn:
    the link reset once, that request resumed from its prompt and its output so far
    (Worker._link_recovered), the other from the store, no error; each request's tokens a run's
    without the failure."""
    from opentpu.host.board import BoardBackend
    from opentpu.host.cards import LinkError
    from test_batch35 import _cards, _x2
    W, spec = _tiny(H=512, f=512, kinds=("linear", "attn"))
    p, q = _prompts(9, [10, 7])
    cfg = _x2(monkeypatch)
    wait = BoardBackend.wait

    def run(fail):
        eng = Engine(spec, W, cap=CAP, cfg=cfg, resident=True, layer_major=2, backend=_cards(cfg))
        w = SC.Worker(eng, _Tok(), quantum=2)
        t, fired, recovers = eng.backend.board.t, [], []
        recover = t.recover

        def failing(self, *a, **k):
            st = wait(self, *a, **k)
            if fail and not fired and w.stats["generated"] >= 3:
                fired.append(1)
                raise LinkError("a test's failed completion check")
            return st
        monkeypatch.setattr(BoardBackend, "wait", failing)
        monkeypatch.setattr(t, "recover", lambda *a: (recovers.append(1), recover(*a))[1])
        w.start()
        try:
            with w.cv:
                rs = [w.submit(SC.Request(x, max_new=8, stop_ids=frozenset())) for x in (p, q)]
            for r in rs:
                r.wait(900)
            assert all(r.error is None and r.finish == "length" for r in rs), [r.error for r in rs]
            return [r.out for r in rs], recovers, fired, dict(w.stats)
        finally:
            w.stop()
    want, _, _, st0 = run(False)
    got, recovers, fired, st = run(True)
    assert fired == [1] and st0["saves"] >= 1, (fired, st0)
    assert got == want and recovers == [1], (got, want, recovers)
    assert st.get("link_recoveries") == 1 and st["errors"] == 0, st
    assert st["prompt_tokens"] == st0["prompt_tokens"], (st, st0)


def test_raw_images_take_the_batch_area(tmp_path, monkeypatch):
    """chat35.raw_images' maps of the raw image are copy-on-write: --slots N's batch area
    (batch.build: penalty factors 1, -inf logits) is written into them, the raw file left as it
    is, and each slice's [base, end) joins RUNS' runs (the area lies in the file's holes: the
    scrub would leave zeros there, only the runs are written)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "boards" / "bcu1525" / "chat35"))
    import chat35 as C35
    from opentpu.llm import batch as BA
    W, spec = _tiny()[:2]
    cfg = _cfg(monkeypatch)
    img = spec.image(cfg, CAP, 2, 2, lookup=True)
    bat = BA.alloc(img, cfg, 2, 1 << 20)
    raw = tmp_path / "x.raw"
    n = bat["end"]
    with open(raw, "wb") as f:              # two slices, data in each one's first 4 KiB, holes after
        for b in (0, n):
            f.seek(b)
            f.write(b"\1" * 4096)
        f.truncate(2 * n)
    monkeypatch.setattr(BA, "build", BA.build)          # (batch_runs wraps it: undone after)
    monkeypatch.setitem(C35.RUNS, "runs", [C35.data_runs(raw, b, n) for b in (0, n)])
    maps = [np.memmap(raw, np.uint8, "c", offset=b, shape=(n,)) for b in (0, n)]
    C35.batch_runs()
    C35.batch_runs()                        # (once)
    BA.build(maps, bat, spec, cfg)
    ones = np.ones(4, np.float32).view(np.uint8)
    for m, r in zip(maps, C35.RUNS["runs"]):
        assert r[-1] == (bat["base"], bat["end"]) and (bat["base"], bat["end"]) not in r[:-1]
        assert (m[bat["pa"][1]:bat["pa"][1] + len(ones)] == ones).all()
    on_disk = np.fromfile(raw, np.uint8)
    assert on_disk.sum() == 2 * 4096 and len(on_disk) == 2 * n
