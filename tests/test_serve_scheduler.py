"""The server's worker (opentpu/serve/scheduler.py) on the ISA simulator with tiny models:
requests submitted together run in turns, each context switched out to the KV store and back,
and every request's greedy tokens equal its own run on a fresh engine, bit for bit (the card's
decode loop and the host's steps alike; Qwen3, LFM2 and Qwen3.5: KV caches only, convolution
rings, DeltaNet states); a prompt that continues a saved context feeds only the rest; stop
strings, max_new and cancels end requests; the store on disk serves a second worker."""
import threading
import time

import numpy as np
import pytest

from opentpu.llm.qwen3 import Engine
from opentpu.serve import scheduler as SC
from opentpu.serve import store as ST

from test_serve_kvio import _model

torch = pytest.importorskip("torch")

CAP = 512


class _Tok:
    """One character per token id."""

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(0x100 + int(i)) for i in ids)


def _engine(spec, W, cfg, card=True):
    return Engine(spec, W, cap=CAP, cfg=cfg, resident=card, prompt_runs=card)


def _ref(spec, W, cfg, prompt, n, stop):
    """Greedy tokens of a fresh engine (stop ids end it, not included)."""
    e = _engine(spec, W, cfg, card=False)
    lg, out = e.prefill(prompt), []
    while len(out) < n:
        t = int(np.argmax(lg))
        if t in stop:
            break
        out.append(t)
        if len(out) == n:
            break
        lg = e.step(t)
    return out


def _prompts(seed, sizes, base=None):
    r = np.random.default_rng(seed)
    return [(list(base) if base else []) + [int(x) for x in r.integers(0, 1000, n)]
            for n in sizes]


@pytest.mark.parametrize("case, card", [("qwen3", True), ("qwen3", False), ("lfm2", True),
                                        ("qwen35", True)])
def test_requests_in_turns_are_their_own_runs(case, card):
    """Four requests at once, quantum 3: each runs in several turns, the others' contexts in
    between; their tokens are their own fresh runs'. Then a fifth whose prompt continues the
    first's: it feeds only what is new."""
    W, spec, cfg = _model(case)
    stop = frozenset({int(spec.eos[0])} if spec.eos else ())
    eng = _engine(spec, W, cfg, card)
    w = SC.Worker(eng, _Tok(), quantum=3, max_running=4).start()
    try:
        ps = _prompts(1, [21, 34, 9, 27])
        reqs = [w.submit(SC.Request(p, max_new=10, stop_ids=stop)) for p in ps]
        for r in reqs:
            r.wait(timeout=600)
        for p, r in zip(ps, reqs):
            assert r.finish in ("length", "stop") and r.error is None, r.error
            assert r.out == _ref(spec, W, cfg, p, 10, stop), r.rid
        assert sum(r.swaps for r in reqs) >= 3 and w.stats["restores"] >= 3
        # the first request's conversation, continued
        nxt = ps[0] + reqs[0].out + _prompts(2, [6])[0]
        r = w.submit(SC.Request(nxt, max_new=8, stop_ids=stop)).wait(timeout=600)
        assert r.out == _ref(spec, W, cfg, nxt, 8, stop)
        if not eng.stateful:
            assert r.cached >= len(ps[0]) + len(reqs[0].out) - 1
        else:                               # (its end saved: the prompt and the reply less one)
            assert r.cached == len(ps[0]) + len(reqs[0].out) - 1 or r.cached == 0
        assert r.prefill_tokens == len(nxt) - r.cached
    finally:
        w.stop()


@pytest.mark.parametrize("title_first", [False, True])
def test_a_title_beside_the_main_prompt_feeds_nothing_again(title_first):
    """A coding client's short title request beside its main prompt, one slot (LFM2: convolution
    rings, one sequence), in either order: each feeds its prompt once, each has its fresh run's
    tokens, and the main conversation's next turns feed only what is new: a context switched out
    comes back from the store."""
    W, spec, cfg = _model("lfm2")
    stop = frozenset()
    eng = _engine(spec, W, cfg)
    w = SC.Worker(eng, _Tok(), quantum=3, max_running=4).start()
    try:
        main, title = _prompts(41, [300, 19])
        order = [title, main] if title_first else [main, title]
        reqs = [w.submit(SC.Request(p, max_new=12, stop_ids=stop, at=len(p) - 2)) for p in order]
        for r in reqs:
            r.wait(timeout=600)
        rt, rm = reqs if title_first else reqs[::-1]
        assert rm.error is None and rt.error is None
        assert rm.out == _ref(spec, W, cfg, main, 12, stop)
        assert rt.out == _ref(spec, W, cfg, title, 12, stop)
        assert rm.prefill_tokens == len(main) and rt.prefill_tokens == len(title)
        assert w.stats["prefill_tokens"] == len(main) + len(title)
        assert w.stats["restores"] >= 1             # (one was switched out and came back)
        # a client's next turn re-renders the reply (not the device's tokens): from the reply's
        # start (Server.reply_start, before the generation prompt's 2 tokens here), whichever
        # context is on the device now
        nxt = main[:-2] + _prompts(43, [9])[0]
        r = w.submit(SC.Request(nxt, max_new=5, stop_ids=stop, at=len(nxt) - 2)).wait(timeout=600)
        assert r.out == _ref(spec, W, cfg, nxt, 5, stop)
        assert r.cached == len(main) - 2 and r.prefill_tokens == 9
        # the reply as the device gave it: from its end
        nxt = nxt + r.out + _prompts(42, [6])[0]
        r = w.submit(SC.Request(nxt, max_new=4, stop_ids=stop)).wait(timeout=600)
        assert r.out == _ref(spec, W, cfg, nxt, 4, stop)
        assert r.cached == len(nxt) - 7 and r.prefill_tokens == 7
    finally:
        w.stop()


def test_shared_prefix_from_the_store_and_the_device():
    """Requests sharing a 260-token system prompt: the second takes it from the device (the
    first's context rolled back), the third from the store (another context ran between);
    each feeds only its own part, with its fresh run's tokens."""
    W, spec, cfg = _model("qwen3")
    stop = frozenset()
    w = SC.Worker(_engine(spec, W, cfg), _Tok(), quantum=4).start()
    try:
        sysp = _prompts(5, [260])[0]
        a, b = _prompts(6, [12, 15], base=sysp)
        other = _prompts(7, [30])[0]
        ra = w.submit(SC.Request(a, max_new=5, stop_ids=stop)).wait(600)
        rb = w.submit(SC.Request(b, max_new=5, stop_ids=stop)).wait(600)
        assert rb.cached == 260 and w.stats["device_reuses"] >= 1
        w.submit(SC.Request(other, max_new=3, stop_ids=stop)).wait(600)
        c = sysp + _prompts(8, [10])[0]
        rc = w.submit(SC.Request(c, max_new=5, stop_ids=stop)).wait(600)
        assert rc.cached >= 260 and w.stats["restores"] >= 1
        for p, r in ((a, ra), (b, rb), (c, rc)):
            assert r.out == _ref(spec, W, cfg, p, 5, stop)
        u = w.usage()
        assert u["prefix_cache"]["shared_pages"] >= 1 and u["kv_store"]["ram_bytes"] > 0
    finally:
        w.stop()


def test_stop_strings_max_new_and_cancel():
    W, spec, cfg = _model("qwen3")
    w = SC.Worker(_engine(spec, W, cfg), _Tok(), quantum=4).start()
    try:
        p = _prompts(9, [15])[0]
        want = _ref(spec, W, cfg, p, 12, frozenset())
        text = _Tok().decode(want)
        stop_s = text[5:7]
        r = w.submit(SC.Request(p, max_new=12, stop_ids=frozenset(), stop_strs=(stop_s,)))
        got = "".join(r.stream(timeout=600))
        assert got == text[:text.find(stop_s)] and r.finish == "stop" and stop_s not in got
        r = w.submit(SC.Request(p, max_new=4, stop_ids=frozenset())).wait(600)
        assert r.finish == "length" and r.out == want[:4] and r.text == text[:4]
        # a cancel: before it runs, and while it runs
        gate = threading.Event()
        r1 = SC.Request(_prompts(10, [20])[0], max_new=200, stop_ids=frozenset())
        r1.cancel()
        w.submit(r1).wait(600)
        assert r1.finish == "abort"
        r2 = w.submit(SC.Request(_prompts(11, [20])[0], max_new=300, stop_ids=frozenset()))
        it = r2.stream(timeout=600)
        next(it)
        r2.cancel()
        r2.wait(600)
        assert r2.finish == "abort" and len(r2.out) < 300
        gate.set()
    finally:
        w.stop()


def test_store_on_disk_serves_another_worker(tmp_path):
    """A worker whose store spills to disk; a second worker on the same directory continues the
    first's saved context (its prompt's prefix restored from the files)."""
    W, spec, cfg = _model("qwen3")
    mk = ST.model_key("tiny-qwen3")
    stop = frozenset()
    s1 = ST.BlobStore(64 << 10, 1 << 30, tmp_path)      # 64 KiB of RAM: spills
    w = SC.Worker(_engine(spec, W, cfg), _Tok(), store=s1, model_key=mk).start()
    p = _prompts(12, [300])[0]
    try:
        r = w.submit(SC.Request(p, max_new=4, stop_ids=stop)).wait(600)
        w.submit(SC.Request(_prompts(13, [20])[0], max_new=2, stop_ids=stop)).wait(600)
    finally:
        w.stop()
    s1.flush()
    assert s1.usage()["disk_blobs"] >= 2
    assert s1.usage()["writer_errors"] == 0 and s1.usage()["writer_alive"] is True
    s2 = ST.BlobStore(1 << 30, 1 << 30, tmp_path)
    w2 = SC.Worker(_engine(spec, W, cfg), _Tok(), store=s2, model_key=mk).start()
    try:
        q = p + r.out + _prompts(14, [5])[0]
        r2 = w2.submit(SC.Request(q, max_new=4, stop_ids=stop)).wait(600)
        assert r2.cached >= 300 and r2.out == _ref(spec, W, cfg, q, 4, stop)
        assert s2.usage()["disk_hits"] >= 1
    finally:
        w2.stop()


def test_requests_that_do_not_fit_are_refused():
    W, spec, cfg = _model("qwen3")
    w = SC.Worker(_engine(spec, W, cfg), _Tok())
    with pytest.raises(ValueError):
        w.submit(SC.Request(list(range(CAP)), max_new=1))
    with pytest.raises(ValueError):
        w.submit(SC.Request([], max_new=1))
    r = SC.Request(list(range(CAP - 3)), max_new=50)
    w.submit(r)
    assert r.max_new == 3


def _until(f, timeout=600.0):
    t = time.monotonic() + timeout
    while not f():
        assert time.monotonic() < t, "timed out"
        time.sleep(0.01)


def test_a_chats_reply_start_stays_on_the_device_then_goes_to_the_store():
    """Qwen3.5 (DeltaNet states) with a snapshot: a request's reply start (at) is a snapshot on
    the device, not a store save on its path (nor one a token short of its prompt). Its next
    turn, waiting meanwhile, rolls the device back there (no store traffic); once the queue is
    empty the turn's reply start goes to the store (the device back to it, saved), so a later
    turn after another conversation restores it. Every request's tokens are its fresh run's;
    each request's phases say where its time went."""
    W, spec, cfg = _model("qwen35")
    stop = frozenset()
    eng = Engine(spec, W, cap=CAP, cfg=cfg, resident=True, prompt_runs=True, snapshots=1)
    w = SC.Worker(eng, _Tok(), quantum=64, max_running=1).start()
    try:
        a1 = _prompts(20, [30])[0]
        a2 = a1[:26] + _prompts(21, [12])[0]
        r1 = w.submit(SC.Request(a1, max_new=6, stop_ids=stop, at=26))
        r2 = w.submit(SC.Request(a2, max_new=5, stop_ids=stop, at=len(a2) - 3))
        r1.wait(600)
        r2.wait(600)
        assert r1.out == _ref(spec, W, cfg, a1, 6, stop)
        assert r2.out == _ref(spec, W, cfg, a2, 5, stop)
        assert r1.phases.get("snapshot", 0) > 0 and "save" not in r1.phases
        assert r2.cached == 26 and r2.phases.get("rollback", 0) > 0
        assert "save" not in r2.phases and "restore" not in r2.phases
        # idle: r2's reply start into the store (rolled back to it), nothing else; none a
        # token short of a prompt
        _until(lambda: w.stats["idle_saves"] == 1)
        assert w.stats["saves"] == 1 and w.stats["rollbacks"] == 2
        assert w.stats["snapshots"] == 2 and w.stats["device_reuses"] == 1
        assert w.dev == a2[:len(a2) - 3] and not w.dirty and w.snap_at is None
        assert w.cache.match(a2, limit=len(a2) - 1).k == len(a2) - 3
        b = _prompts(22, [20])[0]
        rb = w.submit(SC.Request(b, max_new=3, stop_ids=stop, at=17)).wait(600)
        assert "save" not in rb.phases and rb.out == _ref(spec, W, cfg, b, 3, stop)
        a3 = a2[:len(a2) - 3] + _prompts(23, [9])[0]
        r3 = w.submit(SC.Request(a3, max_new=4, stop_ids=stop, at=len(a3) - 2)).wait(600)
        assert r3.cached == len(a2) - 3 and r3.phases.get("restore", 0) > 0
        assert r3.out == _ref(spec, W, cfg, a3, 4, stop)
        # its restore by part (the ISA simulator: direct, no copy), the bytes it wrote
        ph, L = r3.phases, w.io.layout
        parts = ("store", "host", "prog", "link", "load", "run", "adopt")
        assert abs(sum(ph[f"restore.{k}"] for k in parts) - ph["restore"]) < 1e-6
        assert ph["restore.link"] > 0 and ph["restore.load"] == 0
        assert "restore.gc" in ph and ph["restore.telemetry"] == 0      # (no card sampler)
        assert r3.io["restore"]["link_bytes"] == L.page_bytes(len(a2) - 3) + L.state_bytes
        assert r3.io["restore"]["copies"] == 0
        last = w.usage()["last_io"]
        assert last["restore"]["s"] == round(ph["restore"], 4) and last["save"]["link_bytes"] > 0
        # a flush: the store and the device's context both forgotten
        _until(lambda: not w.busy() and w.snap_at is None and not w.dirty)
        w.flush()
        r4 = w.submit(SC.Request(a3, max_new=4, stop_ids=stop, at=len(a3) - 2)).wait(600)
        assert r4.cached == 0 and r4.out == r3.out
    finally:
        w.stop()


def test_a_tool_calls_reply_stays_on_the_device():
    """A reply that holds its request's keep_reply (the API's "<tool_call>" with tools): the
    idle save keeps it on the device (no rollback), its reply start saved from the snapshot,
    then the whole context; the agent's next request (the reply, then a tool's result) feeds
    only the result; a request that parts at the reply start restores it there. A reply
    without it is rolled back as a chat's."""
    W, spec, cfg = _model("qwen35")
    stop = frozenset()
    eng = Engine(spec, W, cap=CAP, cfg=cfg, resident=True, prompt_runs=True, snapshots=1)
    w = SC.Worker(eng, _Tok(), quantum=64, max_running=1, prefix_block=0).start()
    try:
        a = _prompts(40, [30])[0]
        ref = _ref(spec, W, cfg, a, 6, stop)
        call = _Tok().decode(ref[2:4])                  # (the reply's text holds it)
        r1 = w.submit(SC.Request(a, max_new=6, stop_ids=stop, at=len(a) - 2,
                                 keep_reply=call)).wait(600)
        assert r1.out == ref
        _until(lambda: w.stats["idle_saves"] == 2)
        n1 = len(w.dev)
        assert n1 >= len(a) + len(ref) - 1 and w.dev == (a + ref)[:n1]
        assert w.stats["rollbacks"] == 0 and not w.dirty and w.snap_at is None
        assert sorted(e.n for e in w.cache.entries.values()) == [len(a) - 2, n1]
        q = a + ref + _prompts(41, [8])[0]
        r2 = w.submit(SC.Request(q, max_new=4, stop_ids=stop, at=len(a) - 2)).wait(600)
        assert r2.cached == n1 and r2.out == _ref(spec, W, cfg, q, 4, stop)
        assert "restore" not in r2.phases and w.stats["device_reuses"] == 1
        _until(lambda: w.stats["idle_saves"] == 3)      # (no snapshot: its context, saved)
        a3 = a[:len(a) - 2] + _prompts(42, [7])[0]
        r3 = w.submit(SC.Request(a3, max_new=4, stop_ids=stop, at=len(a3) - 2,
                                 keep_reply="\x00")).wait(600)
        assert r3.cached == len(a) - 2 and r3.phases.get("restore", 0) > 0
        assert r3.out == _ref(spec, W, cfg, a3, 4, stop)
        _until(lambda: w.stats["rollbacks"] == 1)       # (not in its reply: rolled back)
        assert w.dev == a3[:len(a3) - 2]
    finally:
        w.stop()


def test_every_reply_kept_with_the_thinking_kept():
    """keep_reply "" (the API's with --preserve-thinking): the reply start at the prompt's last
    fed position, a snapshot; the idle save keeps the reply on the device (no rollback): the
    next turn (the prompt, the reply, new tokens) feeds only the new ones; a turn whose reply
    part differs (its text tokenized otherwise) restores the reply start. A switch to another
    conversation before the idle keeps the reply too: in the store, the next turn restores
    all of it."""
    W, spec, cfg = _model("qwen35")
    stop = frozenset()
    eng = Engine(spec, W, cap=CAP, cfg=cfg, resident=True, prompt_runs=True, snapshots=1)
    w = SC.Worker(eng, _Tok(), quantum=64, max_running=1, prefix_block=0).start()
    try:
        a = _prompts(50, [30])[0]
        ref = _ref(spec, W, cfg, a, 6, stop)
        r1 = w.submit(SC.Request(a, max_new=6, stop_ids=stop, at=len(a) - 1,
                                 keep_reply="")).wait(600)
        assert r1.out == ref and r1.phases.get("snapshot", 0) > 0
        _until(lambda: w.stats["idle_saves"] == 2)
        n1 = len(w.dev)
        assert w.stats["rollbacks"] == 0 and w.dev == (a + ref)[:n1]
        assert sorted(e.n for e in w.cache.entries.values()) == [len(a) - 1, n1]
        q = a + ref + _prompts(51, [8])[0]
        r2 = w.submit(SC.Request(q, max_new=4, stop_ids=stop, at=len(q) - 1,
                                 keep_reply="")).wait(600)
        assert r2.cached == n1 and r2.out == _ref(spec, W, cfg, q, 4, stop)
        _until(lambda: w.stats["idle_saves"] == 4)
        q2 = a + [(ref[0] + 1) % 1000] + _prompts(52, [6])[0]
        r3 = w.submit(SC.Request(q2, max_new=3, stop_ids=stop)).wait(600)
        assert r3.cached == len(a) - 1 and r3.out == _ref(spec, W, cfg, q2, 3, stop)
        # a switch before the idle: the reply start and the whole context saved
        b, c = _prompts(53, [24, 20])
        rb = w.submit(SC.Request(b, max_new=5, stop_ids=stop, at=len(b) - 1, keep_reply=""))
        rc = w.submit(SC.Request(c, max_new=2, stop_ids=stop))
        rb.wait(600)
        rc.wait(600)
        assert rc.phases.get("save", 0) > 0 and w.stats["rollbacks"] == 0
        nb = len(b) + len(rb.out) - 1
        q3 = b + rb.out + _prompts(54, [5])[0]
        r4 = w.submit(SC.Request(q3, max_new=3, stop_ids=stop)).wait(600)
        assert r4.cached == nb and r4.out == _ref(spec, W, cfg, q3, 3, stop)
    finally:
        w.stop()


def test_a_chats_next_turn_pre_empts_the_idle_save():
    """A chat's next turn sent the moment its reply ends (an idle save of the reply start under
    way, its device reads slowed to 0.1 s each) pre-empts the save: it gives up between two
    reads (nothing stored), the turn goes on from the device (cached = the reply start, no
    save, no restore) with no wait in the queue, and its tokens are its fresh run's. Another
    conversation sent during the next idle save waits for it (its phases' idle_save) and does
    not pre-empt it."""
    W, spec, cfg = _model("qwen35")
    stop = frozenset()
    eng = Engine(spec, W, cap=CAP, cfg=cfg, resident=True, prompt_runs=True, snapshots=1)
    w = SC.Worker(eng, _Tok(), quantum=64, max_running=1)
    check = w.io._check
    reads = []

    def slow():                         # (a card's device reads: one every 0.1 s)
        reads.append(w.io.abort is not None)
        time.sleep(0.1)
        check()
    w.io._check = slow
    nxt = {}

    def ended(r):                       # (the next request once the idle save is under way)
        if r.rid in nxt:
            q = nxt.pop(r.rid)

            def send():
                _until(lambda: w._idling is not None, 60)
                w.submit(q)
            threading.Thread(target=send, daemon=True).start()
    w.observers.append(ended)
    w.start()
    try:
        a1 = _prompts(30, [30])[0]
        a2 = a1[:26] + _prompts(31, [12])[0]
        b = _prompts(32, [20])[0]
        r1 = SC.Request(a1, max_new=5, stop_ids=stop, at=26)
        r2 = SC.Request(a2, max_new=4, stop_ids=stop, at=len(a2) - 3)
        rb = SC.Request(b, max_new=3, stop_ids=stop, at=17)
        r1.rid, r2.rid, rb.rid = "r1", "r2", "rb"
        nxt.update(r1=r2, r2=rb)        # (r2 once r1 ends, rb once r2 does)
        w.submit(r1)
        for r in (r1, r2, rb):
            _until(lambda: r.finish is not None, 600)
            assert r.error is None, r.error
        assert r1.out == _ref(spec, W, cfg, a1, 5, stop)
        assert r2.out == _ref(spec, W, cfg, a2, 4, stop)
        assert rb.out == _ref(spec, W, cfg, b, 3, stop)
        # r2 pre-empted r1's idle save: no wait, the device's context, nothing saved for it
        assert r2.t_start - r2.t_submit < 0.25, r2.t_start - r2.t_submit
        assert r2.cached == 26 and "save" not in r2.phases and "restore" not in r2.phases
        assert r2.phases.get("idle_save", 0) < 0.25
        assert w.stats["idle_preempts"] == 1
        # rb waited for r2's idle save (not pre-empted: rb needs it, its switch would make it)
        assert rb.phases.get("idle_save", 0) > 0.05 and "save" not in rb.phases
        _until(lambda: not w.busy() and w.snap_at is None and not w.dirty)
        assert w.stats["idle_preempts"] == 1 and w.stats["idle_saves"] == 2
        assert sorted(e.n for e in w.cache.entries.values()) == [17, len(a2) - 3]
        assert any(reads) and w.io.abort is None
    finally:
        w.stop()
