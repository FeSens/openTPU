"""Prefixes kept by the batched Qwen3.5 worker (opentpu/serve/batching35.py): a model with state
continues only at a saved context's whole length, so the worker's feed runs stop at the prompt's
cuts (its marks, the multiples of prefix_block) and each cut goes into the store through its
slot's io. Another conversation sharing the prefix restores it and feeds only the rest. All at x2
(S = 4: both BCU1525s as one machine, CardsTransport over IsaCards), the tiny 16-head untied
Qwen3.5-MoE at x2's width: a context restored from a prefix and fed to the end is the same bytes
(logits, KV pages, each DeltaNet state and convolution window) as the prompt fed from 0, also
for a prompt that parts right after the cut; a session restores while another decodes; a prefix
outlives a server; a client leaving, or a read failing, mid-feed leaves the store consistent."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from opentpu.llm import batch35 as B35
from opentpu.llm.qwen3 import Engine
from opentpu.serve import batching35 as BW35
from opentpu.serve import scheduler as SC
from opentpu.serve import store as ST

from test_serve_lanes35 import _engine, _state, tiny_x2, x2  # noqa: F401
from test_serve_scheduler import _Tok, _prompts

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

CAP = 1024
STOP = frozenset()
MARK = 280              # (the system prompt's end: past a whole page of 256, a partial one after)


def _worker(eng, **kw):
    return BW35.BatchWorker35(eng, _Tok(), **{"prefill_budget": 64, "prefix_block": 128,
                                              "prefix_skip": 16, **kw})


class _Ref:
    """Greedy tokens of a fresh single-sequence engine on the same machine."""

    def __init__(self, x2):
        W, spec, cfg = x2
        self.e = Engine(spec, W, cap=CAP, cfg=cfg, resident=True, layer_major=2)

    def __call__(self, prompt, n: int) -> list:
        self.e.reset()
        lg, out = self.e.prefill(prompt), []
        while True:
            out.append(int(np.argmax(lg)))
            if len(out) == n:
                return out
            lg = self.e.step(out[-1])


def _same(w, s0: int, s1: int, n: int, lg0, lg1, whole_only: bool = False) -> None:
    """Slots s0 and s1 hold one context of n tokens: the logits after it, its KV pages (whole_only:
    those; a partial page's rows past n are what the slot held before), its states and windows."""
    assert np.array_equal(lg0, lg1)
    L = w.ios[s0].layout
    pages = [(i, m) for i, m in L.split(n) if m == L.P or not whole_only]
    for (i, m), a, b in zip(pages, w.ios[s0].read_pages(pages), w.ios[s1].read_pages(pages)):
        assert a.any() and np.array_equal(a, b), (s0, s1, i, m)
    a, b = _state(w.eng, s0), _state(w.eng, s1)
    assert a.keys() == b.keys() and any(k[2] == "window" for k in a)
    for k in a:
        assert a[k].any() and np.array_equal(a[k], b[k]), (s0, s1, k)
    assert np.array_equal(w.ios[s0].read_state(), w.ios[s1].read_state())


def _restore(w, s: int, prompt, k: int, have: int = 0) -> None:
    m = w.cache.match(prompt, limit=len(prompt) - 1)
    assert m.k == k and m.entry.prefix
    assert w.cache.restore(w.ios[s], m, have=have) == k
    w.eng.adopt(k, seq=s)


def test_a_restored_prefix_fed_to_the_end_is_the_prompt_from_zero(x2):
    """The one that matters. A prompt fed from 0 with its prefix kept at cuts (in the middle of a
    page, at a page's end, just past one); each cut restored into another slot and fed to the end
    gives the same logits, pages, states and windows as the slot that fed all of it. Prompts that
    part right after a cut (at a page's end, and in a partial page), restored over a slot whose
    context shares the whole page (have), the same as the prompt fed from 0."""
    eng = _engine(x2, 4)
    w = _worker(eng)
    feed = lambda s, toks, logits=False: B35.prefill_slot(eng, toks, s, logits=logits)
    assert w.ios[0].layout.P == 256 and w.ios[3].slot == 3 and eng.cfg.CHASH
    a = _prompts(30, [300])[0]
    i = 0
    for c in (128, 256, MARK):
        feed(0, a[i:c])
        w.cache.save(w.ios[0], a[:c], prefix=True)
        i = c
    lg0 = feed(0, a[i:], True)
    for s, c in ((1, MARK), (2, 256), (3, 128)):
        _restore(w, s, a[:c + 1], c)
        _same(w, 0, s, len(a), lg0, feed(s, a[c:], True))
    # parting right after a cut, the first token after it another
    for c, s0, s1 in ((256, 0, 1), (MARK, 2, 3)):
        b = a[:c] + _prompts(31, [24])[0]
        assert b[c] != a[c]
        eng.adopt(0, seq=s0)
        lg0 = feed(s0, b, True)
        _restore(w, s1, b, c, have=c)
        _same(w, s0, s1, len(b), lg0, feed(s1, b[c:], True), whole_only=True)


def test_new_sessions_restore_the_shared_prefix(x2):
    """A's prompt kept at the blocks and its mark as it is fed; B (the same system prompt, another
    question) restores the mark and feeds its question; D parts right after a block and restores
    that block. Each one's tokens a fresh engine's, every run on a key the warm makes, and no
    step's program compiled once warm_steps' are live, in whichever slot a restore lands."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "boards" / "bcu1525" / "chat35"))
    import warm35x2 as X
    ref = _Ref(x2)
    eng = _engine(x2, 2)
    seen, run = set(), eng._layer_run
    eng._layer_run = lambda key: (seen.add(key), run(key))[1]
    w = _worker(eng)
    assert w.steps.slots and w.warm_steps() == 3
    w.start()
    try:
        system = _prompts(30, [MARK])[0]
        qa, qb, qd = _prompts(31, [12, 9, 10])
        a = system + qa
        ra = w.submit(SC.Request(a, max_new=5, stop_ids=STOP, marks=(MARK,))).wait(1800)
        assert ra.error is None and ra.out == ref(a, 5)
        assert sorted(e.n for e in w.cache.entries.values() if e.prefix) == [128, 256, MARK]
        assert w.stats["prefix_saves"] == 3 and ra.cached == 0
        assert ra.phases["prefix_save"] > 0 and w.stats["prefix_save_s"] > 0
        assert w.usage()["last_io"]["prefix_save"]["link_bytes"] > 0
        # (slot 0 again, holding A's context: the whole page they share stays)
        b = system + qb
        rb = w.submit(SC.Request(b, max_new=5, stop_ids=STOP, marks=(MARK,))).wait(1800)
        assert rb.error is None and rb.cached == MARK and rb.out == ref(b, 5)
        assert rb.prefill_tokens == len(qb) and rb.phases["restore"] > 0
        assert w.stats["restore_s"] > 0
        assert w.usage()["last_io"]["restore"]["link_bytes"] < eng_context_bytes(w, MARK)
        assert w.cache.usage()["prefix_hits"] == 1
        assert w.stats["prefix_saves"] == 3              # (B fed from the mark: nothing new)
        d = system[:128] + _prompts(32, [40])[0] + qd
        assert d[128] != system[128]
        rd = w.submit(SC.Request(d, max_new=5, stop_ids=STOP)).wait(1800)
        assert rd.error is None and rd.cached == 128 and rd.out == ref(d, 5)
        assert w.cache.usage()["prefix_hit_tokens"] == MARK + 128
        assert w.steps.timing["compiled_inline"] == 0, w.steps.timing
    finally:
        w.stop()
    want = {k for b in range(1, CAP // eng.block + 1) for k in X.layer_keys(x2[1], b, 2)}
    assert seen and seen <= want, seen - want


def eng_context_bytes(w, n: int) -> int:
    """A context's bytes of n tokens: its pages and state."""
    L = w.ios[0].layout
    return sum(L.page_bytes(m) for _, m in L.split(n)) + L.state_bytes


def test_a_save_and_a_restore_beside_decoding_slots(x2, monkeypatch):
    """r1 decodes in slot 0 all along (steps: the ISA cards' decode loop runs to its end at once).
    r2 feeds in slot 1 with its cuts kept through slot 1's io, then decodes; r3 restores r2's
    prefix into slot 2, between their steps. Slot 0's state differs at every cut, so a save through
    the wrong slot's io shows in r3's tokens; slot 1's lanes are not slot 2's. All a fresh
    engine's."""
    monkeypatch.setattr(Engine, "can_generate", property(lambda self: False))
    ref = _Ref(x2)
    eng = _engine(x2, 3)
    w = _worker(eng).start()

    def until(r, n):
        while len(r.out) < n and r.finish is None:
            time.sleep(0.05)
    try:
        p1 = _prompts(40, [20])[0]
        system = _prompts(30, [MARK])[0]
        q2, q3 = _prompts(31, [12, 9])
        r1 = w.submit(SC.Request(p1, max_new=800, stop_ids=STOP))
        until(r1, 2)
        r2 = w.submit(SC.Request(system + q2, max_new=800, stop_ids=STOP, marks=(MARK,)))
        until(r2, 1)
        assert r2.slot == 1 and r2.error is None and w.stats["prefix_saves"] == 3
        r3 = w.submit(SC.Request(system + q3, max_new=5, stop_ids=STOP, marks=(MARK,)))
        r3.wait(1800)
        assert r3.error is None and r3.slot == 2 and r3.cached == MARK
        assert r3.out == ref(system + q3, 5)
        assert r1.finish is None and r2.finish is None  # (decoding all along)
        for r in (r1, r2):
            r.cancelled.set()
        for r, p in ((r1, p1), (r2, system + q2)):
            r.wait(1800)
            assert r.finish == "abort" and r.error is None and r.out == ref(p, len(r.out))
    finally:
        w.stop()


def test_a_next_turn_restores_its_reply_start(x2):
    """A prompt's reply start (Request.at) is kept as it is fed, once, not as a prefix: the next
    turn parts from the reply there (another reply text than the tokens decoded) and restores it,
    from another slot than the one that saved it; its tokens a fresh engine's."""
    ref = _Ref(x2)
    eng = _engine(x2, 2)
    w = _worker(eng).start()
    try:
        p = _prompts(30, [40])[0]
        at = len(p) - 2
        r = w.submit(SC.Request(p, max_new=4, stop_ids=STOP, at=at)).wait(1800)
        assert r.error is None and r.out == ref(p, 4) and r.slot == 0
        e = [e for e in w.cache.entries.values() if e.n == at]
        assert len(e) == 1 and not e[0].prefix and w.stats["prefix_saves"] == 0
        # slot 0 taken (another prompt runs): the next turn in slot 1
        nxt = p[:at] + _prompts(31, [15])[0]
        with w.cv:
            other = w.submit(SC.Request(_prompts(41, [10])[0], max_new=3, stop_ids=STOP))
            rn = w.submit(SC.Request(nxt, max_new=4, stop_ids=STOP, at=len(nxt) - 2))
        other.wait(1800)
        rn.wait(1800)
        assert rn.error is None and rn.slot == 1 and rn.cached == at and rn.out == ref(nxt, 4)
    finally:
        w.stop()


def test_a_prefix_outlives_the_server(x2, tmp_path):
    """Prefix entries are on disk at once: a server killed (no shutdown's save of its RAM) leaves
    them, and the next server on the directory restores them from the disk onto a fresh device."""
    ref = _Ref(x2)
    system = _prompts(30, [MARK])[0]
    qa, qb = _prompts(31, [12, 9])
    store = ST.BlobStore(1 << 30, 1 << 30, tmp_path)
    w = _worker(_engine(x2, 2), store=store, model_key="m").start()
    a = system + qa
    ra = w.submit(SC.Request(a, max_new=3, stop_ids=STOP, marks=(MARK,))).wait(1800)
    assert ra.error is None
    w.stop(persist=False)
    store.flush()

    store2 = ST.BlobStore(1 << 30, 1 << 30, tmp_path)
    w2 = _worker(_engine(x2, 2), store=store2, model_key="m").start()
    try:
        kept = sorted(e.n for e in w2.cache.entries.values() if e.prefix)
        assert kept == [128, 256, MARK]
        assert not any(k in store2.ram for e in w2.cache.entries.values() for k in e.blobs())
        b = system + qb
        rb = w2.submit(SC.Request(b, max_new=5, stop_ids=STOP, marks=(MARK,))).wait(1800)
        assert rb.error is None and rb.cached == MARK and rb.out == ref(b, 5)
        assert store2.stats["disk_hits"] > 0
    finally:
        w2.stop()


def test_a_store_left_part_way_restores_only_what_is_whole(x2, tmp_path):
    """A directory as a killed server and a stopped writer leave it (a cut's tail cut short, a
    .tmp, a blob no entry uses): the next server loads the whole entries and restores the longest.
    A page's file gone under a loaded entry: no part of it reaches the slot, the reply is the
    prompt's from zero."""
    ref = _Ref(x2)
    system = _prompts(30, [MARK])[0]
    qa, qb, qc = _prompts(31, [12, 9, 7])
    store = ST.BlobStore(1 << 30, 1 << 30, tmp_path)
    w = _worker(_engine(x2, 2), store=store, model_key="m").start()
    ra = w.submit(SC.Request(system + qa, max_new=3, stop_ids=STOP, marks=(MARK,))).wait(1800)
    assert ra.error is None
    w.stop(persist=False)
    store.flush()
    cut = next(e for e in w.cache.entries.values() if e.prefix and e.n == MARK)
    f = tmp_path / f"{cut.tail}.bin"
    f.write_bytes(f.read_bytes()[:100])
    (tmp_path / f".{cut.tail}.tmp").write_bytes(b"part of a file")
    orphan = f"{w.cache.mk}-p9-" + "ab" * 16
    (tmp_path / f"{orphan}.bin").write_bytes(bytes(100))

    store2 = ST.BlobStore(1 << 30, 1 << 30, tmp_path)
    w2 = _worker(_engine(x2, 2), store=store2, model_key="m").start()
    try:
        assert sorted(e.n for e in w2.cache.entries.values() if e.prefix) == [128, 256]
        assert not list(tmp_path.glob(".*.tmp")) and orphan not in store2
        b = system + qb
        rb = w2.submit(SC.Request(b, max_new=5, stop_ids=STOP, marks=(MARK,))).wait(1800)
        assert rb.error is None and rb.cached == 256 and rb.out == ref(b, 5)
        _consistent(w2)
    finally:
        w2.stop(persist=False)
    store2.flush()

    store3 = ST.BlobStore(1 << 30, 1 << 30, tmp_path)
    w3 = _worker(_engine(x2, 2), store=store3, model_key="m").start()
    try:
        e = max(w3.cache.entries.values(), key=lambda o: o.n)
        assert e.n == MARK and not any(k in store3.ram for k in e.blobs())
        (tmp_path / f"{e.pages[0]}.bin").unlink()
        c = system + qc
        rc = w3.submit(SC.Request(c, max_new=5, stop_ids=STOP)).wait(1800)
        assert rc.error is None and rc.cached == 0 and rc.out == ref(c, 5)
        assert w3.stats["restores"] == 0 and e.key not in w3.cache.entries
        _consistent(w3)
    finally:
        w3.stop(persist=False)


def _consistent(w) -> None:
    """Every entry's blobs are in the store, each of its size."""
    L = w.ios[0].layout
    for e in w.cache.entries.values():
        for k in e.pages:
            assert len(w.store.get(k)) == L.page_bytes()
        assert len(w.store.get(e.tail)) == (L.page_bytes(e.m) if e.m else 0) + L.state_bytes


def test_a_client_leaving_mid_feed_leaves_the_store_consistent(x2):
    """The client leaves at the feed's first cut: that cut is kept, the prompt stops, the slot is
    free. The next session restores the cut, or the slot's own context, and its tokens are a
    fresh engine's."""
    ref = _Ref(x2)
    eng = _engine(x2, 2)
    w = _worker(eng)
    keep, reqs = w._keep_prefix, []

    def left(r, io=None):
        keep(r, io)
        reqs.append(r)
        r.cancelled.set()
    w._keep_prefix = left
    w.start()
    try:
        system = _prompts(30, [MARK])[0]
        qa, qb = _prompts(31, [12, 9])
        r = w.submit(SC.Request(system + qa, max_new=5, stop_ids=STOP, marks=(MARK,))).wait(1800)
        assert r.finish == "abort" and r.error is None and r.out == [] and reqs == [r]
        assert [e.n for e in w.cache.entries.values()] == [128] and len(r.fed) == 128
        assert w.slot_req == [None, None]
        _consistent(w)
        w._keep_prefix = keep
        b = system + qb
        rb = w.submit(SC.Request(b, max_new=5, stop_ids=STOP, marks=(MARK,))).wait(1800)
        assert rb.error is None and rb.cached == 128 and rb.out == ref(b, 5)
        assert sorted(e.n for e in w.cache.entries.values() if e.prefix) == [128, 256, MARK]
        _consistent(w)
    finally:
        w.stop()


def test_a_read_failing_at_a_cut_stores_nothing_of_it(x2):
    """A device read failing in a cut's save: the request errors, that cut has no entry and no
    tail, what earlier cuts kept stands, and the next request runs."""
    ref = _Ref(x2)
    eng = _engine(x2, 2)
    w = _worker(eng)
    read, calls = w.ios[0].read_state, []

    def failing():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("the read failed")
        return read()
    w.ios[0].read_state = failing
    w.start()
    try:
        system = _prompts(30, [MARK])[0]
        qa, qb = _prompts(31, [12, 9])
        r = w.submit(SC.Request(system + qa, max_new=5, stop_ids=STOP, marks=(MARK,))).wait(1800)
        assert r.finish == "error" and "the read failed" in r.error
        assert [e.n for e in w.cache.entries.values()] == [128]
        _consistent(w)
        b = system + qb
        rb = w.submit(SC.Request(b, max_new=5, stop_ids=STOP, marks=(MARK,))).wait(1800)
        assert rb.error is None and rb.cached == 128 and rb.out == ref(b, 5)
        _consistent(w)
    finally:
        w.stop()


def test_prefix_block_zero_keeps_none_and_a_feed_has_no_cuts():
    """prefix_block 0 and a model without state keep nothing; a cut ends a feed run short and
    takes the iteration's whole budget's place in _rows_at's count of what this budget ends."""
    w = BW35.BatchWorker35.__new__(BW35.BatchWorker35)
    w.budget, w.prefix_block, w.prefix_skip, w.stateful = 512, 0, 16, True
    r = NS(marks=(100,), fed=[1] * 60, _feed=[1] * 700, _cuts=[])
    assert w._prefix_cuts(r, 0, 800) == set() and w._part(r) == 512
    w.prefix_block = 512
    assert w._prefix_cuts(r, 0, 1100) == {100, 512, 1024}
    assert w._prefix_cuts(r, 90, 1100) == {512, 1024}       # (the mark: within the skip)
    w.stateful = False
    assert w._prefix_cuts(r, 0, 1100) == set()
    r._cuts = [100, 512]                        # (the next cut's near)
    assert w._part(r) == 40
    r._cuts, r._feed = [], [1] * 7              # (the feed's end)
    assert w._part(r) == 7
    # a prompt whose last part stops at a cut does not join this iteration's step
    x = NS(finish=None, next=1, slot=0, fed=[1] * 100, _feed=[1] * 20, _cuts=[110], out=[],
           max_new=8, _bset=True)
    w.running, w.prefilling = [x], [x]
    w.steps = NS(kinds={0: "g"})
    assert w._rows_at(0) == ([], [])
    x._cuts = []
    assert w._rows_at(0) == ([(0, 120)], ["g"])
