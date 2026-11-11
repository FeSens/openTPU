"""The batching worker (opentpu/serve/batching.py) on the ISA simulator: a tiny Qwen3 with 4
KV slots. Requests submitted together run in their slots, their decode steps batched (one run
for every decoding request, each next token picked on the device): every request's greedy
tokens equal its own fresh run's; a seeded sampled request gives the same text alone and among
others; a prompt sharing a prefix with another slot's context takes its pages by a device copy,
one sharing a saved context from the store; more requests than slots wait for one; a lone
request in slot 0 runs the card's decode loop; stop strings and cancels end requests."""
import dataclasses

import pytest

from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
from opentpu.serve import batching as BW
from opentpu.serve import scheduler as SC

from test_serve_kvio import _model
from test_serve_scheduler import _Tok, _prompts, _ref

torch = pytest.importorskip("torch")

CAP = 512


def _setup(B=4, **kw):
    W, spec, cfg1 = _model("qwen3")
    cfg = device_config(spec, CAP, batch=B, rows=PREFILL_ROWS, lookup=True, S=1, **kw)
    cfg = dataclasses.replace(cfg, DRAM_BYTES=2 * cfg.DRAM_BYTES)
    eng = Engine(spec, W, cap=CAP, cfg=cfg, batch=B, resident=True, prompt_runs=True,
                 batch_area=True)
    return eng, W, spec, cfg1


def test_requests_batched_in_slots_are_their_own_runs():
    eng, W, spec, cfg1 = _setup()
    stop = frozenset()
    w = BW.BatchWorker(eng, _Tok()).start()
    try:
        ps = _prompts(1, [21, 34, 9, 27, 15, 12])          # six requests, four slots
        reqs = [w.submit(SC.Request(p, max_new=9, stop_ids=stop)) for p in ps]
        for r in reqs:
            r.wait(timeout=900)
        for p, r in zip(ps, reqs):
            assert r.finish == "length" and r.error is None, r.error
            assert r.out == _ref(spec, W, cfg1, p, 9, stop), r.rid
        st = w.usage()
        assert st["steps"] > 0 and st["step_rows"] > st["steps"]      # rows batched
        assert {r.slot for r in reqs} == {0, 1, 2, 3}
        # a request continuing the first conversation feeds only what is new
        nxt = ps[0] + reqs[0].out + _prompts(2, [6])[0]
        r = w.submit(SC.Request(nxt, max_new=5, stop_ids=stop)).wait(900)
        assert r.out == _ref(spec, W, cfg1, nxt, 5, stop)
        assert r.cached >= len(ps[0]) + len(reqs[0].out) - 1
    finally:
        w.stop()


def test_shared_prefix_from_another_slot_and_from_the_store():
    eng, W, spec, cfg1 = _setup()
    stop = frozenset()
    w = BW.BatchWorker(eng, _Tok()).start()
    try:
        sysp = _prompts(5, [270])[0]
        a, b = _prompts(6, [10, 13])
        a, b = sysp + a, sysp + b
        r0 = w.submit(SC.Request(a, max_new=3, stop_ids=stop)).wait(900)
        with w.cv:                                  # bound in one iteration: a takes its slot
            ra = w.submit(SC.Request(a, max_new=40, stop_ids=stop))       # back, b copies
            rb = w.submit(SC.Request(b, max_new=6, stop_ids=stop))        # its pages from it
        rb.wait(900)
        ra.wait(900)
        assert r0.slot == ra.slot and ra.cached == len(a) - 1
        assert rb.cached == 270 and w.stats["slot_copies"] >= 1
        # fill every slot with other contexts: a's slot is overwritten, its context saved
        others = [w.submit(SC.Request(p, max_new=2, stop_ids=stop))
                  for p in _prompts(7, [20, 21, 22, 23, 24])]
        for r in others:
            r.wait(900)
        c = a + ra.out + _prompts(8, [7])[0]
        rc = w.submit(SC.Request(c, max_new=5, stop_ids=stop)).wait(900)
        assert rc.cached >= len(a) and w.stats["restores"] >= 1
        for p, r, n in ((a, ra, 40), (b, rb, 6), (c, rc, 5)):
            assert r.out == _ref(spec, W, cfg1, p, n, stop)
    finally:
        w.stop()


def test_sampled_requests_alone_and_together():
    """A seeded sampled request's tokens are the same alone (slot 0: the card's loop) and with
    others batched beside it (each row its own uniforms, picked on the device)."""
    eng, W, spec, cfg1 = _setup()
    stop = frozenset()
    samp = SC.Sampling(0.9, 16, 0.9, 1.2, seed=11)
    p = _prompts(9, [18])[0]
    w = BW.BatchWorker(eng, _Tok()).start()
    try:
        alone = w.submit(SC.Request(p, max_new=12, stop_ids=stop, sampling=samp)).wait(900)
        assert w.stats["card_loops"] >= 1
        w.flush()
        others = [w.submit(SC.Request(q, max_new=14, stop_ids=stop,
                                      sampling=SC.Sampling(1.0, 32, 0.95, 1.0, seed=i)))
                  for i, q in enumerate(_prompts(10, [22, 25]))]
        together = w.submit(SC.Request(p, max_new=12, stop_ids=stop, sampling=samp))
        together.wait(900)
        for r in others:
            r.wait(900)
        assert together.out == alone.out
        assert w.stats["step_rows"] > w.stats["steps"]
    finally:
        w.stop()


class _Halts:
    """The ISA simulator's backend with a card's run_generate: a run's tokens as they land, and
    on the armed run, after `k` of them, `arrive()` (another request comes) and the loop halts
    as stop() halts the card's (the run's other uniforms drawn, not used)."""

    def __init__(self, be, k: int):
        self.be, self.k, self.arrive = be, k, None

    def __getattr__(self, name):
        return getattr(self.be, name)

    def run_generate(self, progs, out, nb, on_token, stop, state):
        import numpy as np
        from opentpu.llm import generate as G
        st = self.be.run(progs)
        w = self.be.read(0, out, 4 * nb).view(np.uint32)
        k = int(np.argmax(w == G.OUT_MARK)) if (w == G.OUT_MARK).any() else nb
        got = [int(x) for x in w[:k].view(np.float32)]
        if self.arrive is not None and len(got) > self.k:
            got = got[:self.k]
            self.arrive()
            self.arrive = None
            assert stop()
        for t in got:
            on_token(t)
        return st, got


def test_sampled_request_halted_in_the_card_loop_keeps_its_uniforms():
    """A seeded sampled request decoding alone in slot 0 (the card's loop, which draws a run's
    uniforms ahead) that another request's arrival halts after 3 tokens goes on in the steps
    with the uniform after its last token's: its tokens are its reply alone's."""
    eng, W, spec, cfg1 = _setup()
    eng.backend = _Halts(eng.backend, 3)
    stop = frozenset()
    samp = SC.Sampling(0.9, 16, 0.9, 1.0, seed=4)
    p = _prompts(11, [16])[0]
    w = BW.BatchWorker(eng, _Tok()).start()
    try:
        alone = w.submit(SC.Request(p, max_new=14, stop_ids=stop, sampling=samp)).wait(900)
        w.flush()
        late = []
        eng.backend.arrive = lambda: late.append(w.submit(SC.Request(
            _prompts(12, [20])[0], max_new=6, stop_ids=stop,
            sampling=SC.Sampling(1.0, 32, 0.95, 1.0, seed=2))))
        loops = w.stats["card_loops"]
        again = w.submit(SC.Request(p, max_new=14, stop_ids=stop, sampling=samp)).wait(900)
        late[0].wait(900)
        assert w.stats["card_loops"] > loops and eng.backend.arrive is None
        assert again.error is None and again.out == alone.out
    finally:
        w.stop()


def test_stop_strings_and_cancel_in_slots():
    eng, W, spec, cfg1 = _setup()
    w = BW.BatchWorker(eng, _Tok()).start()
    try:
        p = _prompts(11, [15])[0]
        want = _ref(spec, W, cfg1, p, 12, frozenset())
        text = _Tok().decode(want)
        long = w.submit(SC.Request(_prompts(12, [20])[0], max_new=200, stop_ids=frozenset()))
        r = w.submit(SC.Request(p, max_new=12, stop_ids=frozenset(), stop_strs=(text[5:7],)))
        assert "".join(r.stream(timeout=900)) == text[:5] and r.finish == "stop"
        long.cancel()
        long.wait(900)
        assert long.finish == "abort" and len(long.out) < 200
        assert all(x is None for x in w.slot_req)
    finally:
        w.stop()


@pytest.mark.parametrize("mixed", [False, True])
def test_the_coming_steps_rows_are_told_ahead(mixed):
    """_rows_at: prompts of 9 and 30 tokens with a feed budget of 8 an iteration. The first is
    fed by the budget in iteration 0 and decodes in its step; the second is then fed by the
    budget (8 tokens an iteration), or with mixed runs rides that step's spare rows (MCOLS - 1
    tokens each), and decodes from the step after its last. The rows predicted at the start
    are the rows each iteration's step takes."""
    eng, W, spec, cfg1 = _setup()
    w = BW.BatchWorker(eng, _Tok(), prefill_budget=8, mixed=mixed)
    ps = _prompts(9, [9, 30])
    rs = [SC.Request(p, max_new=20, sampling=SC.Sampling(0.0, 0, 1.0, 1.0, None),
                     stop_ids=frozenset({-1})) for p in ps]
    for r in rs:
        w.running.append(r)
        w._bind(r)
    q = w.steps.feed_rows()
    if mixed:                                   # iterations until the second decodes
        n = 1 + -(-(len(ps[1]) - 1) // (q - 1))
    else:
        n = -(-(len(ps[1]) - 1) // w.budget)
    pred = {k: w._rows_at(k)[0] for k in range(n + 1)}
    seen = []
    run, mix = w.steps.run, w.steps.mixed
    w.steps.run = lambda rows, *a, **k: (seen.append(list(rows)), run(rows, *a, **k))[1]
    w.steps.mixed = lambda rows, *a, **k: (seen.append(list(rows)), mix(rows, *a, **k))[1]
    for it in range(n + 1):                     # the loop's body
        m = w._mixed_step() if w.prefilling and w.mixed else "no"
        if m != "ran":
            b = w.budget if m == "no" else 1
            while w.prefilling and b > 0:
                b -= w._prefill_run(w.prefilling[0])
            w._decode_step(w._decoding())
    assert seen == [pred[k] for k in range(n + 1)]
    assert seen[-1] == sorted([(rs[0].slot, 8 + n), (rs[1].slot, 29)])


def test_requests_pack_into_a_block_of_slots():
    """A step runs each block of feed_rows() slots in a run of its own: a request takes a free
    slot in the block holding the most requests, before one holding a longer prefix of its
    prompt (copied on the device instead) or the least recently used."""
    eng, W, spec, cfg1 = _setup(B=8, MCOLS=4)                    # (the card's: two blocks)
    w = BW.BatchWorker(eng, _Tok())
    q = w.steps.feed_rows()
    assert q == 4
    w.slot_used = [0.0 if s >= q else 1.0 for s in range(8)]     # (the other block's older)
    rs = [SC.Request(p, max_new=4) for p in _prompts(13, [9] * q)]
    w.slot_dev[q] = list(rs[1].prompt[:5])                         # (rs[1]'s prefix there)
    for r in rs:
        w.running.append(r)
        w._bind(r)
    assert sorted(r.slot for r in rs) == list(range(q))
    assert rs[1].cached == 5 and w.stats["slot_copies"] == 1
