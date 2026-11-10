"""otpu-serve --mtp (opentpu/serve/scheduler.py's worker with the MTP decoder) on a tiny streamed
Qwen3.5-MoE with its MTP drafter (tests/test_mtp_moe.py's), on the ISA simulator: requests in
turns are their own plain greedy runs, and a context switched out and back holds the MTP layer's
KV cache, the committed slot's state and the draft bit for bit (kvio's pages and state); a prompt
that goes on from a saved reply start with another token appends the MTP layer's row before it
again from the held residual row, bit for bit a fresh decoder's; a loop halted on the device
mid-run (the host's stop word, written while the card runs) continues from where it halted."""
import dataclasses

import numpy as np
import pytest

from opentpu.llm import generate as G
from opentpu.llm.mtp import MTPDecoder, MTPStats
from opentpu.serve import scheduler as SC

from test_mtp import _mtp_kv, _states
from test_mtp_moe import CFG, _plain, tiny_moe_mtp     # noqa: F401 (the fixture)
from test_mtp_moe import _layered as _layered32
from test_serve_scheduler import _Tok, _prompts

pytest.importorskip("torch")

CFG64 = dataclasses.replace(CFG, DRAM_BYTES=1 << 26)   # (the tiled expert-major image's room)


def _layered(spec, W, em):
    return _layered32(spec, W, em, cfg=CFG64)


def _ctx(w, n: int) -> list:
    """The device's MTP context of n positions: the MTP layer's KV cache, the committed slot's
    DeltaNet states and windows, and the draft."""
    d = w.dec
    return _mtp_kv(w.eng, n) + _states(w.eng, w.eng.spec, d.slot) + [np.array([d.draft])]


def _same(a, b) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_mtp_requests_in_turns_are_their_own_runs(tiny_moe_mtp):
    """Four requests at once, quantum 3: each runs in several turns, the others' contexts in
    between; their tokens are plain greedy decode's. Each switch out saves the MTP context
    (the MTP layer's KV cache, the committed slot's state, the draft after the next token) and
    the switch back puts it back the same (the state in slot 0). A request whose reply ends
    right after its prompt (max_new 1) leaves its last row's draft pending: the next switch
    runs it before the save, and a prompt that goes on from that context with its token
    restores it with the MTP layer's row a fresh decoder has there."""
    W, spec = tiny_moe_mtp
    w = SC.Worker(_layered(spec, W, False), _Tok(), quantum=3, max_running=4, mtp=True)
    saved, back = {}, []
    save, restored = w._save_device, w._mtp_restored

    def save_(pin_for=None, r=None, state_at=None):
        save(pin_for, r=r, state_at=state_at)
        if pin_for is not None:
            saved[pin_for.rid] = _ctx(w, len(w.dev))

    def restored_(r, e, want, k):
        restored(r, e, want, k)
        if r.fed:
            back.append((r.rid, w.dec.slot, _same(_ctx(w, k), saved[r.rid])))
    w._save_device, w._mtp_restored = save_, restored_
    w.start()
    try:
        ps = _prompts(1, [21, 34, 9, 27])
        reqs = [w.submit(SC.Request(p, max_new=10 if i else 1)) for i, p in enumerate(ps)]
        for r in reqs:
            r.wait(timeout=1200)
        for p, r in zip(ps, reqs):
            assert r.finish == "length" and r.error is None, r.error
            assert r.out == _plain(spec, W, p, len(r.out)), r.rid
        assert sum(r.swaps for r in reqs) >= 3 and w.stats["restores"] >= 3
        assert back and all(ok and slot == 0 for _, slot, ok in back), back
        assert w.stats["mtp_iters"] == sum(r.mtp_iters for r in reqs) > 0
        nxt = ps[0] + reqs[0].out + _prompts(2, [6])[0]
        r = w.submit(SC.Request(nxt, max_new=1)).wait(1200)
        assert r.out == _plain(spec, W, nxt, 1) and r.cached == len(ps[0])
        assert w.stats["refills"] == 0          # (the entry's follow: the first token)
        fresh = MTPDecoder(_layered(spec, W, False))
        fresh.prefill(nxt, MTPStats(), lazy=True)
        n = len(nxt) - 1
        assert _same(_mtp_kv(w.eng, n) + _states(w.eng, spec, w.dec.slot),
                     _mtp_kv(fresh.eng, n) + _states(fresh.eng, spec, fresh.slot))
        with pytest.raises(ValueError, match="sampling"):       # (not the device's loop's)
            w.submit(SC.Request(ps[0], sampling=SC.Sampling(temperature=0.7, top_k=0)))
    finally:
        w.stop()


@pytest.mark.parametrize("em", [False, True])
def test_mtp_prompt_from_a_held_reply_start(tiny_moe_mtp, em, monkeypatch):
    """A request with a reply start (at) saves its prompt's part before it with the token after
    it (follow) and the residual row before it (held). Another prompt that shares that part and
    goes on with another token restores it (cached = at) and its prefill appends the MTP layer's
    row at at - 1 again from the held row (MTPDecoder.refill): the MTP layer's KV cache and the
    committed state after its prompt are a fresh decoder's, bit for bit. Without the held row
    the row at at - 1 is the first prompt's (the tokens stay plain greedy decode's: only the
    drafts read it)."""
    W, spec = tiny_moe_mtp
    sysp, other = _prompts(5, [30, 17])
    at = len(sysp)
    a, b = sysp + [101, 7, 9, 400, 33, 5], sysp + [222, 8, 61, 14, 900, 3, 2, 70]
    fresh = MTPDecoder(_layered(spec, W, em))
    fresh.prefill(b, MTPStats(), lazy=True)
    n = len(b) - 1                          # (the last row's: pending, the loop's first run)
    want = _mtp_kv(fresh.eng, n) + _states(fresh.eng, spec, fresh.slot)

    def run(held: bool):
        w = SC.Worker(_layered(spec, W, em), _Tok(), mtp=True)
        if not held:
            monkeypatch.setattr(w.cache, "held_row", lambda e: None)
        w.start()
        try:
            ra = w.submit(SC.Request(a, max_new=4, at=at)).wait(1200)
            w.submit(SC.Request(other, max_new=3)).wait(1200)      # (another context between)
            rb = w.submit(SC.Request(b, max_new=1)).wait(1200)
            assert ra.out == _plain(spec, W, a, 4) and rb.out == _plain(spec, W, b, 1)
            assert rb.cached == at and w.stats["refills"] == held
            return _mtp_kv(w.eng, n) + _states(w.eng, spec, w.dec.slot)
        finally:
            w.stop()
    assert _same(run(True), want)
    got, kv = run(False), len(_mtp_kv(fresh.eng, n))
    assert _same(got[kv:], want[kv:]) and not _same(got[:kv], want[:kv])   # (the state alike)


def test_mtp_loop_halted_mid_run_continues(tiny_moe_mtp):
    """The card's halt (BoardBackend.run_generate's stop word, written while the loop runs) on
    the ISA simulator: run_generate here runs the loop with the WAITW hook watching out[], a
    second request arrives after the first token lands, and the stop word goes in at the next
    poll; the device halts after the token in flight. The first request's turn ends there, the
    second runs, and the first continues from the draft the halted run left: both are plain
    greedy decode's."""
    W, spec = tiny_moe_mtp
    eng = _layered(spec, W, False)
    be, cfg = eng.backend, eng.cfg
    w = SC.Worker(eng, _Tok(), quantum=4, max_running=4, mtp=True)
    arrive, runs = [], []

    def run_generate(programs, out, n, on_token=None, stop=None, state=None):
        m = be.machine
        host, seen, halted = m.host, [0], [False]

        def land():
            o = be.read(0, out, 4 * n).view(np.uint32)
            while seen[0] < n and o[seen[0]] != G.OUT_MARK:
                if on_token is not None:
                    on_token(int(o[seen[0]:seen[0] + 1].view(np.float32)[0]))
                seen[0] += 1
                if arrive:
                    arrive.pop()()

        def poll(mach):
            land()
            if stop is not None and not halted[0] and stop():
                for s in range(cfg.S):
                    be.write(s, state + 4 * G.S_HALT, np.ones(1, np.float32))
                halted[0] = True
            if host is not None:
                host(mach)
        m.host = poll
        try:
            st = be.run(programs)
        finally:
            m.host = host
        land()
        o = be.read(0, out, 4 * seen[0]).view(np.float32)
        runs.append((n, seen[0], halted[0]))
        return st, [int(x) for x in o]
    be.run_generate = run_generate
    w.start()
    try:
        pa, pb = _prompts(9, [26, 19])
        rb = []
        arrive.append(lambda: rb.append(w.submit(SC.Request(pb, max_new=6))))
        ra = w.submit(SC.Request(pa, max_new=16)).wait(1200)
        rb[0].wait(1200)
        assert ra.out == _plain(spec, W, pa, 16) and rb[0].out == _plain(spec, W, pb, 6)
        assert runs[0][2] and runs[0][1] < runs[0][0], runs     # (halted, tokens left)
        assert ra.swaps >= 1 and len(runs) >= 3
    finally:
        w.stop()


def test_mtp_dense_requests_in_turns_and_a_reply_start():
    """A dense Qwen3.5 with its MTP layer (tests/test_mtp.py's tiny model; its prompt in prompt
    runs, the first draft in the prefill): requests in turns, and one that goes on from
    another's reply start with another token, are plain greedy decode's. Its decoder holds no
    residual row: the part before the reply's start ends a token earlier (at - 1), and the MTP
    layer's row there took the token both prompts share."""
    from opentpu.llm.mtp import mtp_engine
    from test_mtp import CFG as DCFG
    from test_mtp import _mtp_weights
    from test_mtp import _plain as _dense
    from test_qwen35 import _tiny_model
    _, W, spec = _tiny_model(8, init=0.2)
    W = _mtp_weights(W, spec)
    w = SC.Worker(mtp_engine(spec, W, cap=512, cfg=DCFG), _Tok(), quantum=3, mtp=True)
    assert not w.dec.holds
    w.start()
    try:
        sysp, *ps = _prompts(3, [24, 13, 30, 8])
        at = len(sysp)
        a, b = sysp + [17, 5, 300, 9], sysp + [44, 6, 2, 71, 8]
        reqs = [w.submit(SC.Request(p, max_new=8)) for p in ps]
        ra = w.submit(SC.Request(a, max_new=6, at=at))
        for r in reqs + [ra]:
            r.wait(1200)
        rb = w.submit(SC.Request(b, max_new=6)).wait(1200)
        for p, r in zip(ps + [a, b], reqs + [ra, rb]):
            assert r.error is None and r.out == _dense(spec, W, p, len(r.out), {}), r.rid
        assert w.stats["restores"] >= 2 and rb.cached == at - 1 and w.stats["refills"] == 0
    finally:
        w.stop()
