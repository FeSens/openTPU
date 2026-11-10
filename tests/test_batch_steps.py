"""Batched decode steps (opentpu/llm/batch.py) on the ISA simulator: an Engine(batch=4) image
with its lookup tables and batch area; three prompts prefilled into slots 0, 1 and 3 (slot 0
by its prompt runs, the others by the rows kernel with device inputs), then steps that feed a
token to each and pick the next on the device. Greedy rows give each sequence's own greedy
decode (a single-sequence engine's), sampled rows (top-k, top-p, temperature, a repetition
penalty) the decode loop's reference pick on that engine's logits with the row's uniform, bit
for bit; rows join and leave the steps; slot 0 keeps the generate loop."""
import dataclasses

import numpy as np
import pytest

from opentpu.llm import batch as BA
from opentpu.llm import generate as G
from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config

from test_serve_kvio import _model

torch = pytest.importorskip("torch")

CAP = 512


def _engines():
    W, spec, cfg1 = _model("qwen3")
    cfg = device_config(spec, CAP, batch=4, rows=PREFILL_ROWS, lookup=True, S=1)
    cfg = dataclasses.replace(cfg, DRAM_BYTES=2 * cfg.DRAM_BYTES)
    eng = Engine(spec, W, cap=CAP, cfg=cfg, batch=4, resident=True, prompt_runs=True,
                 batch_area=True)

    def single():
        return Engine(spec, W, cap=CAP, cfg=cfg1, resident=True, prompt_runs=True)
    return eng, single, spec


def _prompts(seed, sizes):
    r = np.random.default_rng(seed)
    return [[int(x) for x in r.integers(0, 1000, n)] for n in sizes]


def test_greedy_rows_are_each_sequences_decode():
    eng, single, spec = _engines()
    assert eng.bat is not None and eng.can_generate
    slots = [0, 1, 3]
    ps = _prompts(0, [21, 9, 30])
    st = BA.Steps(eng)
    nxt = {}
    for s, p in zip(slots, ps):
        nxt[s] = int(np.argmax(eng.prefill(p, seq=s)))
        st.set_sampling(s, None)
    want = {}
    for s, p in zip(slots, ps):
        e = single()
        lg, out = e.prefill(p), []
        for _ in range(7):
            t = int(np.argmax(lg))
            out.append(t)
            lg = e.step(t)
        want[s] = out
    got = {s: [nxt[s]] for s in slots}
    live = list(slots)
    for i in range(6):
        if i == 3:
            live = [0, 3]                   # slot 1 leaves; its row comes back after
        if i == 5:
            live = [0, 1, 3]
        if i == 5:                          # slot 1 catches up alone first
            while len(got[1]) < len(got[0]):
                t = st.run([(1, eng.poss[1])], [got[1][-1]])[0]
                got[1].append(t)
        rows = [(s, eng.poss[s]) for s in live if len(got[s]) < 7]
        if not rows:
            break
        out = st.run(rows, [got[s][-1] for s, _ in rows])
        for (s, _), t in zip(rows, out):
            got[s].append(t)
    for s in slots:
        assert got[s][:7] == want[s][:len(got[s][:7])], s
        assert len(got[s]) >= 6


@pytest.mark.parametrize("pen", [1.0, 1.3])
def test_sampled_rows_pick_as_the_decode_loop(pen):
    """Rows with sampling: each step's pick equals generate.reference_pick on the logits of a
    single-sequence engine fed the same tokens, with the same uniform; a greedy row beside."""
    eng, single, spec = _engines()
    slots = [0, 2]
    ps = _prompts(1, [17, 25])
    samps = {0: G.Sampling(0.8, 20, 0.9, pen), 2: None}
    st = BA.Steps(eng)
    refs, ctx, nxt = {}, {}, {}
    for s, p in zip(slots, ps):
        lg = eng.prefill(p, seq=s)
        nxt[s] = int(np.argmax(lg))
        ctx[s] = list(p) + [nxt[s]]
        st.set_sampling(s, samps[s], ctx[s])
        e = single()
        e.prefill(p)
        refs[s] = e
    rng = np.random.default_rng(5)
    for _ in range(6):
        u = rng.random(2).astype(np.float32)
        rows = [(s, eng.poss[s]) for s in slots]
        out = st.run(rows, [nxt[s] for s in slots], uniforms=u)
        for r, s in enumerate(slots):
            lg = refs[s].step(nxt[s])
            if samps[s] is None:
                want = int(np.argmax(lg))
            else:
                want = G.reference_pick(lg, samps[s], ctx[s], np.minimum(
                    u[r], np.float32(1 - 2.0 ** -24)), 1, None)
            assert out[r] == want, (s, r)
            nxt[s] = out[r]
            ctx[s].append(out[r])


def test_sampled_rows_with_head_chunks_off_the_sampler_blocks(monkeypatch):
    """Three sampled rows whose LM head chunk (head_rows_chunk) would not be whole sampler
    blocks (generate.BLK): x2's three rows, TMEM / 24 = 5,456 vocabulary rows, 85 blocks and
    16, here HEAD_CHUNK 336 (5 blocks and 16) over 1,000, top-k 3 (3 of the 16 blocks
    gathered, as the 35B's 20 of 970). The chunks keep whole blocks, so each block's maximum
    covers the logits its gather reads: every pick is the decode loop's (the card's B = 4
    window, 11-05: three sampled rows after the fourth ended picked from block maxima of other
    logits)."""
    from opentpu.llm import qwen3 as Q3
    monkeypatch.setattr(Q3, "HEAD_CHUNK", 336)
    eng, single, spec = _engines()
    slots = [0, 1, 3]
    ps = _prompts(4, [13, 22, 8])
    samp = G.Sampling(1.0, 3, 1.0, 1.0)
    st = BA.Steps(eng)
    refs, ctx, nxt = {}, {}, {}
    for s, p in zip(slots, ps):
        nxt[s] = int(np.argmax(eng.prefill(p, seq=s)))
        ctx[s] = list(p) + [nxt[s]]
        st.set_sampling(s, samp, ctx[s])
        refs[s] = single()
        refs[s].prefill(p)
    rng, sampled = np.random.default_rng(6), 0
    for _ in range(8):
        u = np.minimum(rng.random(3), 1 - 2.0 ** -24).astype(np.float32)
        out = st.run([(s, eng.poss[s]) for s in slots], [nxt[s] for s in slots], uniforms=u)
        for r, s in enumerate(slots):
            lg = refs[s].step(nxt[s])
            want = G.reference_pick(lg, samp, ctx[s], u[r], 1, None)
            assert out[r] == want, (s, r)
            sampled += want != int(np.argmax(lg))
            nxt[s] = want
            ctx[s].append(want)
    assert sampled > 0


def test_slot0_keeps_the_generate_loop_and_steps_follow_it():
    eng, single, spec = _engines()
    p = _prompts(2, [19])[0]
    e = single()
    t0 = int(np.argmax(e.prefill(p)))
    want = e.generate_card(t0, 10)
    t = int(np.argmax(eng.prefill(p)))
    got = eng.generate_card(t, 5)               # the loop on the device ...
    st = BA.Steps(eng)
    st.set_sampling(0, None)
    x = got[-1]
    for _ in range(5):                          # ... then batched steps
        x = st.run([(0, eng.poss[0])], [x])[0]
        got.append(x)
    assert got == want


def test_rows_split_into_groups_that_fit_imem():
    """With IMEM for one row's program only, a step of three rows runs in three groups: the
    same tokens as one run of all three."""
    eng, single, spec = _engines()
    slots, ps = [0, 1, 2], _prompts(3, [11, 14, 9])
    st = BA.Steps(eng)
    nxt = {}
    for s, p in zip(slots, ps):
        nxt[s] = int(np.argmax(eng.prefill(p, seq=s)))
        st.set_sampling(s, None)
    one = max(len(BA.compile_step(eng.image, eng.bat, [(s, eng.poss[s])], [None], eng.block,
                                  st.span([(s, eng.poss[s])]))[0]) for s in slots)
    st.limit = one + 2
    rows = [(s, eng.poss[s]) for s in slots]
    assert [len(g) for g in st.plan(rows)] == [1, 1, 1]
    got = {s: [nxt[s]] for s in slots}
    for _ in range(4):
        out = st.run([(s, eng.poss[s]) for s in slots], [got[s][-1] for s in slots])
        for s, t in zip(slots, out):
            got[s].append(t)
    for s, p in zip(slots, ps):
        e = single()
        lg, want = e.prefill(p), []
        for _ in range(5):
            want.append(int(np.argmax(lg)))
            lg = e.step(want[-1])
        assert got[s] == want, s
    assert st.runs == 12


def test_one_program_serves_a_span_of_steps_across_blocks():
    """A program serves its rows' steps (the step word t) until a row reaches the end of its
    attention block; that row then runs in the next block's program. The tokens are each
    sequence's decode; programs are compiled per span, not per step."""
    eng, single, spec = _engines()
    blk = eng.block
    slots = [0, 2]
    ps = _prompts(4, [blk - 3, 12])
    st = BA.Steps(eng)
    nxt = {}
    for s, p in zip(slots, ps):
        nxt[s] = int(np.argmax(eng.prefill(p, seq=s)))
        st.set_sampling(s, None)
    assert st.span([(0, eng.poss[0]), (2, eng.poss[2])]) == 3
    got = {s: [nxt[s]] for s in slots}
    for _ in range(8):
        out = st.run([(s, eng.poss[s]) for s in slots], [got[s][-1] for s in slots])
        for s, t in zip(slots, out):
            got[s].append(t)
    for s, p in zip(slots, ps):
        e = single()
        lg, want = e.prefill(p), []
        for _ in range(9):
            want.append(int(np.argmax(lg)))
            lg = e.step(want[-1])
        assert got[s] == want, s
    assert st.timing["programs"] == 2 and st.timing["reused"] == 6, st.timing


def test_feed_runs_then_the_last_token_picks_on_the_device():
    """A prompt fed in feed runs (MCOLS tokens, no LM head; from a multiple of MCOLS on, across
    an attention block) but its last token, which a step's row feeds: its pick and the next
    ones are the prompt's greedy decode. A feed program serves its block's runs."""
    eng, single, spec = _engines()
    blk = eng.block
    st = BA.Steps(eng)
    q = st.feed_rows()
    ps = {1: _prompts(6, [blk + 10])[0], 3: _prompts(7, [7])[0]}
    for s, p in ps.items():
        eng.prefill(p[:3], seq=s)               # (a cached prefix: the feed starts unaligned)
        st.set_sampling(s, None)
    p0 = st.timing["programs"]
    for s, p in ps.items():
        st.feed(s, p[3:-1])
        assert eng.poss[s] == len(p) - 1
    # slot 1: a head run (1 token to the multiple of q), a program per block, the last run's
    # (blk + 10 - 1 - 3 - 1 = blk + 5 tokens: 1 + blk + 4); slot 3: a head run of 1, then 2
    assert st.timing["programs"] - p0 <= 3 + 2, st.timing
    got = {s: [] for s in ps}
    nxt = {s: p[-1] for s, p in ps.items()}
    for _ in range(4):
        rows = [(s, eng.poss[s]) for s in ps]
        out = st.run(rows, [nxt[s] for s, _ in rows])
        for (s, _), t in zip(rows, out):
            got[s].append(t)
            nxt[s] = t
    for s, p in ps.items():
        e = single()
        lg, want = e.prefill(p), []
        for _ in range(4):
            want.append(int(np.argmax(lg)))
            lg = e.step(want[-1])
        assert got[s] == want, s
    assert q == eng.cfg.MCOLS


def test_a_joining_rows_step_is_compiled_ahead():
    """prefetch_rows compiles a coming step's program (a slot joining the decoding ones) in the
    background: the step then compiles nothing inline, and its tokens are the decode's."""
    eng, single, spec = _engines()
    eng.pipeline = True                         # (a compile thread on the simulator)
    st = BA.Steps(eng)
    ps = _prompts(8, [13, 21])
    nxt = {}
    for s, p in zip([0, 2], ps):
        nxt[s] = int(np.argmax(eng.prefill(p, seq=s)))
        st.set_sampling(s, None)
    st.run([(0, eng.poss[0])], [nxt[0]])        # slot 0 decodes alone ...
    st.prefetch_rows([(0, eng.poss[0]), (2, eng.poss[2])], [None, None])
    inline = st.timing["compiled_inline"]
    a = st.run([(0, eng.poss[0]), (2, eng.poss[2])], [nxt[0], nxt[2]])   # ... slot 2 joins
    assert st.timing["compiled_inline"] == inline, st.timing
    e = single()
    e.prefill(ps[1])
    assert a[1] == int(np.argmax(e.step(nxt[2])))


def test_feed_kinds_from_another_process():
    """A feed row's kind as the compile worker process gets it (unpickled: an equal string, not
    the same object) compiles as FEED."""
    import pickle
    eng, single, spec = _engines()
    kind = pickle.loads(pickle.dumps("".join(["fe", "ed"])))
    rows = [(1, 8 + j) for j in range(4)]
    progs = BA.compile_step(eng.image, eng.bat, rows, [kind] * 4, eng.block, 4)
    assert len(progs[0]) > 0


def test_a_leaving_rows_slot_pads_until_the_new_program():
    """A row leaves a 3-row group: the other two run in the live program for PAD_AHEAD steps,
    the free slot's row fed a dummy token past its context, then in their own program (compiled
    for that step): their tokens are their decodes; the left slot's context is intact."""
    eng, single, spec = _engines()
    eng.pipeline = True
    st = BA.Steps(eng)
    slots, ps = [0, 1, 2], _prompts(10, [11, 14, 9])
    got = {}
    for s, p in zip(slots, ps):
        got[s] = [int(np.argmax(eng.prefill(p, seq=s)))]
        st.set_sampling(s, None)
    for _ in range(2):
        out = st.run([(s, eng.poss[s]) for s in slots], [got[s][-1] for s in slots])
        for s, t in zip(slots, out):
            got[s].append(t)
    n1 = eng.poss[1]                           # slot 1 leaves here
    live = [0, 2]
    for _ in range(BA.PAD_AHEAD + 2):
        out = st.run([(s, eng.poss[s]) for s in live], [got[s][-1] for s in live], free={1})
        for s, t in zip(live, out):
            got[s].append(t)
    assert st.timing["padded"] == BA.PAD_AHEAD, st.timing
    assert eng.poss[1] == n1 + BA.PAD_AHEAD
    eng.adopt(n1, seq=1)                       # slot 1's context goes on as if never padded
    for _ in range(2):
        got[1].append(st.run([(1, eng.poss[1])], [got[1][-1]])[0])
    for s, p in zip(slots, ps):
        e = single()
        lg, want = e.prefill(p), []
        for _ in range(len(got[s])):
            want.append(int(np.argmax(lg)))
            lg = e.step(want[-1])
        assert got[s] == want, s


def test_mixed_runs_feed_a_prompt_beside_decoding_rows():
    """Decoding rows and a prompt's feed rows in one run (two run-time values: the decoding
    rows advance one a run, the feed rows k): the decoding rows' tokens are their decodes, the
    fed prompt's last token then picks its greedy decode; one program serves the prompt's runs
    (its feed rows compiled at their block's start)."""
    eng, single, spec = _engines()
    st = BA.Steps(eng)
    ps = _prompts(11, [13, 9, 23])
    got = {}
    for s in (0, 1):
        got[s] = [int(np.argmax(eng.prefill(ps[s], seq=s)))]
        st.set_sampling(s, None)
    st.set_sampling(2, None)
    feed = list(ps[2][:-1])                     # slot 2's prompt, 2 tokens a run beside them
    p0 = st.timing["programs"]
    while feed:
        k = min(2, len(feed))
        rows = [(s, eng.poss[s]) for s in (0, 1)]
        assert st.mixed_ready(rows, 2, k)
        out = st.mixed(rows, [got[0][-1], got[1][-1]], None, 2, feed[:k])
        del feed[:k]
        for s, t in zip((0, 1), out):
            got[s].append(t)
    assert eng.poss[2] == len(ps[2]) - 1
    assert st.timing["programs"] - p0 == 1, st.timing     # 11 runs of 2 tokens, one program
    got[2] = [st.run([(2, eng.poss[2])], [ps[2][-1]])[0]]
    for s, p in enumerate(ps):
        e = single()
        lg, want = e.prefill(p), []
        for _ in range(len(got[s])):
            want.append(int(np.argmax(lg)))
            lg = e.step(want[-1])
        assert got[s] == want, s
