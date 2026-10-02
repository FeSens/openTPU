"""A prompt's prefill from programs at run-time positions (docs/prefill.md, opentpu/llm/prefill.py)
on the ISA simulator: the logits and the image's DRAM (KV caches, DeltaNet states and windows,
the MTP layer's cache) equal those of compile-time runs of the same split bit for bit, from
position 0 and from a later position (a chat's next turn), across an attention bucket's end,
plain and MTP; the programs are compiled once per bucket, kind and rows."""
import dataclasses

import numpy as np
import pytest

from opentpu.isasim import board_config
from opentpu.llm import prefill as PF
from opentpu.llm.mtp import MTPDecoder, MTPStats, mtp_engine
from opentpu.llm.qwen3 import PREFILL_ROWS, RUN_WORDS, Engine, device_config

from test_mtp import CFG, FP4, _mtp_kv, _mtp_weights, _states
from test_qwen35 import _tiny_model

pytest.importorskip("torch")


def test_split():
    """R_max rows a run; the rows before conv_k - 1 in a compile-time run; a run-time run's rows
    in one bucket; the last run L."""
    assert PF.split(0, 10, 4, 256, 4) == [(0, 4, "P"), (4, 4, "P"), (8, 2, "L")]
    assert PF.split(1, 3, 4, 256, 4) == [(1, 2, "L")]
    assert PF.split(250, 263, 4, 256, 4) == [(250, 4, "P"), (254, 2, "P"), (256, 4, "P"),
                                             (260, 3, "L")]
    assert PF.split(37, 38, 4, 256, 4) == [(37, 1, "L")]
    # R_max per bucket: 3 rows in bucket 1, 1 in bucket 2
    assert PF.split(250, 259, lambda b: 3 if b == 1 else 1, 256, 4) == [
        (250, 3, "P"), (253, 3, "P"), (256, 1, "P"), (257, 1, "P"), (258, 1, "L")]


def _static(eng, toks, R_max):
    """Engine.prefill as compile-time runs (compile_rows) of the prompt runs' split."""
    p0, img, B = eng.pos, eng.image, eng.block
    for p, R, kind in PF.split(p0, p0 + len(toks), R_max, B, PF.conv_k(eng)):
        rows, lr, tk = [(0, p + j) for j in range(R)], [R - 1] if kind == "L" else [], \
            toks[p - p0:p - p0 + R]
        lg = eng._run_rows(rows, tk, lr, img.compile_rows(rows, lr, B, tokens=tk))
        eng.pos = p + R
    return lg[0]


def _drams(eng, P):
    """Each slice's DRAM but the prompt runs' own words: out[0 .. P] and the state's run
    words (tpos, LFM2's ring words: qwen3.RUN_WORDS)."""
    g = eng.image.lookup["gen"]
    out = []
    for s in eng.backend.machine.slices:
        d = s.dram.copy()
        d[g["out"]:g["out"] + 4 * (P + 1)] = 0
        for w in RUN_WORDS.values():
            d[g["state"] + 4 * w:g["state"] + 4 * w + 4] = 0
        out.append(d)
    return out


def _model(case):
    if case.startswith(("qwen3", "lfm2")):  # Qwen3 (no convolutions: every run at a run-time
        from test_autodecode import _tiny   # position), LFM2 (its 3-tap convolutions' ring)
        W, spec = _tiny(case.split("-")[0])
        if case == "qwen3-emb8":
            spec = dataclasses.replace(spec, embed="int8")
        S = 2 if case.endswith("-design") else 1
        if case.endswith("-board"):         # MCOLS 4: runs of two passes (8 rows)
            return W, spec, CFG, {}
        return W, spec, device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=S), {}
    if case == "kh16":
        _, W, spec = _tiny_model(16, 16, init=0.2)
        return W, spec, board_config(DRAM_BYTES=1 << 26, DSTEP=True, STREAM=True, PAIR=True), {}
    _, W, spec = _tiny_model(8, init=0.2)
    if case == "emb8":
        spec = dataclasses.replace(spec, embed="int8")
    if case == "design":
        return W, spec, device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=2), {}
    return W, spec, CFG, FP4 if case == "fp4" else {}


@pytest.mark.parametrize("case, P1, P2", [
    ("kh8", 250, 20),       # the second prompt from 250 across the bucket's end at 256
    ("kh16", 37, 30),       # the 0.8B's / 2B's 16 DeltaNet heads
    ("emb8", 37, 30),       # the int8 embedding (a scaled row gather per token)
    ("fp4", 37, 30),        # 4-bit weights with PAIR: runs of more than MCOLS / 2 rows
    ("design", 37, 30),     # two slices, 8 MXU columns: runs of 8 rows
    ("qwen3", 250, 20),     # Qwen3 (qwen3_rows), one slice
    ("qwen3-design", 37, 30),
    ("qwen3-emb8", 37, 30),     # the int8 embedding (Llama-likes: SmolLM3, Phi-4-mini)
    ("qwen3-board", 250, 20),   # the board's MXU (MCOLS 4): runs of 8 rows, two passes
    ("lfm2", 250, 20),
    ("lfm2-design", 37, 30),
])
def test_prompt_runs_are_compile_time_runs(case, P1, P2):
    """Two prompts, the second from where the first left (a chat's next turn): each one's last
    logits and then every slice's DRAM equal those of compile-time runs (compile_rows) of the
    same split, word for word; the programs: one per kind, rows and bucket (and the compile-time
    first run), none per prompt."""
    W, spec, cfg, kw = _model(case)
    r = np.random.default_rng(P1)
    p1, p2 = ([int(t) for t in r.integers(0, 1000, n)] for n in (P1, P2))
    a = Engine(spec, W, cap=512, cfg=cfg, resident=True, prompt_runs=True, **kw)
    assert PF.supported(a)
    got = [a.prefill(p1), a.prefill(p2)]
    rm = a._prompt_rmax                      # the tiny models: every bucket's the most rows
    full = a.image.rows if not hasattr(spec, "mtp") else min(cfg.MCOLS, a.image.rows)
    assert set(rm.values()) == {(full, True)}
    R_max = lambda blocks: rm[blocks][0]
    b = Engine(spec, W, cap=512, cfg=cfg, resident=True, **kw)
    want = [_static(b, p1, R_max), _static(b, p2, R_max)]
    assert all(np.array_equal(x, y) for x, y in zip(got, want))
    assert [s["rows"] for s in a.stats] == [s["rows"] for s in b.stats]
    assert a.pos == b.pos == P1 + P2
    assert all(np.array_equal(x, y) for x, y in zip(_drams(a, P1 + P2), _drams(b, P1 + P2)))
    K = PF.conv_k(a)
    runs = PF.split(0, P1, R_max, 256, K) + PF.split(P1, P1 + P2, R_max, 256, K)
    keys = {(k, R, "at", p) if p < K - 1 else (k, R, p // 256 + 1)
            for p, R, k in runs} | {("L", R, b) for b, (R, _) in rm.items()}  # (r_max's probes)
    assert sorted(rm) == sorted({p // 256 + 1 for p, _, _ in runs})
    assert {(k[1], k[2], k[3], k[4]) if k[3] == "at" else (k[1], k[2], k[3])
            for k in a._prompt_progs} == keys
    if case == "fp4":
        return
    # int8 weights: a row's arithmetic is the decode kernel's in any split, so today's prefill
    # (fit_chunk's runs) gives the same logits, states and windows, and the next step's logits
    c = Engine(spec, W, cap=512, cfg=cfg, resident=True, **kw)
    assert all(np.array_equal(x, c.prefill(t)) for x, t in zip(got, (p1, p2)))
    if "linear" in getattr(spec, "kinds", ()):     # Qwen3.5's DeltaNet states and windows
        assert all(np.array_equal(x, y) for x, y in zip(_states(a, spec, 0), _states(c, spec, 0)))
    assert np.array_equal(a.step(5), c.step(5))



def test_todays_route_where_it_fits_more_rows(monkeypatch):
    """prefill.covers: where today's prefill (compile_rows) fits more rows in a bucket than its
    prompt run (here bucket 2's prompt programs held to 2 rows), a prompt that reaches the
    bucket takes today's runs (their rows, logits and the next step); one in bucket 1 still
    takes prompt runs."""
    from opentpu.compiler import CompileError
    W, spec, cfg, _ = _model("qwen3")
    r = np.random.default_rng(3)
    p1, p2 = ([int(t) for t in r.integers(0, 1000, n)] for n in (30, 240))
    a = Engine(spec, W, cap=512, cfg=cfg, resident=True, prompt_runs=True)
    real = a.image.compile_prompt_run

    def held(blocks, R, *args, **kw):
        if blocks == 2 and R > 2:
            raise CompileError("TMEM exhausted (the test's bucket 2)")
        return real(blocks, R, *args, **kw)
    monkeypatch.setattr(a.image, "compile_prompt_run", held)
    c = Engine(spec, W, cap=512, cfg=cfg, resident=True)
    R = min(cfg.MCOLS, a.image.rows)
    assert R > 3
    assert np.array_equal(a.prefill(p1), c.prefill(p1))
    assert [s["rows"] for s in a.stats] == [n for _, n, _ in PF.split(0, 30, R, 256, 1)]
    na, nc = len(a.stats), len(c.stats)
    assert np.array_equal(a.prefill(p2), c.prefill(p2))
    assert a._prompt_rmax == {1: (R, True), 2: (2, False)}
    assert [s["rows"] for s in a.stats[na:]] == [s["rows"] for s in c.stats[nc:]]
    assert {k for k in a._prompt_progs if k[3] == 2} == {("prompt", "L", 2, 2, False, 0)}
    assert np.array_equal(a.step(5), c.step(5))

def _static_mtp(dec, R_max):
    """MTPDecoder.prefill as compile-time runs of the prompt runs' split: the rows (hidden,
    the decoder's slot), then the MTP layer over them."""
    def prefill(tokens, st, chunk=None, pick=None, on_run=None):
        eng, img = dec.eng, dec.img
        toks = [int(t) for t in tokens]
        p0 = eng.pos
        P = p0 + len(toks)
        if p0 == 0:
            dec.slot = 0
        for p, R, kind in PF.split(p0, P, R_max, eng.block, PF.conv_k(eng)):
            i = p - p0
            dec._run(img.compile_rows([(0, p + j) for j in range(R)],
                                      [R - 1] if kind == "L" else [], eng.block,
                                      tokens=toks[i:i + R], slot=dec.slot, hidden=True),
                     "prefill", R, st)
            nxt = toks[i + 1:i + R + 1]
            if kind == "L":
                a0 = int(np.argmax(dec._logits(R)[R - 1]))
                nxt = nxt + [a0]
            draft = dec._draft(p, nxt, st, "mtp prefill")[R - 1]
        eng.pos, dec.draft = P, draft
        return a0, draft
    return prefill


@pytest.mark.parametrize("P1", [40, 245])
def test_mtp_prompt_runs_are_compile_time_runs(P1):
    """MTP: a prompt, greedy decoding, then the next turn's prompt (the last token and new
    ones, from the decoder's committed slot; at P1 245 across the bucket's end): the tokens,
    the first drafts and every slice's DRAM (the MTP layer's KV cache too) equal those of
    compile-time runs of the same split; plain prompt runs give the same first token and
    tokens (one split for plain and MTP)."""
    _, W, spec = _tiny_model(8, init=0.2)
    W = _mtp_weights(W, spec)
    r = np.random.default_rng(P1)
    p1, p2 = [int(t) for t in r.integers(0, 1000, P1)], [int(t) for t in r.integers(0, 1000, 9)]
    a = MTPDecoder(mtp_engine(spec, W, cap=512, cfg=CFG, prompt_runs=True))
    b = MTPDecoder(mtp_engine(spec, W, cap=512, cfg=CFG))
    b.prefill = _static_mtp(b, lambda blocks: PF.r_max(a.eng, blocks))
    sa, sb = a.generate(p1, max_new=6), b.generate(p1, max_new=6)
    assert sa.tokens == sb.tokens and a.slot == b.slot and a.eng.pos == b.eng.pos
    plain = Engine(spec, W, cap=512, cfg=CFG, resident=True, prompt_runs=True)
    assert plain.generate(p1, max_new=6) == sa.tokens
    assert [x[:2] for x in sa.runs] == [x[:2] for x in sb.runs]
    q = [sa.tokens[-1]] + p2
    got, want = a.prefill(q, MTPStats()), b.prefill(q, MTPStats())
    assert got == want and a.eng.pos == b.eng.pos == P1 + 5 + len(q)
    n = a.eng.pos
    assert all(np.array_equal(x, y) for x, y in zip(_drams(a.eng, n), _drams(b.eng, n)))
    assert all(np.array_equal(x, y) for x, y in
               zip(_states(a.eng, spec, a.slot), _states(b.eng, spec, b.slot)))
    assert all(np.array_equal(x, y) for x, y in zip(_mtp_kv(a.eng, n), _mtp_kv(b.eng, n)))
    assert {k[-1] for k in a.eng._prompt_progs} == {0, a.slot}    # (M's: slot 0)
    # int8: today's prefill (fit_chunk's runs, the MTP layer over 4 rows a run) the same
    c = MTPDecoder(mtp_engine(spec, W, cap=512, cfg=CFG))
    assert c.generate(p1, max_new=6).tokens == sa.tokens and c.prefill(q, MTPStats()) == got
    assert all(np.array_equal(x, y) for x, y in
               zip(_states(a.eng, spec, a.slot), _states(c.eng, spec, c.slot)))
    assert all(np.array_equal(x, y) for x, y in zip(_mtp_kv(a.eng, n), _mtp_kv(c.eng, n)))


def test_prompt_programs_come_from_the_cache(tmp_path, monkeypatch):
    """The prompt runs' programs through the program cache (Engine prog_cache): a new process's
    MTP engine reads every one from disk, none compiled; the device loops after prompt runs
    (MTP's and plain's) give the tokens of today's prefill (int8 weights)."""
    from opentpu import progcache as PC
    from opentpu.llm.mtp import NK
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path))
    PC.clear()
    _, W, spec = _tiny_model(8, init=0.2)
    W = _mtp_weights(W, spec)
    prompt = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 20)]

    def run(cache, prompt_runs=True):
        dec = MTPDecoder(mtp_engine(spec, W, cap=512, cfg=CFG, prog_cache=cache,
                                    prompt_runs=prompt_runs))
        return dec.generate_card(prompt, max_new=10, stop=[]).tokens, dec.eng

    want, _ = run(False, False)
    assert run(False)[0] == want and len(set(want)) > 4
    plain = Engine(spec, W, cap=512, cfg=CFG, resident=True, prompt_runs=True)
    t0 = int(np.argmax(plain.prefill(prompt)))
    assert [t0] + plain.generate_card(t0, 9, stop_ids=[]) == want
    run(True)
    PC.clear()
    s = dict(PC.stats)
    got, eng = run(True)
    assert got == want and PC.stats["compile"] == s["compile"]
    # (and R_max: progcache.fact)
    assert PC.stats["disk"] - s["disk"] == NK + len(eng._prompt_progs) + len(eng._prompt_rmax)
    PC.clear()


def test_bucket_1_programs_are_warmed_at_start(tmp_path, monkeypatch):
    """prefill.warm: an engine with the pipeline compiles bucket 1's prompt programs on a thread
    as it starts, so a prompt in bucket 1 compiles none and gives the logits of an engine
    without it; a new process's engine takes R_max and the programs from the program cache
    (no compile)."""
    from opentpu import progcache as PC
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path))
    PC.clear()
    _, W, spec = _tiny_model(8, init=0.2)
    prompt = [int(t) for t in np.random.default_rng(6).integers(0, 1000, 21)]
    eng = Engine(spec, W, cap=512, cfg=CFG, resident=True, prompt_runs=True, pipeline=True,
                 prog_cache=True)
    eng._prompt_warm.join(timeout=600)
    warmed = set(eng._prompt_progs)
    assert len(warmed) == 2 + PF.r_max(eng, 1)          # first run, P, L of every R
    lg = eng.prefill(prompt)
    assert set(eng._prompt_progs) == warmed
    assert np.array_equal(lg, Engine(spec, W, cap=512, cfg=CFG, resident=True,
                                     prompt_runs=True).prefill(prompt))
    PC.clear()
    s = dict(PC.stats)
    again = Engine(spec, W, cap=512, cfg=CFG, resident=True, prompt_runs=True, prog_cache=True)
    assert np.array_equal(again.prefill(prompt), lg) and PC.stats["compile"] == s["compile"]
    PC.clear()
