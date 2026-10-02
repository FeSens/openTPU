"""A prompt's prefill from programs at run-time positions (docs/prefill.md).

The host writes the prompt's tokens to the generate area's out[] (the token at each position),
then runs one program per run of up to R_max rows: P (rows), L (the prompt's last run: its last
row's logits) and, on an MTP image, M (the MTP layer over the run's rows) after each. A program
takes its position from the generate state's tpos word and its tokens from out[], so it serves
every run of its bucket and kind, and the engine keeps it (and the program cache across
processes): the host compiles nothing per prompt. The rows before conv_k - 1 (a new context's
first run) are at compile-time positions, cached the same way.

The runs' split depends only on the first position, the prompt's length, R_max and the attention
block: R_max rows a run, cut where a run would cross a bucket's end (a run-time run's rows stay in
its bucket). R_max: the most rows of one MXU pass (MCOLS, at most the image's rows) whose last
bucket's L program compiles and fits IMEM. Plain and MTP engines of a model take the same split,
so their logits are the same bit for bit.
"""
from __future__ import annotations

import time

import numpy as np

from .. import isa as I
from ..compiler import CompileError
from . import generate as G


def split(p0: int, P: int, R_max: int, block: int, K: int) -> list[tuple[int, int, str]]:
    """The runs (first position, rows, kind) of the prompt positions [p0, P)."""
    runs, p = [], p0
    while p < P:
        n = min(R_max, P - p)
        if p >= K - 1:                      # a run-time run: its rows in one bucket
            n = min(n, block - p % block)
        runs.append((p, n, "L" if p + n == P else "P"))
        p += n
    return runs


def supported(eng) -> bool:
    """The engine's image runs prompts this way: a dense model's resident image, without
    rows the host writes before each run."""
    img = eng.image
    return (hasattr(img, "compile_prompt_run") and bool(getattr(img, "lookup", None))
            and getattr(eng.spec, "moe", None) is None
            and not (eng._host_rows is not None and eng._host_rows([0]))
            and getattr(eng, "row_server", None) is None)


def programs(eng, p: int, R: int, kind: str, hidden: bool = False, slot: int = 0):
    """The run's programs (kept by the engine; through its program cache), or on the card
    (runs_words) the program assembled once: a run of another program loads its words, not
    an assembly of them (3-5 ms a run; MTP's rows and M runs alternate). hidden, slot: MTP's
    rows (their hidden stored, the states' slot); M's take neither."""
    img, block, K = eng.image, eng.block, eng.spec.conv_k
    if kind == "M":
        hidden, slot = False, 0
    done = eng.__dict__.setdefault("_prompt_progs", {})
    if p < K - 1:
        what = ("prompt", kind, R, "at", p, hidden, slot)

        def compile():
            return img.compile_prompt_run(0, R, kind, block, hidden, slot, p0=p), None
    else:
        blocks = p // block + 1
        what = ("prompt", kind, R, blocks, hidden, slot)

        def compile():
            return img.compile_prompt_run(blocks, R, kind, block, hidden, slot), None
    if what not in done:
        progs = eng.cached(what, compile)[0]
        if not G.fits(img, progs):
            raise CompileError(f"prompt run {what}: {max(map(len, progs))} instructions, "
                               f"IMEM {img.cfg.IMEM_WORDS // 8}")
        prep = getattr(eng.backend, "prepare", None)
        if getattr(eng.backend, "runs_words", False) and len(progs) == 1:
            progs = np.asarray(I.assemble(progs[0]), np.uint32)
        elif prep is not None:
            prep(progs)
        done[what] = progs
    return done[what]


def r_max(eng) -> int:
    """The most rows of a run: one MXU pass (MCOLS) at most, the image's rows at most, and the
    last bucket's (plain) L program compiled and fitting IMEM. MTP's runs take the same R_max
    (their programs must fit it), so plain and MTP prefills of a model split alike."""
    if "_prompt_rmax" in eng.__dict__:
        return eng._prompt_rmax
    img, block = eng.image, eng.block
    last = (img.cap - 1) // block * block               # the last bucket's first position
    p = max(last, eng.spec.conv_k - 1)
    for R in range(min(img.cfg.MCOLS, img.rows, img.cap - p), 0, -1):
        try:
            programs(eng, p, R, "L")
        except CompileError as e:
            if not any(w in str(e) for w in ("TMEM", "ACT RAM", "IMEM")):
                raise
            continue
        eng._prompt_rmax = R
        return R
    raise CompileError("no prompt run fits")


def write_tokens(eng, p0: int, tokens) -> None:
    """The prompt's tokens to out[p0 ..] (every slice's)."""
    g = eng.image.lookup["gen"]
    t = np.asarray(tokens, np.float32)
    for s in range(eng.cfg.S):
        eng.backend.write(s, g["out"] + 4 * p0, t)


def run(eng, progs, p: int, R: int) -> dict:
    """One run at position p (its tpos word written first)."""
    if p >= eng.spec.conv_k - 1:
        g = eng.image.lookup["gen"]
        for s in range(eng.cfg.S):
            eng.backend.write(s, g["state"] + 4 * G.S_TPOS,
                              np.full(1, p % eng.block, np.float32))
    st = eng.backend.run(progs)
    st["rows"] = R
    return st


def logits(eng, R: int) -> np.ndarray:
    """The L run's last row's logits."""
    io, S, v, v_loc = eng.image.io, eng.cfg.S, eng.spec.vocab, eng.image.v_loc
    return np.concatenate([eng.backend.read(s, io["logits"] + 4 * ((R - 1) * v + s * v_loc),
                                            4 * v_loc).view(np.float32) for s in range(S)])


def chunks(eng, tokens):
    """Engine.prefill_chunks's runs this way: yields (the run's tokens, logits after the
    prompt's last token with the last run, else None)."""
    tokens = [int(t) for t in tokens]
    p0 = eng.poss[0]
    P = p0 + len(tokens)
    if P > eng.cap:
        raise RuntimeError("KV cache full")
    if not tokens:
        return
    R_max = r_max(eng)
    write_tokens(eng, p0, tokens)
    for p, R, kind in split(p0, P, R_max, eng.block, eng.spec.conv_k):
        progs = programs(eng, p, R, kind)
        if kind == "L":
            eng._prefetch(P)            # the first decode step's, while the device runs
        eng.stats.append(run(eng, progs, p, R))
        eng.poss[0] = p + R
        yield tokens[p - p0:p - p0 + R], logits(eng, R) if kind == "L" else None


def mtp(dec, tokens, st, pick=None, on_run=None):
    """MTPDecoder.prefill's runs this way: each run's rows (their hidden stored, the states in
    the decoder's slot), then the MTP layer over them (M: the tokens after the rows, out[P] the
    prompt's last logits' pick a0). Returns (a0, the first draft); on_run as the decoder's."""
    eng = dec.eng
    toks = [int(t) for t in tokens]
    p0 = eng.pos
    P = p0 + len(toks)
    a0 = draft = None
    t0 = time.perf_counter()
    R_max = r_max(eng)
    st.compile_s += time.perf_counter() - t0
    write_tokens(eng, p0, toks)
    for p, R, kind in split(p0, P, R_max, eng.block, eng.spec.conv_k):
        t0 = time.perf_counter()
        rows, mp = programs(eng, p, R, kind, True, dec.slot), programs(eng, p, R, "M")
        st.compile_s += time.perf_counter() - t0
        st.runs.append(("prefill", R, run(eng, rows, p, R)))
        if kind == "L":
            lg = logits(eng, R)
            a0 = int(np.argmax(lg)) if pick is None else int(pick(lg))
            write_tokens(eng, P, [a0])
        st.runs.append(("mtp prefill", R, run(eng, mp, p, R)))
        draft = dec._drafts(R)[R - 1]
        eng.pos = p + R
        if on_run is not None and on_run(toks[p - p0:p - p0 + R]) and p + R < P:
            return None, None
    return a0, draft
