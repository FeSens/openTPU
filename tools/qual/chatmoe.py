"""otpu-chat with a mixture-of-experts model streamed from its pool (docs/offload.md 14), scripted:
a few turns of one conversation through otpu-chat's own Chat, engine and sampler, on the ISA
simulator or the card, each turn's tokens and numbers as JSON. The run on the simulator with the
card's configuration (tools/qual/refs.py cfg) is the card's reference bit for bit: --check
compares every turn's tokens with it.

    python3 tools/qual/chatmoe.py MODEL --cfg cfg.pkl --out ref.json              # ISA simulator
    python3 tools/qual/chatmoe.py MODEL --backend board --check ref.json --out card.json

MODEL: a short name of opentpu.llm.MODELS (lfm2-8b, qwen35-35b, gemma4-26b) or a checkpoint
directory. The turns default to TURNS (two: a question, then one about the answer), greedy
(--sample: the model's sampling defaults, the uniforms from --seed's generator: the card's loop
picks the same tokens as the simulator's), --max-new tokens each. The pool, slots and prompt
settings are otpu-chat's (opentpu/host/streamed.py; --em-rows: expert-major's least rows a
call, docs/offload.md 13.16). Each turn lists its prompt's calls (a layer-major prompt's
prefill_layers: a model with state has two, up to the snapshot at the reply's start and after):
rows, expert-major or not, wall and device seconds, the expert server's counts over it and the
host's time on its requests and idle parts (--trace: each call's runs and the server's timeline
too, and each turn's host phases up to its first token, _Phases). On a tree whose
streamed.engine has no em_rows (production's) it runs without --em-rows.
The pool's page cache is recorded with them: the fraction of its packed experts in it (mincore,
PoolFile.resident, as tools/offload/pool_disk.py checks its arms) after the load and before and
after each turn (--pool-calls: each call too; a 17.5 GB pool's check takes 60-90 ms, which
the turn's wall times then include), and each call's and turn's pool reads by where they came
from (PoolFile.io: "cached", "disk"); a turn that read from the disk or began under 95% is
marked "cold" (its prompt's misses waited on the disk: session r2m8's 35B, 2026-10-13, 2,966
misses served in 40.2 s against production's 2.9 s, the same calls, read from the SSD at
~13.5 ms each).
--mtp (Qwen3.5-MoE): the MTP drafter's image (streamed.engine(mtp=): its pool the -mtp one,
pack_pool.py extend), each reply by the MTP loop on the device as otpu-chat --mtp decodes it
(docs/offload.md 15.6), greedy or sampled: the same tokens as plain decode's (--check against a
plain run's JSON), with each turn's iterations and accepted drafts; --mtp-from: the drafter's
checkpoint when not the model's (MiniCPM-V 4.7: its base Qwen3.5-35B-A3B). Exit 1 when --check
finds another token.

Images (a model with a vision side: MiniCPM-V, docs/minicpmv.md): --image N:PATH attaches PATH
to turn N (1-based; repeated), --max-slices bounds an image's slices; each turn reports its
images and their crops' wall seconds through the card's tower (vision_s), and the JSON keeps
each image's sha256 (the reference's must be the same bytes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

TURNS = ("What is the capital of France? Answer in one sentence.",
         "And what is it famous for? Two sentences.")
DATE = (2026, 9, 29)        # "today" for a chat template that states it (refs.py's DATE)


class _Pinned:
    """The tokenizer with its chat template's date pinned (a reference must not expire)."""

    def __init__(self, tok):
        self.tok = tok

    def __getattr__(self, k):
        return getattr(self.tok, k)

    def __call__(self, *a, **kw):
        return self.tok(*a, **kw)

    def apply_chat_template(self, *a, **kw):
        import datetime
        day = datetime.date(*DATE)
        return self.tok.apply_chat_template(*a, strftime_now=lambda fmt: day.strftime(fmt), **kw)


# the expert server's counts a prompt's call reports (their change over it): its requests'
# hits and misses, expert-major's need lines, their ids queued (not in a slot), already in
# one and landed, the needs end_prefill sent, idle parts landed whole (prefetched) and the
# polls that held one back near a run's end (holds, stage_holds), the bytes sent
CALL_COUNTS = ("hits", "misses", "need_lines", "needs_queued", "need_hits", "needs_landed",
               "drained", "prefetched", "holds", "stage_holds", "bytes")


def _pool(eng):
    """The engine's pool file (offload.PoolFile) and its packed experts' bytes; (None, 0) for a
    pool in host RAM."""
    pf = getattr(getattr(eng, "server", None), "pool_file", None)
    if pf is None or pf.ids is None or not len(pf.ids):
        return None, 0
    return pf, len(pf.ids) * pf.slot


def _resident(pf, total: int) -> float | None:
    """The fraction of the pool's packed experts in the page cache now (PoolFile.resident), or
    None where the host cannot tell."""
    res = None if pf is None else pf.resident(pf.ids)
    return None if res is None else round(res / total, 4)


def _io(pf) -> dict:
    """PoolFile.io's counts now (a copy): {cached | disk | unknown: [reads, s, bytes, absent]}."""
    return {k: list(v) for k, v in ((pf.io or {}) if pf is not None else {}).items()}


def _reads(io0: dict, io1: dict) -> dict:
    """The pool's reads between two _io copies by where they came from: [reads, s, MB]."""
    out = {}
    for k, v in io1.items():
        d = [x - y for x, y in zip(v, io0.get(k, [0] * len(v)))]
        if d[0]:
            out[k] = [d[0], round(d[1], 3), round(d[2] / 1e6, 1)]
    return out


def _calls(eng, calls: list, trace: list | None = None, pool_calls: bool = False) -> None:
    """Each prefill_layers call of eng appended to `calls`: its rows, expert-major or not (the
    gate's choice, streamed.gate_expert_major, else the Engine's), wall and device seconds,
    runs, the server's CALL_COUNTS over it, and from the server's events its requests and the
    host's seconds on them (serve_s: each seen to served, while the card waits) and on idle
    parts (parts_s); with `trace` (a list) also its runs (key, start, end, cycles; s from the
    call's start) and those events (seen, done, kind, layer or expert, misses or bytes)."""
    base, srv = eng.prefill_layers, eng.server
    pf, total = _pool(eng)
    res = (lambda: _resident(pf, total)) if pool_calls else (lambda: None)  # noqa: E731
    khz = (getattr(eng.backend, "info", None) or {}).get("core_khz")
    runs, t0 = [], [None]
    if trace is not None:
        lr, b, key = eng._layer_run, eng.backend, [None]

        def layer_run(k):
            key[0] = k
            return lr(k)
        eng._layer_run = layer_run
        if hasattr(b, "start"):                 # (the card's: start, then wait)
            start, wait = b.start, b.wait

            def started(*a, **kw):
                if t0[0] is not None:
                    runs.append([key[0], time.perf_counter() - t0[0]])
                return start(*a, **kw)

            def waited(*a, **kw):
                st = wait(*a, **kw)
                if t0[0] is not None and runs and len(runs[-1]) == 2:
                    runs[-1] += [time.perf_counter() - t0[0], (st or {}).get("cycles")]
                return st
            b.start, b.wait = started, waited

    def prefill_layers(tokens, logits: bool = True, **kw):     # (kw: an MTP image's, vis)
        c0 = {k: getattr(srv, k, 0) for k in CALL_COUNTS}
        res0, io0 = res(), _io(pf)              # (the pool's page cache: outside the timing)
        k, t = len(eng.stats), time.perf_counter()
        srv.events = []                         # (the server's timeline: each poll that did work)
        if trace is not None:
            runs.clear()
            t0[0] = t
        try:
            return base(tokens, logits, **kw)
        finally:
            dt = time.perf_counter() - t
            cyc = sum((x or {}).get("cycles") or 0 for x in eng.stats[k:])
            last = getattr(eng, "em_last", None)
            em = last[1] if last is not None and last[0] == len(tokens) else eng.expert_major
            ev, srv.events = srv.events or [], None
            req = [e for e in ev if e[2] == "d"]
            calls.append(dict(rows=len(tokens), em=bool(em), s=round(dt, 3),
                              device_s=round(cyc / (khz * 1e3), 3) if khz else None,
                              runs=len(eng.stats) - k, requests=len(req),
                              serve_s=round(sum(e[1] - e[0] for e in req), 3),
                              parts_s=round(sum(e[1] - e[0] for e in ev if e[2] == "p"), 3),
                              **{c: getattr(srv, c, 0) - c0[c] for c in CALL_COUNTS},
                              pool_before=res0, pool_after=res(),
                              reads=_reads(io0, _io(pf))))
            if trace is not None:
                t0[0] = None
                trace.append(dict(rows=len(tokens), em=bool(em), runs=[list(r) for r in runs],
                                  events=[[round(e[0] - t, 6), round(e[1] - t, 6)] +
                                          list(e[2:]) for e in ev]))
    eng.prefill_layers = prefill_layers


# the host's phases of a turn up to its first token (--trace, _Phases), and the ones that
# hold others (left out of _Phases.line's sum: their parts are phases of their own)
PHASES = ("template", "rollback", "mtp prefill", "prefill_layers", "end_prefill", "mtp snapshot",
          "snapshot", "a0", "run")
HOLDERS = ("mtp prefill", "mtp snapshot")


class _Phases:
    """--trace: each turn's host phases from its start (before Chat.ask) to its first token,
    [phase, start, end, cycles] (s from the turn's start; a device run's cycles, else None),
    in the order they end: "template" (Chat._template: the conversation rendered, the prompt's
    and reply_start's), "rollback" (Chat._rollback: to a snapshot, its restore's runs in it),
    "mtp prefill" (MTPDecoder.prefill: the MTP decoder's prompt, a nested one for its first
    conv_k - 1 rows), "prefill_layers" (each call, as `calls`), "end_prefill" (the expert
    server's slots back to decode's, in a call), "snapshot" (Engine.snapshot: the state's
    copy), "mtp snapshot" (MTPDecoder.snapshot: the snapshot and the held row's read), "a0"
    (the first pick: the first token from the prompt's logits; none starts after it) and "run"
    (each device run outside the calls: a snapshot's copies, MTP's first rows, a draft). Host
    bookkeeping: the same calls, runs and tokens."""

    def __init__(self, chat, eng):
        self.t0, self.done, self.out, self.depth = None, False, [], 0
        self.wrap(chat, "_template", "template")
        self.wrap(chat, "_rollback", "rollback")
        self.wrap(eng, "snapshot", "snapshot")
        self.wrap(eng, "prefill_layers", "prefill_layers", call=True)
        srv = getattr(eng, "server", None)
        if srv is not None and hasattr(srv, "end_prefill"):
            self.wrap(srv, "end_prefill", "end_prefill")
        if chat.mtp is not None:
            self.wrap(chat.mtp, "prefill", "mtp prefill")
            self.wrap(chat.mtp, "snapshot", "mtp snapshot")
        chat.pick = _FirstPick(chat.pick, self)
        b = eng.backend
        if hasattr(b, "start"):                 # (the card's: run is start, then wait)
            start, wait, t = b.start, b.wait, [None]

            def started(*a, **kw):
                t[0] = time.perf_counter() if self.live and not self.depth else None
                return start(*a, **kw)

            def waited(*a, **kw):
                st = wait(*a, **kw)
                if t[0] is not None:
                    self.add("run", t[0], time.perf_counter(), (st or {}).get("cycles"))
                    t[0] = None
                return st
            b.start, b.wait = started, waited
        else:
            self.wrap(b, "run", "run", run=True)

    @property
    def live(self) -> bool:
        return self.t0 is not None and not self.done

    def add(self, name: str, t: float, t1: float, cycles=None) -> None:
        self.out.append([name, round(t - self.t0, 6), round(t1 - self.t0, 6), cycles])

    def wrap(self, obj, attr: str, name: str, call: bool = False, run: bool = False) -> None:
        """obj.attr stamped `name` (call: a prefill_layers call, whose runs are its own; run:
        a device run, outside the calls only, its cycles too)."""
        fn = getattr(obj, attr)

        def timed(*a, **kw):
            if not self.live or (run and self.depth):
                return fn(*a, **kw)
            t, r = time.perf_counter(), None
            self.depth += call
            try:
                r = fn(*a, **kw)
                return r
            finally:
                self.depth -= call
                self.add(name, t, time.perf_counter(),
                         r.get("cycles") if run and isinstance(r, dict) else None)
        setattr(obj, attr, timed)

    def begin(self) -> None:
        self.t0, self.done, self.out = time.perf_counter(), False, []

    def end(self) -> list:
        self.t0 = None
        return self.out

    @staticmethod
    def line(phases: list, ttft: float | None) -> str:
        """A turn's phases summed by name, and the rest of its TTFT outside them (the union of
        the phases' spans but HOLDERS')."""
        tot: dict = {}
        for p, a, b, _ in phases:
            n, s = tot.get(p, (0, 0.0))
            tot[p] = (n + 1, s + b - a)
        spans = sorted((a, b) for p, a, b, _ in phases if p not in HOLDERS)
        cov, end = 0.0, 0.0
        for a, b in spans:
            cov += max(0.0, b - max(a, end))
            end = max(end, b)
        rest = "" if ttft is None else f"; the rest of TTFT {max(0.0, ttft - cov):.3f} s"
        return ", ".join(f"{p} {tot[p][1]:.3f} s" + (f" ({tot[p][0]})" if tot[p][0] > 1 else "")
                         for p in PHASES if p in tot) + rest


class _FirstPick:
    """The chat's sampler (its attributes as they are), its first call in a turn stamped "a0"
    (_Phases): the turn's first token."""

    def __init__(self, pick, ph: _Phases):
        self._pick, self._ph = pick, ph

    def __getattr__(self, k):
        return getattr(self._pick, k)

    def __call__(self, *a, **kw):
        ph = self._ph
        if not ph.live:
            return self._pick(*a, **kw)
        t = time.perf_counter()
        try:
            return self._pick(*a, **kw)
        finally:
            ph.add("a0", t, time.perf_counter())
            ph.done = True


def run(model, backend: str = "isa", cfg_file: str | None = None, dev: str | None = None,
        turns=TURNS, max_new: int = 16, cap: int = 2048, pool: str | None = None,
        experts: int = 0, layer_major: int | None = None, sample: bool = False,
        seed: int = 0, create: bool = False, tok=None, em_rows: int | None = None,
        trace: list | None = None, em_pass: int | None = None, mtp: bool = False,
        mtp_from: str | None = None, images=(), max_slices: int | None = None,
        vision_staged: bool | None = None, pool_calls: bool = False,
        tile_rows: int | None = None, phases: list | None = None) -> dict:
    from opentpu.host import chat as C
    from opentpu.host import streamed as ST
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec, model_dir
    path = model_dir(model)
    if tok is None:
        from transformers import AutoTokenizer
        tok = _Pinned(AutoTokenizer.from_pretrained(path))
    spec = load_spec(path)
    wformat, head, formats = ST.defaults(spec)
    be, cfg = C.make_backend(backend, spec, cap, dev, path.name)
    if cfg_file is not None:                    # (the card's: refs.py cfg)
        want = pickle.loads(Path(cfg_file).read_bytes())
        if cfg is not None and cfg != want:
            raise SystemExit(f"the device's configuration is not {cfg_file}'s: {cfg} / {want}")
        cfg = want
    t = time.time()
    eng = ST.engine(path, cfg or board_config(), be, cap, pool or ST.DEFAULT, experts=experts,
                    wformat=wformat, head_format=head, formats=formats, layer_major=layer_major,
                    create=create, snapshots=C.SNAPSHOTS,     # (otpu-chat's: its turns roll back)
                    vision_staged=vision_staged,              # (MiniCPM-V: None, the Spec's)
                    **({} if em_rows is None else {"em_rows": em_rows}),    # (13.16)
                    **({} if em_pass is None else {"em_pass": em_pass}),    # (13.18)
                    **({"mtp": True, "mtp_from": mtp_from} if mtp else {}),    # (15.6)
                    **({} if tile_rows is None else {"tile_rows": tile_rows, "tile_runs": True}))
    load_s = time.time() - t
    if tile_rows is not None and not getattr(eng, "layer_tile", False):
        raise SystemExit("--tile-rows: no tiled expert-major runs on this engine (their mask "
                         "tiles: an image built without OTPU_TILE=0; expert-major prefill)")
    pf, total = _pool(eng)
    if pf is not None and pf.io is None:        # (the reads by where they came from)
        pf.io = {}
    calls: list = []
    if eng.layer_major:                         # (the prompt's calls, each turn's)
        _calls(eng, calls, trace, pool_calls)
    sp = C.sampling(spec, argparse.Namespace(), path)
    pick = C.sampler(sp["temperature"] if sample else 0, sp["top_k"], sp["top_p"], seed,
                     sp["repetition_penalty"], getattr(spec, "softcap", None))
    khz = (getattr(eng.backend, "info", None) or {}).get("core_khz")
    chat = C.Chat(eng, tok, False, pick, max_new, clock_mhz=khz / 1e3 if khz else 0.0)
    chat.max_slices = max_slices
    ph = None if phases is None else _Phases(chat, eng)     # (--trace: each turn's phases)
    out = dict(model=path.name, backend=backend, cfg=cfg_file, sample=sample, seed=seed,
               weights=C.weights_label(eng.image, wformat),
               experts=ST.describe(eng, eng.pool_path), layer_major=eng.layer_major,
               expert_major=eng.expert_major, em_rows=getattr(eng, "em_rows", 0),
               em_pass=getattr(eng, "em_pass", None),
               tile_rows=getattr(eng, "tile_rows", None) if getattr(eng, "layer_tile", False)
               else None, on_card=chat.on_card, load_s=round(load_s, 1), max_slices=max_slices,
               vision_staged=getattr(getattr(eng.image, "vision", None), "resident", True) is False,
               images=[dict(turn=i, path=str(f), sha256=hashlib.sha256(
                   Path(f).read_bytes()).hexdigest()) for i, f in images],
               pool_gb=round(total / 1e9, 2), pool_at_load=_resident(pf, total), turns=[])
    if mtp:
        if not chat.use_mtp:
            raise SystemExit("--mtp: these sampling settings stay on the host (no MTP loop)")
        out.update(mtp=True, mtp_from=mtp_from)
    for k, text in enumerate(turns, 1):
        for i, f in images:
            if i == k:
                chat.attach(str(f))
        n0, c0 = len(chat.fed), len(calls)
        res0, io0 = _resident(pf, total), _io(pf)
        k0 = None if trace is None else len(trace)
        if ph is not None:
            ph.begin()
        reply, tr = chat.ask(text)
        if ph is not None:
            phases.append(dict(ttft_s=tr.ttft_s, calls=[k0, len(trace)] if trace is not None
                               else None, phases=ph.end()))
        res1, reads = _resident(pf, total), _reads(io0, _io(pf))
        r = lambda x, k=3: None if x is None else round(x, k)      # noqa: E731
        out["turns"].append(dict(
            text=text, reply=reply, tokens=list(chat._reply), end=tr.end, fed_before=n0,
            fed=[int(t) for t in chat.fed],
            prompt_tokens=tr.prefill_tokens, reused=tr.reused, restarted=tr.restarted,
            context=tr.context, ttft_s=r(tr.ttft_s),
            prefill_s=r(tr.prefill_s), prefill_tok_s=r(tr.prefill_tok_s, 2),
            prefill_dev_tok_s=r(tr.prefill_dev_tok_s, 2), gen_tokens=tr.gen_tokens,
            decode_tok_s=r(tr.decode_tok_s, 2), decode_dev_tok_s=r(tr.decode_dev_tok_s, 2),
            hits=tr.expert_hits, misses=tr.expert_misses,
            hit_rate=r(tr.expert_hit_rate, 4), mb_per_token=r(tr.streamed_mb_per_token, 2),
            prefill_misses=tr.prefill_misses, images=tr.images, vision_s=r(tr.vision_s),
            calls=calls[c0:], line=tr.line(),
            pool_before=res0, pool_after=res1, reads=reads,
            cold=bool(reads.get("disk")) or any(c["pool_before"] is not None and
                                                c["pool_before"] < 0.95 for c in calls[c0:])
            or (res0 is not None and res0 < 0.95),
            **({"mtp_iters": tr.mtp_iters, "mtp_accepted": tr.mtp_accepted} if mtp else {})))
    out["host_mem"] = ST.host_mem()
    return out


def check(got: dict, ref: dict) -> list:
    """The turns whose tokens differ from the reference's: (turn, first different index)."""
    bad = []
    for i, (a, b) in enumerate(zip(got["turns"], ref["turns"])):
        if a["tokens"] != b["tokens"]:
            bad.append((i, next((j for j, (x, y) in enumerate(zip(a["tokens"], b["tokens"]))
                                 if x != y), min(len(a["tokens"]), len(b["tokens"])))))
    if len(got["turns"]) != len(ref["turns"]):
        bad.append((min(len(got["turns"]), len(ref["turns"])), None))
    if [x["sha256"] for x in got.get("images", [])] != \
            [x["sha256"] for x in ref.get("images", [])]:
        bad.append((None, "images"))
    return bad


def _pct(x) -> str:
    return "?" if x is None else f"{100 * x:.1f}%"


def _rd(reads: dict) -> str:
    return ", ".join(f"{k} {v[0]} ({v[1]} s, {v[2]} MB)"
                     for k, v in sorted(reads.items())) or "none"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--backend", choices=("isa", "board"), default="isa")
    ap.add_argument("--dev", help="the card's device prefix (default: /dev/otpu0 if present, "
                    "else /dev/xdma0)")
    ap.add_argument("--cfg", help="the card's configuration, pickled (tools/qual/refs.py cfg); "
                                  "on the card it must be the device's")
    ap.add_argument("--turn", action="append", help="a user turn (repeated; default TURNS)")
    ap.add_argument("--max-new", type=int, default=16, help="tokens a reply at most")
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--pool", help="the expert pool (default: the checkpoint's, as otpu-chat)")
    ap.add_argument("--pool-create", action="store_true",
                    help="a missing pool is made, each expert packed on its first use (a "
                         "reference on a host with the whole checkpoint and no pool)")
    ap.add_argument("--experts", type=int, default=0, help="slots a MoE layer (0: fill)")
    ap.add_argument("--layer-major", type=int, metavar="R")
    ap.add_argument("--em-rows", type=int, metavar="N",
                    help="expert-major from N new rows a call on (default: the family's, "
                         "streamed.EM_ROWS; 0: every call)")
    ap.add_argument("--em-pass", type=int, metavar="C",
                    help="expert-major's entries a pass (default: the configuration's, "
                         "moe.em_pass_rows)")
    ap.add_argument("--tile-rows", type=int, metavar="T",
                    help="expert-major runs tiled, T rows a run (Engine.tile_rows: the layer's "
                         "weights once for T rows; default 8 for Qwen3.5-MoE, off with "
                         "OTPU_TILE=0)")
    ap.add_argument("--trace", help="each turn's calls' runs and the server's timeline, and its "
                                    "host phases up to the first token (PHASES), JSON")
    ap.add_argument("--mtp", action="store_true",
                    help="Qwen3.5-MoE: decode with the MTP drafter, the loop on the device "
                         "(its pool: the -mtp one, pack_pool.py extend)")
    ap.add_argument("--mtp-from", metavar="DIR",
                    help="the MTP drafter's checkpoint when not the model's (--mtp)")
    ap.add_argument("--image", action="append", default=[], metavar="N:PATH",
                    help="an image for turn N (1-based; repeated): a model with a vision side")
    ap.add_argument("--max-slices", type=int, help="an image's slices at most (default the "
                                                   "processor's)")
    vis = ap.add_mutually_exclusive_group()
    vis.add_argument("--vision-staged", action="store_true",
                     help="MiniCPM-V's tower weights staged through the expert slots "
                          "(minicpmv.Spec.vision_staged, the default; the pool grows by their "
                          "entries)")
    vis.add_argument("--vision-resident", action="store_true",
                     help="MiniCPM-V's tower weights resident in the image")
    ap.add_argument("--pool-calls", action="store_true",
                    help="the pool's page cache before and after each call too (mincore: "
                         "60-90 ms a check of a 17.5 GB pool, in the turn's wall times)")
    ap.add_argument("--sample", action="store_true", help="the model's sampling defaults")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", help="a reference run's JSON: its tokens, turn by turn")
    ap.add_argument("--out", help="this run's JSON")
    a = ap.parse_args(argv)
    tr, ph = ([], []) if a.trace else (None, None)
    images = []
    for x in a.image:
        n, _, f = x.partition(":")
        if not n.isdigit() or not Path(f).is_file():
            ap.error(f"--image {x}: N:PATH, PATH a file")
        images.append((int(n), Path(f)))
    r = run(a.model, a.backend, a.cfg, a.dev, a.turn or TURNS, a.max_new, a.cap, a.pool,
            a.experts, a.layer_major, a.sample, a.seed, a.pool_create, em_rows=a.em_rows,
            trace=tr, em_pass=a.em_pass, mtp=a.mtp or a.mtp_from is not None,
            mtp_from=a.mtp_from, images=images, max_slices=a.max_slices,
            vision_staged=True if a.vision_staged else False if a.vision_resident else None,
            pool_calls=a.pool_calls, tile_rows=a.tile_rows, phases=ph)
    for i, t in enumerate(r["turns"]):
        print(f"turn {i + 1}: {t['reply']!r}\n  {t['line']}", flush=True)
        print(f"  pool: {_pct(t['pool_before'])} -> {_pct(t['pool_after'])} of its "
              f"{r['pool_gb']} GB in the page cache; reads {_rd(t['reads'])}"
              + ("; COLD (misses read from the disk)" if t["cold"] else ""), flush=True)
        for c in t["calls"]:
            print(f"    call: {c['rows']} rows {'expert' if c['em'] else 'layer'}-major, "
                  f"{c['s']} s (device {c['device_s']}), {c['runs']} runs; misses "
                  f"{c['misses']} ({c['requests']} requests, served in {c['serve_s']} s), needs "
                  f"{c['needs_queued']} ({c['need_lines']} lines), parts {c['prefetched']} "
                  f"({c['parts_s']} s), holds {c['holds']}; pool {_pct(c['pool_before'])} -> "
                  f"{_pct(c['pool_after'])} in the page cache, reads {_rd(c['reads'])}",
                  flush=True)
        if ph is not None:
            print(f"    phases: {_Phases.line(ph[i]['phases'], t['ttft_s'])}", flush=True)
    if a.trace:
        Path(a.trace).write_text(json.dumps(dict(
            run_fields=["key", "start", "end", "cycles"],
            event_fields=["seen", "done", "kind", "layer or expert", "misses or bytes"],
            calls=tr, phase_fields=["phase", "start", "end", "cycles"], turns=ph)))
    rc = 0
    if a.check:
        ref = json.loads(Path(a.check).read_text())
        bad = check(r, ref)
        r["check"] = dict(ref=a.check, ok=not bad, differ=bad)
        print(f"[{'FAIL' if bad else 'PASS'}] {r['model']} against {a.check}: " +
              (", ".join("other images" if i is None else f"turn {i + 1} from token {j}"
                         for i, j in bad) if bad else
               f"{len(r['turns'])} turns' tokens the same"), flush=True)
        rc = 1 if bad else 0
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
