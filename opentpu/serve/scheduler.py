"""The device's worker (otpu-serve): requests from any number of clients, one model on the card,
each request's context switched in and out of the device's KV cache.

One thread owns the Engine. A request is a prompt's token ids (the API renders the chat), its
sampling and its stop conditions; the worker admits up to `max_running` at a time and runs them
in turns:

- a new request's prompt continues the longest prefix the device can take back: what the
  device holds (a common prefix where the KV caches hold the whole context; where there is
  state, a context the prompt continues), or a context saved in the KV store
  (opentpu/serve/store.py: host RAM, then disk), whichever is longer; only the rest is fed
  (prefill), then the first token is picked from the last row's logits;
- a running request decodes a quantum of tokens per turn, on the card's decode loop
  (Engine.generate_card: greedy or with the request's sampling, its uniforms from the request's
  own generator) or, for sampling the device does not take, step by step on the host;
- before the device's context is overwritten it is saved to the store (its new pages and its
  tail; an active request's pinned until it ends), so that the request it belongs to, or a
  later one sharing its prefix, continues it instead of feeding it again. A finished request's
  is saved as soon as the queue is empty (the worker idle), so a request that comes later has
  nothing to save before its prompt;
- a model with state (Request.at, a chat's reply start) keeps the context there for the next
  turn: a snapshot on the device (Engine(snapshots=), as otpu-chat's), which a next turn
  continuing it rolls back to (no store traffic) and which goes to the store (the device rolled
  back to it, then saved) once the worker is idle or before another context takes the device;
  without snapshots the context there is saved as the prompt is fed. The turn past the reply
  start is not saved: the next turn's rendering parts there;
- a request that continues the context an idle save is writing (a chat's next turn, sent at
  once) pre-empts it: the save gives up between two device reads (nothing stored, the device
  as it was) and the request goes on from the device; the next idle saves what the device then
  holds. Any other request waits for the save, which its switch would make anyway;
- a turn whose run fails on x2's link (cards.LinkError) resets the link and resumes its request
  from its prompt and its output so far (Worker._link_recovered, as otpu-chat35x2's turn), once
  a request; any other failure ends the request in error.

New requests come first (their prefill), then the running ones in turn. While one request runs
alone, it runs to its end; the card's loop is halted after the token in flight as soon as
another request arrives (stop(), polled while the card runs).

With mtp (an MTP engine, opentpu/llm/mtp.py; otpu-serve --mtp) every request decodes with the
MTP drafter, its loop on the device (MTPDecoder.loop_card): the prompt goes through the
decoder's prefill (the MTP layer's K / V rows with it; the first token picked on the host from
the last row's logits as the device would, its draft left to the loop's first run), and a
context's bytes hold the MTP layer's KV cache and the committed slot's state (kvio). A context
saved for its request to come back to keeps its draft (Request.draft); one saved before the
reply's start (Request.at) keeps the residual row before it, so that a prompt going on there with
another token appends the MTP layer's row again (MTPDecoder.refill; store.PrefixCache's
follow and held). Only sampling the device's loop takes is served.
"""
from __future__ import annotations

import itertools
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from ..host.chat import Detok, sampler as host_sampler
from ..llm import generate as G
from ..llm.mtp import MTPStats
from . import kvio as KV
from . import telemetry as TM
from . import store as ST

STATS_KEEP = 1024       # the engine's per-run stats kept
PREFIX_BLOCK = 2048     # a model with state: a prompt's prefix kept every this many positions
PREFIX_SKIP = 256       # ... but none this close to where its feed starts
QUANTUM = 32            # tokens a running request decodes per turn while others wait
CHUNK = 64              # tokens per decode run on a backend that does not poll stop() mid-run


@dataclass
class Sampling:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0 and self.repetition_penalty == 1.0


@dataclass(eq=False)                    # (identity: requests are equal only to themselves)
class Request:
    """One generation. prompt: the token ids fed; at: where a model with state keeps a saved
    context in the prompt (a chat's reply start: the next turn's rendering parts there), or
    None; marks: where it keeps the prompt's prefix in the store as it is fed, for other
    conversations (the last message boundary before the final user turn: a system prompt's
    end; Worker.prefix_block's block positions too); keep_reply: a reply whose text holds it
    stays on the device past the idle save or a switch (a tool call: the agent's next request
    continues it; "": every reply, the API's with the thinking kept), its reply start saved
    from the snapshot (at may then be the prompt's last fed position); None: none; stop_ids
    end it (not part of the output);
    stop_strs end it at the text's first match (cut there)."""
    prompt: list
    max_new: int = 256
    sampling: Sampling = field(default_factory=Sampling)
    stop_ids: frozenset = frozenset()
    stop_strs: tuple = ()
    at: int | None = None
    marks: tuple = ()
    keep_reply: str | None = None
    rid: str = ""
    skip_special: bool = True         # the text without the tokenizer's special tokens
    images: list = field(default_factory=list)  # its images (minicpmv.ImageInput, in order):
    #                                   their tokens in prompt are their keys (minicpmv.keyed)
    # progress (the worker's)
    out: list = field(default_factory=list)
    text: str = ""
    finish: str | None = None         # "stop", "length", "abort", "error"
    error: str | None = None
    cached: int = 0                   # prompt tokens the device did not have to feed
    prefill_tokens: int = 0
    t_submit: float = 0.0
    t_start: float | None = None      # the worker's first turn of it (its wait in the queue ends)
    t_first: float | None = None
    t_done: float | None = None
    prefill_s: float = 0.0            # seconds its prompt was fed (its turns' prefill)
    # seconds of its turns in the worker's phases: "switch" (the device made its context: within
    # it "save" (another context into the store), "restore" (its own written back), "rollback"
    # (the device back to a snapshot)), "snapshot" (its reply start kept on the device) and
    # "save" too where its reply start went to the store instead; "idle_save": of its queue time,
    # what it waited for an idle save. A save's and a restore's parts: "save.<part>" and
    # "restore.<part>" (_io_parts: store, host, prog, link, load, run, adopt; gc within them)
    phases: dict = field(default_factory=dict)
    # the bytes its saves and restores moved: {"save" | "restore": {link_bytes, link_calls,
    # copies, copied_bytes}} (kvio.ContextIO.moved)
    io: dict = field(default_factory=dict)
    turns: int = 0
    swaps: int = 0
    events: queue.Queue = field(default_factory=queue.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    # the worker's state of it
    fed: list = field(default_factory=list)      # tokens in its context (the device's or saved)
    next: int | None = None           # its next token: picked, not fed
    entry: object = None              # its saved context while it is off the device (pinned)
    slot: int | None = None           # its KV slot (batching.BatchWorker)
    draft: int | None = None          # MTP: the draft after `next` while it is off the device
    mtp_iters: int = 0                # MTP: its loop's iterations and accepted drafts
    mtp_accepted: int = 0
    _bset: bool = False               # its slot's sampling words written (BatchWorker)
    _shown: int = 0                   # chars of text sent
    _detok: object = None
    _pick: object = None

    def cancel(self) -> None:
        self.cancelled.set()

    def stream(self, timeout: float | None = None):
        """The text deltas as they come; ends when the request does."""
        while True:
            kind, val = self.events.get(timeout=timeout)
            if kind == "delta":
                yield val
            else:
                return

    def wait(self, timeout: float | None = None) -> "Request":
        for _ in self.stream(timeout):
            pass
        return self


class Worker:
    """The device's thread: Worker(engine, tokenizer, ...).start(); submit(request). store: a
    store.BlobStore for the saved contexts (None: host RAM of `ram_bytes`); model_key: what
    makes the contexts' bytes (store.model_key)."""

    def __init__(self, eng, tok, store=None, model_key: str = "model",
                 max_running: int = 8, quantum: int = QUANTUM, ram_bytes: int = 4 << 30,
                 area=False, mtp: bool = False, prefix_block: int = PREFIX_BLOCK,
                 prefix_skip: int = PREFIX_SKIP):
        self.eng, self.tok = eng, tok
        # a model with state: a prompt's prefix kept in the store as it is fed, at its marks
        # and every prefix_block positions (0: neither), none within prefix_skip of where the
        # feed starts (_prefill, _keep_prefix)
        self.prefix_block, self.prefix_skip = prefix_block, prefix_skip
        self.tok_lock = threading.Lock()
        self.io = KV.ContextIO(eng, area=area)      # (False: kvio.staging_area's)
        self.last_io: dict = {}         # the last save's and restore's parts (_io_parts)
        self.store = store if store is not None else ST.BlobStore(ram_bytes)
        self.dec = None                   # the MTP decoder (mtp=True)
        if mtp:
            from ..llm.mtp import MTPDecoder
            if not getattr(eng, "can_generate", False):
                raise ValueError("MTP decoding runs the device's loop: an engine that has it")
            self.dec = MTPDecoder(eng)
        self.follow: int | None = None    # MTP: the token the device context's last MTP row
        #                                   took (None: not known)
        self.cache = ST.PrefixCache(self.store, self.io.layout, model_key, mtp=mtp)
        self.max_running, self.quantum = max_running, quantum
        self.stateful = self.io.layout.stateful
        self.softcap = getattr(eng.spec, "softcap", None)
        self.vision = None                # minicpmv.Vision, at the first image
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.cv = threading.Condition()
        self.dev: list = []               # the tokens the device's context holds
        self.dirty = False                # ... not saved in the store yet
        self.snap_at: int | None = None   # a snapshot on the device of dev[:snap_at]'s state (a
        #                                   request's reply start), not in the store yet
        self.reply_kept = False           # the device's context ends in a reply its next request
        #                                   continues (Request.keep_reply): the idle save keeps it
        self._flush_gen = 0               # flush()es so far; _drop_dev: one asked the worker to
        self._drop_dev = False            # forget the device's context
        self._idling = None               # the context an idle save writes, while it does; _pre:
        self._pre = (None, False)         # (the next request, whether it pre-empts that save)
        self.owner: Request | None = None
        self._rr = 0
        self._ids = itertools.count(1)
        self._stop = False
        self.thread = None
        # monitors (opentpu/serve/metrics.py): observers(r) on the worker's thread as each
        # request ends (cheap: they queue a record); on_idle() there once the queue is empty
        # (chat35's layer runs ahead); feeding (r, the device's position, tokens to feed,
        # perf_counter, stats' prefill_tokens) while a prompt is fed: its progress is the
        # device's position past it
        self.observers: list = []
        self.on_idle = None
        self.feeding: tuple | None = None
        self.stats = {"requests": 0, "finished": 0, "aborted": 0, "errors": 0,
                      "prompt_tokens": 0, "cached_tokens": 0, "prefill_tokens": 0,
                      "generated": 0, "decode_s": 0.0, "prefill_s": 0.0, "switch_s": 0.0,
                      "saves": 0, "restores": 0, "device_reuses": 0, "turns": 0,
                      "restore_s": 0.0, "save_s": 0.0, "snapshots": 0, "rollbacks": 0,
                      "idle_saves": 0, "idle_preempts": 0, "prefix_saves": 0,
                      "prefix_save_s": 0.0}
        if mtp:
            self.stats.update(mtp_iters=0, mtp_accepted=0, refills=0)

    # ---- the clients' side
    def start(self) -> "Worker":
        self.thread = threading.Thread(target=self._loop, daemon=True, name="otpu-serve-worker")
        self.thread.start()
        return self

    def stop(self, persist: bool = True) -> None:
        """End the worker (requests in flight aborted); persist: the device's context and the
        store's RAM onto its disk, for a later server."""
        with self.cv:
            self._stop = True
            self.cv.notify_all()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if persist and self.store.dir is not None:
            try:
                self._save_at()
                self._save_device()
            except BaseException:
                traceback.print_exc()
            self.store.persist()

    def submit(self, r: Request) -> Request:
        if not r.prompt:
            raise ValueError("an empty prompt")
        if len(r.prompt) >= self.eng.cap:
            raise ValueError(f"the prompt's {len(r.prompt)} tokens leave no room in the "
                             f"context of {self.eng.cap}")
        if r.images and getattr(getattr(self.eng, "image", None), "vision", None) is None:
            raise ValueError("this model takes no images")
        if sum(1 for t in r.prompt if t < 0) != sum(im.tokens for im in r.images):
            raise ValueError("token ids are >= 0 (an image's tokens: its keys, minicpmv.keyed)")
        r.rid = r.rid or f"r{next(self._ids)}"
        r.stop_ids = frozenset(r.stop_ids) or frozenset(self.eng.spec.eos)
        if len(r.stop_ids) > G.N_STOP:
            raise ValueError(f"{len(r.stop_ids)} stop token ids: the card's loop takes "
                             f"{G.N_STOP}")
        if self.dec is not None and not self._card_picks(r):
            raise ValueError("this server decodes with MTP on the device: sampling with top_k "
                             f"1 .. {G.K_MAX} (or greedy) and repetition_penalty >= 1 only")
        r.t_submit = time.perf_counter()
        r.max_new = max(0, min(r.max_new, self.eng.cap - len(r.prompt)))
        with self.cv:
            self.waiting.append(r)
            self.stats["requests"] += 1
            self.cv.notify_all()
        return r

    def flush(self) -> None:
        """Forget the saved contexts and the device's (the next request starts from what the
        store and the device do not hold). The device's context is the worker's to drop (its
        idle save may be running): it does before its next request; a save that ran across
        the flush is cleared again (_idle)."""
        with self.cv:
            self._flush_gen += 1
            if not self.running:
                self._drop_dev = True
            self.cv.notify_all()
        self.cache.clear()

    def busy(self) -> int:
        with self.cv:
            return len(self.waiting) + len(self.running)

    # ---- the loop
    def _trim(self) -> None:
        """The engine's per-run stats, the last STATS_KEEP (a server runs for days; between
        iterations, so no engine call is slicing them)."""
        st = getattr(self.eng, "stats", None)
        if isinstance(st, list) and len(st) > 4 * STATS_KEEP:
            del st[:-STATS_KEEP]

    def _loop(self) -> None:
        while True:
            self._trim()
            with self.cv:
                while not self._stop and not self.waiting and not self.running:
                    self._dropped()
                    self.cv.wait()
                if self._stop:
                    for r in self.waiting + self.running:
                        self._end(r, "abort")
                    return
                self._dropped()
                r = self._pick()
            if r is None:
                continue
            try:
                self._turn(r)
            except BaseException as e:      # the request fails, the server goes on
                if self._link_recovered(e, r):     # (the link reset: r resumes)
                    continue
                r.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                self.stats["errors"] += 1
                self.owner, self.dev, self.dirty, self.snap_at = None, [], False, None
                try:
                    self.eng.reset()
                    self._mtp_reset()
                except BaseException:
                    pass
                self._end(r, "error")
            if not self.busy():
                self._idle()

    def _dropped(self) -> None:
        """flush()'s: the device's context forgotten (under cv, no request running)."""
        if self._drop_dev and not self.running:
            self.dev, self.dirty, self.snap_at, self._drop_dev = [], False, None, False

    def _idle(self) -> None:
        """The queue empty after a turn: the device's context into the store while nothing
        waits (a snapshot's: the reply start, the device back to it, or with a reply kept
        (reply_kept) the reply start and then the whole context, the device where it is; else
        the context itself; MTP's left to the next switch), so the next request's switch has
        nothing to save; then
        on_idle (chat35's layer runs ahead). A request that comes meanwhile and continues that
        context pre-empts the save (_preempts); any other waits for it (its phases'
        "idle_save")."""
        if self.dec is None and self.owner is None and (self.snap_at is not None or self.dirty):
            g = self._flush_gen
            t0 = time.perf_counter()
            try:
                with self.cv:
                    self._dropped()
                    self._idling = list(self.dev[:self.snap_at] if self.snap_at is not None
                                        and not self.reply_kept else self.dev)
                    self._pre = (None, False)
                saved = self.stats["saves"]
                self.io.abort = self._preempts
                if not self._preempts():
                    if self.reply_kept:
                        self._save_reply()
                    else:
                        self._save_at()
                    self._save_device()
                else:
                    self.stats["idle_preempts"] += 1
                self.stats["idle_saves"] += self.stats["saves"] - saved
            except KV.Aborted:                  # (rolled back, unsaved: dirty, as it was)
                self.stats["idle_preempts"] += 1
            except BaseException:
                traceback.print_exc()
                self.dev, self.dirty, self.snap_at = [], False, None
                try:
                    self.eng.reset()
                except BaseException:
                    pass
            finally:
                self.io.abort = None
                t1 = time.perf_counter()
                with self.cv:
                    self._idling = None
                    for r in self.waiting:      # (the requests it held up)
                        r.phases["idle_save"] = r.phases.get("idle_save", 0.0) + \
                            t1 - max(t0, r.t_submit)
            if self._flush_gen != g:            # (a flush meanwhile: what this saved too)
                self.cache.clear()
        if self.on_idle is not None and not self.busy():
            try:
                self.on_idle()
            except BaseException:
                traceback.print_exc()

    def _preempts(self) -> bool:
        """Whether the next request continues the context the idle save writes (its prompt
        goes on from all of it: a chat's next turn), so it needs no save first: the save gives
        up (between its device reads: kvio's abort) and the request goes first. Asked once a
        request (the next one, as _pick takes them)."""
        with self.cv:
            r = self.waiting[0] if self.waiting else None
        if r is None or self._idling is None:
            return False
        if self._pre[0] is not r:
            n = len(self._idling)
            self._pre = (r, len(r.prompt) > n and self._common(self._idling, r.prompt) == n)
        return self._pre[1]

    def _pick(self) -> Request | None:
        """New requests first (up to max_running running), then the running ones in turn."""
        for r in [r for r in self.running if r.cancelled.is_set()]:
            self._end(r, "abort")
        for r in [r for r in self.waiting if r.cancelled.is_set()]:
            self._end(r, "abort")
        if self.waiting and len(self.running) < self.max_running:
            r = self.waiting.pop(0)
            self.running.append(r)
            return r
        if not self.running:
            return None
        self._rr = (self._rr + 1) % len(self.running)
        if self.owner in self.running and len(self.running) == 1:
            return self.owner
        return self.running[self._rr]

    def _others(self, r: Request) -> bool:
        """Another request waits for the device (no lock: polled while the card runs)."""
        return bool(self.waiting) or len(self.running) > 1

    def _link_recovered(self, e, r: Request | None = None) -> bool:
        """After r's turn failed on the link (cards.LinkError: x2's two cards), as chat35x2's
        turn: the link reset (CardsTransport.recover: the cores stopped, LRST) and the device's
        context dropped (the run may have written past it); r, and a request whose context only
        the device held (a save the failure cut short), resume with the prompt and the output
        so far as the context to make (_want: the store's longest match, the rest fed again),
        the others from the store as they left. Once a request; none with the MTP decoder
        (chat35x2's). False: not a link's failure, or no recovery (the error path's)."""
        from ..host.cards import LinkError
        t = getattr(getattr(self.eng.backend, "board", None), "t", None)
        if r is None or not isinstance(e, LinkError) or not hasattr(t, "recover") or \
                self.dec is not None or getattr(r, "_relinked", False):
            return False
        traceback.print_exc()
        try:
            t.recover()
        except BaseException:
            traceback.print_exc()
            return False
        self.stats["link_recoveries"] = self.stats.get("link_recoveries", 0) + 1
        self.owner, self.dev, self.dirty, self.snap_at = None, [], False, None
        self.reply_kept = False
        self.eng.reset()
        for x in [r] + [x for x in self.running if x is not r and x.finish is None and x.fed
                        and x.entry is None]:
            if x.entry is not None:
                self.cache.unpin(x.entry)
                x.entry = None
            x._relinked, x.fed, x.next = True, [], None
        if r.finish is not None:
            self._end(r, r.finish)
        return True

    @staticmethod
    def _want(r: Request) -> list:
        """The context a request's feed makes, its last token fed by its first step: its prompt,
        or after a link recovery its prompt and its output so far."""
        return list(r.prompt) + list(r.out) if getattr(r, "_relinked", False) else r.prompt

    def _turn(self, r: Request) -> None:
        r.turns += 1
        self.stats["turns"] += 1
        t0 = time.perf_counter()
        if r.t_start is None:
            r.t_start = t0
        self._switch_to(r)
        self.stats["switch_s"] += time.perf_counter() - t0
        if r.next is None and r.finish is None:
            self._feed(r)
        if r.finish is None and r.next is not None:
            self._decode(r)
        if r.finish is not None:
            self._end(r, r.finish)

    # ---- contexts
    @staticmethod
    def _common(a, b) -> int:
        n = min(len(a), len(b))
        if not n:
            return 0
        eq = np.asarray(a[:n], np.int64) == np.asarray(b[:n], np.int64)
        return n if eq.all() else int(np.argmin(eq))

    @staticmethod
    def _phase(r: Request | None, k: str, t0: float) -> float:
        dt = time.perf_counter() - t0
        if r is not None:
            r.phases[k] = r.phases.get(k, 0.0) + dt
        return dt

    def _io_mark(self, io=None) -> tuple:
        """kvio's counters now, for a phase's parts (_io_parts). io: a slot's (default self.io)."""
        io = self.io if io is None else io
        return dict(io.t), dict(io.moved), KV.gc_seconds()

    def _io_parts(self, r: Request | None, kind: str, total: float, mark: tuple, io=None,
                  **more: float) -> None:
        """A save's or restore's `total` seconds by part since `mark` into r.phases as
        kind.part: kvio's prog, link, load and run (kvio.ContextIO.t), "host" kvio's own host
        work, more's (adopt), "store" the rest (the store's lookup or insert: its hashes and
        copies on the host; a save's settle); within them (not added) "gc" the GC's seconds and
        "telemetry" a card sample's (telemetry.overlap: its thread holds the interpreter).
        The bytes and runs into r.io[kind]. The last of each kind, a request's or not (an idle
        save), in last_io[kind] too (/api/stats' kv). io: _io_mark's."""
        t0, m0, g0 = mark
        io = self.io if io is None else io
        t, m = io.t, io.moved
        parts = {k: t[k] - t0[k] for k in KV.PARTS}
        io_s = t["io"] - t0["io"]
        parts["host"] = max(0.0, io_s - sum(parts.values()))
        parts.update(more)
        parts["store"] = max(0.0, total - io_s - sum(more.values()))
        parts["gc"] = KV.gc_seconds() - g0
        now = time.perf_counter()
        parts["telemetry"] = TM.overlap(now - total, now)
        moved = {k: m[k] - m0[k] for k in m}
        self.last_io[kind] = {"s": round(total, 4), **{k: round(v, 4) for k, v in parts.items()},
                              **moved}
        if r is None:
            return
        for k, v in parts.items():
            r.phases[f"{kind}.{k}"] = r.phases.get(f"{kind}.{k}", 0.0) + v
        mv = r.io.setdefault(kind, dict.fromkeys(m, 0))
        for k, v in moved.items():
            mv[k] += v

    def _save_device(self, pin_for: Request | None = None, r: Request | None = None,
                     state_at: int | None = None) -> None:
        """The device's context into the store (what the store lacks of it). MTP: a pending
        draft settled first (the MTP layer's last row), the request's draft kept with it. r:
        the request whose turn it delays (its phases' "save"). state_at: a snapshot holding the
        context's state (kvio.snapshot_at), read from there."""
        if not self.dev or (not self.dirty and pin_for is None):
            return
        t0, mark = time.perf_counter(), self._io_mark()
        self._settle()
        e = self.cache.save(self.io, list(self.dev), pin=pin_for is not None, follow=self.follow,
                            state_at=state_at)
        self.stats["saves"] += 1
        dt = self._phase(r, "save", t0)
        self.stats["save_s"] += dt
        self._io_parts(r, "save", dt, mark)
        self.dirty = False
        if pin_for is not None:
            pin_for.entry = e
            if self.dec is not None:
                pin_for.draft = self.dec.draft

    def _rollback(self, r: Request | None) -> int:
        """The device back to its snapshot (Engine.rollback): dev[:snap_at], unsaved."""
        k, self.snap_at = self.snap_at, None
        t0 = time.perf_counter()
        t = self.eng.rollback(k)
        self._phase(r, "rollback", t0)
        self.stats["rollbacks"] += 1
        self.dev, self.dirty = list(self.dev[:t]), t > 0  # (a new list: dev is a request's fed)
        return t

    def _save_at(self, r: Request | None = None) -> None:
        """The snapshot's context (a request's reply start) into the store: the device rolled
        back to it and saved (it holds that context then, saved), the state read from the
        snapshot (one range, no copy); what lay past it is dropped (a chat's reply, which its
        next turn's rendering does not repeat)."""
        if self.snap_at is not None:
            t = self._rollback(r)
            at = KV.snapshot_at(self.eng, t) if self.io.slot == 0 else None
            self._save_device(r=r, state_at=at)

    def _save_reply(self, r: Request | None = None) -> None:
        """The snapshot's context (the reply start) into the store, its state read from the
        snapshot, the device left past it, unsaved (a reply kept: the next request continues
        it, or parts at the reply start and restores this); a snapshot it cannot read:
        _save_at. r: the request whose turn it delays (its phases' "save")."""
        k = self.snap_at
        at = KV.snapshot_at(self.eng, k) if k is not None and self.io.slot == 0 else None
        if at is None:
            return self._save_at(r)
        t0, mark = time.perf_counter(), self._io_mark()
        self.cache.save(self.io, list(self.dev[:k]), state_at=at)
        self.snap_at = None
        self.stats["saves"] += 1
        dt = self._phase(r, "save", t0)
        self.stats["save_s"] += dt
        self._io_parts(r, "save", dt, mark)

    def _switch_to(self, r: Request) -> None:
        """Make the device's context r's: a running request's own (as it left it), or for a new
        one the longest prefix of its prompt the device or the store has."""
        if self.owner is r:
            return
        t0 = time.perf_counter()
        try:
            self._switch(r)
        finally:
            self._phase(r, "switch", t0)

    def _switch(self, r: Request) -> None:
        old, kept = self.owner, self.reply_kept
        self.reply_kept = False
        if old is not None and old.finish is None and old.fed:
            old.swaps += 1
            self._save_device(pin_for=old, r=r)     # it comes back to this
        self.owner = None
        if self.snap_at is not None:
            # a snapshot's context: r continues it past the snapshot (a chat's next turn): the
            # device back to it; r continues the device's whole context: kept (the snapshot
            # dropped); else into the store before r's context takes the device (a reply kept:
            # the reply start, then the whole context below)
            c = self._common(self.dev, r.fed or r.prompt)
            if not r.fed and self.snap_at <= c < len(self.dev):
                self._rollback(r)
            elif (r.fed or c < len(self.dev)) and kept:
                self._save_reply(r)
            elif r.fed or c < len(self.dev):
                self._save_at(r)
        c = self._common(self.dev, r.fed or r.prompt)
        if r.fed:                               # r ran before: exactly its context
            want, m = r.fed, ST.Match(len(r.fed), r.entry)
            exact = len(want)
            dev_k = exact if c == exact and (not self.stateful or len(self.dev) == exact) \
                else 0
        else:
            want = self._want(r)
            limit = len(want) - 1               # (a token to feed: its logits)
            m = self.cache.match(want, limit=limit)
            if self.stateful:                   # the device's context, continued
                dev_k = len(self.dev) if c == len(self.dev) <= limit else 0
            else:                               # any common prefix
                dev_k = min(c, limit)
        if dev_k > 0 and dev_k >= m.k:          # the device holds it
            if dev_k < len(self.dev):
                self._save_device(r=r)          # (what lies past it is overwritten)
            self.snap_at = None                 # (adopt drops the snapshots)
            self.eng.adopt(dev_k)
            k = dev_k
            self.stats["device_reuses"] += 1
            if self.dec is not None and r.fed and r.draft is not None:
                self.dec.draft = r.draft
        else:
            self._save_device(r=r)
            self.snap_at = None
            t0, mark = time.perf_counter(), self._io_mark()
            if self.dec is not None:            # (the state into slot 0)
                self.io.slot = 0
            k = self.cache.restore(self.io, m, have=c) if m.k else 0
            if k:
                ta = time.perf_counter()
                self.eng.adopt(k)               # (precompiles in flight waited for)
                ta = time.perf_counter() - ta
                self.stats["restores"] += 1
                dt = self._phase(r, "restore", t0)
                self.stats["restore_s"] += dt
                self._io_parts(r, "restore", dt, mark, adopt=ta)
                self._mtp_restored(r, m.entry, want, k)
            else:
                self.eng.reset()
                self._mtp_reset()
        if r.fed:
            if r.entry is not None:
                self.cache.unpin(r.entry)
                r.entry = None
            if k != len(r.fed):                 # its saved context is gone (dropped): again
                r.fed, r.next = [], None
                self.eng.reset()
                self._mtp_reset()
                k = 0
        else:
            r.fed = list(want[:k])
            if want is r.prompt:                # (a link's resume: its first match counted)
                r.cached = k
                self.stats["cached_tokens"] += k
        self.dev = r.fed                        # (the same list: it grows as r runs)
        self.dirty = True
        self.owner = r
        self._rope(r)
        if self.eng.pos != len(r.fed):
            raise RuntimeError(f"the device is at {self.eng.pos}, the context has {len(r.fed)}")

    # ---- the MTP decoder's state (mtp)
    def _mtp_reset(self) -> None:
        """A new context: the decoder's committed slot 0, nothing pending."""
        d = self.dec
        if d is not None:
            d.slot, d.draft, d.pending, d.refill, d.held = 0, None, None, None, {}
            self.follow = None

    def _mtp_restored(self, r: Request, e, want, k: int) -> None:
        """The device's context restored from entry e (its state in slot 0): r's own (its
        draft back), or the first k tokens of r's prompt: the MTP layer's row at k - 1 took
        e.follow; a prompt going on with another token appends it again from the held residual
        row (MTPDecoder.refill), where the entry keeps one."""
        d = self.dec
        if d is None:
            return
        d.slot, d.pending, d.refill, d.held = 0, None, None, {}
        if r.fed:
            d.draft, self.follow = r.draft, r.next
            return
        d.draft, self.follow = None, e.follow
        if e.follow != int(self._real(r, list(want[k:k + 1]))[0]):
            h = self.cache.held_row(e)
            if h is not None and d.holds:
                d.refill = (k - 1, h)
                self.stats["refills"] += 1

    def _settle(self) -> None:
        """MTP: a lazy prefill's pending draft run (the MTP layer's row at the prompt's last
        position, the first token's), so the device's context is whole; the state's slot for
        kvio."""
        d = self.dec
        if d is None:
            return
        if d.pending is not None:
            self.follow = d.pending[1]
            d._settle()
        self.io.slot = d.slot

    # ---- images (MiniCPM-V)
    def _real(self, r: Request, keys) -> list:
        """The token ids of r's keys (its images' tokens: the image token)."""
        if not r.images:
            return keys
        from ..llm import minicpmv as MV
        return MV.unkeyed(keys, self.eng.spec.vision.image_token)

    def _rope(self, r: Request) -> None:
        """The RoPE table's rows for r's context: its images' M-RoPE positions (and s + delta
        past its prompt, the decode's), or the table's own rows (a text-only one, after an
        image's). Only the rows that differ are written."""
        if not r.images and self.vision is None:
            return
        if self.vision is None:
            from ..llm import minicpmv as MV
            self.vision = MV.Vision(self.eng)
        if r.images:
            self.vision.positions(self._real(r, r.prompt),
                                  [s for im in r.images for s in im.sizes])
        else:
            self.vision.pristine()

    # ---- prefill and decode
    def _sampler(self, r: Request):
        if r._pick is None:
            s = r.sampling
            r._pick = host_sampler(0 if s.greedy else s.temperature, s.top_k, s.top_p,
                                   s.seed, s.repetition_penalty, self.softcap)
        return r._pick

    def _card_picks(self, r: Request) -> bool:
        """The device's decode loop picks r's tokens (greedy, or a sampling it takes)."""
        if not getattr(self.eng, "can_generate", False):
            return False
        return r.sampling.greedy or G.Sampling.fits(**self._sampler(r).params)

    def _feed(self, r: Request) -> None:
        """_prefill, timed (r.prefill_s) and watched (feeding: a monitor reads the device's
        position for the prompt's progress, a layer-major prompt's too, which yields once)."""
        t0 = time.perf_counter()
        self.feeding = (r, self.eng.pos, len(self._want(r)) - len(r.fed), t0,
                        self.stats["prefill_tokens"])
        try:
            self._prefill(r)
        finally:
            self.feeding = None
            r.prefill_s += time.perf_counter() - t0

    def _count_runs(self, k: int | None) -> int | None:
        """The engine's device runs from its stats entry k on into stats' prefill_runs (a
        layer-major prompt's entry: its "runs", a layer loop's runs in a bucket one; another
        entry a run); its entry count now (None: an engine without stats). The key from the
        first count on: a worker that feeds its prompts otherwise (BatchWorker) shows none."""
        est = getattr(self.eng, "stats", None)
        if not isinstance(est, list):
            return None
        if k is not None:
            self.stats["prefill_runs"] = self.stats.get("prefill_runs", 0) + sum(
                (e or {}).get("runs", 1) for e in est[k:])
        return len(est)

    def _prefill(self, r: Request) -> None:
        """Feed r's prompt from r.fed on. When the device's loop picks r's tokens, all but its
        last token, which becomes r.next: the decode loop feeds it and picks the first token
        on the device (no logits to the host); else the host picks it from the logits."""
        if self.dec is not None:
            return self._prefill_mtp(r)
        t0 = time.perf_counter()
        ids = self._want(r)                     # (its key: with images, their keys)
        real = self._real(r, ids)
        n = len(r.fed)
        vis = self.vision.prompt(real, r.images, n) if r.images else None
        last = len(ids) - 1 if self._card_picks(r) else len(ids)
        end = last + (r.keep_reply is not None)     # (a reply kept: its start may be the last)
        at = r.at if self.stateful and r.at is not None and n < r.at < end else None
        keep = self._prefix_cuts(r, n, last) if self.dec is None else set()
        cut = sorted(keep | ({at} if at is not None else set()))
        logits = None
        if ids is r.prompt:                     # (a link's resume: counted at its first feed)
            self.stats["prompt_tokens"] += len(ids)
        for a, b in zip([n] + cut, cut + [last]):
            if a >= b:
                continue
            kw = {} if b == len(ids) else {"logits": False}
            if vis is not None:
                kw["vis"] = vis[a - n:b - n]
            i, k = a, self._count_runs(None)
            for part, logits in self.eng.prefill_chunks(real[a:b], **kw):
                r.fed += ids[i:i + len(part)]
                i += len(part)
                r.prefill_tokens += len(part)
                self.stats["prefill_tokens"] += len(part)
                k = self._count_runs(k)
                if r.cancelled.is_set():
                    r.finish = "abort"
                    self.stats["prefill_s"] += time.perf_counter() - t0
                    return
            if b == at:                         # kept where the next turn parts (only there:
                self._keep_at(r)                # the last part ends a token short too)
            elif b in keep:                     # a prefix for other conversations
                self._keep_prefix(r)
        self.stats["prefill_s"] += time.perf_counter() - t0
        if last < len(ids):                     # (the prompt's last token: the decode feeds it)
            r.next = ids[-1]
            r.prefill_tokens += 1
            self.stats["prefill_tokens"] += 1
            return
        t = int(np.argmax(logits)) if r.sampling.greedy else \
            int(self._sampler(r)(logits, self._real(r, r.fed)))
        self._emit(r, t)

    def _prefix_cuts(self, r: Request, n: int, last: int) -> set:
        """Where r's prompt fed from n is kept as a prefix (a model with state, prefix_block):
        its marks and the multiples of prefix_block in (n + prefix_skip, last)."""
        B, skip = self.prefix_block, self.prefix_skip
        if not self.stateful or not B:
            return set()
        blocks = range((n + skip) // B * B + B, last, B)
        return {b for b in (*r.marks, *blocks) if n + skip < b < last}

    def _keep_prefix(self, r: Request, io=None) -> None:
        """r's prompt so far (the device there, mid-prompt) into the store as a prefix entry
        (store.Entry.prefix, on disk at once), for another conversation that shares it: the
        pages the store lacks and the state at this position. io: r's slot's (default self.io)."""
        io = self.io if io is None else io
        t0, mark = time.perf_counter(), self._io_mark(io)
        self.cache.save(io, list(r.fed), prefix=True)
        self.stats["prefix_saves"] += 1
        dt = self._phase(r, "prefix_save", t0)
        self.stats["prefix_save_s"] += dt
        self._io_parts(None, "prefix_save", dt, mark, io=io)

    def _keep_at(self, r: Request) -> None:
        """r's context at its reply start (the device there): a snapshot on the device, for its
        next turn's rollback (and the store at the worker's idle, _save_at); without snapshots
        (Engine(snapshots=0)) into the store now."""
        t0, mark = time.perf_counter(), self._io_mark()
        if self.eng.snapshot():
            self.snap_at = len(r.fed)
            self.stats["snapshots"] += 1
            self._phase(r, "snapshot", t0)
            return
        self.cache.save(self.io, list(r.fed))
        self.stats["saves"] += 1
        dt = self._phase(r, "save", t0)
        self.stats["save_s"] += dt
        self._io_parts(r, "save", dt, mark)

    def _prefill_mtp(self, r: Request) -> None:
        """_prefill with the MTP decoder: the whole prompt through MTPDecoder.prefill (lazy:
        the last row's draft is the loop's first run), the first token picked on the host from
        the last row's logits as the device's loop would (greedy: the argmax; sampled:
        generate.reference_pick with the request's first uniform), and r.next. Up to the
        reply's start (r.at; a decoder that does not hold the row before it: a token earlier)
        a part of its own, saved with the token after it and that row."""
        t0 = time.perf_counter()
        d, ids = self.dec, r.prompt
        real = self._real(r, ids)
        n = len(r.fed)
        vis = self.vision.prompt(real, r.images, n) if r.images else None
        at = r.at if self.stateful and r.at is not None else None
        if at is not None and not d.holds:
            at -= 1
        if at is not None and not n < at < len(ids):
            at = None
        self.stats["prompt_tokens"] += len(ids)
        st = MTPStats()
        k = [n, self._count_runs(None)]

        def on_run(part) -> bool:
            r.fed += ids[k[0]:k[0] + len(part)]
            k[0] += len(part)
            r.prefill_tokens += len(part)
            self.stats["prefill_tokens"] += len(part)
            k[1] = self._count_runs(k[1])
            return r.cancelled.is_set()

        def rows(a: int, b: int) -> dict:
            return {} if vis is None else {"vis": vis[a - n:b - n]}
        samp = None if r.sampling.greedy else G.Sampling(**self._sampler(r).params)
        if at is not None:
            d.prefill(real[n:at], st, on_run=on_run, follow=int(real[at]), **rows(n, at))
            if r.cancelled.is_set():
                r.finish = "abort"
                self.stats["prefill_s"] += time.perf_counter() - t0
                return
            self.follow = int(real[at])
            self.io.slot = d.slot
            self.cache.save(self.io, list(r.fed), follow=self.follow, held=d.kept_row())
            self.stats["saves"] += 1

        def pick(lg) -> int:            # (the device loop's pick of the first token)
            if samp is None:
                return int(np.argmax(lg))
            u = np.minimum(np.float32(self._sampler(r).rng.random()), np.float32(1 - 2.0 ** -24))
            return int(G.reference_pick(lg, samp, list(real), u, self.eng.cfg.S, self.softcap))
        a = len(r.fed)
        a0, _ = d.prefill(real[a:], st, pick=pick, on_run=on_run, lazy=True,
                          **rows(a, len(ids)))
        self.stats["prefill_s"] += time.perf_counter() - t0
        if a0 is None:                          # (cancelled)
            r.finish = "abort"
            return
        self.follow = None if d.pending is not None else a0     # (lazy: the loop's first run)
        self._emit(r, a0)

    def _emit(self, r: Request, t: int) -> None:
        """Token t generated: the request's output and its text, its end when it stops."""
        if r.finish is not None:
            return
        if t in r.stop_ids:
            r.next = None
            r.finish = "stop"
            return
        if r.t_first is None:
            r.t_first = time.perf_counter()
        r.out.append(t)
        r.next = t
        self.stats["generated"] += 1
        if r._detok is None:
            r._detok = _Detok(_Locked(self.tok, self.tok_lock), r.skip_special)
        r.text += r._detok.add(t)
        cut = None
        for s in r.stop_strs:
            i = r.text.find(s, max(0, r._shown - len(s)))
            if i >= 0 and (cut is None or i < cut):
                cut = i
        if cut is not None:
            r.text = r.text[:cut]
            r.finish = "stop"
        elif len(r.out) >= r.max_new:
            r.finish = "length"
        hold = 0                                # what may start a stop string: held back
        if r.finish is None:
            for s in r.stop_strs:
                for j in range(min(len(s) - 1, len(r.text)), 0, -1):
                    if r.text.endswith(s[:j]):
                        hold = max(hold, j)
                        break
        end = len(r.text) - hold
        if end > r._shown:
            r.events.put(("delta", r.text[r._shown:end]))
            r._shown = end

    def _decode(self, r: Request) -> None:
        """Up to the request's end, or a quantum when others wait. Invariant: the device has
        fed r.fed (eng.pos == len(r.fed)); r.next is picked, not fed."""
        if self.dec is not None:
            return self._decode_mtp(r)
        t0 = time.perf_counter()
        eng = self.eng
        k0 = len(r.out)
        while r.finish is None and r.next is not None:
            if len(r.fed) >= eng.cap:           # no room to feed the next token
                r.finish = "length"
                break
            if r.cancelled.is_set():
                r.finish = "abort"
                break
            left = r.max_new - len(r.out)
            n = left if not self._others(r) else min(left, self.quantum)
            if not getattr(eng.backend, "run_generate", None):
                n = min(n, CHUNK)               # (no stop() mid-run: a cancel waits for it)
            pick = self._sampler(r)
            card = getattr(eng, "can_generate", False)
            samp = None
            if card and not r.sampling.greedy:
                card = G.Sampling.fits(**pick.params)
                samp = G.Sampling(**pick.params) if card else None
            t = r.next
            if card:
                def stop() -> bool:
                    return r.finish is not None or r.cancelled.is_set() or self._yield(r, k0)
                r.fed.append(t)
                got = eng.generate_card(t, n, stop_ids=sorted(r.stop_ids),
                                        on_token=lambda x: self._emit(r, x), stop=stop,
                                        sampling=samp, context=list(self._real(r, r.fed)),
                                        rng=pick.rng if samp is not None else None)
                r.fed += got[:-1]
                if not got:
                    r.fed.pop()
                    break
                if got[-1] in r.stop_ids:
                    r.next = None
                    r.finish = r.finish or "stop"
            else:                               # the host picks each token
                ctx = list(self._real(r, r.fed))
                for _ in range(n):
                    r.fed.append(t)
                    ctx.append(t)
                    lg = eng.step(t)
                    t = int(np.argmax(lg)) if r.sampling.greedy else int(pick(lg, ctx))
                    self._emit(r, t)
                    if r.finish is not None or len(r.fed) >= eng.cap:
                        break
            if eng.pos != len(r.fed):
                raise RuntimeError(f"the device is at {eng.pos}, the context has {len(r.fed)}")
            if self._yield(r, k0):
                break
        self.stats["decode_s"] += time.perf_counter() - t0

    def _decode_mtp(self, r: Request) -> None:
        """_decode with the MTP loop on the device (MTPDecoder.loop_card from r.next and the
        decoder's draft: after a lazy prefill none, its first run drafts it): its tokens as they
        land, halted after the token in flight when r's turn ends (stop()). The decoder leaves
        the draft after the last token and the state in its committed slot."""
        t0 = time.perf_counter()
        eng, d = self.eng, self.dec
        k0 = len(r.out)
        while r.finish is None and r.next is not None:
            if r.cancelled.is_set():
                r.finish = "abort"
                break
            left = r.max_new - len(r.out)
            n = left if not self._others(r) else min(left, self.quantum)
            if not getattr(eng.backend, "run_generate", None):
                n = min(n, CHUNK)               # (no stop() mid-run: a cancel waits for it)
            pick = self._sampler(r)
            samp = None if r.sampling.greedy else G.Sampling(**pick.params)
            t = r.next
            first = [True]

            def took(x: int) -> None:           # (the loop emits t first: r's already)
                if not first[0]:
                    self._emit(r, x)
                first[0] = False

            def stop() -> bool:
                return r.finish is not None or r.cancelled.is_set() or self._yield(r, k0)
            r.fed.append(t)
            st = d.loop_card(t, d.draft, n + 1, stop=sorted(r.stop_ids), on_token=took,
                             samp=samp, context=list(self._real(r, r.fed)),
                             rng=pick.rng if samp is not None else None, halt=stop)
            got = st.tokens[1:]
            r.mtp_iters += st.iterations
            r.mtp_accepted += sum(st.accepted)
            self.stats["mtp_iters"] += st.iterations
            self.stats["mtp_accepted"] += sum(st.accepted)
            if not got:                         # (no room: the draft ran on its own)
                r.fed.pop()
                self.follow = t
                r.finish = r.finish or "length"
                break
            r.fed += got[:-1]
            self.follow = got[-1]
            if got[-1] in r.stop_ids:
                r.next = None
                r.finish = r.finish or "stop"
            if eng.pos != len(r.fed):
                raise RuntimeError(f"the device is at {eng.pos}, the context has {len(r.fed)}")
            if self._yield(r, k0):
                break
        self.stats["decode_s"] += time.perf_counter() - t0

    def _yield(self, r: Request, k0: int) -> bool:
        """r's turn ends: a new request can start (its prefill first: time to first token), or
        r has decoded its quantum while others run."""
        done = len(r.out) - k0
        if self.waiting and len(self.running) < self.max_running and done > 0:
            return True
        return self._others(r) and done >= self.quantum

    def _end(self, r: Request, why: str) -> None:
        with self.cv:
            if r in self.running:
                self.running.remove(r)
            if r in self.waiting:
                self.waiting.remove(r)
        if r.entry is not None:
            self.cache.unpin(r.entry)
            r.entry = None
        r.finish = r.finish or why
        if r.finish == "abort":
            self.stats["aborted"] += 1
        else:
            self.stats["finished"] += 1
        r.t_done = time.perf_counter()
        if r._shown < len(r.text):
            r.events.put(("delta", r.text[r._shown:]))
            r._shown = len(r.text)
        r.events.put(("done", r.finish))
        if self.owner is r:                     # the device keeps its context (unsaved)
            self.owner = None
            self.reply_kept = r.keep_reply is not None and r.keep_reply in r.text
        for f in self.observers:
            try:
                f(r)
            except BaseException:
                traceback.print_exc()

    def usage(self) -> dict:
        with self.cv:
            q = {"waiting": len(self.waiting), "running": len(self.running)}
        return {**q, **self.stats, "device_context": len(self.dev),
                "kv_store": self.store.usage(), "prefix_cache": self.cache.usage(),
                "kvio": dict(self.io.stats),
                "last_io": {k: dict(v) for k, v in self.last_io.items()}}


class _Detok(Detok):
    """chat.Detok, the special tokens' text kept or skipped."""

    def __init__(self, tok, skip: bool):
        super().__init__(tok)
        self.skip = skip

    def _dec(self, ids) -> str:
        return self.tok.decode(ids, skip_special_tokens=self.skip)


class _Locked:
    """A tokenizer whose decode holds a lock (the API's threads encode with it meanwhile)."""

    def __init__(self, tok, lock):
        self.tok, self.lock = tok, lock

    def decode(self, ids, **kw):
        with self.lock:
            return self.tok.decode(ids, **kw)
