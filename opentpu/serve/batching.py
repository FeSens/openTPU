"""The device's worker with KV slots (continuous batching): an Engine(batch=B, batch_area=True)
image holds B sequences' KV caches; up to B requests run at once, each in its own slot.

Each iteration of the loop:

1. waiting requests take free slots: the slot whose context shares the longest prefix with the
   prompt; the prompt then continues the longest prefix found in that slot, in another slot
   (its pages copied on the device), or in the KV store (host RAM, then disk); a slot's context
   that would be overwritten is saved to the store first;
2. one run of the oldest prompt still being fed (chunked prefill: the decoding requests wait one
   run at most): up to MCOLS of its tokens, every weight streamed once, no LM head
   (batch.Steps.feed). A prompt feeds all but its last token this way;
3. one batched step for every decoding request (opentpu/llm/batch.py): a run feeds each one its
   token and picks its next on the device, greedy or with its own sampling; the host reads one
   word per request. A request's first step feeds its prompt's last token, so its first token
   is picked on the device too: the host never reads logits.

A request that decodes alone in slot 0 with nothing waiting runs the card's decode loop instead
(Engine.generate_card), halted after the token in flight when another arrives. Requests past
the B slots wait. Only models whose image holds several sequences batch (Qwen3, the Llama-likes:
SmolLM3, Phi-4-mini); the others take scheduler.Worker's turns.
"""
from __future__ import annotations

import time
import traceback

from ..llm import batch as BA
from ..llm import generate as G
from . import kvio as KV
from . import scheduler as SC


PREFILL_BUDGET = 64     # prompt tokens fed per iteration (SGLang's chunked prefill size)


def batches(eng) -> bool:
    """The engine runs batched steps."""
    return getattr(eng, "bat", None) is not None


class BatchWorker(SC.Worker):
    def __init__(self, eng, tok, store=None, model_key: str = "model", ram_bytes: int = 4 << 30,
                 area=False, quantum: int = SC.QUANTUM, prefill_budget: int = PREFILL_BUDGET,
                 mixed: bool = False):
        if not batches(eng):
            raise ValueError("BatchWorker needs Engine(batch=B > 1, batch_area=True)")
        B = eng.batch
        super().__init__(eng, tok, store=store, model_key=model_key, max_running=B,
                         quantum=quantum, ram_bytes=ram_bytes, area=area)
        if self.stateful:
            raise ValueError("batched slots hold KV caches only (no recurrent state)")
        self.B = B
        self.budget = max(1, int(prefill_budget))
        self.mixed = bool(mixed)
        self.ios = [self.io] + [KV.ContextIO(eng, seq=s, area=self.io.area) for s in range(1, B)]
        self.slot_req: list = [None] * B          # the request in each slot
        self.slot_dev: list = [[] for _ in range(B)]   # the tokens each slot's caches hold
        self.slot_dirty = [False] * B
        self.slot_used = [0.0] * B
        self.steps = BA.Steps(eng)
        self.steps.warm()
        self.prefilling: list = []                # requests being fed, oldest first
        self.stats.update(steps=0, step_rows=0, step_runs=0, slot_copies=0, card_loops=0,
                          step_s=0.0)

    # ---- the loop
    def _loop(self) -> None:
        while True:
            self._trim()
            with self.cv:
                while not self._stop and not self.waiting and not self.running:
                    self.cv.wait()
                if self._stop:
                    for r in self.waiting + self.running:
                        self._end(r, "abort")
                    return
                for r in [r for r in self.waiting + self.running if r.cancelled.is_set()
                          and r.finish is None]:
                    r.finish = "abort"
                admit, nfree = [], len(self._free_slots())
                while self.waiting and len(admit) < nfree:
                    r = self.waiting.pop(0)
                    self.running.append(r)
                    admit.append(r)
            try:
                for r in [r for r in self.running if r.finish is not None]:
                    self._end(r, r.finish)
                for r in admit:
                    if r.finish is None:
                        self._bind(r)
                self._ahead()
                mixed = self._mixed_step() if self.prefilling and self.mixed else "no"
                if mixed != "ran":
                    # the oldest prompts up to the budget; one feed run while a mixed run's
                    # program compiles
                    budget = self.budget if mixed == "no" else 1
                    while self.prefilling and budget > 0:
                        budget -= self._prefill_run(self.prefilling[0])
                    dec = self._decoding()
                    if dec:
                        self._decode_step(dec)
                for r in [r for r in self.running if r.finish is not None]:
                    self._end(r, r.finish)
            except BaseException as e:
                if self._link_recovered(e):       # (the link reset: each request resumes)
                    continue
                traceback.print_exc()             # every running request fails; the server
                #                                   goes on with an empty device
                self.stats["errors"] += 1
                for r in list(self.running):
                    r.error = f"{type(e).__name__}: {e}"
                    self._end(r, "error")
                self._reset_device()

    def _reset_device(self) -> None:
        self.prefilling = []
        self.steps.drop()
        for s in range(self.B):
            self.slot_req[s], self.slot_dev[s], self.slot_dirty[s] = None, [], False
            try:
                self.eng.reset(s)
            except BaseException:
                pass

    def _link_recovered(self, e, r=None) -> bool:
        """After a run that failed on the link (cards.LinkError: x2's two cards), as chat35x2's
        turn: the link reset (CardsTransport.recover: the cores stopped, LRST) and the slots'
        contexts dropped (the run may have written its rows' caches and states); each request
        running then goes back to the queue's head and resumes with its prompt and its output so
        far as the context to make (_want: the store's longest match, the rest fed again). Once
        a request: one the link fails again ends in error, as any other failure. False: not a
        link's failure, or no recovery (the error path's). (r: scheduler.Worker's, a turn's
        request; a batched run is every running request's.)"""
        from ..host.cards import LinkError
        t = getattr(getattr(self.eng.backend, "board", None), "t", None)
        live = [r for r in self.running if r.finish is None]
        if not isinstance(e, LinkError) or not hasattr(t, "recover") or \
                any(getattr(r, "_relinked", False) for r in live):
            return False
        traceback.print_exc()
        try:
            t.recover()
        except BaseException:
            traceback.print_exc()
            return False
        self.stats["link_recoveries"] = self.stats.get("link_recoveries", 0) + 1
        self._reset_device()
        with self.cv:
            for r in live:
                self.running.remove(r)
                r._relinked, r.slot, r.next = True, None, None
            self.waiting[:0] = live
        return True

    def _free_slots(self) -> list[int]:
        return [s for s in range(self.B) if self.slot_req[s] is None]

    # ---- slots
    def _save_slot(self, s: int, r: SC.Request | None = None) -> None:
        """Slot s's context into the store (r: the request it makes way for, its "save" phase)."""
        if self.slot_dirty[s] and self.slot_dev[s]:
            io = self.ios[s]
            t0, mark = time.perf_counter(), self._io_mark(io)
            self.cache.save(io, list(self.slot_dev[s]))
            self.stats["saves"] += 1
            dt = self._phase(r, "save", t0)
            self.stats["save_s"] += dt
            self._io_parts(r, "save", dt, mark, io=io)
        self.slot_dirty[s] = False

    def _bind(self, r: SC.Request) -> None:
        """r into a free slot, its prompt's longest prefix in place."""
        t0 = time.perf_counter()
        r.t_start = r.t_start or t0
        want, P = self._want(r), self.io.layout.P
        limit = len(want) - 1
        free = self._free_slots()
        common = [self._common(self.slot_dev[s], want) for s in range(self.B)]
        # the slot: one in the block of feed_rows() slots (an MXU pass's rows; a step runs each
        # block in a run of its own) holding the most requests, so that they take the fewest
        # runs; then the one holding the longest prefix (another slot's is copied on the card),
        # slot 0 (the card's loop runs there), the least recently used
        q = self.steps.feed_rows()
        held = [sum(y // q == x // q for y in range(self.B) if self.slot_req[y] is not None)
                for x in range(self.B)]
        s = max(free, key=lambda x: (held[x], min(common[x], limit), x == 0, -self.slot_used[x]))
        own = min(common[s], limit)
        other = max((x for x in range(self.B) if x != s), key=lambda x: common[x], default=None)
        k_other = min(common[other], limit) if other is not None else 0
        m = self.cache.match(want, limit=limit)
        k = max(own, k_other, m.k)
        if k < len(self.slot_dev[s]):             # the slot's context past k is overwritten
            self._save_slot(s)
        if k == own:
            self.eng.adopt(own, seq=s)
            if own:
                self.stats["device_reuses"] += 1
        elif k == k_other:
            have = own // P
            self.ios[s].copy_from(self.ios[other], [(i, P) for i in range(have, -(-k // P))])
            self.eng.adopt(k, seq=s)
            self.stats["slot_copies"] += 1
        else:
            k = self.cache.restore(self.ios[s], m, have=own)
            self.eng.adopt(k, seq=s)
            if k:
                self.stats["restores"] += 1
        r.fed = list(want[:k])
        r.slot = s
        if want is r.prompt:                    # (a resume's context: not the request's prompt)
            r.cached = k
            self.stats["cached_tokens"] += k
            self.stats["prompt_tokens"] += len(want)
        self.slot_req[s], self.slot_dev[s], self.slot_dirty[s] = r, r.fed, True
        self.slot_used[s] = time.monotonic()
        r._feed = list(want[k:-1])               # fed in runs; the last token, a step's row
        r.next = want[-1]
        r.prefill_tokens += 1
        self.stats["prefill_tokens"] += 1
        if r._feed:
            self.prefilling.append(r)
            q = self.steps.feed_rows()
            p, n = k, len(r._feed)
            head = min(n, (q - p % q) % q)
            self.steps.prefetch_feed(s, p + n - (n - head) % q, (n - head) % q)
        r._bset = False
        self.stats["switch_s"] += time.perf_counter() - t0

    # ---- prefill and decode
    def _prefill_run(self, r: SC.Request) -> int:
        """One feed run of r's prompt (up to MCOLS tokens, to the next multiple of it); the
        tokens fed."""
        t0 = time.perf_counter()
        q = self.steps.feed_rows()
        p = self.eng.poss[r.slot]
        part = r._feed[:q - p % q]              # (a whole run: the budget may go over by less)
        self.steps.feed(r.slot, part)
        del r._feed[:len(part)]
        r.fed += part
        r.prefill_tokens += len(part)
        self.stats["prefill_tokens"] += len(part)
        self.stats["prefill_s"] += time.perf_counter() - t0
        r.prefill_s += time.perf_counter() - t0
        if r.cancelled.is_set():
            r.finish = "abort"
        if not r._feed or r.finish is not None:
            self.prefilling.remove(r)
        return max(1, len(part))

    def _rows_at(self, k: int):
        """The decode rows (slot, position) and kinds of the step k iterations on, as far as
        they can be told: the decoding requests that have not reached max_tokens by then, k
        positions on, and the prompts fed by then (the feed budget's runs, or a mixed step's
        spare rows), from their last token on (stop tokens, new requests and a mixed step
        waiting for its program cannot be told)."""
        rows, kinds = [], []
        for x in self.running:
            if x.finish is None and x.next is not None and x not in self.prefilling \
                    and x.slot is not None and x.max_new - len(x.out) > k:
                rows.append((x.slot, len(x.fed) + k))
                kinds.append(self.steps.kinds.get(x.slot) if x._bset else
                             BA.kind_of(self._samp(x)))
        q = self.steps.feed_rows()
        left = [len(x._feed) for x in self.prefilling]
        at = [self.eng.poss[x.slot] for x in self.prefilling]
        i, ndec = 0, len(rows)
        for it in range(k + 1):
            # _mixed_step's spare rows (the prompt done then decodes from the next step), else
            # the budget's feed runs (it decodes in this one)
            mixed = self.mixed and 0 < ndec < q
            b = q - ndec if mixed else self.budget
            while i < len(left) and b > 0:
                n = min(left[i], b if mixed else q - at[i] % q)
                left[i] -= n
                at[i] += n
                b -= n
                if left[i] == 0:
                    x, j = self.prefilling[i], it + mixed
                    if j <= k:
                        rows.append((x.slot, len(x.prompt) - 1 + (k - j)))
                        kinds.append(BA.kind_of(self._samp(x)))
                    ndec += 1
                    i += 1
                if mixed:
                    break
        order = sorted(range(len(rows)), key=lambda j: rows[j][0])     # as _decode_step's
        return [rows[j] for j in order], [kinds[j] for j in order]

    def _ahead(self) -> None:
        """Compile ahead this iteration's and the next one's steps when their rows differ from
        the last step's (a prompt joins, a request reaches max_tokens)."""
        for k in (0, 1):
            rows, kinds = self._rows_at(k)
            if not rows or any(p >= self.eng.cap for _, p in rows):
                continue
            if len(rows) == 1 and rows[0][0] == 0 and not self.waiting and not self.prefilling:
                continue                        # (the card's decode loop)
            self.steps.prefetch_rows(rows, kinds)

    def _samp(self, r: SC.Request):
        """The request's sampling on the device: None (greedy) or a generate.Sampling (top-k
        clamped to the device's K_MAX; 0: K_MAX)."""
        s = r.sampling
        if s.greedy:
            return None
        k = s.top_k if 0 < s.top_k <= G.K_MAX else G.K_MAX
        return G.Sampling(s.temperature, k, s.top_p, max(1.0, s.repetition_penalty))

    def _decoding(self) -> list:
        """The requests that decode this step, in slot order (a row set's program key)."""
        dec = [r for r in self.running if r.finish is None and r.next is not None
               and r not in self.prefilling]
        for r in dec:
            if len(r.fed) >= self.eng.cap:
                r.finish = "length"
        return sorted((r for r in dec if r.finish is None), key=lambda r: r.slot)

    def _set_sampling(self, dec) -> None:
        for r in dec:
            if not r._bset:
                self.steps.set_sampling(r.slot, self._samp(r), list(r.fed) + [r.next])
                r._bset = True

    def _inputs(self, dec):
        """The step's rows, tokens and uniforms (each slot's sampling set on its first)."""
        self._set_sampling(dec)
        uni = None
        if any(self._samp(r) is not None for r in dec):
            uni = [float(self._sampler(r).rng.random()) if self._samp(r) is not None else 0.0
                   for r in dec]
        return [(r.slot, len(r.fed)) for r in dec], [r.next for r in dec], uni

    def _outputs(self, dec, out, t0: float, runs: int) -> None:
        self.stats["steps"] += 1
        self.stats["step_rows"] += len(dec)
        self.stats["step_runs"] += runs
        for r, t in zip(dec, out):
            r.fed.append(r.next)
            self._emit(r, t)
        dt = time.perf_counter() - t0
        self.stats["step_s"] += dt
        self.stats["decode_s"] += dt
        for r in dec:
            self.slot_used[r.slot] = time.monotonic()

    def _decode_step(self, dec: list) -> None:
        if not dec:
            return
        if len(dec) == 1 and dec[0].slot == 0 and not self.waiting and not self.prefilling \
                and self.eng.can_generate:
            return self._card_loop(dec[0])
        t0 = time.perf_counter()
        rows, toks, uni = self._inputs(dec)
        n0 = self.steps.runs
        out = self.steps.run(rows, toks, uni, free=set(self._free_slots()))
        self._outputs(dec, out, t0, self.steps.runs - n0)

    def _mixed_step(self) -> str:
        """With fewer decoding requests than an MXU pass's rows and a prompt being fed: one run
        of their step with the oldest prompt's next tokens in the spare rows (Steps.mixed), so
        the decoding requests do not wait for its feed runs. "ran"; "pending" (its program
        compiles: one feed run this time, then the step); "no" (none applies)."""
        dec = self._decoding()
        q = self.steps.feed_rows()
        if not dec or len(dec) >= q:
            return "no"
        r = self.prefilling[0]
        p = self.eng.poss[r.slot]
        k = min(q - len(dec), len(r._feed), self.eng.block - p % self.eng.block)
        self._set_sampling(dec)                 # (the rows' kinds: the program's key)
        ready = self.steps.mixed_ready([(x.slot, len(x.fed)) for x in dec], r.slot, k)
        if not ready:
            return "no" if ready is None else "pending"
        t0 = time.perf_counter()
        rows, toks, uni = self._inputs(dec)
        n0 = self.steps.runs
        part = r._feed[:k]
        out = self.steps.mixed(rows, toks, uni, r.slot, part)
        del r._feed[:k]
        r.fed += part
        r.prefill_tokens += k
        self.stats["prefill_tokens"] += k
        self.stats["mixed_runs"] = self.stats.get("mixed_runs", 0) + 1
        if r.cancelled.is_set():
            r.finish = "abort"
        if not r._feed or r.finish is not None:
            self.prefilling.remove(r)
        self._outputs(dec, out, t0, self.steps.runs - n0)
        return "ran"

    def _card_loop(self, r: SC.Request) -> None:
        """r alone, in slot 0: the card's decode loop until it ends or another request comes."""
        t0 = time.perf_counter()
        eng = self.eng
        left = r.max_new - len(r.out)
        n = left if getattr(eng.backend, "run_generate", None) else min(left, SC.CHUNK)
        pick = self._sampler(r)
        samp = self._samp(r)
        st0 = pick.rng.bit_generator.state if samp is not None else None
        t = r.next
        r.fed.append(t)
        got = eng.generate_card(t, n, stop_ids=sorted(r.stop_ids),
                                on_token=lambda x: self._emit(r, x),
                                stop=lambda: (r.finish is not None or r.cancelled.is_set()
                                              or bool(self.waiting)),
                                sampling=samp, context=list(r.fed),
                                rng=pick.rng if samp is not None else None)
        if st0 is not None:      # one uniform a token picked: a run that stop() halted drew
            pick.rng.bit_generator.state = st0       # its whole run's ahead, and the steps
            pick.rng.random(len(got))                # after it go on from the next one
        r.fed += got[:-1]
        if not got:
            r.fed.pop()
        elif got[-1] in r.stop_ids:
            r.next = None
            r.finish = r.finish or "stop"
        r._bset = False                         # (the slot's penalty factors miss these)
        if eng.poss[0] != len(r.fed):
            raise RuntimeError(f"slot 0 is at {eng.poss[0]}, the context has {len(r.fed)}")
        self.stats["card_loops"] += 1
        self.stats["decode_s"] += time.perf_counter() - t0

    def _end(self, r: SC.Request, why: str) -> None:
        s = getattr(r, "slot", None)
        if s is not None and self.slot_req[s] is r:
            self.slot_req[s] = None
            self.slot_used[s] = time.monotonic()
        if r in self.prefilling:
            self.prefilling.remove(r)
        super()._end(r, why)

    def stop(self, persist: bool = True) -> None:
        with self.cv:
            self._stop = True
            self.cv.notify_all()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if persist and self.store.dir is not None:
            for s in range(self.B):
                try:
                    self._save_slot(s)
                except BaseException:
                    traceback.print_exc()
            self.store.persist()

    def flush(self) -> None:
        """Forget the saved contexts and what the slots hold (no request runs)."""
        self.cache.clear()
        for s in range(self.B):
            if self.slot_req[s] is None:
                self.slot_dev[s], self.slot_dirty[s] = [], False

    def usage(self) -> dict:
        u = super().usage()
        tm = dict(self.steps.timing)
        khz = (getattr(self.eng.backend, "info", None) or {}).get("core_khz")
        if khz:
            tm["device_s"] = round(tm["cycles"] / (khz * 1e3), 3)
        u["step_timing"] = tm
        u["slots"] = [{"slot": s, "context": len(self.slot_dev[s]),
                       "request": getattr(self.slot_req[s], "rid", None)}
                      for s in range(self.B)]
        u["device_context"] = sum(len(d) for d in self.slot_dev)
        return u
