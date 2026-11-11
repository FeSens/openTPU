"""The Qwen3.5 / Qwen3.6-MoE worker with sequence slots (batched decode, opentpu/llm/batch35.py):
an Engine(batch=B, batch_area=True, layer_major=R) image holds B sequences' KV caches,
DeltaNet states and convolution windows; up to B requests run at once, each in its own slot.

Each iteration of the loop (batching.BatchWorker's, with the model's state):

1. waiting requests take free slots: the slot whose context the prompt continues (a context
   with state continues from its whole length only: a chat's next turn), else the longest
   context saved in the KV store that the prompt continues (its pages and state written into
   the slot, but the whole pages the slot's context shares with the prompt), else from position
   0; a slot's context that would be overwritten is saved to the store first;
2. up to prefill_budget tokens of the oldest prompt still being fed, by the layer-major runs
   at its slot (batch35.prefill_slot): all but its last token. A run stops at the prompt's next
   cut (scheduler.Worker._prefix_cuts: its marks and the multiples of prefix_block), where the
   slot's context goes into the store as a prefix (_keep_prefix), and the iteration goes on to
   the step: another conversation sharing the prefix restores it (a saved context's whole
   length) and feeds only the rest. It stops too at the prompt's reply start (Request.at), kept
   for the conversation's next turn, whose rendering parts there;
3. one batched step for every decoding request (batch35.Steps): a run feeds each one its token
   and picks its next on the device, greedy or with its own sampling. A request's first step
   feeds its prompt's last token, so its first token is picked on the device too.

A request that decodes alone in slot 0 with nothing waiting runs the card's decode loop
(Engine.generate_card), as batching.BatchWorker's. A step's rows never pad into a free slot
(Steps' free): a row fed there would move that slot's state past its saved context. Where the
image takes them (batch35.slots_fit) the steps run over slots: one program per slots and kinds
for every position. A run that fails on x2's link resets it and resumes every running request
(batching.BatchWorker._link_recovered), as otpu-chat35x2's turn does.
"""
from __future__ import annotations

import time

from ..llm import batch35 as B35
from . import batching as BW
from . import kvio as KV
from . import scheduler as SC

PREFILL_BUDGET = 512    # prompt tokens fed per iteration (layer runs: a run a layer a chunk)


def batches(eng) -> bool:
    """The engine runs a Qwen3.5 image's batched steps."""
    return BW.batches(eng) and bool(getattr(eng.image, "slot_runs", False))


class BatchWorker35(BW.BatchWorker):
    def __init__(self, eng, tok, store=None, model_key: str = "model", ram_bytes: int = 4 << 30,
                 area=False, quantum: int = SC.QUANTUM, prefill_budget: int = PREFILL_BUDGET,
                 prefix_block: int = SC.PREFIX_BLOCK, prefix_skip: int = SC.PREFIX_SKIP):
        if not batches(eng) or not eng.layer_major:
            raise ValueError("BatchWorker35 needs a Qwen3.5 Engine(batch=B > 1, batch_area=True, "
                             "layer_major=R)")
        B = eng.batch
        SC.Worker.__init__(self, eng, tok, store=store, model_key=model_key, max_running=B,
                           quantum=quantum, ram_bytes=ram_bytes, area=area,
                           prefix_block=prefix_block, prefix_skip=prefix_skip)
        self.B = B
        self.budget = max(1, int(prefill_budget))
        self.mixed = False
        self.ios = [self.io] + [KV.ContextIO(eng, seq=s, area=self.io.area) for s in range(1, B)]
        self.slot_req: list = [None] * B
        self.slot_dev: list = [[] for _ in range(B)]
        self.slot_dirty = [False] * B
        self.slot_used = [0.0] * B
        self.steps = B35.Steps(eng, slots=B35.slots_fit(eng))
        self.prefilling: list = []
        self.stats.update(steps=0, step_rows=0, step_runs=0, slot_copies=0, card_loops=0,
                          step_s=0.0)

    def warm_steps(self, samplings=()) -> int:
        """Steps over slots (Steps.slots): every program the server's requests step on, made live
        before the first one, so that none compiles mid-chat: each set of slots, each row
        greedy or one of samplings' kinds (scheduler.Sampling: the API's defaults), batch.AHEAD
        of them compiling at once (on x2 in the slice workers; more, and Steps drops the oldest
        results), then taken, each one's IMEM fit known to Steps.plan (else its first plan
        compiles the rows' program again to learn it). All of them stay live (Steps.live: the
        x2 B = 4 warm's 80 past batch.LIVE's 64). The number made; none without slots (span
        programs are keyed by their rows' positions). Before start()."""
        import itertools
        from types import SimpleNamespace
        st = self.steps
        if not st.slots:
            return 0
        kinds = list(dict.fromkeys([None] + [BW.BA.kind_of(self._samp(SimpleNamespace(sampling=s)))
                                             for s in samplings]))
        groups = [([(s, 0) for s in slots], ks) for n in range(1, self.B + 1)
                  for slots in itertools.combinations(range(self.B), n)
                  for ks in itertools.product(kinds, repeat=n)]
        st.live = max(st.live, len(groups) + BW.BA.LIVE)
        for i in range(0, len(groups), BW.BA.AHEAD):
            part = groups[i:i + BW.BA.AHEAD]
            for rows, ks in part:
                st._prefetch(st._key(rows, ks))
            for rows, ks in part:
                progs, _ = st._take(rows, ks)
                st._fit[st._sig(st._key(rows, ks))] = st._size(progs) <= st.limit
        return len(groups)

    def _bind(self, r: SC.Request) -> None:
        """r into a free slot: the one whose whole context r's prompt continues, else slot 0 (the
        card's loop runs there), else the one sharing the most tokens with the prompt, the least
        recently used of equals. There the store's longest such context is restored (but the
        whole pages among the tokens shared), else it starts from position 0."""
        t0 = time.perf_counter()
        r.t_start = r.t_start or t0
        want, K = self._want(r), self.eng.spec.conv_k
        if len(want) < K:                        # (the step's rows: from conv_k - 1 on)
            r.error = f"a prompt of {len(want)} tokens: batched decode needs {K} or more"
            r.finish = "error"
            return
        limit = len(want) - 1
        free = self._free_slots()
        common = {x: self._common(want, self.slot_dev[x]) for x in free}

        def own(x):                             # (a context with state: its whole length)
            n = len(self.slot_dev[x])
            return n if n <= limit and common[x] == n else 0
        s = max(free, key=lambda x: (own(x), x == 0, common[x], -self.slot_used[x]))
        k0 = own(s)
        m = self.cache.match(want, limit=limit)
        k = max(k0, m.k)
        if self.slot_dev[s] and k0 < len(self.slot_dev[s]) or k > k0:
            self._save_slot(s, r)                # (the slot's context is overwritten)
        if k == k0:
            self.eng.adopt(k0, seq=s)
            if k0:
                self.stats["device_reuses"] += 1
        else:
            io, t1 = self.ios[s], time.perf_counter()
            mark = self._io_mark(io)
            k = self.cache.restore(io, m, have=common[s])
            ta = time.perf_counter()
            self.eng.adopt(k, seq=s)
            ta = time.perf_counter() - ta
            if k:
                self.stats["restores"] += 1
                dt = self._phase(r, "restore", t1)
                self.stats["restore_s"] += dt
                self._io_parts(r, "restore", dt, mark, io=io, adopt=ta)
        r.fed = list(want[:k])
        r.slot = s
        if want is r.prompt:                    # (a resume's context: not the request's prompt)
            r.cached = k
            self.stats["cached_tokens"] += k
            self.stats["prompt_tokens"] += len(want)
        self.slot_req[s], self.slot_dev[s], self.slot_dirty[s] = r, r.fed, True
        self.slot_used[s] = time.monotonic()
        r._feed = list(want[k:-1])              # fed by layer runs; the last token, a step's row
        keep = self._prefix_cuts(r, k, limit)
        r._at = r.at if r.at is not None and k < r.at <= limit and r.at not in keep else None
        r._cuts = sorted(keep | ({r._at} if r._at is not None else set()))
        r.next = want[-1]
        r.prefill_tokens += 1
        self.stats["prefill_tokens"] += 1
        if r._feed:
            self.prefilling.append(r)
        r._bset = False
        self.stats["switch_s"] += time.perf_counter() - t0

    def _part(self, r: SC.Request) -> int:
        """The tokens of r's prompt the next _prefill_run takes: the budget's, to its next cut."""
        n = min(self.budget, len(r._feed))
        return min(n, r._cuts[0] - len(r.fed)) if r._cuts else n

    def _prefill_run(self, r: SC.Request) -> int:
        """Up to the budget of r's prompt by layer runs at its slot, its context kept at a cut
        (through its slot's io); what it took of the iteration's budget: the tokens fed, all of
        it at a cut (the decoding requests' step comes next)."""
        t0 = time.perf_counter()
        part = r._feed[:self._part(r)]
        B35.prefill_slot(self.eng, part, r.slot, logits=False)
        del r._feed[:len(part)]
        r.fed += part
        r.prefill_tokens += len(part)
        self.stats["prefill_tokens"] += len(part)
        dt = time.perf_counter() - t0
        self.stats["prefill_s"] += dt
        r.prefill_s += dt
        cut = bool(r._cuts) and len(r.fed) == r._cuts[0]
        if cut:
            del r._cuts[0]
            if len(r.fed) == r._at:
                self._keep_reply_start(r)
            else:
                self._keep_prefix(r, self.ios[r.slot])
        if r.cancelled.is_set():
            r.finish = "abort"
        if not r._feed or r.finish is not None:
            self.prefilling.remove(r)
        return self.budget if cut else max(1, len(part))

    def _keep_reply_start(self, r: SC.Request) -> None:
        """r's context at its reply start into the store through its slot's io (no snapshots on
        a slot): the conversation's next turn restores it, whatever the reply renders as."""
        io = self.ios[r.slot]
        t0, mark = time.perf_counter(), self._io_mark(io)
        self.cache.save(io, list(r.fed))
        self.stats["saves"] += 1
        dt = self._phase(r, "save", t0)
        self.stats["save_s"] += dt
        self._io_parts(r, "save", dt, mark, io=io)

    def _rows_at(self, k: int):
        """The decode rows (slot, position) and kinds k iterations on: the decoding requests
        that have not reached max_tokens by then, with the prompts whose runs this iteration's
        budget ends (_ahead compiles their first step's programs before those runs: a request
        joining the batch waits for no compile)."""
        joins, b = set(), self.budget           # (the loop's: the oldest prompts while the
        for x in self.prefilling:               # budget lasts, up to the budget each)
            if b <= 0:
                break
            n = self._part(x)
            b -= max(1, n)
            if n == len(x._feed):
                joins.add(id(x))
        rows, kinds = [], []
        for x in sorted(self.running, key=lambda x: x.slot if x.slot is not None else -1):
            if x.finish is None and x.next is not None and x.slot is not None \
                    and (x not in self.prefilling or id(x) in joins) \
                    and x.max_new - len(x.out) > k:
                rows.append((x.slot, len(x.fed) + len(x._feed) + k))
                kinds.append(self.steps.kinds.get(x.slot) if x._bset else
                             BW.BA.kind_of(self._samp(x)))
        return rows, kinds

    def _ahead(self) -> None:
        """batching.BatchWorker's, and with two or more requests decoding the step of each set
        of all of them but one (a program from their positions to the first one's block end):
        any one may end on a stop token at any step, and the others go on without waiting for a
        compile (slot 0 alone with no request waiting runs the card's decode loop, which needs
        none). One at a time, once nothing else compiles (Steps.prefetch_group)."""
        super()._ahead()
        rows, kinds = self._rows_at(0)
        if len(rows) >= 2 and all(p < self.eng.cap for _, p in rows):
            loop = self.eng.can_generate and not self.waiting and not self.prefilling
            for i in range(len(rows)):
                sub, sk = rows[:i] + rows[i + 1:], kinds[:i] + kinds[i + 1:]
                if len(sub) > 1 or sub[0][0] != 0 or not loop:
                    self.steps.prefetch_group(sub, sk)

    def _decode_step(self, dec: list) -> None:
        if not dec:
            return
        if len(dec) == 1 and dec[0].slot == 0 and not self.waiting and not self.prefilling \
                and self.eng.can_generate:
            return self._card_loop(dec[0])
        t0 = time.perf_counter()
        rows, toks, uni = self._inputs(dec)
        n0 = self.steps.runs
        out = self.steps.run(rows, toks, uni)    # (no free slots' padding: their state)
        self._outputs(dec, out, t0, self.steps.runs - n0)
