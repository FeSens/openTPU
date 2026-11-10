"""otpu-serve's monitor: the server's numbers for Prometheus (GET /metrics) and for the dashboard
(GET /api/stats, the live stream GET /api/events, the page at /; docs/dashboard.md).

Where they come from:
- each finished request (Worker.observers: on the worker's thread, it only queues a record):
  its time in the queue, to the first token (TTFT), feeding its prompt, decoding, end to end, its
  prompt, cached and generated tokens, its finish;
- the worker's counters and queue, sampled every `interval` seconds (1) by the monitor's thread:
  tokens generated and fed (a prompt being fed counts the device's position past its start, so
  a layer-major prompt, which the engine yields in one part, moves while it runs), the running
  and waiting requests, the prompt in flight (tokens done of its total, the rate, the ETA), the
  device's context (KV positions used of the model's), the KV store and prefix cache;
- the card (telemetry.CardSampler, its own thread every 2 s; none on the simulators) and the
  host (telemetry.HostStats);
- the program cache (progstate.saves): whether it saves, and the wait entries computed.

The monitor's thread builds one snapshot per tick and hands it to every /api/events stream (the
browser draws it: the server renders nothing); a scrape of /metrics folds the queued records
and formats what the last tick saw. The worker never waits for the monitor.

Prometheus names: vLLM's (`vllm:` prefix, label model_name) where they mean the same here --
vllm:num_requests_running / _waiting, vllm:kv_cache_usage_perc (the device context's positions
over the model's context: one sequence), vllm:prompt_tokens_total, vllm:generation_tokens_total,
vllm:request_success_total{finished_reason}, vllm:time_to_first_token_seconds,
vllm:e2e_request_latency_seconds, vllm:time_per_output_token_seconds,
vllm:request_queue_time_seconds, vllm:request_prefill_time_seconds,
vllm:request_decode_time_seconds, vllm:request_prompt_tokens, vllm:request_generation_tokens,
vllm:prefix_cache_queries_total / vllm:prefix_cache_hits_total (prompt tokens, and those the
device or the store held) -- and `otpu:` for the card, the store and the host (otpu:die_temp
_celsius, otpu:power_watts{source}, otpu:utilization_ratio{unit}, ...). A value the card does
not give is left out (no sample), never a zero."""
from __future__ import annotations

import bisect
import json
import threading
import time
from collections import deque

from . import fans as FN
from . import progstate as PS

# vLLM's buckets (vllm/engine/metrics.py), the TTFT's past 160 s for a long prompt on the card
TTFT_BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0,
                7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0)
E2E_BUCKETS = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
               120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0)
TPOT_BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0,
                7.5, 10.0, 20.0, 40.0, 80.0)
TOKEN_BUCKETS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000,
                 100000, 200000)
RECENT = 200            # finished requests kept for the table and the percentiles
SERIES = 1800           # the server's points (one a tick: 30 min at 1 s)
CARD_SERIES = 900       # the card's (one a sample: 30 min at 2 s)
DECODE_WINDOW = 5.0     # seconds the live decode tok/s is measured over
PROG_WINDOW = 60.0      # ... the wait entries computed are counted over
PREFILL_WINDOW = 15.0   # ... the prefill's (a layer-major prompt moves a chunk at a time: while
                        # one is fed, its tokens so far over its time so far)


class Histogram:
    def __init__(self, buckets):
        self.le = tuple(buckets)
        self.counts = [0] * (len(self.le) + 1)
        self.sum = 0.0
        self.n = 0

    def observe(self, v: float) -> None:
        self.counts[bisect.bisect_left(self.le, v)] += 1
        self.sum += v
        self.n += 1

    def lines(self, name: str, labels: str) -> list[str]:
        out, c = [], 0
        for le, k in zip(self.le + (float("inf"),), self.counts):
            c += k
            out.append(f'{name}_bucket{{{labels},le="{_num(le)}"}} {c}')
        out.append(f"{name}_sum{{{labels}}} {_num(self.sum)}")
        out.append(f"{name}_count{{{labels}}} {self.n}")
        return out


def _num(v) -> str:
    if v == float("inf"):
        return "+Inf"
    if isinstance(v, int):
        return str(v)
    return repr(round(float(v), 6))


def _esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")


def _pct(xs, q: float):
    """The q-th quantile of xs (nearest rank), None for none."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def _r(x, n: int = 3):
    return None if x is None else round(float(x), n)


class Monitor:
    """The monitor of a worker (scheduler.Worker or batching.BatchWorker): start(); stop().
    card: a telemetry.CardSampler (or DemoCard) or None; card_note: why there is none (shown as
    n/a); info: the server's static facts (backend, weights, ...); progs: the program cache's
    state (progstate.saves; None: no such section, the demo's); fans: fan_pid's state and the
    route that sets its gains (fans.state; None: neither, the demo's)."""

    def __init__(self, worker, name: str, card=None, card_note: str | None = None,
                 interval: float = 1.0, info: dict | None = None, host=None, progs=PS.saves,
                 fans=FN.state):
        from .telemetry import HostStats
        self.w, self.name, self.card, self.card_note = worker, name, card, card_note
        self.interval = interval
        self.info = dict(info or {})
        self.host = host if host is not None else HostStats()
        self.t0 = time.time()
        self.wall = time.time() - time.perf_counter()     # perf_counter -> unix time
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self._ticking = threading.Lock()                  # (one tick at a time)
        self._q: deque = deque()                          # records queued by the worker
        self.recent_q: deque = deque(maxlen=RECENT)
        self.rec_version = 0
        self.series: deque = deque(maxlen=SERIES)
        self.card_series: deque = deque(maxlen=CARD_SERIES)
        self._marks: deque = deque()                      # (t, generated, fed) for the rates
        self.progs = progs
        self.fans = fans
        self._prog_marks: deque = deque()                 # (t, entries computed, their seconds)
        self.h = {"ttft": Histogram(TTFT_BUCKETS), "e2e": Histogram(E2E_BUCKETS),
                  "tpot": Histogram(TPOT_BUCKETS), "queue": Histogram(E2E_BUCKETS),
                  "prefill": Histogram(TTFT_BUCKETS), "decode": Histogram(E2E_BUCKETS),
                  "prompt": Histogram(TOKEN_BUCKETS), "gen": Histogram(TOKEN_BUCKETS)}
        self.finished = {"stop": 0, "length": 0, "abort": 0, "error": 0}
        self.latest: dict | None = None
        self.seq = 0                                      # ticks so far
        self.cpu_s = 0.0                                  # the monitor thread's CPU seconds
        self._host_last = None
        self._stop = threading.Event()
        self.thread = None
        worker.observers.append(self._ended)
        if card is not None:
            card.on_sample = self._card_sample

    # ---- the worker's side (its thread: a record queued, nothing else)
    def _ended(self, r) -> None:
        self._q.append((r.rid, r.t_submit, r.t_start, r.t_first, r.t_done, r.prefill_s,
                        len(r.prompt), r.cached, r.prefill_tokens, len(r.out), r.finish,
                        r.turns, r.swaps, r.error, dict(getattr(r, "phases", None) or {}),
                        {k: dict(v) for k, v in (getattr(r, "io", None) or {}).items()}))

    def _fold(self) -> None:
        """The queued records into the histograms and the recent table (under the lock)."""
        while self._q:
            (rid, t_sub, t_start, t_first, t_done, pre_s, plen, cached, fed, nout, fin, turns,
             swaps, err, phases, io) = self._q.popleft()
            t_done = t_done or time.perf_counter()
            e2e = t_done - t_sub
            queue = (t_start - t_sub) if t_start else e2e
            ttft = (t_first - t_sub) if t_first else None
            dec = (t_done - t_first) if t_first else None
            tpot = dec / (nout - 1) if dec is not None and nout > 1 else None
            fin = fin if fin in self.finished else "error" if fin else "abort"
            self.finished[fin] += 1
            h = self.h
            h["e2e"].observe(e2e)
            h["queue"].observe(queue)
            h["prompt"].observe(plen)
            h["gen"].observe(nout)
            if ttft is not None:
                h["ttft"].observe(ttft)
            if pre_s:
                h["prefill"].observe(pre_s)
            if dec is not None:
                h["decode"].observe(dec)
            if tpot is not None:
                h["tpot"].observe(tpot)
            self.recent_q.append({
                "rid": rid, "t": round(self.wall + t_sub, 3), "status": fin, "error": err,
                "prompt": plen, "cached": cached, "fed": fed, "completion": nout,
                "queue_s": _r(queue), "ttft_s": _r(ttft), "prefill_s": _r(pre_s),
                "prefill_tps": _r(fed / pre_s, 2) if pre_s and fed else None,
                "decode_tps": _r((nout - 1) / dec, 2) if dec and nout > 1 else None,
                "e2e_s": _r(e2e), "turns": turns, "swaps": swaps,
                "phases": {k: _r(v, 4) for k, v in phases.items()}, "io": io})
            self.rec_version += 1

    def _card_sample(self, c: dict) -> None:
        """A card sample (the sampler's thread): its point on the card's series."""
        u = c.get("util") or {}
        pt = {"t": round(c["t"], 2), "run": u.get("RUNNING"), "mxu": u.get("MXU_BUSY"),
              "mac": u.get("MXU_MAC"), "vpu": u.get("VPU_BUSY"), "dma": u.get("DMA_BUSY"),
              "dram": u.get("DRAM"), "gbs": c.get("dram_gbs"), "die": c.get("die_c"),
              "reg": (c.get("reg") or {}).get("hot_c"), "w": (c.get("power") or {}).get("w")}
        for i, x in enumerate(c.get("cards") or ()):      # (die, reg: the hottest of the cards)
            pt[f"die{i}"], pt[f"reg{i}"] = x.get("die_c"), (x.get("reg") or {}).get("hot_c")
        with self.lock:
            self.card_series.append(pt)

    # ---- the monitor's thread
    def start(self) -> "Monitor":
        self.tick()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="otpu-monitor")
        self.thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self.cv:
            self.cv.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.card is not None:
            self.card.stop()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            c0 = time.thread_time()
            try:
                self.tick()
            except Exception:           # (a monitor never takes the server down)
                import traceback
                traceback.print_exc()
            self.cpu_s += time.thread_time() - c0

    def _live(self, now: float) -> dict:
        """The worker's state now: queue, the request in flight, counters (no engine call)."""
        w = self.w
        st = w.stats
        with w.cv:
            running, waiting = list(w.running), list(w.waiting)
        owner = getattr(w, "owner", None)
        feeding = getattr(w, "feeding", None)
        pos = int(getattr(w.eng, "pos", 0) or 0)
        cap = int(getattr(w.eng, "cap", 0) or 0)
        if getattr(w, "slot_dev", None) is not None:     # (BatchWorker: its slots' contexts)
            pos, cap = sum(len(d) for d in w.slot_dev), cap * len(w.slot_dev)
        fed = st.get("prefill_tokens", 0)
        inflight = None
        if feeding is not None:
            r, base, n, t0, fed0 = feeding
            done = max(0, min(n, int(getattr(w.eng, "pos", 0) or 0) - base))
            fed = max(fed, fed0 + done)     # (a layer-major prompt counts once it is fed)
            el = now - t0
            rate = done / el if done and el > 0 else None
            with self.lock:
                prior = _pct([x["prefill_tps"] for x in self.recent_q], 0.5)
            eta = (n - done) / rate if rate else (n - done) / prior if prior else None
            inflight = {"rid": r.rid, "phase": "prefill", "done": done, "of": n,
                        "prompt": len(r.prompt), "cached": r.cached, "elapsed_s": _r(el, 1),
                        "rate_tps": _r(rate, 2), "eta_s": _r(eta, 1),
                        "eta_from": "this prompt" if rate else "recent prompts" if prior
                        else None, "max_new": r.max_new,
                        "queue_s": _r((r.t_start or now) - r.t_submit, 2)}
        elif owner is not None and owner.finish is None and owner.t_first is not None:
            r = owner
            n = len(r.out)
            el = now - r.t_first
            inflight = {"rid": r.rid, "phase": "decode", "tokens": n, "max_new": r.max_new,
                        "prompt": len(r.prompt), "cached": r.cached,
                        "rate_tps": _r((n - 1) / el, 2) if n > 1 and el > 0 else None,
                        "ttft_s": _r(r.t_first - r.t_submit, 2), "elapsed_s": _r(el, 1)}

        def item(r, state):
            return {"rid": r.rid, "state": state, "prompt": len(r.prompt), "cached": r.cached,
                    "completion": len(r.out), "max_new": r.max_new,
                    "age_s": _r(now - r.t_submit, 1),
                    "ttft_s": _r(r.t_first - r.t_submit, 2) if r.t_first else None}
        items = []
        for r in running:
            state = ("prefill" if feeding is not None and feeding[0] is r else
                     "decode" if r is owner else "paused" if r.t_first or r.fed else "starting")
            items.append(item(r, state))
        items += [item(r, "queued") for r in waiting]
        return {"running": len(running), "waiting": len(waiting), "items": items[:64],
                "inflight": inflight, "pos": pos, "cap": cap,
                "generated": st.get("generated", 0), "fed": fed,
                "prompt_tokens": st.get("prompt_tokens", 0),
                "cached_tokens": st.get("cached_tokens", 0),
                "prefill_runs": st.get("prefill_runs")}

    def _rates(self, now: float, gen: int, fed: int) -> tuple:
        m = self._marks
        m.append((now, gen, fed))
        while m and now - m[0][0] > max(DECODE_WINDOW, PREFILL_WINDOW) + 1:
            m.popleft()

        def rate(i: int, win: float):
            old = next((x for x in m if now - x[0] <= win + 1e-6), None)
            if old is None or now - old[0] < 0.5:
                return 0.0 if len(m) > 1 else None
            return (m[-1][i] - old[i]) / (now - old[0])
        return rate(1, DECODE_WINDOW), rate(2, PREFILL_WINDOW)

    def tick(self) -> dict:
        """One snapshot (the monitor's thread; /api/stats before the first tick)."""
        with self._ticking:
            return self._tick()

    def _tick(self) -> dict:
        now = time.perf_counter()
        live = self._live(now)
        dec_tps, pre_tps = self._rates(now, live["generated"], live["fed"])
        inf = live["inflight"]
        if inf is not None and inf["phase"] == "prefill" and inf["rate_tps"]:
            pre_tps = inf["rate_tps"]       # (a prompt in flight: its own rate, not its parts')
        try:
            u = self.w.usage()
        except Exception:
            u = {}
        host = self.host.sample()
        card = self.card.latest if self.card is not None else None
        progs = self._prog_cache(now)
        fans = None
        if self.fans is not None:
            try:
                fans = self.fans()
            except Exception:
                pass
        with self.lock:
            self._fold()
            recent = list(self.recent_q)
            rec_version = self.rec_version
            cap = live["cap"]
            pt = {"t": round(self.wall + now, 2), "dec": _r(dec_tps, 2), "pre": _r(pre_tps, 2),
                  "run": live["running"], "wait": live["waiting"], "pos": live["pos"],
                  "cpu": host.get("cpu")}
            self.series.append(pt)
            last = recent[-50:]
            store, pc = u.get("kv_store") or {}, u.get("prefix_cache") or {}
            snap = {
                "v": 1, "seq": self.seq, "t": pt["t"], "rec_version": rec_version,
                "server": {"model": self.name, "uptime_s": round(time.time() - self.t0, 1),
                           "started": round(self.t0, 1),
                           "cap": int(getattr(self.w.eng, "cap", 0) or 0),
                           "slots": int(getattr(self.w.eng, "batch", 1) or 1),
                           "state": live["inflight"]["phase"] if live["inflight"] else
                           "busy" if live["running"] else "idle", **self.info},
                "queue": {"running": live["running"], "waiting": live["waiting"],
                          "items": live["items"]},
                "inflight": live["inflight"],
                "throughput": {"decode_tps": _r(dec_tps, 2), "prefill_tps": _r(pre_tps, 2),
                               "generated": live["generated"], "fed": live["fed"],
                               "prompt_tokens": live["prompt_tokens"],
                               "cached_tokens": live["cached_tokens"],
                               "prefill_runs": live["prefill_runs"]},
                "latency": {"ttft_p50": _pct([x["ttft_s"] for x in last], 0.5),
                            "ttft_p90": _pct([x["ttft_s"] for x in last], 0.9),
                            "e2e_p50": _pct([x["e2e_s"] for x in last], 0.5),
                            "decode_tps_p50": _pct([x["decode_tps"] for x in last], 0.5),
                            "prefill_tps_p50": _pct([x["prefill_tps"] for x in last], 0.5),
                            "n": len(last)},
                "requests": {"total": u.get("requests"), "finished": u.get("finished"),
                             "aborted": u.get("aborted"), "errors": u.get("errors"),
                             **{f"finish_{k}": v for k, v in self.finished.items()}},
                "kv": {"positions": live["pos"], "cap": cap,
                       "usage": _r(live["pos"] / cap, 5) if cap else None,
                       "device_context": u.get("device_context"),
                       "store": {"ram_used": store.get("ram_bytes"),
                                 "ram_limit": store.get("ram_limit"),
                                 "disk_used": store.get("disk_bytes"),
                                 "disk_limit": store.get("disk_limit"),
                                 "ram_blobs": store.get("ram_blobs"),
                                 "disk_blobs": store.get("disk_blobs")} if store else None,
                       "store_stats": {k: store.get(k) for k in ("ram_hits", "disk_hits",
                                                                 "misses", "spilled",
                                                                 "dropped", "writer_errors",
                                                                 "writer_alive") if k in store},
                       "prefix": dict(pc) if pc else None,
                       "saves": u.get("saves"), "restores": u.get("restores"),
                       "device_reuses": u.get("device_reuses"),
                       "snapshots": u.get("snapshots"), "rollbacks": u.get("rollbacks"),
                       "idle_saves": u.get("idle_saves"), "idle_preempts": u.get("idle_preempts"),
                       "last_io": u.get("last_io"),        # (the last save's, restore's parts)
                       "page_bytes": self.info.get("kv_page_bytes"),
                       "dram": self._dram(card)},
                "prog_cache": progs,
                "fans": fans,
                "card": card if card is not None else None,
                "card_note": self.card_note if self.card is None else
                (self.card.error if card is None and self.card.error else
                 "waiting for the first sample" if card is None else None),
                "card_error": getattr(self.card, "error", None),
                "host": host,
                "monitor": {"interval_s": self.interval, "ticks": self.seq,
                            "cpu_s": round(self.cpu_s, 3),
                            "card_cpu_s": _r(getattr(self.card, "cpu_s", None), 3),
                            "card_samples": getattr(self.card, "samples", None),
                            "card_interval_s": getattr(self.card, "interval", None),
                            "card_took_s": getattr(self.card, "took_s", None),
                            "card_took_max_s": getattr(self.card, "took_max_s", None),
                            "card_spacing_s": getattr(self.card, "spacing_s", None),
                            "card_spacing_max_s": getattr(self.card, "spacing_max_s", None)},
                "point": pt,
                "card_point": self.card_series[-1] if self.card_series else None,
            }
            self.latest = snap
            self.seq += 1
            self.cv.notify_all()
        return snap

    def _prog_cache(self, now: float) -> dict | None:
        """The program cache's state with the wait entries computed over the last PROG_WINDOW:
        what a chunk loses to them while saves are off."""
        if self.progs is None:
            return None
        try:
            s = self.progs()
        except Exception:
            return None
        m = self._prog_marks
        m.append((now, s["entries_computed"], s["entries_s"]))
        while now - m[0][0] > PROG_WINDOW:
            m.popleft()
        return dict(s, entries_last_min=s["entries_computed"] - m[0][1],
                    entries_s_last_min=round(s["entries_s"] - m[0][2], 3))

    def _dram(self, card) -> dict | None:
        """The card's DRAM map (image, weights, KV capacity and use, snapshots, programs, free):
        this process's runner status (BoardBackend.status), else the status file smi read."""
        st = getattr(getattr(self.w.eng, "backend", None), "status", None)
        d = getattr(st, "data", {}).get("dram") if st is not None else None
        if d is None and card is not None:
            d = (card.get("process") or {}).get("dram")
        return dict(d) if isinstance(d, dict) else None

    # ---- readers (HTTP handler threads)
    def snapshot(self, recent: bool = True, history: bool = False) -> dict:
        """The last tick's snapshot (/api/stats): with the recent requests' table, and with the
        series (the dashboard's first event)."""
        if self.latest is None:
            self.tick()
        with self.lock:
            out = dict(self.latest)
            if recent:
                out["recent"] = list(reversed(list(self.recent_q)[-50:]))
            if history:
                out["series"] = list(self.series)
                out["card_series"] = list(self.card_series)
        return out

    def wait(self, seq: int, timeout: float = 15.0) -> dict | None:
        """The first snapshot after tick `seq` (an /api/events stream's), None on timeout or
        once the monitor stops."""
        with self.cv:
            ok = self.cv.wait_for(lambda: self._stop.is_set() or (
                self.latest is not None and self.latest["seq"] > seq), timeout)
            return self.latest if ok and not self._stop.is_set() else None

    def stopped(self) -> bool:
        return self._stop.is_set()

    def recent(self, n: int = 50) -> list:
        """The last n finished requests, newest first."""
        with self.lock:
            return list(reversed(list(self.recent_q)[-n:]))

    # ---- Prometheus
    def prometheus(self) -> str:
        """GET /metrics: the text exposition format (version 0.0.4)."""
        snap = self.latest or self.tick()
        with self.lock:
            self._fold()
            h = {k: (v.le, list(v.counts), v.sum, v.n) for k, v in self.h.items()}
            fin = dict(self.finished)
        L = f'model_name="{_esc(self.name)}"'
        out: list[str] = []

        def head(name, kind, text):
            out.append(f"# HELP {name} {text}")
            out.append(f"# TYPE {name} {kind}")

        def gauge(name, v, text, labels=""):
            if v is None:
                return
            head(name, "gauge", text)
            out.append(f"{name}{{{L}{labels}}} {_num(v)}")

        def counter(name, v, text):
            if v is None:
                return
            head(name + "_total", "counter", text)
            out.append(f"{name}_total{{{L}}} {_num(v)}")

        def hist(name, key, text):
            head(name, "histogram", text)
            le, counts, s, n = h[key]
            hh = Histogram(le)
            hh.counts, hh.sum, hh.n = counts, s, n
            out.extend(hh.lines(name, L))
        q, th, kv = snap["queue"], snap["throughput"], snap["kv"]
        gauge("vllm:num_requests_running", q["running"],
              "Requests on the device now (in turns: admitted, prefill or decode).")
        gauge("vllm:num_requests_waiting", q["waiting"], "Requests waiting to be admitted.")
        gauge("vllm:kv_cache_usage_perc", kv["usage"],
              "KV cache usage: the device context's positions over the model's context (1 = "
              "full).")
        counter("vllm:prompt_tokens", th["prompt_tokens"], "Prompt tokens of the requests "
                "admitted (cached ones included).")
        counter("vllm:generation_tokens", th["generated"], "Tokens generated.")
        counter("otpu:prefill_device_runs", th.get("prefill_runs"), "Device runs the host "
                "started to feed prompts (a layer loop's runs in a bucket: one).")
        head("vllm:request_success_total", "counter", "Requests finished, by finish reason.")
        for k, v in fin.items():
            out.append(f'vllm:request_success_total{{{L},finished_reason="{k}"}} {v}')
        counter("vllm:prefix_cache_queries", th["prompt_tokens"],
                "Prompt tokens looked up in the prefix cache (the device's context and the KV "
                "store).")
        counter("vllm:prefix_cache_hits", th["cached_tokens"],
                "Prompt tokens the device's context or the KV store held (not fed).")
        hist("vllm:time_to_first_token_seconds", "ttft", "Time to the first token (from the "
             "request's arrival).")
        hist("vllm:e2e_request_latency_seconds", "e2e", "End-to-end request latency.")
        hist("vllm:time_per_output_token_seconds", "tpot", "Mean time per output token after "
             "the first, per request.")
        hist("vllm:request_queue_time_seconds", "queue", "Time in the queue before the first "
             "turn on the device.")
        hist("vllm:request_prefill_time_seconds", "prefill", "Time feeding the prompt.")
        hist("vllm:request_decode_time_seconds", "decode", "Time from the first token to the "
             "last.")
        hist("vllm:request_prompt_tokens", "prompt", "Prompt tokens per request.")
        hist("vllm:request_generation_tokens", "gen", "Generated tokens per request.")
        gauge("otpu:decode_tokens_per_second", th["decode_tps"],
              f"Tokens generated per second over the last {DECODE_WINDOW:g} s.")
        gauge("otpu:prefill_tokens_per_second", th["prefill_tps"],
              f"Prompt tokens fed per second over the last {PREFILL_WINDOW:g} s.")
        inf = snap.get("inflight") or {}
        if inf.get("phase") == "prefill":
            gauge("otpu:prompt_progress_tokens", inf["done"], "The prompt in flight: tokens fed.")
            gauge("otpu:prompt_remaining_tokens", inf["of"] - inf["done"],
                  "The prompt in flight: tokens left to feed.")
        gauge("otpu:kv_positions_used", kv["positions"], "The device context's positions.")
        gauge("otpu:kv_positions_capacity", kv["cap"], "The model's context (positions).")
        st = kv.get("store") or {}
        if st:
            head("otpu:kv_store_bytes", "gauge", "The KV store's saved contexts, by tier.")
            for tier in ("ram", "disk"):
                if st.get(f"{tier}_used") is not None:
                    out.append(f'otpu:kv_store_bytes{{{L},tier="{tier}"}} {st[f"{tier}_used"]}')
        pc = kv.get("prefix") or {}
        if "prefix_hits" in pc:
            counter("otpu:prefix_entry_hits", pc["prefix_hits"], "Restores from a prompt's "
                    "prefix kept as it was fed (another conversation's shared system prompt).")
            counter("otpu:prefix_entry_hit_tokens", pc["prefix_hit_tokens"], "Prompt tokens "
                    "restored from those prefixes.")
        gauge("otpu:uptime_seconds", snap["server"]["uptime_s"], "Seconds since the server "
              "started.")
        pcs = snap.get("prog_cache") or {}
        if pcs.get("saving") is not None:
            gauge("otpu:prog_cache_saving", 1 if pcs["saving"] else 0, "The program cache saves "
                  "(0: its disk is under the floor, programs compiled and wait entries computed "
                  "are not kept, and a long prompt computes them on every pass).")
            gauge("otpu:prog_cache_free_bytes", pcs["free_bytes"], "Free bytes on the program "
                  "cache's disk.")
            gauge("otpu:prog_cache_floor_bytes", pcs["floor_bytes"], "The free bytes a save must "
                  "leave.")
        counter("otpu:programs_compiled", pcs.get("programs_compiled"), "Programs compiled by "
                "this process.")
        counter("otpu:programs_written", pcs.get("programs_written"), "Programs it saved to the "
                "program cache.")
        counter("otpu:wait_entries_computed", pcs.get("entries_computed"), "Wait entries computed "
                "(not found in memory or on disk).")
        counter("otpu:wait_entries_compute_seconds", pcs.get("entries_s"), "Their seconds.")
        warm = snap["server"].get("warm") or {}
        gauge("otpu:prog_cache_warm_tokens", warm.get("tokens"), "A prompt's layer runs are on "
              "disk through this many tokens (at the server's start).")
        c = snap.get("card")
        if c is not None:
            gauge("otpu:card_up", 1 if c.get("ok") else 0, "The accelerator answers on PCIe.")
            gauge("otpu:die_temp_celsius", c.get("die_c"), "The FPGA die (the governor's "
                  "reading, or the SYSMON's register when the governor's is gone).")
            gauge("otpu:regulator_temp_celsius", (c.get("reg") or {}).get("hot_c"),
                  "The VCCINT regulator's hottest temperature (the BMC watch).")
            p = c.get("power") or {}
            gauge("otpu:power_watts", p.get("w"), "Power: measured (on the BCU1525 the LTC3884's"
                  " READ_PIN, a lower bound of VCCINT's input, not checked against a meter) or the"
                  " Vivado estimate.", f',source="{p.get("source")}"')
            g = c.get("gov") or {}
            if g:
                gauge("otpu:governor_run", 1 if g.get("state") == "RUN" else 0,
                      "The thermal governor in RUN (1) or SAFE (0).")
            u = c.get("util") or {}
            if u:
                head("otpu:utilization_ratio", "gauge", "Unit busy share since the last sample "
                     "(counter deltas).")
                for k, v in u.items():
                    out.append(f'otpu:utilization_ratio{{{L},unit="{k}"}} {_num(v)}')
            if c.get("dram_rd_gbs") is not None:
                head("otpu:dram_bandwidth_bytes_per_second", "gauge", "The card's DRAM traffic.")
                for k, f in (("read", "dram_rd_gbs"), ("write", "dram_wr_gbs")):
                    if c.get(f) is not None:
                        out.append(f'otpu:dram_bandwidth_bytes_per_second{{{L},dir="{k}"}} '
                                   f"{_num(round(c[f] * 1e9))}")
        hs = snap.get("host") or {}
        gauge("otpu:host_cpu_ratio", hs.get("cpu"), "The host's CPU busy share (all cores).")
        gauge("otpu:host_memory_used_bytes", hs.get("mem_used"), "The host's memory in use.")
        gauge("otpu:process_cpu_ratio", hs.get("proc_cpu"), "This server's CPU (1 = a core).")
        return "\n".join(out) + "\n"


def sse(event: str, obj) -> bytes:
    """One server-sent event: `event: <name>` and its data as one line of JSON."""
    return (f"event: {event}\ndata: ".encode() + json.dumps(obj, separators=(",", ":")).encode()
            + b"\n\n")
