"""otpu-serve --demo: the monitor and the dashboard over simulated requests and a simulated card
(no model, no card; docs/dashboard.md). Every number it shows is made up and says so: the
server's name ends in "(demo)", /api/stats carries "demo": true, the card's samples "source":
"demo". The chat and completion routes answer 503.

DemoWorker has what metrics.Monitor reads of a scheduler.Worker (stats, the queue, the request
on the device, `feeding`, usage(), observers) and runs requests the way the 35B runs them on the
BCU1525 (one sequence; a prompt fed at ~31 tokens/s, layer-major; ~33 tokens/s decode; a chat's
next turn feeds only its new tokens, the device holding the rest). backfill() writes the last
`minutes` of history (finished requests, the series) so a fresh dashboard has charts to draw."""
from __future__ import annotations

import random
import threading
import time

from . import scheduler as SC

PREFILL_TPS = 31.0      # the 35B's layer-major prompt on one BCU1525 (a 28k-token prompt ~15 min)
DECODE_TPS = 33.0       # ... its decode
CAP = 262144


class _Eng:
    def __init__(self, cap: int):
        self.cap, self.batch, self.pos = cap, 1, 0
        self.backend = _Backend(self)


class _Backend:
    """The DRAM map a BoardBackend's status holds (board.dram_layout's keys), the 35B's sizes
    on the BCU1525's 64 GiB, roughly."""

    def __init__(self, eng):
        self.eng = eng
        self.status = self

    @property
    def data(self) -> dict:
        G = 1 << 30
        kv = 262144 * 20480
        img = int(19.6 * G) + kv + int(0.1 * G)
        return {"dram": {"total": 64 * G, "image": img, "weights": int(19.6 * G),
                         "kv_capacity": kv, "kv_used": int(kv * self.eng.pos / self.eng.cap),
                         "program": 64 << 20, "snapshots": int(0.1 * G),
                         "free": 64 * G - img - (64 << 20)}}


class DemoWorker:
    def __init__(self, cap: int = CAP, seed: int = 7, prefill_tps: float = PREFILL_TPS,
                 decode_tps: float = DECODE_TPS):
        self.eng = _Eng(cap)
        self.r = random.Random(seed)
        self.prefill_tps, self.decode_tps = prefill_tps, decode_tps
        self.cv = threading.Condition()
        self.running: list = []
        self.waiting: list = []
        self.owner = None
        self.feeding = None
        self.observers: list = []
        self.on_idle = None
        self.stats = {"requests": 0, "finished": 0, "aborted": 0, "errors": 0,
                      "prompt_tokens": 0, "cached_tokens": 0, "prefill_tokens": 0,
                      "generated": 0, "decode_s": 0.0, "prefill_s": 0.0, "switch_s": 0.0,
                      "saves": 0, "restores": 0, "device_reuses": 0, "turns": 0,
                      "restore_s": 0.0, "save_s": 0.0}
        self.store = {"ram_bytes": 0, "ram_limit": 4 << 30, "disk_bytes": 0, "disk_limit": 0,
                      "ram_blobs": 0, "disk_blobs": 0, "ram_hits": 0, "disk_hits": 0,
                      "misses": 0, "spilled": 0, "dropped": 0}
        self.prefix = {"entries": 0, "tokens": 0, "lookups": 0, "hit_tokens": 0}
        self._ids = 0
        self._stop = threading.Event()
        self.thread = None
        self.page_bytes = 20480 * 256           # (a 256-position page of 10 attention layers)

    # ---- what the monitor reads
    def usage(self) -> dict:
        with self.cv:
            q = {"waiting": len(self.waiting), "running": len(self.running)}
        return {**q, **self.stats, "device_context": self.eng.pos,
                "kv_store": dict(self.store), "prefix_cache": dict(self.prefix)}

    def busy(self) -> int:
        with self.cv:
            return len(self.running) + len(self.waiting)

    def level(self) -> float:
        """The card's load for DemoCard: 2 a prompt, 1 decoding, 0 idle."""
        return 2.0 if self.feeding is not None else 1.0 if self.owner is not None else 0.0

    def flush(self) -> None:
        pass

    def stop(self, persist: bool = True) -> None:
        self._stop.set()
        with self.cv:
            self.cv.notify_all()

    # ---- the simulation
    def _request(self, plen: int, cached: int, max_new: int) -> SC.Request:
        self._ids += 1
        r = SC.Request(prompt=[0] * plen, max_new=max_new, rid=f"r{self._ids}")
        r.cached = cached
        r.t_submit = time.perf_counter()
        return r

    def _next(self) -> tuple:
        """(prompt tokens, cached, max_new) of the next simulated request: mostly a chat's next
        turn on the device's context, sometimes a new chat, now and then a long document."""
        u = self.r.random()
        pos = self.eng.pos
        if u < 0.08 or pos > 120000:                        # a long document, a new chat
            n = self.r.choice([9000, 14000, 21000, 28000])
            return n, 0, self.r.randint(300, 900)
        if u < 0.25:                                        # a new chat
            return self.r.randint(40, 900), 0, self.r.randint(60, 700)
        new = self.r.randint(20, 400)                       # the next turn
        return pos + new, pos, self.r.randint(40, 600)

    def submit(self, plen: int, cached: int, max_new: int) -> SC.Request:
        r = self._request(plen, cached, max_new)
        with self.cv:
            self.waiting.append(r)
            self.stats["requests"] += 1
            self.cv.notify_all()
        return r

    def start(self, first=None) -> "DemoWorker":
        """Run the simulation; first: (prompt, cached, max_new, the fraction already fed) of a
        request on the device at the start (the screenshot's long prompt)."""
        self.thread = threading.Thread(target=self._loop, args=(first,), daemon=True,
                                       name="otpu-demo-worker")
        self.thread.start()
        threading.Thread(target=self._arrivals, daemon=True, name="otpu-demo-arrivals").start()
        return self

    def _arrivals(self) -> None:
        while not self._stop.wait(self.r.expovariate(1 / 25.0)):
            if self.busy() < 4:
                self.submit(*self._next())

    def _loop(self, first) -> None:
        if first is not None:
            plen, cached, max_new, frac = first
            r = self.submit(plen, cached, max_new)
            self._run(r, frac)
        while not self._stop.is_set():
            with self.cv:
                while not self.waiting and not self._stop.is_set():
                    self.cv.wait(0.5)
                if self._stop.is_set():
                    return
                r = self.waiting[0]
            self._run(r, 0.0)

    def _sleep(self, s: float) -> bool:
        return self._stop.wait(s)

    def _run(self, r: SC.Request, frac: float) -> None:
        with self.cv:
            if r in self.waiting:
                self.waiting.remove(r)
            self.running.append(r)
        st = self.stats
        r.t_start = time.perf_counter()
        st["prompt_tokens"] += len(r.prompt)
        st["cached_tokens"] += r.cached
        self.prefix["lookups"] += 1
        if r.cached:
            st["device_reuses"] += 1
            self.prefix["hit_tokens"] += r.cached
        else:
            self.eng.pos = 0
        n = len(r.prompt) - r.cached
        done0 = int(frac * n)
        t0 = time.perf_counter() - done0 / self.prefill_tps
        r.t_start = min(r.t_start, t0)
        r.t_submit = min(r.t_submit, t0 - 0.4)
        base = self.eng.pos
        self.eng.pos = base + done0
        self.feeding = (r, base, n, t0, st["prefill_tokens"])
        chunk = 128                             # (a chunk of the image's prefill rows)
        while self.eng.pos - base < n:
            if self._sleep(min(chunk, n - (self.eng.pos - base)) / self.prefill_tps
                           * self.r.uniform(0.92, 1.08)):
                return
            self.eng.pos = min(base + n, self.eng.pos + chunk)
        self.feeding = None
        r.prefill_tokens = n
        st["prefill_tokens"] += n
        r.prefill_s = time.perf_counter() - t0
        st["prefill_s"] += r.prefill_s
        self.owner = r
        t1 = time.perf_counter()
        stop_at = self.r.randint(max(1, r.max_new // 3), r.max_new)
        while len(r.out) < stop_at:
            if self._sleep(self.r.uniform(0.9, 1.1) / self.decode_tps):
                return
            r.out.append(0)
            if r.t_first is None:
                r.t_first = time.perf_counter()
            st["generated"] += 1
            self.eng.pos += 1
        st["decode_s"] += time.perf_counter() - t1
        r.finish = "stop" if stop_at < r.max_new else "length"
        r.t_done = time.perf_counter()
        r.turns = 1
        self.owner = None
        st["finished"] += 1
        st["saves"] += 1
        kv = self.eng.pos * 20480                       # (bytes of its pages in the store)
        self.store["ram_bytes"] = min(self.store["ram_limit"], self.store["ram_bytes"] + kv // 4)
        self.store["ram_blobs"] += 1
        self.prefix["entries"] = min(64, self.prefix["entries"] + 1)
        self.prefix["tokens"] += self.eng.pos
        with self.cv:
            self.running.remove(r)
        for f in self.observers:
            f(r)

    def backfill(self, mon, minutes: float = 30.0) -> None:
        """The last `minutes` of simulated history into the monitor: finished requests and the
        server's and the card's series."""
        now = time.perf_counter()
        t = now - 60 * minutes
        pos, gen, fed = 0, 0, 0
        busy = []                                       # (start, end, kind) intervals
        while True:
            plen, cached, max_new = self._next_from(pos)
            n = plen - cached
            pre = n / self.prefill_tps * self.r.uniform(0.95, 1.05)
            k = self.r.randint(max(1, max_new // 3), max_new)
            dec = k / self.decode_tps * self.r.uniform(0.97, 1.03)
            if t + pre + dec > now - 20:
                break
            r = self._request(plen, cached, max_new)
            r.t_submit, r.t_start = t, t + 0.05
            r.prefill_s, r.prefill_tokens = pre, n
            r.t_first = t + 0.05 + pre + 1 / self.decode_tps
            r.t_done = t + 0.05 + pre + dec
            r.out = [0] * k
            r.finish = "stop" if k < max_new else "length"
            r.turns = 1
            busy += [(t + 0.05, t + 0.05 + pre, "pre", n), (t + 0.05 + pre, r.t_done, "dec", k)]
            for f in self.observers:
                f(r)
            st = self.stats
            st["requests"] += 1
            st["finished"] += 1
            st["prompt_tokens"] += plen
            st["cached_tokens"] += cached
            st["prefill_tokens"] += n
            st["generated"] += k
            pos = plen + k if cached or plen < 120000 else 0
            t = r.t_done + self.r.expovariate(1 / 30.0)
        self.eng.pos = pos
        self.prefix.update(entries=12, tokens=pos * 3, lookups=st["requests"],
                           hit_tokens=st["cached_tokens"])
        self.store.update(ram_bytes=min(self.store["ram_limit"], pos * 20480 * 3), ram_blobs=12)

        def load(x: float):
            return next(((k, n) for a, b, k, n in busy if a <= x < b), (None, 0))
        wall = mon.wall
        with mon.lock:
            for i in range(int(60 * minutes)):
                x = now - 60 * minutes + i
                k, _ = load(x)
                mon.series.append({"t": round(wall + x, 2),
                                   "dec": round(self.decode_tps * self.r.uniform(0.98, 1.02), 2)
                                   if k == "dec" else 0.0,
                                   "pre": round(self.prefill_tps * self.r.uniform(0.98, 1.02), 2)
                                   if k == "pre" else 0.0,
                                   "run": int(k is not None), "wait": 0, "pos": None,
                                   "cpu": round(0.05 + 0.04 * (k is not None)
                                                + self.r.uniform(0, 0.02), 4)})
        card = mon.card
        if card is not None and hasattr(card, "sample"):
            level = card.busy
            for i in range(int(30 * minutes)):
                x = now - 60 * minutes + 2 * i
                k, _ = load(x)
                card.busy = (lambda v: (lambda: v))(2.0 if k == "pre" else 1.0 if k else 0.0)
                c = card.sample()
                c["t"] = wall + x
                mon._card_sample(c)
            card.busy = level

    def _next_from(self, pos: int) -> tuple:
        save = self.eng.pos
        self.eng.pos = pos
        try:
            return self._next()
        finally:
            self.eng.pos = save


class DemoServer:
    """What api.make_handler needs of a Server, over a DemoWorker: no model behind it."""

    def __init__(self, worker: DemoWorker, monitor, name: str, api_key=None):
        self.w, self.monitor, self.name, self.api_key = worker, monitor, name, api_key
        self.names = {name}
        self.info = {"demo": True}
        self.created = int(time.time())

    def health(self):
        return 200, {"status": "ok", "demo": True}

    def models(self) -> dict:
        return {"object": "list", "data": [
            {"id": self.name, "object": "model", "created": self.created, "owned_by": "opentpu",
             "root": self.name, "max_model_len": self.w.eng.cap}]}

    def stats(self) -> dict:
        return dict(self.monitor.snapshot(recent=False), demo=True)

    def _none(self, *a, **kw):
        from .api import APIError
        raise APIError(503, "otpu-serve --demo: no model behind this server", kind="demo")
    chat = completion = generate = _none


def open_demo(name: str = "qwen36-35b (demo)", minutes: float = 30.0, first=None,
              interval: float = 1.0, card_interval: float = 2.0, seed: int = 7,
              api_key=None, start: bool = True):
    """(DemoServer, DemoWorker, Monitor) with `minutes` of history, running when start."""
    from . import metrics as M
    from . import telemetry as TM
    w = DemoWorker(seed=seed)
    card = TM.DemoCard(busy=w.level, interval=card_interval, seed=seed)
    mon = M.Monitor(w, name, card=card, interval=interval, progs=None, fans=None,
                    info={"demo": True, "backend": "board", "label": "otpu-serve35",
                          "weights": "fp4 + attn,dn,shared=nvfp4, head int8",
                          "cap": CAP, "kv_page_bytes": w.page_bytes, "slots": 1,
                          "kv_ram_bytes": 4 << 30, "kv_disk_bytes": 0,
                          "dram_bytes": 64 << 30})
    w.backfill(mon, minutes)
    srv = DemoServer(w, mon, name, api_key)
    if start:
        w.start(first)
        mon.start()
        card.start()
    return srv, w, mon


def main(a) -> int:
    from .api import serve
    srv, w, mon = open_demo(api_key=a.api_key, first=(28000, 0, 600, 0.41))
    for plen, cached, max_new in ((28400, 28000, 400), (900, 0, 300)):    # (two waiting)
        w.submit(plen, cached, max_new)
    httpd = serve(srv, a.host, a.port)
    print(f"otpu-serve --demo: simulated requests and card on http://{a.host}:{a.port}/ "
          f"(no model: the chat routes answer 503)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        mon.stop()
        w.stop()
    return 0
