"""otpu-serve's telemetry: the card's state as otpu-smi reads it, and the host's CPU and memory, on
a thread of their own at a low rate, for the monitor (opentpu/serve/metrics.py: /metrics, the
dashboard).

CardSampler calls smi.query every `interval` seconds (default 2) on a transport of its own
(XdmaTransport without DMA: register reads, as otpu-smi beside a chat): the bitstream, link,
utilization since the last sample (the counters' SNAP deltas), the governor, the die, the power,
the runner's status file. It reads exactly what smi's default does: the governor's die, the
SYSMON's register only once smi says the governor's reading is gone (never --sysmon: an AXI read
of the SYSMON's DRP registers can stop the governor's temperature bus, opentpu/host/gov.py), and
the BMC bridge's watch registers (no session). A bridge without the watch (BMC1) needs a session
under the BMC lock that fan_pid shares: that one is read every `bmc_every` seconds (default 10),
the last reading carried with its age between them. The I2C buses are not read (no PMBus on the
BCU1525; --telemetry-i2c reads them). Nothing here is ever on a request's path: the worker never
waits for a sample, and a sample that fails is reported in the next snapshot (`error`), not
raised.

sample() is the compact form the monitor serves (`card` in /api/stats): every value None where the
card, the bitstream or the BMC does not give it, never a made-up number; demo=True (otpu-serve
--demo, tests) is a synthetic card that says so ("source": "demo")."""
from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path

UNITS = ("RUNNING", "MXU_BUSY", "MXU_MAC", "VPU_BUSY", "QNT_BUSY", "DMA_BUSY")
INTERVAL = 2.0          # seconds between card samples (fan_pid reads the governor every 2 s)
BMC_EVERY = 10.0        # seconds between BMC sessions on a bridge without the watch

# the samples' spans (perf_counter), any sampler's: its thread holds the interpreter for most of
# one (smi's Python), so another thread's work then waits (overlap: a save's or restore's share)
_SPANS: list = []
_ACTIVE: dict = {}
_SPANS_N = 64
_spans_lock = threading.Lock()


def overlap(t0: float, t1: float) -> float:
    """Seconds of [t0, t1] (perf_counter) during which a card sample ran (one in flight too)."""
    now = time.perf_counter()
    with _spans_lock:
        spans = _SPANS + [(a, now) for a in _ACTIVE.values()]
    return sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in spans)


def _r1(v):
    return None if v is None else round(float(v), 1)


def compact(d: dict) -> dict:
    """smi.query's result -> the dashboard's card: what each panel shows, None where n/a."""
    from ..host import smi
    out = {"t": d.get("time") or time.time(), "device": d.get("device"), "ok": bool(d.get("ok")),
           "link": d.get("link"), "pcie": d.get("pcie"), "pcie_max": d.get("pcie_max"),
           "source": "card", "bitstream": None, "util": None, "dram_gbs": None,
           "dram_rd_gbs": None, "dram_wr_gbs": None, "die_c": None, "die_src": None,
           "gov": None, "reg": None, "power": None, "process": None, "ddr_mts": None}
    bs = d.get("bitstream")
    if bs:
        out["bitstream"] = {"build": None if bs.get("build_id") is None else
                            f"{bs['build_id']:08x}", "core_mhz": bs.get("core_mhz"),
                            "D": bs.get("D"), "MCOLS": bs.get("MCOLS"), "LANES": bs.get("LANES")}
        out["ddr_mts"] = bs.get("ddr_mts")
    u = d.get("util")
    if u:
        out["util"] = {k: round(float(u[k]), 4) for k in UNITS if u.get(k) is not None}
        out["util"]["DRAM"] = round(float(u.get("DRAM") or 0.0), 4)
        for k in ("dram_gbs", "dram_rd_gbs", "dram_wr_gbs"):
            v = d.get(k)
            out[k] = None if v is None else round(float(v), 3)
    bc = d.get("bcu")
    if bc:
        g = bc["gov"]
        die, src = smi.die_now(bc)
        out["die_c"], out["die_src"] = (None if die is None else round(die, 2)), src or None
        # causes: STATUS [22:16], why the last RUN ended (sticky until CMD CLEAR: in RUN, the
        # previous stop's); trips_thermal: TRIPS [15:0], the thermal stops since the governor's
        # reset (the die, the alarms, the regulator, a reading gone; never the host's SAFE)
        out["gov"] = {"state": "RUN" if g["run"] else "SAFE", "acc_on": g["acc_on"],
                      "dimms_on": g["dimms_on"], "causes": g["causes"], "run_s": g["run_ms"] / 1e3,
                      "heartbeat_ms": g.get("heartbeat_ms"), "trip_c": _r1(g.get("trip_c")),
                      "resume_c": _r1(g.get("resume_c")), "reg_trip_c": _r1(g.get("reg_trip_c")),
                      "alarm": g["alarm"], "ot": g["ot"], "frozen": bc.get("gov_frozen"),
                      "trips_thermal": g["trips_thermal"],
                      "trips_heartbeat": g.get("trips_heartbeat")}
        w = (bc.get("bmc") or {}).get("watch")
        if w:
            out["reg"] = {"hot_c": w.get("hot_c"), "p0_c": w.get("p0_c"), "p1_c": w.get("p1_c"),
                          "ctl_c": w.get("ctl_c"), "age_ms": w.get("hot_age_ms"),
                          "stopped": w.get("stopped")}
        elif g.get("has_reg") and g.get("reg_c") is not None:
            out["reg"] = {"hot_c": g["reg_c"], "p0_c": None, "p1_c": None, "ctl_c": None,
                          "age_ms": g.get("reg_age_ms"), "stopped": False}
    elif d.get("temp_c") is not None:               # another card: its XADC die (R_TEMP)
        out["die_c"], out["die_src"] = round(float(d["temp_c"]), 2), "XADC"
    # power: what smi measured, as it labels it, never checked against a meter ("verified"
    # False). On the BCU1525 it is the LTC3884's READ_PIN (the BMC watch's W_PIN, or a BMC1
    # session's): the input through that controller's sense resistor, which UG1268's VCCINT
    # (two LTC3874 slaves beside it, off PMBus) may not all pass: a lower bound of VCCINT's
    # input (docs/bcu1525.md section 13). The Vivado estimate rides along when there is one.
    ms = d.get("measured") or {}
    est = d.get("power") if d.get("power") and d["power"].get("w") is not None else None
    if ms.get("w") is not None:
        bmc = ms.get("source") == "watch" or any(r.get("bus") == "bmc"
                                                  for r in ms.get("rails") or ())
        out["power"] = {"w": round(float(ms["w"]), 2), "source": "measured",
                        "label": ms.get("label") or "measured", "age_ms": ms.get("age_ms"),
                        "vin": ms.get("vin"), "how": ms.get("how"), "verified": False,
                        "reading": "LTC3884 READ_PIN" if bmc and ms.get("how") == "PIN" else
                        ms.get("how"),
                        "estimate_w": None if est is None else round(float(est["w"]), 2)}
    elif est is not None:
        out["power"] = {"w": round(float(est["w"]), 2), "source": "estimate",
                        "label": "Vivado estimate from utilization", "age_ms": None, "vin": None,
                        "how": None, "verified": False, "reading": None, "estimate_w": None}
    st = d.get("process")
    if st:
        out["process"] = {"pid": st.get("pid"), "model": st.get("model"),
                          "weights": st.get("weights"), "stale": st.get("stale"),
                          "tokens": st.get("tokens"), "dram": st.get("dram")}
    return out


class CardSampler:
    """smi.query every `interval` seconds on a thread (start(); stop()). latest: the last
    compact sample (None before the first), error: the last failure's text (None after a good
    sample); samples, failures; cpu_s: the thread's CPU seconds (time.thread_time), its cost.
    open_transport(dev) -> a register transport (default XdmaTransport(dev, dma=False));
    on_sample(card) after each one (the monitor's history)."""

    def __init__(self, dev: str, interval: float = INTERVAL, open_transport=None,
                 bmc_every: float = BMC_EVERY, i2c: bool = False, on_sample=None):
        self.dev, self.interval, self.bmc_every, self.i2c = dev, max(0.5, interval), bmc_every, i2c
        self.open_transport = open_transport
        self.on_sample = on_sample
        self.latest: dict | None = None
        self.error: str | None = None
        self.samples = self.failures = 0
        self.cpu_s = 0.0
        self._t = None
        self._prev = None               # the last sample's counters (utilization since them)
        self._track: dict = {}          # smi's frozen-die tracking across samples
        self._pj = None                 # the power report, resolved once (smi.power_report)
        self._bmc1 = False              # a bridge without the watch: sessions bmc_every apart
        self._bmc_at = -1e9
        self._last_measured = None
        # the samples' timing (s): the last one's duration and the time since the one before,
        # and the most of each (a slow sample spaces the series' points beyond `interval`)
        self.took_s = self.took_max_s = self.spacing_s = self.spacing_max_s = None
        self._at = None
        self._stop = threading.Event()
        self.thread = None

    def start(self) -> "CardSampler":
        self.thread = threading.Thread(target=self._loop, daemon=True, name="otpu-telemetry")
        self.thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def _transport(self):
        if self._t is None:
            if self.open_transport is not None:
                self._t = self.open_transport(self.dev)
            else:
                from ..host.board import XdmaTransport
                self._t = XdmaTransport(self.dev, dma=False)
        return self._t

    def sample(self) -> dict:
        """One sample now (the thread's; tests call it directly)."""
        from ..host import smi
        from ..host import bmc as B
        c0 = time.thread_time()
        now = time.monotonic()
        if self._at is not None:
            self.spacing_s = round(now - self._at, 3)
            self.spacing_max_s = max(self.spacing_max_s or 0.0, self.spacing_s)
        self._at = now
        with _spans_lock:
            _ACTIVE[id(self)] = p0 = time.perf_counter()
        try:
            bmc = not self._bmc1 or now - self._bmc_at >= self.bmc_every
            d = smi.query(self._transport(), self.dev, prev=self._prev,
                          power_json=self._pj or smi.POWER_JSON, i2c=self.i2c, bmc=bmc,
                          sysmon=False, track=self._track)
            if self._pj is None and d.get("ok"):
                self._pj = smi.power_report(d) or Path("/nonexistent/power.json")
            self._prev = d.get("counters")
            br = (d.get("bcu") or {}).get("bmc")
            if bmc and br:
                self._bmc1 = br.get("id") == B.ID_BMC
                if self._bmc1:
                    self._bmc_at = now
                    if d.get("measured"):
                        self._last_measured = (now, d["measured"])
            elif self._bmc1 and self._last_measured and not d.get("measured"):
                at, m = self._last_measured           # (between sessions: the last, aged)
                d["measured"] = dict(m, age_ms=round(1e3 * (now - at)))
            out = compact(d)
            self.latest, self.error = out, None
            self.samples += 1
            return out
        except Exception as e:          # (OSError, a BMC error: told in the snapshot)
            self.failures += 1
            self.error = f"{type(e).__name__}: {e}"
            self._t = None if isinstance(e, OSError) else self._t
            self._prev = None
            raise
        finally:
            with _spans_lock:
                _ACTIVE.pop(id(self), None)
                _SPANS.append((p0, time.perf_counter()))
                del _SPANS[:-_SPANS_N]
            self.cpu_s += time.thread_time() - c0
            self.took_s = round(time.monotonic() - now, 3)
            self.took_max_s = max(self.took_max_s or 0.0, self.took_s)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = self.sample()
                if self.on_sample is not None:
                    self.on_sample(out)
            except Exception:
                pass
            self._stop.wait(self.interval)


def _all(vals) -> bool:
    return all(v is not None for v in vals)


def machine(cards: list, names: list, devs: list) -> dict:
    """Cards' compact samples (None: a card's failed) as one machine's, the form the monitor and
    the dashboard read: the hottest die and regulator, the summed DRAM traffic and power, the
    units' mean busy, the governor in RUN only when every card's is (its causes and thermal
    trips every card's); "cards" each card's own sample by name. A value a card lacks is None
    here too, never the others' sum or mean alone."""
    ok = [c for c in cards if c is not None]
    out = {"t": max(c["t"] for c in ok) if ok else time.time(), "device": "+".join(devs),
           "ok": len(ok) == len(cards) and all(c.get("ok") for c in ok), "source": "card",
           "cards": [dict(c, name=n) if c is not None else {"name": n, "device": d, "ok": False}
                     for c, n, d in zip(cards, names, devs)]}
    full = len(ok) == len(cards)
    for k in ("link", "pcie", "pcie_max"):
        v = [c.get(k) for c in ok]
        out[k] = (v[0] if len(set(v)) == 1 else " / ".join(map(str, v))) if full and _all(v) \
            else None
    b = [c.get("bitstream") for c in ok]
    out["bitstream"] = dict(b[0], build="/".join(sorted({x["build"] for x in b if x.get("build")})))\
        if full and _all(b) else None
    v = [c.get("ddr_mts") for c in ok]
    out["ddr_mts"] = v[0] if full and _all(v) and len(set(v)) == 1 else None
    us = [c.get("util") for c in ok]
    if full and _all(us):
        keys = set.intersection(*(set(u) for u in us))
        out["util"] = {k: round(sum(u[k] for u in us) / len(us), 4) for k in sorted(keys)}
    else:
        out["util"] = None
    for k in ("dram_gbs", "dram_rd_gbs", "dram_wr_gbs"):
        v = [c.get(k) for c in ok]
        out[k] = round(sum(v), 3) if full and _all(v) else None
    dies = [c.get("die_c") for c in ok]
    if full and _all(dies):
        i = max(range(len(ok)), key=lambda j: dies[j])
        out["die_c"] = dies[i]
        out["die_src"] = f"{ok[i].get('die_src') or 'n/a'}, {names[i]}"
    else:
        out["die_c"] = out["die_src"] = None
    gs = [c.get("gov") for c in ok]
    if full and _all(gs):
        dimms = [tuple(g.get("dimms_on") or ()) for g in gs]
        out["gov"] = {
            "state": "RUN" if all(g["state"] == "RUN" for g in gs) else "SAFE",
            "acc_on": all(g.get("acc_on") for g in gs),
            "dimms_on": list(dimms[0]) if len(set(dimms)) == 1 else None,
            "causes": sorted({x for g in gs for x in (g.get("causes") or ())}),
            "run_s": min(g["run_s"] for g in gs),
            "heartbeat_ms": max(g["heartbeat_ms"] for g in gs)
            if _all(g.get("heartbeat_ms") for g in gs) else None,
            "trip_c": min(g["trip_c"] for g in gs) if _all(g.get("trip_c") for g in gs) else None,
            "resume_c": min(g["resume_c"] for g in gs)
            if _all(g.get("resume_c") for g in gs) else None,
            "reg_trip_c": min(g["reg_trip_c"] for g in gs)
            if _all(g.get("reg_trip_c") for g in gs) else None,
            "alarm": any(g.get("alarm") for g in gs), "ot": any(g.get("ot") for g in gs),
            "frozen": any(g.get("frozen") for g in gs),
            "trips_thermal": sum(g.get("trips_thermal") or 0 for g in gs),
            "trips_heartbeat": sum(g.get("trips_heartbeat") or 0 for g in gs)}
    else:
        out["gov"] = None
    rs = [c.get("reg") for c in ok]
    out["reg"] = max(rs, key=lambda r: r.get("hot_c") if r.get("hot_c") is not None else -1e9) \
        if full and _all(rs) else None
    ps = [c.get("power") for c in ok]
    if full and _all(ps) and _all(p.get("w") for p in ps) and \
            len({(p.get("source"), p.get("reading")) for p in ps}) == 1:
        p0 = ps[0]
        out["power"] = dict(p0, w=round(sum(p["w"] for p in ps), 2),
                            label=f"{p0.get('label') or p0.get('source')}, {len(ps)} cards summed",
                            age_ms=max(p.get("age_ms") or 0 for p in ps), vin=None,
                            estimate_w=round(sum(p["estimate_w"] for p in ps), 2)
                            if _all(p.get("estimate_w") for p in ps) else None)
    else:
        out["power"] = None
    out["process"] = ok[0].get("process") if ok and cards[0] is not None else None
    return out


class CardsSampler:
    """Cards in tensor parallel (otpu-serve35x2: the two BCU1525s as one machine) as one
    CardSampler's interface: a CardSampler a card (its own register transport, smi's default
    reads, its own BMC bridge under that bridge's lock), the cards sampled one after the other
    every `interval` seconds on one thread. latest: machine()'s view of the last samples (a card
    that failed: None in it, its error in `error` by name). devs: the cards' devices, card 0
    first; names: theirs to show (default "card 0", "card 1", ...)."""

    def __init__(self, devs, interval: float = INTERVAL, open_transport=None,
                 bmc_every: float = BMC_EVERY, i2c: bool = False, on_sample=None, names=None):
        self.devs = list(devs)
        self.names = list(names) if names else [f"card {c}" for c in range(len(self.devs))]
        self.dev = "+".join(self.devs)
        self.cards = [CardSampler(d, interval, open_transport, bmc_every, i2c)
                      for d in self.devs]
        self.interval = self.cards[0].interval
        self.on_sample = on_sample
        self.latest: dict | None = None
        self.error: str | None = None
        self.samples = self.failures = 0
        self.took_s = self.took_max_s = self.spacing_s = self.spacing_max_s = None
        self._at = None
        self._stop = threading.Event()
        self.thread = None

    @property
    def cpu_s(self) -> float:
        return sum(c.cpu_s for c in self.cards)

    def start(self) -> "CardsSampler":
        self.thread = threading.Thread(target=self._loop, daemon=True, name="otpu-telemetry")
        self.thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def sample(self) -> dict:
        """Every card's sample now, then the machine's (raises when no card answered)."""
        now = time.monotonic()
        if self._at is not None:
            self.spacing_s = round(now - self._at, 3)
            self.spacing_max_s = max(self.spacing_max_s or 0.0, self.spacing_s)
        self._at = now
        outs, errs = [], []
        try:
            for n, c in zip(self.names, self.cards):
                try:
                    outs.append(c.sample())
                except Exception:
                    outs.append(None)
                    errs.append(f"{n}: {c.error}")
        finally:
            self.took_s = round(time.monotonic() - now, 3)
            self.took_max_s = max(self.took_max_s or 0.0, self.took_s)
        self.error = "; ".join(errs) or None
        if all(o is None for o in outs):
            self.failures += 1
            raise RuntimeError(self.error)
        out = machine(outs, self.names, self.devs)
        self.latest = out
        self.samples += 1
        if errs:
            self.failures += 1
        return out

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = self.sample()
                if self.on_sample is not None:
                    self.on_sample(out)
            except Exception:
                pass
            self._stop.wait(self.interval)


class DemoCard:
    """A synthetic BCU1525 for otpu-serve --demo (and the dashboard's screenshot): smooth
    utilization, temperature and power that follow `busy()` (0 idle .. 1 decoding, 2 prefill).
    Every sample says "source": "demo"."""

    def __init__(self, busy=lambda: 0.0, interval: float = INTERVAL, on_sample=None,
                 seed: int = 1):
        import random
        self.busy, self.interval, self.on_sample = busy, interval, on_sample
        self.r = random.Random(seed)
        self.latest, self.error, self.samples, self.failures, self.cpu_s = None, None, 0, 0, 0.0
        self._die, self._pw, self._t0 = 58.0, 31.0, time.time()
        self._stop = threading.Event()
        self.thread = None

    def start(self) -> "DemoCard":
        self.thread = threading.Thread(target=self._loop, daemon=True, name="otpu-demo-card")
        self.thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def sample(self) -> dict:
        b = float(self.busy())
        j = lambda s: self.r.gauss(0.0, s)                      # noqa: E731
        if b >= 2:      # a layer-major prompt: the MXU busy, DeltaNet-bound
            u = {"RUNNING": 0.97, "MXU_BUSY": 0.62, "MXU_MAC": 0.48, "VPU_BUSY": 0.31,
                 "QNT_BUSY": 0.06, "DMA_BUSY": 0.21, "DRAM": 0.55}
        elif b >= 1:    # decode: DRAM-bound
            u = {"RUNNING": 0.93, "MXU_BUSY": 0.34, "MXU_MAC": 0.17, "VPU_BUSY": 0.22,
                 "QNT_BUSY": 0.04, "DMA_BUSY": 0.41, "DRAM": 0.86}
        else:
            u = {"RUNNING": 0.0, "MXU_BUSY": 0.0, "MXU_MAC": 0.0, "VPU_BUSY": 0.0,
                 "QNT_BUSY": 0.0, "DMA_BUSY": 0.0, "DRAM": 0.0}
        u = {k: round(min(1.0, max(0.0, v + (j(0.015) if v else 0.0))), 4) for k, v in u.items()}
        gbs = u["DRAM"] * 150e6 * 128 / 1e9 * 2
        tgt_die = 58.5 + 9.0 * min(b, 1.6)
        self._die += 0.18 * (tgt_die - self._die) + j(0.08)
        tgt_pw = 30.0 + 21.0 * u["RUNNING"] * (0.6 + 0.5 * u["DRAM"])
        self._pw += 0.35 * (tgt_pw - self._pw) + j(0.25)
        reg = self._die - 9.0 + 0.6 * math.sin((time.time() - self._t0) / 40)
        out = {"t": time.time(), "device": "/dev/xdma0", "ok": True, "source": "demo",
               "link": "ok", "pcie": "8.0 GT/s PCIe x16", "pcie_max": "8.0 GT/s PCIe x16",
               "bitstream": {"build": "8bf10e8f", "core_mhz": 150.0, "D": 128, "MCOLS": 8,
                             "LANES": 8}, "ddr_mts": 2400,
               "util": u, "dram_gbs": round(gbs, 3), "dram_rd_gbs": round(gbs * 0.97, 3),
               "dram_wr_gbs": round(gbs * 0.03, 3), "die_c": round(self._die, 2),
               "die_src": "governor",
               "gov": {"state": "RUN", "acc_on": True, "dimms_on": [0, 1, 2, 3], "causes": [],
                       "run_s": time.time() - self._t0 + 3600.0, "heartbeat_ms": 310,
                       "trip_c": 80.0, "resume_c": 65.0, "reg_trip_c": 78.0, "alarm": False,
                       "ot": False, "frozen": None, "trips_thermal": 0, "trips_heartbeat": 0},
               "reg": {"hot_c": round(reg, 2), "p0_c": round(reg - 1.5, 2),
                       "p1_c": round(reg, 2), "ctl_c": round(reg - 4.0, 2), "age_ms": 180,
                       "stopped": False},
               "power": {"w": round(self._pw, 2), "source": "measured",
                         "label": "VCCINT regulator input", "age_ms": 240, "vin": 12.08,
                         "how": "PIN", "verified": False, "reading": "LTC3884 READ_PIN",
                         "estimate_w": None},
               "process": None}
        self.latest = out
        self.samples += 1
        return out

    def _loop(self) -> None:
        while not self._stop.is_set():
            out = self.sample()
            if self.on_sample is not None:
                self.on_sample(out)
            self._stop.wait(self.interval)


# ------------------------------------------------------------------------------ the host
class HostStats:
    """The host's CPU (busy share since the last call, all cores), memory and load, and this
    process's CPU share and resident memory: /proc on Linux (the card host), psutil elsewhere if
    it is installed; None for what neither gives."""

    def __init__(self):
        self._cpu = None            # (busy, total) jiffies at the last call
        self._proc = None           # (process CPU seconds, monotonic) at the last call
        self.ncpu = os.cpu_count() or 1
        try:
            import psutil
            self.ps = psutil
            psutil.cpu_percent(None)
        except Exception:
            self.ps = None

    def _proc_stat(self):
        try:
            f = Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]
        except OSError:
            return None
        v = [int(x) for x in f]
        idle = v[3] + (v[4] if len(v) > 4 else 0)
        return sum(v) - idle, sum(v)

    def _meminfo(self):
        try:
            kv = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, _, rest = line.partition(":")
                kv[k] = int(rest.split()[0]) * 1024
            return kv["MemTotal"], kv["MemTotal"] - kv.get("MemAvailable", kv["MemFree"])
        except (OSError, KeyError, ValueError, IndexError):
            return None

    def _rss(self):
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            pass
        if self.ps is not None:
            try:
                return self.ps.Process().memory_info().rss
            except Exception:
                return None
        return None

    def sample(self) -> dict:
        out = {"cpu": None, "ncpu": self.ncpu, "mem_total": None, "mem_used": None,
               "load1": None, "proc_cpu": None, "proc_rss": self._rss()}
        s = self._proc_stat()
        if s is not None:
            if self._cpu is not None and s[1] > self._cpu[1]:
                out["cpu"] = round((s[0] - self._cpu[0]) / (s[1] - self._cpu[1]), 4)
            self._cpu = s
        elif self.ps is not None:
            out["cpu"] = round(self.ps.cpu_percent(None) / 100.0, 4)
        m = self._meminfo()
        if m is None and self.ps is not None:
            vm = self.ps.virtual_memory()
            m = (vm.total, vm.total - vm.available)
        if m is not None:
            out["mem_total"], out["mem_used"] = m
        try:
            out["load1"] = round(os.getloadavg()[0], 2)
        except OSError:
            pass
        t = os.times()
        now = (t.user + t.system, time.monotonic())
        if self._proc is not None and now[1] > self._proc[1]:
            out["proc_cpu"] = round((now[0] - self._proc[0]) / (now[1] - self._proc[1]), 4)
        self._proc = now
        return out
