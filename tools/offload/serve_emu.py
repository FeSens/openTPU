#!/usr/bin/env python3
"""Path (a)'s host side alone, on the card's host without the card (docs/offload.md 10.8): the
real ExpertServer, BoardDram, PoolFile and XdmaTransport code, with each XDMA call replaced by a
wait of its cost on the card (asleep, the GIL released, as in the driver; its last 60 us spun for
an exact end), and the card's requests replayed from a session.

The card here posts each request (a session trace's layer and misses: hits drawn from the
server's slots, misses from outside them) a device gap after the previous served landed, and
answers the poll's reads of seq and of the row. It records when each request is posted, seen,
its last new entry landed and served landed. The DMA calls are logged, so the link's idle time
between them can be split by what comes before and after.

  TRACE  a moe_card --hint-trace timeline ([t0, t1, kind, layer, misses] per poll: its "d"
         requests and their gaps), or a moe_card result (misses_per_request_decode, the layers
         in turn, --gap's device gap)
  POOL   a split-format pool file of the model's slot size (--pool-n: its first experts only,
         g mod N; keep them in the page cache)

Prints one JSON line (appended to --out):
- detect_s: post -> seen, summed;
- crit_s: seen -> the request's last new entry landed (what the card waits for);
- window_s: seen -> poll returns (moe_card's poll, less its seq read);
- link_busy_s: the calls' modelled time;
- gaps: the link's idle between consecutive calls in a window, by the calls' kinds
  (D an expert's part, e a new entry, c a victim's clear, s served; "q": nothing queued when
  the first ended): [count, s, us each].

--legacy serves as before docs/offload.md 10.8 (moe_card --legacy-serve). Needs a C compiler
(cc) for the wait, Linux for its timer slack.
Usage: PYTHONPATH=. python tools/offload/serve_emu.py TRACE POOL --model q35 [--legacy]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

WAIT_C = r"""
#include <time.h>
#include <sys/prctl.h>
static long now(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000000000L + t.tv_nsec;
}
void spin_ns(long ns) { long e = now() + ns; while (now() < e); }
static __thread int slack;
/* a DMA call's wait: asleep as in the driver, its last 60 us spun */
void wait_ns(long ns) {
  long e = now() + ns, s = ns - 60000;
  if (s > 0) {
    if (!slack) { prctl(PR_SET_TIMERSLACK, 1000UL, 0, 0, 0); slack = 1; }
    struct timespec t = {s / 1000000000L, s % 1000000000L}; nanosleep(&t, 0);
  }
  while (now() < e);
}
"""

# the card's configurations (sessions 11 and 12): experts per layer, per request, MoE layers,
# slots per layer, slot bytes
MODELS = {"q35": (256, 8, 40, 42, 1671168), "g26": (128, 8, 30, 18, 3446784)}


def wait_lib():
    d = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "otpu"
    so = d / f"serve_emu-{os.uname().machine}.so"
    if not so.exists():
        d.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as t:
            c = Path(t) / "wait.c"
            c.write_text(WAIT_C)
            subprocess.check_call(["cc", "-O2", "-shared", "-fPIC", "-o", str(so), str(c)])
    lib = ctypes.CDLL(str(so))          # (CDLL: the GIL released during a call)
    lib.spin_ns.argtypes = lib.wait_ns.argtypes = [ctypes.c_long]
    return lib


def requests(path, layers: int, gap: float, detect: float):
    """[(layer, misses)] and each one's device gap after the served before it."""
    x = json.loads(Path(path).read_text())
    if isinstance(x, dict):             # a moe_card result: its decode's misses, layers in turn
        m = x["misses_per_request_decode"]
        return [(i % layers, int(v)) for i, v in enumerate(m)], [gap or 1.5e-3] * len(m)
    d = [e for e in x if e[2] == "d"]
    g = [2e-3] + [max(0.2e-3, d[i][0] - d[i - 1][1] - detect) for i in range(1, len(d))]
    return [(int(e[3]) % layers, int(e[4])) for e in d], [gap] * len(d) if gap else g


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trace")
    ap.add_argument("pool")
    ap.add_argument("--model", choices=sorted(MODELS), default="q35")
    ap.add_argument("--legacy", action="store_true", help="serve as before 10.8")
    ap.add_argument("--n", type=int, default=0, help="the first N requests (0: all)")
    ap.add_argument("--gap", type=float, default=0.0,
                    help="the device's gap from served to its next post, s (default: the "
                         "trace's, less --detect)")
    ap.add_argument("--detect", type=float, default=0.1e-3,
                    help="the card's detection latency in the trace's gaps (s)")
    ap.add_argument("--pool-n", type=int, default=0, help="experts in POOL (0: the model's)")
    ap.add_argument("--h2c-call", type=float, default=20e-6, help="a write call's cost (s)")
    ap.add_argument("--h2c-bps", type=float, default=1.365e9, help="the link's write rate")
    ap.add_argument("--c2h-call", type=float, default=22e-6, help="a beat read's cost (s)")
    ap.add_argument("--reg", type=float, default=1.5e-6, help="a register read's cost (s)")
    ap.add_argument("--out", help="append the result's JSON line here")
    a = ap.parse_args()
    from opentpu.host import board as bd
    from opentpu.host import offload as off

    lib = wait_lib()
    E, k, nl, slots, slot = MODELS[a.model]
    L = off.Layout.build(0x1000_0000, E, k, [slots] * nl, slot)
    G = E * nl
    R, gaps = requests(a.trace, nl, a.gap, a.detect)
    if a.n:
        R, gaps = R[:a.n], gaps[:a.n]
    st = dict(r=0, posted=None, ids=[], post=[], seen=[], entry=[], served=[], busy=0.0)
    rng = np.random.default_rng(1)
    beats, calls, srv_ref = {}, [], []

    def addr_of(ch, o):                 # a beat's logical address (CHASH)
        m = o // 64
        return m * 128 + 64 * (ch ^ (int(m).bit_count() & 1))

    def post_due(now):
        """The next request, if it is posted by now: its ids drawn against the slots."""
        r = st["r"]
        if r >= len(R) or st["posted"] == r:
            return
        t_prev = st["served"][r - 1] if r else st["t0"]
        if t_prev is None or now < t_prev + gaps[r]:
            return
        srv = srv_ref[0]
        j, m = R[r]
        have = [g for g in srv.lru[j] if g not in srv.pending]
        m = min(m, k, E - len(have))
        out = [g for g in range(j * E, (j + 1) * E) if g not in srv.lru[j]]
        ids = [int(g) for g in list(rng.choice(have, k - m, replace=False)) +
               list(rng.choice(out, m, replace=False))]
        rng.shuffle(ids)
        st["ids"].append(ids)           # (this draw's own time is not the card's)
        st["post"].append(t_prev + gaps[r] + time.perf_counter() - now)
        for x in ("seen", "entry", "served"):
            st[x].append(None)
        st["posted"] = r

    def card_read(addr, n):
        b = np.zeros(64, np.uint8)
        if addr == L.mbox:
            post_due(time.perf_counter())
            r = st["posted"]
            if r is not None and st["seen"][r] is None and st["served"][r] is None:
                st["seen"][r] = time.perf_counter()
            b[:8] = np.frombuffer(np.array([0 if r is None else r + 1, k], np.float32)
                                  .tobytes(), np.uint8)
        elif addr == L.row:
            ids = st["ids"][st["posted"]]
            b[:4 * len(ids)] = np.frombuffer(np.array(ids, np.float32).tobytes(), np.uint8)
        return b[:n]

    def card_write(addr, data) -> str:
        now = time.perf_counter()
        if addr == L.served:
            r = int(np.frombuffer(bytes(data[:4]), np.float32)[0]) - 1
            if 0 <= r < len(st["served"]):
                st["served"][r] = now
                st["r"] = r + 1
            return "s"
        if L.dir <= addr < L.dir + 8 * G:
            w = np.frombuffer(bytes(data), np.uint32).copy()
            old, beats[addr] = beats.get(addr), w
            new = w[1::2] != 0
            if old is not None:
                new &= old[1::2] == 0
            if new.any() and st["posted"] is not None:
                st["entry"][st["posted"]] = now
            return "e" if new.any() else "c"
        return "?"

    class FakeOs:
        """board.py's os: pwrite / lseek / readinto on the fake device files (fds -2, -3)."""
        pos = 0

        def __getattr__(self, name):
            return getattr(os, name)

        def pwrite(self, fd, buf, at):
            if fd != -2:
                return os.pwrite(fd, buf, at)
            n = len(buf)
            t = a.h2c_call + n / a.h2c_bps
            t0 = time.perf_counter()
            lib.wait_ns(int(t * 1e9))
            t1 = time.perf_counter()
            st["busy"] += t
            ch, o = (1, at - bd.BASE[1]) if at >= bd.BASE[1] else (0, at)
            kind = card_write(addr_of(ch, o), np.frombuffer(buf, np.uint8)) if n == 64 else "D"
            calls.append((t0, t1, kind, mem._q.qsize() == 0))
            return n

        def lseek(self, fd, at, how):
            if fd != -3:
                return os.lseek(fd, at, how)
            self.pos = at
            return at

        def readinto(self, fd, mv):
            if fd != -3:
                return os.readinto(fd, mv)
            lib.spin_ns(int(a.c2h_call * 1e9))
            ch, o = (1, self.pos - bd.BASE[1]) if self.pos >= bd.BASE[1] else (0, self.pos)
            mv[:] = card_read(addr_of(ch, o), len(mv)).tobytes()
            return len(mv)

    bd.os = FakeOs()
    t = object.__new__(bd.XdmaTransport)
    t.dev, t.devname, t._otpu_lock = "/dev/serve-emu", "serve-emu", None
    t._dma = bd._DmaLock("serve-emu")       # (its own flock file: <run dir>/serve-emu.dma)
    t.h2c, t.c2h, t.words, t.regs = -2, -3, None, None

    class FakeBoard:
        chash = True

        def __init__(self, t):
            self.t = t

        def write(self, addr, b):           # (the card's own words: load's mailbox)
            lib.wait_ns(int((a.h2c_call + len(b) / a.h2c_bps) * 1e9))

    class Backend:
        board = FakeBoard(t)
    mem = off.BoardDram(Backend(), L)
    pf = off.PoolFile(a.pool, L.slot_bytes, True)
    nfile = a.pool_n or G
    srv = off.ExpertServer(mem, L, lambda g: pf.get(g % nfile), policy="lfu")
    srv_ref.append(srv)
    srv.load([j * E + e for j in range(nl) for e in range(E)])
    idle = t.host_idle
    if a.legacy:                            # (moe_card --legacy-serve)
        mem.lead, srv.clear_late, pf.iov, idle = None, False, False, bd.HOST_IDLE
    pf.io, srv.events = {}, []              # (as moe_card's decode)
    dma0, wait0 = mem.dma_s, mem.wait_s
    calls.clear()
    st["t0"] = time.perf_counter()
    st["busy"] = 0.0
    while st["r"] < len(R):                 # BoardBackend._serve's loop
        lib.spin_ns(int(a.reg * 1e9))       # (R_STATUS)
        if not srv.poll() and idle:
            time.sleep(idle)
    wall = time.perf_counter() - st["t0"]

    ev = [e for e in srv.events if e[2] == "d"]
    w = np.array([e[1] - e[0] for e in ev])
    mi = np.array([e[4] for e in ev])
    seen = np.array(st["seen"], float)
    det = seen - np.array(st["post"])
    crit = np.array([np.nan if e is None else e for e in st["entry"]], float) - seen
    fit = np.linalg.lstsq(np.vstack([np.ones_like(mi), mi]).T.astype(float), w * 1e3,
                          rcond=None)[0]
    gaps_ = {}
    for x, y in zip(calls[:-1], calls[1:]):
        if x[2] == "s":                     # served: the window's end
            continue
        g = gaps_.setdefault(f"{x[2]}>{y[2]}" + ("q" if x[3] else ""), [0, 0.0])
        g[0] += 1
        g[1] += y[0] - x[1]
    print(json.dumps(r := dict(
        model=a.model, legacy=a.legacy, requests=len(w), misses=int(mi.sum()),
        wall_s=round(wall, 3), window_s=round(w.sum(), 3), crit_s=round(np.nansum(crit), 3),
        detect_s=round(det.sum(), 3), link_busy_s=round(st["busy"], 3),
        dma_s=round(mem.dma_s - dma0, 3), stage_wait_s=round(mem.wait_s - wait0, 3),
        fit_window_ms=[round(x, 4) for x in fit],
        window_ms_by_misses={int(m): round(float(np.median(w[mi == m])) * 1e3, 3)
                             for m in range(k + 1) if (mi == m).sum() > 3},
        reads={x: [y[0], round(y[1], 3)] for x, y in pf.io.items()},
        gaps={x: [y[0], round(y[1], 3), round(y[1] / y[0] * 1e6, 1)]
              for x, y in sorted(gaps_.items())})))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    sys.exit(main())
