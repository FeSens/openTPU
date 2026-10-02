#!/usr/bin/env python3
"""Path (a)'s host side alone, on the card's host without the card (docs/offload.md 10.8): the
real ExpertServer, BoardDram, PoolFile and XdmaTransport code, with each XDMA call replaced by a
wait of its cost on the card (asleep, the GIL released, as in the driver; its last 60 us spun for
an exact end), and the card's requests replayed from a session.

The card here posts each request (a session trace's layer and misses: hits drawn from the
server's slots, misses from outside them) and answers the poll's reads of seq and of the row. It
records when each request is posted, seen, its last tag landed (docs/offload.md 10.11: the last
missing expert's) and served landed. The DMA calls are logged, so the link's idle time between
them can be split by what comes before and after. When the card posts:
- default: the trace's gap after each served (less --detect), or --gap;
- --post-c C.json: the card's compute from the previous request's critical end, per request
  (card_fit.py post-c on the trace's run), and no sooner than served (its fence);
- --card-w F,m:W,...: with --post-c, the card's own compute too (docs/offload.md 10.13): no
  sooner than the critical end + F, nor than the previous post + W(m) for a request of m
  misses (the card computing its present experts while the link brings the rest); a C above
  F + 1 ms (a token's end) stays. card_fit.py w fits F and W on a session's runs.
The link: --h2c-call + bytes / --h2c-bps a write call, --beat-call a 64-byte one, --size-cost
the card's measured cost of the parts' sizes, --c2h-call a beat read. Gen2 on the card (the
link: g2check's calibration; --card-w: session 16's fit), the 35B:
  --size-cost 167936:162.0e-6,667648:298.3e-6,835584:329.5e-6 --h2c-call 6.3e-6
  --h2c-bps 2.585e9 --beat-call 35.6e-6 --c2h-call 19.4e-6
  --card-w 1.705e-3,1:2.88e-3,2:3.48e-3,3:4.13e-3,4:4.77e-3,5:5.56e-3,6:6.19e-3,7:6.77e-3
the 26B: --size-cost 344064:213.5e-6,1379328:658.8e-6,1723392:672.9e-6 --beat-call 49.3e-6
  --c2h-call 20.3e-6 --card-w 2.956e-3,1:6.39e-3,2:7.08e-3,3:7.89e-3 (the same call and rate).

  TRACE  a moe_card --hint-trace timeline ([t0, t1, kind, layer, misses] per poll: its "d"
         requests and their gaps), or a moe_card result (misses_per_request_decode, the layers
         in turn, --gap's device gap)
  POOL   a split-format pool file of the model's slot size (--pool-n: its first experts only,
         g mod N; keep them in the page cache)

Prints one JSON line (appended to --out):
- detect_s: post -> seen, summed;
- crit_s: seen -> the request's last tag landed (what the card waits for);
- window_s: seen -> poll returns (moe_card's poll, less its seq read);
- link_busy_s: the calls' modelled time;
- gaps: the link's idle between consecutive calls in a window, by the calls' kinds
  (a the answer, D an expert's part, T its last (the tag its last beat), e the directory, t
  a tag's clear, s served; "q": nothing queued when the first ended): [count, s, us each].

--legacy serves as before docs/offload.md 10.8 (moe_card --legacy-serve). Needs a C compiler
(cc) for the wait, Linux for its timer slack (elsewhere the waits are as exact as the OS
sleeps). Run it on the card's host only between card sessions: it reads the pool and spins a
core.
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
#ifdef __linux__
#include <sys/prctl.h>
#endif
static long now(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000000000L + t.tv_nsec;
}
void spin_ns(long ns) { long e = now() + ns; while (now() < e); }
static __thread int slack;
/* a DMA call's wait: asleep as in the driver, its last 60 us spun (Linux: 1 us timer slack) */
void wait_ns(long ns) {
  long e = now() + ns, s = ns - 60000;
  if (s > 0) {
#ifdef __linux__
    if (!slack) { prctl(PR_SET_TIMERSLACK, 1000UL, 0, 0, 0); slack = 1; }
#endif
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
    so = d / f"serve_emu-{sys.platform}-{os.uname().machine}.so"
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


def card_busy() -> str:
    """Who uses the card, as runstate sees it, without touching its lock (DeviceLock: a monitor
    must not flock it): a live holder in a device's lock file, a live runner's status, or a live
    session in the card sessions' quiet file (env.sh's OTPU_QUIET: between a session's steps).
    "" when none."""
    from opentpu.host.runstate import pid_alive, read_status, run_dir
    who = []
    for lock in sorted(run_dir().glob("xdma*.lock")):
        if lock.name.count(".") > 1:        # (xdma0.i2c.lock: the board's I2C)
            continue
        try:
            pid = int(lock.read_text().split()[0])
        except (OSError, ValueError, IndexError):
            continue
        if pid_alive(pid):
            who.append(f"{lock.name} held by {pid}")
        st = read_status(lock.stem)
        if st and not st["stale"]:
            who.append(f"{lock.stem} runner {st['pid']}")
    q = Path(os.environ.get("OTPU_QUIET", Path.home() / "otpu-build" / "QUIET"))
    try:
        pid = int(q.read_text().split()[0])
        if pid_alive(pid):
            who.append(f"{q} session {pid}")
    except (OSError, ValueError, IndexError):
        pass
    return "; ".join(who)


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
    ap.add_argument("--beat-call", type=float, default=0.0,
                    help="a 64-byte write call's whole cost (s; default --h2c-call's: session "
                         "13's card measured 49-61 us)")
    ap.add_argument("--h2c-bps", type=float, default=1.365e9, help="the link's write rate")
    ap.add_argument("--c2h-call", type=float, default=22e-6, help="a beat read's cost (s)")
    ap.add_argument("--reg", type=float, default=1.5e-6, help="a register read's cost (s)")
    ap.add_argument("--out", help="append the result's JSON line here")
    ap.add_argument("--tag", default="", help="a label for the result line")
    ap.add_argument("--post-c", help="JSON list: per request the card's time from the previous "
                    "request's critical end (its last new entry / tag landed; its post with no "
                    "miss) to this post (c_r.py); default: the trace's gap after served")
    ap.add_argument("--card-w", default="",
                    help="F,m:W,...: the card model with its own compute (session 16): the next "
                         "post no sooner than the critical end + F (s), nor than its post + W(m) "
                         "for a request of m misses (the card never waiting); a trace C_r above "
                         "F + 1 ms (a token's end) is kept as is")
    ap.add_argument("--size-cost", default="",
                    help="pwrite bytes:seconds,...: the card's measured cost of a write of about "
                         "that size (within 256 bytes)")
    ap.add_argument("--win-write", type=float, default=0.0,
                    help="B: a 64-byte host write through the BAR window, s of the worker's CPU "
                         "(no link time; 0: a DMA call)")
    ap.add_argument("--win-load", type=float, default=0.0,
                    help="B: a 32-bit load through the window (s): the poll's seq one, the row "
                         "its words and the count (0: a C2H call)")
    ap.add_argument("--dump", help="per request: post, seen, critical end, served (npz)")
    ap.add_argument("--no-willneed", action="store_true", help="no PoolFile.willneed (ahead)")
    ap.add_argument("--probe", help="per request: the host's stamps from seen to the first data "
                    "DMA (JSON)")
    ap.add_argument("--force", action="store_true",
                    help="run while the card is in use (card_busy: its timing would suffer)")
    ap.add_argument("--dir-beats", action="store_true",
                    help="each changed directory beat as its own 64-byte write (not a span)")
    a = ap.parse_args()
    busy = card_busy()
    if busy and not a.force:
        sys.exit(f"serve_emu: the card is in use ({busy}): run it between card sessions")
    from opentpu.host import board as bd
    from opentpu.host import offload as off

    lib = wait_lib()
    E, k, nl, slots, slot = MODELS[a.model]
    L = off.Layout.build(0x1000_0000, E, k, [slots] * nl, slot)
    G = E * nl
    R, gaps = requests(a.trace, nl, a.gap, a.detect)
    if a.n:
        R, gaps = R[:a.n], gaps[:a.n]
    st = dict(r=0, posted=None, ids=[], post=[], seen=[], entry=[], served=[], busy=0.0,
              win=0.0, nwin=0, nload=0)
    onecall = bool(getattr(L, "tag", 0))       # design A's layout (else the entry protocol)
    post_c = json.loads(Path(a.post_c).read_text()) if a.post_c else None
    if post_c is not None and a.n:
        post_c = post_c[:a.n]
    card_f, card_w = None, {}
    if a.card_w:
        v = a.card_w.split(",")
        card_f = float(v[0])
        card_w = {int(x): float(y) for x, y in (u.split(":") for u in v[1:])}
    size_cost = [(int(x), float(y)) for x, y in
                 (v.split(":") for v in a.size_cost.split(",") if v)]
    rng = np.random.default_rng(1)
    calls, srv_ref, beats = [], [], {}

    def addr_of(ch, o):                 # a beat's logical address (CHASH)
        m = o // 64
        return m * 128 + 64 * (ch ^ (int(m).bit_count() & 1))

    def post_due(now):
        """The next request, if it is posted by now: its ids drawn against the slots."""
        r = st["r"]
        if r >= len(R) or st["posted"] == r:
            return
        if not r:
            due = st["t0"] + gaps[0]
        elif st["served"][r - 1] is None:
            return
        elif post_c is not None:            # the card's compute from the critical end; its
            ce = st["entry"][r - 1]         # fence: served
            ce = st["post"][r - 1] if ce is None else ce
            c = post_c[r]
            if card_f is not None and st["entry"][r - 1] is not None and c <= card_f + 1e-3:
                c = card_f                  # (the card's own compute: W below)
            due = max(ce + c, st["served"][r - 1])
            if card_f is not None and st["entry"][r - 1] is not None:
                due = max(due, st["post"][r - 1] + card_w.get(R[r - 1][1], 0.0))
        else:
            due = st["served"][r - 1] + gaps[r]
        if now < due:
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
        st["post"].append(due + time.perf_counter() - now)
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

    tags = {s + L.tag for s in L.all_slots()} if onecall else set()

    def card_write(addr, data, n) -> str:
        """A call of n bytes whose last beat is at addr landed."""
        now = time.perf_counter()
        if addr == L.served:
            r = int(np.frombuffer(bytes(data[:4]), np.float32)[0]) - 1
            if 0 <= r < len(st["served"]):
                st["served"][r] = now
                st["r"] = r + 1
            return "s"
        if onecall and addr == L.answer:
            return "a"
        if L.dir <= addr < L.dir + 8 * G:
            if onecall:
                return "e"
            w = np.frombuffer(bytes(data), np.uint32).copy()     # (the entry protocol: a
            old, beats[addr] = beats.get(addr), w                # new entry is critical)
            new = w[1::2] != 0
            if old is not None:
                new &= old[1::2] == 0
            if new.any() and st["posted"] is not None:
                st["entry"][st["posted"]] = now
            return "e" if new.any() else "c"
        if addr in tags:
            if n == 64:
                return "t"
            if st["posted"] is not None:        # an expert's last call: its tag landed
                st["entry"][st["posted"]] = now
            return "T"
        return "D" if n > 64 else "?"

    class FakeOs:
        """board.py's os: pwrite / lseek / readinto on the fake device files (fds -2, -3)."""
        pos = 0

        def __getattr__(self, name):
            return getattr(os, name)

        def pwrite(self, fd, buf, at):
            if fd != -2:
                return os.pwrite(fd, buf, at)
            n = len(buf)
            win = n == 64 and a.win_write
            if win:
                t = a.win_write
            elif n == 64 and a.beat_call:
                t = a.beat_call
            else:
                t = next((y for x, y in size_cost if abs(n - x) <= 256),
                         (a.beat_call if n <= 4096 and a.beat_call else a.h2c_call) + n / a.h2c_bps)
            t0 = time.perf_counter()
            lib.wait_ns(int(t * 1e9))
            t1 = time.perf_counter()
            if win:
                st["win"] += t
                st["nwin"] += 1
            else:
                st["busy"] += t
            ch, o = (1, at - bd.BASE[1]) if at >= bd.BASE[1] else (0, at)
            kind = card_write(addr_of(ch, o + n - 64), np.frombuffer(buf, np.uint8)[-64:], n)
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
            ch, o = (1, self.pos - bd.BASE[1]) if self.pos >= bd.BASE[1] else (0, self.pos)
            if a.win_load:                  # seq: one load; the row: its words and the count
                ad = addr_of(ch, o)
                nl = 1 if ad == L.mbox else -(-len(mv) // 4) + 1
                st["nload"] += nl
                lib.spin_ns(int(nl * a.win_load * 1e9))
            else:
                lib.spin_ns(int(a.c2h_call * 1e9))
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
    if not a.no_willneed:               # (as moe_card: its default)
        srv.ahead = pf.willneed
    if hasattr(pf, "defer_touch"):      # (as moe.serve)
        srv.pool_file, pf.defer_touch = pf, True
    stamps = []
    if a.probe:                         # the host's steps from seen to the first data DMA
        def stamp(name):
            r = st["posted"]
            if r is not None:
                stamps.append((r, name, time.perf_counter()))

        def wrap(obj, name, before=None, after=None):
            f = getattr(obj, name)

            def w(*x, **kw):
                if before is None or before(*x):
                    stamp(name + ">")
                out = f(*x, **kw)
                if after is None or after(*x):
                    stamp(name + "<")
                return out
            setattr(obj, name, w)
        wrap(mem, "read", lambda addr, n: addr == L.row, lambda addr, n: addr == L.row)
        wrap(srv, "serve")
        wrap(srv, "_slot")
        if getattr(srv, "ahead", None) is not None:
            wrap(srv, "ahead")
        wrap(srv, "pool")
        wrap(mem, "write_slot")
        wrap(mem, "write", lambda addr, d: addr == getattr(L, "answer", -1),
             lambda addr, d: addr == getattr(L, "answer", -1))
        wrap(mem, "_put")
        for o, n in ((pf, "_absent"), (pf, "_touch"), (pf, "_read_iov"), (mem, "_pair"),
                     (mem, "_staging"), (mem, "_iov"), (mem, "_cuts")):
            if hasattr(o, n):
                wrap(o, n)
    if a.dir_beats and hasattr(srv, "_dir_flush"):
        def dir_flush():                    # each changed beat alone
            if not srv._dirty:
                return
            per = off.LINE // 8
            for b in sorted({g // per for g in srv._dirty}):
                mem.write(L.entry(b * per), srv.dirv[b * per:(b + 1) * per])
            srv._dirty.clear()
        srv._dir_flush = dir_flush
    srv.load([j * E + e for j in range(nl) for e in range(E)])
    idle = t.host_idle
    if a.legacy:                            # (moe_card --legacy-serve)
        mem.lead, srv.clear_late, pf.iov, idle = None, False, False, bd.HOST_IDLE
    pf.io, srv.events = {}, []              # (as moe_card's decode)
    dma0, wait0 = mem.dma_s, mem.wait_s
    calls.clear()
    st["t0"] = time.perf_counter()
    st["busy"] = st["win"] = 0.0
    st["nwin"] = st["nload"] = 0
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
    kinds = {}
    for x in calls:
        kinds[x[2]] = kinds.get(x[2], 0) + 1
    print(json.dumps(r := dict(
        tag=a.tag, model=a.model, onecall=onecall, legacy=a.legacy, requests=len(w),
        misses=int(mi.sum()), win_s=round(st["win"], 3), win_writes=st["nwin"],
        win_loads=st["nload"], calls=kinds,
        wall_s=round(wall, 3), window_s=round(w.sum(), 3), crit_s=round(np.nansum(crit), 3),
        detect_s=round(det.sum(), 3), link_busy_s=round(st["busy"], 3),
        dma_s=round(mem.dma_s - dma0, 3), stage_wait_s=round(mem.wait_s - wait0, 3),
        fit_window_ms=[round(x, 4) for x in fit],
        window_ms_by_misses={int(m): round(float(np.median(w[mi == m])) * 1e3, 3)
                             for m in range(k + 1) if (mi == m).sum() > 3},
        reads={x: [y[0], round(y[1], 3)] for x, y in pf.io.items()},
        gaps={x: [y[0], round(y[1], 3), round(y[1] / y[0] * 1e6, 1)]
              for x, y in sorted(gaps_.items())})))
    if a.probe:
        Path(a.probe).write_text(json.dumps(dict(seen=st["seen"], stamps=stamps,
                                                 calls=[(x[0], x[1], x[2]) for x in calls])))
    if a.dump:
        np.savez(a.dump, post=np.array(st["post"], float), seen=seen,
                 entry=np.array([np.nan if e is None else e for e in st["entry"]], float),
                 served=np.array([np.nan if e is None else e for e in st["served"]], float),
                 misses=mi, calls=np.array([(x[0], x[1]) for x in calls]),
                 kinds=np.array([x[2] for x in calls]))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    sys.exit(main())
