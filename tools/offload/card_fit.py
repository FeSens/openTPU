#!/usr/bin/env python3
"""The card's side of serve_emu.py, from a card session's runs (moe_card --hint-trace timelines
with their .calls.json; docs/offload.md 10.13):

  post-c TRACE OUT [--onecall]   per request, the card's time from the previous request's
      critical end to its post (serve_emu --post-c). The critical end of a request with misses
      is its last tag landed: design A's last data call (--onecall), else the old path's first
      64-byte call after it (its last new entry); of a request with none, its post.
  w RUN... [--k K] [--cap S]   F and W(m) (serve_emu --card-w) from design A runs: the next post
      = max(critical end + F, post + W(m)). F is the median critical end -> next post of the
      requests with 3 or more misses (the card waited for their last tag); W(m) per miss count m
      the value that fits the runs' next posts best (median absolute error), requests whose
      next post came within --cap s of their critical end only (a token's end excluded). Also
      W0, the median post -> next post of the requests with no miss: the card's own time, the
      floor for levers that shorten the windows far (W(m) holds the fit runs' waits).

A post is the next request's seen less DET (the poll's detection, 15 us). Prints the fit and the
measured / modelled time after the critical end by misses; w prints the --card-w string.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

DET = 15e-6
BIG = 100_000                           # a data call (an expert's part), bytes


def _requests(trace: str, onecall: bool = True):
    """Per request of the trace ("d" polls): its post, critical end, misses and the next one's
    post, as rows [post, ce, next post, misses] (the last request has no next)."""
    ev = [e for e in json.loads(Path(trace).read_text()) if e[2] == "d"]
    calls = np.array(json.loads(Path(trace[:-5] + ".calls.json").read_text()))
    idx = np.searchsorted(calls[:, 0], [e[0] for e in ev])
    out = []
    for i, e in enumerate(ev):
        ce = e[0] - DET
        if e[4]:
            j = k = idx[i]
            while k < len(calls) and calls[k, 0] < e[1]:
                k += 1
            w = calls[j:k]
            big = np.nonzero(w[:, 2] >= BIG)[0]
            if len(big):
                last = big[-1]
                if onecall:
                    ce = w[last, 1]
                else:
                    nxt = [c for c in w[last + 1:] if c[2] == 64]
                    ce = nxt[0][1] if nxt else w[last, 1]
        nxt = ev[i + 1][0] - DET if i + 1 < len(ev) else np.nan
        out.append((e[0] - DET, ce, nxt, e[4]))
    return np.array(out, float)


def post_c(a) -> None:
    r = _requests(a.trace, a.onecall)
    c = [0.0] + list(r[1:, 0] - r[:-1, 1])
    json.dump(c, open(a.out, "w"))
    print(f"{a.trace}: {len(r)} requests; C median {np.median(c[1:]) * 1e3:.3f} ms, "
          f"sum {np.sum(c):.2f} s -> {a.out}")


def fit_w(a) -> None:
    files = [f for d in a.runs for f in sorted(glob.glob(f"{d}/*.trace.json"))] or a.runs
    r = np.vstack([_requests(f)[:-1] for f in files])
    r = r[(r[:, 3] > 0) & (r[:, 2] - r[:, 1] < a.cap)]
    post, ce, nxt, m = r.T
    F = float(np.median((nxt - ce)[m >= 3]))
    W = {}
    for k in range(1, a.k + 1):
        s = m == k
        if s.sum() < 30:
            continue
        ws = np.arange(0.0, a.cap, 10e-6)
        err = [np.median(np.abs(np.maximum(ce[s] + F, post[s] + w) - nxt[s])) for w in ws]
        W[k] = float(ws[int(np.argmin(err))])
    model = np.maximum(ce + F, post + np.array([W.get(int(x), 0.0) for x in m]))
    allr = np.vstack([_requests(f)[:-1] for f in files])
    hits = allr[(allr[:, 3] == 0) & (allr[:, 2] - allr[:, 0] < a.cap)]
    W0 = float(np.median(hits[:, 2] - hits[:, 0])) if len(hits) else float("nan")
    by = " ".join(f"{k}:{np.median((nxt - ce)[m == k]) * 1e6:.0f}/"
                  f"{np.median((model - ce)[m == k]) * 1e6:.0f}" for k in W)
    print(f"{len(files)} runs, {len(r)} requests: F {F * 1e6:.0f} us; after the critical end "
          f"(us, measured/model) by misses {by}; next posts' error median "
          f"{np.median(model - nxt) * 1e6:+.0f} us")
    print(f"W0 {W0 * 1e6:.0f} us: the card's own time a request ({len(hits)} with no miss)")
    print("--card-w " + ",".join([f"{F:.4g}"] + [f"{k}:{w:.4g}" for k, w in W.items() if w > 0]))
    print(f"--card-w {F:.4g},0:{W0:.4g}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("post-c")
    p.add_argument("trace")
    p.add_argument("out")
    p.add_argument("--onecall", action="store_true", help="design A's runs (the tag last)")
    p.set_defaults(fn=post_c)
    p = sub.add_parser("w")
    p.add_argument("runs", nargs="+", help="run directories (their *.trace.json) or traces")
    p.add_argument("--k", type=int, default=8, help="the model's experts per request")
    p.add_argument("--cap", type=float, default=8e-3, help="s: a token's end above it")
    p.set_defaults(fn=fit_w)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
