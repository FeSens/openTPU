#!/usr/bin/env python3
"""Decode slot hits after a prompt, by the slots' state at the prompt's end (docs/offload.md 13.18).

Offline, on offload's router traces (~/otpu-build/offload/traces on the build host: <model>.<text>.npz, each MoE
layer's top-k of 2048 tokens, L<i>_idx). Per MoE layer S slots (the card's: the 35B 41 = 1640 / 40, the 26B
16 = 480 / 30). Every window of a prompt of P rows (its last TAIL rows layer-major, as the chat's "157E +
4L") and then T decode rows (the trace's next rows: a proxy for a reply's routing), every layer and text;
the decode under LRU. The slots' state at the prompt's end:
  lm    layer-major: LRU over the prompt's rows, each row's experts in turn;
  em    expert-major today: the experts loaded in the pass table's order (each one's first entry in the
        chunk), the last S kept, then the tail rows;
  rec   expert-major with its passes and loads ordered by each expert's last row: the last S by last
        appearance (layer-major's state, no more bytes), then the tail;
  freq  the prompt's S most frequent experts, then the tail.
Prints each state's decode hit rate, misses a token and MB a token.

  python3 tools/offload/em_slotsim.py TRACES [--prompt 157] [--decode 16] [--tail 4]
"""
import argparse
from pathlib import Path

import numpy as np

MODELS = {"qwen35-35b": (41, 1.712), "gemma26": (16, 3.81)}    # slots a layer, MB an expert (MX pools)
KINDS = ("lm", "em", "rec", "freq")


def touch(state: list, ids, S: int, count=None) -> None:
    """LRU: state holds the slots' experts, the most recent last; ids a row's experts."""
    h = 0
    for e in ids:
        if e in state:
            state.remove(e)
            h += 1
        elif len(state) >= S:
            state.pop(0)
        state.append(e)
    if count is not None:
        count[0] += h
        count[1] += len(ids) - h


def end_state(kind: str, rows: np.ndarray, S: int, tail: int) -> list:
    """The slots after a prompt of `rows` ([P, k] expert ids) under `kind`."""
    if kind == "lm":
        st: list = []
        for r in rows:
            touch(st, r, S)
        return st
    em, last_rows = rows[:len(rows) - tail], rows[len(rows) - tail:]
    if kind == "em":
        order = list(dict.fromkeys(int(e) for r in em for e in r))
    elif kind == "rec":
        last = {int(e): i for i, r in enumerate(em) for e in r}
        order = sorted(last, key=last.get)
    else:
        c = np.bincount(em.reshape(-1))
        order = [int(e) for e in np.argsort(c, kind="stable")[::-1][:S]][::-1]
    st = order[-S:]
    for r in last_rows:
        touch(st, r, S)
    return st


def run(traces: Path, model: str, P: int, T: int, tail: int) -> dict:
    S, _ = MODELS[model]
    res = {k: [0, 0] for k in KINDS}
    layers = 0
    for f in sorted(traces.glob(f"{model}.*.npz")):
        d = np.load(f)
        keys = sorted((k for k in d.files if k.endswith("_idx")), key=lambda k: int(k[1:-4]))
        layers = len(keys)
        n = d[keys[0]].shape[0]
        for o in range(0, n - P - T + 1, P + T):
            for key in keys:
                idx = d[key].astype(int)
                for kind in KINDS:
                    st = end_state(kind, idx[o:o + P], S, tail)
                    for r in idx[o + P:o + P + T]:
                        touch(st, r, S, res[kind])
    return dict(layers=layers, slots=S, **{k: tuple(v) for k, v in res.items()})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("traces", type=Path)
    ap.add_argument("--prompt", type=int, default=157)
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--tail", type=int, default=4)
    a = ap.parse_args(argv)
    for m, (S, mb) in MODELS.items():
        r = run(a.traces, m, a.prompt, a.decode, a.tail)
        print(f"{m}: prompt {a.prompt}, decode {a.decode}, {S} slots a layer, {r['layers']} MoE layers")
        for k in KINDS:
            h, mi = r[k]
            k_ = 8 * r["layers"] * mi / (h + mi)             # misses a token (top-8 a layer)
            print(f"  {k:5s} hits {h / (h + mi):.4f}  misses a token {k_:6.1f}  MB a token {k_ * mb:7.1f}")


if __name__ == "__main__":
    main()
