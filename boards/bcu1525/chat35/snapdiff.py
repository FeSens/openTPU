"""snapdiff.py: an engine's context as it stands in each slice's DRAM (every KV cache array up to the engine's position, by
page, and the DeltaNet states: opentpu.serve.kvio.Layout's regions), saved by chat35x2.py (CHAT35_SNAP=DIR), and two
such snapshots compared: snapdiff.py A.npz B.npz prints each slice's arrays SAME or DIFF (bytes that differ, the first
page and position), the states the same way, and exits 0 when all are the same. Card against IsaCards at the first key
whose logits differ: which cache or state went first (docs: the card-vs-ISA snapshot diff)."""
import json
import sys

import numpy as np


def snap(eng) -> dict:
    """{name: bytes}: slice s's KV array j up to the engine's position ("s{s}.kv{j}") and its DeltaNet states
    ("s{s}.st"), with "meta" (the position, each array's base / width / tile, the page's positions P)."""
    from opentpu.serve.kvio import Layout
    lay = Layout(eng.image, eng.cfg.S)
    pages = -(-eng.pos // lay.P)
    out = {}
    for s in range(eng.cfg.S):
        for j, a in enumerate(lay.arrays):
            n = pages * lay.P * a.w
            out[f"s{s}.kv{j}"] = np.asarray(eng.backend.read(s, a.base, n), np.uint8) if n else np.zeros(0, np.uint8)
        st = [np.asarray(eng.backend.read(s, b, n), np.uint8) for b, n in lay.state]
        out[f"s{s}.st"] = np.concatenate(st) if st else np.zeros(0, np.uint8)
    meta = {"pos": int(eng.pos), "P": int(lay.P), "S": int(eng.cfg.S),
            "arrays": [[int(a.base), int(a.w), bool(a.tile)] for a in lay.arrays],
            "state": [[int(b), int(n)] for b, n in lay.state],
            "regions": [[int(b), int(n)] for b, n in eng.image.state_regions(0) if n] if lay.stateful else []}
    out["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    return out


def save(eng, path) -> None:
    np.savez(path, **snap(eng))


def state_at(meta, i: int) -> str:
    """Where byte i of the concatenated states sits: its DRAM address and the image's state region (state_regions'
    order: layer by layer, a DeltaNet layer's windows and states) holding it."""
    for b, n in meta["state"]:
        if i < n:
            a = b + i
            for r, (rb, rn) in enumerate(meta.get("regions", [])):
                if rb <= a < rb + rn:
                    return f" (DRAM {a:#x}: region {r} of {len(meta['regions'])}, {a - rb} bytes into its {rn})"
            return f" (DRAM {a:#x})"
        i -= n
    return ""


def main(a, b) -> int:
    A, B = np.load(a), np.load(b)
    ma, mb = (json.loads(bytes(z["meta"]).decode()) for z in (A, B))
    print(f"A pos {ma['pos']}, B pos {mb['pos']} (P {ma['P']}, S {ma['S']}, {len(ma['arrays'])} KV arrays, "
          f"{sum(n for _, n in ma['state'])} state bytes a slice)")
    bad = 0
    for k in sorted((set(A.files) & set(B.files)) - {"meta"}, key=lambda k: (int(k[1:k.index(".")]), k)):
        x, y = A[k], B[k]
        n = min(len(x), len(y))
        d = np.flatnonzero(x[:n] != y[:n])
        if not len(d) and len(x) == len(y):
            print(f"{k}: SAME ({n} bytes)")
            continue
        bad += 1
        where = ""
        if ".kv" in k and len(d):
            base, w, tile = ma["arrays"][int(k.split("kv")[1])]
            page = int(d[0]) // (ma["P"] * w)
            where = f", first in page {page}" + ("" if tile else f" position {page * ma['P'] + int(d[0]) % (ma['P'] * w) // w}")
        elif len(d):
            where = f", first at state byte {int(d[0])}" + state_at(ma, int(d[0]))
        print(f"{k}: DIFF {len(d)} of {n} bytes{where}" + (f" (lengths {len(x)} / {len(y)})" if len(x) != len(y) else ""))
    print("snapdiff: ALL SAME" if not bad else f"snapdiff: {bad} regions differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
