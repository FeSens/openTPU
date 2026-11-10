"""The layer loop (opentpu/llm/prefill_loop.py) on the board model: one 35B layer's N layer-major runs of R rows, one device run
each (the engine's prefill_layers today), against the same runs as one layer loop
(prefill_loop.compile_layer_loop, Engine(layer_loop=True)): device cycles and instructions of
both, the bits checked (docs/pf500.md).

    python3 tools/exp/pf500_loop.py isa|sim [--target bcu1525-wg:bhw16] [--kinds dn,attn] [--R 2]
                                        [--N 8] [--pos 1024] [--json FILE]

pf500's cut, zero image and routing (route_ids over the N R rows, route_words: each row's residual
one-hot), layer LAYER[kind] at positions pos, pos + R, ... in one attention bucket. isa: the runs
one after another on one DRAM (A), the loop on a fresh one (B): their residual rows must be equal,
and the ledger's units and instructions of both. sim: the board model after the ISA runs, the runs
(each its own load / set_args / run, the words written once) then the routing words again and the
loop (the ISA's C: the loop on A's DRAM, the board's state as it finds it): each run's and the
loop's instructions and residual rows checked against the ISA's, cycles summed over the runs
against the loop's. The host's gaps between runs (a run's start and wait) are not in the
board's cycles: the loop's saving there is the card's (chat35 CHAT35_LOOP).
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pf500 as P                                               # noqa: E402


def compile_both(img, kind: str, R: int, N: int, pos: int):
    """([(programs, ARG words)] the N runs, (programs, ARG words) their loop)."""
    from opentpu.compiler import arg_words
    from opentpu.llm.prefill_loop import compile_layer_loop
    from opentpu.llm.qwen3 import ATTN_BLOCK as B, RunPos
    li, K = P.LAYER[kind], img.spec.conv_k
    if pos % B + N * R > B:
        raise ValueError(f"{N} runs of {R} rows from {pos}: past the attention block")
    blocks = pos // B + 1
    progs, ra = img.compile_layer_run(li, blocks, R=R, embedded=True)
    runs = [(progs, arg_words(ra, {"row": i * R, **RunPos.values(0, pos + i * R, K, B)}))
            for i in range(N)]
    lp, la = compile_layer_loop(img, li, blocks, n=R)
    v = RunPos.values(0, pos, K, B)
    loop = (lp, arg_words(la, {**v, "tpos0": v["tpos"], "row0": 0, "runs": N, "tpos": 0,
                               "row": 0}))
    return runs, loop


def residual(img, dram, rows: int) -> np.ndarray:
    xb = img.io["xbuf"]
    return np.array(dram[xb:xb + 4 * img.spec.hidden * rows])


def read(sb, addr: int, n: int) -> np.ndarray:
    got = sb.read(addr, n, 0)
    return np.frombuffer(got, np.uint8) if isinstance(got, (bytes, bytearray)) else np.asarray(got)


def board_seq(sb, img, seq, words) -> dict:
    """The programs one after another on the board, each its own run (the words written
    first): cycles and instructions of each."""
    b, at = sb.board, P.prog_at(img)
    for s, ws in enumerate(words):
        for a, v in ws:
            b.write(a, v, s)
    out = []
    for progs, args in seq:
        b.load(at, progs if len(progs) > 1 else progs[0])
        b.set_args(list(args) + [0] * (8 - len(args)))
        t0 = time.time()
        st = b.run()
        out.append({"cycles": int(st["cycles"]), "instructions": list(st["instructions"])})
        print(f"board: {out[-1]['cycles']} cycles in {time.time() - t0:.0f} s", file=sys.stderr,
              flush=True)
    return {"cycles": sum(r["cycles"] for r in out), "runs": out}


def one(a, tg, img, sb, kind: str) -> dict:
    R, N = a.R, a.N
    rec = {"mode": a.cmd, "target": tg.name, "kind": kind, "R": R, "N": N, "pos": a.pos}
    t0 = time.time()
    runs, loop = compile_both(img, kind, R, N, a.pos)
    rec["compile_s"] = round(time.time() - t0, 1)
    rec["program"] = {"run": len(runs[0][0][0]), "loop": len(loop[0][0])}
    ids = P.route_ids(N * R, img.spec.moe.E, img.spec.moe.k, a.seed)
    words, live = P.route_words(img, P.LAYER[kind], ids)
    # A: the runs on one DRAM; B: the loop on a fresh one; C: the loop on A's (the board's)
    da, ka, ra = [], [], []
    for j, (progs, args) in enumerate(runs):
        ra.append(P.run_isa(tg, progs, args, words if j == 0 else None, live, ka, da))
    rA = residual(img, da[0], N * R)
    db, kb = [], []
    rb = P.run_isa(tg, loop[0], loop[1], words, live, kb, db)
    rB = residual(img, db[0], N * R)
    rec["bits_equal"] = bool(np.array_equal(rA, rB))
    del db, kb
    tot = {u: sum(r["total"][u] for r in ra) for u in P.UNITS}
    rec["isa"] = {"runs": {"instructions": sum(r["instructions"] for r in ra), "total": tot},
                  "loop": {"instructions": rb["instructions"],
                           "total": {u: rb["total"][u] for u in P.UNITS}}}
    if sb is None:
        return rec
    kc: list = []
    P.run_isa(tg, loop[0], loop[1], words, live, kc, da)
    rC = residual(img, da[0], N * R)
    xb, n = img.io["xbuf"], 4 * img.spec.hidden * N * R
    t0 = time.time()
    br = board_seq(sb, img, runs, words)
    bad = [] if np.array_equal(read(sb, xb, n), rA) else [
        "runs: residual rows differ from the ISA's"]
    bl = board_seq(sb, img, [loop], words)
    rec["sim_s"] = round(time.time() - t0)
    for j, (r, m) in enumerate(zip(br["runs"], ka)):
        if r["instructions"] != [sl.icount for sl in m.slices]:
            bad.append(f"run {j}: instructions {r['instructions']} != ISA")
    if bl["runs"][0]["instructions"] != [sl.icount for sl in kc[0].slices]:
        bad.append(f"loop: instructions {bl['runs'][0]['instructions']} != ISA")
    if not np.array_equal(read(sb, xb, n), rC):
        bad.append("loop: residual rows differ from the ISA's")
    rec["ok"] = not bad and rec["bits_equal"]
    if bad:
        rec["mismatch"] = "; ".join(bad)
    rec["sim"] = {"runs": br, "loop": bl["runs"][0],
                  "saved": br["cycles"] - bl["cycles"],
                  "saved%": round(100 * (br["cycles"] - bl["cycles"]) / max(br["cycles"], 1), 2)}
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=("isa", "sim"))
    p.add_argument("--target", default="bcu1525-wg:bhw16")
    p.add_argument("--kinds", default="dn,attn")
    p.add_argument("--R", type=int, default=2)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--pos", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--model", default=P.MODEL)
    p.add_argument("--json", default=None)
    p.add_argument("--plus", default=P.K7, help="sim: the board model's plusargs")
    p.add_argument("--params", default="", help="sim: its build parameters, K=V,...")
    a = p.parse_args(argv)
    tg = P.target(a.target)
    img = P.image(tg, a.model, a.pos)
    sb = P.SimBoard(tg, P.prog_at(img), a.plus, a.params, 65536) if a.cmd == "sim" else None
    rows = []
    for kind in a.kinds.split(","):
        try:
            rec = one(a, tg, img, sb, kind)
        except Exception as e:
            rec = {"kind": kind, "error": f"{type(e).__name__}: {e}"[:300]}
        print(json.dumps(rec), flush=True)
        rows.append(rec)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"peak RSS {rss / (2**30 if sys.platform == 'darwin' else 2**20):.2f} GiB",
          file=sys.stderr)
    if a.json:
        Path(a.json).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
