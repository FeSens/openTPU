"""Decode-step cost of a model's first N layers on the card, from the free-running counters:
where a token's cycles go, layer kind by layer kind, to set against tools/perf_qwen.py's
simulated cycles for the same --layers.

    python3 tools/q35_split.py [--model qwen35] [--layers 1,3,4,0] [--pos 60] [--reps 3]
                               [--wformat fp4] [--head-format int8] [--json out.json]

For each N (0: every layer) the model is cut to its first N layers (the LM head stays whole),
its image written to the card, and --reps decode steps run at position --pos (per-position
programs, the KV cache before --pos empty: the timing does not depend on it). Prints per N the
step's cycles (the min over the reps) and the counters over one step: DRAM read / write bytes,
MXU busy / MAC / starved, VPU, DMA, TMEM_DENY and DRAM_WAIT cycles.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu.host import chat as C  # noqa: E402
from opentpu.host import regs as R  # noqa: E402
from opentpu.llm import load_spec, model_dir  # noqa: E402
from opentpu.llm import qwen3 as Q  # noqa: E402

KEYS = ("RUNNING", "MXU_BUSY", "MXU_MAC", "MXU_STARVE", "VPU_BUSY", "QNT_BUSY", "DMA_BUSY",
        "TMEM_DENY", "DRAM_RD", "DRAM_WR", "DRAM_WAIT", "INSTR")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35")
    ap.add_argument("--layers", default="1,3,4,0")
    ap.add_argument("--pos", type=int, default=60)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--wformat", default="fp4")
    ap.add_argument("--head-format", default="int8")
    ap.add_argument("--backend", default="board", choices=["board", "board-sim"])
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    path = model_dir(a.model)
    full = load_spec(path)
    W = Q.load_weights(path)
    cap = 256 * (a.pos // 256 + 1)
    out = []
    for n in (int(x) for x in a.layers.split(",")):
        spec = dataclasses.replace(full, **({"kinds": full.kinds[:n]} if hasattr(full, "kinds")
                                            else {"layers": n})) if n else full
        backend, cfg = C.make_backend(a.backend, spec, cap, a.dev, path.name)
        eng = Q.Engine(spec, W, cap=cap, cfg=cfg, backend=backend, wformat=a.wformat,
                       head_format=a.head_format, pipeline=False)
        board = eng.backend.board
        steps = []
        try:
            for _ in range(a.reps):
                eng.pos = a.pos
                s0 = board.snapshot()
                eng.step(791)
                s1 = board.snapshot()
                d = {k: s1[k] - s0[k] for k in KEYS if k in s0}
                d["cycles"] = eng.stats[-1]["cycles"]
                steps.append(d)
        finally:
            eng.backend.close()
        best = min(steps, key=lambda d: d["cycles"])
        best["layers"] = spec.layers if hasattr(spec, "layers") else n
        best["kinds"] = list(getattr(spec, "kinds", []))
        best["all_cycles"] = [d["cycles"] for d in steps]
        out.append(best)
        run = max(1, best.get("RUNNING", best["cycles"]))
        print(f"N={n or 'all'} ({best['layers']} layers): {best['cycles']} cycles "
              f"(reps {best['all_cycles']}); DRAM read {best['DRAM_RD'] * R.DRAM_BEAT / 1e6:.2f} MB,"
              f" write {best['DRAM_WR'] * R.DRAM_BEAT / 1e6:.2f} MB; " + ", ".join(
                  f"{k} {100 * best[k] / run:.1f}%" for k in
                  ("MXU_BUSY", "MXU_MAC", "MXU_STARVE", "VPU_BUSY", "DMA_BUSY", "TMEM_DENY",
                   "DRAM_WAIT") if k in best), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
