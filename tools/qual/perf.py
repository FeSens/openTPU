"""Card qualification: prefill tok/s on a long prompt, then decode with the card's free-running
counters (device cycles, DRAM traffic and efficiency, MXU starvation) and wall time.

    python3 tools/qual/perf.py MODEL WFORMAT [HEAD_FORMAT] [--prompt 512] [--tokens 64]

Run from a host tree (the repo root). One engine per call, with resident decode where the model
has it (the check that resident decode computes the ISA simulator's tokens is refs.py card
--resident, after the warm soak).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from opentpu.host import chat as C  # noqa: E402
from opentpu.llm import load_spec, model_dir  # noqa: E402
from opentpu.llm import qwen3 as Q  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("wformat")
    ap.add_argument("head_format", nargs="?")
    ap.add_argument("--prompt", type=int, default=512, help="prefill tokens")
    ap.add_argument("--tokens", type=int, default=64, help="decode tokens")
    a = ap.parse_args()
    m, wf, hf = a.model, a.wformat, (a.head_format if a.head_format not in (None, "-") else None)
    path = model_dir(m)
    spec, tok = load_spec(path), AutoTokenizer.from_pretrained(path)
    backend, cfg = C.make_backend("board", spec, 2048, "/dev/xdma0", path.name)
    eng = Q.Engine(spec, Q.load_weights(path), cap=2048, cfg=cfg, backend=backend,
                   wformat=wf, head_format=hf, resident=True)
    board, info = eng.backend.board, eng.backend.info
    khz = info["core_khz"]
    text = (ROOT / "tools/data/austen_pp_ch1.txt").read_text()
    ids = tok(text)["input_ids"][:a.prompt]
    t0, n0 = time.perf_counter(), len(eng.stats)
    logits = eng.prefill(ids)
    wall = time.perf_counter() - t0
    pcyc = sum(s["cycles"] for s in eng.stats[n0:])
    print(f"{path.name} ({wf}{', head ' + hf if hf else ''}): prefill {len(ids)} tokens in "
          f"{len(eng.stats) - n0} runs: device {len(ids) * khz * 1e3 / pcyc:.1f} tok/s, "
          f"wall {len(ids) / wall:.1f} tok/s (compile included); resident={eng.resident}")
    t = int(np.argmax(logits))
    n, s0 = a.tokens, board.snapshot()
    n0 = len(eng.stats)
    t0 = time.perf_counter()
    for _ in range(n):
        t = int(np.argmax(eng.step(t)))
    eng._drain()
    wall = time.perf_counter() - t0
    s1 = board.snapshot()
    dev = sum(s["cycles"] for s in eng.stats[n0:])
    d = {k: s1[k] - s0[k] for k in ("DRAM_RD", "DRAM_WR", "RUNNING", "MXU_STARVE", "DRAM_WAIT")
         if k in s0}
    run_s = d["RUNNING"] / (khz * 1e3)
    peak = 2 * 8 * info["ddr_mts"] * 1e6 / 1e9 if info.get("ddr_mts") else 17.06
    gbs = (d["DRAM_RD"] + d["DRAM_WR"]) * 64 / run_s / 1e9
    print(f"  decode {n} tokens: {dev / n / 1e6:.3f} Mcycles/token, device "
          f"{n * khz * 1e3 / dev:.2f} tok/s, wall {n / wall:.2f} tok/s (argmax loop, not "
          f"streamed); DRAM read {d['DRAM_RD'] * 64 / n / 1e6:.1f} MB/token, write "
          f"{d['DRAM_WR'] * 64 / n / 1e6:.2f} MB/token; while running {gbs:.2f} GB/s "
          f"(read {d['DRAM_RD'] * 64 / run_s / 1e9:.2f}) = {100 * gbs / peak:.0f}% of "
          f"{peak:.1f} GB/s"
          + (f"; MXU_STARVE {100 * d['MXU_STARVE'] / d['RUNNING']:.0f}%" if "MXU_STARVE" in d else "")
          + (f"; DRAM_WAIT {100 * d['DRAM_WAIT'] / d['RUNNING']:.0f}%" if "DRAM_WAIT" in d else ""))
    eng.backend.close()


if __name__ == "__main__":
    main()
