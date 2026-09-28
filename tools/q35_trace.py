"""One decode step of a model cut to its first N layers, instruction by instruction, on the card
(the hardware trace, keep first) or on the RTL simulator (DDR3-1066 bank model, as
tools/perf_qwen.py --ddr 1066): dumps every traced instruction's timing to JSON, to set the
two side by side (where the card's cycles go that the model does not charge).

    python3 tools/q35_trace.py card|sim --out f.json [--model qwen35] [--layers 1] [--pos 60]
                                        [--wformat fp4] [--head-format int8] [--mhz 120.755]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu import isa as I  # noqa: E402
from opentpu import rtlsim  # noqa: E402
from opentpu.llm import load_spec, model_dir  # noqa: E402
from opentpu.llm import qwen3 as Q  # noqa: E402
from opentpu.profile import ddr3_plusargs, parse  # noqa: E402


def _dump(p, progs, cycles, out, extra):
    recs = []
    for r in sorted(p.slice_recs(0), key=lambda r: r.idx):
        ins = progs[0][r.pc]
        src = ins.src[1] if len(ins.src) > 1 else (ins.src[0] if ins.src else ("?", 0, "?"))
        recs.append({"idx": r.idx, "pc": r.pc, "op": r.op, "name": r.name, "detail": r.detail,
                     "dispatch": r.dispatch, "ready": r.ready, "start": r.start,
                     "release": r.release, "end": r.end, "portb": getattr(r, "portb", 0),
                     "src": f"{Path(src[0]).name}:{src[1]} {src[2]}"})
    Path(out).write_text(json.dumps({"cycles": cycles, "recs": recs, **extra}))
    print(f"{out}: {cycles} cycles, {len(recs)} instructions traced", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("where", choices=["card", "sim"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen35")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--pos", type=int, default=60)
    ap.add_argument("--wformat", default="fp4")
    ap.add_argument("--head-format", default="int8")
    ap.add_argument("--mhz", type=float, default=120.755)
    ap.add_argument("--dev", default="/dev/xdma0")
    a = ap.parse_args(argv)
    path = model_dir(a.model)
    spec = load_spec(path)
    if a.layers:
        spec = dataclasses.replace(spec, **({"kinds": spec.kinds[:a.layers]}
                                            if hasattr(spec, "kinds") else {"layers": a.layers}))
    W = Q.load_weights(path)
    cap = 256 * (a.pos // 256 + 1)
    if a.where == "card":
        from opentpu.host import chat as C
        from opentpu.host.hwlens import records_to_trace
        backend, cfg = C.make_backend("board", spec, cap, a.dev, path.name)
        eng = Q.Engine(spec, W, cap=cap, cfg=cfg, backend=backend, wformat=a.wformat,
                       head_format=a.head_format, pipeline=False)
        try:
            eng.pos = a.pos
            eng.step(791)                   # warm: the same run untraced
            eng.pos = a.pos
            progs = eng.image.compile_step(a.pos)
            eng.backend.trace = {"keep": "first"}
            eng._program = lambda pos: progs        # the traced run's programs, for parse
            eng.step(791)
            st = eng.stats[-1]
        finally:
            eng.backend.close()
        tr = st["trace"]
        p = parse(records_to_trace(tr["records"], progs), cfg, progs, "card")
        _dump(p, progs, st["cycles"], a.out,
              {"trace": {k: tr[k] for k in ("count", "drop", "depth", "lost")}})
        return 0
    from opentpu.isasim import board_config
    wkw = dict(wformat=a.wformat, head_format=a.head_format, rows=Q.PREFILL_ROWS)
    need = spec.image(board_config(DRAM_BYTES=1 << 40), cap, **wkw).nbytes
    cfg = board_config(DRAM_BYTES=1 << max(20, (need - 1).bit_length()))
    img = spec.image(cfg, cap, **wkw)
    dram = img.build(W)[0]
    x = np.asarray(W["model.embed_tokens.weight"][791], np.float32)
    c, s = Q.rope_tables(spec, a.pos)
    for key, v in (("x", x), ("cos", c), ("sin", s)):
        b = np.ascontiguousarray(v, np.float32).view(np.uint8).ravel()
        dram[img.io[key]:img.io[key] + b.size] = b
    progs = img.compile_step(a.pos)
    mts = 3200 / 3
    _, _, st = rtlsim.run(cfg, progs, [dram], trace=True, uarch={**rtlsim.BOARD_UARCH, "AXI_BL": 8},
                          axi=True, boot=True, stall=0, bw=100, lat=round(0.3 * a.mhz), arc=4,
                          max_cycles=1 << 40, plusargs=ddr3_plusargs(mts, a.mhz))
    p = parse(st["trace"], cfg, progs, "sim")
    _dump(p, progs, st["cycles"], a.out, {})
    return 0


if __name__ == "__main__":
    sys.exit(main())
