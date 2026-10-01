"""Co-simulated card time of a MoE model's prefill parts at R rows a program (docs/offload.md:
multi-row MoE prefill). RTL with the card's LiteDRAM controllers (rtlsim LDC, DDR3-1066, the core
at 133.33 MHz), the board's configuration (MCOLS 4, PAIR, DSTEP, STREAM). Timing does not depend
on the data, so the images are zeros: no checkpoint is read (config.json only).

    python3 tools/moe_prefill_cosim.py rows MODEL_DIR --variant dense|moe --R 1,2,4 --prefix 0:0,1:0,5
    python3 tools/moe_prefill_cosim.py expert MODEL_DIR --M 1,2,3,4,8

rows: the model cut to each prefix of layers (Gemma 4: checkpoint layers, ':' between
prefixes; Qwen3.5: a count), R rows at positions pos.. (no logits); `dense`: the MoE block
removed (the attention / mixer and the dense MLP or shared expert alone), `moe`: one row with
the MoE block, every expert present (each entry names its layer's first slot) and `served`
past every request (the timing hack: no fence or slot waits). A layer's cost is the difference
of two prefixes. expert: one expert's FFN (kernels.mlp.swiglu_down from its slot) on M rows,
less the same program without it. One JSON line per run.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu import language as ol  # noqa: E402
from opentpu import rtlsim  # noqa: E402
from opentpu.compiler import Affine, Tensor  # noqa: E402
from opentpu.isasim import board_config  # noqa: E402
from opentpu.kernels.lib import gelu_tanh  # noqa: E402
from opentpu.kernels.mlp import swiglu_down  # noqa: E402
from opentpu.llm import moe as MO  # noqa: E402
from opentpu.profile import ddr3_plusargs  # noqa: E402

MHZ = 133.33
DRAM = 1 << 30                  # one simulator build for every run


def cfg_of():
    return board_config(PAIR=True, DSTEP=True, STREAM=True, DRAM_BYTES=DRAM)


def cosim(cfg, progs, dram, profile=False) -> int:
    plus = rtlsim.ldc_plusargs(1066, MHZ) + ddr3_plusargs(3200 / 3, MHZ)[2:]
    _, _, st = rtlsim.run(cfg, progs, [dram], uarch=rtlsim.BOARD_UARCH, axi=True, boot=True,
                          stall=0, bw=100, lat=round(0.3 * MHZ), max_cycles=1 << 40, ldc=1066,
                          trace=profile, plusargs=["+axi_dram=1", "+axi_map=1"] + plus)
    if profile:
        print_profile(cfg, progs, st)
    return st["cycles"]


def _where(ins):
    """The instruction's place: its line in moe.py (the MoE block), else its function in the
    model file, else its innermost function."""
    for f, line, fn in ins.src:
        if Path(f).name == "moe.py":
            return f"moe.py:{line} {fn}"
    for f, line, fn in ins.src:
        if Path(f).name in ("gemma4.py", "qwen35.py", "qwen3.py"):
            return f"{Path(f).name} {fn}"
    return ins.src[0][2] if ins.src else "?"


def print_profile(cfg, progs, st):
    from opentpu.profile import parse
    p = parse(st["trace"], cfg, progs, "cosim")
    p.cycles = st["cycles"]
    ph = p.phases(lambda r: _where(progs[0][r.pc]))
    print(f"profile: {p.cycles} cycles", file=sys.stderr)
    for k, v in sorted(ph.items(), key=lambda x: -x[1]["cycles"])[:40]:
        print(f"  {v['cycles']:9d} {100 * v['cycles'] / p.cycles:5.1f}%  chunks {v['portb']:8d}"
              f"  VPU {v['VPU']:8d} MXU {v['MXU']:8d}  {k}", file=sys.stderr)


def load_model(d):
    mt = json.loads((Path(d) / "config.json").read_text()).get("model_type")
    if mt == "gemma4":
        from opentpu.llm import gemma4 as M
        return "gemma4", M, M.Spec.from_hf(d)
    if mt == "lfm2_moe":                    # (expert only)
        from opentpu.llm import lfm2 as M
        return "lfm2", M, M.Spec.from_hf(d)
    from opentpu.llm import qwen35 as M
    return "qwen35", M, M.Spec.from_hf(d)


def model_image(kind, M, spec, prefix, variant, R, cfg, cap):
    if kind == "gemma4":
        s = spec.truncated([int(x) for x in prefix.split(",")])
        if variant == "dense":
            s = dataclasses.replace(s, experts=0, top_k=0, expert_ffn=0)
            return s, s.image(cfg, cap, rows=R, wformat="int8", head_format="fp4", formats="")
        return s, s.image(cfg, cap, rows=R, wformat="int8", head_format="fp4",
                          formats="experts=fp4", experts=s.top_k)
    n = int(prefix)
    s = dataclasses.replace(spec, kinds=spec.kinds[:n])
    if variant == "dense":
        s = dataclasses.replace(s, moe=None)
        return s, s.image(cfg, cap, rows=R, wformat="fp4", head_format="int8")
    return s, s.image(cfg, cap, rows=R, wformat="fp4", head_format="int8", experts=s.moe.k)


def timing_hack(img, dram):
    """served past every request; every entry present at its layer's first slot."""
    Lo = img.offload
    dram[Lo.served:Lo.served + 4] = np.frombuffer(np.float32(1e6).tobytes(), np.uint8)
    ent = np.zeros((Lo.layers * Lo.E, 2), np.uint32)
    for j, (a, _) in enumerate(Lo.slots):
        ent[j * Lo.E:(j + 1) * Lo.E, 0] = a
    ent[:, 1] = np.frombuffer(np.float32(1.0).tobytes(), np.uint32)[0]
    b = ent.view(np.uint8).reshape(-1)
    dram[Lo.dir:Lo.dir + b.size] = b


def cmd_rows(a):
    kind, M, spec = load_model(a.model)
    cfg = cfg_of()
    for prefix in a.prefix.split(":"):
        for R in [int(x) for x in a.R.split(",")]:
            if a.variant == "moe" and R != 1:
                continue
            rec = dict(model=Path(a.model).name, variant=a.variant, prefix=prefix, R=R,
                       pos=a.pos)
            try:
                s, img = model_image(kind, M, spec, prefix, a.variant, R, cfg, a.cap)
                rows = [(0, a.pos + r) for r in range(R)]
                progs = img.compile_rows(rows, [], img.block if hasattr(img, "block")
                                         else 256) if kind == "gemma4" or a.variant == "dense" \
                    else img.compile_step(a.pos)
            except Exception as e:          # TMEM / ACT RAM / IMEM: the fit answer
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
                print(json.dumps(rec), flush=True)
                continue
            rec["instructions"] = max(len(p) for p in progs)
            rec["image_mib"] = round(img.nbytes / 2**20)
            if a.compile_only:
                print(json.dumps(rec), flush=True)
                continue
            dram = np.zeros(img.nbytes, np.uint8)
            if a.variant == "moe":
                timing_hack(img, dram)
            t = time.time()
            rec["cycles"] = cosim(cfg, progs, dram, a.profile)
            rec["ms"] = round(rec["cycles"] / (MHZ * 1e3), 3)
            rec["sim_s"] = round(time.time() - t)
            print(json.dumps(rec), flush=True)


@ol.jit
def _expert_rows(m, M, with_expert=True):
    x = ol.load(m.x[0:M, :])
    xe = ol.quantize(x)
    if with_expert:
        ex = m.fmt.descs(Affine(m.slot))
        o = swiglu_down(xe, ex.wg, ex.wu, ex.wd, act=gelu_tanh)
        ol.store(m.y[0:M, :], o)
    else:
        ol.store(m.y[0:M, :], x)


def cmd_expert(a):
    kind, Mo, spec = load_model(a.model)
    mo = spec.moe
    cfg = cfg_of()
    H = spec.hidden
    fmt = MO.ExpertFormat(H, mo.ffn, cfg.D, "fp4")
    for M in [int(x) for x in a.M.split(",")]:
        nb = 2 * 4 * M * H
        slot = -(-nb // 4096) * 4096
        m = SimpleNamespace(x=Tensor(Affine(0), (M, H), (H, 1)),
                            y=Tensor(Affine(4 * M * H), (M, H), (H, 1)), fmt=fmt, slot=slot)
        dram = np.zeros(slot + fmt.nbytes, np.uint8)
        out = dict(model=Path(a.model).name, M=M, slot_bytes=fmt.nbytes)
        cyc = []
        for we in (True, False):
            progs = [_expert_rows.trace(cfg, 0, {"m": m, "M": M, "with_expert": we}).finish()]
            cyc.append(cosim(cfg, progs, dram))
        out.update(cycles=cyc[0], base=cyc[1], expert_cycles=cyc[0] - cyc[1],
                   expert_ms=round((cyc[0] - cyc[1]) / (MHZ * 1e3), 4),
                   dram_bound_ms=round(fmt.nbytes / 14.1e9 * 1e3, 4))
        print(json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows")
    r.add_argument("model")
    r.add_argument("--variant", choices=("dense", "moe"), default="dense")
    r.add_argument("--R", default="1,2,4")
    r.add_argument("--prefix", default="0:0,1:0,5")
    r.add_argument("--pos", type=int, default=256)
    r.add_argument("--cap", type=int, default=1024)
    r.add_argument("--compile-only", action="store_true")
    r.add_argument("--profile", action="store_true", help="per place: cycles (stderr)")
    e = sub.add_parser("expert")
    e.add_argument("model")
    e.add_argument("--M", default="1,2,3,4,8")
    a = ap.parse_args()
    {"rows": cmd_rows, "expert": cmd_expert}[a.cmd](a)


if __name__ == "__main__":
    main()
