"""Timing-only program transforms that bound what a faster vector unit (VPU) or stream engine (SE)
would save (docs/vpu_pump.md). Each replaces or shrinks instructions in place, so the program's
length, its loops and every other unit's work stay as compiled; only the cycles of the co-simulated
run mean anything afterwards (the results are wrong, as with tools/decode_gaps.py --drop).

    modes (comma list; apply(prog, "half,halfse")):
      novop     every VOP a NOP: an infinitely fast VPU (the VPU's whole share)
      nocomp    the composite VOPs (EXP2, EXP2SUB, RECIP, RSQRT, LOG2) NOPs
      comp1     each composite VOP as a one-slot elementwise op of its shape (EXP2SUB -> SUB with
                its B; the others -> ADD of a scalar): composites at the elementwise rate (8 words
                a cycle) and latency (one slot, 7 cycles) -- the bound of any composite speed-up
      half      every VOP's element work halved: a VPU of twice the throughput (16 lanes, or 8
                lanes pumped at 2x) with today's latencies (see _half)
      halfcomp  the composite VOPs' work halved: a composite path of twice the throughput
      halfew    the non-composite VOPs' work halved (half = halfew + halfcomp)
      nose      every DSTEP / STREAM a NOP: an infinitely fast stream engine, and no state traffic
      halfse    each DSTEP's rows halved: an SE of twice the datapath rate; also halves its state
                traffic (DRAM), which a faster SE would not, so it overstates (docs/vpu_pump.md)
      qse       each DSTEP's rows quartered: an SE of four times the datapath rate (as halfse)
      nostate   each DSTEP reads and writes no state (ZERO, and flag bit 7: otpu_dma's
                simulation-only DF_SIM_NOWR, needs +ds_sim=1): the datapath's time with its state
                traffic free (docs/se_engine.md)
      qmin      each QACT one D-block (its rows' first block): the quantizer free, its dependencies
                kept (it still reads its input's first block and writes its ACT range's first one)
                -- the bound of any QACT speed-up (docs/weight_groups.md)
      qhalf     each QACT's blocks halved (ceil): a quantizer of twice the rate, or of one pass over
                the row (its max known before it: pass 0 done by the producer)
      nossq     each row sum of squares over >= 1,024 columns (an RMSNorm's statistic: hidden-wide
                rows) one chunk of 8: the bound of computing it in QACT's pass 0 beside the max
                (docs/dataflow.md 4.3, NORM; the scalar 1 / rms chain after it is kept)
      nogather  each GATHER a BAR: the collective's moves free, the slices' synchronization kept
                (the bound of any GATHER speed-up)
      halfmm    each MM of M > 1 rows takes ceil(M / 2) (its replay halved): an MXU of two
                activation rows a step (MCOLS 16 at WG 8, or its DSPs pumped at 2x), its drain and
                epilogue at today's rate (the bound of any rows-a-step widening; M = 1 MMs as built)
      w2mm      each grouped int8 MM of M > 1 rows and N > 8 a 4-bit one (WF int4, groups of 16
                rows): two int8 groups a step (two weight rows a DSP on A and D, the 4-bit path's
                second column end), its results, epilogue and drain as built (TMEM's writes
                kept); it reads half the chunks from DRAM, so it understates where DRAM binds
      exp2l     each EXP2SUB a RECIP of its shape: an exp2 of two passes (a degree-4 polynomial,
                6 ops: 4 columns a cycle, latency 2 T) instead of three
      fsum      each RSUM of attention.py's a NOP: the row sums of the softmax taken with its
                EXP2SUB (a fused sum on the composite's output, no VPU slot)
      st16      each DSTEP with flag ST16 (isa.F_DST16): its DRAM state in 16 bits, half the
                chunks each way (needs the board's ST16, Config.ST16; the 16-bit state study's
                timing on programs compiled for the fp32 state: each head's state the first half
                of its area. The images of a Config.ST16 target do it for real: Spec.state16)

`keep(ins)` (optional) protects instructions that must stay exact because a later RLD turns their
values into addresses or loop counts (the MoE routing's bookkeeping: tools/moe_prefill_cosim.py).
"""
from __future__ import annotations

import dataclasses

from opentpu import isa as I

COMPOSITE = {I.V_EXP2, I.V_EXP2SUB, I.V_RECIP, I.V_RSQRT, I.V_LOG2, I.V_SIGMOID, I.V_SILU}
SUMS = {I.V_RSUM, I.V_RSSQ, I.V_RDOT}
MODES = ("novop", "nocomp", "comp1", "half", "halfcomp", "halfew", "nose", "halfse", "qse",
         "nostate", "qmin", "qhalf", "nossq", "nogather", "st16", "halfmm", "w2mm", "exp2l",
         "fsum")
F_SIM_NOWR = 0x80                   # otpu_dma (simulation, +ds_sim=1): the DSTEP writes nothing


def vfunc(ins) -> int:
    return (ins.w[5] >> 16) & 0xFF


def _shape(ins) -> tuple[int, int]:
    return ins.w[3] & 0xFFFF, ins.w[3] >> 16


def _with(ins, **w) -> I.Instr:
    words = list(ins.w)
    for k, v in w.items():
        words[int(k[1:]) - 1] = I.u32(v)
    return dataclasses.replace(ins, w=words, comment=f"hacked {ins.comment}")


def _half(ins) -> I.Instr:
    """The VOP with half its element work, as a VPU of twice the lanes would see it: the columns
    halved (ceil), so ceil(cols / 2) / 8 chunks = cols / 16 a row; a row sum of at most 64
    columns (padded to 64: 8 chunks, 4 at 16 lanes) takes half its rows instead (a 1-row one
    keeps its 8 chunks, 4 cycles more than 16 lanes would take)."""
    rows, cols = _shape(ins)
    if vfunc(ins) in SUMS and cols <= 64:
        rows = -(-rows // 2)
    else:
        cols = -(-cols // 2)
    return _with(ins, w4=rows | (cols << 16))


def _in_attention(ins) -> bool:
    return any("attention.py" in str(f) for f in (ins.src or ()))


def _w2(ins) -> bool:
    """A grouped int8 MM (LG >= 1, WF 0, no KT) of M > 1 rows and more than one group."""
    m, nrows = (ins.w[5] >> 16) & 0xFF, ins.w[3] & 0xFFFF
    return (ins.rd >= 1 and (ins.flags >> I.WF_SHIFT) & 3 == I.W8 and not ins.w[4] & 1
            and m > 1 and nrows > 1 << ins.rd)


def _comp1(ins) -> I.Instr:
    f = vfunc(ins)
    if f == I.V_EXP2SUB:                # A - B, B in its own mode
        return _with(ins, w6=(ins.w[5] & 0xFF00FFFF) | (I.V_SUB << 16))
    # A only: x + 0.0 (B a scalar immediate; the register part of imm is not read: rd = 0)
    w6 = (ins.w[5] & 0x0000FFFF) | (I.V_ADD << 16) | (I.B_SCALAR << 24)
    return dataclasses.replace(_with(ins, w6=w6, w7=0), rd=0)


def apply(prog, modes: str, keep=None):
    """prog with the modes' transforms (module docstring); returns (new program, counts)."""
    ms = [m for m in (modes or "").split(",") if m]
    for m in ms:
        if m not in MODES:
            raise ValueError(f"vpu_hack: unknown mode {m!r} (modes: {', '.join(MODES)})")
    if "half" in ms:
        ms = [m for m in ms if m != "half"] + ["halfew", "halfcomp"]
    n = dict.fromkeys(ms, 0)
    out = []
    for ins in prog:
        new = ins
        if keep is not None and keep(ins):
            out.append(ins)
            continue
        if ins.op == I.VOP:
            comp = vfunc(ins) in COMPOSITE
            if "nossq" in ms and vfunc(ins) == I.V_RSSQ and _shape(ins)[1] >= 1024:
                new, n["nossq"] = _with(ins, w4=_shape(ins)[0] | (8 << 16)), n["nossq"] + 1
            elif "novop" in ms:
                new, n["novop"] = I.nop(f"dropped {ins.comment}"), n["novop"] + 1
            elif comp and "nocomp" in ms:
                new, n["nocomp"] = I.nop(f"dropped {ins.comment}"), n["nocomp"] + 1
            elif "fsum" in ms and vfunc(ins) == I.V_RSUM and _in_attention(ins):
                new, n["fsum"] = I.nop(f"dropped {ins.comment}"), n["fsum"] + 1
            else:
                if vfunc(ins) == I.V_EXP2SUB and "exp2l" in ms:
                    new = _with(ins, w6=(ins.w[5] & 0xFF00FFFF) | (I.V_RECIP << 16))
                    n["exp2l"] += 1
                if comp and "comp1" in ms:
                    new, n["comp1"] = _comp1(new), n["comp1"] + 1
                if comp and "halfcomp" in ms:
                    new, n["halfcomp"] = _half(new), n["halfcomp"] + 1
                elif not comp and "halfew" in ms:
                    new, n["halfew"] = _half(new), n["halfew"] + 1
        elif ins.op in (I.DSTEP, I.STREAM):
            if "nose" in ms:
                new, n["nose"] = I.nop(f"dropped {ins.comment}"), n["nose"] + 1
            else:
                for m, k in (("halfse", 2), ("qse", 4)):
                    if m in ms and ins.op == I.DSTEP:
                        rows, cols = _shape(new)
                        new, n[m] = _with(new, w4=-(-rows // k) | (cols << 16)), n[m] + 1
                if "st16" in ms and ins.op == I.DSTEP:
                    new = dataclasses.replace(new, flags=new.flags | I.F_DST16,
                                              comment=f"hacked {ins.comment}")
                    n["st16"] += 1
                if "nostate" in ms and ins.op == I.DSTEP:
                    new = dataclasses.replace(new, flags=new.flags | I.F_DZERO | F_SIM_NOWR,
                                              comment=f"hacked {ins.comment}")
                    n["nostate"] += 1
        elif ins.op == I.MM and "halfmm" in ms and (ins.w[5] >> 16) & 0xFF > 1:
            m = (ins.w[5] >> 16) & 0xFF
            new = _with(ins, w6=(ins.w[5] & 0xFF00FFFF) | (-(-m // 2) << 16))
            n["halfmm"] += 1
        elif ins.op == I.MM and "w2mm" in ms and _w2(ins):
            fl = (ins.flags & ~(3 << I.WF_SHIFT)) | (I.W4I << I.WF_SHIFT)
            new = dataclasses.replace(ins, rd=ins.rd + 1, flags=fl, comment=f"hacked {ins.comment}")
            n["w2mm"] += 1
        elif ins.op == I.GATHER and "nogather" in ms:
            new, n["nogather"] = I.bar(), n["nogather"] + 1
        elif ins.op == I.QACT and ("qmin" in ms or "qhalf" in ms):
            kb = ins.w[1] >> 16
            m = "qmin" if "qmin" in ms else "qhalf"
            kb = 1 if m == "qmin" else -(-kb // 2)
            new, n[m] = _with(ins, w2=(ins.w[1] & 0xFFFF) | (kb << 16)), n[m] + 1
        out.append(new)
    return out, n


PASSES = {I.V_EXP2: 3, I.V_EXP2SUB: 3, I.V_RECIP: 2, I.V_RSQRT: 4, I.V_LOG2: 4,
          I.V_SIGMOID: 6, I.V_SILU: 6}      # (VSIG: six passes at CNS 3, four at 5)


def issue_cycles(ins, lanes: int = 8) -> int:
    """The VPU's (or the stream engine's) issue cycles for an instruction as built (docs/stream.md
    11): a chunk of `lanes` columns a cycle; a composite holds slot 0 for its passes through the
    three stages (EXP2 3, RECIP 2, RSQRT / LOG2 4, SIGMOID / SILU 6: 2.67, 4, 2 and 1.33 columns a
    cycle); a row sum's row is padded to 64 columns (whole partial-loop rounds); OUTER first fills
    its column buffers (at least 2 chunks); DSTEP / STREAM 8 state words a cycle. Latency not
    counted."""
    if ins.op in (I.DSTEP, I.STREAM):
        if ins.op == I.DSTEP:
            rows, cols = _shape(ins)
        else:
            import re
            m = re.search(r"stream (\d+)x(\d+)", ins.comment or "")
            rows, cols = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        return -(-rows * cols // 8)
    if ins.op != I.VOP:
        return 0
    rows, cols = _shape(ins)
    f = vfunc(ins)
    ch = -(-cols // lanes)
    if f in SUMS:
        rl = 64 // lanes
        return rows * -(-ch // rl) * rl
    if f == I.V_OUTER or (f == I.V_MAC and (ins.w[5] >> 24) & 3 == I.B_COL):   # (+ the fill)
        return max(2, ch) + rows * ch
    return rows * ch * PASSES.get(f, 1)


def multiplicity(prog) -> list[int]:
    """Each instruction's executions in one run: the product of its enclosing LOOP counts (a
    count from a register, R[ra] + w2 with ra != 0, is taken as w2: compile-time programs have
    none)."""
    out, stack = [], []           # stack: (last pc of the body, count)
    for pc, ins in enumerate(prog):
        while stack and pc > stack[-1][0]:
            stack.pop()
        m = 1
        for _, c in stack:
            m *= c
        out.append(m)
        if ins.op == I.LOOP:
            stack.append((pc + ins.w[0], ins.w[1]))
    return out


def inventory(prog, cfg) -> dict:
    """The run's static work: per VOP function (and DSTEP / STREAM) its instructions and issue
    cycles (issue_cycles), and the port-B chunks of everything (opentpu.profile's model): the
    vector work against the memory stream, before any overlap."""
    from opentpu import profile as PR
    mult = multiplicity(prog)
    fn, portb = {}, 0
    for ins, m in zip(prog, mult):
        if ins.op == I.LOOP:
            continue
        portb += m * PR._describe(ins, cfg)[3]
        if ins.op in (I.VOP, I.DSTEP, I.STREAM):
            k = I.VFUNCS.get(vfunc(ins), "?") if ins.op == I.VOP else PR.OPNAMES[ins.op]
            a = fn.setdefault(k, [0, 0])
            a[0] += m
            a[1] += m * issue_cycles(ins, cfg.LANES)
    comp = ("exp2", "exp2sub", "recip", "rsqrt", "log2")
    return {"portb": portb, "by_func": fn,
            "vop_issue": sum(v[1] for k, v in fn.items() if k not in ("DSTEP", "STREAM")),
            "comp_issue": sum(v[1] for k, v in fn.items() if k in comp),
            "se_issue": sum(v[1] for k, v in fn.items() if k in ("DSTEP", "STREAM"))}


def _prefix(spec, layers: str | None):
    """The model cut to its first N layers, or Gemma 4's comma list of checkpoint layers."""
    if not layers:
        return spec
    ls = [int(x) for x in layers.split(",")]
    if hasattr(spec, "truncated"):
        return spec.truncated(ls if len(ls) > 1 else range(ls[0]))
    if hasattr(spec, "kinds"):
        return dataclasses.replace(spec, kinds=spec.kinds[:ls[0]])
    return dataclasses.replace(spec, layers=ls[0])


def compile_case(model, kind: str, n: int, pos: int, mcols: int | None = None,
                 wformat: str = "fp4", head: str = "int8", cap: int | None = None,
                 layers: str | None = None, dram_bytes: int = 1 << 40, full: bool = False,
                 sets: dict | None = None):
    """(cfg, program) of a case, compiled only (no weights): kind "decode" (compile_step at pos),
    "rows" (n prompt rows at pos.., no logits) or "batch" (n slots at pos, Qwen3 family:
    opentpu/llm/batch.py's step with the picks); `layers` cuts the model (_prefix). full: (cfg,
    programs, image, the batch area or None, spec). sets: Config fields over the board's (another
    target's arch, e.g. S 4)."""
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec, model_dir
    spec = _prefix(load_spec(model_dir(model)), layers)
    R = n if kind in ("rows", "batch") else 1
    cap = cap or 256 * ((pos + R - 1) // 256 + 1)
    mk = {"MCOLS": mcols} if mcols else {}
    cfg = board_config(DRAM_BYTES=dram_bytes, **{**(sets or {}), **mk})
    kw = dict(wformat=wformat, head_format=head, rows=R)
    if kind == "batch":
        kw.update(batch=n, lookup=True)
    img = spec.image(cfg, cap, **kw)
    area = None
    if kind == "decode":
        progs = img.compile_step(pos)
    elif kind == "rows":
        progs = img.compile_rows([(0, pos + r) for r in range(n)], [])
    else:
        from opentpu.llm import batch as BA
        area = BA.alloc(img, cfg, n, -(-img.nbytes // 4096) * 4096)
        progs = BA.compile_step(img, area, [(s, pos) for s in range(n)], [None] * n, 256)
    return (cfg, progs, img, area, spec) if full else (cfg, progs[0])


MHZ = 133.33
COSIM_DRAM = 1 << 31            # one simulator build per MCOLS for every zero-image run


def cosim_case(model, kind, n, pos, mcols=None, layers=None, hack="", wformat="fp4",
               head="int8") -> dict:
    """A case co-simulated on a zero DRAM image (the timing does not depend on the data; only
    config.json is read): the RTL at the board's micro-architecture with LiteDRAM's controller
    at DDR3-1066 and the core at 133.33 MHz, as tools/moe_prefill_cosim.py runs it; `hack`
    applies this module's modes first. Returns the run's record (cycles)."""
    import time

    import numpy as np

    from opentpu import rtlsim
    from opentpu.profile import ddr3_plusargs
    cfg, progs, img, area, spec = compile_case(model, kind, n, pos, mcols, wformat, head,
                                         layers=layers, dram_bytes=COSIM_DRAM, full=True)
    rec = dict(model=str(model).rstrip("/").split("/")[-1], kind=kind, n=n, pos=pos,
               mcols=cfg.MCOLS, layers=layers, hack=hack, wformat=wformat, head=head,
               instructions=max(len(p) for p in progs), image_mib=round(img.nbytes / 2**20))
    progs = [apply(p, hack)[0] for p in progs]
    end = area["end"] if area else img.nbytes
    if end > COSIM_DRAM:
        rec["error"] = f"image {end} B over the simulated DRAM"
        return rec
    dram = np.zeros(end, np.uint8)
    if area:
        from opentpu.llm import batch as BA
        BA.build([dram], area, spec, cfg)
    plus = rtlsim.ldc_plusargs(1066, MHZ) + ddr3_plusargs(3200 / 3, MHZ)[2:]
    t = time.time()
    _, _, st = rtlsim.run(cfg, progs, [dram], uarch=rtlsim.BOARD_UARCH, axi=True, boot=True,
                          stall=0, bw=100, lat=round(0.3 * MHZ), max_cycles=1 << 40, ldc=1066,
                          plusargs=["+axi_dram=1", "+axi_map=1"] + plus)
    rec.update(cycles=int(st["cycles"]), sim_s=round(time.time() - t))
    return rec


def main(argv=None) -> int:
    """python3 tools/vpu_hack.py MODEL KIND N POS [MCOLS]: the static inventory of a case (one
    JSON line): the VPU's and the stream engine's issue cycles against the port-B chunks.
    python3 tools/vpu_hack.py cosim MODEL KIND N POS [--mcols M] [--layers L] [--hack MODES]
    [--wformat F] [--head-format H]: the case co-simulated on a zero image (cosim_case), one JSON
    line."""
    import argparse
    import json
    import sys
    a = (argv if argv is not None else sys.argv[1:])
    if a and a[0] == "cosim":
        ap = argparse.ArgumentParser(prog="vpu_hack.py cosim")
        for k in ("model", "kind"):
            ap.add_argument(k)
        ap.add_argument("n", type=int)
        ap.add_argument("pos", type=int)
        ap.add_argument("--mcols", type=int)
        ap.add_argument("--layers")
        ap.add_argument("--hack", default="")
        ap.add_argument("--wformat", default="fp4")
        ap.add_argument("--head-format", default="int8")
        o = ap.parse_args(a[1:])
        print(json.dumps(cosim_case(o.model, o.kind, o.n, o.pos, o.mcols, o.layers, o.hack,
                                    o.wformat, o.head_format)), flush=True)
        return 0
    model, kind, n, pos = a[0], a[1], int(a[2]), int(a[3])
    mcols = int(a[4]) if len(a) > 4 else None
    cfg, prog = compile_case(model, kind, n, pos, mcols)
    inv = inventory(prog, cfg)
    print(json.dumps({"model": model.rstrip("/").split("/")[-1], "kind": kind, "n": n,
                      "pos": pos, "mcols": cfg.MCOLS, "instructions": len(prog), **inv}))
    return 0



def moe_keep(ins) -> bool:
    """The MoE routing's bookkeeping VOPs (moe.py's own, on fewer than 1,024 words: the router's
    scores and the top-k, the experts' ids, offsets, rows and flags, the rotations), whose values
    later become RLD addresses and loop counts: kept exact so a hacked run still reads valid
    directory entries and runs k experts. The experts' FFNs and the combine stay hackable."""
    if ins.op != I.VOP:
        return False
    rows, cols = _shape(ins)
    inner = next((fn for f, _, fn in ins.src if f.endswith("moe.py")), None)
    own = ins.src and ins.src[0][0].endswith("moe.py")
    return inner is not None and own and rows * cols < 1024


if __name__ == "__main__":
    raise SystemExit(main())
