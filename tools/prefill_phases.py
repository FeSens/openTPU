"""A prefill run's cycles, DRAM bytes and unit busy time by phase (tools/board_perf.py prefill;
docs/tiled_prefill.md): each traced instruction's phase from its source frames (projections,
attention, MLP, DeltaNet, inputs, other), its DRAM bytes from its operands (profile._describe's
port-B chunks and port-A scale words), split by what they are: weights (MMs' streamed rows and
their scales), KV (the attention's K / V^T streams and its appends), the DeltaNet states
(DSTEP's read and write), activations (every other LD / ST / QST).

    summarize(profile, programs, cfg) -> dict      (from a parsed trace: opentpu.profile.parse)
"""
from __future__ import annotations

from opentpu import isa as I
from opentpu import quant as Q

# source frames (innermost few: Instr.src) of each phase, tried in this order
CORE = {"scores", "finish", "prime", "_scores", "_finish", "_attend_heads", "_attend_tile",
        "_tile_scores", "_tile_finish", "_heads_loop", "_attention_loop", "_tile_blocks"}
MLP = {"_mlp_tile", "_mlp", "swiglu_down", "_down_split", "gate_up", "_sum_parts",
       "_add_parts", "moe_ffn_rows", "_expert_rows", "_experts"}
PROJ = {"_attention_tile", "_attention_rows", "_norm_heads", "_rope_rows_padded", "_rope_gate",
        "_gemma_attention_tile"}
DN = {"_deltanet_rows", "_deltanet_dstep", "_rows_pipelined", "_dn_norm", "_silu_dn",
      "key_heads", "dsteps", "store_window", "head_in", "_out_proj", "mm_x", "mm_z", "gates",
      "_l2n", "l2norm_rows", "softplus", "segment", "prep", "post", "fetch", "unrolled"}
INPUTS = {"_inputs_rows", "_embed_word", "_gather", "gather_row"}
PHASES = (("attn-core", CORE), ("mlp", MLP), ("attn-proj", PROJ), ("deltanet", DN),
          ("inputs", INPUTS))
KINDS = ("weights", "kv", "dn_state", "act")


def kv_mm(ins) -> bool:
    """An MM that streams a KV cache: the K cache's scores (RSC: Sw alone a row) or V^T's
    finish (UNIT, ASCALE); the projections' weights take neither."""
    return ins.op == I.MM and bool(ins.flags & (I.F_RSC | I.F_ASCALE | I.F_UNIT))


def phase(ins) -> str:
    """The instruction's phase: the first of PHASES one of its source frames names; the
    projections' MMs that the attention's pipeline emits (its frames) are attn-proj's."""
    fns = {f for _, _, f in ins.src}
    for name, fs in PHASES:
        if fns & fs:
            if name == "attn-core" and ins.op == I.MM and not kv_mm(ins):
                return "attn-proj"
            return name
    return "other"


def nbytes(ins, cfg) -> dict:
    """The DRAM bytes an instruction moves, by kind (one execution)."""
    op, w, D = ins.op, ins.w, cfg.D
    if op == I.MM:
        N, KB = w[3] & 0xFFFF, (w[3] >> 16) & 0xFF
        wf = (ins.flags >> I.WF_SHIFT) & 3
        wb = N * KB * (D if wf == I.W8 else D // 2)
        if ins.flags & I.F_UNIT or ins.flags & I.F_ASCALE:
            sb = 0
        elif ins.flags & I.F_RSC:
            sb = 4 * N
        else:
            sb = N * (4 + KB * Q.SCALE_DATA["int8" if wf == I.W8 else "fp4"])
        return {"mm": wb + sb}
    if op in (I.LD, I.ST):
        return {"ldst": 4 * w[2]}
    if op == I.DSTEP:
        rows, cols = w[3] & 0xFFFF, w[3] >> 16
        n = 4 * rows * cols
        return {"dstep": n if ins.flags & I.F_DZERO else 2 * n}
    if op == I.STREAM:
        import re
        m = re.search(r"stream (\d+)x(\d+)", ins.comment)
        n = 4 * int(m.group(1)) * int(m.group(2)) if m else 0
        return {"dstep": n if ins.flags & I.F_SZERO else 2 * n}
    if op == I.QST:
        rows, KB = w[3] & 0xFFFF, w[3] >> 16
        n = rows * KB * D
        nw = n // 2 if ins.flags & I.F_HALF else n
        return {"qst": nw + 4 * rows * KB}
    return {}


def kind_of(ph: str, ins, b: dict) -> dict:
    """The bytes b of an instruction of phase ph, by KINDS."""
    out = {}
    for k, v in b.items():
        if k == "mm":
            out["kv" if kv_mm(ins) and ph.startswith("attn") else "weights"] = v
        elif k == "dstep":
            out["dn_state"] = v
        elif k == "qst":
            out["kv" if ph in ("attn-core", "attn-proj") else "act"] = v
        else:
            out["act"] = v
    return out


def _union(iv) -> int:
    tot, cs, ce = 0, None, None
    for a, b in sorted(iv):
        if ce is None or a > ce:
            tot += 0 if ce is None else ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    return tot + (0 if ce is None else ce - cs)


def summarize(p, programs, cfg, s: int = 0) -> dict:
    """Per phase: cycles (profile.phases' charge: from the latest end before a run of the
    phase's instructions to the latest end within it, so they add up to the run's), the
    instructions traced, DRAM bytes by kind, and the DMA / MXU / VPU busy cycles (the union of
    the phase's own instructions' intervals; DSTEP and STREAM run on the DMA unit). Then the
    totals over the run."""
    prog = programs[s]
    out: dict = {}
    done = 0
    for r in sorted(p.slice_recs(s), key=lambda r: r.idx):
        ins = prog[r.pc]
        ph = phase(ins)
        a = out.setdefault(ph, {"cycles": 0, "n": 0, **{k: 0 for k in KINDS},
                                "iv": {0: [], 1: [], 3: []}, "dstep_iv": []})
        a["n"] += 1
        for k, v in kind_of(ph, ins, nbytes(ins, cfg)).items():
            a[k] += v
        if r.start >= 0 and r.end >= 0:
            if r.unit in a["iv"]:
                a["iv"][r.unit].append((r.start, r.end))
            if r.op in (I.DSTEP, I.STREAM):
                a["dstep_iv"].append((r.start, r.end))
        if r.end > done:
            a["cycles"] += r.end - done
            done = r.end
    tot = {"cycles": 0, "n": 0, **{k: 0 for k in KINDS}}
    alliv = {0: [], 1: [], 3: [], "d": []}
    for a in out.values():
        for u, iv in a.pop("iv").items():
            a[{0: "DMA", 1: "MXU", 3: "VPU"}[u]] = _union(iv)
            alliv[u] += iv
        alliv["d"] += a["dstep_iv"]
        a["DSTEP"] = _union(a.pop("dstep_iv"))
        for k in tot:
            tot[k] += a[k]
    tot |= {"DMA": _union(alliv[0]), "MXU": _union(alliv[1]), "VPU": _union(alliv[3]),
            "DSTEP": _union(alliv["d"])}
    return {"phases": out, "total": tot, "traced": len(p.slice_recs(s))}
