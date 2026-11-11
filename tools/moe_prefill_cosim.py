"""Co-simulated card time of a MoE model's prefill parts at R rows a program (docs/offload.md:
multi-row MoE prefill). RTL with the card's LiteDRAM controllers (rtlsim LDC, DDR3-1066, the core
at 133.33 MHz), the board's configuration (MCOLS 4, PAIR, DSTEP, STREAM). Timing does not depend
on the data, so the images are zeros: no checkpoint is read (config.json only).

    python3 tools/moe_prefill_cosim.py rows MODEL_DIR --variant dense|moe --R 1,2,4 --prefix 0:0,1:0,5
    python3 tools/moe_prefill_cosim.py expert MODEL_DIR --M 1,2,3,4,8
    python3 tools/moe_prefill_cosim.py em MODEL_DIR --rows 16,131,512 [--used N | --route IDS.npy]
        [--set MCOLS=16] [--isa] [--profile]
    python3 tools/moe_prefill_cosim.py mtp MODEL_DIR --prefix 4:8 [--pos 256]

rows: the model cut to each prefix of layers (Gemma 4: checkpoint layers, ':' between
prefixes; Qwen3.5: a count), R rows at positions pos.. (no logits); `dense`: the MoE block
removed (the attention / mixer and the dense MLP or shared expert alone), `moe`: one row with
the MoE block, every expert present (each entry names its layer's first slot) and `served`
past every request (the timing hack: no fence or slot waits). A layer's cost is the difference
of two prefixes. expert: one expert's FFN (kernels.mlp.swiglu_down from its slot) on M rows,
less the same program without it. em: expert-major's expert run of one MoE layer
(compile_expert_run, docs/offload.md 13.11) after its mixer runs over a chunk of `rows` rows,
with the server's words as the expert-major prefill has them once every expert the chunk needs
has landed: the records (each row's k global ids, weights and 1 / rms in its tail) in the
scratch on the first slots, each expert the rows chose in a slot of its own after the scratch,
with its tag set and its directory entry {slot, 1.0}. The routing: each row k distinct experts,
uniform, of `used` (the layer's distinct experts in a chunk; by default the prompts' measured 144
of the 35B's 256, 75 of the 26B's 128), or a router trace's layer-local ids (--route, .npy
[rows, k]). After the run every used expert's tag must be zero and every (row, rank) output
place written (checked; --isa: the ISA simulator, no cycles, its passes counted by their WAITWs
and checked against the bucketing's, isa_passes). Its record: passes (numpy's
bucketing), cycles, ms, ms a pass, the experts' bytes streamed (each pass streams its expert),
the run's DRAM bytes by kind and a row (em_bytes), and layer-major's passes on the same ids in
runs of 2 rows (lm2: each run's union; the synthetic routing's adjacent rows are independent,
so against a prompt's it overstates them).
mtp (Qwen3.5-MoE, docs/offload.md 15.6): the decode loop's programs on the card at a prefix of
layers, the production formats (streamed.FORMATS), every expert present and the host's
mailboxes served (the timing hack): the plain loop's one token (compile_generate: the step and
its argmax) and the MTP loop's iteration (opentpu/llm/mtp.py compile_gen: V, the verify of the
token and its draft, then D, the MTP layer over both rows and the 32K draft head, on V's DRAM),
their HALT CHAINs made HALTs. The image's head is the whole vocabulary's; a layer's cost is the
difference of two prefixes. Also the slots a layer that fill the card's DRAM (fit_experts) for
the plain image and the MTP one (its MTP layer's MoE one more layer of the slots).
--set KEY=VALUE overrides the board configuration (MCOLS=16, ...). One JSON line per run.
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

from opentpu import isa as I  # noqa: E402
from opentpu import language as ol  # noqa: E402
from opentpu import quant as Q  # noqa: E402
from opentpu import rtlsim  # noqa: E402
from opentpu.compiler import Affine, Tensor  # noqa: E402
from opentpu.isasim import board_config  # noqa: E402
from opentpu.kernels.lib import gelu_tanh  # noqa: E402
from opentpu.kernels.mlp import swiglu_down  # noqa: E402
from opentpu.llm import moe as MO  # noqa: E402
from opentpu.profile import ddr3_plusargs  # noqa: E402

MHZ = 133.33
DRAM = 1 << 30                  # one simulator build for every run


def cfg_of(**over):
    return board_config(**{**dict(PAIR=True, DSTEP=True, STREAM=True, DRAM_BYTES=DRAM), **over})


def _rtl(cfg, progs, dram, args=None, profile=False):
    """(DRAM after the run, stats): the RTL with the card's LiteDRAM controllers."""
    plus = rtlsim.ldc_plusargs(1066, MHZ) + ddr3_plusargs(3200 / 3, MHZ)[2:]
    drams, _, st = rtlsim.run(cfg, progs, [dram], uarch=rtlsim.BOARD_UARCH, axi=True,
                              boot=True, stall=0, bw=100, lat=round(0.3 * MHZ),
                              max_cycles=1 << 40, ldc=1066, args=args, trace=profile,
                              plusargs=["+axi_dram=1", "+axi_map=1"] + plus)
    if profile:
        print_profile(cfg, progs, st)
    return drams[0], st


def cosim(cfg, progs, dram, profile=False) -> int:
    return _rtl(cfg, progs, dram, profile=profile)[1]["cycles"]


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
    if mt in ("gemma4", "gemma4_text"):
        from opentpu.llm import gemma4 as M
        return "gemma4", M, M.Spec.from_hf(d)
    if mt == "lfm2_moe":                    # (expert only)
        from opentpu.llm import lfm2 as M
        return "lfm2", M, M.Spec.from_hf(d)
    from opentpu.llm import qwen35 as M
    return "qwen35", M, M.Spec.from_hf(d)


def model_image(kind, M, spec, prefix, variant, R, cfg, cap, **kw):
    """The model cut to the prefix; `moe`: its experts in fp4, kw["experts"] slots a layer
    (default top-k)."""
    if kind == "gemma4":
        s = spec.truncated([int(x) for x in prefix.split(",")])
        if variant == "dense":
            s = dataclasses.replace(s, experts=0, top_k=0, expert_ffn=0)
            return s, s.image(cfg, cap, rows=R, wformat="int8", head_format="fp4", formats="")
        return s, s.image(cfg, cap, rows=R, wformat="int8", head_format="fp4",
                          formats="experts=fp4", **{"experts": s.top_k, **kw})
    n = int(prefix)
    s = dataclasses.replace(spec, kinds=spec.kinds[:n])
    if variant == "dense":
        s = dataclasses.replace(s, moe=None)
        return s, s.image(cfg, cap, rows=R, wformat="fp4", head_format="int8")
    return s, s.image(cfg, cap, rows=R, wformat="fp4", head_format="int8",
                      **{"experts": s.moe.k, **kw})


def vpu_hack(progs, modes):
    """tools/vpu_hack.py's transforms (docs/vpu_pump.md), the routing's bookkeeping kept exact."""
    if not modes:
        return progs
    sys.path.insert(0, str(ROOT / "tools"))
    import vpu_hack as VH
    return [VH.apply(p, modes, keep=VH.moe_keep)[0] for p in progs]


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
                       pos=a.pos, hack=a.hack)
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
            progs = vpu_hack(progs, a.hack)
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


def cmd_layer(a):
    """The layer-major prefill's program (compile_layer_run) of the prefix's last layer, one row
    at position pos, every expert present (the timing hack)."""
    from opentpu.compiler import arg_words
    from opentpu.llm.qwen3 import RunPos
    kind, M, spec = load_model(a.model)
    cfg = cfg_of()
    for prefix in a.prefix.split(":"):
        s, img = model_image(kind, M, spec, prefix, "moe", 1, cfg, a.cap, lookup=True)
        li, block = s.layers - 1, getattr(img, "block", 256)
        progs, ra = img.compile_layer_run(li, a.pos // block + 1, block, R=a.R,
                                          embedded=a.R > 1)
        args = arg_words(ra, dict(RunPos.values(791, a.pos, 1, block), row=3))
        progs = vpu_hack(progs, a.hack)
        dram = np.zeros(img.nbytes, np.uint8)
        timing_hack(img, dram)
        t = time.time()
        plus = rtlsim.ldc_plusargs(1066, MHZ) + ddr3_plusargs(3200 / 3, MHZ)[2:]
        _, _, st = rtlsim.run(cfg, progs, [dram], uarch=rtlsim.BOARD_UARCH, axi=True, boot=True,
                              stall=0, bw=100, lat=round(0.3 * MHZ), max_cycles=1 << 40,
                              ldc=1066, args=args, plusargs=["+axi_dram=1", "+axi_map=1"] + plus)
        print(json.dumps(dict(model=Path(a.model).name, prefix=prefix, layer=li, R=a.R, pos=a.pos,
                              hack=a.hack,
                              instructions=len(progs[0]), args=len(ra), cycles=st["cycles"],
                              ms=round(st["cycles"] / (MHZ * 1e3), 3),
                              sim_s=round(time.time() - t))), flush=True)


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


EM_USED = {"qwen35": 144, "gemma4": 75}    # a layer's distinct experts in a prompt chunk
EM_SENT = 7.0                               # an output place the run has not written


def _no_chain(progs):
    """The programs with each HALT CHAIN a plain HALT (one program timed alone)."""
    return [[I.halt(comment="(chain)") if x.op == I.HALT and x.flags & I.F_CHAIN else x
             for x in p] for p in progs]


def _put(dram, at: int, v) -> None:
    b = np.ascontiguousarray(v).view(np.uint8).ravel()
    dram[at:at + b.size] = b


def cmd_mtp(a):
    """The plain decode loop's token against the MTP loop's iteration (V then D) at each
    prefix of layers; the slots a layer of the whole model's images."""
    from opentpu.host import streamed as ST
    from opentpu.host.offload import LINE
    from opentpu.llm import generate as G
    from opentpu.llm import mtp as MT
    kind, M, spec = load_model(a.model)
    if kind != "qwen35":
        raise SystemExit("mtp: a Qwen3.5-MoE model")
    wformat, head, formats = ST.defaults(spec)
    full = board_config(PAIR=True, DSTEP=True, STREAM=True)      # the card's 4 GiB
    out = dict(model=Path(a.model).name, cap=a.cap_slots)
    for name, sp, rows in (("plain", spec, 1), ("mtp", dataclasses.replace(spec, mtp=True), 2)):
        img = ST.probe(sp, full, a.cap_slots, rows=rows, wformat=wformat, head_format=head,
                       formats=formats, lookup=True)
        out[f"slots_{name}"] = ST.fit_experts(sp, full, a.cap_slots, img)
        out[f"layers_{name}"] = img.offload.layers
    print(json.dumps(out), flush=True)
    cfg = cfg_of()
    blk, pos = 256, a.pos
    b = pos // blk + 1
    for prefix in a.prefix.split(":"):
        n = int(prefix)
        rec = dict(model=Path(a.model).name, prefix=n, pos=pos, formats=formats)
        for name, mtp in (("plain", False), ("mtp", True)):
            s = dataclasses.replace(spec, kinds=spec.kinds[:n], mtp=mtp)
            img = s.image(cfg, a.cap, rows=2 if mtp else 1, wformat=wformat, head_format=head,
                          formats=formats, lookup=True, experts=s.moe.k)
            dram = np.zeros(img.nbytes, np.uint8)
            timing_hack(img, dram)
            _put(dram, img.lookup["mbox"] + 2 * LINE, np.float32(1e6))     # rows served
            g = img.lookup["gen"]
            st = G.state_words(s, 791, pos, 8, [], blk)
            K = s.conv_k
            if not mtp:
                _put(dram, g["state"], st)
                progs = img.compile_generate(b, max((b - 1) * blk, K - 1), blk, chain=False)
                t = time.time()
                _, r = _rtl(cfg, progs, dram)
                rec.update(plain=r["cycles"], plain_ms=round(r["cycles"] / (MHZ * 1e3), 3),
                           plain_ins=len(progs[0]), plain_sim_s=round(time.time() - t))
                continue
            st[MT.S_DRAFT], st[MT.S_PAR] = 792, 0
            _put(dram, g["state"], st)
            for k in ("V0", "D"):
                progs = _no_chain(MT.compile_gen(img, b, MT.KINDS.index(k), blk))
                t = time.time()
                dram, r = _rtl(cfg, progs, dram)
                rec.update({k: r["cycles"], f"{k}_ms": round(r["cycles"] / (MHZ * 1e3), 3),
                            f"{k}_ins": len(progs[0]), f"{k}_sim_s": round(time.time() - t)})
        print(json.dumps(rec), flush=True)


def em_ids(rows: int, k: int, E: int, used: int, seed: int, route=None) -> np.ndarray:
    """[rows, k] layer-local expert ids: the trace's first rows, or each row k distinct of
    `used` experts (a random set), uniform."""
    if route is not None:
        ids = np.load(route).astype(np.int64)
        if ids.ndim != 2 or ids.shape[1] != k or len(ids) < rows or ids.min() < 0 \
                or ids.max() >= E:
            raise ValueError(f"{route}: [rows >= {rows}, {k}] ids below {E}, has {ids.shape}")
        return ids[:rows]
    rng = np.random.default_rng(seed)
    pool = rng.permutation(E)[:max(k, min(used, E))]
    return np.stack([rng.choice(pool, k, replace=False) for _ in range(rows)])


def em_passes(ids: np.ndarray, E: int, C: int = 2, S: int = 0) -> tuple:
    """(passes, distinct experts, rows loaded): the bucketing's passes of up to C entries of one
    expert. A pass loads and stores C rows (a part-filled one repeats its first entry), and an
    expert's last pass S when it has 1 .. S entries (the short pass, moe.em_short_rows)."""
    c = np.bincount(ids.reshape(-1), minlength=E)
    p, r = -(-c // C), c % C
    short = (r > 0) & (r <= S)
    return int(p.sum()), int((c > 0).sum()), int((p * C - short * (C - S)).sum())


def em_bytes(ids: np.ndarray, passes: int, loads: int, fb: int, H: int) -> dict:
    """The expert run's DRAM traffic by kind, in bytes of whole 64-byte lines: each pass streams
    its expert (fb), reads its rows' X and 1 / rms and writes their outputs (`loads` rows in
    all: em_passes), reads its directory entry and zeroes its slot's tag; the tables read each
    row's ids."""
    def ln(b):
        return -(-b // 64) * 64
    rows, k = ids.shape
    return dict(weights=passes * fb, rows_in=loads * (ln(4 * H) + 64),
                rows_out=loads * ln(4 * H), tables=rows * ln(4 * k), words=passes * 2 * 64)


def lm_passes(ids: np.ndarray, R: int = 2) -> int:
    """Layer-major's expert passes on the same ids, runs of R rows: each run's union once."""
    return sum(len(np.unique(ids[r:r + R])) for r in range(0, len(ids), R))


def em_layer_slots(L, scratch: int, distinct: int, k: int) -> int:
    """Slots a layer for an expert run's image (Layout L's layers, at most E slots each): its
    scratch of `scratch` bytes, then every used expert in a slot of its own (em_dram) and at
    least the 3 E slots expert-major needs by default (streamed.expert_major_fits), so that, as
    on the card, the run's experts' bound starts past the scratch (moe.em_expert_slots; an image
    of fewer than em_layers layers has every slot as the bound)."""
    need = -(-scratch // L.pitch) + max(distinct, 3 * L.E)
    return min(L.E, max(k, -(-need // len(L.slots))))


def em_layers(L, scratch: int) -> int:
    """The MoE layers an expert run's image needs for em_layer_slots' 3 E slots past the scratch
    at E slots a layer (4 on the 35B: 25 + 768 slots)."""
    return -(-(-(-scratch // L.pitch) + 3 * L.E) // L.E)


def em_dram(img, li: int, ids: np.ndarray, seed=None):
    """(DRAM, the experts' slots): the zero image with the words the expert run of layer li
    reads once every expert its chunk needs has landed (the module docstring). seed: random
    data instead of zeros (the rows' X and 1 / rms, the layer's gains, each used expert's
    weights packed into its slot), for compare's bit-for-bit check."""
    from opentpu.compiler import Affine as Af
    dsc = img.descriptors(0)
    dev, lw, Lo = dsc.moe_dev, dsc.layer(li), img.offload
    E, H, (rows, k) = Lo.E, dev.fmt.H, ids.shape
    dram = np.zeros(-(-img.nbytes // 4) * 4, np.uint8)

    def put(addr, a):
        a = np.ascontiguousarray(a)
        dram[addr:addr + a.nbytes] = a.view(np.uint8).reshape(-1)

    put(Af.of(lw.gbase.base).const, np.float32([li * E]))       # the layer's j * E
    rng = None if seed is None else np.random.default_rng(seed)
    if rng is not None:
        for gn in ("g_post", "g_exp"):
            if getattr(lw, gn, None) is not None:
                put(Af.of(getattr(lw, gn).base).const,
                    (0.5 + rng.random(H)).astype(np.float32))
    tail = 4 * H * (k + 2)
    for r in range(rows):
        a = dev.em_base + r * dev.em_rec
        if rng is not None:
            put(a, rng.standard_normal(H).astype(np.float32))   # X
        put(a + 4 * H, np.full(k * H, EM_SENT, np.float32))     # OUT: the run writes them
        put(a + tail, (li * E + ids[r]).astype(np.float32))     # global ids
        put(a + tail + 4 * k, np.full(k, 1.0 / k, np.float32))  # weights
        put(a + tail + 8 * k, np.float32([1.0 if rng is None else 0.5 + rng.random()]))
    # the experts past the server's scratch (Layout.scratch: the image's prefill rows' records,
    # moe.em_slots), where the run's slot bound has them (dev.em_slots)
    scratch = len(MO.em_slots(Lo, max(rows, dev.em_rows) * dev.em_rec))
    used, slots = np.unique(ids), Lo.all_slots()
    if scratch + len(used) > len(slots):
        raise ValueError(f"{scratch} scratch slots and {len(used)} experts: the image has "
                         f"{len(slots)} slots")
    es = slots[scratch:scratch + len(used)]
    f, lanes = dev.fmt, getattr(img, "slot_lanes", None)
    for e, a in zip(used, es):
        if rng is not None:                                     # its weights
            w = [0.05 * rng.standard_normal(sh).astype(np.float32)
                 for sh in ((f.F0, H), (f.F0, H), (H, f.F0))]
            data = f.pack(*w)
            if lanes is not None:       # with the channel hash: lanes by address (pool_build's)
                wg, D, own = lanes
                for o, n in own:
                    data[o:o + n] = Q.hash_rows(data[o:o + n], a + o, wg, D)
            put(a, data)
        put(Lo.dir + 8 * (li * E + int(e)), np.array([a, 0x3F800000], np.uint32))  # {slot, 1.0}
        put(a + Lo.tag, np.uint32([1]))                         # landed
    return dram, es


def em_outputs(after, img, rows: int, k: int) -> np.ndarray:
    """The records' OUT rows [rows, k, H] (the expert run's outputs), as uint32 bits."""
    dev = img.descriptors(0).moe_dev
    H, w = dev.fmt.H, after.view(np.uint32)
    return np.stack([w[(dev.em_base + r * dev.em_rec + 4 * H) // 4:][:k * H].reshape(k, H)
                     for r in range(rows)])


def em_check(after, img, li: int, ids: np.ndarray, es) -> None:
    """Every used expert's tag zeroed, every (row, rank) output place written."""
    dev, Lo = img.descriptors(0).moe_dev, img.offload
    H, (rows, k) = dev.fmt.H, ids.shape
    w = after.view(np.uint32)
    bad = [a for a in es if w[(a + Lo.tag) // 4]]
    if bad:
        raise AssertionError(f"{len(bad)} of {len(es)} experts' tags not zeroed")
    f = after.view(np.float32)
    for r in range(rows):
        o = (dev.em_base + r * dev.em_rec + 4 * H) // 4
        left = [i for i in range(k) if (f[o + i * H:o + (i + 1) * H] == EM_SENT).any()]
        if left:
            raise AssertionError(f"row {r}: ranks {left} not written")


def _sets(items) -> dict:
    """--set KEY=VALUE pairs: the board configuration's overrides (ints; true / false)."""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        b = {"true": True, "false": False}.get(v.lower())
        out[k] = int(v, 0) if b is None else b
    return out


def cmd_em(a):
    """Expert-major's expert run (module docstring)."""
    from opentpu.compiler import arg_words
    from opentpu.isasim import Machine
    kind, M, spec = load_model(a.model)
    if kind not in ("qwen35", "gemma4"):
        raise SystemExit("em: Qwen3.5-MoE or Gemma 4 MoE")
    over = _sets(a.set)
    cfg = cfg_of(**over)
    prefix = a.prefix or ("1" if kind == "qwen35" else "0")
    s, img = model_image(kind, M, spec, prefix, "moe", 1, cfg, a.cap)
    mo, li = s.moe, s.layers - 1
    used = a.used or EM_USED[kind]
    for rows in [int(x) for x in a.rows.split(",")]:
        ids = em_ids(rows, mo.k, mo.E, used, a.seed, a.route)
        distinct = em_passes(ids, mo.E)[1]
        L = img.offload
        n = em_layer_slots(L, max(rows, img.prefill_rows)
                           * MO.em_record(s.hidden, mo.k, MO.em_qd(img.cfg)),
                           distinct, mo.k)
        s, img = model_image(kind, M, spec, prefix, "moe", 1, cfg, a.cap, experts=n)
        progs, ra = img.compile_expert_run(li, a.em_pass)
        C, S = img.em_pass                          # (the pass's entries, a short pass's)
        passes, _, loads = em_passes(ids, mo.E, C, S)
        args = arg_words(ra, {"rows": rows, "entries": rows * mo.k})
        dram, es = em_dram(img, li, ids, a.seed if a.random else None)
        fb = img.fmt.nbytes
        rec = dict(model=Path(a.model).name, layer=li, rows=rows, k=mo.k, E=mo.E,
                   used=distinct, entries=rows * mo.k, C=C, S=S, passes=passes,
                   instructions=len(progs[0]), slots=n, image_mib=round(img.nbytes / 2**20),
                   cfg={**{k_: getattr(cfg, k_) for k_ in ("MCOLS", "LANES", "PAIR", "DSTEP",
                                                            "STREAM")}, **over},
                   expert_mib=round(passes * fb / 2**20, 1),
                   dram_bound_ms=round(passes * (fb + 16 * s.hidden) / 14.1e9 * 1e3, 3))
        mv = em_bytes(ids, passes, loads, fb, s.hidden)
        lm = lm_passes(ids)
        rec.update(bytes=mv, bytes_a_row=round(sum(mv.values()) / rows),
                   weight_bytes_a_row=round(mv["weights"] / rows), lm2_passes=lm,
                   lm2_weight_bytes_a_row=round(lm * fb / rows))
        t = time.time()
        if a.isa:
            m = Machine(cfg, progs, [dram], args)
            sl, waits = m.slices[0], [0]
            step = sl.step

            def counted():              # the passes the program ran: one WAITW each
                waits[0] += sl.prog[sl.pc].op == I.WAITW
                step()
            sl.step = counted
            m.run(max_steps=1 << 34)
            after = sl.dram[:len(dram)]
            rec.update(icount=sl.icount, isa_passes=waits[0])
            if waits[0] != passes:
                raise AssertionError(f"the ISA ran {waits[0]} passes, the bucketing {passes}")
        else:
            after, st = _rtl(cfg, progs, dram, args, a.profile)
            rec.update(cycles=st["cycles"], ms=round(st["cycles"] / (MHZ * 1e3), 3),
                       ms_a_pass=round(st["cycles"] / (MHZ * 1e3) / max(passes, 1), 4))
        em_check(after, img, li, ids, es)
        if a.compare:                   # the same DRAM through passes of a.compare entries
            p2, _ = img.compile_expert_run(li, a.compare)
            m2 = Machine(cfg, p2, [dram], args).run(max_steps=1 << 34)
            o1 = em_outputs(after, img, rows, mo.k)
            o2 = em_outputs(m2.slices[0].dram[:len(dram)], img, rows, mo.k)
            rec.update(compare=a.compare, differing_words=int((o1 != o2).sum()),
                       nonzero_words=int((o1 != 0).sum()))
            if (o1 != o2).any():
                raise AssertionError(f"{int((o1 != o2).sum())} output words differ from passes "
                                     f"of {a.compare}")
        rec.update(check="ok", sim_s=round(time.time() - t))
        print(json.dumps(rec), flush=True)


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
    r.add_argument("--hack", default="", help="tools/vpu_hack.py modes (timing only)")
    la = sub.add_parser("layer")
    la.add_argument("model")
    la.add_argument("--prefix", default="0,1:0,5")
    la.add_argument("--pos", type=int, default=256)
    la.add_argument("--R", type=int, default=1, help="rows a run (all routed alike: the zero "
                    "image's router ties, so a run's union is k experts)")
    la.add_argument("--cap", type=int, default=1024)
    la.add_argument("--hack", default="", help="tools/vpu_hack.py modes (timing only)")
    e = sub.add_parser("expert")
    e.add_argument("model")
    e.add_argument("--M", default="1,2,3,4,8")
    m = sub.add_parser("em")
    m.add_argument("model")
    m.add_argument("--rows", default="131", help="the chunk's rows (a list)")
    m.add_argument("--prefix", default=None, help="layers (Qwen3.5 a count, default 1; Gemma 4 "
                   "checkpoint layers, default 0): the last one's expert run")
    m.add_argument("--used", type=int, default=0, help="distinct experts the chunk uses "
                   "(default: Qwen3.5 144, Gemma 4 75)")
    m.add_argument("--route", default=None, help="router trace: .npy [rows, k] layer-local ids")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--cap", type=int, default=1024)
    m.add_argument("--set", action="append", help="board configuration KEY=VALUE (repeat)")
    m.add_argument("--isa", action="store_true", help="the ISA simulator (the check, no cycles)")
    m.add_argument("--profile", action="store_true", help="per place: cycles (stderr)")
    m.add_argument("--random", action="store_true", help="random rows, gains and experts "
                   "(seeded by --seed) instead of zeros")
    m.add_argument("--compare", type=int, default=0, help="with --isa: also run passes of this "
                   "many entries on the same DRAM; the outputs must be equal")
    m.add_argument("--em-pass", type=int, default=None, help="entries a pass (default: the "
                   "configuration's, moe.em_pass_rows)")
    t = sub.add_parser("mtp")
    t.add_argument("model")
    t.add_argument("--prefix", default="4:8", help="layer counts (':' between)")
    t.add_argument("--pos", type=int, default=256, help="the token's position (the verify's "
                   "rows at pos, pos + 1)")
    t.add_argument("--cap", type=int, default=1024)
    t.add_argument("--cap-slots", type=int, default=2048, help="the KV capacity of the slot "
                   "count (otpu-chat's, chatmoe's)")
    a = ap.parse_args()
    {"rows": cmd_rows, "expert": cmd_expert, "layer": cmd_layer, "em": cmd_em,
     "mtp": cmd_mtp}[a.cmd](a)


if __name__ == "__main__":
    main()
