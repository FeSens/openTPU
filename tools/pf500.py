"""pf500: the shared benchmark of Qwen3.6-35B-A3B prefill (docs/pf500.md;
the scoreboard docs/pf500.md). One layer of each kind of the 35B (a DeltaNet layer and a
gated-attention layer, each with its MoE: every layer of the model has one) through the prefill's
programs at R rows a run, on a target's configuration, on a zero image (the timing does not depend
on the data, only config.json is read) whose routing is set: each row's k experts chosen (uniform
over the E experts, distinct in a row, seed 1; --route zero: the zero image's own, every row the
same k), by its residual row (one-hot), its layer's post-attention gain (ones) and router (row r's
experts' weights in column r), the only weights not zero.

    python3 tools/pf500.py isa [--target T,...] [--R 2,64,512] [--kinds dn,attn] [--pos 1024]
                               [--route uniform|zero] [--model DIR] [--json FILE]
    python3 tools/pf500.py sim ... [--plus +a=b,...]      the board model (RTL): see `sim` below

Paths (--path): lm, layer-major (Image.compile_layer_run: the R rows through layer li, its MoE
moe_ffn_rows at S = 1, a row at a time at S > 1: the chat's CHAT35_LM runs; R <= qwen35.RUN_ROWS
today); em, expert-major on the resident pool (items 2 and 3: Image(em=True), the chunk's rows in
their records; layer li's mixer and MoE prologue in tiles of --T rows, tiled.compile_layer_tile,
then the layer's expert run over the R rows, moe_expert_run: each expert once on the rows that
chose it; with DNCHUNK a DeltaNet layer's tiles as every tile's prep, core, then post), the runs in
turn on one DRAM, their work summed. A point that does not compile is a row with its error.
A target "+KEY[=VAL]" overrides its arch fields in the isa mode (bcu1525-x2:wmx+FSUM: wide-mxu's
EXP2SUB row sums, the compiler's exp2sub fsum + take, before a variant states them); compile
choices outside the CFG_HASH (DNCHUNK) in both modes.

isa: the ISA simulator with tools/dataflow.py's counters (the run is bit-exact control flow; MMs
skip their products where their weights are zero). A row: per phase (tools/prefill_phases.py's:
deltanet, attn-proj, attn-core, mlp = the MoE, its router and shared expert, inputs, other) and
per unit the work in cycles at the target's rates (no stalls, no overlap):
  MXU   the array's cycles: dataflow's mxu_adv, an MM in groups by the pass rule (mm_cycles;
        MXU_PUMP: two activation rows a pass)
  VPU   VOP issue cycles (dataflow.vpu_cycles; VPU_CNS 5: the composites in two passes; FSUM:
        the RSUM an EXP2SUB's sums replace is a take of FSUM_TAKE cycles)
  SE    the stream engine's DSTEP / STREAM cycles
  Q     the quantizer's TMEM read cycles (QACT, QST)
  DRAM  the bytes read and written over the slice's peak (bytes a core cycle: DIMMs x 19.2 GB/s
        at the core clock), and over the decode's measured 88 % of it (dram88)
  COLL  collectives x a latency each (S 1: 0; the hub's 88 / 152 at S 2 / 4 inside a card, + 2 L
        across the cards) + the gathered words into a slice at LANES a cycle
bound = the largest unit (a lower bound on the run's cycles), serial = their sum. DRAM bytes by
kind (weights and their scales, kv, the DeltaNet state, activations). The model: its layers of
each kind from --model's config.json (layout: layer_types, the measured layer of a kind its
first past layer 0; Qwen3.6-35B-A3B's 30 DeltaNet and 10 attention), cycles a token = the sum of
each kind's layers x its layer's cycles / R, tok/s at the core clock (150 MHz); a family the
runs do not serve (GAP: GPT-OSS, LFM2, Gemma 4) a row with its gap and layout. The lead's G0:
the card's 43 tok/s (layer-major R 2 on bcu1525-wg:bhw16).

sim: the same programs and words on the board model, after the ISA run (its instruction counts
and the run's residual rows checked against the board's): S = 1 tb_bcuwg (board_perf.bcuwg_transport:
SE_R 2, RLY 0x22555599, DMA_DEPTH 512), the x2 targets tb_x2 (cards.X2SimTransport, each card
tb_bcuwg's S = 2 at floorplan c, the link model at L 40), both with the card-calibrated memory
path K7 (`+cp=3333,+dimm_mc=1,+mc_ba=15,+mc_rw=62,+mc_strict=1,+mc_xbsw=16,+mc_xrtw=8,+mc_xwtr=8,
+mc_xgq=12`: S = 1 and S = 2 decode within ~1 % of the cards). A row: cycles, per phase the
trace's cycles (profile.phases' latest-end charge) and the DMA / MXU / VPU / DSTEP busy unions
(tools/prefill_phases.summarize), the run's DRAM bytes by kind; the model row as isa's from the
cycles. --path em: the point's runs in turn on the board's DRAM (the words written before the
first, the DRAM carried run to run), each run's instruction counts checked, then the chunk's R
records against the ISA's; cycles and busy unions summed, cycles by run label (by_run_cycles).
A Verilator build of the target's board model, then a run a point: heavy.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import dataflow as DF                                           # noqa: E402
import prefill_phases as PP                                     # noqa: E402

from opentpu import isa as I                                    # noqa: E402
from opentpu import isasim as SIM                               # noqa: E402
from opentpu import quant as Q                                  # noqa: E402
from opentpu import target as T                                 # noqa: E402

K7 = ("+cp=3333,+dimm_mc=1,+mc_ba=15,+mc_rw=62,+mc_strict=1,"
      "+mc_xbsw=16,+mc_xrtw=8,+mc_xwtr=8,+mc_xgq=12")

# the 35B's config.json (only it is read): the checkpoint's, else the copy beside this tool
MODEL = os.environ.get("PF500_MODEL") or str(next(
    (d for d in (ROOT / "models" / "Qwen3.6-35B-A3B", ROOT / "tools/pf500_models/Qwen3.6-35B-A3B")
     if (d / "config.json").exists()), ROOT / "tools/pf500_models/Qwen3.6-35B-A3B"))
TARGETS = ("bcu1525-wg:bhw16", "bcu1525-x2:st16", "bcu1525-x2:wmx")
MHZ = 150.0
DIMM_GBS = 19.2                 # a DDR4-2400 DIMM's peak
DRAM_EFF = 0.88                 # the decode's measured share of it (bcu-wg: 86-90 %)
CUT_VOCAB = 4096                # the cut's vocabulary (no layer run reads the head or the table;
#                                 the x2 cuts' q36-4: the low area 536 -> ~30 MiB)
# a config.json's model_type: its family (streamed.FORMATS' key). The bench's runs are the
# Qwen3.5-MoE image's (expert-major: tiled.LAYER_KERNELS' qwen35 layer tiles at S > 1 and
# Image.compile_expert_run); another family's model is a row with its gap, not a measurement
FAMILY = {"qwen3_5_moe": "qwen35", "qwen3_5_moe_text": "qwen35", "qwen3_5": "qwen35",
          "qwen3_5_text": "qwen35", "gpt_oss": "gptoss", "lfm2": "lfm2", "lfm2_moe": "lfm2",
          "gemma4": "gemma4", "gemma4_text": "gemma4"}
GAP = {"gptoss": "gptoss.Image has no expert-major prefill (no em image, no compile_expert_run) "
                 "and no tiled layer run (tiled.LAYER_KERNELS: gemma4, qwen35)",
       "lfm2": "lfm2.Spec.check: MoE layers run on one slice (S = 1); no expert-major prefill "
               "and no tiled layer run (tiled.LAYER_KERNELS: gemma4, qwen35)",
       "gemma4": "pf500's runs are Qwen3.5's (DeltaNet parts, RunPos, the qwen35 expert run); "
                 "gemma4's layer tiles are not wired in"}
# a layer_types entry: the bench's kind (Qwen3.5's DeltaNet dn and gated attention attn; the
# other families' kinds name their layers in the layout and the gap row)
KIND = {"linear_attention": "dn", "full_attention": "attn", "sliding_attention": "swa",
        "conv": "conv"}


@dataclasses.dataclass(frozen=True)
class Layout:
    """A model's layers as the bench measures them, from its config.json (text_config's where it
    has one): each layer's kind (layer_types, KIND), the cut (the model's first layers, every
    kind's measured layer in it: the kind's first layer past layer 0, which reads the
    embedding), each kind's layers in the model, its MoE layers (all but num_dense_layers when
    it has experts), and the family's gap (None: the bench runs it)."""
    family: str
    kinds: tuple
    cut: int
    layer: dict
    counts: dict
    moe: int
    gap: str | None


def layout(model: str) -> Layout:
    c = json.loads((Path(model) / "config.json").read_text())
    t = c.get("text_config") or c
    mt = c.get("model_type")
    fam = FAMILY.get(mt) or FAMILY.get(t.get("model_type"))
    if fam is None:
        return Layout(str(mt), (), 0, {}, {}, 0, f"model_type {mt!r}: not a family the bench "
                      f"knows ({', '.join(sorted(set(FAMILY)))})")
    kinds = tuple(KIND.get(x, x) for x in t["layer_types"])
    layer = {k: next((i for i, x in enumerate(kinds) if x == k and i > 0), kinds.index(k))
             for k in dict.fromkeys(kinds)}
    counts = {k: kinds.count(k) for k in layer}
    E = t.get("num_experts") or t.get("num_local_experts") or 0
    moe = len(kinds) - int(t.get("num_dense_layers") or 0) if E else 0
    gap = GAP.get(fam)
    if gap is None and moe != len(kinds):
        gap = f"{len(kinds) - moe} dense layers: the bench's em runs are every layer's MoE"
    return Layout(fam, kinds, max(layer.values()) + 1, layer, counts, moe, gap)


_L = layout(MODEL)                 # the default model's (the 35B's: DeltaNet 0-2, attention 3)
CUT, LAYER, MODEL_LAYERS = _L.cut, _L.layer, _L.counts      # 4, {dn 1, attn 3}, {dn 30, attn 10}
PHASES = ("deltanet", "attn-proj", "attn-core", "mlp", "inputs", "other")
UNITS = ("MXU", "VPU", "SE", "Q", "DRAM", "COLL")
# VPU_CNS 5's composites (the wide-mxu note: EXP2 / EXP2SUB / RSQRT / LOG2 at 4 columns a cycle)
PASSES5 = {I.V_EXP2: 2, I.V_EXP2SUB: 2, I.V_RECIP: 2, I.V_RSQRT: 2, I.V_LOG2: 2,
           I.V_SIGMOID: 4, I.V_SILU: 4}      # VSIG (otpu_se_comp SIG): four passes each
# FSUM (wide-mxu, otpu_vpu VPU_FSUM): an EXP2SUB with its row sums streams as a plain one; the
# RSUM it replaces is a take, an empty VOP of ~2 cycles (no TMEM traffic, no rows x cols stream)
FSUM_TAKE = 2


# ---------------------------------------------------------------------------- the target
@dataclasses.dataclass
class Tgt:
    name: str
    cfg: object
    cards: int
    spc: int                    # slices a card
    dimms: int                  # DIMMs a slice
    pump: bool
    cns: int
    link_l: int = 40            # the QSFP link's one-way core cycles (qsfp-link s6: ~39)
    qstb: bool = False          # QUANT_QSTB: QST rows (ROW) through the quantizer's row buffer

    @property
    def dram_bpc(self) -> float:
        """A slice's peak DRAM bytes a core cycle."""
        return self.dimms * DIMM_GBS * 1e3 / MHZ

    def gather_lat(self) -> int:
        """A GATHER's latency on the slice's critical path (co-sim measurements at S = 2 / 4,
        COLL_PIPE: 88 / 152 cycles a 256-word GATHER; across cards the link's round trip)."""
        S = self.cfg.S
        if S == 1:
            return 0
        lat = 88 if S == 2 else 152
        return lat + (2 * self.link_l if self.cards > 1 else 0)


def target(name: str) -> Tgt:
    """A target by name; "+KEY[=VAL]" after it overrides arch fields, the ISA mode only (e.g.
    bcu1525-x2:wmx+FSUM: EXP2SUB's row sums before a variant states them); +QUANT_QSTB the
    quantizer's QSTB (a micro-architecture knob: the ledger's QST rows read once)."""
    base, *over = name.split("+")
    t = T.load(base)
    kw = {}
    for o in over:
        k, _, v = o.partition("=")
        kw[k] = True if not v else int(v, 0) if v.lstrip("-").isdigit() or v.startswith("0x") else v
    qstb = kw.pop("QUANT_QSTB", None)
    cfg = T.config(t, **kw)
    if cfg.FSUM and not (cfg.DSTEP and cfg.LANES == 8 and cfg.D >= 32):
        raise ValueError(f"{name}: FSUM needs DSTEP, LANES 8, D >= 32")
    cards = int(t.platform.get("cards", 1))
    dimms = 4 if t.name == "bcu1525-wg" else 2      # (bcu1525-s2 / -x2: two DIMMs a slice)
    ua = {**t.arch, **t.uarch}
    return Tgt(name, cfg, cards, cfg.S // cards, dimms, bool(ua.get("MXU_PUMP") or False),
               int(ua.get("VPU_CNS") or 3),
               qstb=bool(ua.get("QUANT_QSTB") or False) if qstb is None else bool(qstb))


# ---------------------------------------------------------------------------- the image
_IMAGES: dict = {}


def image(tg: Tgt, model: str, pos: int, em: bool = False, rows: int = 512):
    """The model cut to its layout's first layers (vocabulary CUT_VOCAB) on the target's
    configuration, as the chat's engine builds it (streamed.FORMATS: fp4 experts resident,
    nvfp4 attention / DeltaNet / shared MLP, int8 head; lookup tables; `rows` prefill rows, the
    chat's 512), KV capacity the next 4096 past pos; em: the expert-major records
    (Image(em=True)). A family the runs do not serve (the layout's gap): ValueError."""
    from opentpu.host.streamed import FORMATS
    from opentpu.llm import load_spec
    cap = 4096 * (pos // 4096 + 1)
    key = (tg.name, model, cap, em, rows)
    if key not in _IMAGES:
        L = layout(model)
        if L.gap:
            raise ValueError(f"{L.family}: {L.gap}")
        spec = load_spec(model)
        s = dataclasses.replace(spec, kinds=spec.kinds[:L.cut], vocab=CUT_VOCAB)
        wf, hf, fm = FORMATS[L.family]
        img = s.image(tg.cfg, cap, 1, 1, wformat=wf, head_format=hf, formats=fm,
                      lookup=True, prefill_rows=rows, **({"em": True} if em else {}))
        img.__dict__["_pf500_layout"] = L
        _IMAGES[key] = img
    return _IMAGES[key]


def compile_runs(img, kind: str, R: int, pos: int, path: str = "lm", T: int = 16,
                 C: int | None = None) -> list:
    """[(label, programs, ARG words)]: the layout's layer of `kind`'s runs of R rows from
    position pos (row 0 of the chunk). lm: one layer run; em: tiles of at most T rows (none
    across an attention block) and the layer's expert run (passes of at most C entries: default the
    image's, Image.compile_expert_run). A DeltaNet layer with Config.DNCHUNK: every tile's
    prep, every tile's core, every tile's post (tiled.dn_parts: the core tiles' DSTEPs, the
    recurrence, or where the image has it the chunked core, Image.compile_dn_run, over the whole
    chunks of DNCHUNK rows), labelled prep / core / post and the rows (the chunked core: dnc)."""
    from opentpu.compiler import arg_words
    from opentpu.llm import tiled as TL
    from opentpu.llm.qwen3 import ATTN_BLOCK as B, RunPos
    li, K = img.__dict__.get("_pf500_layout", _L).layer[kind], img.spec.conv_k
    if path == "lm":
        progs, ra = img.compile_layer_run(li, pos // B + 1, R=R)
        return [("lm", progs, arg_words(ra, {"row": 0, **RunPos.values(0, pos, K, B)}))]
    if path != "em":
        raise ValueError(f"path {path!r}: lm or em")
    if R > img.prefill_rows:
        raise ValueError(f"{R} rows: the image's chunk holds {img.prefill_rows}")
    cache = img.__dict__.setdefault("_pf500_runs", {})
    core = TL.dn_core(img) if img.cfg.DNCHUNK and kind == "dn" else None
    cut = TL.core_end(core, 0, R)
    tiles, i = [], 0                    # (Engine.prefill_layers' runs of the layer)
    while i < R:
        p = pos + i
        n = min(T, R - i, B - p % B, cut - i if cut and i < cut else T)
        tiles.append(((li, p // B + 1, n, True, "tile"), dict(RunPos.values(0, p, K, B), row=i),
                      None))
        i += n
    if img.cfg.DNCHUNK and kind == "dn":
        tiles = TL.dn_merged(TL.dn_parts(tiles, core), img.cfg.DNMERGE)
    runs = []
    for key, vals, _ in tiles:
        if key not in cache:
            cache[key] = img.compile_dn_run(key[1]) if key[0] == "dnc" else \
                TL.compile_layer_tile(img, li, key[1], key[2],
                                      **({"part": key[-1]} if key[-1] != "tile" else {}))
        progs, ra = cache[key]
        lab = f"dnc{vals['rows']}" if key[0] == "dnc" else \
            f"{key[-1][3:].split(':')[0] if key[-1] != 'tile' else 'tile'}{key[2]}" + \
            (f"x{key[-1].split(':')[1]}" if ":" in str(key[-1]) else "")
        runs.append((lab, progs, arg_words(ra, vals)))
    if ("x", li, C) not in cache:
        cache["x", li, C] = img.compile_expert_run(li, C)
    progs, ra = cache["x", li, C]
    runs.append(("expert", progs, arg_words(ra, {"rows": R, "entries": R * img.spec.moe.k})))
    return runs


# ---------------------------------------------------------------------------- the routing
def route_ids(R: int, E: int, k: int, seed: int = 1) -> np.ndarray:
    """Each row's k distinct experts, uniform over E (tools/moe_prefill_cosim.em_ids' rule at
    used = E)."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(E, k, replace=False) for _ in range(R)])


def route_words(img, li: int, ids: np.ndarray, em: bool = False) -> tuple[list, list]:
    """(per slice [(address, bytes)], the router's DRAM ranges): row r's residual one-hot at
    column r (em: in its record's X), layer li's post-attention gain ones, its router's expert
    rows ids[r, j]'s weight in column r k - j (so row r's top k are ids[r] in that order; the
    shared expert's gate row and every other weight zero). At S > 1 the router's rows are split
    as Image.build splits them."""
    spec, cfg = img.spec, img.cfg
    H, E, S, D = spec.hidden, spec.moe.E, cfg.S, cfg.D
    R, k = ids.shape
    if R > H:
        raise ValueError(f"{R} rows: one-hot routing needs R <= hidden ({H})")
    rw = np.zeros((E + 1, H), np.float32)
    for r in range(R):
        rw[ids[r], r] = k - np.arange(k)
    if S > 1:
        rw = np.pad(rw, ((0, S * img.er - len(rw)), (0, 0)))
        parts = [rw[s * img.er:(s + 1) * img.er] for s in range(S)]
    else:
        parts = [rw]
    if em:                          # (each row's record: X its first H words)
        from opentpu.llm.moe import em_qd, em_record
        rec = em_record(H, k, em_qd(img.cfg)) // 4
        x = np.zeros((R, rec), np.float32)
        xa = img.io["em"]
    else:
        x, xa = np.zeros((R, H), np.float32), img.io["xbuf"]
    x[np.arange(R), np.arange(R)] = 1.0
    words, live = [], []
    for s in range(S):
        lw = img.descriptors(s).layer(li)
        rt = lw.router
        d0, s0 = rt.data.const, rt.scale.const
        q, sc = Q.quantize_mxu(parts[s], "int8", D)
        dq, ds = Q.grouped(q, sc, "int8", H, cfg.WG, D, at=d0 if cfg.CHASH else None)
        dq, ds = np.ascontiguousarray(dq).view(np.uint8).reshape(-1), \
            np.ascontiguousarray(ds).view(np.uint8).reshape(-1)
        words.append([(d0, dq), (s0, ds),
                      (lw.g_post.base.const, np.ones(H, np.float32).view(np.uint8)),
                      (xa, x.view(np.uint8).reshape(-1))])
        live.append([(d0, d0 + dq.size)])
    return words, live


# ---------------------------------------------------------------------------- the ISA ledger
def mm_cycles(ins, pump: bool) -> int:
    """An MM in groups (WG > 1: every MM's): the array's cycles by otpu_mxu's pass rule. A pass is one
    activation row, or two under MXU_PUMP (rows 2p and 2p + 1; an odd M's last pass row 2p alone),
    and takes KB advances; its result FIFO rows (a row's one, 4-bit two) are pushed a cycle each, and
    a pass's last advance follows the last pass's at least their rows later (pu_hold; unpumped the
    4-bit rows' lk1), so a pass takes max(KB, its result rows) cycles: G groups x the passes' sum.
    KB 1 int8 pumped is 2 cycles a pass, not dataflow's halved M (the DeltaNet core's MMs, 2x)."""
    w = ins.w
    N, KB, M = w[3] & 0xFFFF, (w[3] >> 16) & 0xFF, max(1, (w[5] >> 16) & 0xFF)
    G = -(-N // (1 << ins.rd))
    per = 1 if (ins.flags >> I.WF_SHIFT) & 3 == I.W8 else 2      # result FIFO rows an activation row
    rows = [2] * (M // 2) + [1] * (M % 2) if pump and M > 1 else [1] * M
    return G * sum(max(KB, r * per) for r in rows)


class PSlice(DF.LedgerSlice):
    """dataflow's slice with each class's phase, the target's MXU / VPU rates, MMs over live
    (non-zero) weights computed, no DRAM flow tags (their 4 bytes a beat of a 32 GiB slice)."""

    live: tuple = ()
    pump = False
    cns = 3

    def __init__(self, cfg, sid, program, dram, fast=True):
        SIM.Slice.__init__(self, cfg, sid, program, dram)
        self.fast = fast
        self.cls, self.names, self.acc, self.phase = {}, [], [], []
        Tw = cfg.TMEM_WORDS
        self.t_w = np.full(Tw, -1, np.int32)
        self.t_n = np.zeros(Tw, np.uint8)
        self.t_f = np.full(Tw, -1, np.int32)
        from collections import defaultdict
        self.uses, self.flows, self.dflows = defaultdict(int), defaultdict(int), defaultdict(int)
        self.d_w = np.full(1, -1, np.int32)
        self.prog_bytes = 0
        self.cur = -1
        self.mxu = []               # per class: the array's cycles at the target's rate
        self.vpu5 = []              # per class: VOP issue cycles at the target's VPU_CNS
        self.ngather = []           # per class: collectives (GATHER, BAR)
        self.gwords = []            # per class: a slice's words a GATHER sends

    def klass(self, ins) -> int:
        ph = PP.phase(ins)
        k = (DF.opkey(ins), DF.site(ins), ph)
        c = self.cls.get(k)
        if c is None:
            c = self.cls[k] = len(self.names)
            self.names.append(k[:2])
            self.acc.append(DF.Acc())
            self.phase.append(ph)
            self.mxu.append(0)
            self.vpu5.append(0)
            self.ngather.append(0)
            self.gwords.append(0)
        return c

    def _a_mm(self, ins, a):
        before = a.mxu_adv
        post = super()._a_mm(ins, a)
        adv = a.mxu_adv - before
        if ins.rd:                  # groups: the passes' cycles (mm_cycles)
            adv = mm_cycles(ins, self.pump)
        elif self.pump:             # (rows of one: no pumped machine has them, WG > 1)
            M = (ins.w[5] >> 16) & 0xFF
            adv = adv * -(-M // 2) // max(M, 1)
        self.mxu[self.cur] += adv
        return post

    def _a_vop(self, ins, a):
        post = super()._a_vop(ins, a)
        if DF.vfunc(ins) == I.V_RSUM and ins.flags & I.F_FSUM:
            self.vpu5[self.cur] += FSUM_TAKE
        elif self.cns >= 5 and DF.vfunc(ins) in PASSES5:
            rows, cols = ins.w[3] & 0xFFFF, ins.w[3] >> 16
            self.vpu5[self.cur] += rows * -(-cols // self.cfg.LANES) * PASSES5[DF.vfunc(ins)]
        else:
            self.vpu5[self.cur] += DF.vpu_cycles(ins, self.cfg.LANES)
        return post

    def step(self) -> None:
        """A collective (the machine runs it, not execute): its class's count and words."""
        if self.pc < len(self.prog) and self.prog[self.pc].op in (I.GATHER, I.BAR):
            ins = self.prog[self.pc]
            c = self.klass(ins)
            self.ngather[c] += 1
            if ins.op == I.GATHER:
                self.gwords[c] += (ins.w[2] & 0xFFFF) * (ins.w[2] >> 16)
        super().step()

    def _mm_fast(self, ins) -> None:
        sa = self.da(ins.ra, ins.w[0])
        if any(lo <= sa < hi for lo, hi in self.live):
            return SIM.Slice.execute(self, ins)
        return super()._mm_fast(ins)


class PMachine(DF.LedgerMachine):
    def __init__(self, cfg, programs, args, tg: Tgt, live):
        assert len(programs) == cfg.S
        SIM._fit_imem(cfg, programs)
        self.cfg = cfg
        self.slices = []
        for s in range(cfg.S):
            sl = PSlice(cfg, s, programs[s], None)
            sl.live, sl.pump, sl.cns = tuple(live[s]) if live else (), tg.pump, tg.cns
            sl.QSTB = tg.qstb
            self.slices.append(sl)
        self.args = args
        self.collectives = 0
        self.chains = 0
        self.host = None
        for s in self.slices:
            s.R = SIM._regs(args)


def run_isa(tg: Tgt, progs, args, words=None, live=None, keep=None, drams=None) -> dict:
    """The ledger of slice 0, by phase and unit (cycles at the target's rates); keep: a list
    the machine is appended to (the board model's check); drams: the slices' DRAM to run on
    (a list: empty, it takes this run's, so the next run sees what this one wrote)."""
    m = PMachine(tg.cfg, progs, args, tg, live)
    if drams:
        for sl, d in zip(m.slices, drams):
            sl.dram = d
    elif drams is not None:
        drams.extend(sl.dram for sl in m.slices)
    if keep is not None:
        keep.append(m)
    for s, sl in enumerate(m.slices):
        for a, v in (words[s] if words else ()):
            sl.dram[a:a + v.size] = v
    t0 = time.time()
    m.run(max_steps=1 << 40)
    s = m.slices[0]
    cfg = tg.cfg
    st16 = 0.5 if getattr(cfg, "ST16", "") else 1.0     # (dataflow counts 4-byte states)
    ph = {p: {"MXU": 0, "VPU": 0, "SE": 0, "Q": 0, "COLL": 0, "rd": {}, "wr": {}, "n": 0}
          for p in PHASES}
    for c, (name, a) in enumerate(zip(s.names, s.acc)):
        d = ph[s.phase[c]]
        d["n"] += a.n
        d["MXU"] += s.mxu[c]
        if name[0].startswith("VOP"):
            d["VPU"] += s.vpu5[c]
        elif name[0] in ("DSTEP", "STREAM"):
            d["SE"] += a.vpu
        d["Q"] += max(a.rdc[p] for p in ("Q", "Q2", "Q3"))    # (own ports, read together)
        d["COLL"] += s.ngather[c] * tg.gather_lat() + cfg.S * s.gwords[c] // cfg.LANES
        d["collectives"] = d.get("collectives", 0) + s.ngather[c]
        for kind, v in a.dram_rd.items():
            v = v * st16 if kind == "state" else v
            d["rd"][kind] = d["rd"].get(kind, 0) + int(v)
        for kind, v in a.dram_wr.items():
            v = v * st16 if kind == "state" else v
            d["wr"][kind] = d["wr"].get(kind, 0) + int(v)
    for d in ph.values():
        d["bytes"] = sum(v for k, v in d["rd"].items()) + sum(
            v for k, v in d["wr"].items() if k != "qst_beats")
        d["DRAM"] = round(d["bytes"] / tg.dram_bpc)
    ph = {p: d for p, d in ph.items() if d["n"]}
    tot = {u: sum(d[u] for d in ph.values()) for u in UNITS}
    tot["bytes"] = sum(d["bytes"] for d in ph.values())
    tot["dram88"] = round(tot["DRAM"] / DRAM_EFF)
    return {"isa_s": round(time.time() - t0, 1), "instructions": s.icount,
            "program": len(progs[0]), "phases": ph, "total": tot}


def merge_runs(rs: list) -> dict:
    """run_isa's results of a layer's runs as one: instructions, phases' units and bytes summed
    (the program: the longest)."""
    if len(rs) == 1:
        return rs[0]
    ph: dict = {}
    for r in rs:
        for p, d in r["phases"].items():
            o = ph.setdefault(p, {"rd": {}, "wr": {}})
            for k, v in d.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        o[k][kk] = o[k].get(kk, 0) + vv
                else:
                    o[k] = o.get(k, 0) + v
    tot = {u: sum(r["total"][u] for r in rs) for u in (*UNITS, "bytes", "dram88")}
    return {"isa_s": round(sum(r["isa_s"] for r in rs), 1),
            "instructions": sum(r["instructions"] for r in rs),
            "program": max(r["program"] for r in rs), "phases": ph, "total": tot,
            "runs": {lab: sum(1 for x in rs if x.get("label") == lab)
                     for lab in dict.fromkeys(x.get("label") for x in rs)},
            "by_run": {lab: {u: sum(x["total"][u] for x in rs if x.get("label") == lab)
                             for u in UNITS}
                       for lab in dict.fromkeys(x.get("label") for x in rs)}}


def kinds_of(ph: dict) -> dict:
    """DRAM bytes by kind: weights (and their scales), kv, the DeltaNet state, activations."""
    out = {"weights": 0, "kv": 0, "dn_state": 0, "act": 0}
    for d in ph.values():
        for rw in ("rd", "wr"):
            for k, v in d[rw].items():
                if k == "qst_beats":
                    continue
                out["weights" if k in ("weights", "scales") else "kv" if k.startswith(
                    ("kv", "ld.kv")) else "dn_state" if k == "state" else "act"] += v
    return out


def bound_of(tot: dict) -> tuple[int, str]:
    u = max(UNITS, key=lambda k: tot[k])
    return int(tot[u]), u


# ---------------------------------------------------------------------------- the board model
def prog_at(img) -> int:
    """Where the programs go: past the image, or past an Image.high's low area (as
    BoardBackend loads them; board_perf.prefill_case's)."""
    from opentpu.llm.qwen3 import image_end
    return -(-(image_end(img) if getattr(img, "high", False) else img.nbytes) // 4096) * 4096


class SimBoard:
    """The target's board model as a Board, built once (the Verilator build is its params'):
    S = 1 tb_bcuwg, x2 tb_x2 (two cards of tb_bcuwg's S = 2, the link model)."""

    def __init__(self, tg: Tgt, at: int, plus: str, params: str, trace_depth: int):
        sys.path.insert(0, str(ROOT / "tools"))
        import board_perf as FP
        from opentpu import rtlsim
        from opentpu.host.board import Board
        # small C++ units, PF500_VJ compiler workers (x2_cosim's: the build within ~4 GB)
        fl = list(rtlsim.VFLAGS)
        if "-j" in fl:
            fl[fl.index("-j") + 1] = os.environ.get("PF500_VJ", "4")
        if "--output-groups" not in fl:
            fl += ["--output-groups", "0"]
        rtlsim.VFLAGS = fl
        t = T.load(tg.name.split("+")[0])
        if T.arch_hash(tg.cfg) != T.arch_hash(T.config(t)):
            raise ValueError(f"{tg.name}: arch overrides are the ISA mode's (the board model is "
                             f"a target's build; compile choices such as DNCHUNK are fine)")
        dense = 1 << (FP.prog_end(tg.cfg, at) - 1).bit_length()
        pl = [x for x in plus.split(",") if x]
        self.ct = None
        if tg.cards == 1:
            pr = f"SE_R=2,RLY=0x22555599,DMA_DEPTH=512,TRACE_DEPTH={trace_depth},TRACE_QD=256"
            self.tr = FP.bcuwg_transport(t, dense, pl, ",".join(filter(None, [pr, params])))
        else:
            from importlib.util import module_from_spec, spec_from_file_location
            from opentpu.host.cards import CardsTransport, X2SimTransport
            spec = spec_from_file_location("bcu_fp", ROOT / "boards/bcu1525/scripts/bcu_fp.py")
            fp = module_from_spec(spec)
            spec.loader.exec_module(fp)
            plan = fp.plan("c")
            pp = {"SE_R": 2, "DMA_DEPTH": 512, "RLY": plan["rly"], "DMAP": plan["dmap"],
                  "TRACE_DEPTH": trace_depth, "TRACE_QD": 256}
            for kv in filter(None, params.split(",")):
                k, v = kv.split("=", 1)
                pp[k] = int(v, 0)
            # (t itself: its name drops the variant, and the build would be the base target's)
            sim = X2SimTransport(target=t, ch_bytes=dense // tg.cfg.S * tg.cfg.S // 8,
                                 params=pp, plusargs=pl + ["+max_cycles=100000000"],
                                 link_lat=tg.link_l)
            self.tr = self.ct = CardsTransport(sim.cards)
        self.board = Board(self.tr, lock=False)
        self.written: list = []

    def run(self, progs, args, words, at: int, issued: int, fresh: bool = True) -> dict:
        """One run; fresh (a point's first run): the last point's words back to zero, then this
        point's written (else the DRAM as the run before left it: an expert-major sequence)."""
        b = self.board
        if fresh:
            for s, ws in enumerate(self.written):      # (the last point's words back to zero)
                for a, v in ws:
                    b.write(a, np.zeros(v.size, np.uint8), s)
            self.written = words or []
            for s, ws in enumerate(self.written):
                for a, v in ws:
                    b.write(a, v, s)
        b.load(at, progs if len(progs) > 1 else progs[0])
        if self.ct is not None:
            self.ct.collectives = issued
        if args is not None:
            b.set_args(list(args) + [0] * (8 - len(args)))
        return b.run(trace={"keep": "first"})

    def read(self, addr, n, s):
        return self.board.read(addr, n, s)


def run_sim(tg: Tgt, sb: SimBoard, img, progs, args, words, live, isa_m, prof=None) -> dict:
    """One point on the board model: cycles, the trace's phases and unit busy unions; checked
    against the ISA run (instruction counts, the R residual rows written). prof: a list the
    parsed trace (profile.parse's) is appended to."""
    from opentpu.host.hwlens import drop_orphans, records_to_trace
    from opentpu.profile import parse
    t0 = time.time()
    st = sb.run(progs, args, words, prog_at(img), isa_m.collectives)
    out = {"sim_s": round(time.time() - t0), "cycles": int(st["cycles"])}
    want = [sl.icount for sl in isa_m.slices]
    bad = [] if list(st["instructions"]) == want else [
        f"instructions {list(st['instructions'])} != ISA {want}"]
    xb = img.io["xbuf"]
    n = 4 * img.spec.hidden * isa_m.R_rows
    for s, sl in enumerate(isa_m.slices):
        got = sb.read(xb, n, s)
        if not np.array_equal(got, sl.dram[xb:xb + n]):
            bad.append(f"slice {s}: residual rows differ from the ISA's")
    out["ok"] = not bad
    if bad:
        out["mismatch"] = "; ".join(bad)
    tr = st["trace"]
    p = parse(drop_orphans(records_to_trace(tr["records"], progs)), tg.cfg, progs, "card")
    sm = PP.summarize(p, progs, tg.cfg)
    if prof is not None:
        prof.append(p)
    out["trace"] = {k: tr[k] for k in ("count", "drop", "depth", "lost")}
    out["phases"] = sm["phases"]
    out["busy"] = {k: sm["total"][k] for k in ("DMA", "MXU", "VPU", "DSTEP")}
    out["busy%"] = {k: round(100 * v / max(out["cycles"], 1), 1) for k, v in out["busy"].items()}
    out["dram_kinds"] = {k: sm["total"][k] for k in PP.KINDS}
    return out


def nonzero_spans(d: np.ndarray, end: int, page: int = 4096) -> list:
    """(address, bytes) spans of d[:end] (end a multiple of page) holding a nonzero byte, by page."""
    on = np.concatenate([d[c:min(end, c + (1 << 28))].view(np.uint64).reshape(-1, page // 8).any(1)
                         for c in range(0, end, 1 << 28)]) if end else np.zeros(0, bool)
    e = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0]))
    return [(int(a) * page, int(b - a) * page) for a, b in zip(e[0::2], e[1::2])]


def run_diag(p, progs) -> dict:
    """Where a run's time goes on the board model (slice 0 of its trace): each unit's busy union;
    per class (phase:op) its count and cycles in flight (start -> end), waiting on its dependencies
    (dispatch -> ready) and on its unit (ready -> start), and, for an MM, in flight before its
    dependencies cleared (start -> release: the MXU parked on the chain); the MXU's idle cycles
    (profile.mxu_gaps: before the first MM, after the last, between) and the gaps by the class
    that finished last before the MM was released."""
    cls = {}
    name = {}
    for r in p.slice_recs(0):
        if r.start < 0:
            continue
        k = name.get(r.pc)
        if k is None:
            ins = progs[0][r.pc]
            k = name[r.pc] = f"{PP.phase(ins)}:{DF.opkey(ins)}"
        a = cls.setdefault(k, [0, 0, 0, 0, 0])
        a[0] += 1
        a[1] += r.end - r.start
        a[2] += max(0, r.ready - r.dispatch)
        a[3] += max(0, r.start - r.ready)
        if r.op == I.MM and r.release >= 0:
            a[4] += max(0, r.release - r.start)
    g = p.mxu_gaps(0)
    blame = {}
    for (_, pc), v in g["blame"].items():
        k = name.get(pc, "?") if pc >= 0 else "?"
        blame[k] = blame.get(k, 0) + v
    return {"cycles": p.cycles, "busy": p.unit_busy(0), "cls": cls,
            "mxu_idle": {"prologue": g["prologue"], "epilogue": g["epilogue"],
                         "gaps": sum(b - a for a, b, _ in g["gaps"]), "blame": blame}}


def add_diag(acc: dict, d: dict) -> None:
    """Sum run_diag d into acc (a label's runs)."""
    if not acc:
        acc.update(json.loads(json.dumps(d)))
        acc["runs"] = 1
        return
    acc["runs"] += 1
    acc["cycles"] += d["cycles"]
    for k, v in d["busy"].items():
        acc["busy"][k] = acc["busy"].get(k, 0) + v
    for k, v in d["cls"].items():
        a = acc["cls"].setdefault(k, [0] * len(v))
        for i, x in enumerate(v):
            a[i] += x
    m, n = acc["mxu_idle"], d["mxu_idle"]
    for k in ("prologue", "epilogue", "gaps"):
        m[k] += n[k]
    for k, v in n["blame"].items():
        m["blame"][k] = m["blame"].get(k, 0) + v


def dump_recs(path: str, p, progs) -> None:
    """A run's trace records on slice 0, for analysis off the board: one JSON line per
    instruction, [pc, op, unit, dispatch, ready, start, release, end, stats] (stats: its unit's
    counters at its completion, the U line's: the MXU's starve / bp / frz / deny, QUANT's and
    the VPU's frz; {} for the others), after a line with each pc's class (phase:op) and
    comment, and slice 0's activity windows (the P / Q lines' counters by key, "c" their ends).
    The trace's other slices go to PATH-sN.jsonl in the same form, and slice 0's header has
    every traced slice's last end ("ends": a run's cycles are its slowest slice's)."""
    sl = sorted({r.slice for r in getattr(p, "recs", [])}) or [0]
    per = {s: sorted(p.slice_recs(s), key=lambda r: r.idx) for s in sl}
    ends = {s: max((r.end for r in rs), default=0) for s, rs in per.items()}
    wins = getattr(p, "buckets", None) or [{}]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for s, rs in per.items():
        pg = progs[s] if s < len(progs) else progs[0]
        head = {"cycles": p.cycles if s == 0 else ends[s], "pcs": {
            r.pc: [f"{PP.phase(pg[r.pc])}:{DF.opkey(pg[r.pc])}", pg[r.pc].comment[:80]]
            for r in rs}, "windows": wins[s] if s < len(wins) else {}}
        if s == 0:
            head["ends"] = ends
        with open(path if s == 0 else f"{path.removesuffix('.jsonl')}-s{s}.jsonl", "w") as f:
            f.write(json.dumps(head) + "\n")
            for r in rs:
                f.write(json.dumps([r.pc, r.op, r.unit, r.dispatch, r.ready, r.start, r.release,
                                    r.end, r.stats]) + "\n")


def run_sim_em(tg: Tgt, sb: SimBoard, img, runs, words, keep: list, R: int,
               dump: str | None = None) -> dict:
    """An expert-major point's runs on the board model in order (the DRAM carried run to run):
    cycles and the traces' unit busy unions summed, cycles by run label; checked against the ISA
    runs (keep: their machines, in order): each run's instruction counts, then the chunk's R
    records (residual, routes, the shared expert's rows). dump: a path prefix, each label's
    first run's records to PREFIX-LABEL.jsonl (dump_recs)."""
    from opentpu.host.hwlens import drop_orphans, records_to_trace
    from opentpu.llm.moe import em_qd, em_record
    from opentpu.profile import parse
    t0 = time.time()
    out = {"cycles": 0, "busy": dict.fromkeys(("DMA", "MXU", "VPU", "DSTEP"), 0),
           "by_run_cycles": {}, "trace_lost": 0, "diag": {}}
    bad = []
    for j, ((lab, progs, args), m) in enumerate(zip(runs, keep)):
        st = sb.run(progs, args, words if j == 0 else None, prog_at(img), m.collectives,
                    fresh=j == 0)
        c = int(st["cycles"])
        want = [sl.icount for sl in m.slices]
        if list(st["instructions"]) != want:
            bad.append(f"run {j} ({lab}): instructions {list(st['instructions'])} != ISA {want}")
        tr = st["trace"]
        p = parse(drop_orphans(records_to_trace(tr["records"], progs)), tg.cfg, progs, "card")
        sm = PP.summarize(p, progs, tg.cfg)
        out["cycles"] += c
        out["by_run_cycles"][lab] = out["by_run_cycles"].get(lab, 0) + c
        out["trace_lost"] += int(tr["drop"]) + int(tr["lost"])
        for k in out["busy"]:
            out["busy"][k] += sm["total"][k]
        if dump and lab not in out["diag"]:
            dump_recs(f"{dump}-{lab}.jsonl", p, progs)
        add_diag(out["diag"].setdefault(lab, {}), run_diag(p, progs))
    base, n = img.io["em"], R * em_record(img.spec.hidden, img.spec.moe.k, em_qd(img.cfg))
    for s, sl in enumerate(keep[-1].slices):
        if not np.array_equal(sb.read(base, n, s), sl.dram[base:base + n]):
            bad.append(f"slice {s}: the chunk's records differ from the ISA's")
    out["ok"] = not bad
    if bad:
        out["mismatch"] = "; ".join(bad)[:600]
    # (what the runs wrote, back to zero before the next point: the ISA's DRAM nonzero below the
    # programs, as the board's if it matched)
    w = list(sb.written) + [[]] * (len(keep[-1].slices) - len(sb.written))
    for s, sl in enumerate(keep[-1].slices):
        w[s] = list(w[s]) + [(a, np.broadcast_to(np.uint8(0), (n,)))
                             for a, n in nonzero_spans(sl.dram, prog_at(img))]
    sb.written = w
    out["sim_s"] = round(time.time() - t0)
    out["busy%"] = {k: round(100 * v / max(out["cycles"], 1), 1) for k, v in out["busy"].items()}
    return out


# ---------------------------------------------------------------------------- the commands
def arm_row(a, rec: dict) -> None:
    """--json with --em-pipe arms: each point also to its arm's own file as it finishes
    (FILE-pN.jsonl beside FILE: an arm's answer before the next arm runs)."""
    if getattr(a, "json", None) and rec.get("em_pipe") is not None:
        f = Path(a.json)
        with open(f.with_name(f"{f.stem}-p{rec['em_pipe']}{f.suffix}"), "a") as o:
            o.write(json.dumps(rec) + "\n")


def cmd_isa(a) -> list:
    rows = []
    L = layout(a.model)
    kinds = (getattr(a, "kinds", None) or ",".join(L.counts)).split(",")
    for tname in a.target.split(","):
        tg = target(tname)
        em = a.path == "em"
        if L.gap:                       # (the family's gap: a row a point with the layout)
            for R in [int(x) for x in a.R.split(",")]:
                rec = {"mode": a.cmd, "target": tname, "S": tg.cfg.S, "path": a.path, "R": R,
                       "kind": "model", "pos": a.pos, "route": a.route, "family": L.family,
                       "layers": L.counts, "moe": L.moe, "error": f"{L.family}: {L.gap}"}
                print(json.dumps(rec), flush=True)
                rows.append(rec)
            continue
        img = image(tg, a.model, a.pos, em, max([512] + [int(x) for x in a.R.split(",")]))
        E, k = img.spec.moe.E, img.spec.moe.k
        sb = SimBoard(tg, prog_at(img), a.plus, a.params, a.trace_depth) \
            if a.cmd == "sim" else None
        arms = [int(x) for x in getattr(a, "em_pipe", "2").split(",")] if em else [None]
        for R, ep in [(int(x), ep) for x in a.R.split(",") for ep in arms]:
            per = {}
            if ep is not None:                  # (moe.EM_PIPE: the expert run's form, 2 / 1 / 0)
                from opentpu.llm import moe as MO
                MO.EM_PIPE = ep
                for key in [k for k in img.__dict__.get("_pf500_runs", {}) if k[0] == "x"]:
                    del img._pf500_runs[key]
            for kind in kinds:
                rec = {"mode": a.cmd, "target": tname, "S": tg.cfg.S, "path": a.path,
                       "R": R, "kind": kind, "pos": a.pos, "route": a.route}
                if ep is not None:
                    rec["em_pipe"] = ep
                try:
                    t0 = time.time()
                    runs = compile_runs(img, kind, R, a.pos, a.path, getattr(a, "T", 16),
                                        getattr(a, "C", None))
                    rec["compile_s"] = round(time.time() - t0, 1)
                    if em:
                        rec["em_pass"] = list(getattr(img, "em_pass", None) or ())
                        rec["em_program"] = len(runs[-1][1][0])
                except Exception as e:      # TMEM / IMEM / rows: the fit answer
                    rec["error"] = f"{type(e).__name__}: {e}"[:300]
                    print(json.dumps(rec), flush=True)
                    rows.append(rec)
                    arm_row(a, rec)
                    continue
                words = live = None
                if a.route != "zero":
                    ids = route_ids(R, E, k, a.seed)
                    words, live = route_words(img, L.layer[kind], ids, em)
                    rec["experts"] = int(np.unique(ids).size)
                keep: list = []
                drams: list = []
                rs = []
                for j, (lab, progs, args) in enumerate(runs):
                    rs.append(run_isa(tg, progs, args, words if j == 0 else None, live, keep,
                                      drams) | {"label": lab})
                r = merge_runs(rs)
                r.pop("label", None)
                b, u = bound_of(r["total"])
                rec.update(r)
                rec |= {"bound": b, "binding": u,
                        "serial": int(sum(r["total"][x] for x in UNITS)),
                        "dram_kinds": kinds_of(r["phases"])}
                if sb is not None:
                    rec["isa"] = {x: rec.pop(x) for x in ("phases", "total", "dram_kinds")}
                    if em:
                        rec |= run_sim_em(tg, sb, img, runs, words, keep, R, a.dump and
                                          f"{a.dump}/{tname.replace(':', '_')}-R{R}-{kind}"
                                          + ("" if ep is None else f"-p{ep}"))
                    else:
                        keep[0].R_rows = R
                        rec |= run_sim(tg, sb, img, progs, args, words, live, keep[0])
                per[kind] = rec
                print(json.dumps(rec), flush=True)
                rows.append(rec)
                arm_row(a, rec)
            if set(per) == set(L.counts):
                m = model_row(tg, R, per, a, L.counts)
                print(json.dumps(m), flush=True)
                rows.append(m)
    return rows


def model_row(tg: Tgt, R: int, per: dict, a, counts: dict | None = None) -> dict:
    """The model's cycles a token from its layers' runs: each kind's layers (counts, the layout's;
    the 35B's 30 dn + 10 attn) / R, at the bound and with the DRAM at the decode's share
    (bound88: the bound with dram88 for DRAM)."""
    counts = counts or MODEL_LAYERS

    def cyc(f):
        return sum(counts[k] * f(per[k]) for k in counts) / R

    def b88(r):
        return max(max(r["total"][u] for u in UNITS if u != "DRAM"), r["total"]["dram88"])

    if a.cmd == "sim":
        c = cyc(lambda r: r["cycles"])
        busy = {u: round(100 * cyc(lambda r, u=u: r["busy"][u]) / c, 1)
                for u in ("DMA", "MXU", "VPU", "DSTEP")}
        return {"mode": a.cmd, "target": tg.name, "S": tg.cfg.S, "path": a.path, "R": R,
                "kind": "model", "pos": a.pos, "route": a.route,
                "ok": all(per[k].get("ok") for k in counts),
                "Mcyc/tok": round(c / 1e6, 3), "tok/s": round(MHZ * 1e6 / c, 1),
                "isa bound tok/s": round(MHZ * 1e6 / cyc(lambda r: r["bound"]), 1),
                "busy%": busy}
    c, c88 = cyc(lambda r: r["bound"]), cyc(b88)
    busy = {u: round(100 * cyc(lambda r, u=u: r["total"][u]) / c88, 1) for u in UNITS}
    return {"mode": a.cmd, "target": tg.name, "S": tg.cfg.S, "path": a.path, "R": R,
            "kind": "model", "pos": a.pos, "route": a.route,
            "Mcyc/tok": round(c / 1e6, 3), "tok/s": round(MHZ * 1e6 / c, 1),
            "Mcyc/tok88": round(c88 / 1e6, 3), "tok/s88": round(MHZ * 1e6 / c88, 1),
            "busy%88": busy}


def table(rows: list) -> str:
    out = ["| target | R | kind | instr | MXU | VPU | SE | Q | DRAM | COLL | bound | MB |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("error"):
            out.append(f"| {r['target']} | {r['R']} | {r['kind']} | not compiled: "
                       f"{r['error'][:60]} |||||||||")
        elif r["kind"] == "model" and r["mode"] == "sim":
            out.append(f"| {r['target']} | {r['R']} | model (sim) | | busy % {r['busy%']} "
                       f"|||||| {r['Mcyc/tok']} M/tok = {r['tok/s']} tok/s (ISA bound "
                       f"{r['isa bound tok/s']}) ok {r['ok']} | |")
        elif r["kind"] == "model":
            out.append(f"| {r['target']} | {r['R']} | model | | | | | | | | "
                       f"{r['Mcyc/tok']} M/tok = {r['tok/s']} tok/s "
                       f"(DRAM 88 %: {r['tok/s88']}) | |")
        elif r["mode"] == "sim":
            t = r["isa"]["total"]
            out.append(f"| {r['target']} | {r['R']} | {r['kind']} (sim) | {r['instructions']} | "
                       f"busy % {r['busy%']} |||||| {r['cycles']:,} cycles (ISA bound "
                       f"{r['bound']:,} {r['binding']}) ok {r['ok']} | {t['bytes'] / 1e6:.1f} |")
        else:
            t = r["total"]
            out.append(f"| {r['target']} | {r['R']} | {r['kind']} | {r['instructions']} | "
                       + " | ".join(f"{t[u]:,}" for u in UNITS)
                       + f" | {r['bound']:,} {r['binding']} | {t['bytes'] / 1e6:.1f} |")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=("isa", "sim"))
    p.add_argument("--target", default=",".join(TARGETS))
    p.add_argument("--R", default="2")
    p.add_argument("--kinds", default=None, help="default: the layout's kinds")
    p.add_argument("--pos", type=int, default=1024)
    p.add_argument("--path", default="lm", choices=("lm", "em"))
    p.add_argument("--T", type=int, default=16, help="em: a tile's rows (<= ACT_ROWS)")
    p.add_argument("--C", type=int, default=None,
                   help="em: an expert pass's most entries (default: the image's, moe.em_exact / "
                        "em_pass_order)")
    p.add_argument("--gscale", action="store_true",
                   help="em: moe.EM_GSCALE (the router weights as the down MMs' row scale)")
    p.add_argument("--route", default="uniform", choices=("uniform", "zero"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--json", default=None)
    p.add_argument("--plus", default=K7, help="sim: the board model's plusargs")
    p.add_argument("--params", default="", help="sim: its build parameters, K=V,...")
    p.add_argument("--trace-depth", type=int, default=65536)
    p.add_argument("--dump", default=None,
                   help="sim em: a directory for each run label's first trace (dump_recs)")
    p.add_argument("--em-pipe", default="2",
                   help="em: moe.EM_PIPE per arm, e.g. 2,1,0 (each point in each, one board build)")
    a = p.parse_args(argv)
    if a.gscale:
        from opentpu.llm import moe as MO
        MO.EM_GSCALE = True
    rows = cmd_isa(a)
    for r in rows:
        r["gscale"] = a.gscale
    print(table(rows))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"peak RSS {rss / (2**30 if sys.platform == 'darwin' else 2**20):.2f} GiB",
          file=sys.stderr)
    if a.json:
        Path(a.json).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if any(r.get("ok") is False for r in rows):     # a point's check failed: scripts see it
        sys.exit(1)


if __name__ == "__main__":
    main()
