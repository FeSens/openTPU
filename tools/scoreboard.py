"""The workload scoreboard: what a hardware configuration gives on the four workloads the user
weighs equally (2026-10-11): single-user decode, prefill, multi-user batched decode, MoE.

    python3 tools/scoreboard.py [report] [--board ypcb|bcu1525] [what-if options]
    python3 tools/scoreboard.py compare --mcols 8            # a what-if against its board
    python3 tools/scoreboard.py whatifs [--board B]           # the standard what-if set
    python3 tools/scoreboard.py validate                      # the model against card / co-sim
    python3 tools/scoreboard.py boards | models | fit

What-if options (each overrides the board preset):
  --mcols N        MXU columns: MCOLS x D MACs a cycle, MCOLS rows a pass (int8; MCOLS / 2 PAIRed
                   4-bit)
  --act-rows N     rows a weight pass (MM replay, "virtual MCOLS"): a run of R rows streams the
                   weights ceil(R / N) times; the MXU still takes ceil(R / MCOLS) cycles a block
  --mxu-pump X     MACs a cycle x X at the same rows a pass (a double-pumped MXU)
  --tmem-words N   TMEM words: the prefill rows a run (tools/scoreboard_rmax.json, the
                   compiler's fit)
  --rows N         prefill rows a run, forced (tiling); --imem 0: no IMEM limit (looped
                   attention), else instructions
  --batch-rows N   rows of one batched-decode run (default: rows a pass)
  --vpu X          vector unit and DSTEP throughput x X
  -S N             slices (tensor-parallel: weights, KV heads and experts split)
  --mhz F          core clock
  --mem ddr3|ddr4|hbm --mts N --channels N --mem-eff E --gbps G (a slice's memory GB/s)
  --port-bytes N   core port bytes a cycle a slice (128: two 64-byte channel beats)
  --pcie-gbps G    host link for offloaded experts; --fast-gb G memory a slice holds
  --moe-rows all|gather   a MoE layer run's union experts on all R rows, or each on its own
  --wformat fp4|int8|auto  --ctx N  --models a,b  --json FILE  --markdown

Python (the other experiments):

    sys.path.insert(0, "tools"); import scoreboard as SB
    hw = SB.board("ypcb", mcols=8, tmem_words=262144)
    sc = SB.scoreboard(hw)            # {"decode": {...}, "prefill", "batch", "moe"}
    SB.compare(SB.board("ypcb"), hw)  # ratio a workload and their geometric mean ("score")

The model (docs/scoreboard.md): per layer and run of R rows, max(DRAM bytes / bandwidth, MXU
cycles) + the DeltaNet recurrence + a calibrated overhead a row; the bytes and MACs from the
real layer shapes (tools/scoreboard_models.json: each checkpoint's config.json and safetensors
headers), the MXU's row groups, the KV cache read in whole 256-token blocks, DSTEP at 8 state
words a cycle; the constants fitted to the YPCB card's counters, checked against the RTL
co-simulation (tools/scoreboard_cosim.py). Pure Python, no numpy.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_FILE = HERE / "scoreboard_models.json"
RMAX_FILE = HERE / "scoreboard_rmax.json"
COSIM_RUNS_FILE = HERE / "scoreboard_cosim_runs.jsonl"   # build/sbcosim.sh's runs (the simulation host)
D = 128                         # the MXU's depth, one weight block; the port's chunk is D bytes
BLOCK = 256                     # the attention block: decode and prompt runs read whole blocks
GB = 1e9
MB = 1e6

# ------------------------------------------------------------------------------- the models
NAMES = {"qwen3": "Qwen3-0.6B", "lfm2": "LFM2.5-230M", "qwen35": "Qwen3.5-0.8B",
         "qwen35-2b": "Qwen3.5-2B", "qwen35-4b": "Qwen3.5-4B", "smollm3": "SmolLM3-3B",
         "phi4-mini": "Phi-4-mini-instruct", "lfm2-2.6b": "LFM2-2.6B",
         "gemma4": "gemma-4-E2B-it", "gemma4-e4b": "gemma-4-E4B-it",
         "lfm2-8b": "LFM2.5-8B-A1B", "qwen35-35b": "Qwen3.5-35B-A3B",
         "gemma4-26b": "gemma-4-26B-A4B-it"}
SHORT = {v: k for k, v in NAMES.items()}
DENSE = ["qwen3", "lfm2", "qwen35", "qwen35-2b", "qwen35-4b", "smollm3", "phi4-mini",
         "lfm2-2.6b", "gemma4"]
BATCHED = ["qwen3", "smollm3", "phi4-mini"]     # batched-decode kernels today: Qwen3, Llama-likes
MOE = ["gemma4-26b", "qwen35-35b"]
MIXES = {"phi4-mini": "mlp@4-29=fp4", "qwen35-4b": "delta=fp4,mlp=fp4",
         "smollm3": "gateup@9-35=fp4", "lfm2-2.6b": "conv=fp4,mlp=fp4"}   # formats.MIXES
MOE_FORMATS = {"gemma4-26b": ("int8", "fp4", "experts=fp4"),            # layers, head, formats
               "qwen35-35b": ("fp4", "int8", ""), "lfm2-8b": ("fp4", "int8", "")}
BITS = {"int8": 8.25, "fp4": 4.25, "int4": 4.25}   # storage bits a weight, block scales included

_MODELS: dict | None = None


def models() -> dict:
    global _MODELS
    if _MODELS is None:
        _MODELS = json.loads(MODELS_FILE.read_text())
    return _MODELS


def model(name: str) -> dict:
    key = NAMES.get(name, name)
    if key not in models():
        raise KeyError(f"unknown model {name!r}: {', '.join(NAMES)}")
    return models()[key]


def short(m: dict) -> str:
    return SHORT.get(m["name"], m["name"])


def layers(m: dict):
    """The model's layers in order: (index, layer)."""
    i = 0
    for n, lay in m["runs"]:
        for _ in range(n):
            yield i, lay
            i += 1


def _pad(n: float, m: int = D) -> int:
    return -(-int(n) // m) * m


SKIP = re.compile(r"(^|\.)(visual|vision_tower|vision_model|audio_tower|audio_model|"
                  r"embed_vision|embed_audio|mtp)(\.|_)")


def shapes(name: str, cfg: dict, tensors: dict) -> dict:
    """A model's shapes from its config.json and safetensors headers ({name: [dtype, shape]}):
    per layer its kind, the parameters of each weight group (attn, gateup, down, delta, conv,
    router, shared, ple; padded to D, Gemma 4's FFN widths to 2D as its image pads them), its
    attention / DeltaNet / MoE dimensions; the LM head and Gemma 4's PLE projection."""
    t = cfg.get("text_config", cfg)
    mt = t.get("model_type")
    fam = {"qwen3": "qwen3", "lfm2": "lfm2", "lfm2_moe": "lfm2", "qwen3_5_text": "qwen35",
           "qwen3_5_moe_text": "qwen35", "gemma4_text": "gemma4", "phi3": "llama",
           "smollm3": "llama", "llama": "llama"}[mt]
    H, L, V = t["hidden_size"], t["num_hidden_layers"], t["vocab_size"]
    kinds = t.get("layer_types") or ["full_attention"] * L
    ffn_pad = 2 * D if fam == "gemma4" else D
    n_shared = t.get("num_kv_shared_layers") or 0
    per, head, embed, ple = {}, 0, 0, {}
    for k, (_, sh) in tensors.items():
        if SKIP.search(k):
            continue
        mm = re.search(r"layers\.(\d+)\.(.*)$", k)
        if not mm:
            if k.endswith("lm_head.weight"):
                head = _pad(sh[0]) * _pad(sh[1])
            elif k.endswith("embed_tokens.weight"):
                embed = sh[0] * sh[1]
            elif k.endswith("embed_tokens_per_layer.weight"):
                ple["table"] = sh[0] * sh[1]
            elif k.endswith("per_layer_model_projection.weight"):
                ple["proj"] = _pad(sh[0]) * _pad(sh[1])
            continue
        per.setdefault(int(mm.group(1)), {})[mm.group(2)] = sh
    head = head or _pad(V) * _pad(H)               # tied: the head streams the embedding
    out = []
    for i in range(L):
        kind = {"full_attention": "attn", "sliding_attention": "sliding",
                "linear_attention": "delta", "conv": "conv"}[kinds[i]]
        shared = fam == "gemma4" and i >= L - n_shared
        g = {}
        for nm, sh in per[i].items():
            if len(sh) != 2 or "conv1d" in nm or re.search(r"experts\.\d+\.", nm):
                continue
            if shared and re.search(r"self_attn\.(k|v)_proj", nm):
                continue                          # a KV-shared layer reads an earlier one's
            n, kd = sh
            if "shared" not in nm and "router" not in nm and \
                    re.search(r"(gate_proj|up_proj|\.w1\.|\.w3\.|gate_up_proj)", nm):
                n = _pad(n, ffn_pad)
            if "shared" not in nm and re.search(r"(down_proj|\.w2\.)", nm):
                kd = _pad(kd, ffn_pad)
            grp = ("attn" if "self_attn" in nm else "delta" if "linear_attn" in nm else
                   "conv" if re.search(r"conv\.(in|out)_proj", nm) else
                   "shared" if "shared_expert" in nm else
                   "router" if re.search(r"(router\.proj|mlp\.gate\.weight|"
                                         r"feed_forward\.gate\.weight)", nm) else
                   "gateup" if re.search(r"(gate_proj|up_proj|\.w1\.|\.w3\.|gate_up)", nm)
                   else "down" if re.search(r"(down_proj|\.w2\.)", nm) else
                   "ple" if "per_layer" in nm else "other")
            g[grp] = g.get(grp, 0) + _pad(n) * _pad(kd)
        lay = {"kind": kind, "groups": dict(sorted(g.items()))}
        if kind in ("attn", "sliding"):
            if fam == "gemma4" and kinds[i] == "full_attention":
                lay["kind"] = "global"
                d = t.get("global_head_dim") or t["head_dim"]
                kvh = t.get("num_global_key_value_heads") or t["num_key_value_heads"]
            else:
                d = t.get("head_dim") or H // t["num_attention_heads"]
                kvh = t["num_key_value_heads"]
            lay.update(q_heads=t["num_attention_heads"], kv_heads=kvh, head_dim=d,
                       window=(t.get("sliding_window") or 0) if kind == "sliding" else 0,
                       kv_shared=shared)
        elif kind == "delta":
            lay["deltanet"] = dict(v_heads=t["linear_num_value_heads"],
                                   k_heads=t["linear_num_key_heads"],
                                   dk=t["linear_key_head_dim"], dv=t["linear_value_head_dim"],
                                   conv_k=t.get("linear_conv_kernel_dim", 4))
        if any("experts" in nm for nm in per[i]):
            E = t.get("num_experts") or t.get("num_local_experts")
            F = t.get("moe_intermediate_size") or t.get("expert_intermediate_size")
            lay["moe"] = dict(experts=E, top_k=t.get("num_experts_per_tok") or
                              t.get("top_k_experts"),
                              expert_params=3 * _pad(F, ffn_pad) * _pad(H))
        out.append(lay)
    runs = []
    for lay in out:
        if runs and runs[-1][1] == lay:
            runs[-1][0] += 1
        else:
            runs.append([1, lay])
    res = dict(name=name, family=fam, model_type=mt, hidden=H, vocab=V, n_layers=L,
               head=head, embed=embed, untied=bool(head != _pad(V) * _pad(H) or
                                                   not t.get("tie_word_embeddings", True)),
               runs=runs)
    if ple.get("proj"):
        res["ple"] = dict(proj=ple["proj"], table=ple.get("table", 0),
                          row=t["hidden_size_per_layer_input"] * L)
    return res


def safetensors_headers(d) -> dict:
    import struct
    out = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        out.update({k: [v["dtype"], v["shape"]] for k, v in h.items()})
    return out


# ------------------------------------------------------------------------------- formats
def formats_of(name: str, wformat: str) -> tuple:
    """(layer format, head format, formats string) of a weight choice: fp4 (int8 head), int8,
    mix / auto (the recommended mix, else int8: otpu-chat's default); a MoE model takes its
    production formats."""
    if name in MOE_FORMATS:
        return MOE_FORMATS[name]
    if wformat == "fp4":
        return "fp4", "int8", ""
    if wformat in ("mix", "auto"):
        return "int8", "int8", MIXES.get(name, "")
    return wformat, wformat, ""


def _rules(formats: str):
    out = []
    for item in (formats or "").replace(" ", "").split(","):
        if item:
            key, _, fmt = item.partition("=")
            kind, _, span = key.partition("@")
            lo, dash, hi = span.partition("-")
            out.append((kind, int(lo) if lo else None,
                        (int(hi) if dash else int(lo)) if lo else None, fmt))
    return out


def group_format(group: str, i: int, base: str, formats: str) -> str:
    """A weight group's format in layer i: a ranged rule over an unranged one, the first of
    equals ("mlp" is gateup and down; "experts" the routed experts; "head" unranged)."""
    best = None
    for kind, lo, hi, fmt in _rules(formats):
        if kind == group or (kind == "mlp" and group in ("gateup", "down")):
            if lo is None:
                best = best or (1, fmt)
            elif lo <= i <= hi and (best is None or best[0] == 1):
                best = (0, fmt)
    return best[1] if best else base


# ------------------------------------------------------------------------------- hardware
@dataclasses.dataclass(frozen=True)
class HW:
    """A hardware configuration. A slice's port moves port_bytes a core cycle (two 64-byte
    channel beats on the YPCB); its memory delivers channels x mts x 8 bytes a second (or
    gbps) at mem_eff while streaming; the slice streams at the lower of the two."""
    name: str = "ypcb"
    S: int = 1                      # slices (tensor-parallel: weights, KV heads, experts split)
    mcols: int = 4                  # MXU columns: MCOLS x D MACs a cycle, MCOLS rows a pass
    act_rows: int = 0               # rows a weight pass (MM replay; 0: mcols)
    mxu_pump: float = 1.0           # MACs a cycle x pump at the same rows a pass
    pair: bool = True               # 4-bit MMs of <= MCOLS / 2 rows take two blocks a cycle
    tmem_words: int = 65536
    imem: int = 4096                # instructions a program (0: no limit, looped attention)
    vpu: float = 1.0                # vector unit and DSTEP throughput x vpu
    mhz: float = 133.33
    mem: str = "ddr3"
    mts: float = 1066.67
    channels: int = 2               # memory channels a slice
    mem_eff: float = 0.97           # data share while streaming (the card: 8.04 K cycles a MB)
    port_bytes: int = 128           # core port bytes a cycle a slice
    port_eff: float = 1.0            # the co-sim at an ideal memory: decode 4% faster than LiteDRAM
    gbps: float = 0.0               # a slice's peak memory GB/s (0: channels x mts x 8)
    fast_gb: float = 4.0            # GiB a slice holds for the model (32-bit addresses: 4)
    far_gbps: float = 0.0           # a second, slower memory tier on the card
    far_gb: float = 0.0
    pcie_gbps: float = 2.2          # host -> card for offloaded experts (Gen2 x8, a miss's rate)
    rows: int = 0                   # prefill rows a run (0: the TMEM / IMEM fit)
    batch_rows: int = 0             # rows of a batched decode run (0: rows a pass)
    moe_rows: str = "all"           # a layer run's union experts on all R rows, or "gather"
    moe_ids: int = 16               # expert ids a MoE layer run's request holds (R k <= 16)
    kernel: str = "today"           # prompt runs: today's compile_rows, or "tiled" (the tile
                                    # kernel: packed attention, overlapped projections)
    host_run_ms: float = 0.6        # the host's time a run (prompt runs, batched steps)
    note: str = ""

    def replace(self, **kw) -> "HW":
        return dataclasses.replace(self, **kw)

    @property
    def rows_pass(self) -> int:
        return self.act_rows or self.mcols

    def mem_gbps(self) -> float:
        """A slice's memory bandwidth while streaming, GB/s."""
        return (self.gbps or self.channels * self.mts * 8 / 1e3) * self.mem_eff

    def bw(self) -> float:
        """A slice's streaming bandwidth in bytes a core cycle: the port or the memory."""
        return min(self.port_bytes * self.port_eff, self.mem_gbps() * GB / (self.mhz * 1e6))

    def bound(self) -> str:
        return "port" if self.port_bytes * self.port_eff <= self.mem_gbps() * GB / (
            self.mhz * 1e6) else "memory"

    def describe(self) -> str:
        mem = (f"{self.mem.upper()} {self.gbps:g} GB/s" if self.gbps else
               f"{self.mem.upper()}-{self.mts:g} x {self.channels}")
        return (f"{self.name}: S {self.S}, MCOLS {self.mcols} (rows a pass {self.rows_pass}, "
                f"MXU x{self.mxu_pump:g}), TMEM {self.tmem_words // 1024}K words, VPU "
                f"x{self.vpu:g}, {self.mhz:g} MHz, {mem} a slice: {self.mem_gbps():.1f} "
                f"GB/s, {self.bw():.1f} B a "
                f"cycle ({self.bound()}-bound), {self.S * self.bw() * self.mhz / 1e3:.1f} "
                f"GB/s in all; host link {self.pcie_gbps:g} GB/s")


BOARDS = {
    "ypcb": HW(note="YPCB-00338 production: xc7k480t, 2 x DDR3-1066 (LiteDRAM), 133.33 MHz, "
                    "MCOLS 4, TMEM 64K words, PCIe Gen2 x8"),
    "bcu1525": HW(name="bcu1525", S=2, mhz=100.0, mem="ddr4", mts=1600, mem_eff=0.95,
                  pcie_gbps=9.0,
                  note="BCU1525: VU9P, S = 2 slices on 2 DDR4 channels each (4 GiB a slice "
                       "addressed of 2 x 16 GB), PCIe Gen3 x16; bring-up: DDR4-1600, 100 MHz"),
}


def board(b: str = "ypcb", **kw) -> HW:
    """A board preset with what-if overrides: board("bcu1525", mhz=200, mts=2400)."""
    if b not in BOARDS:
        raise KeyError(f"unknown board {b!r}: {', '.join(BOARDS)}")
    return BOARDS[b].replace(**kw) if kw else BOARDS[b]


# ------------------------------------------------------------------------------- calibration
# Fitted on the YPCB card (docs/scoreboard.md; `validate` shows the agreement, `fit` refits
# kappa):
# - a layer's overhead outside the bandwidth and MXU bounds (the vector unit's work and the
#   dependency chains between a layer's phases), in units of U = ROW_CYCLES x hidden / 1024:
#   kappa_pass a weight pass + kappa_row a row of the pass (rows of a further pass overlap
#   the stream of the next: h of a row), (kappa_pass, kappa_row) by model, least squares over
#   its card decode, prefill and batched points (else the family's mean);
# - vpu_share: the share of it that scales with the vector unit (the RTL at 16 lanes);
# - a DeltaNet layer's recurrence: DSTEP at 8 state words a cycle plus a fill a head; one row's
#   hides behind the layer's weight stream, every further row's adds (the RTL's Qwen3.5-0.8B
#   prefill at R = 4: 1.95 M of 6.82 M cycles are DSTEP's). With it run the DeltaNet's own
#   VOPs (head_in and the conv's sigmoid at 0.875 cycles a conv channel each, the gate's 112 a
#   v-head, norms and SiLU ~1.3 a v-head's value: vpu-pump's inventory, 15.7 K a row on the
#   0.8B and 2B, 22.7 K on the 4B), dn_vop of a further row's exposed; dn_slack of the gap
#   between the stream and the MXU hides part of them and of the DSTEPs (the campaign's
#   DeltaNet layers at MCOLS 4 and 8, fitted with the kappas: the 2B's MXU-bound layer at
#   MCOLS 4 takes 84 K over the MXU for 108 K of DSTEP, the 0.8B's stream-bound one at
#   MCOLS 8 166 K over the stream);
# - attention streams a row's KV once a group of MCOLS query heads of a KV head (the code's
#   parts: gemma4._attention, qwen3._attend_heads);
# - run: cycles a device run; head_row: a logits row's sink (argmax / sampling);
# - coll: S > 1, cycles a layer and row for the all-gathers (the co-sim at S 2, Qwen3 at the
#   port's rate: the layer halves, plus ~0.4 K at decode and ~1.5 K a row at R 4);
# - moe_layer / moe_expert: a MoE layer's routing a row, an expert's start (the card's
#   zero-miss layer runs); link_lead_ms: the host's time to see a decode layer's misses and
#   start their DMA; a layer's link time past what the card hides (the dense MLP / shared
#   expert and the hit experts) is exposed; prefill_exposed: the share of a prompt's
#   offloaded experts' link time the card does not hide; moe_run_ms: the host's time a
#   layer run (the card's 35B and 26B); miss_scale: the card's misses against the router
#   traces' cache model.
ROW_CYCLES = 7000.0
CAL = {
    "kappa": {},                    # {model: (kappa_pass, kappa_row)}: KAPPA below
    "kappa_family": {},             # (the family's mean; set on first use)
    "h": 0.14,                      # a row past the first MCOLS: its share of a row overhead
    "over_form": "run",             # the overhead: "run" (kappa_pass a run, kappa_row a row)
                                    # or "pass" (kappa_pass a weight pass, kappa_row the first
                                    # MCOLS rows, h the rest's share)
    "vpu_share": 0.8,
    "run": 5000.0,
    "head_row": 2000.0,
    "dstep_fill": 200.0,
    "dstep_words": 8.0,
    "dn_vop": 0.4,                  # a DeltaNet row's VOPs after the first: the share exposed
    "dn_slack": 0.24,               # ... less this share of |MXU - stream| (the campaign's
                                    # 6 layers and the card, fitted with the kappas)
    "dn_hide": 0.5,                 # ... which hides at most this share of them
    "moe_alpha": 0.6,               # a run's union of R rows' experts: k R^alpha (consecutive
                                    # tokens share experts), at most the prompt's curve
    "att_serial": True,             # attention between the matmuls, not under their stream
    "att_head": 0.0,                # cycles a row's entry a KV head (and q-group pass) ...
    "att_blk": 0.0,                 # ... and a 256-token block of it (0: the KV stream binds)
    "tile_layer": 0.0,              # the tile kernel: cycles a layer and run outside the MXU
    "tile_row": 4874.0,             # ... and a row (x hidden / 1024)
    "prompt_run": 1.0,              # a prompt run's layers over the model's (the card: 1.0)
    "tile_card": 1.07,              # the tile kernel's prompt runs over its co-sim
    "coll": 1000.0,
    "moe_layer": 6000.0,
    "moe_expert": 600.0,
    "link_lead_ms": 0.15,           # the host's lead a request (it sees it, starts the DMA)
    "moe_run_ms": 0.2,              # the host's time a layer run (start, HALTED, between)
    "prefill_exposed": 0.85,
    "miss_scale": 1.2,
}


KAPPA = {     # `scoreboard.py fit` (2026-10-12: serial attention, a run + a row, DeltaNet VOPs)
    "gemma4": (3.7091, 0.898), "lfm2": (0.0286, 0.2592), "lfm2-2.6b": (0.4248, 0.3043),
    "phi4-mini": (0.5077, 0.3273), "qwen3": (0.2666, 0.7267), "qwen35": (0.0, 1.1987),
    "qwen35-2b": (0.9086, 0.0), "qwen35-4b": (0.588, 0.0), "smollm3": (0.4921, 0.6348)}
CAL["kappa"].update(KAPPA)


def _family_kappa():
    fam = {}
    for n, k in CAL["kappa"].items():
        fam.setdefault(model(n)["family"], []).append(k)
    CAL["kappa_family"] = {f: tuple(sum(x[j] for x in v) / len(v) for j in (0, 1))
                           for f, v in fam.items()}


# ------------------------------------------------------------------------------- the time model
def att_tokens(lay: dict, ctx: int) -> int:
    """Positions a row at context ctx (its own included) reads from the KV cache: whole
    256-token blocks (the decode programs' buckets, the prompt runs' masked block); a sliding
    layer at most its window and a block."""
    n = _pad(ctx + 1, BLOCK)
    if lay.get("window"):
        n = min(n, _pad(lay["window"], BLOCK) + BLOCK)
    return n


def kv_token_bytes(d: int) -> float:
    """KV bytes a token and KV head: K int8 + an fp32 scale a D block, V int8 + its scale."""
    d = _pad(d)
    return 2 * d + 4 * (d // D) + 4


def mxu_cycles(params: float, fmt: str, R: int, hw: HW) -> float:
    """Cycles the MXU takes for `params` weights on R rows: int8 one block (D weights) a cycle
    a group of MCOLS rows; 4-bit (PAIR) two blocks a cycle a group of MCOLS / 2 rows."""
    if fmt == "int8" or not hw.pair:
        c = params / D * math.ceil(R / hw.mcols)
    else:
        c = params / (2 * D) * math.ceil(2 * R / hw.mcols)
    return c / hw.mxu_pump


def kappa(m: dict) -> tuple:
    """(kappa_pass, kappa_row) of a model."""
    if not CAL["kappa_family"] and CAL["kappa"]:
        _family_kappa()
    return CAL["kappa"].get(short(m), CAL["kappa_family"].get(m["family"], (0.3, 0.7)))


def over_cycles(m: dict, hw: HW, R: int) -> float:
    """A layer's overhead for a run of R rows at the configuration's vector unit.
    over_form "run" (the default): kappa_pass a run, kappa_row every row (tiling's co-sim:
    compile_rows' cycles a row are flat in R and MCOLS, the card's prompt runs 4-11% above);
    the vector unit's share is vpu_share of the rows after the first (vpu-pump's co-sim:
    every VOP's work halved takes 0.35% of decode and 2.7-4.3% of Qwen3's runs of 4).
    over_form "pass": kappa_pass a weight pass, kappa_row a row of the first MCOLS, h the
    rest's share."""
    kp, kr = kappa(m)
    s = CAL["vpu_share"]
    U = ROW_CYCLES * m["hidden"] / 1024
    if CAL["over_form"] == "run":
        return (kp + kr) * U + kr * (R - 1) * U * (s / hw.vpu + 1 - s)
    p = hw.rows_pass
    reff = min(R, hw.mcols) + CAL["h"] * max(0, R - hw.mcols)
    return (kp * math.ceil(R / p) + kr * reff) * U * (s / hw.vpu + 1 - s)


def dstep_cycles(lay: dict, hw: HW) -> float:
    dn = lay["deltanet"]
    return dn["dk"] * dn["dv"] / (CAL["dstep_words"] * hw.vpu) + CAL["dstep_fill"]


def dn_vop_cycles(lay: dict, hw: HW) -> float:
    """A DeltaNet row's own VOPs (not DSTEP), issue cycles at the configuration's vector
    unit: vpu-pump's static inventory of Qwen3.5 (vpu_hack.inventory)."""
    dn = lay["deltanet"]
    conv = 2 * dn["k_heads"] * dn["dk"] + dn["v_heads"] * dn["dv"]
    return (1.75 * conv + 112 * dn["v_heads"] + 1.3 * dn["v_heads"] * dn["dv"]) / hw.vpu


def layer_cycles(m: dict, i: int, lay: dict, hw: HW, fmt: tuple, ctxs: list,
                 experts: float | None = None, split: bool = False, prompt: bool = False):
    """Cycles of layer i (on each slice) for a run of len(ctxs) rows, row r at context
    ctxs[r]. experts: the distinct routed experts the run streams (MoE; default the rows'
    expected union). split: (cycles, {bytes, mem, mxu, rec, over})."""
    base, _, formats = fmt
    R, S = len(ctxs), hw.S
    wbytes = mxu = 0.0
    for g, p in lay["groups"].items():
        f = group_format(g, i, base, formats)
        wbytes += p * BITS[f] / 8
        mxu += mxu_cycles(p, f, R, hw)
    byts = math.ceil(R / hw.rows_pass) * wbytes
    over = over_cycles(m, hw, R) / S          # the vector work splits with the heads (co-sim S 2)
    rec = att = 0.0
    tiled = hw.kernel == "tiled" and R > 1 and ctxs[-1] - ctxs[0] == R - 1   # a prompt run
    if lay["kind"] in ("attn", "sliding", "global"):
        kvb = kv_token_bytes(lay["head_dim"])
        G = max(1, lay["q_heads"] // lay["kv_heads"])
        passes = math.ceil(G / hw.mcols)
        kv = lat = 0.0
        if tiled:
            # the tile kernel (the tiling agent's): MCOLS / G rows share an entry a KV head
            # (packed attention), its K and V products on the MXU, the projections of the next
            # head streaming meanwhile: the layer is bound by the MXU with attention's MMs added
            pack = max(1, hw.mcols // G)
            for g0 in range(0, R, pack):
                n = att_tokens(lay, ctxs[min(R, g0 + pack) - 1])
                kv += lay["kv_heads"] * kvb * n * passes
                mxu += lay["kv_heads"] * passes * 2 * n * _pad(lay["head_dim"]) / D / \
                    hw.mxu_pump
            byts += kv + (0 if lay.get("kv_shared") else R * lay["kv_heads"] * kvb)
        else:
            for c in ctxs:
                n = att_tokens(lay, c)
                kv += lay["kv_heads"] * kvb * n * passes
                byts += 0 if lay.get("kv_shared") else lay["kv_heads"] * kvb  # the row's K, V
                lat += lay["kv_heads"] * passes * (CAL["att_head"] + CAL["att_blk"] * n / BLOCK)
            if CAL["att_serial"]:
                # attention runs between the layer's matmuls (its q, k, v before, o after): a
                # row's entry a KV head streams the head's blocks (the card's prompt runs and
                # batched steps: the KV stream binds it, att_blk 0) and waits on its scores,
                # softmax and V product
                s_ = CAL["vpu_share"]
                att = max(kv / S / hw.bw(), lat / S * (s_ / hw.vpu + 1 - s_))
            else:
                byts += kv
    elif lay["kind"] == "delta":
        dn = lay["deltanet"]
        # the recurrence: the first row's DSTEPs hide behind the layer's weight stream (their
        # state traffic shares the port with it); every further row's run after, bound by
        # DSTEP's datapath, which covers their own state traffic
        byts += 2 * dn["v_heads"] * dn["dk"] * dn["dv"] * 4
        rec = (R - 1) * dn["v_heads"] / S * dstep_cycles(lay, hw)
    elif lay["kind"] == "conv":
        byts += R * 2 * 3 * m["hidden"] * 4                             # the conv window
    if "moe" in lay:
        mo = lay["moe"]
        f = group_format("experts", i, base, formats)
        if experts is None:
            experts = union(mo["experts"], mo["top_k"], R, MOE_EEFF.get(short(m)))
        byts += experts * mo["expert_params"] * BITS[f] / 8
        rows_e = R if hw.moe_rows == "all" else max(1, round(R * mo["top_k"] / experts))
        mxu += experts * mxu_cycles(mo["expert_params"], f, rows_e, hw)
        over += R * CAL["moe_layer"] + experts * CAL["moe_expert"]
    if S > 1:
        over += R * CAL["coll"]
    mem = byts / S / hw.bw()
    if lay["kind"] == "delta" and R > 1:
        # the further rows' DeltaNet VOPs run with their DSTEPs; the idle side of the stream
        # and the MXU hides part of both (the campaign's layers at MCOLS 4 and 8), at most
        # dn_hide of them (the campaign's R <= 4 hid at most 37%; a long run on a deep ACT
        # RAM, one weight pass for its MXU's many, is not measured)
        rec += CAL["dn_vop"] * (R - 1) * dn_vop_cycles(lay, hw) / S
        rec -= min(CAL["dn_hide"] * rec, CAL["dn_slack"] * abs(mxu / S - mem))
    if tiled and lay["kind"] in ("attn", "sliding", "global"):
        # what the tile kernel leaves outside the MXU: a layer's and a row's (its co-sim at
        # MCOLS 4 and 8, T 4-16: the vector work it does not overlap, the MMs' fills)
        s_ = CAL["vpu_share"]
        over = (CAL["tile_layer"] + CAL["tile_row"] * R) * m["hidden"] / 1024 * \
            (s_ / hw.vpu + 1 - s_) / S + (R * CAL["coll"] if S > 1 else 0.0)
        if prompt:
            # fitted on its co-sim (compile_rows-like); a prompt run on the card costs more:
            # the model's today's kernel (the card's prompt runs) sits ~7% above compile_rows'
            # co-sim at every R and MCOLS (tiling's), the card's own compile_rows B0 4.7% under
            mxu *= CAL["tile_card"]
            over *= CAL["tile_card"]
    cyc = max(mem, mxu / S) + att + rec + over
    if split:
        return cyc, dict(bytes=byts + (kv if att and not tiled else 0.0), mem=mem, mxu=mxu / S,
                         rec=rec,
                         over=over, att=att,
                         b_mem=mem if mem >= mxu / S else 0.0,
                         b_mxu=mxu / S if mxu / S > mem else 0.0)
    return cyc


def union(E: int, k: int, R: int, E_eff: float | None = None) -> float:
    """Expected distinct experts among R consecutive tokens' top-k picks: k R^alpha
    (consecutive tokens share experts: the card's zero-miss layer runs at R = 1 and 2), at
    most as from E_eff equally likely experts (E_eff < E: the router's skew over a prompt,
    MOE_EEFF)."""
    e = min(E, E_eff or E)
    return min(float(E), k * R ** CAL["moe_alpha"], e * (1 - (1 - k / e) ** R))


def head_cycles(m: dict, hw: HW, fmt: tuple, R: int) -> float:
    """The LM head for R logits rows: it streams once a pass of rows; each row's sink."""
    _, hf, formats = fmt
    hf = group_format("head", 0, hf, formats)
    p = m["head"]
    byts = p * BITS[hf] / 8 * math.ceil(R / hw.rows_pass)
    return max(byts / hw.S / hw.bw(), mxu_cycles(p, hf, R, hw) / hw.S) + R * CAL["head_row"]


def ple_cycles(m: dict, hw: HW, fmt: tuple, R: int) -> float:
    """Gemma 4's PLE model projection, a row's."""
    if "ple" not in m:
        return 0.0
    f = group_format("ple", 0, fmt[0], fmt[2])
    p = m["ple"]["proj"]
    return max(p * BITS[f] / 8 * math.ceil(R / hw.rows_pass) / hw.S / hw.bw(),
               mxu_cycles(p, f, R, hw) / hw.S)


def run_cycles(m: dict, hw: HW, fmt: tuple, ctxs: list, logits: int = 1,
               experts: float | None = None, split: bool = False, prompt: bool = False):
    """Cycles of one device run: len(ctxs) rows through every layer, `logits` of them with
    logits (the LM head streams once for them). prompt: a prompt run (prefill.run's run-time
    rows: its run words, mask loads and per-row V^T appends cost CAL prompt_run over
    compile_rows' layers)."""
    ple = ple_cycles(m, hw, fmt, len(ctxs))
    tot = CAL["run"] + ple
    parts = dict(bytes=0.0, mem=0.0, mxu=0.0, rec=0.0, over=CAL["run"], att=0.0, b_mem=0.0,
                 b_mxu=0.0)
    lay_tot = 0.0
    for i, lay in layers(m):
        c, p = layer_cycles(m, i, lay, hw, fmt, ctxs, experts, split=True, prompt=prompt)
        lay_tot += c
        for k in parts:
            parts[k] += p[k]
    tot += lay_tot
    if prompt and len(ctxs) > 1:
        extra = lay_tot * (CAL["prompt_run"] - 1)
        tot += extra
        parts["over"] += extra
    if logits:
        tot += head_cycles(m, hw, fmt, logits)
    if split:
        # the head and the PLE projection stream or compute like a layer's weights
        R = len(ctxs)
        side = []
        if logits:
            _, hf, formats = fmt
            hf = group_format("head", 0, hf, formats)
            side.append((m["head"] * BITS[hf] / 8 * math.ceil(logits / hw.rows_pass),
                         mxu_cycles(m["head"], hf, logits, hw)))
            parts["over"] += logits * CAL["head_row"]
        if "ple" in m:
            f = group_format("ple", 0, fmt[0], fmt[2])
            side.append((m["ple"]["proj"] * BITS[f] / 8 * math.ceil(R / hw.rows_pass),
                         mxu_cycles(m["ple"]["proj"], f, R, hw)))
        for byts, mx in side:
            a, b = byts / hw.S / hw.bw(), mx / hw.S
            parts["b_mem" if a >= b else "b_mxu"] += max(a, b)
        return tot, parts
    return tot


UNITS = ("stream", "MXU", "attention", "DSTEP", "overhead", "host link", "host")


def shares(parts: dict, hw: HW, extra: dict | None = None) -> dict:
    """The share of a time each unit binds: the weight stream (the port or the memory,
    whichever is slower), the MXU, attention (its KV stream and latency), DeltaNet's DSTEP
    recurrence, the overhead (vector unit, control, runs); extra: other parts, in cycles
    (host link, host)."""
    d = {"stream": parts["b_mem"], "MXU": parts["b_mxu"], "attention": parts.get("att", 0.0),
         "DSTEP": parts["rec"], "overhead": parts["over"]}
    d.update(extra or {})
    tot = sum(d.values()) or 1.0
    return {k: v / tot for k, v in d.items()}


def bound(sh: dict, hw: HW) -> str:
    """The unit with the largest share, the stream named by what bounds it (port / memory)."""
    k = max(sh, key=sh.get)
    name = hw.bound() if k == "stream" else k
    return f"{name} {100 * sh[k]:.0f}%"


def bytes_per_token(m: dict, fmt: tuple, ctx: int) -> float:
    """A decode token's DRAM bytes at context ctx (all slices)."""
    hw = BOARDS["ypcb"]
    tot = sum(layer_cycles(m, i, lay, hw, fmt, [ctx], split=True)[1]["bytes"]
              for i, lay in layers(m))
    tot += m["head"] * BITS[group_format("head", 0, fmt[1], fmt[2])] / 8
    if "ple" in m:
        tot += m["ple"]["proj"] * BITS[group_format("ple", 0, fmt[0], fmt[2])] / 8
    return tot


# ------------------------------------------------------------------------------- workloads
def decode(name: str, hw: HW, wformat: str = "fp4", ctx: int = 512) -> dict:
    """Single-user decode at context ctx, device tokens a second (the card's decode loop picks
    every token: the host adds nothing a token)."""
    m = model(name)
    fmt = formats_of(name, wformat)
    cyc, parts = run_cycles(m, hw, fmt, [ctx], split=True)
    sh = shares(parts, hw)
    return dict(tok_s=hw.mhz * 1e6 / cyc, mcycles=cyc / 1e6,
                mb_token=bytes_per_token(m, fmt, ctx) / MB, shares=sh, bound=bound(sh, hw))


_RMAX: dict | None = None


def rmax_table() -> dict:
    """The compiler's prefill rows (tools/scoreboard_cosim.py probe): {"model|tmem|mcols|
    act_rows": {bucket: [TMEM / ACT RAM rows, also IMEM rows]}}."""
    global _RMAX
    if _RMAX is None:
        _RMAX = json.loads(RMAX_FILE.read_text()) if RMAX_FILE.exists() else {}
    return _RMAX


RMAX_CARD = {"qwen3": 9, "lfm2": 8, "qwen35": 4, "qwen35-2b": 4, "qwen35-4b": 3,
             "smollm3": 4, "phi4-mini": 3, "lfm2-2.6b": 4, "gemma4": 4}


def _fit_rows(name: str, hw: HW, bucket: int) -> tuple:
    """(TMEM rows, IMEM rows) of a prefill run in a bucket: the probe's, interpolated between
    its buckets (1 and 4), else the card's R_max scaled by the TMEM words."""
    t = rmax_table().get(f"{NAMES.get(name, name)}|{hw.tmem_words}|{hw.mcols}|{hw.act_rows}")
    if not t:
        cands = [(k, v) for k, v in rmax_table().items()
                 if k.startswith(NAMES.get(name, name) + "|") and
                 int(k.split("|")[1]) == hw.tmem_words]
        t = cands[0][1] if cands else None
    if not t:
        r = max(1, int(RMAX_CARD.get(name, 4) * hw.tmem_words / 65536))
        return r, r
    bs = sorted(int(b) for b in t)
    lo = max([b for b in bs if b <= bucket] or bs[:1])
    hi = min([b for b in bs if b >= bucket] or bs[-1:])
    a, b = t[str(lo)], t[str(hi)]
    w = 0 if hi == lo else (bucket - lo) / (hi - lo)
    return (int(a[0] + (b[0] - a[0]) * w), int(a[1] + (b[1] - a[1]) * w))


def _rule_rows(r: int, hw: HW, fmt: str) -> int:
    p = hw.rows_pass
    if r > p:
        return (r // p) * p
    if fmt == "fp4" and hw.pair and r > 1 and r % 2 and r > hw.mcols // 2:
        return r - 1
    return r


def prefill_rows(name: str, hw: HW, bucket: int = 1, fmt: str = "fp4") -> int:
    """Rows a prefill run in a bucket: --rows, else the most the TMEM fit allows (and IMEM's
    unless imem 0) as the host's rule takes them: whole passes above one pass, and for 4-bit
    an odd R's R - 1 (PAIR: prefer_rows); of those, and of the fewer whole passes and MXU
    groups, the R the model finds fastest a row (the host's rule weighs its cost a row)."""
    if hw.rows:
        return hw.rows
    tm, im = _fit_rows(name, hw, bucket)
    top = _rule_rows(max(1, min(tm, im) if hw.imem else tm), hw, fmt)
    cands = {top} | {r for r in range(hw.mcols, top, hw.mcols)}
    if len(cands) == 1:
        return top
    m = model(name)
    f = formats_of(name, "fp4" if fmt == "fp4" else "auto" if fmt == "mix" else fmt)
    c0 = (bucket - 1) * BLOCK + BLOCK // 2
    cost = {r: run_cycles(m, hw, f, [c0 + i for i in range(r)], logits=0) / r for r in cands}
    best = min(cost.values())
    return max(r for r in cands if cost[r] <= best * 1.001)


def prefill(name: str, hw: HW, wformat: str = "fp4", P: int = 1024, p0: int = 0) -> dict:
    """A P-token prompt from position p0 in prompt runs (a run stays in its 256-token bucket
    when R < 256; the last run's last row takes the logits). TTFT = the runs' device time
    plus the host's time a run (warm program cache)."""
    m = model(name)
    fmt = formats_of(name, wformat)
    cyc, runs, p, rows = 0.0, 0, p0, []
    acc = {}
    while p < p0 + P:
        R = prefill_rows(name, hw, p // BLOCK + 1, fmt[0] if not fmt[2] else "mix")
        n = min(R, p0 + P - p, BLOCK - p % BLOCK if R < BLOCK else R)
        last = p + n >= p0 + P
        c, parts = run_cycles(m, hw, fmt, [p + r for r in range(n)], logits=1 if last else 0,
                              split=True, prompt=True)
        cyc += c
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + v
        runs += 1
        rows.append(R)
        p += n
    t = cyc / (hw.mhz * 1e6) + runs * hw.host_run_ms / 1e3
    sh = shares(acc, hw, {"host": runs * hw.host_run_ms * hw.mhz * 1e3})
    out = dict(rows=rows[0], rows_last=rows[-1], runs=runs, ttft_s=t, tok_s=P / t,
               device_s=cyc / (hw.mhz * 1e6), shares=sh, bound=bound(sh, hw),
               kernel=hw.kernel)
    if hw.kernel == "tiled":
        # the host takes today's kernel (and its rows) where that is faster
        other = prefill(name, hw.replace(kernel="today", rows=0), wformat, P, p0)
        if other["ttft_s"] < t:
            return other
    return out


def batch(name: str, hw: HW, users: int, wformat: str = "fp4", ctx: int = 512) -> dict:
    """Batched decode of `users` sequences at context ctx: a step runs them in blocks of
    batch_rows (rows a pass), one run a block, every weight streamed once a run, each row's
    logits picked on the card. Total and per-user tokens a second (device and host)."""
    m = model(name)
    fmt = formats_of(name, wformat)
    rb = hw.batch_rows or hw.rows_pass
    cyc, runs, left = 0.0, 0, users
    acc = {}
    while left > 0:
        n = min(rb, left)
        c, parts = run_cycles(m, hw, fmt, [ctx] * n, logits=n, split=True)
        cyc += c
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + v
        runs += 1
        left -= n
    host = runs * hw.host_run_ms / 1e3 if users > 1 else 0.0
    t = cyc / (hw.mhz * 1e6) + host
    sh = shares(acc, hw, {"host": host * hw.mhz * 1e6})
    return dict(users=users, total_tok_s=users / t, user_tok_s=1 / t, runs=runs,
                device_ms=cyc / (hw.mhz * 1e3), shares=sh, bound=bound(sh, hw))


# ------------------------------------------------------------------------------- MoE
# Misses a decode token (of L x k) against the share of the pool in slots: the per-layer LRU
# over four 2048-token router traces (docs/offload.md section 3); decayed use (the default)
# misses ~12% less (LFU); the card's chats miss more than the traces (miss_scale)
CACHE = {"gemma4-26b": [(0.05, 129.8), (0.10, 102.4), (0.19, 57.3), (0.20, 55.3),
                        (0.30, 31.4), (0.50, 8.9), (0.75, 1.2), (1.0, 0.0)],
         "qwen35-35b": [(0.05, 169.4), (0.10, 125.4), (0.13, 110.2), (0.20, 80.8),
                        (0.30, 52.8), (0.50, 22.6), (0.75, 5.9), (1.0, 0.0)]}
LFU = 0.88
MOE_EEFF = {"gemma4-26b": 75.0, "qwen35-35b": 145.0}   # a layer's distinct experts over a prompt
RESERVE_GB = 0.3                                        # KV, I/O, programs beside the slots


def _interp(tab, x):
    if x <= tab[0][0]:
        return tab[0][1]
    for (x0, y0), (x1, y1) in zip(tab, tab[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return 0.0


def moe_fit(name: str, hw: HW) -> dict:
    """Where a MoE model's experts live: slots in the fast memory beside the rest of the image
    (the S slices' memory, each expert split across them), then the far tier's, the rest on
    the host. The untied embedding table stays on the host (the 35B's)."""
    m = model(name)
    base, hf, formats = formats_of(name, "auto")
    dense = sum(p * BITS[group_format(g, i, base, formats)] / 8
                for i, lay in layers(m) for g, p in lay["groups"].items())
    dense += m["head"] * BITS[hf] / 8
    lay = next(x for _, x in layers(m) if "moe" in x)
    mo = lay["moe"]
    nmoe = sum(1 for _, x in layers(m) if "moe" in x)
    eb = mo["expert_params"] * BITS[group_format("experts", 0, base, formats)] / 8
    pool = nmoe * mo["experts"] * eb
    fast = hw.S * (hw.fast_gb * 2**30 - RESERVE_GB * GB) - dense
    slots = max(0.0, min(pool, fast)) / eb
    far = min(pool - slots * eb, hw.far_gb * GB) / eb if hw.far_gbps else 0.0
    return dict(dense_gb=dense / GB, pool_gb=pool / GB, expert_mb=eb / MB, slots=int(slots),
                frac=slots * eb / pool, far_slots=int(far), experts=nmoe * mo["experts"],
                layers=nmoe, top_k=mo["top_k"], E=mo["experts"],
                resident=slots * eb >= pool * 0.999)


def _misses(name: str, frac: float) -> float:
    return 0.0 if frac >= 0.999 else _interp(CACHE[name], frac) * LFU * CAL["miss_scale"]


def moe_decode(name: str, hw: HW, ctx: int = 512) -> dict:
    """MoE decode: the card's time with the routed experts present, plus the exposed link
    time of the misses (from the host over PCIe, or from the far tier)."""
    m = model(name)
    base, _, formats = fmt = formats_of(name, "auto")
    fit = moe_fit(name, hw)
    f = hw.mhz * 1e6
    cc, parts = run_cycles(m, hw, fmt, [ctx], split=True)
    card = cc / f
    misses = _misses(name, fit["frac"])                 # a token's, beyond the fast memory
    host = _misses(name, min(1.0, (fit["slots"] + fit["far_slots"]) / fit["experts"])) \
        if fit["far_slots"] else misses                 # ... and beyond the far tier too
    far = misses - host
    eb = fit["expert_mb"] * MB
    fast = eb / hw.S / (hw.bw() * f)                    # an expert from the fast memory, s
    exposed = link = 0.0
    k = fit["top_k"]
    for i, lay in layers(m):
        if "moe" not in lay:
            continue
        mh, mf = host / fit["layers"], far / fit["layers"]
        # what the card computes while the misses stream: the part beside the request (the
        # dense MLP, the shared expert) and the experts it has
        beside = sum(p * BITS[group_format(g, i, base, formats)] / 8
                     for g, p in lay["groups"].items() if g in ("gateup", "down", "shared"))
        hide = beside / hw.S / (hw.bw() * f) + max(0.0, k - mh - mf) * fast
        lk = CAL["link_lead_ms"] / 1e3 * (mh > 0) + mh * eb / (hw.pcie_gbps * GB)
        link += lk
        exposed += max(0.0, lk - hide)
        if mf:                                          # read from the far tier instead
            exposed += mf * (eb / (hw.far_gbps * GB) - fast)
    t = card + exposed
    sh = shares(parts, hw, {"host link": exposed * f})
    return dict(tok_s=1 / t, card_tok_s=1 / card, misses=misses, far_misses=far,
                link_s=link, exposed_s=exposed, resident=fit["resident"],
                slots=fit["slots"], frac=fit["frac"], shares=sh, bound=bound(sh, hw))


def moe_prefill(name: str, hw: HW, P: int = 1024, R: int | None = None) -> dict:
    """MoE prefill, layer-major (docs/offload.md 13): runs of R rows a layer (default 2, the
    card's; --rows, at most the request's ids / top-k: --moe-ids), every slot serving
    the running layer, so a layer's misses are its prompt's distinct experts; each run
    streams its rows' union of experts. A DeltaNet model's first conv_k - 1 rows run as
    decode steps (docs/offload.md 13.6). The layer runs are their own kernel: --kernel does
    not apply."""
    m = model(name)
    fmt = formats_of(name, "auto")
    hw = hw.replace(kernel="today")
    fit = moe_fit(name, hw)
    R = R or min(hw.rows or max(2, hw.moe_ids // fit["top_k"]),
                 max(1, hw.moe_ids // fit["top_k"]))
    eeff = MOE_EEFF[name]
    steps = 0
    dn = next((x["deltanet"] for _, x in layers(m) if "deltanet" in x), None)
    if dn and P > 1:
        steps = min(P - 1, dn.get("conv_k", 4) - 1)
    cyc, runs = 0.0, 0
    acc = dict(b_mem=0.0, b_mxu=0.0, rec=0.0, over=0.0, att=0.0)
    for i, lay in layers(m):
        p = steps
        while p < P:
            n = min(R, P - p)
            ex = union(lay["moe"]["experts"], lay["moe"]["top_k"], n, eeff) \
                if "moe" in lay else None
            c, parts = layer_cycles(m, i, lay, hw, fmt, [p + r for r in range(n)], ex,
                                    split=True)
            cyc += c + CAL["run"]
            for k in acc:
                acc[k] += parts[k]
            acc["over"] += CAL["run"]
            runs += 1
            p += n
    cyc += head_cycles(m, hw, fmt, 1)
    device = cyc / (hw.mhz * 1e6)
    link = 0.0
    if not fit["resident"]:
        # a layer's distinct experts over the prompt; the slots hold the decode's share of
        # them, the rest comes from the far tier or the host, once each (pooled slots)
        U = union(fit["E"], fit["top_k"], P, eeff) * fit["layers"]
        far = fit["far_slots"] / fit["experts"]
        eb = fit["expert_mb"] * MB
        link = U * (1 - fit["frac"] - far) * eb / (hw.pcie_gbps * GB)
        if far:
            link += U * far * eb / (hw.far_gbps * GB)
    host = runs * CAL["moe_run_ms"] / 1e3
    t = device + CAL["prefill_exposed"] * link + host
    if steps:
        t += steps / moe_decode(name, hw, 0)["tok_s"]
    f = hw.mhz * 1e6
    sh = shares(acc, hw, {"host link": CAL["prefill_exposed"] * link * f, "host": host * f})
    return dict(ttft_s=t, tok_s=P / t, device_s=device, link_s=link, rows=R, runs=runs,
                shares=sh, bound=bound(sh, hw))


# ------------------------------------------------------------------------------- the scoreboard
def scoreboard(hw: HW, wformat: str = "fp4", ctx: int = 512, names=None,
               users=(1, 4, 8, 16), prompts=(256, 1024)) -> dict:
    """Every workload on a configuration: decode and prefill of the dense models, batched
    decode of the batched ones, decode and prefill of the MoEs."""
    names = names or DENSE
    out = {"hw": dataclasses.asdict(hw), "wformat": wformat, "ctx": ctx,
           "decode": {}, "prefill": {}, "batch": {}, "moe": {}}
    for n in names:
        if n not in MOE:
            out["decode"][n] = decode(n, hw, wformat, ctx)
            out["prefill"][n] = {P: prefill(n, hw, wformat, P) for P in prompts}
    for n in [x for x in names if x in BATCHED] or BATCHED:
        out["batch"][n] = {u: batch(n, hw, u, wformat, ctx) for u in users}
    for n in MOE:
        out["moe"][n] = dict(decode=moe_decode(n, hw, ctx), fit=moe_fit(n, hw),
                             prefill={P: moe_prefill(n, hw, P) for P in prompts})
    return out


def _gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def summary(sc: dict) -> dict:
    """One number a workload, geometric means over the models: decode tok/s; prefill tok/s at
    the longest prompt; batched total tok/s at the most users; MoE sqrt(decode x prefill)."""
    P = max(next(iter(sc["prefill"].values())))
    U = max(next(iter(sc["batch"].values())))
    return dict(decode=_gmean([v["tok_s"] for v in sc["decode"].values()]),
                prefill=_gmean([v[P]["tok_s"] for v in sc["prefill"].values()]),
                batch=_gmean([v[U]["total_tok_s"] for v in sc["batch"].values()]),
                moe=_gmean([math.sqrt(v["decode"]["tok_s"] * v["prefill"][P]["tok_s"])
                            for v in sc["moe"].values()]))


def bounds(sc: dict) -> dict:
    """What binds each workload: the units' shares of the time averaged over the models (the
    summary's points: the longest prompt, the most users, MoE decode and prefill), the
    largest named ("port 81%", "MXU 64%", "overhead 40%", "host link 55%")."""
    hw = HW(**sc["hw"])
    P = max(next(iter(sc["prefill"].values())))
    U = max(next(iter(sc["batch"].values())))
    groups = dict(decode=[v["shares"] for v in sc["decode"].values()],
                  prefill=[v[P]["shares"] for v in sc["prefill"].values()],
                  batch=[v[U]["shares"] for v in sc["batch"].values()],
                  moe=[x for v in sc["moe"].values()
                       for x in (v["decode"]["shares"], v["prefill"][P]["shares"])])
    out = {}
    for k, xs in groups.items():
        if xs:
            keys = {u for x in xs for u in x}
            out[k] = bound({u: sum(x.get(u, 0.0) for x in xs) / len(xs) for u in keys}, hw)
    return out


def compare(base: HW, new: HW, full: bool = False, **kw):
    """Each workload's ratio new / base and their geometric mean, "score" (the four weighed
    equally); full: also (the base's bounds, the new configuration's bounds)."""
    sa, sb = scoreboard(base, **kw), scoreboard(new, **kw)
    a, b = summary(sa), summary(sb)
    r = {k: b[k] / a[k] for k in a}
    r["score"] = _gmean(list(r.values()))
    return (r, bounds(sa), bounds(sb)) if full else r


WHATIFS = [
    ("MCOLS 8", dict(mcols=8)),
    ("TMEM x2 (8 ACT RAM rows)", dict(tmem_words=131072, act_rows=8)),
    ("TMEM x4 (16 ACT RAM rows)", dict(tmem_words=262144, act_rows=16)),
    ("64 rows a run, today's kernel", dict(rows=64, act_rows=64, imem=0)),
    ("tile kernel, 16 rows a run (64K TMEM)", dict(kernel="tiled", rows=16)),
    ("tile kernel, 16 rows, MCOLS 8", dict(kernel="tiled", rows=16, mcols=8)),
    ("8 users in one pass (replay)", dict(batch_rows=8, act_rows=8)),
    ("VPU x2", dict(vpu=2.0)),
    ("MXU pumped x2", dict(mxu_pump=2.0)),
    ("MCOLS 8 + TMEM x4 + VPU x2", dict(mcols=8, tmem_words=262144, act_rows=16, vpu=2.0)),
    ("MXU x2 + TMEM x4 + VPU x2 (pumped)", dict(mxu_pump=2.0, tmem_words=262144, act_rows=16,
                                                 vpu=2.0)),
    ("port 256 B a cycle (memory permitting)", dict(port_bytes=256)),
    ("port 256 B + MXU x2", dict(port_bytes=256, mxu_pump=2.0)),
    ("MoE runs of 8 rows (64-id requests)", dict(moe_ids=64)),
    ("MoE runs of 8 rows, each row its experts", dict(moe_ids=64, moe_rows="gather")),
]


def whatifs(b: str = "ypcb", wformat: str = "fp4", items=None) -> list:
    """[(label, overrides, ratios, the board's bounds, the what-if's bounds)]."""
    base = board(b)
    out = []
    for label, kw in items or WHATIFS:
        r, bb, bn = compare(base, base.replace(**kw), full=True, wformat=wformat)
        out.append((label, kw, r, bb, bn))
    return out


# ------------------------------------------------------------------------------- card / co-sim
# What the model is checked against. Card: production builds at 133.33 MHz (pa e4db91c9,
# fmvf 542fc43a, fixmath 1025907f), tools/qual/perf.py's decode counters (positions 512-575,
# int8 head) and prompt runs (docs/prefill.md 7), otpu-serve's batched steps (docs/serve.md),
# the streamed MoEs (docs/offload.md 13.7, 14.4). Co-sim: the RTL with LiteDRAM's controller
# at DDR3-1066 (docs/litedram.md, docs/prefill.md 8; build/sbcosim.jsonl: this tool's runs).
CARD_DECODE = [   # (model, (layers, head, formats), Mcycles a token, MB a token measured, src)
    ("qwen3", ("fp4", "int8", ""), 3.730, 444.5, "fmvf qual"),
    ("qwen3", ("int8", "int8", ""), 5.506, 664.5, "fmvf qual"),
    ("lfm2", ("fp4", "int8", ""), 1.369, 164.5, "fmvf qual"),
    ("lfm2", ("int8", "int8", ""), 2.025, 245.5, "fmvf qual"),
    ("qwen35", ("fp4", "int8", ""), 4.817, 582.4, "fmvf qual"),
    ("qwen35", ("int8", "int8", ""), 6.818, 831.4, "fmvf qual"),
    ("phi4-mini", ("int8", "int8", ""), 32.936, 4011, "pa mix1"),
    ("phi4-mini", ("int8", "int8", "mlp@4-27=fp4"), 25.640, 3105, "pa mix1"),
    ("phi4-mini", ("fp4", "int8", ""), 19.971, 2401, "pa mix1"),
    ("smollm3", ("int8", "int8", ""), 26.309, 3203, "pa mix1"),
    ("smollm3", ("int8", "int8", "gateup@9-35=fp4"), 21.409, 2594, "pa mix1"),
    ("smollm3", ("fp4", "int8", ""), 14.980, 1796, "pa mix1"),
    ("qwen35-4b", ("int8", "int8", "delta=fp4,mlp=fp4"), 23.384, 2770, "pa mix1"),
    ("qwen35-4b", ("fp4", "int8", ""), 22.206, 2623, "pa mix1"),
    ("qwen35-2b", ("fp4", "int8", ""), 10.800, 1282, "pa mix1"),
    ("lfm2-2.6b", ("int8", "int8", ""), 21.764, 2661, "pa mix2"),
    ("lfm2-2.6b", ("int8", "int8", "conv=fp4,mlp=fp4"), 12.298, 1486, "pa mix2"),
    ("lfm2-2.6b", ("fp4", "int8", ""), 11.958, 1444, "pa mix2"),
    ("gemma4", ("int8", "int8", ""), 20.875, 2496, "fmvf mix4"),
    ("gemma4", ("int8", "int8", "attn@15-34=fp4,mlp@15-34=fp4"), 15.751, 1855, "fmvf mix4"),
    ("gemma4", ("fp4", "int8", ""), 13.375, 1558, "fmvf mix4"),
]
CARD_PREFILL = [  # (model, formats, prompt P, rows a run, device Mcycles, source)
    ("qwen3", ("int8", "int8", ""), 236, 4, 288.0, "fmvf prompt runs, 59 runs"),
    ("qwen3", ("int8", "int8", ""), 236, 8, 268.7, "fmvf off: compile_rows, 30 runs of 8"),
    ("lfm2", ("int8", "int8", ""), 238, 4, 96.5, "fmvf prompt runs, 60 runs"),
    ("lfm2", ("int8", "int8", ""), 238, 8, 92.7, "fmvf off: compile_rows, 31 runs"),
    ("lfm2-2.6b", ("int8", "int8", "conv=fp4,mlp=fp4"), 238, 4, 1185.5, "fmvf prompt runs"),
    ("smollm3", ("int8", "int8", "gateup@9-35=fp4"), 291, 4, 1757.7,
     "fmvf prompt runs, 73 runs"),
    ("phi4-mini", ("int8", "int8", "mlp@4-27=fp4"), 227, 3, 2066.2,
     "fmvf prompt runs, 76 runs"),
    ("gemma4", ("int8", "int8", ""), 224, 4, 1056.2, "fmvf prompt runs, 56 runs"),
    ("qwen35-4b", ("fp4", "int8", ""), 237, 2, 2113.1, "fmvf, 119 runs"),
    ("qwen35-4b", ("fp4", "int8", ""), 237, 3, 2441.7, "fmvf, 79 runs"),
    ("qwen35", ("fp4", "int8", ""), 237, 4, 412.8, "pa: compile_rows, 3.096 s, 60 runs"),
    ("qwen35-2b", ("fp4", "int8", ""), 237, 4, 752.6, "pa: compile_rows, 5.645 s, 60 runs"),
    # docs/prefill.md 7: a first prompt (A0) and a next turn of 13 tokens (B1), prompt runs
    ("qwen3", ("int8", "int8", ""), 24, 4, 30.5, "fmvf A0, 6 runs"),
    ("qwen3", ("int8", "int8", ""), 13, 4, 21.4, "fmvf B1, 4 runs", 267),
    ("lfm2", ("int8", "int8", ""), 21, 4, 9.9, "fmvf A0, 6 runs"),
    ("lfm2", ("int8", "int8", ""), 13, 4, 7.0, "fmvf B1, 4 runs", 269),
    ("lfm2-2.6b", ("int8", "int8", "conv=fp4,mlp=fp4"), 21, 4, 111.3, "fmvf A0, 6 runs"),
    ("lfm2-2.6b", ("int8", "int8", "conv=fp4,mlp=fp4"), 13, 3, 91.5, "fmvf B1, 5 runs", 269),
    ("smollm3", ("int8", "int8", "gateup@9-35=fp4"), 80, 4, 482.4, "fmvf A0, 20 runs"),
    ("smollm3", ("int8", "int8", "gateup@9-35=fp4"), 13, 4, 94.3, "fmvf B1, 4 runs", 322),
    ("phi4-mini", ("int8", "int8", "mlp@4-27=fp4"), 15, 3, 141.2, "fmvf A0, 5 runs"),
    ("phi4-mini", ("int8", "int8", "mlp@4-27=fp4"), 13, 3, 135.4, "fmvf B1, 5 runs", 258),
    ("gemma4", ("int8", "int8", ""), 13, 4, 76.2, "fmvf A0, 4 runs"),
    ("gemma4", ("int8", "int8", ""), 13, 4, 76.2, "fmvf B1, 4 runs", 224),
]
CARD_BATCH = [    # (model, formats, rows, context, device ms a run, source)
    ("qwen3", ("int8", "int8", ""), 1, 64, 39.5, "1025907f serve"),
    ("qwen3", ("int8", "int8", ""), 2, 64, 41.6, "1025907f serve"),
    ("qwen3", ("int8", "int8", ""), 4, 64, 46.3, "1025907f serve"),
    ("qwen3", ("int8", "int8", ""), 1, 1024, 43.1, "1025907f serve"),
    ("qwen3", ("int8", "int8", ""), 2, 1024, 48.8, "1025907f serve"),
    ("qwen3", ("int8", "int8", ""), 4, 1024, 60.6, "1025907f serve"),
]
COSIM = [         # (model, formats, rows, position, cycles, what, hw overrides)
    ("qwen35", ("fp4", "int8", ""), 4, 544, 6820714, "prefill run, R 4 (prefill.md 8)", {}),
    ("qwen35", ("fp4", "int8", ""), 4, 544, 4 * 1413000, "R 4, MCOLS 8 (prefill.md 8)",
     {"mcols": 8}),
    ("qwen3", ("fp4", "int8", ""), 0, 544, 3696899, "decode (litedram.md)", {}),
    ("lfm2", ("fp4", "int8", ""), 0, 544, 1355035, "decode (litedram.md)", {}),
    ("qwen35", ("fp4", "int8", ""), 0, 544, 4765281, "decode (litedram.md)", {}),
    ("qwen3", ("int8", "int8", ""), 0, 544, 5471471, "decode (litedram.md)", {}),
    ("lfm2", ("int8", "int8", ""), 0, 544, 2012426, "decode (litedram.md)", {}),
    ("qwen35", ("int8", "int8", ""), 0, 544, 6764651, "decode (litedram.md)", {}),
] + [  # the tiling agent's (tools/perf_tiled.py, branch tiled-prefill; otpu_seq's TAW fixed):
       # today's compile_rows, ACT_ROWS max(MCOLS, R), the first 2 layers, rows from pos, no
       # logits; cycles a row x R. The card's prompt runs cost ~6% more (tile_card)
    ("qwen3", (f, "int8", ""), R, pos, round(c * 1e3 * R),
     f"2 layers, MCOLS {mc}, R {R}" + (f" at {pos}" if pos != 256 else "") + " (tiling)",
     {"mcols": mc, "act_rows": max(mc, R), "tmem_words": 65536 if R <= 8 else 262144}, 2)
    for f, mc, pos, cs in (("fp4", 4, 256, (83.6, 82.6, 83.1, 84.7)),
                           ("fp4", 8, 256, (57.7, 53.8, 53.0, 54.0)),
                           ("int8", 4, 256, (89.2, 83.8, 83.8, 85.0)),
                           ("int8", 8, 256, (89.2, 56.8, 54.5, 54.4)),
                           ("fp4", 4, 1024, (110.8, None, None, None)),
                           ("fp4", 8, 1024, (84.1, 82.0, None, None)))
    for R, c in zip((4, 8, 16, 32), cs) if c]
COSIM_TILED = [   # the tiling agent's tile kernel at TMEM 64K: Qwen3-0.6B, 2 layers, rows from
                  # pos, no logits; cycles a row (2026-10-12; its MXU 96-97% busy at MCOLS 4)
    ("qwen3", (f, "int8", ""), T, pos, round(c * 1e3 * T),
     f"2 layers, tile kernel, MCOLS {mc}, T {T}" + (f" at {pos}" if pos != 256 else ""),
     {"kernel": "tiled", "mcols": mc, "act_rows": T if T > mc else 0}, 2)
    for f, mc, pos, cs in (("fp4", 4, 256, (79.5, 76.5, 76.4)), ("int8", 4, 256, (84.7, 76.5, 76.4)),
                           ("fp4", 8, 256, (49.1, 45.1, 44.4)), ("int8", 8, 256, (82.5, 48.8, 44.4)),
                           ("fp4", 4, 1024, (None, None, 92.4)), ("fp4", 8, 1024, (None, None, 58.4)))
    for T, c in zip((4, 8, 16), cs) if c]
LAYER_COSIM_TILING = [  # the tiling agent's slopes (2 - 1 layers), today's compile_rows on the
                        # card's configuration (ACT_ROWS 4, TMEM 64K): checked, not fitted
    ("qwen3", ("int8", "int8", ""), (1, 2), 4, 100, 40.9e3, "tiling, compile_rows"),
    ("qwen3", ("int8", "int8", ""), (1, 2), 4, 256, 44.6e3, "tiling, compile_rows"),
    ("qwen3", ("int8", "int8", ""), (1, 2), 4, 267, 44.6e3, "tiling, compile_rows"),
    ("qwen3", ("fp4", "int8", ""), (1, 2), 4, 256, 41.8e3, "tiling, compile_rows"),
]
LAYER_COSIM = [   # (model, formats, layers [lo, hi), rows, position, cycles a row and layer, src)
    # docs/prefill.md 8: the slope from 4 to 8 layers, no logits
    ("qwen35-4b", ("fp4", "int8", ""), (4, 8), 3, 544, 322.3e3, "prefill.md 8"),
    ("qwen35-4b", ("fp4", "int8", ""), (4, 8), 2, 544, 278.0e3, "prefill.md 8"),
    ("phi4-mini", ("int8", "int8", "mlp=fp4"), (4, 5), 3, 128, 277.9e3, "prefill.md 8"),
    ("phi4-mini", ("int8", "int8", "mlp=fp4"), (4, 5), 2, 128, 279.2e3, "prefill.md 8"),
    ("phi4-mini", ("int8", "int8", ""), (4, 5), 3, 128, 291.0e3, "prefill.md 8"),
    ("phi4-mini", ("int8", "int8", ""), (4, 5), 2, 128, 431.1e3, "prefill.md 8"),
    ("smollm3", ("int8", "int8", "gateup=fp4"), (4, 5), 3, 3968, 255.1e3, "prefill.md 8"),
    ("smollm3", ("int8", "int8", "gateup=fp4"), (4, 5), 2, 3968, 281.2e3, "prefill.md 8"),
    ("smollm3", ("int8", "int8", ""), (4, 5), 3, 3968, 263.6e3, "prefill.md 8"),
    ("smollm3", ("int8", "int8", ""), (4, 5), 2, 3968, 372.0e3, "prefill.md 8"),
]
CARD_MOE = [      # (model, quantity, measured, source)
    ("qwen35-35b", "decode tok/s", 5.30, "s17, 128 tokens, 103 misses a token"),
    ("gemma4-26b", "decode tok/s", 3.70, "14.4, 128 tokens, 88 misses a token"),
    ("qwen35-35b", "prefill s (134 tokens)", 13.32, "pfcomp, R 2 pooled"),
    ("gemma4-26b", "prefill s (124 tokens)", 16.25, "pfcomp2, R 2 pooled"),
]


MOE_LAYER_RUNS = [  # (model, layer kind, rows, ms a zero-miss layer run, source: offload.md 13.7)
    ("qwen35-35b", "delta", 1, 2.48, "pftime"), ("qwen35-35b", "delta", 2, 3.38, "pfpool"),
    ("qwen35-35b", "attn", 1, 1.98, "pftime"), ("qwen35-35b", "attn", 2, 2.61, "pfpool"),
    ("gemma4-26b", "sliding", 2, 7.29, "pfpool"), ("gemma4-26b", "global", 2, 7.81, "pfpool"),
]


def _prompt_runs(P: int, R: int, p0: int = 0) -> list:
    """A prompt's runs: (rows' contexts, logits rows), R rows a run inside a bucket."""
    out, p = [], p0
    while p < p0 + P:
        n = min(R, p0 + P - p, BLOCK - p % BLOCK if R < BLOCK else R)
        out.append(([p + r for r in range(n)], int(p + n >= p0 + P)))
        p += n
    return out


def points() -> list:
    """Every card and co-sim point the dense model is fitted or checked on: dicts of group,
    model, fmt, runs [(contexts, logits rows)], hw overrides, measured cycles, label."""
    out = []
    for name, fmt, mc, _, src in CARD_DECODE:
        out.append(dict(group="decode", model=name, fmt=fmt, runs=[([544], 1)], hw={},
                        y=mc * 1e6, label=f"{name} {'/'.join(x for x in fmt if x)} ({src})"))
    for name, fmt, P, R, mc, src, *p0 in CARD_PREFILL:
        p0 = p0[0] if p0 else 0
        at = f" at {p0}" if p0 else ""
        out.append(dict(group="prefill", model=name, fmt=fmt, runs=_prompt_runs(P, R, p0),
                        hw={}, y=mc * 1e6, label=f"{name} {P} tokens{at}, R {R} ({src})",
                        prompt="compile_rows" not in src))
    for name, fmt, R, ctx, ms, src in CARD_BATCH:
        out.append(dict(group="batched", model=name, fmt=fmt, runs=[([ctx] * R, R)], hw={},
                        y=ms * 133.33e3, label=f"{name} {R} rows at {ctx} ({src})"))
    for name, fmt, R, pos, cyc, what, kw, *nl in COSIM:
        runs = [([pos], 1)] if R == 0 else [([pos + r for r in range(R)], 0)]
        out.append(dict(group="co-sim 2-layer" if nl else "co-sim", model=name, fmt=fmt,
                        runs=runs, hw=kw, y=cyc,
                        label=f"{name} {fmt[0]} {what}", n_layers=nl[0] if nl else 0))
    for name, fmt, R, pos, cyc, what, kw, nl in COSIM_TILED:
        out.append(dict(group="co-sim tile kernel", model=name, fmt=fmt,
                        runs=[([pos + r for r in range(R)], 0)], hw=kw, y=cyc,
                        label=f"{name} {fmt[0]} {what}", n_layers=nl))
    for grp, rows in (("co-sim layer", LAYER_COSIM),
                      ("co-sim layer (compile_rows)", LAYER_COSIM_TILING)):
        for name, fmt, span, R, pos, cyc, src in rows:
            fs = "/".join(x for x in fmt if x)
            out.append(dict(group=grp, model=name, fmt=fmt, hw={}, y=cyc, span=span,
                            ctxs=[pos + r for r in range(R)], nofit=grp != "co-sim layer",
                            label=f"{name} {fs} layer, R {R} at {pos} ({src})"))
    return out


def truncated(m: dict, n: int) -> dict:
    """The model's first n layers."""
    runs, left = [], n
    for k, lay in m["runs"]:
        if left <= 0:
            break
        runs.append([min(k, left), lay])
        left -= k
    return dict(m, runs=runs, n_layers=n)


def point_cycles(pt: dict) -> float:
    hw = BOARDS["ypcb"].replace(**pt["hw"])
    m = model(pt["model"])
    if "span" in pt:                                    # a row's cycles a layer
        lo, hi = pt["span"]
        L = list(layers(m))
        return sum(layer_cycles(m, i, L[i][1], hw, pt["fmt"], pt["ctxs"])
                   for i in range(lo, hi)) / (hi - lo) / len(pt["ctxs"])
    if pt.get("n_layers"):
        m = truncated(m, pt["n_layers"])
    return sum(run_cycles(m, hw, pt["fmt"], ctxs, logits=lg, prompt=pt.get("prompt", False))
               for ctxs, lg in pt["runs"])


def _with_kappa(name, k, f):
    save = CAL["kappa"].get(name)
    CAL["kappa"][name] = k
    try:
        return f()
    finally:
        if save is None:
            CAL["kappa"].pop(name)
        else:
            CAL["kappa"][name] = save


def _lsq(rows):
    """min sum ((b + x a1 + y a2 - t) / t)^2 over x, y >= 0; rows (b, a1, a2, t)."""
    def solve(use):
        S = [[0.0] * 2 for _ in range(2)]
        v = [0.0, 0.0]
        for b, a1, a2, t in rows:
            a = (a1 / t, a2 / t)
            r = (t - b) / t
            for i in range(2):
                v[i] += a[i] * r * use[i]
                for j in range(2):
                    S[i][j] += a[i] * a[j] * use[i] * use[j]
        if use == (1, 1):
            det = S[0][0] * S[1][1] - S[0][1] * S[1][0]
            if abs(det) < 1e-18:
                return None
            return ((v[0] * S[1][1] - v[1] * S[0][1]) / det,
                    (S[0][0] * v[1] - S[1][0] * v[0]) / det)
        i = use.index(1)
        x = v[i] / S[i][i] if S[i][i] else 0.0
        return (x, 0.0) if i == 0 else (0.0, x)

    def err(k):
        return sum(((b + k[0] * a1 + k[1] * a2 - t) / t) ** 2 for b, a1, a2, t in rows)
    cands = [k for k in (solve((1, 1)), solve((1, 0)), solve((0, 1))) if k is not None and
             min(k) >= 0]
    return min(cands or [(0.0, 0.0)], key=err)


def fit_model(name: str, pts: list) -> tuple:
    """(kappa_pass, kappa_row) of a model: least squares (relative) over its points."""
    rows = []
    for pt in pts:
        b = _with_kappa(name, (0.0, 0.0), lambda: point_cycles(pt))
        a1 = _with_kappa(name, (1.0, 0.0), lambda: point_cycles(pt)) - b
        a2 = _with_kappa(name, (0.0, 1.0), lambda: point_cycles(pt)) - b
        rows.append((b, a1, a2, pt["y"]))
    k = _lsq(rows)
    return round(k[0], 4), round(k[1], 4)


def fit(pts=None) -> dict:
    """kappa of every model with card points (decode, prefill, batched; the co-sim's
    prefill run where the card has none)."""
    pts = pts if pts is not None else [p for p in points() if not p["hw"] and
                                       not p.get("n_layers") and not p.get("nofit")]
    out = {}
    for name in sorted({p["model"] for p in pts}):
        out[name] = fit_model(name, [p for p in pts if p["model"] == name])
    return out


def validate(loo: bool = True) -> list:
    """(group, point, measured, model, error %, leave-one-out error %) for every point; the
    leave-one-out model refits the point's model without it (decode points: its other
    formats)."""
    out = []
    pts = points()
    fitted = [p for p in pts if not p["hw"] and not p.get("n_layers") and not p.get("nofit")]
    for pt in pts:
        got = point_cycles(pt)
        e = 100 * (got / pt["y"] - 1)
        el = None
        if loo:
            rest = [p for p in fitted if p is not pt and p["model"] == pt["model"]]
            if rest:
                k = fit_model(pt["model"], rest)
                el = 100 * (_with_kappa(pt["model"], k, lambda: point_cycles(pt)) / pt["y"] - 1)
        out.append((pt["group"], pt["label"], pt["y"] / 1e6, got / 1e6, e, el))
    hw = BOARDS["ypcb"]
    for name, kind, R, ms, src in MOE_LAYER_RUNS:
        m = model(name)
        fmt = formats_of(name, "auto")
        i, lay = next((i, x) for i, x in layers(m) if x["kind"] == kind and "moe" in x)
        ex = union(lay["moe"]["experts"], lay["moe"]["top_k"], R, MOE_EEFF[name])
        got = (layer_cycles(m, i, lay, hw, fmt, [256 + r for r in range(R)], ex) +
               CAL["run"]) / (hw.mhz * 1e3)
        out.append(("MoE layer run ms", f"{name} {kind} R {R} zero-miss ({src})", ms, got,
                    100 * (got / ms - 1), None))
    for name, q, val, src in CARD_MOE:
        if q.startswith("decode"):
            got = moe_decode(name, hw, 128)["tok_s"]
        else:
            got = moe_prefill(name, hw, int(re.search(r"\((\d+)", q).group(1)))["ttft_s"]
        out.append(("MoE", f"{name} {q} ({src})", val, got, 100 * (got / val - 1), None))
    if COSIM_RUNS_FILE.exists():
        for x in cosim_layers(COSIM_RUNS_FILE):
            out.append(("co-sim campaign", f"{x['what']}, {x['config']}", x["measured"] / 1e6,
                        x["model"] / 1e6, x["error"], None))
    return out


def cosim_layers(path) -> list:
    """A co-sim campaign's layers (tools/scoreboard_cosim.py run, prefixes 1:2...): each pair of
    consecutive prefixes of one configuration gives the last layer's cycles, the run's own
    cost and its LM head cancelling. Returns dicts of the configuration, the layer, its
    measured and modelled cycles."""
    recs = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    recs = [r for r in recs if "cycles" in r and r.get("layers", "").isdigit()]
    key = ("model", "rows", "pos", "batch", "wformat", "head_format", "formats", "mcols",
           "act_rows", "lanes", "tmem", "mem", "S", "mhz", "bw", "lat", "mts")
    by = {}
    for r in recs:
        by.setdefault(tuple(r.get(k) for k in key), {})[int(r["layers"])] = r["cycles"]
    out = []
    for k, cyc in by.items():
        r = dict(zip(key, k))
        name = SHORT.get(r["model"])
        if not name:
            continue
        m = model(name)
        L = list(layers(m))
        hw = BOARDS["ypcb"].replace(mcols=r["mcols"], act_rows=r["act_rows"] or 0,
                                    vpu=r["lanes"] / 8, tmem_words=r["tmem"], S=r["S"],
                                    mhz=r["mhz"])
        if r["mem"] == "port":
            hw = hw.replace(gbps=1e4 * r["bw"] / 100)
        fmt = (r["wformat"], r["head_format"], r["formats"] or "")
        R, pos = r["rows"], r["pos"]
        ctxs = [pos] if R == 0 else [pos] * R if r["batch"] else [pos + i for i in range(R)]
        ns = sorted(cyc)
        for a, b in zip(ns, ns[1:]):
            got = sum(layer_cycles(m, i, L[i][1], hw, fmt, ctxs) for i in range(a, b))
            what = (f"{name} {fmt[0]} layer {a}{'-' + str(b - 1) if b - a > 1 else ''} "
                    f"({L[a][1]['kind']}) " +
                    ("decode" if R == 0 else f"{'batch' if r['batch'] else 'R'} {R}") +
                    f" at {pos}")
            cfg = ", ".join(f"{n} {r[n]}" for n, d in (("mcols", 4), ("act_rows", 0),
                                                       ("lanes", 8), ("tmem", 65536),
                                                       ("S", 1), ("mem", "ldc"))
                            if r[n] not in (d, None))
            out.append(dict(what=what, config=cfg or "ypcb", measured=cyc[b] - cyc[a],
                            model=got, error=100 * (got / (cyc[b] - cyc[a]) - 1), rec=r))
    return out


def cosim_report(path, md: bool = False) -> str:
    rows = cosim_layers(path)
    L = ["| layer | configuration | co-sim cycles | model | error |", "|---|---|---:|---:|---:|"] \
        if md else []
    for x in rows:
        L.append(f"| {x['what']} | {x['config']} | {x['measured']:,} | {x['model']:,.0f} | "
                 f"{x['error']:+.1f}% |" if md else
                 f"{x['what']:44s} {x['config']:36s} {x['measured']:11,} {x['model']:11,.0f} "
                 f"{x['error']:+6.1f}%")
    if rows:
        e = [abs(x["error"]) for x in rows]
        L += ["", f"co-sim layers: mean |error| {sum(e) / len(e):.1f}% (max {max(e):.1f}%), "
                  f"{len(e)} layers"]
    return "\n".join(L)


def validate_report(md: bool = False) -> str:
    rows = validate()
    L = []
    if md:
        L += ["| group | point | measured | model | error | leave-one-out |",
              "|---|---|---:|---:|---:|---:|"]
    for g, w, a, b, e, el in rows:
        loo = "-" if el is None else f"{el:+.1f}%"
        L.append(f"| {g} | {w} | {a:,.3f} | {b:,.3f} | {e:+.1f}% | {loo} |" if md else
                 f"{g:17s} {w:64s} {a:9.3f} {b:9.3f} {e:+6.1f}% {loo:>7s}")
    groups = {}
    for g, _, _, _, e, el in rows:
        groups.setdefault(g, []).append((abs(e), None if el is None else abs(el)))
    L.append("")
    for g, v in groups.items():
        lo = [x for _, x in v if x is not None]
        L.append(f"{g}: mean |error| {sum(x for x, _ in v) / len(v):.1f}% (max "
                 f"{max(x for x, _ in v):.1f}%)" +
                 (f", leave-one-out {sum(lo) / len(lo):.1f}% (max {max(lo):.1f}%)" if lo else "")
                 + f", {len(v)} points")
    return "\n".join(L)


# ------------------------------------------------------------------------------- output
def _f(x, nd=1):
    return "-" if x is None else f"{x:,.{nd}f}"


def render(sc: dict, md: bool = False) -> str:
    hw = HW(**sc["hw"])
    L = []

    def table(head, rows):
        if md:
            L.append("| " + " | ".join(head) + " |")
            L.append("|" + "|".join("---" if j == 0 else "---:" for j in range(len(head))) + "|")
            L.extend("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
        else:
            w = [max(len(str(c)) for c in col) for col in zip(head, *rows)]
            for r in [head] + rows:
                L.append("  ".join(str(c).ljust(w[0]) if j == 0 else str(c).rjust(w[j])
                                   for j, c in enumerate(r)))
        L.append("")
    L += [("**" if md else "") + hw.describe() + ("**" if md else ""), ""]
    s = summary(sc)
    P = max(next(iter(sc["prefill"].values())))
    U = max(next(iter(sc["batch"].values())))
    L += [f"summary (geometric means, {sc['wformat']} weights, context {sc['ctx']}): decode "
          f"{s['decode']:.1f} tok/s, prefill {P} {s['prefill']:.1f} tok/s, batched {U} users "
          f"{s['batch']:.1f} tok/s in all, MoE {s['moe']:.2f} (sqrt of decode x prefill {P} "
          f"tok/s)", ""]
    bd = bounds(sc)
    L += ["bound (the unit taking the largest share of the time, averaged over the models): " +
          ", ".join(f"{k} {bd[k]}" for k in ("decode", "prefill", "batch", "moe") if k in bd),
          ""]
    Ps = list(next(iter(sc["prefill"].values())))
    head = ["model", "decode tok/s", "MB/token"] + \
        [h for P in Ps for h in (f"prefill {P} tok/s", f"TTFT {P} s")] + ["rows a run"]
    rows = []
    head += ["decode bound", f"prefill {Ps[-1]} bound"]
    for n, d in sc["decode"].items():
        pf = sc["prefill"][n]
        rr = pf[Ps[-1]]
        rows.append([n, _f(d["tok_s"]), _f(d["mb_token"], 0)] +
                    [x for P in Ps for x in (_f(pf[P]["tok_s"]), _f(pf[P]["ttft_s"], 2))] +
                    [str(rr["rows"]) + ("" if rr["rows_last"] == rr["rows"] else
                                        f"..{rr['rows_last']}"), d["bound"], rr["bound"]])
    table(head, rows)
    Us = list(next(iter(sc["batch"].values())))
    table(["batched decode, total / each tok/s"] + [f"{u} users" for u in Us] +
          [f"{Us[-1]} users bound"],
          [[n] + [f"{v[u]['total_tok_s']:.1f} / {v[u]['user_tok_s']:.1f}" for u in Us] +
           [v[Us[-1]]["bound"]] for n, v in sc["batch"].items()])
    head = ["MoE", "experts on the card", "decode tok/s", "misses a token"] + \
        [h for P in Ps for h in (f"prefill {P} tok/s", f"TTFT {P} s")] + \
        ["decode bound", f"prefill {Ps[-1]} bound"]
    rows = []
    for n, v in sc["moe"].items():
        f, d = v["fit"], v["decode"]
        where = "all" if d["resident"] else f"{100 * f['frac']:.0f}% ({f['slots']} slots)" + \
            (f" + {f['far_slots']} far" if f["far_slots"] else "")
        rows.append([n, where, _f(d["tok_s"], 2), _f(d["misses"], 0)] +
                    [x for P in Ps for x in (_f(v["prefill"][P]["tok_s"]),
                                             _f(v["prefill"][P]["ttft_s"], 1))] +
                    [d["bound"], v["prefill"][Ps[-1]]["bound"]])
    table(head, rows)
    return "\n".join(L)


def whatifs_report(b: str = "ypcb", md: bool = False, wformat: str = "fp4") -> str:
    res = whatifs(b, wformat)
    W = ("decode", "prefill", "batch", "moe")
    head = ["what-if (against " + b + ")", "decode", "prefill", "batched", "MoE", "score",
            "bound after: decode / prefill / batched / MoE"]
    L = []
    if md:
        L += ["| " + " | ".join(head) + " |", "|---|---:|---:|---:|---:|---:|---|"]
    else:
        L.append(f"{head[0]:44s} {'decode':>8s} {'prefill':>8s} {'batched':>8s} {'MoE':>8s} "
                 f"{'score':>8s}  {head[-1]}")
    rows = [(f"({b} itself)", {k: 1.0 for k in W + ("score",)}, res[0][3])] + \
        [(label, r, bn) for label, _, r, _, bn in res]
    for label, r, bn in rows:
        cells = [f"{r[k]:.3f}x" for k in W + ("score",)]
        bd = " / ".join(bn.get(k, "-") for k in W)
        L.append("| " + " | ".join([label] + cells + [bd]) + " |" if md else
                 f"{label:44s} " + " ".join(f"{c:>8s}" for c in cells) + "  " + bd)
    return "\n".join(L)


# ------------------------------------------------------------------------------- CLI
OVERRIDES = ("S", "mcols", "act_rows", "mxu_pump", "tmem_words", "rows", "batch_rows",
             "moe_rows", "moe_ids", "kernel", "vpu", "mhz", "mem", "mts", "channels", "mem_eff", "port_bytes", "gbps",
             "pcie_gbps", "fast_gb", "imem")


def hw_from_args(a) -> HW:
    kw = {k: getattr(a, k) for k in OVERRIDES if getattr(a, k, None) is not None}
    if kw:
        kw["name"] = a.board + " + " + ", ".join(f"{k} {v}" for k, v in kw.items())
    return board(a.board, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 2)[2].split("Python")[0])
    ap.add_argument("cmd", nargs="?", default="report",
                    choices=["report", "compare", "whatifs", "validate", "fit", "boards",
                             "models", "gen-models", "cosim"])
    ap.add_argument("paths", nargs="*", help="gen-models: checkpoint directories; cosim: "
                    "tools/scoreboard_cosim.py run's JSON lines")
    ap.add_argument("--board", default="ypcb", choices=list(BOARDS))
    ap.add_argument("-S", type=int)
    for k, t in (("mcols", int), ("act-rows", int), ("mxu-pump", float), ("tmem-words", int),
                 ("rows", int), ("batch-rows", int), ("vpu", float), ("mhz", float),
                 ("mts", float), ("channels", int), ("mem-eff", float), ("port-bytes", int),
                 ("gbps", float), ("pcie-gbps", float), ("fast-gb", float), ("imem", int),
                 ("moe-ids", int)):
        ap.add_argument("--" + k, type=t)
    ap.add_argument("--mem", choices=["ddr3", "ddr4", "hbm"])
    ap.add_argument("--moe-rows", choices=["all", "gather"])
    ap.add_argument("--kernel", choices=["today", "tiled"])
    ap.add_argument("--wformat", default="fp4", choices=["fp4", "int8", "auto", "mix"])
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--models", help="comma list of short names (default: " +
                    ",".join(DENSE) + ")")
    ap.add_argument("--json")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args(argv)
    names = a.models.split(",") if a.models else None
    if a.cmd == "boards":
        for k, v in BOARDS.items():
            print(f"{k:8s} {v.note}\n         {v.describe()}")
        return
    if a.cmd == "models":
        for k, v in NAMES.items():
            m = model(k)
            print(f"{k:12s} {v:22s} {m['family']:7s} {m['n_layers']:3d} layers, hidden "
                  f"{m['hidden']:5d}, kappa {kappa(m)}")
        return
    if a.cmd == "gen-models":
        res = json.loads(MODELS_FILE.read_text()) if MODELS_FILE.exists() else {}
        for p in map(Path, a.paths):
            res[p.name] = shapes(p.name, json.loads((p / "config.json").read_text()),
                                 safetensors_headers(p))
            print(f"{p.name}: {res[p.name]['family']}, {res[p.name]['n_layers']} layers")
        MODELS_FILE.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        return
    if a.cmd == "fit":
        print(json.dumps(fit(), indent=1))
        return
    if a.cmd == "cosim":
        for p in a.paths:
            print(cosim_report(p, md=a.markdown))
        return
    if a.cmd == "validate":
        print(validate_report(md=a.markdown))
        return
    if a.cmd == "whatifs":
        print(whatifs_report(a.board, md=a.markdown, wformat=a.wformat))
        return
    hw = hw_from_args(a)
    sc = scoreboard(hw, a.wformat, a.ctx, names)
    if a.cmd == "compare":
        base = board(a.board)
        print(render(scoreboard(base, a.wformat, a.ctx, names), a.markdown))
        print(render(sc, a.markdown))
        r, bb, bn = compare(base, hw, full=True, wformat=a.wformat, ctx=a.ctx, names=names)
        print("what-if / board: " + ", ".join(f"{k} {v:.3f}x" for k, v in r.items()))
        print("bound, board -> what-if: " + ", ".join(f"{k} {bb[k]} -> {bn[k]}" for k in bn))
    else:
        print(render(sc, a.markdown))
    if a.json:
        Path(a.json).write_text(json.dumps(sc, indent=1, default=str))


if __name__ == "__main__":
    main()
