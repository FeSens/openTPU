"""Targets: one file per device, `targets/<name>.toml` (docs/targets.md).

A target describes one build of openTPU in four layers:

- arch: the `opentpu.isasim.Config` fields, what the programs depend on. Two targets with the
  same arch run the same compiled programs.
- uarch: RTL parameters that change timing only (MXU_IMPL, TMEM_IMPL, RPB / WPB, ...).
- tech: the cell family the RTL is built with (rtl/tech/<family>), the part or the PDK.
- platform: the board around the accelerator (memory, channels, host link, clock).

Everything that depends on the target comes from here: `config()` for the compiler and the
simulators, `sim_uarch()` for the Verilator builds, `mem_model()` for the co-simulation's channel
model, `rtl_params()` for the FPGA and ASIC tops, `arch_hash()` for the device's CFG_HASH register
and the host's check. `current()` is the target the environment selects (OTPU_TARGET, default
the YPCB-00338 card).

    python3 -m opentpu.target show [NAME]       the target, resolved
    python3 -m opentpu.target json [NAME]       the same as JSON (flows read it)
    python3 -m opentpu.target generics [NAME]   Vivado generics (NAME=VALUE ...)
    python3 -m opentpu.target hash NAME [K=V]   arch_hash, 8 hex digits (K: an FPGA build's
                                                MCOLS / LANES / ACT_ROWS / DSTEP knobs)
    python3 -m opentpu.target env [NAME]        shell assignments T_<PARAM>=<value> (flows)
    python3 -m opentpu.target files NAME        the RTL file list (sources())
    python3 -m opentpu.target list              the targets

NAME may name a variant, NAME:VARIANT (a target file's [variants.<VARIANT>]: arch and uarch
values over the file's; the file's [target] variant is the default): OTPU_TARGET=
ypcb-00338:m8, make -C boards/ypcb-00338 bit TARGET=ypcb-00338:m8.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:                      # Python 3.10
    import tomli as tomllib                      # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ROOT / "targets"
DEFAULT = "ypcb-00338"
KINDS = ("fpga", "asic", "sim")
FAMILIES = ("generic", "xilinx7", "xilinxus", "asic")
# uarch keys every target names (timing only; the RTL's parameters of the same names)
UARCH_KEYS = ("MXU_IMPL", "MXU_CL", "VPU_CL", "ULANES", "TMEM_IMPL", "RPB", "WPB", "WIN",
              "FIFO_DEPTH", "TRACE_DEPTH", "PQ_WIN", "CHASH")
# and the optional ones (absent: the RTL's default): MXU_E, the MXU's epilogue lanes (0: MCOLS / 2;
# above MCOLS: MCOLS); QUANT_R1, RSCALE's row factor read once a row (otpu_quant R1); QUANT_PW,
# a buffered QACT pass 1's chunks a cycle (otpu_quant PW; 1: absent); SE_R, the streams on an
# engine of their own with SE_R rows a cycle (otpu_slice, otpu_se_eng; 0: the VPU's engine;
# docs/se_engine.md); MXU_QD, the MXU's commands queued (otpu_mxu QD, 2..7; 2: absent;
# docs/native_channels.md); MXU_RF_ROWS, the MXU's result FIFO rows (otpu_mxu RF_ROWS: a power of
# two >= 16, fewer than its rule's throttling one-advance rows; 0: absent, the rule); DMA_DEPTH,
# the DMA's LD chunk buffer, a DSTEP's read-ahead (otpu_dma DEPTH, a power of two; absent: 128);
# DMA_LDQ, the DMA's commands in flight (otpu_dma LDQ: 1, absent; 2, an LD's DRAM reads under
# the command before it, docs/dma_ldq.md), timing only;
# with [arch] ICACHE the instruction cache's lines of 32 instructions, IC_LINES (otpu_seq; a power
# of two, absent: 16), and the lines its prefetcher fills ahead, IC_PF (1 .. 4, absent: 2);
# VPU_CNS, the composites' stages per lane (otpu_vpu CNS: 3, absent; 5, EXP2 / EXP2SUB / RSQRT /
# LOG2 in two passes, bit-exact); MXU_PUMP, the MXU's array pumped on the 2x clock, two activation
# rows a pass (otpu_mxu PUMP, WG > 1: 1; 0, absent), bit-exact; QUANT_SWW, the quantizer's SW port
# words: QST's pass 1 in rows mode 4 x QUANT_SWW bytes a cycle (otpu_quant SWW: 1, absent; 4 with
# 4 x 4 <= ULANES x 2 and <= ULANES x QUANT_PW), bit-exact; QUANT_QSTB, QST rows in ROW mode through
# the quantizer's row buffer, pass 1 under the next rows' pass 0 (otpu_quant QSTB: 0, absent; 1
# with 4 x QUANT_SWW = ULANES x QUANT_PW), bit-exact
UARCH_OPT = ("MXU_E", "QUANT_R1", "QUANT_PW", "SE_R", "MXU_QD", "MXU_RF_ROWS", "DMA_DEPTH",
             "DMA_LDQ", "IC_LINES", "IC_PF", "VPU_CNS", "MXU_PUMP", "QUANT_SWW", "QUANT_QSTB")
# with TMEM_IMPL 3 the one copy's knobs (otpu_slice TMEM_<K>, docs/tmem_nocopy.md;
# vivado/tmem_knobs.tcl's TMEM_X)
TMEM3_KEYS = ("TMEM_NSB", "TMEM_SBP", "TMEM_SBH", "TMEM_RPL", "TMEM_TDP", "TMEM_GST", "TMEM_PMP",
              "TMEM_PRK", "TMEM_SDM")
# and otpu_native_dram's per-channel depths in beats (otpu_board / otpu_top NAT_*: RD, AD, XD, WQD,
# APF; absent: the adapter's defaults, the cards'): B's, A's and W's reads in flight, the SW queue,
# A's read runs; NAT_FILL (SW_FILL: a partial SW beat read and written whole, 1, or written with
# its byte mask, 0: a memory that takes masks); NAT_WPOST (WPOST: B's and SW's write data
# posted to the channel at the take, the WG targets' bridges; needs NAT_FILL 0); NAT_SWG (SWG, with
# NAT_WPOST: one SW gather beat for every channel and one posted beat a cycle on one bus, the
# the WG targets' area cut; absent: a beat a channel); and NAT_XPOST (XPOST:
# port W's beats posted into the MXU's chunk FIFO by the channels, the WG targets' bridges; needs WG > 1
# and a D chunk one beat; docs/native_channels.md); NAT_XSTG, register stages each way between a
# slice and its adapter (otpu_xstg: a floorplan with the adapters an SLR from their slices;
# absent: 0, wired), and NAT_XSTG_P the posted W beats' and their counts' (absent: NAT_XSTG's);
# NAT_RHB / NAT_WHB, the channels' read / write data in half beats (otpu_native_dram and the WG targets'
# bridges RHB / WHB, R1 / R2: NAT_RHB needs NAT_XPOST, NAT_WHB NAT_WPOST and NAT_SWG, both beat_bytes
# 128; absent: whole beats, the cards'). None of them is in the Config (arch_hash): the programs
# do not depend on them
NAT_KEYS = ("NAT_RD", "NAT_AD", "NAT_XD", "NAT_WQD", "NAT_APF", "NAT_FILL", "NAT_WPOST",
            "NAT_SWG", "NAT_XPOST", "NAT_XSTG", "NAT_XSTG_P", "NAT_RHB", "NAT_WHB")
# and the RAM styles (strings; otpu_board / otpu_top parameters of the same names): TMEM_STYLE,
# the pumped one copy's RAMs (TMEM_IMPL 3, TMEM_PMP 1; "ultra": UltraRAM; absent: "block");
# IMEM_STYLE, IMEM's and its wait entries' (otpu_seq STYLE; absent: "auto", the tool's pick);
# IMEM_WC_STYLE, the wait entries' alone (otpu_seq WSTYLE; absent: IMEM_STYLE's);
# MXU_RF_STYLE, the MXU's result FIFO's copies (otpu_mxu RF_STYLE; absent: "block", read ahead;
# "distributed": LUT RAM read at its address, the WG targets': a 64-row copy fills an eighth of its block
# RAMs, and the read ahead's bypass costs more LUTs than the LUT RAM took); MXU_FIFO_STYLE, the
# MXU's chunk FIFO lanes (otpu_mxu FIFO_STYLE; absent: "block"; "ultra": UltraRAM, at NAT_XPOST
# each lane's posted beats staged across the clocks in LUT RAM)
STYLE_KEYS = ("TMEM_STYLE", "IMEM_STYLE", "IMEM_WC_STYLE", "MXU_RF_STYLE", "MXU_FIFO_STYLE")
# and the multi-slice board's crossings (otpu_board parameters of the same names, S > 1; absent:
# the board's defaults): COLL_PIPE, the collective hub's reads in flight as many as the round
# trip through the crossings takes (otpu_coll_hub NR > 1, and the writes' credits with them; 0:
# one read at a time); XRS, the register stages each way between a slice and the control block /
# the collective unit (otpu_xr, >= 2). The board model's from the target (opentpu/host/board.py),
# and sim_uarch's with COLL_HUB (otpu_top's hub, the co-simulation's); OTPU_COLL_PIPE / OTPU_XRS
# override both
COLL_KEYS = ("COLL_PIPE", "XRS")
STYLES = ("block", "ultra")
KEY_STYLES = {"MXU_RF_STYLE": ("block", "distributed")}    # (the others: STYLES)
# and the RAMs' split between block RAM and UltraRAM (integers; otpu_board / otpu_top parameters of
# the same names; absent: 0): ACT_UCOLS, the ACT RAM's columns in UltraRAM (otpu_actram UCOLS,
# 0 .. MCOLS)
URAM_KEYS = ("ACT_UCOLS",)
# arch keys a target may leave out, with their value then (the machine before the field)
ARCH_DEFAULTS = {"WG": 1, "OUTER": True, "DSTEP2": False, "KT": False, "AW": 32, "BHW": 0,
                 "ICACHE": False, "SEXACT": False, "ST16": "", "FSUM": False, "VSIG": False,
                 "VMAC": False, "KVSPLIT": 0, "SPLITQ": False, "ST16S": False, "QHL": False,
                 "QSTHL": False, "OUTER1": False, "SPLMM": False, "DNCHUNK": 0, "QONCE": False,
                 "DNMERGE": 0}
ST16_CODES = {"": 0, "e6m9": 1, "bf16": 2}    # [arch] ST16: otpu_dma ST16 (isa.ST16_CODE)
AW_MAX = 40                     # DRAM address bits a target may state (Config.AW: 32 .. AW_MAX)
# Config fields a target does not state: config() derives them (CHASH: the memory's channel hash,
# [uarch] CHASH, as the group chunks' row order at WG > 1; GALL: WG > 1, the MXU at WG > 1 runs
# every MM in groups, docs/weight_groups.md; SE_OWN: [uarch] SE_R > 0 with DSTEP, the streams
# on an engine of their own, isasim.se_own)
DERIVED = ("CHASH", "GALL", "SE_OWN", "WC_WIN")
PLATFORM_KEYS = ("board", "core_mhz", "mem", "channels_per_slice", "channel_bytes", "beat_bytes",
                 "host")
# The slice's TMEM read ports (otpu_slice NRP) and the most lanes a simulation may set
# (OTPU_LANES=16): RPB = 0 ("every read lane": TMEM replicated per read port) simulates as
# NRP x 16, the board build as NRP x LANES; both select the slice's write-mask arbiter.
NRP = 8
SIM_LANES_MAX = 16


# The RTL, as file lists under rtl/ in compile order: the core (no vendor cell, no FPGA-only
# memory), a cell family, the FPGA boards' shell (control block, trace, the native DRAM adapter,
# the channel bridges) and a board's top. sources() puts a target's together; the flows take
# theirs from it or are checked against it (tests/test_target.py: rtlsim.RTL_SOURCES, the board
# Makefile, create_project.tcl, tools/synth).
CORE = ["vpu/otpu_fp.sv", "vpu/otpu_fpipe.sv", "top/otpu_pkg.sv", "mem/otpu_tmem.sv",
        "mem/otpu_actram.sv", "seq/otpu_seq.sv", "vpu/otpu_vtree.sv", "dma/otpu_dma.sv",
        "mxu/otpu_mxu.sv", "vpu/otpu_quant.sv", "vpu/otpu_se_comp.sv", "vpu/otpu_se_tail.sv",
        "vpu/otpu_vpu.sv", "top/otpu_coll.sv", "top/otpu_slice.sv"]
TECH = {"generic": ["tech/generic/otpu_tech_dsp.sv", "tech/generic/otpu_tech_ram.sv"],
        "xilinx7": ["tech/xilinx7/otpu_tech_dsp.sv", "tech/generic/otpu_tech_ram.sv"],
        "xilinxus": ["tech/xilinxus/otpu_tech_dsp.sv", "tech/generic/otpu_tech_ram.sv"],
        # the RAMs on SRAM macros: the leaves (otpu_sram_*) come from tools/asic/flow.py
        "asic": ["tech/generic/otpu_tech_dsp.sv", "tech/asic/otpu_tech_ram.sv"]}
SHELL = ["shell/otpu_ctrl.sv", "shell/otpu_trace.sv", "shell/otpu_board.sv",
         "mem/otpu_native_dram.sv", "mem/otpu_xstg.sv", "shell/otpu_afifo.sv",
         "shell/otpu_mem_ch.sv",
         "shell/otpu_axi_split2.sv"]
BOARDS = {"ypcb-00338": ["boards/ypcb-00338/otpu_native_sys.sv",
                         "boards/ypcb-00338/otpu_fpga_top_ld.sv"],
          "bcu1525": ["boards/bcu1525/otpu_bcu_sys.sv", "boards/bcu1525/otpu_bmc_watch.sv",
                      "boards/bcu1525/otpu_bmc_uart.sv", "boards/bcu1525/otpu_bcu_gov.sv",
                      "boards/bcu1525/otpu_axil_off.sv", "boards/bcu1525/otpu_axi_off.sv",
                      "boards/bcu1525/otpu_fpga_top_bcu.sv"],
          # the BCU1525 at WG 8 (boards/bcu1525/vivado/create_project_wg.tcl): the channel
          # bridges and register slices, the DIMMs' pairs and hmuxes, the governor and the
          # BMC's bridge
          "bcu1525-wg": ["mem/otpu_hbm_ch.sv", "mem/otpu_axi_rs.sv",
                         "boards/bcu1525/otpu_mig_pair.sv", "boards/bcu1525/otpu_mig_hmux.sv",
                         "boards/bcu1525/otpu_bcu_wg_sys.sv",
                         "boards/bcu1525/otpu_bmc_watch.sv", "boards/bcu1525/otpu_bmc_uart.sv",
                         "boards/bcu1525/otpu_bcu_gov.sv", "boards/bcu1525/otpu_axil_off.sv",
                         "boards/bcu1525/otpu_axi_off.sv",
                         "boards/bcu1525/otpu_fpga_top_bcu_wg.sv"]}


class TargetError(ValueError):
    pass


@dataclass(frozen=True)
class Target:
    name: str
    kind: str
    platform_id: int
    description: str
    arch: dict
    uarch: dict
    tech: dict
    platform: dict
    path: Path
    variant: str = ""                   # the [variants.<name>] applied ("": none)

    @property
    def family(self) -> str:
        return self.tech["family"]

    @property
    def core_khz(self) -> int:
        """The target clock in kHz (an FPGA's MMCM may round it: create_project.tcl)."""
        return round(self.platform["core_mhz"] * 1000)

    @property
    def act_rows(self) -> int:
        return self.arch["ACT_ROWS"] or self.arch["MCOLS"]

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("name", "variant", "kind", "platform_id",
                                          "description", "arch", "uarch", "tech", "platform")}
        d["arch_hash"] = f"{arch_hash(config(self)):08x}"
        return d


def names() -> list[str]:
    return sorted(p.stem for p in TARGETS.glob("*.toml"))


def _config_fields() -> dict:
    from .isasim import Config
    return {f.name: f for f in dataclasses.fields(Config)}


@lru_cache(maxsize=None)
def load(name: str | None = None) -> Target:
    """The target `name` (a name in targets/ or a path to a .toml file; None: current()), with
    a variant: `name:variant` (a [variants.<variant>] table of the file: arch and uarch values
    over the file's), else the file's own [target] variant, if it names one."""
    if name is None:
        return current()
    name, _, var = str(name).partition(":")
    p = Path(name)
    path = p if p.suffix == ".toml" and p.exists() else TARGETS / f"{name}.toml"
    if not path.exists():
        raise TargetError(f"no target {name!r}: targets/ has {', '.join(names())}")
    with path.open("rb") as f:
        raw = tomllib.load(f)
    t = raw.get("target", {})
    for sec in ("target", "arch", "uarch", "tech", "platform"):
        if sec not in raw:
            raise TargetError(f"{path}: no [{sec}] section")
    variants = raw.get("variants", {})
    base = t.get("variant", "")         # the file states its default variant's values
    if base:
        if base not in variants:
            raise TargetError(f"{path}: [target] variant {base!r} has no [variants.{base}]")
        diff = sorted(k for k, v in variants[base].items()
                      if raw["arch"].get(k, raw["uarch"].get(k)) != v)
        if diff:
            raise TargetError(f"{path}: [arch] / [uarch] {diff} differ from the default variant "
                              f"{base!r}'s")
    var = var or base
    if var:
        if var not in variants:
            raise TargetError(f"{path}: no variant {var!r} (its variants: "
                              f"{', '.join(sorted(variants)) or 'none'})")
        raw["arch"], raw["uarch"] = dict(raw["arch"]), dict(raw["uarch"])
        for k, v in variants[var].items():
            if k in raw["arch"] or k in ARCH_DEFAULTS:
                raw["arch"][k] = v
            elif k in raw["uarch"] or k in UARCH_KEYS + UARCH_OPT + TMEM3_KEYS + NAT_KEYS + \
                    STYLE_KEYS + URAM_KEYS + COLL_KEYS:
                raw["uarch"][k] = v
            else:
                raise TargetError(f"{path}: variant {var!r}: {k} is not an [arch] or [uarch] key")
    cf = _config_fields()
    raw["arch"] = ARCH_DEFAULTS | raw["arch"]
    bad = sorted(set(raw["arch"]) - (set(cf) - set(DERIVED)))
    if bad:
        raise TargetError(f"{path}: [arch] keys {bad} are not Config fields a target states")
    missing = sorted(set(cf) - set(raw["arch"]) - set(DERIVED))
    if missing:
        raise TargetError(f"{path}: [arch] lacks {missing} (every Config field is stated)")
    missing = [k for k in UARCH_KEYS if k not in raw["uarch"]]
    if missing:
        raise TargetError(f"{path}: [uarch] lacks {missing}")
    missing = [k for k in PLATFORM_KEYS if k not in raw["platform"]]
    if missing:
        raise TargetError(f"{path}: [platform] lacks {missing}")
    if t.get("kind") not in KINDS:
        raise TargetError(f"{path}: kind {t.get('kind')!r} not in {KINDS}")
    if raw["tech"].get("family") not in FAMILIES:
        raise TargetError(f"{path}: tech family {raw['tech'].get('family')!r} not in {FAMILIES}")
    if t.get("name", path.stem) != path.stem:
        raise TargetError(f"{path}: name {t.get('name')!r} differs from the file's")
    a = raw["arch"]
    if a["S"] < 1 or a["D"] % 32 or a["MCOLS"] < 1 or a["LANES"] & (a["LANES"] - 1):
        raise TargetError(f"{path}: bad arch (S >= 1, D a multiple of 32, LANES a power of two)")
    wg = a["WG"]
    if wg < 1 or wg & (wg - 1) or wg > a["MCOLS"]:
        raise TargetError(f"{path}: WG {wg}: a power of two, at most MCOLS (a group's rows each "
                          f"take a column)")
    if not 2 <= raw["uarch"].get("MXU_QD", 2) <= 7:
        raise TargetError(f"{path}: MXU_QD {raw['uarch']['MXU_QD']}: 2..7 (otpu_mxu QD)")
    rf = raw["uarch"].get("MXU_RF_ROWS", 0)
    if rf and (rf < 16 or rf & (rf - 1)):
        raise TargetError(f"{path}: MXU_RF_ROWS {rf}: 0, or a power of two >= 16 "
                          f"(otpu_mxu RF_ROWS)")
    if a["ICACHE"] and (a["D"] > 128 or not 1 <= raw["uarch"].get("IC_PF", 2) <= 4
                        or raw["uarch"].get("IC_LINES", 16) & (raw["uarch"].get("IC_LINES", 16) - 1)
                        or a["IMEM_WORDS"] // 256 <= raw["uarch"].get("IC_LINES", 16)):
        raise TargetError(f"{path}: ICACHE (otpu_seq) needs D <= 128, IC_LINES a power of two "
                          f"below the longest program's lines (IMEM_WORDS / 256), IC_PF 1 .. 4")
    if a["ST16"] not in ST16_CODES or (a["ST16"] and not (a["DSTEP"] and a["STREAM"]
                                                          and a["D"] <= 128)):
        raise TargetError(f"{path}: ST16 {a['ST16']!r}: one of {sorted(ST16_CODES)}, with DSTEP "
                          f"and STREAM (the DMA's 16-bit state, otpu_dma) and D <= 128 (a row "
                          f"of 64 columns whole chunks)")
    if a["FSUM"] and not (a["DSTEP"] and a["LANES"] == 8 and a["D"] >= 32):
        raise TargetError(f"{path}: FSUM (EXP2SUB's row sums, otpu_vpu) needs the composites on "
                          f"the stream engine's path: DSTEP, LANES 8, D >= 32")
    if a["QHL"] and not (a["SPLITQ"] and a["SEQ_WC"]):
        raise TargetError(f"{path}: QHL (QACT flag HL, otpu_quant) needs SPLITQ and SEQ_WC (the "
                          f"wait entries cover its lo blocks; the footprint scoreboard does not)")
    if a["QSTHL"] and not (a["SPLITQ"] and a["SEQ_WC"] and a["WG"] > 1
                           and raw["uarch"].get("QUANT_QSTB")):
        raise TargetError(f"{path}: QSTHL (QST HL, otpu_quant) needs SPLITQ, SEQ_WC (the wait "
                          f"entries cover its lo part), GRP (WG > 1, GALL) and the QST row buffer "
                          f"(uarch QUANT_QSTB)")
    if a["QONCE"] and not a["SEQ_WC"]:
        raise TargetError(f"{path}: QONCE (QACT flag ST, LDA) needs SEQ_WC (the wait entries "
                          f"cover their DRAM and ACT RAM ranges; the footprint scoreboard does "
                          f"not)")
    if a["QONCE"]:
        u = raw["uarch"]
        al = int(u.get("ULANES", a["LANES"])) * int(u.get("QUANT_PW", 1))
        w = min(a["D"] // 4, a["LANES"])
        if not (int(u.get("QUANT_SWW", 1)) * 4 == al and al >= 8 and (4 * w) % al == 0
                and 2 <= w <= 16):
            raise TargetError(f"{path}: QONCE needs the quantizer's SW port as wide as its ACT "
                              f"writes (4 x QUANT_SWW = ULANES x QUANT_PW, at least 8: otpu_quant "
                              f"QONCE) and the DMA's beat a multiple of them (otpu_dma LDA)")
    if a["OUTER1"] and (a["OUTER"] or not a["VMAC"]):
        raise TargetError(f"{path}: OUTER1 (VOP OUTER decay one, otpu_vpu OUTER1) is OUTER's "
                          f"decay one on MAC's datapath: VMAC on, OUTER off")
    if a["SPLMM"] and not (a["SEQ_WC"] and a["WG"] > 1):
        raise TargetError(f"{path}: SPLMM (MM SPL, otpu_mxu SPL) needs SEQ_WC (the wait entries "
                          f"cover its lo part) and GRP (WG > 1, GALL)")
    if a["VSIG"] and not (a["DSTEP"] and a["LANES"] == 8 and a["D"] >= 32):
        raise TargetError(f"{path}: VSIG (VOP SIGMOID / SILU, otpu_se_comp SIG) needs the "
                          f"composites on the stream engine's path: DSTEP, LANES 8, D >= 32")
    if raw["uarch"].get("DMA_LDQ", 1) not in (1, 2):
        raise TargetError(f"{path}: DMA_LDQ {raw['uarch']['DMA_LDQ']}: 1, or 2 (otpu_dma LDQ)")
    if raw["arch"]["SEQ_WC"] and 16 < raw["uarch"]["WIN"] != 32:
        raise TargetError(f"{path}: SEQ_WC (otpu_seq's wait entries) needs WIN <= 16, or 32 "
                          f"(Config.WC_WIN: the 4-bit counts saturate at 15)")
    for k in STYLE_KEYS:
        if raw["uarch"].get(k, "block") not in KEY_STYLES.get(k, STYLES):
            raise TargetError(f"{path}: {k} {raw['uarch'][k]!r} not in {KEY_STYLES.get(k, STYLES)}")
    if not 0 <= raw["uarch"].get("ACT_UCOLS", 0) <= a["MCOLS"]:
        raise TargetError(f"{path}: ACT_UCOLS {raw['uarch']['ACT_UCOLS']}: the ACT RAM's columns "
                          f"in UltraRAM, 0 .. MCOLS")
    if raw["uarch"].get("XRS", 2) < 2:
        raise TargetError(f"{path}: XRS {raw['uarch']['XRS']}: the crossings' stages each way, "
                          f">= 2 (one on each side)")
    if raw["uarch"].get("NAT_WPOST") and raw["uarch"].get("NAT_FILL", True):
        raise TargetError(f"{path}: NAT_WPOST needs NAT_FILL false (a fill read's data is merged "
                          f"as its write goes out)")
    if raw["uarch"].get("NAT_SWG") and not raw["uarch"].get("NAT_WPOST"):
        raise TargetError(f"{path}: NAT_SWG needs NAT_WPOST (one posted beat a cycle on one bus)")
    if wg > 1 and a["PAIR"]:
        raise TargetError(f"{path}: WG > 1 needs PAIR false (the MXU runs every MM in groups, "
                          f"GALL, and groups take no PAIR)")
    if raw["uarch"].get("NAT_XPOST") and (
            t.get("kind") != "fpga" or wg == 1 or raw["platform"].get("beat_bytes") != a["D"]):
        raise TargetError(f"{path}: NAT_XPOST (port W's beats posted into the MXU's chunk FIFO) "
                          f"needs an FPGA target with WG > 1 and beat_bytes = D")
    if raw["uarch"].get("NAT_RHB") and (not raw["uarch"].get("NAT_XPOST")
                                         or raw["platform"].get("beat_bytes") != 128):
        raise TargetError(f"{path}: NAT_RHB (read data in half beats) needs NAT_XPOST (only ports "
                          f"A's and B's reads come back on n_rdata) and beat_bytes 128")
    if raw["uarch"].get("NAT_WHB") and (not raw["uarch"].get("NAT_SWG")
                                         or raw["platform"].get("beat_bytes") != 128):
        raise TargetError(f"{path}: NAT_WHB (write data in half beats) needs NAT_WPOST and NAT_SWG "
                          f"(the posted beat on one bus) and beat_bytes 128")
    aw, nb = a["AW"], a["DRAM_BYTES"]
    if not 32 <= aw <= AW_MAX:
        raise TargetError(f"{path}: AW {aw}: DRAM addresses of 32 to {AW_MAX} bits")
    if nb > 1 << aw:
        raise TargetError(f"{path}: DRAM_BYTES over {(1 << aw) >> 30} GiB: DRAM addresses are AW "
                          f"{aw} bits per slice ([arch] AW, docs/targets.md)")
    if nb < 1 or nb & (nb - 1):
        raise TargetError(f"{path}: DRAM_BYTES {nb}: a power of two (CFG_DRAM holds its log2)")
    bhw = a["BHW"]
    if bhw and (t["kind"] != "fpga" or not 1 << bhw < nb
                or bhw - (raw["platform"]["channels_per_slice"] - 1).bit_length() < 17):
        raise TargetError(f"{path}: BHW {bhw}: the bank-half window (otpu_mig_pair) needs an FPGA "
                          f"target, less than the slice's DRAM_BYTES and at least 2^17 bytes a "
                          f"channel (the MIG map's bank-high bit 16 below its top)")
    pl = raw["platform"]
    nc = pl.get("cards", 1)         # cards in tensor parallel, each its S / cards slices (cards())
    if nc < 1 or a["S"] % nc:
        raise TargetError(f"{path}: cards {nc}: S {a['S']} slices over whole cards, the same "
                          f"number on each")
    if t["kind"] == "fpga":         # the native adapter's channels (otpu_native_dram NCH, CB)
        nch, cb = pl["channels_per_slice"], pl["beat_bytes"]
        if nch < 2 or nch & (nch - 1) or cb not in (64, 128) or a["D"] % cb or (nch * cb) % a["D"]:
            raise TargetError(f"{path}: channels_per_slice {nch}, beat_bytes {cb}: the native "
                              f"adapter takes a power of two of at least 2 channels and beats of "
                              f"64 or 128 bytes, a D chunk whole beats, a beat on every channel "
                              f"whole chunks")
        if nch * pl["channel_bytes"] != a["DRAM_BYTES"]:
            raise TargetError(f"{path}: {nch} channels x {pl['channel_bytes']} bytes is not the "
                              f"slice's DRAM_BYTES {a['DRAM_BYTES']}")
    # (after the channels' own checks) a group step is one beat on every channel (port W), a
    # row a beat (port W's lane c column c's: docs/isa.md "Weight groups", CHASH's row order)
    if wg > 1 and (wg * a["D"] != raw["platform"]["channels_per_slice"] *
                   raw["platform"].get("beat_bytes", 64) or
                   raw["platform"].get("beat_bytes", 64) != a["D"]):
        raise TargetError(f"{path}: WG x D = {wg * a['D']} bytes a step, not one beat of D bytes "
                          f"on every channel (channels_per_slice x beat_bytes, beat_bytes = D)")
    return Target(name=path.stem, kind=t["kind"], platform_id=int(t.get("platform_id", 0)),
                  description=t.get("description", ""), arch=dict(a), uarch=dict(raw["uarch"]),
                  tech=dict(raw["tech"]), platform=dict(raw["platform"]), path=path,
                  variant=var)


@lru_cache(maxsize=None)
def _variants(path: Path) -> tuple[tuple[str, ...], str]:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return tuple(sorted(raw.get("variants", {}))), raw.get("target", {}).get("variant", "")


def variants(t: Target | str) -> tuple[str, ...]:
    """The variants a target's file states ([variants.<name>]: load(f"{name}:{variant}"))."""
    t = t if isinstance(t, Target) else load(t)
    return _variants(t.path)[0]


def full_name(t: Target) -> str:
    """The target's name with its variant, load's NAME:VARIANT (the file's own default variant,
    its [target] variant: the name alone)."""
    return f"{t.name}:{t.variant}" if t.variant and t.variant != _variants(t.path)[1] else t.name


def cards(t: Target | str | None = None) -> int:
    """The cards the target's slices span ([platform] cards, 1 unless stated): global slice s on
    card s // (S / cards), its local slice s % (S / cards). Each card's build has S / cards
    slices (its CFG_SLICES); one compile at S, each card loading its slices' programs and DRAM
    (opentpu.host.cards)."""
    t = load(t) if t is None or isinstance(t, str) else t
    return t.platform.get("cards", 1)


def by_platform(pid: int) -> Target | None:
    """The target whose platform_id a device reports (CFG_PLATFORM), if this tree has it (its
    file's own values: a variant shares its platform id, opentpu.host.board check_target)."""
    for n in names():
        t = load(n)
        if t.platform_id == pid:
            return t
    return None


def current() -> Target:
    """The target OTPU_TARGET names (default the YPCB-00338 card)."""
    return load(os.environ.get("OTPU_TARGET") or DEFAULT)


def config(t: Target | str | None = None, **kw):
    """The target's `Config` (the hardware's truth: its optional features on; CHASH its memory's
    at WG > 1, GALL at WG > 1, SE_OWN its SE_R's), `kw` overriding fields."""
    from .isasim import Config, se_own
    t = t if isinstance(t, Target) else load(t)
    a = {**t.arch, **kw}
    a.setdefault("CHASH", a["WG"] > 1 and bool(t.uarch["CHASH"]))
    a.setdefault("GALL", a["WG"] > 1)
    a.setdefault("SE_OWN", se_own(t.uarch.get("SE_R", 0), a["DSTEP"], a["LANES"], a["D"]))
    a.setdefault("WC_WIN", 32 if a["SEQ_WC"] and t.uarch["WIN"] > 16 else 16)
    return Config(**a)


# The numerics the programs and images assume, hashed with the Config: "mx", the MX MXU and
# quantizer (docs/isa.md MM, QACT, QST; docs/mx_mxu.md; CAPS bit29). A host of the previous
# numerics computes the Config's hash without it, so it refuses an MX bitstream wherever it
# checks the CFG_HASH (opentpu/host/board.py device_config: ConfigMismatch), and an MX host
# refuses a bitstream of the previous numerics the same way.
NUMERICS = "mx"


def arch_hash(cfg) -> int:
    """A stable 32-bit hash of a Config's fields and the numerics (the CFG_HASH register;
    programs compiled for one Config run on any device whose Config hashes the same). SEQ_WC
    (EXPERIMENTAL, the sequencer's wait entries) is in it only when on: the hash of every
    build without it stays, and a host that writes no entries refuses a build with it (its
    hash leaves it out: opentpu/host/board.py check_hash). WG (weight groups) is in it only
    above 1, so every machine before the field hashes as before, CHASH (the group chunks'
    row order) and GALL (EXPERIMENTAL) only at WG > 1 with them on, OUTER only when off (a VPU
    without VOP OUTER), DSTEP2 only when on (DSTEP's pair) and SE_OWN only when on (the streams
    on an engine of their own: the programs' DeltaNet bursts), KT only when on (MM's K tail),
    AW (DRAM address bits) only above 32, BHW (the bank-half window) only when on, WC_WIN (the
    window the wait entries assume) only above 16, ICACHE (IMEM an instruction cache: IMEM_WORDS
    the longest program) only when on, ST16 (DSTEP's state in 16 bits, its format) only when
    on, FSUM (EXP2SUB's fused row sums) only when on, SPLITQ (QACT / QST flag LO) only when
    on, ST16S (STREAM's ST16R / ST16W) only when on, QHL (QACT flag HL) and QSTHL (QST HL)
    only when on, VSIG
    (VOP SIGMOID / SILU), VMAC (VOP MAC), OUTER1 (VOP OUTER decay one), QONCE (QACT ST,
    LDA) and SPLMM (MM SPL) only when on, SEXACT, KVSPLIT, DNCHUNK and DNMERGE never
    (compile choices, the same machine)."""
    d = dict(dataclasses.asdict(cfg), NUMERICS=NUMERICS)
    d.pop("SEXACT", None)
    d.pop("KVSPLIT", None)
    d.pop("DNCHUNK", None)
    d.pop("DNMERGE", None)
    if d.get("AW", 32) == 32:
        d.pop("AW", None)
    if not d.get("SEQ_WC"):
        d.pop("SEQ_WC", None)
    if not d.get("ICACHE"):
        d.pop("ICACHE", None)
    if not d.get("ST16"):
        d.pop("ST16", None)
    if not d.get("FSUM"):
        d.pop("FSUM", None)
    if not d.get("SPLITQ"):
        d.pop("SPLITQ", None)
    if not d.get("ST16S"):
        d.pop("ST16S", None)
    if not d.get("QHL"):
        d.pop("QHL", None)
    if not d.get("QSTHL"):
        d.pop("QSTHL", None)
    if not d.get("VSIG"):
        d.pop("VSIG", None)
    if not d.get("VMAC"):
        d.pop("VMAC", None)
    if not d.get("OUTER1"):
        d.pop("OUTER1", None)
    if not d.get("QONCE"):
        d.pop("QONCE", None)
    if not d.get("SPLMM"):
        d.pop("SPLMM", None)
    if not d.get("GALL") or d.get("WG", 1) == 1:
        d.pop("GALL", None)
    if d.get("OUTER", True):
        d.pop("OUTER", None)
    if not d.get("DSTEP2"):
        d.pop("DSTEP2", None)
    if not d.get("SE_OWN"):
        d.pop("SE_OWN", None)
    if not d.get("KT"):
        d.pop("KT", None)
    if not d.get("BHW"):
        d.pop("BHW", None)
    if d.get("WC_WIN", 16) == 16:
        d.pop("WC_WIN", None)
    if not d.get("CHASH") or d.get("WG", 1) == 1:
        d.pop("CHASH", None)
    if d.get("WG", 1) == 1:
        d.pop("WG", None)
    s = json.dumps({k: (int(v) if isinstance(v, bool) else v) for k, v in sorted(d.items())},
                   separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big")


def sim_uarch(t: Target | str | None = None) -> dict:
    """The Verilator builds' micro-architecture for `t` (rtlsim.BOARD_UARCH): the keys the
    simulation top takes. RPB 0 (every read lane) is NRP x 16, as for any LANES <= 16."""
    t = t if isinstance(t, Target) else load(t)
    u = t.uarch
    d = {"WIN": u["WIN"], "RPB": u["RPB"] or NRP * SIM_LANES_MAX, "WPB": u["WPB"],
         "FIFO_DEPTH": u["FIFO_DEPTH"]}
    if u["TMEM_IMPL"]:
        d["TMEM_IMPL"] = u["TMEM_IMPL"]
    if u["TMEM_IMPL"] == 3:                     # the one copy's knobs the target states
        d.update({k: int(u[k]) for k in TMEM3_KEYS if k in u})
    # the RAM styles as Verilator -G values (SystemVerilog string literals)
    d.update({k: f'"{u[k]}"' for k in STYLE_KEYS if k in u})
    d.update({k: int(u[k]) for k in URAM_KEYS if u.get(k)})
    if u.get("MXU_E"):
        d["MXU_E"] = u["MXU_E"]
    if u.get("QUANT_R1"):
        d["QUANT_R1"] = 1
    if u.get("QUANT_PW", 1) > 1:
        d["QUANT_PW"] = int(u["QUANT_PW"])
    if u.get("QUANT_SWW", 1) > 1:
        d["QUANT_SWW"] = int(u["QUANT_SWW"])
    if u.get("QUANT_QSTB"):
        d["QUANT_QSTB"] = 1
    if u.get("SE_R"):
        d["SE_R"] = u["SE_R"]
    if u.get("DMA_DEPTH"):
        d["DMA_DEPTH"] = int(u["DMA_DEPTH"])
    if u.get("DMA_LDQ", 1) != 1:
        d["DMA_LDQ"] = int(u["DMA_LDQ"])
    if u.get("MXU_QD", 2) != 2:
        d["MXU_QD"] = int(u["MXU_QD"])
    if u.get("MXU_RF_ROWS"):
        d["MXU_RF_ROWS"] = int(u["MXU_RF_ROWS"])
    if u.get("VPU_CNS", 3) != 3:
        d["VPU_CNS"] = int(u["VPU_CNS"])
    if u.get("MXU_PUMP"):
        d["MXU_PUMP"] = 1
    if t.arch["SEQ_WC"]:
        d["SEQ_WC"] = 1
    if t.arch["FSUM"]:                          # otpu_vpu's fused row sums
        d["VPU_FSUM"] = 1
    if t.arch["SPLITQ"]:                        # otpu_quant's LO flag
        d["QUANT_SPLIT"] = 1
    if t.arch["ST16S"] and t.arch["ST16"]:      # otpu_dma's ST16R / ST16W
        d["ST16S"] = 1
    if t.arch["QHL"] and t.arch["SPLITQ"]:      # otpu_quant's HL flag
        d["QUANT_HL"] = 1
    if t.arch["QSTHL"] and t.arch["SPLITQ"]:    # otpu_quant's QST HL
        d["QUANT_QSTHL"] = 1
    if t.arch["VSIG"]:                          # otpu_vpu's SIGMOID / SILU
        d["VPU_SIG"] = 1
    if t.arch["VMAC"]:                          # otpu_vpu's VOP MAC
        d["VPU_MAC"] = 1
    if t.arch["OUTER1"]:                        # otpu_vpu's VOP OUTER decay one
        d["VPU_OUTER1"] = 1
    if t.arch["QONCE"]:                         # otpu_quant's QACT ST, otpu_dma's LDA
        d["QONCE"] = 1
    if t.arch["SPLMM"]:                         # otpu_mxu's MM SPL
        d["MXU_SPL"] = 1
    if t.arch["ICACHE"]:                        # otpu_seq's instruction cache
        d["ICACHE"] = 1
        d.update({k: int(u[k]) for k in ("IC_LINES", "IC_PF") if k in u})
    if t.kind == "fpga" and t.arch["S"] > 1:    # otpu_board's collective at S > 1 (its hub)
        d["COLL_HUB"] = 1
        d.update({k: int(u[k]) for k in COLL_KEYS if k in u})
    nch, cb = t.platform["channels_per_slice"], t.platform["beat_bytes"]
    if t.kind == "fpga" and (nch, cb) != (2, 64):   # the tops' default: two 64-byte channels
        d.update(NCH=nch, CB=cb)
    d.update({k: int(u[k]) for k in NAT_KEYS if k in u})
    return d


def mem_model(t: Target | str | None = None, mhz: float | None = None) -> list[str]:
    """The co-simulation's channel model for `t` at `mhz` (default the target's clock): the
    native memory model's arguments (sim/verilator/otpu_native_mem.sv) from the target's sim_*
    platform keys -- per channel the sustained rate as bytes per core cycle (+nat_bpc, in
    thousandths), the read latency in core cycles (+axi_lat) and the refresh-like duty loss
    (+nat_refi / +nat_rfc, cycles) -- or none for a target without them (the DDR3 cards': their
    controller co-simulated, or profile.ddr3_plusargs)."""
    t = t if isinstance(t, Target) else load(t)
    p = t.platform
    if "sim_ch_gbps" not in p:
        return []
    mhz = mhz or p["core_mhz"]
    cyc = lambda ns: round(ns * mhz / 1000)
    return [f"+nat_bpc={round(p['sim_ch_gbps'] * 1e6 / mhz)}", f"+axi_lat={cyc(p['sim_lat_ns'])}",
            f"+nat_refi={cyc(p.get('sim_refi_ns', 0))}", f"+nat_rfc={cyc(p.get('sim_rfc_ns', 0))}"]


def rtl_params(t: Target | str | None = None) -> dict:
    """The parameters of a board / ASIC top for `t`: arch sizes, uarch, and the identity the
    control block reports (CFG_* registers, register map 4). Flows pass the ones their top
    takes."""
    t = t if isinstance(t, Target) else load(t)
    a, u, p = t.arch, t.uarch, t.platform
    cfg = config(t)
    tm3 = {k: int(u[k]) for k in TMEM3_KEYS if k in u} if u["TMEM_IMPL"] == 3 else {}
    tm3 |= {"SE_R": int(u["SE_R"])} if u.get("SE_R") else {}
    tm3 |= {"DMA_DEPTH": int(u["DMA_DEPTH"])} if u.get("DMA_DEPTH") else {}
    tm3 |= {"DMA_LDQ": int(u["DMA_LDQ"])} if u.get("DMA_LDQ", 1) != 1 else {}
    nat = {k: int(u[k]) for k in NAT_KEYS if k in u}
    nat |= {k: str(u[k]) for k in STYLE_KEYS if k in u}     # (strings: a flow quotes them)
    nat |= {k: int(u[k]) for k in URAM_KEYS if u.get(k)}
    nat |= {k: int(u[k]) for k in COLL_KEYS if k in u}
    return tm3 | nat | {
        "S": a["S"], "D": a["D"], "MCOLS": a["MCOLS"], "ACT_ROWS": t.act_rows,
        "ACT_BLOCKS": a["ACT_BLOCKS"], "TMEM_WORDS": a["TMEM_WORDS"],
        "IMEM_WORDS": a["IMEM_WORDS"], "LANES": a["LANES"], "DSTEP": int(a["DSTEP"]),
        "MXU_IMPL": u["MXU_IMPL"], "MXU_CL": u["MXU_CL"], "MXU_E": u.get("MXU_E", 0),
        "VPU_CL": u["VPU_CL"], "ULANES": u["ULANES"], "TMEM_IMPL": u["TMEM_IMPL"],
        "RPB": u["RPB"] or NRP * a["LANES"], "WPB": u["WPB"], "WIN": u["WIN"],
        "FIFO_DEPTH": u["FIFO_DEPTH"], "TRACE_DEPTH": u["TRACE_DEPTH"], "PQ_WIN": u["PQ_WIN"],
        "CHASH": int(u["CHASH"]), "SEQ_WC": int(a["SEQ_WC"]),
        "DDR_MTS": int(p.get("mem_mts", 0)),
        "NCH": p["channels_per_slice"], "CB": p["beat_bytes"],
        "CH_LOG2": (p["channel_bytes"] - 1).bit_length(),
        "DRAM_LOG2": (a["DRAM_BYTES"] - 1).bit_length(),
        "PLATFORM": t.platform_id,
        "ARCH_HASH": arch_hash(cfg),
    } | ({"AW": a["AW"]} if a["AW"] > 32 else {}) \
      | ({"WG": a["WG"]} if a["WG"] > 1 else {}) \
      | ({"OUTER": 0} if not a["OUTER"] else {}) \
      | ({"DSTEP2": 1} if a["DSTEP2"] else {}) \
      | ({"KT": 1} if a["KT"] else {}) \
      | ({"ST16": ST16_CODES[a["ST16"]]} if a["ST16"] else {}) \
      | ({"VPU_FSUM": 1} if a["FSUM"] else {}) \
      | ({"QUANT_SPLIT": 1} if a["SPLITQ"] else {}) \
      | ({"ST16S": 1} if a["ST16S"] and a["ST16"] else {}) \
      | ({"QUANT_HL": 1} if a["QHL"] and a["SPLITQ"] else {}) \
      | ({"QUANT_QSTHL": 1} if a["QSTHL"] and a["SPLITQ"] else {}) \
      | ({"VPU_SIG": 1} if a["VSIG"] else {}) \
      | ({"VPU_MAC": 1} if a["VMAC"] else {}) \
      | ({"VPU_OUTER1": 1} if a["OUTER1"] else {}) \
      | ({"QONCE": 1} if a["QONCE"] else {}) \
      | ({"MXU_SPL": 1} if a["SPLMM"] else {}) \
      | ({"ICACHE": 1} | {k: int(u[k]) for k in ("IC_LINES", "IC_PF") if k in u}
         if a["ICACHE"] else {}) \
      | ({"QUANT_R1": 1} if u.get("QUANT_R1") else {}) \
      | ({"QUANT_PW": int(u["QUANT_PW"])} if u.get("QUANT_PW", 1) > 1 else {}) \
      | ({"QUANT_SWW": int(u["QUANT_SWW"])} if u.get("QUANT_SWW", 1) > 1 else {}) \
      | ({"QUANT_QSTB": 1} if u.get("QUANT_QSTB") else {}) \
      | ({"MXU_QD": int(u["MXU_QD"])} if u.get("MXU_QD", 2) != 2 else {}) \
      | ({"MXU_RF_ROWS": int(u["MXU_RF_ROWS"])} if u.get("MXU_RF_ROWS") else {}) \
      | ({"VPU_CNS": int(u["VPU_CNS"])} if u.get("VPU_CNS", 3) != 3 else {}) \
      | ({"MXU_PUMP": 1} if u.get("MXU_PUMP") else {})  # (absent: before)


def sources(t: Target | str | None = None, family: str | None = None) -> list[str]:
    """The RTL files of `t` under rtl/, in compile order: the core, the cell family's cells
    (`family` overrides the target's: the simulators build every target with "generic"), and
    for an FPGA the shell and the board's top (once the board has one)."""
    t = t if isinstance(t, Target) else load(t)
    tech = TECH[family or t.family]
    core = CORE[:CORE.index("mxu/otpu_mxu.sv") + 1] + tech + CORE[CORE.index("mxu/otpu_mxu.sv") + 1:]
    if t.kind != "fpga":
        return core
    return core + SHELL + BOARDS.get(t.platform["board"], [])


def build_config(t: Target | str | None = None, **knobs):
    """The Config of an FPGA build of `t` with the board flow's knobs (run_vivado.sh MCOLS,
    LANES, ACT_ROWS, TMEM_WORDS, DSTEP, SEQ_WC, WG, CHASH; a board model's IMEM_WORDS; the
    CFG_HASH it reports): ACT_ROWS
    equal to MCOLS is 0 (no replay), DSTEP 0 leaves out the stream engine too (STREAM), CHASH
    (otpu_board's: the channels' hash) is the Config's at WG > 1; OUTER 0 a VPU without VOP OUTER
    (otpu_board's; the board model's, SimTransport); DSTEP2 DSTEP's pair (with DSTEP); PAIR,
    column reuse (CAPS bit5: otpu_board's at WG 1 only, its MXU at WG > 1 runs every MM in
    groups) follows a WG knob; SE_R the streams' engine (SE_OWN at SE_R > 0, with DSTEP); KT
    MM's K tail (otpu_mxu's, at WG > 1); ICACHE IMEM an instruction cache (otpu_seq),
    IMEM_WORDS the longest program (CFG_IMEM); ST16 DSTEP's state in 16 bits (e6m9, bf16, or
    0 / empty: none; with DSTEP); FSUM EXP2SUB's fused row sums (otpu_vpu's, with DSTEP);
    SPLITQ QACT / QST flag LO (otpu_quant's SPLIT); ST16S STREAM's ST16R / ST16W (otpu_dma's,
    with ST16); QHL QACT flag HL (otpu_quant's HL, with SPLITQ); QSTHL QST HL (otpu_quant's
    QSTHL, with SPLITQ and the uarch's QUANT_QSTB); VSIG VOP SIGMOID / SILU
    (otpu_vpu's, with DSTEP); VMAC VOP MAC (otpu_vpu's); OUTER1 VOP OUTER decay one
    (otpu_vpu's, with VMAC and OUTER off); QONCE QACT flag ST and LDA (otpu_quant's and
    otpu_dma's, with SEQ_WC); SPLMM MM SPL (otpu_mxu's, with SEQ_WC at WG > 1)."""
    t = t if isinstance(t, Target) else load(t)
    a = dict(t.arch)
    for k in ("MCOLS", "LANES", "ACT_ROWS", "TMEM_WORDS", "IMEM_WORDS", "WG"):
        if knobs.get(k) is not None:
            a[k] = int(knobs[k])
    if a["ACT_ROWS"] == a["MCOLS"]:
        a["ACT_ROWS"] = 0
    if knobs.get("WG") is not None:
        a["PAIR"] = a["WG"] == 1
    if knobs.get("DSTEP") is not None:
        a["DSTEP"] = a["STREAM"] = bool(int(knobs["DSTEP"]))
    if knobs.get("SEQ_WC") is not None:
        a["SEQ_WC"] = bool(int(knobs["SEQ_WC"]))
    if knobs.get("CHASH") is not None:
        a["CHASH"] = a["WG"] > 1 and bool(int(knobs["CHASH"]))
    if knobs.get("OUTER") is not None:
        a["OUTER"] = bool(int(knobs["OUTER"]))
    if knobs.get("DSTEP2") is not None:
        a["DSTEP2"] = bool(int(knobs["DSTEP2"]))
    a["DSTEP2"] = a["DSTEP2"] and a["DSTEP"]
    if knobs.get("PAIR") is not None:
        a["PAIR"] = bool(int(knobs["PAIR"]))
    if knobs.get("KT") is not None:
        a["KT"] = bool(int(knobs["KT"]))
    if knobs.get("ICACHE") is not None:
        a["ICACHE"] = bool(int(knobs["ICACHE"]))
    if knobs.get("ST16") is not None:
        v = str(knobs["ST16"])
        a["ST16"] = {"0": "", "1": "e6m9", "2": "bf16"}.get(v, v)
        if a["ST16"] not in ST16_CODES:
            raise TargetError(f"ST16 {v!r}: e6m9, bf16 or 0")
    a["ST16"] = a["ST16"] if a["DSTEP"] else ""
    if knobs.get("FSUM") is not None:
        a["FSUM"] = bool(int(knobs["FSUM"]))
    a["FSUM"] = a["FSUM"] and a["DSTEP"] and a["LANES"] == 8 and a["D"] >= 32
    if knobs.get("SPLITQ") is not None:
        a["SPLITQ"] = bool(int(knobs["SPLITQ"]))
    if knobs.get("ST16S") is not None:
        a["ST16S"] = bool(int(knobs["ST16S"]))
    a["ST16S"] = a["ST16S"] and bool(a["ST16"])
    if knobs.get("QHL") is not None:
        a["QHL"] = bool(int(knobs["QHL"]))
    a["QHL"] = a["QHL"] and a["SPLITQ"]
    if knobs.get("QSTHL") is not None:
        a["QSTHL"] = bool(int(knobs["QSTHL"]))
    a["QSTHL"] = a["QSTHL"] and a["SPLITQ"] and bool(t.uarch.get("QUANT_QSTB"))
    if knobs.get("VSIG") is not None:
        a["VSIG"] = bool(int(knobs["VSIG"]))
    a["VSIG"] = a["VSIG"] and a["DSTEP"] and a["LANES"] == 8 and a["D"] >= 32
    if knobs.get("VMAC") is not None:
        a["VMAC"] = bool(int(knobs["VMAC"]))
    if knobs.get("OUTER1") is not None:
        a["OUTER1"] = bool(int(knobs["OUTER1"]))
    a["OUTER1"] = a["OUTER1"] and a["VMAC"] and not a["OUTER"]
    if knobs.get("QONCE") is not None:
        a["QONCE"] = bool(int(knobs["QONCE"]))
    a["QONCE"] = a["QONCE"] and a["SEQ_WC"]
    if knobs.get("SPLMM") is not None:
        a["SPLMM"] = bool(int(knobs["SPLMM"]))
    a["SPLMM"] = a["SPLMM"] and a["SEQ_WC"] and a["WG"] > 1
    if knobs.get("SE_R") is not None:
        from .isasim import se_own
        a["SE_OWN"] = se_own(int(knobs["SE_R"]), a["DSTEP"], a["LANES"], a["D"])
    return config(t, **a)


def _fmt(v) -> str:
    return str(int(v)) if isinstance(v, bool) else str(v)


def _knobs(args: list[str]) -> dict:
    """K=V arguments (empty values ignored)."""
    out = {}
    for a in args:
        k, _, v = a.partition("=")
        if v != "":
            out[k] = v
    return out


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, name = argv[0], (argv[1] if len(argv) > 1 else None)
    if cmd == "list":
        for n in names():
            t = load(n)
            print(f"{n:20s} {t.kind:5s} {t.description}")
        return 0
    t = load(name)
    if cmd == "json":
        print(json.dumps(t.to_dict() | {"rtl_params": rtl_params(t)}, indent=1, default=str))
    elif cmd == "show":
        print(f"{t.name}{':' + t.variant if t.variant else ''} ({t.kind}): {t.description}")
        for sec in ("arch", "uarch", "tech", "platform"):
            print(f"  [{sec}] " + " ".join(f"{k}={_fmt(v)}" for k, v in getattr(t, sec).items()))
        print(f"  arch_hash {arch_hash(config(t)):08x}")
    elif cmd == "generics":
        print(" ".join(f"{k}={_fmt(v)}" for k, v in rtl_params(t).items()))
    elif cmd == "files":                        # files NAME [FAMILY]: rtl/ paths, one a line
        print("\n".join(f"rtl/{f}" for f in sources(t, argv[2] if len(argv) > 2 else None)))
    elif cmd == "hash":                  # hash NAME [MCOLS=4 LANES=8 ACT_ROWS=4 DSTEP=1 SEQ_WC=0]
        print(f"{arch_hash(build_config(t, **_knobs(argv[2:]))):08x}")
    elif cmd == "env":                           # shell assignments T_<PARAM>=<value>
        for k, v in rtl_params(t).items():
            print(f"T_{k}={v:08x}" if k == "ARCH_HASH" else f"T_{k}={_fmt(v)}")
        print(f"T_CORE_MHZ={t.platform['core_mhz']}")
        print(f"T_PART={t.tech.get('part', '')}")
        print(f"T_FAMILY={t.family}")
    else:
        print(f"unknown command {cmd!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
