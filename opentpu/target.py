"""Targets: one file per device, `targets/<name>.toml` (docs/targets.md).

A target describes one build of openTPU in four layers:

- arch: the `opentpu.isasim.Config` fields, what the programs depend on. Two targets with the
  same arch run the same compiled programs.
- uarch: RTL parameters that change timing only (MXU_IMPL, TMEM_IMPL, RPB / WPB, ...).
- tech: the cell family the RTL is built with (rtl/tech/<family>), the part or the PDK.
- platform: the board around the accelerator (memory, channels, host link, clock).

Everything that depends on the target comes from here: `config()` for the compiler and the
simulators, `sim_uarch()` for the Verilator builds, `rtl_params()` for the FPGA and ASIC tops,
`arch_hash()` for the device's CFG_HASH register and the host's check. `current()` is the target
the environment selects (OTPU_TARGET, default the YPCB-00338 card).

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
# above MCOLS: MCOLS); with TMEM_IMPL 3 the one copy's knobs (otpu_slice TMEM_<K>,
# docs/tmem_nocopy.md; vivado/tmem_knobs.tcl's TMEM_X)
TMEM3_KEYS = ("TMEM_NSB", "TMEM_SBP", "TMEM_SBH", "TMEM_RPL", "TMEM_TDP", "TMEM_GST", "TMEM_PMP",
              "TMEM_PRK", "TMEM_SDM")
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
         "mem/otpu_native_dram.sv", "shell/otpu_afifo.sv", "shell/otpu_mem_ch.sv",
         "shell/otpu_axi_split2.sv"]
BOARDS = {"ypcb-00338": ["shell/otpu_pcie_mm.sv", "boards/ypcb-00338/otpu_native_sys.sv",
                         "boards/ypcb-00338/otpu_fpga_top_ld.sv"],   # (otpu_pcie_mm: LitePCIe's DMA)
          "bcu1525": ["boards/bcu1525/otpu_bcu_sys.sv", "boards/bcu1525/otpu_fpga_top_bcu.sv"]}


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
            if k in raw["arch"]:
                raw["arch"][k] = v
            elif k in raw["uarch"] or k in UARCH_KEYS + ("MXU_E",) + TMEM3_KEYS:
                raw["uarch"][k] = v
            else:
                raise TargetError(f"{path}: variant {var!r}: {k} is not an [arch] or [uarch] key")
    cf = _config_fields()
    bad = sorted(set(raw["arch"]) - set(cf))
    if bad:
        raise TargetError(f"{path}: [arch] keys {bad} are not Config fields")
    missing = sorted(set(cf) - set(raw["arch"]))
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
    if raw["arch"]["SEQ_WC"] and raw["uarch"]["WIN"] > 16:
        raise TargetError(f"{path}: SEQ_WC (otpu_seq's wait entries) needs WIN <= 16: 4-bit "
                          f"counts name a unit's last 15 instructions")
    if a["DRAM_BYTES"] > 1 << 32:
        raise TargetError(f"{path}: DRAM_BYTES over 4 GiB: DRAM addresses are 32 bits per slice "
                          f"(more slices, or the reserved window register: docs/targets.md)")
    return Target(name=path.stem, kind=t["kind"], platform_id=int(t.get("platform_id", 0)),
                  description=t.get("description", ""), arch=dict(a), uarch=dict(raw["uarch"]),
                  tech=dict(raw["tech"]), platform=dict(raw["platform"]), path=path,
                  variant=var)


def by_platform(pid: int) -> Target | None:
    """The target whose platform_id a device reports (CFG_PLATFORM), if this tree has it."""
    for n in names():
        t = load(n)
        if t.platform_id == pid:
            return t
    return None


def current() -> Target:
    """The target OTPU_TARGET names (default the YPCB-00338 card)."""
    return load(os.environ.get("OTPU_TARGET") or DEFAULT)


def config(t: Target | str | None = None, **kw):
    """The target's `Config` (the hardware's truth: its optional features on), `kw` overriding
    fields."""
    from .isasim import Config
    t = t if isinstance(t, Target) else load(t)
    return Config(**{**t.arch, **kw})


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
    hash leaves it out: opentpu/host/board.py check_hash)."""
    d = dict(dataclasses.asdict(cfg), NUMERICS=NUMERICS)
    if not d.get("SEQ_WC"):
        d.pop("SEQ_WC", None)
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
    if u.get("MXU_E"):
        d["MXU_E"] = u["MXU_E"]
    if t.arch["SEQ_WC"]:
        d["SEQ_WC"] = 1
    return d


def rtl_params(t: Target | str | None = None) -> dict:
    """The parameters of a board / ASIC top for `t`: arch sizes, uarch, and the identity the
    control block reports (CFG_* registers, register map 4). Flows pass the ones their top
    takes."""
    t = t if isinstance(t, Target) else load(t)
    a, u, p = t.arch, t.uarch, t.platform
    cfg = config(t)
    tm3 = {k: int(u[k]) for k in TMEM3_KEYS if k in u} if u["TMEM_IMPL"] == 3 else {}
    return tm3 | {
        "S": a["S"], "D": a["D"], "MCOLS": a["MCOLS"], "ACT_ROWS": t.act_rows,
        "ACT_BLOCKS": a["ACT_BLOCKS"], "TMEM_WORDS": a["TMEM_WORDS"],
        "IMEM_WORDS": a["IMEM_WORDS"], "LANES": a["LANES"], "DSTEP": int(a["DSTEP"]),
        "MXU_IMPL": u["MXU_IMPL"], "MXU_CL": u["MXU_CL"], "MXU_E": u.get("MXU_E", 0),
        "VPU_CL": u["VPU_CL"], "ULANES": u["ULANES"], "TMEM_IMPL": u["TMEM_IMPL"],
        "RPB": u["RPB"] or NRP * a["LANES"], "WPB": u["WPB"], "WIN": u["WIN"],
        "FIFO_DEPTH": u["FIFO_DEPTH"], "TRACE_DEPTH": u["TRACE_DEPTH"], "PQ_WIN": u["PQ_WIN"],
        "CHASH": int(u["CHASH"]), "SEQ_WC": int(a["SEQ_WC"]),
        "DDR_MTS": int(p.get("mem_mts", 0)),
        "NCH": p["channels_per_slice"],
        "CH_LOG2": (p["channel_bytes"] - 1).bit_length(),
        "DRAM_LOG2": (a["DRAM_BYTES"] - 1).bit_length(),
        "PLATFORM": t.platform_id,
        "ARCH_HASH": arch_hash(cfg),
    }


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
    LANES, ACT_ROWS, TMEM_WORDS, DSTEP, SEQ_WC; the CFG_HASH it reports): ACT_ROWS equal to
    MCOLS is 0 (no replay), DSTEP 0 leaves out the stream engine too (STREAM)."""
    t = t if isinstance(t, Target) else load(t)
    a = dict(t.arch)
    for k in ("MCOLS", "LANES", "ACT_ROWS", "TMEM_WORDS"):
        if knobs.get(k) is not None:
            a[k] = int(knobs[k])
    if a["ACT_ROWS"] == a["MCOLS"]:
        a["ACT_ROWS"] = 0
    if knobs.get("DSTEP") is not None:
        a["DSTEP"] = a["STREAM"] = bool(int(knobs["DSTEP"]))
    if knobs.get("SEQ_WC") is not None:
        a["SEQ_WC"] = bool(int(knobs["SEQ_WC"]))
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
