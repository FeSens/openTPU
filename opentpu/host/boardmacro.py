"""otpu_board's parameters as a board wrapper's defines. A wrapper that instantiates otpu_board
inside its own system (rtl/boards/bcu1525/otpu_bcu_wg_sys.sv: the channel bridges, the DIMMs'
pairs) sets the slices, their channels and the clock at the instance and takes every other
parameter from the OTPU_BOARD macro: boards/bcu1525/scripts/make_wg.py's for the bitstream (the
target's rtl_params) and the board model's (board.SimTransport bcuwg) are this module's."""
from __future__ import annotations

import re
from pathlib import Path

# the wrapper's own: the slices and their channels, the core clock, no DDR3, I2C or host
# calibration, set at the instance
WRAP_OWN = ("S", "NCH", "CB", "CORE_KHZ", "DDR_MTS", "HAS_I2C", "HOSTCAL")


def board_param_names() -> list[str]:
    """otpu_board's parameters, in rtl/shell/otpu_board.sv's order."""
    src = (Path(__file__).resolve().parents[2] / "rtl/shell/otpu_board.sv").read_text()
    head = src[src.index("module otpu_board #("):]
    head = head[:head.index("\n) (")]
    return re.findall(r"^\s*parameter\s+(?:\w+\s+)*?(?:\[[^\]]*\]\s*)?(\w+)\s*=", head, re.M)


def board_macro(p: dict) -> str:
    """OTPU_BOARD's text: `p`'s otpu_board parameters other than the wrapper's own (WRAP_OWN) as
    overrides, ", .K(V)" in otpu_board's order: strings quoted (TMEM_STYLE; one already a
    Verilog literal, 32'h.. or quoted, as given), ARCH_HASH and BUILD_ID in 32-bit hex. A key
    otpu_board does not declare is an error (the wrapper would build without it)."""
    names = board_param_names()
    if bad := sorted(set(p) - set(names)):
        raise ValueError(f"not otpu_board parameters: {', '.join(bad)}")

    def v(k: str, x) -> str:
        if isinstance(x, str):
            return x if x.startswith(("32'h", '"')) else f'"{x}"'
        return f"32'h{x:08x}" if k in ("ARCH_HASH", "BUILD_ID") else str(int(x))
    return "".join(f", .{k}({v(k, p[k])})" for k in names if k in p and k not in WRAP_OWN)


def board_defines(p: dict) -> str:
    """The wrapper's defines from `p` (otpu_board's parameters): OTPU_BOARD (board_macro),
    OTPU_WPOST, the board's NAT_WPOST for the bridges (posted write data), OTPU_XPOST, its
    NAT_XPOST (port W's beats posted into the MXUs' chunk FIFOs), OTPU_RHB and OTPU_WHB, its
    NAT_RHB / NAT_WHB (the native ports' read / write data in half beats), and OTPU_XSW, log2
    of its FIFO_DEPTH (otpu_board's default 1024): the bridges' chunk FIFO entry bits."""
    xsw = (int(p.get("FIFO_DEPTH", 1024)) - 1).bit_length()
    return (f"`define OTPU_BOARD {board_macro(p)}\n`define OTPU_WPOST {int(p.get('NAT_WPOST', 0))}\n"
            f"`define OTPU_XPOST {int(p.get('NAT_XPOST', 0))}\n`define OTPU_RHB {int(p.get('NAT_RHB', 0))}\n"
            f"`define OTPU_WHB {int(p.get('NAT_WHB', 0))}\n`define OTPU_XSW {xsw}\n")


# rtl_params keys that are no otpu_board parameter: they follow from NCH and CB (the wrapper's
# own, checked against its channels)
DERIVED = ("CH_LOG2", "DRAM_LOG2")
# the arch knobs a --set changes ARCH_HASH with (as opentpu.target.build_config)
KNOBS = ("MCOLS", "LANES", "ACT_ROWS", "TMEM_WORDS", "DSTEP", "SEQ_WC", "WG", "CHASH", "OUTER",
         "DSTEP2", "SE_R", "KT", "ICACHE", "IMEM_WORDS")


def build_id(root: Path | None = None) -> int:
    """The tree's commit: COMMIT (a copy shipped without .git: the short sha, +dirty), else
    git's."""
    import subprocess
    root = root or Path(__file__).resolve().parents[2]
    c = root / "COMMIT"
    sha = c.read_text().strip() if c.exists() else subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short=8", "HEAD"], capture_output=True,
        text=True).stdout.strip()
    try:
        return int(sha[:8], 16)
    except ValueError:
        return 0


def board_params(t, nc: int, cb: int, sets: dict | None = None) -> dict:
    """otpu_board's parameters for a wrapper's build: the target's rtl_params with `sets` (K: V
    overrides; the arch knobs' change ARCH_HASH as build_config's) and BUILD_ID, checked against
    the wrapper's nc channels of cb bytes and otpu_board's parameters (OTPU_BOARD:
    board_macro). The bridges post port W's beats into the MXUs' chunk FIFOs (the target's
    NAT_XPOST): a build without them only when asked for (sets NAT_XPOST 0)."""
    from opentpu import target as T
    p = {k: v for k, v in T.rtl_params(t).items() if k not in DERIVED}
    if p["NCH"] * p["S"] > nc or p["CB"] != cb:
        raise SystemExit(f"{t.name} has {p['S']} x {p['NCH']} channels of {p['CB']} bytes, the "
                         f"board {nc} of {cb}")
    sets = sets or {}
    names = board_param_names()
    for k, v in sets.items():
        if k not in names or k in WRAP_OWN + ("ARCH_HASH", "BUILD_ID"):
            raise SystemExit(f"--set {k}: not an otpu_board parameter the target sets")
        try:
            p[k] = v if isinstance(v, int) else int(v, 0)
        except ValueError:                  # a string parameter (TMEM_STYLE)
            p[k] = v
    if not p.get("NAT_XPOST") and "NAT_XPOST" not in sets:
        raise SystemExit(f"{t.name} at NAT_XPOST 0: the bridges post port W's beats (the "
                         f"target's NAT_XPOST); --set NAT_XPOST=0 builds one without")
    knobs = {k: p[k] for k in KNOBS if k in sets}
    if knobs:
        p["ARCH_HASH"] = T.arch_hash(T.build_config(t, **knobs))
    p["BUILD_ID"] = build_id()
    try:
        board_macro(p)
    except ValueError as e:
        raise SystemExit(f"{t.name}'s rtl_params: {e}")
    return p
