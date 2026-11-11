#!/usr/bin/env python3
"""The BCU1525 WG build's generated inputs (targets/bcu1525-wg.toml; boards/bcu1525/vivado/
create_project_wg.tcl reads them from OUT):

    python3 boards/bcu1525/scripts/make_wg.py OUT [--core-mhz F] [--set K=V ...] [--slr N]
        [--rly N0,N1,N2,N3] [--target bcu1525-wg[:VARIANT] | bcu1525-s2[:VARIANT] [--plan a|b|c]]
        [--target bcu1525-x2:st16 --plan c]  # one S=2 card of the S=4 machine

OUT/otpu_bcu_wg_defines.vh: otpu_board's parameters (OTPU_BOARD, OTPU_WPOST .. OTPU_XSW: the
target's rtl_params with --set overrides, the arch knobs' with their ARCH_HASH, and BUILD_ID the
tree's commit: opentpu.host.boardmacro's board_params and board_defines; and
OTPU_WG_CAW, the channels' beat address bits from the target's channel_bytes, and OTPU_WG_BHW, the
pairs' bank-half window from the target's BHW: a channel's bytes, log2), a global include in
the project. OUT/wg_params.tcl: CORE_MHZ, CORE_KHZ (the block design's clock, an even integer
divide of its 1200 MHz VCO: acc_clk2x at half of it), RLY (channel j's bridge pipeline
otpu_hbm_ch N, DIMM j >> 1's --rly entry), WG_CAW (the same as OTPU_WG_CAW: the top's CAW and
bd_wg.tcl's DMA windows), BUILD_ID, ARCH_HASH and RTL, the target's sources in compile order
(opentpu.target sources).

--slr N (default 2): the slice's SLR (constraints/bcu1525_wg_floorplan.tcl's, OTPU_WG_SLR). C0 is
in SLR0, C1 and C2 in SLR1 (with PCIe), C3 in SLR2. --rly's default follows from it: a DIMM k
crossings from the slice 4 k + 1 stages, one in the slice's SLR 2 (at its edges). SLR2
(5,5,2 for C1-C3 and 9 for C0) leaves SLR1 to PCIe, C1, C2 and the DMA's interconnect: with the
slice there too it was 94.8% of its slices and failed timing on its 300 MHz paths (2026-10-28);
SLR1 gives 5,2,2,5.

--target bcu1525-s2 (two slices, two DIMMs each: docs/bcu1525_s2.md): --plan picks
the floorplan (bcu_fp.py; default c: slice 0 in SLR0 on C0 + C1, slice 1 in SLR2 on C2 + C3),
which sets the slices' SLRs (WG_SLRS, the floorplan's pblocks), DMAP (OTPU_WG_DMAP and WG_DMAP:
the DIMM slots' DIMMs; the floorplan's pipeline stages by SLR) and RLY (16 channels, 64 bits);
OTPU_WG_S and WG_S the slices. --slr and --rly are
the one slice's. Either target takes a variant (default the file's own): bcu1525-wg:bhw the
bank-half window (OTPU_WG_BHW, its CFG_HASH with BHW).

--target bcu1525-x2:st16 builds one card (NCARDS 2, S 2, PLATFORM 7, CFG_HASH E331F79C),
derived from the compiler's S=4 target before hashing. OTPU_WG_X2 enables the QSFP ports and
endpoint in the same WG top; WG_NCARDS enables the extra BD clock, Aurora IP and constraints.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from opentpu import target as T                    # noqa: E402
from opentpu.host import boardmacro as BM          # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bcu_fp                                      # noqa: E402

TARGET = "bcu1525-wg"
TARGETS = ("bcu1525-wg", "bcu1525-s2", "bcu1525-x2")
VCO_MHZ = 1200                                     # bd_wg.tcl's clock wizard


def core_khz(mhz: float) -> int:
    div = round(VCO_MHZ / mhz)
    if div % 2 or abs(VCO_MHZ / div - mhz) > 0.01:
        raise SystemExit(f"make_wg.py: {mhz} MHz is no even divide of the {VCO_MHZ} MHz VCO "
                         f"(acc_clk2x at half the divide): 100 (12), 150 (8), 120 (10) ..")
    return round(VCO_MHZ * 1000 / div)


DIMM_SLR = (0, 1, 1, 2)                            # C0 .. C3


def rly_for(slr: int) -> list[int]:
    """Each DIMM's bridge pipelines with the slice in SLR slr."""
    return [4 * abs(slr - d) + 1 if d != slr else 2 for d in DIMM_SLR]


def caw(channel_bytes: int, beat_bytes: int, s: int = 1) -> int:
    """The channels' beat address bits: 22 (512 MiB a channel) .. 25 (4 GiB, all of an 8 GB DIMM's
    half; 26, 16 GB DIMMs, needs another MIG part and its AXI at 34 bits: bd_wg.tcl). At s 2 a
    DIMM holds four: 22 .. 24."""
    n = (channel_bytes // beat_bytes - 1).bit_length()
    top = 26 - s
    if channel_bytes != beat_bytes << n or not 22 <= n <= top:
        raise SystemExit(f"make_wg.py: channel_bytes {channel_bytes}: 2^22 .. 2^{top} beats of "
                         f"{beat_bytes} bytes ({2 * s} channels a DIMM)")
    return n


def rly(per_dimm: list[int], nch: int) -> int:
    """RLY's bits: channel j's N in RLY[4 j +: 4], DIMM j >> 1's."""
    if len(per_dimm) != nch // 2 or any(not 0 <= n <= 15 for n in per_dimm):
        raise SystemExit(f"make_wg.py: --rly {per_dimm}: {nch // 2} entries of 0 .. 15")
    return sum(per_dimm[j >> 1] << (4 * j) for j in range(nch))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--core-mhz", type=float)
    ap.add_argument("--set", action="append", default=[], metavar="K=V")
    ap.add_argument("--slr", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--rly", help="N0,N1,N2,N3 (default: --slr's)")
    ap.add_argument("--target", default=TARGET,
                    help=f"{' or '.join(TARGETS)}, or a variant ({TARGET}:bhw)")
    ap.add_argument("--plan", choices=sorted(bcu_fp.FLOORPLANS),
                    help="bcu1525-s2's floorplan (bcu_fp.py; default c)")
    a = ap.parse_args()
    if a.target.partition(":")[0] not in TARGETS:
        raise SystemExit(f"make_wg.py: --target {a.target}: {' or '.join(TARGETS)}, or a variant "
                         f"of it")
    t = T.load(a.target)
    sets = dict(kv.split("=", 1) for kv in a.set)
    ncards = T.cards(t)
    # The compiler target is the whole machine. Build one card, including its hash, from the
    # local slice count BEFORE board_params hashes it (x2-sw's per-card contract). Never emit
    # the machine's S=4 hash and then override the RTL's S. S/PLATFORM are wiring, not --set knobs.
    if ncards > 1:
        if ncards != 2 or t.arch["S"] != 4 or {"S", "PLATFORM"} & sets.keys():
            raise SystemExit("make_wg.py: x2 requires two cards, four global slices, no S/PLATFORM override")
        t = replace(t, arch=dict(t.arch, S=t.arch["S"] // ncards))
        if a.plan not in (None, "c"):
            raise SystemExit("make_wg.py: x2 uses plan c (the hub in SLR1)")
    ns = int(T.rtl_params(t)["S"])
    bp = BM.board_params(t, ns * t.platform["channels_per_slice"], t.platform["beat_bytes"],
                              sets)
    if bp["S"] not in (1, 2):
        raise SystemExit(f"make_wg.py: {a.target} at S {bp['S']}: the top has one or two slices")
    mhz = a.core_mhz or float(t.platform["core_mhz"])
    khz = core_khz(mhz)
    cw = caw(t.platform["channel_bytes"], t.platform["beat_bytes"], bp["S"])
    if bp["S"] == 1:
        if a.plan:
            raise SystemExit(f"make_wg.py: --plan is bcu1525-s2's (two slices), {a.target} has one")
        r = rly([int(n) for n in a.rly.split(",")] if a.rly else rly_for(a.slr), bp["NCH"])
        slrs, dmap, fp = (a.slr,), 0xE4, ""
    else:
        if a.rly or a.slr != 2:
            raise SystemExit("make_wg.py: --slr / --rly are one slice's; two slices take --plan")
        pl = bcu_fp.plan(a.plan or "c", bp["NCH"])
        r, slrs, dmap, fp = pl["rly"], pl["slr"], pl["dmap"], pl["fp"]
    rw = 32 * bp["S"]
    bhw = t.arch["BHW"] and t.arch["BHW"] - (t.platform["channels_per_slice"] - 1).bit_length()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "otpu_bcu_wg_defines.vh").write_text(
        "// written by boards/bcu1525/scripts/make_wg.py\n"
        "`ifndef OTPU_BCU_WG_DEFINES\n`define OTPU_BCU_WG_DEFINES\n"
        f"{BM.board_defines(bp)}`define OTPU_WG_CAW {cw}\n`define OTPU_WG_BHW {bhw}\n"
        f"`define OTPU_WG_S {bp['S']}\n`define OTPU_WG_DMAP 8'h{dmap:02X}\n"
        # the image's identity for the governor (answered in SAFE: otpu_bcu_gov's IMAGE words)
        f"`define OTPU_WG_PLATFORM {int(bp['PLATFORM'])}\n"
        f"`define OTPU_WG_CFG_HASH 32'h{bp['ARCH_HASH']:08X}\n"
        f"`define OTPU_WG_BUILD_ID 32'h{bp['BUILD_ID']:08X}\n`endif\n")
    if ncards > 1:
        vh = a.out / "otpu_bcu_wg_defines.vh"
        vh.write_text(vh.read_text().replace("`endif\n",
            f"`define OTPU_WG_X2\n`define OTPU_WG_HASH 32'h{bp['ARCH_HASH']:08X}\n`endif\n"))
    rtl = " ".join(f"rtl/{f}" for f in T.sources(t))
    (a.out / "wg_params.tcl").write_text(
        "# written by boards/bcu1525/scripts/make_wg.py\n"
        f"set CORE_MHZ {mhz:g}\nset CORE_KHZ {khz}\nset RLY {r:0{rw // 4}x}\nset WG_SLR {slrs[0]}\n"
        f"set WG_S {bp['S']}\nset WG_SLRS [list {' '.join(map(str, slrs))}]\nset WG_PLAN {fp or '-'}\n"
        f"set WG_DMAP {dmap:02X}\nset WG_CAW {cw}\n"
        f"set BUILD_ID {bp['BUILD_ID']:08x}\nset ARCH_HASH {bp['ARCH_HASH']:08x}\n"
        f"set WG_NCARDS {ncards}\n"
        f"set RTL [list {rtl}]\n")
    where = (f"slice in SLR{slrs[0]}" if bp["S"] == 1 else
             f"floorplan {fp}: slices in SLR{slrs[0]} and SLR{slrs[1]}, DMAP 8'h{dmap:02X}")
    print(f"make_wg: {a.out}: {a.target}, core {khz} kHz, {where}, RLY {rw}'h{r:0{rw // 4}x}, CAW {cw}, BHW {bhw}, BUILD_ID {bp['BUILD_ID']:08x}, "
          f"ARCH_HASH {bp['ARCH_HASH']:08x}, MCOLS {bp['MCOLS']} WG {bp['WG']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
