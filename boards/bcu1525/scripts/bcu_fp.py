"""The BCU1525's floorplans at two slices (targets/bcu1525-s2.toml; docs/bcu1525_s2.md):
which SLR each slice sits in and which two DIMMs it owns, and what follows from them for the build
(make_wg.py) and the board model (tools/board_perf.py --params FP=..; tb_bcuwg): DMAP, otpu_bcu_wg_sys's
DIMM slots on the DIMMs (slot i on DIMM DMAP[2 i +: 2], slice s on slots 2 s and 2 s + 1), and RLY,
each board channel's bridge pipeline (otpu_hbm_ch N): 4 k + 1 stages for a DIMM k SLR crossings from
its slice, 2 for one in the slice's own SLR (docs/bcu1525_wg.md).

The VU9P's DIMMs (docs/bcu1525.md): C0 in SLR0, C1 and C2 in SLR1 (with PCIe and XDMA), C3 in SLR2.

  a  slice 0 in SLR1 on C1 + C2 (both local), slice 1 in SLR2 on C3 (local) + C0 (two crossings)
  b  slice 0 in SLR1 on C0 (one crossing) + C1, slice 1 in SLR2 on C2 (one crossing) + C3
  c  slice 0 in SLR0 on C0 + C1 (one crossing), slice 1 in SLR2 on C2 (one crossing) + C3; SLR1
     left to the shell (XDMA, the four MIGs' AXI sides, the control block and the hub)

b and c own the same DIMMs (slice 0 C0 + C1, slice 1 C2 + C3): DMAP 0xE4, the identity, so the
host's DMA windows (sc_dma's DIMM d at d x 8 GiB) need no permutation; a's does (the top's g_dma).

At one slice (bcu1525-wg) slot i is DIMM i (DMAP 0xE4) and RLY follows make_wg.py --slr.
"""
from __future__ import annotations

DIMM_SLR = (0, 1, 1, 2)                            # C0 .. C3

# floorplan: (the slices' SLRs, each slice's two DIMMs: its slots 2 s, 2 s + 1)
FLOORPLANS = {
    "a": ((1, 2), ((1, 2), (3, 0))),
    "b": ((1, 2), ((0, 1), (2, 3))),
    "c": ((0, 2), ((0, 1), (2, 3))),
}
NC = 4                                             # channels a DIMM at two slices


def stages(slr: int, dimm: int) -> int:
    """A bridge's pipeline between its halves for DIMM `dimm` and a slice in SLR `slr`."""
    k = abs(DIMM_SLR[dimm] - slr)
    return 4 * k + 1 if k else 2


def plan(fp: str, nch: int = 8) -> dict:
    """Floorplan `fp`'s SLRs, DIMMs, DMAP and RLY (the 4-bit N of board channel g at bits 4 g),
    and the SLR crossings of each slice's weight streams."""
    if fp not in FLOORPLANS:
        raise ValueError(f"floorplan {fp!r}: one of {', '.join(sorted(FLOORPLANS))}")
    slrs, dimms = FLOORPLANS[fp]
    slots = [d for pair in dimms for d in pair]
    dmap = sum(d << (2 * i) for i, d in enumerate(slots))
    rly = 0
    for g in range(len(slrs) * nch):
        s, i = g // nch, g // NC
        rly |= stages(slrs[s], slots[i]) << (4 * g)
    cross = [[abs(DIMM_SLR[d] - slrs[s]) for d in dimms[s]] for s in range(len(slrs))]
    return {"fp": fp, "slr": slrs, "dimms": dimms, "dmap": dmap, "rly": rly, "crossings": cross}


def params(fp: str, nch: int = 8) -> str:
    """tb_bcuwg's / otpu_bcu_wg_sys's DMAP and RLY for floorplan `fp`, as board_perf --params takes them."""
    p = plan(fp, nch)
    return f"DMAP=0x{p['dmap']:02x},RLY=0x{p['rly']:0{len(p['slr']) * nch}x}"


if __name__ == "__main__":
    for k in sorted(FLOORPLANS):
        p = plan(k)
        print(k, p["slr"], p["dimms"], f"DMAP 0x{p['dmap']:02x}", f"RLY 0x{p['rly']:016x}",
              "crossings", p["crossings"], params(k))
