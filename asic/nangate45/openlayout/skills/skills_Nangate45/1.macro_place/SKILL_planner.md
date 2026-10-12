---
name: openroad-macro-placement
description: >
  Action playbook and TCL recipes for manual / scripted macro placement
  in OpenROAD. Use this skill whenever the user is doing macro placement,
  floorplanning, working with SRAM/IP block locations, asks about
  place_macro, macro orientation codes (R0/R90/R180/R270/MX/MY/MXR90/
  MYR90), setPlacementStatus, ord::get_db_block, save_image of a
  floorplan, write_db checkpoints, the macro-placement loop (plan →
  place → fix overlap → refine), or works on benchmarks like ariane,
  MacroPlacement where the task is positioning macros
  around the core. Output the script for macro placement.
---

## Planning rules

- **Push macros to the periphery .** Macros are routing obstacles; placing
  them in the core forces detours and creates congestion. Peripheral
  placement also simplifies the PG mesh and helps IR drop on high-power
  macros. 
- **Push macros to the corner.** 
- **Orient so macro pins do not face outward.** Rotate / flip only when a
  macro's highlighted pin side points outward toward the nearest core/die edge.
  Preserve row/column alignment while flipping. 
- **Leave room around each macro** for signal routing and the PG grid.
- **Eliminate dead pockets.** Empty space between macros wastes area —
  shift macros or adjust aspect ratio to convert it into usable
  standard-cell region.
- **Avoid narrow channels between blocks and boundary.** Skinny row strips between
  macros cause CTS and routing pain — widen or eliminate them. Do not leave channels between boundary with macros.
- **Keep the standard-cell region near aspect ratio 1.0 and contiguous.**
  Square is densest; aspect ratios above ~4 degrade density. Fragmented
  logic regions hurt density and complicate CTS. Do not place one or two macros to break, make small channels.
- **The area near the pin group should leave for stdcell.** The macros interact with pin group normally less than stdcell group, so the macro cluster can put away of pin group. Do not put macros in the center to make compact shape.
- **The macros in the same cluster should be placed closer.** According to the macro cluster file.
- **Point pin sides inward — toward the central standard-cell region and toward each other macro (pin-to-pin). Non-pin sides can then abut back-to-back to save space.**
- **Don't stack the macros too deep if possible(more than 3).** 
- **The gap between macros, 3 is ok if it is pin side, two non-pin side can abut.** 
- **Center-free is the objective.** Keep the central standard-cell region contiguous and usable; pushing macros to the periphery is only the heuristic that usually achieves this. When macro-to-periphery placement conflicts with center contiguity, prioritize the contiguous center and relax \"hug the edge\".

## Place a macro

If a macro is already placed/fixed, unlock it first by setting status to
`NONE`, then call `place_macro`:

```tcl
set inst [[ord::get_db_block] findInst "<full/hierarchical/instance/name>"]
$inst setPlacementStatus "NONE"

place_macro \
    -macro_name "<full/hierarchical/instance/name>" \
    -location {<x_um> <y_um>} \
    -orient <ORIENT_CODE>
```

`-location` is in microns and refers to the macro's origin (lower-left
in R0 orientation). Hierarchical instance names use forward slashes
(e.g. `i_cache_subsystem/i_nbdcache/sram_block_1__data_sram/...`).



## Orientation codes

| Code     | Meaning |
|----------|---------|
| `R0`     | No rotation, no mirror (default) |
| `R90`    | Rotated 90° CCW |
| `R180`   | Rotated 180° |
| `R270`   | Rotated 270° CCW (= 90° CW) |
| `MX`     | Mirror across X-axis (flip top↔bottom) |
| `MY`     | Mirror across Y-axis (flip left↔right) |
| `MXR90`  | Mirror X, then rotate 90° CCW |
| `MYR90`  | Mirror Y, then rotate 90° CCW |

Picking orientation: face the macro's pin side toward where its
connected logic / IO sits. SRAM pins are usually on one edge — rotate
so that edge faces the consumer logic.
