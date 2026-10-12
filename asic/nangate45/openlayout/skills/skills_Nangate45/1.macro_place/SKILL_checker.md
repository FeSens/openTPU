---
name: macro-place-checker
description: Verification gates for the macro placement stage in an OpenROAD iteration loop. Use this skill whenever macro placement has just finished, detailed placement has just finished, or post-route timing has missed its goal — anytime the user is running macro placement exploration, sweeping floorplans, deciding whether to advance to CTS/routing, or choosing between a small macro nudge and a full replan. Trigger even if the user doesn't say "gate" or "check" — phrases like "is this placement good", "should I run CTS now", "timing still fails, what next", or "macros look right?" all qualify.
---

# Macro Placement Checker

Three verification gates sit inside the macro placement iteration loop. Run them in order. Skipping any of them wastes runtime: late-stage failures that could have been caught upstream, a "finished" placement that's actually wasteful, or a refinement loop that nudges macros when it should be replanning the whole shape.

## When to run

- **Gate A** — immediately after macro placement completes, before launching detailed placement.
- **Gate B** — after routing, only if post-route timing misses the goal (`WNS < 0` or `TNS < 0`).

If a gate fails, states the reason for planner, so it can refine.

## Gate A — Before detailed placement / CTS

Once macros are placed, **first compare the implemented placement against the
Planner's stated plan, then judge whether the placement is good enough to launch
the rest of the flow**:

- **Shape/topology:** Does the placed macro outline match the requested shape, side cluster, corner cluster, or split topology? Does the standard-cell region make a good shape(not fragmented cause by one or two macros) and contiguous. Does Push macros to the periphery if possible?
- **Pin group of die:** Does the area near pin group of die leave for stdcell? Some cases, we have to place next to pin group like pin group on four boundary, that is fine.
- **Orientation intent:** Use the Gate-A image with macro pin-side highlights. The simplified Gate-A rule is: a macro's highlighted pin side should not face outward toward the nearest core/die edge. For macros placed on the core periphery, fail only when the highlighted pin side points outward; request a rotate/flip fix that preserves the current row/column alignment. Should adjacent macros to face pin-side-to-pin-side, but do not fail because neighboring macros are not facing each other.
- **Pin-side packing:** Reserve routing/stdcell space on the highlighted pin side, where the macro actually connects. Macros may be packed closely on non-pin sides with only the required placement/PG clearance. 
- **Routing channels:** Are channels between macros wide enough for routing + PG? Don't stack too deep.
- **Stdcell area:** Does the predicted stdcell area look continuous and ~aspect-ratio 1? If pins are spread across multiple sides, do the macros spread out to match while reserving the middle for stdcells and forming a coherent shape such as a ring?

If the implementation does not match the plan, fail Gate A before judging
general quality. In the findings, state the concrete mismatch in one sentence,
starting with `Shape mismatch:` when topology, occupied regions, or opening
direction is wrong. Then request a Placer refinement/redo against the original
Planner strategy rather than advancing to detailed placement.

When Gate A fails because of macro pin-side orientation or packing, make the
feedback executable for Planner/Placer: name the affected side/group or macro
indices if visible, state which highlighted pin side should face which direction
(`rotate/flip so pins do not face outward`). Explicitly say that the current
row/column coordinates should be preserved. The feedback should tell the Placer
to flip orientations only; it should not ask the Planner to replan, and it
should not request pin-side-to-pin-side facing between neighboring macros. Do not
only say "fix orientation" or "make it tighter".

If the Planner strategy contains both a named topology and a region list, both
must be satisfied. Passing the region list alone is not enough when the visible
outline does not match the named shape/opening direction.

If any check fails, **refine the macro placement now** (re-run for the affected macros). Detailed placement and CTS are expensive — do not start them on a floorplan you already know is wrong.


## Gate B — On timing miss, pick the refinement scale

If post-route or detailed placement timing misses the goal (`WNS < 0` or `TNS < 0`), **decide the refinement scale before touching the placement**. Fine-grain nudges and coarse-grain replans fix different problems — using the wrong lever burns iterations.

Go **fine-grain** (move individual macros) when:

- Stdcell area is sparse, or stdcells have leaked into channels between macros and caused routing congestion
- A specific channel is too narrow, but the overall shape is right
- One or two macros sit slightly off their planned region
- Gate A/B was mostly fine and only localized issues remain

Go **coarse-grain** (discard the shape) when:

- Stdcell area is dense and continuous (Gate B passed) but timing still misses by a meaningful margin
- The current shape leaves critical paths fundamentally long — e.g., every macro is on the periphery and a critical net has to cross the whole die
- Two or three fine-grain refinements in a row haven't moved WNS

When going coarse, **try a genuinely different topology** — cluster near the dominant pin side, ring around stdcells, group timing-critical macros adjacent to each other. Don't replay a small variation of a shape that already failed.

Why split it this way: a dense stdcell layout that still fails timing means the *plan* is wrong, not the placement detail — more nudging won't fix it. A sparse layout with routing issues means the plan is right but execution is loose — replanning the whole shape throws away work that was actually fine.

## Reporting

After each gate, report in this format so the loop driver can act on it:

```
Gate: A | B 
Result: PASS | FAIL
Findings: <one-line summary of what was checked / what failed>
Next action: ADVANCE (to next stage) | REFINE fine-grain | REFINE coarse-grain (new shape)
```
