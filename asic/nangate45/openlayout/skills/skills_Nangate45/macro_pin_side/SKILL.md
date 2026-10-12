# Skill: Macro pin-side highlight

Detect the **signal-pin side(s)** of every macro from OpenROAD DB geometry and draw a
thick highlight line on **every** pin-bearing edge in a whole-die PNG. A macro may have
pins on more than one side (e.g. an SRAM with pins on both L and R) — each real pin side
is highlighted, not just the single dominant one.

**Key idea:** never infer the pin side visually from a screenshot. The side comes
from `iterm getBBox`, which already reflects the macro's *current placed
orientation* (R0/R90/MX/MY…). The screenshot is only the canvas we annotate.

## What it does

1. OpenROAD loads the placement (`--odb`, or `--def` + LEFs).
2. For every **BLOCK** instance:
   - take the placed bbox (`inst getBBox`),
   - iterate `inst getITerms`, keep only `getSigType == SIGNAL`,
   - compute each pin center from `iterm getBBox`,
   - measure its distance to the macro L/R/T/B edges and count it toward the
     nearest edge,
   - dominant side = edge with the most pins (`NONE` if no signal pins).
3. Writes structured text, one line per macro (the per-edge `counts` are the source
   of truth; pin **side(s)** are derived from them in Python — see step 5):
   ```
   MACRO_PIN_SIDE name=<inst> orient=<orient> side=<L/R/T/B/NONE> dominant=<n> \
       counts=L:<n>,R:<n>,T:<n>,B:<n> bbox=(x0,y0,x1,y1)
   ```
   (bbox in microns. `side` = the single dominant edge; `counts` carry all four so a
   multi-side macro can be recovered without re-running OpenROAD.)
4. `save_image` of the whole die, following the **whole-die-with-pins** convention
   of `skills_Nangate45/image_snapshot`: the save area is the union of the die box
   and the top-level IO-pin (bterm) boxes, padded by `margin`, and the signal/clock
   bterm boxes are recorded so they can be redrawn (OpenROAD rasterizes thin pins to
   near-invisibility at die zoom).
5. Python derives each macro's pin **side(s)** from the per-edge counts with
   `pin_sides(counts, min_pins)`: an edge is a pin side when it carries at least
   `--min-pins` signal pins (default **1** = every edge with any pins). Each pin is
   already binned to its geometrically nearest edge, so this reports **all** real
   pin sides — a macro with address/data on L+R and control on T+B shows **four**
   bars (e.g. `L:40 R:38 T:10 B:10` → `L,R,T,B`). Then it maps each macro/pin bbox
   micron→pixel (with Y-flip) and draws:
   - the **IO pins** as small **cyan** boxes (so the die-edge pins stay visible), then
   - a **black-shadow + bright-red** bar on **each** pin side of every macro, drawn
     **just inside that macro**, hugging the pin edge and **inset to ~70%** of the
     edge length. Keeping it inside (not in the channel) means the bar is never
     confused with the neighbouring macro across an abutting edge,
   - a **numbered label** (yellow-on-black) drawn **once per macro**, on the dominant
     pin side. The number is the macro's index in the printed summary /
     `MACRO_PIN_SIDE` order, so you can read bar → macro name. Disable with `--no-label`.

   The stdout summary prints `sides=<comma list>` and a `side_combos` histogram
   (e.g. `{'LR': 12}`) so multi-side macros are obvious without opening the image.

## Usage

```bash
conda activate coder

python skills/skills_Nangate45/macro_pin_side/macro_pin_side.py \
    --def     Benchmark/PlaceReasoner-Bench/bp_fe/def/u65_placed.def \
    --lef-dir Benchmark/PlaceReasoner-Bench/bp_fe/lef \
    --out-dir /…/OpenLayout_macro/work_images \
    --design  bp_fe_u65
```

From a checkpoint ODB instead (one `read_db` restores tech + LEF + placement):

```bash
python macro_pin_side.py --odb run/floorplan.odb --out-dir work_images --design foo
```

### Options
- `--lef LEF1 LEF2 …` — explicit LEFs (tech first) instead of `--lef-dir` glob.
  `--lef-dir` globs `*.lef` and reads any name containing `tech` first.
- `--width` (default 1600) — `save_image` pixel width; height keeps aspect.
- `--margin` (default 0.03) — pad fraction on the die∪pins union, `pad = max(w,h) * margin`.
- `--min-pins` (default 1) — an edge is highlighted as a pin side when it carries at
  least this many signal pins. Default 1 shows **every** edge that has any pins (a
  four-side macro gets four bars). Raise it (e.g. `3`) to suppress edges with only a
  stray pin or two.
- `--no-pins` — skip the IO-pin overlay (pins on by default, matching `image_snapshot`).
- `--no-label` — skip the numbered macro labels next to each bar.
- `--openroad` (default `/usr/bin/openroad`).

### Outputs (in `--out-dir`, prefixed `<design>_`)
- `<design>_pin_side.txt` — the `MACRO_PIN_SIDE` lines (machine-readable).
- `<design>_iopins.txt` — `IO_PIN sig=… box=(…)` lines for the bterm overlay.
- `<design>_pin_side.png` — annotated whole-die-with-pins image.
- `<design>_raw.png` — un-annotated OpenROAD render.
- `<design>_pin_side.tcl` — the generated TCL (re-runnable / inspectable).

The driver also prints a per-macro summary and the side distribution.

## Notes / gotchas
- **Headless render needs `xvfb-run`** in this repo (`offscreen`/`xcb` both fail —
  see memory `feedback_openroad_save_image`). The driver already wraps OpenROAD in
  `xvfb-run -a`.
- `parse_sides()` and `draw_overlay()` are importable if you want to reuse the
  highlight on an image you rendered elsewhere — pass the same micron `area` that
  `save_image -area` used (the driver reads it from the `IMAGE_AREA` stdout line).
- Verified on bp_fe `u65_placed.def`: 11 BLOCK macros, 8×R / 3×L; red line lands on
  the correct geometric edge (right edge present, left edge clean) under the Y-flip.
```
