# Skill: Whole-die-with-pins image snapshot

Render a PNG that shows the **entire die plus the top-level IO pins**, so a
checker (or human) can see the macro placement *and* which die edges the pins
sit on. This is exactly what the agent flow saves at every stage; use the
standalone tool to capture it on demand from any checkpoint.

## What "whole die with pins" means

The saved area is the **die box**, expanded outward by a small `margin` so the
boundary pins are not clipped, and (if enabled) the pin boxes are redrawn on top
because OpenROAD rasterizes thin pins to near-invisibility at the die zoom.

Settings live in `config_macro.yaml`:

```yaml
image:
  save_width: 1024     # save_image -width (px); height keeps the layout aspect
  margin: 0.03         # fractional pad around die/pins (0.03 = 3% of the larger side)
  pin_overlay: true    # redraw bterm boxes over the PNG so pins stay visible
```

## The code path (flow)

1. `macroflow/recipes.py` `DIE_CORE` + `PIN_COORDS` query the die box and every
   bterm pin box from OpenROAD.
2. `macroflow/context.py` `FlowState.image_area(margin)` unions the die (+ any
   out-of-die pin boxes) and pads by `margin`.
3. `macroflow/openroad.py` `ORSession.save_image()` emits
   `save_image -area {x0 y0 x1 y1} -width 1024 image.png`.
4. If `pin_overlay: true`, `macroflow/context.py` `draw_pin_overlay()` draws the
   pin boxes on top of the OpenROAD PNG.

For **bp_fe** (`DIE_AREA 0 0 800 600`, `margin 0.03`) the area becomes
`-24 -24 824 624`, so the manual OpenROAD command is:

```tcl
save_image -area {-24 -24 824 624} -width 1024 whole_die_with_pins.png
```

`margin` is a fraction of the larger die side: `pad = max(w, h) * margin`,
so `800 * 0.03 = 24`.

## Standalone tool: `save_die_image.py`

Reuses the same pipeline (`load_image_settings`, `DIE_CORE`/`PIN_COORDS`,
`FlowState.image_area`, `draw_pin_overlay`), so a manual capture is identical to
the flow's snapshots. Defaults read from `config_macro.yaml`.

```bash
conda activate coder

# From a checkpoint ODB (read_db restores tech + LEF + placement):
python save_die_image.py \
    --odb work/<run>/saves_db/floorplan.odb \
    -o whole_die_with_pins.png

# From a DEF (reads the dataset's lef/*.lef, tech LEF first, then the DEF):
python save_die_image.py \
    --data-dir Benchmark/PlaceReasoner-Bench/bp_fe \
    --def def/pre_place.def \
    -o whole_die_with_pins.png
```

Useful overrides: `--width 2048`, `--margin 0.05`, `--no-pins` (skip the
overlay), `--pins` (force it on). The tool prints the die/core/pin counts and
the exact `save_image` command it emits.

## Notes

- ODB mode is preferred: a single `read_db` restores tech + LEF + placement.
- DEF mode needs the LEFs; it globs `lef/*.lef` and loads any file whose name
  contains `tech` first.
- The session runs headless (`QT_QPA_PLATFORM=minimal`); no X server needed.
