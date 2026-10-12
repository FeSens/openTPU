---
name: openroad-init-script
description: >
  Generate a parameterized OpenROAD initialization TCL script for the
  physical-design read-in stage — loads PDK (liberty + LEF), reads the
  design (Verilog and/or DEF), constraints (SDC), and sets up parasitics.
  Use this skill whenever the user asks for an OpenROAD init / setup /
  read-in script, wants to adapt an existing init script to a new design
  or PDK (Nangate45, Sky130, ASAP7, GF180), works with benchmarks like
  ariane, gcd, aes, ibex, MacroPlacement, or mentions
  read_liberty, read_lef, read_def, read_verilog, read_sdc,
  define_corners, set_wire_rc, or setDefaultParasitics — including when
  they paste a TCL snippet and ask to "change the paths".
---

# OpenROAD Init Script

Generates the read-in stage of an OpenROAD flow. Use
`assets/init_template.tcl` as the starting shape.

## Required order

Reordering breaks things, often silently:

1. **Path variables** at the top (`pdk_dir`, `design_dir`, `top_design`)
2. **`define_corners`** — Nangate45 has only `tt`; ASAP7/Sky130 have `ss tt ff`
3. **`read_liberty`** — std cells first (sets unit system), then macros
4. **`read_lef`** — tech LEF first, then cell LEF, then macro LEFs
5. **`setDefaultParasitics` proc** wrapping `set_wire_rc -signal/-clock -layer ...`
6. **`set proj_name` / `set top_design`**
7. **`read_verilog`**
8. **`read_def`**
9. **`read_sdc`**


## Wire-RC layers per PDK

`set_wire_rc -layer` uses the PDK's actual layer naming — don't invent it.

| PDK       | Signal   | Clock    |
|-----------|----------|----------|
| Nangate45 | `metal3` | `metal5` |
| Sky130HD  | `met3`   | `met5`   |
| ASAP7     | `M3`     | `M5`     |
| GF180     | `Metal3` | `Metal5` |

## Output style

- Variables at the top, body uses `${var}`
- Terse `# action` comments before each block (match the user's comment
  style if they showed one)
- Keep / add `puts "Init ..."` lines before slow reads
- Mirror the source script's shape — same section order, same density.
  This is a path/parameter swap, not a refactor.

## Things to skip / preserve based on intent

- **`macro_placed.def` or similar already-floorplanned DEF** → do NOT
  add `initialize_floorplan`; the user wants to preserve the existing
  core size and macro locations
- **DEF-only** (no Verilog) → drop `read_verilog`
- **Verilog-only** (no DEF) → drop `read_def`

## Pitfalls

- LEF before liberty → wrong units for `set_wire_rc`
- DEF before Verilog → modules not yet in DB
- `set_wire_rc` before any `read_liberty` → garbage values
- Hardcoded paths in the body instead of `${var}`
