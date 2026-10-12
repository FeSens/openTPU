#!/bin/bash
# openTPU's nangate45-small ASIC block through an OpenLayout-style flow (README.md): Yosys flat
# synthesis, a 60% floorplan, planned macro placement candidates, OpenROAD place / CTS / route.
#
#   run.sh prep                 the SRAM macros and their map (tools/asic/flow.py prep small)
#   run.sh synth                yosys-slang elaboration + scripts/synth.tcl
#   run.sh floorplan            60% utilization, aspect ratio 1, I/O pins
#   run.sh dataflow             macro clusters (OpenLayout dataflow/rtlmp)
#   run.sh macro CAND           candidates/gen_candidates.py + place_macro, pin-side image
#   run.sh baseline             rtl_macro_placer (defaults) as candidate "rtlmp"
#   run.sh pnr CAND             placement, CTS, routing (scripts/pnr.tcl)
#   run.sh snapshot CAND        whole-die PNGs of the macro placement and the routed design
#   run.sh all [CAND]           everything for one candidate (default cand2)
#
# Environment:
#   ORFS_HOME        an OpenROAD-flow-scripts checkout (only flow/platforms/nangate45 is read),
#                    or PLATFORM_DIR = that directory
#   YOSYS            yosys (0.40+), default "yosys"; YOSYS_SLANG the slang plugin ("slang")
#   OPENROAD         openroad, default "openroad"
#   OL_SKILLS        OpenLayout's skills, default skills/ here (a copy of
#                    https://github.com/fenggev56/OpenLayout's, GPL-3.0: skills/LICENSE): the
#                    macro (the pin-side image) and snapshot steps use them
#   OPENLAYOUT_HOME  an OpenLayout checkout: its dataflow/ is used by the dataflow step
#   OL_WORK          work directory, default build/asic/small-openlayout
#   OL_THREADS       OpenROAD threads (8); CLOCK_PERIOD_PS: ABC's target (3300)
set -euo pipefail
OL_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$OL_DIR/../../.." && pwd)
POINT=small
PREP=$ROOT/build/asic/$POINT
export OL_DIR
export OL_WORK=${OL_WORK:-$ROOT/build/asic/$POINT-openlayout}
export OL_MACROS=$PREP/macros
export PLATFORM_DIR=${PLATFORM_DIR:-${ORFS_HOME:?set ORFS_HOME or PLATFORM_DIR}/flow/platforms/nangate45}
YOSYS=${YOSYS:-yosys}
YOSYS_SLANG=${YOSYS_SLANG:-slang}
OPENROAD=${OPENROAD:-openroad}
OPENLAYOUT_HOME=${OPENLAYOUT_HOME:-}
OL_SKILLS=${OL_SKILLS:-$OL_DIR/skills}
PIN_SIDE=$OL_SKILLS/skills_Nangate45/macro_pin_side/macro_pin_side.py
mkdir -p "$OL_WORK"

need() { [ -e "$1" ] || { echo "run.sh: missing $1 ($2)" >&2; exit 1; }; }

prep() {
  (cd "$ROOT" && python3 tools/asic/flow.py prep $POINT)
}

synth() {
  need "$PREP/config.mk" "run.sh prep"
  local out=$OL_WORK/synth
  mkdir -p "$out"
  # the point's sources and parameters, as tools/asic/flow.py gives them to ORFS
  local files params gp=""
  files=$(sed -n 's/^export VERILOG_FILES = //p' "$PREP/config.mk")
  params=$(sed -n 's/^export VERILOG_TOP_PARAMS = //p' "$PREP/config.mk")
  set -- $params
  while [ $# -ge 2 ]; do gp="$gp -G $1=$2"; shift 2; done
  "$YOSYS" -m "$YOSYS_SLANG" -q -l "$out/elab.log" -p \
    "read_slang -D SYNTHESIS --ignore-assertions --top otpu_asic_top $gp $files; hierarchy -top otpu_asic_top; write_rtlil $out/otpu_asic_top.il"
  OL_RTLIL=$out/otpu_asic_top.il OL_SYNTH_OUT=$out \
    "$YOSYS" -c "$OL_DIR/scripts/synth.tcl" -l "$out/synth.log" > /dev/null
  grep -E "Number of cells|Chip area" "$out/synth_stat.txt" | tail -2
  echo "macros: $(wc -l < "$out/macros.txt")"
}

floorplan() {
  need "$OL_WORK/synth/otpu_asic_top_synth.v" "run.sh synth"
  "$OPENROAD" -no_init -exit "$OL_DIR/scripts/floorplan.tcl" > "$OL_WORK/floorplan.log" 2>&1
  grep -E "^(Die|Core) |Design area" "$OL_WORK/floorplan.log"
}

dataflow() {
  need "$OL_WORK/floorplan.def" "run.sh floorplan"
  need "$OPENLAYOUT_HOME/dataflow/rtlmp/run.py" "set OPENLAYOUT_HOME to an OpenLayout checkout"
  local lefs="--lef $PLATFORM_DIR/lef/NangateOpenCellLibrary.tech.lef --lef $PLATFORM_DIR/lef/NangateOpenCellLibrary.macro.mod.lef"
  for f in "$OL_MACROS"/*.lef; do lefs="$lefs --lef $f"; done
  mkdir -p "$OL_WORK/dataflow"
  for mode in "" "--macro-only"; do
    (cd "$OPENLAYOUT_HOME/dataflow" && python3 rtlmp/run.py --verilog "$OL_WORK/synth/otpu_asic_top_synth.v" \
      $lefs --def "$OL_WORK/floorplan.def" --design otpu_small --out-dir "$OL_WORK/dataflow" \
      --cache "$OL_WORK/dataflow/netlist.pkl" $mode) 2>&1 | tail -3
  done
  rm -f "$OL_WORK/dataflow/netlist.pkl"
}

pin_side() {   # the Gate-A image of a candidate (skill macro_pin_side)
  [ -e "$PIN_SIDE" ] || { echo "no pin-side image: $PIN_SIDE missing (OL_SKILLS)"; return 0; }
  python3 "$PIN_SIDE" --odb "$OL_WORK/$1/macro_placed.odb" --lef-dir "$OL_MACROS" \
    --out-dir "$OL_WORK/$1/images" --design otpu_small_$1 --openroad "$OPENROAD" > "$OL_WORK/$1/pin_side.log" 2>&1
  echo "$OL_WORK/$1/images/otpu_small_$1_pin_side.png"
}

macro() {
  need "$OL_WORK/floorplan.odb" "run.sh floorplan"
  python3 "$OL_DIR/candidates/gen_candidates.py" "$OL_WORK/floorplan.log" "$OL_WORK" > /dev/null
  need "$OL_WORK/$1/macro_place.tcl" "candidates: cand1 cand2 cand3"
  OL_CAND=$1 "$OPENROAD" -no_init -exit "$OL_DIR/scripts/macro_place.tcl" > "$OL_WORK/$1/macro_place.log" 2>&1
  grep -E "MACRO_CHECK|ERROR" "$OL_WORK/$1/macro_place.log"
  pin_side "$1"
}

baseline() {
  need "$OL_WORK/floorplan.odb" "run.sh floorplan"
  mkdir -p "$OL_WORK/rtlmp"
  "$OPENROAD" -no_init -exit "$OL_DIR/scripts/rtlmp_place.tcl" > "$OL_WORK/rtlmp/macro_place.log" 2>&1
  pin_side rtlmp
}

pnr() {
  need "$OL_WORK/$1/macro_placed.odb" "run.sh macro $1"
  OL_CAND=$1 "$OPENROAD" -no_init -exit "$OL_DIR/scripts/pnr.tcl" > "$OL_WORK/$1/pnr${OL_START:+_$OL_START}.log" 2>&1
  cat "$OL_WORK/$1/reports/ppa_${OL_STOP:-route}.txt"
}

snapshot() {   # skill image_snapshot: the whole die with its pins
  local t=$OL_SKILLS/skills_openroad_flow/image_snapshot/template/snapshot.tcl
  need "$t" "OL_SKILLS: OpenLayout's skills"
  mkdir -p "$OL_WORK/$1/images"
  for s in macro_placed reports/otpu_asic_top_routed; do
    [ -f "$OL_WORK/$1/$s.odb" ] || continue
    local png=$OL_WORK/$1/images/otpu_small_$1_$(basename $s | sed 's/otpu_asic_top_//').png
    printf 'set in_odb %s\nset out_png %s\nset img_width 1600\nsource %s\n' "$OL_WORK/$1/$s.odb" "$png" "$t" > "$OL_WORK/$1/snapshot.tcl"
    xvfb-run -a "$OPENROAD" -no_init -exit "$OL_WORK/$1/snapshot.tcl" > "$OL_WORK/$1/snapshot.log" 2>&1
    echo "$png"
  done
}

step=${1:-}
case "$step" in
  prep|synth|floorplan|dataflow|baseline) $step ;;
  macro|pnr|snapshot) $step "${2:?candidate: cand1 cand2 cand3 rtlmp}" ;;
  all) c=${2:-cand2}; prep; synth; floorplan; dataflow; macro "$c"; pnr "$c"; snapshot "$c" ;;
  *) sed -n '2,25p' "$0"; exit 2 ;;
esac
