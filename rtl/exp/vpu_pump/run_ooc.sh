#!/bin/bash
# EXPERIMENTAL (docs/vpu_pump.md): the experiment's Vivado runs, one after another (the build host only,
# through ~/otpu-build/slot.sh). Usage, from the tree:
#   rtl/exp/vpu_pump/run_ooc.sh <outdir> [step ...]
# steps:
#   lane0     8 lanes of today's otpu_fmma at 133.33 MHz (vpu_pump_top VAR 0)
#   lane1     4 pumped units, today's 6 stages, at 266.67 MHz (VAR 1)
#   lane2     4 pumped deep units, 12 stages (XR 1), synthesis retiming (VAR 2)
#   lane2n    the same without retiming
#   lane3     4 pumped deep units, 18 stages (XR 2), retiming
#   lane4     4 pumped split units (vp_fmma_split: 10 stages, the two long ones split by hand)
#   vpu8se    otpu_vpu as built (LANES 8, the stream engine: COMP8 + the tail) at 7.5 ns
#   vpu8      otpu_vpu LANES 8 without the stream engine (CL 2 long lanes) at 7.5 ns
#   vpu8cl4   the same with CL 4
#   vpu16     otpu_vpu LANES 16 (no stream engine: SE needs 8 lanes), CL 4, at 7.5 ns
#   vpu8se2x  otpu_vpu as built at 3.75 ns (the whole VPU at 266.67 MHz: a probe)
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
O=${1:?outdir}; shift
mkdir -p "$O"
export PATH=$HOME/.local/bin:$PATH
viv() {   # name tcl args...
  local n=$1; shift
  echo "START $n $(date '+%F %T')" >> "$O/progress.txt"
  local t0=$(date +%s)
  (cd "$O" && timeout 5400 vivado -mode batch -nojournal -log "$O/$n.log" -source "$@" > "$O/$n.out" 2>&1)
  echo "DONE $n rc=$? $(( $(date +%s) - t0 ))s" >> "$O/progress.txt"
  grep -h "^OTPU_" "$O/$n.out" | sed "s/^/$n /" >> "$O/progress.txt"
}
L=$ROOT/rtl/exp/vpu_pump/ooc_lane.tcl
V=$ROOT/rtl/exp/vpu_pump/ooc_vpu.tcl
for s in "$@"; do
  case $s in
    lane0)    viv lane0 "$L" -tclargs "$O/lane0" 0 1 0 ;;
    lane1)    viv lane1 "$L" -tclargs "$O/lane1" 1 1 0 ;;
    lane2)    viv lane2 "$L" -tclargs "$O/lane2" 2 1 1 ;;
    lane2n)   viv lane2n "$L" -tclargs "$O/lane2n" 2 1 0 ;;
    lane3)    viv lane3 "$L" -tclargs "$O/lane3" 2 2 1 ;;
    lane4)    viv lane4 "$L" -tclargs "$O/lane4" 3 1 0 ;;
    vpu8se)   viv vpu8se "$V" -tclargs "$O/vpu8se" 7.5 8 2 1 ;;
    vpu8)     viv vpu8 "$V" -tclargs "$O/vpu8" 7.5 8 2 0 ;;
    vpu8cl4)  viv vpu8cl4 "$V" -tclargs "$O/vpu8cl4" 7.5 8 4 0 ;;
    vpu16)    viv vpu16 "$V" -tclargs "$O/vpu16" 7.5 16 4 0 ;;
    vpu8se2x) viv vpu8se2x "$V" -tclargs "$O/vpu8se2x" 3.75 8 2 1 ;;
    *) echo "unknown step $s" >> "$O/progress.txt" ;;
  esac
done
echo "EXIT $(date '+%F %T')" >> "$O/progress.txt"
