#!/bin/bash
# EXPERIMENTAL (docs/mx_mxu.md): the experiment's Vivado runs in order, one at a time (the build host
# only, under ~/otpu-build/slot.sh). Usage, from the tree:
#   rtl/exp/mx_mxu/run_ooc.sh <outdir> <step> ...
# steps: k7-base4, k7-base8 (base_top MCOLS 4 / 8, DSP48E1), k7-mx<NCOL>e<E> (mx_top pumped),
# us-base4, us-base8 (base_top, DSP48E2), us-mx<NCOL>e<E> (mx_top packed). A step whose
# progress line says DONE rc=0 is skipped.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
O=${1:?outdir}; shift
mkdir -p "$O"
export PATH=$HOME/.local/bin:/opt/Xilinx/2026.1/Vivado/bin:$PATH
K7=xc7k480tffg1156-2
US=xcku5p-ffvb676-2-e
viv() {   # name top fam part generics...
  local n=$1; shift
  if grep -q "^DONE $n rc=0" "$O/progress.txt" 2>/dev/null; then return; fi
  echo "START $n $(date '+%F %T') MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB" >> "$O/progress.txt"
  local t0=$(date +%s)
  (cd "$O" && timeout 10800 vivado -mode batch -nojournal -log "$O/$n.log" \
     -source "$ROOT/rtl/exp/mx_mxu/ooc.tcl" -tclargs "$O/$n" "$@" > "$O/$n.out" 2>&1)
  echo "DONE $n rc=$? $(( $(date +%s) - t0 ))s" >> "$O/progress.txt"
  grep -h "^OTPU_" "$O/$n.out" >> "$O/progress.txt"
}
for s in "$@"; do
  case $s in
    k7-base*) viv "$s" base_top x7 $K7 MCOLS=${s#k7-base} ;;
    us-base*) viv "$s" base_top us $US MCOLS=${s#us-base} ;;
    k7-mx*) c=${s#k7-mx}; viv "$s" mx_top x7 $K7 NCOL=${c%e*} E=${c#*e} ;;
    us-mx*) c=${s#us-mx}; viv "$s" mx_top us $US NCOL=${c%e*} E=${c#*e} ;;
    *) echo "unknown step $s" >> "$O/progress.txt" ;;
  esac
done
echo "EXIT $(date '+%F %T')" >> "$O/progress.txt"
