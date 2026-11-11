#!/bin/bash
# EXPERIMENTAL (otpu_ppk.sv): the pumped packed MXU test's Vivado runs, one at a time, in the order
# given. Each waits until MemAvailable >= MEM_GB (default 12) before it starts. Needs Vivado with a
# VU9P license (XILINXD_LICENSE_FILE as the BCU1525 builds set it). Usage, from the tree's root:
#   rtl/exp/pump_us/run_ooc.sh <outdir> [step ...]
# steps (clk_s 150 MHz, clk_f 300 MHz):
#   col     one pumped column, as small as the real one: 128 PE + 36 column-end DSP48E2s
#   mxu8    the array's fan-out: 8 columns (1,024 PE + 288 column-end DSPs) in SLR2
#   colh, mxu8h   the same with HOLD 1 (every chain end into clk_s over two fast cycles)
#   colb    one column with BF 0 (the activation's fast register is the DSP's B1)
# Results: <outdir>/progress.txt (the OTPU_ lines), <outdir>/<step>/ (reports, routed.dcp).
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
O=${1:?outdir}; shift
mkdir -p "$O"
export PATH=$HOME/.local/bin:/opt/Xilinx/2026.1/Vivado/bin:$PATH
MEM_GB=${MEM_GB:-12}
waitmem() {
  for _ in $(seq 1 240); do
    m=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "$m" -ge "$MEM_GB" ] && return 0
    echo "$(date '+%T') waiting: MemAvailable ${m} GB" >> "$O/progress.txt"
    sleep 60
  done
  return 1
}
viv() {   # name K=V...
  local n=$1; shift
  waitmem || { echo "SKIP $n (memory)" >> "$O/progress.txt"; return; }
  echo "START $n $(date '+%F %T') $*" >> "$O/progress.txt"
  local t0=$(date +%s)
  (cd "$O" && timeout 14400 vivado -mode batch -nojournal -log "$O/$n.log" \
     -source "$ROOT/rtl/exp/pump_us/ooc.tcl" -tclargs "$O/$n" "$@" > "$O/$n.out" 2>&1)
  echo "DONE $n rc=$? $(( $(date +%s) - t0 ))s" >> "$O/progress.txt"
  grep -h "^OTPU_" "$O/$n.out" >> "$O/progress.txt"
}
for s in "$@"; do
  case $s in
    col)   viv col   NCOL=1 ;;
    colh)  viv colh  NCOL=1 HOLD=1 ;;
    colb)  viv colb  NCOL=1 BF=0 ;;
    mxu8)  viv mxu8  NCOL=8 PBLOCK=SLR2 ;;
    mxu8h) viv mxu8h NCOL=8 HOLD=1 PBLOCK=SLR2 ;;
    *) echo "unknown step $s" >> "$O/progress.txt" ;;
  esac
done
echo "EXIT $(date '+%F %T')" >> "$O/progress.txt"
