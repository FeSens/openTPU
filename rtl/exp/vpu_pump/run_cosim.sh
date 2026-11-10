#!/bin/bash
# EXPERIMENTAL (docs/vpu_pump.md): co-simulation lists, P runs at a time. Each line of a list is
# "TAG perf_qwen-arguments", "TAG moe: moe_prefill_cosim-arguments" or "TAG zero: vpu_hack.py
# cosim-arguments" (a zero image from config.json, a layer prefix); every run is the board's
# configuration (PAIR, DSTEP, STREAM; LiteDRAM's controller at DDR3-1066, the core at 133.33 MHz).
# Results: <outdir>/<TAG>.log, and perf_qwen's JSON lines in <outdir>/<list name>.jsonl. A tag
# whose log ends in rc=0 is skipped. LOCK=<file>: each run under lockf on that file (a laptop's
# shared heavy-job lock). Usage, from anywhere:
#   rtl/exp/vpu_pump/run_cosim.sh <outdir> <P> <list> ...
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
O=${1:?outdir}; P=${2:?parallel}; shift 2
mkdir -p "$O"
export OTPU_PAIR=1 OTPU_DSTEP=1 OTPU_STREAM=1 PYTHONPATH=$ROOT
one() {   # list tag args...
  local js=$1 tag=$2; shift 2
  grep -q "^rc=0" "$O/$tag.log" 2>/dev/null && return 0
  local t0=$(date +%s)
  cd "$ROOT" || return 1
  if [ "$1" = "env:" ]; then     # "env: K=V ... --": the run's own environment
    shift
    while [ "$1" != "--" ]; do export "$1"; shift; done
    shift
  fi
  local lk=()
  [ -n "$LOCK" ] && lk=(lockf -k "$LOCK")
  if [ "$1" = "moe:" ]; then
    shift
    "${lk[@]}" python3 tools/moe_prefill_cosim.py "$@" > "$O/$tag.log" 2>&1
  elif [ "$1" = "zero:" ]; then
    shift
    "${lk[@]}" python3 tools/vpu_hack.py cosim "$@" > "$O/$tag.log" 2>&1
  else
    "${lk[@]}" python3 tools/perf_qwen.py --ddr 1066 --mhz 133.33 --ldc --json "$js" "$@" \
      > "$O/$tag.log" 2>&1
  fi
  echo "rc=$? $(( $(date +%s) - t0 ))s" >> "$O/$tag.log"
  echo "$(date '+%T') $tag $(tail -1 "$O/$tag.log")" >> "$O/progress.txt"
}
export -f one
export O ROOT LOCK
for list in "$@"; do
  js="$O/$(basename "$list" .txt).jsonl"
  grep -v '^#' "$list" | grep -v '^ *$' | while read -r tag args; do echo "$js $tag $args"; done
done | xargs -P "$P" -L 1 bash -c 'one $0 "$@"'
echo "EXIT $(date '+%F %T')" >> "$O/progress.txt"
