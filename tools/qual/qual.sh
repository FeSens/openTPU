#!/bin/bash
# Card qualification of a candidate bitstream (docs/board.md, "Qualifying a bitstream").
#
#   otpu-lock --wait 3600 -- bash tools/qual/qual.sh DEPLOY_DIR [fast|full]
#
# DEPLOY_DIR holds otpu.bit (a path, or a name under ~/otpu-build). Runs from the host tree the
# script is in. Environment: REST (the bitstream to leave on the card; default the candidate),
# OUT (results; default /tmp/qual-<deploy>), REFCACHE (tools/qual/refs.py), LOAD=0 (no JTAG load:
# qualify the bitstream the card runs, e.g. on opentpu, whose root port once did not bring the
# link back on a hot rescan after a reload; the selftest's config line names the build).
#
# fast (~25 min with cached references): selftest (with the RDOT/OUTER/LOG2 op checks), the ISA
#   references in the background, prefill + DRAM efficiency for the six configurations, the
#   streamed decode (decode_profile) of the three 4-bit ones, a 3 min warm soak, diag with the
#   quick memory test, then after the soak token-exact against the ISA simulator for all six,
#   per-position and resident, and a final selftest.
# full (~45 min): also a cold diag with the full march C- (2 x 2.6 min), decode_profile for all
#   six, rw_bench, a 5 min soak and the full march in the warm diag.
# Every phase prints its duration; the table is at the end and in $OUT/phases.tsv.
set -u
DEP=${1:?deploy dir}; PROFILE=${2:-fast}
case $DEP in /*) BIT=$DEP/otpu.bit ;; *) BIT=~/otpu-build/$DEP/otpu.bit ;; esac
NAME=$(basename "$DEP")
H=$(cd "$(dirname "$0")/../.." && pwd); P=${PYTHON:-~/otpu-venv/bin/python}
REST=${REST:-$BIT}; OUT=${OUT:-/tmp/qual-$NAME}; mkdir -p "$OUT"
cd "$H" || exit 1; export PYTHONPATH=$H
RUNS="qwen3:int8:- lfm2:int8:- qwen35:int8:- qwen3:fp4:int8 lfm2:fp4:int8 qwen35:fp4:int8"
if [ "$PROFILE" = full ]; then
  SOAK=300; COLD=1; WARM_MEM=full; DP_RUNS=$RUNS; RW=1
else
  SOAK=180; COLD=0; WARM_MEM=quick; DP_RUNS="qwen3:fp4:int8 lfm2:fp4:int8 qwen35:fp4:int8"; RW=0
fi
: > "$OUT/phases.tsv"
PH=""; PT=0; T00=$(date +%s)
memgb() { awk '/MemAvailable/ {printf "%.1f", $2 / 1048576}' /proc/meminfo; }
phase() {
  local now; now=$(date +%s)
  if [ -n "$PH" ]; then printf '%s\t%d\n' "$PH" $((now - PT)) >> "$OUT/phases.tsv"
    echo "--- $PH: $((now - PT)) s"; fi
  PH=$1; PT=$now
  [ -n "$PH" ] && echo "=== $PH $(date +%T)  load $(cut -d' ' -f1 /proc/loadavg), MemAvailable $(memgb) GiB"
}
# root helpers: opentpu has narrow sudo rules (/etc/sudoers.d/60-otpu-card) for these exact commands
rescan() { if [ -x /usr/local/sbin/otpu-rescan ]; then sudo -n /usr/local/sbin/otpu-rescan; else sudo ~/otpu-venv/bin/otpu-setup --rescan; fi; }
others() { sudo -n lsof -t /dev/xdma0_* 2>/dev/null | head -1; }
quiet() { for _ in $(seq 120); do [ -z "$(others)" ] && return 0; sleep 5; done
  echo "device open by pid $(others)"; return 1; }
load() {
  quiet || return 1
  openFPGALoader -c digilent_hs2 --freq 10000000 "$1" 2>&1 | tail -1
  for _ in $(seq 60); do
    rescan 2>&1 | tail -1 | tee "$OUT/rescan"
    grep -q "ID 0x4f545055" "$OUT/rescan" && return 0; sleep 10; done
  echo "rescan failed for 10 minutes"; return 1; }
selftest() { timeout 1800 $P -m opentpu.host.selftest 2>&1 \
  | grep -E "\[(PASS|FAIL)\]|config|ALL|stopped|hint" | tee -a "$OUT/checks.txt"; }
diag() {  # $1 label, $2 quick|full
  timeout 3600 $P -m opentpu.host.diag --mem "$2" --soak 20 > "$OUT/diag-$1.txt" 2>&1
  grep -E "\[(PASS|FAIL)\].*(RDOT|OUTER|LOG2)" "$OUT/diag-$1.txt"
  sed -n '/^summary/,$p' "$OUT/diag-$1.txt" | tee -a "$OUT/checks.txt"
}
temp() { $P -m opentpu.host.smi 2>&1 | grep -oE "Temp [^ ]+ [^ │]+" | head -1; }
finish() {
  phase ""
  echo "=== phases ($PROFILE)"; column -t -s $'\t' "$OUT/phases.tsv"
  echo "total $(( ($(date +%s) - T00) / 60 )) min; $(grep -c '\[FAIL\]' "$OUT/checks.txt" 2>/dev/null || echo 0) FAIL lines; results in $OUT"
  echo "QUAL DONE $NAME $(date +%T)"
}
trap finish EXIT

echo "################ $NAME, $PROFILE profile, host tree $(git -C "$H" rev-parse --short HEAD) $(date +%T)"
phase "load + selftest"
if [ "${LOAD:-1}" = 0 ]; then echo "LOAD=0: no JTAG load, the card keeps its bitstream"
else load "$BIT" || exit 1; fi
selftest
$P tools/qual/refs.py cfg "$OUT/cfg.pkl" --name "$NAME" | cut -c1-200

phase "references (background)"
( $P tools/qual/refs.py compute "$OUT/cfg.pkl" > "$OUT/refs.log" 2>&1; echo "refs exit $?" >> "$OUT/refs.log" ) &
sleep 3; head -8 "$OUT/refs.log"

if [ $COLD = 1 ]; then phase "diag cold (full march)"; temp; diag cold full; fi

phase "prefill + decode counters (6)"
for r in $RUNS; do IFS=: read -r m w h <<< "$r"
  timeout 1800 $P tools/qual/perf.py "$m" "$w" "$h" 2>&1 | grep -vE "Warning|warn\("
done

phase "decode_profile ($(echo $DP_RUNS | wc -w))"
for r in $DP_RUNS; do IFS=: read -r m w h <<< "$r"; args="--wformat $w"; [ "$h" != "-" ] && args="$args --head-format $h"
  timeout 1800 $P tools/decode_profile.py --model "$m" --greedy --tokens 96 $args \
    --json "$OUT/dp-$m-$w-$h.json" > "$OUT/dp-$m-$w-$h.txt" 2>&1
  grep -E " on board|^host critical|^wall|Error" "$OUT/dp-$m-$w-$h.txt"
done

if [ $RW = 1 ]; then phase "rw_bench"
  timeout 900 $P tools/rw_bench.py card --modes mm,mm+st,mm+dstep,dstep 2>&1 | tail -4; fi

phase "warm soak ${SOAK} s"; temp
t0=$(date +%s); n=0
while [ $(( $(date +%s) - t0 )) -lt $SOAK ]; do
  timeout 900 $P tools/decode_profile.py --model qwen3 --greedy --tokens 256 \
    --prompt "Write a long essay about the history of France." 2>&1 | grep -E "^wall|Error|error"
  n=$((n+1))
done; echo "warm soak: $n runs in $(( $(date +%s) - t0 )) s"; temp

phase "diag warm ($WARM_MEM memory test)"; diag warm $WARM_MEM

phase "token-exact after the soak (6 x per-position + resident)"
grep -E "FAILED|refs exit" "$OUT/refs.log"
for r in $RUNS; do IFS=: read -r m w h <<< "$r"
  for res in "" --resident; do
    timeout 1800 $P tools/qual/refs.py card "$OUT/cfg.pkl" "$m" "$w" "$h" 32 $res 2>&1 \
      | grep -E "\] model|Error|Traceback" | tee -a "$OUT/checks.txt"
  done
done

phase "final selftest"
if [ "$REST" != "$BIT" ] && [ "${LOAD:-1}" != 0 ]; then load "$REST" || exit 1; fi
selftest
sudo -n dmesg | grep -iE "xdma.*(timed out|error|fail)" | tail -3
