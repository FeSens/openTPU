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
#   per-position and resident, and a final selftest. A bitstream with the decode loop (CAPS
#   bit30) also runs it for all six, token-exact against the same references, and
#   decode_profile --card-loop, greedy and sampled, for the 4-bit ones (wall against device
#   tok/s); GEN=0 / 1
#   overrides the bitstream's bit. A bitstream with WAITW (CAPS bit31; WAITW=0 / 1 overrides it)
#   runs tools/qual/waitw.py after the warm diag: 200 rounds of the host writing data, then a
#   flag, while the card waits on the flag and then reads the data, and a WAITW timeout.
# full (~45 min): also a cold diag with the full march C- (2 x 2.6 min), decode_profile for all
#   six, rw_bench, a 5 min soak and the full march in the warm diag.
# Every phase prints its duration; the table is at the end and in $OUT/phases.tsv.
#
# A check that crashes counts: every step's output goes to $OUT/logs/, and a step that exits
# non-zero (a timeout too) or prints a Python exception is a FAIL line in $OUT/checks.txt, as is
# a token-exact run that did not pass. The model phases need the tree's checkpoints (models/, as
# opentpu.llm.model_dir finds them: a staged tree needs its models link); without them they are
# skipped and that is a FAIL line. SOAK (seconds) overrides the profile's warm soak.
set -u
DEP=${1:?deploy dir}; PROFILE=${2:-fast}
case $DEP in /*) BIT=$DEP/otpu.bit ;; *) BIT=~/otpu-build/$DEP/otpu.bit ;; esac
NAME=$(basename "$DEP")
H=$(cd "$(dirname "$0")/../.." && pwd); P=${PYTHON:-~/otpu-venv/bin/python}
REST=${REST:-$BIT}; OUT=${OUT:-/tmp/qual-$NAME}; mkdir -p "$OUT"
cd "$H" || exit 1; export PYTHONPATH=$H
RUNS="qwen3:int8:- lfm2:int8:- qwen35:int8:- qwen3:fp4:int8 lfm2:fp4:int8 qwen35:fp4:int8"
if [ "$PROFILE" = full ]; then
  SOAK=${SOAK:-300}; COLD=1; WARM_MEM=full; DP_RUNS=$RUNS; RW=1
else
  SOAK=${SOAK:-180}; COLD=0; WARM_MEM=quick; DP_RUNS="qwen3:fp4:int8 lfm2:fp4:int8 qwen35:fp4:int8"; RW=0
fi
: > "$OUT/phases.tsv"; : > "$OUT/checks.txt"; mkdir -p "$OUT/logs"
PH=""; PT=0; T00=$(date +%s)
memgb() { awk '/MemAvailable/ {printf "%.1f", $2 / 1048576}' /proc/meminfo 2>/dev/null; }
phase() {
  local now; now=$(date +%s)
  if [ -n "$PH" ]; then printf '%s\t%d\n' "$PH" $((now - PT)) >> "$OUT/phases.tsv"
    echo "--- $PH: $((now - PT)) s"; fi
  PH=$1; PT=$now
  [ -n "$PH" ] && echo "=== $PH $(date +%T)  load $(cut -d' ' -f1 /proc/loadavg 2>/dev/null), MemAvailable $(memgb) GiB"
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
    rescan 2>&1 | tee -a "$OUT/rescan.log" | tail -1 | tee "$OUT/rescan"   # rescan.log: every step (relink)
    grep -q "ID 0x4f545055" "$OUT/rescan" && return 0; sleep 10; done
  echo "rescan failed for 10 minutes"; return 1; }
# a FAIL line: into checks.txt, and on stderr (not into the callers' filters and tees)
fail() { echo "  [FAIL] $PH: $*" >> "$OUT/checks.txt"; echo "  [FAIL] $PH: $*" >&2; }
# run LABEL CMD...: CMD's output (stdout and stderr) into $OUT/logs/NNN-LABEL.txt, then printed
# for the caller's filter. A non-zero exit (timeout's 124 too) or a Python exception is a FAIL
# line: a crash never passes silently through the filters below. Returns 1 when it is one.
EXC='^Traceback \(most recent call last\)|^([A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*(Error|Exception)(: |$)'
run() {
  local label=$1; shift
  local n; n=$(( $(ls "$OUT/logs" | wc -l) + 1 ))        # counted: run may be in a subshell
  local log; log="$OUT/logs/$(printf %03d $n)-$(printf %s "$label" | tr -c 'A-Za-z0-9_.-' _).txt"
  "$@" > "$log" 2>&1; local rc=$?
  local exc; exc=$(grep -E "$EXC" "$log" | grep -v '^Traceback' | tail -1 | cut -c1-200)
  local bad=0
  if [ $rc -ne 0 ] || grep -qE "$EXC" "$log"; then
    fail "$label: exit $rc${exc:+, $exc} ($log)"; bad=1
  fi
  cat "$log"
  return $bad
}
selftest() { run selftest timeout 1800 $P -m opentpu.host.selftest \
  | grep -E "\[(PASS|FAIL)\]|config|ALL|stopped|hint" | tee -a "$OUT/checks.txt"; }
diag() {  # $1 label, $2 quick|full
  run "diag $1" timeout 3600 $P -m opentpu.host.diag --mem "$2" --soak 20 > "$OUT/diag-$1.txt"
  grep -E "\[(PASS|FAIL)\].*(RDOT|OUTER|LOG2)" "$OUT/diag-$1.txt"
  sed -n '/^summary/,$p' "$OUT/diag-$1.txt" | tee -a "$OUT/checks.txt"
}
# the model phases' checkpoints, where the tools look for them (opentpu.llm.model_dir)
models_ok() {
  local m d bad=""
  for m in $(for r in $RUNS; do echo "${r%%:*}"; done | sort -u); do
    d=$($P -c "from opentpu.llm import model_dir; print(model_dir('$m'))" 2>/dev/null)
    [ -n "$d" ] && [ -f "$d/config.json" ] || bad="$bad $m (${d:-model_dir failed})"
  done
  [ -z "$bad" ] && return 0
  fail "no checkpoint for$bad: $H/models does not hold the models (a staged tree needs its" \
       "models link, e.g. models -> ~/openTPU/models); the model phases are skipped"
  return 1
}
temp() { $P -m opentpu.host.smi 2>&1 | grep -oE "Temp [^ ]+ [^ │]+" | head -1; }
finish() {
  phase ""
  echo "=== phases ($PROFILE)"; column -t -s $'\t' "$OUT/phases.tsv"
  echo "total $(( ($(date +%s) - T00) / 60 )) min; $(grep -c '\[FAIL\]' "$OUT/checks.txt") FAIL lines" \
       "($(grep -c '\[PASS\]' "$OUT/checks.txt") PASS); results in $OUT"
  echo "QUAL DONE $NAME $(date +%T)"
}
trap finish EXIT

echo "################ $NAME, $PROFILE profile, host tree $(git -C "$H" rev-parse --short HEAD) $(date +%T)"
phase "load + selftest"
if [ "${LOAD:-1}" = 0 ]; then echo "LOAD=0: no JTAG load, the card keeps its bitstream"
else load "$BIT" || exit 1; fi
selftest
run "refs cfg" $P tools/qual/refs.py cfg "$OUT/cfg.pkl" --name "$NAME" | cut -c1-200
MODELS=1; models_ok || MODELS=0
CAPS=$($P -c "from opentpu.host.board import Board, XdmaTransport
c = Board(XdmaTransport('/dev/xdma0', dma=False), check=False, lock=False).info()['caps']
print(int(bool(c.get('gen'))), int(bool(c.get('waitw'))))" 2>/dev/null || echo 0 0)
GEN=${GEN:-${CAPS% *}}; WAITW=${WAITW:-${CAPS#* }}
echo "decode loop on the card (CAPS bit30): $([ "$GEN" = 1 ] && echo yes || echo no)," \
     "WAITW (CAPS bit31): $([ "$WAITW" = 1 ] && echo yes || echo no)"

if [ $MODELS = 1 ]; then phase "references (background)"
  ( $P tools/qual/refs.py compute "$OUT/cfg.pkl" > "$OUT/refs.log" 2>&1; echo "refs exit $?" >> "$OUT/refs.log" ) &
  sleep 3; head -8 "$OUT/refs.log"
fi

if [ $COLD = 1 ]; then phase "diag cold (full march)"; temp; diag cold full; fi

if [ $MODELS = 1 ]; then
phase "prefill + decode counters (6)"
for r in $RUNS; do IFS=: read -r m w h <<< "$r"
  run "perf $m $w $h" timeout 1800 $P tools/qual/perf.py "$m" "$w" "$h" | grep -vE "Warning|warn\("
done

phase "decode_profile ($(set -- $DP_RUNS; echo $#))"
for r in $DP_RUNS; do IFS=: read -r m w h <<< "$r"; args="--wformat $w"; [ "$h" != "-" ] && args="$args --head-format $h"
  run "decode_profile $m $w $h" timeout 1800 $P tools/decode_profile.py --model "$m" --greedy \
    --tokens 96 $args --json "$OUT/dp-$m-$w-$h.json" > "$OUT/dp-$m-$w-$h.txt"
  grep -E " on board|^host critical|^wall|Error" "$OUT/dp-$m-$w-$h.txt"
done
fi

if [ $RW = 1 ]; then phase "rw_bench"
  run rw_bench timeout 900 $P tools/rw_bench.py card --modes mm,mm+st,mm+dstep,dstep | tail -4; fi

if [ $MODELS = 1 ]; then
phase "warm soak ${SOAK} s"; temp
t0=$(date +%s); n=0
while [ $(( $(date +%s) - t0 )) -lt $SOAK ]; do
  run "soak run $((n + 1))" timeout 900 $P tools/decode_profile.py --model qwen3 --greedy \
    --tokens 256 --prompt "Write a long essay about the history of France." > "$OUT/soak-run.txt"
  ok=$?; grep -E "^wall|Error|error" "$OUT/soak-run.txt"
  n=$((n+1))
  [ $ok = 0 ] || break          # a failed run ends the soak (its FAIL line), not a loop of them
done; echo "warm soak: $n runs in $(( $(date +%s) - t0 )) s"; temp
fi

phase "diag warm ($WARM_MEM memory test)"; diag warm $WARM_MEM

if [ "$WAITW" = 1 ]; then phase "WAITW on the host's writes"
  run waitw timeout 900 $P tools/qual/waitw.py --rounds 200 | grep -E "\[(PASS|FAIL)\]" \
    | tee -a "$OUT/checks.txt"
fi

if [ $MODELS = 1 ]; then
phase "token-exact after the soak (6 x per-position + resident)"
grep -E "FAILED|refs exit" "$OUT/refs.log"
p0=$(grep -c '\[PASS\] model' "$OUT/checks.txt")
for r in $RUNS; do IFS=: read -r m w h <<< "$r"
  for res in "" --resident; do
    run "token-exact $m $w $h$res" timeout 1800 $P tools/qual/refs.py card "$OUT/cfg.pkl" "$m" "$w" "$h" 32 $res \
      | grep -E "\] model|Error|Traceback" | tee -a "$OUT/checks.txt"
  done
done
want=$(( $(echo $RUNS | wc -w) * 2 )); got=$(( $(grep -c '\[PASS\] model' "$OUT/checks.txt") - p0 ))
[ "$got" -eq "$want" ] || fail "$got of $want token-exact runs passed"
if [ "$GEN" = 1 ]; then
phase "decode loop on the card ($(set -- $RUNS; echo $#) token-exact + $(set -- $DP_RUNS; echo $#) x 2 decode_profile)"
p0=$(grep -c '\[PASS\] model' "$OUT/checks.txt")
for r in $RUNS; do IFS=: read -r m w h <<< "$r"
  run "card loop $m $w $h" timeout 1800 $P tools/qual/refs.py card "$OUT/cfg.pkl" "$m" "$w" "$h" 32 --card-loop \
    | grep -E "\] model|Error|Traceback" | tee -a "$OUT/checks.txt"
done
want=$(set -- $RUNS; echo $#); got=$(( $(grep -c '\[PASS\] model' "$OUT/checks.txt") - p0 ))
[ "$got" -eq "$want" ] || fail "$got of $want card-loop token-exact runs passed"
for r in $DP_RUNS; do IFS=: read -r m w h <<< "$r"; args="--wformat $w"; [ "$h" != "-" ] && args="$args --head-format $h"
  for mode in greedy sampled; do         # sampled: the model's chat defaults (chat.SAMPLING)
    run "decode_profile card loop $m $w $h $mode" timeout 1800 $P tools/decode_profile.py --model "$m" \
      $([ $mode = greedy ] && echo --greedy) --tokens 96 $args --card-loop \
      --json "$OUT/dpl-$m-$w-$h-$mode.json" > "$OUT/dpl-$m-$w-$h-$mode.txt"
    grep -E "decode loop on the card|Error" "$OUT/dpl-$m-$w-$h-$mode.txt"
  done
done
fi
wait                            # the references' job (refs.py card waited for what it needed)
rc=$(sed -n 's/^refs exit //p' "$OUT/refs.log" | tail -1)
{ [ "${rc:-?}" = 0 ] && ! grep -qE "$EXC" "$OUT/refs.log"; } || fail "refs compute: exit ${rc:-?} ($OUT/refs.log)"
fi

phase "final selftest"
if [ "$REST" != "$BIT" ] && [ "${LOAD:-1}" != 0 ]; then load "$REST" || exit 1; fi
selftest
sudo -n dmesg | grep -iE "xdma.*(timed out|error|fail)" | tail -3
