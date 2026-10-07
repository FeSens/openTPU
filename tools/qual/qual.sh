#!/bin/bash
# Card qualification of a candidate bitstream (docs/board.md, "Qualifying a bitstream").
#
#   bash tools/qual/qual.sh DEPLOY_DIR [fast|full]
#
# Started without the card lock, it quantizes the 4-bit runs' weights into the image cache
# first (opentpu.host.prebuild, at nice 19), then takes the lock (otpu-lock, waiting up to
# LOCK_WAIT seconds, default 3600) and runs under it; started under `otpu-lock -- ...` it runs
# at once, and the tools quantize what the cache does not hold under the lock.
# DEPLOY_DIR holds otpu.bit (a path, or a name under ~/otpu-build). Runs from the host tree the
# script is in. Environment: REST (the bitstream to leave on the card; default the candidate),
# OUT (results; default /tmp/qual-<deploy>), REFCACHE (tools/qual/refs.py), LOAD=0 (no JTAG load:
# qualify the bitstream the card runs, e.g. on opentpu, whose root port once did not bring the
# link back on a hot rescan after a reload), RUNS (the model runs MODEL:WF:HF, default the six
# of Qwen3, LFM2.5 and Qwen3.5 in int8 and 4-bit; e.g. RUNS="lfm2-2.6b:int8:- lfm2-2.6b:fp4:int8"
# for the other models of opentpu.llm.MODELS), LONG (the long prompt's run, MODEL:WF:HF, default
# lfm2:int8:-; empty: none), NTOK (the token-exact runs' tokens, default 32).
#
# The card must run the candidate: its BUILD_ID (Board.info) must be EXPECT, else the last 8
# hex digits of the deploy's name (deploy_fpfix_e776703a -> e776703a), checked after the JTAG
# load (with LOAD=0 at the start) and again at the end; after REST's load, REST's build
# (REST_EXPECT, else its directory's name's, the link resolved). A failed JTAG load, another
# build on the card or a probe of BUILD_ID and CAPS that fails is a FAIL line and stops the
# qual (REST is loaded back if the candidate was).
#
# fast (~25 min with cached references): selftest (with the RDOT/OUTER/LOG2 op checks), the ISA
#   references in the background, prefill + DRAM efficiency for the six configurations, the
#   streamed decode (decode_profile) of the three 4-bit ones, a 3 min warm soak, diag with the
#   quick memory test, then after the soak token-exact against the ISA simulator: for all six,
#   per-position and resident, NTOK greedy tokens past EOS with every logits vector bit for bit,
#   and LONG's long prompt (240 tokens, the decode across position 256: a run per attention
#   bucket) per-position, resident and resident in prompt runs (docs/prefill.md); then a final
#   selftest. A bitstream with the decode loop (CAPS bit30) also runs it for all six and LONG's
#   two (plain and in prompt runs), token-exact against the same references, and
#   decode_profile --card-loop, greedy and sampled, for the 4-bit ones (wall against device
#   tok/s); GEN=0 / 1 overrides the bitstream's bit. A bitstream with WAITW (CAPS bit31;
#   WAITW=0 / 1 overrides it) runs tools/qual/waitw.py after the warm diag: 200 rounds of the
#   host writing data, then a flag, while the card waits on the flag and then reads the data,
#   and a WAITW timeout. Before the final selftest, tools/qual/turnaround.py: TURN seconds
#   (default 30) of fp4 weight reads beside 64 KiB stores and loads (the DRAM's read / write
#   turnarounds; fastmux's rtw 3), the data against the ISA simulator, and both channels' ECC
#   counters, which must be 0 then; and the kernel's xdma errors since the load (dmesg), each a
#   FAIL line.
# full (~45 min): also a cold diag with the full march C- (2 x 2.6 min), decode_profile for all
#   six, rw_bench, a 5 min soak and the full march in the warm diag.
# Every phase prints its duration; the table is at the end and in $OUT/phases.tsv.
#
# A check that crashes counts: every step's output goes to $OUT/logs/, and a step that exits
# non-zero (a timeout too) or prints a Python exception is a FAIL line in $OUT/checks.txt, as is
# a token-exact run that did not pass (or did not test what it says: resident decode not
# engaged, prompt runs not taken). The model phases need the tree's checkpoints (models/, as
# opentpu.llm.model_dir finds them: a staged tree needs its models link); without them they are
# skipped and that is a FAIL line. The references' job gets REFS_WAIT seconds (default 900)
# after the last token-exact run, then it is killed and that is a FAIL line. SOAK (seconds)
# overrides the profile's warm soak; PREBUILD=0 skips the prebuild. The exit status is 0 only
# when the qual ran to its end with no FAIL line.
set -u
DEP=${1:?deploy dir}; PROFILE=${2:-fast}
case $DEP in /*) BIT=$DEP/otpu.bit ;; *) BIT=~/otpu-build/$DEP/otpu.bit ;; esac
NAME=$(basename "$DEP")
H=$(cd "$(dirname "$0")/../.." && pwd); P=${PYTHON:-~/otpu-venv/bin/python}
REST=${REST:-$BIT}; OUT=${OUT:-/tmp/qual-$NAME}; mkdir -p "$OUT"
cd "$H" || exit 1; export PYTHONPATH=$H
RUNS=${RUNS:-"qwen3:int8:- lfm2:int8:- qwen35:int8:- qwen3:fp4:int8 lfm2:fp4:int8 qwen35:fp4:int8"}
LONG=${LONG-lfm2:int8:-}; NTOK=${NTOK:-32}
# Started without the card lock: first the 4-bit runs' weights into the image cache
# (opentpu.host.prebuild at nice 19, outside the lock; opentpu/qcache.py), so the session
# quantizes nothing under the lock; then this script again, under it. PREBUILD=0: no prebuild.
if [ -z "${OTPU_LOCK_HELD:-}" ]; then
  PB=""
  if [ "${PREBUILD:-1}" != 0 ]; then
    for r in $RUNS $LONG; do case ":${r#*:}:" in *:fp4:*|*:int4:*) PB="$PB --prebuild $r" ;; esac; done
    kept=${REFCACHE:-$HOME/otpu-build/refcache}/configs/$NAME.pkl
    [ -n "$PB" ] && [ -f "$kept" ] && PB="$PB --prebuild-cfg $kept"
  fi
  echo "=== prebuild outside the card lock${PB:+:$PB}; then the lock (up to ${LOCK_WAIT:-3600} s) $(date +%T)"
  exec $P -m opentpu.host.runstate --wait "${LOCK_WAIT:-3600}" $PB -- bash "$H/tools/qual/qual.sh" "$@"
fi
if [ "$PROFILE" = full ]; then
  SOAK=${SOAK:-300}; COLD=1; WARM_MEM=full; DP_RUNS=$RUNS; RW=1
else
  SOAK=${SOAK:-180}; COLD=0; WARM_MEM=quick; RW=0
  DP_RUNS=""; for r in $RUNS; do [[ $r == *:fp4:* ]] && DP_RUNS="$DP_RUNS $r"; done
fi
: > "$OUT/phases.tsv"; : > "$OUT/checks.txt"; mkdir -p "$OUT/logs"
PH=""; PT=0; T00=$(date +%s)
REFS=""; STOPPED=""; LOADED=0; RESTORED=""; K0=0; GENC=0; WAITWC=0
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
# load BIT: JTAG (openFPGALoader's exit status counts; its output in $OUT/jtag.log), then the
# rescan until the card answers; 1, and why, on a failure. Which build the card then runs is
# check_build's to say: every openTPU bitstream answers the rescan with the same ID.
load() {
  quiet || return 1
  echo "=== $1 $(date +%T)" >> "$OUT/jtag.log"
  openFPGALoader -c digilent_hs2 --freq 10000000 "$1" >> "$OUT/jtag.log" 2>&1; local rc=$?
  tail -1 "$OUT/jtag.log"
  [ $rc = 0 ] || { echo "openFPGALoader exit $rc ($OUT/jtag.log)"; return 1; }
  for _ in $(seq 60); do
    rescan 2>&1 | tee -a "$OUT/rescan.log" | tail -1 | tee "$OUT/rescan"   # rescan.log: every step (relink)
    grep -q "ID 0x4f545055" "$OUT/rescan" && return 0; sleep 10; done
  echo "rescan failed for 10 minutes"; return 1; }
# FAIL and PASS lines into checks.txt; a FAIL line on stderr too (not into the callers' filters
# and tees)
fail() { echo "  [FAIL] $PH: $*" >> "$OUT/checks.txt"; echo "  [FAIL] $PH: $*" >&2; }
pass() { echo "  [PASS] $PH: $*" | tee -a "$OUT/checks.txt"; }
# stop [WHY]: the qual ends here (finish: REST back, the summary, exit status 1)
stop() { [ $# -gt 0 ] && fail "$*"; STOPPED=1; echo "the qual stops here"; exit 1; }
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
hexid() { printf '%s\n' "$1" | sed -n 's/.*_\([0-9a-f]\{8\}\)$/\1/p'; }
# check_build BUILD WHAT: the card's BUILD_ID (and CAPS bits 30 and 31, into GENC and WAITWC)
# from Board.info: a PASS line when it is BUILD, else a FAIL line and status 1
check_build() {
  local got=""
  [ -n "$1" ] || { fail "$2: no build id to check the card against (REST_EXPECT, or a" \
                        "directory named ..._<8 hex digits>)"; return 1; }
  read -r got GENC WAITWC <<< "$(run "probe ($2)" $P -c "from opentpu.host.board import Board, XdmaTransport
i = Board(XdmaTransport('/dev/xdma0', dma=False), check=False, lock=False).info()
c, b = i['caps'] or {}, i['build_id']
print('PROBE', 'none' if b is None else '%08x' % b, int(bool(c.get('gen'))), int(bool(c.get('waitw'))))" \
    | sed -n 's/^PROBE //p' | tail -1)"
  [ -n "$got" ] || { fail "$2: the probe of BUILD_ID and CAPS failed"; return 1; }
  [ "$got" = "$1" ] || { fail "$2: the card runs build $got, not $1"; return 1; }
  pass "$2: the card runs build $got"
}
# REST back on the card: its load, then its build
restore() {
  RESTORED=1
  load "$REST" || { fail "the JTAG load of REST ($REST) failed"; return 1; }
  check_build "${REST_EXPECT:-$(hexid "$(cd "$(dirname "$REST")" 2>/dev/null && basename "$(pwd -P)")")}" \
              "REST ($REST)"
}
# the kernel log's last timestamp (s since boot), then its xdma errors after it: a FAIL line each
klog_t() { sudo -n dmesg 2>/dev/null | sed -n 's/^\[ *\([0-9][0-9]*\.[0-9]*\)\].*/\1/p' | tail -1; }
xdma_errors() {
  local k; k=$(sudo -n dmesg 2>&1) || { fail "dmesg: not readable ($(printf %s "$k" | tail -1))"; return; }
  printf '%s\n' "$k" | awk -v t0="$K0" 'match($0, /^\[ *[0-9]+\.[0-9]+\]/) {
      if (substr($0, 2, RLENGTH - 2) + 0 > t0 + 0) print }' \
    | grep -iE "xdma.*(timed out|error|fail)" | cut -c1-200 | while IFS= read -r l; do fail "dmesg: $l"; done
}
selftest() { run selftest timeout 1800 $P -m opentpu.host.selftest \
  | grep -E "\[(PASS|FAIL)\]|config|ALL|stopped|hint" | tee -a "$OUT/checks.txt"; }
diag() {  # $1 label, $2 quick|full
  run "diag $1" timeout 3600 $P -m opentpu.host.diag --mem "$2" --soak 20 > "$OUT/diag-$1.txt"
  grep -E "\[(PASS|FAIL)\].*(RDOT|OUTER|LOG2)" "$OUT/diag-$1.txt"
  sed -n '/^summary/,$p' "$OUT/diag-$1.txt" | tee -a "$OUT/checks.txt"
}
# exact LABEL MODEL WF HF [refs.py card's flags]: a token-exact run, its line into checks.txt
exact() { local label=$1; shift
  run "$label" timeout 1800 $P tools/qual/refs.py card "$OUT/cfg.pkl" "$1" "$2" "$3" "$NTOK" "${@:4}" \
    | grep -E "\] model|Error|Traceback" | tee -a "$OUT/checks.txt"; }
npass() { grep -c '\[PASS\] model' "$OUT/checks.txt"; }
# the references' job (this shell's, still running); refs_wait: up to REFS_WAIT s more for its
# end, then killed (refs.py compute kills its jobs); a non-zero exit, an exception in its log
# or the kill is a FAIL line
refs_running() { [ -n "$REFS" ] && jobs -p | grep -qx "$REFS" && kill -0 "$REFS" 2>/dev/null; }
refs_wait() {
  local t0 w=${REFS_WAIT:-900} rc; t0=$(date +%s)
  while refs_running && [ $(( $(date +%s) - t0 )) -lt "$w" ]; do sleep 2; done
  if refs_running; then
    kill "$REFS"; wait "$REFS"; REFS=""
    fail "refs compute: still running $w s after the last token-exact run: killed ($OUT/refs.log)"
    return
  fi
  wait "$REFS"; rc=$?; REFS=""
  echo "refs exit $rc" >> "$OUT/refs.log"
  { [ $rc = 0 ] && ! grep -qE "$EXC" "$OUT/refs.log"; } || fail "refs compute: exit $rc ($OUT/refs.log)"
}
# the model phases' checkpoints, where the tools look for them (opentpu.llm.model_dir)
models_ok() {
  local m d bad=""
  for m in $(for r in $RUNS $LONG; do echo "${r%%:*}"; done | sort -u); do
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
  local rc=$? nf
  if refs_running; then kill "$REFS"; wait "$REFS"; echo "killed the references' job (pid $REFS)"; fi
  [ $rc != 0 ] && [ -z "$STOPPED" ] && fail "qual.sh exited early (status $rc)"
  if [ $rc != 0 ] && [ "$LOADED" = 1 ] && [ -z "$RESTORED" ] && [ "$REST" != "$BIT" ]; then
    phase "REST back on the card"; restore; fi
  phase ""
  echo "=== phases ($PROFILE)"; column -t -s $'\t' "$OUT/phases.tsv"
  nf=$(grep -c '\[FAIL\]' "$OUT/checks.txt")
  echo "total $(( ($(date +%s) - T00) / 60 )) min; $nf FAIL lines" \
       "($(grep -c '\[PASS\]' "$OUT/checks.txt") PASS); results in $OUT"
  if [ "$nf" = 0 ] && [ $rc = 0 ]; then echo "QUAL DONE $NAME $(date +%T): PASS"; exit 0; fi
  echo "QUAL DONE $NAME $(date +%T): FAIL"; exit 1
}
trap finish EXIT
trap 'exit 143' TERM HUP INT    # killed: finish too (the references' job, REST, the summary)

# (a staged copy of the tree has no .git: its COMMIT file, as the deploy kits write it, if any)
HC=$(git -C "$H" rev-parse --short HEAD 2>/dev/null || head -c 7 "$H/COMMIT" 2>/dev/null)
echo "################ $NAME, $PROFILE profile, host tree ${HC:-unknown} $(date +%T)"
phase "load + selftest"
EXPECT=${EXPECT:-$(hexid "$NAME")}
[ -n "$EXPECT" ] || stop "no build id to expect: set EXPECT ($NAME does not end in _<8 hex digits>)"
if [ "${LOAD:-1}" = 0 ]; then echo "LOAD=0: no JTAG load, the card keeps its bitstream"
else load "$BIT" || stop "the JTAG load of $BIT failed (above; $OUT/jtag.log, $OUT/rescan.log)"
  LOADED=1; fi
K0=$(klog_t); K0=${K0:-0}
check_build "$EXPECT" "the candidate" || stop
selftest
run "refs cfg" $P tools/qual/refs.py cfg "$OUT/cfg.pkl" --name "$NAME" | cut -c1-200
MODELS=1; models_ok || MODELS=0
GEN=${GEN:-$GENC}; WAITW=${WAITW:-$WAITWC}
echo "decode loop on the card (CAPS bit30): $([ "$GEN" = 1 ] && echo yes || echo no)," \
     "WAITW (CAPS bit31): $([ "$WAITW" = 1 ] && echo yes || echo no)"
NRUNS=$(set -- $RUNS; echo $#); NLONG=$(set -- $LONG; echo $#)

if [ $MODELS = 1 ]; then phase "references (background)"
  $P tools/qual/refs.py compute "$OUT/cfg.pkl" --ntok "$NTOK" --runs $RUNS \
    ${LONG:+$LONG:long $LONG:long,pr} > "$OUT/refs.log" 2>&1 &
  REFS=$!
  sleep 3; head -8 "$OUT/refs.log"
fi

if [ $COLD = 1 ]; then phase "diag cold (full march)"; temp; diag cold full; fi

if [ $MODELS = 1 ]; then
phase "prefill + decode counters ($NRUNS)"
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
phase "token-exact after the soak ($NRUNS x per-position + resident${LONG:+, the long prompt x 3})"
grep -E "FAILED" "$OUT/refs.log"
p0=$(npass)
for r in $RUNS; do IFS=: read -r m w h <<< "$r"
  for res in "" --resident; do exact "token-exact $m $w $h$res" "$m" "$w" "$h" $res; done
done
if [ -n "$LONG" ]; then IFS=: read -r m w h <<< "$LONG"
  for res in "" --resident "--resident --prompt-runs"; do
    exact "token-exact long $m $w $h${res:+ $res}" "$m" "$w" "$h" --long $res; done
fi
want=$(( NRUNS * 2 + NLONG * 3 )); got=$(( $(npass) - p0 ))
[ "$got" -eq "$want" ] || fail "$got of $want token-exact runs passed"
if [ "$GEN" = 1 ]; then
phase "decode loop on the card ($NRUNS${LONG:+ + 2 long} token-exact + $(set -- $DP_RUNS; echo $#) x 2 decode_profile)"
p0=$(npass)
for r in $RUNS; do IFS=: read -r m w h <<< "$r"
  exact "card loop $m $w $h" "$m" "$w" "$h" --card-loop
done
if [ -n "$LONG" ]; then IFS=: read -r m w h <<< "$LONG"
  for pr in "" --prompt-runs; do
    exact "card loop long $m $w $h${pr:+ $pr}" "$m" "$w" "$h" --long --card-loop $pr; done
fi
want=$(( NRUNS + NLONG * 2 )); got=$(( $(npass) - p0 ))
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
refs_wait                       # (refs.py card waited for each reference it needed)
fi

phase "DRAM turnarounds + ECC (${TURN:-30} s)"
run turnaround timeout 900 $P tools/qual/turnaround.py --seconds "${TURN:-30}" \
  | grep -E "\[(PASS|FAIL)\]" | tee -a "$OUT/checks.txt"
xdma_errors                     # since the load (before REST's, which brings its own)

phase "final selftest"
if [ "$REST" != "$BIT" ] && [ "${LOAD:-1}" != 0 ]; then restore || stop
else check_build "$EXPECT" "the candidate, at the end" || stop; fi
selftest
