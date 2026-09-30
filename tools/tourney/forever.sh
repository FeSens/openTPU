#!/usr/bin/env bash
# The fmax tournament, continuously (docs/tourney.md, "Running continuously"): one round per
# component in turn, and after every WHOLE_EVERY component rounds one whole-design round, the
# FOREVER_WHOLE components in turn (otpu_full: RTL across modules; otpu_impl: Vivado directives
# and placement). It never finishes on its own:
#   touch /tmp/otpu-tourney-stop     stop before the next round (the running round completes)
#   touch /tmp/otpu-tourney-pause    wait before the next round while the file exists
#   /tmp/otpu-tourney-hosts          the Vivado hosts and their caps (remote.py), read per job
#   /tmp/otpu-tourney-comps          the components of the next pass (whitespace separated; a
#                                    component listed twice runs twice per pass), read per pass
#   /tmp/otpu-tourney-whole          the whole-design components to take turns (in place of
#                                    FOREVER_WHOLE), read before each whole-design round
# Each round's champion first merges BASE (origin/main, fetched per round) when it has moved.
# K=2 agent slots per round; K_<comp>=n overrides one component (e.g. K_otpu_mxu=3).
# A second loop beside it runs with its own OBJECTIVE (e.g. unit: the per-unit OOC tournament,
# `make tourney-units`), control files (FOREVER_STOP, FOREVER_PAUSE, FOREVER_COMPS_FILE; the
# hosts file is remote.py's OTPU_HOSTS_FILE) and WHOLE_EVERY=0 (no whole-design rounds).
set -u
cd "$(git rev-parse --show-toplevel)"
# the units on the measured worst path families first (the DRAM adapter's command picker, TMEM's
# bank arbiter, MXU control and drain, the sequencer, the DMA), then the rest
COMPS=${FOREVER_COMPS:-"otpu_native_dram otpu_tmem otpu_mxu otpu_seq otpu_dma otpu_xunit otpu_vpu otpu_quant otpu_coll otpu_actram"}
COMPS_FILE=${FOREVER_COMPS_FILE:-/tmp/otpu-tourney-comps}
WHOLE=(${FOREVER_WHOLE:-otpu_full otpu_impl})
WHOLE_FILE=${FOREVER_WHOLE_FILE:-/tmp/otpu-tourney-whole}
EVERY=${WHOLE_EVERY:-3}
K=${K:-2}
BASE=${BASE:-origin/main}
TARGET=${TARGET_MHZ:-133.33}
OBJECTIVE=${OBJECTIVE:-fmax}
STOP=${FOREVER_STOP:-/tmp/otpu-tourney-stop} PAUSE=${FOREVER_PAUSE:-/tmp/otpu-tourney-pause}
# the orchestrator checks the same files before it starts
export OTPU_TOURNEY_STOP=$STOP OTPU_TOURNEY_PAUSE=$PAUSE

round() {   # one round of component $1
  local c=$1 k var
  [[ -e $STOP ]] && { echo "[forever] $STOP: stopping"; exit 0; }
  [[ -f tools/tourney/components/$c.yaml ]] || { echo "[forever] no component $c: skipped"; return 0; }
  while [[ -e $PAUSE ]]; do sleep 60; done
  var="K_$c"; k=${!var:-$K}
  git fetch -q origin 2>/dev/null || echo "[forever] git fetch failed; using the last $BASE"
  echo "[forever] $(date '+%F %T') round: $c (K=$k)"
  python3 -m tools.tourney.orchestrator --objective "$OBJECTIVE" --target-mhz "$TARGET" --comp "$c" \
    --rounds 1 --slots "$k" --eval vivado-remote --base "$BASE" \
    || echo "[forever] $(date '+%F %T') $c: round failed (exit $?); going on"
}

whole() {   # the whole-design components: the whole file when it names some, else WHOLE
  local f=""
  [[ -f $WHOLE_FILE ]] && f=$(tr -s '[:space:]' ' ' < "$WHOLE_FILE")
  f=${f# }; f=${f% }
  echo "${f:-${WHOLE[*]}}"
}

comps() {   # the next pass: the comps file when it names components, else COMPS
  local f=""
  [[ -f $COMPS_FILE ]] && f=$(tr -s '[:space:]' ' ' < "$COMPS_FILE")
  f=${f# }; f=${f% }
  echo "${f:-$COMPS}"
}

n=0 w=0
while :; do
  pass=$(comps)
  echo "[forever] $(date '+%F %T') pass: $pass"
  for c in $pass; do
    round "$c"
    n=$((n + 1))
    if (( EVERY > 0 && n % EVERY == 0 )); then
      read -r -a ws <<< "$(whole)"
      round "${ws[w % ${#ws[@]}]}"
      w=$((w + 1))
    fi
  done
done
