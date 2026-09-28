#!/usr/bin/env bash
# The fmax tournament, continuously (docs/tourney.md, "Running continuously"): one round per
# component in turn, and a whole-design round (FOREVER_WHOLE) after every WHOLE_EVERY component
# rounds. It never finishes on its own:
#   touch /tmp/otpu-tourney-stop     stop before the next round (the running round completes)
#   touch /tmp/otpu-tourney-pause    wait before the next round while the file exists
#   /tmp/otpu-tourney-hosts          the Vivado hosts and their caps (remote.py), read per job
# Each round's champion first merges BASE (origin/main, fetched per round) when it has moved.
# K=2 agent slots per round; K_<comp>=n overrides one component (e.g. K_otpu_mxu=3).
set -u
cd "$(git rev-parse --show-toplevel)"
COMPS=${FOREVER_COMPS:-"otpu_dma otpu_seq otpu_tmem otpu_coll otpu_xunit otpu_mxu otpu_axi_dram otpu_vpu otpu_quant otpu_actram"}
WHOLE=${FOREVER_WHOLE:-"otpu_impl"}
EVERY=${WHOLE_EVERY:-4}
K=${K:-2}
BASE=${BASE:-origin/main}
TARGET=${TARGET_MHZ:-125.49}
STOP=/tmp/otpu-tourney-stop PAUSE=/tmp/otpu-tourney-pause

round() {   # one round of component $1
  local c=$1 k var
  [[ -e $STOP ]] && { echo "[forever] $STOP: stopping"; exit 0; }
  while [[ -e $PAUSE ]]; do sleep 60; done
  var="K_$c"; k=${!var:-$K}
  git fetch -q origin 2>/dev/null || echo "[forever] git fetch failed; using the last $BASE"
  echo "[forever] $(date '+%F %T') round: $c (K=$k)"
  python3 -m tools.tourney.orchestrator --objective fmax --target-mhz "$TARGET" --comp "$c" \
    --rounds 1 --slots "$k" --eval vivado-remote --base "$BASE" \
    || echo "[forever] $(date '+%F %T') $c: round failed (exit $?); going on"
}

n=0
while :; do
  for c in $COMPS; do
    round "$c"
    n=$((n + 1))
    if (( n % EVERY == 0 )); then
      for w in $WHOLE; do round "$w"; done
    fi
  done
done
