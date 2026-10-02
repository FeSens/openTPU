#!/bin/bash
# offload card sessions 11 + 12 (production build, no reload), d29bfe9's programs (session 9's
# references, refs-d29bfe9) in two trees: A = tree-s11 (session 10's host code + Layout.pitch +
# DROPOTHER's guard), B = this one (A + PoolFile's mapped reads, offload-mglru).
#   11: A's g26r twice (the 26B's slots whole RUN blocks apart: every expert read in place;
#       session 10: 2.39-2.42 tok/s, direct 5618 of 11276), the other files dropped (DROPOTHER)
#   12: q35e128r A B A B on a host whose other checkpoints a process mapped and touched (HOG:
#       the Qwen3.5 0.8B / 2B / 4B, session 10's 13.6 GB; DROPOTHER=""): the pool read through
#       read() (A) against touched through its map (B; docs/offload.md 10.7). Session 10:
#       3.05 tok/s with them in the page cache, 3.79 without.
# About 20 min (2026-10-01 21:30-21:50, results: docs/offload.md 10.6 and 10.7). It stops at a
# mismatch, a timeout or an error (a FAIL, a selftest's).
# Run: otpu-lock --wait 10800 -- tools/offload/sessions/session12.sh   (log: O/s12/session.log)
set -u
SESSION=${SESSION:-s12}; RF=${RF:-${O:-$HOME/otpu-build/offload/card2}/refs-d29bfe9}
source "$(dirname "$0")/env.sh"             # (its RF defaults to O: set before)
exec > >(tee -a $R/session.log) 2>&1
TB=$T; TA=${TA:-$O/../tree-s11}
DIRS="$HOME/openTPU/models:$O:$O/../g26"
HOGS="$HOME/openTPU/models/Qwen3.5-0.8B:$HOME/openTPU/models/Qwen3.5-2B:$HOME/openTPU/models/Qwen3.5-4B"
echo "session12 start $(date +%T) tree B $rev, A $(cat $TA/COMMIT 2>/dev/null) mem $(mem) GB"
bad() { sleep 1; grep -c -E "\[FAIL\]|exit 124|Traceback|STOP" $R/session.log; }
b0=$(bad)
stop() { [ $(bad) -gt $b0 ] && { echo "STOP $(date +%T): a mismatch, a timeout or an error"; return 0; }; return 1; }
run() {   # tree, its name, runs, then card_moe.sh's environment
  local tt=$1 out=$2 runs=$3; shift 3
  echo "--- $out: tree $tt, $runs, $*"
  env "$@" T=$tt RF=$RF R=$R/$out RUNS="$runs" bash "$here/card_moe.sh"
}
go=1
run $TA s11 "g26r g26r" DROPOTHER=$DIRS HOG=
stop && go=0
if [ $go = 1 ]; then
  for k in 1 2; do
    for t in A B; do
      if [ $t = A ]; then tt=$TA; else tt=$TB; fi
      run $tt $t$k q35e128r DROPOTHER= HOG=$HOGS
      stop && { go=0; break 2; }
    done
  done
fi
echo "session12 end $(date +%T)"
python - $R <<'PY'
import glob, json, sys
r = sys.argv[1]
for f in sorted(glob.glob(f"{r}/*/*.json")):
    if f.endswith(".trace.json"):
        continue
    c = json.load(open(f))
    h, m = c.get("host_decode_s") or {}, c.get("host_mem") or {}
    rd = h.get("reads") or {}
    pw = c.get("pool_warm") or {}
    print(f"  {f[len(r) + 1:]}: tok/s {c['tok_s_wall']}, stage {h.get('stage')} s, poll "
          f"{h.get('poll')}, dma {h.get('dma_s')}, direct {h.get('direct')}; reads cached "
          f"{rd.get('cached')} disk {rd.get('disk')}; pool resident "
          f"{(pw.get('at_decode') or {}).get('resident_gb')} -> "
          f"{(pw.get('at_end') or {}).get('resident_gb')} GB; cached "
          f"{(m.get('at_decode') or {}).get('cached')} GB")
PY
date > $R/DONE
