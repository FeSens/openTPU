#!/bin/bash
# offload card session 13 (production build, no reload), d29bfe9's programs (session 9's
# references, refs-d29bfe9) in one tree: session 12's host code + docs/offload.md 10.8's serving
# (preadv_iov, a request's first miss a fifth first, the victims' clears after its experts, the
# poll loop spinning), against the same tree's --legacy-serve (as session 12 served), A B A B:
#   q35e128sl / q35e128s: the 35B at 128 tokens (session 12's B2: 3.76 tok/s, windows 21.5 s,
#     DMA 16.6; predicted about 4.0 now);
#   g26sl / g26s: the 26B at 128 tokens (session 11: 2.65 / 2.69, windows 32.0 s, DMA 27.2;
#     predicted about 2.8).
# Each with its timeline and BoardDram's DMA calls (--hint-trace: <run>.trace.json and
# <run>.trace.calls.json). The other big files dropped before each run (DROPOTHER's default).
# About 18 min. It stops at a mismatch, a timeout or an error (a FAIL, a selftest's).
# Run: otpu-lock --wait 10800 -- tools/offload/sessions/session13.sh   (log: O/s13/session.log)
set -u
SESSION=${SESSION:-s13}; RF=${RF:-${O:-$HOME/otpu-build/offload/card2}/refs-d29bfe9}
source "$(dirname "$0")/env.sh"             # (its RF defaults to O: set before)
exec > >(tee -a $R/session.log) 2>&1
echo "session13 start $(date +%T) tree $rev mem $(mem) GB"
bad() { sleep 1; grep -c -E "\[FAIL\]|exit 124|Traceback|STOP" $R/session.log; }
b0=$(bad)
stop() { [ $(bad) -gt $b0 ] && { echo "STOP $(date +%T): a mismatch, a timeout or an error"; return 0; }; return 1; }
run() {   # its name, runs
  local out=$1 runs=$2
  echo "--- $out: $runs"
  env T=$T RF=$RF R=$R/$out RUNS="$runs" bash "$here/card_moe.sh"
}
go=1
for k in 1 2; do
  run L$k q35e128sl; stop && { go=0; break; }
  run N$k q35e128s; stop && { go=0; break; }
done
if [ $go = 1 ]; then
  run gL1 g26sl
  stop || run gN1 g26s
fi
echo "session13 end $(date +%T)"
python - $R <<'PY'
import glob, json, sys
r = sys.argv[1]
for f in sorted(glob.glob(f"{r}/*/*.json")):
    if ".trace." in f:
        continue
    c = json.load(open(f))
    h = c.get("host_decode_s") or {}
    rd = h.get("reads") or {}
    print(f"  {f[len(r) + 1:]}: legacy {c.get('legacy_serve')} tok/s {c['tok_s_wall']} "
          f"(device {c['tok_s_device']}), poll {h.get('poll')}, dma {h.get('dma_s')}, stage "
          f"{h.get('stage')}, flush {h.get('flush')}, reads cached {rd.get('cached')} disk "
          f"{rd.get('disk')}, polls {h.get('polls_reads')}, match {c.get('match')}")
PY
date > $R/DONE
