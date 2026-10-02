#!/bin/bash
# offload card session 14 (production: Gen2 x8, deploy_g2fix_0885d436; no reload), session 13's
# programs (d29bfe9's, refs-d29bfe9) in session 13's tree with PollPacer (docs/offload.md 10.10):
#   q35e128s, q35e128sp: the 35B with the default serving on Gen2 (Gen1, session 13: 4.01 / 4.04
#     tok/s; predicted ~5.0, 4.9-5.1: 10.9), then with the decode's polls paced;
#   g26s, g26s50, g26sp, g26s, g26sp: the 26B on Gen2 (Gen1 2.78; predicted ~3.6, 3.56-3.63) and
#     the poll's A/B: spinning, a 50 us sleep, paced (MXU_STARVE rose with the spin, 10.8).
# Each with its timeline and DMA calls (--hint-trace). The other big files dropped before each
# run (DROPOTHER's default). About 19 min. It stops at a mismatch, a timeout or an error.
# Run: otpu-lock --wait 10800 -- tools/offload/sessions/session14.sh   (log: O/s14/session.log)
set -u
SESSION=${SESSION:-s14}; RF=${RF:-${O:-$HOME/otpu-build/offload/card2}/refs-d29bfe9}
source "$(dirname "$0")/env.sh"             # (its RF defaults to O: set before)
exec > >(tee -a $R/session.log) 2>&1
echo "session14 start $(date +%T) tree $rev mem $(mem) GB"
bad() { sleep 1; grep -c -E "\[FAIL\]|exit 124|Traceback|STOP" $R/session.log; }
b0=$(bad)
stop() { [ $(bad) -gt $b0 ] && { echo "STOP $(date +%T): a mismatch, a timeout or an error"; return 0; }; return 1; }
run() {   # its name, runs
  echo "--- $1: $2"
  env T=$T RF=$RF R=$R/$1 RUNS="$2" bash "$here/card_moe.sh"
}
for x in "N1 q35e128s" "P1 q35e128sp" "gS1 g26s" "gF1 g26s50" "gP1 g26sp" "gS2 g26s" "gP2 g26sp"; do
  run $x
  stop && break
done
echo "session14 end $(date +%T)"
python - $R <<'PY'
import glob, json, sys
r = sys.argv[1]
for f in sorted(glob.glob(f"{r}/*/*.json")):
    if ".trace." in f:
        continue
    c = json.load(open(f))
    h, d = c.get("host_decode_s") or {}, c.get("device_counters") or {}
    g = {k: round(d.get(k, 0) / 1e9, 3) for k in ("RUNNING", "MXU_STARVE", "DMA_BUSY", "DRAM_WAIT")}
    print(f"  {f[len(r) + 1:]}: poll_idle {c.get('poll_idle')} tok/s {c['tok_s_wall']} (device "
          f"{c['tok_s_device']}), poll {h.get('poll')}, dma {h.get('dma_s')} ({h.get('dma_gbs')} "
          f"GB/s), polls {h.get('polls_reads')}, pacer {c.get('pacer')}, G cycles {g}, "
          f"match {c.get('match')}")
PY
date > $R/DONE
