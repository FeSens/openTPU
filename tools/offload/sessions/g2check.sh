#!/bin/bash
# offload: the Gen2 x8 check (docs/offload.md 10.9) on production g2fix 0885d436, no reload:
# session 13 tree (b40fc6f, d29bfe9 programs, refs-d29bfe9), q35e128s x2 then g26s, with traces.
# About 6 min (2026-10-02 00:31-00:37; results: docs/offload.md 10.9). It stops at a mismatch,
# a timeout or an error.
# Run: otpu-lock --wait 3600 -- tools/offload/sessions/g2check.sh   (log: O/g2c/session.log)
set -u
SESSION=${SESSION:-g2c}; RF=${RF:-${O:-$HOME/otpu-build/offload/card2}/refs-d29bfe9}
source "$(dirname "$0")/env.sh"
exec > >(tee -a $R/session.log) 2>&1
echo "g2check start $(date +%T) tree $rev mem $(mem) GB"
bad() { sleep 1; grep -c -E "\[FAIL\]|exit 124|Traceback|STOP" $R/session.log; }
b0=$(bad)
stop() { [ $(bad) -gt $b0 ] && { echo "STOP $(date +%T): a mismatch, a timeout or an error"; return 0; }; return 1; }
run() { echo "--- $1: $2"; env T=$T RF=$RF R=$R/$1 RUNS="$2" bash "$here/card_moe.sh"; }
for x in "N1 q35e128s" "N2 q35e128s" "gN1 g26s"; do
  run $x
  stop && break
done
echo "g2check end $(date +%T)"
python - $R <<'PY'
import glob, json, sys
r = sys.argv[1]
for f in sorted(glob.glob(f"{r}/*/*.json")):
    if ".trace." in f:
        continue
    c = json.load(open(f))
    h, d = c.get("host_decode_s") or {}, c.get("device_counters") or {}
    g = {k: round(d.get(k, 0) / 1e9, 3) for k in ("RUNNING", "MXU_STARVE", "DMA_BUSY", "DRAM_WAIT")}
    print(f"  {f[len(r) + 1:]}: tok/s {c['tok_s_wall']} (device {c['tok_s_device']}), poll "
          f"{h.get('poll')}, dma {h.get('dma_s')} ({h.get('dma_gbs')} GB/s), polls "
          f"{h.get('polls_reads')}, G {g}, match {c.get('match')} sha {c.get('prefill_logits_sha')}")
PY
date > $R/DONE
