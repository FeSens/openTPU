#!/bin/bash
# offload card session 16 (docs/offload.md 10.13, the request's head), on the bitstream as loaded:
# main (OLD: design A, the answer first) against this tree (the answer after the first expert's
# first part, the other misses' willneed after it, touches deferred, a reused mincore vector,
# the DMA lock reentrant), A B A B, q35e128s and g26s with their timelines and DMA calls. Host
# only: the same programs, so both match RF bit for bit (main's 35B programs changed since
# f725c2b: refs-8100ffb; the 26B's did not: refs-f725c2b). Predicted (serve_emu): the head -0.5
# to -0.6 s per 128 tokens; the 35B 5.08-5.16 -> 5.20-5.27 tok/s, the 26B 3.58 -> 3.63-3.65.
#   OLD     main's tree (default O/../tree-main: git archive of main with COMMIT)
#   RF      the references (default O/refs-s16: q35ref16.json from refs-8100ffb, q26ref16.json
#           from refs-f725c2b)
# About 20 min. It stops at a mismatch, a timeout or an error.
# Run: otpu-lock --wait 10800 -- tools/offload/sessions/session16.sh   (log: O/s16/session.log)
set -u
SESSION=${SESSION:-s16}; RF=${RF:-${O:-$HOME/otpu-build/offload/card2}/refs-s16}
source "$(dirname "$0")/env.sh"             # (its RF defaults to O: set before)
OLD=${OLD:-$(dirname "$O")/tree-main}
exec > >(tee -a $R/session.log) 2>&1
echo "session16 start $(date +%T) tree $rev old $(cat $OLD/COMMIT) mem $(mem) GB"
bad() { sleep 1; grep -c -E "\[FAIL\]|exit 124|Traceback|STOP" $R/session.log; }
b0=$(bad)
stop() { [ $(bad) -gt $b0 ] && { echo "STOP $(date +%T): a mismatch, a timeout or an error"; return 0; }; return 1; }
run() {   # its name, runs, tree
  echo "--- $1: $2 ($(cat $3/COMMIT 2>/dev/null || echo $3), RF $RF)"
  env T=$3 RF=$RF R=$R/$1 RUNS="$2" bash "$3/tools/offload/sessions/card_moe.sh"
}
stop || for x in "M1 q35e128s $OLD" "H1 q35e128s $T" "M2 q35e128s $OLD" "H2 q35e128s $T" \
                 "gM1 g26s $OLD" "gH1 g26s $T" "gM2 g26s $OLD" "gH2 g26s $T"; do
  run $x
  stop && break
done
echo "session16 end $(date +%T)"
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
