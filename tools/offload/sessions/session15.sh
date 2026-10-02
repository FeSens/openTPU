#!/bin/bash
# offload card session 15 (docs/offload.md 10.11, design A), on the bitstream as loaded:
# - the tag's order on it first (tools/qual/waitw.py --tag-rounds 2000: an expert's 1-4 MiB with
#   the tag in its DMA's last beat while the card waits on it; it stops on a FAIL);
# - then A against each miss's entry on the same bitstream, A B A B: A's parent (OLD: main
#   cb3dce5, every other change the same) and this tree (the answer and the tags), q35e128s and
#   g26s with their timelines and DMA calls, against refs-d29bfe9 (the math is unchanged: the
#   same tokens and prefill sha; refs-f725c2b confirms it on the ISA simulator). Predicted: the
#   35B -0.6 to -0.8 s per 128 tokens (+2.4-3.3%), the 26B -0.4 to -0.9 s (+1-2.5%); the 35B's
#   64-byte calls 31,346 -> about 10,000.
#   OLD  A's parent's tree (default O/../tree-cb3dce5: git archive of main cb3dce5)
# About 22 min. It stops at a mismatch, a timeout or an error.
# Run: otpu-lock --wait 10800 -- tools/offload/sessions/session15.sh   (log: O/s15/session.log)
set -u
SESSION=${SESSION:-s15}; RF=${RF:-${O:-$HOME/otpu-build/offload/card2}/refs-d29bfe9}
source "$(dirname "$0")/env.sh"             # (its RF defaults to O: set before)
OLD=${OLD:-$(dirname "$O")/tree-cb3dce5}
exec > >(tee -a $R/session.log) 2>&1
echo "session15 start $(date +%T) tree $rev old $(cat $OLD/COMMIT) mem $(mem) GB"
bad() { sleep 1; grep -c -E "\[FAIL\]|exit 124|Traceback|STOP" $R/session.log; }
b0=$(bad)
stop() { [ $(bad) -gt $b0 ] && { echo "STOP $(date +%T): a mismatch, a timeout or an error"; return 0; }; return 1; }
echo "--- waitw: the tag in the data's last beat"
timeout 900 python tools/qual/waitw.py --rounds 50 --tag-rounds 2000; echo "waitw exit $?"
run() {   # its name, runs, tree
  echo "--- $1: $2 ($(cat $3/COMMIT 2>/dev/null || echo $3))"
  env T=$3 RF=$RF R=$R/$1 RUNS="$2" bash "$3/tools/offload/sessions/card_moe.sh"
}
stop || for x in "L1 q35e128s $OLD" "A1 q35e128s $T" "L2 q35e128s $OLD" "A2 q35e128s $T" \
                 "gL1 g26s $OLD" "gA1 g26s $T" "gL2 g26s $OLD" "gA2 g26s $T"; do
  run $x
  stop && break
done
echo "session15 end $(date +%T)"
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
