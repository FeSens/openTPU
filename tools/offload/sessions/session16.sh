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
# Then the 35B's layer-major prefill (pooled, R = 2, lmtime's 134-token prompt PF_HF, its sha
# PF_SHA and tokens), main against this tree: onecall's prefill serve rose 6.56 -> 7.10 s.
#   EXPECT  the card's build (default e4db91c9: production pa); another one stops it
# About 26 min. It stops at a mismatch, a timeout or an error.
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
EXPECT=${EXPECT:-e4db91c9}
PF_HF=${PF_HF:-$HOME/otpu-build/moepf/lmtime/q35w-hf.json} PF_SHA=${PF_SHA:-95426ebacc3b40a9}
PF_TOK="[8160, 579, 264, 7047, 1817, 421]"
pf() {    # its name, tree: the 35B's prompt layer-major, R = 2, pooled, with its timeline
  echo "--- $1: 35B layer-major prefill ($(cat $2/COMMIT))"
  t0=$SECONDS; cat $O/pool-q35-fp4.split.bin > /dev/null; echo "  pool read in $((SECONDS - t0)) s"
  (cd $2 && PYTHONPATH=$2 timeout 1800 python tools/offload/moe_card.py $O/Qwen3.5-35B-A3B \
    --check $PF_HF -n 16 --pool $O/pool-q35-fp4.split.bin --cfg $R/cfg-dev.pkl --experts 0 --card \
    --policy lfu --embed-table host --hints off --layer-major 2 --prefill-trace $R/$1.trace.json \
    --out $R/$1.json > $R/$1.log 2>&1)
  echo "  exit $? $(date +%T)"; grep -E "Error|Traceback" $R/$1.log | tail -3
  python - $R/$1.json $PF_SHA "$PF_TOK" <<'PY' || echo "STOP: $1 differs"
import json, sys
d = json.load(open(sys.argv[1]))
tok = json.loads(sys.argv[3])
ok = d["prefill_logits_sha"] == sys.argv[2] and d["tokens"][:len(tok)] == tok
print(f"  [{'PASS' if ok else 'FAIL'}] prefill sha {d['prefill_logits_sha']} vs {sys.argv[2]}, tokens "
      f"{'same' if d['tokens'][:len(tok)] == tok else 'DIFFER'}; prefill {d['prefill_s']} s, "
      f"requests {d.get('prefill_requests')} misses {d.get('prefill_misses')}")
pt = d["prefill_time"]
print("  prefill_time " + json.dumps({k: pt.get(k) for k in ("wall_s", "runs", "device_s", "between_s", "requests", "serve_s")}))
sys.exit(0 if ok else 1)
PY
}
st=$(timeout 600 python -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "\[(PASS|FAIL)\]|ALL|build" | tail -14
echo "$st" | grep -q "build $EXPECT" || echo "STOP: the card's build is not $EXPECT"
timeout 300 python tools/qual/refs.py cfg $R/cfg-dev.pkl > /dev/null 2>&1; echo "cfg exit $?"
stop || for x in "M1 q35e128s $OLD" "H1 q35e128s $T" "M2 q35e128s $OLD" "H2 q35e128s $T"; do
  run $x
  stop && break
done
stop || for x in "pfM $OLD" "pfH $T"; do
  pf $x
  stop && break
done
stop || for x in "gM1 g26s $OLD" "gH1 g26s $T" "gM2 g26s $OLD" "gH2 g26s $T"; do
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
