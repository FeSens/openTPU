#!/bin/bash
# offload card session 9 (production build B, Gen1, no reload): the experts' MMs paired (main
# d29bfe9, moe-pair) against unpaired, in one session. Tree B is this one (d29bfe9 + this
# script; its ISA-simulator references in RFB, made by reference.sh at d29bfe9), tree A main
# 7d879e6, before moe-pair (TA; its references in RFA: sessions 7 and 8's, and LFM2.5-8B's
# 160 tokens made at 7d879e6). A B A B, each with:
#   q35e128a  Qwen3.5-35B-A3B, the table on the host, 128 tokens (session 7: 3.84, 3.87 tok/s)
#   g26a      gemma-4-26B-A4B, int8 / fp4 experts / fp4 head, 128 tokens (session 8: 2.77, 2.77)
#   8b160     LFM2.5-8B-A1B, 160 tokens (session 3: 10.64), when its pool is staged
# each run against its own tree's reference (tokens, prefill sha). The co-simulation projects
# +3-4% for the 35B and the 26B (link-bound at Gen1) and +35-40% for LFM2.5-8B. The first live
# run of the repo's card_moe.sh; results: docs/offload.md 10.5. About 35 min (AB=B: tree B
# alone, about 18). It stops at a mismatch, a timeout or an error (a FAIL, a selftest's).
# WAITPOOL=s: LFM2.5-8B's pool is streamed in while this holds the lock (opentpu's disk keeps it
# only for the session): the sparse file made here, then pack_pool.py send | recv from a host
# with the whole checkpoint, ending with POOL.done; up to s seconds, a pool not whole removed.
# DROPPOOL=1: the pool removed at the end.
# Run: otpu-lock --wait 10800 -- tools/offload/sessions/session9.sh   (log: O/s9/session.log)
set -u
SESSION=s9; source "$(dirname "$0")/env.sh"
exec > >(tee -a $R/session.log) 2>&1
TB=$T; TA=${TA:-$O/../tree-7d879e6}; RFA=${RFA:-$O/refs-7d879e6} RFB=${RFB:-$O/refs-d29bfe9}
echo "session9 start $(date +%T) tree B $rev, A $(cat $TA/COMMIT 2>/dev/null) mem $(mem) GB"
P=$O/pool-lfm25-8b-fp4.split.bin
if [ -n "${WAITPOOL:-}" ]; then
  [ -f $P ] || python tools/offload/pack_pool.py $O/LFM2.5-8B-A1B $P init
  t0=$SECONDS; until [ -f $P.done ] || [ $((SECONDS - t0)) -gt $WAITPOOL ]; do sleep 10; done
  if python -c "import numpy, sys; sys.exit(not numpy.fromfile('$P.packed', numpy.uint8).all())"; then
    echo "  LFM2.5-8B's pool whole after $((SECONDS - t0)) s"
  else
    echo "  LFM2.5-8B's pool not whole: removed, 8b160 skipped"; rm -f $P $P.packed $P.format $P.done
  fi
fi
bad() { sleep 1; grep -c -E "\[FAIL\]|exit 124|Traceback|STOP" $R/session.log; }
b0=$(bad)
for k in 1 2; do
  for t in ${AB:-A B}; do
    if [ $t = A ]; then tt=$TA rf=$RFA; else tt=$TB rf=$RFB; fi
    echo "--- tree $t ($tt), round $k"
    T=$tt RF=$rf R=$R/$t$k RUNS="${RUNS:-q35e128a g26a 8b160}" bash "$here/card_moe.sh"
    [ $(bad) -gt $b0 ] && { echo "STOP $(date +%T): a mismatch, a timeout or an error"; break 2; }
  done
done
echo "session9 end $(date +%T)"
[ -n "${DROPPOOL:-}" ] && rm -f $P $P.packed $P.format $P.done && echo "  LFM2.5-8B's pool removed"
python - $R <<'PY'
import glob, json, os, sys
r = sys.argv[1]
for name in ("q35card128ea", "g26card128a", "card160"):
    v = {t: [json.load(open(f)) for f in sorted(glob.glob(f"{r}/{t}[12]/{name}.json"))]
         for t in ("A", "B")}
    if not v["B"]:
        continue
    s = {t: [x["tok_s_wall"] for x in v[t]] for t in v}
    same = len({tuple(x["tokens"]) for x in v["B"]}) == 1
    gain = (f", B/A {sum(s['B']) / len(s['B']) / (sum(s['A']) / len(s['A'])) - 1:+.1%}"
            if s["A"] else "")
    print(f"  {name}: tok/s A {s['A']} B {s['B']}{gain}; B's runs the same tokens: {same}")
PY
date > $R/DONE
