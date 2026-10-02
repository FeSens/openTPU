#!/bin/bash
# offload card session 17, on the bitstream as loaded (EXPECT: production fmvf 542fc43a), in one
# lock, this tree throughout:
# (b) LFM2.5-8B-A1B token-exact (design A owes it, docs/offload.md 10.11): 8b16, and 8b160 with
#     LFM160=1. Its pool is in /dev/shm (LFMPOOL, copied before the lock: the host's disk is at
#     its floor) behind links in O, all deleted right after (b). Skipped if LFMPOOL is missing.
# (a) the decode's re-baseline on the fused build: q35e128s twice (the A runs of (c)), g26s twice.
# (c) the 35B's capped hints (docs/offload.md 12.7): q35e128s (A), q35e128hn (B: pre, top 4, n 1)
#     and with TOP8=1 q35e128hn8 (C: top 8) in a palindrome, A B C C B A (A B B A without C), so
#     a linear drift cancels for each. Predicted for B: about +5% tok/s over A (contention
#     0.13-0.17 s a GB: +5 to +6%; at 0.34: +1.6%); top 8 about +7%. Rule: hints on by default
#     for the 35B only if B gains >= 2% over A with every run bit for bit.
# Each model's first run after the others' pools leave the page cache runs slower (gemma4's
# pfhint2: 35B 13.16 against 12.46 s, demand serve 6.83 -> 5.93 s), so a discarded warm-up run
# of the base config (W, gW: checked, not compared) comes before the 35B's and the 26B's; the
# summary gives each pair's difference. LFM2's runs are a token-exact check, not compared.
# RF holds the ISA simulator's references: q35ref16 and q26ref16 (refs-s16: the programs are
# unchanged since), q35ref16h (the hint programs with the caps, refs-hintcap), ref16 and ref160
# (LFM2's, refs-lfm-3ef23f6). Without q35ref16h, HREF=q35ref16 links it to q35ref16 (hints
# change no logit; logged), else the B runs are skipped.
# About 35 min. It stops at a mismatch, a timeout or an error.
# Run: otpu-lock --wait 10800 -- tools/offload/sessions/session17.sh   (log: O/s17/session.log)
set -u
SESSION=${SESSION:-s17}; RF=${RF:-${O:-$HOME/otpu-build/offload/card2}/refs-s17}
source "$(dirname "$0")/env.sh"             # (its RF defaults to O: set before)
LFMPOOL=${LFMPOOL:-/dev/shm/offload-lfm/pool-lfm25-8b-fp4.split.bin}
EXPECT=${EXPECT:-542fc43a}
exec > >(tee -a $R/session.log) 2>&1
echo "session17 start $(date +%T) tree $rev RF $RF mem $(mem) GB"
bad() { sleep 1; grep -c -E "\[FAIL\]|exit 124|Traceback|STOP" $R/session.log; }
b0=$(bad)
stop() { [ $(bad) -gt $b0 ] && { echo "STOP $(date +%T): a mismatch, a timeout or an error"; return 0; }; return 1; }
run() {   # its name, runs
  echo "--- $1: $2 ($rev, RF $RF)"
  env RF=$RF R=$R/$1 RUNS="$2" bash "$T/tools/offload/sessions/card_moe.sh"
}
st=$(timeout 600 python -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "\[(PASS|FAIL)\]|ALL|build" | tail -14
echo "$st" | grep -q "build $EXPECT" || echo "STOP: the card's build is not $EXPECT"
if [ ! -f $RF/q35ref16h.json ] && [ -n "${HREF:-}" ]; then
  ln -s $HREF.json $RF/q35ref16h.json && echo "q35ref16h: $HREF's (no ISA run of the hint programs)"
fi
P=$O/pool-lfm25-8b-fp4.split.bin
lfm_gone() { rm -f $P $P.packed $P.format; rm -rf "$(dirname $LFMPOOL)"; echo "  LFM2 pool deleted $(date +%T); /dev/shm: $(df -h /dev/shm | tail -1)"; }
if ! stop && [ -f $LFMPOOL ] && [ ! -e $P ]; then
  ln -s $LFMPOOL $P && cp $LFMPOOL.packed $LFMPOOL.format $O/
  run b8 "8b16${LFM160:+ 8b160}"
  lfm_gone
else
  echo "(b) skipped: $([ -f $LFMPOOL ] || echo "no $LFMPOOL")$([ -e $P ] && echo " $P exists")"
fi
stop || for x in "W q35e128s" "A1 q35e128s" "B1 q35e128hn" ${TOP8:+"C1 q35e128hn8"} \
                 ${TOP8:+"C2 q35e128hn8"} "B2 q35e128hn" "A2 q35e128s" "gW g26s" "gA1 g26s" \
                 "gA2 g26s"; do
  run $x
  stop && break
done
echo "session17 end $(date +%T)"
python - $R <<'PY'
import glob, json, sys
import numpy as np
r = sys.argv[1]
by = {}
for f in sorted(glob.glob(f"{r}/*/*.json")):
    if ".trace." in f:
        continue
    c = json.load(open(f))
    h, d = c.get("host_decode_s") or {}, c.get("device_counters") or {}
    hi = c.get("hints") or {}
    run = f[len(r) + 1:].split("/")[0]
    by.setdefault(run.rstrip("0123456789"), []).append(c)
    print(f"  {f[len(r) + 1:]}: tok/s {c['tok_s_wall']} (device {c['tok_s_device']}), generate "
          f"{c.get('generate_s')} s, misses {c.get('misses_per_token_decode')} a decode token, MB "
          f"{(c.get('bytes_per_token_decode') or 0) / 1e6:.0f}, serve {h.get('serve')}, hints "
          f"{ {k: hi.get(k) for k in ('served', 'prefetched', 'promoted', 'dropped', 'withdrawn')} }, "
          f"match {c.get('match')} sha {c.get('prefill_logits_sha')}")
a = by.get("A")                             # (the warm-up runs W, gW not compared)
for k in ("B", "C"):
    if a and by.get(k):
        ta, tb = (np.mean([c["tok_s_wall"] for c in x]) for x in (a, by[k]))
        pairs = ", ".join(f"{y['tok_s_wall'] / x['tok_s_wall'] - 1:+.1%}" for x, y in zip(a, by[k]))
        print(f"  {k} against A: tok/s {tb:.3f} against {ta:.3f} ({tb / ta - 1:+.1%}); by pair "
              f"({k}1 / A1, {k}2 / A2): {pairs}")
PY
date > $R/DONE
