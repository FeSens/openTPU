#!/bin/bash
# Layer-major prefill on the card (production build B, Gen1, no reload; docs/offload.md 13):
# 16 tokens of gemma-4-26B-A4B as session 8's g26a (int8 layers, fp4 experts and head, 18 slots
# a layer by decayed use, wiki.txt's first paragraph) and of Qwen3.5-35B-A3B as q35e (the
# table on the host, no hints), each prompt token by token (g26t16, q35t16) or layer by layer
# in runs of 1 or 2 rows (g26lm1/2, q35lm1/2: each layer's own slots; g26lmp2, q35lmp2: R = 2
# with the slots pooled). RUNS picks the runs (default: the 35B's three with per-layer slots,
# then the pooled ones last, so that a pooled failure costs only its own runs).
# A model's runs must give the same prefill logits and tokens (layer-major is bit-exact with
# token by token on the ISA simulator, pooled or not); each is also checked against the ISA
# simulator's reference (reference.sh) and HF's; and against the runs of an earlier session in
# O/lm (678b976: g26t16 / lm1 / lm2, per-layer slots). prefill_s: the prompt's time.
# Run: otpu-lock --wait 10800 -- tools/offload/sessions/layer_major.sh   (log: O/lm2/session.log)
set -u
SESSION=${SESSION:-lm2}; source "$(dirname "$0")/env.sh"
exec > >(tee -a $R/session.log) 2>&1
[ -e $O/gemma-4-26B-A4B ] || ln -s ${G26:-$HOME/openTPU/models/gemma-4-26B-A4B} $O/gemma-4-26B-A4B
echo "layer_major start $(date +%T) tree $rev mem $(mem) GB"
check() {      # the model's runs so far (and O/lm's) agree, and this run wrote its result
  python - $R $1 <<'PY'
import glob, json, sys
o, run = sys.argv[1:]
m = run[:3]
fs = glob.glob(f"{o}/{m}card16*.json") + glob.glob(f"{o}/../lm/{m}card16*.json")
n = len(glob.glob(f"{o}/{m}card16*.json"))
same = len({(json.load(open(f))["prefill_logits_sha"], tuple(json.load(open(f))["tokens"]))
            for f in fs}) == 1
sys.exit(0 if same and n else 1)
PY
}
for run in ${RUNS:-q35t16 q35lm1 q35lm2 q35lmp2 g26lmp2}; do   # one at a time: stop at the
  n0=$(ls $R/${run:0:3}card16*.json 2>/dev/null | wc -l)        # first failed or differing run
  RUNS=$run bash "$here/card_moe.sh"
  n1=$(ls $R/${run:0:3}card16*.json 2>/dev/null | wc -l)
  if [ "$n1" -le "$n0" ] || ! check $run; then echo "STOP after $run: no result, or it differs"; break; fi
done
echo "layer_major end $(date +%T)"
python - $R <<'PY'
import glob, json, os, sys
o = sys.argv[1]
for m in ("g26", "q35"):
    r = {(d + " " if d else "") + (os.path.basename(f)[len(m) + 6:-5] or "t16"): json.load(open(f))
         for d in ("", "lm") for f in sorted(glob.glob(f"{o}/{'../' + d + '/' if d else ''}"
                                                       f"{m}card16*.json"))
         if d != "lm" or os.path.realpath(f"{o}/../lm") != os.path.realpath(o)}
    if not r:
        continue
    same = len({(x["prefill_logits_sha"], tuple(x["tokens"])) for x in r.values()}) == 1
    print(f"  [{'PASS' if same else 'FAIL'}] {m}: {len(r)} runs, prefill logits and tokens the "
          f"same: {same}")
    for n, x in r.items():
        print(f"  {m} {n}: prefill {x['prefill_s']} s, sha {x['prefill_logits_sha']}, pooled "
              f"{x.get('pooled')}, layer-major requests {x.get('prefill_requests')} misses "
              f"{x.get('prefill_misses')}, decode {x['tok_s_wall']} tok/s")
PY
date > $R/DONE
