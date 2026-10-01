#!/bin/bash
# Layer-major prefill on the card (production build B, Gen1, no reload; docs/offload.md 13):
# gemma-4-26B-A4B as session 8's g26a (int8 layers, fp4 experts and head, 18 slots a layer by
# decayed use, wiki.txt's first paragraph), 16 tokens, its prompt token by token (g26t16), then
# layer by layer in runs of 1 and 2 rows (g26lm1, g26lm2); then Qwen3.5-35B-A3B the same way
# as q35e (q35t16, q35lm1, q35lm2; RUNS picks the runs). Each model's three runs' prefill
# logits and tokens must be the same (layer-major is bit-exact with token by token on the ISA
# simulator), and each against the ISA simulator's reference (reference.sh) and HF's.
# Per-layer slots (not pooled yet): the layer-major runs move about as many experts as token
# by token; their prefill_s is the card's compute and the runs' overheads. About 25 min a
# model.
# Run: otpu-lock --wait 10800 -- tools/offload/sessions/layer_major.sh   (log: O/lm/session.log)
set -u
SESSION=lm; source "$(dirname "$0")/env.sh"
exec > >(tee -a $R/session.log) 2>&1
[ -e $O/gemma-4-26B-A4B ] || ln -s ${G26:-$HOME/openTPU/models/gemma-4-26B-A4B} $O/gemma-4-26B-A4B
echo "layer_major start $(date +%T) tree $rev mem $(mem) GB"
RUNS="${RUNS:-g26t16 g26lm1 g26lm2 q35t16 q35lm1 q35lm2}" bash "$here/card_moe.sh"
echo "layer_major end $(date +%T)"
python - $R <<'PY'
import json, os, sys
o = sys.argv[1]
for m, t in (("g26", ""), ("q35", "t")):
    r = {n: json.load(open(f"{o}/{m}card16{n}.json")) for n in (t, "lm1", "lm2")
         if os.path.exists(f"{o}/{m}card16{n}.json")}
    if not r:
        continue
    same = len({(x["prefill_logits_sha"], tuple(x["tokens"])) for x in r.values()}) == 1
    print(f"  [{'PASS' if len(r) == 3 and same else 'FAIL'}] {m}: the runs' prefill logits and "
          f"tokens the same: {same} ({len(r)} runs)")
    for n, x in r.items():
        print(f"  {m}{n or 't16'}: prefill {x['prefill_s']} s, sha {x['prefill_logits_sha']}, "
              f"layer-major requests {x.get('prefill_requests')} misses "
              f"{x.get('prefill_misses')}, decode {x['tok_s_wall']} tok/s")
PY
date > $R/DONE
