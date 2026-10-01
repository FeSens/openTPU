#!/bin/bash
# offload card session 8 (2026-10-01, production build B, Gen1, no reload; tree 2a0b962, the DMA
# guard): Gemma 4 26B-A4B with its experts streamed (docs/offload.md 11): int8 layers, fp4
# experts (the split pool, G26POOL) and fp4 head, 18 slots a layer by decayed use, 128 tokens
# after wiki.txt's first paragraph, twice. Then one 35B run as session 7's q35e128a: the guard's
# cost against O/s7's. Each run's first 16 tokens and prefill against the ISA simulator's
# reference (q26ref16: reference.sh), the two 26B runs' 128 tokens against each other.
# Results: docs/offload.md 11.5. About 35 min.
# Run: otpu-lock --wait 10800 -- tools/offload/sessions/session8.sh   (log: O/s8/session.log)
set -u
SESSION=s8; source "$(dirname "$0")/env.sh"
exec > >(tee -a $R/session.log) 2>&1
[ -e $O/gemma-4-26B-A4B ] || ln -s ${G26:-$HOME/openTPU/models/gemma-4-26B-A4B} $O/gemma-4-26B-A4B
echo "session8 start $(date +%T) tree $rev mem $(mem) GB"
RUNS="${RUNS:-g26a g26b q35e128a}" bash "$here/card_moe.sh"
echo "session8 end $(date +%T)"
python - $R $O/s7 <<'PY'
import json, os, sys
o, s7 = sys.argv[1:]
r = {n: json.load(open(f"{o}/g26card128{n}.json")) for n in ("a", "b")
     if os.path.exists(f"{o}/g26card128{n}.json")}
print("  26B: 128 tokens the same in both runs:",
      len(r) == 2 and len({tuple(x["tokens"]) for x in r.values()}) == 1)
def polls(x):
    h = x["host_decode_s"]
    return f"polls' reads {1e6 * h['polls_read_s'] / max(h['polls_reads'], 1):.1f} us x {h['polls_reads']}"
for n, x in r.items():
    print(f"  g26{n}: {x['tok_s_wall']} tok/s, {polls(x)}")
if os.path.exists(f"{o}/q35card128ea.json") and os.path.exists(f"{s7}/q35card128ea.json"):
    c, s = (json.load(open(f"{d}/q35card128ea.json")) for d in (o, s7))
    print(f"  35B: tokens as session 7's: {c['tokens'] == s['tokens']}; tok/s {c['tok_s_wall']} "
          f"(s7 {s['tok_s_wall']}); {polls(c)} (s7 {polls(s)})")
PY
date > $R/DONE
