#!/bin/bash
# offload card session 7 (2026-10-01, production build B, Gen1, no reload; tree 91e17ec,
# hint-tune): the embedding table on the host against on the card at 128 tokens, each twice and
# interleaved (16 tokens vary by ~6% on the host), the pool read into the page cache before each
# run, the polls' beat read. Each run's first 16 tokens and prefill against q35ref16, the four
# runs' 128 tokens against each other. Results: docs/offload.md 10.4. About 12 min.
# Run: otpu-lock --wait 10800 -- tools/offload/sessions/session7.sh   (log: O/s7/session.log)
set -u
SESSION=s7; source "$(dirname "$0")/env.sh"
exec > >(tee -a $R/session.log) 2>&1
echo "session7 start $(date +%T) tree $rev mem $(mem) GB"
RUNS="${RUNS:-q35e128a q35c128a q35e128b q35c128b}" bash "$here/card_moe.sh"
echo "session7 end $(date +%T)"
python - $R <<'PY'
import json, sys
r = {n: json.load(open(f"{sys.argv[1]}/q35card128{n}.json")) for n in ("ea", "ca", "eb", "cb")}
print("  128 tokens the same in all four runs:", len({tuple(x["tokens"]) for x in r.values()}) == 1)
PY
date > $R/DONE
