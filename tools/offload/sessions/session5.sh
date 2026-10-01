#!/bin/bash
# offload card session 5 (2026-10-01, production build B, Gen1, no reload; tree 7a048e5,
# offload-hint): Qwen3.5-35B-A3B, 16 tokens, slots by decayed use filling the DRAM (--experts 0):
#   q35c   the embedding table on the card, no hints (34-35 slots a layer)
#   q35e   the table on the host (42 slots a layer), no hints
#   q35eh  the table on the host, hints on
# each token-exact against the ISA simulator's q35ref16. The pool must hold all 10240 experts
# (the card host's checkpoint has none: pack_pool.py's send / recv). Results: docs/offload.md
# 10.4, 12.5. About 35 min (3 image builds).
# Run: otpu-lock --wait 10800 -- tools/offload/sessions/session5.sh   (log: O/s5/session.log)
set -u
SESSION=s5; source "$(dirname "$0")/env.sh"
exec > >(tee -a $R/session.log) 2>&1
echo "session5 start $(date +%T) tree $rev mem $(mem) GB"
RUNS="${RUNS:-q35c q35e q35eh}" bash "$here/card_moe.sh"
echo "session5 end $(date +%T)"
date > $R/DONE
