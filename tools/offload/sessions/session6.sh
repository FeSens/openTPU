#!/bin/bash
# offload card session 6 (2026-10-01, production build B, Gen1, no reload; tree d0835eb: main
# ee7fa64 + ExpertServer's drop and timeline): session 5's follow-up on the router hints, which
# lost 5%. The full 35B pool, its page cache warm:
#   q35e    table on the host, no hints (session 5's noise)
#   q35eht  + hints (512 KiB parts), with the decode's timeline
#   q35ehd  + a request withdrawing its layer's unnamed hints
#   q35ehs  + 128 KiB parts
#   q35ehds + both
#   q35c    table on the card, no hints
# each token-exact against q35ref16. Results: docs/offload.md 12.6. About 12 min.
# Run: otpu-lock --wait 10800 -- tools/offload/sessions/session6.sh   (log: O/s6/session.log)
set -u
SESSION=s6; source "$(dirname "$0")/env.sh"
exec > >(tee -a $R/session.log) 2>&1
echo "session6 start $(date +%T) tree $rev mem $(mem) GB"
RUNS="${RUNS:-q35e q35eht q35ehd q35ehs q35ehds q35c}" bash "$here/card_moe.sh"
echo "session6 end $(date +%T)"
date > $R/DONE
