#!/bin/bash
# offload card session 10 (2026-10-01, production build 72256074, no reload): where the 35B's
# host staging goes (docs/offload.md 10.6; results there). Session 9's 35B lost 48 ms a token to
# staging on main 7d879e6 and 117-210 ms on d29bfe9, at the same misses and DMA. Its tree: main
# d29bfe9 with offload-relmem's host code (its programs d29bfe9's: program_sha.py --cfg 35B
# 2fa1b39f, 26B 48151736, so RF = session 9's refs-d29bfe9). Rounds (ROUNDS, comma-separated):
#   s10:  q35e128t q35e128r q35e128rw g26a g26r (DROPOTHER off; RF then fell to O's, 7d879e6's:
#         its FAILs were the references', the prefill sha d29bfe9's; the 2nd round never ran)
#   s10b: q35e128t q35e128r g26a g26r, the other checkpoints out of the page cache (DROPOTHER)
#   then: q35e128t as session 9's q35e128a with each request's window (--hint-trace),
#   q35e128r + --release-weights, q35e128rw + --willneed, g26r g26a + --release-weights. Since,
#   moe_card releases and queues by default: q35e128k is the old q35e128t (--keep-weights
#   --no-willneed), and card_moe.sh drops the other files unless DROPOTHER="".
# It stops at a mismatch, a timeout or an error (a FAIL, a selftest's).
# Run: SESSION=s10b ROUNDS="..." otpu-lock --wait 10800 -- tools/offload/sessions/session10.sh
set -u
SESSION=${SESSION:-s10}; RF=${RF:-${O:-$HOME/otpu-build/offload/card2}/refs-d29bfe9}
source "$(dirname "$0")/env.sh"             # (its RF defaults to O: set before)
exec > >(tee -a $R/session.log) 2>&1
echo "session10 start $(date +%T) tree $rev mem $(mem) GB"
bad() { sleep 1; grep -c -E "\[FAIL\]|exit 124|Traceback|STOP" $R/session.log; }
b0=$(bad)
k=0
IFS=, read -r -a rounds <<< "${ROUNDS:-q35e128t q35e128r q35e128rw g26a g26r,q35e128r q35e128t g26r g26a}"
for runs in "${rounds[@]}"; do
  k=$((k + 1))
  echo "--- round $k: $runs"
  RF=$RF R=$R/r$k RUNS="$runs" bash "$here/card_moe.sh"
  [ $(bad) -gt $b0 ] && { echo "STOP $(date +%T): a mismatch, a timeout or an error"; break; }
done
echo "session10 end $(date +%T)"
python - $R <<'PY'
import glob, json, sys
r = sys.argv[1]
for f in sorted(glob.glob(f"{r}/r[0-9]/*.json")):
    if f.endswith(".trace.json"):
        continue
    c = json.load(open(f))
    h, m = c.get("host_decode_s") or {}, c.get("host_mem") or {}
    rd = h.get("reads") or {}
    at = lambda k: (m.get(k) or {})                                   # noqa: E731
    pw = c.get("pool_warm") or {}
    print(f"  {f[len(r) + 1:]}: tok/s {c['tok_s_wall']}, misses/token "
          f"{c['misses_per_token_decode']}, stage {h.get('stage')} s (wait {h.get('stage_wait')}), "
          f"poll {h.get('poll')}, dma {h.get('dma_s')}; reads cached {rd.get('cached')} disk "
          f"{rd.get('disk')}; rss {at('at_decode').get('rss')} -> {at('at_end').get('rss')} "
          f"(file {at('at_decode').get('rss_file')}, children {at('at_decode').get('children_rss')}),"
          f" available {at('at_decode').get('available')} -> {at('at_end').get('available')}; "
          f"pool resident {(pw.get('at_decode') or {}).get('resident_gb')} -> "
          f"{(pw.get('at_end') or {}).get('resident_gb')} GB")
PY
date > $R/DONE
