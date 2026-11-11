#!/bin/bash
# Session pfB (docs/offload.md 13.11, 13.13, 13.14, 13.15): expert-major MoE in the layer-major prefill
# on the card, A/B against today's default. Run 2026-10-12 on the card host.
# Tree: GitHub main ee8f118d plus this script (its opentpu/ the same as production's host/, 23b68089),
# a git archive at ~/otpu-build/moe-ab/tree-ee8f118d (T). Card: production deploy_mt4_91a9c2d3 as
# loaded (EXPECT 91a9c2d3, register map 4); no bitstream change.
# Arms (every one: R = 2, pooled slots, --idle-parts v2 --ahead-part 4096, slots filling the DRAM, lfu):
#   A = today's default (no layer-ahead), B = --expert-major, C = --layer-ahead hint (C=0: none).
# Per model a discarded warm-up run W (B: a failure of B stops the session before a measured run),
# then A B C C B A (A B B A without C); each pair's difference in the summary:
#   35B: Qwen3.5-35B-A3B, lmtime's 134-token prompt (moepf/lmtime/q35w-hf.json), 16 tokens;
#   26B: gemma-4-26B-A4B-it (the -it checkpoint and its pool g26it/, since 2026-10-09), q26-hf.json's
#        124 rows, 16 tokens.
# Pass: every run bit for bit (its prefill logits' sha and its 16 tokens) with its model's W. The
# 2026-10-02 keys (35B 95426ebacc3b40a9, the base 26B's q26ref16) predate fix-math (2026-10-08) and the
# 26B's -it checkpoint, so they are printed for information only. A build other than EXPECT, a
# failed selftest, a device configuration other than production's, a failed or timed-out run or a
# mismatch stops it; no retries.
# The images go to a cache of this session's own (IMG, removed after each model: none hits
# production's ~/otpu-build/qcache/img, whose key holds chat's KV capacity), so each model builds
# once (W). The programs compile in each run's process (no disk program cache: as every session).
# About 25 min under the lock.
# Run (no pipe): setsid nohup ~/.local/bin/otpu-lock --wait 600 -- bash $T/tools/offload/sessions/pfB.sh
#   > ~/otpu-build/moe-ab/launch.log 2>&1 < /dev/null &
set -u
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*' | grep -vE '^OTPU_LOCK_(HELD|PID|WAIT)$'); do unset $v; done
MA=${MA:-$HOME/otpu-build/moe-ab}
T=${T:-$MA/tree-ee8f118d}
R=${R:-$MA/pfB}
IMG=${IMG:-$MA/img}
C=${C:-1}
export OTPU_IMG_CACHE=$IMG OTPU_IMG_CACHE_GB=8 OTPU_IMG_CACHE_FREE_GB=6
source $T/tools/offload/sessions/env.sh       # (O: card2; OTPU_QUIET claimed, removed at exit)
exec > >(tee -a $R/session.log) 2>&1
EXPECT=${EXPECT:-91a9c2d3}
PRODCFG=${PRODCFG:-$HOME/otpu-build/deploy_mt4_91a9c2d3/qual/cfg.pkl}
G26IT=$HOME/otpu-build/offload/g26it
DROPOTHER=$HOME/openTPU/models:$O:$G26IT:$HOME/otpu-build/qcache/img
Q35="Qwen3.5-35B-A3B pool-q35-fp4.split.bin $HOME/otpu-build/moepf/lmtime/q35w-hf.json"
G26="gemma-4-26B-A4B-it ../g26it/pool-g26it-fp4.split.bin $O/q26-hf.json"
QO="--embed-table host --hints off"
GO="--wformat int8 --formats experts=fp4 --head-format fp4"
ALL="--layer-major 2 --idle-parts v2 --ahead-part 4096"
declare -A ARM=([A]="" [B]="--expert-major" [C]="--layer-ahead hint")
# the 2026-10-02 card runs' (pfv2 pA1, before fix-math): information only
OLD35='{"sha": "95426ebacc3b40a9", "tokens": [8160, 579, 264, 7047, 1817, 421, 11177, 310, 279, 11488, 2468, 25, 271, 16, 13, 220]}'
fail=0
echo "pfB start $(date +%T) tree $rev T $T mem $(mem) GB, /home $(df -BG --output=avail /home | tail -1 | tr -d ' ') free; $(uptime)"
st=$(timeout 600 python -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "$st" | grep -q "build $EXPECT" || { echo "STOP: the card's build is not $EXPECT"; exit 2; }
echo "$st" | grep -q "register map 4" || { echo "STOP: not register map 4"; exit 2; }
echo "$st" | grep -q "ALL PASS" || { echo "STOP: selftest"; exit 2; }
timeout 300 python tools/qual/refs.py cfg $R/cfg-dev.pkl > /dev/null 2>&1; echo "cfg exit $?"
python - $R/cfg-dev.pkl $PRODCFG <<'PY' || { echo "STOP: the device configuration"; exit 2; }
import dataclasses, pickle, sys
a, b = (pickle.load(open(f, "rb")) for f in sys.argv[1:])
d = {f.name: (getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
     if getattr(a, f.name) != getattr(b, f.name)}
print("  cfg: production's (qual/cfg.pkl)" if not d else f"  cfg DIFFERS from production's: {d}")
sys.exit(1 if d else 0)
PY
run() {   # model (q|g), run name, arm, timeout
  local m=$1 run=$2 arm=$3 to=$4 md pool hf extra
  if [ $m = q ]; then read -r md pool hf <<< "$Q35"; extra="$QO"; else read -r md pool hf <<< "$G26"; extra="$GO"; fi
  extra="$extra $ALL ${ARM[$arm]}"
  echo "=== $run ($arm) $(date +%T): $extra"
  python - $O/$md $O/$pool $DROPOTHER <<'PY'
import os, sys                      # every other file of 100 MB or more out of the page cache
keep = {os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])}  # (card_moe.sh's DROPOTHER)
ours = tuple(os.path.realpath(os.path.expanduser(d)) + os.sep for d in ("~/openTPU", "~/otpu-build"))
gb = n = 0
seen = set()
for d in sys.argv[3].split(":"):
    for root, _, files in os.walk(os.path.expanduser(d)):
        for f in files:
            r = os.path.realpath(os.path.join(root, f))
            if r in seen or not r.startswith(ours) or not os.path.isfile(r) \
                    or os.path.getsize(r) < 100 << 20 or r in keep or os.path.dirname(r) in keep:
                continue
            seen.add(r)
            fd = os.open(r, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            gb, n = gb + os.path.getsize(r) / 1e9, n + 1
print(f"  dropped from the page cache: {n} other files of {gb:.1f} GB")
PY
  local t0=$SECONDS; cat $O/$pool > /dev/null; echo "  pool read in $((SECONDS - t0)) s, mem $(mem) GB"
  timeout $to python tools/offload/moe_card.py $O/$md --check $hf -n 16 --pool $O/$pool \
    --cfg $R/cfg-dev.pkl --experts 0 --card --policy lfu $extra --prefill-trace $R/$run.trace.json \
    --out $R/$run.json > $R/$run.log 2>&1
  local rc=$?; echo "  exit $rc $(date +%T)"; grep -E "Error|Traceback|timed out" $R/$run.log | tail -3
  [ $rc -eq 0 ] && [ -f $R/$run.json ] || { echo "STOP: $run (exit $rc)"; return 1; }
  python - $R/$run.json $R/${m}W.json "$OLD35" $m <<'PY' || { echo "STOP: $run differs from ${m}W"; return 1; }
import json, sys
d, w = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ok = d["prefill_logits_sha"] == w["prefill_logits_sha"] and d["tokens"] == w["tokens"]
pt = d["prefill_time"]
print(f"  [{'PASS' if ok else 'FAIL'}] prefill sha {d['prefill_logits_sha']} tokens "
      f"{'same as' if d['tokens'] == w['tokens'] else 'DIFFER from'} W's ({w['prefill_logits_sha']}); "
      f"prefill wall {pt['wall_s']} s (device {pt['device_s']}, halted {pt['halted_s']}, between "
      f"{pt['between_s']}, serve {pt['serve_s']}), requests {d.get('prefill_requests')} misses "
      f"{d.get('prefill_misses')}; load {d['load_s']} s; decode {d['tok_s_wall']} tok/s")
print(f"  need {json.dumps(pt.get('need'))} ahead {json.dumps(pt.get('ahead'))} idle_parts "
      f"{json.dumps(d.get('idle_parts'))}")
if sys.argv[1] == sys.argv[2]:      # (W: the 2026-10-02 runs', information only)
    if sys.argv[4] == "q":
        o = json.loads(sys.argv[3])
        print(f"  (info) vs 2026-10-02 (pfv2, before fix-math): sha {'same' if o['sha'] == d['prefill_logits_sha'] else 'differs'}"
              f" ({o['sha']}), tokens {'same' if o['tokens'] == d['tokens'] else 'DIFFER'}")
    else:
        print(f"  (info) vs the base 26B's HF tokens (q26-hf.json): first diff {d['first_diff']}")
    print(f"  tokens {d['tokens']}")
sys.exit(0 if ok else 1)
PY
}
order="W A1 B1 C1 C2 B2 A2"; [ "$C" = 1 ] || order="W A1 B1 B2 A2"
for m in q g; do
  for x in $order; do
    arm=${x:0:1}; [ $x = W ] && arm=B
    to=600; [ $x = W ] && to=1500
    run $m $m$x $arm $to || { fail=1; break 2; }
  done
  echo "  image cache: $(du -sh $IMG 2>/dev/null | cut -f1); removed"; rm -rf $IMG
done
echo "=== final selftest $(date +%T)"
timeout 600 python -m opentpu.host.selftest 2>&1 | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
rm -rf $IMG
echo "pfB end $(date +%T) fail $fail"
python - $R <<'PY'
import json, os, sys
r = sys.argv[1]
def get(run):
    f = f"{r}/{run}.json"
    return json.load(open(f)) if os.path.exists(f) else None
for name, m in (("35B prefill", "q"), ("26B prefill", "g")):
    a1, a2 = get(m + "A1"), get(m + "A2")
    for x in ("B", "C"):
        x1, x2 = get(m + x + "1"), get(m + x + "2")
        if None in (a1, a2, x1, x2):
            print(f"  {name} {x}: incomplete"); continue
        for key in ("wall_s", "device_s", "halted_s", "between_s", "serve_s"):
            v = lambda c: c["prefill_time"][key] or 0.0     # noqa: E731
            d1, d2 = v(x1) - v(a1), v(x2) - v(a2)
            ma, mx = (v(a1) + v(a2)) / 2, (v(x1) + v(x2)) / 2
            print(f"  {name} {key}: A {v(a1):.3f} {v(a2):.3f}, {x} {v(x1):.3f} {v(x2):.3f}; pairs "
                  f"{x}-A {d1:+.3f} {d2:+.3f}; mean {x}-A {mx - ma:+.3f}")
        for y, c in ((x + "1", x1), (x + "2", x2)):
            pt = c["prefill_time"]
            print(f"    {y} runs {pt['runs']} requests {c.get('prefill_requests')} misses "
                  f"{c.get('prefill_misses')} need {json.dumps(pt.get('need'))} idle_parts "
                  f"{json.dumps(c.get('idle_parts'))}")
PY
date > $R/DONE
echo "PFB EXIT $fail $(date +%T)"
