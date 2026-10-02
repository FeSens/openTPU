#!/bin/bash
# Session 18 (offload, docs/offload.md 13.12): the halt-aware hold re-measured with its fix
# (offload-hold: run_clock during a run's serve; pfv2 had holds 0), and the 35B decode hints'
# confirm against the default parts (ra since pfv2). One otpu-lock; per pair one discarded
# warm-up run, then A B B A, each pair's difference reported.
# (1) the 35B's layer-major prefill (R = 2, pooled, lmtime's 134-token prompt, 16 tokens) with
#     the layer-ahead predictor and whole-expert parts (--layer-ahead hint --ahead-part 4096):
#     A = --idle-parts ra, B = --idle-parts v2. Pass: lmtime's sha 95426ebacc3b40a9 and tokens.
#     B must show holds > 0.
# (2) the 35B decode, 128 tokens: A = q35e128s (no hints; ref q35ref16), B = the hints (pre,
#     top 4, n 1, whole-expert parts, --hint-drop; ref q35ref16h) with the default parts;
#     hint traces kept. Hints become the default only at >= 2% with both pairs positive.
# Host only against main's programs, so session 17's references hold (RF: card2/refs-s17). A
# build other than EXPECT, a failed selftest, a failed or timed-out run or a mismatch stops it;
# no retries. About 20 min.
# Run: setsid -f otpu-lock --wait 3600 -- bash tools/offload/sessions/session18.sh   (no pipe)
set -u
SESSION=s18
RF=${RF:-$HOME/otpu-build/offload/card2/refs-s17}
source "$(dirname "$0")/env.sh"
exec > >(tee -a $R/session.log) 2>&1
EXPECT=${EXPECT:-542fc43a}
DROPOTHER=$HOME/openTPU/models:$O:$O/../g26
LM=${LM:-$HOME/otpu-build/moepf/lmtime/q35w-hf.json}
P="--embed-table host --hints off --layer-major 2 --layer-ahead hint --ahead-part 4096"
HN="--embed-table host --hints on --hint-drop --hint-n 1 --hint-top 4 --hint-part 1632"
Q=Qwen3.5-35B-A3B; QP=pool-q35-fp4.split.bin
declare -A RUN=(         # checkpoint, pool, HF's, tokens, reference, moe_card's other options
  [pW]="$Q $QP $LM 16 sha:95426ebacc3b40a9 $P --idle-parts ra"
  [pA1]="$Q $QP $LM 16 sha:95426ebacc3b40a9 $P --idle-parts ra"
  [pB1]="$Q $QP $LM 16 sha:95426ebacc3b40a9 $P --idle-parts v2"
  [pB2]="$Q $QP $LM 16 sha:95426ebacc3b40a9 $P --idle-parts v2"
  [pA2]="$Q $QP $LM 16 sha:95426ebacc3b40a9 $P --idle-parts ra"
  [dW]="$Q $QP $O/q35-hf.json 128 $RF/q35ref16.json --embed-table host --hints off"
  [dA1]="$Q $QP $O/q35-hf.json 128 $RF/q35ref16.json --embed-table host --hints off --hint-trace TRACE"
  [dB1]="$Q $QP $O/q35-hf.json 128 $RF/q35ref16h.json $HN --hint-trace TRACE"
  [dB2]="$Q $QP $O/q35-hf.json 128 $RF/q35ref16h.json $HN --hint-trace TRACE"
  [dA2]="$Q $QP $O/q35-hf.json 128 $RF/q35ref16.json --embed-table host --hints off --hint-trace TRACE")
Q35TOK="[8160, 579, 264, 7047, 1817, 421]"        # (lmtime's first tokens)
echo "session18 start $(date +%T) tree $rev mem $(mem) GB; $(uptime)"
st=$(timeout 600 python -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "\[(PASS|FAIL)\]|ALL" | tail -14
echo "$st" | grep -q "build $EXPECT" || { echo "STOP: the card's build is not $EXPECT"; exit 2; }
echo "$st" | grep -q "ALL PASS" || { echo "STOP: selftest"; exit 2; }
timeout 300 python tools/qual/refs.py cfg $R/cfg-dev.pkl > /dev/null 2>&1; echo "cfg exit $?"
for run in pW pA1 pB1 pB2 pA2 dW dA1 dB1 dB2 dA2; do
  read -r md pool hf n ref extra <<< "${RUN[$run]}"
  extra=${extra//TRACE/$R/$run.hint.json}
  ptr=""; [ $n = 16 ] && ptr="--prefill-trace $R/$run.trace.json"
  echo "=== $run $(date +%T) ($extra)"
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
  t0=$SECONDS; cat $O/$pool > /dev/null; echo "  pool read in $((SECONDS - t0)) s, mem $(mem) GB"
  timeout 900 python tools/offload/moe_card.py $O/$md --check $hf -n $n --pool $O/$pool \
    --cfg $R/cfg-dev.pkl --experts 0 --card --policy lfu $extra $ptr \
    --out $R/$run.json > $R/$run.log 2>&1
  rc=$?; echo "  exit $rc $(date +%T)"; grep -E "Error|Traceback" $R/$run.log | tail -3
  [ $rc -eq 0 ] && [ -f $R/$run.json ] || { echo "STOP: $run"; break; }
  python - $R/$run.json $ref "$Q35TOK" <<'PY' || { echo "STOP: $run differs"; break; }
import json, sys
d = json.load(open(sys.argv[1]))
if sys.argv[2].startswith("sha:"):
    sha, tok = sys.argv[2][4:], json.loads(sys.argv[3])
else:
    r = json.load(open(sys.argv[2]))
    sha, tok = r["prefill_logits_sha"], r["tokens"]
ok = d["prefill_logits_sha"] == sha and d["tokens"][:len(tok)] == tok
print(f"  [{'PASS' if ok else 'FAIL'}] prefill sha {d['prefill_logits_sha']} vs {sha}, tokens "
      f"{'same' if d['tokens'][:len(tok)] == tok else 'DIFFER'}; prefill {d['prefill_s']} s, "
      f"decode {d['tok_s_wall']} tok/s (device {d.get('tok_s_device')}), misses/token "
      f"{d.get('misses_per_token_decode')}")
print("  idle_parts " + json.dumps(d.get("idle_parts")) + " hints " + json.dumps(d.get("hints")))
pt = d.get("prefill_time")
if pt:
    print("  prefill_time " + json.dumps({k: pt.get(k) for k in ("wall_s", "runs", "device_s",
                                                                "between_s", "requests", "serve_s")}))
sys.exit(0 if ok else 1)
PY
done
echo "=== final selftest $(date +%T)"
timeout 600 python -m opentpu.host.selftest 2>&1 | grep -E "\[(PASS|FAIL)\]|ALL" | tail -14
echo "session18 end $(date +%T)"
python - $R <<'PY'
import json, os, sys
r = sys.argv[1]
def get(run):
    f = f"{r}/{run}.json"
    return json.load(open(f)) if os.path.exists(f) else None
for name, runs, key in (("35B prefill, ra -> v2", "p", "prefill_s"),
                        ("35B decode, hints top 4 + ra", "d", "tok_s_wall")):
    a1, b1, b2, a2 = (get(runs + x) for x in ("A1", "B1", "B2", "A2"))
    if None in (a1, b1, b2, a2):
        print(f"  {name}: incomplete"); continue
    v = lambda c: c[key] if key != "prefill_s" else c["prefill_time"]["wall_s"]   # noqa: E731
    d1, d2 = v(b1) - v(a1), v(b2) - v(a2)
    ma, mb = (v(a1) + v(a2)) / 2, (v(b1) + v(b2)) / 2
    rel = f" ({mb / ma - 1:+.2%})" if key == "tok_s_wall" else ""
    print(f"  {name} ({key}): A {v(a1):.3f} {v(a2):.3f}, B {v(b1):.3f} {v(b2):.3f}; pairs B-A "
          f"{d1:+.3f} {d2:+.3f}; means B-A {mb - ma:+.3f}{rel}")
    for x, c in (("B1", b1), ("B2", b2)):
        print(f"    {x} idle_parts {json.dumps(c.get('idle_parts'))}")
PY
date > $R/DONE
