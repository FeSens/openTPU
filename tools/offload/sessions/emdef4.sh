#!/bin/bash
# Session emdef4 (docs/offload.md 13.16): the expert-major threshold's swap check, short. emdef3 was token-exact and
# the 35B passed; the 26B's first turn (21 rows, layer-major in both arms) took +0.20 / +0.14 s against P, with the
# same requests (the host served them 3-8% slower in N; on the ISA simulator N's and P's layer-major calls run the
# same programs, requests and slots, and their host time a request is the same). Gemma 4's EM_ROWS is now 20: its
# 21-row call expert-major (X - P over four pairs -0.086 s TTFT, -0.18 to -0.56 s device). Qwen3.5's stays 44.
# Trees: the candidate deploy's host, ~/otpu-build/deploy_em_91a9c2d3/host (T: the commit with EM_ROWS 44 / 20, or it
# stops); production's host (PH: deploy_mt4_91a9c2d3/host, main 23b68089), read-only, no bytecode written. Card:
# production deploy_mt4_91a9c2d3 as loaded (EXPECT 91a9c2d3, register map 4). One otpu-lock:
# 1. selftest (T); the device configuration must be production's qual/cfg.pkl;
# 2. the production qual's chatmoe step from T (RUNS="qwen35-35b gemma4-26b"), token-exact against the ISA references
#    (the 26B's two turns now expert-major: 21 and 27 rows);
# 3. otpu-chat by name from T, one greedy answer each, its reply against production's mt4 chat-<model>.txt;
# 4. per model emdef2's first three turns (about 24, 31 and 49 new tokens on the 35B: calls of 20, 27 and 45 rows,
#    each with a 4-row call after it; the 26B's 21, 27 and 46), arms P (T's chatmoe.py on production's opentpu) and
#    N (T), every run traced: a discarded warm-up (N), then ORDER (default P1 N1 N2 P2; for more pairs, balanced:
#    ORDER="P1 N1 N2 P2 N3 P3 P4 N4", about 8 min more). Every run's tokens must equal the warm-up's, and its first
#    two turns the ISA references';
# 5. selftest; the summary: per model and turn each run's TTFT and device s, N - P per pair (Nk - Pk) and of the
#    means, the calls, and each arm's layer-major calls' host ms a miss.
# The swap's rule : N no worse than P at every turn, token-exact.
# A build other than EXPECT, a failed selftest, another configuration, a FAIL, a failed or timed-out run or a mismatch
# stops it (then the final selftest). About 20 min.
# Run (no pipe): setsid nohup ~/.local/bin/otpu-lock --wait 600 -- bash $T/tools/offload/sessions/emdef4.sh
#   > ~/otpu-build/moe-ab/emdef4.launch.log 2>&1 < /dev/null &
set -u
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*' | grep -vE '^OTPU_LOCK_(HELD|PID|WAIT)$'); do unset $v; done
export OTPU_IMG_CACHE_GB=30 OTPU_IMG_CACHE_FREE_GB=6      # (the card host's, from its ~/.bashrc)
MA=${MA:-$HOME/otpu-build/moe-ab}
T=${T:-$HOME/otpu-build/deploy_em_91a9c2d3/host}
PH=${PH:-$HOME/otpu-build/deploy_mt4_91a9c2d3/host}
R=${R:-$MA/emdef4}
source $T/tools/offload/sessions/env.sh       # (O: card2; OTPU_QUIET claimed, removed at exit)
exec > >(tee -a $R/session.log) 2>&1
EXPECT=${EXPECT:-91a9c2d3}
ORDER=${ORDER:-P1 N1 N2 P2}                    # (after the warm-up W)
PD=$HOME/otpu-build/deploy_mt4_91a9c2d3        # (production's qual/cfg.pkl and chat-<model>.txt)
DROPOTHER=$HOME/openTPU/models:$O:$HOME/otpu-build/offload/g26it
T1="What is the capital of France? Answer in one sentence."
T2="And what is it famous for? Two sentences."
T3="Which three museums in Paris should a first-time visitor see, and how long does each take?"
declare -A M=([q]="qwen35-35b Qwen3.5-35B-A3B pool-q35-fp4.split.bin"
              [g]="gemma4-26b gemma-4-26B-A4B-it ../g26it/pool-g26it-fp4.split.bin")
fail=0
echo "emdef4 start $(date +%T) T $T ($(cat $T/COMMIT)) PH $PH ($(cat $PH/COMMIT)) mem $(mem) GB, /home $(df -BG --output=avail /home | tail -1 | tr -d ' ') free; $(uptime)"
grep -q '^EM_ROWS = {"qwen35": 44, "gemma4": 20}' $T/opentpu/host/streamed.py \
  && grep -q "serve_s" $T/tools/qual/chatmoe.py \
  || { echo "STOP: T is not the commit with EM_ROWS 44 / 20 and chatmoe.py's serve_s"; exit 2; }
echo "  T's EM_ROWS: $(grep '^EM_ROWS = ' $T/opentpu/host/streamed.py)"
echo "  P's opentpu: $(cd /tmp && PYTHONPATH=$PH PYTHONDONTWRITEBYTECODE=1 python -c 'import opentpu; print(opentpu.__file__)')"
busy() { echo "  host: $(cut -d' ' -f1-3 /proc/loadavg); $(ps -eo pcpu,comm --sort=-pcpu | sed -n 2,4p | tr -s ' ' | tr '\n' ';')"; }
st=$(timeout 600 python -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "$st" | grep -q "build $EXPECT" || { echo "STOP: the card's build is not $EXPECT"; exit 2; }
echo "$st" | grep -q "register map 4" || { echo "STOP: not register map 4"; exit 2; }
echo "$st" | grep -q "ALL PASS" || { echo "STOP: selftest"; exit 2; }
timeout 300 python tools/qual/refs.py cfg $R/cfg-dev.pkl > /dev/null 2>&1; echo "cfg exit $?"
python - $R/cfg-dev.pkl $PD/qual/cfg.pkl <<'PY' || { echo "STOP: the device configuration"; exit 2; }
import dataclasses, pickle, sys
a, b = (pickle.load(open(f, "rb")) for f in sys.argv[1:])
d = {f.name: (getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
     if getattr(a, f.name) != getattr(b, f.name)}
print("  cfg: production's (qual/cfg.pkl)" if not d else f"  cfg DIFFERS from production's: {d}")
sys.exit(1 if d else 0)
PY
warm() {   # checkpoint, pool (in O): every other big file out of the page cache, the pool read into it
  python - $O/$1 $O/$2 $DROPOTHER <<'PY'
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
  local t0=$SECONDS; cat $O/$2 > /dev/null; echo "  pool read in $((SECONDS - t0)) s, mem $(mem) GB"
}
# 2. the production qual's chatmoe step, from T
echo "=== chatmoe.sh $(date +%T)"
env RUNS="qwen35-35b gemma4-26b" SESSION=chatmoe R=$R/chatmoe timeout 3600 bash $T/tools/qual/chatmoe.sh \
  > $R/chatmoe.log 2>&1
echo "  CHATMOE EXIT $? $(date +%T)"
grep -E "\[(PASS|FAIL|SKIP)\] [^ ]+ against|  turn [0-9]|    call: |Error|Traceback" $R/chatmoe.log | cut -c1-260
[ "$(grep -c "turns' tokens the same" $R/chatmoe.log)" = 2 ] && ! grep -q "\[FAIL\]" $R/chatmoe.log \
  || { echo "STOP: chatmoe"; fail=1; }
# 3. otpu-chat by name, from T
for m in q g; do
  [ $fail = 0 ] || break
  read -r name md pool <<< "${M[$m]}"
  echo "=== otpu-chat $name $(date +%T)"
  warm $md $pool
  (cd /tmp && timeout 1200 python -m opentpu.host.chat --backend board --model $name --greedy --max-new 24 \
     --prompt "$T1" > $R/chat-$name.txt 2>&1)
  echo "  CHAT $name EXIT $? $(date +%T)"
  python - $R/chat-$name.txt $PD/chat-$name.txt <<'PY' || { echo "STOP: otpu-chat $name"; fail=1; }
import sys
def part(f):
    ln = [x for x in open(f).read().splitlines() if x.strip()]
    reply = [x for x in ln if not x.startswith(("loading ", "experts:", "weights:", "loaded in", "[TTFT"))]
    return reply, [x for x in ln if x.startswith(("loaded in", "[TTFT"))]
(a, sa), (b, sb) = part(sys.argv[1]), part(sys.argv[2])
print(f"  [{'PASS' if a == b and a else 'FAIL'}] reply {a!r} {'=' if a == b else '!='} production's (mt4)")
for x, y in zip(sa, sb):
    print(f"    {x[:200]}\n    (mt4) {y[:200]}")
sys.exit(0 if a == b and a else 1)
PY
done
# 4. P (production's opentpu, T's tool) and N (T), per model: W, then ORDER
for m in q g; do
  [ $fail = 0 ] || break
  read -r name md pool <<< "${M[$m]}"
  for x in W $ORDER; do
    tree=$T; [ ${x:0:1} = P ] && tree=$PH          # (the opentpu it runs; the tool is T's)
    extra=(); [ $x = W ] || extra+=(--trace $R/$m$x.trace.json)
    echo "=== $m$x $(date +%T) ($name, opentpu $(cat $tree/COMMIT)${extra[*]:+, ${extra[*]}})"
    warm $md $pool
    busy
    (cd $T && env PYTHONPATH=$tree PYTHONDONTWRITEBYTECODE=1 timeout 1500 python $T/tools/qual/chatmoe.py \
       $O/$md --pool $O/$pool --backend board --cfg $R/cfg-dev.pkl --turn "$T1" --turn "$T2" --turn "$T3" \
       "${extra[@]}" --out $R/$m$x.json > $R/$m$x.log 2>&1)
    rc=$?; echo "  exit $rc $(date +%T)"; grep -E "Error|Traceback" $R/$m$x.log | tail -3
    [ $rc = 0 ] && [ -f $R/$m$x.json ] || { echo "STOP: $m$x"; fail=1; break; }
    python - $R/$m$x.json $R/${m}W.json $RF/chatmoe-$name.json <<'PY' || { echo "STOP: $m$x differs"; fail=1; break; }
import json, sys
d, w, ref = (json.load(open(f)) for f in sys.argv[1:])
tok = [t["tokens"] for t in d["turns"]]
ok_w = tok == [t["tokens"] for t in w["turns"]]
ok_r = tok[:2] == [t["tokens"] for t in ref["turns"][:2]]     # (the references: TURNS' two)
print(f"  [{'PASS' if ok_w and ok_r else 'FAIL'}] tokens {'=' if ok_w else '!='} W's, first two turns "
      f"{'=' if ok_r else '!='} the ISA reference's; layer-major {d['layer_major']}, em_rows "
      f"{d.get('em_rows', '-')}, load {d['load_s']} s")
for i, t in enumerate(d["turns"]):
    print(f"  turn {i + 1}: prompt {t['prompt_tokens']} (reused {t['reused']}) TTFT {t['ttft_s']} s, prefill "
          f"{t['prefill_tok_s']} tok/s (device {t['prefill_dev_tok_s']}), {t['prefill_misses']} misses; decode "
          f"{t['gen_tokens']} at {t['decode_tok_s']} tok/s (device {t['decode_dev_tok_s']}); end {t['end']}")
    for c in t.get("calls", []):
        print(f"    call {c['rows']} rows {'EM' if c['em'] else 'LM'}: {c['s']} s (device {c['device_s']}), "
              f"{c['runs']} runs, misses {c['misses']} ({c['requests']} requests, served in {c['serve_s']} s), "
              f"needs {c['needs_queued']} (hits {c['need_hits']}), parts {c['prefetched']} ({c['parts_s']} s), "
              f"holds {c['holds']}")
sys.exit(0 if ok_w and ok_r else 1)
PY
  done
done
echo "=== final selftest $(date +%T)"
timeout 600 python -m opentpu.host.selftest 2>&1 | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "emdef4 end $(date +%T) fail $fail"
python - $R "$ORDER" <<'PY'
import json, os, sys
r, order = sys.argv[1], sys.argv[2].split()
def get(run):
    f = f"{r}/{run}.json"
    return json.load(open(f)) if os.path.exists(f) else None
for name, m in (("35B", "q"), ("26B", "g")):
    a = {x: get(m + x) for x in order}
    if None in a.values():
        print(f"  {name}: incomplete"); continue
    P, N = [x for x in order if x[0] == "P"], [x for x in order if x[0] == "N"]
    pairs = [(n, "P" + n[1:]) for n in N if "P" + n[1:] in a]
    for i in range(len(a[P[0]]["turns"])):
        def v(x, k="ttft_s"):
            return a[x]["turns"][i][k] or 0.0
        def dev(x):
            t = a[x]["turns"][i]
            return t["prompt_tokens"] / t["prefill_dev_tok_s"]
        def mean(xs, f=v):
            return sum(f(x) for x in xs) / len(xs)
        def lm(x):          # the layer-major calls' host time a miss, ms
            cs = [c for c in a[x]["turns"][i].get("calls", []) if not c["em"] and c["misses"]]
            m_ = sum(c["misses"] for c in cs)
            return f"{1e3 * sum(c['serve_s'] for c in cs) / m_:.3f}" if m_ else "-"
        calls = " + ".join(f"{c['rows']}{'E' if c['em'] else 'L'}" for c in a[N[0]]["turns"][i].get("calls", []))
        print(f"  {name} turn {i + 1} ({a[P[0]]['turns'][i]['prompt_tokens']} tokens; N {calls}): TTFT N - P "
              + " ".join(f"{v(n) - v(p):+.3f}" for n, p in pairs) + f", of the means {mean(N) - mean(P):+.3f}; "
              f"device N - P of the means {mean(N, dev) - mean(P, dev):+.3f}")
        print("      TTFT " + ", ".join(f"{x} {v(x):.3f}" for x in order) + "; device "
              + ", ".join(f"{x} {dev(x):.3f}" for x in order))
        print("      layer-major calls' host ms a miss: " + ", ".join(f"{x} {lm(x)}" for x in order))
PY
date > $R/DONE
echo "EMDEF4 EXIT $fail $(date +%T)"
