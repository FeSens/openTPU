#!/bin/bash
# Session emc1 (docs/offload.md 13.18): expert-major's passes of C entries on the MX host. The expert run's passes
# take C = 4 entries of an expert on the card (moe.em_pass_rows: MCOLS 4 with PAIR, an expert's last pass of 1-2
# entries at PAIR's rate), today's take 2; each row's math is the same (the MX MXU's integer sums), so the tokens
# must be the same. A second question rides along: does expert-major's slot fill after the prompt cost decode hits
# (MX's 26B decode hits 0.604 against mt4's 0.661, 361.9 against 280.7 MB/token)? Arm L runs the same tree and
# engine with every call layer-major.
# Trees: T, the emc commit's host tree (TC; stage it with git archive, below), its COMMIT file holding the sha.
# Card: production deploy_mx_8b33206e as loaded (EXPECT 8b33206e, register map 4); its qual/cfg.pkl, the MX pools
# (pool-*-mx.split.bin) and the ISA references of its qual (deploy_mx_8b33206e/chatmoe: RF). One otpu-lock:
# 1. selftest (T); the device configuration must be production's qual/cfg.pkl;
# 2. per model (the 35B, the 26B) emdef2's three turns and a fourth of about 130 new tokens (wiki1.txt to sum up:
#    the co-sims' 131 / 124 rows, 13.17), arms A (T, --em-pass 2: today's passes), B (T, the configuration's C = 4)
#    and L (T, --em-rows 100000: every call layer-major), every A / B run traced: a discarded warm-up W (B's: the
#    image builds once for T, every arm's), then ORDER (default: ABBA x 4 with L twice,
#    "A1 B1 B2 A2 L1 B3 A3 A4 B4 L2"). Every run's tokens must equal W's, and its first two turns the ISA
#    references';
# 3. selftest; the summary: per model and turn the calls, B - A per pair and of the means (TTFT, device s, decode hit
#    rate, decode MB/token), then expert-major against layer-major (the A and B runs' means against L's: decode hit
#    rate, MB/token, TTFT).
# A build other than EXPECT, a failed selftest, another configuration, a failed or timed-out run or a mismatch stops
# it (then the final selftest). About 25 min (35B runs ~55 s, 26B ~65 s, 11 each; W builds T's images).
# Stage T (from a laptop, before the session; no host work while a session runs):
#   git archive TC | ssh CARD_HOST 'D=~/otpu-build/moe-ab/emc1/host; mkdir -p $D && tar -x -C $D'
#   ssh CARD_HOST 'echo <TC, full sha> > ~/otpu-build/moe-ab/emc1/host/COMMIT'
# Run (no pipe): setsid nohup ~/.local/bin/otpu-lock --wait 600 -- bash $T/tools/offload/sessions/emc1.sh
#   > ~/otpu-build/moe-ab/emc1.launch.log 2>&1 < /dev/null &
set -u
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*' | grep -vE '^OTPU_LOCK_(HELD|PID|WAIT)$'); do unset $v; done
export OTPU_IMG_CACHE_GB=30 OTPU_IMG_CACHE_FREE_GB=6      # (the card host's, from its ~/.bashrc)
MA=${MA:-$HOME/otpu-build/moe-ab}
T=${T:-$MA/emc1/host}
PD=${PD:-$HOME/otpu-build/deploy_mx_8b33206e}  # (production: qual/cfg.pkl, the chatmoe references)
RF=${RF:-$PD/chatmoe}
R=${R:-$MA/emc1/runs}
source $T/tools/offload/sessions/env.sh       # (O: card2; OTPU_QUIET claimed, removed at exit)
exec > >(tee -a $R/session.log) 2>&1
EXPECT=${EXPECT:-8b33206e}
TC=${TC:-}                                     # (the emc commit T must be; empty: any with --em-pass)
ORDER=${ORDER:-A1 B1 B2 A2 L1 B3 A3 A4 B4 L2}  # (after the warm-up W)
DROPOTHER=$HOME/openTPU/models:$O:$HOME/otpu-build/offload/g26it
T1="What is the capital of France? Answer in one sentence."
T2="And what is it famous for? Two sentences."
T3="Which three museums in Paris should a first-time visitor see, and how long does each take?"
T4="Sum this up in one sentence: $(cat $T/tools/offload/sessions/wiki1.txt)"
declare -A M=([q]="qwen35-35b Qwen3.5-35B-A3B pool-q35-mx.split.bin"
              [g]="gemma4-26b gemma-4-26B-A4B-it ../g26it/pool-g26it-mx.split.bin")
declare -A ARGS=([W]="" [A]="--em-pass 2" [B]="" [L]="--em-rows 100000")
fail=0
echo "emc1 start $(date +%T) T $T ($(cat $T/COMMIT)) PD $PD RF $RF mem $(mem) GB, /home $(df -BG --output=avail /home | tail -1 | tr -d ' ') free; $(uptime)"
grep -q '^def em_pass_rows' $T/opentpu/llm/moe.py && grep -q '"--em-pass"' $T/tools/qual/chatmoe.py \
  || { echo "STOP: T has no C-entry passes (moe.em_pass_rows, chatmoe.py --em-pass)"; exit 2; }
[ -z "$TC" ] || [ "$(cut -c1-${#TC} $T/COMMIT)" = "$TC" ] || { echo "STOP: T is not $TC"; exit 2; }
for m in q g; do
  read -r name md pool <<< "${M[$m]}"
  [ -f $O/$pool ] && [ -f $RF/chatmoe-$name.json ] || { echo "STOP: $O/$pool or $RF/chatmoe-$name.json missing"; exit 2; }
done
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
# 2. per model: W, then ORDER (A: --em-pass 2, B: the configuration's, L: every call layer-major), all from T
for m in q g; do
  [ $fail = 0 ] || break
  read -r name md pool <<< "${M[$m]}"
  for x in W $ORDER; do
    read -r -a extra <<< "${ARGS[${x:0:1}]}"
    case ${x:0:1} in A|B) extra+=(--trace $R/$m$x.trace.json);; esac
    echo "=== $m$x $(date +%T) ($name${extra[*]:+, ${extra[*]}})"
    warm $md $pool
    busy
    (cd $T && timeout 1800 python $T/tools/qual/chatmoe.py $O/$md --pool $O/$pool --backend board \
       --cfg $R/cfg-dev.pkl --turn "$T1" --turn "$T2" --turn "$T3" --turn "$T4" "${extra[@]}" --out $R/$m$x.json \
       > $R/$m$x.log 2>&1)
    rc=$?; echo "  exit $rc $(date +%T)"; grep -E "Error|Traceback" $R/$m$x.log | tail -3
    [ $rc = 0 ] && [ -f $R/$m$x.json ] || { echo "STOP: $m$x"; fail=1; break; }
    python - $R/$m$x.json $R/${m}W.json $RF/chatmoe-$name.json <<'PY' || { echo "STOP: $m$x differs"; fail=1; break; }
import json, sys
d, w, ref = (json.load(open(f)) for f in sys.argv[1:])
tok = [t["tokens"] for t in d["turns"]]
ok_w = tok == [t["tokens"] for t in w["turns"]]
ok_r = tok[:2] == [t["tokens"] for t in ref["turns"][:2]]     # (the references: TURNS' two)
print(f"  [{'PASS' if ok_w and ok_r else 'FAIL'}] tokens {'=' if ok_w else '!='} W's, first two turns "
      f"{'=' if ok_r else '!='} the ISA reference's; em_rows {d.get('em_rows', '-')}, em_pass "
      f"{d.get('em_pass', '-')}, load {d['load_s']} s")
for i, t in enumerate(d["turns"]):
    print(f"  turn {i + 1}: prompt {t['prompt_tokens']} (reused {t['reused']}) TTFT {t['ttft_s']} s, prefill "
          f"{t['prefill_tok_s']} tok/s (device {t['prefill_dev_tok_s']}), {t['prefill_misses']} misses; decode "
          f"{t['gen_tokens']} at {t['decode_tok_s']} tok/s (device {t['decode_dev_tok_s']}), hits {t['hit_rate']}, "
          f"{t['mb_per_token']} MB/token; end {t['end']}")
    for c in t.get("calls", []):
        print(f"    call {c['rows']} rows {'EM' if c['em'] else 'LM'}: {c['s']} s (device {c['device_s']}), "
              f"{c['runs']} runs, misses {c['misses']} ({c['requests']} requests, served in {c['serve_s']} s), "
              f"needs {c['needs_queued']} (hits {c['need_hits']}), holds {c['holds']}")
sys.exit(0 if ok_w and ok_r else 1)
PY
  done
done
echo "=== final selftest $(date +%T)"
timeout 600 python -m opentpu.host.selftest 2>&1 | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "emc1 end $(date +%T) fail $fail"
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
    A, B, L = ([x for x in order if x[0] == c] for c in "ABL")
    pairs = [(b, "A" + b[1:]) for b in B if "A" + b[1:] in a]
    for i in range(len(a[B[0]]["turns"])):
        def v(x, k="ttft_s"):
            return a[x]["turns"][i][k] or 0.0
        def dev(x):
            t = a[x]["turns"][i]
            return t["prompt_tokens"] / t["prefill_dev_tok_s"]
        def mean(xs, f=v, k="ttft_s"):
            return sum(f(x, k) if f is v else f(x) for x in xs) / len(xs) if xs else float("nan")
        def calls(x):
            return " + ".join(f"{c['rows']}{'E' if c['em'] else 'L'}" for c in a[x]["turns"][i].get("calls", []))
        print(f"  {name} turn {i + 1} ({a[B[0]]['turns'][i]['prompt_tokens']} tokens; B {calls(B[0])}"
              + (f", L {calls(L[0])}" if L else "") + "):")
        for k, lab, fmt in (("ttft_s", "TTFT s", "+.3f"), ("hit_rate", "decode hits", "+.4f"),
                            ("mb_per_token", "decode MB/token", "+.2f")):
            print(f"      {lab}: B - A " + " ".join(f"{v(b, k) - v(p, k):{fmt}}" for b, p in pairs)
                  + f", of the means {mean(B, v, k) - mean(A, v, k):{fmt}}"
                  + (f"; em (A, B) - L of the means {mean(A + B, v, k) - mean(L, v, k):{fmt}}" if L else "")
                  + "; " + ", ".join(f"{x} {v(x, k):{fmt[1:]}}" for x in order))
        print(f"      device s: B - A of the means {mean(B, dev) - mean(A, dev):+.3f}"
              + (f", em (A, B) - L {mean(A + B, dev) - mean(L, dev):+.3f}" if L else "") + "; "
              + ", ".join(f"{x} {dev(x):.3f}" for x in order))
PY
date > $R/DONE
echo "EMC1 EXIT $fail $(date +%T)"
