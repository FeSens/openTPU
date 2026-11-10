#!/bin/bash
# Session les1 (docs/offload.md 15.4): what LES gives the card, production's host against les's on production's
# bitstream. les = release-2's r2 (4ead4e53) + two host changes, both bit for bit:
#   S  the shared expert beside the request (moe.moe_ffn / moe_ffn_rows): Qwen3.5-MoE only, its programs change;
#   O  an expert-major need line counts as a request, and the finished layer's experts go to the pooled order's end
#      by their last need line (docs/offload.md 13.18, ExpertServer.need / _at): the slots after an EM prompt.
# Arms: A = production's host tree (PD/host, d8fe2c15), B = les's (TB, its COMMIT). Both models, five turns of
# --max-new 32: emc1's four (the 35B's turns 3-4 expert-major, 45 / ~157 rows; the 26B's 1-4) and a short fifth.
# Predicted (tools/offload/lesim.py on session 16's card fit; docs/offload.md 15): S, the 35B's decode about -2.5% a
# token (F less 0.136 ms on every request with misses; ~-5 ms a token at the chat's ~120 misses), its layer-major
# prompts a little shorter (~0.14 ms a request with misses); the 26B nothing (no shared expert). O, the decode after
# an expert-major prompt: emc1's 35B turn 4 lost 0.035 hits / 18.9 MB a token against layer-major; em_slotsim bounds
# the fix at 177.0 -> 166.7 MB (16 decode rows); the hit rates and MB a token are deterministic per arm (emc1: equal
# in every run of an arm), so B - A on them is O's effect alone; its time is the rest of a later turn's decode
# change beyond S's per-token change in turns 1-2.
# Card: production deploy_r2m4_b10245f5 as loaded (EXPECT b10245f5, register map 5); its qual/cfg.pkl; the MX pools;
# the ISA references: PD/chatmoe (TURNS' two, production's tree: RF) and les's five-turn ones (RL: chatmoe5-*.json,
# computed on the simulation host from TB on deploy_r2m4_d352e53d.pkl = r2's configuration). One otpu-lock:
# 1. selftest; the device configuration must be production's qual/cfg.pkl;
# 2. per model (the 35B, then the 26B): a discarded warm-up of each arm (WA, WB: each tree builds its own images),
#    then ORDER (default ABBA x 2: "A1 B1 B2 A2 A3 B3 B4 A4"), each run after the pool's read; every run's tokens
#    must equal RL's in all five turns and RF's first two (RF's at --max-new 16: turn 1's eos, turn 2's 16 first);
# 3. selftest; the summary per model and turn: B - A per pair and of the means (TTFT, prefill device s, decode
#    device ms a token, decode hit rate, MB a token), then the split: S = the 35B's decode ms a token B - A over
#    the turns before its first expert-major prompt (1-2), O = each later turn's hits and MB a token B - A, and its
#    ms a token beyond S's (the 26B: no S, all of B - A is O's).
# A build other than EXPECT, a failed selftest, another configuration, a failed or timed-out run or a token mismatch
# stops it (then the final selftest). About 40 min (20 runs of 60-90 s, each after a pool read of ~40 s; W builds).
# Stage (from a laptop, before the session; no host work while a session runs):
#   git archive <les sha> | ssh CARD_HOST 'D=~/otpu-build/les/les1/host; mkdir -p $D && tar -x -C $D'
#   ssh CARD_HOST 'echo <les sha, full> > ~/otpu-build/les/les1/host/COMMIT'
#   scp chatmoe5-qwen35-35b.json chatmoe5-gemma4-26b.json CARD_HOST:otpu-build/les/les1/refs/
# Run (no pipe): setsid nohup ~/.local/bin/otpu-lock --wait 600 -- bash ~/otpu-build/les/les1/host/tools/offload/sessions/les1.sh
#   > ~/otpu-build/les/les1.launch.log 2>&1 < /dev/null &
set -u
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*' | grep -vE '^OTPU_LOCK_(HELD|PID|WAIT)$'); do unset $v; done
export OTPU_IMG_CACHE_GB=30 OTPU_IMG_CACHE_FREE_GB=6      # (the card host's, from its ~/.bashrc)
LS=${LS:-$HOME/otpu-build/les/les1}
PD=${PD:-$HOME/otpu-build/deploy_r2m4_b10245f5}  # (production: host/, qual/cfg.pkl, chatmoe/)
TA=${TA:-$PD/host}
TB=${TB:-$LS/host}
RF=${RF:-$PD/chatmoe}
RL=${RL:-$LS/refs}
R=${R:-$LS/runs}
T=$TB
source $TB/tools/offload/sessions/env.sh       # (O: card2; OTPU_QUIET claimed, removed at exit)
exec > >(tee -a $R/session.log) 2>&1
EXPECT=${EXPECT:-b10245f5}
MAP=${MAP:-register map 5}
ORDER=${ORDER:-A1 B1 B2 A2 A3 B3 B4 A4}       # (after the warm-ups WA, WB)
MAXNEW=${MAXNEW:-32}
DROPOTHER=$HOME/openTPU/models:$O:$HOME/otpu-build/offload/g26it
T1="What is the capital of France? Answer in one sentence."
T2="And what is it famous for? Two sentences."
T3="Which three museums in Paris should a first-time visitor see, and how long does each take?"
T4="Sum this up in one sentence: $(cat $TB/tools/offload/sessions/wiki1.txt)"
T5="Name two emperors of that period."
declare -A M=([q]="qwen35-35b Qwen3.5-35B-A3B pool-q35-mx.split.bin"
              [g]="gemma4-26b gemma-4-26B-A4B-it ../g26it/pool-g26it-mx.split.bin")
fail=0
echo "les1 start $(date +%T) A $TA ($(cat $TA/COMMIT)) B $TB ($(cat $TB/COMMIT)) PD $PD RF $RF RL $RL mem $(mem) GB," \
  "/home $(df -BG --output=avail /home | tail -1 | tr -d ' ') free; $(uptime)"
grep -q 'sh = swiglu_down(xs, lw.wg, lw.wu, lw.wd)' $TB/opentpu/llm/moe.py && grep -q 'def _count' $TB/opentpu/host/offload.py \
  || { echo "STOP: TB is not les (the shared expert beside the request, need's count)"; exit 2; }
grep -q 'def _count' $TA/opentpu/host/offload.py && { echo "STOP: TA already has les's server"; exit 2; }
for m in q g; do
  read -r name md pool <<< "${M[$m]}"
  [ -f $O/$pool ] && [ -f $RF/chatmoe-$name.json ] && [ -f $RL/chatmoe5-$name.json ] \
    || { echo "STOP: $O/$pool, $RF/chatmoe-$name.json or $RL/chatmoe5-$name.json missing"; exit 2; }
done
busy() { echo "  host: $(cut -d' ' -f1-3 /proc/loadavg); $(ps -eo pcpu,comm --sort=-pcpu | sed -n 2,4p | tr -s ' ' | tr '\n' ';')"; }
st=$(timeout 600 python -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "$st" | grep -q "build $EXPECT" || { echo "STOP: the card's build is not $EXPECT"; exit 2; }
echo "$st" | grep -q "$MAP" || { echo "STOP: not $MAP"; exit 2; }
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
# 2. per model: WA, WB, then ORDER; A from TA, B from TB, each its own PYTHONPATH
for m in q g; do
  [ $fail = 0 ] || break
  read -r name md pool <<< "${M[$m]}"
  for x in WA WB $ORDER; do
    case $x in WA|A*) tr=$TA;; *) tr=$TB;; esac
    echo "=== $m$x $(date +%T) ($name, $(basename $(dirname $tr))/$(basename $tr) $(cut -c1-8 $tr/COMMIT))"
    warm $md $pool
    busy
    (cd $tr && PYTHONPATH=$tr timeout 1800 python $tr/tools/qual/chatmoe.py $O/$md --pool $O/$pool --backend board \
       --cfg $R/cfg-dev.pkl --turn "$T1" --turn "$T2" --turn "$T3" --turn "$T4" --turn "$T5" --max-new $MAXNEW \
       --out $R/$m$x.json > $R/$m$x.log 2>&1)
    rc=$?; echo "  exit $rc $(date +%T)"; grep -E "Error|Traceback" $R/$m$x.log | tail -3
    [ $rc = 0 ] && [ -f $R/$m$x.json ] || { echo "STOP: $m$x"; fail=1; break; }
    python - $R/$m$x.json $RL/chatmoe5-$name.json $RF/chatmoe-$name.json <<'PY' || { echo "STOP: $m$x differs"; fail=1; break; }
import json, sys
d, rl, rf = (json.load(open(f)) for f in sys.argv[1:])
tok = [t["tokens"] for t in d["turns"]]
ok_l = tok == [t["tokens"] for t in rl["turns"]]
ok_f = all(x == t["tokens"] if t["end"] == "eos" else x[:len(t["tokens"])] == t["tokens"]   # (RF's at --max-new 16:
           for x, t in zip(tok, rf["turns"][:2]))                                         #  turn 2 a prefix)
print(f"  [{'PASS' if ok_l and ok_f else 'FAIL'}] tokens {'=' if ok_l else '!='} les's five-turn ISA reference, "
      f"first two {'=' if ok_f else '!='} production's; em_rows {d.get('em_rows', '-')}, load {d['load_s']} s")
for i, t in enumerate(d["turns"]):
    print(f"  turn {i + 1}: prompt {t['prompt_tokens']} (reused {t['reused']}) TTFT {t['ttft_s']} s, prefill "
          f"{t['prefill_tok_s']} tok/s (device {t['prefill_dev_tok_s']}), {t['prefill_misses']} misses; decode "
          f"{t['gen_tokens']} at {t['decode_tok_s']} tok/s (device {t['decode_dev_tok_s']}), hits {t['hit_rate']}, "
          f"{t['mb_per_token']} MB/token; end {t['end']}")
    for c in t.get("calls", []):
        print(f"    call {c['rows']} rows {'EM' if c['em'] else 'LM'}: {c['s']} s (device {c['device_s']}), "
              f"{c['runs']} runs, misses {c['misses']} ({c['requests']} requests, served in {c['serve_s']} s), "
              f"needs {c['needs_queued']} (hits {c['need_hits']}), holds {c['holds']}")
sys.exit(0 if ok_l and ok_f else 1)
PY
  done
done
echo "=== final selftest $(date +%T)"
timeout 600 python -m opentpu.host.selftest 2>&1 | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "les1 end $(date +%T) fail $fail"
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
    A, B = ([x for x in order if x[0] == c] for c in "AB")
    pairs = [(b, "A" + b[1:]) for b in B if "A" + b[1:] in a]
    n = len(a[B[0]]["turns"])
    def tv(x, i, k):
        t = a[x]["turns"][i]
        if k == "dev_s":                    # the prompt's device seconds
            return t["prompt_tokens"] / t["prefill_dev_tok_s"] if t["prefill_dev_tok_s"] else 0.0
        if k == "ms_tok":                   # the decode's device ms a token
            return 1000.0 / t["decode_dev_tok_s"] if t["decode_dev_tok_s"] else 0.0
        return t[k] or 0.0
    def mean(xs, i, k):
        return sum(tv(x, i, k) for x in xs) / len(xs)
    split = {}
    for i in range(n):
        calls = " + ".join(f"{c['rows']}{'E' if c['em'] else 'L'}" for c in a[B[0]]["turns"][i].get("calls", []))
        print(f"  {name} turn {i + 1} ({a[B[0]]['turns'][i]['prompt_tokens']} tokens; {calls}):")
        for k, lab, fmt in (("ttft_s", "TTFT s", "+.3f"), ("dev_s", "prompt device s", "+.3f"),
                            ("ms_tok", "decode device ms/token", "+.2f"), ("hit_rate", "decode hits", "+.4f"),
                            ("mb_per_token", "decode MB/token", "+.2f")):
            d = mean(B, i, k) - mean(A, i, k)
            split[(i, k)] = (d, mean(A, i, k))
            print(f"      {lab}: B - A " + " ".join(f"{tv(b, i, k) - tv(p, i, k):{fmt}}" for b, p in pairs)
                  + f", of the means {d:{fmt}}; " + ", ".join(f"{x} {tv(x, i, k):{fmt[1:]}}" for x in order))
    em = [i for i in range(n) if any(c["em"] for c in a[B[0]]["turns"][i].get("calls", []))]
    clean = [i for i in range(n) if not em or i < em[0]]    # (decodes no expert-major prompt came before)
    s_rel = 0.0
    if m == "q" and clean:              # S: the 35B's decode before any expert-major prompt (O can't act there)
        s = sum(split[(i, "ms_tok")][0] for i in clean) / len(clean)
        s_rel = sum(split[(i, "ms_tok")][0] / split[(i, "ms_tok")][1] for i in clean) / len(clean)
        print(f"  {name} split: S (the shared expert) = decode B - A over turns {[i + 1 for i in clean]}: "
              f"{s:+.2f} ms a token ({s_rel * 100:+.2f}%)")
    else:                               # (the 26B: no shared expert, byte-identical programs)
        print(f"  {name} split: no S (no shared expert): B - A is O's alone, and the noise in turns "
              f"{[i + 1 for i in clean]}")
    for i in range(n):
        if i not in clean:
            d, base = split[(i, "ms_tok")]
            print(f"  {name} split: turn {i + 1}: O (the slots after an expert-major prompt) hits "
                  f"{split[(i, 'hit_rate')][0]:+.4f}, MB/token {split[(i, 'mb_per_token')][0]:+.2f}; decode "
                  f"{d:+.2f} ms a token, beyond S {d - s_rel * base:+.2f}")
PY
date > $R/DONE
echo "LES1 EXIT $fail $(date +%T)"
