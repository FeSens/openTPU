#!/bin/bash
# Session mtp2 (docs/offload.md 15.6): mtp1c's A / B again with the two TTFT fixes, and a traced pair. A = production's
# host tree (PD/host: deploy_hostr21_b10245f5, d966c3df, with S), plain decode from production's pool; B = mcpv-mtp
# (TB, its COMMIT: les-mtp's MoE MTP drafter with MiniCPM-V on main, and F1b: the loop's first run drafts the prompt's
# last row; F2: the chat's snapshot at the reply's start holds the row before it), chatmoe --mtp from the MTP pool (QM,
# mtp1's). The 35B's five les1 turns of --max-new 32. Predicted TTFT B - A from mtp1c's parts with F1b and F2
# (MODEL, turns 1-5): -0.31 +0.06 +0.33 +0.33 +0.59 s (docs/offload.md 15.6); decode as mtp1c's (1.127x).
# Card: production deploy_hostr21_b10245f5 as loaded (EXPECT b10245f5, register map 5); its qual/cfg.pkl; the ISA
# references: PD/chatmoe (TURNS' two: RF) and les's five-turn ones (RL: chatmoe5-qwen35-35b.json in mtp1's refs/).
# One otpu-lock:
# 1. selftest; the device configuration must be production's qual/cfg.pkl;
# 2. a discarded warm-up of each arm (WA, WB: each tree builds its own image), then ORDER (default ABBA x 2), then
#    TRACE (default AT BT: one run of each arm with chatmoe --trace, each call's runs and the expert server's
#    timeline; outside the timed runs), each run after its pool's read (both pools wholly in the page cache before
#    their runs, fincore, else STOP); every run's tokens must equal RL's in all five turns and RF's first two;
# 3. selftest; the summary per turn as mtp1's, B - A's TTFT against MODEL, and from the traced pair the expert-major
#    calls (turns 3-5) by run kind: device ms, wall ms, the gaps between runs, the server's parts and requests.
# A build other than EXPECT, a failed selftest, another configuration, a failed or timed-out run or a token mismatch
# stops it (then the final selftest). About 50 min: 12 runs of 70-120 s, W's builds, the pools' reads (~130 s from
# the SSD at each switch of arm: the two pools do not fit the card host's page cache together).
# Stage (from a laptop; no host work while a session runs; mtp1's pool, checkpoint and refs as
# they are in ~/otpu-build/les/mtp1, only read):
#   git archive <mcpv-mtp sha> | ssh CARD_HOST 'D=~/otpu-build/les/mtp2/host; mkdir -p $D && tar -x -C $D'
#   ssh CARD_HOST 'echo <mcpv-mtp sha, full> > ~/otpu-build/les/mtp2/host/COMMIT; mkdir -p ~/otpu-build/les/mtp2/runs'
# Run (no pipe): QM=$HOME/otpu-build/les/mtp1/pool-q35-mx-mtp.split.bin setsid nohup ~/.local/bin/otpu-lock --wait 600 --
#   bash ~/otpu-build/les/mtp2/host/tools/offload/sessions/mtp2.sh > ~/otpu-build/les/mtp2.launch.log 2>&1 < /dev/null &
set -u
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*' | grep -vE '^OTPU_LOCK_(HELD|PID|WAIT)$'); do unset $v; done
export OTPU_IMG_CACHE_GB=30 OTPU_IMG_CACHE_FREE_GB=6      # (the card host's, from its ~/.bashrc)
LS=${LS:-$HOME/otpu-build/les/mtp2}
M1=${M1:-$HOME/otpu-build/les/mtp1}               # (mtp1's MTP checkpoint and refs)
PD=${PD:-$HOME/otpu-build/deploy_hostr21_b10245f5}  # (production: host/, qual/cfg.pkl, chatmoe/)
TA=${TA:-$PD/host}
TB=${TB:-$LS/host}
RF=${RF:-$PD/chatmoe}
RL=${RL:-$M1/refs}
R=${R:-$LS/runs}
T=$TB
source $TB/tools/offload/sessions/env.sh       # (O: card2; OTPU_QUIET claimed, removed at exit)
exec > >(tee -a $R/session.log) 2>&1
EXPECT=${EXPECT:-b10245f5}
MAP=${MAP:-register map 5}
ORDER=${ORDER:-A1 B1 B2 A2 A3 B3 B4 A4}       # (after the warm-ups WA, WB)
TRACE=${TRACE-AT BT}                            # (after ORDER: --trace; empty: none)
MODEL=${MODEL:--0.31 0.06 0.33 0.33 0.59}       # (TTFT B - A predicted, turns 1-5, s)
MAXNEW=${MAXNEW:-32}
S_MS=${S_MS:-0}                                 # (A has S: B - A is MTP alone)
MD=Qwen3.5-35B-A3B
MDB=${MDB:-$M1/$MD-mtp}                         # (B's checkpoint: card2's files linked, and mtp.safetensors)
QA=${QA:-pool-q35-mx.split.bin}                 # (in O)
QM=${QM:?the MTP pool (choose its disk: beside QA on /home, or on /data and kept warm)}
DROPOTHER=${DROPOTHER:-$HOME/openTPU:$HOME/otpu-build:/data/otpu}   # (all of ours: mtp1b, 2026-10-14, stopped
                                # before its first run, MiniCPM-V's pool on /data/otpu cached since session hostr21)
T1="What is the capital of France? Answer in one sentence."
T2="And what is it famous for? Two sentences."
T3="Which three museums in Paris should a first-time visitor see, and how long does each take?"
T4="Sum this up in one sentence: $(cat $TB/tools/offload/sessions/wiki1.txt)"
T5="Name two emperors of that period."
name=qwen35-35b
fail=0
echo "mtp2 start $(date +%T) A $TA ($(cat $TA/COMMIT)) B $TB ($(cat $TB/COMMIT)) PD $PD RF $RF RL $RL QM $QM" \
  "mem $(mem) GB, /home $(df -BG --output=avail /home | tail -1 | tr -d ' ') free; $(uptime)"
grep -q 'mtp_from' $TB/opentpu/host/streamed.py && grep -q '"--mtp"' $TB/tools/qual/chatmoe.py \
  && grep -q 'def _settle' $TB/opentpu/llm/mtp.py && grep -q 'def holds' $TB/opentpu/llm/mtp.py \
  || { echo "STOP: TB is not mcpv-mtp with F1b and F2 (streamed.engine's mtp, chatmoe --mtp, MTPDecoder._settle, holds)"; exit 2; }
grep -q 'mtp_from' $TA/opentpu/host/streamed.py && { echo "STOP: TA already has les-mtp's engine"; exit 2; }
[ -f $O/$QA ] && [ -f $QM ] && [ -f $RF/chatmoe-$name.json ] && [ -f $RL/chatmoe5-$name.json ] \
  || { echo "STOP: $O/$QA, $QM, $RF/chatmoe-$name.json or $RL/chatmoe5-$name.json missing"; exit 2; }
[ -f $MDB/mtp.safetensors ] && [ -f $MDB/config.json ] || { echo "STOP: $MDB is not B's checkpoint"; exit 2; }
(cd $TB && PYTHONPATH=$TB python tools/offload/pack_pool.py $MDB $QM info --mtp) 2>&1 | tail -3 > $R/qm-info.txt
cat $R/qm-info.txt
grep -q "10496 of 10496 experts packed" $R/qm-info.txt && grep -q "the same key" $R/qm-info.txt \
  || { echo "STOP: the MTP pool is not whole or not this image's"; exit 2; }
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
warm() {   # checkpoint, pool: every other big file out of the page cache, the pool read into it
  python - $1 $2 $DROPOTHER <<'PY'
import os, sys                      # every other file of 100 MB or more out of the page cache
keep = {os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])}  # (card_moe.sh's DROPOTHER)
keep |= {os.path.realpath(os.path.join(sys.argv[1], f)) for f in os.listdir(sys.argv[1])}  # (B's links)
ours = tuple(os.path.realpath(os.path.expanduser(d)) + os.sep for d in ("~/openTPU", "~/otpu-build", "/data/otpu"))
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
  local t0=$SECONDS k
  for k in 1 2; do                      # (every block read; a second read if pages were dropped)
    dd if=$2 of=/dev/null bs=16M status=none
    [ "$(res $2)" -ge 99 ] && break
  done
  echo "  pool read in $((SECONDS - t0)) s, $(res $2)% of it in the page cache, mem $(mem) GB, cached" \
    "$(awk '/^Cached:/ {print int($2/1048576)}' /proc/meminfo) GB"
  [ "$(res $2)" -ge 99 ]
}
res() {    # the percentage of a file in the page cache (util-linux fincore)
  local r=$(fincore -b -n -o RES "$1" 2>/dev/null | tr -d ' ')
  echo $(( ${r:-0} * 100 / $(stat -c %s "$1") ))
}
# 2. WA, WB, then ORDER; A from TA (plain, production's pool), B from TB (--mtp, the MTP pool)
for x in WA WB $ORDER $TRACE; do
  case $x in WA|A*) tr=$TA; md=$O/$MD; pool=$O/$QA; extra="";; *) tr=$TB; md=$MDB; pool=$QM; extra="--mtp";; esac
  echo "=== q$x $(date +%T) ($name, $(basename $(dirname $tr))/$(basename $tr) $(cut -c1-8 $tr/COMMIT) $extra)"
  warm $md $pool || { echo "STOP: $pool not wholly in the page cache (both arms' pools must be warm)"; fail=1; break; }
  case $x in ?T) extra="$extra --trace $R/q$x.trace.json";; esac   # (TRACE's: outside ORDER's timing)
  busy
  (cd $tr && PYTHONPATH=$tr timeout 1800 python $tr/tools/qual/chatmoe.py $md --pool $pool --backend board \
     --cfg $R/cfg-dev.pkl --turn "$T1" --turn "$T2" --turn "$T3" --turn "$T4" --turn "$T5" --max-new $MAXNEW \
     $extra --out $R/q$x.json > $R/q$x.log 2>&1)
  rc=$?; echo "  exit $rc $(date +%T); after it $(res $pool)% of its pool in the page cache"
  grep -E "Error|Traceback" $R/q$x.log | tail -3
  [ $rc = 0 ] && [ -f $R/q$x.json ] || { echo "STOP: q$x"; fail=1; break; }
  python - $R/q$x.json $RL/chatmoe5-$name.json $RF/chatmoe-$name.json <<'PY' || { echo "STOP: q$x differs"; fail=1; break; }
import json, sys
d, rl, rf = (json.load(open(f)) for f in sys.argv[1:])
tok = [t["tokens"] for t in d["turns"]]
ok_l = tok == [t["tokens"] for t in rl["turns"]]
ok_f = all(x == t["tokens"] if t["end"] == "eos" else x[:len(t["tokens"])] == t["tokens"]   # (RF's at --max-new 16:
           for x, t in zip(tok, rf["turns"][:2]))                                         #  turn 2 a prefix)
print(f"  [{'PASS' if ok_l and ok_f else 'FAIL'}] tokens {'=' if ok_l else '!='} les's five-turn ISA reference, "
      f"first two {'=' if ok_f else '!='} production's; mtp {d.get('mtp', False)}, {d['experts']}, load {d['load_s']} s")
for i, t in enumerate(d["turns"]):
    acc = f", MTP {t['mtp_accepted']} of {t['mtp_iters']} drafts" if t.get("mtp_iters") else ""
    print(f"  turn {i + 1}: prompt {t['prompt_tokens']} (reused {t['reused']}) TTFT {t['ttft_s']} s, prefill "
          f"{t['prefill_tok_s']} tok/s (device {t['prefill_dev_tok_s']}), {t['prefill_misses']} misses; decode "
          f"{t['gen_tokens']} at {t['decode_tok_s']} tok/s (device {t['decode_dev_tok_s']}), hits {t['hit_rate']}, "
          f"{t['mb_per_token']} MB/token{acc}; end {t['end']}")
    for c in t.get("calls", []):
        print(f"    call {c['rows']} rows {'EM' if c['em'] else 'LM'}: {c['s']} s (device {c['device_s']}), "
              f"{c['runs']} runs, misses {c['misses']} ({c['requests']} requests, served in {c['serve_s']} s), "
              f"needs {c['needs_queued']} (hits {c['need_hits']}), holds {c['holds']}")
sys.exit(0 if ok_l and ok_f else 1)
PY
done
echo "=== final selftest $(date +%T)"
timeout 600 python -m opentpu.host.selftest 2>&1 | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "mtp2 end $(date +%T) fail $fail"
python - $R "$ORDER" "$S_MS" "$MODEL" <<'PY'
import json, os, sys
r, order, s_ms = sys.argv[1], sys.argv[2].split(), float(sys.argv[3])
model = [float(x) for x in sys.argv[4].split()]
a = {x: json.load(open(f"{r}/q{x}.json")) if os.path.exists(f"{r}/q{x}.json") else None for x in order}
if None in a.values():
    print("  35B: incomplete"); sys.exit(0)
A, B = ([x for x in order if x[0] == c] for c in "AB")
pairs = [(b, "A" + b[1:]) for b in B if "A" + b[1:] in a]
n = len(a[B[0]]["turns"])
def tv(x, i, k):
    t = a[x]["turns"][i]
    if k == "dev_s":                    # the prompt's device seconds
        return t["prompt_tokens"] / t["prefill_dev_tok_s"] if t["prefill_dev_tok_s"] else 0.0
    if k == "ms_tok":                   # the decode's device ms a token
        return 1000.0 / t["decode_dev_tok_s"] if t["decode_dev_tok_s"] else 0.0
    return t.get(k) or 0.0
def mean(xs, i, k):
    return sum(tv(x, i, k) for x in xs) / len(xs)
tot = {"A": [0.0, 0], "B": [0.0, 0]}
for i in range(n):
    calls = " + ".join(f"{c['rows']}{'E' if c['em'] else 'L'}" for c in a[B[0]]["turns"][i].get("calls", []))
    print(f"  35B turn {i + 1} ({a[B[0]]['turns'][i]['prompt_tokens']} tokens; {calls}):")
    for k, lab, fmt in (("ttft_s", "TTFT s", "+.3f"), ("dev_s", "prompt device s", "+.3f"),
                        ("ms_tok", "decode device ms/token", "+.2f"), ("hit_rate", "decode hits", "+.4f"),
                        ("mb_per_token", "decode MB/token", "+.2f")):
        d = mean(B, i, k) - mean(A, i, k)
        print(f"      {lab}: B - A " + " ".join(f"{tv(b, i, k) - tv(p, i, k):{fmt}}" for b, p in pairs)
              + f", of the means {d:{fmt}}; " + ", ".join(f"{x} {tv(x, i, k):{fmt[1:]}}" for x in order))
    it = sum(a[x]["turns"][i].get("mtp_iters", 0) for x in B)
    ac = sum(a[x]["turns"][i].get("mtp_accepted", 0) for x in B)
    ma, mb = mean(A, i, "ms_tok"), mean(B, i, "ms_tok")
    g = a[B[0]]["turns"][i]["gen_tokens"]
    for c, m in (("A", ma), ("B", mb)):
        tot[c][0] += m * g
        tot[c][1] += g
    print(f"      TTFT B - A of the means {mean(B, i, 'ttft_s') - mean(A, i, 'ttft_s'):+.3f} s against the "
          f"model's {model[i]:+.2f} (mtp1c's parts less F1b's and F2's)" if i < len(model) else "")
    print(f"      MTP: acceptance {ac / max(1, it):.3f} ({ac} of {it}); decode {ma / mb if mb else 0:.3f}x "
          f"(A {ma:.1f}, B {mb:.1f} ms a token); beyond S ({s_ms:+.2f} ms) {ma / (mb - s_ms) if mb else 0:.3f}x")
ma, mb = (tot[c][0] / max(1, tot[c][1]) for c in "AB")
print(f"  35B decode over the turns (token-weighted): A {ma:.1f}, B {mb:.1f} ms a token: {ma / mb:.3f}x; MTP alone "
      f"(B less S's {s_ms:+.2f}) {ma / (mb - s_ms):.3f}x")
PY
[ -f $R/qAT.trace.json ] && [ -f $R/qBT.trace.json ] && python - $R <<'PY'
import json, sys                # the traced pair's prompt calls in turns 3-5 (expert-major's), by run kind
r = sys.argv[1]
tr = {x: json.load(open(f"{r}/q{x}.trace.json"))["calls"] for x in ("AT", "BT")}
js = {x: json.load(open(f"{r}/q{x}.json")) for x in ("AT", "BT")}
def kind(k):                    # (qwen3.layer_programs' keys)
    if not isinstance(k, list):
        return str(k)
    return {"x": "expert", "m": "mtp kv", "v": "vis", -1: "embed"}.get(k[0], "layer")
for i in (2, 3, 4):
    out = {}
    for x in ("AT", "BT"):
        n0 = sum(len(t["calls"]) for t in js[x]["turns"][:i])
        cs = js[x]["turns"][i]["calls"]
        k, rows, parts, req = {}, [], 0.0, 0.0
        for c, cj in zip(tr[x][n0:n0 + len(cs)], cs):
            cyc = sum(run[3] or 0 for run in c["runs"] if len(run) >= 4)
            khz = cyc / (cj["device_s"] * 1e3) if cj.get("device_s") else 133333.0
            rows.append(f"{c['rows']} {'EM' if c['em'] else 'LM'}")
            prev = None
            for run in c["runs"]:
                if len(run) < 4:
                    continue
                key, t0, t1, cy = run[:4]
                d = k.setdefault(kind(key), [0, 0.0, 0.0, 0.0])
                d[0] += 1
                d[1] += (cy or 0) / khz
                d[2] += 1e3 * (t1 - t0)
                d[3] += 0.0 if prev is None else 1e3 * max(0.0, t0 - prev)
                prev = t1
            parts += sum(e[1] - e[0] for e in c["events"] if e[2] == "p")
            req += sum(e[1] - e[0] for e in c["events"] if e[2] == "d")
        out[x] = (rows, k, parts, req, js[x]["turns"][i]["ttft_s"])
    print(f"  turn {i + 1}: TTFT {out['AT'][4]} / {out['BT'][4]} s (AT / BT); calls " +
          "; ".join(f"{x} {', '.join(o[0])}" for x, o in out.items()))
    tot = [[0.0] * 3 for _ in range(2)]
    for kd in sorted(set(out["AT"][1]) | set(out["BT"][1])):
        a, b = (out[x][1].get(kd, [0, 0.0, 0.0, 0.0]) for x in ("AT", "BT"))
        for t, v in zip(tot, (a, b)):
            t[0] += v[1]; t[1] += v[2]; t[2] += v[3]
        print(f"    {kd:7s} runs {a[0]:5d} / {b[0]:5d}; device ms {a[1]:8.1f} / {b[1]:8.1f} ({b[1] - a[1]:+7.1f}); "
              f"wall ms {a[2]:8.1f} / {b[2]:8.1f} ({b[2] - a[2]:+7.1f}); gaps before ms {a[3]:7.1f} / {b[3]:7.1f} "
              f"({b[3] - a[3]:+7.1f})")
    a, b = tot
    print(f"    all     device ms {a[0]:.1f} / {b[0]:.1f} ({b[0] - a[0]:+.1f}); wall ms {a[1]:.1f} / {b[1]:.1f} "
          f"({b[1] - a[1]:+.1f}); gaps ms {a[2]:.1f} / {b[2]:.1f} ({b[2] - a[2]:+.1f}); server parts "
          f"{out['AT'][2]:.3f} / {out['BT'][2]:.3f} s, requests {out['AT'][3]:.3f} / {out['BT'][3]:.3f} s")
PY
date > $R/DONE
echo "MTP2 EXIT $fail $(date +%T)"
