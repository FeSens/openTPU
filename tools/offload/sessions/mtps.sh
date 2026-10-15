#!/bin/bash
# Session mtps (docs/offload.md 15.6): otpu-serve with and without --mtp on the streamed 35B, for MTP by default in the
# server. Both arms from this tree (T: serve-mtp), so only --mtp differs: A = otpu-serve from production's pool (O/QA,
# card2's checkpoint), B = otpu-serve --mtp from the MTP pool (QM, mtp1's checkpoint MDB). Each run: the arm's pool read
# into the page cache (~130 s at a switch of arm: the two pools do not fit the page cache together), the server up
# (/health 200), tools/serve_bench.py: its check part (each prompt alone, then all at once, a repeat: the same replies),
# throughput at 1, 2 and 4 clients greedy and 4 sampled (the model's sampling, seeded), a three-turn conversation and
# four arrivals one every 1.5 s, --max-tokens MAXTOK; then the server's /v1/stats (the context switches: restores,
# restore_s, save_s; MTP's iterations and accepted drafts) and its stop. Order: a warm-up of each arm (WA, WB: the check
# part alone; this tree's images' first builds), then ORDER (A1 B1 B2 A2).
# Checks: every serve_bench check part PASS; B's check replies A's (greedy: MTP's loop is plain decode's bit for bit).
# Default-on rule, B against A (the means of the arm's runs): aggregate tok/s >= A's at 1, 2 and 4 clients and >= 1.05x
# A's at 4; the TTFT medians (throughput levels, conversation turns, arrivals) at most 0.3 s more; the arrivals' gap p99
# at most 1.2x A's; and the checks. The summary prints each measure per run, B - A or B / A, the restore ms a context
# switch (restore_s / restores) of each arm, and the rule's verdict (MTPS DEFAULT-ON: YES / NO, and why).
# ADDON=1: the same on the dense Qwen3.5-4B (its own mtp.* layer; A plain, B --mtp; no pools), W4A W4B A41 B41 after
# the 35B, reported apart (not in the rule).
# On a tree whose tiled expert-major runs take the states' slot 0 only (before mtp-slot-fix), both arms run untiled
# (OTPU_TILE=0).
# A build other than EXPECT, a failed selftest, a server that does not come up, a failed check or a run past its timeout
# stops it (then the final selftest). About 70 min at MAXTOK 128 (6 runs of the 35B, ~8-10 min each with the pool reads;
# the add-on ~20 min more).
# Stage (from the Mac; no host work while a session runs):
#   git archive <serve-mtp sha> | ssh opentpu 'D=~/otpu-build/les/mtps/host; mkdir -p $D && tar -x -C $D'
#   ssh opentpu 'echo <serve-mtp sha, full> > ~/otpu-build/les/mtps/host/COMMIT; mkdir -p ~/otpu-build/les/mtps/runs'
# Run (no pipe), inside a held session (OTPU_LOCK_HELD set) directly, else under otpu-lock:
#   EXPECT=<build> setsid -f ~/.local/bin/otpu-lock --wait 600 -- bash ~/otpu-build/les/mtps/host/tools/offload/\
#     sessions/mtps.sh > ~/otpu-build/les/mtps.launch.log 2>&1 < /dev/null
set -u
[ -n "${OTPU_LOCK_HELD:-}" ] || { echo "run under otpu-lock"; exit 1; }
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*' | grep -vE '^OTPU_LOCK_(HELD|PID|WAIT)$'); do unset $v; done
export OTPU_IMG_CACHE_GB=30 OTPU_IMG_CACHE_FREE_GB=6      # (the card host's, from its ~/.bashrc)
LS=${LS:-$HOME/otpu-build/les/mtps}
M1=${M1:-$HOME/otpu-build/les/mtp1}               # (mtp1's MTP checkpoint and pool)
R=${R:-$LS/runs}
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh   # (T: this tree; O: card2; OTPU_QUIET claimed)
exec > >(tee -a $R/session.log) 2>&1
EXPECT=${EXPECT:?the card build as the selftest prints it, e.g. 039b34d5}
ORDER=${ORDER:-A1 B1 B2 A2}                     # (after the warm-ups WA, WB)
MAXTOK=${MAXTOK:-128}
ADDON=${ADDON:-0}
PORT=${PORT:-8021}
MD=Qwen3.5-35B-A3B
MDB=${MDB:-$M1/$MD-mtp}                         # (B's checkpoint: card2's files linked, and mtp.safetensors)
QA=${QA:-pool-q35-mx.split.bin}                 # (in O, read only)
QM=${QM:-$M1/pool-q35-mx-mtp.split.bin}
M4=${M4:-qwen35-4b}                             # (the add-on's model: a name or a checkpoint directory)
DROPOTHER=${DROPOTHER:-$HOME/openTPU:$HOME/otpu-build:/data/otpu}
fail=0
echo "mtps start $(date +%T) T $T ($rev) O $O QM $QM MDB $MDB ORDER $ORDER MAXTOK $MAXTOK ADDON $ADDON mem $(mem) GB;" \
  "$(uptime)"
grep -q '"--mtp"' $T/opentpu/serve/api.py && grep -q 'def _decode_mtp' $T/opentpu/serve/scheduler.py \
  || { echo "STOP: T has no otpu-serve --mtp"; exit 2; }
[ -f $O/$QA ] && [ -f $QM ] && [ -f $MDB/mtp.safetensors ] && [ -d $O/$MD ] \
  || { echo "STOP: $O/$QA, $QM, $MDB/mtp.safetensors or $O/$MD missing"; exit 2; }
st=$(timeout 600 python -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "$st" | grep -q "build $EXPECT" || { echo "STOP: the card's build is not $EXPECT"; exit 2; }
echo "$st" | grep -q "ALL PASS" || { echo "STOP: selftest"; exit 2; }
if grep -q "a tiled layer run takes the states' slot 0" $T/opentpu/llm/qwen3.py; then
  export OTPU_TILE=0      # (this tree's tiled expert-major runs take the states' slot 0 only: MTP's slot 1 raises)
  echo "  OTPU_TILE=0 for both arms: no tiled runs in MTP's slot 1 on this tree (mtp-slot-fix)"
fi
for p in "$O/$MD $O/$QA" "$MDB $QM --mtp"; do
  set -- $p
  (python tools/offload/pack_pool.py $1 $2 info ${3:-}) 2>&1 | tail -3 > $R/info-$(basename $2).txt
  cat $R/info-$(basename $2).txt
  grep -q "the same key" $R/info-$(basename $2).txt && grep -qE "([0-9]+) of \1 experts packed" $R/info-$(basename $2).txt \
    || { echo "STOP: $2 is not whole or not this image's"; exit 2; }
done
tree() { local x; for x in "$@"; do echo $x; tree $(pgrep -P $x); done; }
live() { local x; for x in "$@"; do kill -0 $x 2> /dev/null && echo $x; done; }
stop_tree() {   # hostr21.sh's: TERM to the server's process group (timeout's), 30 s, then KILL
  local p=$1 g w all left
  g=$(ps -o pgid= -p $p | tr -d ' ')
  [ "$g" = "$p" ] || g=""
  all=$(tree $p | tr '\n' ' ')
  if [ -n "$g" ]; then kill -- -$g 2> /dev/null; else kill $all 2> /dev/null; fi
  for w in $(seq 1 30); do
    left=$( { [ -n "$g" ] && pgrep -g $g; live $all; } | sort -u | tr '\n' ' ')
    [ -z "${left// }" ] && break; sleep 1; done
  if [ -n "${left// }" ]; then echo "  stop_tree: $left still alive after 30 s: KILL"
    [ -n "$g" ] && kill -9 -- -$g 2> /dev/null; kill -9 $left 2> /dev/null; sleep 2; fi
  wait $p 2> /dev/null
}
xdma_free() {
  local w u
  for w in $(seq 1 60); do u=$(lsof -t /dev/xdma0_* /dev/otpu0_* 2> /dev/null | tr '\n' ' '); [ -z "$u" ] && return 0; sleep 1; done
  echo "  the card's nodes still open by pids $u: FAIL"; return 1
}
warm() {   # checkpoint, pool: every other big file out of the page cache, the pool read into it (mtp2.sh's)
  python - $1 $2 $DROPOTHER <<'PY'
import os, sys
keep = {os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])}
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
  for k in 1 2; do
    dd if=$2 of=/dev/null bs=16M status=none
    [ "$(res $2)" -ge 99 ] && break
  done
  echo "  pool read in $((SECONDS - t0)) s, $(res $2)% of it in the page cache, mem $(mem) GB"
  [ "$(res $2)" -ge 99 ]
}
res() {
  local r=$(fincore -b -n -o RES "$1" 2>/dev/null | tr -d ' ')
  echo $(( ${r:-0} * 100 / $(stat -c %s "$1") ))
}
one() {   # run name, model, pool or "", extra server flags, serve_bench flags
  local x=$1 md=$2 pool=$3 flags=$4 bench=$5 sp rb
  echo "=== $x $(date +%T) ($md $flags)"
  if [ -n "$pool" ] && [ "$(res $pool)" -lt 99 ]; then warm $md $pool || { echo "STOP: $pool not in the page cache"; return 1; }; fi
  rm -rf $R/kv-$x
  (cd /tmp && exec env PYTHONPATH=$T timeout 3000 python -m opentpu.serve.api --model $md --backend board \
     ${pool:+--pool $pool} $flags --port $PORT --kv-ram-gb 2 --kv-dir $R/kv-$x) > $R/$x.serve.log 2>&1 &
  sp=$!
  (cd $T && timeout 2700 python tools/serve_bench.py --port $PORT --max-tokens $MAXTOK $bench --out $R/$x.json \
     > $R/$x.bench.log 2>&1)
  rb=$?
  echo "  serve_bench exit $rb $(date +%T)"; grep -E "check:|throughput:|^turn |arrivals:|did not come up" $R/$x.bench.log \
    | cut -c1-260
  stop_tree $sp; xdma_free || rb=1
  grep -E "Error|Traceback" $R/$x.serve.log | tail -3
  [ $rb = 0 ] && grep -q "check: PASS" $R/$x.bench.log
}
WARM="--skip throughput,conversation,users,arrivals --check-prompts 2 --check-tokens 16"
FULL="--levels 1,2,4 --sampled-levels 4 --turns 3 --arrivals 4 --users 0"
for x in WA WB $ORDER; do
  case $x in A*|WA) md=$O/$MD; pool=$O/$QA; flags="";; *) md=$MDB; pool=$QM; flags="--mtp";; esac
  case $x in W?) bench=$WARM;; *) bench=$FULL;; esac
  one q$x $md $pool "$flags" "$bench" || { echo "STOP: q$x"; fail=1; break; }
done
if [ $fail = 0 ] && [ "$ADDON" = 1 ]; then
  for x in W4A W4B A41 B41; do
    case $x in *A*) flags="";; *) flags="--mtp";; esac
    case $x in W*) bench=$WARM;; *) bench=$FULL;; esac
    one q$x $M4 "" "$flags" "$bench" || { echo "STOP: q$x (the add-on)"; fail=1; break; }
  done
fi
echo "=== final selftest $(date +%T)"
timeout 600 python -m opentpu.host.selftest 2>&1 | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
[ $fail = 0 ] && python - $R "$ORDER" $ADDON <<'PY' || fail=1
import json, os, statistics as S, sys
r, order, addon = sys.argv[1], sys.argv[2].split(), sys.argv[3] == "1"
def load(x):
    return json.load(open(f"{r}/q{x}.json"))
def measures(d):
    m = {}
    for t in d.get("throughput", []):
        k = f"{t['clients']}{'s' if t['sampled'] else ''}"
        m[f"tok/s {k}"] = t["aggregate_tok_s"]
        m[f"TTFT {k}"] = t["ttft_s"]["median"]
    conv = d.get("conversation", [])
    if conv:
        m["TTFT conv"] = S.median(c["ttft_s"] for c in conv)
    a = d.get("arrivals")
    if a:
        m["TTFT arrivals"] = S.median(a["ttft_s"])
        m["gap p99 ms"] = a["gap_ms"]["p99"]
    st = d.get("stats", {})
    m["restore ms"] = round(1e3 * st["restore_s"] / st["restores"], 1) if st.get("restores") else None
    m["save ms"] = round(1e3 * st["save_s"] / st["saves"], 1) if st.get("saves") else None
    if st.get("mtp_iters"):
        m["acceptance"] = round(st["mtp_accepted"] / st["mtp_iters"], 3)
    return m
def pair(As, Bs, label):
    a, b = [load(x) for x in As], [load(x) for x in Bs]
    ok = all(x["check"]["pass"] for x in a + b)
    same = all(x["check"]["solo"] == a[0]["check"]["solo"] for x in a + b)
    print(f"  [{'PASS' if ok and same else 'FAIL'}] {label}: check parts {'PASS' if ok else 'FAIL'}, B's replies "
          f"{'=' if same else '!='} A's")
    ma, mb = ([measures(x) for x in xs] for xs in (a, b))
    keys = [k for k in ma[0] if all(k in m and m[k] is not None for m in ma + mb)]
    mean = {c: {k: S.mean(m[k] for m in ms) for k in keys} for c, ms in (("A", ma), ("B", mb))}
    for k in keys:
        rel = f"B / A {mean['B'][k] / mean['A'][k]:.3f}" if mean["A"][k] else ""
        print(f"    {k:14s} " + "  ".join(f"{x} {m[k]}" for x, m in zip(As + Bs, ma + mb)) +
              f"  | A {mean['A'][k]:.3f} B {mean['B'][k]:.3f}  B - A {mean['B'][k] - mean['A'][k]:+.3f}  {rel}")
    return ok and same, mean
ok, mean = pair([x for x in order if x[0] == "A"], [x for x in order if x[0] == "B"], "35B")
A, B = mean["A"], mean["B"]
why = []
for c in ("1", "2", "4"):
    k = f"tok/s {c}"
    if k in A and B[k] < A[k]:
        why.append(f"{k} {B[k]:.2f} < A's {A[k]:.2f}")
if "tok/s 4" in A and B["tok/s 4"] < 1.05 * A["tok/s 4"]:
    why.append(f"tok/s 4 {B['tok/s 4']:.2f} under 1.05x A's {A['tok/s 4']:.2f}")
for k in [k for k in A if k.startswith("TTFT")]:
    if B[k] > A[k] + 0.3:
        why.append(f"{k} {B[k]:.3f} s, A's + {B[k] - A[k]:.3f}")
if "gap p99 ms" in A and B["gap p99 ms"] > 1.2 * A["gap p99 ms"]:
    why.append(f"gap p99 {B['gap p99 ms']:.1f} ms over 1.2x A's {A['gap p99 ms']:.1f}")
if not ok:
    why.append("a check failed")
print(f"MTPS DEFAULT-ON: {'YES' if not why else 'NO'}" + ("" if not why else " (" + "; ".join(why) + ")"))
if addon and all(os.path.exists(f"{r}/q{x}.json") for x in ("A41", "B41")):
    ok4, _ = pair(["A41"], ["B41"], "4B add-on (not in the rule)")
    ok = ok and ok4
sys.exit(0 if ok else 1)
PY
date > $R/DONE
echo "MTPS EXIT $fail $(date +%T)"
