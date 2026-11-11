#!/bin/bash
# Session step mtpt (docs/offload.md 15.6): mtp2's TTFT left +0.10..0.13 s of B - A outside the prompt's calls in turns
# 2-5 (mtp1c +0.13..0.16); one traced run of each arm with chatmoe --trace's host phases (tools/qual/chatmoe.py
# _Phases: rollback, each prefill_layers and its end_prefill, the snapshot and MTP's held row's read, MTP's own prefill,
# a0, the device runs outside the calls) puts each turn's TTFT in its parts. Both arms from this tree (T: the stamps
# branch), so only MTP differs: A = plain decode from production's pool (O/QA, card2's checkpoint), B = --mtp from the
# MTP pool (QM, mtp1's checkpoint MDB). The 35B's five les1 turns of --max-new 32, as mtp2.
# Self-contained, ~10 min, on whatever bitstream the card has (EXPECT: its build, as selftest prints it): selftest,
# the device's configuration (refs.py cfg), both pools' keys against this tree's image (pack_pool.py info); then per
# arm its pool read into the page cache (~70 s at a switch), a warm-up run (the image's first build for this tree and
# configuration, one turn, untimed) and the traced run; selftest. Tokens: A's and B's the same in every turn (MTP's
# greedy loop is plain decode's bit for bit), and with REF (a five-turn chatmoe JSON on this configuration, e.g. mtp1's
# refs/chatmoe5-qwen35-35b.json on production's m4) each arm's equal to it. A build other than EXPECT, a failed
# selftest, a pool not this image's, a failed run or another token stops it (then the final selftest).
# On a tree whose tiled expert-major runs take the states' slot 0 only (before mtp-slot-fix: MTP's slot 1 raises
# at the first expert-major prompt after a loop that ended there), both arms run untiled (OTPU_TILE=0).
# Output: R/q{WA,AT,WB,BT}.{log,json}, R/q{AT,BT}.trace.json, and per turn A's and B's TTFT, the calls, each phase's
# seconds and the rest of the TTFT outside them, B - A; the last line MTPT EXIT <fails>.
# Stage (from a laptop; no host work while a session runs):
#   git archive <stamps sha> | ssh CARD_HOST 'D=~/otpu-build/les/mtpt/host; mkdir -p $D && tar -x -C $D'
#   ssh CARD_HOST 'echo <stamps sha, full> > ~/otpu-build/les/mtpt/host/COMMIT; mkdir -p ~/otpu-build/les/mtpt/runs'
# Run (no pipe), inside a held session (OTPU_LOCK_HELD set) directly, else under otpu-lock:
#   EXPECT=<build> setsid -f ~/.local/bin/otpu-lock --wait 600 -- bash ~/otpu-build/les/mtpt/host/tools/offload/\
#     sessions/mtpt.sh > ~/otpu-build/les/mtpt.launch.log 2>&1 < /dev/null
set -u
[ -n "${OTPU_LOCK_HELD:-}" ] || { echo "run under otpu-lock"; exit 1; }
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*' | grep -vE '^OTPU_LOCK_(HELD|PID|WAIT)$'); do unset $v; done
export OTPU_IMG_CACHE_GB=30 OTPU_IMG_CACHE_FREE_GB=6      # (the card host's, from its ~/.bashrc)
LS=${LS:-$HOME/otpu-build/les/mtpt}
M1=${M1:-$HOME/otpu-build/les/mtp1}               # (mtp1's MTP checkpoint and pool)
R=${R:-$LS/runs}
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh   # (T: this tree; O: card2; OTPU_QUIET claimed)
exec > >(tee -a $R/session.log) 2>&1
EXPECT=${EXPECT:?the card build as the selftest prints it, e.g. 039b34d5}
REF=${REF:-}                                    # (a five-turn reference on this configuration; empty: A = B only)
MAXNEW=${MAXNEW:-32}
MD=Qwen3.5-35B-A3B
MDB=${MDB:-$M1/$MD-mtp}                         # (B's checkpoint: card2's files linked, and mtp.safetensors)
QA=${QA:-pool-q35-mx.split.bin}                 # (in O, read only)
QM=${QM:-$M1/pool-q35-mx-mtp.split.bin}
DROPOTHER=${DROPOTHER:-$HOME/openTPU:$HOME/otpu-build:/data/otpu}
T1="What is the capital of France? Answer in one sentence."
T2="And what is it famous for? Two sentences."
T3="Which three museums in Paris should a first-time visitor see, and how long does each take?"
T4="Sum this up in one sentence: $(cat $T/tools/offload/sessions/wiki1.txt)"
T5="Name two emperors of that period."
fail=0
echo "mtpt start $(date +%T) T $T ($rev) O $O QM $QM MDB $MDB REF ${REF:-none} mem $(mem) GB; $(uptime)"
grep -q 'class _Phases' $T/tools/qual/chatmoe.py && grep -q 'def holds' $T/opentpu/llm/mtp.py \
  || { echo "STOP: T has no chatmoe phases or no MTP held rows"; exit 2; }
[ -f $O/$QA ] && [ -f $QM ] && [ -f $MDB/mtp.safetensors ] && [ -d $O/$MD ] && { [ -z "$REF" ] || [ -f "$REF" ]; } \
  || { echo "STOP: $O/$QA, $QM, $MDB/mtp.safetensors, $O/$MD or REF missing"; exit 2; }
st=$(timeout 600 python -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "$st" | grep -q "build $EXPECT" || { echo "STOP: the card's build is not $EXPECT"; exit 2; }
echo "$st" | grep -q "ALL PASS" || { echo "STOP: selftest"; exit 2; }
if grep -q "a tiled layer run takes the states' slot 0" $T/opentpu/llm/qwen3.py; then
  export OTPU_TILE=0      # (this tree's tiled expert-major runs take the states' slot 0 only: MTP's slot 1 raises)
  echo "  OTPU_TILE=0 for both arms: no tiled runs in MTP's slot 1 on this tree (mtp-slot-fix)"
fi
timeout 300 python tools/qual/refs.py cfg $R/cfg-dev.pkl > /dev/null 2>&1; echo "cfg exit $?"
[ -f $R/cfg-dev.pkl ] || { echo "STOP: no device configuration"; exit 2; }
python - $R/cfg-dev.pkl <<'PY'
import pickle, sys
c = pickle.load(open(sys.argv[1], "rb"))
print(f"  cfg: MCOLS {c.MCOLS}, S {c.S}, DRAM {c.DRAM_BYTES >> 20} MiB")
PY
for p in "$O/$MD $O/$QA" "$MDB $QM --mtp"; do
  set -- $p
  (python tools/offload/pack_pool.py $1 $2 info ${3:-}) 2>&1 | tail -3 > $R/info-$(basename $2).txt
  cat $R/info-$(basename $2).txt
  grep -q "the same key" $R/info-$(basename $2).txt && grep -qE "([0-9]+) of \1 experts packed" $R/info-$(basename $2).txt \
    || { echo "STOP: $2 is not whole or not this image's"; exit 2; }
done
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
res() {    # the percentage of a file in the page cache (util-linux fincore)
  local r=$(fincore -b -n -o RES "$1" 2>/dev/null | tr -d ' ')
  echo $(( ${r:-0} * 100 / $(stat -c %s "$1") ))
}
for x in WA AT WB BT; do
  case $x in ?A|A?) md=$O/$MD; pool=$O/$QA; extra="";; *) md=$MDB; pool=$QM; extra="--mtp";; esac
  case $x in W?) turns=(--turn "$T1"); mx=8;;
             *) turns=(--turn "$T1" --turn "$T2" --turn "$T3" --turn "$T4" --turn "$T5"); mx=$MAXNEW
                extra="$extra --trace $R/q$x.trace.json";; esac
  echo "=== q$x $(date +%T) ($extra)"
  case $x in W?) warm $md $pool || { echo "STOP: $pool not wholly in the page cache"; fail=1; break; };; esac
  (timeout 1200 python tools/qual/chatmoe.py $md --pool $pool --backend board --cfg $R/cfg-dev.pkl "${turns[@]}" \
     --max-new $mx $extra --out $R/q$x.json > $R/q$x.log 2>&1)
  rc=$?; echo "  exit $rc $(date +%T); after it $(res $pool)% of its pool in the page cache"
  grep -E "Error|Traceback" $R/q$x.log | tail -3
  [ $rc = 0 ] && [ -f $R/q$x.json ] || { echo "STOP: q$x"; fail=1; break; }
  grep -E "^turn |phases:" $R/q$x.log | cut -c1-400
  if [ -n "$REF" ] && [ $x != WA ] && [ $x != WB ]; then
    python - $R/q$x.json $REF <<'PY' || { echo "STOP: q$x differs from REF"; fail=1; break; }
import json, sys
d, ref = (json.load(open(f)) for f in sys.argv[1:])
ok = [t["tokens"] for t in d["turns"]] == [t["tokens"] for t in ref["turns"]]
print(f"  [{'PASS' if ok else 'FAIL'}] tokens {'=' if ok else '!='} REF's five turns")
sys.exit(0 if ok else 1)
PY
  fi
done
echo "=== final selftest $(date +%T)"
timeout 600 python -m opentpu.host.selftest 2>&1 | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
[ $fail = 0 ] && python - $R <<'PY' || fail=1
import json, sys                # each turn: A's and B's TTFT in phases (chatmoe _Phases), B - A
r = sys.argv[1]
PH = ("template", "rollback", "mtp prefill", "prefill_layers", "end_prefill", "mtp snapshot", "snapshot", "a0",
      "run")
HOLD = ("mtp prefill", "mtp snapshot")
js = {x: json.load(open(f"{r}/q{x}.json")) for x in ("AT", "BT")}
tr = {x: json.load(open(f"{r}/q{x}.trace.json"))["turns"] for x in ("AT", "BT")}
ok = [t["tokens"] for t in js["AT"]["turns"]] == [t["tokens"] for t in js["BT"]["turns"]]
print(f"  [{'PASS' if ok else 'FAIL'}] tokens: A's and B's the same in all {len(js['AT']['turns'])} turns")
def parts(t):
    tot, spans = {}, []
    for p, a, b, _ in t["phases"]:
        n, s = tot.get(p, (0, 0.0))
        tot[p] = (n + 1, s + b - a)
        if p not in HOLD:
            spans.append((a, b))
    cov, end = 0.0, 0.0
    for a, b in sorted(spans):
        cov += max(0.0, b - max(a, end))
        end = max(end, b)
    tot["rest"] = (1, max(0.0, (t["ttft_s"] or 0.0) - cov))
    return tot
for i, (ta, tb) in enumerate(zip(tr["AT"], tr["BT"])):
    pa, pb = parts(ta), parts(tb)
    ca, cb = (" + ".join(f"{c['rows']}{'E' if c['em'] else 'L'} {c['s']}" for c in js[x]["turns"][i]["calls"])
              for x in ("AT", "BT"))
    print(f"  turn {i + 1}: TTFT A {ta['ttft_s']:.3f} B {tb['ttft_s']:.3f} ({tb['ttft_s'] - ta['ttft_s']:+.3f}) s; "
          f"calls A {ca}; B {cb}")
    for p in PH + ("rest",):
        if p in pa or p in pb:
            (na, sa), (nb, sb) = pa.get(p, (0, 0.0)), pb.get(p, (0, 0.0))
            print(f"      {p:15s} A {sa:7.3f} ({na}) B {sb:7.3f} ({nb})  B - A {sb - sa:+.3f}")
    ra = [q for q in ta["phases"] if q[0] == "run"]
    rb = [q for q in tb["phases"] if q[0] == "run"]
    print(f"      runs outside the calls: A {len(ra)} ({sum(q[3] or 0 for q in ra)} cycles), "
          f"B {len(rb)} ({sum(q[3] or 0 for q in rb)} cycles)")
sys.exit(0 if ok else 1)
PY
date > $R/DONE
echo "MTPT EXIT $fail $(date +%T)"
