#!/bin/bash
# Session emprol (docs/tiled_prefill.md 14.3): em-prologue's card A/B. A tile's MoE prologue takes its rows at once and
# posts its need lines between the shared expert's passes (each post waits for the host to serve the one before: the
# co-simulation serves at once, so what the host's poll costs is the card's to show), and a tiled MoE image's DeltaNet
# rows run a tile's at a time where MCOLS takes them. Co-simulated (T = 8, position 256): MiniCPM-V's mixer -6.3% a row
# on m4 (-1.15 s of the 488-row image prompt), -13.3% on m8 (-2.25 s); qwen35-35b's -2.2% / -7.1%.
# Arms: A = the release's host (A, default the r2.1 deploy's host: batch3), B = em-prologue's (B), each its own
# opentpu on PYTHONPATH and its own tools/qual/chatmoe.py, read only (no bytecode written). Card: production as
# loaded (EXPECT, MAP), the configuration of REFCFG. Models and prompts (chatmoe.py, --pool-calls):
#   mcpv  MiniCPM-V 4.7: doc.png at 9 slices, one turn of 32 tokens (the image request: 488 prompt rows)
#   q35   qwen35-35b: a ~260-token text prompt (the em-tile references' q35turn), 16 tokens
# Per model: the pool into the page cache (every other file of 100 MB or more out of it), a warm-up run in each arm
# (WA, WB: each tree's images and programs built and cached, discarded; WA's tokens the reference), then ORDER (default
# A1 B1 B2 A2; ORDER="A1 B1 B2 A2 B3 A3 A4 B4" for more pairs). Every run's tokens must equal WA's (the same bits).
# Summary per model: each run's TTFT, prefill and device seconds, B - A per pair and of the means, the expert-major
# call's runs and its host time. The rule : B no worse than A on either model, token-exact.
# A build other than EXPECT, a failed selftest, another configuration, a missing file, a failed run or a token
# mismatch stops it (then the final selftest). About 45 min (the warm-ups ~10 min each model with first builds).
# Stage (from a laptop): B's tree (git archive of em-prologue's commit) to ~/otpu-build/emprol/host with COMMIT;
#   A defaults to the r2.1 deploy's host (its COMMIT is printed). Run (no pipe), e.g.:
#   EXPECT=039b34d5 A=$HOME/otpu-build/production/host \
#   REFCFG=$HOME/otpu-build/production/refcache/configs/cfg_E3.pkl setsid -f ~/.local/bin/otpu-lock --wait 3600 -- \
#     bash ~/otpu-build/emprol/host/tools/offload/sessions/emprol.sh > ~/otpu-build/emprol/session.log 2>&1 < /dev/null
# (No apostrophe inside a ${...:?...} / ${...:-...} word: bash takes it as an open quote, to the next one.)
set -u
[ -n "${OTPU_LOCK_HELD:-}" ] || { echo "run under otpu-lock"; exit 1; }
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*' | grep -vE '^OTPU_LOCK_(HELD|PID|WAIT)$'); do unset $v; done
export OTPU_IMG_CACHE_GB=30 OTPU_IMG_CACHE_FREE_GB=6      # (the card host's, from its ~/.bashrc)
E=${E:-$HOME/otpu-build/emprol}
A=${A:-$HOME/otpu-build/deploy_rel21_039b34d5/host}
B=${B:-$E/host}
EXPECT=${EXPECT:?the card build to expect, e.g. 039b34d5}
MAP=${MAP:-register map 5}
REFCFG=${REFCFG:?the configuration the arms run on, e.g. the deploy refcache/configs/cfg_E3.pkl}
ORDER=${ORDER:-A1 B1 B2 A2}
P=$HOME/otpu-venv/bin/python
MO=/data/otpu/minicpmv; C2=$HOME/otpu-build/offload/card2
PYLIB=${PYLIB:-$MO/pylib}                       # (MiniCPM-V's tower: torchvision, pillow, transformers)
R=$E/run-$(date +%m%d-%H%M); mkdir -p $R
exec > >(tee -a $R/session.log) 2>&1
Q=$HOME/otpu-build/QUIET          # the whole session is quiet: no prebuilds or OOC jobs on the host
q=$(cat $Q 2>/dev/null); if [ -z "$q" ] || ! kill -0 "$q" 2>/dev/null; then echo $$ > $Q; trap '[ "$(cat $Q 2>/dev/null)" = $$ ] && rm -f $Q' EXIT; fi
mem() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
declare -A MD=([mcpv]="$MO/MiniCPM-V-4.7-35B-A3B $MO/pool-mcpv35-mx.split.bin"
               [q35]="$C2/Qwen3.5-35B-A3B $C2/pool-q35-mx.split.bin")
cat > $R/q35turn.txt <<'TXT'
Summarize the following text in two sentences.

The printing press with movable type changed how knowledge travelled across Europe. Before it, books were copied by hand, a slow and costly process that kept most texts in the libraries of monasteries, universities and wealthy families. A single scribe might spend months on one volume, and every copy could introduce new errors. Around 1450, Johannes Gutenberg combined several existing technologies into a practical system: metal type cast in a hand mould, an oil-based ink that stuck to metal, and a screw press adapted from those used to make wine and paper. Within fifty years, presses were running in more than two hundred cities, and millions of books had been printed. Prices fell, literacy spread beyond the clergy and the nobility, and scholars in distant cities could work from identical texts, which made it easier to compare results and correct mistakes. Pamphlets and broadsheets carried news and arguments quickly, and the Reformation, the scientific revolution and the growth of national languages all drew on this new capacity to reproduce words cheaply and accurately. At the same time, authorities worried about what could now be spread so easily, and many states introduced licences and censorship to control the presses.
TXT
declare -A ARGS=([mcpv]="--image 1:$MO/img/doc.png --turn 'What does this document say? Two sentences.' --max-new 32"
                 [q35]="--turn \"\$(cat $R/q35turn.txt)\" --max-new 16")
fail=0
echo "emprol start $(date +%T): A $A ($(cat $A/COMMIT 2>/dev/null)), B $B ($(cat $B/COMMIT 2>/dev/null)); mem $(mem) GB; $(uptime)"
grep -q "def need_rows" $B/opentpu/llm/moe.py && grep -q "each=None" $B/opentpu/kernels/mlp.py \
  || { echo "STOP: B is not em-prologue's tree (moe.need_rows, swiglu_down(each=))"; exit 2; }
! grep -q "def need_rows" $A/opentpu/llm/moe.py || { echo "STOP: A already has em-prologue"; exit 2; }
for m in mcpv q35; do
  read -r md pool <<< "${MD[$m]}"
  [ -f $md/config.json ] && [ -f $pool ] && [ -f $pool.key ] || { echo "STOP: $md or $pool missing"; exit 2; }
done
[ -f $MO/img/doc.png ] && [ -f $REFCFG ] || { echo "STOP: $MO/img/doc.png or $REFCFG missing"; exit 2; }
for t in $A $B; do
  echo "  $t: $(cd /tmp && PYTHONPATH=$t PYTHONDONTWRITEBYTECODE=1 $P -c 'import opentpu, opentpu.progcache as p; print(opentpu.__file__, p.source())')"
done
st=$(cd /tmp && PYTHONPATH=$A timeout 600 $P -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "$st" | grep -q "build $EXPECT" || { echo "STOP: the card's build is not $EXPECT"; exit 2; }
echo "$st" | grep -q "$MAP" || { echo "STOP: not $MAP"; exit 2; }
echo "$st" | grep -q "ALL PASS" || { echo "STOP: selftest"; exit 2; }
(cd $A && PYTHONPATH=$A PYTHONDONTWRITEBYTECODE=1 timeout 300 $P tools/qual/refs.py cfg $R/cfg-dev.pkl > /dev/null 2>&1)
$P - $R/cfg-dev.pkl $REFCFG <<'PY' || { echo "STOP: the device configuration is not REFCFG's"; exit 2; }
import dataclasses, pickle, sys
a, b = (pickle.load(open(f, "rb")) for f in sys.argv[1:])
d = {f.name: (getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
     if getattr(a, f.name) != getattr(b, f.name)}
print("  cfg: REFCFG's" if not d else f"  cfg DIFFERS (device, REFCFG): {d}")
sys.exit(1 if d else 0)
PY
warm() {      # $1 checkpoint, $2 pool: every other file of 100 MB or more out of the page cache, the pool into it
  $P - $1 $2 $MO:$C2:$HOME/openTPU/models:$HOME/otpu-build/offload <<'PY'
import os, sys
keep = {os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])}
ours = tuple(os.path.realpath(os.path.expanduser(d)) + os.sep
             for d in ("~/openTPU", "~/otpu-build", "/data/otpu"))
gb = n = 0
seen = set()
for d in sys.argv[3].split(":"):
    for root, _, files in os.walk(d):
        for f in files:
            r = os.path.realpath(os.path.join(root, f))
            if r in seen or not r.startswith(ours) or not os.path.isfile(r) or \
                    os.path.getsize(r) < 100 << 20 or r in keep or os.path.dirname(r) in keep:
                seen.add(r)
                continue
            seen.add(r)
            fd = os.open(r, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            gb, n = gb + os.path.getsize(r) / 1e9, n + 1
print(f"  dropped from the page cache: {n} other files of {gb:.1f} GB")
PY
  local t0=$SECONDS
  dd if=$2 of=/dev/null bs=16M status=none
  echo "  (pool read in $((SECONDS - t0)) s, mem $(mem) GB)"
}
run() {       # $1 model, $2 run name (WA, WB, A1, B2, ...): its arm's tree and chatmoe.py; WA's tokens the reference
  local m=$1 n=$2 t; read -r md pool <<< "${MD[$m]}"
  case $n in WA|A*) t=$A ;; *) t=$B ;; esac
  case $n in WA) chk="" ;; *) chk="--check $R/$m-WA.json" ;; esac
  eval "(cd $t && env PYTHONPATH=$t:$PYLIB PYTHONDONTWRITEBYTECODE=1 timeout 3600 $P tools/qual/chatmoe.py $md \
    --pool $pool --backend board --cfg $R/cfg-dev.pkl --pool-calls $chk --out $R/$m-$n.json ${ARGS[$m]})" \
    > $R/$m-$n.log 2>&1
  local rc=$?
  echo "  $m $n ($t) exit $rc $(date +%T); $(grep -E "\[(PASS|FAIL)\]|Error|Traceback" $R/$m-$n.log | tail -2 | cut -c1-200)"
  [ $rc = 0 ] && { [ $n = WA ] || grep -q "\[PASS\]" $R/$m-$n.log; } || { echo "STOP: $m $n"; fail=1; }
}
for m in mcpv q35; do
  [ $fail = 0 ] || break
  read -r md pool <<< "${MD[$m]}"
  echo "=== $m $(date +%T)"
  warm $md $pool
  for n in WA WB $ORDER; do
    [ $fail = 0 ] || break
    run $m $n
  done
done
echo "=== summary $(date +%T)"
$P - $R "$ORDER" <<'PY'
import json, sys
from pathlib import Path
R, order = Path(sys.argv[1]), sys.argv[2].split()
for m in ("mcpv", "q35"):
    rows = {}
    for n in order:
        f = R / f"{m}-{n}.json"
        if not f.exists():
            continue
        t = json.load(open(f))["turns"][0]
        dev = t["prompt_tokens"] / t["prefill_dev_tok_s"] if t.get("prefill_dev_tok_s") else None
        rows[n] = (t["ttft_s"], t["prefill_s"], dev, t.get("calls"))
        print(f"  {m} {n}: TTFT {t['ttft_s']} s, prefill {t['prefill_s']} s ({t['prompt_tokens']} tokens), device "
              f"{dev and round(dev, 2)} s, decode {t['decode_tok_s']} tok/s")
    pairs = sorted({n[1:] for n in rows})
    d = [(rows["B" + k][0] - rows["A" + k][0], rows["B" + k][1] - rows["A" + k][1])
         for k in pairs if "A" + k in rows and "B" + k in rows]
    for k, (dt, dp) in zip(pairs, d):
        print(f"  {m} pair {k}: B - A TTFT {dt:+.2f} s, prefill {dp:+.2f} s")
    if d:
        print(f"  {m} mean B - A: TTFT {sum(x for x, _ in d) / len(d):+.2f} s, prefill "
              f"{sum(y for _, y in d) / len(d):+.2f} s over {len(d)} pairs "
              f"({'B no worse' if all(x <= 0.05 for x, _ in d) else 'B WORSE in a pair'})")
PY
st=$(cd /tmp && PYTHONPATH=$A timeout 600 $P -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "ALL|stopped|FAIL" | tail -3
echo "EMPROL SESSION END $(date +%T) fail=$fail"
