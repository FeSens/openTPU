#!/bin/bash
# Session mcpv3 (docs/minicpmv.md 6): MiniCPM-V 4.7 (35B-A3B) on the card with this tree's defaults:
# the staged vision, the tower's wait entries computed ahead (opentpu/host/vision_warm.py) and the
# faster wait analysis (waits._MemoPass). Also the sliced reference's first run, and what the pool's
# disk costs (/data is a 7200 rpm disk). Runs (RUNS, default "warmup sliced thumb text tower pooldisk"):
#   warmup    chatmoe.py's text turns, unmeasured: the image's first build for this tree (its keys
#             are new: ~100-200 s, ~28 GB at its peak, which takes the pool out of the page cache);
#             the engine's vision_warm process starts
#   sliced    doc.png at 9 slices (7 crops of 1,008 and 1,024 patches), one turn of 32 tokens,
#             token-exact against its reference; its vision_s against mcpv2's ~41 s plus ~23 s for
#             a patch count's first crop
#   thumb     cats.jpg at max-slices 1, two turns, token-exact against thumb's reference (staged,
#             the default now)
#   text      chatmoe.py's two text turns, token-exact
#   tower     tools/minicpmv_tower.py time on cats.jpg (one crop) and doc.png: each crop's start
#             seconds (a first crop's ~23 s of wait entries computed at its start, before
#             vision_warm), and the odd-stride residual's gain (vit_out_mlp, docs/minicpmv.md 6):
#             out, out2 and last's device time a run against the co-simulation's expected
#             (3.27 ms a 16-row group; mcpv2's 4.57) and each crop's against mcpv2's ~5.77 s
#   pooldisk  tools/offload/pool_disk.py: one warm process, the same text turn with the pool in the
#             page cache and dropped from it (ABBA): decode tok/s and a miss's read time from the
#             disk against the page cache
# Each measured run follows the reset of mcpv2: every other file of 100 MB or more under O,
# ~/openTPU/models and ~/otpu-build/offload leaves the page cache and the pool is read into it.
# Card: production deploy_r2m4_b10245f5 as loaded (EXPECT b10245f5, register map 5); its
# qual/cfg.pkl. O (/data/otpu/minicpmv): the slim checkpoint, the pool (its 446 vision entries
# packed by mcpv2), img/, refs/ (minicpmv-text/thumb/sliced.json), pylib/.
# POOL_CALLS=1: chatmoe.py --pool-calls (the pool's page cache around each call too).
# One otpu-lock: selftest (the configuration must be production's), the runs, selftest. A build
# other than EXPECT, a failed selftest, another configuration or a missing file stops it; a failed
# run or a token mismatch is reported and the next run goes on. The last line: MCPV3 PASS or
# MCPV3 FAIL. About 25 min.
# Stage (from a laptop, before the session; no host work while a session runs):
#   git archive <sha> | ssh CARD_HOST 'D=~/otpu-build/minicpmv/mcpv3/host; mkdir -p $D && tar -x -C $D'
#   ssh CARD_HOST 'echo <sha> > ~/otpu-build/minicpmv/mcpv3/host/COMMIT'
# Run (no pipe): setsid nohup ~/.local/bin/otpu-lock --wait 600 -- bash \
#   ~/otpu-build/minicpmv/mcpv3/host/tools/offload/sessions/mcpv3.sh > ~/otpu-build/minicpmv/mcpv3.launch.log 2>&1 \
#   < /dev/null &
set -u
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*' | grep -vE '^OTPU_LOCK_(HELD|PID|WAIT)$'); do unset $v; done
export OTPU_IMG_CACHE_GB=30 OTPU_IMG_CACHE_FREE_GB=6      # (the card host's, from its ~/.bashrc)
SESSION=${SESSION:-mcpv3}
O=${O:-/data/otpu/minicpmv}
RF=${RF:-$O/refs}
PD=${PD:-$HOME/otpu-build/deploy_r2m4_b10245f5}        # (production: qual/cfg.pkl)
REFCFG=${REFCFG:-$PD/qual/cfg.pkl}
PYLIB=${PYLIB:-$O/pylib}
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"       # (T: this tree; R = O/SESSION; OTPU_QUIET claimed)
export PYTHONPATH=$T:$PYLIB
exec > >(tee -a $R/session.log) 2>&1
EXPECT=${EXPECT:-b10245f5}
MAP=${MAP:-register map 5}
MD=MiniCPM-V-4.7-35B-A3B
POOL=pool-mcpv35-mx.split.bin
RUNS=${RUNS:-warmup sliced thumb text tower pooldisk}
THUMB="--image 1:$O/img/cats.jpg --max-slices 1 --turn 'Describe this image in one sentence.' --turn 'What colors stand out? One sentence.'"
declare -A RUN=(        # chatmoe.py's options
  [warmup]=""
  [text]=""
  [sliced]="--image 1:$O/img/doc.png --turn 'What does this document say? Two sentences.' --max-new 32"
  [thumb]="$THUMB")
declare -A REFOF=([warmup]=text [text]=text [sliced]=sliced [thumb]=thumb)
fail=0
echo "mcpv3 start $(date +%T) T $T ($rev) O $O RF $RF mem $(mem) GB, /data $(df -BG --output=avail /data | tail -1 | tr -d ' ') free; $(uptime)"
[ -f $T/opentpu/host/vision_warm.py ] && grep -q '_MemoPass' $T/opentpu/waits.py \
  && grep -q 'vision_staged: bool = True' $T/opentpu/llm/minicpmv.py \
  && grep -q 'odd: bool = True' $T/opentpu/kernels/vit.py \
  || { echo "STOP: T is not the minicpmv tree with vision_warm, waits._MemoPass, staged by default and the odd-stride residual"; exit 2; }
python -c "import torchvision, PIL, transformers; print('  python:', torchvision.__version__, PIL.__version__,
  transformers.__version__)" || { echo "STOP: no torchvision / pillow / transformers under PYLIB $PYLIB"; exit 2; }
[ -f $O/$MD/config.json ] && [ -f $O/$POOL ] && [ -f $O/$POOL.key ] && [ -f $O/img/cats.jpg ] && \
  [ -f $O/img/doc.png ] && [ -f $REFCFG ] || { echo "STOP: $O/$MD, $O/$POOL, its key, img/ or $REFCFG missing"; exit 2; }
grep -q '"vision"' $O/$POOL.key || { echo "STOP: the pool's key has no vision entries (mcpv2's staged run packs them)"; exit 2; }
for m in $RUNS; do
  [ -n "${REFOF[$m]:-}" ] || continue
  [ -f $RF/minicpmv-${REFOF[$m]}.json ] || { echo "STOP: no reference $RF/minicpmv-${REFOF[$m]}.json"; exit 2; }
done
st=$(timeout 600 python -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "$st" | grep -q "build $EXPECT" || { echo "STOP: the card's build is not $EXPECT"; exit 2; }
echo "$st" | grep -q "$MAP" || { echo "STOP: not $MAP"; exit 2; }
echo "$st" | grep -q "ALL PASS" || { echo "STOP: selftest"; exit 2; }
timeout 300 python tools/qual/refs.py cfg $R/cfg-dev.pkl > /dev/null 2>&1; echo "cfg exit $?"
python - $R/cfg-dev.pkl $REFCFG <<'PY' || { echo "STOP: the device configuration is not the references'"; exit 2; }
import dataclasses, pickle, sys
a, b = (pickle.load(open(f, "rb")) for f in sys.argv[1:])
d = {f.name: (getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
     if getattr(a, f.name) != getattr(b, f.name)}
print("  cfg: the references' configuration" if not d else f"  cfg DIFFERS (device, ref): {d}")
sys.exit(1 if d else 0)
PY
warm() {      # every other file of 100 MB or more out of the page cache, then the pool into it
  python - $O/$MD $O/$POOL $O:$HOME/openTPU/models:$HOME/otpu-build/offload <<'PY'
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
  dd if=$O/$POOL of=/dev/null bs=16M status=none      # (every block read)
  echo "  (pool read in $((SECONDS - t0)) s, mem $(mem) GB, cached $(awk '/^Cached:/ {print int($2/1048576)}' /proc/meminfo) GB)"
}
summary() {   # a chatmoe.py JSON's lines
  python - $1 <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"  {r['weights']}; {r['experts']}; staged vision {r.get('vision_staged')}; load {r['load_s']} s, "
      f"host {r['host_mem']}")
for i, t in enumerate(r["turns"]):
    print(f"  turn {i + 1}: {t.get('images', 0)} images (vision {t.get('vision_s')} s); prompt "
          f"{t['prompt_tokens']} tokens (reused {t['reused']}) in {t['prefill_s']} s ({t['prefill_tok_s']} "
          f"tok/s, device {t['prefill_dev_tok_s']}), {t['prefill_misses']} misses; TTFT {t['ttft_s']} s; decode "
          f"{t['gen_tokens']} tokens at {t['decode_tok_s']} tok/s (device {t['decode_dev_tok_s']}), hits "
          f"{t['hit_rate']}, {t['mb_per_token']} MB/token; end {t['end']}\n    {t['reply']!r}")
PY
}
chat() {      # chatmoe.py run $1 (RUN's options), checked against REFOF's reference
  eval "timeout 3600 python tools/qual/chatmoe.py $O/$MD --pool $O/$POOL --backend board --cfg $R/cfg-dev.pkl \
    --check $RF/minicpmv-${REFOF[$1]}.json --out $R/minicpmv-$1.json ${POOL_CALLS:+--pool-calls} ${RUN[$1]}" \
    > $R/minicpmv-$1.log 2>&1
  local rc=$?
  echo "  exit $rc $(date +%T)"; grep -E "\[(PASS|FAIL)\]|Error|Traceback" $R/minicpmv-$1.log | tail -3
  [ $rc = 0 ] && grep -q "\[PASS\]" $R/minicpmv-$1.log || fail=1
  [ -f $R/minicpmv-$1.json ] && summary $R/minicpmv-$1.json
}
for m in $RUNS; do
  echo "=== $m $(date +%T)"
  case $m in
    warmup) chat warmup ;;
    tower)
      warm
      timeout 2400 python tools/minicpmv_tower.py time $O/$MD --backend board --cfg $R/cfg-dev.pkl \
        --pool $O/$POOL --image $O/img/cats.jpg:1 --image $O/img/doc.png --reps 1 --out $R/tower.json \
        > $R/tower.log 2>&1
      rc=$?; echo "  exit $rc $(date +%T)"; grep -vE "^\[transformers" $R/tower.log | grep -vE "^    " | tail -12
      [ $rc = 0 ] || fail=1
      [ -f $R/tower.json ] && python - $R/tower.json <<'PY'
import json, math, sys
G, G0 = 3.27e-3, 4.57e-3    # s a 16-row group of vit_out_mlp with the MLP: co-sim after the fix, mcpv2's card
for im in json.load(open(sys.argv[1]))["images"]:
    for c in im["crops"]:
        h, w = c["size"]
        g = {"out": math.ceil(h * w / 16), "out2": math.ceil(h * w / 64), "last": math.ceil(h * w / 64)}
        ph = " ".join(f"{k} {1e3 * x['device_s'] / x['runs']:.1f} ms a run (expected {1e3 * g[k] * G:.1f}, "
                      f"mcpv2 ~{1e3 * g[k] * G0:.1f})" for k, x in c["phases"].items() if k in g)
        print(f"  crop {h}x{w}: device {c['device_s']:.3f} s (mcpv2 ~5.77 at 1,008-1,024 patches, expected ~4.8); {ph}")
PY
      ;;
    pooldisk)
      timeout 2400 python tools/offload/pool_disk.py $O/$MD --pool $O/$POOL --backend board \
        --cfg $R/cfg-dev.pkl --out $R/pooldisk.json > $R/pooldisk.log 2>&1
      rc=$?; echo "  exit $rc $(date +%T)"; grep -vE "^\[transformers" $R/pooldisk.log | tail -6
      [ $rc = 0 ] || fail=1 ;;
    *) warm; chat $m ;;
  esac
done
echo "=== MiniCPM-V's pool and checkpoint out of the page cache $(date +%T)"   # (the card host's 31 GB hold
python - $O/$POOL $O/$MD <<'PY'           # one 35B pool: after session hostr21 this one's 18 GB stayed, and the next
import os, sys                            # session's warm held 63% of production's 35B pool)
gb = n = 0
for a in sys.argv[1:]:
    fs = [os.path.join(r, f) for r, _, fl in os.walk(a) for f in fl] if os.path.isdir(a) else [a]
    for f in fs:
        if os.path.isfile(f):
            fd = os.open(f, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            gb, n = gb + os.path.getsize(f) / 1e9, n + 1
print(f"  dropped from the page cache: {n} files of {gb:.1f} GB")
PY
echo "  mem $(mem) GB, cached $(awk '/^Cached:/ {print int($2/1048576)}' /proc/meminfo) GB"
echo "=== final selftest $(date +%T)"
st=$(timeout 600 python -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "$st" | grep -q "ALL PASS" || fail=1
echo "MCPV3 $([ $fail = 0 ] && echo PASS || echo FAIL) $(date +%T)"
exit $fail
