#!/bin/bash
# Session mcpv2 (docs/minicpmv.md 6): MiniCPM-V 4.7 (35B-A3B) on the card, where the time goes and the staged vision.
# mcpv1 passed (text, thumb, serve token-exact) with thumb's one crop at 30.3 s of vision, otpu-serve's image TTFTs at
# 16.6 / 15.5 s and text's first turns cold (its image's first build, 200.8 s at a 28.4 GB peak of the host's 31 GB,
# took the pool out of the page cache: the pool is on /data, a 7200 rpm disk). Runs (RUNS, default
# "tower text sliced staged stower"), each after every other file of 100 MB or more under O, ~/openTPU/models and
# ~/otpu-build/offload leaves the page cache and the pool is read into it:
#   tower    tools/minicpmv_tower.py time: cats.jpg at max-slices 1 (one crop), then doc.png at 9 slices (7 crops),
#            each step timed (host preprocessing, the programs' build, host rows, their writes, each run's start and
#            wait against its device time by phase, the rows' read), then minicpmv.Vision.encode twice as a whole
#   text     chatmoe.py's two text turns again, the image built already (mcpv1's were cold): the warm decode
#   sliced   doc.png at 9 slices, one turn of 32 tokens, token-exact against its reference
#   staged   cats.jpg at max-slices 1, two turns, the tower's weights staged through the expert slots
#            (chatmoe.py --vision-staged): a warm-up (its image's first build; the pool grows by the 446 vision
#            entries, packed on first use), the pool read again, then the run, token-exact against thumb's reference
#            (staged rows are the resident tower's bit for bit): its slots a layer, decode tok/s and hits against
#            mcpv1's thumb (resident)
#   stower   tools/minicpmv_tower.py time --vision-staged on cats.jpg (max-slices 1) and doc.png: the placement's
#            seconds (Vision.stage) beside the crops'
# Card: production deploy_r2m4_b10245f5 as loaded (EXPECT b10245f5, register map 5); its qual/cfg.pkl. O
# (/data/otpu/minicpmv): as mcpv1 (the slim checkpoint, the pool, img/, refs/, pylib/). The staged runs grow the pool
# file in place by 763.6 MB (its key's "vision"; the experts unchanged: SUMS-pool's sum no longer holds for the file).
# One otpu-lock: selftest (the configuration must be production's), the runs, selftest. A build other than EXPECT, a
# failed selftest, another configuration or a missing file stops it; a failed run or a token mismatch is reported and
# the next run goes on. The last line: MCPV2 PASS or MCPV2 FAIL. About 30 min (the staged image's first build ~4 min).
# Stage (from a laptop, before the session; no host work while a session runs):
#   git archive <sha> | ssh CARD_HOST 'D=~/otpu-build/minicpmv/mcpv2/host; mkdir -p $D && tar -x -C $D'
#   ssh CARD_HOST 'echo <sha> > ~/otpu-build/minicpmv/mcpv2/host/COMMIT'
# Run (no pipe): setsid nohup ~/.local/bin/otpu-lock --wait 600 -- bash \
#   ~/otpu-build/minicpmv/mcpv2/host/tools/offload/sessions/mcpv2.sh > ~/otpu-build/minicpmv/mcpv2.launch.log 2>&1 \
#   < /dev/null &
set -u
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*' | grep -vE '^OTPU_LOCK_(HELD|PID|WAIT)$'); do unset $v; done
export OTPU_IMG_CACHE_GB=30 OTPU_IMG_CACHE_FREE_GB=6      # (the card host's, from its ~/.bashrc)
SESSION=${SESSION:-mcpv2}
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
RUNS=${RUNS:-tower text sliced staged stower}
THUMB="--image 1:$O/img/cats.jpg --max-slices 1 --turn 'Describe this image in one sentence.' --turn 'What colors stand out? One sentence.'"
declare -A RUN=(        # chatmoe.py's options
  [text]=""
  [sliced]="--image 1:$O/img/doc.png --turn 'What does this document say? Two sentences.' --max-new 32"
  [staged]="--vision-staged $THUMB")
declare -A REFOF=([text]=text [sliced]=sliced [staged]=thumb)
fail=0
echo "mcpv2 start $(date +%T) T $T ($rev) O $O RF $RF mem $(mem) GB, /data $(df -BG --output=avail /data | tail -1 | tr -d ' ') free; $(uptime)"
grep -q 'vision_staged: bool | None = None' $T/opentpu/host/streamed.py && [ -f $T/tools/minicpmv_tower.py ] \
  || { echo "STOP: T is not the minicpmv branch with minicpmv_tower.py and streamed.engine(vision_staged=)"; exit 2; }
python -c "import torchvision, PIL, transformers; print('  python:', torchvision.__version__, PIL.__version__,
  transformers.__version__)" || { echo "STOP: no torchvision / pillow / transformers under PYLIB $PYLIB"; exit 2; }
[ -f $O/$MD/config.json ] && [ -f $O/$POOL ] && [ -f $O/$POOL.key ] && [ -f $O/img/cats.jpg ] && \
  [ -f $O/img/doc.png ] && [ -f $REFCFG ] || { echo "STOP: $O/$MD, $O/$POOL, its key, img/ or $REFCFG missing"; exit 2; }
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
chat() {      # chatmoe.py run $1 (RUN's options), checked against REFOF's reference; $2: the output's name
  eval "timeout 3600 python tools/qual/chatmoe.py $O/$MD --pool $O/$POOL --backend board --cfg $R/cfg-dev.pkl \
    --check $RF/minicpmv-${REFOF[$1]}.json --out $R/minicpmv-$2.json ${RUN[$1]}" > $R/minicpmv-$2.log 2>&1
  local rc=$?
  echo "  exit $rc $(date +%T)"; grep -E "\[(PASS|FAIL)\]|Error|Traceback" $R/minicpmv-$2.log | tail -3
  [ $rc = 0 ] && grep -q "\[PASS\]" $R/minicpmv-$2.log || fail=1
  [ -f $R/minicpmv-$2.json ] && summary $R/minicpmv-$2.json
}
tower() {     # minicpmv_tower.py time, $1 its output's name, the rest its options
  local n=$1; shift
  timeout 2400 python tools/minicpmv_tower.py time $O/$MD --backend board --cfg $R/cfg-dev.pkl --pool $O/$POOL \
    --image $O/img/cats.jpg:1 --image $O/img/doc.png --reps 2 --out $R/$n.json "$@" > $R/$n.log 2>&1
  local rc=$?
  echo "  exit $rc $(date +%T)"; grep -vE "^\[transformers" $R/$n.log | grep -vE "^    " | tail -14
  [ $rc = 0 ] || fail=1
}
for m in $RUNS; do
  echo "=== $m $(date +%T)"
  case $m in
    tower) warm; tower tower ;;
    stower) warm; tower stower --vision-staged ;;
    staged)
      echo "  warm-up (the staged image's first build, the pieces packed into the pool)"
      chat staged staged-w
      warm; chat staged staged ;;
    *) warm; chat $m $m ;;
  esac
done
echo "=== final selftest $(date +%T)"
st=$(timeout 600 python -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "$st" | grep -q "ALL PASS" || fail=1
ls -la $O/$POOL $O/$POOL.key
echo "MCPV2 $([ $fail = 0 ] && echo PASS || echo FAIL) $(date +%T)"
exit $fail
