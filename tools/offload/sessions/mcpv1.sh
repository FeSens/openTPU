#!/bin/bash
# Session mcpv1 (docs/minicpmv.md 6): MiniCPM-V 4.7 (35B-A3B) on the card for the first time, through otpu-chat's
# engine and Chat (tools/qual/chatmoe.py): the language model's experts streamed from the MX pool, the vision tower
# resident in int8 (Spec.vision_staged False), MiniCPM-V's formats (streamed.FORMATS["minicpmv"]: fp4 experts,
# DeltaNet int8 in layers 0-19, the shared expert int8, the rest nvfp4, head int8). Runs, each token-exact against
# the ISA simulator's reference (REF=1, below):
#   text    chatmoe.py's two text turns (TURNS, --max-new 16): the language model alone
#   thumb   cats.jpg at --max-slices 1 (its thumbnail: one crop), two turns
#   sliced  doc.png at the processor's 9 slices (7 crops), one turn of --max-new 32
#   serve   otpu-serve on the card, tools/serve_bench.py's images part (docs/serve.md Images) at max_slices 1: cats.jpg
#           with thumb's first question (its reply must be thumb's reference's first turn's), again (the context
#           reused past the image), its next turn, doc.png with the same question (no context past it reused), a
#           text-only question; each request's cached tokens and TTFT (no reference of its own)
# Each chatmoe.py turn reports its images and their crops' seconds through the card's tower (vision_s), TTFT, prefill
# and decode tok/s (wall and device) and the slots' hits.
# Card: production deploy_r2m4_b10245f5 as loaded (EXPECT b10245f5, register map 5); its qual/cfg.pkl.
# O (/data/otpu/minicpmv) holds MiniCPM-V-4.7-35B-A3B/ (the slim checkpoint), pool-mcpv35-mx.split.bin (+ .packed,
# .format, .key), img/ (cats.jpg, doc.png), refs/minicpmv-<run>.json and pylib/ (PYLIB: the simulation host's ~/mcpv/tf518,
# torchvision 0.29.1+cpu, pillow 12.3.0 and transformers 5.18.0 over the venv's torch 2.14.0: the image's decode and
# resize as HF's processor, the packages the references ran with; without torchvision the resize differs).
# One otpu-lock:
# 1. selftest; the device configuration must be production's qual/cfg.pkl;
# 2. per run (RUNS, default "text thumb sliced serve"): every other file of 100 MB or more under O, ~/openTPU/models
#    and ~/otpu-build/offload out of the page cache, the pool read into it, then chatmoe.py --check its reference
#    (serve: otpu-serve on PORT, default 8017, and serve_bench.py against it, then the server stopped);
# 3. selftest; the summary per run and turn.
# A build other than EXPECT, a failed selftest, another configuration or a missing file stops it; a failed run or a
# token mismatch is reported and the next run goes on. The last line: MCPV1 PASS or MCPV1 FAIL. About 35 min (the
# image's first build on the card host ~5 min, otpu-serve's ~5 min more, the pool's read ~40 s a run).
# References (REF=1: the same runs on the ISA simulator with REFCFG, no card, no selftest), on the simulation host from this
# tree, one run a call (each in a slot of its own), into RF (serve has none: thumb's):
#   REF=1 O=~/mcpv PYLIB=~/mcpv/tf518 REFCFG=~/mcpv/cfg-b10245f5.pkl RUNS=thumb bash tools/offload/sessions/mcpv1.sh
#   (O there: MiniCPM-V-4.7-35B-A3B -> slim/MiniCPM-V-4.7-35B-A3B, the pool, img/; RF default O/refs)
# Stage (from a laptop, before the session; no host work while a session runs):
#   git archive <sha> | ssh CARD_HOST 'D=~/otpu-build/minicpmv/mcpv1/host; mkdir -p $D && tar -x -C $D'
#   ssh CARD_HOST 'echo <sha> > ~/otpu-build/minicpmv/mcpv1/host/COMMIT'
#   ssh SIM_HOST 'tar -C ~/mcpv -cf - tf518 img refs' | ssh CARD_HOST 'cd /data/otpu/minicpmv && tar -xf - && \
#     rm -rf pylib && mv tf518 pylib'
# Run (no pipe): setsid nohup ~/.local/bin/otpu-lock --wait 600 -- bash \
#   ~/otpu-build/minicpmv/mcpv1/host/tools/offload/sessions/mcpv1.sh > ~/otpu-build/minicpmv/mcpv1.launch.log 2>&1 \
#   < /dev/null &
set -u
REF=${REF:-0}
if [ $REF = 1 ]; then
  OTPU_QUIET=${OTPU_QUIET:-/dev/null}               # (no card host: nothing to keep quiet)
  SESSION=${SESSION:-mcpv1-ref}
else
  for v in $(env | grep -o '^OTPU_[A-Z0-9_]*' | grep -vE '^OTPU_LOCK_(HELD|PID|WAIT)$'); do unset $v; done
  export OTPU_IMG_CACHE_GB=30 OTPU_IMG_CACHE_FREE_GB=6      # (the card host's, from its ~/.bashrc)
  SESSION=${SESSION:-mcpv1}
fi
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
RUNS=${RUNS:-text thumb sliced serve}
PORT=${PORT:-8017}
declare -A RUN=(        # chatmoe.py's options
  [text]=""
  [thumb]="--image 1:$O/img/cats.jpg --max-slices 1 --turn 'Describe this image in one sentence.' --turn 'What colors stand out? One sentence.'"
  [sliced]="--image 1:$O/img/doc.png --turn 'What does this document say? Two sentences.' --max-new 32")
fail=0
echo "mcpv1 start $(date +%T) REF $REF T $T ($rev) O $O RF $RF REFCFG $REFCFG mem $(mem) GB; $(uptime)"
grep -q '"minicpmv": ("fp4", "int8", "delta@0-19=int8,attn=nvfp4,delta=nvfp4,mlp=int8")' \
  $T/opentpu/host/streamed.py && grep -q "def place" $T/opentpu/host/offload.py \
  || { echo "STOP: T is not the minicpmv branch (MiniCPM-V's formats, the expert server's place)"; exit 2; }
python -c "import torchvision, PIL, transformers; print('  python:', torchvision.__version__, PIL.__version__,
  transformers.__version__)" || { echo "STOP: no torchvision / pillow / transformers under PYLIB $PYLIB"; exit 2; }
[ -f $O/$MD/config.json ] && [ -f $O/$POOL ] && [ -f $O/$POOL.key ] && [ -f $O/img/cats.jpg ] && \
  [ -f $O/img/doc.png ] && [ -f $REFCFG ] || { echo "STOP: $O/$MD, $O/$POOL, its key, img/ or $REFCFG missing"; exit 2; }
if [ $REF = 1 ]; then
  mkdir -p $RF
  for m in ${RUNS/serve/}; do
    echo "=== ref $m $(date +%T)"
    t0=$SECONDS
    eval "python tools/qual/chatmoe.py $O/$MD --pool $O/$POOL --backend isa --cfg $REFCFG \
      --out $RF/minicpmv-$m.json ${RUN[$m]}" > $R/ref-$m.log 2>&1
    echo "  exit $? in $((SECONDS - t0)) s $(date +%T)"; grep -E "^turn|Error|Traceback" $R/ref-$m.log | tail -6
  done
  echo "MCPV1 REF DONE $(date +%T)"
  exit 0
fi
for m in $RUNS; do
  [ $m = serve ] && m=thumb
  [ -f $RF/minicpmv-$m.json ] || { echo "STOP: no reference $RF/minicpmv-$m.json"; exit 2; }
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
for m in $RUNS; do
  echo "=== $m $(date +%T)"
  python - $O/$MD $O/$POOL $O:$HOME/openTPU/models:$HOME/otpu-build/offload <<'PY'
import os, sys                                # every other file of 100 MB or more out of the
keep = {os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])}    # page cache
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
  t0=$SECONDS; cat $O/$POOL > /dev/null       # the pool in the page cache (the RAM tier)
  echo "  (pool read in $((SECONDS - t0)) s, mem $(mem) GB)"
  if [ $m = serve ]; then
    timeout 3600 python -m opentpu.serve.api --model $O/$MD --backend board --pool $O/$POOL --port $PORT \
      --kv-ram-gb 2 --kv-dir $R/kv > $R/serve.log 2>&1 &
    sp=$!
    timeout 2400 python tools/serve_bench.py --port $PORT --skip check,throughput,conversation,users,arrivals \
      --image $O/img/cats.jpg --image $O/img/doc.png --image-slices 1 --image-tokens 16 \
      --image-ref $RF/minicpmv-thumb.json --wait 1200 --out $R/serve-bench.json > $R/serve-bench.log 2>&1
    rc=$?
    kill $sp 2>/dev/null; wait $sp 2>/dev/null
    echo "  exit $rc $(date +%T); server: $(grep -E "^weights|Error|Traceback" $R/serve.log | tail -2 | tr '\n' ' ')"
    [ $rc = 0 ] || fail=1
    python - $R/serve-bench.json <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1]))["images"]
except (OSError, KeyError, ValueError) as e:
    sys.exit(f"  no images part: {e!r}")
print(f"  images {'PASS' if r['pass'] else 'FAIL'}: thumb's reply {r.get('ref_equal')}, again equal "
      f"{r['again_equal']}")
for k in ("first", "again", "turn2", "others", "text"):
    for x in (r[k] if k == "others" else [r[k]]):
        print(f"  {k}: prompt {x['prompt_tokens']}, cached {x['cached']}, TTFT {x['ttft_s']} s, "
              f"{x['finish']}: {x['text']!r}")
PY
    continue
  fi
  eval "timeout 3600 python tools/qual/chatmoe.py $O/$MD --pool $O/$POOL --backend board \
    --cfg $R/cfg-dev.pkl --check $RF/minicpmv-$m.json --out $R/minicpmv-$m.json ${RUN[$m]}" \
    > $R/minicpmv-$m.log 2>&1
  rc=$?
  echo "  exit $rc $(date +%T)"; grep -E "\[(PASS|FAIL)\]|Error|Traceback" $R/minicpmv-$m.log | tail -3
  [ $rc = 0 ] && grep -q "\[PASS\]" $R/minicpmv-$m.log || fail=1
  [ -f $R/minicpmv-$m.json ] || continue
  python - $R/minicpmv-$m.json <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"  {r['weights']}; {r['experts']}; layer-major {r['layer_major']}, loop on the card "
      f"{r['on_card']}, load {r['load_s']} s, host {r['host_mem']}")
for i, t in enumerate(r["turns"]):
    print(f"  turn {i + 1}: {t.get('images', 0)} images (vision {t.get('vision_s')} s); prompt "
          f"{t['prompt_tokens']} tokens (reused {t['reused']}"
          f"{', restarted' if t['restarted'] else ''}) in {t['prefill_s']} s "
          f"({t['prefill_tok_s']} tok/s, device {t['prefill_dev_tok_s']}), {t['prefill_misses']} "
          f"misses; TTFT {t['ttft_s']} s; decode {t['gen_tokens']} tokens at {t['decode_tok_s']} "
          f"tok/s (device {t['decode_dev_tok_s']}), hits {t['hit_rate']}, "
          f"{t['mb_per_token']} MB/token; end {t['end']}\n    {t['reply']!r}")
PY
done
echo "=== final selftest $(date +%T)"
st=$(timeout 600 python -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "$st" | grep -q "ALL PASS" || fail=1
echo "MCPV1 $([ $fail = 0 ] && echo PASS || echo FAIL) $(date +%T)"
exit $fail
