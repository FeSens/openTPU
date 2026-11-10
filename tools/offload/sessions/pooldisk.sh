#!/bin/bash
# Session pooldisk (docs/minicpmv.md 6): what an expert pool's disk costs a warm process, on the card host's two disks.
# mcpv3b's dropped arms were invalid (the engine maps the pool: posix_fadvise DONTNEED left it cached); pool_disk.py now
# releases those maps first and checks every arm with mincore. Pools (each its model's engine, otpu-chat's):
#   mcpv  MiniCPM-V 4.7's, /data (sdb, a 7200 rpm disk)
#   q35   qwen35-35b's, production's file on /home (sda, the SATA SSD): read only (its .packed and .key's sums and the
#         pool's size and mtime are checked before and after)
# Per pool: a warm-up process (pool_disk.py --arms "": the engine's open, which builds the image into the image cache
# if this tree's keys are new, and the warm-up turn; unmeasured), then the measured one: --arms cached,dropped,dropped,cached, the same turn
# greedy. The arms' tok/s, misses and reads by where they came from (page cache or disk): a miss's service time per
# disk. One otpu-lock: selftest (production's build and configuration), the pools, selftest. A failed run, an INVALID
# arm or a changed q35 file is a FAIL; the last line: POOLDISK PASS or POOLDISK FAIL. About 20 min.
# Stage (from a laptop): git archive <sha> | ssh CARD_HOST 'D=~/otpu-build/minicpmv/pooldisk/host; mkdir -p $D &&
#   tar -x -C $D'; ssh CARD_HOST 'echo <sha> > ~/otpu-build/minicpmv/pooldisk/host/COMMIT'
# Run (no pipe): setsid nohup ~/.local/bin/otpu-lock --wait 600 -- bash \
#   ~/otpu-build/minicpmv/pooldisk/host/tools/offload/sessions/pooldisk.sh > ~/otpu-build/minicpmv/pooldisk.launch.log \
#   2>&1 < /dev/null &
set -u
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*' | grep -vE '^OTPU_LOCK_(HELD|PID|WAIT)$'); do unset $v; done
export OTPU_IMG_CACHE_GB=30 OTPU_IMG_CACHE_FREE_GB=6      # (the card host's, from its ~/.bashrc)
SESSION=${SESSION:-pooldisk}
O=${O:-/data/otpu/minicpmv}
C2=${C2:-$HOME/otpu-build/offload/card2}
PD=${PD:-$HOME/otpu-build/deploy_r2m4_b10245f5}        # (production: qual/cfg.pkl)
REFCFG=${REFCFG:-$PD/qual/cfg.pkl}
PYLIB=${PYLIB:-$O/pylib}
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"       # (T: this tree; R = O/SESSION; OTPU_QUIET claimed)
export PYTHONPATH=$T:$PYLIB
exec > >(tee -a $R/session.log) 2>&1
EXPECT=${EXPECT:-b10245f5}
MAP=${MAP:-register map 5}
POOLS=${POOLS:-mcpv q35}
declare -A MODEL=([mcpv]=$O/MiniCPM-V-4.7-35B-A3B [q35]=$C2/Qwen3.5-35B-A3B)
declare -A POOL=([mcpv]=$O/pool-mcpv35-mx.split.bin [q35]=$C2/pool-q35-mx.split.bin)
fail=0
echo "pooldisk start $(date +%T) T $T ($rev) R $R mem $(mem) GB; $(uptime)"
grep -q 'MADV_DONTNEED' $T/tools/offload/pool_disk.py && grep -q '"valid"\|valid=valid' $T/tools/offload/pool_disk.py \
  || { echo "STOP: T's pool_disk.py does not release the engine's maps or check its arms"; exit 2; }
for p in $POOLS; do
  [ -f ${MODEL[$p]}/config.json ] && [ -f ${POOL[$p]} ] && [ -f ${POOL[$p]}.key ] && [ -f ${POOL[$p]}.packed ] \
    || { echo "STOP: $p's ${MODEL[$p]} or ${POOL[$p]} (.key, .packed) missing"; exit 2; }
done
[ -f $REFCFG ] || { echo "STOP: no $REFCFG"; exit 2; }
st=$(timeout 600 python -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "$st" | grep -q "build $EXPECT" || { echo "STOP: the card's build is not $EXPECT"; exit 2; }
echo "$st" | grep -q "$MAP" || { echo "STOP: not $MAP"; exit 2; }
echo "$st" | grep -q "ALL PASS" || { echo "STOP: selftest"; exit 2; }
timeout 300 python tools/qual/refs.py cfg $R/cfg-dev.pkl > /dev/null 2>&1; echo "cfg exit $?"
python - $R/cfg-dev.pkl $REFCFG <<'PY' || { echo "STOP: the device configuration is not production's"; exit 2; }
import dataclasses, pickle, sys
a, b = (pickle.load(open(f, "rb")) for f in sys.argv[1:])
d = {f.name: (getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
     if getattr(a, f.name) != getattr(b, f.name)}
print("  cfg: production's configuration" if not d else f"  cfg DIFFERS (device, production): {d}")
sys.exit(1 if d else 0)
PY
mark() {      # a pool's files as they are: size and mtime of the pool, sums of its .packed and .key
  echo "$(stat -c '%s %Y' $1) $(sha256sum $1.packed $1.key | cut -c1-16 | tr '\n' ' ')"
}
for p in $POOLS; do
  echo "=== $p $(date +%T): ${POOL[$p]} on $(df --output=source ${POOL[$p]} | tail -1)"
  before=$(mark ${POOL[$p]})
  timeout 1200 python tools/offload/pool_disk.py ${MODEL[$p]} --pool ${POOL[$p]} --backend board \
    --cfg $R/cfg-dev.pkl --arms "" --out $R/pooldisk-$p-warmup.json > $R/pooldisk-$p-warmup.log 2>&1
  rc=$?; echo "  warm-up exit $rc $(date +%T)"; grep -vE "^\[transformers" $R/pooldisk-$p-warmup.log | tail -3
  [ $rc = 0 ] || { fail=1; continue; }
  timeout 1500 python tools/offload/pool_disk.py ${MODEL[$p]} --pool ${POOL[$p]} --backend board \
    --cfg $R/cfg-dev.pkl --out $R/pooldisk-$p.json > $R/pooldisk-$p.log 2>&1
  rc=$?; echo "  exit $rc $(date +%T)"; grep -vE "^\[transformers" $R/pooldisk-$p.log | tail -6
  [ $rc = 0 ] || fail=1
  python - $R/pooldisk-$p.json <<'PY' || fail=1
import json, sys
r = json.load(open(sys.argv[1]))
bad = [a["arm"] for a in r["arms"] if not a.get("valid")]
print(f"  arms valid: {len(r['arms']) - len(bad)} of {len(r['arms'])}" + (f" (INVALID: {bad})" if bad else ""))
sys.exit(1 if bad or len(r["arms"]) != 4 else 0)
PY
  after=$(mark ${POOL[$p]})
  [ "$before" = "$after" ] && echo "  pool files unchanged: $after" || { echo "  POOL FILES CHANGED: $before -> $after"; fail=1; }
done
echo "=== final selftest $(date +%T)"
st=$(timeout 600 python -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "$st" | grep -q "ALL PASS" || fail=1
echo "POOLDISK $([ $fail = 0 ] && echo PASS || echo FAIL) $(date +%T)"
exit $fail
