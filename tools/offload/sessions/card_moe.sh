#!/bin/bash
# offload card session runs (production bitstream, no reload): MoE models with their experts
# streamed from a packed pool file on the host into DRAM slots, the card routing and computing
# every expert inside its generate loop. Each run is checked against the ISA simulator's run with
# the bitstream's configuration (its prefill logits' sha256 and its tokens: a longer run against
# the reference's prefix) and against HF's.
#   8b16   LFM2.5-8B-A1B, 16 tokens (hf.json's prompt, ref16.json)
#   8b160  LFM2.5-8B-A1B, 160 tokens (hf160.json's prompt, ref160.json): the steady state
#   q35    Qwen3.5-35B-A3B, 16 tokens at 32 slots per layer (q35-hf.json, q35ref16.json)
#   q35lfu, 8b160lfu: the same with the slots replaced by least decayed use (--policy lfu)
#   q35c, q35e, q35eh: the 35B by decayed use, its slots filling the DRAM (--experts 0), with
#     the embedding table on the card or the host (--embed-table) and the router's hints off or
#     on (--hints; docs/offload.md 5.9 and 12)
#   q35eht, q35ehd, q35ehs, q35ehds: q35eh with the decode's hint timeline (--hint-trace), and
#     with a request withdrawing its layer's unnamed hints (--hint-drop) and/or 128 KiB parts
#   q35e128a/b, q35c128a/b: q35e and q35c at 128 tokens (their first 16 against the reference)
#   g26a, g26b: gemma-4-26B-A4B (int8 layers, fp4 experts and head, slots filling the DRAM: 18 a
#     layer), 128 tokens after wiki.txt's first paragraph (q26-hf.json, q26ref16.json: 16 tokens)
#   g26t16, g26lm1, g26lm2: the 26B at 16 tokens, its prompt token by token (as g26a) or layer
#     by layer in runs of 1 or 2 rows (--layer-major; docs/offload.md 13)
# Before each run the other pools leave the page cache and the run's pool is read into it. A run
# whose files are not staged in O is skipped. Selftest before and after.
# Run: otpu-lock --wait 3600 -- tools/offload/sessions/card_moe.sh   (RUNS="8b16 8b160 q35";
# env.sh's T, O, R, RF; PSFX=.split or "" for the pools' format; G26POOL)
set -u
source "$(dirname "$0")/env.sh"
PSFX=${PSFX-.split}       # the pool files: the split format (.split.bin), or "" for the slot format
G26POOL=${G26POOL:-../g26/pool-g26-fp4.split.bin}     # (relative to O)
echo "card_moe start $(date +%T) tree $rev mem $(mem) GB"
timeout 1800 python -m opentpu.host.selftest 2>&1 | grep -E "\[(PASS|FAIL)\]|config" | tail -12
timeout 300 python tools/qual/refs.py cfg $R/cfg-dev.pkl > /dev/null 2>&1; echo "cfg exit $?"
python - $R/cfg-dev.pkl $REFCFG <<'PY'
import dataclasses, pickle, sys
a, b = (pickle.load(open(f, "rb")) for f in sys.argv[1:])
d = {f.name: (getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
     if getattr(a, f.name) != getattr(b, f.name)}
print("  cfg: the references' configuration" if not d else f"  cfg DIFFERS from the references' (device, ref): {d}")
sys.exit(1 if d else 0)
PY
[ $? = 0 ] || { echo "STOP: the device configuration is not the references'"; exit 2; }
Q35="Qwen3.5-35B-A3B pool-q35-fp4$PSFX.bin"
declare -A RUN=(         # checkpoint, pool, tokens, HF's, reference, output, slots (0: fill), policy,
                         # moe_card's other options
  [8b16]="LFM2.5-8B-A1B pool-lfm25-8b-fp4$PSFX.bin 16 hf.json ref16 card16 0 lru"
  [8b160]="LFM2.5-8B-A1B pool-lfm25-8b-fp4$PSFX.bin 160 hf160.json ref160 card160 0 lru"
  [8b160lfu]="LFM2.5-8B-A1B pool-lfm25-8b-fp4$PSFX.bin 160 hf160.json ref160 card160lfu 0 lfu"
  [q35]="$Q35 16 q35-hf.json q35ref16 q35card16 32 lru"
  [q35lfu]="$Q35 16 q35-hf.json q35ref16 q35card16lfu 32 lfu"
  [q35c]="$Q35 16 q35-hf.json q35ref16 q35card16c 0 lfu --embed-table card --hints off"
  [q35e]="$Q35 16 q35-hf.json q35ref16 q35card16e 0 lfu --embed-table host --hints off"
  [q35eh]="$Q35 16 q35-hf.json q35ref16 q35card16eh 0 lfu --embed-table host --hints on"
  [q35eht]="$Q35 16 q35-hf.json q35ref16 q35card16eht 0 lfu --embed-table host --hints on --hint-trace $R/q35card16eht.trace.json"
  [q35ehd]="$Q35 16 q35-hf.json q35ref16 q35card16ehd 0 lfu --embed-table host --hints on --hint-drop --hint-trace $R/q35card16ehd.trace.json"
  [q35ehs]="$Q35 16 q35-hf.json q35ref16 q35card16ehs 0 lfu --embed-table host --hints on --hint-part 128 --hint-trace $R/q35card16ehs.trace.json"
  [q35ehds]="$Q35 16 q35-hf.json q35ref16 q35card16ehds 0 lfu --embed-table host --hints on --hint-drop --hint-part 128 --hint-trace $R/q35card16ehds.trace.json"
  [q35e128a]="$Q35 128 q35-hf.json q35ref16 q35card128ea 0 lfu --embed-table host --hints off"
  [q35e128b]="$Q35 128 q35-hf.json q35ref16 q35card128eb 0 lfu --embed-table host --hints off"
  [q35c128a]="$Q35 128 q35-hf.json q35ref16 q35card128ca 0 lfu --embed-table card --hints off"
  [q35c128b]="$Q35 128 q35-hf.json q35ref16 q35card128cb 0 lfu --embed-table card --hints off"
  [g26a]="gemma-4-26B-A4B $G26POOL 128 q26-hf.json q26ref16 g26card128a 0 lfu --wformat int8 --formats experts=fp4 --head-format fp4"
  [g26b]="gemma-4-26B-A4B $G26POOL 128 q26-hf.json q26ref16 g26card128b 0 lfu --wformat int8 --formats experts=fp4 --head-format fp4"
  [g26t16]="gemma-4-26B-A4B $G26POOL 16 q26-hf.json q26ref16 g26card16 0 lfu --wformat int8 --formats experts=fp4 --head-format fp4"
  [g26lm1]="gemma-4-26B-A4B $G26POOL 16 q26-hf.json q26ref16 g26card16lm1 0 lfu --wformat int8 --formats experts=fp4 --head-format fp4 --layer-major 1"
  [g26lm2]="gemma-4-26B-A4B $G26POOL 16 q26-hf.json q26ref16 g26card16lm2 0 lfu --wformat int8 --formats experts=fp4 --head-format fp4 --layer-major 2")
for name in ${RUNS:-8b16 8b160 q35}; do
  read -r md pool n hf ref out ex pol extra <<< "${RUN[$name]}"
  if [ ! -f $O/$md/config.json ] || [ ! -f $O/$pool ] || [ ! -f $RF/$ref.json ]; then
    echo "  [SKIP] $name: not staged"; continue; fi
  echo "=== $name $(date +%T)"
  python - $O/$pool $O/pool-*.bin $O/$(dirname $G26POOL)/pool-*.bin <<'PY'
import os, sys                                # the other pools out of the page cache (session 8:
for f in set(map(os.path.realpath, sys.argv[2:])) - {os.path.realpath(sys.argv[1])}:   # the
    if os.path.exists(f):                     # 26B's pages, read twice, outlived the 35B's, read
        fd = os.open(f, os.O_RDONLY)          # once, and its staging read the disk)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
PY
  t0=$SECONDS; cat $O/$pool > /dev/null      # the RAM tier warm before each timed run, and how
  python - $O/$pool <<'PY'                    # much of the pool the page cache holds
import os, sys
from opentpu.host.offload import PoolFile
p = sys.argv[1]
n = os.path.getsize(p + ".packed")
print(f"  pool warm: {PoolFile(p, os.path.getsize(p) // n, True).resident(range(n)) / 1e9:.2f} of "
      f"{os.path.getsize(p) / 1e9:.2f} GB resident")
PY
  echo "  (read in $((SECONDS - t0)) s)"
  timeout 3600 python tools/offload/moe_card.py $O/$md --check $O/$hf -n $n --pool $O/$pool \
    --cfg $R/cfg-dev.pkl --experts $ex --card --policy $pol ${extra:-} --out $R/$out.json \
    > $R/$out.log 2>&1
  echo "  exit $? $(date +%T)"; grep -E "Error|Traceback" $R/$out.log | tail -3
  [ -f $R/$out.json ] || continue
  python - $R/$out.json $RF/$ref.json $O/$hf <<'PY'
import json, sys
c, r, h = (json.load(open(f)) for f in sys.argv[1:])
same = c["tokens"][:len(r["tokens"])] == r["tokens"]      # (a longer run: the reference's prefix)
dh = next((i for i, (a, b) in enumerate(zip(c["tokens"], h["tokens"])) if a != b), None)
print(f"  [{'PASS' if same and c['prefill_logits_sha'] == r['prefill_logits_sha'] else 'FAIL'}] "
      f"vs ISA sim: tokens {'same' if same else 'DIFFER'}, prefill sha {c['prefill_logits_sha']} "
      f"vs {r['prefill_logits_sha']}; vs HF: first diff {dh}")
k = ("tok_s_wall", "tok_s_device", "hits", "misses", "misses_per_token_decode",
     "misses_per_token_decode_2nd_half", "bytes_per_token_decode", "host_decode_s", "load_s",
     "prefill_s", "generate_s", "experts_per_layer", "policy", "pool_warm", "embed_host",
     "hints", "layer_major", "prefill_requests", "prefill_misses")
print("  " + json.dumps({x: c.get(x) for x in k}))
PY
done
echo "=== final selftest $(date +%T)"
timeout 1800 python -m opentpu.host.selftest 2>&1 | grep -E "\[(PASS|FAIL)\]|config|ALL" | tail -12
echo "CARD_MOE DONE $(date +%T)"
