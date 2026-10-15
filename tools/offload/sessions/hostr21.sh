#!/bin/bash
# Session hostr21: the host release batch host-r21 (MiniCPM-V, the pool's warm gate, S = the shared expert beside the
# request, chatmoe's pool residency, qual.sh's ECC_EACH) qualified on production's bitstream b10245f5 as loaded (no JTAG
# load, LOAD=0; production stays on the card), under ONE otpu-lock. Its deploy directory D
# (~/otpu-build/deploy_hostr21_b10245f5): otpu.bit -> production's, host/ (this tree, COMMIT, models -> ~/openTPU/models),
# refcache/ (tools/qual/refs.py's references for this tree's keys, computed on hwe-bench on production's configuration,
# configs/deploy_hostr21_b10245f5.pkl), chatmoe/ (the streamed MoE models' references: qwen35-35b recomputed on this
# tree, lfm2-8b and gemma4-26b production's, whose programs S does not change).
# PART (env): A = qual.sh fast (ECC_EACH=1) + E2B-it + otpu-chat on eight models against production's replies;
#             B = chatmoe.sh (lfm2-8b qwen35-35b gemma4-26b, POOL_CALLS=1) + the warm gate on otpu-serve (qwen35-35b);
#             C = MiniCPM-V (mcpv3.sh RUNS="warmup sliced thumb text", POOL_CALLS=1, its references in /data);
#             all (default) = A, B, C. Selftest first (build b10245f5, register map 5, MCOLS 4 at 133.333 MHz: else
#             no session), an ECC check after each part and at the end, scrub, selftest.
# The warm gate (PART B): qwen35-35b's pool (production's file, read only) dropped from the page cache, otpu-serve
#   opened on it: /health polled every 2 s (each change of answer logged: 503 "warming" with its progress, then 200),
#   the server's "pool warm:" lines, serve_bench's check part through the warming 503 (wait_up); then the pool read
#   back into the page cache. PASS: a 503 "warming" seen, then 200 within the cap, "pool warm: done", serve_bench's
#   check passed.
# Every line that says FAIL, NOT CLEAN or EXIT <non-zero> is the lead's to read; the last line: HOSTR21 SESSION END.
# About 2.5 h (A ~70 min with the images' first builds for this tree, B ~45, C ~40).
# Stage (from the Mac): see the deploy directory's README. Run (no pipe; the lead's GO):
#   setsid -f ~/.local/bin/otpu-lock --wait 3600 -- bash ~/otpu-build/deploy_hostr21_b10245f5/host/tools/offload/\
#     sessions/hostr21.sh > ~/otpu-build/deploy_hostr21_b10245f5/session.log 2>&1 < /dev/null
set -u
PART=${PART:-all}
D=${D:-$HOME/otpu-build/deploy_hostr21_b10245f5}; H=$D/host
P=$HOME/otpu-venv/bin/python; RC=$D/refcache
PD=${PD:-$HOME/otpu-build/deploy_r2m4_b10245f5}        # production (its chat-*.txt, qual/cfg.pkl)
C2=$HOME/otpu-build/offload/card2
[ -n "${OTPU_LOCK_HELD:-}" ] || { echo "run under otpu-lock"; exit 1; }
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*' | grep -vE '^OTPU_LOCK_(HELD|PID|WAIT)$'); do unset $v; done
export OTPU_IMG_CACHE_GB=30 OTPU_IMG_CACHE_FREE_GB=6      # (the card host's, from its ~/.bashrc)
Q=$HOME/otpu-build/QUIET          # the whole session is quiet: no prebuilds or OOC jobs on the host
q=$(cat $Q 2>/dev/null); if [ -z "$q" ] || ! kill -0 "$q" 2>/dev/null; then echo $$ > $Q; trap '[ "$(cat $Q 2>/dev/null)" = $$ ] && rm -f $Q' EXIT; fi
cs() { echo "  img cache $(du -sm $HOME/otpu-build/qcache/img 2>/dev/null | cut -f1) MB, disk $(df -BG --output=avail /home | tail -1 | tr -d ' ') free, mem $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) GB"; }
ecc() {   # $1 = label: selftest's ECC counts since the last scrub; any error = NOT CLEAN
  echo "=== ECC check: $1 $(date +%T)"; cd /tmp && env PYTHONPATH=$H timeout 900 $P -m opentpu.host.selftest > $D/selftest_ecc_$1.log 2>&1
  grep -qE "\[PASS\] ecc .*channel 0: 0 corrected, 0 uncorrectable, channel 1: 0 corrected, 0 uncorrectable" $D/selftest_ecc_$1.log \
    && echo "ECC CLEAN ($1)" || echo "ECC NOT CLEAN ($1): $(grep -E " ecc " $D/selftest_ecc_$1.log | cut -c1-200)"
}
tree() { local x; for x in "$@"; do echo $x; tree $(pgrep -P $x); done; }    # these PIDs and all under them
live() { local x; for x in "$@"; do kill -0 $x 2> /dev/null && echo $x; done; }  # those of them alive
stop_tree() {   # $1: a background job's PID, timeout's (exec'd), the leader of the server's process group: TERM to the
  local p=$1 g w all left   # group (uutils' timeout passes a TERM to its child only, not to the child's children: a
  g=$(ps -o pgid= -p $p | tr -d ' ')   # compile worker would outlive it), up to 30 s for all of it; then KILL
  [ "$g" = "$p" ] || g=""              # (not its own group's leader: its tree instead)
  all=$(tree $p | tr '\n' ' ')
  if [ -n "$g" ]; then kill -- -$g 2> /dev/null; else kill $all 2> /dev/null; fi
  for w in $(seq 1 30); do
    left=$( { [ -n "$g" ] && pgrep -g $g; live $all; } | sort -u | tr '\n' ' ')
    [ -z "${left// }" ] && break; sleep 1; done
  if [ -n "${left// }" ]; then echo "  stop_tree: $left still alive after 30 s: KILL"
    [ -n "$g" ] && kill -9 -- -$g 2> /dev/null; kill -9 $left 2> /dev/null; sleep 2; fi
  wait $p 2> /dev/null
}
xdma_free() {   # the card's device files closed by every process of this user (60 s at most)
  local w u
  for w in $(seq 1 60); do u=$(lsof -t /dev/xdma0_* /dev/otpu0_* 2> /dev/null | tr '\n' ' '); [ -z "$u" ] && return 0; sleep 1; done
  echo "  the card's nodes still open by pids $u: FAIL"; return 1
}
echo "HOSTR21 SESSION START $(date +%T) PART $PART host $(cat $H/COMMIT) up since $(uptime -s)"
for f in $D/otpu.bit $H/models $RC/configs/$(basename $D).pkl $D/chatmoe/chatmoe-qwen35-35b.json; do
  [ -e $f ] || { echo "STOP: no $f"; exit 2; }; done
echo "=== selftest $(date +%T)"; cd /tmp && env PYTHONPATH=$H timeout 900 $P -m opentpu.host.selftest 2>&1 | tee $D/selftest$PART.log | grep -E "config|bandwidth|ALL|FAIL|ecc"
grep -q "build b10245f5 " $D/selftest$PART.log || { echo "BUILD MISMATCH: the card is not build b10245f5: no session"; exit 3; }
grep -q "register map 5" $D/selftest$PART.log || { echo "REGMAP 5 MISSING: no session"; exit 3; }
grep -qE "MCOLS=4 .*core 133.333 MHz" $D/selftest$PART.log || { echo "CONFIG NOT m4 133.33: no session"; exit 3; }
grep -q "ALL PASS" $D/selftest$PART.log || { echo "SELFTEST FAIL: no session"; exit 3; }
case $PART in all|A)
  cs; cd $H && env -u RUNS LOAD=0 ECC_EACH=1 OUT=$D/qual REFCACHE=$RC timeout 5400 bash $H/tools/qual/qual.sh $D fast > $D/qual_fast.log 2>&1
  echo "QUAL EXIT $? $(date +%T)"; tail -2 $D/qual_fast.log; grep -F "[FAIL]" $D/qual/checks.txt | head -8 | cut -c1-220
  grep -E "NOTE|DRAM turnarounds, ECC|ecc " $D/qual/checks.txt | head -6 | cut -c1-220; cs
  echo "=== qual E2B-it gemma4:int8:- $(date +%T)"
  cd $H && env -u RUNS RUNS="gemma4:int8:-" LONG="" LOAD=0 SOAK=60 ECC_EACH=1 OUT=$D/qual_e2b REFCACHE=$RC timeout 3600 bash $H/tools/qual/qual.sh $D fast > $D/qual_e2b.log 2>&1
  echo "QUAL E2B EXIT $? $(date +%T)"; tail -2 $D/qual_e2b.log; grep -F "[FAIL]" $D/qual_e2b/checks.txt 2>/dev/null | head -8 | cut -c1-220; cs
  Qn="What is the capital of France? Answer in one sentence."
  for m in gemma4-e4b gemma4 qwen35-4b lfm2-2.6b qwen35 lfm2-8b qwen35-35b gemma4-26b; do
    echo "=== otpu-chat $m $(date +%T)"
    cd /tmp && env PYTHONPATH=$H timeout 2400 $P -m opentpu.host.chat --backend board --model $m --greedy --max-new 24 --prompt "$Qn" > $D/chat-$m.txt 2>&1
    echo "CHAT $m EXIT $? $(date +%T)"; grep -vE "^\s*$" $D/chat-$m.txt | cut -c1-220 | tail -3
    echo "  hostr21: $(grep -oE "\(device [0-9.]+\) tok/s|[0-9.]+ Mcycles/token" $D/chat-$m.txt | tr "\n" " ")"
    echo "  prod:    $(grep -oE "\(device [0-9.]+\) tok/s|[0-9.]+ Mcycles/token" $PD/chat-$m.txt 2>/dev/null | tr "\n" " ")"
  done; cs
  ecc after-A ;;
esac
case $PART in all|B)
  CF=$(ls $D/chatmoe/*.pkl | head -1)
  echo "=== chatmoe lfm2-8b qwen35-35b gemma4-26b $(date +%T) (refcfg $CF, POOL_CALLS=1)"
  cd $H && env RUNS="lfm2-8b qwen35-35b gemma4-26b" RF=$D/chatmoe REFCFG=$CF POOL_CALLS=1 SESSION=chatmoe-hostr21-$(date +%H%M) \
    timeout 7200 bash tools/qual/chatmoe.sh > $D/chatmoe.log 2>&1
  echo "CHATMOE EXIT $? $(date +%T)"; grep -E "\[(PASS|FAIL|SKIP)\]|Error|cfg|COLD|load" $D/chatmoe.log | cut -c1-240
  echo "=== warm gate: otpu-serve qwen35-35b, its pool dropped from the page cache $(date +%T)"
  POOL=$C2/pool-q35-mx.split.bin; before=$(stat -c '%s %Y' $POOL; sha256sum $POOL.packed $POOL.key | cut -c1-16 | tr '\n' ' ')
  env PYTHONPATH=$H $P - $POOL <<'PY'
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)          # (read only: its pages out of the page cache)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
from opentpu.host.offload import PoolFile
p, n = sys.argv[1], os.path.getsize(sys.argv[1] + ".packed")
print(f"  pool: {PoolFile(p, os.path.getsize(p) // n, True, False).resident(range(n)) / 1e9:.2f} of "
      f"{os.path.getsize(p) / 1e9:.2f} GB in the page cache")
PY
  # (exec: $! is timeout's own PID, the leader of the server's process group; `cd ... && env ... &` gave a subshell's,
  # and killing it left the server holding /dev/xdma0: session hostr21, 2026-10-14)
  (cd /tmp && exec env PYTHONPATH=$H timeout 2400 $P -m opentpu.serve.api --model $C2/Qwen3.5-35B-A3B --backend board \
    --pool $POOL --port 8019 --kv-ram-gb 2 --kv-dir $D/kv) > $D/serve.log 2>&1 &
  sp=$!
  env PYTHONPATH=$H $P - <<'PY' > $D/serve-health.log 2>&1
import http.client, json, time
t0, last = time.time(), None
while time.time() - t0 < 1800:
    try:
        c = http.client.HTTPConnection("127.0.0.1", 8019, timeout=10)
        c.request("GET", "/health")
        r = c.getresponse()
        st, body = r.status, json.loads(r.read() or b"null")
    except (OSError, ValueError):
        st, body = None, None
    b = body if isinstance(body, dict) else {}
    k = (st, b.get("status"), b.get("warm"), round(b.get("progress") or 0, 1))
    if k != last:
        print(f"{time.time() - t0:7.1f} s: {st} {body}", flush=True)
        last = k
    if st == 200:
        break
    time.sleep(2)
PY
  cat $D/serve-health.log
  cd $H && env PYTHONPATH=$H timeout 1800 $P tools/serve_bench.py --port 8019 --skip throughput,conversation,users,arrivals \
    --check-prompts 2 --check-tokens 16 --out $D/serve-bench.json > $D/serve-bench.log 2>&1
  rb=$?; echo "SERVE_BENCH EXIT $rb $(date +%T)"; grep -E "warming|check:|did not come up" $D/serve-bench.log | cut -c1-200
  sleep 1; stop_tree $sp; xdma_free || ok_dev=0
  grep -E "^pool warm|^otpu-serve|Error|Traceback" $D/serve.log | cut -c1-220
  ok=${ok_dev:-1}
  grep -q ' 503 .*warming' $D/serve-health.log || { ok=0; echo "  warm gate: no 503 warming seen"; }
  grep -q ' 200 ' $D/serve-health.log || { ok=0; echo "  warm gate: no 200 after it"; }
  grep -q '^pool warm: done' $D/serve.log || { ok=0; echo "  warm gate: the warm did not end done"; }
  [ $rb = 0 ] && grep -q "check: PASS" $D/serve-bench.log || { ok=0; echo "  warm gate: serve_bench's check did not pass"; }
  echo "WARM GATE $([ $ok = 1 ] && echo PASS || echo FAIL) $(date +%T)"
  after=$(stat -c '%s %Y' $POOL; sha256sum $POOL.packed $POOL.key | cut -c1-16 | tr '\n' ' ')
  [ "$before" = "$after" ] && echo "  q35 pool unchanged: $after" || echo "  q35 pool CHANGED: $before -> $after (FAIL)"
  t0=$SECONDS; cat $POOL > /dev/null; echo "  q35 pool read back into the page cache in $((SECONDS - t0)) s"; cs
  ecc after-B ;;
esac
case $PART in all|C)
  echo "=== MiniCPM-V (mcpv3.sh warmup sliced thumb text, POOL_CALLS=1) $(date +%T)"
  cd $H && env RUNS="warmup sliced thumb text" POOL_CALLS=1 SESSION=mcpv-hostr21 timeout 5400 \
    bash $H/tools/offload/sessions/mcpv3.sh > $D/mcpv.log 2>&1
  echo "MCPV EXIT $? $(date +%T)"; grep -E "^===|\[(PASS|FAIL)\]|STOP|MCPV3|turn 1:" $D/mcpv.log | cut -c1-220; cs
  ecc after-C ;;
esac
ecc end-$PART
cd $H && env PYTHONPATH=$H timeout 600 $P $PD/scrub.py > $D/scrub$PART.log 2>&1; echo "SCRUB EXIT $? $(date +%T)"; tail -3 $D/scrub$PART.log
echo "=== selftest $(date +%T)"; cd /tmp && env PYTHONPATH=$H timeout 900 $P -m opentpu.host.selftest 2>&1 | tee $D/selftest2$PART.log | grep -E "config|bandwidth|ALL|FAIL"
echo "HOSTR21 SESSION END (PART $PART) $(date +%T)"
