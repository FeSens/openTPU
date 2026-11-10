#!/bin/bash
# b35.sh [TREE [ARGS...]]: c35.sh build (the 262,144-position Qwen3.6-35B-A3B chat image) into img/ (logs/b35.out), its process group
# killed if MemAvailable falls under MINGB (default 8) GB; logs/b35.exit at the end.
D=$(cd "$(dirname "$0")" && pwd)
min=${MINGB:-8}
mkdir -p "$D/logs"
cd "$D" || exit 2
echo "== b35 $(date +%T)"
setsid ./c35.sh build "$@" > logs/b35.out 2>&1 &
p=$!
low=0
while kill -0 $p 2>/dev/null; do
  a=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  if [ "$a" -lt "$min" ]; then echo "$(date +%T) MemAvailable $a GB < $min: killing the build"; kill -9 -- -$p; low=1; fi
  sleep 5
done
wait $p; rc=$?
echo "rc $rc low $low $(date +%T)"
echo "$rc" > logs/b35.exit
