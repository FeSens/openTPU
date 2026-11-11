#!/bin/bash
# b35x2.sh [TREE [ARGS...]]: c35x2.sh build (the two-card 262,144-position Qwen3.6-35B-A3B chat image; with ARGS
# otpu-chat's, e.g. otpu-serve35x2 --slots 2's: tree-SHA --slots 2 --cap 131072) into img/ (logs/b35x2.out), its
# process group killed if MemAvailable falls under MINGB (default 8) GB (its peak RSS logged); logs/b35x2.exit at the end
# (LOG=NAME: logs/NAME.out and logs/NAME.exit, e.g. the slotted build's b35x2s).
D=$(cd "$(dirname "$0")" && pwd)
min=${MINGB:-8}
mkdir -p "$D/logs"
cd "$D" || exit 2
echo "== b35x2 $(date +%T)"
n=${LOG:-b35x2}
setsid ./c35x2.sh build "$@" > logs/$n.out 2>&1 &
p=$!
low=0; peak=0
while kill -0 $p 2>/dev/null; do
  a=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  r=$(ps -o rss= -g $p 2>/dev/null | awk '{s += $1} END {print int(s / 1048576)}')
  [ "${r:-0}" -gt $peak ] && peak=$r
  if [ "$a" -lt "$min" ]; then echo "$(date +%T) MemAvailable $a GB < $min: killing the build"; kill -9 -- -$p; low=1; fi
  sleep 5
done
wait $p; rc=$?
echo "rc $rc low $low peak RSS $peak GB $(date +%T)"
echo "$rc" > logs/$n.exit
