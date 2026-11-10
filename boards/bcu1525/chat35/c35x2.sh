#!/bin/bash
# c35x2.sh key|build [TREE [ARGS...]]: chat35x2.py's image of Qwen3.6-35B-A3B for both BCU1525s (CHAT35_KEY /
# CHAT35_BUILD) in TREE (default tree: a link to the shipped tree), ARGS otpu-chat's after chat35x2's own (otpu-serve35x2
# --slots 2's image: --slots 2 --cap 131072), with otpu-chat35x2's OTPU_* environment: OTPU_TARGET bcu1525-x2:st16 alone
# (no OTPU_DEV: the cards' names go to the chat as an argument, so the key holds none); the image cache img/ here.
D=$(cd "$(dirname "$0")" && pwd)
mode=$1; tree=${2:-tree}
shift; shift
cd "$D/$tree" || exit 2
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*'); do unset "$v"; done
export OTPU_TARGET=bcu1525-x2:st16 OTPU_IMG_CACHE=$D/img OTPU_IMG_CACHE_GB=60 PYTHONPATH=. \
  OTPU_IMAGE_CACHE_GB=${QCACHE_GB:-30}
case $mode in
  key) export CHAT35_KEY=1 ;;
  build) export CHAT35_BUILD=1 ;;
  *) echo "c35x2.sh key|build"; exit 2 ;;
esac
exec nice -n 10 "${BCU_PY:-python3}" -u "$D/chat35x2.py" "$@"
