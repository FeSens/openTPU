#!/bin/bash
# c35.sh key|build [TREE [ARGS...]]: chat35.py's image of Qwen3.6-35B-A3B (CHAT35_KEY / CHAT35_BUILD) in TREE (default
# tree-TREE), ARGS otpu-chat's after chat35's own (otpu-serve35 --slots 2's image: --slots 2 --cap 131072)
# with the card run's OTPU_* environment (OTPU_TARGET bcu1525-wg:bhw16; the image cache img/ here).
D=$(cd "$(dirname "$0")" && pwd)
mode=$1; tree=${2:-tree-TREE}
cd "$D/$tree" || exit 2
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*'); do unset "$v"; done
export OTPU_TARGET=bcu1525-wg:bhw16 OTPU_IMG_CACHE=$D/img OTPU_IMG_CACHE_GB=100 PYTHONPATH=. \
  OTPU_IMAGE_CACHE_GB=${QCACHE_GB:-30}
case $mode in
  key) export CHAT35_KEY=1 ;;
  build) export CHAT35_BUILD=1 ;;
  *) echo "c35.sh key|build"; exit 2 ;;
esac
exec nice -n 10 "${BCU_PY:-python3}" -u "$D/chat35.py" "${@:3}"
