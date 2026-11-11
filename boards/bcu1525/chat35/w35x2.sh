#!/bin/bash
# w35x2.sh [N] [TREE] [LB]: warm35x2.py (every bucket's programs of the two-card 35B chat image; with LB, the layer-major
# prompt's runs of buckets 1 .. LB) with the chat's keyed
# environment (OTPU_TARGET bcu1525-x2:st16: otpu-chat35x2's) into qprog/ here, in N processes (default 4), from TREE
# (default tree); the arguments from warm35x2.pkl here (a chat's; AHEAD=1: written first with no card, chat35x2.py
# CHAT35_AHEAD, its sampler the chat's default). The cache then goes to the card host's cardimg/x2/qprog.
D=$(cd "$(dirname "$0")" && pwd)
n=${1:-4}; tree=${2:-tree}; lb=${3:-0}
cd "$D/$tree" || exit 2
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*'); do unset "$v"; done
export OTPU_TARGET=bcu1525-x2:st16 OTPU_PROG_CACHE=$D/qprog OTPU_PROG_CACHE_GB=8 PYTHONPATH=.
mkdir -p "$D/qprog"
if [ "${AHEAD:-0}" = 1 ]; then
  CHAT35_AHEAD=1 "${BCU_PY:-python3}" -u "$D/chat35x2.py" < /dev/null || exit 1
fi
exec "${BCU_PY:-python3}" -u "$D/warm35x2.py" "$n" "$lb"
