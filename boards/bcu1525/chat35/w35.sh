#!/bin/bash
# w35.sh [N] [TREE]: warm35.py (every bucket's programs of the 35B chat image) with the chat's keyed environment (OTPU_TARGET
# bcu1525-wg:bhw16: otpu-chat35's) into qprog35/ here, in N processes (default 4), from TREE (default
# tree-TREE); the arguments from warm35.pkl here (a chat's). The cache then goes to the card host's cardimg/qprog35.
D=$(cd "$(dirname "$0")" && pwd)
n=${1:-4}; tree=${2:-tree-TREE}
cd "$D/$tree" || exit 2
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*'); do unset "$v"; done
export OTPU_TARGET=bcu1525-wg:bhw16 OTPU_PROG_CACHE=$D/qprog35 OTPU_PROG_CACHE_GB=8 PYTHONPATH=.
mkdir -p "$D/qprog35"
exec "${BCU_PY:-python3}" -u "$D/warm35.py" "$n"
