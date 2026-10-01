#!/bin/bash
# The ISA simulator's reference for card runs (docs/offload.md 10): moe_card on the simulator with
# the card's generate loop, slots and configuration (REFCFG: tools/qual/refs.py cfg on the card's
# host), on a host with the whole pool (the 26B's: 77 min on omarchy; its prefill steps a token at
# a time). Its prefill logits' sha256 and its tokens are what card_moe.sh checks a run against.
#   reference.sh MODEL POOL HF.json N OUT.json [moe_card options]
# e.g. session 8's: reference.sh gemma-4-26B-A4B ../g26/pool-g26-fp4.split.bin q26-hf.json 16 \
#   q26ref16.json --wformat int8 --formats experts=fp4 --head-format fp4   (paths in O)
# Started at MEMSTART GB of MemAvailable (default 16), under a MEMMAX memory limit (default 14G,
# systemd-run --user --scope where there is one) at nice 19; killed, not frozen, on two 30 s
# samples under MEMFLOOR GB (default 6). Log: O/OUT.log.
set -u
here=$(cd "$(dirname "$0")" && pwd)
T=${T:-$(cd "$here/../../.." && pwd)}; O=${O:-$HOME/otpu-build/offload/card2}
REFCFG=${REFCFG:-$O/cfg-c2830d6a.pkl}
model=$1 pool=$2 hf=$3 n=$4 out=$5; shift 5
cd "$T" || exit 1
export PATH=${OTPU_VENV:-$HOME/otpu-venv}/bin:$PATH PYTHONPATH=$T OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mem() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
exec > >(tee -a $O/$out.log) 2>&1
while [ $(mem) -lt ${MEMSTART:-16} ]; do sleep 60; done
echo "reference start $(date +%T) mem $(mem) GB: $model $out"
lim=()
command -v systemd-run > /dev/null && lim=(systemd-run --user --scope -p MemoryMax=${MEMMAX:-14G} -p MemorySwapMax=0 --quiet)
"${lim[@]}" nice -n 19 python tools/offload/moe_card.py $O/$model --check $O/$hf -n $n \
  --pool $O/$pool --cfg $REFCFG --experts 0 "$@" --out $O/$out &
pid=$!; low=0
while kill -0 $pid 2>/dev/null; do
  m=$(mem)
  if [ "$m" -lt ${MEMFLOOR:-6} ]; then low=$((low+1)); else low=0; fi
  if [ $low -ge 2 ]; then pkill -P $pid; kill $pid; echo "KILLED $(date +%T) mem $m GB"; break; fi
  sleep 30
done
wait $pid; echo "reference exit $? $(date +%T)"
