#!/bin/bash
# Hugging Face's greedy tokens for card runs (bf16 on the CPU, moe_card --hf): the prompt and the
# 8 best logits a step, on a host with the whole checkpoint (the 26B's: omarchy, 8 GiB of weights
# in RAM and the rest from disk). A checkpoint without a chat template takes the prompt as plain
# text (the 26B's download is the base model).
#   hf_reference.sh MODEL OUT.json N [PROMPT_FILE]    (MODEL a checkpoint directory; OUT in O;
#   no PROMPT_FILE: moe_card's question, as the 35B's and LFM2.5's references)
# e.g. session 8's: hf_reference.sh ~/openTPU/models/gemma-4-26B-A4B q26-hf.json 16 \
#   tools/offload/sessions/wiki1.txt   (the first paragraph of the router traces' wiki text)
# MAXMEM of weights in RAM (default 8GiB), under a MEMMAX memory limit (default 12G) at nice 19; killed, not frozen, on two 15 s samples
# under MEMFLOOR GB of MemAvailable (default 6) or DISKFLOOR GB free (default 41); the
# accelerate offload folder removed after.
set -u
here=$(cd "$(dirname "$0")" && pwd)
T=${T:-$(cd "$here/../../.." && pwd)}; O=${O:-$HOME/otpu-build/offload/card2}
model=$1 out=$2 n=$3 prompt=${4:-}
cd "$T" || exit 1
export PATH=${OTPU_VENV:-$HOME/otpu-venv}/bin:$PATH PYTHONPATH=$T OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mem() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
disk() { df -BG --output=avail ~ | tail -1 | tr -dc 0-9; }
echo "hf_reference start $(date +%T) mem $(mem) GB disk $(disk) GB"
lim=()
command -v systemd-run > /dev/null && lim=(systemd-run --user --scope -p MemoryMax=${MEMMAX:-12G} -p MemorySwapMax=0 --quiet)
"${lim[@]}" nice -n 19 python tools/offload/moe_card.py $model --hf $O/$out -n $n \
  --max-memory ${MAXMEM:-8GiB} ${prompt:+--prompt "$(cat $prompt)"} &
pid=$!; low=0
while kill -0 $pid 2>/dev/null; do
  m=$(mem); d=$(disk)
  if [ "$m" -lt ${MEMFLOOR:-6} ] || [ "$d" -lt ${DISKFLOOR:-41} ]; then low=$((low+1)); else low=0; fi
  if [ $low -ge 2 ]; then pkill -P $pid; kill $pid; echo "KILLED $(date +%T) mem $m GB disk $d GB"; break; fi
  sleep 15
done
wait $pid; echo "hf_reference exit $? $(date +%T)"
rm -rf "$(dirname $model)/.offload-moe-card"
