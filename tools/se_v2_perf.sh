#!/usr/bin/env bash
# SE v2: one decode token per model on the RTL, v1 against v2 (OTPU_SE=v2: COMP8 and ONE_TREE)
# with tools/perf_qwen.py at the checkpoint configuration (OTPU_DSTEP=1 OTPU_PAIR=1, the
# DDR3-1066 bank model, 120 MHz); the v2 runs use --check (bit-exact with the ISA simulator).
# Full outputs in perf_se_v2/.
# Usage: tools/se_v2_perf.sh [model:layers ...]    default: qwen35:4 lfm2:2 qwen3:2
set -u
runs=${*:-qwen35:4 lfm2:2 qwen3:2}
mkdir -p perf_se_v2
for r in $runs; do
  m=${r%%:*}; L=${r##*:}
  for se in v1 v2; do
    f=perf_se_v2/$m-L$L-$se.txt
    chk=""; [[ $se == v2 ]] && chk="--check"
    OTPU_DSTEP=1 OTPU_PAIR=1 OTPU_SE=$se \
      python tools/perf_qwen.py --model $m --layers $L --ddr 1066 --mhz 120 $chk > $f 2>&1
    echo "=== $m layers $L $se (exit $?)"
    grep -E "^layers=|ms/token|bit-exact|Traceback|Error|^  (VPU|vpu|U_VPU|DMA|MXU|Q) " $f | head -12
  done
done
