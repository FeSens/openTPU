#!/bin/bash
# EXPERIMENTAL (docs/mx_mxu.md): tb_mx (mx_top, NCOL 2, E 2) in xsim on the unisim models, for
# one family: x7 (pumped DSP48E1s, MMCME2_ADV) or us (packed DSP48E2s). Vivado's simulator: on
# the build host only (never on a laptop). Usage, from the tree's root:
#   rtl/exp/mx_mxu/run_xsim.sh <workdir> x7|us
set -e
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
W=${1:?workdir}
FAM=${2:?x7 or us}
mkdir -p "$W" && cd "$W"
XV=$(dirname "$(readlink -f "$(command -v vivado)")")/..
[ -f "$XV/data/verilog/src/glbl.v" ] || XV=/opt/Xilinx/2026.1/Vivado
# otpu_skew alone (xvlog does not take otpu_mxu.sv's use-before-declaration)
sed -n '/^module otpu_skew/,/^endmodule/p' "$ROOT/rtl/mxu/otpu_mxu.sv" > otpu_skew.sv
if [ "$FAM" = x7 ]; then
  F="$ROOT/rtl/tech/xilinx7/otpu_tech_dsp.sv $ROOT/rtl/exp/dsp_pump/otpu_pump_col.sv $ROOT/rtl/exp/mx_mxu/otpu_mx_x7.sv"
else
  F="$ROOT/rtl/tech/xilinxus/otpu_tech_dsp.sv $ROOT/rtl/exp/mx_mxu/otpu_mx_us.sv"
fi
xvlog -sv "$ROOT/rtl/vpu/otpu_fp.sv" "$ROOT/rtl/vpu/otpu_fpipe.sv" otpu_skew.sv $F \
  "$ROOT/rtl/exp/mx_mxu/otpu_mx_lane.sv" "$ROOT/rtl/exp/mx_mxu/mx_top.sv" \
  "$ROOT/rtl/exp/mx_mxu/tb_mx.sv" > xvlog_$FAM.log
xvlog "$XV/data/verilog/src/glbl.v" >> xvlog_$FAM.log
# xsim's front end links libncurses.so.5 / libtinfo.so.5; a host with only ncurses 6 (the build host,
# Arch) gets local links to the 6 libraries
if ! ldconfig -p 2>/dev/null | grep -q libncurses.so.5; then
  mkdir -p lib5
  ln -sf /usr/lib/libncursesw.so.6 lib5/libncurses.so.5
  ln -sf /usr/lib/libtinfo.so.6 lib5/libtinfo.so.5
  export LD_LIBRARY_PATH=$PWD/lib5${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
xelab -L unisims_ver -s sim_$FAM tb_mx glbl > xelab_$FAM.log
xsim sim_$FAM -R > xsim_$FAM.log
grep -E "TB_RESULT|MISMATCH" xsim_$FAM.log | head -30
