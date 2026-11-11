#!/bin/bash
# EXPERIMENTAL (otpu_ppk.sv): tb_ppk in xsim on AMD's unisim DSP48E2 (the check of the behavioural
# model run_vl.sh uses). Vivado's simulator, no license needed; on the build host. Usage,
# from the tree's root:
#   rtl/exp/pump_us/run_xsim.sh <workdir> [NCOL [HOLD [BF [WF]]]]     (defaults 2 0 1 3)
set -e
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
E=$ROOT/rtl/exp/pump_us
W=${1:?workdir}
NCOL=${2:-2}; HOLD=${3:-0}; BF=${4:-1}; WF=${5:-3}
N=n${NCOL}h${HOLD}b${BF}w${WF}
mkdir -p "$W" && cd "$W"
export PATH=$HOME/.local/bin:/opt/Xilinx/2026.1/Vivado/bin:$PATH
XV=$(dirname "$(readlink -f "$(command -v vivado)")")/..
[ -f "$XV/data/verilog/src/glbl.v" ] || XV=/opt/Xilinx/2026.1/Vivado
# otpu_skew alone (xvlog does not take otpu_mxu.sv's use-before-declaration)
sed -n '/^module otpu_skew/,/^endmodule/p' "$ROOT/rtl/mxu/otpu_mxu.sv" > otpu_skew.sv
xvlog -sv otpu_skew.sv "$ROOT/rtl/tech/xilinxus/otpu_tech_dsp.sv" "$E/otpu_ppk.sv" \
  "$E/ppk_top.sv" "$E/tb_ppk.sv" > xvlog.log
xvlog "$XV/data/verilog/src/glbl.v" >> xvlog.log
# xsim's front end links libncurses.so.5 / libtinfo.so.5; a host with only ncurses 6 (Arch)
# gets local links to the 6 libraries
if ! ldconfig -p 2>/dev/null | grep -q libncurses.so.5; then
  mkdir -p lib5
  ln -sf /usr/lib/libncursesw.so.6 lib5/libncurses.so.5
  ln -sf /usr/lib/libtinfo.so.6 lib5/libtinfo.so.5
  export LD_LIBRARY_PATH=$PWD/lib5${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
# the unisims' GSR holds the DSPs' registers for the first 100 ns: tb_ppk's first chunk comes at
# cycle 40 of 4 ns
xelab -L unisims_ver --timescale 1ns/1ps -generic_top "NCOL=$NCOL" -generic_top "HOLD=$HOLD" \
  -generic_top "BF=$BF" -generic_top "WF=$WF" -s "sim_$N" tb_ppk glbl > "xelab_$N.log"
xsim "sim_$N" -R > "xsim_$N.log"
grep -E "TB_RESULT|MISMATCH" "xsim_$N.log" | head -12
