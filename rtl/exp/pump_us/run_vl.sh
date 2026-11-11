#!/bin/bash
# EXPERIMENTAL (otpu_ppk.sv): Verilator checks of the pumped packed MXU test. From the tree's root:
#   rtl/exp/pump_us/run_vl.sh lint                    lint ppk_top (unisim stand-ins) and tb_ppk
#   rtl/exp/pump_us/run_vl.sh sim <workdir> [name ...]   build and run tb_ppk (a lab host)
# The simulation's DSP48E2 is the behavioural one (sim/dsp48e2_beh.sv); run_xsim.sh runs the same
# testbench on AMD's model. Names (default all):
#   n1 NCOL 1 nvfp4 | n1w8 int8 (den) | n1i4 int4 | n1h HOLD 1 | n1b BF 0 | n1hb HOLD 1 BF 0 (mxfp4)
#   n2 NCOL 2 | n8 NCOL 8 (the array) HOLD 1
set -e
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
E=$ROOT/rtl/exp/pump_us
mode=${1:?lint or sim}; shift
skew() {  # otpu_skew alone (otpu_mxu.sv needs the package and the fp files)
  sed -n '/^module otpu_skew/,/^endmodule/p' "$ROOT/rtl/mxu/otpu_mxu.sv" > "$1/otpu_skew.sv"
}
RTL="$ROOT/rtl/tech/xilinxus/otpu_tech_dsp.sv $E/otpu_ppk.sv $E/ppk_top.sv"
if [ "$mode" = lint ]; then
  T=$(mktemp -d)
  skew "$T"
  rc=0
  verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-PINCONNECTEMPTY \
    --top-module ppk_top "$T/otpu_skew.sv" $RTL "$E/sim/dsp48e2_beh.sv" "$E/sim/mmcme4_stub.sv" \
    "$ROOT/boards/bcu1525/sim/unisim_stubs.sv" -GNCOL=8 || rc=1
  verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-PINCONNECTEMPTY --timing \
    --top-module tb_ppk "$T/otpu_skew.sv" $RTL "$E/sim/dsp48e2_beh.sv" "$E/tb_ppk.sv" || rc=1
  rm -rf "$T"
  [ $rc = 0 ] && echo "LINT OK"
  exit $rc
fi
W=${1:?workdir}; shift
mkdir -p "$W"
skew "$W"
one() {   # name -G...
  local n=$1; shift
  rm -rf "${W:?}/$n"
  verilator --binary --timing -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL \
    --build-jobs 2 --x-assign 0 --x-initial 0 -O2 --top-module tb_ppk "$@" --Mdir "$W/$n" -o sim \
    "$W/otpu_skew.sv" $RTL "$E/sim/dsp48e2_beh.sv" "$E/tb_ppk.sv" > "$W/$n.build.log" 2>&1 \
    || { echo "BUILD FAILED $n (see $W/$n.build.log)"; tail -20 "$W/$n.build.log"; return 1; }
  "$W/$n/sim" > "$W/$n.log" 2>&1 || true
  grep -E "TB_RESULT|MISMATCH|Fatal|fatal|Error" "$W/$n.log" | head -12 | sed "s/^/$n: /"
}
names=${*:-n1 n1w8 n1i4 n1h n1b n1hb n2 n8}
for n in $names; do
  case $n in
    n1)   one n1   -GNCOL=1 -GWF=3 -GSEED=1 ;;
    n1w8) one n1w8 -GNCOL=1 -GWF=0 -GSEED=2 ;;
    n1i4) one n1i4 -GNCOL=1 -GWF=1 -GSEED=3 ;;
    n1h)  one n1h  -GNCOL=1 -GWF=3 -GHOLD=1 -GSEED=4 ;;
    n1b)  one n1b  -GNCOL=1 -GWF=3 -GBF=0 -GSEED=5 ;;
    n1hb) one n1hb -GNCOL=1 -GWF=2 -GHOLD=1 -GBF=0 -GSEED=6 ;;
    n2)   one n2   -GNCOL=2 -GWF=3 -GSEED=7 ;;
    n8)   one n8   -GNCOL=8 -GWF=0 -GHOLD=1 -GSEED=8 -GNCHUNK=120 ;;
    *) echo "unknown $n" ;;
  esac
done
