#!/bin/bash
# Synthesize one module for the xc7 fabric with yosys (+slang) and report its size and the
# longest logic path. usage: sta.sh <outdir> <top> "<-G P=V ...>" <files...>
# Prints: <top> arrival=<ps> luts=.. ff=.. dsp=.. bram36=.. bram18=.. lutram=..
# The arrival time is logic only (cell delays from the Xilinx specify models, no routing), from
# any register or input to any register or output.
Y=${YOSYS:-$HOME/bonetto/riscv-autoarch/.toolchain/oss-cad-suite/bin/yosys}
out=$1; top=$2; params=$3; shift 3
mkdir -p "$out"
s="$out/$top"
$Y -m slang -p "read_slang $* $params --top $top -DSYNTHESIS; synth_xilinx -family xc7 -flatten -abc9 -noiopad; read_verilog -lib -specify +/xilinx/cells_sim.v +/xilinx/cells_xtra.v; tee -q -o $s.stat stat; tee -q -o $s.sta sta" > $s.ylog 2>&1
arr=$(grep -m1 "Latest arrival" $s.sta | sed 's/.* is //; s/://')
# stat lists the cells twice (module, then design totals): count the first listing only
first=$(sed "/=== design hierarchy ===/q" $s.stat)
cnt() { echo "$first" | grep -E "^ +[0-9]+ +($1)\$" | awk '{s+=$1} END {print s+0}'; }
echo "$top arrival=${arr:-?} luts=$(cnt 'LUT[1-6]') ff=$(cnt 'FD[A-Z]*') dsp=$(cnt DSP48E1)" \
     "bram36=$(cnt RAMB36E1) bram18=$(cnt RAMB18E1) lutram=$(cnt 'RAM[0-9A-Z]+|SRL[A-Z0-9]*')"
