#!/bin/bash
# Formal equivalence of the board's glue at one slice: otpu_board + otpu_ctrl at git revision REV
# against the worktree's, with the YPCB build's parameters, the units they instantiate (the slice,
# the DRAM adapter, the collective unit, the trace buffer) cut out as black boxes and their ports
# turned into the module's (yosys expose -evert), then equiv_make / equiv_simple / equiv_induct.
# A change that keeps S = 1 the same logic proves equal; one that changes it leaves unproven
# $equiv cells (exit 1).
#   usage: tools/synth/board_equiv.sh [REV (default HEAD)] [-G P=V ...]
Y=${YOSYS:-$HOME/bonetto/riscv-autoarch/.toolchain/oss-cad-suite/bin/yosys}
W=$(cd "$(dirname "$0")/../.." && pwd)
rev=${1:-HEAD}; shift || true
d=$(mktemp -d "${TMPDIR:-/tmp}/board_equiv.XXXXXX")
trap 'rm -rf "$d"' EXIT
git -C "$W" show "$rev:rtl/shell/otpu_board.sv" > "$d/old_board.sv" || exit 2
git -C "$W" show "$rev:rtl/shell/otpu_ctrl.sv" > "$d/old_ctrl.sv" || exit 2
R=$W/rtl
LIB="$R/top/otpu_pkg.sv $R/vpu/otpu_fp.sv $R/vpu/otpu_fpipe.sv $R/mem/otpu_tmem.sv $R/mem/otpu_actram.sv
     $R/seq/otpu_seq.sv $R/vpu/otpu_vtree.sv $R/dma/otpu_dma.sv $R/mxu/otpu_mxu.sv
     $R/tech/generic/otpu_tech_dsp.sv $R/tech/generic/otpu_tech_ram.sv $R/vpu/otpu_quant.sv $R/vpu/otpu_se_comp.sv
     $R/vpu/otpu_se_tail.sv $R/vpu/otpu_vpu.sv $R/top/otpu_coll.sv $R/top/otpu_slice.sv
     $R/shell/otpu_trace.sv $R/mem/otpu_native_dram.sv $R/mem/otpu_xstg.sv"
BB="--blackboxed-module otpu_slice --blackboxed-module otpu_native_dram --blackboxed-module otpu_coll
    --blackboxed-module otpu_trace"
G="-G MCOLS=4 -G ACT_ROWS=4 -G MXU_IMPL=2 -G VPU_CL=2 -G ULANES=8 -G PLATFORM=1 $*"
LIB=$(echo $LIB); BB=$(echo $BB)    # one line each: a newline ends a yosys command
CUT="expose -evert otpu_board/c:u_slice otpu_board/c:u_mem otpu_board/c:*u_coll otpu_board/c:*u_trace"
PREP="hierarchy -top otpu_board; proc; memory; flatten; $CUT; opt_clean"
$Y -m slang -p "
read_slang $LIB $d/old_ctrl.sv $d/old_board.sv --top otpu_board $G $BB -DSYNTHESIS --allow-use-before-declare
$PREP; rename otpu_board gold; design -stash gold
read_slang $LIB $R/shell/otpu_ctrl.sv $R/shell/otpu_board.sv --top otpu_board $G $BB -DSYNTHESIS --allow-use-before-declare
$PREP; rename otpu_board gate; design -stash gate
design -copy-from gold -as gold gold
design -copy-from gate -as gate gate
equiv_make gold gate equiv; hierarchy -top equiv
equiv_simple -seq 5; equiv_induct; equiv_status -assert" > "$d/equiv.log" 2>&1
rc=$?
grep -E "Found [0-9]+ \\\$equiv cells|are proven|ERROR" "$d/equiv.log" | tail -3
[ $rc = 0 ] && echo "board_equiv: otpu_board + otpu_ctrl at $rev and the worktree are equivalent" \
  || echo "board_equiv: NOT equivalent (or failed), exit $rc"
exit $rc
