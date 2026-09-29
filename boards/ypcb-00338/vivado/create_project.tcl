# Create the Vivado project for openTPU on the YPCB-00338.
#   vivado -mode batch -source create_project.tcl -tclargs [DDR_SPEED] [OUT_DIR] [MCOLS] [CORE_MHZ] [BUILD_ID] [VPU_CL] [LANES] [ACT_ROWS] [DSTEP] [AXI_BL]
# DDR_SPEED: 800 (default), 1066, or the out-of-spec 1300, 1333, 1600 (docs/board.md). OUT_DIR: default ../../../build/vivado (repository build/).
# BUILD_ID: 8 hex digits for the BUILD_ID register (default: the first 8 hex digits of the
# repository's git commit, else 0).

set here [file normalize [file dirname [info script]]]
set root [file normalize $here/../../..]
set DDR_SPEED [expr {[llength $argv] > 0 ? [lindex $argv 0] : 800}]
set out [expr {[llength $argv] > 1 ? [file normalize [lindex $argv 1]] : "$root/build/vivado"}]
set MCOLS [expr {[llength $argv] > 2 ? [lindex $argv 2] : 2}]
set CORE_MHZ [expr {[llength $argv] > 3 ? [lindex $argv 3] : 100}]
set BUILD_ID [expr {[llength $argv] > 4 ? [lindex $argv 4] : ""}]
set VPU_CL [expr {[llength $argv] > 5 ? [lindex $argv 5] : 2}]
set LANES [expr {[llength $argv] > 6 ? [lindex $argv 6] : 8}]
set ACT_ROWS [expr {[llength $argv] > 7 ? [lindex $argv 7] : $MCOLS}]
# DSTEP 0: the DMA's DeltaNet head step left out (CAPS bit6 = 0; the compiler emits VOPs)
set DSTEP [expr {[llength $argv] > 8 ? [lindex $argv 8] : 1}]
# AXI_BL: the accelerator's port B read burst, beats (8 default, up to 64; bd.tcl's MAX_BURST_LENGTH follows)
set AXI_BL [expr {[llength $argv] > 9 ? [lindex $argv 9] : 8}]
if {$BUILD_ID eq ""} {
  if {[catch {exec git -C $root rev-parse HEAD} BUILD_ID]} { set BUILD_ID 0 }
  set BUILD_ID [string range $BUILD_ID 0 7]
}
if {![regexp {^[0-9a-fA-F]{1,8}$} $BUILD_ID]} { set BUILD_ID 0 }
# the core clock the block design makes (bd.tcl: CORE_MHZ rounded to the MMCM's 1/8 divider
# steps of its VCO, 1000 MHz for DDR3-1333, else 800), for the CORE_KHZ register
set VCO [expr {$DDR_SPEED == 1333 ? 1000 : 800}]
set CORE_KHZ [expr {round($VCO * 1000.0 / (round(double($VCO) / $CORE_MHZ * 8) / 8.0))}]
puts "CORE_KHZ $CORE_KHZ, BUILD_ID $BUILD_ID"
set MIG_DIR $here/mig

if {![file exists $MIG_DIR/mig_ddr3_ch0.prj]} {
  error "run boards/ypcb-00338/scripts/gen_mig_prj.py first (it writes vivado/mig/*.prj)"
}

create_project -force otpu $out -part xc7k480tffg1156-2
set_property target_language Verilog [current_project]
set_property default_lib xil_defaultlib [current_project]

# ---- RTL (SystemVerilog)
set rtl [list \
  rtl/vpu/otpu_fp.sv rtl/vpu/otpu_fpipe.sv rtl/top/otpu_pkg.sv \
  rtl/mem/otpu_tmem.sv rtl/mem/otpu_axi_dram.sv rtl/mem/otpu_actram.sv \
  rtl/seq/otpu_seq.sv rtl/vpu/otpu_vtree.sv rtl/dma/otpu_dma.sv \
  rtl/mxu/otpu_mxu.sv \
  rtl/vpu/otpu_quant.sv rtl/vpu/otpu_se_comp.sv rtl/vpu/otpu_se_tail.sv rtl/vpu/otpu_vpu.sv rtl/top/otpu_coll.sv \
  rtl/top/otpu_slice.sv \
  rtl/boards/ypcb-00338/otpu_ctrl.sv rtl/boards/ypcb-00338/otpu_trace.sv \
  rtl/boards/ypcb-00338/otpu_board.sv rtl/boards/ypcb-00338/otpu_fpga_top.sv]
foreach f $rtl {
  add_files -norecurse $root/$f
  set_property file_type SystemVerilog [get_files $root/$f]
}
# SYNTHESIS removes the simulation-only checks and dumps
set_property verilog_define {SYNTHESIS} [get_filesets sources_1]

# ---- block design
source $here/bd.tcl
make_wrapper -files [get_files otpu_bd.bd] -top
add_files -norecurse [glob $out/otpu.gen/sources_1/bd/otpu_bd/hdl/otpu_bd_wrapper.v]
set_property top otpu_fpga_top [current_fileset]
set_property generic "MCOLS=$MCOLS ACT_ROWS=$ACT_ROWS VPU_CL=$VPU_CL LANES=$LANES CORE_KHZ=$CORE_KHZ BUILD_ID=32'h$BUILD_ID DDR_MTS=$DDR_SPEED DSTEP=1'b$DSTEP AXI_BL=$AXI_BL" [current_fileset]

# ---- constraints
add_files -fileset constrs_1 -norecurse [list \
  $root/boards/ypcb-00338/constraints/otpu_top.xdc \
  $root/boards/ypcb-00338/constraints/otpu_ddr3_pins.xdc]

# ---- strategies: timing-driven, the accelerator is the critical part
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]

puts "project created in $out (DDR3-$DDR_SPEED)"
