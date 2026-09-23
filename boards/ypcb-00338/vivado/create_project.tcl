# Create the Vivado project for openTPU on the YPCB-00338.
#   vivado -mode batch -source create_project.tcl -tclargs [DDR_SPEED] [OUT_DIR]
# DDR_SPEED: 800 (default) or 1066. OUT_DIR: default ../../../build/vivado (repository build/).

set here [file normalize [file dirname [info script]]]
set root [file normalize $here/../../..]
set DDR_SPEED [expr {[llength $argv] > 0 ? [lindex $argv 0] : 800}]
set out [expr {[llength $argv] > 1 ? [file normalize [lindex $argv 1]] : "$root/build/vivado"}]
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
  rtl/seq/otpu_seq.sv rtl/dma/otpu_dma.sv rtl/mxu/otpu_mxu.sv \
  rtl/vpu/otpu_quant.sv rtl/vpu/otpu_vpu.sv rtl/top/otpu_coll.sv rtl/top/otpu_slice.sv \
  rtl/boards/ypcb-00338/otpu_ctrl.sv rtl/boards/ypcb-00338/otpu_board.sv \
  rtl/boards/ypcb-00338/otpu_fpga_top.sv]
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

# ---- constraints
add_files -fileset constrs_1 -norecurse [list \
  $root/boards/ypcb-00338/constraints/otpu_top.xdc \
  $root/boards/ypcb-00338/constraints/otpu_ddr3_pins.xdc]

# ---- strategies: timing-driven, the accelerator is the critical part
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]

puts "project created in $out (DDR3-$DDR_SPEED)"
