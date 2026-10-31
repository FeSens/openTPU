# Create the Vivado project for openTPU on the SQRL FK33 (xcvu33p-fsvh2104-2-e; docs/sqrl-fk33.md): the HBM IP (hbm_0,
# both stacks, 8 GB; rtl/boards/sqrl-fk33/otpu_hbm.sv instantiates it), the block design (bd.tcl: XDMA, the clocks,
# the SYSMON) and the board top (rtl/boards/sqrl-fk33/otpu_fpga_top_fk33.sv: one slice on two HBM channels).
#   vivado -mode batch -source create_project.tcl -tclargs OUT_DIR MCOLS CORE_MHZ BUILD_ID VPU_CL LANES ACT_ROWS DSTEP
#     PLATFORM ARCH_HASH
# (run_vivado.sh takes them from the target, targets/sqrl-fk33.toml; the RTL list below is opentpu.target
# sources("sqrl-fk33")). Environment: UCLK_MHZ (default 250), OTPU_FAST (Vivado's default strategies), OTPU_IP_CACHE, TMEM_URAM (default 1:
# the TMEM in UltraRAM, boards/sqrl-fk33/vivado/tmem_uram.xdc).
# Built and run with Vivado 2019.2 (the VU33P needs a license that covers it; the HBM IP 1.0, XDMA 4.1).

set here [file normalize [file dirname [info script]]]
set root [file normalize $here/../../..]
proc hexarg {i} {
  if {[llength $::argv] > $i} { return [lindex $::argv $i] }
  return ""
}
set out [expr {[llength $argv] > 0 ? [file normalize [lindex $argv 0]] : "$root/build/vivado_sqrl-fk33"}]
set MCOLS [expr {[llength $argv] > 1 ? [lindex $argv 1] : 4}]
set CORE_MHZ [expr {[llength $argv] > 2 ? [lindex $argv 2] : 250}]
set BUILD_ID [hexarg 3]
set VPU_CL [expr {[llength $argv] > 4 ? [lindex $argv 4] : 2}]
set LANES [expr {[llength $argv] > 5 ? [lindex $argv 5] : 8}]
set ACT_ROWS [expr {[llength $argv] > 6 ? [lindex $argv 6] : $MCOLS}]
set DSTEP [expr {[llength $argv] > 7 ? [lindex $argv 7] : 1}]
set PLATFORM [expr {[llength $argv] > 8 ? [lindex $argv 8] : 5}]
set ARCH_HASH [hexarg 9]
if {$ARCH_HASH eq ""} { set ARCH_HASH 0 }
if {![regexp {^[0-9a-fA-F]{1,8}$} $ARCH_HASH]} { error "ARCH_HASH $ARCH_HASH: 1 to 8 hex digits" }
if {$BUILD_ID eq ""} {
  if {[catch {exec git -C $root rev-parse HEAD} BUILD_ID]} { set BUILD_ID 0 }
  set BUILD_ID [string range $BUILD_ID 0 7]
}
if {![regexp {^[0-9a-fA-F]{1,8}$} $BUILD_ID]} { error "BUILD_ID $BUILD_ID: 1 to 8 hex digits" }
set PART xcvu33p-fsvh2104-2-e
# uclk: the HBM's AXI ports, its adapters and otpu_mem_ch's controller side (UCLK_MHZ in the environment)
set UCLK_MHZ [expr {[info exists ::env(UCLK_MHZ)] && $::env(UCLK_MHZ) ne "" ? $::env(UCLK_MHZ) : 250}]
puts "BUILD_ID $BUILD_ID, PLATFORM $PLATFORM, ARCH_HASH $ARCH_HASH, core $CORE_MHZ MHz, uclk $UCLK_MHZ MHz"

create_project -force otpu $out -part $PART
if {[info exists ::env(OTPU_IP_CACHE)] && $::env(OTPU_IP_CACHE) ne ""} {
  file mkdir $::env(OTPU_IP_CACHE)
  config_ip_cache -use_cache_location $::env(OTPU_IP_CACHE)
}
set_property target_language Verilog [current_project]

# ---- RTL: the core and its UltraScale+ cells, the shell, the board's system (otpu_fk33_sys), the HBM path, the top
set rtl [list \
  rtl/vpu/otpu_fp.sv rtl/vpu/otpu_fpipe.sv rtl/top/otpu_pkg.sv \
  rtl/mem/otpu_tmem.sv rtl/mem/otpu_actram.sv \
  rtl/seq/otpu_seq.sv rtl/vpu/otpu_vtree.sv rtl/dma/otpu_dma.sv \
  rtl/mxu/otpu_mxu.sv rtl/tech/xilinxus/otpu_tech_dsp.sv rtl/tech/generic/otpu_tech_ram.sv \
  rtl/vpu/otpu_quant.sv rtl/vpu/otpu_se_comp.sv rtl/vpu/otpu_se_tail.sv rtl/vpu/otpu_vpu.sv \
  rtl/top/otpu_coll.sv rtl/top/otpu_slice.sv \
  rtl/shell/otpu_ctrl.sv rtl/shell/otpu_trace.sv \
  rtl/shell/otpu_board.sv rtl/mem/otpu_native_dram.sv \
  rtl/shell/otpu_afifo.sv rtl/shell/otpu_mem_ch.sv \
  rtl/shell/otpu_axi_split2.sv rtl/boards/sqrl-fk33/otpu_fk33_sys.sv \
  rtl/boards/sqrl-fk33/otpu_hbm_port.sv rtl/boards/sqrl-fk33/otpu_hbm.sv \
  rtl/boards/sqrl-fk33/otpu_fpga_top_fk33.sv]
foreach f $rtl {
  add_files -norecurse $root/$f
  set_property file_type SystemVerilog [get_files $root/$f]
}
set_property verilog_define {SYNTHESIS} [get_filesets sources_1]

# ---- the HBM IP: both stacks (8 GB), global switches on, all 32 AXI ports (the top ties off the unused ones), the
# AXI ports on uclk, the reference and APB clocks 100 MHz; temperature-compensated refresh (2019.2's default is off)
create_ip -name hbm -vendor xilinx.com -library ip -version 1.0 -module_name hbm_0
set hbm [get_ips hbm_0]
set_property -dict [list CONFIG.USER_HBM_DENSITY {8GB} CONFIG.USER_HBM_STACK {2} CONFIG.USER_MEMORY_DISPLAY {8192}] $hbm
set cfg [list CONFIG.USER_SWITCH_ENABLE_00 {TRUE} CONFIG.USER_SWITCH_ENABLE_01 {TRUE} \
  CONFIG.USER_HBM_REF_CLK_0 {100} CONFIG.USER_HBM_REF_CLK_1 {100} \
  CONFIG.USER_APB_PCLK_0 {100} CONFIG.USER_APB_PCLK_1 {100} CONFIG.USER_MC_ENABLE_APB_01 {TRUE} \
  CONFIG.USER_AXI_CLK_FREQ $UCLK_MHZ CONFIG.USER_AXI_CLK1_FREQ $UCLK_MHZ]
for {set i 0} {$i < 32} {incr i} { lappend cfg CONFIG.USER_SAXI_[format %02d $i] {true} }
for {set i 0} {$i < 16} {incr i} { lappend cfg CONFIG.USER_MC${i}_REF_TEMP_COMP {true} }
set_property -dict $cfg $hbm
generate_target {instantiation_template synthesis} $hbm

# ---- block design
source $here/bd.tcl   ;# CORE_MHZ, UCLK_MHZ in; CORE_HZ out
make_wrapper -files [get_files otpu_bd.bd] -top
add_files -norecurse [lindex [glob $out/otpu.gen/sources_1/bd/otpu_bd/hdl/otpu_bd_wrapper.v \
                                   $out/otpu.srcs/sources_1/bd/otpu_bd/hdl/otpu_bd_wrapper.v] 0]
set_property top otpu_fpga_top_fk33 [current_fileset]
set CORE_KHZ [expr {round($CORE_HZ / 1000.0)}]
set SEQ_WC [expr {[info exists ::env(OTPU_SEQ_WC_GEN)] && $::env(OTPU_SEQ_WC_GEN) eq "1" ? 1 : 0}]
set_property generic "SEQ_WC=1'b$SEQ_WC MCOLS=$MCOLS ACT_ROWS=$ACT_ROWS VPU_CL=$VPU_CL LANES=$LANES CORE_KHZ=$CORE_KHZ BUILD_ID=32'h$BUILD_ID DSTEP=1'b$DSTEP PLATFORM=$PLATFORM ARCH_HASH=32'h$ARCH_HASH" [current_fileset]

# ---- constraints: the pins and configuration; the TMEM in UltraRAM (a synthesis property); then, as unmanaged Tcl
# on the synthesized netlist, otpu_mem_ch's crossings (fk33_mem_ch.tcl, scoped to the module) and the top's
set cons $root/boards/sqrl-fk33/constraints
add_files -fileset constrs_1 -norecurse $cons/fk33_pins.xdc
if {![info exists ::env(TMEM_URAM)] || $::env(TMEM_URAM) ne "0"} {
  set uram $here/tmem_uram.xdc
  add_files -fileset constrs_1 -norecurse $uram
  set_property USED_IN_IMPLEMENTATION false [get_files $uram]
  puts "TMEM in UltraRAM (tmem_uram.xdc)"
}
set memch $cons/fk33_mem_ch.tcl
set late [list $memch $cons/fk33_top.tcl]
add_files -fileset constrs_1 -norecurse $late
set_property SCOPED_TO_REF otpu_mem_ch [get_files $memch]
set_property PROCESSING_ORDER LATE [get_files $late]
set_property USED_IN_SYNTHESIS false [get_files $late]

# The xcvu33p (slice, shell and channels in its one SLR) congests: the openTPU_FK FK33 build with the Explore
# directives routed at congestion level 6 and failed timing; spreading at placement with AggressiveExplore met it.
if {[info exists ::env(OTPU_FAST)] && $::env(OTPU_FAST) eq "1"} {
  set_property strategy {Vivado Synthesis Defaults} [get_runs synth_1]
  set_property strategy {Vivado Implementation Defaults} [get_runs impl_1]
} else {
  set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
  set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
  set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE AltSpreadLogic_high [get_runs impl_1]
  set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
  set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
  set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
}
set_property report_strategy {No Reports} [get_runs synth_1]
set_property report_strategy {No Reports} [get_runs impl_1]
puts "project created in $out (SQRL FK33, S=1, HBM 2 x 2 GiB, Gen3 x8)"
