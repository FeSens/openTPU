# Create the Vivado project for the BCU1525 at WG 8 (targets/bcu1525-wg.toml; docs/bcu1525_wg.md):
# the accelerator (one slice of 8 channels, the WG targets') on the four DDR4-2400 UDIMMs through AMD's MIG,
# the block design bd_wg.tcl (XDMA Gen3, the clocks, the SYSMON, the MIGs behind their
# SmartConnects) and the top rtl/boards/bcu1525/otpu_fpga_top_bcu_wg.sv.
#   vivado -mode batch -source create_project_wg.tcl -tclargs OUT_DIR
# OUT_DIR holds boards/bcu1525/scripts/make_wg.py's output (run_vivado.sh IMAGE=wg writes it
# first): wg_params.tcl (CORE_MHZ, CORE_KHZ, RLY, WG_CAW, BUILD_ID, ARCH_HASH, the RTL:
# opentpu.target sources("bcu1525-wg")) and otpu_bcu_wg_defines.vh (otpu_board's parameters,
# OTPU_BOARD), a global include. WG_CAW (the channels' beat address bits) is the top's CAW and
# bd_wg.tcl's DMA windows. WG_S (bcu1525-s2: 2; absent: 1) the slices: RLY 32 x WG_S bits, bd_wg.tcl's
# windows a DIMM's 2 x WG_S channels. Environment: PCIE_LANES (bd_wg.tcl), OTPU_FAST (Vivado's default
# strategies), OTPU_IP_CACHE. At 150 MHz the top's CORE_KHZ1 is 133333: the RTL carries the
# fallback clock's constants too (otpu_ctrl.sv's otpu_clk_k), and the same synthesis implements at
# 133.33 MHz with OTPU_CLK_HOOK=clk133.tcl (bcu1525_wg_check.tcl, before opt_design).
set here [file normalize [file dirname [info script]]]
set root [file normalize $here/../../..]
if {[llength $argv] < 1} { error "create_project_wg.tcl: OUT_DIR (make_wg.py's output)" }
set out [file normalize [lindex $argv 0]]
source $out/wg_params.tcl
foreach v {CORE_MHZ CORE_KHZ RLY WG_CAW BUILD_ID ARCH_HASH RTL} {
  if {![info exists $v]} { error "create_project_wg.tcl: $out/wg_params.tcl sets no $v" }
}
if {![info exists WG_S]} { set WG_S 1 }
if {![info exists WG_NCARDS]} { set WG_NCARDS 1 }
set RLYW [expr {32 * $WG_S}]
if {[string length $RLY] != $RLYW / 4} { error "create_project_wg.tcl: RLY $RLY: $RLYW bits at WG_S $WG_S" }
set PART xcvu9p-fsgd2104-2L-e
set CORE_KHZ1 [expr {$CORE_KHZ == 150000 ? 133333 : $CORE_KHZ}]
puts "BCU1525 WG: $WG_S slice(s), core $CORE_KHZ kHz (fallback $CORE_KHZ1), RLY $RLYW'h$RLY, CAW $WG_CAW, BUILD_ID $BUILD_ID, ARCH_HASH $ARCH_HASH"

create_project -force otpu $out -part $PART
if {[info exists ::env(OTPU_IP_CACHE)] && $::env(OTPU_IP_CACHE) ne ""} {
  file mkdir $::env(OTPU_IP_CACHE)
  config_ip_cache -use_cache_location $::env(OTPU_IP_CACHE)
}
set_property target_language Verilog [current_project]

# ---- RTL, otpu_board's parameters first (a global include: every file sees OTPU_BOARD)
foreach f $RTL {
  add_files -norecurse $root/$f
  set_property file_type SystemVerilog [get_files $root/$f]
}
set vh $out/otpu_bcu_wg_defines.vh
add_files -norecurse $vh
set_property file_type {Verilog Header} [get_files $vh]
set_property is_global_include true [get_files $vh]
# OTPU_FPGA_INIT: no memories' power-on reset (otpu_slice rst_mem; docs/asic.md "Power-up state")
set_property verilog_define {SYNTHESIS OTPU_FPGA_INIT} [get_filesets sources_1]
if {$WG_NCARDS == 2} {
  if {$WG_S != 2} { error "x2 requires WG_S 2" }
  source $root/boards/bcu1525/x2/project_x2.tcl
}

# ---- block design
source $here/bd_wg.tcl
if {$WG_NCARDS == 2} { source $root/boards/bcu1525/x2/aurora_ip.tcl }
make_wrapper -files [get_files otpu_bdw.bd] -top
add_files -norecurse [glob $out/otpu.gen/sources_1/bd/otpu_bdw/hdl/otpu_bdw_wrapper.v]
set_property top otpu_fpga_top_bcu_wg [current_fileset]
set_property generic "CORE_KHZ=$CORE_KHZ CORE_KHZ1=$CORE_KHZ1 PCIE_LANES=$PCIE_LANES RLY=$RLYW'h$RLY CAW=$WG_CAW" [current_fileset]

# ---- constraints: the pins (scripts/gen_wg_pins.py); then, after the IPs' own, as unmanaged Tcl
# on the synthesized netlist: the bridges' crossings (the WG targets', scoped to their core halves), the
# MXU lanes' posted-beat staging (scoped), the TMEM's pumped RAMs (the YPCB's otpu_clk2x.tcl), the
# floorplan (bcu1525_wg_floorplan.tcl: OTPU_WG_FP) and the top's (bcu1525_wg_top.tcl, last). opt_design's pre-hook (bcu1525_wg_check.tcl) stops the run
# on any of them failing.
set cons $root/boards/bcu1525/constraints
# the board clocks no IP creates (clk300_1, pcie_refclk), in the normal order: the IPs' late
# constraints (XDMA's pcie4_ip_late.xdc) group them
add_files -fileset constrs_1 -norecurse $cons/bcu1525_wg_clocks.xdc
add_files -fileset constrs_1 -norecurse $cons/bcu1525_wg_pins.xdc
# after the IPs' (as the MIG probes' pins): the board clocks' LVDS over a MIG's own I/O standard for
# its system clock
set_property PROCESSING_ORDER LATE [get_files $cons/bcu1525_wg_pins.xdc]
set late [list $cons/otpu_hbm_ch.tcl $cons/otpu_mxu_stage.tcl \
               $root/boards/ypcb-00338/constraints/otpu_clk2x.tcl $cons/bcu1525_wg_floorplan.tcl \
               $cons/bcu1525_wg_top.tcl]
if {$WG_NCARDS == 2} { lappend late $root/boards/bcu1525/x2/constraints_x2.tcl }
add_files -fileset constrs_1 -norecurse $late
set_property SCOPED_TO_REF otpu_hbm_cs [get_files $cons/otpu_hbm_ch.tcl]
set_property SCOPED_TO_REF otpu_mxu_stage [get_files $cons/otpu_mxu_stage.tcl]
set_property PROCESSING_ORDER LATE [get_files $late]
set_property USED_IN_SYNTHESIS false [get_files $late]
foreach f $late {
  if {![string match -nocase *tcl* [get_property FILE_TYPE [get_files $f]]]} {
    error "$f: FILE_TYPE [get_property FILE_TYPE [get_files $f]], expected TCL (unmanaged)"
  }
}
set_property STEPS.OPT_DESIGN.TCL.PRE $here/bcu1525_wg_check.tcl [get_runs impl_1]
# place_design's: clk300_1's BUFG drives its two MMCMs (C1's MIG, the clock wizard) over the backbone
set_property STEPS.PLACE_DESIGN.TCL.PRE $here/bcu1525_wg_place_pre.tcl [get_runs impl_1]

if {[info exists ::env(OTPU_FAST)] && $::env(OTPU_FAST) eq "1"} {
  set_property strategy {Vivado Synthesis Defaults} [get_runs synth_1]
  set_property strategy {Vivado Implementation Defaults} [get_runs impl_1]
} else {
  set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
  set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
  set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
}
set_property report_strategy {No Reports} [get_runs synth_1]
set_property report_strategy {No Reports} [get_runs impl_1]
puts "project created in $out (BCU1525 WG, Gen3 x$PCIE_LANES, 4 x DDR4-2400 MIG, core $CORE_MHZ MHz)"
