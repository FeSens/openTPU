# Create the Vivado project for openTPU on the YPCB-00338: the LiteDRAM core
# (boards/ypcb-00338/litedram/), each channel behind otpu_mem_ch (bd_native.tcl, otpu_fpga_top_ld).
#   vivado -mode batch -source create_project.tcl -tclargs [DDR_SPEED] [OUT_DIR] [MCOLS] [CORE_MHZ] [BUILD_ID] [VPU_CL] [LANES] [ACT_ROWS] [DSTEP] [MXU]
# The defaults build MCOLS=4 and the systolic MXU.
# DDR_SPEED: 1066 only (the LiteDRAM core is generated for DDR3-1066: tools/litedram/gen_core.py).
# OUT_DIR: default ../../../build/vivado (repository build/).
# BUILD_ID: 8 hex digits for the BUILD_ID register (default: the first 8 hex digits of the
# repository's git commit, else 0).

set here [file normalize [file dirname [info script]]]
set root [file normalize $here/../../..]
set DDR_SPEED [expr {[llength $argv] > 0 ? [lindex $argv 0] : 1066}]
if {$DDR_SPEED != 1066} { error "DDR3-$DDR_SPEED: the LiteDRAM core is generated for DDR3-1066" }
set out [expr {[llength $argv] > 1 ? [file normalize [lindex $argv 1]] : "$root/build/vivado"}]
set MCOLS [expr {[llength $argv] > 2 ? [lindex $argv 2] : 4}]
set CORE_MHZ [expr {[llength $argv] > 3 ? [lindex $argv 3] : 133.33}]
set BUILD_ID [expr {[llength $argv] > 4 ? [lindex $argv 4] : ""}]
set VPU_CL [expr {[llength $argv] > 5 ? [lindex $argv 5] : 2}]
set LANES [expr {[llength $argv] > 6 ? [lindex $argv 6] : 8}]
set ACT_ROWS [expr {[llength $argv] > 7 ? [lindex $argv 7] : $MCOLS}]
# DSTEP 0: the DMA's DeltaNet head step left out (CAPS bit6 = 0; the compiler emits VOPs)
set DSTEP [expr {[llength $argv] > 8 ? [lindex $argv 8] : 1}]
# MXU: the MXU's dot product (docs/mxu_systolic.md): systolic (MXU_IMPL 2, the default) or tree (0)
set MXU [expr {[llength $argv] > 9 ? [lindex $argv 9] : "systolic"}]
set MXU_IMPL [expr {$MXU eq "tree" ? 0 : 2}]
if {$BUILD_ID eq ""} {
  if {[catch {exec git -C $root rev-parse HEAD} BUILD_ID]} { set BUILD_ID 0 }
  set BUILD_ID [string range $BUILD_ID 0 7]
}
if {![regexp {^[0-9a-fA-F]{1,8}$} $BUILD_ID]} { set BUILD_ID 0 }
# the core clock the block design makes (bd_native.tcl: CORE_MHZ rounded to the MMCM's 1/8
# divider steps of its 800 MHz VCO), for the CORE_KHZ register
set VCO 800
set CORE_KHZ [expr {round($VCO * 1000.0 / (round(double($VCO) / $CORE_MHZ * 8) / 8.0))}]
puts "CORE_KHZ $CORE_KHZ, BUILD_ID $BUILD_ID, MXU_IMPL $MXU_IMPL"

create_project -force otpu $out -part xc7k480tffg1156-2
# The block design's IP synthesis (minutes: xdma above all) in a cache shared by every build on
# the host (OTPU_IP_CACHE, run_vivado.sh): each build is a fresh project, so the
# project's own cache never hit. Keyed on the IP's configuration, part and Vivado version.
if {[info exists ::env(OTPU_IP_CACHE)] && $::env(OTPU_IP_CACHE) ne ""} {
  file mkdir $::env(OTPU_IP_CACHE)
  config_ip_cache -use_cache_location $::env(OTPU_IP_CACHE)
}
set_property target_language Verilog [current_project]
set_property default_lib xil_defaultlib [current_project]

# ---- RTL (SystemVerilog): the accelerator (otpu_board, its DRAM adapter otpu_native_dram), the
# channels' native ports (one otpu_mem_ch per channel, XDMA's split: otpu_native_sys), the top and
# the generated LiteDRAM core (Verilog)
set rtl [list \
  rtl/vpu/otpu_fp.sv rtl/vpu/otpu_fpipe.sv rtl/top/otpu_pkg.sv \
  rtl/mem/otpu_tmem.sv rtl/mem/otpu_actram.sv \
  rtl/seq/otpu_seq.sv rtl/vpu/otpu_vtree.sv rtl/dma/otpu_dma.sv \
  rtl/mxu/otpu_mxu.sv \
  rtl/vpu/otpu_quant.sv rtl/vpu/otpu_se_comp.sv rtl/vpu/otpu_se_tail.sv rtl/vpu/otpu_vpu.sv rtl/top/otpu_coll.sv \
  rtl/top/otpu_slice.sv \
  rtl/boards/ypcb-00338/otpu_ctrl.sv rtl/boards/ypcb-00338/otpu_trace.sv \
  rtl/boards/ypcb-00338/otpu_board.sv rtl/mem/otpu_native_dram.sv \
  rtl/boards/ypcb-00338/otpu_afifo.sv rtl/boards/ypcb-00338/otpu_mem_ch.sv \
  rtl/boards/ypcb-00338/otpu_axi_split2.sv rtl/boards/ypcb-00338/otpu_native_sys.sv \
  rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv]
add_files -norecurse $root/boards/ypcb-00338/litedram/otpu_litedram.v
# the core's identifier ROM ("openTPU LiteDRAM"), which the Verilog $readmemh's by file name:
# synthesis finds it as a project file
add_files -norecurse $root/boards/ypcb-00338/litedram/otpu_litedram_mem.init
set_property FILE_TYPE {Memory Initialization Files} [get_files otpu_litedram_mem.init]
foreach f $rtl {
  add_files -norecurse $root/$f
  set_property file_type SystemVerilog [get_files $root/$f]
}
# SYNTHESIS removes the simulation-only checks and dumps
set_property verilog_define {SYNTHESIS} [get_filesets sources_1]

# ---- block design
source $here/bd_native.tcl
make_wrapper -files [get_files otpu_bd.bd] -top
add_files -norecurse [glob $out/otpu.gen/sources_1/bd/otpu_bd/hdl/otpu_bd_wrapper.v]
set_property top otpu_fpga_top_ld [current_fileset]
set_property generic "MCOLS=$MCOLS ACT_ROWS=$ACT_ROWS VPU_CL=$VPU_CL LANES=$LANES CORE_KHZ=$CORE_KHZ BUILD_ID=32'h$BUILD_ID DDR_MTS=$DDR_SPEED DSTEP=1'b$DSTEP MXU_IMPL=$MXU_IMPL" [current_fileset]

# ---- constraints: the board (otpu_top_ld.xdc) and the core's pads, VREF and synchronizers
# (otpu_litedram.xdc); then, after the IP's constraints, as unmanaged Tcl on the implemented
# netlist, each otpu_mem_ch's crossings (scoped to the module) and the top's (otpu_top_native.tcl)
set cons $root/boards/ypcb-00338/constraints
add_files -fileset constrs_1 -norecurse [list $cons/otpu_top_ld.xdc \
  $root/boards/ypcb-00338/litedram/otpu_litedram.xdc]
set late [list $cons/otpu_mem_ch.tcl $cons/otpu_top_native.tcl]
add_files -fileset constrs_1 -norecurse $late
set_property SCOPED_TO_REF otpu_mem_ch [get_files $cons/otpu_mem_ch.tcl]
set_property PROCESSING_ORDER LATE [get_files $late]
set_property USED_IN_SYNTHESIS false [get_files $late]
foreach f $late {
  if {![string match -nocase *tcl* [get_property FILE_TYPE [get_files $f]]]} {
    error "$f: FILE_TYPE [get_property FILE_TYPE [get_files $f]], expected TCL (unmanaged)"
  }
}

# FLOORPLAN=1 (run_vivado.sh): soft Pblocks that keep the accelerator's units by their partners
# (constraints/otpu_floorplan.tcl), unmanaged Tcl read late in implementation like the crossings
# above; off by default
if {[info exists ::env(OTPU_FLOORPLAN)] && $::env(OTPU_FLOORPLAN) ni {"" 0}} {
  set fp $cons/otpu_floorplan.tcl
  add_files -fileset constrs_1 -norecurse $fp
  set_property PROCESSING_ORDER LATE [get_files $fp]
  set_property USED_IN_SYNTHESIS false [get_files $fp]
  puts "floorplan: $fp"
}

# ---- strategies: timing-driven, the accelerator is the critical part. OTPU_FAST=1 (run_vivado.sh
# FAST=1): a development build at a relaxed clock (CORE_MHZ 100), with Vivado's default synthesis
# (no retiming) and implementation (build.tcl: no post-route phys_opt, no impl_directives.tcl)
if {[info exists ::env(OTPU_FAST)] && $::env(OTPU_FAST) eq "1"} {
  set_property strategy {Vivado Synthesis Defaults} [get_runs synth_1]
  set_property strategy {Vivado Implementation Defaults} [get_runs impl_1]
} else {
  set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
  set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
  set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
}
# build.tcl writes its own reports from the opened runs: none inside the runs
set_property report_strategy {No Reports} [get_runs synth_1]
set_property report_strategy {No Reports} [get_runs impl_1]

puts "project created in $out (DDR3-$DDR_SPEED, LiteDRAM)"
