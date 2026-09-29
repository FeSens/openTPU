# Create the Vivado project for openTPU on the YPCB-00338.
#   vivado -mode batch -source create_project.tcl -tclargs [DDR_SPEED] [OUT_DIR] [MCOLS] [CORE_MHZ] [BUILD_ID] [VPU_CL] [LANES] [ACT_ROWS] [DSTEP] [AXI_BL] [MEM] [MXU]
# DDR_SPEED: 800 (default), 1066, or the out-of-spec 1300, 1333, 1600 (docs/board.md). OUT_DIR: default ../../../build/vivado (repository build/).
# BUILD_ID: 8 hex digits for the BUILD_ID register (default: the first 8 hex digits of the
# repository's git commit, else 0).
# MEM: the memory controllers and how the accelerator and XDMA reach them.
#   mig (default)  the two MIGs' AXI ports behind the SmartConnect (bd.tcl, otpu_fpga_top)
#   mig_native     the two MIGs' native ports, each behind otpu_mig_native and otpu_mem_ch
#                  (bd_native.tcl, otpu_fpga_top_mn; gen_mig_prj.py --native for the .prj files;
#                  the MIGs are RTL-level IP here, IP integrator takes the MIG with AXI only)
#   litedram       the LiteDRAM core boards/ypcb-00338/litedram/, each channel behind otpu_mem_ch
#                  (bd_native.tcl, otpu_fpga_top_ld; DDR3-1066 whatever DDR_SPEED says)

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
set MEM [expr {[llength $argv] > 10 ? [lindex $argv 10] : "mig"}]
if {$MEM ni {mig mig_native litedram}} { error "MEM must be mig, mig_native or litedram, not $MEM" }
if {$MEM eq "litedram"} { set DDR_SPEED 1066 }
# MXU: the MXU's dot product (docs/mxu_systolic.md): systolic (MXU_IMPL 2, the default) or tree (0)
set MXU [expr {[llength $argv] > 11 ? [lindex $argv 11] : "systolic"}]
set MXU_IMPL [expr {$MXU eq "tree" ? 0 : 2}]
if {$BUILD_ID eq ""} {
  if {[catch {exec git -C $root rev-parse HEAD} BUILD_ID]} { set BUILD_ID 0 }
  set BUILD_ID [string range $BUILD_ID 0 7]
}
if {![regexp {^[0-9a-fA-F]{1,8}$} $BUILD_ID]} { set BUILD_ID 0 }
# the core clock the block design makes (bd.tcl: CORE_MHZ rounded to the MMCM's 1/8 divider
# steps of its VCO, 1000 MHz for DDR3-1333, else 800), for the CORE_KHZ register
set VCO [expr {$DDR_SPEED == 1333 ? 1000 : 800}]
set CORE_KHZ [expr {round($VCO * 1000.0 / (round(double($VCO) / $CORE_MHZ * 8) / 8.0))}]
puts "CORE_KHZ $CORE_KHZ, BUILD_ID $BUILD_ID, MXU_IMPL $MXU_IMPL"
set MIG_DIR [expr {$MEM eq "mig_native" ? "$here/mig_native" : "$here/mig"}]

if {$MEM eq "mig" && ![file exists $MIG_DIR/mig_ddr3_ch0.prj]} {
  error "run boards/ypcb-00338/scripts/gen_mig_prj.py first (it writes vivado/mig/*.prj)"
}
# mig_native: .prj files with the native interface (an AXI one would build the AXI MIG). Not at
# DDR3-1333 / 1600: build.tcl's IDELAYE2_FINEDELAY patch covers the block design's MIGs only.
if {$MEM eq "mig_native"} {
  if {$DDR_SPEED in {1333 1600}} { error "MEM=mig_native: DDR3-$DDR_SPEED is not supported (build.tcl's MIG PHY patch)" }
  foreach ch {0 1} {
    set f $MIG_DIR/mig_ddr3_ch${ch}.prj
    if {![file exists $f]} { error "run boards/ypcb-00338/scripts/gen_mig_prj.py --native first" }
    set fh [open $f]; set x [read $fh]; close $fh
    if {![regexp {<PortInterface>NATIVE</PortInterface>} $x]} {
      error "$f has the AXI interface: run boards/ypcb-00338/scripts/gen_mig_prj.py --native"
    }
  }
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
# the native builds: the channels' native ports (otpu_native_dram in otpu_board, one otpu_mem_ch
# per channel, XDMA's split: otpu_native_sys) and the controllers' side (otpu_mig_native; the
# generated LiteDRAM core, Verilog)
if {$MEM ne "mig"} {
  set rtl [lreplace $rtl end end rtl/mem/otpu_native_dram.sv \
    rtl/boards/ypcb-00338/otpu_afifo.sv rtl/boards/ypcb-00338/otpu_mem_ch.sv \
    rtl/boards/ypcb-00338/otpu_axi_split2.sv rtl/boards/ypcb-00338/otpu_native_sys.sv]
  if {$MEM eq "mig_native"} {
    lappend rtl rtl/boards/ypcb-00338/otpu_mig_native.sv rtl/boards/ypcb-00338/otpu_fpga_top_mn.sv
  } else {
    lappend rtl rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv
    add_files -norecurse $root/boards/ypcb-00338/litedram/otpu_litedram.v
  }
}
foreach f $rtl {
  add_files -norecurse $root/$f
  set_property file_type SystemVerilog [get_files $root/$f]
}
# SYNTHESIS removes the simulation-only checks and dumps
set_property verilog_define {SYNTHESIS} [get_filesets sources_1]

# mig_native: the two MIGs as RTL-level IP, instantiated in otpu_fpga_top_mn (u_mig/mig_<c>),
# each from its native .prj as bd.tcl gives its MIG cells theirs; the module is the .prj's
# ModuleName, mig_ddr3_ch<c>. Their out-of-context runs are made here, so synth_1 runs them first
# as it does the block design's.
if {$MEM eq "mig_native"} {
  set migdef [lindex [lsort -decreasing [get_ipdefs -all xilinx.com:ip:mig_7series:*]] 0]
  if {$migdef eq ""} { error "IP mig_7series not found in this Vivado installation" }
  foreach ch {0 1} {
    create_ip -vlnv $migdef -module_name mig_ddr3_ch$ch
    set ip [get_ips mig_ddr3_ch$ch]
    file copy -force $MIG_DIR/mig_ddr3_ch${ch}.prj [get_property IP_DIR $ip]/mig_ddr3_ch${ch}.prj
    set_property -dict [list CONFIG.BOARD_MIG_PARAM {Custom} CONFIG.MIG_DONT_TOUCH_PARAM {Custom} \
      CONFIG.RESET_BOARD_INTERFACE {Custom} CONFIG.XML_INPUT_FILE mig_ddr3_ch${ch}.prj] $ip
  }
  set migs [get_ips {mig_ddr3_ch0 mig_ddr3_ch1}]
  generate_target all $migs
  # one IP per create_ip_run (a collection: "[Vivado 12-3445] ... provide only one sub-design")
  foreach ch {0 1} { create_ip_run [get_files [get_property IP_FILE [get_ips mig_ddr3_ch$ch]]] }
}

# ---- block design
source $here/[expr {$MEM eq "mig" ? "bd.tcl" : "bd_native.tcl"}]
make_wrapper -files [get_files otpu_bd.bd] -top
add_files -norecurse [glob $out/otpu.gen/sources_1/bd/otpu_bd/hdl/otpu_bd_wrapper.v]
set_property top [dict get {mig otpu_fpga_top mig_native otpu_fpga_top_mn litedram otpu_fpga_top_ld} $MEM] [current_fileset]
set_property generic "MCOLS=$MCOLS ACT_ROWS=$ACT_ROWS VPU_CL=$VPU_CL LANES=$LANES CORE_KHZ=$CORE_KHZ BUILD_ID=32'h$BUILD_ID DDR_MTS=$DDR_SPEED DSTEP=1'b$DSTEP AXI_BL=$AXI_BL MXU_IMPL=$MXU_IMPL" [current_fileset]

# ---- constraints
if {$MEM ne "litedram"} {
  add_files -fileset constrs_1 -norecurse [list \
    $root/boards/ypcb-00338/constraints/otpu_top.xdc \
    $root/boards/ypcb-00338/constraints/otpu_ddr3_pins.xdc]
} else {
  # the board (otpu_top_ld.xdc) and the core's pads, VREF and synchronizers (otpu_litedram.xdc)
  add_files -fileset constrs_1 -norecurse [list \
    $root/boards/ypcb-00338/constraints/otpu_top_ld.xdc \
    $root/boards/ypcb-00338/litedram/otpu_litedram.xdc]
}
# the native builds: then, after the IP's constraints, as unmanaged Tcl on the implemented
# netlist, each otpu_mem_ch's crossings (scoped to the module) and the top's (otpu_top_native.tcl)
if {$MEM ne "mig"} {
  set cons $root/boards/ypcb-00338/constraints
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
}

# ---- strategies: timing-driven, the accelerator is the critical part
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]

puts "project created in $out (DDR3-$DDR_SPEED, $MEM)"
