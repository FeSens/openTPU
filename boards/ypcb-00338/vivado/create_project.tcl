# Create the Vivado project for openTPU on the YPCB-00338: the LiteDRAM core
# (boards/ypcb-00338/litedram/), each channel behind otpu_mem_ch, and the LitePCIe core
# (boards/ypcb-00338/litepcie/: its PHY's hard block wrapper, the pcie_7x IP, made here from
# pcie_s7.tcl) with otpu_pcie_mm (bd_native.tcl, otpu_fpga_top_ld).
#   vivado -mode batch -source create_project.tcl -tclargs [DDR_SPEED] [OUT_DIR] [MCOLS] [CORE_MHZ] [BUILD_ID] [VPU_CL] [LANES] [ACT_ROWS] [DSTEP] [MXU] [PLATFORM] [ARCH_HASH] [MXU_E] [TMEM_IMPL] [TMEM_RQ] [TMEM_X]
# The defaults build MCOLS=4 and the systolic MXU.
# DDR_SPEED: 1066 only (the LiteDRAM core is generated for DDR3-1066: tools/litedram/gen_core.py).
# OUT_DIR: default ../../../build/vivado (repository build/).
# BUILD_ID: 8 hex digits for the BUILD_ID register (default: the first 8 hex digits of the
# repository's git commit, else 0).

set here [file normalize [file dirname [info script]]]
set root [file normalize $here/../../..]
# hexarg I: argument I as given ("" when absent). Not through expr's ?:, which reads a
# numeric-looking string as a number: BUILD_ID 83740e79 came out 8.374e+83, failed the hex
# check and was built as 0 (dev build 83740e79, 2026-10-12); 01234567 would be octal
proc hexarg {i} {
  if {[llength $::argv] > $i} { return [lindex $::argv $i] }
  return ""
}
set DDR_SPEED [expr {[llength $argv] > 0 ? [lindex $argv 0] : 1066}]
if {$DDR_SPEED != 1066} { error "DDR3-$DDR_SPEED: the LiteDRAM core is generated for DDR3-1066" }
set out [expr {[llength $argv] > 1 ? [file normalize [lindex $argv 1]] : "$root/build/vivado"}]
set MCOLS [expr {[llength $argv] > 2 ? [lindex $argv 2] : 4}]
set CORE_MHZ [expr {[llength $argv] > 3 ? [lindex $argv 3] : 133.33}]
set BUILD_ID [hexarg 4]
set VPU_CL [expr {[llength $argv] > 5 ? [lindex $argv 5] : 2}]
set LANES [expr {[llength $argv] > 6 ? [lindex $argv 6] : 8}]
set ACT_ROWS [expr {[llength $argv] > 7 ? [lindex $argv 7] : $MCOLS}]
# DSTEP 0: the DMA's DeltaNet head step left out (CAPS bit6 = 0; the compiler emits VOPs)
set DSTEP [expr {[llength $argv] > 8 ? [lindex $argv 8] : 1}]
# MXU: the MXU's dot product (docs/mxu_systolic.md): systolic (MXU_IMPL 2), the only one
set MXU [expr {[llength $argv] > 9 ? [lindex $argv 9] : "systolic"}]
if {$MXU ne "systolic"} { error "MXU $MXU: the systolic MXU is the only one" }
set MXU_IMPL 2
# PLATFORM, ARCH_HASH: the CFG_PLATFORM / CFG_HASH registers (run_vivado.sh: the target file,
# targets/ypcb-00338.toml, through opentpu/target.py; ARCH_HASH 8 hex digits)
set PLATFORM [expr {[llength $argv] > 10 ? [lindex $argv 10] : 1}]
set ARCH_HASH [hexarg 11]
if {$ARCH_HASH eq ""} { set ARCH_HASH 0 }
# MXU_E: the MXU's epilogue lanes (timing only; the target's [uarch] MXU_E, 0: MCOLS / 2)
set MXU_E [expr {[llength $argv] > 12 ? [lindex $argv 12] : 4}]
if {![regexp {^[0-9a-fA-F]{1,8}$} $ARCH_HASH]} { error "ARCH_HASH $ARCH_HASH: 1 to 8 hex digits" }
# TRACE_DEPTH (environment, run_vivado.sh): the trace buffer's records, a power of two; 0 leaves
# it out (CAPS trace bit 0: 32 RAMB36, ~2.3K LUT, ~3.9K FF; docs/infra_hunt.md)
set TRACE_DEPTH [expr {[info exists ::env(TRACE_DEPTH)] && $::env(TRACE_DEPTH) ne "" ? $::env(TRACE_DEPTH) : 16384}]
if {![regexp {^[0-9]+$} $TRACE_DEPTH] || ($TRACE_DEPTH != 0 && ($TRACE_DEPTH & ($TRACE_DEPTH - 1)))} {
  error "TRACE_DEPTH $TRACE_DEPTH: a power of two, or 0"
}
# COLL_GATHER (environment, run_vivado.sh): 0, the default since release r2, builds the collective
# unit without GATHER (BAR alone: CAPS2 bit0 = 0; the host refuses GATHER programs); 1 with it
# (docs/infra_hunt.md 5)
set COLL_GATHER [expr {[info exists ::env(COLL_GATHER)] && $::env(COLL_GATHER) ne "" ? $::env(COLL_GATHER) : 0}]
if {$COLL_GATHER ni {0 1}} { error "COLL_GATHER $COLL_GATHER: 0 or 1" }
# TMEM_WORDS (environment, run_vivado.sh: the target's [arch] TMEM_WORDS, in its CFG_HASH): the
# TMEM's words, a power of two (the CFG_TMEM register; the card's m4 64K, m8 128K: 128 RAMB36)
set TMEM_WORDS [expr {[info exists ::env(TMEM_WORDS)] && $::env(TMEM_WORDS) ne "" ? $::env(TMEM_WORDS) : 65536}]
if {![regexp {^[0-9]+$} $TMEM_WORDS] || $TMEM_WORDS < 1024 || ($TMEM_WORDS & ($TMEM_WORDS - 1))} {
  error "TMEM_WORDS $TMEM_WORDS: a power of two, 1024 or more"
}
# TMEM_IMPL: the TMEM (otpu_tmem): 0 a copy per read port (production), 1 banked (three copies;
# the slice's banked arbiter, docs/tmem_pump.md), 3 one copy (docs/tmem_nocopy.md; TMEM_X its
# knobs, tmem_knobs.tcl: by default 2 RAMs a bank pumped at 2x on core_clk2x). Timing only: the
# programs do not change. The three follow MXU_E (arguments 13, 14, 15)
set TMEM_IMPL [expr {[llength $argv] > 13 ? [lindex $argv 13] : 0}]
# TMEM_RQ (TMEM_IMPL 1): the banks' addresses from the requests by priority, not the grants
set TMEM_RQ [expr {[llength $argv] > 14 ? [lindex $argv 14] : 0}]
if {$TMEM_RQ ni {0 1}} { error "TMEM_RQ $TMEM_RQ: 0 or 1" }
set TMEM_X [expr {[llength $argv] > 15 ? [lindex $argv 15] : ""}]
source $here/tmem_knobs.tcl   ;# TMEM_GEN, CLK2X (bd_native.tcl's core_clk2x)
if {$BUILD_ID eq ""} {
  if {[catch {exec git -C $root rev-parse HEAD} BUILD_ID]} { set BUILD_ID 0 }
  set BUILD_ID [string range $BUILD_ID 0 7]
}
if {![regexp {^[0-9a-fA-F]{1,8}$} $BUILD_ID]} { error "BUILD_ID $BUILD_ID: 1 to 8 hex digits" }
# the core clock the block design makes (bd_native.tcl: CORE_MHZ rounded to the MMCM's 1/8
# divider steps of its 800 MHz VCO), for the CORE_KHZ register
set VCO 800
set CORE_KHZ [expr {round($VCO * 1000.0 / (round(double($VCO) / $CORE_MHZ * 8) / 8.0))}]
puts "CORE_KHZ $CORE_KHZ, BUILD_ID $BUILD_ID, MXU_IMPL $MXU_IMPL, MXU_E $MXU_E, TMEM_IMPL $TMEM_IMPL, TMEM_RQ $TMEM_RQ, PLATFORM $PLATFORM, ARCH_HASH $ARCH_HASH, TRACE_DEPTH $TRACE_DEPTH, COLL_GATHER $COLL_GATHER, TMEM_WORDS $TMEM_WORDS"

create_project -force otpu $out -part xc7k480tffg1156-2
# The IP synthesis (minutes: the PCIe block's wrapper above all) in a cache shared by every build on
# the host (OTPU_IP_CACHE, run_vivado.sh): each build is a fresh project, so the
# project's own cache never hit. Keyed on the IP's configuration, part and Vivado version.
if {[info exists ::env(OTPU_IP_CACHE)] && $::env(OTPU_IP_CACHE) ne ""} {
  file mkdir $::env(OTPU_IP_CACHE)
  config_ip_cache -use_cache_location $::env(OTPU_IP_CACHE)
}
set_property target_language Verilog [current_project]
set_property default_lib xil_defaultlib [current_project]

# ---- RTL (SystemVerilog): the accelerator (otpu_board, its DRAM adapter otpu_native_dram), the
# channels' native ports (one otpu_mem_ch per channel, the host DMA's split: otpu_native_sys), the
# host DMA's master (otpu_pcie_mm), the top and the generated LiteDRAM and LitePCIe cores (Verilog)
set rtl [list \
  rtl/vpu/otpu_fp.sv rtl/vpu/otpu_fpipe.sv rtl/top/otpu_pkg.sv \
  rtl/mem/otpu_tmem.sv rtl/mem/otpu_actram.sv \
  rtl/seq/otpu_seq.sv rtl/vpu/otpu_vtree.sv rtl/dma/otpu_dma.sv \
  rtl/mxu/otpu_mxu.sv rtl/tech/xilinx7/otpu_tech_dsp.sv rtl/tech/generic/otpu_tech_ram.sv \
  rtl/vpu/otpu_quant.sv rtl/vpu/otpu_se_comp.sv rtl/vpu/otpu_se_tail.sv rtl/vpu/otpu_vpu.sv rtl/top/otpu_coll.sv \
  rtl/top/otpu_slice.sv \
  rtl/shell/otpu_ctrl.sv rtl/shell/otpu_trace.sv \
  rtl/shell/otpu_board.sv rtl/mem/otpu_native_dram.sv \
  rtl/shell/otpu_afifo.sv rtl/shell/otpu_mem_ch.sv \
  rtl/shell/otpu_axi_split2.sv rtl/shell/otpu_pcie_mm.sv rtl/boards/ypcb-00338/otpu_native_sys.sv \
  rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv]
add_files -norecurse $root/boards/ypcb-00338/litedram/otpu_litedram.v
# the core's identifier ROM ("openTPU LiteDRAM"), which the Verilog $readmemh's by file name:
# synthesis finds it as a project file
add_files -norecurse $root/boards/ypcb-00338/litedram/otpu_litedram_mem.init
set_property FILE_TYPE {Memory Initialization Files} [get_files otpu_litedram_mem.init]
# the LitePCIe core and its identifier ROM ("openTPU LitePCIe"), the same way
add_files -norecurse $root/boards/ypcb-00338/litepcie/otpu_litepcie.v
add_files -norecurse $root/boards/ypcb-00338/litepcie/otpu_litepcie_mem.init
set_property FILE_TYPE {Memory Initialization Files} [get_files otpu_litepcie_mem.init]
# the PCIe hard block's wrapper the core instantiates (pcie_s7: Gen2 x8, 128 bits, 250 MHz, the
# card's PCI identity), synthesized out of context by build.tcl
source $root/boards/ypcb-00338/litepcie/pcie_s7.tcl
foreach f $rtl {
  add_files -norecurse $root/$f
  set_property file_type SystemVerilog [get_files $root/$f]
}
# SYNTHESIS removes the simulation-only checks and dumps
set_property verilog_define {SYNTHESIS} [get_filesets sources_1]

# ---- block design
source $here/bd_native.tcl   ;# sets PCIE_GEN (environment, run_vivado.sh) for the generics; CLK2X
make_wrapper -files [get_files otpu_bd.bd] -top
add_files -norecurse [glob $out/otpu.gen/sources_1/bd/otpu_bd/hdl/otpu_bd_wrapper.v]
set_property top otpu_fpga_top_ld [current_fileset]
# otpu_seq WAITC (run_vivado.sh: the target's [arch] SEQ_WC, or SEQ_WC=1; CFG_UARCH bit24)
set SEQ_WC [expr {[info exists ::env(OTPU_SEQ_WC_GEN)] && $::env(OTPU_SEQ_WC_GEN) eq "1" ? 1 : 0}]
set_property generic "SEQ_WC=1'b$SEQ_WC MCOLS=$MCOLS ACT_ROWS=$ACT_ROWS VPU_CL=$VPU_CL LANES=$LANES CORE_KHZ=$CORE_KHZ BUILD_ID=32'h$BUILD_ID DDR_MTS=$DDR_SPEED DSTEP=1'b$DSTEP MXU_IMPL=$MXU_IMPL MXU_E=$MXU_E TMEM_IMPL=$TMEM_IMPL TMEM_RQ=1'b$TMEM_RQ PCIE_GEN=$PCIE_GEN PLATFORM=$PLATFORM ARCH_HASH=32'h$ARCH_HASH TRACE_DEPTH=$TRACE_DEPTH COLL_GATHER=1'b$COLL_GATHER TMEM_WORDS=$TMEM_WORDS$TMEM_GEN" [current_fileset]

# ---- constraints: the board (otpu_top_ld.xdc) and the core's pads, VREF and synchronizers
# (otpu_litedram.xdc); then, after the IP's constraints, as unmanaged Tcl on the implemented
# netlist, each otpu_mem_ch's crossings (scoped to the module) and the top's (otpu_top_native.tcl);
# with CLK2X the paths between core_clk and core_clk2x (otpu_clk2x.tcl); the MXU's systolic array
# as a grid of LOCs, each column on its DSP column beside its ACT RAM memory, generated from the
# netlist at MCOLS 8 and up (otpu_mxu_place.tcl; below, the placer's: OTPU_MXU_PLACE=1 / 0 overrides);
# the memory PHY's OSERDES data flops in pblocks beside their pads (otpu_ddr_phy_place.tcl;
# OTPU_DDR_PHY_PLACE=0 leaves them to the placer)
set cons $root/boards/ypcb-00338/constraints
add_files -fileset constrs_1 -norecurse [list $cons/otpu_top_ld.xdc \
  $root/boards/ypcb-00338/litedram/otpu_litedram.xdc $root/boards/ypcb-00338/litepcie/otpu_litepcie.xdc]
set late [list $cons/otpu_litepcie_clk.tcl $cons/otpu_mem_ch.tcl $cons/otpu_top_native.tcl]
if {$CLK2X} { lappend late $cons/otpu_clk2x.tcl }
lappend late $cons/otpu_mxu_place.tcl $cons/otpu_ddr_phy_place.tcl
add_files -fileset constrs_1 -norecurse $late
set_property SCOPED_TO_REF otpu_mem_ch [get_files $cons/otpu_mem_ch.tcl]
# LiteX writes the core's constraints for the core as the top (get_nets {*pclk_sel})
set_property SCOPED_TO_REF otpu_litepcie [get_files otpu_litepcie.xdc]
set_property PROCESSING_ORDER LATE [get_files $late]
set_property USED_IN_SYNTHESIS false [get_files $late]
foreach f $late {
  if {![string match -nocase *tcl* [get_property FILE_TYPE [get_files $f]]]} {
    error "$f: FILE_TYPE [get_property FILE_TYPE [get_files $f]], expected TCL (unmanaged)"
  }
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
