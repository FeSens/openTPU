# EXPERIMENTAL: otpu_mxu out of context at bcu1525-x2's parameters (the slice's: WG 8, MCOLS 8,
# XPOST, KT, the UltraRAM FIFO lanes) with the pumped array (PUMP 1), on the BCU1525's part. The
# core clock and clk_f are two primary clocks in phase (CORE_NS, CORE_NS / 2); after synthesis
# boards/ypcb-00338/constraints/otpu_clk2x.tcl's sections (the pumped MXU's multicycle set, as the
# card builds read it). Vivado with a VU9P license:
#   vivado -mode batch -nojournal -source rtl/exp/pump_us/ooc_mxu.tcl -tclargs <out> [K=V ...]
# K: PUMP 1, PHOLD 1 (the chain ends held into clk), CORE_NS 6.666 (150 MHz; clk_f 300),
# PBLOCK none | SLR0 | SLR1 | SLR2 | CLOCKREGION_XaYb:CLOCKREGION_XcYd, PART xcvu9p-fsgd2104-2L-e,
# PLACE / PHYS / ROUTE directives (Default). Ports are unconstrained (no I/O delays): the paths
# timed are the array's own. Prints OTPU_* summary lines; reports go to <out>.
set out [file normalize [lindex $argv 0]]
array set o {PUMP 1 PHOLD 1 CORE_NS 6.666 PBLOCK SLR2 PART xcvu9p-fsgd2104-2L-e PLACE Default
             PHYS Default ROUTE Default}
foreach kv [lrange $argv 1 end] {
  set i [string first = $kv]
  set k [string range $kv 0 [expr {$i - 1}]]
  if {$i < 1 || ![info exists o($k)]} { error "ooc_mxu.tcl: unknown argument $kv" }
  set o($k) [string range $kv [expr {$i + 1}] end]
}
set root [file normalize [file join [file dirname [info script]] ../../..]]
file mkdir $out
foreach k [lsort [array names o]] { puts "OTPU_ARG $k $o($k)" }

create_project -in_memory -part $o(PART)
set_property source_mgmt_mode None [current_project]
read_verilog -sv [list $root/rtl/vpu/otpu_fp.sv $root/rtl/top/otpu_pkg.sv $root/rtl/vpu/otpu_fpipe.sv \
  $root/rtl/mxu/otpu_mxu.sv $root/rtl/tech/xilinxus/otpu_tech_dsp.sv \
  $root/rtl/tech/generic/otpu_tech_ram.sv]
# bcu1525-x2's otpu_mxu parameters (otpu_slice's instance at the target's rtl_params)
set gen [list D=128 MCOLS=8 ROWS=16 DEPTH=2048 LANES=8 TLANES=8 CL=16 E=8 WG=8 AB=128 XPOST=1 QB=2 \
  CHASH=1 TQ=1 RF_STYLE="distributed" FIFO_STYLE="ultra" QD=4 WASK=1 AASK=1 KT=1 DAW=36 \
  PUMP=$o(PUMP) PHOLD=$o(PHOLD)]
set ga {}
foreach g $gen { lappend ga -generic $g }
synth_design -top otpu_mxu -part $o(PART) -mode out_of_context -flatten_hierarchy rebuilt \
  -verilog_define SYNTHESIS {*}$ga
create_clock -period $o(CORE_NS) -name core_clk [get_ports clk]
create_clock -period [expr {$o(CORE_NS) / 2.0}] -name core_clk2x [get_ports clk_f]
set cs [get_clocks core_clk]
set cf [get_clocks core_clk2x]
source $root/boards/ypcb-00338/constraints/otpu_clk2x.tcl
if {$o(PBLOCK) ne "none"} {
  set pb [create_pblock pb_mxu]
  add_cells_to_pblock $pb -top
  resize_pblock $pb -add $o(PBLOCK)
  puts "OTPU_PBLOCK $o(PBLOCK)"
}
report_utilization -file $out/util_synth.txt
opt_design
place_design -directive $o(PLACE)
phys_opt_design -directive $o(PHYS)
route_design -directive $o(ROUTE)
write_checkpoint -force $out/routed.dcp
report_utilization -file $out/util.txt
report_utilization -hierarchical -hierarchical_depth 4 -file $out/util_hier.txt
report_timing_summary -max_paths 20 -file $out/timing_summary.txt
report_clock_interaction -file $out/clock_interaction.txt
report_exceptions -file $out/exceptions.txt
report_methodology -file $out/methodology.txt

set dsp [llength [get_cells -hierarchical -filter {REF_NAME == DSP48E2}]]
set lut [llength [get_cells -hierarchical -filter {REF_NAME =~ LUT*}]]
set ff  [llength [get_cells -hierarchical -filter {IS_SEQUENTIAL && REF_NAME =~ FD*}]]
puts "OTPU_UTIL DSP48E2 $dsp LUT $lut FF $ff"
# the worst path by clock pair, one line each, its 10 worst in timing_<tag>.txt
proc otpu_path {tag from to} {
  global out
  set q [get_timing_paths -quiet -from $from -to $to -setup -max_paths 10 -nworst 1]
  set h [get_timing_paths -quiet -from $from -to $to -hold -max_paths 1 -nworst 1]
  if {![llength $q]} { puts "OTPU_PATH $tag none"; return }
  set w [lindex $q 0]
  puts "OTPU_PATH $tag setup [get_property SLACK $w] req [get_property REQUIREMENT $w] levels [get_property LOGIC_LEVELS $w] [get_property STARTPOINT_PIN $w] -> [get_property ENDPOINT_PIN $w] | hold [get_property SLACK [lindex $h 0]]"
  report_timing -of_objects $q -file $out/timing_$tag.txt
}
otpu_path SS $cs $cs
otpu_path FF $cf $cf
otpu_path SF $cs $cf
otpu_path FS $cf $cs
set p [lindex [get_timing_paths -setup -max_paths 1 -nworst 1] 0]
set h [lindex [get_timing_paths -hold -max_paths 1 -nworst 1] 0]
set fails [llength [get_timing_paths -quiet -setup -max_paths 100000 -slack_lesser_than 0]]
puts "OTPU_WNS [get_property SLACK $p] OTPU_WHS [get_property SLACK $h] OTPU_FAILING_ENDPOINTS_MAX100K $fails"
puts "OTPU_DONE"
