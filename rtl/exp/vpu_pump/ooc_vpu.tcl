# EXPERIMENTAL (docs/vpu_pump.md): out-of-context synthesis + implementation of otpu_vpu (the VPU
# with the stream engine's core side, otpu_se_comp and otpu_se_tail) on the YPCB's part, at one
# clock. Vivado on the build host only:
#   vivado -mode batch -nojournal -source rtl/exp/vpu_pump/ooc_vpu.tcl -tclargs <out> <PERIOD_NS> \
#     <LANES> <CL> <HAS_SE> [WBUF] [PART]
# Production: 7.5 8 2 1 1 (CL is unused with the stream engine: COMP8). Prints OTPU_* lines.
set out [file normalize [lindex $argv 0]]
set period [lindex $argv 1]
set lanes [lindex $argv 2]
set cl [lindex $argv 3]
set se [lindex $argv 4]
set wbuf [expr {[llength $argv] > 5 ? [lindex $argv 5] : 1}]
set part [expr {[llength $argv] > 6 ? [lindex $argv 6] : "xc7k480tffg1156-2"}]
set root [file normalize [file join [file dirname [info script]] ../../..]]
file mkdir $out

create_project -in_memory -part $part
set_property source_mgmt_mode None [current_project]
read_verilog -sv [list $root/rtl/vpu/otpu_fp.sv $root/rtl/vpu/otpu_fpipe.sv \
  $root/rtl/top/otpu_pkg.sv $root/rtl/tech/generic/otpu_tech_ram.sv $root/rtl/vpu/otpu_vtree.sv \
  $root/rtl/vpu/otpu_se_comp.sv $root/rtl/vpu/otpu_se_tail.sv $root/rtl/vpu/otpu_vpu.sv]
synth_design -top otpu_vpu -part $part -mode out_of_context -flatten_hierarchy rebuilt \
  -verilog_define SYNTHESIS -generic LANES=$lanes -generic CL=$cl -generic HAS_SE=$se \
  -generic WBUF=$wbuf -generic SID=0
create_clock -period $period -name clk [get_ports clk]
report_utilization -file $out/util_synth.txt
report_utilization -hierarchical -hierarchical_depth 3 -file $out/util_hier_synth.txt
opt_design
place_design
phys_opt_design
route_design
report_utilization -file $out/util.txt
report_utilization -hierarchical -hierarchical_depth 3 -file $out/util_hier.txt
report_timing_summary -max_paths 20 -file $out/timing_summary.txt
set f [open $out/util.txt]; set t [read $f]; close $f
foreach {k pat} {LUT "LUT as Logic" LUTRAM "LUT as Memory" FF "Slice Registers" DSP "DSPs" SLICE "Slice" BRAM "Block RAM Tile"} {
  if {[regexp -line "^\\|\\s*$pat\\s*\\|\\s*(\[0-9.\]+)" $t -> v]} { puts "OTPU_UTIL $k $v" }
}
set n 0
foreach q [get_timing_paths -setup -max_paths 20 -nworst 1] {
  if {$n < 5} {
    puts "OTPU_PATH [get_property SLACK $q] levels [get_property LOGIC_LEVELS $q] [get_property STARTPOINT_PIN $q] -> [get_property ENDPOINT_PIN $q]"
  }
  incr n
}
set p [lindex [get_timing_paths -setup -max_paths 1 -nworst 1] 0]
set h [lindex [get_timing_paths -hold -max_paths 1 -nworst 1] 0]
puts "OTPU_WNS [get_property SLACK $p] OTPU_WHS [get_property SLACK $h]"
puts "OTPU_DONE"
