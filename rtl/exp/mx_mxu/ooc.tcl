# EXPERIMENTAL (docs/mx_mxu.md): out-of-context synthesis + implementation of the MX MXU
# experiment's tops (mx_top: the MX datapath; base_top: today's) on one part. Vivado on the build host:
#   vivado -mode batch -nojournal -source rtl/exp/mx_mxu/ooc.tcl -tclargs <out> <top> <fam> \
#     <part> [GENERIC=VALUE ...]
# top: mx_top | base_top; fam: x7 (DSP48E1 cells, mx_top pumped at 2 x clk_s from an MMCM) |
# us (DSP48E2 cells, mx_top packed, one clock). clk_s 133.33 MHz. Prints OTPU_* summary lines;
# reports go to <out>.
set out  [file normalize [lindex $argv 0]]
set top  [lindex $argv 1]
set fam  [lindex $argv 2]
set part [lindex $argv 3]
set gens [lrange $argv 4 end]
set root [file normalize [file join [file dirname [info script]] ../../..]]
file mkdir $out

create_project -in_memory -part $part
set_property source_mgmt_mode None [current_project]
set src [list $root/rtl/top/otpu_pkg.sv $root/rtl/vpu/otpu_fp.sv $root/rtl/vpu/otpu_fpipe.sv \
  $root/rtl/mxu/otpu_mxu.sv]
if {$fam eq "x7"} {
  lappend src $root/rtl/tech/xilinx7/otpu_tech_dsp.sv $root/rtl/exp/dsp_pump/otpu_pump_col.sv \
    $root/rtl/exp/mx_mxu/otpu_mx_x7.sv
} else {
  lappend src $root/rtl/tech/xilinxus/otpu_tech_dsp.sv $root/rtl/exp/mx_mxu/otpu_mx_us.sv
}
lappend src $root/rtl/exp/mx_mxu/otpu_mx_lane.sv $root/rtl/exp/mx_mxu/mx_top.sv \
  $root/rtl/exp/mx_mxu/base_top.sv
read_verilog -sv $src
set gargs {}
foreach g $gens { lappend gargs -generic $g }
set t0 [clock seconds]
synth_design -top $top -part $part -mode out_of_context -flatten_hierarchy rebuilt \
  -verilog_define SYNTHESIS {*}$gargs
puts "OTPU_SYNTH_S [expr {[clock seconds] - $t0}]"
create_clock -period 7.5 -name clk_in [get_ports clk_in]
set cs [get_clocks -quiet -of_objects [get_nets -quiet -hierarchical -filter {NAME =~ *u_clk/clk_s}]]
if {![llength $cs]} { set cs [get_clocks clk_in] }
set cf {}
if {$fam eq "x7"} {
  set cs [get_clocks -quiet -of_objects [get_pins -hierarchical -filter {NAME =~ *u_clk/u_mmcm/CLKOUT0}]]
  set cf [get_clocks -quiet -of_objects [get_pins -hierarchical -filter {NAME =~ *u_clk/u_mmcm/CLKOUT1}]]
}
foreach {t c} [list slow $cs fast $cf] {
  if {[llength $c]} { puts "OTPU_CLOCK $t [get_property NAME $c] [get_property PERIOD $c]" }
}
# the pumped DSPs' operand pins load only on the slow edges: a slow register in front of them
# has the whole slow period (rtl/exp/dsp_pump/ooc.tcl)
set n 0
if {$fam eq "x7" && $top eq "mx_top"} {
  set ppe [get_cells -hierarchical -filter {REF_NAME == DSP48E1 && NAME =~ *u_ppe*}]
  set pins [get_pins -of_objects $ppe -filter {REF_PIN_NAME =~ A[*] || REF_PIN_NAME =~ B[*] || REF_PIN_NAME =~ D[*]}]
  set n [llength $pins]
  set_multicycle_path 2 -setup -end -from $cs -to $pins
  set_multicycle_path 1 -hold -end -from $cs -to $pins
}
puts "OTPU_MCP_PINS $n"
report_utilization -file $out/util_synth.txt
set t0 [clock seconds]
opt_design
place_design
phys_opt_design
route_design
puts "OTPU_IMPL_S [expr {[clock seconds] - $t0}]"
write_checkpoint -force $out/routed.dcp
report_utilization -file $out/util.txt
report_utilization -hierarchical -hierarchical_depth 8 -file $out/util_hier.txt
report_timing_summary -max_paths 10 -file $out/timing_summary.txt
report_clock_interaction -file $out/clock_interaction.txt
report_pulse_width -file $out/pulse_width.txt
report_methodology -file $out/methodology.txt
report_power -file $out/power.txt
set tags [list SS $cs $cs]
if {[llength $cf]} { lappend tags FF $cf $cf SF $cs $cf FS $cf $cs }
foreach {tag from to} $tags {
  set q [lindex [get_timing_paths -quiet -from $from -to $to -setup -max_paths 1 -nworst 1] 0]
  set h [lindex [get_timing_paths -quiet -from $from -to $to -hold -max_paths 1 -nworst 1] 0]
  if {[llength $q]} {
    puts "OTPU_PATH $tag setup [get_property SLACK $q] req [get_property REQUIREMENT $q] levels [get_property LOGIC_LEVELS $q] [get_property STARTPOINT_PIN $q] -> [get_property ENDPOINT_PIN $q] | hold [get_property SLACK $h]"
    report_timing -from $from -to $to -max_paths 10 -nworst 1 -file $out/timing_$tag.txt
  } else {
    puts "OTPU_PATH $tag none"
  }
}
set p [lindex [get_timing_paths -setup -max_paths 1 -nworst 1] 0]
set h [lindex [get_timing_paths -hold -max_paths 1 -nworst 1] 0]
puts "OTPU_WNS [get_property SLACK $p] OTPU_WHS [get_property SLACK $h]"
puts "OTPU_DONE"
