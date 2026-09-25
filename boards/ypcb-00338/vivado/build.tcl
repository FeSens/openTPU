# Build the bitstream: synthesis, implementation (opt, place, phys_opt, route, post-route
# phys_opt), reports and bitstream.
#   vivado -mode batch -source build.tcl -tclargs [OUT_DIR] [JOBS] [impl|full] [IMPL_STRATEGY]
# With "impl", the existing synthesis (and IP runs) are kept and only implementation reruns:
# for constraint or implementation-strategy changes that do not touch the RTL.
# Expects the project from create_project.tcl. Outputs in OUT_DIR/reports and
# OUT_DIR/otpu.bit (+ otpu.ltx when debug cores exist, + otpu.bin/.mcs for the BPI flash);
# reports/power.json feeds otpu-smi's power estimate.

set here [file normalize [file dirname [info script]]]
set root [file normalize $here/../../..]
set out [expr {[llength $argv] > 0 ? [file normalize [lindex $argv 0]] : "$root/build/vivado"}]
set jobs [expr {[llength $argv] > 1 ? [lindex $argv 1] : 8}]
set impl_only [expr {[lindex $argv 2] eq "impl"}]
# an implementation strategy (e.g. Performance_Explore) when the default run misses timing
set strategy [lindex $argv 3]

open_project $out/otpu.xpr
file mkdir $out/reports

if {!$impl_only} {
  # ---- IP (block design) out-of-context runs first
  generate_target all [get_files otpu_bd.bd]
  export_ip_user_files -of_objects [get_files otpu_bd.bd] -no_script -sync -force -quiet
  create_ip_run [get_files otpu_bd.bd]

  # ---- synthesis
  reset_run synth_1
  launch_runs synth_1 -jobs $jobs
  wait_on_run synth_1
  if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { error "synthesis failed" }
  open_run synth_1
  report_utilization -hierarchical -hierarchical_depth 4 -file $out/reports/synth_util_hier.rpt
  report_timing_summary -max_paths 20 -file $out/reports/synth_timing.rpt
  close_design
} else {
  if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { error "impl: no finished synthesis to reuse" }
  # constraint edits mark synthesis out of date; keep its netlist anyway
  set_property NEEDS_REFRESH false [get_runs synth_1]
}

# ---- implementation to the routed design
if {$strategy ne ""} { set_property strategy $strategy [get_runs impl_1] }
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
reset_run impl_1
launch_runs impl_1 -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "implementation failed" }
open_run impl_1

report_timing_summary -max_paths 50 -report_unconstrained -warn_on_violation \
  -file $out/reports/timing_summary.rpt
report_timing -max_paths 30 -sort_by group -nworst 1 -file $out/reports/timing_worst.rpt
report_clock_interaction -file $out/reports/clock_interaction.rpt
report_clocks -file $out/reports/clocks.rpt
report_cdc -details -file $out/reports/cdc.rpt
report_utilization -file $out/reports/util.rpt
report_utilization -hierarchical -hierarchical_depth 5 -file $out/reports/util_hier.rpt
report_drc -file $out/reports/drc.rpt
report_methodology -file $out/reports/methodology.rpt
# power: hierarchical enough to reach the slice's units (otpu-smi's estimate; opentpu/host/power.py)
report_power -hierarchical_depth 12 -file $out/reports/power.rpt
report_power -hierarchical_depth 12 -format xml -file $out/reports/power.xml
if {[catch {exec env PYTHONPATH=$root python3 -m opentpu.host.power $out/reports/power.xml \
              -o $out/reports/power.json} msg]} {
  puts "power.json not written ($msg): run python3 -m opentpu.host.power reports/power.rpt -o reports/power.json"
} else { puts $msg }
report_io -file $out/reports/io.rpt

# per-clock worst slack in one line each (quick look)
set fh [open $out/reports/SUMMARY.txt w]
set wns [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]]
set whs [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -hold] 0]]
puts $fh "WNS $wns ns   WHS $whs ns"
foreach clk [get_clocks] {
  set p [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup -to $clk] 0]
  if {$p ne ""} {
    set period [get_property PERIOD $clk]
    set slack [get_property SLACK $p]
    set fmax [expr {$slack eq "" ? "n/a" : [format %.1f [expr {1000.0 / ($period - $slack)}]]}]
    puts $fh [format "%-40s period %7.3f ns  slack %8s ns  fmax %s MHz" $clk $period $slack $fmax]
  }
}
close $fh
puts [exec cat $out/reports/SUMMARY.txt]

# ---- bitstream, written from the routed design opened above
source $here/bitstream_pre.tcl
write_bitstream -force $out/otpu.bit
if {[llength [get_debug_cores -quiet]]} { write_debug_probes -force $out/otpu.ltx }
# BPI x16 flash image (for a permanent load; see docs/board.md). The flash has address lines
# A1..A25 on a x16 bus: 64 MB.
write_cfgmem -force -format mcs -size 64 -interface BPIx16 -loadbit "up 0x0 $out/otpu.bit" \
  $out/otpu.mcs
puts "bitstream: $out/otpu.bit"
if {$wns < 0} { puts "WARNING: timing not met (WNS $wns ns) -- see reports/timing_summary.rpt" }
