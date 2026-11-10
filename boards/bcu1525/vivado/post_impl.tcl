# The reports, SUMMARY.txt, the bitstream and the flash image of an implemented design (open):
# build.tcl's (after open_run impl_1) and impl_resume.tcl's. Needs $out (the build's OUT_DIR).
report_timing_summary -max_paths 50 -report_unconstrained -warn_on_violation \
  -file $out/reports/timing_summary.rpt
report_timing -max_paths 30 -sort_by group -nworst 1 -file $out/reports/timing_worst.rpt
report_design_analysis -congestion -file $out/reports/congestion.rpt
report_clock_interaction -file $out/reports/clock_interaction.rpt
report_cdc -details -file $out/reports/cdc.rpt
report_utilization -file $out/reports/util.rpt
report_utilization -slr -file $out/reports/util_slr.rpt
report_utilization -hierarchical -hierarchical_depth 5 -file $out/reports/util_hier.rpt
report_drc -file $out/reports/drc.rpt
report_methodology -file $out/reports/methodology.rpt
report_power -hierarchical_depth 12 -file $out/reports/power.rpt
report_io -file $out/reports/io.rpt
# the pulse-width skew checks (the MIG PHYs' Min / Max Skew, pulse_skew.tcl): reports/pulse_skew.rpt,
# the failing ones' count and worst in SUMMARY.txt
if {[catch {report_pulse_width -min_skew -max_skew -significant_digits 3 \
               -file $out/reports/pulse_skew.rpt} e]} {
  puts "post_impl.tcl: report_pulse_width -min_skew -max_skew: $e; -all instead"
  report_pulse_width -all -significant_digits 3 -file $out/reports/pulse_skew.rpt
}
source $here/pulse_skew.tcl
lassign [::otpu::pulse_skew $out/reports/pulse_skew.rpt] skew_n skew_ws skew_wl

set fh [open $out/reports/SUMMARY.txt w]
set wns [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]]
set whs [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -hold] 0]]
puts $fh "WNS $wns ns   WHS $whs ns"
puts $fh [expr {$skew_n ? "pulse-width skew checks failing: $skew_n, worst $skew_wl" \
                         : "pulse-width skew checks failing: 0"}]
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

# x2 is a functional development image: hold >= 0 and setup >= OTPU_X2_WNS_MIN (default 0) before a
# bitstream is emitted. Run its netlist/placement checks here too: a partial implementation is not a qual.
if {[llength [get_cells -quiet u_link]]} { source $here/../x2/post_x2.tcl }

write_bitstream -force $out/otpu.bit
if {[llength [get_debug_cores -quiet]]} { write_debug_probes -force $out/otpu.ltx }
# the configuration flash: Micron MT25QU01G, 1 Gb (128 MB), SPI x4 (UG1268)
write_cfgmem -force -format mcs -size 128 -interface SPIx4 -loadbit "up 0x0 $out/otpu.bit" $out/otpu.mcs
puts "bitstream: $out/otpu.bit"
if {$wns < 0} { puts "WARNING: timing not met (WNS $wns ns) -- see reports/timing_summary.rpt" }
