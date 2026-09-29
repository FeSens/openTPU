# Out-of-context Vivado check of otpu_mem_ch's clock-domain crossings, with its constraints
# (boards/ypcb-00338/constraints/otpu_mem_ch.xdc, read scoped and unmanaged as the build will):
# synthesis, then report_cdc, report_methodology, the clock interaction, the timing exceptions
# and whether every query of the XDC found its cells; with ROUTE=1 also place and route, the
# timing summary, the bus skew and the CDC and methodology reports again. Non-project mode, on
# omarchy (never on the Mac):
#
#   vivado -mode batch -nojournal -source tools/memch_ooc.tcl -tclargs <outdir> [RMW] [ROUTE]
#
# RMW: 1 (LiteDRAM, the default) or 0 (MIG). ROUTE: 0 (the default) or 1. Clock periods (ns) from
# the environment: MEMCH_CLK (core, default 7.968: 125.5 MHz, about the best routed core clock),
# MEMCH_UCLK (7.5: 133.33 MHz, DDR3-1066), MEMCH_XCLK (8.0: axi_aclk). The module's ports get
# input and output delays of 30% of their clock's period (n_* rst: clk, x_* xrst: xclk, c_* urst:
# uclk), so every path is timed. Reports go to <outdir>; the summary is printed at the end.
set out [file normalize [lindex $argv 0]]
set rmw [expr {[llength $argv] > 1 ? [lindex $argv 1] : 1}]
set route [expr {[llength $argv] > 2 ? [lindex $argv 2] : 0}]
set root [file normalize [file join [file dirname [info script]] ..]]
proc env_or {name dflt} { if {[info exists ::env($name)]} { return $::env($name) }; return $dflt }
set t_clk [env_or MEMCH_CLK 7.968]
set t_ucl [env_or MEMCH_UCLK 7.5]
set t_xcl [env_or MEMCH_XCLK 8.0]
file mkdir $out

set_part xc7k480t-ffg1156-2
read_verilog -sv [list $root/rtl/boards/ypcb-00338/otpu_afifo.sv $root/rtl/boards/ypcb-00338/otpu_mem_ch.sv]
synth_design -top otpu_mem_ch -mode out_of_context -generic RMW=$rmw -flatten_hierarchy rebuilt

create_clock -name clk -period $t_clk [get_ports clk]
create_clock -name uclk -period $t_ucl [get_ports uclk]
create_clock -name xclk -period $t_xcl [get_ports xclk]
foreach {pats c t} [list {n_* rst} clk $t_clk {x_* xrst} xclk $t_xcl {c_* urst} uclk $t_ucl] {
  set ins  [get_ports -quiet -filter {DIRECTION == IN} $pats]
  set outs [get_ports -quiet -filter {DIRECTION == OUT} $pats]
  if {[llength $ins]}  { set_input_delay  -clock $c [expr {0.3 * $t}] $ins }
  if {[llength $outs]} { set_output_delay -clock $c [expr {0.3 * $t}] $outs }
}
set xdc $root/boards/ypcb-00338/constraints/otpu_mem_ch.xdc
read_xdc -unmanaged -ref otpu_mem_ch $xdc
# the scoped read must have applied the constraints (if it did not, read the file unscoped, the
# same thing when otpu_mem_ch is the top, and say so)
proc n_maxdelay {} { return [regexp -all -line {max_delay} [report_exceptions -return_string]] }
set scoped [n_maxdelay]
if {$scoped == 0} { read_xdc -unmanaged $xdc }
set nmd [n_maxdelay]

# every query of the XDC must find cells (the top is otpu_mem_ch, so the names are the same)
set empty {}
set pats {a_rs1_reg x_rs1_reg a_hs1_reg x_hs1_reg a_wacc_g_reg[*] a_wacc_s1_reg[*] x_wacc_g_reg[*] x_wacc_s1_reg[*]}
foreach f {u_aq u_ad u_ar u_xq u_xd u_xr} {
  lappend pats $f/wgray_reg\[*\] $f/wgray_r1_reg\[*\] $f/rgray_reg\[*\] $f/rgray_w1_reg\[*\]
}
set qlog {}
foreach p $pats {
  set n [llength [get_cells -quiet $p]]
  lappend qlog "  $p: $n cells"
  if {$n == 0} { lappend empty $p }
}
foreach f {u_aq u_ad u_ar u_xq u_xd u_xr} {
  set n [llength [get_pins -quiet -filter {DIRECTION == OUT && IS_LEAF} -of_objects [get_cells -quiet -hierarchical -filter "NAME =~ *$f/mem_reg*"]]]
  lappend qlog "  $f RAM outputs: $n pins"
  if {$n == 0} { lappend empty "$f RAM outputs" }
}

proc reports {out tag} {
  report_clock_interaction -delay_type min_max -file $out/clock_interaction_$tag.rpt
  report_cdc -details -file $out/cdc_details_$tag.rpt
  report_cdc -file $out/cdc_$tag.rpt
  report_methodology -file $out/methodology_$tag.rpt
  report_timing_summary -max_paths 20 -file $out/timing_$tag.rpt
}
reports $out synth
report_exceptions -file $out/exceptions.rpt
catch {report_exceptions -ignored -file $out/exceptions_ignored.rpt}
check_timing -verbose -file $out/check_timing.rpt
report_utilization -hierarchical -file $out/util_hier.rpt
report_utilization -file $out/util.rpt
if {$route} {
  opt_design
  place_design
  route_design
  reports $out routed
  report_bus_skew -file $out/bus_skew.rpt
  write_checkpoint -force $out/otpu_mem_ch_routed.dcp
}

puts "==== otpu_mem_ch OOC (RMW=$rmw, route=$route, clk $t_clk / uclk $t_ucl / xclk $t_xcl ns): $out"
puts [expr {$scoped ? "XDC read scoped (-ref otpu_mem_ch): $nmd max_delay lines in report_exceptions"
                    : "XDC: the scoped read applied nothing; read unscoped instead: $nmd max_delay lines"}]
puts "XDC queries:"
puts [join $qlog "\n"]
puts [expr {[llength $empty] ? "XDC queries that matched nothing: $empty" : "XDC queries: all matched"}]
puts "Clock interaction:"
puts [report_clock_interaction -delay_type min_max -return_string]
puts "CDC summary:"
puts [report_cdc -return_string]
puts "Methodology summary:"
set m [report_methodology -return_string]
set i [string first "Summary" $m]
puts [string range $m $i [expr {$i + 2500}]]
