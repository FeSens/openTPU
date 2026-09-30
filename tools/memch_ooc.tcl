# Out-of-context Vivado check of otpu_mem_ch's clock-domain crossings, with its constraints
# (boards/ypcb-00338/constraints/otpu_mem_ch.tcl, read scoped and unmanaged as the build will):
# synthesis, then report_cdc (with the XDC's waivers; also every check per endpoint, where the
# Vivado version has -all_checks_per_endpoint), report_methodology, the waivers, the clock
# interaction, the timing exceptions (each set_max_delay applied once) and whether every query of
# the XDC found its objects; with ROUTE=1 also place and route, the timing summary, the bus skew
# and the CDC and methodology reports again. Last, every source of a path between the clocks is
# checked against the crossings the XDC constrains. Non-project mode, on omarchy (never on the
# Mac):
#
#   vivado -mode batch -nojournal -source tools/memch_ooc.tcl -tclargs <outdir> [ROUTE]
#
# ROUTE: 0 (the default) or 1. Clock periods (ns) from
# the environment: MEMCH_CLK (core, default 7.968: 125.5 MHz, about the best routed core clock),
# MEMCH_UCLK (7.5: 133.33 MHz, DDR3-1066), MEMCH_XCLK (8.0: axi_aclk). The module's ports get
# input and output delays of 30% of their clock's period (n_* rst: clk, x_* xrst: xclk, c_* urst:
# uclk), so every path is timed. Reports go to <outdir>; the summary is printed at the end.
set out [file normalize [lindex $argv 0]]
set route [expr {[llength $argv] > 1 ? [lindex $argv 1] : 0}]
set root [file normalize [file join [file dirname [info script]] ..]]
proc env_or {name dflt} { if {[info exists ::env($name)]} { return $::env($name) }; return $dflt }
set t_clk [env_or MEMCH_CLK 7.968]
set t_ucl [env_or MEMCH_UCLK 7.5]
set t_xcl [env_or MEMCH_XCLK 8.0]
file mkdir $out

set_part xc7k480t-ffg1156-2
read_verilog -sv [list $root/rtl/boards/ypcb-00338/otpu_afifo.sv $root/rtl/boards/ypcb-00338/otpu_mem_ch.sv]
synth_design -top otpu_mem_ch -mode out_of_context -flatten_hierarchy rebuilt

create_clock -name clk -period $t_clk [get_ports clk]
create_clock -name uclk -period $t_ucl [get_ports uclk]
create_clock -name xclk -period $t_xcl [get_ports xclk]
foreach {pats c t} [list {n_* rst} clk $t_clk {x_* xrst} xclk $t_xcl {c_* urst} uclk $t_ucl] {
  set ins  [get_ports -quiet -filter {DIRECTION == IN} $pats]
  set outs [get_ports -quiet -filter {DIRECTION == OUT} $pats]
  if {[llength $ins]}  { set_input_delay  -clock $c [expr {0.3 * $t}] $ins }
  if {[llength $outs]} { set_output_delay -clock $c [expr {0.3 * $t}] $outs }
}
set xdc $root/boards/ypcb-00338/constraints/otpu_mem_ch.tcl
set fh [open $xdc]; set xdc_text [read $fh]; close $fh
set n_xmd [regexp -all -line {^\s*set_max_delay\M} $xdc_text]
set n_xwv [regexp -all -line {^\s*create_waiver\M} $xdc_text]
read_xdc -unmanaged -ref otpu_mem_ch $xdc
# The scoped read must have applied every set_max_delay once: report_exceptions lists the active
# ones (max_dpo=, or max= without -datapath_only), -ignored the overridden ones (a second read).
# Only if the scoped read applied none is the file read unscoped (the same thing when otpu_mem_ch
# is the top), and the summary says so.
proc n_maxdelay {args} { return [regexp -all -line {\mmax(_dpo)?=} [report_exceptions {*}$args -return_string]] }
proc n_waivers {} { if {[catch {llength [get_waivers -quiet]} n]} { return "?" }; return $n }
set scoped [expr {[n_maxdelay] > 0}]
if {!$scoped} { read_xdc -unmanaged $xdc }
set nmd [n_maxdelay]
set nmd_ign [n_maxdelay -ignored]
set nwv [n_waivers]

# every query of the XDC must find cells (the top is otpu_mem_ch, so the names are the same)
set empty {}
set pats {a_rs1_reg x_rs1_reg a_hs1_reg x_hs1_reg e_s1_reg a_wacc_g_reg[*] a_wacc_s1_reg[*] x_wacc_g_reg[*] x_wacc_s1_reg[*]}
foreach f {u_aq u_ad u_ar u_xq u_xd u_xr} {
  lappend pats $f/wgray_reg\[*\] $f/wbin_reg\[*\] $f/wgray_r1_reg\[*\] $f/rgray_reg\[*\] $f/rbin_reg\[*\]* $f/rgray_w1_reg\[*\]
}
# the waivers' endpoints
lappend pats g_port\[*\].oc_reg\[*\] g_port\[*\].oc_v_reg g_port\[*\].u_oq/wp_reg* \
  g_port\[*\].u_of/wp_reg* g_port\[*\].u_tag/wp_reg* run_reg\[*\] cur_x_reg a_out_reg\[*\] \
  x_out_reg\[*\] a_pend_reg\[*\] x_pend_reg\[*\] a_seq_reg\[*\] x_seq_reg\[*\] a_nq_reg\[*\] \
  x_nq_reg\[*\] n_rdata_reg\[*\] rm_busy_reg rm_x_reg
set qlog {}
foreach p $pats {
  set n [llength [get_cells -quiet $p]]
  lappend qlog "  $p: $n cells"
  if {$n == 0} { lappend empty $p }
}
foreach f {u_aq u_ad u_ar u_xq u_xd u_xr} {
  set cs [get_cells -quiet -hierarchical -filter "NAME =~ *$f/mem_reg*"]
  set n [llength [get_pins -quiet -filter {DIRECTION == OUT && IS_LEAF} -of_objects $cs]]
  set k [llength [get_pins -quiet -filter {IS_LEAF && REF_PIN_NAME == CLK} -of_objects $cs]]
  lappend qlog "  $f RAM outputs: $n pins, write clock pins: $k"
  if {$n == 0 || $k == 0} { lappend empty "$f RAM" }
}
foreach f {u_oq u_of u_tag} {
  set n [llength [get_pins -quiet -filter {IS_LEAF && (REF_PIN_NAME == I || REF_PIN_NAME == WE)} -of_objects [get_cells -quiet -hierarchical -filter "NAME =~ *$f/mem_reg*"]]]
  lappend qlog "  $f RAM data and write-enable inputs: $n pins"
  if {$n == 0} { lappend empty "$f RAM inputs" }
}

# report_cdc -details names one startpoint per endpoint. This lists every group of registers (a
# register or RAM array, replicas included) and every input port with a path from one of the
# three clocks into another, and marks those that are not one of the constrained crossings: the
# FIFOs' RAMs and pointers (a pointer's top bit may sit in the binary register), the reset,
# hold and error-bit synchronizers' sources and the write-accept gray counts.
set xs_expect {
  clk>uclk  {^(u_aq|u_ad)/(mem_reg|wgray_reg|wbin_reg)$|^u_ar/(rgray_reg|rbin_reg)$|^a_req_reg$}
  xclk>uclk {^(u_xq|u_xd)/(mem_reg|wgray_reg|wbin_reg)$|^u_xr/(rgray_reg|rbin_reg)$|^x_req_reg$}
  uclk>clk  {^u_ar/(mem_reg|wgray_reg|wbin_reg)$|^(u_aq|u_ad)/(rgray_reg|rbin_reg)$|^(a_hold_reg|a_wacc_g_reg|c_err_reg)$}
  uclk>xclk {^u_xr/(mem_reg|wgray_reg|wbin_reg)$|^(u_xq|u_xd)/(rgray_reg|rbin_reg)$|^(x_hold_reg|x_wacc_g_reg)$}
}
proc cross_sources {expect} {
  set clks {clk uclk xclk}
  set grp [dict create]
  set ckp [get_pins -quiet -hierarchical -filter {IS_LEAF && (REF_PIN_NAME == C || REF_PIN_NAME == CLK)}]
  foreach n [get_property NAME $ckp] {
    set k [regsub {/[^/]+$} $n {}]
    set k [regsub {/mem_reg.*$} $k {/mem_reg}]
    set k [regsub {(\[[0-9]+\])?(_rep(__[0-9]+)?)?$} $k {}]
    dict lappend grp $k $n
  }
  set ports [dict create]
  foreach n [get_property NAME [all_inputs]] {
    set k [regsub {\[[0-9]+\]$} $n {}]
    if {$k ni $clks} { dict lappend ports $k $n }
  }
  set lines {}
  set bad 0
  dict for {k ns} $grp {
    set sc [get_property -quiet NAME [get_clocks -quiet -of_objects [get_pins [lindex $ns 0]]]]
    if {[llength $sc] != 1} continue
    foreach dc $clks {
      if {$dc eq $sc} continue
      if {![llength [get_timing_paths -quiet -from [get_pins $ns] -to [get_clocks $dc] -max_paths 1]]} continue
      set ok [expr {[dict exists $expect $sc>$dc] && [regexp [dict get $expect $sc>$dc] $k]}]
      if {!$ok} { incr bad }
      lappend lines [format "  %-5s -> %-5s %-24s %5d cells %s" $sc $dc $k [llength $ns] [expr {$ok ? "" : "  NOT A CONSTRAINED CROSSING"}]]
    }
  }
  dict for {k ns} $ports {
    foreach dc $clks {
      set tp [get_timing_paths -quiet -from [get_ports $ns] -to [get_clocks $dc] -max_paths 1]
      if {![llength $tp]} continue
      set sc [get_property -quiet STARTPOINT_CLOCK $tp]
      if {![catch {get_property NAME $sc} n] && $n ne ""} { set sc $n }
      if {$sc eq "" || $sc eq $dc} continue
      set ok [expr {[dict exists $expect $sc>$dc] && [regexp [dict get $expect $sc>$dc] $k]}]
      if {!$ok} { incr bad }
      lappend lines [format "  %-5s -> %-5s %-24s %5d ports%s" $sc $dc $k [llength $ns] [expr {$ok ? "" : "  NOT A CONSTRAINED CROSSING"}]]
    }
  }
  return [list $bad [lsort $lines]]
}

proc reports {out tag} {
  report_clock_interaction -delay_type min_max -file $out/clock_interaction_$tag.rpt
  report_cdc -details -file $out/cdc_details_$tag.rpt
  catch {report_cdc -details -all_checks_per_endpoint -file $out/cdc_all_$tag.rpt}
  report_cdc -file $out/cdc_$tag.rpt
  report_methodology -file $out/methodology_$tag.rpt
  report_timing_summary -max_paths 20 -file $out/timing_$tag.rpt
}
reports $out synth
catch {report_waivers -file $out/waivers.rpt}
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
if {[catch {cross_sources $xs_expect} xs]} { set xs [list "?" [list "  cross_sources failed: $xs"]] }

puts "==== otpu_mem_ch OOC (route=$route, clk $t_clk / uclk $t_ucl / xclk $t_xcl ns): $out"
puts [expr {$scoped ? "XDC read scoped (-ref otpu_mem_ch)" : "XDC: the scoped read applied nothing; read unscoped instead"}]
puts "XDC set_max_delay: $nmd active of $n_xmd in the file, $nmd_ign ignored (overridden); waivers: $nwv (the file has $n_xwv create_waiver)"
puts "XDC queries:"
puts [join $qlog "\n"]
puts [expr {[llength $empty] ? "XDC queries that matched nothing: $empty" : "XDC queries: all matched"}]
puts "Sources of paths between the clocks (register or RAM arrays, input ports): [lindex $xs 0] not constrained crossings"
puts [join [lindex $xs 1] "\n"]
puts "Clock interaction:"
puts [report_clock_interaction -delay_type min_max -return_string]
puts "CDC summary:"
puts [report_cdc -return_string]
puts "Methodology summary:"
set m [report_methodology -return_string]
set i [string first "Summary" $m]
puts [string range $m $i [expr {$i + 2500}]]
