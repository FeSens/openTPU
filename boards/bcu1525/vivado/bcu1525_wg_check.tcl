# opt_design's pre-hook in the BCU1525 WG build (create_project_wg.tcl): the late constraint files
# ran at link_design in this same Vivado, and a failed command or a missed cell there is only a
# CRITICAL WARNING, so the run stops here on what they recorded (as the WG targets' otpu_impl.tcl): each
# file's misses (::otpu::fail) and what each finished, against the design's cells. otpu_hbm_ch.tcl
# a bridge's core half (otpu_hbm_cs) each, otpu_mxu_stage.tcl an MXU lane's staging each
# (::otpu_mxu_stage_ok), bcu1525_wg_top.tcl and bcu1525_wg_floorplan.tcl once, otpu_clk2x.tcl every
# pumped MXU column (otpu_mxpc, MXU_PUMP).
namespace eval ::otpu {}
set wg_fail [expr {[info exists ::otpu::fail] ? $::otpu::fail : {}}]
proc wg_count {ref} {
  return [llength [get_cells -quiet -hierarchical -filter "ORIG_REF_NAME == $ref || REF_NAME == $ref"]]
}
foreach {f n} [list otpu_hbm_ch [wg_count otpu_hbm_cs] bcu1525_wg_top 1 bcu1525_wg_floorplan 1] {
  set k [expr {[info exists ::otpu::ok($f)] ? $::otpu::ok($f) : 0}]
  if {$k != $n} { lappend wg_fail "$f.tcl finished $k of the build's $n" }
  puts "bcu1525_wg_check.tcl: $f.tcl finished $k of $n"
}
set n [wg_count otpu_mxu_stage]
if {[wg_count otpu_link]} {
  foreach f {x2_floorplan x2_cdc} {
    set k [expr {[info exists ::otpu::ok($f)] ? $::otpu::ok($f) : 0}]
    if {$k != 1} { lappend wg_fail "$f finished $k of 1" }
    puts "bcu1525_wg_check.tcl: $f finished $k of 1"
  }
}
set k [expr {[info exists ::otpu_mxu_stage_ok] ? $::otpu_mxu_stage_ok : 0}]
if {$k != $n} { lappend wg_fail "otpu_mxu_stage.tcl constrained $k of the build's $n" }
puts "bcu1525_wg_check.tcl: otpu_mxu_stage.tcl constrained $k of $n"
# the pumped MXU's columns (MXU_PUMP): otpu_clk2x.tcl's weights and chain ends at a core period
set n [wg_count otpu_mxpc]
if {$n} {
  set k [expr {[info exists ::otpu::ok(otpu_clk2x_mxu)] ? $::otpu::ok(otpu_clk2x_mxu) : 0}]
  if {$k != $n} { lappend wg_fail "otpu_clk2x.tcl constrained $k of the build's $n pumped MXU columns" }
  puts "bcu1525_wg_check.tcl: otpu_clk2x.tcl constrained $k of $n pumped MXU columns"
}
# every port on its pin (build.tcl checks it after synthesis too; an implementation of a synthesis
# whose reports did not finish has only this one)
set nopin [get_ports -quiet -filter {PACKAGE_PIN == ""}]
if {[llength $nopin]} { lappend wg_fail "[llength $nopin] ports without a pin: [lrange $nopin 0 7]" }
puts "bcu1525_wg_check.tcl: [llength [get_ports]] ports, [llength $nopin] without a pin"
if {[llength $wg_fail]} {
  error "bcu1525_wg_check.tcl: [llength $wg_fail] constraint checks failed: [join [lrange $wg_fail 0 7] {; }]"
}
report_exceptions -file [file normalize ../../reports/wg_exceptions.rpt]   ;# (the run directory: OUT/otpu.runs/impl_1)
puts "bcu1525_wg_check.tcl: every constraint file finished"
# OTPU_CLK_HOOK (run_vivado.sh CLK_HOOK): a Tcl file on the linked, constrained netlist before
# opt_design, a bare name this directory's: clk133.tcl implements the 150 MHz synthesis at 133.33
# (it stops the run on anything that does not read back). Before WG_STOP=link: a link-only run
# tries it on the real netlist.
if {[info exists ::env(OTPU_CLK_HOOK)] && $::env(OTPU_CLK_HOOK) ne ""} {
  set wg_hook $::env(OTPU_CLK_HOOK)
  if {[file pathtype $wg_hook] ne "absolute"} {
    set wg_hook [file join [file dirname [file normalize [info script]]] $wg_hook]
  }
  if {![file exists $wg_hook]} { error "bcu1525_wg_check.tcl: OTPU_CLK_HOOK $wg_hook: no such file" }
  puts "bcu1525_wg_check.tcl: OTPU_CLK_HOOK $wg_hook"
  set ::otpu::clk_rpt_dir [file normalize ../../reports]
  source $wg_hook
}
# OTPU_WG_STOP=link (run_vivado.sh WG_STOP=link): the run stops here, its netlist linked and
# constrained, before opt_design: a pre-check of the implementation's memory (the two-slice build's)
if {[info exists ::env(OTPU_WG_STOP)] && $::env(OTPU_WG_STOP) eq "link"} {
  error "bcu1525_wg_check.tcl: stopped at the implementation's link (OTPU_WG_STOP=link), before opt_design"
}
