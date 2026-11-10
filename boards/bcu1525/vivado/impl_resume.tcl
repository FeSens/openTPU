# The BCU1525 WG build's implementation from its last finished step, without the project: non-project
# mode on the project's impl_1 checkpoints (opt, then this script's placed, physopt and routed), then
# post_impl.tcl's reports and bitstream.
#   vivado -mode batch -source impl_resume.tcl -tclargs OUT_DIR
# The project's impl_1 must have finished opt_design (its link, the late constraints and
# bcu1525_wg_check.tcl: the checkpoint holds them); a step this script finished leaves
# .resume_<step>.done beside its checkpoint, so a run killed mid-step (the watchdog's memory floor)
# starts that step again. Directives: OTPU_WG_PLACE, OTPU_WG_PHYS, OTPU_WG_ROUTE (default Default);
# OTPU_WG_POSTPHYS (a directive, AggressiveExplore ...) adds a phys_opt_design after the route.
# OTPU_WG_PREROUTE: a Tcl file (this directory's, or a path) sourced before route_design
# (route_holdfix.tcl: the router keeps fixing hold past Route 35-514's bailout). OTPU_WG_PREPHYS:
# one sourced on the placed design before phys_opt_design (phys_screp.tcl: the SmartConnects'
# high fan-out resets replicated by placement).
set out [file normalize [lindex $argv 0]]
set here [file normalize [file dirname [info script]]]
set rd $out/otpu.runs/impl_1
set top otpu_fpga_top_bcu_wg
file mkdir $out/reports
proc dir {k} { expr {[info exists ::env($k)] && $::env($k) ne "" ? $::env($k) : "Default"} }
set postphys [expr {[info exists ::env(OTPU_WG_POSTPHYS)] ? $::env(OTPU_WG_POSTPHYS) : ""}]
set steps [concat {opt placed physopt routed} [expr {$postphys ne "" ? "postphys" : ""}]]
set last ""
if {[file exists $rd/.opt_design.end.rst] && [file exists $rd/${top}_opt.dcp]} { set last opt }
# (a marker older than opt_design's end is an earlier implementation's: impl_1 ran again since)
foreach s [lrange $steps 1 end] {
  set m $rd/.resume_$s.done
  if {$last ne "" && [file exists $rd/${top}_$s.dcp] && [file exists $m]
      && [file mtime $m] >= [file mtime $rd/.opt_design.end.rst]} { set last $s }
}
if {$last eq ""} { error "impl_resume.tcl: $rd has no finished opt_design (run the project's impl_1 first)" }
set i [lsearch $steps $last]
puts "impl_resume.tcl: from ${top}_$last.dcp"
open_checkpoint $rd/${top}_$last.dcp
# The DDR4 banks' INTERNAL_VREF: the MIGs train each data byte's own (HPIO_VREF). No constraint sets
# one, yet the four address banks read 0.600 (SSTL12's) and conflict with their data byte's POD12
# (BIIVRC-1, dev100c): reset on every POD12 bank, here and again before the reports.
proc wg_vref_reset {at} {
  set was {}
  foreach k [lsort -unique [get_iobanks -of_objects [get_ports -filter {IOSTANDARD =~ POD12*}]]] {
    set v [get_property INTERNAL_VREF $k]
    if {$v ne ""} { lappend was $k:$v; reset_property INTERNAL_VREF $k }
  }
  puts "impl_resume.tcl: INTERNAL_VREF at $at: [expr {[llength $was] ? "reset on $was" : "none"}]"
}
wg_vref_reset $last
proc done {rd top s} {
  write_checkpoint -force $rd/${top}_$s.dcp
  close [open $rd/.resume_$s.done w]
  puts "impl_resume.tcl: $s at [clock format [clock seconds] -gmt 1 -format %H:%M:%SZ]"
}
if {$i < 1} {
  source $here/bcu1525_wg_place_pre.tcl
  place_design -directive [dir OTPU_WG_PLACE]
  done $rd $top placed
}
# OTPU_WG_STOP=placed: a place trial, the placed design's reports (reports/placed_*.rpt: utilization
# by SLR and by hierarchy, the placer's congestion, the estimated timing), no phys_opt, route or
# bitstream; impl_resume.tcl again without it goes on from the placed checkpoint
if {[info exists ::env(OTPU_WG_STOP)] && $::env(OTPU_WG_STOP) eq "placed"} {
  if {$i >= 2} { error "impl_resume.tcl: OTPU_WG_STOP placed, but $rd is past place ($last)" }
  report_utilization -slr -file $out/reports/placed_util_slr.rpt
  report_utilization -hierarchical -hierarchical_depth 5 -file $out/reports/placed_util_hier.rpt
  report_design_analysis -congestion -file $out/reports/placed_congestion.rpt
  report_timing_summary -max_paths 30 -file $out/reports/placed_timing.rpt
  # each SLR's LUTs and flip-flops by instance (a cell's first four levels of hierarchy): what shares
  # the slice SLRs with the slices (reports/placed_slr_cells.txt: SLR, LUTs, FFs, instance)
  if {[catch {
    set f [open $out/reports/placed_slr_cells.txt w]
    foreach s [get_slrs] {
      set cs [get_cells -quiet -of_objects [get_sites -quiet -of_objects $s -filter {SITE_TYPE =~ SLICE*}]]
      foreach {d pat} {nl {REF_NAME =~ LUT*} nf {REF_NAME =~ FD*}} {
        set $d [dict create]
        foreach nm [get_property NAME [filter -quiet $cs $pat]] {
          set q [lrange [split $nm /] 0 end-1]
          dict incr $d [expr {[llength $q] ? [join [lrange $q 0 3] /] : "(top)"}]
        }
      }
      foreach p [lsort -unique [concat [dict keys $nl] [dict keys $nf]]] {
        puts $f "$s [expr {[dict exists $nl $p] ? [dict get $nl $p] : 0}]\
                 [expr {[dict exists $nf $p] ? [dict get $nf $p] : 0}] $p"
      }
    }
    close $f
  } e]} { puts "impl_resume.tcl: placed_slr_cells.txt: $e" }
  # the staging checks' rows 6, 7, 10 and 11 (docs/bcu1525_wg.md) on the placed
  # design, so a place trial needs no Vivado of its own for them: the logic levels by group, the
  # clocks and their interactions, CDC, the methodology checks (DSP / RAM pipelining, TIMING-*) and
  # the exceptions Vivado ignored; a report that fails is logged, the trial goes on
  foreach {rpt cmd} {
    placed_levels.rpt {report_design_analysis -logic_level_distribution -logic_level_dist_paths 5000}
    placed_clocks.rpt {report_clocks}
    placed_clock_interaction.rpt {report_clock_interaction}
    placed_cdc.rpt {report_cdc -details}
    placed_methodology.rpt {report_methodology}
    placed_exceptions_ignored.rpt {report_exceptions -ignored}
  } {
    if {[catch {{*}$cmd -file $out/reports/$rpt} e]} { puts "impl_resume.tcl: $rpt: $e" }
  }
  puts "impl_resume.tcl: OTPU_WG_STOP placed: stopped after place_design"
  return
}
proc hook {k} {
  if {[info exists ::env($k)] && $::env($k) ne ""} {
    set f [file join $::here $::env($k)]
    if {![file exists $f]} { error "impl_resume.tcl: $k: no $f" }
    puts "impl_resume.tcl: $k $f"
    uplevel #0 [list source $f]
  }
}
if {$i < 2} {
  hook OTPU_WG_PREPHYS
  phys_opt_design -directive [dir OTPU_WG_PHYS]
  done $rd $top physopt
}
if {$i < 3} {
  hook OTPU_WG_PREROUTE
  route_design -directive [dir OTPU_WG_ROUTE]
  done $rd $top routed
}
if {$postphys ne "" && $i < 4} {
  phys_opt_design -directive $postphys
  done $rd $top postphys
}
wg_vref_reset [lindex $steps end]
source $here/post_impl.tcl
