# A place-only floorplan screen of the BCU1525 WG build (bcuwg_run_t4.sh TCL=FILE): an
# implementation's opt checkpoint (OTPU_SCREEN_DCP), cells moved between its pblocks
# (OTPU_SCREEN_MOVES), place_design (OTPU_WG_PLACE, default Default, as impl_resume.tcl), then the
# placed reports into OUT_DIR/reports/screen_*. OTPU_SCREEN_SAVE 1: the placed checkpoint too,
# OUT_DIR/screen_placed.dcp, right after place_design (b1r_stage.sh puts it in a new run's impl_1 as
# its placed step: bcuwg_run_t4.sh DCP=1 goes on with impl_resume.tcl's phys_opt, route, reports and
# bitstream on this placement).
#   vivado -mode batch -source place_screen.tcl -tclargs OUT_DIR
# BCUWG_TREE (the staged tree) gives bcu1525_wg_place_pre.tcl. Moves, words (all into SLR1's
# pblock_ctl, beside the hub; "none": no move, the floorplan as built):
#   x2port  the x2 link's core-clock ports (u_link/g_l[p].u_port) and the hub adapters' leaves in
#           SLR2 (floorplan_x2.tcl's: otpu_coll_lx, its u_dn_rx / u_up_tx stages): the crossing
#           moves to the ports' FIFO and credit side, on the core clock (wg_x2port.tcl, also B2's
#           OTPU_WG_X2PORT 1 in bcu1525_wg_place_pre.tcl: not both)
#   cport   each slice's collective port (g_xs.g_x[s].u_cport) with its crossing's slice-side
#           stages (u_dn_rx, u_up_tx): the crossing moves to the port's slice side, on the core
#           clock (wg_cport.tcl, also B2's OTPU_WG_CPORT 1 in bcu1525_wg_place_pre.tcl: not both)
#   umem    each slice's adapter (u_mem) and its 8 bridges' core halves into pblock_ctl, u_xs's
#           pipes split on the slice SLR's boundary with SLR1, each pair on its Laguna registers
#           (wg_umem.tcl, also B2's OTPU_WG_UMEM 1 in bcu1525_wg_place_pre.tcl: not both).
#           NAT_XSTG >= 2 (x2's 2: no spare stage, the slice's port registers drive the pair;
#           screen_cross.rpt's paths take the pairs' pins too)
# The lanes stay: they run on the Aurora user clock (402.8 MHz) beside the GT quads.
# OTPU_WG_MIGKO (bcu1525_wg_place_pre.tcl's, wg_migko.tcl): the keep-outs beside the DIMMs' PHYs;
# screen_summary.txt gets each one's cells placed inside it and the other cells on its sites.
# OTPU_SCREEN_CDG 1: CLOCK_DELAY_GROUP otpu_acc on acc_clk's and acc_clk2x's nets before
# place_design (B1's netlist: the wizard's clkout2_buf and the top's u_acc2x_buf, two outputs of one
# MMCM; from B2 the top's BUFGCE_DIV pair, u_acc_buf and u_acc2x_buf: bcu1525_wg_top.tcl's
# OTPU_WG_CDG, UG949). Always: the 1x / 2x crossings' hold endpoints under 0, -0.1,
# -0.182 (the pair's hold uncertainty, gone), -0.3 and -0.5 ns, each way, into screen_summary.txt.
# OTPU_CLK_HOOK (a bare name this directory's, as bcu1525_wg_check.tcl's): sourced on the opened
# checkpoint before the moves (clk133.tcl: the screen at 133.33 MHz, the MMCM's and the selects'
# edits after opt_design here; its reports clk_k.rpt and clk133_clocks.rpt in OUT_DIR/reports).
# Reports: screen_util_slr.rpt, screen_congestion.rpt, screen_timing.rpt (estimated), screen_cross.rpt
# (the worst paths through the moved cells' pins, their data path delays against B2's 6.667 ns),
# screen_slr_cells.txt (impl_resume.tcl's placed_slr_cells.txt), screen_summary.txt (the estimated
# WNS / WHS and each clock's worst setup slack, each SLR's used CLB sites and LUTs, the worst path
# through the moved cells, the worst path of the 1000 worst endpoints whose start and end sit in two
# SLRs, the 1x / 2x crossings' hold).
set out [file normalize [lindex $argv 0]]
proc senv {k d} { expr {[info exists ::env($k)] && $::env($k) ne "" ? $::env($k) : $d} }
set dcp [senv OTPU_SCREEN_DCP ""]
set tree [senv BCUWG_TREE ""]
set moves [senv OTPU_SCREEN_MOVES "x2port cport"]
if {$dcp eq "" || ![file exists $dcp]} { error "place_screen.tcl: OTPU_SCREEN_DCP '$dcp': no checkpoint" }
set pre $tree/boards/bcu1525/vivado/bcu1525_wg_place_pre.tcl
if {![file exists $pre]} { error "place_screen.tcl: BCUWG_TREE '$tree': no $pre" }
if {$moves eq "none"} { set moves {} }
foreach m $moves {
  if {$m ni {x2port cport umem}} { error "place_screen.tcl: OTPU_SCREEN_MOVES: '$m' (x2port, cport, umem or none)" }
}
file mkdir $out/reports
set t0 [clock seconds]
puts "place_screen.tcl: $dcp, moves '$moves', place [senv OTPU_WG_PLACE Default]"
open_checkpoint $dcp
set hook [senv OTPU_CLK_HOOK ""]
if {$hook ne ""} {
  if {[file pathtype $hook] ne "absolute"} { set hook [file join [file dirname [file normalize [info script]]] $hook] }
  if {![file exists $hook]} { error "place_screen.tcl: OTPU_CLK_HOOK $hook: no such file" }
  puts "place_screen.tcl: OTPU_CLK_HOOK $hook"
  namespace eval ::otpu {}
  set ::otpu::clk_rpt_dir $out/reports
  source $hook
}
set ctl [get_pblocks -quiet pblock_ctl]
if {[llength $ctl] != 1} { error "place_screen.tcl: no pblock_ctl (plan c's SLR1 pblock)" }
set moved {}
set sllx {}
if {"x2port" in $moves} {
  if {[senv OTPU_WG_X2PORT 0] eq "1"} { error "place_screen.tcl: OTPU_SCREEN_MOVES x2port with OTPU_WG_X2PORT 1 (bcu1525_wg_place_pre.tcl moves it too)" }
  source [file join [file dirname [file normalize [info script]]] wg_x2port.tcl]
  lassign [::otpu::wg_x2port $ctl place_screen.tcl] ports lx rest
  set moved [concat $moved $ports]
}
if {"cport" in $moves} {
  if {[senv OTPU_WG_CPORT 0] eq "1"} { error "place_screen.tcl: OTPU_SCREEN_MOVES cport with OTPU_WG_CPORT 1 (bcu1525_wg_place_pre.tcl moves it too)" }
  source [file join [file dirname [file normalize [info script]]] wg_cport.tcl]
  lassign [::otpu::wg_cport $ctl place_screen.tcl] c f
  set moved [concat $moved [lindex $c 0] [lindex $c 3]]
}
if {"umem" in $moves} {
  if {[senv OTPU_WG_UMEM 0] eq "1"} { error "place_screen.tcl: OTPU_SCREEN_MOVES umem with OTPU_WG_UMEM 1 (bcu1525_wg_place_pre.tcl moves it too)" }
  source [file join [file dirname [file normalize [info script]]] wg_umem.tcl]
  lassign [::otpu::wg_umem $ctl place_screen.tcl] c sll
  set moved [concat $moved $c]
  set sllx [concat $sllx $sll]
}
foreach c $moved {
  puts "place_screen.tcl: $c in [get_pblocks -quiet -of_objects [get_cells -quiet $c]]"
}
set cdg [senv OTPU_SCREEN_CDG 0]
if {$cdg} {
  # (B1's: the wizard's clkout2_buf; from B2 the pair, the wizard's clkout2_buf core_clk's)
  set cp [expr {[llength [get_cells -quiet u_acc_buf]] ? {u_acc_buf/O u_acc2x_buf/O}
                : {u_bd/otpu_bdw_i/clk_wiz_0/inst/clkout2_buf/O u_acc2x_buf/O}}]
  set cn [get_nets -quiet -of_objects [get_pins -quiet $cp]]
  if {[llength $cn] != 2} { error "place_screen.tcl: OTPU_SCREEN_CDG: acc_clk's and acc_clk2x's nets '$cn'" }
  set_property CLOCK_DELAY_GROUP otpu_acc $cn
  puts "place_screen.tcl: CLOCK_DELAY_GROUP otpu_acc on [join $cn]"
}
source $pre
place_design -directive [senv OTPU_WG_PLACE Default]
puts "place_screen.tcl: placed at [clock format [clock seconds] -gmt 1 -format %H:%M:%SZ] ([expr {[clock seconds] - $t0}] s)"
if {[senv OTPU_SCREEN_SAVE 0]} {
  write_checkpoint -force $out/screen_placed.dcp
  puts "place_screen.tcl: $out/screen_placed.dcp at [clock format [clock seconds] -gmt 1 -format %H:%M:%SZ]"
}
report_utilization -slr -file $out/reports/screen_util_slr.rpt
report_design_analysis -congestion -file $out/reports/screen_congestion.rpt
report_timing_summary -max_paths 30 -file $out/reports/screen_timing.rpt
# the moved cells' pins: the paths across their new crossings (B2's budget 6.667 ns at 150 MHz)
set pins [get_pins -quiet -of_objects [concat $moved $sllx]]
if {[llength $pins]} {
  report_timing -through $pins -max_paths 40 -nworst 1 -sort_by slack -file $out/reports/screen_cross.rpt
}
if {[catch {
  set f [open $out/reports/screen_summary.txt w]
  puts $f "place_screen.tcl: $dcp, moves '$moves', place [senv OTPU_WG_PLACE Default], clock hook '$hook', [expr {[clock seconds] - $t0}] s"
  set ws [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]]
  set wh [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -hold] 0]]
  puts $f "placement WNS $ws ns WHS $wh ns"
  foreach clk [get_clocks] {
    set p [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup -to $clk] 0]
    if {[llength $p]} { puts $f [format "  %-44s setup %8s ns" [get_property NAME $clk] [get_property SLACK $p]] }
  }
  foreach s [get_slrs] {
    set st [get_sites -quiet -of_objects $s -filter {SITE_TYPE =~ SLICE*}]
    set used [llength [filter -quiet $st {IS_USED == 1}]]
    set nl [llength [filter -quiet [get_cells -quiet -of_objects $st] {REF_NAME =~ LUT*}]]
    puts $f [format "%s CLB %d / %d (%.2f%%) LUT cells %d" $s $used [llength $st] \
      [expr {100.0 * $used / max(1, [llength $st])}] $nl]
  }
  if {[llength $pins]} {
    set p [lindex [get_timing_paths -through $pins -max_paths 1 -nworst 1] 0]
    if {[llength $p]} {
      puts $f [format "worst path through the moved cells: slack %s ns, data path %s ns (%s -> %s)" \
        [get_property SLACK $p] [get_property DATAPATH_DELAY $p] [get_property STARTPOINT_PIN $p] \
        [get_property ENDPOINT_PIN $p]]
    }
  }
  # the worst SLR crossing: the first of the 1000 worst endpoints' paths whose start and end cells
  # sit in two SLRs (ports skipped)
  if {[catch {
    set xp ""
    foreach p [get_timing_paths -max_paths 1000 -nworst 1 -setup] {
      set a [get_cells -quiet -of_objects [get_pins -quiet [get_property STARTPOINT_PIN $p]]]
      set b [get_cells -quiet -of_objects [get_pins -quiet [get_property ENDPOINT_PIN $p]]]
      if {![llength $a] || ![llength $b]} continue
      set sa [get_slrs -quiet -of_objects $a]; set sb [get_slrs -quiet -of_objects $b]
      if {[llength $sa] == 1 && [llength $sb] == 1 && $sa ne $sb} { set xp $p; break }
    }
    if {[llength $xp]} {
      puts $f [format "worst SLR crossing: slack %s ns, data path %s ns, %s -> %s (%s -> %s)" \
        [get_property SLACK $xp] [get_property DATAPATH_DELAY $xp] $sa $sb \
        [get_property STARTPOINT_PIN $xp] [get_property ENDPOINT_PIN $xp]]
    } else {
      puts $f "worst SLR crossing: none in the 1000 worst endpoints"
    }
  } e]} { puts $f "worst SLR crossing: $e" }
  # OTPU_WG_MIGKO's keep-outs (bcu1525_wg_place_pre.tcl): each one's cells inside, others inside
  if {[llength [info commands ::otpu::migko_report]]} {
    foreach l [::otpu::migko_report] { puts $f "migko: $l" }
  }
  close $f
} e]} { puts "place_screen.tcl: screen_summary.txt: $e" }
# the 1x / 2x crossings' hold endpoints by slack (one path an endpoint)
if {[catch {
  set ca [get_clocks -quiet -of_objects [get_pins -quiet {u_sys/u_board/cal_s1_reg[0]/C}]]
  set cf [get_clocks -quiet -of_objects [get_pins -quiet u_acc2x_buf/O]]
  if {[llength $ca] != 1 || [llength $cf] != 1} { error "clocks '$ca' / '$cf'" }
  set f [open $out/reports/screen_summary.txt a]
  puts $f "CLOCK_DELAY_GROUP on acc_clk / acc_clk2x: $cdg"
  foreach {a b} [list $ca $cf $cf $ca] {
    set line "hold [get_property NAME $a] -> [get_property NAME $b], endpoints under"
    foreach th {0 -0.1 -0.182 -0.3 -0.5} {
      set n [llength [get_timing_paths -quiet -hold -from $a -to $b -max_paths 200000 -nworst 1 -slack_lesser_than $th]]
      append line " $th: $n"
    }
    puts $f $line
  }
  close $f
} e]} { puts "place_screen.tcl: the crossings' hold: $e" }
catch {puts [exec cat $out/reports/screen_summary.txt]}
# each SLR's LUTs and flip-flops by instance (impl_resume.tcl's placed_slr_cells.txt)
if {[catch {
  set f [open $out/reports/screen_slr_cells.txt w]
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
} e]} { puts "place_screen.tcl: screen_slr_cells.txt: $e" }
puts "place_screen.tcl: done in [expr {[clock seconds] - $t0}] s"
