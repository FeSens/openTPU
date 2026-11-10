# boards/bcu1525/vivado/wg_x2port.tcl on a mock of the x2 top's opt netlist (tests/test_bcuwg_build.py):
# x2/floorplan_x2.tcl's pblocks (u_link whole and the hub adapters' SLR2 leaves in pblock_x2_link,
# their u_dn_tx / u_up_rx in pblock_ctl) and the Vivado commands the move takes.
#   tclsh vivado_mock_x2port.tcl CASE WG_X2PORT_TCL
# CASE: ok, nopb (no pblock_x2_link), oneport (one link port), nolx (no adapter leaves in
# pblock_x2_link). Prints the returned lists' sizes and each cell's pblock ("in CELL PBLOCK", "-"
# for none); exit 1 with the move's error.
lassign $argv case tcl
array set IN {}   ;# cell -> pblock
set CELLS {u_link}
set PBS {pblock_ctl}
if {$case ne "nopb"} { lappend PBS pblock_x2_link }
set IN(u_link) pblock_x2_link
foreach p {0 1} {
  set l "u_link/g_l\[$p\]"
  foreach c {u_lane u_txf u_rxf u_popped} { lappend CELLS $l.$c }
  if {!($case eq "oneport" && $p == 1)} { lappend CELLS $l.u_port }
  set a "u_sys/u_board/g_xs.g_link.g_l\[$p\].u_lx"
  foreach h {u_dn_tx u_up_rx} { lappend CELLS $a/$h; set IN($a/$h) pblock_ctl }
  if {$case ne "nolx"} {
    foreach r {s_reg[0] s_reg[1] v_reg} { lappend CELLS $a/$r; set IN($a/$r) pblock_x2_link }
  }
}
proc mock_filter {cells f} {
  if {![regexp {^NAME\s*(=~|!~)\s*"?([^"]+)"?$} $f - op pat]} { error "mock: filter '$f'" }
  set r {}
  foreach c $cells {
    if {[string match $pat $c] == ($op eq "=~")} { lappend r $c }
  }
  return $r
}
proc get_pblocks {args} {
  set n [lindex $args end]
  return [expr {$n in $::PBS ? $n : ""}]
}
proc get_cells {args} {
  set of ""; set f ""; set pat ""
  for {set i 0} {$i < [llength $args]} {incr i} {
    set a [lindex $args $i]
    switch -- $a {
      -quiet {}
      -of_objects { set of [lindex $args [incr i]] }
      -filter { set f [lindex $args [incr i]] }
      default { set pat $a }
    }
  }
  if {$of ne ""} {
    set r {}
    foreach c $::CELLS { if {[info exists ::IN($c)] && $::IN($c) eq $of} { lappend r $c } }
  } else {
    # a name pattern without -hierarchical: its own level only (the wildcards do not cross a "/")
    set r {}
    set d [regexp -all {/} $pat]
    foreach c $::CELLS {
      if {[string match $pat $c] && [regexp -all {/} $c] == $d} { lappend r $c }
    }
  }
  if {$f ne ""} { set r [mock_filter $r $f] }
  return $r
}
proc filter {cells f} { mock_filter $cells $f }
proc remove_cells_from_pblock {pb cells} {
  foreach c $cells {
    if {![info exists ::IN($c)] || $::IN($c) ne $pb} { error "mock: $c is not in $pb" }
    unset ::IN($c)
  }
}
proc add_cells_to_pblock {pb cells} {
  if {$pb ni $::PBS} { error "mock: no pblock $pb" }
  foreach c $cells { set ::IN($c) $pb }
}
source $tcl
if {[catch {::otpu::wg_x2port pblock_ctl mock} r]} { puts $r; exit 1 }
lassign $r ports lx rest
puts "returned [llength $ports] [llength $lx] [llength $rest]"
foreach c [lsort $CELLS] { puts "in $c [expr {[info exists IN($c)] ? $IN($c) : "-"}]" }
