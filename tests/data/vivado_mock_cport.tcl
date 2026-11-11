# boards/bcu1525/vivado/wg_cport.tcl on a mock of the two-slice top's opt netlist
# (tests/test_bcuwg_build.py): the floorplan's pblocks (each slice's u_cport and u_dn_rx in its
# pblock_slice<s>, u_up_tx in none: the placer's, or its parent's) and the Vivado commands the move takes.
#   tclsh vivado_mock_cport.tcl CASE WG_CPORT_TCL
# CASE: ok, nopb (no pblock_slice1), nocport (slice 1 without its u_cport). Prints the returned lists'
# sizes and each cell's pblock ("in CELL PBLOCK", "-" for none); exit 1 with the move's error.
lassign $argv case tcl
array set IN {}   ;# cell -> pblock
set CELLS {}
set PBS {pblock_ctl pblock_slice0}
if {$case ne "nopb"} { lappend PBS pblock_slice1 }
foreach s {0 1} {
  set x "u_sys/u_board/g_xs.g_x\[$s\]"
  if {!($case eq "nocport" && $s == 1)} { lappend CELLS $x.u_cport; set IN($x.u_cport) pblock_slice$s }
  lappend CELLS $x.u_dn_rx $x.u_up_tx
  set IN($x.u_dn_rx) pblock_slice$s
  lappend CELLS "u_sys/u_board/g_xs.g_x\[$s\].u_other"
  set IN(u_sys/u_board/g_xs.g_x\[$s\].u_other) pblock_slice$s
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
  set r {}
  if {$of ne ""} {
    foreach c $::CELLS { if {[info exists ::IN($c)] && $::IN($c) eq $of} { lappend r $c } }
  } else {
    foreach c $::CELLS { if {[string match $pat $c]} { lappend r $c } }
  }
  if {$f ne ""} { set r [mock_filter $r $f] }
  return $r
}
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
if {[catch {::otpu::wg_cport pblock_ctl mock} r]} { puts $r; exit 1 }
lassign $r moved from
puts "returned [llength $moved] [llength $from]"
foreach c [lsort $CELLS] { puts "in $c [expr {[info exists IN($c)] ? $IN($c) : "-"}]" }
