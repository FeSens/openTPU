# cport, the BCU1525 WG floorplan move on an opt_design netlist at two slices: each slice's
# collective port (u_sys/u_board/g_xs.g_x[s].u_cport) with its crossing's slice-side stages (u_dn_rx,
# u_up_tx) from pblock_slice<s> into SLR1's pblock_ctl, beside the hub: the crossing moves to the
# port's slice side, on the core clock (6.667 ns at 150 MHz). B1's placement: 1.58K LUTs a slice SLR.
# Sourced by place_screen.tcl (OTPU_SCREEN_MOVES cport) and bcu1525_wg_place_pre.tcl (OTPU_WG_CPORT 1,
# run_vivado.sh WG_CPORT: B2's place). ::otpu::wg_cport CTL WHO moves both slices' and returns
# {MOVED FROM}: the moved cells, and the ones of them that were in their slice's pblock; any surprise
# in the netlist or the pblocks is an error naming WHO.
namespace eval ::otpu {}
proc ::otpu::cport_cells {who what pat} {
  set c [get_cells -quiet $pat]
  if {[llength $c] != 1} { error "$who: cport: [llength $c] $what ($pat, expected 1)" }
  return $c
}
proc ::otpu::wg_cport {ctl who} {
  set moved {}
  set from {}
  foreach s {0 1} {
    set pb [get_pblocks -quiet pblock_slice$s]
    if {[llength $pb] != 1} { error "$who: cport: no pblock_slice$s" }
    set x "u_sys/u_board/g_xs.g_x?$s?"
    set c {}
    foreach {h what} {u_cport "collective port" u_dn_rx "dn_rx stage" u_up_tx "up_tx stage"} {
      set c [concat $c [::otpu::cport_cells $who "slice $s $what" $x.$h]]
    }
    # (the floorplan's pblock_slice<s> holds the ones it took: put back only those)
    set in [get_cells -quiet -of_objects $pb -filter {NAME =~ u_sys/u_board/g_xs.g_x*}]
    set f {}
    foreach k $c { if {$k in $in} { lappend f $k } }
    if {[llength $f]} { remove_cells_from_pblock $pb $f }
    add_cells_to_pblock $ctl $c
    set moved [concat $moved $c]
    set from [concat $from $f]
    puts "$who: cport: slice $s: [join $c] into pblock_ctl ([llength $f] from pblock_slice$s)"
  }
  return [list $moved $from]
}
