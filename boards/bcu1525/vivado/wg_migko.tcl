# migko, the BCU1525 WG build's MIG keep-out on an opt_design netlist (OTPU_WG_MIGKO, run_vivado.sh
# WG_MIGKO; default off): per DIMM d an EXCLUDE_PLACEMENT pblock pblock_migko<d>, IS_SOFT, of the
# W SLICE columns (OTPU_WG_MIGKO_W, default 12) on its PHY's fabric side over its banks' clock-region
# rows, holding the CLB cells (LUTs, carries, muxes, LUT RAMs, SRLs, flip-flops) of its MIG's
# u_ddr4_mem_intfc (the PHY's fabric half, the calibration, the controller; u_ddr_axi, the AXI side,
# stays with the placer, as do the MIG's block RAMs and DSPs): the core kept out of the slices whose
# routes reach the bitslices.
#
# B1 (b1pt-4d589467, 2026-11-04, text reports): 7 failing Min Skew checks, each between an
# RXTX_BITSLICE's D[1] and D[2] routes from the fabric, over all four PHYs (worst -0.177 ns, ddr4_0
# byte 3 at BITSLICE_RX_TX_X0Y101; -0.101 ddr4_2 X0Y491, -0.078 ddr4_1 X1Y452, -0.064 ddr4_3
# X1Y582); ddr4_0's routed congestion East level 5 at CLEL_R_X38Y92..CLEL_R_X53Y123 (its
# u_ddr4_mem_intfc 25%, u_ddr_mc_pi 14%, u_slice/u_vpu 10%) and its failing setup endpoints on SLICE
# X55-68 Y94-178. The PHY's fabric side faces right: every MIG cell on B1's timing paths sits at
# SLICE X >= 55 (bitslice column X0: C0 and C2) or X >= 112 (X1: C1 and C3), the core's cells fill
# X54 and X111, and placed routes into the bitslices' D pins were 0.29-0.31 ns from X55-57 against
# 1.24-1.38 ns from X65-74 (2.0-2.5 ns routed). The rows: a DIMM's I/O sites' (IOB_X<c>Y<y>: 52 a
# bank, one bank a clock-region row of 60 SLICE rows; the board's banks 40-42 C0, 65-67 C1, 46-48 C2,
# 70-72 C3: rows 1-3, 6-8, 7-9, 11-13), so C0 SLICE_X55Y60:SLICE_X66Y239, C1 X112Y360:X123Y539, C2
# X55Y420:X66Y599, C3 X112Y660:X123Y839 at W 12. The size: B1's u_ddr4_mem_intfc 11.0-11.5K LUTs
# and 15.0-15.2K flip-flops a DIMM, 2,160 CLB sites at W 12 (66% of their LUTs, 44% of their
# flip-flops); at B1's 5.36 LUTs a used CLB the MIG takes about 2,130 placed anyway, so the four
# strips (8,640 CLBs, 5.8% of the VU9P's) move the core off them rather than taking it room.
# The strips are not nested in the floorplan's SLR pblocks (SLR2 holds pblock_slice1 and x2's
# pblock_x2_link over the same rows): they overlap them, as those two do each other.
# Sourced by bcu1525_wg_place_pre.tcl (place_design's pre-hook, so place_screen.tcl's screens take
# it too). ::otpu::wg_migko SPEC W WHO makes them (SPEC: 1 every DIMM, or DIMM numbers "0 3") and
# returns {DIMM PBLOCK RANGE ...}; any surprise in the netlist is an error naming WHO.
# ::otpu::migko_report after place_design: a line a strip (its cells inside / placed, others inside).
namespace eval ::otpu {}
# the first SLICE column on each bitslice column's fabric side (VU9P, B1's placement above)
set ::otpu::migko_x0 {0 55 1 112}
proc ::otpu::migko_dimms {spec who} {
  if {$spec eq "1"} { return {0 1 2 3} }
  set ds [lsort -unique [regexp -all -inline {\S+} $spec]]
  if {![llength $ds]} { error "$who: migko: OTPU_WG_MIGKO '$spec' (1 or DIMM numbers 0-3)" }
  foreach d $ds {
    if {$d ni {0 1 2 3}} { error "$who: migko: OTPU_WG_MIGKO '$spec' (1 or DIMM numbers 0-3)" }
  }
  return $ds
}
# DIMM d's I/O: {column rows} from its ports' IOB sites (each port's LOC, else its package pin's
# site), a site's CLOCK_REGION where Vivado gives one checked against its row
proc ::otpu::migko_io {d who} {
  set ports [get_ports -quiet "ddr4_c${d}_*"]
  if {![llength $ports]} { error "$who: migko: no ddr4_c${d}_* ports" }
  set cols {}
  set rows {}
  foreach p $ports {
    set s [get_property -quiet LOC $p]
    if {![regexp {^IOB_X(\d+)Y(\d+)$} $s - x y]} {
      set s [get_sites -quiet -of_objects [get_package_pins -quiet -of_objects $p]]
      if {![regexp {^IOB_X(\d+)Y(\d+)$} $s - x y]} { error "$who: migko: DIMM $d port $p: no IOB site ('$s')" }
    }
    set r [expr {$y / 52}]
    set cr [get_property -quiet CLOCK_REGION [get_sites -quiet $s]]
    if {[regexp {^X\d+Y(\d+)$} $cr - ry] && $ry != $r} {
      error "$who: migko: DIMM $d port $p at $s: clock region $cr, not row $r (52 I/O sites a row)"
    }
    lappend cols $x
    lappend rows $r
  }
  set cols [lsort -unique $cols]
  set rows [lsort -integer -unique $rows]
  if {[llength $cols] != 1} { error "$who: migko: DIMM $d's ports on IOB columns '$cols' (one expected)" }
  if {[lindex $rows 0] / 5 != [lindex $rows end] / 5} {
    error "$who: migko: DIMM $d's ports on clock-region rows '$rows' (two SLRs)"
  }
  return [list [lindex $cols 0] $rows]
}
proc ::otpu::wg_migko {spec w who} {
  if {![string is integer -strict $w] || $w < 1 || $w > 28} {
    error "$who: migko: OTPU_WG_MIGKO_W '$w' (SLICE columns, 1-28)"
  }
  set made {}
  foreach d [::otpu::migko_dimms $spec $who] {
    set mi u_bd/otpu_bdw_i/ddr4_$d/inst/u_ddr4_mem_intfc
    if {[llength [get_pblocks -quiet pblock_migko$d]]} { error "$who: migko: pblock_migko$d exists" }
    lassign [::otpu::migko_io $d $who] c rows
    if {![dict exists $::otpu::migko_x0 $c]} { error "$who: migko: DIMM $d on IOB column $c (0 or 1)" }
    set x0 [dict get $::otpu::migko_x0 $c]
    set rg SLICE_X${x0}Y[expr {60 * [lindex $rows 0]}]:SLICE_X[expr {$x0 + $w - 1}]Y[expr {60 * [lindex $rows end] + 59}]
    set cells [get_cells -quiet -hierarchical -filter \
      "IS_PRIMITIVE && NAME =~ $mi/* && (PRIMITIVE_GROUP == CLB || PRIMITIVE_GROUP == REGISTER)"]
    set nl 0
    set nf 0
    foreach r [get_property -quiet REF_NAME $cells] {
      if {[string match LUT* $r]} { incr nl } elseif {[string match FD* $r]} { incr nf }
    }
    if {!$nl || !$nf} { error "$who: migko: $mi: [llength $cells] CLB cells, $nl LUTs, $nf flip-flops" }
    set held [get_pblocks -quiet -of_objects $cells]
    if {[llength $held]} { error "$who: migko: $mi's cells already in '[join $held]'" }
    set pb [create_pblock pblock_migko$d]
    resize_pblock $pb -add $rg
    set ns [llength [get_sites -quiet -of_objects $pb]]
    if {!$ns} { error "$who: migko: pblock_migko$d $rg: no sites" }
    set_property IS_SOFT TRUE $pb
    set_property EXCLUDE_PLACEMENT TRUE $pb
    add_cells_to_pblock $pb $cells
    puts [format "%s: migko: pblock_migko%d %s (%d sites, IOB column X%d rows %d-%d, SLR%d): %s's %d CLB cells, %d LUTs (%.0f%% of the sites') and %d flip-flops (%.0f%%)" \
      $who $d $rg $ns $c [lindex $rows 0] [lindex $rows end] [expr {[lindex $rows 0] / 5}] $mi \
      [llength $cells] $nl [expr {100.0 * $nl / (8 * $ns)}] $nf [expr {100.0 * $nf / (16 * $ns)}]]
    lappend made $d $pb $rg
  }
  return $made
}
# after place_design: a line a pblock_migko<d>, its cells placed inside its sites / placed, and the
# other cells on them (EXCLUDE_PLACEMENT soft: the placer may break either)
proc ::otpu::migko_report {} {
  set lines {}
  foreach pb [lsort [get_pblocks -quiet pblock_migko*]] {
    set st [get_sites -quiet -of_objects $pb]
    set in [dict create]
    foreach s $st { dict set in $s 1 }
    set mine [get_cells -quiet -of_objects $pb]
    set np 0
    set ni 0
    foreach l [get_property -quiet LOC $mine] {
      if {$l eq ""} continue
      incr np
      if {[dict exists $in $l]} { incr ni }
    }
    set m [dict create]
    foreach c $mine { dict set m $c 1 }
    set no 0
    foreach c [get_cells -quiet -of_objects $st] { if {![dict exists $m $c]} { incr no } }
    lappend lines "$pb ([get_property -quiet GRID_RANGES $pb], [llength $st] sites): $ni of $np placed cells inside, $no other cells inside"
  }
  return $lines
}
