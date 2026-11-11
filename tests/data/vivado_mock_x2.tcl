# Execute the real plan-c floorplan first, then add the link hierarchy. This checks the actual
# adapter split selectors and GT LOCs, including a missing-stage failure. No Vivado emulation
# of placement or timing is claimed: the list of leaves below is deliberately tiny.
lassign $argv wgout here out x2mode
set argv [list 2 $wgout fp $here $out]
source [file dirname [info script]]/vivado_mock_wg.tcl
hc u_link; lc u_link/f FDRE
hc u_link_phy; lc u_link_phy/f FDRE
foreach p {0 1} {
  set a "u_sys/u_board/g_xs.g_link.g_l\[$p\].u_lx"
  hc $a
  foreach h {u_dn_tx u_up_rx u_dn_rx u_up_tx} { hc $a/$h; lc $a/$h/r FDRE }
  lc $a/tq FDRE
  lc "u_link_phy/g_q\[$p\].u_aur/u_common/u_common" GTYE4_COMMON
}
if {$x2mode eq "missing"} { unset C(u_sys/u_board/g_xs.g_link.g_l\[1\].u_lx/u_up_rx) }
rename fx wg_fx
proc fx {f} {
  regsub -all {NAME !~ ([^ ()|&]+)} $f {![string match {\1} $name]} f
  return [wg_fx $f]
}
rename set_property wg_set_property
proc set_property {p v o} {
  if {$p eq "LOC" && [string match GTYE4_COMMON_* $v]} { foreach c $o { set ::LOC($c) $v }; return }
  wg_set_property $p $v $o
}
rename get_pblocks wg_get_pblocks
proc get_pblocks {args} {
  set p [lindex $args end]
  if {[string index $p 0] eq "-" || $p eq ""} { return [wg_get_pblocks] }
  return [lsearch -all -inline -glob [wg_get_pblocks] $p]
}
proc get_clock_regions {args} { return {X0Y10 X5Y14} }
set rc [catch {source $here/../x2/floorplan_x2.tcl} e]
if {$rc} { puts "FAIL $e"; exit 0 }
foreach p {0 1} {
  set a "u_sys/u_board/g_xs.g_link.g_l\[$p\].u_lx"
  foreach h {u_dn_tx u_up_rx} {
    if {"$a/$h" ni $PBC(pblock_ctl) || "$a/$h/r" in $PBC(pblock_x2_link)} { error "hub stage misplaced" }
  }
  foreach h {u_dn_rx u_up_tx} { if {"$a/$h/r" ni $PBC(pblock_x2_link)} { error "adapter stage misplaced" } }
  if {"$a/tq" ni $PBC(pblock_x2_link)} { error "adapter FIFO misplaced" }
}
if {$PBR(pblock_x2_link) ne "CLOCKREGION_X0Y10 CLOCKREGION_X5Y14"} { error "clock region resource prefix missing" }
if {$LOC(u_link_phy/g_q\[0\].u_aur/u_common/u_common) ne "GTYE4_COMMON_X1Y12" ||
    $LOC(u_link_phy/g_q\[1\].u_aur/u_common/u_common) ne "GTYE4_COMMON_X1Y11"} { error "wrong GT common sites" }
puts "PASS x2 floorplan"
