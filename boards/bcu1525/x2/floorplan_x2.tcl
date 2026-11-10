# Link port / PHY and the hub adapters in SLR2, beside quads 230/231. The hub stays in SLR1
# (plan c); its otpu_coll_lx XRS=2 stages straddle that crossing. Select adapter LEAVES so that
# placing the hub-side stages elsewhere does not also constrain them through an ancestor.
# CLOCKREGION_ is required by resize_pblock (plain X0Y10 is not a valid resource range).
namespace eval ::otpu {}
proc ::otpu::x2_cells {what pat} {
  set c [get_cells -quiet $pat]
  if {![llength $c]} { error "x2: no $what ($pat)" }
  return $c
}
set x2pb [create_pblock pblock_x2_link]
set cr [get_clock_regions -quiet -of_objects [get_slrs SLR2]]
if {![llength $cr]} { error "x2: no SLR2 clock regions" }
foreach r $cr { resize_pblock $x2pb -add CLOCKREGION_$r }
set_property IS_SOFT FALSE $x2pb
add_cells_to_pblock $x2pb [::otpu::x2_cells endpoint {u_link}]
add_cells_to_pblock $x2pb [::otpu::x2_cells PHY {u_link_phy}]
set ctl [get_pblocks -quiet pblock_ctl]
if {[llength $ctl] != 1} { error "x2: plan c needs pblock_ctl in SLR1" }
foreach p {0 1} {
  set a "u_sys/u_board/g_xs.g_link.g_l?$p?.u_lx"
  foreach h {u_dn_tx u_up_rx} {
    add_cells_to_pblock $ctl [::otpu::x2_cells "port $p hub stage $h" $a/$h]
  }
  set cells [get_cells -quiet -hierarchical -filter \
    "NAME =~ $a/* && IS_PRIMITIVE == 1 && NAME !~ $a/u_dn_tx/* && NAME !~ $a/u_up_rx/*"]
  if {![llength $cells]} { error "x2: port $p has no adapter leaves" }
  add_cells_to_pblock $x2pb $cells
  puts "x2 floorplan: port $p adapter [llength $cells] leaves in SLR2; u_dn_tx/u_up_rx in SLR1"
}
foreach {p site} {0 GTYE4_COMMON_X1Y12 1 GTYE4_COMMON_X1Y11} {
  set c [::otpu::x2_cells "quad $p common" "u_link_phy/g_q?$p?.u_aur/u_common/u_common"]
  if {[llength $c] != 1} { error "x2: port $p has [llength $c] GT commons" }
  set_property LOC $site $c
  puts "x2 floorplan: $site $c"
}
incr ::otpu::ok(x2_floorplan)
