# x2port, the BCU1525 x2 floorplan move on an opt_design netlist: the x2 link's core-clock ports
# (u_link/g_l[p].u_port, otpu_link_port) and the hub adapters' leaves in SLR2 (x2/floorplan_x2.tcl's:
# otpu_coll_lx, its u_dn_rx / u_up_tx stages) from pblock_x2_link into SLR1's pblock_ctl, beside the
# hub. The rest of u_link (the lanes on Aurora's 402.8 MHz user_clk, the TX / RX FIFOs, the
# synchronizers) stays in pblock_x2_link beside the GT quads, so the SLR crossing moves to the
# ports' FIFO and credit side, on the core clock.
# Sourced by place_screen.tcl (OTPU_SCREEN_MOVES x2port) and bcu1525_wg_place_pre.tcl (OTPU_WG_X2PORT 1,
# run_vivado.sh WG_X2PORT: B2's place). ::otpu::wg_x2port CTL WHO moves both ports and returns
# {PORTS LX REST}: the two ports, the adapter leaves moved with them and the u_link cells left in
# pblock_x2_link; any surprise in the netlist or the pblocks is an error naming WHO.
namespace eval ::otpu {}
proc ::otpu::x2port_cells {who what pat} {
  set c [get_cells -quiet $pat]
  if {![llength $c]} { error "$who: x2port: no $what ($pat)" }
  return $c
}
proc ::otpu::wg_x2port {ctl who} {
  set x2pb [get_pblocks -quiet pblock_x2_link]
  if {[llength $x2pb] != 1} { error "$who: x2port: no pblock_x2_link" }
  # u_link is in pblock_x2_link whole: put back its children but the ports
  set ports [::otpu::x2port_cells $who "link ports" {u_link/g_l?*?.u_port}]
  if {[llength $ports] != 2} { error "$who: x2port: [llength $ports] link ports (expected 2)" }
  remove_cells_from_pblock $x2pb [::otpu::x2port_cells $who "link endpoint" u_link]
  set rest [filter [get_cells u_link/*] {NAME !~ "u_link/g_l?*?.u_port"}]
  add_cells_to_pblock $x2pb $rest
  set lx [get_cells -quiet -of_objects $x2pb -filter {NAME =~ u_sys/u_board/g_xs.g_link.g_l*.u_lx/*}]
  if {![llength $lx]} { error "$who: x2port: no adapter leaves in pblock_x2_link" }
  remove_cells_from_pblock $x2pb $lx
  add_cells_to_pblock $ctl [concat $ports $lx]
  puts "$who: x2port: [join $ports], [llength $lx] adapter leaves into pblock_ctl; [llength $rest] u_link cells stay in pblock_x2_link"
  return [list $ports $lx $rest]
}
