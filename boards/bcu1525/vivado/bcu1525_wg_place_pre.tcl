# place_design's pre-hook in the BCU1525 WG build (create_project_wg.tcl), on opt_design's netlist.
# clk300_1's pad (IOB_X1Y281, SLR1) feeds, through one IBUFDS (bd_wg.tcl's refbuf_c1), both DIMM
# C1's MIG (No_Buffer: its MMCM locked by the IP at MMCM_X1Y7) and the clock wizard's MMCM; opt_design
# puts a BUFG on it (Opt 31-194). A BUFG driven from a pad driving two MMCMs must sit in one MMCM's
# clock region with the other adjacent (rule_bufg_mmcm_2loads), and in the pad's region
# (rule_gclkio_bufg): the pad two regions under the MIG's MMCM fails both (Place 30-675, 30-716,
# "IO Clock Placer failed", 2026-10-28). Each BUFG on that net drives its MMCMs over the clock
# backbone (CLOCK_DEDICATED_ROUTE BACKBONE on its output, the placer's own suggestion): the BUFG in
# the pad's region, the MMCMs where they are. (Pins by -of_objects: the cells' names hold brackets,
# GEN_IBUFDS[0], IBUF_OUT[0]_BUFG_inst, a pattern's wildcard.)
set ib [get_cells -quiet -hierarchical -filter {REF_NAME == IBUFCTRL && NAME =~ u_bd/otpu_bdw_i/refbuf_c1/*}]
if {[llength $ib] != 1} { error "bcu1525_wg_place_pre.tcl: clk300_1's IBUFCTRL: '$ib' (expected one)" }
proc wg_out_net {cell} { get_nets -of_objects [get_pins -of_objects $cell -filter {REF_PIN_NAME == O}] }
set n [wg_out_net $ib]
if {[llength $n] != 1} { error "bcu1525_wg_place_pre.tcl: $ib's output net: '$n'" }
set bufs [get_cells -quiet -of_objects [get_pins -quiet -leaf -of_objects $n -filter {DIRECTION == IN}] \
  -filter {REF_NAME =~ BUFG*}]
if {[llength $bufs] == 0} { error "bcu1525_wg_place_pre.tcl: no BUFG on clk300_1's net $n" }
foreach b $bufs {
  set o [wg_out_net $b]
  set mmcm [get_cells -quiet -of_objects [get_pins -quiet -leaf -of_objects $o -filter {DIRECTION == IN}] \
    -filter {REF_NAME =~ MMCM*}]
  set_property CLOCK_DEDICATED_ROUTE BACKBONE $o
  puts "bcu1525_wg_place_pre.tcl: $b -> [llength $mmcm] MMCMs ($mmcm): CLOCK_DEDICATED_ROUTE BACKBONE on $o"
}
# OTPU_WG_UMEM 1 (run_vivado.sh WG_UMEM; two slices, plan c's pblock_ctl): wg_umem.tcl's move, each
# slice's adapter and its bridges' core halves into SLR1 and u_xs's pipes split on the SLR
# boundary's Laguna registers, as place_screen.tcl's umem screens placed it. Any miss stops the run.
if {[info exists ::env(OTPU_WG_UMEM)] && $::env(OTPU_WG_UMEM) eq "1"} {
  set wg_ctl [get_pblocks -quiet pblock_ctl]
  if {[llength $wg_ctl] != 1} { error "bcu1525_wg_place_pre.tcl: OTPU_WG_UMEM: no pblock_ctl (plan c's SLR1 pblock)" }
  source [file join [file dirname [file normalize [info script]]] wg_umem.tcl]
  lassign [::otpu::wg_umem $wg_ctl bcu1525_wg_place_pre.tcl] wg_umc wg_ums
  puts "bcu1525_wg_place_pre.tcl: OTPU_WG_UMEM: [llength $wg_umc] cells into pblock_ctl, [llength $wg_ums] Laguna registers"
  unset wg_ctl wg_umc wg_ums
}
# OTPU_WG_X2PORT 1 (run_vivado.sh WG_X2PORT; x2's floorplan, plan c's pblock_ctl): wg_x2port.tcl's
# move, the x2 link's core-clock ports and the hub adapters' SLR2 leaves into SLR1, as
# place_screen.tcl's x2port screens placed them. Any miss stops the run.
if {[info exists ::env(OTPU_WG_X2PORT)] && $::env(OTPU_WG_X2PORT) eq "1"} {
  set wg_ctl [get_pblocks -quiet pblock_ctl]
  if {[llength $wg_ctl] != 1} { error "bcu1525_wg_place_pre.tcl: OTPU_WG_X2PORT: no pblock_ctl (plan c's SLR1 pblock)" }
  source [file join [file dirname [file normalize [info script]]] wg_x2port.tcl]
  lassign [::otpu::wg_x2port $wg_ctl bcu1525_wg_place_pre.tcl] wg_x2p wg_x2l wg_x2r
  unset wg_ctl wg_x2p wg_x2l wg_x2r
}
# OTPU_WG_CPORT 1 (run_vivado.sh WG_CPORT; two slices, plan c's pblock_ctl): wg_cport.tcl's move,
# each slice's collective port and its crossing's slice-side stages into SLR1, as place_screen.tcl's
# cport screens placed them. Any miss stops the run.
if {[info exists ::env(OTPU_WG_CPORT)] && $::env(OTPU_WG_CPORT) eq "1"} {
  set wg_ctl [get_pblocks -quiet pblock_ctl]
  if {[llength $wg_ctl] != 1} { error "bcu1525_wg_place_pre.tcl: OTPU_WG_CPORT: no pblock_ctl (plan c's SLR1 pblock)" }
  source [file join [file dirname [file normalize [info script]]] wg_cport.tcl]
  lassign [::otpu::wg_cport $wg_ctl bcu1525_wg_place_pre.tcl] wg_cpm wg_cpf
  unset wg_ctl wg_cpm wg_cpf
}
# OTPU_WG_MIGKO (run_vivado.sh WG_MIGKO; unset, empty or 0: off): wg_migko.tcl's keep-out, an
# EXCLUDE_PLACEMENT soft pblock beside each DIMM's PHY (1: all four; DIMM numbers: those),
# OTPU_WG_MIGKO_W SLICE columns wide (default 12). Any miss stops the run.
if {[info exists ::env(OTPU_WG_MIGKO)] && $::env(OTPU_WG_MIGKO) ni {"" 0}} {
  source [file join [file dirname [file normalize [info script]]] wg_migko.tcl]
  set wg_mkw [expr {[info exists ::env(OTPU_WG_MIGKO_W)] && $::env(OTPU_WG_MIGKO_W) ne "" ? $::env(OTPU_WG_MIGKO_W) : 12}]
  set wg_mk [::otpu::wg_migko $::env(OTPU_WG_MIGKO) $wg_mkw bcu1525_wg_place_pre.tcl]
  puts "bcu1525_wg_place_pre.tcl: OTPU_WG_MIGKO '$::env(OTPU_WG_MIGKO)' W $wg_mkw: [expr {[llength $wg_mk] / 3}] keep-outs"
  unset wg_mkw wg_mk
}
