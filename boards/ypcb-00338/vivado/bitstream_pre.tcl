# Sourced by build.tcl just before write_bitstream, with the routed design open.
#
# REQP-83 (IDELAYE2 with DELAY_SRC IDATAIN and nothing on IDATAIN): each DDR3 channel's
# ddr3_reset_n sits in a spare pin of a data byte group (CH0 AD31, CH1 F18). MIG counts that pin
# in the byte lane's PHY bit mask, so the lane gets an input path (IDELAY + ISERDES) for it; the
# pin is an output driven from fabric, so that IDELAY has no pad input, and the lane's DQ map
# never reads its ISERDES. The check is harmless for exactly these two cells; stop if any other
# cell trips it.
set bad {}
foreach c [get_cells -hier -filter {REF_NAME == IDELAYE2 && DELAY_SRC == IDATAIN}] {
  if {[llength [get_nets -quiet -of [get_pins $c/IDATAIN]]] == 0 ||
      [get_property TYPE [get_nets -quiet -of [get_pins $c/IDATAIN]]] eq "GROUND"} {
    if {![string match "*/mig_?/*ddr_phy_4lanes_0.u_ddr_phy_4lanes/ddr_byte_lane_D.ddr_byte_lane_D/ddr_byte_group_io/input_?1?.iserdes_dq_.idelay_dq.idelaye2" $c]} {
      lappend bad $c
    }
  }
}
if {[llength $bad]} { error "unexpected IDELAYE2 without IDATAIN: $bad" }
set_property SEVERITY {Warning} [get_drc_checks REQP-83]
