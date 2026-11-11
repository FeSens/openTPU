# impl_resume.tcl's OTPU_WG_PREPHYS=phys_screp.tcl: the SmartConnects' synchronous resets of high
# fan-out (their nodes' m_sc_areset_r / s_sc_areset_r registers) replicated by placement before
# phys_opt_design (phys_opt_design -force_replication_on_nets: a copy beside each cluster of loads).
# They are the IP's own registers, out of the RTL's reach: B1's worst xdma_aclk path was sc_dma's
# m03_w_node m_sc_areset_r (fo 608, SLICE_X101Y429) into its payload FIFO's output registers' R at
# Y545, 5.35 ns of route at 4 ns. The DIMMs' resets in u_sys the same way: otpu_bcu_wg_sys's
# register a load after each DIMM's synchronizer (g_ch[*].wrs_b, a bridge's AXI half; wrs_s, its
# register slices; g_dimm[*].wrs_p, the pair; B1r's g_ars[0].s_reg[1] into the pair's FIFOs and
# the slices' rings, -0.39 ns at 300 MHz, ~97 % route). OTPU_SCREP_FO: the fan-out from which a
# reset is replicated (default 128).
set fo [expr {[info exists ::env(OTPU_SCREP_FO)] && $::env(OTPU_SCREP_FO) ne "" ? $::env(OTPU_SCREP_FO) : 128}]
set regs [get_cells -quiet -hier -filter {IS_PRIMITIVE && REF_NAME =~ FD* && NAME =~ u_bd/*/sc_*/inst/*_sc_areset_r_reg}]
set wrs [get_cells -quiet -hier -filter {IS_PRIMITIVE && REF_NAME =~ FD* && NAME =~ u_sys/g_*.wrs_?_reg}]
set nets {}
foreach r [concat $regs $wrs] {
  set n [get_nets -quiet -of_objects [get_pins -quiet $r/Q]]
  if {[llength $n] == 1 && [get_property FLAT_PIN_COUNT $n] - 1 >= $fo} { lappend nets $n }
}
puts "phys_screp.tcl: [llength $regs] SmartConnect reset registers, [llength $wrs] DIMM reset copies, [llength $nets] of fan-out >= $fo"
foreach n $nets { puts "phys_screp.tcl: fan-out [expr {[get_property FLAT_PIN_COUNT $n] - 1}] $n" }
if {[llength $nets]} { phys_opt_design -force_replication_on_nets $nets }
