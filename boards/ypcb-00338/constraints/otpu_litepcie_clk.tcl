# LitePCIe's PHY clocks (otpu_litepcie u_lp, LitePCIe's S7PCIEPHY). Unmanaged Tcl, read in
# implementation only and late (create_project.tcl), on the netlist: it selects cells by type.
#
# The PHY's MMCM makes the pipe clocks from the lane 0 GT's TXOUTCLK (the IP's txoutclk_x0y0,
# 100 MHz): 125 and 250 MHz, userclk1 (500 MHz, the hard block's core at Gen2 x8) and userclk2
# (250 MHz: the user clock, the top's xclk). The pipe clock pclk is a BUFGCTRL between the 125 and
# 250 MHz outputs (the Gen1 / Gen2 rate), and Vivado propagates both clocks through it. As in
# Xilinx's own pipe clock module's constraints (the XDMA IP carried them), each gets a generated
# clock on the mux's output, and the two are physically exclusive: only one drives pclk at a
# time, so no path runs from one to the other.
set lp_mux [get_cells -quiet -hierarchical -filter {REF_NAME == BUFGCTRL && NAME =~ u_lp/*}]
if {[llength $lp_mux] == 1} {
  set lp_i0 [get_pins $lp_mux/I0]
  set lp_i1 [get_pins $lp_mux/I1]
  set lp_c0 [get_clocks -quiet -of_objects $lp_i0]
  set lp_c1 [get_clocks -quiet -of_objects $lp_i1]
  if {[llength $lp_c0] == 1 && [llength $lp_c1] == 1} {
    create_generated_clock -name lp_pclk_125 -source $lp_i0 -divide_by 1 -add -master_clock $lp_c0 \
      [get_pins $lp_mux/O]
    create_generated_clock -name lp_pclk_250 -source $lp_i1 -divide_by 1 -add -master_clock $lp_c1 \
      [get_pins $lp_mux/O]
    set_clock_groups -name lp_pclk_mux -physically_exclusive -group lp_pclk_125 -group lp_pclk_250
    puts "otpu_litepcie_clk.tcl: pclk mux [get_property NAME $lp_mux]: [get_property NAME $lp_c0] / [get_property NAME $lp_c1]"
  } else {
    puts "CRITICAL WARNING: \[otpu_litepcie_clk.tcl\] the pclk mux's inputs carry '$lp_c0' / '$lp_c1', expected one clock each"
  }
} else {
  puts "CRITICAL WARNING: \[otpu_litepcie_clk.tcl\] [llength $lp_mux] BUFGCTRL in u_lp, expected the pclk mux"
}
