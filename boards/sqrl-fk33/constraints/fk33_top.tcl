# The FK33 top's crossings (unmanaged Tcl, read late on the synthesized netlist; create_project.tcl). Every clock comes
# from one MMCM on XDMA's axi_aclk, so Vivado would time these as synchronous paths: each synchronizer's first stage
# is cut. otpu_mem_ch's own crossings are fk33_mem_ch.tcl's (scoped to it).
proc fk33_sync_to {pattern what} {
  set c [get_cells -quiet -hier -filter "NAME =~ $pattern"]
  if {[llength $c] == 0} {
    puts "CRITICAL WARNING: \[fk33_top.tcl\] no cells '$pattern': $what unconstrained"
    return
  }
  set_false_path -to $c
  puts "fk33_top.tcl: $what, [llength $c] synchronizer flip-flops"
}
fk33_sync_to {hcal_u1_reg*} {HBM calibration (APB clk100) into uclk}
fk33_sync_to {u_sys/u_board/cal_s1_reg*} {HBM calibration and CATTRIP into core_clk (otpu_board)}
fk33_sync_to {lrs1_reg} {the control masters' reset (xdma_aclk) into core_clk}
# the block design's reset blocks: their asynchronous inputs (the MMCM's lock, axi_aresetn)
set p [get_pins -quiet -hier -filter {NAME =~ u_bd/*rst_*/ext_reset_in || NAME =~ u_bd/*rst_*/dcm_locked}]
if {[llength $p]} {
  set_false_path -through $p
  puts "fk33_top.tcl: reset inputs, [llength $p] pins"
} else {
  puts "CRITICAL WARNING: \[fk33_top.tcl\] no proc_sys_reset inputs"
}
# The HBM IP's debug core makes Vivado insert a debug hub, whose clock must be free running (Chipscope 16-213
# otherwise): clk100, the HBM's APB clock
set n [get_nets -quiet clk100]
if {[llength [get_debug_cores -quiet dbg_hub]] && [llength $n] == 1} {
  set_property C_CLK_INPUT_FREQ_HZ 100000000 [get_debug_cores dbg_hub]
  set_property C_ENABLE_CLK_DIVIDER false [get_debug_cores dbg_hub]
  set_property C_USER_SCAN_CHAIN 1 [get_debug_cores dbg_hub]
  connect_debug_port dbg_hub/clk $n
  puts "fk33_top.tcl: debug hub on clk100"
} else {
  puts "fk33_top.tcl: no debug hub ([llength [get_debug_cores -quiet]] debug cores), or clk100 not found ([llength $n] nets)"
}
