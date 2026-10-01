# Clock-domain crossings of the top (create_project.tcl: otpu_fpga_top_ld) outside otpu_mem_ch
# (whose own are in otpu_mem_ch.tcl, scoped to the module). Unmanaged Tcl, read in implementation
# only and late (PROCESSING_ORDER LATE: after the IP's and LiteX's constraints, on the netlist
# synthesis made), because it selects cells by clock and by fan-out.
#
# The clocks: core_clk (the block design's MMCM) and the channels' controller clock (the
# LiteDRAM core's sys, its MMCM on the same 50 MHz buffer) come from the 50 MHz oscillator, so
# Vivado relates them;
# xdma_aclk (125 MHz; 250 at PCIE_GEN 2) comes from the PCIe reference clock and is asynchronous
# to all of them, but nothing declares it so: no set_clock_groups and no clock-to-clock false
# path, which would take priority over otpu_mem_ch's set_max_delay -datapath_only and leave its
# crossings untimed. Every path between two of the clocks is therefore a max delay without clock
# skew, here, in otpu_mem_ch.tcl, in LiteX's XDC (its synchronizers' first stages are false paths
# by their mr_ff / ars_ff attributes), in the SmartConnect's own constraints (the control
# registers, xdma_aclk -> core_clk; the LiteDRAM CSRs too at PCIE_GEN 2) or a false path into a
# 2-flip-flop synchronizer (otpu_top_ld.xdc: the temperature, the I2C pins).
#
# The clocks by the pins of otpu_mem_ch u_sys/u_ch0, the ones its scoped constraints use.
set c_core [get_clocks -quiet -of_objects [get_pins -quiet u_sys/u_ch0/clk]]
set c_ucl  [get_clocks -quiet -of_objects [get_pins -quiet u_sys/u_ch0/uclk]]
set c_x    [get_clocks -quiet -of_objects [get_pins -quiet u_sys/u_ch0/xclk]]
foreach v {c_core c_ucl c_x} {
  if {[llength [set $v]] != 1} {
    puts "CRITICAL WARNING: \[otpu_top_native.tcl\] $v is '[set $v]', expected one clock: crossings unconstrained"
  }
}
set t_core [get_property -quiet -min PERIOD $c_core]
set t_ucl  [get_property -quiet -min PERIOD $c_ucl]
set t_x    [get_property -quiet -min PERIOD $c_x]
# The LiteDRAM core's CSR clock (its ctl_clk): xdma_aclk, core_clk at PCIE_GEN 2
# (otpu_fpga_top_ld.sv), whichever of the two clocks registers in u_ld. Not from the pin
# u_ld/ctl_clk: synthesis does not keep it (7b1cc919 found no clock there, kept the crossing's
# max delays on xdma_aclk and left core_clk <-> sys timed as synchronous, 2.5 ns apart).
set c_ctl {}
foreach c [concat $c_core $c_x] {
  if {[llength [filter -quiet [all_registers -clock $c] {NAME =~ u_ld/*}]]} { lappend c_ctl $c }
}
if {[llength $c_ctl] != 1} {
  puts "CRITICAL WARNING: \[otpu_top_native.tcl\] the LiteDRAM CSR clock is '$c_ctl', expected one of core_clk and xdma_aclk to clock registers in u_ld: crossing unconstrained"
}
set t_ctl  [get_property -quiet -min PERIOD $c_ctl]

# ---- the LiteDRAM core (u_ld)
if {[llength [get_cells -quiet u_ld]]} {
  # The core's CSR port (BAR0 0x10000), AXI-Lite in its ctl_clk (c_ctl: xdma_aclk, core_clk at
  # PCIE_GEN 2), crossed into sys (both channels' controller clock, c_ucl) by LiteX's
  # AXILiteClockDomainCrossing (stream AsyncFIFOs: gray pointers through MultiRegs, whose first
  # stages LiteX's XDC false-paths; the storage written in one clock and read in the other). One
  # destination period, datapath only, into the registers of the other clock inside the core and
  # nowhere else.
  if {[llength $c_ucl] == 1 && [llength $c_ctl] == 1} {
    set ld_sys [filter -quiet [all_registers -clock $c_ucl] {NAME =~ u_ld/*}]
    set ld_ctl [filter -quiet [all_registers -clock $c_ctl] {NAME =~ u_ld/*}]
    if {[llength $ld_sys]}  { set_max_delay -datapath_only -from $c_ctl -to $ld_sys $t_ucl }
    if {[llength $ld_ctl]}  { set_max_delay -datapath_only -from $c_ucl -to $ld_ctl $t_ctl }
    puts "otpu_top_native.tcl: LiteDRAM CSR crossing, [llength $ld_sys] sys / [llength $ld_ctl] [get_property NAME $c_ctl] registers in u_ld"
  }
  # LiteX's MMCM and PLL reset chains (per primitive 8 FDCE in the 50 MHz clock that feeds it,
  # crg_mmcm_reset -> s7mmcm_reset0..): their reset, the ctrl CSR's soc_rst, comes from sys. The
  # standalone images false-path sys -> clkin (WLCRG's add_false_path_constraints); the core's
  # XDC does not carry it. From sys into those registers alone (on the routed 14875bf: -2.780 ns
  # at a 2.5 ns requirement, reset_wr_stb -> FDCE / FDCE_8).
  set c_50  [get_clocks -quiet sys_clk_50]
  set ld_50 [filter -quiet [all_registers -clock $c_50] {NAME =~ u_ld/*}]
  if {[llength $ld_50] && [llength $c_ucl] == 1} {
    set_false_path -from $c_ucl -to $ld_50
    puts "otpu_top_native.tcl: LiteDRAM MMCM / PLL reset chains, [llength $ld_50] sys_clk_50 registers in u_ld"
  } else {
    puts "CRITICAL WARNING: \[otpu_top_native.tcl\] LiteDRAM reset chains: [llength $ld_50] sys_clk_50 registers in u_ld"
  }
  # The calibration flags (the core's cal_ready CSRs, sys) into the accelerator's synchronizer
  # (STATUS CALIB0/1; 2 flip-flops, ASYNC_REG): one core_clk period, datapath only (otpu_top_ld.xdc
  # has no false path for them).
  set cal_s1 [get_cells -quiet -hier -filter {NAME =~ u_sys/u_board/cal_s1_reg*}]
  if {[llength $cal_s1] && [llength $c_ucl] == 1 && [llength $c_core] == 1} {
    set_max_delay -datapath_only -from $c_ucl -to $cal_s1 $t_core
  } else {
    puts "CRITICAL WARNING: \[otpu_top_native.tcl\] calibration flags: [llength $cal_s1] synchronizer cells"
  }
}

# ---- report_cdc waivers for XDMA's read data. Each channel's XDMA read-data FIFO u_xr
# (distributed RAM, written in the controller clock) is read asynchronously onto its R channel
# (x_rdata, x_rid, x_rresp, x_rlast), which otpu_axi_split2 muxes combinationally into XDMA's
# M_AXI (at PCIE_GEN 2 into otpu_dma_split's register slice); its paths are limited to one
# xdma_aclk period by otpu_mem_ch.tcl (-through the RAM's outputs). The endpoints are XDMA's (and
# otpu_axi_split2's order FIFO, popped on the last beat; the slice's registers at PCIE_GEN 2),
# outside the module, so the waivers are here, from the RAMs' write clocks to exactly the
# endpoints the RAMs' outputs reach outside u_ch0 / u_ch1. Safe by the pointer protocol: XDMA
# takes a beat only with x_rvalid, the FIFO's registered not-empty in xdma_aclk, and an entry is
# written two controller-clock cycles before its write pointer crosses and not rewritten before
# the read pointer crosses back.
set xr_ram [get_cells -quiet -hierarchical -filter {NAME =~ u_sys/u_ch?/u_xr/mem_reg*}]
set xr_src [get_pins -quiet -filter {IS_LEAF && REF_PIN_NAME == CLK} -of_objects $xr_ram]
set xr_out [get_pins -quiet -filter {IS_LEAF && DIRECTION == OUT} -of_objects $xr_ram]
set xr_dst {}
if {[llength $xr_out]} {
  set xr_dst [filter [all_fanout -quiet -from $xr_out -endpoints_only -flat] {NAME !~ u_sys/u_ch?/*}]
}
if {[llength $xr_src] && [llength $xr_dst]} {
  foreach id {CDC-1 CDC-4 CDC-13 CDC-14 CDC-15 CDC-16 CDC-26} {
    if {[catch {create_waiver -type CDC -id $id -user "otpu_native_sys" -description "otpu_mem_ch u_xr's head (distributed RAM, written in the controller clock) read asynchronously onto XDMA's R channel through otpu_axi_split2; taken only with x_rvalid, the FIFO's registered not-empty in xdma_aclk, when the entry is stable" -from $xr_src -to $xr_dst} msg]} {
      puts "CRITICAL WARNING: \[otpu_top_native.tcl\] $id waiver for XDMA's read data not created: $msg"
    }
  }
} else {
  puts "CRITICAL WARNING: \[otpu_top_native.tcl\] XDMA read-data waivers: [llength $xr_src] RAM clock pins, [llength $xr_dst] endpoints"
}

# ---- TIMING-9 (unknown CDC logic: a crossing under set_max_delay -datapath_only with no
# double-registered synchronizer at the capture side) names no cells; it flags the FIFO heads
# above and in otpu_mem_ch.tcl, read across the clocks by design and covered by the report_cdc
# waivers. The waiver takes the whole check, so a new crossing of that kind has to show up in
# report_cdc (which build.tcl writes, reports/cdc.rpt) rather than here.
if {[catch {create_waiver -type METHODOLOGY -id {TIMING-9} -user "otpu_native_sys" -description "otpu_mem_ch's FIFO heads (distributed RAM read asynchronously across the clocks, guarded by the gray pointers) and XDMA's read data from them; each crossing is waived in report_cdc by its two ends"} msg]} {
  puts "CRITICAL WARNING: \[otpu_top_native.tcl\] TIMING-9 waiver not created: $msg"
}
