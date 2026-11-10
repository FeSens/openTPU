# Clock-domain crossings of the BCU1525 top (otpu_fpga_top_bcu) outside otpu_mem_ch (whose own are
# boards/ypcb-00338/constraints/otpu_mem_ch.tcl, scoped to the module: create_project.tcl adds it
# for every otpu_mem_ch here too). Unmanaged Tcl, read in implementation only and late, as the
# YPCB's otpu_top_native.tcl, whose reasoning it follows: no clock pair is declared asynchronous
# (that would take priority over otpu_mem_ch's set_max_delay -datapath_only), so every path between
# two clocks is a max delay without skew, here, in otpu_mem_ch.tcl, in LiteX's XDC or in the
# SmartConnects' own constraints. UNVERIFIED: written before a Vivado with UltraScale+ parts was
# available (docs/bcu1525.md section 9).
#
# The clocks: core_clk (the block design's MMCM on clk300_1), acc_clk (the same MMCM's second
# output at core_clk's divide and phase behind a BUFGCE: the accelerator's, u_sys's; synchronous
# with core_clk, so the paths between them are timed as they are), xdma_aclk (XDMA's, from the PCIe
# reference clock) and each DIMM's controller clock (its core's sys_clk, its MMCM on clk300_<d>):
# board channel j's is the uclk of u_sys/g_ch[j].u_ch. core_clk from the top's net (u_ch's clk is
# acc_clk); the governor, the CSR ports' master side and the DIMM synchronizers run on it.
set c_core [get_clocks -quiet -of_objects [get_nets -quiet core_clk]]
set c_acc  [get_clocks -quiet -of_objects [get_pins -quiet {u_sys/g_ch[0].u_ch/clk}]]
set c_x    [get_clocks -quiet -of_objects [get_pins -quiet {u_sys/g_ch[0].u_ch/xclk}]]
foreach v {c_core c_acc c_x} {
  if {[llength [set $v]] != 1} {
    puts "CRITICAL WARNING: \[bcu1525_top.tcl\] $v is '[set $v]', expected one clock: crossings unconstrained"
  }
}
set t_core [get_property -quiet -min PERIOD $c_core]
set t_x    [get_property -quiet -min PERIOD $c_x]
# board channel j -> DIMM (otpu_fpga_top_bcu dimm())
set dimm_of {0 0 1 1 2 3 3 2}
foreach {j d} $dimm_of {
  set c_ucl [get_clocks -quiet -of_objects [get_pins -quiet "u_sys/g_ch\[$j\].u_ch/uclk"]]
  if {[llength $c_ucl] != 1} {
    puts "CRITICAL WARNING: \[bcu1525_top.tcl\] channel $j (DIMM C$d): controller clock '$c_ucl'"
    continue
  }
  set t_ucl [get_property -quiet -min PERIOD $c_ucl]
  # DIMM d's CSR port (AXI-Lite on core_clk) crossed into its sys clock inside the core (LiteX's
  # AXILiteClockDomainCrossing): one destination period, datapath only, into the core's registers
  set ld_sys [filter -quiet [all_registers -clock $c_ucl] "NAME =~ u_ld$d/*"]
  set ld_ctl [filter -quiet [all_registers -clock $c_core] "NAME =~ u_ld$d/*"]
  if {[llength $ld_sys]} { set_max_delay -datapath_only -from $c_core -to $ld_sys $t_ucl }
  if {[llength $ld_ctl]} { set_max_delay -datapath_only -from $c_ucl -to $ld_ctl $t_core }
  # its calibration and uncorrectable-ECC flags into otpu_board's synchronizers (cal_s1 / ded_s1
  # bit j, acc_clk), its sys reset into the CSR port's (ld_crs1 bit d) and its ready into the
  # governor's DIMM gates' (cal_c1 bit d, both core_clk): one core_clk period
  foreach pat [list "u_sys/u_board/cal_s1_reg\[$j\]" "u_sys/u_board/ded_s1_reg\[$j\]" "ld_crs1_reg\[$d\]" "cal_c1_reg\[$d\]"] {
    set s1 [get_cells -quiet $pat]
    if {[llength $s1]} {
      set_max_delay -datapath_only -from $c_ucl -to $s1 $t_core
    } else {
      puts "CRITICAL WARNING: \[bcu1525_top.tcl\] no synchronizer $pat"
    }
  }
  # the core's own MMCM reset chain in its 300 MHz input clock, reset from sys (the YPCB's
  # sys_clk_50 case) and from the governor (dimm_on, the core's rst: core_clk): false paths into
  # those registers alone; the governor's UART select into the core's UART receiver (a two-flop
  # synchronizer), as bcu1525_mem_top.tcl
  set c_in [get_clocks -quiet -of_objects [get_pins -quiet "g_bufg\[$d\].u_bufg/O"]]
  if {[llength $c_in] == 1} {
    set ld_in [filter -quiet [all_registers -clock $c_in] "NAME =~ u_ld$d/*"]
    if {[llength $ld_in]} {
      set_false_path -from $c_ucl -to $ld_in
      set gd [get_cells -quiet {u_gov/dimm_on_reg[*]}]
      if {[llength $gd]} { set_false_path -from $gd -to $ld_in } else {
        puts "CRITICAL WARNING: \[bcu1525_top.tcl\] no u_gov/dimm_on_reg"
      }
    } else {
      puts "CRITICAL WARNING: \[bcu1525_top.tcl\] DIMM C$d: no registers on its 300 MHz input"
    }
  }
  # (an empty -from or -to is an error, which ends this script: every set below is checked)
  set gu [get_cells -quiet {u_gov/usel_reg[*]}]
  if {[llength $gu] && [llength $ld_sys]} { set_false_path -from $gu -to $ld_sys } else {
    puts "CRITICAL WARNING: \[bcu1525_top.tcl\] DIMM C$d: UART select [llength $gu] / [llength $ld_sys] registers"
  }
  puts "bcu1525_top.tcl: channel $j on DIMM C$d, [get_property NAME $c_ucl]: [llength $ld_sys] sys / [llength $ld_ctl] core_clk registers in u_ld$d"
}

# The DIMMs' delay controllers: each core has its own IDELAYCTRL (LiteX's USPIDELAYCTRL, on its MMCM's
# 400 MHz reference) for its IDELAYE3 / ODELAYE3 in TIME mode. Four ungrouped controllers fail
# placement's DRC (PLIDC-7), so each core's controller and delays form one IODELAY_GROUP, which
# Vivado replicates the controller across (the banks of that DIMM's pins).
foreach d {0 1 2 3} {
  set dly [get_cells -quiet -hierarchical -filter "NAME =~ u_ld$d/* && (REF_NAME == IDELAYCTRL || REF_NAME == IDELAYE3 || REF_NAME == ODELAYE3)"]
  set ctl [filter -quiet $dly {REF_NAME == IDELAYCTRL}]
  if {[llength $ctl] == 1} {
    set_property IODELAY_GROUP otpu_ld$d $dly
    puts "bcu1525_top.tcl: DIMM C$d IODELAY_GROUP otpu_ld$d: [llength $ctl] IDELAYCTRL, [expr {[llength $dly] - 1}] delays"
  } else {
    puts "CRITICAL WARNING: \[bcu1525_top.tcl\] u_ld$d has [llength $ctl] IDELAYCTRLs ([llength $dly] delay cells): no IODELAY_GROUP"
  }
}

# the governor's DMA gates (otpu_axi_off off / dead per channel, registered in core_clk) into
# their synchronizers in xdma_aclk (xo1 / xd1): one xdma_aclk period
set xs1 [get_cells -quiet {xo1_reg[*] xd1_reg[*]}]
if {[llength $xs1] && [llength $c_x] == 1 && [llength $c_core] == 1} {
  set_max_delay -datapath_only -from $c_core -to $xs1 $t_x
  puts "bcu1525_top.tcl: [llength $xs1] DMA gate synchronizers"
} else {
  puts "CRITICAL WARNING: \[bcu1525_top.tcl\] the DMA gates: [llength $xs1] synchronizer cells"
}

# report_cdc waivers for XDMA's read data: each channel's u_xr (distributed RAM written in its
# controller clock) read asynchronously onto its R channel into sc_dma (otpu_mem_ch.tcl bounds the
# paths); safe by the pointer protocol (otpu_top_native.tcl)
set xr_ram [get_cells -quiet -hierarchical -filter {NAME =~ u_sys/g_ch*.u_ch/u_xr/mem_reg*}]
set xr_src [get_pins -quiet -filter {IS_LEAF && REF_PIN_NAME == CLK} -of_objects $xr_ram]
set xr_out [get_pins -quiet -filter {IS_LEAF && DIRECTION == OUT} -of_objects $xr_ram]
set xr_dst {}
if {[llength $xr_out]} {
  set xr_dst [filter [all_fanout -quiet -from $xr_out -endpoints_only -flat] {NAME !~ u_sys/g_ch*.u_ch/*}]
}
if {[llength $xr_src] && [llength $xr_dst]} {
  foreach id {CDC-1 CDC-4 CDC-13 CDC-14 CDC-15 CDC-16 CDC-26} {
    if {[catch {create_waiver -type CDC -id $id -user "otpu_bcu_sys" -description "otpu_mem_ch u_xr's head read asynchronously onto XDMA's R channel (sc_dma); taken only with x_rvalid, the FIFO's registered not-empty in xdma_aclk" -from $xr_src -to $xr_dst} msg]} {
      puts "CRITICAL WARNING: \[bcu1525_top.tcl\] $id waiver not created: $msg"
    }
  }
} else {
  puts "CRITICAL WARNING: \[bcu1525_top.tcl\] XDMA read-data waivers: [llength $xr_src] RAM clock pins, [llength $xr_dst] endpoints"
}
if {[catch {create_waiver -type METHODOLOGY -id {TIMING-9} -user "otpu_bcu_sys" -description "otpu_mem_ch's FIFO heads and XDMA's read data from them; each crossing is waived in report_cdc by its two ends"} msg]} {
  puts "CRITICAL WARNING: \[bcu1525_top.tcl\] TIMING-9 waiver not created: $msg"
}

# the SYSMON's over-temperature threshold as configured (the build's report: INIT_53 holds the OT
# upper code and, in its 4 low bits 0x3, its enable; without them the SYSMON's 125 C default)
set sm [get_cells -quiet -hierarchical -filter {REF_NAME == SYSMONE4}]
foreach c $sm {
  puts "bcu1525_top.tcl: $c INIT_53 [get_property -quiet INIT_53 $c] INIT_57 [get_property -quiet INIT_57 $c] INIT_50 [get_property -quiet INIT_50 $c] INIT_54 [get_property -quiet INIT_54 $c] INIT_41 [get_property -quiet INIT_41 $c]"
}

# the internal VREF of every bank with DDR4 data (POD12: 0.84 V, 70 % of VDDQ). Vivado chose 0.600
# (SSTL12's) for the address banks 42 / 47 / 65 / 70, and bank 47 also holds DIMM C2's bytes 5
# (dq[40..47]): the 2026-10-21 image needed an ECO for it (BIIVRC-1; docs/bcu1525.md section 12)
set pod [get_ports -quiet -filter {IOSTANDARD =~ POD12*}]
if {[catch {get_iobanks -of_objects $pod} banks]} {
  puts "CRITICAL WARNING: \[bcu1525_top.tcl\] the POD12 ports' banks: $banks"
} else {
  foreach b $banks { set_property INTERNAL_VREF 0.84 $b }
  puts "bcu1525_top.tcl: INTERNAL_VREF 0.84 on banks [lsort $banks]"
}
