# Clock-domain crossings of the BCU1525's memory-test top (otpu_fpga_top_bcu_mem; docs/bcu1525.md
# section 14), as bcu1525_top.tcl's for the full top: unmanaged Tcl, implementation only, late.
# core_clk (the block design's MMCM on clk300_1) runs the governor, the BMC's bridge and the CSR
# ports' master side; each DIMM's core runs its own sys clock (its MMCM on clk300_<d>).
# leaf pins and the top's nets, not the modules' ports: the top leaves each core's sys_clk output
# unread, so synthesis drops it (the 2026-10-26 build found no clock on either)
set c_core [get_clocks -quiet -of_objects [get_nets -quiet core_clk]]
if {[llength $c_core] != 1} {
  puts "CRITICAL WARNING: \[bcu1525_mem_top.tcl\] core_clk is '$c_core', expected one clock: crossings unconstrained"
}
set t_core [get_property -quiet -min PERIOD $c_core]
foreach d {0 1 2 3} {
  set c_ucl [get_clocks -quiet -of_objects [get_pins -quiet u_ld$d/BUFGCE_DIV/O]]
  if {[llength $c_ucl] != 1} {
    puts "CRITICAL WARNING: \[bcu1525_mem_top.tcl\] DIMM C$d: controller clock '$c_ucl'"
    continue
  }
  set t_ucl [get_property -quiet -min PERIOD $c_ucl]
  # the CSR port crossed inside the core (LiteX's AXILiteClockDomainCrossing): one destination
  # period, datapath only
  set ld_sys [filter -quiet [all_registers -clock $c_ucl] "NAME =~ u_ld$d/*"]
  set ld_ctl [filter -quiet [all_registers -clock $c_core] "NAME =~ u_ld$d/*"]
  if {[llength $ld_sys]} { set_max_delay -datapath_only -from $c_core -to $ld_sys $t_ucl }
  if {[llength $ld_ctl]} { set_max_delay -datapath_only -from $c_ucl -to $ld_ctl $t_core }
  # its sys reset into the CSR port's synchronizer (ld_crs1 bit d): one core_clk period
  set s1 [get_cells -quiet "ld_crs1_reg\[$d\]"]
  if {[llength $s1]} {
    set_max_delay -datapath_only -from $c_ucl -to $s1 $t_core
  } else {
    puts "CRITICAL WARNING: \[bcu1525_mem_top.tcl\] no synchronizer ld_crs1_reg\[$d\]"
  }
  # the core's MMCM reset chain on its 300 MHz input: from its own sys clock (LiteX's soc reset)
  # and from the governor (dimm_on, the core's rst: core_clk) false paths into those registers
  # alone; the governor's UART select into the core's UART receiver (a two-flop synchronizer)
  set c_in [get_clocks -quiet -of_objects [get_pins -quiet "g_bufg\[$d\].u_bufg/O"]]
  if {[llength $c_in] == 1} {
    set ld_in [filter -quiet [all_registers -clock $c_in] "NAME =~ u_ld$d/*"]
    if {[llength $ld_in]} {
      set_false_path -from $c_ucl -to $ld_in
      set gd [get_cells -quiet {u_gov/dimm_on_reg[*]}]
      if {[llength $gd]} { set_false_path -from $gd -to $ld_in } else {
        puts "CRITICAL WARNING: \[bcu1525_mem_top.tcl\] no u_gov/dimm_on_reg"
      }
    } else {
      puts "CRITICAL WARNING: \[bcu1525_mem_top.tcl\] DIMM C$d: no registers on its 300 MHz input"
    }
  }
  # (an empty -from or -to is an error, which ends this script: every set below is checked)
  set gu [get_cells -quiet {u_gov/usel_reg[*]}]
  if {[llength $gu] && [llength $ld_sys]} { set_false_path -from $gu -to $ld_sys } else {
    puts "CRITICAL WARNING: \[bcu1525_mem_top.tcl\] DIMM C$d: UART select [llength $gu] / [llength $ld_sys] registers"
  }
  puts "bcu1525_mem_top.tcl: DIMM C$d, [get_property NAME $c_ucl]: [llength $ld_sys] sys / [llength $ld_ctl] core_clk registers in u_ld$d"
}
# the board UART's output mux (uart_sel, core_clk; each core's TX, its sys clock): to a pin
# without an output delay, unconstrained either way

# each core's IDELAYCTRL and delays in one IODELAY_GROUP (bcu1525_top.tcl's PLIDC-7)
foreach d {0 1 2 3} {
  set dly [get_cells -quiet -hierarchical -filter "NAME =~ u_ld$d/* && (REF_NAME == IDELAYCTRL || REF_NAME == IDELAYE3 || REF_NAME == ODELAYE3)"]
  set ctl [filter -quiet $dly {REF_NAME == IDELAYCTRL}]
  if {[llength $ctl] == 1} {
    set_property IODELAY_GROUP otpu_ld$d $dly
    puts "bcu1525_mem_top.tcl: DIMM C$d IODELAY_GROUP otpu_ld$d: [llength $ctl] IDELAYCTRL, [expr {[llength $dly] - 1}] delays"
  } else {
    puts "CRITICAL WARNING: \[bcu1525_mem_top.tcl\] u_ld$d has [llength $ctl] IDELAYCTRLs ([llength $dly] delay cells): no IODELAY_GROUP"
  }
}

# the SYSMON's over-temperature threshold as configured (the build's report: INIT_53 holds the OT
# upper code and, in its 4 low bits 0x3, its enable; without them the SYSMON's 125 C default)
set sm [get_cells -quiet -hierarchical -filter {REF_NAME == SYSMONE4}]
foreach c $sm {
  puts "bcu1525_mem_top.tcl: $c INIT_53 [get_property -quiet INIT_53 $c] INIT_57 [get_property -quiet INIT_57 $c] INIT_50 [get_property -quiet INIT_50 $c] INIT_54 [get_property -quiet INIT_54 $c] INIT_41 [get_property -quiet INIT_41 $c]"
}

# the internal VREF of every bank with DDR4 data (POD12: 0.84 V, 70 % of VDDQ). Vivado chose 0.600
# (SSTL12's) for the address banks 42 / 47 / 65 / 70, and bank 47 also holds DIMM C2's bytes 5
# (dq[40..47]): the 2026-10-21 image needed an ECO for it (BIIVRC-1; docs/bcu1525.md section 12)
set pod [get_ports -quiet -filter {IOSTANDARD =~ POD12*}]
if {[catch {get_iobanks -of_objects $pod} banks]} {
  puts "CRITICAL WARNING: \[bcu1525_mem_top.tcl\] the POD12 ports' banks: $banks"
} else {
  foreach b $banks { set_property INTERNAL_VREF 0.84 $b }
  puts "bcu1525_mem_top.tcl: INTERNAL_VREF 0.84 on banks [lsort $banks]"
}
