# The SLR floorplan of the FK35 build (docs/sqrl-fk35.md). Unmanaged Tcl (cell lookups with checks), read in
# implementation only and late (create_project.tcl). The BCU1525's bcu1525_floorplan.tcl is the model; the layout is
# the openTPU_FK FK35 block-design build's, which met 250 / 250 MHz on this part.
#
# The xcvu35p's two SLRs: both HBM stacks (all 32 AXI ports) at the bottom edge of SLR0, the PCIe block (PCIE4C_X1Y0,
# GTY quads 224-227) in SLR0 too.
#   - pb_mem, SLR0 (the memory side): every channel's otpu_mem_ch, its HBM adapter (otpu_hbm_port) and the u_rx half
#     of its native-port crossing. The block design (XDMA) and the HBM IP stay unconstrained: their sites are fixed
#     in SLR0 anyway.
#   - pb_cmp, SLR1 (the compute side): otpu_board, i.e. both slices, their native DRAM blocks and the shell (control
#     block, collective unit, trace and the slices' crossings, otpu_board g_xs), and the u_tx halves.
# Slice 0, the shell and the memory side in SLR0 (the BCU1525's split: the shell with the PCIe block) filled ~93% of
# SLR0's usable CLBs in the openTPU_FK FK35 build and did not route; this way SLR0 is about a quarter full and SLR1 about
# two thirds (the two-channel slices; with DUAL's eight channels SLR0 about half).
#
# What crosses SLR1 -> SLR0 (core_clk): each channel's native port (eight with DUAL), through two otpu_native_reg
# (otpu_fk35_sys SLL, all ones): u_tx with the slice, u_rx with otpu_mem_ch; the read data, write count and error flags leave u_rx from
# flip-flops into u_tx's (flop to flop: USER_SLL_REG on both puts them in the Laguna sites), the commands and write
# data leave u_tx from its skid buffers' output registers (Laguna TX) into u_rx's skid entry. The core reset crosses
# as otpu_fk35_sys's copies (rst_b with the slices, rst_c[j] with each channel). The control registers (the block
# design's SmartConnect -> otpu_board's AXI-Lite isolator) cross as AXI-Lite, the die temperature as a register.
proc fk35_pb {name slr cells} {
  set found {}
  foreach c $cells {
    set x [get_cells -quiet $c]
    if {[llength $x]} { lappend found $x } else {
      puts "CRITICAL WARNING: \[fk35_floorplan.tcl\] $name: no cell $c"
    }
  }
  set pb [create_pblock $name]
  resize_pblock $pb -add $slr
  if {[llength $found]} { add_cells_to_pblock $pb [join $found] }
  puts "fk35_floorplan.tcl: $name in $slr, [llength $found] of [llength $cells] cells"
}
#
# MEMX (the top's, OTPU_MEMX=1): the slices' memory adapters (otpu_native_xdram2: u_dram and the crossing's memory
# side u_m) in SLR0 with the channels, the crossing's slice side (u_s) in SLR1 with the rest of otpu_board; the
# crossing is on the slices' ports (u_s / u_m sll_* registers, Laguna), the native ports stay in SLR0
set xm [list u_sys/u_board/g_memx.u_mem "u_sys/u_board/g_slice\[1\].g_memx.u_mem"]
set memx [llength [get_cells -quiet [lindex $xm 0]]]
set mem {}
if {$memx} {
  # otpu_board's cells but the adapters (kept as cells: otpu_native_xdram2, otpu_board), by name
  set cmp [list u_sys/rst_b_reg]
  foreach c [get_cells u_sys/u_board/* -filter {NAME !~ "*g_memx.u_mem"}] { lappend cmp [get_property NAME $c] }
  foreach x $xm { lappend mem "$x/u_m" "$x/u_dram"; lappend cmp "$x/u_s" }
} else {
  set cmp [list u_sys/u_board u_sys/rst_b_reg]
}
foreach j {0 1 2 3 4 5 6 7} {
  lappend mem "u_sys/g_ch\[$j\].u_ch" "g_hbm\[$j\].u_port" "u_sys/rst_c_reg\[$j\]"
  if {!$memx} {
    lappend mem "u_sys/g_ch\[$j\].g_sll.u_rx"
    lappend cmp "u_sys/g_ch\[$j\].g_sll.u_tx"
  }
}
fk35_pb pb_mem SLR0 $mem
fk35_pb pb_cmp SLR1 $cmp

# Gen4 (XDMA's 512-bit AXI: sc_dma's S00 has 512 data bits; Gen3's 256): XDMA (54k LUTs, Gen3's 24k) and sc_dma's
# 512-bit entry (its S00 entry pipeline and nodes, the width conversion) in a hard pblock, the two right columns of
# SLR0 (the PCIe block at X7Y0) and the top row's right half; sc_dma's 128-bit masters stay free. Unconstrained,
# f35d000c spread XDMA over X64-232 Y0-239, a fifth of it among channels 2-5 and the memory adapters in X2-X5 rows
# 0-2, whose paths then failed (-0.52 ns core, -0.40 uclk). Other cells may still use those regions.
if {[llength [get_pins -quiet {u_bd/otpu_bd_i/sc_dma/S00_AXI_wdata[511]}]]} {
  set pbc [get_cells u_bd/otpu_bd_i/xdma_0]
  foreach c {s00_entry_pipeline s00_nodes} {
    set x [get_cells -quiet u_bd/otpu_bd_i/sc_dma/inst/$c]
    if {[llength $x]} { lappend pbc $x } else { puts "CRITICAL WARNING: \[fk35_floorplan.tcl\] pb_xdma: no sc_dma $c" }
  }
  set pb [create_pblock pb_xdma]
  resize_pblock $pb -add {CLOCKREGION_X6Y0:CLOCKREGION_X7Y3 CLOCKREGION_X3Y3:CLOCKREGION_X5Y3}
  set_property IS_SOFT false $pb
  add_cells_to_pblock $pb $pbc
  puts "fk35_floorplan.tcl: pb_xdma, [llength $pbc] cells ($pbc) in X6Y0:X7Y3 X3Y3:X5Y3"
}

if {$memx} {
  # Laguna registers on the slices' port crossings: each side's sll_* registers (u_s's requests out and replies in,
  # u_m's requests in and replies out)
  foreach x $xm {
    set sll {}
    foreach pat [list "$x/u_s/sll_*_reg*" "$x/u_m/sll_*_reg*"] {
      set sll [concat $sll [get_cells -quiet $pat -filter {REF_NAME =~ FD*}]]
    }
    if {[llength $sll]} {
      set_property USER_SLL_REG TRUE $sll
      puts "fk35_floorplan.tcl: $x, [llength $sll] Laguna registers"
    } else {
      puts "CRITICAL WARNING: \[fk35_floorplan.tcl\] $x: no crossing registers found"
    }
  }
}

# Laguna registers on the flop-to-flop crossings, per channel: the read data, write count and error flags (u_rx's
# s_* registers drive u_tx's directly) and the commands' / write data's output registers (u_tx's m_* registers).
# (Not with MEMX, whose channels do not cross; a return does not end a script read as a constraint.)
foreach j [expr {$memx ? {} : {0 1 2 3 4 5 6 7}}] {
  set sll {}
  set g "u_sys/g_ch\[$j\].g_sll"
  foreach pat [list "$g.u_rx/s_rdata_reg*" "$g.u_rx/s_wdone_reg*" "$g.u_rx/s_err_reg" "$g.u_tx/s_rdata_reg*" \
               "$g.u_tx/s_wdone_reg*" "$g.u_tx/s_err_reg" "$g.u_tx/u_cmd/m_data_reg*" "$g.u_tx/u_wd/m_data_reg*"] {
    set sll [concat $sll [get_cells -quiet $pat -filter {REF_NAME =~ FD*}]]
  }
  if {[llength $sll]} {
    set_property USER_SLL_REG TRUE $sll
    puts "fk35_floorplan.tcl: channel $j, [llength $sll] Laguna registers"
  } else {
    puts "CRITICAL WARNING: \[fk35_floorplan.tcl\] channel $j: no crossing registers found"
  }
}
