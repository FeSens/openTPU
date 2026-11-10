# Block design "otpu_bdw" for the BCU1525 at WG 8 (create_project_wg.tcl; targets/bcu1525-wg.toml;
# bd.tcl, the LiteDRAM build's, is its model): PCIe (XDMA, Gen3 x16), the clocks, resets, the
# SYSMON, the control interconnect, XDMA's DMA split onto the DIMMs, and the four DIMMs' memory
# controllers (AMD's DDR4 IP, "MIG", DDR4-2400, AXI4 512 bits at its ui_clk, 300 MHz), each behind
# an AXI Clock Converter (cc_mem<d>) from aclk: one port a DIMM, which the top's otpu_mig_hmux
# shares between the DIMM's pair of accelerator channels and the host's DMA (out of the block
# design at aclk through a SmartConnect, sc_h<d>). No SmartConnect on the accelerator's stream: one
# takes L + 1 cycles for an L-beat burst's R (8 beats: 88.9%, the card's ~86% stream ceiling;
# xsim, 2026-10-29), the clock converter a beat a cycle. The accelerator, its bridges and the
# pairs (otpu_bcu_wg_sys), the hmuxes, the gates and the address remap of the host's DMA, the
# governor and the BMC's bridge are the top's (rtl/boards/bcu1525/otpu_fpga_top_bcu_wg.sv).
# External interfaces:
#   M_AXI_CTL     AXI4-Lite master -> the accelerator's control registers (BAR0 0x00000, core_clk)
#   M_AXI_GOV     AXI4-Lite master -> the thermal governor (otpu_bcu_gov, BAR0 0x10000, core_clk)
#   M_AXI_BMC     AXI4-Lite master -> the BMC's UART bridge (otpu_bmc_uart, BAR0 0x20000, core_clk)
#                 (the SYSMON at BAR0 0x30000 inside; no DIMM CSRs: the MIGs have none without ECC)
#   M_AXI_DMA<d>  AXI4 masters, 512 bits, no IDs, 64-bit addresses, xdma_aclk: XDMA's DMA (and
#                 jtag_axi_1's) for DIMM slot d, card addresses d x W .. (W = 2^(WG_CAW + 7 + WG_S):
#                 1 GiB at WG_CAW 22, 8 GiB at 25 (WG_S 2: 24); the top gates it, permutes the
#                 offset (virtual channel j's 2^(WG_CAW + 7) at j x 2^(WG_CAW + 7) of the card, the
#                 the WG targets' layout) and sends slot d to its DIMM, DMAP's (two slices' floorplan a))
#   S_AXI_H<d>    AXI4 slaves, 512 bits, no IDs, xdma_aclk: the same, gated and remapped (the
#                 DIMM's byte address, 33 bits), into sc_h<d>
#   M_AXI_H<d>    AXI4 masters, 512 bits, no IDs, 33-bit addresses, aclk: sc_h<d>'s, the host's
#                 DMA for DIMM d into the top's otpu_mig_hmux (full-width INCR bursts: no narrow)
#   S_AXI_P<d>    AXI4 slaves, 512 bits, 4 ID bits, aclk: DIMM d's port (otpu_mig_hmux: its pair,
#                 otpu_mig_pair, and the host's DMA) into cc_mem<d>
#   ddr4_c<d>     DIMM d's pins; c300_<d>: its 300 MHz board clock (d = 0, 2, 3: the MIG's own
#                 differential input; DIMM C1's clock is shared, below)
# Clocks: clk300_1 (C1's bank, SLR1, beside the PCIe block) through one IBUFDS (refbuf_c1) into
# both DIMM C1's MIG (System_Clock No_Buffer) and, through a global buffer, the clock wizard:
# core_clk (CORE_MHZ: the control side), acc_clk2x (2 x CORE_MHZ, unbuffered: the top makes the
# accelerator's clocks from it, acc_clk2x and acc_clk at its half, with a BUFGCE_DIV pair on one
# enable, stopped by the governor, and the enable's own BUFG, otpu_fpga_top_bcu_wg; the TMEM's
# pumped RAMs, TMEM_PMP 1) and aclk (300 MHz: the bridges' AXI side, the pairs, the hmuxes, the
# clock converters' slave side, sc_h's master side; ungated). All from one MMCM at VCO 1200 MHz
# (300 x 4): core_clk at VCO / CORE_DIV, acc_clk2x at VCO / (CORE_DIV / 2), aclk at VCO / 4.
# xdma_aclk: 250 MHz.
# Each MIG's ui_clk (its own MMCM on its board clock) is its AXI clock; rst_ui<d> makes its AXI
# reset from its ui_clk_sync_rst. mig_rst<d> (from the top, the governor's: DIMM d off) holds the
# MIG in reset (its MMCM too: the DIMM unclocked), sc_rstn<d> (the top's, in aclk) sc_h<d> and
# cc_mem<d>'s slave side, and through rst_cc<d> (in ui_clk, with ui_clk_sync_rst) its master side:
# both sides reset together whenever the top resets the DIMM.
# Variables (set before sourcing): CORE_MHZ (default 100), WG_S (the slices, make_wg.py's: a DIMM
# holds 2 x WG_S channels; default 1), WG_CAW (the channels' beat address bits, make_wg.py's: 22
# 512 MiB a channel, 25 4 GiB, all of an 8 GB DIMM at WG_S 1, 24 at 2; default 22). 26 (16 GB
# DIMMs, 64 GB) is later work: another MemoryPart, the MIGs' AXI and S_AXI_H / M_AXI_H / S_AXI_P at
# 34 bits.
# Environment: PCIE_LANES (16; 8: Gen3 x8, 256-bit AXI).

proc ip_vlnv {name} {
  set defs [lsort -decreasing [get_ipdefs -all "xilinx.com:ip:${name}:*"]]
  if {[llength $defs] == 0} { error "IP $name not found in this Vivado installation" }
  return [lindex $defs 0]
}

create_bd_design otpu_bdw
current_bd_design otpu_bdw

# the clocks: MMCM divides of the 1200 MHz VCO (core_clk and the 2x rising together), the 2x on
# CLKOUT0 (DIVIDE_F, the MMCM's one fractional divide) and core_clk on CLKOUT1: a 150 MHz
# synthesis (4.0 and 8) is implemented at 133.33 MHz too (4.5 and 9: clk133.tcl, OTPU_CLK_HOOK)
if {![info exists CORE_MHZ]} { set CORE_MHZ 100 }
set VCO 1200
set CORE_DIV [expr {round(double($VCO) / $CORE_MHZ)}]
if {$CORE_DIV % 2 || abs($VCO / double($CORE_DIV) - $CORE_MHZ) > 0.01} {
  error "bd_wg.tcl: core_clk needs an even integer divide of the ${VCO} MHz VCO (acc_clk2x at half of it), not $CORE_MHZ MHz"
}
set CORE_HZ [expr {$VCO * 1000000 / $CORE_DIV}]
set CLK2X_HZ [expr {2 * $CORE_HZ}]
set ACLK_HZ 300000000
set PCIE_LANES [expr {[info exists ::env(PCIE_LANES)] && $::env(PCIE_LANES) ne "" ? $::env(PCIE_LANES) : 16}]
if {$PCIE_LANES ni {8 16}} { error "bd_wg.tcl: PCIE_LANES $PCIE_LANES (8 or 16)" }
set AXI_BITS [expr {$PCIE_LANES == 16 ? 512 : 256}]
set XDMA_MHZ 250
set XDMA_HZ [expr {$XDMA_MHZ * 1000000}]
# a DIMM's window of the card's DMA addresses (2 x WG_S channels of 2^(WG_CAW + 7) bytes), in GiB
if {![info exists WG_CAW]} { set WG_CAW 22 }
if {![info exists WG_S]} { set WG_S 1 }
if {$WG_S ni {1 2} || $WG_CAW ni {22 23 24 25} || $WG_CAW + $WG_S > 26} {
  error "bd_wg.tcl: WG_S $WG_S, WG_CAW $WG_CAW: an 8 GB DIMM holds 2 x WG_S channels of 2^(WG_CAW + 7) bytes (WG_CAW 22 .. 26 - WG_S)"
}
set DIMM_GIB [expr {1 << ($WG_CAW - 23 + $WG_S)}]
set bd_vu9p [string match -nocase xcvu9p* [get_property PART [current_project]]]
puts "bd_wg: core_clk $CORE_HZ Hz (divide $CORE_DIV), acc_clk2x $CLK2X_HZ, aclk $ACLK_HZ; Gen3 x$PCIE_LANES, $AXI_BITS bits"

# ------------------------------------------------------------------ external ports
set pcie_refclk [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 pcie_refclk]
set_property CONFIG.FREQ_HZ 100000000 $pcie_refclk
create_bd_port -dir I -type rst pcie_perstn
set_property CONFIG.POLARITY ACTIVE_LOW [get_bd_ports pcie_perstn]
foreach p {core_clk acc_clk2x aclk xdma_aclk} { create_bd_port -dir O -type clk $p }
if {[info exists WG_NCARDS] && $WG_NCARDS == 2} {
  create_bd_port -dir O -type clk init_clk
  set_property CONFIG.FREQ_HZ 100000000 [get_bd_ports init_clk]
}
create_bd_port -dir O -type rst core_rstn
create_bd_port -dir O -type rst xdma_aresetn
create_bd_port -dir O temp_alarm
create_bd_port -dir O ot_alarm
create_bd_port -dir O -from 9 -to 0 device_temp
create_bd_port -dir O pcie_link_up
foreach d {0 1 2 3} {
  set c [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 c300_$d]
  set_property CONFIG.FREQ_HZ 300000000 $c
  create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddr4_rtl:1.0 ddr4_c$d
  create_bd_port -dir I -type rst mig_rst$d
  set_property CONFIG.POLARITY ACTIVE_HIGH [get_bd_ports mig_rst$d]
  create_bd_port -dir I -type rst sc_rstn$d
  set_property CONFIG.POLARITY ACTIVE_LOW [get_bd_ports sc_rstn$d]
  create_bd_port -dir O calib$d
}

set lites {M_AXI_CTL M_AXI_BMC M_AXI_GOV}
foreach p $lites {
  set m [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 $p]
  set_property -dict [list CONFIG.PROTOCOL AXI4LITE CONFIG.DATA_WIDTH 32 CONFIG.ADDR_WIDTH 32 \
    CONFIG.HAS_BURST 0 CONFIG.HAS_LOCK 0 CONFIG.HAS_PROT 0 CONFIG.HAS_CACHE 0 CONFIG.HAS_QOS 0 \
    CONFIG.HAS_REGION 0 CONFIG.FREQ_HZ $CORE_HZ] $m
}
set dmas {}
set hosts {}
set pairs {}
set hms {}
foreach d {0 1 2 3} {
  set m [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI_DMA$d]
  set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH $AXI_BITS CONFIG.ADDR_WIDTH 64 \
    CONFIG.HAS_BURST 0 CONFIG.HAS_LOCK 0 CONFIG.HAS_PROT 0 CONFIG.HAS_CACHE 0 \
    CONFIG.HAS_QOS 0 CONFIG.HAS_REGION 0 CONFIG.FREQ_HZ $XDMA_HZ] $m
  lappend dmas M_AXI_DMA$d
  # the slaves: AXI4 without the optional signals (the top ties AxSIZE / AxBURST: full-width INCR)
  set h [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_AXI_H$d]
  set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH $AXI_BITS CONFIG.ADDR_WIDTH 33 \
    CONFIG.ID_WIDTH 0 CONFIG.HAS_LOCK 0 CONFIG.HAS_PROT 0 CONFIG.HAS_CACHE 0 CONFIG.HAS_QOS 0 \
    CONFIG.HAS_REGION 0 CONFIG.FREQ_HZ $XDMA_HZ] $h
  lappend hosts S_AXI_H$d
  # sc_h<d>'s master: the host's DMA at aclk, full-width INCR (the hmux has no AxSIZE / AxBURST)
  set hm [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI_H$d]
  set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 512 CONFIG.ADDR_WIDTH 33 \
    CONFIG.ID_WIDTH 0 CONFIG.HAS_BURST 0 CONFIG.HAS_LOCK 0 CONFIG.HAS_PROT 0 CONFIG.HAS_CACHE 0 \
    CONFIG.HAS_QOS 0 CONFIG.HAS_REGION 0 CONFIG.SUPPORTS_NARROW_BURST 0 \
    CONFIG.FREQ_HZ $ACLK_HZ] $hm
  lappend hms M_AXI_H$d
  set p [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_AXI_P$d]
  set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 512 CONFIG.ADDR_WIDTH 33 \
    CONFIG.ID_WIDTH 4 CONFIG.HAS_LOCK 0 CONFIG.HAS_PROT 0 CONFIG.HAS_CACHE 0 CONFIG.HAS_QOS 0 \
    CONFIG.HAS_REGION 0 CONFIG.FREQ_HZ $ACLK_HZ] $p
  lappend pairs S_AXI_P$d
}
set_property CONFIG.ASSOCIATED_BUSIF [join $lites :] [get_bd_ports core_clk]
set_property CONFIG.ASSOCIATED_RESET {core_rstn} [get_bd_ports core_clk]
set_property CONFIG.ASSOCIATED_BUSIF [join [concat $dmas $hosts] :] [get_bd_ports xdma_aclk]
set_property CONFIG.ASSOCIATED_RESET {xdma_aresetn} [get_bd_ports xdma_aclk]
set_property CONFIG.ASSOCIATED_BUSIF [join [concat $pairs $hms] :] [get_bd_ports aclk]
set_property CONFIG.ASSOCIATED_RESET {sc_rstn0:sc_rstn1:sc_rstn2:sc_rstn3} [get_bd_ports aclk]
foreach {p f} [list core_clk $CORE_HZ acc_clk2x $CLK2X_HZ aclk $ACLK_HZ xdma_aclk $XDMA_HZ] {
  set_property CONFIG.FREQ_HZ $f [get_bd_ports $p]
}

# ------------------------------------------------------------------ clocks and resets
# clk300_1: one IBUFDS for DIMM C1's MIG (No_Buffer) and the clock wizard (through a BUFG)
set rb [create_bd_cell -type ip -vlnv [ip_vlnv util_ds_buf] refbuf_c1]
set_property CONFIG.C_BUF_TYPE {IBUFDS} $rb
connect_bd_intf_net [get_bd_intf_ports c300_1] [get_bd_intf_pins refbuf_c1/CLK_IN_D]
set gb [create_bd_cell -type ip -vlnv [ip_vlnv util_ds_buf] refbuf_g]
set_property CONFIG.C_BUF_TYPE {BUFG} $gb
connect_bd_net [get_bd_pins refbuf_c1/IBUF_OUT] [get_bd_pins refbuf_g/BUFG_I]

set clk [create_bd_cell -type ip -vlnv [ip_vlnv clk_wiz] clk_wiz_0]
set_property -dict [list \
  CONFIG.PRIM_IN_FREQ {300.000} CONFIG.PRIM_SOURCE {Global_buffer} \
  CONFIG.USE_RESET {false} CONFIG.USE_LOCKED {true} CONFIG.NUM_OUT_CLKS {3} \
  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ [expr {$CLK2X_HZ / 1.0e6}] CONFIG.CLK_OUT1_PORT {acc_clk2x} \
  CONFIG.CLKOUT1_DRIVES {No_buffer} \
  CONFIG.CLKOUT2_USED {true} CONFIG.CLKOUT2_REQUESTED_OUT_FREQ [expr {$CORE_HZ / 1.0e6}] \
  CONFIG.CLKOUT2_REQUESTED_PHASE {0.000} CONFIG.CLK_OUT2_PORT {core_clk} \
  CONFIG.CLKOUT3_USED {true} CONFIG.CLKOUT3_REQUESTED_OUT_FREQ {300.000} CONFIG.CLK_OUT3_PORT {aclk} \
] $clk
if {[info exists WG_NCARDS] && $WG_NCARDS == 2} {
  set_property -dict [list CONFIG.NUM_OUT_CLKS {4} CONFIG.CLKOUT4_USED {true} \
    CONFIG.CLKOUT4_REQUESTED_OUT_FREQ {100.000} CONFIG.CLK_OUT4_PORT {init_clk}] $clk
}
connect_bd_net [get_bd_pins refbuf_g/BUFG_O] [get_bd_pins clk_wiz_0/clk_in1]
# the VCO and the divides as above, whatever the wizard would pick (always: clk133.tcl takes
# exactly these, CLKOUT0 the 2x at CORE_DIV / 2, CLKOUT1 core_clk at CORE_DIV, CLKOUT2 aclk at 4)
set want [dict create core_clk $CORE_HZ acc_clk2x $CLK2X_HZ aclk $ACLK_HZ]
if {[info exists WG_NCARDS] && $WG_NCARDS == 2} { dict set want init_clk 100000000 }
set_property -dict [list CONFIG.OVERRIDE_MMCM {true} CONFIG.MMCM_DIVCLK_DIVIDE {1} \
  CONFIG.MMCM_CLKFBOUT_MULT_F [format %.3f [expr {$VCO / 300.0}]] \
  CONFIG.MMCM_CLKOUT0_DIVIDE_F [format %.3f [expr {$CORE_DIV / 2.0}]] \
  CONFIG.MMCM_CLKOUT1_DIVIDE $CORE_DIV CONFIG.MMCM_CLKOUT2_DIVIDE 4] $clk
if {[dict exists $want init_clk]} { set_property CONFIG.MMCM_CLKOUT3_DIVIDE 12 $clk }
dict for {p f} $want {
  set got [get_property CONFIG.FREQ_HZ [get_bd_pins clk_wiz_0/$p]]
  if {$got != $f} { error "bd_wg.tcl: the clock wizard makes $p at $got Hz, not $f" }
  connect_bd_net [get_bd_pins clk_wiz_0/$p] [get_bd_ports $p]
}
if {[get_bd_pins -quiet clk_wiz_0/acc_clk2x_ce] ne ""} { error "bd_wg.tcl: acc_clk2x buffered" }

# core reset: until the MMCM locks, and while the host asserts PCIe PERST#
set rst_core [create_bd_cell -type ip -vlnv [ip_vlnv proc_sys_reset] rst_core]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins rst_core/slowest_sync_clk]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins rst_core/dcm_locked]
connect_bd_net [get_bd_ports pcie_perstn] [get_bd_pins rst_core/ext_reset_in]
connect_bd_net [get_bd_pins rst_core/peripheral_aresetn] [get_bd_ports core_rstn]

# ------------------------------------------------------------------ PCIe: XDMA, Gen3 x16 (x8)
# As bd.tcl: the YPCB's identity (device 7028, class 1200, subsystem 10EE:4F54) and its H2C setting
# (xdma_rnum_rids 8), GTY quads 224-227 (select_quad 227)
set ibuf [create_bd_cell -type ip -vlnv [ip_vlnv util_ds_buf] refclk_buf]
set_property CONFIG.C_BUF_TYPE {IBUFDSGTE} $ibuf
connect_bd_intf_net $pcie_refclk [get_bd_intf_pins refclk_buf/CLK_IN_D]
set xdma [create_bd_cell -type ip -vlnv [ip_vlnv xdma] xdma_0]
set cfg [list \
  CONFIG.mode_selection {Advanced} \
  CONFIG.pl_link_cap_max_link_width "X$PCIE_LANES" \
  CONFIG.pl_link_cap_max_link_speed {8.0_GT/s} \
  CONFIG.axi_data_width "${AXI_BITS}_bit" \
  CONFIG.axisten_freq $XDMA_MHZ \
  CONFIG.ref_clk_freq {100_MHz} \
  CONFIG.pf0_device_id {7028} \
  CONFIG.pf0_class_code_base {12} CONFIG.pf0_class_code_sub {00} \
  CONFIG.pf0_class_code_interface {00} \
  CONFIG.pf0_subsystem_vendor_id {10EE} CONFIG.pf0_subsystem_id {4F54} \
  CONFIG.pf0_revision_id {01} \
  CONFIG.xdma_rnum_chnl {1} CONFIG.xdma_wnum_chnl {1} CONFIG.xdma_rnum_rids {8} \
  CONFIG.axilite_master_en {true} CONFIG.axilite_master_size {1} \
  CONFIG.axilite_master_scale {Megabytes} \
  CONFIG.pciebar2axibar_axil_master {0x00000000} \
  CONFIG.xdma_axi_intf_mm {AXI_Memory_Mapped} \
  CONFIG.plltype {QPLL1} \
]
if {$bd_vu9p} { lappend cfg CONFIG.en_gt_selection {true} CONFIG.select_quad {GTY_Quad_227} }
set_property -dict $cfg $xdma
connect_bd_net [get_bd_pins refclk_buf/IBUF_DS_ODIV2] [get_bd_pins xdma_0/sys_clk]
connect_bd_net [get_bd_pins refclk_buf/IBUF_OUT] [get_bd_pins xdma_0/sys_clk_gt]
connect_bd_net [get_bd_ports pcie_perstn] [get_bd_pins xdma_0/sys_rst_n]
make_bd_intf_pins_external [get_bd_intf_pins xdma_0/pcie_mgt]
set_property NAME pcie_mgt [get_bd_intf_ports -of [get_bd_intf_nets -of [get_bd_intf_pins xdma_0/pcie_mgt]]]
connect_bd_net [get_bd_pins xdma_0/user_lnk_up] [get_bd_ports pcie_link_up]
connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_ports xdma_aclk]
connect_bd_net [get_bd_pins xdma_0/axi_aresetn] [get_bd_ports xdma_aresetn]

# XDMA's DMA (and jtag_axi_1's) onto the four DIMMs, in xdma_aclk, at XDMA's width
set scd [create_bd_cell -type ip -vlnv [ip_vlnv smartconnect] sc_dma]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {4} CONFIG.NUM_CLKS {1}] $scd
connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_pins sc_dma/aclk]
connect_bd_net [get_bd_pins xdma_0/axi_aresetn] [get_bd_pins sc_dma/aresetn]
connect_bd_intf_net [get_bd_intf_pins xdma_0/M_AXI] [get_bd_intf_pins sc_dma/S00_AXI]
set jtd [create_bd_cell -type ip -vlnv [ip_vlnv jtag_axi] jtag_axi_1]
set_property -dict [list CONFIG.PROTOCOL {0} CONFIG.M_AXI_ADDR_WIDTH {64} \
  CONFIG.M_AXI_DATA_WIDTH {32}] $jtd
connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_pins jtag_axi_1/aclk]
connect_bd_net [get_bd_pins xdma_0/axi_aresetn] [get_bd_pins jtag_axi_1/aresetn]
connect_bd_intf_net [get_bd_intf_pins jtag_axi_1/M_AXI] [get_bd_intf_pins sc_dma/S01_AXI]
foreach d {0 1 2 3} {
  connect_bd_intf_net [get_bd_intf_pins sc_dma/M0${d}_AXI] [get_bd_intf_ports M_AXI_DMA$d]
}

# ------------------------------------------------------------------ the DIMMs: MIG, clock converter
# DDR4-2400 (tCK 833 ps; the board clock 300 MHz, the IP's 3332 ps input), the GOODRAM
# GR2400D464L17S/8G UDIMMs as Micron's MTA8ATF1G64AZ-2G3 (1 rank, x8, 8 GB), DM without DBI, no
# ECC, CL 17 / CWL 12; AXI4 512 bits (one 64-byte burst a beat), ROW_COLUMN_BANK (the banks below
# the column: a stream's bursts rotate over the bank groups); the DIMMs' pins on the top's
# constraints (bcu1525_wg_pins.xdc). Probed on C0's pins on its own (the IP's example design,
# placed and routed clean, 2026-10-27). AxiAddressWidth 33: the DIMM's 8 GB, both channels at WG_CAW 25
# (16 GB DIMMs would take a 2-rank or x4 part and 34 bits: later work).
set MIG_CFG [list \
  CONFIG.C0.DDR4_TimePeriod 833 CONFIG.C0.DDR4_InputClockPeriod 3332 \
  CONFIG.C0.DDR4_MemoryType UDIMMs CONFIG.C0.DDR4_MemoryPart MTA8ATF1G64AZ-2G3 \
  CONFIG.C0.DDR4_DataWidth 64 CONFIG.C0.DDR4_DataMask DM_NO_DBI CONFIG.C0.DDR4_Ecc false \
  CONFIG.C0.DDR4_CasLatency 17 CONFIG.C0.DDR4_CasWriteLatency 12 \
  CONFIG.C0.DDR4_AxiSelection true CONFIG.C0.DDR4_AxiDataWidth 512 \
  CONFIG.C0.DDR4_AxiAddressWidth 33 CONFIG.C0.DDR4_AxiIDWidth 4 \
  CONFIG.C0.DDR4_Mem_Add_Map ROW_COLUMN_BANK]
# OTPU_WG_MIGRD s62 (run_vivado.sh WG_MIGRD, vivado/mig_map.tcl): ORDERING Strict, with the read
# channel's FIFOs deeper, for more reads in flight than the user interface's read buffer's 32
set MIG_ORD [expr {[info exists ::env(OTPU_WG_MIGRD)] && $::env(OTPU_WG_MIGRD) eq "s62" ? "Strict" : "Normal"}]
lappend MIG_CFG CONFIG.C0.DDR4_Ordering $MIG_ORD
# OTPU_WG_HRS (run_vivado.sh WG_HRS; unset, empty or 0: off; 1: C0 and C3, the DIMMs off XDMA's
# SLR1; or DIMM numbers): DIMM d's host DMA port through rs_h<d>, an AXI register slice in its
# Multi_SLR_Crossing mode (15, one crossing: each channel's boundary flops USER_SLL_REG, on the
# Laguna registers), between sc_h<d>'s M00 (beside XDMA) and M_AXI_H<d> (the top's hmux, beside its
# MIG). Screen 1 (b1fp1cdg, 2026-11-04): its worst path, -0.863 ns, was hmux[0]'s R ring through a
# LUT into the SLL and sc_h0's R FIFO LUT RAM; the host's DMA only, off the accelerator's stream.
proc hrs_dimms {spec} {
  if {$spec in {"" 0}} { return {} }
  if {$spec eq "1"} { return {0 3} }
  set ds [lsort -unique [regexp -all -inline {\S+} $spec]]
  foreach d $ds { if {$d ni {0 1 2 3}} { error "bd_wg: OTPU_WG_HRS '$spec' (1 or DIMM numbers 0-3)" } }
  return $ds
}
set HRS [hrs_dimms [expr {[info exists ::env(OTPU_WG_HRS)] ? $::env(OTPU_WG_HRS) : ""}]]
# OTPU_WG_PRS (run_vivado.sh WG_PRS; unset or empty: on at WG_S 2, off at 1; 0 off, 1 on): each
# DIMM's port S_AXI_P<d> (the top's hmux) through rs_p<d>, an AXI register slice at aclk with AW,
# AR, R and B fully registered (1) and W straight (0: B1r's paths are R's), into cc_mem<d> (B1r:
# hmux[0]'s R ring full flag into cc_mem0's R FIFO, -0.39 ns at 300 MHz, ~97 % route). One cycle
# more each way on AW, AR, R and B.
proc prs_on {spec s} {
  if {$spec eq ""} { return [expr {$s == 2}] }
  if {$spec ni {0 1}} { error "bd_wg: OTPU_WG_PRS '$spec' (0 or 1; unset: on at WG_S 2)" }
  return $spec
}
set PRS [prs_on [expr {[info exists ::env(OTPU_WG_PRS)] ? $::env(OTPU_WG_PRS) : ""}] $WG_S]
foreach d {0 1 2 3} {
  set mig [create_bd_cell -type ip -vlnv [ip_vlnv ddr4] ddr4_$d]
  set_property -dict [concat $MIG_CFG [list CONFIG.System_Clock \
    [expr {$d == 1 ? "No_Buffer" : "Differential"}]]] $mig
  foreach {k want} [list C0.DDR4_TimePeriod 833 C0.DDR4_MemoryType UDIMMs C0.DDR4_AxiDataWidth 512 \
                      C0.DDR4_DataMask DM_NO_DBI C0.DDR4_Mem_Add_Map ROW_COLUMN_BANK \
                      C0.DDR4_Ordering $MIG_ORD] {
    set got [get_property CONFIG.$k $mig]
    if {$got ne $want} { error "ddr4_$d: $k is $got, expected $want" }
  }
  connect_bd_intf_net [get_bd_intf_pins ddr4_$d/C0_DDR4] [get_bd_intf_ports ddr4_c$d]
  if {$d == 1} {
    connect_bd_net [get_bd_pins refbuf_c1/IBUF_OUT] [get_bd_pins ddr4_$d/c0_sys_clk_i]
  } else {
    connect_bd_intf_net [get_bd_intf_ports c300_$d] [get_bd_intf_pins ddr4_$d/C0_SYS_CLK]
  }
  connect_bd_net [get_bd_ports mig_rst$d] [get_bd_pins ddr4_$d/sys_rst]
  connect_bd_net [get_bd_pins ddr4_$d/c0_init_calib_complete] [get_bd_ports calib$d]
  # its AXI reset from its ui_clk's
  # (its ext_reset_in active high from ui_clk_sync_rst's polarity: read-only, checked below)
  create_bd_cell -type ip -vlnv [ip_vlnv proc_sys_reset] rst_ui$d
  connect_bd_net [get_bd_pins ddr4_$d/c0_ddr4_ui_clk] [get_bd_pins rst_ui$d/slowest_sync_clk]
  connect_bd_net [get_bd_pins ddr4_$d/c0_ddr4_ui_clk_sync_rst] [get_bd_pins rst_ui$d/ext_reset_in]
  connect_bd_net [get_bd_pins rst_ui$d/peripheral_aresetn] [get_bd_pins ddr4_$d/c0_ddr4_aresetn]
  # the DIMM's port (the top's hmux: its pair and the host) across into the MIG's ui_clk, a beat a
  # cycle. Its slave side in reset with sc_rstn<d>; its master side through rst_cc<d> (ui_clk):
  # sc_rstn<d> (active low) or the MIG's ui_clk_sync_rst (active high, read-only: checked below)
  set cc [create_bd_cell -type ip -vlnv [ip_vlnv axi_clock_converter] cc_mem$d]
  set_property -dict [list CONFIG.PROTOCOL {AXI4} CONFIG.DATA_WIDTH {512} CONFIG.ADDR_WIDTH {33} \
    CONFIG.ID_WIDTH {4} CONFIG.ACLK_ASYNC {1} CONFIG.SYNCHRONIZATION_STAGES {3}] $cc
  create_bd_cell -type ip -vlnv [ip_vlnv proc_sys_reset] rst_cc$d
  connect_bd_net [get_bd_pins ddr4_$d/c0_ddr4_ui_clk] [get_bd_pins rst_cc$d/slowest_sync_clk]
  connect_bd_net [get_bd_ports sc_rstn$d] [get_bd_pins rst_cc$d/ext_reset_in]
  connect_bd_net [get_bd_pins ddr4_$d/c0_ddr4_ui_clk_sync_rst] [get_bd_pins rst_cc$d/aux_reset_in]
  connect_bd_net [get_bd_pins clk_wiz_0/aclk] [get_bd_pins cc_mem$d/s_axi_aclk]
  connect_bd_net [get_bd_ports sc_rstn$d] [get_bd_pins cc_mem$d/s_axi_aresetn]
  connect_bd_net [get_bd_pins ddr4_$d/c0_ddr4_ui_clk] [get_bd_pins cc_mem$d/m_axi_aclk]
  connect_bd_net [get_bd_pins rst_cc$d/peripheral_aresetn] [get_bd_pins cc_mem$d/m_axi_aresetn]
  if {$PRS} {
    set rs [create_bd_cell -type ip -vlnv [ip_vlnv axi_register_slice] rs_p$d]
    set_property -dict [list CONFIG.PROTOCOL {AXI4} CONFIG.DATA_WIDTH {512} CONFIG.ADDR_WIDTH {33} \
      CONFIG.ID_WIDTH {4} CONFIG.REG_AW {1} CONFIG.REG_AR {1} CONFIG.REG_W {0} CONFIG.REG_R {1} \
      CONFIG.REG_B {1}] $rs
    foreach {k want} {REG_AW 1 REG_AR 1 REG_W 0 REG_R 1 REG_B 1 ID_WIDTH 4} {
      set got [get_property CONFIG.$k $rs]
      if {$got ne $want} { error "rs_p$d: $k is $got, expected $want" }
    }
    connect_bd_net [get_bd_pins clk_wiz_0/aclk] [get_bd_pins rs_p$d/aclk]
    connect_bd_net [get_bd_ports sc_rstn$d] [get_bd_pins rs_p$d/aresetn]
    connect_bd_intf_net [get_bd_intf_ports S_AXI_P$d] [get_bd_intf_pins rs_p$d/S_AXI]
    connect_bd_intf_net [get_bd_intf_pins rs_p$d/M_AXI] [get_bd_intf_pins cc_mem$d/S_AXI]
    puts "bd_wg: rs_p$d (OTPU_WG_PRS): S_AXI_P$d -> register slice (AW AR R B full, W straight) -> cc_mem$d"
  } else {
    connect_bd_intf_net [get_bd_intf_ports S_AXI_P$d] [get_bd_intf_pins cc_mem$d/S_AXI]
  }
  connect_bd_intf_net [get_bd_intf_pins cc_mem$d/M_AXI] [get_bd_intf_pins ddr4_$d/C0_DDR4_S_AXI]
  # the host's DMA (xdma_aclk, XDMA's width) out at aclk, 512 bits, to the top's hmux
  set sch [create_bd_cell -type ip -vlnv [ip_vlnv smartconnect] sc_h$d]
  set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1} CONFIG.NUM_CLKS {2}] $sch
  connect_bd_net [get_bd_pins clk_wiz_0/aclk] [get_bd_pins sc_h$d/aclk]
  connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_pins sc_h$d/aclk1]
  connect_bd_net [get_bd_ports sc_rstn$d] [get_bd_pins sc_h$d/aresetn]
  connect_bd_intf_net [get_bd_intf_ports S_AXI_H$d] [get_bd_intf_pins sc_h$d/S00_AXI]
  if {$d in $HRS} {
    set rs [create_bd_cell -type ip -vlnv [ip_vlnv axi_register_slice] rs_h$d]
    set_property -dict [list CONFIG.PROTOCOL {AXI4} CONFIG.DATA_WIDTH {512} CONFIG.ADDR_WIDTH {33} \
      CONFIG.ID_WIDTH {0} CONFIG.REG_AW {15} CONFIG.REG_AR {15} CONFIG.REG_W {15} CONFIG.REG_R {15} \
      CONFIG.REG_B {15} CONFIG.NUM_SLR_CROSSINGS {1}] $rs
    foreach {k want} {REG_AW 15 REG_AR 15 REG_W 15 REG_R 15 REG_B 15 NUM_SLR_CROSSINGS 1} {
      set got [get_property CONFIG.$k $rs]
      if {$got ne $want} { error "rs_h$d: $k is $got, expected $want" }
    }
    connect_bd_net [get_bd_pins clk_wiz_0/aclk] [get_bd_pins rs_h$d/aclk]
    connect_bd_net [get_bd_ports sc_rstn$d] [get_bd_pins rs_h$d/aresetn]
    connect_bd_intf_net [get_bd_intf_pins sc_h$d/M00_AXI] [get_bd_intf_pins rs_h$d/S_AXI]
    connect_bd_intf_net [get_bd_intf_pins rs_h$d/M_AXI] [get_bd_intf_ports M_AXI_H$d]
    puts "bd_wg: rs_h$d (OTPU_WG_HRS): sc_h$d/M00_AXI -> Multi_SLR_Crossing register slice -> M_AXI_H$d"
  } else {
    connect_bd_intf_net [get_bd_intf_pins sc_h$d/M00_AXI] [get_bd_intf_ports M_AXI_H$d]
  }
}

# ------------------------------------------------------------------ SYSMON (die temperature)
# as bd.tcl: the temperature alarm at 80 C (released 70), OT at 84 C (released 65)
set sysmon [create_bd_cell -type ip -vlnv [ip_vlnv system_management_wiz] sysmon]
set OT_UP 84.0
set OT_DOWN 65.0
set ALM_UP 80.0
set ALM_DOWN 70.0
set_property -dict [list CONFIG.INTERFACE_SELECTION {Enable_AXI} \
  CONFIG.DCLK_FREQUENCY [expr {int(ceil($CORE_HZ / 1.0e6))}] \
  CONFIG.SEQUENCER_MODE {Continuous} CONFIG.CHANNEL_ENABLE_VP_VN {false} \
  CONFIG.ENABLE_TEMP_BUS {true} \
  CONFIG.OT_ALARM {true} CONFIG.TEMPERATURE_ALARM_OT_TRIGGER $OT_UP \
  CONFIG.TEMPERATURE_ALARM_OT_RESET $OT_DOWN \
  CONFIG.USER_TEMP_ALARM {true} CONFIG.TEMPERATURE_ALARM_TRIGGER $ALM_UP \
  CONFIG.TEMPERATURE_ALARM_RESET $ALM_DOWN \
  CONFIG.VCCINT_ALARM {false} CONFIG.VCCAUX_ALARM {false}] $sysmon
foreach {k want} [list OT_ALARM true TEMPERATURE_ALARM_OT_TRIGGER $OT_UP \
                       TEMPERATURE_ALARM_OT_RESET $OT_DOWN USER_TEMP_ALARM true \
                       TEMPERATURE_ALARM_TRIGGER $ALM_UP TEMPERATURE_ALARM_RESET $ALM_DOWN] {
  set got [get_property CONFIG.$k $sysmon]
  if {$got != $want} { error "sysmon: $k is $got, expected $want" }
}
foreach {pin port} {ot_out ot_alarm user_temp_alarm_out temp_alarm} {
  if {[get_bd_pins -quiet sysmon/$pin] eq ""} {
    error "sysmon: no pin $pin (its pins: [get_bd_pins -of [get_bd_cells sysmon]])"
  }
  connect_bd_net [get_bd_pins sysmon/$pin] [get_bd_ports $port]
}
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins sysmon/s_axi_aclk]
connect_bd_net [get_bd_pins rst_core/peripheral_aresetn] [get_bd_pins sysmon/s_axi_aresetn]
connect_bd_net [get_bd_pins sysmon/temp_out] [get_bd_ports device_temp]

# ------------------------------------------------------------------ control interconnect
set scl [create_bd_cell -type ip -vlnv [ip_vlnv smartconnect] sc_ctl]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {4} CONFIG.NUM_CLKS {2}] $scl
connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_pins sc_ctl/aclk]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins sc_ctl/aclk1]
connect_bd_net [get_bd_pins rst_core/peripheral_aresetn] [get_bd_pins sc_ctl/aresetn]
connect_bd_intf_net [get_bd_intf_pins xdma_0/M_AXI_LITE] [get_bd_intf_pins sc_ctl/S00_AXI]
set jtl [create_bd_cell -type ip -vlnv [ip_vlnv jtag_axi] jtag_axi_0]
set_property -dict [list CONFIG.PROTOCOL {2} CONFIG.M_AXI_ADDR_WIDTH {32} \
  CONFIG.M_AXI_DATA_WIDTH {32}] $jtl
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins jtag_axi_0/aclk]
connect_bd_net [get_bd_pins rst_core/peripheral_aresetn] [get_bd_pins jtag_axi_0/aresetn]
connect_bd_intf_net [get_bd_intf_pins jtag_axi_0/M_AXI] [get_bd_intf_pins sc_ctl/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M00_AXI] [get_bd_intf_ports M_AXI_CTL]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M01_AXI] [get_bd_intf_pins sysmon/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M02_AXI] [get_bd_intf_ports M_AXI_BMC]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M03_AXI] [get_bd_intf_ports M_AXI_GOV]

# ------------------------------------------------------------------ address map
# BAR0: 0x00000 the control registers, 0x10000 the governor, 0x20000 the BMC's UART bridge,
# 0x30000 the SYSMON. The card's DMA addresses: DIMM d at d x DIMM_GIB GiB (M_AXI_DMA<d>; the top
# permutes the offset into the DIMM: virtual channel 2d + s at (2d + s) x DIMM_GIB / 2 GiB, 1 GiB
# windows at WG_CAW 22, 8 GiB at 25: the whole of the four 8 GB DIMMs). Each DIMM's port
# (S_AXI_P<d>, through cc_mem<d>): the MIG's 8 GB at 0; its host's DMA (S_AXI_H<d>, sc_h<d>'s):
# M_AXI_H<d>'s 8 GB at 0.
foreach lite [list [get_bd_addr_spaces xdma_0/M_AXI_LITE] [get_bd_addr_spaces jtag_axi_0/Data]] {
  assign_bd_address -offset 0x00000000 -range 64K -target_address_space $lite [get_bd_addr_segs M_AXI_CTL/Reg] -force
  assign_bd_address -offset 0x00010000 -range 64K -target_address_space $lite [get_bd_addr_segs M_AXI_GOV/Reg] -force
  assign_bd_address -offset 0x00020000 -range 64K -target_address_space $lite [get_bd_addr_segs M_AXI_BMC/Reg] -force
  assign_bd_address -offset 0x00030000 -range 64K -target_address_space $lite \
    [get_bd_addr_segs -of [get_bd_intf_pins sysmon/S_AXI_LITE]] -force
}
foreach sp [list [get_bd_addr_spaces xdma_0/M_AXI] [get_bd_addr_spaces jtag_axi_1/Data]] {
  foreach d {0 1 2 3} {
    assign_bd_address -offset [format 0x%09X [expr {$d * $DIMM_GIB << 30}]] -range ${DIMM_GIB}G \
      -target_address_space $sp [get_bd_addr_segs M_AXI_DMA$d/Reg] -force
  }
}
foreach d {0 1 2 3} {
  set seg [get_bd_addr_segs -of [get_bd_intf_pins ddr4_$d/C0_DDR4_S_AXI]]
  set seg [lsearch -all -inline $seg *ADDRESS_BLOCK*]
  if {[llength $seg] != 1} { error "ddr4_$d: its memory segment is '$seg'" }
  assign_bd_address -offset 0x0 -range 8G -target_address_space [get_bd_addr_spaces S_AXI_P$d] \
    $seg -force
  assign_bd_address -offset 0x0 -range 8G -target_address_space [get_bd_addr_spaces S_AXI_H$d] \
    [get_bd_addr_segs M_AXI_H$d/Reg] -force
}

validate_bd_design

foreach {cell want} {rst_core 0 rst_ui0 1 rst_ui1 1 rst_ui2 1 rst_ui3 1} {
  set got [get_property CONFIG.C_EXT_RESET_HIGH [get_bd_cells $cell]]
  if {$got != $want} { error "$cell: C_EXT_RESET_HIGH is $got, expected $want" }
}
foreach d {0 1 2 3} {
  foreach {k want} {C_EXT_RESET_HIGH 0 C_AUX_RESET_HIGH 1} {
    set got [get_property CONFIG.$k [get_bd_cells rst_cc$d]]
    if {$got != $want} { error "rst_cc$d: $k is $got, expected $want" }
  }
}
foreach p [concat {pcie_mgt pcie_refclk} $dmas $hosts $hms $pairs $lites \
                  {ddr4_c0 ddr4_c1 ddr4_c2 ddr4_c3 c300_0 c300_1 c300_2 c300_3}] {
  if {[get_bd_intf_ports $p] eq ""} { error "block design port $p missing: [get_bd_intf_ports]" }
}
foreach {k want} [list pf0_device_id 7028 pf0_class_code 120000 pf0_subsystem_id 4F54 \
                       pf0_revision_id 01 axisten_freq $XDMA_MHZ \
                       pl_link_cap_max_link_width X$PCIE_LANES axi_data_width ${AXI_BITS}_bit] {
  set got [get_property CONFIG.$k [get_bd_cells xdma_0]]
  if {$got ne $want} { error "xdma_0: $k is $got, expected $want" }
}
foreach p $dmas {
  set w [get_property CONFIG.DATA_WIDTH [get_bd_intf_ports $p]]
  set id [get_property CONFIG.ID_WIDTH [get_bd_intf_ports $p]]
  if {$w != $AXI_BITS || $id != 0} { error "$p: $w bits, $id ID bits (expected $AXI_BITS, none)" }
}
foreach p $hms {
  set w [get_property CONFIG.DATA_WIDTH [get_bd_intf_ports $p]]
  set id [get_property CONFIG.ID_WIDTH [get_bd_intf_ports $p]]
  if {$w != 512 || $id != 0} { error "$p: $w bits, $id ID bits (expected 512, none)" }
}
# OTPU_WG_HRS: rs_h<d>'s S_AXI must carry M_AXI_H<d>'s SUPPORTS_NARROW_BURST 0 back to sc_h<d>, so at
# 256 bits its upsizer still sends only whole 512-bit beats (the hmux has no AxSIZE)
foreach d $HRS {
  set want [get_property CONFIG.SUPPORTS_NARROW_BURST [get_bd_intf_ports M_AXI_H$d]]
  set got [get_property CONFIG.SUPPORTS_NARROW_BURST [get_bd_intf_pins rs_h$d/S_AXI]]
  puts "bd_wg: rs_h$d/S_AXI SUPPORTS_NARROW_BURST '$got' (M_AXI_H$d '$want')"
  if {$AXI_BITS < 512 && $got ne $want} {
    error "rs_h$d/S_AXI: SUPPORTS_NARROW_BURST '$got', M_AXI_H$d's '$want': sc_h$d could send narrow beats at $AXI_BITS bits"
  }
}
puts "bd_wg: four DIMMs (MIG DDR4-2400, AXI4 512), M_AXI_DMA0..3 $AXI_BITS bits, M_AXI_H0..3 512"
save_bd_design
