# Block design "otpu_bd" for the YPCB-00338: PCIe (XDMA), two DDR3 controllers (MIG), clocks,
# resets and the interconnect. The accelerator (otpu_board) sits outside the block design, in
# otpu_fpga_top.sv, on these external interfaces (all on core_clk):
#   M_AXI_CTL  AXI4-Lite master  -> the control registers (host BAR0 offset 0)
#   S_AXI_M0   AXI4 slave, 512b  <- the accelerator's channel-0 master
#   S_AXI_M1   AXI4 slave, 512b  <- the accelerator's channel-1 master
# and two status ports in other clock domains, synchronized by the accelerator: calib (the
# MIGs' calibration flags) and device_temp (MIG channel 0's XADC die-temperature code, 12 bits,
# its ui_clk domain).
# Address map (every master): MIG0 at 0x0000_0000, MIG1 at 0x8000_0000, 2 GiB each -- the map
# of host/board.py and rtl/mem/otpu_axi_dram.sv.
#
# Clock plan (50 MHz board oscillator, AA28 -> MMCM, VCO 800 MHz):
#   core_clk   100.000 MHz  accelerator, control, interconnect core side (CORE_MHZ: 800 / D,
#                           D a multiple of 1/8, e.g. 80 / 75.3 / 66.7 as a timing fallback)
#   clk_200    200.000 MHz  MIG reference (IDELAYCTRL) and, for DDR3-800, MIG system clock
#   clk_267    266.667 MHz  MIG system clock for DDR3-1066 (DDR_SPEED=1066)
#   ui_clk0/1  100 / 133 MHz  MIG user clocks (4:1 of 400 / 533 MHz)
#   axi_aclk   250 MHz      XDMA (Gen2 x8, 128-bit)
#
# Variables (set before sourcing): DDR_SPEED (800 | 1066), MIG_DIR (dir of the .prj files).

if {![info exists DDR_SPEED]} { set DDR_SPEED 800 }
if {![info exists MIG_DIR]} { set MIG_DIR [file normalize [file dirname [info script]]/mig] }

# the newest installed version of an IP
proc ip_vlnv {name} {
  set defs [lsort -decreasing [get_ipdefs -all "xilinx.com:ip:${name}:*"]]
  if {[llength $defs] == 0} { error "IP $name not found in this Vivado installation" }
  return [lindex $defs 0]
}

create_bd_design otpu_bd
current_bd_design otpu_bd

# core clock: CORE_MHZ (from create_project.tcl) rounded to the MMCM's 1/8 divider steps
if {![info exists CORE_MHZ]} { set CORE_MHZ 100 }
set CORE_DIV [expr {round(800.0 / $CORE_MHZ * 8) / 8.0}]
set CORE_MHZ_ACT [format %.3f [expr {800.0 / $CORE_DIV}]]
set CORE_HZ [expr {round(800.0e6 / $CORE_DIV)}]
puts "core_clk: $CORE_MHZ_ACT MHz (MMCM divide $CORE_DIV)"

# ------------------------------------------------------------------ external ports
create_bd_port -dir I -type clk -freq_hz 50000000 sys_clk_50
set pcie_refclk [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 pcie_refclk]
set_property CONFIG.FREQ_HZ 100000000 $pcie_refclk
create_bd_port -dir I -type rst pcie_perstn
set_property CONFIG.POLARITY ACTIVE_LOW [get_bd_ports pcie_perstn]
create_bd_port -dir O -type clk core_clk
create_bd_port -dir O -type rst core_rstn
create_bd_port -dir O -from 1 -to 0 calib
create_bd_port -dir O -from 11 -to 0 device_temp
create_bd_port -dir O pcie_link_up

set m_ctl [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI_CTL]
set_property -dict [list CONFIG.PROTOCOL AXI4LITE CONFIG.DATA_WIDTH 32 CONFIG.ADDR_WIDTH 32 \
  CONFIG.HAS_BURST 0 CONFIG.HAS_LOCK 0 CONFIG.HAS_PROT 0 CONFIG.HAS_CACHE 0 CONFIG.HAS_QOS 0 \
  CONFIG.HAS_REGION 0] $m_ctl
foreach p {S_AXI_M0 S_AXI_M1} {
  set s [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 $p]
  set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 512 CONFIG.ADDR_WIDTH 32 \
    CONFIG.ID_WIDTH 1 CONFIG.HAS_REGION 0 CONFIG.NUM_READ_OUTSTANDING 64 \
    CONFIG.NUM_WRITE_OUTSTANDING 16 CONFIG.MAX_BURST_LENGTH 1 CONFIG.FREQ_HZ $CORE_HZ] $s
}
set_property CONFIG.ASSOCIATED_BUSIF {M_AXI_CTL:S_AXI_M0:S_AXI_M1} [get_bd_ports core_clk]
set_property CONFIG.ASSOCIATED_RESET {core_rstn} [get_bd_ports core_clk]
set_property CONFIG.FREQ_HZ $CORE_HZ [get_bd_ports core_clk]

# ------------------------------------------------------------------ clocks and resets
# The board reset pin (R28) is not wired: resets come from the MMCM lock (and PCIe PERST#).
set clk [create_bd_cell -type ip -vlnv [ip_vlnv clk_wiz] clk_wiz_0]
set_property -dict [list \
  CONFIG.PRIM_IN_FREQ {50.000} CONFIG.PRIM_SOURCE {Single_ended_clock_capable_pin} \
  CONFIG.USE_RESET {false} CONFIG.USE_LOCKED {true} \
  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $CORE_MHZ_ACT \
  CONFIG.CLKOUT2_USED {true} CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {200.000} \
  CONFIG.CLKOUT3_USED {true} CONFIG.CLKOUT3_REQUESTED_OUT_FREQ {266.667} \
  CONFIG.NUM_OUT_CLKS {3} CONFIG.MMCM_DIVCLK_DIVIDE {1} CONFIG.MMCM_CLKFBOUT_MULT_F {16.000} \
  CONFIG.MMCM_CLKOUT0_DIVIDE_F $CORE_DIV CONFIG.MMCM_CLKOUT1_DIVIDE {4} \
  CONFIG.MMCM_CLKOUT2_DIVIDE {3} \
  CONFIG.CLK_OUT1_PORT {core_clk} CONFIG.CLK_OUT2_PORT {clk_200} CONFIG.CLK_OUT3_PORT {clk_267} \
] $clk
connect_bd_net [get_bd_ports sys_clk_50] [get_bd_pins clk_wiz_0/clk_in1]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_ports core_clk]

# power-on reset held until the MMCM locks
set rst_core [create_bd_cell -type ip -vlnv [ip_vlnv proc_sys_reset] rst_core]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins rst_core/slowest_sync_clk]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins rst_core/dcm_locked]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins rst_core/ext_reset_in]
connect_bd_net [get_bd_pins rst_core/peripheral_aresetn] [get_bd_ports core_rstn]

# ------------------------------------------------------------------ PCIe: XDMA, Gen2 x8
set ibuf [create_bd_cell -type ip -vlnv [ip_vlnv util_ds_buf] refclk_buf]
set_property CONFIG.C_BUF_TYPE {IBUFDSGTE} $ibuf
connect_bd_intf_net $pcie_refclk [get_bd_intf_pins refclk_buf/CLK_IN_D]

set xdma [create_bd_cell -type ip -vlnv [ip_vlnv xdma] xdma_0]
set_property -dict [list \
  CONFIG.mode_selection {Advanced} \
  CONFIG.pl_link_cap_max_link_width {X8} \
  CONFIG.pl_link_cap_max_link_speed {5.0_GT/s} \
  CONFIG.axi_data_width {128_bit} \
  CONFIG.axisten_freq {250} \
  CONFIG.ref_clk_freq {100_MHz} \
  CONFIG.pf0_device_id {7028} \
  CONFIG.xdma_rnum_chnl {1} CONFIG.xdma_wnum_chnl {1} \
  CONFIG.axilite_master_en {true} CONFIG.axilite_master_size {1} \
  CONFIG.axilite_master_scale {Megabytes} \
  CONFIG.pciebar2axibar_axil_master {0x00000000} \
  CONFIG.xdma_axi_intf_mm {AXI_Memory_Mapped} \
  CONFIG.plltype {QPLL1} \
] $xdma
# The PCIe block and GT quad: Vivado picks the default for the part; the card's lanes are on
# MGT pins F2..U4 with the reference clock on J8. If placement fails, set
#   CONFIG.pcie_blk_locn and CONFIG.select_quad  (see docs/board.md, "PCIe placement").
if {[info exists PCIE_BLK_LOCN]} { set_property CONFIG.pcie_blk_locn $PCIE_BLK_LOCN $xdma }
if {[info exists PCIE_QUAD]} { set_property CONFIG.select_quad $PCIE_QUAD $xdma }

connect_bd_net [get_bd_pins refclk_buf/IBUF_OUT] [get_bd_pins xdma_0/sys_clk]
connect_bd_net [get_bd_ports pcie_perstn] [get_bd_pins xdma_0/sys_rst_n]
make_bd_intf_pins_external [get_bd_intf_pins xdma_0/pcie_mgt]
set_property NAME pcie_mgt [get_bd_intf_ports -of [get_bd_intf_nets -of [get_bd_intf_pins xdma_0/pcie_mgt]]]
connect_bd_net [get_bd_pins xdma_0/user_lnk_up] [get_bd_ports pcie_link_up]

# ------------------------------------------------------------------ DDR3: two MIGs
set sysclk [expr {$DDR_SPEED == 1066 ? "clk_wiz_0/clk_267" : "clk_wiz_0/clk_200"}]
foreach ch {0 1} {
  set mig [create_bd_cell -type ip -vlnv [ip_vlnv mig_7series] mig_$ch]
  set dir [get_property IP_DIR [get_ips [get_property CONFIG.Component_Name $mig]]]
  file copy -force [file join $MIG_DIR mig_ddr3_ch${ch}.prj] [file join $dir mig_ddr3_ch${ch}.prj]
  set_property -dict [list CONFIG.BOARD_MIG_PARAM {Custom} CONFIG.MIG_DONT_TOUCH_PARAM {Custom} \
    CONFIG.RESET_BOARD_INTERFACE {Custom} CONFIG.XML_INPUT_FILE mig_ddr3_ch${ch}.prj] $mig
  connect_bd_net [get_bd_pins $sysclk] [get_bd_pins mig_$ch/sys_clk_i]
  connect_bd_net [get_bd_pins clk_wiz_0/clk_200] [get_bd_pins mig_$ch/clk_ref_i]
  connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins mig_$ch/sys_rst]
  make_bd_intf_pins_external [get_bd_intf_pins mig_$ch/DDR3]
  set_property NAME DDR3_$ch [get_bd_intf_ports -of [get_bd_intf_nets -of [get_bd_intf_pins mig_$ch/DDR3]]]
  # AXI reset in the controller's user clock domain
  set r [create_bd_cell -type ip -vlnv [ip_vlnv proc_sys_reset] rst_mig_$ch]
  connect_bd_net [get_bd_pins mig_$ch/ui_clk] [get_bd_pins rst_mig_$ch/slowest_sync_clk]
  connect_bd_net [get_bd_pins mig_$ch/ui_clk_sync_rst] [get_bd_pins rst_mig_$ch/ext_reset_in]
  set_property CONFIG.C_EXT_RESET_HIGH {1} $r
  connect_bd_net [get_bd_pins mig_$ch/mmcm_locked] [get_bd_pins rst_mig_$ch/dcm_locked]
  connect_bd_net [get_bd_pins rst_mig_$ch/peripheral_aresetn] [get_bd_pins mig_$ch/aresetn]
}
# one XADC: channel 1 takes the die temperature from channel 0, and so does the accelerator
# (the TEMP register)
connect_bd_net [get_bd_pins mig_0/device_temp] [get_bd_pins mig_1/device_temp_i] \
  [get_bd_ports device_temp]

set cat [create_bd_cell -type ip -vlnv [ip_vlnv xlconcat] calib_cat]
set_property CONFIG.NUM_PORTS {2} $cat
connect_bd_net [get_bd_pins mig_0/init_calib_complete] [get_bd_pins calib_cat/In0]
connect_bd_net [get_bd_pins mig_1/init_calib_complete] [get_bd_pins calib_cat/In1]
connect_bd_net [get_bd_pins calib_cat/dout] [get_bd_ports calib]

# ------------------------------------------------------------------ interconnect
# Memory: XDMA (250 MHz, 128b) and the accelerator's two masters (100 MHz, 512b) onto both
# controllers. SmartConnect converts width and clock.
set scm [create_bd_cell -type ip -vlnv [ip_vlnv smartconnect] sc_mem]
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {2} CONFIG.NUM_CLKS {4}] $scm
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins sc_mem/aclk]
connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_pins sc_mem/aclk1]
connect_bd_net [get_bd_pins mig_0/ui_clk] [get_bd_pins sc_mem/aclk2]
connect_bd_net [get_bd_pins mig_1/ui_clk] [get_bd_pins sc_mem/aclk3]
connect_bd_net [get_bd_pins rst_core/interconnect_aresetn] [get_bd_pins sc_mem/aresetn]
connect_bd_intf_net [get_bd_intf_pins xdma_0/M_AXI] [get_bd_intf_pins sc_mem/S00_AXI]
connect_bd_intf_net [get_bd_intf_ports S_AXI_M0] [get_bd_intf_pins sc_mem/S01_AXI]
connect_bd_intf_net [get_bd_intf_ports S_AXI_M1] [get_bd_intf_pins sc_mem/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_mem/M00_AXI] [get_bd_intf_pins mig_0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_mem/M01_AXI] [get_bd_intf_pins mig_1/S_AXI]

# Control: XDMA's AXI-Lite master (BAR0) into the core clock domain
set scl [create_bd_cell -type ip -vlnv [ip_vlnv smartconnect] sc_ctl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1} CONFIG.NUM_CLKS {2}] $scl
connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_pins sc_ctl/aclk]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins sc_ctl/aclk1]
connect_bd_net [get_bd_pins xdma_0/axi_aresetn] [get_bd_pins sc_ctl/aresetn]
connect_bd_intf_net [get_bd_intf_pins xdma_0/M_AXI_LITE] [get_bd_intf_pins sc_ctl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M00_AXI] [get_bd_intf_ports M_AXI_CTL]

# ------------------------------------------------------------------ address map
foreach space [list [get_bd_addr_spaces xdma_0/M_AXI] [get_bd_addr_spaces S_AXI_M0] \
                    [get_bd_addr_spaces S_AXI_M1]] {
  assign_bd_address -offset 0x00000000 -range 2G -target_address_space $space \
    [get_bd_addr_segs mig_0/memmap/memaddr] -force
  assign_bd_address -offset 0x80000000 -range 2G -target_address_space $space \
    [get_bd_addr_segs mig_1/memmap/memaddr] -force
}
assign_bd_address -offset 0x00000000 -range 64K -target_address_space \
  [get_bd_addr_spaces xdma_0/M_AXI_LITE] [get_bd_addr_segs M_AXI_CTL/Reg] -force

validate_bd_design
save_bd_design
