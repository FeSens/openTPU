# Block design "otpu_bd" for the SQRL FK33 (create_project.tcl; boards/bcu1525/vivado/bd.tcl is its model): PCIe
# (XDMA), the clocks, resets, the SYSMON, the control interconnect and XDMA's DMA split onto the four channels. No
# memory controller: the HBM IP, its adapters and the accelerator are outside, in the top
# (rtl/boards/sqrl-fk33/otpu_fpga_top_fk33.sv). External interfaces:
#   M_AXI_CTL      AXI4-Lite master -> the accelerator's control registers (BAR0 0x00000, core_clk)
#   M_AXI_DMA<j>   AXI4 masters, 128 bits, no IDs, xdma_aclk -> channel j's bridge (otpu_mem_ch): XDMA's DMA
#                  through the SmartConnect sc_dma, which decodes card address bits 32:31 (channel j at j x 2 GiB;
#                  opentpu.host.board.channel_map) and converts the width
# and device_temp (the SYSMON's die temperature, 10 bits, core_clk; BAR0 0x30000: its registers).
#
# PCIe: the settings of the openTPU_FK FK33 / FK35 builds that run on the cards (the FK35's bd.tcl, four DMA ports: XDMA on
# PCIE4C_X1Y0, the GTY lanes of the card's edge connector, SQRL's identity 1E24:1533 / 1E24:0001 rev A3, which the
# Windows driver sqrlfk.sys binds to) without their Tandem PCIe: the bitstream is loaded over JTAG (or from the flash)
# whole. Gen3 x8, 256-bit AXI at 250 MHz. The card has no clock input besides the PCIe reference, so the clock
# wizard runs from XDMA's axi_aclk (250 MHz): core_clk (CORE_MHZ), uclk (UCLK_MHZ: the HBM's AXI ports, the adapters,
# otpu_mem_ch's controller side) and clk100 (the HBM's reference and APB clock); each resets until its lock and with
# axi_aresetn.
# Variables (set before sourcing): CORE_MHZ, UCLK_MHZ (default 250 both).
proc ip_vlnv {name} {
  set defs [lsort -decreasing [get_ipdefs -all "xilinx.com:ip:${name}:*"]]
  if {[llength $defs] == 0} { error "IP $name not found in this Vivado installation" }
  return [lindex $defs 0]
}
create_bd_design otpu_bd
current_bd_design otpu_bd
if {![info exists CORE_MHZ]} { set CORE_MHZ 250 }
if {![info exists UCLK_MHZ]} { set UCLK_MHZ 250 }
set XDMA_MHZ 250
set XDMA_HZ [expr {$XDMA_MHZ * 1000000}]
set AXI_BITS 256
# ------------------------------------------------------------------ external ports
set pcie_refclk [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 pcie_refclk]
set_property CONFIG.FREQ_HZ 100000000 $pcie_refclk
create_bd_port -dir I -type rst pcie_perstn
set_property CONFIG.POLARITY ACTIVE_LOW [get_bd_ports pcie_perstn]
foreach p {core_clk uclk clk100 xdma_aclk} { create_bd_port -dir O -type clk $p }
foreach p {core_rstn urstn clk100_rstn xdma_aresetn} { create_bd_port -dir O -type rst $p }
create_bd_port -dir O -from 9 -to 0 device_temp
create_bd_port -dir O pcie_link_up
set m [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI_CTL]
set_property -dict [list CONFIG.PROTOCOL AXI4LITE CONFIG.DATA_WIDTH 32 CONFIG.ADDR_WIDTH 32 \
  CONFIG.HAS_BURST 0 CONFIG.HAS_LOCK 0 CONFIG.HAS_PROT 0 CONFIG.HAS_CACHE 0 CONFIG.HAS_QOS 0 \
  CONFIG.HAS_REGION 0] $m
# ------------------------------------------------------------------ PCIe: XDMA
set ibuf [create_bd_cell -type ip -vlnv [ip_vlnv util_ds_buf] refclk_buf]
set_property CONFIG.C_BUF_TYPE {IBUFDSGTE} $ibuf
connect_bd_intf_net $pcie_refclk [get_bd_intf_pins refclk_buf/CLK_IN_D]
set xdma [create_bd_cell -type ip -vlnv [ip_vlnv xdma] xdma_0]
set_property -dict [list CONFIG.mode_selection {Advanced} CONFIG.pcie_blk_locn {PCIE4C_X1Y0}] $xdma
set_property -dict [list \
  CONFIG.cfg_mgmt_if {false} \
  CONFIG.pl_link_cap_max_link_width {X8} CONFIG.pl_link_cap_max_link_speed {8.0_GT/s} \
  CONFIG.axi_data_width "${AXI_BITS}_bit" CONFIG.axisten_freq $XDMA_MHZ CONFIG.ref_clk_freq {100_MHz} \
  CONFIG.vendor_id {1E24} CONFIG.pf0_device_id {1533} \
  CONFIG.pf0_revision_id {A3} CONFIG.pf0_subsystem_vendor_id {1E24} CONFIG.pf0_subsystem_id {0001} \
  CONFIG.pf0_class_code_base {12} CONFIG.pf0_class_code_sub {00} CONFIG.pf0_class_code_interface {00} \
  CONFIG.xdma_rnum_chnl {1} CONFIG.xdma_wnum_chnl {1} CONFIG.xdma_rnum_rids {8} \
  CONFIG.axilite_master_en {true} CONFIG.axilite_master_size {1} CONFIG.axilite_master_scale {Megabytes} \
  CONFIG.pciebar2axibar_axil_master {0x00000000} \
  CONFIG.xdma_axi_intf_mm {AXI_Memory_Mapped} CONFIG.plltype {QPLL1} \
] $xdma
connect_bd_net [get_bd_pins refclk_buf/IBUF_DS_ODIV2] [get_bd_pins xdma_0/sys_clk]
connect_bd_net [get_bd_pins refclk_buf/IBUF_OUT] [get_bd_pins xdma_0/sys_clk_gt]
connect_bd_net [get_bd_ports pcie_perstn] [get_bd_pins xdma_0/sys_rst_n]
make_bd_intf_pins_external [get_bd_intf_pins xdma_0/pcie_mgt]
set_property NAME pcie_mgt [get_bd_intf_ports -of [get_bd_intf_nets -of [get_bd_intf_pins xdma_0/pcie_mgt]]]
connect_bd_net [get_bd_pins xdma_0/user_lnk_up] [get_bd_ports pcie_link_up]
connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_ports xdma_aclk]
connect_bd_net [get_bd_pins xdma_0/axi_aresetn] [get_bd_ports xdma_aresetn]
# ------------------------------------------------------------------ clocks and resets
set clk [create_bd_cell -type ip -vlnv [ip_vlnv clk_wiz] clk_wiz_0]
set_property -dict [list CONFIG.PRIM_SOURCE {No_buffer} CONFIG.PRIM_IN_FREQ $XDMA_MHZ.000 \
  CONFIG.CLKOUT2_USED {true} CONFIG.CLKOUT3_USED {true} CONFIG.NUM_OUT_CLKS {3} \
  CONFIG.CLK_OUT1_PORT {core_clk} CONFIG.CLK_OUT2_PORT {uclk} CONFIG.CLK_OUT3_PORT {clk100} \
  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $CORE_MHZ CONFIG.CLKOUT2_REQUESTED_OUT_FREQ $UCLK_MHZ \
  CONFIG.CLKOUT3_REQUESTED_OUT_FREQ {100.000} CONFIG.RESET_TYPE {ACTIVE_LOW} CONFIG.RESET_PORT {resetn}] $clk
connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_pins clk_wiz_0/clk_in1]
connect_bd_net [get_bd_pins xdma_0/axi_aresetn] [get_bd_pins clk_wiz_0/resetn]
foreach {r c o} {rst_core core_clk core_rstn rst_uclk uclk urstn rst_clk100 clk100 clk100_rstn} {
  create_bd_cell -type ip -vlnv [ip_vlnv proc_sys_reset] $r
  connect_bd_net [get_bd_pins clk_wiz_0/$c] [get_bd_pins $r/slowest_sync_clk]
  connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins $r/dcm_locked]
  connect_bd_net [get_bd_pins xdma_0/axi_aresetn] [get_bd_pins $r/ext_reset_in]
  connect_bd_net [get_bd_pins $r/peripheral_aresetn] [get_bd_ports $o]
}
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_ports core_clk]
connect_bd_net [get_bd_pins clk_wiz_0/uclk] [get_bd_ports uclk]
connect_bd_net [get_bd_pins clk_wiz_0/clk100] [get_bd_ports clk100]
set CORE_HZ [get_property CONFIG.FREQ_HZ [get_bd_pins clk_wiz_0/core_clk]]
set UCLK_HZ [get_property CONFIG.FREQ_HZ [get_bd_pins clk_wiz_0/uclk]]
puts "core_clk $CORE_HZ Hz, uclk $UCLK_HZ Hz (clock wizard on XDMA's axi_aclk)"
set_property -dict [list CONFIG.FREQ_HZ $CORE_HZ CONFIG.ASSOCIATED_BUSIF M_AXI_CTL CONFIG.ASSOCIATED_RESET core_rstn] \
  [get_bd_ports core_clk]
# ------------------------------------------------------------------ XDMA's DMA onto the four channels
set dmas {M_AXI_DMA0 M_AXI_DMA1 M_AXI_DMA2 M_AXI_DMA3}
set scd [create_bd_cell -type ip -vlnv [ip_vlnv smartconnect] sc_dma]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {4} CONFIG.NUM_CLKS {1}] $scd
connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_pins sc_dma/aclk]
connect_bd_net [get_bd_pins xdma_0/axi_aresetn] [get_bd_pins sc_dma/aresetn]
connect_bd_intf_net [get_bd_intf_pins xdma_0/M_AXI] [get_bd_intf_pins sc_dma/S00_AXI]
foreach p $dmas {
  set m [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 $p]
  set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 128 CONFIG.ADDR_WIDTH 33 \
    CONFIG.HAS_BURST 0 CONFIG.HAS_LOCK 0 CONFIG.HAS_PROT 0 CONFIG.HAS_CACHE 0 \
    CONFIG.HAS_QOS 0 CONFIG.HAS_REGION 0 CONFIG.FREQ_HZ $XDMA_HZ] $m
  connect_bd_intf_net [get_bd_intf_pins sc_dma/M0[string index $p end]_AXI] $m
}
set_property -dict [list CONFIG.ASSOCIATED_BUSIF [join $dmas :] CONFIG.ASSOCIATED_RESET xdma_aresetn \
  CONFIG.FREQ_HZ $XDMA_HZ] [get_bd_ports xdma_aclk]
# ------------------------------------------------------------------ SYSMON (die temperature)
# temp_out: SYSMONE4's 10-bit code (the host's formula is the platform's: UltraScale+, as the BCU1525)
set sysmon [create_bd_cell -type ip -vlnv [ip_vlnv system_management_wiz] sysmon]
set_property CONFIG.ENABLE_TEMP_BUS {true} $sysmon
set_property -dict [list CONFIG.INTERFACE_SELECTION {Enable_AXI} \
  CONFIG.DCLK_FREQUENCY [expr {int(ceil($CORE_HZ / 1.0e6))}] CONFIG.SEQUENCER_MODE {Default} \
  CONFIG.CHANNEL_ENABLE_VP_VN {false} CONFIG.OT_ALARM {false} CONFIG.USER_TEMP_ALARM {false} \
  CONFIG.VCCINT_ALARM {false} CONFIG.VCCAUX_ALARM {false}] $sysmon
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins sysmon/s_axi_aclk]
connect_bd_net [get_bd_pins rst_core/peripheral_aresetn] [get_bd_pins sysmon/s_axi_aresetn]
connect_bd_net [get_bd_pins sysmon/temp_out] [get_bd_ports device_temp]
# ------------------------------------------------------------------ control interconnect
set scl [create_bd_cell -type ip -vlnv [ip_vlnv smartconnect] sc_ctl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2} CONFIG.NUM_CLKS {2}] $scl
connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_pins sc_ctl/aclk]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins sc_ctl/aclk1]
connect_bd_net [get_bd_pins xdma_0/axi_aresetn] [get_bd_pins sc_ctl/aresetn]
connect_bd_intf_net [get_bd_intf_pins xdma_0/M_AXI_LITE] [get_bd_intf_pins sc_ctl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M00_AXI] [get_bd_intf_ports M_AXI_CTL]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M01_AXI] [get_bd_intf_pins sysmon/S_AXI_LITE]
# ------------------------------------------------------------------ address map
# BAR0 (1 MB): 0x00000 the control registers, 0x30000 the SYSMON (the YPCB's and BCU1525's layout). XDMA's DMA:
# 8 GiB, M_AXI_DMA<j> at j x 2 GiB (explicit offsets: 2019.2's Tcl 8.5 format %X truncates to 32 bits)
set lite [get_bd_addr_spaces xdma_0/M_AXI_LITE]
assign_bd_address -offset 0x00000000 -range 64K -target_address_space $lite [get_bd_addr_segs M_AXI_CTL/Reg] -force
assign_bd_address -offset 0x00030000 -range 64K -target_address_space $lite \
  [get_bd_addr_segs -of [get_bd_intf_pins sysmon/S_AXI_LITE]] -force
foreach {j off} {0 0x000000000 1 0x080000000 2 0x100000000 3 0x180000000} {
  assign_bd_address -offset $off -range 2G \
    -target_address_space [get_bd_addr_spaces xdma_0/M_AXI] [get_bd_addr_segs M_AXI_DMA$j/Reg] -force
}
validate_bd_design
foreach p [concat {pcie_mgt pcie_refclk M_AXI_CTL} $dmas] {
  if {[get_bd_intf_ports $p] eq ""} { error "block design port $p missing: [get_bd_intf_ports]" }
}
foreach {k want} [list vendor_id 1E24 pf0_device_id 1533 pcie_blk_locn PCIE4C_X1Y0 axisten_freq $XDMA_MHZ \
                       pl_link_cap_max_link_width X8 axi_data_width ${AXI_BITS}_bit] {
  set got [get_property CONFIG.$k [get_bd_cells xdma_0]]
  if {$got ne $want} { error "xdma_0: $k is $got, expected $want" }
}
foreach p $dmas {
  if {[get_property CONFIG.DATA_WIDTH [get_bd_intf_ports $p]] != 128} { error "$p: expected 128 bits" }
  if {[get_property CONFIG.ID_WIDTH [get_bd_intf_ports $p]] != 0} { error "$p: expected no ID bits" }
}
save_bd_design
