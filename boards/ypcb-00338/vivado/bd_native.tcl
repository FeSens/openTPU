# Block design "otpu_bd" for the YPCB-00338 (create_project.tcl): PCIe (XDMA), the core clock,
# resets, the XADC and the control interconnect. No memory controllers and no memory
# interconnect: the accelerator (otpu_board) and XDMA's DMA meet on each channel in otpu_mem_ch
# (otpu_native_sys), outside the block design, in front of the LiteDRAM core. External interfaces:
#   M_AXI_CTL     AXI4-Lite master -> the accelerator's control registers (BAR0 0x0000, core_clk)
#   M_AXI_DMA     AXI4 master, 128b -> the channels (XDMA's DMA, xdma_aclk; bit 31 = channel)
#   M_AXI_MEMCAL  AXI4-Lite master -> the LiteDRAM core's CSRs (BAR0 0x10000, 64 KB, xdma_aclk:
#                 the core crosses them into its own clock; opentpu/host/memcal.py)
# and device_temp (the XADC die-temperature code, core_clk; BAR0 0x30000: the XADC registers).
#
# Clocks: the 50 MHz clock arrives already on a global buffer (otpu_fpga_top_ld.sv: the LiteDRAM
# core's MMCMs share it), so the clock wizard has no input buffer and makes core_clk only.
# xdma_aclk 125 MHz (Gen1 x8, 128 bits).
#
# Variables (set before sourcing): CORE_MHZ.

# the newest installed version of an IP
proc ip_vlnv {name} {
  set defs [lsort -decreasing [get_ipdefs -all "xilinx.com:ip:${name}:*"]]
  if {[llength $defs] == 0} { error "IP $name not found in this Vivado installation" }
  return [lindex $defs 0]
}

create_bd_design otpu_bd
current_bd_design otpu_bd

# core clock: CORE_MHZ rounded to the MMCM's 1/8 divider steps of its VCO
if {![info exists CORE_MHZ]} { set CORE_MHZ 100 }
set VCO 800
set CORE_DIV [expr {round(double($VCO) / $CORE_MHZ * 8) / 8.0}]
set CORE_MHZ_ACT [format %.3f [expr {double($VCO) / $CORE_DIV}]]
set CORE_HZ [expr {int(floor($VCO * 1.0e6 / $CORE_DIV))}]
puts "core_clk: $CORE_MHZ_ACT MHz (MMCM divide $CORE_DIV)"

# ------------------------------------------------------------------ external ports
create_bd_port -dir I -type clk -freq_hz 50000000 sys_clk_50
set pcie_refclk [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 pcie_refclk]
set_property CONFIG.FREQ_HZ 100000000 $pcie_refclk
create_bd_port -dir I -type rst pcie_perstn
set_property CONFIG.POLARITY ACTIVE_LOW [get_bd_ports pcie_perstn]
create_bd_port -dir O -type clk core_clk
create_bd_port -dir O -type rst core_rstn
create_bd_port -dir O -type clk xdma_aclk
create_bd_port -dir O -type rst xdma_aresetn
create_bd_port -dir O -from 11 -to 0 device_temp
create_bd_port -dir O pcie_link_up

set lites {M_AXI_CTL M_AXI_MEMCAL}
foreach p $lites {
  set m [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 $p]
  set_property -dict [list CONFIG.PROTOCOL AXI4LITE CONFIG.DATA_WIDTH 32 CONFIG.ADDR_WIDTH 32 \
    CONFIG.HAS_BURST 0 CONFIG.HAS_LOCK 0 CONFIG.HAS_PROT 0 CONFIG.HAS_CACHE 0 CONFIG.HAS_QOS 0 \
    CONFIG.HAS_REGION 0] $m
}
set_property CONFIG.FREQ_HZ 125000000 [get_bd_intf_ports M_AXI_MEMCAL]
set_property CONFIG.ASSOCIATED_BUSIF {M_AXI_CTL} [get_bd_ports core_clk]
set_property CONFIG.ASSOCIATED_RESET {core_rstn} [get_bd_ports core_clk]
set_property CONFIG.FREQ_HZ $CORE_HZ [get_bd_ports core_clk]

# ------------------------------------------------------------------ clocks and resets
set clk [create_bd_cell -type ip -vlnv [ip_vlnv clk_wiz] clk_wiz_0]
set_property -dict [list \
  CONFIG.PRIM_IN_FREQ {50.000} CONFIG.PRIM_SOURCE {No_buffer} \
  CONFIG.USE_RESET {false} CONFIG.USE_LOCKED {true} \
  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $CORE_MHZ_ACT CONFIG.NUM_OUT_CLKS {1} \
  CONFIG.MMCM_DIVCLK_DIVIDE {1} \
  CONFIG.MMCM_CLKFBOUT_MULT_F [format %.3f [expr {$VCO / 50.0}]] \
  CONFIG.MMCM_CLKOUT0_DIVIDE_F $CORE_DIV CONFIG.CLK_OUT1_PORT {core_clk} \
] $clk
connect_bd_net [get_bd_ports sys_clk_50] [get_bd_pins clk_wiz_0/clk_in1]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_ports core_clk]
# With one output the wizard picks its own M / D / O and may miss VCO / CORE_DIV
# (133.33: M 62.625, D 4, O 5.875 = 133.245 MHz; 150: 148.828 for 148.837) and validation fails
# on core_clk's FREQ_HZ. Then the MMCM settings are the ones above, in override mode. Where the
# wizard's own are exact they stay (100: M 20 / O 10, VCO 1000, the qualified images; 120.755).
if {[get_property CONFIG.FREQ_HZ [get_bd_pins clk_wiz_0/core_clk]] != $CORE_HZ} {
  set_property -dict [list CONFIG.OVERRIDE_MMCM {true} CONFIG.MMCM_DIVCLK_DIVIDE {1} \
    CONFIG.MMCM_CLKFBOUT_MULT_F [format %.3f [expr {$VCO / 50.0}]] \
    CONFIG.MMCM_CLKOUT0_DIVIDE_F $CORE_DIV] $clk
}
# core_clk as the MMCM makes it: the port's FREQ_HZ and create_project.tcl's CORE_KHZ assume it
set f [get_property CONFIG.FREQ_HZ [get_bd_pins clk_wiz_0/core_clk]]
if {$f != $CORE_HZ} { error "bd_native.tcl: the clock wizard makes core_clk at $f Hz, not $CORE_HZ" }
puts "core_clk: $f Hz (MMCM M [get_property CONFIG.MMCM_CLKFBOUT_MULT_F $clk],\
  D [get_property CONFIG.MMCM_DIVCLK_DIVIDE $clk], O [get_property CONFIG.MMCM_CLKOUT0_DIVIDE_F $clk])"

# core reset: until the MMCM locks, and while the host asserts PCIe PERST#
set rst_core [create_bd_cell -type ip -vlnv [ip_vlnv proc_sys_reset] rst_core]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins rst_core/slowest_sync_clk]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins rst_core/dcm_locked]
connect_bd_net [get_bd_ports pcie_perstn] [get_bd_pins rst_core/ext_reset_in]
connect_bd_net [get_bd_pins rst_core/peripheral_aresetn] [get_bd_ports core_rstn]

# ------------------------------------------------------------------ PCIe: XDMA, Gen1 x8
# The production settings and identity, those of every bitstream since the first (the host tells
# the memory build from CAPS, not from the PCI identity).
# xdma_rnum_rids: the H2C engine's outstanding read requests, 8 x the driver's 512-byte MRRS =
# 4 KiB. At the default 32 (16 KiB) the engine laps its read buffer when the card holds up its
# writes (a run's or a card->host read's traffic on the channel): the data jumps 8 KiB, or goes
# stale, and the engine keeps the slip until the bitstream is loaded again (docs/host.md, "XDMA's
# H2C overrun"); host calls of 4 KiB, which keep 4 KiB in flight at most, were clean.
set ibuf [create_bd_cell -type ip -vlnv [ip_vlnv util_ds_buf] refclk_buf]
set_property CONFIG.C_BUF_TYPE {IBUFDSGTE} $ibuf
connect_bd_intf_net $pcie_refclk [get_bd_intf_pins refclk_buf/CLK_IN_D]

set xdma [create_bd_cell -type ip -vlnv [ip_vlnv xdma] xdma_0]
set_property -dict [list \
  CONFIG.mode_selection {Advanced} \
  CONFIG.pl_link_cap_max_link_width {X8} \
  CONFIG.pl_link_cap_max_link_speed {2.5_GT/s} \
  CONFIG.axi_data_width {128_bit} \
  CONFIG.axisten_freq {125} \
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
] $xdma
connect_bd_net [get_bd_pins refclk_buf/IBUF_OUT] [get_bd_pins xdma_0/sys_clk]
connect_bd_net [get_bd_ports pcie_perstn] [get_bd_pins xdma_0/sys_rst_n]
make_bd_intf_pins_external [get_bd_intf_pins xdma_0/pcie_mgt]
set_property NAME pcie_mgt [get_bd_intf_ports -of [get_bd_intf_nets -of [get_bd_intf_pins xdma_0/pcie_mgt]]]
connect_bd_net [get_bd_pins xdma_0/user_lnk_up] [get_bd_ports pcie_link_up]
connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_ports xdma_aclk]
connect_bd_net [get_bd_pins xdma_0/axi_aresetn] [get_bd_ports xdma_aresetn]

# XDMA's DMA master leaves the design as it is (otpu_axi_split2 takes it in the top)
make_bd_intf_pins_external [get_bd_intf_pins xdma_0/M_AXI]
set_property NAME M_AXI_DMA [get_bd_intf_ports -of [get_bd_intf_nets -of [get_bd_intf_pins xdma_0/M_AXI]]]
set_property CONFIG.ASSOCIATED_BUSIF {M_AXI_DMA:M_AXI_MEMCAL} [get_bd_ports xdma_aclk]
set_property CONFIG.ASSOCIATED_RESET {xdma_aresetn} [get_bd_ports xdma_aclk]
set_property CONFIG.FREQ_HZ 125000000 [get_bd_ports xdma_aclk]

# ------------------------------------------------------------------ XADC (die temperature)
# temp_out for the accelerator's TEMP register, the XADC registers at BAR0 0x30000.
set xadc [create_bd_cell -type ip -vlnv [ip_vlnv xadc_wiz] xadc_temp]
set_property -dict [list CONFIG.INTERFACE_SELECTION {Enable_AXI} CONFIG.DCLK_FREQUENCY {100} \
  CONFIG.XADC_STARUP_SELECTION {single_channel} CONFIG.SINGLE_CHANNEL_SELECTION {TEMPERATURE} \
  CONFIG.TIMING_MODE {Continuous} CONFIG.ENABLE_TEMP_BUS {true} \
  CONFIG.OT_ALARM {false} CONFIG.USER_TEMP_ALARM {false} CONFIG.VCCINT_ALARM {false} \
  CONFIG.VCCAUX_ALARM {false} CONFIG.CHANNEL_ENABLE_VP_VN {false}] $xadc
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins xadc_temp/s_axi_aclk]
connect_bd_net [get_bd_pins rst_core/peripheral_aresetn] [get_bd_pins xadc_temp/s_axi_aresetn]
connect_bd_net [get_bd_pins xadc_temp/temp_out] [get_bd_ports device_temp]

# ------------------------------------------------------------------ control interconnect
# XDMA's AXI-Lite master (BAR0): the control registers (into core_clk), the XADC (core_clk) and
# the LiteDRAM CSRs (in xdma_aclk)
set scl [create_bd_cell -type ip -vlnv [ip_vlnv smartconnect] sc_ctl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {3} CONFIG.NUM_CLKS {2}] $scl
connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_pins sc_ctl/aclk]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins sc_ctl/aclk1]
connect_bd_net [get_bd_pins xdma_0/axi_aresetn] [get_bd_pins sc_ctl/aresetn]
connect_bd_intf_net [get_bd_intf_pins xdma_0/M_AXI_LITE] [get_bd_intf_pins sc_ctl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M00_AXI] [get_bd_intf_ports M_AXI_CTL]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M01_AXI] [get_bd_intf_pins xadc_temp/s_axi_lite]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M02_AXI] [get_bd_intf_ports M_AXI_MEMCAL]

# ------------------------------------------------------------------ address map
# BAR0: 0x0 the control registers, 0x10000 the LiteDRAM CSRs (R_MEMCAL; both channels', by
# csr.csv), 0x30000 the XADC. XDMA's DMA: the whole 32-bit space to M_AXI_DMA
# (channel 0 at 0x0000_0000, channel 1 at 0x8000_0000, 2 GiB each: opentpu/host/board.py)
set lite [get_bd_addr_spaces xdma_0/M_AXI_LITE]
assign_bd_address -offset 0x00000000 -range 64K -target_address_space $lite [get_bd_addr_segs M_AXI_CTL/Reg] -force
assign_bd_address -offset 0x00010000 -range 64K -target_address_space $lite [get_bd_addr_segs M_AXI_MEMCAL/Reg] -force
assign_bd_address -offset 0x00030000 -range 64K -target_address_space $lite \
  [get_bd_addr_segs -of [get_bd_intf_pins xadc_temp/s_axi_lite]] -force
assign_bd_address -offset 0x00000000 -range 4G -target_address_space [get_bd_addr_spaces xdma_0/M_AXI] \
  [get_bd_addr_segs M_AXI_DMA/Reg] -force

validate_bd_design

# the external ports the top connects to (IP integrator renames a clashing port silently)
foreach p [concat {pcie_mgt pcie_refclk M_AXI_DMA} $lites] {
  if {[get_bd_intf_ports $p] eq ""} { error "block design port $p missing: [get_bd_intf_ports]" }
}
if {[llength [get_bd_nets -of [get_bd_ports device_temp]]] == 0 || \
    [get_bd_pins -of [get_bd_nets -of [get_bd_ports device_temp]] -filter {DIR == O}] eq ""} {
  error "device_temp has no driver"
}
foreach {cell want} {rst_core 0} {
  set got [get_property CONFIG.C_EXT_RESET_HIGH [get_bd_cells $cell]]
  if {$got != $want} { error "$cell: C_EXT_RESET_HIGH is $got, expected $want" }
}
foreach {k want} {pf0_device_id 7028 pf0_class_code 120000 pf0_subsystem_id 4F54 pf0_revision_id 01} {
  set got [get_property CONFIG.$k [get_bd_cells xdma_0]]
  if {$got ne $want} { error "xdma_0: $k is $got, expected $want" }
}
save_bd_design
