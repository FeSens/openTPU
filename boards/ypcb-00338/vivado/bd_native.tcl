# Block design "otpu_bd" for the YPCB-00338 builds whose memory channels are native ports behind
# otpu_mem_ch (create_project.tcl MEM=mig_native or MEM=litedram; bd.tcl is the AXI MIG build's):
# PCIe (XDMA), the core clock, resets, the XADC and the control interconnect, and for mig_native
# the two MIGs with their native user interface. There is no memory interconnect: the
# accelerator (otpu_board with MEM_NATIVE) and XDMA's DMA meet on each channel in otpu_mem_ch
# (otpu_native_sys), outside the block design. The same name as bd.tcl's design, so build.tcl
# serves every build. External interfaces:
#   M_AXI_CTL     AXI4-Lite master -> the accelerator's control registers (BAR0 0x0000, core_clk)
#   M_AXI_DMA     AXI4 master, 128b -> the channels (XDMA's DMA, xdma_aclk; bit 31 = channel)
#   M_AXI_MEMCAL  (litedram) AXI4-Lite master -> the LiteDRAM core's CSRs (BAR0 0x10000, 64 KB,
#                 xdma_aclk: the core crosses them into its own clock; opentpu/host/memcal.py)
#   DDR3_0/1      (mig_native) the channels' DDR3 pins, as bd.tcl's
#   mig<c>_*      (mig_native) channel c's MIG native interface (app_*, otpu_mig_native), its
#                 user clock ui_clk and reset ui_clk_sync_rst
# and device_temp (the XADC die-temperature code, core_clk; BAR0 0x30000: the XADC registers),
# calib (mig_native: the MIGs' init_calib_complete, each in its ui_clk; the LiteDRAM core has its
# own ready bits, set by the host's calibration). BAR0 0x10000 / 0x20000 (the AXI MIGs' ECC
# registers in bd.tcl) are unmapped for mig_native: their S_AXI_CTRL comes with the AXI interface.
#
# Clocks: mig_native as bd.tcl (the 50 MHz pin into the clock wizard: core_clk, clk_200 for the
# MIGs' IDELAYCTRL and DDR3-800 / 1300 / 1600 system clock, clk_mig for DDR3-1066 / 1333; the
# MIGs' ui_clk 4:1 of the memory clock). litedram: the 50 MHz clock arrives already on a global
# buffer (otpu_fpga_top_ld.sv: the LiteDRAM core's MMCMs share it), so the clock wizard has no
# input buffer and makes core_clk only. xdma_aclk 125 MHz (Gen1 x8, 128 bits).
#
# Variables (set before sourcing): MEM (mig_native | litedram), DDR_SPEED (mig_native: 800 |
# 1066 | 1300 | 1333 | 1600), MIG_DIR (the .prj files, native interface: gen_mig_prj.py --native),
# CORE_MHZ.

if {![info exists MEM]} { set MEM mig_native }
if {$MEM ni {mig_native litedram}} { error "bd_native.tcl: MEM is $MEM (mig_native or litedram)" }
set mig [expr {$MEM eq "mig_native"}]
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

# core clock: CORE_MHZ rounded to the MMCM's 1/8 divider steps of its VCO (as bd.tcl)
if {![info exists CORE_MHZ]} { set CORE_MHZ 100 }
set VCO [expr {$mig && $DDR_SPEED == 1333 ? 1000 : 800}]
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
if {$mig} { create_bd_port -dir O -from 1 -to 0 calib }

set lites [expr {$mig ? {M_AXI_CTL} : {M_AXI_CTL M_AXI_MEMCAL}}]
foreach p $lites {
  set m [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 $p]
  set_property -dict [list CONFIG.PROTOCOL AXI4LITE CONFIG.DATA_WIDTH 32 CONFIG.ADDR_WIDTH 32 \
    CONFIG.HAS_BURST 0 CONFIG.HAS_LOCK 0 CONFIG.HAS_PROT 0 CONFIG.HAS_CACHE 0 CONFIG.HAS_QOS 0 \
    CONFIG.HAS_REGION 0] $m
}
if {!$mig} { set_property CONFIG.FREQ_HZ 125000000 [get_bd_intf_ports M_AXI_MEMCAL] }
set_property CONFIG.ASSOCIATED_BUSIF {M_AXI_CTL} [get_bd_ports core_clk]
set_property CONFIG.ASSOCIATED_RESET {core_rstn} [get_bd_ports core_clk]
set_property CONFIG.FREQ_HZ $CORE_HZ [get_bd_ports core_clk]

# ------------------------------------------------------------------ clocks and resets
set clk [create_bd_cell -type ip -vlnv [ip_vlnv clk_wiz] clk_wiz_0]
if {$mig} {
  set_property -dict [list \
    CONFIG.PRIM_IN_FREQ {50.000} CONFIG.PRIM_SOURCE {Single_ended_clock_capable_pin} \
    CONFIG.USE_RESET {false} CONFIG.USE_LOCKED {true} \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $CORE_MHZ_ACT \
    CONFIG.CLKOUT2_USED {true} CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {200.000} \
    CONFIG.CLKOUT3_USED {true} CONFIG.CLKOUT3_REQUESTED_OUT_FREQ [format %.3f [expr {$VCO / 3.0}]] \
    CONFIG.NUM_OUT_CLKS {3} CONFIG.MMCM_DIVCLK_DIVIDE {1} \
    CONFIG.MMCM_CLKFBOUT_MULT_F [format %.3f [expr {$VCO / 50.0}]] \
    CONFIG.MMCM_CLKOUT0_DIVIDE_F $CORE_DIV CONFIG.MMCM_CLKOUT1_DIVIDE [expr {$VCO / 200}] \
    CONFIG.MMCM_CLKOUT2_DIVIDE {3} \
    CONFIG.CLK_OUT1_PORT {core_clk} CONFIG.CLK_OUT2_PORT {clk_200} CONFIG.CLK_OUT3_PORT {clk_mig} \
  ] $clk
} else {
  set_property -dict [list \
    CONFIG.PRIM_IN_FREQ {50.000} CONFIG.PRIM_SOURCE {No_buffer} \
    CONFIG.USE_RESET {false} CONFIG.USE_LOCKED {true} \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $CORE_MHZ_ACT CONFIG.NUM_OUT_CLKS {1} \
    CONFIG.MMCM_DIVCLK_DIVIDE {1} \
    CONFIG.MMCM_CLKFBOUT_MULT_F [format %.3f [expr {$VCO / 50.0}]] \
    CONFIG.MMCM_CLKOUT0_DIVIDE_F $CORE_DIV CONFIG.CLK_OUT1_PORT {core_clk} \
  ] $clk
}
connect_bd_net [get_bd_ports sys_clk_50] [get_bd_pins clk_wiz_0/clk_in1]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_ports core_clk]

# core reset: until the MMCM locks, and while the host asserts PCIe PERST# (as bd.tcl)
set rst_core [create_bd_cell -type ip -vlnv [ip_vlnv proc_sys_reset] rst_core]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins rst_core/slowest_sync_clk]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins rst_core/dcm_locked]
connect_bd_net [get_bd_ports pcie_perstn] [get_bd_pins rst_core/ext_reset_in]
connect_bd_net [get_bd_pins rst_core/peripheral_aresetn] [get_bd_ports core_rstn]

# ------------------------------------------------------------------ PCIe: XDMA, Gen1 x8
# The production settings and identity, exactly as bd.tcl (the host tells the memory build from
# CAPS, not from the PCI identity).
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
  CONFIG.xdma_rnum_chnl {1} CONFIG.xdma_wnum_chnl {1} \
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
set_property CONFIG.ASSOCIATED_BUSIF [expr {$mig ? "M_AXI_DMA" : "M_AXI_DMA:M_AXI_MEMCAL"}] [get_bd_ports xdma_aclk]
set_property CONFIG.ASSOCIATED_RESET {xdma_aresetn} [get_bd_ports xdma_aclk]
set_property CONFIG.FREQ_HZ 125000000 [get_bd_ports xdma_aclk]

# ------------------------------------------------------------------ DDR3: two MIGs, native (mig_native)
# As bd.tcl's MIGs (the same .prj flow, clocks, sys_rst and XADC), with the native user
# interface: its pins, the user clock and its reset leave the design as mig<c>_<pin>. The MIGs'
# app_sr_req / app_ref_req / app_zq_req stay unconnected, as in bd.tcl (IP integrator ties them
# low); app_rd_data_end, the ECC error flags and the self-refresh / refresh / ZQ acknowledges are
# not used.
set mig_pins {app_addr app_cmd app_en app_rdy app_wdf_data app_wdf_mask app_wdf_wren app_wdf_end
              app_wdf_rdy app_rd_data app_rd_data_valid ui_clk ui_clk_sync_rst}
if {$mig} {
  set sysclk [expr {$DDR_SPEED in {1066 1333} ? "clk_wiz_0/clk_mig" : "clk_wiz_0/clk_200"}]
  foreach ch {0 1} {
    set m [create_bd_cell -type ip -vlnv [ip_vlnv mig_7series] mig_$ch]
    set dir [get_property IP_DIR [get_ips [get_property CONFIG.Component_Name $m]]]
    file copy -force [file join $MIG_DIR mig_ddr3_ch${ch}.prj] [file join $dir mig_ddr3_ch${ch}.prj]
    set_property -dict [list CONFIG.BOARD_MIG_PARAM {Custom} CONFIG.MIG_DONT_TOUCH_PARAM {Custom} \
      CONFIG.RESET_BOARD_INTERFACE {Custom} CONFIG.XML_INPUT_FILE mig_ddr3_ch${ch}.prj] $m
    connect_bd_net [get_bd_pins $sysclk] [get_bd_pins mig_$ch/sys_clk_i]
    connect_bd_net [get_bd_pins clk_wiz_0/clk_200] [get_bd_pins mig_$ch/clk_ref_i]
    connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins mig_$ch/sys_rst]
    make_bd_intf_pins_external -name DDR3_$ch [get_bd_intf_pins mig_$ch/DDR3]
    if {[get_bd_intf_pins -quiet mig_$ch/S_AXI] ne ""} {
      error "mig_$ch has an AXI interface: run gen_mig_prj.py --native (MEM=mig_native)"
    }
    foreach p $mig_pins {
      set pin [get_bd_pins -quiet mig_$ch/$p]
      if {$pin eq ""} {
        error "mig_$ch has no pin $p; its pins: [lsort [get_bd_pins -of_objects [get_bd_cells mig_$ch]]]"
      }
      make_bd_pins_external -name mig${ch}_$p $pin
    }
  }
}

# ------------------------------------------------------------------ XADC (die temperature)
# As bd.tcl: temp_out for the accelerator's TEMP register (and the MIGs' temperature-compensated
# read timing), the XADC registers at BAR0 0x30000.
set xadc [create_bd_cell -type ip -vlnv [ip_vlnv xadc_wiz] xadc_temp]
set_property -dict [list CONFIG.INTERFACE_SELECTION {Enable_AXI} CONFIG.DCLK_FREQUENCY {100} \
  CONFIG.XADC_STARUP_SELECTION {single_channel} CONFIG.SINGLE_CHANNEL_SELECTION {TEMPERATURE} \
  CONFIG.TIMING_MODE {Continuous} CONFIG.ENABLE_TEMP_BUS {true} \
  CONFIG.OT_ALARM {false} CONFIG.USER_TEMP_ALARM {false} CONFIG.VCCINT_ALARM {false} \
  CONFIG.VCCAUX_ALARM {false} CONFIG.CHANNEL_ENABLE_VP_VN {false}] $xadc
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins xadc_temp/s_axi_aclk]
connect_bd_net [get_bd_pins rst_core/peripheral_aresetn] [get_bd_pins xadc_temp/s_axi_aresetn]
if {$mig} {
  connect_bd_net [get_bd_pins xadc_temp/temp_out] [get_bd_pins mig_0/device_temp_i] \
    [get_bd_pins mig_1/device_temp_i] [get_bd_ports device_temp]
  set cat [create_bd_cell -type ip -vlnv [ip_vlnv xlconcat] calib_cat]
  set_property CONFIG.NUM_PORTS {2} $cat
  connect_bd_net [get_bd_pins mig_0/init_calib_complete] [get_bd_pins calib_cat/In0]
  connect_bd_net [get_bd_pins mig_1/init_calib_complete] [get_bd_pins calib_cat/In1]
  connect_bd_net [get_bd_pins calib_cat/dout] [get_bd_ports calib]
} else {
  connect_bd_net [get_bd_pins xadc_temp/temp_out] [get_bd_ports device_temp]
}

# ------------------------------------------------------------------ control interconnect
# XDMA's AXI-Lite master (BAR0): the control registers (into core_clk), the XADC (core_clk) and,
# for litedram, the LiteDRAM CSRs (in xdma_aclk)
set scl [create_bd_cell -type ip -vlnv [ip_vlnv smartconnect] sc_ctl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI [expr {$mig ? 2 : 3}] CONFIG.NUM_CLKS {2}] $scl
connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_pins sc_ctl/aclk]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins sc_ctl/aclk1]
connect_bd_net [get_bd_pins xdma_0/axi_aresetn] [get_bd_pins sc_ctl/aresetn]
connect_bd_intf_net [get_bd_intf_pins xdma_0/M_AXI_LITE] [get_bd_intf_pins sc_ctl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M00_AXI] [get_bd_intf_ports M_AXI_CTL]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M01_AXI] [get_bd_intf_pins xadc_temp/s_axi_lite]
if {!$mig} { connect_bd_intf_net [get_bd_intf_pins sc_ctl/M02_AXI] [get_bd_intf_ports M_AXI_MEMCAL] }

# ------------------------------------------------------------------ address map
# BAR0: 0x0 the control registers, 0x10000 the LiteDRAM CSRs (litedram: R_MEMCAL; both
# channels', by csr.csv), 0x30000 the XADC. XDMA's DMA: the whole 32-bit space to M_AXI_DMA
# (channel 0 at 0x0000_0000, channel 1 at 0x8000_0000, 2 GiB each: opentpu/host/board.py)
set lite [get_bd_addr_spaces xdma_0/M_AXI_LITE]
assign_bd_address -offset 0x00000000 -range 64K -target_address_space $lite [get_bd_addr_segs M_AXI_CTL/Reg] -force
if {!$mig} {
  assign_bd_address -offset 0x00010000 -range 64K -target_address_space $lite [get_bd_addr_segs M_AXI_MEMCAL/Reg] -force
}
assign_bd_address -offset 0x00030000 -range 64K -target_address_space $lite \
  [get_bd_addr_segs -of [get_bd_intf_pins xadc_temp/s_axi_lite]] -force
assign_bd_address -offset 0x00000000 -range 4G -target_address_space [get_bd_addr_spaces xdma_0/M_AXI] \
  [get_bd_addr_segs M_AXI_DMA/Reg] -force

validate_bd_design

# the external ports the top connects to (IP integrator renames a clashing port silently)
set want [concat {pcie_mgt pcie_refclk M_AXI_DMA} $lites [expr {$mig ? {DDR3_0 DDR3_1} : {}}]]
foreach p $want {
  if {[get_bd_intf_ports $p] eq ""} { error "block design port $p missing: [get_bd_intf_ports]" }
}
if {$mig} {
  foreach ch {0 1} {
    foreach p $mig_pins {
      if {[get_bd_ports -quiet mig${ch}_$p] eq ""} { error "block design port mig${ch}_$p missing" }
    }
  }
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
