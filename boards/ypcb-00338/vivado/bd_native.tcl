# Block design "otpu_bd" for the YPCB-00338 (create_project.tcl): the core clock, resets, the XADC
# and the control interconnect. No PCIe core and no memory controllers: LitePCIe
# (otpu_litepcie, tools/litepcie/gen_core.py) and the LiteDRAM core sit in the top
# (otpu_fpga_top_ld), and the accelerator and the host's DMA (otpu_pcie_mm) meet on each channel
# in otpu_mem_ch (otpu_native_sys). External interfaces:
#   S_AXI_CTL     AXI4-Lite slave, LitePCIe's BAR0 master (its user window, BAR0 0x00000-0x7FFFF;
#                 pcie_clk, the PCIe user clock)
#   M_AXI_CTL     AXI4-Lite master -> the accelerator's control registers (BAR0 0x0000, core_clk)
#   M_AXI_MEMCAL  AXI4-Lite master -> the LiteDRAM core's CSRs (BAR0 0x10000, 64 KB, core_clk: the
#                 core crosses them into its own clock; opentpu/host/memcal.py)
# and device_temp (the XADC die-temperature code, core_clk; BAR0 0x30000: the XADC registers).
#
# Clocks: the 50 MHz clock arrives already on a global buffer (otpu_fpga_top_ld.sv: the LiteDRAM
# core's MMCMs share it), so the clock wizard has no input buffer and makes core_clk only.
# pcie_clk 250 MHz (LitePCIe's, Gen2 x8 at 128 bits), from the top.
#
# Variables (set before sourcing): CORE_MHZ, CLK2X (below). PCIE_GEN (environment): 2, Gen2 x8,
# the only setting (the XDMA builds' Gen1 x8, PCIE_GEN 1, went with XDMA).

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
# CLK2X (set before sourcing, default 0; create_project.tcl: 1 with the TMEM's pumped RAMs,
# TMEM_PMP 1): core_clk2x, the MMCM's second output at twice core_clk from the same VCO, both
# dividers integers and no phase shift, so its edges rise with core_clk's (core_clk's divider must
# then be even). With CLK2X 0 the port is core_clk itself: the wrapper keeps one form, and the RTL
# leaves it unused (otpu_fpga_top_*: clk_f). constraints/otpu_clk2x.tcl times the paths between them.
if {![info exists CLK2X]} { set CLK2X 0 }
if {$CLK2X && ($CORE_DIV != floor($CORE_DIV) || int($CORE_DIV) % 2)} {
  error "bd_native.tcl: CLK2X needs an even MMCM divide for core_clk, not $CORE_DIV ($CORE_MHZ_ACT MHz; 133.33: 6, 100: 8)"
}
set CLK2X_HZ [expr {$CLK2X ? int(floor($VCO * 1.0e6 / ($CORE_DIV / 2))) : $CORE_HZ}]
if {$CLK2X} { puts "core_clk2x: [expr {$CLK2X_HZ / 1.0e6}] MHz (MMCM divide [expr {int($CORE_DIV) / 2}])" }
set PCIE_GEN [expr {[info exists ::env(PCIE_GEN)] && $::env(PCIE_GEN) ne "" ? $::env(PCIE_GEN) : 2}]
if {$PCIE_GEN != 2} { error "bd_native.tcl: PCIE_GEN $PCIE_GEN: LitePCIe's PHY here is Gen2 x8 (2) only" }
set PCIE_HZ 250000000
puts "PCIe: LitePCIe, Gen2 x8, pcie_clk 250 MHz"

# ------------------------------------------------------------------ external ports
create_bd_port -dir I -type clk -freq_hz 50000000 sys_clk_50
create_bd_port -dir I -type rst pcie_perstn
set_property CONFIG.POLARITY ACTIVE_LOW [get_bd_ports pcie_perstn]
create_bd_port -dir I -type clk -freq_hz $PCIE_HZ pcie_clk
create_bd_port -dir I -type rst pcie_rstn
set_property CONFIG.POLARITY ACTIVE_LOW [get_bd_ports pcie_rstn]
create_bd_port -dir O -type clk core_clk
create_bd_port -dir O -type clk core_clk2x
create_bd_port -dir O -type rst core_rstn
create_bd_port -dir O -from 11 -to 0 device_temp

set lites {M_AXI_CTL M_AXI_MEMCAL}
foreach p $lites {
  set m [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 $p]
  set_property -dict [list CONFIG.PROTOCOL AXI4LITE CONFIG.DATA_WIDTH 32 CONFIG.ADDR_WIDTH 32 \
    CONFIG.HAS_BURST 0 CONFIG.HAS_LOCK 0 CONFIG.HAS_PROT 0 CONFIG.HAS_CACHE 0 CONFIG.HAS_QOS 0 \
    CONFIG.HAS_REGION 0] $m
}
# LitePCIe's BAR0 master (AXI-Lite with AxPROT)
set s_ctl [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_AXI_CTL]
set_property -dict [list CONFIG.PROTOCOL AXI4LITE CONFIG.DATA_WIDTH 32 CONFIG.ADDR_WIDTH 32 \
  CONFIG.HAS_BURST 0 CONFIG.HAS_LOCK 0 CONFIG.HAS_PROT 1 CONFIG.HAS_CACHE 0 CONFIG.HAS_QOS 0 \
  CONFIG.HAS_REGION 0 CONFIG.FREQ_HZ $PCIE_HZ] $s_ctl
set_property CONFIG.ASSOCIATED_BUSIF {S_AXI_CTL} [get_bd_ports pcie_clk]
set_property CONFIG.ASSOCIATED_RESET {pcie_rstn} [get_bd_ports pcie_clk]
# the LiteDRAM CSRs in core_clk, the SmartConnect crossing them as it does the control registers
# (otpu_fpga_top_ld.sv: the core's ctl_clk)
set_property CONFIG.FREQ_HZ $CORE_HZ [get_bd_intf_ports M_AXI_MEMCAL]
set_property CONFIG.ASSOCIATED_BUSIF {M_AXI_CTL:M_AXI_MEMCAL} [get_bd_ports core_clk]
set_property CONFIG.ASSOCIATED_RESET {core_rstn} [get_bd_ports core_clk]
set_property CONFIG.FREQ_HZ $CORE_HZ [get_bd_ports core_clk]
set_property CONFIG.FREQ_HZ $CLK2X_HZ [get_bd_ports core_clk2x]

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
# CLK2X: the second output, its divider half core_clk's; the MMCM's settings always overridden
# (VCO 800 MHz), so both come from the same VCO by integer dividers
if {$CLK2X} {
  set_property -dict [list CONFIG.NUM_OUT_CLKS {2} CONFIG.CLKOUT2_USED {true} \
    CONFIG.CLKOUT2_REQUESTED_OUT_FREQ [format %.3f [expr {$CLK2X_HZ / 1.0e6}]] \
    CONFIG.CLKOUT2_REQUESTED_PHASE {0.000} CONFIG.CLK_OUT2_PORT {core_clk2x}] $clk
  set_property -dict [list CONFIG.OVERRIDE_MMCM {true} CONFIG.MMCM_DIVCLK_DIVIDE {1} \
    CONFIG.MMCM_CLKFBOUT_MULT_F [format %.3f [expr {$VCO / 50.0}]] \
    CONFIG.MMCM_CLKOUT0_DIVIDE_F $CORE_DIV CONFIG.MMCM_CLKOUT1_DIVIDE [expr {int($CORE_DIV) / 2}]] $clk
  connect_bd_net [get_bd_pins clk_wiz_0/core_clk2x] [get_bd_ports core_clk2x]
  set f2 [get_property CONFIG.FREQ_HZ [get_bd_pins clk_wiz_0/core_clk2x]]
  if {$f2 != $CLK2X_HZ} { error "bd_native.tcl: the clock wizard makes core_clk2x at $f2 Hz, not $CLK2X_HZ" }
} else {
  connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_ports core_clk2x]
}
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

# ------------------------------------------------------------------ XADC (die temperature)
# temp_out for the accelerator's TEMP register, the XADC registers at BAR0 0x30000. Its DCLK is
# core_clk: the wizard divides it down to ADCCLK (at most 26 MHz, DS182) from DCLK_FREQUENCY, so
# that is core_clk's (in whole MHz, rounded up: the divider errs high). The fixed 100 gave a
# divider of 4: 33 MHz at core_clk 133.33 MHz, out of spec (build.tcl checks the divider the
# netlist has).
set XADC_DCLK_MHZ [expr {int(ceil($CORE_HZ / 1.0e6))}]
set xadc [create_bd_cell -type ip -vlnv [ip_vlnv xadc_wiz] xadc_temp]
set_property -dict [list CONFIG.INTERFACE_SELECTION {Enable_AXI} CONFIG.DCLK_FREQUENCY $XADC_DCLK_MHZ \
  CONFIG.XADC_STARUP_SELECTION {single_channel} CONFIG.SINGLE_CHANNEL_SELECTION {TEMPERATURE} \
  CONFIG.TIMING_MODE {Continuous} CONFIG.ENABLE_TEMP_BUS {true} \
  CONFIG.OT_ALARM {false} CONFIG.USER_TEMP_ALARM {false} CONFIG.VCCINT_ALARM {false} \
  CONFIG.VCCAUX_ALARM {false} CONFIG.CHANNEL_ENABLE_VP_VN {false}] $xadc
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins xadc_temp/s_axi_aclk]
connect_bd_net [get_bd_pins rst_core/peripheral_aresetn] [get_bd_pins xadc_temp/s_axi_aresetn]
connect_bd_net [get_bd_pins xadc_temp/temp_out] [get_bd_ports device_temp]

# ------------------------------------------------------------------ control interconnect
# LitePCIe's BAR0 master: the control registers (into core_clk), the XADC (core_clk) and the
# LiteDRAM CSRs (core_clk)
set scl [create_bd_cell -type ip -vlnv [ip_vlnv smartconnect] sc_ctl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {3} CONFIG.NUM_CLKS {2}] $scl
connect_bd_net [get_bd_ports pcie_clk] [get_bd_pins sc_ctl/aclk]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins sc_ctl/aclk1]
connect_bd_net [get_bd_ports pcie_rstn] [get_bd_pins sc_ctl/aresetn]
connect_bd_intf_net [get_bd_intf_ports S_AXI_CTL] [get_bd_intf_pins sc_ctl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M00_AXI] [get_bd_intf_ports M_AXI_CTL]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M01_AXI] [get_bd_intf_pins xadc_temp/s_axi_lite]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M02_AXI] [get_bd_intf_ports M_AXI_MEMCAL]

# ------------------------------------------------------------------ address map
# BAR0's user window: 0x0 the control registers, 0x10000 the LiteDRAM CSRs (R_MEMCAL; both
# channels', by csr.csv), 0x30000 the XADC. LitePCIe's own CSRs (0x80000 up) never reach here.
set lite [get_bd_addr_spaces S_AXI_CTL]
assign_bd_address -offset 0x00000000 -range 64K -target_address_space $lite [get_bd_addr_segs M_AXI_CTL/Reg] -force
assign_bd_address -offset 0x00010000 -range 64K -target_address_space $lite [get_bd_addr_segs M_AXI_MEMCAL/Reg] -force
assign_bd_address -offset 0x00030000 -range 64K -target_address_space $lite \
  [get_bd_addr_segs -of [get_bd_intf_pins xadc_temp/s_axi_lite]] -force

validate_bd_design

# the external ports the top connects to (IP integrator renames a clashing port silently)
foreach p [concat {S_AXI_CTL} $lites] {
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
save_bd_design
