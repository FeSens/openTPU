# Block design "otpu_bdm" of the BCU1525's memory-test image (create_project_mem.tcl; the top
# rtl/boards/bcu1525/otpu_fpga_top_bcu_mem.sv; docs/bcu1525.md section 14): no PCIe. A JTAG-to-AXI
# master (jtag_axi, AXI4-Lite) is the host's way in, over the board's FT4232H and Vivado's hardware
# server; the control clock; the SYSMON with its alarms; the interconnect at the full image's BAR0
# offsets:
#   M_AXI_GOV       0x10000  the thermal governor (otpu_bcu_gov, 64 KB)
#   M_AXI_BMC       0x20000  the board management controller's UART bridge (otpu_bmc_uart)
#   (the SYSMON)    0x30000  its registers
#   M_AXI_MEMCAL<d> 0x40000 + d x 0x10000  DIMM C<d>'s LiteDRAM core's CSRs
# and device_temp (the die's 10-bit code), temp_alarm (the SYSMON's temperature alarm ALM[0]: up at
# 80 C, down at 70 C) and ot_alarm (its over-temperature alarm: up at 84 C, which with the
# bitstream's OVERTEMPSHUTDOWN shuts the FPGA down about 10 ms later; down at 65 C, below which it
# can be configured again), all in core_clk. The governor trips at 80 C from the code by itself.
#
# Clocks: core_clk 100 MHz from clk300_in (DIMM C1's 300 MHz, on a global buffer in the top);
# core_locked its MMCM's lock out, core_rstn the top's reset of the AXI side in.

proc ip_vlnv {name} {
  set defs [lsort -decreasing [get_ipdefs -all "xilinx.com:ip:${name}:*"]]
  if {[llength $defs] == 0} { error "IP $name not found in this Vivado installation" }
  return [lindex $defs 0]
}

create_bd_design otpu_bdm
current_bd_design otpu_bdm

set CORE_HZ 100000000
set OT_UP 84.0
set OT_DOWN 65.0
set ALM_UP 80.0
set ALM_DOWN 70.0

# ------------------------------------------------------------------ external ports
create_bd_port -dir I -type clk -freq_hz 300000000 clk300_in
create_bd_port -dir O -type clk core_clk
create_bd_port -dir O core_locked
create_bd_port -dir I -type rst core_rstn
set_property CONFIG.POLARITY ACTIVE_LOW [get_bd_ports core_rstn]
create_bd_port -dir O -from 9 -to 0 device_temp
create_bd_port -dir O temp_alarm
create_bd_port -dir O ot_alarm
set lites {M_AXI_GOV M_AXI_BMC M_AXI_MEMCAL0 M_AXI_MEMCAL1 M_AXI_MEMCAL2 M_AXI_MEMCAL3}
foreach p $lites {
  set m [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 $p]
  set_property -dict [list CONFIG.PROTOCOL AXI4LITE CONFIG.DATA_WIDTH 32 CONFIG.ADDR_WIDTH 32 \
    CONFIG.HAS_BURST 0 CONFIG.HAS_LOCK 0 CONFIG.HAS_PROT 0 CONFIG.HAS_CACHE 0 CONFIG.HAS_QOS 0 \
    CONFIG.HAS_REGION 0 CONFIG.FREQ_HZ $CORE_HZ] $m
}
set_property CONFIG.ASSOCIATED_BUSIF [join $lites :] [get_bd_ports core_clk]
set_property CONFIG.ASSOCIATED_RESET {core_rstn} [get_bd_ports core_clk]
set_property CONFIG.FREQ_HZ $CORE_HZ [get_bd_ports core_clk]

# ------------------------------------------------------------------ clock and reset
set clk [create_bd_cell -type ip -vlnv [ip_vlnv clk_wiz] clk_wiz_0]
set_property -dict [list \
  CONFIG.PRIM_IN_FREQ {300.000} CONFIG.PRIM_SOURCE {Global_buffer} \
  CONFIG.USE_RESET {false} CONFIG.USE_LOCKED {true} \
  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {100.000} CONFIG.NUM_OUT_CLKS {1} \
  CONFIG.CLK_OUT1_PORT {core_clk} \
] $clk
connect_bd_net [get_bd_ports clk300_in] [get_bd_pins clk_wiz_0/clk_in1]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_ports core_clk]
set f [get_property CONFIG.FREQ_HZ [get_bd_pins clk_wiz_0/core_clk]]
if {$f != $CORE_HZ} { error "bd_mem.tcl: the clock wizard makes core_clk at $f Hz, not $CORE_HZ" }

# the reset is the top's (otpu_fpga_top_bcu_mem: core_locked through a shift register on core_clk),
# not proc_sys_reset's: its external reset's polarity follows what drives it, and a constant gives
# it none to follow
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_ports core_locked]

# ------------------------------------------------------------------ JTAG-to-AXI
set jtag [create_bd_cell -type ip -vlnv [ip_vlnv jtag_axi] jtag_axi_0]
set_property -dict [list CONFIG.PROTOCOL {2} CONFIG.M_AXI_ADDR_WIDTH {32} \
  CONFIG.M_AXI_DATA_WIDTH {32}] $jtag
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins jtag_axi_0/aclk]
connect_bd_net [get_bd_ports core_rstn] [get_bd_pins jtag_axi_0/aresetn]

# ------------------------------------------------------------------ SYSMON
# Continuous sequencer: in Default mode the wizard fixes the temperature alarm at 85 / 60 C and
# gives no user_temp_alarm_out
set sysmon [create_bd_cell -type ip -vlnv [ip_vlnv system_management_wiz] sysmon]
set_property -dict [list CONFIG.INTERFACE_SELECTION {Enable_AXI} CONFIG.DCLK_FREQUENCY {100} \
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
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins sysmon/s_axi_aclk]
connect_bd_net [get_bd_ports core_rstn] [get_bd_pins sysmon/s_axi_aresetn]
connect_bd_net [get_bd_pins sysmon/temp_out] [get_bd_ports device_temp]
foreach {pin port} {ot_out ot_alarm user_temp_alarm_out temp_alarm} {
  if {[get_bd_pins -quiet sysmon/$pin] eq ""} {
    error "sysmon: no pin $pin (its pins: [get_bd_pins -of [get_bd_cells sysmon]])"
  }
  connect_bd_net [get_bd_pins sysmon/$pin] [get_bd_ports $port]
}

# ------------------------------------------------------------------ interconnect
set sc [create_bd_cell -type ip -vlnv [ip_vlnv smartconnect] sc_ctl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {7} CONFIG.NUM_CLKS {1}] $sc
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins sc_ctl/aclk]
connect_bd_net [get_bd_ports core_rstn] [get_bd_pins sc_ctl/aresetn]
connect_bd_intf_net [get_bd_intf_pins jtag_axi_0/M_AXI] [get_bd_intf_pins sc_ctl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M00_AXI] [get_bd_intf_ports M_AXI_GOV]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M01_AXI] [get_bd_intf_ports M_AXI_BMC]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M02_AXI] [get_bd_intf_pins sysmon/S_AXI_LITE]
foreach d {0 1 2 3} {
  connect_bd_intf_net [get_bd_intf_pins sc_ctl/M0[expr {$d + 3}]_AXI] [get_bd_intf_ports M_AXI_MEMCAL$d]
}

# ------------------------------------------------------------------ address map
set sp [get_bd_addr_spaces jtag_axi_0/Data]
assign_bd_address -offset 0x00010000 -range 64K -target_address_space $sp [get_bd_addr_segs M_AXI_GOV/Reg] -force
assign_bd_address -offset 0x00020000 -range 64K -target_address_space $sp [get_bd_addr_segs M_AXI_BMC/Reg] -force
assign_bd_address -offset 0x00030000 -range 64K -target_address_space $sp \
  [get_bd_addr_segs -of [get_bd_intf_pins sysmon/S_AXI_LITE]] -force
foreach d {0 1 2 3} {
  assign_bd_address -offset [format 0x%08X [expr {0x40000 + $d * 0x10000}]] -range 64K \
    -target_address_space $sp [get_bd_addr_segs M_AXI_MEMCAL$d/Reg] -force
}

validate_bd_design
foreach p $lites {
  if {[get_bd_intf_ports $p] eq ""} { error "block design port $p missing: [get_bd_intf_ports]" }
}
save_bd_design
