# openTPU on the Inspur YPCB-00338 with LiteDRAM (create_project.tcl; top otpu_fpga_top_ld):
# board pins, clocks, configuration (the MIG builds' otpu_top.xdc, less the MIG's lines, until
# those builds were removed); the DDR3 pins, I/O standards and VREF are the LiteDRAM core's
# (boards/ypcb-00338/litedram/otpu_litedram.xdc), the clock-domain crossings that need the
# implemented netlist are in otpu_top_native.tcl (unmanaged, read late) and otpu_mem_ch.tcl.

# ---------------------------------------------------------------- 50 MHz oscillator
set_property PACKAGE_PIN AA28 [get_ports SYS_CLK]
set_property IOSTANDARD LVCMOS18 [get_ports SYS_CLK]
create_clock -name sys_clk_50 -period 20.000 [get_ports SYS_CLK]

# ---------------------------------------------------------------- LEDs
set_property PACKAGE_PIN P30 [get_ports {led[0]}]
set_property PACKAGE_PIN M30 [get_ports {led[1]}]
set_property PACKAGE_PIN N30 [get_ports {led[2]}]
set_property IOSTANDARD LVCMOS18 [get_ports {led[*]}]
set_false_path -to [get_ports {led[*]}]

# ---------------------------------------------------------------- I2C (open drain)
# Two buses the host bit-bangs through otpu_ctrl's I2C_CTRL / I2C_IN (opentpu/host/i2c.py):
# the LM73 temperature sensor's bus (SCL N24, SDA N25, ALERT P25) and the PCIe edge
# connector's SMBus (SCL R26, SDA R27), shared with the host's SMBus master. Pins from the
# vendor XDC; the board's pull-ups are not documented, so the weak internal PULLUP is a
# backup. The pins are released (high-Z) unless the host sets a drive-low bit.
set_property PACKAGE_PIN N24 [get_ports lm73_scl]
set_property PACKAGE_PIN N25 [get_ports lm73_sda]
set_property PACKAGE_PIN P25 [get_ports lm73_alert_n]
set_property PACKAGE_PIN R26 [get_ports smb_scl]
set_property PACKAGE_PIN R27 [get_ports smb_sda]
set_property IOSTANDARD LVCMOS18 [get_ports {lm73_scl lm73_sda lm73_alert_n smb_scl smb_sda}]
set_property PULLUP true [get_ports {lm73_scl lm73_sda lm73_alert_n smb_scl smb_sda}]
set_property DRIVE 4 [get_ports {lm73_scl lm73_sda smb_scl smb_sda}]
set_property SLEW SLOW [get_ports {lm73_scl lm73_sda smb_scl smb_sda}]
set_false_path -to [get_ports {lm73_scl lm73_sda smb_scl smb_sda}]
set_false_path -from [get_ports {lm73_scl lm73_sda lm73_alert_n smb_scl smb_sda}]

# ---------------------------------------------------------------- PCIe
# Reference clock (100 MHz from the slot) on MGTREFCLK0_116 (J8). The lanes are on GT banks 116
# and 115: lane 0..3 = TX F2 H2 K2 M2 = MGTXTXP3..0_116, lane 4..7 = N4 P2 T2 U4 =
# MGTXTXP3..0_115, i.e. lane i on GTXE2_CHANNEL_X0Y(23 - i). The XDMA's own constraints place
# the lanes one quad lower (X0Y19..X0Y12, banks 115/114), which is also illegal with the
# refclk in bank 116; these LOCs (user XDC, applied after the IP's) move them onto the card's
# pins. (select_quad is an UltraScale option; 7-series XDMA ignores it.)
set_property LOC GTXE2_CHANNEL_X0Y23 [get_cells -hier -filter {NAME =~ *pipe_lane?0?.gt_wrapper_i/gtx_channel.gtxe2_channel_i}]
set_property LOC GTXE2_CHANNEL_X0Y22 [get_cells -hier -filter {NAME =~ *pipe_lane?1?.gt_wrapper_i/gtx_channel.gtxe2_channel_i}]
set_property LOC GTXE2_CHANNEL_X0Y21 [get_cells -hier -filter {NAME =~ *pipe_lane?2?.gt_wrapper_i/gtx_channel.gtxe2_channel_i}]
set_property LOC GTXE2_CHANNEL_X0Y20 [get_cells -hier -filter {NAME =~ *pipe_lane?3?.gt_wrapper_i/gtx_channel.gtxe2_channel_i}]
set_property LOC GTXE2_CHANNEL_X0Y19 [get_cells -hier -filter {NAME =~ *pipe_lane?4?.gt_wrapper_i/gtx_channel.gtxe2_channel_i}]
set_property LOC GTXE2_CHANNEL_X0Y18 [get_cells -hier -filter {NAME =~ *pipe_lane?5?.gt_wrapper_i/gtx_channel.gtxe2_channel_i}]
set_property LOC GTXE2_CHANNEL_X0Y17 [get_cells -hier -filter {NAME =~ *pipe_lane?6?.gt_wrapper_i/gtx_channel.gtxe2_channel_i}]
set_property LOC GTXE2_CHANNEL_X0Y16 [get_cells -hier -filter {NAME =~ *pipe_lane?7?.gt_wrapper_i/gtx_channel.gtxe2_channel_i}]
set_property PACKAGE_PIN J8 [get_ports pcie_refclk_clk_p]
create_clock -name pcie_refclk -period 10.000 [get_ports pcie_refclk_clk_p]
set_property PACKAGE_PIN Y26 [get_ports pcie_perstn]
set_property IOSTANDARD LVCMOS18 [get_ports pcie_perstn]
set_property PULLUP true [get_ports pcie_perstn]
set_false_path -from [get_ports pcie_perstn]

# ---------------------------------------------------------------- clock domain crossings
# The accelerator synchronizes the die-temperature code and the I2C pin levels
# itself (2 flip-flops per bit, ASYNC_REG; the temperature is taken only when two samples
# agree). The calibration flags (the LiteDRAM core's c0_ready / c1_ready, in its sys clock) get
# a max delay instead of a false path, in otpu_top_native.tcl: sys and core_clk are also one of
# otpu_mem_ch's clock pairs, and nothing here may make that pair asynchronous. The control
# registers cross from xdma_aclk to core_clk in the SmartConnect, which carries its own
# constraints.
set_false_path -to [get_cells -hier -filter {NAME =~ *u_board/tmp_s1_reg*}]
set_false_path -to [get_cells -hier -filter {NAME =~ *u_board/i2c_s1_reg*}]

# The 50 MHz clock reaches the block design's MMCM (core_clk) and the LiteDRAM core's sys MMCM
# and IDELAY reference PLL from one BUFG (otpu_fpga_top_ld.sv), which the AA28 pin's IBUF drives
# directly. Each channel's write clock MMCM (WL7DDRPHY's WriteClocks, cascaded from sys; the
# core's XDC has their crossings' uncertainty and the serializer resets' max delay) sits in its
# own banks' clock regions, as on the card images (tools/litedram/ld_test.py --mmcm-locs,
# docs/litedram.md section 8): X0Y2 (bank 13) for channel 0, X0Y6 (bank 17, the command bank) for
# channel 1. Unplaced, the placer put each among the other channel's banks.
set_property LOC MMCME2_ADV_X0Y2 [get_cells u_ld/ldmmcm0]
set_property LOC MMCME2_ADV_X0Y6 [get_cells u_ld/ldmmcm1]

# ---------------------------------------------------------------- configuration
# Configuration banks at 1.8 V (the board's flash and IO are LVCMOS18); BPI x16 flash.
set_property CFGBVS GND [current_design]
set_property CONFIG_VOLTAGE 1.8 [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.UNUSEDPIN PULLNONE [current_design]
set_property CONFIG_MODE BPI16 [current_design]
set_property BITSTREAM.CONFIG.BPI_SYNC_MODE DISABLE [current_design]
