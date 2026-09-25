# openTPU on the Inspur YPCB-00338 (xc7k480t-ffg1156-2): board pins, clocks, configuration.
# Pins from the vendor/reverse-engineered board XDC (vendor_ypcb003381p1.xdc, TiferKing/
# ypcb_00338_1p1_hack). DDR3 locations: otpu_ddr3_pins.xdc (generated) plus the MIG IP.

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
# The accelerator synchronizes the calibration flags and the die-temperature code itself (2
# flip-flops per bit, ASYNC_REG; the temperature is taken only when two samples agree);
# everything else crosses in the SmartConnects, which carry their own constraints.
set_false_path -to [get_cells -hier -filter {NAME =~ *u_board/cal_s1_reg*}]
set_false_path -to [get_cells -hier -filter {NAME =~ *u_board/tmp_s1_reg*}]

# The MIGs' system clock comes from the MMCM on the 50 MHz pin (same I/O column as the DDR3
# banks) through the CMT backbone. If placement reports "sub-optimal placement for a clock-
# capable IO pin and MMCM/PLL pair", uncomment:
# set_property CLOCK_DEDICATED_ROUTE BACKBONE [get_nets -hier -filter {NAME =~ *clk_wiz_0*clk_200*}]

# ---------------------------------------------------------------- configuration
# Configuration banks at 1.8 V (the board's flash and IO are LVCMOS18); BPI x16 flash.
set_property CFGBVS GND [current_design]
set_property CONFIG_VOLTAGE 1.8 [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.UNUSEDPIN PULLNONE [current_design]
set_property CONFIG_MODE BPI16 [current_design]
set_property BITSTREAM.CONFIG.BPI_SYNC_MODE DISABLE [current_design]

# ---------------------------------------------------------------- DDR3 (MIG) placement
# ddr3_reset_n of channel 0 (AD31, IOB_X0Y138) sits in a spare pin of a data byte group
# (bank index 2, group 3: CH0 byte lane 6). MIG counts it in that lane's PHY bit mask and gives
# the lane an ISERDES for bit position 1, which belongs on AD31's ILOGIC site. Left free, the
# placer put it outside the lane's clock region and the lane's phaser clocks became unroutable.
set_property LOC ILOGIC_X0Y138 [get_cells -hier -filter {NAME =~ */mig_0/*ddr_phy_4lanes_0.u_ddr_phy_4lanes/ddr_byte_lane_D.ddr_byte_lane_D/ddr_byte_group_io/input_?1?.iserdes_dq_.iserdesdq}]
