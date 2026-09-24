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
# Reference clock (100 MHz from the slot) on MGTREFCLK J8; the lanes (TX F2 H2 K2 M2 N4 P2 T2
# U4) are placed by the XDMA IP with its GT locations (select the PCIe block / quad in bd.tcl
# if the default does not match these pins, see docs/board.md).
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
