#!/usr/bin/env python3

# litex-boards' platforms/ypcb_00338_1p1.py (2025), for tools/litedram/ld_test.py: the DDR3 pins
# match boards/ypcb-00338/constraints/ddr3_ch{0,1}.pins.xdc (checked pin by pin); the default
# clock is the 50 MHz oscillator the production design uses.
#
# This file is part of LiteX-Boards.
#
# Copyright (c) 2025 Florent Kermarrec <florent@enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# The YPCB-00338-1P1 boards has been documented by @TiferKing:
# - https://github.com/TiferKing/ypcb_00338_1p1_hack
# - https://www.tiferking.cn/index.php/2024/12/19/650/

# Features:
# - XC7K480T-FFG1156-2.
# - PCIe Gen2/3 x8.
# - Dual 72-bit DDR3 (64 + 8 ECC).
# - Linear BPI Flash (x16).
# - LM73 temperature sensor + SMBus on edge connector.
# - Three user LEDs.

from litex.build.generic_platform import *
from litex.build.xilinx import Xilinx7SeriesPlatform
from litex.build.openfpgaloader import OpenFPGALoader

# IOs ----------------------------------------------------------------------------------------------

_io = [
    # Clk / Rst.
    ("clk50", 0, Pins("AA28"), IOStandard("LVCMOS18")),
    ("rst_n", 0, Pins("R28"),  IOStandard("LVCMOS18")),
    ("clk200", 0,
        Subsignal("p", Pins("AH27"), IOStandard("LVDS_25")),
        Subsignal("n", Pins("AH28"), IOStandard("LVDS_25")),
    ),
    ("clk200_1", 0,
        Subsignal("p", Pins("G25"), IOStandard("LVDS_25")),
        Subsignal("n", Pins("G26"), IOStandard("LVDS_25")),
    ),

    # Leds.
    ("user_led", 0, Pins("P30"), IOStandard("LVCMOS18")), # Red.
    ("user_led", 1, Pins("M30"), IOStandard("LVCMOS18")), # Green.
    ("user_led", 2, Pins("N30"), IOStandard("LVCMOS18")), # Yellow.

    # LM73 temperature sensor (I²C + ALERT).
    ("lm73", 0,
        Subsignal("alert", Pins("P25")),
        Subsignal("scl",   Pins("N24")),
        Subsignal("sda",   Pins("N25")),
        IOStandard("LVCMOS18")
    ),

    # PCIe (Gen2 X8).
     ("pcie_i2c", 0,
        Subsignal("scl", Pins("R26")),
        Subsignal("sda", Pins("R27")),
        IOStandard("LVCMOS18")
    ),
    ("pcie_x1", 0,
        Subsignal("clk_p", Pins("J8")),
        Subsignal("clk_n", Pins("J7")),
        Subsignal("rst_n", Pins("Y26"), IOStandard("LVCMOS18")),
        Subsignal("tx_p",  Pins("F2")),
        Subsignal("tx_n",  Pins("F1")),
        Subsignal("rx_p",  Pins("H6")),
        Subsignal("rx_n",  Pins("H5")),
    ),
    ("pcie_x4", 0,
        Subsignal("clk_p", Pins("J8")),
        Subsignal("clk_n", Pins("J7")),
        Subsignal("rst_n", Pins("Y26"), IOStandard("LVCMOS18")),
        Subsignal("tx_p",  Pins("F2 H2 K2 M2")),
        Subsignal("tx_n",  Pins("F1 H1 K1 M1")),
        Subsignal("rx_p",  Pins("H6 J4 K6 L4")),
        Subsignal("rx_n",  Pins("H5 J3 K5 L3")),
    ),
    ("pcie_x8", 0,
        Subsignal("clk_p", Pins("J8")),
        Subsignal("clk_n", Pins("J7")),
        Subsignal("rst_n", Pins("Y26"), IOStandard("LVCMOS18")),
        Subsignal("tx_p",  Pins("F2 H2 K2 M2 N4 P2 T2 U4")),
        Subsignal("tx_n",  Pins("F1 H1 K1 M1 N3 P1 T1 U3")),
        Subsignal("rx_p",  Pins("H6 J4 K6 L4 M6 P6 R4 T6")),
        Subsignal("rx_n",  Pins("H5 J3 K5 L3 M5 P5 E3 T5")),
    ),

    # Linear Flash (BPI x16).
    ("linear_flash", 0,
        Subsignal("adr", Pins(
            # A[1] … A[25] (little-endian order).
            "AD26 AC25 AC29 AC28 AD27 AC27 AB25 AB28 AB27 AB26",
            "AA26 AA31 AA30 AB33 AB32  Y32  P32  R32  U33  T31",
            "T30   U31  U30  N34  P34"
        )),
        Subsignal("dq", Pins(
            # DQ[0] … DQ[15].
            "AA33 AA34 Y33 Y34 V32 V33 W31 W32 W30 V25",
            "W25 V29 W29 V28 W24 Y24"
        )),
        Subsignal("wen",  Pins("T34")),
        Subsignal("oen",  Pins("T33")),
        Subsignal("ce_n", Pins("V30")),
        Subsignal("adv",  Pins("M31")),
        IOStandard("LVCMOS18"),
        Misc("SLEW=FAST"),
    ),

    # DDR3 SDRAM – Channel 0.
    ("ddram", 0,
        # Address / bank / control.
        Subsignal("a", Pins(
            "AK27 AN23 AL24 AK26 AH24 AH25 AL26 AJ24",
            "AJ25 AM23 AL28 AL25 AM25 AK24 AM27"
        ), IOStandard("SSTL15")),
        Subsignal("ba",      Pins("AM26 AP24 AN28"), IOStandard("SSTL15")),
        Subsignal("ras_n",   Pins("AJ29"), IOStandard("SSTL15")),
        Subsignal("cas_n",   Pins("AP26"), IOStandard("SSTL15")),
        Subsignal("we_n",    Pins("AN27"), IOStandard("SSTL15")),
        Subsignal("cs_n",    Pins("AK28"), IOStandard("SSTL15")),
        Subsignal("cke",     Pins("AP27"), IOStandard("SSTL15")),
        Subsignal("odt",     Pins("AK29"), IOStandard("SSTL15")),
        Subsignal("reset_n", Pins("AD31"), IOStandard("SSTL15")),

        # Data / strobe / clock.
        Subsignal("dq", Pins(
            "AG17 AG16 AH17 AJ19 AH18 AH19 AJ16 AJ17",
            "AL20 AN17 AL19 AM16 AL18 AL16 AM20 AN18",
            "AL23 AN20 AK23 AP19 AN22 AN19 AM22 AP20",
            "AJ21 AH22 AK21 AG21 AG22 AG20 AH23 AG23",
            "AJ32 AK32 AK31 AL30 AL34 AL31 AK34 AL29",
            "AJ34 AH32 AJ30 AH34 AF31 AG30 AG31 AF30",
            "AE32 AC33 AF33 AC32 AD34 AC34 AE33 AE31",
            "AE26 AF29 AE24 AF28 AF24 AG25 AF26 AF25",
            "AN34 AP30 AM33 AN29 AP32 AP29 AM31 AP31" # ECC
        ), IOStandard("SSTL15"), Misc("IN_TERM=UNTUNED_SPLIT_40")),
        Subsignal("dqs_p", Pins(
            "AK16 AM17 AP21 AH20 AK33 AG33 AE34 AE27 AN32"
        ), IOStandard("DIFF_SSTL15"), Misc("IN_TERM=UNTUNED_SPLIT_40")),
        Subsignal("dqs_n", Pins(
            "AK17 AM18 AP22 AJ20 AL33 AH33 AF34 AE28 AP33"
        ), IOStandard("DIFF_SSTL15"), Misc("IN_TERM=UNTUNED_SPLIT_40")),
        Subsignal("clk_p", Pins("AN25"), IOStandard("DIFF_SSTL15")),
        Subsignal("clk_n", Pins("AP25"), IOStandard("DIFF_SSTL15")),
        Misc("SLEW=FAST"),
    ),

    # DDR3 SDRAM – Channel 1.
    ("ddram", 1,
        # Address / bank / control.
        Subsignal("a", Pins(
            "E27 C27 B28 D27 C24 D24 C25 A24",
            "A25 J24 F26 D26 H25 D25 B26"
        ), IOStandard("SSTL15")),
        Subsignal("ba",      Pins("F24 J25 E24"), IOStandard("SSTL15")),
        Subsignal("ras_n",   Pins("E28"), IOStandard("SSTL15")),
        Subsignal("cas_n",   Pins("E26"), IOStandard("SSTL15")),
        Subsignal("we_n",    Pins("F25"), IOStandard("SSTL15")),
        Subsignal("cs_n",    Pins("F28"), IOStandard("SSTL15")),
        Subsignal("cke",     Pins("A28"), IOStandard("SSTL15")),
        Subsignal("odt",     Pins("B27"), IOStandard("SSTL15")),
        Subsignal("reset_n", Pins("F18"), IOStandard("SSTL15")),

        # Data / strobe / clock.
        Subsignal("dq", Pins(
            "A29 B33 A31 C33 C32 A30 B30 A33",
            "D31 F33 D30 D29 E33 E34 E31 F34",
            "B23 A21 C23 B20 B22 A23 C20 B21",
            "G31 G32 F29 F31 E29 G33 H33 H32",
            "B18 C17 C19 B16 A18 A16 C18 B17",
            "K27 L24 K24 L28 K26 M27 L25 M26",
            "F16 E18 E16 H19 H17 H20 E17 H18",
            "D20 F21 E23 G21 G20 D21 F20 F23",
            "L34 K34 K31 K33 L31 J30 L33 J34"  # ECC
        ), IOStandard("SSTL15"), Misc("IN_TERM=UNTUNED_SPLIT_40")),
        Subsignal("dqs_p", Pins(
            "B31 D34 A19 H29 D16 K28 G17 G22 K32"
        ), IOStandard("DIFF_SSTL15"), Misc("IN_TERM=UNTUNED_SPLIT_40")),
        Subsignal("dqs_n", Pins(
            "B32 C34 A20 H30 D17 K29 G18 G23 J32"
        ), IOStandard("DIFF_SSTL15"), Misc("IN_TERM=UNTUNED_SPLIT_40")),
        Subsignal("clk_p", Pins("B25"), IOStandard("DIFF_SSTL15")),
        Subsignal("clk_n", Pins("A26"), IOStandard("DIFF_SSTL15")),
        Misc("SLEW=FAST"),
    ),
]

# Connectors ---------------------------------------------------------------------------------------

_connectors = []

# Platform -----------------------------------------------------------------------------------------

class Platform(Xilinx7SeriesPlatform):
    default_clk_name   = "clk50"
    default_clk_period = 1e9/50e6

    def __init__(self, toolchain="vivado"):
        device = "xc7k480t-ffg1156-2"
        Xilinx7SeriesPlatform.__init__(self, device, _io, _connectors, toolchain=toolchain)
        self.add_platform_command("set_property INTERNAL_VREF 0.750 [get_iobanks 11]")
        self.add_platform_command("set_property INTERNAL_VREF 0.750 [get_iobanks 12]")
        self.add_platform_command("set_property INTERNAL_VREF 0.750 [get_iobanks 13]")
        self.add_platform_command("set_property INTERNAL_VREF 0.750 [get_iobanks 14]")
        self.add_platform_command("set_property INTERNAL_VREF 0.750 [get_iobanks 15]")
        self.add_platform_command("set_property INTERNAL_VREF 0.750 [get_iobanks 16]")
        self.add_platform_command("set_property INTERNAL_VREF 0.750 [get_iobanks 17]")
        self.add_platform_command("set_property INTERNAL_VREF 0.750 [get_iobanks 18]")

    def create_programmer(self):
        return OpenFPGALoader(fpga_part="xc7k480t", cable="digilent_hs2", freq=20e6)

    def do_finalize(self, fragment):
        Xilinx7SeriesPlatform.do_finalize(self, fragment)
        self.add_period_constraint(self.lookup_request(self.default_clk_name, loose=True), self.default_clk_period)
