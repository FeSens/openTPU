// Lint-only declarations: primitive signatures from Yosys cells_xtra.v;
// Aurora 13.0 SupportLevel 0 interface from s6's lt_aurora. No primitive or IP behavior is
// modeled. These stubs are never synthesis sources or evidence of a working physical link.
/* verilator lint_off DECLFILENAME */
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNDRIVEN */
/* verilator lint_off ASCRANGE */
module BUFG(input wire I, output wire O);
  assign O = I;
endmodule

module BUFG_GT(O, CE, CEMASK, CLR, CLRMASK, DIV, I);
    parameter SIM_DEVICE = "ULTRASCALE";
    parameter STARTUP_SYNC = "FALSE";
    (* clkbuf_driver *)
    output O;
    input CE;
    input CEMASK;
    input CLR;
    input CLRMASK;
    input [2:0] DIV;
    input I;
endmodule

module STARTUPE3(CFGCLK, CFGMCLK, DI, EOS, PREQ, DO, DTS, FCSBO, FCSBTS, GSR, GTS, KEYCLEARB, PACK, USRCCLKO, USRCCLKTS, USRDONEO, USRDONETS);
    parameter PROG_USR = "FALSE";
    parameter real SIM_CCLK_FREQ = 0.0;
    output CFGCLK;
    output CFGMCLK;
    output [3:0] DI;
    output EOS;
    output PREQ;
    input [3:0] DO;
    input [3:0] DTS;
    input FCSBO;
    input FCSBTS;
    input GSR;
    input GTS;
    input KEYCLEARB;
    input PACK;
    input USRCCLKO;
    input USRCCLKTS;
    input USRDONEO;
    input USRDONETS;
endmodule

module GTYE4_COMMON(DRPDO, DRPRDY, PMARSVDOUT0, PMARSVDOUT1, QPLL0FBCLKLOST, QPLL0LOCK, QPLL0OUTCLK, QPLL0OUTREFCLK, QPLL0REFCLKLOST, QPLL1FBCLKLOST, QPLL1LOCK, QPLL1OUTCLK, QPLL1OUTREFCLK, QPLL1REFCLKLOST, QPLLDMONITOR0, QPLLDMONITOR1, REFCLKOUTMONITOR0, REFCLKOUTMONITOR1, RXRECCLK0SEL, RXRECCLK1SEL, SDM0FINALOUT
, SDM0TESTDATA, SDM1FINALOUT, SDM1TESTDATA, UBDADDR, UBDEN, UBDI, UBDWE, UBMDMTDO, UBRSVDOUT, UBTXUART, BGBYPASSB, BGMONITORENB, BGPDB, BGRCALOVRD, BGRCALOVRDENB, DRPADDR, DRPCLK, DRPDI, DRPEN, DRPWE, GTGREFCLK0
, GTGREFCLK1, GTNORTHREFCLK00, GTNORTHREFCLK01, GTNORTHREFCLK10, GTNORTHREFCLK11, GTREFCLK00, GTREFCLK01, GTREFCLK10, GTREFCLK11, GTSOUTHREFCLK00, GTSOUTHREFCLK01, GTSOUTHREFCLK10, GTSOUTHREFCLK11, PCIERATEQPLL0, PCIERATEQPLL1, PMARSVD0, PMARSVD1, QPLL0CLKRSVD0, QPLL0CLKRSVD1, QPLL0FBDIV, QPLL0LOCKDETCLK
, QPLL0LOCKEN, QPLL0PD, QPLL0REFCLKSEL, QPLL0RESET, QPLL1CLKRSVD0, QPLL1CLKRSVD1, QPLL1FBDIV, QPLL1LOCKDETCLK, QPLL1LOCKEN, QPLL1PD, QPLL1REFCLKSEL, QPLL1RESET, QPLLRSVD1, QPLLRSVD2, QPLLRSVD3, QPLLRSVD4, RCALENB, SDM0DATA, SDM0RESET, SDM0TOGGLE, SDM0WIDTH
, SDM1DATA, SDM1RESET, SDM1TOGGLE, SDM1WIDTH, UBCFGSTREAMEN, UBDO, UBDRDY, UBENABLE, UBGPI, UBINTR, UBIOLMBRST, UBMBRST, UBMDMCAPTURE, UBMDMDBGRST, UBMDMDBGUPDATE, UBMDMREGEN, UBMDMSHIFT, UBMDMSYSRST, UBMDMTCK, UBMDMTDI);
    parameter [0:0] AEN_QPLL0_FBDIV = 1'b1;
    parameter [0:0] AEN_QPLL1_FBDIV = 1'b1;
    parameter [0:0] AEN_SDM0TOGGLE = 1'b0;
    parameter [0:0] AEN_SDM1TOGGLE = 1'b0;
    parameter [0:0] A_SDM0TOGGLE = 1'b0;
    parameter [8:0] A_SDM1DATA_HIGH = 9'b000000000;
    parameter [15:0] A_SDM1DATA_LOW = 16'b0000000000000000;
    parameter [0:0] A_SDM1TOGGLE = 1'b0;
    parameter [15:0] BIAS_CFG0 = 16'h0000;
    parameter [15:0] BIAS_CFG1 = 16'h0000;
    parameter [15:0] BIAS_CFG2 = 16'h0000;
    parameter [15:0] BIAS_CFG3 = 16'h0000;
    parameter [15:0] BIAS_CFG4 = 16'h0000;
    parameter [15:0] BIAS_CFG_RSVD = 16'h0000;
    parameter [15:0] COMMON_CFG0 = 16'h0000;
    parameter [15:0] COMMON_CFG1 = 16'h0000;
    parameter [15:0] POR_CFG = 16'h0000;
    parameter [15:0] PPF0_CFG = 16'h0F00;
    parameter [15:0] PPF1_CFG = 16'h0F00;
    parameter QPLL0CLKOUT_RATE = "FULL";
    parameter [15:0] QPLL0_CFG0 = 16'h391C;
    parameter [15:0] QPLL0_CFG1 = 16'h0000;
    parameter [15:0] QPLL0_CFG1_G3 = 16'h0020;
    parameter [15:0] QPLL0_CFG2 = 16'h0F80;
    parameter [15:0] QPLL0_CFG2_G3 = 16'h0F80;
    parameter [15:0] QPLL0_CFG3 = 16'h0120;
    parameter [15:0] QPLL0_CFG4 = 16'h0002;
    parameter [9:0] QPLL0_CP = 10'b0000011111;
    parameter [9:0] QPLL0_CP_G3 = 10'b0000011111;
    parameter integer QPLL0_FBDIV = 66;
    parameter integer QPLL0_FBDIV_G3 = 80;
    parameter [15:0] QPLL0_INIT_CFG0 = 16'h0000;
    parameter [7:0] QPLL0_INIT_CFG1 = 8'h00;
    parameter [15:0] QPLL0_LOCK_CFG = 16'h01E8;
    parameter [15:0] QPLL0_LOCK_CFG_G3 = 16'h21E8;
    parameter [9:0] QPLL0_LPF = 10'b1011111111;
    parameter [9:0] QPLL0_LPF_G3 = 10'b1111111111;
    parameter [0:0] QPLL0_PCI_EN = 1'b0;
    parameter [0:0] QPLL0_RATE_SW_USE_DRP = 1'b0;
    parameter integer QPLL0_REFCLK_DIV = 1;
    parameter [15:0] QPLL0_SDM_CFG0 = 16'h0040;
    parameter [15:0] QPLL0_SDM_CFG1 = 16'h0000;
    parameter [15:0] QPLL0_SDM_CFG2 = 16'h0000;
    parameter QPLL1CLKOUT_RATE = "FULL";
    parameter [15:0] QPLL1_CFG0 = 16'h691C;
    parameter [15:0] QPLL1_CFG1 = 16'h0020;
    parameter [15:0] QPLL1_CFG1_G3 = 16'h0020;
    parameter [15:0] QPLL1_CFG2 = 16'h0F80;
    parameter [15:0] QPLL1_CFG2_G3 = 16'h0F80;
    parameter [15:0] QPLL1_CFG3 = 16'h0120;
    parameter [15:0] QPLL1_CFG4 = 16'h0002;
    parameter [9:0] QPLL1_CP = 10'b0000011111;
    parameter [9:0] QPLL1_CP_G3 = 10'b0000011111;
    parameter integer QPLL1_FBDIV = 66;
    parameter integer QPLL1_FBDIV_G3 = 80;
    parameter [15:0] QPLL1_INIT_CFG0 = 16'h0000;
    parameter [7:0] QPLL1_INIT_CFG1 = 8'h00;
    parameter [15:0] QPLL1_LOCK_CFG = 16'h01E8;
    parameter [15:0] QPLL1_LOCK_CFG_G3 = 16'h21E8;
    parameter [9:0] QPLL1_LPF = 10'b1011111111;
    parameter [9:0] QPLL1_LPF_G3 = 10'b1111111111;
    parameter [0:0] QPLL1_PCI_EN = 1'b0;
    parameter [0:0] QPLL1_RATE_SW_USE_DRP = 1'b0;
    parameter integer QPLL1_REFCLK_DIV = 1;
    parameter [15:0] QPLL1_SDM_CFG0 = 16'h0000;
    parameter [15:0] QPLL1_SDM_CFG1 = 16'h0000;
    parameter [15:0] QPLL1_SDM_CFG2 = 16'h0000;
    parameter [15:0] RSVD_ATTR0 = 16'h0000;
    parameter [15:0] RSVD_ATTR1 = 16'h0000;
    parameter [15:0] RSVD_ATTR2 = 16'h0000;
    parameter [15:0] RSVD_ATTR3 = 16'h0000;
    parameter [1:0] RXRECCLKOUT0_SEL = 2'b00;
    parameter [1:0] RXRECCLKOUT1_SEL = 2'b00;
    parameter [0:0] SARC_ENB = 1'b0;
    parameter [0:0] SARC_SEL = 1'b0;
    parameter [15:0] SDM0INITSEED0_0 = 16'b0000000000000000;
    parameter [8:0] SDM0INITSEED0_1 = 9'b000000000;
    parameter [15:0] SDM1INITSEED0_0 = 16'b0000000000000000;
    parameter [8:0] SDM1INITSEED0_1 = 9'b000000000;
    parameter SIM_DEVICE = "ULTRASCALE_PLUS";
    parameter SIM_MODE = "FAST";
    parameter SIM_RESET_SPEEDUP = "TRUE";
    parameter [15:0] UB_CFG0 = 16'h0000;
    parameter [15:0] UB_CFG1 = 16'h0000;
    parameter [15:0] UB_CFG2 = 16'h0000;
    parameter [15:0] UB_CFG3 = 16'h0000;
    parameter [15:0] UB_CFG4 = 16'h0000;
    parameter [15:0] UB_CFG5 = 16'h0400;
    parameter [15:0] UB_CFG6 = 16'h0000;
    output [15:0] DRPDO;
    output DRPRDY;
    output [7:0] PMARSVDOUT0;
    output [7:0] PMARSVDOUT1;
    output QPLL0FBCLKLOST;
    output QPLL0LOCK;
    output QPLL0OUTCLK;
    output QPLL0OUTREFCLK;
    output QPLL0REFCLKLOST;
    output QPLL1FBCLKLOST;
    output QPLL1LOCK;
    output QPLL1OUTCLK;
    output QPLL1OUTREFCLK;
    output QPLL1REFCLKLOST;
    output [7:0] QPLLDMONITOR0;
    output [7:0] QPLLDMONITOR1;
    output REFCLKOUTMONITOR0;
    output REFCLKOUTMONITOR1;
    output [1:0] RXRECCLK0SEL;
    output [1:0] RXRECCLK1SEL;
    output [3:0] SDM0FINALOUT;
    output [14:0] SDM0TESTDATA;
    output [3:0] SDM1FINALOUT;
    output [14:0] SDM1TESTDATA;
    output [15:0] UBDADDR;
    output UBDEN;
    output [15:0] UBDI;
    output UBDWE;
    output UBMDMTDO;
    output UBRSVDOUT;
    output UBTXUART;
    input BGBYPASSB;
    input BGMONITORENB;
    input BGPDB;
    input [4:0] BGRCALOVRD;
    input BGRCALOVRDENB;
    input [15:0] DRPADDR;
    input DRPCLK;
    input [15:0] DRPDI;
    input DRPEN;
    input DRPWE;
    input GTGREFCLK0;
    input GTGREFCLK1;
    input GTNORTHREFCLK00;
    input GTNORTHREFCLK01;
    input GTNORTHREFCLK10;
    input GTNORTHREFCLK11;
    input GTREFCLK00;
    input GTREFCLK01;
    input GTREFCLK10;
    input GTREFCLK11;
    input GTSOUTHREFCLK00;
    input GTSOUTHREFCLK01;
    input GTSOUTHREFCLK10;
    input GTSOUTHREFCLK11;
    input [2:0] PCIERATEQPLL0;
    input [2:0] PCIERATEQPLL1;
    input [7:0] PMARSVD0;
    input [7:0] PMARSVD1;
    input QPLL0CLKRSVD0;
    input QPLL0CLKRSVD1;
    input [7:0] QPLL0FBDIV;
    input QPLL0LOCKDETCLK;
    input QPLL0LOCKEN;
    input QPLL0PD;
    input [2:0] QPLL0REFCLKSEL;
    input QPLL0RESET;
    input QPLL1CLKRSVD0;
    input QPLL1CLKRSVD1;
    input [7:0] QPLL1FBDIV;
    input QPLL1LOCKDETCLK;
    input QPLL1LOCKEN;
    input QPLL1PD;
    input [2:0] QPLL1REFCLKSEL;
    input QPLL1RESET;
    input [7:0] QPLLRSVD1;
    input [4:0] QPLLRSVD2;
    input [4:0] QPLLRSVD3;
    input [7:0] QPLLRSVD4;
    input RCALENB;
    input [24:0] SDM0DATA;
    input SDM0RESET;
    input SDM0TOGGLE;
    input [1:0] SDM0WIDTH;
    input [24:0] SDM1DATA;
    input SDM1RESET;
    input SDM1TOGGLE;
    input [1:0] SDM1WIDTH;
    input UBCFGSTREAMEN;
    input [15:0] UBDO;
    input UBDRDY;
    input UBENABLE;
    input [1:0] UBGPI;
    input [1:0] UBINTR;
    input UBIOLMBRST;
    input UBMBRST;
    input UBMDMCAPTURE;
    input UBMDMDBGRST;
    input UBMDMDBGUPDATE;
    input [3:0] UBMDMREGEN;
    input UBMDMSHIFT;
    input UBMDMSYSRST;
    input UBMDMTCK;
    input UBMDMTDI;
endmodule

module IBUFDS_GTE4(O, ODIV2, CEB, I, IB);
    parameter [0:0] REFCLK_EN_TX_PATH = 1'b0;
    parameter [1:0] REFCLK_HROW_CK_SEL = 2'b00;
    parameter [1:0] REFCLK_ICNTL_RX = 2'b00;
    output O;
    output ODIV2;
    input CEB;
    (* iopad_external_pin *)
    input I;
    (* iopad_external_pin *)
    input IB;
endmodule

module aur_q0 (
  input wire [0:255] s_axi_tx_tdata,
  input wire s_axi_tx_tvalid,
  output wire s_axi_tx_tready,
  output wire [0:255] m_axi_rx_tdata,
  output wire m_axi_rx_tvalid,
  input wire [0:3] rxp, rxn,
  output wire [0:3] txp, txn,
  input wire refclk1_in,
  output wire hard_err, soft_err, channel_up,
  output wire [0:3] lane_up,
  input wire mmcm_not_locked, user_clk, sync_clk, reset_pb, gt_rxcdrovrden_in, power_down,
  input wire [2:0] loopback,
  input wire pma_init,
  output wire gt_pll_lock,
  input wire gt_qpllclk_quad1_in, gt_qpllrefclk_quad1_in, gt_qplllock_quad1_in, gt_qpllrefclklost_quad1,
  output wire gt_to_common_qpllreset_out,
  input wire [9:0] gt0_drpaddr,
  input wire [15:0] gt0_drpdi,
  input wire gt0_drpen, gt0_drpwe,
  output wire [15:0] gt0_drpdo,
  output wire gt0_drprdy,
  input wire [9:0] gt1_drpaddr,
  input wire [15:0] gt1_drpdi,
  input wire gt1_drpen, gt1_drpwe,
  output wire [15:0] gt1_drpdo,
  output wire gt1_drprdy,
  input wire [9:0] gt2_drpaddr,
  input wire [15:0] gt2_drpdi,
  input wire gt2_drpen, gt2_drpwe,
  output wire [15:0] gt2_drpdo,
  output wire gt2_drprdy,
  input wire [9:0] gt3_drpaddr,
  input wire [15:0] gt3_drpdi,
  input wire gt3_drpen, gt3_drpwe,
  output wire [15:0] gt3_drpdo,
  output wire gt3_drprdy,
  input wire init_clk,
  output wire link_reset_out, sys_reset_out,
  output wire [3:0] gt_powergood,
  output wire tx_out_clk, bufg_gt_clr_out
);
endmodule

module aur_q1 (
  input wire [0:255] s_axi_tx_tdata,
  input wire s_axi_tx_tvalid,
  output wire s_axi_tx_tready,
  output wire [0:255] m_axi_rx_tdata,
  output wire m_axi_rx_tvalid,
  input wire [0:3] rxp, rxn,
  output wire [0:3] txp, txn,
  input wire refclk1_in,
  output wire hard_err, soft_err, channel_up,
  output wire [0:3] lane_up,
  input wire mmcm_not_locked, user_clk, sync_clk, reset_pb, gt_rxcdrovrden_in, power_down,
  input wire [2:0] loopback,
  input wire pma_init,
  output wire gt_pll_lock,
  input wire gt_qpllclk_quad1_in, gt_qpllrefclk_quad1_in, gt_qplllock_quad1_in, gt_qpllrefclklost_quad1,
  output wire gt_to_common_qpllreset_out,
  input wire [9:0] gt0_drpaddr,
  input wire [15:0] gt0_drpdi,
  input wire gt0_drpen, gt0_drpwe,
  output wire [15:0] gt0_drpdo,
  output wire gt0_drprdy,
  input wire [9:0] gt1_drpaddr,
  input wire [15:0] gt1_drpdi,
  input wire gt1_drpen, gt1_drpwe,
  output wire [15:0] gt1_drpdo,
  output wire gt1_drprdy,
  input wire [9:0] gt2_drpaddr,
  input wire [15:0] gt2_drpdi,
  input wire gt2_drpen, gt2_drpwe,
  output wire [15:0] gt2_drpdo,
  output wire gt2_drprdy,
  input wire [9:0] gt3_drpaddr,
  input wire [15:0] gt3_drpdi,
  input wire gt3_drpen, gt3_drpwe,
  output wire [15:0] gt3_drpdo,
  output wire gt3_drprdy,
  input wire init_clk,
  output wire link_reset_out, sys_reset_out,
  output wire [3:0] gt_powergood,
  output wire tx_out_clk, bufg_gt_clr_out
);
endmodule
/* verilator lint_on ASCRANGE */
/* verilator lint_on UNDRIVEN */
/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on UNUSEDPARAM */
/* verilator lint_on DECLFILENAME */
