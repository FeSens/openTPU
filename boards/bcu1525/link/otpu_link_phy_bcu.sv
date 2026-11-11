// BCU1525 x2 physical layer: two Aurora 64B/66B x4 streaming cores at 25.78125 Gb/s, their
// external GT commons and user clocks, adapted from qsfp-link 9133ed3f's s6 v6 image. Port 0
// is QSFP0 / quad 231 (X1Y48-51); port 1 is QSFP1 / quad 230 (X1Y44-47). Both use U9's
// 161.1328125 MHz on quad 231 MGTREFCLK1: quad 230 takes that buffered clock through
// GTSOUTHREFCLK10, QPLL0REFCLKSEL 110. Quad 230's U12 reference is not used or buffered.
//
// Power-on release is counted from STARTUPE3 CFGMCLK (2^24 cycles, nominally 335 ms), then
// synchronized into the board's independent 100 MHz init_clk; rst reasserts the Aurora resets.
// No SI5335 reset or frequency-select port exists here. The flash and configuration pins stay
// three-stated. The board must own exactly one STARTUPE3; integrate this instance with any
// existing configuration/flash owner before building. The Aurora aur_q0 / aur_q1 IP must be
// generated with the s6 streaming / SupportLevel 0 settings and its placement / CDC constraints.
//
// Source/lint only: this wrapper has NEVER been through Vivado or tested on hardware. The
// proven measurement belongs to the original s6 image, not to this new integration.
`default_nettype none
/* verilator lint_off PROCASSINIT */
/* verilator lint_off PINCONNECTEMPTY */
module otpu_link_phy_bcu (
  input wire init_clk,
  input wire rst,
  input wire qsfp0_mgt_refclk_1_p,
  input wire qsfp0_mgt_refclk_1_n,
  input wire [1:0][3:0] rxp, rxn,
  output wire [1:0][3:0] txp, txn,
  output wire [1:0] qsfp_resetl, qsfp_lpmode, qsfp_modsell,
  output wire [1:0] user_clk,
  input wire [1:0][255:0] tx_tdata,
  input wire [1:0] tx_tvalid,
  output wire [1:0] tx_tready,
  output wire [1:0][255:0] rx_tdata,
  output wire [1:0] rx_tvalid,
  output wire [1:0] channel_up,
  output wire [1:0][3:0] lane_up,
  output wire [1:0] soft_err, hard_err
);
  wire cfgmclk, cfgclk;
  STARTUPE3 #(.PROG_USR("FALSE"), .SIM_CCLK_FREQ(0.0)) u_startup (
    .CFGCLK(), .CFGMCLK(cfgmclk), .DI(), .EOS(), .PREQ(), .DO(4'b0000), .DTS(4'b1111),
    .FCSBO(1'b1), .FCSBTS(1'b1), .GSR(1'b0), .GTS(1'b0), .KEYCLEARB(1'b1), .PACK(1'b0),
    .USRCCLKO(1'b0), .USRCCLKTS(1'b1), .USRDONEO(1'b1), .USRDONETS(1'b1));
  BUFG u_cfgclk (.I(cfgmclk), .O(cfgclk));
  logic [24:0] por = '0;
  always_ff @(posedge cfgclk) if (!por[24]) por <= por + 1'b1;
  wire aur_rst;
  otpu_link_sync_bit #(.N(3), .INIT(1'b1)) u_rst (.clk(init_clk), .d(rst || !por[24]), .q(aur_rst));
  assign qsfp_resetl = 2'b11;
  assign qsfp_lpmode = 2'b00;
  assign qsfp_modsell = 2'b00;

  wire [1:0] unused_refbuf;
  for (genvar p = 0; p < 2; p++) begin : g_q
    wire [17:0] unused_aur;
    wire unused_refmon;
    (* keep_hierarchy = "yes" *) otpu_link_aurora_bcu #(.Q(p), .NB(p)) u_aur (
      .refclk_p(p == 0 ? qsfp0_mgt_refclk_1_p : 1'b0),
      .refclk_n(p == 0 ? qsfp0_mgt_refclk_1_n : 1'b0),
      .refclk_in(unused_refbuf[0]), .refclk_out(unused_refbuf[p]),
      .rxp(rxp[p]), .rxn(rxn[p]), .txp(txp[p]), .txn(txn[p]), .init_clk,
      .reset_pb(aur_rst), .pma_init(aur_rst), .loopback(3'b000),
      .user_clk(user_clk[p]), .tx_tdata(tx_tdata[p]), .tx_tvalid(tx_tvalid[p]), .tx_tready(tx_tready[p]),
      .rx_tdata(rx_tdata[p]), .rx_tvalid(rx_tvalid[p]), .channel_up(channel_up[p]), .lane_up(lane_up[p]),
      .soft_err(soft_err[p]), .hard_err(hard_err[p]), .aur(unused_aur), .refclk_mon(unused_refmon));
  end
endmodule
/* verilator lint_on PINCONNECTEMPTY */
/* verilator lint_on PROCASSINIT */
`default_nettype wire
