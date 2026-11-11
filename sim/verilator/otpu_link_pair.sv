// Two cards' link ports joined by two cables, for simulation (docs/x2_link.md
// section 6): otpu_link as card a (CARD_ID 0) and card b (1), NCARDS 2, each link's Aurora core
// and cable an otpu_link_chan (cable p: a's port p to b's port p, as s6's cabling). One core clock
// and one user_clk for both cards. The hub-side ends per card and link ([p]) as otpu_link's; the
// cards' X2 windows on c_* ports of their own. The real-RTL counterpart of two otpu_link_sim
// instances (cables 0 and 1).
`default_nettype none

module otpu_link_cards import otpu_link_pkg::*; #(
  parameter int D   = 58,                  // user_clk cycles through Aurora and the cable
  parameter int CC  = 7,
  parameter int UPD = 64,
  parameter bit B_SWAP = 0,
  parameter int ID = 0
) (
  input  logic                  clk,
  input  logic                  uclk,
  input  logic                  rst,
  input  logic [7:0]            a_card,
  input  logic [7:0]            a_ncards,
  input  logic [7:0]            b_card,
  input  logic [7:0]            b_ncards,
  input  logic [1:0][WW-1:0]    a_s_tdata,
  input  logic [1:0]            a_s_tvalid,
  output logic [1:0]            a_s_tready,
  input  logic [1:0]            a_s_tlast,
  output logic [1:0][WW-1:0]    a_m_tdata,
  output logic [1:0]            a_m_tvalid,
  input  logic [1:0]            a_m_tready,
  output logic [1:0]            a_m_tlast,
  output logic [1:0]            a_m_thdr,
  output logic [1:0]            a_up,
  output logic [1:0]            a_rx_err,
  output logic [1:0]            a_abort_in,
  input  logic [1:0]            a_abort,
  input  logic                  a_c_we,
  input  logic [7:0]            a_c_addr,
  input  logic [31:0]           a_c_wd,
  output logic [31:0]           a_c_rd,
  input  logic [1:0][WW-1:0]    b_s_tdata,
  input  logic [1:0]            b_s_tvalid,
  output logic [1:0]            b_s_tready,
  input  logic [1:0]            b_s_tlast,
  output logic [1:0][WW-1:0]    b_m_tdata,
  output logic [1:0]            b_m_tvalid,
  input  logic [1:0]            b_m_tready,
  output logic [1:0]            b_m_tlast,
  output logic [1:0]            b_m_thdr,
  output logic [1:0]            b_up,
  output logic [1:0]            b_rx_err,
  output logic [1:0]            b_abort_in,
  input  logic [1:0]            b_abort,
  input  logic                  b_c_we,
  input  logic [7:0]            b_c_addr,
  input  logic [31:0]           b_c_wd,
  output logic [31:0]           b_c_rd
);
  logic [1:0][WW-1:0] a_txd, a_rxd, b_txd, b_rxd;
  logic [1:0]         a_txv, a_txr, a_rxv, b_txv, b_txr, b_rxv, chup, soft_error;
  wire [1:0] b_chup = B_SWAP ? {chup[0], chup[1]} : chup;
  wire [1:0] b_soft = B_SWAP ? {soft_error[0], soft_error[1]} : soft_error;
  logic [1:0][3:0]    lnup;
  wire [1:0][3:0] b_lnup = B_SWAP ? {lnup[0], lnup[1]} : lnup;
  logic urst;
  otpu_link_sync_bit #(.N(3), .INIT(1'b1)) u_urst (.clk(uclk), .d(rst), .q(urst));

  otpu_link #(.CFG(32'h5EED_0001)) u_a (
    .clk, .rst,
    .s_tdata(a_s_tdata), .s_tvalid(a_s_tvalid), .s_tready(a_s_tready), .s_tlast(a_s_tlast),
    .m_tdata(a_m_tdata), .m_tvalid(a_m_tvalid), .m_tready(a_m_tready), .m_tlast(a_m_tlast),
    .m_thdr(a_m_thdr), .up(a_up), .rx_err(a_rx_err), .abort_in(a_abort_in), .abort(a_abort),
    .card(a_card), .ncards(a_ncards), .c_we(a_c_we), .c_addr(a_c_addr), .c_wd(a_c_wd), .c_rd(a_c_rd),
    .user_clk({uclk, uclk}), .tx_tdata(a_txd), .tx_tvalid(a_txv), .tx_tready(a_txr),
    .rx_tdata(a_rxd), .rx_tvalid(a_rxv), .channel_up(chup), .lane_up(lnup), .soft_err(soft_error));
  otpu_link #(.CFG(32'h5EED_0001)) u_b (
    .clk, .rst,
    .s_tdata(b_s_tdata), .s_tvalid(b_s_tvalid), .s_tready(b_s_tready), .s_tlast(b_s_tlast),
    .m_tdata(b_m_tdata), .m_tvalid(b_m_tvalid), .m_tready(b_m_tready), .m_tlast(b_m_tlast),
    .m_thdr(b_m_thdr), .up(b_up), .rx_err(b_rx_err), .abort_in(b_abort_in), .abort(b_abort),
    .card(b_card), .ncards(b_ncards), .c_we(b_c_we), .c_addr(b_c_addr), .c_wd(b_c_wd), .c_rd(b_c_rd),
    .user_clk({uclk, uclk}), .tx_tdata(b_txd), .tx_tvalid(b_txv), .tx_tready(b_txr),
    .rx_tdata(b_rxd), .rx_tvalid(b_rxv), .channel_up(b_chup), .lane_up(b_lnup), .soft_err(b_soft));
  for (genvar p = 0; p < 2; p++) begin : g_c
    otpu_link_chan #(.W(WW), .D(D), .CC(CC), .UPD(UPD), .ID(ID + p)) u_ch (
      .clk(uclk), .rst(urst),
      .a_tx_tdata(a_txd[p]), .a_tx_tvalid(a_txv[p]), .a_tx_tready(a_txr[p]),
      .a_rx_tdata(a_rxd[p]), .a_rx_tvalid(a_rxv[p]),
      .b_tx_tdata(b_txd[p ^ int'(B_SWAP)]), .b_tx_tvalid(b_txv[p ^ int'(B_SWAP)]), .b_tx_tready(b_txr[p ^ int'(B_SWAP)]),
      .b_rx_tdata(b_rxd[p ^ int'(B_SWAP)]), .b_rx_tvalid(b_rxv[p ^ int'(B_SWAP)]),
      .channel_up(chup[p]), .lane_up(lnup[p]), .soft_err(soft_error[p]));
  end
endmodule

`default_nettype wire

// One cable, with exactly otpu_link_sim's ports: it can replace the model at the hub without
// changing the instance connections. Two real two-port endpoints are clocked at 150 MHz by the
// caller and 402.83 MHz by the simulation-only user clock here. The unused cable is idle. CA/CB,
// NC and PA/PB choose the endpoint configuration / cable (crossed cabling is also supported).
// D, CC and UPD model Aurora, as otpu_link_cards; latency is measured, not imposed as L. This
// adapter exposes the actual endpoint's status/hello snapshot; all CSR tests use otpu_link_cards.
`default_nettype none
module otpu_link_pair import otpu_link_pkg::*; #(
  parameter int ID = 0,
  parameter int CA = 0, PA = 0, CB = 1, PB = 0, NC = 2,
  parameter int D = 58, CC = 7, UPD = 64
) (
  input logic clk, rst,
  input logic [WW-1:0] a_s_tdata,
  input logic a_s_tvalid,
  output logic a_s_tready,
  input logic a_s_tlast,
  output logic [WW-1:0] a_m_tdata,
  output logic a_m_tvalid,
  input logic a_m_tready,
  output logic a_m_tlast, a_m_thdr, a_up, a_rx_err, a_abort_in,
  input logic a_abort,
  input logic [WW-1:0] b_s_tdata,
  input logic b_s_tvalid,
  output logic b_s_tready,
  input logic b_s_tlast,
  output logic [WW-1:0] b_m_tdata,
  output logic b_m_tvalid,
  input logic b_m_tready,
  output logic b_m_tlast, b_m_thdr, b_up, b_rx_err, b_abort_in,
  input logic b_abort,
  output logic [31:0] a_stat, a_hello, b_stat, b_hello
);
  timeunit 1ps; timeprecision 1ps;
  logic uclk = 0;
  always #1241 uclk = !uclk;
  logic [1:0][WW-1:0] asd, bsd, amd, bmd;
  logic [1:0] asv, bsv, asl, bsl, asr, bsr, amv, bmv, amr, bmr, aml, bml, amh, bmh;
  logic [1:0] au, bu, ae, be, aai, bai, aa, ba;
  logic [31:0] unused_ard, unused_brd;
  initial if (PA < 0 || PA > 1 || PB < 0 || PB > 1) $fatal(1, "pair: PA/PB must be 0 or 1");
  always_comb begin
    asd = '0; asv = '0; asl = '0; amr = '0; aa = '0;
    bsd = '0; bsv = '0; bsl = '0; bmr = '0; ba = '0;
    asd[PA] = a_s_tdata; asv[PA] = a_s_tvalid; asl[PA] = a_s_tlast;
    amr[PA] = a_m_tready; aa[PA] = a_abort;
    bsd[PB] = b_s_tdata; bsv[PB] = b_s_tvalid; bsl[PB] = b_s_tlast;
    bmr[PB] = b_m_tready; ba[PB] = b_abort;
    a_s_tready = asr[PA]; a_m_tdata = amd[PA]; a_m_tvalid = amv[PA];
    a_m_tlast = aml[PA]; a_m_thdr = amh[PA]; a_up = au[PA]; a_rx_err = ae[PA]; a_abort_in = aai[PA];
    b_s_tready = bsr[PB]; b_m_tdata = bmd[PB]; b_m_tvalid = bmv[PB];
    b_m_tlast = bml[PB]; b_m_thdr = bmh[PB]; b_up = bu[PB]; b_rx_err = be[PB]; b_abort_in = bai[PB];
  end
  otpu_link_cards #(.D(D), .CC(CC), .UPD(UPD), .B_SWAP(PA != PB), .ID(ID)) u_cards (
    .clk, .uclk, .rst, .a_card(8'(CA)), .a_ncards(8'(NC)), .b_card(8'(CB)), .b_ncards(8'(NC)),
    .a_s_tdata(asd), .a_s_tvalid(asv), .a_s_tready(asr), .a_s_tlast(asl),
    .a_m_tdata(amd), .a_m_tvalid(amv), .a_m_tready(amr), .a_m_tlast(aml), .a_m_thdr(amh),
    .a_up(au), .a_rx_err(ae), .a_abort_in(aai), .a_abort(aa),
    .a_c_we(1'b0), .a_c_addr(8'h0), .a_c_wd(32'd0), .a_c_rd(unused_ard),
    .b_s_tdata(bsd), .b_s_tvalid(bsv), .b_s_tready(bsr), .b_s_tlast(bsl),
    .b_m_tdata(bmd), .b_m_tvalid(bmv), .b_m_tready(bmr), .b_m_tlast(bml), .b_m_thdr(bmh),
    .b_up(bu), .b_rx_err(be), .b_abort_in(bai), .b_abort(ba),
    .b_c_we(1'b0), .b_c_addr(8'h0), .b_c_wd(32'd0), .b_c_rd(unused_brd));
  assign a_stat = u_cards.u_a.r_stat[PA];
  assign b_stat = u_cards.u_b.r_stat[PB];
  assign a_hello = u_cards.u_a.r_hello[PA];
  assign b_hello = u_cards.u_b.r_hello[PB];
endmodule
`default_nettype wire
