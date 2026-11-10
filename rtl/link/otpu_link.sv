// The x2 link port (docs/x2_link.md): two links (port 0 = QSFP0 / quad 231,
// port 1 = QSFP1 / quad 230), each an Aurora 64B/66B x4 streaming core's user side. Per link:
// otpu_link_port on the core clock (the hub adapter's AXI-Stream ends, credits, CRCs), the TX FIFO
// (2^TAW words) and the RX FIFO (RXW words: the receive buffer), otpu_link_lane on the core's
// user_clk, and the crossings. Placed in SLR2 beside the quads; the hub's adapter sits next to it
// and owns the crossing to SLR1.
// Per link, to the hub: up (channel and lanes up, a hello since the last X2_CARD change / link
// reset that agrees: the peer's NCARDS = ours, its card != ours), rx_err (a pulse: a header's HCRC
// or CHK failed, a trailer's DCRC failed, the RX FIFO overflowed, or the channel dropped while
// up), abort_in (a pulse: the peer sent ABORT); from it, abort (a pulse: send ABORT).
// Registers: the X2 window (otpu_ctrl 0x300-0x3FF; offsets R_* in otpu_link_pkg): card and
// ncards come in from otpu_ctrl's X2_CARD; X2_CTRL (c_we at R_CTRL: clear stickies, hello,
// ping; bit 2 ignored) is written here; c_rd reads the window (combinational in c_addr: otpu_ctrl registers it).
// Resets: rst is board reset OR otpu_ctrl's >=16-core-cycle LRST pulse; the user_clk sides
// follow through synchronizers. This port does not decode or extend LRST. The run's core reset must not reach rst (a card that starts later finds the
// other card's first messages waiting).
`default_nettype none

module otpu_link import otpu_link_pkg::*; #(
  parameter int TAW = 9,                 // TX FIFO: 2^TAW words
  parameter logic [31:0] CFG = 32'h0     // the design hash sent in hello
) (
  input  wire                  clk,
  input  wire                  rst,
  // the hub adapter, per link
  input  wire  [1:0][WW-1:0]   s_tdata,
  input  wire  [1:0]           s_tvalid,
  output logic [1:0]           s_tready,
  input  wire  [1:0]           s_tlast,
  output logic [1:0][WW-1:0]   m_tdata,
  output logic [1:0]           m_tvalid,
  input  wire  [1:0]           m_tready,
  output logic [1:0]           m_tlast,
  output logic [1:0]           m_thdr,
  output logic [1:0]           up,
  output logic [1:0]           rx_err,
  output logic [1:0]           abort_in,
  /* verilator lint_off SYMRSVDWORD */
  input  wire  [1:0]           abort,
  /* verilator lint_on SYMRSVDWORD */
  // registers
  input  wire  [7:0]           card,
  input  wire  [7:0]           ncards,
  input  wire                  c_we,
  input  wire  [7:0]           c_addr,
  input  wire  [31:0]          c_wd,
  output logic [31:0]          c_rd,
  // the Aurora cores' user sides, per link (each on its user_clk)
  input  wire  [1:0]           user_clk,
  output logic [1:0][WW-1:0]   tx_tdata,
  output logic [1:0]           tx_tvalid,
  input  wire  [1:0]           tx_tready,
  input  wire  [1:0][WW-1:0]   rx_tdata,
  input  wire  [1:0]           rx_tvalid,
  input  wire  [1:0]           channel_up,
  input  wire  [1:0][3:0]      lane_up,
  input  wire  [1:0]           soft_err
);
  localparam int RAW = $clog2(RXW);
  localparam int CRED_INIT = RXW - 8;

  // X2_CTRL pulses
  logic x_clr, x_hello, x_ping;
  always_ff @(posedge clk) begin
    x_clr <= !rst && c_we && c_addr == R_CTRL && c_wd[X_CLR];
    x_hello <= !rst && c_we && c_addr == R_CTRL && c_wd[X_HELLO];
    x_ping <= !rst && c_we && c_addr == R_CTRL && c_wd[X_PING];
  end
  // a change of card / ncards: a new hello out, and the received ones count again from here
  logic [15:0] cn_d;
  logic        cn_chg;
  always_ff @(posedge clk) begin
    cn_d <= {ncards, card};
    cn_chg <= !rst && cn_d != {ncards, card};
  end
  wire rst_c = rst;

  logic [1:0][31:0] r_stat, r_hello, r_crc, r_frm, r_ping, r_txw, r_rxw, r_cw, r_soft;

  for (genvar p = 0; p < 2; p++) begin : g_l
    wire uclk = user_clk[p];
    logic rst_u;
    otpu_link_sync_bit #(.N(3), .INIT(1'b1)) u_rst_u (.clk(uclk), .d(rst_c), .q(rst_u));

    // FIFOs
    logic          tf_we, tf_rvalid, tf_re, rf_we, rf_full, rf_rvalid, rf_re;
    logic [WW+3:0] tf_wdata, tf_rdata, rf_wdata, rf_rdata;
    logic [TAW:0]  tf_wfree, unused_tf_count;
    logic [RAW:0]  unused_rf_free, unused_rf_count;
    logic          unused_tf_full;
    otpu_link_afifo #(.W(WW + 4), .AW(TAW)) u_txf (
      .wclk(clk), .wrst(rst_c), .we(tf_we), .wdata(tf_wdata), .wfull(unused_tf_full), .wfree(tf_wfree),
      .rclk(uclk), .rrst(rst_u), .re(tf_re), .rdata(tf_rdata), .rvalid(tf_rvalid), .rcount(unused_tf_count));
    otpu_link_afifo #(.W(WW + 4), .AW(RAW)) u_rxf (
      .wclk(uclk), .wrst(rst_u), .we(rf_we), .wdata(rf_wdata), .wfull(rf_full), .wfree(unused_rf_free),
      .rclk(clk), .rrst(rst_c), .re(rf_re), .rdata(rf_rdata), .rvalid(rf_rvalid), .rcount(unused_rf_count));

    // credits across
    logic [15:0] popped_c, popped_u, cr_rx_u, cr_rx_c, avail;
    otpu_link_sync_count #(.W(16)) u_popped (.sclk(clk), .srst(rst_c), .sbin(popped_c),
                                             .dclk(uclk), .dbin(popped_u));
    otpu_link_sync_word #(.W(16)) u_cr (.sclk(uclk), .srst(rst_u), .sd(cr_rx_u),
                                        .dclk(clk), .drst(rst_c), .dd(cr_rx_c));

    // pulses into user_clk
    logic hello_u, ping_u, abort_u;
    otpu_link_sync_pulse u_p_hel (.sclk(clk), .sp(x_hello || cn_chg), .dclk(uclk), .dp(hello_u));
    otpu_link_sync_pulse u_p_png (.sclk(clk), .sp(x_ping), .dclk(uclk), .dp(ping_u));
    otpu_link_sync_pulse u_p_abt (.sclk(clk), .sp(abort[p]), .dclk(uclk), .dp(abort_u));

    // X2_CARD is stable while running; transfer the two fields together, including changes
    // made while the links are idle. The lane notices the synchronized change and sends hello.
    logic [15:0] cn_u;
    otpu_link_sync_word #(.W(16)) u_w_cn (.sclk(clk), .srst(rst_c), .sd({ncards, card}),
                                          .dclk(uclk), .drst(rst_u), .dd(cn_u));

    // user_clk side
    logic        hdr_err_u, ovf_u, abort_rx_u, unused_plock_u, ping_done_u;
    logic [31:0] peer_u, unused_pcfg_u, unused_ping_uw, txw_u, rxw_u, soft_u, unused_perr_u;
    logic [7:0]  hcnt_u;
    otpu_link_lane u_lane (
      .clk(uclk), .rst(rst_u),
      .tx_tdata(tx_tdata[p]), .tx_tvalid(tx_tvalid[p]), .tx_tready(tx_tready[p]),
      .rx_tdata(rx_tdata[p]), .rx_tvalid(rx_tvalid[p]), .channel_up(channel_up[p]),
      .soft_err(soft_err[p]),
      .tf_data(tf_rdata), .tf_valid(tf_rvalid), .tf_re,
      .rf_we, .rf_data(rf_wdata), .rf_full,
      .popped(popped_u), .cr_rx(cr_rx_u),
      .prbs_en(1'b0), .prbs_len(16'd0), .cred_th(16'd64),
      .card(cn_u[7:0]), .ncards(cn_u[15:8]), .port(1'(p)), .cfg(CFG),
      .hello_req(hello_u), .ping_req(ping_u), .abort_req(abort_u), .inject(1'b0),
      .hdr_err(hdr_err_u), .ovf(ovf_u), .abort_rx(abort_rx_u), .peer(peer_u), .peer_cfg(unused_pcfg_u),
      .hello_cnt(hcnt_u), .ping(unused_ping_uw), .ping_done(ping_done_u), .prbs_locked(unused_plock_u),
      .n_txw(txw_u), .n_rxw(rxw_u), .n_soft(soft_u), .n_perr(unused_perr_u));

    // core side
    logic len_err, hcrc_err, dcrc_err, frame, cred_wait;
    otpu_link_port #(.CRED_INIT(CRED_INIT), .TAW(TAW)) u_port (
      .clk, .rst(rst_c),
      .s_tdata(s_tdata[p]), .s_tvalid(s_tvalid[p]), .s_tready(s_tready[p]), .s_tlast(s_tlast[p]),
      .m_tdata(m_tdata[p]), .m_tvalid(m_tvalid[p]), .m_tready(m_tready[p]), .m_tlast(m_tlast[p]),
      .m_thdr(m_thdr[p]),
      .tf_we, .tf_wdata, .tf_wfree, .rf_rdata, .rf_rvalid, .rf_re,
      .cr_rx(cr_rx_c), .popped(popped_c), .avail, .len_err, .hcrc_err, .dcrc_err, .frame,
      .cred_wait);

    // the user_clk side into the core clock
    logic       chup_c, lanes_c, hdr_err_c, ovf_c, abort_c;
    logic [7:0] hcnt_c;
    logic [31:0] unused_peer_c, txw_c, rxw_c, soft_c;
    logic [39:0] hello_c;
    logic ping_done_c;
    otpu_link_sync_bit u_y_up (.clk, .d(channel_up[p]), .q(chup_c));
    otpu_link_sync_bit u_y_ln (.clk, .d(&lane_up[p]), .q(lanes_c));
    otpu_link_sync_pulse u_p_her (.sclk(uclk), .sp(hdr_err_u), .dclk(clk), .dp(hdr_err_c));
    otpu_link_sync_pulse u_p_ovf (.sclk(uclk), .sp(ovf_u), .dclk(clk), .dp(ovf_c));
    otpu_link_sync_pulse u_p_abi (.sclk(uclk), .sp(abort_rx_u), .dclk(clk), .dp(abort_c));
    otpu_link_sync_word #(.W(40)) u_w_pr (.sclk(uclk), .srst(rst_u), .sd({hcnt_u, peer_u}),
                                          .dclk(clk), .drst(rst_c), .dd(hello_c));
    assign {hcnt_c, unused_peer_c} = hello_c;
    otpu_link_sync_pulse u_p_done (.sclk(uclk), .sp(ping_done_u), .dclk(clk), .dp(ping_done_c));
    otpu_link_sync_count #(.W(32)) u_c_tx (.sclk(uclk), .srst(rst_u), .sbin(txw_u),
                                           .dclk(clk), .dbin(txw_c));
    otpu_link_sync_count #(.W(32)) u_c_rx (.sclk(uclk), .srst(rst_u), .sbin(rxw_u),
                                           .dclk(clk), .dbin(rxw_c));
    otpu_link_sync_count #(.W(32)) u_c_sf (.sclk(uclk), .srst(rst_u), .sbin(soft_u),
                                           .dclk(clk), .dbin(soft_c));

    // up, rx_err, the stickies and counts
    logic [7:0]  hrec;                   // hello count at the last card change
    logic        hnew, agree, up_q, s_crc, s_wd, s_abt, s_drop;
    logic [30:0] ping_age;
    logic [31:0] ping_core;
    logic ping_busy;
    logic [24:0] wcyc;
    logic [31:0] n_crc, n_frm, n_cw;
    always_comb begin
      agree = unused_peer_c[31] && unused_peer_c[15:8] == ncards && unused_peer_c[7:0] != card;
    end
    always_ff @(posedge clk) begin
      up_q <= !rst_c && !cn_chg && chup_c && lanes_c && hnew && agree;
      rx_err[p] <= !rst_c && (len_err || hcrc_err || dcrc_err || hdr_err_c || ovf_c ||
                                (up_q && !(chup_c && lanes_c)));
      abort_in[p] <= !rst_c && abort_c;
      if (rst_c) begin
        hrec <= '0; hnew <= 1'b0;
        ping_core <= '0; ping_age <= '0; ping_busy <= 1'b0;
        {s_crc, s_wd, s_abt, s_drop} <= '0;
        wcyc <= '0;
        n_crc <= '0; n_frm <= '0; n_cw <= '0;
      end else begin
        hrec <= hcnt_c;
        if (cn_chg) hnew <= 1'b0;
        else if (hcnt_c != hrec) hnew <= 1'b1;
        if (x_ping) begin ping_age <= '0; ping_busy <= 1'b1; end
        else if (ping_busy) ping_age <= ping_age + 1'b1;
        if (ping_done_c && ping_busy) begin
          ping_core <= {1'b1, ping_age};
          ping_busy <= 1'b0;
        end
        if (x_clr) {s_crc, s_wd, s_abt, s_drop} <= '0;
        if (len_err || hcrc_err || dcrc_err || hdr_err_c || ovf_c ||
            (up_q && !(chup_c && lanes_c))) s_crc <= 1'b1;
        if (abort_c) s_abt <= 1'b1;
        if (up_q && !(chup_c && lanes_c)) s_drop <= 1'b1;
        wcyc <= cred_wait ? wcyc + {24'd0, !wcyc[24]} : '0;
        if (wcyc[24]) s_wd <= 1'b1;
        n_crc <= n_crc + 32'(hcrc_err) + 32'(dcrc_err) + 32'(hdr_err_c) + 32'(ovf_c) + 32'(len_err) +
                 32'(up_q && !(chup_c && lanes_c));
        n_frm <= n_frm + 32'(frame);
        n_cw <= n_cw + 32'(cred_wait);
      end
    end
    assign up[p] = up_q;
    always_comb begin
      r_stat[p] = '0;
      r_stat[p][P_CHUP] = chup_c;
      r_stat[p][P_HELLO] = hnew;
      r_stat[p][P_CRC] = s_crc;
      r_stat[p][P_WDOG] = s_wd;
      r_stat[p][P_ABORT] = s_abt;
      r_stat[p][P_LANES] = lanes_c;
      r_stat[p][P_DROP] = s_drop;
      r_stat[p][P_UP] = up_q;
      r_stat[p][31:16] = avail;
      r_hello[p] = unused_peer_c[31] ? {HELLO_MAGIC, 4'd0, unused_peer_c[15:8], 3'd0, unused_peer_c[16], unused_peer_c[7:0]} : '0;
      r_crc[p] = n_crc;
      r_frm[p] = n_frm;
      r_ping[p] = ping_core;
      r_txw[p] = txw_c;
      r_rxw[p] = rxw_c;
      r_cw[p] = n_cw;
      r_soft[p] = soft_c;
    end
  end

  // the window
  always_comb begin
    c_rd = '0;
    case (c_addr)
      R_INFO:          c_rd = {16'(RXW), 8'(LMAX), PROTO};
      R_PORT + 8'h00:  c_rd = r_stat[0];
      R_PORT + 8'h04:  c_rd = r_hello[0];
      R_PORT + 8'h08:  c_rd = r_crc[0];
      R_PORT + 8'h0C:  c_rd = r_frm[0];
      R_PORT + 8'h10:  c_rd = r_stat[1];
      R_PORT + 8'h14:  c_rd = r_hello[1];
      R_PORT + 8'h18:  c_rd = r_crc[1];
      R_PORT + 8'h1C:  c_rd = r_frm[1];
      R_PING:          c_rd = r_ping[0];
      R_PING + 8'h04:  c_rd = r_ping[1];
      R_CNT + 8'h00:   c_rd = r_txw[0];
      R_CNT + 8'h04:   c_rd = r_rxw[0];
      R_CNT + 8'h08:   c_rd = r_cw[0];
      R_CNT + 8'h0C:   c_rd = r_soft[0];
      R_CNT + 8'h10:   c_rd = r_txw[1];
      R_CNT + 8'h14:   c_rd = r_rxw[1];
      R_CNT + 8'h18:   c_rd = r_cw[1];
      R_CNT + 8'h1C:   c_rd = r_soft[1];
      default:         c_rd = '0;
    endcase
  end
endmodule

`default_nettype wire
