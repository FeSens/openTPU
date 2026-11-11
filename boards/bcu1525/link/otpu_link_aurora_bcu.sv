// One QSFP28 cage's Aurora 64B/66B link (docs/x2_link.md): the core as the
// build generated it for quad Q (aur_q0: quad 231, X1Y48-51, QSFP0; aur_q1: quad 230, X1Y44-47,
// QSFP1; 4 lanes, streaming, no flow control, the shared logic outside: SupportLevel 0) at the
// 25.78125 Gb/s line rate, and that shared logic as the IP's example design has it (<ip>_support.v):
// the reference clock's buffer (the cage's MGTREFCLK1, the SI5335 at 161.1328125 MHz), the quad's
// QPLL0 from it (otpu_link_gtcommon_bcu; NB 1 / 2: from refclk_in, the
// buffered MGTREFCLK1 of the quad above / below, over the adjacent-quad path, this quad's own then
// only to refclk_mon), user_clk on a BUFG_GT from lane
// 0's TXOUTCLK (held cleared 2^16 init_clk cycles after the core's bufg_gt_clr_out), the core's
// reset logic (pma_init debounced in init_clk and delayed, reset_pb debounced in user_clk and
// held while pma_init is; reset_pb comes in through a synchronizer here).
// The core's data ports are [0:255]: this side's bit 255 is the core's bit 0 (lane 0's first).
// aur: {gt_powergood[3:0], 0, refclklost, qpll lock, gt_pll_lock, user_clk active, sys_reset_out,
// link_reset_out, hard_err, soft_err, lane_up[3:0], channel_up} (user_clk / init_clk levels: read
// through synchronizers).
// Adapted from qsfp-link 9133ed3f's lt_aurora (s6 v6): fixed to 25G, the quad-230 local
// reference buffer omitted because both quads use U9. The shared reset and user-clock sequence
// is otherwise unchanged. This integration has been linted only, never built or run on a card.
// Initial state is FPGA power-up state; unused DRP outputs and ascending IP buses are deliberate.
/* verilator lint_off PROCASSINIT */
/* verilator lint_off PINCONNECTEMPTY */
/* verilator lint_off ASCRANGE */
/* verilator lint_off UNUSEDSIGNAL */
`default_nettype none

module otpu_link_aurora_bcu #(parameter int Q = 0, parameter int NB = 0) (
  input  wire          refclk_p,
  input  wire          refclk_n,
  input  wire          refclk_in,     // NB 1 / 2: the neighbour quad's buffered MGTREFCLK1
  output wire          refclk_out,    // this quad's buffered MGTREFCLK1 (a neighbour's refclk_in)
  input  wire  [3:0]   rxp,
  input  wire  [3:0]   rxn,
  output wire  [3:0]   txp,
  output wire  [3:0]   txn,
  input  wire          init_clk,
  input  wire          reset_pb,
  input  wire          pma_init,
  input  wire  [2:0]   loopback,
  output wire          user_clk,
  input  wire  [255:0] tx_tdata,
  input  wire          tx_tvalid,
  output wire          tx_tready,
  output wire  [255:0] rx_tdata,
  output wire          rx_tvalid,
  output wire          channel_up,
  output wire  [3:0]   lane_up,
  output wire          soft_err,
  output wire          hard_err,
  output wire  [17:0]  aur,
  output wire          refclk_mon     // the reference clock / 2 on a BUFG_GT (lt_freq)
);
  // the reference clock
  wire refclk, refclk_odiv2;
  if (NB == 0) begin : g_ref
    IBUFDS_GTE4 #(.REFCLK_HROW_CK_SEL(2'b00)) u_refbuf (
      .I(refclk_p), .IB(refclk_n), .CEB(1'b0), .O(refclk), .ODIV2(refclk_odiv2));
    BUFG_GT u_refmon (.I(refclk_odiv2), .CE(1'b1), .CEMASK(1'b0), .CLR(1'b0), .CLRMASK(1'b0),
                      .DIV(3'd1), .O(refclk_mon));
  end else begin : g_neighbour
    assign refclk = 1'b0;
    assign refclk_odiv2 = 1'b0;
    assign refclk_mon = 1'b0;
  end
  assign refclk_out = refclk;
  wire pll_ref = NB != 0 ? refclk_in : refclk;

  // the quad's PLL
  wire qpll_reset, qpll_lock, qpll_clk, qpll_refclk, qpll_lost;
  (* keep_hierarchy = "yes" *) otpu_link_gtcommon_bcu #(.NB(NB)) u_common (
    .refclk(pll_ref), .lockdetclk(init_clk), .reset(qpll_reset), .lock(qpll_lock),
    .outclk(qpll_clk), .outrefclk(qpll_refclk), .refclklost(qpll_lost));

  // user_clk
  wire tx_out_clk, bufg_gt_clr;
  logic        clr_dly = 1'b1;
  logic [15:0] clr_cnt = '0;
  always_ff @(posedge init_clk or posedge bufg_gt_clr)
    if (bufg_gt_clr) begin
      clr_dly <= 1'b1;
      clr_cnt <= '0;
    end else begin
      clr_cnt <= clr_cnt + 1'b1;
      if (&clr_cnt) clr_dly <= 1'b0;
    end
  BUFG_GT u_ubuf (.I(tx_out_clk), .CE(1'b1), .CEMASK(1'b0), .CLR(clr_dly), .CLRMASK(1'b0),
                  .DIV(3'd0), .O(user_clk));
  (* ASYNC_REG = "TRUE" *) logic [1:0] act = '0;
  always_ff @(posedge user_clk or posedge clr_dly)
    if (clr_dly) act <= '0;
    else act <= {act[0], 1'b1};
  wire uactive = act[1];

  // resets (<ip>_SUPPORT_RESET_LOGIC)
  (* ASYNC_REG = "TRUE" *) logic [3:0] gt_db = '1;
  logic        gt_rst = 1'b1;
  logic [19:0] gt_dly = '1;
  always_ff @(posedge init_clk) begin
    gt_db <= {gt_db[2:0], pma_init};
    gt_rst <= &gt_db;
    gt_dly <= {gt_dly[18:0], gt_rst};
  end
  wire pma_init_core = gt_dly[18];
  wire gt_rst_u;
  otpu_link_sync_bit #(.N(5), .INIT(1'b1)) u_gtrst (.clk(user_clk), .d(gt_rst), .q(gt_rst_u));
  wire        rpb_u;
  otpu_link_sync_bit #(.N(2), .INIT(1'b1)) u_rpb (.clk(user_clk), .d(reset_pb), .q(rpb_u));
  logic [3:0] rs_db = '1;
  logic       sys_rst = 1'b1;
  always_ff @(posedge user_clk) begin
    rs_db <= gt_rst_u ? 4'hF : {rs_db[2:0], rpb_u};
    sys_rst <= &rs_db;
  end

  // the core: bit 255 here is its bit 0
  wire [0:255] txd = tx_tdata;
  wire [0:255] rxd;
  assign rx_tdata = rxd;
  wire [0:3] lup, txp_c, txn_c;
  assign lane_up = lup;
  assign txp = txp_c;
  assign txn = txn_c;
  wire [0:3] rxp_c = rxp, rxn_c = rxn;
  wire       gt_pll_lock, link_reset_out, sys_reset_out;
  wire [3:0] gt_powergood;
  `define LT_AUR_PORTS \
    .s_axi_tx_tdata(txd), .s_axi_tx_tvalid(tx_tvalid), .s_axi_tx_tready(tx_tready), \
    .m_axi_rx_tdata(rxd), .m_axi_rx_tvalid(rx_tvalid), \
    .rxp(rxp_c), .rxn(rxn_c), .txp(txp_c), .txn(txn_c), .refclk1_in(1'b0), \
    .hard_err(hard_err), .soft_err(soft_err), .channel_up(channel_up), .lane_up(lup), \
    .mmcm_not_locked(!uactive), .user_clk(user_clk), .sync_clk(user_clk), .reset_pb(sys_rst), \
    .gt_rxcdrovrden_in(1'b0), .power_down(1'b0), .loopback(loopback), .pma_init(pma_init_core), \
    .gt_pll_lock(gt_pll_lock), \
    .gt_qpllclk_quad1_in(qpll_clk), .gt_qpllrefclk_quad1_in(qpll_refclk), \
    .gt_qplllock_quad1_in(qpll_lock), .gt_qpllrefclklost_quad1(qpll_lost), \
    .gt_to_common_qpllreset_out(qpll_reset), \
    .gt0_drpaddr(10'd0), .gt0_drpdi(16'd0), .gt0_drpen(1'b0), .gt0_drpwe(1'b0), .gt0_drpdo(), .gt0_drprdy(), \
    .gt1_drpaddr(10'd0), .gt1_drpdi(16'd0), .gt1_drpen(1'b0), .gt1_drpwe(1'b0), .gt1_drpdo(), .gt1_drprdy(), \
    .gt2_drpaddr(10'd0), .gt2_drpdi(16'd0), .gt2_drpen(1'b0), .gt2_drpwe(1'b0), .gt2_drpdo(), .gt2_drprdy(), \
    .gt3_drpaddr(10'd0), .gt3_drpdi(16'd0), .gt3_drpen(1'b0), .gt3_drpwe(1'b0), .gt3_drpdo(), .gt3_drprdy(), \
    .init_clk(init_clk), .link_reset_out(link_reset_out), .sys_reset_out(sys_reset_out), \
    .gt_powergood(gt_powergood), .tx_out_clk(tx_out_clk), .bufg_gt_clr_out(bufg_gt_clr)
  if (Q == 0) begin : g_q0
    aur_q0 u_core (`LT_AUR_PORTS);
  end else begin : g_q1
    aur_q1 u_core (`LT_AUR_PORTS);
  end
  `undef LT_AUR_PORTS

  assign aur = {gt_powergood, 1'b0, qpll_lost, qpll_lock, gt_pll_lock, uactive, sys_reset_out,
                link_reset_out, hard_err, soft_err, lup, channel_up};
endmodule

`default_nettype wire

/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on ASCRANGE */
/* verilator lint_on PINCONNECTEMPTY */
/* verilator lint_on PROCASSINIT */
