// otpu_native_dram2 behind a registered crossing of the slice's ports (otpu_board MEMX): the adapter
// in the memory's SLR, the slice in another (the FK35: the slices filled SLR1's block RAMs with
// their adapters beside them, docs/sqrl-fk35.md). The slice side (u_s) and the memory side (u_m)
// meet flip-flop to flip-flop (their sll_* registers, the Laguna sites: fk35_floorplan.tcl); the
// native ports go from the adapter (u_dram) to the channels as they are.
//
// Ports A, SW and B each cross into a queue of XN entries on the memory side (otpu_xq: its head in
// a register, as the slice's port registers present a request); the slice side gives the slice a
// port's rdy while it holds a credit, one per queue entry, a credit coming back when the adapter
// takes the entry. The queues may reorder against each other, as the adapter's own A, SW and B
// queues do (otpu_native_dram "Ordering": the slice orders what depends across ports by wr_idle).
// The read data come back through three registers (one on the memory side, then the crossing's
// two), without backpressure, as from the adapter.
//
// wr_idle and rd_idle are the adapter's, registered with the takes of the cycle before (so each
// value covers every entry whose credit has come back with it) and moved with the read data, and
// high only with every credit back: no request of the slice's is in a queue or on its way (the
// slice side's are registers of the next cycle's value). A read's data reaches the slice three
// cycles after the adapter gives it; the adapter's rd_idle after its last read is a cycle later
// still. a_flush crosses through the same registers as the requests, so an A read taken after it
// reaches the adapter after it (a read taken before it may then miss a held beat: it fetches its
// beat again). The adapter's reset crosses as a register per side; the slice side gives no rdy
// for HOLD cycles after it, until the memory side and the registers between are out of it too.
module otpu_xq #(
  parameter int W = 64,
  parameter int N = 16
) (
  input  logic         clk,
  input  logic         rst,
  input  logic         in_v,
  input  logic [W-1:0] in_d,
  output logic         out_v,
  output logic [W-1:0] out_d,
  input  logic         out_take
);
  localparam int NW = $clog2(N);
  (* ram_style = "distributed" *) logic [W-1:0] m [N];
  logic [NW-1:0] wp, rp;
  logic [NW:0]   n;                              // entries in m (the head register not counted)
  wire ld = n != 0 && (!out_v || out_take);
  always_ff @(posedge clk) begin
    if (in_v) m[wp] <= in_d;
    if (ld) out_d <= m[rp];
    if (rst) begin
      wp <= '0; rp <= '0; n <= '0; out_v <= 1'b0;
    end else begin
      wp <= wp + NW'(in_v);
      rp <= rp + NW'(ld);
      n <= n + (NW + 1)'(in_v) - (NW + 1)'(ld);
      if (ld) out_v <= 1'b1;
      else if (out_take) out_v <= 1'b0;
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk)
    if (!rst && in_v && n == (NW + 1)'(N) && !ld) $fatal(1, "otpu_xq: overflow");
`endif
endmodule

// the slice side: credits, the requests' and the flush's registers out, the replies' in
module otpu_xdram_s #(
  parameter int WA = 1,
  parameter int WS = 1,
  parameter int WB = 1,
  parameter int WR = 1,                          // A reply bits
  parameter int WQ = 1,                          // B reply bits
  parameter int XN = 16,
  parameter int HOLD = 12
) (
  input  logic          clk,
  input  logic          rst,
  input  logic          a_flush,
  output logic          a_rdy,
  input  logic          a_req,
  input  logic [WA-1:0] a_d,
  output logic          sw_rdy,
  input  logic          sw_req,
  input  logic [WS-1:0] sw_d,
  output logic          b_rdy,
  input  logic          b_req,
  input  logic [WB-1:0] b_d,
  output logic          a_rvalid,
  output logic [WR-1:0] a_r,
  output logic          b_rvalid,
  output logic [WQ-1:0] b_r,
  output logic          wr_idle,
  output logic          rd_idle,
  // to the memory side (sll_tx_*) and from it (into sll_rx_*)
  output logic          sll_tx_rst, sll_tx_fl, sll_tx_av, sll_tx_sv, sll_tx_bv,
  output logic [WA-1:0] sll_tx_ad,
  output logic [WS-1:0] sll_tx_sd,
  output logic [WB-1:0] sll_tx_bd,
  input  logic          m_av, m_bv,
  input  logic [WR-1:0] m_ad,
  input  logic [WQ-1:0] m_bd,
  input  logic [4:0]    m_st                     // {A, SW, B taken the cycle before, wr_idle, rd_idle}
);
  localparam int CW = $clog2(XN + 1);
  localparam int HW = $clog2(HOLD + 1);
  logic [CW-1:0] cr_a, cr_s, cr_b;
  logic [HW-1:0] hold;
  logic          sll_rx_av, sll_rx_bv;
  logic [WR-1:0] sll_rx_ad;
  logic [WQ-1:0] sll_rx_bd;
  logic [4:0]    sll_rx_st;
  wire en = hold == 0;
  assign a_rdy  = en && cr_a != 0;
  assign sw_rdy = en && cr_s != 0;
  assign b_rdy  = en && cr_b != 0;
  wire ta = a_req && a_rdy, ts = sw_req && sw_rdy, tb = b_req && b_rdy;
  wire home = cr_a == CW'(XN) && cr_s == CW'(XN) && cr_b == CW'(XN);
  // wr_idle and rd_idle are registers, loaded with what en && home && sll_rx_st give in the next
  // cycle (the credits' next values): the same values, a flip-flop at the start of their long
  // route into the slice's sequencer and DMA (the FK35's f35d0009: sll_rx_st -> the idle LUT ->
  // ic_base / w_n, 3.4 ns of route, -0.12 ns)
  wire [CW-1:0] cr_a_n = cr_a - CW'(ta) + CW'(sll_rx_st[4]);
  wire [CW-1:0] cr_s_n = cr_s - CW'(ts) + CW'(sll_rx_st[3]);
  wire [CW-1:0] cr_b_n = cr_b - CW'(tb) + CW'(sll_rx_st[2]);
  wire home_n = en ? cr_a_n == CW'(XN) && cr_s_n == CW'(XN) && cr_b_n == CW'(XN) : home;
  wire en_n = en || hold == HW'(1);
  logic wr_idle_q, rd_idle_q;
  always_ff @(posedge clk) begin
    wr_idle_q <= !rst && en_n && home_n && m_st[1];
    rd_idle_q <= !rst && en_n && home_n && m_st[0];
  end
  assign wr_idle = wr_idle_q;
  assign rd_idle = rd_idle_q;
  assign a_rvalid = sll_rx_av; assign a_r = sll_rx_ad;
  assign b_rvalid = sll_rx_bv; assign b_r = sll_rx_bd;
  always_ff @(posedge clk) begin
    sll_tx_rst <= rst;
    sll_tx_fl <= a_flush;
    sll_tx_av <= ta; sll_tx_sv <= ts; sll_tx_bv <= tb;
    sll_tx_ad <= a_d; sll_tx_sd <= sw_d; sll_tx_bd <= b_d;
    sll_rx_av <= m_av; sll_rx_bv <= m_bv; sll_rx_ad <= m_ad; sll_rx_bd <= m_bd;
    sll_rx_st <= m_st;
    if (rst) begin
      hold <= HW'(HOLD);
      cr_a <= CW'(XN); cr_s <= CW'(XN); cr_b <= CW'(XN);
      sll_tx_av <= 1'b0; sll_tx_sv <= 1'b0; sll_tx_bv <= 1'b0;
      sll_rx_av <= 1'b0; sll_rx_bv <= 1'b0; sll_rx_st <= '0;
    end else if (!en) begin
      hold <= hold - 1'b1;
    end else begin
      cr_a <= cr_a - CW'(ta) + CW'(sll_rx_st[4]);
      cr_s <= cr_s - CW'(ts) + CW'(sll_rx_st[3]);
      cr_b <= cr_b - CW'(tb) + CW'(sll_rx_st[2]);
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk)
    if (!rst && en && (cr_a > CW'(XN) || cr_s > CW'(XN) || cr_b > CW'(XN)))
      $fatal(1, "otpu_xdram_s: a credit came back that was not given");
  // the idle registers hold what the credits and the status give now
  bit q_ok;
  initial q_ok = 1'b0;
  always @(posedge clk) begin
    if (rst) q_ok <= 1'b1;
    if (q_ok && !rst && (wr_idle_q != (en && home && sll_rx_st[1]) ||
                         rd_idle_q != (en && home && sll_rx_st[0])))
      $fatal(1, "otpu_xdram_s: idle registers %0d %0d, not en %0d home %0d status %b", wr_idle_q,
             rd_idle_q, en, home, sll_rx_st[1:0]);
  end
`endif
endmodule

// the memory side: the requests' registers in, their queues, the replies' and the status's registers out
module otpu_xdram_m #(
  parameter int WA = 1,
  parameter int WS = 1,
  parameter int WB = 1,
  parameter int WR = 1,
  parameter int WQ = 1,
  parameter int XN = 16
) (
  input  logic          clk,
  output logic          rst_m,                   // the adapter's reset, crossed
  output logic          a_flush,
  input  logic          a_rdy,
  output logic          a_req,
  output logic [WA-1:0] a_d,
  input  logic          sw_rdy,
  output logic          sw_req,
  output logic [WS-1:0] sw_d,
  input  logic          b_rdy,
  output logic          b_req,
  output logic [WB-1:0] b_d,
  input  logic          a_rvalid,
  input  logic [WR-1:0] a_r,
  input  logic          b_rvalid,
  input  logic [WQ-1:0] b_r,
  input  logic          wr_idle,
  input  logic          rd_idle,
  input  logic          s_rst, s_fl, s_av, s_sv, s_bv,
  input  logic [WA-1:0] s_ad,
  input  logic [WS-1:0] s_sd,
  input  logic [WB-1:0] s_bd,
  output logic          sll_tx_av, sll_tx_bv,
  output logic [WR-1:0] sll_tx_ad,
  output logic [WQ-1:0] sll_tx_bd,
  output logic [4:0]    sll_tx_st,
  // the channels' read and write beats this cycle, registered for the board's counters (the slice's SLR)
  input  logic [3:0]    n_rvalid, n_wvalid, n_wready,
  output logic [3:0]    n_rbeat, n_wbeat
);
  logic          sll_rx_rst, sll_rx_fl, sll_rx_av, sll_rx_sv, sll_rx_bv;
  logic [WA-1:0] sll_rx_ad;
  logic [WS-1:0] sll_rx_sd;
  logic [WB-1:0] sll_rx_bd;
  logic [2:0]    tk_d;                           // A, SW, B taken (last cycle)
  // the replies and the status a register before the crossing's (p_*): the adapter's read data
  // leave its read buffers' block RAMs through their selection, then cross the SLR from a flip-flop
  // (f35d0009: u_rbm -> 3 LUTs -> sll_tx_bd at the Laguna site, -0.18 ns). The whole bundle moves
  // a cycle later, the status with the data, so rd_idle still follows the last read's data
  logic          p_av, p_bv;
  logic [WR-1:0] p_ad;
  logic [WQ-1:0] p_bd;
  logic [4:0]    p_st;
  assign rst_m = sll_rx_rst;
  assign a_flush = sll_rx_fl;
  always_ff @(posedge clk) begin
    sll_rx_rst <= s_rst;
    sll_rx_fl <= s_fl;
    sll_rx_av <= s_av; sll_rx_sv <= s_sv; sll_rx_bv <= s_bv;
    sll_rx_ad <= s_ad; sll_rx_sd <= s_sd; sll_rx_bd <= s_bd;
    tk_d <= {a_req && a_rdy, sw_req && sw_rdy, b_req && b_rdy};
    p_av <= a_rvalid; p_bv <= b_rvalid; p_ad <= a_r; p_bd <= b_r;
    p_st <= {tk_d, wr_idle, rd_idle};
    sll_tx_av <= p_av; sll_tx_bv <= p_bv; sll_tx_ad <= p_ad; sll_tx_bd <= p_bd;
    sll_tx_st <= p_st;
    n_rbeat <= n_rvalid;
    n_wbeat <= n_wvalid & n_wready;
    if (rst_m) begin
      sll_rx_av <= 1'b0; sll_rx_sv <= 1'b0; sll_rx_bv <= 1'b0;
      tk_d <= '0; p_av <= 1'b0; p_bv <= 1'b0; p_st <= '0;
      sll_tx_av <= 1'b0; sll_tx_bv <= 1'b0; sll_tx_st <= '0;
    end
  end
  otpu_xq #(.W(WA), .N(XN)) u_qa (
    .clk, .rst(rst_m), .in_v(sll_rx_av), .in_d(sll_rx_ad), .out_v(a_req), .out_d(a_d),
    .out_take(a_req && a_rdy));
  otpu_xq #(.W(WS), .N(XN)) u_qs (
    .clk, .rst(rst_m), .in_v(sll_rx_sv), .in_d(sll_rx_sd), .out_v(sw_req), .out_d(sw_d),
    .out_take(sw_req && sw_rdy));
  otpu_xq #(.W(WB), .N(XN)) u_qb (
    .clk, .rst(rst_m), .in_v(sll_rx_bv), .in_d(sll_rx_bd), .out_v(b_req), .out_d(b_d),
    .out_take(b_req && b_rdy));
endmodule

module otpu_native_xdram2 #(
  parameter int D = 128,
  parameter bit CHASH = 1'b1,
  parameter int RD = 128,
  parameter int OQ = 512,
  parameter int AW = 32,
  parameter int BAW = 25,
  parameter int QMEM = 0,
  parameter int XN = 16                              // entries a port's queue (credits)
) (
  input  logic              clk,
  input  logic              rst,
  input  logic              a_flush,
  output logic              a_rdy_x,
  input  logic              a_req_x,
  input  logic              a_we_x,
  input  logic [AW-1:0]     a_addr_x,
  input  logic [31:0]       a_wdata_x,
  input  logic [3:0]        a_be_x,
  output logic              a_rvalid,
  output logic [255:0]      a_rdata,
  output logic              sw_rdy,
  input  logic              sw_req,
  input  logic [AW-1:0]     sw_addr,
  input  logic [31:0]       sw_wdata,
  input  logic [3:0]        sw_be,
  output logic              b_rdy,
  input  logic              b_req,
  input  logic              b_tag,
  input  logic              b_we,
  input  logic [D/4-1:0]    b_wmask,
  input  logic [D*8-1:0]    b_wdata,
  input  logic [AW-1:0]     b_addr,
  input  logic              b_par,
  input  logic              b_two,
  output logic              b_rvalid,
  output logic              b_rtag,
  output logic [D*8-1:0]    b_rdata,
  output logic [D*8-1:0]    b_rdata2,
  output logic              wr_idle,
  output logic              rd_idle,
  output logic [3:0]        n_cvalid,
  input  logic [3:0]        n_cready,
  output logic [3:0]        n_cwe,
  output logic [3:0][BAW-1:0] n_caddr,
  output logic [3:0]        n_wvalid,
  input  logic [3:0]        n_wready,
  output logic [3:0][511:0] n_wdata,
  output logic [3:0][63:0]  n_wmask,
  input  logic [3:0]        n_rvalid,
  input  logic [3:0][511:0] n_rdata,
  input  logic [3:0][15:0]  n_wdone,
  output logic [3:0]        n_rbeat,    // n_rvalid, n_wvalid && n_wready a cycle late (otpu_board's
  output logic [3:0]        n_wbeat     //   DRAM_RD / DRAM_WR counts, in the slice's SLR)
);
  localparam int WA = 1 + AW + 32 + 4;               // {we, addr, wdata, be}
  localparam int WS = AW + 32 + 4;                   // {addr, wdata, be}
  localparam int WB = 2 + D / 4 + D * 8 + AW + 2;    // {tag, we, wmask, wdata, addr, par, two}
  localparam int WR = 256;
  localparam int WQ = 1 + 2 * D * 8;                 // {rtag, rdata, rdata2}

  // the adapter's side of the crossing
  logic          m_rst, m_flush, m_a_rdy, m_a_req, m_sw_rdy, m_sw_req, m_b_rdy, m_b_req;
  logic [WA-1:0] m_a_d;
  logic [WS-1:0] m_sw_d;
  logic [WB-1:0] m_b_d;
  logic          m_a_rvalid, m_b_rvalid, m_b_rtag, m_wr_idle, m_rd_idle;
  logic [255:0]  m_a_rdata;
  logic [D*8-1:0] m_b_rdata, m_b_rdata2;
  // between the sides
  logic          x_rst, x_fl, x_av, x_sv, x_bv, y_av, y_bv;
  logic [WA-1:0] x_ad;
  logic [WS-1:0] x_sd;
  logic [WB-1:0] x_bd;
  logic [WR-1:0] y_ad, s_a_r;
  logic [WQ-1:0] y_bd, s_b_r;
  logic [4:0]    y_st;
  logic [3:0]    m_n_wvalid;
  assign n_wvalid = m_n_wvalid;

  (* KEEP_HIERARCHY = "YES" *)   // (the floorplan places u_s, u_m and u_dram by name)
  otpu_xdram_s #(.WA(WA), .WS(WS), .WB(WB), .WR(WR), .WQ(WQ), .XN(XN)) u_s (
    .clk, .rst, .a_flush,
    .a_rdy(a_rdy_x), .a_req(a_req_x), .a_d({a_we_x, a_addr_x, a_wdata_x, a_be_x}),
    .sw_rdy, .sw_req, .sw_d({sw_addr, sw_wdata, sw_be}),
    .b_rdy, .b_req, .b_d({b_tag, b_we, b_wmask, b_wdata, b_addr, b_par, b_two}),
    .a_rvalid, .a_r(s_a_r), .b_rvalid, .b_r(s_b_r), .wr_idle, .rd_idle,
    .sll_tx_rst(x_rst), .sll_tx_fl(x_fl), .sll_tx_av(x_av), .sll_tx_sv(x_sv), .sll_tx_bv(x_bv),
    .sll_tx_ad(x_ad), .sll_tx_sd(x_sd), .sll_tx_bd(x_bd),
    .m_av(y_av), .m_bv(y_bv), .m_ad(y_ad), .m_bd(y_bd), .m_st(y_st));
  assign a_rdata = s_a_r;
  assign {b_rtag, b_rdata, b_rdata2} = s_b_r;

  (* KEEP_HIERARCHY = "YES" *)
  otpu_xdram_m #(.WA(WA), .WS(WS), .WB(WB), .WR(WR), .WQ(WQ), .XN(XN)) u_m (
    .clk, .rst_m(m_rst), .a_flush(m_flush),
    .a_rdy(m_a_rdy), .a_req(m_a_req), .a_d(m_a_d), .sw_rdy(m_sw_rdy), .sw_req(m_sw_req),
    .sw_d(m_sw_d), .b_rdy(m_b_rdy), .b_req(m_b_req), .b_d(m_b_d),
    .a_rvalid(m_a_rvalid), .a_r(m_a_rdata), .b_rvalid(m_b_rvalid),
    .b_r({m_b_rtag, m_b_rdata, m_b_rdata2}), .wr_idle(m_wr_idle), .rd_idle(m_rd_idle),
    .s_rst(x_rst), .s_fl(x_fl), .s_av(x_av), .s_sv(x_sv), .s_bv(x_bv),
    .s_ad(x_ad), .s_sd(x_sd), .s_bd(x_bd),
    .sll_tx_av(y_av), .sll_tx_bv(y_bv), .sll_tx_ad(y_ad), .sll_tx_bd(y_bd), .sll_tx_st(y_st),
    .n_rvalid, .n_wvalid(m_n_wvalid), .n_wready, .n_rbeat, .n_wbeat);

  logic          d_a_we, d_b_tag, d_b_we, d_b_par, d_b_two;
  logic [AW-1:0] d_a_addr, d_sw_addr, d_b_addr;
  logic [31:0]   d_a_wdata, d_sw_wdata;
  logic [3:0]    d_a_be, d_sw_be;
  logic [D/4-1:0] d_b_wmask;
  logic [D*8-1:0] d_b_wdata;
  assign {d_a_we, d_a_addr, d_a_wdata, d_a_be} = m_a_d;
  assign {d_sw_addr, d_sw_wdata, d_sw_be} = m_sw_d;
  assign {d_b_tag, d_b_we, d_b_wmask, d_b_wdata, d_b_addr, d_b_par, d_b_two} = m_b_d;

  (* KEEP_HIERARCHY = "YES" *)
  otpu_native_dram2 #(.D(D), .CHASH(CHASH), .RD(RD), .OQ(OQ), .AW(AW), .BAW(BAW), .QMEM(QMEM)) u_dram (
    .clk, .rst(m_rst), .a_flush(m_flush),
    .a_rdy_x(m_a_rdy), .a_req_x(m_a_req), .a_we_x(d_a_we), .a_addr_x(d_a_addr),
    .a_wdata_x(d_a_wdata), .a_be_x(d_a_be), .a_rvalid(m_a_rvalid), .a_rdata(m_a_rdata),
    .sw_rdy(m_sw_rdy), .sw_req(m_sw_req), .sw_addr(d_sw_addr), .sw_wdata(d_sw_wdata),
    .sw_be(d_sw_be),
    .b_rdy(m_b_rdy), .b_req(m_b_req), .b_tag(d_b_tag), .b_we(d_b_we), .b_wmask(d_b_wmask),
    .b_wdata(d_b_wdata), .b_addr(d_b_addr), .b_par(d_b_par), .b_two(d_b_two),
    .b_rvalid(m_b_rvalid), .b_rtag(m_b_rtag), .b_rdata(m_b_rdata), .b_rdata2(m_b_rdata2),
    .wr_idle(m_wr_idle), .rd_idle(m_rd_idle),
    .n_cvalid, .n_cready, .n_cwe, .n_caddr, .n_wvalid(m_n_wvalid), .n_wready, .n_wdata, .n_wmask,
    .n_rvalid, .n_rdata, .n_wdone);
endmodule
