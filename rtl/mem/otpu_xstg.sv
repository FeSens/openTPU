// A slice's interface to its DRAM adapter (otpu_native_dram) through N register stages each way
// (otpu_board / otpu_top NAT_XSTG; N 0: the signals wired, no cells). For a floorplan that puts
// the adapters and their channels' core halves an SLR away from their slices (the WG targets'):
// every signal between a slice and its adapter then crosses in flip-flops.
//
// A direction is N registers with nothing between them (otpu_xstg_p: g_s[i].r_reg, stage 0 at
// the source; no reset, no enable, no shift-register extraction, so that a floorplan can put
// them on the SLR crossing's sites). They start at 0 (the FPGA's configuration, the simulators'
// --x-initial 0) and their sources hold 0 in reset, which lasts longer than N cycles.
//
// Request ports (A, SW, B, W). The slice keeps its contract (a request held in its port register
// until rdy; rdy from registers only). Its request bit, the rdy it saw and the fields go forward,
// and the adapter's side takes a request where both bits arrive set, into a queue of
// QN >= 2N + 2 entries (otpu_xstg_q) whose head is the adapter's request. The slice's rdy is the
// queue's room for 2N + 1 more, N stages back. A rdy the slice sees is the room of N cycles
// before, and a request it lets go arrives N cycles later, so at most 2N + 1 requests arrive on
// one room: the ones already in flight both ways, and this cycle's. No logic of the handshake
// sits on the slice's side. Port A's and W's queue heads are flip-flops (otpu_xstg_q HR: read
// ahead), as the slice's port registers were: the adapter takes them into the channel hash, the
// run compares and its queues' write enables in one cycle (port A's grant chain at 150 MHz on the
// m8 card already 12 levels); B's and SW's are the LUT RAM's read at a registered address (their
// inputs came late from the slice's grant chains, b_par its fold, carried as a field).
// Reads come back in N stages, and so do the adapter's wr_idle and rd_idle, each with its queues'
// (a request queued is not yet the adapter's). A request the slice sent in the last 2N cycles is
// in neither: in flight, or queued with its queue's state not yet back. So the slice's side holds
// wr_idle low for the 2N cycles after each write it sends, and rd_idle after each read (w_h, r_h:
// 2N flip-flops each and a LUT, the only logic on the slice's side).
// The adapter's reset (m_rst: the slice's sys_rst) and a_flush's terms (rst, ld_start, a_inval at
// the slice) cross in N stages: the adapter leaves reset N cycles after the slice, before any
// request can arrive; a flush reaches it before the requests the slice sends after it.
// The posted W beats (XPOST: p_we, p_q, p_slot, p_d on p_clk, from the channels to the MXU) cross
// in NP p_clk stages of their own (default N: p_clk is the faster clock, the bridges' AXI clock, so
// it may want more), p_slot as one copy, its copies made in the last stage (g_psc[k], the
// MXU's RAM groups'); the channels' counts of them (p_cntg) take NP p_clk stages too
// before the adapter's synchronizer. A count thus reaches its synchronizer no earlier than its
// beats reach the MXU, by construction, whatever the clocks: the count's crossing into the core
// clock (set_max_delay -datapath_only and set_bus_skew from the last p_cntg stage: bcu1525_wg_top.tcl)
// is the same as without stages, and the beats' last stage into the MXU's staging RAMs is a p_clk
// path. The adapter's decoded counts (p_cnt) then cross to the slice in N core-clock stages:
// later still, so p_cnt is never ahead of its beats.
module otpu_xstg #(
  parameter int N   = 0,
  parameter int NP  = N,                   // the posts' and their counts' stages (p_clk)
  parameter int D   = 128,
  parameter int NCH = 2,
  parameter int CB  = 64,
  parameter int AB  = 32,                  // port A's read bytes (otpu_native_dram AB)
  parameter int XSW = 9,                   // a chunk FIFO entry's bits (otpu_native_dram XSW)
  parameter int XQB = 0,                   // a posted beat's writes, 2^XQB (otpu_board NAT_XQB)
  parameter int AW  = 32,                  // byte address bits (otpu_seq AW)
  parameter int SWW = 1,                   // the SW port's words (otpu_quant SWW)
  parameter int WA  = (AW > 34) ? AW - 2 : 32  // the ports' word address bits (derived)
) (
  input  logic clk,
  input  logic p_clk,                      // the posts' clock (otpu_board p_clk)
  // ---- the slice's side (otpu_slice's port registers)
  input  logic                    s_rst,   // the slice's sys_rst
  input  logic [2:0]              s_flush, // a_flush's terms: rst, ld_start, a_inval
  output logic                    s_a_rdy,
  input  logic                    s_a_req,
  input  logic                    s_a_we,
  input  logic [WA-1:0]           s_a_addr,
  input  logic [31:0]             s_a_wdata,
  input  logic [3:0]              s_a_be,
  output logic                    s_a_rvalid,
  output logic [AB*8-1:0]         s_a_rdata,
  output logic                    s_sw_rdy,
  input  logic                    s_sw_req,
  input  logic [WA-1:0]           s_sw_addr,
  input  logic [32*SWW-1:0]       s_sw_wdata,
  input  logic [4*SWW-1:0]        s_sw_be,
  output logic                    s_b_rdy,
  input  logic                    s_b_req,
  input  logic                    s_b_tag,
  input  logic                    s_b_we,
  input  logic [D/4-1:0]          s_b_wmask,
  input  logic [D*8-1:0]          s_b_wdata,
  input  logic [WA-1:0]           s_b_addr,
  input  logic [$clog2(NCH)-1:0]  s_b_par,
  output logic                    s_b_rvalid,
  output logic                    s_b_rtag,
  output logic [D*8-1:0]          s_b_rdata,
  output logic                    s_w_rdy,
  input  logic                    s_w_req,
  input  logic [WA-1:0]           s_w_addr,
  input  logic [XSW-1:0]          s_w_slot,
  input  logic                    s_w_one,
  output logic [NCH-1:0]          s_w_bv,
  output logic [NCH*CB*8-1:0]     s_w_bd,
  output logic                    s_wr_idle,
  output logic                    s_rd_idle,
  output logic [NCH-1:0][XSW+1:0] s_p_cnt,
  output logic [NCH-1:0]          s_p_we,
  output logic [NCH-1:0][1:0][((XQB > 0) ? (1 << XQB) : 1)-1:0] s_p_q,
  output logic [NCH-1:0][4:0][XSW-1:0] s_p_slot,
  output logic [NCH-1:0][(D*8 >> XQB)-1:0] s_p_d,
  // ---- the adapter's side (otpu_native_dram's slice side)
  output logic                    m_rst,
  output logic                    m_flush,
  input  logic                    m_a_rdy,
  output logic                    m_a_req,
  output logic                    m_a_we,
  output logic [WA-1:0]           m_a_addr,
  output logic [31:0]             m_a_wdata,
  output logic [3:0]              m_a_be,
  input  logic                    m_a_rvalid,
  input  logic [AB*8-1:0]         m_a_rdata,
  input  logic                    m_sw_rdy,
  output logic                    m_sw_req,
  output logic [WA-1:0]           m_sw_addr,
  output logic [32*SWW-1:0]       m_sw_wdata,
  output logic [4*SWW-1:0]        m_sw_be,
  input  logic                    m_b_rdy,
  output logic                    m_b_req,
  output logic                    m_b_tag,
  output logic                    m_b_we,
  output logic [D/4-1:0]          m_b_wmask,
  output logic [D*8-1:0]          m_b_wdata,
  output logic [WA-1:0]           m_b_addr,
  output logic [$clog2(NCH)-1:0]  m_b_par,
  input  logic                    m_b_rvalid,
  input  logic                    m_b_rtag,
  input  logic [D*8-1:0]          m_b_rdata,
  input  logic                    m_w_rdy,
  output logic                    m_w_req,
  output logic [WA-1:0]           m_w_addr,
  output logic [XSW-1:0]          m_w_slot,
  output logic                    m_w_one,
  input  logic [NCH-1:0]          m_w_bv,
  input  logic [NCH*CB*8-1:0]     m_w_bd,
  input  logic                    m_wr_idle,
  input  logic                    m_rd_idle,
  input  logic [NCH-1:0][XSW+1:0] m_p_cnt,
  output logic [NCH-1:0][XSW+1:0] m_p_cntg,
  // ---- the channels' posts (p_clk)
  input  logic [NCH-1:0]          c_p_we,
  input  logic [NCH-1:0][1:0][((XQB > 0) ? (1 << XQB) : 1)-1:0] c_p_q,
  input  logic [NCH-1:0][4:0][XSW-1:0] c_p_slot,
  input  logic [NCH-1:0][(D*8 >> XQB)-1:0] c_p_d,
  input  logic [NCH-1:0][XSW+1:0] c_p_cntg
);
  localparam int CL = $clog2(NCH), PW = (D * 8) >> XQB, QH = (XQB > 0) ? 1 << XQB : 1;
  initial if (N < 0 || N > 8 || NP < 0 || NP > 8)
    $fatal(1, "otpu_xstg: N %0d, NP %0d (0 .. 8 stages)", N, NP);

  if (N == 0) begin : g_wire
    assign m_rst = s_rst, m_flush = |s_flush;
    assign s_a_rdy = m_a_rdy, m_a_req = s_a_req, m_a_we = s_a_we, m_a_addr = s_a_addr;
    assign m_a_wdata = s_a_wdata, m_a_be = s_a_be, s_a_rvalid = m_a_rvalid, s_a_rdata = m_a_rdata;
    assign s_sw_rdy = m_sw_rdy, m_sw_req = s_sw_req, m_sw_addr = s_sw_addr;
    assign m_sw_wdata = s_sw_wdata, m_sw_be = s_sw_be;
    assign s_b_rdy = m_b_rdy, m_b_req = s_b_req, m_b_tag = s_b_tag, m_b_we = s_b_we;
    assign m_b_wmask = s_b_wmask, m_b_wdata = s_b_wdata, m_b_addr = s_b_addr, m_b_par = s_b_par;
    assign s_b_rvalid = m_b_rvalid, s_b_rtag = m_b_rtag, s_b_rdata = m_b_rdata;
    assign s_w_rdy = m_w_rdy, m_w_req = s_w_req, m_w_addr = s_w_addr, m_w_slot = s_w_slot;
    assign m_w_one = s_w_one, s_w_bv = m_w_bv, s_w_bd = m_w_bd;
    assign s_wr_idle = m_wr_idle, s_rd_idle = m_rd_idle, s_p_cnt = m_p_cnt;
  end else begin : g_stg
    // ---- the adapter's reset and a_flush's terms
    logic [2:0] fl;
    otpu_xstg_p #(.W(4), .N(N)) u_ctl (.clk, .d({s_rst, s_flush}), .q({m_rst, fl}));
    assign m_flush = |fl;

    // ---- the request ports: a queue each on the adapter's side
    logic a_none, sw_none, b_none, w_none;
    otpu_xstg_q #(.W(WA + 37), .N(N), .HR(1'b1)) u_qa (
      .clk, .m_rst, .s_req(s_a_req), .s_d({s_a_we, s_a_addr, s_a_wdata, s_a_be}), .s_rdy(s_a_rdy),
      .m_req(m_a_req), .m_d({m_a_we, m_a_addr, m_a_wdata, m_a_be}), .m_rdy(m_a_rdy),
      .m_none(a_none));
    otpu_xstg_q #(.W(WA + 36 * SWW), .N(N)) u_qs (
      .clk, .m_rst, .s_req(s_sw_req), .s_d({s_sw_addr, s_sw_wdata, s_sw_be}), .s_rdy(s_sw_rdy),
      .m_req(m_sw_req), .m_d({m_sw_addr, m_sw_wdata, m_sw_be}), .m_rdy(m_sw_rdy),
      .m_none(sw_none));
    otpu_xstg_q #(.W(2 + D / 4 + D * 8 + WA + CL), .N(N)) u_qb (
      .clk, .m_rst, .s_req(s_b_req),
      .s_d({s_b_tag, s_b_we, s_b_wmask, s_b_wdata, s_b_addr, s_b_par}), .s_rdy(s_b_rdy),
      .m_req(m_b_req), .m_d({m_b_tag, m_b_we, m_b_wmask, m_b_wdata, m_b_addr, m_b_par}),
      .m_rdy(m_b_rdy), .m_none(b_none));
    if (NCH * CB > D) begin : g_w                // port W (otpu_native_dram HAS_W)
      otpu_xstg_q #(.W(WA + 1 + XSW), .N(N), .HR(1'b1)) u_qw (
        .clk, .m_rst, .s_req(s_w_req), .s_d({s_w_addr, s_w_slot, s_w_one}), .s_rdy(s_w_rdy),
        .m_req(m_w_req), .m_d({m_w_addr, m_w_slot, m_w_one}), .m_rdy(m_w_rdy), .m_none(w_none));
    end else begin : g_no_w                      // never ready, never valid (as the adapter's)
      assign s_w_rdy = 1'b0, m_w_req = 1'b0, m_w_addr = '0, m_w_slot = '0, m_w_one = 1'b0;
      assign w_none = 1'b1;
    end

    // ---- reads, the adapter's idles with its queues', its decoded post counts: N stages back
    otpu_xstg_p #(.W(1 + AB * 8), .N(N)) u_ra (.clk, .d({m_a_rvalid, m_a_rdata}),
                                              .q({s_a_rvalid, s_a_rdata}));
    otpu_xstg_p #(.W(2 + D * 8), .N(N)) u_rb (.clk, .d({m_b_rvalid, m_b_rtag, m_b_rdata}),
                                             .q({s_b_rvalid, s_b_rtag, s_b_rdata}));
    otpu_xstg_p #(.W(NCH + NCH * CB * 8), .N(N)) u_rw (.clk, .d({m_w_bv, m_w_bd}),
                                                      .q({s_w_bv, s_w_bd}));
    otpu_xstg_p #(.W(NCH * (XSW + 2)), .N(N)) u_rc (.clk, .d(m_p_cnt), .q(s_p_cnt));
    logic wi_d, ri_d;
    otpu_xstg_p #(.W(2), .N(N)) u_ri (
      .clk, .d({m_wr_idle && a_none && sw_none && b_none, m_rd_idle && a_none && b_none && w_none}),
      .q({wi_d, ri_d}));
    // the slice's sends of the last 2N cycles (bit 0: the last cycle's)
    logic [2*N-1:0] w_h, r_h;
    always_ff @(posedge clk)
      if (s_rst) begin
        w_h <= '0;
        r_h <= '0;
      end else begin
        w_h <= {w_h[2*N-2:0], (s_a_req && s_a_we && s_a_rdy) || (s_sw_req && s_sw_rdy) ||
                              (s_b_req && s_b_we && s_b_rdy)};
        r_h <= {r_h[2*N-2:0], (s_a_req && !s_a_we && s_a_rdy) || (s_b_req && !s_b_we && s_b_rdy) ||
                              (s_w_req && s_w_rdy)};
      end
`ifndef OTPU_MUT_XHIST
    assign s_wr_idle = wi_d && !(|w_h);
    assign s_rd_idle = ri_d && !(|r_h);
`else                                            // (mutant: the idles without the sends in flight)
    assign s_wr_idle = wi_d;
    assign s_rd_idle = ri_d;
`endif
  end

  // ---- the posted beats and their counts, NP p_clk stages each
  otpu_xstg_p #(.W(NCH * (1 + 2 * QH + PW)), .N(NP)) u_pp (
    .clk(p_clk), .d({c_p_we, c_p_q, c_p_d}), .q({s_p_we, s_p_q, s_p_d}));
  if (NP == 0) begin : g_ps_wire
    assign s_p_slot = c_p_slot;
  end else begin : g_ps                           // one copy, then each copy's last stage
    logic [NCH-1:0][XSW-1:0] sl0, sl;
    for (genvar c = 0; c < NCH; c++) begin : g_c
      assign sl0[c] = c_p_slot[c][0];
    end
    otpu_xstg_p #(.W(NCH * XSW), .N(NP - 1)) u_ps (.clk(p_clk), .d(sl0), .q(sl));
    for (genvar k = 0; k < 5; k++) begin : g_psc
      logic [NCH-1:0][XSW-1:0] q;
      otpu_xstg_p #(.W(NCH * XSW), .N(1)) u_p (.clk(p_clk), .d(sl), .q(q));
      for (genvar c = 0; c < NCH; c++) begin : g_c
        assign s_p_slot[c][k] = q[c];
      end
    end
  end
  for (genvar c = 0; c < NCH; c++) begin : g_pc     // a channel's count: its own cells (the
    otpu_xstg_p #(.W(XSW + 2), .N(NP)) u_p (         //   bus-skew constraint's, bcu1525_wg_top.tcl)
      .clk(p_clk), .d(c_p_cntg[c]), .q(m_p_cntg[c]));
  end
endmodule

// N register stages, nothing between them; g_s[0] at the source. N 0: a wire.
module otpu_xstg_p #(
  parameter int W = 1,
  parameter int N = 1
) (
  input  logic         clk,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  if (N == 0) begin : g_wire
    assign q = d;
  end else begin : g_pipe
    for (genvar i = 0; i < N; i++) begin : g_s
      (* shreg_extract = "no" *) logic [W-1:0] r;   // r_reg: a stage (USER_SLL_REG cells, as otpu_xr's)
      if (i == 0) begin : g_first
        always_ff @(posedge clk) r <= d;
      end else begin : g_next
        always_ff @(posedge clk) r <= g_s[i-1].r;
      end
    end
    assign q = g_s[N-1].r;
  end
endmodule

// A request port across N stages (otpu_xstg): forward the slice's request bit, the rdy it saw
// and its fields; on the adapter's side a queue of QN entries (LUT RAM, the head read at its
// address) takes a request where both bits arrive set, and its head is the adapter's request.
// HR: the head (m_req, m_d) from flip-flops instead, the same cycle by cycle: on a pop the RAM's
// word at the next head's address (rp, a register: read ahead), and a request arriving at a
// queue empty but for the head popped goes into it as well (write through).
// The slice's rdy: the queue's room for 2N + 1 more requests, N stages back (registered: the
// slice's contract). The queue resets with the adapter (m_rst); m_none: it holds no request.
module otpu_xstg_q #(
  parameter int W  = 1,
  parameter int N  = 1,
  parameter bit HR = 1'b0                    // the head from flip-flops
) (
  input  logic         clk,
  input  logic         m_rst,
  input  logic         s_req,
  input  logic [W-1:0] s_d,
  output logic         s_rdy,
  output logic         m_req,
  output logic [W-1:0] m_d,
  input  logic         m_rdy,
  output logic         m_none
);
  localparam int QN = 2 ** $clog2(2 * N + 2), QW = $clog2(QN);
  logic [1:0] fv;                            // the request bit and the rdy it went with
  logic [W-1:0] fd;
  otpu_xstg_p #(.W(2 + W), .N(N)) u_f (.clk, .d({s_req, s_rdy, s_d}), .q({fv, fd}));
  wire push = fv[1] && fv[0];
  wire pop = m_req && m_rdy;
  (* ram_style = "distributed" *) logic [W-1:0] mem [QN];
  logic [QW-1:0] wp, rp;                     // rp: the head's address (HR: the next head's)
  logic [QW:0] n;                            // entries held
  always_ff @(posedge clk) begin
    if (push) mem[wp] <= fd;
    if (m_rst) begin
      wp <= '0;
      rp <= QW'(HR);
      n <= '0;
    end else begin
      wp <= wp + QW'(push);
      rp <= rp + QW'(pop);
      n <= n + (QW + 1)'(push) - (QW + 1)'(pop);
    end
  end
  if (HR) begin : g_hr
    logic hv;
    logic [W-1:0] hd;
    always_ff @(posedge clk) begin
      hv <= !m_rst && n + (QW + 1)'(push) - (QW + 1)'(pop) != '0;
`ifndef OTPU_MUT_XHEAD
      if (n == (QW + 1)'(pop)) hd <= fd;     // empty but for the head popped: the request arriving
`else                                        // (mutant: no write through behind a pop)
      if (n == '0) hd <= fd;
`endif
      else if (pop) hd <= mem[rp];
    end
    assign m_req = hv;
    assign m_d = hd;
  end else begin : g_ra
    assign m_req = n != '0;
    assign m_d = mem[rp];
  end
  assign m_none = n == '0;
`ifndef OTPU_MUT_XROOM
  localparam int ROOM = 2 * N + 1;
`else
  localparam int ROOM = 2 * N;                   // (mutant: room for one request too few)
`endif
  otpu_xstg_p #(.W(1), .N(N)) u_r (.clk, .d((QW + 1)'(QN) - n >= (QW + 1)'(ROOM)), .q(s_rdy));
`ifndef SYNTHESIS
`ifndef OTPU_WO_UNCHECKED                         // (the mutants' run: the testbench must catch it)
  always @(posedge clk)
    if (!m_rst && push && !pop && n == (QW + 1)'(QN))
      $fatal(1, "otpu_xstg_q: a request arrived at a full queue (%0d entries)", QN);
`endif
`endif
endmodule
