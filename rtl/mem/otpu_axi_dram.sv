// Slice DRAM ports (as otpu_dram) on two AXI4 memory channels (the board's two DDR3
// controllers), 512-bit data, one 64-byte beat per transaction.
//
// Address map: the slice's byte address space is interleaved over the channels in 64-byte
// beats: logical beat b = addr / 64 lives on channel b % 2 at BASE[b % 2] + (b / 2) * 64. A
// D = 128-byte chunk (port B; requests are chunk aligned) is therefore one beat on each
// channel, and a streamed operand uses both channels evenly. The host applies the same map
// when it fills and reads the memory (host/board.py).
//
// Port B: chunk reads and word-masked chunk writes. Port A: single-word reads and byte-enabled
// word writes. Port SW: byte-enabled word writes (the quantizer's QST stores), independent of A
// so that they never hold up the MXU's scale reads. A reads that fall in the beat of the previous A read (the MXU's scale stream:
// 16 scales per beat) reuse it without a DRAM access, until any write is accepted.
// Requests are taken when req && rdy; rdy depends on registered state only. Reads return in
// order per port (the B tag with its data). Responses never back up: a read beat is issued on
// AXI only when its response FIFO has room reserved. wr_idle: every accepted write has its
// AXI write response.
module otpu_axi_dram #(
  parameter int D = 128,
  parameter int QD = 4,                              // request queue depth per channel and port
  parameter int RD = 64,                             // B read beats in flight per channel
  parameter int AD = 16,                             // A read beats in flight per channel
  parameter logic [31:0] BASE0 = 32'h0000_0000,
  parameter logic [31:0] BASE1 = 32'h8000_0000
) (
  input  logic              clk,
  input  logic              rst,
  // slice side
  output logic              a_rdy,
  input  logic              a_req,
  input  logic              a_we,
  input  logic [31:0]       a_addr,     // word address
  input  logic [31:0]       a_wdata,
  input  logic [3:0]        a_be,
  output logic              a_rvalid,
  output logic [31:0]       a_rdata,
  output logic              sw_rdy,
  input  logic              sw_req,
  input  logic [31:0]       sw_addr,    // word address
  input  logic [31:0]       sw_wdata,
  input  logic [3:0]        sw_be,
  output logic              b_rdy,
  input  logic              b_req,
  input  logic              b_tag,
  input  logic              b_we,
  input  logic [D/4-1:0]    b_wmask,
  input  logic [D*8-1:0]    b_wdata,
  input  logic [31:0]       b_addr,     // word address (chunk aligned)
  output logic              b_rvalid,
  output logic              b_rtag,
  output logic [D*8-1:0]    b_rdata,
  output logic              wr_idle,
  // AXI4 masters, one per channel (ID 0: port B, ID 1: port A)
  output logic [1:0]            m_awvalid,
  input  logic [1:0]            m_awready,
  output logic [1:0][31:0]      m_awaddr,
  output logic [1:0]            m_awid,
  output logic [1:0]            m_wvalid,
  input  logic [1:0]            m_wready,
  output logic [1:0][511:0]     m_wdata,
  output logic [1:0][63:0]      m_wstrb,
  input  logic [1:0]            m_bvalid,
  output logic [1:0]            m_bready,
  input  logic [1:0]            m_bid,
  input  logic [1:0][1:0]       m_bresp,
  output logic [1:0]            m_arvalid,
  input  logic [1:0]            m_arready,
  output logic [1:0][31:0]      m_araddr,
  output logic [1:0]            m_arid,
  input  logic [1:0]            m_rvalid,
  output logic [1:0]            m_rready,
  input  logic [1:0]            m_rid,
  input  logic [1:0][511:0]     m_rdata,
  input  logic [1:0][1:0]       m_rresp,
  input  logic [1:0]            m_rlast,
  output logic                  err         // sticky: an AXI error response
);
  initial if (D != 128) $fatal(1, "otpu_axi_dram: D must be 128 (one beat per channel)");
  localparam int QW = $clog2(QD);
  localparam int RW = $clog2(RD);
  localparam int AW_ = $clog2(AD);
  localparam int OD = 2 * RD;                        // B tags / A order entries in flight (2^k)
  localparam int OW = $clog2(OD);

  function automatic logic [31:0] chan_addr(input logic [31:0] word_addr, input logic c);
    logic [31:0] beat;
    beat = word_addr >> 4;
    return (c ? BASE1 : BASE0) + ((beat >> 1) << 6);
  endfunction

  // ------------------------------------------------------------------ request queues
  // per channel: qb (port B beats), qa (port A beats)
  typedef struct packed {
    logic         we;
    logic [31:0]  addr;        // AXI address
    logic [511:0] data;
    logic [15:0]  wmask;       // word enables
  } qb_t;
  typedef struct packed {
    logic         we;
    logic [31:0]  addr;
    logic [3:0]   idx;         // word in the beat
    logic [31:0]  data;
    logic [3:0]   be;
  } qa_t;

  qb_t        qb [2][QD];
  logic [QW:0] qb_n [2];
  logic [QW-1:0] qb_h [2];
  qa_t        qa [2][QD];
  logic [QW:0] qa_n [2];
  logic [QW-1:0] qa_h [2];
  qa_t        qw [2][QD];                   // port SW writes
  logic [QW:0] qw_n [2];
  logic [QW-1:0] qw_h [2];

  // order of B reads (tags) and of A reads (channel, word, reuse)
  logic         bt_q [OD];
  logic [OW:0]  bt_n;
  logic [OW-1:0] bt_h;
  typedef struct packed { logic c; logic [3:0] idx; logic reuse; } ao_t;
  ao_t          ao_q [OD];
  logic [OW:0]  ao_n;
  logic [OW-1:0] ao_h;

  // A beat reuse
  logic         al_v;
  logic [27:0]  al_beat;
  wire  [27:0]  a_beat = a_addr[31:4];

  assign b_rdy = (qb_n[0] < QD) && (qb_n[1] < QD) && (bt_n < OD);
  assign a_rdy = (qa_n[0] < QD) && (qa_n[1] < QD) && (ao_n < OD);
  assign sw_rdy = (qw_n[0] < QD) && (qw_n[1] < QD);
  wire sw_take = sw_req && sw_rdy;
  wire sw_ch = sw_addr[4];
  wire b_take = b_req && b_rdy;
  wire a_take = a_req && a_rdy;
  wire a_ch = a_addr[4];
  wire a_reuse = !a_we && al_v && al_beat == a_beat;

  // ------------------------------------------------------------------ response FIFOs
  logic [511:0] rb_q [2][RD];
  logic [RW:0]  rb_n [2], rb_res [2];          // stored; stored + in flight
  logic [RW-1:0] rb_h [2], rb_t [2];
  logic [511:0] ra_q [2][AD];
  logic [AW_:0] ra_n [2], ra_res [2];
  logic [AW_-1:0] ra_h [2], ra_t [2];

  // ------------------------------------------------------------------ per-channel issue
  logic [1:0] ar_b, ar_a, w_b, w_a, w_w;         // this cycle's AR / write source
  logic [1:0] aw_done, w_done;                   // current write head: halves already taken
  logic [1:0] wsrc [2];                          // current write's source: 0 qb, 1 qa, 2 qw
  logic [1:0] wcur;                              // a write is in progress
  qb_t hb [2];
  qa_t ha [2], hw [2], hs [2];
  always_comb begin
    for (int c = 0; c < 2; c++) begin
      hb[c] = qb[c][qb_h[c]];
      ha[c] = qa[c][qa_h[c]];
      hw[c] = qw[c][qw_h[c]];
      // reads: A first (rare), then B; each needs reserved response room
      ar_a[c] = (qa_n[c] != 0) && !ha[c].we && (ra_res[c] < AD);
      ar_b[c] = !ar_a[c] && (qb_n[c] != 0) && !hb[c].we && (rb_res[c] < RD);
      m_arvalid[c] = ar_a[c] || ar_b[c];
      m_araddr[c] = ar_a[c] ? ha[c].addr : hb[c].addr;
      m_arid[c] = ar_a[c];
      // writes: the head write of qw, qa or qb (kept until both AW and W are taken)
      if (wcur[c]) begin
        w_w[c] = (wsrc[c] == 2'd2);
        w_a[c] = (wsrc[c] == 2'd1);
        w_b[c] = (wsrc[c] == 2'd0);
      end else begin
        w_w[c] = (qw_n[c] != 0);
        w_a[c] = !w_w[c] && (qa_n[c] != 0) && ha[c].we;
        w_b[c] = !w_w[c] && !w_a[c] && (qb_n[c] != 0) && hb[c].we;
      end
      hs[c] = w_w[c] ? hw[c] : ha[c];
      m_awvalid[c] = (w_w[c] || w_a[c] || w_b[c]) && !aw_done[c];
      m_wvalid[c] = (w_w[c] || w_a[c] || w_b[c]) && !w_done[c];
      m_awaddr[c] = (w_w[c] || w_a[c]) ? hs[c].addr : hb[c].addr;
      m_awid[c] = w_w[c] || w_a[c];
      m_wdata[c] = '0;
      m_wstrb[c] = '0;
      if (w_w[c] || w_a[c]) begin
        m_wdata[c][32 * hs[c].idx +: 32] = hs[c].data;
        m_wstrb[c][4 * hs[c].idx +: 4] = hs[c].be;
      end else begin
        m_wdata[c] = hb[c].data;
        for (int k = 0; k < 16; k++) m_wstrb[c][4 * k +: 4] = {4{hb[c].wmask[k]}};
      end
      m_bready[c] = 1'b1;
      m_rready[c] = 1'b1;
    end
  end

  // ------------------------------------------------------------------ merge
  wire b_out = (bt_n != 0) && (rb_n[0] != 0) && (rb_n[1] != 0);
  ao_t aoh;
  assign aoh = ao_q[ao_h];
  logic [511:0] a_last;                          // the beat of the last fetched A read
  wire a_out = (ao_n != 0) && (aoh.reuse || ra_n[aoh.c] != 0);
  wire [511:0] a_src = aoh.reuse ? a_last : ra_q[aoh.c][ra_h[aoh.c]];
  assign b_rvalid = b_out;
  assign b_rtag = bt_q[bt_h];
  assign b_rdata = {rb_q[1][rb_h[1]], rb_q[0][rb_h[0]]};
  assign a_rvalid = a_out;
  assign a_rdata = a_src[32 * aoh.idx +: 32];

  // ------------------------------------------------------------------ writes outstanding
  logic [15:0] wr_n;
  assign wr_idle = (wr_n == 0);

  always_ff @(posedge clk) begin
    if (rst) begin
      for (int c = 0; c < 2; c++) begin
        qb_n[c] <= '0; qb_h[c] <= '0; qa_n[c] <= '0; qa_h[c] <= '0;
        qw_n[c] <= '0; qw_h[c] <= '0;
        rb_n[c] <= '0; rb_res[c] <= '0; rb_h[c] <= '0; rb_t[c] <= '0;
        ra_n[c] <= '0; ra_res[c] <= '0; ra_h[c] <= '0; ra_t[c] <= '0;
      end
      bt_n <= '0; bt_h <= '0; ao_n <= '0; ao_h <= '0;
      al_v <= 1'b0;
      wr_n <= '0;
      aw_done <= '0; w_done <= '0; wcur <= '0;
      wsrc[0] <= '0; wsrc[1] <= '0;
      err <= 1'b0;
    end else begin
      logic [15:0] wn;
      wn = wr_n;
      for (int c = 0; c < 2; c++) begin
        logic [QW:0] nb, na, nw;
        logic [RW:0] rbn, rbr;
        logic [AW_:0] ran, rar;
        logic popb, popa, popw;
        nb = qb_n[c]; na = qa_n[c]; nw = qw_n[c];
        rbn = rb_n[c]; rbr = rb_res[c]; ran = ra_n[c]; rar = ra_res[c];
        popb = 1'b0; popa = 1'b0; popw = 1'b0;
        // ---- accept
        if (b_take && (!b_we || b_wmask[16 * c +: 16] != 0)) begin
          qb_t e;
          e.we = b_we;
          e.addr = chan_addr(b_addr + 32'(16 * c), c[0]);
          e.data = b_wdata[512 * c +: 512];
          e.wmask = b_we ? b_wmask[16 * c +: 16] : '0;
          qb[c][QW'(qb_h[c] + nb)] <= e;
          nb = nb + 1;
          if (b_we) wn = wn + 1;
        end
        if (a_take && a_ch == c[0] && !a_reuse) begin
          qa_t e;
          e.we = a_we;
          e.addr = chan_addr(a_addr, c[0]);
          e.idx = a_addr[3:0];
          e.data = a_wdata;
          e.be = a_be;
          qa[c][QW'(qa_h[c] + na)] <= e;
          na = na + 1;
          if (a_we) wn = wn + 1;
        end
        if (sw_take && sw_ch == c[0]) begin
          qa_t e;
          e.we = 1'b1;
          e.addr = chan_addr(sw_addr, c[0]);
          e.idx = sw_addr[3:0];
          e.data = sw_wdata;
          e.be = sw_be;
          qw[c][QW'(qw_h[c] + nw)] <= e;
          nw = nw + 1;
          wn = wn + 1;
        end
        // ---- AR
        if (m_arvalid[c] && m_arready[c]) begin
          if (ar_a[c]) begin popa = 1'b1; rar = rar + 1; end
          else begin popb = 1'b1; rbr = rbr + 1; end
        end
        // ---- AW / W (a write leaves its queue once both are taken)
        if (w_w[c] || w_a[c] || w_b[c]) begin
          logic awd, wd;
          awd = aw_done[c] || (m_awvalid[c] && m_awready[c]);
          wd = w_done[c] || (m_wvalid[c] && m_wready[c]);
          if (awd && wd) begin
            if (w_w[c]) popw = 1'b1; else if (w_a[c]) popa = 1'b1; else popb = 1'b1;
            aw_done[c] <= 1'b0; w_done[c] <= 1'b0; wcur[c] <= 1'b0;
          end else begin
            aw_done[c] <= awd; w_done[c] <= wd; wcur[c] <= 1'b1;
            wsrc[c] <= w_w[c] ? 2'd2 : w_a[c] ? 2'd1 : 2'd0;
          end
        end
        if (popb) begin qb_h[c] <= qb_h[c] + 1; nb = nb - 1; end
        if (popa) begin qa_h[c] <= qa_h[c] + 1; na = na - 1; end
        if (popw) begin qw_h[c] <= qw_h[c] + 1; nw = nw - 1; end
        // ---- R
        if (m_rvalid[c]) begin
          if (m_rresp[c][1]) err <= 1'b1;
          if (m_rid[c]) begin
            ra_q[c][ra_t[c]] <= m_rdata[c];
            ra_t[c] <= ra_t[c] + 1;
            ran = ran + 1;
          end else begin
            rb_q[c][rb_t[c]] <= m_rdata[c];
            rb_t[c] <= rb_t[c] + 1;
            rbn = rbn + 1;
          end
        end
        // ---- B
        if (m_bvalid[c]) begin
          if (m_bresp[c][1]) err <= 1'b1;
          wn = wn - 1;
        end
        // ---- merge pops
        if (b_out) begin
          rb_h[c] <= rb_h[c] + 1;
          rbn = rbn - 1; rbr = rbr - 1;
        end
        if (a_out && !aoh.reuse && aoh.c == c[0]) begin
          ra_h[c] <= ra_h[c] + 1;
          ran = ran - 1; rar = rar - 1;
        end
        qb_n[c] <= nb; qa_n[c] <= na; qw_n[c] <= nw;
        rb_n[c] <= rbn; rb_res[c] <= rbr; ra_n[c] <= ran; ra_res[c] <= rar;
      end
      wr_n <= wn;
      // ---- order FIFOs
      begin
        logic [OW:0] btn, aon;
        btn = bt_n; aon = ao_n;
        if (b_take && !b_we) begin
          bt_q[OW'(bt_h + btn)] <= b_tag;
          btn = btn + 1;
        end
        if (b_out) begin bt_h <= bt_h + 1; btn = btn - 1; end
        if (a_take && !a_we) begin
          ao_q[OW'(ao_h + aon)] <= '{c: a_ch, idx: a_addr[3:0], reuse: a_reuse};
          aon = aon + 1;
        end
        if (a_out) begin
          ao_h <= ao_h + 1; aon = aon - 1;
          if (!aoh.reuse) a_last <= ra_q[aoh.c][ra_h[aoh.c]];
        end
        bt_n <= btn; ao_n <= aon;
      end
      // ---- A beat reuse: the beat of the last A read, forgotten on any write
      if ((b_take && b_we) || (a_take && a_we) || sw_take) al_v <= 1'b0;
      else if (a_take && !a_we) begin
        al_v <= 1'b1;
        al_beat <= a_beat;
      end
    end
  end
endmodule
