// The x2 link's co-sim model (docs/x2_link.md): one cable between two
// cards' link ends (end a, end b), both on one core clock. Each end is otpu_link's per-link
// interface: lt_port's AXI-Stream ends (TX s_*, RX m_* with m_thdr), up, rx_err, abort_in and
// abort. The two-card setup takes two instances (A.p0 - B.p0, A.p1 - B.p1).
// Per direction: the sender takes a header only when the receiver's RX FIFO has room for the
// whole message (msg_words(LEN): the header, LEN words and otpu_link's trailer, which the model
// counts but does not carry), as lt_port does; a word taken in cycle c is on the receiver's m_*
// from cycle c + L (L >= 1: L 0 is taken as 1); the receiver's pops come back to the sender as
// credits L cycles later. RATE_Q8: the cable's words a cycle x 256 (591 = 2.31 = 11.1 GB/s at
// 150 MHz; a word a cycle at most is taken, so it binds only below 256), the trailer one more.
// `up` rises UPD cycles after reset (otpu_link: the channel up and an agreeing hello).
// Plusargs (ID = the instance's id; the end whose messages: a or b):
//   +lnk<ID>_L=<cycles>          L for both directions
//   +lnk<ID>a_flip=<n>           a bit of end a's n-th message (from 0) flips on the cable: with a
//                                payload, its first payload word's bit 0, and rx_err at b as the
//                                last payload word is taken; LEN 0, the header is dropped, rx_err
//   +lnk<ID>a_drop=<n>           end a's n-th message vanishes (no rx_err; its credits come back)
//   +lnk<ID>_stall=<t>:<c>       neither end sends for c cycles from cycle t (since reset)
//   +lnk<ID>_down=<t>            the link drops at cycle t: up low, rx_err at both ends, no traffic
//   +lnk<ID>a_abort=<t>          end b sees abort_in at cycle t + L, as if end a's hub aborted
// (b_ the same for end b's messages.) Checks ($fatal): a header's TYPE DATA, LEN <= LMAX, HCRC
// bits [255:224] zero; tlast with the header's LEN. The a_stat / a_hello outputs are what the end's
// registers would read (section 5: status, the far end's hello from CB / PB / NC).
`default_nettype none

module otpu_link_sim import otpu_link_pkg::*; #(
  parameter int L       = 40,
  parameter int RXW     = otpu_link_pkg::RXW,
  parameter int RATE_Q8 = 591,
  parameter int ID      = 0,
  parameter int UPD     = 16,
  parameter int CA = 0, PA = 0,           // end a's card and port
  parameter int CB = 1, PB = 0,           // end b's
  parameter int NC = 2
) (
  input  logic          clk,
  input  logic          rst,
  input  logic [WW-1:0] a_s_tdata,
  input  logic          a_s_tvalid,
  output logic          a_s_tready,
  input  logic          a_s_tlast,
  output logic [WW-1:0] a_m_tdata,
  output logic          a_m_tvalid,
  input  logic          a_m_tready,
  output logic          a_m_tlast,
  output logic          a_m_thdr,
  output logic          a_up,
  output logic          a_rx_err,
  output logic          a_abort_in,
  input  logic          a_abort,
  input  logic [WW-1:0] b_s_tdata,
  input  logic          b_s_tvalid,
  output logic          b_s_tready,
  input  logic          b_s_tlast,
  output logic [WW-1:0] b_m_tdata,
  output logic          b_m_tvalid,
  input  logic          b_m_tready,
  output logic          b_m_tlast,
  output logic          b_m_thdr,
  output logic          b_up,
  output logic          b_rx_err,
  output logic          b_abort_in,
  input  logic          b_abort,
  output logic [31:0]   a_stat,
  output logic [31:0]   a_hello,
  output logic [31:0]   b_stat,
  output logic [31:0]   b_hello
);
  int   cyc, lp, st_t, st_c, down_t;
  logic up, down, stall;
  initial begin
    string s;
    lp = L;
    void'($value$plusargs("link_lat=%d", lp));  // X2SimTransport: both cables
    void'($value$plusargs($sformatf("lnk%0d_L=%%d", ID), lp));
    if (lp < 1) lp = 1;
    if (lp > 1000) $fatal(1, "otpu_link_sim %0d: L %0d (at most 1000)", ID, lp);
    st_t = -1; st_c = 0; down_t = -1;
    if ($value$plusargs($sformatf("lnk%0d_stall=%%s", ID), s))
      if ($sscanf(s, "%d:%d", st_t, st_c) != 2) $fatal(1, "otpu_link_sim %0d: +stall=t:c", ID);
    void'($value$plusargs($sformatf("lnk%0d_down=%%d", ID), down_t));
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      cyc <= 0;
      up <= 1'b0;
      down <= 1'b0;
    end else begin
      cyc <= cyc + 1;
      if (cyc == down_t) down <= 1'b1;
      up <= !down && cyc != down_t && cyc >= UPD;
    end
  end
  assign stall = st_t >= 0 && cyc >= st_t && cyc < st_t + st_c;
  wire drop_ev = !rst && cyc == down_t && down_t >= 0;
  assign a_up = up;
  assign b_up = up;

  logic        ab_err, ba_err, ab_ain, ba_ain;
  logic [15:0] ab_avail, ba_avail;
  logic [31:0] ab_nerr, ba_nerr, ab_nfr, ba_nfr;
  logic        ab_wd, ba_wd;
  otpu_link_sim_dir #(.L(L), .RXW(RXW), .RATE_Q8(RATE_Q8), .ID(ID), .E("a")) u_ab (
    .clk, .rst, .up, .stall, .lp,
    .s_tdata(a_s_tdata), .s_tvalid(a_s_tvalid), .s_tready(a_s_tready), .s_tlast(a_s_tlast),
    .m_tdata(b_m_tdata), .m_tvalid(b_m_tvalid), .m_tready(b_m_tready), .m_tlast(b_m_tlast),
    .m_thdr(b_m_thdr), .abort(a_abort), .abort_in(ab_ain), .rx_err(ab_err), .avail(ab_avail),
    .n_err(ab_nerr), .n_frames(ab_nfr), .wdog(ab_wd));
  otpu_link_sim_dir #(.L(L), .RXW(RXW), .RATE_Q8(RATE_Q8), .ID(ID), .E("b")) u_ba (
    .clk, .rst, .up, .stall, .lp,
    .s_tdata(b_s_tdata), .s_tvalid(b_s_tvalid), .s_tready(b_s_tready), .s_tlast(b_s_tlast),
    .m_tdata(a_m_tdata), .m_tvalid(a_m_tvalid), .m_tready(a_m_tready), .m_tlast(a_m_tlast),
    .m_thdr(a_m_thdr), .abort(b_abort), .abort_in(ba_ain), .rx_err(ba_err), .avail(ba_avail),
    .n_err(ba_nerr), .n_frames(ba_nfr), .wdog(ba_wd));
  assign a_rx_err = ba_err || drop_ev;
  assign b_rx_err = ab_err || drop_ev;
  assign a_abort_in = ba_ain;
  assign b_abort_in = ab_ain;

  // the ends' registers (section 5): status and the far end's hello
  logic a_crc, b_crc, a_abt, b_abt, dropped;
  always_ff @(posedge clk) begin
    if (rst) begin
      a_crc <= 1'b0; b_crc <= 1'b0; a_abt <= 1'b0; b_abt <= 1'b0; dropped <= 1'b0;
    end else begin
      if (a_rx_err) a_crc <= 1'b1;
      if (b_rx_err) b_crc <= 1'b1;
      if (a_abort_in) a_abt <= 1'b1;
      if (b_abort_in) b_abt <= 1'b1;
      if (drop_ev) dropped <= 1'b1;
    end
  end
  always_comb begin
    a_stat = '0;
    a_stat[P_CHUP] = up; a_stat[P_HELLO] = up; a_stat[P_CRC] = a_crc; a_stat[P_WDOG] = ab_wd;
    a_stat[P_ABORT] = a_abt; a_stat[P_LANES] = up; a_stat[P_DROP] = dropped; a_stat[P_UP] = up;
    a_stat[31:16] = ab_avail;
    b_stat = '0;
    b_stat[P_CHUP] = up; b_stat[P_HELLO] = up; b_stat[P_CRC] = b_crc; b_stat[P_WDOG] = ba_wd;
    b_stat[P_ABORT] = b_abt; b_stat[P_LANES] = up; b_stat[P_DROP] = dropped; b_stat[P_UP] = up;
    b_stat[31:16] = ba_avail;
    a_hello = up ? {HELLO_MAGIC, 4'd0, 8'(NC), 4'(PB), 8'(CB)} : '0;
    b_hello = up ? {HELLO_MAGIC, 4'd0, 8'(NC), 4'(PA), 8'(CA)} : '0;
  end
endmodule

// One direction of otpu_link_sim: end E's TX to the other end's RX.
module otpu_link_sim_dir import otpu_link_pkg::*; #(
  parameter int    L       = 40,
  parameter int    RXW     = 2048,
  parameter int    RATE_Q8 = 591,
  parameter int    ID      = 0,
  parameter string E       = "a"
) (
  input  logic          clk,
  input  logic          rst,
  input  logic          up,
  input  logic          stall,
  input  int            lp,               // L (the plusarg's)
  input  logic [WW-1:0] s_tdata,
  input  logic          s_tvalid,
  output logic          s_tready,
  input  logic          s_tlast,
  output logic [WW-1:0] m_tdata,
  output logic          m_tvalid,
  input  logic          m_tready,
  output logic          m_tlast,
  output logic          m_thdr,
  input  logic          abort,
  output logic          abort_in,
  output logic          rx_err,
  output logic [15:0]   avail,
  output logic [31:0]   n_err,
  output logic [31:0]   n_frames,
  output logic          wdog
);
  localparam int CRED_INIT = RXW - 8;      // as otpu_link's (a few words for the parser)
  localparam int R = 1024;                 // the cable's ring: a slot a cycle, L < R
  localparam int AW = $clog2(RXW);
  typedef struct packed {
    logic          v, hdr, last, err;
    logic [WW-1:0] d;
  } w_t;

  int flip_n, drop_n, abort_t;
  initial begin
    flip_n = -1; drop_n = -1; abort_t = -1;
    void'($value$plusargs($sformatf("lnk%0d%s_flip=%%d", ID, E), flip_n));
    void'($value$plusargs($sformatf("lnk%0d%s_drop=%%d", ID, E), drop_n));
    void'($value$plusargs($sformatf("lnk%0d%s_abort=%%d", ID, E), abort_t));
  end

  w_t          cable [R];                  // the word due at the receiver in cycle (index)
  logic [7:0]  crd   [R];                  // credits due back at the sender
  logic        ab    [R];                  // an abort due at the receiver
  logic        er    [R];                  // an rx_err due at the receiver (a dropped header)
  w_t          rxf   [RXW];                // the receiver's RX FIFO
  logic [AW:0] wp, rp;
  int          cyc, tok, msgn, wait_c;
  logic        first, m_flip, m_drop;
  logic [15:0] rem, len, sent, ret;

  wire [15:0] hlen = s_tdata[47:32];
  always_comb begin
    avail = 16'(CRED_INIT) + ret - sent;
    s_tready = up && !stall && tok >= 256 && (!first || {1'b0, avail} >= msg_words(hlen));
    m_tvalid = wp != rp;
    {m_tdata, m_thdr, m_tlast} = {rxf[rp[AW-1:0]].d, rxf[rp[AW-1:0]].hdr, rxf[rp[AW-1:0]].last};
  end

  always_ff @(posedge clk) begin
    rx_err <= 1'b0;
    abort_in <= 1'b0;
    if (rst) begin
      for (int i = 0; i < R; i++) begin
        cable[i] <= '0; crd[i] <= '0; ab[i] <= 1'b0; er[i] <= 1'b0;
      end
      wp <= '0; rp <= '0;
      cyc <= 0; tok <= 256; msgn <= 0; wait_c <= 0;
      first <= 1'b1; rem <= '0; len <= '0; sent <= '0; ret <= '0;
      m_flip <= 1'b0; m_drop <= 1'b0;
      n_err <= '0; n_frames <= '0; wdog <= 1'b0;
    end else begin
      automatic int now = cyc % R, due = (cyc + lp - 1) % R;
      automatic w_t dv = cable[now];
      automatic w_t nw = '0;
      automatic logic [7:0] fr = '0;
      automatic logic [AW:0] wpn = wp;
      automatic logic drop_hdr = 1'b0;
      automatic int tk = (tok + RATE_Q8 > 4 * 256) ? 4 * 256 : tok + RATE_Q8;
      cyc <= cyc + 1;
      ret <= ret + 16'(crd[now]);
      crd[now] <= '0;
      cable[now] <= '0;
      if (ab[now]) abort_in <= 1'b1;
      ab[now] <= 1'b0;
      if (er[now]) begin rx_err <= 1'b1; n_err <= n_err + 1; end
      er[now] <= 1'b0;
      if (abort || cyc == abort_t) begin
        if (lp == 1) abort_in <= 1'b1; else ab[due] <= 1'b1;
      end
      // the sender
      if (s_tvalid && first && !s_tready && up && !stall && tok >= 256) begin
        wait_c <= wait_c + 1;
        if (wait_c >= (1 << 24)) wdog <= 1'b1;
      end else wait_c <= 0;
      if (s_tvalid && s_tready) begin
        automatic logic lastw;
        automatic logic fl = m_flip, dr = m_drop;
        if (first) begin
          if (s_tdata[63:56] != T_DATA || hlen > 16'(LMAX) || s_tdata[255:224] != '0)
            $fatal(1, "otpu_link_sim %0d%s: header TYPE %02x LEN %0d HCRC bits %08x (DATA, LEN <= %0d, 0)",
                   ID, E, s_tdata[63:56], hlen, s_tdata[255:224], LMAX);
          fl = msgn == flip_n;
          dr = msgn == drop_n;
          m_flip <= fl; m_drop <= dr;
          msgn <= msgn + 1;
          sent <= sent + 16'(msg_words(hlen));
          len <= hlen; rem <= hlen;
          lastw = hlen == 0;
          first <= lastw;
          tk = tk - 256;
        end else begin
          lastw = rem == 16'd1;
          rem <= rem - 1'b1;
          first <= lastw;
          tk = tk - 256 - (lastw ? 256 : 0);      // the trailer
        end
        if (s_tlast != lastw)
          $fatal(1, "otpu_link_sim %0d%s: tlast %0d on a word whose LEN says %0d", ID, E, s_tlast, lastw);
        nw.v = 1'b1;
        nw.hdr = first;
        nw.last = lastw;
        nw.d = s_tdata;
        if (fl && first && hlen == 0) begin       // a flipped header alone: dropped, rx_err
          nw.v = 1'b0;
          drop_hdr = 1'b1;
        end
        if (fl && !first && len != 0 && rem == len) nw.d[0] = !nw.d[0];
        if (fl && lastw && !first) nw.err = 1'b1;
        if (dr || drop_hdr) begin                 // its credits come back as if it were taken
          automatic int d2 = (cyc + 2 * lp - 1) % R;
          crd[d2] <= crd[d2] + 8'(first ? 1 : (lastw ? 2 : 1));
          nw.v = 1'b0;
          if (drop_hdr) begin
            if (lp == 1) begin rx_err <= 1'b1; n_err <= n_err + 1; end else er[due] <= 1'b1;
          end
        end
      end
      // the cable: the word due now, then (L 1) the word just taken
      if (dv.v) begin
        if (wpn - rp == (AW+1)'(RXW)) $fatal(1, "otpu_link_sim %0d%s: RX FIFO overflow", ID, E);
        rxf[wpn[AW-1:0]] <= dv;
        wpn = wpn + 1'b1;
      end
      if (nw.v) begin
        if (lp == 1) begin
          rxf[wpn[AW-1:0]] <= nw;
          wpn = wpn + 1'b1;
        end else cable[due] <= nw;
      end
      wp <= wpn;
      tok <= tk;
      // the receiver
      if (m_tvalid && m_tready) begin
        automatic w_t h = rxf[rp[AW-1:0]];
        rp <= rp + 1'b1;
        fr = (h.last && !h.hdr) ? 8'd2 : 8'd1;    // the trailer's word with the last
        if (h.hdr) n_frames <= n_frames + 1;
        if (h.err) begin rx_err <= 1'b1; n_err <= n_err + 1; end
        if (lp == 1) ret <= ret + 16'(crd[now]) + 16'(fr);
        else crd[due] <= crd[due] + fr;
      end
    end
  end
endmodule

`default_nettype wire
