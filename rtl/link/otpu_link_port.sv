// One x2 link's core side (docs/x2_link.md), from the link test's
// lt_port (s6): the hub adapter's AXI-Stream ends, the credits, and the port's CRCs.
//   TX (s_*): a message is a header word (the hub's DATA: TYPE, LEN, ARG, AUX and its fields in
//   [223:64]) and LEN payload words, tlast on the last. The port takes a header only when the
//   peer's RX FIFO has room for the whole message (avail >= msg_words(LEN): the header, the
//   payload, the trailer), so a message never waits for credits once begun. A word taken goes to
//   the TX FIFO a cycle later from a register (w1): a header with FLAGS (F_CRED), CHK and HCRC
//   filled in, a payload word as it is, and after the last payload word the trailer with the
//   payload's CRC-32 (s_tready low that cycle). len_err: tlast disagreed with LEN, LEN > LMAX or
//   TYPE not DATA (counted; the message goes as given).
//   RX (m_*): the RX FIFO's DATA messages ({hdr, last, cred, trl, word}, otpu_link_lane's) through
//   a two-word output buffer. A header's HCRC is checked as it leaves the FIFO: a bad one is not
//   delivered, nor its message's other words (hcrc_err). The payload's CRC-32 is kept as its words
//   leave; the trailer is checked (dcrc_err on a mismatch) and dropped. frame: a header delivered.
// Credits are words of the peer's RX FIFO: avail = CRED_INIT + (the peer's returned total, cr_rx)
// - (the words sent, sent); popped counts credit-bearing words actually taken by the hub,
// plus stripped trailers and discarded words, which otpu_link_lane returns to the peer. Moving
// a word from the RX FIFO into an output register does not yet return its credit. A two-event
// credit return is serialized so the Gray counter crossing still steps by at most one/cycle.
`default_nettype none

module otpu_link_port import otpu_link_pkg::*; #(
  parameter int CRED_INIT = RXW - 8,
  parameter int TAW = 9                  // the TX FIFO: 2^TAW words
) (
  input  wire              clk,
  input  wire              rst,
  input  wire  [WW-1:0]    s_tdata,
  input  wire              s_tvalid,
  output logic             s_tready,
  input  wire              s_tlast,
  output logic [WW-1:0]    m_tdata,
  output logic             m_tvalid,
  input  wire              m_tready,
  output logic             m_tlast,
  output logic             m_thdr,
  // the TX FIFO's write side: {hdr, last, 0, 0, word}
  output logic             tf_we,
  output logic [WW+3:0]    tf_wdata,
  input  wire  [TAW:0]     tf_wfree,
  // the RX FIFO's read side (first word falls through): {hdr, last, cred, trl, word}
  input  wire  [WW+3:0]    rf_rdata,
  input  wire              rf_rvalid,
  output logic             rf_re,
  input  wire  [15:0]      cr_rx,
  output logic [15:0]      popped,
  output logic [15:0]      avail,
  output logic             len_err,     // pulses
  output logic             hcrc_err,
  output logic             dcrc_err,
  output logic             frame,
  output logic             cred_wait    // a header waits for credits this cycle
);
  // ---------------------------------------------------------------- TX
  logic        first;
  logic [15:0] rem, sent;
  wire  [15:0] len = s_tdata[47:32];
  logic        fits;
  // w1: the word taken last cycle; trl: the trailer is due this cycle
  logic          w1_v, w1_hdr, w1_last, w1_pl, trl;
  logic [WW-1:0] w1_d;
  logic [31:0]   dcrc;
  always_comb begin
    avail = 16'(CRED_INIT) + cr_rx - sent;
    fits = {1'b0, avail} >= msg_words(len);
    // room for w1, the trailer and this word; no word the cycle w1 is the last payload word
    s_tready = !rst && tf_wfree >= (TAW + 1)'(3) && !(w1_v && w1_pl && w1_last) && (!first || fits);
    cred_wait = s_tvalid && first && !fits;
  end
  logic last;
  always_comb last = first ? (len == 16'd0) : (rem == 16'd1);
  always_ff @(posedge clk) begin
    len_err <= 1'b0;
    if (rst) begin
      first <= 1'b1;
      sent <= '0;
      rem <= '0;
      w1_v <= 1'b0;
      trl <= 1'b0;
    end else begin
      w1_v <= s_tvalid && s_tready;
      trl <= w1_v && w1_pl && w1_last;
      if (s_tvalid && s_tready) begin
        w1_d <= s_tdata;
        w1_hdr <= first;
        w1_pl <= !first;
        w1_last <= last;
        len_err <= s_tlast != last ||
                   (first && (len > 16'(LMAX) || s_tdata[63:56] != T_DATA));
        if (first) begin
          sent <= sent + 16'(msg_words(len));
          rem <= len;
          first <= len == 16'd0;
        end else begin
          rem <= rem - 1'b1;
          first <= rem == 16'd1;
        end
      end
    end
  end
  // the FIFO write: w1 (sealed if a header), or the trailer
  always_ff @(posedge clk) begin
    if (w1_v && w1_hdr) dcrc <= CRC_INIT;
    else if (w1_v) dcrc <= crc32w(dcrc, w1_d);
  end
  always_comb begin
    tf_we = !rst && (w1_v || trl);
    if (trl) tf_wdata = {1'b0, 1'b1, 2'b00, mk_trl(dcrc)};
    else if (w1_hdr) tf_wdata = {1'b1, w1_last, 2'b00, seal(w1_d, 8'(1 << F_CRED))};
    else tf_wdata = {1'b0, 1'b0, 2'b00, w1_d};
  end

  // ---------------------------------------------------------------- RX
  wire           f_hdr = rf_rdata[WW+3], f_last = rf_rdata[WW+2], f_cred = rf_rdata[WW+1],
                 f_trl = rf_rdata[WW];
  wire [WW-1:0]  f_d = rf_rdata[WW-1:0];
  logic          drop;                   // inside a message whose header failed
  logic [31:0]   rcrc;
  // the output buffer: q0 (m_*), q1
  logic          q0_v, q1_v, q0_l, q1_l, q0_h, q1_h, q0_c, q1_c;
  logic [2:0]    credit_pending;
  logic [3:0]    credits_due;
  logic [WW-1:0] q0_d, q1_d;
  logic          ld, bypass, hok;
  logic [WW-1:0] hub_d;
  always_comb begin
    hok = hdr_ok(f_d);
    hub_d = f_d;
    if (f_hdr) begin hub_d[255:224] = '0; hub_d[55:48] = '0; hub_d[7:0] = '0; end
    // a word that goes to the hub needs a slot (taken only with at most one word held, so no
    // path from m_tready to the FIFO); dropped words and trailers go at once
    bypass = f_trl || (drop && !f_hdr) || (f_hdr && !hok);
    ld = !rst && rf_rvalid && !bypass && !(q0_v && q1_v);
    // Wait for the buffered payload to be delivered before checking its trailer. Thus
    // dcrc_err always follows delivery, including when the hub withholds m_tready.
    rf_re = !rst && rf_rvalid && (bypass || !(q0_v && q1_v)) &&
            !(f_trl && (q0_v || q1_v));
    m_tdata = q0_d;
    m_tvalid = q0_v;
    m_tlast = q0_l;
    m_thdr = q0_h;
    credits_due = {1'b0, credit_pending} + 4'(q0_v && m_tready && q0_c) +
                  4'(rf_re && f_cred && bypass);
  end
  always_ff @(posedge clk) begin
    hcrc_err <= 1'b0;
    dcrc_err <= 1'b0;
    frame <= 1'b0;
    if (rst) begin
      drop <= 1'b0;
      q0_v <= 1'b0;
      q1_v <= 1'b0;
      popped <= '0;
      credit_pending <= '0;
    end else begin
      if (credits_due != 0) popped <= popped + 1'b1;
      credit_pending <= 3'(credits_due - 4'(credits_due != 0));
      if (rf_re) begin
        if (f_hdr) begin
          rcrc <= CRC_INIT;
          if (!hok) begin
            hcrc_err <= 1'b1;
            drop <= !f_last;             // its payload and trailer follow
          end else begin drop <= 1'b0; frame <= 1'b1; end
        end else if (f_trl) begin
          if (!drop && (f_d[31:0] != rcrc || f_d[63:56] != T_TRL)) dcrc_err <= 1'b1;
          drop <= 1'b0;
        end else if (!drop) rcrc <= crc32w(rcrc, f_d);
      end
      // the output buffer
      case ({ld, q0_v && m_tready})
        2'b01: begin q0_v <= q1_v; q0_d <= q1_d; q0_l <= q1_l; q0_h <= q1_h; q0_c <= q1_c; q1_v <= 1'b0; end
        2'b10: if (!q0_v) begin q0_v <= 1'b1; q0_d <= hub_d; q0_l <= f_last; q0_h <= f_hdr; q0_c <= f_cred; end
               else begin q1_v <= 1'b1; q1_d <= hub_d; q1_l <= f_last; q1_h <= f_hdr; q1_c <= f_cred; end
        2'b11: if (q1_v) begin
                 q0_d <= q1_d; q0_l <= q1_l; q0_h <= q1_h; q0_c <= q1_c;
                 q1_d <= hub_d; q1_l <= f_last; q1_h <= f_hdr; q1_c <= f_cred;
               end else begin
                 q0_d <= hub_d; q0_l <= f_last; q0_h <= f_hdr; q0_c <= f_cred;
               end
        default: ;
      endcase
    end
  end
endmodule

`default_nettype wire
