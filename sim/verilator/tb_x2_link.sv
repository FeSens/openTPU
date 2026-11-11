`timescale 1ps / 1ps
// Unit test of the x2 link port (rtl/link/otpu_link.sv; tests/test_x2_link.py): two cards' ports
// joined by two cables (otpu_link_pair: the RTL with otpu_link_chan in place of Aurora; MODEL 1:
// two otpu_link_sim instead, the same traffic and checks). Ends e = 2 card + port (a.p0, a.p1,
// b.p0, b.p1); e talks to e ^ 2. Each end's hub-side agent sends messages (the hub's DATA: LEN
// 0 .. LMAX, ARG = the message number, SRC = e, ROW / COL from it, payload words from pat()) and
// takes what arrives, m_tready withheld at random (+stall percent) or for one window (+hold).
// Checked: every message whole and in order as sent (the header's TYPE, LEN, ARG, SRC, ROW, COL;
// the payload words; tlast), no rx_err; with +tolerate (a corrupted wire word, +chan<p>a_flip) at
// most one message from that end lost or altered and an rx_err at the receiver; up at every end
// after start-up, the hello registers naming the far card and port (RTL); +lat: an idle message's
// header on the far m_* so many core cycles after it was taken; +abort: b.p0's abort_in after
// a.p0's abort; +chan0_down: rx_err and up low at both ends of cable 0.
// Plusargs: +seed=, +msgs= (an end), +stall=, +gap=, +len=, +hold=, +lat, +abort=, +tolerate.
// Prints TBXL DONE with the counts and PASS, or FAIL.
`default_nettype none

module tb_x2_link;
  import otpu_link_pkg::*;
  parameter int MODEL = 0;
  parameter int L = 40;                    // MODEL 1's latency
  localparam int CP = 6667, UP = 2482;     // 150 MHz, 402.83 MHz

  logic clk = 1'b0, uclk = 1'b0;
  always #(CP / 2) clk = !clk;
  always #(UP / 2) uclk = !uclk;
  logic rst = 1'b1;

  logic [3:0][WW-1:0] s_tdata, m_tdata;
  logic [3:0]         s_tvalid, s_tready, s_tlast, m_tvalid, m_tready, m_tlast, m_thdr;
  logic [3:0]         up, rx_err, abort_in, abort;
  logic [1:0]         c_we;
  logic [1:0][7:0]    c_addr;
  logic [1:0][31:0]   c_wd, c_rd;
  assign c_we = '0;
  assign c_wd = '0;

  if (MODEL) begin : g_m
    for (genvar p = 0; p < 2; p++) begin : g_c
      otpu_link_sim #(.L(L), .ID(p), .PA(p), .PB(p)) u_c (
        .clk, .rst,
        .a_s_tdata(s_tdata[p]), .a_s_tvalid(s_tvalid[p]), .a_s_tready(s_tready[p]), .a_s_tlast(s_tlast[p]),
        .a_m_tdata(m_tdata[p]), .a_m_tvalid(m_tvalid[p]), .a_m_tready(m_tready[p]), .a_m_tlast(m_tlast[p]),
        .a_m_thdr(m_thdr[p]), .a_up(up[p]), .a_rx_err(rx_err[p]), .a_abort_in(abort_in[p]), .a_abort(abort[p]),
        .b_s_tdata(s_tdata[2+p]), .b_s_tvalid(s_tvalid[2+p]), .b_s_tready(s_tready[2+p]), .b_s_tlast(s_tlast[2+p]),
        .b_m_tdata(m_tdata[2+p]), .b_m_tvalid(m_tvalid[2+p]), .b_m_tready(m_tready[2+p]), .b_m_tlast(m_tlast[2+p]),
        .b_m_thdr(m_thdr[2+p]), .b_up(up[2+p]), .b_rx_err(rx_err[2+p]), .b_abort_in(abort_in[2+p]), .b_abort(abort[2+p]),
        .a_stat(), .a_hello(), .b_stat(), .b_hello());
    end
    assign c_rd = '0;
  end else begin : g_r
    otpu_link_cards u_p (
      .clk, .uclk, .rst,
      .a_card(8'd0), .a_ncards(8'd2), .b_card(8'd1), .b_ncards(8'd2),
      .a_s_tdata(s_tdata[1:0]), .a_s_tvalid(s_tvalid[1:0]), .a_s_tready(s_tready[1:0]), .a_s_tlast(s_tlast[1:0]),
      .a_m_tdata(m_tdata[1:0]), .a_m_tvalid(m_tvalid[1:0]), .a_m_tready(m_tready[1:0]), .a_m_tlast(m_tlast[1:0]),
      .a_m_thdr(m_thdr[1:0]), .a_up(up[1:0]), .a_rx_err(rx_err[1:0]), .a_abort_in(abort_in[1:0]), .a_abort(abort[1:0]),
      .a_c_we(c_we[0]), .a_c_addr(c_addr[0]), .a_c_wd(c_wd[0]), .a_c_rd(c_rd[0]),
      .b_s_tdata(s_tdata[3:2]), .b_s_tvalid(s_tvalid[3:2]), .b_s_tready(s_tready[3:2]), .b_s_tlast(s_tlast[3:2]),
      .b_m_tdata(m_tdata[3:2]), .b_m_tvalid(m_tvalid[3:2]), .b_m_tready(m_tready[3:2]), .b_m_tlast(m_tlast[3:2]),
      .b_m_thdr(m_thdr[3:2]), .b_up(up[3:2]), .b_rx_err(rx_err[3:2]), .b_abort_in(abort_in[3:2]), .b_abort(abort[3:2]),
      .b_c_we(c_we[1]), .b_c_addr(c_addr[1]), .b_c_wd(c_wd[1]), .b_c_rd(c_rd[1]));
  end

  logic [3:0][15:0] min_credit;
  if (!MODEL) begin : g_checks
    for (genvar p = 0; p < 2; p++) begin : g_p
      always @(posedge uclk) if (!rst) begin
        if (g_r.u_p.u_a.g_l[p].u_rxf.used_w > (12'(RXW)) ||
            g_r.u_p.u_b.g_l[p].u_rxf.used_w > (12'(RXW)))
          $fatal(1, "RX FIFO exceeded 2048 words");
        if (g_r.u_p.u_a.g_l[p].ovf_u || g_r.u_p.u_b.g_l[p].ovf_u)
          $fatal(1, "RX FIFO overflow");
      end
      always @(posedge clk) begin
        if (rst) begin min_credit[p] <= 16'(RXW-8); min_credit[p+2] <= 16'(RXW-8); end
        else begin
        if (g_r.u_p.u_a.g_l[p].avail < min_credit[p]) min_credit[p] <= g_r.u_p.u_a.g_l[p].avail;
        if (g_r.u_p.u_b.g_l[p].avail < min_credit[p+2]) min_credit[p+2] <= g_r.u_p.u_b.g_l[p].avail;
        if (g_r.u_p.u_a.g_l[p].avail > 16'(RXW - 8) ||
            g_r.u_p.u_b.g_l[p].avail > 16'(RXW - 8))
          $fatal(1, "credits exceeded initial window");
        end
      end
    end
  end

  function automatic logic [WW-1:0] pat(input int e, input int n, input int j);
    logic [WW-1:0] w;
    for (int l = 0; l < 8; l++)
      w[32*l +: 32] = 32'(n) * 32'h9E37_79B1 + 32'(j) * 32'h85EB_CA77 + 32'(l) * 32'hC2B2_AE3D + 32'(e) * 32'h27D4_EB2F;
    return w;
  endfunction
  function automatic logic [31:0] xs(input logic [31:0] x);
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return x;
  endfunction

  int seed = 1, msgs = 300, stall = 0, gap = 0, flen = -1, hold = -1, abort_t = -1, down_t = -1;
  bit lat = 0, tol = 0;
  initial begin
    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("msgs=%d", msgs));
    void'($value$plusargs("stall=%d", stall));
    void'($value$plusargs("gap=%d", gap));
    void'($value$plusargs("len=%d", flen));
    void'($value$plusargs("hold=%d", hold));
    void'($value$plusargs("abort=%d", abort_t));
    void'($value$plusargs("chan0_down=%d", down_t));
    void'($value$plusargs("lnk0_down=%d", down_t));
    lat = $test$plusargs("lat");
    tol = $test$plusargs("tolerate");
  end

  int cyc, errs, t_up;
  bit go;                                  // every end up (the agents start then)
  int sent_m [4], got_m [4], lost [4], bad [4], nerr [4], ab_at [4];
  int sn [4], srem [4], slen [4], sidle [4];
  logic [31:0] rs [4], rr [4];
  int rn [4], rj [4], rlen [4];
  bit in_msg [4], cur_bad [4];
  int t_sent [4][$];
  logic [31:0] digest [4];
  int lat_min = 1 << 30, lat_max = 0;

  task automatic fail(input string m);
    if (errs < 10) $display("TBXL ERROR cyc %0d: %s", cyc, m);
    errs++;
  endtask

  always_comb begin
    for (int e = 0; e < 4; e++) begin
      s_tvalid[e] = 1'b0; s_tdata[e] = '0; s_tlast[e] = 1'b0;
      if (go && sn[e] < msgs && sidle[e] == 0) begin
        s_tvalid[e] = 1'b1;
        if (srem[e] < 0) begin
          s_tdata[e] = mk_data(16'(sn[e]), slen[e] == 0 ? A_BAR : A_PIECE, 16'(slen[e]), 8'(sn[e]), 8'(e), 4'd1, 16'(sn[e]), 16'(~sn[e]));
          s_tlast[e] = slen[e] == 0;
        end else begin
          s_tdata[e] = pat(e, sn[e], slen[e] - srem[e]);
          s_tlast[e] = srem[e] == 1;
        end
      end
      m_tready[e] = !rst && !(hold >= 0 && cyc >= t_up + 2000 && cyc < t_up + 2000 + hold) &&
                    (rr[e] % 100 >= 32'(stall));
      abort[e] = e == 0 && cyc == abort_t;
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      cyc <= 0; go <= 1'b0; t_up <= 0;
      for (int e = 0; e < 4; e++) begin
        sn[e] <= 0; srem[e] <= -1; sidle[e] <= 0;
        rs[e] <= 32'(seed) * 32'h2545_F491 + 32'(e) + 1;
        rr[e] <= 32'(seed) * 32'h9E37_79B9 + 32'(e) + 7;
        slen[e] <= flen >= 0 ? flen : int'((32'(seed) * 32'h2545_F491 + 32'(e) + 1) % (LMAX + 1));
        rn[e] <= 0; rj[e] <= 0; in_msg[e] <= 1'b0; cur_bad[e] <= 1'b0;
        digest[e] <= CRC_INIT;
        sent_m[e] <= 0; got_m[e] <= 0; lost[e] <= 0; bad[e] <= 0; nerr[e] <= 0; ab_at[e] <= -1;
      end
    end else begin
      cyc <= cyc + 1;
      if (!go && &up) begin go <= 1'b1; t_up <= cyc; end
      for (int e = 0; e < 4; e++) begin
        automatic int o = e ^ 2;          // e's messages arrive at o
        rr[e] <= xs(rr[e]);
        if (sidle[e] > 0) sidle[e] <= sidle[e] - 1;
        if (s_tvalid[e] && s_tready[e]) begin
          if (srem[e] < 0) begin
            if (lat) t_sent[e].push_back(cyc);
            srem[e] <= slen[e];
          end else srem[e] <= srem[e] - 1;
          if (s_tlast[e]) begin
            automatic logic [31:0] r = xs(rs[e]);
            rs[e] <= r;
            sent_m[e] <= sent_m[e] + 1;
            sn[e] <= sn[e] + 1;
            srem[e] <= -1;
            slen[e] <= flen >= 0 ? flen : int'(r % (LMAX + 1));
            if (lat) sidle[e] <= 200;
            else if (r[31:24] % 100 < 8'(gap)) sidle[e] <= int'(r[23:20]);
          end
        end
        if (m_tvalid[o] && m_tready[o]) begin
          digest[e] <= crc32w(digest[e], m_tdata[o]);
          if (m_thdr[o]) begin
            automatic int n = int'(m_tdata[o][31:16]);
            automatic int ln = int'(m_tdata[o][47:32]);
            if (in_msg[e]) begin
              fail($sformatf("end %0d: a header inside a message", o));
              in_msg[e] <= 1'b0;
            end
            if (n != (rn[e] & 16'hFFFF)) begin
              if (tol && n > (rn[e] & 16'hFFFF)) lost[e] <= lost[e] + n - (rn[e] & 16'hFFFF);
              else fail($sformatf("end %0d: message %0d, expected %0d", o, n, rn[e]));
            end
            if (m_tdata[o][63:56] != T_DATA || m_tdata[o][H_SRC +: 8] != 8'(e) ||
                m_tdata[o][H_ROW +: 16] != 16'(n) || m_tdata[o][H_COL +: 16] != 16'(~n) ||
                m_tdata[o][15:8] != (ln == 0 ? A_BAR : A_PIECE))
              fail($sformatf("end %0d: message %0d's header %h", o, n, m_tdata[o][127:0]));
            if (!tol && m_tdata[o] != mk_data(16'(n), ln == 0 ? A_BAR : A_PIECE,
                  16'(ln), 8'(n), 8'(e), 4'd1, 16'(n), 16'(~n)))
              fail($sformatf("end %0d: complete header differs", o));
            if (m_tlast[o] != (ln == 0)) fail($sformatf("end %0d: tlast on a header", o));
            rn[e] <= n + 1;
            rlen[e] <= ln;
            rj[e] <= 0;
            in_msg[e] <= ln != 0;
            cur_bad[e] <= 1'b0;
            if (ln == 0) got_m[e] <= got_m[e] + 1;
            if (lat && t_sent[e].size() > 0) begin
              automatic int d = cyc - t_sent[e].pop_front();
              if (d < lat_min) lat_min <= d;
              if (d > lat_max) lat_max <= d;
            end
          end else begin
            automatic logic badw = m_tdata[o] != pat(e, rn[e] - 1, rj[e]);
            if (!in_msg[e]) fail($sformatf("end %0d: a payload word outside a message", o));
            if (badw && !tol) fail($sformatf("end %0d: message %0d word %0d", o, rn[e] - 1, rj[e]));
            if (m_tlast[o] != (rj[e] + 1 == rlen[e])) fail($sformatf("end %0d: tlast", o));
            rj[e] <= rj[e] + 1;
            if (badw && !cur_bad[e]) begin bad[e] <= bad[e] + 1; cur_bad[e] <= 1'b1; end
            if (rj[e] + 1 == rlen[e]) begin
              in_msg[e] <= 1'b0;
              got_m[e] <= got_m[e] + 1;
            end
          end
        end
        if (rx_err[e]) nerr[e] <= nerr[e] + 1;
        if (abort_in[e] && ab_at[e] < 0) ab_at[e] <= cyc;
      end
    end
  end

  function automatic int done_cnt(input int e);   // e's messages accounted for at e ^ 2
    return got_m[e] + lost[e];
  endfunction

  initial begin
    int tmo;
    bit all;
    c_addr = '0;
    repeat (8) @(posedge clk);
    rst <= 1'b0;
    tmo = 100 * msgs * (LMAX + 3) + 40000 + (hold > 0 ? hold : 0);
    while (cyc < tmo) begin
      @(posedge clk);
      all = go;
      for (int e = 0; e < 4; e++) if (sn[e] < msgs || done_cnt(e) < msgs - (tol ? 1 : 0)) all = 0;
      if (all && cyc > abort_t + 200 && !(tol && cyc < t_up + 3000)) break;
      if (down_t >= 0 && cyc > 3000 + t_up && go) break;
    end
    repeat (300) @(posedge clk);         // anything late (tolerate: the last messages, rx_err)
    if (!go) fail("the ends never came up");
    if (down_t < 0)
      for (int e = 0; e < 4; e++)
        if (done_cnt(e) < msgs - (tol ? 1 : 0)) fail($sformatf("end %0d's messages: got %0d lost %0d of %0d", e, got_m[e], lost[e], msgs));
    for (int e = 0; e < 4; e++) begin
      automatic bit hit = tol && e == 2;   // a.p0's messages, corrupted on cable 0, arrive at b.p0
      if (down_t >= 0) begin
        if ((e == 0 || e == 2) && (nerr[e] < 1 || up[e])) fail($sformatf("down: end %0d rx_err %0d up %0d", e, nerr[e], up[e]));
      end else if (hit) begin
        if (nerr[e] < 1 || lost[0] + bad[0] > 1) fail($sformatf("flip: rx_err %0d lost %0d bad %0d", nerr[e], lost[0], bad[0]));
      end else if (nerr[e] != 0 || lost[e ^ 2] != 0 || bad[e ^ 2] != 0)
        fail($sformatf("end %0d: rx_err %0d, lost %0d bad %0d", e, nerr[e], lost[e ^ 2], bad[e ^ 2]));
    end
    if (abort_t >= 0 && ab_at[2] < 0) fail("abort_in never came");
    if (!MODEL && down_t < 0) begin       // the hello registers: the far card and port
      for (int c = 0; c < 2; c++)
        for (int p = 0; p < 2; p++) begin
          automatic logic [31:0] h, s;
          c_addr[c] = R_PORT + 8'(16 * p) + 8'h04;
          #1;
          h = c_rd[c];
          c_addr[c] = R_PORT + 8'(16 * p);
          #1;
          s = c_rd[c];
          if (h != {HELLO_MAGIC, 4'd0, 8'd2, 4'(p), 8'(1 - c)} || !s[P_UP] || !s[P_HELLO])
            fail($sformatf("card %0d port %0d: hello %h status %h", c, p, h, s));
        end
    end
    $display("TBXL DONE MODEL=%0d msgs=%0d got=%0d,%0d,%0d,%0d lost=%0d bad=%0d rx_err=%0d,%0d,%0d,%0d lat=%0d..%0d abort=%0d up_at=%0d cycles=%0d",
             MODEL, msgs, got_m[0], got_m[1], got_m[2], got_m[3], lost[0], bad[0],
             nerr[0], nerr[1], nerr[2], nerr[3], lat_min, lat_max,
             ab_at[2] >= 0 ? ab_at[2] - abort_t : -1, t_up, cyc);
    if (!MODEL && hold > 0) begin
      logic [31:0] count;
      for (int e = 0; e < 4; e++) begin
        c_addr[e/2] = R_CNT + 8'(16*(e%2)) + 8'h08; #1; count = c_rd[e/2];
        if (count == 0 || min_credit[e] >= 34) fail("RX hold never exhausted credits / credit-wait counter");
      end
    end
    $display("TBXL DIGEST %h %h %h %h", digest[0], digest[1], digest[2], digest[3]);
    if (errs == 0) $display("TBXL PASS"); else $display("TBXL FAIL %0d", errs);
    $finish;
  end
endmodule

`default_nettype wire
