`timescale 1ps/1ps
// Unit test of otpu_link_sim (sim/verilator/otpu_link_sim.sv; tests/test_x2_link.py): two
// hub-side agents, one at each end of a cable, send messages (LEN 0 .. LMAX, the hub's DATA
// header with ARG = the message number, payload words from pat()) and take what arrives, the
// receivers' m_tready withheld at random (+stall percent) or for one long window (+hold=cycles
// from cycle 2000: the sender must stop on credits, nothing lost). Checked: every message arrives
// whole and in order, as sent (a +lnk0a_flip message with its first payload word's bit 0
// flipped and one rx_err; a dropped one missing); the sender's words in flight never exceed the
// credits; +lat: idle messages arrive L cycles after they are taken; +abort: b's abort_in L
// cycles after a's abort; +lnk0_down: rx_err at both ends and up low.
// Plusargs: +seed=, +msgs= (a direction), +stall=, +gap= (a sender's idle percent), +len=
// (fixed LEN), +hold=, +lat, +abort=<cycle>. Prints TBLS DONE with the counts and PASS, or FAIL.
`default_nettype none

module tb_x2_link_sim;
  import otpu_link_pkg::*;
  parameter int L = 40;
  parameter int PA = 0, PB = 0;
  parameter bit PAIR = 0;  // use the exact same ports with the real RTL cable pair
  parameter int RATE_Q8 = 591;
  localparam int CRED_INIT = RXW - 8;

  logic clk = 1'b0;
  always #(PAIR ? 3333 : 5) clk = !clk;
  logic rst = 1'b1;

  logic [1:0][WW-1:0] s_tdata, m_tdata;
  logic [1:0]         s_tvalid, s_tready, s_tlast, m_tvalid, m_tready, m_tlast, m_thdr;
  logic [1:0]         up, rx_err, abort_in, abort;
  logic [1:0][31:0]   stat, hello;

  if (PAIR) begin : g_pair
  otpu_link_pair #(.ID(0), .PA(PA), .PB(PB)) dut (
    .clk, .rst,
    .a_s_tdata(s_tdata[0]), .a_s_tvalid(s_tvalid[0]), .a_s_tready(s_tready[0]), .a_s_tlast(s_tlast[0]),
    .a_m_tdata(m_tdata[0]), .a_m_tvalid(m_tvalid[0]), .a_m_tready(m_tready[0]), .a_m_tlast(m_tlast[0]),
    .a_m_thdr(m_thdr[0]), .a_up(up[0]), .a_rx_err(rx_err[0]), .a_abort_in(abort_in[0]), .a_abort(abort[0]),
    .b_s_tdata(s_tdata[1]), .b_s_tvalid(s_tvalid[1]), .b_s_tready(s_tready[1]), .b_s_tlast(s_tlast[1]),
    .b_m_tdata(m_tdata[1]), .b_m_tvalid(m_tvalid[1]), .b_m_tready(m_tready[1]), .b_m_tlast(m_tlast[1]),
    .b_m_thdr(m_thdr[1]), .b_up(up[1]), .b_rx_err(rx_err[1]), .b_abort_in(abort_in[1]), .b_abort(abort[1]),
    .a_stat(stat[0]), .a_hello(hello[0]), .b_stat(stat[1]), .b_hello(hello[1]));
  end else begin : g_sim
  otpu_link_sim #(.L(L), .RATE_Q8(RATE_Q8), .ID(0), .PA(PA), .PB(PB)) dut (
    .clk, .rst,
    .a_s_tdata(s_tdata[0]), .a_s_tvalid(s_tvalid[0]), .a_s_tready(s_tready[0]), .a_s_tlast(s_tlast[0]),
    .a_m_tdata(m_tdata[0]), .a_m_tvalid(m_tvalid[0]), .a_m_tready(m_tready[0]), .a_m_tlast(m_tlast[0]),
    .a_m_thdr(m_thdr[0]), .a_up(up[0]), .a_rx_err(rx_err[0]), .a_abort_in(abort_in[0]), .a_abort(abort[0]),
    .b_s_tdata(s_tdata[1]), .b_s_tvalid(s_tvalid[1]), .b_s_tready(s_tready[1]), .b_s_tlast(s_tlast[1]),
    .b_m_tdata(m_tdata[1]), .b_m_tvalid(m_tvalid[1]), .b_m_tready(m_tready[1]), .b_m_tlast(m_tlast[1]),
    .b_m_thdr(m_thdr[1]), .b_up(up[1]), .b_rx_err(rx_err[1]), .b_abort_in(abort_in[1]), .b_abort(abort[1]),
    .a_stat(stat[0]), .a_hello(hello[0]), .b_stat(stat[1]), .b_hello(hello[1]));
  end

  function automatic logic [WW-1:0] pat(input int n, input int j);
    logic [WW-1:0] w;
    for (int l = 0; l < 8; l++) w[32*l +: 32] = 32'(n) * 32'h9E37_79B1 + 32'(j) * 32'h85EB_CA77 + 32'(l) * 32'hC2B2_AE3D;
    return w;
  endfunction
  function automatic logic [31:0] xs(input logic [31:0] x);
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return x;
  endfunction

  int seed = 1, msgs = 400, stall = 0, gap = 0, flen = -1, hold = -1, abort_t = -1, down_t = -1;
  int flip_n [2], drop_n [2];
  bit lat = 0;
  initial begin
    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("msgs=%d", msgs));
    void'($value$plusargs("stall=%d", stall));
    void'($value$plusargs("gap=%d", gap));
    void'($value$plusargs("len=%d", flen));
    void'($value$plusargs("hold=%d", hold));
    void'($value$plusargs("abort=%d", abort_t));
    void'($value$plusargs("lnk0_down=%d", down_t));
    lat = $test$plusargs("lat");
    flip_n = '{-1, -1}; drop_n = '{-1, -1};
    void'($value$plusargs("lnk0a_flip=%d", flip_n[0]));
    void'($value$plusargs("lnk0b_flip=%d", flip_n[1]));
    void'($value$plusargs("lnk0a_drop=%d", drop_n[0]));
    void'($value$plusargs("lnk0b_drop=%d", drop_n[1]));
  end

  int cyc, errs;
  int sent_m [2], sent_w [2], got_m [2], got_w [2], nerr [2], ab_at [2], inflight_max [2];
  int t_sent [2][$];                      // the cycle each idle message's header was taken
  int lat_min, lat_max;
  // sender state
  int sn [2], srem [2], slen [2], sidle [2];
  logic [31:0] rs [2], rr [2];
  // receiver state
  int rn [2], rj [2], rlen [2], rexp [2];
  bit in_msg [2];

  task automatic fail(input string m);
    if (errs < 10) $display("TBLS ERROR cyc %0d: %s", cyc, m);
    errs++;
  endtask

  always_comb begin
    for (int e = 0; e < 2; e++) begin
      s_tvalid[e] = 1'b0; s_tdata[e] = '0; s_tlast[e] = 1'b0;
      if (!rst && sn[e] < msgs && sidle[e] == 0) begin
        s_tvalid[e] = 1'b1;
        if (srem[e] < 0) begin
          s_tdata[e] = mk_data(16'(sn[e]), A_PIECE, 16'(slen[e]), 8'(sn[e]), 8'(e), 4'd1, 16'(sn[e]), 16'(~sn[e]));
          s_tlast[e] = slen[e] == 0;
        end else begin
          s_tdata[e] = pat(sn[e], slen[e] - srem[e]);
          s_tlast[e] = srem[e] == 1;
        end
      end
      m_tready[e] = !rst && !(hold >= 0 && cyc >= 2000 && cyc < 2000 + hold) && (rr[e] % 100 >= 32'(stall));
    end
  end
  assign abort[0] = cyc == abort_t;
  assign abort[1] = 1'b0;

  always_ff @(posedge clk) begin
    if (rst) begin
      cyc <= 0;
      for (int e = 0; e < 2; e++) begin
        sn[e] <= 0; srem[e] <= -1; sidle[e] <= 0;
        rs[e] <= 32'(seed) * 32'h2545_F491 + 32'(e) + 1;
        rr[e] <= 32'(seed) * 32'h9E37_79B9 + 32'(e) + 7;
        slen[e] <= flen >= 0 ? flen : int'((32'(seed) * 32'h2545_F491 + 32'(e) + 1) % (LMAX + 1));
        rn[e] <= 0; rj[e] <= 0; in_msg[e] <= 1'b0;
        sent_m[e] <= 0; sent_w[e] <= 0; got_m[e] <= 0; got_w[e] <= 0; nerr[e] <= 0; ab_at[e] <= -1;
        inflight_max[e] <= 0;
      end
      lat_min <= 1 << 30; lat_max <= 0;
    end else begin
      cyc <= cyc + 1;
      for (int e = 0; e < 2; e++) begin
        automatic int o = 1 - e;          // the far end: e's messages arrive at o
        rr[e] <= xs(rr[e]);
        // sender e
        if (sidle[e] > 0) sidle[e] <= sidle[e] - 1;
        if (s_tvalid[e] && s_tready[e]) begin
          sent_w[e] <= sent_w[e] + 1;
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
            if (lat) sidle[e] <= 2 * L + 20;
            else if (r[31:24] % 100 < 8'(gap)) sidle[e] <= int'(r[23:20]);
          end
        end
        // receiver o takes e's messages
        if (m_tvalid[o] && m_tready[o]) begin
          got_w[e] <= got_w[e] + 1;
          if (m_thdr[o]) begin
            automatic int n = int'(m_tdata[o][31:16]);
            automatic int expn = rn[e];
            while (expn == drop_n[e] || (expn == flip_n[e] && rlen_of(expn) == 0)) expn++;
            if (in_msg[e]) fail($sformatf("end %0d: a header inside a message", o));
            if (n != (expn & 16'hFFFF))
              fail($sformatf("end %0d: message %0d, expected %0d", o, n, expn));
            if (m_tdata[o][63:56] != T_DATA || m_tdata[o][H_SRC +: 8] != 8'(e) ||
                m_tdata[o][H_ROW +: 16] != 16'(n) || m_tdata[o][H_COL +: 16] != 16'(~n))
              fail($sformatf("end %0d: message %0d's header %h", o, n, m_tdata[o][127:0]));
            rn[e] <= expn + 1;
            rlen[e] <= int'(m_tdata[o][47:32]);
            rj[e] <= 0;
            in_msg[e] <= m_tdata[o][47:32] != 0;
            if (m_tlast[o] != (m_tdata[o][47:32] == 0)) fail("tlast on a header");
            if (m_tdata[o][47:32] == 0) got_m[e] <= got_m[e] + 1;
            if (lat && t_sent[e].size() > 0) begin
              automatic int d = cyc - t_sent[e].pop_front();
              if (d < lat_min) lat_min <= d;
              if (d > lat_max) lat_max <= d;
            end
          end else begin
            automatic logic [WW-1:0] x = pat(rn[e] - 1, rj[e]);
            if (rn[e] - 1 == flip_n[e] && rj[e] == 0) x[0] = !x[0];
            if (!in_msg[e]) fail($sformatf("end %0d: a payload word outside a message", o));
            if (m_tdata[o] != x) fail($sformatf("end %0d: message %0d word %0d", o, rn[e] - 1, rj[e]));
            if (m_tlast[o] != (rj[e] + 1 == rlen[e])) fail($sformatf("end %0d: tlast", o));
            rj[e] <= rj[e] + 1;
            if (rj[e] + 1 == rlen[e]) begin
              in_msg[e] <= 1'b0;
              got_m[e] <= got_m[e] + 1;
            end
          end
        end
        if (rx_err[o]) nerr[o] <= nerr[o] + 1;
        if (abort_in[o] && ab_at[o] < 0) ab_at[o] <= cyc;
        // words in flight against the credits (the sender's view: sent - returned)
        if (int'(16'(CRED_INIT) - (o == 1 ? stat[0][31:16] : stat[1][31:16])) > inflight_max[e])
          inflight_max[e] <= int'(16'(CRED_INIT) - (o == 1 ? stat[0][31:16] : stat[1][31:16]));
      end
    end
  end
  // a message's LEN as the sender drew it (for the flipped-header case: LEN 0 is dropped)
  int lens [2][$];
  always_ff @(posedge clk)
    if (!rst)
      for (int e = 0; e < 2; e++)
        if (s_tvalid[e] && s_tready[e] && srem[e] < 0) lens[e].push_back(slen[e]);
  function automatic int rlen_of(input int n);
    return (n < lens[0].size()) ? lens[0][n] : 1;   // flips are injected on end a's messages
  endfunction

  initial begin
    int tmo;
    repeat (5) @(posedge clk);
    rst <= 1'b0;
    tmo = 200 * msgs * (LMAX + 2) + 20000 + (hold > 0 ? hold : 0);
    while (cyc < tmo) begin
      @(posedge clk);
      if (down_t < 0 && sn[0] >= msgs && sn[1] >= msgs && got_m[0] + (drop_n[0] >= 0) + (flip_n[0] >= 0 && rlen_of(flip_n[0]) == 0) >= msgs
          && got_m[1] + (drop_n[1] >= 0) >= msgs && cyc > abort_t + L + 5) break;
      if (down_t >= 0 && cyc > down_t + 50) break;
    end
    repeat (5) @(posedge clk);
    if (down_t < 0 && (got_m[0] + (drop_n[0] >= 0) + (flip_n[0] >= 0 && rlen_of(flip_n[0]) == 0) < msgs || got_m[1] + (drop_n[1] >= 0) < msgs))
      fail($sformatf("timeout: got %0d / %0d of %0d", got_m[0], got_m[1], msgs));
    if (inflight_max[0] > CRED_INIT || inflight_max[1] > CRED_INIT) fail("credits exceeded");
    if (down_t < 0 && nerr[1] != (flip_n[0] >= 0 ? 1 : 0)) fail($sformatf("rx_err at b %0d", nerr[1]));
    if (down_t >= 0 && (nerr[0] < 1 || nerr[1] < 1 || up != 2'b00)) fail("down: rx_err / up");
    if (abort_t >= 0 && ab_at[1] != abort_t + (L < 1 ? 1 : L)) fail($sformatf("abort_in at %0d", ab_at[1]));
    if (lat && (lat_min != (L < 1 ? 1 : L) || lat_max != (L < 1 ? 1 : L)))
      fail($sformatf("latency %0d..%0d, L %0d", lat_min, lat_max, L));
    if (!(down_t >= 0) && (hello[0][7:0] != 8'd1 || hello[0][31:24] != HELLO_MAGIC || !stat[0][P_UP]))
      fail($sformatf("registers: hello %h stat %h", hello[0], stat[0]));
    $display("TBLS DONE L=%0d msgs=%0d words=%0d,%0d got=%0d,%0d rx_err=%0d,%0d inflight_max=%0d,%0d lat=%0d..%0d abort=%0d cycles=%0d",
             L, msgs, sent_w[0], sent_w[1], got_m[0], got_m[1], nerr[0], nerr[1],
             inflight_max[0], inflight_max[1], lat_min, lat_max, ab_at[1], cyc);
    if (errs == 0) $display("TBLS PASS"); else $display("TBLS FAIL %0d", errs);
    $finish;
  end
endmodule

`default_nettype wire
