`timescale 1ps/1ps
// Directed x2 port checks alongside tb_x2_link's random traffic: CSR ownership, hello agreement
// and changes, bidirectional ABORT, ping in core cycles, reset of queued data / credits / counts,
// channel and lane loss, and precise header / payload / dropped-word faults. The channel's
// semantic fault plusargs select ARG 0. A subsequent ARG 1 frame must recover with exact data.
// Error pulses are recorded even while m_tready is withheld; a payload DCRC error must follow
// the last payload's delivery. The monitor retains every delivered word, not just frame counts.
`default_nettype none
module tb_x2_link_ctrl;
  import otpu_link_pkg::*;
  logic clk = 0, uclk = 0, rst = 1;
  always #3333 clk = !clk;
  always #1241 uclk = !uclk;
  logic [3:0][WW-1:0] sd, md;
  logic [3:0] sv, sr, sl, mv, mr, ml, mh, up, er, ai, ab;
  logic [1:0] cw;
  logic [1:0][7:0] ca, card, ncards;
  logic [1:0][31:0] wd, rd;
  otpu_link_cards dut (
    .clk, .uclk, .rst, .a_card(card[0]), .a_ncards(ncards[0]), .b_card(card[1]), .b_ncards(ncards[1]),
    .a_s_tdata(sd[1:0]), .a_s_tvalid(sv[1:0]), .a_s_tready(sr[1:0]), .a_s_tlast(sl[1:0]),
    .a_m_tdata(md[1:0]), .a_m_tvalid(mv[1:0]), .a_m_tready(mr[1:0]), .a_m_tlast(ml[1:0]),
    .a_m_thdr(mh[1:0]), .a_up(up[1:0]), .a_rx_err(er[1:0]), .a_abort_in(ai[1:0]), .a_abort(ab[1:0]),
    .a_c_we(cw[0]), .a_c_addr(ca[0]), .a_c_wd(wd[0]), .a_c_rd(rd[0]),
    .b_s_tdata(sd[3:2]), .b_s_tvalid(sv[3:2]), .b_s_tready(sr[3:2]), .b_s_tlast(sl[3:2]),
    .b_m_tdata(md[3:2]), .b_m_tvalid(mv[3:2]), .b_m_tready(mr[3:2]), .b_m_tlast(ml[3:2]),
    .b_m_thdr(mh[3:2]), .b_up(up[3:2]), .b_rx_err(er[3:2]), .b_abort_in(ai[3:2]), .b_abort(ab[3:2]),
    .b_c_we(cw[1]), .b_c_addr(ca[1]), .b_c_wd(wd[1]), .b_c_rd(rd[1]));

  typedef struct packed {logic [WW-1:0] d; logic h, l;} beat_t;
  beat_t received[4][$];
  int errors[4], aborts[4], last_at[4], err_at[4], cyc;
  always @(posedge clk) begin
    if (rst) begin
      cyc = 0;
      for (int e = 0; e < 4; e++) begin
        received[e].delete(); errors[e] = 0; aborts[e] = 0; last_at[e] = -1; err_at[e] = -1;
      end
    end else begin
      cyc++;
      for (int e = 0; e < 4; e++) begin
        if (mv[e] && mr[e]) begin
          received[e].push_back('{md[e], mh[e], ml[e]});
          if (ml[e]) last_at[e] = cyc;
        end
        if (er[e]) begin errors[e]++; err_at[e] = cyc; end
        if (ai[e]) aborts[e]++;
      end
    end
  end
  task automatic ck(input bit ok, input string why);
    if (!ok) $fatal(1, "TBXC cycle %0d: %s", cyc, why);
  endtask
  task automatic tick(input int n);
    repeat(n) @(negedge clk);
  endtask
  task automatic wait_up;
    for (int i = 0; i < 10000; i++) begin
      tick(1);
      if (&up) return;
    end
    ck(0, "hello timeout");
  endtask
  task automatic write_c(input int c, input logic[7:0] addr, input logic[31:0] value);
    @(negedge clk); ca[c] = addr; wd[c] = value; cw[c] = 1;
    @(negedge clk); cw[c] = 0; wd[c] = 0;
  endtask
  task automatic read_c(input int c, input logic[7:0] addr, output logic[31:0] value);
    ca[c] = addr; #1; value = rd[c];
  endtask
  function automatic logic[WW-1:0] payload(input int seq, input int j);
    for (int l = 0; l < 8; l++) payload[32*l+:32] = 32'(seq*1000+j*8+l+1);
  endfunction
  task automatic send_word(input int e, input logic[WW-1:0] w, input bit last);
    @(negedge clk); sd[e] = w; sl[e] = last; sv[e] = 1;
    for (int i = 0; i < 10000; i++) begin
      @(posedge clk);
      if (sr[e]) begin @(negedge clk); sv[e] = 0; return; end
    end
    ck(0, "send timed out");
  endtask
  task automatic send_msg(input int e, input int seq, input int len);
    send_word(e, mk_data(16'(seq), len == 0 ? A_BAR : A_PIECE, 16'(len), 8'(e), 8'(e/2), 1, 0, 0), len == 0);
    for (int j = 0; j < len; j++) send_word(e, payload(seq,j), j == len-1);
  endtask
  task automatic check_msg(input int e, input int start, input int seq, input int len, input bit flip);
    logic [WW-1:0] expected;
    ck(received[e].size() >= start+len+1, "short received frame");
    expected = mk_data(16'(seq), len == 0 ? A_BAR : A_PIECE, 16'(len), 8'(e^2), 8'((e^2)/2), 1, 0, 0);
    ck(received[e][start] == {expected, 1'b1, (len == 0)}, "header differs");
    for (int j = 0; j < len; j++) begin
      expected = payload(seq,j) ^ WW'(flip && j == 0);
      ck(received[e][start+j+1] == {expected, 1'b0, (j == len-1)}, "payload/last differs");
    end
  endtask
  task automatic pulse_abort(input logic[3:0] mask);
    @(negedge clk); ab = mask;
    @(negedge clk); ab = 0;
  endtask

  int mode, kind, src;
  initial begin
    mode = 0; kind = 0; src = 0;
    void'($value$plusargs("case=%d", mode));
    void'($value$plusargs("kind=%d", kind));
    void'($value$plusargs("src=%d", src));
    sd = 0; sv = 0; sl = 0; mr = '1; ab = 0; ca = 0; cw = 0; wd = 0;
    card[0] = 0; card[1] = mode == 1 ? 0 : 1;
    ncards[0] = 2; ncards[1] = mode == 1 ? 4 : 2;
    tick(20); rst = 0;
    if (mode == 1) begin
      logic [31:0] status;
      tick(2000);
      ck(up == 0, "incompatible card/ncards accepted");
      ncards[1] = 2;
      tick(2000);
      ck(up == 0, "same card accepted");
      card[1] = 1;
      wait_up();
      ncards[0] = 4;
      tick(2000);
      ck(up == 0, "card configuration change did not invalidate hello");
      ncards[0] = 2;
      wait_up();
      // More than 256 hellos must not clear up when the wire counter wraps.
      for (int i = 0; i < 270; i++) begin
        write_c(0, R_CTRL, 1 << X_HELLO);
        write_c(1, R_CTRL, 1 << X_HELLO);
        tick(100);
        ck(&up, "hello count wrapped or manual hello cleared up");
      end
      read_c(0, R_PORT, status);
      ck(status[P_HELLO] && status[P_UP], "hello status lost");
    end else begin
      wait_up(); tick(100);
      case(mode)
        0: begin
          logic [31:0] value, before_words;
          for (int c = 0; c < 2; c++) begin
            read_c(c, R_INFO, value); ck(value == {16'(RXW),8'(LMAX),PROTO}, "X2_INFO");
            for (int a = 0; a < 4; a++) begin
              read_c(c, a == 0 ? 8'h00 : a == 1 ? 8'h0C : a == 2 ? 8'h60 : 8'h64, value);
              ck(value == 0, "port claimed another block's CSR");
            end
          end
          for (int e = 0; e < 4; e++) send_msg(e, 0, e+1);
          tick(500);
          for (int e = 0; e < 4; e++) check_msg(e^2, 0, 0, e+1, 0);
          pulse_abort('1); tick(500);
          for (int e = 0; e < 4; e++) ck(aborts[e] == 1, "bidirectional abort");
          for (int c = 0; c < 2; c++) begin
            write_c(c, R_CTRL, 1 << X_PING); tick(500);
            for (int p = 0; p < 2; p++) begin
              read_c(c, R_PING + 8'(4*p), value);
              ck(value[31] && value[30:0] >= 40 && value[30:0] < 250, "ping is not a core-cycle RTT");
              read_c(c, R_PORT + 8'(16*p) + 8'h0C, value); ck(value == 1, "frames received counter");
              read_c(c, R_PORT + 8'(16*p), value); ck(value[P_ABORT], "abort sticky");
              read_c(c, R_CNT + 8'(16*p), value); ck(value > 0, "TX words counter");
              read_c(c, R_CNT + 8'(16*p) + 4, value); ck(value > 0, "RX words counter");
            end
            read_c(c, R_CNT, before_words);
            write_c(c, R_CTRL, 1 << X_CLR); tick(100);
            read_c(c, R_PORT, value); ck(!value[P_ABORT] && !value[P_CRC], "clear stickies");
            read_c(c, R_CNT, value); ck(value >= before_words, "clear erased counters");
          end
          // A write to X2_CTRL.LRST alone is ignored, including parked receive data.
          mr[2] = 0; send_msg(0, 1, 4); tick(200);
          write_c(0, R_CTRL, 1 << X_LRST); write_c(1, R_CTRL, 1 << X_LRST); tick(100);
          ck(&up, "link locally decoded LRST"); mr[2] = 1; tick(200);
          check_msg(2, 2, 1, 4, 0);
          // The real reset discards parked words on every end and resets the entire window.
          mr = 0;
          for (int e = 0; e < 4; e++) send_msg(e, 2, 4);
          tick(200); rst = 1; tick(20);
          for (int c = 0; c < 2; c++) begin
            for (int p = 0; p < 2; p++) begin
              read_c(c, R_PORT + 8'(16*p), value);
              ck(value[31:16] == 16'(RXW-8) && value[8:2] == 0, "reset credits/stickies/up");
              read_c(c, R_PORT + 8'(16*p) + 8, value); ck(value == 0, "reset CRC count");
              read_c(c, R_PORT + 8'(16*p) + 12, value); ck(value == 0, "reset frame count");
              read_c(c, R_PING + 8'(4*p), value); ck(value == 0, "reset ping");
              for (int a = 0; a < 4; a++) begin
                read_c(c, R_CNT + 8'(16*p+4*a), value); ck(value == 0, "reset counters");
              end
            end
          end
          rst = 0; mr = '1; wait_up(); tick(500);
          for (int e = 0; e < 4; e++) begin
            ck(received[e].size() == 0, "stale word after reset");
            ck(errors[e] == 0 && aborts[e] == 0, "stale event after reset");
          end
          for (int e = 0; e < 4; e++) send_msg(e, 3, 0);
          tick(500);
          for (int e = 0; e < 4; e++) check_msg(e^2, 0, 3, 0, 0);
        end
        2: begin
          int dst, n, old_len;
          logic[31:0] value;
          dst = src ^ 2; old_len = kind == 6 ? 0 : 4;
          mr[dst] = 0;
          send_msg(src, 0, old_len);
          tick(400);
          ck(received[dst].size() == 0, "hub popped while held");
          if (kind == 1 || kind == 2) begin
            ck(errors[dst] == 0, "DCRC reported before delivery");
            // Even prefetched output words stay charged until the hub takes them.
            if (dst == 2) ck(dut.u_b.g_l[0].popped_c == 0, "credit returned before hub pop");
          end
          mr[dst] = 1; tick(400);
          send_msg(src, 1, 3); tick(800);
          n = received[dst].size();
          ck(errors[dst] >= 1, "fault raised no rx_err");
          ck(n >= 4, "no recovered frame");
          check_msg(dst, n-4, 1, 3, 0);
          if (kind == 0 || kind == 3 || kind == 6 || kind == 7 || kind == 8)
            ck(n == 4, "invalid header delivered");
          if (kind == 1 || kind == 2) begin
            ck(n == 9 && errors[dst] == 1, "DCRC fault count/delivery");
            check_msg(dst, 0, 0, 4, kind == 1);
            ck(err_at[dst] > 0, "DCRC pulse missing");
          end
          read_c(dst/2, R_PORT + 8'(16*(dst%2)), value); ck(value[P_CRC], "CRC sticky missing");
          read_c(dst/2, R_PORT + 8'(16*(dst%2)) + 8, value);
          ck(value == 32'(errors[dst]), "CRC error counter differs from pulses");
          for (int e = 0; e < 4; e++) if (e != dst) ck(errors[e] == 0, "fault leaked to another end");
          write_c(dst/2, R_CTRL, 1 << X_CLR); tick(200);
          read_c(dst/2, R_PORT + 8'(16*(dst%2)), value); ck(!value[P_CRC], "CRC clear");
          read_c(dst/2, R_PORT + 8'(16*(dst%2)) + 8, value); ck(value > 0, "clear reset CRC count");
        end
        3,4: begin
          logic [31:0] value;
          tick(3000);
          ck(!up[0] && !up[2] && up[1] && up[3], "channel/lane loss up");
          ck(errors[0] == 1 && errors[2] == 1, "channel/lane loss rx_err");
          for (int c = 0; c < 2; c++) begin
            read_c(c, R_PORT, value);
            ck(value[P_DROP] && value[P_CRC] && !value[P_LANES], "drop stickies/lane status");
            ck(value[P_CHUP] == (mode == 4), "channel status on lane-only loss");
          end
        end
        5: begin
          logic [31:0] value;
          tick(3000);
          for (int c = 0; c < 2; c++) begin
            read_c(c, R_CNT + 12, value); ck(value == 17, "soft error count");
          end
        end
        6: begin
          logic [WW-1:0] w;
          // Independent, byte-at-a-time MSB-first CRC-32 reference vectors (no final XOR).
          ck(hcrc('0) == 32'h6811F1FE && hcrc('1) == 32'h473B38FE, "HCRC zero/ones vector");
          ck(crc32w(CRC_INIT,'0) == 32'h4A55AF67 &&
             crc32w(CRC_INIT,'1) == 32'h2F2AC900, "DCRC zero/ones vector");
          w = '0;
          for (int j = 0; j < 28; j++) w[223-8*j -: 8] = 8'(j);
          ck(hcrc(w) == 32'h02617444 && (hcrc_hi(w)^hcrc_lo(w)) == hcrc(w), "HCRC byte vector/halves");
          for (int j = 0; j < 32; j++) w[255-8*j -: 8] = 8'(j);
          ck(crc32w(CRC_INIT,w) == 32'h8F819950, "DCRC byte vector");
        end
        default: ck(0, "unknown test case");
      endcase
    end
    $display("TBXC PASS case=%0d kind=%0d src=%0d cycles=%0d",mode,kind,src,cyc);
    $finish;
  end
  initial begin #2000000000; $fatal(1,"TBXC wall timeout"); end
endmodule
`default_nettype wire
