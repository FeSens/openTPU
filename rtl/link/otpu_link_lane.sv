// One x2 link's user_clk side (docs/x2_link.md): the wire format in front of an
// Aurora 64B/66B x4 core in streaming mode (otpu_link_pkg), from the link test's lt_link (s6). All
// of it runs on the core's user_clk (402.8 MHz at 25.78125 Gb/s), every stage a register stage.
// TX: a message at a time, chosen at message boundaries: a control word (a PONG echo, ABORT, CRED,
// IDENT = hello, PING), else a message from the TX FIFO (otpu_link_port's: sealed header, payload,
// trailer; cut through, its words as the core side writes them), else a PRBS message (prbs_en). The
// last stage stamps PING / PONG in the cycle Aurora takes them and sits behind a two-word skid, so
// Aurora's tready reaches no further than it.
// RX: the parser follows the headers (CHK, a known type and, but for PING / PONG / PRBS, HCRC;
// else the word is dropped, hdr_err on the first of a run, so a misframed message's words go): a
// DATA message's header, payload and (LEN > 0) trailer go to the RX FIFO, flagged {hdr, last (the
// last payload word, or a LEN 0 header), cred, trl}; a PING is echoed from here as a PONG; a PONG
// gives the round trip (ping: [15:0] user_clk cycles, [30:16] the far end's turnaround, [31]
// valid); CRED, IDENT and ABORT go to registers / pulses; PRBS words to the checker.
// Credits: the core takes no message whose words the peer's RX FIFO has no room for. This side
// returns the credit-bearing words the core took (popped) as a CRED word when cred_th are
// due or the oldest is 1024 cycles old. Unrecognizable wire words never manufacture credits:
// a corrupt/dropped frame leaves conservative outstanding credits until the host resets both
// ends for token retry. A valid header in a missing trailer's position restarts the parser.
// Hello: at reset release, at channel up, on hello_req, every 2^21 cycles (5 ms), every 2^12 (10 us)
// until the peer's has come (since reset / hello_req), and in answer to one that has not heard ours.
// CRED, IDENT and ABORT go out sealed (HCRC) from registers kept up to date beside the chooser.
`default_nettype none

module otpu_link_lane import otpu_link_pkg::*; (
  input  wire              clk,
  input  wire              rst,
  // Aurora 64B/66B, streaming
  output logic [WW-1:0]    tx_tdata,
  output logic             tx_tvalid,
  input  wire              tx_tready,
  input  wire  [WW-1:0]    rx_tdata,
  input  wire              rx_tvalid,
  input  wire              channel_up,
  input  wire              soft_err,
  // the TX FIFO (first word falls through): {hdr, last (the message's last wire word), 0, 0, word}
  input  wire  [WW+3:0]    tf_data,
  input  wire              tf_valid,
  output logic             tf_re,
  // the RX FIFO: {hdr, last, cred, trl, word}
  output logic             rf_we,
  output logic [WW+3:0]    rf_data,
  input  wire              rf_full,
  // credits (mod 2^16)
  input  wire  [15:0]      popped,      // credit-bearing words the core side has taken (synchronized)
  output logic [15:0]      cr_rx,       // the peer's credit total, as last received
  // settings (static while traffic runs) and pulses, in clk
  input  wire              prbs_en,
  input  wire  [15:0]      prbs_len,
  input  wire  [15:0]      cred_th,
  input  wire  [7:0]       card,
  input  wire  [7:0]       ncards,
  input  wire              port,
  input  wire  [31:0]      cfg,
  input  wire              hello_req,
  input  wire              ping_req,
  input  wire              abort_req,
  input  wire              inject,      // flip bit 0 of the next PRBS word sent
  // results
  output logic             hdr_err,     // pulse: a run of dropped words began (no header where one
                                        // was due)
  output logic             ovf,         // pulse: the RX FIFO was full
  output logic             abort_rx,    // pulse: an ABORT came
  output logic [31:0]      peer,        // the last hello: [7:0] card, [15:8] ncards, [16] port, [31] seen
  output logic [31:0]      peer_cfg,
  output logic [7:0]       hello_cnt,   // hellos received (mod 256)
  output logic             ping_done,   // a PONG received (core measures its own round trip)
  output logic [31:0]      ping,
  output logic             prbs_locked,
  output logic [31:0]      n_txw,       // wire words sent
  output logic [31:0]      n_rxw,       // wire words received
  output logic [31:0]      n_soft,      // cycles with soft_err
  output logic [31:0]      n_perr       // PRBS words in error
);
  logic [31:0] ucnt;
  always_ff @(posedge clk) ucnt <= rst ? '0 : ucnt + 1'b1;

  function automatic logic [63:0] hdr(input logic [7:0] t, input logic [15:0] len,
                                      input logic [15:0] arg, input logic [7:0] aux);
    logic [63:0] h;
    h = {t, 8'h00, len, arg, aux, 8'h00};
    h[7:0] = hdr_chk(h);
    return h;
  endfunction

  // ---------------------------------------------------------------- TX
  logic          echo_set;          // from RX: a PONG to send (echo_w)
  logic [WW-1:0] echo_w;
  logic          echo_pend;
  logic [15:0]   cred_total, cred_sent, cred_diff;
  logic [10:0]   cred_age;
  logic          cred_due;
  logic [20:0]   itmr;
  logic          ident_due, ping_due, abort_due, up_d0;
  logic [15:0]   cn_d;
  wire cn_chg = cn_d != {ncards, card};
  always_ff @(posedge clk) cn_d <= {ncards, card};
  logic          heard, id_reply;   // from RX: the peer's hello came / it had not heard ours
  logic [WW-1:0] cred_w, ident_w, abort_w;
  logic [15:0]   cred_wv;
  localparam logic [1:0] K_NONE = 2'd0, K_PING = 2'd1, K_PONG = 2'd2;
  logic          ctl_v;
  logic [1:0]    ctl_k;
  logic [WW-1:0] ctl_w;
  logic          take_ctl;
  always_ff @(posedge clk) begin
    cred_wv <= cred_total;
    cred_w <= seal({{(WW-64){1'b0}}, hdr(T_CRED, 16'd0, cred_total, 8'd0)}, 8'h0);
    ident_w <= seal({{(WW-128){1'b0}}, cfg, PROTO, 6'd0, heard, port, ncards, card,
                     hdr(T_IDENT, 16'd0, 16'd0, 8'd0)}, 8'h0);
    abort_w <= seal({{(WW-64){1'b0}}, hdr(T_ABORT, 16'd0, 16'd0, card)}, 8'h0);
    cred_total <= popped;
    cred_diff <= cred_total - cred_sent;
    cred_due <= cred_diff != 0 && (cred_diff >= cred_th || cred_age[10]);
    up_d0 <= channel_up;
    if (rst) begin
      echo_pend <= 1'b0;
      cred_sent <= '0;
      cred_age <= '0;
      itmr <= '0;
      ident_due <= 1'b1;             // a hello at reset release
      ping_due <= 1'b0;
      abort_due <= 1'b0;
      ctl_v <= 1'b0;
    end else begin
      cred_age <= (cred_diff == 0) ? '0 : cred_age + {10'd0, ~cred_age[10]};
      itmr <= itmr + 1'b1;
      if (&itmr || (!heard && &itmr[11:0]) || hello_req || cn_chg || id_reply || (channel_up && !up_d0))
        ident_due <= 1'b1;
      if (ping_req) ping_due <= 1'b1;
      if (abort_req) abort_due <= 1'b1;
      if (echo_set) echo_pend <= 1'b1;
      if (take_ctl) ctl_v <= 1'b0;
      else if (!ctl_v) begin
        ctl_w <= '0;
        ctl_k <= K_NONE;
        if (echo_pend && !echo_set) begin
          ctl_w <= echo_w;
          ctl_k <= K_PONG;
          ctl_v <= 1'b1;
          echo_pend <= 1'b0;
        end else if (abort_due) begin
          ctl_w <= abort_w;
          ctl_v <= 1'b1;
          abort_due <= 1'b0;
        end else if (cred_due) begin
          ctl_w <= cred_w;
          ctl_v <= 1'b1;
          cred_sent <= cred_wv;
          cred_age <= '0;
        end else if (ident_due && channel_up) begin
          ctl_w <= ident_w;
          ctl_v <= 1'b1;
          ident_due <= 1'b0;
        end else if (ping_due && channel_up) begin
          ctl_w[63:0] <= hdr(T_PING, 16'd0, 16'd0, 8'd0);
          ctl_k <= K_PING;
          ctl_v <= 1'b1;
          ping_due <= 1'b0;
        end
      end
    end
  end

  // o1: the word chosen for the wire
  typedef enum logic [1:0] {S_IDLE, S_CORE, S_PRBS} txst_t;
  txst_t st;
  logic [15:0]   prem, pseq;
  logic          prem1;
  logic [WW-1:0] gen, prbs_hdr_w;
  logic          inj_pend;
  logic          o1_v, o2_v, o2b_v;
  logic [WW-1:0] o1_d, o2_d, o2b_d;
  logic [1:0]    o1_k, o2_k, o2b_k;
  logic          up_rdy, adv1;
  assign up_rdy = !o2b_v;
  assign adv1 = !o1_v || up_rdy;
  wire tf_hdr = tf_data[WW+3], tf_last = tf_data[WW+2];
  always_comb begin
    take_ctl = 1'b0;
    tf_re = 1'b0;
    if (adv1 && channel_up)
      case (st)
        S_IDLE: if (ctl_v) take_ctl = 1'b1;
                else if (tf_valid) tf_re = 1'b1;
        S_CORE: tf_re = tf_valid;
        default: ;
      endcase
  end
  always_ff @(posedge clk) begin
    prbs_hdr_w <= {{(WW-64){1'b0}}, hdr(T_PRBS, prbs_len, pseq, 8'd0)};
    if (rst) begin
      st <= S_IDLE;
      o1_v <= 1'b0;
      gen <= {64'h0123_4567_89AB_CDEF, 64'hFEDC_BA98_7654_3210, 64'h0F1E_2D3C_4B5A_6978, 64'h8796_A5B4_C3D2_E1F0};
      pseq <= '0;
      inj_pend <= 1'b0;
    end else begin
      if (adv1) begin
        o1_v <= 1'b0;
        o1_k <= K_NONE;
        if (channel_up)
          case (st)
            S_IDLE:
              if (ctl_v) begin
                o1_d <= ctl_w;
                o1_k <= ctl_k;
                o1_v <= 1'b1;
              end else if (tf_valid) begin
                // a word that is not a header here is the TX FIFO out of step (dropped)
                o1_d <= tf_data[WW-1:0];
                o1_v <= tf_hdr;
                if (tf_hdr && !tf_last) st <= S_CORE;
              end else if (prbs_en && prbs_len != 0) begin
                o1_d <= prbs_hdr_w;
                o1_v <= 1'b1;
                st <= S_PRBS;
                prem <= prbs_len;
                prem1 <= prbs_len == 16'd1;
                pseq <= pseq + 1'b1;
              end
            S_CORE:
              if (tf_valid) begin
                o1_d <= tf_data[WW-1:0];
                o1_v <= 1'b1;
                if (tf_last) st <= S_IDLE;
              end
            S_PRBS: begin
              o1_d <= gen ^ {{(WW-1){1'b0}}, inj_pend};
              inj_pend <= 1'b0;
              o1_v <= 1'b1;
              gen <= xs256(gen);
              prem <= prem - 1'b1;
              prem1 <= prem == 16'd2;
              if (prem1) st <= S_IDLE;
            end
            default: st <= S_IDLE;
          endcase
      end
      if (inject) inj_pend <= 1'b1;
    end
  end

  // o2 (+ its skid o2b): to Aurora, PING / PONG stamped for the cycle Aurora takes the word
  function automatic logic [WW-1:0] stamp(input logic [WW-1:0] w, input logic [1:0] k, input logic [31:0] now);
    stamp = w;
    if (k == K_PING) stamp[TS_ATX +: 32] = now;
    if (k == K_PONG) stamp[T_TURN +: 32] = now - w[TS_BRX +: 32];
  endfunction
  logic go1, take2;
  assign go1 = o1_v && up_rdy;
  assign take2 = o2_v && tx_tready;
  always_ff @(posedge clk) begin
    if (rst) begin
      o2_v <= 1'b0;
      o2b_v <= 1'b0;
    end else begin
      if (!o2_v || take2) begin
        if (o2b_v) begin
          o2_d <= stamp(o2b_d, o2b_k, ucnt + 1'b1);
          o2_k <= o2b_k;
          o2_v <= 1'b1;
          o2b_v <= go1;
          o2b_d <= o1_d;
          o2b_k <= o1_k;
        end else begin
          o2_d <= stamp(o1_d, o1_k, ucnt + 1'b1);
          o2_k <= o1_k;
          o2_v <= go1;
        end
      end else begin
        o2_d <= stamp(o2_d, o2_k, ucnt + 1'b1);
        if (go1) begin
          o2b_d <= o1_d;
          o2b_k <= o1_k;
          o2b_v <= 1'b1;
        end
      end
    end
  end
  assign tx_tdata = o2_d;
  assign tx_tvalid = o2_v;

  // ---------------------------------------------------------------- RX
  // q: CHK and the HCRC's halves; r0: the word, a header if it passes
  logic          q_v, q_chk, q_len0, r0_v, r0_chk, r0_len0;
  logic [31:0]   q_ch, q_cl;
  logic [WW-1:0] q_d, r0_d;
  always_ff @(posedge clk) begin
    q_v <= rx_tvalid && channel_up && !rst;
    q_d <= rx_tdata;
    q_chk <= hdr_chk(rx_tdata[63:0]) == rx_tdata[7:0] && known(rx_tdata[63:56]) &&
             ((rx_tdata[63:56] == T_DATA && rx_tdata[47:32] <= 16'(LMAX)) ||
              (rx_tdata[63:56] == T_PRBS) || rx_tdata[47:32] == 0);
    q_len0 <= rx_tdata[47:32] == 16'd0;
    q_ch <= hcrc_hi(rx_tdata);
    q_cl <= hcrc_lo(rx_tdata);
    r0_v <= q_v && channel_up && !rst;
    r0_d <= q_d;
    r0_chk <= q_chk && (!sealed(q_d[63:56]) || (q_ch ^ q_cl) == q_d[255:224]);
    r0_len0 <= q_len0;
  end
  localparam logic [2:0] R_NONE = 3'd0, R_FIFO = 3'd1, R_ECHO = 3'd2, R_CRED = 3'd3, R_IDENT = 3'd4,
                         R_PRBS = 3'd5, R_BAD = 3'd6, R_CTL = 3'd7;
  logic          at_hdr, to_prbs, msg_cred, rem1, rem2, has_trl;
  logic          trunc_err;
  wire restart_hdr = !at_hdr && has_trl && rem1 && r0_chk && sealed(r0_d[63:56]);
  logic [16:0]   rem;
  logic [2:0]    r1_k;
  logic          r1_hdr, r1_last, r1_cred, r1_trl;
  logic [WW-1:0] r1_d;
  wire [7:0] r0_t = r0_d[63:56];
  always_ff @(posedge clk) begin
    r1_k <= R_NONE;
    r1_d <= r0_d;
    trunc_err <= !rst && r0_v && restart_hdr;
    if ((at_hdr || restart_hdr) && r0_chk && r0_t == T_PING) r1_d[TS_BRX +: 32] <= ucnt - 32'd2;
    if ((at_hdr || restart_hdr) && r0_chk && r0_t == T_PONG) r1_d[T_URTT +: 32] <= ucnt - 32'd2 - r0_d[TS_ATX +: 32];
    if (rst || !channel_up) begin
      at_hdr <= 1'b1;
    end else if (r0_v) begin
      if (at_hdr || restart_hdr) begin
        at_hdr <= 1'b1;
        r1_hdr <= 1'b1;
        r1_last <= r0_len0;
        r1_trl <= 1'b0;
        r1_cred <= r0_d[48 + F_CRED];
        msg_cred <= r0_d[48 + F_CRED];
        if (!r0_chk) r1_k <= R_BAD;
        else begin
          case (r0_t)
            T_DATA:  r1_k <= R_FIFO;
            T_PRBS:  r1_k <= R_NONE;
            T_PING:  r1_k <= R_ECHO;
            T_CRED:  r1_k <= R_CRED;
            T_IDENT: r1_k <= R_IDENT;
            default: r1_k <= R_CTL;           // PONG, ABORT
          endcase
          to_prbs <= r0_t == T_PRBS;
          has_trl <= r0_t == T_DATA;
          if (!r0_len0) begin
            // a DATA message: LEN payload words and the trailer; PRBS: LEN words
            at_hdr <= 1'b0;
            rem <= {1'b0, r0_d[47:32]} + 17'(r0_t == T_DATA);
            rem1 <= r0_t != T_DATA && r0_d[47:32] == 16'd1;
            rem2 <= r0_t == T_DATA && r0_d[47:32] == 16'd1;
          end
        end
      end else begin
        r1_hdr <= 1'b0;
        r1_last <= has_trl ? rem2 : rem1;
        r1_trl <= has_trl && rem1;
        r1_cred <= msg_cred;
        r1_k <= to_prbs ? R_PRBS : R_FIFO;
        rem <= rem - 1'b1;
        rem1 <= rem == 17'd2;
        rem2 <= rem == 17'd3;
        if (rem1) at_hdr <= 1'b1;
      end
    end
  end

  // r2: the message's actions
  wire [7:0] r1_t = r1_d[63:56];
  logic in_bad;
  always_ff @(posedge clk) begin
    rf_we <= !rst && r1_k == R_FIFO;
    rf_data <= {r1_hdr, r1_last, r1_cred, r1_trl, r1_d};
    echo_set <= 1'b0;
    hdr_err <= !rst && ((r1_k == R_BAD && !in_bad) || trunc_err);
    id_reply <= r1_k == R_IDENT && !r1_d[81];
    abort_rx <= !rst && r1_k == R_CTL && r1_t == T_ABORT;
    ping_done <= !rst && r1_k == R_CTL && r1_t == T_PONG;
    if (rst) begin
      cr_rx <= '0;
      peer <= '0;
      peer_cfg <= '0;
      hello_cnt <= '0;
      ping <= '0;
      heard <= 1'b0;
      in_bad <= 1'b0;
    end else begin
      if (r1_k == R_BAD) in_bad <= 1'b1;
      else if (r1_k != R_NONE) in_bad <= 1'b0;
      if (hello_req || cn_chg) heard <= 1'b0;
      else if (r1_k == R_IDENT) heard <= 1'b1;
      case (r1_k)
        R_ECHO: begin
          echo_w <= r1_d;
          echo_w[63:56] <= T_PONG;
          echo_w[48 + F_CRED] <= 1'b0;
          echo_w[7:0] <= r1_d[7:0] ^ T_PING ^ T_PONG ^ {7'd0, r1_d[48 + F_CRED]} << F_CRED;
          echo_set <= 1'b1;
        end
        R_CRED: cr_rx <= r1_d[31:16];
        R_IDENT: begin
          peer <= {1'b1, 14'd0, r1_d[80], r1_d[79:72], r1_d[71:64]};
          peer_cfg <= r1_d[127:96];
          hello_cnt <= hello_cnt + 1'b1;
        end
        R_CTL: if (r1_t == T_PONG) ping <= {1'b1, r1_d[T_TURN +: 15], r1_d[T_URTT +: 16]};
        R_BAD: ;
        default: ;
      endcase
    end
  end
  assign ovf = rf_we && rf_full;

  // PRBS checker: the expected word from the first word seen, then the pattern on its own;
  // 8 bad words in a row drop the lock
  logic          plock;
  logic [WW-1:0] pexp;
  logic [2:0]    run;
  logic          pbad;
  always_ff @(posedge clk) begin
    pbad <= 1'b0;
    if (rst) begin
      plock <= 1'b0;
      run <= '0;
    end else if (r1_k == R_PRBS) begin
      if (!plock) begin
        pexp <= xs256(r1_d);
        plock <= 1'b1;
        run <= '0;
      end else begin
        pexp <= xs256(pexp);
        pbad <= r1_d != pexp;
        run <= (r1_d != pexp) ? run + 1'b1 : '0;
        if (r1_d != pexp && run == 3'd7) plock <= 1'b0;
      end
    end
  end
  assign prbs_locked = plock;

  // counters (each steps by at most one a cycle: read across with otpu_link_sync_count)
  always_ff @(posedge clk)
    if (rst) begin
      n_txw <= '0; n_rxw <= '0; n_soft <= '0; n_perr <= '0;
    end else begin
      n_txw <= n_txw + 32'(take2);
      n_rxw <= n_rxw + 32'(r0_v);
      n_soft <= n_soft + 32'(soft_err);
      n_perr <= n_perr + 32'(pbad);
    end
endmodule

`default_nettype wire
