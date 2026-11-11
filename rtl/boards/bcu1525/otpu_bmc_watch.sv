// The BMC bridge's own reads of VCCINT's regulator (otpu_bmc_uart WATCH; docs/bcu1525.md sections
// 13 and 14). The card's power chips are on the board management controller's I2C bus, not the
// FPGA's: the fabric asks the BMC for them on its UART, as the host's tools do
// (opentpu/host/bmc.py), so that the thermal governor (otpu_bcu_gov) holds the regulator's limit
// with the host gone.
//
// A round every POLL_MS: 00 5C 03 (it ends a frame left half sent at the BMC; bmc.py's sync) and
// SETTLE_MS of quiet, the BMC firmware's version until the BMC has answered it (one of
// 0x10002..0x10005, the versions the protocol is known for; another stops the watch for good, as
// bmc.py refuses it: fw_bad), then five reads of the LTC3884 at 0x5B, a request and its reply
// each: page 0's and page 1's READ_TEMPERATURE_1 (PAGE_PLUS_READ), READ_TEMPERATURE_2 (the
// controller's die), READ_PIN and READ_VIN. The requests are the constants REQ (bmc.py's bytes;
// tests/test_bmc.py checks them on the line): reads only, so nothing here can change a setting
// of the regulator's. A reply counts when its frame, its length, its command, its sum, the BMC's
// I2C status, the block's count and the SMBus PEC check, and a temperature also when it is
// between -40 and 150 C: its word replaces the reading's and the reading's age restarts. A reply
// that fails a check, or none within REPLY_MS of the request's last byte, counts in n_bad, and
// REPLY_MS of quiet and the sync come before the next request.
//
// The line is the watch's between the host's holds: the host asks (hold_req, otpu_bmc_uart's
// HOLD), the watch gives it at the end of a round (own drops) and takes it back when the host
// gives it up or HOLD_MS after it gave it (revoke, one cycle: otpu_bmc_uart then clears the
// request); a round follows either at once. The readings' ages run through a hold.
module otpu_bmc_watch #(
  parameter int CLK_HZ    = 100_000_000,
  parameter int CLK_HZ1   = CLK_HZ,        // the fallback clock (otpu_clk_k), or CLK_HZ: none
  parameter int POLL_MS   = 500,            // a round's start to the next one's
  parameter int REPLY_MS  = 200,            // a reply's limit, and the quiet after a failure
  parameter int SETTLE_MS = 5,              // after the sync
  parameter int HOLD_MS   = 1000            // the host's hold at most
) (
  input  logic        clk,
  input  logic        rst,
  // the line, through otpu_bmc_uart's transmitter and receiver
  output logic        own,                  // the watch has the line (reset 1)
  input  logic        hold_req,             // the host asks for it
  output logic        revoke,               // one cycle: HOLD_MS ended the host's hold
  input  logic        tx_idle,              // the transmitter idle
  output logic        tx_go,                // one cycle: tx_byte to the transmitter (own only)
  output logic [7:0]  tx_byte,
  input  logic        rx_take,              // a byte received (rx_byte), one cycle
  input  logic [7:0]  rx_byte,
  input  logic        rx_err,               // a byte lost on the line (parity, framing), one cycle
  // the readings: 0 page 0's temperature, 1 page 1's, 2 the controller's die, 3 PIN, 4 VIN
  output logic [4:0][15:0] word,            // the last good reply's word (LINEAR11)
  output logic [4:0][15:0] age,             // ms since it (saturating; all ones before the first)
  output logic signed [15:0] hot,           // the hottest of readings 0..2, in quarter degrees C
  output logic [15:0] hot_age,              // the oldest of their ages
  output logic [31:0] fw,                   // the firmware's version (0 until it answered)
  output logic        fw_bad,               // not one the protocol is known for: the watch stopped
  output logic [15:0] n_ok,                 // replies taken (saturating)
  output logic [15:0] n_bad                 // exchanges failed (saturating)
);
  localparam int MS = CLK_HZ / 1000;
  localparam int MS1 = CLK_HZ1 / 1000;
  localparam int MW = $clog2(MS > MS1 ? MS : MS1);
  localparam int NQ = 7, QL = 16;           // 0 the version, 1..5 the readings, 6 the sync
  localparam logic [2:0] Q_SYNC = 3'd6;
  localparam logic [7:0] ESC = 8'h5C;
`ifndef SYNTHESIS
  initial begin
    if (MS < 2 || MS1 < 2) $fatal(1, "otpu_bmc_watch: CLK_HZ %0d, CLK_HZ1 %0d", CLK_HZ, CLK_HZ1);
    if (POLL_MS > 65535 || REPLY_MS > 65535 || HOLD_MS > 65535 || SETTLE_MS < 1)
      $fatal(1, "otpu_bmc_watch: POLL_MS %0d, REPLY_MS %0d, SETTLE_MS %0d, HOLD_MS %0d",
             POLL_MS, REPLY_MS, SETTLE_MS, HOLD_MS);
  end
`endif

  // ---- the requests' frames (bmc.wrap_frame(bmc.encode_inner(...)); no 5C inside to double),
  // their lengths, the replies' frames' lengths (inside 5C 02 .. 5C 03) and the PEC's start for
  // each read (CRC-8 over 0xB6, the bytes written, 0xB7)
  localparam logic [7:0] REQ [NQ][QL] = '{
    '{8'h5C, 8'h02, 8'h0C, 8'h00, 8'h00, 8'h0C, 8'h00, 8'h5C, 8'h03,                     // version
      8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00},
    '{8'h5C, 8'h02, 8'h0A, 8'h00, 8'h07, 8'h5B, 8'h84, 8'h04, 8'h06, 8'h02, 8'h00, 8'h8D,  // page 0
      8'h89, 8'h01, 8'h5C, 8'h03},                                                       // 0x8D
    '{8'h5C, 8'h02, 8'h0A, 8'h00, 8'h07, 8'h5B, 8'h84, 8'h04, 8'h06, 8'h02, 8'h01, 8'h8D,  // page 1
      8'h8A, 8'h01, 8'h5C, 8'h03},                                                       // 0x8D
    '{8'h5C, 8'h02, 8'h0A, 8'h00, 8'h04, 8'h5B, 8'h81, 8'h03, 8'h8E, 8'h7B, 8'h01, 8'h5C,  // 0x8E
      8'h03, 8'h00, 8'h00, 8'h00},
    '{8'h5C, 8'h02, 8'h0A, 8'h00, 8'h04, 8'h5B, 8'h81, 8'h03, 8'h97, 8'h84, 8'h01, 8'h5C,  // 0x97
      8'h03, 8'h00, 8'h00, 8'h00},
    '{8'h5C, 8'h02, 8'h0A, 8'h00, 8'h04, 8'h5B, 8'h81, 8'h03, 8'h88, 8'h75, 8'h01, 8'h5C,  // 0x88
      8'h03, 8'h00, 8'h00, 8'h00},
    '{8'h00, 8'h5C, 8'h03, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,  // sync
      8'h00, 8'h00, 8'h00, 8'h00}
  };
  localparam logic [4:0] REQ_N [NQ] = '{5'd9, 5'd16, 5'd16, 5'd13, 5'd13, 5'd13, 5'd3};
  localparam logic [3:0] REP_N [NQ] = '{4'd9, 4'd10, 4'd10, 4'd9, 4'd9, 4'd9, 4'd0};
  localparam logic [7:0] PEC0 [NQ] = '{8'h00, 8'h8B, 8'hE0, 8'hFB, 8'h11, 8'h85, 8'h00};

  function automatic logic [7:0] crc8(input logic [7:0] c, input logic [7:0] b);
    logic [7:0] x = c ^ b;
    for (int i = 0; i < 8; i++) x = x[7] ? {x[6:0], 1'b0} ^ 8'h07 : {x[6:0], 1'b0};
    return x;
  endfunction
  // LINEAR11 (an 11-bit mantissa, a 5-bit exponent, both signed) in quarter units
  function automatic logic signed [31:0] quarter(input logic [15:0] w);
    logic signed [31:0] y = 32'(signed'(w[10:0]));
    logic signed [5:0]  e = 6'(signed'(w[15:11])) + 6'sd2;
    return e[5] ? y >>> 6'(-e) : y <<< e;
  endfunction

  // ---- the millisecond: MS cycles, or MS1 once clk133.tcl set otpu_clk_sel's INIT
  logic [MW-1:0] ms_div, ms_m1;
  logic tick;
  if (CLK_HZ1 != CLK_HZ) begin : g_clk1
    logic sel;
    otpu_clk_sel u_sel (.clk, .sel);
    otpu_clk_k #(.W(MW), .K0(MW'(MS - 1)), .K1(MW'(MS1 - 1))) u_ms_m1 (.sel, .k(ms_m1));
  end else begin : g_clk0
    assign ms_m1 = MW'(MS - 1);
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      ms_div <= '0;
      tick <= 1'b0;
    end else begin
      ms_div <= ms_div == ms_m1 ? '0 : ms_div + 1'b1;
      tick <= ms_div == ms_m1;
    end
  end

  typedef enum logic [3:0] {S_ROUND, S_SEND, S_SETTLE, S_REPLY, S_CHECK, S_NEXT, S_GAP, S_IDLE,
                            S_HOST, S_BACK, S_STOP} st_e;
  st_e         st;
  logic [2:0]  q, cur;                      // the request of the round, the one on the line
  logic [4:0]  k;                           // its bytes sent
  logic        need_sync, fw_ok;
  logic [15:0] t_ms, since;                 // ms in this state, since the round began

  // ---- the reply's frame: 5C 02 starts one, 5C 03 ends it, 5C 5C is a 5C in it; outside a
  // frame bytes are dropped. Byte p_i: 0 the command, 2 the payload's length, 3.. the payload
  // (its first byte the BMC's status for I2C), the last two the sum of the others
  logic        p_esc, p_act, p_bad, p_end, p_err;
  logic [3:0]  p_i;
  logic [7:0]  p_cmd, p_len, p_slo, p_shi, p_crc;
  logic [7:0]  p_pl [5];
  logic [15:0] p_sum;
  wire         rx_in = rx_take && st == S_REPLY;
  wire         d_go  = rx_in && p_act && (p_esc ? rx_byte == ESC : rx_byte != ESC);
  wire [3:0]   rn    = REP_N[cur];
  always_ff @(posedge clk) begin
    p_end <= 1'b0;
    p_err <= 1'b0;
    if (rst || st != S_REPLY) begin
      p_esc <= 1'b0;
      p_act <= 1'b0;
    end else if (rx_in) begin
      if (p_esc) begin
        p_esc <= 1'b0;
        if (rx_byte == 8'h02) begin
          p_act <= 1'b1;
          p_i <= '0;
          p_sum <= '0;
          p_crc <= PEC0[cur];
          p_bad <= 1'b0;
        end else if (p_act && rx_byte == 8'h03) begin
          p_act <= 1'b0;
          p_end <= 1'b1;
        end else if (p_act && rx_byte != ESC) begin
          p_act <= 1'b0;                    // an escape of nothing: the reply is wrong
          p_err <= 1'b1;
        end
      end else if (rx_byte == ESC) begin
        p_esc <= 1'b1;
      end
    end
    if (d_go) begin
      if (p_i == rn) p_bad <= 1'b1;         // longer than the reply
      else p_i <= p_i + 1'b1;
      if (p_i < rn - 4'd2) p_sum <= p_sum + 16'(rx_byte);
      if (p_i >= 4'd4 && p_i < rn - 4'd2) p_crc <= crc8(p_crc, rx_byte);
      if (p_i == 4'd0) p_cmd <= rx_byte;
      if (p_i == 4'd2) p_len <= rx_byte;
      if (p_i >= 4'd3 && p_i < 4'd8) p_pl[3'(p_i - 4'd3)] <= rx_byte;
      if (p_i == rn - 4'd2) p_slo <= rx_byte;
      if (p_i == rn - 4'd1) p_shi <= rx_byte;
    end
  end
  wire paged = cur == 3'd1 || cur == 3'd2;
  wire p_ok  = !p_bad && p_i == rn && p_cmd == (cur == 3'd0 ? 8'h8C : 8'h8A) &&
               p_len == 8'(rn - 4'd5) && p_sum == {p_shi, p_slo} &&
               (cur == 3'd0 || (p_pl[0] == 8'd1 && p_crc == 8'd0 && (!paged || p_pl[1] == 8'd2)));

  // ---- the sequence
  logic [15:0] r_word;                      // the reply's word
  logic [31:0] r_ver;
  logic signed [31:0] r_q;                  // its value in quarter units
  logic signed [15:0] tq [3];               // the temperatures, quarter degrees
  logic [2:0]  ri;                          // the reading (cur - 1)
  logic        commit;                      // reading ri's word and age, this cycle
  wire         q_last = q == 3'd5;
  assign ri = cur - 3'd1;

  always_ff @(posedge clk) begin
    tx_go <= 1'b0;
    revoke <= 1'b0;
    commit <= 1'b0;
    if (tick && t_ms != '1) t_ms <= t_ms + 1'b1;
    if (tick && since != '1) since <= since + 1'b1;
    if (rst) begin
      st <= S_IDLE;
      own <= 1'b1;
      q <= '0;
      cur <= '0;
      k <= '0;
      need_sync <= 1'b1;
      fw_ok <= 1'b0;
      fw_bad <= 1'b0;
      fw <= '0;
      t_ms <= '0;
      since <= 16'(POLL_MS);                // a round at once
      n_ok <= '0;
      n_bad <= '0;
      tx_byte <= '0;
    end else begin
      case (st)
        S_ROUND: begin                      // the next request (the sync first when due)
          cur <= need_sync ? Q_SYNC : q;
          k <= '0;
          st <= S_SEND;
        end
        S_SEND:
          if (!tx_go && tx_idle) begin
            if (k == REQ_N[cur]) begin      // its last byte is out
              t_ms <= '0;
              if (cur == Q_SYNC) begin
                need_sync <= 1'b0;
                st <= S_SETTLE;
              end else begin
                st <= S_REPLY;
              end
            end else begin
              tx_go <= 1'b1;
              tx_byte <= REQ[cur][k[3:0]];
              k <= k + 1'b1;
            end
          end
        S_SETTLE:
          if (t_ms >= 16'(SETTLE_MS)) st <= S_ROUND;
        S_REPLY:
          if (p_end) begin
            st <= S_CHECK;
          end else if (p_err || rx_err || t_ms >= 16'(REPLY_MS)) begin
            if (n_bad != '1) n_bad <= n_bad + 1'b1;
            t_ms <= '0;
            st <= S_GAP;
          end
        S_CHECK:
          if (p_ok) begin
            r_word <= paged ? {p_pl[3], p_pl[2]} : {p_pl[2], p_pl[1]};
            r_ver <= {p_pl[3], p_pl[2], p_pl[1], p_pl[0]};
            r_q <= quarter(paged ? {p_pl[3], p_pl[2]} : {p_pl[2], p_pl[1]});
            st <= S_NEXT;
          end else begin
            if (n_bad != '1) n_bad <= n_bad + 1'b1;
            t_ms <= '0;
            st <= S_GAP;
          end
        S_NEXT: begin                       // the reply taken; the next request or the round's end
          if (cur == 3'd0) begin
            fw <= r_ver;
            if (r_ver >= 32'h10002 && r_ver <= 32'h10005) begin
              fw_ok <= 1'b1;
              if (n_ok != '1) n_ok <= n_ok + 1'b1;
              q <= 3'd1;
              st <= S_ROUND;
            end else begin
              fw_bad <= 1'b1;
              st <= S_STOP;
            end
          end else if (cur <= 3'd3 && (r_q < -32'sd160 || r_q > 32'sd600)) begin
            if (n_bad != '1) n_bad <= n_bad + 1'b1;   // a temperature out of -40 .. 150 C
            t_ms <= '0;
            st <= S_GAP;
          end else begin
            commit <= 1'b1;
            if (n_ok != '1) n_ok <= n_ok + 1'b1;
            if (q_last) begin
              st <= hold_req ? S_HOST : S_IDLE;
              if (hold_req) begin own <= 1'b0; t_ms <= '0; end
            end else begin
              q <= q + 1'b1;
              st <= S_ROUND;
            end
          end
        end
        S_GAP:                              // after a failure: quiet, then the sync
          if (t_ms >= 16'(REPLY_MS)) begin
            need_sync <= 1'b1;
            if (q == 3'd0 || q_last) begin  // no version, no reading: the round ends
              st <= hold_req ? S_HOST : S_IDLE;
              if (hold_req) begin own <= 1'b0; t_ms <= '0; end
            end else begin
              q <= q + 1'b1;
              st <= S_ROUND;
            end
          end
        S_IDLE:
          if (hold_req) begin
            own <= 1'b0;
            t_ms <= '0;
            st <= S_HOST;
          end else if (since >= 16'(POLL_MS)) begin
            since <= '0;
            q <= fw_ok ? 3'd1 : 3'd0;
            need_sync <= 1'b1;
            st <= S_ROUND;
          end
        S_HOST:                             // the host's line
          if (!hold_req || t_ms >= 16'(HOLD_MS)) begin
            revoke <= hold_req;
            st <= S_BACK;
          end
        S_BACK:                             // back once the transmitter is idle
          if (tx_idle) begin
            own <= 1'b1;
            if (fw_bad) begin
              st <= S_STOP;
            end else begin
              since <= '0;
              q <= fw_ok ? 3'd1 : 3'd0;
              need_sync <= 1'b1;
              st <= S_ROUND;
            end
          end
        S_STOP:                             // firmware unknown: no request, the host's holds
          if (hold_req) begin
            own <= 1'b0;
            t_ms <= '0;
            st <= S_HOST;
          end
        default: st <= S_IDLE;
      endcase
    end
  end

  // ---- the readings: words, ages, the temperatures' hottest and oldest
  always_ff @(posedge clk) begin
    if (rst) begin
      word <= '0;
      age <= '1;
      for (int i = 0; i < 3; i++) tq[i] <= '0;
      hot <= '0;
      hot_age <= '1;
    end else begin
      for (int i = 0; i < 5; i++) begin
        if (commit && ri == 3'(i)) begin
          word[i] <= r_word;
          age[i] <= '0;
        end else if (tick && age[i] != '1) begin
          age[i] <= age[i] + 1'b1;
        end
      end
      for (int i = 0; i < 3; i++)
        if (commit && ri == 3'(i)) tq[i] <= 16'(r_q);
      hot <= tq[0] > tq[1] ? (tq[0] > tq[2] ? tq[0] : tq[2]) : (tq[1] > tq[2] ? tq[1] : tq[2]);
      hot_age <= age[0] > age[1] ? (age[0] > age[2] ? age[0] : age[2])
                                 : (age[1] > age[2] ? age[1] : age[2]);
    end
  end
endmodule
