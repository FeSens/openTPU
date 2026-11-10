// Unit test of otpu_bmc_uart's watch (rtl/boards/bcu1525/otpu_bmc_watch.sv; tests/test_bmc.py) at
// 2000 clocks a millisecond and 16 clocks a bit (the card's ratio of bit to millisecond), POLL_MS
// 40, REPLY_MS 4, HOLD_MS 10, through the bridge's AXI4-Lite port. The testbench is the BMC: a
// script (+script=FILE, $readmemh, 16-bit words; tests/test_bmc.py writes it with bmc.py's
// framing, PEC and LINEAR11) of the bytes expected on the line (0x1nnn: nnn bytes follow), the
// reply to send (0x2nnn: nnn bytes, bit8 a byte with a bad parity bit), quiet (0x3nnn: no byte
// for nnn ms) and its end (0). `served` counts its entries done; the flow below checks the
// registers at its points (+expect=FILE: the readings' words and the host's reply). Checked: the
// rounds (the sync, the firmware's version first and again until it answers, the five reads),
// each failure counted and skipped (a lost reply, a wrong PEC, a wrong sum, the BMC's NACK, a
// parity error, a temperature out of range), a 5C doubled in a reply, the readings' words, ages
// and hottest (W_*, reg_hot, reg_age); the host's hold granted between rounds and at a round's
// end, never in one, its bytes on the line and its reply in its RX FIFO, its release, HOLD_MS's
// revocation (REVOKED, the request cleared), its TX bytes held without the line; an unknown
// firmware stopping the watch (no byte more), holds still granted. Prints PASS, or the errors
// and FAIL.
module tb_bmc_watch;
  localparam int MSC = 2000, DIV = 16, BIT = 10 * DIV;     // BIT: a bit's time (the clock is 10)
  localparam int POLL = 40, REPLY = 4, HOLDT = 10;

  logic clk = 1'b0;
  always #5 clk = !clk;
  logic rst = 1'b1;

  logic [15:0] awaddr = '0, araddr = '0;
  logic [31:0] wdata = '0, rdata;
  logic [3:0]  wstrb = 4'hf;
  logic awvalid = 1'b0, wvalid = 1'b0, bready = 1'b1, arvalid = 1'b0, rready = 1'b1;
  logic awready, wready, bvalid, arready, rvalid;
  logic [1:0] bresp, rresp;
  logic tx, drv = 1'b1;
  logic signed [15:0] reg_hot;
  logic [15:0] reg_age;

  otpu_bmc_uart #(.CLK_HZ(MSC * 1000), .BAUD(MSC * 1000 / DIV), .PARITY(2), .DEPTH(16),
                  .WATCH(1'b1), .POLL_MS(POLL), .REPLY_MS(REPLY), .HOLD_MS(HOLDT)) dut (
    .clk, .rst,
    .s_awaddr(awaddr), .s_awvalid(awvalid), .s_awready(awready), .s_wdata(wdata),
    .s_wstrb(wstrb), .s_wvalid(wvalid), .s_wready(wready), .s_bresp(bresp), .s_bvalid(bvalid),
    .s_bready(bready), .s_araddr(araddr), .s_arvalid(arvalid), .s_arready(arready),
    .s_rdata(rdata), .s_rresp(rresp), .s_rvalid(rvalid), .s_rready(rready), .tx, .rx(drv),
    .reg_hot, .reg_age);

  int errs = 0;
  task automatic fail(input string s);
    errs++;
    if (errs <= 30) $display("ERROR at %0t: %s", $time, s);
  endtask
  task automatic expect_eq(input string what, input logic [31:0] got, input logic [31:0] want);
    if (got !== want) fail($sformatf("%s: %h, expected %h", what, got, want));
  endtask

  // ---- AXI4-Lite master (tb_bmc_uart's)
  int n_aw = 0, n_ar = 0, n_b = 0, n_r = 0;
  logic [31:0] r_data;
  logic [1:0]  r_resp, b_resp;
  always_ff @(posedge clk) begin
    if (awvalid && awready) n_aw <= n_aw + 1;
    if (arvalid && arready) n_ar <= n_ar + 1;
    if (bvalid && bready) begin n_b <= n_b + 1; b_resp <= bresp; end
    if (rvalid && rready) begin n_r <= n_r + 1; r_data <= rdata; r_resp <= rresp; end
  end
  task automatic wr(input logic [15:0] a, input logic [31:0] d);
    int k0 = n_aw, b0 = n_b;
    @(negedge clk);
    awaddr = a; awvalid = 1'b1; wdata = d; wvalid = 1'b1;
    while (n_aw == k0) @(negedge clk);
    awvalid = 1'b0; wvalid = 1'b0;
    while (n_b == b0) @(negedge clk);
  endtask
  task automatic rd(input logic [15:0] a, output logic [31:0] d);
    int k0 = n_ar, r0 = n_r;
    @(negedge clk);
    araddr = a; arvalid = 1'b1;
    while (n_ar == k0) @(negedge clk);
    arvalid = 1'b0;
    while (n_r == r0) @(negedge clk);
    d = r_data;
  endtask
  localparam logic [15:0] RX = 16'h0, TX = 16'h4, STAT = 16'h8, CTRL = 16'hC, LEVEL = 16'h10,
                          ID = 16'h14, HOLD = 16'h1C, W_HOT = 16'h20, W_P0 = 16'h24,
                          W_P1 = 16'h28, W_CTL = 16'h2C, W_PIN = 16'h30, W_VIN = 16'h34,
                          W_FW = 16'h38, W_CNT = 16'h3C;

  // ---- the BMC's side of the line: every frame on TX (start 0, even parity, stop 1) into line_q
  byte line_q[$];
  always @(negedge tx) begin
    logic [10:0] f;
    #(BIT / 2);
    for (int i = 0; i < 11; i++) begin
      f[i] = tx;
      if (i < 10) #(BIT);
    end
    if (f[0] !== 1'b0 || f[10] !== 1'b1 || f[9] !== ^f[8:1])
      fail($sformatf("frame on the line %b", f));
    line_q.push_back(f[8:1]);
  end
  task automatic frame(input byte b, input bit bad_par = 0);
    drv = 1'b0; #(BIT);
    for (int i = 0; i < 8; i++) begin drv = b[i]; #(BIT); end
    drv = ^b ^ bad_par; #(BIT);
    drv = 1'b1; #(BIT);
  endtask
  task automatic ms(input int n);
    repeat (n * MSC) @(negedge clk);
  endtask

  // ---- the script
  logic [15:0] script [8192];
  logic [31:0] expv [256];
  int served = 0;
  bit script_done = 0;
  task automatic run_script();
    int pc = 0, n;
    forever begin
      n = int'(script[pc][11:0]);
      case (script[pc][15:12])
        4'h0: break;
        4'h1: begin                                       // bytes expected on the line
          int k = 0;
          while (line_q.size() < n && k < 3 * POLL * MSC) begin @(negedge clk); k++; end
          if (line_q.size() < n) begin
            fail($sformatf("entry %0d: %0d of %0d bytes on the line", served, line_q.size(), n));
            break;
          end
          for (int i = 0; i < n; i++) begin
            byte b = line_q.pop_front();
            if (b != script[pc + 1 + i][7:0])
              fail($sformatf("entry %0d byte %0d: %h, expected %h", served, i, b,
                             script[pc + 1 + i][7:0]));
          end
          pc += 1 + n;
          if (script[pc][15:12] != 4'h2) served++;
        end
        4'h2: begin                                       // the reply
          #(3 * BIT);
          for (int i = 0; i < n; i++) frame(script[pc + 1 + i][7:0], script[pc + 1 + i][8]);
          pc += 1 + n;
          served++;
        end
        4'h3: begin                                       // quiet
          ms(n);
          if (line_q.size() != 0) fail($sformatf("entry %0d: %0d bytes in quiet", served,
                                                 line_q.size()));
          pc += 1;
          served++;
        end
        default: begin
          fail($sformatf("script word %0d: %h", pc, script[pc]));
          break;
        end
      endcase
    end
    script_done = 1;
  endtask
  task automatic wait_served(input int n);
    while (served < n && !script_done) @(negedge clk);
    if (served < n) fail($sformatf("the script ended at entry %0d, before %0d", served, n));
    repeat (100) @(negedge clk);                          // the watch takes the last reply
  endtask

  // ---- the flow
  localparam byte VER_REQ [9] = '{8'h5C, 8'h02, 8'h0C, 8'h00, 8'h00, 8'h0C, 8'h00, 8'h5C, 8'h03};
  logic [31:0] v, v2;
  byte got[$];
  string path;
  initial begin
    if (!$value$plusargs("script=%s", path)) begin $display("FAIL: no +script"); $finish; end
    $readmemh(path, script);
    if (!$value$plusargs("expect=%s", path)) begin $display("FAIL: no +expect"); $finish; end
    $readmemh(path, expv);
    repeat (5) @(negedge clk);
    rst = 1'b0;
    fork run_script(); join_none
    rd(ID, v);    expect_eq("ID", v, 32'h424D_4332);
    rd(W_HOT, v); expect_eq("W_HOT before a reading", v, 32'hFFFF_0000);
    if (reg_age !== 16'hFFFF) fail("reg_age before a reading");

    // rounds 1 (the version's reply lost) and 2 (the version, the five reads)
    wait_served(9);
    rd(W_FW, v);  expect_eq("W_FW", v, 32'h10004);
    rd(W_CNT, v); expect_eq("W_CNT after round 2", v, {16'd1, 16'd6});
    for (int i = 0; i < 5; i++) begin
      rd(16'(W_P0 + 4 * i), v);
      expect_eq($sformatf("reading %0d's word", i), v[15:0], expv[i][15:0]);
      if (v[31:16] > 15) fail($sformatf("reading %0d's age %0d after round 2", i, v[31:16]));
    end
    rd(W_HOT, v); expect_eq("W_HOT's hottest (52 C)", v[15:0], 208);
    if (v[31:16] > 15) fail($sformatf("W_HOT's age %0d", v[31:16]));
    if (reg_hot !== 16'sd208 || reg_age > 15) fail($sformatf("reg_hot %0d, reg_age %0d", reg_hot, reg_age));
    rd(STAT, v);  expect_eq("STATUS's hold bits", v[13:11], 0);
    rd(HOLD, v);  expect_eq("HOLD", v, 0);
    rd(LEVEL, v); expect_eq("the host's RX FIFO without the line", v[15:0], 0);

    // round 3: a wrong PEC, a wrong sum, the BMC's NACK, a parity error; VIN taken
    wait_served(19);
    rd(W_CNT, v); expect_eq("W_CNT after round 3", v, {16'd5, 16'd7});
    rd(W_P0, v);  expect_eq("page 0's word kept", v[15:0], expv[0][15:0]);
    if (v[31:16] < 40) fail($sformatf("page 0's age %0d after round 3", v[31:16]));
    rd(W_CTL, v); expect_eq("the controller's word kept", v[15:0], expv[2][15:0]);
    rd(W_PIN, v); expect_eq("PIN's word kept", v[15:0], expv[3][15:0]);
    rd(W_VIN, v); expect_eq("VIN's new word", v[15:0], expv[5][15:0]);
    if (v[31:16] > 2) fail($sformatf("VIN's age %0d after round 3", v[31:16]));

    // round 4: page 0 at 160 C (out of range), page 1's reply with a 5C (75.5 C: the hottest)
    wait_served(26);
    rd(W_CNT, v); expect_eq("W_CNT after round 4", v, {16'd6, 16'd11});
    rd(W_P0, v);  expect_eq("page 0's word kept (160 C refused)", v[15:0], expv[0][15:0]);
    rd(W_P1, v);  expect_eq("page 1's word (a 5C doubled)", v[15:0], expv[6][15:0]);
    rd(W_HOT, v); expect_eq("W_HOT's hottest (75.5 C)", v[15:0], expv[7][15:0]);
    if (v[31:16] < 60) fail($sformatf("W_HOT's age %0d: page 0's, refused twice", v[31:16]));
    rd(LEVEL, v); expect_eq("the host's RX FIFO after four rounds", v[15:0], 0);

    // round 5: a hold between rounds (granted at once), the host's sync and version request on
    // the line, its reply in the RX FIFO; the ages run through the hold; the release, a round
    ms(1);
    wr(HOLD, 1);
    repeat (20) @(negedge clk);
    rd(HOLD, v);  expect_eq("HOLD between rounds", v, 3);
    rd(STAT, v);  if (!v[11]) fail($sformatf("HELD not in STATUS %h", v));
    wr(CTRL, 3);
    wr(TX, 32'h00); wr(TX, 32'h5C); wr(TX, 32'h03);
    foreach (VER_REQ[i]) wr(TX, {24'd0, VER_REQ[i]});
    rd(W_P0, v);
    wait_served(27);
    ms(1);
    got = {};
    forever begin rd(RX, v2); if (!v2[31]) break; got.push_back(v2[7:0]); end
    if (got.size() != int'(expv[8])) fail($sformatf("the host's reply: %0d bytes, expected %0d", got.size(), expv[8]));
    foreach (got[i]) if (got[i] != expv[9 + i][7:0]) fail($sformatf("the host's reply byte %0d: %h", i, got[i]));
    rd(W_P0, v2);
    if (v2[31:16] <= v[31:16]) fail($sformatf("page 0's age %0d then %0d in the hold", v[31:16], v2[31:16]));
    rd(W_CNT, v); expect_eq("W_CNT in the hold", v, {16'd6, 16'd11});
    wr(HOLD, 0);
    wait_served(33);
    rd(HOLD, v);  expect_eq("HOLD after the release", v, 0);
    rd(W_CNT, v); expect_eq("W_CNT after round 5", v, {16'd6, 16'd16});
    rd(W_P0, v);  if (v[31:16] > 15) fail($sformatf("page 0's age %0d after round 5", v[31:16]));

    // round 6: a hold asked in a round (granted at its end, not before); the host's request on
    // the line, unanswered; HOLD_MS ends the hold (REVOKED, the request cleared), a round follows;
    // a TX byte without the line stays in the FIFO
    wait (served >= 34);
    wr(HOLD, 1);
    rd(HOLD, v);  expect_eq("HOLD asked in a round", v, 1);
    wait (served >= 38);
    rd(HOLD, v);  expect_eq("HOLD before the round's last reply", v, 1);
    wait_served(39);
    rd(HOLD, v);  expect_eq("HOLD at the round's end", v, 3);
    foreach (VER_REQ[i]) wr(TX, {24'd0, VER_REQ[i]});
    wait_served(40);
    ms(HOLDT + 1);
    rd(HOLD, v);  expect_eq("HOLD after HOLD_MS", v, 0);
    rd(STAT, v);  if (!v[12] || v[11]) fail($sformatf("STATUS %h after HOLD_MS (REVOKED, not HELD)", v));
    rd(STAT, v);  if (v[12]) fail("REVOKED not cleared by the STATUS read");
    wr(TX, 32'hA5);
    repeat (5 * BIT) @(negedge clk);
    rd(LEVEL, v); expect_eq("a TX byte without the line", v[31:16], 1);
    wr(CTRL, 1);
    rd(LEVEL, v); expect_eq("LEVEL after the TX reset", v[31:16], 0);
    wait_served(46);
    rd(W_CNT, v); expect_eq("W_CNT after round 6 and the next", v, {16'd6, 16'd26});

    // a reset, then a firmware the protocol is not known for: the watch stops; holds still
    ms(1);
    rst = 1'b1;
    repeat (5) @(negedge clk);
    rst = 1'b0;
    wait_served(48);
    rd(STAT, v);  if (!v[13]) fail($sformatf("STATUS %h: the watch not stopped", v));
    rd(W_FW, v);  expect_eq("W_FW (unknown)", v, 32'h20001);
    rd(W_CNT, v); expect_eq("W_CNT stopped", v, 0);
    rd(W_HOT, v); expect_eq("W_HOT stopped", v, 32'hFFFF_0000);
    if (reg_age !== 16'hFFFF) fail("reg_age stopped");
    wr(HOLD, 1);
    repeat (20) @(negedge clk);
    rd(HOLD, v);  expect_eq("HOLD with the watch stopped", v, 3);
    wr(HOLD, 0);
    repeat (5 * BIT) @(negedge clk);
    rd(HOLD, v);  expect_eq("HOLD released, stopped", v, 0);
    wait (script_done);
    ms(1);
    if (line_q.size() != 0) fail($sformatf("%0d bytes on the line after the script", line_q.size()));

    $display("entries %0d", served);
    if (errs == 0) $display("PASS");
    else $display("FAIL (%0d errors)", errs);
    $finish;
  end
  initial begin
    #(1000 * MSC * 10);
    $display("FAIL: timeout (entry %0d)", served);
    $finish;
  end
endmodule
