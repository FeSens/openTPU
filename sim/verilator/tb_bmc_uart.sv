// Unit test of otpu_bmc_uart (rtl/boards/bcu1525/otpu_bmc_uart.sv; tests/test_bmc.py) without its
// watch (WATCH 0; tb_bmc_watch.sv has it) at 16 clocks a bit and 16-byte FIFOs, through its
// AXI4-Lite port. The line is looped back (TX into RX) or
// driven by the testbench (frames with a bad parity bit or a low stop bit, a break, a glitch), and
// a checker of its own samples every frame on TX at its bits' middles (start 0, even parity, stop
// 1). Checked: the registers' values, the bytes in order through both FIFOs, the frames on the
// line, the error flags (set by the event, cleared by a STATUS read) and the FIFOs' limits (a full
// TX FIFO drops a write, a full RX FIFO loses a byte: overrun), CTRL's FIFO resets (the frame on
// the line finishes whole), and AW given before W. Prints PASS, or the errors and FAIL.
module tb_bmc_uart;
  localparam int DIV = 16, DEPTH = 16, BIT = 10 * DIV;    // BIT: a bit's time (the clock is 10)

  logic clk = 1'b0;
  always #5 clk = !clk;
  logic rst = 1'b1;

  logic [15:0] awaddr = '0, araddr = '0;
  logic [31:0] wdata = '0, rdata;
  logic [3:0]  wstrb = 4'hf;
  logic awvalid = 1'b0, wvalid = 1'b0, bready = 1'b1, arvalid = 1'b0, rready = 1'b1;
  logic awready, wready, bvalid, arready, rvalid;
  logic [1:0] bresp, rresp;
  logic tx, rx, loop = 1'b1, drv = 1'b1;
  assign rx = loop ? tx : drv;

  otpu_bmc_uart #(.CLK_HZ(DIV * 100_000), .BAUD(100_000), .PARITY(2), .DEPTH(DEPTH),
                  .WATCH(1'b0)) dut (
    .clk, .rst,
    .s_awaddr(awaddr), .s_awvalid(awvalid), .s_awready(awready), .s_wdata(wdata),
    .s_wstrb(wstrb), .s_wvalid(wvalid), .s_wready(wready), .s_bresp(bresp), .s_bvalid(bvalid),
    .s_bready(bready), .s_araddr(araddr), .s_arvalid(arvalid), .s_arready(arready),
    .s_rdata(rdata), .s_rresp(rresp), .s_rvalid(rvalid), .s_rready(rready), .tx, .rx,
    .reg_hot(), .reg_age());

  int errs = 0;
  task automatic fail(input string s);
    errs++;
    if (errs <= 20) $display("ERROR at %0t: %s", $time, s);
  endtask
  task automatic expect_eq(input string what, input logic [31:0] got, input logic [31:0] want);
    if (got !== want) fail($sformatf("%s: %h, expected %h", what, got, want));
  endtask

  // ---- AXI4-Lite master: drives at the falling edge; the handshakes are counted at the rising one
  // (n_aw, n_ar, n_b, n_r), where the slave takes them
  int n_aw = 0, n_ar = 0, n_b = 0, n_r = 0;
  logic [31:0] r_data;
  logic [1:0]  r_resp, b_resp;
  always_ff @(posedge clk) begin
    if (awvalid && awready) n_aw <= n_aw + 1;
    if (arvalid && arready) n_ar <= n_ar + 1;
    if (bvalid && bready) begin n_b <= n_b + 1; b_resp <= bresp; end
    if (rvalid && rready) begin n_r <= n_r + 1; r_data <= rdata; r_resp <= rresp; end
  end
  task automatic wr(input logic [15:0] a, input logic [31:0] d, input bit w_late = 0);
    int k0 = n_aw, b0 = n_b;
    @(negedge clk);
    awaddr = a; awvalid = 1'b1; wdata = d; wvalid = !w_late;
    if (w_late) begin
      repeat (3) @(negedge clk);
      if (n_aw != k0) fail("AW taken without W");
      wvalid = 1'b1;
    end
    while (n_aw == k0) @(negedge clk);
    awvalid = 1'b0; wvalid = 1'b0;
    while (n_b == b0) @(negedge clk);
    if (b_resp != 2'b00) fail($sformatf("write %h: BRESP %0d", a, b_resp));
  endtask
  task automatic rd(input logic [15:0] a, output logic [31:0] d);
    int k0 = n_ar, r0 = n_r;
    @(negedge clk);
    araddr = a; arvalid = 1'b1;
    while (n_ar == k0) @(negedge clk);
    arvalid = 1'b0;
    while (n_r == r0) @(negedge clk);
    d = r_data;
    if (r_resp != 2'b00) fail($sformatf("read %h: RRESP %0d", a, r_resp));
  endtask
  localparam logic [15:0] RX = 16'h0, TX = 16'h4, STAT = 16'h8, CTRL = 16'hC, LEVEL = 16'h10,
                          ID = 16'h14, DIVR = 16'h18;

  // ---- the line checker: every frame on TX
  byte wire_q[$];
  int  n_frames = 0;
  always @(negedge tx) begin
    logic [10:0] f;
    #(BIT / 2);
    for (int i = 0; i < 11; i++) begin
      f[i] = tx;
      if (i < 10) #(BIT);
    end
    n_frames++;
    if (f[0] !== 1'b0 || f[10] !== 1'b1 || f[9] !== ^f[8:1])
      fail($sformatf("frame on the line %b", f));
    wire_q.push_back(f[8:1]);
  end

  // ---- the testbench drives RX (loop = 0)
  task automatic frame(input byte b, input bit bad_par = 0, input bit low_stop = 0);
    drv = 1'b0; #(BIT);
    for (int i = 0; i < 8; i++) begin drv = b[i]; #(BIT); end
    drv = ^b ^ bad_par; #(BIT);
    drv = !low_stop; #(BIT);                          // (a low stop bit stays low: a break)
  endtask

  // ---- the flow
  logic [31:0] v;
  byte sent[$], got[$];
  task automatic drain_rx(output byte q[$]);
    q = {};
    forever begin
      rd(RX, v);
      if (!v[31]) break;
      q.push_back(v[7:0]);
    end
  endtask
  task automatic wait_tx_idle();
    int k = 0;
    do begin rd(STAT, v); k++; end while (!v[9] && k < 10000);
    if (!v[9]) fail("TX never idle");
    #(2 * BIT);
  endtask

  initial begin
    repeat (5) @(negedge clk);
    rst = 1'b0;
    repeat (5) @(negedge clk);
    rd(ID, v);    expect_eq("ID", v, 32'h424D_4331);
    rd(DIVR, v);  expect_eq("DIV", v, DIV);
    rd(STAT, v);  expect_eq("STATUS at reset (TX empty and idle, line high, HELD)", v, 32'hE04);
    rd(RX, v);    expect_eq("RX empty", v, 0);

    // loopback: bytes through TX, the line and RX, in order; AW before W once
    sent = '{8'h00, 8'h5C, 8'h03, 8'hA5, 8'hFF, 8'h80, 8'h01, 8'h5C};
    foreach (sent[i]) wr(TX, sent[i], i == 2);
    rd(LEVEL, v);
    if (v[31:16] == 0) fail("LEVEL: no TX bytes queued");
    wait_tx_idle();
    rd(LEVEL, v); expect_eq("LEVEL after the loopback", v, sent.size());
    rd(STAT, v);  expect_eq("STATUS after the loopback (RX valid)", v & 32'h1FF, 32'h005);
    drain_rx(got);
    if (got != sent) fail($sformatf("RX %p, sent %p", got, sent));
    if (wire_q != sent) fail($sformatf("line %p, sent %p", wire_q, sent));

    // TX FIFO full: DEPTH + 4 writes at once, the last ones dropped; every taken byte on the line
    wire_q = {};
    loop = 1'b0;
    for (int i = 0; i < DEPTH + 4; i++) wr(TX, 8'(i * 7 + 1));
    rd(STAT, v);
    if (!v[8]) fail("TX_DROP not set by writes to a full FIFO");
    rd(STAT, v);
    if (v[8]) fail("TX_DROP not cleared by the STATUS read");
    wait_tx_idle();
    if (wire_q.size() < DEPTH || wire_q.size() > DEPTH + 1)
      fail($sformatf("%0d frames after %0d writes into %0d bytes", wire_q.size(), DEPTH + 4, DEPTH));
    foreach (wire_q[i]) if (wire_q[i] != 8'(i * 7 + 1)) fail($sformatf("frame %0d: %h", i, wire_q[i]));

    // CTRL bit0 while sending: the frame on the line finishes, the rest is dropped
    wire_q = {};
    for (int i = 0; i < 6; i++) wr(TX, 8'h30 + 8'(i));
    #(3 * BIT);
    wr(CTRL, 1);
    rd(LEVEL, v); expect_eq("LEVEL after the TX reset", v[31:16], 0);
    wait_tx_idle();
    if (wire_q.size() != 1 || wire_q[0] != 8'h30)
      fail($sformatf("TX reset mid-frame: line %p", wire_q));

    // RX errors from the testbench's frames
    frame(8'h41, 1);                                  // bad parity: no byte, the flag
    #(BIT);
    rd(STAT, v);
    if (!v[7] || v[0]) fail($sformatf("parity error: STATUS %h", v));
    rd(STAT, v);
    if (v[7]) fail("parity flag not cleared by the STATUS read");
    frame(8'h42, 0, 1);                               // low stop bit, then a break
    #(20 * BIT);
    drv = 1'b1;
    #(2 * BIT);
    rd(STAT, v);
    if (!v[6] || v[0]) fail($sformatf("framing error: STATUS %h", v));
    drv = 1'b0; #(BIT / 4); drv = 1'b1;               // a glitch: not a start bit
    #(12 * BIT);
    frame(8'h43);
    #(BIT);
    drain_rx(got);
    if (got.size() != 1 || got[0] != 8'h43)
      fail($sformatf("after the break and the glitch: RX %p", got));
    rd(STAT, v);
    if (v[7:5] != 0) fail($sformatf("flags after a good frame: STATUS %h", v));

    // RX FIFO full: DEPTH + 2 frames, two lost (overrun); CTRL bit1 empties it
    for (int i = 0; i < DEPTH + 2; i++) frame(8'(i));
    #(BIT);
    rd(LEVEL, v); expect_eq("LEVEL with RX full", v[15:0], DEPTH);
    rd(STAT, v);
    if (!v[5] || !v[1]) fail($sformatf("overrun: STATUS %h", v));
    rd(RX, v);    expect_eq("the oldest byte kept", v, 32'h8000_0000);
    wr(CTRL, 2);
    rd(LEVEL, v); expect_eq("LEVEL after the RX reset", v[15:0], 0);
    rd(RX, v);    expect_eq("RX after the RX reset", v, 0);

    $display("frames on the line %0d", n_frames);
    if (errs == 0) $display("PASS");
    else $display("FAIL (%0d errors)", errs);
    $finish;
  end
  initial begin
    #(4000 * BIT);
    $display("FAIL: timeout");
    $finish;
  end
endmodule
