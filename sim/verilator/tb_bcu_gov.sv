// Unit test of otpu_bcu_gov (rtl/boards/bcu1525/otpu_bcu_gov.sv; tests/test_bcu_gov.py) at 10
// clocks a millisecond and OFF_DLY 16, through its AXI4-Lite port, the SYSMON's code and alarms
// driven by the testbench. Checked: the registers after reset (SAFE, every domain off, UART_SEL 1),
// each condition GO needs (a valid code, the die at or below RESUME, no alarm, a heartbeat) and
// each cause that ends RUN (the die at TRIP, the temperature alarm, the OT alarm, a code out of
// range, no heartbeat, the host's SAFE) with its sticky bit and counter; the order going off (the
// accelerator at once, the DIMMs OFF_DLY cycles later, none turned on by CTRL meanwhile); TRIP,
// RESUME and HB_MS kept within their limits; WARN, CLEAR, the highest code, RUN_MS. With HAS_REG
// (the regulator's reading driven here, as otpu_bmc_uart's watch gives it): GO refused above
// REG_RESUME and with a reading as old as REG_STALE, RUN ended at REG_TRIP and by a stale
// reading (causes 21, 22: thermal), REG_WARN, REG_TRIP / REG_RESUME / REG_STALE within their
// limits, the regulator's highest and its CLEAR. The code's age at TEMP_STALE_MS 1000 here
// (TEMP_AGE, TEMP_STALE): counted from each change, held while the temperature alarm stands; a
// code held TEMP_STALE_MS drops TEMP_VALID, ends RUN (cause 23 alone: thermal) and refuses GO
// until it moves; one that changes within each TEMP_STALE_MS keeps RUN. Prints PASS, or the
// errors and FAIL.
module tb_bcu_gov;
  localparam int MSC = 10, OFF = 16, STALE = 1000;

  logic clk = 1'b0;
  always #5 clk = !clk;
  logic rst = 1'b1;

  logic [11:0] awaddr = '0, araddr = '0;
  logic [31:0] wdata = '0, rdata;
  logic [3:0]  wstrb = 4'hf;
  logic awvalid = 1'b0, wvalid = 1'b0, bready = 1'b1, arvalid = 1'b0, rready = 1'b1;
  logic awready, wready, bvalid, arready, rvalid;
  logic [1:0] bresp, rresp;
  logic [9:0] temp = '0;
  logic temp_alarm = 1'b0, ot_alarm = 1'b0;
  logic signed [15:0] reg_t = 16'sd200;                      // 50 C, quarter degrees
  logic [15:0] reg_age = 16'd5;
  logic acc_on;
  logic [3:0] dimm_on, dimm_ok;
  logic [1:0] uart_sel;
  logic [2:0] led;

  otpu_bcu_gov #(.CLK_HZ(MSC * 1000), .OFF_DLY(OFF), .HAS_REG(1'b1), .TEMP_STALE_MS(STALE)) dut (
    .clk, .rst, .temp, .temp_alarm, .ot_alarm, .reg_t, .reg_age, .acc_on, .dimm_on, .dimm_ok,
    .uart_sel, .led,
    .s_awaddr(awaddr), .s_awvalid(awvalid), .s_awready(awready), .s_wdata(wdata),
    .s_wstrb(wstrb), .s_wvalid(wvalid), .s_wready(wready), .s_bresp(bresp), .s_bvalid(bvalid),
    .s_bready(bready), .s_araddr(araddr), .s_arvalid(arvalid), .s_arready(arready),
    .s_rdata(rdata), .s_rresp(rresp), .s_rvalid(rvalid), .s_rready(rready));

  int errs = 0;
  task automatic fail(input string s);
    errs++;
    if (errs <= 20) $display("ERROR at %0t: %s", $time, s);
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
  task automatic wr(input logic [11:0] a, input logic [31:0] d, input bit w_late = 0);
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
  task automatic rd(input logic [11:0] a, output logic [31:0] d);
    int k0 = n_ar, r0 = n_r;
    @(negedge clk);
    araddr = a; arvalid = 1'b1;
    while (n_ar == k0) @(negedge clk);
    arvalid = 1'b0;
    while (n_r == r0) @(negedge clk);
    d = r_data;
    if (r_resp != 2'b00) fail($sformatf("read %h: RRESP %0d", a, r_resp));
  endtask
  localparam logic [11:0] ID = 12'h00, STAT = 12'h04, CTRL = 12'h08, CMD = 12'h0C, TEMP = 12'h10,
                          TRIP = 12'h14, RESUME = 12'h18, HBMS = 12'h1C, KICK = 12'h20,
                          TRIPS = 12'h24, RUNMS = 12'h28, LIMITS = 12'h2C, REG = 12'h30,
                          RTRIP = 12'h34, RRES = 12'h38, RSTALE = 12'h3C, RAGE = 12'h40,
                          RLIM = 12'h44, TAGE = 12'h48, TSTALE = 12'h4C;
  localparam logic [31:0] GO = 1, SAFE = 2, CLEAR = 4;

  // ---- the domains' order: the accelerator off no later than the DIMMs, the DIMMs OFF cycles
  // after it (counted here), and never a DIMM turned on while RUN is 0
  int off_gap = -1, acc_off_at = 0, cyc = 0;
  logic acc_q = 1'b0;
  logic [3:0] dimm_q = '0;
  always_ff @(posedge clk) begin
    automatic int a = (acc_q && !acc_on) ? cyc : acc_off_at;
    cyc <= cyc + 1;
    acc_q <= acc_on;
    dimm_q <= dimm_on;
    acc_off_at <= a;
    if (dimm_q != 0 && dimm_on == 0) off_gap <= cyc - a;
    if (!rst && (dimm_on & ~dimm_q) != 0 && !dut.run) fail("a DIMM turned on outside RUN");
    if (!rst && acc_on && !dut.run) fail("the accelerator on outside RUN");
    if (!rst && (dimm_ok & ~dimm_on) != 0) fail($sformatf("dimm_ok %b ahead of dimm_on %b", dimm_ok, dimm_on));
    if (!rst && dimm_ok != 0 && !dut.run) fail("dimm_ok outside RUN");
  end

  logic [31:0] v;
  task automatic ms(input int n);
    repeat (n * MSC) @(negedge clk);
  endtask
  // GO from SAFE with every domain, a kick first; expect RUN or not
  task automatic go_expect(input bit want, input string what);
    wr(KICK, 0);
    wr(CMD, GO);
    repeat (3) @(negedge clk);
    rd(STAT, v);
    if (v[0] !== want) fail($sformatf("%s: RUN %0d, expected %0d (STATUS %h)", what, v[0], want, v));
  endtask
  task automatic expect_stop(input int cause_bit, input string what);
    repeat (3) @(negedge clk);
    rd(STAT, v);
    if (v[0] !== 1'b0) fail($sformatf("%s: still RUN (STATUS %h)", what, v));
    if (!v[16 + cause_bit]) fail($sformatf("%s: cause bit %0d clear (STATUS %h)", what, 16 + cause_bit, v));
    if (acc_on) fail($sformatf("%s: accelerator still on", what));
    repeat (OFF + 4) @(negedge clk);
    if (dimm_on != 0) fail($sformatf("%s: DIMMs %b still on after OFF_DLY", what, dimm_on));
  endtask

  initial begin
    repeat (5) @(negedge clk);
    rst = 1'b0;
    repeat (5) @(negedge clk);
    rd(ID, v);     expect_eq("ID", v, 32'h474F_5631);
    rd(LIMITS, v); expect_eq("LIMITS", v, {6'd0, 10'd704, 6'd0, 10'd724});
    rd(CTRL, v);   expect_eq("CTRL at reset (UART_SEL 1)", v, 32'h100);
    rd(TRIP, v);   expect_eq("TRIP at reset", v, 724);
    rd(RESUME, v); expect_eq("RESUME at reset", v, 674);
    rd(HBMS, v);   expect_eq("HB_MS at reset", v, 2000);
    rd(STAT, v);   expect_eq("STATUS at reset (no reading, no heartbeat; HAS_REG, REG_OK, REG_COOL)", v, 32'h6080);
    rd(RLIM, v);   expect_eq("REG_LIMITS", v, {16'd300, 16'd312});
    rd(RTRIP, v);  expect_eq("REG_TRIP at reset", v, 312);
    rd(RRES, v);   expect_eq("REG_RESUME at reset", v, 300);
    rd(RSTALE, v); expect_eq("REG_STALE at reset", v, 10000);
    rd(RAGE, v);   expect_eq("REG_AGE", v, 5);
    rd(REG, v);    expect_eq("REG (now, the highest since reset)", v, {16'd200, 16'd200});
    if (acc_on || dimm_on != 0) fail("a domain on after reset");
    if (uart_sel != 2'd1) fail("UART_SEL at reset");

    // GO's conditions: no heartbeat yet (a valid, cool reading)
    wr(CTRL, 32'h01F);
    temp = 10'd660;                                          // 48 C
    repeat (4) @(negedge clk);
    wr(CMD, GO);
    repeat (3) @(negedge clk);
    rd(STAT, v);   if (v[0]) fail("GO without a heartbeat");
    // no reading
    temp = 10'd0;
    repeat (4) @(negedge clk);
    go_expect(0, "GO with no temperature reading");
    // too warm: above RESUME (674)
    temp = 10'd700;
    repeat (4) @(negedge clk);
    go_expect(0, "GO above RESUME");
    temp = 10'd674;
    repeat (4) @(negedge clk);
    temp_alarm = 1'b1;
    repeat (4) @(negedge clk);
    go_expect(0, "GO with the temperature alarm");
    temp_alarm = 1'b0;
    ot_alarm = 1'b1;
    repeat (4) @(negedge clk);
    go_expect(0, "GO with the OT alarm");
    ot_alarm = 1'b0;
    repeat (4) @(negedge clk);
    go_expect(1, "GO at RESUME, a heartbeat, no alarm");
    rd(STAT, v);   expect_eq("STATUS in RUN", v & 32'h1F1F, 32'h1F1D);  // acc, DIMMs, valid, hb, cool, run
    if (!acc_on || dimm_on != 4'hF || dimm_ok != 4'hF) fail("RUN: the domains not on");

    // the host's SAFE: the accelerator at once, the DIMMs OFF cycles later
    wr(CMD, SAFE);
    @(negedge clk);
    if (dimm_ok != 0 || dimm_on != 4'hF) fail($sformatf("SAFE: dimm_ok %b, dimm_on %b (expected 0, f)", dimm_ok, dimm_on));
    expect_stop(3, "host SAFE");
    if (off_gap < OFF - 2 || off_gap > OFF + 3) fail($sformatf("DIMMs off %0d cycles after the accelerator, expected ~%0d", off_gap, OFF));

    // CTRL during the delay turns no DIMM on; a partial set runs only those
    go_expect(1, "GO again");
    wr(CTRL, 32'h003);
    repeat (2) @(negedge clk);
    if (dimm_on != 4'h3 || acc_on) fail($sformatf("CTRL 3 in RUN: DIMMs %b acc %0d", dimm_on, acc_on));
    wr(CMD, SAFE);
    wr(CTRL, 32'h00F);                                      // within OFF_DLY of SAFE
    repeat (OFF + 4) @(negedge clk);
    if (dimm_on != 0) fail("a DIMM on after SAFE (CTRL written in the delay)");
    wr(CTRL, 32'h01F);

    // the die at TRIP
    wr(CMD, CLEAR);
    go_expect(1, "GO for the TRIP case");
    temp = 10'd714;                                          // TRIP - 10: WARN, still RUN
    repeat (4) @(negedge clk);
    rd(STAT, v);   if (!v[0] || !v[1]) fail($sformatf("WARN at TRIP - 10 (STATUS %h)", v));
    temp = 10'd724;
    expect_stop(0, "die at TRIP");
    rd(STAT, v);   if (!v[31]) fail("thermal trip bit31");
    rd(TRIPS, v);  expect_eq("TRIPS after one thermal", v, 32'h0000_0001);
    rd(TEMP, v);   expect_eq("TEMP (now, highest)", v, {6'd0, 10'd724, 6'd0, 10'd724});
    // hot: GO refused until RESUME
    temp = 10'd690;
    repeat (4) @(negedge clk);
    go_expect(0, "GO between RESUME and TRIP");
    temp = 10'd670;
    repeat (4) @(negedge clk);

    // the SYSMON's temperature alarm, its OT alarm, a code out of range
    go_expect(1, "GO for the alarm case");
    temp_alarm = 1'b1;
    expect_stop(1, "temperature alarm");
    temp_alarm = 1'b0;
    repeat (4) @(negedge clk);
    go_expect(1, "GO for the OT case");
    ot_alarm = 1'b1;
    expect_stop(4, "OT alarm");
    ot_alarm = 1'b0;
    repeat (4) @(negedge clk);
    go_expect(1, "GO for the no-reading case");
    temp = 10'd0;
    expect_stop(4, "no reading");
    temp = 10'd670;
    repeat (4) @(negedge clk);
    rd(TRIPS, v);  expect_eq("TRIPS after four thermal", v, 32'h0000_0004);

    // the heartbeat: HB_MS clamped (5 -> 100), RUN kept by kicks, ended without them
    wr(HBMS, 5);
    rd(HBMS, v);   expect_eq("HB_MS clamped up", v, 100);
    wr(HBMS, 50000);
    rd(HBMS, v);   expect_eq("HB_MS clamped down", v, 10000);
    wr(HBMS, 100);
    go_expect(1, "GO for the heartbeat case");
    for (int i = 0; i < 5; i++) begin ms(50); wr(KICK, 0); end
    rd(STAT, v);   if (!v[0]) fail("RUN ended with kicks every 50 ms");
    rd(RUNMS, v);  if (v < 240 || v > 260) fail($sformatf("RUN_MS %0d, expected ~250", v));
    rd(KICK, v);   if (v > 2) fail($sformatf("ms since the kick %0d", v));
    ms(110);
    expect_stop(2, "no heartbeat");
    rd(TRIPS, v);  expect_eq("TRIPS after a heartbeat trip", v, 32'h0001_0004);

    // TRIP and RESUME within their limits, a lowered TRIP trips lower
    wr(TRIP, 900);
    rd(TRIP, v);   expect_eq("TRIP clamped", v, 724);
    wr(RESUME, 800);
    rd(RESUME, v); expect_eq("RESUME clamped", v, 704);
    wr(RESUME, 674);
    wr(TRIP, 700);
    go_expect(1, "GO with TRIP 700");
    temp = 10'd700;
    expect_stop(0, "die at a lowered TRIP");

    // the regulator: GO above REG_RESUME (300, 75 C) and with a stale reading refused; WARN at
    // REG_TRIP - 8 (76 C); RUN ended at REG_TRIP (312) and by a reading REG_STALE old
    temp = 10'd660;
    wr(TRIP, 724);
    wr(HBMS, 2000);
    repeat (4) @(negedge clk);
    reg_t = 16'sd301;
    repeat (2) @(negedge clk);
    rd(STAT, v);   if (v[14] || !v[13]) fail($sformatf("REG_COOL above REG_RESUME (STATUS %h)", v));
    go_expect(0, "GO above REG_RESUME");
    reg_t = 16'sd300;
    reg_age = 16'd10000;
    repeat (2) @(negedge clk);
    rd(STAT, v);   if (v[13] || v[14]) fail($sformatf("REG_OK with a reading REG_STALE old (STATUS %h)", v));
    go_expect(0, "GO with a stale regulator reading");
    reg_age = 16'd9999;
    go_expect(1, "GO at REG_RESUME, a reading in time");
    reg_t = 16'sd304;
    repeat (4) @(negedge clk);
    rd(STAT, v);   if (!v[0] || !v[15]) fail($sformatf("REG_WARN at REG_TRIP - 8 (STATUS %h)", v));
    if (!led[2]) fail("led[2] without REG_WARN");
    reg_t = 16'sd312;
    expect_stop(5, "regulator at REG_TRIP");
    rd(STAT, v);   if (!v[31]) fail("thermal bit31 for the regulator");
    rd(TRIPS, v);  expect_eq("TRIPS after the regulator's trip", v, 32'h0001_0006);
    reg_t = 16'sd280;
    go_expect(1, "GO for the stale case");
    reg_age = 16'd10000;
    expect_stop(6, "regulator unread");
    rd(TRIPS, v);  expect_eq("TRIPS after the stale reading", v, 32'h0001_0007);
    reg_age = 16'd20;
    // the limits: REG_TRIP and REG_RESUME clamped at the build's, REG_STALE within 1000..30000;
    // a lowered REG_TRIP and REG_STALE trip lower
    wr(RTRIP, 400);
    rd(RTRIP, v);  expect_eq("REG_TRIP clamped", v, 312);
    wr(RRES, 350);
    rd(RRES, v);   expect_eq("REG_RESUME clamped", v, 300);
    wr(RSTALE, 10);
    rd(RSTALE, v); expect_eq("REG_STALE clamped up", v, 1000);
    wr(RSTALE, 50000);
    rd(RSTALE, v); expect_eq("REG_STALE clamped down", v, 30000);
    wr(RSTALE, 2000);
    wr(RTRIP, 290);
    go_expect(1, "GO with REG_TRIP 290");
    reg_t = 16'sd290;
    expect_stop(5, "regulator at a lowered REG_TRIP");
    reg_t = 16'sd280;
    go_expect(1, "GO with REG_STALE 2000");
    reg_age = 16'd2000;
    expect_stop(6, "regulator unread for a lowered REG_STALE");
    reg_age = 16'd20;
    rd(REG, v);    expect_eq("REG (now, highest)", v, {16'd312, 16'd280});

    // the code's age: from each change (TEMP_AGE), held while the temperature alarm stands
    rd(TSTALE, v); expect_eq("TEMP_STALE", v, STALE);
    wr(CMD, CLEAR);
    temp = 10'd661;
    repeat (4) @(negedge clk);
    rd(TAGE, v);   if (v > 1) fail($sformatf("TEMP_AGE %h just after a change", v));
    ms(100);
    rd(TAGE, v);   if (v[16:0] < 99 || v[16:0] > 101 || v[31]) fail($sformatf("TEMP_AGE %h after 100 ms", v));
    temp_alarm = 1'b1;
    ms(300);
    rd(TAGE, v);   if (v[16:0] < 99 || v[16:0] > 102) fail($sformatf("TEMP_AGE %h: not held under the alarm", v));
    temp_alarm = 1'b0;
    repeat (4) @(negedge clk);
    // a code held TEMP_STALE_MS: TEMP_VALID drops, RUN ends with cause 23 alone, GO is refused
    go_expect(1, "GO with a fresh code");
    ms(STALE - 200);
    wr(KICK, 0);
    rd(STAT, v);   if (!v[0] || !v[4]) fail($sformatf("RUN and TEMP_VALID before TEMP_STALE_MS (STATUS %h)", v));
    ms(150);
    expect_stop(7, "a code held TEMP_STALE_MS");
    rd(STAT, v);   if (v[4] || !v[31] || v[20]) fail($sformatf("stale: TEMP_VALID, bit31, bit20 (STATUS %h)", v));
    rd(TAGE, v);   if (!v[31] || v[16:0] < STALE) fail($sformatf("TEMP_AGE %h after TEMP_STALE_MS", v));
    rd(TRIPS, v);  expect_eq("TRIPS after the stale code", v, 32'h0001_000A);
    go_expect(0, "GO with a stale code");
    temp = 10'd662;
    repeat (4) @(negedge clk);
    rd(TAGE, v);   if (v > 1) fail($sformatf("TEMP_AGE %h after the code moved", v));
    rd(STAT, v);   if (!v[4]) fail($sformatf("TEMP_VALID once the code moved (STATUS %h)", v));
    // a code that changes within each TEMP_STALE_MS keeps RUN
    go_expect(1, "GO once the code moved");
    for (int i = 0; i < 3; i++) begin
      ms(STALE * 6 / 10);
      temp = (i % 2 == 0) ? 10'd663 : 10'd662;
      wr(KICK, 0);
    end
    rd(STAT, v);   if (!v[0] || !v[4]) fail($sformatf("RUN with a code that moves (STATUS %h)", v));
    wr(CMD, SAFE);

    // CLEAR: the causes and the highest code; UART_SEL; AW before W
    temp = 10'd660;
    repeat (4) @(negedge clk);
    wr(CMD, CLEAR, 1);
    rd(STAT, v);   expect_eq("causes after CLEAR", v & 32'h80FF_0000, 0);
    rd(REG, v);    expect_eq("REG after CLEAR", v, {16'd280, 16'd280});
    rd(TEMP, v);   expect_eq("TEMP after CLEAR", v, {6'd0, 10'd660, 6'd0, 10'd660});
    wr(CTRL, 32'h31F);
    if (uart_sel != 2'd3) fail("UART_SEL 3");

    repeat (10) @(negedge clk);
    if (errs == 0) $display("PASS");
    else $display("FAIL (%0d errors)", errs);
    $finish;
  end
endmodule
