// Replays one channel's native command trace (otpu_native_mem's +nat_trace lines: cycle, channel,
// we, beat; +trace=FILE, +ch=C) through one channel's LiteDRAM controller (otpu_ldc_ch.v, from
// tools/litedram/gen_ldc.py) on its first user port, the second idle (the one-port core's
// behaviour), as fast as it takes them, or (+paced=1) each no earlier than its traced cycle
// scaled by +scale=N/1000. Write data is always offered, read data always taken. Prints the
// controller cycles from the first command to the last beat moved, and the bank / row changes of
// the trace. Built with rtlsim.build("tb_ldc_replay", [otpu_ldc_ch.v, this file]).
module tb_ldc_replay;
  logic clk = 1'b0, rst = 1'b1;
  always #5 clk = ~clk;

  logic cv, cr, cwe, wv, wr, rv;
  logic [24:0] ca;
  otpu_ldc_ch u_dut (                  // the trace on the first user port, the second idle
    .sys_clk(clk), .sys_rst(rst),
    .p0_cmd_valid(cv), .p0_cmd_ready(cr), .p0_cmd_we(cwe), .p0_cmd_addr(ca),
    .p0_wdata_valid(wv), .p0_wdata_ready(wr), .p0_wdata_data('0), .p0_wdata_we('1),
    .p0_rdata_valid(rv), .p0_rdata_ready(1'b1), .p0_rdata_data(),
    .p1_cmd_valid(1'b0), .p1_cmd_ready(), .p1_cmd_we(1'b0), .p1_cmd_addr('0),
    .p1_wdata_valid(1'b0), .p1_wdata_ready(), .p1_wdata_data('0), .p1_wdata_we('1),
    .p1_rdata_valid(), .p1_rdata_ready(1'b1), .p1_rdata_data());

  longint tcyc [$];
  logic [24:0] taddr [$];
  bit twe [$];
  int ch = 0, paced = 0, scale = 1000;
  longint nrd = 0, nwr = 0, grd = 0, gwr = 0, idx = 0, cyc = 0, start = -1, wcmd = 0;
  longint banksw = 0, rowsw = 0;
  string fn;
  initial begin
    integer fd, n;
    longint t;
    int c, we, m;
    int pb = -1, pr = -1;
    if (!$value$plusargs("trace=%s", fn)) $fatal(1, "+trace= missing");
    void'($value$plusargs("ch=%d", ch));
    void'($value$plusargs("paced=%d", paced));
    void'($value$plusargs("scale=%d", scale));
    fd = $fopen(fn, "r");
    if (fd == 0) $fatal(1, "no trace");
    while (!$feof(fd)) begin
      n = $fscanf(fd, "%d %d %d %d\n", t, c, we, m);
      if (n != 4) break;
      if (c != ch) continue;
      tcyc.push_back(t);
      taddr.push_back(25'(m));
      twe.push_back(we != 0);
      if (we != 0) gwr++; else grd++;
      if ((m >> 7) % 8 != pb) banksw++;
      if ((m >> 7) != pr) rowsw++;
      pb = (m >> 7) % 8;
      pr = m >> 7;
    end
    $fclose(fd);
    $display("REPLAY trace ch%0d: %0d commands (%0d reads, %0d writes), %0d bank changes, %0d row changes",
             ch, tcyc.size(), grd, gwr, banksw, rowsw);
    repeat (8) @(posedge clk);
    rst = 1'b0;
  end

  // the next command, offered once its (scaled) traced cycle has come when paced
  always_comb begin
    cv = !rst && idx < tcyc.size() &&
         (paced == 0 || (tcyc[idx] - tcyc[0]) * scale / 1000 <= cyc - start);
    ca = idx < tcyc.size() ? taddr[idx] : '0;
    cwe = idx < tcyc.size() ? twe[idx] : 1'b0;
    // write data offered from its command on (a write command goes with its data valid)
    wv = !rst && nwr < wcmd + ((cv && cwe) ? 1 : 0);
  end
  always @(posedge clk) begin
    if (!rst) begin
      if (start < 0 && idx == 0) start <= cyc;
      cyc <= cyc + 1;
      if (cv && cr) idx <= idx + 1;
      if (cv && cr && cwe) wcmd <= wcmd + 1;
      if (rv) nrd <= nrd + 1;
      if (wv && wr) nwr <= nwr + 1;
      if (tcyc.size() != 0 && idx == tcyc.size() && nrd == grd && nwr >= gwr) begin
        $display("REPLAY ch%0d cycles=%0d reads=%0d writes=%0d beats/cycle=%.4f",
                 ch, cyc - start, nrd, nwr, real'(grd + gwr) / real'(cyc - start));
        $finish;
      end
      if (cyc > 64'd1 << 34) $fatal(1, "replay did not finish");
    end
  end
endmodule
