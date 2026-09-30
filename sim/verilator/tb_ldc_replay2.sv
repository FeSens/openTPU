// tb_ldc_replay with two user ports: the trace's commands go, in order, into one queue per port
// (+qd=N entries each) chosen by the channel beat's bit +sel=B (7: the bank's low bit, one 8 KiB
// row per port in turn) and each queue drains into its port on its own (a command to one port
// never waits behind the other's). Same address, same port: no hazard crosses ports. Prints the
// cycles from the first command to the last beat moved, and the largest read skew between the
// ports (the read data an adapter would have to hold to return it in command order). Needs the
// controller with two user ports: tools/litedram/gen_ldc.py --ports 2 OUT.v.
module tb_ldc_replay2;
  logic clk = 1'b0, rst = 1'b1;
  always #5 clk = ~clk;

  logic [1:0] cv, cr, cwe, wv, wr, rv;
  logic [1:0][24:0] ca;
  otpu_ldc_ch u_dut (
    .sys_clk(clk), .sys_rst(rst),
    .p0_cmd_valid(cv[0]), .p0_cmd_ready(cr[0]), .p0_cmd_we(cwe[0]), .p0_cmd_addr(ca[0]),
    .p0_wdata_valid(wv[0]), .p0_wdata_ready(wr[0]), .p0_wdata_data('0), .p0_wdata_we('1),
    .p0_rdata_valid(rv[0]), .p0_rdata_ready(1'b1), .p0_rdata_data(),
    .p1_cmd_valid(cv[1]), .p1_cmd_ready(cr[1]), .p1_cmd_we(cwe[1]), .p1_cmd_addr(ca[1]),
    .p1_wdata_valid(wv[1]), .p1_wdata_ready(wr[1]), .p1_wdata_data('0), .p1_wdata_we('1),
    .p1_rdata_valid(rv[1]), .p1_rdata_ready(1'b1), .p1_rdata_data());

  logic [24:0] taddr [$];
  bit twe [$];
  int ch = 0, qd = 16, sel = 7;
  longint grd = 0, gwr = 0, idx = 0, cyc = 0, start = -1;
  longint nrd [2], nwr [2], wcmd [2], rcmd [2];
  longint qn [2];                        // queued, not yet taken by the port
  logic [24:0] qa [2][$];
  bit qw [2][$];
  longint skew = 0;
  bit rport [$];                         // the port of every read, in command order
  longint held [2], maxheld [2];         // returned reads waiting for an earlier one (other port)
  string fn;
  initial begin
    integer fd, n;
    longint t;
    int c, we, m;
    if (!$value$plusargs("trace=%s", fn)) $fatal(1, "+trace= missing");
    void'($value$plusargs("ch=%d", ch));
    void'($value$plusargs("qd=%d", qd));
    void'($value$plusargs("sel=%d", sel));
    fd = $fopen(fn, "r");
    if (fd == 0) $fatal(1, "no trace");
    while (!$feof(fd)) begin
      n = $fscanf(fd, "%d %d %d %d\n", t, c, we, m);
      if (n != 4) break;
      if (c != ch) continue;
      taddr.push_back(25'(m));
      twe.push_back(we != 0);
      if (we != 0) gwr++; else grd++;
    end
    $fclose(fd);
    nrd = '{0, 0}; nwr = '{0, 0}; wcmd = '{0, 0}; rcmd = '{0, 0};
    held = '{0, 0}; maxheld = '{0, 0};
    $display("REPLAY2 trace ch%0d: %0d commands, split on bit %0d, queues of %0d", ch,
             taddr.size(), sel, qd);
    repeat (8) @(posedge clk);
    rst = 1'b0;
  end

  always_comb
    for (int p = 0; p < 2; p++) begin
      cv[p] = !rst && qa[p].size() != 0;
      ca[p] = qa[p].size() != 0 ? qa[p][0] : '0;
      cwe[p] = qa[p].size() != 0 ? qw[p][0] : 1'b0;
      wv[p] = !rst && nwr[p] < wcmd[p] + ((cv[p] && cwe[p]) ? 1 : 0);
    end

  always @(posedge clk) begin
    if (!rst) begin
      int p;
      if (start < 0) start <= cyc;
      cyc <= cyc + 1;
      for (int q = 0; q < 2; q++) begin
        if (cv[q] && cr[q]) begin
          if (cwe[q]) wcmd[q] <= wcmd[q] + 1; else rcmd[q] <= rcmd[q] + 1;
          void'(qa[q].pop_front()); void'(qw[q].pop_front());
        end
        if (rv[q]) nrd[q] <= nrd[q] + 1;
        if (wv[q] && wr[q]) nwr[q] <= nwr[q] + 1;
      end
      // in order: the next trace command into its port's queue, if there is room
      if (idx < taddr.size()) begin
        p = int'(taddr[idx][sel]);
        if (qa[p].size() < qd) begin
          if (!twe[idx]) rport.push_back(p[0]);
          qa[p].push_back(taddr[idx]);
          qw[p].push_back(twe[idx]);
          idx <= idx + 1;
        end
      end
      // in-order return: reads come back per port in order; the adapter releases the oldest
      // read once its port has returned it
      begin
        longint h [2];
        h = held;
        for (int q = 0; q < 2; q++) if (rv[q]) h[q]++;
        while (rport.size() != 0 && h[rport[0]] != 0) begin
          h[rport[0]]--;
          void'(rport.pop_front());
        end
        held <= h;
        for (int q = 0; q < 2; q++) if (h[q] > maxheld[q]) maxheld[q] <= h[q];
      end
      if (taddr.size() != 0 && idx == taddr.size() && qa[0].size() == 0 && qa[1].size() == 0 &&
          nrd[0] + nrd[1] == grd && nwr[0] + nwr[1] >= gwr) begin
        $display("REPLAY2 ch%0d cycles=%0d reads=%0d writes=%0d beats/cycle=%.4f held max %0d / %0d",
                 ch, cyc - start, nrd[0] + nrd[1], nwr[0] + nwr[1],
                 real'(grd + gwr) / real'(cyc - start), maxheld[0], maxheld[1]);
        $finish;
      end
      if (cyc > 64'd1 << 34) $fatal(1, "replay did not finish");
    end
  end
endmodule
