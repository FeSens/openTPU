// Unit test of otpu_axil_off (rtl/boards/bcu1525/otpu_axil_off.sv; tests/test_bcu_gov.py) in front
// of otpu_axil_iso and a model of a LiteDRAM core's CSR port: it takes every access and answers 4
// cycles later, unless its clock is stopped (`dead`: taken, never answered), and forgets what it
// had in reset. As the BCU1525 tops wire them: the iso's s_rst and the gate's off rise together
// when the governor turns the DIMM off. Checked: accesses pass while on; while off every access is
// answered SLVERR (0xDEAD0FF0) at once; one taken by the core when it dies is answered SLVERR by
// the iso; accesses pass again when it is back on. Prints PASS, or the errors and FAIL.
module tb_axil_off;
  logic clk = 1'b0;
  always #5 clk = !clk;
  logic rst = 1'b1, off = 1'b1, dead = 1'b1;

  logic [15:0] awaddr = '0, araddr = '0;
  logic [31:0] wdata = '0, rdata;
  logic awvalid = 1'b0, wvalid = 1'b0, bready = 1'b1, arvalid = 1'b0, rready = 1'b1;
  logic awready, wready, bvalid, arready, rvalid;
  logic [1:0] bresp, rresp;
  // gate -> iso
  logic [15:0] g_awaddr, g_araddr;
  logic [31:0] g_wdata, g_rdata;
  logic [3:0]  g_wstrb;
  logic [1:0]  g_bresp, g_rresp;
  logic g_awvalid, g_awready, g_wvalid, g_wready, g_bvalid, g_bready;
  logic g_arvalid, g_arready, g_rvalid, g_rready;
  // iso -> core model
  logic [15:0] c_awaddr, c_araddr;
  logic [31:0] c_wdata, c_rdata;
  logic [3:0]  c_wstrb;
  logic [1:0]  c_bresp, c_rresp;
  logic c_awvalid, c_awready, c_wvalid, c_wready, c_bvalid, c_bready;
  logic c_arvalid, c_arready, c_rvalid, c_rready;

  otpu_axil_off #(.AW(16)) u_off (
    .clk, .rst, .off,
    .m_awaddr(awaddr), .m_awvalid(awvalid), .m_awready(awready), .m_wdata(wdata),
    .m_wstrb(4'hf), .m_wvalid(wvalid), .m_wready(wready), .m_bresp(bresp), .m_bvalid(bvalid),
    .m_bready(bready), .m_araddr(araddr), .m_arvalid(arvalid), .m_arready(arready),
    .m_rdata(rdata), .m_rresp(rresp), .m_rvalid(rvalid), .m_rready(rready),
    .s_awaddr(g_awaddr), .s_awvalid(g_awvalid), .s_awready(g_awready), .s_wdata(g_wdata),
    .s_wstrb(g_wstrb), .s_wvalid(g_wvalid), .s_wready(g_wready), .s_bresp(g_bresp),
    .s_bvalid(g_bvalid), .s_bready(g_bready), .s_araddr(g_araddr), .s_arvalid(g_arvalid),
    .s_arready(g_arready), .s_rdata(g_rdata), .s_rresp(g_rresp), .s_rvalid(g_rvalid),
    .s_rready(g_rready));
  otpu_axil_iso #(.AW(16)) u_iso (
    .clk, .m_rst(rst), .s_rst(off),
    .m_awaddr(g_awaddr), .m_awvalid(g_awvalid), .m_awready(g_awready), .m_wdata(g_wdata),
    .m_wstrb(g_wstrb), .m_wvalid(g_wvalid), .m_wready(g_wready), .m_bresp(g_bresp),
    .m_bvalid(g_bvalid), .m_bready(g_bready), .m_araddr(g_araddr), .m_arvalid(g_arvalid),
    .m_arready(g_arready), .m_rdata(g_rdata), .m_rresp(g_rresp), .m_rvalid(g_rvalid),
    .m_rready(g_rready),
    .s_awaddr(c_awaddr), .s_awvalid(c_awvalid), .s_awready(c_awready), .s_wdata(c_wdata),
    .s_wstrb(c_wstrb), .s_wvalid(c_wvalid), .s_wready(c_wready), .s_bresp(c_bresp),
    .s_bvalid(c_bvalid), .s_bready(c_bready), .s_araddr(c_araddr), .s_arvalid(c_arvalid),
    .s_arready(c_arready), .s_rdata(c_rdata), .s_rresp(c_rresp), .s_rvalid(c_rvalid),
    .s_rready(c_rready));

  // the core's CSR port: one register file word per address, an access answered 4 cycles after
  // it is taken, never while dead; reset (off) forgets the access
  logic [31:0] regs [16];
  logic pend = 1'b0, pend_w = 1'b0;
  logic [3:0] pend_a;
  int wait_n = 0;
  assign c_awready = !pend && c_awvalid && c_wvalid;
  assign c_wready  = c_awready;
  assign c_arready = !pend && !(c_awvalid && c_wvalid);
  always_ff @(posedge clk) begin
    if (c_awvalid && c_awready) begin
      pend <= 1'b1; pend_w <= 1'b1; pend_a <= c_awaddr[5:2]; wait_n <= 4;
      regs[c_awaddr[5:2]] <= c_wdata;
    end else if (c_arvalid && c_arready) begin
      pend <= 1'b1; pend_w <= 1'b0; pend_a <= c_araddr[5:2]; wait_n <= 4;
    end
    if (pend && !dead && wait_n > 0) wait_n <= wait_n - 1;
    if (pend && !dead && wait_n == 1) begin
      if (pend_w) c_bvalid <= 1'b1; else begin c_rvalid <= 1'b1; c_rdata <= regs[pend_a]; end
    end
    if (c_bvalid && c_bready) begin c_bvalid <= 1'b0; pend <= 1'b0; end
    if (c_rvalid && c_rready) begin c_rvalid <= 1'b0; pend <= 1'b0; end
    if (off) begin pend <= 1'b0; c_bvalid <= 1'b0; c_rvalid <= 1'b0; end
  end
  assign c_bresp = 2'b00;
  assign c_rresp = 2'b00;

  int errs = 0;
  task automatic fail(input string s);
    errs++;
    if (errs <= 20) $display("ERROR at %0t: %s", $time, s);
  endtask

  // the master: one access, its response within `limit` cycles (else an error)
  task automatic wr(input logic [15:0] a, input logic [31:0] d, input logic [1:0] want, input string what);
    int n = 0;
    @(negedge clk);
    awaddr = a; wdata = d; awvalid = 1'b1; wvalid = 1'b1;
    while (!(awready && wready)) begin @(negedge clk); if (++n > 200) begin fail({what, ": AW never taken"}); return; end end
    @(negedge clk);
    awvalid = 1'b0; wvalid = 1'b0;
    while (!bvalid) begin @(negedge clk); if (++n > 200) begin fail({what, ": no B"}); return; end end
    if (bresp != want) fail($sformatf("%s: BRESP %0d, expected %0d", what, bresp, want));
  endtask
  task automatic rd(input logic [15:0] a, input logic [1:0] want, input logic [31:0] wd, input string what,
                   input bit chk = 1);
    int n = 0;
    @(negedge clk);
    araddr = a; arvalid = 1'b1;
    while (!arready) begin @(negedge clk); if (++n > 200) begin fail({what, ": AR never taken"}); return; end end
    @(negedge clk);
    arvalid = 1'b0;
    while (!rvalid) begin @(negedge clk); if (++n > 200) begin fail({what, ": no R"}); return; end end
    if (rresp != want || (chk && rdata != wd))
      fail($sformatf("%s: RRESP %0d data %h, expected %0d %h", what, rresp, rdata, want, wd));
  endtask

  initial begin
    repeat (4) @(negedge clk);
    rst = 1'b0;
    repeat (4) @(negedge clk);
    // off after reset: answered here
    rd(16'h0004, 2'b10, 32'hDEAD_0FF0, "read while off");
    wr(16'h0004, 32'h1, 2'b10, "write while off");
    // on: through to the core
    dead = 1'b0; off = 1'b0;
    repeat (4) @(negedge clk);
    wr(16'h0008, 32'hCAFE_0001, 2'b00, "write while on");
    rd(16'h0008, 2'b00, 32'hCAFE_0001, "read while on");
    // the core dies with a read in flight: the iso answers it SLVERR when off rises
    fork
      rd(16'h0008, 2'b10, 0, "read in flight as the core dies", 0);   // the iso's SLVERR: any data
      begin
        @(posedge c_arvalid);
        dead = 1'b1;
        repeat (20) @(negedge clk);
        off = 1'b1;
      end
    join
    // ...then this side answers
    rd(16'h000C, 2'b10, 32'hDEAD_0FF0, "read after the core died");
    wr(16'h000C, 32'h2, 2'b10, "write after the core died");
    // back on: through again (the model forgot nothing it was written before)
    dead = 1'b0; off = 1'b0;
    repeat (4) @(negedge clk);
    rd(16'h0008, 2'b00, 32'hCAFE_0001, "read after it is back on");
    // off rising while the master holds a write's valids before they are taken (a long access
    // ahead of it): taken here, answered SLVERR
    fork
      rd(16'h0008, 2'b00, 32'hCAFE_0001, "read ahead of the withdrawn write");
      begin
        @(negedge clk);
        @(negedge clk);
        wr(16'h0004, 32'h3, 2'b10, "write withdrawn from the core");
      end
      begin
        @(posedge rvalid);                  // the read's response with the master: off now
        @(negedge clk);
        off = 1'b1;
      end
    join
    repeat (4) @(negedge clk);
    if (errs == 0) $display("PASS");
    else $display("FAIL (%0d errors)", errs);
    $finish;
  end
endmodule
