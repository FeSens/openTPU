// The fixed X2 window in otpu_ctrl: the card register, progress watchdog and sequence,
// 16-cycle link reset independent of RUN, every link CSR passed through, reserved words zero,
// and COLL_STATUS reflected in STATUS ERROR. A combinational CSR stand-in returns its address;
// concurrent read/write traffic checks that the read mux never observes the write's address.
// NC 1 checks the old capability word and the disabled window. No accelerator is needed here.
module tb_x2_ctrl;
  parameter int NC = 2;
  logic clk = 1'b0, rst = 1'b1;
  always #5 clk = ~clk;
  logic [11:0] awaddr = '0, araddr = '0;
  logic awvalid = 0, wvalid = 0, arvalid = 0, awready, wready, arready;
  logic [31:0] wdata = '0, rdata;
  logic bvalid, rvalid, run, lrst, c_we;
  logic [7:0] card, ncards, c_addr;
  logic [31:0] wd, c_wd, c_rd;
  logic [7:0] status = '0;
  logic [15:0] seq = 16'h1234;
  int nwrites = 0, nreset = 0, checks = 0;
  assign c_rd = 32'hABCD0000 | {24'd0, c_addr};
  always @(posedge clk) begin
    if (c_we) begin
      if (c_addr != awaddr[7:0] || c_wd != wdata) $fatal(1, "CSR write mismatch");
      nwrites++;
    end
    if (lrst) nreset++;
  end
  otpu_ctrl #(.NC(NC), .TRACE_DEPTH(0), .PLATFORM(7)) dut (
    .clk, .rst, .s_awaddr(awaddr), .s_awvalid(awvalid), .s_awready(awready),
    .s_wdata(wdata), .s_wstrb(4'hF), .s_wvalid(wvalid), .s_wready(wready),
    .s_bresp(), .s_bvalid(bvalid), .s_bready(1'b1),
    .s_araddr(araddr), .s_arvalid(arvalid), .s_arready(arready),
    .s_rdata(rdata), .s_rresp(), .s_rvalid(rvalid), .s_rready(1'b1),
    .run, .ld_start(), .arg(), .ld_addr(), .ld_n(), .ld_busy(1'b0), .halted(1'b0),
    .error(1'b0), .wait_to(1'b0), .icount('0), .wr_idle(1'b1), .axi_err(1'b0),
    .calib(2'b11), .ecc_ded(1'b0), .b_rd(1'b0), .b_wr(1'b0), .a_rd(1'b0),
    .a_wr(1'b0), .b_wait(1'b0), .temp_v(1'b0), .temp('0), .mxu_busy(1'b0),
    .mxu_mac(1'b0), .mxu_starve(1'b0), .vpu_busy(1'b0), .qnt_busy(1'b0),
    .dma_busy(1'b0), .tmem_deny(1'b0), .dram_rd('0), .dram_wr('0), .dram_wait(1'b0),
    .instr('0), .tr_en(), .tr_stop(), .tr_clear(), .tr_addr(), .tr_count('0),
    .tr_drop('0), .tr_busy(1'b0), .tr_rdata('0), .i2c_lo(), .i2c_in('0),
    .x2_card(card), .x2_ncards(ncards), .x2_lrst(lrst), .coll_wd(wd),
    .coll_status(status), .x2_seq(seq), .x2_c_we(c_we), .x2_c_addr(c_addr),
    .x2_c_wd(c_wd), .x2_c_rd(c_rd));
  task automatic write_reg(input logic [11:0] a, input logic [31:0] d);
    @(negedge clk); awaddr = a; wdata = d; awvalid = 1; wvalid = 1;
    do @(posedge clk); while (!awready);
    @(negedge clk); awvalid = 0; wvalid = 0;
    do @(posedge clk); while (!bvalid);
  endtask
  task automatic read_reg(input logic [11:0] a, output logic [31:0] d);
    @(negedge clk); araddr = a; arvalid = 1;
    do @(posedge clk); while (!arready);
    @(negedge clk); arvalid = 0;
    do @(posedge clk); while (!rvalid);
    d = rdata;
  endtask
  task automatic check_reg(input logic [11:0] a, input logic [31:0] want);
    logic [31:0] d;
    read_reg(a, d);
    if (d != want) $fatal(1, "reg %h = %h, want %h", a, d, want);
    checks++;
  endtask
  initial begin
    repeat (10) @(negedge clk); rst = 0;
    check_reg(12'h05C, 32'h1 | (NC > 1 ? 32'h8000 : 0));
    check_reg(12'h098, 7);
    check_reg(12'h300, NC > 1 ? 32'h100 : 0);
    check_reg(12'h364, NC > 1 ? 32'd1 << 26 : 0);
    check_reg(12'h30C, NC > 1 ? 32'h1234 : 0);
    write_reg(12'h300, 32'h201);
    check_reg(12'h300, NC > 1 ? 32'h201 : 0);
    write_reg(12'h364, 1234);
    check_reg(12'h364, NC > 1 ? 1234 : 0);
    for (int a = 'h308; a < 'h360; a += 4) if (a != 'h30C) begin
      check_reg(12'(a), NC > 1 ? 32'hABCD0000 | 32'(a & 255) : 0);
      write_reg(12'(a), 32'(a));
    end
    check_reg(12'h368, 0);
    check_reg(12'h3FC, 0);
    write_reg(12'h304, 15);
    check_reg(12'h304, 0);
    repeat (20) @(posedge clk);
    if (nreset != (NC > 1 ? 16 : 0)) $fatal(1, "LRST cycles %0d", nreset);
    if (nwrites != (NC > 1 ? 22 : 0)) $fatal(1, "CSR writes %0d", nwrites);
    status = 8'h63;
    check_reg(12'h360, NC > 1 ? 32'h63 : 0);
    check_reg(12'h00C, NC > 1 ? 32'h6A : 32'h68);
    write_reg(12'h008, 1);
    write_reg(12'h008, 0);
    check_reg(12'h300, NC > 1 ? 32'h201 : 0);
    check_reg(12'h364, NC > 1 ? 1234 : 0);
    // Concurrent channels, with the write arriving one cycle after AR: r_p[0] is then high.
    fork
      check_reg(12'h318, NC > 1 ? 32'hABCD0018 : 0);
      begin @(negedge clk); write_reg(12'h324, 99); end
    join
    $display("TBXCTRL PASS NC=%0d checks=%0d", NC, checks);
    $finish;
  end
  initial begin #100000; $fatal(1, "control test hangs"); end
endmodule
