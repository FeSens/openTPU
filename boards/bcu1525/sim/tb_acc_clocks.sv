// The WG top's accelerator clocks (otpu_fpga_top_bcu_wg: the enable's register and the BUFGCE_DIV
// pair, /1 and /2 on one enable and one clear; tests/test_bcuwg_build.py copies that block out of
// the top as otpu_wg_acc_clocks) under random starts and stops of acc_ce, a core_clk register as
// the sequencer's (0.1 ns after the edge: a flip-flop's clock to out): runs and gaps of 1 to 24
// core cycles, one in four a single cycle. core_clk and the MMCM's 2x output rise together at every
// other 2x edge (one VCO, integer divides). In the middle of every 2x half period it checks both
// clocks against acc_ce's value before the last core edge (e): acc_clk == core_clk && e and
// acc_clk2x == the 2x output && e. So acc_clk rises only with core_clk, a 1x cycle holds two 2x
// edges, both clocks stop low, and each run starts and stops one core cycle after acc_ce. Each
// start raises the enable out of a stop with the clear high (counted: clr2x high at the sample
// before ce2x rises, single-cycle gaps included). +seed=N picks the run lengths; +start2x=N starts the 2x output N of its periods late
// (configuration's start-up: odd N at the 2x edge between two core edges), acc_ce low until it
// runs; with +bufgce_div_o1 (unisim_stubs.sv) the /2 starts at the other phase. Prints PASS or
// FAIL.
`timescale 1ns / 1ps
module tb_acc_clocks;
  localparam int NRUNS = 4000;
  logic clk2x_u = 1'b0, core_clk = 1'b0;
  int start2x;
  logic [4:0] left;
  initial begin                                                         // rises at 2 + 4m, m >= N
    if (!$value$plusargs("start2x=%d", start2x)) start2x = 0;
    left = 5'(3 + start2x);                  // (acc_ce low until the 2x output runs)
    #(4 * start2x);
    forever begin #2 clk2x_u = 1'b1; #2 clk2x_u = 1'b0; end
  end
  initial forever begin #2 core_clk = 1'b1; #4 core_clk = 1'b0; #2; end         // rises at 2 + 8n

  int unsigned seed;
  logic [31:0] x;
  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    x = 32'h9e3779b9 ^ seed;
    if (x == 0) x = 1;
  end
  function automatic logic [31:0] xs(input logic [31:0] v);   // xorshift32
    v ^= v << 13; v ^= v >> 17; v ^= v << 5;
    return v;
  endfunction

  logic acc_ce = 1'b0, e = 1'b0;
  int runs = 0, gap1 = 0, on1 = 0;
  logic [1:0] eh = '0;                       // e's last two values (single-cycle runs and gaps)
  always @(posedge core_clk) begin
    e <= acc_ce;                             // acc_ce before this edge
    if (left == 0) begin
      logic [31:0] y;
      y = xs(x);
      x <= y;
      acc_ce <= #0.1 !acc_ce;
      left <= y[1:0] == 0 ? 5'd0 : 5'(y[8:4] % 24);
      if (!acc_ce) runs <= runs + 1;
    end else left <= left - 1'b1;
    eh <= {eh[0], e};
    if (eh == 2'b10 && e) gap1 <= gap1 + 1;
    if (eh == 2'b01 && !e) on1 <= on1 + 1;
  end

  logic acc_clk, acc_clk2x;
  otpu_wg_acc_clocks dut (.acc_clk2x_u(clk2x_u), .acc_ce, .acc_clk, .acc_clk2x);

  int n1 = 0, n2 = 0, errs = 0, clr_on = 0;
  logic pc = 1'b1, pe = 1'b0;                // clr2x and ce2x at the last sample
  always @(posedge acc_clk) n1++;
  always @(posedge acc_clk2x) n2++;
  initial begin
    #1;
    forever begin                            // at 1 + 2k ns (the edges at even ns)
      if (acc_clk !== (core_clk && e) || acc_clk2x !== (clk2x_u && e)) begin
        if (errs < 10)
          $display("FAIL at %0t: acc_clk %b acc_clk2x %b, core_clk %b 2x %b e %b", $time, acc_clk,
                   acc_clk2x, core_clk, clk2x_u, e);
        errs++;
      end
      if (dut.ce2x && !pe && pc) clr_on++;
      pc = dut.clr2x;
      pe = dut.ce2x;
      if (runs >= NRUNS && !e && !acc_ce) begin
        if (errs == 0 && n2 == 2 * n1 && gap1 > 50 && on1 > 50 && clr_on == runs)
          $display("PASS runs %0d (single-cycle gaps %0d, runs %0d, starts with the clear high %0d), acc_clk rises %0d, acc_clk2x %0d, start2x %0d",
                   runs, gap1, on1, clr_on, n1, n2, start2x);
        else
          $display("FAIL errors %0d, acc_clk rises %0d, acc_clk2x %0d, single-cycle gaps %0d, runs %0d, starts with the clear high %0d of %0d",
                   errs, n1, n2, gap1, on1, clr_on, runs);
        $finish;
      end
      #2;
    end
  end
endmodule
