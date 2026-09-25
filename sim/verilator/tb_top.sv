// Simulation harness: memories load prog_<s>.hex / dram_<s>.bin from +dir=..., the machine runs
// until every slice halts, then TMEM and DRAM of every slice are dumped for comparison.
module tb_top;
  parameter int S          = 1;
  parameter int D          = 32;
  parameter int MCOLS      = 8;
  parameter int ACT_BLOCKS = 64;
  parameter int TMEM_WORDS = 1 << 16;
  parameter int IMEM_WORDS = 1 << 16;
  parameter int DRAM_WORDS = 1 << 18;
  parameter int DRAM_LAT   = 8;
  parameter int LANES      = 8;
  parameter int WIN        = 32;
  parameter int RPB        = 4;
  parameter int WPB        = 2;
  parameter int MXU_IMPL   = 0;
  parameter int MXU_CL     = 16;
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1;
  parameter int ULANES     = LANES;
  parameter int AXI        = 0;
  parameter int FIFO_DEPTH = 128;

  logic clk = 1'b0, sys_rst = 1'b1, rst = 1'b1, dump = 1'b0;
  logic ld_start = 1'b0, ld_busy;
  logic [31:0] ld_addr = '0, ld_n = '0;
  logic all_halted, any_error;
  logic [31:0] icount [S];
  longint cycles = 0, max_cycles = 50_000_000;

  otpu_top #(.S(S), .D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS), .TMEM_WORDS(TMEM_WORDS),
             .IMEM_WORDS(IMEM_WORDS), .DRAM_WORDS(DRAM_WORDS), .DRAM_LAT(DRAM_LAT),
             .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB), .MXU_IMPL(MXU_IMPL), .MXU_CL(MXU_CL), .VPU_CL(VPU_CL), .ULANES(ULANES), .AXI(AXI), .FIFO_DEPTH(FIFO_DEPTH)) dut (
    .clk, .sys_rst, .rst, .ld_start, .ld_addr, .ld_n, .ld_busy, .all_halted, .any_error, .icount,
    .dump);

  always #5 clk = ~clk;

  initial begin
    void'($value$plusargs("max_cycles=%d", max_cycles));
    repeat (3) @(posedge clk);
    sys_rst = 1'b0;
    // +boot: the program sits in DRAM at +boot_addr (bytes), +boot_n instructions; the loader
    // copies it into IMEM before the slices leave reset (as on the board)
    if ($test$plusargs("boot")) begin
      void'($value$plusargs("boot_addr=%d", ld_addr));
      void'($value$plusargs("boot_n=%d", ld_n));
      @(posedge clk);
      ld_start = 1'b1;
      @(posedge clk);
      ld_start = 1'b0;
      while (ld_busy) @(posedge clk);
    end
    @(posedge clk);
    rst = 1'b0;
    while (!all_halted && cycles < max_cycles) begin
      @(posedge clk);
      cycles++;
    end
    dump = 1'b1;
    @(posedge clk);
    dump = 1'b0;
    repeat (2) @(posedge clk);     // the slices print their trace events a cycle late
    $display("RESULT cycles=%0d halted=%0d error=%0d", cycles, all_halted, any_error);
    for (int s = 0; s < S; s++) $display("SLICE %0d icount=%0d", s, icount[s]);
    $finish;
  end
endmodule
