// Simulation harness: memories load prog_<s>.hex / dram_<s>.bin from +dir=..., the machine runs
// until every slice halts, then TMEM and DRAM of every slice are dumped for comparison.
// +runs=N: N runs in one simulation, as the host runs a program again (the memory path, sys_rst,
// is not reset between them): after a run halts, the core's reset (AXI = 1: the memory model
// writes the next run's pokeb lines, the host's writes between the runs), +reload: the program
// loaded again, +r<k>_arg<j>=N: run k's arguments (else the run before's), then the next run.
module tb_top;
  parameter int S          = 1;
  parameter int D          = 32;
  parameter int MCOLS      = 8;
  parameter int ACT_BLOCKS = 64;
  parameter int ACT_ROWS   = MCOLS;
  parameter int TMEM_WORDS = 1 << 16;
  parameter int IMEM_WORDS = 1 << 16;
  parameter int DRAM_WORDS = 1 << 18;
  parameter int DRAM_LAT   = 8;
  parameter int LANES      = 8;
  parameter int WIN        = 32;
  parameter int RPB        = 4;
  parameter int WPB        = 2;
  parameter int MXU_IMPL   = 2;
  parameter int MXU_CL     = 16;
  parameter int MXU_E      = 0;       // MXU epilogue lanes (0: MCOLS / 2)
  parameter int QUANT_XD   = 1024;    // quantizer row buffer chunks (0: two passes over TMEM)
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1;
  parameter int ULANES     = LANES;
  parameter int AXI        = 0;       // otpu_top's memory path (1: the board's, native; 2: the
                                      // card's channels, LiteDRAM's controller)
  parameter int FIFO_DEPTH = 128;
  parameter int TMEM_IMPL  = 0;       // TMEM: 0 a copy per read port, 1 banked (otpu_slice)
  parameter bit SEQ_WC     = 1'b0;    // EXPERIMENTAL: otpu_seq's wait counts (docs/seq_vpu.md)
  parameter bit TMEM_RQ    = 1'b0;    // TMEM_IMPL 1: request-priority bank addresses
  parameter int TMEM_NCP   = 3;       // TMEM_IMPL 1: copies
  parameter int TMEM_NSB   = 8;       // TMEM_IMPL 3: sub-banks per bank
  parameter int TMEM_SBP   = 3;       // TMEM_IMPL 3: the sub-bank field's word-address bit
  parameter int TMEM_SBH   = 0;       // TMEM_IMPL 3: the sub-bank hashed
  parameter bit TMEM_RPL   = 1'b0;    // TMEM_IMPL 3: lost reads replayed
  parameter bit TMEM_TDP   = 1'b0;    // TMEM_IMPL 3: true dual-port RAMs
  parameter bit TMEM_GST   = 1'b0;    // TMEM_IMPL 3: QUANT 3 and COLL the DMA's guests
  parameter int TMEM_PMP   = 0;       // TMEM_TDP: 2 the pumped RAMs' model
  parameter int TMEM_PRK   = 0;       // TMEM_IMPL 3: the read ports' ranks (otpu_slice)
  parameter int TMEM_SDM   = -1;      // TMEM_TDP: the read ports' RAM sides (otpu_slice)
  parameter bit COLL_GATHER = 1'b1;   // 0: the BAR-only collective unit (otpu_coll_bar) at S = 1,
                                      // as the card builds it (rtlsim.BOARD_UARCH); S > 1 has GATHER
  parameter bit DUAL       = 1'b0;    // port B chunk pairs: int8 MM PAIR (otpu_top)

  logic clk = 1'b0, clk_f = 1'b0, sys_rst = 1'b1, rst = 1'b1, dump = 1'b0;
  logic ld_start = 1'b0, ld_busy;
  logic [31:0] ld_addr = '0, ld_n = '0;
  logic [31:0] rinit [8] = '{default: '0};    // +arg0=N .. +arg7=N: R8..R15 at the start
  logic all_halted, any_error;
  logic [31:0] icount [S];
  longint cycles = 0, max_cycles = 50_000_000;

  otpu_top #(.S(S), .D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS), .ACT_ROWS(ACT_ROWS), .TMEM_WORDS(TMEM_WORDS),
             .IMEM_WORDS(IMEM_WORDS), .DRAM_WORDS(DRAM_WORDS), .DRAM_LAT(DRAM_LAT),
             .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB), .MXU_IMPL(MXU_IMPL), .MXU_CL(MXU_CL), .MXU_E(MXU_E), .QUANT_XD(QUANT_XD), .VPU_CL(VPU_CL), .ULANES(ULANES), .AXI(AXI), .FIFO_DEPTH(FIFO_DEPTH),
             .TMEM_IMPL(TMEM_IMPL), .TMEM_RQ(TMEM_RQ), .TMEM_NCP(TMEM_NCP),
             .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH),
             .TMEM_RPL(TMEM_RPL), .TMEM_TDP(TMEM_TDP), .TMEM_GST(TMEM_GST),
             .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK), .TMEM_SDM(TMEM_SDM), .SEQ_WC(SEQ_WC),
             .COLL_GATHER(COLL_GATHER || S > 1), .DUAL(DUAL)) dut (
    .clk, .clk_f, .sys_rst, .rst, .rinit, .ld_start, .ld_addr, .ld_n, .ld_busy, .all_halted, .any_error,
    .icount, .dump);

  // TMEM_PMP 1: clk_f (2 x clk) rises with each edge of clk, in clk's own process (no race between
  // the two domains); clk's edges stay where they are (the memory models' clocks are absolute)
  if (TMEM_PMP == 1) begin : g_clkf
    initial forever begin
      #2 clk_f = 1'b0;
      #3 begin clk = ~clk; clk_f = 1'b1; end
    end
  end else begin : g_clk
    always #5 clk = ~clk;
  end

  initial begin
    void'($value$plusargs("max_cycles=%d", max_cycles));
    begin
      string nm;
      for (int k = 0; k < 8; k++) begin
        nm = $sformatf("arg%0d=%%d", k);
        void'($value$plusargs(nm, rinit[k]));
      end
    end
    // the testbench changes its signals on the falling edge: race-free against the design's
    // flip-flops and against a memory model's own clock (AXI = 2)
    repeat (3) @(negedge clk);
    sys_rst = 1'b0;
    // +boot: the program sits in DRAM at +boot_addr (bytes), +boot_n instructions; the loader
    // copies it into IMEM before the slices leave reset (as on the board)
    if ($test$plusargs("boot")) begin
      void'($value$plusargs("boot_addr=%d", ld_addr));
      void'($value$plusargs("boot_n=%d", ld_n));
      @(negedge clk);
      ld_start = 1'b1;
      @(negedge clk);
      ld_start = 1'b0;
      while (ld_busy) @(negedge clk);
    end
    @(negedge clk);
    rst = 1'b0;
    while (!all_halted && cycles < max_cycles) begin
      @(negedge clk);
      cycles++;
    end
    begin
      int runs = 1;
      string nm;
      void'($value$plusargs("runs=%d", runs));
      for (int r = 1; r < runs && all_halted && !any_error; r++) begin
        repeat (256) @(negedge clk);     // the halted run's last writes land
        rst = 1'b1;                      // (the memory model: the host's writes between them)
        for (int k = 0; k < 8; k++) begin
          nm = $sformatf("r%0d_arg%0d=%%d", r, k);
          void'($value$plusargs(nm, rinit[k]));
        end
        repeat (4) @(negedge clk);
        if ($test$plusargs("reload")) begin
          ld_start = 1'b1;
          @(negedge clk);
          ld_start = 1'b0;
          while (ld_busy) @(negedge clk);
        end
        @(negedge clk);
        rst = 1'b0;
        @(negedge clk);
        while (!all_halted && cycles < max_cycles) begin
          @(negedge clk);
          cycles++;
        end
      end
    end
    // the memory path finishes what it took before the dump: a halted slice's last writes are
    // counted once the channel takes their commands, their data a few cycles later
    repeat (256) @(negedge clk);
    dump = 1'b1;
    @(negedge clk);
    dump = 1'b0;
    repeat (2) @(posedge clk);     // the slices print their trace events a cycle late
    $display("RESULT cycles=%0d halted=%0d error=%0d", cycles, all_halted, any_error);
    for (int s = 0; s < S; s++) $display("SLICE %0d icount=%0d", s, icount[s]);
    $finish;
  end
endmodule
