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
  parameter int MXU_QD     = 2;       // MXU commands queued (otpu_mxu QD; 2..7)
  parameter int QUANT_XD   = 1024;    // quantizer row buffer chunks (0: two passes over TMEM)
  parameter bit QUANT_R1   = 1'b0;    // RSCALE's row factor read once a row (otpu_quant R1)
  parameter int QUANT_PW   = 1;       // a buffered QACT pass 1's chunks a cycle (otpu_quant PW)
  parameter bit QUANT_SPLIT = 1'b0;   // QACT / QST flag LO (otpu_quant SPLIT; Config.SPLITQ)
  parameter bit QUANT_HL    = 1'b0;  // QACT flag HL (otpu_quant HL; Config.QHL)
  parameter int QUANT_SWW  = 1;       // the SW port's words (otpu_quant SWW; timing only)
  parameter bit QUANT_QSTB  = 1'b0;   // QST rows through the row buffer (otpu_quant QSTB; timing only)
  parameter bit QUANT_QSTHL = 1'b0;   // QST HL: a GRP ROW QST's hi and lo from one read (otpu_quant QSTHL; Config.QSTHL)
  parameter bit QONCE      = 1'b0;    // QACT flag ST and LDA (Config.QONCE)
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1;
  parameter int ULANES     = LANES;
  parameter int AXI        = 0;       // otpu_top's memory path (1: the board's, native; 2: the
                                      // card's channels, LiteDRAM's controller)
  parameter int FIFO_DEPTH = 128;
  parameter int TMEM_IMPL  = 0;       // TMEM: 0 a copy per read port, 1 banked (otpu_slice)
  parameter bit SEQ_WC     = 1'b0;    // EXPERIMENTAL: otpu_seq's wait counts (docs/seq_vpu.md)
  parameter int SE_R       = 0;       // streams on an engine of their own (otpu_slice)
  parameter int DMA_DEPTH  = 128;     // otpu_dma DEPTH (otpu_slice)
  parameter int DMA_LDQ    = 1;       // otpu_dma LDQ (otpu_slice)
  parameter int MXU_PUMP   = 0;       // the MXU's array pumped on clk_f (otpu_mxu PUMP)
  parameter int VPU_FSUM   = 0;       // otpu_vpu FSUM: EXP2SUB's fused row sums (Config.FSUM)
  parameter int VPU_SIG    = 0;       // otpu_vpu VSIG: VOP SIGMOID / SILU (Config.VSIG)
  parameter int VPU_MAC    = 0;       // otpu_vpu MAC: VOP MAC (Config.VMAC)
  parameter int VPU_OUTER1 = 0;       // otpu_vpu OUTER1: VOP OUTER decay one on MAC's (Config.OUTER1)
  parameter int VPU_CNS    = 3;       // otpu_vpu CNS: the composites' stages per lane (3, or 5: two passes)
  parameter bit OUTER      = 1'b1;    // VOP OUTER (0: a VPU without it, Config.OUTER)
  parameter bit DSTEP2     = 1'b0;    // DSTEP's pair (Config.DSTEP2)
  parameter int ST16       = 0;       // a stream's DRAM state in 16 bits (Config.ST16)
  parameter bit ST16S      = 1'b0;    // with ST16: STREAM's flags ST16R / ST16W (Config.ST16S)
  parameter bit KT         = 1'b0;    // MM's K tail (Config.KT; WG > 1)
  parameter int MXU_SPL    = 0;       // MM SPL, a split's terms as one MM (Config.SPLMM; WG > 1)
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
  parameter string TMEM_STYLE = "block"; // TMEM_PMP 1: the pumped RAMs' (otpu_slice)
  parameter string IMEM_STYLE = "auto";  // IMEM's RAMs (otpu_slice)
  parameter string IMEM_WC_STYLE = "";   // its wait entries', if not IMEM_STYLE's (otpu_slice)
  parameter string MXU_RF_STYLE = "block";  // the MXU's result FIFO (otpu_slice)
  parameter string MXU_FIFO_STYLE = "block";  // the MXU's chunk FIFO lanes (otpu_slice)
  parameter int MXU_RF_ROWS = 0;      // the MXU's result FIFO rows (otpu_slice; 0: its rule)
  parameter int ACT_UCOLS  = 0;       // the ACT RAM's columns in UltraRAM (otpu_slice)
  parameter bit COLL_GATHER = 1'b1;   // 0: the BAR-only collective unit (otpu_coll_bar) at S = 1,
                                      // as the card builds it (rtlsim.BOARD_UARCH); S > 1 has GATHER
  parameter bit COLL_HUB   = 1'b0;    // S > 1: otpu_board's collective path (otpu_top COLL_HUB;
                                      // rtlsim.BOARD_UARCH of an FPGA target at S > 1)
  parameter int XRS        = 2;       // COLL_HUB: the stages each way (otpu_top XRS)
  parameter bit COLL_PIPE  = 1'b0;    // COLL_HUB: the hub's reads in flight (otpu_top COLL_PIPE)
  parameter int NCH        = 2;       // the memory's channels and beat bytes (otpu_top; the
  parameter int CB         = 64;      //   target's: opentpu.target.sim_uarch), the adapter's
  parameter int NAT_RD     = 256 / NCH; //   depths (the target's [uarch] NAT_*)
  parameter int NAT_AD     = 64;
  parameter int NAT_XD     = 64;
  parameter int NAT_WQD    = 32;
  parameter int NAT_APF    = 32;
  parameter bit NAT_FILL   = 1'b1;
  parameter bit NAT_WPOST  = 1'b0;
  parameter bit NAT_SWG    = 1'b0;
  parameter bit NAT_XPOST  = 1'b0;
  parameter int NAT_XQB    = 0;       // NAT_XPOST: a beat's writes, 2^NAT_XQB (otpu_top)
  parameter int NAT_XSTG   = 0;       // stages between a slice and its adapter (otpu_top)
  parameter int NAT_XSTG_P = NAT_XSTG; // the posts' (otpu_top)
  parameter bit NAT_RHB    = 1'b0;    // the channels' read / write data in half beats (otpu_top)
  parameter bit NAT_WHB    = 1'b0;
  parameter int WG         = 1;       // weight rows a group (otpu_slice; the config's WG)
  parameter bit CHASH      = 1'b1;    // the channels' hash (otpu_top; at WG > 1 the Config's)
  parameter int AW         = 32;      // DRAM byte address bits (otpu_top AW; AXI 0 at AW > 32:
                                      //   the words above DRAM_WORDS sparse, otpu_dram)
  parameter bit ICACHE     = 1'b0;    // IMEM an instruction cache (otpu_top; the program in DRAM:
  parameter int IC_LINES   = 16;      //   +boot)
  parameter int IC_PF      = 2;

  logic clk = 1'b0, clk_f = 1'b0, sys_rst = 1'b1, rst = 1'b1, dump = 1'b0;
  logic ld_start = 1'b0, ld_busy;
  logic [31:0] ld_addr = '0, ld_n = '0;
  logic [31:0] rinit [8] = '{default: '0};    // +arg0=N .. +arg7=N: R8..R15 at the start
  logic all_halted, any_error;
  logic [31:0] icount [S];
  longint cycles = 0, max_cycles = 50_000_000;

  otpu_top #(.S(S), .D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS), .ACT_ROWS(ACT_ROWS), .TMEM_WORDS(TMEM_WORDS),
             .IMEM_WORDS(IMEM_WORDS), .DRAM_WORDS(DRAM_WORDS), .DRAM_LAT(DRAM_LAT),
             .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB), .MXU_IMPL(MXU_IMPL), .MXU_CL(MXU_CL), .MXU_E(MXU_E), .MXU_QD(MXU_QD), .QUANT_XD(QUANT_XD), .QUANT_R1(QUANT_R1), .QUANT_PW(QUANT_PW), .QUANT_SPLIT(QUANT_SPLIT), .QUANT_HL(QUANT_HL), .QUANT_SWW(QUANT_SWW), .QUANT_QSTB(QUANT_QSTB), .QUANT_QSTHL(QUANT_QSTHL), .QONCE(QONCE), .VPU_CL(VPU_CL), .ULANES(ULANES), .AXI(AXI), .FIFO_DEPTH(FIFO_DEPTH),
             .TMEM_IMPL(TMEM_IMPL), .TMEM_RQ(TMEM_RQ), .TMEM_NCP(TMEM_NCP),
             .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH),
             .TMEM_RPL(TMEM_RPL), .TMEM_TDP(TMEM_TDP), .TMEM_GST(TMEM_GST),
             .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK), .TMEM_SDM(TMEM_SDM),
             .TMEM_STYLE(TMEM_STYLE), .IMEM_STYLE(IMEM_STYLE), .ACT_UCOLS(ACT_UCOLS),
             .MXU_RF_STYLE(MXU_RF_STYLE), .MXU_FIFO_STYLE(MXU_FIFO_STYLE),
             .MXU_RF_ROWS(MXU_RF_ROWS),
             .IMEM_WC_STYLE(IMEM_WC_STYLE),
             .SEQ_WC(SEQ_WC), .COLL_GATHER(COLL_GATHER || S > 1), .COLL_HUB(COLL_HUB), .XRS(XRS),
             .COLL_PIPE(COLL_PIPE),
             .NCH(NCH), .CB(CB), .NAT_RD(NAT_RD), .NAT_AD(NAT_AD), .NAT_XD(NAT_XD),
             .NAT_WQD(NAT_WQD), .NAT_APF(NAT_APF), .NAT_FILL(NAT_FILL), .NAT_WPOST(NAT_WPOST), .NAT_SWG(NAT_SWG), .NAT_XPOST(NAT_XPOST), .NAT_XQB(NAT_XQB), .NAT_XSTG(NAT_XSTG), .NAT_XSTG_P(NAT_XSTG_P), .NAT_RHB(NAT_RHB), .NAT_WHB(NAT_WHB), .WG(WG), .CHASH(CHASH), .SE_R(SE_R), .DMA_DEPTH(DMA_DEPTH), .DMA_LDQ(DMA_LDQ), .MXU_PUMP(MXU_PUMP), .VPU_FSUM(VPU_FSUM), .VPU_SIG(VPU_SIG), .VPU_MAC(VPU_MAC), .VPU_OUTER1(VPU_OUTER1), .VPU_CNS(VPU_CNS),
             .OUTER(OUTER), .DSTEP2(DSTEP2), .ST16(ST16), .ST16S(ST16S), .KT(KT), .MXU_SPL(MXU_SPL), .ICACHE(ICACHE), .IC_LINES(IC_LINES),
             .IC_PF(IC_PF), .AW(AW)) dut (
    .clk, .clk_f, .sys_rst, .rst, .rinit, .ld_start, .ld_addr, .ld_n, .ld_busy, .all_halted, .any_error,
    .icount, .dump);

  // TMEM_PMP 1: clk_f (2 x clk) rises with each edge of clk, in clk's own process (no race between
  // the two domains); clk's edges stay where they are (the memory models' clocks are absolute)
  if (TMEM_PMP == 1 || MXU_PUMP != 0) begin : g_clkf         // (and the pumped MXU's)
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
