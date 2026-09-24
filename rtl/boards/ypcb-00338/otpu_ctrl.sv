// Host control registers (AXI4-Lite slave, reached from the host through the PCIe bridge's
// BAR). All in the core clock domain. Register map version 2 (docs/observability.md: the
// contract with the host); 12 address bits are decoded (the map repeats every 4 KiB).
//
//   0x00 ID        RO  0x4F545055 ("OTPU")
//   0x04 VERSION   RO  {D[15:0], MCOLS[7:0], LANES[7:0]}
//   0x08 CTRL      RW  bit0 RUN: 1 releases the slice from reset (write 0, then 1, per run)
//                      bit1 LOAD (write 1: copy PROG_N instructions from PROG_ADDR into IMEM;
//                      only while RUN = 0)
//                      bit2 CLEAR: zero the per-run counters (0x18 .. 0x34)
//   0x0C STATUS    RO  bit0 HALTED, bit1 ERROR (illegal instruction), bit2 LOADING,
//                      bit3 WR_IDLE, bit4 AXI_ERR (sticky), bit5 CALIB0, bit6 CALIB1, bit7 RUN
//   0x10 PROG_ADDR RW  program byte address in the slice's DRAM (chunk aligned)
//   0x14 PROG_N    RW  program length in instructions
//   0x18 CYCLES    RO  core cycles since RUN rose, until HALTED (low 32 bits)
//   0x1C CYCLES_HI RO
//   0x20 ICOUNT    RO  instructions retired
//   0x24 B_RD      RO  port B read requests taken (chunks)
//   0x28 B_WR      RO  port B write requests taken
//   0x2C A_RD      RO  port A read requests taken
//   0x30 SW_WR     RO  scalar (QST) write requests taken
//   0x34 B_STALL   RO  cycles a port B request waited for the memory
//   0x38 SCRATCH   RW  (host bring-up test)
//   0x3C REGMAP    RO  register map version (2)
//   0x40 CAPS      RO  bit0 trace buffer, bit1 temperature, [15:8] log2(trace depth),
//                      [23:16] log2(P/Q window cycles)
//   0x44 CORE_KHZ  RO  the core clock in kHz (build parameter)
//   0x48 BUILD_ID  RO  build parameter (the low 32 bits of the git commit)
//   0x4C TEMP      RO  bit31 valid, [11:0] XADC die-temperature code
//   0x50 SNAP      W   latch every free-running counter into its shadow; R: snapshots taken
//   0x100 + 8k     RO  free-running counter k's shadow (64 bits, low word first), k =
//                      UPTIME RUNNING MXU_BUSY MXU_MAC VPU_BUSY QNT_BUSY DMA_BUSY TMEM_DENY
//                      DRAM_RD DRAM_WR DRAM_WAIT INSTR; cleared by reset only
//   0x200 TRACE_CTRL RW  bit0 ENABLE (record while RUN), bit1 CLEAR (write 1), bit2 STOP_WHEN_FULL,
//                        bit3 BUSY (read only: events not yet in the buffer)
//   0x204 TRACE_COUNT RO records written since the clear (saturating)
//   0x208 TRACE_DROP RO  events lost to a full capture queue
//   0x20C TRACE_ADDR RW  the record to read
//   0x210 TRACE_LO   RO  its bits [31:0]
//   0x214 TRACE_HI   RO  its bits [63:32]; reading it increments TRACE_ADDR
// Reads answer three cycles after the address is taken (a registered address, a registered
// multiplexer, the data register), which also gives the trace RAM time after TRACE_ADDR moves.
module otpu_ctrl #(
  parameter int D     = 128,
  parameter int MCOLS = 2,
  parameter int LANES = 8,
  parameter int CORE_KHZ = 100000,
  parameter logic [31:0] BUILD_ID = 32'h0,
  parameter int TRACE_DEPTH = 16384,     // 0: no trace buffer
  parameter int PQ_WIN = 1024,           // the trace's P/Q window (cycles)
  parameter bit HAS_TEMP = 1'b1
) (
  input  logic        clk,
  input  logic        rst,
  // AXI4-Lite slave
  input  logic [11:0] s_awaddr,
  input  logic        s_awvalid,
  output logic        s_awready,
  input  logic [31:0] s_wdata,
  input  logic [3:0]  s_wstrb,
  input  logic        s_wvalid,
  output logic        s_wready,
  output logic [1:0]  s_bresp,
  output logic        s_bvalid,
  input  logic        s_bready,
  input  logic [11:0] s_araddr,
  input  logic        s_arvalid,
  output logic        s_arready,
  output logic [31:0] s_rdata,
  output logic [1:0]  s_rresp,
  output logic        s_rvalid,
  input  logic        s_rready,
  // core
  output logic        run,
  output logic        ld_start,
  output logic [31:0] ld_addr,
  output logic [31:0] ld_n,
  input  logic        ld_busy,
  input  logic        halted,
  input  logic        error,
  input  logic [31:0] icount,
  input  logic        wr_idle,
  input  logic        axi_err,
  input  logic [1:0]  calib,
  input  logic        b_rd, b_wr, a_rd, a_wr, b_wait,
  input  logic        temp_v,
  input  logic [11:0] temp,
  // activity for the free-running counters (any registered or combinational source)
  input  logic        mxu_busy, mxu_mac, vpu_busy, qnt_busy, dma_busy, tmem_deny,
  input  logic [1:0]  dram_rd, dram_wr,  // 64-byte beats this cycle
  input  logic        dram_wait,
  input  logic [1:0]  instr,             // instructions retired this cycle
  // trace buffer (otpu_trace)
  output logic        tr_en,
  output logic        tr_stop,
  output logic        tr_clear,
  output logic [31:0] tr_addr,
  input  logic [31:0] tr_count,
  input  logic [31:0] tr_drop,
  input  logic        tr_busy,
  input  logic [63:0] tr_rdata
);
  localparam int NFR = 12;
  localparam logic [31:0] CAPS = {8'd0, 8'($clog2(PQ_WIN)),
                                  8'(TRACE_DEPTH != 0 ? $clog2(TRACE_DEPTH) : 0),
                                  6'd0, HAS_TEMP, TRACE_DEPTH != 0};

  logic [63:0] cycles;
  logic [31:0] n_brd, n_bwr, n_ard, n_awr, n_bst, scratch;
  logic        clear, snap;
  logic [1:0]  ld_pend_q;                  // LOAD written, loader not yet visibly busy
  wire         ld_pend = ld_start || (|ld_pend_q);
  always_ff @(posedge clk) ld_pend_q <= rst ? 2'b00 : {ld_pend_q[0], ld_start};

  // ---- write channel: take address and data together
  logic       w_go, tr_inc;
  assign s_awready = s_awvalid && s_wvalid && !s_bvalid;
  assign s_wready  = s_awready;
  assign w_go      = s_awready;
  assign s_bresp   = 2'b00;

  always_ff @(posedge clk) begin
    ld_start <= 1'b0;
    clear <= 1'b0;
    snap <= 1'b0;
    tr_clear <= 1'b0;
    if (rst) begin
      run <= 1'b0;
      s_bvalid <= 1'b0;
      ld_addr <= '0; ld_n <= '0; scratch <= '0;
      tr_en <= 1'b0; tr_stop <= 1'b0; tr_addr <= '0;
    end else begin
      if (s_bvalid && s_bready) s_bvalid <= 1'b0;
      if (tr_inc) tr_addr <= tr_addr + 1;
      if (w_go) begin
        s_bvalid <= 1'b1;
        case (s_awaddr[11:2])
          10'h002: begin
            run <= s_wdata[0];
            ld_start <= s_wdata[1] && !s_wdata[0] && !run;
            clear <= s_wdata[2];
          end
          10'h004: ld_addr <= s_wdata;
          10'h005: ld_n <= s_wdata;
          10'h00E: scratch <= s_wdata;
          10'h014: snap <= 1'b1;
          10'h080: begin
            tr_en <= s_wdata[0];
            tr_clear <= s_wdata[1];
            tr_stop <= s_wdata[2];
          end
          10'h083: tr_addr <= s_wdata;
          default: ;
        endcase
      end
    end
  end

  // ---- per-run counters
  always_ff @(posedge clk) begin
    if (rst || clear) begin
      cycles <= '0; n_brd <= '0; n_bwr <= '0; n_ard <= '0; n_awr <= '0; n_bst <= '0;
    end else begin
      if (run && !halted) cycles <= cycles + 1;
      if (b_rd) n_brd <= n_brd + 1;
      if (b_wr) n_bwr <= n_bwr + 1;
      if (a_rd) n_ard <= n_ard + 1;
      if (a_wr) n_awr <= n_awr + 1;
      if (b_wait) n_bst <= n_bst + 1;
    end
  end

  // ---- free-running counters (reset only), their increments registered first; SNAP copies
  // them all into the shadows in one cycle
  logic [1:0]  fr_inc [NFR];
  logic [63:0] fr [NFR], fr_s [NFR];
  logic [31:0] n_snap;
  always_ff @(posedge clk) begin
    fr_inc[0]  <= 2'd1;
    fr_inc[1]  <= 2'(run && !halted);
    fr_inc[2]  <= 2'(mxu_busy);
    fr_inc[3]  <= 2'(mxu_mac);
    fr_inc[4]  <= 2'(vpu_busy);
    fr_inc[5]  <= 2'(qnt_busy);
    fr_inc[6]  <= 2'(dma_busy);
    fr_inc[7]  <= 2'(tmem_deny);
    fr_inc[8]  <= dram_rd;
    fr_inc[9]  <= dram_wr;
    fr_inc[10] <= 2'(dram_wait);
    fr_inc[11] <= instr;
    if (rst) begin
      for (int k = 0; k < NFR; k++) begin fr[k] <= '0; fr_s[k] <= '0; fr_inc[k] <= '0; end
      n_snap <= '0;
    end else begin
      for (int k = 0; k < NFR; k++) fr[k] <= fr[k] + 64'(fr_inc[k]);
      if (snap) begin
        for (int k = 0; k < NFR; k++) fr_s[k] <= fr[k];
        n_snap <= n_snap + 1;
      end
    end
  end

  // ---- read channel: one read at a time; address, multiplexer and data registered
  logic [9:0]  r_a;
  logic [1:0]  r_p;
  logic [31:0] r_d;
  assign s_arready = !s_rvalid && r_p == 2'b00;
  assign s_rresp = 2'b00;
  assign tr_inc = r_p[1] && r_a == 10'h085;
  always_ff @(posedge clk) begin
    if (rst) begin
      s_rvalid <= 1'b0;
      r_p <= 2'b00;
    end else begin
      if (s_rvalid && s_rready) s_rvalid <= 1'b0;
      r_p <= {r_p[0], s_arvalid && s_arready};
      if (r_p[1]) begin
        s_rvalid <= 1'b1;
        s_rdata <= r_d;
      end
    end
    if (s_arvalid && s_arready) r_a <= s_araddr[11:2];
    case (r_a)
      10'h000: r_d <= 32'h4F54_5055;
      10'h001: r_d <= {16'(D), 8'(MCOLS), 8'(LANES)};
      10'h002: r_d <= {31'd0, run};
      10'h003: r_d <= {24'd0, run, calib, axi_err, wr_idle, ld_busy || ld_pend, error, halted};
      10'h004: r_d <= ld_addr;
      10'h005: r_d <= ld_n;
      10'h006: r_d <= cycles[31:0];
      10'h007: r_d <= cycles[63:32];
      10'h008: r_d <= icount;
      10'h009: r_d <= n_brd;
      10'h00A: r_d <= n_bwr;
      10'h00B: r_d <= n_ard;
      10'h00C: r_d <= n_awr;
      10'h00D: r_d <= n_bst;
      10'h00E: r_d <= scratch;
      10'h00F: r_d <= 32'd2;
      10'h010: r_d <= CAPS;
      10'h011: r_d <= 32'(CORE_KHZ);
      10'h012: r_d <= BUILD_ID;
      10'h013: r_d <= {temp_v, 19'd0, temp};
      10'h014: r_d <= n_snap;
      10'h080: r_d <= {28'd0, tr_busy, tr_stop, 1'b0, tr_en};
      10'h081: r_d <= tr_count;
      10'h082: r_d <= tr_drop;
      10'h083: r_d <= tr_addr;
      10'h084: r_d <= tr_rdata[31:0];
      10'h085: r_d <= tr_rdata[63:32];
      default:
        if (r_a[9:6] == 4'h1 && r_a[5:1] < 5'(NFR))
          r_d <= r_a[0] ? fr_s[r_a[5:1]][63:32] : fr_s[r_a[5:1]][31:0];
        else
          r_d <= 32'hDEAD_BEEF;
    endcase
  end
endmodule
