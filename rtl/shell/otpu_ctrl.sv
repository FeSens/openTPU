// Host control registers (AXI4-Lite slave, reached from the host through the PCIe bridge's
// BAR). All in the core clock domain. Register map version 5 (docs/observability.md: the
// contract with the host); 12 address bits are decoded (the map repeats every 4 KiB).
//
//   0x00 ID        RO  0x4F545055 ("OTPU")
//   0x04 VERSION   RO  {D[15:0], MCOLS[7:0], LANES[7:0]}
//   0x08 CTRL      RW  bit0 RUN: 1 releases the slice from reset (write 0, then 1, per run); it
//                      rises only while both channels are calibrated (CALIB0 and CALIB1), else
//                      the write leaves it 0 (STATUS RUN)
//                      bit1 LOAD (write 1: copy PROG_N instructions from PROG_ADDR into IMEM;
//                      only while RUN = 0 and both channels are calibrated)
//                      bit2 CLEAR: zero the per-run counters (0x18 .. 0x34)
//                      Stopping a run (RUN 1 -> 0) stops the slice at once; what it gave the memory
//                      still goes: wait for WR_IDLE before writing the DRAM the run wrote or read.
//                      Its reads still in flight are dropped (a LOAD and the next run start after
//                      them)
//   0x0C STATUS    RO  bit0 HALTED, bit1 ERROR (illegal instruction; or WAIT_TO), bit2 LOADING,
//                      bit3 WR_IDLE (every write the slice gave the memory is taken by the DDR3
//                      controllers), bit4 AXI_ERR (sticky: a DDR3 controller broke its port contract,
//                      otpu_mem_ch n_err; on the MIG builds an AXI error response), bit5 CALIB0,
//                      bit6 CALIB1, bit7 RUN,
//                      bit8 WAIT_TO (with ERROR: a WAITW timed out; as ERROR, until RUN falls),
//                      bit9 ECC_DED (a channel's ECC counted an uncorrectable 64-bit word since its
//                      counters were last cleared: the LiteDRAM CSR ecc_clear / ecc1_clear; the
//                      word's data was wrong, and a partial write merged into it wrote it back
//                      with good check bits)
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
//   0x3C REGMAP    RO  register map version (5; 4 before CAPS2, 3 before the CFG_* registers)
//   0x40 CAPS      RO  bit0 trace buffer, bit1 temperature, bit2 I2C pins, bit3 DDR_MTS,
//                      bit4 4-bit MM weights (MM flags WF, docs/isa.md), bit5 column reuse (MM
//                      PAIR / QACT DUP), bit6 DSTEP (the DMA's DeltaNet head step), bit7
//                      hashed channel interleave (otpu_native_dram CHASH), bit24 ACT_ROWS,
//                      [15:8] log2(trace depth), [23:16] log2(P/Q window cycles),
//                      bit25 the run's arguments (ARG0..7), bit26 STREAM (the stream engine's
//                      hardware subset, docs/stream.md; with DSTEP), bit27 the DDR3
//                      controllers want the host's calibration (LiteDRAM:
//                      opentpu/host/memcal.py), bit29 MX: the MX numerics (MM's exact
//                      accumulation with power-of-two block scales, QACT / QST's offsets;
//                      docs/isa.md, docs/mx_mxu.md; every bitstream since), bit30 GEN: the
//                      decode loop's instructions
//                      (RLD, VOP ARGMAX, HALT CHAIN; docs/isa.md, docs/autodecode.md), bit31
//                      WAITW (the DMA waits for a word the host writes; docs/isa.md)
//   0x44 CORE_KHZ  RO  the core clock in kHz (build parameter)
//   0x48 BUILD_ID  RO  build parameter (the low 32 bits of the git commit)
//   0x4C TEMP      RO  bit31 valid, [11:0] XADC die-temperature code
//   0x50 SNAP      W   latch every free-running counter into its shadow; R: snapshots taken
//   0x54 DDR_MTS   RO  the DDR3 data rate in MT/s (build parameter; CAPS bit3 when nonzero)
//   0x58 ACT_ROWS  RO  ACT RAM rows: the most rows of one MM (build parameter; CAPS bit24)
//   0x5C CAPS2     RO  (version 5; CAPS is full) bit0 GATHER: the collective unit runs GATHER
//                      (always at S > 1; at S = 1 unless built without it, otpu_board COLL_GATHER:
//                      then BAR alone, and a GATHER raises ERROR)
//   0x60 + 4k      RW  ARG0..ARG7: the run's arguments, R8..R15 when RUN rises (CAPS bit25;
//                      docs/isa.md "Arguments"); reset 0
//   0x80 CFG_SLICES   RO  slices S (version 4: the configuration the target file built,
//                         opentpu/target.py; docs/targets.md)
//   0x84 CFG_TMEM     RO  TMEM words per slice
//   0x88 CFG_IMEM     RO  IMEM words per slice (8 per instruction)
//   0x8C CFG_ACT      RO  ACT RAM blocks
//   0x90 CFG_DRAM     RO  [7:0] log2 DRAM bytes per slice, [15:8] memory channels per slice,
//                         [23:16] log2 bytes a slice addresses per channel
//   0x94 CFG_UARCH    RO  [3:0] MXU_IMPL, [7:4] TMEM_IMPL, [15:8] VPU_CL, [23:16] ULANES,
//                         [24] SEQ_WC (otpu_seq's wait entries: every program needs them, docs/isa.md),
//                         [25] QUAD (a 4-bit MM PAIR of MCOLS/4 rows four blocks an advance, otpu_mxu)
//   0x98 CFG_PLATFORM RO  the target's platform id (1 YPCB-00338, 2 BCU1525; 256 and up ASICs)
//   0x9C CFG_HASH     RO  opentpu.target.arch_hash of the target's Config at build time
//   0xA0 + 4s         (reserved: slice s's DRAM window, docs/targets.md)
//   0xC0 + 4s ICOUNT_s RO slice s's instructions retired (S > 1; ICOUNT is slice 0's)
//   0x100 + 8k     RO  free-running counter k's shadow (64 bits, low word first), k =
//                      UPTIME RUNNING MXU_BUSY MXU_MAC VPU_BUSY QNT_BUSY DMA_BUSY TMEM_DENY
//                      DRAM_RD DRAM_WR DRAM_WAIT INSTR MXU_STARVE (version 3); cleared by
//                      reset only
//   0x200 TRACE_CTRL RW  bit0 ENABLE (record while RUN), bit1 CLEAR (write 1), bit2 STOP_WHEN_FULL,
//                        bit3 BUSY (read only: events not yet in the buffer)
//   0x204 TRACE_COUNT RO records written since the clear (saturating in its high bits: otpu_trace)
//   0x208 TRACE_DROP RO  events lost to a full capture queue
//   0x20C TRACE_ADDR RW  the record to read
//   0x210 TRACE_LO   RO  its bits [31:0]
//   0x214 TRACE_HI   RO  its bits [63:32]; reading it increments TRACE_ADDR
//   0x220 I2C_CTRL   RW  open-drain pins, 1 = drive low, 0 = release: bit0 SCL0, bit1 SDA0
//                        (LM73 sensor bus), bit2 SCL1, bit3 SDA1 (PCIe edge SMBus); reset 0
//   0x224 I2C_IN     RO  the pin levels (synchronized by otpu_board): bit0 SCL0, bit1 SDA0,
//                        bit2 SCL1, bit3 SDA1, bit4 ALERT0 (LM73 ALERT, active low)
// Reads answer four cycles after the address is taken (a registered address, a registered
// multiplexer per register group, the group's choice registered, the data register), which also
// gives the trace RAM time after TRACE_ADDR moves.
module otpu_ctrl #(
  parameter int D     = 128,
  parameter int MCOLS = 2,
  parameter int LANES = 8,
  parameter int ACT_ROWS = MCOLS,
  parameter int CORE_KHZ = 100000,
  parameter logic [31:0] BUILD_ID = 32'h0,
  parameter int DDR_MTS = 0,             // DDR3 data rate (MT/s); 0: not given
  parameter int TRACE_DEPTH = 16384,     // 0: no trace buffer
  parameter int PQ_WIN = 1024,           // the trace's P/Q window (cycles)
  parameter bit HAS_TEMP = 1'b1,
  parameter bit HAS_I2C = 1'b1,          // CAPS bit2: the I2C pins are wired (otpu_fpga_top_ld)
  parameter bit CHASH = 1'b0,            // CAPS bit7: the hashed channel interleave (otpu_native_dram)
  parameter bit DSTEP = 1'b1,            // CAPS bit6 and bit26: the stream engine (DSTEP, STREAM)
  parameter bit HOSTCAL = 1'b0,          // CAPS bit27: the host calibrates the DDR3 controllers
  parameter bit COLL_GATHER = 1'b1,      // CAPS2 bit0 (with S > 1): the collective unit runs GATHER
  parameter int LDP = 2,                 // cycles after LOAD before the loaders' busy shows (>= 2)
  // the configuration registers (CFG_*, register map 4): the target's values
  parameter int S          = 1,
  parameter int TMEM_WORDS = 1 << 16,
  parameter int IMEM_WORDS = 1 << 15,
  parameter int ACT_BLOCKS = 128,
  parameter int DRAM_LOG2  = 32,         // log2 DRAM bytes per slice (otpu_board: DAW, 32 .. 36)
  parameter int NCH        = 2,          // memory channels per slice
  parameter int CH_LOG2    = 31,         // log2 bytes a slice addresses per channel (DAW > 32:
                                         // DAW - log2 NCH; 31 at DAW 32)
  parameter int MXU_IMPL   = 2,
  parameter int TMEM_IMPL  = 0,
  parameter int VPU_CL     = 2,
  parameter int ULANES     = 8,
  parameter bit SEQ_WC     = 1'b0,    // EXPERIMENTAL: otpu_seq's wait entries (WAITC, docs/seq_vpu.md; CFG_UARCH bit24)
  parameter bit QUAD       = 1'b0,    // the MXU's QUAD (CFG_UARCH bit25: in the Config, so in CFG_HASH)
  parameter int PLATFORM   = 0,
  parameter logic [31:0] ARCH_HASH = 32'h0,
  // the free-running counters' increment widths (2 and 2 for the YPCB's one slice on two
  // channels): DRAM_RD / DRAM_WR the beats every slice's channels move in a cycle, INSTR the
  // instructions every slice retires in a cycle (up to 2 each)
  parameter int CW         = $clog2(S * NCH + 1),
  parameter int IW         = $clog2(2 * S + 1)
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
  output logic [31:0] arg [8],           // ARG0..7 -> the sequencer's R8..R15
  output logic [31:0] ld_addr,
  output logic [31:0] ld_n,
  input  logic        ld_busy,
  input  logic        halted,
  input  logic        error,
  input  logic        wait_to,
  input  logic [S-1:0][31:0] icount,  // each slice's (ICOUNT: slice 0's; ICOUNT_s at 0xC0 + 4s)
  input  logic        wr_idle,
  input  logic        axi_err,
  input  logic [1:0]  calib,
  input  logic        ecc_ded,           // STATUS ECC_DED (synchronized: otpu_board)
  input  logic        b_rd, b_wr, a_rd, a_wr, b_wait,
  input  logic        temp_v,
  input  logic [11:0] temp,
  // activity for the free-running counters (any registered or combinational source)
  input  logic        mxu_busy, mxu_mac, vpu_busy, qnt_busy, dma_busy, tmem_deny,
  input  logic        mxu_starve,        // the MXU streams a command and has no chunk
  input  logic [CW-1:0] dram_rd, dram_wr,  // 64-byte beats this cycle (every slice's channels)
  input  logic        dram_wait,
  input  logic [IW-1:0] instr,           // instructions retired this cycle (every slice's)
  // trace buffer (otpu_trace)
  output logic        tr_en,
  output logic        tr_stop,
  output logic        tr_clear,
  output logic [31:0] tr_addr,
  input  logic [31:0] tr_count,
  input  logic [31:0] tr_drop,
  input  logic        tr_busy,
  input  logic [63:0] tr_rdata,
  // I2C pins (the host bit-bangs them: opentpu/host/i2c.py)
  output logic [3:0]  i2c_lo,
  input  logic [4:0]  i2c_in
);
  localparam int NFR = 13;
  localparam logic [31:0] CAPS = {1'b1, 1'b1, 1'b1, 1'b0, HOSTCAL, DSTEP, 2'd3,  // bit31 WAITW, bit30 GEN, bit29 MX, bit27 HOSTCAL, bit26 STREAM, bit25 ARG, bit24 ACT_ROWS
                                  8'($clog2(PQ_WIN)),
                                  8'(TRACE_DEPTH != 0 ? $clog2(TRACE_DEPTH) : 0),
                                  CHASH, DSTEP, 1'b1, 1'b1, DDR_MTS != 0, HAS_I2C, HAS_TEMP,
                                  TRACE_DEPTH != 0};

  localparam logic [31:0] CAPS2 = {31'd0, COLL_GATHER || S > 1};   // bit0 GATHER

  logic [63:0] cycles;
  logic [31:0] n_brd, n_bwr, n_ard, n_awr, n_bst, scratch;
  logic        clear, snap;
  logic [LDP-1:0] ld_pend_q;               // LOAD written, loader not yet visibly busy
  wire         ld_pend = ld_start || (|ld_pend_q);
  always_ff @(posedge clk) ld_pend_q <= rst ? '0 : {ld_pend_q[LDP-2:0], ld_start};

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
      i2c_lo <= '0;
      for (int k = 0; k < 8; k++) arg[k] <= '0;
    end else begin
      if (s_bvalid && s_bready) s_bvalid <= 1'b0;
      if (tr_inc) tr_addr <= tr_addr + 1;
      if (w_go) begin
        s_bvalid <= 1'b1;
        case (s_awaddr[11:2])
          10'h002: begin
            run <= s_wdata[0] && (run || &calib);
            ld_start <= s_wdata[1] && !s_wdata[0] && !run && &calib;
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
          10'h088: i2c_lo <= s_wdata[3:0];
          10'h018, 10'h019, 10'h01A, 10'h01B, 10'h01C, 10'h01D, 10'h01E, 10'h01F:
            arg[s_awaddr[4:2]] <= s_wdata;
          default: ;
        endcase
      end
    end
  end

  // ---- per-run counters. Their clear is a register of its own, rst or a CLEAR write (in the
  // cycle clear rises; rst a cycle late), replicated by synthesis over their 224 flip-flops
  // (250 MHz, FK35: rst -> the rst || clear LUT -> n_bwr's reset pins, 1 level, 96% route, -0.45 ns)
  (* max_fanout = 64 *) logic pc_clr;
  always_ff @(posedge clk) pc_clr <= rst || (w_go && s_awaddr[11:2] == 10'h002 && s_wdata[2]);
  always_ff @(posedge clk) begin
    if (pc_clr) begin
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
  localparam int FW = CW > IW ? CW : IW;
  logic [15:0][31:0] ic_all;  // ICOUNT_s (S > 1)
  always_comb begin
    ic_all = '0;
    ic_all[S-1:0] = icount;
  end
  logic [FW-1:0] fr_inc [NFR];
  logic [63:0] fr [NFR], fr_s [NFR];
  logic [31:0] n_snap;
  // their reset, a register per counter (and n_snap's), next to it, a cycle after rst (250 MHz,
  // FK35: rst -> fr_s, fr, 0-1 levels, 98% route, -0.31 ns); keep: not merged back into one
  (* keep = "true", shreg_extract = "no" *) logic [NFR:0] fr_rst;
  always_ff @(posedge clk) fr_rst <= {(NFR + 1){rst}};
  always_ff @(posedge clk) begin
    fr_inc[0]  <= FW'(1);
    fr_inc[1]  <= FW'(run && !halted);
    fr_inc[2]  <= FW'(mxu_busy);
    fr_inc[3]  <= FW'(mxu_mac);
    fr_inc[4]  <= FW'(vpu_busy);
    fr_inc[5]  <= FW'(qnt_busy);
    fr_inc[6]  <= FW'(dma_busy);
    fr_inc[7]  <= FW'(tmem_deny);
    fr_inc[8]  <= FW'(dram_rd);
    fr_inc[9]  <= FW'(dram_wr);
    fr_inc[10] <= FW'(dram_wait);
    fr_inc[11] <= FW'(instr);
    fr_inc[12] <= FW'(mxu_starve);
    for (int k = 0; k < NFR; k++)
      if (fr_rst[k]) begin
        fr[k] <= '0; fr_s[k] <= '0; fr_inc[k] <= '0;
      end else begin
        fr[k] <= fr[k] + 64'(fr_inc[k]);
        if (snap) fr_s[k] <= fr[k];
      end
    if (fr_rst[NFR]) n_snap <= '0;
    else if (snap) n_snap <= n_snap + 1;
  end

  // ---- read channel: one read at a time; the address, a multiplexer per register group (r_g:
  // 0x000-0x01F, 0x020-0x03F, 0x040-0x07F the counters' shadows, 0x080-0x08F), the group's choice
  // (r_d) and the data registered (250 MHz, FK35: r_a -> r_d through one multiplexer over every
  // register and the default's compares, 6 levels, -0.43 ns). TRACE_HI's increment (tr_inc) comes
  // the cycle after its data is in r_g[3].
  logic [9:0]  r_a;
  logic [2:0]  r_p;
  logic [31:0] r_d;
  logic [31:0] r_g [4];
  logic [3:0]  r_gs;                    // r_a's group (one-hot; none: 0xDEADBEEF)
  assign s_arready = !s_rvalid && r_p == 3'b000;
  assign s_rresp = 2'b00;
  assign tr_inc = r_p[1] && r_a == 10'h085;
  always_ff @(posedge clk) begin
    if (rst) begin
      s_rvalid <= 1'b0;
      r_p <= 3'b000;
    end else begin
      if (s_rvalid && s_rready) s_rvalid <= 1'b0;
      r_p <= {r_p[1:0], s_arvalid && s_arready};
      if (r_p[2]) begin
        s_rvalid <= 1'b1;
        s_rdata <= r_d;
      end
    end
    if (s_arvalid && s_arready) r_a <= s_araddr[11:2];
    r_gs <= {r_a[9:4] == 6'h08, r_a[9:6] == 4'h1, r_a[9:5] == 5'h01, r_a[9:5] == 5'h00};
    case (r_a[4:0])
      5'h00: r_g[0] <= 32'h4F54_5055;
      5'h01: r_g[0] <= {16'(D), 8'(MCOLS), 8'(LANES)};
      5'h02: r_g[0] <= {31'd0, run};
      5'h03: r_g[0] <= {22'd0, ecc_ded, wait_to, run, calib, axi_err, wr_idle, ld_busy || ld_pend,
                        error, halted};
      5'h04: r_g[0] <= ld_addr;
      5'h05: r_g[0] <= ld_n;
      5'h06: r_g[0] <= cycles[31:0];
      5'h07: r_g[0] <= cycles[63:32];
      5'h08: r_g[0] <= icount[0];
      5'h09: r_g[0] <= n_brd;
      5'h0A: r_g[0] <= n_bwr;
      5'h0B: r_g[0] <= n_ard;
      5'h0C: r_g[0] <= n_awr;
      5'h0D: r_g[0] <= n_bst;
      5'h0E: r_g[0] <= scratch;
      5'h0F: r_g[0] <= 32'd5;
      5'h10: r_g[0] <= CAPS;
      5'h11: r_g[0] <= 32'(CORE_KHZ);
      5'h12: r_g[0] <= BUILD_ID;
      5'h13: r_g[0] <= {temp_v, 19'd0, temp};
      5'h14: r_g[0] <= n_snap;
      5'h15: r_g[0] <= 32'(DDR_MTS);
      5'h16: r_g[0] <= 32'(ACT_ROWS);
      5'h17: r_g[0] <= CAPS2;
      default: r_g[0] <= arg[r_a[2:0]];                 // 0x18 .. 0x1F
    endcase
    if (r_a[4]) begin                                   // 0x30 .. 0x3F: ICOUNT_s
      if (S > 1 && 32'(r_a[3:0]) < 32'(S)) r_g[1] <= ic_all[r_a[3:0]];
      else r_g[1] <= 32'hDEAD_BEEF;
    end else begin
      case (r_a[3:0])
        4'h0: r_g[1] <= 32'(S);
        4'h1: r_g[1] <= 32'(TMEM_WORDS);
        4'h2: r_g[1] <= 32'(IMEM_WORDS);
        4'h3: r_g[1] <= 32'(ACT_BLOCKS);
        4'h4: r_g[1] <= {8'd0, 8'(CH_LOG2), 8'(NCH), 8'(DRAM_LOG2)};
        4'h5: r_g[1] <= {6'd0, QUAD, SEQ_WC, 8'(ULANES), 8'(VPU_CL), 4'(TMEM_IMPL), 4'(MXU_IMPL)};
        4'h6: r_g[1] <= 32'(PLATFORM);
        4'h7: r_g[1] <= ARCH_HASH;
        default: r_g[1] <= 32'hDEAD_BEEF;
      endcase
    end
    if (r_a[5:1] < 5'(NFR)) r_g[2] <= r_a[0] ? fr_s[r_a[5:1]][63:32] : fr_s[r_a[5:1]][31:0];
    else r_g[2] <= 32'hDEAD_BEEF;
    case (r_a[3:0])
      4'h0: r_g[3] <= {28'd0, tr_busy, tr_stop, 1'b0, tr_en};
      4'h1: r_g[3] <= tr_count;
      4'h2: r_g[3] <= tr_drop;
      4'h3: r_g[3] <= tr_addr;
      4'h4: r_g[3] <= tr_rdata[31:0];
      4'h5: r_g[3] <= tr_rdata[63:32];
      4'h8: r_g[3] <= {28'd0, i2c_lo};
      4'h9: r_g[3] <= {27'd0, i2c_in};
      default: r_g[3] <= 32'hDEAD_BEEF;
    endcase
    r_d <= r_gs[0] ? r_g[0] : r_gs[1] ? r_g[1] : r_gs[2] ? r_g[2] : r_gs[3] ? r_g[3] : 32'hDEAD_BEEF;
  end
`ifndef SYNTHESIS
  // the trace ring's record is three cycles behind TRACE_ADDR (otpu_trace: the RAM's read and
  // two output registers): TRACE_LO / TRACE_HI take it three or more cycles after TRACE_ADDR moved
  logic [31:0] ta_q;
  int          ta_age;
  always_ff @(posedge clk) begin
    ta_q <= tr_addr;
    ta_age <= (tr_addr != ta_q) ? 1 : (ta_age < 8 ? ta_age + 1 : ta_age);
    if (!rst && r_p[0] && (r_a == 10'h084 || r_a == 10'h085) && (tr_addr != ta_q || ta_age < 3))
      $fatal(1, "otpu_ctrl: a trace record taken %0d cycles after TRACE_ADDR moved", ta_age);
  end
`endif
endmodule

// AXI4-Lite between a master and a slave that reset apart (otpu_board: the control registers;
// otpu_fpga_top_ld: the LiteDRAM core's CSR port). The block design's SmartConnect, the master,
// resets with XDMA's axi_aresetn (PERST#, and a PCIe hot reset or link down too); the control
// registers reset with the core reset (PERST#, not a hot reset), the LiteDRAM core's CSR port with
// the core's own clocks only. A slave that answered an access after its master's reset handed the
// fresh master a response it never asked for. One access at a time: a write's AW and W are taken
// together (as otpu_ctrl takes them), a read's AR, and given to the slave; its response is taken
// from the slave and held for the master. m_rst, the master's reset: the master's side forgets
// the access (a held response is dropped), but what the slave was given is completed with it (AW
// and W both, its response taken and dropped), so the slave never sees half an access or answers
// into the master's next one; a new access waits for that. s_rst, the slave's reset (the LiteDRAM
// core's sys reset: its PLL, or its ctrl_reset CSR): an access it had is answered SLVERR. An access
// costs three cycles more (the request and the response registered).
module otpu_axil_iso #(
  parameter int AW = 32
) (
  input  logic          clk,
  input  logic          m_rst,
  input  logic          s_rst,
  // the master's side
  input  logic [AW-1:0] m_awaddr,
  input  logic          m_awvalid,
  output logic          m_awready,
  input  logic [31:0]   m_wdata,
  input  logic [3:0]    m_wstrb,
  input  logic          m_wvalid,
  output logic          m_wready,
  output logic [1:0]    m_bresp,
  output logic          m_bvalid,
  input  logic          m_bready,
  input  logic [AW-1:0] m_araddr,
  input  logic          m_arvalid,
  output logic          m_arready,
  output logic [31:0]   m_rdata,
  output logic [1:0]    m_rresp,
  output logic          m_rvalid,
  input  logic          m_rready,
  // the slave's side
  output logic [AW-1:0] s_awaddr,
  output logic          s_awvalid,
  input  logic          s_awready,
  output logic [31:0]   s_wdata,
  output logic [3:0]    s_wstrb,
  output logic          s_wvalid,
  input  logic          s_wready,
  input  logic [1:0]    s_bresp,
  input  logic          s_bvalid,
  output logic          s_bready,
  output logic [AW-1:0] s_araddr,
  output logic          s_arvalid,
  input  logic          s_arready,
  input  logic [31:0]   s_rdata,
  input  logic [1:0]    s_rresp,
  input  logic          s_rvalid,
  output logic          s_rready
);
  // busy: an access is with the slave, or its response with the master (m_bvalid / m_rvalid);
  // wr: it is a write; drop: the master was reset since it gave it (its response goes nowhere).
  // Both resets are on at configuration (the valids clear)
  logic busy = 1'b0, wr = 1'b0, drop = 1'b0;
  // a write's ready waits for both its valids (as otpu_ctrl's), a read's does not (as otpu_ctrl's)
  assign m_awready = !s_rst && !m_rst && !busy && m_awvalid && m_wvalid;
  assign m_wready  = m_awready;
  assign m_arready = !s_rst && !m_rst && !busy && !(m_awvalid && m_wvalid);
  assign s_bready  = busy && wr && !m_bvalid;
  assign s_rready  = busy && !wr && !m_rvalid;
  always_ff @(posedge clk) begin
    if (m_awready) begin
      busy <= 1'b1; wr <= 1'b1; drop <= 1'b0;
      s_awvalid <= 1'b1; s_wvalid <= 1'b1;
      s_awaddr <= m_awaddr; s_wdata <= m_wdata; s_wstrb <= m_wstrb;
    end else if (m_arvalid && m_arready) begin
      busy <= 1'b1; wr <= 1'b0; drop <= 1'b0;
      s_arvalid <= 1'b1;
      s_araddr <= m_araddr;
    end
    if (s_awvalid && s_awready) s_awvalid <= 1'b0;
    if (s_wvalid && s_wready) s_wvalid <= 1'b0;
    if (s_arvalid && s_arready) s_arvalid <= 1'b0;
    // the slave's response: to the master, or dropped (the access is then over)
    if (s_bvalid && s_bready) begin
      if (drop || m_rst) busy <= 1'b0;
      else m_bvalid <= 1'b1;
      m_bresp <= s_bresp;
    end
    if (s_rvalid && s_rready) begin
      if (drop || m_rst) busy <= 1'b0;
      else m_rvalid <= 1'b1;
      m_rresp <= s_rresp; m_rdata <= s_rdata;
    end
    if ((m_bvalid && m_bready) || (m_rvalid && m_rready)) begin
      busy <= 1'b0; m_bvalid <= 1'b0; m_rvalid <= 1'b0;
    end
    // the master's reset: a held response is dropped (the access is over), one still with the
    // slave will be
    if (m_rst) begin
      m_bvalid <= 1'b0; m_rvalid <= 1'b0;
      if (m_bvalid || m_rvalid) busy <= 1'b0;
      else if (busy) drop <= 1'b1;
    end
    // the slave's reset: what it was given is gone; an access still with it is answered SLVERR
    // (the master is not left waiting), or over if its master was reset too
    if (s_rst) begin
      s_awvalid <= 1'b0; s_wvalid <= 1'b0; s_arvalid <= 1'b0;
      if (busy && !m_bvalid && !m_rvalid) begin
        if (drop || m_rst) busy <= 1'b0;
        else begin m_bvalid <= wr; m_rvalid <= !wr; m_bresp <= 2'b10; m_rresp <= 2'b10; end
      end
    end
  end
endmodule
