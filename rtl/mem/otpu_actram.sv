// ACT RAM: the MXU's stationary operand. ROWS rows x BLOCKS blocks x D int8, plus one 64-bit
// scale record per (row, block): {da3, da2, da1, da0, Sa}, the row's fp32 Sa and the block's four
// sub-block offsets (QACT, docs/isa.md "Quantization"). The quantizer writes up to LANES consecutive bytes (and one scale) per
// cycle; its byte index is LANES aligned, so the bytes sit in one block. With `w_dup` (QACT DUP)
// the bytes and the scale also go to row w_row + w_off (w_off: the QACT's row count; DUP rows
// stay below MCOLS). The MXU reads one block of MCOLS rows per cycle: rows g*MCOLS ..
// g*MCOLS+MCOLS-1 of group `r_grp` (ROWS > MCOLS: the MXU replays a weight chunk for each group
// of an MM), block r_blk, or r_blk2 for the rows in r_hi (MM PAIR: the odd blocks).
//
// Block RAM: one memory per column j (rows j, j+MCOLS, ...) and piece p (bytes p*PB ..
// p*PB+PB-1: four 64-bit block RAMs at PB = 32), GROUPS*BLOCKS words of PB bytes with byte
// write enables, group-major; with UCOLS, columns 0 .. UCOLS-1's in UltraRAM, the same shapes
// (a piece four 72-bit URAM288 wide, 16 a column at D = 128; the scale records stay in block
// RAM). The read is registered (the MXU's first pipeline register) and advances with the
// column's `ren` bit (otpu_mxu: the columns of the MM's rows); a read of a block written at the
// same edge returns the old bytes in block RAM (in UltraRAM: the tool's port order; no consumed
// read depends on it, below). Writes
// land three cycles after they are presented: the quantizer's write is registered as it arrives
// and again (LANES bytes, their index and row, the scale), so it crosses from the quantizer in
// that narrow form in two hops and the lane placement (LANES -> D bytes) starts at a flip-flop
// by the block RAMs, then registered again in place (bytes; the block address, column select
// and group once per column and piece). The quantizer reports done two cycles after its last
// write and the sequencer releases a dependent MM two cycles after that (otpu_quant), so a
// consumed read (r_use, the MXU's pop) never names a block whose write is still in flight: the
// simulation checks it.
//
// ASK (otpu_mxu AASK: the activations' chain-stage skew as read addresses): the bytes in ASK
// stages, byte i in stage i % ASK, each stage's D / ASK bytes a RAM of their own (a piece's PB
// bytes are ASK / NP stages: the same block RAMs), stage s read s cycles after its request (the
// block, the enables and r_use through a register line of ASK - 1 taps, g_sk), so r_data's byte i
// is the read of i % ASK cycles ago, as i % ASK registers after the read gave it. Every column
// reads the same block (r_hi low: no PAIR; the simulation checks it); the scale records are read
// at the request. A delayed read is still its MM's block: the MM completes only after that
// stage's products, and a QACT writing the block waits for it (otpu_seq); the simulation checks
// the write-in-flight rule at each stage's read too.
//
// DPORT = 1 (Config.QONCE: LDA, docs/isa.md "Quantize once"): a second writer, the DMA's (d_*:
// LANES bytes or a record a cycle, the quantizer's shapes, no DUP), taken at the input register
// in place of the quantizer's. The DMA writes only in a cycle with no quantizer write at its write
// stage (otpu_dma act_busy: the quantizer first), so the two never meet.
module otpu_actram #(
  parameter int D      = 32,
  parameter int MCOLS  = 8,
  parameter int ROWS   = MCOLS,     // a multiple of MCOLS, MCOLS a power of two if more
  parameter int BLOCKS = 64,
  parameter int LANES  = 8,
  parameter int UCOLS  = 0,         // columns in UltraRAM (0 .. MCOLS)
  parameter int ASK    = 0,         // the reads skewed by stage (above): 0, or the stages
  parameter bit DPORT  = 1'b0       // the DMA's write port (LDA, above)
) (
  input  logic                   clk,
  input  logic                   rst,         // synchronous: the write enables (below)
  input  logic [LANES-1:0]       we,          // lane l writes byte w_idx + l
  input  logic [7:0]             w_row,
  input  logic [31:0]            w_idx,       // byte index within the row: block*D + i
  input  logic [LANES-1:0][7:0]  w_data,
  input  logic                   w_dup,
  input  logic [7:0]             w_off,
  input  logic                   swe,
  input  logic [7:0]             s_row,
  input  logic [15:0]            s_blk,
  input  logic [63:0]            s_data,
  // DPORT: the DMA's writes (LDA)
  input  logic [LANES-1:0]       d_we,
  input  logic [7:0]             d_row,
  input  logic [31:0]            d_idx,
  input  logic [LANES-1:0][7:0]  d_data,
  input  logic                   d_swe,
  input  logic [7:0]             d_srow,
  input  logic [15:0]            d_sblk,
  input  logic [63:0]            d_sdata,
  input  logic [MCOLS-1:0]       ren,         // per column
  input  logic                   r_use,       // this edge's read is consumed (simulation checks)
  input  logic [15:0]            r_blk,
  input  logic [15:0]            r_blk2,
  input  logic [MCOLS-1:0]       r_hi,
  input  logic [7:0]             r_grp,
  output wire  [MCOLS*D*8-1:0]   r_data,      // row j at [j*D*8 +: D*8], byte i at [+8i]
  output wire  [MCOLS*64-1:0]    r_scale
);
  localparam int BW = $clog2(BLOCKS);
  localparam int DW = $clog2(D);
  localparam int GROUPS = ROWS / MCOLS;
  localparam int GW = (GROUPS > 1) ? $clog2(GROUPS) : 1;
  localparam int PB = (D > 32) ? 32 : D;      // bytes per piece
  localparam int NP = D / PB;
  localparam int RAW = $clog2(GROUPS * BLOCKS);  // the RAMs' address bits
  localparam int SB = (ASK > 0) ? D / ASK : PB;  // ASK: a stage's bytes, a piece's stages
  localparam int SPP = PB / SB;
  initial if (D % LANES != 0) $fatal(1, "otpu_actram: LANES must divide D");
  initial if (ASK != 0 && (ASK < 2 || D % ASK != 0 || ASK % NP != 0))
    $fatal(1, "otpu_actram: ASK %0d is not 0, or at least 2, dividing D, a multiple of %0d", ASK,
           NP);
  initial if (D % PB != 0) $fatal(1, "otpu_actram: D must be a multiple of 32 above 32");
  initial if (UCOLS < 0 || UCOLS > MCOLS) $fatal(1, "otpu_actram: UCOLS %0d not in 0 .. MCOLS", UCOLS);
  initial if (ROWS % MCOLS != 0 || (GROUPS > 1 && (MCOLS & (MCOLS - 1)) != 0))
    $fatal(1, "otpu_actram: ROWS must be a multiple of MCOLS, a power of two if larger");
  // a row's column and group
  function automatic int col(input logic [7:0] r);
    return (MCOLS > 1) ? int'(r) % MCOLS : 0;
  endfunction
  function automatic logic [GW-1:0] grp(input logic [7:0] r);
    return (GROUPS > 1) ? GW'(int'(r) / MCOLS) : '0;
  endfunction
  // a write to row r (with dup, also to row r + off) selects column j, in the group returned
  function automatic logic sel(input int j, input logic [7:0] r, input logic dup,
                               input logic [7:0] off);
    return (col(r) == j) || (dup && col(8'(r + off)) == j);
  endfunction
  function automatic logic [GW-1:0] sgrp(input int j, input logic [7:0] r, input logic [7:0] off);
    return (col(r) == j) ? grp(r) : grp(8'(r + off));
  endfunction
  // the quantizer's write as it arrives (_i) and again (_h): the long hop from the quantizer is
  // these two narrow registers (on 959b425 at 133.33 MHz the quantizer sat 600 RPM units from
  // these block RAMs and the wide write register below, fed straight from it, missed timing; on
  // 812bb01, with one, its replicas by the placement logic were the worst path, 96% route). The
  // enables (and the placement's selects below) take the reset; the bytes, rows and indices
  // are data, taken only under them
  (* shreg_extract = "no" *) logic [LANES-1:0]      we_i, we_h;
  (* shreg_extract = "no" *) logic [7:0]            w_row_i, w_off_i, s_row_i;
  (* shreg_extract = "no" *) logic [7:0]            w_row_h, w_off_h, s_row_h;
  (* shreg_extract = "no" *) logic [DW+BW-1:0]      w_idx_i, w_idx_h;
  (* shreg_extract = "no" *) logic [LANES-1:0][7:0] w_data_i, w_data_h;
  (* shreg_extract = "no" *) logic                  w_dup_i, swe_i, w_dup_h, swe_h;
  (* shreg_extract = "no" *) logic [BW-1:0]         s_blk_i, s_blk_h;
  (* shreg_extract = "no" *) logic [63:0]           s_data_i, s_data_h;
  wire dw_n = DPORT && (|d_we);              // DPORT: the DMA's bytes this cycle
  wire ds_n = DPORT && d_swe;                // ... its record
  always_ff @(posedge clk) begin
    we_i <= rst ? '0 : dw_n ? d_we : we;
    w_row_i <= dw_n ? d_row : w_row;
    w_idx_i <= dw_n ? d_idx[DW+BW-1:0] : w_idx[DW+BW-1:0];
    w_data_i <= dw_n ? d_data : w_data;
    w_dup_i <= !(dw_n || ds_n) && w_dup;
    w_off_i <= w_off;
    swe_i <= rst ? 1'b0 : ds_n || swe;
    s_row_i <= ds_n ? d_srow : s_row;
    s_blk_i <= ds_n ? d_sblk[BW-1:0] : s_blk[BW-1:0];
    s_data_i <= ds_n ? d_sdata : s_data;
    we_h <= rst ? '0 : we_i;
    w_row_h <= w_row_i;
    w_idx_h <= w_idx_i;
    w_data_h <= w_data_i;
    w_dup_h <= w_dup_i;
    w_off_h <= w_off_i;
    swe_h <= rst ? 1'b0 : swe_i;
    s_row_h <= s_row_i;
    s_blk_h <= s_blk_i;
    s_data_h <= s_data_i;
  end

  // the write, as byte enables and bytes in place (registered; the block address, column select
  // and group per column and piece in g_row), and the scale write (registered); per column:
  // selected, and the group of the row it writes
  logic [D-1:0]     wbe;
  logic [D*8-1:0]   wd;
  logic [MCOLS-1:0] ssel;
  logic [GW-1:0]    sg [MCOLS];
  logic [BW-1:0]    sb;
  logic [63:0]      sd;
  // the placement register's inputs: per column, selected and the group of the row it writes
  logic [MCOLS-1:0] wsel_n, ssel_n;
  logic [GW-1:0]    wg_n [MCOLS], sg_n [MCOLS];
  always_comb
    for (int j = 0; j < MCOLS; j++) begin
      wsel_n[j] = sel(j, w_row_h, w_dup_h, w_off_h) && (|we_h);
      ssel_n[j] = swe_h && sel(j, s_row_h, w_dup_h, w_off_h);
      wg_n[j] = sgrp(j, w_row_h, w_off_h);
      sg_n[j] = sgrp(j, s_row_h, w_off_h);
    end
  // the write's index is LANES aligned (otpu_quant), so byte i is lane i % LANES's, written when
  // the index is in i's lane group: a decode for the enables, and the lanes copied across D for
  // the bytes (the enables pick them), no byte rotator
  always_ff @(posedge clk) begin
    for (int i = 0; i < D; i++) begin
      wbe[i] <= we_h[i % LANES] && (32'(w_idx_h[DW-1:0]) / LANES == i / LANES);
      wd[8*i +: 8] <= w_data_h[i % LANES];
    end
    ssel <= rst ? '0 : ssel_n;
    for (int j = 0; j < MCOLS; j++) sg[j] <= sg_n[j];
    sb <= s_blk_h;
    sd <= s_data_h;
  end
`ifndef SYNTHESIS
  // the checks start once the reset has held two edges (registers start arbitrary: otpu_tmem)
  bit rst_p, rst_seen;
  initial begin rst_p = 1'b0; rst_seen = 1'b0; end
  always @(posedge clk) begin rst_p <= rst; if (rst && rst_p) rst_seen <= 1'b1; end
  wire chk = rst_seen && !rst;
  always @(posedge clk)
    if (chk && DPORT && ((|d_we && |we) || (d_swe && swe)))
      $fatal(1, "otpu_actram: the DMA and the quantizer write in one cycle");
  always @(posedge clk)
    if (chk && |we_h && 32'(w_idx_h[DW-1:0]) % LANES != 0)
      $fatal(1, "otpu_actram: write index %0d not LANES (%0d) aligned", w_idx_h[DW-1:0], LANES);
`endif

  // ASK: stage s's read at tap s of a line (tap 0 the request): the block, every column's (no
  // r_hi), and the columns' enables; r_use too in simulation, for the checks in g_row
  if (ASK > 0) begin : g_sk
    (* shreg_extract = "no" *) logic [ASK-1:0][GW+BW-1:0] sra;
    (* shreg_extract = "no" *) logic [ASK-1:0][MCOLS-1:0] sren;
    assign sra[0] = (GROUPS > 1) ? {r_grp[GW-1:0], r_blk[BW-1:0]} : (GW+BW)'(r_blk[BW-1:0]);
    assign sren[0] = ren;
    always_ff @(posedge clk)
      for (int k = 1; k < ASK; k++) begin
        sra[k] <= sra[k-1];
        sren[k] <= sren[k-1];
      end
`ifndef SYNTHESIS
    logic [ASK-1:0] suse;
    assign suse[0] = r_use;
    always @(posedge clk) for (int k = 1; k < ASK; k++) suse[k] <= suse[k-1];
    always @(posedge clk)
      if (chk && |ren && r_hi != '0) $fatal(1, "otpu_actram: ASK, and a read with r_hi %b", r_hi);
`endif
  end

  for (genvar j = 0; j < MCOLS; j++) begin : g_row
    localparam string ST = (j < UCOLS) ? "ultra" : "auto";
    logic [63:0]    rs;
    wire [BW-1:0] rb = r_hi[j] ? r_blk2[BW-1:0] : r_blk[BW-1:0];
    wire [GW+BW-1:0] ra = (GROUPS > 1) ? {r_grp[GW-1:0], rb} : (GW+BW)'(rb);
    wire [GW+BW-1:0] sa = (GROUPS > 1) ? {sg[j], sb} : (GW+BW)'(sb);
    // a piece's block RAMs take their write address from a copy of the block address (and
    // select, group) of their own: on 812bb01 one wb drove all MCOLS * D / 8 of them across
    // the systolic array's span, 6.7 ns of wire with no logic
    for (genvar p = 0; p < NP; p++) begin : g_pc
      (* keep *) logic [BW-1:0] wb;
      (* keep *) logic          wsel;
      (* keep *) logic [GW-1:0] wg;
      always_ff @(posedge clk) begin
        wb <= w_idx_h[DW +: BW];
        wsel <= rst ? 1'b0 : wsel_n[j];
        wg <= wg_n[j];
      end
      wire [GW+BW-1:0] wa = (GROUPS > 1) ? {wg, wb} : (GW+BW)'(wb);
      if (ASK == 0) begin : g_p
        logic [PB*8-1:0] rd;
        // the piece's bytes: a technology RAM with byte enables (rtl/tech)
        otpu_ram_1r1w_be #(.W(PB*8), .N(GROUPS*BLOCKS), .STYLE(ST)) u_act (
          .clk, .we(wsel), .wbe(wbe[p*PB +: PB]), .wa(wa[RAW-1:0]), .wd(wd[p*PB*8 +: PB*8]),
          .re(ren[j]), .ra(ra[RAW-1:0]), .rd);
        assign r_data[j*D*8 + p*PB*8 +: PB*8] = rd;
      end else begin : g_s
        // ASK: the piece's SPP stages p * SPP + t, a RAM each, read at their tap of g_sk; in
        // block RAM (a stage's SB bytes alone make a RAM the tool may put in LUTs)
        for (genvar t = 0; t < SPP; t++) begin : g_t
          localparam int S = p * SPP + t;
          logic [SB-1:0]   be;
          logic [SB*8-1:0] bd, rd;
          for (genvar k = 0; k < SB; k++) begin : g_k
            assign be[k] = wbe[S + ASK*k];
            assign bd[8*k +: 8] = wd[8*(S + ASK*k) +: 8];
            assign r_data[(j*D + S + ASK*k)*8 +: 8] = rd[8*k +: 8];
          end
          otpu_ram_1r1w_be #(.W(SB*8), .N(GROUPS*BLOCKS), .STYLE((j < UCOLS) ? "ultra" : "block"))
            u_act (.clk, .we(wsel), .wbe(be), .wa(wa[RAW-1:0]), .wd(bd), .re(g_sk.sren[S][j]),
                   .ra(g_sk.sra[S][RAW-1:0]), .rd);
        end
      end
    end
    // the column's block scale records
    otpu_ram_1r1w #(.W(64), .N(GROUPS*BLOCKS), .STYLE("auto")) u_asc (
      .clk, .we(ssel[j]), .wa(sa[RAW-1:0]), .wd(sd), .re(ren[j]), .ra(ra[RAW-1:0]), .rd(rs));
    assign r_scale[j*64 +: 64] = rs;
`ifndef SYNTHESIS
    // a consumed read of a block (or its scale) with a write in any register: it would return
    // the old bytes
    wire [GW+BW-1:0] wa = g_pc[0].wa;
    wire             wsel_1 = sel(j, w_row_i, w_dup_i, w_off_i) && (|we_i);
    wire             ssel_1 = swe_i && sel(j, s_row_i, w_dup_i, w_off_i);
    wire [GW+BW-1:0] wa_1 = (GROUPS > 1) ? {sgrp(j, w_row_i, w_off_i), w_idx_i[DW +: BW]}
                                         : (GW+BW)'(w_idx_i[DW +: BW]);
    wire [GW+BW-1:0] sa_1 = (GROUPS > 1) ? {sgrp(j, s_row_i, w_off_i), s_blk_i} : (GW+BW)'(s_blk_i);
    wire [GW+BW-1:0] wa_n = (GROUPS > 1) ? {wg_n[j], w_idx_h[DW +: BW]} : (GW+BW)'(w_idx_h[DW +: BW]);
    wire [GW+BW-1:0] sa_n = (GROUPS > 1) ? {sg_n[j], s_blk_h} : (GW+BW)'(s_blk_h);
    always @(posedge clk)
      if (chk && ren[j] && r_use && ((g_pc[0].wsel && wa == ra) || (wsel_n[j] && wa_n == ra) ||
                           (wsel_1 && wa_1 == ra) || (ssel[j] && sa == ra) ||
                           (ssel_n[j] && sa_n == ra) || (ssel_1 && sa_1 == ra)))
        $fatal(1, "otpu_actram: column %0d block %0d read while its write is in flight at %0t",
               j, ra, $time);
    if (ASK > 0) begin : g_skc
      // ASK: and stage s's read, at its tap (the scale records are read at the request)
      always @(posedge clk)
        for (int s = 1; s < ASK; s++)
          if (chk && g_sk.sren[s][j] && g_sk.suse[s] &&
              ((g_pc[0].wsel && wa == g_sk.sra[s]) || (wsel_n[j] && wa_n == g_sk.sra[s]) ||
               (wsel_1 && wa_1 == g_sk.sra[s])))
            $fatal(1, "otpu_actram: column %0d stage %0d reads block %0d in flight at %0t", j, s,
                   g_sk.sra[s], $time);
    end
`endif
  end
endmodule
