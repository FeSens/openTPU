// ACT RAM: the MXU's stationary operand. ROWS rows x BLOCKS blocks x D int8, plus one fp32
// scale per (row, block). The quantizer writes up to LANES consecutive bytes (and one scale) per
// cycle; its byte index is LANES aligned, so the bytes sit in one block. With `w_dup` (QACT DUP)
// the bytes and the scale also go to row w_row + w_off (w_off: the QACT's row count; DUP rows
// stay below MCOLS). The MXU reads one block of MCOLS rows per cycle: rows g*MCOLS ..
// g*MCOLS+MCOLS-1 of group `r_grp` (ROWS > MCOLS: the MXU replays a weight chunk for each group
// of an MM), block r_blk, or r_blk2 for the rows in r_hi (MM PAIR: the odd blocks).
//
// Block RAM: one memory per column j (rows j, j+MCOLS, ...) and piece p (bytes p*PB ..
// p*PB+PB-1: four 64-bit block RAMs at PB = 32), GROUPS*BLOCKS words of PB bytes with byte
// write enables, group-major. The read is registered (the MXU's first pipeline register) and
// advances with `ren`; a read of a block written at the same edge returns the old bytes. Writes
// land three cycles after they are presented: the quantizer's write is registered as it arrives
// and again (LANES bytes, their index and row, the scale), so it crosses from the quantizer in
// that narrow form in two hops and the lane placement (LANES -> D bytes) starts at a flip-flop
// by the block RAMs, then registered again in place (bytes; the block address, column select
// and group once per column and piece). The quantizer reports done two cycles after its last
// write and the sequencer releases a dependent MM two cycles after that (otpu_quant), so a
// consumed read (r_use, the MXU's pop) never names a block whose write is still in flight: the
// simulation checks it.
module otpu_actram #(
  parameter int D      = 32,
  parameter int MCOLS  = 8,
  parameter int ROWS   = MCOLS,     // a multiple of MCOLS, MCOLS a power of two if more
  parameter int BLOCKS = 64,
  parameter int LANES  = 8
) (
  input  logic                   clk,
  input  logic [LANES-1:0]       we,          // lane l writes byte w_idx + l
  input  logic [7:0]             w_row,
  input  logic [31:0]            w_idx,       // byte index within the row: block*D + i
  input  logic [LANES-1:0][7:0]  w_data,
  input  logic                   w_dup,
  input  logic [7:0]             w_off,
  input  logic                   swe,
  input  logic [7:0]             s_row,
  input  logic [15:0]            s_blk,
  input  logic [31:0]            s_data,
  input  logic                   ren,
  input  logic                   r_use,       // this edge's read is consumed (simulation checks)
  input  logic [15:0]            r_blk,
  input  logic [15:0]            r_blk2,
  input  logic [MCOLS-1:0]       r_hi,
  input  logic [7:0]             r_grp,
  output wire  [MCOLS*D*8-1:0]   r_data,      // row j at [j*D*8 +: D*8], byte i at [+8i]
  output wire  [MCOLS*32-1:0]    r_scale
);
  localparam int BW = $clog2(BLOCKS);
  localparam int DW = $clog2(D);
  localparam int GROUPS = ROWS / MCOLS;
  localparam int GW = (GROUPS > 1) ? $clog2(GROUPS) : 1;
  localparam int PB = (D > 32) ? 32 : D;      // bytes per piece
  localparam int NP = D / PB;
  initial if (D % LANES != 0) $fatal(1, "otpu_actram: LANES must divide D");
  initial if (D % PB != 0) $fatal(1, "otpu_actram: D must be a multiple of 32 above 32");
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
  // 812bb01, with one, its replicas by the placement logic were the worst path, 96% route)
  (* shreg_extract = "no" *) logic [LANES-1:0]      we_i, we_h;
  (* shreg_extract = "no" *) logic [7:0]            w_row_i, w_off_i, s_row_i;
  (* shreg_extract = "no" *) logic [7:0]            w_row_h, w_off_h, s_row_h;
  (* shreg_extract = "no" *) logic [DW+BW-1:0]      w_idx_i, w_idx_h;
  (* shreg_extract = "no" *) logic [LANES-1:0][7:0] w_data_i, w_data_h;
  (* shreg_extract = "no" *) logic                  w_dup_i, swe_i, w_dup_h, swe_h;
  (* shreg_extract = "no" *) logic [BW-1:0]         s_blk_i, s_blk_h;
  (* shreg_extract = "no" *) logic [31:0]           s_data_i, s_data_h;
  always_ff @(posedge clk) begin
    we_i <= we;
    w_row_i <= w_row;
    w_idx_i <= w_idx[DW+BW-1:0];
    w_data_i <= w_data;
    w_dup_i <= w_dup;
    w_off_i <= w_off;
    swe_i <= swe;
    s_row_i <= s_row;
    s_blk_i <= s_blk[BW-1:0];
    s_data_i <= s_data;
    we_h <= we_i;
    w_row_h <= w_row_i;
    w_idx_h <= w_idx_i;
    w_data_h <= w_data_i;
    w_dup_h <= w_dup_i;
    w_off_h <= w_off_i;
    swe_h <= swe_i;
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
  logic [31:0]      sd;
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
  always_ff @(posedge clk) begin
    wbe <= '0;
    wd <= '0;
    for (int l = 0; l < LANES; l++) begin
      wbe[w_idx_h[DW-1:0] + DW'(l)] <= we_h[l];
      wd[8 * (w_idx_h[DW-1:0] + DW'(l)) +: 8] <= w_data_h[l];
    end
    ssel <= ssel_n;
    for (int j = 0; j < MCOLS; j++) sg[j] <= sg_n[j];
    sb <= s_blk_h;
    sd <= s_data_h;
  end
`ifndef SYNTHESIS
  initial begin we_i = '0; swe_i = 1'b0; we_h = '0; swe_h = 1'b0; ssel = '0; end
`endif

  for (genvar j = 0; j < MCOLS; j++) begin : g_row
    logic [31:0]    asc [GROUPS*BLOCKS];
    logic [31:0]    rs;
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
      logic [PB*8-1:0] act [GROUPS*BLOCKS];
      logic [PB*8-1:0] rd;
      always_ff @(posedge clk) begin
        wb <= w_idx_h[DW +: BW];
        wsel <= wsel_n[j];
        wg <= wg_n[j];
      end
      wire [GW+BW-1:0] wa = (GROUPS > 1) ? {wg, wb} : (GW+BW)'(wb);
      always_ff @(posedge clk) begin
        for (int b = 0; b < PB; b++)
          if (wsel && wbe[p*PB + b]) act[wa][8 * b +: 8] <= wd[8 * (p*PB + b) +: 8];
        if (ren) rd <= act[ra];
      end
      assign r_data[j*D*8 + p*PB*8 +: PB*8] = rd;
`ifndef SYNTHESIS
      initial begin wsel = 1'b0; for (int b = 0; b < GROUPS*BLOCKS; b++) act[b] = '0; end
`endif
    end
    always_ff @(posedge clk) begin
      if (ssel[j]) asc[sa] <= sd;
      if (ren) rs <= asc[ra];
    end
    assign r_scale[j*32 +: 32] = rs;
`ifndef SYNTHESIS
    initial for (int b = 0; b < GROUPS*BLOCKS; b++) asc[b] = '0;
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
      if (ren && r_use && ((g_pc[0].wsel && wa == ra) || (wsel_n[j] && wa_n == ra) ||
                           (wsel_1 && wa_1 == ra) || (ssel[j] && sa == ra) ||
                           (ssel_n[j] && sa_n == ra) || (ssel_1 && sa_1 == ra)))
        $fatal(1, "otpu_actram: column %0d block %0d read while its write is in flight at %0t",
               j, ra, $time);
`endif
  end
endmodule
