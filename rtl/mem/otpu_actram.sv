// ACT RAM: the MXU's stationary operand. ROWS rows x BLOCKS blocks x D int8, plus one fp32
// scale per (row, block). The quantizer writes up to LANES consecutive bytes (and one scale) per
// cycle; its byte index is LANES aligned, so the bytes sit in one block. With `w_dup` (QACT DUP)
// the bytes and the scale also go to row w_row + w_off (w_off: the QACT's row count; DUP rows
// stay below MCOLS). The MXU reads one block of MCOLS rows per cycle: rows g*MCOLS ..
// g*MCOLS+MCOLS-1 of group `r_grp` (ROWS > MCOLS: the MXU replays a weight chunk for each group
// of an MM), block r_blk, or r_blk2 for the rows in r_hi (MM PAIR: the odd blocks).
//
// Block RAM: one memory per column j (rows j, j+MCOLS, ...), GROUPS*BLOCKS words of D bytes
// with byte write enables, group-major. The read is registered (the MXU's first pipeline
// register) and advances with `ren`; a read of a block written at the same edge returns the old
// bytes. Writes land two cycles after they are presented: the quantizer's write is registered as
// it arrives (LANES bytes, their index and row, the scale), so it crosses from the quantizer in
// that narrow form and the lane placement (LANES -> D bytes) starts at a flip-flop by the block
// RAMs, then registered again in place (bytes, column selects, their groups). The quantizer
// reports done two cycles after its last write and the sequencer releases a dependent MM two
// cycles after that (otpu_quant), so a consumed read (r_use, the MXU's pop) never names a block
// whose write is still in flight: the simulation checks it.
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
  initial if (D % LANES != 0) $fatal(1, "otpu_actram: LANES must divide D");
  initial if (ROWS % MCOLS != 0 || (GROUPS > 1 && (MCOLS & (MCOLS - 1)) != 0))
    $fatal(1, "otpu_actram: ROWS must be a multiple of MCOLS, a power of two if larger");
  // a row's column and group
  function automatic int col(input logic [7:0] r);
    return (MCOLS > 1) ? int'(r) % MCOLS : 0;
  endfunction
  function automatic logic [GW-1:0] grp(input logic [7:0] r);
    return (GROUPS > 1) ? GW'(int'(r) / MCOLS) : '0;
  endfunction
  // the quantizer's write as it arrives: the long hop from the quantizer is this narrow register
  // (on 959b425 at 133.33 MHz the quantizer sat 600 RPM units from these block RAMs and the wide
  // write register below, fed straight from it, missed timing)
  logic [LANES-1:0]      we_i;
  logic [7:0]            w_row_i, w_off_i, s_row_i;
  logic [DW+BW-1:0]      w_idx_i;
  logic [LANES-1:0][7:0] w_data_i;
  logic                  w_dup_i, swe_i;
  logic [BW-1:0]         s_blk_i;
  logic [31:0]           s_data_i;
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
  end
  wire [7:0] w_row2 = w_row_i + w_off_i;      // the DUP copy's row
  wire [7:0] s_row2 = s_row_i + w_off_i;

  // the write, as a block address, byte enables and bytes in place (registered), and the
  // scale write (registered); per column: selected, and the group of the row it writes
  logic [BW-1:0]    wb;
  logic [D-1:0]     wbe;
  logic [D*8-1:0]   wd;
  logic [MCOLS-1:0] wsel, ssel;
  logic [GW-1:0]    wg [MCOLS], sg [MCOLS];
  logic [BW-1:0]    sb;
  logic [31:0]      sd;
  // the second register's inputs: per column, selected and the group of the row it writes
  logic [MCOLS-1:0] wsel_n, ssel_n;
  logic [GW-1:0]    wg_n [MCOLS], sg_n [MCOLS];
  always_comb
    for (int j = 0; j < MCOLS; j++) begin
      wsel_n[j] = ((col(w_row_i) == j) || (w_dup_i && col(w_row2) == j)) && (|we_i);
      ssel_n[j] = swe_i && ((col(s_row_i) == j) || (w_dup_i && col(s_row2) == j));
      wg_n[j] = (col(w_row_i) == j) ? grp(w_row_i) : grp(w_row2);
      sg_n[j] = (col(s_row_i) == j) ? grp(s_row_i) : grp(s_row2);
    end
  always_ff @(posedge clk) begin
    wb <= w_idx_i[DW +: BW];
    wbe <= '0;
    wd <= '0;
    for (int l = 0; l < LANES; l++) begin
      wbe[w_idx_i[DW-1:0] + DW'(l)] <= we_i[l];
      wd[8 * (w_idx_i[DW-1:0] + DW'(l)) +: 8] <= w_data_i[l];
    end
    wsel <= wsel_n;
    ssel <= ssel_n;
    for (int j = 0; j < MCOLS; j++) begin
      wg[j] <= wg_n[j];
      sg[j] <= sg_n[j];
    end
    sb <= s_blk_i;
    sd <= s_data_i;
  end
`ifndef SYNTHESIS
  initial begin we_i = '0; swe_i = 1'b0; wsel = '0; ssel = '0; end
`endif

  for (genvar j = 0; j < MCOLS; j++) begin : g_row
    logic [D*8-1:0] act [GROUPS*BLOCKS];
    logic [31:0]    asc [GROUPS*BLOCKS];
    logic [D*8-1:0] rd;
    logic [31:0]    rs;
    wire [BW-1:0] rb = r_hi[j] ? r_blk2[BW-1:0] : r_blk[BW-1:0];
    wire [GW+BW-1:0] ra = (GROUPS > 1) ? {r_grp[GW-1:0], rb} : (GW+BW)'(rb);
    wire [GW+BW-1:0] wa = (GROUPS > 1) ? {wg[j], wb} : (GW+BW)'(wb);
    wire [GW+BW-1:0] sa = (GROUPS > 1) ? {sg[j], sb} : (GW+BW)'(sb);
    always_ff @(posedge clk) begin
      for (int b = 0; b < D; b++)
        if (wsel[j] && wbe[b]) act[wa][8 * b +: 8] <= wd[8 * b +: 8];
      if (ren) rd <= act[ra];
    end
    always_ff @(posedge clk) begin
      if (ssel[j]) asc[sa] <= sd;
      if (ren) rs <= asc[ra];
    end
    assign r_data[j*D*8 +: D*8] = rd;
    assign r_scale[j*32 +: 32] = rs;
`ifndef SYNTHESIS
    initial for (int b = 0; b < GROUPS*BLOCKS; b++) begin act[b] = '0; asc[b] = '0; end
    // a consumed read of a block (or its scale) with a write in either register: it would
    // return the old bytes
    wire [GW+BW-1:0] wa_n = (GROUPS > 1) ? {wg_n[j], w_idx_i[DW +: BW]} : (GW+BW)'(w_idx_i[DW +: BW]);
    wire [GW+BW-1:0] sa_n = (GROUPS > 1) ? {sg_n[j], s_blk_i} : (GW+BW)'(s_blk_i);
    always @(posedge clk)
      if (ren && r_use && ((wsel[j] && wa == ra) || (wsel_n[j] && wa_n == ra) ||
                           (ssel[j] && sa == ra) || (ssel_n[j] && sa_n == ra)))
        $fatal(1, "otpu_actram: column %0d block %0d read while its write is in flight at %0t", j, ra, $time);
`endif
  end
endmodule
