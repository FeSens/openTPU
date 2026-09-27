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
// bytes.
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
  // a row's column and group; the word address {group, block}
  function automatic int col(input logic [7:0] r);
    return (MCOLS > 1) ? int'(r) % MCOLS : 0;
  endfunction
  function automatic logic [GW+BW-1:0] wa(input logic [7:0] r, input logic [BW-1:0] b);
    return (GROUPS > 1) ? {GW'(int'(r) / MCOLS), b} : (GW+BW)'(b);
  endfunction
  wire [7:0] w_row2 = w_row + w_off;          // the DUP copy's row
  wire [7:0] s_row2 = s_row + w_off;

  // the write, as a block address, byte enables and bytes in place
  wire [BW-1:0]   wb = w_idx[DW +: BW];
  logic [D-1:0]   wbe;
  logic [D*8-1:0] wd;
  always_comb begin
    wbe = '0;
    wd = '0;
    for (int l = 0; l < LANES; l++) begin
      wbe[w_idx[DW-1:0] + DW'(l)] = we[l];
      wd[8 * (w_idx[DW-1:0] + DW'(l)) +: 8] = w_data[l];
    end
  end

  for (genvar j = 0; j < MCOLS; j++) begin : g_row
    logic [D*8-1:0] act [GROUPS*BLOCKS];
    logic [31:0]    asc [GROUPS*BLOCKS];
    logic [D*8-1:0] rd;
    logic [31:0]    rs;
    wire  sel1 = (col(w_row) == j);
    wire  sel = sel1 || (w_dup && col(w_row2) == j);
    wire  ssel1 = (col(s_row) == j);
    wire  ssel = ssel1 || (w_dup && col(s_row2) == j);
    wire [GW+BW-1:0] waddr = sel1 ? wa(w_row, wb) : wa(w_row2, wb);
    wire [GW+BW-1:0] saddr = ssel1 ? wa(s_row, s_blk[BW-1:0]) : wa(s_row2, s_blk[BW-1:0]);
    wire [BW-1:0] rb = r_hi[j] ? r_blk2[BW-1:0] : r_blk[BW-1:0];
    wire [GW+BW-1:0] ra = (GROUPS > 1) ? {r_grp[GW-1:0], rb} : (GW+BW)'(rb);
    always_ff @(posedge clk) begin
      for (int b = 0; b < D; b++)
        if (sel && wbe[b]) act[waddr][8 * b +: 8] <= wd[8 * b +: 8];
      if (ren) rd <= act[ra];
    end
    always_ff @(posedge clk) begin
      if (swe && ssel) asc[saddr] <= s_data;
      if (ren) rs <= asc[ra];
    end
    assign r_data[j*D*8 +: D*8] = rd;
    assign r_scale[j*32 +: 32] = rs;
`ifndef SYNTHESIS
    initial for (int b = 0; b < GROUPS*BLOCKS; b++) begin act[b] = '0; asc[b] = '0; end
`endif
  end
endmodule
