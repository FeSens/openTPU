// ACT RAM: the MXU's stationary operand. MCOLS rows x BLOCKS blocks x D int8, plus one fp32
// scale per (row, block). The quantizer writes up to LANES consecutive bytes (and one scale) per
// cycle; its byte index is LANES aligned, so the bytes sit in one block. The MXU reads one block
// of every row per cycle.
//
// Block RAM: one memory per row, BLOCKS words of D bytes with byte write enables. The read is
// registered (the MXU's first pipeline register) and advances with `ren`; a read of a block
// written at the same edge returns the old bytes.
module otpu_actram #(
  parameter int D      = 32,
  parameter int MCOLS  = 8,
  parameter int BLOCKS = 64,
  parameter int LANES  = 8
) (
  input  logic                   clk,
  input  logic [LANES-1:0]       we,          // lane l writes byte w_idx + l
  input  logic [7:0]             w_row,
  input  logic [31:0]            w_idx,       // byte index within the row: block*D + i
  input  logic [LANES-1:0][7:0]  w_data,
  input  logic                   swe,
  input  logic [7:0]             s_row,
  input  logic [15:0]            s_blk,
  input  logic [31:0]            s_data,
  input  logic                   ren,
  input  logic [15:0]            r_blk,
  output wire  [MCOLS*D*8-1:0]   r_data,      // row j at [j*D*8 +: D*8], byte i at [+8i]
  output wire  [MCOLS*32-1:0]    r_scale
);
  localparam int BW = $clog2(BLOCKS);
  localparam int DW = $clog2(D);
  localparam int RW = (MCOLS > 1) ? $clog2(MCOLS) : 1;
  initial if (D % LANES != 0) $fatal(1, "otpu_actram: LANES must divide D");

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
    logic [D*8-1:0] act [BLOCKS];
    logic [31:0]    asc [BLOCKS];
    logic [D*8-1:0] rd;
    logic [31:0]    rs;
    wire  sel = (w_row[RW-1:0] == RW'(j));
    always_ff @(posedge clk) begin
      for (int b = 0; b < D; b++)
        if (sel && wbe[b]) act[wb][8 * b +: 8] <= wd[8 * b +: 8];
      if (ren) rd <= act[r_blk[BW-1:0]];
    end
    always_ff @(posedge clk) begin
      if (swe && s_row[RW-1:0] == RW'(j)) asc[s_blk[BW-1:0]] <= s_data;
      if (ren) rs <= asc[r_blk[BW-1:0]];
    end
    assign r_data[j*D*8 +: D*8] = rd;
    assign r_scale[j*32 +: 32] = rs;
`ifndef SYNTHESIS
    initial for (int b = 0; b < BLOCKS; b++) begin act[b] = '0; asc[b] = '0; end
`endif
  end
endmodule
