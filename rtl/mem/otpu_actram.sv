// ACT RAM: the MXU's stationary operand. MCOLS rows x BLOCKS blocks x D int8, plus one fp32
// scale per (row, block). The quantizer writes up to LANES consecutive bytes (and one scale) per
// cycle; the MXU reads one block of every row per cycle.
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
  input  logic [15:0]            r_blk,
  output logic [MCOLS*D*8-1:0]   r_data,      // row j at [j*D*8 +: D*8], byte i at [+8i]
  output logic [MCOLS*32-1:0]    r_scale
);
  localparam int IW = $clog2(BLOCKS * D);
  localparam int BW = $clog2(BLOCKS);
  logic [7:0]  act [MCOLS][BLOCKS * D];
  logic [31:0] asc [MCOLS][BLOCKS];

  always_ff @(posedge clk) begin
    for (int l = 0; l < LANES; l++)
      if (we[l]) act[w_row[$clog2(MCOLS)-1:0]][IW'(w_idx + l)] <= w_data[l];
    if (swe) asc[s_row[$clog2(MCOLS)-1:0]][s_blk[BW-1:0]] <= s_data;
  end

  always_comb begin
    for (int j = 0; j < MCOLS; j++) begin
      for (int i = 0; i < D; i++)
        r_data[(j*D + i)*8 +: 8] = act[j][IW'(r_blk * D + i)];
      r_scale[j*32 +: 32] = asc[j][r_blk[BW-1:0]];
    end
  end

`ifndef SYNTHESIS
  initial begin
    for (int j = 0; j < MCOLS; j++) begin
      for (int i = 0; i < BLOCKS * D; i++) act[j][i] = '0;
      for (int b = 0; b < BLOCKS; b++) asc[j][b] = '0;
    end
  end
`endif
endmodule
