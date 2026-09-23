// Behavioural per-slice DRAM for simulation (the board version sits behind a DDR3/HBM
// controller). Port A: one 32-bit word read or byte-enabled write per cycle. Port B: one
// D-byte burst per cycle, read (the MXU's streamed operand, DMA loads) or word-masked write
// (DMA stores). Reads return LAT cycles later, in order; data is sampled at the request.
module otpu_dram #(
  parameter int WORDS = 1 << 18,
  parameter int D     = 32,
  parameter int LAT   = 8,
  parameter int SID   = 0
) (
  input  logic              clk,
  input  logic              a_req,
  input  logic              a_we,
  input  logic [31:0]       a_addr,     // word address
  input  logic [31:0]       a_wdata,
  input  logic [3:0]        a_be,
  output logic              a_rvalid,
  output logic [31:0]       a_rdata,
  input  logic              b_req,
  input  logic              b_tag,      // requester tag, returned with the read data
  input  logic              b_we,
  input  logic [D/4-1:0]    b_wmask,    // word enables for a burst write
  input  logic [D*8-1:0]    b_wdata,
  input  logic [31:0]       b_addr,     // word address of the chunk
  output logic              b_rvalid,
  output logic              b_rtag,
  output logic [D*8-1:0]    b_rdata,
  input  logic              dump
);
  localparam int AW = $clog2(WORDS);
  localparam int CW = D / 4;

  logic [31:0] mem [WORDS];

  logic [LAT-1:0]    av, bv, bt;
  logic [31:0]       ad [LAT];
  logic [D*8-1:0]    bd [LAT];

  always_ff @(posedge clk) begin
    if (a_req && a_we) begin
      for (int b = 0; b < 4; b++)
        if (a_be[b]) mem[a_addr[AW-1:0]][8*b +: 8] <= a_wdata[8*b +: 8];
    end
    av[0] <= a_req && !a_we;
    ad[0] <= mem[a_addr[AW-1:0]];
    bv[0] <= b_req && !b_we;
    bt[0] <= b_tag;
    for (int w = 0; w < CW; w++) bd[0][32*w +: 32] <= mem[AW'(b_addr + w)];
    if (b_req && b_we)
      for (int w = 0; w < CW; w++) if (b_wmask[w]) mem[AW'(b_addr + w)] <= b_wdata[32*w +: 32];
    for (int k = 1; k < LAT; k++) begin
      av[k] <= av[k-1];
      ad[k] <= ad[k-1];
      bv[k] <= bv[k-1];
      bt[k] <= bt[k-1];
      bd[k] <= bd[k-1];
    end
  end
  assign a_rvalid = av[LAT-1];
  assign a_rdata  = ad[LAT-1];
  assign b_rvalid = bv[LAT-1];
  assign b_rtag   = bt[LAT-1];
  assign b_rdata  = bd[LAT-1];

`ifndef SYNTHESIS
  // Images are raw binary: dram_<SID>.bin holds big-endian words ($fread order) and the dump
  // dram_out_<SID>.bin is little-endian (the byte order of the DRAM itself).
  string dir;
  integer fd, nread;
  initial begin
    av = '0;
    bv = '0;
    bt = '0;
    for (int i = 0; i < WORDS; i++) mem[i] = '0;
    if ($value$plusargs("dir=%s", dir)) begin
      fd = $fopen($sformatf("%s/dram_%0d.bin", dir, SID), "rb");
      if (fd != 0) begin
        nread = $fread(mem, fd);
        $fclose(fd);
      end
    end
  end
  always @(posedge clk) if (dump) begin
    fd = $fopen($sformatf("%s/dram_out_%0d.bin", dir, SID), "wb");
    for (int i = 0; i < WORDS; i++) $fwrite(fd, "%u", mem[i]);
    $fclose(fd);
  end
`endif
endmodule
