// Behavioural per-slice DRAM for simulation (the board version sits behind a DDR3/HBM
// controller). Port A: one read of the aligned 32 bytes holding word a_addr (the MXU's scale
// lines) or one byte-enabled word write per cycle; port SW: one
// byte-enabled word write per cycle (the quantizer's QST stores). Port B: one
// D-byte burst per cycle, read (the MXU's streamed operand, DMA loads) or word-masked write
// (DMA stores). Reads return LAT cycles later, in order; data is sampled at the request. A read
// with b_two (the MXU's int8 PAIR, otpu_mxu DUAL) also returns the next chunk in b_rdata2.
// A slice of more than 4 GiB (DAW > 32, otpu_pkg): the word addresses are AW bits, {REG, 30 bits},
// and the memory is NREG windows of WORDS words, one a 4 GiB region (2^(DAW - 32)), each decoded
// by the low bits of the address in its region (an address and the one 4 GiB above it are
// different words; inside a region the window repeats, as the one memory of DAW 32 does through
// its 4 GiB). The images and the dump are the windows in region order.
module otpu_dram #(
  parameter int WORDS = 1 << 18,
  parameter int D     = 32,
  parameter int LAT   = 8,
  parameter int SID   = 0,
  parameter int AW    = 32,           // word-address bits (otpu_pkg AWW)
  parameter int NREG  = 1             // windows (regions): 1, or 2^(DAW - 32)
) (
  input  logic              clk,
  input  logic              a_req,
  input  logic              a_we,
  input  logic [AW-1:0]     a_addr,     // word address
  input  logic [31:0]       a_wdata,
  input  logic [3:0]        a_be,
  output logic              a_rvalid,
  output logic [255:0]      a_rdata,    // words (a_addr & ~7) .. (a_addr | 7), the lowest first
  input  logic              sw_req,
  input  logic [AW-1:0]     sw_addr,
  input  logic [31:0]       sw_wdata,
  input  logic [3:0]        sw_be,
  input  logic              b_req,
  input  logic              b_tag,      // requester tag, returned with the read data
  input  logic              b_we,
  input  logic [D/4-1:0]    b_wmask,    // word enables for a burst write
  input  logic [D*8-1:0]    b_wdata,
  input  logic [AW-1:0]     b_addr,     // word address of the chunk
  input  logic              b_two,      // a read of the chunk and the next one
  output logic              b_rvalid,
  output logic              b_rtag,
  output logic [D*8-1:0]    b_rdata,
  output logic [D*8-1:0]    b_rdata2,   // b_two: the next chunk
  output logic              rd_idle,    // no read in flight
  input  logic              dump
);
  localparam int MW = $clog2(WORDS);
  localparam int CW = D / 4;
  localparam int NW = WORDS * NREG;
  initial if (NREG > 1 && ((1 << MW) != WORDS || MW > 30 || (1 << $clog2(NREG)) != NREG))
    $fatal(1, "otpu_dram: NREG %0d windows of %0d words (powers of two, at most 2^30 words)", NREG,
           WORDS);

  logic [31:0] mem [NW];
  // the memory word of word address a: its low MW bits (NREG 1, as before), or its region's window
  function automatic int mi(input logic [AW-1:0] a);
    if (NREG == 1) return int'(a[MW-1:0]);
    return int'((a >> 30) % NREG) * WORDS + int'(a[MW-1:0]);
  endfunction

  logic [LAT-1:0]    av, bv, bt;
  logic [255:0]      ad [LAT];
  logic [D*8-1:0]    bd [LAT], bd2 [LAT];

`ifndef SYNTHESIS
  // the host's writes during the run (WAITW's tests): <dir>/poke_<SID>.txt, lines "cycle word
  // value" (hex, in cycle order), each word written as the cycle count reaches its cycle
  localparam int NPOKE = 64;
  longint      pk_c [NPOKE];
  logic [63:0] pk_a [NPOKE];
  logic [31:0] pk_v [NPOKE];
  int          pk_n = 0, pk_i = 0;
  longint      pk_cyc = 0;
`endif
  always_ff @(posedge clk) begin
`ifndef SYNTHESIS
    pk_cyc <= pk_cyc + 1;
    if (pk_i < pk_n && pk_cyc >= pk_c[pk_i]) begin
      mem[mi(AW'(pk_a[pk_i]))] <= pk_v[pk_i];
      pk_i <= pk_i + 1;
    end
`endif
    if (a_req && a_we) begin
      for (int b = 0; b < 4; b++)
        if (a_be[b]) mem[mi(a_addr)][8*b +: 8] <= a_wdata[8*b +: 8];
    end
    if (sw_req) begin
      for (int b = 0; b < 4; b++)
        if (sw_be[b]) mem[mi(sw_addr)][8*b +: 8] <= sw_wdata[8*b +: 8];
    end
    av[0] <= a_req && !a_we;
    for (int w = 0; w < 8; w++) ad[0][32*w +: 32] <= mem[mi({a_addr[AW-1:3], 3'(w)})];
    bv[0] <= b_req && !b_we;
    bt[0] <= b_tag;
    for (int w = 0; w < CW; w++) bd[0][32*w +: 32] <= mem[mi(b_addr + AW'(w))];
    for (int w = 0; w < CW; w++) bd2[0][32*w +: 32] <= b_two ? mem[mi(b_addr + AW'(CW + w))] : '0;
    if (b_req && b_we)
      for (int w = 0; w < CW; w++) if (b_wmask[w]) mem[mi(b_addr + AW'(w))] <= b_wdata[32*w +: 32];
    for (int k = 1; k < LAT; k++) begin
      av[k] <= av[k-1];
      ad[k] <= ad[k-1];
      bv[k] <= bv[k-1];
      bt[k] <= bt[k-1];
      bd[k] <= bd[k-1];
      bd2[k] <= bd2[k-1];
    end
  end
  assign a_rvalid = av[LAT-1];
  assign a_rdata  = ad[LAT-1];
  assign b_rvalid = bv[LAT-1];
  assign b_rtag   = bt[LAT-1];
  assign b_rdata  = bd[LAT-1];
  assign b_rdata2 = bd2[LAT-1];
  assign rd_idle  = !(|av) && !(|bv);

`ifndef SYNTHESIS
  // Images are raw binary: dram_<SID>.bin holds big-endian words ($fread order) and the dump
  // dram_out_<SID>.bin is little-endian (the byte order of the DRAM itself).
  string dir;
  integer fd, nread;
  initial begin
    av = '0;
    bv = '0;
    bt = '0;
    for (int i = 0; i < NW; i++) mem[i] = '0;
    if ($value$plusargs("dir=%s", dir)) begin
      fd = $fopen($sformatf("%s/dram_%0d.bin", dir, SID), "rb");
      if (fd != 0) begin
        nread = $fread(mem, fd);
        $fclose(fd);
      end
      fd = $fopen($sformatf("%s/poke_%0d.txt", dir, SID), "r");
      if (fd != 0) begin
        while (pk_n < NPOKE && $fscanf(fd, "%h %h %h", pk_c[pk_n], pk_a[pk_n], pk_v[pk_n]) == 3)
          pk_n++;
        $fclose(fd);
      end
    end
  end
  always @(posedge clk) if (dump) begin
    fd = $fopen($sformatf("%s/dram_out_%0d.bin", dir, SID), "wb");
    for (int i = 0; i < NW; i++) $fwrite(fd, "%u", mem[i]);
    $fclose(fd);
  end
`endif
endmodule
