// Behavioural per-slice DRAM for simulation (the board version sits behind a DDR3/HBM
// controller). Port A: one read of the aligned AB bytes holding word a_addr (the MXU's scale
// lines; AB 32, or a channel beat at WG > 1) or one byte-enabled word write per cycle; port SW: one
// byte-enabled write of SWW words per cycle (the quantizer's QST stores: the aligned SWW words
// holding word sw_addr). Port B: one
// D-byte burst per cycle, read (the MXU's streamed operand, DMA loads) or word-masked write
// (DMA stores). Port W (P > D: otpu_native_dram's, P = NCH x CB): one P-byte read per cycle (a
// wide chunk), its CB-byte lanes as the adapter's (CHASH: lane c the chunk's beat c ^ h, h the
// XOR fold of the chunk's index into log2(P / CB) bits; else in order). Reads return LAT cycles
// later, in order; data is sampled at the request. The first WORDS words are dense (an address
// above them wraps at AW 32); at AW > 32 every word at or above WORDS is in a sparse store (see
// the images below), so an address that loses a bit above 31 reads or writes the wrong word. In
// reset (sys_rst) it takes no request, as a controller in reset does: the slices' requests while
// their own resets take hold (registers start arbitrary) neither write nor return.
module otpu_dram #(
  parameter int WORDS = 1 << 18,
  parameter int D     = 32,
  parameter int LAT   = 8,
  parameter int SID   = 0,
  parameter int P     = D,              // port W's wide chunk (bytes; none unless P > D)
  parameter int AB    = 32,             // port A's read bytes
  parameter int CB    = P,              // port W's lanes (bytes) and their hash (otpu_native_dram's)
  parameter bit CHASH = 1'b0,
  parameter int AW    = 32,             // byte address bits (otpu_seq AW)
  parameter int SWW   = 1,              // port SW's words (otpu_quant SWW)
  parameter int WA    = (AW > 34) ? AW - 2 : 32   // the ports' word address bits (derived)
) (
  input  logic              clk,
  input  logic              rst,
  input  logic              a_req,
  input  logic              a_we,
  input  logic [WA-1:0]     a_addr,     // word address
  input  logic [31:0]       a_wdata,
  input  logic [3:0]        a_be,
  output logic              a_rvalid,
  output logic [AB*8-1:0]   a_rdata,    // words (a_addr & ~(AB/4-1)) .. (a_addr | AB/4-1), the lowest first
  input  logic              sw_req,
  input  logic [WA-1:0]     sw_addr,
  input  logic [32*SWW-1:0] sw_wdata,
  input  logic [4*SWW-1:0]  sw_be,
  input  logic              b_req,
  input  logic              b_tag,      // requester tag, returned with the read data
  input  logic              b_we,
  input  logic [D/4-1:0]    b_wmask,    // word enables for a burst write
  input  logic [D*8-1:0]    b_wdata,
  input  logic [WA-1:0]     b_addr,     // word address of the chunk
  output logic              b_rvalid,
  output logic              b_rtag,
  output logic [D*8-1:0]    b_rdata,
  input  logic              w_req,
  input  logic [WA-1:0]     w_addr,     // word address of the wide chunk
  output logic              w_rvalid,
  output logic [P*8-1:0]    w_rdata,
  output logic              rd_idle,    // no read in flight
  input  logic              dump
);
  localparam int MW = $clog2(WORDS);
  localparam int CW = D / 4;
  localparam int AL = $clog2(AB / 4);               // word-address bits in a port A read

  logic [31:0] mem [WORDS];
  // AW > 32: the words at or above WORDS (hv), written on the falling edge from the cycle's
  // writes (hq, in the order the dense words take theirs), so every rising edge reads them as
  // the dense words read (the writes of the cycles before)
  typedef struct { longint a; logic [31:0] d; logic [3:0] be; } hw_t;
  logic [31:0] hv [longint];
  hw_t         hq [$];
  function automatic bit hi_of(input logic [WA-1:0] a);
    return AW > 32 && longint'(a) >= longint'(WORDS);
  endfunction
  function automatic logic [31:0] rd(input logic [WA-1:0] a);
    if (hi_of(a)) return hv.exists(longint'(a)) ? hv[longint'(a)] : '0;
    return mem[a[MW-1:0]];
  endfunction
  always @(negedge clk)
    while (hq.size() > 0) begin
      hw_t w;
      logic [31:0] v;
      w = hq.pop_front();
      v = hv.exists(w.a) ? hv[w.a] : '0;
      for (int b = 0; b < 4; b++) if (w.be[b]) v[8*b +: 8] = w.d[8*b +: 8];
      hv[w.a] = v;
    end

  logic [LAT-1:0]    av, bv, bt;
  logic [AB*8-1:0]   ad [LAT];
  logic [D*8-1:0]    bd [LAT];

`ifndef SYNTHESIS
  // the host's writes during the run (WAITW's tests): <dir>/poke_<SID>.txt, lines "cycle word
  // value" (hex, in cycle order), each word written as the cycle count reaches its cycle
  localparam int NPOKE = 64;
  longint      pk_c [NPOKE];
  logic [WA-1:0] pk_a [NPOKE];
  logic [31:0] pk_v [NPOKE];
  int          pk_n = 0, pk_i = 0;
  longint      pk_cyc = 0;
`endif
  always_ff @(posedge clk) begin
`ifndef SYNTHESIS
    pk_cyc <= pk_cyc + 1;
    if (pk_i < pk_n && pk_cyc >= pk_c[pk_i]) begin
      if (hi_of(pk_a[pk_i])) hq.push_back('{longint'(pk_a[pk_i]), pk_v[pk_i], 4'hF});
      else mem[pk_a[pk_i][MW-1:0]] <= pk_v[pk_i];
      pk_i <= pk_i + 1;
    end
`endif
    if (!rst && a_req && a_we) begin
      if (hi_of(a_addr)) hq.push_back('{longint'(a_addr), a_wdata, a_be});
      else
        for (int b = 0; b < 4; b++)
          if (a_be[b]) mem[a_addr[MW-1:0]][8*b +: 8] <= a_wdata[8*b +: 8];
    end
    if (!rst && sw_req)
      for (int w = 0; w < SWW; w++) begin
        logic [WA-1:0] sa;
        sa = (sw_addr & ~WA'(SWW - 1)) | WA'(w);
        if (hi_of(sa)) begin
          if (sw_be[4*w +: 4] != 0) hq.push_back('{longint'(sa), sw_wdata[32*w +: 32], sw_be[4*w +: 4]});
        end else
          for (int b = 0; b < 4; b++)
            if (sw_be[4*w + b]) mem[sa[MW-1:0]][8*b +: 8] <= sw_wdata[32*w + 8*b +: 8];
      end
    av[0] <= !rst && a_req && !a_we;
    for (int w = 0; w < AB / 4; w++) ad[0][32*w +: 32] <= rd({a_addr[WA-1:AL], AL'(w)});
    bv[0] <= !rst && b_req && !b_we;
    bt[0] <= b_tag;
    for (int w = 0; w < CW; w++) bd[0][32*w +: 32] <= rd(WA'(b_addr + w));
    if (!rst && b_req && b_we)
      for (int w = 0; w < CW; w++)
        if (b_wmask[w]) begin
          if (hi_of(WA'(b_addr + w)))
            hq.push_back('{longint'(WA'(b_addr + w)), b_wdata[32*w +: 32], 4'hF});
          else mem[MW'(b_addr + w)] <= b_wdata[32*w +: 32];
        end
    for (int k = 1; k < LAT; k++) begin
      av[k] <= !rst && av[k-1];
      ad[k] <= ad[k-1];
      bv[k] <= !rst && bv[k-1];
      bt[k] <= bt[k-1];
      bd[k] <= bd[k-1];
    end
  end
  assign a_rvalid = av[LAT-1];
  assign a_rdata  = ad[LAT-1];
  assign b_rvalid = bv[LAT-1];
  assign b_rtag   = bt[LAT-1];
  assign b_rdata  = bd[LAT-1];
  logic [LAT-1:0] wv;
  if (P > D) begin : g_w
    localparam int NL = P / CB, CL = (NL > 1) ? $clog2(NL) : 1, PWW = $clog2(P / 4);
    logic [P*8-1:0] wd [LAT];
    logic [CL-1:0] wh;                              // the chunk's hash (otpu_native_dram fold)
    always_comb begin
      wh = '0;
      if (CHASH && NL > 1) for (int i = PWW; i < WA; i++) wh[(i - PWW) % CL] ^= w_addr[i];
    end
    always_ff @(posedge clk) begin
      wv[0] <= !rst && w_req;
      for (int c = 0; c < NL; c++)
        for (int w = 0; w < CB / 4; w++)
          wd[0][CB*8*c + 32*w +: 32] <= rd(WA'(w_addr + (CB / 4) * 32'(CL'(c) ^ wh) + w));
      for (int k = 1; k < LAT; k++) begin
        wv[k] <= !rst && wv[k-1];
        wd[k] <= wd[k-1];
      end
    end
    assign w_rdata = wd[LAT-1];
  end else begin : g_nw
    assign wv = '0;
    assign w_rdata = '0;
  end
  assign w_rvalid = wv[LAT-1];
  assign rd_idle  = !(|av) && !(|bv) && !(|wv);

`ifndef SYNTHESIS
  // Images are raw binary: dram_<SID>.bin holds big-endian words ($fread order) and the dump
  // dram_out_<SID>.bin is little-endian (the byte order of the DRAM itself). At AW > 32 the
  // sparse words: dram_hi_<SID>.txt in (absent: none; a word not in it reads 0), and
  // dram_hi_out_<SID>.txt out (every word held there, loaded or written), lines "<word address>
  // <word>" in hex, the word's byte j (byte address 4 x address + j) at its bits 8j + 7 .. 8j.
  string dir;
  integer fd, nread;
  longint hw_a;
  logic [31:0] hw_d;
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
      fd = AW > 32 ? $fopen($sformatf("%s/dram_hi_%0d.txt", dir, SID), "r") : 0;
      if (fd != 0) begin
        while ($fscanf(fd, "%h %h", hw_a, hw_d) == 2)
          if (hw_a < longint'(WORDS) || hw_a >= (longint'(1) << WA))
            $fatal(1, "otpu_dram: dram_hi_%0d.txt word %h outside %h .. %h", SID, hw_a, WORDS,
                   (longint'(1) << WA) - 1);
          else hv[hw_a] = hw_d;
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
    for (int i = 0; i < WORDS; i++) $fwrite(fd, "%u", mem[i]);
    $fclose(fd);
    if (AW > 32) begin
      fd = $fopen($sformatf("%s/dram_hi_out_%0d.txt", dir, SID), "w");
      foreach (hv[a]) $fwrite(fd, "%0h %h\n", a, hv[a]);
      $fclose(fd);
    end
  end
`endif
endmodule
