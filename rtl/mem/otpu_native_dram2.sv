// A slice's DRAM ports on four memory channels (DUAL): two otpu_native_dram adapters, each on two
// channels, the slice's D-byte chunks split between them by parity. Chunk m of the slice's byte
// space is chunk m >> 1 of adapter p = m & 1, on channels 2p and 2p + 1, so a chunk is still one
// beat on each of two channels and a pair of adjacent chunks one beat on each of the four: the MXU's
// int8 PAIR stream (otpu_mxu DUAL) reads 256 bytes a cycle. The host applies the same map
// (opentpu/host/board.py: a slice's channel 2p + c holds adapter p's channel c).
//
// Port B: a single chunk (read or write) goes to its adapter; a pair (b_two, reads only: the
// chunk at b_addr and the next one) to both, the first chunk's adapter for b_rdata, the other's
// for b_rdata2. b_rdy takes both adapters' (registers), so a request never waits for one alone.
// Port A (one word or the 32-byte line of a word: one chunk) and port SW (one word) go to the
// word's adapter. Each adapter returns its reads in order; the reads leave here in the slice's
// order: an order FIFO per port names the adapter(s) of every read taken, and its head's
// adapters' data is taken (b_rrdy / a_rrdy) when all of it is there (b_ravail / a_ravail: the
// adapters hold it until then). A slice address's two adapters never hold the same byte, so
// the adapters' own ordering (otpu_native_dram "Ordering") is the slice's; wr_idle and rd_idle
// are both adapters'. AW, BAW: the slice's word-address and the adapters' channel-beat widths
// (otpu_native_dram); the adapters' BAW covers the chunk index without its parity bit: a
// 2^DAW-byte slice on four channels has BAW = DAW - 8.
module otpu_native_dram2 #(
  parameter int D = 128,
  parameter bit CHASH = 1'b1,
  parameter int RD = 128,                            // B read beats in flight per channel
  parameter int OQ = 512,                            // reads in flight a port (order FIFO), 2^k
  parameter int AW = 32,                             // the slice's word addresses (otpu_pkg AWW)
  parameter int BAW = 25,                            // the channel beat address bits
  parameter int QMEM = 0                             // the adapters' queue memories (otpu_native_dram)
) (
  input  logic              clk,
  input  logic              rst,
  input  logic              a_flush,
  output logic              a_rdy_x,
  input  logic              a_req_x,
  input  logic              a_we_x,
  input  logic [AW-1:0]     a_addr_x,   // word address
  input  logic [31:0]       a_wdata_x,
  input  logic [3:0]        a_be_x,
  output logic              a_rvalid,
  output logic [255:0]      a_rdata,
  output logic              sw_rdy,
  input  logic              sw_req,
  input  logic [AW-1:0]     sw_addr,
  input  logic [31:0]       sw_wdata,
  input  logic [3:0]        sw_be,
  output logic              b_rdy,
  input  logic              b_req,
  input  logic              b_tag,
  input  logic              b_we,
  input  logic [D/4-1:0]    b_wmask,
  input  logic [D*8-1:0]    b_wdata,
  input  logic [AW-1:0]     b_addr,     // word address (chunk aligned)
  input  logic              b_par,      // ^b_addr[AW-1:5]
  input  logic              b_two,      // a read of the chunk and the next one
  output logic              b_rvalid,
  output logic              b_rtag,
  output logic [D*8-1:0]    b_rdata,
  output logic [D*8-1:0]    b_rdata2,   // b_two: the next chunk
  output logic              wr_idle,
  output logic              rd_idle,
  output logic [3:0]        n_cvalid,
  input  logic [3:0]        n_cready,
  output logic [3:0]        n_cwe,
  output logic [3:0][BAW-1:0] n_caddr,
  output logic [3:0]        n_wvalid,
  input  logic [3:0]        n_wready,
  output logic [3:0][511:0] n_wdata,
  output logic [3:0][63:0]  n_wmask,
  input  logic [3:0]        n_rvalid,
  input  logic [3:0][511:0] n_rdata,
  input  logic [3:0][15:0]  n_wdone
);
  initial if (D != 128) $fatal(1, "otpu_native_dram2: D must be 128");
  localparam int CB = $clog2(D / 4);                 // the chunk's word bits (5): addr[CB] its parity
  localparam int OW = $clog2(OQ);

  // an address in adapter p: the chunk index without its parity bit
  function automatic logic [AW-1:0] inner(input logic [AW-1:0] a);
    return {1'b0, a[AW-1:CB+1], a[CB-1:0]};
  endfunction

  logic [1:0]       p_a_rdy, p_sw_rdy, p_b_rdy, p_wr_idle, p_rd_idle;
  logic [1:0]       p_a_req, p_sw_req, p_b_req;
  logic [1:0][AW-1:0] p_b_addr;
  logic [1:0]       p_b_par;
  logic [1:0]       p_a_rvalid, p_b_rvalid, p_b_rtag;
  logic [1:0][255:0] p_a_rdata;
  logic [1:0][D*8-1:0] p_b_rdata;
  logic [1:0]       p_b_ravail, p_b_rrdy, p_a_ravail, p_a_rrdy;

  // ---- requests
  wire       bp = b_addr[CB];                        // the first chunk's adapter
  wire [AW-1:0] bi = inner(b_addr);
  // the pair's second chunk: the same inner chunk in adapter 1 (bp 0), the next one in adapter 0
  wire [AW-1:0] bi2 = bp ? bi + AW'(D / 4) : bi;
  wire       bpar = b_par ^ b_addr[CB];              // ^bi[AW-1:5] (b_par: ^b_addr[AW-1:5])
  wire       bpar2 = bp ? ^bi2[AW-1:CB] : bpar;
  assign b_rdy = &p_b_rdy;
  assign a_rdy_x = &p_a_rdy;
  assign sw_rdy = &p_sw_rdy;
  // a request goes to its adapter(s) only in a cycle both are ready (what the slice sees): an
  // adapter ready alone would take it again in every cycle the slice holds it
  always_comb
    for (int p = 0; p < 2; p++) begin
      p_b_req[p]  = b_req && b_rdy && (b_two || bp == p[0]);
      p_b_addr[p] = (b_two && bp != p[0]) ? bi2 : bi;
      p_b_par[p]  = (b_two && bp != p[0]) ? bpar2 : bpar;
      p_a_req[p]  = a_req_x && a_rdy_x && a_addr_x[CB] == p[0];
      p_sw_req[p] = sw_req && sw_rdy && sw_addr[CB] == p[0];
    end
  assign wr_idle = &p_wr_idle;

  // ---- order FIFOs: B {two, first adapter, the slice's tag}, A {adapter}
  logic [2:0]  bq [OQ];
  logic [OW:0] bq_n;
  logic [OW-1:0] bq_h;
  logic        aq [OQ];
  logic [OW:0] aq_n;
  logic [OW-1:0] aq_h;
  wire b_take = b_req && b_rdy && !b_we;
  wire a_take = a_req_x && a_rdy_x && !a_we_x;
  wire [2:0] bq_in = {b_two, bp, b_tag};
  // the heads, registered (the LUT RAM read stays out of the take enables): after a pop the next
  // entry (written at least a cycle before), or the entry pushed into an empty queue
  logic [2:0] bqh;
  logic       aqh;
  wire        bh_two = bqh[2], bh_p = bqh[1];
  // the head's chunks are all in: both adapters' heads (a pair), else its adapter's
  wire b_ok = (bq_n != 0) && (bh_two ? &p_b_ravail : p_b_ravail[bh_p]);
  always_comb
    for (int p = 0; p < 2; p++) begin
      p_b_rrdy[p] = b_ok && (bh_two || bh_p == p[0]);
      p_a_rrdy[p] = (aq_n != 0) && aqh == p[0];
    end
  wire a_pop = (aq_n != 0) && p_a_ravail[aqh];
  always_ff @(posedge clk) begin
    if (b_ok) bqh <= (bq_n == (OW + 1)'(1)) ? bq_in : bq[OW'(bq_h + 1'b1)];
    else if (bq_n == 0) bqh <= bq_in;
    if (a_pop) aqh <= (aq_n == (OW + 1)'(1)) ? a_addr_x[CB] : aq[OW'(aq_h + 1'b1)];
    else if (aq_n == 0) aqh <= a_addr_x[CB];
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      bq_n <= '0; bq_h <= '0; aq_n <= '0; aq_h <= '0;
    end else begin
      bq_n <= bq_n + (b_take ? 1'b1 : 1'b0) - (b_ok ? 1'b1 : 1'b0);
      if (b_ok) bq_h <= bq_h + 1'b1;
      aq_n <= aq_n + (a_take ? 1'b1 : 1'b0) - (a_pop ? 1'b1 : 1'b0);
      if (a_pop) aq_h <= aq_h + 1'b1;
    end
    if (b_take) bq[OW'(bq_h + bq_n)] <= bq_in;
    if (a_take) aq[OW'(aq_h + aq_n)] <= a_addr_x[CB];
  end
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst) begin
    if (b_take && bq_n[OW]) $fatal(1, "otpu_native_dram2: B order FIFO overflow");
    if (a_take && aq_n[OW]) $fatal(1, "otpu_native_dram2: A order FIFO overflow");
    if (b_req && b_rdy && b_two && b_we) $fatal(1, "otpu_native_dram2: a pair write");
    if (b_ok && p_b_rtag[bh_p] != bqh[0]) $fatal(1, "otpu_native_dram2: B tag differs");
    for (int p = 0; p < 2; p++)
      if (p_b_rvalid[p] != (b_ok && (bh_two || bh_p == p[0])))
        $fatal(1, "otpu_native_dram2: adapter %0d B data out of order", p);
  end
`endif

  // ---- read data in order: B this cycle (b_rvalid: the adapters' b_out), A the next (each
  // adapter registers it; one adapter's a cycle)
  assign b_rvalid = b_ok;
  assign b_rtag   = bqh[0];
  assign b_rdata  = p_b_rdata[bh_p];
  assign b_rdata2 = p_b_rdata[!bh_p];
  assign a_rvalid = |p_a_rvalid;
  assign a_rdata  = p_a_rvalid[1] ? p_a_rdata[1] : p_a_rdata[0];
  assign rd_idle  = &p_rd_idle && bq_n == 0 && aq_n == 0;

  for (genvar p = 0; p < 2; p++) begin : g_ad
    otpu_native_dram #(.D(D), .CHASH(CHASH), .RD(RD), .AW(AW), .BAW(BAW), .QMEM(QMEM)) u_ad (
      .clk, .rst, .a_flush,
      .a_rdy_x(p_a_rdy[p]), .a_req_x(p_a_req[p]), .a_we_x(a_we_x), .a_addr_x(inner(a_addr_x)),
      .a_wdata_x, .a_be_x, .a_rvalid(p_a_rvalid[p]), .a_rdata(p_a_rdata[p]),
      .sw_rdy(p_sw_rdy[p]), .sw_req(p_sw_req[p]), .sw_addr(inner(sw_addr)), .sw_wdata, .sw_be,
      .b_rdy(p_b_rdy[p]), .b_req(p_b_req[p]), .b_tag, .b_we, .b_wmask, .b_wdata,
      .b_addr(p_b_addr[p]), .b_par(p_b_par[p]), .b_rvalid(p_b_rvalid[p]), .b_rtag(p_b_rtag[p]),
      .b_rdata(p_b_rdata[p]), .b_ravail(p_b_ravail[p]), .b_rrdy(p_b_rrdy[p]),
      .a_ravail(p_a_ravail[p]), .a_rrdy(p_a_rrdy[p]),
      .wr_idle(p_wr_idle[p]), .rd_idle(p_rd_idle[p]),
      .n_cvalid(n_cvalid[2*p +: 2]), .n_cready(n_cready[2*p +: 2]), .n_cwe(n_cwe[2*p +: 2]),
      .n_caddr(n_caddr[2*p +: 2]), .n_wvalid(n_wvalid[2*p +: 2]), .n_wready(n_wready[2*p +: 2]),
      .n_wdata(n_wdata[2*p +: 2]), .n_wmask(n_wmask[2*p +: 2]), .n_rvalid(n_rvalid[2*p +: 2]),
      .n_rdata(n_rdata[2*p +: 2]), .n_wdone(n_wdone[2*p +: 2]));
  end
endmodule
