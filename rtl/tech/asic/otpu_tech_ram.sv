// The technology RAMs on SRAM macros (an ASIC's rtl/tech; the generic file infers the same
// modules). Each RAM is tiled onto leaf macros of at most WMAX bits by NMAX words: width tiles side
// by side, depth tiles (N a multiple of NMAX) with a registered read bank select. The leaves
// (otpu_sram_1r1w, otpu_sram_1rw, otpu_sram_1rw1r) are generated per design by tools/asic/flow.py,
// one branch per macro size the design uses (asic/<pdk>'s macros: tools/asic/fakeram.py for
// NanGate45).
//
// The macros' semantics are the generic RAMs': registered reads with an enable (the output holds
// otherwise), writes masked per bit. A 1R1W macro's read of the word written at the same edge is
// undefined; no user depends on it (otpu_tmem bypasses it). A 1RW1R macro reads first on both
// ports (otpu_tmem IMPL 3 TDP's replays read a word in the cycle its write lands: a real PDK's
// dual-port macro has to offer it, docs/asic.md). STYLE and ZERO are the FPGA's and the
// simulation's, unused here.
package otpu_tech_asic;
  localparam int WMAX = 128;    // bits per macro
  localparam int NMAX = 4096;   // words per macro
endpackage

// one tiled RAM: KIND 0 1R1W, 1 1RW (a: ra; en: re | we)
module otpu_ram_tiled #(
  parameter int W    = 32,
  parameter int N    = 1024,
  parameter int KIND = 0
) (
  input  logic                 clk,
  input  logic                 we,
  input  logic [W-1:0]         wm,      // per bit
  input  logic [$clog2(N)-1:0] wa,
  input  logic [W-1:0]         wd,
  input  logic                 re,
  input  logic [$clog2(N)-1:0] ra,
  output logic [W-1:0]         rd
);
  import otpu_tech_asic::*;
  localparam int AW = $clog2(N);
  localparam int NW = (W + WMAX - 1) / WMAX;          // width tiles
  localparam int TN = (N < NMAX) ? N : NMAX;          // words per tile
  localparam int NN = N / TN;                         // depth tiles
  localparam int TA = $clog2(TN);
  initial if (NN * TN != N) $fatal(1, "otpu_ram_tiled: %0d words is not a multiple of %0d", N, NMAX);
  // each depth tile's read data (unpacked: Verilator 5.047 read q[rbank] of a packed [NN][W]
  // wrongly here, W 40, N 8192; tests/test_tech_ram.py)
  logic [W-1:0] q [NN];
  logic [(NN > 1 ? $clog2(NN) : 1)-1:0] rbank;   // the tile the last read read
  // the depth tile an address is in (0 with one tile)
  localparam int BW = (NN > 1) ? $clog2(NN) : 1;
  logic [BW-1:0] rt, wt;
  if (NN > 1) begin : g_sel
    assign rt = ra[AW-1:TA];
    assign wt = wa[AW-1:TA];
  end else begin : g_nosel
    assign rt = '0;
    assign wt = '0;
  end
  for (genvar d = 0; d < NN; d++) begin : g_d
    wire rsel = (rt == BW'(d));
    wire wsel = (wt == BW'(d));
    for (genvar c = 0; c < NW; c++) begin : g_c
      localparam int CW = (c == NW - 1) ? W - c * WMAX : WMAX;
      if (KIND == 0) begin : g_2p
        otpu_sram_1r1w #(.W(CW), .N(TN)) u_m (
          .clk, .re(re && rsel), .ra(ra[TA-1:0]), .we(we && wsel), .wa(wa[TA-1:0]),
          .wd(wd[c*WMAX +: CW]), .wm(wm[c*WMAX +: CW]), .rd(q[d][c*WMAX +: CW]));
      end else begin : g_1p
        // one port: the write's address when writing, the read's otherwise
        otpu_sram_1rw #(.W(CW), .N(TN)) u_m (
          .clk, .en((re && rsel) || (we && wsel)), .we(we && wsel),
          .a(we ? wa[TA-1:0] : ra[TA-1:0]), .wd(wd[c*WMAX +: CW]), .wm(wm[c*WMAX +: CW]),
          .rd(q[d][c*WMAX +: CW]));
      end
    end
  end
  if (NN > 1) begin : g_bank
    always_ff @(posedge clk) if (re) rbank <= rt;
    assign rd = q[rbank];
  end else begin : g_one
    assign rbank = '0;
    assign rd = q[0];
  end
endmodule

// one tiled two-port RAM (1RW1R): port A a read or a write, port B a read, each depth tile with
// its own registered read select per port; the leaves (otpu_sram_1rw1r) read first
module otpu_ram_tiled_dp #(
  parameter int W    = 32,
  parameter int N    = 1024
) (
  input  logic                 clk,
  input  logic                 ena,
  input  logic                 wea,
  input  logic [$clog2(N)-1:0] aa,
  input  logic [W-1:0]         wda,
  output logic [W-1:0]         rda,
  input  logic                 enb,
  input  logic [$clog2(N)-1:0] ab,
  output logic [W-1:0]         rdb
);
  import otpu_tech_asic::*;
  localparam int AW = $clog2(N);
  localparam int NW = (W + WMAX - 1) / WMAX;
  localparam int TN = (N < NMAX) ? N : NMAX;
  localparam int NN = N / TN;
  localparam int TA = $clog2(TN);
  localparam int BW = (NN > 1) ? $clog2(NN) : 1;
  initial if (NN * TN != N) $fatal(1, "otpu_ram_tiled_dp: %0d words is not a multiple of %0d", N, NMAX);
  logic [W-1:0] qa [NN];
  logic [W-1:0] qb [NN];
  logic [BW-1:0] ta, tb;
  if (NN > 1) begin : g_sel
    assign ta = aa[AW-1:TA];
    assign tb = ab[AW-1:TA];
  end else begin : g_nosel
    assign ta = '0;
    assign tb = '0;
  end
  for (genvar d = 0; d < NN; d++) begin : g_d
    wire asel = (ta == BW'(d));
    wire bsel = (tb == BW'(d));
    for (genvar c = 0; c < NW; c++) begin : g_c
      localparam int CW = (c == NW - 1) ? W - c * WMAX : WMAX;
      otpu_sram_1rw1r #(.W(CW), .N(TN)) u_m (
        .clk, .ena(ena && asel), .wea(wea && asel), .aa(aa[TA-1:0]), .wd(wda[c*WMAX +: CW]),
        .wm({CW{1'b1}}), .rda(qa[d][c*WMAX +: CW]), .enb(enb && bsel), .ab(ab[TA-1:0]),
        .rdb(qb[d][c*WMAX +: CW]));
    end
  end
  if (NN > 1) begin : g_bank
    logic [BW-1:0] bka, bkb;
    always_ff @(posedge clk) begin
      if (ena) bka <= ta;
      if (enb) bkb <= tb;
    end
    assign rda = qa[bka];
    assign rdb = qb[bkb];
  end else begin : g_one
    assign rda = qa[0];
    assign rdb = qb[0];
  end
endmodule

module otpu_ram_1r1w #(
  parameter int    W     = 32,
  parameter int    N     = 1024,
  parameter string STYLE = "block",
  parameter bit    ZERO  = 1'b1,
  parameter bit    OREG  = 1'b0      // the read a second edge later (the generic file's)
) (
  input  logic                 clk,
  input  logic                 we,
  input  logic [$clog2(N)-1:0] wa,
  input  logic [W-1:0]         wd,
  input  logic                 re,
  input  logic [$clog2(N)-1:0] ra,
  output logic [W-1:0]         rd
);
  logic [W-1:0] rq;
  otpu_ram_tiled #(.W(W), .N(N), .KIND(0)) u_t (
    .clk, .we, .wm('1), .wa, .wd, .re, .ra, .rd(rq));
  if (OREG) begin : g_oreg
    always_ff @(posedge clk) rd <= rq;
  end else begin : g_nore
    assign rd = rq;
  end
endmodule

// two clocks (the generic file's): no macro here; only the WG targets' posted chunk FIFO uses it
module otpu_ram_1r1w_2c #(
  parameter int    W     = 32,
  parameter int    N     = 1024,
  parameter string STYLE = "block",
  parameter bit    ZERO  = 1'b1
) (
  input  logic                 wclk,
  input  logic                 we,
  input  logic [$clog2(N)-1:0] wa,
  input  logic [W-1:0]         wd,
  input  logic                 rclk,
  input  logic                 re,
  input  logic [$clog2(N)-1:0] ra,
  output logic [W-1:0]         rd
);
  initial $fatal(1, "otpu_ram_1r1w_2c: no two-clock SRAM macro (otpu_mxu XPOST is the WG targets')");
  assign rd = '0;
endmodule

// two clocks with field writes (the generic file's): no macro here either (otpu_mxu XPOST's
// remainder RAM)
module otpu_ram_1r1w_2c_be #(
  parameter int    W     = 36,
  parameter int    N     = 512,
  parameter int    NB    = 4,
  parameter string STYLE = "block",
  parameter bit    ZERO  = 1'b1
) (
  input  logic                 wclk,
  input  logic                 we,
  input  logic [NB-1:0]        wbe,
  input  logic [$clog2(N)-1:0] wa,
  input  logic [W-1:0]         wd,
  input  logic                 rclk,
  input  logic                 re,
  input  logic [$clog2(N)-1:0] ra,
  output logic [W-1:0]         rd
);
  initial $fatal(1, "otpu_ram_1r1w_2c_be: no two-clock SRAM macro (otpu_mxu XPOST is the WG targets')");
  assign rd = '0;
endmodule

// Read ahead (rtl/tech/generic has the contract): small and asynchronous-read, so flip-flops and
// a read multiplexer at the registered address, as the units' LUT RAMs were; no macro.
module otpu_ram_1r1w_ra #(
  parameter int    W     = 32,
  parameter int    N     = 64,
  parameter string STYLE = "block",
  parameter bit    ZERO  = 1'b1
) (
  input  logic                 clk,
  input  logic                 we,
  input  logic [$clog2(N)-1:0] wa,
  input  logic [W-1:0]         wd,
  input  logic [$clog2(N)-1:0] ra_n,
  output logic [W-1:0]         rd
);
  logic [W-1:0]         mem [N];
  logic [$clog2(N)-1:0] ra;
  always_ff @(posedge clk) begin
    if (we) mem[wa] <= wd;
    ra <= ra_n;
  end
  assign rd = mem[ra];
`ifndef SYNTHESIS
  initial begin
    ra = '0;
    if (ZERO) for (int i = 0; i < N; i++) mem[i] = '0;
  end
`endif
endmodule

module otpu_ram_1r1w_be #(
  parameter int    W     = 64,
  parameter int    N     = 1024,
  parameter string STYLE = "block",
  parameter bit    ZERO  = 1'b1
) (
  input  logic                 clk,
  input  logic                 we,
  input  logic [W/8-1:0]       wbe,
  input  logic [$clog2(N)-1:0] wa,
  input  logic [W-1:0]         wd,
  input  logic                 re,
  input  logic [$clog2(N)-1:0] ra,
  output logic [W-1:0]         rd
);
  logic [W-1:0] wm;
  always_comb for (int k = 0; k < W / 8; k++) wm[8*k +: 8] = {8{wbe[k]}};
  otpu_ram_tiled #(.W(W), .N(N), .KIND(0)) u_t (
    .clk, .we, .wm, .wa, .wd, .re, .ra, .rd);
endmodule

module otpu_ram_1rw #(
  parameter int    W     = 32,
  parameter int    N     = 1024,
  parameter string STYLE = "block",
  parameter bit    ZERO  = 1'b1
) (
  input  logic                 clk,
  input  logic                 en,
  input  logic                 we,
  input  logic [$clog2(N)-1:0] a,
  input  logic [W-1:0]         wd,
  output logic [W-1:0]         rd
);
  otpu_ram_tiled #(.W(W), .N(N), .KIND(1)) u_t (
    .clk, .we(en && we), .wm('1), .wa(a), .wd, .re(en && !we), .ra(a), .rd);
endmodule

module otpu_ram_1rw1r #(
  parameter int    W     = 32,
  parameter int    N     = 1024,
  parameter string STYLE = "block",
  parameter bit    ZERO  = 1'b1
) (
  input  logic                 clk,
  input  logic                 ena,
  input  logic                 wea,
  input  logic [$clog2(N)-1:0] aa,
  input  logic [W-1:0]         wda,
  output logic [W-1:0]         rda,
  input  logic                 enb,
  input  logic [$clog2(N)-1:0] ab,
  output logic [W-1:0]         rdb
);
  otpu_ram_tiled_dp #(.W(W), .N(N)) u_t (.clk, .ena, .wea, .aa, .wda, .rda, .enb, .ab, .rdb);
endmodule
