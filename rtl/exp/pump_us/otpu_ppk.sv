// EXPERIMENTAL: not in any file list, not in the board build. An out-of-context timing test only
// (ooc.tcl), with a Verilator check of its arithmetic (run_vl.sh, tb_ppk.sv).
//
// One column of the BCU1525's WG 8 MXU (otpu_mxu IMPL 2 at WG > 1: packed PEs, otpu_pe PK, on
// DSP48E2s, rtl/tech/xilinxus) pumped: its 128 DSP48E2s run at clk_f = 2 x clk_s and make, in a
// slow cycle, the packed products of TWO activation rows (x0, x1: two of the advance's rows)
// against the same weight chunk. The column does two of today's columns' work: option (i) of the
// 2x MXU note (300 MHz under the 150 MHz core). The fabric stays at clk_s except, per position,
// the activation's 2:1 select and its register (b_f) and the phase toggles; clk_f comes from the
// same MMCM, phase-aligned (every slow rising edge is a fast rising edge).
//
//   clk_s : the activation skews and hop (x0, x1), the weight skews and decode, the column ends
//           (four otpu_mxce: rows 0 / 1 x halves S0 / S1, as otpu_mxu's u_ce / u_ce2 per row)
//   clk_f : the DSP48E2 (A2 / D loaded on the slow edges, the pre-adder's AD register holding
//           the packed weight wlo + whi * 2^16 for two fast cycles, B = the activation, x0 and x1
//           on alternate fast edges, M, P and the PCIN cascade carrying the two rows' packed sums
//           interleaved), b_f, the per-pair phase toggles, the chain ends' holding registers
//
// Fast-edge schedule (fast edges f; the slow edges are the even f; t the chunk's cycle at the
// column's inputs; k = i % CLS the chain stage; COL adds 2 COL to every f below). P_k captures
// row r's running sum at f = 6 + k + r + 2t: the cascade moves a stage per fast cycle with the
// two rows a fast cycle apart. Back through M (f - 1), B and AD (f - 2), b_f (f - 3):
//   AD  loads at the edges of k's parity (CEAD = k even ? pe : po) and holds two fast cycles
//   A2 / D (the decoded weights) load on the slow edges (CE = pe) from a skew of ceil(k/2):
//       a two-fast-cycle path from clk_s (multicycle 2, ooc.tcl)
//   b_f takes x0 on the edges of k's other parity and x1 on k's parity (sel = k even ? pe : po),
//       from skews of floor(k/2) + 1 (x0) and floor((k+1)/2) + 1 (x1): the captures on odd
//       edges are one-fast-period paths from clk_s
// pe is 1 on the even edges, po on the odd ones: per pair of positions a toggle of each
// (dont_touch, reset-aligned), so no phase net fans out beyond two DSPs.
// Chain ends: row 0's P_15 at f = 21 + 2t, row 1's at 22 + 2t. HOLD 0: the column end takes row
// 0 from P on the slow edge 22 + 2t (one fast period) and row 1 from a register loaded on the odd
// edge 23 + 2t (one more); its rows' A / D presented in slow cycles t + 10 and t + 11. HOLD 1:
// row 0 held on the even edge 22 + 2t, row 1 on the odd edge 23 + 2t and again on the even edge
// 24 + 2t, so every return into clk_s is a two-fast-cycle path (multicycle, ooc.tcl), a slow
// cycle later (t + 11, t + 12).
// Packed sums (otpu_mxu's header): P = S0 + 2^15 + S1 * 2^16 (the chain's first stage adds RND =
// 2^15, |S0|, |S1| < 2^15 for 4-bit weights, |w| <= 12, and int8's nibbles), S0 = P[15:0] - 2^15
// (bit 15 inverted), S1 = P >> 16. Row r's u_ce takes S0 (and, int8, 16 S1 on D: den) with B
// bm[2r], its u_ce2 S1 with bm[2r + 1]; acc[2r + h] for chunk t in slow cycle
// t + COL + 15 + HOLD + r (first: the chunk's row of blocks starts, as otpu_mxce's).
module otpu_ppk_col #(
  parameter int D = 128,
  parameter int CLS = 16,
  parameter int COL = 0,            // the column (its skews: COL more; COL > 0 hops x from COL - 1)
  parameter bit BF = 1'b1,          // b_f a fabric register (else B1, BREG 2: the select into B)
  parameter bit HOLD = 1'b0         // chain ends held into clk_s over two fast cycles
) (
  input  logic                    clk_s,
  input  logic                    clk_f,
  input  logic                    rst_s,      // clk_s: the fast phase aligns as it falls
  input  logic [D-1:0][7:0]       x0p,        // COL 0: row 0's bytes, chunk t in cycle t; else
  input  logic [D-1:0][7:0]       x1p,        // column COL - 1's skewed rows (x0h, x1h)
  output logic [D-1:0][7:0]       x0h,
  output logic [D-1:0][7:0]       x1h,
  input  logic [D-1:0][7:0]       w,          // the column's lane: chunk t in cycle t
  input  logic [1:0]              wf,         // the weights' format (otpu_mxu's; static here)
  input  logic [3:0][7:0][16:0]   bm,         // B (m << sh) for row r, half h at 2r + h, cycle t
  input  logic                    first,      // chunk t starts its row of blocks, cycle t
  output logic [3:0][47:0]        acc
);
  localparam int NCH = D / CLS;
  localparam int NP = CLS / 2;
  localparam int L0 = 10 + (HOLD ? 1 : 0);    // row 0's chain ends at the column ends (cycle)
  initial if (NCH != 8 || CLS != 16) $fatal(1, "otpu_ppk_col: D 128, CLS 16 only");
  localparam logic [1:0] WF_W8 = 2'd0;

  // otpu_mxu's decode (dec4, wdec): a byte's two 4-bit weights, or int8's nibbles
  function automatic logic [7:0] dec4(input logic [3:0] c, input logic fp);
    if (!fp) return {{4{c[3]}}, c};
    case (c)
      4'd5: return 8'd6;
      4'd6: return 8'd8;
      4'd7: return 8'd12;
      4'd8: return 8'd0;
      4'd9: return -8'sd1;
      4'd10: return -8'sd2;
      4'd11: return -8'sd3;
      4'd12: return -8'sd4;
      4'd13: return -8'sd6;
      4'd14: return -8'sd8;
      4'd15: return -8'sd12;
      default: return 8'(c);
    endcase
  endfunction
  function automatic logic [15:0] wdec(input logic [7:0] b, input logic [1:0] f);
    if (f == WF_W8) return {{4{b[7]}}, b[7:4], 4'd0, b[3:0]};
    return {dec4(b[7:4], f != 2'd1), dec4(b[3:0], f != 2'd1)};
  endfunction

  // ---- the fast phase: per chain a reset copy, per pair of positions (and per chain end) a pe
  // and a po toggle. rst_c falls on a slow edge; rst_f a fast edge later (odd), where pe is
  // loaded with 1 (so it reads 1 on the even edges) and po with 0
  (* dont_touch = "true" *) logic rst_c;
  (* dont_touch = "true" *) logic [NCH-1:0] rst_f;
  (* dont_touch = "true" *) logic [NCH-1:0][NP-1:0] pe_t, po_t;
  (* dont_touch = "true" *) logic [NCH-1:0] pe_h, po_h;
  always_ff @(posedge clk_s) rst_c <= rst_s;
  always_ff @(posedge clk_f) begin
    rst_f <= {NCH{rst_c}};
    for (int c = 0; c < NCH; c++) begin
      pe_t[c] <= rst_f[c] ? {NP{1'b1}} : ~pe_t[c];
      po_t[c] <= rst_f[c] ? {NP{1'b0}} : ~po_t[c];
      pe_h[c] <= rst_f[c] ? 1'b1 : ~pe_h[c];
      po_h[c] <= rst_f[c] ? 1'b0 : ~po_h[c];
    end
  end
  // the format: a copy per chain (otpu_mxu's wfc), int8's D at the column ends (den)
  (* dont_touch = "true" *) logic [NCH-1:0][1:0] wfc;
  logic w8;
  always_ff @(posedge clk_s) begin
    wfc <= {NCH{wf}};
    w8 <= wf == WF_W8;
  end

  logic [NCH-1:0][31:0] pl;                        // the chain ends' P
  for (genvar c = 0; c < NCH; c++) begin : g_ch
    logic [47:0] pc [CLS];
    for (genvar k = 0; k < CLS; k++) begin : g_k
      localparam int I = c * CLS + k;
      localparam bit ODD = (k % 2) == 1;
      // the rows' bytes: column 0 skews them, the others hop column COL - 1's
      logic [7:0] x0s, x1s;
      if (COL == 0) begin : g_sk
        otpu_skew #(.W(8), .N(k / 2 + 1))       u_x0 (.clk(clk_s), .en(1'b1), .d(x0p[I]), .q(x0s));
        otpu_skew #(.W(8), .N((k + 1) / 2 + 1)) u_x1 (.clk(clk_s), .en(1'b1), .d(x1p[I]), .q(x1s));
      end else begin : g_hop
        always_ff @(posedge clk_s) begin
          x0s <= x0p[I];
          x1s <= x1p[I];
        end
      end
      assign x0h[I] = x0s;
      assign x1h[I] = x1s;
      // the weights: the lane's byte skewed ceil(k / 2) + COL, decoded into A2 / D
      logic [7:0] ws;
      otpu_skew #(.W(8), .N((k + 1) / 2 + COL)) u_w (.clk(clk_s), .en(1'b1), .d(w[I]), .q(ws));
      wire [15:0] wd = wdec(ws, wfc[c]);
      // the activation: x1 on the edges of k's parity, x0 on the others
      wire sel = ODD ? po_t[c][k / 2] : pe_t[c][k / 2];
      wire [7:0] xm = sel ? x1s : x0s;
      logic [7:0] act;
      if (BF) begin : g_bf
        logic [7:0] b_f;
        always_ff @(posedge clk_f) b_f <= xm;
        assign act = b_f;
      end else begin : g_b1
        assign act = xm;
      end
      logic [31:0] pk;
      otpu_ppk_pe #(.FIRST(k == 0), .BF(BF)) u_ppe (
        .clk_f, .ce_w(pe_t[c][k / 2]), .ce_ad(sel), .wlo(wd[7:0]), .whi(wd[15:8]), .act,
        .pcin(k == 0 ? 48'd0 : pc[k == 0 ? 0 : k - 1]), .pcout(pc[k]), .p(pk));
      if (k == CLS - 1) begin : g_end
        assign pl[c] = pk;
      end
    end
  end

  // ---- the chain ends into clk_s: row 0's and row 1's packed sums per group (q0, q1)
  logic [NCH-1:0][31:0] q0, q1;
  if (HOLD) begin : g_hold
    logic [NCH-1:0][31:0] hold0, hold1, hold1e;
    always_ff @(posedge clk_f)
      for (int c = 0; c < NCH; c++) begin
        if (pe_h[c]) begin
          hold0[c] <= pl[c];
          hold1e[c] <= hold1[c];
        end
        if (po_h[c]) hold1[c] <= pl[c];
      end
    assign q0 = hold0;
    assign q1 = hold1e;
  end else begin : g_nhold
    logic [NCH-1:0][31:0] hold1;
    always_ff @(posedge clk_f)
      for (int c = 0; c < NCH; c++) if (po_h[c]) hold1[c] <= pl[c];
    assign q0 = pl;
    assign q1 = hold1;
  end

  // ---- column ends (clk_s): per row, S0 (and int8's 16 S1) with bm[2r], S1 with bm[2r + 1]
  for (genvar r = 0; r < 2; r++) begin : g_row
    logic [7:0][23:0] ca, cd, ca2;
    logic [7:0][16:0] bmd, bmd2;
    logic             fst;
    always_comb
      for (int g = 0; g < 8; g++) begin
        logic [31:0] q;
        q = (r == 0) ? q0[g] : q1[g];
        ca[g] = {{9{!q[15]}}, q[14:0]};
        cd[g] = {{4{q[31]}}, q[31:16], 4'd0};
        ca2[g] = {{8{q[31]}}, q[31:16]};
      end
    otpu_skew #(.W(8 * 17), .N(L0 + r + COL)) u_bm (.clk(clk_s), .en(1'b1), .d(bm[2 * r]),
                                                    .q(bmd));
    otpu_skew #(.W(8 * 17), .N(L0 + r + COL)) u_bm2 (.clk(clk_s), .en(1'b1), .d(bm[2 * r + 1]),
                                                     .q(bmd2));
    otpu_skew #(.W(1), .N(L0 + r + 3 + COL)) u_f (.clk(clk_s), .en(1'b1), .d(first), .q(fst));
    otpu_mxce #(.CPG(2)) u_ce (.clk(clk_s), .a(ca), .d(cd), .den(w8), .bm(bmd), .first(fst),
                               .acc(acc[2 * r]));
    otpu_mxce #(.CPG(1)) u_ce2 (.clk(clk_s), .a(ca2), .d('0), .den(1'b0), .bm(bmd2), .first(fst),
                                .acc(acc[2 * r + 1]));
  end
endmodule


// A position of the pumped packed column: one DSP48E2 at clk_f. A2 = wlo, D = whi * 2^16 (loaded
// with ce_w, the slow edges), AD = A2 + D (loaded with ce_ad: held two fast cycles), B = the
// activation (BF: from the fabric's b_f, BREG 1; else B1 and B2, BREG 2), M = B * AD and
// P = PCIN + M (FIRST: P = M + RND, RND = 2^15): every register on, INMODE and OPMODE constant.
module otpu_ppk_pe #(parameter bit FIRST = 1'b0, parameter bit BF = 1'b1) (
  input  logic        clk_f,
  input  logic        ce_w,
  input  logic        ce_ad,
  input  logic [7:0]  wlo,
  input  logic [7:0]  whi,
  input  logic [7:0]  act,
  input  logic [47:0] pcin,
  output logic [47:0] pcout,
  output logic [31:0] p
);
  logic [47:0] pf;
  DSP48E2 #(
    .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .AMULTSEL("AD"), .BMULTSEL("B"),
    .PREADDINSEL("A"), .USE_MULT("MULTIPLY"), .USE_WIDEXOR("FALSE"),
    .USE_SIMD("ONE48"), .AREG(1), .ACASCREG(1), .BREG(BF ? 1 : 2), .BCASCREG(BF ? 1 : 2),
    .CREG(0), .DREG(1), .ADREG(1), .MREG(1), .PREG(1), .INMODEREG(0), .OPMODEREG(0),
    .ALUMODEREG(0), .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
    .AUTORESET_PATDET("NO_RESET"), .MASK(48'h3fffffffffff), .PATTERN(48'h0),
    .RND(FIRST ? 48'h8000 : 48'h0), .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
  ) u_dsp (
    .CLK(clk_f),
    .A({{22{wlo[7]}}, wlo}), .B({{10{act[7]}}, act}), .C(48'd0), .D({{3{whi[7]}}, whi, 16'd0}),
    .INMODE(5'b00100), .OPMODE(FIRST ? 9'b10_000_01_01 : 9'b00_001_01_01),
    .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
    .CEA1(1'b0), .CEA2(ce_w), .CEB1(BF ? 1'b0 : 1'b1), .CEB2(1'b1), .CEC(1'b0), .CED(ce_w),
    .CEAD(ce_ad), .CEM(1'b1), .CEP(1'b1), .CEALUMODE(1'b0), .CECTRL(1'b0), .CECARRYIN(1'b0),
    .CEINMODE(1'b0),
    .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
    .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
    .ACIN(30'd0), .BCIN(18'd0), .PCIN(pcin), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
    .ACOUT(), .BCOUT(), .PCOUT(pcout), .P(pf), .CARRYCASCOUT(), .MULTSIGNOUT(), .CARRYOUT(),
    .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT(), .XOROUT());
  assign p = pf[31:0];
endmodule
