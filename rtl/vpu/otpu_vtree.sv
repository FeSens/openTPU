// Folding tree of the isum_64 reductions (docs/isa.md "Sums"): the row sums of the LANES-lane
// partial loops (RSUM, RSSQ, RDOT in the VPU; the two row dots of DSTEP in the DMA). The
// partial loop delivers a row's final partials on `pacc`, LANES per cycle, one sub-row per
// cycle (sub = 0..RL-1, RL = 64 / LANES): `cap` marks a cycle that carries final partials,
// `row_last` the row's last. The root comes out RD cycles after the row's last partials
// (root_v). Rows must start a multiple of RL cycles apart (the adders follow a fixed
// schedule); every register advances with `en`.
module otpu_vtree
  import otpu_fp::*;
#(
  parameter int LANES = 8,
  parameter int LA    = 4
) (
  input  logic       clk,
  input  logic       rst,
  input  logic       en,
  input  f32_t       pacc [LANES],
  input  logic       cap,
  input  logic       row_last,
  input  logic [7:0] sub,
  output f32_t       root,
  output logic       root_v
);
  localparam int NP = 64;                    // isum_64 partials
  localparam int RL = NP / LANES;            // partial loop length in chunks
  localparam int LW = $clog2(LANES);

  // Folding tree (level n = 32, 16, .., 1: x[i] += x[i+n], i < n), streamed. Partial p sits in
  // row p / LANES, lane p % LANES, and a row's final partials arrive on `pacc` one row per
  // cycle (sub 0..RL-1). Row levels (n >= LANES; j = 0..LR-1, H = RL >> (j+1) pairs: row r +=
  // row r + H) run on the LANES adders u_tree; pair r of level j issues I_j + r cycles after
  // sub 0 is on `pacc`, its operands taken from delay lines (level 0: `pacc`, level j >= 1:
  // `tr_y`). Level j issues in the residues [H, 2H) mod RL, and rows start a multiple of RL
  // cycles apart (nch is padded to a multiple of RL, rows issue back to back, reductions run
  // alone), so the adders are never double-booked and nothing stalls (TP: the last row level
  // issues one cycle early, in residue 0, which no level uses). Lane levels (n < LANES)
  // run on a dedicated pipeline of adders (see xl).
  localparam int LR = $clog2(RL);             // row levels
  function automatic int tree_h(input int j);   // pairs of row level j
    return RL >> (j + 1);
  endfunction
  function automatic int tree_i(input int j);   // issue offset I_j of row level j
    int t;
    t = tree_h(0);
    for (int i = 1; i <= j; i++)
      t = t - tree_h(i) + RL * ((LA + 2 * tree_h(i) + RL - 1) / RL);
    return t;
  endfunction
  // level j >= 1 reads row r from tr_y delayed tree_da(j), row r + H delayed tree_db(j)
  function automatic int tree_da(input int j);
    return tree_i(j) - tree_i(j - 1) - LA;
  endfunction
  function automatic int tree_db(input int j);
    return tree_da(j) - tree_h(j);
  endfunction
  function automatic int tree_dmax();
    int m;
    m = 1;
    for (int j = 1; j < LR; j++) if (tree_da(j) > m) m = tree_da(j);
    return m;
  endfunction
  // LN2: the three lane levels of LANES = 8 fold onto two adders (see u_ln), root 2 cycles later
  localparam bit LN2 = (LANES == 8) && (RL == 8) && (LA == 4);
  // TP (LN2's tree): the last row level issues one cycle early, in the free residue 0, so every
  // u_tree operand is one 2:1 mux with a flip-flop select (see g_tp); DL and RD stay the same
  localparam bit TP = LN2;
  localparam int DMAX = TP ? 2 : tree_dmax();
  localparam int DL = tree_i(LR - 1) - (RL - 1);                // row_last -> last row level
  localparam int RD = tree_i(LR - 1) + LA * (1 + LW) - (RL - 1) // row_last -> root
                      + (LN2 ? 2 : 0);

  // level strobes: cap of the subs a level pairs up, delayed to the level's issue cycles
  // (level 0 is the default); the last row level and the root share one shift register
  logic [LR-1:1] lv, lvn;                     // lvn: lv one cycle early (TP)
  logic [RD-1:0] rsr;
  always_ff @(posedge clk)
    if (rst) rsr <= '0;
    else if (en) rsr <= {rsr[RD-2:0], cap && row_last};
  assign lv[LR-1] = rsr[TP ? DL - 2 : DL - 1];
  assign lvn[LR-1] = 1'b0;
  wire root_v_i = rsr[RD-1];
  for (genvar j = 1; j < LR - 1; j++) begin : g_lv
    localparam int DJ = tree_i(j) - (RL - tree_h(j));
    logic [DJ-1:0] sr;
    always_ff @(posedge clk)
      if (rst) sr <= '0;
      else if (en) sr <= {sr[DJ-2:0], cap && sub >= 8'(RL - tree_h(j))};
    assign lv[j] = sr[DJ-1];
    assign lvn[j] = sr[(DJ >= 2) ? DJ - 2 : 0];
  end

  f32_t tr_y [LANES];
  f32_t tr_y1 [LANES];                        // tr_y delayed one cycle (td[1])
  for (genvar k = 0; k < LANES; k++) begin : g_tree
    f32_t pdd, oa, ob;
    logic [DMAX:1][31:0] td;                 // tr_y[k] delayed 1..DMAX
    otpu_delay #(.W(32), .N(RL / 2 - 1)) u_pd (.clk, .en, .d(pacc[k]), .q(pdd));
    always_ff @(posedge clk) if (en) begin
      td[1] <= tr_y[k];
      for (int d = 2; d <= DMAX; d++) td[d] <= td[d-1];
    end
    assign tr_y1[k] = td[1];
    if (TP) begin : g_tp
      // level 0 at 4..7 (residues after sub 0 is on `pacc`), level 1 at 10, 11, level 2 at 16:
      //   a = la ? td[2] : pacc,  b = lv[1] ? tr_y : qb,  qb = (a cycle earlier) lv2n ? tr_y : pdd
      // level 0: pacc (row r + 4) + qb = pd (row r), the pair swapped (fp_add is commutative bit
      // for bit); level 1: td[2] (row r) + tr_y (row r + 2); level 2: td[2] (row 0) + qb = td[1]
      // (row 1). Each operand: one adder output behind a 2:1 mux with a flip-flop select.
      f32_t qb;
      logic la;
      wire  lv2n = rsr[DL - 3];              // level 2 one cycle early
      always_ff @(posedge clk)
        if (rst) la <= 1'b0;
        else if (en) la <= lvn[1] || lv2n;
      always_ff @(posedge clk) if (en) qb <= lv2n ? tr_y[k] : pdd;
      assign oa = la ? td[2] : pacc[k];
      assign ob = lv[1] ? tr_y[k] : qb;
    end else begin : g_gen
      f32_t pd;
      f32_t sa [LR], sb [LR];                // level j's operands (row r, row r + H)
      wire  [DMAX:0][31:0] tt = {td, tr_y[k]};
      // pd = pacc delayed RL/2; its last stage has a reset so it stays a flip-flop, not a
      // shift-register LUT (the value after reset is never used)
      always_ff @(posedge clk)
        if (rst) pd <= '0;
        else if (en) pd <= pdd;
      assign sa[0] = pd;
      assign sb[0] = pacc[k];
      for (genvar j = 1; j < LR; j++) begin : g_op
        assign sa[j] = tt[tree_da(j)];
        assign sb[j] = tt[tree_db(j)];
      end
      always_comb begin
        oa = sa[0]; ob = sb[0];
        for (int j = 1; j < LR; j++)
          if (lv[j]) begin oa = sa[j]; ob = sb[j]; end
      end
    end
    otpu_fadd #(.LAT(LA)) u_tree (.clk, .en, .a(oa), .b(ob), .y(tr_y[k]));
  end

  // lane levels n = LANES >> q: xl[q][k] = xl[q-1][k] + xl[q-1][k + n], k < n; the root is
  // xl[LW][0], RD cycles after the row's last partial. Level q issues LA * q cycles after the
  // last row level (l2_v for q = 2), rows a multiple of RL cycles apart: unless LA is a
  // multiple of RL, level 2 never meets level 1 and runs on level 1's adders k < LANES/4
  // (SH2; xl[2][k] is then xl[1][k] LA cycles later)
  localparam bit SH2 = !LN2 && (LW >= 2) && (LA % RL != 0);
  wire  l2_v = rsr[DL + 2 * LA - 1];
  f32_t xl [LW + 1][LANES];
  f32_t xs [LANES];                           // SH2: the shared adders' results (level 1 or 2)
  logic ln_l1b, ln_l2, ln_l3;                 // LN2 issue strobes (L1a is rsr[DL + LA - 1])
  for (genvar k = 0; k < LANES; k++) begin : g_xl0
    assign xl[0][k] = tr_y[k];
  end
  // LN2 (E = the last row level's issue + LA + 1 -- TP issues it one cycle early -- the
  // residues mod RL = 8 in brackets): u_ln[m] issues
  //   L1a at E     (5): xl[0][m]     + xl[0][m + 4]   (from td[1])
  //   L1b at E+1   (6): xl[0][m + 2] + xl[0][m + 6]   (from td[2])
  //   L2  at E+5   (2): xs[m] at E+4 + xs[m]          (L1a + L1b = xl[1][m] + xl[1][m + 2])
  //   L3  at E+10  (7): xs[0] + xs[1] at E+9, m = 0   (L2 results = xl[2][0] + xl[2][1])
  // and the root is xs[0] at E+14. The residues are distinct and rows are a multiple of RL
  // apart, so neither adder is ever double-booked; same operands in the same order as xl.
  // The operands are registered one granted cycle ahead (qa, qb): a has no mux, b one 2:1 mux
  // with a flip-flop select in front of the L1b result xs[m].
  if (LN2) begin : g_ln
    wire l1a_n = rsr[DL + LA - 2];            // L1a / L1b one cycle early
    wire l1b_n = rsr[DL + LA - 1];
    assign ln_l1b = rsr[DL + LA];
    assign ln_l2  = rsr[DL + 2 * LA];
    assign ln_l3  = rsr[DL + 2 * LA + 5];
    for (genvar m = 0; m < 2; m++) begin : g_m
      f32_t qa, qb, ob;
      always_ff @(posedge clk) if (en) begin
        qa <= l1a_n ? tr_y[m]     : l1b_n ? tr_y1[m + 2] : xs[m];
        qb <= l1a_n ? tr_y[m + 4] : l1b_n ? tr_y1[m + 6] : xs[1];
      end
      assign ob = ln_l2 ? xs[m] : qb;
      otpu_fadd #(.LAT(LA)) u_ln (.clk, .en, .a(qa), .b(ob), .y(xs[m]));
    end
  end else begin : g_lg
    assign ln_l1b = 1'b0;
    assign ln_l2  = 1'b0;
    assign ln_l3  = 1'b0;
    for (genvar q = 1; q <= LW; q++) begin : g_xl
      for (genvar k = 0; k < (LANES >> q); k++) begin : g_k
        if (SH2 && q == 2) begin : g_sh
          assign xl[q][k] = xs[k];
        end else if (SH2 && q == 1 && k < (LANES >> 2)) begin : g_mux
          f32_t oa, ob;
          assign oa = l2_v ? xs[k] : xl[0][k];
          assign ob = l2_v ? xl[1][k + (LANES >> 2)] : xl[0][k + (LANES >> 1)];
          otpu_fadd #(.LAT(LA)) u_add (.clk, .en, .a(oa), .b(ob), .y(xs[k]));
          assign xl[q][k] = xs[k];
        end else begin : g_add
          otpu_fadd #(.LAT(LA)) u_add (.clk, .en, .a(xl[q-1][k]), .b(xl[q-1][k + (LANES >> q)]),
                                       .y(xl[q][k]));
        end
      end
    end
  end

  assign root = LN2 ? xs[0] : xl[LW][0];
  assign root_v = root_v_i;
`ifndef SYNTHESIS
  // the tree schedule needs rows a multiple of RL cycles apart: level 0 (issuing while the
  // subs >= RL/2 are on `pacc`) must never meet another level
  always_ff @(posedge clk)
    if (!rst && en && cap && sub >= 8'(RL / 2) && lv != 0)
      $fatal(1, "otpu_vtree: tree schedule collision");
  // lane level 2 shares level 1's adders: it must never issue with level 1 (LN2: the lane
  // levels L1a, L1b, L2 and L3 share u_ln and must never issue together)
  always_ff @(posedge clk)
    if (!rst && en && (LN2 ? !$onehot0({rsr[DL + LA - 1], ln_l1b, ln_l2, ln_l3})
                           : SH2 && l2_v && rsr[DL + LA - 1]))
      $fatal(1, "otpu_vtree: lane level schedule collision");
`endif
endmodule
