// The stream engine's tail (docs/stream.md, sections 4 and 8): every stage of a stream after
// REDUCE A, L = LANES words (one segment) per cycle. It is DSTEP's datapath (otpu_dstep)
// without dot A: A runs on the VPU's slot-0 partial loop and folding tree (otpu_vpu), which
// take the X registers (xd, xa, xm) and hand back each row's kv (kv_v). The DMA fills the
// per-stream vectors and scalars, feeds the stream's segments in DRAM order and takes the
// updated segments (y) and the row outputs (o). With cfg = GDN (a_en, a_sel 0, DELTA, G = K0,
// q_en) it is bit-exact with DSTEP, i.e. the VOP sequence RDOT, MUL, SUB, MUL, OUTER, RDOT.
//
//   X   the input segment (register), with A's vector at its columns (xa) and its meta (xm)
//   A   (the VPU) kv = isum_64(S[r, c] * xa(c)), TA + RD cycles after X
//   D   d from kv, the row scalar x(r), K0 and K1 by cfg.dmode, three multiply-adds (the VPU's
//       MUL, SUB, MUL) into a FIFO:
//         DELTA   d = (x - kv * K0) * K1        DELTA1  d = (x - kv) * K1
//         SCALE   d = x * K1                    DOT     d = kv * K1
//       always DELTA's three slots: slot 1 = (kv or +0) * (K0 or 1) + -0, slot 2 = (x or
//       p1) * 1 + (-p1 or -0), slot 3 = s2 * K1 + -0. mul(z, 1) == z and add(z, -0) == z
//       exactly for every z under FTZ (-0 and NaN included), so each mode is its op list
//   U   Y[r, c] = S[r, c] * G(c) + d * B(c) (the VPU's OUTER), G = K0, slot 3 or 1.0, B slot
//       1; S read back from a delay buffer T_U cycles after X; the updated segment leaves (y)
//   Q   o = isum_64(Y[r, c] * Q(c)), Q slot 0: a partial loop and tree; o leaves (o_v) with
//       cfg.q_en. ONE_TREE: no tree here; Q's final partials go to the VPU's u_vt
//       (qp_*) in windows between A's, and the root comes back (qo_*): T_U is padded so that
//       a row's Q window is DQ = 104 pe cycles after its A window (see T_U). cfg.pad64 (with
//       ONE_TREE): cols = 64 presented as 16 segments per row, the last 8 +0 (the DMA drops
//       their Y); the column buffers read +0 there
//
// A's vector: slot 1 (k) or slot 2 (a, cfg.a_sel) have one buffer at X. An SF_K fill writes
// it unless a_sel, an SF_A fill always, so xa needs no mux. B keeps its own copy of slot 1.
//
// Every register advances with `pe` (registered by the DMA): the pipeline freezes as a whole,
// so the trees' fixed schedules and the delay T_U hold in pe cycles. The DMA feeds a row's
// segments on consecutive pe cycles (bubbles, in_v = 0, only after the last row) and cols is a
// multiple of 64, so rows start a multiple of RL pe cycles apart, as otpu_vtree needs. The VPU
// runs A on every segment whatever cfg.a_en (kv_v times the rows of the d stage).
module otpu_se_tail
  import otpu_pkg::*;
  import otpu_fp::*;
#(
  parameter int LANES = 8,
  parameter int TA    = 7,       // pe cycles from the X registers to A's partials (VPU slot 0)
  // ONE_TREE: Q's final partials go to the VPU's tree (qp_*, QD registers there) and its root
  // comes back (qo_*): no tree of its own (docs/stream.md 11.3)
  parameter bit ONE_TREE = 1'b0,
  parameter int QD    = 1,
  // COMP8: U and Q are also the VPU's composite stages (otpu_se_comp) in VOP mode (cm): their
  // units advance on cen (the VPU's enable, which is pe in stream mode) (docs/stream.md 11.2)
  parameter bit COMP8 = 1'b0
) (
  input  logic             clk,
  input  logic             rst,
  input  logic             init,        // a stream starts: counters cleared
  input  ss_cfg_t          cfg,         // valid from before the fill to the stream's last o
  // fill (the DMA's TMEM reads): a vector segment at index fi, a row scalar segment, K0, K1
  input  logic [2:0]       fk,          // SF_*
  input  logic [4:0]       fi,
  input  f32_t             fd [LANES],
  // stream
  input  logic             pe,
  input  logic             in_v,
  input  f32_t             in_d [LANES],
  output f32_t             xd [LANES],  // X: the segment, A's vector at its columns, its meta
  output f32_t             xa [LANES],
  output ss_meta_t         xm,
  input  f32_t             kv,          // A: a row's dot (from the VPU's tree)
  input  logic             kv_v,
  output logic             y_v,         // an updated segment (with pe)
  output f32_t             y_d [LANES],
  output logic             o_v,         // a row's o (with pe)
  output f32_t             o_d,
  // ONE_TREE: Q's final partials to the VPU's tree (with pe), the row's o back
  output logic             qp_cap,
  output logic             qp_row_last,
  output logic [7:0]       qp_sub,
  output f32_t             qp_d [LANES],
  input  logic             qo_v,
  input  f32_t             qo_d,
  // COMP8: U (y = a*b + c*e) and Q (y = a*b + c) as generic stages, SL cen-cycles from the
  // cycle their operands are presented (u_sel / q_sel) to u_y / q_y
  input  logic             cm,
  input  logic             cen,
  input  logic             u_sel,
  input  f32_t             u_a [LANES],
  input  f32_t             u_b [LANES],
  input  f32_t             u_c [LANES],
  input  f32_t             u_e [LANES],
  output f32_t             u_y [LANES],
  input  logic             q_sel,
  input  f32_t             q_a [LANES],
  input  f32_t             q_b [LANES],
  input  f32_t             q_c [LANES],
  output f32_t             q_y [LANES]
);
  localparam int L = LANES;
  localparam int LM = 2, LA = 4;
  localparam int NP = 64;
  localparam int RL = NP / L;
  localparam int CBD = 256 / L;               // column buffers: 256 columns over the lanes
  localparam int CBW = $clog2(CBD);
  localparam int LW = $clog2(L);
  localparam int SL = 1 + LM + LA;
  // X -> U: the last segment of a row reaches the tree TA later, its root RD later
  // (otpu_vtree: at most 28 for LANES 8), d three multiply-adds (3 SL) and a FIFO write after
  // that; a row's first segment is ns - 1 ahead of its last (ns <= CBD)
  localparam int RD_MAX = 28;
  localparam int T_U0 = (CBD - 1) + TA + RD_MAX + 3 * SL + 3;
  // ONE_TREE: a row's Q window reaches the shared tree DQ = T_U + SL + LM + LA + QD - TA pe
  // cycles after its A window. At ns = 16, 24 and 32 the windows (the last RL segments of each
  // row) never meet, and stay aligned mod RL, iff DQ = RL * o with o odd and not a multiple of
  // 3: T_U is padded to the smallest such DQ (DQ 97 -> 104 at TA 7, QD 1)
  function automatic int q_pad(input int t0);
    int d, o;
    d = t0 + SL + LM + LA + QD - TA;
    o = (d + RL - 1) / RL;
    while (o % 2 == 0 || o % 3 == 0) o++;
    return o * RL - d;
  endfunction
  localparam int T_U = T_U0 + (ONE_TREE ? q_pad(T_U0) : 0);
  localparam int UBD = 128;                   // delay buffer (> T_U)
  localparam int DFD = 8;                     // d FIFO
  initial if (L != 8 || T_U >= UBD || RL < LA + 1)
    $fatal(1, "otpu_se_tail: unsupported LANES / TA");

  localparam f32_t F_NZ = 32'h8000_0000;      // -0: (a*b) + -0 == a*b exactly

  // ---------------------------------------------------------------- the modes (static)
  // registered from cfg (it is set before the fill and holds through the stream) and from
  // K0 (set by the fill, before the stream)
  logic [5:0] ns;
  logic       c_asel, c_kz, c_a2p, c_c2z, c_gcol, c_qen, c_pad;
  f32_t       e, beta;                        // K0, K1
  f32_t       b1, gk;                         // slot 1's b (K0 or 1); G unless a column
  always_ff @(posedge clk) begin
    ns     <= cfg.ns;
    c_asel <= cfg.a_sel;
    c_kz   <= !cfg.a_en;                                    // kv = +0 without A
    c_a2p  <= cfg.dmode == SD_DOT;                          // slot 2's a = p1 (else x)
    c_c2z  <= cfg.dmode == SD_SCALE || cfg.dmode == SD_DOT; // slot 2's c = -0 (else -p1)
    c_gcol <= cfg.g_src == SG_COL;
    c_qen  <= cfg.q_en;
    c_pad  <= ONE_TREE && cfg.pad64;
    b1     <= (cfg.dmode == SD_DELTA) ? e : F_ONE;
    gk     <= (cfg.g_src == SG_ONE) ? F_ONE : e;
  end

  // ---------------------------------------------------------------- per-stream vectors
  f32_t kx [L];                               // A's vector at the input segment's columns
  f32_t ku [L];                               // B(c) of the segment at U
  f32_t qu [L];                               // Q(c) of the segment at U
  f32_t gu [L];                               // G(c) of the segment at U
  logic [CBW-1:0] xj, uj;                     // segment index in its row at X / at U
  f32_t vr;                                   // x of the row whose kv is on the d stage
  logic [7:0] rv;                             // that row
  f32_t vl [L];                               // x(rv - rv % L + l)
  wire  fa = (fk == SF_K && !c_asel) || fk == SF_A;
  for (genvar l = 0; l < L; l++) begin : g_buf
    f32_t kb [CBD], kb2 [CBD], qb [CBD], vb [CBD], gb [CBD];
    always_ff @(posedge clk) begin
      if (fa) kb[fi] <= fd[l];
      if (fk == SF_K) kb2[fi] <= fd[l];
      if (fk == SF_Q) qb[fi] <= fd[l];
      if (fk == SF_X) vb[fi] <= fd[l];
      if (fk == SF_G) gb[fi] <= fd[l];
    end
    // pad64: the columns past 64 (segments 8..15) read +0
    assign kx[l] = (c_pad && xj[CBW-1:3] != '0) ? F_ZERO : kb[xj];
    assign ku[l] = (c_pad && uj[CBW-1:3] != '0) ? F_ZERO : kb2[uj];
    assign qu[l] = (c_pad && uj[CBW-1:3] != '0) ? F_ZERO : qb[uj];
    assign gu[l] = (c_pad && uj[CBW-1:3] != '0) ? F_ZERO : gb[uj];
    assign vl[l] = vb[rv[7:LW]];
  end
  assign vr = vl[rv[LW-1:0]];
  always_ff @(posedge clk) begin
    if (fk == SF_K0) e <= fd[0];
    if (fk == SF_K1) beta <= fd[0];
  end

  // ---------------------------------------------------------------- X: the input segment
  function automatic ss_meta_t meta_of(input logic v, input logic [CBW-1:0] j,
                                       input logic [5:0] n);
    ss_meta_t m;
    m.v = v;
    m.j = 5'(j);
    m.first = v && (32'(j) < RL);
    m.final_ = v && (32'(j) + RL >= 32'(n));
    m.row_last = v && (32'(j) + 1 == 32'(n));
    m.sub = 8'(32'(j) % RL);
    return m;
  endfunction

  logic [CBW-1:0] jn;                         // segment index of the next input in its row
  ss_meta_t x0;
  always_ff @(posedge clk) begin
    if (rst || init) begin
      jn <= '0;
      x0 <= '0;
    end else if (pe) begin
      x0 <= meta_of(in_v, jn, ns);
      if (in_v) jn <= (32'(jn) + 1 == 32'(ns)) ? '0 : jn + 1'b1;
    end
    if (pe) begin
      xd <= in_d;
      xa <= kx;
    end
  end
  assign xj = jn;
  assign xm = x0;

  // ---------------------------------------------------------------- D: d from kv, x, K0, K1
  // three multiply-adds with registered inputs (see the header for the modes)
  f32_t  d1a, d2a, d2c, d3a, p1, s2, dd, vd;
  logic  d1v, d2v, d3v, dv;
  always_ff @(posedge clk) begin
    if (rst || init) begin
      rv <= '0;
    end else if (pe && kv_v) begin
      rv <= rv + 1'b1;
    end
    if (pe) begin
      d1a <= c_kz ? F_ZERO : kv;
      d2a <= c_a2p ? p1 : vd;
      d2c <= c_c2z ? F_NZ : fneg(p1);
      d3a <= s2;
    end
  end
  otpu_delay #(.W(1), .N(1)) u_d1v (.clk, .en(pe), .d(kv_v), .q(d1v));
  otpu_delay #(.W(32), .N(SL)) u_vd (.clk, .en(pe), .d(vr), .q(vd));
  otpu_fmadd #(.LM(LM), .LA(LA)) u_d1 (.clk, .en(pe), .a(d1a), .b(b1), .c(F_NZ), .y(p1));
  otpu_delay #(.W(1), .N(SL)) u_d2v (.clk, .en(pe), .d(d1v), .q(d2v));
  otpu_fmadd #(.LM(LM), .LA(LA)) u_d2 (.clk, .en(pe), .a(d2a), .b(F_ONE), .c(d2c), .y(s2));
  otpu_delay #(.W(1), .N(SL)) u_d3v (.clk, .en(pe), .d(d2v), .q(d3v));
  otpu_fmadd #(.LM(LM), .LA(LA)) u_d3 (.clk, .en(pe), .a(d3a), .b(beta), .c(F_NZ), .y(dd));
  otpu_delay #(.W(1), .N(LM + LA)) u_dv (.clk, .en(pe), .d(d3v), .q(dv));
  // b1 and beta are read by u_d1 / u_d3 straight from their registers (set before the stream)

  // d FIFO: pushed as each row's d is ready, popped at each row's first segment at U
  f32_t          dq [DFD];
  logic [$clog2(DFD):0] dq_n;
  logic [$clog2(DFD)-1:0] dq_h, dq_t;
  f32_t          d_cur;
  wire           d_pop;

  // ---------------------------------------------------------------- U: the delayed segments
  // a block RAM delay line of T_U pe cycles: written at wc, read (registered) at wc - (T_U - 1)
  ss_meta_t um;
  f32_t  ud [L];
  logic [$clog2(UBD)-1:0] wc;
  (* ram_style = "block" *) logic [L*32+$bits(ss_meta_t)-1:0] ub [UBD];
  logic [L*32+$bits(ss_meta_t)-1:0] ub_q, ub_d;
  always_comb begin
    ub_d[L*32 +: $bits(ss_meta_t)] = x0;
    for (int l = 0; l < L; l++) ub_d[32 * l +: 32] = xd[l];
  end
  always_ff @(posedge clk) begin
    if (pe) begin
      ub[wc] <= ub_d;
      ub_q <= ub[wc - $clog2(UBD)'(T_U - 1)];
    end
  end
  always_ff @(posedge clk)
    if (rst || init) wc <= '0;
    else if (pe) wc <= wc + 1'b1;
  // the first T_U pe cycles read words never written: their valid bit is masked
  logic [$clog2(UBD):0] warm;
  always_ff @(posedge clk)
    if (rst || init) warm <= '0;
    else if (pe && warm != ($clog2(UBD)+1)'(T_U)) warm <= warm + 1'b1;
  always_comb begin
    um = ub_q[L*32 +: $bits(ss_meta_t)];
    if (warm != ($clog2(UBD)+1)'(T_U)) um = '0;
    for (int l = 0; l < L; l++) ud[l] = ub_q[32 * l +: 32];
  end
  assign uj = CBW'(um.j);
  assign d_pop = pe && um.v && um.j == '0;

  always_ff @(posedge clk) begin
    if (rst || init) begin
      dq_n <= '0; dq_h <= '0; dq_t <= '0;
    end else if (pe) begin
      if (dv) begin
        dq[dq_t] <= dd;
        dq_t <= dq_t + 1'b1;
      end
      if (d_pop) dq_h <= dq_h + 1'b1;
      dq_n <= dq_n + ($clog2(DFD)+1)'(dv) - ($clog2(DFD)+1)'(d_pop);
    end
    if (d_pop) d_cur <= dq[dq_h];
  end
`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && pe && ((d_pop && dq_n == 0) || (dv && !d_pop && 32'(dq_n) == DFD)))
      $fatal(1, "otpu_se_tail: d FIFO %s", (d_pop && dq_n == 0) ? "underflow" : "overflow");
`endif
  f32_t d_row;
  assign d_row = (um.v && um.j == '0) ? dq[dq_h] : d_cur;

  // Y = S G + d B: the VPU's OUTER slot (a*b + c*e), inputs registered. pad64: a padded
  // segment (S = +0 from the DMA, B = +0) also takes G = d = +0, so its Y is +0 (no inf * 0
  // reaches Q) and its Q terms are +0 (its Y is dropped by the DMA)
  f32_t  nw [L];
  ss_meta_t um_y;
  wire   upad = c_pad && um.j[4:3] != '0;
  // U's and Q's units: on pe, or (COMP8) on cen, with the composite's operands in VOP mode
  wire   ue = COMP8 ? cen : pe;
  wire   uc = COMP8 && cm;
  for (genvar l = 0; l < L; l++) begin : g_u
    f32_t ra, rg, rc, re;
    always_ff @(posedge clk) if (ue) begin
      if (uc) begin
        ra <= u_a[l]; rg <= u_b[l]; rc <= u_c[l]; re <= u_e[l];
      end else begin
        ra <= ud[l]; rg <= upad ? F_ZERO : c_gcol ? gu[l] : gk; rc <= upad ? F_ZERO : d_row;
        re <= ku[l];
      end
    end
    otpu_fmma #(.LM(LM), .LA(LA)) u_ma (.clk, .en(ue), .a(ra), .b(rg), .c(rc), .e(re), .y(nw[l]));
    assign u_y[l] = nw[l];
  end
  otpu_delay #(.W($bits(ss_meta_t)), .N(SL)) u_uy (.clk, .en(pe), .d(um), .q(um_y));
  assign y_v = um_y.v;
  assign y_d = nw;

  // ---------------------------------------------------------------- Q: o = Y . q
  f32_t  qy [L];                              // q(c) aligned with nw
  for (genvar l = 0; l < L; l++) begin : g_qd
    otpu_delay #(.W(32), .N(SL)) u_q (.clk, .en(pe), .d(qu[l]), .q(qy[l]));
  end
  f32_t  pq [L];
  ss_meta_t yq_q, yq_t;
  otpu_delay #(.W($bits(ss_meta_t)), .N(LM)) u_yq (.clk, .en(pe), .d(um_y), .q(yq_q));
  otpu_delay #(.W($bits(ss_meta_t)), .N(LA)) u_yt (.clk, .en(pe), .d(yq_q), .q(yq_t));
  logic  fq_e;
  otpu_delay #(.W(1), .N(LM - 1)) u_fq (.clk, .en(pe), .d(um_y.first), .q(fq_e));
  // COMP8 in VOP mode: a*b + c from input registers (qa, qb, qc), c reaching the adder through
  // fbq (the partial loop's register) as the product does
  for (genvar l = 0; l < L; l++) begin : g_lq
    f32_t tq, prev, fbd, fbq, qa, qb, qc, qc2, ma, mb;
    if (COMP8) begin : g_qi
      always_ff @(posedge clk) if (ue) begin
        qa <= q_a[l]; qb <= q_b[l]; qc <= q_c[l]; qc2 <= qc;
      end
      assign ma = uc ? qa : nw[l];
      assign mb = uc ? qb : qy[l];
    end else begin : g_qs
      assign ma = nw[l];
      assign mb = qy[l];
      assign qc2 = F_ZERO;
    end
    otpu_fmul #(.LAT(LM)) u_m (.clk, .en(ue), .a(ma), .b(mb), .y(tq));
    otpu_delay #(.W(32), .N(RL - LA - 1)) u_fb (.clk, .en(ue), .d(pq[l]), .q(fbd));
    always_ff @(posedge clk) if (ue) fbq <= uc ? qc2 : fq_e ? F_ZERO : fbd;
    assign prev = fbq;
    otpu_fadd #(.LAT(LA)) u_a (.clk, .en(ue), .a(prev), .b(tq), .y(pq[l]));
    assign q_y[l] = pq[l];
  end
  if (ONE_TREE) begin : g_q1
    // the final partials to the VPU's tree (in Q's windows, see T_U), the root back
    assign qp_cap = yq_t.v && yq_t.final_ && c_qen;
    assign qp_row_last = yq_t.row_last;
    assign qp_sub = yq_t.sub;
    assign qp_d = pq;
    assign o_d = qo_d;
    assign o_v = qo_v && c_qen;
  end else begin : g_q2
    logic tq_v;
    otpu_vtree #(.LANES(L), .LA(LA)) u_tq (.clk, .rst(rst || init), .en(pe), .pacc(pq),
                                           .cap(yq_t.v && yq_t.final_), .row_last(yq_t.row_last),
                                           .sub(yq_t.sub), .root(o_d), .root_v(tq_v));
    assign o_v = tq_v && c_qen;
    assign qp_cap = 1'b0;
    assign qp_row_last = 1'b0;
    assign qp_sub = '0;
    for (genvar l = 0; l < L; l++) begin : g_qz
      assign qp_d[l] = '0;
    end
  end
endmodule
