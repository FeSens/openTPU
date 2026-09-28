// DSTEP datapath (docs/isa.md DSTEP): one Gated DeltaNet head step on state rows streaming
// through, L = LANES words (one segment) per cycle, bit-exact with the ISA's RDOT, MUL, SUB,
// MUL, OUTER, RDOT sequence. The DMA (otpu_dma) fills the per-head vectors, feeds the state's
// segments in DRAM order and takes the updated segments and the row outputs o.
//
//   X   the input segment (register), with k(c) from the column buffer
//   A   kv = isum_64(S[r, c] * k(c)): the RDOT partial loop (RL = 64 / L chunks) and the folding
//       tree (otpu_vtree)
//   D   d = (v(r) - kv * e) * beta: three multiply-adds (the VPU's MUL, SUB, MUL), into a FIFO
//   U   S'[r, c] = S[r, c] * e + d * k(c) (the VPU's OUTER), S read back from a delay buffer
//       T_U cycles after X; the updated segment leaves (y)
//   Q   o = isum_64(S'[r, c] * q(c)): a second partial loop and tree; o leaves (o_v)
//
// Every register advances with `pe` (registered by the DMA): the pipeline freezes as a whole,
// so the trees' fixed schedules and the delay T_U hold in pe cycles. The DMA feeds a row's
// segments on consecutive pe cycles (bubbles, in_v = 0, only after the last row) and cols is a
// multiple of 64, so rows start a multiple of RL pe cycles apart, as otpu_vtree needs.
module otpu_dstep
  import otpu_fp::*;
#(
  parameter int LANES = 8
) (
  input  logic             clk,
  input  logic             rst,
  input  logic             init,        // a DSTEP starts: counters cleared
  input  logic [5:0]       ns,          // segments per row: cols / LANES
  // fill (the DMA's TMEM reads): q, k, v segments at index fi, e, beta (lane 0)
  input  logic             fq, fk, fv, fe, fb,
  input  logic [4:0]       fi,
  input  f32_t             fd [LANES],
  // stream
  input  logic             pe,
  input  logic             in_v,
  input  f32_t             in_d [LANES],
  output logic             y_v,         // an updated segment (with pe)
  output f32_t             y_d [LANES],
  output logic             o_v,         // a row's o (with pe)
  output f32_t             o_d
);
  localparam int L = LANES;
  localparam int LM = 2, LA = 4;
  localparam int NP = 64;
  localparam int RL = NP / L;
  localparam int CBD = 256 / L;               // column buffers: 256 columns over the lanes
  localparam int CBW = $clog2(CBD);
  localparam int LW = $clog2(L);
  // X -> U: the last segment of a row reaches the tree LM + LA later, its root RD later
  // (otpu_vtree: at most 28 for LANES 8 or 16), d three multiply-adds (21) and a FIFO write
  // after that; a row's first segment is ns - 1 ahead of its last (ns <= CBD)
  localparam int RD_MAX = 28;
  localparam int T_U = (CBD - 1) + LM + LA + RD_MAX + 3 * (1 + LM + LA) + 3;
  localparam int UBD = 128;                   // delay buffer (> T_U)
  localparam int DFD = 8;                     // d FIFO
  initial if (T_U >= UBD || RL < LA + 1 || NP % L != 0)
    $fatal(1, "otpu_dstep: unsupported LANES");

  localparam f32_t F_NZ = 32'h8000_0000;      // -0: (a*b) + -0 == a*b exactly

  // ---------------------------------------------------------------- per-head vectors
  f32_t e, beta;
  f32_t kx [L];                               // k(c) of the input segment
  f32_t ku [L];                               // k(c) of the segment at U
  f32_t qu [L];                               // q(c) of the segment at U
  logic [CBW-1:0] xj, uj;                     // segment index in its row at X / at U
  f32_t vr;                                   // v of the row whose root is on tree A
  logic [7:0] rv;                             // that row
  f32_t vl [L];                               // v(rv - rv % L + l)
  for (genvar l = 0; l < L; l++) begin : g_buf
    f32_t kb [CBD], kb2 [CBD], qb [CBD], vb [CBD];
    always_ff @(posedge clk) begin
      if (fk) begin kb[fi] <= fd[l]; kb2[fi] <= fd[l]; end
      if (fq) qb[fi] <= fd[l];
      if (fv) vb[fi] <= fd[l];
    end
    assign kx[l] = kb[xj];
    assign ku[l] = kb2[uj];
    assign qu[l] = qb[uj];
    assign vl[l] = vb[rv[7:LW]];
  end
  assign vr = vl[rv[LW-1:0]];
  always_ff @(posedge clk) begin
    if (fe) e <= fd[0];
    if (fb) beta <= fd[0];
  end

  // ---------------------------------------------------------------- X: the input segment
  typedef struct packed {
    logic             v;
    logic             first;     // chunk index in its row < RL (partials start at +0)
    logic             final_;    // chunk index >= ns - RL (partials become final)
    logic             row_last;
    logic [7:0]       sub;       // chunk index mod RL
    logic [CBW-1:0]   j;
  } sm_t;
  function automatic sm_t meta_of(input logic v, input logic [CBW-1:0] j, input logic [5:0] n);
    sm_t m;
    m.v = v;
    m.j = j;
    m.first = v && (32'(j) < RL);
    m.final_ = v && (32'(j) + RL >= 32'(n));
    m.row_last = v && (32'(j) + 1 == 32'(n));
    m.sub = 8'(32'(j) % RL);
    return m;
  endfunction

  logic [CBW-1:0] jn;                         // segment index of the next input in its row
  sm_t   x0;
  f32_t  xd [L], xk [L];
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
      xk <= kx;
    end
  end
  assign xj = jn;

  // ---------------------------------------------------------------- A: kv partial loop + tree
  // (the VPU's RDOT, with a squarer-style multiplier and adder per lane: pacc = prev + S * k)
  f32_t  pa [L];
  sm_t   xa_q, xa_t;                          // the meta at the adder inputs / aligned with pa
  otpu_delay #(.W($bits(sm_t)), .N(LM)) u_xq (.clk, .en(pe), .d(x0), .q(xa_q));
  otpu_delay #(.W($bits(sm_t)), .N(LA)) u_xt (.clk, .en(pe), .d(xa_q), .q(xa_t));
  logic  fa_e;                                // x0.first LM - 1 later
  otpu_delay #(.W(1), .N(LM - 1)) u_fa (.clk, .en(pe), .d(x0.first), .q(fa_e));
  for (genvar l = 0; l < L; l++) begin : g_la
    f32_t tq, prev, fbd, fbq;
    otpu_fmul #(.LAT(LM)) u_m (.clk, .en(pe), .a(xd[l]), .b(xk[l]), .y(tq));
    otpu_delay #(.W(32), .N(RL - LA - 1)) u_fb (.clk, .en(pe), .d(pa[l]), .q(fbd));
    always_ff @(posedge clk) if (pe) fbq <= fa_e ? F_ZERO : fbd;
    assign prev = fbq;
    otpu_fadd #(.LAT(LA)) u_a (.clk, .en(pe), .a(prev), .b(tq), .y(pa[l]));
  end
  f32_t  kv;
  logic  kv_v;
  otpu_vtree #(.LANES(L), .LA(LA)) u_ta (.clk, .rst(rst || init), .en(pe), .pacc(pa),
                                         .cap(xa_t.v && xa_t.final_), .row_last(xa_t.row_last),
                                         .sub(xa_t.sub), .root(kv), .root_v(kv_v));

  // ---------------------------------------------------------------- D: d = (v - kv e) beta
  // three multiply-adds with registered inputs, the VPU's MUL (a*b + -0), SUB (v*1 + -p), MUL
  localparam int SL = 1 + LM + LA;
  f32_t  d1a, d2a, d2c, d3a, p1, s2, dd, vd;
  logic  d1v, d2v, d3v, dv;
  always_ff @(posedge clk) begin
    if (rst || init) begin
      rv <= '0;
    end else if (pe && kv_v) begin
      rv <= rv + 1'b1;
    end
    if (pe) begin
      d1a <= kv;
      d2a <= vd;
      d2c <= fneg(p1);
      d3a <= s2;
    end
  end
  otpu_delay #(.W(1), .N(1)) u_d1v (.clk, .en(pe), .d(kv_v), .q(d1v));
  otpu_delay #(.W(32), .N(SL)) u_vd (.clk, .en(pe), .d(vr), .q(vd));
  otpu_fmadd #(.LM(LM), .LA(LA)) u_d1 (.clk, .en(pe), .a(d1a), .b(e), .c(F_NZ), .y(p1));
  otpu_delay #(.W(1), .N(SL)) u_d2v (.clk, .en(pe), .d(d1v), .q(d2v));
  otpu_fmadd #(.LM(LM), .LA(LA)) u_d2 (.clk, .en(pe), .a(d2a), .b(F_ONE), .c(d2c), .y(s2));
  otpu_delay #(.W(1), .N(SL)) u_d3v (.clk, .en(pe), .d(d2v), .q(d3v));
  otpu_fmadd #(.LM(LM), .LA(LA)) u_d3 (.clk, .en(pe), .a(d3a), .b(beta), .c(F_NZ), .y(dd));
  otpu_delay #(.W(1), .N(LM + LA)) u_dv (.clk, .en(pe), .d(d3v), .q(dv));
  // e and beta are read by u_d1 / u_d3 straight from their registers (set in the fill)

  // d FIFO: pushed as each row's d is ready, popped at each row's first segment at U
  f32_t          dq [DFD];
  logic [$clog2(DFD):0] dq_n;
  logic [$clog2(DFD)-1:0] dq_h, dq_t;
  f32_t          d_cur;
  wire           d_pop;

  // ---------------------------------------------------------------- U: the delayed segments
  // a block RAM delay line of T_U pe cycles: written at wc, read (registered) at wc - (T_U - 1)
  sm_t   um;
  f32_t  ud [L];
  logic [$clog2(UBD)-1:0] wc;
  (* ram_style = "block" *) logic [L*32+$bits(sm_t)-1:0] ub [UBD];
  logic [L*32+$bits(sm_t)-1:0] ub_q, ub_d;
  always_comb begin
    ub_d[L*32 +: $bits(sm_t)] = x0;
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
    um = ub_q[L*32 +: $bits(sm_t)];
    if (warm != ($clog2(UBD)+1)'(T_U)) um = '0;
    for (int l = 0; l < L; l++) ud[l] = ub_q[32 * l +: 32];
  end
  assign uj = um.j;
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
      $fatal(1, "otpu_dstep: d FIFO %s", (d_pop && dq_n == 0) ? "underflow" : "overflow");
`endif
  f32_t d_row;
  assign d_row = (um.v && um.j == '0) ? dq[dq_h] : d_cur;

  // S' = S e + d k: the VPU's OUTER slot (a*b + c*e), inputs registered
  f32_t  nw [L];
  sm_t   um_y;
  for (genvar l = 0; l < L; l++) begin : g_u
    f32_t ra, rc, re;
    always_ff @(posedge clk) if (pe) begin
      ra <= ud[l]; rc <= d_row; re <= ku[l];
    end
    otpu_fmma #(.LM(LM), .LA(LA)) u_ma (.clk, .en(pe), .a(ra), .b(e), .c(rc), .e(re), .y(nw[l]));
  end
  otpu_delay #(.W($bits(sm_t)), .N(SL)) u_uy (.clk, .en(pe), .d(um), .q(um_y));
  assign y_v = um_y.v;
  assign y_d = nw;

  // ---------------------------------------------------------------- Q: o = S' . q
  f32_t  qy [L];                              // q(c) aligned with nw
  for (genvar l = 0; l < L; l++) begin : g_qd
    otpu_delay #(.W(32), .N(SL)) u_q (.clk, .en(pe), .d(qu[l]), .q(qy[l]));
  end
  f32_t  pq [L];
  sm_t   yq_q, yq_t;
  otpu_delay #(.W($bits(sm_t)), .N(LM)) u_yq (.clk, .en(pe), .d(um_y), .q(yq_q));
  otpu_delay #(.W($bits(sm_t)), .N(LA)) u_yt (.clk, .en(pe), .d(yq_q), .q(yq_t));
  logic  fq_e;
  otpu_delay #(.W(1), .N(LM - 1)) u_fq (.clk, .en(pe), .d(um_y.first), .q(fq_e));
  for (genvar l = 0; l < L; l++) begin : g_lq
    f32_t tq, prev, fbd, fbq;
    otpu_fmul #(.LAT(LM)) u_m (.clk, .en(pe), .a(nw[l]), .b(qy[l]), .y(tq));
    otpu_delay #(.W(32), .N(RL - LA - 1)) u_fb (.clk, .en(pe), .d(pq[l]), .q(fbd));
    always_ff @(posedge clk) if (pe) fbq <= fq_e ? F_ZERO : fbd;
    assign prev = fbq;
    otpu_fadd #(.LAT(LA)) u_a (.clk, .en(pe), .a(prev), .b(tq), .y(pq[l]));
  end
  otpu_vtree #(.LANES(L), .LA(LA)) u_tq (.clk, .rst(rst || init), .en(pe), .pacc(pq),
                                         .cap(yq_t.v && yq_t.final_), .row_last(yq_t.row_last),
                                         .sub(yq_t.sub), .root(o_d), .root_v(o_v));
endmodule
