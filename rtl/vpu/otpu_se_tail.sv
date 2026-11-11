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
//       MUL, SUB, MUL) on one unit, SL pe cycles apart, into a FIFO:
//         DELTA   d = (x - kv * K0) * K1        DELTA1  d = (x - kv) * K1
//         SCALE   d = x * K1                    DOT     d = kv * K1
//       always DELTA's three slots: slot 1 = (kv or +0) * (K0 or 1) + -0, slot 2 = (x or
//       p1) * 1 + (-p1 or -0), slot 3 = s2 * K1 + -0. mul(z, 1) == z and add(z, -0) == z
//       exactly for every z under FTZ (-0 and NaN included), so each mode is its op list
//   U   Y[r, c] = S[r, c] * G(c) + d * B(c) (the VPU's OUTER), G = K0, slot 3 or 1.0, B slot
//       1; S read back from a delay buffer T_U cycles after X; the updated segment leaves (y)
//   Q   o = isum_64(Y[r, c] * Q(c)), Q slot 0: a partial loop here; o leaves (o_v) with
//       cfg.q_en. One tree (docs/stream.md 11.3): Q's final partials go to the VPU's u_vt
//       (qp_*) in windows between A's, and the root comes back (qo_*): T_U is padded so that
//       a row's Q window is DQ = 104 pe cycles after its A window (see T_U). cfg.pad64: cols =
//       64 presented as 16 segments per row, the last 8 +0 (the DMA drops their Y); the
//       column buffers read +0 there
//
// A's vector: slot 1 (k) or slot 2 (a, cfg.a_sel) have one buffer at X. An SF_K fill writes
// it unless a_sel, an SF_A fill always, so xa needs no mux. B keeps its own copy of slot 1.
//
// Every register advances with `pe` (registered by the DMA): the pipeline freezes as a whole,
// so the trees' fixed schedules and the delay T_U hold in pe cycles. The DMA feeds a row's
// segments on consecutive pe cycles (bubbles, in_v = 0, only after the last row) and cols is a
// multiple of 64, so rows start a multiple of RL pe cycles apart, as otpu_vtree needs. The VPU
// runs A on every segment whatever cfg.a_en (kv_v times the rows of the d stage).
//
// R row groups (otpu_se_eng's; the VPU's tail is R = 1): each pe cycle takes R segments, the same
// segment of R consecutive rows (group g: rows R i + g), and each group has its own d stage,
// delayed segment, U and Q lanes and row output; the column buffers, the modes, the control and
// the d FIFO's pointers are shared (every group is at the same column in the same cycle), and a
// group's x is the buffer word of its row (rows R i .. R i + R - 1 sit in R different lanes'
// buffers). Every row's operations are those of R = 1, so the results are the same bit for bit.
// The ports per group: in_d, xd, y_d and qp_d hold group g's lanes at g LANES .. g LANES + LANES - 1,
// kv, o_d and qo_d its word at [g]; with R = 1 they are R = 1's ports. The VOP-mode stages (u_*,
// q_*, otpu_se_comp's) are group 0's.
//
// DSTEP2 (cfg.pair, DSTEP's pair; docs/isa.md): two heads in one stream, rows r >= rows / 2 head
// b's. The fill writes head b's vectors after head a's in the column buffers (cols 128: indices
// 16..31) and its K0 and K1 at fi 1; each stage takes its row's head (X and U count their rows,
// D compares its row, Q delays U's) and reads head b's index (+ 16), K0 and K1 for head b's rows.
module otpu_se_tail
  import otpu_pkg::*;
  import otpu_fp::*;
#(
  parameter int LANES = 8,
  parameter int TA    = 7,       // pe cycles from the X registers to A's partials (VPU slot 0)
  // Q's final partials go to the VPU's tree (qp_*, QD registers there) and its root comes back
  // (qo_*): no tree of its own (docs/stream.md 11.3)
  parameter int QD    = 1,
  parameter int R     = 1,       // row groups (rows a pe cycle): 1, 2, 4 or 8
  parameter bit DSTEP2 = 1'b0    // DSTEP's pair (cfg.pair): the heads' second vectors and K0, K1
) (
  input  logic             clk,
  input  logic             rst,
  input  logic             init,        // a stream starts: counters cleared
  input  ss_cfg_t          cfg,         // valid from before the fill to the stream's last o
  // fill (the DMA's TMEM reads): a vector segment at index fi, a row scalar segment, K0, K1
  input  logic [2:0]       fk,          // SF_*
  input  logic [4:0]       fi [LANES],  // the segment's index, a copy per lane (the VPU's g_sfi)
  input  f32_t             fd [LANES],
  // stream
  input  logic             pe,
  input  logic             in_v,
  input  f32_t             in_d [R*LANES],
  output f32_t             xd [R*LANES],  // X: the segments, A's vector at their columns, meta
  output f32_t             xa [LANES],
  output ss_meta_t         xm,
  input  logic [R-1:0][31:0] kv,        // A: each group's row dot (from the VPU's tree)
  input  logic             kv_v,
  output logic             y_v,         // updated segments (with pe)
  output f32_t             y_d [R*LANES],
  output logic             o_v,         // the rows' o (with pe)
  output logic [R-1:0][31:0] o_d,
  // Q's final partials to the VPU's tree (with pe), the rows' o back
  output logic             qp_cap,
  output logic             qp_row_last,
  output logic [7:0]       qp_sub,
  output f32_t             qp_d [R*LANES],
  input  logic             qo_v,
  input  logic [R-1:0][31:0] qo_d,
  // U (y = a*b + c*e) and Q (y = a*b + c) are also the VPU's composite stages (otpu_se_comp)
  // in VOP mode: generic stages, SL cen-cycles from the cycle their operands are presented
  // (u_sel / q_sel) to u_y / q_y. Their units advance on cen (the VPU's enable, which is pe in
  // stream mode) (docs/stream.md 11.2). VOP mode comes as copies of one bit, two per lane (cm_u:
  // U's operand registers, cm_q: Q's multiplier operands and partial loop), each its own
  // flip-flop in otpu_vpu, so no one net reaches every lane's selects; the enable too (cen: a
  // copy per lane, the VPU's en_l)
  input  logic [LANES-1:0] cm_u,
  input  logic [LANES-1:0] cm_q,
  input  logic [LANES-1:0] cen,
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
  // A row's Q window reaches the shared tree DQ = T_U + SL + LM + LA + QD - TA pe
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
  localparam int T_U = T_U0 + q_pad(T_U0);
  localparam int UBD = 128;                   // delay buffer (> T_U)
  localparam int DFD = 8;                     // d FIFO
  localparam int RW = $clog2(R);
  // (D's one unit a group: a row's three ops are SL and 2 SL pe cycles after its kv, and rows
  // start a multiple of RL pe cycles apart, so neither may be a multiple of RL)
  initial if (L != 8 || T_U >= UBD || RL < LA + 1 || R < 1 || R > L || R != (1 << RW) ||
              SL % RL == 0 || (2 * SL) % RL == 0)
    $fatal(1, "otpu_se_tail: unsupported LANES / TA / R");

  localparam f32_t F_NZ = 32'h8000_0000;      // -0: (a*b) + -0 == a*b exactly

  // ---------------------------------------------------------------- the modes (static)
  // registered from cfg (it is set before the fill and holds through the stream) and from
  // K0 (set by the fill, before the stream)
  logic [5:0] ns;
  logic       c_asel, c_kz, c_a2p, c_c2z, c_gcol, c_qen, c_pad;
  f32_t       e, beta;                        // K0, K1
  f32_t       b1, gk;                         // slot 1's b (K0 or 1); G unless a column
  logic       c_pair;                         // DSTEP2: a pair's stream
  logic [7:0] c_hr;                           // ... its head b's first row
  f32_t       e_b, beta_b, b1_b, gk_b;        // ... head b's K0, K1, b and G
  always_ff @(posedge clk) begin
    ns     <= cfg.ns;
    c_asel <= cfg.a_sel;
    c_kz   <= !cfg.a_en;                                    // kv = +0 without A
    c_a2p  <= cfg.dmode == SD_DOT;                          // slot 2's a = p1 (else x)
    c_c2z  <= cfg.dmode == SD_SCALE || cfg.dmode == SD_DOT; // slot 2's c = -0 (else -p1)
    c_gcol <= cfg.g_src == SG_COL;
    c_qen  <= cfg.q_en;
    c_pad  <= cfg.pad64;
    b1     <= (cfg.dmode == SD_DELTA) ? e : F_ONE;
    gk     <= (cfg.g_src == SG_ONE) ? F_ONE : e;
    c_pair <= DSTEP2 && cfg.pair;
    c_hr   <= cfg.rows[8:1];
    b1_b   <= (cfg.dmode == SD_DELTA) ? e_b : F_ONE;
    gk_b   <= (cfg.g_src == SG_ONE) ? F_ONE : e_b;
  end

  // ---------------------------------------------------------------- per-stream vectors
  f32_t kx [L];                               // A's vector at the input segment's columns
  f32_t ku [L];                               // B(c) of the segment at U
  f32_t qv [L];                               // Q(c) of the segment SL - 1 pe cycles after U
  f32_t gu [L];                               // G(c) of the segment at U
  logic [CBW-1:0] xj, uj, qj;                 // segment index in its row at X / at U / at qv
  logic           hx, hu, hq;                 // DSTEP2: the row there is a pair's head b's
  // The buffers are written only by the fills, before a stream, so a word is read when it is
  // used (at an index delayed with it) instead of being delayed itself
  f32_t vr [R];                               // x of the rows whose p1 is on the d unit
  logic [7:0] rv;                             // the rows' steps by kv (group g's row: R rv + g)
  wire  [7:0] rr = 8'(32'(rv) << RW);          // group 0's row at its kv
  logic [7:0] rd;                             // ... SL pe cycles later, at its p1
  f32_t vl [L];                               // x(rd - rd % L + l)
  wire  fa = (fk == SF_K && !c_asel) || fk == SF_A;
  for (genvar l = 0; l < L; l++) begin : g_buf
    f32_t kb [CBD], kb2 [CBD], qb [CBD], vb [CBD], gb [CBD];
    always_ff @(posedge clk) begin
      if (fa) kb[fi[l]] <= fd[l];
      if (fk == SF_K) kb2[fi[l]] <= fd[l];
      if (fk == SF_Q) qb[fi[l]] <= fd[l];
      if (fk == SF_X) vb[fi[l]] <= fd[l];
      if (fk == SF_G) gb[fi[l]] <= fd[l];
    end
    // pad64: the columns past 64 (segments 8..15) read +0. A pair's head b: index + 16 (its ns)
    assign kx[l] = (c_pad && xj[CBW-1:3] != '0) ? F_ZERO : kb[{xj[CBW-1] | hx, xj[CBW-2:0]}];
    assign ku[l] = (c_pad && uj[CBW-1:3] != '0) ? F_ZERO : kb2[{uj[CBW-1] | hu, uj[CBW-2:0]}];
    assign qv[l] = qb[{qj[CBW-1] | hq, qj[CBW-2:0]}];   // (pad64: zeroed at qy)
    assign gu[l] = (c_pad && uj[CBW-1:3] != '0) ? F_ZERO : gb[uj];
    assign vl[l] = vb[rd[7:LW]];
  end
  for (genvar g = 0; g < R; g++) begin : g_vr
    assign vr[g] = vl[rd[LW-1:0] + LW'(g)];
  end
  wire hk = DSTEP2 && fi[0][0];                        // (a pair's head b: fi 1, lane 0's copy)
  always_ff @(posedge clk) begin
    if (fk == SF_K0 && !hk) e <= fd[0];
    if (fk == SF_K1 && !hk) beta <= fd[0];
    if (fk == SF_K0 && hk) e_b <= fd[0];
    if (fk == SF_K1 && hk) beta_b <= fd[0];
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
  // a pair's head at X: the rows started (R a row step) against head b's first
  logic [8:0] xr;
  always_ff @(posedge clk)
    if (rst || init) xr <= '0;
    else if (pe && in_v && 32'(jn) + 1 == 32'(ns)) xr <= xr + 9'(R);
  assign hx = DSTEP2 && c_pair && xr >= {1'b0, c_hr};

  // ---------------------------------------------------------------- D: d from kv, x, K0, K1
  // three multiply-adds (see the header for the modes) on one unit a group, with registered
  // inputs: a row's slot 1 is presented with its kv, slot 2 SL pe cycles later with p1 (the
  // unit's result then), slot 3 SL after that with s2, and d is out 3 SL after the kv. Rows
  // start a multiple of RL pe cycles apart (cols is a multiple of 64), SL and 2 SL are not
  // multiples of RL, so no two rows' slots meet
  f32_t  dd [R];
  logic  e2, e3, dv;                          // a row's slot 2 / 3 is presented now; d is out
  always_ff @(posedge clk)
    if (rst || init) begin
      rv <= '0;
    end else if (pe && kv_v) begin
      rv <= rv + 1'b1;
    end
  // the valids in reset registers (otpu_vdelay): the lines advance only with pe, inside a
  // stream, so a power-up one would push the first stream's d FIFO
  otpu_vdelay #(.N(SL)) u_e2 (.clk, .rst, .en(pe), .d(kv_v), .q(e2));
  otpu_vdelay #(.N(SL)) u_e3 (.clk, .rst, .en(pe), .d(e2), .q(e3));
  otpu_vdelay #(.N(SL)) u_dv (.clk, .rst, .en(pe), .d(e3), .q(dv));
  otpu_delay #(.W(8), .N(SL)) u_rd (.clk, .en(pe), .d(rr), .q(rd));
`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && pe && 32'(kv_v) + 32'(e2) + 32'(e3) > 1)
      $fatal(1, "otpu_se_tail: two rows' d slots meet (kv_v %0d e2 %0d e3 %0d)", kv_v, e2, e3);
`endif
  // a pair's head at D: slot 1's row (rr, with its kv) and slot 3's (2 SL pe cycles later)
  logic  hd1, hd3;
  assign hd1 = DSTEP2 && c_pair && {1'b0, rr} >= {1'b0, c_hr};
  if (DSTEP2) begin : g_hd
    otpu_delay #(.W(1), .N(2 * SL)) u_hd3 (.clk, .en(pe), .d(hd1), .q(hd3));
  end else begin : g_nhd
    assign hd3 = 1'b0;
  end
  for (genvar g = 0; g < R; g++) begin : g_d
    f32_t  ra, rb, rc, y;
    always_ff @(posedge clk)
      if (pe) begin
        ra <= e3 ? y : e2 ? (c_a2p ? y : vr[g]) : c_kz ? F_ZERO : kv[g];
        rb <= e3 ? (hd3 ? beta_b : beta) : e2 ? F_ONE : hd1 ? b1_b : b1;
        rc <= (e2 && !c_c2z) ? fneg(y) : F_NZ;
      end
    otpu_fmadd #(.LM(LM), .LA(LA)) u_d (.clk, .en(pe), .a(ra), .b(rb), .c(rc), .y);
    assign dd[g] = y;
  end

  // d FIFO: pushed as each row's d is ready, popped at each row's first segment at U (one
  // entry holds every group's d)
  f32_t          dq [DFD][R];
  logic [$clog2(DFD):0] dq_n;
  logic [$clog2(DFD)-1:0] dq_h, dq_t;
  f32_t          d_cur [R];
  wire           d_pop;

  // ---------------------------------------------------------------- U: the delayed segments
  // a block RAM delay line of T_U pe cycles: written at wc, read (registered) at wc - (T_U - 1)
  ss_meta_t um;
  f32_t  ud [R*L];
  logic [$clog2(UBD)-1:0] wc;
  logic [R*L*32+$bits(ss_meta_t)-1:0] ub_q, ub_d;
  always_comb begin
    ub_d[R*L*32 +: $bits(ss_meta_t)] = x0;
    for (int l = 0; l < R*L; l++) ub_d[32 * l +: 32] = xd[l];
  end
  otpu_ram_1r1w #(.W(R*L*32+$bits(ss_meta_t)), .N(UBD), .STYLE("block")) u_ub (
    .clk, .we(pe), .wa(wc), .wd(ub_d), .re(pe), .ra(wc - $clog2(UBD)'(T_U - 1)), .rd(ub_q));
  always_ff @(posedge clk)
    if (rst || init) wc <= '0;
    else if (pe) wc <= wc + 1'b1;
  // the first T_U pe cycles read words never written: their valid bit is masked
  logic [$clog2(UBD):0] warm;
  always_ff @(posedge clk)
    if (rst || init) warm <= '0;
    else if (pe && warm != ($clog2(UBD)+1)'(T_U)) warm <= warm + 1'b1;
  always_comb begin
    um = ub_q[R*L*32 +: $bits(ss_meta_t)];
    if (warm != ($clog2(UBD)+1)'(T_U)) um = '0;
    for (int l = 0; l < R*L; l++) ud[l] = ub_q[32 * l +: 32];
  end
  assign uj = CBW'(um.j);
  assign d_pop = pe && um.v && um.j == '0;
  // a pair's head at U: U's rows done (R a row step) against head b's first
  logic [8:0] ur;
  always_ff @(posedge clk)
    if (rst || init) ur <= '0;
    else if (pe && um.v && um.row_last) ur <= ur + 9'(R);
  assign hu = DSTEP2 && c_pair && ur >= {1'b0, c_hr};

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
  // the checks start once the reset has held two edges (registers start arbitrary, and so does a
  // unit's reset register until the slice's reset pipeline fills: otpu_slice)
  bit rst_p, rst_seen;
  initial begin rst_p = 1'b0; rst_seen = 1'b0; end
  always @(posedge clk) begin rst_p <= rst; if (rst && rst_p) rst_seen <= 1'b1; end
  always_ff @(posedge clk)
    if (!rst && rst_seen && pe && ((d_pop && dq_n == 0) || (dv && !d_pop && 32'(dq_n) == DFD)))
      $fatal(1, "otpu_se_tail: d FIFO %s", (d_pop && dq_n == 0) ? "underflow" : "overflow");
`endif
  f32_t d_row [R];
  for (genvar g = 0; g < R; g++) begin : g_dr
    assign d_row[g] = (um.v && um.j == '0) ? dq[dq_h][g] : d_cur[g];
  end

  // Y = S G + d B: the VPU's OUTER slot (a*b + c*e), inputs registered. pad64: a padded
  // segment (S = +0 from the DMA, B = +0) also takes G = d = +0, so its Y is +0 (no inf * 0
  // reaches Q) and its Q terms are +0 (its Y is dropped by the DMA)
  f32_t  nw [R*L];
  ss_meta_t um_y;
  wire   upad = c_pad && um.j[4:3] != '0;
  // U's and Q's units: on cen (column lane l % L's copy), with the composite's operands in VOP
  // mode (group 0)
  // (lane l of group l / L: column lane l % L; group 0's lanes are l < L)
  for (genvar l = 0; l < R*L; l++) begin : g_u
    f32_t ra, rg, rc, re;
    if (l < L) begin : g_vop                    // group 0: the composite's operands too
      always_ff @(posedge clk) if (cen[l]) begin
        if (cm_u[l]) begin
          ra <= u_a[l]; rg <= u_b[l]; rc <= u_c[l]; re <= u_e[l];
        end else begin
          ra <= ud[l]; rg <= upad ? F_ZERO : c_gcol ? gu[l] : hu ? gk_b : gk;
          rc <= upad ? F_ZERO : d_row[0]; re <= ku[l];
        end
      end
      assign u_y[l] = nw[l];
    end else begin : g_st
      always_ff @(posedge clk) if (cen[l % L]) begin
        ra <= ud[l]; rg <= upad ? F_ZERO : c_gcol ? gu[l % L] : hu ? gk_b : gk;
        rc <= upad ? F_ZERO : d_row[l / L]; re <= ku[l % L];
      end
    end
    otpu_fmma #(.LM(LM), .LA(LA)) u_ma (.clk, .en(cen[l % L]), .a(ra), .b(rg), .c(rc), .e(re), .y(nw[l]));
  end
  // (v in reset registers, as the d line's; the rest under it)
  localparam int MB = $bits(ss_meta_t);
  logic [MB-1:0] ump, umyp;
  assign ump = um;
  otpu_vdelay #(.N(SL)) u_uyv (.clk, .rst, .en(pe), .d(ump[MB-1]), .q(umyp[MB-1]));
  otpu_delay #(.W(MB - 1), .N(SL)) u_uy (.clk, .en(pe), .d(ump[MB-2:0]), .q(umyp[MB-2:0]));
  assign um_y = umyp;
  assign y_v = um_y.v;
  assign y_d = nw;

  // ---------------------------------------------------------------- Q: o = Y . q
  // q(c) aligned with nw: the buffer read at the segment's index SL - 1 pe cycles after U, into
  // a register (+0 for a pad64 segment)
  f32_t  qy [L];
  logic  qpd;
  otpu_delay #(.W(CBW + 1), .N(SL - 1)) u_qj (.clk, .en(pe), .d({upad, uj}), .q({qpd, qj}));
  if (DSTEP2) begin : g_hq                    // (a pair's head at Q: U's, delayed with qj)
    otpu_delay #(.W(1), .N(SL - 1)) u_hq (.clk, .en(pe), .d(hu), .q(hq));
  end else begin : g_nhq
    assign hq = 1'b0;
  end
  for (genvar l = 0; l < L; l++) begin : g_qd
    always_ff @(posedge clk) if (pe) qy[l] <= qpd ? F_ZERO : qv[l];
  end
  // Without O no valid row enters Q: the DMA stops pe once the last Y is out, and a row still
  // on its way through Q would come out in the next stream
  f32_t  pq [R*L];
  ss_meta_t yq_d, yq_q, yq_t;
  assign yq_d = c_qen ? um_y : '0;
  logic [MB-1:0] yqdp, yqqp, yqtp;
  assign yqdp = yq_d;
  otpu_vdelay #(.N(LM)) u_yqv (.clk, .rst, .en(pe), .d(yqdp[MB-1]), .q(yqqp[MB-1]));
  otpu_delay #(.W(MB - 1), .N(LM)) u_yq (.clk, .en(pe), .d(yqdp[MB-2:0]), .q(yqqp[MB-2:0]));
  otpu_vdelay #(.N(LA)) u_ytv (.clk, .rst, .en(pe), .d(yqqp[MB-1]), .q(yqtp[MB-1]));
  otpu_delay #(.W(MB - 1), .N(LA)) u_yt (.clk, .en(pe), .d(yqqp[MB-2:0]), .q(yqtp[MB-2:0]));
  assign yq_q = yqqp;
  assign yq_t = yqtp;
  logic  fq_e;
  otpu_delay #(.W(1), .N(LM - 1)) u_fq (.clk, .en(pe), .d(yq_d.first), .q(fq_e));
  // in VOP mode: a*b + c from input registers (qa, qb, qc), c reaching the adder through fbq
  // (the partial loop's register) as the product does
  for (genvar l = 0; l < R*L; l++) begin : g_lq
    f32_t tq, prev, fbd, fbq, qa, qb, qc, qc2, ma, mb;
    if (l < L) begin : g_cm                     // the composites' operands (group 0)
      always_ff @(posedge clk) if (cen[l]) begin
        qa <= q_a[l]; qb <= q_b[l]; qc <= q_c[l]; qc2 <= qc;
      end
      assign ma = cm_q[l] ? qa : nw[l];
      assign mb = cm_q[l] ? qb : qy[l];
      always_ff @(posedge clk) if (cen[l]) fbq <= cm_q[l] ? qc2 : fq_e ? F_ZERO : fbd;
      assign q_y[l] = pq[l];
    end else begin : g_st
      assign ma = nw[l];
      assign mb = qy[l % L];
      always_ff @(posedge clk) if (cen[l % L]) fbq <= fq_e ? F_ZERO : fbd;
    end
    otpu_fmul #(.LAT(LM)) u_m (.clk, .en(cen[l % L]), .a(ma), .b(mb), .y(tq));
    otpu_delay #(.W(32), .N(RL - LA - 1)) u_fb (.clk, .en(cen[l % L]), .d(pq[l]), .q(fbd));
    assign prev = fbq;
    otpu_fadd #(.LAT(LA)) u_a (.clk, .en(cen[l % L]), .a(prev), .b(tq), .y(pq[l]));
  end
  // the final partials to the VPU's tree (in Q's windows, see T_U), the root back
  assign qp_cap = yq_t.v && yq_t.final_ && c_qen;
  assign qp_row_last = yq_t.row_last;
  assign qp_sub = yq_t.sub;
  assign qp_d = pq;
  assign o_d = qo_d;
  assign o_v = qo_v && c_qen;
endmodule

// A stream engine of its own (otpu_slice SE_R > 0; docs/se_engine.md): the stream engine of
// otpu_vpu (dot A on slot 0's partial loop and the folding tree, then otpu_se_tail) as a unit
// beside the VPU, with R row groups. The DMA's streams (DSTEP, STREAM) run here, so the VPU
// keeps running VOPs while a state steps (in the VPU's engine they take turns: a stream holds
// the VPU, docs/stream.md 4.2), and R rows step a pe cycle: R x LANES state words, R segments in
// (the same segment of rows R i .. R i + R - 1), R updated segments and R row outputs out.
// Per group: LANES multiply-adds of dot A in a loop of RL pe cycles (the RDOT path's bits: slot
// 0's a * b + c from registered inputs; c is the loop's own partial, so the VPU's c * 1.0 is left
// out: mul(z, 1) == z under FTZ), its tree (dot A's final partials and the tail's dot Q's in
// windows between them, as otpu_vpu's u_vt), the tail's group. The DMA's side is the
// VPU's: ss_gnt a cycle after ss_req (nothing to drain), every ss_* input registered twice (SE_LAG
// 2), ss_y_v / ss_o_v qualified with the engine's pe.
module otpu_se_eng
  import otpu_pkg::*;
  import otpu_fp::*;
#(
  parameter int LANES = 8,
  parameter int R     = 1,
  parameter bit DSTEP2 = 1'b0       // DSTEP's pair (otpu_se_tail)
) (
  input  logic               clk,
  input  logic               rst,
  input  logic               ss_req,
  output logic               ss_gnt,
  input  ss_cfg_t            ss_cfg,
  input  logic               ss_pe,
  input  logic               ss_in_v,
  input  f32_t               ss_in_d [R*LANES],
  input  logic [2:0]         ss_fk,
  input  logic [4:0]         ss_fi,
  input  f32_t               ss_fd [LANES],
  output logic               ss_y_v,
  output f32_t               ss_y_d [R*LANES],
  output logic               ss_o_v,
  output logic [R-1:0][31:0] ss_o_d
);
  localparam int L = LANES;
  localparam int LM = 2, LA = 4;
  localparam int RL = 64 / L;
  localparam int TA = 1 + LM + LA;          // X -> dot A's partials (slot 0's input register + LM + LA)
  localparam int DF = RL - TA;              // the loop's feedback delay (the VPU's RMA)
  localparam f32_t F_NZ = 32'h8000_0000;
  initial if (L != 8 || DF < 1) $fatal(1, "otpu_se_eng: LANES 8");

  // the grant, and the inputs through two registers (otpu_vpu's g_se)
  logic       ss_rq, ss_act, s_init, sen_q, sen, s1_in_v, s_in_v;
  logic [2:0] s1_fk, s_fk;
  logic [4:0] s1_fi, s_fi;
  f32_t       s1_in_d [R*L], s_in_d [R*L], s1_fd [L], s_fd [L];
  ss_cfg_t    s_cfg;
  wire        ss_act_d = ss_req && ss_rq;
  always_ff @(posedge clk) begin
    if (rst) begin
      ss_rq <= 1'b0; ss_act <= 1'b0; s_init <= 1'b0; sen_q <= 1'b0; sen <= 1'b0;
      s1_fk <= SF_NONE; s_fk <= SF_NONE;
    end else begin
      ss_rq <= ss_req;
      ss_act <= ss_act_d;
      s_init <= ss_act_d && !ss_act;
      sen_q <= ss_act_d && ss_pe;
      sen <= sen_q;
      s1_fk <= ss_act ? ss_fk : SF_NONE;
      s_fk <= s1_fk;
    end
    s1_fi <= ss_fi; s1_fd <= ss_fd; s_fi <= s1_fi; s_fd <= s1_fd;
    s1_in_v <= ss_in_v; s1_in_d <= ss_in_d; s_in_v <= s1_in_v; s_in_d <= s1_in_d;
    s_cfg <= ss_cfg;
  end
  assign ss_gnt = ss_act;
  wire pe = sen;

  // dot A, per group and lane
  f32_t xd [R*L], xa [L], pacc [R*L];
  ss_meta_t xm, mt;
  for (genvar i = 0; i < R*L; i++) begin : g_a
    f32_t ra, rb, rc, fbd, fbq;
    always_ff @(posedge clk) if (pe) begin
      ra <= xd[i]; rb <= xa[i % L];
    end
    always_ff @(posedge clk)
      if (rst)     rc <= F_NZ;
      else if (pe) rc <= xm.first ? F_ZERO : fbq;
    otpu_fmadd #(.LM(LM), .LA(LA)) u_ma (.clk, .en(pe), .a(ra), .b(rb), .c(rc), .y(pacc[i]));
    otpu_delay #(.W(32), .N(DF - 1)) u_fb (.clk, .en(pe), .d(pacc[i]), .q(fbd));
    always_ff @(posedge clk) if (pe) fbq <= fbd;
  end
  otpu_delay #(.W($bits(ss_meta_t)), .N(TA)) u_mt (.clk, .en(pe), .d(xm), .q(mt));

  // the trees: dot A's final partials, or the tail's dot Q's in their windows (as otpu_vpu's)
  logic       q_cap, q_rl, vt_q, t_qc, t_qrl, t_yv, t_ov;
  logic [7:0] q_sub, t_qsub;
  f32_t       q_pd [R*L], t_qd [R*L];
  logic [R-1:0][31:0] root, root_o;
  logic [R-1:0] root_v;
  wire  a_cap = mt.v && mt.final_;
  wire  cap = a_cap || q_cap;
  wire  vt_rl = q_cap ? q_rl : mt.row_last;
  wire  [7:0] vt_sub = q_cap ? q_sub : mt.sub;
  for (genvar g = 0; g < R; g++) begin : g_t
    f32_t vt_in [L];
    for (genvar l = 0; l < L; l++) begin : g_vtin
      assign vt_in[l] = q_cap ? q_pd[g * L + l] : pacc[g * L + l];
    end
    otpu_vtree #(.LANES(L), .LA(LA)) u_vt (.clk, .rst(rst || s_init), .en(pe), .pacc(vt_in),
                                           .cap, .row_last(vt_rl), .sub(vt_sub), .root(root[g]),
                                           .root_o(root_o[g]), .root_v(root_v[g]));
  end
  always_ff @(posedge clk) if (pe) begin
    q_cap <= t_qc;
    q_rl <= t_qrl;
    q_sub <= t_qsub;
    q_pd <= t_qd;
  end
  // the roots come out in window order (every group's together): a FIFO of their kinds routes
  // them to kv (A) or back to the tail (Q)
  logic [7:0] kq;
  logic [2:0] kh, kt;
  always_ff @(posedge clk)
    if (rst || s_init) begin
      kh <= '0; kt <= '0;
    end else if (pe) begin
      if (cap && vt_rl) begin
        kq[kt] <= q_cap;
        kt <= kt + 1'b1;
      end
      if (root_v[0]) kh <= kh + 1'b1;
    end
  assign vt_q = kq[kh];
`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && pe && a_cap && q_cap) $fatal(1, "otpu_se_eng: A and Q windows meet on u_vt");
`endif

  f32_t z [L];
  logic [4:0] s_fil [L];                    // the tail's fill index per lane (one register here)
  for (genvar l = 0; l < L; l++) begin : g_z
    assign z[l] = '0;
    assign s_fil[l] = s_fi;
  end
  otpu_se_tail #(.LANES(L), .TA(TA), .QD(1), .R(R), .DSTEP2(DSTEP2)) u_tail (
    .clk, .rst, .init(s_init), .cfg(s_cfg), .fk(s_fk), .fi(s_fil), .fd(s_fd),
    .pe, .in_v(s_in_v), .in_d(s_in_d), .xd, .xa, .xm,
    .kv(root), .kv_v(root_v[0] && !vt_q), .y_v(t_yv), .y_d(ss_y_d), .o_v(t_ov), .o_d(ss_o_d),
    .qp_cap(t_qc), .qp_row_last(t_qrl), .qp_sub(t_qsub), .qp_d(t_qd),
    .qo_v(root_v[0] && vt_q), .qo_d(root_o),
    .cm_u('0), .cm_q('0), .cen({L{pe}}), .u_sel(1'b0), .u_a(z), .u_b(z), .u_c(z), .u_e(z), .u_y(),
    .q_sel(1'b0), .q_a(z), .q_b(z), .q_c(z), .q_y());
  assign ss_y_v = sen && t_yv;
  assign ss_o_v = sen && t_ov;
endmodule
