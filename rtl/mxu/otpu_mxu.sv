// MXU: streams N rows x KB blocks of int8 from DRAM (port B, one D-byte chunk per cycle,
// prefetched through a FIFO) and their fp32 block scales (port A), and reuses each chunk across
// up to MCOLS stationary rows held in ACT RAM (docs/isa.md, MM). Per chunk and column j:
//   t[j][k] = (i2f(sum_i act[j][i] * w[i]) * ws) * ascale[j]
//   acc[j]  = isum_4(t[j][0..KB-1])      interleaved partials p[k mod 4], then (p0+p2)+(p1+p3)
// After the last block of a row the M results are written to TMEM (optionally accumulated,
// optionally rescaled: y = old * alpha[j] + acc). With RMAX the running max of every written
// value per column is written after the last row (the max order is total: any order is exact).
//
// Pipelined for the FPGA clock:
//   pop | products | +4 | +4 | tree | i2f (2) | *ws (2) | *ascale (2) | partial loop (4) | combine
// The partial loop is exactly 4 pipeline advances long (one 4-stage adder), so block k meets
// the partial of block k-4 of the same row. The compute pipeline
// advances when a chunk is popped, or with a bubble whenever the next chunk would start a new
// row (or the command has no chunks left); it only freezes in the middle of a row.
// Finished rows go to a result FIFO; the drain writes up to LANES results per cycle, through a
// pipelined read-modify-write for ACC/ASCALE, and holds while the TMEM grant is withheld.
//
// The MXU holds two commands: the issuer streams the newer one's chunks while the consumer
// finishes (and drains) the older one. The issuer yields DRAM ports to the DMA and QST
// (a_gnt/b_gnt). On the FPGA the products map to DSP48 multipliers.
module otpu_mxu
  import otpu_pkg::*;
  import otpu_fp::*;
#(
  parameter int D     = 32,
  parameter int MCOLS = 8,
  parameter int DEPTH = 16,
  parameter int LANES = 8,       // TMEM banks (MCOLS > LANES drains a row in several cycles)
  parameter int IMPL  = 0,       // integer dot product: 0 adder tree, 1 DSP cascade chains
  parameter int CL    = 16,      // IMPL 1: products per cascade chain (D / CL chains)
  parameter int SID   = 0
) (
  input  logic                  clk,
  input  logic                  rst,
  input  logic                  start,       // accept: the issuer may stream it
  input  logic                  go,          // release: the consumer may use it (in order)
  input  cmd_t                  cmd,
  output logic                  rdy,
  output logic                  done,
  output logic                  computing,   // a chunk is consumed this cycle (profiling)
  // ACT RAM read
  output logic [15:0]           act_blk,
  output logic                  act_ren,     // the ACT RAM read register advances (with S0)
  input  logic [MCOLS*D*8-1:0]  act_data,
  input  logic [MCOLS*32-1:0]   act_scale,
  // DRAM port A (scales) and B (chunks)
  output logic                  a_req,
  output logic [31:0]           a_addr,
  input  logic                  a_gnt,
  input  logic                  a_rvalid,
  input  logic [31:0]           a_rdata,
  output logic                  b_req,
  output logic [31:0]           b_addr,
  input  logic                  b_gnt,
  input  logic                  b_rvalid,
  input  logic [D*8-1:0]        b_rdata,
  // TMEM (read port for ACC, write port)
  output logic [LANES-1:0]        t_ren,
  output logic [LANES-1:0][31:0]  t_raddr,
  input  logic [LANES-1:0][31:0]  t_rdata,
  output logic [LANES-1:0]        t_wen,
  output logic [LANES-1:0][31:0]  t_waddr,
  output logic [LANES-1:0][31:0]  t_wdata,
  input  logic                    t_gnt
);
  localparam int BW = $clog2(LANES);
  localparam int PW = $clog2(DEPTH);
  localparam int LM = 2, LA = 4;
  localparam int NPART = 4;                 // MM partials (isum_4)
  localparam int RF = 32;                   // result FIFO rows
  localparam int RFW = $clog2(RF);
  localparam int MW = $clog2(MCOLS) + 1;
  localparam int NL = (LANES < MCOLS) ? LANES : MCOLS;   // lanes the drain can fill
  initial if (D % 16 != 0) $fatal(1, "otpu_mxu: D must be a multiple of 16");

  // ================================================================== issuer
  logic        i_act, i_unit;
  logic [31:0] i_left, i_rs, i_srs;
  logic [15:0] i_KB, i_k;
  logic [31:0] row_addr, chunk_addr, srow_addr, scale_addr;
  logic [31:0] issued, popped;

  // ================================================================== command queue (2)
  logic [31:0] q_out [2], q_total [2];
  logic [15:0] q_KB [2], q_ors [2];
  logic [31:0] q_mxo [2];                     // RMAX output base: out + M * ors (no multiply later)
  logic [7:0]  q_M [2], q_ab [2];
  logic        q_unit [2], q_acc [2], q_rmax [2], q_asc [2], q_go [2];
  logic [31:0] q_asa [2];
  logic [31:0] q_jo [2][MCOLS];            // j * ors
  logic        q_h;
  logic [1:0]  q_n;

  wire [31:0] c_out = q_out[q_h];
  wire [31:0] c_total = q_total[q_h];
  wire [15:0] c_KB = q_KB[q_h];
  wire [7:0]  c_M = q_M[q_h], c_ab = q_ab[q_h];
  wire        c_unit = q_unit[q_h], c_acc = q_acc[q_h], c_rmax = q_rmax[q_h];
  wire        c_asc = q_asc[q_h];
  wire [31:0] c_asa = q_asa[q_h];
  wire        c_act = (q_n != 0) && q_go[q_h];
  logic [31:0] alpha [MCOLS];
  logic [1:0]  al_st;                       // ASCALE factors: 0 to load, 1 loading, 2 loaded
  logic [7:0]  al_i, mx_i;                  // next ASCALE factor to load / RMAX value to write

  // FIFOs of chunks and of their scales (the two DRAM ports return independently, in order;
  // an issued chunk's slot is reserved, so neither FIFO can overflow)
  logic [D*8-1:0] f_data  [DEPTH];
  logic [31:0]    f_scale [DEPTH];
  logic [PW-1:0]  f_head, f_tail, s_head, s_tail;
  logic [PW:0]    f_count, s_count;

  // ================================================================== consumer control
  logic [15:0] ck;
  logic [31:0] c_pop;
  logic [RFW:0] rows_live;                  // rows popped (first block) and not yet drained
  wire last_k   = (ck + 1 == c_KB);
  wire more     = c_act && (c_pop < c_total);
  wire pop      = more && (f_count != 0) && (c_unit || s_count != 0) && (ck != 0 || rows_live < RF);
  wire en_c     = pop || !(more && ck != 0);       // freeze only in the middle of a row
  wire want_iss = i_act && ((issued - popped) < DEPTH);
  wire go_iss   = want_iss && b_gnt && (i_unit || a_gnt);

  assign rdy = !i_act && (q_n < 2);
  assign computing = pop;
  assign b_req  = go_iss;
  assign b_addr = go_iss ? (chunk_addr >> 2) : '0;
  assign a_req  = go_iss && !i_unit;
  assign a_addr = a_req ? (scale_addr >> 2) : '0;
  assign act_blk = 16'(c_ab) + ck;
  assign act_ren = en_c;

  // ================================================================== compute pipeline
  typedef struct packed {
    logic       v;
    logic       last;     // last block of its row
    logic       first;    // block index < 4: the partial starts at +0
    logic       fin;      // block index >= KB - 4: the partial is final after this block
    logic [1:0] q;        // block index mod 4
  } cm_t;

  cm_t                 m0, m1, m2, m3, m4, m5, m6;
  logic [D*8-1:0]      w0;
  logic [MCOLS*D*8-1:0] a0;
  f32_t                ws0, ws1, ws2, ws3, ws4, ws5, ws6;
  i2f_mid_t            im [MCOLS];
  f32_t                as0 [MCOLS];
  logic signed [31:0]  s4 [MCOLS];
  f32_t                fi [MCOLS];
  // S0's ACT RAM block and scales are the ACT RAM's registered read
  assign a0 = act_data;
  // the scale FIFO's registered read is kept free of logic (so it maps into the block RAM's
  // output register); unit-scale chunks are substituted after it
  f32_t ws0r;
  logic cu0;
  assign ws0 = cu0 ? F_ONE : ws0r;
  always_comb for (int j = 0; j < MCOLS; j++) as0[j] = act_scale[j*32 +: 32];

  always_ff @(posedge clk) if (en_c) begin
    // S0: the popped chunk, its ACT RAM block and scales
    m0 <= '0;
    if (pop) begin
      m0.v <= 1'b1;
      m0.last <= last_k;
      m0.first <= (ck < 16'(NPART));
      m0.fin <= (32'(ck) + NPART >= 32'(c_KB));
      m0.q <= ck[1:0];
    end
    w0 <= f_data[f_head];
    ws0r <= f_scale[s_head];
    cu0 <= c_unit;
    // S5, S6: int -> fp32
    for (int j = 0; j < MCOLS; j++) im[j] <= i2f_s1(s4[j]);
    for (int j = 0; j < MCOLS; j++) fi[j] <= i2f_s2(im[j]);
    m5 <= m4; ws5 <= ws4;
    m6 <= m5;
  end
  // ws6 feeds the first fp multiplier's B operand: a reset flop, never an SRL tap
  always_ff @(posedge clk) if (rst) ws6 <= '0; else if (en_c) ws6 <= ws5;

  // dot-product latency S0 -> s4 (the tree: 4 register levels)
  localparam int NG = D / CL;
  localparam int TL = (NG <= 1) ? 0 : (NG <= 4) ? 1 : (NG <= 16) ? 2 : 3;
  localparam int LDOT = (IMPL == 0) ? 4 : CL + 1 + TL - (TL >= 3 ? 1 : 0);

  // ---- S1 .. S4: the exact integer dot products of the chunk with every column's ACT block;
  // s4 (with m4, ws4) is the chunk's result LDOT - 4 cycles after S0 + 4.
  if (IMPL == 0) begin : g_tree
    // products, then a 4 / 4 / 8 adder tree (one register level each)
    // Columns 2p and 2p+1 share the weight byte, so one multiplier (a DSP48 with its pre-adder)
    // makes both: pp = (a0*2^16 + a1) * w = (a0*w)*2^16 + a1*w (a 25-bit A: shift 17 overflows).
    // a1*w = pp[15:0] (signed), a0*w = pp[31:16] (signed) + pp[15] (the low field's borrow).
    // An odd last column keeps a plain product.
    // pp and pr are packed so they are registers the DSPs absorb (MREG), not memories that are
    // mapped to fabric flops after DSP packing; every field is read through $signed().
    logic [(MCOLS + 1) / 2 - 1:0][D-1:0][31:0] pp;
    logic [D-1:0][15:0]                         pr;
    logic signed [19:0] s2 [MCOLS][D/4];
    logic signed [23:0] s3 [MCOLS][D/16];
    always_ff @(posedge clk) if (en_c) begin
      for (int p = 0; p < MCOLS / 2; p++)
        for (int i = 0; i < D; i++) begin
          logic signed [24:0] pa;
          pa = $signed({a0[(2*p*D + i)*8 +: 8], 16'b0}) + 25'($signed(a0[((2*p+1)*D + i)*8 +: 8]));
          pp[p][i] <= 32'(int'(pa) * int'($signed(w0[i*8 +: 8])));
        end
      if (MCOLS % 2 == 1)
        for (int i = 0; i < D; i++)
          pr[i] <= 16'(int'($signed(a0[((MCOLS-1)*D + i)*8 +: 8])) * int'($signed(w0[i*8 +: 8])));
      for (int j = 0; j < MCOLS; j++) begin
        for (int g = 0; g < D / 4; g++) begin
          logic signed [19:0] t;
          t = '0;
          for (int k = 0; k < 4; k++)
            if (j % 2 == 1) t = t + 20'($signed(pp[j/2][4*g+k][15:0]));
            else if (j + 1 < MCOLS)
              t = t + 20'($signed(pp[j/2][4*g+k][31:16])) + 20'(pp[j/2][4*g+k][15]);
            else t = t + 20'($signed(pr[4*g+k]));
          s2[j][g] <= t;
        end
        for (int g = 0; g < D / 16; g++)
          s3[j][g] <= 24'(s2[j][4*g]) + 24'(s2[j][4*g+1]) + 24'(s2[j][4*g+2]) + 24'(s2[j][4*g+3]);
        begin
          logic signed [31:0] t;
          t = '0;
          for (int g = 0; g < D / 16; g++) t = t + 32'(s3[j][g]);
          s4[j] <= t;
        end
      end
      m1 <= m0; ws1 <= ws0;
      m2 <= m1; ws2 <= ws1;
      m3 <= m2; ws3 <= ws2;
      m4 <= m3; ws4 <= ws3;
    end
  end else begin : g_casc
    // Systolic accumulate chains (DSP48 A*B + PCIN cascades): the D positions form NG = D / CL
    // chains of CL; position i = g*CL + k enters stage k of chain g k cycles after S0 (operand
    // skew in shift registers), so a new chunk enters every cycle. Stage k: a registered product
    // (the DSP's M register) added to stage k-1's running sum (its P register). The NG chain
    // sums meet in a small adder tree (TL register levels of up to 4 inputs). Exact integers.
    initial if (D % CL != 0 || NG > 64) $fatal(1, "otpu_mxu: CL must divide D, D/CL <= 64");
    // operand skew: position k of every chain is delayed k cycles (weights shared by columns)
    logic [7:0] ws_k [D];                               // skewed weight bytes
    logic [7:0] as_k [MCOLS][D];                        // skewed activation bytes
    for (genvar i = 0; i < D; i++) begin : g_wsk
      otpu_delay #(.W(8), .N(i % CL)) u_w (.clk, .en(en_c), .d(w0[i*8 +: 8]), .q(ws_k[i]));
      for (genvar j = 0; j < MCOLS; j++) begin : g_ask
        otpu_delay #(.W(8), .N(i % CL)) u_a (.clk, .en(en_c), .d(a0[(j*D + i)*8 +: 8]),
                                            .q(as_k[j][i]));
      end
    end
    // the chains
    logic signed [15:0] mreg [MCOLS][D];                // stage products (M registers)
    logic signed [23:0] preg [MCOLS][D];                // running sums (P registers)
    always_ff @(posedge clk) if (en_c) begin
      for (int j = 0; j < MCOLS; j++)
        for (int i = 0; i < D; i++) begin
          mreg[j][i] <= 16'(int'($signed(as_k[j][i])) * int'($signed(ws_k[i])));
          preg[j][i] <= ((i % CL == 0) ? 24'sd0 : preg[j][i - 1]) + 24'(mreg[j][i]);
        end
    end
    // chain sums (stage CL-1 of each chain) -> tree
    logic signed [31:0] t1 [MCOLS][(NG + 3) / 4];
    logic signed [31:0] t2 [MCOLS][(NG + 15) / 16];
    always_ff @(posedge clk) if (en_c) begin
      for (int j = 0; j < MCOLS; j++) begin
        for (int u = 0; u < (NG + 3) / 4; u++) begin
          logic signed [31:0] t;
          t = '0;
          for (int v = 0; v < 4; v++)
            if (4 * u + v < NG) t = t + 32'(preg[j][(4 * u + v) * CL + CL - 1]);
          t1[j][u] <= t;
        end
        for (int u = 0; u < (NG + 15) / 16; u++) begin
          logic signed [31:0] t;
          t = '0;
          for (int v = 0; v < 4; v++)
            if (4 * u + v < (NG + 3) / 4) t = t + t1[j][4 * u + v];
          t2[j][u] <= t;
        end
      end
    end
    always_comb
      for (int j = 0; j < MCOLS; j++) begin
        logic signed [31:0] t;
        t = '0;
        case (TL)
          0: t = 32'(preg[j][CL - 1]);
          1: t = t1[j][0];
          2: t = t2[j][0];
          default: for (int u = 0; u < (NG + 15) / 16; u++) t = t + t2[j][u];
        endcase
        s4[j] = t;
      end
    // the meta and the weight scale travel alongside (S0 -> S4 position)
    otpu_delay #(.W($bits(cm_t)), .N(LDOT)) u_m4 (.clk, .en(en_c), .d(m0), .q(m4));
    otpu_delay #(.W(32), .N(LDOT)) u_ws4 (.clk, .en(en_c), .d(ws0), .q(ws4));
  end

  // the ACT scale travels with the chunk to the second multiplier (S0 + LDOT + 2 + LM)
  f32_t t1 [MCOLS], t2 [MCOLS], pacc [MCOLS];
  cm_t  mt_p, mt, ma;                        // meta at the adder inputs / outputs
  // the delay lines into the fp operands end in reset flops (a reset can't go into an SRL, so the
  // last stage is an FDRE with a fast clock-to-out); same total length and enable
  otpu_delay #(.W($bits(cm_t)), .N(2 * LM - 1)) u_mt (.clk, .en(en_c), .d(m6), .q(mt_p));
  always_ff @(posedge clk) if (rst) mt <= '0; else if (en_c) mt <= mt_p;
  otpu_delay #(.W($bits(cm_t)), .N(LA)) u_ma (.clk, .en(en_c), .d(mt), .q(ma));
  for (genvar j = 0; j < MCOLS; j++) begin : g_col
    f32_t fb, prev, as_p, asq;
    otpu_delay #(.W(32), .N(LDOT + 2 + LM - 1)) u_as (.clk, .en(en_c), .d(as0[j]), .q(as_p));
    always_ff @(posedge clk) if (rst) asq <= '0; else if (en_c) asq <= as_p;
    otpu_fmul #(.LAT(LM)) u_m1 (.clk, .en(en_c), .a(fi[j]), .b(ws6), .y(t1[j]));
    otpu_fmul #(.LAT(LM)) u_m2 (.clk, .en(en_c), .a(t1[j]), .b(asq), .y(t2[j]));
    // partial loop: pacc(block k) = pacc(block k - 4) + t(k), exactly NPART advances
    otpu_delay #(.W(32), .N(NPART - LA)) u_fb (.clk, .en(en_c), .d(pacc[j]), .q(fb));
    assign prev = mt.first ? F_ZERO : fb;
    otpu_fadd #(.LAT(LA)) u_acc (.clk, .en(en_c), .a(prev), .b(t2[j]), .y(pacc[j]));
  end

  // collect the final partials of a row; at its last block combine (p0+p2)+(p1+p3).
  // The last block latches the combine operands into pset (its own slot <- pacc, slots the row
  // never filled <- +0), so the adders read flops; pset's data input is always pacc.
  f32_t       pset [MCOLS][NPART];
  logic [NPART-1:0] pmask;
  always_ff @(posedge clk) begin
    if (rst) pmask <= '0;
    else if (en_c && ma.v && ma.fin) begin
      if (ma.last) pmask <= '0;
      else pmask[ma.q] <= 1'b1;
      for (int q = 0; q < NPART; q++)
        for (int j = 0; j < MCOLS; j++)
          if (ma.q == 2'(q)) pset[j][q] <= pacc[j];
          else if (ma.last && !pmask[q]) pset[j][q] <= F_ZERO;
    end
  end
  wire launch = ma.v && ma.last;
  f32_t c01 [MCOLS], c23 [MCOLS], rowv [MCOLS];
  logic lv1, lv2;
  for (genvar j = 0; j < MCOLS; j++) begin : g_comb
    otpu_fadd #(.LAT(LA)) u_c01 (.clk, .en(en_c), .a(pset[j][0]), .b(pset[j][2]), .y(c01[j]));
    otpu_fadd #(.LAT(LA)) u_c23 (.clk, .en(en_c), .a(pset[j][1]), .b(pset[j][3]), .y(c23[j]));
    otpu_fadd #(.LAT(LA)) u_c (.clk, .en(en_c), .a(c01[j]), .b(c23[j]), .y(rowv[j]));
  end
  // one advance to latch pset, then the two adder levels
  otpu_delay #(.W(1), .N(2 * LA + 1)) u_lv (.clk, .en(en_c), .d(launch), .q(lv2));

  // ================================================================== result FIFO
  f32_t        rf_v [RF][MCOLS];
  logic [RFW-1:0] rf_h, rf_t;
  logic [RFW:0]   rf_n;
  wire         rf_push = en_c && lv2;

  // ================================================================== drain
  // lanes this cycle: results dj .. dj+ncnt-1 of the head row, stopping at a bank conflict
  logic [31:0] d_row;                        // out + n of the head row
  logic [7:0]  dj;
  logic [7:0]  ncnt;
  logic [LANES-1:0][31:0] daddr_l, dval_l;
  logic [LANES-1:0][7:0]  dcol_l;
  wire  drain_go = (rf_n != 0) && (!c_asc || al_st == 2'd2);
  always_comb begin
    logic [LANES-1:0] used;
    logic stop;
    logic [31:0] ad;
    used = '0; stop = 1'b0; ncnt = '0; daddr_l = '0; dval_l = '0; dcol_l = '0; ad = '0;
    for (int k = 0; k < NL; k++) begin
      if (k < MCOLS && !stop && 32'(dj) + 32'(k) < 32'(c_M)) begin
        ad = d_row + q_jo[q_h][MW'(32'(dj) + 32'(k))];
        if (!used[ad[BW-1:0]]) begin
          used[ad[BW-1:0]] = 1'b1;
          daddr_l[k] = ad;
          dval_l[k] = rf_v[rf_h][MW'(32'(dj) + 32'(k))];
          dcol_l[k] = dj + 8'(k);
          ncnt = ncnt + 1;
        end else stop = 1'b1;
      end else stop = 1'b1;
    end
  end
  wire drain_row_done = drain_go && (dj + ncnt == c_M);

  // read-modify-write pipeline for ACC: read now, data next cycle, (old*alpha)+new, write
  typedef struct packed {
    logic                   v;
    logic [NL-1:0]          m;
    logic [NL-1:0][31:0]    ad, nv;
    logic [NL-1:0][7:0]     col;
  } rmw_t;
  rmw_t r0, rw;                               // r0: data arriving now; rw: at the write stage
  rmw_t rx;                                   // RMAX (no ACC): drained lanes, compared next cycle
  f32_t ry [NL];
  for (genvar k = 0; k < NL; k++) begin : g_rmw
    f32_t al;
    assign al = c_asc ? alpha[r0.col[k][MW-2:0]] : F_ONE;
    otpu_fmadd #(.LM(LM), .LA(LA)) u_y (.clk, .en(t_gnt), .a(t_rdata[k]), .b(al), .c(r0.nv[k]),
                                        .y(ry[k]));
  end
  otpu_delay #(.W($bits(rmw_t)), .N(LM + LA)) u_rw (.clk, .en(t_gnt), .d(r0), .q(rw));
  logic [3:0] rmw_n;                          // rows' lanes in flight (any nonzero = busy)

  // RMAX
  f32_t mx [MCOLS];
  logic [MCOLS-1:0] mx_have;
  logic mx_done;

  wire c_drained = c_act && (c_pop == c_total) && (rows_live == 0) && (rmw_n == 0) && !r0.v && !rx.v;
  wire mx_go     = c_drained && c_rmax && !mx_done && (c_total != 0);
  wire c_fin     = c_drained && (!c_rmax || mx_done || c_total == 0);
  wire al_go     = c_act && c_asc && al_st == 2'd0;

  always_comb begin
    t_ren = '0; t_raddr = '0; t_wen = '0; t_waddr = '0; t_wdata = '0;
    if (drain_go) begin
      for (int k = 0; k < LANES; k++) begin
        if (32'(k) < 32'(ncnt)) begin
          if (c_acc) begin
            t_ren[k] = 1'b1;
            t_raddr[k] = daddr_l[k];
          end else begin
            t_wen[k] = 1'b1;
            t_waddr[k] = daddr_l[k];
            t_wdata[k] = dval_l[k];
          end
        end
      end
    end
    if (rw.v) begin
      for (int k = 0; k < NL; k++) begin
        if (rw.m[k]) begin
          t_wen[k] = 1'b1;
          t_waddr[k] = rw.ad[k];
          t_wdata[k] = ry[k];
        end
      end
    end
    // ASCALE factors and RMAX values move LANES words per cycle (consecutive words: distinct
    // banks)
    if (al_go) begin
      for (int k = 0; k < LANES; k++) begin
        if (32'(al_i) + 32'(k) < 32'(c_M)) begin
          t_ren[k] = 1'b1;
          t_raddr[k] = c_asa + 32'(al_i) + 32'(k);
        end
      end
    end
    if (mx_go) begin
      for (int k = 0; k < LANES; k++) begin
        if (32'(mx_i) + 32'(k) < 32'(c_M)) begin
          t_wen[k] = 1'b1;
          t_waddr[k] = q_mxo[q_h] + 32'(mx_i) + 32'(k);
          t_wdata[k] = mx[MW'(32'(mx_i) + 32'(k))];
        end
      end
    end
  end

  // ---- statistics for the profiler (per completed command)
  logic [31:0] cyc, st_starve, st_bp, st_frz, st_deny;

  always_ff @(posedge clk) begin
    done <= 1'b0;
    if (rst) begin
      i_act <= 1'b0;
      q_h <= 1'b0; q_n <= '0;
      issued <= '0; popped <= '0;
      f_head <= '0; f_tail <= '0; f_count <= '0;
      s_head <= '0; s_tail <= '0; s_count <= '0;
      ck <= '0; c_pop <= '0; rows_live <= '0;
      rf_h <= '0; rf_t <= '0; rf_n <= '0;
      dj <= '0; mx_done <= 1'b0; mx_have <= '0;
      al_st <= 2'd0; al_i <= '0; mx_i <= '0;
      r0 <= '0; rx <= '0; rmw_n <= '0;
      cyc <= '0; st_starve <= '0; st_bp <= '0; st_frz <= '0; st_deny <= '0;
    end else begin
      logic [1:0] qn;
      logic [RFW:0] rn;
      logic [RFW:0] rl;
      cyc <= cyc + 1;
      qn = q_n;
      rn = rf_n;
      rl = rows_live;
      // ---- accept a command
      if (start) begin
        logic qi;
        qi = q_h ^ (q_n != 0);
        q_out[qi]   <= cmd.w3;
        q_total[qi] <= 32'(cmd.w4[15:0]) * 32'(cmd.w4[31:16]);
        q_KB[qi]    <= cmd.w4[31:16];
        q_ors[qi]   <= cmd.w6[15:0];
        q_mxo[qi]   <= cmd.w3 + 32'(cmd.w6[23:16]) * 32'(cmd.w6[15:0]);
        q_M[qi]     <= cmd.w6[23:16];
        q_ab[qi]    <= cmd.w6[31:24];
        q_unit[qi]  <= cmd.flags[0];
        q_acc[qi]   <= cmd.flags[1];
        q_rmax[qi]  <= cmd.flags[2];
        q_asc[qi]   <= cmd.flags[3];
        q_go[qi]    <= 1'b0;
        q_asa[qi]   <= cmd.w2;
        for (int j = 0; j < MCOLS; j++) q_jo[qi][j] <= 32'(j) * 32'(cmd.w6[15:0]);
        qn = qn + 1;
        if (cmd.w4[15:0] != 0 && cmd.w4[31:16] != 0) begin
          i_act  <= 1'b1;
          i_left <= 32'(cmd.w4[15:0]) * 32'(cmd.w4[31:16]);
          i_KB   <= cmd.w4[31:16];
          i_k    <= '0;
          i_rs   <= cmd.w5;
          i_srs  <= cmd.w7;
          i_unit <= cmd.flags[0];
          row_addr <= cmd.w1; chunk_addr <= cmd.w1;
          srow_addr <= cmd.w2; scale_addr <= cmd.w2;
        end
      end
      // ---- release (in order): the head if not yet released, else the second entry
      if (go) begin
        if (q_n != 0 && !q_go[q_h]) q_go[q_h] <= 1'b1;
        else q_go[~q_h] <= 1'b1;
      end
      // ---- issue one chunk request
      if (go_iss) begin
        issued <= issued + 1;
        i_left <= i_left - 1;
        if (i_left == 1) i_act <= 1'b0;
        if (i_k + 1 == i_KB) begin
          i_k <= '0;
          row_addr <= row_addr + i_rs;
          chunk_addr <= row_addr + i_rs;
          srow_addr <= srow_addr + i_srs;
          scale_addr <= srow_addr + i_srs;
        end else begin
          i_k <= i_k + 1;
          chunk_addr <= chunk_addr + D;
          scale_addr <= scale_addr + 4;
        end
      end
      if (want_iss && !go_iss) st_deny <= st_deny + 1;
      // ---- FIFO pushes
      if (b_rvalid) begin
        f_data[f_tail] <= b_rdata;
        f_tail <= f_tail + 1;
      end
      if (a_rvalid) begin
        f_scale[s_tail] <= a_rdata;
        s_tail <= s_tail + 1;
      end
      f_count <= f_count + (b_rvalid ? 1 : 0) - (pop ? 1 : 0);
      s_count <= s_count + (a_rvalid ? 1 : 0) - ((pop && !c_unit) ? 1 : 0);
      // ---- pop one chunk
      if (more && f_count == 0) st_starve <= st_starve + 1;
      if (more && f_count != 0 && !pop) st_bp <= st_bp + 1;
      if (pop) begin
        f_head <= f_head + 1;
        if (!c_unit) s_head <= s_head + 1;
        popped <= popped + 1;
        c_pop <= c_pop + 1;
        if (ck == 0) rl = rl + 1;
        ck <= last_k ? '0 : ck + 1;
      end
      // ---- a finished row enters the result FIFO
      if (rf_push) begin
        for (int j = 0; j < MCOLS; j++) rf_v[rf_t][j] <= rowv[j];
        rf_t <= rf_t + 1;
        rn = rn + 1;
      end
      // ---- drain (holds while the TMEM grant is withheld)
      if (t_gnt) begin
        r0 <= '0;
        rx <= '0;
        if (drain_go) begin
          if (c_acc) begin
            r0.v <= 1'b1;
            for (int k = 0; k < NL; k++) r0.m[k] <= (32'(k) < 32'(ncnt));
            r0.ad <= daddr_l[NL-1:0];
            r0.nv <= dval_l[NL-1:0];
            r0.col <= dcol_l[NL-1:0];
          end
          if (c_rmax && !c_acc) begin
            rx.v <= 1'b1;
            for (int k = 0; k < NL; k++) rx.m[k] <= (32'(k) < 32'(ncnt));
            rx.nv <= dval_l[NL-1:0];
            rx.col <= dcol_l[NL-1:0];
          end
          if (drain_row_done) begin
            dj <= '0;
            d_row <= d_row + 1;
            rf_h <= rf_h + 1;
            rn = rn - 1;
            rl = rl - 1;
          end else begin
            dj <= dj + ncnt;
          end
        end
        // registered so the lane selection and the max compare are in different cycles
        if (rx.v) begin
          for (int k = 0; k < NL; k++) begin
            if (rx.m[k]) begin
              logic [MW-2:0] j;
              j = rx.col[k][MW-2:0];
              if (!mx_have[j] || fp_gt(rx.nv[k], mx[j])) mx[j] <= ftz(rx.nv[k]);
              mx_have[j] <= 1'b1;
            end
          end
        end
        if (rw.v && c_rmax) begin
          for (int k = 0; k < NL; k++) begin
            if (rw.m[k]) begin
              logic [MW-2:0] j;
              j = rw.col[k][MW-2:0];
              if (!mx_have[j] || fp_gt(ry[k], mx[j])) mx[j] <= ftz(ry[k]);
              mx_have[j] <= 1'b1;
            end
          end
        end
        rmw_n <= rmw_n + ((drain_go && c_acc) ? 4'd1 : 4'd0) - (rw.v ? 4'd1 : 4'd0);
        if (mx_go) begin
          if (32'(mx_i) + LANES >= 32'(c_M)) mx_done <= 1'b1;
          else mx_i <= mx_i + 8'(LANES);
        end
        if (al_go) al_st <= 2'd1;
        if (al_st == 2'd1) begin
          for (int k = 0; k < LANES; k++)
            if (32'(al_i) + 32'(k) < MCOLS) alpha[MW'(32'(al_i) + 32'(k))] <= t_rdata[k];
          if (32'(al_i) + LANES >= 32'(c_M)) al_st <= 2'd2;
          else begin
            al_i <= al_i + 8'(LANES);
            al_st <= 2'd0;
          end
        end
      end else if (drain_go || rw.v || mx_go || al_go) begin
        st_frz <= st_frz + 1;
      end
      rf_n <= rn;
      rows_live <= rl;
      // ---- the consumer's command is complete
      if (c_fin && !pop) begin
        done <= 1'b1;
        q_h <= ~q_h;
        qn = qn - 1;
        ck <= '0; c_pop <= '0;
        dj <= '0;
        d_row <= (start && q_n == 2'd1) ? cmd.w3 : q_out[~q_h];
        mx_done <= 1'b0; mx_have <= '0;
        al_st <= 2'd0; al_i <= '0; mx_i <= '0;
`ifndef SYNTHESIS
        if (trace) $display("T%0d U c=%0d u=1 starve=%0d bp=%0d frz=%0d deny=%0d", SID, cyc,
                            st_starve, st_bp, st_frz, st_deny);
`endif
        st_starve <= '0; st_bp <= '0; st_frz <= '0; st_deny <= '0;
      end
      // the head's output base (set when a command becomes head)
      if (start && q_n == 0) d_row <= cmd.w3;
      q_n <= qn;
    end
  end

`ifndef SYNTHESIS
  bit trace;
  initial trace = $test$plusargs("trace");
`endif
endmodule
