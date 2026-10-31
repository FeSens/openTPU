// Quantizer for QACT (TMEM rows -> ACT RAM) and QST (TMEM rows -> DRAM bytes), docs/isa.md
// "Quantization": per row (x after the optional RSCALE / CSCALE factors) amax = max|x|, the row's
// S = amax/127 and inv = 127*recip(amax); per group g (QACT: the sub-blocks of D/4 elements, QST:
// the D-blocks) its own max a_g and offset d_g = min(31, e(amax) - e(a_g) - (f(a_g) > f(amax)))
// (0 for a_g = 0 and in ROW mode); q = q8(x * (inv * 2^d_g)).
//
// Each row is read twice, both passes LANES elements a cycle (QST's pass 1: one byte, or a word
// of four, a cycle):
//   pass 0: read | prescale: (x*r)*c (2 multipliers) | chunk max (two stages, also per group
//           where a chunk holds several QACT groups) | fold: the row's max and the group maxes
//           (a_g, kept in four RAMs: group g in RAM g%4, word g/4) | the scale unit (recip: 2
//           Newton steps and the correction on multiply-add slots, then *127 and *1/127)
//   QST:    the row's scale row through DRAM port A: ROW: the word S; else S, then a word of four
//           d bytes a cycle (the four groups of one group word), the pad bytes 0
//   pass 1: read (the group word of the chunk with it) | d (4) | per lane inv * 2^d (an exponent
//           add) | prescale | x * (inv * 2^d), q8 | write (QACT: the chunk's bytes, and at a
//           block's first chunk its record {da3, da2, da1, da0, S}; QST: the bytes)
// A row's pass 1 starts once its scale is out of the scale unit (and QST's scale row written);
// the next row's pass 0 follows its pass 1 at once.
//
// QACT rows of up to XD x LANES elements go through the row buffer instead (XD > 0): pass 0
// also pushes the prescaled chunks into a FIFO (block RAM, XD chunks) and the reader goes on to
// the next row's pass 0 at once; pass 1 pops the row's chunks from the FIFO once its scale is
// out (a TMEM read and the prescale fewer), while the following rows' pass 0 run. Up to NB = 4
// rows are between the passes: the group maxes and the rows' scales are kept per row in NB banks
// (row r in bank r % 4), and pass 0 of row r waits for pass 1 of row r - 4 to have read them; the
// FIFO's free entries are counted at the reads (credits). A row's chunks are popped in order, so
// the result is the two-pass one bit for bit.
//
// The TMEM grant (gnt, the slice's arbiter: withheld while DRAM port A cannot take QST's write,
// while RSCALE's guest read waits for the DMA's copy, for a replay) holds only the reader: the
// position, the request registers, the chunk whose reads are asked (m0a) and the one whose data
// is due (m0). Everything behind it (the prescale, the maxes, the scale unit, the offsets, the
// quantizing multipliers, the row buffer's pops, the writer: thousands of registers, every DSP)
// runs without an enable, its chunks carrying valid bits; a chunk enters it on the grant that
// takes its data (the first after its read's, as when the whole unit froze: a replayed read's
// data comes a cycle late, under a withheld grant), so every result is the frozen design's. The
// reader keeps what it waits for from there (a row's scale, sq_p). QACT's writes need no grant
// (the ACT RAM is the quantizer's alone); QST's go through a queue whose head is port A's
// request (a_want from flip-flops), taken under the grant, and the reader issues a QST pass 1
// read only with a free entry (credits). (250 MHz, FK35, the 200 MHz build re-timed at 4 ns:
// the grant was every register's enable, 7285 loads behind a BUFGCE: u_quant ackw / u_mem qw_n
// -> a_want -> the grant -> invl, g_inv, u_sc, g_pre, g_q, ..., 3-4 levels, 92% route, -0.70 to
// -0.81 ns, thousands of endpoints.)

// N-cycle delay line (N >= 2) like otpu_delay, but its last stage has a synchronous reset: a
// flip-flop with a reset is not packed into the shift-register LUTs in front of it, so the
// multiplier operands it feeds leave a real flip-flop and not an SRL's slow clock-to-out.
module otpu_qdly #(parameter int W = 32, parameter int N = 2) (
  input  logic         clk,
  input  logic         rst,
  input  logic         en,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  initial if (N < 2) $fatal(1, "otpu_qdly: N must be at least 2");
  logic [W-1:0] r [N - 1];
  (* shreg_extract = "no", keep *) logic [W-1:0] qr;
  always_ff @(posedge clk) if (en) begin
    r[0] <= d;
    for (int k = 1; k < N - 1; k++) r[k] <= r[k-1];
  end
  always_ff @(posedge clk)
    if (rst) qr <= '0;
    else if (en) qr <= r[N-2];
  assign q = qr;
endmodule

// y = a * b as otpu_fp's fp_mul (flush to zero, round to nearest even) in three stages, for the
// prescale's first multiplier, whose result is the second one's DSP operand: the significands'
// product (the DSP's M/P) with the sign, the exponent sum and the special result | the
// rounding's operands: the significand by p[47], it + 1, the round-up bit, e0 = e + p[47] and its
// cases | the result. In otpu_fmul's two stages the whole rounding (8 levels, 3 a carry chain) ran
// from one DSP's P register into the next one's B input register (250 MHz, FK35: g_pre[].u_r ->
// u_c2, -0.54 to -0.79 ns, 76% route); here 4 levels behind the P register, 2-3 in front of the
// next DSP. Written from the operation, not from otpu_fp's stage internals (whose owner may
// change them); the simulation checks every result against fp_mul.
module otpu_qfmul3
  import otpu_fp::*;
(
  input  logic clk,
  input  f32_t a,
  input  f32_t b,
  output f32_t y
);
  // stage 1: an exponent field of 0 is a flushed zero, 255 an infinity or a NaN
  logic [47:0]        p;
  logic               s, sp;
  f32_t               sv;
  logic signed [10:0] e;
  always_ff @(posedge clk) begin
    logic za, zb, ia, ib;
    za = (a[30:23] == 8'd0);
    zb = (b[30:23] == 8'd0);
    ia = (a[30:23] == 8'hFF);
    ib = (b[30:23] == 8'hFF);
    p  <= {1'b1, a[22:0]} * {1'b1, b[22:0]};
    s  <= a[31] ^ b[31];
    e  <= 11'(a[30:23]) + 11'(b[30:23]) - 11'sd127;
    sp <= ia || ib || za || zb;
    sv <= (ia || ib) ? (((ia && a[22:0] != 0) || (ib && b[22:0] != 0) || za || zb) ? F_NAN
                        : {a[31] ^ b[31], 8'hFF, 23'd0})
                     : {a[31] ^ b[31], 31'd0};
  end
  // stage 2: the significand mm (24 bits from p[47] or p[46]), its guard g and sticky st, the
  // round-up bit, mm + 1 (its low 23 bits) and the exponent before rounding, e0 = e + p[47]
  logic               sp2, s2, rnd, am, ovf, e254, neg, ez;
  f32_t               sv2;
  logic [22:0]        mm, mi;
  logic [7:0]         e8, e8i;
  always_ff @(posedge clk) begin
    logic [23:0]        t;
    logic               g, st;
    logic signed [10:0] e0;
    if (p[47]) begin
      t = p[47:24]; g = p[23]; st = |p[22:0];
    end else begin
      t = p[46:23]; g = p[22]; st = |p[21:0];
    end
    e0 = e + 11'(p[47]);
    sp2  <= sp;
    sv2  <= sv;
    s2   <= s;
    rnd  <= g && (st || t[0]);
    am   <= &t;
    mm   <= t[22:0];
    mi   <= t[22:0] + 23'd1;
    e8   <= e0[7:0];
    e8i  <= e0[7:0] + 8'd1;
    ovf  <= (e0 >= 11'sd255);
    e254 <= (e0 == 11'sd254);
    neg  <= (e0 < 11'sd0);
    ez   <= (e0 == 11'sd0);
  end
  // stage 3: rounding up carries out of the 24 bits only when mm is all ones (c): the
  // significand's low bits are then mm + 1's (0) and the exponent field e0 + 1
  always_ff @(posedge clk) begin
    logic c;
    c = rnd && am;
    if (sp2) y <= sv2;
    else if (ovf || (e254 && c)) y <= {s2, 8'hFF, 23'd0};
    else if (neg) y <= {s2, 31'd0};
    else if (ez) y <= am ? {s2, 8'd1, 23'd0} : {s2, 31'd0};
    else y <= {s2, c ? e8i : e8, rnd ? mi : mm};
  end
`ifndef SYNTHESIS
  f32_t w [3];
  int unsigned nw;                            // (registers start arbitrary)
  initial nw = 0;
  always @(posedge clk) begin
    if (nw >= 3 && y != w[2])
      $fatal(1, "otpu_qfmul3: %h is not fp_mul's %h", y, w[2]);
    w[0] <= fp_mul(a, b);
    w[1] <= w[0];
    w[2] <= w[1];
    if (nw < 3) nw <= nw + 1;
  end
`endif
endmodule

module otpu_qscale
  import otpu_fp::*;
#(parameter int LM = 2, parameter int LA = 4, parameter int TW = 2) (
  input  logic          clk,
  input  logic          rst,
  input  logic          en,
  input  logic          iv,
  input  logic [TW-1:0] itag,
  input  f32_t          amax,        // >= 0 (an absolute value)
  output logic          ov,
  output logic [TW-1:0] otag,
  output f32_t          sc,
  output f32_t          inv
);
  localparam int SL = LM + LA;
  localparam int P = SL + LM;               // one Newton step: u_t, then u_y
  localparam int LAT = 2 * P + 2 * SL + LM; // two Newton steps, the correction (u_e, u_y3), u_inv
  // input register (the amax selection in front of the unit is its own pipeline stage)
  f32_t          amax_q;
  logic          iv_q;
  logic [TW-1:0] itag_q;
  always_ff @(posedge clk) if (en) begin
    amax_q <= amax; iv_q <= iv; itag_q <= itag;
  end
  // seed stage: |amax|, its flags, -a (a = amax's significand, in [1, 2)), the reciprocal seed
  // and the result's exponent adjustment (otpu_fp's rcp_ek), registered (amax is an otpu_fmul
  // result or +0, so it is already flushed: no ftz)
  f32_t ax, y0, kx;
  logic zero, big;
  logic [8:0] ek;
  always_ff @(posedge clk) if (en) begin
    f32_t a;
    a = {1'b0, amax_q[30:0]};
    ax <= a;
    kx <= {9'h17F, a[22:0]};
    zero <= (a == 0);
    big <= (a >= 32'h7E80_0000);
    ek <= rcp_ek(a);
    y0 <= RECIP_MAGIC - {9'h07F, a[22:0]};
  end
  // recip(a): y = y * (2 - a*y), twice, from the magic seed, then e = 1 - a*y, y = y*e + y; the
  // delay lines into the multiplier operands end in a flip-flop with a reset (otpu_qdly). The
  // ISA's y*t + (-0) is y*t for every product (no fp_mul result is changed by adding -0), so u_y
  // is a plain multiply, not padded: its output register feeds the next multipliers directly (no
  // SRL in front of their DSPs), and k1 is delayed by the step's SL + LM to meet it.
  f32_t y [4], t [2], k1 [3], yd [3], e;
  assign y[0] = y0;
  assign k1[0] = kx;
  for (genvar i = 0; i < 2; i++) begin : g_it
    otpu_fmadd #(.LM(LM), .LA(LA)) u_t (.clk, .en, .a(k1[i]), .b(y[i]), .c(F_TWO), .y(t[i]));
    otpu_qdly #(.W(32), .N(SL)) u_yd (.clk, .rst, .en, .d(y[i]), .q(yd[i]));
    otpu_qdly #(.W(32), .N(P)) u_k (.clk, .rst, .en, .d(k1[i]), .q(k1[i + 1]));
    otpu_fmul #(.LAT(LM)) u_y (.clk, .en, .a(yd[i]), .b(t[i]), .y(y[i + 1]));
  end
  otpu_fmadd #(.LM(LM), .LA(LA)) u_e (.clk, .en, .a(k1[2]), .b(y[2]), .c(F_ONE), .y(e));
  otpu_qdly #(.W(32), .N(SL)) u_yd2 (.clk, .rst, .en, .d(y[2]), .q(yd[2]));
  otpu_fmadd #(.LM(LM), .LA(LA)) u_y3 (.clk, .en, .a(yd[2]), .b(e), .c(yd[2]), .y(y[3]));
  // inv = 127 * (y * 2^ek) (y in [0.5, 1]) is (127 * y) * 2^ek bit for bit: 127 * y is in
  // [63.5, 127] and the product is normal, so its rounding does not change with the scaling,
  // except that it overflows to +inf where the field passes 254. big: 127 * (+0) = +0, forced
  // on the product (bd delayed beside zd) instead of on the operand.
  f32_t invm, scm, scd;
  logic zd, bd, zd2, bd2;
  logic [8:0] ekd, ek2, fe;
  otpu_qdly #(.W(11), .N(LAT - LM)) u_f (.clk, .rst, .en, .d({zero, big, ek}), .q({zd, bd, ekd}));
  otpu_fmul #(.LAT(LM)) u_inv (.clk, .en, .a(F_127), .b(y[3]), .y(invm));
  otpu_fmul #(.LAT(LM)) u_sc (.clk, .en, .a(ax), .b(F_INV127), .y(scm));
  otpu_delay #(.W(32), .N(LAT - LM)) u_scd (.clk, .en, .d(scm), .q(scd));
  otpu_delay #(.W(11), .N(LM)) u_z2 (.clk, .en, .d({zd, bd, ekd}), .q({zd2, bd2, ek2}));
  otpu_delay #(.W(1 + TW), .N(LAT + 1)) u_v (.clk, .en, .d({iv_q, itag_q}), .q({ov, otag}));
  assign fe = {1'b0, invm[30:23]} + ek2;     // 7..259
  assign inv = (zd2 || bd2) ? F_ZERO : (fe >= 9'd255) ? F_INF : {1'b0, fe[7:0], invm[22:0]};
  assign sc = zd2 ? F_ZERO : scd;
endmodule

module otpu_quant
  import otpu_pkg::*;
  import otpu_fp::*;
#(
  parameter int D     = 32,
  parameter int LANES = 8,
  parameter int TLANES = LANES,             // TMEM's banks (t_rben; at least LANES)
  parameter int GR    = 1024,               // group maxes of a row (QACT 4*KB, QST KB; not ROW)
  parameter int XD    = 1024,               // row buffer chunks (QACT rows <= XD*LANES); 0: none
  parameter int SID   = 0
) (
  input  logic                    clk,
  input  logic                    rst,
  input  logic                    start,
  input  cmd_t                    cmd,
  output logic                    rdy,
  output logic                    done,
  input  logic                    gnt,        // TMEM grant: the reader holds when low
  // TMEM read (port A lanes)
  output logic [LANES-1:0]        t_ren,
  output logic [LANES-1:0][31:0]  t_raddr,
  input  logic [LANES-1:0][31:0]  t_rdata,
  output logic [LANES-1:0]        t_ren2,
  output logic [LANES-1:0][31:0]  t_raddr2,
  // t_ren / t_ren2 by TMEM bank (TLANES banks; otpu_tmem PBR: the ports' rotator merge,
  // registered with them)
  output logic [TLANES-1:0]       t_rben, t_rben2,
  output logic [TLANES-1:0][31:0] t_rbad, t_rbad2,
  input  logic [LANES-1:0][31:0]  t_rdata2,
  output logic                    t_ren3,     // RSCALE: the row factor (one word)
  output logic [31:0]             t_raddr3,
  input  logic [31:0]             t_rdata3,
  // ACT RAM write
  output logic [LANES-1:0]        act_we,
  output logic [7:0]              act_row,
  output logic [31:0]             act_idx,
  output logic [LANES-1:0][7:0]   act_data,
  output logic                    act_dup,    // QACT DUP: also rows act_row + act_off
  output logic [7:0]              act_off,
  output logic                    asc_we,
  output logic [7:0]              asc_row,
  output logic [15:0]             asc_blk,
  output logic [63:0]             asc_data,   // {da3, da2, da1, da0, S}
  // DRAM port A (QST writes). a_want: a write is waiting (before the grant); the slice withholds
  // the grant while the port cannot take it. QST completes once its writes are acknowledged.
  output logic                    a_want,
  input  logic                    wr_idle,
  output logic                    a_req,
  output logic                    a_we,
  output logic [31:0]             a_addr,
  output logic [31:0]             a_wdata,
  output logic [3:0]              a_be,
  // profiling: a cycle after a QACT ends, its cycles frozen by the TMEM grant
  output logic                    pf_u,
  output logic [31:0]             pf_frz
);
  localparam int LM = 2, LA = 4;
  localparam int LR = 3;                    // x * r: otpu_qfmul3 (the rounding in two stages)
  localparam int QL = 2 + LR + LM;          // prescale latency: the data's and the operands'
                                            // registers, x * r, then * c
  localparam int DL = $clog2(D);
  localparam int QG = D / 4;                // QACT group (sub-block) elements
  localparam int QGL = $clog2(QG);
  localparam int QN = (LANES > QG) ? LANES / QG : 1;   // QACT groups a chunk holds
  localparam int PL = LANES / QN;           // a chunk's lanes per QACT group
  localparam int GWN = GR / 4;              // group words (four group maxes each)
  localparam int GA = $clog2(GWN);
  localparam int NB = 4;                    // row banks: group maxes and scales of rows in flight
  localparam int XW = (XD > 1) ? $clog2(XD) : 1;
  localparam int QD = 32;                   // QST's write queue (entries; credits)
  localparam int QDW = $clog2(QD);
  initial if (XD == 1 || (XD & (XD - 1)) != 0) $fatal(1, "otpu_quant: XD 0 or a power of two >= 2");
  initial if (D % LANES != 0 || (D & (D - 1)) != 0)
    $fatal(1, "otpu_quant: D a power of two, a multiple of LANES");
  initial if (LANES < 4) $fatal(1, "otpu_quant: QST word writes need LANES >= 4");
  initial if ((LANES & (LANES - 1)) != 0) $fatal(1, "otpu_quant: LANES must be a power of two");
  initial if (GR % 4 != 0 || (GWN & (GWN - 1)) != 0) $fatal(1, "otpu_quant: GR 4 x a power of two");

  // ------------------------------------------------------------------ command
  logic        busy, is_st, rowm, csf, rsf, ackw, ack2, dup;
  // QACT through the row buffer; it also selects the writer's operands (512 bits at LANES 8)
  (* max_fanout = 64 *) logic bm;
  logic [31:0] csb, rsb, srs, drs, es;
  logic [15:0] rows, KB;
  logic [7:0]  ab;
  logic [31:0] G;                            // a row's elements
  logic [31:0] GW;                           // ... read in pass 1 (QST HALF: the first half)
  logic [31:0] sbase, sstep;                 // QST: the row's scale row; the next one's step
  logic [15:0] swn;                          // QST: a scale row's words
  // cycles an instruction was frozen by the TMEM grant, for the profiler: the condition is
  // registered (st_c) and summed a cycle late, off the grant path; the count is st_frz + st_c
  // (a start clears it: the unit was idle the cycle before, st_c is 0)
  logic [31:0] st_frz;
  logic        st_c;
  assign rdy = !busy;
  // wr_idle registered: QST's end waits a cycle more (ack2), and the slice's write count and the
  // board's channel logic in front of it do not reach busy (250 MHz, FK35: g_sll u_tx s_wdone
  // -> busy, 7 levels, -0.26 ns)
  logic        wr_idle_q;
  always_ff @(posedge clk) wr_idle_q <= wr_idle;

  // ------------------------------------------------------------------ reader
  // walks rows r and their elements e, twice (pass 0: the maxes, pass 1: quantize); it advances
  // on the cycles the TMEM grant is given
  logic [15:0] r;
  logic [31:0] e;                            // element in the row
  logic [31:0] row_src;                      // TMEM address of the row
  logic [31:0] badr, brs;                    // QST byte address dst + r*drs + e*es (mod 2^32)
                                             // at (r, e) / at e = 0
  logic        pass;
  logic        wide;                         // QST pass 1: four contiguous bytes (a word) a cycle
  logic        rd_wait;                      // waiting for the row's scale (and scale row)
  logic        rd_done;
  logic        sw_ph;                        // QST: the scale row's writes
  logic [15:0] sw_i;                         // ... its word
  // elements per read: QST pass 1 one byte (or one word) a cycle, everything else LANES
  wire [31:0] ew = (is_st && pass) ? (wide ? 32'd4 : 32'd1) : 32'(LANES);
  wire [31:0] bstep = wide ? 32'd4 : es;     // QST pass 1: badr per read
  wire [31:0] Gr = pass ? GW : G;            // the elements of this pass
  wire rlast = (e + ew >= Gr);
  // the elements left in the row, Gr - e, kept alongside e (set wherever e or the pass moves):
  // lane l reads while l < ew and l < rem, a compare with a small constant instead of the adder
  // and compare of e + l < Gr, so the read enables and TMEM's block RAM enables do not follow two
  // carry chains (133.33 MHz, 110ec6d: e -> e + l -> < Gr -> the Q2 port's lane and bank enables
  // -> block RAM ENB, 12 levels, +0.258 ns)
  logic [31:0] rem;
  logic [LANES-1:0] rmask;
  always_comb for (int l = 0; l < LANES; l++) rmask[l] = 32'(l) < ew && 32'(l) < rem;
  // the row buffer: pass 1's row r1 (popping while p1a), its element, the FIFO's pointers and
  // its entries taken by issued reads (credits)
  logic        p1a;
  logic [15:0] r1;
  logic [31:0] e1c;
  logic [XW-1:0] xw, xr;
  logic [XW:0] xocc;                         // <= XD: below XD while its top bit is clear
  logic [2:0]  dr;                           // r - r1 (<= NB)
  // QST's write credits: the pass 1 reads issued whose write is not yet out of the queue (one
  // write each), and that count below QD / zero, registered with it
  logic [QDW:0] qcr;
  logic        qc_ok, qc_z;
  // buffered: a read needs a free FIFO entry, and row r's banks free (pass 1 of row r - NB done);
  // QST pass 1: a free queue entry; one LUT on the read enables' path
  wire xok = !bm || (!xocc[XW] && !dr[2]);
  wire qok = !is_st || !pass || qc_ok;
  wire riss = busy && !ackw && !rd_done && !rd_wait && !sw_ph && xok && qok;

  typedef struct packed {
    logic             v;
    logic             pass;
    logic [31:0]      e;         // element index within the row (of lane 0)
    logic             last;      // last chunk of the pass
    logic             glast;     // ... of the instruction
    logic [15:0]      row;
    logic [LANES-1:0] mask;
    logic [31:0]      baddr;     // QST: the byte address (badr at the read)
  } rm_t;
  // m0a: the chunk whose read request is in the request register (TMEM reads it this cycle);
  // m0: the chunk whose data arrives now, held (with its group word) until the grant that takes
  // it into the datapath; m0g: m0 entering it (valid on that grant); mp: after the prescale
  rm_t m0a, m0, m0g, mp;
  always_ff @(posedge clk) if (rst) m0 <= '0; else if (gnt) m0 <= m0a;
  always_comb begin
    m0g = m0;
    m0g.v = m0.v && gnt;
  end
  // buffered pass 1: a chunk popped (its data and group word arriving), then +1, +2 (the writer's
  // stage, like mp); the offsets' stages take their chunk from mb0 or m0
  rm_t mb0, mb1, mb2;
  rm_t m1s;
  assign m1s = bm ? mb0 : m0;

  // TMEM's input stage (release r2, docs/tmem_nocopy.md 7): the three read requests (t_*_c) go
  // to TMEM from flip-flops, a cycle after they are formed, advancing with the grant; the
  // position's arithmetic (row_src + e + l, csb + e + l, rsb + r) and the issue test end at the
  // register, not at the TMEM's ranks and pumped banks. The chunk waits a cycle in m0a for it.
  // The requests by TMEM bank too (t_rben*, otpu_tmem PBR), so TMEM's ranks and compares start at
  // flip-flops, not at its rotation of the lanes (r150d, the m8 card at a real 150 MHz:
  // t_raddr -> the rotation -> the ranks -> a pumped bank's address, 6 levels)
  logic [LANES-1:0]        t_ren_c, t_ren2_c;
  logic [LANES-1:0][31:0]  t_raddr_c, t_raddr2_c;
  logic [TLANES-1:0]       t_rben_c, t_rben2_c;
  logic [TLANES-1:0][31:0] t_rbad_c, t_rbad2_c;
  logic                    t_ren3_c;
  logic [31:0]             t_raddr3_c;
  otpu_tmem_breq #(.LANES(LANES), .TLANES(TLANES)) u_tb1 (.en(t_ren_c), .a0(t_raddr_c[0]),
                                                          .ben(t_rben_c), .bad(t_rbad_c));
  otpu_tmem_breq #(.LANES(LANES), .TLANES(TLANES)) u_tb2 (.en(t_ren2_c), .a0(t_raddr2_c[0]),
                                                          .ben(t_rben2_c), .bad(t_rbad2_c));
  always_ff @(posedge clk)
    if (rst) begin t_ren <= '0; t_ren2 <= '0; t_ren3 <= 1'b0; t_rben <= '0; t_rben2 <= '0; end
    else if (gnt) begin
      t_ren <= t_ren_c; t_ren2 <= t_ren2_c; t_ren3 <= t_ren3_c;
      t_raddr <= t_raddr_c; t_raddr2 <= t_raddr2_c; t_raddr3 <= t_raddr3_c;
      t_rben <= t_rben_c; t_rben2 <= t_rben2_c; t_rbad <= t_rbad_c; t_rbad2 <= t_rbad2_c;
    end
  // (the addresses follow the position alone: a lane's counts only while its enable is up)
  always_comb begin
    t_ren_c = '0; t_ren2_c = '0; t_ren3_c = 1'b0;
    for (int l = 0; l < LANES; l++) begin
      t_raddr_c[l] = row_src + e + 32'(l);
      t_raddr2_c[l] = csb + e + 32'(l);
    end
    t_raddr3_c = rsb + 32'(r);
    if (riss) begin
      for (int l = 0; l < LANES; l++) begin
        if (rmask[l]) begin
          t_ren_c[l] = 1'b1;
          if (csf) t_ren2_c[l] = 1'b1;
        end
      end
      if (rsf) t_ren3_c = 1'b1;
    end
  end
`ifndef SYNTHESIS
  bit rem_rst;                                       // (registers start arbitrary)
  initial rem_rst = 1'b0;
  always @(posedge clk) begin
    if (rst) rem_rst <= 1'b1;
    if (!rst && rem_rst && busy && !ackw && rem != Gr - e)
      $fatal(1, "otpu_quant: rem %0d is not Gr %0d - e %0d", rem, Gr, e);
  end
`endif

  // ------------------------------------------------------------------ prescale
  // The TMEM read data lands in plain flip-flops (kept: not taken into the DSPs' input
  // registers), nothing in front of them; the CSCALE and RSCALE selects come a stage later, and
  // each lane takes its own copy of the row factor (250 MHz, FK35: TMEM's UltraRAM -> its lane
  // muxes -> x1 / x2 / u_r's A input, 4-5 levels, -0.61 to -0.69 ns, behind the grant's enable;
  // r3 on eight lanes' unregistered DSP inputs -> u_r's P, 7 levels, -0.21 ns). m0g goes along
  // (QL), with QST's byte address in it
  f32_t xp [LANES];
  (* keep *) f32_t r3;
  always_ff @(posedge clk) r3 <= t_rdata3;
  for (genvar l = 0; l < LANES; l++) begin : g_pre
    (* keep *) f32_t x1, x2, rl;
    f32_t x1b, v1, cd;
    always_ff @(posedge clk) begin
      x1 <= t_rdata[l];
      x2 <= t_rdata2[l];
      x1b <= x1;
      rl <= rsf ? r3 : F_ONE;
    end
    otpu_qfmul3 u_r (.clk, .a(x1b), .b(rl), .y(v1));
    otpu_delay #(.W(32), .N(LR + 1)) u_c (.clk, .en(1'b1), .d(csf ? x2 : F_ONE), .q(cd));
    otpu_fmul #(.LAT(LM)) u_c2 (.clk, .en(1'b1), .a(v1), .b(cd), .y(xp[l]));
  end
  otpu_delay #(.W($bits(rm_t)), .N(QL)) u_mp (.clk, .en(1'b1), .d(m0g), .q(mp));

  // fp_gt for the max path: every operand there is a non-negative flushed value (an otpu_fmul
  // result with the sign cleared, +0, or one taken from those), where fkey is {1, a[30:0]}, so
  // the order is the unsigned order of a[30:0] (NaN still above +inf) and needs no ftz
  function automatic logic mgt(input f32_t a, input f32_t b);
    return a[30:0] > b[30:0];
  endfunction

  // ------------------------------------------------------------------ pass 0: maxes
  // The chunk's max over contiguous lanes in two registered stages: pairs, then the rest; on the
  // way, the max of each QACT group the chunk holds (QN of PL lanes, where D/4 < LANES)
  localparam int HL = LANES / 2;
  f32_t  ca [HL];
  f32_t  cmax;
  f32_t  pmx [QN];
  rm_t   mh, mc;
  always_ff @(posedge clk) begin
    f32_t v [LANES];
    // xp is flushed (otpu_fmul), so clearing the sign is fabs
    for (int l = 0; l < LANES; l++) v[l] = mp.mask[l] ? {1'b0, xp[l][30:0]} : F_ZERO;
    for (int l = 0; l < HL; l++) ca[l] <= mgt(v[2*l + 1], v[2*l]) ? v[2*l + 1] : v[2*l];
    mh <= mp;
  end
  always_ff @(posedge clk) begin
    f32_t v [HL];
    f32_t pm [QN];
    for (int l = 0; l < HL; l++) v[l] = ca[l];
    for (int p = 0; p < QN; p++) pm[p] = v[p * PL / 2];          // (PL 2: the pairs)
    for (int s = 1; s < HL; s = s * 2) begin                      // v[l] covers 4s lanes after
      for (int l = 0; l + s < HL; l = l + 2 * s)
        if (mgt(v[l + s], v[l])) v[l] = v[l + s];
      if (4 * s == PL) for (int p = 0; p < QN; p++) pm[p] = v[p * PL / 2];
    end
    cmax <= v[0];
    pmx <= pm;
    mc <= mh;
  end

  // the row's running max with this chunk folded in (a row's first chunk restarts it): the
  // scale unit's input at the row's last chunk
  f32_t amx_r, nmax;
  assign nmax = (mc.e != 0 && mgt(amx_r, cmax)) ? amx_r : cmax;
  // the group maxes: QST's D-blocks, QACT's sub-blocks of D/4 (one or more chunks each, or QN a
  // chunk), folded like the row's and written at the group's last chunk to the group RAMs
  // (group g in RAM g%4, word g/4), a stage later
  f32_t gacc, gnew;
  wire [31:0] mce = mc.e, mcn = mc.e + LANES;
  wire gfst = is_st ? (mce[DL-1:0] == 0) : (QN > 1 || mce[QGL-1:0] == 0);
  wire glst = is_st ? (mcn[DL-1:0] == 0) : (QN > 1 || mcn[QGL-1:0] == 0);
  wire [31:0] gix = is_st ? (mce >> DL) : (mce >> QGL);   // the chunk's (first) group
  assign gnew = (!gfst && mgt(gacc, cmax)) ? gacc : cmax;
  logic [3:0]       gw_we;
  logic [GA+1:0]    gw_a;                    // {row bank, word}
  logic [3:0][30:0] gw_d;
  always_ff @(posedge clk) begin
    gw_we <= '0;
    if (mc.v && !mc.pass && !rowm && glst) begin
      if (is_st || QN == 1) begin
        gw_we[gix[1:0]] <= 1'b1;
        gw_d[gix[1:0]] <= gnew[30:0];
      end else begin
        for (int p = 0; p < QN; p++) begin
          gw_we[2'(gix[1:0] + 2'(p))] <= 1'b1;
          gw_d[2'(gix[1:0] + 2'(p))] <= pmx[p][30:0];
        end
      end
      gw_a <= {mc.row[1:0], GA'(gix >> 2)};
    end
  end
  // the group RAMs' reads: a pass 1 chunk's group word (QACT: the block's four groups; QST: the
  // word of its block) at its request, from m0a (the word arrives with m0 and is held with it:
  // the read advances with the grant); a buffered pass 1 reads it at the pop (with mb0, every
  // cycle); QST's scale row phase reads the next scale-row word. NB banks of GWN words, row r in
  // bank r % NB
  logic [3:0][30:0] grd;
  wire [GA+1:0] gr_a = sw_ph ? {r[1:0], GA'(sw_i)} : bm ? {r1[1:0], GA'(e1c >> DL)}
                     : is_st ? {m0a.row[1:0], GA'(m0a.e >> (DL + 2))}
                     : {m0a.row[1:0], GA'(m0a.e >> DL)};
  wire gr_re = bm || gnt;
  for (genvar s = 0; s < 4; s++) begin : g_gr
    otpu_ram_1r1w #(.W(31), .N(NB * GWN), .STYLE("block")) u_g (
      .clk, .we(gw_we[s]), .wa(gw_a), .wd(gw_d[s]), .re(gr_re), .ra(gr_a), .rd(grd[s]));
  end

  // scale unit (its amax is don't-care while sq_iv is low: sc/inv are read only under sq_ov)
  logic        sq_iv, sq_ov;
  logic [1:0]  sq_ot;
  f32_t        sq_amax, sq_sc, sq_inv;
  assign sq_iv   = mc.v && mc.last && !mc.pass;
  assign sq_amax = nmax;
  otpu_qscale #(.LM(LM), .LA(LA), .TW(2)) u_sc (.clk, .rst, .en(1'b1), .iv(sq_iv),
    .itag(mc.row[1:0]), .amax(sq_amax), .ov(sq_ov), .otag(sq_ot), .sc(sq_sc), .inv(sq_inv));
  // the rows' S, inv and amax by bank (row r in r % NB: written at its scale's input / output,
  // read through its pass 1); sv: the bank's scale is out and its pass 1 not yet started
  f32_t       sc_q [NB], inv_q [NB], amx_q [NB];
  logic [NB-1:0] sv;

  // ------------------------------------------------------------------ pass 1: offsets
  // d = min(31, e(amax) - e(a) - (f(a) > f(amax))), 0 for a = 0 and in ROW mode (a <= amax).
  // A carry select: e(amax) - e(a) and e(amax) - e(a) - 1 (+ ~e(a)) beside the fraction compare,
  // which picks one, saturated (the compare was the subtract's operand: the group RAMs' output,
  // the compare's and the subtract's carry chains and the saturation in series, 9 levels, the
  // m8 card at a 150 MHz equivalent -0.026 ns)
  function automatic logic [4:0] dval(input logic [30:0] a, input f32_t amax, input logic row);
    logic [8:0] d0, d1;
    if (row || a == 0) return 5'd0;
    d0 = 9'(amax[30:23]) - 9'(a[30:23]);
    d1 = 9'(amax[30:23]) + {1'b1, ~a[30:23]};
    if (a[22:0] > amax[22:0]) return (d1[8:5] != 0) ? 5'd31 : d1[4:0];
    return (d0[8:5] != 0) ? 5'd31 : d0[4:0];
  endfunction
`ifndef SYNTHESIS
  function automatic logic [4:0] dval_ref(input logic [30:0] a, input f32_t amax, input logic row);
    logic [8:0] d;
    if (row || a == 0) return 5'd0;
    d = 9'(amax[30:23]) - 9'(a[30:23]) - ((a[22:0] > amax[22:0]) ? 9'd1 : 9'd0);
    return (d > 9'd31) ? 5'd31 : d[4:0];
  endfunction
  // (dq below takes dval of grd every cycle: the group RAMs' read follows the grant)
  always @(posedge clk) if (!rst)
    for (int s = 0; s < 4; s++)
      if (dval(grd[s], amx_q[m1s.row[1:0]], rowm) != dval_ref(grd[s], amx_q[m1s.row[1:0]], rowm))
        $fatal(1, "otpu_quant: dval is not the reference's");
`endif
  // inv * 2^d: inv is +0, +inf or positive and normal (otpu_qscale), so the product is exact:
  // the exponent field plus d, +inf where it reaches 255
  function automatic f32_t p2mul(input f32_t inv, input logic [4:0] d);
    logic [8:0] x;
    if (inv[30:23] == 8'd0) return F_ZERO;
    x = 9'(inv[30:23]) + 9'(d);
    return (x >= 9'd255) ? F_INF : {1'b0, x[7:0], inv[22:0]};
  endfunction
  // the chunk's group word (with m0 or mb0), its four d a stage later, each lane's inv * 2^d and
  // the block's record bytes the next, delayed to the prescale's output
  logic [3:0][4:0] dq;
  logic [31:0]     e1;
  logic [1:0]      b1;                       // the chunk's row bank
  always_ff @(posedge clk) begin
    for (int s = 0; s < 4; s++) dq[s] <= dval(grd[s], amx_q[m1s.row[1:0]], rowm);
    e1 <= m1s.e;
    b1 <= m1s.row[1:0];
  end
  f32_t        invl [LANES], invp [LANES];
  logic [31:0] dal, dap;
  always_ff @(posedge clk) begin
    for (int l = 0; l < LANES; l++) begin
      logic [31:0] el;
      el = e1 + 32'(l);
      invl[l] <= p2mul(inv_q[b1], dq[is_st ? el[DL+1:DL] : el[QGL+1:QGL]]);
    end
    dal <= {3'b0, dq[3], 3'b0, dq[2], 3'b0, dq[1], 3'b0, dq[0]};
  end
  for (genvar l = 0; l < LANES; l++) begin : g_inv
    otpu_delay #(.W(32), .N(QL - 2)) u_i (.clk, .en(1'b1), .d(invl[l]), .q(invp[l]));
  end
  otpu_delay #(.W(32), .N(QL - 2)) u_da (.clk, .en(1'b1), .d(dal), .q(dap));
`ifndef SYNTHESIS
  // p2mul is fp_mul(inv, 2^d) for every inv the scale unit gives
  always @(posedge clk) if (!rst)
    for (int l = 0; l < LANES; l++) begin
      logic [31:0] el;
      el = e1 + 32'(l);
      if (p2mul(inv_q[b1], dq[is_st ? el[DL+1:DL] : el[QGL+1:QGL]]) !=
          fp_mul(inv_q[b1], {1'b0, 8'(8'd127 + 8'(dq[is_st ? el[DL+1:DL] : el[QGL+1:QGL]])), 23'd0}))
        $fatal(1, "otpu_quant: inv %h * 2^d is not fp_mul's", inv_q[b1]);
    end
`endif

  // ------------------------------------------------------------------ writer
  typedef struct packed {
    logic             v;
    logic [LANES-1:0] mask;
    logic [7:0]       row;
    logic [31:0]      idx;       // ACT byte index
    logic             sw;        // write the scale record (first chunk of a block)
    logic [15:0]      blk;
    logic [31:0]      da;        // the block's offsets {da3, da2, da1, da0}
    logic             st;        // QST byte
    logic [31:0]      baddr;     // QST byte address
    logic             fin;       // last write of the instruction
    logic [31:0]      sc;        // the row's S (the scale record)
  } wm_t;
  wm_t   wq, wqd;                            // at the multiplier input / at the write stage
  f32_t  wx [LANES], winv [LANES];
  // buffered pass 1: the popped chunk's data, delayed to the offsets' output (mb2)
  f32_t  xd [LANES], xb1 [LANES], xb2 [LANES];
  always_ff @(posedge clk) begin
    xb1 <= xd;
    xb2 <= xb1;
  end
  always_ff @(posedge clk)
    if (rst) begin
      mb1 <= '0;
      mb2 <= '0;
    end else begin
      mb1 <= mb0;
      mb2 <= mb1;
    end
  // the row buffer: pass 0's prescaled chunks (buffered QACT), popped by pass 1 in order (the
  // read registered: the popped chunk's data arrives with mb0)
  wire xpush = bm && mp.v && !mp.pass;
  if (XD > 0) begin : g_xb
    logic [LANES*32-1:0] xwd, xrd;
    always_comb for (int l = 0; l < LANES; l++) xwd[32*l +: 32] = xp[l];
    otpu_ram_1r1w #(.W(LANES * 32), .N(XD), .STYLE("block")) u_xb (
      .clk, .we(xpush), .wa(xw), .wd(xwd), .re(1'b1), .ra(xr), .rd(xrd));
    always_comb for (int l = 0; l < LANES; l++) xd[l] = xrd[32*l +: 32];
  end else begin : g_nxb
    always_comb for (int l = 0; l < LANES; l++) xd[l] = F_ZERO;
  end
`ifndef SYNTHESIS
  // the FIFO: a pop takes a chunk pushed before it, a push never overruns (the credits)
  int unsigned xs_n;
  always @(posedge clk)
    if (rst || start) xs_n = 0;
    else if (busy) begin
      if (p1a && xs_n == 0) $fatal(1, "otpu_quant: row buffer popped empty");
      xs_n = xs_n + (xpush ? 1 : 0) - (p1a ? 1 : 0);
      if (xs_n > XD) $fatal(1, "otpu_quant: row buffer overrun");
    end
`endif
  // the writer's chunk: a buffered pass 1's pop (mb2) or a pass 1 chunk out of the prescale (mp),
  // never both in one instruction, so bm picks the operands: a static select (its own copies,
  // bm's max_fanout), not mb2.v and mp's valid bits on every operand bit (250 MHz, FK35: u_mp's
  // qr -> winv_r, 1 level, 95% route, -0.46 ns). The fields are don't-care without wq.v
  always_comb begin
    rm_t m;
    m = bm ? mb2 : mp;
    wq.v = bm ? mb2.v : (mp.v && mp.pass);
    wq.mask = m.mask;
    for (int l = 0; l < LANES; l++) begin
      wx[l] = bm ? xb2[l] : xp[l];
      winv[l] = bm ? invl[l] : invp[l];
    end
    wq.row = m.row[7:0];
    wq.idx = (32'(ab) * D + m.e) & ~32'(LANES - 1);   // QACT: e steps by LANES (QST: unused)
    wq.sw = !is_st && (m.e[DL-1:0] == 0);
    wq.blk = 16'(ab) + 16'(m.e >> DL);
    wq.da = bm ? dal : dap;
    wq.st = is_st;
    wq.baddr = mp.baddr;
    wq.fin = m.glast;
    wq.sc = sc_q[m.row[1:0]];
  end
  // the writer's selection registered in front of the multipliers
  wm_t   wq_r;
  f32_t  wx_r [LANES], winv_r [LANES];
  always_ff @(posedge clk) begin
    wq_r <= wq;
    wx_r <= wx;
    winv_r <= winv;
  end
  // q8_s1 with the shift (and its guard bit k = sh - 1) decoded from e by a table, not computed
  // as 150 - e: no carry chain between e and ip/g/st (the e compares feed only zero/sat). No
  // ftz: it changes only m, and only when e == 0, where zero makes q8_s2 return 0 anyway.
  // A NaN product (0 * inf: a zero beside a tiny amax, whose inv is +inf) quantizes to 0.
  function automatic q8_mid_t qq8_s1(input f32_t x_in);
    q8_mid_t q;
    f32_t x;
    logic [7:0]  e;
    logic [23:0] m;
    logic [4:0]  sh, k;
    x = x_in;
    e = x[30:23];
    m = {1'b1, x[22:0]};
    q = '0;
    q.s = x[31];
    q.zero = (e < 8'd126) || is_nan(x);
    q.sat = (e >= 8'd134);
    sh = 5'd17; k = 5'd16;                      // zero or sat (as q8_s1)
    for (int c = 126; c < 134; c++)
      if (e == 8'(c)) begin sh = 5'(150 - c); k = 5'(149 - c); end
    q.ip = 32'(m) >> sh;
    q.g = m[k];
    q.st = |(m & ~(24'hFF_FFFF << k));          // |m[k-1:0]
    return q;
  endfunction
  logic [LANES-1:0][7:0] qb;
  for (genvar l = 0; l < LANES; l++) begin : g_q
    f32_t p;
    q8_mid_t qm;
    otpu_fmul #(.LAT(LM)) u_q (.clk, .en(1'b1), .a(wx_r[l]), .b(winv_r[l]), .y(p));
    always_ff @(posedge clk) begin
      qm <= qq8_s1(p);
      qb[l] <= q8_s2(qm);
    end
  end
  otpu_qdly #(.W($bits(wm_t)), .N(LM + 2)) u_wq (.clk, .rst, .en(1'b1), .d(wq_r), .q(wqd));  // ends in a flip-flop

  // ------------------------------------------------------------------ QST's write queue
  // QST's byte / word writes wait here for DRAM port A: the head is the request (a_want from
  // flip-flops), taken in a cycle the grant is up (the slice withholds it while the port cannot
  // take a write), so a write leaves exactly as before: none after the port's ready falls. QD
  // entries in distributed RAM; the reader issues a pass 1 read only with a credit (qcr: the
  // reads issued whose write is not yet out), so the datapath, which no longer stops, never
  // overruns it, and at full rate (QD covers the ~14 cycles from an issue to its write's pop).
  // The scale row's words go out directly, between a row's passes, once every data write before
  // them is out (qc_z)
  typedef struct packed {
    logic        fin;
    logic [29:0] addr;
    logic [31:0] data;
    logic [3:0]  be;
  } qw_t;
  (* ram_style = "distributed" *) logic [$bits(qw_t)-1:0] qf_m [QD];
  qw_t            qf_in, qf_hd;
  logic [QDW-1:0] qf_w, qf_r;
  logic [QDW:0]   qf_n;
  logic           qf_v;                      // not empty (qf_n != 0, registered with it; 0 at
                                             // power-up too: no a_want before the reset)
  wire qf_push = busy && !ackw && wqd.v && wqd.st;
  wire qf_pop  = qf_v && gnt;
  always_comb begin
    qf_in.fin = wqd.fin;
    qf_in.addr = wqd.baddr[31:2];
    if (wqd.mask[1]) begin                    // a word (wide: the row is word aligned)
      qf_in.data = {qb[3], qb[2], qb[1], qb[0]};
      qf_in.be = 4'hF;
    end else begin
      qf_in.data = {4{qb[0]}};
      qf_in.be = 4'b0001 << wqd.baddr[1:0];
    end
  end
  always_ff @(posedge clk) if (qf_push) qf_m[qf_w] <= qf_in;
  assign qf_hd = qf_m[qf_r];
  always_ff @(posedge clk)
    if (rst || start) begin
      qf_w <= '0; qf_r <= '0; qf_n <= '0; qf_v <= 1'b0;
      qcr <= '0; qc_ok <= 1'b1; qc_z <= 1'b1;
    end else begin
      logic [QDW:0] n, c;
      n = qf_n + (qf_push ? 1'b1 : 1'b0) - (qf_pop ? 1'b1 : 1'b0);
      c = qcr + ((gnt && riss && is_st && pass) ? 1'b1 : 1'b0) - (qf_pop ? 1'b1 : 1'b0);
      if (qf_push) qf_w <= qf_w + 1'b1;
      if (qf_pop) qf_r <= qf_r + 1'b1;
      qf_n <= n;
      qf_v <= (n != 0);
      qcr <= c;
      qc_ok <= (c < QD);
      qc_z <= (c == 0);
    end
`ifndef SYNTHESIS
  // pushes stay within the credits, the queue within QD (from the first reset on)
  always @(posedge clk)
    if (!rst && rem_rst && !start && (32'(qf_n) > 32'(qcr) || 32'(qcr) > QD))
      $fatal(1, "otpu_quant: QST queue %0d writes, %0d credits taken (of %0d)", qf_n, qcr, QD);
`endif

  // QST's scale row: S, then (not ROW) the d words: word j holds groups 4j .. 4j+3, the group
  // RAMs' word j (read the cycle before), the bytes of groups past KB 0
  logic [31:0] dword;
  always_comb begin
    dword = '0;
    for (int s = 0; s < 4; s++)
      if (32'(sw_i) * 4 + 32'(s) < 32'(KB) + 4)        // groups 4(sw_i - 1) + s < KB
        dword[8*s +: 8] = {3'b0, dval(grd[s], amx_q[r[1:0]], 1'b0)};
  end
  // a DRAM write is waiting (independent of the grant, which the slice derives from it)
  assign a_want = qf_v || sw_ph;

  assign act_dup = dup;
  assign act_off = rows[7:0];

  always_comb begin
    act_we = '0; act_row = '0; act_idx = '0; act_data = '0;
    asc_we = 1'b0; asc_row = '0; asc_blk = '0; asc_data = '0;
    a_req = 1'b0; a_we = 1'b0; a_addr = '0; a_wdata = '0; a_be = '0;
    if (busy && !ackw && wqd.v && !wqd.st) begin
      act_row = wqd.row;
      // QACT writes are whole chunks (idx LANES aligned, every mask bit set); saying so here
      // folds the ACT RAM's byte rotator to a chunk decode
      act_idx = wqd.idx & ~32'(LANES - 1);
      for (int l = 0; l < LANES; l++) begin
        act_we[l] = wqd.mask[0];
        act_data[l] = qb[l];
      end
      if (wqd.sw) begin
        asc_we = 1'b1;
        asc_row = wqd.row;
        asc_blk = wqd.blk;
        asc_data = {wqd.da, wqd.sc};
      end
    end
    if (qf_v) begin
      a_req = gnt;
      a_we = gnt;
      a_addr = {2'b00, qf_hd.addr};
      a_wdata = qf_hd.data;
      a_be = qf_hd.be;
    end else if (sw_ph) begin
      a_req = gnt;
      a_we = gnt;
      a_addr = (sbase + {sw_i, 2'b00}) >> 2;
      a_wdata = (sw_i == 0) ? sc_q[r[1:0]] : dword;
      a_be = 4'hF;
    end
  end
`ifndef SYNTHESIS
  // the scale row goes out between a row's passes: no data write queued or arriving then
  always @(posedge clk)
    if (!rst && busy && sw_ph && (qf_push || qf_v))
      $fatal(1, "otpu_quant: a QST data write during the scale row");
`endif

  // ------------------------------------------------------------------ sequencing
  // done is reported two cycles after the instruction's last ACT RAM write (done_w, then done)
  // and the ACT RAM lands the write two edges after done rises (three registers: otpu_actram).
  // The sequencer clears the dependency at the edge after done rises and releases the dependent
  // MM (urel) at the next, the one that lands the write, so the MM's first read of the block is
  // at least an edge after it landed; otpu_actram's simulation check stops on a consumed read of
  // a block still in flight. QST's done follows the last write out of the queue, once the DRAM
  // has taken every write (wr_idle, registered: ack2 skips the cycle it does not cover yet).
  // The reader advances with the grant; the rest (the pops, the counters' datapath sides, the
  // maxes, the scales, the last write) every cycle. A row's scale out of the scale unit (sq_ov,
  // one cycle) is kept for the reader (sq_p) until a grant takes it (QST: once its data writes
  // are out, qc_z)
  logic done_w;
  logic sq_p;
  wire  sq_f = sq_ov || sq_p;
  wire  sq_take = sq_f && !bm && gnt && (!is_st || qc_z);
  always_ff @(posedge clk) done <= !rst && done_w;
  always_ff @(posedge clk) begin
    done_w <= 1'b0;
    pf_u <= 1'b0;
    st_c <= busy && !gnt;
    st_frz <= st_frz + 32'(st_c);
    if (rst) begin
      busy <= 1'b0;
      ackw <= 1'b0;
      ack2 <= 1'b0;
      sw_ph <= 1'b0;
      m0a <= '0;
      mb0 <= '0;
      p1a <= 1'b0;
      bm <= 1'b0;
      sq_p <= 1'b0;
    end else if (ackw) begin
      m0a <= '0;
      mb0 <= '0;
      ack2 <= 1'b1;
      if (ack2 && wr_idle_q) begin
        ackw <= 1'b0;
        ack2 <= 1'b0;
        busy <= 1'b0;
        done_w <= 1'b1;
      end
    end else if (start) begin
      logic qst;
      logic [15:0] nr, kb;
      logic [31:0] nw;
      qst = (cmd.op == OP_QST);
      nr = qst ? cmd.w4[15:0] : 16'(cmd.w2[7:0]);
      kb = qst ? cmd.w4[31:16] : cmd.w2[31:16];
      nw = (32'(kb) + 32'd3) >> 2;                // d words of a scale row
      is_st <= qst;
      rowm  <= cmd.flags[0];
      csf   <= !qst && cmd.flags[1];
      csb   <= cmd.w4;
      rsf   <= !qst && cmd.flags[2];
      rsb   <= cmd.w5;
      dup   <= !qst && cmd.flags[3];
      rows  <= nr;
      KB    <= kb;
      G     <= 32'(kb) * D;
      rem   <= 32'(kb) * D;
      if (qst) begin
        srs  <= cmd.w5;
        drs  <= cmd.w6;
        es   <= cmd.w7;
        // HALF (ROW mode): the scale is the whole row's, pass 1 writes its first half
        GW   <= (32'(kb) * D) >> ((cmd.flags[0] && cmd.flags[1]) ? 1 : 0);
        sstep <= cmd.flags[0] ? 32'd4 : 32'd4 + (nw << 2);
        swn  <= cmd.flags[0] ? 16'd1 : 16'(32'd1 + nw);
      end else begin
        ab   <= cmd.w2[15:8];
        srs  <= cmd.w3;
        GW   <= 32'(kb) * D;
      end
      r <= '0; e <= '0;
      badr <= cmd.w2; brs <= cmd.w2;             // QST dst (unused by QACT)
      sbase <= cmd.w3;                           // QST sdst
      row_src <= cmd.w1;
      pass <= 1'b0; rd_wait <= 1'b0; wide <= 1'b0; sw_ph <= 1'b0; sw_i <= '0;
      m0a <= '0;
      st_frz <= '0;
      bm <= XD > 0 && !qst && 32'(kb) * D <= 32'(XD) * LANES;
      p1a <= 1'b0; r1 <= '0; e1c <= '0; xw <= '0; xr <= '0; xocc <= '0; dr <= '0; sv <= '0;
      mb0 <= '0;
      sq_p <= 1'b0;
      if (nr == 0 || kb == 0) begin
        rd_done <= 1'b1;
        done_w <= 1'b1;
      end else begin
        rd_done <= 1'b0;
        busy <= 1'b1;
      end
    end else if (busy) begin
      // ---- reader (with the grant)
      if (gnt) begin
        m0a <= '0;
        if (riss) begin
          m0a.v <= 1'b1;
          m0a.pass <= pass;
          m0a.e <= e;
          m0a.last <= rlast;
          m0a.row <= r;
          m0a.mask <= rmask;
          m0a.glast <= pass && rlast && (32'(r) + 1 == 32'(rows));
          m0a.baddr <= badr;
          if (!rlast) begin
            e <= e + ew;
            rem <= rem - ew;
            badr <= badr + bstep;
          end else if (!pass && !bm) begin          // wait for the row's scale, re-read it
            rd_wait <= 1'b1;
            e <= '0;
            rem <= G;
            badr <= brs;
            wide <= is_st && es == 32'd1 && brs[1:0] == 2'd0;
          end else begin                             // the next row (buffered: after pass 0)
            e <= '0;
            rem <= G;
            pass <= 1'b0;
            wide <= 1'b0;
            r <= r + 1;
            if (32'(r) + 1 == 32'(rows)) rd_done <= 1'b1;
            brs <= brs + drs;
            badr <= brs + drs;
            row_src <= row_src + srs;
            sbase <= sbase + sstep;
          end
        end
        // the row's scale: QST writes its scale row, then pass 1 starts (buffered: pass 1 takes
        // it when it gets to the row, below)
        if (sq_take) begin
          if (is_st) begin
            sw_ph <= 1'b1;
            sw_i <= '0;
          end else begin
            pass <= 1'b1;
            rem <= GW;                               // pass 1 re-reads from e = 0
            rd_wait <= 1'b0;
          end
        end
        if (sw_ph) begin
          if (sw_i + 16'd1 == swn) begin
            sw_ph <= 1'b0;
            pass <= 1'b1;
            rem <= GW;
            rd_wait <= 1'b0;
          end else sw_i <= sw_i + 16'd1;
        end
      end
      sq_p <= sq_f && !bm && !sq_take;
      // ---- buffered pass 1: once row r1's scale is out, pop its chunks (one a cycle, the
      // group word read with each); a row's start takes a cycle
      mb0 <= '0;
      if (p1a) begin
        mb0.v <= 1'b1;
        mb0.e <= e1c;
        mb0.row <= r1;
        mb0.mask <= '1;
        mb0.last <= (e1c + LANES >= G);
        mb0.glast <= (e1c + LANES >= G) && (32'(r1) + 1 == 32'(rows));
        xr <= xr + 1'b1;
        if (e1c + LANES >= G) begin
          p1a <= 1'b0;
          r1 <= r1 + 1'b1;
        end else e1c <= e1c + LANES;
      end else if (bm && 32'(r1) < 32'(rows) && sv[r1[1:0]]) begin
        p1a <= 1'b1;
        e1c <= '0;
      end
      // the FIFO's entries taken (a buffered read) and given back (a pop); rows between the
      // passes (r - r1)
      xocc <= xocc + ((bm && riss && gnt) ? 1'b1 : 1'b0) - (p1a ? 1'b1 : 1'b0);
      if (xpush) xw <= xw + 1'b1;
      dr <= dr + ((bm && riss && rlast && gnt) ? 3'd1 : 3'd0)
               - ((p1a && e1c + LANES >= G) ? 3'd1 : 3'd0);
      // ---- the maxes
      if (mc.v && !mc.pass) begin
        amx_r <= nmax;
        gacc <= gnew;
      end
      if (sq_iv) amx_q[mc.row[1:0]] <= nmax;
      // ---- the row's scale, by bank
      if (sq_ov) begin
        sc_q[sq_ot] <= sq_sc;
        inv_q[sq_ot] <= sq_inv;
        if (bm) sv[sq_ot] <= 1'b1;
      end
      if (!p1a && bm && 32'(r1) < 32'(rows) && sv[r1[1:0]]) sv[r1[1:0]] <= 1'b0;
      // ---- the last write: QACT's at the writer, QST's out of the queue
      if (wqd.v && wqd.fin && !wqd.st) begin
        busy <= 1'b0;
        done_w <= 1'b1;
        pf_u <= 1'b1;
        pf_frz <= st_frz + 32'(st_c);
      end
      if (qf_pop && qf_hd.fin) ackw <= 1'b1;        // the last write is taken this cycle
    end
  end
`ifndef SYNTHESIS
  // the ACT RAM write above takes a QACT write as a whole LANES-aligned chunk (act_we from
  // mask[0], act_idx with its low bits cleared): check that every QACT write is one
  always @(posedge clk)
    if (!rst && busy && !ackw && wqd.v && !wqd.st
        && (wqd.mask != '1 || (wqd.idx & 32'(LANES - 1)) != 0))
      $fatal(1, "otpu_quant: QACT write not a whole LANES-aligned chunk (mask %b, idx %0d)",
             wqd.mask, wqd.idx);
  // a row's group maxes fit the group RAMs
  always @(posedge clk)
    if (!rst && start && !cmd.flags[0] &&
        ((cmd.op == OP_QST) ? 32'(cmd.w4[31:16]) : 4 * 32'(cmd.w2[31:16])) > GR)
      $fatal(1, "otpu_quant: %0d groups a row, at most %0d", (cmd.op == OP_QST) ?
             32'(cmd.w4[31:16]) : 4 * 32'(cmd.w2[31:16]), GR);
`endif
endmodule
