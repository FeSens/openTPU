// Quantizer for QACT (TMEM rows -> ACT RAM) and QST (TMEM rows -> DRAM bytes), docs/isa.md
// "Quantization": per row (x after the optional RSCALE / CSCALE factors) amax = max|x|, the row's
// S = amax/127 and inv = 127*recip(amax); per group g (QACT: the sub-blocks of D/4 elements, QST:
// the D-blocks) its own max a_g and offset d_g = min(31, e(amax) - e(a_g) - (f(a_g) > f(amax)))
// (0 for a_g = 0 and in ROW mode); q = q8(x * (inv * 2^d_g)).
//
// Each row is read twice, both passes LANES elements a cycle (QST's pass 1: one byte, or a word
// of four, a cycle), everything advancing on the cycles the TMEM grant is given:
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
// PW > 1: a buffered pass 1 pops PW chunks a cycle (the row buffer in PW banks, chunk i in bank
// i % PW: a row starts at a multiple of D / LANES >= PW chunks), and the writer (the lanes'
// inv * 2^d, multipliers and q8) and the ACT RAM write are LANES x PW lanes: a row's pass 1
// takes 1 / PW of its pass 0's cycles (a decode QACT, one row, waits for both). An unbuffered
// pass 1 and QST stay at LANES a cycle (a QACT chunk's bytes in its lane group of the write).
// PW 1 is the quantizer without it.
//
// RSCALE's row factor is one word a row. R1 = 0 reads it with every chunk (Q3, beside Q and
// Q2); R1 = 1 reads it with the row's first chunk and holds it for the others (timing only:
// with CSCALE, the 17 reads of Q + Q2 + Q3 in a cycle did not fit the one-copy TMEM's banks, a
// 40-block QACT took 1,653 cycles against 1,333).
//
// QST GRP (WG > 1, GALL: docs/weight_groups.md; docs/isa.md QST) stores rows n = R[rd] + w6
// + r (the sequencer adds R[rd]) of an int8 matrix in groups of R = WG rows, chunks P = R x D:
// the same bytes and scales as QST's, at other addresses. A byte's address is {the chunk's
// address above P, its lane ^ the chunk's fold (CHASH), its byte}, from registers: rows mode
// the row's group chunk (stepped by P a D-block, by KB x P a group of rows), lane n % R, byte
// e % D, pass 1 a word a cycle; COL (V^T, ROW) element e's group chunk (stepped by es every R
// elements), lane e % R, byte n % D, a byte a cycle. Rows mode writes its scales into the
// group's area (stepped by es a group): S at 4 (n % R), block k's d at 4 R + k R + n % R, a
// byte a cycle; COL its S words as ROW does. The first row's group offsets (n0 / R) x KB P and
// (n0 / R) x es are two products, registered, during the first row's pass 0.
//
// SPLIT = 1 (Config.SPLITQ, CAPS2 bit17): QACT flag LO (bit4) / QST flag LO (bit7) writes the
// split's lo part instead of q, lo = sat127(rint(256 p) - 256 q8(p)) for the same product p =
// x * (inv * 2^d_g) (half to even; NaN 0), and the scale S * 2^-8 (flushed below the normals; the
// offsets unchanged), so q + lo / 256 is p to 2^-9: an operand split in two int8 halves for the
// MXU (chunked DeltaNet). The lanes' q8 then comes from the same shift, floor(256 |p|) in 16 bits
// (e 118 .. 133) and its guard and sticky bits; SPLIT = 0 is the quantizer without it.
//
// QHL = 1 (with SPLIT; Config.QHL, CAPS2 bit21): QACT flag HL (bit5) writes both halves from one
// read: q into blocks ab .. ab + KB - 1 and lo into ab + KB .. ab + 2 KB - 1, each block with its
// record (S, and S * 2^-8; the same offsets). Its rows go through the row buffer (KB x D <=
// QHL_ELEMS 8192 <= XD x LANES), and pass 1 pops each of a row's chunks twice, q then lo (the
// pop, its group word and its offsets as before; the second pop advances the FIFO): the writer's
// lo flag (hb0 .. hb2 beside mb0 .. mb2, then the multipliers' latency) picks lo's bytes, block
// and scale. A row's pass 1 takes twice its cycles, the TMEM reads and pass 0 of a second QACT
// fewer. otpu_pkg's QACT footprint stays KB blocks: HL needs the wait entries (SEQ_WC), which
// cover lo's blocks. QHL = 0 is the quantizer without it.
//
// SWW > 1 (QUANT_SWW; timing only): DRAM port A is SWW words wide (a_wdata 32 SWW bits, a_be 4
// SWW; a_addr still a word's: the write's bytes are the aligned 4 SWW holding it, those a_be
// enables), and QST's pass 1 in rows mode with element stride 1 and a row 4 SWW aligned (GRP
// rows: always) reads and writes QSW = 4 SWW contiguous bytes a cycle: lanes 0 .. LANES - 1
// through the Q port, the rest through Q2 (QST reads no CSCALE: csb is then the row's address +
// LANES), whose data skips the prescale (QST has no factors: x * 1 * 1 is x for the writer's
// product) on a delay line of its latency, into the writer's lanes LANES .. QSW - 1. A word or a
// byte write (the scale rows, the other QST modes) is replicated over the port with its enables.
// SWW = 1 is the quantizer without it.
//
// QSTB = 1 (QUANT_QSTB; timing only, with the row buffer and SWW > 1 at 4 SWW = LANES x PW): QST
// rows in ROW mode whose pass 1 is QSW bytes a cycle (GRP rows; else element stride 1 and dst and
// the row stride 4 SWW aligned; not HALF, of up to XD x LANES elements) go through the row buffer as
// QACT's do: pass 0 pushes the row's chunks and goes on to the next row, and once a row's scale is
// out its pass 1 pops PW chunks a cycle into the writer, QSW bytes to the SW port. The row's
// scale word goes to the writer first, in the cycle its pass 1 starts (sb0 .. sb2 beside mb0 ..
// mb2, then the writer's ss), so the port takes the scale row and the bytes in the writes' order.
// A buffered row's byte and scale addresses (badr, brs, sbase; GRP gb, grb, gln, gar) step with
// its pops, not with the reader. An unbuffered row waits for its scale before pass 1 (~67 cycles
// a 128-element row at SWW 4, 11-04); a buffered one takes pass 0's G / LANES cycles. QSTB = 0
// is the quantizer without it.
//
// QSTHL = 1 (with QSTB and SPLIT; Config.QSTHL, CAPS2 bit22): QST HL (w5 bit31; GRP ROW rows, not
// COL, HALF or LO; w5[19:0] srs) writes a row's hi part and, from the same read, its lo part
// HLD = w5[30:20] KiB after it (bytes and scale words): its buffered pass 1 writes hi's scale
// word, then lo's (sbh), then pops each of its chunks twice, hi then lo, as QACT HL's (p1h, the
// second pop advancing), the lo flag (hb0 .. hb2) picking the lanes' lo bytes and the scale S *
// 2^-8, lo's chunks at gb + HLD folded as their own (glb: the channel hash) and lo's scale word
// HLD after hi's (mb_baddr); HLD a multiple of P = WG x D. A row takes 2 + 2 G /
// QSW cycles of pass 1 (18 at 128 elements, SWW 4) beside pass 0's G / LANES (16). QSTHL = 0 is
// the quantizer without it.
//
// QONCE = 1 (Config.QONCE, CAPS2 bit24; docs/isa.md "Quantize once"): QACT flag ST (bit6) also
// stores each row's bytes and records in DRAM, through the SW port (4 SWW = LANES x PW): row r at
// R[rc] + w3 + r w7 (srs then in w6), its KB x D bytes as ACT RAM gets them, then its KB records of
// 8 bytes ({da3..da0, S}, block k's at KB x D + 8 k). The bytes go out with their ACT RAM write,
// in the same cycle (the port's write as QST's: the grant waits for the port); the records
// through a FIFO (QREC deep), popped in cycles without a byte write, or, when the FIFO is full
// and a block's first write (a push) waits, while a_hold holds the writer (the slice withholds
// the grant: the pop needs the port alone, a_rdy). Not with HL. The QACT completes once the FIFO
// is empty and its writes are acknowledged (wr_idle), as QST does. act_wv: a QACT write is at the
// write stage (the DMA's LDA does not write ACT RAM then: otpu_dma act_busy). QONCE = 0 is the
// quantizer without it.

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
    amax_q <= amax; itag_q <= itag;
  end
  always_ff @(posedge clk) if (rst) iv_q <= 1'b0; else if (en) iv_q <= iv;
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
  // delay lines into the multiplier operands end in a flip-flop (otpu_delay; no reset: data under
  // the scale unit's valid, u_v). The ISA's y*t + (-0) is y*t for every product (no fp_mul result
  // is changed by adding -0), so u_y is a plain multiply, not padded: its output register feeds
  // the next multipliers directly (no SRL in front of their DSPs), and k1 is delayed by the step's
  // SL + LM to meet it.
  f32_t y [4], t [2], k1 [3], yd [3], e;
  assign y[0] = y0;
  assign k1[0] = kx;
  for (genvar i = 0; i < 2; i++) begin : g_it
    otpu_fmadd #(.LM(LM), .LA(LA)) u_t (.clk, .en, .a(k1[i]), .b(y[i]), .c(F_TWO), .y(t[i]));
    otpu_delay #(.W(32), .N(SL)) u_yd (.clk, .en, .d(y[i]), .q(yd[i]));
    otpu_delay #(.W(32), .N(P)) u_k (.clk, .en, .d(k1[i]), .q(k1[i + 1]));
    otpu_fmul #(.LAT(LM)) u_y (.clk, .en, .a(yd[i]), .b(t[i]), .y(y[i + 1]));
  end
  otpu_fmadd #(.LM(LM), .LA(LA)) u_e (.clk, .en, .a(k1[2]), .b(y[2]), .c(F_ONE), .y(e));
  otpu_delay #(.W(32), .N(SL)) u_yd2 (.clk, .en, .d(y[2]), .q(yd[2]));
  otpu_fmadd #(.LM(LM), .LA(LA)) u_y3 (.clk, .en, .a(yd[2]), .b(e), .c(yd[2]), .y(y[3]));
  // inv = 127 * (y * 2^ek) (y in [0.5, 1]) is (127 * y) * 2^ek bit for bit: 127 * y is in
  // [63.5, 127] and the product is normal, so its rounding does not change with the scaling,
  // except that it overflows to +inf where the field passes 254. big: 127 * (+0) = +0, forced
  // on the product (bd delayed beside zd) instead of on the operand.
  f32_t invm, scm, scd;
  logic zd, bd, zd2, bd2;
  logic [8:0] ekd, ek2, fe;
  otpu_delay #(.W(11), .N(LAT - LM)) u_f (.clk, .en, .d({zero, big, ek}), .q({zd, bd, ekd}));
  otpu_fmul #(.LAT(LM)) u_inv (.clk, .en, .a(F_127), .b(y[3]), .y(invm));
  otpu_fmul #(.LAT(LM)) u_sc (.clk, .en, .a(ax), .b(F_INV127), .y(scm));
  otpu_delay #(.W(32), .N(LAT - LM)) u_scd (.clk, .en, .d(scm), .q(scd));
  otpu_delay #(.W(11), .N(LM)) u_z2 (.clk, .en, .d({zd, bd, ekd}), .q({zd2, bd2, ek2}));
  // the valid in reset registers (otpu_vdelay: a power-up one would set the row's scale), the
  // tag beside it
  otpu_vdelay #(.N(LAT + 1)) u_v (.clk, .rst, .en, .d(iv_q), .q(ov));
  otpu_delay #(.W(TW), .N(LAT + 1)) u_t (.clk, .en, .d(itag_q), .q(otag));
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
  parameter bit R1    = 1'b0,               // RSCALE's row factor read once a row (above)
  parameter int PW    = 1,                  // a buffered pass 1's chunks a cycle (above)
  parameter bit SPLIT = 1'b0,               // QACT / QST flag LO: the split's lo part (above)
  parameter bit QHL   = 1'b0,               // QACT flag HL: q and lo from one read (above)
  parameter int SWW   = 1,                  // DRAM port A's words: QST pass 1's 4 SWW bytes (above)
  parameter bit QSTB  = 1'b0,               // QST rows (ROW) through the row buffer (above)
  parameter bit QSTHL = 1'b0,               // QST HL: a GRP ROW QST's hi and lo from one read (above)
  parameter bit QONCE = 1'b0,               // QACT flag ST: the rows' bytes and records to DRAM too (above)
  parameter int WG    = 1,                  // QST GRP's rows a group (above; 1: no QST GRP)
  parameter bit CHASH = 1'b0,               // ... its chunks' lanes in the channel hash's order
  parameter int SID   = 0,
  parameter int AW    = 32,                 // DRAM byte address bits (otpu_seq AW): QST's dst, sdst
  parameter int WA    = waw(AW)             // the SW port's word address bits (derived)
) (
  input  logic                    clk,
  input  logic                    rst,
  input  logic                    start,
  input  cmd_t                    cmd,
  output logic                    rdy,
  output logic                    done,
  input  logic                    gnt,        // TMEM grant: hold everything when low
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
  output logic [LANES*PW-1:0]     act_we,     // (LANES x PW lanes)
  output logic [7:0]              act_row,
  output logic [31:0]             act_idx,
  output logic [LANES*PW-1:0][7:0] act_data,
  output logic                    act_dup,    // QACT DUP: also rows act_row + act_off
  output logic [7:0]              act_off,
  output logic                    asc_we,
  output logic [7:0]              asc_row,
  output logic [15:0]             asc_blk,
  output logic [63:0]             asc_data,   // {da3, da2, da1, da0, S}
  // DRAM port A (QST writes). a_want: a write is waiting (before the grant); the slice withholds
  // the grant while the port cannot take it. QST completes once its writes are acknowledged.
  output logic                    a_want,
  output logic                    a_hold,     // QONCE: hold the writer for a record (above)
  input  logic                    a_rdy,      // QONCE: the port takes a write this cycle
  output logic                    act_wv,     // QONCE: a QACT write at the write stage (above)
  input  logic                    wr_idle,
  output logic                    a_req,
  output logic                    a_we,
  output logic [WA-1:0]           a_addr,
  output logic [32*SWW-1:0]       a_wdata,    // (SWW > 1: the aligned 4 SWW bytes holding a_addr)
  output logic [4*SWW-1:0]        a_be,
  // profiling: a cycle after a QACT ends, its cycles frozen by the TMEM grant
  output logic                    pf_u,
  output logic [31:0]             pf_frz
);
  localparam int LM = 2, LA = 4;
  localparam int QL = 2 * LM + 1;           // prescale latency (with its input register)
  localparam int DL = $clog2(D);
  localparam int QG = D / 4;                // QACT group (sub-block) elements
  localparam int QGL = $clog2(QG);
  localparam int QN = (LANES > QG) ? LANES / QG : 1;   // QACT groups a chunk holds
  localparam int PL = LANES / QN;           // a chunk's lanes per QACT group
  localparam int GWN = GR / 4;              // group words (four group maxes each)
  localparam int GA = $clog2(GWN);
  localparam int NB = 4;                    // row banks: group maxes and scales of rows in flight
  localparam int XW = (XD > 1) ? $clog2(XD) : 1;
  localparam int QW = LANES * PW;           // the writer's lanes
  localparam int PWB = $clog2(PW);          // (0 at PW 1)
  localparam int SL = $clog2(SWW);          // (0 at SWW 1)
  localparam int QSW = 4 * SWW;             // SWW > 1: QST pass 1's bytes a read (above)
  initial if (XD == 1 || (XD & (XD - 1)) != 0) $fatal(1, "otpu_quant: XD 0 or a power of two >= 2");
  initial if (PW < 1 || (PW & (PW - 1)) != 0 || QW > D || (PW > 1 && XD > 0 && XD < 2 * PW))
    $fatal(1, "otpu_quant: PW a power of two, LANES x PW <= D, XD 0 or >= 2 x PW");
  initial if (D % LANES != 0 || (D & (D - 1)) != 0)
    $fatal(1, "otpu_quant: D a power of two, a multiple of LANES");
  initial if (LANES < 4) $fatal(1, "otpu_quant: QST word writes need LANES >= 4");
  initial if ((LANES & (LANES - 1)) != 0) $fatal(1, "otpu_quant: LANES must be a power of two");
  initial if (GR % 4 != 0 || (GWN & (GWN - 1)) != 0) $fatal(1, "otpu_quant: GR 4 x a power of two");
  initial if (SWW < 1 || (SWW & (SWW - 1)) != 0 ||
              (SWW > 1 && (QSW > 2 * LANES || QSW > QW || QSW >= D)))
    $fatal(1, "otpu_quant: SWW a power of two; above 1, 4 SWW <= 2 LANES, <= LANES x PW, < D");
  initial if (QHL && (!SPLIT || XD * LANES < 8192))
    $fatal(1, "otpu_quant: QHL needs SPLIT and the row buffer for QHL_ELEMS 8192 (XD x LANES)");
  initial if (QSTHL && (!QSTB || !SPLIT || XD * LANES < 8192))
    $fatal(1, "otpu_quant: QSTHL needs QSTB, SPLIT and XD x LANES >= QHL_ELEMS 8192");
  initial if (QSTB && (XD == 0 || SWW == 1 || QSW != QW))
    $fatal(1, "otpu_quant: QSTB needs the row buffer and SWW > 1 with 4 SWW = LANES x PW");
  initial if (QONCE && (SWW == 1 || QSW != QW))
    $fatal(1, "otpu_quant: QONCE needs SWW > 1 with 4 SWW = LANES x PW");

  wire en = gnt;

  // ------------------------------------------------------------------ command
  logic        busy, is_st, rowm, csf, rsf, ackw, dup;
  logic        lom;                        // SPLIT: the instruction's flag LO
  logic        stm;                        // QONCE: QACT's flag ST
  logic        hlm;                        // HL: the instruction's flag HL (QSTHL: QST's w5 bit31)
  localparam bit HLA = QHL || QSTHL;         // (either HL: a buffered chunk popped twice)
  logic        sbh;                        // QSTHL: the row's lo scale word is next (hi's done)
  logic [AW-1:0] hldr;                       // QSTHL: lo's offset, HLD KiB
  logic [31:0] abh;                          // HL: lo's first block (ab + KB) and its byte
  logic [15:0] abk;                          //   index (ab + KB) x D
  logic        bm;                           // QACT (QSTB: QST rows) through the row buffer
  wire         qbm = QSTB && bm;             // (the reader's address steps' guard: none at QSTB 0)
  logic [31:0] csb, rsb, srs, drs, es;
  logic [15:0] rows, KB;
  logic [7:0]  ab;
  logic [31:0] G;                            // a row's elements
  logic [31:0] GW;                           // ... read in pass 1 (QST HALF: the first half)
  logic [AW-1:0] sbase;                      // QST: the row's scale row (the steps sstep, es,
  logic [31:0] sstep;                        //   drs, gkp zero-extended); the next one's step
  logic [15:0] swn;                          // QST: a scale row's words (GRP rows: S, d bytes)
  // QST GRP (WG > 1, above): grp, gcol (COL); gb the chunk's address (P-aligned), grb the row's
  // first (rows: its group's chunk 0; COL: dst + (n / D) P), gkp KB x P, gar the row's scale
  // area (rows), gln the row's lane n % R (rows), gno its byte n % D (COL), gsp the scale
  // phase's write address (rows); gi the start's products in flight
  localparam int LG = $clog2(WG);
  localparam int LGW = (LG > 0) ? LG : 1;
  localparam int LP = LG + DL;
  logic          grp, gcol;
  logic [AW-1:0] gb, grb, gar, gsp;
  logic [31:0]   gkp;
  logic [LGW-1:0] gln;
  logic [DL-1:0] gno;
  logic [1:0]    gi;
  // the start's products (n0 / R) x KB P and (n0 / R) x es: operands registered at the start,
  // two product registers (block multipliers: their pipeline registers)
  logic [31:0]   gm_n, gm_ko, gm_eo, gm_k1, gm_e1, gm_k, gm_e;
  always_ff @(posedge clk) begin
    gm_k1 <= gm_n * gm_ko;
    gm_e1 <= gm_n * gm_eo;
    gm_k <= gm_k1;
    gm_e <= gm_e1;
  end
  // cycles an instruction was frozen by the TMEM grant, for the profiler: the condition is
  // registered (st_c) and summed a cycle late, off the grant path; the count is st_frz + st_c
  // (a start clears it: the unit was idle the cycle before, st_c is 0)
  logic [31:0] st_frz;
  logic        st_c;
  assign rdy = !busy;

  // ------------------------------------------------------------------ reader
  // walks rows r and their elements e, twice (pass 0: the maxes, pass 1: quantize)
  logic [15:0] r;
  logic [31:0] e;                            // element in the row
  logic [31:0] row_src;                      // TMEM address of the row
  logic [AW-1:0] badr, brs;                  // QST byte address dst + r*drs + e*es (mod 2^AW)
                                             // at (r, e) / at e = 0
  logic        pass;
  logic        wide;                         // QST pass 1: four contiguous bytes (a word) a cycle
  logic        wsw;                          // ... SWW > 1: QSW contiguous bytes a cycle (above)
  logic        rd_wait;                      // waiting for the row's scale (and scale row)
  logic        rd_done;
  logic        sw_ph;                        // QST: the scale row's writes
  logic [15:0] sw_i;                         // ... its word
  // elements per read: QST pass 1 one byte (or one word) a cycle, everything else LANES
  wire [31:0] ew = (is_st && pass) ? (wsw ? 32'(QSW) : wide ? 32'd4 : 32'd1) : 32'(LANES);
  wire [31:0] bstep = wsw ? 32'(QSW) : wide ? 32'd4 : es;   // QST pass 1: badr per read
  wire [31:0] Gr = pass ? GW : G;            // the elements of this pass
  wire rlast = (e + ew >= Gr);
  // the elements left in the row, Gr - e, kept alongside e (set wherever e or the pass moves):
  // lane l reads while l < ew and l < rem, a compare with a small constant instead of the adder
  // and compare of e + l < Gr, so the read enables and TMEM's block RAM enables do not follow two
  // carry chains (133.33 MHz, 110ec6d: e -> e + l -> < Gr -> the Q2 port's lane and bank enables
  // -> block RAM ENB, 12 levels, +0.258 ns)
  logic [31:0] rem;
  logic [LANES-1:0] rmask, rmask2;           // (rmask2: wsw's lanes LANES + l, through Q2)
  always_comb for (int l = 0; l < LANES; l++) begin
    rmask[l] = 32'(l) < ew && 32'(l) < rem;
    rmask2[l] = SWW > 1 && wsw && 32'(LANES + l) < ew && 32'(LANES + l) < rem;
  end
  // the row buffer: pass 1's row r1 (popping while p1a), its element, the FIFO's pointers and
  // its entries taken by issued reads (credits)
  logic        p1a;
  logic        p1h;                        // HL: the chunk's lo pop is next (its q pop done)
  wire         p1x = p1a && !(HLA && hlm && !p1h);   // a pop that advances (HL: lo's)
  logic [15:0] r1;
  logic [31:0] e1c;
  logic [XW-1:0] xw, xr;
  logic [XW:0] xocc;                         // <= XD: below XD while its top bit is clear
  logic [2:0]  dr;                           // r - r1 (<= NB)
  // buffered: a read needs a free FIFO entry, and row r's banks free (pass 1 of row r - NB done);
  // one LUT on the read enables' path
  wire xok = !bm || (!xocc[XW] && !dr[2]);
  wire riss = busy && !ackw && !rd_done && !rd_wait && !sw_ph && xok;

  typedef struct packed {
    logic             v;
    logic             pass;
    logic [31:0]      e;         // element index within the row (of lane 0)
    logic             last;      // last chunk of the pass
    logic             glast;     // ... of the instruction
    logic [15:0]      row;
    logic [LANES-1:0] mask;
  } rm_t;
  // m0a: the chunk whose read request is in the request register (TMEM reads it this cycle);
  // m0: the chunk whose data arrives now; mp: after the prescale
  rm_t m0a, m0, mp;
  // (the chunk's valid alone resets, as mh's and mc's: the other fields are data under it)
  always_ff @(posedge clk) begin
    if (en) m0 <= m0a;
    if (rst) m0.v <= 1'b0;
  end
  // SWW > 1: a chunk's wsw side {ws, mask2} beside m0a, m0 and mp (QST wsw: QSW bytes, lanes
  // LANES.. in mask2), outside rm_t: at SWW 1 nothing reads w2a, and m0's delay is as before
  logic [LANES:0] w2a, w2p;
  // buffered pass 1: a chunk popped (its data and group word arriving), then +1, +2 (the writer's
  // stage, like mp); the offsets' stages take their chunk from mb0 or m0
  rm_t mb0, mb1, mb2;
  logic hb0, hb1, hb2;                       // HL: ... lo's pop (outside rm_t: m0's delay as before)
  logic sb0, sb1, sb2;                       // QSTB: ... the row's scale word (its pass 1's start)
  rm_t m1s;
  assign m1s = bm ? mb0 : m0;

  // TMEM's input stage (release r2, docs/tmem_nocopy.md 7): the three read requests (t_*_c) go
  // to TMEM from flip-flops, a cycle after they are formed, advancing with the enable; the
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
    else if (en) begin
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
        if (rmask2[l]) t_ren2_c[l] = 1'b1;
      end
      if (rsf && (!R1 || e == 0)) t_ren3_c = 1'b1;
    end
  end
`ifndef SYNTHESIS
  // the checks start once the reset has held two edges (registers start arbitrary, and so does a
  // unit's reset register until the slice's reset pipeline fills: otpu_slice)
  bit rst_p, rst_seen;
  initial begin rst_p = 1'b0; rst_seen = 1'b0; end
  always @(posedge clk) begin rst_p <= rst; if (rst && rst_p) rst_seen <= 1'b1; end
  always @(posedge clk) begin
    if (!rst && rst_seen && busy && !ackw && rem != Gr - e)
      $fatal(1, "otpu_quant: rem %0d is not Gr %0d - e %0d", rem, Gr, e);
  end
`endif

  // ------------------------------------------------------------------ prescale
  // the TMEM read data is registered first (no path from the block RAMs into the multipliers'
  // DSP inputs in one cycle); m0 and the byte address are delayed to match (QL)
  f32_t xp [LANES];
  f32_t r3;
  always_ff @(posedge clk)
    if (en) r3 <= !rsf ? F_ONE : (!R1 || (m0.v && m0.e == 0)) ? t_rdata3 : r3;
  for (genvar l = 0; l < LANES; l++) begin : g_pre
    f32_t v1, cd, x1, x2;
    always_ff @(posedge clk) if (en) begin
      x1 <= t_rdata[l];
      x2 <= csf ? t_rdata2[l] : F_ONE;
    end
    otpu_fmul #(.LAT(LM)) u_r (.clk, .en, .a(x1), .b(r3), .y(v1));
    otpu_delay #(.W(32), .N(LM)) u_c (.clk, .en, .d(x2), .q(cd));
    otpu_fmul #(.LAT(LM)) u_c2 (.clk, .en, .a(v1), .b(cd), .y(xp[l]));
  end
  // the chunk's valid in reset registers (otpu_vdelay), as m0's, the rest under it
  localparam int MB = $bits(rm_t);
  logic [MB-1:0] m0p, mpp;
  assign m0p = m0;
  otpu_vdelay #(.N(QL)) u_mpv (.clk, .rst, .en, .d(m0p[MB-1]), .q(mpp[MB-1]));
  otpu_delay #(.W(MB - 1), .N(QL)) u_mp (.clk, .en, .d(m0p[MB-2:0]), .q(mpp[MB-2:0]));
  assign mp = mpp;
  if (SWW > 1) begin : g_w2
    logic [LANES:0] w2;
    always_ff @(posedge clk) if (rst) w2 <= '0; else if (en) w2 <= w2a;
    otpu_delay #(.W(LANES + 1), .N(QL)) u_w2 (.clk, .en, .d(w2), .q(w2p));
  end else begin : g_nw2
    assign w2p = '0;
  end
  // SWW > 1: wsw's lanes LANES + l (Q2's data), delayed to the prescale's output (no factors)
  f32_t xq [LANES];
  if (SWW > 1) begin : g_xq
    for (genvar l = 0; l < LANES; l++) begin : g_l
      otpu_delay #(.W(32), .N(QL)) u_x (.clk, .en, .d(t_rdata2[l]), .q(xq[l]));
    end
  end else begin : g_nxq
    always_comb for (int l = 0; l < LANES; l++) xq[l] = F_ZERO;
  end

  // QST GRP's byte address: {gb above P, lane ^ fold(gb / P), byte} (rows: lane n % R, byte e %
  // D; COL: lane e % R, byte n % D; QSTB's buffered rows: e1c, the pop's element). QSTHL: a lo
  // pop's chunk is gb + HLD (glb, registered at its hi pop: gb steps only at lo's), folded as
  // its own chunk
  logic [AW-1:0] gbadr, glb;
  wire [31:0] ge = qbm ? e1c : e;
  if (QSTHL) begin : g_glb
    always_ff @(posedge clk) if (en) glb <= gb + hldr;
  end else begin : g_nglb
    assign glb = '0;
  end
  always_comb begin
    logic [LGW-1:0] h, l;
    logic [AW-1:0] c;
    c = (QSTHL && hlm && p1a && p1h) ? glb : gb;
    h = '0;
    if (CHASH && LG > 0)
      for (int b = LP; b < AW; b++) h[(b - LP) % LGW] = h[(b - LP) % LGW] ^ c[b];
    l = gcol ? LGW'(e) : gln;
    gbadr = c | ((LG > 0) ? 32'(l ^ h) << DL : 32'd0) | 32'(gcol ? gno : ge[DL-1:0]);
  end
  // QST's byte address dst + row*drs + e*es: the reader's running sum badr, taken in the issue
  // cycle like m0a and delayed with the chunk (m0a, m0: two registers) through the prescale (QL)
  logic [AW-1:0] mp_baddr;
  otpu_delay #(.W(AW), .N(QL + 2)) u_ba (.clk, .en, .d(grp ? gbadr : badr), .q(mp_baddr));
  // QSTB: a pop's byte address (its pass 1 row's badr / gbadr), or at the row's pass 1 start its
  // scale word's (GRP rows: S at 4 (n % R) in the group's area; else sbase), taken in the pop
  // cycle like mb0 and delayed with it to mb2 (three registers); QSTHL: lo's scale word + HLD
  // (its pops: gbadr's glb)
  logic [AW-1:0] mb_baddr;
  if (QSTB) begin : g_qb
    otpu_delay #(.W(AW), .N(3)) u_bb (.clk, .en,
      .d(p1a ? (grp ? gbadr : badr)
             : (grp ? gar + AW'({gln, 2'b00}) : sbase) + ((QSTHL && hlm && sbh) ? hldr : AW'(0))),
      .q(mb_baddr));
  end else begin : g_nqb
    assign mb_baddr = '0;
  end

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
  always_ff @(posedge clk) if (en) begin
    f32_t v [LANES];
    // xp is flushed (otpu_fmul), so clearing the sign is fabs
    for (int l = 0; l < LANES; l++) v[l] = mp.mask[l] ? {1'b0, xp[l][30:0]} : F_ZERO;
    for (int l = 0; l < HL; l++) ca[l] <= mgt(v[2*l + 1], v[2*l]) ? v[2*l + 1] : v[2*l];
  end
  // the chunks' valids reset, their other fields are data: every use of one is under its .v
  // (gw_go and sq_iv take mc.v, the maxes fold under it), so the reset loads 2 flip-flops, not
  // 2 x $bits(rm_t)
  always_ff @(posedge clk) begin
    if (en) begin
      mh <= mp;
      mc <= mh;
    end
    if (rst) begin
      mh.v <= 1'b0;
      mc.v <= 1'b0;
    end
  end
  always_ff @(posedge clk) if (en) begin
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
  wire gw_go = mc.v && !mc.pass && !rowm && glst;
  always_ff @(posedge clk)                   // the write enables reset (control), the rest data
    if (rst) gw_we <= '0;
    else if (en) begin
      gw_we <= '0;
      if (gw_go) begin
        if (is_st || QN == 1) gw_we[gix[1:0]] <= 1'b1;
        else for (int p = 0; p < QN; p++) gw_we[2'(gix[1:0] + 2'(p))] <= 1'b1;
      end
    end
  always_ff @(posedge clk) if (en) begin
    if (gw_go) begin
      if (is_st || QN == 1) begin
        gw_d[gix[1:0]] <= gnew[30:0];
      end else begin
        for (int p = 0; p < QN; p++) gw_d[2'(gix[1:0] + 2'(p))] <= pmx[p][30:0];
      end
      gw_a <= {mc.row[1:0], GA'(gix >> 2)};
    end
  end
  // the group RAMs' reads: the chunk's group word at a pass 1 read (QACT: the block's four
  // groups; QST: the word of its block; buffered: at a pop), the next scale-row word in QST's
  // scale row phase; NB banks of GWN words, row r in bank r % NB
  // grd: the group word read at the chunk's issue (the group RAMs are not behind TMEM's input
  // stage), the cycle its read request is in the request register; grd_d: a cycle later, with
  // m0 (the TMEM data). A buffered pass 1 reads it at the pop, with mb0 (no TMEM read).
  logic [3:0][30:0] grd, grd_d;
  always_ff @(posedge clk) if (en) grd_d <= grd;
  // (GRP rows: a d byte a cycle, the word of group sw_i - 1 a cycle ahead)
  wire [GA+1:0] gr_a = sw_ph ? {r[1:0], GA'((grp && !gcol) ? sw_i >> 2 : sw_i)}
                     : is_st ? {r[1:0], GA'(e >> (DL + 2))}
                     : bm ? {r1[1:0], GA'(e1c >> DL)} : {r[1:0], GA'(e >> DL)};
  for (genvar s = 0; s < 4; s++) begin : g_gr
    otpu_ram_1r1w #(.W(31), .N(NB * GWN), .STYLE("block")) u_g (
      .clk, .we(en && gw_we[s]), .wa(gw_a), .wd(gw_d[s]), .re(en), .ra(gr_a), .rd(grd[s]));
  end

  // scale unit (its amax is don't-care while sq_iv is low: sc/inv are read only under sq_ov)
  logic        sq_iv, sq_ov;
  logic [1:0]  sq_ot;
  f32_t        sq_amax, sq_sc, sq_inv;
  assign sq_iv   = mc.v && mc.last && !mc.pass;
  assign sq_amax = nmax;
  otpu_qscale #(.LM(LM), .LA(LA), .TW(2)) u_sc (.clk, .rst, .en, .iv(sq_iv), .itag(mc.row[1:0]),
    .amax(sq_amax), .ov(sq_ov), .otag(sq_ot), .sc(sq_sc), .inv(sq_inv));
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
  always @(posedge clk) if (!rst && rst_seen && en)
    for (int s = 0; s < 4; s++)
      if (dval(bm ? grd[s] : grd_d[s], amx_q[m1s.row[1:0]], rowm)
          != dval_ref(bm ? grd[s] : grd_d[s], amx_q[m1s.row[1:0]], rowm))
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
  // the read's group word at m0, its four d a stage later, each lane's inv * 2^d and the
  // block's record bytes the next, delayed to the prescale's output
  logic [3:0][4:0] dq;
  logic [31:0]     e1;
  logic [1:0]      b1;                       // the chunk's row bank
  always_ff @(posedge clk) if (en) begin
    for (int s = 0; s < 4; s++) dq[s] <= dval(bm ? grd[s] : grd_d[s], amx_q[m1s.row[1:0]], rowm);
    e1 <= m1s.e;
    b1 <= m1s.row[1:0];
  end
  f32_t        invl [QW], invp [LANES];      // (invp: an unbuffered pass 1's, LANES)
  logic [31:0] dal, dap;
  always_ff @(posedge clk) if (en) begin
    for (int l = 0; l < QW; l++) begin
      logic [31:0] el;
      el = e1 + 32'(l);
      invl[l] <= p2mul(inv_q[b1], dq[is_st ? el[DL+1:DL] : el[QGL+1:QGL]]);
    end
    dal <= {3'b0, dq[3], 3'b0, dq[2], 3'b0, dq[1], 3'b0, dq[0]};
  end
  for (genvar l = 0; l < LANES; l++) begin : g_inv
    otpu_delay #(.W(32), .N(QL - 2)) u_i (.clk, .en, .d(invl[l]), .q(invp[l]));
  end
  otpu_delay #(.W(32), .N(QL - 2)) u_da (.clk, .en, .d(dal), .q(dap));
`ifndef SYNTHESIS
  // p2mul is fp_mul(inv, 2^d) for every inv the scale unit gives (inv_w: the row banks it gave
  // one since the reset)
  logic [NB-1:0] inv_w;
  always @(posedge clk) if (!rst && rst_seen && en)
    for (int l = 0; l < QW; l++) begin
      logic [31:0] el;
      el = e1 + 32'(l);
      if (inv_w[b1] && p2mul(inv_q[b1], dq[is_st ? el[DL+1:DL] : el[QGL+1:QGL]]) !=
          fp_mul(inv_q[b1], {1'b0, 8'(8'd127 + 8'(dq[is_st ? el[DL+1:DL] : el[QGL+1:QGL]])), 23'd0}))
        $fatal(1, "otpu_quant: inv %h * 2^d is not fp_mul's", inv_q[b1]);
    end
`endif

  // ------------------------------------------------------------------ writer
  typedef struct packed {
    logic             v;
    logic [QW-1:0]    mask;
    logic [7:0]       row;
    logic [31:0]      idx;       // ACT byte index
    logic             sw;        // write the scale record (first chunk of a block)
    logic [15:0]      blk;
    logic [31:0]      da;        // the block's offsets {da3, da2, da1, da0}
    logic             st;        // QST byte
    logic             ws;        // ... QSW bytes (wsw)
    logic             ss;        // ... QSTB: the row's scale word sc instead (at baddr)
    logic [AW-1:0]    baddr;     // QST byte address
    logic             fin;       // last write of the instruction
    logic [31:0]      sc;        // the row's S (the scale record)
  } wm_t;
  wm_t   wq, wqd;                            // at the multiplier input / at the write stage
  f32_t  wx [QW], winv [QW];
  // buffered pass 1: the popped chunks' data, delayed to the offsets' output (mb2)
  f32_t  xd [QW], xb1 [QW], xb2 [QW];
  always_ff @(posedge clk) if (en) begin
    xb1 <= xd;
    xb2 <= xb1;
  end
  always_ff @(posedge clk) begin
    if (en) begin
      mb1 <= mb0;
      mb2 <= mb1;
      hb1 <= HLA && hb0;
      hb2 <= HLA && hb1;
      sb1 <= QSTB && sb0;
      sb2 <= QSTB && sb1;
    end
    if (rst) begin                           // the valids (the writer's selection: m), HL's and
      mb1.v <= 1'b0;                         // QSTB's flags
      mb2.v <= 1'b0;
      hb1 <= 1'b0;
      hb2 <= 1'b0;
      sb1 <= 1'b0;
      sb2 <= 1'b0;
    end
  end
  // the row buffer: pass 0's prescaled chunks (buffered QACT), popped by pass 1 in order, PW a
  // pop (the read registered: the popped chunks' data arrives with mb0); PW banks of XD / PW
  // chunks, chunk i in bank i % PW, word i / PW (a pop: one word of every bank)
  wire xpush = bm && mp.v && !mp.pass;
  if (XD > 0) begin : g_xb
    for (genvar b = 0; b < PW; b++) begin : g_bk
      logic [LANES*32-1:0] xwd, xrd;
      always_comb for (int l = 0; l < LANES; l++) xwd[32*l +: 32] = xp[l];
      otpu_ram_1r1w #(.W(LANES * 32), .N(XD / PW), .STYLE("block")) u_xb (
        .clk, .we(en && xpush && 32'(xw) % PW == b), .wa(xw[XW-1:PWB]), .wd(xwd), .re(en),
        .ra(xr[XW-1:PWB]), .rd(xrd));
      always_comb for (int l = 0; l < LANES; l++) xd[b * LANES + l] = xrd[32*l +: 32];
    end
  end else begin : g_nxb
    always_comb for (int l = 0; l < QW; l++) xd[l] = F_ZERO;
  end
`ifndef SYNTHESIS
  // the FIFO: a pop takes a chunk pushed before it, a push never overruns (the credits)
  int unsigned xs_n;
  always @(posedge clk)
    if (rst || start) xs_n = 0;
    else if (rst_seen && busy && en) begin
      if (p1a && xs_n < PW) $fatal(1, "otpu_quant: row buffer popped empty");
      xs_n = xs_n + (xpush ? 1 : 0) - (p1x ? PW : 0);
      if (xs_n > XD) $fatal(1, "otpu_quant: row buffer overrun");
    end
`endif
  // (a buffered chunk is wsw only as a QSTB row's: QSW = QW bytes a pop)
  wire m_ws = SWW > 1 && (mb2.v ? QSTB && is_st : w2p[LANES]);
  wire m_hl = HLA && mb2.v && hb2;                    // HL: lo's write (blocks ab + KB ..)
  wire m_ss = QSTB && mb2.v && sb2;                   // QSTB: the row's scale word
  always_comb begin
    rm_t   m;
    wq = '0;
    for (int l = 0; l < QW; l++) begin
      wx[l] = F_ZERO;
      winv[l] = F_ZERO;
    end
    m = mb2.v ? mb2 : mp;                    // (mb2 only when buffered, mp.pass only when not)
    if (mb2.v || (mp.v && mp.pass)) begin
      wq.v = 1'b1;
      // a pop's PW chunks; an unbuffered chunk's LANES lanes go to every lane group, its mask
      // to its own (QACT: its place in the LANES x PW-aligned write; QST: group 0)
      // (wsw: QSW lanes, Q2's from LANES on)
      if (mb2.v) wq.mask = '1;
      else if (m_ws)
        for (int l = 0; l < QW; l++) wq.mask[l] = (l < LANES) ? m.mask[l % LANES]
                                                  : (l < 2 * LANES) && w2p[l % LANES];
      else
        for (int g = 0; g < PW; g++)
          if (is_st ? (g == 0) : (32'(g) == (m.e / LANES) % PW))
            wq.mask[g*LANES +: LANES] = m.mask;
      for (int l = 0; l < QW; l++) begin
        wx[l] = mb2.v ? xb2[l] : (m_ws && l >= LANES) ? xq[l % LANES] : xp[l % LANES];
        winv[l] = mb2.v ? invl[l] : invp[l % LANES];
      end
      wq.row = m.row[7:0];
      wq.idx = ((m_hl ? abh : 32'(ab) * D) + m.e) & ~32'(QW - 1);   // QACT: e steps by LANES
      wq.sw = !is_st && (m.e[DL-1:0] == 0);                          // (QST: unused)
      wq.blk = (m_hl ? abk : 16'(ab)) + 16'(m.e >> DL);
      wq.da = mb2.v ? dal : dap;
      wq.st = is_st;
      wq.ws = m_ws;
      wq.ss = m_ss;
      wq.baddr = (QSTB && mb2.v) ? mb_baddr : mp_baddr;
      wq.fin = m.glast;
      wq.sc = (lom || m_hl) ? sc_lo(sc_q[m.row[1:0]]) : sc_q[m.row[1:0]];
    end
  end
  // the writer's selection registered in front of the multipliers
  wm_t   wq_r;
  f32_t  wx_r [QW], winv_r [QW];
  always_ff @(posedge clk) begin
    if (en) begin
      wq_r <= wq;
      wx_r <= wx;
      winv_r <= winv;
    end
    if (rst) wq_r.v <= 1'b0;                 // the write's valid; the rest is data under it
  end
  // HL: the write's lo flag, from wq_r's stage to the lanes' second stage (lm: LM + 1 later)
  logic hq_l;
  if (HLA) begin : g_hq
    logic [LM+1:0] hq;
    always_ff @(posedge clk) if (en) hq <= {hq[LM:0], wq.v && m_hl};
    assign hq_l = hq[LM+1];
  end else begin : g_nhq
    assign hq_l = 1'b0;
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
  logic [QW-1:0][7:0] qb;
  for (genvar l = 0; l < QW; l++) begin : g_q
    f32_t p;
    otpu_fmul #(.LAT(LM)) u_q (.clk, .en, .a(wx_r[l]), .b(winv_r[l]), .y(p));
    if (SPLIT) begin : g_lo
      lo_mid_t lm;
      always_ff @(posedge clk) if (en) begin
        lm <= qlo_s1(p);
        qb[l] <= (lom || hq_l) ? qlo_s2(lm) : qlo_q8(lm);
      end
    end else begin : g_q8
      q8_mid_t qm;
      always_ff @(posedge clk) if (en) begin
        qm <= qq8_s1(p);
        qb[l] <= q8_s2(qm);
      end
    end
  end
  // the write's valid in reset registers (otpu_vdelay), the rest data in otpu_delay (ends in a
  // flip-flop; no reset: every use of wqd's other fields is under wqd.v)
  localparam int WB = $bits(wm_t);
  logic [WB-1:0] wqp, wqdp;
  assign wqp = wq_r;
  otpu_vdelay #(.N(LM + 2)) u_wqv (.clk, .rst, .en, .d(wqp[WB-1]), .q(wqdp[WB-1]));
  otpu_delay #(.W(WB - 1), .N(LM + 2)) u_wq (.clk, .en, .d(wqp[WB-2:0]), .q(wqdp[WB-2:0]));
  assign wqd = wqdp;

  // QST's scale row: S, then (not ROW) the d words: word j holds groups 4j .. 4j+3, the group
  // RAMs' word j (read the cycle before), the bytes of groups past KB 0
  logic [31:0] dword;
  always_comb begin
    dword = '0;
    for (int s = 0; s < 4; s++)
      if (32'(sw_i) * 4 + 32'(s) < 32'(KB) + 4)        // groups 4(sw_i - 1) + s < KB
        dword[8*s +: 8] = {3'b0, dval(grd[s], amx_q[r[1:0]], 1'b0)};
  end
  // QST GRP rows: the d byte of group sw_i - 1
  logic [7:0] gdb;
  always_comb gdb = {3'b0, dval(grd[2'(sw_i - 16'd1)], amx_q[r[1:0]], 1'b0)};
  // QST's S word (LO: S * 2^-8)
  f32_t scw;
  always_comb scw = lom ? sc_lo(sc_q[r[1:0]]) : sc_q[r[1:0]];
  // a DRAM write is waiting (independent of the grant, which the slice derives from it)
  assign a_want = (busy && !ackw && wqd.v && (wqd.st || (QONCE && stm))) || sw_ph;
  assign act_wv = QONCE && busy && !ackw && wqd.v && !wqd.st;

  // QONCE (ST): the bytes' DRAM row base (qrb: row qlr's, qrn: the next row's; qbase the write's)
  // and the record FIFO (QREC entries: a record's word address and {da, S}), pushed with a
  // block's first byte write, popped by the port alone (above)
  localparam int QREC = 32;
  localparam int QRL = $clog2(QREC);
  logic [AW-1:0]  qrb, qrn, qdrs, qkbd;
  logic [31:0]    qabd;
  logic [7:0]     qlr;
  logic [QRL:0]   qrw, qrr;
  logic [WA+63:0] q_ent, q_head;
  logic [32*SWW-1:0] q_rwd;                 // the head record's write: 8 of the port's 4 SWW bytes
  logic [4*SWW-1:0]  q_rbe;
  wire            q_rf_nz = QONCE && (qrw != qrr);
  wire [AW-1:0]   qbase = (wqd.row == qlr) ? qrb : qrn;
  wire            q_bw = QONCE && stm && busy && !ackw && wqd.v && !wqd.st;  // a byte write waits
  assign a_hold = q_bw && wqd.sw && (qrw - qrr) >= (QRL+1)'(QREC - 1);    // ... and a full FIFO
  wire            q_push = q_bw && wqd.sw && gnt;
  wire            q_pop = q_rf_nz && (!q_bw || a_hold) && a_rdy;
  if (QONCE) begin : g_qo
    (* ram_style = "distributed" *) logic [WA+63:0] qrf [QREC];
    always_comb
      q_ent = {WA'((qbase + qkbd + AW'({wqd.blk - 16'(ab), 3'b000})) >> 2), wqd.da, wqd.sc};
    always_ff @(posedge clk) if (q_push) qrf[qrw[QRL-1:0]] <= q_ent;
    assign q_head = qrf[qrr[QRL-1:0]];
    assign q_rwd = {(SWW / 2){q_head[63:0]}};
    assign q_rbe = (4*SWW)'(8'hFF) << (4 * (32'(q_head[64 +: SL]) & ~32'd1));
    always_ff @(posedge clk)
      if (rst) begin
        qrw <= '0;
        qrr <= '0;
      end else begin
        if (q_push) qrw <= qrw + 1'b1;
        if (q_pop) qrr <= qrr + 1'b1;
      end
    // the rows' bases: the next row's at its first byte write (rows in order, each written)
    always_ff @(posedge clk)
      if (start) begin
        qrb <= AW'(cmd_a(cmd, 3));
        qrn <= AW'(cmd_a(cmd, 3)) + AW'(cmd.w7);
        qdrs <= AW'(cmd.w7);
        qlr <= '0;
        qkbd <= AW'(32'(cmd.w2[31:16]) * D);
        qabd <= 32'(cmd.w2[15:8]) * D;
      end else if (q_bw && gnt && wqd.row != qlr) begin
        qlr <= wqd.row;
        qrb <= qrn;
        qrn <= qrn + qdrs;
      end
`ifndef SYNTHESIS
    always @(posedge clk)
      if (!rst && rst_seen && ((q_push && (qrw - qrr) == (QRL+1)'(QREC)) || (a_hold && gnt)))
        $fatal(1, "otpu_quant: QACT ST's record FIFO overrun (a_hold %0d, gnt %0d)", a_hold, gnt);
    always @(posedge clk)
      if (!rst && rst_seen && q_bw && gnt && wqd.row != qlr && wqd.row != qlr + 8'd1)
        $fatal(1, "otpu_quant: QACT ST's row %0d after row %0d", wqd.row, qlr);
`endif
  end else begin : g_nqo
    assign q_ent = '0;
    assign q_head = '0;
    assign q_rwd = '0;
    assign q_rbe = '0;
    always_comb begin
      qrw = '0; qrr = '0; qrb = '0; qrn = '0; qdrs = '0; qkbd = '0; qabd = '0; qlr = '0;
    end
  end

  assign act_dup = dup;
  assign act_off = rows[7:0];

  // (SWW > 1: a word's or a byte's place in the port's 4 SWW bytes, its address's low bits)
  logic [SL+1:0] wbo, sbo;
  logic [AW-1:0] sra;                        // the scale row's word address (bytes; not GRP)
  always_comb begin
    act_we = '0; act_row = '0; act_idx = '0; act_data = '0;
    asc_we = 1'b0; asc_row = '0; asc_blk = '0; asc_data = '0;
    a_req = 1'b0; a_we = 1'b0; a_addr = '0; a_wdata = '0; a_be = '0;
    wbo = wqd.baddr[SL+1:0];
    sra = sbase + {sw_i, 2'b00};
    sbo = sra[SL+1:0];
    if (busy && !ackw && wqd.v) begin
      if (wqd.st) begin
        a_req = 1'b1;
        a_we = 1'b1;
        a_addr = wqd.baddr >> 2;
        if (QSTB && wqd.ss) begin                 // a buffered row's S (LO: S * 2^-8)
          a_wdata = {SWW{wqd.sc}};
          a_be = (4*SWW)'(4'hF) << (4 * (wbo >> 2));
        end else if (SWW > 1 && wqd.ws) begin     // QSW bytes (wsw: the row is QSW aligned)
          for (int l = 0; l < QSW; l++) begin
            a_wdata[8*l +: 8] = qb[l];
            a_be[l] = wqd.mask[l];
          end
        end else if (wqd.mask[1]) begin           // a word (wide: the row is word aligned)
          a_wdata = {SWW{qb[3], qb[2], qb[1], qb[0]}};
          a_be = (4*SWW)'(4'hF) << (4 * (wbo >> 2));
        end else begin
          a_wdata = {QSW{qb[0]}};
          a_be = (4*SWW)'(1) << wbo;
        end
      end else begin
        act_row = wqd.row;
        // QACT writes are whole lane groups (idx LANES x PW aligned, a group's mask bits all set
        // or clear); saying so here folds the ACT RAM's byte rotator to a decode
        act_idx = wqd.idx & ~32'(QW - 1);
        for (int l = 0; l < QW; l++) begin
          act_we[l] = wqd.mask[(l / LANES) * LANES];
          act_data[l] = qb[l];
        end
        if (wqd.sw) begin
          asc_we = 1'b1;
          asc_row = wqd.row;
          asc_blk = wqd.blk;
          asc_data = {wqd.da, wqd.sc};
        end
        if (QONCE && stm) begin                   // ST: the same bytes to DRAM (QSW = QW)
          a_req = 1'b1;
          a_we = 1'b1;
          a_addr = WA'((qbase + AW'(wqd.idx - qabd)) >> 2);
          for (int l = 0; l < QSW; l++) begin
            a_wdata[8*l +: 8] = qb[l];
            a_be[l] = wqd.mask[(l / LANES) * LANES];
          end
        end
      end
    end
    if (sw_ph) begin
      a_req = 1'b1;
      a_we = 1'b1;
      if (grp && !gcol) begin                     // S, then a d byte a cycle
        a_addr = gsp >> 2;
        a_wdata = (sw_i == 0) ? {SWW{scw}} : {QSW{gdb}};
        a_be = (sw_i == 0) ? (4*SWW)'(4'hF) << (4 * (gsp[SL+1:0] >> 2))
                           : (4*SWW)'(1) << gsp[SL+1:0];
      end else begin
        a_addr = sra >> 2;
        a_wdata = {SWW{(sw_i == 0) ? scw : dword}};
        a_be = (4*SWW)'(4'hF) << (4 * (sbo >> 2));
      end
    end
    if (!gnt) begin
      act_we = '0; asc_we = 1'b0; a_req = 1'b0; a_we = 1'b0;
    end
    if (q_pop) begin                              // QONCE ST: a record (q_rw*)
      a_req = 1'b1;
      a_we = 1'b1;
      a_addr = q_head[64 +: WA];
      a_wdata = q_rwd;
      a_be = q_rbe;
    end
  end
`ifndef SYNTHESIS
  // the scale row goes out between a row's passes: no data write is in flight then
  always @(posedge clk)
    if (!rst && rst_seen && busy && sw_ph && wqd.v)
      $fatal(1, "otpu_quant: a QST data write during the scale row");
`endif

  // ------------------------------------------------------------------ sequencing
  // done is reported two cycles after the instruction's last ACT RAM write (done_w, then done)
  // and the ACT RAM lands the write two edges after done rises (three registers: otpu_actram).
  // The sequencer clears the dependency at the edge after done rises and releases the dependent
  // MM (urel) at the next, the one that lands the write, so the MM's first read of the block is
  // at least an edge after it landed; otpu_actram's simulation check stops on a consumed read of
  // a block still in flight
  logic done_w;
  always_ff @(posedge clk) done <= !rst && done_w;
  always_ff @(posedge clk) begin
    done_w <= 1'b0;
    pf_u <= 1'b0;
    st_c <= busy && !gnt;
    st_frz <= st_frz + 32'(st_c);
    if (rst) begin
      busy <= 1'b0;
      ackw <= 1'b0;
      sw_ph <= 1'b0;
      grp <= 1'b0;
      gi <= '0;
      m0a <= '0;
      w2a <= '0;
      mb0 <= '0;
      hb0 <= 1'b0;
      sb0 <= 1'b0;
      sbh <= 1'b0;
      p1a <= 1'b0;
      p1h <= 1'b0;
      bm <= 1'b0;
`ifndef SYNTHESIS
      inv_w <= '0;
`endif
    end else if (ackw) begin
      m0a <= '0;
      w2a <= '0;
      mb0 <= '0;
      sb0 <= 1'b0;
      hb0 <= 1'b0;
      if (wr_idle && !q_rf_nz) begin
        ackw <= 1'b0;
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
      // (QST at SWW > 1: Q2's reads, wsw's lanes LANES.., from the row's address + LANES)
      csb   <= (SWW > 1 && qst) ? cmd.w1 + 32'(LANES) : cmd.w4;
      rsf   <= !qst && cmd.flags[2];
      rsb   <= cmd.w5;
      dup   <= !qst && cmd.flags[3];
      lom   <= SPLIT && (qst ? cmd.flags[7] : cmd.flags[4]);
      stm   <= QONCE && !qst && cmd.flags[6];
      hlm   <= (QHL && !qst && cmd.flags[5]) ||
               (QSTHL && WG > 1 && qst && cmd.w5[31] && cmd.flags[2] && !cmd.flags[3]);
      hldr  <= QSTHL ? AW'({cmd.w5[30:20], 10'b0}) : '0;
      rows  <= nr;
      KB    <= kb;
      G     <= 32'(kb) * D;
      rem   <= 32'(kb) * D;
      if (qst) begin
        srs  <= (QSTHL && cmd.w5[31]) ? 32'(cmd.w5[19:0]) : cmd.w5;
        drs  <= cmd.w6;
        es   <= cmd.w7;
        // HALF (ROW mode): the scale is the whole row's, pass 1 writes its first half
        GW   <= (32'(kb) * D) >> ((cmd.flags[0] && cmd.flags[1]) ? 1 : 0);
        sstep <= cmd.flags[0] ? 32'd4 : 32'd4 + (nw << 2);
        swn  <= cmd.flags[0] ? 16'd1 : (WG > 1 && cmd.flags[2]) ? 16'd1 + kb : 16'(32'd1 + nw);
      end else begin
        ab   <= cmd.w2[15:8];
        srs  <= (QONCE && cmd.flags[6]) ? cmd.w6 : cmd.w3;     // (ST: w3 its DRAM rows)
        GW   <= 32'(kb) * D;
        abk  <= QHL ? 16'(cmd.w2[15:8]) + kb : '0;
        abh  <= QHL ? (32'(cmd.w2[15:8]) + 32'(kb)) * D : '0;
      end
      r <= '0; e <= '0;
      badr <= AW'(cmd_a(cmd, 2)); brs <= AW'(cmd_a(cmd, 2));   // QST dst (unused by QACT)
      sbase <= AW'(cmd_a(cmd, 3));                             // QST sdst
      row_src <= cmd.w1;
      pass <= 1'b0; rd_wait <= 1'b0; wide <= 1'b0; wsw <= 1'b0; sw_ph <= 1'b0; sw_i <= '0;
      // QST GRP: the first row's lane / byte; its group offsets are the products' (gi)
      grp  <= WG > 1 && qst && cmd.flags[2];
      gcol <= WG > 1 && qst && cmd.flags[2] && cmd.flags[3];
      gi   <= (WG > 1 && qst && cmd.flags[2]) ? 2'd3 : 2'd0;
      gln  <= LGW'(cmd.w6);
      gno  <= cmd.w6[DL-1:0];
      gkp  <= 32'(kb) << LP;
      gm_n <= cmd.w6 >> LG; gm_ko <= 32'(kb) << LP; gm_eo <= cmd.w7;
      m0a <= '0;
      w2a <= '0;
      st_frz <= '0;
      // QSTB: ROW, not HALF, pass 1 QSW bytes a cycle for every row (GRP rows; else element
      // stride 1, dst and the row stride QSW aligned)
      bm <= XD > 0 && 32'(kb) * D <= 32'(XD) * LANES &&
            (!qst || (QSTB && cmd.flags[0] && !cmd.flags[1] &&
                      ((WG > 1 && cmd.flags[2]) ? !cmd.flags[3]
                       : cmd.w7 == 32'd1 && (cmd.w2 & 32'(QSW - 1)) == 0 &&
                         (cmd.w6 & 32'(QSW - 1)) == 0)));
      p1a <= 1'b0; p1h <= 1'b0; r1 <= '0; e1c <= '0; xw <= '0; xr <= '0; xocc <= '0; dr <= '0;
      sv <= '0;
      mb0 <= '0;
      sb0 <= 1'b0;
      hb0 <= 1'b0;
      sbh <= 1'b0;
      if (nr == 0 || kb == 0) begin
        rd_done <= 1'b1;
        done_w <= 1'b1;
      end else begin
        rd_done <= 1'b0;
        busy <= 1'b1;
      end
    end else if (busy && en) begin
      logic fin;
      fin = 1'b0;
      // ---- QST GRP: the first row's group (rows: dst + (n0 / R) KB P, its area sdst + (n0 / R)
      // es; COL: dst + (n0 / D) P) once the products are out (gi 3 -> 0: the third enabled
      // cycle, after the operands' and the products' two registers)
      if (gi != 0) begin
        gi <= gi - 2'd1;
        if (gi == 2'd1) begin
          grb <= gcol ? brs + ((drs & ~32'(D - 1)) << LG) : brs + gm_k;
          gb  <= gcol ? brs + ((drs & ~32'(D - 1)) << LG) : brs + gm_k;
          gar <= sbase + gm_e;
        end
      end
      // ---- reader
      m0a <= '0;
      w2a <= '0;
      if (riss) begin
        m0a.v <= 1'b1;
        m0a.pass <= pass;
        m0a.e <= e;
        m0a.last <= rlast;
        m0a.row <= r;
        m0a.mask <= rmask;
        w2a <= {SWW > 1 && wsw, rmask2};
        m0a.glast <= pass && rlast && (32'(r) + 1 == 32'(rows));
        if (!rlast) begin
          e <= e + ew;
          rem <= rem - ew;
          if (!qbm) badr <= badr + bstep;         // (QSTB's buffered rows: the pops step it)
          // GRP pass 1: rows the next D-block's chunk (a word, or wsw's QSW bytes, a read), COL
          // the next R elements'
          if (grp && pass && (gcol ? LGW'(e) == LGW'(WG - 1)
                              : (SWW > 1 && wsw) ? (e[DL-1:0] | DL'(QSW - 1)) == DL'(D - 1)
                                                 : &e[DL-1:2]))
            gb <= gb + (gcol ? es : 32'(1) << LP);
        end else if (!pass && !bm) begin          // wait for the row's scale, re-read it
          rd_wait <= 1'b1;
          e <= '0;
          rem <= G;
          badr <= brs;
          wide <= is_st && (grp ? !gcol : es == 32'd1 && brs[1:0] == 2'd0);
          wsw <= SWW > 1 && is_st && (grp ? !gcol : es == 32'd1 && (brs & AW'(QSW - 1)) == 0);
        end else begin                             // the next row (buffered: after pass 0)
          e <= '0;
          rem <= G;
          pass <= 1'b0;
          wide <= 1'b0;
          wsw <= 1'b0;
          r <= r + 1;
          if (32'(r) + 1 == 32'(rows)) rd_done <= 1'b1;
          if (!grp && !qbm) begin
            brs <= brs + drs;
            badr <= brs + drs;
          end
          row_src <= row_src + srs;
          if (SWW > 1 && is_st) csb <= csb + srs;  // (wsw's Q2 reads: row_src + LANES)
          if (!qbm) sbase <= sbase + sstep;
          // GRP: rows the next lane (past the group's last: the next group and area); COL the
          // next byte (past the chunk's last: the next column chunk). (Buffered: QSTB's pops
          // step the byte and scale addresses at their row's end, below.)
          if (grp && !gcol && !qbm) begin
            gln <= gln + 1'b1;
            gb <= (gln == LGW'(WG - 1)) ? grb + gkp : grb;
            if (gln == LGW'(WG - 1)) begin
              grb <= grb + gkp;
              gar <= gar + es;
            end
          end
          if (gcol) begin
            gno <= gno + 1'b1;
            gb <= (&gno) ? grb + (32'(1) << LP) : grb;
            if (&gno) grb <= grb + (32'(1) << LP);
          end
        end
      end
      // ---- buffered pass 1: once row r1's scale is out, pop its chunks (one a cycle, the
      // group word read with each; HL: each twice, q then lo, the second advancing); a row's
      // start takes a cycle
      mb0 <= '0;
      hb0 <= 1'b0;
      sb0 <= 1'b0;
      if (p1a) begin
        mb0.v <= 1'b1;
        mb0.e <= e1c;
        mb0.row <= r1;
        mb0.mask <= '1;
        mb0.last <= p1x && (e1c + QW >= G);
        mb0.glast <= p1x && (e1c + QW >= G) && (32'(r1) + 1 == 32'(rows));
        hb0 <= HLA && hlm && p1h;
        if (HLA && hlm) p1h <= !p1h;
        if (p1x) begin
          xr <= xr + XW'(PW);
          if (e1c + QW >= G) begin
            p1a <= 1'b0;
            r1 <= r1 + 1'b1;
          end else e1c <= e1c + QW;
          // QSTB: the row's byte address on by the pop's QSW bytes (GRP rows: past a D-block's
          // last, the next block's chunk); at its last pop the next row's addresses, as the
          // reader steps an unbuffered row's
          if (QSTB && is_st) begin
            if (e1c + QW < G) begin
              if (!grp) badr <= badr + AW'(QSW);
              else if ((e1c[DL-1:0] | DL'(QSW - 1)) == DL'(D - 1)) gb <= gb + (AW'(1) << LP);
            end else begin
              if (!grp) begin
                brs <= brs + drs;
                badr <= brs + drs;
              end
              sbase <= sbase + sstep;
              if (grp) begin
                gln <= gln + 1'b1;
                gb <= (gln == LGW'(WG - 1)) ? grb + gkp : grb;
                if (gln == LGW'(WG - 1)) begin
                  grb <= grb + gkp;
                  gar <= gar + es;
                end
              end
            end
          end
        end
      end else if (bm && 32'(r1) < 32'(rows) && sv[r1[1:0]]) begin
        // QSTB: the row's scale word into the writer first (mb_baddr: its address); QSTHL: hi's,
        // then lo's (sbh), then pass 1
        if (!(QSTHL && hlm && is_st && !sbh)) begin
          p1a <= 1'b1;
          e1c <= '0;
        end
        if (QSTHL && hlm && is_st) sbh <= !sbh;
        if (QSTB && is_st) begin
          mb0.v <= 1'b1;
          mb0.row <= r1;
          mb0.mask <= '1;
          sb0 <= 1'b1;
          hb0 <= QSTHL && hlm && sbh;
        end
      end
      // the FIFO's entries taken (a buffered read) and given back (a pop); rows between the
      // passes (r - r1)
      xocc <= xocc + ((bm && riss) ? 1'b1 : 1'b0) - (p1x ? (XW+1)'(PW) : '0);
      if (xpush) xw <= xw + 1'b1;
      dr <= dr + ((bm && riss && rlast) ? 3'd1 : 3'd0) - ((p1x && e1c + QW >= G) ? 3'd1 : 3'd0);
      // ---- the maxes
      if (mc.v && !mc.pass) begin
        amx_r <= nmax;
        gacc <= gnew;
      end
      if (sq_iv) amx_q[mc.row[1:0]] <= nmax;
      // ---- the row's scale: QST writes its scale row, then pass 1 starts (buffered: pass 1
      // takes it when it gets to the row)
      if (sq_ov) begin
        sc_q[sq_ot] <= sq_sc;
        inv_q[sq_ot] <= sq_inv;
`ifndef SYNTHESIS
        inv_w[sq_ot] <= 1'b1;
`endif
        if (bm) begin
          sv[sq_ot] <= 1'b1;
        end else if (is_st) begin
          sw_ph <= 1'b1;
          sw_i <= '0;
          gsp <= gar + {gln, 2'b00};               // (GRP rows: the row's S)
        end else begin
          pass <= 1'b1;
          rem <= GW;                               // pass 1 re-reads from e = 0
          rd_wait <= 1'b0;
        end
      end
      if (!p1a && bm && 32'(r1) < 32'(rows) && sv[r1[1:0]] && !(QSTHL && hlm && is_st && !sbh))
        sv[r1[1:0]] <= 1'b0;                     // (QSTHL: at lo's scale word, pass 1's start)
      if (sw_ph) begin
        // (GRP rows: block 0's d byte at 4 R + n % R, each next R on)
        gsp <= (sw_i == 0) ? gar + (32'(4 * WG) | 32'(gln)) : gsp + 32'(WG);
        if (sw_i + 16'd1 == swn) begin
          sw_ph <= 1'b0;
          pass <= 1'b1;
          rem <= GW;
          rd_wait <= 1'b0;
        end else sw_i <= sw_i + 16'd1;
      end
      // ---- the last write
      if (wqd.v && wqd.fin) fin = 1'b1;
      if (fin && (is_st || (QONCE && stm))) begin
        ackw <= 1'b1;                              // the last write is taken this cycle
      end else if (fin) begin
        busy <= 1'b0;
        done_w <= 1'b1;
        pf_u <= 1'b1;
        pf_frz <= st_frz + 32'(st_c);
      end
    end
  end
`ifndef SYNTHESIS
  // the ACT RAM write above takes a QACT write as whole lane groups at a LANES x PW-aligned
  // index (act_we from each group's first mask bit, act_idx with its low bits cleared): check
  // that every QACT write is
  always @(posedge clk)
    if (!rst && rst_seen && busy && !ackw && wqd.v && !wqd.st) begin
      if (wqd.mask == '0 || (wqd.idx & 32'(QW - 1)) != 0)
        $fatal(1, "otpu_quant: QACT write empty or not LANES x PW aligned (idx %0d)", wqd.idx);
      for (int g = 0; g < PW; g++)
        if (wqd.mask[g*LANES +: LANES] != '0 && wqd.mask[g*LANES +: LANES] != '1)
          $fatal(1, "otpu_quant: QACT write not whole lane groups (mask %b)", wqd.mask);
    end
  // QST GRP: the machine's groups (WG > 1), COL with ROW, the alignments (docs/isa.md QST); the
  // first row's group offsets before its scale row and pass 1
  always @(posedge clk)
    if (!rst && rst_seen && start && cmd.op == OP_QST && cmd.flags[2]) begin
      if (WG == 1 || 32'(cmd.flags[6:4]) != 32'(LG))
        $fatal(1, "otpu_quant: QST GRP in groups of %0d rows, the machine's WG %0d",
               1 << cmd.flags[6:4], WG);
      if (cmd.flags[3] && !cmd.flags[0]) $fatal(1, "otpu_quant: QST GRP COL needs ROW");
      if ((cmd.w2 & ((32'(1) << LP) - 1)) != 0 || cmd.w3[1:0] != 0 ||
          (cmd.w7 & (cmd.flags[3] ? (32'(1) << LP) - 1 : 32'd3)) != 0)
        $fatal(1, "otpu_quant: QST GRP: dst %0h, sdst %0h, es %0h not aligned", cmd.w2,
               cmd.w3, cmd.w7);
    end
  always @(posedge clk)
    if (!rst && rst_seen && busy && grp && gi != 0 && (pass || sw_ph || p1a || sb0))
      $fatal(1, "otpu_quant: QST GRP's first row before its group offsets");
  // HL: through the row buffer, not with LO or DUP (docs/isa.md QACT)
  always @(posedge clk)
    if (!rst && start && QHL && cmd.op == OP_QACT && cmd.flags[5] &&
        (cmd.flags[4] || cmd.flags[3] || 32'(cmd.w2[31:16]) * D > 32'(XD) * LANES))
      $fatal(1, "otpu_quant: QACT HL with LO or DUP, or %0d x %0d elements a row past the row buffer",
             cmd.w2[31:16], D);
  // QSTHL: QST HL on GRP ROW rows, not COL, HALF or LO, through the row buffer (docs/isa.md QST)
  always @(posedge clk)
    if (!rst && start && QSTHL && cmd.op == OP_QST && cmd.w5[31] &&
        (!(WG > 1 && cmd.flags[2]) || cmd.flags[3] || !cmd.flags[0] || cmd.flags[1] ||
         cmd.flags[7] || cmd.w5[30:20] == 0 || ({cmd.w5[30:20], 10'b0} & ((32'(1) << LP) - 1)) != 0 ||
         32'(cmd.w4[31:16]) * D > 32'(XD) * LANES))
      $fatal(1, "otpu_quant: QST HL off GRP ROW rows, with COL, HALF, LO, HLD 0 or not a multiple of P, or %0d x %0d elements a row past the row buffer",
             cmd.w4[31:16], D);
  // a row's group maxes fit the group RAMs
  always @(posedge clk)
    if (!rst && rst_seen && start && !cmd.flags[0] &&
        ((cmd.op == OP_QST) ? 32'(cmd.w4[31:16]) : 4 * 32'(cmd.w2[31:16])) > GR)
      $fatal(1, "otpu_quant: %0d groups a row, at most %0d", (cmd.op == OP_QST) ?
             32'(cmd.w4[31:16]) : 4 * 32'(cmd.w2[31:16]), GR);
`endif
endmodule
