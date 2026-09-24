// VPU: LANES fp32 elements per cycle along a row of a [rows, cols] TMEM tile. Operand A from
// read port A, operand B from read port B (full tile, per-column), one broadcast word (per-row)
// or the scalar immediate (docs/isa.md, VOP).
//
// Pipelined for the FPGA clock. Split lanes: lanes 0..CL-1 are chains of NSLOT multiply-add
// slots (slot = (a*b)+c with its own input register, SL = 1 + LM + LA cycles; slot 1 has three
// more for the EXP2 range reduction); lanes CL..LANES-1 have slot 0 only. The composite
// functions (EXP2, RECIP, RSQRT) are issued CL columns per cycle onto the long lanes, every
// other function LANES columns per cycle. Composite functions
// are slot programs, exactly the fp_add/fp_mul sequences of the ISA:
//   ADD/SUB/RSUB/MUL  1 slot     EXP2/EXP2SUB 9 (sub, range reduction, 7 Horner steps)
//   RECIP             6 (3 Newton steps)      RSQRT  10 (h = x/2, 3 Newton steps)
//   MAX/MIN/COPY/ABS/FILL 0 (written the cycle after the read)
// The result is written from the slot where the function ends, so latency depends on the
// function. Several elementwise instructions overlap in the pipeline: every entry carries its
// function, and an instruction may start while others are in flight only if its latency is
// at least theirs, so writes and completions stay in start order. (The sequencer makes a VOP
// wait for the VOPs it depends on to complete.) Reductions run alone.
//
// Reductions: RMAX folds each chunk with a max tree (the order is total, so any order is
// exact). RSUM/RSSQ implement isum_64: lane l owns the partials l, l+LANES, ... and adds each
// chunk's term into the partial last updated RL = 64/LANES chunks ago -- a loop of exactly RL
// cycles through a pipelined adder. Rows are padded with +0 terms to a multiple of 64 columns;
// the final partials of a row come out LANES per cycle, in order, and a streaming folding tree
// sums them while the next row streams: the levels across rows on LANES adders fed by delay
// lines on a fixed schedule, the levels across lanes on a pipeline of adders (level 2 reuses
// level 1's, which are idle then).
//
// Everything (reads, lane pipelines, the tree, writes) advances on cycles the TMEM grant is
// given; nothing stalls.
module otpu_vpu
  import otpu_pkg::*;
  import otpu_fp::*;
#(
  parameter int LANES = 8,
  parameter int CL    = (LANES >= 8) ? LANES / 4 : 1,   // lanes with the composite functions
  parameter int SID   = 0
) (
  input  logic                    clk,
  input  logic                    rst,
  input  logic                    start,
  input  cmd_t                    cmd,
  output logic                    rdy,
  output logic                    done,
  input  logic                    gnt,        // TMEM grant: hold everything when low
  output logic [LANES-1:0]        ta_en,
  output logic [LANES-1:0][31:0]  ta_addr,
  input  logic [LANES-1:0][31:0]  ta_data,
  output logic [LANES-1:0]        tb_en,
  output logic [LANES-1:0][31:0]  tb_addr,
  input  logic [LANES-1:0][31:0]  tb_data,
  output logic [LANES-1:0]        tw_en,
  output logic [LANES-1:0][31:0]  tw_addr,
  output logic [LANES-1:0][31:0]  tw_data
);
  localparam int LM = 2, LA = 4;
  localparam int SL = 1 + LM + LA;          // cycles per slot
  localparam int NSLOT = 10;
  localparam int NP = 64;                    // RSUM/RSSQ partials (isum_64)
  localparam int RL = NP / LANES;            // partial loop length in chunks
  localparam int LW = $clog2(LANES);
  localparam int NCL = (CL < LANES) ? CL : LANES;
  localparam int CLW = $clog2(NCL);
  initial if (RL < LA || NP % LANES != 0)
    $fatal(1, "otpu_vpu: LANES must be a power of two <= 16");

  localparam f32_t F_NZ = 32'h8000_0000;     // -0: (a*b) + -0 == a*b exactly

  // ------------------------------------------------------------------ command state
  logic        issuing, red_act;
  logic [31:0] dst, a, b, imm;
  logic [15:0] rows, cols, drs, ars, brs;
  logic [7:0]  func;
  logic [1:0]  bmode;
  logic [15:0] ir, ic, nch, ch;               // issue: row, column, chunks per row, chunk
  logic [31:0] a_row, b_row, d_row;           // row base addresses (no multipliers)
  logic [3:0]  nslots;                         // slots of this function
  logic [7:0]  tag;                            // instruction tag
  logic [31:0] cyc, st_frz;

  function automatic logic [3:0] n_slots(input logic [7:0] f);
    case (f)
      V_ADD, V_SUB, V_RSUB, V_MUL: return 4'd1;
      V_EXP2, V_EXP2SUB:           return 4'd9;
      V_RECIP:                     return 4'd6;
      V_RSQRT:                     return 4'd10;
      default:                     return 4'd0;
    endcase
  endfunction

  function automatic logic comp_f(input logic [7:0] f);
    return n_slots(f) > 1;
  endfunction
  wire [15:0] iwid = comp_f(func) ? 16'(NCL) : 16'(LANES);   // columns issued per cycle

  wire is_sum = (func == V_RSUM) || (func == V_RSSQ);
  wire is_red = is_sum || (func == V_RMAX);

  function automatic logic red_f(input logic [7:0] f);
    return f == V_RSUM || f == V_RSSQ || f == V_RMAX;
  endfunction

  // command queue (the sequencer starts instructions into it)
  cmd_t        cq [2];
  logic [1:0]  cq_n;                          // queued
  logic        cq_h;
  logic [3:0]  ew_n;                          // elementwise instructions issued, not done
  logic [3:0]  last_tap;                      // slots of the last one started
  assign rdy = (cq_n < 2);
  cmd_t hc;
  assign hc = cq[cq_h];
  wire  [7:0] hf = hc.w6[23:16];
  wire  h_empty = (hc.w4[15:0] == 0) || (hc.w4[31:16] == 0);
  wire  can_begin = (cq_n != 0) && !issuing && !red_act &&
                    ((red_f(hf) || h_empty) ? (ew_n == 0) :
                                              (ew_n == 0 || n_slots(hf) >= last_tap));
  wire  busy = issuing || red_act || ew_n != 0;


  wire en = gnt;

  // ------------------------------------------------------------------ issue
  logic [LANES-1:0] imask;
  logic             irow_last, iall_last;
  always_comb begin
    ta_en = '0; ta_addr = '0; tb_en = '0; tb_addr = '0;
    imask = '0;
    irow_last = (ch + 1 == nch);
    iall_last = irow_last && (ir + 1 == rows);
    if (issuing) begin
      for (int l = 0; l < LANES; l++) begin
        if (32'(ic) + 32'(l) < 32'(cols) && 16'(l) < iwid) begin
          imask[l] = 1'b1;
          ta_en[l] = (func != V_FILL);
          ta_addr[l] = a_row + 32'(ic) + 32'(l);
          if (is_binary(func)) case (bmode)
            B_FULL: begin
              tb_en[l] = 1'b1;
              tb_addr[l] = b_row + 32'(ic) + 32'(l);
            end
            B_COL: begin
              tb_en[l] = 1'b1;
              tb_addr[l] = b + 32'(ic) + 32'(l);
            end
            default: ;
          endcase
        end
      end
      if (bmode == B_ROW && is_binary(func) && imask[0]) begin
        tb_en[0] = 1'b1;
        tb_addr[0] = b_row;
      end
    end
  end

  // meta of a chunk; m0 describes the data arriving this cycle (read last cycle)
  typedef struct packed {
    logic             v;
    logic [7:0]       tag;
    logic [7:0]       func;
    logic [1:0]       bmode;
    logic [31:0]      imm;
    logic [LANES-1:0] mask;
    logic [31:0]      waddr;     // elementwise: dst row + column; reductions: dst row
    logic             row_last;  // last chunk of its row
    logic             all_last;  // last chunk of the instruction
    logic             first;     // reductions: chunk index < RL (partials start at +0)
    logic             final_;    // reductions: chunk index >= nch - RL (partials become final)
    logic [7:0]       sub;       // chunk index mod RL
  } meta_t;
  // mi: the chunk read this cycle; m0: that chunk one granted cycle later, together with its
  // TMEM data registered (xa, xb), so no path runs from the TMEM block RAMs into the lanes'
  // multipliers in one cycle
  meta_t m0, mi;
  logic [LANES-1:0][31:0] xa, xb;
  always_ff @(posedge clk)
    if (rst) m0 <= '0;
    else if (en) begin
      m0 <= mi;
      xa <= ta_data;
      xb <= tb_data;
    end

  // the parts of the meta the later stages read: along the lane taps, and for the reductions
  typedef struct packed {
    logic             v;
    logic [7:0]       func;
    logic [LANES-1:0] mask;
    logic [31:0]      waddr;
    logic             all_last;
  } mt_t;
  typedef struct packed {
    logic             v;
    logic [7:0]       tag;
    logic [31:0]      waddr;
    logic             row_last;
    logic             all_last;
    logic             first;
    logic             final_;
    logic [7:0]       sub;
  } rm_t;
  rm_t m0r;
  assign m0r = '{v: m0.v, tag: m0.tag, waddr: m0.waddr, row_last: m0.row_last,
                 all_last: m0.all_last, first: m0.first, final_: m0.final_, sub: m0.sub};

  function automatic logic live(input rm_t m, input logic [7:0] t);
    return m.v && m.tag == t;
  endfunction

  // ------------------------------------------------------------------ elementwise lanes
  typedef struct packed {
    f32_t        v, k1, k2;
    logic [8:0]  ii;
    logic [2:0]  f;
  } lst_t;

  // Lane state fields live across slot s's delay line: read by a later slot or by the end tap
  // (EXP2 9: f v ii, RECIP 6: f k2, RSQRT 10: f k2) before being rewritten. Derived from the
  // slot programs in g_slot -- recheck when editing one. v is never live: every slot followed
  // by a read of v writes v itself. f is always live; dead fields are not carried.
  function automatic logic k1_live(input int s);
    return s <= 7;
  endfunction
  function automatic logic k2_live(input int s);
    return s <= 8 && s != 3 && s != 6;
  endfunction
  function automatic logic ii_live(input int s);
    return s >= 1 && s <= 8;
  endfunction

  mt_t   mtap [NSLOT + 1];
  f32_t  lres [LANES];

  assign mtap[0] = '{v: m0.v, func: m0.func, mask: m0.mask, waddr: m0.waddr,
                     all_last: m0.all_last};
  // slot 1 has three extra input stages for the EXP2 range reduction (clamp, floor, i2f)
  localparam int PRE1 = 3;
  mt_t   msl [NSLOT];                        // meta at each slot's input mux
  for (genvar s = 0; s < NSLOT; s++) begin : g_mdel
    if (s == 1) begin : g_pre
      otpu_delay #(.W($bits(mt_t)), .N(PRE1)) u_p (.clk, .en, .d(mtap[s]), .q(msl[s]));
    end else begin : g_nopre
      assign msl[s] = mtap[s];
    end
    otpu_delay #(.W($bits(mt_t)), .N(SL)) u_d (.clk, .en, .d(msl[s]), .q(mtap[s + 1]));
  end

  // which boundary holds an entry at its function's last slot (at most one: see header)
  logic [3:0] wtap;
  mt_t        mo;
  always_comb begin
    wtap = '0;
    mo = '0;
    for (int s = NSLOT; s >= 0; s--)
      if (mtap[s].v && !red_f(mtap[s].func) && n_slots(mtap[s].func) == 4'(s)) begin
        wtap = 4'(s);
        mo = mtap[s];
      end
  end

  // long lanes: all functions
  for (genvar l = 0; l < NCL; l++) begin : g_lane
    localparam int NS = NSLOT;                     // slots of this lane
    f32_t x, y;
    lst_t st [NS + 1];
    assign x = xa[l];
    assign y = (m0.bmode == B_SCALAR) ? m0.imm : (m0.bmode == B_ROW) ? xb[0] : xb[l];

    // boundary 0: the simple functions' results and the composite functions' setup
    always_comb begin
      f32_t xz, ax;
      xz = ftz(x);
      ax = {1'b0, xz[30:0]};
      st[0] = '0;
      case (m0.func)
        V_MAX:   st[0].v = fp_max(x, y);
        V_MIN:   st[0].v = fp_min(x, y);
        V_COPY:  st[0].v = xz;
        V_ABS:   st[0].v = fabs(x);
        V_FILL:  st[0].v = ftz(y);
        V_RECIP: begin
          st[0].k1 = {1'b1, ax[30:0]};                       // -|x|
          st[0].k2 = ftz(RECIP_MAGIC - ax);                  // seed
          st[0].f  = {xz[31], (ax >= 32'h7E80_0000), (ax == 0)};
        end
        V_RSQRT: begin
          st[0].k2 = RSQRT_MAGIC - (xz >> 1);
          st[0].f  = {2'b00, (xz[31] || xz[30:0] == 0 || xz == F_INF)};
        end
        default: ;
      endcase
    end

    for (genvar s = 0; s < NS; s++) begin : g_slot
      f32_t       ia, ib, ic_, r;
      lst_t       sin_, sd, sti;
      logic [1:0] dest;                  // 0: none, 1: v, 2: k1, 3: k2
      logic       negd;                  // store fneg(result)
      if (s == 1) begin : g_pre
        // EXP2 range reduction: xf = x clamped, i = floor(xf) | -i2f(i) (into k2, unused here)
        // three stages: clamp | floor | i2f
        lst_t p0, p1, p2;
        logic e1, e2;                    // the entry in p0 / p1 is an EXP2
        always_ff @(posedge clk) if (en) begin
          lst_t t;
          logic lo, hi;
          t = st[s];
          if (mtap[s].func == V_EXP2 || mtap[s].func == V_EXP2SUB) begin
            lo = fp_gt(F_M126, t.v);
            hi = !fp_gt(F_128, t.v);
            t.v = (lo || hi) ? F_ZERO : t.v;
            t.f = {1'b0, hi, lo};
          end
          p0 <= t;
          e1 <= (mtap[s].func == V_EXP2 || mtap[s].func == V_EXP2SUB);
          t = p0;
          if (e1) t.ii = 9'(ffloor(p0.v));
          p1 <= t;
          e2 <= e1;
          t = p1;
          if (e2) t.k2 = fneg(i2f(32'($signed(p1.ii))));
          p2 <= t;
        end
        assign sti = p2;
      end else begin : g_nopre
        assign sti = st[s];
      end
      always_comb begin
        lst_t t;
        t = sti;
        ia = F_ZERO; ib = F_ONE; ic_ = F_NZ; dest = 2'd0; negd = 1'b0;
        case (msl[s].func)
          V_MUL:  if (s == 0) begin ia = x; ib = y; dest = 2'd1; end
          V_ADD:  if (s == 0) begin ia = x; ic_ = y; dest = 2'd1; end
          V_SUB:  if (s == 0) begin ia = x; ic_ = fneg(y); dest = 2'd1; end
          V_RSUB: if (s == 0) begin ia = y; ic_ = fneg(x); dest = 2'd1; end
          V_EXP2, V_EXP2SUB: begin
            if (s == 0) begin
              ia = x; ic_ = (msl[s].func == V_EXP2SUB) ? fneg(y) : F_NZ; dest = 2'd1;
            end else if (s == 1) begin
              ia = t.v; ic_ = t.k2; dest = 2'd2;                          // f = xf - i
            end else if (s == 2) begin
              ia = EXP2_C7; ib = t.k1; ic_ = EXP2_C6; dest = 2'd1;
            end else if (s <= 8) begin
              ia = t.v; ib = t.k1; dest = 2'd1;
              case (s)
                3: ic_ = EXP2_C5;
                4: ic_ = EXP2_C4;
                5: ic_ = EXP2_C3;
                6: ic_ = EXP2_C2;
                7: ic_ = EXP2_C1;
                default: ic_ = EXP2_C0;
              endcase
            end
          end
          V_RECIP: if (s < 6) begin
            if (s % 2 == 0) begin ia = t.k1; ib = t.k2; ic_ = F_TWO; dest = 2'd1; end   // 2 - |x|y
            else begin ia = t.k2; ib = t.v; dest = 2'd3; end                              // y = y*t
          end
          V_RSQRT: begin
            if (s == 0) begin ia = F_HALF; ib = ftz(x); dest = 2'd2; negd = 1'b1; end     // -h
            else if ((s - 1) % 3 == 0) begin ia = t.k2; ib = t.k2; dest = 2'd1; end       // y*y
            else if ((s - 1) % 3 == 1) begin ia = t.k1; ib = t.v; ic_ = F_1P5; dest = 2'd1; end
            else begin ia = t.k2; ib = t.v; dest = 2'd3; end                              // y*c
          end
          default: ;
        endcase
        sin_ = t;
      end
      f32_t       ra, rb, rc;
      lst_t       sr;
      logic [1:0] dr, dd;
      logic       nr, nd;
      always_ff @(posedge clk) if (en) begin
        ra <= ia; rb <= ib; rc <= ic_; sr <= sin_; dr <= dest; nr <= negd;
      end
      otpu_fmadd #(.LM(LM), .LA(LA)) u_ma (.clk, .en, .a(ra), .b(rb), .c(rc), .y(r));
      // the live fields only (see k1_live); the dead ones read as 0 and are never used
      assign sd.v = '0;
      if (k1_live(s)) begin : g_k1
        otpu_delay #(.W(32), .N(LM + LA)) u_k1 (.clk, .en, .d(sr.k1), .q(sd.k1));
      end else begin : g_nk1
        assign sd.k1 = '0;
      end
      if (k2_live(s)) begin : g_k2
        otpu_delay #(.W(32), .N(LM + LA)) u_k2 (.clk, .en, .d(sr.k2), .q(sd.k2));
      end else begin : g_nk2
        assign sd.k2 = '0;
      end
      if (ii_live(s)) begin : g_ii
        otpu_delay #(.W(9), .N(LM + LA)) u_ii (.clk, .en, .d(sr.ii), .q(sd.ii));
      end else begin : g_nii
        assign sd.ii = '0;
      end
      otpu_delay #(.W(3), .N(LM + LA)) u_f (.clk, .en, .d(sr.f), .q(sd.f));
      otpu_delay #(.W(3), .N(LM + LA)) u_dst (.clk, .en, .d({dr, nr}), .q({dd, nd}));
      always_comb begin
        f32_t rr;
        rr = nd ? fneg(r) : r;
        st[s + 1] = sd;
        case (dd)
          2'd1: st[s + 1].v = rr;
          2'd2: st[s + 1].k1 = rr;
          2'd3: st[s + 1].k2 = rr;
          default: ;
        endcase
      end
    end

    // the result of the entry that ends at a tap this cycle (taps: 0, 1, 6, 9, 10)
    always_comb begin
      lst_t t;
      t = st[wtap];
      case (mo.func)
        V_EXP2, V_EXP2SUB:
          lres[l] = t.f[0] ? F_ZERO : t.f[1] ? F_INF : (t.v + {t.ii, 23'd0});
        V_RECIP:
          lres[l] = t.f[0] ? F_ZERO : t.f[1] ? {t.f[2], 31'd0} : (t.f[2] ? fneg(t.k2) : t.k2);
        V_RSQRT:
          lres[l] = t.f[0] ? F_ZERO : t.k2;
        default:
          lres[l] = t.v;
      endcase
    end
  end

  // short lanes: slot 0 only. The composite functions are never issued here (imask), so their
  // setup, operand cases and result muxes are left out.
  for (genvar l = NCL; l < LANES; l++) begin : g_slane
    f32_t x, y;
    f32_t st [2];                                  // v at boundaries 0 and 1
    assign x = xa[l];
    assign y = (m0.bmode == B_SCALAR) ? m0.imm : (m0.bmode == B_ROW) ? xb[0] : xb[l];

    always_comb begin
      st[0] = '0;
      case (m0.func)
        V_MAX:   st[0] = fp_max(x, y);
        V_MIN:   st[0] = fp_min(x, y);
        V_COPY:  st[0] = ftz(x);
        V_ABS:   st[0] = fabs(x);
        V_FILL:  st[0] = ftz(y);
        default: ;
      endcase
    end
    f32_t ia, ib, ic_, ra, rb, rc;
    always_comb begin
      ia = F_ZERO; ib = F_ONE; ic_ = F_NZ;
      case (msl[0].func)
        V_MUL:  begin ia = x; ib = y; end
        V_ADD:  begin ia = x; ic_ = y; end
        V_SUB:  begin ia = x; ic_ = fneg(y); end
        V_RSUB: begin ia = y; ic_ = fneg(x); end
        default: ;
      endcase
    end
    always_ff @(posedge clk) if (en) begin
      ra <= ia; rb <= ib; rc <= ic_;
    end
    otpu_fmadd #(.LM(LM), .LA(LA)) u_ma (.clk, .en, .a(ra), .b(rb), .c(rc), .y(st[1]));

    // the result of the entry that ends at tap 0 or 1 this cycle (other taps: masked here)
    assign lres[l] = (wtap == 0) ? st[0] : st[1];
  end

  // ------------------------------------------------------------------ RMAX
  // pairwise max tree over the lanes of the chunk (masked lanes drop out), one registered
  // stage per level: stage 0 from the TMEM read data (LANES -> HL), stage j halves HL >> (j-1)
  localparam int HL = LANES / 2;
  localparam int ML = $clog2(HL);             // levels after the first
  f32_t  mxh_v [ML + 1][HL];                  // stage j holds HL >> j values
  logic  mxh_h [ML + 1][HL];
  rm_t   mxh_m [ML + 1];
  f32_t  mxc_q;
  rm_t   mxm_q;
  f32_t  mx_run;
  logic  mx_have;
  always_ff @(posedge clk) if (en) begin
    f32_t v [LANES];
    logic h [LANES];
    for (int l = 0; l < LANES; l++) begin
      v[l] = ftz(xa[l]);
      h[l] = m0.mask[l];
    end
    for (int l = 0; l < HL; l++) begin
      mxh_v[0][l] <= (!h[l] || (h[l + HL] && fp_gt(v[l + HL], v[l]))) ? v[l + HL] : v[l];
      mxh_h[0][l] <= h[l] || h[l + HL];
    end
    mxh_m[0] <= m0r;
    for (int j = 1; j <= ML; j++) begin
      for (int l = 0; l < (HL >> j); l++) begin
        mxh_v[j][l] <= (!mxh_h[j-1][l] || (mxh_h[j-1][l + (HL >> j)] &&
                        fp_gt(mxh_v[j-1][l + (HL >> j)], mxh_v[j-1][l]))) ?
                       mxh_v[j-1][l + (HL >> j)] : mxh_v[j-1][l];
        mxh_h[j][l] <= mxh_h[j-1][l] || mxh_h[j-1][l + (HL >> j)];
      end
      mxh_m[j] <= mxh_m[j-1];
    end
  end
  assign mxc_q = mxh_v[ML][0];
  assign mxm_q = mxh_m[ML];
  f32_t mx_new;
  assign mx_new = (mx_have && fp_gt(mx_run, mxc_q)) ? mx_run : mxc_q;

  // ------------------------------------------------------------------ RSUM / RSSQ
  f32_t  pacc [LANES];                  // partial after adding this chunk's term
  rm_t   mtq, mt;                       // meta at the adder inputs / aligned with `pacc`
  otpu_delay #(.W($bits(rm_t)), .N(LM)) u_mtq (.clk, .en, .d(m0r), .q(mtq));
  otpu_delay #(.W($bits(rm_t)), .N(LA)) u_mt (.clk, .en, .d(mtq), .q(mt));
  f32_t  fbd [LANES];                   // pacc delayed RL - LA - 1 (also read by the tree)
  logic  first_e;                       // mtq.first one cycle early
  otpu_delay #(.W(1), .N(LM - 1)) u_first (.clk, .en, .d(m0.first), .q(first_e));
  for (genvar l = 0; l < LANES; l++) begin : g_red
    f32_t xin, tq, prev;
    assign xin = m0.mask[l] ? ftz(xa[l]) : F_ZERO;
    otpu_fmul #(.LAT(LM)) u_sq (.clk, .en, .a(xin), .b((m0.func == V_RSSQ) ? xin : F_ONE), .y(tq));
    // pacc(chunk c) = pacc(chunk c - RL) + term(c): a loop of exactly RL cycles
    if (RL - LA >= 1) begin : g_fbq
      // the loop's last stage is a flip-flop with a sync clear (the first chunks' +0): no
      // shift-register LUT and no mux in front of the adder
      f32_t fbq;
      otpu_delay #(.W(32), .N(RL - LA - 1)) u_fb (.clk, .en, .d(pacc[l]), .q(fbd[l]));
      always_ff @(posedge clk) if (en) fbq <= first_e ? F_ZERO : fbd[l];
      assign prev = fbq;
    end else begin : g_fbw
      f32_t fb;
      otpu_delay #(.W(32), .N(RL - LA)) u_fb (.clk, .en, .d(pacc[l]), .q(fb));
      assign prev = mtq.first ? F_ZERO : fb;
      assign fbd[l] = F_ZERO;
    end
    otpu_fadd #(.LAT(LA)) u_acc (.clk, .en, .a(prev), .b(tq), .y(pacc[l]));
  end

  // Folding tree (level n = 32, 16, .., 1: x[i] += x[i+n], i < n), streamed. Partial p sits in
  // row p / LANES, lane p % LANES, and a row's final partials arrive on `pacc` one row per
  // cycle (sub 0..RL-1). Row levels (n >= LANES; j = 0..LR-1, H = RL >> (j+1) pairs: row r +=
  // row r + H) run on the LANES adders u_tree; pair r of level j issues I_j + r cycles after
  // sub 0 is on `pacc`, its operands taken from delay lines (level 0: `pacc`, level j >= 1:
  // `tr_y`). Level j issues in the residues [H, 2H) mod RL, and rows start a multiple of RL
  // cycles apart (nch is padded to a multiple of RL, rows issue back to back, reductions run
  // alone), so the adders are never double-booked and nothing stalls. Lane levels (n < LANES)
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
  localparam int DMAX = tree_dmax();
  localparam int DL = tree_i(LR - 1) - (RL - 1);                // row_last -> last row level
  localparam int RD = tree_i(LR - 1) + LA * (1 + LW) - (RL - 1); // row_last -> root

  // a final partial of this reduction is on `pacc`
  wire cap = red_act && is_sum && live(mt, tag) && mt.final_;

  // level strobes: cap of the subs a level pairs up, delayed to the level's issue cycles
  // (level 0 is the default); the last row level and the root share one shift register
  logic [LR-1:1] lv;
  logic [RD-1:0] rsr;
  always_ff @(posedge clk)
    if (rst) rsr <= '0;
    else if (en) rsr <= {rsr[RD-2:0], cap && mt.row_last};
  assign lv[LR-1] = rsr[DL-1];
  wire root_v = rsr[RD-1];
  for (genvar j = 1; j < LR - 1; j++) begin : g_lv
    localparam int DJ = tree_i(j) - (RL - tree_h(j));
    logic [DJ-1:0] sr;
    always_ff @(posedge clk)
      if (rst) sr <= '0;
      else if (en) sr <= {sr[DJ-2:0], cap && mt.sub >= 8'(RL - tree_h(j))};
    assign lv[j] = sr[DJ-1];
  end

  f32_t tr_y [LANES];
  for (genvar k = 0; k < LANES; k++) begin : g_tree
    f32_t pd, pdd, oa, ob;
    f32_t sa [LR], sb [LR];                  // level j's operands (row r, row r + H)
    logic [DMAX:1][31:0] td;                 // tr_y[k] delayed 1..DMAX
    wire  [DMAX:0][31:0] tt = {td, tr_y[k]};
    // pd = pacc delayed RL/2; its last stage has a reset so it stays a flip-flop, not a
    // shift-register LUT (the value after reset is never used)
    if (RL / 2 == RL - LA) begin : g_pdf
      assign pdd = fbd[k];
    end else begin : g_pdd
      otpu_delay #(.W(32), .N(RL / 2 - 1)) u_pd (.clk, .en, .d(pacc[k]), .q(pdd));
    end
    always_ff @(posedge clk)
      if (rst) pd <= '0;
      else if (en) pd <= pdd;
    always_ff @(posedge clk) if (en) begin
      td[1] <= tr_y[k];
      for (int d = 2; d <= DMAX; d++) td[d] <= td[d-1];
    end
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
    otpu_fadd #(.LAT(LA)) u_tree (.clk, .en, .a(oa), .b(ob), .y(tr_y[k]));
  end

  // lane levels n = LANES >> q: xl[q][k] = xl[q-1][k] + xl[q-1][k + n], k < n; the root is
  // xl[LW][0], RD cycles after the row's last partial. Level q issues LA * q cycles after the
  // last row level (l2_v for q = 2), rows a multiple of RL cycles apart: unless LA is a
  // multiple of RL, level 2 never meets level 1 and runs on level 1's adders k < LANES/4
  // (SH2; xl[2][k] is then xl[1][k] LA cycles later)
  localparam bit SH2 = (LW >= 2) && (LA % RL != 0);
  wire  l2_v = rsr[DL + 2 * LA - 1];
  f32_t xl [LW + 1][LANES];
  f32_t xs [LANES];                           // SH2: the shared adders' results (level 1 or 2)
  for (genvar k = 0; k < LANES; k++) begin : g_xl0
    assign xl[0][k] = tr_y[k];
  end
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

  // rows finish in order: the next root goes to wr_row
  logic [31:0]   wr_row;
  logic [15:0]   rows_done;

  // ------------------------------------------------------------------ TMEM writes
  // The writes are computed here (cw_*) and registered (tw_*): a TMEM write is performed one
  // granted cycle after the cycle that produced it, so no path runs from the TMEM read data
  // through the lanes into the TMEM write port. `done` follows its instruction's last write.
  logic [LANES-1:0]       cw_en;
  logic [LANES-1:0][31:0] cw_addr, cw_data;
  logic                   done_i, dpend;
  always_ff @(posedge clk) begin
    if (rst) begin
      tw_en <= '0;
      done <= 1'b0; dpend <= 1'b0;
    end else begin
      if (gnt) begin
        tw_en <= cw_en; tw_addr <= cw_addr; tw_data <= cw_data;
      end
      done <= (done_i || dpend) && gnt;
      dpend <= (done_i || dpend) && !gnt;
    end
  end
  always_comb begin
    cw_en = '0; cw_addr = '0; cw_data = '0;
    if (mo.v) begin
      for (int l = 0; l < LANES; l++) begin
        if (mo.mask[l]) begin
          cw_en[l] = 1'b1;
          cw_addr[l] = mo.waddr + 32'(l);
          cw_data[l] = lres[l];
        end
      end
    end
    if (red_act && func == V_RMAX && live(mxm_q, tag) && mxm_q.row_last) begin
      cw_en[0] = 1'b1;
      cw_addr[0] = mxm_q.waddr;
      cw_data[0] = mx_new;
    end
    if (red_act && is_sum && root_v) begin
      cw_en[0] = 1'b1;
      cw_addr[0] = wr_row;
      cw_data[0] = xl[LW][0];
    end
  end

  // ------------------------------------------------------------------ sequencing
  always_ff @(posedge clk) begin
    logic fin, ewfin;
    done_i <= 1'b0;
    cyc <= cyc + 1;
    fin = 1'b0;
    ewfin = 1'b0;
    if (busy && !en) st_frz <= st_frz + 1;
    if (rst) begin
      cq_n <= '0; cq_h <= 1'b0;
      issuing <= 1'b0; red_act <= 1'b0;
      ew_n <= '0; last_tap <= '0;
      mi <= '0;
      tag <= '0;
      cyc <= '0;
    end else begin
      logic [3:0] ewn;
      logic [1:0] qn;
      ewn = ew_n;
      qn = cq_n;
      // ---- accept from the sequencer
      if (start) begin
        cq[cq_h ^ cq_n[0]] <= cmd;
        qn = qn + 1;
      end
      // ---- begin issuing the head instruction
      if (can_begin && en) begin
        logic [15:0] c16, nchunks;
        cq_h <= ~cq_h;
        qn = qn - 1;
        if (h_empty) begin
          done_i <= 1'b1;
        end else begin
          dst <= hc.w1; a <= hc.w2; b <= hc.w3;
          rows <= hc.w4[15:0]; cols <= hc.w4[31:16];
          drs <= hc.w5[15:0]; ars <= hc.w5[31:16];
          brs <= hc.w6[15:0]; func <= hf; bmode <= hc.w6[25:24];
          imm <= hc.w7;
          nslots <= n_slots(hf);
          c16 = hc.w4[31:16];
          nchunks = comp_f(hf) ? (c16 + 16'(NCL) - 1) >> CLW : (c16 + 16'(LANES) - 1) >> LW;
          if (hf == V_RSUM || hf == V_RSSQ)
            nchunks = ((nchunks + 16'(RL) - 1) / 16'(RL)) * 16'(RL);
          nch <= nchunks;
          ir <= '0; ic <= '0; ch <= '0;
          a_row <= hc.w2; b_row <= hc.w3; d_row <= hc.w1;
          tag <= tag + 1;
          issuing <= 1'b1;
          st_frz <= '0;
          if (red_f(hf)) begin
            red_act <= 1'b1;
            wr_row <= hc.w1; rows_done <= '0;
            mx_have <= 1'b0;
          end else begin
            ewn = ewn + 1;
            last_tap <= n_slots(hf);
          end
        end
      end
      if (en) begin
        // ---- issue the next chunk (its data arrives next cycle, described by mi)
        mi <= '0;
        if (issuing) begin
          mi.v <= 1'b1;
          mi.tag <= tag;
          mi.func <= func;
          mi.bmode <= bmode;
          mi.imm <= imm;
          mi.mask <= imask;
          mi.waddr <= is_red ? d_row : d_row + 32'(ic);
          mi.row_last <= irow_last;
          mi.all_last <= iall_last;
          mi.first <= (ch < 16'(RL));
          mi.final_ <= (ch + 16'(RL) >= nch);
          mi.sub <= 8'(ch % 16'(RL));
          if (irow_last) begin
            ic <= '0; ch <= '0;
            a_row <= a_row + 32'(ars);
            b_row <= b_row + 32'(brs);
            d_row <= d_row + 32'(drs);
            if (ir + 1 == rows) issuing <= 1'b0;
            else ir <= ir + 1;
          end else begin
            ic <= ic + iwid;
            ch <= ch + 1;
          end
        end
        // ---- elementwise: an instruction's last chunk is written
        if (mo.v && mo.all_last) begin
          ewfin = 1'b1;
          ewn = ewn - 1;
        end
        // ---- RMAX
        if (red_act && func == V_RMAX && live(mxm_q, tag)) begin
          if (mxm_q.row_last) mx_have <= 1'b0;
          else begin
            mx_run <= mx_new;
            mx_have <= 1'b1;
          end
          if (mxm_q.all_last) fin = 1'b1;
        end
        // ---- RSUM/RSSQ: a row's sum is written this cycle (see TMEM writes)
        if (red_act && is_sum && root_v) begin
          wr_row <= wr_row + 32'(drs);
          rows_done <= rows_done + 1;
          if (rows_done + 1 == rows) fin = 1'b1;
        end
      end
      ew_n <= ewn;
      cq_n <= qn;
      if (fin) red_act <= 1'b0;
      if (fin || ewfin) begin
        done_i <= 1'b1;
`ifndef SYNTHESIS
        if (trace) $display("T%0d U c=%0d u=3 frz=%0d", SID, cyc, st_frz);
`endif
      end
    end
  end
`ifndef SYNTHESIS
  bit trace;
  initial trace = $test$plusargs("trace");
  // the tree schedule needs rows a multiple of RL cycles apart: level 0 (issuing while the
  // subs >= RL/2 are on `pacc`) must never meet another level
  always_ff @(posedge clk)
    if (!rst && en && cap && mt.sub >= 8'(RL / 2) && lv != 0)
      $fatal(1, "otpu_vpu: RSUM/RSSQ tree schedule collision");
  // lane level 2 shares level 1's adders: it must never issue with level 1
  always_ff @(posedge clk)
    if (!rst && en && SH2 && l2_v && rsr[DL + LA - 1])
      $fatal(1, "otpu_vpu: RSUM/RSSQ lane level schedule collision");
`endif
endmodule
