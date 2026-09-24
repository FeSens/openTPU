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
// the 64 final partials are captured into one of two buffers and summed by a folding tree
// (LANES adders) while the next row streams.
//
// The stream (reads, lane pipelines, stream writes) advances on cycles the TMEM grant is given
// and the tree is not holding it back; the tree advances whenever the grant is given.
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
  logic        issuing, red_act, stall;
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
  wire  cmd_t hc = cq[cq_h];
  wire  [7:0] hf = hc.w6[23:16];
  wire  h_empty = (hc.w4[15:0] == 0) || (hc.w4[31:16] == 0);
  wire  can_begin = (cq_n != 0) && !issuing && !red_act &&
                    ((red_f(hf) || h_empty) ? (ew_n == 0) :
                                              (ew_n == 0 || n_slots(hf) >= last_tap));
  wire  busy = issuing || red_act || ew_n != 0;


  wire en  = gnt && !stall;                  // the stream
  wire ent = gnt;                            // the reduction tree

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
    if (stall) begin
      ta_en = '0; tb_en = '0;
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
  meta_t m0;

  function automatic logic live(input meta_t m, input logic [7:0] t);
    return m.v && m.tag == t;
  endfunction

  // ------------------------------------------------------------------ elementwise lanes
  typedef struct packed {
    f32_t        v, k1, k2;
    logic [8:0]  ii;
    logic [2:0]  f;
  } lst_t;

  meta_t mtap [NSLOT + 1];
  f32_t  lres [LANES];

  assign mtap[0] = m0;
  // slot 1 has three extra input stages for the EXP2 range reduction (clamp, floor, i2f)
  localparam int PRE1 = 3;
  meta_t msl [NSLOT];                        // meta at each slot's input mux
  for (genvar s = 0; s < NSLOT; s++) begin : g_mdel
    if (s == 1) begin : g_pre
      otpu_delay #(.W($bits(meta_t)), .N(PRE1)) u_p (.clk, .en, .d(mtap[s]), .q(msl[s]));
    end else begin : g_nopre
      assign msl[s] = mtap[s];
    end
    otpu_delay #(.W($bits(meta_t)), .N(SL)) u_d (.clk, .en, .d(msl[s]), .q(mtap[s + 1]));
  end

  // which boundary holds an entry at its function's last slot (at most one: see header)
  logic [3:0] wtap;
  meta_t      mo;
  always_comb begin
    wtap = '0;
    mo = '0;
    for (int s = NSLOT; s >= 0; s--)
      if (mtap[s].v && !red_f(mtap[s].func) && n_slots(mtap[s].func) == 4'(s)) begin
        wtap = 4'(s);
        mo = mtap[s];
      end
  end

  for (genvar l = 0; l < LANES; l++) begin : g_lane
    localparam int NS = (l < NCL) ? NSLOT : 1;     // slots of this lane
    f32_t x, y;
    lst_t st [NS + 1];
    assign x = ta_data[l];
    assign y = (m0.bmode == B_SCALAR) ? m0.imm : (m0.bmode == B_ROW) ? tb_data[0] : tb_data[l];

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
      otpu_delay #(.W($bits(lst_t)), .N(LM + LA)) u_st (.clk, .en, .d(sr), .q(sd));
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
      if (NS == NSLOT) t = st[wtap];
      else t = (wtap == 0) ? st[0] : st[NS];       // short lane: the simple functions only
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

  // ------------------------------------------------------------------ RMAX
  // pairwise max tree over the lanes of the chunk (masked lanes drop out), in two registered
  // stages: the first level (from the TMEM read data), then the rest
  localparam int HL = LANES / 2;
  f32_t  mxh_v [HL];
  logic  mxh_h [HL];
  meta_t mxh_m;
  f32_t  mxc_q;
  meta_t mxm_q;
  f32_t  mx_run;
  logic  mx_have;
  always_ff @(posedge clk) if (en) begin
    f32_t v [LANES];
    logic h [LANES];
    for (int l = 0; l < LANES; l++) begin
      v[l] = ftz(ta_data[l]);
      h[l] = m0.mask[l];
    end
    for (int l = 0; l < HL; l++) begin
      mxh_v[l] <= (!h[l] || (h[l + HL] && fp_gt(v[l + HL], v[l]))) ? v[l + HL] : v[l];
      mxh_h[l] <= h[l] || h[l + HL];
    end
    mxh_m <= m0;
  end
  always_ff @(posedge clk) if (en) begin
    f32_t v [HL];
    logic h [HL];
    for (int l = 0; l < HL; l++) begin
      v[l] = mxh_v[l];
      h[l] = mxh_h[l];
    end
    for (int w = HL / 2; w >= 1; w = w / 2) begin
      for (int l = 0; l < w; l++) begin
        if (!h[l] || (h[l + w] && fp_gt(v[l + w], v[l]))) v[l] = v[l + w];
        h[l] = h[l] || h[l + w];
      end
    end
    mxc_q <= v[0];
    mxm_q <= mxh_m;
  end
  f32_t mx_new;
  assign mx_new = (mx_have && fp_gt(mx_run, mxc_q)) ? mx_run : mxc_q;

  // ------------------------------------------------------------------ RSUM / RSSQ
  f32_t  pacc [LANES];                  // partial after adding this chunk's term
  meta_t mtq, mt;                       // meta at the adder inputs / aligned with `pacc`
  otpu_delay #(.W($bits(meta_t)), .N(LM)) u_mtq (.clk, .en, .d(m0), .q(mtq));
  otpu_delay #(.W($bits(meta_t)), .N(LA)) u_mt (.clk, .en, .d(mtq), .q(mt));
  for (genvar l = 0; l < LANES; l++) begin : g_red
    f32_t xin, tq, prev, fb;
    assign xin = m0.mask[l] ? ftz(ta_data[l]) : F_ZERO;
    otpu_fmul #(.LAT(LM)) u_sq (.clk, .en, .a(xin), .b((m0.func == V_RSSQ) ? xin : F_ONE), .y(tq));
    // pacc(chunk c) = pacc(chunk c - RL) + term(c): a loop of exactly RL cycles
    otpu_delay #(.W(32), .N(RL - LA)) u_fb (.clk, .en, .d(pacc[l]), .q(fb));
    assign prev = mtq.first ? F_ZERO : fb;
    otpu_fadd #(.LAT(LA)) u_acc (.clk, .en, .a(prev), .b(tq), .y(pacc[l]));
  end

  // Final partials of a row are captured into one of NTB buffers; each buffer runs its own
  // folding tree (level n = 32, 16, .., 1: x[i] += x[i+n], i < n) and the trees of different
  // rows interleave on the LANES tree adders (one buffer issues up to LANES adds per cycle).
  // Partial p sits in row p / LANES, lane p % LANES: levels n >= LANES add rows of the same
  // lane, levels n < LANES add lanes of row 0, so tree adder k only reads lane k (and lane
  // k + n) and only writes lane k -- narrow muxes instead of any-of-64 reads.
  localparam int NTB = 4;
  localparam int TBW = $clog2(NTB);
  f32_t          pb [NTB][RL][LANES];
  logic [NTB-1:0] tb_act;                 // captured, tree in progress
  logic [6:0]    tb_n [NTB], tb_i [NTB];   // pairs in the level, next pair
  logic [2:0]    tb_inf [NTB];             // issue cycles in flight
  logic [31:0]   tb_dst [NTB];
  logic [TBW-1:0] cap_sel;
  logic [15:0]   rows_done;

  // pick the lowest-numbered buffer that can issue
  logic          tr_go;
  logic [TBW-1:0] tr_b;
  always_comb begin
    tr_go = 1'b0; tr_b = '0;
    for (int b = NTB - 1; b >= 0; b--)
      if (tb_act[b] && tb_i[b] < tb_n[b] && !(tb_i[b] == 0 && tb_inf[b] != 0)) begin
        tr_go = 1'b1; tr_b = TBW'(b);
      end
  end
  // issue register: the selected buffer and its pair range; pb is read the next cycle (the
  // issue counts as in flight from selection, so a level never reads before its inputs land)
  logic          is_go;
  logic [TBW-1:0] is_b;
  logic [6:0]    is_i, is_n;
  f32_t          tr_y [LANES];
  logic [LANES-1:0] tr_m;
  logic [6:0]    tr_dst_q [LA];
  logic [TBW-1:0] tr_b_q [LA];
  logic [LANES-1:0] tr_v_q [LA];
  for (genvar k = 0; k < LANES; k++) begin : g_tree
    logic [6:0] ia, ib;
    assign ia = is_i + 7'(k);
    assign ib = ia + is_n;
    assign tr_m[k] = is_go && (ia < is_n);
    f32_t oa, ob;
    assign oa = pb[is_b][ia[5:0] >> LW][k];
    assign ob = (is_n >= 7'(LANES)) ? pb[is_b][ib[5:0] >> LW][k]
                                    : pb[is_b][0][ib[LW-1:0]];
    otpu_fadd #(.LAT(LA)) u_tree (.clk, .en(ent), .a(oa), .b(ob), .y(tr_y[k]));
  end

  // finished rows wait here for the (single) TMEM write lane
  f32_t          rq_v [NTB];
  logic [31:0]   rq_a [NTB];
  logic [TBW-1:0] rq_h;
  logic [TBW:0]  rq_n;

  // the last RL chunks of a row deliver its final partials: hold the stream if the buffer
  // they go to still runs a tree
  assign stall = red_act && is_sum && live(mt, tag) && mt.final_ && tb_act[cap_sel];

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
    if (mo.v && !stall) begin
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
    if (red_act && is_sum && rq_n != 0) begin
      cw_en[0] = 1'b1;
      cw_addr[0] = rq_a[rq_h];
      cw_data[0] = rq_v[rq_h];
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
      m0 <= '0;
      tag <= '0;
      cyc <= '0;
      tb_act <= '0;
      rq_n <= '0;
      for (int k = 0; k < LA; k++) tr_v_q[k] <= '0;
      is_go <= 1'b0;
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
            tb_act <= '0; cap_sel <= '0; rq_n <= '0; rq_h <= '0; rows_done <= '0;
            for (int bb = 0; bb < NTB; bb++) tb_inf[bb] <= '0;
            mx_have <= 1'b0;
            for (int k = 0; k < LA; k++) tr_v_q[k] <= '0;
            is_go <= 1'b0;
          end else begin
            ewn = ewn + 1;
            last_tap <= n_slots(hf);
          end
        end
      end
      if (en) begin
        // ---- issue the next chunk (its data arrives next cycle, described by m0)
        m0 <= '0;
        if (issuing) begin
          m0.v <= 1'b1;
          m0.tag <= tag;
          m0.func <= func;
          m0.bmode <= bmode;
          m0.imm <= imm;
          m0.mask <= imask;
          m0.waddr <= is_red ? d_row : d_row + 32'(ic);
          m0.row_last <= irow_last;
          m0.all_last <= iall_last;
          m0.first <= (ch < 16'(RL));
          m0.final_ <= (ch + 16'(RL) >= nch);
          m0.sub <= 8'(ch % 16'(RL));
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
        // ---- RSUM/RSSQ: capture the final partials of a row
        if (red_act && is_sum && live(mt, tag) && mt.final_) begin
          for (int l = 0; l < LANES; l++) pb[cap_sel][mt.sub[5:0]][l] <= pacc[l];
          if (mt.row_last) begin
            tb_act[cap_sel] <= 1'b1;
            tb_n[cap_sel] <= 7'(NP / 2);
            tb_i[cap_sel] <= '0;
            tb_dst[cap_sel] <= mt.waddr;
            cap_sel <= cap_sel + 1;
          end
        end
      end
      if (ent && red_act && is_sum) begin
        logic [TBW:0] rn;
        logic pushed;
        rn = rq_n;
        pushed = 1'b0;
        // ---- tree adds
        is_go <= tr_go;
        is_b <= tr_b;
        is_i <= tb_i[tr_b];
        is_n <= tb_n[tr_b];
        tr_v_q[0] <= tr_m;
        tr_dst_q[0] <= is_i;
        tr_b_q[0] <= is_b;
        for (int k = 1; k < LA; k++) begin
          tr_v_q[k] <= tr_v_q[k-1];
          tr_dst_q[k] <= tr_dst_q[k-1];
          tr_b_q[k] <= tr_b_q[k-1];
        end
        for (int k = 0; k < LANES; k++)
          if (tr_v_q[LA-1][k]) pb[tr_b_q[LA-1]][tr_dst_q[LA-1][5:0] >> LW][k] <= tr_y[k];
        for (int bb = 0; bb < NTB; bb++) begin
          logic [2:0] inf;
          inf = tb_inf[bb];
          if (tr_go && tr_b == TBW'(bb)) begin
            inf = inf + 1;
            tb_i[bb] <= tb_i[bb] + 7'(LANES);
          end
          if (tr_v_q[LA-1] != 0 && tr_b_q[LA-1] == TBW'(bb)) inf = inf - 1;
          tb_inf[bb] <= inf;
          // ---- level done: next level, or the root is ready (one per cycle)
          if (tb_act[bb] && tb_i[bb] >= tb_n[bb] && tb_inf[bb] == 0 &&
              !(tr_go && tr_b == TBW'(bb))) begin
            if (tb_n[bb] != 7'd1) begin
              tb_n[bb] <= tb_n[bb] >> 1;
              tb_i[bb] <= '0;
            end else if (!pushed && 32'(rn) < NTB) begin
              pushed = 1'b1;
              rq_v[TBW'(rq_h + rn)] <= pb[bb][0][0];
              rq_a[TBW'(rq_h + rn)] <= tb_dst[bb];
              rn = rn + 1;
              tb_act[bb] <= 1'b0;
            end
          end
        end
        // ---- a finished row is written this cycle (see TMEM writes)
        if (rq_n != 0) begin
          rq_h <= rq_h + 1;
          rn = rn - 1;
          rows_done <= rows_done + 1;
          if (rows_done + 1 == rows) fin = 1'b1;
        end
        rq_n <= rn;
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
`endif
endmodule
