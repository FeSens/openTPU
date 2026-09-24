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
  output logic [LANES-1:0][31:0]  tw_data,
  // profiling: a cycle after an instruction ends, its cycles frozen by the TMEM grant
  output logic                    pf_u,
  output logic [31:0]             pf_frz
);
  localparam int LM = 2, LA = 4;
  localparam int SL = 1 + LM + LA;          // cycles per slot
  localparam int NSLOT = 10;
  localparam int NP = 64;                    // RSUM/RSSQ partials (isum_64)
  localparam int RL = NP / LANES;            // partial loop length in chunks
  localparam int LW = $clog2(LANES);
  localparam int NCL = (CL < LANES) ? CL : LANES;
  localparam int CLW = $clog2(NCL);
  // TMEM word address bits (the TMEM has far fewer words and $fatals past them); the ports
  // zero-extend to 32 bits
  localparam int AW = 20;
  initial if (RL < LA || NP % LANES != 0)
    $fatal(1, "otpu_vpu: LANES must be a power of two <= 16");

  localparam f32_t F_NZ = 32'h8000_0000;     // -0: (a*b) + -0 == a*b exactly

  // ------------------------------------------------------------------ command state
  logic        issuing, red_act;
  logic [AW-1:0] dst, a, b;
  logic [31:0] imm;
  logic [15:0] rows, cols, drs, ars, brs;
  logic [7:0]  func;
  logic [1:0]  bmode;
  logic [15:0] ir, ic, nch, ch;               // issue: row, column, chunks per row, chunk
  logic [AW-1:0] a_row, b_row, d_row;         // row base addresses (no multipliers)
  logic [3:0]  nslots;                         // slots of this function
  logic [7:0]  tag;                            // instruction tag
  // cycles an instruction was frozen by the TMEM grant, for the profiler: the condition is
  // registered (st_c) and summed a cycle late, off the grant path; the count is st_frz + st_c
  // (an instruction begins on a granted cycle, so a begin never drops a pending frozen cycle
  // of its own)
  logic [31:0] st_frz;
  logic        st_c;

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

  // the classes of functions the lane taps after slot 0 tell apart (slot 0 reads m0's func)
  localparam logic [2:0] C_OTH = 3'd0, C_ONE = 3'd1, C_EXP = 3'd2, C_RCP = 3'd3, C_RSQ = 3'd4;
  function automatic logic [2:0] f_cls(input logic [7:0] f);
    case (f)
      V_ADD, V_SUB, V_RSUB, V_MUL: return C_ONE;
      V_EXP2, V_EXP2SUB:           return C_EXP;
      V_RECIP:                     return C_RCP;
      V_RSQRT:                     return C_RSQ;
      default:                     return C_OTH;    // the reductions too
    endcase
  endfunction
  // a function of the class: the same as each of its functions at every slot s >= 1 (EXP2SUB
  // differs from EXP2 at slot 0 only); C_OTH's has no slots
  function automatic logic [7:0] cls_f(input logic [2:0] c);
    case (c)
      C_ONE:   return V_ADD;
      C_EXP:   return V_EXP2;
      C_RCP:   return V_RECIP;
      C_RSQ:   return V_RSQRT;
      default: return V_COPY;
    endcase
  endfunction

  // TMEM word address base + c + l, AW bits, zero-extended to the port
  function automatic logic [31:0] tm_addr(input logic [AW-1:0] base, input logic [15:0] c,
                                          input int l);
    logic [AW-1:0] t;
    t = base + AW'(c) + AW'(l);
    return 32'(t);
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
          ta_addr[l] = tm_addr(a_row, ic, l);
          if (is_binary(func)) case (bmode)
            B_FULL: begin
              tb_en[l] = 1'b1;
              tb_addr[l] = tm_addr(b_row, ic, l);
            end
            B_COL: begin
              tb_en[l] = 1'b1;
              tb_addr[l] = tm_addr(b, ic, l);
            end
            default: ;
          endcase
        end
      end
      if (bmode == B_ROW && is_binary(func) && imask[0]) begin
        tb_en[0] = 1'b1;
        tb_addr[0] = 32'(b_row);
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
    logic [AW-1:0]    waddr;     // elementwise: dst row + column; reductions: dst row
    logic             row_last;  // last chunk of its row
    logic             all_last;  // last chunk of the instruction
    logic             first;     // reductions: chunk index < RL (partials start at +0)
    logic             final_;    // reductions: chunk index >= nch - RL (partials become final)
    logic [7:0]       sub;       // chunk index mod RL
    logic             sq;        // func == V_RSSQ (the squarer's operand select, from a flip-flop)
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
    logic [2:0]       cls;       // f_cls(func)
    logic [LANES-1:0] mask;
    logic [AW-1:0]    waddr;
    logic             all_last;
  } mt_t;
  typedef struct packed {
    logic             v;
    logic [7:0]       tag;
    logic [AW-1:0]    waddr;
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

  // RSUM/RSSQ on the lanes' slot-0 multiply-add when the partial loop (RL cycles) holds the
  // input register, the multiplier and the adder plus a feedback flip-flop (DF of them);
  // otherwise (RL = 4) a squarer and an adder of their own per lane (see RSUM / RSSQ)
  localparam bit RMA = (RL >= 2 + LM + LA);
  localparam int DF  = RL - (1 + LM + LA);
  f32_t  pacc [LANES];                  // partial after adding this chunk's term
  f32_t  rsa [LANES], rsb [LANES];      // RMA: the term's factors, at m0
  f32_t  rfb [LANES];                   // RMA: the partial RL chunks ago (+0 for the first)

  assign mtap[0] = '{v: m0.v, cls: f_cls(m0.func), mask: m0.mask, waddr: m0.waddr,
                     all_last: m0.all_last};
  // after slot 0 the lines carry the mask bits of lanes < NCL only (the others read 0): only
  // the composite functions get past tap 1, and they are issued on those lanes (iwid). Tap 1
  // keeps the whole mask.
  localparam int MTW = $bits(mt_t) - (LANES - NCL);
  function automatic logic [MTW-1:0] mt_pk(input mt_t m);
    return {m.v, m.cls, m.mask[NCL-1:0], m.waddr, m.all_last};
  endfunction
  function automatic mt_t mt_up(input logic [MTW-1:0] p);
    mt_t m;
    m = '0;
    {m.v, m.cls, m.mask[NCL-1:0], m.waddr, m.all_last} = p;
    return m;
  endfunction
  // slot 1 has three extra input stages for the EXP2 range reduction (clamp, floor, i2f)
  localparam int PRE1 = 3;
  mt_t   msl [NSLOT];                        // meta at each slot's input mux
  for (genvar s = 0; s < NSLOT; s++) begin : g_mdel
    if (s == 1) begin : g_pre
      logic [MTW-1:0] q;
      otpu_delay #(.W(MTW), .N(PRE1)) u_p (.clk, .en, .d(mt_pk(mtap[s])), .q(q));
      assign msl[s] = mt_up(q);
    end else begin : g_nopre
      assign msl[s] = mtap[s];
    end
    if (s == 0) begin : g_d0
      otpu_delay #(.W($bits(mt_t)), .N(SL)) u_d (.clk, .en, .d(msl[s]), .q(mtap[s + 1]));
    end else begin : g_dn
      logic [MTW-1:0] q;
      otpu_delay #(.W(MTW), .N(SL)) u_d (.clk, .en, .d(mt_pk(msl[s])), .q(q));
      assign mtap[s + 1] = mt_up(q);
    end
  end

  // the taps where functions end: 0 (MAX..FILL), T_EW (ADD/SUB/RSUB/MUL), T_RC, T_EX, T_RS
  localparam int T_EW = n_slots(V_ADD), T_RC = n_slots(V_RECIP), T_EX = n_slots(V_EXP2),
                 T_RS = n_slots(V_RSQRT);
  function automatic logic end_tap(input int s);
    return s == 0 || s == T_EW || s == T_RC || s == T_EX || s == T_RS;
  endfunction

  // which boundary holds an entry at its function's last slot (at most one: see header); each
  // end tap is tied to its functions, so the lanes read every result from a fixed tap. Past
  // tap 0 the class decides (the reductions' C_OTH has no slots).
  logic hit [NSLOT + 1];
  mt_t  mo;
  always_comb begin
    mo = '0;
    for (int s = 0; s <= NSLOT; s++) begin
      if (s == 0) hit[s] = mtap[s].v && !red_f(m0.func) && n_slots(m0.func) == 4'd0;
      else        hit[s] = end_tap(s) && mtap[s].v && n_slots(cls_f(mtap[s].cls)) == 4'(s);
      if (hit[s]) mo = mtap[s];
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
          if (mtap[s].cls == C_EXP) begin
            lo = fp_gt(F_M126, t.v);
            hi = !fp_gt(F_128, t.v);
            t.v = (lo || hi) ? F_ZERO : t.v;
            t.f = {1'b0, hi, lo};
          end
          p0 <= t;
          e1 <= (mtap[s].cls == C_EXP);
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
      // the slot's function: m0's at slot 0, the class's after it
      logic [7:0] sf;
      assign sf = (s == 0) ? m0.func : cls_f(msl[s].cls);
      always_comb begin
        lst_t t;
        t = sti;
        ia = F_ZERO; ib = F_ONE; ic_ = F_NZ; dest = 2'd0; negd = 1'b0;
        case (sf)
          V_MUL:  if (s == 0) begin ia = x; ib = y; dest = 2'd1; end
          V_ADD:  if (s == 0) begin ia = x; ic_ = y; dest = 2'd1; end
          V_SUB:  if (s == 0) begin ia = x; ic_ = fneg(y); dest = 2'd1; end
          V_RSUB: if (s == 0) begin ia = y; ic_ = fneg(x); dest = 2'd1; end
          V_RSUM, V_RSSQ: if (s == 0 && RMA) begin ia = rsa[l]; ib = rsb[l]; ic_ = rfb[l]; end
          V_EXP2, V_EXP2SUB: begin
            if (s == 0) begin
              ia = x; ic_ = (sf == V_EXP2SUB) ? fneg(y) : F_NZ; dest = 2'd1;
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
        ra <= ia; rb <= ib; sr <= sin_; dr <= dest; nr <= negd;
      end
      // rc has a sync reset so it can't join u_c's two stages in a shift-register LUT: the
      // adder's c operand then leaves a flip-flop (clk->Q 0.3 ns) instead of an SRL (1.5 ns).
      // F_NZ is every slot's default c, so bits constant across a slot's cases stay constant.
      always_ff @(posedge clk)
        if (rst)     rc <= F_NZ;
        else if (en) rc <= ic_;
      otpu_fmadd #(.LM(LM), .LA(LA)) u_ma (.clk, .en, .a(ra), .b(rb), .c(rc), .y(r));
      if (s == 0 && RMA) begin : g_pacc
        assign pacc[l] = r;
      end
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

    // the result of the entry that ends at a tap this cycle: each composite function's result
    // from its own tap (EXP2 T_EX, RECIP T_RC, RSQRT T_RS), the others' v from tap 0 or T_EW
    f32_t ex_r, rc_r, rs_r;
    lst_t te, tc, ts;
    assign te = st[T_EX];
    assign tc = st[T_RC];
    assign ts = st[T_RS];
    assign ex_r = te.f[0] ? F_ZERO : te.f[1] ? F_INF : (te.v + {te.ii, 23'd0});
    assign rc_r = tc.f[0] ? F_ZERO : tc.f[1] ? {tc.f[2], 31'd0} : (tc.f[2] ? fneg(tc.k2) : tc.k2);
    assign rs_r = ts.f[0] ? F_ZERO : ts.k2;
    assign lres[l] = hit[T_EX] ? ex_r : hit[T_RC] ? rc_r : hit[T_RS] ? rs_r :
                     hit[0] ? st[0].v : st[T_EW].v;
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
      case (m0.func)
        V_MUL:  begin ia = x; ib = y; end
        V_ADD:  begin ia = x; ic_ = y; end
        V_SUB:  begin ia = x; ic_ = fneg(y); end
        V_RSUB: begin ia = y; ic_ = fneg(x); end
        V_RSUM, V_RSSQ: if (RMA) begin ia = rsa[l]; ib = rsb[l]; ic_ = rfb[l]; end
        default: ;
      endcase
    end
    always_ff @(posedge clk) if (en) begin
      ra <= ia; rb <= ib;
    end
    // sync reset keeps rc out of u_c's SRL (see g_slot)
    always_ff @(posedge clk)
      if (rst)     rc <= F_NZ;
      else if (en) rc <= ic_;
    otpu_fmadd #(.LM(LM), .LA(LA)) u_ma (.clk, .en, .a(ra), .b(rb), .c(rc), .y(st[1]));
    if (RMA) begin : g_pacc
      assign pacc[l] = st[1];
    end

    // the result of the entry that ends at tap 0 or 1 this cycle (other taps: masked here)
    assign lres[l] = hit[0] ? st[0] : st[1];
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
  // RMA: the term enters the lane's slot 0 through its input register (ra/rb/rc), so it reaches
  // the adder one cycle later than a squarer of its own would
  rm_t   mtq, mt;                       // meta at the adder inputs / aligned with `pacc`
  otpu_delay #(.W($bits(rm_t)), .N(RMA ? LM + 1 : LM)) u_mtq (.clk, .en, .d(m0r), .q(mtq));
  otpu_delay #(.W($bits(rm_t)), .N(LA)) u_mt (.clk, .en, .d(mtq), .q(mt));
  logic  first_e;                       // !RMA: mtq.first one cycle early
  if (RMA) begin : g_nfe
    assign first_e = 1'b0;
  end else begin : g_fe
    otpu_delay #(.W(1), .N(LM - 1)) u_first (.clk, .en, .d(m0.first), .q(first_e));
  end
  for (genvar l = 0; l < LANES; l++) begin : g_red
    f32_t sa, sb;
    // sign/exponent masked; the significand raw from xa (the multiplier flushes a zero exponent
    // itself and reads the significand only when both exponents are normal), so the term is
    // the same as that of m0.mask[l] ? ftz(xa[l]) : +0
    assign sa = {m0.mask[l] & xa[l][31], m0.mask[l] ? xa[l][30:23] : 8'd0, xa[l][22:0]};
    assign sb = m0.sq ? sa : F_ONE;
    assign rsa[l] = sa;
    assign rsb[l] = sb;
    // pacc(chunk c) = pacc(chunk c - RL) + term(c): a loop of exactly RL cycles
    if (RMA) begin : g_ma
      // through slot 0's u_ma (term + prev: the add is commutative bit for bit), fed back by
      // fbq, which slot 0 reads (as c) the cycle after it's written: cleared on the chunks that
      // are on m0 then (mi), the first ones' +0. rc keeps the adder's c off a shift register.
      f32_t fbq, fbd;
      otpu_delay #(.W(32), .N(DF - 1)) u_fb (.clk, .en, .d(pacc[l]), .q(fbd));
      always_ff @(posedge clk) if (en) fbq <= mi.first ? F_ZERO : fbd;
      assign rfb[l] = fbq;
    end else begin : g_acc
      f32_t tq, prev;
      assign rfb[l] = F_ZERO;
      otpu_fmul #(.LAT(LM)) u_sq (.clk, .en, .a(sa), .b(sb), .y(tq));
      if (RL - LA >= 1) begin : g_fbq
        // the loop's last stage is a flip-flop with a sync clear (the first chunks' +0): no
        // shift-register LUT and no mux in front of the adder
        f32_t fbq, fbd;
        otpu_delay #(.W(32), .N(RL - LA - 1)) u_fb (.clk, .en, .d(pacc[l]), .q(fbd));
        always_ff @(posedge clk) if (en) fbq <= first_e ? F_ZERO : fbd;
        assign prev = fbq;
      end else begin : g_fbw
        f32_t fb;
        otpu_delay #(.W(32), .N(RL - LA)) u_fb (.clk, .en, .d(pacc[l]), .q(fb));
        assign prev = mtq.first ? F_ZERO : fb;
      end
      otpu_fadd #(.LAT(LA)) u_acc (.clk, .en, .a(prev), .b(tq), .y(pacc[l]));
    end
  end

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

  // a final partial of this reduction is on `pacc`
  wire cap = red_act && is_sum && live(mt, tag) && mt.final_;

  // level strobes: cap of the subs a level pairs up, delayed to the level's issue cycles
  // (level 0 is the default); the last row level and the root share one shift register
  logic [LR-1:1] lv, lvn;                     // lvn: lv one cycle early (TP)
  logic [RD-1:0] rsr;
  always_ff @(posedge clk)
    if (rst) rsr <= '0;
    else if (en) rsr <= {rsr[RD-2:0], cap && mt.row_last};
  assign lv[LR-1] = rsr[TP ? DL - 2 : DL - 1];
  assign lvn[LR-1] = 1'b0;
  wire root_v = rsr[RD-1];
  for (genvar j = 1; j < LR - 1; j++) begin : g_lv
    localparam int DJ = tree_i(j) - (RL - tree_h(j));
    logic [DJ-1:0] sr;
    always_ff @(posedge clk)
      if (rst) sr <= '0;
      else if (en) sr <= {sr[DJ-2:0], cap && mt.sub >= 8'(RL - tree_h(j))};
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

  // rows finish in order: the next root goes to wr_row
  logic [AW-1:0] wr_row;
  logic [15:0]   rows_done;

  // ------------------------------------------------------------------ TMEM writes
  // The writes are computed here (cw_*) and registered (tw_*): a TMEM write is performed one
  // granted cycle after the cycle that produced it, so no path runs from the TMEM read data
  // through the lanes into the TMEM write port. `done` follows its instruction's last write.
  logic [LANES-1:0]         cw_en;
  logic [LANES-1:0][AW-1:0] cw_addr, tw_a;
  logic [LANES-1:0][31:0]   cw_data;
  logic                     done_i, dpend;
  for (genvar l = 0; l < LANES; l++) begin : g_twa
    assign tw_addr[l] = 32'(tw_a[l]);
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      tw_en <= '0;
      done <= 1'b0; dpend <= 1'b0;
    end else begin
      if (gnt) begin
        tw_en <= cw_en; tw_a <= cw_addr; tw_data <= cw_data;
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
          cw_addr[l] = mo.waddr + AW'(l);
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
      cw_data[0] = LN2 ? xs[0] : xl[LW][0];
    end
  end

  // ------------------------------------------------------------------ sequencing
  always_ff @(posedge clk) begin
    logic fin, ewfin;
    done_i <= 1'b0;
    pf_u <= 1'b0;
    fin = 1'b0;
    ewfin = 1'b0;
    st_c <= busy && !en;
    st_frz <= st_frz + 32'(st_c);
    if (rst) begin
      cq_n <= '0; cq_h <= 1'b0;
      issuing <= 1'b0; red_act <= 1'b0;
      ew_n <= '0; last_tap <= '0;
      mi <= '0;
      tag <= '0;
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
          dst <= AW'(hc.w1); a <= AW'(hc.w2); b <= AW'(hc.w3);
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
          a_row <= AW'(hc.w2); b_row <= AW'(hc.w3); d_row <= AW'(hc.w1);
          tag <= tag + 1;
          issuing <= 1'b1;
          st_frz <= '0;
          if (red_f(hf)) begin
            red_act <= 1'b1;
            wr_row <= AW'(hc.w1); rows_done <= '0;
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
          mi.waddr <= is_red ? d_row : d_row + AW'(ic);
          mi.row_last <= irow_last;
          mi.all_last <= iall_last;
          mi.first <= (ch < 16'(RL));
          mi.final_ <= (ch + 16'(RL) >= nch);
          mi.sub <= 8'(ch % 16'(RL));
          mi.sq <= (func == V_RSSQ);
          if (irow_last) begin
            ic <= '0; ch <= '0;
            a_row <= a_row + AW'(ars);
            b_row <= b_row + AW'(brs);
            d_row <= d_row + AW'(drs);
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
          wr_row <= wr_row + AW'(drs);
          rows_done <= rows_done + 1;
          if (rows_done + 1 == rows) fin = 1'b1;
        end
      end
      ew_n <= ewn;
      cq_n <= qn;
      if (fin) red_act <= 1'b0;
      if (fin || ewfin) begin
        done_i <= 1'b1;
        pf_u <= 1'b1;
        pf_frz <= st_frz + 32'(st_c);
      end
    end
  end
`ifndef SYNTHESIS
  // the tree schedule needs rows a multiple of RL cycles apart: level 0 (issuing while the
  // subs >= RL/2 are on `pacc`) must never meet another level
  always_ff @(posedge clk)
    if (!rst && en && cap && mt.sub >= 8'(RL / 2) && lv != 0)
      $fatal(1, "otpu_vpu: RSUM/RSSQ tree schedule collision");
  // lane level 2 shares level 1's adders: it must never issue with level 1 (LN2: the lane
  // levels L1a, L1b, L2 and L3 share u_ln and must never issue together)
  always_ff @(posedge clk)
    if (!rst && en && (LN2 ? !$onehot0({rsr[DL + LA - 1], ln_l1b, ln_l2, ln_l3})
                           : SH2 && l2_v && rsr[DL + LA - 1]))
      $fatal(1, "otpu_vpu: RSUM/RSSQ lane level schedule collision");
`endif
endmodule
