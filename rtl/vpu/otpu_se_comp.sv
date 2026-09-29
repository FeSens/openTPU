// The stream engine's composite functions (docs/stream.md section 11, SE v2): EXP2, EXP2SUB,
// RECIP, RSQRT and LOG2 on all LANES lanes, microcoded over NS generic multiply-add stages per
// lane (in SE: VPU slot 0, the tail's U and the tail's Q). Each function is its op list from
// docs/isa.md -- the same rounded mul-then-add steps, operands and constants as the VPU's
// 10-slot chains -- so it is bit-exact with opentpu/fp32.py by construction.
//
// A chunk (LANES columns of one instruction) runs in passes around a loop:
//
//   S0 -> RR -> S1 -> ... -> S(NS-1) -> loop register -> S0 (next pass) or out
//
// Op k of a function runs at stage k % NS of pass k / NS; stages past a function's last op run
// the identity v = v*1 + -0 (exact for every flushed value). RR is the EXP2 range reduction
// (range flags | floor | -i2f, three register stages, today's slot-1 pre-stages), applied on pass 0
// only (op 0 is at S0); LOG2 takes i2f(e) there. T = NS*SL + 4 cycles from S0 to S0.
//
//   func      ops  passes (NS = 3)   columns per cycle (LANES = 8)
//   EXP2       9   3                 2.67    (op 0 is v = x*1 + -0, then f, 7 Horner steps)
//   EXP2SUB    9   3                 2.67    (op 0 is v = x*1 + -y)
//   RECIP      6   2                 4
//   RSQRT     10   4                 2
//   LOG2      10   4                 2
//
// Stage s of a lane is a unit y = add(mul(a, b), c) (an otpu_fmma with e = 1.0, or an
// otpu_fmul then otpu_fadd) whose operands (st_a, st_b, st_c, st_e) are registered at the
// cycle they are presented and whose result (st_y) comes back SL = 1 + LM + LA cycles later.
// EXT = 1 leaves the units to their owners (in SE: VPU slot 0, the tail's U and Q), which take
// st_* into their input registers when st_sel[s]; EXT = 0 instantiates generic ones here.
// Everything this module computes besides the units is here: the per-lane setup (the seeds and
// the LOG2 split), RR, the state carried along the stages (k1, k2, ii, f; v is the result of
// the stage before, as in the chains), the operand muxes, the per-stage control (a ROM on the
// chunk's function and pass, shared by the lanes) and the final fix-ups.
//
// Issue: a chunk enters at S0 (in_v, with A and B per lane) unless a chunk returns to S0 that
// cycle. `hold` says so HA cycles ahead (it is a register tap: the loop is a fixed schedule),
// so the issuer (the VPU: read address -> mi -> m0 -> S0 is HA = 2) holds its read instead of
// buffering it. A chunk with P passes finishes (out_v, out_d, out_m, out_meta) P*T cycles after it
// entered; chunks finish in entry order as long as a chunk never enters while chunks with
// more passes are in flight (the VPU's latency rule: an instruction may start only if its
// latency is at least that of those in flight). Everything advances with `en`.
module otpu_se_comp
  import otpu_pkg::*;
  import otpu_fp::*;
#(
  parameter int LANES = 8,
  parameter int MW    = 64,       // the core's opaque chunk meta (out_meta = in_meta)
  parameter int NS    = 3,        // stages per lane
  parameter int HA    = 2,        // `hold` leads the S0 cycle it protects by HA cycles
  parameter bit EXT   = 1'b1      // the stage units are outside (st_*); 0: generic ones inside
) (
  input  logic             clk,
  input  logic             rst,
  input  logic             en,
  // entry at S0: a chunk of function in_f (V_EXP2, V_EXP2SUB, V_RECIP, V_RSQRT, V_LOG2)
  input  logic             in_v,
  input  logic [7:0]       in_f,
  input  f32_t             in_a [LANES],     // operand A
  input  f32_t             in_b [LANES],     // operand B (EXP2SUB)
  input  logic [LANES-1:0] in_m,             // lane mask (carried)
  input  logic [MW-1:0]    in_meta,
  output logic             hold,             // an entry HA cycles from now would collide
  // the stage units (EXT): operands at stage s's input mux this cycle (st_sel[s]), the
  // result SL cycles later; y = a*b + c*e (e = 1.0; the Q stage takes a*b + c)
  output logic [NS-1:0]    st_sel,
  output f32_t             st_a [NS][LANES],
  output f32_t             st_b [NS][LANES],
  output f32_t             st_c [NS][LANES],
  output f32_t             st_e [NS][LANES],
  input  f32_t             st_y [NS][LANES],
  // a finished chunk, n_pass(f) * T cycles after it entered (taken with en)
  output logic             out_v,
  output f32_t             out_d [LANES],
  output logic [LANES-1:0] out_m,
  output logic [MW-1:0]    out_meta
);
  localparam int LM = 2, LA = 4;
  localparam int SL = 1 + LM + LA;         // a stage: input registers, multiply, add
  localparam int RRL = 3;                  // RR: clamp | floor | i2f
  localparam int T = NS * SL + RRL + 1;    // S0 to S0 (the loop register is the 1)
  initial if (NS < 1 || HA < 1 || HA >= T)
    $fatal(1, "otpu_se_comp: unsupported NS / HA");

  localparam f32_t F_NZ = 32'h8000_0000;   // -0: (a*b) + -0 == a*b exactly

  // ---------------------------------------------------------------- functions and microcode
  localparam logic [2:0] CC_NONE = 3'd0, CC_EXP = 3'd1, CC_EXS = 3'd2, CC_RCP = 3'd3,
                         CC_RSQ = 3'd4, CC_LOG = 3'd5;
  function automatic logic [2:0] f_cc(input logic [7:0] f);
    case (f)
      V_EXP2:    return CC_EXP;
      V_EXP2SUB: return CC_EXS;
      V_RECIP:   return CC_RCP;
      V_RSQRT:   return CC_RSQ;
      V_LOG2:    return CC_LOG;
      default:   return CC_NONE;
    endcase
  endfunction
  function automatic int n_ops(input logic [2:0] c);
    case (c)
      CC_EXP, CC_EXS: return 9;
      CC_RCP:         return 6;
      CC_RSQ, CC_LOG: return 10;
      default:        return 1;
    endcase
  endfunction
  // (each case a constant, so no divider is built for a variable c)
  function automatic logic [3:0] n_pass(input logic [2:0] c);
    case (c)
      CC_EXP, CC_EXS: return 4'((9 + NS - 1) / NS);
      CC_RCP:         return 4'((6 + NS - 1) / NS);
      CC_RSQ, CC_LOG: return 4'((10 + NS - 1) / NS);
      default:        return 4'd1;
    endcase
  endfunction

  // One op: y = a*b + c into dst (negated with neg). The multiply is commutative bit for bit,
  // so each op puts its variable factor in a and b takes k1, k2 or a constant; a first Horner
  // coefficient is written into v by the op before (kv: v when dst is not v), so a never
  // takes a constant. At S0 a also takes the input A and c the negated input B.
  localparam logic [1:0] A_V = 2'd0, A_K1 = 2'd1, A_K2 = 2'd2, A_X = 2'd3;
  localparam logic [1:0] B_K1 = 2'd0, B_K2 = 2'd1, B_KB = 2'd2;
  localparam logic [1:0] C_KC = 2'd0, C_K2 = 2'd1, C_NY = 2'd2;
  localparam logic [1:0] D_NO = 2'd0, D_V = 2'd1, D_K1 = 2'd2, D_K2 = 2'd3;
  typedef struct packed {
    logic [1:0] as;
    logic [1:0] bs;
    logic [1:0] cs;
    logic [1:0] dst;
    logic       neg;
    f32_t       kb;
    f32_t       kc;
    f32_t       kv;
  } uc_t;

  function automatic f32_t exp2_c(input int i);
    case (i)
      0: return EXP2_C0;  1: return EXP2_C1;  2: return EXP2_C2;  3: return EXP2_C3;
      4: return EXP2_C4;  5: return EXP2_C5;  6: return EXP2_C6;  default: return EXP2_C7;
    endcase
  endfunction
  function automatic f32_t log2_c(input int i);
    case (i)
      1: return LOG2_C1;  2: return LOG2_C2;  3: return LOG2_C3;  4: return LOG2_C4;
      5: return LOG2_C5;  6: return LOG2_C6;  7: return LOG2_C7;  8: return LOG2_C8;
      default: return LOG2_C9;
    endcase
  endfunction

  // op k of function c (the VPU chains' slot k, factors in either order); k past the last op:
  // v = v*1 + -0
  function automatic uc_t ucode(input logic [2:0] c, input int k);
    uc_t u;
    u = '{as: A_V, bs: B_KB, cs: C_KC, dst: D_V, neg: 1'b0, kb: F_ONE, kc: F_NZ, kv: F_ZERO};
    case (c)
      CC_EXP, CC_EXS: begin
        if (k == 0) begin                                            // v = x (- y)
          u.as = A_X; u.cs = (c == CC_EXS) ? C_NY : C_KC;
        end else if (k == 1) begin                                   // f = xf - i2f(i); v = C7
          u.cs = C_K2; u.dst = D_K1; u.kv = EXP2_C7;
        end else if (k <= 8) begin                                   // p*f + C6 .. C0
          u.bs = B_K1; u.kc = exp2_c(8 - k);
        end
      end
      CC_RCP:
        if (k < 6) begin
          if (k % 2 == 0) begin                                      // t = 2 - |x| y
            u.as = A_K1; u.bs = B_K2; u.kc = F_TWO;
          end else begin                                             // y = t y (the last
            u.bs = B_K2; u.dst = (k == 5) ? D_V : D_K2;               // into v)
          end
        end
      CC_RSQ:
        if (k == 0) begin                                            // -h = -(x 0.5)
          u.as = A_X; u.kb = F_HALF; u.dst = D_K1; u.neg = 1'b1;
        end else if (k < 10) begin
          if ((k - 1) % 3 == 0) begin                                // y y
            u.as = A_K2; u.bs = B_K2;
          end else if ((k - 1) % 3 == 1) begin                       // 1.5 - (y y) h
            u.bs = B_K1; u.kc = F_1P5;
          end else begin                                             // y = t y (the last
            u.bs = B_K2; u.dst = (k == 9) ? D_V : D_K2;               // into v)
          end
        end
      CC_LOG:
        if (k == 0) begin                                            // t = m - 1; v = C9
          u.as = A_K1; u.kc = F_M1; u.dst = D_K1; u.kv = LOG2_C9;
        end else if (k <= 8) begin                                   // q*t + C8 .. C1
          u.bs = B_K1; u.kc = log2_c(9 - k);
        end else if (k == 9) begin                                   // q*t + i2f(e)
          u.bs = B_K1; u.cs = C_K2;
        end
      default: ;
    endcase
    return u;
  endfunction

  // What stage s's ops use, over every function and pass (elaboration time): the operand and
  // destination muxes keep only those inputs. At NS = 3, e.g., Q's c is always a constant and
  // Q never writes k1.
  function automatic logic [3:0] st_use(input int s, input int w);   // w: 0 a, 1 b, 2 c, 3 dst
    logic [3:0] m;
    uc_t u;
    m = '0;
    for (int c = 1; c <= 5; c++)
      for (int p = 0; p < int'(n_pass(3'(c))); p++) begin
        u = ucode(3'(c), p * NS + s);
        case (w)
          0:       m[u.as] = 1'b1;
          1:       m[u.bs] = 1'b1;
          2:       m[u.cs] = 1'b1;
          default: m[u.dst] = 1'b1;
        endcase
      end
    return m;
  endfunction
  // ... and whether one of them writes a constant into v (kv)
  function automatic logic kv_use(input int s);
    uc_t u;
    for (int c = 1; c <= 5; c++)
      for (int p = 0; p < int'(n_pass(3'(c))); p++) begin
        u = ucode(3'(c), p * NS + s);
        if (u.dst != D_V && u.kv != F_ZERO) return 1'b1;
      end
    return 1'b0;
  endfunction

  // stage s's control for a chunk of function c in pass p: a ROM over (c, p) built from ucode
  // with constant arguments (k's arithmetic, % 2 and % 3, never reaches the hardware)
  function automatic uc_t uc_rom(input logic [2:0] c, input logic [3:0] p, input int s);
    uc_t u;
    u = ucode(CC_NONE, 0);
    for (int ci = 1; ci <= 5; ci++)
      for (int pi = 0; pi < 16; pi++)
        if (c == 3'(ci) && p == 4'(pi)) u = ucode(3'(ci), pi * NS + s);
    return u;
  endfunction

  // ---------------------------------------------------------------- state and meta
  typedef struct packed {
    f32_t       v, k1, k2;
    logic [8:0] ii;
    logic [2:0] f;
  } cst_t;
  localparam int KW = 32 + 32 + 9 + 3;     // the fields carried along a stage (not v)
  typedef struct packed {
    logic [2:0]       cls;
    logic [3:0]       pass;
    logic [LANES-1:0] mask;
    logic [MW-1:0]    m;
  } cm_t;

  // pass 0's state from x (the chains' boundary 0). Fields a function does not read before
  // writing them are don't-cares: RECIP and RSQRT share one seed subtractor, and k1 and ii
  // are computed whatever the function.
  function automatic cst_t setup(input logic [2:0] c, input f32_t x);
    cst_t t;
    f32_t xz, ax, sd;
    logic ge, z, rcp;
    xz = ftz(x);
    ax = {1'b0, xz[30:0]};
    ge = (xz[22:0] >= LOG2_SQRT2);
    z  = (xz[30:0] == 0);
    rcp = (c == CC_RCP);
    sd = (rcp ? RECIP_MAGIC : RSQRT_MAGIC) - (rcp ? ax : (xz >> 1));
    t = '0;
    t.k2 = rcp ? ftz(sd) : sd;                                        // the seed
    t.k1 = (c == CC_LOG) ? {1'b0, ge ? 8'd126 : 8'd127, xz[22:0]}    // m in [sqrt(1/2), sqrt(2))
                         : {1'b1, ax[30:0]};                          // -|x|
    t.ii = 9'(xz[30:23]) - 9'd127 + 9'(ge);                           // LOG2's e
    case (c)
      CC_RCP: t.f = {xz[31], (ax >= 32'h7E80_0000), (ax == 0)};
      CC_RSQ: t.f = {2'b00, (xz[31] || z || xz == F_INF)};
      CC_LOG: t.f = {!z && (xz[31] || is_nan(xz)), xz == F_INF, z};  // NaN, +inf, -inf
      default: ;
    endcase
    return t;
  endfunction

  // RR's floor (ffloor on its domain): x flushed and in [-126, 128) or +-0, so |x| < 128 and
  // the integer part is the top 7 bits of the significand at most (outside it, EXP2's result
  // is a flag's +0 or +inf and the value here does not matter)
  function automatic logic [8:0] rr_floor(input f32_t x);
    logic [2:0] k;
    logic [6:0] ip;
    logic       fr;
    if (x[30:23] < 8'd127) return (x[31] && x[30:23] != 0) ? 9'h1FF : 9'd0;
    k  = 3'(x[30:23] - 8'd127);                   // 0..6
    ip = 7'({1'b1, x[22:17]} >> (3'd6 - k));
    fr = |(x[22:0] & (23'h7F_FFFF >> k));         // the fraction bits below the point
    return x[31] ? -(9'(ip) + 9'(fr)) : 9'(ip);
  endfunction

  // i2f of a 9-bit integer of magnitude <= 255: exact, so equal to i2f
  function automatic f32_t i2f9(input logic [8:0] n);
    logic [7:0]  m;
    logic [2:0]  p;
    logic [30:0] sh;
    m = n[8] ? 8'(-n) : n[7:0];
    if (m == 0) return F_ZERO;
    p = 3'd0;
    for (int j = 1; j < 8; j++) if (m[j]) p = 3'(j);
    sh = {m, 23'd0} >> p;                         // the leading one at bit 23
    return {n[8], 8'd127 + 8'(p), sh[22:0]};
  endfunction

  function automatic f32_t finish(input logic [2:0] c, input cst_t t);
    case (c)
      CC_EXP, CC_EXS: return t.f[0] ? F_ZERO : t.f[1] ? F_INF : (t.v + {t.ii, 23'd0});
      CC_RCP:         return t.f[0] ? F_ZERO : t.f[1] ? {t.f[2], 31'd0} :
                             (t.f[2] ? fneg(t.v) : t.v);
      CC_RSQ:         return t.f[0] ? F_ZERO : t.v;
      CC_LOG:         return t.f[0] ? F_NINF : t.f[2] ? F_NAN : t.f[1] ? F_INF : t.v;
      default:        return F_ZERO;
    endcase
  endfunction

  // ---------------------------------------------------------------- the loop's schedule
  // vpos[j] / cpos[j]: a chunk was presented to S0 j + 1 cycles ago / and it continues after
  // that pass. The loop register holds the chunk presented T cycles ago (vpos[T-1]).
  logic [T-1:0] vpos, cpos;
  cm_t          lp_m;                     // the loop register's chunk
  cst_t         lp_st [LANES];
  logic         lp_v, ret, fin;
  assign lp_v = vpos[T-1];
  assign fin  = (lp_m.pass + 4'd1 >= n_pass(lp_m.cls));
  // ret: it returns to S0 this cycle (= lp_v && !fin), decided a cycle early from the chunk
  // entering the loop register, with S0's control for it (uc_rt): S0's operand selects then
  // leave flip-flops
  logic         ret_r;
  assign ret  = ret_r;
  assign hold = cpos[T - HA - 1];         // it will return HA cycles from now
  assign out_v = lp_v && fin;
  assign out_m = lp_m.mask;
  assign out_meta = lp_m.m;
  for (genvar l = 0; l < LANES; l++) begin : g_out
    assign out_d[l] = finish(lp_m.cls, lp_st[l]);
  end

  cm_t  mi [NS];                           // meta at stage s's input (presented)
  cm_t  mo [NS];                           // ... at its output (SL later)
  cm_t  mp [NS];                           // ... after it (RR for s = 0)
  wire  p0 = in_v || ret;
  assign mi[0] = ret ? '{cls: lp_m.cls, pass: lp_m.pass + 4'd1, mask: lp_m.mask, m: lp_m.m} :
                       '{cls: f_cc(in_f), pass: 4'd0, mask: in_m, m: in_meta};
  always_ff @(posedge clk)
    if (rst) begin
      vpos <= '0; cpos <= '0;
    end else if (en) begin
      vpos <= {vpos[T-2:0], p0};
      cpos <= {cpos[T-2:0], p0 && (mi[0].pass + 4'd1 < n_pass(mi[0].cls))};
    end
`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && en && in_v && ret) $fatal(1, "otpu_se_comp: entry while a chunk returns");
`endif

  // stage presentation offsets from S0
  function automatic int s_off(input int s);
    return (s == 0) ? 0 : SL + RRL + (s - 1) * SL;
  endfunction
  for (genvar s = 0; s < NS; s++) begin : g_sel
    if (s == 0) begin : g_s0
      assign st_sel[s] = p0;
    end else begin : g_sn
      assign st_sel[s] = vpos[s_off(s) - 1];
    end
    otpu_delay #(.W($bits(cm_t)), .N(SL)) u_md (.clk, .en, .d(mi[s]), .q(mo[s]));
    if (s == 0) begin : g_rrm
      otpu_delay #(.W($bits(cm_t)), .N(RRL)) u_mr (.clk, .en, .d(mo[s]), .q(mp[s]));
    end else begin : g_nrm
      assign mp[s] = mo[s];
    end
    if (s + 1 < NS) begin : g_nx
      assign mi[s + 1] = mp[s];
    end
  end
  always_ff @(posedge clk) if (en) lp_m <= mp[NS - 1];
  uc_t uc_rt;
  always_ff @(posedge clk)
    if (rst) ret_r <= 1'b0;
    else if (en) ret_r <= vpos[T-2] && (mp[NS - 1].pass + 4'd1 < n_pass(mp[NS - 1].cls));
  always_ff @(posedge clk)
    if (en) uc_rt <= uc_rom(mp[NS - 1].cls, mp[NS - 1].pass + 4'd1, 0);
`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && ret != (lp_v && !fin)) $fatal(1, "otpu_se_comp: ret out of step");
`endif

  // per-stage control: operand selects and constants at the input, destination at the output
  uc_t uci [NS], uco [NS];
  for (genvar s = 0; s < NS; s++) begin : g_uc
    if (s == 0) begin : g_u0
      assign uci[s] = ret ? uc_rt : uc_rom(f_cc(in_f), 4'd0, 0);
    end else begin : g_un
      assign uci[s] = uc_rom(mi[s].cls, mi[s].pass, s);
    end
    assign uco[s] = uc_rom(mo[s].cls, mo[s].pass, s);
  end
  // RR's flags, from the meta after S0: EXP2's range reduction / LOG2's i2f(e), pass 0 only
  wire rr_e = (mo[0].pass == 0) && (mo[0].cls == CC_EXP || mo[0].cls == CC_EXS);
  wire rr_g = (mo[0].pass == 0) && (mo[0].cls == CC_LOG);
  logic e1, e2, g1, g2;
  always_ff @(posedge clk) if (en) begin
    e1 <= rr_e; g1 <= rr_g;
    e2 <= e1;   g2 <= g1;
  end

  // ---------------------------------------------------------------- lanes
  for (genvar l = 0; l < LANES; l++) begin : g_lane
    cst_t sti [NS], sto [NS], stp [NS];    // state at each stage's input, output, after it
    assign sti[0] = ret ? lp_st[l] : setup(f_cc(in_f), in_a[l]);

    for (genvar s = 0; s < NS; s++) begin : g_st
      localparam logic [3:0] AU = st_use(s, 0), BU = st_use(s, 1), CU = st_use(s, 2),
                             DU = st_use(s, 3);
      localparam logic       KVU = kv_use(s);
      f32_t a, b, c, y;
      always_comb begin
        cst_t t;
        t = sti[s];
        case (uci[s].as)
          A_K1:    a = AU[A_K1] ? t.k1 : t.v;
          A_K2:    a = AU[A_K2] ? t.k2 : t.v;
          A_X:     a = (s == 0 && AU[A_X]) ? in_a[l] : t.v;     // pass 0 is at S0
          default: a = t.v;
        endcase
        case (uci[s].bs)
          B_K1:    b = BU[B_K1] ? t.k1 : uci[s].kb;
          B_K2:    b = BU[B_K2] ? t.k2 : uci[s].kb;
          default: b = uci[s].kb;
        endcase
        case (uci[s].cs)
          C_K2:    c = CU[C_K2] ? t.k2 : uci[s].kc;
          C_NY:    c = (s == 0 && CU[C_NY]) ? fneg(in_b[l]) : uci[s].kc;
          default: c = uci[s].kc;
        endcase
      end
      assign st_a[s][l] = a;
      assign st_b[s][l] = b;
      assign st_c[s][l] = c;
      assign st_e[s][l] = F_ONE;
      if (EXT) begin : g_ext
        assign y = st_y[s][l];
      end else begin : g_int
        f32_t ra, rb, rc;
        always_ff @(posedge clk) if (en) begin
          ra <= a; rb <= b; rc <= c;
        end
        otpu_fmadd #(.LM(LM), .LA(LA)) u_ma (.clk, .en, .a(ra), .b(rb), .c(rc), .y(y));
      end
      // the carried fields, SL cycles (v is the stage's result or dead)
      logic [KW-1:0] kd;
      otpu_delay #(.W(KW), .N(SL)) u_k (.clk, .en, .d({sti[s].k1, sti[s].k2, sti[s].ii, sti[s].f}),
                                        .q(kd));
      always_comb begin
        f32_t r;
        r = uco[s].neg ? fneg(y) : y;
        sto[s] = '0;
        {sto[s].k1, sto[s].k2, sto[s].ii, sto[s].f} = kd;
        sto[s].v = KVU ? uco[s].kv : F_ZERO;
        case (uco[s].dst)
          D_V:  sto[s].v  = r;
          D_K1: if (DU[D_K1]) sto[s].k1 = r;
          D_K2: if (DU[D_K2]) sto[s].k2 = r;
          default: ;
        endcase
      end

      if (s == 0) begin : g_rr
        // RR: EXP2 f = (hi, lo) | i = floor(v) | k2 = -i2f(i); LOG2 k2 = i2f(e)
        cst_t q0, q1, q2;
        always_ff @(posedge clk) if (en) begin
          cst_t t;
          logic lo, hi;
          t = sto[s];
          if (rr_e) begin
            // x < -126 or x >= 128 (or NaN): the result is +0 or +inf whatever the steps
            // compute, so x is not clamped to 0 (the flags decide at the end)
            lo = fp_gt(F_M126, t.v);
            hi = !fp_gt(F_128, t.v);
            t.f = {1'b0, hi, lo};
          end
          q0 <= t;
          t = q0;
          if (e1) t.ii = rr_floor(q0.v);
          q1 <= t;
          t = q1;
          if (e2 || g2) begin
            t.k2 = i2f9(q1.ii);
            t.k2[31] = t.k2[31] ^ e2;
          end
          q2 <= t;
        end
        assign stp[s] = q2;
      end else begin : g_nrr
        assign stp[s] = sto[s];
      end
      if (s + 1 < NS) begin : g_nx
        assign sti[s + 1] = stp[s];
      end
    end
    always_ff @(posedge clk) if (en) lp_st[l] <= stp[NS - 1];
  end
endmodule
