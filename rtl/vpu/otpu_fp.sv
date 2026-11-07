// openTPU fp32 arithmetic (docs/isa.md "Arithmetic").
// IEEE-754 binary32, round-to-nearest-even, flush-to-zero on inputs and outputs.
// Bit-exact with opentpu/fp32.py. Functions are combinational; composite functions (exp2,
// recip, rsqrt, log2) are fixed sequences of fp_add/fp_mul.
package otpu_fp;

  typedef logic [31:0] f32_t;

  localparam f32_t F_ZERO  = 32'h0000_0000;
  localparam f32_t F_ONE   = 32'h3F80_0000;
  localparam f32_t F_TWO   = 32'h4000_0000;
  localparam f32_t F_HALF  = 32'h3F00_0000;
  localparam f32_t F_1P5   = 32'h3FC0_0000;
  localparam f32_t F_127   = 32'h42FE_0000;
  localparam f32_t F_M126  = 32'hC2FC_0000;   // -126.0
  localparam f32_t F_128   = 32'h4300_0000;   //  128.0
  localparam f32_t F_INF   = 32'h7F80_0000;
  localparam f32_t F_NINF  = 32'hFF80_0000;
  localparam f32_t F_M1    = 32'hBF80_0000;   // -1.0
  localparam f32_t F_NAN   = 32'h7FC0_0000;
  localparam f32_t F_INV127 = 32'h3C01_0204;  // f32(1/127)

  // 2^f ~ C0 + f*(C1 + ... + f*C7) on [0, 1]: minimax (relative error, degree 7, C0 = 1), fp32
  // (same as opentpu/fp32.py).
  localparam f32_t EXP2_C0 = 32'h3F80_0000;
  localparam f32_t EXP2_C1 = 32'h3F31_7218;
  localparam f32_t EXP2_C2 = 32'h3E75_FDE9;
  localparam f32_t EXP2_C3 = 32'h3D63_593E;
  localparam f32_t EXP2_C4 = 32'h3C1D_8534;
  localparam f32_t EXP2_C5 = 32'h3AAF_E2F6;
  localparam f32_t EXP2_C6 = 32'h3916_C07C;
  localparam f32_t EXP2_C7 = 32'h37B3_C7D6;

  // log2(1+t) ~ t*(C1 + t*(C2 + ... + t*C9)), minimax, fp32 (same as opentpu/fp32.py).
  localparam f32_t LOG2_C1 = 32'h3FB8_AA3B;
  localparam f32_t LOG2_C2 = 32'hBF38_AA38;
  localparam f32_t LOG2_C3 = 32'h3EF6_39EB;
  localparam f32_t LOG2_C4 = 32'hBEB8_AE27;
  localparam f32_t LOG2_C5 = 32'h3E93_69C2;
  localparam f32_t LOG2_C6 = 32'hBE74_ADF2;
  localparam f32_t LOG2_C7 = 32'h3E5C_E48E;
  localparam f32_t LOG2_C8 = 32'hBE54_3E8E;
  localparam f32_t LOG2_C9 = 32'h3E00_DB73;
  localparam logic [22:0] LOG2_SQRT2 = 23'h35_04F3;   // mantissa bits of sqrt(2)

  localparam logic [31:0] RECIP_MAGIC = 32'h7EF3_11C3;
  localparam logic [31:0] RSQRT_MAGIC = 32'h5F37_59DF;

  function automatic f32_t ftz(input f32_t a);
    return (a[30:23] == 8'd0) ? {a[31], 31'd0} : a;
  endfunction

  function automatic f32_t fneg(input f32_t a);
    return {~a[31], a[30:0]};
  endfunction

  function automatic f32_t fabs(input f32_t a);
    f32_t t;
    t = ftz(a);
    return {1'b0, t[30:0]};
  endfunction

  function automatic logic is_nan(input f32_t a);
    return (a[30:23] == 8'hFF) && (a[22:0] != 0);
  endfunction

  // Leading-zero counts as balanced trees (a priority chain is too slow for the FPGA clock).
  function automatic logic [5:0] lzc32(input logic [31:0] m);
    logic [4:0] n;
    logic [31:0] x;
    if (m == 0) return 6'd32;
    x = m; n = '0;
    if (x[31:16] == 0) begin n[4] = 1'b1; x = x << 16; end
    if (x[31:24] == 0) begin n[3] = 1'b1; x = x << 8; end
    if (x[31:28] == 0) begin n[2] = 1'b1; x = x << 4; end
    if (x[31:30] == 0) begin n[1] = 1'b1; x = x << 2; end
    if (x[31] == 0)    begin n[0] = 1'b1; end
    return {1'b0, n};
  endfunction

  // Normalize while counting: the shift stages of the leading-zero search are the normalize
  // shifter (one mux chain, not an LZC plus a separate barrel shifter). m == 0 -> n = 31, y = 0.
  typedef struct packed {
    logic [4:0]  n;
    logic [27:0] y;       // m << n
  } norm28_t;

  function automatic norm28_t norm28(input logic [27:0] m);
    norm28_t r;
    r.y = m; r.n = '0;
    if (r.y[27:12] == 0) begin r.n[4] = 1'b1; r.y = r.y << 16; end
    if (r.y[27:20] == 0) begin r.n[3] = 1'b1; r.y = r.y << 8; end
    if (r.y[27:24] == 0) begin r.n[2] = 1'b1; r.y = r.y << 4; end
    if (r.y[27:26] == 0) begin r.n[1] = 1'b1; r.y = r.y << 2; end
    if (r.y[27] == 0)    begin r.n[0] = 1'b1; r.y = r.y << 1; end
    return r;
  endfunction

  // Align while collecting the sticky: each right-shift stage ORs the bits it drops.
  // Equals {x >> k, |x[k-1:0]} for k = 0..31 (x is 27 bits, so k >= 27 gives y = 0).
  typedef struct packed {
    logic        st;
    logic [26:0] y;
  } align27_t;

  function automatic align27_t align27(input logic [26:0] x, input logic [4:0] k);
    align27_t r;
    r.y = x; r.st = 1'b0;
    if (k[4]) begin r.st = r.st | (|r.y[15:0]); r.y = r.y >> 16; end
    if (k[3]) begin r.st = r.st | (|r.y[7:0]);  r.y = r.y >> 8;  end
    if (k[2]) begin r.st = r.st | (|r.y[3:0]);  r.y = r.y >> 4;  end
    if (k[1]) begin r.st = r.st | (|r.y[1:0]);  r.y = r.y >> 2;  end
    if (k[0]) begin r.st = r.st | r.y[0];       r.y = r.y >> 1;  end
    return r;
  endfunction

  // |x[k-1:0]| for a variable k (0..32)
  function automatic logic sticky_below(input logic [31:0] x, input logic [5:0] k);
    logic [31:0] mask;
    mask = ~(32'hFFFF_FFFF << k);            // a shift by >= 32 gives 0: all ones
    return |(x & mask);
  endfunction

  // ---- fp_mul in two stages (the product is the register boundary: on the FPGA it sits in
  // the DSP48 M/P registers).
  typedef struct packed {
    logic               sp;      // special result (zero, inf, NaN) already known
    logic [31:0]        sv;
    logic               s;
    logic signed [10:0] e;
    logic [47:0]        p;
  } fmul_mid_t;

  function automatic fmul_mid_t fp_mul_s1(input f32_t a_in, input f32_t b_in);
    fmul_mid_t m;
    f32_t a, b;
    a = ftz(a_in);
    b = ftz(b_in);
    m = '0;
    m.s = a[31] ^ b[31];
    if (a[30:23] == 8'hFF || b[30:23] == 8'hFF) begin
      m.sp = 1'b1;
      m.sv = (is_nan(a) || is_nan(b) || a[30:0] == 0 || b[30:0] == 0) ? F_NAN
             : {m.s, 8'hFF, 23'd0};
    end else if (a[30:23] == 0 || b[30:23] == 0) begin
      m.sp = 1'b1;
      m.sv = {m.s, 31'd0};
    end
    // The DSP operands skip the flush: ftz only changes a zero-exponent input, and then sp is
    // set and m.p is never read.
    m.p = {1'b1, a_in[22:0]} * {1'b1, b_in[22:0]};
    m.e = 11'(a[30:23]) + 11'(b[30:23]) - 11'sd127;
    return m;
  endfunction

  // Rounding carries out (mr[24]) only when mm was all ones: mr == 2^24, so mr[22:0] == 0 with
  // or without a >> 1. The exponent bump c is checked against e0 (the exponent before rounding),
  // so the overflow/underflow flags don't wait on the carry chain.
  // e0 == 0 (a product in [2^-127, 2^-126)): IEEE rounds it on the subnormal grid (2^-149, one
  // bit coarser than mm's 2^-150), then the flush keeps only a result of 2^-126. That is every
  // product >= 2^-126 - 2^-150 (the tie goes to 2^-126, the even one), i.e. mm all ones whatever
  // g and st say; rounding at 24 bits would need g too (0x3F000000 * 0x00FFFFFF is 2^-126).
  // Written so that nothing waits on another carry chain (250 MHz, FK35: the product's top bit
  // h (fo ~60) -> the mm mux -> the rounding adder with rnd as its carry-in -> c -> the e0 tests,
  // 10 levels, -0.77 ns in the MXU's g_rmw): the increment mm + 1 runs beside the rounding
  // decision, which then picks (mr) and gives the carry-out as rnd && all-ones (c); the exponent
  // tests are on m.e with h choosing, and the field is e + h, or e + h + 1 picked by c. The same
  // function as before, bit for bit.
  // It is two steps, fp_mul_s2a (everything but the picks) and fp_mul_s2b (the picks), which
  // otpu_fmul SPLIT registers apart (a 3-stage multiplier); fp_mul_s2 is their composition.
  typedef struct packed {
    logic        sp;
    logic [31:0] sv;
    logic        s;
    logic        rnd;      // round up
    logic        a1;       // mm all ones (rnd && a1: the rounding carries out)
    logic        ge255, eq254, lt0, eq0;   // e0 = e + h >= 255, == 254, < 0, == 0
    logic [22:0] mm, mi;   // the significand's fraction, and it + 1
    logic [7:0]  eh, eh1;  // 8'(e0), and it + 1
  } fmul_rnd_t;

  function automatic fmul_rnd_t fp_mul_s2a(input fmul_mid_t m);
    fmul_rnd_t r;
    logic h, g, st;
    logic signed [10:0] e;
    logic [23:0] mm;
    h = m.p[47];
    e = m.e;
    mm = h ? m.p[47:24] : m.p[46:23];
    g = h ? m.p[23] : m.p[22];
    st = (|m.p[21:0]) || (h && m.p[22]);
    r.sp = m.sp;
    r.sv = m.sv;
    r.s = m.s;
    r.rnd = g && (st || mm[0]);
    r.a1 = (&m.p[46:24]) && (h || m.p[23]);      // &mm
    r.mm = mm[22:0];
    r.mi = mm[22:0] + 23'd1;
    r.ge255 = h ? (e >= 11'sd254) : (e >= 11'sd255);
    r.eq254 = h ? (e == 11'sd253) : (e == 11'sd254);
    r.lt0   = h ? (e < -11'sd1) : (e < 11'sd0);
    r.eq0   = h ? (e == -11'sd1) : (e == 11'sd0);
    r.eh  = 8'(e) + 8'(h);
    r.eh1 = 8'(e) + 8'(h) + 8'd1;
    return r;
  endfunction

  function automatic f32_t fp_mul_s2b(input fmul_rnd_t r);
    logic c;
    c = r.rnd && r.a1;                           // mm + rnd carries out
    if (r.sp) return r.sv;
    if (r.ge255 || (r.eq254 && c)) return {r.s, 8'hFF, 23'd0};
    if (r.lt0) return {r.s, 31'd0};
    if (r.eq0) return r.a1 ? {r.s, 8'd1, 23'd0} : {r.s, 31'd0};
    return {r.s, c ? r.eh1 : r.eh, r.rnd ? r.mi : r.mm};   // 8'(e0) + 8'(c), mm + rnd
  endfunction

  function automatic f32_t fp_mul_s2(input fmul_mid_t m);
    return fp_mul_s2b(fp_mul_s2a(m));
  endfunction

  function automatic f32_t fp_mul(input f32_t a_in, input f32_t b_in);
    return fp_mul_s2(fp_mul_s1(a_in, b_in));
  endfunction

  // ---- fp_add in four stages: unpack + compare | align + add | normalize | round.
  // Only inf/NaN operands and exact cancellation are special; zero operands ride the datapath
  // (a zero significand). The special value is {sign, exp_all_ones, quiet}, see fadd_sv.
  typedef struct packed {
    logic        sp;
    logic [2:0]  sv;
    logic        sa;       // sign of the larger operand (the result sign unless it cancels)
    logic        sub;      // effective subtraction
    logic [7:0]  e;
    logic [7:0]  d;        // exponent difference (>= 0)
    logic [26:0] ma, mb;   // significands with 3 guard bits, mb not yet aligned
  } fadd_p1_t;

  typedef struct packed {
    logic        sp;
    logic [2:0]  sv;
    logic        sa;
    logic        sub;
    logic [7:0]  e;        // exponent of the larger operand (e <= 254 unless sp)
    logic [27:0] sum;
  } fadd_p2_t;

  typedef struct packed {
    logic               sp;
    logic [2:0]         sv;
    logic               s;
    logic signed [8:0]  e;    // e0 - 1, in [-31, 254] unless sp (s4 adds the 1 back)
    logic [26:0]        mn;   // normalized: mn[26] set, guard/round/sticky in [2:0]
  } fadd_nm_t;

  localparam logic [2:0] FADD_SV_NAN  = 3'b011;
  localparam logic [2:0] FADD_SV_ZERO = 3'b000;

  // Expand the 3-bit special value: NaN -> F_NAN, {s,2'b10} -> +-inf, 3'b000 -> F_ZERO.
  function automatic f32_t fadd_sv(input logic [2:0] sv);
    return {sv[2], {8{sv[1]}}, sv[0], 22'd0};
  endfunction

  // The swap (the larger magnitude first) is the compare of the flushed magnitudes, made from the
  // raw ones: the flush only zeroes a zero-exponent operand, whose raw magnitude is already below
  // every normal one, so the raw compare orders every pair the same except two flushed zeros
  // (no swap). Both exponent differences are formed beside the compare, which then picks one:
  // compare -> swap -> subtract was the serial path (250 MHz, FK35: u_vt q_cap -> u_tree s1.d,
  // the flush's zero tests, the compare, the swap (fo 64), the subtract, 9 levels, -0.87 ns).
  // The same fields as before, bit for bit.
  function automatic fadd_p1_t fp_add_s1(input f32_t a_in, input f32_t b_in);
    fadd_p1_t r;
    logic za, zb, sw;
    logic [7:0] ea, eb, dab, dba;
    logic [26:0] xa, xb;
    ea = a_in[30:23];
    eb = b_in[30:23];
    za = (ea == 8'd0);
    zb = (eb == 8'd0);
    // inf/NaN; the datapath fields below are don't-care when sp. A zero operand is not special:
    // its significand is 0 (hidden bit = exponent nonzero), and align/add/round return the other.
    // (The flush leaves inf and NaN alone, so these read the raw operands.)
    r.sp = (ea == 8'hFF || eb == 8'hFF);
    if (is_nan(a_in) || is_nan(b_in)) r.sv = FADD_SV_NAN;
    else if (ea == 8'hFF && eb == 8'hFF && a_in[31] != b_in[31]) r.sv = FADD_SV_NAN;
    else r.sv = {(ea == 8'hFF) ? a_in[31] : b_in[31], 2'b10};
    sw = (b_in[30:0] > a_in[30:0]) && !(za && zb);
    dab = ea - eb;
    dba = eb - ea;
    xa = {!za, za ? 23'd0 : a_in[22:0], 3'b000};
    xb = {!zb, zb ? 23'd0 : b_in[22:0], 3'b000};
    r.sa = sw ? b_in[31] : a_in[31];
    r.sub = a_in[31] ^ b_in[31];
    r.e = sw ? eb : ea;
    r.d = sw ? dba : dab;
    r.ma = sw ? xb : xa;
    r.mb = sw ? xa : xb;
    return r;
  endfunction

  function automatic fadd_p2_t fp_add_s2(input fadd_p1_t r);
    fadd_p2_t q;
    logic [26:0] mb;
    align27_t al;
    q.sp = r.sp; q.sv = r.sv; q.sa = r.sa; q.sub = r.sub;
    q.e = r.e;                               // the normalize's +1 is folded into s4
    // d = 0 shifts nothing (st = 0); d > 26 overrides, so the shifter only sees d[4:0].
    al = align27(r.mb, r.d[4:0]);
    mb = (r.d > 8'd26) ? 27'd1 : (al.y | {26'd0, al.st});
    q.sum = r.sub ? ({1'b0, r.ma} - {1'b0, mb}) : ({1'b0, r.ma} + {1'b0, mb});
    return q;
  endfunction

  // norm28's count and shift with the count as a tree (the zero flags and counts of 4-bit groups,
  // combined pairwise), so all five bits come in a few levels instead of each after the shifts
  // before it (norm28: n0 after four shift stages); the shift takes n4 and n3 as they come (the
  // 16- and 8-bit zero tests) and n1, n0 together last. The same {n, y} as norm28 (m == 0: 31, 0).
  function automatic norm28_t norm28t(input logic [27:0] m);
    norm28_t r;
    logic [31:0] x;
    logic [7:0]  z4;
    logic [1:0]  c4 [8];
    logic [3:0]  z8;
    logic [2:0]  c8 [4];
    logic [1:0]  z16;
    logic [3:0]  c16 [2];
    logic [27:0] y1, y2, y3;
    x = {m, 4'b0000};
    for (int g = 0; g < 8; g++) begin
      z4[g] = (x[4*g +: 4] == 4'd0);
      c4[g] = x[4*g + 3] ? 2'd0 : x[4*g + 2] ? 2'd1 : x[4*g + 1] ? 2'd2 : 2'd3;
    end
    for (int h = 0; h < 4; h++) begin
      z8[h] = z4[2*h + 1] && z4[2*h];
      c8[h] = z4[2*h + 1] ? {1'b1, c4[2*h]} : {1'b0, c4[2*h + 1]};
    end
    for (int k = 0; k < 2; k++) begin
      z16[k] = z8[2*k + 1] && z8[2*k];
      c16[k] = z8[2*k + 1] ? {1'b1, c8[2*k]} : {1'b0, c8[2*k + 1]};
    end
    r.n = z16[1] ? {1'b1, c16[0]} : {1'b0, c16[1]};
    y1 = r.n[4] ? (m << 16) : m;
    y2 = r.n[3] ? (y1 << 8) : y1;
    y3 = r.n[2] ? (y2 << 4) : y2;
    r.y = y3 << r.n[1:0];
    return r;
  endfunction

  // One normalize for add and sub over the 28-bit sum, counted and shifted together by norm28.
  // An add carry-out gives n = 0 (the sticky OR folds sum[1:0]); otherwise sum[26] is set, n = 1,
  // and y[0] is 0. A sub never carries out, so n = lzc27(sum[26:0]) + 1, which the
  // +1 folded into s4's constants absorbs. A zero add (0+0) leaves mn = 0 and e0 = 1 - 31 < 0:
  // s4 returns {sa, 0}, keeping the -0 of (-0)+(-0). A zero sub is the cancellation special;
  // norm28 moves any nonzero sum up to y[27] (16+8+4+2+1 >= 27), so !y[27] is sum == 0, tested
  // on the sum directly. The count is norm28t's tree (250 MHz, FK35: the tail's Q adders s2.sum
  // -> norm28's serial count -> e = e - n, 9 levels, -0.67 ns; -> mn 7 levels, -0.57 ns).
  function automatic fadd_nm_t fp_add_s3(input fadd_p2_t q);
    fadd_nm_t n;
    norm28_t nz;
    logic zs;
    nz = norm28t(q.sum);
    zs = (q.sum == 28'd0);
    n.sp = q.sp;
    n.sv = q.sv;
    n.s = q.sa;
    if (q.sub && zs && !q.sp) begin
      n.sp = 1'b1;
      n.sv = FADD_SV_ZERO;
    end
    n.mn = {nz.y[27:2], nz.y[1] | nz.y[0]};
    n.e = 9'(q.e) - 9'(nz.n);                // e0 - 1
    return n;
  endfunction

  // x = n.e = e0 - 1: each test is the one on e0 (the exponent before rounding) with e0 = x + 1.
  function automatic f32_t fp_add_s4(input fadd_nm_t n);
    logic g, rs, c;
    logic signed [8:0] x;
    logic [23:0] mm;
    logic [24:0] mr;
    if (n.sp) return fadd_sv(n.sv);
    x = n.e;
    g  = n.mn[2];
    rs = n.mn[1] | n.mn[0];
    mm = n.mn[26:3];
    mr = {1'b0, mm} + ((g && (rs || mm[0])) ? 25'd1 : 25'd0);
    c = mr[24];   // as in fp_mul_s2: no post-round shift, flags off e0
    if (x >= 9'sd254 || (x == 9'sd253 && c)) return {n.s, 8'hFF, 23'd0};
    if (x < -9'sd1 || (x == -9'sd1 && !c)) return {n.s, 31'd0};
    return {n.s, 8'(x) + {6'd0, c, ~c}, mr[22:0]};   // 8'(e0) + 8'(c)
  endfunction

  function automatic f32_t fp_add(input f32_t a_in, input f32_t b_in);
    return fp_add_s4(fp_add_s3(fp_add_s2(fp_add_s1(a_in, b_in))));
  endfunction

  function automatic f32_t fp_sub(input f32_t a, input f32_t b);
    return fp_add(a, fneg(b));
  endfunction

  // a > b in the total order of flushed values: -inf < ... < -0 < +0 < ... < +inf < NaN.
  function automatic logic [31:0] fkey(input f32_t a_in);
    f32_t a;
    a = ftz(a_in);
    return a[31] ? ~a : {1'b1, a[30:0]};
  endfunction

  // fkey(a) > fkey(b), with the flush beside the comparator instead of in front of it: the
  // unflushed keys order every pair with nonzero exponents the same, and a flushed operand is
  // a signed zero, which the other's sign alone orders (above every negative, below every
  // positive; +0 > -0)
  function automatic logic fp_gt(input f32_t a, input f32_t b);
    logic za, zb, gr;
    za = (a[30:23] == 8'd0);
    zb = (b[30:23] == 8'd0);
    gr = (a[31] ? ~a : {1'b1, a[30:0]}) > (b[31] ? ~b : {1'b1, b[30:0]});
    return za ? (zb ? (!a[31] && b[31]) : b[31]) : (zb ? !a[31] : gr);
  endfunction

  function automatic f32_t fp_max(input f32_t a, input f32_t b);
    return fp_gt(a, b) ? ftz(a) : ftz(b);
  endfunction

  function automatic f32_t fp_min(input f32_t a, input f32_t b);
    return fp_gt(b, a) ? ftz(a) : ftz(b);
  endfunction

  // fp_max (mn = 0) or fp_min (mn = 1) on one comparator: when neither operand is above the
  // other their keys are equal, and so are their flushed values (fkey is one-to-one)
  function automatic f32_t fp_mm(input f32_t a, input f32_t b, input logic mn);
    return (fp_gt(a, b) ^ mn) ? ftz(a) : ftz(b);
  endfunction

  // i2f in two stages: magnitude + leading zeros | normalize + round.
  typedef struct packed {
    logic        z, s;
    logic [5:0]  lz;
    logic [31:0] mag;
  } i2f_mid_t;

  function automatic i2f_mid_t i2f_s1(input logic signed [31:0] x);
    i2f_mid_t m;
    m.z = (x == 0);
    m.s = x[31];
    m.mag = x[31] ? (~x + 32'd1) : x;
    m.lz = lzc32(m.mag);
    return m;
  endfunction

  function automatic f32_t i2f_s2(input i2f_mid_t m);
    logic g, st;
    logic [31:0] nrm;
    logic [23:0] mm;
    logic [24:0] mr;
    logic [8:0] e;
    if (m.z) return F_ZERO;
    nrm = m.mag << m.lz;
    e = 9'd158 - 9'(m.lz);
    mm = nrm[31:8];
    g = nrm[7];
    st = |nrm[6:0];
    mr = {1'b0, mm} + ((g && (st || mm[0])) ? 25'd1 : 25'd0);
    if (mr[24]) begin
      mr = mr >> 1;
      e = e + 1;
    end
    return {m.s, e[7:0], mr[22:0]};
  endfunction

  function automatic f32_t i2f(input logic signed [31:0] x);
    return i2f_s2(i2f_s1(x));
  endfunction

  // Round half to even, saturate to [-127, 127], NaN -> 0; in two stages: shift | round + saturate.
  typedef struct packed {
    logic        zero, sat, s;
    logic [31:0] ip;
    logic        g, st;
  } q8_mid_t;

  function automatic q8_mid_t q8_s1(input f32_t x_in);
    q8_mid_t q;
    f32_t x;
    int e, sh;
    logic [23:0] m;
    x = ftz(x_in);
    e = int'(x[30:23]);
    q = '0;
    q.s = x[31];
    q.zero = (e < 126) || is_nan(x);
    q.sat = (e >= 134);
    m = {1'b1, x[22:0]};
    sh = (q.zero || q.sat) ? 17 : 150 - e;       // 17..24
    q.ip = 32'(m) >> sh;
    q.g = m[sh - 1];
    q.st = sticky_below({8'd0, m}, 6'(sh - 1));
    return q;
  endfunction

  function automatic logic [7:0] q8_s2(input q8_mid_t q);
    logic [31:0] ip;
    if (q.zero) return 8'd0;
    if (q.sat) return q.s ? 8'h81 : 8'h7F;
    ip = q.ip;
    if (q.g && (q.st || ip[0])) ip = ip + 1;
    if (ip > 127) ip = 127;
    return q.s ? 8'(-ip) : 8'(ip);
  endfunction

  function automatic logic [7:0] q8(input f32_t x_in);
    return q8_s2(q8_s1(x_in));
  endfunction

  // floor() of x in [-126, 128) as an integer.
  function automatic int ffloor(input f32_t x_in);
    f32_t x;
    int e, sh, ip;
    logic [23:0] m;
    logic frac;
    x = ftz(x_in);
    if (x[30:0] == 0) return 0;
    e = int'(x[30:23]);
    if (e < 127) return x[31] ? -1 : 0;
    m = {1'b1, x[22:0]};
    sh = 150 - e;                       // 16..23 for |x| < 256
    ip = int'(32'(m) >> sh);
    frac = sticky_below({8'd0, m}, 6'(sh));
    if (x[31]) return -(ip + (frac ? 1 : 0));
    return ip;
  endfunction

  function automatic f32_t fp_exp2(input f32_t x_in);
    f32_t x, f, p;
    int i;
    logic [31:0] r;
    x = ftz(x_in);
    if (is_nan(x)) return F_INF;          // as the chains: their op 0 (x*1 + -0) makes it +NaN
    if (fp_gt(F_M126, x)) return F_ZERO;
    if (!fp_gt(F_128, x)) return F_INF;
    i = ffloor(x);
    f = fp_sub(x, i2f(i));
    p = EXP2_C7;
    p = fp_add(fp_mul(p, f), EXP2_C6);
    p = fp_add(fp_mul(p, f), EXP2_C5);
    p = fp_add(fp_mul(p, f), EXP2_C4);
    p = fp_add(fp_mul(p, f), EXP2_C3);
    p = fp_add(fp_mul(p, f), EXP2_C2);
    p = fp_add(fp_mul(p, f), EXP2_C1);
    p = fp_add(fp_mul(p, f), EXP2_C0);
    r = p + (i << 23);
    return r;
  endfunction

  // recip runs on the significand a (x's fraction with exponent field 127, in [1, 2)) and its
  // result y (in [0.5, 1]) gets 127 - (x's field) added to its exponent field: rcp_ek, a 9-bit
  // two's complement (-125..126 below 2^126; it stays normal).
  function automatic logic [8:0] rcp_ek(input f32_t x);
    return 9'd127 - 9'(x[30:23]);
  endfunction

  // rsqrt's range scaling (docs/isa.md): the Newton steps are exact under x -> x * 2^2k,
  // y -> y * 2^-k while nothing flushes, so out-of-range inputs run scaled and the result's
  // exponent field is adjusted back. RSQ_LO / RSQ_HI: exponent field <= 2 runs on 16x (h = x/2
  // flushes at field 1), field 250..254 on x/16 (y*y flushes at >= 252).
  function automatic logic rsq_lo(input f32_t x);
    return x[30:23] <= 8'd2;
  endfunction
  function automatic logic rsq_hi(input f32_t x);
    return x[30:23] >= 8'd250 && x[30:23] != 8'hFF;
  endfunction

  function automatic f32_t fp_recip(input f32_t x_in);
    f32_t x, a, y;
    x = ftz(x_in);
    if (x[30:0] == 0) return F_ZERO;
    if (x[30:0] >= 31'h7E80_0000) return {x[31], 31'b0};   // |x| >= 2^126 (incl. inf): flushes
    a = {9'h07F, x[22:0]};                                  // the significand, in [1, 2)
    y = RECIP_MAGIC - a;
    for (int k = 0; k < 2; k++) y = fp_mul(y, fp_sub(F_TWO, fp_mul(a, y)));
    y = fp_add(fp_mul(y, fp_sub(F_ONE, fp_mul(a, y))), y);     // y*e + y, e = 1 - a*y
    y = y + {rcp_ek(x), 23'd0};
    return x[31] ? fneg(y) : y;
  endfunction

  function automatic f32_t fp_rsqrt(input f32_t x_in);
    f32_t x, y, h;
    logic lo, hi;
    x = ftz(x_in);
    if (x[31] || x[30:0] == 0 || x == 32'h7F80_0000) return F_ZERO;
    lo = rsq_lo(x);
    hi = rsq_hi(x);
    if (lo) x = x + 32'h0200_0000;                        // 16x
    if (hi) x = x - 32'h0200_0000;                        // x / 16
    y = RSQRT_MAGIC - (x >> 1);
    h = fp_mul(F_HALF, x);
    for (int k = 0; k < 2; k++) y = fp_mul(y, fp_sub(F_1P5, fp_mul(h, fp_mul(y, y))));
    y = fp_add(fp_mul(y, fp_sub(F_HALF, fp_mul(h, fp_mul(y, y)))), y);   // e = 0.5 - h*(y*y)
    if (lo) y = y + 32'h0100_0000;                        // * 4
    if (hi) y = y - 32'h0100_0000;                        // / 4
    return y;
  endfunction

  // log2: x = 2^e * m, m in [sqrt(1/2), sqrt(2)); t = m - 1 (exact); Horner from C9; + i2f(e)
  function automatic f32_t fp_log2(input f32_t x_in);
    f32_t x, m, t, q;
    logic ge;
    logic signed [31:0] e;
    x = ftz(x_in);
    if (x[30:0] == 0) return F_NINF;
    if (x[31] || is_nan(x)) return F_NAN;
    if (x == F_INF) return F_INF;
    ge = (x[22:0] >= LOG2_SQRT2);
    e = 32'(x[30:23]) - 32'sd127 + 32'(ge);
    m = {1'b0, ge ? 8'd126 : 8'd127, x[22:0]};
    t = fp_add(m, F_M1);
    q = LOG2_C9;
    q = fp_add(fp_mul(q, t), LOG2_C8);
    q = fp_add(fp_mul(q, t), LOG2_C7);
    q = fp_add(fp_mul(q, t), LOG2_C6);
    q = fp_add(fp_mul(q, t), LOG2_C5);
    q = fp_add(fp_mul(q, t), LOG2_C4);
    q = fp_add(fp_mul(q, t), LOG2_C3);
    q = fp_add(fp_mul(q, t), LOG2_C2);
    q = fp_add(fp_mul(q, t), LOG2_C1);
    return fp_add(fp_mul(q, t), i2f(e));
  endfunction

endpackage
