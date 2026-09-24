// openTPU fp32 arithmetic (docs/isa.md "Arithmetic").
// IEEE-754 binary32, round-to-nearest-even, flush-to-zero on inputs and outputs.
// Bit-exact with opentpu/fp32.py. Functions are combinational; composite functions (exp2,
// recip, rsqrt) are fixed sequences of fp_add/fp_mul.
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
  localparam f32_t F_NAN   = 32'h7FC0_0000;
  localparam f32_t F_INV127 = 32'h3C01_0204;  // f32(1/127)

  // Taylor coefficients ln2^k / k!, k = 0..7, rounded to fp32 (same as opentpu/fp32.py).
  localparam f32_t EXP2_C0 = 32'h3F80_0000;
  localparam f32_t EXP2_C1 = 32'h3F31_7218;
  localparam f32_t EXP2_C2 = 32'h3E75_FDF0;
  localparam f32_t EXP2_C3 = 32'h3D63_5847;
  localparam f32_t EXP2_C4 = 32'h3C1D_955B;
  localparam f32_t EXP2_C5 = 32'h3AAE_C3FF;
  localparam f32_t EXP2_C6 = 32'h3921_8489;
  localparam f32_t EXP2_C7 = 32'h377F_E5FE;

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
  function automatic f32_t fp_mul_s2(input fmul_mid_t m);
    logic g, st, c;
    logic signed [10:0] e0;
    logic [23:0] mm;
    logic [24:0] mr;
    if (m.sp) return m.sv;
    if (m.p[47]) begin
      mm = m.p[47:24]; g = m.p[23]; st = |m.p[22:0];
    end else begin
      mm = m.p[46:23]; g = m.p[22]; st = |m.p[21:0];
    end
    e0 = m.e + 11'(m.p[47]);
    mr = {1'b0, mm} + ((g && (st || mm[0])) ? 25'd1 : 25'd0);
    c = mr[24];
    if (e0 >= 11'sd255 || (e0 == 11'sd254 && c)) return {m.s, 8'hFF, 23'd0};
    if (e0 < 11'sd0 || (e0 == 11'sd0 && !c)) return {m.s, 31'd0};
    return {m.s, 8'(e0) + 8'(c), mr[22:0]};
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
    logic [7:0]  e;        // exponent + 1 (fits: e <= 254 unless sp)
    logic [27:0] sum;
  } fadd_p2_t;

  typedef struct packed {
    logic               sp;
    logic [2:0]         sv;
    logic               s;
    logic signed [9:0]  e;
    logic [26:0]        mn;   // normalized: mn[26] set, guard/round/sticky in [2:0]
  } fadd_nm_t;

  localparam logic [2:0] FADD_SV_NAN  = 3'b011;
  localparam logic [2:0] FADD_SV_ZERO = 3'b000;

  // Expand the 3-bit special value: NaN -> F_NAN, {s,2'b10} -> +-inf, 3'b000 -> F_ZERO.
  function automatic f32_t fadd_sv(input logic [2:0] sv);
    return {sv[2], {8{sv[1]}}, sv[0], 22'd0};
  endfunction

  function automatic fadd_p1_t fp_add_s1(input f32_t a_in, input f32_t b_in);
    fadd_p1_t r;
    f32_t a, b, t;
    a = ftz(a_in);
    b = ftz(b_in);
    // inf/NaN; the datapath fields below are don't-care when sp. A zero operand is not special:
    // its significand is 0 (hidden bit = exponent nonzero), and align/add/round return the other.
    r.sp = (a[30:23] == 8'hFF || b[30:23] == 8'hFF);
    if (is_nan(a) || is_nan(b)) r.sv = FADD_SV_NAN;
    else if (a[30:23] == 8'hFF && b[30:23] == 8'hFF && a[31] != b[31]) r.sv = FADD_SV_NAN;
    else r.sv = {(a[30:23] == 8'hFF) ? a[31] : b[31], 2'b10};
    if (b[30:0] > a[30:0]) begin
      t = a; a = b; b = t;
    end
    r.sa = a[31];
    r.sub = a[31] ^ b[31];
    r.e = a[30:23];
    r.d = a[30:23] - b[30:23];
    r.ma = {a[30:23] != 8'd0, a[22:0], 3'b000};
    r.mb = {b[30:23] != 8'd0, b[22:0], 3'b000};
    return r;
  endfunction

  function automatic fadd_p2_t fp_add_s2(input fadd_p1_t r);
    fadd_p2_t q;
    logic [26:0] mb;
    logic stk;
    q.sp = r.sp; q.sv = r.sv; q.sa = r.sa; q.sub = r.sub;
    q.e = r.e + 8'd1;                        // pre-incremented for the 28-bit normalize in s3
    mb = r.mb;
    if (r.d > 8'd26) begin
      mb = 27'd1;
    end else if (r.d != 0) begin
      stk = sticky_below({5'd0, mb}, 6'(r.d));
      mb = (mb >> r.d) | {26'd0, stk};
    end
    q.sum = r.sub ? ({1'b0, r.ma} - {1'b0, mb}) : ({1'b0, r.ma} + {1'b0, mb});
    return q;
  endfunction

  // One normalize for add and sub over the 28-bit sum, counted and shifted together by norm28.
  // An add carry-out gives n = 0 (the sticky OR folds sum[1:0]); otherwise sum[26] is set, n = 1,
  // and y[0] is 0. A sub never carries out, so n = lzc27(sum[26:0]) + 1, which the
  // pre-incremented q.e absorbs. A zero add (0+0) leaves mn = 0 and e = 1 - 31 < 0: s4 returns
  // {sa, 0}, keeping the -0 of (-0)+(-0). A zero sub is the cancellation special.
  function automatic fadd_nm_t fp_add_s3(input fadd_p2_t q);
    fadd_nm_t n;
    norm28_t nz;
    n.sp = q.sp;
    n.sv = q.sv;
    n.s = q.sa;
    if (q.sub && q.sum == 0 && !q.sp) begin
      n.sp = 1'b1;
      n.sv = FADD_SV_ZERO;
    end
    nz = norm28(q.sum);
    n.mn = {nz.y[27:2], nz.y[1] | nz.y[0]};
    n.e = 10'(q.e) - 10'(nz.n);
    return n;
  endfunction

  function automatic f32_t fp_add_s4(input fadd_nm_t n);
    logic g, rs, c;
    logic signed [9:0] e0;
    logic [23:0] mm;
    logic [24:0] mr;
    if (n.sp) return fadd_sv(n.sv);
    e0 = n.e;
    g  = n.mn[2];
    rs = n.mn[1] | n.mn[0];
    mm = n.mn[26:3];
    mr = {1'b0, mm} + ((g && (rs || mm[0])) ? 25'd1 : 25'd0);
    c = mr[24];   // as in fp_mul_s2: no post-round shift, flags off e0
    if (e0 >= 10'sd255 || (e0 == 10'sd254 && c)) return {n.s, 8'hFF, 23'd0};
    if (e0 < 10'sd0 || (e0 == 10'sd0 && !c)) return {n.s, 31'd0};
    return {n.s, 8'(e0) + 8'(c), mr[22:0]};
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

  function automatic logic fp_gt(input f32_t a, input f32_t b);
    return fkey(a) > fkey(b);
  endfunction

  function automatic f32_t fp_max(input f32_t a, input f32_t b);
    return fp_gt(a, b) ? ftz(a) : ftz(b);
  endfunction

  function automatic f32_t fp_min(input f32_t a, input f32_t b);
    return fp_gt(b, a) ? ftz(a) : ftz(b);
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

  // Round half to even, saturate to [-127, 127]; in two stages: shift | round + saturate.
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
    q.zero = (e < 126);
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

  function automatic f32_t fp_recip(input f32_t x_in);
    f32_t x, ax, y;
    x = ftz(x_in);
    ax = {1'b0, x[30:0]};
    if (ax == 0) return F_ZERO;
    if (ax >= 32'h7E80_0000) return {x[31], 31'b0};      // |x| >= 2^126 (incl. inf): flushes
    y = ftz(RECIP_MAGIC - ax);
    for (int k = 0; k < 3; k++) y = fp_mul(y, fp_sub(F_TWO, fp_mul(ax, y)));
    return x[31] ? fneg(y) : y;
  endfunction

  function automatic f32_t fp_rsqrt(input f32_t x_in);
    f32_t x, y, h;
    x = ftz(x_in);
    if (x[31] || x[30:0] == 0 || x == 32'h7F80_0000) return F_ZERO;
    y = RSQRT_MAGIC - (x >> 1);
    h = fp_mul(F_HALF, x);
    for (int k = 0; k < 3; k++) y = fp_mul(y, fp_sub(F_1P5, fp_mul(h, fp_mul(y, y))));
    return y;
  endfunction

endpackage
