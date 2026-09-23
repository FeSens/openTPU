// Pipelined fp32 operators built from the staged functions of otpu_fp. Every register advances
// with `en`: a unit freezes as a whole while its TMEM grant is withheld.
//
//   otpu_fmul    y = a * b                LAT >= 2 (s1 | s2, extra cycles appended)
//   otpu_fadd    y = a + b                LAT >= 4 (s1 | s2 | s3 | s4, extra cycles appended)
//   otpu_fmadd   y = (a * b) + c          two roundings, as in the ISA; LAT = LM + LA
//   otpu_delay   N-cycle delay line (SRL-friendly: no reset)

module otpu_delay #(parameter int W = 32, parameter int N = 1) (
  input  logic         clk,
  input  logic         en,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  if (N == 0) begin : g_wire
    assign q = d;
  end else begin : g_regs
    logic [W-1:0] r [N];
    always_ff @(posedge clk) if (en) begin
      r[0] <= d;
      for (int k = 1; k < N; k++) r[k] <= r[k-1];
    end
    assign q = r[N-1];
  end
endmodule

module otpu_fmul
  import otpu_fp::*;
#(parameter int LAT = 2) (
  input  logic  clk,
  input  logic  en,
  input  f32_t  a,
  input  f32_t  b,
  output f32_t  y
);
  fmul_mid_t m;
  f32_t r;
  always_ff @(posedge clk) if (en) begin
    m <= fp_mul_s1(a, b);
    r <= fp_mul_s2(m);
  end
  otpu_delay #(.W(32), .N(LAT - 2)) u_pad (.clk, .en, .d(r), .q(y));
endmodule

module otpu_fadd
  import otpu_fp::*;
#(parameter int LAT = 4) (
  input  logic  clk,
  input  logic  en,
  input  f32_t  a,
  input  f32_t  b,
  output f32_t  y
);
  fadd_p1_t s1;
  fadd_p2_t s2;
  fadd_nm_t s3;
  f32_t r;
  always_ff @(posedge clk) if (en) begin
    s1 <= fp_add_s1(a, b);
    s2 <= fp_add_s2(s1);
    s3 <= fp_add_s3(s2);
    r  <= fp_add_s4(s3);
  end
  otpu_delay #(.W(32), .N(LAT - 4)) u_pad (.clk, .en, .d(r), .q(y));
endmodule

module otpu_fmadd
  import otpu_fp::*;
#(parameter int LM = 2, parameter int LA = 4) (
  input  logic  clk,
  input  logic  en,
  input  f32_t  a,
  input  f32_t  b,
  input  f32_t  c,
  output f32_t  y
);
  f32_t p, cd;
  otpu_fmul #(.LAT(LM)) u_m (.clk, .en, .a, .b, .y(p));
  otpu_delay #(.W(32), .N(LM)) u_c (.clk, .en, .d(c), .q(cd));
  otpu_fadd #(.LAT(LA)) u_a (.clk, .en, .a(p), .b(cd), .y);
endmodule
