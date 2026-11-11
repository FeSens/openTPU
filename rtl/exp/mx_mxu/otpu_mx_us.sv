// EXPERIMENTAL (docs/mx_mxu.md): not in any file list, not in the board build.
//
// The MX MXU column on UltraScale+ (DSP48E2; xcku5p as the stand-in for the BCU1525's
// VU9P): one physical column whose 128 DSPs make two products each at the core clock
// by INT8 packing, and two accumulating column ends. The 7-series version (pumped) is
// otpu_mx_x7.sv, which also states the numerics.
//
// Packing (docs/dsp_pumping.md 6(c)): A = x1 * 2^19 on the 27-bit pre-adder, D = x0, B = w, so
// the product is x1*w*2^19 + x0*w and a 16-long PCIN chain sums P = S1 * 2^19 + S0 with
// |S0| <= 16 * 127 * 128 < 2^18 (activations in [-127, 127], weights in [-128, 127]): at the
// chain end S0 = sext(P[18:0]), S1 = P[47:19] + P[18]. Exact, one clock.

package otpu_mx_fam;
  localparam int LATC = 23;
  localparam bit PUMPED = 1'b0;
endpackage

// One clock (clk_f is clk_s: nothing here runs faster).
module otpu_mx_clk (
  input  logic clk_in,
  output logic clk_s,
  output logic clk_f,
  output logic locked
);
  BUFG u_bs (.I(clk_in), .O(clk_s));
  assign clk_f = clk_s;
  assign locked = 1'b1;
endmodule

// The shared weight skew: position i delayed k = i % CLS (otpu_mxu's g_wsk).
module otpu_mx_front #(parameter int D = 128, parameter int CLS = 16) (
  input  logic               clk_s,
  input  logic [D-1:0][15:0] w,
  output logic [D-1:0][15:0] wc
);
  for (genvar i = 0; i < D; i++) begin : g_w
    otpu_skew #(.W(16), .N(i % CLS)) u_w (.clk(clk_s), .en(1'b1), .d(w[i]), .q(wc[i]));
  end
endmodule

// The accumulating column end (DSP48E2): see otpu_mx_x7.sv.
module otpu_mx_colend (
  input  logic              clk,
  input  logic [3:0][23:0]  a,
  input  logic [3:0][23:0]  d,
  input  logic [3:0][16:0]  bm,
  input  logic              first,
  output logic [47:0]       acc
);
  logic [3:0][47:0] pc, pf;
  for (genvar b = 0; b < 4; b++) begin : g_d
    DSP48E2 #(
      .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .AMULTSEL("AD"), .BMULTSEL("B"),
      .PREADDINSEL("A"), .USE_MULT("MULTIPLY"), .USE_WIDEXOR("FALSE"),
      .USE_SIMD("ONE48"), .AREG(1), .ACASCREG(1), .BREG(1), .BCASCREG(1), .CREG(0), .DREG(1),
      .ADREG(0), .MREG(1), .PREG(b == 3 ? 1 : 0), .INMODEREG(0), .OPMODEREG(0), .ALUMODEREG(0),
      .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
      .AUTORESET_PATDET("NO_RESET"), .MASK(48'h3fffffffffff), .PATTERN(48'h0),
      .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
    ) u_dsp (
      .CLK(clk),
      .A({{6{a[b][23]}}, a[b]}), .B({1'b0, bm[b]}), .C(48'd0), .D({{3{d[b][23]}}, d[b]}),
      .INMODE(5'b00100), .OPMODE(b == 0 ? 9'b00_000_01_01 : 9'b00_001_01_01),
      .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
      .CEA1(1'b1), .CEA2(1'b1), .CEB1(1'b1), .CEB2(1'b1), .CEC(1'b0), .CED(1'b1), .CEAD(1'b0),
      .CEM(1'b1), .CEP(b == 3), .CEALUMODE(1'b0), .CECTRL(1'b0), .CECARRYIN(1'b0),
      .CEINMODE(1'b0),
      .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
      .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
      .ACIN(30'd0), .BCIN(18'd0), .PCIN(b == 0 ? 48'd0 : pc[b == 0 ? 0 : b - 1]),
      .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
      .ACOUT(), .BCOUT(), .PCOUT(pc[b]), .P(pf[b]), .CARRYCASCOUT(), .MULTSIGNOUT(),
      .CARRYOUT(), .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT(), .XOROUT());
  end
  DSP48E2 #(
    .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .AMULTSEL("A"), .BMULTSEL("B"),
    .PREADDINSEL("A"), .USE_MULT("NONE"), .USE_WIDEXOR("FALSE"),
    .USE_SIMD("ONE48"), .AREG(0), .ACASCREG(0), .BREG(0), .BCASCREG(0), .CREG(0), .DREG(0),
    .ADREG(0), .MREG(0), .PREG(1), .INMODEREG(0), .OPMODEREG(1), .ALUMODEREG(0),
    .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
    .AUTORESET_PATDET("NO_RESET"), .MASK(48'h3fffffffffff), .PATTERN(48'h0),
    .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
  ) u_acc (
    .CLK(clk),
    .A(30'd0), .B(18'd0), .C(48'd0), .D(27'd0),
    .INMODE(5'b00000), .OPMODE(first ? 9'b00_001_00_00 : 9'b00_001_00_10),
    .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
    .CEA1(1'b0), .CEA2(1'b0), .CEB1(1'b0), .CEB2(1'b0), .CEC(1'b0), .CED(1'b0), .CEAD(1'b0),
    .CEM(1'b0), .CEP(1'b1), .CEALUMODE(1'b0), .CECTRL(1'b1), .CECARRYIN(1'b0),
    .CEINMODE(1'b0),
    .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
    .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
    .ACIN(30'd0), .BCIN(18'd0), .PCIN(pc[3]), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
    .ACOUT(), .BCOUT(), .PCOUT(), .P(acc), .CARRYCASCOUT(), .MULTSIGNOUT(),
    .CARRYOUT(), .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT(), .XOROUT());
endmodule

// A packed position: DSP48E2, A = x1 * 2^19 and D = x0 into the pre-adder (AREG / DREG 1),
// B = w (BREG 1), M registered, P = PCIN + M (FIRST: P = M), PREG on (the chain end too: its
// split is in fabric).
module otpu_pack_pe #(parameter bit FIRST = 1'b0) (
  input  logic               clk,
  input  logic [7:0]         x0,
  input  logic [7:0]         x1,
  input  logic [7:0]         w,
  input  logic [47:0]        pcin,
  output logic [47:0]        pcout,
  output logic [47:0]        p
);
  DSP48E2 #(
    .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .AMULTSEL("AD"), .BMULTSEL("B"),
    .PREADDINSEL("A"), .USE_MULT("MULTIPLY"), .USE_WIDEXOR("FALSE"),
    .USE_SIMD("ONE48"), .AREG(1), .ACASCREG(1), .BREG(1), .BCASCREG(1), .CREG(0), .DREG(1),
    .ADREG(0), .MREG(1), .PREG(1), .INMODEREG(0), .OPMODEREG(0), .ALUMODEREG(0),
    .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
    .AUTORESET_PATDET("NO_RESET"), .MASK(48'h3fffffffffff), .PATTERN(48'h0),
    .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
  ) u_dsp (
    .CLK(clk),
    .A({{3{x1[7]}}, x1, 19'd0}), .B({{10{w[7]}}, w}), .C(48'd0), .D({{19{x0[7]}}, x0}),
    .INMODE(5'b00100), .OPMODE(FIRST ? 9'b00_000_01_01 : 9'b00_001_01_01),
    .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
    .CEA1(1'b1), .CEA2(1'b1), .CEB1(1'b1), .CEB2(1'b1), .CEC(1'b0), .CED(1'b1), .CEAD(1'b0),
    .CEM(1'b1), .CEP(1'b1), .CEALUMODE(1'b0), .CECTRL(1'b0), .CECARRYIN(1'b0),
    .CEINMODE(1'b0),
    .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
    .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
    .ACIN(30'd0), .BCIN(18'd0), .PCIN(pcin), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
    .ACOUT(), .BCOUT(), .PCOUT(pcout), .P(p), .CARRYCASCOUT(), .MULTSIGNOUT(), .CARRYOUT(),
    .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT(), .XOROUT());
endmodule

// One physical column, two stationary rows, COL hops from the front, all at clk_s: packed PE
// chains (position k's operands k + COL cycles late), the chain ends split in fabric into the
// two rows' sums, two accumulating column ends. Ports as the 7-series otpu_mx_col2.
module otpu_mx_col2 #(parameter int D = 128, parameter int CLS = 16, parameter int COL = 0) (
  input  logic                  clk_s,
  input  logic                  clk_f,
  input  logic                  rst_s,
  input  logic [D-1:0][7:0]     x0,
  input  logic [D-1:0][7:0]     x1,
  input  logic [D-1:0][15:0]    wc_prev,
  output logic [D-1:0][15:0]    wc,
  input  logic                  hsel,
  input  logic [31:0]           sw_lo,
  input  logic [31:0]           sw_hi,
  input  logic [31:0]           da0,
  input  logic [31:0]           da1,
  input  logic                  fmt,
  input  logic                  v,
  input  logic                  first,
  input  logic                  last,
  output logic [47:0]           acc0,
  output logic [47:0]           acc1,
  output logic                  av
);
  localparam int NCH = D / CLS;
  initial if (NCH != 8 || CLS != 16) $fatal(1, "otpu_mx_col2: D 128, CLS 16 only");
  logic [D-1:0][15:0] hop;                     // (the last column's: unused, trimmed)
  always_ff @(posedge clk_s) hop <= wc_prev;
  assign wc = hop;

  // chain ends: P of position 15, registered (chunk t's in cycle t + 18 + COL), split
  logic [NCH-1:0][23:0] s0c, s1c;
  for (genvar c = 0; c < NCH; c++) begin : g_ch
    logic [47:0] pc [CLS];
    logic [47:0] pe;
    for (genvar k = 0; k < CLS; k++) begin : g_k
      localparam int I = c * CLS + k;
      logic [7:0] a0, a1;
      otpu_skew #(.W(8), .N(k + COL)) u_x0 (.clk(clk_s), .en(1'b1), .d(x0[I]), .q(a0));
      otpu_skew #(.W(8), .N(k + COL)) u_x1 (.clk(clk_s), .en(1'b1), .d(x1[I]), .q(a1));
      wire [7:0] w = hsel ? wc_prev[I][15:8] : wc_prev[I][7:0];
      logic [47:0] pk;
      otpu_pack_pe #(.FIRST(k == 0)) u_pe (
        .clk(clk_s), .x0(a0), .x1(a1), .w,
        .pcin(k == 0 ? 48'd0 : pc[k == 0 ? 0 : k - 1]), .pcout(pc[k]), .p(pk));
      if (k == CLS - 1) begin : g_end
        assign pe = pk;
      end
    end
    assign s0c[c] = {{5{pe[18]}}, pe[18:0]};
    assign s1c[c] = 24'(pe[47:19] + 29'(pe[18]));
  end
  logic [3:0][23:0] a0e, d0e, a1e, d1e;
  always_comb
    for (int b = 0; b < 4; b++) begin
      a0e[b] = s0c[2*b]; d0e[b] = s0c[2*b+1];
      a1e[b] = s1c[2*b]; d1e[b] = s1c[2*b+1];
    end

  wire [31:0] sws = hsel ? sw_hi : sw_lo;
  logic [32:0] si;
  logic [31:0] dq0, dq1;
  otpu_skew #(.W(33), .N(18 + COL)) u_s (.clk(clk_s), .en(1'b1), .d({fmt, sws}), .q(si));
  otpu_skew #(.W(32), .N(18 + COL)) u_d0 (.clk(clk_s), .en(1'b1), .d(da0), .q(dq0));
  otpu_skew #(.W(32), .N(18 + COL)) u_d1 (.clk(clk_s), .en(1'b1), .d(da1), .q(dq1));
  function automatic logic [3:0][16:0] bms(input logic [32:0] s, input logic [31:0] da);
    for (int b = 0; b < 4; b++) begin
      logic signed [9:0] sh;
      logic [5:0] m;
      sh = (s[32] ? 10'sd16 : 10'sd11) - 10'(s[7:0]) - 10'(da[8*b +: 8]);
      m = s[32] ? 6'd1 : s[8 + 6*b +: 6];
      bms[b] = (sh < 0) ? 17'd0 : (17'(m) << sh[4:0]);
    end
  endfunction
  logic f, vl;
  otpu_skew #(.W(1), .N(20 + COL)) u_f (.clk(clk_s), .en(1'b1), .d(v && first), .q(f));
  otpu_skew #(.W(1), .N(23 + COL)) u_l (.clk(clk_s), .en(1'b1), .d(v && last), .q(vl));
  logic [47:0] p0, p1, q0, q1;
  otpu_mx_colend u_ce0 (.clk(clk_s), .a(a0e), .d(d0e), .bm(bms(si, dq0)), .first(f), .acc(p0));
  otpu_mx_colend u_ce1 (.clk(clk_s), .a(a1e), .d(d1e), .bm(bms(si, dq1)), .first(f), .acc(p1));
  always_ff @(posedge clk_s) begin
    q0 <= p0;
    q1 <= p1;
  end
  assign acc0 = q0;
  assign acc1 = q1;
  assign av = vl;
endmodule
