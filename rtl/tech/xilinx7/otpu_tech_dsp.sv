// The systolic MXU's arithmetic cells (otpu_mxu IMPL 2) on Xilinx 7-series DSP48E1s (the
// YPCB-00338's xc7k480t). rtl/tech/generic holds the equivalent behavioural models (the
// simulators'), rtl/tech/xilinxus the UltraScale+ DSP48E2 version; the file list picks one
// (opentpu/target.py, docs/targets.md).

// One position of the systolic MXU (IMPL 2): a DSP48E1 with the activation in its B register
// (BREG 1), the two weight streams on A (low) and D (high), one selected by the pre-adder's input
// gates (INMODE: sel 0 -> A, 1 -> D), the product in M and the chain's running sum in P:
// P = PCIN + M (FIRST: P = M). A chain's LAST position leaves P unregistered: the column end's A
// / D input registers take its place (otpu_mxce), so the route to the column end is not in
// front of its pre-adder and multiplier. The select is registered in the DSP (INMODEREG, loaded
// with en): sel is its next value, so the fabric's route to the DSP column ends at the INMODE
// register instead of before the pre-adder and multiplier (133.33 MHz, 812bb01: hi0 -> INMODE,
// 0 levels, 4.4 ns of route and a 2.4 ns setup, +0.174 ns). WREG registers the weights too (A
// and D registers, loaded with en: wlo and whi are their next values). rtl/tech/generic holds the
// equivalent behavioural model.
// PK (otpu_mxu at WG > 1, docs/weight_groups.md): D = whi * 2^16 and sel 1 adds it to A
// in the pre-adder (INMODE: A never gated), the product act * (wlo + whi * 2^16), the chain's sum
// 32 bits wide (p); under sel a chain's FIRST position adds C = 2^15 (OPMODE's Z, its register
// loaded with INMODE's), so the column end splits a sum by its bits.
module otpu_pe #(parameter bit FIRST = 1'b0, parameter bit LAST = 1'b0, parameter bit WREG = 1'b0,
                 parameter bit PK = 1'b0) (
  input  logic               clk,
  input  logic               en,
  input  logic [7:0]         act,
  input  logic [7:0]         wlo,               // WREG: the weights from the next cycle on
  input  logic [7:0]         whi,
  input  logic               sel,               // the select from the next cycle on
  input  logic [47:0]        pcin,
  output logic [47:0]        pcout,
  output logic signed [(PK ? 32 : 24)-1:0] p
);
  logic [47:0] pf;
  DSP48E1 #(
    .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .USE_DPORT("TRUE"), .USE_MULT("MULTIPLY"),
    .USE_SIMD("ONE48"), .AREG(WREG ? 1 : 0), .ACASCREG(WREG ? 1 : 0), .BREG(1), .BCASCREG(1),
    .CREG(0), .DREG(WREG ? 1 : 0),
    .ADREG(0), .MREG(1), .PREG(LAST ? 0 : 1), .INMODEREG(1), .OPMODEREG((FIRST && PK) ? 1 : 0),
    .ALUMODEREG(0), .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
    .AUTORESET_PATDET("NO_RESET"), .MASK(48'h3fffffffffff), .PATTERN(48'h0),
    .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
  ) u_dsp (
    .CLK(clk),
    .A({{22{wlo[7]}}, wlo}), .B({{10{act[7]}}, act}), .C((FIRST && PK) ? 48'h8000 : 48'd0),
    .D(PK ? {whi[7], whi, 16'd0} : {{17{whi[7]}}, whi}),
    .INMODE({2'b00, sel, PK ? 1'b0 : sel, 1'b0}),
    .OPMODE(FIRST ? {(PK && sel) ? 3'b011 : 3'b000, 4'b01_01} : 7'b001_01_01),
    .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
    .CEA1(1'b0), .CEA2(WREG ? en : 1'b0), .CEB1(en), .CEB2(en), .CEC(1'b0),
    .CED(WREG ? en : 1'b0), .CEAD(1'b0),
    .CEM(en), .CEP(LAST ? 1'b0 : en), .CEALUMODE(1'b0), .CECTRL((FIRST && PK) ? en : 1'b0),
    .CECARRYIN(1'b0),
    .CEINMODE(en),
    .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
    .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
    .ACIN(30'd0), .BCIN(18'd0), .PCIN(pcin), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
    .ACOUT(), .BCOUT(), .PCOUT(pcout), .P(pf), .CARRYCASCOUT(), .MULTSIGNOUT(), .CARRYOUT(),
    .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT());
  assign p = pf[(PK ? 32 : 24)-1:0];
endmodule


// The MX column end (otpu_mxu, docs/mx_mxu.md): nine DSP48E1s. DSP g (g < 8) takes group g's
// chain end on A (and, at two chains a group, the second on D: the pre-adder sums them; den 0
// gates D off, INMODE) and its
// B = m << sh on B (unsigned, < 2^17), all registered at E, and multiplies into M. Groups 0..3:
// M at E+1, their sum down a cascade in the same cycle (P = PCIN + M, DSP 0: P = M), registered
// by DSP 3 at E+2 (p3: the production column end's four cascaded post-adders, which close at
// 133.33 MHz). Groups 4..7 a cycle later (B's second register, BREG 2, and the pre-adder's,
// ADREG, or at one chain a group A's second, AREG 2): M at E+2, P = PCIN + M from p3 on,
// registered by DSP 7 at E+3 (p7). DSP 8 accumulates
// the row: P = (first ? 0 : P) + PCIN at E+4, first in its OPMODE register (loaded at E+3: X is P
// or 0). Every sum stays on the cascades. Exact: |a + d| < 2^24 in the 25-bit pre-adder, B in
// 18 bits signed, a row's sum < 2^47. rtl/tech/generic holds the equivalent behavioural model.
module otpu_mxce #(parameter int CPG = 1) (
  input  logic                clk,
  input  logic [7:0][23:0]    a,        // group g's chain end: the pre-adder's A ...
  input  logic [7:0][23:0]    d,        // ... and D (CPG 2: the group's second chain, or
                                        // otpu_mxu's packed int8 high halves; else unused)
  input  logic                den,      // D into the pre-adder (CPG 2; constant while it streams)
  input  logic [7:0][16:0]    bm,       // B = m << sh, with a / d
  input  logic                first,    // the row's first block (two cycles after a / d / bm)
  output logic [47:0]         acc
);
  initial if (CPG != 1 && CPG != 2) $fatal(1, "otpu_mxce: CPG %0d", CPG);
  logic [8:0][47:0] pc, pf;
  for (genvar g = 0; g < 8; g++) begin : g_m
    // groups 4..7: a register more before the multiplier (the pre-adder's, or A's second)
    localparam bit LATE = (g >= 4);
    localparam int AR = (LATE && CPG == 1) ? 2 : 1;
    DSP48E1 #(
      .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .USE_DPORT(CPG == 2 ? "TRUE" : "FALSE"),
      .USE_MULT("MULTIPLY"), .USE_SIMD("ONE48"), .AREG(AR), .ACASCREG(AR), .BREG(LATE ? 2 : 1),
      .BCASCREG(LATE ? 2 : 1), .CREG(0), .DREG(CPG == 2 ? 1 : 0),
      .ADREG((LATE && CPG == 2) ? 1 : 0), .MREG(1),
      .PREG((g == 3 || g == 7) ? 1 : 0), .INMODEREG(0), .OPMODEREG(0), .ALUMODEREG(0),
      .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
      .AUTORESET_PATDET("NO_RESET"), .MASK(48'h3fffffffffff), .PATTERN(48'h0),
      .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
    ) u_dsp (
      .CLK(clk),
      .A({{6{a[g][23]}}, a[g]}), .B({1'b0, bm[g]}), .C(48'd0),
      .D((CPG == 2) ? {d[g][23], d[g]} : 25'd0),
      .INMODE((CPG == 2) ? {2'b00, den, 2'b00} : 5'b00000),
      .OPMODE(g == 0 ? 7'b000_01_01 : 7'b001_01_01),
      .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
      .CEA1(AR == 2 ? 1'b1 : 1'b0), .CEA2(1'b1), .CEB1(LATE ? 1'b1 : 1'b0), .CEB2(1'b1),
      .CEC(1'b0), .CED(CPG == 2 ? 1'b1 : 1'b0), .CEAD((LATE && CPG == 2) ? 1'b1 : 1'b0),
      .CEM(1'b1), .CEP((g == 3 || g == 7) ? 1'b1 : 1'b0), .CEALUMODE(1'b0), .CECTRL(1'b0),
      .CECARRYIN(1'b0), .CEINMODE(1'b0),
      .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
      .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
      .ACIN(30'd0), .BCIN(18'd0), .PCIN(g == 0 ? 48'd0 : pc[g == 0 ? 0 : g - 1]),
      .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
      .ACOUT(), .BCOUT(), .PCOUT(pc[g]), .P(pf[g]), .CARRYCASCOUT(), .MULTSIGNOUT(),
      .CARRYOUT(), .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT());
  end
  // the accumulator: X = P (0 for a row's first block), Y = 0, Z = PCIN
  DSP48E1 #(
    .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .USE_DPORT("FALSE"), .USE_MULT("NONE"),
    .USE_SIMD("ONE48"), .AREG(0), .ACASCREG(0), .BREG(0), .BCASCREG(0), .CREG(0), .DREG(0),
    .ADREG(0), .MREG(0), .PREG(1), .INMODEREG(0), .OPMODEREG(1), .ALUMODEREG(0),
    .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
    .AUTORESET_PATDET("NO_RESET"), .MASK(48'h3fffffffffff), .PATTERN(48'h0),
    .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
  ) u_acc (
    .CLK(clk),
    .A(30'd0), .B(18'd0), .C(48'd0), .D(25'd0),
    .INMODE(5'b00000), .OPMODE(first ? 7'b001_00_00 : 7'b001_00_10),
    .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
    .CEA1(1'b0), .CEA2(1'b0), .CEB1(1'b0), .CEB2(1'b0), .CEC(1'b0), .CED(1'b0), .CEAD(1'b0),
    .CEM(1'b0), .CEP(1'b1), .CEALUMODE(1'b0), .CECTRL(1'b1), .CECARRYIN(1'b0),
    .CEINMODE(1'b0),
    .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
    .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
    .ACIN(30'd0), .BCIN(18'd0), .PCIN(pc[7]), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
    .ACOUT(), .BCOUT(), .PCOUT(pc[8]), .P(pf[8]), .CARRYCASCOUT(), .MULTSIGNOUT(),
    .CARRYOUT(), .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT());
  assign acc = pf[8];
endmodule


// A position of the pumped packed column (otpu_mxu PUMP, otpu_mxpc): one DSP48E2's registers at
// clk_f = 2 x clk. A2 = wlo and D = whi * 2^16 (loaded with ce_w: the slow edges), AD = A2 + D
// (loaded with ce_ad, held two fast cycles), B = the activation (BF: one register, the fabric's
// b_f in front of it; else two, B1 and B2), M = B * AD and P = PCIN + M (FIRST: P = M + 2^15,
// the DSP's RND), every register on: rtl/tech/xilinxus maps it onto one DSP48E2. The 7-series
// cards never pump the MXU (the tops' MXU_PUMP 0): here the generic model, the families' one
// cell set.
module otpu_ppe #(parameter bit FIRST = 1'b0, parameter bit BF = 1'b1) (
  input  logic        clk_f,
  input  logic        ce_w,
  input  logic        ce_ad,
  input  logic [7:0]  wlo,
  input  logic [7:0]  whi,
  input  logic [7:0]  act,
  input  logic [47:0] pcin,
  output logic [47:0] pcout,
  output logic [31:0] p
);
  logic signed [7:0]  a2, d2, b1, b2;
  logic signed [26:0] ad;
  logic signed [44:0] m;
  logic [47:0]        pr;
  wire  signed [7:0]  b = BF ? b1 : b2;
  always_ff @(posedge clk_f) begin
    if (ce_w) begin
      a2 <= wlo;
      d2 <= whi;
    end
    if (ce_ad) ad <= 27'(a2) + 27'(d2) * 27'sd65536;
    b1 <= act;
    b2 <= b1;
    m <= 45'(b) * 45'(ad);
    pr <= (FIRST ? 48'h8000 : pcin) + 48'(m);
  end
  assign pcout = pr;
  assign p = pr[31:0];
endmodule
