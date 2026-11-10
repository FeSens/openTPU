// SIMULATION ONLY (run_vl.sh): a behavioural DSP48E2 for Verilator, the subset the pumped packed
// column (otpu_ppk_pe) and the column ends (rtl/tech/xilinxus otpu_mxce) use. AMD's unisim model
// is the reference (run_xsim.sh); this one follows UG579's datapath:
//   A1 / A2 (AREG 0-2, CEA1 / CEA2; INMODE[0] picks A1 for the multiplier), B1 / B2 likewise
//   (INMODE[4]), D (DREG, CED), the pre-adder AD = (INMODE[2] ? D : 0) +/- (INMODE[1] ? 0 : A)
//   (INMODE[3] subtracts: D - A; PREADDINSEL "A" only), its register (ADREG, CEAD), the 27 x 18
//   multiplier (AMULTSEL "A" or "AD", BMULTSEL "B"), M (MREG, CEM), INMODE / OPMODE registers
//   (INMODEREG / CEINMODE, OPMODEREG / CECTRL), the ALU at ALUMODE 0000 (P = W + X + Y + Z) with
//   X in {0, M, P}, Y in {0, M}, Z in {0, PCIN, P, PCIN >>> 17, P >>> 17}, W in {0, P, RND}, and P
//   (PREG, CEP). Anything else stops the simulation. Resets are not modelled (tied 0 here).
/* verilator lint_off UNUSEDPARAM */
module DSP48E2 #(
  parameter string A_INPUT = "DIRECT", B_INPUT = "DIRECT", AMULTSEL = "A", BMULTSEL = "B",
  parameter string PREADDINSEL = "A", USE_MULT = "MULTIPLY", USE_SIMD = "ONE48",
  parameter string USE_WIDEXOR = "FALSE",
  parameter AREG = 1, ACASCREG = 1, BREG = 1, BCASCREG = 1, CREG = 1, DREG = 1, ADREG = 1,
  parameter MREG = 1, PREG = 1, INMODEREG = 1, OPMODEREG = 1, ALUMODEREG = 1, CARRYINREG = 1,
  parameter CARRYINSELREG = 1,
  parameter string USE_PATTERN_DETECT = "NO_PATDET", AUTORESET_PATDET = "NO_RESET",
  parameter logic [47:0] MASK = 48'h3fffffffffff, PATTERN = 48'h0,
  parameter string SEL_MASK = "MASK", SEL_PATTERN = "PATTERN", parameter logic [47:0] RND = 48'h0
) (
  input  logic        CLK,
  input  logic [29:0] A, ACIN,
  input  logic [17:0] B, BCIN,
  input  logic [47:0] C, PCIN,
  input  logic [26:0] D,
  input  logic [4:0]  INMODE,
  input  logic [8:0]  OPMODE,
  input  logic [3:0]  ALUMODE,
  input  logic [2:0]  CARRYINSEL,
  input  logic        CARRYIN, CARRYCASCIN, MULTSIGNIN,
  input  logic        CEA1, CEA2, CEB1, CEB2, CEC, CED, CEAD, CEM, CEP, CEALUMODE, CECTRL,
  input  logic        CECARRYIN, CEINMODE,
  input  logic        RSTA, RSTB, RSTC, RSTD, RSTM, RSTP, RSTALLCARRYIN, RSTALUMODE, RSTCTRL,
  input  logic        RSTINMODE,
  output logic [29:0] ACOUT,
  output logic [17:0] BCOUT,
  output logic [47:0] PCOUT, P,
  output logic [3:0]  CARRYOUT,
  output logic [7:0]  XOROUT,
  output logic        CARRYCASCOUT, MULTSIGNOUT, OVERFLOW, UNDERFLOW, PATTERNDETECT,
  output logic        PATTERNBDETECT
);
  initial begin
    if (A_INPUT != "DIRECT" || B_INPUT != "DIRECT" || PREADDINSEL != "A" || BMULTSEL != "B" ||
        USE_SIMD != "ONE48" || (AMULTSEL != "A" && AMULTSEL != "AD") ||
        (USE_MULT != "MULTIPLY" && USE_MULT != "NONE"))
      $fatal(1, "DSP48E2 (behavioural): unsupported configuration");
  end
  logic [29:0] a1, a2;
  logic [17:0] b1, b2;
  logic [26:0] dr, adr;
  logic [44:0] mr;
  logic [47:0] pr;
  logic [4:0]  inr;
  logic [8:0]  opr;
  wire  [4:0]  inm = INMODEREG ? inr : INMODE;
  wire  [8:0]  opm = OPMODEREG ? opr : OPMODE;
  // the multiplier's A and B paths (UG579: INMODE[0] / [4] pick A1 / B1, else A2 / B2)
  wire  [29:0] a_1 = (AREG == 2) ? a1 : A;
  wire  [29:0] a_2 = (AREG == 0) ? A : a2;
  wire  [29:0] a_m = inm[0] ? a_1 : a_2;
  wire  [17:0] b_1 = (BREG == 2) ? b1 : B;
  wire  [17:0] b_2 = (BREG == 0) ? B : b2;
  wire  [17:0] b_m = inm[4] ? b_1 : b_2;
  wire  [26:0] d_q = DREG ? dr : D;
  wire  [26:0] a_p = inm[1] ? 27'd0 : a_m[26:0];
  wire  [26:0] d_p = inm[2] ? d_q : 27'd0;
  wire  [26:0] ad_c = inm[3] ? d_p - a_p : d_p + a_p;
  wire  [26:0] ad_q = ADREG ? adr : ad_c;
  wire  [26:0] mul_a = (AMULTSEL == "AD") ? ad_q : a_m[26:0];
  wire  [44:0] m_c = (USE_MULT == "NONE") ? 45'd0     // operands extended to the product's 45 bits
                   : $signed({{18{mul_a[26]}}, mul_a}) * $signed({{27{b_m[17]}}, b_m});
  wire  [44:0] m_q = MREG ? mr : m_c;
  wire  [47:0] m48 = {{3{m_q[44]}}, m_q};
  logic [47:0] x, y, z, w, p_c;
  always_comb begin
    case (opm[1:0])
      2'b00: x = '0;
      2'b01: x = m48;
      2'b10: x = pr;
      default: x = 'x;
    endcase
    case (opm[3:2])
      2'b00: y = '0;
      2'b01: y = '0;                       // with X = M: the two partial products, M in all
      default: y = 'x;
    endcase
    case (opm[6:4])
      3'b000: z = '0;
      3'b001: z = PCIN;
      3'b010: z = pr;
      3'b101: z = 48'($signed(PCIN) >>> 17);
      3'b110: z = 48'($signed(pr) >>> 17);
      default: z = 'x;
    endcase
    case (opm[8:7])
      2'b00: w = '0;
      2'b01: w = pr;
      2'b10: w = RND;
      default: w = 'x;
    endcase
    p_c = w + x + y + z;
  end
  always_ff @(posedge CLK) begin
    if (CEA1) a1 <= A;
    if (CEA2) a2 <= (AREG == 2) ? a1 : A;
    if (CEB1) b1 <= B;
    if (CEB2) b2 <= (BREG == 2) ? b1 : B;
    if (CED) dr <= D;
    if (CEAD) adr <= ad_c;
    if (CEM) mr <= m_c;
    if (CEINMODE) inr <= INMODE;
    if (CECTRL) opr <= OPMODE;
    if (CEP) pr <= p_c;
  end
  // the subset: M only through X = Y = 01, ALUMODE 0000
  always @(posedge CLK)
    if ((opm[1:0] == 2'b01) != (opm[3:2] == 2'b01) || opm[1:0] == 2'b11 || opm[3:2] > 2'b01 ||
        opm[6:4] == 3'b011 || opm[6:4] == 3'b100 || opm[6:4] == 3'b111 || opm[8:7] == 2'b11 ||
        ALUMODE != 4'b0000)
      $fatal(1, "DSP48E2 (behavioural): OPMODE %b ALUMODE %b not modelled", opm, ALUMODE);
  assign P = PREG ? pr : p_c;
  assign PCOUT = P;
  assign ACOUT = (ACASCREG == 0) ? A : a2;
  assign BCOUT = (BCASCREG == 0) ? B : b2;
  assign {CARRYOUT, XOROUT, CARRYCASCOUT, MULTSIGNOUT, OVERFLOW, UNDERFLOW, PATTERNDETECT,
          PATTERNBDETECT} = '0;
endmodule
/* verilator lint_on UNUSEDPARAM */
