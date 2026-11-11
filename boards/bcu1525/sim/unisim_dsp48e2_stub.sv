// Lint-only stand-in for the Xilinx DSP48E2 primitive (unisim) with the ports the systolic MXU's
// cells (rtl/tech/xilinxus) connect, and no behaviour: `make lint` lints the top with the
// primitive the builds use (Vivado builds with the real one; tests/xsim checks the cells against
// AMD's model, docs/bcu1525.md section 7).
module DSP48E2 #(
  parameter A_INPUT = "DIRECT", B_INPUT = "DIRECT", AMULTSEL = "A", BMULTSEL = "B",
  parameter PREADDINSEL = "A", USE_MULT = "MULTIPLY", USE_SIMD = "ONE48", USE_WIDEXOR = "FALSE",
  parameter AREG = 1, ACASCREG = 1, BREG = 1, BCASCREG = 1, CREG = 1, DREG = 1, ADREG = 1,
  parameter MREG = 1, PREG = 1, INMODEREG = 1, OPMODEREG = 1, ALUMODEREG = 1, CARRYINREG = 1,
  parameter CARRYINSELREG = 1,
  parameter USE_PATTERN_DETECT = "NO_PATDET", AUTORESET_PATDET = "NO_RESET",
  parameter logic [47:0] MASK = 48'h3fffffffffff, PATTERN = 48'h0,
  parameter SEL_MASK = "MASK", SEL_PATTERN = "PATTERN", parameter logic [47:0] RND = 48'h0
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
  assign {ACOUT, BCOUT, PCOUT, P, CARRYOUT, XOROUT, CARRYCASCOUT, MULTSIGNOUT, OVERFLOW,
          UNDERFLOW, PATTERNDETECT, PATTERNBDETECT} = '0;
endmodule
