// Lint-only stand-in for the Xilinx DSP48E1 primitive (unisim) with the ports otpu_mxu's systolic
// cells (MXU_IMPL 2) connect under SYNTHESIS, and no behaviour: `make lint` defines SYNTHESIS, so
// the tops are linted with the primitive the builds use. Vivado builds with the real one.
module DSP48E1 #(
  parameter A_INPUT = "DIRECT", B_INPUT = "DIRECT", USE_DPORT = "FALSE", USE_MULT = "MULTIPLY",
  parameter USE_SIMD = "ONE48", AREG = 1, ACASCREG = 1, BREG = 1, BCASCREG = 1, CREG = 1,
  parameter DREG = 1, ADREG = 1, MREG = 1, PREG = 1, INMODEREG = 1, OPMODEREG = 1,
  parameter ALUMODEREG = 1, CARRYINREG = 1, CARRYINSELREG = 1,
  parameter USE_PATTERN_DETECT = "NO_PATDET", AUTORESET_PATDET = "NO_RESET",
  parameter logic [47:0] MASK = 48'h3fffffffffff, PATTERN = 48'h0,
  parameter SEL_MASK = "MASK", SEL_PATTERN = "PATTERN"
) (
  input  logic        CLK,
  input  logic [29:0] A, ACIN,
  input  logic [17:0] B, BCIN,
  input  logic [47:0] C, PCIN,
  input  logic [24:0] D,
  input  logic [4:0]  INMODE,
  input  logic [6:0]  OPMODE,
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
  output logic        CARRYCASCOUT, MULTSIGNOUT, OVERFLOW, UNDERFLOW, PATTERNDETECT, PATTERNBDETECT
);
  assign {ACOUT, BCOUT, PCOUT, P, CARRYOUT, CARRYCASCOUT, MULTSIGNOUT, OVERFLOW, UNDERFLOW,
          PATTERNDETECT, PATTERNBDETECT} = '0;
endmodule
