// LINT ONLY (run_vl.sh lint): a stand-in for the unisim MMCME4_ADV with the ports ppk_top connects
// and no behaviour (the testbench clocks ppk_core itself; Vivado builds with the real one).
/* verilator lint_off UNUSEDPARAM */
module MMCME4_ADV #(
  parameter BANDWIDTH = "OPTIMIZED", COMPENSATION = "INTERNAL", parameter int DIVCLK_DIVIDE = 1,
  parameter real CLKFBOUT_MULT_F = 5.0, CLKFBOUT_PHASE = 0.0, CLKIN1_PERIOD = 0.0,
  parameter real CLKOUT0_DIVIDE_F = 1.0, CLKOUT0_PHASE = 0.0, CLKOUT0_DUTY_CYCLE = 0.5,
  parameter int CLKOUT1_DIVIDE = 1, parameter real CLKOUT1_PHASE = 0.0,
  parameter real CLKOUT1_DUTY_CYCLE = 0.5
) (
  input  logic        CLKIN1, CLKIN2, CLKINSEL, CLKFBIN, PWRDWN, RST, CDDCREQ,
  input  logic [6:0]  DADDR,
  input  logic        DCLK, DEN, DWE, PSCLK, PSEN, PSINCDEC,
  input  logic [15:0] DI,
  output logic [15:0] DO,
  output logic        CLKFBOUT, CLKFBOUTB, CLKOUT0, CLKOUT0B, CLKOUT1, CLKOUT1B, CLKOUT2,
  output logic        CLKOUT2B, CLKOUT3, CLKOUT3B, CLKOUT4, CLKOUT5, CLKOUT6, LOCKED,
  output logic        CDDCDONE, DRDY, PSDONE, CLKINSTOPPED, CLKFBSTOPPED
);
  assign {DO, CLKFBOUT, CLKFBOUTB, CLKOUT0, CLKOUT0B, CLKOUT1, CLKOUT1B, CLKOUT2, CLKOUT2B,
          CLKOUT3, CLKOUT3B, CLKOUT4, CLKOUT5, CLKOUT6, LOCKED, CDDCDONE, DRDY, PSDONE,
          CLKINSTOPPED, CLKFBSTOPPED} = '0;
endmodule
/* verilator lint_on UNUSEDPARAM */
