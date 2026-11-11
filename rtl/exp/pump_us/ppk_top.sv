// EXPERIMENTAL (otpu_ppk.sv): the out-of-context tops of the pumped packed MXU test.
//
// ppk_core: NCOL pumped columns (otpu_ppk_col) as otpu_mxu's WG 8 array places them: every
// column takes the same two rows, column 0 skewing them and column j hopping column j - 1's a
// cycle later; column j its own lane's weights (w_in[j]) and the shared scales and first, both
// skewed j more. NCOL 1 is one column (128 + 36 DSP48E2s), NCOL 8 the BCU1525's array (1,024 +
// 288). Every input is registered in clk_s (the ports carry no timing), the outputs too.
// ppk_top: ppk_core with its clocks from one MMCME4_ADV, as bd_wg.tcl makes core_clk and
// acc_clk2x (VCO 1200 MHz = 300 x 4; clk_s = VCO / DIV_S, clk_f = VCO / DIV_F, each through a
// BUFGCE), so Vivado times the real skew between them.
module ppk_core #(
  parameter int NCOL = 1,
  parameter int D = 128,
  parameter bit BF = 1'b1,
  parameter bit HOLD = 1'b0
) (
  input  logic                              clk_s,
  input  logic                              clk_f,
  input  logic                              rst,
  input  logic [D-1:0][7:0]                 x0_in,
  input  logic [D-1:0][7:0]                 x1_in,
  input  logic [NCOL-1:0][D-1:0][7:0]       w_in,
  input  logic [1:0]                        wf_in,
  input  logic [3:0][7:0][16:0]             bm_in,
  input  logic                              first_in,
  output logic [NCOL-1:0][3:0][47:0]        acc_out   // chunk t's (driven in cycle t): row r
);                                                    // in cycle t + 17 + HOLD + r + column
  logic [1:0] rs;
  always_ff @(posedge clk_s) rs <= {rs[0], rst};
  wire rst_s = rs[1];
  logic [D-1:0][7:0]           x0, x1;
  logic [NCOL-1:0][D-1:0][7:0] w;
  logic [1:0]                  wf;
  logic [3:0][7:0][16:0]       bm;
  logic                        first;
  always_ff @(posedge clk_s) begin
    x0 <= x0_in; x1 <= x1_in; w <= w_in; wf <= wf_in; bm <= bm_in; first <= first_in;
  end
  logic [NCOL:0][D-1:0][7:0]   xh0, xh1;                 // the rows' hop: column j's in j
  logic [NCOL-1:0][3:0][47:0]  acc;
  assign xh0[0] = x0;
  assign xh1[0] = x1;
  for (genvar j = 0; j < NCOL; j++) begin : g_col
    otpu_ppk_col #(.D(D), .COL(j), .BF(BF), .HOLD(HOLD)) u_col (
      .clk_s, .clk_f, .rst_s, .x0p(xh0[j]), .x1p(xh1[j]), .x0h(xh0[j + 1]), .x1h(xh1[j + 1]),
      .w(w[j]), .wf, .bm, .first, .acc(acc[j]));
  end
  always_ff @(posedge clk_s) acc_out <= acc;
endmodule


module ppk_top #(
  parameter int NCOL = 1,
  parameter int D = 128,
  parameter bit BF = 1'b1,
  parameter bit HOLD = 1'b0,
  parameter real CLKIN_NS = 3.333,
  parameter real MULT = 4.0,
  parameter real DIV_S = 8.0,
  parameter int DIV_F = 4
) (
  input  logic                              clk_in,     // 300 MHz (create_clock in ooc.tcl)
  input  logic                              rst,
  input  logic [D-1:0][7:0]                 x0_in,
  input  logic [D-1:0][7:0]                 x1_in,
  input  logic [NCOL-1:0][D-1:0][7:0]       w_in,
  input  logic [1:0]                        wf_in,
  input  logic [3:0][7:0][16:0]             bm_in,
  input  logic                              first_in,
  output logic [NCOL-1:0][3:0][47:0]        acc_out
);
  logic clkfb, clk_s_u, clk_f_u, clk_s, clk_f, locked;
  MMCME4_ADV #(
    .BANDWIDTH("OPTIMIZED"), .COMPENSATION("INTERNAL"), .DIVCLK_DIVIDE(1),
    .CLKFBOUT_MULT_F(MULT), .CLKFBOUT_PHASE(0.0), .CLKIN1_PERIOD(CLKIN_NS),
    .CLKOUT0_DIVIDE_F(DIV_S), .CLKOUT0_PHASE(0.0), .CLKOUT0_DUTY_CYCLE(0.5),
    .CLKOUT1_DIVIDE(DIV_F), .CLKOUT1_PHASE(0.0), .CLKOUT1_DUTY_CYCLE(0.5)
  ) u_mmcm (
    .CLKIN1(clk_in), .CLKIN2(1'b0), .CLKINSEL(1'b1), .CLKFBIN(clkfb), .CLKFBOUT(clkfb),
    .CLKFBOUTB(), .CLKOUT0(clk_s_u), .CLKOUT0B(), .CLKOUT1(clk_f_u), .CLKOUT1B(),
    .CLKOUT2(), .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6(),
    .LOCKED(locked), .PWRDWN(1'b0), .RST(1'b0), .CDDCREQ(1'b0), .CDDCDONE(),
    .DADDR(7'd0), .DCLK(1'b0), .DEN(1'b0), .DI(16'd0), .DO(), .DRDY(), .DWE(1'b0),
    .PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0), .PSDONE(), .CLKINSTOPPED(), .CLKFBSTOPPED());
  BUFGCE u_bs (.I(clk_s_u), .CE(1'b1), .O(clk_s));
  BUFGCE u_bf (.I(clk_f_u), .CE(1'b1), .O(clk_f));
  ppk_core #(.NCOL(NCOL), .D(D), .BF(BF), .HOLD(HOLD)) u_core (
    .clk_s, .clk_f, .rst(rst | ~locked), .x0_in, .x1_in, .w_in, .wf_in, .bm_in, .first_in,
    .acc_out);
endmodule
