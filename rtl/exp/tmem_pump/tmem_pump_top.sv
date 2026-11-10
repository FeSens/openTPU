// EXPERIMENTAL (docs/tmem_pump.md): the out-of-context top of the TMEM experiment. The slice's
// TMEM ports (otpu_slice: 8 read ports, the MXU's with the crossbar, Q3 and COLL guests on the
// DMA's copy at IMPL 0; 4 write ports), every input registered in clk_s and every read lane's data
// registered in clk_s (the ports carry no timing), around one of:
//   VAR 0  otpu_tmem IMPL 0: a copy per read port (6 copies, the card's)
//   VAR 1  otpu_tmem IMPL 1: banked, NCP 3 copies of 1R1W RAMs (the ASIC's)
//   VAR 2  otpu_tmem_pump IMPL 0: 3 pumped copies (6 slots: the card's behaviour)
//   VAR 3  otpu_tmem_pump IMPL 1, NCP 4: 2 pumped copies (4 banked slots)
//   VAR 4  otpu_tmem_pump IMPL 1, NCP 2: 1 pumped copy (2 banked slots)
//   VAR 5  otpu_tmem_pump IMPL 1, K = PK (3 or 4): 1 copy pumped at PK x, 2 PK - 1 banked slots
//          (requests registered: reads a cycle later)
// The clocks come from one MMCM (FAMILY 0: MMCME2_ADV, 7 series; 1: MMCME4_ADV, UltraScale+):
// VCO = the input clock x MULT, clk_s = VCO / DIV_S, clk_f = VCO / DIV_F, both through global
// buffers, so Vivado times the real skew between them.
module tmem_pump_top #(
  parameter int VAR = 2,
  parameter int WORDS = 1 << 16,
  parameter int FAMILY = 0,
  parameter string STYLE = "block",
  parameter real CLKIN_NS = 7.5,
  parameter real MULT = 8.0,
  parameter real DIV_S = 8.0,
  parameter int DIV_F = 4,
  parameter int PK = 3,
  parameter int CASC = 0
) (
  input  logic                          clk_in,
  input  logic [7:0][7:0]               r_en_in,
  input  logic [7:0][7:0]               r_req_in,
  input  logic [7:0][7:0][31:0]         r_addr_in,
  input  logic [3:0][7:0]               w_en_in,
  input  logic [3:0][7:0]               w_req_in,
  input  logic [3:0]                    w_gnt_in,
  input  logic [3:0][7:0][31:0]         w_addr_in,
  input  logic [3:0][7:0][31:0]         w_data_in,
  output logic [7:0][7:0][31:0]         r_data_out
);
  localparam int LANES = 8, NRP = 8, NWP = 4;
  localparam int P_MXU = 0, P_DMA = 1, P_Q = 2, P_VA = 3, P_VB = 4, P_Q3 = 5, P_Q2 = 6, P_COLL = 7;
  localparam int W_MXU = 1;
  localparam logic [NRP-1:0] GEN_R = NRP'(1) << P_MXU;
  localparam logic [NWP-1:0] GEN_W = NWP'(1) << W_MXU;
  localparam logic [NRP-1:0] SH_MASK = (NRP'(1) << P_Q3) | (NRP'(1) << P_COLL);
  //                                       COLL  Q2    Q3    VB    VA    Q     DMA   MXU
  localparam logic [NRP-1:0][2:0] CPY3 = {3'd2, 3'd1, 3'd2, 3'd1, 3'd0, 3'd0, 3'd2, 3'd0};
  localparam logic [NRP-1:0][2:0] CPY4 = {3'd3, 3'd2, 3'd1, 3'd1, 3'd0, 3'd3, 3'd2, 3'd0};
  localparam logic [NRP-1:0][2:0] CPYB = {3'd1, 3'd1, 3'd1, 3'd1, 3'd0, 3'd0, 3'd1, 3'd0};
  localparam logic [NRP-1:0][2:0] CPY5 = {3'd1, 3'd4, 3'd1, 3'd4, 3'd3, 3'd2, 3'd1, 3'd0};
  localparam logic [NRP-1:0][2:0] CPY7 = {3'd1, 3'd6, 3'd5, 3'd4, 3'd3, 3'd2, 3'd1, 3'd0};

  logic clkfb, clk_s_u, clk_f_u, clk_s, clk_f;
  if (FAMILY == 0) begin : g_k7
    MMCME2_ADV #(
      .BANDWIDTH("OPTIMIZED"), .COMPENSATION("INTERNAL"), .DIVCLK_DIVIDE(1),
      .CLKFBOUT_MULT_F(MULT), .CLKFBOUT_PHASE(0.0), .CLKIN1_PERIOD(CLKIN_NS),
      .CLKOUT0_DIVIDE_F(DIV_S), .CLKOUT0_PHASE(0.0), .CLKOUT0_DUTY_CYCLE(0.5),
      .CLKOUT1_DIVIDE(DIV_F), .CLKOUT1_PHASE(0.0), .CLKOUT1_DUTY_CYCLE(0.5)
    ) u_mmcm (
      .CLKIN1(clk_in), .CLKIN2(1'b0), .CLKINSEL(1'b1), .CLKFBIN(clkfb), .CLKFBOUT(clkfb),
      .CLKFBOUTB(), .CLKOUT0(clk_s_u), .CLKOUT0B(), .CLKOUT1(clk_f_u), .CLKOUT1B(),
      .CLKOUT2(), .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6(),
      .LOCKED(), .PWRDWN(1'b0), .RST(1'b0),
      .DADDR(7'd0), .DCLK(1'b0), .DEN(1'b0), .DI(16'd0), .DO(), .DRDY(), .DWE(1'b0),
      .PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0), .PSDONE(), .CLKINSTOPPED(), .CLKFBSTOPPED());
    BUFG u_bs (.I(clk_s_u), .O(clk_s));
    BUFG u_bf (.I(clk_f_u), .O(clk_f));
  end else begin : g_us
    MMCME4_ADV #(
      .BANDWIDTH("OPTIMIZED"), .COMPENSATION("INTERNAL"), .DIVCLK_DIVIDE(1),
      .CLKFBOUT_MULT_F(MULT), .CLKFBOUT_PHASE(0.0), .CLKIN1_PERIOD(CLKIN_NS),
      .CLKOUT0_DIVIDE_F(DIV_S), .CLKOUT0_PHASE(0.0), .CLKOUT0_DUTY_CYCLE(0.5),
      .CLKOUT1_DIVIDE(DIV_F), .CLKOUT1_PHASE(0.0), .CLKOUT1_DUTY_CYCLE(0.5)
    ) u_mmcm (
      .CLKIN1(clk_in), .CLKIN2(1'b0), .CLKINSEL(1'b1), .CLKFBIN(clkfb), .CLKFBOUT(clkfb),
      .CLKFBOUTB(), .CLKOUT0(clk_s_u), .CLKOUT0B(), .CLKOUT1(clk_f_u), .CLKOUT1B(),
      .CLKOUT2(), .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6(),
      .LOCKED(), .PWRDWN(1'b0), .RST(1'b0), .CDDCREQ(1'b0), .CDDCDONE(),
      .DADDR(7'd0), .DCLK(1'b0), .DEN(1'b0), .DI(16'd0), .DO(), .DRDY(), .DWE(1'b0),
      .PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0), .PSDONE(), .CLKINSTOPPED(), .CLKFBSTOPPED());
    BUFGCE u_bs (.I(clk_s_u), .CE(1'b1), .O(clk_s));
    BUFGCE u_bf (.I(clk_f_u), .CE(1'b1), .O(clk_f));
  end

  logic [NRP-1:0][LANES-1:0]       r_en, r_req;
  logic [NRP-1:0][LANES-1:0][31:0] r_addr, r_data;
  logic [NWP-1:0][LANES-1:0]       w_en, w_req;
  logic [NWP-1:0]                  w_gnt;
  logic [NWP-1:0][LANES-1:0][31:0] w_addr, w_data;
  always_ff @(posedge clk_s) begin
    r_en <= r_en_in; r_req <= r_req_in; r_addr <= r_addr_in;
    w_en <= w_en_in; w_req <= w_req_in; w_gnt <= w_gnt_in; w_addr <= w_addr_in;
    w_data <= w_data_in;
    r_data_out <= r_data;
  end

  if (VAR == 0 || VAR == 1) begin : g_ref
    otpu_tmem #(.WORDS(WORDS), .LANES(LANES), .NRP(NRP), .NWP(NWP), .WPB(1), .GEN_R(GEN_R),
                .GEN_W(GEN_W), .SH_HOST(P_DMA), .SH_MASK(VAR == 0 ? SH_MASK : '0), .SID(0),
                .IMPL(VAR), .NCP(3), .CPY(CPY3)) u_tmem (
      .clk(clk_s), .clk_f(1'b0), .rst(1'b0), .r_en, .r_req, .r_addr, .r_ben('0), .r_bad('0), .r_data, .w_en, .w_req, .w_gnt, .w_addr,
      .w_data, .dump(1'b0));
  end else if (VAR == 5) begin : g_pumpk
    otpu_tmem_pump #(.WORDS(WORDS), .LANES(LANES), .NRP(NRP), .NWP(NWP), .GEN_R(GEN_R),
                     .GEN_W(GEN_W), .SH_HOST(P_DMA), .SH_MASK('0), .IMPL(1), .NCP(2 * PK - 1),
                     .CPY(PK == 3 ? CPY5 : CPY7), .STYLE(STYLE), .K(PK), .CASC(CASC)) u_tmem (
      .clk(clk_s), .clk_f, .r_en, .r_req, .r_addr, .r_ben('0), .r_bad('0), .r_data, .w_en, .w_req, .w_gnt, .w_addr,
      .w_data);
  end else begin : g_pump
    otpu_tmem_pump #(.WORDS(WORDS), .LANES(LANES), .NRP(NRP), .NWP(NWP), .GEN_R(GEN_R),
                     .GEN_W(GEN_W), .SH_HOST(P_DMA), .SH_MASK(VAR == 2 ? SH_MASK : '0),
                     .IMPL(VAR == 2 ? 0 : 1), .NCP(VAR == 3 ? 4 : 2),
                     .CPY(VAR == 3 ? CPY4 : CPYB), .STYLE(STYLE)) u_tmem (
      .clk(clk_s), .clk_f, .r_en, .r_req, .r_addr, .r_ben('0), .r_bad('0), .r_data, .w_en, .w_req, .w_gnt, .w_addr,
      .w_data);
  end
endmodule
