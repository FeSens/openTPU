// FPGA top for the Inspur YPCB-00338 (xc7k480t-ffg1156-2): the block design otpu_bd (PCIe
// XDMA, two DDR3 MIGs, clocks, interconnect; boards/ypcb-00338/vivado/bd.tcl) and the
// accelerator otpu_board on its core-clock AXI interfaces. Port names follow the block design
// wrapper (<interface>_<signal>) so the MIG and XDMA constraints apply unchanged.
// AXI4 port connections P_<signal> from master bundle M and slave-response bundle S
`define OTPU_AXI(P, M, S) \
    .P``_awid(M.awid), .P``_awaddr(M.awaddr), .P``_awlen(M.awlen), .P``_awsize(M.awsize), \
    .P``_awburst(M.awburst), .P``_awlock(M.awlock), .P``_awcache(M.awcache), \
    .P``_awprot(M.awprot), .P``_awqos(M.awqos), .P``_awvalid(M.awvalid), \
    .P``_awready(S.awready), .P``_wdata(M.wdata), .P``_wstrb(M.wstrb), .P``_wlast(M.wlast), \
    .P``_wvalid(M.wvalid), .P``_wready(S.wready), .P``_bid(S.bid), .P``_bresp(S.bresp), \
    .P``_bvalid(S.bvalid), .P``_bready(M.bready), .P``_arid(M.arid), .P``_araddr(M.araddr), \
    .P``_arlen(M.arlen), .P``_arsize(M.arsize), .P``_arburst(M.arburst), \
    .P``_arlock(M.arlock), .P``_arcache(M.arcache), .P``_arprot(M.arprot), \
    .P``_arqos(M.arqos), .P``_arvalid(M.arvalid), .P``_arready(S.arready), \
    .P``_rid(S.rid), .P``_rdata(S.rdata), .P``_rresp(S.rresp), .P``_rlast(S.rlast), \
    .P``_rvalid(S.rvalid), .P``_rready(M.rready)

module otpu_fpga_top #(
  parameter int MCOLS = 2,                  // MXU columns (activation rows per weight chunk)
  parameter int VPU_CL = 2,                 // VPU lanes with the composite functions (exp2, ...)
  parameter int LANES = 8,                  // VPU lanes / TMEM banks (8 or 16)
  parameter int ULANES = 8,                 // TMEM lanes of the MXU and the quantizer
  parameter int CORE_KHZ = 100000,          // core_clk as the block design makes it (CORE_KHZ register)
  parameter logic [31:0] BUILD_ID = 32'h0   // the git commit (BUILD_ID register)
) (
  // board
  input  logic        SYS_CLK,              // 50 MHz, AA28
  output logic [2:0]  led,                  // P30 red, M30 green, N30 yellow
  // PCIe Gen2 x8
  input  logic        pcie_refclk_clk_p,    // J8 (MGTREFCLK)
  input  logic        pcie_refclk_clk_n,
  input  logic        pcie_perstn,          // Y26
  input  logic [7:0]  pcie_mgt_rxp,
  input  logic [7:0]  pcie_mgt_rxn,
  output logic [7:0]  pcie_mgt_txp,
  output logic [7:0]  pcie_mgt_txn,
  // DDR3 channel 0
  inout  wire  [71:0] DDR3_0_dq,
  inout  wire  [8:0]  DDR3_0_dqs_p,
  inout  wire  [8:0]  DDR3_0_dqs_n,
  output logic [14:0] DDR3_0_addr,
  output logic [2:0]  DDR3_0_ba,
  output logic        DDR3_0_ras_n,
  output logic        DDR3_0_cas_n,
  output logic        DDR3_0_we_n,
  output logic        DDR3_0_reset_n,
  output logic [0:0]  DDR3_0_ck_p,
  output logic [0:0]  DDR3_0_ck_n,
  output logic [0:0]  DDR3_0_cke,
  output logic [0:0]  DDR3_0_cs_n,
  output logic [0:0]  DDR3_0_odt,
  // DDR3 channel 1
  inout  wire  [71:0] DDR3_1_dq,
  inout  wire  [8:0]  DDR3_1_dqs_p,
  inout  wire  [8:0]  DDR3_1_dqs_n,
  output logic [14:0] DDR3_1_addr,
  output logic [2:0]  DDR3_1_ba,
  output logic        DDR3_1_ras_n,
  output logic        DDR3_1_cas_n,
  output logic        DDR3_1_we_n,
  output logic        DDR3_1_reset_n,
  output logic [0:0]  DDR3_1_ck_p,
  output logic [0:0]  DDR3_1_ck_n,
  output logic [0:0]  DDR3_1_cke,
  output logic [0:0]  DDR3_1_cs_n,
  output logic [0:0]  DDR3_1_odt
);
  logic        core_clk, core_rstn, pcie_link_up;
  logic [1:0]  calib;
  logic [11:0] device_temp;                  // XADC die-temperature code (xadc_temp in bd.tcl)

  // ---- control (AXI4-Lite, BD master -> accelerator)
  logic [31:0] ctl_awaddr, ctl_araddr, ctl_wdata, ctl_rdata;
  logic [3:0]  ctl_wstrb;
  logic [1:0]  ctl_bresp, ctl_rresp;
  logic ctl_awvalid, ctl_awready, ctl_wvalid, ctl_wready, ctl_bvalid, ctl_bready;
  logic ctl_arvalid, ctl_arready, ctl_rvalid, ctl_rready;

  // ---- memory masters (accelerator -> BD slave), one bundle per channel
  typedef struct packed {
    logic [0:0]   awid;   logic [31:0] awaddr; logic [7:0] awlen; logic [2:0] awsize;
    logic [1:0]   awburst; logic awlock; logic [3:0] awcache; logic [2:0] awprot;
    logic [3:0]   awqos;  logic awvalid;
    logic [511:0] wdata;  logic [63:0] wstrb; logic wlast; logic wvalid;
    logic         bready;
    logic [0:0]   arid;   logic [31:0] araddr; logic [7:0] arlen; logic [2:0] arsize;
    logic [1:0]   arburst; logic arlock; logic [3:0] arcache; logic [2:0] arprot;
    logic [3:0]   arqos;  logic arvalid;
    logic         rready;
  } m_t;
  typedef struct packed {
    logic awready; logic wready;
    logic [0:0] bid; logic [1:0] bresp; logic bvalid;
    logic arready;
    logic [0:0] rid; logic [511:0] rdata; logic [1:0] rresp; logic rlast; logic rvalid;
  } s_t;
  m_t m0, m1;
  s_t s0, s1;

  otpu_bd_wrapper u_bd (
    .sys_clk_50(SYS_CLK),
    .pcie_refclk_clk_p, .pcie_refclk_clk_n, .pcie_perstn,
    .pcie_mgt_rxp, .pcie_mgt_rxn, .pcie_mgt_txp, .pcie_mgt_txn,
    .pcie_link_up, .core_clk, .core_rstn, .calib, .device_temp,
    .DDR3_0_dq, .DDR3_0_dqs_p, .DDR3_0_dqs_n, .DDR3_0_addr, .DDR3_0_ba, .DDR3_0_ras_n,
    .DDR3_0_cas_n, .DDR3_0_we_n, .DDR3_0_reset_n, .DDR3_0_ck_p, .DDR3_0_ck_n, .DDR3_0_cke,
    .DDR3_0_cs_n, .DDR3_0_odt,
    .DDR3_1_dq, .DDR3_1_dqs_p, .DDR3_1_dqs_n, .DDR3_1_addr, .DDR3_1_ba, .DDR3_1_ras_n,
    .DDR3_1_cas_n, .DDR3_1_we_n, .DDR3_1_reset_n, .DDR3_1_ck_p, .DDR3_1_ck_n, .DDR3_1_cke,
    .DDR3_1_cs_n, .DDR3_1_odt,
    .M_AXI_CTL_awaddr(ctl_awaddr), .M_AXI_CTL_awvalid(ctl_awvalid),
    .M_AXI_CTL_awready(ctl_awready), .M_AXI_CTL_wdata(ctl_wdata), .M_AXI_CTL_wstrb(ctl_wstrb),
    .M_AXI_CTL_wvalid(ctl_wvalid), .M_AXI_CTL_wready(ctl_wready),
    .M_AXI_CTL_bresp(ctl_bresp), .M_AXI_CTL_bvalid(ctl_bvalid), .M_AXI_CTL_bready(ctl_bready),
    .M_AXI_CTL_araddr(ctl_araddr), .M_AXI_CTL_arvalid(ctl_arvalid),
    .M_AXI_CTL_arready(ctl_arready), .M_AXI_CTL_rdata(ctl_rdata), .M_AXI_CTL_rresp(ctl_rresp),
    .M_AXI_CTL_rvalid(ctl_rvalid), .M_AXI_CTL_rready(ctl_rready),
    `OTPU_AXI(S_AXI_M0, m0, s0),
    `OTPU_AXI(S_AXI_M1, m1, s1)
  );

  // core reset: synchronous, active high
  logic core_rst;
  always_ff @(posedge core_clk) core_rst <= !core_rstn;

  logic [2:0] board_led;
  otpu_board #(.MCOLS(MCOLS), .VPU_CL(VPU_CL), .LANES(LANES), .ULANES(ULANES), .CORE_KHZ(CORE_KHZ), .BUILD_ID(BUILD_ID)) u_board (
    .clk(core_clk), .rst(core_rst), .calib, .temp(device_temp), .led(board_led),
    .s_ctl_awaddr(ctl_awaddr[11:0]), .s_ctl_awvalid(ctl_awvalid), .s_ctl_awready(ctl_awready),
    .s_ctl_wdata(ctl_wdata), .s_ctl_wstrb(ctl_wstrb), .s_ctl_wvalid(ctl_wvalid),
    .s_ctl_wready(ctl_wready), .s_ctl_bresp(ctl_bresp), .s_ctl_bvalid(ctl_bvalid),
    .s_ctl_bready(ctl_bready), .s_ctl_araddr(ctl_araddr[11:0]), .s_ctl_arvalid(ctl_arvalid),
    .s_ctl_arready(ctl_arready), .s_ctl_rdata(ctl_rdata), .s_ctl_rresp(ctl_rresp),
    .s_ctl_rvalid(ctl_rvalid), .s_ctl_rready(ctl_rready),
    `OTPU_AXI(m0_axi, m0, s0),
    `OTPU_AXI(m1_axi, m1, s1)
  );

  // LEDs (polarity unverified on this card): led[0] red = heartbeat, led[1] green = PCIe link
  // up and both DDR3 channels calibrated, led[2] yellow = the accelerator runs or halted cleanly
  assign led = {board_led[2] | board_led[1], pcie_link_up & (&calib), board_led[0]};
endmodule

`undef OTPU_AXI
