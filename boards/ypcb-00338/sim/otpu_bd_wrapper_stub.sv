// Lint-only stand-in for the Vivado-generated block design wrapper (boards/ypcb-00338/vivado/
// bd.tcl): the ports Vivado will generate, no behaviour. Used by `make lint` to check
// otpu_fpga_top offline; Vivado builds with the real otpu_bd_wrapper.v.
module otpu_bd_wrapper (
  input  logic        sys_clk_50,
  input  logic        pcie_refclk_clk_p, pcie_refclk_clk_n, pcie_perstn,
  input  logic [7:0]  pcie_mgt_rxp, pcie_mgt_rxn,
  output logic [7:0]  pcie_mgt_txp, pcie_mgt_txn,
  output logic        pcie_link_up, core_clk, core_rstn,
  output logic [1:0]  calib,
  output logic [11:0] device_temp,
  inout  wire  [71:0] DDR3_0_dq, DDR3_1_dq,
  inout  wire  [8:0]  DDR3_0_dqs_p, DDR3_0_dqs_n, DDR3_1_dqs_p, DDR3_1_dqs_n,
  output logic [14:0] DDR3_0_addr, DDR3_1_addr,
  output logic [2:0]  DDR3_0_ba, DDR3_1_ba,
  output logic        DDR3_0_ras_n, DDR3_0_cas_n, DDR3_0_we_n, DDR3_0_reset_n,
  output logic        DDR3_1_ras_n, DDR3_1_cas_n, DDR3_1_we_n, DDR3_1_reset_n,
  output logic [0:0]  DDR3_0_ck_p, DDR3_0_ck_n, DDR3_0_cke, DDR3_0_cs_n, DDR3_0_odt,
  output logic [0:0]  DDR3_1_ck_p, DDR3_1_ck_n, DDR3_1_cke, DDR3_1_cs_n, DDR3_1_odt,
  output logic [31:0] M_AXI_CTL_awaddr, M_AXI_CTL_araddr, M_AXI_CTL_wdata,
  output logic [3:0]  M_AXI_CTL_wstrb,
  output logic        M_AXI_CTL_awvalid, M_AXI_CTL_wvalid, M_AXI_CTL_bready,
  output logic        M_AXI_CTL_arvalid, M_AXI_CTL_rready,
  input  logic        M_AXI_CTL_awready, M_AXI_CTL_wready, M_AXI_CTL_bvalid,
  input  logic        M_AXI_CTL_arready, M_AXI_CTL_rvalid,
  input  logic [1:0]  M_AXI_CTL_bresp, M_AXI_CTL_rresp,
  input  logic [31:0] M_AXI_CTL_rdata,
`define STUB_S(P) \
  input  logic [0:0] P``_awid, P``_arid, \
  input  logic [31:0] P``_awaddr, P``_araddr, \
  input  logic [7:0] P``_awlen, P``_arlen, \
  input  logic [2:0] P``_awsize, P``_arsize, P``_awprot, P``_arprot, \
  input  logic [1:0] P``_awburst, P``_arburst, \
  input  logic P``_awlock, P``_arlock, \
  input  logic [3:0] P``_awcache, P``_arcache, P``_awqos, P``_arqos, \
  input  logic P``_awvalid, P``_arvalid, P``_wvalid, P``_wlast, P``_bready, P``_rready, \
  input  logic [511:0] P``_wdata, \
  input  logic [63:0] P``_wstrb, \
  output logic P``_awready, P``_arready, P``_wready, P``_bvalid, P``_rvalid, P``_rlast, \
  output logic [0:0] P``_bid, P``_rid, \
  output logic [1:0] P``_bresp, P``_rresp, \
  output logic [511:0] P``_rdata
  `STUB_S(S_AXI_M0),
  `STUB_S(S_AXI_M1)
);
`undef STUB_S
endmodule
