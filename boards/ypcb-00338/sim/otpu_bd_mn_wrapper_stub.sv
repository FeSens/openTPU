// Lint-only stand-in for the Vivado-generated block design wrapper of the native MIG build
// (boards/ypcb-00338/vivado/bd_native.tcl, MEM=mig_native): the ports Vivado will generate, no
// behaviour. Used by `make lint-mn` to check otpu_fpga_top_mn offline; Vivado builds with the
// real otpu_bd_wrapper.v (and the unisim IOBUF). The DDR3 ports as bd.tcl's
// (otpu_bd_wrapper_stub.sv), each MIG's native interface as mig<c>_<pin>, M_AXI_DMA as XDMA's
// M_AXI.
module otpu_bd_wrapper (
  input  logic        sys_clk_50,
  input  logic        pcie_refclk_clk_p, pcie_refclk_clk_n, pcie_perstn,
  input  logic [7:0]  pcie_mgt_rxp, pcie_mgt_rxn,
  output logic [7:0]  pcie_mgt_txp, pcie_mgt_txn,
  output logic        pcie_link_up, core_clk, core_rstn, xdma_aclk, xdma_aresetn,
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
`define STUB_MIG(P) \
  output logic         P``_ui_clk, P``_ui_clk_sync_rst, \
  input  logic [28:0]  P``_app_addr, \
  input  logic [2:0]   P``_app_cmd, \
  input  logic         P``_app_en, P``_app_wdf_wren, P``_app_wdf_end, \
  input  logic [511:0] P``_app_wdf_data, \
  input  logic [63:0]  P``_app_wdf_mask, \
  output logic         P``_app_rdy, P``_app_wdf_rdy, P``_app_rd_data_valid, \
  output logic [511:0] P``_app_rd_data
`define STUB_L(P) \
  output logic [31:0] P``_awaddr, P``_araddr, P``_wdata, \
  output logic [3:0]  P``_wstrb, \
  output logic        P``_awvalid, P``_wvalid, P``_bready, P``_arvalid, P``_rready, \
  input  logic        P``_awready, P``_wready, P``_bvalid, P``_arready, P``_rvalid, \
  input  logic [1:0]  P``_bresp, P``_rresp, \
  input  logic [31:0] P``_rdata
  `STUB_L(M_AXI_CTL),
  `STUB_MIG(mig0),
  `STUB_MIG(mig1),
  output logic [3:0]   M_AXI_DMA_awid, M_AXI_DMA_arid,
  output logic [63:0]  M_AXI_DMA_awaddr, M_AXI_DMA_araddr,
  output logic [7:0]   M_AXI_DMA_awlen, M_AXI_DMA_arlen,
  output logic [2:0]   M_AXI_DMA_awsize, M_AXI_DMA_arsize, M_AXI_DMA_awprot, M_AXI_DMA_arprot,
  output logic [1:0]   M_AXI_DMA_awburst, M_AXI_DMA_arburst,
  output logic         M_AXI_DMA_awlock, M_AXI_DMA_arlock,
  output logic [3:0]   M_AXI_DMA_awcache, M_AXI_DMA_arcache,
  output logic         M_AXI_DMA_awvalid, M_AXI_DMA_arvalid, M_AXI_DMA_wvalid, M_AXI_DMA_wlast,
  output logic         M_AXI_DMA_bready, M_AXI_DMA_rready,
  output logic [127:0] M_AXI_DMA_wdata,
  output logic [15:0]  M_AXI_DMA_wstrb,
  input  logic         M_AXI_DMA_awready, M_AXI_DMA_arready, M_AXI_DMA_wready,
  input  logic         M_AXI_DMA_bvalid, M_AXI_DMA_rvalid, M_AXI_DMA_rlast,
  input  logic [3:0]   M_AXI_DMA_bid, M_AXI_DMA_rid,
  input  logic [1:0]   M_AXI_DMA_bresp, M_AXI_DMA_rresp,
  input  logic [127:0] M_AXI_DMA_rdata
);
`undef STUB_L
`undef STUB_MIG
endmodule

// The Xilinx IOBUF primitive (unisim) as otpu_fpga_top_mn uses it: T = 1 floats the pad.
module IOBUF (inout wire IO, input logic I, input logic T, output logic O);
  assign IO = T ? 1'bz : I;
  assign O = IO;
endmodule
