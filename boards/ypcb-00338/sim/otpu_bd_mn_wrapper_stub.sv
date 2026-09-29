// Lint-only stand-ins for the native MIG build (MEM=mig_native): the block design wrapper
// (boards/ypcb-00338/vivado/bd_native.tcl) and the two MIG IP modules create_project.tcl makes
// (mig_ddr3_ch0 / 1: native interface, ECC, no DM pins, XADC and clock buffers outside), with the
// ports Vivado will generate and no behaviour. Used by `make lint-mn` to check otpu_fpga_top_mn
// offline; Vivado builds with the real ones (and the unisim IOBUF). M_AXI_DMA is XDMA's M_AXI.
module otpu_bd_wrapper (
  input  logic        sys_clk_50,
  input  logic        pcie_refclk_clk_p, pcie_refclk_clk_n, pcie_perstn,
  input  logic [7:0]  pcie_mgt_rxp, pcie_mgt_rxn,
  output logic [7:0]  pcie_mgt_txp, pcie_mgt_txn,
  output logic        pcie_link_up, core_clk, core_rstn, xdma_aclk, xdma_aresetn,
  output logic [11:0] device_temp,
  output logic        mig_sys_clk, mig_ref_clk, mig_locked,
`define STUB_L(P) \
  output logic [31:0] P``_awaddr, P``_araddr, P``_wdata, \
  output logic [3:0]  P``_wstrb, \
  output logic        P``_awvalid, P``_wvalid, P``_bready, P``_arvalid, P``_rready, \
  input  logic        P``_awready, P``_wready, P``_bvalid, P``_arready, P``_rvalid, \
  input  logic [1:0]  P``_bresp, P``_rresp, \
  input  logic [31:0] P``_rdata
  `STUB_L(M_AXI_CTL),
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
endmodule

// The MIG 7-series native top as mig_ddr3_ch<c> (the .prj's ModuleName)
`define STUB_MIG(M) \
module M ( \
  inout  wire  [71:0]  ddr3_dq, \
  inout  wire  [8:0]   ddr3_dqs_p, ddr3_dqs_n, \
  output logic [14:0]  ddr3_addr, \
  output logic [2:0]   ddr3_ba, \
  output logic         ddr3_ras_n, ddr3_cas_n, ddr3_we_n, ddr3_reset_n, \
  output logic [0:0]   ddr3_ck_p, ddr3_ck_n, ddr3_cke, ddr3_cs_n, ddr3_odt, \
  input  logic [28:0]  app_addr, \
  input  logic [2:0]   app_cmd, \
  input  logic         app_en, app_wdf_end, app_wdf_wren, \
  input  logic [511:0] app_wdf_data, \
  input  logic [63:0]  app_wdf_mask, \
  output logic [511:0] app_rd_data, \
  output logic         app_rd_data_end, app_rd_data_valid, app_rdy, app_wdf_rdy, \
  input  logic         app_sr_req, app_ref_req, app_zq_req, \
  output logic         app_sr_active, app_ref_ack, app_zq_ack, \
  output logic [7:0]   app_ecc_multiple_err, \
  output logic         ui_clk, ui_clk_sync_rst, init_calib_complete, \
  input  logic         sys_clk_i, clk_ref_i, \
  input  logic [11:0]  device_temp_i, \
  input  logic         sys_rst); \
endmodule
`STUB_MIG(mig_ddr3_ch0)
`STUB_MIG(mig_ddr3_ch1)
`undef STUB_MIG

// The Xilinx IOBUF primitive (unisim) as otpu_fpga_top_mn uses it: T = 1 floats the pad.
module IOBUF (inout wire IO, input logic I, input logic T, output logic O);
  assign IO = T ? 1'bz : I;
  assign O = IO;
endmodule
