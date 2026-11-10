// Lint-only stand-in for the Vivado-generated wrapper of the BCU1525 WG build's block design
// (boards/bcu1525/vivado/bd_wg.tcl): the ports as bd_wg.tcl declares them, no behaviour.
// tests/test_bcuwg_build.py lints otpu_fpga_top_bcu_wg against it; Vivado builds with the real
// otpu_bdw_wrapper.v. The DMA masters (M_AXI_DMA<d>, sc_dma's; M_AXI_H<d>, sc_h's: 512 bits at
// aclk) have no IDs and no burst / lock / cache / prot / qos / region signals (the top leaves their
// size signals unconnected: not listed); the slaves S_AXI_H<d> (no IDs) and S_AXI_P<d> (4 ID bits)
// take size and burst; addresses: M_AXI_DMA's 64 bits, M_AXI_H's and the slaves' 33 (the MIGs'
// AxiAddressWidth, an 8 GB DIMM); each DIMM's MIG interface is a single-rank x8 UDIMM's (one
// clock pair, chip select, CKE and ODT).
// acc_clk2x is the clock wizard's output unbuffered (the top makes acc_clk2x and acc_clk from it).
module otpu_bdw_wrapper #(parameter int PCIE_LANES = 16, parameter int AXB = 512) (
  input  logic        pcie_refclk_clk_p, pcie_refclk_clk_n, pcie_perstn,
  input  logic [PCIE_LANES-1:0] pcie_mgt_rxp, pcie_mgt_rxn,
  output logic [PCIE_LANES-1:0] pcie_mgt_txp, pcie_mgt_txn,
  output logic        pcie_link_up, core_clk, core_rstn, acc_clk2x, aclk, xdma_aclk,
  output logic        xdma_aresetn, temp_alarm, ot_alarm,
`ifdef OTPU_WG_X2
  output logic        init_clk,
`endif
  output logic [9:0]  device_temp,
`define STUB_L(P) \
  output logic [31:0] P``_awaddr, P``_araddr, P``_wdata, \
  output logic [3:0]  P``_wstrb, \
  output logic        P``_awvalid, P``_wvalid, P``_bready, P``_arvalid, P``_rready, \
  input  logic        P``_awready, P``_wready, P``_bvalid, P``_arready, P``_rvalid, \
  input  logic [1:0]  P``_bresp, P``_rresp, \
  input  logic [31:0] P``_rdata
`define STUB_S(P, DW, IW) \
  input  logic [32:0]     P``_awaddr, P``_araddr, \
  input  logic [7:0]      P``_awlen, P``_arlen, \
  input  logic [2:0]      P``_awsize, P``_arsize, \
  input  logic [1:0]      P``_awburst, P``_arburst, \
  input  logic            P``_awvalid, P``_arvalid, P``_wvalid, P``_wlast, P``_bready, P``_rready, \
  input  logic [DW-1:0]   P``_wdata, \
  input  logic [DW/8-1:0] P``_wstrb, \
  output logic            P``_awready, P``_arready, P``_wready, P``_bvalid, P``_rvalid, P``_rlast, \
  output logic [1:0]      P``_bresp, P``_rresp, \
  output logic [DW-1:0]   P``_rdata
`define STUB_D(D) \
  input  logic        c300_``D``_clk_p, c300_``D``_clk_n, \
  output logic [16:0] ddr4_c``D``_adr, \
  output logic [1:0]  ddr4_c``D``_ba, ddr4_c``D``_bg, \
  output logic        ddr4_c``D``_act_n, ddr4_c``D``_reset_n, \
  output logic [0:0]  ddr4_c``D``_ck_t, ddr4_c``D``_ck_c, ddr4_c``D``_cke, ddr4_c``D``_cs_n, \
  output logic [0:0]  ddr4_c``D``_odt, \
  inout  wire  [7:0]  ddr4_c``D``_dm_n, ddr4_c``D``_dqs_t, ddr4_c``D``_dqs_c, \
  inout  wire  [63:0] ddr4_c``D``_dq, \
  input  logic        mig_rst``D, sc_rstn``D, \
  output logic        calib``D, \
  output logic [63:0]  M_AXI_DMA``D``_awaddr, M_AXI_DMA``D``_araddr, \
  output logic [7:0]   M_AXI_DMA``D``_awlen, M_AXI_DMA``D``_arlen, \
  output logic         M_AXI_DMA``D``_awvalid, M_AXI_DMA``D``_arvalid, M_AXI_DMA``D``_wvalid, \
  output logic         M_AXI_DMA``D``_wlast, M_AXI_DMA``D``_bready, M_AXI_DMA``D``_rready, \
  output logic [AXB-1:0]   M_AXI_DMA``D``_wdata, \
  output logic [AXB/8-1:0] M_AXI_DMA``D``_wstrb, \
  input  logic         M_AXI_DMA``D``_awready, M_AXI_DMA``D``_arready, M_AXI_DMA``D``_wready, \
  input  logic         M_AXI_DMA``D``_bvalid, M_AXI_DMA``D``_rvalid, M_AXI_DMA``D``_rlast, \
  input  logic [1:0]   M_AXI_DMA``D``_bresp, M_AXI_DMA``D``_rresp, \
  input  logic [AXB-1:0]   M_AXI_DMA``D``_rdata, \
  output logic [32:0]  M_AXI_H``D``_awaddr, M_AXI_H``D``_araddr, \
  output logic [7:0]   M_AXI_H``D``_awlen, M_AXI_H``D``_arlen, \
  output logic         M_AXI_H``D``_awvalid, M_AXI_H``D``_arvalid, M_AXI_H``D``_wvalid, \
  output logic         M_AXI_H``D``_wlast, M_AXI_H``D``_bready, M_AXI_H``D``_rready, \
  output logic [511:0] M_AXI_H``D``_wdata, \
  output logic [63:0]  M_AXI_H``D``_wstrb, \
  input  logic         M_AXI_H``D``_awready, M_AXI_H``D``_arready, M_AXI_H``D``_wready, \
  input  logic         M_AXI_H``D``_bvalid, M_AXI_H``D``_rvalid, M_AXI_H``D``_rlast, \
  input  logic [1:0]   M_AXI_H``D``_bresp, M_AXI_H``D``_rresp, \
  input  logic [511:0] M_AXI_H``D``_rdata, \
  `STUB_S(S_AXI_H``D, AXB, 0), \
  `STUB_S(S_AXI_P``D, 512, 4), \
  input  logic [3:0]  S_AXI_P``D``_awid, S_AXI_P``D``_arid, \
  output logic [3:0]  S_AXI_P``D``_bid, S_AXI_P``D``_rid
  `STUB_L(M_AXI_CTL),
  `STUB_L(M_AXI_BMC),
  `STUB_L(M_AXI_GOV),
  `STUB_D(0),
  `STUB_D(1),
  `STUB_D(2),
  `STUB_D(3)
);
`undef STUB_L
`undef STUB_S
`undef STUB_D
endmodule
