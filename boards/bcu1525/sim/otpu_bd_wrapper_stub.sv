// Lint-only stand-in for the Vivado-generated wrapper of the BCU1525's block design
// (boards/bcu1525/vivado/bd.tcl): the ports Vivado will generate, no behaviour. `make lint` checks
// otpu_fpga_top_bcu offline against it; Vivado builds with the real otpu_bd_wrapper.v. The DMA
// masters are sc_dma's, made external as declared in bd.tcl: AXI4, 33-bit address, 128-bit data,
// no IDs and no burst / lock / cache / prot / qos / region signals (the top leaves their size
// signals, full-width beats, unconnected: not listed here).
module otpu_bd_wrapper #(parameter int PCIE_LANES = 16) (
  input  logic        clk300_in,
  input  logic        pcie_refclk_clk_p, pcie_refclk_clk_n, pcie_perstn,
  input  logic [PCIE_LANES-1:0] pcie_mgt_rxp, pcie_mgt_rxn,
  output logic [PCIE_LANES-1:0] pcie_mgt_txp, pcie_mgt_txn,
  output logic        pcie_link_up, core_clk, core_clk2x, core_rstn, xdma_aclk, xdma_aresetn,
  output logic        acc_clk, temp_alarm, ot_alarm,
  input  logic        acc_clk_ce,
  output logic [9:0]  device_temp,
`define STUB_L(P) \
  output logic [31:0] P``_awaddr, P``_araddr, P``_wdata, \
  output logic [3:0]  P``_wstrb, \
  output logic        P``_awvalid, P``_wvalid, P``_bready, P``_arvalid, P``_rready, \
  input  logic        P``_awready, P``_wready, P``_bvalid, P``_arready, P``_rvalid, \
  input  logic [1:0]  P``_bresp, P``_rresp, \
  input  logic [31:0] P``_rdata
`define STUB_D(P) \
  output logic [32:0]  P``_awaddr, P``_araddr, \
  output logic [7:0]   P``_awlen, P``_arlen, \
  output logic         P``_awvalid, P``_arvalid, P``_wvalid, P``_wlast, P``_bready, P``_rready, \
  output logic [127:0] P``_wdata, \
  output logic [15:0]  P``_wstrb, \
  input  logic         P``_awready, P``_arready, P``_wready, P``_bvalid, P``_rvalid, P``_rlast, \
  input  logic [1:0]   P``_bresp, P``_rresp, \
  input  logic [127:0] P``_rdata
  `STUB_L(M_AXI_CTL),
  `STUB_L(M_AXI_MEMCAL0),
  `STUB_L(M_AXI_MEMCAL1),
  `STUB_L(M_AXI_MEMCAL2),
  `STUB_L(M_AXI_MEMCAL3),
  `STUB_L(M_AXI_BMC),
  `STUB_L(M_AXI_GOV),
  `STUB_D(M_AXI_DMA0),
  `STUB_D(M_AXI_DMA1),
  `STUB_D(M_AXI_DMA2),
  `STUB_D(M_AXI_DMA3)
);
`undef STUB_L
`undef STUB_D
endmodule
