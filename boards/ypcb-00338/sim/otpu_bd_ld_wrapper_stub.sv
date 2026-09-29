// Lint-only stand-in for the Vivado-generated block design wrapper of the LiteDRAM build
// (boards/ypcb-00338/vivado/bd_native.tcl, MEM=litedram): the ports Vivado will generate, no
// behaviour. Used by `make lint-ld` to check otpu_fpga_top_ld offline; Vivado builds with the
// real otpu_bd_wrapper.v (and the unisim IBUF, BUFG, IOBUF). M_AXI_DMA is XDMA's M_AXI made
// external (64-bit address, 4-bit IDs, 128-bit data).
module otpu_bd_wrapper (
  input  logic        sys_clk_50,
  input  logic        pcie_refclk_clk_p, pcie_refclk_clk_n, pcie_perstn,
  input  logic [7:0]  pcie_mgt_rxp, pcie_mgt_rxn,
  output logic [7:0]  pcie_mgt_txp, pcie_mgt_txn,
  output logic        pcie_link_up, core_clk, core_rstn, xdma_aclk, xdma_aresetn,
  output logic [11:0] device_temp,
`define STUB_L(P) \
  output logic [31:0] P``_awaddr, P``_araddr, P``_wdata, \
  output logic [3:0]  P``_wstrb, \
  output logic        P``_awvalid, P``_wvalid, P``_bready, P``_arvalid, P``_rready, \
  input  logic        P``_awready, P``_wready, P``_bvalid, P``_arready, P``_rvalid, \
  input  logic [1:0]  P``_bresp, P``_rresp, \
  input  logic [31:0] P``_rdata
  `STUB_L(M_AXI_CTL),
  `STUB_L(M_AXI_MEMCAL),
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

// The Xilinx primitives (unisim) as otpu_fpga_top_ld uses them
module IOBUF (inout wire IO, input logic I, input logic T, output logic O);
  assign IO = T ? 1'bz : I;
  assign O = IO;
endmodule
module IBUF (input logic I, output logic O);
  assign O = I;
endmodule
module BUFG (input logic I, output logic O);
  assign O = I;
endmodule
