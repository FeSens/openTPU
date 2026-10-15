// Lint-only stand-in for the Vivado-generated block design wrapper of the LiteDRAM build
// (boards/ypcb-00338/vivado/bd_native.tcl): the ports Vivado will generate, no behaviour. Used by
// `make lint` to check otpu_fpga_top_ld offline; Vivado builds with the real otpu_bd_wrapper.v
// (and the unisim IBUF, BUFG, IOBUF). S_AXI_CTL is LitePCIe's BAR0 master (AXI-Lite with AxPROT).
module otpu_bd_wrapper (
  input  logic        sys_clk_50, pcie_perstn, pcie_clk, pcie_rstn,
  output logic        core_clk, core_clk2x, core_rstn,
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
  input  logic [31:0] S_AXI_CTL_awaddr, S_AXI_CTL_araddr, S_AXI_CTL_wdata,
  input  logic [2:0]  S_AXI_CTL_awprot, S_AXI_CTL_arprot,
  input  logic [3:0]  S_AXI_CTL_wstrb,
  input  logic        S_AXI_CTL_awvalid, S_AXI_CTL_wvalid, S_AXI_CTL_bready, S_AXI_CTL_arvalid,
  input  logic        S_AXI_CTL_rready,
  output logic        S_AXI_CTL_awready, S_AXI_CTL_wready, S_AXI_CTL_bvalid, S_AXI_CTL_arready,
  output logic        S_AXI_CTL_rvalid,
  output logic [1:0]  S_AXI_CTL_bresp, S_AXI_CTL_rresp,
  output logic [31:0] S_AXI_CTL_rdata
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
