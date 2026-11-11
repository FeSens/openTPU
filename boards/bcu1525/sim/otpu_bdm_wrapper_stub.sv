// Lint-only stand-in for the Vivado-generated wrapper of the BCU1525 memory-test image's block
// design (boards/bcu1525/vivado/bd_mem.tcl): the ports Vivado will generate, no behaviour. `make
// lint` checks otpu_fpga_top_bcu_mem offline against it.
module otpu_bdm_wrapper (
  input  logic        clk300_in, core_rstn,
  output logic        core_clk, core_locked, temp_alarm, ot_alarm,
  output logic [9:0]  device_temp,
`define STUB_L(P) \
  output logic [31:0] P``_awaddr, P``_araddr, P``_wdata, \
  output logic [3:0]  P``_wstrb, \
  output logic        P``_awvalid, P``_wvalid, P``_bready, P``_arvalid, P``_rready, \
  input  logic        P``_awready, P``_wready, P``_bvalid, P``_arready, P``_rvalid, \
  input  logic [1:0]  P``_bresp, P``_rresp, \
  input  logic [31:0] P``_rdata
  `STUB_L(M_AXI_GOV),
  `STUB_L(M_AXI_BMC),
  `STUB_L(M_AXI_MEMCAL0),
  `STUB_L(M_AXI_MEMCAL1),
  `STUB_L(M_AXI_MEMCAL2),
  `STUB_L(M_AXI_MEMCAL3)
);
`undef STUB_L
endmodule
