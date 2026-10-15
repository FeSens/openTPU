// Lint-only stand-in for the generated LitePCIe core (boards/ypcb-00338/litepcie/otpu_litepcie.v,
// tools/litepcie/gen_core.py): its ports, no behaviour (the core instantiates the pcie_7x IP and
// unisim primitives, which Verilator does not have). Used by `make lint`.
module otpu_litepcie (
  input  wire          pcie_clk_p, pcie_clk_n, pcie_rst_n,
  input  wire  [7:0]   pcie_rx_p, pcie_rx_n,
  output wire  [7:0]   pcie_tx_p, pcie_tx_n,
  output wire          clk, rst, link_up,
  output wire  [31:0]  mmap_axi_lite_awaddr, mmap_axi_lite_araddr, mmap_axi_lite_wdata,
  output wire  [2:0]   mmap_axi_lite_awprot, mmap_axi_lite_arprot,
  output wire  [3:0]   mmap_axi_lite_wstrb,
  output wire          mmap_axi_lite_awvalid, mmap_axi_lite_wvalid, mmap_axi_lite_bready,
  output wire          mmap_axi_lite_arvalid, mmap_axi_lite_rready,
  input  wire          mmap_axi_lite_awready, mmap_axi_lite_wready, mmap_axi_lite_bvalid,
  input  wire          mmap_axi_lite_arready, mmap_axi_lite_rvalid,
  input  wire  [1:0]   mmap_axi_lite_bresp, mmap_axi_lite_rresp,
  input  wire  [31:0]  mmap_axi_lite_rdata,
  output wire          h2c_cmd_valid, input wire h2c_cmd_ready,
  output wire  [31:0]  h2c_cmd_addr, output wire [23:0] h2c_cmd_len, output wire [7:0] h2c_cmd_flags,
  output wire          h2c_dat_valid, input wire h2c_dat_ready,
  output wire  [127:0] h2c_dat_data, output wire h2c_dat_last,
  output wire          c2h_cmd_valid, input wire c2h_cmd_ready,
  output wire  [31:0]  c2h_cmd_addr, output wire [23:0] c2h_cmd_len, output wire [7:0] c2h_cmd_flags,
  input  wire          c2h_dat_valid, output wire c2h_dat_ready,
  input  wire  [127:0] c2h_dat_data, input wire c2h_dat_last,
  input  wire          mm_h2c_done, mm_err,
  input  wire  [31:0]  mm_status,
  output wire          mm_rst
);
endmodule
