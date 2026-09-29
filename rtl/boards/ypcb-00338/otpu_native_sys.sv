// The accelerator and XDMA's DMA on two native memory channels (otpu_fpga_top_ld: LiteDRAM's
// native ports): otpu_board (otpu_native_dram on the channels' native masters n_*), XDMA's AXI
// master split by address bit 31 onto the channels (otpu_axi_split2), and one otpu_mem_ch per
// channel in front of that channel's controller port c_* in the controller's clock (uclk[c];
// partial beats read-modify-written there: LiteDRAM's ECC port takes whole beats only). Its
// crossings' constraints: constraints/otpu_mem_ch.tcl (scoped to each otpu_mem_ch) and
// otpu_top_native.tcl.
//   HOSTCAL: the host calibrates the controllers (CAPS bit27: LiteDRAM, whose calibration the
//        host runs through its CSRs; opentpu/host/memcal.py).
//   calib: each channel's calibration flag in its controller's clock (STATUS CALIB0/1; synchronized
//        in otpu_board).
module otpu_native_sys #(
  parameter int MCOLS = 2,
  parameter int ACT_ROWS = MCOLS,
  parameter int VPU_CL = 2,
  parameter int MXU_IMPL = 0,            // MXU dot product: 0 adder tree, 2 systolic
  parameter int LANES = 8,
  parameter int ULANES = 8,
  parameter int CORE_KHZ = 100000,
  parameter logic [31:0] BUILD_ID = 32'h0,
  parameter int DDR_MTS = 0,
  parameter bit DSTEP = 1'b1,
  parameter bit HOSTCAL = 1'b0
) (
  // core clock and reset (synchronous, high), XDMA's clock and reset
  input  logic                  clk,
  input  logic                  rst,
  input  logic                  xclk,
  input  logic                  xrst,
  // board status and pins (otpu_board)
  input  logic [1:0]            calib,
  input  logic [11:0]           temp,
  output logic [2:0]            led,
  output logic [3:0]            i2c_lo,
  input  logic [4:0]            i2c_pin,
  // control registers: AXI4-Lite slave, core_clk (BAR0 0x0)
  input  logic [11:0]           s_ctl_awaddr,
  input  logic                  s_ctl_awvalid,
  output logic                  s_ctl_awready,
  input  logic [31:0]           s_ctl_wdata,
  input  logic [3:0]            s_ctl_wstrb,
  input  logic                  s_ctl_wvalid,
  output logic                  s_ctl_wready,
  output logic [1:0]            s_ctl_bresp,
  output logic                  s_ctl_bvalid,
  input  logic                  s_ctl_bready,
  input  logic [11:0]           s_ctl_araddr,
  input  logic                  s_ctl_arvalid,
  output logic                  s_ctl_arready,
  output logic [31:0]           s_ctl_rdata,
  output logic [1:0]            s_ctl_rresp,
  output logic                  s_ctl_rvalid,
  input  logic                  s_ctl_rready,
  // XDMA's DMA master: AXI4, 128 bits, xclk (INCR bursts of full-width beats)
  input  logic                  x_awvalid,
  output logic                  x_awready,
  input  logic [3:0]            x_awid,
  input  logic [31:0]           x_awaddr,
  input  logic [7:0]            x_awlen,
  input  logic                  x_wvalid,
  output logic                  x_wready,
  input  logic [127:0]          x_wdata,
  input  logic [15:0]           x_wstrb,
  input  logic                  x_wlast,
  output logic                  x_bvalid,
  input  logic                  x_bready,
  output logic [3:0]            x_bid,
  output logic [1:0]            x_bresp,
  input  logic                  x_arvalid,
  output logic                  x_arready,
  input  logic [3:0]            x_arid,
  input  logic [31:0]           x_araddr,
  input  logic [7:0]            x_arlen,
  output logic                  x_rvalid,
  input  logic                  x_rready,
  output logic [3:0]            x_rid,
  output logic [127:0]          x_rdata,
  output logic [1:0]            x_rresp,
  output logic                  x_rlast,
  // the channels' controller ports, channel c in uclk[c] (otpu_mem_ch's c_*)
  input  logic [1:0]            uclk,
  input  logic [1:0]            urst,
  output logic [1:0]            c_cmd_valid,
  input  logic [1:0]            c_cmd_ready,
  output logic [1:0]            c_cmd_we,
  output logic [1:0][24:0]      c_cmd_addr,
  output logic [1:0]            c_wdata_valid,
  input  logic [1:0]            c_wdata_ready,
  output logic [1:0][511:0]     c_wdata_data,
  output logic [1:0][63:0]      c_wdata_we,
  input  logic [1:0]            c_rdata_valid,
  input  logic [1:0][511:0]     c_rdata_data
);
  // the accelerator's native masters, per channel
  logic [1:0]        n_cvalid, n_cready, n_cwe, n_wvalid, n_wready, n_rvalid;
  logic [1:0][24:0]  n_caddr;
  logic [1:0][511:0] n_wdata, n_rdata;
  logic [1:0][63:0]  n_wmask;
  logic [1:0][15:0]  n_wdone;
  // XDMA's per channel (otpu_axi_split2's masters)
  logic [1:0]        m_awvalid, m_awready, m_wvalid, m_wready, m_bvalid, m_bready;
  logic [1:0]        m_arvalid, m_arready, m_rvalid, m_rready, m_rlast;
  logic [1:0][3:0]   m_bid, m_rid;
  logic [1:0][1:0]   m_bresp, m_rresp;
  logic [1:0][127:0] m_rdata;

  otpu_axi_split2 #(.IDW(4), .DW(128)) u_split (
    .clk(xclk), .rst(xrst),
    .s_awvalid(x_awvalid), .s_awready(x_awready), .s_awid(x_awid), .s_awaddr(x_awaddr),
    .s_awlen(x_awlen), .s_wvalid(x_wvalid), .s_wready(x_wready), .s_wdata(x_wdata),
    .s_wstrb(x_wstrb), .s_wlast(x_wlast), .s_bvalid(x_bvalid), .s_bready(x_bready),
    .s_bid(x_bid), .s_bresp(x_bresp), .s_arvalid(x_arvalid), .s_arready(x_arready),
    .s_arid(x_arid), .s_araddr(x_araddr), .s_arlen(x_arlen), .s_rvalid(x_rvalid),
    .s_rready(x_rready), .s_rid(x_rid), .s_rdata(x_rdata), .s_rresp(x_rresp), .s_rlast(x_rlast),
    .m_awvalid, .m_awready, .m_wvalid, .m_wready, .m_bvalid, .m_bready, .m_bid, .m_bresp,
    .m_arvalid, .m_arready, .m_rvalid, .m_rready, .m_rid, .m_rdata, .m_rresp, .m_rlast);

  // one bridge per channel; XDMA's address, ID, length and write data go to both (the split's
  // valids select one)
  otpu_mem_ch #(.XIDW(4)) u_ch0 (
    .clk, .rst,
    .n_cvalid(n_cvalid[0]), .n_cready(n_cready[0]), .n_cwe(n_cwe[0]), .n_caddr(n_caddr[0]),
    .n_wvalid(n_wvalid[0]), .n_wready(n_wready[0]), .n_wdata(n_wdata[0]), .n_wmask(n_wmask[0]),
    .n_rvalid(n_rvalid[0]), .n_rdata(n_rdata[0]), .n_wdone(n_wdone[0]),
    .xclk, .xrst,
    .x_awvalid(m_awvalid[0]), .x_awready(m_awready[0]), .x_awid, .x_awaddr, .x_awlen,
    .x_wvalid(m_wvalid[0]), .x_wready(m_wready[0]), .x_wdata, .x_wstrb, .x_wlast,
    .x_bvalid(m_bvalid[0]), .x_bready(m_bready[0]), .x_bid(m_bid[0]), .x_bresp(m_bresp[0]),
    .x_arvalid(m_arvalid[0]), .x_arready(m_arready[0]), .x_arid, .x_araddr, .x_arlen,
    .x_rvalid(m_rvalid[0]), .x_rready(m_rready[0]), .x_rid(m_rid[0]), .x_rdata(m_rdata[0]),
    .x_rresp(m_rresp[0]), .x_rlast(m_rlast[0]),
    .uclk(uclk[0]), .urst(urst[0]),
    .c_cmd_valid(c_cmd_valid[0]), .c_cmd_ready(c_cmd_ready[0]), .c_cmd_we(c_cmd_we[0]),
    .c_cmd_addr(c_cmd_addr[0]),
    .c_wdata_valid(c_wdata_valid[0]), .c_wdata_ready(c_wdata_ready[0]),
    .c_wdata_data(c_wdata_data[0]), .c_wdata_we(c_wdata_we[0]),
    .c_rdata_valid(c_rdata_valid[0]), .c_rdata_data(c_rdata_data[0]));
  otpu_mem_ch #(.XIDW(4)) u_ch1 (
    .clk, .rst,
    .n_cvalid(n_cvalid[1]), .n_cready(n_cready[1]), .n_cwe(n_cwe[1]), .n_caddr(n_caddr[1]),
    .n_wvalid(n_wvalid[1]), .n_wready(n_wready[1]), .n_wdata(n_wdata[1]), .n_wmask(n_wmask[1]),
    .n_rvalid(n_rvalid[1]), .n_rdata(n_rdata[1]), .n_wdone(n_wdone[1]),
    .xclk, .xrst,
    .x_awvalid(m_awvalid[1]), .x_awready(m_awready[1]), .x_awid, .x_awaddr, .x_awlen,
    .x_wvalid(m_wvalid[1]), .x_wready(m_wready[1]), .x_wdata, .x_wstrb, .x_wlast,
    .x_bvalid(m_bvalid[1]), .x_bready(m_bready[1]), .x_bid(m_bid[1]), .x_bresp(m_bresp[1]),
    .x_arvalid(m_arvalid[1]), .x_arready(m_arready[1]), .x_arid, .x_araddr, .x_arlen,
    .x_rvalid(m_rvalid[1]), .x_rready(m_rready[1]), .x_rid(m_rid[1]), .x_rdata(m_rdata[1]),
    .x_rresp(m_rresp[1]), .x_rlast(m_rlast[1]),
    .uclk(uclk[1]), .urst(urst[1]),
    .c_cmd_valid(c_cmd_valid[1]), .c_cmd_ready(c_cmd_ready[1]), .c_cmd_we(c_cmd_we[1]),
    .c_cmd_addr(c_cmd_addr[1]),
    .c_wdata_valid(c_wdata_valid[1]), .c_wdata_ready(c_wdata_ready[1]),
    .c_wdata_data(c_wdata_data[1]), .c_wdata_we(c_wdata_we[1]),
    .c_rdata_valid(c_rdata_valid[1]), .c_rdata_data(c_rdata_data[1]));

  otpu_board #(.MCOLS(MCOLS), .ACT_ROWS(ACT_ROWS), .VPU_CL(VPU_CL), .MXU_IMPL(MXU_IMPL),
               .LANES(LANES), .ULANES(ULANES),
               .CORE_KHZ(CORE_KHZ), .BUILD_ID(BUILD_ID), .DDR_MTS(DDR_MTS), .DSTEP(DSTEP),
               .HOSTCAL(HOSTCAL)) u_board (
    .clk, .rst, .calib, .temp, .led, .i2c_lo, .i2c_pin,
    .s_ctl_awaddr, .s_ctl_awvalid, .s_ctl_awready, .s_ctl_wdata, .s_ctl_wstrb, .s_ctl_wvalid,
    .s_ctl_wready, .s_ctl_bresp, .s_ctl_bvalid, .s_ctl_bready, .s_ctl_araddr, .s_ctl_arvalid,
    .s_ctl_arready, .s_ctl_rdata, .s_ctl_rresp, .s_ctl_rvalid, .s_ctl_rready,
    .n_cvalid, .n_cready, .n_cwe, .n_caddr, .n_wvalid, .n_wready, .n_wdata, .n_wmask,
    .n_rvalid, .n_rdata, .n_wdone);
endmodule
