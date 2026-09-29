// FPGA top for the Inspur YPCB-00338 (xc7k480t-ffg1156-2) with the MIGs' native interface
// (create_project.tcl MEM=mig_native; otpu_fpga_top.sv is the AXI MIG build): the block design
// otpu_bd (PCIe XDMA, the clocks, the XADC, the control interconnect;
// boards/ypcb-00338/vivado/bd_native.tcl), the two DDR3 MIGs as RTL-level IP with their native
// user interface (u_mig: mig_ddr3_ch0 / 1 from create_project.tcl, the AXI build's .prj with
// PortInterface NATIVE; IP integrator takes the MIG with AXI only), one otpu_mig_native per
// channel (otpu_mem_ch's controller port onto the MIG's app_* interface, in its ui_clk; partial
// beats as wr_bytes, the ECC controller's read-modify-write), and otpu_native_sys: the
// accelerator on the channels' native ports (otpu_board with MEM_NATIVE: otpu_native_dram) and
// XDMA's DMA master, meeting in one otpu_mem_ch per channel (RMW = 0). The MIGs calibrate
// themselves, as in the AXI build (STATUS CALIB0/1 from init_calib_complete; CAPS bit27 clear).
//
// Clocks: core_clk and the MIGs' system and reference clocks from the block design's MMCM on the
// 50 MHz pin (as bd.tcl); each MIG's ui_clk (4:1 of its memory clock: 133.33 MHz at DDR3-1066)
// is its channel's controller clock; xdma_aclk (125 MHz) is XDMA's. otpu_mem_ch crosses between
// them (constraints/otpu_mem_ch.tcl, otpu_top_native.tcl: no asynchronous clock groups).
// Resets: core_rst from the block design (MMCM lock, PCIe PERST#), xrst from XDMA's axi_aresetn,
// the MIGs' sys_rst from the clock wizard's lock (as bd.tcl), each channel's controller side from
// its MIG's ui_clk_sync_rst.
// Port names are otpu_fpga_top.sv's (DDR3_<c>_<signal>), and the MIG instances sit at
// u_mig/mig_<c> (as the block design's .../mig_<c>), so constraints/otpu_top.xdc (with its mig_0
// ILOGIC placement) and otpu_ddr3_pins.xdc apply unchanged.
module otpu_fpga_top_mn #(
  parameter int MCOLS = 2,                  // MXU columns (activation rows per weight chunk)
  parameter int ACT_ROWS = MCOLS,           // ACT RAM rows (> MCOLS: MM replay, one weight pass)
  parameter int VPU_CL = 2,                 // VPU lanes with the composite functions (exp2, ...)
  parameter int MXU_IMPL = 0,               // MXU dot product: 0 adder tree, 2 systolic (docs/mxu_systolic.md)
  parameter int LANES = 8,                  // VPU lanes / TMEM banks (8 or 16)
  parameter int ULANES = 8,                 // TMEM lanes of the MXU and the quantizer
  parameter int CORE_KHZ = 100000,          // core_clk as the block design makes it (CORE_KHZ register)
  parameter logic [31:0] BUILD_ID = 32'h0,  // the git commit (BUILD_ID register)
  parameter int DDR_MTS = 0,                // the DDR3 data rate the MIGs run (DDR_MTS register)
  parameter bit DSTEP = 1'b1,               // the DMA's DSTEP datapath (CAPS bit6; 0: left out)
  parameter int AXI_BL = 8,                 // (the AXI adapter's; unused here, kept for the build's generics)
  parameter int AXI_WBL = 8
) (
  // board
  input  logic        SYS_CLK,              // 50 MHz, AA28
  output logic [2:0]  led,                  // P30 red, M30 green, N30 yellow
  // I2C, open drain (otpu_ctrl I2C_CTRL / I2C_IN; the host bit-bangs them)
  inout  wire         lm73_scl,             // N24  the LM73 temperature sensor's bus
  inout  wire         lm73_sda,             // N25
  input  logic        lm73_alert_n,         // P25
  inout  wire         smb_scl,              // R26  the PCIe edge connector's SMBus
  inout  wire         smb_sda,              // R27
  // PCIe Gen2 x8
  input  logic        pcie_refclk_clk_p,    // J8 (MGTREFCLK)
  input  logic        pcie_refclk_clk_n,
  input  logic        pcie_perstn,          // Y26
  input  logic [7:0]  pcie_mgt_rxp,
  input  logic [7:0]  pcie_mgt_rxn,
  output logic [7:0]  pcie_mgt_txp,
  output logic [7:0]  pcie_mgt_txn,
  // DDR3 channel 0
  inout  wire  [71:0] DDR3_0_dq,
  inout  wire  [8:0]  DDR3_0_dqs_p,
  inout  wire  [8:0]  DDR3_0_dqs_n,
  output logic [14:0] DDR3_0_addr,
  output logic [2:0]  DDR3_0_ba,
  output logic        DDR3_0_ras_n,
  output logic        DDR3_0_cas_n,
  output logic        DDR3_0_we_n,
  output logic        DDR3_0_reset_n,
  output logic [0:0]  DDR3_0_ck_p,
  output logic [0:0]  DDR3_0_ck_n,
  output logic [0:0]  DDR3_0_cke,
  output logic [0:0]  DDR3_0_cs_n,
  output logic [0:0]  DDR3_0_odt,
  // DDR3 channel 1
  inout  wire  [71:0] DDR3_1_dq,
  inout  wire  [8:0]  DDR3_1_dqs_p,
  inout  wire  [8:0]  DDR3_1_dqs_n,
  output logic [14:0] DDR3_1_addr,
  output logic [2:0]  DDR3_1_ba,
  output logic        DDR3_1_ras_n,
  output logic        DDR3_1_cas_n,
  output logic        DDR3_1_we_n,
  output logic        DDR3_1_reset_n,
  output logic [0:0]  DDR3_1_ck_p,
  output logic [0:0]  DDR3_1_ck_n,
  output logic [0:0]  DDR3_1_cke,
  output logic [0:0]  DDR3_1_cs_n,
  output logic [0:0]  DDR3_1_odt
);
  logic        core_clk, core_rstn, xdma_aclk, xdma_aresetn, pcie_link_up;
  logic [1:0]  calib;
  logic [11:0] device_temp;                  // XADC die-temperature code (core_clk)

  // ---- control (AXI4-Lite, BD master -> accelerator, core_clk)
  logic [31:0] ctl_awaddr, ctl_araddr, ctl_wdata, ctl_rdata;
  logic [3:0]  ctl_wstrb;
  logic [1:0]  ctl_bresp, ctl_rresp;
  logic ctl_awvalid, ctl_awready, ctl_wvalid, ctl_wready, ctl_bvalid, ctl_bready;
  logic ctl_arvalid, ctl_arready, ctl_rvalid, ctl_rready;
  // ---- XDMA's DMA master (AXI4, 128 bits, xdma_aclk)
  logic [3:0]   dm_awid, dm_arid, dm_bid, dm_rid;
  logic [63:0]  dm_awaddr, dm_araddr;
  logic [7:0]   dm_awlen, dm_arlen;
  logic [127:0] dm_wdata, dm_rdata;
  logic [15:0]  dm_wstrb;
  logic [1:0]   dm_bresp, dm_rresp;
  logic dm_awvalid, dm_awready, dm_wlast, dm_wvalid, dm_wready, dm_bvalid, dm_bready;
  logic dm_arvalid, dm_arready, dm_rlast, dm_rvalid, dm_rready;
  // ---- the MIGs' clocks and reset from the block design, their user clocks and resets
  logic        mig_sys_clk, mig_ref_clk, mig_locked;
  logic [1:0]  ui_clk, ui_rst;

  otpu_bd_wrapper u_bd (
    .sys_clk_50(SYS_CLK),
    .pcie_refclk_clk_p, .pcie_refclk_clk_n, .pcie_perstn,
    .pcie_mgt_rxp, .pcie_mgt_rxn, .pcie_mgt_txp, .pcie_mgt_txn,
    .pcie_link_up, .core_clk, .core_rstn, .xdma_aclk, .xdma_aresetn, .device_temp,
    .mig_sys_clk, .mig_ref_clk, .mig_locked,
    .M_AXI_CTL_awaddr(ctl_awaddr), .M_AXI_CTL_awvalid(ctl_awvalid),
    .M_AXI_CTL_awready(ctl_awready), .M_AXI_CTL_wdata(ctl_wdata), .M_AXI_CTL_wstrb(ctl_wstrb),
    .M_AXI_CTL_wvalid(ctl_wvalid), .M_AXI_CTL_wready(ctl_wready),
    .M_AXI_CTL_bresp(ctl_bresp), .M_AXI_CTL_bvalid(ctl_bvalid), .M_AXI_CTL_bready(ctl_bready),
    .M_AXI_CTL_araddr(ctl_araddr), .M_AXI_CTL_arvalid(ctl_arvalid),
    .M_AXI_CTL_arready(ctl_arready), .M_AXI_CTL_rdata(ctl_rdata), .M_AXI_CTL_rresp(ctl_rresp),
    .M_AXI_CTL_rvalid(ctl_rvalid), .M_AXI_CTL_rready(ctl_rready),
    .M_AXI_DMA_awid(dm_awid), .M_AXI_DMA_awaddr(dm_awaddr), .M_AXI_DMA_awlen(dm_awlen),
    .M_AXI_DMA_awvalid(dm_awvalid), .M_AXI_DMA_awready(dm_awready),
    .M_AXI_DMA_wdata(dm_wdata), .M_AXI_DMA_wstrb(dm_wstrb), .M_AXI_DMA_wlast(dm_wlast),
    .M_AXI_DMA_wvalid(dm_wvalid), .M_AXI_DMA_wready(dm_wready),
    .M_AXI_DMA_bid(dm_bid), .M_AXI_DMA_bresp(dm_bresp), .M_AXI_DMA_bvalid(dm_bvalid),
    .M_AXI_DMA_bready(dm_bready),
    .M_AXI_DMA_arid(dm_arid), .M_AXI_DMA_araddr(dm_araddr), .M_AXI_DMA_arlen(dm_arlen),
    .M_AXI_DMA_arvalid(dm_arvalid), .M_AXI_DMA_arready(dm_arready),
    .M_AXI_DMA_rid(dm_rid), .M_AXI_DMA_rdata(dm_rdata), .M_AXI_DMA_rresp(dm_rresp),
    .M_AXI_DMA_rlast(dm_rlast), .M_AXI_DMA_rvalid(dm_rvalid), .M_AXI_DMA_rready(dm_rready),
    // XDMA's bursts are INCR of full-width beats (otpu_mem_ch relies on it): these are unused
    .M_AXI_DMA_awsize(), .M_AXI_DMA_awburst(), .M_AXI_DMA_awlock(), .M_AXI_DMA_awcache(),
    .M_AXI_DMA_awprot(), .M_AXI_DMA_arsize(), .M_AXI_DMA_arburst(), .M_AXI_DMA_arlock(),
    .M_AXI_DMA_arcache(), .M_AXI_DMA_arprot()
  );

  // resets: synchronous, active high, one register each (otpu_mem_ch crosses them registered)
  logic core_rst, xrst;
  always_ff @(posedge core_clk) core_rst <= !core_rstn;
  always_ff @(posedge xdma_aclk) xrst <= !xdma_aresetn;

  // ---- the channels' controller ports (otpu_mem_ch <-> otpu_mig_native), channel c in ui_clk[c]
  logic [1:0]        c_cmd_valid, c_cmd_ready, c_cmd_we, c_cmd_partial;
  logic [1:0]        c_wdata_valid, c_wdata_ready, c_rdata_valid;
  logic [1:0][24:0]  c_cmd_addr;
  logic [1:0][511:0] c_wdata_data, c_rdata_data;
  logic [1:0][63:0]  c_wdata_we;
  otpu_mig_pair u_mig (
    .sys_clk(mig_sys_clk), .ref_clk(mig_ref_clk), .locked(mig_locked), .temp(device_temp),
    .ui_clk, .ui_rst, .calib,
    .c_cmd_valid, .c_cmd_ready, .c_cmd_we, .c_cmd_addr, .c_cmd_partial, .c_wdata_valid,
    .c_wdata_ready, .c_wdata_data, .c_wdata_we, .c_rdata_valid, .c_rdata_data,
    .DDR3_0_dq, .DDR3_0_dqs_p, .DDR3_0_dqs_n, .DDR3_0_addr, .DDR3_0_ba, .DDR3_0_ras_n,
    .DDR3_0_cas_n, .DDR3_0_we_n, .DDR3_0_reset_n, .DDR3_0_ck_p, .DDR3_0_ck_n, .DDR3_0_cke,
    .DDR3_0_cs_n, .DDR3_0_odt,
    .DDR3_1_dq, .DDR3_1_dqs_p, .DDR3_1_dqs_n, .DDR3_1_addr, .DDR3_1_ba, .DDR3_1_ras_n,
    .DDR3_1_cas_n, .DDR3_1_we_n, .DDR3_1_reset_n, .DDR3_1_ck_p, .DDR3_1_ck_n, .DDR3_1_cke,
    .DDR3_1_cs_n, .DDR3_1_odt);

  // I2C: each line released (high-Z, pulled up) unless its I2C_CTRL bit drives it low
  logic [3:0] i2c_lo, i2c_lvl;
  IOBUF u_iob_scl0 (.IO(lm73_scl), .I(1'b0), .T(!i2c_lo[0]), .O(i2c_lvl[0]));
  IOBUF u_iob_sda0 (.IO(lm73_sda), .I(1'b0), .T(!i2c_lo[1]), .O(i2c_lvl[1]));
  IOBUF u_iob_scl1 (.IO(smb_scl),  .I(1'b0), .T(!i2c_lo[2]), .O(i2c_lvl[2]));
  IOBUF u_iob_sda1 (.IO(smb_sda),  .I(1'b0), .T(!i2c_lo[3]), .O(i2c_lvl[3]));

  // ---- the accelerator and XDMA on the channels
  logic [2:0] board_led;
  otpu_native_sys #(.MCOLS(MCOLS), .ACT_ROWS(ACT_ROWS), .VPU_CL(VPU_CL), .MXU_IMPL(MXU_IMPL),
                    .LANES(LANES),
                    .ULANES(ULANES), .CORE_KHZ(CORE_KHZ), .BUILD_ID(BUILD_ID), .DDR_MTS(DDR_MTS),
                    .DSTEP(DSTEP), .AXI_BL(AXI_BL), .AXI_WBL(AXI_WBL), .RMW(1'b0)) u_sys (
    .clk(core_clk), .rst(core_rst), .xclk(xdma_aclk), .xrst,
    .calib, .temp(device_temp), .led(board_led), .i2c_lo, .i2c_pin({lm73_alert_n, i2c_lvl}),
    .s_ctl_awaddr(ctl_awaddr[11:0]), .s_ctl_awvalid(ctl_awvalid), .s_ctl_awready(ctl_awready),
    .s_ctl_wdata(ctl_wdata), .s_ctl_wstrb(ctl_wstrb), .s_ctl_wvalid(ctl_wvalid),
    .s_ctl_wready(ctl_wready), .s_ctl_bresp(ctl_bresp), .s_ctl_bvalid(ctl_bvalid),
    .s_ctl_bready(ctl_bready), .s_ctl_araddr(ctl_araddr[11:0]), .s_ctl_arvalid(ctl_arvalid),
    .s_ctl_arready(ctl_arready), .s_ctl_rdata(ctl_rdata), .s_ctl_rresp(ctl_rresp),
    .s_ctl_rvalid(ctl_rvalid), .s_ctl_rready(ctl_rready),
    .x_awvalid(dm_awvalid), .x_awready(dm_awready), .x_awid(dm_awid), .x_awaddr(dm_awaddr[31:0]),
    .x_awlen(dm_awlen), .x_wvalid(dm_wvalid), .x_wready(dm_wready), .x_wdata(dm_wdata),
    .x_wstrb(dm_wstrb), .x_wlast(dm_wlast), .x_bvalid(dm_bvalid), .x_bready(dm_bready),
    .x_bid(dm_bid), .x_bresp(dm_bresp), .x_arvalid(dm_arvalid), .x_arready(dm_arready),
    .x_arid(dm_arid), .x_araddr(dm_araddr[31:0]), .x_arlen(dm_arlen), .x_rvalid(dm_rvalid),
    .x_rready(dm_rready), .x_rid(dm_rid), .x_rdata(dm_rdata), .x_rresp(dm_rresp),
    .x_rlast(dm_rlast),
    .uclk(ui_clk), .urst(ui_rst),
    .c_cmd_valid, .c_cmd_ready, .c_cmd_we, .c_cmd_addr, .c_cmd_partial, .c_wdata_valid,
    .c_wdata_ready, .c_wdata_data, .c_wdata_we, .c_rdata_valid, .c_rdata_data);

  // LEDs (as otpu_fpga_top): led[0] red = heartbeat, led[1] green = PCIe link up and both DDR3
  // channels calibrated, led[2] yellow = the accelerator runs or halted cleanly
  assign led = {board_led[2] | board_led[1], pcie_link_up & (&calib), board_led[0]};
endmodule

// Both DDR3 channels' MIGs (RTL-level IP mig_ddr3_ch0 / 1, native interface, ECC, DDR3 pins as
// the top's) with their otpu_mig_native shims. The instances are named mig_0 / mig_1 one level
// down, as in the AXI build's block design (.../mig_0), which otpu_top.xdc's placement of channel
// 0's ILOGIC for ddr3_reset_n matches (*/mig_0/*).
module otpu_mig_pair (
  input  logic              sys_clk,     // the MIGs' system clock (No Buffer), reference clock
  input  logic              ref_clk,     // (200 MHz, IDELAYCTRL), and the clock wizard's lock
  input  logic              locked,      // (their active-low sys_rst)
  input  logic [11:0]       temp,        // XADC die temperature (device_temp_i)
  output logic [1:0]        ui_clk,      // channel c's user clock and reset
  output logic [1:0]        ui_rst,
  output logic [1:0]        calib,       // init_calib_complete, in ui_clk[c]
  // the channels' generic controller ports (otpu_mem_ch), channel c in ui_clk[c]
  input  logic [1:0]        c_cmd_valid,
  output logic [1:0]        c_cmd_ready,
  input  logic [1:0]        c_cmd_we,
  input  logic [1:0][24:0]  c_cmd_addr,
  input  logic [1:0]        c_cmd_partial,
  input  logic [1:0]        c_wdata_valid,
  output logic [1:0]        c_wdata_ready,
  input  logic [1:0][511:0] c_wdata_data,
  input  logic [1:0][63:0]  c_wdata_we,
  output logic [1:0]        c_rdata_valid,
  output logic [1:0][511:0] c_rdata_data,
  // DDR3 channel 0
  inout  wire  [71:0] DDR3_0_dq,
  inout  wire  [8:0]  DDR3_0_dqs_p,
  inout  wire  [8:0]  DDR3_0_dqs_n,
  output logic [14:0] DDR3_0_addr,
  output logic [2:0]  DDR3_0_ba,
  output logic        DDR3_0_ras_n,
  output logic        DDR3_0_cas_n,
  output logic        DDR3_0_we_n,
  output logic        DDR3_0_reset_n,
  output logic [0:0]  DDR3_0_ck_p,
  output logic [0:0]  DDR3_0_ck_n,
  output logic [0:0]  DDR3_0_cke,
  output logic [0:0]  DDR3_0_cs_n,
  output logic [0:0]  DDR3_0_odt,
  // DDR3 channel 1
  inout  wire  [71:0] DDR3_1_dq,
  inout  wire  [8:0]  DDR3_1_dqs_p,
  inout  wire  [8:0]  DDR3_1_dqs_n,
  output logic [14:0] DDR3_1_addr,
  output logic [2:0]  DDR3_1_ba,
  output logic        DDR3_1_ras_n,
  output logic        DDR3_1_cas_n,
  output logic        DDR3_1_we_n,
  output logic        DDR3_1_reset_n,
  output logic [0:0]  DDR3_1_ck_p,
  output logic [0:0]  DDR3_1_ck_n,
  output logic [0:0]  DDR3_1_cke,
  output logic [0:0]  DDR3_1_cs_n,
  output logic [0:0]  DDR3_1_odt
);
  // the MIGs' native interfaces, channel c in ui_clk[c]
  logic [1:0][28:0]  app_addr;
  logic [1:0][2:0]   app_cmd;
  logic [1:0]        app_en, app_rdy, app_wdf_wren, app_wdf_end, app_wdf_rdy, app_rd_data_valid;
  logic [1:0][511:0] app_wdf_data, app_rd_data;
  logic [1:0][63:0]  app_wdf_mask;

  otpu_mig_native u_mn0 (
    .c_cmd_valid(c_cmd_valid[0]), .c_cmd_ready(c_cmd_ready[0]), .c_cmd_we(c_cmd_we[0]),
    .c_cmd_addr(c_cmd_addr[0]), .c_cmd_partial(c_cmd_partial[0]),
    .c_wdata_valid(c_wdata_valid[0]), .c_wdata_ready(c_wdata_ready[0]),
    .c_wdata_data(c_wdata_data[0]), .c_wdata_we(c_wdata_we[0]),
    .c_rdata_valid(c_rdata_valid[0]), .c_rdata_data(c_rdata_data[0]),
    .app_addr(app_addr[0]), .app_cmd(app_cmd[0]), .app_en(app_en[0]), .app_rdy(app_rdy[0]),
    .app_wdf_data(app_wdf_data[0]), .app_wdf_mask(app_wdf_mask[0]),
    .app_wdf_wren(app_wdf_wren[0]), .app_wdf_end(app_wdf_end[0]),
    .app_wdf_rdy(app_wdf_rdy[0]), .app_rd_data(app_rd_data[0]),
    .app_rd_data_valid(app_rd_data_valid[0]));
  otpu_mig_native u_mn1 (
    .c_cmd_valid(c_cmd_valid[1]), .c_cmd_ready(c_cmd_ready[1]), .c_cmd_we(c_cmd_we[1]),
    .c_cmd_addr(c_cmd_addr[1]), .c_cmd_partial(c_cmd_partial[1]),
    .c_wdata_valid(c_wdata_valid[1]), .c_wdata_ready(c_wdata_ready[1]),
    .c_wdata_data(c_wdata_data[1]), .c_wdata_we(c_wdata_we[1]),
    .c_rdata_valid(c_rdata_valid[1]), .c_rdata_data(c_rdata_data[1]),
    .app_addr(app_addr[1]), .app_cmd(app_cmd[1]), .app_en(app_en[1]), .app_rdy(app_rdy[1]),
    .app_wdf_data(app_wdf_data[1]), .app_wdf_mask(app_wdf_mask[1]),
    .app_wdf_wren(app_wdf_wren[1]), .app_wdf_end(app_wdf_end[1]),
    .app_wdf_rdy(app_wdf_rdy[1]), .app_rd_data(app_rd_data[1]),
    .app_rd_data_valid(app_rd_data_valid[1]));

  // the MIGs: self-refresh, refresh and ZQ requests off (the controller's own schedule); the
  // read-data end flag, the ECC error flags and the acknowledges are not used
  mig_ddr3_ch0 mig_0 (
    .ddr3_dq(DDR3_0_dq), .ddr3_dqs_p(DDR3_0_dqs_p), .ddr3_dqs_n(DDR3_0_dqs_n),
    .ddr3_addr(DDR3_0_addr), .ddr3_ba(DDR3_0_ba), .ddr3_ras_n(DDR3_0_ras_n),
    .ddr3_cas_n(DDR3_0_cas_n), .ddr3_we_n(DDR3_0_we_n), .ddr3_reset_n(DDR3_0_reset_n),
    .ddr3_ck_p(DDR3_0_ck_p), .ddr3_ck_n(DDR3_0_ck_n), .ddr3_cke(DDR3_0_cke),
    .ddr3_cs_n(DDR3_0_cs_n), .ddr3_odt(DDR3_0_odt),
    .app_addr(app_addr[0]), .app_cmd(app_cmd[0]), .app_en(app_en[0]),
    .app_wdf_data(app_wdf_data[0]), .app_wdf_end(app_wdf_end[0]), .app_wdf_mask(app_wdf_mask[0]),
    .app_wdf_wren(app_wdf_wren[0]), .app_rd_data(app_rd_data[0]),
    .app_rd_data_valid(app_rd_data_valid[0]), .app_rdy(app_rdy[0]), .app_wdf_rdy(app_wdf_rdy[0]),
    .app_sr_req(1'b0), .app_ref_req(1'b0), .app_zq_req(1'b0),
    .ui_clk(ui_clk[0]), .ui_clk_sync_rst(ui_rst[0]), .init_calib_complete(calib[0]),
    .sys_clk_i(sys_clk), .clk_ref_i(ref_clk), .device_temp_i(temp), .sys_rst(locked));
  mig_ddr3_ch1 mig_1 (
    .ddr3_dq(DDR3_1_dq), .ddr3_dqs_p(DDR3_1_dqs_p), .ddr3_dqs_n(DDR3_1_dqs_n),
    .ddr3_addr(DDR3_1_addr), .ddr3_ba(DDR3_1_ba), .ddr3_ras_n(DDR3_1_ras_n),
    .ddr3_cas_n(DDR3_1_cas_n), .ddr3_we_n(DDR3_1_we_n), .ddr3_reset_n(DDR3_1_reset_n),
    .ddr3_ck_p(DDR3_1_ck_p), .ddr3_ck_n(DDR3_1_ck_n), .ddr3_cke(DDR3_1_cke),
    .ddr3_cs_n(DDR3_1_cs_n), .ddr3_odt(DDR3_1_odt),
    .app_addr(app_addr[1]), .app_cmd(app_cmd[1]), .app_en(app_en[1]),
    .app_wdf_data(app_wdf_data[1]), .app_wdf_end(app_wdf_end[1]), .app_wdf_mask(app_wdf_mask[1]),
    .app_wdf_wren(app_wdf_wren[1]), .app_rd_data(app_rd_data[1]),
    .app_rd_data_valid(app_rd_data_valid[1]), .app_rdy(app_rdy[1]), .app_wdf_rdy(app_wdf_rdy[1]),
    .app_sr_req(1'b0), .app_ref_req(1'b0), .app_zq_req(1'b0),
    .ui_clk(ui_clk[1]), .ui_clk_sync_rst(ui_rst[1]), .init_calib_complete(calib[1]),
    .sys_clk_i(sys_clk), .clk_ref_i(ref_clk), .device_temp_i(temp), .sys_rst(locked));
endmodule
