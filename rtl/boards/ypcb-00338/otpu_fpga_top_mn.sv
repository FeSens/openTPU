// FPGA top for the Inspur YPCB-00338 (xc7k480t-ffg1156-2) with the MIGs' native interface
// (create_project.tcl MEM=mig_native; otpu_fpga_top.sv is the AXI MIG build): the block design
// otpu_bd (PCIe XDMA, the two DDR3 MIGs with their native user interface, clocks, the XADC, the
// control interconnect; boards/ypcb-00338/vivado/bd_native.tcl), one otpu_mig_native per channel
// (otpu_mem_ch's controller port onto the MIG's app_* interface, in its ui_clk; partial beats as
// wr_bytes, the ECC controller's read-modify-write) and otpu_native_sys: the accelerator on the
// channels' native ports (otpu_board with MEM_NATIVE: otpu_native_dram) and XDMA's DMA master,
// meeting in one otpu_mem_ch per channel (RMW = 0). The MIGs calibrate themselves, as in the
// AXI build (STATUS CALIB0/1 from init_calib_complete; CAPS bit27 clear).
//
// Clocks: core_clk and the MIGs' system and reference clocks from the block design's MMCM on the
// 50 MHz pin (as bd.tcl); each MIG's ui_clk (4:1 of its memory clock: 133.33 MHz at DDR3-1066)
// is its channel's controller clock; xdma_aclk (125 MHz) is XDMA's. otpu_mem_ch crosses between
// them (constraints/otpu_mem_ch.tcl, otpu_top_native.tcl: no asynchronous clock groups).
// Resets: core_rst from the block design (MMCM lock, PCIe PERST#), xrst from XDMA's axi_aresetn,
// each channel's controller side from its MIG's ui_clk_sync_rst.
// Port names follow the block design wrapper (DDR3_<c>_<signal>) as otpu_fpga_top.sv's, so the
// MIG constraints and constraints/otpu_top.xdc, otpu_ddr3_pins.xdc apply unchanged.
module otpu_fpga_top_mn #(
  parameter int MCOLS = 2,                  // MXU columns (activation rows per weight chunk)
  parameter int ACT_ROWS = MCOLS,           // ACT RAM rows (> MCOLS: MM replay, one weight pass)
  parameter int VPU_CL = 2,                 // VPU lanes with the composite functions (exp2, ...)
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
  // ---- the MIGs' native interfaces, channel c in its ui_clk[c]
  logic [1:0]        ui_clk, ui_rst;
  logic [1:0][28:0]  app_addr;
  logic [1:0][2:0]   app_cmd;
  logic [1:0]        app_en, app_rdy, app_wdf_wren, app_wdf_end, app_wdf_rdy, app_rd_data_valid;
  logic [1:0][511:0] app_wdf_data, app_rd_data;
  logic [1:0][63:0]  app_wdf_mask;

  otpu_bd_wrapper u_bd (
    .sys_clk_50(SYS_CLK),
    .pcie_refclk_clk_p, .pcie_refclk_clk_n, .pcie_perstn,
    .pcie_mgt_rxp, .pcie_mgt_rxn, .pcie_mgt_txp, .pcie_mgt_txn,
    .pcie_link_up, .core_clk, .core_rstn, .xdma_aclk, .xdma_aresetn, .calib, .device_temp,
    .DDR3_0_dq, .DDR3_0_dqs_p, .DDR3_0_dqs_n, .DDR3_0_addr, .DDR3_0_ba, .DDR3_0_ras_n,
    .DDR3_0_cas_n, .DDR3_0_we_n, .DDR3_0_reset_n, .DDR3_0_ck_p, .DDR3_0_ck_n, .DDR3_0_cke,
    .DDR3_0_cs_n, .DDR3_0_odt,
    .DDR3_1_dq, .DDR3_1_dqs_p, .DDR3_1_dqs_n, .DDR3_1_addr, .DDR3_1_ba, .DDR3_1_ras_n,
    .DDR3_1_cas_n, .DDR3_1_we_n, .DDR3_1_reset_n, .DDR3_1_ck_p, .DDR3_1_ck_n, .DDR3_1_cke,
    .DDR3_1_cs_n, .DDR3_1_odt,
    .mig0_ui_clk(ui_clk[0]), .mig0_ui_clk_sync_rst(ui_rst[0]),
    .mig0_app_addr(app_addr[0]), .mig0_app_cmd(app_cmd[0]), .mig0_app_en(app_en[0]),
    .mig0_app_rdy(app_rdy[0]), .mig0_app_wdf_data(app_wdf_data[0]),
    .mig0_app_wdf_mask(app_wdf_mask[0]), .mig0_app_wdf_wren(app_wdf_wren[0]),
    .mig0_app_wdf_end(app_wdf_end[0]), .mig0_app_wdf_rdy(app_wdf_rdy[0]),
    .mig0_app_rd_data(app_rd_data[0]), .mig0_app_rd_data_valid(app_rd_data_valid[0]),
    .mig1_ui_clk(ui_clk[1]), .mig1_ui_clk_sync_rst(ui_rst[1]),
    .mig1_app_addr(app_addr[1]), .mig1_app_cmd(app_cmd[1]), .mig1_app_en(app_en[1]),
    .mig1_app_rdy(app_rdy[1]), .mig1_app_wdf_data(app_wdf_data[1]),
    .mig1_app_wdf_mask(app_wdf_mask[1]), .mig1_app_wdf_wren(app_wdf_wren[1]),
    .mig1_app_wdf_end(app_wdf_end[1]), .mig1_app_wdf_rdy(app_wdf_rdy[1]),
    .mig1_app_rd_data(app_rd_data[1]), .mig1_app_rd_data_valid(app_rd_data_valid[1]),
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
  for (genvar c = 0; c < 2; c++) begin : g_mig
    otpu_mig_native u_mn (
      .c_cmd_valid(c_cmd_valid[c]), .c_cmd_ready(c_cmd_ready[c]), .c_cmd_we(c_cmd_we[c]),
      .c_cmd_addr(c_cmd_addr[c]), .c_cmd_partial(c_cmd_partial[c]),
      .c_wdata_valid(c_wdata_valid[c]), .c_wdata_ready(c_wdata_ready[c]),
      .c_wdata_data(c_wdata_data[c]), .c_wdata_we(c_wdata_we[c]),
      .c_rdata_valid(c_rdata_valid[c]), .c_rdata_data(c_rdata_data[c]),
      .app_addr(app_addr[c]), .app_cmd(app_cmd[c]), .app_en(app_en[c]), .app_rdy(app_rdy[c]),
      .app_wdf_data(app_wdf_data[c]), .app_wdf_mask(app_wdf_mask[c]),
      .app_wdf_wren(app_wdf_wren[c]), .app_wdf_end(app_wdf_end[c]),
      .app_wdf_rdy(app_wdf_rdy[c]), .app_rd_data(app_rd_data[c]),
      .app_rd_data_valid(app_rd_data_valid[c]));
  end

  // I2C: each line released (high-Z, pulled up) unless its I2C_CTRL bit drives it low
  logic [3:0] i2c_lo, i2c_lvl;
  IOBUF u_iob_scl0 (.IO(lm73_scl), .I(1'b0), .T(!i2c_lo[0]), .O(i2c_lvl[0]));
  IOBUF u_iob_sda0 (.IO(lm73_sda), .I(1'b0), .T(!i2c_lo[1]), .O(i2c_lvl[1]));
  IOBUF u_iob_scl1 (.IO(smb_scl),  .I(1'b0), .T(!i2c_lo[2]), .O(i2c_lvl[2]));
  IOBUF u_iob_sda1 (.IO(smb_sda),  .I(1'b0), .T(!i2c_lo[3]), .O(i2c_lvl[3]));

  // ---- the accelerator and XDMA on the channels
  logic [2:0] board_led;
  otpu_native_sys #(.MCOLS(MCOLS), .ACT_ROWS(ACT_ROWS), .VPU_CL(VPU_CL), .LANES(LANES),
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
