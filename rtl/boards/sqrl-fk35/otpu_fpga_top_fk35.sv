// FPGA top for the SQRL FK35 (xcvu35p-fsvh2104-2-e: two SLRs, 8 GB HBM2; docs/sqrl-fk35.md): the block design
// otpu_bd (XDMA, the clocks, the SYSMON, the control interconnect and XDMA's DMA split onto the channels:
// boards/sqrl-fk35/vivado/bd.tcl), the HBM IP (otpu_hbm: hbm_0 with its 32 AXI ports as arrays), one otpu_hbm_port
// per channel and otpu_fk35_sys (the accelerator at S = 2 and XDMA's DMA, meeting in one otpu_mem_ch per channel;
// the BCU1525's otpu_bcu_sys is its model). The FK33's top (rtl/boards/sqrl-fk33) is this one at S = 1.
//
// Channels (DUAL, targets/sqrl-fk35.toml): board channel j = 4s + c (slice s, its channel c) is HBM bytes
// j x 1 GiB .. + 1 GiB, pseudo-channels 4j .. 4j + 3, the four of mini-switch j: slice 0 on the first stack, slice 1
// on the second. Its otpu_hbm_port (PCB 2) has NS = 2 AXI ports per controller port: port a = p + 2k (controller
// port p, sub-port k) on the IP's AXI_(4j + a), so sub-port k's two pseudo-channels are those of its own two ports and
// every access stays in the channel's mini-switch; all 32 AXI ports are used. The HBM's global switches are on (any
// port reaches all 8 GB); each slice addresses its 4 GiB (otpu_native_dram2: 1 GiB of each of its channels).
// Without DUAL (DUAL = 0 here) the four channels of the two-channel slices: channel j = 2s + c at j x 2 GiB, PCB 3,
// AXI_(8j + 4k + p).
//
// Clocks (bd.tcl): everything from XDMA's axi_aclk (250 MHz, the card has no other clock input): core_clk, uclk
// (250 MHz: otpu_mem_ch's controller side, the adapters and the HBM AXI ports) and clk100 (the HBM's reference
// and APB clock). A PCIe link reset restarts the HBM too. otpu_mem_ch crosses between core_clk, xdma_aclk and uclk.
// Resets: core_rst from the block design (MMCM lock, axi_aresetn), xrst from XDMA's axi_aresetn; the HBM ports
// and the adapters stay in reset until both stacks have calibrated (apb_complete_0/1, from the IP's APB clock).
// The parameters are the target's (targets/sqrl-fk35.toml: opentpu.target rtl_params; run_vivado.sh passes them).
module otpu_fpga_top_fk35 #(
  parameter int S          = 2,
  parameter int D          = 128,
  parameter int MCOLS      = 4,
  parameter int ACT_ROWS   = MCOLS,
  parameter int ACT_BLOCKS = 128,
  parameter int TMEM_WORDS = 1 << 16,
  parameter int IMEM_WORDS = 1 << 15,
  parameter int FIFO_DEPTH = 1024,
  parameter int LANES      = 8,
  parameter int WIN        = 16,
  parameter int RPB        = 8 * LANES,
  parameter int WPB        = 1,
  parameter int TMEM_IMPL  = 0,
  parameter int MXU_IMPL   = 2,
  parameter int MXU_CL     = 16,
  parameter int VPU_CL     = 2,
  parameter int ULANES     = 8,
  parameter int TRACE_DEPTH = 16384,
  parameter int PQ_WIN     = 1024,
  parameter bit CHASH      = 1'b1,
  parameter bit DSTEP      = 1'b1,
  parameter int CORE_KHZ   = 250000,        // core_clk as the block design makes it
  parameter logic [31:0] BUILD_ID = 32'h0,  // the git commit (BUILD_ID register)
  parameter int DDR_MTS    = 1800,          // HBM2 at 900 MHz (DDR_MTS register)
  parameter bit SEQ_WC     = 1'b0,
  parameter int PLATFORM   = 4,             // targets/sqrl-fk35.toml (CFG_PLATFORM)
  parameter logic [31:0] ARCH_HASH = 32'h0, // its arch_hash (CFG_HASH; run_vivado.sh)
  parameter int PCIE_LANES = 8,
  parameter int QMEM       = 0,             // the memory adapters' queues: 0 block RAM, 1 LUT RAM, 2 and
                                            // the B FIFOs in UltraRAM (otpu_native_dram; OTPU_QMEM)
  parameter bit MEMX       = 1'b1,          // the adapters in SLR0, the crossing on the slices' ports
                                            // (otpu_board MEMX; OTPU_MEMX=0: the native ports cross)
  parameter bit QUAD       = 1'b0           // the target's QUAD: 4-bit MM PAIR four blocks an advance
                                            // (otpu_mxu; run_vivado.sh OTPU_QUAD from targets/sqrl-fk35.toml)
) (
  output logic [6:0]  led,                  // 0..3 green, 4..6 the RGB LED (high: on)
  output logic        pcie_clkreq,
  input  logic        pcie_refclk_p,
  input  logic        pcie_refclk_n,
  input  logic        pcie_perst_n,
  input  logic [PCIE_LANES-1:0] pcie_mgt_rxp,
  input  logic [PCIE_LANES-1:0] pcie_mgt_rxn,
  output logic [PCIE_LANES-1:0] pcie_mgt_txp,
  output logic [PCIE_LANES-1:0] pcie_mgt_txn
);
  // DUAL (the target's): each slice on four HBM channels, its port B in chunk pairs; XDMA's DMA split onto the eight
  // channels (bd.tcl NCH 8), each in a 2 GiB window of which the slice uses the first 1 GiB
  localparam bit DUAL = 1'b1;
  localparam int NC  = (DUAL ? 4 : 2) * S;
  localparam int NP  = 32;                  // the HBM IP's AXI ports
  localparam int CHW = DUAL ? 30 : 31;      // log2 of a channel's HBM bytes
  localparam int PCB = DUAL ? 2 : 3;        // its pseudo-channels' log2
`ifndef SYNTHESIS
  initial if (S != 2) $fatal(1, "otpu_fpga_top_fk35: S = 2 (eight HBM channels, DUAL; or four)");
`endif
  logic        core_clk, core_rstn, uclk, urstn, clk100, clk100_rstn, xdma_aclk, xdma_aresetn, pcie_link_up;
  logic [9:0]  device_temp;                 // SYSMONE4 die-temperature code (core_clk)
  logic [31:0] ctl_awaddr, ctl_araddr, ctl_wdata, ctl_rdata;
  logic [3:0]  ctl_wstrb;
  logic [1:0]  ctl_bresp, ctl_rresp;
  logic ctl_awvalid, ctl_awready, ctl_wvalid, ctl_wready, ctl_bvalid, ctl_bready;
  logic ctl_arvalid, ctl_arready, ctl_rvalid, ctl_rready;
  logic [NC-1:0][33:0]  dm_awaddr, dm_araddr;
  logic [NC-1:0][7:0]   dm_awlen, dm_arlen;
  logic [NC-1:0][127:0] dm_wdata, dm_rdata;
  logic [NC-1:0][15:0]  dm_wstrb;
  logic [NC-1:0][1:0]   dm_bresp, dm_rresp;
  logic [NC-1:0] dm_awvalid, dm_awready, dm_wvalid, dm_wready, dm_wlast, dm_bvalid, dm_bready;
  logic [NC-1:0] dm_arvalid, dm_arready, dm_rvalid, dm_rready, dm_rlast;
`define BD_DMA(J) \
    .M_AXI_DMA``J``_awaddr(dm_awaddr[J]), .M_AXI_DMA``J``_awlen(dm_awlen[J]), \
    .M_AXI_DMA``J``_awvalid(dm_awvalid[J]), .M_AXI_DMA``J``_awready(dm_awready[J]), \
    .M_AXI_DMA``J``_wdata(dm_wdata[J]), .M_AXI_DMA``J``_wstrb(dm_wstrb[J]), \
    .M_AXI_DMA``J``_wlast(dm_wlast[J]), .M_AXI_DMA``J``_wvalid(dm_wvalid[J]), \
    .M_AXI_DMA``J``_wready(dm_wready[J]), .M_AXI_DMA``J``_bresp(dm_bresp[J]), \
    .M_AXI_DMA``J``_bvalid(dm_bvalid[J]), .M_AXI_DMA``J``_bready(dm_bready[J]), \
    .M_AXI_DMA``J``_araddr(dm_araddr[J]), .M_AXI_DMA``J``_arlen(dm_arlen[J]), \
    .M_AXI_DMA``J``_arvalid(dm_arvalid[J]), .M_AXI_DMA``J``_arready(dm_arready[J]), \
    .M_AXI_DMA``J``_rdata(dm_rdata[J]), .M_AXI_DMA``J``_rresp(dm_rresp[J]), \
    .M_AXI_DMA``J``_rlast(dm_rlast[J]), .M_AXI_DMA``J``_rvalid(dm_rvalid[J]), \
    .M_AXI_DMA``J``_rready(dm_rready[J])
  otpu_bd_wrapper u_bd (
    .pcie_refclk_clk_p(pcie_refclk_p), .pcie_refclk_clk_n(pcie_refclk_n),
    .pcie_perstn(pcie_perst_n),
    .pcie_mgt_rxp, .pcie_mgt_rxn, .pcie_mgt_txp, .pcie_mgt_txn,
    .pcie_link_up, .core_clk, .core_rstn, .uclk, .urstn, .clk100, .clk100_rstn,
    .xdma_aclk, .xdma_aresetn, .device_temp,
    .M_AXI_CTL_awaddr(ctl_awaddr), .M_AXI_CTL_awvalid(ctl_awvalid), .M_AXI_CTL_awready(ctl_awready),
    .M_AXI_CTL_wdata(ctl_wdata), .M_AXI_CTL_wstrb(ctl_wstrb), .M_AXI_CTL_wvalid(ctl_wvalid),
    .M_AXI_CTL_wready(ctl_wready), .M_AXI_CTL_bresp(ctl_bresp), .M_AXI_CTL_bvalid(ctl_bvalid),
    .M_AXI_CTL_bready(ctl_bready), .M_AXI_CTL_araddr(ctl_araddr), .M_AXI_CTL_arvalid(ctl_arvalid),
    .M_AXI_CTL_arready(ctl_arready), .M_AXI_CTL_rdata(ctl_rdata), .M_AXI_CTL_rresp(ctl_rresp),
    .M_AXI_CTL_rvalid(ctl_rvalid), .M_AXI_CTL_rready(ctl_rready),
    `BD_DMA(0), `BD_DMA(1), `BD_DMA(2), `BD_DMA(3), `BD_DMA(4), `BD_DMA(5), `BD_DMA(6), `BD_DMA(7));
`undef BD_DMA
  assign pcie_clkreq = 1'b0;
  logic core_rst, xrst;
  always_ff @(posedge core_clk) core_rst <= !core_rstn;
  always_ff @(posedge xdma_aclk) xrst <= !xdma_aresetn;
  (* ASYNC_REG = "TRUE" *) logic lrs1 = 1'b1, lrs2 = 1'b1;
  always_ff @(posedge core_clk) begin lrs1 <= xrst; lrs2 <= lrs1; end
  // ---- HBM: both stacks calibrated (apb_complete, from the APB clock) releases the AXI ports' side
  logic [1:0] hbm_calib, hbm_cattrip;
  (* ASYNC_REG = "TRUE" *) logic [1:0] hcal_u1 = '0, hcal_u2 = '0;
  always_ff @(posedge uclk) begin hcal_u1 <= hbm_calib; hcal_u2 <= hcal_u1; end
  wire u_rst = !urstn || !(&hcal_u2);
  logic [NP-1:0][32:0]  h_awaddr, h_araddr;
  logic [NP-1:0][5:0]   h_awid, h_arid, h_bid, h_rid;
  logic [NP-1:0][3:0]   h_awlen, h_arlen;
  logic [NP-1:0][2:0]   h_awsize, h_arsize;
  logic [NP-1:0][1:0]   h_awburst, h_arburst, h_bresp, h_rresp;
  logic [NP-1:0]        h_awvalid, h_awready, h_wvalid, h_wready, h_wlast;
  logic [NP-1:0]        h_bvalid, h_bready, h_arvalid, h_arready, h_rvalid, h_rready, h_rlast;
  logic [NP-1:0][255:0] h_wdata, h_rdata;
  logic [NP-1:0][31:0]  h_wstrb, h_wpar;
  otpu_hbm u_hbm (
    .hbm_ref(clk100), .aclk(uclk), .aresetn(urstn), .pclk(clk100), .presetn(clk100_rstn),
    .h_awaddr, .h_awburst, .h_awid, .h_awlen, .h_awsize, .h_awvalid, .h_awready,
    .h_wdata, .h_wlast, .h_wstrb, .h_wpar, .h_wvalid, .h_wready, .h_bready, .h_bid, .h_bresp, .h_bvalid,
    .h_araddr, .h_arburst, .h_arid, .h_arlen, .h_arsize, .h_arvalid, .h_arready,
    .h_rready, .h_rdata, .h_rid, .h_rlast, .h_rresp, .h_rvalid,
    .calib(hbm_calib), .cattrip(hbm_cattrip), .temp0(), .temp1());
  // ---- the accelerator and XDMA on the channels (sc_dma decoded address bits 33:31: the channel's 2 GiB window)
  logic [NC-1:0][30:0] dm_awaddr_c, dm_araddr_c;
  for (genvar j = 0; j < NC; j++) begin : g_dma
    assign dm_awaddr_c[j] = dm_awaddr[j][30:0];
    assign dm_araddr_c[j] = dm_araddr[j][30:0];
  end
  logic [NC-1:0][1:0]        c_cmd_valid, c_cmd_ready, c_cmd_we, c_wdata_valid, c_wdata_ready, c_rdata_valid;
  logic [NC-1:0][1:0][24:0]  c_cmd_addr;
  logic [NC-1:0][1:0][511:0] c_wdata_data, c_rdata_data;
  logic [NC-1:0][1:0][63:0]  c_wdata_we;
  logic [2:0] board_led;
  // calib: both stacks up and no catastrophic temperature trip (otpu_board refuses RUN and LOAD without it and
  // synchronizes it)
  otpu_fk35_sys #(.S(S), .D(D), .MCOLS(MCOLS), .ACT_ROWS(ACT_ROWS), .ACT_BLOCKS(ACT_BLOCKS),
                 .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS), .FIFO_DEPTH(FIFO_DEPTH),
                 .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB), .MXU_IMPL(MXU_IMPL),
                 .MXU_CL(MXU_CL), .VPU_CL(VPU_CL), .ULANES(ULANES), .TRACE_DEPTH(TRACE_DEPTH),
                 .PQ_WIN(PQ_WIN), .CHASH(CHASH), .DSTEP(DSTEP), .CORE_KHZ(CORE_KHZ),
                 .BUILD_ID(BUILD_ID), .SEQ_WC(SEQ_WC), .DDR_MTS(DDR_MTS), .PLATFORM(PLATFORM),
                 .ARCH_HASH(ARCH_HASH), .TMEM_IMPL(TMEM_IMPL), .DUAL(DUAL), .QUAD(QUAD), .NQMEM(QMEM), .MEMX(MEMX)) u_sys (
    .clk(core_clk), .clk_f(1'b0), .rst(core_rst), .ctl_mrst(lrs2), .xclk(xdma_aclk), .xrst,
    .calib({NC{&hbm_calib && !(|hbm_cattrip)}}), .ded('0), .temp({device_temp, 2'b00}),
    .led(board_led), .i2c_lo(), .i2c_pin(5'h1F),
    .s_ctl_awaddr(ctl_awaddr[11:0]), .s_ctl_awvalid(ctl_awvalid), .s_ctl_awready(ctl_awready),
    .s_ctl_wdata(ctl_wdata), .s_ctl_wstrb(ctl_wstrb), .s_ctl_wvalid(ctl_wvalid),
    .s_ctl_wready(ctl_wready), .s_ctl_bresp(ctl_bresp), .s_ctl_bvalid(ctl_bvalid),
    .s_ctl_bready(ctl_bready), .s_ctl_araddr(ctl_araddr[11:0]), .s_ctl_arvalid(ctl_arvalid),
    .s_ctl_arready(ctl_arready), .s_ctl_rdata(ctl_rdata), .s_ctl_rresp(ctl_rresp),
    .s_ctl_rvalid(ctl_rvalid), .s_ctl_rready(ctl_rready),
    .x_awvalid(dm_awvalid), .x_awready(dm_awready), .x_awaddr(dm_awaddr_c), .x_awlen(dm_awlen),
    .x_wvalid(dm_wvalid), .x_wready(dm_wready), .x_wdata(dm_wdata), .x_wstrb(dm_wstrb),
    .x_wlast(dm_wlast), .x_bvalid(dm_bvalid), .x_bready(dm_bready), .x_bresp(dm_bresp),
    .x_arvalid(dm_arvalid), .x_arready(dm_arready), .x_araddr(dm_araddr_c), .x_arlen(dm_arlen),
    .x_rvalid(dm_rvalid), .x_rready(dm_rready), .x_rdata(dm_rdata), .x_rresp(dm_rresp),
    .x_rlast(dm_rlast),
    .uclk({NC{uclk}}), .urst({NC{u_rst}}),
    .c_cmd_valid, .c_cmd_ready, .c_cmd_we, .c_cmd_addr, .c_wdata_valid, .c_wdata_ready,
    .c_wdata_data, .c_wdata_we, .c_rdata_valid, .c_rdata_data);
  // ---- the HBM adapters: channel j at j x 1 GiB, its AXI port a on the IP's AXI_(4j + a) (without DUAL: j x 2 GiB,
  // AXI_(8j + 4k + p))
  localparam logic [NP-1:0] HP_USED = DUAL ? 32'hFFFF_FFFF : 32'h3333_3333;
  for (genvar j = 0; j < NC; j++) begin : g_hbm
    logic [3:0][32:0]  awaddr, araddr;
    logic [3:0][5:0]   awid, arid, bid, rid;
    logic [3:0][3:0]   awlen, arlen;
    logic [3:0][2:0]   awsize, arsize;
    logic [3:0][1:0]   awburst, arburst, bresp, rresp;
    logic [3:0]        awvalid, awready, wvalid, wready, wlast, bvalid, bready;
    logic [3:0]        arvalid, arready, rvalid, rready, rlast;
    logic [3:0][255:0] wdata, rdata;
    logic [3:0][31:0]  wstrb, wpar;
    otpu_hbm_port #(.CH_BASE(33'(j) << CHW), .PCB(PCB), .NS(2), .RD(64)) u_port (
      .uclk, .urst(u_rst),
      .c_cmd_valid(c_cmd_valid[j]), .c_cmd_ready(c_cmd_ready[j]), .c_cmd_we(c_cmd_we[j]),
      .c_cmd_addr(c_cmd_addr[j]), .c_wdata_valid(c_wdata_valid[j]), .c_wdata_ready(c_wdata_ready[j]),
      .c_wdata_data(c_wdata_data[j]), .c_wdata_we(c_wdata_we[j]),
      .c_rdata_valid(c_rdata_valid[j]), .c_rdata_data(c_rdata_data[j]),
      .m_awaddr(awaddr), .m_awid(awid), .m_awlen(awlen), .m_awsize(awsize), .m_awburst(awburst),
      .m_awvalid(awvalid), .m_awready(awready), .m_wdata(wdata), .m_wstrb(wstrb), .m_wlast(wlast),
      .m_wvalid(wvalid), .m_wready(wready), .m_wid(), .m_wdata_parity(wpar),
      .m_bid(bid), .m_bresp(bresp), .m_bvalid(bvalid), .m_bready(bready),
      .m_araddr(araddr), .m_arid(arid), .m_arlen(arlen), .m_arsize(arsize), .m_arburst(arburst),
      .m_arvalid(arvalid), .m_arready(arready), .m_rid(rid), .m_rdata(rdata), .m_rresp(rresp),
      .m_rlast(rlast), .m_rvalid(rvalid), .m_rready(rready),
      .st_rcmd(), .st_wcmd(), .st_rout(), .st_wout(), .st_dep(), .st_busy(), .st_err_cnt(), .st_err(),
      .st_perr());
    for (genvar a = 0; a < 4; a++) begin : g_ax
      localparam int H = DUAL ? 4 * j + a : 8 * j + 4 * (a / 2) + a % 2;
      assign h_awaddr[H] = awaddr[a];   assign h_awid[H] = awid[a];     assign h_awlen[H] = awlen[a];
      assign h_awsize[H] = awsize[a];   assign h_awburst[H] = awburst[a]; assign h_awvalid[H] = awvalid[a];
      assign awready[a] = h_awready[H];
      assign h_wdata[H] = wdata[a];     assign h_wstrb[H] = wstrb[a];   assign h_wlast[H] = wlast[a];
      assign h_wvalid[H] = wvalid[a];   assign h_wpar[H] = wpar[a];     assign wready[a] = h_wready[H];
      assign bid[a] = h_bid[H];         assign bresp[a] = h_bresp[H];   assign bvalid[a] = h_bvalid[H];
      assign h_bready[H] = bready[a];
      assign h_araddr[H] = araddr[a];   assign h_arid[H] = arid[a];     assign h_arlen[H] = arlen[a];
      assign h_arsize[H] = arsize[a];   assign h_arburst[H] = arburst[a]; assign h_arvalid[H] = arvalid[a];
      assign arready[a] = h_arready[H];
      assign rid[a] = h_rid[H];         assign rdata[a] = h_rdata[H];   assign rresp[a] = h_rresp[H];
      assign rlast[a] = h_rlast[H];     assign rvalid[a] = h_rvalid[H]; assign h_rready[H] = rready[a];
    end
  end
  for (genvar p = 0; p < NP; p++) begin : g_hbm_off
    if (!HP_USED[p]) begin : g_off
      assign h_awaddr[p] = '0;  assign h_awid[p] = '0;   assign h_awlen[p] = '0;
      assign h_awsize[p] = '0;  assign h_awburst[p] = '0; assign h_awvalid[p] = 1'b0;
      assign h_wdata[p] = '0;   assign h_wstrb[p] = '0;   assign h_wlast[p] = 1'b0;
      assign h_wvalid[p] = 1'b0; assign h_wpar[p] = '0;   assign h_bready[p] = 1'b1;
      assign h_araddr[p] = '0;  assign h_arid[p] = '0;   assign h_arlen[p] = '0;
      assign h_arsize[p] = '0;  assign h_arburst[p] = '0; assign h_arvalid[p] = 1'b0;
      assign h_rready[p] = 1'b1;
    end
  end
  // LEDs: green 0 heartbeat, green 1 PCIe link up and both stacks calibrated, green 2 the accelerator runs or
  // halted cleanly; green 3 and the RGB LED off
  assign led = {4'b0000, board_led[2] | board_led[1], pcie_link_up & (&hbm_calib), board_led[0]};
endmodule
