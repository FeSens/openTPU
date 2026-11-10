// FPGA top of the BCU1525's memory-test image ("image M", docs/bcu1525.md section 14): the four
// DDR4 UDIMMs' LiteDRAM cores (otpu_ld4, boards/bcu1525/litedram_udimm/, as otpu_fpga_top_bcu's)
// and their CSRs, the board management controller's UART bridge (otpu_bmc_uart, its watch reading
// VCCINT's regulator for the governor) and the thermal governor (otpu_bcu_gov), reached over JTAG: the block design otpu_bdm (vivado/bd_mem.tcl) is a
// JTAG-to-AXI master, the control clock, the SYSMON and the interconnect, at the full image's BAR0
// offsets (0x10000 the governor, 0x20000 the BMC's bridge, 0x30000 the SYSMON, 0x40000 + d x
// 0x10000 DIMM C<d>'s core), so opentpu/host's register code is the same. No PCIe, no accelerator:
// the cores' native ports are idle and each core's own BIST (its CSRs) tests its DIMM. The card has
// no fan (2026-10-21: 112 C), so every core starts held in reset and runs only while the governor
// grants it (CTRL / CMD GO): its rst stops the core's MMCM, its clocks and its DIMM's.
//
// A DIMM the governor turns off: its core's rst (otpu_ld4's 8-flip-flop chain on its 300 MHz input,
// into its MMCM's RST); its CSR port's otpu_axil_iso slave reset and the core's ctl_rst, and
// otpu_axil_off in front, for the time it is off and UP_CYC cycles after it is turned on (the MMCM's
// lock, the core's IDELAYCTRL reset and sys reset: until then the core's CSR crossing has no live
// side; LiteDRAM's sys_rst stays low while its clocks are stopped, so it cannot say so itself). An
// access to it meanwhile is answered SLVERR at once.
//
// The UART (BF18 / BB20, the board's USB UART) is DIMM C<UART_SEL>'s core's (the governor's CTRL;
// C1's after configuration): its BIOS's calibration log.
module otpu_fpga_top_bcu_mem #(
  parameter int CORE_KHZ = 100000,          // core_clk (bd_mem.tcl)
  parameter int UP_CYC   = 1 << 17          // a DIMM's CSRs answered here after turning on (1.3 ms)
) (
  input  logic        clk300_0_p,
  input  logic        clk300_0_n,
  input  logic        clk300_1_p,
  input  logic        clk300_1_n,
  input  logic        clk300_2_p,
  input  logic        clk300_2_n,
  input  logic        clk300_3_p,
  input  logic        clk300_3_n,
  output logic [2:0]  led,
  input  logic        uart_rx,
  output logic        uart_tx,
  output logic        bmc_tx,
  input  logic        bmc_rx,
  output logic [13:0] ddr4_c0_a,
  output logic        ddr4_c0_we_n,
  output logic        ddr4_c0_cas_n,
  output logic        ddr4_c0_ras_n,
  output logic        ddr4_c0_act_n,
  output logic [1:0]  ddr4_c0_ba,
  output logic [1:0]  ddr4_c0_bg,
  output logic        ddr4_c0_cke,
  output logic        ddr4_c0_odt,
  output logic        ddr4_c0_cs_n,
  output logic        ddr4_c0_reset_n,
  output logic        ddr4_c0_clk_p,
  output logic        ddr4_c0_clk_n,
  output logic        ddr4_c0_par,
  output logic        ddr4_c0_cke1,
  output logic        ddr4_c0_odt1,
  output logic [2:0]  ddr4_c0_cs_hi,
  output logic [7:0]  ddr4_c0_dm,
  inout  wire  [63:0] ddr4_c0_dq,
  inout  wire  [7:0]  ddr4_c0_dqs_p,
  inout  wire  [7:0]  ddr4_c0_dqs_n,
  output logic [13:0] ddr4_c1_a,
  output logic        ddr4_c1_we_n,
  output logic        ddr4_c1_cas_n,
  output logic        ddr4_c1_ras_n,
  output logic        ddr4_c1_act_n,
  output logic [1:0]  ddr4_c1_ba,
  output logic [1:0]  ddr4_c1_bg,
  output logic        ddr4_c1_cke,
  output logic        ddr4_c1_odt,
  output logic        ddr4_c1_cs_n,
  output logic        ddr4_c1_reset_n,
  output logic        ddr4_c1_clk_p,
  output logic        ddr4_c1_clk_n,
  output logic        ddr4_c1_par,
  output logic        ddr4_c1_cke1,
  output logic        ddr4_c1_odt1,
  output logic [2:0]  ddr4_c1_cs_hi,
  output logic [7:0]  ddr4_c1_dm,
  inout  wire  [63:0] ddr4_c1_dq,
  inout  wire  [7:0]  ddr4_c1_dqs_p,
  inout  wire  [7:0]  ddr4_c1_dqs_n,
  output logic [13:0] ddr4_c2_a,
  output logic        ddr4_c2_we_n,
  output logic        ddr4_c2_cas_n,
  output logic        ddr4_c2_ras_n,
  output logic        ddr4_c2_act_n,
  output logic [1:0]  ddr4_c2_ba,
  output logic [1:0]  ddr4_c2_bg,
  output logic        ddr4_c2_cke,
  output logic        ddr4_c2_odt,
  output logic        ddr4_c2_cs_n,
  output logic        ddr4_c2_reset_n,
  output logic        ddr4_c2_clk_p,
  output logic        ddr4_c2_clk_n,
  output logic        ddr4_c2_par,
  output logic        ddr4_c2_cke1,
  output logic        ddr4_c2_odt1,
  output logic [2:0]  ddr4_c2_cs_hi,
  output logic [7:0]  ddr4_c2_dm,
  inout  wire  [63:0] ddr4_c2_dq,
  inout  wire  [7:0]  ddr4_c2_dqs_p,
  inout  wire  [7:0]  ddr4_c2_dqs_n,
  output logic [13:0] ddr4_c3_a,
  output logic        ddr4_c3_we_n,
  output logic        ddr4_c3_cas_n,
  output logic        ddr4_c3_ras_n,
  output logic        ddr4_c3_act_n,
  output logic [1:0]  ddr4_c3_ba,
  output logic [1:0]  ddr4_c3_bg,
  output logic        ddr4_c3_cke,
  output logic        ddr4_c3_odt,
  output logic        ddr4_c3_cs_n,
  output logic        ddr4_c3_reset_n,
  output logic        ddr4_c3_clk_p,
  output logic        ddr4_c3_clk_n,
  output logic        ddr4_c3_par,
  output logic        ddr4_c3_cke1,
  output logic        ddr4_c3_odt1,
  output logic [2:0]  ddr4_c3_cs_hi,
  output logic [7:0]  ddr4_c3_dm,
  inout  wire  [63:0] ddr4_c3_dq,
  inout  wire  [7:0]  ddr4_c3_dqs_p,
  inout  wire  [7:0]  ddr4_c3_dqs_n
);
  // ---- clocks: each DIMM's 300 MHz on one global buffer (C1's also the block design's)
  logic [3:0] c300_ib, c300;
  IBUFDS u_ib300_0 (.I(clk300_0_p), .IB(clk300_0_n), .O(c300_ib[0]));
  IBUFDS u_ib300_1 (.I(clk300_1_p), .IB(clk300_1_n), .O(c300_ib[1]));
  IBUFDS u_ib300_2 (.I(clk300_2_p), .IB(clk300_2_n), .O(c300_ib[2]));
  IBUFDS u_ib300_3 (.I(clk300_3_p), .IB(clk300_3_n), .O(c300_ib[3]));
  for (genvar d = 0; d < 4; d++) begin : g_bufg
    BUFG u_bufg (.I(c300_ib[d]), .O(c300[d]));
  end

  // ---- the block design: JTAG-to-AXI, core_clk, the SYSMON
  logic        core_clk, core_locked, temp_alarm, ot_alarm;
  logic [9:0]  device_temp;
  logic [31:0] gv_awaddr, gv_araddr, gv_wdata, gv_rdata;
  logic [3:0]  gv_wstrb;
  logic [1:0]  gv_bresp, gv_rresp;
  logic gv_awvalid, gv_awready, gv_wvalid, gv_wready, gv_bvalid, gv_bready;
  logic gv_arvalid, gv_arready, gv_rvalid, gv_rready;
  logic [31:0] bmc_awaddr, bmc_araddr, bmc_wdata, bmc_rdata;
  logic [3:0]  bmc_wstrb;
  logic [1:0]  bmc_bresp, bmc_rresp;
  logic bmc_awvalid, bmc_awready, bmc_wvalid, bmc_wready, bmc_bvalid, bmc_bready;
  logic bmc_arvalid, bmc_arready, bmc_rvalid, bmc_rready;
  logic [3:0][31:0] mc_awaddr, mc_araddr, mc_wdata, mc_rdata;
  logic [3:0][3:0]  mc_wstrb;
  logic [3:0][1:0]  mc_bresp, mc_rresp;
  logic [3:0] mc_awvalid, mc_awready, mc_wvalid, mc_wready, mc_bvalid, mc_bready;
  logic [3:0] mc_arvalid, mc_arready, mc_rvalid, mc_rready;
`define BD_LITE(P, S) \
    .P``_awaddr(S``_awaddr), .P``_awvalid(S``_awvalid), .P``_awready(S``_awready), \
    .P``_wdata(S``_wdata), .P``_wstrb(S``_wstrb), .P``_wvalid(S``_wvalid), \
    .P``_wready(S``_wready), .P``_bresp(S``_bresp), .P``_bvalid(S``_bvalid), \
    .P``_bready(S``_bready), .P``_araddr(S``_araddr), .P``_arvalid(S``_arvalid), \
    .P``_arready(S``_arready), .P``_rdata(S``_rdata), .P``_rresp(S``_rresp), \
    .P``_rvalid(S``_rvalid), .P``_rready(S``_rready)
`define BD_MC(D) \
    .M_AXI_MEMCAL``D``_awaddr(mc_awaddr[D]), .M_AXI_MEMCAL``D``_awvalid(mc_awvalid[D]), \
    .M_AXI_MEMCAL``D``_awready(mc_awready[D]), .M_AXI_MEMCAL``D``_wdata(mc_wdata[D]), \
    .M_AXI_MEMCAL``D``_wstrb(mc_wstrb[D]), .M_AXI_MEMCAL``D``_wvalid(mc_wvalid[D]), \
    .M_AXI_MEMCAL``D``_wready(mc_wready[D]), .M_AXI_MEMCAL``D``_bresp(mc_bresp[D]), \
    .M_AXI_MEMCAL``D``_bvalid(mc_bvalid[D]), .M_AXI_MEMCAL``D``_bready(mc_bready[D]), \
    .M_AXI_MEMCAL``D``_araddr(mc_araddr[D]), .M_AXI_MEMCAL``D``_arvalid(mc_arvalid[D]), \
    .M_AXI_MEMCAL``D``_arready(mc_arready[D]), .M_AXI_MEMCAL``D``_rdata(mc_rdata[D]), \
    .M_AXI_MEMCAL``D``_rresp(mc_rresp[D]), .M_AXI_MEMCAL``D``_rvalid(mc_rvalid[D]), \
    .M_AXI_MEMCAL``D``_rready(mc_rready[D])
  otpu_bdm_wrapper u_bd (
    .clk300_in(c300[1]), .core_clk, .core_locked, .core_rstn, .device_temp, .temp_alarm, .ot_alarm,
    `BD_LITE(M_AXI_GOV, gv), `BD_LITE(M_AXI_BMC, bmc),
    `BD_MC(0), `BD_MC(1), `BD_MC(2), `BD_MC(3));
`undef BD_LITE
`undef BD_MC

  // the reset (the block design's AXI side and this top's): from configuration until core_clk's MMCM
  // has been locked for 16 cycles, and at once when it loses lock
  logic [15:0] rsr = '0;
  always_ff @(posedge core_clk) rsr <= core_locked ? {rsr[14:0], 1'b1} : '0;
  wire core_rstn = rsr[15];
  logic core_rst = 1'b1;
  always_ff @(posedge core_clk) core_rst <= !core_rstn;

  // ---- the governor
  logic       acc_on;                       // no accelerator in this image
  logic [3:0] dimm_on;
  logic [1:0] uart_sel;
  logic signed [15:0] reg_hot;              // VCCINT's regulator (the BMC's bridge's watch)
  logic [15:0] reg_age;
  otpu_bcu_gov #(.CLK_HZ(CORE_KHZ * 1000), .HAS_REG(1'b1)) u_gov (
    .clk(core_clk), .rst(core_rst), .temp(device_temp), .temp_alarm, .ot_alarm, .reg_t(reg_hot),
    .reg_age, .acc_on, .dimm_on, .dimm_ok(), .uart_sel, .led,
    .s_awaddr(gv_awaddr[11:0]), .s_awvalid(gv_awvalid), .s_awready(gv_awready),
    .s_wdata(gv_wdata), .s_wstrb(gv_wstrb), .s_wvalid(gv_wvalid), .s_wready(gv_wready),
    .s_bresp(gv_bresp), .s_bvalid(gv_bvalid), .s_bready(gv_bready),
    .s_araddr(gv_araddr[11:0]), .s_arvalid(gv_arvalid), .s_arready(gv_arready),
    .s_rdata(gv_rdata), .s_rresp(gv_rresp), .s_rvalid(gv_rvalid), .s_rready(gv_rready));

  // ---- the BMC's UART bridge
  otpu_bmc_uart #(.CLK_HZ(CORE_KHZ * 1000)) u_bmc (
    .clk(core_clk), .rst(core_rst),
    .s_awaddr(bmc_awaddr[15:0]), .s_awvalid(bmc_awvalid), .s_awready(bmc_awready),
    .s_wdata(bmc_wdata), .s_wstrb(bmc_wstrb), .s_wvalid(bmc_wvalid), .s_wready(bmc_wready),
    .s_bresp(bmc_bresp), .s_bvalid(bmc_bvalid), .s_bready(bmc_bready),
    .s_araddr(bmc_araddr[15:0]), .s_arvalid(bmc_arvalid), .s_arready(bmc_arready),
    .s_rdata(bmc_rdata), .s_rresp(bmc_rresp), .s_rvalid(bmc_rvalid), .s_rready(bmc_rready),
    .tx(bmc_tx), .rx(bmc_rx), .reg_hot, .reg_age);

  // ---- the DDR4 cores' CSR ports: off (dimm_on low, or within UP_CYC of rising), else their sys
  // reset in core_clk (two flip-flops)
  logic [3:0] d_sys_clk, d_sys_rst, d_tx;
  (* ASYNC_REG = "TRUE" *) logic [3:0] ld_crs1 = '1, ld_crs2 = '1;
  always_ff @(posedge core_clk) begin ld_crs1 <= d_sys_rst; ld_crs2 <= ld_crs1; end
  logic [3:0][$clog2(UP_CYC):0] up_cnt;
  logic [3:0] csr_off, csr_rst;
  logic [15:0] lc_awaddr [4], lc_araddr [4], li_awaddr [4], li_araddr [4];
  logic [31:0] lc_wdata [4], lc_rdata [4], li_wdata [4], li_rdata [4];
  logic [3:0]  lc_wstrb [4], li_wstrb [4];
  logic [1:0]  lc_bresp [4], lc_rresp [4], li_bresp [4], li_rresp [4];
  logic [3:0] lc_awvalid, lc_awready, lc_wvalid, lc_wready, lc_bvalid, lc_bready;
  logic [3:0] lc_arvalid, lc_arready, lc_rvalid, lc_rready;
  logic [3:0] li_awvalid, li_awready, li_wvalid, li_wready, li_bvalid, li_bready;
  logic [3:0] li_arvalid, li_arready, li_rvalid, li_rready;
  for (genvar d = 0; d < 4; d++) begin : g_csr
    always_ff @(posedge core_clk) begin
      if (core_rst || !dimm_on[d]) up_cnt[d] <= '0;
      else if (up_cnt[d] != ($clog2(UP_CYC) + 1)'(UP_CYC)) up_cnt[d] <= up_cnt[d] + 1'b1;
      csr_off[d] <= core_rst || !dimm_on[d] || up_cnt[d] != ($clog2(UP_CYC) + 1)'(UP_CYC);
      csr_rst[d] <= csr_off[d] || ld_crs2[d];
    end
    otpu_axil_off #(.AW(16)) u_off (
      .clk(core_clk), .rst(core_rst), .off(csr_off[d]),
      .m_awaddr(mc_awaddr[d][15:0]), .m_awvalid(mc_awvalid[d]), .m_awready(mc_awready[d]),
      .m_wdata(mc_wdata[d]), .m_wstrb(mc_wstrb[d]), .m_wvalid(mc_wvalid[d]),
      .m_wready(mc_wready[d]), .m_bresp(mc_bresp[d]), .m_bvalid(mc_bvalid[d]),
      .m_bready(mc_bready[d]), .m_araddr(mc_araddr[d][15:0]), .m_arvalid(mc_arvalid[d]),
      .m_arready(mc_arready[d]), .m_rdata(mc_rdata[d]), .m_rresp(mc_rresp[d]),
      .m_rvalid(mc_rvalid[d]), .m_rready(mc_rready[d]),
      .s_awaddr(li_awaddr[d]), .s_awvalid(li_awvalid[d]), .s_awready(li_awready[d]),
      .s_wdata(li_wdata[d]), .s_wstrb(li_wstrb[d]), .s_wvalid(li_wvalid[d]),
      .s_wready(li_wready[d]), .s_bresp(li_bresp[d]), .s_bvalid(li_bvalid[d]),
      .s_bready(li_bready[d]), .s_araddr(li_araddr[d]), .s_arvalid(li_arvalid[d]),
      .s_arready(li_arready[d]), .s_rdata(li_rdata[d]), .s_rresp(li_rresp[d]),
      .s_rvalid(li_rvalid[d]), .s_rready(li_rready[d]));
    otpu_axil_iso #(.AW(16)) u_iso (
      .clk(core_clk), .m_rst(core_rst), .s_rst(csr_rst[d] || csr_off[d]),
      .m_awaddr(li_awaddr[d]), .m_awvalid(li_awvalid[d]), .m_awready(li_awready[d]),
      .m_wdata(li_wdata[d]), .m_wstrb(li_wstrb[d]), .m_wvalid(li_wvalid[d]),
      .m_wready(li_wready[d]), .m_bresp(li_bresp[d]), .m_bvalid(li_bvalid[d]),
      .m_bready(li_bready[d]), .m_araddr(li_araddr[d]), .m_arvalid(li_arvalid[d]),
      .m_arready(li_arready[d]), .m_rdata(li_rdata[d]), .m_rresp(li_rresp[d]),
      .m_rvalid(li_rvalid[d]), .m_rready(li_rready[d]),
      .s_awaddr(lc_awaddr[d]), .s_awvalid(lc_awvalid[d]), .s_awready(lc_awready[d]),
      .s_wdata(lc_wdata[d]), .s_wstrb(lc_wstrb[d]), .s_wvalid(lc_wvalid[d]),
      .s_wready(lc_wready[d]), .s_bresp(lc_bresp[d]), .s_bvalid(lc_bvalid[d]),
      .s_bready(lc_bready[d]), .s_araddr(lc_araddr[d]), .s_arvalid(lc_arvalid[d]),
      .s_arready(lc_arready[d]), .s_rdata(lc_rdata[d]), .s_rresp(lc_rresp[d]),
      .s_rvalid(lc_rvalid[d]), .s_rready(lc_rready[d]));
  end

  // ---- the DDR4 cores, [d] = DIMM C<d>: rst from the governor; native ports idle
`define LD4(N) \
  otpu_ld4 u_ld``N ( \
    .clk300(c300[N]), .rst(!dimm_on[N]), .sys_clk(d_sys_clk[N]), .sys_rst(d_sys_rst[N]), \
    .ctl_clk(core_clk), .ctl_rst(csr_rst[N]), \
    .ctl_awvalid(lc_awvalid[N]), .ctl_awready(lc_awready[N]), .ctl_awaddr(lc_awaddr[N]), \
    .ctl_wvalid(lc_wvalid[N]), .ctl_wready(lc_wready[N]), .ctl_wdata(lc_wdata[N]), \
    .ctl_wstrb(lc_wstrb[N]), .ctl_bvalid(lc_bvalid[N]), .ctl_bready(lc_bready[N]), \
    .ctl_bresp(lc_bresp[N]), .ctl_arvalid(lc_arvalid[N]), .ctl_arready(lc_arready[N]), \
    .ctl_araddr(lc_araddr[N]), .ctl_rvalid(lc_rvalid[N]), .ctl_rready(lc_rready[N]), \
    .ctl_rdata(lc_rdata[N]), .ctl_rresp(lc_rresp[N]), \
    .c0_cmd_valid(1'b0), .c0_cmd_ready(), .c0_cmd_we(1'b0), .c0_cmd_addr(27'd0), \
    .c0_wdata_valid(1'b0), .c0_wdata_ready(), .c0_wdata_data(512'd0), .c0_wdata_we(64'd0), \
    .c0_rdata_valid(), .c0_rdata_ready(1'b1), .c0_rdata_data(), .c0_ready(), .c0_ded(), \
    .c0b_cmd_valid(1'b0), .c0b_cmd_ready(), .c0b_cmd_we(1'b0), .c0b_cmd_addr(27'd0), \
    .c0b_wdata_valid(1'b0), .c0b_wdata_ready(), .c0b_wdata_data(512'd0), .c0b_wdata_we(64'd0), \
    .c0b_rdata_valid(), .c0b_rdata_ready(1'b1), .c0b_rdata_data(), \
    .serial_rx(uart_sel == 2'(N) ? uart_rx : 1'b1), .serial_tx(d_tx[N]), \
    .ddram_a(ddr4_c``N``_a), .ddram_we_n(ddr4_c``N``_we_n), .ddram_cas_n(ddr4_c``N``_cas_n), \
    .ddram_ras_n(ddr4_c``N``_ras_n), .ddram_act_n(ddr4_c``N``_act_n), \
    .ddram_ba(ddr4_c``N``_ba), .ddram_bg(ddr4_c``N``_bg), .ddram_cke(ddr4_c``N``_cke), \
    .ddram_odt(ddr4_c``N``_odt), .ddram_cs_n(ddr4_c``N``_cs_n), \
    .ddram_reset_n(ddr4_c``N``_reset_n), .ddram_clk_p(ddr4_c``N``_clk_p), \
    .ddram_clk_n(ddr4_c``N``_clk_n), .ddram_dm(ddr4_c``N``_dm), .ddram_dq(ddr4_c``N``_dq), \
    .ddram_dqs_p(ddr4_c``N``_dqs_p), .ddram_dqs_n(ddr4_c``N``_dqs_n)); \
  assign ddr4_c``N``_par = 1'b0; \
  assign ddr4_c``N``_cs_hi = 3'b111; \
  assign ddr4_c``N``_cke1 = 1'b0; \
  assign ddr4_c``N``_odt1 = 1'b0;
  `LD4(0)
  `LD4(1)
  `LD4(2)
  `LD4(3)
`undef LD4
  assign uart_tx = d_tx[uart_sel];
endmodule
